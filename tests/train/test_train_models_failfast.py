from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from train.models import (
    _DIRECT_POSE_COMPONENT_STAT_KEYS,
    _DIRECT_POSE_DEFAULT_STAT_KEYS,
    EventMotionModel,
    MotionJointLoss,
    _DirectPoseGroupNormRequest,
    _DirectPosePayloadRequest,
    _ensure_temporal_axis,
    _masked_group_mean,
    _masked_group_weighted_mean,
    _setdefault_stats,
    _stats_float,
    _stats_float_or,
    _torch_dynamo_is_compiling_safe,
    _torch_onnx_is_in_export_safe,
)


def _state_layout(num_joints: int) -> dict[str, dict[str, int]]:
    rot_dim = int(num_joints) * 6
    angvel_dim = int(num_joints) * 3
    return {
        "RootPosition": {"start": 0, "size": 3},
        "RootVelocity": {"start": 3, "size": 2},
        "BoneRotations6D": {"start": 5, "size": rot_dim},
        "BoneAngularVelocities": {"start": 5 + rot_dim, "size": angvel_dim},
    }


def _output_layout(num_joints: int) -> dict[str, dict[str, int]]:
    return {"BoneRotations6D": {"start": 0, "size": int(num_joints) * 6}}


class _RaisingModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("sentinel leg head failure")


class _RaisingEmbeddingModule(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("sentinel side embedding failure")


class _NamedRaisingModule(nn.Module):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(self.message)


class _RaisingAdaptiveHistoryModule(nn.Module):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message
        self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, pose_hist: torch.Tensor, context: torch.Tensor | None = None):  # type: ignore[override]
        raise RuntimeError(self.message)


class _RaisingFrozenEncoder(nn.Module):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def forward(self, x: torch.Tensor, return_summary: bool | None = None):  # type: ignore[override]
        raise RuntimeError(self.message)


class _BadFeatureSource:
    def __str__(self) -> str:
        raise RuntimeError("sentinel direct feature source failure")


class _BadCtorStringObject:
    def __str__(self) -> str:
        raise RuntimeError("sentinel ctor __str__ failure")


class _BadFloatRuntimeObject:
    def __float__(self) -> float:
        raise RuntimeError("sentinel float runtime failure")


class _BadIntRuntimeObject:
    def __int__(self) -> int:
        raise RuntimeError("sentinel int runtime failure")


class _TypeErrorBoneNameSequence:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> str:
        raise TypeError("sentinel bone-name type failure")


class _RuntimeErrorBoneNameSequence:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> str:
        raise RuntimeError("sentinel bone-name runtime failure")


class _TypeErrorIterable:
    def __iter__(self):
        raise TypeError("sentinel iterable type failure")


class _RuntimeErrorIterable:
    def __iter__(self):
        raise RuntimeError("sentinel iterable runtime failure")


def _build_direct_pose_leg_model(
    *,
    side_routing: bool,
    gate_mode: str = "none",
    side_sign_gate: bool = False,
    use_phase: bool = False,
    phase_mode: str = "concat",
    side_cue: str = "none",
    side_embed_dim: int = 0,
    time_pe_dim: int = 0,
    contact_time_pe_dim: int = 0,
    contact_plan_init_mode: str = "learnable",
    use_event_clock: bool = False,
    angvel_dim: int = 0,
    pose_hist_dim: int = 0,
    period_dim: int = 0,
    lambda_fusion_enable: bool = False,
    lambda_fusion_use_rollout_step: bool = False,
    direct_pose_enable: bool = True,
    direct_pose_leg_enable: bool = True,
    bone_names: tuple[str, ...] | list[str] | None = None,
    **model_overrides,
) -> EventMotionModel:
    if bone_names is None:
        bone_names = ("thigh_l", "thigh_r")
    num_joints = len(bone_names)
    torch.manual_seed(0)
    model_kwargs = dict(
        in_state_dim=5 + num_joints * 6 + num_joints * 3,
        out_motion_dim=num_joints * 6,
        cond_dim=4,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        contact_dim=2,
        angvel_dim=angvel_dim,
        pose_hist_dim=pose_hist_dim,
        period_dim=period_dim,
        state_layout=_state_layout(num_joints),
        output_layout=_output_layout(num_joints),
        bone_names=list(bone_names),
        contact_plan_enable=True,
        contact_plan_hidden=8,
        contact_plan_inject="none",
        contact_plan_init_mode=contact_plan_init_mode,
        use_event_clock=use_event_clock,
        direct_pose_enable=direct_pose_enable,
        direct_pose_hidden=16,
        direct_pose_meas_mode="concat",
        direct_pose_use_phase_z=use_phase,
        direct_pose_phase_z_mode=phase_mode,
        direct_pose_leg_enable=direct_pose_leg_enable,
        direct_pose_leg_bones=("thigh_l", "thigh_r"),
        direct_pose_leg_mode="so3",
        direct_pose_leg_gate_mode=gate_mode,
        direct_pose_leg_side_routing=side_routing,
        direct_pose_leg_side_sign_gate=side_sign_gate,
        direct_pose_leg_side_cue=side_cue,
        direct_pose_leg_side_embed_dim=side_embed_dim,
        direct_pose_time_pe_dim=time_pe_dim,
        contact_plan_time_pe_dim=contact_time_pe_dim,
        lambda_fusion_enable=lambda_fusion_enable,
        lambda_fusion_use_rollout_step=lambda_fusion_use_rollout_step,
    )
    model_kwargs.update(model_overrides)
    model = EventMotionModel(**model_kwargs)
    model.eval()
    return model


def _forward_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    state = torch.randn(1, 2, 23, dtype=torch.float32)
    cond = torch.randn(1, 2, 4, dtype=torch.float32)
    contacts = torch.rand(1, 2, 2, dtype=torch.float32)
    return state, cond, contacts


def _build_motion_joint_loss(
    *,
    num_joints: int = 2,
    bone_names: tuple[str, ...] | list[str] | None = None,
    meta: dict | None = None,
    **kwargs,
) -> MotionJointLoss:
    if bone_names is None:
        bone_names = ("thigh_l", "thigh_r")
    loss = MotionJointLoss(
        output_layout=_output_layout(num_joints),
        meta=meta,
        **kwargs,
    )
    loss.set_bone_names(list(bone_names))
    return loss


def _cause_text(exc: BaseException) -> str:
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        messages.append(str(current))
        current = current.__cause__
    return "\n".join(messages)


def _patch_tensor_zero_once(expected_shape: tuple[int, ...], message: str):
    original_zero = torch.Tensor.zero_
    hit = {"count": 0}

    def _patched_zero(tensor: torch.Tensor, *args, **kwargs):  # type: ignore[no-untyped-def]
        shape = tuple(int(v) for v in tensor.shape)
        if hit["count"] == 0 and shape == tuple(int(v) for v in expected_shape):
            hit["count"] += 1
            raise RuntimeError(message)
        return original_zero(tensor, *args, **kwargs)

    return mock.patch.object(torch.Tensor, "zero_", _patched_zero)


def _patch_register_buffer_once(target_names: tuple[str, ...], message: str):
    original_register_buffer = nn.Module.register_buffer
    wanted = set(target_names)
    hit = {"count": 0}

    def _patched_register_buffer(module: nn.Module, name: str, tensor: torch.Tensor, persistent: bool = True) -> None:
        if hit["count"] == 0 and isinstance(module, EventMotionModel) and name in wanted:
            hit["count"] += 1
            raise RuntimeError(message)
        return original_register_buffer(module, name, tensor, persistent=persistent)

    return mock.patch.object(nn.Module, "register_buffer", _patched_register_buffer)


class TrainModelsFailFastTest(unittest.TestCase):
    def _assert_failure(
        self,
        factory,
        *,
        expected_messages: tuple[str, ...],
        exc_types=(RuntimeError, TypeError, ValueError),
    ) -> None:
        with self.assertRaises(exc_types) as raised:
            factory()

        text = _cause_text(raised.exception)
        for message in expected_messages:
            self.assertIn(message, text)

    def _assert_init_failure(
        self,
        factory,
        *,
        expected_messages: tuple[str, ...],
        exc_types=(RuntimeError, TypeError, ValueError),
    ) -> None:
        self._assert_failure(
            factory,
            expected_messages=expected_messages,
            exc_types=exc_types,
        )

    def _assert_forward_failure(
        self,
        model: EventMotionModel,
        *,
        expected_messages: tuple[str, ...],
        phase_z: torch.Tensor | None = None,
        phase_event_age: torch.Tensor | None = None,
        time_index: torch.Tensor | int | float | None = None,
        rollout_step: torch.Tensor | int | float | None = None,
    ) -> None:
        state, cond, contacts = _forward_inputs()
        angvel = None
        pose_history = None
        if int(getattr(model, "angvel_dim", 0) or 0) > 0:
            angvel = torch.randn(1, 2, int(model.angvel_dim), dtype=torch.float32)
        if int(getattr(model, "pose_hist_dim", 0) or 0) > 0:
            pose_history = torch.randn(1, 2, int(model.pose_hist_dim), dtype=torch.float32)

        with self.assertRaises(RuntimeError) as raised:
            model(
                state,
                cond,
                contacts=contacts,
                angvel=angvel,
                pose_history=pose_history,
                phase_z=phase_z,
                phase_event_age=phase_event_age,
                time_index=time_index,
                rollout_step=rollout_step,
            )

        text = _cause_text(raised.exception)
        for message in expected_messages:
            self.assertIn(message, text)

    def test_motion_joint_loss_module_group_helpers_preserve_weighted_means(self) -> None:
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        mask = torch.tensor([True, False, True])
        joint_weights = torch.tensor([1.0, 99.0, 3.0], dtype=torch.float32)

        self.assertTrue(torch.allclose(_masked_group_mean(values, mask), torch.tensor(3.5)))
        self.assertTrue(torch.allclose(_masked_group_weighted_mean(values, mask, joint_weights), torch.tensor(4.0)))
        self.assertIsNone(_masked_group_mean(values, torch.tensor([False, False, False])))
        self.assertIsNone(_masked_group_weighted_mean(values, mask, torch.tensor([0.0, 1.0, 0.0])))

    def test_motion_joint_loss_module_stats_helpers_preserve_scalar_contract(self) -> None:
        self.assertEqual(_stats_float(torch.tensor(2.5)), 2.5)
        self.assertEqual(_stats_float_or(torch.ones(2), default=-7.0), -7.0)
        self.assertEqual(_stats_float_or(_BadFloatRuntimeObject(), default=3.0), 3.0)

    def test_motion_joint_loss_module_temporal_and_stats_default_helpers(self) -> None:
        rank2 = torch.zeros(2, 3, dtype=torch.float32)
        rank3 = torch.zeros(2, 4, 3, dtype=torch.float32)
        stats = {"keep": 1.0}

        self.assertEqual(tuple(_ensure_temporal_axis(rank2).shape), (2, 1, 3))
        self.assertEqual(tuple(_ensure_temporal_axis(rank3).shape), (2, 4, 3))
        _setdefault_stats(stats, {"keep": 2.0, "new": 3.0})
        self.assertEqual(stats, {"keep": 1.0, "new": 3.0})

    def test_motion_joint_loss_helper_aliases_remain_compatible(self) -> None:
        loss = _build_motion_joint_loss()
        values = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
        mask = torch.tensor([True, False])

        self.assertTrue(torch.allclose(loss._masked_group_mean(values, mask), _masked_group_mean(values, mask)))
        self.assertEqual(loss._stats_float_or(torch.ones(2), default=-1.0), _stats_float_or(torch.ones(2), default=-1.0))

    def test_direct_pose_ctor_unknown_string_modes_use_explicit_defaults(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            phase_mode="bad-phase-mode",
            direct_pose_leg_mode="bad-leg-mode",
            gate_mode="bad-gate-mode",
            direct_pose_leg_contact_order="bad-contact-order",
            direct_pose_leg_side_cue="bad-side-cue",
        )

        self.assertEqual(model.direct_pose_phase_z_mode, "concat")
        self.assertEqual(model.direct_pose_leg_mode, "rot6d_add")
        self.assertEqual(model.direct_pose_leg_gate_mode, "none")
        self.assertEqual(model.direct_pose_leg_contact_order, "lr")
        self.assertEqual(model.direct_pose_leg_contact_ch_l, 0)
        self.assertEqual(model.direct_pose_leg_contact_ch_r, 1)
        self.assertEqual(model.direct_pose_leg_side_cue, "none")
        self.assertEqual(model.direct_pose_leg_side_cue_dim, 0)

    def test_direct_pose_ctor_none_defaults_preserved_for_constructor_cluster(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            phase_mode=None,
            direct_pose_leg_mode=None,
            gate_mode=None,
            direct_pose_leg_max_deg=None,
            direct_pose_leg_gate_power=None,
            direct_pose_leg_scale_log_clip=None,
            direct_pose_leg_scale_clamp_k=None,
            direct_pose_leg_contact_order=None,
            direct_pose_leg_side_embed_dim=None,
            direct_pose_leg_side_cue=None,
            direct_pose_leg_side_cue_tau=None,
        )

        self.assertEqual(model.direct_pose_phase_z_mode, "concat")
        self.assertEqual(model.direct_pose_leg_mode, "rot6d_add")
        self.assertEqual(model.direct_pose_leg_gate_mode, "none")
        self.assertAlmostEqual(model.direct_pose_leg_max_rad, 0.0)
        self.assertEqual(model.direct_pose_leg_gate_power, 1.0)
        self.assertEqual(model.direct_pose_leg_scale_log_clip, 4.0)
        self.assertEqual(model.direct_pose_leg_scale_clamp_k, 0.0)
        self.assertEqual(model.direct_pose_leg_contact_order, "lr")
        self.assertEqual(model.direct_pose_leg_side_embed_dim, 0)
        self.assertEqual(model.direct_pose_leg_side_cue, "none")
        self.assertEqual(model.direct_pose_leg_side_cue_tau, 30.0)

    def test_direct_pose_phase_z_mode_exotic_object_raises(self) -> None:
        self._assert_init_failure(
            lambda: _build_direct_pose_leg_model(
                side_routing=False,
                phase_mode=_BadCtorStringObject(),
            ),
            expected_messages=(
                "direct_pose_phase_z_mode",
                "must be a string or None",
                "actual_type=_BadCtorStringObject",
            ),
        )

    def test_direct_pose_leg_mode_exotic_object_raises(self) -> None:
        self._assert_init_failure(
            lambda: _build_direct_pose_leg_model(
                side_routing=False,
                direct_pose_leg_mode=_BadCtorStringObject(),
            ),
            expected_messages=(
                "direct_pose_leg_mode",
                "must be a string or None",
                "actual_type=_BadCtorStringObject",
            ),
        )

    def test_direct_pose_leg_max_deg_invalid_range_raises(self) -> None:
        self._assert_init_failure(
            lambda: _build_direct_pose_leg_model(
                side_routing=False,
                direct_pose_leg_max_deg=-1.0,
            ),
            expected_messages=(
                "direct_pose_leg_max_deg",
                "range [0, inf)",
                "value=-1.0",
            ),
        )

    def test_direct_pose_leg_gate_mode_exotic_object_raises(self) -> None:
        self._assert_init_failure(
            lambda: _build_direct_pose_leg_model(
                side_routing=False,
                gate_mode=_BadCtorStringObject(),
            ),
            expected_messages=(
                "direct_pose_leg_gate_mode",
                "must be a string or None",
                "actual_type=_BadCtorStringObject",
            ),
        )

    def test_direct_pose_leg_gate_power_invalid_range_raises(self) -> None:
        self._assert_init_failure(
            lambda: _build_direct_pose_leg_model(
                side_routing=False,
                direct_pose_leg_gate_power=0.0,
            ),
            expected_messages=(
                "direct_pose_leg_gate_power",
                "range (0, inf)",
                "value=0.0",
            ),
        )

    def test_direct_pose_leg_scale_log_clip_invalid_range_raises(self) -> None:
        self._assert_init_failure(
            lambda: _build_direct_pose_leg_model(
                side_routing=False,
                direct_pose_leg_scale_log_clip=0.0,
            ),
            expected_messages=(
                "direct_pose_leg_scale_log_clip",
                "range (0, inf)",
                "value=0.0",
            ),
        )

    def test_direct_pose_leg_scale_clamp_k_invalid_type_raises(self) -> None:
        self._assert_init_failure(
            lambda: _build_direct_pose_leg_model(
                side_routing=False,
                direct_pose_leg_scale_clamp_k="bad-scale-clamp",
            ),
            expected_messages=(
                "direct_pose_leg_scale_clamp_k",
                "finite scalar",
                "values <= 1 disable the clamp",
                "value='bad-scale-clamp'",
                "actual_type=str",
            ),
        )

    def test_direct_pose_leg_scale_clamp_k_values_le_one_disable_explicitly(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            direct_pose_leg_scale_clamp_k=0.5,
        )

        self.assertEqual(model.direct_pose_leg_scale_clamp_k, 0.0)

    def test_direct_pose_leg_contact_order_exotic_object_raises(self) -> None:
        self._assert_init_failure(
            lambda: _build_direct_pose_leg_model(
                side_routing=False,
                direct_pose_leg_contact_order=_BadCtorStringObject(),
            ),
            expected_messages=(
                "direct_pose_leg_contact_order",
                "must be a string or None",
                "actual_type=_BadCtorStringObject",
            ),
        )

    def test_direct_pose_leg_side_embed_dim_invalid_type_raises(self) -> None:
        self._assert_init_failure(
            lambda: _build_direct_pose_leg_model(
                side_routing=False,
                direct_pose_leg_side_embed_dim="bad-side-embed-dim",
            ),
            expected_messages=(
                "direct_pose_leg_side_embed_dim",
                "integer scalar in range [0, inf)",
                "value='bad-side-embed-dim'",
                "actual_type=str",
            ),
        )

    def test_direct_pose_leg_side_embed_dim_negative_clamps_to_zero(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            direct_pose_leg_side_embed_dim=-4,
        )

        self.assertEqual(model.direct_pose_leg_side_embed_dim, 0)

    def test_direct_pose_leg_side_cue_exotic_object_raises(self) -> None:
        self._assert_init_failure(
            lambda: _build_direct_pose_leg_model(
                side_routing=False,
                direct_pose_leg_side_cue=_BadCtorStringObject(),
            ),
            expected_messages=(
                "direct_pose_leg_side_cue",
                "must be a string or None",
                "actual_type=_BadCtorStringObject",
            ),
        )

    def test_direct_pose_leg_side_cue_tau_invalid_range_raises(self) -> None:
        self._assert_init_failure(
            lambda: _build_direct_pose_leg_model(
                side_routing=False,
                direct_pose_leg_side_cue_tau=0.0,
            ),
            expected_messages=(
                "direct_pose_leg_side_cue_tau",
                "range (0, inf)",
                "value=0.0",
            ),
        )

    def test_bone_residual_adapter_metadata_failure_disables_adapters(self) -> None:
        with mock.patch.object(
            EventMotionModel,
            "_init_bone_residual_adapters",
            side_effect=ValueError("sentinel adapter metadata failure"),
        ):
            model = _build_direct_pose_leg_model(
                side_routing=False,
                residual_adapter_bones=("thigh_l",),
            )

        self.assertEqual(model._bone_adapter_slices, [])
        self.assertEqual(model._bone_adapter_names, [])
        self.assertEqual(len(model._bone_adapters), 0)

    def test_bone_residual_adapter_runtime_failure_raises(self) -> None:
        with mock.patch.object(
            EventMotionModel,
            "_init_bone_residual_adapters",
            side_effect=RuntimeError("sentinel adapter runtime failure"),
        ):
            self._assert_init_failure(
                lambda: _build_direct_pose_leg_model(
                    side_routing=False,
                    residual_adapter_bones=("thigh_l",),
                ),
                expected_messages=(
                    "sentinel adapter runtime failure",
                ),
                exc_types=(RuntimeError,),
            )

    def test_direct_pose_split_leg_name_type_error_falls_back_to_empty_names(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            direct_pose_split_enable=True,
            direct_pose_leg_enable=False,
            bone_names=("thigh_l", "thigh_r", "spine"),
        )
        model.direct_pose_leg_joint_names = []
        model.direct_pose_leg_joint_idx = [0, 1]

        model._init_direct_pose_routing_metadata(
            bone_names=_TypeErrorBoneNameSequence(),
            output_layout=_output_layout(3),
        )

        self.assertEqual(model.direct_pose_leg_joint_names, [])

    def test_direct_pose_split_leg_name_runtime_error_not_swallowed(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            direct_pose_split_enable=True,
            direct_pose_leg_enable=False,
            bone_names=("thigh_l", "thigh_r", "spine"),
        )
        model.direct_pose_leg_joint_names = []
        model.direct_pose_leg_joint_idx = [0, 1]

        self._assert_failure(
            lambda: model._init_direct_pose_routing_metadata(
                bone_names=_RuntimeErrorBoneNameSequence(),
                output_layout=_output_layout(3),
            ),
            expected_messages=(
                "sentinel bone-name runtime failure",
            ),
            exc_types=(RuntimeError,),
        )

    def test_eval_runtime_control_scalar_parse_failure_falls_back_to_default(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False)

        controls = model._normalize_eval_runtime_controls(
            contact_plan_inject_scale="bad-inject-scale",
            contact_plan_time_bias_scale="bad-time-bias-scale",
        )

        self.assertEqual(controls.contact_plan_inject_scale, 1.0)
        self.assertEqual(controls.contact_plan_time_bias_scale, 1.0)

    def test_eval_runtime_control_scalar_runtime_error_not_swallowed(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False)

        self._assert_failure(
            lambda: model._normalize_eval_runtime_controls(
                contact_plan_inject_scale=_BadFloatRuntimeObject(),
            ),
            expected_messages=(
                "sentinel float runtime failure",
            ),
            exc_types=(RuntimeError,),
        )

    def test_contact_plan_debug_stack_shape_mismatch_falls_back_to_none(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False)
        buffers = model._init_contact_plan_debug_buffers(enabled=True)
        assert buffers is not None
        buffers.contacts_plan_logits_base = [
            torch.zeros(1, 2, dtype=torch.float32),
            torch.zeros(1, 3, dtype=torch.float32),
        ]

        debug_logits = model._finalize_contact_plan_debug_logits(buffers)

        self.assertIsNone(debug_logits.contacts_plan_logits_base)

    def test_contact_plan_debug_stack_type_error_not_swallowed(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False)
        buffers = model._init_contact_plan_debug_buffers(enabled=True)
        assert buffers is not None
        buffers.contacts_plan_logits_base = [torch.zeros(1, 2, dtype=torch.float32), object()]  # type: ignore[list-item]

        with self.assertRaises(TypeError):
            model._finalize_contact_plan_debug_logits(buffers)

    def test_direct_pose_leg_cross_leg_ablation_contact_dim_parse_failure_returns_none(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False)
        model.contact_dim = "bad-contact-dim"  # type: ignore[assignment]
        model.direct_pose_leg_joint_names = ["thigh_r", "thigh_l"]

        result = model._compute_direct_pose_leg_cross_leg_ablation(
            leg_in=torch.zeros(1, 2, 12, dtype=torch.float32),
            direct_feat=None,
            plan_in=None,
            meas_in=None,
            phase_in_direct=None,
            batch_size=1,
            seq_len=2,
            joint_count=2,
            ablation_mode="zero",
        )

        self.assertIsNone(result)

    def test_direct_pose_leg_cross_leg_ablation_contact_dim_runtime_error_not_swallowed(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False)
        model.contact_dim = _BadIntRuntimeObject()  # type: ignore[assignment]
        model.direct_pose_leg_joint_names = ["thigh_r", "thigh_l"]

        self._assert_failure(
            lambda: model._compute_direct_pose_leg_cross_leg_ablation(
                leg_in=torch.zeros(1, 2, 12, dtype=torch.float32),
                direct_feat=None,
                plan_in=None,
                meas_in=None,
                phase_in_direct=None,
                batch_size=1,
                seq_len=2,
                joint_count=2,
                ablation_mode="zero",
            ),
            expected_messages=(
                "sentinel int runtime failure",
            ),
            exc_types=(RuntimeError,),
        )

    def test_direct_pose_leg_cross_leg_ablation_joint_names_parse_failure_returns_none(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False)
        model.contact_dim = 2
        model.direct_pose_leg_joint_names = _TypeErrorIterable()  # type: ignore[assignment]

        result = model._compute_direct_pose_leg_cross_leg_ablation(
            leg_in=torch.zeros(1, 2, 12, dtype=torch.float32),
            direct_feat=None,
            plan_in=None,
            meas_in=None,
            phase_in_direct=None,
            batch_size=1,
            seq_len=2,
            joint_count=2,
            ablation_mode="zero",
        )

        self.assertIsNone(result)

    def test_direct_pose_leg_cross_leg_ablation_joint_names_runtime_error_not_swallowed(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False)
        model.contact_dim = 2
        model.direct_pose_leg_joint_names = _RuntimeErrorIterable()  # type: ignore[assignment]

        self._assert_failure(
            lambda: model._compute_direct_pose_leg_cross_leg_ablation(
                leg_in=torch.zeros(1, 2, 12, dtype=torch.float32),
                direct_feat=None,
                plan_in=None,
                meas_in=None,
                phase_in_direct=None,
                batch_size=1,
                seq_len=2,
                joint_count=2,
                ablation_mode="zero",
            ),
            expected_messages=(
                "sentinel iterable runtime failure",
            ),
            exc_types=(RuntimeError,),
        )

    def test_direct_pose_leg_residual_head_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False)
        model.direct_pose_leg_head = _RaisingModule()
        self._assert_forward_failure(
            model,
            expected_messages=(
                "direct_pose forward failed",
                "direct_pose leg residual forward failed",
                "sentinel leg head failure",
            ),
        )

    def test_direct_pose_side_routed_leg_residual_head_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=True)
        model.direct_pose_leg_head_shared = _RaisingModule()
        self._assert_forward_failure(
            model,
            expected_messages=(
                "direct_pose forward failed",
                "direct_pose side-routed leg residual forward failed",
                "sentinel leg head failure",
            ),
        )

    def test_direct_pose_side_routed_leg_sign_gate_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=True, side_sign_gate=True)
        model.direct_pose_leg_side_sign_gate_head = _RaisingModule()
        self._assert_forward_failure(
            model,
            expected_messages=(
                "direct_pose forward failed",
                "direct_pose side-routed leg residual forward failed",
                "direct_pose side-routed leg sign gate forward failed",
                "sentinel leg head failure",
            ),
        )

    def test_direct_pose_side_routed_leg_learned_gate_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=True, gate_mode="learned")
        model.direct_pose_leg_gate_head_shared = _RaisingModule()
        self._assert_forward_failure(
            model,
            expected_messages=(
                "direct_pose forward failed",
                "direct_pose side-routed leg residual forward failed",
                "direct_pose side-routed leg learned gate forward failed",
                "sentinel leg head failure",
            ),
        )

    def test_direct_pose_side_routed_leg_scale_gate_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=True, gate_mode="scale")
        model.direct_pose_leg_gate_head_shared = _RaisingModule()
        self._assert_forward_failure(
            model,
            expected_messages=(
                "direct_pose forward failed",
                "direct_pose side-routed leg residual forward failed",
                "direct_pose side-routed leg scale gate forward failed",
                "sentinel leg head failure",
            ),
        )

    def test_direct_pose_leg_learned_gate_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False, gate_mode="learned")
        model.direct_pose_leg_gate_head = _RaisingModule()
        self._assert_forward_failure(
            model,
            expected_messages=(
                "direct_pose forward failed",
                "direct_pose leg residual forward failed",
                "direct_pose leg learned gate forward failed",
                "sentinel leg head failure",
            ),
        )

    def test_direct_pose_leg_scale_gate_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False, gate_mode="scale")
        model.direct_pose_leg_gate_head = _RaisingModule()
        self._assert_forward_failure(
            model,
            expected_messages=(
                "direct_pose forward failed",
                "direct_pose leg residual forward failed",
                "direct_pose leg scale gate forward failed",
                "sentinel leg head failure",
            ),
        )

    def test_phase_z_sequence_contract_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False, use_phase=True)
        self._assert_forward_failure(
            model,
            expected_messages=(
                "phase_z sequence contract failed in EventMotionModel.forward",
                "expected phase_z to be broadcastable to (B=1, Tq=2, feat_dim=4)",
                "2D inputs are interpreted as batch-major, not time-major",
                "shape=(2, 4)",
            ),
            phase_z=torch.randn(2, 4, dtype=torch.float32),
        )

    def test_phase_event_age_sequence_contract_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=True,
            side_cue="phase_event_age",
        )
        self._assert_forward_failure(
            model,
            expected_messages=(
                "phase_event_age sequence contract failed in EventMotionModel.forward",
                "expected phase_event_age to be broadcastable to (B=1, Tq=2, feat_dim=2)",
                "shape=(1, 2, 1)",
                "3D input must carry exactly feat_dim=2 features, but got 1",
            ),
            phase_event_age=torch.randn(1, 2, 1, dtype=torch.float32),
        )

    def test_side_routed_phase_z_view_contract_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=True,
            use_phase=True,
            phase_mode="replace_contacts",
        )
        model.contact_plan_head = nn.Linear(int(model.contact_plan_hidden), 1)
        self._assert_forward_failure(
            model,
            expected_messages=(
                "direct_pose forward failed",
                "direct_pose side-routed leg residual forward failed",
                "direct_pose side-routed phase_z contract failed",
                "expected phase_z_in_direct to resolve to (B=1, Tq=2, 2*contact_channels=2)",
                "before per-side view `(B, Tq, contact_channels=1, 2)`",
                "shape=(1, 2, 4)",
                "contact_dim=2",
            ),
            phase_z=torch.randn(1, 2, 4, dtype=torch.float32),
        )

    def test_side_routed_side_embedding_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=True,
            side_embed_dim=4,
        )
        model.direct_pose_leg_side_embed = _RaisingEmbeddingModule(dim=4)
        self._assert_forward_failure(
            model,
            expected_messages=(
                "direct_pose forward failed",
                "direct_pose side-routed leg residual forward failed",
                "direct_pose side-routed side embedding forward failed",
                "expected emb_r/emb_l broadcast to (B, Tq, D)",
                "embed_weight_shape=(2, 4)",
                "sentinel side embedding failure",
            ),
        )

    def test_direct_pose_time_pe_concat_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            time_pe_dim=6,
        )
        original_cat = torch.cat

        def _patched_cat(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...], dim: int = 0) -> torch.Tensor:
            if (
                len(tensors) == 2
                and all(torch.is_tensor(t) for t in tensors)
                and tensors[0].ndim == 3
                and tensors[1].ndim == 3
                and tuple(int(v) for v in tensors[0].shape) == (1, 2, 4)
                and tuple(int(v) for v in tensors[1].shape) == (1, 2, 6)
                and dim == -1
            ):
                raise RuntimeError("sentinel direct time concat failure")
            return original_cat(tensors, dim=dim)

        with mock.patch("train.models.torch.cat", side_effect=_patched_cat):
            self._assert_forward_failure(
                model,
                expected_messages=(
                    "direct_pose forward failed",
                    "direct_pose time PE concat failed",
                    "direct_feat.shape=(1, 2, 4)",
                    "time_pe_direct.shape=(1, 2, 6)",
                    "time_pe_dim=6",
                    "sentinel direct time concat failure",
                ),
            )

    def test_direct_pose_feat_source_contract_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False)
        model.direct_pose_feat_source = _BadFeatureSource()
        self._assert_forward_failure(
            model,
            expected_messages=(
                "direct_pose forward failed",
                "direct_pose_feat_source contract failed",
                "sentinel direct feature source failure",
            ),
        )

    def test_time_index_contract_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            time_pe_dim=6,
        )
        self._assert_forward_failure(
            model,
            expected_messages=(
                "time_index contract failed in EventMotionModel.forward",
                "1D tensor with length 1 or B=1",
                "shape=(3,)",
                "direct_pose_time_pe_dim=6",
            ),
            time_index=torch.randn(3, dtype=torch.float32),
        )

    def test_contact_plan_time_pe_construction_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            contact_time_pe_dim=6,
        )
        model._contact_plan_time_pe_base = "bad-contact-time-base"
        self._assert_forward_failure(
            model,
            expected_messages=(
                "contact_plan time PE construction failed",
                "pe_dim=6",
                "t_grid.shape=(1, 2)",
                "bad-contact-time-base",
            ),
        )

    def test_direct_pose_time_pe_construction_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            time_pe_dim=6,
        )
        model._direct_pose_time_pe_base = "bad-direct-time-base"
        self._assert_forward_failure(
            model,
            expected_messages=(
                "direct_pose time PE construction failed",
                "pe_dim=6",
                "t_grid.shape=(1, 2)",
                "bad-direct-time-base",
            ),
        )

    def test_contact_plan_observed_init_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            contact_plan_init_mode="obs",
        )
        model.contact_plan_init_head = _NamedRaisingModule("sentinel observed init failure")
        self._assert_forward_failure(
            model,
            expected_messages=(
                "contact_plan observed init failed",
                "init_mode='obs'",
                "obs0.shape=(1, 2)",
                "sentinel observed init failure",
            ),
        )

    def test_event_clock_phase_append_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            use_phase=True,
            use_event_clock=True,
        )
        phase_z = torch.full((1, 2, 4), 7.0, dtype=torch.float32)
        original_getitem = torch.Tensor.__getitem__

        def _patched_getitem(self: torch.Tensor, key):  # type: ignore[no-untyped-def]
            if (
                self.ndim == 3
                and tuple(int(v) for v in self.shape) == (1, 2, 4)
                and torch.all(self == 7.0).item()
                and isinstance(key, tuple)
                and len(key) == 2
                and isinstance(key[1], int)
            ):
                raise RuntimeError("sentinel event-clock phase append failure")
            return original_getitem(self, key)

        with mock.patch.object(torch.Tensor, "__getitem__", _patched_getitem):
            self._assert_forward_failure(
                model,
                expected_messages=(
                    "direct_pose phase append failed",
                    "event_clock=on",
                    "phase_input_seq.shape=(1, 2, 4)",
                    "sentinel event-clock phase append failure",
                ),
                phase_z=phase_z,
            )

    def test_event_clock_side_cue_append_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=True,
            side_cue="phase_event_age",
            use_event_clock=True,
        )
        phase_event_age = torch.full((1, 2, 2), 9.0, dtype=torch.float32)
        original_getitem = torch.Tensor.__getitem__

        def _patched_getitem(self: torch.Tensor, key):  # type: ignore[no-untyped-def]
            if (
                self.ndim == 3
                and tuple(int(v) for v in self.shape) == (1, 2, 2)
                and torch.all(self == 9.0).item()
                and isinstance(key, tuple)
                and len(key) == 2
                and isinstance(key[1], int)
            ):
                raise RuntimeError("sentinel event-clock side cue append failure")
            return original_getitem(self, key)

        with mock.patch.object(torch.Tensor, "__getitem__", _patched_getitem):
            self._assert_forward_failure(
                model,
                expected_messages=(
                    "direct_pose side cue append failed",
                    "event_clock=on",
                    "cue_mode='phase_event_age'",
                    "phase_age_seq.shape=(1, 2, 2)",
                    "sentinel event-clock side cue append failure",
                ),
                phase_event_age=phase_event_age,
            )

    def test_non_event_clock_phase_append_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            use_phase=True,
            use_event_clock=False,
        )
        phase_z = torch.full((1, 2, 4), 11.0, dtype=torch.float32)
        original_getitem = torch.Tensor.__getitem__

        def _patched_getitem(self: torch.Tensor, key):  # type: ignore[no-untyped-def]
            if (
                self.ndim == 3
                and tuple(int(v) for v in self.shape) == (1, 2, 4)
                and torch.all(self == 11.0).item()
                and isinstance(key, tuple)
                and len(key) == 2
                and isinstance(key[1], int)
            ):
                raise RuntimeError("sentinel non-event-clock phase append failure")
            return original_getitem(self, key)

        with mock.patch.object(torch.Tensor, "__getitem__", _patched_getitem):
            self._assert_forward_failure(
                model,
                expected_messages=(
                    "direct_pose phase append failed",
                    "event_clock=off",
                    "phase_input_seq.shape=(1, 2, 4)",
                    "sentinel non-event-clock phase append failure",
                ),
                phase_z=phase_z,
            )

    def test_non_event_clock_side_cue_append_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=True,
            side_cue="phase_event_age",
            use_event_clock=False,
        )
        phase_event_age = torch.full((1, 2, 2), 13.0, dtype=torch.float32)
        original_getitem = torch.Tensor.__getitem__

        def _patched_getitem(self: torch.Tensor, key):  # type: ignore[no-untyped-def]
            if (
                self.ndim == 3
                and tuple(int(v) for v in self.shape) == (1, 2, 2)
                and torch.all(self == 13.0).item()
                and isinstance(key, tuple)
                and len(key) == 2
                and isinstance(key[1], int)
            ):
                raise RuntimeError("sentinel non-event-clock side cue append failure")
            return original_getitem(self, key)

        with mock.patch.object(torch.Tensor, "__getitem__", _patched_getitem):
            self._assert_forward_failure(
                model,
                expected_messages=(
                    "direct_pose side cue append failed",
                    "event_clock=off",
                    "cue_mode='phase_event_age'",
                    "phase_age_seq.shape=(1, 2, 2)",
                    "sentinel non-event-clock side cue append failure",
                ),
                phase_event_age=phase_event_age,
            )

    def test_event_clock_time_bias_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            contact_time_pe_dim=6,
            use_event_clock=True,
        )
        model.contact_plan_time_head = _NamedRaisingModule("sentinel event-clock time bias failure")
        self._assert_forward_failure(
            model,
            expected_messages=(
                "contact_plan time bias forward failed",
                "event_clock=on",
                "time_pe.shape=(1, 2, 6)",
                "sentinel event-clock time bias failure",
            ),
        )

    def test_non_event_clock_time_bias_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            contact_time_pe_dim=6,
            use_event_clock=False,
        )
        model.contact_plan_time_head = _NamedRaisingModule("sentinel non-event-clock time bias failure")
        self._assert_forward_failure(
            model,
            expected_messages=(
                "contact_plan time bias forward failed",
                "event_clock=off",
                "time_pe.shape=(1, 2, 6)",
                "sentinel non-event-clock time bias failure",
            ),
        )

    def test_phase_sequence_stack_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            use_phase=True,
        )
        phase_z = torch.full((1, 2, 4), 5.0, dtype=torch.float32)
        original_stack = torch.stack

        def _patched_stack(tensors, dim=0, *args, **kwargs):  # type: ignore[no-untyped-def]
            if (
                isinstance(tensors, list)
                and len(tensors) == 2
                and dim == 1
                and all(torch.is_tensor(t) and tuple(int(v) for v in t.shape) == (1, 4) for t in tensors)
                and all(torch.all(t == 5.0).item() for t in tensors)
            ):
                raise RuntimeError("sentinel phase sequence stack failure")
            return original_stack(tensors, dim=dim, *args, **kwargs)

        with mock.patch("train.models.torch.stack", side_effect=_patched_stack):
            self._assert_forward_failure(
                model,
                expected_messages=(
                    "direct_pose phase sequence stack failed",
                    "phase_dim=4",
                    "element_shapes=[(1, 4), (1, 4)]",
                    "sentinel phase sequence stack failure",
                ),
                phase_z=phase_z,
            )

    def test_side_cue_sequence_stack_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=True,
            side_cue="phase_event_age",
        )
        phase_event_age = torch.full((1, 2, 2), 9.0, dtype=torch.float32)
        original_stack = torch.stack

        def _patched_stack(tensors, dim=0, *args, **kwargs):  # type: ignore[no-untyped-def]
            if (
                isinstance(tensors, list)
                and len(tensors) == 2
                and dim == 1
                and all(torch.is_tensor(t) and tuple(int(v) for v in t.shape) == (1, 2) for t in tensors)
                and all(torch.allclose(t, torch.full((1, 2), 9.0, dtype=t.dtype, device=t.device)) for t in tensors)
            ):
                raise RuntimeError("sentinel side cue sequence stack failure")
            return original_stack(tensors, dim=dim, *args, **kwargs)

        with mock.patch("train.models.torch.stack", side_effect=_patched_stack):
            self._assert_forward_failure(
                model,
                expected_messages=(
                    "direct_pose side cue sequence stack failed",
                    "cue_mode='phase_event_age'",
                    "element_shapes=[(1, 2), (1, 2)]",
                    "sentinel side cue sequence stack failure",
                ),
                phase_event_age=phase_event_age,
            )

    def test_contacts_plan_logits_stack_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
        )
        original_stack = torch.stack
        matching_call_count = {"count": 0}

        def _patched_stack(tensors, dim=0, *args, **kwargs):  # type: ignore[no-untyped-def]
            if (
                isinstance(tensors, list)
                and len(tensors) == 2
                and dim == 1
                and all(torch.is_tensor(t) and tuple(int(v) for v in t.shape) == (1, 2) for t in tensors)
            ):
                matching_call_count["count"] += 1
                if matching_call_count["count"] == 2:
                    raise RuntimeError("sentinel contacts_plan_logits stack failure")
            return original_stack(tensors, dim=dim, *args, **kwargs)

        with mock.patch("train.models.torch.stack", side_effect=_patched_stack):
            self._assert_forward_failure(
                model,
                expected_messages=(
                    "contacts_plan_logits stack failed",
                    "num_steps=2",
                    "element_shapes=[(1, 2), (1, 2)]",
                    "sentinel contacts_plan_logits stack failure",
                ),
            )

    def test_adaptive_history_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            use_event_clock=True,
            pose_hist_dim=12,
        )
        model.adaptive_history_module = _RaisingAdaptiveHistoryModule("sentinel adaptive history failure")
        model._adaptive_history_device = torch.device("cpu")
        self._assert_forward_failure(
            model,
            expected_messages=(
                "adaptive history module forward failed",
                "event_clock=on",
                "pose_history.shape=(1, 2, 12)",
                "sentinel adaptive history failure",
            ),
        )

    def test_frozen_period_feature_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            use_event_clock=True,
            angvel_dim=6,
            pose_hist_dim=12,
            period_dim=4,
        )
        model.frozen_encoder = _RaisingFrozenEncoder("sentinel frozen period failure")
        model.frozen_period_head = nn.Identity()
        self._assert_forward_failure(
            model,
            expected_messages=(
                "frozen period feature forward failed",
                "event_clock=on",
                "period_dim=4",
                "enc_in.shape=(1, 2, 20)",
                "sentinel frozen period failure",
            ),
        )

    def test_lambda_fusion_rollout_step_contract_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            lambda_fusion_enable=True,
            lambda_fusion_use_rollout_step=True,
        )
        self._assert_forward_failure(
            model,
            expected_messages=(
                "lambda_fusion rollout_step contract failed",
                "rollout_step_type=Tensor",
                "rollout_step.shape=(3,)",
            ),
            rollout_step=torch.randn(3, dtype=torch.float32),
        )

    def test_lambda_fusion_forward_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            lambda_fusion_enable=True,
        )
        model.lambda_fusion_head = _NamedRaisingModule("sentinel lambda fusion head failure")
        self._assert_forward_failure(
            model,
            expected_messages=(
                "lambda_fusion forward failed",
                "mode='per_joint'",
                "joint_count=2",
                "sentinel lambda fusion head failure",
            ),
        )

    def test_lambda_fusion_rot6d_layout_failure_raises_at_build_time(self) -> None:
        with mock.patch(
            "train.models.resolve_rot6d_slice",
            side_effect=RuntimeError("sentinel lambda layout failure"),
        ):
            self._assert_init_failure(
                lambda: _build_direct_pose_leg_model(
                    side_routing=False,
                    lambda_fusion_enable=True,
                    direct_pose_enable=False,
                ),
                expected_messages=(
                    "lambda_fusion rot6d layout resolution failed",
                    "field='output_layout'",
                    "out_motion_dim=12",
                    "sentinel lambda layout failure",
                ),
            )

    def test_so3_corrector_rot6d_layout_failure_raises_at_build_time(self) -> None:
        with mock.patch(
            "train.models.resolve_rot6d_slice",
            side_effect=RuntimeError("sentinel so3 layout failure"),
        ):
            self._assert_init_failure(
                lambda: _build_direct_pose_leg_model(
                    side_routing=False,
                    lambda_fusion_enable=False,
                    direct_pose_enable=False,
                ),
                expected_messages=(
                    "so3 corrector rot6d layout resolution failed",
                    "field='output_layout'",
                    "out_motion_dim=12",
                    "sentinel so3 layout failure",
                ),
            )

    def test_lambda_fusion_deterministic_init_failure_raises_at_build_time(self) -> None:
        with _patch_tensor_zero_once((1, 13), "sentinel lambda init failure"):
            self._assert_init_failure(
                lambda: _build_direct_pose_leg_model(
                    side_routing=False,
                    lambda_fusion_enable=True,
                    direct_pose_enable=False,
                    lambda_fusion_mode="global",
                    lambda_fusion_hidden=13,
                ),
                expected_messages=(
                    "lambda_fusion deterministic init failed",
                    "field='lambda_fusion_head[-1]'",
                    "weight_shape=(1, 13)",
                    "logit_init=-2.0",
                    "sentinel lambda init failure",
                ),
            )

    def test_so3_corrector_deterministic_init_failure_raises_at_build_time(self) -> None:
        with _patch_tensor_zero_once((6, 11), "sentinel so3 init failure"):
            self._assert_init_failure(
                lambda: _build_direct_pose_leg_model(
                    side_routing=False,
                    lambda_fusion_enable=False,
                    direct_pose_enable=False,
                    so3_corr_hidden=11,
                ),
                expected_messages=(
                    "so3 corrector deterministic init failed",
                    "field='so3_delta_corrector[-1]'",
                    "weight_shape=(6, 11)",
                    "joint_count=2",
                    "sentinel so3 init failure",
                ),
            )

    def test_contact_plan_init_head_deterministic_init_failure_raises_at_build_time(self) -> None:
        with _patch_tensor_zero_once((8, 13), "sentinel contact-plan init failure"):
            self._assert_init_failure(
                lambda: _build_direct_pose_leg_model(
                    side_routing=False,
                    direct_pose_enable=False,
                    contact_plan_init_mode="obs",
                    contact_plan_init_hidden=13,
                ),
                expected_messages=(
                    "contact_plan init head deterministic init failed",
                    "field='contact_plan_init_head[-1]'",
                    "weight_shape=(8, 13)",
                    "sentinel contact-plan init failure",
                ),
            )

    def test_contact_plan_time_head_deterministic_init_failure_raises_at_build_time(self) -> None:
        with _patch_tensor_zero_once((2, 6), "sentinel contact-plan time failure"):
            self._assert_init_failure(
                lambda: _build_direct_pose_leg_model(
                    side_routing=False,
                    direct_pose_enable=False,
                    contact_time_pe_dim=6,
                ),
                expected_messages=(
                    "contact_plan time head deterministic init failed",
                    "field='contact_plan_time_head'",
                    "weight_shape=(2, 6)",
                    "time_pe_dim=6",
                    "sentinel contact-plan time failure",
                ),
            )

    def test_direct_pose_leg_joint_index_buffer_registration_failure_raises(self) -> None:
        with _patch_register_buffer_once(
            ("direct_pose_leg_joint_idx_tensor",),
            "sentinel leg idx buffer registration failure",
        ):
            self._assert_init_failure(
                lambda: _build_direct_pose_leg_model(
                    side_routing=False,
                ),
                expected_messages=(
                    "direct_pose leg routing metadata registration failed",
                    "field='direct_pose_leg_joint_idx_tensor'",
                    "joint_indices=[0, 1]",
                    "sentinel leg idx buffer registration failure",
                ),
            )

    def test_direct_pose_split_leg_index_buffer_registration_failure_raises(self) -> None:
        with _patch_register_buffer_once(
            ("direct_pose_leg_joint_idx_tensor",),
            "sentinel split leg idx buffer registration failure",
        ):
            self._assert_init_failure(
                lambda: _build_direct_pose_leg_model(
                    side_routing=False,
                    direct_pose_split_enable=True,
                    direct_pose_leg_enable=False,
                ),
                expected_messages=(
                    "direct_pose split leg routing metadata registration failed",
                    "field='direct_pose_leg_joint_idx_tensor'",
                    "joint_indices=[0, 1]",
                    "sentinel split leg idx buffer registration failure",
                ),
            )

    def test_direct_pose_side_position_buffer_registration_failure_raises(self) -> None:
        with _patch_register_buffer_once(
            ("direct_pose_leg_side_pos_r_tensor",),
            "sentinel side position buffer registration failure",
        ):
            self._assert_init_failure(
                lambda: _build_direct_pose_leg_model(
                    side_routing=True,
                ),
                expected_messages=(
                    "direct_pose side routing position buffer registration failed",
                    "direct_pose_leg_side_pos_r_tensor",
                    "pos_r=",
                    "pos_l=",
                    "sentinel side position buffer registration failure",
                ),
            )

    def test_direct_pose_side_embedding_deterministic_init_failure_raises_at_build_time(self) -> None:
        with _patch_tensor_zero_once((2, 4), "sentinel side embedding init failure"):
            self._assert_init_failure(
                lambda: _build_direct_pose_leg_model(
                    side_routing=True,
                    side_embed_dim=4,
                ),
                expected_messages=(
                    "direct_pose side embedding deterministic init failed",
                    "field='direct_pose_leg_side_embed.weight'",
                    "weight_shape=(2, 4)",
                    "side_embed_dim=4",
                    "sentinel side embedding init failure",
                ),
            )

    def test_direct_pose_split_state_missing_leg_index_tensor_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            direct_pose_split_enable=True,
            direct_pose_leg_enable=False,
            bone_names=("thigh_l", "thigh_r", "spine"),
        )
        model.direct_pose_leg_out_idx = None
        self._assert_failure(
            lambda: model._direct_pose_split_state(),
            expected_messages=(
                "direct_pose_split_enable split state contract failed",
                "field='idx_leg'",
                "expected a non-empty 1D integer torch.Tensor",
                "actual_type=NoneType",
            ),
        )

    def test_direct_pose_split_state_disjoint_coverage_mismatch_raises(self) -> None:
        model = _build_direct_pose_leg_model(
            side_routing=False,
            direct_pose_split_enable=True,
            direct_pose_leg_enable=False,
            bone_names=("thigh_l", "thigh_r", "spine"),
        )
        model.direct_pose_nonleg_out_idx = model.direct_pose_nonleg_out_idx[:-1].clone()
        self._assert_failure(
            lambda: model._direct_pose_split_state(),
            expected_messages=(
                "direct_pose_split_enable split state contract failed",
                "fields=('direct_pose_leg_out_idx', 'direct_pose_nonleg_out_idx')",
                "expected full disjoint coverage",
                "out_motion_dim=18",
                "coverage_numel=17",
                "unique_numel=17",
            ),
        )

    def test_event_motion_model_forward_state_shape_contract_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False)
        state, cond, contacts = _forward_inputs()
        self._assert_failure(
            lambda: model(
                state[..., :-1],
                cond,
                contacts=contacts,
            ),
            expected_messages=(
                "state input shape contract failed in EventMotionModel.forward",
                "in_state_dim=23",
                "actual_shape=(1, 2, 22)",
            ),
        )

    def test_event_motion_model_forward_cond_shape_contract_failure_raises(self) -> None:
        model = _build_direct_pose_leg_model(side_routing=False)
        state, cond, contacts = _forward_inputs()
        self._assert_failure(
            lambda: model(
                state,
                cond[..., :-1],
                contacts=contacts,
            ),
            expected_messages=(
                "cond input shape contract failed in EventMotionModel.forward",
                "cond_dim=4",
                "actual_shape=(1, 2, 3)",
            ),
        )

    def test_attention_regularization_non_tensor_geomask_uses_distance_prior(self) -> None:
        loss = _build_motion_joint_loss()
        attn = torch.ones(1, 2, 3, 3, dtype=torch.float32)

        value = loss.compute_attention_regularization(attn, geomask=object())

        self.assertTrue(torch.is_tensor(value))
        self.assertTrue(bool(torch.isfinite(value).detach().cpu().item()))

    def test_attention_regularization_geomask_invalid_rank_raises(self) -> None:
        loss = _build_motion_joint_loss()
        attn = torch.ones(1, 2, 3, 3, dtype=torch.float32)

        self._assert_failure(
            lambda: loss.compute_attention_regularization(
                attn,
                geomask=torch.ones(3, dtype=torch.float32),
            ),
            expected_messages=(
                "geomask for attention regularization",
                "rank 2, 3, or 4",
                "actual_shape=(3,)",
                "actual_ndim=1",
            ),
        )

    def test_attention_regularization_geomask_rank2_bad_shape_raises(self) -> None:
        loss = _build_motion_joint_loss()
        attn = torch.ones(1, 2, 3, 3, dtype=torch.float32)

        self._assert_failure(
            lambda: loss.compute_attention_regularization(
                attn,
                geomask=torch.ones(3, 4, dtype=torch.float32),
            ),
            expected_messages=(
                "geomask rank-2 fallback reshape failed",
                "expected geomask shape (T, T)",
                "geomask_shape=(3, 4)",
                "attention_shape=(2, 3, 3)",
                "T=3",
            ),
        )

    def test_attention_regularization_geomask_rank3_bad_broadcast_raises(self) -> None:
        loss = _build_motion_joint_loss()
        attn = torch.ones(1, 2, 3, 3, dtype=torch.float32)

        self._assert_failure(
            lambda: loss.compute_attention_regularization(
                attn,
                geomask=torch.ones(4, 3, 3, dtype=torch.float32),
            ),
            expected_messages=(
                "geomask broadcast failed in attention regularization",
                "expected rank-3 geomask broadcastable",
                "geomask_shape=(4, 3, 3)",
                "attention_shape=(2, 3, 3)",
                "T=3",
            ),
        )

    def test_attention_regularization_geomask_rank4_bad_fallback_raises(self) -> None:
        loss = _build_motion_joint_loss()
        attn = torch.ones(1, 2, 3, 3, dtype=torch.float32)

        self._assert_failure(
            lambda: loss.compute_attention_regularization(
                attn,
                geomask=torch.ones(2, 4, 3, 3, dtype=torch.float32),
            ),
            expected_messages=(
                "geomask fallback broadcast failed in attention regularization",
                "rank-4 reducible via mean(0)",
                "original_geomask_shape=(2, 4, 3, 3)",
                "fallback_geomask_shape=(4, 3, 3)",
                "attention_shape=(2, 3, 3)",
                "T=3",
            ),
        )

    def test_motion_joint_loss_arm_weight_invalid_raises(self) -> None:
        self._assert_failure(
            lambda: _build_motion_joint_loss(
                direct_pose_loss_arm_weight="bad-arm-weight",
            ),
            expected_messages=(
                "direct_pose_loss_arm_weight",
                "range (0, inf)",
                "value='bad-arm-weight'",
                "type=str",
            ),
        )

    def test_motion_joint_loss_else_weight_invalid_raises(self) -> None:
        self._assert_failure(
            lambda: _build_motion_joint_loss(
                direct_pose_loss_else_weight=0.0,
            ),
            expected_messages=(
                "direct_pose_loss_else_weight",
                "range (0, inf)",
                "value=0.0",
            ),
        )

    def test_motion_joint_loss_group_norm_beta_invalid_raises(self) -> None:
        self._assert_failure(
            lambda: _build_motion_joint_loss(
                direct_pose_loss_group_norm_ema_beta=1.5,
            ),
            expected_messages=(
                "direct_pose_loss_group_norm_ema_beta",
                "(0.0, 0.9999]",
                "value=1.5",
            ),
        )

    def test_motion_joint_loss_group_norm_ratio_min_invalid_raises(self) -> None:
        self._assert_failure(
            lambda: _build_motion_joint_loss(
                direct_pose_loss_group_norm_ratio_min=0.0,
            ),
            expected_messages=(
                "direct_pose_loss_group_norm_ratio_min",
                "range (0, inf)",
                "value=0.0",
            ),
        )

    def test_motion_joint_loss_group_norm_ratio_max_invalid_raises(self) -> None:
        self._assert_failure(
            lambda: _build_motion_joint_loss(
                direct_pose_loss_group_norm_ratio_min=2.0,
                direct_pose_loss_group_norm_ratio_max=1.0,
            ),
            expected_messages=(
                "direct_pose_loss_group_norm_ratio_min/direct_pose_loss_group_norm_ratio_max",
                "ratio_min <= ratio_max",
                "direct_pose_loss_group_norm_ratio_min=2.0",
                "direct_pose_loss_group_norm_ratio_max=1.0",
            ),
        )

    def test_motion_joint_loss_group_norm_eps_invalid_raises(self) -> None:
        self._assert_failure(
            lambda: _build_motion_joint_loss(
                direct_pose_loss_group_norm_eps=0.0,
            ),
            expected_messages=(
                "direct_pose_loss_group_norm_eps",
                "range (0, inf)",
                "value=0.0",
            ),
        )

    def test_motion_joint_loss_ctor_skeleton_offsets_invalid_raises(self) -> None:
        self._assert_failure(
            lambda: _build_motion_joint_loss(
                meta={
                    "skeleton": {
                        "parents": [-1, 0],
                        "ref_local_offsets_m": [[0.0, 0.0], [1.0, 2.0]],
                    }
                },
            ),
            expected_messages=(
                "skeleton.ref_local_offsets_m",
                "shape=(num_joints, 3)",
                "actual_shape=(2, 2)",
            ),
        )

    def test_motion_joint_loss_set_skeleton_offsets_invalid_raises(self) -> None:
        loss = _build_motion_joint_loss()
        self._assert_failure(
            lambda: loss.set_skeleton(
                [-1, 0],
                [[0.0, 0.0], [1.0, 2.0]],
            ),
            expected_messages=(
                "offsets",
                "shape=(num_joints, 3)",
                "actual_shape=(2, 2)",
            ),
        )

    def test_motion_joint_loss_skeleton_cluster_direct_group_masks_exclude_root_and_overlaps(self) -> None:
        bone_names = ("pelvis", "thigh_l", "thigh_r", "upperarm_l", "spine")
        loss = _build_motion_joint_loss(
            num_joints=len(bone_names),
            bone_names=bone_names,
            direct_pose_leg_bones=("thigh_l", "thigh_r"),
            direct_pose_arm_split_enable=True,
            direct_pose_arm_bones=("upperarm_l", "thigh_l"),
        )
        loss.root_idx = 0

        masks = loss._resolve_direct_group_masks(len(bone_names), device=torch.device("cpu"))

        self.assertIsNotNone(masks)
        assert masks is not None
        self.assertEqual(masks["all_ex_root"].tolist(), [False, True, True, True, True])
        self.assertEqual(masks["leg"].tolist(), [False, True, True, False, False])
        self.assertEqual(masks["nonleg"].tolist(), [False, False, False, True, True])
        self.assertEqual(masks["arm"].tolist(), [False, False, False, True, False])
        self.assertEqual(masks["else"].tolist(), [False, False, False, False, True])
        self.assertEqual(masks["trunk"].tolist(), masks["else"].tolist())

    def test_motion_joint_loss_skeleton_cluster_weight_cache_invalidation_preserves_values(self) -> None:
        loss = _build_motion_joint_loss(
            num_joints=3,
            bone_names=("root", "mid", "end"),
            meta={
                "skeleton": {
                    "parents": [-1, 0, 1],
                    "ref_local_offsets_m": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                }
            },
        )

        with mock.patch.object(loss, "_warn_once") as warn_once:
            weights_first = loss._joint_weight_vector(torch.device("cpu"), torch.float32, 3)
            weights_second = loss._joint_weight_vector(torch.device("cpu"), torch.float32, 3)

        self.assertIs(weights_first, weights_second)
        warn_once.assert_called_once()
        self.assertTrue(bool((weights_first.std() > 0.0).detach().cpu().item()))

        loss._invalidate_weight_cache()
        with mock.patch.object(loss, "_warn_once") as warn_once_after:
            weights_recomputed = loss._joint_weight_vector(torch.device("cpu"), torch.float32, 3)

        warn_once_after.assert_not_called()
        self.assertTrue(torch.allclose(weights_first, weights_recomputed))

    def test_motion_joint_loss_skeleton_cluster_init_state_contract(self) -> None:
        loss = _build_motion_joint_loss(
            num_joints=3,
            bone_names=("root", "mid", "end"),
            meta={
                "skeleton": {
                    "parents": [-1, 0, 1],
                    "ref_local_offsets_m": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                }
            },
            w_attn_reg=0.2,
            w_rot_ortho=0.3,
            w_rot_local=0.4,
            w_root_vel=0.5,
            w_root_speed=0.6,
            w_contact_plan=0.7,
            w_contact_meas=0.8,
            w_direct_pose=0.9,
            w_omega_l2=1.1,
            adaptive_bone_weights=True,
        )

        self.assertEqual(loss.w_attn_reg, 0.2)
        self.assertEqual(loss.w_rot_ortho, 0.3)
        self.assertEqual(loss.w_rot_local, 0.4)
        self.assertEqual(loss.w_root_vel, 0.5)
        self.assertEqual(loss.w_root_speed, 0.6)
        self.assertEqual(loss.w_contact_plan, 0.7)
        self.assertEqual(loss.w_contact_meas, 0.8)
        self.assertEqual(loss.w_direct_pose, 0.9)
        self.assertEqual(loss.w_omega_l2, 1.1)
        self.assertTrue(loss.use_adaptive_weights)
        self.assertEqual(loss.parents, [-1, 0, 1])
        self.assertEqual(loss.root_idx, 0)
        self.assertEqual(loss._bone_name_to_idx, {"root": 0, "mid": 1, "end": 2})
        self.assertEqual(loss._loss_group_totals, {})
        self.assertEqual(loss._loss_group_alias["attn"], "aux")
        self.assertEqual(loss._loss_group_alias["direct_pose"], "core")
        self.assertIsInstance(loss._joint_weight_cache, dict)
        self.assertIsInstance(loss._tail_candidate_cache, dict)
        self.assertIsInstance(loss._tail_score_cache, dict)
        self.assertEqual(tuple(loss.bone_offsets.shape), (3, 3))

    def test_motion_joint_loss_skeleton_cluster_relative_rotation_helpers(self) -> None:
        loss = _build_motion_joint_loss(
            num_joints=2,
            bone_names=("root", "child"),
            meta={
                "skeleton": {
                    "parents": [-1, 0],
                    "ref_local_offsets_m": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                }
            },
        )
        root_rot = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        child_rot = torch.eye(3, dtype=torch.float32)
        rotations = torch.stack([root_rot, child_rot], dim=0)
        expected_child_rel = root_rot.transpose(-1, -2).matmul(child_rot)

        parent_relative = loss._parent_relative_matrices(rotations)
        root_relative = loss._root_relative(rotations)

        self.assertTrue(torch.allclose(parent_relative[0], root_rot))
        self.assertTrue(torch.allclose(parent_relative[1], expected_child_rel))
        self.assertTrue(torch.allclose(root_relative[0], child_rot))
        self.assertTrue(torch.allclose(root_relative[1], expected_child_rel))

    def test_motion_joint_loss_skeleton_cluster_limb_stats_helpers(self) -> None:
        loss = _build_motion_joint_loss(
            num_joints=4,
            bone_names=("pelvis", "upperarm_l", "spine", "thigh_r"),
        )
        geo = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

        masks = loss._resolve_limb_masks(4, torch.device("cpu"))
        stats = loss._collect_limb_local_stats(geo)

        self.assertIsNotNone(masks)
        assert masks is not None
        limb_mask, torso_mask = masks
        self.assertEqual(limb_mask.tolist(), [False, True, False, True])
        self.assertEqual(torso_mask.tolist(), [True, False, True, False])
        self.assertEqual(stats["rot_local_limb_count"], 2)
        self.assertEqual(stats["rot_local_torso_count"], 2)
        self.assertAlmostEqual(stats["rot_local_limb_deg"], 0.3 * 180.0 / torch.pi, places=5)
        self.assertAlmostEqual(stats["rot_local_torso_deg"], 0.2 * 180.0 / torch.pi, places=5)
        self.assertAlmostEqual(stats["rot_local_limb_over_torso"], 1.5, places=5)

    def test_motion_joint_loss_skeleton_cluster_tail_candidates_and_scores(self) -> None:
        loss = _build_motion_joint_loss(
            num_joints=5,
            bone_names=("pelvis", "upperarm_l", "spine", "leaf_a", "leaf_b"),
            meta={
                "skeleton": {
                    "parents": [-1, 0, 0, 2, 2],
                    "ref_local_offsets_m": [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                }
            },
        )

        limb_candidates = loss._rot_local_tail_candidates("limbs", 5, torch.device("cpu"), k=2)
        keybone_candidates = loss._rot_local_tail_candidates("keybones", 5, torch.device("cpu"), k=2)

        self.assertIsNotNone(limb_candidates)
        self.assertIsNotNone(keybone_candidates)
        assert limb_candidates is not None
        assert keybone_candidates is not None
        self.assertEqual(limb_candidates.tolist(), [1, 3, 4])
        self.assertEqual(keybone_candidates.tolist(), [0, 1, 3, 4])

        loss.rot_local_tail_select = "ema"
        loss.rot_local_tail_ema_beta = 0.5
        first = loss._rot_local_tail_scores(torch.tensor([1.0, 3.0], dtype=torch.float32)).clone()
        second = loss._rot_local_tail_scores(torch.tensor([3.0, 5.0], dtype=torch.float32))
        self.assertTrue(torch.allclose(first, torch.tensor([1.0, 3.0])))
        self.assertTrue(torch.allclose(second, torch.tensor([2.0, 4.0])))

    def test_motion_joint_loss_applicator_cluster_forward_stats_contract(self) -> None:
        output_layout = {
            "RootVelocity": {"start": 0, "size": 2},
            "BoneRotations6D": {"start": 2, "size": 12},
        }
        loss = MotionJointLoss(
            output_layout=output_layout,
            w_rot_ortho=1.0,
            w_rot_local=1.0,
            w_root_vel=1.0,
            w_root_speed=1.0,
            w_direct_pose=1.0,
            w_contact_plan=1.0,
            w_contact_meas=1.0,
            w_omega_l2=1.0,
            event_clock_lambda_entropy_weight=1.0,
            event_clock_lambda_prior_weight=1.0,
            event_clock_delta_z_l2_weight=1.0,
        )
        loss.set_bone_names(["root", "child"])
        loss.rot_local_tail_weight = 1.0
        loss.rot_local_tail_k = 1
        identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        rot_part = identity_rot6d.repeat(2).view(1, 1, 12)
        pred_out = torch.cat([torch.tensor([[[1.0, 2.0]]], dtype=torch.float32), rot_part], dim=-1)
        gt_motion = torch.cat([torch.zeros(1, 1, 2, dtype=torch.float32), rot_part.clone()], dim=-1)
        pred_motion = {
            "out": pred_out,
            "out_direct": gt_motion.clone(),
            "contacts_plan_logits": torch.zeros(1, 1, 2, dtype=torch.float32),
            "contacts_plan": torch.full((1, 1, 2), 0.5, dtype=torch.float32),
            "contacts_meas": torch.tensor([[[0.75, 0.25]]], dtype=torch.float32),
            "event_clock_lambda_logit": torch.zeros(1, 1, 1, dtype=torch.float32),
            "event_clock_dynamic_prior": torch.full((1, 1, 1), 0.25, dtype=torch.float32),
            "event_clock_delta_z": torch.ones(1, 1, 2, dtype=torch.float32),
            "omega_hat": torch.ones(1, 1, 2, 3, dtype=torch.float32),
        }
        batch = {"contacts": torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)}

        with mock.patch.object(loss, "_warn_once"):
            total_loss, stats = loss(pred_motion, gt_motion, batch=batch)

        expected_keys = {
            "rot_ortho_weighted",
            "rot_local_deg",
            "rot_local_tail_deg",
            "rot_local_tail_k",
            "root_vel_mse",
            "root_speed_mae",
            "direct_pose_objective",
            "direct_pose_weighted",
            "contact_plan_bce",
            "contact_plan_mse",
            "contact_plan_weighted",
            "event_clock_lambda_entropy_weighted",
            "event_clock_lambda_prior_weighted",
            "event_clock_delta_z_l2_weighted",
            "event_clock_lambda_mean",
            "contact_meas_mse",
            "contact_meas_weighted",
            "omega_l2_weighted",
            "loss_group/core",
            "loss_group/aux",
            "loss_group/long",
        }
        self.assertTrue(torch.is_tensor(total_loss))
        self.assertTrue(bool(torch.isfinite(total_loss).detach().cpu().item()))
        self.assertTrue(expected_keys.issubset(stats.keys()))
        self.assertAlmostEqual(stats["root_vel_mse"], 2.5, places=6)
        self.assertAlmostEqual(stats["root_speed_mae"], 5.0 ** 0.5, places=6)
        self.assertEqual(stats["rot_local_tail_k"], 1.0)
        self.assertEqual(stats["event_clock_lambda_mean"], 0.5)
        for key in expected_keys:
            self.assertTrue(torch.isfinite(torch.tensor(float(stats[key]))), msg=key)

    def test_motion_joint_loss_applicator_cluster_direct_pose_defaults_when_pred_not_dict(self) -> None:
        output_layout = {
            "RootVelocity": {"start": 0, "size": 2},
            "BoneRotations6D": {"start": 2, "size": 12},
        }
        loss = MotionJointLoss(
            output_layout=output_layout,
            w_direct_pose=1.0,
        )
        loss.set_bone_names(["root", "child"])
        identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        gt_motion = torch.cat(
            [
                torch.zeros(1, 1, 2, dtype=torch.float32),
                identity_rot6d.repeat(2).view(1, 1, 12),
            ],
            dim=-1,
        )

        total_loss, stats = loss(gt_motion.clone(), gt_motion)

        self.assertTrue(torch.is_tensor(total_loss))
        self.assertEqual(float(total_loss.detach().cpu()), 0.0)
        self.assertEqual(stats["direct_pose_objective"], 0.0)
        self.assertEqual(stats["direct_pose_weighted"], 0.0)
        self.assertEqual(stats["direct_pose_geo"], 0.0)
        self.assertEqual(stats["direct_pose_geo_deg"], 0.0)
        self.assertEqual(stats["direct_pose_split_active"], 0.0)
        self.assertEqual(stats["direct_pose_arm_split_active"], 0.0)
        self.assertEqual(stats["dir_group_norm_used"], 0.0)
        self.assertIn("loss_group/core", stats)
        self.assertIn("loss_group/aux", stats)
        self.assertIn("loss_group/long", stats)

    def test_motion_joint_loss_direct_pose_default_stats_key_contract(self) -> None:
        loss = _build_motion_joint_loss(
            direct_pose_loss_arm_weight=2.5,
            direct_pose_loss_else_weight=0.75,
        )

        defaults = loss._direct_pose_default_stats()
        extra_defaults = loss._direct_pose_extra_defaults()
        expected_default_keys = {
            "direct_pose_geo",
            "direct_pose_geo_deg",
            "direct_pose_objective",
            "direct_pose_weighted",
            "direct_pose_split_active",
            "direct_pose_arm_split_active",
            "dir_base",
            "dir_leg_base",
            "dir_nonleg_base",
            "dir_nonleg_effective_base",
            "dir_arm_base",
            "dir_else_base",
            "leg_over_nonleg",
            "leg_over_nonleg_effective",
            "arm_over_else",
            "direct_pose_arm_else_balance_active",
            "direct_pose_loss_arm_weight",
            "direct_pose_loss_else_weight",
            "dir_group_norm_used",
            "dir_group_norm_leg_raw",
            "dir_group_norm_nonleg_raw",
            "dir_group_norm_leg_clamped",
            "dir_group_norm_nonleg_clamped",
            "dir_group_norm_leg",
            "dir_group_norm_nonleg",
            "dir_group_norm_leg_ema",
            "dir_group_norm_nonleg_ema",
            "dir_group_norm_leg_hit_min",
            "dir_group_norm_leg_hit_max",
            "dir_group_norm_nonleg_hit_min",
            "dir_group_norm_nonleg_hit_max",
            "dir_group_norm_leg_hit_any",
            "dir_group_norm_nonleg_hit_any",
        }

        self.assertSetEqual(set(defaults.keys()), expected_default_keys)
        self.assertTupleEqual(loss._direct_pose_default_stat_keys(), _DIRECT_POSE_DEFAULT_STAT_KEYS)
        self.assertTupleEqual(loss._direct_pose_component_stat_keys(), _DIRECT_POSE_COMPONENT_STAT_KEYS)
        self.assertSetEqual(
            set(extra_defaults.keys()),
            expected_default_keys - {"direct_pose_objective", "direct_pose_weighted"},
        )
        self.assertEqual(defaults["direct_pose_objective"], 0.0)
        self.assertEqual(defaults["direct_pose_weighted"], 0.0)
        self.assertEqual(defaults["direct_pose_loss_arm_weight"], 2.5)
        self.assertEqual(defaults["direct_pose_loss_else_weight"], 0.75)
        self.assertEqual(defaults["dir_group_norm_used"], 0.0)
        self.assertTrue(bool(torch.isnan(torch.tensor(defaults["dir_base"])).item()))
        self.assertNotIn("direct_pose_objective", extra_defaults)
        self.assertNotIn("direct_pose_weighted", extra_defaults)

    def test_motion_joint_loss_direct_pose_component_stats_contract_default_path(self) -> None:
        loss = _build_motion_joint_loss()
        identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        gt_motion = identity_rot6d.repeat(2).view(1, 1, 12)
        stats: dict[str, float] = {}

        total_loss = loss._apply_direct_pose_component(
            gt_motion.new_tensor(0.0),
            stats,
            gt_motion.clone(),
            gt_motion,
            180.0 / torch.pi,
        )

        direct_keys = {
            key for key in stats
            if key.startswith("direct_pose") or key.startswith("dir_") or key.startswith("leg_over") or key.startswith("arm_over")
        }
        self.assertEqual(float(total_loss.detach().cpu()), 0.0)
        self.assertSetEqual(direct_keys, set(_DIRECT_POSE_DEFAULT_STAT_KEYS))

    def test_motion_joint_loss_prepare_direct_pose_pair_normalizes_2d_3d_inputs(self) -> None:
        loss = _build_motion_joint_loss()
        direct_2d = torch.arange(24, dtype=torch.float32).view(2, 12)
        gt_3d = torch.arange(72, dtype=torch.float32).view(2, 3, 12)
        direct_3d = torch.arange(96, dtype=torch.float32).view(2, 4, 12)
        gt_2d = torch.arange(24, dtype=torch.float32).view(2, 12)

        pair = loss._prepare_direct_pose_pair(direct_2d, gt_3d)
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(tuple(pair.direct_seq.shape), (2, 1, 12))
        self.assertEqual(tuple(pair.gt_direct.shape), (2, 1, 12))
        self.assertTrue(torch.equal(pair.direct_seq, direct_2d.unsqueeze(1)))
        self.assertTrue(torch.equal(pair.gt_direct, gt_3d[:, :1]))

        pair = loss._prepare_direct_pose_pair(direct_3d, gt_2d)
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(tuple(pair.direct_seq.shape), (2, 1, 12))
        self.assertEqual(tuple(pair.gt_direct.shape), (2, 1, 12))
        self.assertTrue(torch.equal(pair.direct_seq, direct_3d[:, :1]))
        self.assertTrue(torch.equal(pair.gt_direct, gt_2d.unsqueeze(1)))

        pair = loss._prepare_direct_pose_pair(direct_3d, gt_3d[:, :2])
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(tuple(pair.direct_seq.shape), (2, 2, 12))
        self.assertEqual(tuple(pair.gt_direct.shape), (2, 2, 12))
        self.assertTrue(torch.equal(pair.direct_seq, direct_3d[:, :2]))
        self.assertTrue(torch.equal(pair.gt_direct, gt_3d[:, :2]))

        self.assertIsNone(loss._prepare_direct_pose_pair(direct_3d.unsqueeze(0), gt_3d))

    def test_motion_joint_loss_direct_pose_payload_request_result_types(self) -> None:
        loss = _build_motion_joint_loss()
        identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        direct = identity_rot6d.repeat(2).view(1, 1, 12)
        result = loss._compute_direct_pose_payload_from_request(  # type: ignore[attr-defined]
            _DirectPosePayloadRequest(
                direct=direct,
                gt_motion=direct.clone(),
                deg_per_rad=180.0 / torch.pi,
            )
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(torch.is_tensor(result.objective))
        self.assertIn("direct_pose_geo", result.extra)
        public_payload = loss._compute_direct_pose_payload(direct, direct.clone(), 180.0 / torch.pi)
        self.assertIsNotNone(public_payload)
        assert public_payload is not None
        self.assertEqual(float(public_payload[0].detach().cpu()), float(result.objective.detach().cpu()))
        self.assertEqual(set(public_payload[1].keys()), set(result.extra.keys()))

    def test_motion_joint_loss_group_base_payload_arm_else_balance_contract(self) -> None:
        loss = _build_motion_joint_loss(
            direct_pose_arm_split_enable=True,
            direct_pose_loss_arm_else_balance_enable=True,
            direct_pose_loss_arm_weight=2.0,
            direct_pose_loss_else_weight=1.0,
        )

        payload = loss._compute_direct_pose_group_base_payload(
            dir_base=torch.tensor(3.0),
            dir_leg_base=torch.tensor(2.0),
            dir_nonleg_base=torch.tensor(4.0),
            dir_arm_base=torch.tensor(6.0),
            dir_else_base=torch.tensor(3.0),
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(torch.allclose(payload["dir_base"], torch.tensor(3.0)))
        self.assertTrue(torch.allclose(payload["dir_leg_base"], torch.tensor(2.0)))
        self.assertTrue(torch.allclose(payload["dir_nonleg_base"], torch.tensor(4.0)))
        self.assertTrue(torch.allclose(payload["dir_nonleg_effective_base"], torch.tensor(5.0)))
        self.assertTrue(torch.allclose(payload["dir_arm_base"], torch.tensor(6.0)))
        self.assertTrue(torch.allclose(payload["dir_else_base"], torch.tensor(3.0)))
        self.assertEqual(payload["direct_pose_arm_else_balance_active"], 1.0)
        self.assertEqual(payload["direct_pose_loss_arm_weight"], 2.0)
        self.assertEqual(payload["direct_pose_loss_else_weight"], 1.0)
        self.assertAlmostEqual(payload["leg_over_nonleg"], 0.5, places=6)
        self.assertAlmostEqual(payload["leg_over_nonleg_effective"], 0.4, places=6)
        self.assertAlmostEqual(payload["arm_over_else"], 2.0, places=6)

    def test_motion_joint_loss_group_norm_shared_can_skip_ema_update(self) -> None:
        loss = _build_motion_joint_loss(
            direct_pose_loss_group_norm_w_leg=2.0,
            direct_pose_loss_group_norm_w_nonleg=3.0,
            direct_pose_loss_group_norm_ema_beta=0.5,
        )
        loss._direct_pose_group_norm_ema = {
            "leg": torch.tensor(4.0, dtype=torch.float32),
            "nonleg": torch.tensor(5.0, dtype=torch.float32),
            "marker": "keep",
        }

        objective, payload, ema_update = loss._compute_direct_pose_group_norm_shared(
            torch.tensor(2.0, dtype=torch.float32),
            torch.tensor(6.0, dtype=torch.float32),
            torch.tensor(10.0, dtype=torch.float32),
            update_ema_state=False,
        )

        self.assertEqual(float(objective.detach().cpu()), 7.0)
        self.assertEqual(payload["dir_group_norm_leg_ema"], 4.0)
        self.assertEqual(payload["dir_group_norm_nonleg_ema"], 5.0)
        self.assertEqual(payload["dir_group_norm_w_leg"], 2.0)
        self.assertEqual(payload["dir_group_norm_w_nonleg"], 3.0)
        self.assertTrue(torch.allclose(ema_update["leg"], torch.tensor(3.0)))
        self.assertTrue(torch.allclose(ema_update["nonleg"], torch.tensor(7.5)))
        self.assertEqual(ema_update["marker"], "keep")
        self.assertTrue(torch.allclose(loss._direct_pose_group_norm_ema["leg"], torch.tensor(4.0)))
        self.assertTrue(torch.allclose(loss._direct_pose_group_norm_ema["nonleg"], torch.tensor(5.0)))
        self.assertEqual(loss._direct_pose_group_norm_ema["marker"], "keep")

    def test_motion_joint_loss_group_norm_request_result_types(self) -> None:
        loss = _build_motion_joint_loss(
            direct_pose_loss_group_norm_w_leg=2.0,
            direct_pose_loss_group_norm_w_nonleg=3.0,
            direct_pose_loss_group_norm_ema_beta=0.5,
        )
        request = _DirectPoseGroupNormRequest(
            dir_leg_base=torch.tensor(2.0, dtype=torch.float32),
            dir_nonleg_base=torch.tensor(6.0, dtype=torch.float32),
            dir_nonleg_effective_base=torch.tensor(10.0, dtype=torch.float32),
            update_ema_state=False,
        )

        result = loss._compute_direct_pose_group_norm_from_request(request)  # type: ignore[attr-defined]

        self.assertEqual(float(result.objective.detach().cpu()), 5.0)
        self.assertIn("dir_group_norm_w_leg", result.stats)
        self.assertIn("leg", result.ema_update)
        public_result = loss._compute_direct_pose_group_norm_shared(
            request.dir_leg_base,
            request.dir_nonleg_base,
            request.dir_nonleg_effective_base,
            update_ema_state=False,
        )
        self.assertEqual(float(public_result[0].detach().cpu()), float(result.objective.detach().cpu()))
        self.assertEqual(set(public_result[1].keys()), set(result.stats.keys()))

    def test_motion_joint_loss_prepare_forward_inputs_preserves_dict_and_tensor_contract(self) -> None:
        loss = _build_motion_joint_loss()
        pred_core = torch.randn(1, 2, 12, dtype=torch.float32)
        gt_motion = torch.randn(1, 2, 12, dtype=torch.float32)
        delta_motion = torch.randn(1, 2, 12, dtype=torch.float32)

        pm, gm, delta_pm, delta_fallback = loss._prepare_forward_inputs(
            {
                "out": pred_core,
                "delta": delta_motion,
                "_delta_fallback": True,
            },
            gt_motion,
        )
        self.assertIs(pm, pred_core)
        self.assertIs(gm, gt_motion)
        self.assertIs(delta_pm, delta_motion)
        self.assertTrue(delta_fallback)

        pm, gm, delta_pm, delta_fallback = loss._prepare_forward_inputs(pred_core, gt_motion)
        self.assertIs(pm, pred_core)
        self.assertIs(gm, gt_motion)
        self.assertIsNone(delta_pm)
        self.assertFalse(delta_fallback)

    def test_motion_joint_loss_prepare_aux_supervision_pair_aligns_steps_dtype_and_device(self) -> None:
        loss = _build_motion_joint_loss()
        pred = torch.arange(12, dtype=torch.float32).view(2, 3, 2)
        target = torch.arange(8, dtype=torch.float64).view(2, 2, 2)

        pred_seq, target_seq, steps = loss._prepare_aux_supervision_pair(pred, target)

        self.assertEqual(tuple(pred_seq.shape), (2, 3, 2))
        self.assertEqual(tuple(target_seq.shape), (2, 2, 2))
        self.assertEqual(steps, 2)
        self.assertEqual(target_seq.dtype, pred.dtype)
        self.assertEqual(target_seq.device, pred.device)
        self.assertTrue(torch.equal(pred_seq, pred))
        self.assertTrue(torch.equal(target_seq, target.to(dtype=pred.dtype)))

    def test_motion_joint_loss_submit_component_loss_tracks_group_stats_contract(self) -> None:
        loss = _build_motion_joint_loss()
        loss._init_loss_group_tracker()
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        stats: dict[str, float] = {}

        total_loss = loss._submit_component_loss(
            total_loss,
            stats=stats,
            name="root_vel",
            tensor=torch.tensor(2.0, dtype=torch.float32),
            weight=0.5,
            raw_key="root_vel_mse",
            weighted_key="root_vel_weighted",
        )
        total_loss = loss._submit_component_loss(
            total_loss,
            stats=stats,
            name="contact_plan",
            tensor=torch.tensor(3.0, dtype=torch.float32),
            weight=2.0,
            group="aux",
            raw_key="contact_plan_bce",
            weighted_key="contact_plan_weighted",
            extra={"contact_plan_mse": torch.tensor(1.5, dtype=torch.float32)},
        )
        loss_group_stats = loss._loss_group_stats()

        self.assertAlmostEqual(float(total_loss.detach().cpu()), 7.0, places=6)
        self.assertEqual(stats["root_vel_mse"], 2.0)
        self.assertEqual(stats["root_vel_weighted"], 1.0)
        self.assertEqual(stats["contact_plan_bce"], 3.0)
        self.assertEqual(stats["contact_plan_weighted"], 6.0)
        self.assertEqual(stats["contact_plan_mse"], 1.5)
        self.assertEqual(loss_group_stats["loss_group/core"], 1.0)
        self.assertEqual(loss_group_stats["loss_group/aux"], 6.0)
        self.assertEqual(loss_group_stats["loss_group/long"], 0.0)

    def test_motion_joint_loss_finalize_forward_outputs_adds_loss_group_stats(self) -> None:
        loss = _build_motion_joint_loss()
        total_loss = torch.tensor(5.0, dtype=torch.float32)
        stats = {"base": 1.0}
        loss._loss_group_totals = {"core": 2.0, "aux": 3.0, "long": 4.0}

        finalized_loss, finalized_stats = loss._finalize_forward_outputs(total_loss, stats)

        self.assertIs(finalized_loss, total_loss)
        self.assertIs(finalized_stats, stats)
        self.assertEqual(finalized_stats["base"], 1.0)
        self.assertEqual(finalized_stats["loss_group/core"], 2.0)
        self.assertEqual(finalized_stats["loss_group/aux"], 3.0)
        self.assertEqual(finalized_stats["loss_group/long"], 4.0)

    def test_motion_joint_loss_motion_component_dispatch_order_regression(self) -> None:
        loss = _build_motion_joint_loss()
        total_loss = torch.tensor(1.0, dtype=torch.float32)
        stats: dict[str, float] = {}
        pred_motion = torch.randn(1, 1, 12, dtype=torch.float32)
        gt_motion = torch.randn(1, 1, 12, dtype=torch.float32)
        delta_motion = torch.randn(1, 1, 12, dtype=torch.float32)
        call_order: list[str] = []

        def _rot_ortho(total, stats_arg, pred_arg, delta_arg, delta_fallback):
            call_order.append("rot_ortho")
            self.assertIs(stats_arg, stats)
            self.assertIs(pred_arg, pred_motion)
            self.assertIs(delta_arg, delta_motion)
            self.assertTrue(delta_fallback)
            return total + torch.tensor(2.0, dtype=torch.float32)

        def _rot_local(total, stats_arg, pred_arg, gt_arg, deg_per_rad):
            call_order.append("rot_local")
            self.assertEqual(call_order, ["rot_ortho", "rot_local"])
            self.assertIs(stats_arg, stats)
            self.assertIs(pred_arg, pred_motion)
            self.assertIs(gt_arg, gt_motion)
            self.assertAlmostEqual(deg_per_rad, 180.0 / torch.pi, places=6)
            return total + torch.tensor(3.0, dtype=torch.float32)

        def _root_velocity(total, stats_arg, pred_arg, gt_arg):
            call_order.append("root_velocity")
            self.assertEqual(call_order, ["rot_ortho", "rot_local", "root_velocity"])
            self.assertIs(stats_arg, stats)
            self.assertIs(pred_arg, pred_motion)
            self.assertIs(gt_arg, gt_motion)
            return total + torch.tensor(4.0, dtype=torch.float32)

        with (
            mock.patch.object(loss, "_apply_rot_ortho_component", side_effect=_rot_ortho),
            mock.patch.object(loss, "_apply_rot_local_component", side_effect=_rot_local),
            mock.patch.object(loss, "_apply_root_velocity_components", side_effect=_root_velocity),
        ):
            result = loss._apply_motion_components(
                total_loss,
                stats,
                pred_motion,
                gt_motion,
                delta_motion,
                True,
                180.0 / torch.pi,
            )

        self.assertEqual(call_order, ["rot_ortho", "rot_local", "root_velocity"])
        self.assertAlmostEqual(float(result.detach().cpu()), 10.0, places=6)

    def test_motion_joint_loss_aux_component_dispatch_order_regression(self) -> None:
        loss = _build_motion_joint_loss()
        total_loss = torch.tensor(1.0, dtype=torch.float32)
        stats: dict[str, float] = {}
        pred_motion = {"out": torch.zeros(1, 1, 12, dtype=torch.float32)}
        batch = {"contacts": torch.zeros(1, 1, 2, dtype=torch.float32)}
        call_order: list[str] = []

        def _contact_plan(total, stats_arg, pred_arg, batch_arg):
            call_order.append("contact_plan")
            self.assertIs(stats_arg, stats)
            self.assertIs(pred_arg, pred_motion)
            self.assertIs(batch_arg, batch)
            return total + torch.tensor(2.0, dtype=torch.float32)

        def _event_clock(total, stats_arg, pred_arg):
            call_order.append("event_clock")
            self.assertEqual(call_order, ["contact_plan", "event_clock"])
            self.assertIs(stats_arg, stats)
            self.assertIs(pred_arg, pred_motion)
            return total + torch.tensor(3.0, dtype=torch.float32)

        def _contact_meas(total, stats_arg, pred_arg, batch_arg):
            call_order.append("contact_meas")
            self.assertEqual(call_order, ["contact_plan", "event_clock", "contact_meas"])
            self.assertIs(stats_arg, stats)
            self.assertIs(pred_arg, pred_motion)
            self.assertIs(batch_arg, batch)
            return total + torch.tensor(4.0, dtype=torch.float32)

        def _omega_l2(total, stats_arg, pred_arg):
            call_order.append("omega_l2")
            self.assertEqual(call_order, ["contact_plan", "event_clock", "contact_meas", "omega_l2"])
            self.assertIs(stats_arg, stats)
            self.assertIs(pred_arg, pred_motion)
            return total + torch.tensor(5.0, dtype=torch.float32)

        with (
            mock.patch.object(loss, "_apply_contact_plan_component", side_effect=_contact_plan),
            mock.patch.object(loss, "_apply_event_clock_components", side_effect=_event_clock),
            mock.patch.object(loss, "_apply_contact_meas_component", side_effect=_contact_meas),
            mock.patch.object(loss, "_apply_omega_l2_component", side_effect=_omega_l2),
        ):
            result = loss._apply_aux_components(total_loss, stats, pred_motion, batch)

        self.assertEqual(call_order, ["contact_plan", "event_clock", "contact_meas", "omega_l2"])
        self.assertAlmostEqual(float(result.detach().cpu()), 15.0, places=6)

    def test_motion_joint_loss_forward_dispatch_order_regression(self) -> None:
        loss = _build_motion_joint_loss()
        pred_motion = {
            "out": torch.randn(1, 1, 12, dtype=torch.float32),
            "delta": torch.randn(1, 1, 12, dtype=torch.float32),
            "_delta_fallback": True,
        }
        gt_motion = torch.randn(1, 1, 12, dtype=torch.float32)
        batch = {"contacts": torch.zeros(1, 1, 2, dtype=torch.float32)}
        pred_core_motion = torch.full((1, 1, 12), 3.0, dtype=torch.float32)
        gt_core_motion = torch.full((1, 1, 12), 4.0, dtype=torch.float32)
        delta_motion = torch.full((1, 1, 12), 5.0, dtype=torch.float32)
        call_order: list[str] = []

        def _init_tracker() -> None:
            call_order.append("init")

        def _prepare(pred, gt):
            call_order.append("prep")
            self.assertIs(pred, pred_motion)
            self.assertIs(gt, gt_motion)
            return pred_core_motion, gt_core_motion, delta_motion, True

        def _run_base(pred, gt, attn_weights=None):
            call_order.append("base")
            self.assertIs(pred, pred_core_motion)
            self.assertIs(gt, gt_core_motion)
            self.assertEqual(attn_weights, "sentinel-attn")
            return gt_core_motion.new_tensor(1.0), {"base": 1.0}

        def _apply_motion(total, stats, pred, gt, delta, delta_fallback, deg_per_rad):
            call_order.append("motion")
            self.assertEqual(call_order, ["init", "prep", "base", "motion"])
            self.assertIs(pred, pred_core_motion)
            self.assertIs(gt, gt_core_motion)
            self.assertIs(delta, delta_motion)
            self.assertTrue(delta_fallback)
            self.assertAlmostEqual(deg_per_rad, 180.0 / torch.pi, places=6)
            self.assertEqual(stats, {"base": 1.0})
            return total + gt_core_motion.new_tensor(2.0)

        def _apply_direct(total, stats, pred, gt, deg_per_rad):
            call_order.append("direct")
            self.assertEqual(call_order, ["init", "prep", "base", "motion", "direct"])
            self.assertIs(pred, pred_motion)
            self.assertIs(gt, gt_core_motion)
            self.assertAlmostEqual(deg_per_rad, 180.0 / torch.pi, places=6)
            self.assertEqual(stats, {"base": 1.0})
            return total + gt_core_motion.new_tensor(3.0)

        def _apply_aux(total, stats, pred, batch_arg):
            call_order.append("aux")
            self.assertEqual(call_order, ["init", "prep", "base", "motion", "direct", "aux"])
            self.assertIs(pred, pred_motion)
            self.assertIs(batch_arg, batch)
            self.assertEqual(stats, {"base": 1.0})
            return total + gt_core_motion.new_tensor(4.0)

        def _finalize(total, stats):
            call_order.append("finalize")
            self.assertEqual(call_order, ["init", "prep", "base", "motion", "direct", "aux", "finalize"])
            self.assertAlmostEqual(float(total.detach().cpu()), 10.0, places=6)
            self.assertEqual(stats, {"base": 1.0})
            return total, {"base": 1.0, "finalized": 1.0}

        with (
            mock.patch.object(loss, "_init_loss_group_tracker", side_effect=_init_tracker),
            mock.patch.object(loss, "_prepare_forward_inputs", side_effect=_prepare),
            mock.patch.object(loss, "_run_forward_base", side_effect=_run_base),
            mock.patch.object(loss, "_apply_motion_components", side_effect=_apply_motion),
            mock.patch.object(loss, "_apply_direct_pose_component", side_effect=_apply_direct),
            mock.patch.object(loss, "_apply_aux_components", side_effect=_apply_aux),
            mock.patch.object(loss, "_finalize_forward_outputs", side_effect=_finalize),
        ):
            total_loss, stats = loss.forward(
                pred_motion,
                gt_motion,
                attn_weights="sentinel-attn",
                batch=batch,
            )

        self.assertEqual(call_order, ["init", "prep", "base", "motion", "direct", "aux", "finalize"])
        self.assertAlmostEqual(float(total_loss.detach().cpu()), 10.0, places=6)
        self.assertEqual(stats, {"base": 1.0, "finalized": 1.0})

    def test_motion_joint_loss_direct_pose_component_stats_contract_group_norm_path(self) -> None:
        loss = MotionJointLoss(
            output_layout={"BoneRotations6D": {"start": 0, "size": 18}},
            w_direct_pose=1.0,
            direct_pose_loss_leg_split=True,
            direct_pose_loss_group_norm_enable=True,
            direct_pose_loss_group_norm_w_leg=2.0,
            direct_pose_loss_group_norm_w_nonleg=3.0,
        )
        loss.set_bone_names(["root", "thigh_l", "arm_l"])
        identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        gt_motion = identity_rot6d.repeat(3).view(1, 1, 18)
        pred_motion = {"out_direct": gt_motion.clone()}
        stats: dict[str, float] = {}

        total_loss = loss._apply_direct_pose_component(
            gt_motion.new_tensor(0.0),
            stats,
            pred_motion,
            gt_motion,
            180.0 / torch.pi,
        )

        direct_keys = {
            key for key in stats
            if key.startswith("direct_pose") or key.startswith("dir_") or key.startswith("leg_over") or key.startswith("arm_over")
        }
        self.assertTrue(bool(torch.isfinite(total_loss).detach().cpu().item()))
        self.assertSetEqual(direct_keys, set(_DIRECT_POSE_COMPONENT_STAT_KEYS))
        self.assertEqual(stats["dir_group_norm_w_leg"], 2.0)
        self.assertEqual(stats["dir_group_norm_w_nonleg"], 3.0)
        self.assertIn("direct_pose_objective", stats)
        self.assertIn("direct_pose_weighted", stats)

    def test_torch_dynamo_probe_safe_handles_missing_and_runtime_failure(self) -> None:
        with mock.patch("train.models.torch", SimpleNamespace(), create=False):
            self.assertFalse(_torch_dynamo_is_compiling_safe())

        with mock.patch("train.models.torch", SimpleNamespace(_dynamo=SimpleNamespace(is_compiling=mock.Mock(return_value=True))), create=False):
            self.assertTrue(_torch_dynamo_is_compiling_safe())

        with mock.patch(
            "train.models.torch",
            SimpleNamespace(_dynamo=SimpleNamespace(is_compiling=mock.Mock(side_effect=RuntimeError("sentinel dynamo failure")))),
            create=False,
        ):
            self.assertFalse(_torch_dynamo_is_compiling_safe())

    def test_torch_onnx_probe_safe_handles_missing_and_runtime_failure(self) -> None:
        with mock.patch("train.models.torch", SimpleNamespace(), create=False):
            self.assertFalse(_torch_onnx_is_in_export_safe())

        with mock.patch(
            "train.models.torch",
            SimpleNamespace(onnx=SimpleNamespace(is_in_onnx_export=mock.Mock(return_value=True))),
            create=False,
        ):
            self.assertTrue(_torch_onnx_is_in_export_safe())

        with mock.patch(
            "train.models.torch",
            SimpleNamespace(onnx=SimpleNamespace(is_in_onnx_export=mock.Mock(side_effect=RuntimeError("sentinel onnx failure")))),
            create=False,
        ):
            self.assertFalse(_torch_onnx_is_in_export_safe())

    def test_motion_joint_loss_rot6d_cluster_extract_and_matrix_helpers(self) -> None:
        loss = MotionJointLoss(
            output_layout={"BoneRotations6D": {"start": 0, "size": 12}},
        )
        loss.set_bone_names(["root", "child"])
        raw = torch.arange(12, dtype=torch.float32).view(1, 1, 12) / 10.0
        loss.mu_y = torch.ones(12, dtype=torch.float32)
        loss.std_y = torch.full((12,), 2.0, dtype=torch.float32)

        with mock.patch.object(loss, "_warn_once"):
            extracted = loss._maybe_get_rot6d(raw)
            flat = loss._extract_rot6d_flat(raw, denorm=True, reproject=False, sanitize=False)
            mats = loss._extract_rot6d_mats(raw, denorm=False, reproject=True, sanitize=True)

        self.assertTrue(torch.equal(extracted, raw))
        self.assertTrue(torch.allclose(flat, raw * 2.0 + 1.0))
        self.assertIsNotNone(mats)
        assert mats is not None
        self.assertEqual(tuple(mats.shape), (1, 1, 2, 3, 3))
        self.assertTrue(bool(torch.isfinite(mats).all().detach().cpu().item()))

    def test_motion_joint_loss_rot6d_cluster_objective_helpers(self) -> None:
        loss = MotionJointLoss(
            output_layout={"BoneRotations6D": {"start": 0, "size": 12}},
        )
        loss.set_bone_names(["root", "child"])
        identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        seq = identity_rot6d.repeat(4).view(1, 2, 12)

        with mock.patch.object(loss, "_warn_once"):
            ortho = loss.compute_rot6d_ortho_loss(seq)
            geo, theta, weights = loss.compute_rot6d_geo_loss(seq, seq, return_per_joint=True)
            mats = loss._rot6d_matrices(seq)
            log_loss = loss.compute_rot6d_log_loss(seq, seq)

        self.assertEqual(float(ortho.detach().cpu()), 0.0)
        self.assertEqual(float(geo.detach().cpu()), 0.0)
        self.assertIsNotNone(mats)
        assert mats is not None
        self.assertEqual(tuple(mats.shape), (1, 2, 2, 3, 3))
        self.assertEqual(tuple(theta.shape), (1, 2, 2))
        self.assertEqual(int(weights.numel()), 2)
        self.assertTrue(torch.allclose(theta, torch.zeros_like(theta)))
        self.assertEqual(float(log_loss.detach().cpu()), 0.0)

    def test_motion_joint_loss_payload_arm_weight_override_invalid_raises(self) -> None:
        loss = _build_motion_joint_loss()
        self._assert_failure(
            lambda: loss._compute_direct_pose_group_base_payload(
                dir_base=torch.tensor(1.0),
                dir_leg_base=torch.tensor(2.0),
                dir_nonleg_base=torch.tensor(3.0),
                dir_arm_base=torch.tensor(4.0),
                dir_else_base=torch.tensor(5.0),
                arm_weight="bad-arm-override",
            ),
            expected_messages=(
                "arm_weight",
                "direct-pose group payload",
                "range (0, inf)",
                "value='bad-arm-override'",
            ),
        )

    def test_motion_joint_loss_payload_else_weight_override_invalid_raises(self) -> None:
        loss = _build_motion_joint_loss()
        self._assert_failure(
            lambda: loss._compute_direct_pose_group_base_payload(
                dir_base=torch.tensor(1.0),
                dir_leg_base=torch.tensor(2.0),
                dir_nonleg_base=torch.tensor(3.0),
                dir_arm_base=torch.tensor(4.0),
                dir_else_base=torch.tensor(5.0),
                else_weight=0.0,
            ),
            expected_messages=(
                "else_weight",
                "direct-pose group payload",
                "range (0, inf)",
                "value=0.0",
            ),
        )

    def test_motion_joint_loss_group_norm_runtime_scalar_invalid_raises(self) -> None:
        loss = _build_motion_joint_loss()
        self._assert_failure(
            lambda: loss._compute_direct_pose_group_norm_shared(
                torch.tensor(1.0),
                torch.tensor(2.0),
                torch.tensor(3.0),
                direct_group_beta="bad-beta",
            ),
            expected_messages=(
                "direct_group_beta",
                "(0.0, 0.9999]",
                "value='bad-beta'",
                "type=str",
            ),
        )


if __name__ == "__main__":
    unittest.main()
