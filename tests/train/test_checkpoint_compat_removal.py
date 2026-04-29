from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from train import posttrain as _posttrain
from train import posttrain_build_shell as _posttrain_build_shell
from train.configuration import model_build as _model_build_cfg
from train.checkpoint.load_schema import (
    DirectPoseBuildConfig,
    DirectPoseLoadCompatOptions,
    RemovedCheckpointCompatError,
    RetiredDirectPoseLayoutError,
    normalize_and_validate_direct_pose_ckpt_for_load,
    normalize_direct_pose_split_state_dict_schema,
    prepare_event_motion_ckpt_state_for_load,
    resume_load_weights_compat,
)


class _DummyEventMotionModel(torch.nn.Module):
    def __init__(
        self,
        model_state: dict[str, torch.Tensor] | None = None,
        *,
        split_enable: bool = False,
    ) -> None:
        super().__init__()
        self._model_state = dict(model_state or {})
        self._split_enable = bool(split_enable)

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return dict(self._model_state)

    def _direct_pose_split_state(self) -> dict[str, object] | None:
        if not self._split_enable:
            return None
        return {
            "arm_split": False,
            "idx_leg": torch.tensor([0, 1], dtype=torch.long),
            "idx_nonleg": torch.tensor([2, 3], dtype=torch.long),
            "idx_arm": None,
            "idx_else": None,
        }


def _direct_pose_cfg(**overrides: object) -> DirectPoseBuildConfig:
    values = dict(
        enable=True,
        hidden=32,
        meas_mode="concat",
        feat_source="cond",
        time_pe_dim=0,
        time_pe_base=10000.0,
        use_phase_z=False,
        phase_z_mode="concat",
        split_enable=False,
        arm_split_enable=False,
        arm_bones=None,
        nonleg_proj_dim=0,
        drop_ckpt_weights=False,
    )
    values.update(overrides)
    return DirectPoseBuildConfig(**values)


def _load_options(**overrides: object) -> DirectPoseLoadCompatOptions:
    values = dict(train_direct_pose=True, leg_enable=False, leg_bones=None)
    values.update(overrides)
    return DirectPoseLoadCompatOptions(**values)


class CheckpointCompatRemovalTest(unittest.TestCase):
    def test_strict_payload_rejects_missing_direct_pose_meas_mode(self) -> None:
        payload = {
            "ckpt_in": "/tmp/nonexistent_strict_direct_pose_missing_meas.pt",
            "load_context": "resume",
            "strict_current_model_build": True,
            "train_direct_pose": True,
            "event_clock": "on",
            "width": 32,
            "direct_pose_enable": True,
            "direct_pose_hidden": 32,
            "direct_pose_feat_source": "cond",
            "direct_pose_time_pe_dim": 0,
        }
        with self.assertRaisesRegex(
            SystemExit,
            "direct_pose_meas_mode.*2026-04-28 strict direct-pose shape-inference unload.*no checkpoint shape/posttrain_cfg replacement",
        ):
            _posttrain._cfg_from_payload(payload)

    def test_strict_payload_rejects_retired_direct_pose_override_shim(self) -> None:
        payload = {
            "ckpt_in": "/tmp/nonexistent_strict_direct_pose_override.pt",
            "load_context": "resume",
            "strict_current_model_build": True,
            "train_direct_pose": True,
            "event_clock": "on",
            "width": 32,
            "direct_pose_enable": True,
            "direct_pose_hidden": 32,
            "direct_pose_meas_mode": "concat",
            "direct_pose_meas_mode_override": "concat",
            "direct_pose_feat_source": "cond",
            "direct_pose_time_pe_dim": 0,
        }
        with self.assertRaisesRegex(
            SystemExit,
            "direct_pose_meas_mode_override.*2026-04-28 strict direct-pose shape-inference unload.*no checkpoint shape/posttrain_cfg replacement",
        ):
            _posttrain._cfg_from_payload(payload)

    def test_strict_model_build_ignores_absent_retired_direct_pose_override_attrs(self) -> None:
        payload = {
            "ckpt_in": "/tmp/nonexistent_strict_direct_pose_boundary.pt",
            "load_context": "resume",
            "strict_current_model_build": True,
            "train_direct_pose": True,
            "event_clock": "on",
            "width": 32,
            "direct_pose_enable": True,
            "direct_pose_hidden": 32,
            "direct_pose_meas_mode": "concat",
            "direct_pose_feat_source": "cond",
            "direct_pose_time_pe_dim": 0,
        }
        cfg = _posttrain._cfg_from_payload(payload)
        self.assertIsNone(cfg.direct_pose_hidden_override)
        self.assertIsNone(cfg.direct_pose_meas_mode_override)
        _model_build_cfg._reject_strict_current_direct_pose_shape_inference_inputs(cfg)

    def test_strict_model_build_still_rejects_non_none_retired_direct_pose_override_attrs(self) -> None:
        payload = {
            "ckpt_in": "/tmp/nonexistent_strict_direct_pose_boundary_override.pt",
            "load_context": "resume",
            "strict_current_model_build": True,
            "train_direct_pose": True,
            "event_clock": "on",
            "width": 32,
            "direct_pose_enable": True,
            "direct_pose_hidden": 32,
            "direct_pose_meas_mode": "concat",
            "direct_pose_feat_source": "cond",
            "direct_pose_time_pe_dim": 0,
        }
        cfg = _posttrain._cfg_from_payload(payload)
        object.__setattr__(cfg, "direct_pose_hidden_override", 64)
        with self.assertRaisesRegex(
            SystemExit,
            "direct_pose_hidden_override.*2026-04-28 strict direct-pose shape-inference unload.*no checkpoint shape/posttrain_cfg replacement",
        ):
            _model_build_cfg._reject_strict_current_direct_pose_shape_inference_inputs(cfg)

    def test_payload_rejects_removed_legacy_checkpoint_compat_field(self) -> None:
        payload = {
            "ckpt_in": "/tmp/nonexistent_removed_legacy_flag.pt",
            "load_context": "resume",
            "legacy_checkpoint_compat": False,
            "train_direct_pose": True,
            "event_clock": "on",
            "width": 32,
            "direct_pose_enable": True,
            "direct_pose_hidden": 32,
            "direct_pose_meas_mode": "concat",
            "direct_pose_feat_source": "cond",
            "direct_pose_time_pe_dim": 0,
        }
        with self.assertRaisesRegex(
            SystemExit,
            "config field `legacy_checkpoint_compat` is removed.*tools/migrate_legacy_posttrain_ckpt.py",
        ):
            _posttrain._cfg_from_payload(payload)

    def test_payload_rejects_removed_non_strict_runtime_flag(self) -> None:
        payload = {
            "ckpt_in": "/tmp/nonexistent_removed_non_strict.pt",
            "load_context": "resume",
            "strict_current_model_build": False,
            "train_direct_pose": True,
            "event_clock": "on",
            "width": 32,
            "direct_pose_enable": True,
            "direct_pose_hidden": 32,
            "direct_pose_meas_mode": "concat",
            "direct_pose_feat_source": "cond",
            "direct_pose_time_pe_dim": 0,
        }
        with self.assertRaisesRegex(
            SystemExit,
            "strict_current_model_build=false.*strict/current-only",
        ):
            _posttrain._cfg_from_payload(payload)

    def test_strict_shape_validator_rejects_direct_pose_mismatch(self) -> None:
        model = _DummyEventMotionModel({"direct_pose_head.0.weight": torch.zeros(4, 8)})
        with self.assertRaises(SystemExit) as exc:
            _posttrain_build_shell._validate_strict_current_direct_pose_checkpoint_shapes(
                model=model,
                state_dict={"direct_pose_head.0.weight": torch.zeros(4, 6)},
                stage="unit-test",
            )
        message = str(exc.exception)
        self.assertIn("direct_pose_head.0.weight", message)
        self.assertIn("2026-04-28 strict direct-pose shape-inference unload", message)
        self.assertIn("no load-time shape/posttrain_cfg replacement", message)

    def test_resume_rejects_retired_direct_pose_out_leg_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "old.pt"
            torch.save({"model": {"direct_pose_out_leg.weight": torch.zeros(2, 2)}}, ckpt_path)

            with self.assertRaisesRegex(
                RetiredDirectPoseLayoutError,
                "direct_pose_out_leg\\.\\*.*semantic checkpoint compat removal.*no in-loader replacement",
            ):
                resume_load_weights_compat(torch.nn.Linear(1, 1), str(ckpt_path))

    def test_direct_helper_rejects_monolithic_direct_pose_head_for_split_model(self) -> None:
        state_dict = {
            "direct_pose_head.6.weight": torch.zeros(4, 8),
            "direct_pose_head.6.bias": torch.zeros(4),
        }
        model = _DummyEventMotionModel(split_enable=True)

        with self.assertRaisesRegex(
            RemovedCheckpointCompatError,
            "direct_pose_head\\.6\\.weight.*split readout.*no in-loader replacement",
        ):
            normalize_direct_pose_split_state_dict_schema(model, state_dict)

        self.assertIn("direct_pose_head.6.weight", state_dict)
        self.assertIn("direct_pose_head.6.bias", state_dict)

    def test_prepare_rejects_direct_pose_weight_drop_path(self) -> None:
        state_dict = {"direct_pose_head.0.weight": torch.zeros(4, 6)}
        model = _DummyEventMotionModel({"direct_pose_head.0.weight": torch.zeros(4, 6)})

        with self.assertRaisesRegex(
            RemovedCheckpointCompatError,
            "direct_pose_\\*.*reinit/shape override.*semantic tensor drop.*no in-loader replacement",
        ):
            prepare_event_motion_ckpt_state_for_load(
                state_dict=state_dict,
                model=model,
                ckpt_posttrain_cfg=None,
                contact_dim=1,
                direct_pose_cfg=_direct_pose_cfg(drop_ckpt_weights=True),
                load_options=_load_options(),
            )

        self.assertIn("direct_pose_head.0.weight", state_dict)

    def test_apply_rejects_phase_z_direct_pose_input_dim_mismatch(self) -> None:
        state_dict = {"direct_pose_head.0.weight": torch.zeros(4, 6)}
        model = _DummyEventMotionModel({"direct_pose_head.0.weight": torch.zeros(4, 8)})

        with self.assertRaisesRegex(
            RemovedCheckpointCompatError,
            "direct_pose_head\\.0\\.weight.*phase_z input-dimension.*no in-loader replacement",
        ):
            normalize_and_validate_direct_pose_ckpt_for_load(
                state_dict=state_dict,
                model=model,
                ckpt_posttrain_cfg=None,
                contact_dim=1,
                direct_pose_cfg=_direct_pose_cfg(use_phase_z=True, phase_z_mode="concat"),
                load_options=_load_options(),
                context="apply_direct_pose_ckpt_compat",
            )

    def test_apply_rejects_replace_contacts_phase_tail_mapping(self) -> None:
        src_head = torch.arange(4 * 47, dtype=torch.float32).reshape(4, 47)
        src_leg_head = (1000 + torch.arange(4 * 47, dtype=torch.float32)).reshape(4, 47)
        state_dict = {
            "direct_pose_head.0.weight": src_head.clone(),
            "direct_pose_leg_head.0.weight": src_leg_head.clone(),
        }
        model = _DummyEventMotionModel(
            {
                "direct_pose_head.0.weight": torch.full((4, 43), -1.0),
                "direct_pose_leg_head.0.weight": torch.full((4, 43), -1.0),
            }
        )

        with self.assertRaisesRegex(
            RemovedCheckpointCompatError,
            "direct_pose_head\\.0\\.weight.*phase_z input-dimension.*no in-loader replacement",
        ):
            normalize_and_validate_direct_pose_ckpt_for_load(
                state_dict=state_dict,
                model=model,
                ckpt_posttrain_cfg={
                    "direct_pose_feat_source": "cond",
                    "direct_pose_time_pe_dim": 32,
                    "direct_pose_use_phase_z": True,
                    "direct_pose_phase_z_mode": "concat",
                },
                contact_dim=2,
                direct_pose_cfg=_direct_pose_cfg(
                    use_phase_z=True,
                    phase_z_mode="replace_contacts",
                    feat_source="cond",
                    time_pe_dim=32,
                ),
                load_options=_load_options(leg_enable=True),
                context="apply_direct_pose_ckpt_compat",
            )

    def test_apply_replace_contacts_still_rejects_non_semantic_shape_case(self) -> None:
        state_dict = {"direct_pose_head.0.weight": torch.zeros(4, 48)}
        model = _DummyEventMotionModel({"direct_pose_head.0.weight": torch.zeros(4, 43)})

        with self.assertRaisesRegex(
            RemovedCheckpointCompatError,
            "direct_pose_head\\.0\\.weight.*phase_z input-dimension.*no in-loader replacement",
        ):
            normalize_and_validate_direct_pose_ckpt_for_load(
                state_dict=state_dict,
                model=model,
                ckpt_posttrain_cfg={
                    "direct_pose_feat_source": "cond",
                    "direct_pose_time_pe_dim": 32,
                    "direct_pose_use_phase_z": True,
                    "direct_pose_phase_z_mode": "concat",
                },
                contact_dim=2,
                direct_pose_cfg=_direct_pose_cfg(
                    use_phase_z=True,
                    phase_z_mode="replace_contacts",
                    feat_source="cond",
                    time_pe_dim=32,
                ),
                load_options=_load_options(),
                context="apply_direct_pose_ckpt_compat",
            )

    def test_apply_replace_contacts_rejects_feat_source_mismatch(self) -> None:
        state_dict = {"direct_pose_head.0.weight": torch.zeros(4, 47)}
        model = _DummyEventMotionModel({"direct_pose_head.0.weight": torch.zeros(4, 43)})

        with self.assertRaisesRegex(
            RemovedCheckpointCompatError,
            "direct_pose_head\\.0\\.weight.*phase_z input-dimension.*no in-loader replacement",
        ):
            normalize_and_validate_direct_pose_ckpt_for_load(
                state_dict=state_dict,
                model=model,
                ckpt_posttrain_cfg={
                    "direct_pose_feat_source": "hidden",
                    "direct_pose_time_pe_dim": 32,
                    "direct_pose_use_phase_z": True,
                    "direct_pose_phase_z_mode": "concat",
                },
                contact_dim=2,
                direct_pose_cfg=_direct_pose_cfg(
                    use_phase_z=True,
                    phase_z_mode="replace_contacts",
                    feat_source="cond",
                    time_pe_dim=32,
                ),
                load_options=_load_options(),
                context="apply_direct_pose_ckpt_compat",
            )

    def test_apply_replace_contacts_rejects_time_pe_dim_mismatch(self) -> None:
        state_dict = {"direct_pose_head.0.weight": torch.zeros(4, 47)}
        model = _DummyEventMotionModel({"direct_pose_head.0.weight": torch.zeros(4, 43)})

        with self.assertRaisesRegex(
            RemovedCheckpointCompatError,
            "direct_pose_head\\.0\\.weight.*phase_z input-dimension.*no in-loader replacement",
        ):
            normalize_and_validate_direct_pose_ckpt_for_load(
                state_dict=state_dict,
                model=model,
                ckpt_posttrain_cfg={
                    "direct_pose_feat_source": "cond",
                    "direct_pose_time_pe_dim": 30,
                    "direct_pose_use_phase_z": True,
                    "direct_pose_phase_z_mode": "concat",
                },
                contact_dim=2,
                direct_pose_cfg=_direct_pose_cfg(
                    use_phase_z=True,
                    phase_z_mode="replace_contacts",
                    feat_source="cond",
                    time_pe_dim=32,
                ),
                load_options=_load_options(),
                context="apply_direct_pose_ckpt_compat",
            )

    def test_apply_rejects_direct_pose_leg_head_shape_mismatch(self) -> None:
        state_dict = {"direct_pose_leg_head.0.weight": torch.zeros(4, 6)}
        model = _DummyEventMotionModel({"direct_pose_leg_head.0.weight": torch.zeros(4, 8)})

        with self.assertRaisesRegex(
            RemovedCheckpointCompatError,
            "direct_pose_leg_head\\.\\*.*shape mismatch.*no in-loader replacement",
        ):
            normalize_and_validate_direct_pose_ckpt_for_load(
                state_dict=state_dict,
                model=model,
                ckpt_posttrain_cfg=None,
                contact_dim=1,
                direct_pose_cfg=_direct_pose_cfg(),
                load_options=_load_options(leg_enable=True),
                context="apply_direct_pose_ckpt_compat",
            )

    def test_apply_rejects_direct_pose_leg_bones_override_with_old_leg_tensors(self) -> None:
        state_dict = {"direct_pose_leg_head.0.weight": torch.zeros(4, 6)}
        model = _DummyEventMotionModel({"direct_pose_leg_head.0.weight": torch.zeros(4, 6)})

        with self.assertRaisesRegex(
            RemovedCheckpointCompatError,
            "direct_pose_leg_bones.*ckpt_bones=.*requested_bones=.*no in-loader replacement",
        ):
            normalize_and_validate_direct_pose_ckpt_for_load(
                state_dict=state_dict,
                model=model,
                ckpt_posttrain_cfg={"direct_pose_leg_bones": "thigh_l,thigh_r"},
                contact_dim=1,
                direct_pose_cfg=_direct_pose_cfg(),
                load_options=_load_options(leg_enable=True, leg_bones=("calf_l", "calf_r")),
                context="apply_direct_pose_ckpt_compat",
            )

    def test_apply_rejects_retired_highorder_direct_pose_tensors(self) -> None:
        state_dict = {"direct_pose_leg_head_shared.0.weight": torch.zeros(4, 6)}
        model = _DummyEventMotionModel({"direct_pose_leg_head_shared.0.weight": torch.zeros(4, 6)})

        with self.assertRaisesRegex(
            RemovedCheckpointCompatError,
            "direct_pose_leg_head_shared\\.\\*.*retired high-order.*no in-loader replacement",
        ):
            normalize_and_validate_direct_pose_ckpt_for_load(
                state_dict=state_dict,
                model=model,
                ckpt_posttrain_cfg=None,
                contact_dim=1,
                direct_pose_cfg=_direct_pose_cfg(),
                load_options=_load_options(),
                context="apply_direct_pose_ckpt_compat",
            )


if __name__ == "__main__":
    unittest.main()
