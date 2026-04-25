from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from train import posttrain, training_MPL
from train import posttrain_build_shell as _posttrain_build_shell
from train.history import AdaptiveHistoryModule
from train.models import EventMotionModel, MotionEncoder, PeriodHead


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


def _build_feature_rich_model() -> EventMotionModel:
    bone_names = ["thigh_l", "calf_l", "thigh_r", "calf_r", "arm_l", "arm_r", "spine"]
    num_joints = len(bone_names)
    model = EventMotionModel(
        in_state_dim=5 + num_joints * 6 + num_joints * 3,
        out_motion_dim=num_joints * 6,
        cond_dim=8,
        period_dim=4,
        hidden_dim=48,
        num_layers=3,
        num_heads=4,
        dropout=0.0,
        contact_dim=2,
        angvel_dim=num_joints * 3,
        pose_hist_dim=12,
        state_layout=_make_state_layout(num_joints),
        output_layout=_make_output_layout(num_joints),
        bone_names=bone_names,
        contact_plan_enable=True,
        contact_plan_hidden=16,
        contact_plan_inject="plan_z",
        contact_plan_time_pe_dim=8,
        contact_plan_init_mode="learnable+obs",
        contact_plan_init_hidden=12,
        use_event_clock=True,
        event_clock_hidden_dim=20,
        event_clock_gate_hidden_dim=12,
        direct_pose_enable=True,
        direct_pose_hidden=32,
        direct_pose_meas_mode="concat",
        direct_pose_feat_source="cond+hidden",
        direct_pose_time_pe_dim=8,
        direct_pose_use_phase_z=True,
        direct_pose_phase_z_mode="concat",
        direct_pose_split_enable=True,
        direct_pose_arm_split_enable=True,
        direct_pose_nonleg_proj_dim=10,
        direct_pose_leg_enable=True,
        direct_pose_leg_bones=("thigh_l", "calf_l", "thigh_r", "calf_r"),
        direct_pose_leg_mode="so3",
        direct_pose_leg_gate_mode="learned",
        direct_pose_leg_side_routing=True,
        direct_pose_leg_side_embed_dim=4,
        direct_pose_leg_side_sign_gate=True,
        direct_pose_arm_bones=("arm_l", "arm_r"),
        lambda_fusion_enable=True,
        lambda_fusion_mode="per_joint",
        lambda_fusion_hidden=12,
        lambda_fusion_use_rollout_step=True,
        so3_corr_hidden=12,
    )
    history_module = AdaptiveHistoryModule(
        pose_dim=6,
        hidden_dim=8,
        num_history_frames=2,
        cond_dim=8,
        num_heads=2,
    )
    model.enable_adaptive_history(history_module, pose_hist_len=2)
    model.frozen_encoder = MotionEncoder(
        input_dim=int(model.encoder_input_dim),
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    model.frozen_period_head = PeriodHead(hidden_dim=16, out_dim=int(model.period_dim))
    model.frozen_contact_head = torch.nn.Linear(16, int(model.contact_dim))
    return model


def _build_posttrain_artifacts(model: EventMotionModel) -> _posttrain_build_shell.PostTrainModelArtifacts:
    direct_pose_cfg = SimpleNamespace(
        enable=True,
        hidden=int(model.direct_pose_hidden),
        meas_mode=str(model.direct_pose_meas_mode),
        feat_source=str(model.direct_pose_feat_source),
        time_pe_dim=int(model.direct_pose_time_pe_dim),
        time_pe_base=float(model._direct_pose_time_pe_base),
        use_phase_z=bool(model.direct_pose_use_phase_z),
        phase_z_mode=str(model.direct_pose_phase_z_mode),
        split_enable=bool(model.direct_pose_split_enable),
        arm_split_enable=bool(model.direct_pose_arm_split_enable),
        arm_bones=getattr(model, "direct_pose_arm_bones", None),
        nonleg_proj_dim=int(model.direct_pose_nonleg_proj_dim),
        drop_ckpt_weights=False,
    )
    build_state = _posttrain_build_shell.PostTrainModelBuildState(
        ckpt_posttrain_cfg={"source": "unit"},
        state_dict={},
        width=int(model.hidden_dim),
        contact_dim=int(model.contact_dim),
        angvel_dim=int(model.angvel_dim),
        pose_hist_dim=int(model.pose_hist_dim),
        contact_plan_enable=bool(model.contact_plan_enable),
        contact_plan_hidden=int(model.contact_plan_hidden),
        contact_plan_inject=str(model.contact_plan_inject),
        contact_plan_time_pe_dim=int(model.contact_plan_time_pe_dim),
        contact_plan_init_mode=str(model.contact_plan_init_mode),
        contact_plan_init_hidden=int(model.contact_plan_init_hidden),
        contact_plan_init_dropout=float(model._contact_plan_init_dropout),
        use_event_clock=bool(model.use_event_clock),
        event_clock_hidden_dim=int(model.event_clock_hidden_dim),
        event_clock_gate_hidden_dim=int(model.event_clock_gate_hidden_dim),
        event_clock_max_delta=float(model.event_clock_max_delta),
        period_dim_init=int(model.period_dim),
        direct_pose_cfg=direct_pose_cfg,
        lambda_fusion_enable=bool(model.lambda_fusion_enable),
        lambda_fusion_mode=str(model.lambda_fusion_mode),
        lambda_fusion_hidden=int(model.lambda_fusion_hidden),
        lambda_fusion_dropout=float(model._lambda_fusion_dropout),
        lambda_fusion_logit_init=float(model._lambda_fusion_logit_init),
        lambda_fusion_use_rollout_step=bool(model.lambda_fusion_use_rollout_step),
        direct_pose_leg_gate_mode_model=str(model.direct_pose_leg_gate_mode),
        direct_pose_leg_gate_power_model=float(model.direct_pose_leg_gate_power),
    )
    return _posttrain_build_shell.PostTrainModelArtifacts(
        model=model,
        build_state=build_state,
        direct_pose_feat_source=str(model.direct_pose_feat_source),
        direct_pose_time_pe_dim=int(model.direct_pose_time_pe_dim),
        direct_pose_time_pe_base=float(model._direct_pose_time_pe_base),
        direct_pose_use_phase_z=bool(model.direct_pose_use_phase_z),
        direct_pose_phase_z_mode=str(model.direct_pose_phase_z_mode),
        direct_pose_split_enable=bool(model.direct_pose_split_enable),
        direct_pose_nonleg_proj_dim=int(model.direct_pose_nonleg_proj_dim),
        direct_pose_leg_gate_mode_model=str(model.direct_pose_leg_gate_mode),
        direct_pose_leg_gate_power_model=float(model.direct_pose_leg_gate_power),
    )


class CheckpointFingerprintPhase3SaveTest(unittest.TestCase):
    def test_basetrain_fit_checkpoint_payload_writes_fingerprint_metadata(self) -> None:
        model = _build_feature_rich_model()
        trainer_stub = SimpleNamespace(
            model=model,
            full_config={"run_name": "demo"},
            tf_mode="linear",
            tf_start_epoch=1,
            tf_end_epoch=4,
            tf_max=1.0,
            tf_min=0.1,
            ss_chunk_len=2,
            history_dropout_prob=0.05,
            history_dropout_prob_min=0.01,
            history_dropout_prob_max=0.10,
            enable_grad_connection_test=True,
            freerun_stage_schedule=[{"epochs": 1, "teacher_forcing": 0.5}],
        )

        payload = training_MPL.Trainer._fit_checkpoint_payload(trainer_stub)

        self.assertIn("fingerprint_schema_version", payload)
        self.assertIn("fingerprints", payload)
        self.assertIn("manifest_summary", payload)
        self.assertEqual(
            set(payload["fingerprints"].keys()),
            {
                "io_signature_hash",
                "module_graph_hash",
                "build_order_hash",
                "weights_hash",
                "train_policy_hash",
            },
        )
        self.assertEqual(payload["manifest_summary"]["build_trace"]["pipeline"], "basetrain")

    def test_posttrain_checkpoint_payload_writes_contract_and_fingerprint_metadata(self) -> None:
        model = _build_feature_rich_model()
        artifacts = _build_posttrain_artifacts(model)
        cfg = SimpleNamespace(
            out_dir=Path("/tmp/posttrain_phase3"),
            run_name="demo",
            depth=3,
            num_heads=4,
            dropout=0.0,
            rollout_steps=10,
            rollout_cycles=2,
            rollout_include_boundary=True,
            rollout_random_offset=False,
            time_index_mode="cycle",
            detach_rollout_state=True,
            contact_meas_weight=0.25,
            lambda_reliability_mode="warmup",
            direct_pose_nonleg_train_only=False,
        )
        trainable_slots = {"direct_pose_head": True, "lambda_fusion_head": False}

        with mock.patch("train.posttrain._cfg_to_jsonable", return_value={"run_name": "demo"}):
            payload = posttrain._build_posttrain_checkpoint_payload(
                cfg=cfg,
                artifacts=artifacts,
                train_mode="direct",
                trainable_slots=trainable_slots,
            )

        self.assertIn("checkpoint_contract", payload)
        self.assertIn("build_cfg", payload)
        self.assertIn("fingerprints", payload)
        self.assertIn("manifest_summary", payload)
        self.assertEqual(
            payload["checkpoint_contract"]["name"],
            posttrain.POSTTRAIN_CHECKPOINT_CONTRACT_NAME,
        )
        self.assertEqual(
            payload["checkpoint_contract"]["version"],
            int(posttrain.POSTTRAIN_CHECKPOINT_CONTRACT_VERSION),
        )
        self.assertEqual(payload["manifest_summary"]["build_trace"]["pipeline"], "posttrain")
        configure_step = next(
            step
            for step in payload["manifest_summary"]["build_trace"]["steps"]
            if step["step_id"] == "posttrain.configure_trainable_slots"
        )
        self.assertEqual(configure_step["normalized_config"]["train_mode"], "direct")
        self.assertEqual(
            configure_step["normalized_config"]["trainable_slots"],
            {"direct_pose_head": True, "lambda_fusion_head": False},
        )

    def test_save_posttrain_outputs_persists_fingerprint_block(self) -> None:
        model = _build_feature_rich_model()
        artifacts = _build_posttrain_artifacts(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimpleNamespace(
                out_dir=Path(tmpdir),
                run_name="save",
                depth=3,
                num_heads=4,
                dropout=0.0,
                rollout_steps=5,
                rollout_cycles=1,
                rollout_include_boundary=False,
                rollout_random_offset=False,
                time_index_mode="global",
                detach_rollout_state=False,
                contact_meas_weight=0.0,
                lambda_reliability_mode="none",
                direct_pose_nonleg_train_only=False,
            )
            with mock.patch("train.posttrain._cfg_to_jsonable", return_value={"run_name": "save"}):
                ckpt_path = posttrain._save_posttrain_outputs(
                    cfg=cfg,
                    artifacts=artifacts,
                    train_mode="lambda",
                    trainable_slots={"lambda_fusion_head": True},
                    log_rows=[{"step": 1.0}],
                )
            payload = torch.load(ckpt_path, map_location="cpu")

        self.assertIn("fingerprints", payload)
        self.assertIn("manifest_summary", payload)
        self.assertTrue(ckpt_path.name.startswith("ckpt_last_"))


if __name__ == "__main__":
    unittest.main()
