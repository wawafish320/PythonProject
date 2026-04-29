from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
import unittest

import torch

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
    torch.manual_seed(0)
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
        contact_plan_init_dropout=0.0,
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


def _build_cfg(*, train_mode: str = "direct") -> SimpleNamespace:
    return SimpleNamespace(
        train_direct_pose=bool(train_mode == "direct"),
        train_lambda_head=bool(train_mode == "lambda"),
        direct_pose_leg_train_only=False,
        direct_pose_leg_gate_train_only=False,
        direct_pose_nonleg_train_only=False,
        rollout_steps=10,
        rollout_cycles=2,
        rollout_include_boundary=True,
        rollout_random_offset=False,
        time_index_mode="cycle",
        detach_rollout_state=True,
        contact_meas_weight=0.25,
        lambda_reliability_mode="warmup",
    )


def _build_state(
    model: EventMotionModel,
    *,
    checkpoint_fingerprints: dict[str, str] | None = None,
    checkpoint_manifest_summary: dict[str, object] | None = None,
) -> _posttrain_build_shell.PostTrainModelBuildState:
    direct_pose_leg_cfg = SimpleNamespace(
        enable=bool(getattr(model, "direct_pose_leg_enable", False)),
        bones=tuple(getattr(model, "direct_pose_leg_bones", ()) or ()),
        mode=str(getattr(model, "direct_pose_leg_mode", "none") or "none"),
        side_routing=bool(getattr(model, "direct_pose_leg_side_routing", False)),
    )
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
    return _posttrain_build_shell.PostTrainModelBuildState(
        ckpt_posttrain_cfg={"source": "unit"},
        state_dict={},
        model_build_config=SimpleNamespace(),
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
        direct_pose_leg_cfg=direct_pose_leg_cfg,
        lambda_fusion_enable=bool(model.lambda_fusion_enable),
        lambda_fusion_mode=str(model.lambda_fusion_mode),
        lambda_fusion_hidden=int(model.lambda_fusion_hidden),
        lambda_fusion_dropout=float(model._lambda_fusion_dropout),
        lambda_fusion_logit_init=float(model._lambda_fusion_logit_init),
        lambda_fusion_use_rollout_step=bool(model.lambda_fusion_use_rollout_step),
        direct_pose_leg_gate_mode_model=str(model.direct_pose_leg_gate_mode),
        direct_pose_leg_gate_power_model=float(model.direct_pose_leg_gate_power),
        fingerprint_schema_version=1,
        checkpoint_fingerprints=checkpoint_fingerprints,
        checkpoint_manifest_summary=checkpoint_manifest_summary,
    )


def _render_report(
    summary: _posttrain_build_shell.FingerprintCompareSummary,
    *,
    manifest_summary_present: bool,
) -> str:
    stream = StringIO()
    with redirect_stdout(stream):
        _posttrain_build_shell._emit_posttrain_checkpoint_fingerprint_report(
            summary=summary,
            manifest_summary_present=manifest_summary_present,
        )
    return stream.getvalue()


class CheckpointFingerprintPhase4LoadReportTest(unittest.TestCase):
    def test_full_match_reports_ok(self) -> None:
        model = _build_feature_rich_model()
        cfg = _build_cfg(train_mode="direct")
        build_state = _build_state(
            model,
            checkpoint_manifest_summary={"build_trace": {"pipeline": "posttrain"}},
        )
        current = _posttrain_build_shell._build_posttrain_current_fingerprints(
            cfg=cfg,
            model=model,
            build_state=build_state,
        )
        build_state = replace(build_state, checkpoint_fingerprints=current)

        current_recomputed, summary = _posttrain_build_shell._compare_posttrain_checkpoint_fingerprints(
            cfg=cfg,
            model=model,
            build_state=build_state,
        )
        report = _render_report(summary, manifest_summary_present=True)

        self.assertEqual(current_recomputed, current)
        self.assertEqual(summary.overall_status, "pass")
        self.assertTrue(all(result.status == "match" for result in summary.results))
        self.assertIn("[OK] checkpoint fingerprint comparison summary.", report)
        self.assertIn("manifest_summary=present", report)

    def test_required_segment_mismatch_reports_fail_without_raise(self) -> None:
        model = _build_feature_rich_model()
        cfg = _build_cfg(train_mode="direct")
        build_state = _build_state(
            model,
            checkpoint_manifest_summary={"build_trace": {"pipeline": "posttrain"}},
        )
        current = _posttrain_build_shell._build_posttrain_current_fingerprints(
            cfg=cfg,
            model=model,
            build_state=build_state,
        )
        checkpoint = dict(current)
        checkpoint["module_graph_hash"] = "graph-old"
        build_state = replace(build_state, checkpoint_fingerprints=checkpoint)

        _, summary = _posttrain_build_shell._compare_posttrain_checkpoint_fingerprints(
            cfg=cfg,
            model=model,
            build_state=build_state,
        )
        report = _render_report(summary, manifest_summary_present=True)
        statuses = {result.segment: result.status for result in summary.results}

        self.assertEqual(summary.overall_status, "fail")
        self.assertEqual(statuses["module_graph_hash"], "mismatch")
        self.assertEqual(statuses["io_signature_hash"], "match")
        self.assertIn("module_graph_hash", report)
        self.assertIn("semantic module graph changed", report)

    def test_optional_train_policy_mismatch_is_warn_report_only(self) -> None:
        model = _build_feature_rich_model()
        cfg = _build_cfg(train_mode="lambda")
        build_state = _build_state(
            model,
            checkpoint_manifest_summary={"build_trace": {"pipeline": "posttrain"}},
        )
        current = _posttrain_build_shell._build_posttrain_current_fingerprints(
            cfg=cfg,
            model=model,
            build_state=build_state,
        )
        checkpoint = dict(current)
        checkpoint["train_policy_hash"] = "policy-old"
        build_state = replace(build_state, checkpoint_fingerprints=checkpoint)

        _, summary = _posttrain_build_shell._compare_posttrain_checkpoint_fingerprints(
            cfg=cfg,
            model=model,
            build_state=build_state,
        )
        report = _render_report(summary, manifest_summary_present=True)
        train_policy_result = next(result for result in summary.results if result.segment == "train_policy_hash")

        self.assertEqual(summary.overall_status, "warn")
        self.assertEqual(train_policy_result.status, "mismatch")
        self.assertEqual(train_policy_result.next_action, "optional segment drift observed; compare/report only.")
        self.assertIn("train_policy_hash", report)
        self.assertIn("compare/report only", report)

    def test_missing_fingerprint_block_reports_missing_required(self) -> None:
        model = _build_feature_rich_model()
        cfg = _build_cfg(train_mode="direct")
        build_state = _build_state(model, checkpoint_fingerprints=None, checkpoint_manifest_summary=None)

        _, summary = _posttrain_build_shell._compare_posttrain_checkpoint_fingerprints(
            cfg=cfg,
            model=model,
            build_state=build_state,
        )
        report = _render_report(summary, manifest_summary_present=False)

        self.assertEqual(summary.overall_status, "fail")
        self.assertEqual(len(summary.results), 1)
        self.assertEqual(summary.results[0].segment, "fingerprint_block")
        self.assertEqual(summary.results[0].status, "missing_required")
        self.assertIn("checkpoint missing required fingerprint metadata", report)
        self.assertIn("manifest_summary=missing", report)


if __name__ == "__main__":
    unittest.main()
