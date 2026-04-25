from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import unittest

import torch
from torch import nn

from train.checkpoint import fingerprint as fp
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
    model.frozen_contact_head = nn.Linear(16, int(model.contact_dim))
    return model


class CheckpointFingerprintPhase2Test(unittest.TestCase):
    def test_module_graph_manifest_covers_feature_rich_slots_and_hash_is_order_insensitive(self) -> None:
        model = _build_feature_rich_model()

        manifest = fp.build_event_motion_model_module_graph_manifest(model)
        hash_a = fp.compute_module_graph_hash(manifest)
        hash_b = fp.compute_module_graph_hash(
            fp.ModuleGraphManifest(components=tuple(reversed(manifest.components)))
        )

        self.assertEqual(hash_a, hash_b)

        enabled_slots = {
            component.component_slot
            for component in manifest.components
            if component.enabled
        }
        expected_enabled = {
            "shared_encoder",
            "residual_proj",
            "pasa_attention_block",
            "motion_head",
            "period_encoder",
            "contact_plan_cell",
            "contact_plan_init_z",
            "contact_plan_init_head",
            "contact_plan_head",
            "contact_plan_time_head",
            "event_clock_gate",
            "event_clock_corrector",
            "direct_pose_head",
            "direct_pose_leg_terminal",
            "direct_pose_out_arm",
            "direct_pose_out_else",
            "direct_pose_arm_proj",
            "direct_pose_else_proj",
            "direct_pose_leg_head",
            "direct_pose_leg_gate_head",
            "direct_pose_leg_head_shared",
            "direct_pose_leg_gate_head_shared",
            "direct_pose_leg_side_embed",
            "direct_pose_leg_side_sign_gate_head",
            "lambda_fusion_head",
            "so3_delta_corrector",
            "so3_corr_gate_logit",
            "adaptive_history_module",
            "frozen_encoder",
            "frozen_period_head",
            "frozen_contact_head",
        }
        self.assertTrue(expected_enabled.issubset(enabled_slots))

    def test_module_graph_hash_respects_order_sensitive_consumes(self) -> None:
        direct_a = fp.ComponentManifest(
            component_slot="direct_pose_head",
            component_kind=fp.COMPONENT_SLOT_KIND_MAP["direct_pose_head"],
            enabled=True,
            consumes=("cond", "h_final", "contacts_plan", "contacts_meas"),
            produces=("out_direct",),
            order_sensitive_consumes=True,
        )
        direct_b = fp.ComponentManifest(
            component_slot="direct_pose_head",
            component_kind=fp.COMPONENT_SLOT_KIND_MAP["direct_pose_head"],
            enabled=True,
            consumes=("h_final", "cond", "contacts_plan", "contacts_meas"),
            produces=("out_direct",),
            order_sensitive_consumes=True,
        )
        shared_a = fp.ComponentManifest(
            component_slot="shared_encoder",
            component_kind=fp.COMPONENT_SLOT_KIND_MAP["shared_encoder"],
            enabled=True,
            consumes=("state", "cond"),
            produces=("h",),
        )
        shared_b = fp.ComponentManifest(
            component_slot="shared_encoder",
            component_kind=fp.COMPONENT_SLOT_KIND_MAP["shared_encoder"],
            enabled=True,
            consumes=("cond", "state"),
            produces=("h",),
        )

        direct_hash_a = fp.compute_module_graph_hash(fp.ModuleGraphManifest(components=(direct_a,)))
        direct_hash_b = fp.compute_module_graph_hash(fp.ModuleGraphManifest(components=(direct_b,)))
        shared_hash_a = fp.compute_module_graph_hash(fp.ModuleGraphManifest(components=(shared_a,)))
        shared_hash_b = fp.compute_module_graph_hash(fp.ModuleGraphManifest(components=(shared_b,)))

        self.assertNotEqual(direct_hash_a, direct_hash_b)
        self.assertEqual(shared_hash_a, shared_hash_b)

    def test_build_order_hash_treats_substep_sequence_as_semantic(self) -> None:
        manifest = fp.build_basetrain_build_trace_manifest()
        attach_step = next(step for step in manifest.steps if step.step_id == "basetrain.attach_entry_runtime")
        reversed_attach = replace(attach_step, substeps=tuple(reversed(attach_step.substeps)))
        mutated_steps = tuple(
            reversed_attach if step.step_id == attach_step.step_id else step
            for step in manifest.steps
        )

        hash_a = fp.compute_build_order_hash(manifest)
        hash_b = fp.compute_build_order_hash(
            fp.BuildTraceManifest(pipeline=manifest.pipeline, steps=mutated_steps)
        )

        self.assertNotEqual(hash_a, hash_b)

    def test_weights_hash_is_sorted_by_key_and_changes_with_tensor_content(self) -> None:
        state_a = OrderedDict(
            [
                ("b.weight", torch.ones(2, 2, dtype=torch.float32)),
                ("a.bias", torch.zeros(2, dtype=torch.float32)),
            ]
        )
        state_b = OrderedDict(
            [
                ("a.bias", torch.zeros(2, dtype=torch.float32)),
                ("b.weight", torch.ones(2, 2, dtype=torch.float32)),
            ]
        )
        state_c = OrderedDict(
            [
                ("a.bias", torch.zeros(2, dtype=torch.float32)),
                ("b.weight", torch.full((2, 2), 2.0, dtype=torch.float32)),
            ]
        )

        self.assertEqual(fp.compute_weights_hash(state_a), fp.compute_weights_hash(state_b))
        self.assertNotEqual(fp.compute_weights_hash(state_a), fp.compute_weights_hash(state_c))

    def test_io_signature_hash_sorts_outputs_but_preserves_input_order(self) -> None:
        model = _build_feature_rich_model()
        manifest = fp.build_event_motion_model_io_signature_manifest(model)

        hash_a = fp.compute_io_signature_hash(manifest)
        hash_b = fp.compute_io_signature_hash(
            fp.IOSignatureManifest(inputs=manifest.inputs, outputs=tuple(reversed(manifest.outputs)))
        )
        hash_c = fp.compute_io_signature_hash(
            fp.IOSignatureManifest(
                inputs=(manifest.inputs[1], manifest.inputs[0], *manifest.inputs[2:]),
                outputs=manifest.outputs,
            )
        )

        self.assertEqual(hash_a, hash_b)
        self.assertNotEqual(hash_a, hash_c)

    def test_compare_fingerprints_reports_required_optional_and_mismatch_states(self) -> None:
        current = {
            "io_signature_hash": "io-current",
            "module_graph_hash": "graph-current",
            "build_order_hash": "build-current",
            "weights_hash": "weights-current",
            "train_policy_hash": "policy-current",
        }
        checkpoint = {
            "io_signature_hash": "io-current",
            "module_graph_hash": "graph-old",
            "weights_hash": "weights-old",
        }

        summary = fp.compare_fingerprints(
            checkpoint,
            current,
            short_diff_hints={"module_graph_hash": "direct_pose_head consumes changed"},
        )
        text = fp.format_fingerprint_compare_summary(summary)
        statuses = {result.segment: result.status for result in summary.results}

        self.assertEqual(summary.overall_status, "fail")
        self.assertEqual(statuses["io_signature_hash"], "match")
        self.assertEqual(statuses["module_graph_hash"], "mismatch")
        self.assertEqual(statuses["build_order_hash"], "missing_required")
        self.assertEqual(statuses["weights_hash"], "mismatch")
        self.assertEqual(statuses["train_policy_hash"], "missing_optional")
        self.assertIn("module_graph_hash", text)
        self.assertIn("direct_pose_head consumes changed", text)
        self.assertIn("build_order_hash", text)

    def test_compare_fingerprints_keeps_missing_optional_as_no_check_not_warn(self) -> None:
        current = {
            "io_signature_hash": "io-current",
            "module_graph_hash": "graph-current",
            "build_order_hash": "build-current",
            "weights_hash": "weights-current",
            "train_policy_hash": "policy-current",
        }
        checkpoint = {
            "io_signature_hash": "io-current",
            "module_graph_hash": "graph-current",
            "build_order_hash": "build-current",
            "weights_hash": "weights-current",
        }

        summary = fp.compare_fingerprints(checkpoint, current)
        statuses = {result.segment: result.status for result in summary.results}

        self.assertEqual(summary.overall_status, "pass")
        self.assertEqual(statuses["train_policy_hash"], "missing_optional")


if __name__ == "__main__":
    unittest.main()
