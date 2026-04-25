from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest import mock

from train import posttrain_build_shell as _posttrain_build_shell

from tests.train.test_checkpoint_fingerprint_phase4_load_report import (
    _build_cfg,
    _build_feature_rich_model,
    _build_state,
)

_LOAD_CONTEXT_HINT = "caller must set load_context to one of: resume|chain_hop"


def _build_compare_summary(
    *,
    train_mode: str = "direct",
    mismatch_segment: str | None = None,
    missing_fingerprint_block: bool = False,
):
    model = _build_feature_rich_model()
    cfg = _build_cfg(train_mode=train_mode)
    build_state = _build_state(
        model,
        checkpoint_manifest_summary=None if missing_fingerprint_block else {"build_trace": {"pipeline": "posttrain"}},
    )
    current = _posttrain_build_shell._build_posttrain_current_fingerprints(
        cfg=cfg,
        model=model,
        build_state=build_state,
    )
    checkpoint_fingerprints = None if missing_fingerprint_block else dict(current)
    if checkpoint_fingerprints is not None and mismatch_segment is not None:
        checkpoint_fingerprints[str(mismatch_segment)] = f"{mismatch_segment}-old"
    build_state = replace(build_state, checkpoint_fingerprints=checkpoint_fingerprints)
    _, summary = _posttrain_build_shell._compare_posttrain_checkpoint_fingerprints(
        cfg=cfg,
        model=model,
        build_state=build_state,
    )
    return summary


class CheckpointFingerprintPhase5EnforceTest(unittest.TestCase):
    def test_resume_full_match_does_not_raise(self) -> None:
        summary = _build_compare_summary(train_mode="direct")
        _posttrain_build_shell._enforce_posttrain_checkpoint_fingerprint(summary, load_context="resume")

    def test_resume_required_mismatches_raise(self) -> None:
        for mismatch_segment in ("io_signature_hash", "module_graph_hash", "build_order_hash"):
            with self.subTest(load_context="resume", mismatch_segment=mismatch_segment):
                summary = _build_compare_summary(train_mode="direct", mismatch_segment=mismatch_segment)
                with self.assertRaises(SystemExit) as ctx:
                    _posttrain_build_shell._enforce_posttrain_checkpoint_fingerprint(
                        summary,
                        load_context="resume",
                    )
                self.assertIn(mismatch_segment, str(ctx.exception))

    def test_chain_hop_required_mismatches_do_not_raise(self) -> None:
        for mismatch_segment in ("io_signature_hash", "module_graph_hash", "build_order_hash"):
            with self.subTest(load_context="chain_hop", mismatch_segment=mismatch_segment):
                summary = _build_compare_summary(train_mode="direct", mismatch_segment=mismatch_segment)
                _posttrain_build_shell._enforce_posttrain_checkpoint_fingerprint(
                    summary,
                    load_context="chain_hop",
                )

    def test_missing_fingerprint_block_raises_for_resume_and_chain_hop(self) -> None:
        summary = _build_compare_summary(train_mode="direct", missing_fingerprint_block=True)
        for load_context in ("resume", "chain_hop"):
            with self.subTest(load_context=load_context):
                with self.assertRaises(SystemExit) as ctx:
                    _posttrain_build_shell._enforce_posttrain_checkpoint_fingerprint(
                        summary,
                        load_context=load_context,
                    )
                self.assertIn("fingerprint_block", str(ctx.exception))

    def test_weights_mismatch_is_warning_only(self) -> None:
        summary = _build_compare_summary(train_mode="direct", mismatch_segment="weights_hash")
        _posttrain_build_shell._enforce_posttrain_checkpoint_fingerprint(summary, load_context="resume")

    def test_train_policy_mismatch_is_report_only(self) -> None:
        summary = _build_compare_summary(train_mode="lambda", mismatch_segment="train_policy_hash")
        _posttrain_build_shell._enforce_posttrain_checkpoint_fingerprint(summary, load_context="resume")

    def test_report_includes_load_context_policy_without_phase5_banner(self) -> None:
        summary = _build_compare_summary(train_mode="direct", mismatch_segment="io_signature_hash")
        expected_policies = {
            "resume": "policy=resume-strict",
            "chain_hop": "policy=chain_hop-waiver",
        }
        for load_context, policy_fragment in expected_policies.items():
            with self.subTest(load_context=load_context):
                stream = StringIO()
                with redirect_stdout(stream):
                    _posttrain_build_shell._emit_posttrain_checkpoint_fingerprint_report(
                        summary=summary,
                        manifest_summary_present=True,
                        load_context=load_context,
                    )
                report = stream.getvalue()
                self.assertIn(f"load_context={load_context}", report)
                self.assertIn(policy_fragment, report)
                self.assertNotIn("Phase 5 enforce active", report)

    def test_missing_load_context_raises_with_hint(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _posttrain_build_shell._resolve_posttrain_load_context(SimpleNamespace())
        self.assertIn(_LOAD_CONTEXT_HINT, str(ctx.exception))

    def test_none_load_context_raises_with_hint(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _posttrain_build_shell._resolve_posttrain_load_context(SimpleNamespace(load_context=None))
        self.assertIn(_LOAD_CONTEXT_HINT, str(ctx.exception))

    def test_invalid_load_context_raises_with_hint(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _posttrain_build_shell._resolve_posttrain_load_context(SimpleNamespace(load_context="stage2"))
        self.assertIn(_LOAD_CONTEXT_HINT, str(ctx.exception))

    def test_build_from_ckpt_compares_and_enforces_before_load(self) -> None:
        call_order: list[str] = []
        direct_pose_cfg = SimpleNamespace(
            feat_source="cond+hidden",
            time_pe_dim=8,
            time_pe_base=10000.0,
            use_phase_z=True,
            phase_z_mode="concat",
            split_enable=True,
            nonleg_proj_dim=10,
        )
        build_state = SimpleNamespace(
            direct_pose_cfg=direct_pose_cfg,
            checkpoint_manifest_summary={"build_trace": {"pipeline": "posttrain"}},
            direct_pose_leg_gate_mode_model="learned",
            direct_pose_leg_gate_power_model=1.0,
        )
        summary = _build_compare_summary(train_mode="direct")
        fake_model = object()

        with (
            mock.patch.object(
                _posttrain_build_shell,
                "_resolve_posttrain_load_context",
                side_effect=lambda cfg: call_order.append("load_context") or "resume",
            ),
            mock.patch.object(
                _posttrain_build_shell,
                "_resolve_posttrain_model_build_state",
                side_effect=lambda **kwargs: call_order.append("resolve") or build_state,
            ),
            mock.patch.object(
                _posttrain_build_shell,
                "_instantiate_posttrain_model",
                side_effect=lambda **kwargs: call_order.append("instantiate") or fake_model,
            ),
            mock.patch.object(
                _posttrain_build_shell,
                "_compare_posttrain_checkpoint_fingerprints",
                side_effect=lambda **kwargs: call_order.append("compare") or ({"io_signature_hash": "io"}, summary),
            ),
            mock.patch.object(
                _posttrain_build_shell,
                "_emit_posttrain_checkpoint_fingerprint_report",
                side_effect=lambda **kwargs: call_order.append("emit"),
            ),
            mock.patch.object(
                _posttrain_build_shell,
                "_enforce_posttrain_checkpoint_fingerprint",
                side_effect=lambda *args, **kwargs: call_order.append("enforce"),
            ),
            mock.patch.object(
                _posttrain_build_shell,
                "_load_posttrain_checkpoint_into_model",
                side_effect=lambda **kwargs: call_order.append("load"),
            ),
        ):
            artifacts = _posttrain_build_shell._build_posttrain_model_from_ckpt(
                cfg=SimpleNamespace(),
                ds=mock.sentinel.ds,
                device=mock.sentinel.device,
            )

        self.assertIs(artifacts.model, fake_model)
        self.assertLess(call_order.index("compare"), call_order.index("load"))
        self.assertLess(call_order.index("enforce"), call_order.index("load"))
        self.assertEqual(call_order[:6], ["load_context", "resolve", "instantiate", "compare", "emit", "enforce"])
        self.assertEqual(call_order[6], "load")


if __name__ == "__main__":
    unittest.main()
