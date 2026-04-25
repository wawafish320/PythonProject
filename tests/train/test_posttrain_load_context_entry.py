from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import run_posttrain_nonleg_trunk_ablation as _run_posttrain_nonleg_trunk_ablation
from tools import run_posttrain_runtime_overlay_smoke as _run_posttrain_runtime_overlay_smoke
from train import posttrain as _posttrain


_LOAD_CONTEXT_HINT = "caller must set load_context to one of: resume|chain_hop"


class _CapturePayload(RuntimeError):
    pass


class PostTrainLoadContextEntryTest(unittest.TestCase):
    def test_cfg_from_payload_preserves_load_context(self) -> None:
        cfg = _posttrain._cfg_from_payload(
            {
                "ckpt_in": "models/demo_ckpt.pth",
                "train_direct_pose": True,
                "load_context": "chain_hop",
            }
        )

        self.assertEqual(cfg.load_context, "chain_hop")

    def test_main_missing_load_context_fails_before_dataset_build(self) -> None:
        with (
            mock.patch.object(sys, "argv", ["posttrain.py"]),
            mock.patch.object(
                _posttrain,
                "load_json",
                return_value={"ckpt_in": "models/demo_ckpt.pth", "train_direct_pose": True},
            ),
            mock.patch.object(_posttrain, "_build_dataset_and_loader") as build_dataset,
        ):
            with self.assertRaises(SystemExit) as ctx:
                _posttrain.main()

        self.assertIn(_LOAD_CONTEXT_HINT, str(ctx.exception))
        build_dataset.assert_not_called()

    def test_main_cli_load_context_override_unblocks_preflight(self) -> None:
        with (
            mock.patch.object(sys, "argv", ["posttrain.py", "--load_context", "resume"]),
            mock.patch.object(
                _posttrain,
                "load_json",
                return_value={"ckpt_in": "models/demo_ckpt.pth", "train_direct_pose": True},
            ),
            mock.patch.object(_posttrain, "_build_dataset_and_loader", side_effect=RuntimeError("stop-after-preflight")),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop-after-preflight"):
                _posttrain.main()

    def test_nonleg_trunk_runner_sets_chain_hop_context(self) -> None:
        captured: dict[str, object] = {}

        def _capture(payload: dict[str, object]) -> object:
            captured.update(payload)
            raise _CapturePayload("captured")

        args = SimpleNamespace(
            config="config/posttrain_70r.json",
            trunk_mode="full",
            out_dir="models/out",
            run_name="demo",
            epochs=1,
            steps_per_epoch=20,
            save_step_ckpts="0,1,5,20",
        )
        with (
            mock.patch.object(_run_posttrain_nonleg_trunk_ablation, "_parse_args", return_value=args),
            mock.patch.object(
                _run_posttrain_nonleg_trunk_ablation.posttrain,
                "load_json",
                return_value={"ckpt_in": "models/demo_ckpt.pth", "train_direct_pose": True},
            ),
            mock.patch.object(
                _run_posttrain_nonleg_trunk_ablation.posttrain,
                "_cfg_from_payload",
                side_effect=_capture,
            ),
        ):
            with self.assertRaises(_CapturePayload):
                _run_posttrain_nonleg_trunk_ablation.main()

        self.assertEqual(captured.get("load_context"), "chain_hop")

    def test_runtime_overlay_smoke_sets_resume_context(self) -> None:
        spec = _run_posttrain_runtime_overlay_smoke.FixtureSpec(
            name="demo",
            config_json=Path("config/posttrain_demo.json"),
            ckpt_in=Path("models/demo_ckpt.pth"),
            payload_source="config_json+overrides",
        )
        with mock.patch.object(
            _run_posttrain_runtime_overlay_smoke,
            "load_json",
            return_value={"train_direct_pose": True},
        ):
            payload = _run_posttrain_runtime_overlay_smoke._load_fixture_payload(spec)

        self.assertEqual(payload["load_context"], "resume")


if __name__ == "__main__":
    unittest.main()
