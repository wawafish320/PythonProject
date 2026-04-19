from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from train import posttrain


class PosttrainBuildModelAndTrainerTest(unittest.TestCase):
    def test_sync_posttrain_bone_names_prefers_loss_setter(self) -> None:
        trainer = SimpleNamespace()
        loss_fn = SimpleNamespace()
        loss_fn.set_bone_names = mock.Mock()
        ds = SimpleNamespace(bone_names=["thigh_l", "foot_l"])

        posttrain._sync_posttrain_bone_names(trainer=trainer, loss_fn=loss_fn, ds=ds)

        self.assertEqual(trainer.bone_names, ["thigh_l", "foot_l"])
        loss_fn.set_bone_names.assert_called_once_with(["thigh_l", "foot_l"])

    def test_attach_posttrain_trainer_runtime_orders_shared_loss_and_local_overlay(self) -> None:
        cfg = SimpleNamespace(
            bundle_json=SimpleNamespace(
                expanduser=lambda: SimpleNamespace(resolve=lambda: "/tmp/bundle.json")
            ),
            out_dir="/tmp/out",
            run_name="run",
        )
        ds = SimpleNamespace()
        trainer = SimpleNamespace(loss_fn=SimpleNamespace(), yaw_forward_axis=2)
        expected_norm_spec = {"std": 1.0}
        dataset_artifacts = SimpleNamespace()
        shared_runtime = SimpleNamespace()
        local_overlay = SimpleNamespace()
        call_order: list[str] = []

        def _build_dataset_runtime(trainer, ds, *, bundle_path, norm_spec):
            call_order.append("dataset")
            self.assertEqual(bundle_path, "/tmp/bundle.json")
            self.assertIs(norm_spec, expected_norm_spec)
            return dataset_artifacts

        def _resolve_shared_runtime(**kwargs):
            call_order.append("resolve_shared")
            self.assertIs(kwargs["dataset_artifacts"], dataset_artifacts)
            self.assertEqual(kwargs["trainer_default_yaw_forward_axis"], 2)
            self.assertEqual(kwargs["bundle_json_path"], "/tmp/bundle.json")
            self.assertEqual(kwargs["out_dir"], "/tmp/out")
            self.assertEqual(kwargs["current_run_name"], "run")
            return shared_runtime

        def _apply_shared_runtime(trainer, runtime):
            call_order.append("apply_shared")
            self.assertIs(trainer, trainer_obj)
            self.assertIs(runtime, shared_runtime)

        def _apply_loss_runtime(loss_fn, trainer):
            call_order.append("loss_runtime")
            self.assertIs(loss_fn, trainer_obj.loss_fn)
            self.assertIs(trainer, trainer_obj)

        def _resolve_local_overlay(cfg):
            call_order.append("resolve_local")
            self.assertIs(cfg, cfg_obj)
            return local_overlay

        def _apply_local_overlay(trainer, overlay):
            call_order.append("apply_local")
            self.assertIs(trainer, trainer_obj)
            self.assertIs(overlay, local_overlay)

        cfg_obj, trainer_obj = cfg, trainer
        with (
            mock.patch("train.posttrain.build_and_attach_dataset_runtime", side_effect=_build_dataset_runtime),
            mock.patch("train.posttrain.resolve_shared_trainer_runtime", side_effect=_resolve_shared_runtime),
            mock.patch("train.posttrain.apply_shared_trainer_runtime", side_effect=_apply_shared_runtime),
            mock.patch("train.posttrain.apply_loss_runtime_from_trainer", side_effect=_apply_loss_runtime),
            mock.patch("train.posttrain._resolve_posttrain_local_runtime_overlay", side_effect=_resolve_local_overlay),
            mock.patch("train.posttrain._apply_posttrain_local_runtime_overlay", side_effect=_apply_local_overlay),
            mock.patch("train.posttrain._cfg_to_jsonable", return_value={"run_name": "run"}),
        ):
            posttrain._attach_posttrain_trainer_runtime(
                cfg=cfg,
                ds=ds,
                trainer=trainer,
                norm_spec=expected_norm_spec,
            )

        self.assertEqual(
            call_order,
            ["dataset", "resolve_shared", "apply_shared", "loss_runtime", "resolve_local", "apply_local"],
        )

    def test_build_model_and_trainer_orchestrates_shell_then_runtime_attach(self) -> None:
        cfg = SimpleNamespace()
        ds = SimpleNamespace()
        model = SimpleNamespace()
        norm_spec = {"mu": 0.0}
        trainer = SimpleNamespace(loss_fn=SimpleNamespace())
        call_order: list[str] = []

        def _build_stub(*, cfg, ds, model):
            call_order.append("build")
            self.assertIs(cfg, cfg_obj)
            self.assertIs(ds, ds_obj)
            self.assertIs(model, model_obj)
            return trainer

        def _attach_stub(*, cfg, ds, trainer, norm_spec):
            call_order.append("attach")
            self.assertIs(cfg, cfg_obj)
            self.assertIs(ds, ds_obj)
            self.assertIs(trainer, trainer_obj)
            self.assertIs(norm_spec, norm_spec_obj)

        cfg_obj, ds_obj, model_obj, trainer_obj, norm_spec_obj = cfg, ds, model, trainer, norm_spec
        with (
            mock.patch("train.posttrain._build_posttrain_loss_and_trainer", side_effect=_build_stub),
            mock.patch("train.posttrain._attach_posttrain_trainer_runtime", side_effect=_attach_stub),
        ):
            out = posttrain._build_model_and_trainer(
                cfg=cfg,
                ds=ds,
                model=model,
                norm_spec=norm_spec,
            )

        self.assertIs(out, trainer)
        self.assertEqual(call_order, ["build", "attach"])


if __name__ == "__main__":
    unittest.main()
