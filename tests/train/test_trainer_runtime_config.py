from __future__ import annotations

from types import SimpleNamespace
import unittest

from train.configuration import model_build as mb


class TrainerRuntimeConfigTest(unittest.TestCase):
    def test_train_defaults_and_trainer_kwargs_are_centralized(self) -> None:
        model_build_config = SimpleNamespace(hidden_dim=768)

        cfg = mb.resolve_train_trainer_runtime_config(
            args=SimpleNamespace(),
            model_build_config=model_build_config,
            pin_memory=True,
        )

        self.assertEqual(cfg.lr, mb.DEFAULT_TRAIN_TRAINER_LR)
        self.assertEqual(cfg.grad_clip, mb.DEFAULT_TRAIN_TRAINER_GRAD_CLIP)
        self.assertEqual(cfg.weight_decay, mb.DEFAULT_TRAIN_TRAINER_WEIGHT_DECAY)
        self.assertEqual(cfg.use_amp, mb.DEFAULT_TRAIN_TRAINER_USE_AMP)
        self.assertEqual(cfg.accum_steps, mb.DEFAULT_TRAIN_TRAINER_ACCUM_STEPS)
        self.assertTrue(cfg.pin_memory)
        self.assertEqual(cfg.history_adaptive_hidden, mb.DEFAULT_TRAIN_HISTORY_ADAPTIVE_HIDDEN)
        self.assertEqual(cfg.history_adaptive_heads, mb.DEFAULT_TRAIN_HISTORY_ADAPTIVE_HEADS)
        self.assertEqual(cfg.history_dropout_prob, mb.DEFAULT_TRAIN_HISTORY_DROPOUT_PROB)
        self.assertEqual(cfg.diag_topk, mb.DEFAULT_TRAIN_DIAG_TOPK)
        self.assertEqual(cfg.diag_thr, mb.DEFAULT_TRAIN_DIAG_THR)
        self.assertEqual(cfg.direct_pose_grad_monitor_enable, mb.DEFAULT_DIRECT_POSE_GRAD_MONITOR_ENABLE)
        self.assertEqual(cfg.direct_pose_grad_ratio_gate, mb.DEFAULT_DIRECT_POSE_GRAD_RATIO_GATE)
        self.assertEqual(
            cfg.contacts_pretrain.dropout_injection_mode,
            mb.DEFAULT_TRAINBASE_CONTACTS_PRETRAIN_DROPOUT_INJECTION_MODE,
        )
        self.assertEqual(
            cfg.contacts_pretrain.dropout_prob,
            mb.DEFAULT_TRAINBASE_CONTACTS_PRETRAIN_DROPOUT_PROB,
        )

        args_obj = SimpleNamespace(run_name="unit")
        self.assertEqual(
            cfg.to_trainer_kwargs(args=args_obj),
            {
                "lr": mb.DEFAULT_TRAIN_TRAINER_LR,
                "grad_clip": mb.DEFAULT_TRAIN_TRAINER_GRAD_CLIP,
                "weight_decay": mb.DEFAULT_TRAIN_TRAINER_WEIGHT_DECAY,
                "use_amp": mb.DEFAULT_TRAIN_TRAINER_USE_AMP,
                "accum_steps": mb.DEFAULT_TRAIN_TRAINER_ACCUM_STEPS,
                "pin_memory": True,
                "args": args_obj,
            },
        )

    def test_train_explicit_zero_values_are_not_replaced_by_defaults(self) -> None:
        cfg = mb.resolve_train_trainer_runtime_config(
            args=SimpleNamespace(
                lr=0.0,
                grad_clip=0.0,
                weight_decay=0.0,
                amp=False,
                accum_steps=2,
                history_adaptive_hidden=0,
                history_dropout_prob=0.0,
                diag_thr=0.0,
                direct_pose_grad_ratio_gate=0.0,
                trainbase_contacts_pretrain_dropout_injection_mode="encoder_input",
                trainbase_contacts_pretrain_dropout_prob=0.35,
            ),
            model_build_config=SimpleNamespace(hidden_dim=640),
            pin_memory=False,
        )

        self.assertEqual(cfg.lr, 0.0)
        self.assertEqual(cfg.grad_clip, 0.0)
        self.assertEqual(cfg.weight_decay, 0.0)
        self.assertFalse(cfg.use_amp)
        self.assertEqual(cfg.accum_steps, 2)
        self.assertFalse(cfg.pin_memory)
        self.assertEqual(cfg.history_adaptive_hidden, 640)
        self.assertEqual(cfg.history_dropout_prob, 0.0)
        self.assertEqual(cfg.diag_thr, 0.0)
        self.assertEqual(cfg.direct_pose_grad_ratio_gate, 0.0)
        self.assertEqual(cfg.contacts_pretrain.dropout_injection_mode, "encoder_input")
        self.assertEqual(cfg.contacts_pretrain.dropout_prob, 0.35)

        self.assertEqual(
            cfg.to_adaptive_history_kwargs(),
            {
                "history_hidden_dim": 640,
                "max_history_frames": mb.DEFAULT_TRAIN_HISTORY_ADAPTIVE_MAX_FRAMES,
                "history_heads": mb.DEFAULT_TRAIN_HISTORY_ADAPTIVE_HEADS,
                "train_variable_history": mb.DEFAULT_TRAIN_HISTORY_ADAPTIVE_TRAIN_VARIABLE,
                "history_dropout_prob": 0.0,
                "use_trend_features": mb.DEFAULT_TRAIN_HISTORY_USE_TREND_FEATURES,
            },
        )

    def test_posttrain_defaults_and_trainer_kwargs_are_centralized(self) -> None:
        cfg = mb.resolve_posttrain_trainer_runtime_config(
            cfg=SimpleNamespace(),
            model_build_config=SimpleNamespace(),
        )

        self.assertEqual(cfg.lr, mb.DEFAULT_POSTTRAIN_TRAINER_LR)
        self.assertEqual(cfg.grad_clip, mb.DEFAULT_POSTTRAIN_TRAINER_GRAD_CLIP)
        self.assertEqual(cfg.weight_decay, mb.DEFAULT_POSTTRAIN_TRAINER_WEIGHT_DECAY)
        self.assertEqual(cfg.use_amp, mb.DEFAULT_POSTTRAIN_TRAINER_USE_AMP)
        self.assertEqual(cfg.accum_steps, mb.DEFAULT_POSTTRAIN_TRAINER_ACCUM_STEPS)
        self.assertEqual(cfg.pin_memory, mb.DEFAULT_POSTTRAIN_TRAINER_PIN_MEMORY)
        self.assertEqual(cfg.contacts_pretrain.clamp, mb.DEFAULT_POSTTRAIN_CONTACTS_PRETRAIN_CLAMP)
        self.assertEqual(
            cfg.contacts_pretrain.dropout_injection_mode,
            mb.DEFAULT_POSTTRAIN_CONTACTS_PRETRAIN_DROPOUT_INJECTION_MODE,
        )
        self.assertEqual(
            cfg.contacts_pretrain.dropout_prob,
            mb.DEFAULT_POSTTRAIN_CONTACTS_PRETRAIN_DROPOUT_PROB,
        )
        self.assertEqual(cfg.direct_pose_grad_monitor_enable, mb.DEFAULT_DIRECT_POSE_GRAD_MONITOR_ENABLE)
        self.assertEqual(cfg.direct_pose_grad_ratio_gate, mb.DEFAULT_DIRECT_POSE_GRAD_RATIO_GATE)

        self.assertEqual(
            cfg.to_trainer_kwargs(),
            {
                "lr": mb.DEFAULT_POSTTRAIN_TRAINER_LR,
                "grad_clip": mb.DEFAULT_POSTTRAIN_TRAINER_GRAD_CLIP,
                "weight_decay": mb.DEFAULT_POSTTRAIN_TRAINER_WEIGHT_DECAY,
                "use_amp": mb.DEFAULT_POSTTRAIN_TRAINER_USE_AMP,
                "accum_steps": mb.DEFAULT_POSTTRAIN_TRAINER_ACCUM_STEPS,
                "pin_memory": mb.DEFAULT_POSTTRAIN_TRAINER_PIN_MEMORY,
            },
        )

    def test_posttrain_explicit_zero_values_are_not_replaced_by_defaults(self) -> None:
        cfg = mb.resolve_posttrain_trainer_runtime_config(
            cfg=SimpleNamespace(
                lr=0.0,
                weight_decay=0.0,
                posttrain_contacts_pretrain_clamp=0.0,
                posttrain_contacts_pretrain_dropout_injection_mode="hidden",
                posttrain_contacts_pretrain_dropout_prob=0.2,
                direct_pose_grad_ratio_gate=0.0,
            ),
            model_build_config=SimpleNamespace(),
        )

        self.assertEqual(cfg.lr, 0.0)
        self.assertEqual(cfg.weight_decay, 0.0)
        self.assertEqual(cfg.contacts_pretrain.clamp, 0.0)
        self.assertEqual(cfg.contacts_pretrain.dropout_injection_mode, "hidden")
        self.assertEqual(cfg.contacts_pretrain.dropout_prob, 0.2)
        self.assertEqual(cfg.direct_pose_grad_ratio_gate, 0.0)
        self.assertEqual(
            cfg.to_trainer_kwargs(),
            {
                "lr": 0.0,
                "grad_clip": mb.DEFAULT_POSTTRAIN_TRAINER_GRAD_CLIP,
                "weight_decay": 0.0,
                "use_amp": mb.DEFAULT_POSTTRAIN_TRAINER_USE_AMP,
                "accum_steps": mb.DEFAULT_POSTTRAIN_TRAINER_ACCUM_STEPS,
                "pin_memory": mb.DEFAULT_POSTTRAIN_TRAINER_PIN_MEMORY,
            },
        )

    def test_train_contacts_pretrain_dropout_prob_must_be_in_closed_open_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "trainbase_contacts_pretrain_dropout_prob"):
            mb.resolve_train_trainer_runtime_config(
                args=SimpleNamespace(trainbase_contacts_pretrain_dropout_prob=1.0),
                model_build_config=SimpleNamespace(hidden_dim=640),
                pin_memory=False,
            )

    def test_train_contacts_pretrain_dropout_mode_must_be_known(self) -> None:
        with self.assertRaisesRegex(ValueError, "contacts_pretrain_dropout_injection_mode"):
            mb.resolve_train_trainer_runtime_config(
                args=SimpleNamespace(trainbase_contacts_pretrain_dropout_injection_mode="bad_mode"),
                model_build_config=SimpleNamespace(hidden_dim=640),
                pin_memory=False,
            )


if __name__ == "__main__":
    unittest.main()
