from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from train.models import MotionJointLoss
from train.training_MPL import _parse_train_entry_args


class TrainingMPLEntryConfigCompatTest(unittest.TestCase):
    def test_config_json_accepts_adaptive_bone_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "data": "raw_data/processed_data",
                        "bundle_json": "models/pretrain_template.json",
                        "adaptive_bone_weights": True,
                    }
                ),
                encoding="utf-8",
            )

            args = _parse_train_entry_args(["--config_json", str(cfg_path)])

        self.assertTrue(hasattr(args, "adaptive_bone_weights"))
        self.assertTrue(args.adaptive_bone_weights)

    def test_motion_joint_loss_accepts_compat_flag(self) -> None:
        loss_fn = MotionJointLoss(adaptive_bone_weights=True)

        self.assertTrue(loss_fn.use_adaptive_weights)
        loss_fn._joint_weight_cache = {"sentinel": object()}
        loss_fn._invalidate_weight_cache()
        self.assertEqual(loss_fn._joint_weight_cache, {})


if __name__ == "__main__":
    unittest.main()
