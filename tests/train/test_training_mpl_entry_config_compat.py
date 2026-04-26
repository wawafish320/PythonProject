from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from train.models import MotionJointLoss
from train.training_MPL import _parse_train_entry_args


class TrainingMPLEntryConfigCompatTest(unittest.TestCase):
    def test_config_json_rejects_removed_adaptive_bone_weights(self) -> None:
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

            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    _parse_train_entry_args(["--config_json", str(cfg_path)])

        self.assertEqual(exc.exception.code, 2)

    def test_motion_joint_loss_rejects_removed_adaptive_bone_weights(self) -> None:
        with self.assertRaises(TypeError) as exc:
            MotionJointLoss(adaptive_bone_weights=True)

        self.assertIn("adaptive_bone_weights", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
