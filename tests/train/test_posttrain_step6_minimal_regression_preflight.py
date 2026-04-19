from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from train.validate import run_posttrain_step6_minimal_regression as step6


class PosttrainStep6MinimalRegressionPreflightTest(unittest.TestCase):
    def test_find_missing_case_artifacts_reports_config_and_payload_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing_config_case = step6._Case(
                mode="direct",
                config_path=root / "missing_config.json",
                run_name="run",
                baseline_run_name="base",
            )
            self.assertEqual(
                step6._find_missing_case_artifacts(missing_config_case),
                [("config", root / "missing_config.json")],
            )

            existing_bundle = root / "bundle.json"
            existing_bundle.write_text("{}", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ckpt_in": str(root / "missing_ckpt.pth"),
                        "bundle_json": str(existing_bundle),
                        "paths": [str(root / "missing_walk.npz")],
                    }
                ),
                encoding="utf-8",
            )
            case = step6._Case(
                mode="lambda",
                config_path=config_path,
                run_name="run",
                baseline_run_name="base",
            )

            self.assertEqual(
                step6._find_missing_case_artifacts(case),
                [
                    ("ckpt_in", root / "missing_ckpt.pth"),
                    ("paths[0]", root / "missing_walk.npz"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
