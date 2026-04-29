#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import train.posttrain as posttrain


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--trunk-mode", type=str, default="full", choices=("none", "last", "full"))
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-epoch", type=int, default=20)
    ap.add_argument("--save-step-ckpts", type=str, default="0,1,5,20")
    return ap.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    payload = posttrain.load_json(Path(args.config).expanduser())
    if not isinstance(payload, dict):
        raise SystemExit("[FATAL] config payload must be a dict")
    payload = dict(payload)
    payload["out_dir"] = str(args.out_dir)
    payload["run_name"] = str(args.run_name)
    payload["epochs"] = int(args.epochs)
    payload["steps_per_epoch"] = int(args.steps_per_epoch)
    payload["save_step_ckpts"] = str(args.save_step_ckpts)
    payload["load_context"] = "chain_hop"
    payload["direct_pose_nonleg_train_only"] = True
    payload["direct_pose_nonleg_trunk_mode"] = str(args.trunk_mode)

    os.makedirs(str(args.out_dir), exist_ok=True)
    config_out = Path(args.out_dir) / f"{args.run_name}.nonleg_trunk_config.json"
    _write_json(config_out, payload)
    print(
        "[ablation] delegating to train.posttrain with "
        f"direct_pose_nonleg_trunk_mode={args.trunk_mode}; config={config_out}"
    )
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "train.posttrain", "--config", str(config_out)],
            env=env,
        )
    )


if __name__ == "__main__":
    main()
