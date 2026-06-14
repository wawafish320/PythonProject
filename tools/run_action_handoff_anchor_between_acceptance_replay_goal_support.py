#!/usr/bin/env python3
"""Run anchor-between replay with target endpoint support appended to planner goal.

This is a debug-only wrapper around the existing replay. It keeps the replay code
unchanged and patches only the small contact planner item builder used during
predicted-contact training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import run_action_handoff_anchor_between_acceptance_replay as replay
from tools.run_action_handoff_between_gru_bridge_probe import (
    GOAL_CONTACT_MODES,
    CONTACT_LABEL_THRESHOLD,
    _build_items as _bridge_build_items,
)


_ORIG_THRESHOLD_FN = replay._residual_contact_threshold01_from_raw
_goal_contact_mode = "target_support_end"
_goal_threshold01: Tuple[float, float] = (CONTACT_LABEL_THRESHOLD, CONTACT_LABEL_THRESHOLD)


def _threshold_hook(scaler, raw_threshold):
    global _goal_threshold01
    _goal_threshold01 = _ORIG_THRESHOLD_FN(scaler, raw_threshold)
    return _goal_threshold01


def _build_items_with_goal(clips, *, horizon: int, context_len: int, stride: int):
    return _bridge_build_items(
        clips,
        horizon=int(horizon),
        context_len=int(context_len),
        stride=int(stride),
        goal_contact_mode=str(_goal_contact_mode),
        goal_support_threshold01=_goal_threshold01,
    )


def _parse_wrapper_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--goal-contact-mode",
        choices=GOAL_CONTACT_MODES,
        default="target_support_end",
        help="Planner goal repair passed to the debug contact item builder.",
    )
    args, remaining = parser.parse_known_args()
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], *remaining]
    try:
        base_args = replay.parse_args()
    finally:
        sys.argv = old_argv
    setattr(base_args, "goal_contact_mode", str(args.goal_contact_mode))
    return base_args


def main() -> None:
    global _goal_contact_mode
    args = _parse_wrapper_args()
    _goal_contact_mode = str(args.goal_contact_mode)
    replay._residual_contact_threshold01_from_raw = _threshold_hook
    replay._residual_build_items = _build_items_with_goal
    payload = replay.run(args)

    payload.setdefault("config", {})["goal_contact_mode"] = _goal_contact_mode
    payload.setdefault("schema", {})["predicted_probe_goal_contact"] = {
        "mode": _goal_contact_mode,
        "target_support_end_onehot_dim": 4 if _goal_contact_mode == "target_support_end" else 0,
        "threshold01": [float(v) for v in _goal_threshold01],
    }
    summary_json = Path(payload["artifacts"]["summary_json"])
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    sidecar = summary_json.with_name("goal_contact_mode_sidecar.json")
    sidecar.write_text(
        json.dumps(payload["schema"]["predicted_probe_goal_contact"], indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
