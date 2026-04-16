#!/usr/bin/env python3
"""Static guard for current trainbase/posttrain active configs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

TRAINBASE_ACTIVE_CONFIGS: tuple[str, ...] = (
    "config/exp_phase_mpl.json",
    "config/exp_phase_mpl.clean.json",
    "config/exp_phase_DirectBranch_v1_d1_noreset.json",
)

POSTTRAIN_NEWFLOW_ACTIVE_CONFIGS: tuple[str, ...] = (
    "config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json",
    "config/posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json",
    "config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260227_fromarmchain.json",
    "config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json",
    "config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json",
    "config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json",
)

ACTIVE_CONFIG_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "direct_pose_hinge_",
    "contact_meas_provider",
)
ACTIVE_CONFIG_FORBIDDEN_EXACT_KEYS: tuple[str, ...] = (
    "direct_hinge_delta",
    "diag_input_stats",
    "eval_horizon",
    "eval_warmup",
    "foot_contact_threshold",
    "freerun_horizon",
    "freerun_weight",
    "monitor_batches",
    "patience",
    "tf_warmup_epochs",
)

TRAINBASE_REMOVED_STAGE_ROOT_KEYS: tuple[str, ...] = (
    "teacher_rot_noise_deg_start",
    "teacher_rot_noise_deg_end",
    "teacher_rot_noise_prob_start",
    "teacher_rot_noise_prob_end",
    "targets",
)
TRAINBASE_REMOVED_STAGE_PARAM_KEYS: tuple[str, ...] = (
    "freerun_horizon",
    "freerun_weight",
    "teacher_rot_noise_deg",
    "teacher_rot_noise_prob",
    "input_step_noise_prob",
    "input_noise_profile",
)
TRAINBASE_REMOVED_STAGE_TRAINER_KEYS: tuple[str, ...] = (
    "freerun_horizon",
    "freerun_weight",
)

POSTTRAIN_REMOVED_TARGET_KEYS: tuple[str, ...] = (
    "train_so3_corrector",
    "train_contact_plan_init",
    "train_contact_plan",
    "train_contact_meas",
    "train_contact_td_hazard",
)
POSTTRAIN_RETIRED_SHELL_PREFIXES: tuple[str, ...] = (
    "direct_pose_hinge_",
    "contact_meas_provider",
    "contact_td_hazard_",
    "contact_ttc_",
)
POSTTRAIN_RETIRED_SHELL_EXACT_KEYS: tuple[str, ...] = (
    "direct_hinge_delta",
    "train_contact_ttc",
)


def _load_json(rel_path: str) -> tuple[Path, dict[str, Any]]:
    path = REPO_ROOT / rel_path
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f"top-level JSON must be an object, got {type(payload).__name__}")
    return path, payload


def _find_forbidden_keys(
    payload: dict[str, Any],
    *,
    prefixes: tuple[str, ...],
    exact_keys: tuple[str, ...],
) -> list[str]:
    hits: list[str] = []
    for raw_key in payload.keys():
        key = str(raw_key)
        if key in exact_keys or any(key.startswith(prefix) for prefix in prefixes):
            hits.append(key)
    return sorted(hits)


def _check_trainbase_config(rel_path: str) -> list[str]:
    errs: list[str] = []
    try:
        path, payload = _load_json(rel_path)
    except Exception as exc:
        return [f"{rel_path}: failed to load JSON ({exc})"]
    hits = _find_forbidden_keys(
        payload,
        prefixes=ACTIVE_CONFIG_FORBIDDEN_PREFIXES,
        exact_keys=ACTIVE_CONFIG_FORBIDDEN_EXACT_KEYS,
    )
    if hits:
        errs.append(
            f"{path}: active trainbase config must not contain retired keys: {', '.join(hits)}"
        )
    schedule = payload.get("freerun_stage_schedule")
    if isinstance(schedule, list):
        removed_stage_hits: list[str] = []
        for idx, stage in enumerate(schedule):
            if not isinstance(stage, dict):
                continue
            for key in TRAINBASE_REMOVED_STAGE_ROOT_KEYS:
                if key in stage:
                    removed_stage_hits.append(f"freerun_stage_schedule[{idx}].{key}")
            params = stage.get("params")
            if isinstance(params, dict):
                for key in TRAINBASE_REMOVED_STAGE_PARAM_KEYS:
                    if key in params:
                        removed_stage_hits.append(f"freerun_stage_schedule[{idx}].params.{key}")
            trainer_cfg = stage.get("trainer")
            if isinstance(trainer_cfg, dict):
                for key in TRAINBASE_REMOVED_STAGE_TRAINER_KEYS:
                    if key in trainer_cfg:
                        removed_stage_hits.append(f"freerun_stage_schedule[{idx}].trainer.{key}")
        if removed_stage_hits:
            errs.append(
                f"{path}: active trainbase config must not contain removed stage-schedule keys: "
                + ", ".join(removed_stage_hits)
            )
    return errs


def _check_posttrain_config(rel_path: str) -> list[str]:
    errs: list[str] = []
    try:
        path, payload = _load_json(rel_path)
    except Exception as exc:
        return [f"{rel_path}: failed to load JSON ({exc})"]

    removed_target_hits = [key for key in POSTTRAIN_REMOVED_TARGET_KEYS if key in payload]
    if removed_target_hits:
        errs.append(
            f"{path}: posttrain active config must not contain removed target keys: "
            + ", ".join(removed_target_hits)
        )

    retired_shell_hits = _find_forbidden_keys(
        payload,
        prefixes=POSTTRAIN_RETIRED_SHELL_PREFIXES,
        exact_keys=POSTTRAIN_RETIRED_SHELL_EXACT_KEYS,
    )
    if retired_shell_hits:
        errs.append(
            f"{path}: posttrain active config must not contain retired shell keys: "
            + ", ".join(retired_shell_hits)
        )

    train_direct_pose = bool(payload.get("train_direct_pose", False))
    train_lambda_head = bool(payload.get("train_lambda_head", False))
    if train_direct_pose == train_lambda_head:
        errs.append(
            f"{path}: expected XOR target contract; got train_direct_pose={train_direct_pose}, "
            f"train_lambda_head={train_lambda_head}"
        )

    return errs


def main() -> int:
    errs: list[str] = []

    for rel_path in TRAINBASE_ACTIVE_CONFIGS:
        errs.extend(_check_trainbase_config(rel_path))
    for rel_path in POSTTRAIN_NEWFLOW_ACTIVE_CONFIGS:
        errs.extend(_check_posttrain_config(rel_path))

    if errs:
        print(f"[FAIL] newflow active config guard failed ({len(errs)} issue(s)):")
        for err in errs:
            print(f" - {err}")
        return 1

    checked = len(TRAINBASE_ACTIVE_CONFIGS) + len(POSTTRAIN_NEWFLOW_ACTIVE_CONFIGS)
    print(f"[OK] newflow active config guard passed ({checked} config(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
