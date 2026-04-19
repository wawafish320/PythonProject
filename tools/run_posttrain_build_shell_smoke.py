#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.configuration.io import dump_json, load_json
from train.posttrain import _build_dataset_and_loader, _cfg_from_payload
from train.posttrain_build_shell import (
    _instantiate_posttrain_model,
    _load_posttrain_checkpoint_into_model,
    _resolve_posttrain_model_build_state,
)


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    config_json: Path | None
    ckpt_in: Path
    expected_top_level_keys: tuple[str, ...]
    payload_source: str
    encoder_bundle: Path | None = None
    paths: tuple[Path, ...] | None = None


def _default_fixtures() -> tuple[FixtureSpec, ...]:
    return (
        FixtureSpec(
            name="direct_pose_walkf",
            config_json=Path("config/posttrain_direct_pose_walkf.json"),
            ckpt_in=Path(
                "models/MLPL2_DirectBranch_v1_20260317/exp_phase_DirectBranch_v1_d1_20260317/"
                "ckpt_best_free_exp_phase_DirectBranch_v1_d1_20260317.pth"
            ),
            expected_top_level_keys=("config", "model"),
            payload_source="config_json+overrides",
            encoder_bundle=Path("models/motion_encoder_equiv_stageA.pt"),
            paths=(Path("raw_data/processed_data/Walk_F.npz"),),
        ),
        FixtureSpec(
            name="lambda_head_lowlr72_embedded_cfg",
            config_json=None,
            ckpt_in=Path(
                "models/__tmp_71_lowlr_to72lambda_20260314/lambda/"
                "ckpt_last_WalkF_stage7_lambda_from_lowlr72_20260314.pth"
            ),
            expected_top_level_keys=("model", "posttrain_cfg"),
            payload_source="embedded_posttrain_cfg",
        ),
    )


def _load_fixture_payload(spec: FixtureSpec) -> dict[str, Any]:
    if spec.config_json is not None:
        payload = load_json(spec.config_json)
        if not isinstance(payload, dict):
            raise TypeError(f"{spec.config_json} must contain a JSON object payload.")
        payload = dict(payload)
    else:
        ckpt = torch.load(spec.ckpt_in, map_location="cpu")
        if not isinstance(ckpt, dict) or not isinstance(ckpt.get("posttrain_cfg"), dict):
            raise SystemExit(
                f"[FATAL] fixture {spec.name} requires checkpoint-embedded posttrain_cfg: {spec.ckpt_in}"
            )
        payload = dict(ckpt["posttrain_cfg"])

    payload["ckpt_in"] = str(spec.ckpt_in)
    if spec.encoder_bundle is not None:
        payload["encoder_bundle"] = str(spec.encoder_bundle)
    if spec.paths is not None:
        payload["paths"] = [str(path) for path in spec.paths]
    return payload


def _assert_top_level_keys(*, spec: FixtureSpec, ckpt: Any) -> list[str]:
    if not isinstance(ckpt, dict):
        raise SystemExit(f"[FATAL] fixture {spec.name} did not load a dict checkpoint payload.")
    keys = sorted(str(key) for key in ckpt.keys())
    expected = sorted(spec.expected_top_level_keys)
    if keys != expected:
        raise SystemExit(
            f"[FATAL] fixture {spec.name} top-level keys changed: expected {expected}, got {keys}."
        )
    return keys


def _assert_build_state(*, spec: FixtureSpec, build_state: Any, ds: Any) -> None:
    if int(build_state.width) <= 0:
        raise SystemExit(f"[FATAL] fixture {spec.name} resolved non-positive width={build_state.width}.")
    if int(build_state.contact_dim) != int(getattr(ds, "contact_dim", 0) or 0):
        raise SystemExit(f"[FATAL] fixture {spec.name} contact_dim drifted during build-state resolution.")
    if int(build_state.pose_hist_dim) != int(getattr(ds, "pose_hist_dim", 0) or 0):
        raise SystemExit(f"[FATAL] fixture {spec.name} pose_hist_dim drifted during build-state resolution.")
    if int(build_state.angvel_dim) != int(getattr(ds, "angvel_dim", 0) or 0):
        raise SystemExit(f"[FATAL] fixture {spec.name} angvel_dim drifted during build-state resolution.")


def _assert_model_after_instantiate(*, spec: FixtureSpec, model: Any, build_state: Any) -> None:
    if bool(build_state.contact_plan_enable) and not bool(getattr(model, "contact_plan_enable", False)):
        raise SystemExit(f"[FATAL] fixture {spec.name} lost contact_plan_enable during instantiate.")
    if bool(build_state.direct_pose_cfg.enable) and getattr(model, "direct_pose_head", None) is None:
        raise SystemExit(f"[FATAL] fixture {spec.name} failed to instantiate direct_pose_head.")
    if bool(build_state.lambda_fusion_enable) and getattr(model, "lambda_fusion_head", None) is None:
        raise SystemExit(f"[FATAL] fixture {spec.name} failed to instantiate lambda_fusion_head.")


def _assert_model_after_load(*, spec: FixtureSpec, cfg: Any, model: Any, build_state: Any) -> None:
    if cfg.encoder_bundle is not None and cfg.encoder_bundle.expanduser().is_file():
        if getattr(model, "frozen_encoder", None) is None:
            raise SystemExit(f"[FATAL] fixture {spec.name} lost encoder bundle attach.")
    if bool(build_state.contact_plan_enable) and getattr(model, "contact_plan_cell", None) is None:
        raise SystemExit(f"[FATAL] fixture {spec.name} lost contact_plan path after load.")
    if bool(build_state.direct_pose_cfg.enable) and getattr(model, "direct_pose_head", None) is None:
        raise SystemExit(f"[FATAL] fixture {spec.name} lost direct_pose_head after load.")
    if bool(build_state.lambda_fusion_enable) and getattr(model, "lambda_fusion_head", None) is None:
        raise SystemExit(f"[FATAL] fixture {spec.name} lost lambda_fusion_head after load.")


def _run_fixture(spec: FixtureSpec) -> dict[str, Any]:
    payload = _load_fixture_payload(spec)
    cfg = _cfg_from_payload(payload)
    ckpt = torch.load(cfg.ckpt_in.expanduser(), map_location="cpu")
    top_level_keys = _assert_top_level_keys(spec=spec, ckpt=ckpt)
    _norm_spec, ds, _batch_iter = _build_dataset_and_loader(cfg)
    build_state = _resolve_posttrain_model_build_state(cfg=cfg, ds=ds)
    _assert_build_state(spec=spec, build_state=build_state, ds=ds)
    model = _instantiate_posttrain_model(
        cfg=cfg,
        ds=ds,
        device=torch.device("cpu"),
        build_state=build_state,
    )
    _assert_model_after_instantiate(spec=spec, model=model, build_state=build_state)
    _load_posttrain_checkpoint_into_model(cfg=cfg, model=model, build_state=build_state)
    _assert_model_after_load(spec=spec, cfg=cfg, model=model, build_state=build_state)
    return {
        "name": spec.name,
        "status": "ok",
        "payload_source": spec.payload_source,
        "config_json": str(spec.config_json) if spec.config_json is not None else None,
        "ckpt_in": str(spec.ckpt_in),
        "encoder_bundle": str(cfg.encoder_bundle) if cfg.encoder_bundle is not None else None,
        "paths": [str(path) for path in (cfg.paths or ())],
        "top_level_keys": top_level_keys,
        "build_state": {
            "width": int(build_state.width),
            "contact_dim": int(build_state.contact_dim),
            "angvel_dim": int(build_state.angvel_dim),
            "pose_hist_dim": int(build_state.pose_hist_dim),
            "contact_plan_enable": bool(build_state.contact_plan_enable),
            "contact_plan_inject": str(build_state.contact_plan_inject),
            "direct_pose_enable": bool(build_state.direct_pose_cfg.enable),
            "direct_pose_split_enable": bool(build_state.direct_pose_cfg.split_enable),
            "direct_pose_arm_split_enable": bool(build_state.direct_pose_cfg.arm_split_enable),
            "direct_pose_feat_source": str(build_state.direct_pose_cfg.feat_source),
            "direct_pose_use_phase_z": bool(build_state.direct_pose_cfg.use_phase_z),
            "direct_pose_phase_z_mode": str(build_state.direct_pose_cfg.phase_z_mode),
            "lambda_fusion_enable": bool(build_state.lambda_fusion_enable),
            "lambda_fusion_mode": str(build_state.lambda_fusion_mode),
            "lambda_fusion_use_rollout_step": bool(build_state.lambda_fusion_use_rollout_step),
        },
        "model_paths": {
            "frozen_encoder_attached": getattr(model, "frozen_encoder", None) is not None,
            "contact_plan_enable": bool(getattr(model, "contact_plan_enable", False)),
            "direct_pose_head": getattr(model, "direct_pose_head", None) is not None,
            "lambda_fusion_head": getattr(model, "lambda_fusion_head", None) is not None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run reusable Step 3 posttrain build-shell smoke gate.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("debug_output/_posttrain_build_shell_smokes_20260418"),
        help="Directory for smoke summary artifacts.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "posttrain_build_shell_smoke_summary.json"

    results = [_run_fixture(spec) for spec in _default_fixtures()]
    dump_json(
        summary_path,
        {
            "fixtures": results,
        },
    )
    print(f"[smoke] wrote summary: {summary_path}")
    for result in results:
        build_state = result["build_state"]
        print(
            "[smoke] "
            f"{result['name']}: "
            f"keys={result['top_level_keys']} "
            f"contact_plan={build_state['contact_plan_enable']} "
            f"direct_pose={build_state['direct_pose_enable']} "
            f"lambda_fusion={build_state['lambda_fusion_enable']}"
        )


if __name__ == "__main__":
    main()
