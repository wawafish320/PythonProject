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
from train.posttrain import _build_dataset_and_loader, _build_model_and_trainer, _cfg_from_payload
from train.posttrain_build_shell import _build_posttrain_model_from_ckpt


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    config_json: Path | None
    ckpt_in: Path
    payload_source: str
    encoder_bundle: Path | None = None
    paths: tuple[Path, ...] | None = None


def _default_fixtures() -> tuple[FixtureSpec, ...]:
    return (
        FixtureSpec(
            name="direct_pose_walkf_config_json",
            config_json=Path("config/posttrain_direct_pose_walkf.json"),
            ckpt_in=Path(
                "models/MLPL2_DirectBranch_v1_20260317/exp_phase_DirectBranch_v1_d1_20260317/"
                "ckpt_best_free_exp_phase_DirectBranch_v1_d1_20260317.pth"
            ),
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
    payload["load_context"] = "resume"
    if spec.encoder_bundle is not None:
        payload["encoder_bundle"] = str(spec.encoder_bundle)
    if spec.paths is not None:
        payload["paths"] = [str(path) for path in spec.paths]
    return payload


def _normalize_optional_pathlike(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _assert_overlay_artifacts(*, spec: FixtureSpec, cfg: Any, trainer: Any, loss_fn: Any, model: Any) -> dict[str, Any]:
    if getattr(trainer, "pose_hist_dim", None) is None:
        raise SystemExit(f"[FATAL] fixture {spec.name} lost shared runtime attach: pose_hist_dim missing.")
    if getattr(trainer, "pose_hist_mu", None) is None or getattr(trainer, "pose_hist_std", None) is None:
        raise SystemExit(f"[FATAL] fixture {spec.name} lost shared pose-history runtime attach.")
    if getattr(loss_fn, "mu_y", None) is None:
        raise SystemExit(f"[FATAL] fixture {spec.name} lost loss_fn.mu_y runtime stats.")
    if getattr(loss_fn, "std_y", None) is None:
        raise SystemExit(f"[FATAL] fixture {spec.name} lost loss_fn.std_y runtime stats.")

    expected_clamp = float(getattr(cfg, "posttrain_contacts_pretrain_clamp", 1.0) or 1.0)
    actual_clamp = float(getattr(trainer, "posttrain_contacts_pretrain_clamp", -1.0))
    if abs(actual_clamp - expected_clamp) > 1e-9:
        raise SystemExit(
            f"[FATAL] fixture {spec.name} drifted posttrain contact-pretrain clamp: "
            f"expected {expected_clamp}, got {actual_clamp}."
        )

    expected_raw_spec = _normalize_optional_pathlike(getattr(cfg, "posttrain_contacts_pretrain_affine_stats", None))
    actual_raw_spec = _normalize_optional_pathlike(getattr(trainer, "posttrain_contacts_pretrain_affine_stats_spec", None))
    if actual_raw_spec != expected_raw_spec:
        raise SystemExit(
            f"[FATAL] fixture {spec.name} drifted posttrain contact-pretrain raw spec mapping: "
            f"expected {expected_raw_spec}, got {actual_raw_spec}."
        )

    affine = getattr(trainer, "posttrain_contacts_pretrain_affine", None)
    if expected_raw_spec is None:
        if affine is not None:
            raise SystemExit(f"[FATAL] fixture {spec.name} unexpectedly resolved contact-pretrain affine without raw spec.")
    else:
        if not isinstance(affine, dict):
            raise SystemExit(f"[FATAL] fixture {spec.name} failed to resolve contact-pretrain affine payload.")
        missing = [key for key in ("scale", "bias", "eps") if key not in affine]
        if missing:
            raise SystemExit(
                f"[FATAL] fixture {spec.name} contact-pretrain affine payload missing keys: {missing}."
            )

    neutral_clamp = float(getattr(trainer, "contacts_pretrain_clamp", -1.0))
    if abs(neutral_clamp - actual_clamp) > 1e-9:
        raise SystemExit(
            f"[FATAL] fixture {spec.name} drifted neutral contact-pretrain clamp mapping: "
            f"expected {actual_clamp}, got {neutral_clamp}."
        )
    neutral_raw_spec = _normalize_optional_pathlike(getattr(trainer, "contacts_pretrain_affine_stats_spec", None))
    if neutral_raw_spec != actual_raw_spec:
        raise SystemExit(
            f"[FATAL] fixture {spec.name} drifted neutral contact-pretrain raw spec mapping: "
            f"expected {actual_raw_spec}, got {neutral_raw_spec}."
        )
    neutral_affine = getattr(trainer, "contacts_pretrain_affine", None)
    if neutral_affine != affine:
        raise SystemExit(f"[FATAL] fixture {spec.name} drifted neutral parsed contact-pretrain affine mapping.")
    if not bool(getattr(trainer, "contacts_pretrain_runtime_attached", False)):
        raise SystemExit(f"[FATAL] fixture {spec.name} lost contacts_pretrain_runtime_attached marker.")

    expected_contact_vxy_mode = str(getattr(cfg, "contact_meas_vxy_mode", "abs") or "abs").strip().lower()
    actual_contact_vxy_mode = str(getattr(trainer, "contact_meas_vxy_mode", "") or "").strip().lower()
    if actual_contact_vxy_mode != expected_contact_vxy_mode:
        raise SystemExit(
            f"[FATAL] fixture {spec.name} drifted contact_meas_vxy_mode: "
            f"expected {expected_contact_vxy_mode}, got {actual_contact_vxy_mode}."
        )

    expected_lambda_mode = str(getattr(cfg, "lambda_reliability_mode", "none") or "none")
    actual_lambda_mode = str(getattr(trainer, "lambda_reliability_mode", "") or "")
    if actual_lambda_mode != expected_lambda_mode:
        raise SystemExit(
            f"[FATAL] fixture {spec.name} drifted lambda_reliability_mode: "
            f"expected {expected_lambda_mode}, got {actual_lambda_mode}."
        )

    encoder_bundle = getattr(cfg, "encoder_bundle", None)
    if encoder_bundle is not None and Path(str(encoder_bundle)).expanduser().is_file():
        if getattr(model, "frozen_encoder", None) is None:
            raise SystemExit(f"[FATAL] fixture {spec.name} lost encoder bundle attach.")

    return {
        "name": spec.name,
        "status": "ok",
        "payload_source": spec.payload_source,
        "config_json": str(spec.config_json) if spec.config_json is not None else None,
        "ckpt_in": str(spec.ckpt_in),
        "bundle_json": str(getattr(cfg, "bundle_json", "")),
        "encoder_bundle": str(encoder_bundle) if encoder_bundle is not None else None,
        "paths": [str(path) for path in (getattr(cfg, "paths", None) or ())],
        "trainer_runtime": {
            "pose_hist_dim": int(getattr(trainer, "pose_hist_dim", 0) or 0),
            "has_pose_hist_mu": getattr(trainer, "pose_hist_mu", None) is not None,
            "has_pose_hist_std": getattr(trainer, "pose_hist_std", None) is not None,
            "posttrain_contacts_pretrain_clamp": actual_clamp,
            "posttrain_contacts_pretrain_affine_stats_spec": actual_raw_spec,
            "posttrain_contacts_pretrain_affine_keys": sorted(affine.keys()) if isinstance(affine, dict) else None,
            "contacts_pretrain_clamp": neutral_clamp,
            "contacts_pretrain_affine_stats_spec": neutral_raw_spec,
            "has_contacts_pretrain_affine": neutral_affine is not None,
            "contacts_pretrain_runtime_attached": bool(getattr(trainer, "contacts_pretrain_runtime_attached", False)),
            "contact_meas_vxy_mode": actual_contact_vxy_mode,
            "lambda_reliability_mode": actual_lambda_mode,
        },
        "loss_runtime": {
            "has_mu_y": getattr(loss_fn, "mu_y", None) is not None,
            "has_std_y": getattr(loss_fn, "std_y", None) is not None,
        },
        "model_paths": {
            "frozen_encoder_attached": getattr(model, "frozen_encoder", None) is not None,
            "contact_plan_enable": bool(getattr(model, "contact_plan_enable", False)),
            "direct_pose_head": getattr(model, "direct_pose_head", None) is not None,
            "lambda_fusion_head": getattr(model, "lambda_fusion_head", None) is not None,
        },
    }


def _run_fixture(spec: FixtureSpec) -> dict[str, Any]:
    payload = _load_fixture_payload(spec)
    cfg = _cfg_from_payload(payload)
    norm_spec, ds, _batch_iter = _build_dataset_and_loader(cfg)
    artifacts = _build_posttrain_model_from_ckpt(cfg=cfg, ds=ds, device=torch.device("cpu"))
    model = artifacts.model
    trainer = _build_model_and_trainer(cfg=cfg, ds=ds, model=model, norm_spec=norm_spec)
    return _assert_overlay_artifacts(
        spec=spec,
        cfg=cfg,
        trainer=trainer,
        loss_fn=trainer.loss_fn,
        model=model,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run posttrain runtime overlay smoke gate.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("debug_output/_posttrain_runtime_overlay_smokes_20260418"),
        help="Directory for smoke summary artifact.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "posttrain_runtime_overlay_smoke_summary.json"

    results = [_run_fixture(spec) for spec in _default_fixtures()]
    dump_json(summary_path, {"fixtures": results})
    print(f"[smoke] wrote summary: {summary_path}")
    for result in results:
        runtime = result["trainer_runtime"]
        print(
            "[smoke] "
            f"{result['name']}: "
            f"pose_hist_dim={runtime['pose_hist_dim']} "
            f"affine_spec={runtime['posttrain_contacts_pretrain_affine_stats_spec']} "
            f"lambda_mode={runtime['lambda_reliability_mode']}"
        )


if __name__ == "__main__":
    main()
