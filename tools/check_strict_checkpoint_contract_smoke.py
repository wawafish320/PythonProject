#!/usr/bin/env python3
from __future__ import annotations

"""Smoke checks for default strict checkpoint contract behavior."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import posttrain as _posttrain
from train import posttrain_build_shell as _posttrain_build_shell
from train.checkpoint.load_schema import (
    ContactPlanBuildOverrides,
    RemovedCheckpointCompatError,
    resolve_contact_plan_build_cfg,
)
from train.checkpoint.contract import (
    compute_resolved_build_manifest_hash,
    enforce_strict_current_build_manifest_contract,
)
from train.checkpoint.fingerprint import compare_fingerprints


class _DummyStrictModel(torch.nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._state = dict(state)

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return dict(self._state)


def _minimal_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "ckpt_in": "/tmp/nonexistent_legacy.pth",
        "train_direct_pose": True,
        "load_context": "resume",
        "event_clock": "on",
        "width": 32,
        "direct_pose_enable": True,
        "direct_pose_hidden": 32,
        "direct_pose_meas_mode": "concat",
        "direct_pose_feat_source": "cond",
        "direct_pose_time_pe_dim": 0,
    }
    payload.update(overrides)
    return payload


def main() -> int:
    manifest = {
        "config": {
            "hidden_dim": 32,
            "direct_pose_phase_z_mode": "concat",
            "contact_plan_enable": False,
        },
        "trace": [{"field": "direct_pose_phase_z_mode", "reason": "ignored in hard hash"}],
    }
    manifest_hash = compute_resolved_build_manifest_hash(manifest)
    enforce_strict_current_build_manifest_contract(
        current_manifest=manifest,
        checkpoint_manifest=manifest,
        checkpoint_manifest_hash=manifest_hash,
    )

    try:
        enforce_strict_current_build_manifest_contract(
            current_manifest=manifest,
            checkpoint_manifest=None,
            checkpoint_manifest_hash=None,
        )
    except (SystemExit, RemovedCheckpointCompatError) as exc:
        message = str(exc)
        if "checkpoint missing resolved_build_manifest" not in message:
            raise AssertionError(message)
        if "tools/migrate_legacy_posttrain_ckpt.py" not in message:
            raise AssertionError(message)
    else:
        raise AssertionError("missing strict checkpoint manifest must fail")

    missing_meas_payload = _minimal_payload(
        direct_pose_enable=True,
        direct_pose_hidden=32,
        direct_pose_feat_source="cond",
        direct_pose_time_pe_dim=0,
    )
    missing_meas_payload.pop("direct_pose_meas_mode", None)
    try:
        _posttrain._cfg_from_payload(missing_meas_payload)
    except SystemExit as exc:
        message = str(exc)
        if "direct_pose_meas_mode" not in message:
            raise AssertionError(message)
        if "2026-04-28 strict direct-pose shape-inference unload" not in message:
            raise AssertionError(message)
        if "no checkpoint shape/posttrain_cfg replacement" not in message:
            raise AssertionError(message)
    else:
        raise AssertionError("strict direct-pose config missing meas_mode must fail")

    try:
        _posttrain._cfg_from_payload(
            _minimal_payload(
                direct_pose_enable=True,
                direct_pose_hidden=32,
                direct_pose_meas_mode="concat",
                direct_pose_meas_mode_override="concat",
                direct_pose_feat_source="cond",
                direct_pose_time_pe_dim=0,
            )
        )
    except SystemExit as exc:
        message = str(exc)
        if "direct_pose_meas_mode_override" not in message:
            raise AssertionError(message)
        if "2026-04-28 strict direct-pose shape-inference unload" not in message:
            raise AssertionError(message)
    else:
        raise AssertionError("strict direct-pose meas_mode override shim must fail")

    default_cfg = _posttrain._cfg_from_payload(_minimal_payload())
    if not bool(default_cfg.strict_current_model_build):
        raise AssertionError("strict_current_model_build must default to true")

    try:
        _posttrain._cfg_from_payload(_minimal_payload(legacy_checkpoint_compat=True))
    except SystemExit as exc:
        message = str(exc)
        if "legacy_checkpoint_compat" not in message or "Removed" not in message:
            raise AssertionError(message)
    else:
        raise AssertionError("legacy_checkpoint_compat field must fail-fast")

    try:
        _posttrain._cfg_from_payload(_minimal_payload(strict_current_model_build=False))
    except SystemExit as exc:
        message = str(exc)
        if "strict_current_model_build=false" not in message or "strict/current-only" not in message:
            raise AssertionError(message)
    else:
        raise AssertionError("strict_current_model_build=false must fail-fast")

    current_fingerprints = {
        "io_signature_hash": "io",
        "module_graph_hash": "graph",
        "build_order_hash": "order",
        "weights_hash": "current_weights",
        "train_policy_hash": "policy",
    }
    checkpoint_fingerprints = dict(current_fingerprints)
    checkpoint_fingerprints["weights_hash"] = "old_weights"
    summary = compare_fingerprints(checkpoint_fingerprints, current_fingerprints)
    if not _posttrain_build_shell._posttrain_fingerprint_enforce_required(
        summary,
        load_context="chain_hop",
    ):
        raise AssertionError("strict chain_hop must enforce required weights_hash mismatch")

    state_with_obs_init = {
        "contact_plan_cell.weight_hh": torch.zeros(64, 64),
        "contact_plan_head.4.weight": torch.zeros(2, 64),
        "contact_plan_init_head.1.weight": torch.zeros(13, 7),
    }
    try:
        resolve_contact_plan_build_cfg(
            state_dict=state_with_obs_init,
            in_state_dim=8,
            cond_dim=4,
            contact_dim=2,
            overrides=ContactPlanBuildOverrides(init_mode="learnable"),
        )
    except (SystemExit, RemovedCheckpointCompatError) as exc:
        if "contact_plan_init_head.*" not in str(exc) or "no load-time obs-init replacement" not in str(exc):
            raise AssertionError(str(exc))
    else:
        raise AssertionError("contact_plan_init_head.* must not imply obs-init")

    try:
        resolve_contact_plan_build_cfg(
            state_dict=state_with_obs_init,
            in_state_dim=8,
            cond_dim=4,
            contact_dim=2,
            overrides=ContactPlanBuildOverrides(init_mode="obs"),
        )
    except (SystemExit, RemovedCheckpointCompatError) as exc:
        if "contact_plan_init_head.*" not in str(exc) or "no load-time obs-init replacement" not in str(exc):
            raise AssertionError(str(exc))
    else:
        raise AssertionError("obs-init load-time interpretation must fail even with explicit mode")

    stripped_obs = dict(state_with_obs_init)
    stripped_obs.pop("contact_plan_init_head.1.weight")
    stripped = resolve_contact_plan_build_cfg(
        state_dict=stripped_obs,
        in_state_dim=8,
        cond_dim=4,
        contact_dim=2,
        overrides=ContactPlanBuildOverrides(init_mode="learnable"),
    )
    if stripped.init_mode != "learnable":
        raise AssertionError("migration-time contact_plan_init_head.* strip did not preserve learnable init")

    try:
        _posttrain_build_shell._validate_strict_current_direct_pose_checkpoint_shapes(
            model=_DummyStrictModel({"direct_pose_head.0.weight": torch.zeros(4, 8)}),
            state_dict={"direct_pose_head.0.weight": torch.zeros(4, 6)},
            stage="unit-smoke",
        )
    except SystemExit as exc:
        message = str(exc)
        if "direct_pose_head.0.weight" not in message:
            raise AssertionError(message)
        if "2026-04-28 strict direct-pose shape-inference unload" not in message:
            raise AssertionError(message)
        if "no load-time shape/posttrain_cfg replacement" not in message:
            raise AssertionError(message)
    else:
        raise AssertionError("strict direct-pose checkpoint shape mismatch must fail")

    print("[OK] strict checkpoint contract smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
