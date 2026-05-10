from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .configuration.norm_spec import ContactPretrainRuntime
from .data.dataset import DatasetRuntimeArtifacts
from .history import resolve_pose_hist_runtime_tensors

__all__ = [
    "SharedTrainerRuntime",
    "resolve_shared_trainer_runtime",
    "apply_shared_trainer_runtime",
    "apply_contacts_pretrain_runtime",
    "apply_loss_runtime_from_trainer",
]


@dataclass(frozen=True)
class SharedTrainerRuntime:
    pose_hist_scales: Optional[torch.Tensor]
    pose_hist_mu: Optional[torch.Tensor]
    pose_hist_std: Optional[torch.Tensor]
    yaw_forward_axis: int
    yaw_forward_axis_offset: float
    norm_template_path: Optional[str] = None
    bundle_json_path: Optional[str] = None
    out_dir: Optional[str] = None
    full_config: Optional[Dict[str, Any]] = None
    current_run_name: Optional[str] = None


def _stringify_optional_path(value: Optional[str | Path]) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def resolve_shared_trainer_runtime(
    *,
    dataset_artifacts: DatasetRuntimeArtifacts,
    trainer_default_yaw_forward_axis: int,
    yaw_forward_axis_override: Optional[int] = None,
    yaw_forward_offset_deg_override: Optional[float] = None,
    norm_template_path: Optional[str | Path] = None,
    bundle_json_path: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    full_config: Optional[Dict[str, Any]] = None,
    current_run_name: Optional[str] = None,
) -> SharedTrainerRuntime:
    pose_hist_scales, pose_hist_mu, pose_hist_std = resolve_pose_hist_runtime_tensors(
        dataset_artifacts.dataset
    )

    if yaw_forward_axis_override is not None:
        yaw_forward_axis = int(yaw_forward_axis_override)
    elif dataset_artifacts.forward_axis is not None:
        yaw_forward_axis = int(dataset_artifacts.forward_axis)
    else:
        yaw_forward_axis = int(trainer_default_yaw_forward_axis)

    if yaw_forward_offset_deg_override is not None:
        yaw_forward_axis_offset = float(math.radians(float(yaw_forward_offset_deg_override)))
    else:
        yaw_forward_axis_offset = float(dataset_artifacts.forward_axis_offset)

    return SharedTrainerRuntime(
        pose_hist_scales=pose_hist_scales,
        pose_hist_mu=pose_hist_mu,
        pose_hist_std=pose_hist_std,
        yaw_forward_axis=yaw_forward_axis,
        yaw_forward_axis_offset=yaw_forward_axis_offset,
        norm_template_path=_stringify_optional_path(norm_template_path),
        bundle_json_path=_stringify_optional_path(bundle_json_path),
        out_dir=_stringify_optional_path(out_dir),
        full_config=full_config,
        current_run_name=current_run_name,
    )


def apply_shared_trainer_runtime(trainer: Any, runtime: SharedTrainerRuntime) -> Any:
    renamed_fields = {
        "_norm_template_path": "norm_template_path",
        "_bundle_json_path": "bundle_json_path",
        "_current_run_name": "current_run_name",
    }
    direct_fields = (
        "out_dir",
        "full_config",
        "pose_hist_scales",
        "pose_hist_mu",
        "pose_hist_std",
        "yaw_forward_axis",
        "yaw_forward_axis_offset",
    )

    for trainer_attr, runtime_field in renamed_fields.items():
        setattr(trainer, trainer_attr, getattr(runtime, runtime_field))
    for field_name in direct_fields:
        setattr(trainer, field_name, getattr(runtime, field_name))
    return trainer


def apply_contacts_pretrain_runtime(
    trainer: Any,
    *,
    owner_prefix: str,
    runtime: ContactPretrainRuntime,
) -> Any:
    owner_prefix_txt = str(owner_prefix or "").strip()
    if owner_prefix_txt == "":
        raise ValueError("owner_prefix must be a non-empty string")

    setattr(trainer, "contacts_pretrain_clamp", float(runtime.clamp))
    setattr(trainer, "contacts_pretrain_affine_stats_spec", runtime.affine_stats)
    setattr(trainer, "contacts_pretrain_affine", runtime.affine)
    setattr(trainer, "contacts_pretrain_dropout_injection_mode", str(runtime.dropout_injection_mode))
    setattr(trainer, "contacts_pretrain_dropout_prob", float(runtime.dropout_prob))

    setattr(trainer, f"{owner_prefix_txt}_contacts_pretrain_clamp", float(runtime.clamp))
    setattr(trainer, f"{owner_prefix_txt}_contacts_pretrain_affine_stats_spec", runtime.affine_stats)
    setattr(trainer, f"{owner_prefix_txt}_contacts_pretrain_affine", runtime.affine)
    setattr(trainer, f"{owner_prefix_txt}_contacts_pretrain_dropout_injection_mode", str(runtime.dropout_injection_mode))
    setattr(trainer, f"{owner_prefix_txt}_contacts_pretrain_dropout_prob", float(runtime.dropout_prob))
    setattr(trainer, "contacts_pretrain_runtime_attached", True)
    return trainer


def apply_loss_runtime_from_trainer(
    loss_fn: Any,
    trainer: Any,
    *,
    copy_bundle_meta: bool = False,
    warn_once_fn: Optional[Any] = None,
    warn_key: str = "loss_runtime/bundle_meta",
    warn_message: str = "failed to copy bundle meta onto loss_fn.meta",
) -> Any:
    mu_y = getattr(trainer, "mu_y", None)
    std_y = getattr(trainer, "std_y", None)
    if (mu_y is None) != (std_y is None):
        missing = "std_y" if mu_y is not None else "mu_y"
        present = "mu_y" if mu_y is not None else "std_y"
        raise RuntimeError(
            f"loss runtime has partial normalization stats: {present} is set but {missing} is missing"
        )

    setattr(loss_fn, "mu_y", mu_y)
    setattr(loss_fn, "std_y", std_y)

    if not copy_bundle_meta:
        return loss_fn

    bundle_meta = getattr(trainer, "_bundle_meta", None)
    if not bundle_meta:
        return loss_fn

    try:
        setattr(loss_fn, "meta", dict(bundle_meta))
    except (TypeError, ValueError, RuntimeError) as exc:
        if callable(warn_once_fn):
            warn_once_fn(warn_key, warn_message, exc)
    return loss_fn
