from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch

InjectField = Literal["rootvel", "rot6d", "angvel"]
_ALLOWED_FIELDS: tuple[InjectField, ...] = ("rootvel", "rot6d", "angvel")


@dataclass(frozen=True)
class InjectionRequest:
    source_npz: Path
    inject_at_step: int
    inject_from_step: int
    inject_fields: tuple[InjectField, ...]
    inject_label: str
    inject_pose_hist_full: bool
    length: int


@dataclass(frozen=True)
class InjectionTensorSpec:
    rot6d_shape: tuple[int, int]
    rootvel_shape: tuple[int, int]
    angvel_shape: tuple[int, int]
    dtype: torch.dtype
    device: torch.device


@dataclass
class InjectionPayload:
    rot6d_raw: Optional[torch.Tensor]
    rootvel_raw: Optional[torch.Tensor]
    angvel_raw: Optional[torch.Tensor]
    source_frame_index: int
    spec: InjectionTensorSpec


@dataclass
class InjectionApplyRecord:
    step: int
    step_in_cycle: int
    fields_applied: list[dict]
    pose_hist_tail_rewrite: Optional[dict]


def _slice_width(value: slice, *, name: str) -> int:
    if not isinstance(value, slice) or value.start is None or value.stop is None:
        raise ValueError(f"{name} must be a concrete slice")
    width = int(value.stop - value.start)
    if width <= 0:
        raise ValueError(f"{name} must have positive width")
    return width


def parse_inject_fields(raw: str) -> tuple[InjectField, ...]:
    tokens = [t.strip().lower() for t in str(raw).split(",") if t.strip()]
    if not tokens:
        raise ValueError("inject_fields cannot be empty")
    out: list[InjectField] = []
    seen = set()
    for token in tokens:
        if token not in _ALLOWED_FIELDS:
            raise ValueError(
                f"inject_fields contains unsupported field {token!r}; allowed={','.join(_ALLOWED_FIELDS)}"
            )
        if token not in seen:
            out.append(token)  # type: ignore[arg-type]
            seen.add(token)
    return tuple(out)


def validate_injection_request(request: InjectionRequest) -> None:
    source_npz = Path(request.source_npz)
    if not source_npz.is_file():
        raise FileNotFoundError(f"source_npz does not exist: {source_npz}")
    if int(request.inject_at_step) < 0:
        raise ValueError("inject_at_step must be >= 0")
    if int(request.inject_from_step) < 0:
        raise ValueError("inject_from_step must be >= 0")
    if int(request.length) != 1:
        raise ValueError("Phase-1 contract requires length == 1")
    if not isinstance(request.inject_pose_hist_full, bool):
        raise ValueError("inject_pose_hist_full must be bool")
    if str(request.inject_label).strip() == "":
        raise ValueError("inject_label cannot be empty")
    if not request.inject_fields:
        raise ValueError("inject_fields cannot be empty")
    for field in request.inject_fields:
        if field not in _ALLOWED_FIELDS:
            raise ValueError(f"inject_fields contains unsupported field {field!r}")


def _validate_tensor(
    name: str,
    value: torch.Tensor,
    *,
    expected_shape: tuple[int, int],
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be torch.Tensor")
    if tuple(value.shape) != tuple(expected_shape):
        raise ValueError(f"{name} shape mismatch: got={tuple(value.shape)} expected={tuple(expected_shape)}")
    if value.dtype != expected_dtype:
        raise ValueError(f"{name} dtype mismatch: got={value.dtype} expected={expected_dtype}")
    if value.device != expected_device:
        raise ValueError(f"{name} device mismatch: got={value.device} expected={expected_device}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def validate_payload_against_slices(
    payload: InjectionPayload,
    *,
    rot6d_x_slice: slice,
    angvel_x_slice: Optional[slice],
    rootvel_x_slice: slice,
) -> None:
    if int(payload.source_frame_index) < 0:
        raise ValueError("source_frame_index must be >= 0")
    rot6d_width = _slice_width(rot6d_x_slice, name="rot6d_x_slice")
    rootvel_width = _slice_width(rootvel_x_slice, name="rootvel_x_slice")
    if payload.spec.rot6d_shape != (1, rot6d_width):
        raise ValueError(
            f"spec.rot6d_shape mismatch: got={payload.spec.rot6d_shape} expected={(1, rot6d_width)}"
        )
    if payload.spec.rootvel_shape != (1, rootvel_width):
        raise ValueError(
            f"spec.rootvel_shape mismatch: got={payload.spec.rootvel_shape} expected={(1, rootvel_width)}"
        )
    if angvel_x_slice is not None:
        angvel_width = _slice_width(angvel_x_slice, name="angvel_x_slice")
        if payload.spec.angvel_shape != (1, angvel_width):
            raise ValueError(
                f"spec.angvel_shape mismatch: got={payload.spec.angvel_shape} expected={(1, angvel_width)}"
            )

    non_null_n = 0
    if payload.rot6d_raw is not None:
        non_null_n += 1
        _validate_tensor(
            "rot6d_raw",
            payload.rot6d_raw,
            expected_shape=payload.spec.rot6d_shape,
            expected_dtype=payload.spec.dtype,
            expected_device=payload.spec.device,
        )
    if payload.rootvel_raw is not None:
        non_null_n += 1
        _validate_tensor(
            "rootvel_raw",
            payload.rootvel_raw,
            expected_shape=payload.spec.rootvel_shape,
            expected_dtype=payload.spec.dtype,
            expected_device=payload.spec.device,
        )
    if payload.angvel_raw is not None:
        if angvel_x_slice is None:
            raise ValueError("angvel_raw provided but angvel_x_slice is None")
        non_null_n += 1
        _validate_tensor(
            "angvel_raw",
            payload.angvel_raw,
            expected_shape=payload.spec.angvel_shape,
            expected_dtype=payload.spec.dtype,
            expected_device=payload.spec.device,
        )
    if non_null_n <= 0:
        raise ValueError("InjectionPayload must provide at least one non-null tensor")
