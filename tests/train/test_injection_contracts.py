from __future__ import annotations

from pathlib import Path

import pytest
import torch

from train.validate.injection_contracts import (
    InjectionPayload,
    InjectionRequest,
    InjectionTensorSpec,
    parse_inject_fields,
    validate_injection_request,
    validate_payload_against_slices,
)


def _make_request(tmp_path: Path, *, inject_fields=("rootvel", "rot6d", "angvel"), length=1) -> InjectionRequest:
    src = tmp_path / "target.npz"
    src.write_bytes(b"dummy")
    return InjectionRequest(
        source_npz=src,
        inject_at_step=40,
        inject_from_step=0,
        inject_fields=tuple(inject_fields),  # type: ignore[arg-type]
        inject_label="demo",
        inject_pose_hist_full=True,
        length=length,
    )


def _make_payload(
    *,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    finite: bool = True,
) -> InjectionPayload:
    rot = torch.ones((1, 276), dtype=dtype, device=device)
    root = torch.ones((1, 2), dtype=dtype, device=device)
    ang = torch.ones((1, 138), dtype=dtype, device=device)
    if not finite:
        rot[..., 0] = float("nan")
    return InjectionPayload(
        rot6d_raw=rot,
        rootvel_raw=root,
        angvel_raw=ang,
        source_frame_index=0,
        spec=InjectionTensorSpec(
            rot6d_shape=(1, 276),
            rootvel_shape=(1, 2),
            angvel_shape=(1, 138),
            dtype=dtype,
            device=device,
        ),
    )


def test_parse_inject_fields_dedup_and_order() -> None:
    got = parse_inject_fields("rootvel,rot6d,rootvel,angvel")
    assert got == ("rootvel", "rot6d", "angvel")


def test_parse_inject_fields_empty_fails() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_inject_fields("")


def test_parse_inject_fields_unknown_fails() -> None:
    with pytest.raises(ValueError, match="unsupported field"):
        parse_inject_fields("rootvel,foo")


def test_validate_injection_request_happy_path(tmp_path: Path) -> None:
    req = _make_request(tmp_path)
    validate_injection_request(req)


def test_validate_injection_request_missing_file_fails(tmp_path: Path) -> None:
    req = InjectionRequest(
        source_npz=tmp_path / "missing.npz",
        inject_at_step=0,
        inject_from_step=0,
        inject_fields=("rot6d",),
        inject_label="x",
        inject_pose_hist_full=True,
        length=1,
    )
    with pytest.raises(FileNotFoundError):
        validate_injection_request(req)


def test_validate_injection_request_negative_step_fails(tmp_path: Path) -> None:
    req = _make_request(tmp_path)
    req = InjectionRequest(**{**req.__dict__, "inject_at_step": -1})
    with pytest.raises(ValueError, match="inject_at_step"):
        validate_injection_request(req)


def test_validate_injection_request_length_must_be_one(tmp_path: Path) -> None:
    req = _make_request(tmp_path, length=2)
    with pytest.raises(ValueError, match="length == 1"):
        validate_injection_request(req)


def test_validate_payload_against_slices_happy_path() -> None:
    payload = _make_payload()
    validate_payload_against_slices(
        payload,
        rot6d_x_slice=slice(5, 281),
        angvel_x_slice=slice(281, 419),
        rootvel_x_slice=slice(3, 5),
    )


def test_validate_payload_shape_failfast() -> None:
    payload = _make_payload()
    payload.rot6d_raw = torch.ones((1, 275), dtype=torch.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_payload_against_slices(
            payload,
            rot6d_x_slice=slice(5, 281),
            angvel_x_slice=slice(281, 419),
            rootvel_x_slice=slice(3, 5),
        )


def test_validate_payload_dtype_failfast() -> None:
    payload = _make_payload()
    payload.rot6d_raw = payload.rot6d_raw.to(dtype=torch.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        validate_payload_against_slices(
            payload,
            rot6d_x_slice=slice(5, 281),
            angvel_x_slice=slice(281, 419),
            rootvel_x_slice=slice(3, 5),
        )


def test_validate_payload_device_failfast() -> None:
    payload = _make_payload()
    payload.spec = InjectionTensorSpec(
        rot6d_shape=(1, 276),
        rootvel_shape=(1, 2),
        angvel_shape=(1, 138),
        dtype=torch.float32,
        device=torch.device("meta"),
    )
    with pytest.raises(ValueError, match="device mismatch"):
        validate_payload_against_slices(
            payload,
            rot6d_x_slice=slice(5, 281),
            angvel_x_slice=slice(281, 419),
            rootvel_x_slice=slice(3, 5),
        )


def test_validate_payload_finite_failfast() -> None:
    payload = _make_payload(finite=False)
    with pytest.raises(ValueError, match="non-finite"):
        validate_payload_against_slices(
            payload,
            rot6d_x_slice=slice(5, 281),
            angvel_x_slice=slice(281, 419),
            rootvel_x_slice=slice(3, 5),
        )


def test_validate_payload_requires_non_null_tensor() -> None:
    payload = _make_payload()
    payload.rot6d_raw = None
    payload.rootvel_raw = None
    payload.angvel_raw = None
    with pytest.raises(ValueError, match="at least one non-null"):
        validate_payload_against_slices(
            payload,
            rot6d_x_slice=slice(5, 281),
            angvel_x_slice=slice(281, 419),
            rootvel_x_slice=slice(3, 5),
        )


def test_validate_payload_angvel_requires_slice() -> None:
    payload = _make_payload()
    with pytest.raises(ValueError, match="angvel_x_slice is None"):
        validate_payload_against_slices(
            payload,
            rot6d_x_slice=slice(5, 281),
            angvel_x_slice=None,
            rootvel_x_slice=slice(3, 5),
        )
