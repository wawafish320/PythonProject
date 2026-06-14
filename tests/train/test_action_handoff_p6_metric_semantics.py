from __future__ import annotations

from typing import Any, Dict

from train.validate.run_freerun_cycles import (
    _compute_contact_mismatch_metrics,
    _compute_footslip_pair_metrics,
)


def _contact_step(
    *,
    gt0: float,
    gt1: float,
    meas0: float,
    meas1: float,
    vxy0_cmps: float | None = None,
    vxy1_cmps: float | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ContactGTPerC": [float(gt0), float(gt1)],
        "ContactMeasPerC": [float(meas0), float(meas1)],
    }
    if vxy0_cmps is not None and vxy1_cmps is not None:
        out["ContactMeasWhitebox"] = {"VxyCmpsMean": [float(vxy0_cmps), float(vxy1_cmps)]}
    return out


def test_contact_mismatch_rate_is_binary_threshold_disagreement_rate() -> None:
    step = _contact_step(gt0=0.9, gt1=0.2, meas0=0.1, meas1=0.8)
    rate, frame_or = _compute_contact_mismatch_metrics(step)
    assert rate == 1.0
    assert frame_or == 1.0

    step2 = _contact_step(gt0=0.9, gt1=0.1, meas0=0.8, meas1=0.2)
    rate2, frame_or2 = _compute_contact_mismatch_metrics(step2)
    assert rate2 == 0.0
    assert frame_or2 == 0.0


def test_footslip_uses_whitebox_vxy_cmps_and_dual_frame_gt_contact_gate() -> None:
    # Current contact contract: channel0=right, channel1=left.
    # At step t, only right has dual-frame GT contact (t and t+1 > 0.5).
    step_t = _contact_step(
        gt0=0.9,  # right contact on
        gt1=0.9,  # left contact on at t
        meas0=0.0,
        meas1=0.0,
        vxy0_cmps=25.0,  # right -> 0.25 m/s
        vxy1_cmps=40.0,  # left -> 0.40 m/s (should be gated out by next GT)
    )
    step_next = _contact_step(
        gt0=0.8,  # right stays on
        gt1=0.1,  # left turns off
        meas0=0.0,
        meas1=0.0,
    )
    slip_l, slip_r = _compute_footslip_pair_metrics(contact_step_t=step_t, contact_step_next=step_next)
    assert slip_l is None
    assert slip_r == 0.25


def test_footslip_returns_none_when_no_effective_dual_frame_contact_sample() -> None:
    step_t = _contact_step(
        gt0=0.1,
        gt1=0.2,
        meas0=0.0,
        meas1=0.0,
        vxy0_cmps=10.0,
        vxy1_cmps=20.0,
    )
    step_next = _contact_step(
        gt0=0.2,
        gt1=0.3,
        meas0=0.0,
        meas1=0.0,
    )
    slip_l, slip_r = _compute_footslip_pair_metrics(contact_step_t=step_t, contact_step_next=step_next)
    assert slip_l is None
    assert slip_r is None
