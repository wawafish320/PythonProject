from __future__ import annotations

import pytest

from train.validate.injection_windows import WindowSpec, compute_window_bounds, summarize_window_metrics


def test_compute_window_bounds_happy_path() -> None:
    spec = WindowSpec(entry_window_pre_k=8, entry_window_post_k=8, recovery_window_k=16)
    out = compute_window_bounds(inject_at_step=40, total_steps=120, spec=spec)
    entry = out["entry_window"]
    rec = out["post_inject_recovery"]
    assert entry["t_start"] == 32
    assert entry["t_end"] == 48
    assert entry["window_steps"] == 17
    assert rec["t_start"] == 40
    assert rec["t_end"] == 56
    assert rec["window_steps"] == 17


def test_compute_window_bounds_clamp_left_and_right() -> None:
    spec = WindowSpec(entry_window_pre_k=8, entry_window_post_k=8, recovery_window_k=16)
    left = compute_window_bounds(inject_at_step=2, total_steps=10, spec=spec)
    assert left["entry_window"]["t_start"] == 0
    assert left["entry_window"]["t_end"] == 9
    right = compute_window_bounds(inject_at_step=9, total_steps=10, spec=spec)
    assert right["entry_window"]["t_start"] == 1
    assert right["entry_window"]["t_end"] == 9
    assert right["post_inject_recovery"]["t_end"] == 9


def test_compute_window_bounds_invalid_total_steps_fails() -> None:
    spec = WindowSpec(entry_window_pre_k=1, entry_window_post_k=1, recovery_window_k=1)
    with pytest.raises(ValueError, match="total_steps"):
        compute_window_bounds(inject_at_step=0, total_steps=0, spec=spec)


def test_compute_window_bounds_invalid_inject_step_fails() -> None:
    spec = WindowSpec(entry_window_pre_k=1, entry_window_post_k=1, recovery_window_k=1)
    with pytest.raises(ValueError, match="out of range"):
        compute_window_bounds(inject_at_step=10, total_steps=10, spec=spec)


def test_summarize_window_metrics_happy_path() -> None:
    spec = WindowSpec(entry_window_pre_k=1, entry_window_post_k=1, recovery_window_k=2)
    bounds = compute_window_bounds(inject_at_step=2, total_steps=6, spec=spec)
    per_step = [
        {"GeoLocalDeg": 0.1, "RootStepDispErr": 0.01},
        {"GeoLocalDeg": 0.2, "RootStepDispErr": 0.02},
        {"GeoLocalDeg": 0.3, "RootStepDispErr": 0.03},
        {"GeoLocalDeg": 0.4, "RootStepDispErr": 0.04},
        {"GeoLocalDeg": 0.5, "RootStepDispErr": 0.05},
        {"GeoLocalDeg": 0.6, "RootStepDispErr": 0.06},
    ]
    out = summarize_window_metrics(
        per_step_metrics=per_step,
        bounds=bounds,
        required_metrics=["GeoLocalDeg", "RootStepDispErr"],
    )
    entry = out["entry_window"]["metric_summary"]["GeoLocalDeg"]
    rec = out["post_inject_recovery"]["metric_summary"]["RootStepDispErr"]
    assert entry["n"] == 3
    assert pytest.approx(entry["mean"], rel=0.0, abs=1e-9) == 0.3
    assert rec["n"] == 3
    assert pytest.approx(rec["mean"], rel=0.0, abs=1e-9) == 0.04
    assert rec["peak_step_rel"] == 2


def test_summarize_window_metrics_missing_metric_has_null_summary() -> None:
    spec = WindowSpec(entry_window_pre_k=0, entry_window_post_k=0, recovery_window_k=0)
    bounds = compute_window_bounds(inject_at_step=0, total_steps=1, spec=spec)
    out = summarize_window_metrics(
        per_step_metrics=[{"GeoLocalDeg": 0.1}],
        bounds=bounds,
        required_metrics=["GeoLocalDeg", "ContactMismatchRate"],
    )
    miss = out["entry_window"]["metric_summary"]["ContactMismatchRate"]
    assert miss["n"] == 0
    assert miss["mean"] is None
    assert miss["peak_step_rel"] is None
