from __future__ import annotations

import json
from pathlib import Path

from tools.run_action_handoff_p6_synthetic_boundary_eval import _extract_safety_metrics_from_runner_output


def _write_payload(tmp_path: Path, per_step: list[dict]) -> Path:
    payload = {
        "metrics_per_step": per_step,
        "injection_apply_records": [
            {
                "step": 40,
                "step_in_cycle": 40,
                "fields_applied": [
                    {"field": "rootvel", "requested": True, "applied": True, "reason": "applied"},
                    {"field": "rot6d", "requested": True, "applied": True, "reason": "applied"},
                    {"field": "angvel", "requested": True, "applied": False, "reason": "target_slice_missing"},
                ],
            }
        ],
    }
    p = tmp_path / "Walk_F_freerun_cycles.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_extractor_uses_canonical_contact_mismatch_not_proxy(tmp_path: Path) -> None:
    per_step = [
        {
            "ContactMismatchRate": 0.25,
            "ContactErrAbsMean": 0.75,  # different value; extractor must pick canonical
            "FootSlipBallL": 0.10,
            "FootSlipBallR": 0.20,
            "RootStepDispErr": 0.03,
            "GeoLocalDeg": 5.0,
        },
        {
            "ContactMismatchRate": 0.75,
            "ContactErrAbsMean": 0.25,
            "FootSlipBallL": 0.30,
            "FootSlipBallR": 0.40,
            "RootStepDispErr": 0.07,
            "GeoLocalDeg": 7.0,
        },
    ]
    out = _extract_safety_metrics_from_runner_output(_write_payload(tmp_path, per_step))
    assert out["metric_source_used"]["ContactMismatchRate"] == "ContactMismatchRate"
    assert out["proxy_metric_used"] is False
    assert out["ContactMismatchRate"] == 0.5


def test_extractor_footslip_none_when_no_effective_samples(tmp_path: Path) -> None:
    # Only zeros or nulls should not be treated as valid canonical slip coverage.
    per_step = [
        {
            "ContactMismatchRate": 0.0,
            "FootSlipBallL": 0.0,
            "FootSlipBallR": None,
            "RootStepDispErr": 0.02,
            "GeoLocalDeg": 4.0,
        },
        {
            "ContactMismatchRate": 0.0,
            "FootSlipBallL": 0.0,
            "FootSlipBallR": 0.0,
            "RootStepDispErr": 0.03,
            "GeoLocalDeg": 6.0,
        },
    ]
    out = _extract_safety_metrics_from_runner_output(_write_payload(tmp_path, per_step))
    assert out["FootSlipBallL"] is None
    assert out["FootSlipBallR"] is None
    assert "FootSlipBallL" in out["canonical_metric_missing"]
    assert "FootSlipBallR" in out["canonical_metric_missing"]
