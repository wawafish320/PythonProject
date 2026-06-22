#!/usr/bin/env python3
"""Evaluate provisional P6 smoke acceptance against threshold contract."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_P6_SUMMARY = Path(
    "debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/"
    "p6_synthetic_boundary_eval_summary.json"
)
DEFAULT_CONTRACT = Path(
    "docs/aperiodic_transition/2026-05-26_action_handoff_z_p6_threshold_acceptance_contract.json"
)
REQUIRED_METRICS = (
    "ContactMismatchRate",
    "FootSlipBallL",
    "FootSlipBallR",
    "RootStepDispErr",
    "GeoLocalDeg",
)


def _fatal(msg: str) -> None:
    raise SystemExit(f"[FATAL] {msg}")


def _today_tag() -> str:
    return datetime.now().strftime("%Y%m%d")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        _fatal(f"missing file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fatal(f"failed to parse json {path}: {exc}")


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def _as_finite_float(name: str, v: Any) -> float:
    try:
        f = float(v)
    except Exception:
        _fatal(f"{name} must be numeric, got {v!r}")
    if not math.isfinite(f):
        _fatal(f"{name} must be finite, got {f}")
    return f


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate provisional P6 smoke acceptance.")
    ap.add_argument("--p6-summary", type=Path, default=DEFAULT_P6_SUMMARY)
    ap.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(f"debug_output/_tmp_action_handoff_p6_acceptance_eval_{_today_tag()}"),
    )
    return ap.parse_args()


def _extract_thresholds(contract: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    metrics = contract.get("metrics")
    if not isinstance(metrics, dict):
        _fatal("contract.metrics must be object")
    out: Dict[str, Dict[str, Any]] = {}
    for m in REQUIRED_METRICS:
        spec = metrics.get(m)
        if not isinstance(spec, dict):
            _fatal(f"contract.metrics missing {m}")
        threshold = spec.get("threshold")
        if threshold is None:
            _fatal(f"threshold missing for {m}")
        out[m] = {
            "threshold": _as_finite_float(f"threshold[{m}]", threshold),
            "tier": int(spec.get("tier", 99)),
            "unit": str(spec.get("unit", "")),
            "direction": str(spec.get("direction", "")),
        }
    return out


def _precondition_checks(summary: Dict[str, Any], contract: Dict[str, Any], rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str | None]:
    checks: Dict[str, Any] = {
        "rows_non_empty": len(rows) > 0,
        "total_rows": len(rows),
        "rows_executed_expected_8": len(rows) == 8,
        "canonical_metric_completeness_complete": True,
        "no_proxy_metrics": True,
        "row_binding_audit_passed": True,
        "stress_differentiability_observed": False,
        "rootvel_rot6d_injection_applied": True,
        "expected_warning_policy": {
            "angvel_target_slice_missing_expected": _as_bool(
                ((contract.get("expected_warnings") or {}).get("angvel_target_slice_missing", False))
            )
        },
    }

    if len(rows) == 0:
        return checks, "blocked_metric_incomplete"

    stress = summary.get("stress_differentiability_audit")
    checks["stress_differentiability_observed"] = isinstance(stress, dict) and str(stress.get("status")) == "differentiable_trace_observed"

    for row in rows:
        metrics = row.get("p6_safety_metrics")
        if not isinstance(metrics, dict):
            checks["canonical_metric_completeness_complete"] = False
            checks["no_proxy_metrics"] = False
            continue

        if not _as_bool(metrics.get("canonical_metric_complete", False)):
            checks["canonical_metric_completeness_complete"] = False
        if _as_bool(metrics.get("proxy_metric_used", False)):
            checks["no_proxy_metrics"] = False

        for m in REQUIRED_METRICS:
            if metrics.get(m) is None:
                checks["canonical_metric_completeness_complete"] = False

        bind_audit = metrics.get("row_binding_audit")
        if not isinstance(bind_audit, dict):
            checks["row_binding_audit_passed"] = False
        else:
            if not _as_bool(bind_audit.get("injection_apply_record_matches_inject_at_step", False)):
                checks["row_binding_audit_passed"] = False

        field_apply = metrics.get("injection_field_apply")
        if not isinstance(field_apply, dict):
            checks["rootvel_rot6d_injection_applied"] = False
            continue

        rootvel = field_apply.get("rootvel")
        rot6d = field_apply.get("rot6d")
        angvel = field_apply.get("angvel")
        if not (isinstance(rootvel, dict) and isinstance(rot6d, dict) and isinstance(angvel, dict)):
            checks["rootvel_rot6d_injection_applied"] = False
            continue

        if not _as_bool(rootvel.get("applied", False)) or not _as_bool(rot6d.get("applied", False)):
            checks["rootvel_rot6d_injection_applied"] = False

        if _as_bool(checks["expected_warning_policy"]["angvel_target_slice_missing_expected"]):
            pass
        else:
            if _as_bool(angvel.get("requested", False)) and not _as_bool(angvel.get("applied", False)):
                checks["rootvel_rot6d_injection_applied"] = False

    blocked_reason = None
    if not checks["rows_non_empty"]:
        blocked_reason = "blocked_metric_incomplete"
    elif not checks["canonical_metric_completeness_complete"]:
        blocked_reason = "blocked_metric_incomplete"
    elif not checks["no_proxy_metrics"]:
        blocked_reason = "blocked_metric_incomplete"
    elif not checks["row_binding_audit_passed"]:
        blocked_reason = "blocked_binding_mismatch"
    elif not checks["rootvel_rot6d_injection_applied"]:
        blocked_reason = "blocked_injection_not_applied"

    return checks, blocked_reason


def _eval_row(
    row: Dict[str, Any],
    thresholds: Dict[str, Dict[str, Any]],
    expected_angvel_warning: bool,
    precondition_blocked_reason: str | None,
) -> Dict[str, Any]:
    trial_id = str(row.get("trial_id", ""))
    case_type = str(row.get("case_type", ""))
    metrics = row.get("p6_safety_metrics")
    out: Dict[str, Any] = {
        "trial_id": trial_id,
        "case_type": case_type,
        "tier_results": {},
        "metric_results": {},
        "warnings": [],
        "row_status": None,
    }

    if precondition_blocked_reason is not None:
        out["row_status"] = precondition_blocked_reason
        return out

    if not isinstance(metrics, dict):
        out["row_status"] = "blocked_metric_incomplete"
        return out

    if not _as_bool(metrics.get("canonical_metric_complete", False)) or _as_bool(metrics.get("proxy_metric_used", False)):
        out["row_status"] = "blocked_metric_incomplete"
        return out

    field_apply = metrics.get("injection_field_apply")
    if not isinstance(field_apply, dict):
        out["row_status"] = "blocked_injection_not_applied"
        return out

    rv = field_apply.get("rootvel", {}) if isinstance(field_apply.get("rootvel"), dict) else {}
    r6 = field_apply.get("rot6d", {}) if isinstance(field_apply.get("rot6d"), dict) else {}
    av = field_apply.get("angvel", {}) if isinstance(field_apply.get("angvel"), dict) else {}

    if not (_as_bool(rv.get("applied", False)) and _as_bool(r6.get("applied", False))):
        out["row_status"] = "blocked_injection_not_applied"
        return out

    if _as_bool(av.get("requested", False)) and not _as_bool(av.get("applied", False)):
        if expected_angvel_warning and str(av.get("reason")) == "target_slice_missing":
            out["warnings"].append("angvel_target_slice_missing_expected_under_contract")
        else:
            out["row_status"] = "blocked_injection_not_applied"
            return out

    failed_tiers: Dict[int, List[str]] = {}
    for m in REQUIRED_METRICS:
        value = metrics.get(m)
        if value is None:
            out["row_status"] = "blocked_metric_incomplete"
            return out
        fv = _as_finite_float(f"{trial_id}:{m}", value)
        spec = thresholds[m]
        th = float(spec["threshold"])
        passed = fv <= th
        tier = int(spec["tier"])
        out["metric_results"][m] = {
            "value": fv,
            "threshold": th,
            "pass": passed,
            "tier": tier,
        }
        if not passed:
            failed_tiers.setdefault(tier, []).append(m)

    tier_results: Dict[str, Any] = {}
    for tier in (1, 2, 3):
        failed = failed_tiers.get(tier, [])
        tier_results[f"tier_{tier}"] = {
            "pass": len(failed) == 0,
            "failed_metrics": failed,
        }
    out["tier_results"] = tier_results

    is_all_pass = all(v["pass"] for v in tier_results.values())
    if case_type == "normal":
        out["row_status"] = "normal_accept" if is_all_pass else "normal_fail"
    elif case_type == "weak_stress":
        out["row_status"] = "weak_pass" if is_all_pass else "weak_fallback_required_known_risk"
    else:
        out["row_status"] = "blocked_metric_incomplete"

    return out


def _build_overall(
    row_results: List[Dict[str, Any]],
    preconditions: Dict[str, Any],
    precondition_blocked_reason: str | None,
    threshold_derivation: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    normal = [r for r in row_results if r.get("case_type") == "normal"]
    weak = [r for r in row_results if r.get("case_type") == "weak_stress"]

    normal_accept = sum(1 for r in normal if r.get("row_status") == "normal_accept")
    normal_fail = sum(1 for r in normal if r.get("row_status") == "normal_fail")

    weak_pass = sum(1 for r in weak if r.get("row_status") == "weak_pass")
    weak_fallback = sum(1 for r in weak if r.get("row_status") == "weak_fallback_required_known_risk")

    tier_summary = {
        "tier_1_failed_rows": sum(
            1
            for r in row_results
            if isinstance(r.get("tier_results"), dict)
            and not bool((r["tier_results"].get("tier_1") or {}).get("pass", True))
        ),
        "tier_2_failed_rows": sum(
            1
            for r in row_results
            if isinstance(r.get("tier_results"), dict)
            and not bool((r["tier_results"].get("tier_2") or {}).get("pass", True))
        ),
        "tier_3_failed_rows": sum(
            1
            for r in row_results
            if isinstance(r.get("tier_results"), dict)
            and not bool((r["tier_results"].get("tier_3") or {}).get("pass", True))
        ),
    }

    normal_summary = {
        "rows": len(normal),
        "normal_accept": normal_accept,
        "normal_fail": normal_fail,
    }
    weak_summary = {
        "rows": len(weak),
        "weak_pass": weak_pass,
        "weak_fallback_required_known_risk": weak_fallback,
    }

    if precondition_blocked_reason is not None:
        overall_status = precondition_blocked_reason
    elif "calibration-on-current-smoke" not in str(threshold_derivation) and "signed" not in str(threshold_derivation):
        overall_status = "inconclusive_threshold_missing"
    else:
        normal_tier12_fail = False
        for r in normal:
            tr = r.get("tier_results") or {}
            if not bool((tr.get("tier_1") or {}).get("pass", True)) or not bool((tr.get("tier_2") or {}).get("pass", True)):
                normal_tier12_fail = True
                break

        if normal_tier12_fail:
            overall_status = "p6_smoke_failed_normal_safety"
        elif normal_fail > 0:
            overall_status = "p6_smoke_failed_normal_safety"
        elif weak_fallback > 0:
            overall_status = "p6_smoke_accept_with_known_weak_fallback_required"
        else:
            overall_status = "p6_smoke_accept_all_rows_provisional"

    return overall_status, normal_summary, weak_summary, tier_summary


def _write_md(
    out_path: Path,
    scope: str,
    contract: Dict[str, Any],
    preconditions: Dict[str, Any],
    row_results: List[Dict[str, Any]],
    normal_summary: Dict[str, Any],
    weak_summary: Dict[str, Any],
    tier_summary: Dict[str, Any],
    overall_status: str,
) -> None:
    metrics = contract.get("metrics", {})

    lines: List[str] = []
    lines.append("# P6 Acceptance Evaluation Summary")
    lines.append("")
    lines.append(f"- scope: `{scope}`")
    lines.append(f"- contract_version: `{contract.get('contract_version')}`")
    lines.append(f"- source_artifact: `{contract.get('source_artifact')}`")
    lines.append("")
    lines.append("## Threshold Table")
    lines.append("")
    lines.append("| Metric | Tier | Unit | Direction | Threshold |")
    lines.append("|---|---:|---|---|---:|")
    for m in REQUIRED_METRICS:
        spec = metrics.get(m, {})
        lines.append(
            f"| {m} | {spec.get('tier')} | {spec.get('unit')} | {spec.get('direction')} | {float(spec.get('threshold')):.12g} |"
        )

    lines.append("")
    lines.append("## Per-row Table")
    lines.append("")
    lines.append(
        "| trial_id | case_type | ContactMismatchRate | FootSlipBallL | FootSlipBallR | RootStepDispErr | GeoLocalDeg | tier1 | tier2 | tier3 | row_status |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---|---|---|")
    for r in row_results:
        mm = r.get("metric_results", {})
        t = r.get("tier_results", {})

        def _mv(name: str) -> str:
            d = mm.get(name)
            if not isinstance(d, dict):
                return "NA"
            return f"{float(d.get('value')):.12g}"

        def _tp(tier: str) -> str:
            d = t.get(tier)
            if not isinstance(d, dict):
                return "NA"
            return "pass" if bool(d.get("pass")) else "fail"

        lines.append(
            f"| {r.get('trial_id')} | {r.get('case_type')} | {_mv('ContactMismatchRate')} | {_mv('FootSlipBallL')} | {_mv('FootSlipBallR')} | {_mv('RootStepDispErr')} | {_mv('GeoLocalDeg')} | {_tp('tier_1')} | {_tp('tier_2')} | {_tp('tier_3')} | {r.get('row_status')} |"
        )

    lines.append("")
    lines.append("## Normal vs Weak Classification")
    lines.append("")
    lines.append(f"- normal_summary: `{json.dumps(normal_summary, ensure_ascii=False)}`")
    lines.append(f"- weak_stress_summary: `{json.dumps(weak_summary, ensure_ascii=False)}`")
    lines.append(f"- tier_summary: `{json.dumps(tier_summary, ensure_ascii=False)}`")

    lines.append("")
    lines.append("## Preconditions")
    lines.append("")
    lines.append(f"- precondition_checks: `{json.dumps(preconditions, ensure_ascii=False)}`")

    lines.append("")
    lines.append("## Overall Verdict")
    lines.append("")
    lines.append(f"- overall_status: `{overall_status}`")
    lines.append("- This is provisional smoke acceptance evaluation; production P6 pass is not established.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    summary_path = args.p6_summary.resolve()
    contract_path = args.contract.resolve()

    summary = _load_json(summary_path)
    contract = _load_json(contract_path)

    rows = summary.get("rows")
    if not isinstance(rows, list):
        _fatal("p6 summary missing rows list")

    thresholds = _extract_thresholds(contract)
    preconditions, blocked_reason = _precondition_checks(summary, contract, rows)

    expected_warning = _as_bool(((contract.get("expected_warnings") or {}).get("angvel_target_slice_missing", False)))

    row_results = [_eval_row(r, thresholds, expected_warning, blocked_reason) for r in rows]

    overall_status, normal_summary, weak_summary, tier_summary = _build_overall(
        row_results,
        preconditions,
        blocked_reason,
        str(contract.get("threshold_derivation", "")),
    )

    caveats = [
        "Thresholds are provisional smoke thresholds (calibration-on-current-smoke), not production thresholds.",
        "This evaluation does not establish production P6 pass.",
    ]

    out = {
        "scope": str(contract.get("scope", "provisional_smoke_acceptance")),
        "inputs": {
            "p6_summary": str(summary_path),
            "contract": str(contract_path),
        },
        "contract": {
            "contract_version": contract.get("contract_version"),
            "scope": contract.get("scope"),
            "source_artifact": contract.get("source_artifact"),
            "threshold_derivation": contract.get("threshold_derivation"),
            "metrics": contract.get("metrics"),
            "expected_warnings": contract.get("expected_warnings"),
            "row_class_policy": contract.get("row_class_policy"),
        },
        "precondition_checks": preconditions,
        "row_results": row_results,
        "normal_summary": normal_summary,
        "weak_stress_summary": weak_summary,
        "tier_summary": tier_summary,
        "overall_status": overall_status,
        "caveats": caveats,
        "production_pass_established": False,
    }

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "p6_acceptance_eval_summary.json"
    md_path = out_dir / "p6_acceptance_eval_summary.md"

    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_md(
        md_path,
        str(contract.get("scope", "provisional_smoke_acceptance")),
        contract,
        preconditions,
        row_results,
        normal_summary,
        weak_summary,
        tier_summary,
        overall_status,
    )

    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")


if __name__ == "__main__":
    main()
