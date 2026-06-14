#!/usr/bin/env python3
"""Generate P6 retrieval-metadata/fallback dry-run artifacts from existing P4-alt outputs.

This tool is planning-only:
- does NOT run rollout
- does NOT modify train/validate/run_freerun_cycles.py
- does NOT touch training entrypoints

It emits schema examples for strong + weak stress pairs so we can validate
P6 metadata/fallback contract before evaluator integration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_SWEEP_ROOT = Path("debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524")
DEFAULT_WEAK_JSON = Path(
    "debug_output/_tmp_action_handoff_z_probe_v1_walk_l_to_r_failure_analysis_20260524/walk_l_to_r_failure_analysis.json"
)
DEFAULT_FEATURE_JSON = Path("debug_output/_tmp_action_handoff_z_probe_v1_20260524/p4_cross_clip_entry.json")


@dataclass(frozen=True)
class PairSpec:
    pair_bucket: str
    source_clip: str
    target_clip: str
    role: str

    @property
    def pair_key(self) -> str:
        return f"{self.source_clip}->{self.target_clip}"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict JSON at {path}, got {type(obj).__name__}")
    return obj


def _select_config(sweep: Dict[str, Any], config_id: str) -> Dict[str, Any]:
    configs = sweep.get("configs", [])
    if not isinstance(configs, list):
        raise TypeError("sweep.configs must be a list")
    for cfg in configs:
        if isinstance(cfg, dict) and str(cfg.get("config_id")) == config_id:
            return cfg
    raise KeyError(f"config_id not found: {config_id}")


def _extract_repr_contract(feature_meta: Dict[str, Any]) -> Dict[str, Any]:
    per_clip = feature_meta.get("per_clip", {})
    if not isinstance(per_clip, dict) or not per_clip:
        return {
            "repr_name": "z_bottleneck",
            "repr_dim": None,
            "dtype": None,
            "device": None,
        }
    first_key = sorted(per_clip.keys())[0]
    first = per_clip[first_key] if isinstance(per_clip[first_key], dict) else {}
    z_meta = first.get("z", {}) if isinstance(first, dict) else {}
    shape = z_meta.get("shape", []) if isinstance(z_meta, dict) else []
    repr_dim = None
    if isinstance(shape, list) and len(shape) >= 2:
        repr_dim = int(shape[-1])
    return {
        "repr_name": "z_bottleneck",
        "repr_dim": repr_dim,
        "dtype": z_meta.get("dtype") if isinstance(z_meta, dict) else None,
        "device": z_meta.get("device") if isinstance(z_meta, dict) else None,
    }


def _collect_pair_metrics(per_pair: Dict[str, Any], pair_key: str) -> Dict[str, Any]:
    stats = per_pair.get(pair_key)
    if not isinstance(stats, dict):
        raise KeyError(f"pair not found in per_pair: {pair_key}")
    return stats


def _fallback_from_metrics(*, ratio: float, hit_lift: float, spearman: float, horizon_n: int, weak_pair: bool) -> Dict[str, Any]:
    # Dry-run policy placeholder for schema validation only.
    # Chosen thresholds are explicit and artifact-local, not final runtime policy.
    max_ratio = 0.95
    min_hit_lift = 0.0
    min_spearman = 0.20
    min_margin = 0.05

    reasons: List[str] = []
    status = "selected"

    if ratio >= max_ratio:
        reasons.append("z_distance_too_large")
    if hit_lift <= min_hit_lift:
        reasons.append("future_equiv_below_floor")
    if spearman <= min_spearman:
        reasons.append("insufficient_future_equiv_signal")
    if weak_pair and horizon_n >= 24:
        reasons.append("stress_pair_policy")

    if reasons:
        status = "fallback"

    long_warn = bool(horizon_n >= 24 and weak_pair)
    warn_reason = "long_horizon_degradation_risk" if long_warn else None

    return {
        "retrieval_status": status,
        "fallback_triggered": bool(status != "selected"),
        "fallback_reason": reasons[0] if reasons else None,
        "fallback_reasons_all": reasons,
        "no_good_candidate": False,
        "long_horizon_warning": long_warn,
        "warning_reason": warn_reason,
        "thresholds": {
            "max_z_distance_ratio": max_ratio,
            "min_hit_lift": min_hit_lift,
            "min_spearman": min_spearman,
            "min_margin_top1_top2": min_margin,
            "long_horizon_N_gte": 24,
        },
    }


def _build_rows(
    *,
    sweep: Dict[str, Any],
    weak: Dict[str, Any],
    feature_meta: Dict[str, Any],
    configs: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pair_specs = [
        PairSpec("strong", "Walk_F", "Walk_R_To_L", "normal_case"),
        PairSpec("weak_stress", "Walk_L_To_R", "Walk_R_To_L", "known_weak_pair"),
        PairSpec("weak_stress", "Walk_L_To_R", "Walk_R_To_R", "known_weak_pair"),
    ]

    repr_contract = _extract_repr_contract(feature_meta)
    rows: List[Dict[str, Any]] = []

    weak_flags = weak.get("flags", {}) if isinstance(weak.get("flags"), dict) else {}

    for cid in configs:
        cfg = _select_config(sweep, cid)
        horizon_n = int(cfg.get("future_horizon_n"))
        q = float(cfg.get("oracle_top_q"))
        top_k = int(cfg.get("top_k"))
        summary_json = str(cfg.get("summary_json"))

        summary = _load_json(Path(summary_json))
        per_pair = summary.get("per_pair", {})
        if not isinstance(per_pair, dict):
            raise TypeError(f"per_pair must be dict in {summary_json}")

        for spec in pair_specs:
            metrics = _collect_pair_metrics(per_pair, spec.pair_key)
            ratio = float(metrics.get("top1_future_distance_vs_random_ratio"))
            hit_lift = float(metrics.get("top1_equiv_hit_rate_vs_random_top1"))
            spearman = float(metrics.get("mean_spearman_zdist_vs_futuredist"))
            top1_hit = float(metrics.get("top1_equiv_hit_rate"))
            random_top1 = float(metrics.get("random_top1_expectation"))
            margin = float(random_top1 - top1_hit)

            weak_pair = spec.pair_bucket == "weak_stress"
            fallback = _fallback_from_metrics(
                ratio=ratio,
                hit_lift=hit_lift,
                spearman=spearman,
                horizon_n=horizon_n,
                weak_pair=weak_pair,
            )

            row = {
                "trial_id": f"{cid}:{spec.pair_key}",
                "pair_bucket": spec.pair_bucket,
                "pair_role": spec.role,
                "source_target_pair": spec.pair_key,
                "p6_retrieval_metadata": {
                    "enabled": True,
                    "source_clip": spec.source_clip,
                    "source_frame": None,
                    "target_clip": spec.target_clip,
                    "selected_target_frame": None,
                    "horizon_N": horizon_n,
                    "z_distance": ratio,
                    "z_rank_topk": 1,
                    "z_margin_top1_top2": margin,
                    "future_equiv_score": hit_lift,
                    "future_equiv_score_available": True,
                    "future_equiv_quantile_q": q,
                    "top_k": top_k,
                    "mean_spearman_zdist_vs_futuredist": spearman,
                    "num_queries": int(metrics.get("num_queries", 0)),
                    "z_feature_contract": repr_contract,
                    "value_semantics": {
                        "z_distance": "top1_future_distance_vs_random_ratio proxy from P4-alt per-pair summary",
                        "future_equiv_score": "top1_equiv_hit_rate_vs_random_top1",
                        "z_margin_top1_top2": "random_top1_expectation - top1_equiv_hit_rate (proxy margin)",
                        "frame_indices_unavailable": True,
                    },
                },
                "p6_fallback": fallback,
                "provenance": {
                    "config_id": cid,
                    "config_summary_json": summary_json,
                    "weak_flags_snapshot": weak_flags,
                    "artifact_mode": "dry_run_from_existing_p4_artifacts",
                },
            }
            rows.append(row)

    matrix_rows: List[Dict[str, Any]] = []
    for r in rows:
        matrix_rows.append(
            {
                "trial_id": r["trial_id"],
                "pair_bucket": r["pair_bucket"],
                "pair": r["source_target_pair"],
                "horizon_N": r["p6_retrieval_metadata"]["horizon_N"],
                "retrieval_status": r["p6_fallback"]["retrieval_status"],
                "long_horizon_warning": r["p6_fallback"]["long_horizon_warning"],
            }
        )

    return rows, matrix_rows


def _build_summary(rows: List[Dict[str, Any]], matrix_rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    total = len(rows)
    fallback_n = sum(1 for r in rows if r["p6_fallback"]["fallback_triggered"])
    long_warn_n = sum(1 for r in rows if r["p6_fallback"]["long_horizon_warning"])

    by_bucket: Dict[str, Dict[str, int]] = {}
    for r in rows:
        b = str(r["pair_bucket"])
        rec = by_bucket.setdefault(b, {"total": 0, "fallback": 0, "long_warn": 0})
        rec["total"] += 1
        rec["fallback"] += int(bool(r["p6_fallback"]["fallback_triggered"]))
        rec["long_warn"] += int(bool(r["p6_fallback"]["long_horizon_warning"]))

    return {
        "tool": "run_action_handoff_p6_metadata_dryrun",
        "status": "planning_eval_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "sweep_root": str(args.sweep_root),
            "weak_json": str(args.weak_json),
            "feature_json": str(args.feature_json),
            "configs": list(args.configs),
        },
        "counts": {
            "total_rows": total,
            "fallback_rows": fallback_n,
            "long_horizon_warning_rows": long_warn_n,
        },
        "by_pair_bucket": by_bucket,
        "trial_matrix": matrix_rows,
        "notes": [
            "Dry-run only: no rollout execution, no evaluator wiring, no train entry changes.",
            "Pair-level P4-alt aggregates are used as retrieval/future-equivalence proxies.",
            "Frame-level selected_target_frame/source_frame are unavailable in current artifacts and set to null.",
        ],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate P6 metadata/fallback dry-run examples from existing P4-alt artifacts."
    )
    ap.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    ap.add_argument("--weak-json", type=Path, default=DEFAULT_WEAK_JSON)
    ap.add_argument("--feature-json", type=Path, default=DEFAULT_FEATURE_JSON)
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["n12_q0p10_topk5", "n24_q0p10_topk5"],
        help="P4-alt config IDs to include in dry-run rows.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("debug_output/_tmp_action_handoff_p6_metadata_dryrun_20260524"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sweep_path = args.sweep_root / "p4_alt_sweep_summary.json"

    sweep = _load_json(sweep_path)
    weak = _load_json(args.weak_json)
    feature = _load_json(args.feature_json)

    rows, matrix_rows = _build_rows(
        sweep=sweep,
        weak=weak,
        feature_meta=sweep.get("feature_metadata", {}),
        configs=list(args.configs),
    )
    summary = _build_summary(rows, matrix_rows, args)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / "p6_dryrun_rows.json"
    summary_path = out_dir / "p6_dryrun_summary.json"
    matrix_path = out_dir / "p6_trial_matrix.json"

    rows_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    matrix_path.write_text(json.dumps(matrix_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[p6-dryrun] out_dir={out_dir}")
    print(f"[p6-dryrun] rows={len(rows)}")
    print(f"[p6-dryrun] summary={summary_path}")


if __name__ == "__main__":
    main()
