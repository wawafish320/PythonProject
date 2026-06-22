#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SWEEP_SUMMARY = (
    "debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/"
    "p4_alt_sweep_summary.json"
)
TARGET_SOURCE = "Walk_L_To_R"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"expected dict json: {path}")
    return data


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _f(v: float | None, digits: int = 6) -> str:
    if v is None or not np.isfinite(v):
        return "null"
    return f"{float(v):.{digits}f}"


def _mean_finite(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr, dtype=np.float64))


def _extract_metric(metrics: dict[str, Any], key: str) -> float:
    v = metrics.get(key)
    if v is None:
        return float("nan")
    x = float(v)
    if not np.isfinite(x):
        return float("nan")
    return x


def _summary_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "ratio",
        "hit_lift",
        "spearman",
        "pearson",
        "top1_equiv_hit_rate",
        "topk_equiv_hit_rate",
    ]
    out: dict[str, Any] = {"count": int(len(records))}
    for k in keys:
        out[f"mean_{k}"] = _mean_finite([float(r[k]) for r in records if np.isfinite(float(r[k]))])
    return out


def _rank_sources(source_metrics: dict[str, dict[str, Any]], key: str, higher_is_better: bool) -> dict[str, int]:
    rows: list[tuple[str, float]] = []
    for src, m in source_metrics.items():
        v = _extract_metric(m, key)
        if not np.isfinite(v):
            v = -float("inf") if higher_is_better else float("inf")
        rows.append((src, v))
    rows.sort(key=lambda x: x[1], reverse=higher_is_better)
    return {src: idx + 1 for idx, (src, _) in enumerate(rows)}


def _source_comparison(source_metrics: dict[str, dict[str, Any]], target_source: str) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    metric_specs = [
        ("top1_future_distance_vs_random_ratio", False, "ratio"),
        ("top1_equiv_hit_rate_vs_random_top1", True, "hit_lift"),
        ("mean_spearman_zdist_vs_futuredist", True, "spearman"),
        ("mean_pearson_zdist_vs_futuredist", True, "pearson"),
        ("top1_equiv_hit_rate", True, "top1_equiv_hit_rate"),
        ("topk_equiv_hit_rate", True, "topk_equiv_hit_rate"),
    ]

    for key, higher_is_better, short_name in metric_specs:
        values: list[tuple[str, float]] = []
        for src, m in source_metrics.items():
            values.append((src, _extract_metric(m, key)))

        finite_vals = [(src, v) for src, v in values if np.isfinite(v)]
        if not finite_vals or target_source not in source_metrics:
            comparison[short_name] = {
                "walk_l_to_r_value": None,
                "source_mean": None,
                "source_best": None,
                "source_worst": None,
                "walk_l_to_r_rank": None,
                "num_sources": int(len(source_metrics)),
            }
            continue

        walk_v = _extract_metric(source_metrics[target_source], key)
        arr = np.asarray([v for _, v in finite_vals], dtype=np.float64)
        mean_v = float(np.mean(arr, dtype=np.float64))
        if higher_is_better:
            best_src, best_v = max(finite_vals, key=lambda x: x[1])
            worst_src, worst_v = min(finite_vals, key=lambda x: x[1])
            rank_map = _rank_sources(source_metrics, key, higher_is_better=True)
        else:
            best_src, best_v = min(finite_vals, key=lambda x: x[1])
            worst_src, worst_v = max(finite_vals, key=lambda x: x[1])
            rank_map = _rank_sources(source_metrics, key, higher_is_better=False)

        comparison[short_name] = {
            "walk_l_to_r_value": float(walk_v) if np.isfinite(walk_v) else None,
            "source_mean": mean_v,
            "source_best": {"source": best_src, "value": float(best_v)},
            "source_worst": {"source": worst_src, "value": float(worst_v)},
            "walk_l_to_r_rank": int(rank_map.get(target_source, -1)),
            "num_sources": int(len(source_metrics)),
        }

    return comparison


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Walk_L_To_R source-specific failure analysis for P4-alt sweep.")
    ap.add_argument("--sweep-summary", type=Path, default=Path(DEFAULT_SWEEP_SUMMARY))
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    sweep_path = Path(args.sweep_summary)
    if not sweep_path.exists():
        raise FileNotFoundError(f"missing sweep summary: {sweep_path}")

    sweep = _load_json(sweep_path)
    configs = sweep.get("configs")
    if not isinstance(configs, list) or not configs:
        raise RuntimeError("sweep summary has empty/missing configs")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(
        f"debug_output/_tmp_action_handoff_z_probe_v1_walk_l_to_r_failure_analysis_{date_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    near_random_cfg = (
        sweep.get("stability_criteria", {})
        .get("walk_l_to_r_watch", {})
        .get("near_random_if", {})
    )
    near_random_ratio_thr = float(near_random_cfg.get("top1_future_distance_vs_random_ratio_gte", 0.95))
    near_random_lift_thr = float(near_random_cfg.get("top1_equiv_hit_rate_vs_random_top1_lte", 0.05))
    total_cfg = int(len(configs))
    majority_thr = int(
        sweep.get("walk_l_to_r_watch", {}).get(
            "majority_threshold", math.floor(total_cfg / 2) + 1
        )
    )

    records: list[dict[str, Any]] = []
    horizon_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    q_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    topk_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    per_target_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_target_worst_ratio_count: dict[str, int] = defaultdict(int)
    per_target_worst_lift_count: dict[str, int] = defaultdict(int)
    source_rank_history: dict[str, list[int]] = {"ratio": [], "hit_lift": [], "spearman": []}

    near_random_count = 0

    for cfg in configs:
        cfg_id = str(cfg.get("config_id"))
        n = int(cfg.get("future_horizon_n"))
        q = float(cfg.get("oracle_top_q"))
        top_k = int(cfg.get("top_k"))

        summary_json = cfg.get("summary_json")
        if summary_json is None:
            raise RuntimeError(f"config {cfg_id}: missing summary_json")
        detail_path = Path(str(summary_json))
        if not detail_path.exists():
            raise FileNotFoundError(f"config {cfg_id}: missing detail summary {detail_path}")
        detail = _load_json(detail_path)

        per_source = detail.get("per_source_clip")
        if not isinstance(per_source, dict) or TARGET_SOURCE not in per_source:
            raise RuntimeError(f"config {cfg_id}: missing {TARGET_SOURCE} in per_source_clip")
        walk = per_source[TARGET_SOURCE]

        row = {
            "config_id": cfg_id,
            "future_horizon_n": n,
            "oracle_top_q": q,
            "top_k": top_k,
            "ratio": _extract_metric(walk, "top1_future_distance_vs_random_ratio"),
            "hit_lift": _extract_metric(walk, "top1_equiv_hit_rate_vs_random_top1"),
            "spearman": _extract_metric(walk, "mean_spearman_zdist_vs_futuredist"),
            "pearson": _extract_metric(walk, "mean_pearson_zdist_vs_futuredist"),
            "top1_equiv_hit_rate": _extract_metric(walk, "top1_equiv_hit_rate"),
            "topk_equiv_hit_rate": _extract_metric(walk, "topk_equiv_hit_rate"),
        }

        ratio = row["ratio"]
        lift = row["hit_lift"]
        is_near_random = bool(
            np.isfinite(ratio)
            and np.isfinite(lift)
            and ratio >= near_random_ratio_thr
            and lift <= near_random_lift_thr
        )
        row["near_random"] = is_near_random
        near_random_count += int(is_near_random)

        comp = _source_comparison(per_source, TARGET_SOURCE)
        row["source_comparison"] = comp
        if comp["ratio"]["walk_l_to_r_rank"] is not None:
            source_rank_history["ratio"].append(int(comp["ratio"]["walk_l_to_r_rank"]))
        if comp["hit_lift"]["walk_l_to_r_rank"] is not None:
            source_rank_history["hit_lift"].append(int(comp["hit_lift"]["walk_l_to_r_rank"]))
        if comp["spearman"]["walk_l_to_r_rank"] is not None:
            source_rank_history["spearman"].append(int(comp["spearman"]["walk_l_to_r_rank"]))

        pair = detail.get("per_pair")
        if not isinstance(pair, dict):
            raise RuntimeError(f"config {cfg_id}: missing per_pair")

        walk_pairs = {
            k: v for k, v in pair.items() if isinstance(k, str) and k.startswith(f"{TARGET_SOURCE}->")
        }
        if not walk_pairs:
            raise RuntimeError(f"config {cfg_id}: missing {TARGET_SOURCE}->* per_pair metrics")

        ratio_rows: list[tuple[str, float]] = []
        lift_rows: list[tuple[str, float]] = []

        for pair_key, m in sorted(walk_pairs.items()):
            target = str(pair_key.split("->", 1)[1])
            pair_row = {
                "config_id": cfg_id,
                "future_horizon_n": n,
                "oracle_top_q": q,
                "top_k": top_k,
                "target_clip": target,
                "ratio": _extract_metric(m, "top1_future_distance_vs_random_ratio"),
                "hit_lift": _extract_metric(m, "top1_equiv_hit_rate_vs_random_top1"),
                "spearman": _extract_metric(m, "mean_spearman_zdist_vs_futuredist"),
                "pearson": _extract_metric(m, "mean_pearson_zdist_vs_futuredist"),
                "top1_equiv_hit_rate": _extract_metric(m, "top1_equiv_hit_rate"),
                "topk_equiv_hit_rate": _extract_metric(m, "topk_equiv_hit_rate"),
            }
            per_target_records[target].append(pair_row)
            ratio_rows.append((target, pair_row["ratio"]))
            lift_rows.append((target, pair_row["hit_lift"]))

        finite_ratio_rows = [(t, v) for t, v in ratio_rows if np.isfinite(v)]
        if finite_ratio_rows:
            worst_ratio_target, _ = max(finite_ratio_rows, key=lambda x: x[1])
            per_target_worst_ratio_count[worst_ratio_target] += 1

        finite_lift_rows = [(t, v) for t, v in lift_rows if np.isfinite(v)]
        if finite_lift_rows:
            worst_lift_target, _ = min(finite_lift_rows, key=lambda x: x[1])
            per_target_worst_lift_count[worst_lift_target] += 1

        records.append(row)
        horizon_groups[n].append(row)
        q_groups[f"{q:.2f}"].append(row)
        topk_groups[top_k].append(row)

    by_horizon = {
        str(k): _summary_from_records(v) for k, v in sorted(horizon_groups.items(), key=lambda x: x[0])
    }
    by_q = {k: _summary_from_records(v) for k, v in sorted(q_groups.items(), key=lambda x: float(x[0]))}
    by_topk = {
        str(k): _summary_from_records(v) for k, v in sorted(topk_groups.items(), key=lambda x: x[0])
    }

    short = [r for r in records if int(r["future_horizon_n"]) in (6, 12)]
    long = [r for r in records if int(r["future_horizon_n"]) >= 24]
    short_ratio = _mean_finite([float(r["ratio"]) for r in short if np.isfinite(float(r["ratio"]))])
    short_lift = _mean_finite([float(r["hit_lift"]) for r in short if np.isfinite(float(r["hit_lift"]))])
    short_spear = _mean_finite([float(r["spearman"]) for r in short if np.isfinite(float(r["spearman"]))])
    long_ratio = _mean_finite([float(r["ratio"]) for r in long if np.isfinite(float(r["ratio"]))])
    long_lift = _mean_finite([float(r["hit_lift"]) for r in long if np.isfinite(float(r["hit_lift"]))])
    long_spear = _mean_finite([float(r["spearman"]) for r in long if np.isfinite(float(r["spearman"]))])

    long_degrade_votes = 0
    if short_ratio is not None and long_ratio is not None and long_ratio > short_ratio + 0.03:
        long_degrade_votes += 1
    if short_lift is not None and long_lift is not None and long_lift < short_lift - 0.02:
        long_degrade_votes += 1
    if short_spear is not None and long_spear is not None and long_spear < short_spear - 0.10:
        long_degrade_votes += 1
    long_horizon_degradation = bool((len(long) > 0) and (long_degrade_votes >= 2))

    near_random = bool(near_random_count >= majority_thr)

    per_target_summary: dict[str, Any] = {}
    target_ratio_means: list[float] = []
    for target, rows in sorted(per_target_records.items()):
        summ = _summary_from_records(rows)
        summ["worst_ratio_count"] = int(per_target_worst_ratio_count.get(target, 0))
        summ["worst_hit_lift_count"] = int(per_target_worst_lift_count.get(target, 0))
        per_target_summary[target] = summ
        mean_ratio = summ.get("mean_ratio")
        if mean_ratio is not None:
            target_ratio_means.append(float(mean_ratio))

    target_ratio_range = None
    if len(target_ratio_means) >= 2:
        target_ratio_range = float(max(target_ratio_means) - min(target_ratio_means))

    max_worst_ratio_count = max(per_target_worst_ratio_count.values()) if per_target_worst_ratio_count else 0
    max_worst_lift_count = max(per_target_worst_lift_count.values()) if per_target_worst_lift_count else 0
    target_specific_weakness = bool(
        (target_ratio_range is not None and target_ratio_range >= 0.08)
        or (max_worst_ratio_count >= math.ceil(total_cfg * 0.4))
        or (max_worst_lift_count >= math.ceil(total_cfg * 0.4))
    )

    rank_summary = {}
    for key, values in source_rank_history.items():
        rank_summary[key] = {
            "mean_rank": _mean_finite([float(v) for v in values]),
            "best_rank": int(min(values)) if values else None,
            "worst_rank": int(max(values)) if values else None,
            "num_configs": int(len(values)),
        }

    weakest_targets: list[str] = []
    if per_target_summary:
        worst_by_ratio = max(
            per_target_summary.items(),
            key=lambda kv: float(kv[1]["mean_ratio"]) if kv[1]["mean_ratio"] is not None else -float("inf"),
        )[0]
        weakest_targets.append(worst_by_ratio)
        worst_by_lift = min(
            per_target_summary.items(),
            key=lambda kv: float(kv[1]["mean_hit_lift"]) if kv[1]["mean_hit_lift"] is not None else float("inf"),
        )[0]
        if worst_by_lift not in weakest_targets:
            weakest_targets.append(worst_by_lift)

    if near_random:
        recommended_next_step = (
            "near_random=yes: do not move to P6; run source-specific representation/objective analysis first."
        )
    elif long_horizon_degradation:
        recommended_next_step = (
            "near_random=no with main weakness at N>=24: keep P6 in planning path, treat Walk_L_To_R as known stress "
            "case, and add long-horizon warning into the P4-alt gate report."
        )
    else:
        recommended_next_step = (
            "near_random=no and no strong long-horizon collapse: continue P6 planning with standard source-risk watch."
        )

    if target_specific_weakness and weakest_targets:
        recommended_next_step += (
            " target_specific_weakness=yes: focus later P6/diagnostics on "
            + ", ".join(f"{TARGET_SOURCE}->{t}" for t in weakest_targets)
            + "."
        )

    decision_wording = (
        "P4-alt stability supports recalibrating H3 gate and continuing toward P6 planning, "
        "with Walk_L_To_R as known weak-source risk and P1 magnitude-regression risk unresolved."
    )

    result = {
        "task": "Walk_L_To_R source-specific failure analysis for P4-alt sweep",
        "input_artifacts": {
            "sweep_summary": str(sweep_path.resolve()),
            "per_config_summaries": sorted(
                str(Path(str(cfg["summary_json"])).resolve()) for cfg in configs
            ),
        },
        "decision_wording": decision_wording,
        "walk_l_to_r_per_config": records,
        "grouped_by_horizon": by_horizon,
        "grouped_by_q": by_q,
        "grouped_by_top_k": by_topk,
        "per_pair_aggregate": {
            "per_target": per_target_summary,
            "target_ratio_range": target_ratio_range,
            "worst_ratio_count_by_target": dict(sorted(per_target_worst_ratio_count.items())),
            "worst_hit_lift_count_by_target": dict(sorted(per_target_worst_lift_count.items())),
        },
        "vs_other_sources": {
            "source_rank_summary": rank_summary,
            "per_config_comparison": [
                {
                    "config_id": r["config_id"],
                    "future_horizon_n": r["future_horizon_n"],
                    "oracle_top_q": r["oracle_top_q"],
                    "top_k": r["top_k"],
                    "comparison": r["source_comparison"],
                }
                for r in records
            ],
        },
        "flags": {
            "long_horizon_degradation": long_horizon_degradation,
            "target_specific_weakness": target_specific_weakness,
            "near_random": near_random,
            "near_random_count": int(near_random_count),
            "total_configs": int(total_cfg),
            "near_random_majority_threshold": int(majority_thr),
            "long_horizon_vote_count": int(long_degrade_votes),
        },
        "supporting_stats": {
            "short_horizon_mean": {
                "ratio": short_ratio,
                "hit_lift": short_lift,
                "spearman": short_spear,
            },
            "long_horizon_mean": {
                "ratio": long_ratio,
                "hit_lift": long_lift,
                "spearman": long_spear,
            },
        },
        "recommended_next_step": recommended_next_step,
        "status_wording": "H3 partially supported under recalibrated P4-alt yardstick",
        "p1_risk_note": "P1 point-regression Huber remains weaker than energy/raw and stays unresolved.",
    }

    json_path = out_dir / "walk_l_to_r_failure_analysis.json"
    _dump_json(json_path, result)

    md_lines = [
        "# Walk_L_To_R Failure Analysis (P4-alt Sweep)",
        "",
        f"- Input sweep summary: `{sweep_path.resolve()}`",
        f"- Output dir: `{out_dir.resolve()}`",
        f"- Decision wording: {decision_wording}",
        "- Status wording: H3 partially supported under recalibrated P4-alt yardstick",
        "- P1 risk: point-regression Huber remains weaker than energy/raw (diagnostic risk unresolved)",
        "",
        "## Walk_L_To_R Per-config",
        "",
        "| config_id | N | q | top_k | ratio | hit_lift | spearman | pearson | top1_hit | topk_hit | near_random | rank_ratio | rank_hit_lift | rank_spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        comp = r["source_comparison"]
        md_lines.append(
            "| "
            + f"{r['config_id']} | {r['future_horizon_n']} | {r['oracle_top_q']:.2f} | {r['top_k']}"
            + f" | {_f(r['ratio'])} | {_f(r['hit_lift'])} | {_f(r['spearman'])} | {_f(r['pearson'])}"
            + f" | {_f(r['top1_equiv_hit_rate'])} | {_f(r['topk_equiv_hit_rate'])}"
            + f" | {'Y' if r['near_random'] else 'N'}"
            + f" | {comp['ratio']['walk_l_to_r_rank']} | {comp['hit_lift']['walk_l_to_r_rank']} | {comp['spearman']['walk_l_to_r_rank']} |"
        )

    md_lines.extend([
        "",
        "## Grouped Means by Horizon",
        "",
        "| N | count | mean_ratio | mean_hit_lift | mean_spearman | mean_pearson | mean_top1_hit | mean_topk_hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for n, s in sorted(by_horizon.items(), key=lambda x: int(x[0])):
        md_lines.append(
            f"| {n} | {s['count']} | {_f(s['mean_ratio'])} | {_f(s['mean_hit_lift'])}"
            f" | {_f(s['mean_spearman'])} | {_f(s['mean_pearson'])}"
            f" | {_f(s['mean_top1_equiv_hit_rate'])} | {_f(s['mean_topk_equiv_hit_rate'])} |"
        )

    md_lines.extend([
        "",
        "## Grouped Means by q",
        "",
        "| q | count | mean_ratio | mean_hit_lift | mean_spearman | mean_pearson |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for q_key, s in sorted(by_q.items(), key=lambda x: float(x[0])):
        md_lines.append(
            f"| {q_key} | {s['count']} | {_f(s['mean_ratio'])} | {_f(s['mean_hit_lift'])}"
            f" | {_f(s['mean_spearman'])} | {_f(s['mean_pearson'])} |"
        )

    md_lines.extend([
        "",
        "## Grouped Means by top_k",
        "",
        "| top_k | count | mean_ratio | mean_hit_lift | mean_spearman | mean_pearson |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for k, s in sorted(by_topk.items(), key=lambda x: int(x[0])):
        md_lines.append(
            f"| {k} | {s['count']} | {_f(s['mean_ratio'])} | {_f(s['mean_hit_lift'])}"
            f" | {_f(s['mean_spearman'])} | {_f(s['mean_pearson'])} |"
        )

    md_lines.extend([
        "",
        "## Walk_L_To_R -> Target Pair Aggregate",
        "",
        "| target | count | mean_ratio | mean_hit_lift | mean_spearman | mean_pearson | mean_top1_hit | mean_topk_hit | worst_ratio_count | worst_hit_lift_count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for tgt, s in sorted(per_target_summary.items()):
        md_lines.append(
            f"| {tgt} | {s['count']} | {_f(s['mean_ratio'])} | {_f(s['mean_hit_lift'])}"
            f" | {_f(s['mean_spearman'])} | {_f(s['mean_pearson'])}"
            f" | {_f(s['mean_top1_equiv_hit_rate'])} | {_f(s['mean_topk_equiv_hit_rate'])}"
            f" | {s['worst_ratio_count']} | {s['worst_hit_lift_count']} |"
        )

    md_lines.extend([
        "",
        "## Flags",
        "",
        f"- long_horizon_degradation: {'yes' if long_horizon_degradation else 'no'}",
        f"- target_specific_weakness: {'yes' if target_specific_weakness else 'no'}",
        f"- near_random: {'yes' if near_random else 'no'} ({near_random_count}/{total_cfg}, majority={majority_thr})",
        f"- short_horizon_mean: ratio={_f(short_ratio)}, hit_lift={_f(short_lift)}, spearman={_f(short_spear)}",
        f"- long_horizon_mean: ratio={_f(long_ratio)}, hit_lift={_f(long_lift)}, spearman={_f(long_spear)}",
        "",
        "## Recommended Next Step",
        "",
        f"- {recommended_next_step}",
    ])

    md_path = out_dir / "walk_l_to_r_failure_analysis.md"
    _dump_md(md_path, md_lines)

    print(f"[ok] json: {json_path}")
    print(f"[ok] md: {md_path}")


if __name__ == "__main__":
    main()
