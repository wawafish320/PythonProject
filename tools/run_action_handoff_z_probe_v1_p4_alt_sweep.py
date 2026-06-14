#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from run_action_handoff_z_probe_v1_recalibration import (
    EPS,
    LOCKED_CLIPS,
    _cosine_distance_matrix,
    _mean_finite,
    _npz_scalar_to_text,
    _pairwise_l2_matrix,
    _pearson_corr,
    _spearman_corr,
    _std_finite,
)

DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"

SWEEP_CONFIGS: list[dict[str, Any]] = [
    {"id": "n6_q0p10_topk5", "future_horizon_n": 6, "oracle_top_q": 0.10, "top_k": 5},
    {"id": "n12_q0p10_topk5", "future_horizon_n": 12, "oracle_top_q": 0.10, "top_k": 5},
    {"id": "n24_q0p10_topk5", "future_horizon_n": 24, "oracle_top_q": 0.10, "top_k": 5},
    {"id": "n48_q0p10_topk5", "future_horizon_n": 48, "oracle_top_q": 0.10, "top_k": 5},
    {"id": "n12_q0p05_topk5", "future_horizon_n": 12, "oracle_top_q": 0.05, "top_k": 5},
    {"id": "n12_q0p20_topk5", "future_horizon_n": 12, "oracle_top_q": 0.20, "top_k": 5},
    {"id": "n12_q0p10_topk1", "future_horizon_n": 12, "oracle_top_q": 0.10, "top_k": 1},
    {"id": "n12_q0p10_topk3", "future_horizon_n": 12, "oracle_top_q": 0.10, "top_k": 3},
    {"id": "n12_q0p10_topk10", "future_horizon_n": 12, "oracle_top_q": 0.10, "top_k": 10},
    {"id": "n24_q0p05_topk5", "future_horizon_n": 24, "oracle_top_q": 0.05, "top_k": 5},
]

GLOBAL_RATIO_THR = 0.85
GLOBAL_SPEARMAN_THR = 0.45
GLOBAL_HIT_LIFT_THR = 0.10
SOURCE_RATIO_THR = 0.90
SOURCE_SPEARMAN_THR = 0.30
WALK_WEAK_RATIO_THR = 0.95
WALK_WEAK_HIT_LIFT_THR = 0.05


def _format(v: float | None, digits: int = 6) -> str:
    if v is None or not np.isfinite(v):
        return "null"
    return f"{float(v):.{digits}f}"


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _make_bucket() -> dict[str, list[float]]:
    return {
        "top1_future": [],
        "topk_future": [],
        "rand_future": [],
        "top1_hit": [],
        "topk_hit": [],
        "spearman": [],
        "pearson": [],
        "oracle_count": [],
        "candidate_count": [],
        "random_top1_expectation": [],
    }


def _append_bucket(
    bucket: dict[str, list[float]],
    *,
    top1_future: float,
    topk_future: float,
    rand_future: float,
    top1_hit: float,
    topk_hit: float,
    spearman: float,
    pearson: float,
    oracle_count: float,
    candidate_count: float,
) -> None:
    bucket["top1_future"].append(top1_future)
    bucket["topk_future"].append(topk_future)
    bucket["rand_future"].append(rand_future)
    bucket["top1_hit"].append(top1_hit)
    bucket["topk_hit"].append(topk_hit)
    bucket["spearman"].append(spearman)
    bucket["pearson"].append(pearson)
    bucket["oracle_count"].append(oracle_count)
    bucket["candidate_count"].append(candidate_count)
    bucket["random_top1_expectation"].append(float(oracle_count / max(candidate_count, 1.0)))


def _summarize_bucket(bucket: dict[str, list[float]]) -> dict[str, Any]:
    top1_f = _mean_finite(bucket["top1_future"])
    topk_f = _mean_finite(bucket["topk_future"])
    rand_f = _mean_finite(bucket["rand_future"])
    top1_hit = _mean_finite(bucket["top1_hit"])
    topk_hit = _mean_finite(bucket["topk_hit"])
    random_top1 = _mean_finite(bucket["random_top1_expectation"])
    hit_lift = None
    if top1_hit is not None and random_top1 is not None:
        hit_lift = float(top1_hit - random_top1)
    return {
        "num_queries": int(len(bucket["top1_future"])),
        "mean_top1_future_distance": top1_f,
        "mean_topk_future_distance": topk_f,
        "mean_random_future_distance": rand_f,
        "top1_future_distance_vs_random_ratio": (
            float(top1_f / max(rand_f, EPS)) if top1_f is not None and rand_f is not None else None
        ),
        "topk_future_distance_vs_random_ratio": (
            float(topk_f / max(rand_f, EPS)) if topk_f is not None and rand_f is not None else None
        ),
        "top1_equiv_hit_rate": top1_hit,
        "topk_equiv_hit_rate": topk_hit,
        "random_top1_expectation": random_top1,
        "top1_equiv_hit_rate_vs_random_top1": hit_lift,
        "mean_spearman_zdist_vs_futuredist": _mean_finite(bucket["spearman"]),
        "std_spearman_zdist_vs_futuredist": _std_finite(bucket["spearman"]),
        "mean_pearson_zdist_vs_futuredist": _mean_finite(bucket["pearson"]),
        "mean_oracle_q_count": _mean_finite(bucket["oracle_count"]),
        "mean_candidate_count": _mean_finite(bucket["candidate_count"]),
    }


def _global_pass_like(agg: dict[str, Any]) -> bool:
    ratio = agg.get("top1_future_distance_vs_random_ratio")
    spearman = agg.get("mean_spearman_zdist_vs_futuredist")
    hit_lift = agg.get("top1_equiv_hit_rate_vs_random_top1")
    if ratio is None or spearman is None or hit_lift is None:
        return False
    return bool(
        (float(ratio) < GLOBAL_RATIO_THR)
        and (float(spearman) > GLOBAL_SPEARMAN_THR)
        and (float(hit_lift) > GLOBAL_HIT_LIFT_THR)
    )


def _source_pass_like(src_metrics: dict[str, Any]) -> bool:
    ratio = src_metrics.get("top1_future_distance_vs_random_ratio")
    spearman = src_metrics.get("mean_spearman_zdist_vs_futuredist")
    if ratio is None or spearman is None:
        return False
    return bool((float(ratio) < SOURCE_RATIO_THR) or (float(spearman) > SOURCE_SPEARMAN_THR))


def _run_single_config(
    *,
    config: dict[str, Any],
    z_by_clip: dict[str, np.ndarray],
    future_desc_by_clip: dict[str, np.ndarray],
    random_k: int,
    seed: int,
) -> dict[str, Any]:
    horizon_n = int(config["future_horizon_n"])
    top_k = int(config["top_k"])
    oracle_q = float(config["oracle_top_q"])

    sig_by_clip: dict[str, np.ndarray] = {}
    z_valid_by_clip: dict[str, np.ndarray] = {}
    valid_frames_by_clip: dict[str, np.ndarray] = {}
    for clip in LOCKED_CLIPS:
        z = z_by_clip[clip]
        desc = future_desc_by_clip[clip]
        t = int(min(z.shape[0], desc.shape[0]))
        valid_t = t - horizon_n + 1
        if valid_t <= 2:
            raise RuntimeError(f"{clip}: not enough frames for future horizon N={horizon_n}")
        windows = np.stack(
            [desc[i : i + horizon_n].reshape(-1) for i in range(valid_t)],
            axis=0,
        ).astype(np.float64, copy=False)
        sig_by_clip[clip] = windows
        z_valid_by_clip[clip] = z[:valid_t].astype(np.float64, copy=False)
        valid_frames_by_clip[clip] = np.arange(valid_t, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    per_pair_bucket: dict[str, dict[str, list[float]]] = {}
    per_source_bucket: dict[str, dict[str, list[float]]] = {}
    all_bucket = _make_bucket()

    for src in LOCKED_CLIPS:
        for tgt in LOCKED_CLIPS:
            if src == tgt:
                continue
            pair_id = f"{src}->{tgt}"
            zs = z_valid_by_clip[src]
            zt = z_valid_by_clip[tgt]
            fs = sig_by_clip[src]
            ft = sig_by_clip[tgt]

            zdist = _cosine_distance_matrix(zs, zt)
            fdist = _pairwise_l2_matrix(fs, ft)
            src_t = int(zdist.shape[0])
            tgt_t = int(zdist.shape[1])
            k_eff = min(top_k, tgt_t)
            q_count = max(1, int(math.ceil(oracle_q * float(tgt_t))))
            pair_bucket = per_pair_bucket.setdefault(pair_id, _make_bucket())
            source_bucket = per_source_bucket.setdefault(src, _make_bucket())

            for i in range(src_t):
                zd = zdist[i]
                fd = fdist[i]
                order_z = np.argsort(zd, kind="stable")
                topk_idx = order_z[:k_eff]
                top1_idx = int(topk_idx[0])
                order_f = np.argsort(fd, kind="stable")
                oracle_idx = order_f[:q_count]
                oracle_set = set(int(x) for x in oracle_idx.tolist())
                rand_size = min(max(random_k, k_eff), tgt_t)
                rand_idx = rng.choice(np.arange(tgt_t, dtype=np.int64), size=rand_size, replace=False)

                top1_future = float(fd[top1_idx])
                topk_future = float(np.mean(fd[topk_idx], dtype=np.float64))
                rand_future = float(np.mean(fd[rand_idx], dtype=np.float64))
                top1_hit = 1.0 if top1_idx in oracle_set else 0.0
                topk_hit = 1.0 if any(int(x) in oracle_set for x in topk_idx.tolist()) else 0.0
                spearman = _spearman_corr(zd.astype(np.float64), fd.astype(np.float64))
                pearson = _pearson_corr(zd.astype(np.float64), fd.astype(np.float64))

                _append_bucket(
                    all_bucket,
                    top1_future=top1_future,
                    topk_future=topk_future,
                    rand_future=rand_future,
                    top1_hit=top1_hit,
                    topk_hit=topk_hit,
                    spearman=spearman,
                    pearson=pearson,
                    oracle_count=float(q_count),
                    candidate_count=float(tgt_t),
                )
                _append_bucket(
                    pair_bucket,
                    top1_future=top1_future,
                    topk_future=topk_future,
                    rand_future=rand_future,
                    top1_hit=top1_hit,
                    topk_hit=topk_hit,
                    spearman=spearman,
                    pearson=pearson,
                    oracle_count=float(q_count),
                    candidate_count=float(tgt_t),
                )
                _append_bucket(
                    source_bucket,
                    top1_future=top1_future,
                    topk_future=topk_future,
                    rand_future=rand_future,
                    top1_hit=top1_hit,
                    topk_hit=topk_hit,
                    spearman=spearman,
                    pearson=pearson,
                    oracle_count=float(q_count),
                    candidate_count=float(tgt_t),
                )

    aggregate = _summarize_bucket(all_bucket)
    per_source = {clip: _summarize_bucket(b) for clip, b in sorted(per_source_bucket.items())}
    per_pair = {pair: _summarize_bucket(b) for pair, b in sorted(per_pair_bucket.items())}
    return {
        "task": "P4-alt cross-clip future-equivalence probe sweep item (no MM oracle)",
        "locked_clips": list(LOCKED_CLIPS),
        "probe_definition": {
            "future_horizon_n": horizon_n,
            "top_k": top_k,
            "oracle_top_q": oracle_q,
            "random_k": int(random_k),
            "seed": int(seed),
            "z_distance": "cosine distance (1-cos)",
            "future_equivalence_distance": "L2 distance between flattened future_desc windows of length N",
            "no_mm_oracle": True,
        },
        "valid_frames_per_clip": {
            clip: int(v.shape[0]) for clip, v in sorted(valid_frames_by_clip.items())
        },
        "aggregate": aggregate,
        "per_source_clip": per_source,
        "per_pair": per_pair,
    }


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run P4-alt future-equivalence stability sweep.")
    ap.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--random-k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    if args.random_k < 1:
        raise RuntimeError("--random-k must be >= 1")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_config_dir = out_dir / "per_config"
    per_config_dir.mkdir(parents=True, exist_ok=True)

    z_path = Path(args.z_features)
    if not z_path.exists():
        raise FileNotFoundError(f"missing z-features: {z_path}")

    npz = np.load(z_path, allow_pickle=True)
    clip_order = (
        [_npz_scalar_to_text(x) for x in np.asarray(npz["clip_order"], dtype=object).tolist()]
        if "clip_order" in npz.files
        else list(LOCKED_CLIPS)
    )

    z_by_clip: dict[str, np.ndarray] = {}
    future_desc_by_clip: dict[str, np.ndarray] = {}
    feature_meta: dict[str, Any] = {}
    for clip in LOCKED_CLIPS:
        z_key = f"{clip}__z"
        future_key = f"{clip}__future_desc"
        if z_key not in npz.files:
            raise RuntimeError(f"missing required key: {z_key}")
        if future_key not in npz.files:
            raise RuntimeError(f"missing required key: {future_key} (fail-fast, no schema fallback)")
        z_raw = np.asarray(npz[z_key])
        future_raw = np.asarray(npz[future_key])
        if z_raw.ndim != 2:
            raise RuntimeError(f"{z_key} must be rank-2, got shape={tuple(z_raw.shape)}")
        if future_raw.ndim != 2:
            raise RuntimeError(f"{future_key} must be rank-2, got shape={tuple(future_raw.shape)}")
        z_by_clip[clip] = z_raw.astype(np.float64, copy=False)
        future_desc_by_clip[clip] = future_raw.astype(np.float64, copy=False)
        feature_meta[clip] = {
            "z": {
                "shape": [int(z_raw.shape[0]), int(z_raw.shape[1])],
                "dtype": str(z_raw.dtype),
                "device": "cpu",
            },
            "future_desc": {
                "shape": [int(future_raw.shape[0]), int(future_raw.shape[1])],
                "dtype": str(future_raw.dtype),
                "device": "cpu",
            },
        }

    config_results: list[dict[str, Any]] = []
    per_source_pass_count: dict[str, int] = {clip: 0 for clip in LOCKED_CLIPS}
    walk_l_to_r_near_random_count = 0
    for cfg in SWEEP_CONFIGS:
        result = _run_single_config(
            config=cfg,
            z_by_clip=z_by_clip,
            future_desc_by_clip=future_desc_by_clip,
            random_k=int(args.random_k),
            seed=int(args.seed),
        )
        agg = result["aggregate"]
        global_pass = _global_pass_like(agg)
        source_pass_map: dict[str, bool] = {}
        for src in LOCKED_CLIPS:
            src_metrics = result["per_source_clip"][src]
            src_pass = _source_pass_like(src_metrics)
            source_pass_map[src] = src_pass
            if src_pass:
                per_source_pass_count[src] += 1
        walk_metrics = result["per_source_clip"]["Walk_L_To_R"]
        walk_ratio = walk_metrics.get("top1_future_distance_vs_random_ratio")
        walk_lift = walk_metrics.get("top1_equiv_hit_rate_vs_random_top1")
        if (
            walk_ratio is not None
            and walk_lift is not None
            and float(walk_ratio) >= WALK_WEAK_RATIO_THR
            and float(walk_lift) <= WALK_WEAK_HIT_LIFT_THR
        ):
            walk_l_to_r_near_random_count += 1

        cfg_dir = per_config_dir / str(cfg["id"])
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_json_path = cfg_dir / "p4_alt_future_equivalence_summary.json"
        payload = {
            "config_id": cfg["id"],
            "input_artifact_path": str(z_path.resolve()),
            "feature_metadata": feature_meta,
            **result,
        }
        _dump_json(cfg_json_path, payload)
        config_results.append(
            {
                "config_id": cfg["id"],
                "future_horizon_n": int(cfg["future_horizon_n"]),
                "oracle_top_q": float(cfg["oracle_top_q"]),
                "top_k": int(cfg["top_k"]),
                "summary_json": str(cfg_json_path.resolve()),
                "aggregate": agg,
                "per_source_clip": result["per_source_clip"],
                "global_pass_like": bool(global_pass),
                "per_source_pass_like": source_pass_map,
            }
        )

    total_configs = len(SWEEP_CONFIGS)
    majority_threshold = int(math.floor(total_configs / 2) + 1)
    global_pass_count = int(sum(1 for x in config_results if bool(x["global_pass_like"])))
    global_stable = bool(global_pass_count >= majority_threshold)

    per_source_stability: dict[str, Any] = {}
    for src in LOCKED_CLIPS:
        src_pass_count = int(per_source_pass_count[src])
        src_ratio_vals: list[float] = []
        src_spear_vals: list[float] = []
        src_lift_vals: list[float] = []
        for cfg in config_results:
            m = cfg["per_source_clip"][src]
            if m["top1_future_distance_vs_random_ratio"] is not None:
                src_ratio_vals.append(float(m["top1_future_distance_vs_random_ratio"]))
            if m["mean_spearman_zdist_vs_futuredist"] is not None:
                src_spear_vals.append(float(m["mean_spearman_zdist_vs_futuredist"]))
            if m["top1_equiv_hit_rate_vs_random_top1"] is not None:
                src_lift_vals.append(float(m["top1_equiv_hit_rate_vs_random_top1"]))
        per_source_stability[src] = {
            "pass_like_count": src_pass_count,
            "total_configs": total_configs,
            "majority_threshold": majority_threshold,
            "stable_majority": bool(src_pass_count >= majority_threshold),
            "mean_top1_ratio": (
                float(np.mean(np.asarray(src_ratio_vals, dtype=np.float64), dtype=np.float64))
                if src_ratio_vals
                else None
            ),
            "mean_spearman": (
                float(np.mean(np.asarray(src_spear_vals, dtype=np.float64), dtype=np.float64))
                if src_spear_vals
                else None
            ),
            "mean_top1_hit_lift": (
                float(np.mean(np.asarray(src_lift_vals, dtype=np.float64), dtype=np.float64))
                if src_lift_vals
                else None
            ),
        }

    walk_pass_count = int(per_source_stability["Walk_L_To_R"]["pass_like_count"])
    walk_is_weak = bool(
        (walk_pass_count < majority_threshold)
        or (walk_l_to_r_near_random_count >= majority_threshold)
    )
    all_sources_positive = bool(
        all(bool(per_source_stability[src]["stable_majority"]) for src in LOCKED_CLIPS)
    )

    if not global_stable:
        decision_key = "metric_redesign_required"
        decision_text = "Global not stable: do not rewrite P4 yet; return to metric design / z objective audit."
    elif all_sources_positive and (not walk_is_weak):
        decision_key = "rewrite_design_doc_p4_alt_main_h3_gate"
        decision_text = (
            "Global stable and all sources positive: recommend design doc rewrite "
            "(P1 downgrade, P4-alt main H3 gate, old MM P4 as secondary diagnostic)."
        )
    else:
        decision_key = "source_specific_failure_analysis_before_p6"
        decision_text = (
            "Global stable but source weakness remains (Walk_L_To_R watch item): "
            "run source-specific failure analysis before P6."
        )

    sweep_summary = {
        "task": "P4-alt future-equivalence stability sweep",
        "input_artifact_path": str(z_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "feature_metadata": {
            "clip_order_from_npz": clip_order,
            "per_clip": feature_meta,
        },
        "sweep_configs": SWEEP_CONFIGS,
        "stability_criteria": {
            "global": {
                "top1_future_distance_vs_random_ratio_lt": GLOBAL_RATIO_THR,
                "mean_spearman_zdist_vs_futuredist_gt": GLOBAL_SPEARMAN_THR,
                "top1_equiv_hit_rate_vs_random_top1_gt": GLOBAL_HIT_LIFT_THR,
                "majority_threshold": majority_threshold,
            },
            "per_source": {
                "pass_like_if_any": {
                    "top1_future_distance_vs_random_ratio_lt": SOURCE_RATIO_THR,
                    "mean_spearman_zdist_vs_futuredist_gt": SOURCE_SPEARMAN_THR,
                },
                "majority_threshold": majority_threshold,
            },
            "walk_l_to_r_watch": {
                "near_random_if": {
                    "top1_future_distance_vs_random_ratio_gte": WALK_WEAK_RATIO_THR,
                    "top1_equiv_hit_rate_vs_random_top1_lte": WALK_WEAK_HIT_LIFT_THR,
                },
                "near_random_majority_threshold": majority_threshold,
            },
        },
        "configs": config_results,
        "global_stability": {
            "pass_like_count": global_pass_count,
            "total_configs": total_configs,
            "majority_threshold": majority_threshold,
            "stable_majority": global_stable,
        },
        "per_source_stability": per_source_stability,
        "walk_l_to_r_watch": {
            "pass_like_count": walk_pass_count,
            "near_random_count": int(walk_l_to_r_near_random_count),
            "total_configs": total_configs,
            "majority_threshold": majority_threshold,
            "weak_majority": walk_is_weak,
        },
        "decision": {
            "key": decision_key,
            "text": decision_text,
            "reminder": "P1 point-prediction risk remains unresolved; P4-alt stability alone does not pass v1.",
        },
    }
    summary_json_path = out_dir / "p4_alt_sweep_summary.json"
    _dump_json(summary_json_path, sweep_summary)

    md_lines: list[str] = [
        "# P4-alt Future-Equivalence Stability Sweep",
        "",
        f"- Input artifact: `{z_path.resolve()}`",
        f"- Output dir: `{out_dir.resolve()}`",
        f"- Total configs: {total_configs}, majority threshold: {majority_threshold}",
        "",
        "## Global Table (10 configs)",
        "",
        "| config_id | N | q | top_k | ratio(top1/random) | hit_lift(top1-rand) | spearman | pearson | pass-like |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg in config_results:
        agg = cfg["aggregate"]
        md_lines.append(
            "| "
            f"{cfg['config_id']} | "
            f"{cfg['future_horizon_n']} | "
            f"{cfg['oracle_top_q']:.2f} | "
            f"{cfg['top_k']} | "
            f"{_format(agg['top1_future_distance_vs_random_ratio'])} | "
            f"{_format(agg['top1_equiv_hit_rate_vs_random_top1'])} | "
            f"{_format(agg['mean_spearman_zdist_vs_futuredist'])} | "
            f"{_format(agg['mean_pearson_zdist_vs_futuredist'])} | "
            f"{'Y' if cfg['global_pass_like'] else 'N'} |"
        )
    md_lines.extend(
        [
            "",
            "## Per-source Stability",
            "",
            "| source | pass_like_count | stable_majority | mean_ratio | mean_spearman | mean_hit_lift |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for src in LOCKED_CLIPS:
        row = per_source_stability[src]
        md_lines.append(
            "| "
            f"{src} | "
            f"{row['pass_like_count']}/{row['total_configs']} | "
            f"{'Y' if row['stable_majority'] else 'N'} | "
            f"{_format(row['mean_top1_ratio'])} | "
            f"{_format(row['mean_spearman'])} | "
            f"{_format(row['mean_top1_hit_lift'])} |"
        )
    walk_metrics_rows = []
    for cfg in config_results:
        m = cfg["per_source_clip"]["Walk_L_To_R"]
        walk_metrics_rows.append(
            (
                cfg["config_id"],
                m["top1_future_distance_vs_random_ratio"],
                m["mean_spearman_zdist_vs_futuredist"],
                m["top1_equiv_hit_rate_vs_random_top1"],
                cfg["per_source_pass_like"]["Walk_L_To_R"],
            )
        )
    md_lines.extend(
        [
            "",
            "## Walk_L_To_R Watch",
            "",
            f"- pass_like_count: {walk_pass_count}/{total_configs}",
            f"- near_random_count (ratio>=0.95 and hit_lift<=0.05): {walk_l_to_r_near_random_count}/{total_configs}",
            "",
            "| config_id | ratio(top1/random) | spearman | hit_lift | pass-like |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cfg_id, ratio, spear, lift, passed in walk_metrics_rows:
        md_lines.append(
            "| "
            f"{cfg_id} | {_format(ratio)} | {_format(spear)} | {_format(lift)} | {'Y' if passed else 'N'} |"
        )
    md_lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- {decision_text}",
            "- Reminder: P1 point-prediction risk remains unresolved; P4-alt stability does not alone pass v1.",
        ]
    )
    summary_md_path = out_dir / "p4_alt_sweep_summary.md"
    _dump_md(summary_md_path, md_lines)

    print(f"[ok] p4_alt_sweep_summary.json: {summary_json_path}")
    print(f"[ok] p4_alt_sweep_summary.md: {summary_md_path}")
    print(f"[ok] decision={decision_key} global_pass={global_pass_count}/{total_configs}")


if __name__ == "__main__":
    main()
