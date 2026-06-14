#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

LOCKED_CLIPS = (
    "Walk_F",
    "Walk_L_To_L",
    "Walk_L_To_R",
    "Walk_R_To_L",
    "Walk_R_To_R",
)
TURN_CLIPS = (
    "Walk_L_To_L",
    "Walk_L_To_R",
    "Walk_R_To_L",
    "Walk_R_To_R",
)
EPS = 1e-8

DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"
DEFAULT_P1 = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/p1_predictive_compare.json"
DEFAULT_P4 = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/p4_cross_clip_entry.json"
DEFAULT_SUMMARY = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _npz_scalar_to_text(v: Any) -> str:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    return str(v)


def _format(v: float | None, digits: int = 6) -> str:
    if v is None or not np.isfinite(v):
        return "null"
    return f"{float(v):.{digits}f}"


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.sum(d * d, dtype=np.float64), dtype=np.float64))


def _pairwise_mean_distance(vs: list[np.ndarray]) -> float:
    if len(vs) < 2:
        return 0.0
    vals: list[float] = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            vals.append(_l2(vs[i], vs[j]))
    return float(np.mean(np.asarray(vals, dtype=np.float64), dtype=np.float64)) if vals else 0.0


def _polygon_area(points2d: np.ndarray) -> float:
    if points2d.shape[0] < 3:
        return 0.0
    x = points2d[:, 0]
    y = points2d[:, 1]
    x2 = np.concatenate([x, x[:1]], axis=0)
    y2 = np.concatenate([y, y[:1]], axis=0)
    return float(0.5 * abs(np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1], dtype=np.float64)))


def _pca_2d(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z64 = np.asarray(z, dtype=np.float64)
    zc = z64 - np.mean(z64, axis=0, keepdims=True)
    _, svals, vt = np.linalg.svd(zc, full_matrices=False)
    vars_ = (svals * svals) / max(float(zc.shape[0] - 1), 1.0)
    total = float(np.sum(vars_, dtype=np.float64))
    evr = vars_ / total if total > 0.0 else np.zeros_like(vars_)
    pc2 = vt[:2].T
    proj = zc @ pc2
    return proj.astype(np.float64, copy=False), evr.astype(np.float64, copy=False)


def _knn_temporal_consistency(z: np.ndarray, *, k: int = 5, exclude_window: int = 2) -> dict[str, Any]:
    t = int(z.shape[0])
    dmat = np.sqrt(np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2, dtype=np.float64), dtype=np.float64)
    vals: list[float] = []
    ncnt: list[int] = []
    for i in range(t):
        mask = np.ones(t, dtype=bool)
        lo = max(0, i - exclude_window)
        hi = min(t, i + exclude_window + 1)
        mask[lo:hi] = False
        cand = np.where(mask)[0]
        if cand.size <= 0:
            ncnt.append(0)
            continue
        local = dmat[i, cand]
        order = np.argsort(local)
        kk = int(min(k, order.size))
        chosen = cand[order[:kk]]
        ncnt.append(kk)
        for j in chosen:
            delta = abs(i - int(j))
            cdelta = min(delta, t - delta)
            vals.append(float(cdelta / max(float(t), 1.0)))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "k": int(k),
        "exclude_temporal_window": int(exclude_window),
        "count": int(arr.size),
        "mean": float(np.mean(arr, dtype=np.float64)) if arr.size else None,
        "p50": float(np.percentile(arr, 50.0)) if arr.size else None,
        "neighbor_count_per_query": {
            "mean": float(np.mean(np.asarray(ncnt, dtype=np.float64), dtype=np.float64)) if ncnt else 0.0,
            "min": int(min(ncnt)) if ncnt else 0,
            "max": int(max(ncnt)) if ncnt else 0,
        },
    }


def _turn_mid_window_indices(length: int, k: int) -> tuple[int, int]:
    mid = length // 2
    s = max(0, mid - (k // 2))
    e = s + k
    if e > length:
        e = length
        s = max(0, e - k)
    return int(s), int(e)


def _random_middle_index(length: int, end_window_k: int, rng: np.random.Generator) -> int:
    end_start = max(0, length - end_window_k)
    low = max(0, length // 4)
    high = min(end_start, (3 * length) // 4)
    cands = np.arange(low, high, dtype=np.int64)
    if cands.size <= 0:
        cands = np.arange(0, max(end_start, 1), dtype=np.int64)
    if cands.size <= 0:
        return int(length - 1)
    return int(rng.choice(cands))


def _cosine_distance_matrix(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    src_n = src / np.maximum(np.linalg.norm(src, axis=1, keepdims=True), EPS)
    tgt_n = tgt / np.maximum(np.linalg.norm(tgt, axis=1, keepdims=True), EPS)
    sim = src_n @ tgt_n.T
    sim = np.clip(sim, -1.0, 1.0)
    return (1.0 - sim).astype(np.float64, copy=False)


def _pairwise_l2_matrix(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    src2 = np.sum(src * src, axis=1, dtype=np.float64, keepdims=True)
    tgt2 = np.sum(tgt * tgt, axis=1, dtype=np.float64, keepdims=True).T
    d2 = np.maximum(src2 + tgt2 - 2.0 * (src @ tgt.T), 0.0)
    return np.sqrt(d2, dtype=np.float64)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size <= 1:
        return float("nan")
    rx = np.empty_like(x, dtype=np.float64)
    ry = np.empty_like(y, dtype=np.float64)
    rx[np.argsort(x, kind="stable")] = np.arange(x.size, dtype=np.float64)
    ry[np.argsort(y, kind="stable")] = np.arange(y.size, dtype=np.float64)
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    den = math.sqrt(float(np.sum(rx * rx, dtype=np.float64) * np.sum(ry * ry, dtype=np.float64)))
    if den <= 0.0:
        return float("nan")
    return float(np.sum(rx * ry, dtype=np.float64) / den)


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size <= 1:
        return float("nan")
    xx = x - np.mean(x)
    yy = y - np.mean(y)
    den = math.sqrt(float(np.sum(xx * xx, dtype=np.float64) * np.sum(yy * yy, dtype=np.float64)))
    if den <= 0.0:
        return float("nan")
    return float(np.sum(xx * yy, dtype=np.float64) / den)


def _mean_finite(vals: list[float]) -> float | None:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr, dtype=np.float64)) if arr.size else None


def _std_finite(vals: list[float]) -> float | None:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, dtype=np.float64)) if arr.size else None


def _internal_v2_interpretation(p2: dict[str, Any], p3: dict[str, Any]) -> dict[str, Any]:
    p2_cycle = float(p2["metrics"]["cycle_closure_ratio"])
    p2_knn = float(p2["metrics"]["knn_temporal_consistency"]["mean"])
    p2_pca = float(p2["metrics"]["pca_2d_explained_variance"])
    p3_mono = float(p3["aggregate"]["monotonic_fraction_gt_0p60_count"]) / float(len(TURN_CLIPS))
    p3_slope = float(p3["aggregate"]["slope_negative_count"]) / float(len(TURN_CLIPS))
    p3_tight = float(p3["aggregate"]["end_tightness_ratio"])
    p3_var = float(p3["aggregate"]["mean_end_vs_mid_variance_ratio"])

    phase_locality = "strong" if (p2_knn < 0.10 and p2_pca > 0.65) else ("moderate" if (p2_knn < 0.20 and p2_pca > 0.50) else "weak")
    closure_signal = "weak_closure" if p2_cycle >= 0.75 else ("partial_closure" if p2_cycle >= 0.35 else "strong_closure")
    turn_convergence = "strong" if (p3_mono >= 0.75 and p3_slope >= 0.75) else ("moderate" if (p3_mono >= 0.50 and p3_slope >= 0.50) else "weak")
    turn_endpoint_tightness = "strong" if p3_tight < 0.75 else ("moderate" if p3_tight < 1.0 else "weak")
    turn_end_stability_vs_mid = "weaker_than_mid" if p3_var >= 1.0 else ("similar_or_better_than_mid" if p3_var < 0.75 else "mixed")

    if phase_locality in ("strong", "moderate") and turn_convergence in ("strong", "moderate") and turn_endpoint_tightness in ("strong", "moderate"):
        structure_status = "structured_but_mixed"
    else:
        structure_status = "structure_weak_or_uncertain"

    return {
        "status": structure_status,
        "axes": {
            "phase_locality": phase_locality,
            "cycle_closure": closure_signal,
            "turn_monotonic_convergence": turn_convergence,
            "cross_turn_end_tightness": turn_endpoint_tightness,
            "end_window_variance_vs_mid": turn_end_stability_vs_mid,
        },
        "note": "internal_structure_v2 is diagnostic-only; no hard pass/fail gating.",
    }


def _p4_alt_interpretation(agg: dict[str, Any]) -> dict[str, Any]:
    top1_ratio = float(agg["top1_future_distance_vs_random_ratio"])
    topk_ratio = float(agg["topk_future_distance_vs_random_ratio"])
    top1_hit_vs = float(agg["top1_equiv_hit_rate_vs_random_top1"])
    mean_spear = float(agg["mean_spearman_zdist_vs_futuredist"])

    if top1_ratio < 0.90 and topk_ratio < 0.95 and top1_hit_vs > 0.05 and mean_spear > 0.15:
        status = "supports_recalibration"
    elif top1_ratio > 0.98 and topk_ratio > 0.98 and top1_hit_vs <= 0.01 and mean_spear <= 0.05:
        status = "does_not_support_recalibration"
    else:
        status = "inconclusive"

    return {
        "status": status,
        "note": "P4-alt removes MM oracle and directly tests future-equivalence predictive power from z-neighborhood.",
    }


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run internal_structure_v2 + P4-alt future-equivalence probe.")
    ap.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    ap.add_argument("--internal-out-dir", type=Path, default=None)
    ap.add_argument("--p4-alt-out-dir", type=Path, default=None)
    ap.add_argument("--end-window-k", type=int, default=12)
    ap.add_argument("--future-horizon-n", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--oracle-top-q", type=float, default=0.10)
    ap.add_argument("--random-k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    if args.end_window_k < 2:
        raise RuntimeError("--end-window-k must be >= 2")
    if args.future_horizon_n < 1:
        raise RuntimeError("--future-horizon-n must be >= 1")
    if args.top_k < 1:
        raise RuntimeError("--top-k must be >= 1")
    if not (0.0 < float(args.oracle_top_q) <= 1.0):
        raise RuntimeError("--oracle-top-q must be in (0,1]")

    date_tag = datetime.now().strftime("%Y%m%d")
    internal_out = args.internal_out_dir or Path(f"debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_{date_tag}")
    p4_alt_out = args.p4_alt_out_dir or Path(f"debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_future_equivalence_{date_tag}")
    internal_out.mkdir(parents=True, exist_ok=True)
    p4_alt_out.mkdir(parents=True, exist_ok=True)

    z_path = Path(args.z_features)
    if not z_path.exists():
        raise FileNotFoundError(f"missing z-features: {z_path}")
    p1_path = Path(DEFAULT_P1)
    p4_path = Path(DEFAULT_P4)
    summary_path = Path(DEFAULT_SUMMARY)
    p1 = _load_json(p1_path) if p1_path.exists() else {}
    p4 = _load_json(p4_path) if p4_path.exists() else {}

    npz = np.load(z_path, allow_pickle=True)
    clip_order = [
        _npz_scalar_to_text(x) for x in np.asarray(npz["clip_order"], dtype=object).tolist()
    ] if "clip_order" in npz.files else list(LOCKED_CLIPS)

    z_by_clip: dict[str, np.ndarray] = {}
    hidden_pre_by_clip: dict[str, np.ndarray | None] = {}
    future_desc_by_clip: dict[str, np.ndarray] = {}
    feature_meta: dict[str, Any] = {}

    for clip in LOCKED_CLIPS:
        z = np.asarray(npz[f"{clip}__z"], dtype=np.float64)
        if z.ndim != 2:
            raise RuntimeError(f"{clip}__z must be rank-2")
        hidden = np.asarray(npz[f"{clip}__hidden_pre"]) if f"{clip}__hidden_pre" in npz.files else None
        future = np.asarray(npz[f"{clip}__future_desc"], dtype=np.float64)
        if future.ndim != 2:
            raise RuntimeError(f"{clip}__future_desc must be rank-2")
        z_by_clip[clip] = z
        hidden_pre_by_clip[clip] = hidden
        future_desc_by_clip[clip] = future
        feature_meta[clip] = {
            "z": {"shape": [int(z.shape[0]), int(z.shape[1])], "dtype": str(np.asarray(npz[f"{clip}__z"]).dtype), "device": "cpu"},
            "hidden_pre": {
                "present": bool(hidden is not None),
                "shape": [int(hidden.shape[0]), int(hidden.shape[1])] if hidden is not None else None,
                "dtype": str(hidden.dtype) if hidden is not None else None,
                "device": "cpu" if hidden is not None else None,
            },
            "future_desc": {"shape": [int(future.shape[0]), int(future.shape[1])], "dtype": str(np.asarray(npz[f"{clip}__future_desc"]).dtype), "device": "cpu"},
        }

    # internal_structure_v2 (preserve P2/P3 core numbers)
    walk_f = z_by_clip["Walk_F"]
    t_f = int(walk_f.shape[0])
    mid = t_f // 2
    endpoint_distance = _l2(walk_f[0], walk_f[t_f - 1])
    half_cycle_distance = _l2(walk_f[0], walk_f[mid])
    cycle_closure_ratio = float(endpoint_distance / max(half_cycle_distance, EPS))
    knn = _knn_temporal_consistency(walk_f, k=5, exclude_window=2)
    pca2d, pca_evr = _pca_2d(walk_f)
    pca_var2 = float(np.sum(pca_evr[:2], dtype=np.float64))
    pca_loop_area = _polygon_area(pca2d)

    p3_per_clip: dict[str, Any] = {}
    mono_count = 0
    slope_neg_count = 0
    ratio_vals: list[float] = []
    end_vecs: list[np.ndarray] = []
    end_k = int(args.end_window_k)
    for clip in TURN_CLIPS:
        z = z_by_clip[clip]
        t = int(z.shape[0])
        k = min(end_k, t)
        final = z[t - 1]
        s = t - k
        wnd = z[s:t]
        dist = np.sqrt(np.sum((wnd - final[None, :]) ** 2, axis=1, dtype=np.float64), dtype=np.float64)
        mono = float(np.sum(dist[1:] <= dist[:-1]) / max(k - 1, 1))
        slope = float(np.polyfit(np.arange(k, dtype=np.float64), dist, deg=1)[0])
        end_mean = np.mean(wnd, axis=0, keepdims=True)
        end_var = float(np.mean(np.sum((wnd - end_mean) ** 2, axis=1, dtype=np.float64), dtype=np.float64))
        ms, me = _turn_mid_window_indices(t, k)
        mwnd = z[ms:me]
        if mwnd.shape[0] != k:
            mwnd = z[max(0, me - k):me]
        mid_mean = np.mean(mwnd, axis=0, keepdims=True)
        mid_var = float(np.mean(np.sum((mwnd - mid_mean) ** 2, axis=1, dtype=np.float64), dtype=np.float64))
        ratio = float(end_var / max(mid_var, EPS))
        if mono > 0.60:
            mono_count += 1
        if slope < 0.0:
            slope_neg_count += 1
        ratio_vals.append(ratio)
        end_vecs.append(z[t - 1])
        p3_per_clip[clip] = {
            "num_frames": int(t),
            "effective_k": int(k),
            "end_window_frame_range": [int(s), int(t - 1)],
            "dist_to_final": {
                "start_distance": float(dist[0]),
                "end_distance": float(dist[-1]),
                "mean_distance": float(np.mean(dist, dtype=np.float64)),
                "monotonic_fraction": mono,
                "slope": slope,
            },
            "end_window_variance": end_var,
            "mid_window_frame_range": [int(ms), int(me - 1)],
            "mid_window_variance": mid_var,
            "end_vs_mid_variance_ratio": ratio,
        }

    rng = np.random.default_rng(int(args.seed))
    rand_mid_vecs: list[np.ndarray] = []
    rand_mid_idx: dict[str, int] = {}
    for clip in TURN_CLIPS:
        z = z_by_clip[clip]
        idx = _random_middle_index(int(z.shape[0]), end_k, rng)
        rand_mid_idx[clip] = idx
        rand_mid_vecs.append(z[idx])

    cross_end = _pairwise_mean_distance(end_vecs)
    cross_rand = _pairwise_mean_distance(rand_mid_vecs)
    tight_ratio = float(cross_end / max(cross_rand, EPS))
    mean_ratio = float(np.mean(np.asarray(ratio_vals, dtype=np.float64), dtype=np.float64))

    p2 = {
        "clip": "Walk_F",
        "num_frames": int(t_f),
        "mid_frame_index": int(mid),
        "metrics": {
            "cycle_closure_ratio": cycle_closure_ratio,
            "endpoint_distance": endpoint_distance,
            "half_cycle_distance": half_cycle_distance,
            "knn_temporal_consistency": knn,
            "pca_2d_explained_variance": pca_var2,
            "pca_loop_area": pca_loop_area,
        },
        "diagnostic_focus": "cycle-data-agnostic structural diagnostics (no hard pass/fail)",
    }
    p3 = {
        "turn_clips": list(TURN_CLIPS),
        "end_window_k": int(end_k),
        "per_clip": p3_per_clip,
        "aggregate": {
            "majority_threshold": int(math.floor(len(TURN_CLIPS) / 2) + 1),
            "monotonic_fraction_gt_0p60_count": int(mono_count),
            "slope_negative_count": int(slope_neg_count),
            "mean_end_vs_mid_variance_ratio": mean_ratio,
            "cross_turn_end_tightness": cross_end,
            "cross_turn_random_tightness": cross_rand,
            "end_tightness_ratio": tight_ratio,
            "random_middle_frame_index_seeded": rand_mid_idx,
        },
        "diagnostic_focus": "monotonic convergence and cross-turn end tightness first (no hard pass/fail)",
    }
    internal_diag = _internal_v2_interpretation(p2, p3)

    p1_arms = p1.get("arms", {}) if isinstance(p1, dict) else {}
    p1_energy = p1_arms.get("energy_scalar", {}).get("loss", {}).get("test_weighted_huber")
    p1_raw = p1_arms.get("raw_hidden_pre", {}).get("loss", {}).get("test_weighted_huber")
    p1_z = p1_arms.get("z_bottleneck", {}).get("loss", {}).get("test_weighted_huber")

    internal_json = {
        "task": "internal_structure_v2",
        "input_artifact_path": str(z_path.resolve()),
        "context_artifacts": {
            "p1_predictive_compare": str(p1_path.resolve()) if p1_path.exists() else None,
            "p4_cross_clip_entry": str(p4_path.resolve()) if p4_path.exists() else None,
            "existing_summary": str(summary_path.resolve()) if summary_path.exists() else None,
        },
        "locked_clips": list(LOCKED_CLIPS),
        "feature_metadata": {
            "clip_order_from_npz": clip_order,
            "per_clip": feature_meta,
        },
        "metric_definitions": {
            "eps": EPS,
            "p2": {
                "cycle_closure_ratio": "dist(z_0, z_Tminus1)/max(dist(z_0,z_mid),eps)",
                "endpoint_distance": "dist(z_0, z_Tminus1)",
                "half_cycle_distance": "dist(z_0, z_mid)",
                "knn_temporal_consistency": "k=5 NN in z-space, exclude self and temporal ±2",
                "pca_2d_explained_variance": "Walk_F PCA top-2 EVR sum",
                "pca_loop_area": "Walk_F PCA2D polygon area",
            },
            "p3": {
                "dist_to_final": "||z_t - z_Tminus1|| for last K frames",
                "monotonic_fraction": "descending-step fraction in last K",
                "slope": "linear slope of dist_to_final in last K",
                "end_vs_mid_variance_ratio": "end_window_variance / max(mid_window_variance, eps)",
                "cross_turn_end_tightness": "pairwise mean distance among 4 turn final z",
                "end_tightness_ratio": "cross_turn_end_tightness / max(cross_turn_random_tightness, eps)",
            },
        },
        "p1_risk_context": {
            "energy_test_weighted_huber": p1_energy,
            "raw_hidden_pre_test_weighted_huber": p1_raw,
            "z_bottleneck_test_weighted_huber": p1_z,
            "note": "P1 remains an unresolved predictive sufficiency risk; internal diagnostics do not override it.",
        },
        "p2": p2,
        "p3": p3,
        "diagnostic_interpretation": internal_diag,
        "note": "internal_structure_v2 keeps original P2/P3 numbers but removes hard pass/fail gate semantics.",
    }

    internal_json_path = internal_out / "internal_structure_v2_summary.json"
    _dump_json(internal_json_path, internal_json)

    internal_md = [
        "# Action Handoff z Probe v1 Internal Structure v2",
        "",
        "## P1 risk remains (from existing artifact)",
        f"- z_bottleneck={_format(p1_z)} vs energy={_format(p1_energy)} vs raw_hidden_pre={_format(p1_raw)}.",
        "- 结论：predictive sufficiency 风险仍在，internal v2 不覆盖 P1。",
        "",
        "## P2 diagnostics (cycle-data-agnostic)",
        f"- cycle_closure_ratio={_format(cycle_closure_ratio)}, endpoint={_format(endpoint_distance)}, half_cycle={_format(half_cycle_distance)}",
        f"- knn.mean={_format(knn['mean'])}, knn.p50={_format(knn['p50'])}, pca_2d_explained_variance={_format(pca_var2)}, pca_loop_area={_format(pca_loop_area)}",
        f"- interpretation axes: phase_locality={internal_diag['axes']['phase_locality']}, cycle_closure={internal_diag['axes']['cycle_closure']}",
        "",
        "## P3 diagnostics (monotonic convergence + cross-turn end tightness)",
        f"- monotonic>0.60 count={mono_count}/{len(TURN_CLIPS)}, slope<0 count={slope_neg_count}/{len(TURN_CLIPS)}",
        f"- mean_end_vs_mid_variance_ratio={_format(mean_ratio)}, end_tightness_ratio={_format(tight_ratio)}",
        f"- interpretation axes: turn_monotonic_convergence={internal_diag['axes']['turn_monotonic_convergence']}, cross_turn_end_tightness={internal_diag['axes']['cross_turn_end_tightness']}, end_window_variance_vs_mid={internal_diag['axes']['end_window_variance_vs_mid']}",
        "",
        "## Overall",
        f"- structure_status={internal_diag['status']} (diagnostic-only, no hard pass/fail)",
    ]
    _dump_md(internal_out / "internal_structure_v2_summary.md", internal_md)

    # P4-alt future-equivalence probe (no MM oracle)
    horizon_n = int(args.future_horizon_n)
    top_k = int(args.top_k)
    oracle_q = float(args.oracle_top_q)
    random_k = int(args.random_k)

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
        win = np.stack([desc[i : i + horizon_n].reshape(-1) for i in range(valid_t)], axis=0).astype(np.float64, copy=False)
        sig_by_clip[clip] = win
        z_valid_by_clip[clip] = z[:valid_t].astype(np.float64, copy=False)
        valid_frames_by_clip[clip] = np.arange(valid_t, dtype=np.int64)

    rng2 = np.random.default_rng(int(args.seed))
    per_query_rows: list[dict[str, Any]] = []
    per_pair_bucket: dict[str, dict[str, list[float]]] = {}
    per_source_bucket: dict[str, dict[str, list[float]]] = {}

    for src in LOCKED_CLIPS:
        for tgt in LOCKED_CLIPS:
            if tgt == src:
                continue
            pair = f"{src}->{tgt}"
            zs = z_valid_by_clip[src]
            zt = z_valid_by_clip[tgt]
            fs = sig_by_clip[src]
            ft = sig_by_clip[tgt]

            zdist = _cosine_distance_matrix(zs, zt)
            fdist = _pairwise_l2_matrix(fs, ft)

            src_t, tgt_t = int(zdist.shape[0]), int(zdist.shape[1])
            k_eff = min(top_k, tgt_t)
            q_count = max(1, int(math.ceil(oracle_q * float(tgt_t))))
            pair_bucket = per_pair_bucket.setdefault(pair, {
                "top1_future": [], "topk_future": [], "rand_future": [], "top1_hit": [], "topk_hit": [],
                "spearman": [], "pearson": [], "oracle_count": [], "candidate_count": [],
            })
            src_bucket = per_source_bucket.setdefault(src, {
                "top1_future": [], "topk_future": [], "rand_future": [], "top1_hit": [], "topk_hit": [],
                "spearman": [], "pearson": [], "oracle_count": [], "candidate_count": [],
            })

            for i in range(src_t):
                zd = zdist[i]
                fd = fdist[i]
                order_z = np.argsort(zd, kind="stable")
                topk_idx = order_z[:k_eff]
                top1_idx = int(topk_idx[0])

                order_f = np.argsort(fd, kind="stable")
                oracle_idx = order_f[:q_count]
                oracle_set = set(int(x) for x in oracle_idx.tolist())
                oracle_best = int(order_f[0])

                rand_size = min(max(random_k, k_eff), tgt_t)
                rand_idx = rng2.choice(np.arange(tgt_t, dtype=np.int64), size=rand_size, replace=False)

                top1_future = float(fd[top1_idx])
                topk_future = float(np.mean(fd[topk_idx], dtype=np.float64))
                rand_future = float(np.mean(fd[rand_idx], dtype=np.float64))

                top1_hit = 1.0 if top1_idx == oracle_best else 0.0
                topk_hit = 1.0 if any(int(x) in oracle_set for x in topk_idx.tolist()) else 0.0

                spear = _spearman_corr(zd.astype(np.float64), fd.astype(np.float64))
                pear = _pearson_corr(zd.astype(np.float64), fd.astype(np.float64))

                row = {
                    "source_clip": src,
                    "target_clip": tgt,
                    "source_valid_frame": int(i),
                    "candidate_count": int(tgt_t),
                    "oracle_q_count": int(q_count),
                    "top1_future_distance": top1_future,
                    "topk_future_distance": topk_future,
                    "random_future_distance": rand_future,
                    "top1_equiv_hit": float(top1_hit),
                    "topk_equiv_hit": float(topk_hit),
                    "spearman_zdist_vs_futuredist": spear,
                    "pearson_zdist_vs_futuredist": pear,
                }
                per_query_rows.append(row)

                for b in (pair_bucket, src_bucket):
                    b["top1_future"].append(top1_future)
                    b["topk_future"].append(topk_future)
                    b["rand_future"].append(rand_future)
                    b["top1_hit"].append(float(top1_hit))
                    b["topk_hit"].append(float(topk_hit))
                    b["spearman"].append(spear)
                    b["pearson"].append(pear)
                    b["oracle_count"].append(float(q_count))
                    b["candidate_count"].append(float(tgt_t))

    def _summ(bucket: dict[str, list[float]]) -> dict[str, Any]:
        top1_f = _mean_finite(bucket["top1_future"])
        topk_f = _mean_finite(bucket["topk_future"])
        rand_f = _mean_finite(bucket["rand_future"])
        top1_hit = _mean_finite(bucket["top1_hit"])
        topk_hit = _mean_finite(bucket["topk_hit"])
        return {
            "num_queries": int(len(bucket["top1_future"])),
            "mean_top1_future_distance": top1_f,
            "mean_topk_future_distance": topk_f,
            "mean_random_future_distance": rand_f,
            "top1_future_distance_vs_random_ratio": (float(top1_f / max(rand_f, EPS)) if top1_f is not None and rand_f is not None else None),
            "topk_future_distance_vs_random_ratio": (float(topk_f / max(rand_f, EPS)) if topk_f is not None and rand_f is not None else None),
            "top1_equiv_hit_rate": top1_hit,
            "topk_equiv_hit_rate": topk_hit,
            "mean_spearman_zdist_vs_futuredist": _mean_finite(bucket["spearman"]),
            "std_spearman_zdist_vs_futuredist": _std_finite(bucket["spearman"]),
            "mean_pearson_zdist_vs_futuredist": _mean_finite(bucket["pearson"]),
            "mean_oracle_q_count": _mean_finite(bucket["oracle_count"]),
            "mean_candidate_count": _mean_finite(bucket["candidate_count"]),
        }

    all_bucket = {
        "top1_future": [], "topk_future": [], "rand_future": [], "top1_hit": [], "topk_hit": [],
        "spearman": [], "pearson": [], "oracle_count": [], "candidate_count": [],
    }
    for row in per_query_rows:
        all_bucket["top1_future"].append(float(row["top1_future_distance"]))
        all_bucket["topk_future"].append(float(row["topk_future_distance"]))
        all_bucket["rand_future"].append(float(row["random_future_distance"]))
        all_bucket["top1_hit"].append(float(row["top1_equiv_hit"]))
        all_bucket["topk_hit"].append(float(row["topk_equiv_hit"]))
        all_bucket["spearman"].append(float(row["spearman_zdist_vs_futuredist"]))
        all_bucket["pearson"].append(float(row["pearson_zdist_vs_futuredist"]))
        all_bucket["oracle_count"].append(float(row["oracle_q_count"]))
        all_bucket["candidate_count"].append(float(row["candidate_count"]))

    agg = _summ(all_bucket)
    random_top1_expectation = float(1.0 / max(agg["mean_candidate_count"], 1.0)) if agg["mean_candidate_count"] is not None else None
    if agg["top1_equiv_hit_rate"] is None or random_top1_expectation is None:
        hit_vs_rand = None
    else:
        hit_vs_rand = float(agg["top1_equiv_hit_rate"] - random_top1_expectation)
    agg["random_top1_expectation"] = random_top1_expectation
    agg["top1_equiv_hit_rate_vs_random_top1"] = hit_vs_rand

    p4_alt_diag = _p4_alt_interpretation(agg)

    p4_alt_json = {
        "task": "P4-alt cross-clip future-equivalence probe (no MM oracle)",
        "input_artifact_path": str(z_path.resolve()),
        "locked_clips": list(LOCKED_CLIPS),
        "probe_definition": {
            "question": "Do cross-clip pairs that are near in z also have near GT future_desc over next N frames?",
            "z_distance": "cosine distance (1-cos)",
            "future_equivalence_distance": "L2 distance between flattened future_desc windows of length N",
            "future_horizon_n": int(horizon_n),
            "top_k": int(top_k),
            "oracle_top_q": float(oracle_q),
            "random_k": int(random_k),
            "seed": int(args.seed),
            "no_mm_oracle": True,
        },
        "p1_risk_context": {
            "energy_test_weighted_huber": p1_energy,
            "raw_hidden_pre_test_weighted_huber": p1_raw,
            "z_bottleneck_test_weighted_huber": p1_z,
            "note": "P1 risk remains independent of P4-alt outcome.",
        },
        "aggregate": agg,
        "per_source_clip": {clip: _summ(b) for clip, b in sorted(per_source_bucket.items())},
        "per_pair": {pair: _summ(b) for pair, b in sorted(per_pair_bucket.items())},
        "diagnostic_interpretation": p4_alt_diag,
        "next_path": {
            "if_supports_recalibration": "rewrite P4 around future-equivalence rather than MM-oracle agreement; hold beta/Dz ablation",
            "if_does_not_support_recalibration": "return to beta/Dz and loss-formulation ablation",
            "if_inconclusive": "add targeted controls (future horizon sweep / distance normalization / query subseting) before deciding",
        },
    }
    p4_alt_json_path = p4_alt_out / "p4_alt_future_equivalence_summary.json"
    _dump_json(p4_alt_json_path, p4_alt_json)

    p4_alt_md = [
        "# Action Handoff z Probe v1 P4-alt (Future-Equivalence)",
        "",
        "## Setup",
        "- No MM oracle ground truth.",
        f"- future_horizon_n={horizon_n}, top_k={top_k}, oracle_top_q={_format(oracle_q)}, random_k={random_k}, seed={args.seed}",
        "- 问题：z 近的 cross-clip pair，其 GT future_desc 后续 N 帧是否也近。",
        "",
        "## Aggregate",
        f"- mean_top1_future_distance={_format(agg['mean_top1_future_distance'])}",
        f"- mean_topk_future_distance={_format(agg['mean_topk_future_distance'])}",
        f"- mean_random_future_distance={_format(agg['mean_random_future_distance'])}",
        f"- top1_vs_random_ratio={_format(agg['top1_future_distance_vs_random_ratio'])}, topk_vs_random_ratio={_format(agg['topk_future_distance_vs_random_ratio'])}",
        f"- top1_equiv_hit_rate={_format(agg['top1_equiv_hit_rate'])}, random_top1_expectation={_format(agg['random_top1_expectation'])}, delta={_format(agg['top1_equiv_hit_rate_vs_random_top1'])}",
        f"- mean_spearman={_format(agg['mean_spearman_zdist_vs_futuredist'])}, mean_pearson={_format(agg['mean_pearson_zdist_vs_futuredist'])}",
        "",
        "## Interpretation",
        f"- status={p4_alt_diag['status']}",
        f"- {p4_alt_diag['note']}",
        "",
        "## Path",
        "- supports_recalibration: 重写 P4，暂缓 beta/Dz ablation",
        "- does_not_support_recalibration: 回到 beta/Dz / loss formulation",
        "- inconclusive: 先补控制实验再决策",
    ]
    _dump_md(p4_alt_out / "p4_alt_future_equivalence_summary.md", p4_alt_md)

    print(f"[ok] internal_structure_v2: {internal_json_path}")
    print(f"[ok] p4_alt: {p4_alt_json_path}")
    print(f"[ok] internal_status={internal_diag['status']} p4_alt_status={p4_alt_diag['status']}")


if __name__ == "__main__":
    main()
