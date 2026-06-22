#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _format_float(v: float | None, digits: int = 6) -> str:
    if v is None or not np.isfinite(v):
        return "null"
    return f"{float(v):.{digits}f}"


def _npz_scalar_to_text(v: Any) -> str:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    return str(v)


def _vector_l2(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.sum(diff * diff, dtype=np.float64), dtype=np.float64))


def _pairwise_mean_distance(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0
    dists: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dists.append(_vector_l2(vectors[i], vectors[j]))
    return float(np.mean(np.asarray(dists, dtype=np.float64), dtype=np.float64)) if dists else 0.0


def _polygon_area(points2d: np.ndarray) -> float:
    if points2d.shape[0] < 3:
        return 0.0
    x = points2d[:, 0]
    y = points2d[:, 1]
    x2 = np.concatenate([x, x[:1]], axis=0)
    y2 = np.concatenate([y, y[:1]], axis=0)
    return float(0.5 * abs(np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1], dtype=np.float64)))


def _pca_2d(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if z.ndim != 2:
        raise RuntimeError(f"PCA expects rank-2 tensor, got shape={tuple(z.shape)}")
    z64 = np.asarray(z, dtype=np.float64)
    zc = z64 - np.mean(z64, axis=0, keepdims=True)
    _, svals, v_t = np.linalg.svd(zc, full_matrices=False)
    if svals.size <= 0:
        raise RuntimeError("PCA got empty singular values")
    variances = (svals * svals) / max(float(zc.shape[0] - 1), 1.0)
    total = float(np.sum(variances, dtype=np.float64))
    if total <= 0.0:
        evr = np.zeros_like(variances)
    else:
        evr = variances / total
    pc2 = v_t[:2].T
    proj2 = zc @ pc2
    return proj2.astype(np.float64, copy=False), evr.astype(np.float64, copy=False)


def _knn_temporal_consistency(z: np.ndarray, *, k: int, exclude_window: int) -> dict[str, Any]:
    t = int(z.shape[0])
    if t <= 1:
        raise RuntimeError("kNN temporal consistency requires at least 2 frames")

    dists_mat = np.sqrt(
        np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2, dtype=np.float64),
        dtype=np.float64,
    )

    normalized_circular_distances: list[float] = []
    neighbor_count_per_query: list[int] = []

    for i in range(t):
        exclude = np.zeros(t, dtype=bool)
        exclude[i] = True
        for dt in range(-exclude_window, exclude_window + 1):
            j = i + dt
            if 0 <= j < t:
                exclude[j] = True

        valid_idx = np.where(~exclude)[0]
        if valid_idx.size <= 0:
            neighbor_count_per_query.append(0)
            continue
        local_d = dists_mat[i, valid_idx]
        order = np.argsort(local_d)
        kk = int(min(k, order.size))
        chosen = valid_idx[order[:kk]]
        neighbor_count_per_query.append(int(kk))
        for j in chosen:
            delta = abs(int(i) - int(j))
            circular_delta = min(delta, t - delta)
            normalized_circular_distances.append(float(circular_delta / max(float(t), 1.0)))

    arr = np.asarray(normalized_circular_distances, dtype=np.float64)
    if arr.size <= 0:
        return {
            "k": int(k),
            "exclude_temporal_window": int(exclude_window),
            "count": 0,
            "mean": None,
            "p50": None,
            "neighbor_count_per_query": {
                "mean": float(np.mean(neighbor_count_per_query)) if neighbor_count_per_query else 0.0,
                "min": int(min(neighbor_count_per_query)) if neighbor_count_per_query else 0,
                "max": int(max(neighbor_count_per_query)) if neighbor_count_per_query else 0,
            },
        }

    return {
        "k": int(k),
        "exclude_temporal_window": int(exclude_window),
        "count": int(arr.size),
        "mean": float(np.mean(arr, dtype=np.float64)),
        "p50": float(np.percentile(arr, 50.0)),
        "neighbor_count_per_query": {
            "mean": float(np.mean(np.asarray(neighbor_count_per_query, dtype=np.float64), dtype=np.float64)),
            "min": int(min(neighbor_count_per_query)),
            "max": int(max(neighbor_count_per_query)),
        },
    }


def _p2_interpretation(*, cycle_closure_ratio: float, knn_mean: float, pca_var2: float) -> str:
    pass_like = (
        (cycle_closure_ratio < 0.35)
        and (knn_mean < 0.20)
        and (pca_var2 > 0.50)
    )
    if pass_like:
        return "pass_like"

    clear_fail = (
        int(cycle_closure_ratio >= 0.50)
        + int(knn_mean >= 0.30)
        + int(pca_var2 <= 0.40)
    )
    if clear_fail >= 2:
        return "fail_like"
    return "inconclusive"


def _turn_mid_window_indices(length: int, k: int) -> tuple[int, int]:
    mid_center = length // 2
    start = max(0, mid_center - (k // 2))
    end = start + k
    if end > length:
        end = length
        start = max(0, end - k)
    return int(start), int(end)


def _random_middle_index(length: int, end_window_k: int, rng: np.random.Generator) -> int:
    end_start = max(0, length - end_window_k)
    low = max(0, length // 4)
    high = min(end_start, (3 * length) // 4)
    candidates = np.arange(low, high, dtype=np.int64)
    if candidates.size <= 0:
        candidates = np.arange(0, max(end_start, 1), dtype=np.int64)
    if candidates.size <= 0:
        return int(length - 1)
    return int(rng.choice(candidates))


def _p3_interpretation(
    *,
    monotonic_gt_0p60_count: int,
    slope_negative_count: int,
    n_clips: int,
    mean_end_vs_mid_variance_ratio: float,
    end_tightness_ratio: float,
) -> str:
    majority = int(math.floor(n_clips / 2) + 1)
    pass_like = (
        monotonic_gt_0p60_count >= majority
        and slope_negative_count >= majority
        and mean_end_vs_mid_variance_ratio < 0.75
        and end_tightness_ratio < 0.75
    )
    if pass_like:
        return "pass_like"

    clear_fail = (
        int(monotonic_gt_0p60_count < majority)
        + int(slope_negative_count < majority)
        + int(mean_end_vs_mid_variance_ratio >= 0.95)
        + int(end_tightness_ratio >= 0.95)
    )
    if clear_fail >= 3:
        return "fail_like"
    return "inconclusive"


def _next_decision(*, p2_status: str, p3_status: str, p4_fail: bool) -> str:
    if p2_status == "pass_like" and p3_status == "pass_like" and p4_fail:
        return "re-examine P4 yardstick before beta/Dz ablation"
    if p2_status == "fail_like" and p3_status == "fail_like":
        return "continue beta/Dz focused ablation"
    if p2_status == "pass_like" and p3_status == "fail_like":
        return "diagnose turn/end representation"
    if p2_status == "fail_like" and p3_status == "pass_like":
        return "inconclusive, analyze separately"
    return "inconclusive, collect additional internal-structure evidence before deciding beta/Dz"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe v1 z internal structure (P2/P3) without retraining or beta/Dz sweep."
    )
    parser.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--end-window-k", type=int, default=12)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.end_window_k) <= 1:
        raise RuntimeError("--end-window-k must be >= 2")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir if args.out_dir is not None else Path(
        f"debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_{date_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    z_features_path = Path(args.z_features)
    if not z_features_path.exists():
        raise FileNotFoundError(f"z-features not found: {z_features_path}")

    p1_path = Path(DEFAULT_P1)
    p4_path = Path(DEFAULT_P4)
    summary_path = Path(DEFAULT_SUMMARY)
    if not p1_path.exists():
        raise FileNotFoundError(f"P1 artifact missing: {p1_path}")
    if not p4_path.exists():
        raise FileNotFoundError(f"P4 artifact missing: {p4_path}")

    p1 = _load_json(p1_path)
    p4 = _load_json(p4_path)

    npz = np.load(z_features_path, allow_pickle=True)
    clip_order = [
        _npz_scalar_to_text(x) for x in np.asarray(npz["clip_order"], dtype=object).tolist()
    ] if "clip_order" in npz.files else list(LOCKED_CLIPS)

    missing = [clip for clip in LOCKED_CLIPS if f"{clip}__z" not in npz.files]
    if missing:
        raise RuntimeError(f"z features missing clips: {missing}")

    z_by_clip: dict[str, np.ndarray] = {}
    feature_meta: dict[str, Any] = {}

    for clip in LOCKED_CLIPS:
        z_key = f"{clip}__z"
        z = np.asarray(npz[z_key])
        if z.ndim != 2:
            raise RuntimeError(f"{z_key} must be rank-2, got shape={tuple(z.shape)}")
        if not np.all(np.isfinite(z)):
            raise RuntimeError(f"{z_key} contains non-finite values")
        z_by_clip[clip] = z.astype(np.float64, copy=False)

        hidden_key = f"{clip}__hidden_pre"
        hidden = np.asarray(npz[hidden_key]) if hidden_key in npz.files else None
        if hidden is not None and hidden.ndim != 2:
            raise RuntimeError(f"{hidden_key} must be rank-2 when present, got shape={tuple(hidden.shape)}")
        if hidden is not None and not np.all(np.isfinite(hidden)):
            raise RuntimeError(f"{hidden_key} contains non-finite values")

        feature_meta[clip] = {
            "z": {
                "shape": [int(z.shape[0]), int(z.shape[1])],
                "dtype": str(z.dtype),
                "device": "cpu",
            },
            "hidden_pre": {
                "present": bool(hidden is not None),
                "shape": [int(hidden.shape[0]), int(hidden.shape[1])] if hidden is not None else None,
                "dtype": str(hidden.dtype) if hidden is not None else None,
                "device": "cpu" if hidden is not None else None,
            },
        }

    walk_f = z_by_clip["Walk_F"]
    t_f = int(walk_f.shape[0])
    mid_idx = int(t_f // 2)

    endpoint_distance = _vector_l2(walk_f[0], walk_f[t_f - 1])
    half_cycle_distance = _vector_l2(walk_f[0], walk_f[mid_idx])
    cycle_closure_ratio = float(endpoint_distance / max(half_cycle_distance, EPS))

    knn = _knn_temporal_consistency(walk_f, k=5, exclude_window=2)
    pca_2d, pca_evr = _pca_2d(walk_f)
    pca_var2 = float(np.sum(pca_evr[:2], dtype=np.float64))
    pca_loop_area = _polygon_area(pca_2d)
    knn_mean = float(knn["mean"]) if knn["mean"] is not None else float("nan")

    p2_status = _p2_interpretation(
        cycle_closure_ratio=cycle_closure_ratio,
        knn_mean=knn_mean,
        pca_var2=pca_var2,
    )

    # P2 plot
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(pca_2d[:, 0], pca_2d[:, 1], "-", linewidth=1.5, color="#1f77b4", label="Walk_F z trajectory")
    ax.scatter([pca_2d[0, 0]], [pca_2d[0, 1]], c="#2ca02c", s=70, marker="o", label="start")
    ax.scatter([pca_2d[mid_idx, 0]], [pca_2d[mid_idx, 1]], c="#ff7f0e", s=70, marker="^", label="mid")
    ax.scatter([pca_2d[t_f - 1, 0]], [pca_2d[t_f - 1, 1]], c="#d62728", s=70, marker="s", label="end")
    ax.set_title("Walk_F z PCA(2D) trajectory")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    walk_f_pca_png = out_dir / "walk_f_z_pca.png"
    fig.savefig(walk_f_pca_png, dpi=160)
    plt.close(fig)

    end_k = int(args.end_window_k)
    p3_per_clip: dict[str, Any] = {}
    monotonic_gt_0p60 = 0
    slope_negative = 0
    ratio_values: list[float] = []
    end_vectors: list[np.ndarray] = []

    # curve plot (store first)
    curve_payload: dict[str, dict[str, Any]] = {}

    for clip in TURN_CLIPS:
        z = z_by_clip[clip]
        t = int(z.shape[0])
        k = min(end_k, t)
        if k < 2:
            raise RuntimeError(f"{clip}: effective K is {k}, cannot compute slope/monotonic")

        final = z[t - 1]
        start_idx = t - k
        end_window = z[start_idx:t]
        dist_to_final = np.sqrt(np.sum((end_window - final[None, :]) ** 2, axis=1, dtype=np.float64), dtype=np.float64)

        decreases = int(np.sum(dist_to_final[1:] <= dist_to_final[:-1]))
        monotonic_fraction = float(decreases / max(k - 1, 1))
        frame_x = np.arange(k, dtype=np.float64)
        slope = float(np.polyfit(frame_x, dist_to_final, deg=1)[0])

        end_mean = np.mean(end_window, axis=0, keepdims=True)
        end_window_variance = float(np.mean(np.sum((end_window - end_mean) ** 2, axis=1, dtype=np.float64), dtype=np.float64))

        mid_start, mid_end = _turn_mid_window_indices(t, k)
        mid_window = z[mid_start:mid_end]
        if mid_window.shape[0] != k:
            # keep exact K if possible by padding with nearest indices (defensive for very short clips)
            pad_from = z[max(0, mid_end - k):mid_end]
            mid_window = pad_from
        mid_mean = np.mean(mid_window, axis=0, keepdims=True)
        mid_window_variance = float(np.mean(np.sum((mid_window - mid_mean) ** 2, axis=1, dtype=np.float64), dtype=np.float64))

        ratio = float(end_window_variance / max(mid_window_variance, EPS))

        if monotonic_fraction > 0.60:
            monotonic_gt_0p60 += 1
        if slope < 0.0:
            slope_negative += 1
        ratio_values.append(ratio)
        end_vectors.append(z[t - 1])

        p3_per_clip[clip] = {
            "num_frames": int(t),
            "effective_k": int(k),
            "end_window_frame_range": [int(start_idx), int(t - 1)],
            "dist_to_final": {
                "start_distance": float(dist_to_final[0]),
                "end_distance": float(dist_to_final[-1]),
                "mean_distance": float(np.mean(dist_to_final, dtype=np.float64)),
                "monotonic_fraction": monotonic_fraction,
                "slope": slope,
            },
            "end_window_variance": end_window_variance,
            "mid_window_frame_range": [int(mid_start), int(mid_end - 1)],
            "mid_window_variance": mid_window_variance,
            "end_vs_mid_variance_ratio": ratio,
        }

        curve_payload[clip] = {
            "x": list(range(-k + 1, 1)),
            "y": [float(v) for v in dist_to_final.tolist()],
        }

    rng = np.random.default_rng(0)
    random_mid_vectors: list[np.ndarray] = []
    random_mid_frame_index: dict[str, int] = {}
    for clip in TURN_CLIPS:
        z = z_by_clip[clip]
        idx = _random_middle_index(int(z.shape[0]), end_k, rng)
        random_mid_vectors.append(z[idx])
        random_mid_frame_index[clip] = int(idx)

    cross_turn_end_tightness = _pairwise_mean_distance(end_vectors)
    cross_turn_random_tightness = _pairwise_mean_distance(random_mid_vectors)
    end_tightness_ratio = float(cross_turn_end_tightness / max(cross_turn_random_tightness, EPS))

    mean_ratio = float(np.mean(np.asarray(ratio_values, dtype=np.float64), dtype=np.float64))
    p3_status = _p3_interpretation(
        monotonic_gt_0p60_count=monotonic_gt_0p60,
        slope_negative_count=slope_negative,
        n_clips=len(TURN_CLIPS),
        mean_end_vs_mid_variance_ratio=mean_ratio,
        end_tightness_ratio=end_tightness_ratio,
    )

    # P3 curve plot
    fig2, ax2 = plt.subplots(figsize=(8.0, 5.2))
    for clip in TURN_CLIPS:
        xs = np.asarray(curve_payload[clip]["x"], dtype=np.int64)
        ys = np.asarray(curve_payload[clip]["y"], dtype=np.float64)
        ax2.plot(xs, ys, marker="o", linewidth=1.5, markersize=3.5, label=clip)
    ax2.set_title(f"Turn clips end-window dist_to_final (K={end_k})")
    ax2.set_xlabel("Frame offset to final (0 = final frame)")
    ax2.set_ylabel("L2 distance in z-space")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")
    fig2.tight_layout()
    turn_curve_png = out_dir / "turn_end_distance_curves.png"
    fig2.savefig(turn_curve_png, dpi=160)
    plt.close(fig2)

    p1_arms = p1.get("arms", {})
    p1_energy = p1_arms.get("energy_scalar", {}).get("loss", {}).get("test_weighted_huber")
    p1_raw = p1_arms.get("raw_hidden_pre", {}).get("loss", {}).get("test_weighted_huber")
    p1_z = p1_arms.get("z_bottleneck", {}).get("loss", {}).get("test_weighted_huber")

    p4_gate = p4.get("gate", {})
    p4_metrics = p4_gate.get("z_metrics", {})
    p4_pass = p4_gate.get("z_pass", {})
    p4_fail = not bool(p4_pass.get("pass_all", False))

    decision = _next_decision(p2_status=p2_status, p3_status=p3_status, p4_fail=p4_fail)

    summary = {
        "task": "Action handoff z probe v1 internal structure (P2/P3)",
        "input_artifact_path": str(z_features_path.resolve()),
        "context_artifacts": {
            "p1_predictive_compare": str(p1_path.resolve()),
            "p4_cross_clip_entry": str(p4_path.resolve()),
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
                "cycle_closure_ratio": "dist(z_0, z_Tminus1) / max(dist(z_0, z_mid), eps)",
                "endpoint_distance": "dist(z_0, z_Tminus1)",
                "half_cycle_distance": "dist(z_0, z_mid)",
                "knn_temporal_consistency": "k=5 nearest neighbors in z-space; exclude self and temporal window ±2; report circular temporal distance/T mean and p50",
                "pca_2d_explained_variance": "Walk_F PCA top-2 explained variance ratio sum",
                "pca_loop_area": "Polygon area of Walk_F PCA-2D temporal trajectory",
            },
            "p3": {
                "dist_to_final": "for each turn clip and final K frames, dist_to_final[t]=||z_t-z_Tminus1||",
                "monotonic_fraction": "fraction of descending steps in final K distances",
                "slope": "linear fit slope of frame index vs dist_to_final in final K",
                "end_window_variance": "mean squared distance to end-window mean in z-space",
                "mid_window_variance": "mean squared distance to middle-window mean in z-space",
                "end_vs_mid_variance_ratio": "end_window_variance / max(mid_window_variance, eps)",
                "cross_turn_end_tightness": "pairwise mean distance across final z of 4 turn clips",
                "cross_turn_random_tightness": "pairwise mean distance across one random middle frame per turn clip (seed=0)",
                "end_tightness_ratio": "cross_turn_end_tightness / max(cross_turn_random_tightness, eps)",
            },
        },
        "p1_p4_current_state": {
            "p1": {
                "energy_test_weighted_huber": p1_energy,
                "raw_hidden_pre_test_weighted_huber": p1_raw,
                "z_bottleneck_test_weighted_huber": p1_z,
                "status": "fail_like" if (p1_z is not None and p1_energy is not None and p1_z > p1_energy) else "unknown",
                "note": "From existing artifact; z test loss worse than baseline indicates current P1 failure state.",
            },
            "p4": {
                "z_metrics": p4_metrics,
                "z_pass": p4_pass,
                "status": "fail" if p4_fail else "pass",
                "note": "From existing artifact; global passes but runtime/aggregate fail in current v1 run.",
            },
        },
        "p2": {
            "clip": "Walk_F",
            "num_frames": int(t_f),
            "mid_frame_index": int(mid_idx),
            "metrics": {
                "cycle_closure_ratio": cycle_closure_ratio,
                "endpoint_distance": endpoint_distance,
                "half_cycle_distance": half_cycle_distance,
                "knn_temporal_consistency": knn,
                "pca_2d_explained_variance": pca_var2,
                "pca_loop_area": pca_loop_area,
            },
            "provisional_gate": {
                "rule": "pass_like iff cycle_closure_ratio<0.35 and knn.mean<0.20 and pca_2d_explained_variance>0.50",
                "interpretation": p2_status,
            },
            "artifact": str(walk_f_pca_png.resolve()),
        },
        "p3": {
            "turn_clips": list(TURN_CLIPS),
            "end_window_k": int(end_k),
            "per_clip": p3_per_clip,
            "aggregate": {
                "majority_threshold": int(math.floor(len(TURN_CLIPS) / 2) + 1),
                "monotonic_fraction_gt_0p60_count": int(monotonic_gt_0p60),
                "slope_negative_count": int(slope_negative),
                "mean_end_vs_mid_variance_ratio": mean_ratio,
                "cross_turn_end_tightness": cross_turn_end_tightness,
                "cross_turn_random_tightness": cross_turn_random_tightness,
                "end_tightness_ratio": end_tightness_ratio,
                "random_middle_frame_index_seed0": random_mid_frame_index,
            },
            "provisional_gate": {
                "rule": "pass_like iff majority monotonic_fraction>0.60 and majority slope<0 and mean end_vs_mid_variance_ratio<0.75 and end_tightness_ratio<0.75",
                "interpretation": p3_status,
            },
            "artifact": str(turn_curve_png.resolve()),
        },
        "note": "P2/P3 do not override P1 failure; they only refine failure attribution.",
        "next_decision_matrix": {
            "P2 pass + P3 pass + P4 fail": "re-examine P4 yardstick before beta/Dz ablation",
            "P2/P3 fail": "continue beta/Dz focused ablation",
            "P2 pass + P3 fail": "diagnose turn/end representation",
            "P2 fail + P3 pass": "inconclusive, analyze separately",
        },
        "decision": decision,
    }

    json_path = out_dir / "internal_structure_summary.json"
    _dump_json(json_path, summary)

    lines: list[str] = []
    lines.append("# Action Handoff z Probe v1 Internal Structure (P2/P3)")
    lines.append("")
    lines.append("## Current P1/P4 state (from existing artifacts)")
    lines.append(
        f"- P1 predictive compare: z_bottleneck test_weighted_huber={_format_float(p1_z)} vs energy={_format_float(p1_energy)} vs raw_hidden_pre={_format_float(p1_raw)} (current state: fail-like)."
    )
    lines.append(
        f"- P4 gate: global={_format_float(p4_metrics.get('global'))}, runtime_topk={_format_float(p4_metrics.get('runtime_topk'))}, aggregate_bottom_q={_format_float(p4_metrics.get('aggregate_bottom_q'))}, pass_all={bool(p4_pass.get('pass_all', False))}."
    )
    lines.append("")
    lines.append("## P2 Walk_F phase auto-emergence")
    lines.append(f"- cycle_closure_ratio={_format_float(cycle_closure_ratio)} (threshold < 0.35)")
    lines.append(f"- endpoint_distance={_format_float(endpoint_distance)}")
    lines.append(f"- half_cycle_distance={_format_float(half_cycle_distance)}")
    lines.append(
        f"- knn_temporal_consistency.mean={_format_float(knn.get('mean'))}, p50={_format_float(knn.get('p50'))} (threshold mean < 0.20)"
    )
    lines.append(
        f"- pca_2d_explained_variance={_format_float(pca_var2)} (threshold > 0.50), pca_loop_area={_format_float(pca_loop_area)}"
    )
    lines.append(f"- provisional interpretation: **{p2_status}**")
    lines.append("")
    lines.append("## P3 turn end criterion detectability")
    for clip in TURN_CLIPS:
        m = p3_per_clip[clip]
        d = m["dist_to_final"]
        lines.append(
            "- "
            + f"{clip}: start_distance={_format_float(d['start_distance'])}, mean_distance={_format_float(d['mean_distance'])}, "
            + f"monotonic_fraction={_format_float(d['monotonic_fraction'])}, slope={_format_float(d['slope'])}, "
            + f"end_vs_mid_variance_ratio={_format_float(m['end_vs_mid_variance_ratio'])}"
        )
    lines.append(
        f"- aggregate: monotonic>0.60 count={monotonic_gt_0p60}/{len(TURN_CLIPS)}, slope<0 count={slope_negative}/{len(TURN_CLIPS)}, "
        f"mean_end_vs_mid_variance_ratio={_format_float(mean_ratio)}, end_tightness_ratio={_format_float(end_tightness_ratio)}"
    )
    lines.append(f"- provisional interpretation: **{p3_status}**")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- note: P2/P3 do not override P1 failure; they only refine failure attribution.")
    lines.append(f"- next step decision: **{decision}**")
    lines.append(
        "- beta/Dz ablation remains next step unless the decision is explicitly `re-examine P4 yardstick before beta/Dz ablation`."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {json_path.resolve()}")
    lines.append(f"- {(out_dir / 'walk_f_z_pca.png').resolve()}")
    lines.append(f"- {(out_dir / 'turn_end_distance_curves.png').resolve()}")

    md_path = out_dir / "internal_structure_summary.md"
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(f"[ok] wrote: {walk_f_pca_png}")
    print(f"[ok] wrote: {turn_curve_png}")
    print(f"[ok] P2={p2_status} P3={p3_status} decision={decision}")


if __name__ == "__main__":
    main()
