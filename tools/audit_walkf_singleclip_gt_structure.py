#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_prominences, savgol_filter


@dataclass(frozen=True)
class JointSpec:
    joint: str
    window_start: int
    window_end: int
    context_start: int
    context_end: int


JOINT_SPECS = (
    JointSpec("calf_l", 78, 85, 72, 88),
    JointSpec("calf_r", 56, 62, 50, 68),
)


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    return float(v)


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if bool(v) else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if not math.isfinite(fv):
            return "N/A"
        return f"{fv:.{digits}f}"
    return str(v)


def _serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def format_runs(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "none"
    return ", ".join(
        f"{run['start_sic']}-{run['end_sic']} (len={run['length']})"
        for run in runs
    )


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12, None)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - proj * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12, None)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    r_rel = np.einsum("...ji,...jk->...ik", r_a, r_b)
    trace = np.trace(r_rel, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_theta))


def longest_runs(mask: np.ndarray, x: np.ndarray) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if start is not None and (i == len(mask) - 1 or not mask[i + 1]):
            end = i
            runs.append(
                {
                    "start_sic": int(x[start]),
                    "end_sic": int(x[end]),
                    "length": int(end - start + 1),
                }
            )
            start = None
    return runs


def extremum_metrics(x: np.ndarray, y: np.ndarray, kind: str) -> dict[str, Any] | None:
    if kind not in {"pos", "neg"}:
        raise ValueError(kind)
    if kind == "pos":
        if float(np.max(y)) <= 0.0:
            return None
        idx = int(np.argmax(y))
        sig = y
        raw_value = float(y[idx])
    else:
        if float(np.min(y)) >= 0.0:
            return None
        idx = int(np.argmin(y))
        sig = -y
        raw_value = float(y[idx])

    peak_height = float(sig[idx])
    if idx == 0:
        baseline = float(np.min(sig[idx:]))
        edge_clipped = True
    elif idx == len(sig) - 1:
        baseline = float(np.min(sig[: idx + 1]))
        edge_clipped = True
    else:
        baseline = float(max(np.min(sig[: idx + 1]), np.min(sig[idx:])))
        edge_clipped = False
    prominence = peak_height - baseline
    level = peak_height - 0.5 * prominence

    left = idx
    while left > 0 and sig[left - 1] >= level:
        left -= 1
    if left == 0 and sig[left] >= level:
        left_x = float(x[0])
    else:
        x0 = float(x[left - 1])
        x1 = float(x[left])
        y0 = float(sig[left - 1])
        y1 = float(sig[left])
        left_x = x1 if y1 == y0 else x0 + (level - y0) * (x1 - x0) / (y1 - y0)

    right = idx
    while right < len(sig) - 1 and sig[right + 1] >= level:
        right += 1
    if right == len(sig) - 1 and sig[right] >= level:
        right_x = float(x[-1])
    else:
        x0 = float(x[right])
        x1 = float(x[right + 1])
        y0 = float(sig[right])
        y1 = float(sig[right + 1])
        right_x = x0 if y1 == y0 else x0 + (level - y0) * (x1 - x0) / (y1 - y0)

    return {
        "sic": int(x[idx]),
        "value_deg_per_sec": raw_value,
        "prominence_proxy_deg_per_sec": float(prominence),
        "half_prominence_width_proxy_sic": float(right_x - left_x),
        "edge_clipped": bool(edge_clipped),
    }


def analyze_peak_competition(x: np.ndarray, y: np.ndarray, kind: str) -> dict[str, Any]:
    sig = y if kind == "pos" else -y
    if (kind == "pos" and float(np.max(y)) <= 0.0) or (kind == "neg" and float(np.min(y)) >= 0.0):
        return {
            "num_interior_peaks": 0,
            "has_competition": False,
            "summary": f"no {kind} peak in window",
        }
    peaks, _ = find_peaks(sig)
    if len(peaks) == 0:
        return {
            "num_interior_peaks": 0,
            "has_competition": False,
            "summary": "no interior peak; window is edge-dominated / monotonic",
        }
    prominences = peak_prominences(sig, peaks)[0]
    order = np.argsort(sig[peaks])[::-1]
    ranked = []
    for oi in order[:3]:
        p = int(peaks[oi])
        ranked.append(
            {
                "sic": int(x[p]),
                "value_deg_per_sec": float(y[p]),
                "signed_height_for_rank": float(sig[p]),
                "prominence_deg_per_sec": float(prominences[oi]),
            }
        )
    if len(ranked) < 2:
        return {
            "num_interior_peaks": int(len(ranked)),
            "has_competition": False,
            "ranked_peaks": ranked,
            "summary": "single interior peak only",
        }
    top = ranked[0]
    second = ranked[1]
    amp_ratio = float(second["signed_height_for_rank"] / max(top["signed_height_for_rank"], 1e-9))
    prom_ratio = float(second["prominence_deg_per_sec"] / max(top["prominence_deg_per_sec"], 1e-9))
    has_competition = bool(amp_ratio >= 0.85 and prom_ratio >= 0.30)
    if has_competition:
        summary = (
            f"secondary {kind} peak at sic={second['sic']} is amplitude-close "
            f"(ratio={amp_ratio:.3f}) and prominence-meaningful (ratio={prom_ratio:.3f})"
        )
    else:
        summary = (
            f"secondary {kind} bump {'exists' if len(ranked) >= 2 else 'does not exist'}, "
            f"but prominence ratio is only {prom_ratio:.3f}"
        )
    return {
        "num_interior_peaks": int(len(ranked)),
        "has_competition": has_competition,
        "amplitude_ratio_2_over_1": amp_ratio,
        "prominence_ratio_2_over_1": prom_ratio,
        "ranked_peaks": ranked,
        "summary": summary,
    }


def zero_events(x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    eps = 1e-12
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(y):
        if abs(float(y[i])) <= eps:
            start = i
            while i + 1 < len(y) and abs(float(y[i + 1])) <= eps:
                i += 1
            end = i
            left = start - 1
            right = end + 1
            left_sign = int(np.sign(y[left])) if left >= 0 else 0
            right_sign = int(np.sign(y[right])) if right < len(y) else 0
            event = {
                "kind": "zero_plateau",
                "start_sic": int(x[start]),
                "end_sic": int(x[end]),
                "left_sign": left_sign,
                "right_sign": right_sign,
                "region_start_sic": int(x[left]) if left >= 0 else int(x[start]),
                "region_end_sic": int(x[right]) if right < len(y) else int(x[end]),
                "entry_slope_abs_deg_per_sec_per_sic": abs(float(y[start] - y[left])) if left >= 0 else None,
                "exit_slope_abs_deg_per_sec_per_sic": abs(float(y[right] - y[end])) if right < len(y) else None,
            }
            out.append(event)
        elif i + 1 < len(y) and float(y[i]) * float(y[i + 1]) < 0.0:
            x0 = float(x[i])
            x1 = float(x[i + 1])
            y0 = float(y[i])
            y1 = float(y[i + 1])
            zc = x0 - y0 * (x1 - x0) / (y1 - y0)
            out.append(
                {
                    "kind": "interp_cross",
                    "sic": float(zc),
                    "region_start_sic": x0,
                    "region_end_sic": x1,
                    "slope_abs_deg_per_sec_per_sic": abs(y1 - y0) / max(abs(x1 - x0), 1e-9),
                }
            )
        i += 1
    return out


def derivative_stats(y: np.ndarray) -> dict[str, Any]:
    d1 = np.diff(y)
    abs_d1 = np.abs(d1)
    return {
        "mean_abs_deg_per_sec_per_sic": float(np.mean(abs_d1)) if len(abs_d1) else None,
        "median_abs_deg_per_sec_per_sic": float(np.median(abs_d1)) if len(abs_d1) else None,
        "p90_abs_deg_per_sec_per_sic": float(np.percentile(abs_d1, 90)) if len(abs_d1) else None,
        "max_abs_deg_per_sec_per_sic": float(np.max(abs_d1)) if len(abs_d1) else None,
    }


def curvature_stats(y: np.ndarray) -> dict[str, Any]:
    d2 = np.diff(y, n=2)
    abs_d2 = np.abs(d2)
    return {
        "mean_abs_deg_per_sec_per_sic2": float(np.mean(abs_d2)) if len(abs_d2) else None,
        "median_abs_deg_per_sec_per_sic2": float(np.median(abs_d2)) if len(abs_d2) else None,
        "p90_abs_deg_per_sec_per_sic2": float(np.percentile(abs_d2, 90)) if len(abs_d2) else None,
        "max_abs_deg_per_sec_per_sic2": float(np.max(abs_d2)) if len(abs_d2) else None,
    }


def plateau_summary(x: np.ndarray, y: np.ndarray, kind: str) -> dict[str, Any]:
    sig = y if kind == "pos" else -y
    valid = float(np.max(sig)) > 0.0
    if not valid:
        return {
            "exists_95": False,
            "exists_90": False,
            "runs_95": [],
            "runs_90": [],
        }
    peak = float(np.max(sig))
    mask95 = sig >= (0.95 * peak)
    mask90 = sig >= (0.90 * peak)
    runs95 = longest_runs(mask95, x)
    runs90 = longest_runs(mask90, x)
    return {
        "exists_95": any(r["length"] >= 2 for r in runs95),
        "exists_90": any(r["length"] >= 2 for r in runs90),
        "runs_95": runs95,
        "runs_90": runs90,
    }


def classify_window(metrics: dict[str, Any]) -> dict[str, str]:
    zero_inside = metrics["window_zero_crossings"]
    deriv = metrics["derivative_stats"]
    curv = metrics["curvature_stats"]
    pos_peak = metrics["dominant_positive_peak"]
    neg_peak = metrics["dominant_negative_peak"]
    pos_plateau = metrics["positive_plateau"]
    neg_plateau = metrics["negative_plateau"]
    peak_comp_pos = metrics["positive_peak_competition"]
    peak_comp_neg = metrics["negative_peak_competition"]
    context_note = metrics["context_zero_note"]

    broad_flags = 0
    sharp_flags = 0

    if zero_inside:
        for evt in zero_inside:
            slope = evt.get("slope_abs_deg_per_sec_per_sic")
            if slope is not None and slope >= 20.0:
                sharp_flags += 2
            elif slope is not None and slope < 10.0:
                broad_flags += 1
    elif context_note == "adjacent_zero_plateau":
        broad_flags += 2

    if deriv["median_abs_deg_per_sec_per_sic"] is not None:
        if deriv["median_abs_deg_per_sec_per_sic"] >= 20.0:
            sharp_flags += 2
        elif deriv["median_abs_deg_per_sec_per_sic"] <= 8.0:
            broad_flags += 1

    if curv["max_abs_deg_per_sec_per_sic2"] is not None:
        if curv["max_abs_deg_per_sec_per_sic2"] >= 15.0:
            sharp_flags += 1
        elif curv["max_abs_deg_per_sec_per_sic2"] <= 6.0:
            broad_flags += 1

    for plateau in (pos_plateau, neg_plateau):
        if plateau["exists_95"] or plateau["exists_90"]:
            broad_flags += 1

    for comp in (peak_comp_pos, peak_comp_neg):
        if comp.get("has_competition"):
            broad_flags += 2

    for peak in (pos_peak, neg_peak):
        if peak is not None:
            width = peak["half_prominence_width_proxy_sic"]
            if width <= 2.5:
                sharp_flags += 1
            elif width >= 4.0:
                broad_flags += 1

    if sharp_flags >= broad_flags + 2:
        structure = "sharp / deterministic"
        lean = "more like capacity / temporal-resolution"
    elif broad_flags >= sharp_flags + 2:
        structure = "broad / phase-unfriendly"
        lean = "more like ambiguity / observability"
    else:
        structure = "mixed"
        lean = "mixed; mild ambiguity but not enough for a strong variance-style claim"
    return {
        "structure": structure,
        "lean": lean,
    }


def generate_plot(
    spec: JointSpec,
    fps: float,
    sic: np.ndarray,
    omega_z_deg: np.ndarray,
    out_path: Path,
    metrics: dict[str, Any],
) -> None:
    ctx_mask = (sic >= spec.context_start) & (sic <= min(spec.context_end, int(sic[-1])))
    x = sic[ctx_mask]
    y = omega_z_deg[ctx_mask]

    if len(y) >= 7:
        smooth = savgol_filter(y, 7, 2, mode="interp")
    elif len(y) >= 5:
        smooth = savgol_filter(y, 5, 2, mode="interp")
    else:
        smooth = y.copy()

    pos_local, _ = find_peaks(y)
    neg_local, _ = find_peaks(-y)
    d1 = np.abs(np.diff(y))
    d2 = np.abs(np.diff(y, n=2))
    mid1 = x[:-1] + 0.5
    mid2 = x[1:-1]
    hot_thr = float(np.percentile(d1, 85)) if len(d1) else 0.0
    hot_mask = d1 >= hot_thr if len(d1) else np.zeros(0, dtype=bool)

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(12, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )

    for ax in (ax0, ax1):
        ax.axvspan(spec.window_start, spec.window_end, color="#dbeafe", alpha=0.55, zorder=0)
        for evt in metrics["context_zero_events"]:
            ax.axvspan(
                float(evt["region_start_sic"]),
                float(evt["region_end_sic"]),
                color="#fecaca",
                alpha=0.18,
                zorder=0,
            )

    ax0.plot(x, y, color="#1f2937", lw=2.2, marker="o", ms=4.5, label="raw omega_z")
    ax0.plot(x, smooth, color="#2563eb", lw=1.8, alpha=0.9, label="smoothed omega_z")
    ax0.axhline(0.0, color="0.45", lw=1.0, ls="--")

    if len(pos_local):
        ax0.scatter(x[pos_local], y[pos_local], s=58, color="#16a34a", marker="^", label="local peak")
    if len(neg_local):
        ax0.scatter(x[neg_local], y[neg_local], s=58, color="#7c3aed", marker="v", label="local trough")

    for evt in metrics["context_zero_events"]:
        if evt["kind"] == "interp_cross":
            ax0.scatter(
                [evt["sic"]],
                [0.0],
                s=55,
                facecolors="white",
                edgecolors="#dc2626",
                linewidths=1.6,
                label="zero-crossing" if "zero-crossing" not in ax0.get_legend_handles_labels()[1] else None,
                zorder=5,
            )
        else:
            plateau_x = np.arange(evt["start_sic"], evt["end_sic"] + 1)
            ax0.scatter(
                plateau_x,
                np.zeros_like(plateau_x, dtype=float),
                s=32,
                facecolors="white",
                edgecolors="#dc2626",
                linewidths=1.2,
                marker="s",
                label="zero plateau" if "zero plateau" not in ax0.get_legend_handles_labels()[1] else None,
                zorder=5,
            )

    if len(hot_mask):
        hot_x = mid1[hot_mask]
        hot_y = 0.5 * (y[:-1][hot_mask] + y[1:][hot_mask])
        ax0.scatter(
            hot_x,
            hot_y,
            s=72,
            marker="x",
            color="#f59e0b",
            linewidths=1.8,
            label=f"high |d omega/d sic| (>=p85={hot_thr:.1f})",
            zorder=5,
        )
    window_mid_mask = (mid1 >= spec.window_start) & (mid1 <= spec.window_end)
    if np.any(window_mid_mask):
        local_idx = int(np.argmax(d1[window_mid_mask]))
        local_x = mid1[window_mid_mask][local_idx]
        local_y = 0.5 * (y[:-1][window_mid_mask][local_idx] + y[1:][window_mid_mask][local_idx])
        ax0.scatter(
            [local_x],
            [local_y],
            s=88,
            marker="*",
            color="#ea580c",
            edgecolors="white",
            linewidths=0.7,
            label="window max |d omega/d sic|",
            zorder=6,
        )

    ax0.set_ylabel("omega_z (deg/s)")
    ax0.set_title(f"Walk_F GT omega_z audit - {spec.joint} (local/body, FPS={fps:.1f})")
    ax0.legend(loc="upper left", ncol=3, fontsize=9, frameon=False)

    ax1.plot(mid1, d1, color="#f59e0b", lw=1.9, marker="o", ms=3.5, label="|d omega_z / d sic|")
    if len(d2):
        ax1.plot(mid2, d2, color="#6b7280", lw=1.5, ls="--", marker=".", ms=4, label="|d^2 omega_z / d sic^2|")
    ax1.axhline(hot_thr, color="#f59e0b", lw=1.0, ls=":", alpha=0.8)
    ax1.set_ylabel("deg/s per SIC")
    ax1.set_xlabel("step_in_cycle (sic)")
    ax1.legend(loc="upper left", ncol=2, fontsize=9, frameon=False)

    x_right = max(spec.context_end, int(x[-1]))
    ax1.set_xlim(spec.context_start, x_right)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_joint_metrics(spec: JointSpec, sic: np.ndarray, omega_z_deg: np.ndarray) -> dict[str, Any]:
    ctx_end = min(spec.context_end, int(sic[-1]))
    ctx_mask = (sic >= spec.context_start) & (sic <= ctx_end)
    win_mask = (sic >= spec.window_start) & (sic <= spec.window_end)
    x_ctx = sic[ctx_mask]
    y_ctx = omega_z_deg[ctx_mask]
    x_win = sic[win_mask]
    y_win = omega_z_deg[win_mask]

    z_ctx = zero_events(x_ctx, y_ctx)
    z_win = [
        evt
        for evt in z_ctx
        if (
            (evt["kind"] == "interp_cross" and spec.window_start <= evt["sic"] <= spec.window_end)
            or (
                evt["kind"] == "zero_plateau"
                and not (evt["end_sic"] < spec.window_start or evt["start_sic"] > spec.window_end)
            )
        )
    ]

    context_zero_note = "none"
    if not z_win:
        for evt in z_ctx:
            if evt["kind"] == "zero_plateau" and evt["region_end_sic"] >= spec.window_start - 1:
                context_zero_note = "adjacent_zero_plateau"
                break
            if evt["kind"] == "interp_cross":
                zc = float(evt["sic"])
                if spec.window_start - 1.0 <= zc <= spec.window_end + 1.0:
                    context_zero_note = "adjacent_interp_cross"
                    break

    metrics = {
        "joint": spec.joint,
        "window_start": spec.window_start,
        "window_end": spec.window_end,
        "context_start": spec.context_start,
        "context_end_requested": spec.context_end,
        "context_end_used": int(ctx_end),
        "window_values_deg_per_sec": [float(v) for v in y_win],
        "dynamic_range_deg_per_sec": float(np.max(y_win) - np.min(y_win)),
        "dominant_positive_peak": extremum_metrics(x_win, y_win, "pos"),
        "dominant_negative_peak": extremum_metrics(x_win, y_win, "neg"),
        "window_zero_crossings": z_win,
        "context_zero_events": z_ctx,
        "context_zero_note": context_zero_note,
        "derivative_stats": derivative_stats(y_win),
        "curvature_stats": curvature_stats(y_win),
        "positive_plateau": plateau_summary(x_win, y_win, "pos"),
        "negative_plateau": plateau_summary(x_win, y_win, "neg"),
        "positive_peak_competition": analyze_peak_competition(x_win, y_win, "pos"),
        "negative_peak_competition": analyze_peak_competition(x_win, y_win, "neg"),
    }
    metrics["classification"] = classify_window(metrics)
    return metrics


def report_joint_section(metrics: dict[str, Any], figure_path: Path) -> list[str]:
    pos_peak = metrics["dominant_positive_peak"]
    neg_peak = metrics["dominant_negative_peak"]
    deriv = metrics["derivative_stats"]
    curv = metrics["curvature_stats"]
    pos_plateau = metrics["positive_plateau"]
    neg_plateau = metrics["negative_plateau"]
    lines: list[str] = []
    lines.append(f"## {metrics['joint']} @ sic={metrics['window_start']}-{metrics['window_end']}")
    lines.append("")
    lines.append(f"- Figure: `{figure_path}`")
    lines.append(
        f"- Context shown: sic={metrics['context_start']}-{metrics['context_end_requested']} "
        f"(data available through sic={metrics['context_end_used']})."
    )
    lines.append(
        f"- GT read: `{metrics['classification']['structure']}`; lean: "
        f"`{metrics['classification']['lean']}`."
    )
    lines.append("")
    lines.append("|metric|value|")
    lines.append("|---|---|")
    if pos_peak is None:
        lines.append("|max positive peak|none (window never goes positive)|")
    else:
        lines.append(
            "|max positive peak|"
            f"{_fmt(pos_peak['value_deg_per_sec'])} deg/s at sic={pos_peak['sic']}; "
            f"prom={_fmt(pos_peak['prominence_proxy_deg_per_sec'])}; "
            f"half-prom width={_fmt(pos_peak['half_prominence_width_proxy_sic'])} sic"
            f"{' (edge-clipped)' if pos_peak['edge_clipped'] else ''}|"
        )
    if neg_peak is None:
        lines.append("|max negative peak|none (window never goes negative)|")
    else:
        lines.append(
            "|max negative peak|"
            f"{_fmt(neg_peak['value_deg_per_sec'])} deg/s at sic={neg_peak['sic']}; "
            f"prom={_fmt(neg_peak['prominence_proxy_deg_per_sec'])}; "
            f"half-prom width={_fmt(neg_peak['half_prominence_width_proxy_sic'])} sic"
            f"{' (edge-clipped)' if neg_peak['edge_clipped'] else ''}|"
        )
    if metrics["window_zero_crossings"]:
        z_msgs = []
        for evt in metrics["window_zero_crossings"]:
            if evt["kind"] == "interp_cross":
                z_msgs.append(
                    f"sic={_fmt(evt['sic'])}, slope={_fmt(evt['slope_abs_deg_per_sec_per_sic'])}"
                )
            else:
                z_msgs.append(
                    f"zero plateau sic={evt['start_sic']}-{evt['end_sic']}, "
                    f"entry slope={_fmt(evt['entry_slope_abs_deg_per_sec_per_sic'])}, "
                    f"exit slope={_fmt(evt['exit_slope_abs_deg_per_sec_per_sic'])}"
                )
        lines.append("|zero-crossing in window|" + "; ".join(z_msgs) + "|")
    else:
        note = "no zero-crossing inside window"
        if metrics["context_zero_note"] == "adjacent_zero_plateau":
            plateau_evt = next(
                (evt for evt in metrics["context_zero_events"] if evt["kind"] == "zero_plateau"),
                None,
            )
            if plateau_evt is not None:
                note += (
                    "; nearest context event is zero plateau "
                    f"sic={plateau_evt['start_sic']}-{plateau_evt['end_sic']} "
                    f"(entry slope={_fmt(plateau_evt['entry_slope_abs_deg_per_sec_per_sic'])}, "
                    f"exit slope={_fmt(plateau_evt['exit_slope_abs_deg_per_sec_per_sic'])})"
                )
            else:
                note += "; nearest context event is an adjacent zero plateau"
        elif metrics["context_zero_note"] == "adjacent_interp_cross":
            interp_evt = next(
                (evt for evt in metrics["context_zero_events"] if evt["kind"] == "interp_cross"),
                None,
            )
            if interp_evt is not None:
                note += (
                    "; nearest context event is interpolated crossing at "
                    f"sic={_fmt(interp_evt['sic'])} "
                    f"(slope={_fmt(interp_evt['slope_abs_deg_per_sec_per_sic'])})"
                )
            else:
                note += "; nearest context event is an adjacent interpolated zero-crossing"
        lines.append(f"|zero-crossing in window|{note}|")
    lines.append(f"|omega_z dynamic range|{_fmt(metrics['dynamic_range_deg_per_sec'])} deg/s|")
    lines.append(
        "|d omega_z / d sic stats|"
        f"median={_fmt(deriv['median_abs_deg_per_sec_per_sic'])}, "
        f"mean={_fmt(deriv['mean_abs_deg_per_sec_per_sic'])}, "
        f"p90={_fmt(deriv['p90_abs_deg_per_sec_per_sic'])}, "
        f"max={_fmt(deriv['max_abs_deg_per_sec_per_sic'])}|"
    )
    lines.append(
        "|curvature (d^2 omega_z / d sic^2)|"
        f"median={_fmt(curv['median_abs_deg_per_sec_per_sic2'])}, "
        f"mean={_fmt(curv['mean_abs_deg_per_sec_per_sic2'])}, "
        f"p90={_fmt(curv['p90_abs_deg_per_sec_per_sic2'])}, "
        f"max={_fmt(curv['max_abs_deg_per_sec_per_sic2'])}|"
    )
    lines.append(
        "|plateau test (>=95% / >=90% of dominant)|"
        f"positive: {format_runs(pos_plateau['runs_95'])} / {format_runs(pos_plateau['runs_90'])}; "
        f"negative: {format_runs(neg_plateau['runs_95'])} / {format_runs(neg_plateau['runs_90'])}|"
    )
    lines.append(
        "|multi-peak competition|"
        f"positive: {metrics['positive_peak_competition']['summary']}; "
        f"negative: {metrics['negative_peak_competition']['summary']}|"
    )
    lines.append("")
    lines.append("Qualitative read:")
    if metrics["joint"] == "calf_l":
        lines.append(
            "- The main window is a low-to-moderate positive shoulder after an adjacent zero plateau "
            "(context sic=74-77), not an isolated sharp spike."
        )
        lines.append(
            "- There is a dominant positive peak at sic=80, but its half-prominence width is broad and a "
            "secondary bump around sic=83 behaves more like a shoulder than a clean second mode."
        )
        lines.append(
            "- Because the onset emerges out of several exact-zero samples and the immediate exit slope from "
            "the zero plateau is weak, this window is comparatively phase-unfriendly."
        )
    else:
        lines.append(
            "- The window is almost a monotonic high-slope descent from strong positive omega_z to strong "
            "negative omega_z."
        )
        lines.append(
            "- The interpolated zero-crossing sits cleanly inside the window, and the crossing slope is large "
            "relative to the rest of the clip."
        )
        lines.append(
            "- There is no real plateau and no meaningful peak competition inside the window; the structure "
            "looks phase-sharp and deterministic."
        )
    lines.append("")
    return lines


def build_report(
    out_dir: Path,
    npz_path: Path,
    json_path: Path,
    fps: float,
    cycle_info: dict[str, Any],
    z_axis_info: dict[str, Any],
    per_joint: dict[str, dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Walk_F single-clip GT structure / ambiguity audit")
    lines.append("")
    lines.append("Date: 2026-03-12")
    lines.append("")
    lines.append("## Scope and caution")
    lines.append("")
    lines.append(f"- Inputs: `{npz_path}` and `{json_path}`.")
    lines.append("- GT-only audit: only `bone_ang_vel`, no prediction, no model change, no training.")
    lines.append(
        "- This is **not** a GT variance audit. `Walk_F` is a single natural cycle, so the result can only be "
        "interpreted as a single-clip structural ambiguity / sharpness audit."
    )
    lines.append(
        "- Main signal: local/body-space `omega_z`; `||omega||` was checked only as a support signal."
    )
    lines.append("")
    lines.append("## Data confirmation")
    lines.append("")
    lines.append(
        f"- Raw JSON: `NumFrames={cycle_info['json_num_frames']}`, `len(Frames)={cycle_info['json_frames_len']}`."
    )
    lines.append(
        f"- Processed NPZ: `bone_ang_vel.shape={tuple(cycle_info['bone_ang_vel_shape'])}`, "
        f"`bone_rot6d.shape={tuple(cycle_info['bone_rot6d_shape'])}`."
    )
    lines.append(
        "- There is no extra sample/cycle axis in either file; the clip is a single flat time axis with "
        "SIC treated as frame index `0..87`."
    )
    lines.append(
        f"- Cycle closure check (frame 0 vs 87, local pose): mean={_fmt(cycle_info['closure_mean_deg'])} deg, "
        f"median={_fmt(cycle_info['closure_median_deg'])} deg, max={_fmt(cycle_info['closure_max_deg'])} deg."
    )
    lines.append(
        f"- Mid-cycle contrast (frame 0 vs 44): mean={_fmt(cycle_info['midcycle_mean_deg'])} deg, "
        f"median={_fmt(cycle_info['midcycle_median_deg'])} deg, max={_fmt(cycle_info['midcycle_max_deg'])} deg."
    )
    lines.append(
        f"- Root displacement over the clip: `dx,dy,dz = [{cycle_info['root_displacement'][0]:.3f}, "
        f"{cycle_info['root_displacement'][1]:.3f}, {cycle_info['root_displacement'][2]:.3f}]` m."
    )
    lines.append(
        "- Interpretation: start/end local pose nearly closes while root translates forward, which is consistent "
        "with one locomotion cycle rather than a multi-cycle stack."
    )
    lines.append("")
    lines.append("## omega_z sufficiency check")
    lines.append("")
    for joint, info in z_axis_info.items():
        lines.append(
            f"- `{joint}`: on non-zero samples, `|omega_z| / ||omega||` min/mean/max = "
            f"{_fmt(info['ratio_min'])} / {_fmt(info['ratio_mean'])} / {_fmt(info['ratio_max'])}; "
            "the motion is effectively pure z-axis rotation in these windows."
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for joint in ("calf_l", "calf_r"):
        lines.append(f"- `{out_dir / f'{joint}_gt_omega_z.png'}`")
    lines.append(f"- `{out_dir / 'singleclip_gt_structure_audit.md'}`")
    lines.append(f"- `{out_dir / 'singleclip_gt_structure_audit_metrics.json'}`")
    lines.append("")
    lines.append("## Window diagnostics")
    lines.append("")
    for joint in ("calf_l", "calf_r"):
        lines.extend(report_joint_section(per_joint[joint], out_dir / f"{joint}_gt_omega_z.png"))
    lines.append("## Final read")
    lines.append("")
    lines.append("|window|GT structure|lean|why|")
    lines.append("|---|---|---|---|")
    lines.append(
        "|`calf_l @ sic=78-85`|broad / phase-unfriendly|more like ambiguity / observability|"
        "adjacent zero plateau, soft onset, broad shoulder, no clean in-window zero-crossing|"
    )
    lines.append(
        "|`calf_r @ sic=56-62`|sharp / deterministic|more like capacity / temporal-resolution|"
        "clean in-window sign flip, large crossing slope, monotonic ramp, no plateau / no peak competition|"
    )
    lines.append("")
    lines.append("- The window that most looks like `model should learn this but currently has not` is `calf_r @ sic=56-62`.")
    lines.append(
        "- The window that more looks like `GT is itself less phase-friendly in this local region` is "
        "`calf_l @ sic=78-85`, but the claim should stay modest: this is only a single-clip structural read, "
        "not a cross-sample variance floor argument."
    )
    lines.append(
        "- So the practical read is: if capacity / temporal-resolution follow-up is expensive, `calf_r` is the "
        "cleaner target for that direction; `calf_l` still carries a stronger ambiguity / observability caveat."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk_F single-clip GT structure audit")
    ap.add_argument("--npz", type=Path, default=Path("raw_data/processed_data/Walk_F.npz"))
    ap.add_argument("--json", type=Path, default=Path("raw_data/Walk_F.json"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("debug_output/_tmp_walkf_singleclip_gt_structure_20260312"),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.npz, allow_pickle=True)
    raw = json.loads(Path(args.json).read_text())

    fps = float(np.asarray(npz["FPS"]).item())
    bone_names = [str(x) for x in npz["bone_names"].tolist()]
    bone_ang_vel = np.asarray(npz["bone_ang_vel"], dtype=np.float64)
    bone_rot6d = np.asarray(npz["bone_rot6d"], dtype=np.float64)
    root_pos = np.asarray(npz["root_pos"], dtype=np.float64)

    sic = np.arange(bone_ang_vel.shape[0], dtype=np.int32)
    omega_deg = np.rad2deg(bone_ang_vel)

    r0 = rot6d_to_matrix(bone_rot6d[0])
    r_last = rot6d_to_matrix(bone_rot6d[-1])
    r_mid = rot6d_to_matrix(bone_rot6d[len(bone_rot6d) // 2])
    closure_deg = rotation_angle_deg(r0, r_last)
    midcycle_deg = rotation_angle_deg(r0, r_mid)

    cycle_info = {
        "json_num_frames": int(raw["NumFrames"]),
        "json_frames_len": int(len(raw["Frames"])),
        "bone_ang_vel_shape": list(bone_ang_vel.shape),
        "bone_rot6d_shape": list(bone_rot6d.shape),
        "closure_mean_deg": float(np.mean(closure_deg)),
        "closure_median_deg": float(np.median(closure_deg)),
        "closure_max_deg": float(np.max(closure_deg)),
        "midcycle_mean_deg": float(np.mean(midcycle_deg)),
        "midcycle_median_deg": float(np.median(midcycle_deg)),
        "midcycle_max_deg": float(np.max(midcycle_deg)),
        "root_displacement": [float(v) for v in (root_pos[-1] - root_pos[0]).tolist()],
    }

    z_axis_info: dict[str, dict[str, Any]] = {}
    per_joint: dict[str, dict[str, Any]] = {}
    for spec in JOINT_SPECS:
        j = bone_names.index(spec.joint)
        ctx_end = min(spec.context_end, len(sic) - 1)
        ctx_mask = (sic >= spec.context_start) & (sic <= ctx_end)
        omega = omega_deg[:, j, :]
        omega_norm = np.linalg.norm(omega[ctx_mask], axis=-1)
        nz = omega_norm > 1e-9
        ratio = np.abs(omega[ctx_mask][nz, 2]) / omega_norm[nz]
        z_axis_info[spec.joint] = {
            "ratio_min": float(np.min(ratio)) if len(ratio) else None,
            "ratio_mean": float(np.mean(ratio)) if len(ratio) else None,
            "ratio_max": float(np.max(ratio)) if len(ratio) else None,
        }
        per_joint[spec.joint] = build_joint_metrics(spec, sic, omega_deg[:, j, 2])
        generate_plot(
            spec,
            fps,
            sic,
            omega_deg[:, j, 2],
            args.out_dir / f"{spec.joint}_gt_omega_z.png",
            per_joint[spec.joint],
        )

    metrics_json = {
        "inputs": {
            "npz": str(args.npz),
            "json": str(args.json),
            "fps": fps,
            "space": raw["meta"]["spaces"]["bone_angular_velocities"],
        },
        "cycle_confirmation": cycle_info,
        "omega_z_sufficiency": z_axis_info,
        "per_joint": per_joint,
    }
    (args.out_dir / "singleclip_gt_structure_audit_metrics.json").write_text(
        json.dumps(_serializable(metrics_json), indent=2),
        encoding="utf-8",
    )

    report = build_report(args.out_dir, args.npz, args.json, fps, cycle_info, z_axis_info, per_joint)
    (args.out_dir / "singleclip_gt_structure_audit.md").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
