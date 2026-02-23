#!/usr/bin/env python3
"""
Plot contact-related freerun diagnostics from run_freerun_cycles outputs.

Focus metrics (per-step):
  - ContactMeasMean
  - ContactMeasGtAbsMean
  - ContactErrAbsMean

Default inputs assume you already ran the 4 comparison cases:
  - global / no apply
  - global / apply10
  - cycle  / no apply
  - cycle  / apply10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _series(steps: List[Dict[str, Any]], key: str) -> np.ndarray:
    out = []
    for item in steps:
        v = item.get(key, None) if isinstance(item, dict) else None
        out.append(float(v) if v is not None else np.nan)
    return np.asarray(out, dtype=np.float64)


def _finite_stats(x: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    m = np.isfinite(x)
    if not np.any(m):
        return None, None
    vals = x[m]
    return float(np.mean(vals)), float(np.std(vals))


@dataclass(frozen=True)
class Case:
    label: str
    path: Path


def _plot_column(
    ax_col: List,
    *,
    cases: List[Case],
    cycle_len: Optional[int],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    # Metrics to plot per row: (key, ylabel)
    rows = (
        ("ContactMeasMean", "ContactMeasMean"),
        ("ContactMeasGtAbsMean", "ContactMeasGtAbsMean"),
        ("ContactErrAbsMean", "ContactErrAbsMean"),
    )

    boundary_step = None
    if cycle_len is not None and cycle_len > 1:
        boundary_step = int(cycle_len - 1)  # wrap boundary transition (t=cycle_len-1)

    for row_idx, (key, ylabel) in enumerate(rows):
        ax = ax_col[row_idx]
        for case in cases:
            data = _load_json(case.path)
            steps = data.get("metrics_per_step", [])
            series = _series(steps, key)
            t = np.arange(series.shape[0], dtype=np.int64)

            mu, sig = _finite_stats(series)
            if key == "ContactMeasMean" and (mu is not None) and (sig is not None):
                disp = f"{case.label} (μ={mu:.3f}, σ={sig:.3f})"
            else:
                disp = case.label
            ax.plot(t, series, linewidth=1.5, alpha=0.9, label=disp)

        if boundary_step is not None:
            ax.axvline(boundary_step, linestyle="--", color="k", alpha=0.2, linewidth=1.0)
            ax.text(
                boundary_step + 0.5,
                ax.get_ylim()[1],
                "cycle boundary",
                fontsize=8,
                alpha=0.35,
                va="top",
            )

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)
        if row_idx == 0:
            ax.set_title(title)
        if row_idx == len(rows) - 1:
            ax.set_xlabel("step (transition t→t+1)")

    ax_col[0].legend(loc="upper right", fontsize=8, frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-step contact meas/err diagnostics from freerun_cycles JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--global-noapply",
        type=str,
        default="debug_output/freerun_cycles/compare_absstart_global_noapply/Walk_F_freerun_cycles.json",
        help="Path to global time-index / no so3_corr_apply JSON.",
    )
    parser.add_argument(
        "--global-apply",
        type=str,
        default="debug_output/freerun_cycles/compare_absstart_global_apply10/Walk_F_freerun_cycles.json",
        help="Path to global time-index / so3_corr_apply JSON.",
    )
    parser.add_argument(
        "--cycle-noapply",
        type=str,
        default="debug_output/freerun_cycles/compare_absstart_cycle_noapply/Walk_F_freerun_cycles.json",
        help="Path to cycle time-index / no so3_corr_apply JSON.",
    )
    parser.add_argument(
        "--cycle-apply",
        type=str,
        default="debug_output/freerun_cycles/compare_absstart_cycle_apply10/Walk_F_freerun_cycles.json",
        help="Path to cycle time-index / so3_corr_apply JSON.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="debug_output/freerun_cycles/compare_contacts_meas_err_timeseries.png",
        help="Output image path (png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Walk_F: contacts_meas / contacts_err (global vs cycle, apply vs no-apply)",
        help="Figure title.",
    )
    args = parser.parse_args()

    paths = {
        "global_noapply": Path(args.global_noapply).expanduser(),
        "global_apply": Path(args.global_apply).expanduser(),
        "cycle_noapply": Path(args.cycle_noapply).expanduser(),
        "cycle_apply": Path(args.cycle_apply).expanduser(),
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit("[FATAL] Missing JSON files:\n  - " + "\n  - ".join(missing))

    # Infer cycle_len from any case (they should match).
    probe = _load_json(paths["global_noapply"])
    cycle_len = probe.get("cycle_len")
    try:
        cycle_len = int(cycle_len) if cycle_len is not None else None
    except Exception:
        cycle_len = None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)

    _plot_column(
        list(axes[:, 0]),
        cases=[
            Case("no-apply", paths["global_noapply"]),
            Case("apply10", paths["global_apply"]),
        ],
        cycle_len=cycle_len,
        title="time_index = global (t)",
    )
    _plot_column(
        list(axes[:, 1]),
        cases=[
            Case("no-apply", paths["cycle_noapply"]),
            Case("apply10", paths["cycle_apply"]),
        ],
        cycle_len=cycle_len,
        title="time_index = cycle (t % cycle_len)",
    )

    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

