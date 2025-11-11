#!/usr/bin/env python3
"""Auto-adjust stage schedule parameters based on latest validation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from train.train_configurator import (
    StageMetricAdjuster,
    aggregate_metrics,
    dump_json,
    load_json,
    load_val_metrics,
    summarize_stage_metrics,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-tune stage schedule based on metrics")
    ap.add_argument("--config", default="config/exp_phase_mpl.json", help="Path to config JSON to update")
    ap.add_argument(
        "--metrics",
        default="models/exp_phase_e2e_sc/exp_phase_MLP/metrics",
        help="Directory containing valfree metrics json files",
    )
    ap.add_argument("--dry-run", action="store_true", help="Preview adjustments without writing the config file")
    args = ap.parse_args()

    config_path = Path(args.config)
    metrics_dir = Path(args.metrics)
    if not config_path.is_file():
        ap.error(f"config file not found: {config_path}")
    if not metrics_dir.is_dir():
        ap.error(f"metrics dir not found: {metrics_dir}")

    config = load_json(config_path)
    schedule = config.get("lookahead_stage_schedule")
    if not schedule:
        ap.error("config missing lookahead_stage_schedule")

    val_metrics = load_val_metrics(metrics_dir)
    if not val_metrics:
        ap.error("no valfree metrics found to guide tuning")

    adjuster = StageMetricAdjuster(config)
    changed = adjuster.apply(val_metrics)
    summary = summarize_stage_metrics(schedule, val_metrics)
    for stage in schedule:
        label = stage.get("label") or f"{stage.get('range')}"
        metrics = summary.get(label, {})
        stage_changes = {k.split(".", 1)[-1]: v for k, v in changed.items() if k.startswith(label)}
        print(f"[STAGE] {label}: metrics={metrics or 'n/a'} adjustments={stage_changes or 'no change'}")

    if not changed:
        print("[INFO] no parameter updates needed")
        return

    if args.dry_run:
        print("[INFO] dry-run enabled; config not written")
        return

    dump_json(config_path, config)
    agg = aggregate_metrics(metrics_dir)
    print(f"[OK] saved updated schedule to {config_path} (metrics ave: {agg})")


if __name__ == "__main__":
    main()
