#!/usr/bin/env python3
"""Thin wrapper around train.train_configurator for backward compatibility."""

from __future__ import annotations

import argparse
from pathlib import Path

from train.train_configurator import DatasetProfiler, TrainingConfigBuilder, dump_json, load_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate training config from dataset profile")
    ap.add_argument("--data-root", default="raw_data", help="Directory containing raw JSON clips")
    ap.add_argument("--base", default="config/exp_phase_mpl.json", help="Base config to load (if exists)")
    ap.add_argument("--output", default="config/exp_phase_mpl.json", help="Where to write the generated config")
    ap.add_argument("--profile", action="store_true", help="Force re-profiling even if cached")
    args = ap.parse_args()

    base_cfg = load_json(Path(args.base))
    profiler = DatasetProfiler(Path(args.data_root))
    profile = profiler.profile()
    if not args.profile and base_cfg.get("dataset_profile"):
        print("[strategy_generator] base config already has dataset_profile, overriding with fresh stats.")

    builder = TrainingConfigBuilder(base_cfg)
    cfg = builder.build(profile)
    dump_json(Path(args.output), cfg)
    print(f"[OK] wrote config to {args.output} (epochs={cfg['epochs']} batch={cfg['batch']} lr={cfg['lr']})")


if __name__ == "__main__":
    main()
