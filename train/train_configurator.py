#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - allow both `python -m` and direct execution
    from .configuration import DatasetProfiler, TrainingConfigBuilder, dump_json, load_json
except ImportError:  # executed when invoking `python train/train_configurator.py`
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from train.configuration import DatasetProfiler, TrainingConfigBuilder, dump_json, load_json  # type: ignore


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unified training config manager")
    ap.add_argument("--data-root", default="raw_data", help="Directory containing raw JSON clips")
    ap.add_argument("--base-config", default="config/exp_phase_mpl.json", help="Existing config to load as baseline")
    ap.add_argument("--output", default="config/exp_phase_mpl.json", help="Where to write the updated config")
    ap.add_argument("--profile", action="store_true", help="Recompute dataset profile before building config")
    ap.add_argument("--dry-run", action="store_true", help="Print results without writing files")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    base_cfg = load_json(Path(args.base_config))

    profile = base_cfg.get("dataset_profile")
    if args.profile or not profile:
        profiler = DatasetProfiler(Path(args.data_root))
        profile = profiler.profile()
        print(f"[profile] samples={profile['n_clips']} frames={profile['total_frames']} avg_seq={profile['avg_seq_len']:.1f}")

    builder = TrainingConfigBuilder(base_cfg)
    config = builder.build(profile)
    print(f"[build] epochs={config['epochs']} batch={config['batch']} lr={config['lr']}")

    if args.dry_run:
        print("[dry-run] updated config not written")
    else:
        dump_json(Path(args.output), config)
        print(f"[write] saved config to {args.output}")


if __name__ == "__main__":
    main()
