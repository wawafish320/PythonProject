from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from train.configuration.io import dump_json, load_json

from .profile import DatasetProfiler
from .stages import TrainingConfigBuilder


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Unified training config manager")
    ap.add_argument(
        "--data-root",
        default=str(project_root / "raw_data"),
        help="Directory containing raw JSON clips",
    )
    ap.add_argument(
        "--base-config",
        default=str(project_root / "config" / "exp_phase_mpl.json"),
        help="Existing config to load as baseline",
    )
    ap.add_argument(
        "--output",
        default=str(project_root / "config" / "exp_phase_mpl.json"),
        help="Where to write the updated config",
    )
    ap.add_argument("--profile", action="store_true", help="Recompute dataset profile before building config")
    ap.add_argument("--dry-run", action="store_true", help="Print results without writing files")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    base_cfg = load_json(Path(args.base_config))

    profile = base_cfg.get("dataset_profile")
    if args.profile or not profile:
        profiler = DatasetProfiler(Path(args.data_root))
        profile = profiler.profile()
        print(
            f"[profile] samples={profile['n_clips']} frames={profile['total_frames']} avg_seq={profile['avg_seq_len']:.1f}"
        )

    builder = TrainingConfigBuilder(base_cfg)
    config = builder.build(profile)
    print(f"[build] epochs={config['epochs']} batch={config['batch']} lr={config['lr']}")

    if args.dry_run:
        print("[dry-run] updated config not written")
        return 0

    dump_json(Path(args.output), config)
    print(f"[write] saved config to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
