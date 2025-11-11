#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - allow both `python -m` and direct execution
    from .configuration import (
        DEFAULT_BAYES_SPECS,
        BayesHistory,
        BayesianOptimizer,
        DatasetProfiler,
        StageMetricAdjuster,
        TrainingConfigBuilder,
        aggregate_metrics,
        apply_param_updates,
        compute_history_score,
        dump_json,
        extract_param_vector,
        load_json,
        load_val_metrics,
    )
except ImportError:  # executed when invoking `python train/train_configurator.py`
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from train.configuration import (  # type: ignore
        DEFAULT_BAYES_SPECS,
        BayesHistory,
        BayesianOptimizer,
        DatasetProfiler,
        StageMetricAdjuster,
        TrainingConfigBuilder,
        aggregate_metrics,
        apply_param_updates,
        compute_history_score,
        dump_json,
        extract_param_vector,
        load_json,
        load_val_metrics,
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unified training config manager")
    ap.add_argument("--data-root", default="raw_data", help="Directory containing raw JSON clips")
    ap.add_argument("--metrics-root", default=None, help="Directory containing valfree metrics JSON files")
    ap.add_argument("--base-config", default="config/exp_phase_mpl.json", help="Existing config to load as baseline")
    ap.add_argument("--output", default="config/exp_phase_mpl.json", help="Where to write the updated config")
    ap.add_argument("--history", default="train/bayes_history.json", help="Bayesian optimization history file")
    ap.add_argument("--profile", action="store_true", help="Recompute dataset profile before building config")
    ap.add_argument("--tune", action="store_true", help="Apply metric-driven stage adjustments")
    ap.add_argument("--bayes-opt", action="store_true", help="Enable Bayesian optimizer suggestions")
    ap.add_argument("--bayes-update-history", action="store_true", help="Append current run (config+metrics) to history")
    ap.add_argument("--dry-run", action="store_true", help="Print results without writing files")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for candidate sampling")
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

    metrics_root = Path(args.metrics_root) if args.metrics_root else None
    if args.tune and metrics_root is None:
        raise SystemExit("--tune requires --metrics-root")
    val_metrics = load_val_metrics(metrics_root) if (args.tune and metrics_root) else {}
    if args.tune and metrics_root is not None:
        adjuster = StageMetricAdjuster(config)
        changed = adjuster.apply(val_metrics)
        if changed:
            delta_str = ", ".join(f"{k}={v}" for k, v in changed.items())
            print(f"[tune] updated: {delta_str}")
        else:
            print("[tune] no changes applied")

    metrics_summary = aggregate_metrics(metrics_root) if metrics_root else None
    history = BayesHistory(Path(args.history))

    if args.bayes_update_history and metrics_summary:
        params = extract_param_vector(config, DEFAULT_BAYES_SPECS)
        score = compute_history_score(metrics_summary)
        history.add(params, metrics_summary, score)
        history.save()
        print(f"[history] appended entry score={score:.4f} ({len(history.entries)} total)")

    if args.bayes_opt:
        optimizer = BayesianOptimizer(DEFAULT_BAYES_SPECS, seed=args.seed)
        suggestion = optimizer.suggest(history.entries)
        if suggestion:
            apply_param_updates(config, suggestion)
            formatted = ", ".join(f"{k}={suggestion[k]:.4g}" for k in sorted(suggestion))
            print(f"[bayes] suggestion: {formatted}")
        else:
            print("[bayes] insufficient history, skipped")

    if args.dry_run:
        print("[dry-run] updated config not written")
    else:
        dump_json(Path(args.output), config)
        print(f"[write] saved config to {args.output}")


if __name__ == "__main__":
    main()
