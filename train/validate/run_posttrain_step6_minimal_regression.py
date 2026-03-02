#!/usr/bin/env python3
"""Step6 minimal regression gate for train/posttrain.py refactor."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

METRIC_KEYS = (
    "total",
    "dir_geo",
    "inc_geo",
    "blend_loss",
    "lambda_mean",
    "lambda_eff_mean",
    "gate_sup_loss",
)

DEFAULT_DIRECT_CONFIG = "config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_arm_pe32h512_20260227.json"
DEFAULT_LAMBDA_CONFIG = "config/posttrain_WalkF_stage7_lambda_final_calib_20260226_fromsplitfirst_fullcompat.json"
DEFAULT_OUT_DIR = "models/__tmp_posttrain_smoke"


@dataclass
class _Case:
    mode: str
    config_path: Path
    run_name: str
    baseline_run_name: str


def _run_command(cmd: list[str], *, env: Dict[str, str]) -> None:
    print(f"[step6] RUN: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True, env=env)


def _load_last_metrics(log_path: Path) -> Dict[str, float]:
    payload = json.loads(log_path.read_text())
    rows = payload.get("log") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Invalid posttrain log format: {log_path}")
    last = rows[-1]
    if not isinstance(last, dict):
        raise RuntimeError(f"Invalid final log row format: {log_path}")

    metrics: Dict[str, float] = {}
    for key in METRIC_KEYS:
        val = last.get(key)
        if val is None:
            continue
        try:
            metrics[key] = float(val)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Metric {key!r} is not numeric in {log_path}: {val!r}") from exc
    return metrics


def _assert_artifacts(out_dir: Path, run_name: str) -> Path:
    ckpt_path = out_dir / f"ckpt_last_{run_name}.pth"
    log_path = out_dir / f"posttrain_log_{run_name}.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing log: {log_path}")
    return log_path


def _compare_metrics(current: Dict[str, float], baseline: Dict[str, float], tol: float) -> list[str]:
    failures: list[str] = []
    for key in METRIC_KEYS:
        if key not in current or key not in baseline:
            continue
        curr = float(current[key])
        base = float(baseline[key])
        diff = abs(curr - base)
        print(f"[step6] metric {key:>16s}: current={curr:.10f} baseline={base:.10f} |diff|={diff:.3e}")
        if diff > float(tol):
            failures.append(f"{key}: |{curr} - {base}| = {diff} > tol({tol})")
    return failures


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Step6 minimal posttrain regression checks.")
    ap.add_argument("--python", default=sys.executable, help="Python executable used to run train.posttrain.")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory for smoke runs.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps_per_epoch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_tag", default="step6_20260301", help="Suffix tag used for run names.")

    ap.add_argument("--direct_config", default=DEFAULT_DIRECT_CONFIG)
    ap.add_argument("--lambda_config", default=DEFAULT_LAMBDA_CONFIG)

    ap.add_argument("--baseline_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--baseline_direct_run_name", default="smoke_direct_20260301")
    ap.add_argument("--baseline_lambda_run_name", default="smoke_lambda_20260301")
    ap.add_argument("--metric_tolerance", type=float, default=1e-6)
    ap.add_argument(
        "--skip_baseline_compare",
        action="store_true",
        help="Skip baseline metric comparison (artifact/execution checks still run).",
    )
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        _Case(
            mode="direct",
            config_path=Path(args.direct_config).expanduser().resolve(),
            run_name=f"smoke_direct_{args.run_tag}",
            baseline_run_name=str(args.baseline_direct_run_name),
        ),
        _Case(
            mode="lambda",
            config_path=Path(args.lambda_config).expanduser().resolve(),
            run_name=f"smoke_lambda_{args.run_tag}",
            baseline_run_name=str(args.baseline_lambda_run_name),
        ),
    ]

    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    all_failures: list[str] = []
    for case in cases:
        if not case.config_path.is_file():
            raise FileNotFoundError(f"Config not found: {case.config_path}")

        cmd = [
            str(args.python),
            "-m",
            "train.posttrain",
            "--config",
            str(case.config_path),
            "--epochs",
            str(int(args.epochs)),
            "--steps_per_epoch",
            str(int(args.steps_per_epoch)),
            "--seed",
            str(int(args.seed)),
            "--run_name",
            case.run_name,
            "--out_dir",
            str(out_dir),
        ]
        _run_command(cmd, env=env)

        log_path = _assert_artifacts(out_dir, case.run_name)
        cur_metrics = _load_last_metrics(log_path)
        print(f"[step6] {case.mode} artifacts ready: {log_path}")

        if args.skip_baseline_compare:
            continue

        baseline_log_path = Path(args.baseline_dir).expanduser().resolve() / f"posttrain_log_{case.baseline_run_name}.json"
        if not baseline_log_path.is_file():
            raise FileNotFoundError(
                f"Baseline log not found for {case.mode}: {baseline_log_path}. "
                "Use --skip_baseline_compare if no baseline is available."
            )
        base_metrics = _load_last_metrics(baseline_log_path)
        case_failures = _compare_metrics(cur_metrics, base_metrics, float(args.metric_tolerance))
        if case_failures:
            all_failures.extend([f"[{case.mode}] {msg}" for msg in case_failures])

    if all_failures:
        print("[step6][FAIL] metric regression detected:")
        for msg in all_failures:
            print(f"  - {msg}")
        raise SystemExit(1)

    print("[step6][PASS] minimal regression gate passed.")


if __name__ == "__main__":
    main()
