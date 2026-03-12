#!/usr/bin/env python3
"""
Print markdown tables from `train.validate.run_freerun_cycles` JSON outputs.

Examples
--------
1) Multi-stage table (per-round metric):

  python tools/print_freerun_cycles_tables.py \\
    --metric BlendGeoLocalDeg \\
    --stage "Base (noapply)=debug_output/bridge_table/base_noapply/Walk_F_freerun_cycles.json" \\
    --stage "planinit_obs (noapply)=debug_output/bridge_table/planinit_obs_noapply/Walk_F_freerun_cycles.json" \\
    --stage "lambda_fusion (apply)=debug_output/bridge_table/lambda_apply/Walk_F_freerun_cycles.json" \\
    --stage "so3corr ckpt (lambda apply)=debug_output/bridge_table/so3corr_lambda_apply_rwarmup10/Walk_F_freerun_cycles.json" \\
    --stage "FINAL (lambda+so3 apply)=debug_output/posttrain_direct_pose_rerun/after_keepgate_lambda_retrained/Walk_F_freerun_cycles.json"

2) Baseline vs After table(s):

  python tools/print_freerun_cycles_tables.py \\
    --compare \\
    --baseline debug_output/bridge_table/so3corr_lambda_so3_apply_rwarmup10/Walk_F_freerun_cycles.json \\
    --after debug_output/posttrain_direct_pose_rerun/after_keepgate_lambda_retrained/Walk_F_freerun_cycles.json \\
    --compare-after-label "After" \\
    --compare-metric BlendGeoLocalDeg \\
    --compare-metric DirectGeoLocalDeg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _extract_per_round(obj: Dict, metric: str) -> List[float]:
    mp = obj.get("metrics_per_round", None)
    if not isinstance(mp, list) or not mp:
        raise ValueError("Invalid JSON: missing metrics_per_round list.")
    out: List[float] = []
    for r in mp:
        if not isinstance(r, dict):
            raise ValueError("Invalid JSON: metrics_per_round entries must be dicts.")
        v = r.get(metric, None)
        if v is None:
            raise KeyError(f"Missing metric '{metric}' in metrics_per_round.")
        out.append(float(v))
    return out


def _fmt(metric: str, x: float) -> str:
    # Most freerun metrics are degrees, but some (e.g. RootPosErrMean / LambdaMean) are not.
    if "Deg" in str(metric):
        return f"{x:.2f}°"
    return f"{x:.4f}"


def _print_stage_table(metric: str, stages: List[Tuple[str, Path]]) -> None:
    vals_by_stage: List[Tuple[str, List[float]]] = []
    for label, path in stages:
        obj = _load_json(path)
        vals_by_stage.append((label, _extract_per_round(obj, metric)))

    rounds = min(len(v) for _, v in vals_by_stage)
    headers = ["Round"] + [label for label, _ in vals_by_stage]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for i in range(rounds):
        row = [f"R{i}"]
        for _, v in vals_by_stage:
            row.append(_fmt(metric, v[i]))
        print("| " + " | ".join(row) + " |")
    if rounds >= 5:
        row = ["R1-4 Mean"]
        for _, v in vals_by_stage:
            row.append(_fmt(metric, mean(v[1:5])))
        print("| " + " | ".join(row) + " |")


def _print_compare_table(metric: str, baseline: Path, after: Path, after_label: str) -> None:
    base = _extract_per_round(_load_json(baseline), metric)
    aft = _extract_per_round(_load_json(after), metric)
    rounds = min(len(base), len(aft))

    label = (after_label or "After").strip() or "After"
    print(f"| Round | Baseline {metric} | {label} | 改善 |")
    print("|---|---:|---:|---:|")
    for i in range(rounds):
        print(f"| R{i} | {_fmt(metric, base[i])} | {_fmt(metric, aft[i])} | {_fmt(metric, aft[i] - base[i])} |")
    if rounds >= 5:
        b = mean(base[1:5])
        a = mean(aft[1:5])
        print(f"| R1-4 Mean | {_fmt(metric, b)} | {_fmt(metric, a)} | {_fmt(metric, a - b)} |")


def main() -> None:
    ap = argparse.ArgumentParser(description="Print markdown tables from freerun_cycles JSON outputs.")
    ap.add_argument("--metric", type=str, default="BlendGeoLocalDeg", help="Metric key under metrics_per_round.")
    ap.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Stage spec: 'Label=path/to/Walk_F_freerun_cycles.json' (repeatable).",
    )
    ap.add_argument("--compare", action="store_true", help="Also print baseline-vs-after compare tables.")
    ap.add_argument("--baseline", type=str, default=None, help="Baseline JSON path for --compare.")
    ap.add_argument("--after", type=str, default=None, help="After JSON path for --compare.")
    ap.add_argument(
        "--compare-after-label",
        type=str,
        default="After",
        help="Column title for the 'after' run in --compare table(s).",
    )
    ap.add_argument(
        "--compare-metric",
        action="append",
        default=[],
        help="Metric key(s) for compare table(s) (repeatable).",
    )
    args = ap.parse_args()

    stages: List[Tuple[str, Path]] = []
    for spec in args.stage:
        if "=" not in str(spec):
            raise SystemExit(f"Invalid --stage spec (expected Label=Path): {spec}")
        label, path = str(spec).split("=", 1)
        label = label.strip()
        p = Path(path).expanduser()
        if not p.is_file():
            raise SystemExit(f"--stage file not found: {p}")
        stages.append((label, p))

    if stages:
        _print_stage_table(str(args.metric), stages)

    if args.compare:
        if not args.baseline or not args.after:
            raise SystemExit("--compare requires --baseline and --after.")
        baseline = Path(args.baseline).expanduser()
        after = Path(args.after).expanduser()
        if not baseline.is_file():
            raise SystemExit(f"--baseline file not found: {baseline}")
        if not after.is_file():
            raise SystemExit(f"--after file not found: {after}")
        metrics = [str(x).strip() for x in (args.compare_metric or []) if str(x).strip()]
        if not metrics:
            metrics = [str(args.metric)]
        print()
        for i, m in enumerate(metrics):
            if i > 0:
                print()
            _print_compare_table(m, baseline, after, str(getattr(args, "compare_after_label", "After")))


if __name__ == "__main__":
    main()
