#!/usr/bin/env python3
"""
Pose-only gate runner for E1 vs E2.

Workflow:
1) Run strict freerun eval for each checkpoint (optional if JSON already exists).
2) Aggregate per-seed GeoDirectDeg(=DirectGeoDeg) and DirectGeoLocalDeg.
3) Decide whether E3 can be started.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SEED_PATTERNS = (
    re.compile(r"(?:^|[_-])seed(\d+)(?:$|[_-])", re.IGNORECASE),
    re.compile(r"(?:^|[_-])s(\d+)(?:$|[_-])", re.IGNORECASE),
)


@dataclass
class RunRecord:
    label: str
    seed: int
    ckpt: Path
    freerun_json: Path
    primary: float
    secondary: float
    per_round_primary: List[float]
    per_round_secondary: List[float]


def _run(cmd: List[str], *, cwd: Path) -> str:
    env = os.environ.copy()
    py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "." if not py else f".:{py}"
    print("[CMD]", " ".join(cmd))
    cp = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return cp.stdout


def _parse_seed_id(path: Path, fallback: int) -> int:
    for text in (path.stem, path.name):
        for pat in SEED_PATTERNS:
            m = pat.search(text)
            if m:
                return int(m.group(1))
    return fallback


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _mean(vals: Iterable[float]) -> float:
    items = list(vals)
    if not items:
        return float("nan")
    return float(statistics.fmean(items))


def _std(vals: Iterable[float]) -> float:
    items = list(vals)
    if len(items) <= 1:
        return 0.0
    return float(statistics.pstdev(items))


def _load_round_metrics(path: Path, primary_key: str, secondary_key: str) -> RunRecord:
    obj = json.loads(path.read_text())
    rounds = list(obj.get("metrics_per_round", []) or [])
    if not rounds:
        raise RuntimeError(f"{path}: metrics_per_round is empty.")
    pvals: List[float] = []
    svals: List[float] = []
    for idx, r in enumerate(rounds):
        if not isinstance(r, dict):
            continue
        p = _safe_float(r.get(primary_key))
        s = _safe_float(r.get(secondary_key))
        if p is None or s is None:
            raise RuntimeError(
                f"{path}: round {idx} missing keys {primary_key}/{secondary_key} or value is non-finite."
            )
        pvals.append(p)
        svals.append(s)
    if not pvals or not svals:
        raise RuntimeError(f"{path}: no valid per-round values for {primary_key}/{secondary_key}.")
    return RunRecord(
        label="",
        seed=-1,
        ckpt=Path(""),
        freerun_json=path,
        primary=_mean(pvals),
        secondary=_mean(svals),
        per_round_primary=pvals,
        per_round_secondary=svals,
    )


def _run_freerun(
    *,
    repo_root: Path,
    python_bin: str,
    teacher: Path,
    ckpt: Path,
    out_dir: Path,
    rounds: int,
    force: bool,
) -> str:
    cmd = [
        python_bin,
        "-m",
        "train.validate.run_freerun_cycles",
        "--teacher",
        str(teacher),
        "--model",
        str(ckpt),
        "--rounds",
        str(rounds),
        "--out",
        str(out_dir),
    ]
    if force:
        cmd.append("--force")
    return _run(cmd, cwd=repo_root)


def _summarize_group(runs: List[RunRecord]) -> Dict[str, Any]:
    primary = [r.primary for r in runs]
    secondary = [r.secondary for r in runs]
    return {
        "n": len(runs),
        "primary_mean": _mean(primary) if primary else None,
        "primary_std": _std(primary) if primary else None,
        "secondary_mean": _mean(secondary) if secondary else None,
        "secondary_std": _std(secondary) if secondary else None,
        "seeds": [int(r.seed) for r in runs],
        "runs": [
            {
                "seed": int(r.seed),
                "ckpt": str(r.ckpt),
                "freerun_json": str(r.freerun_json),
                "primary": float(r.primary),
                "secondary": float(r.secondary),
                "per_round_primary": [float(x) for x in r.per_round_primary],
                "per_round_secondary": [float(x) for x in r.per_round_secondary],
            }
            for r in runs
        ],
    }


def _build_gate(
    *,
    e1_runs: List[RunRecord],
    e2_runs: List[RunRecord],
    expected_seeds: int,
    min_primary_seed_wins: int,
    min_secondary_seed_wins: int,
    std_slack: float,
) -> Dict[str, Any]:
    e1_by_seed = {r.seed: r for r in e1_runs}
    e2_by_seed = {r.seed: r for r in e2_runs}
    shared = sorted(set(e1_by_seed.keys()) & set(e2_by_seed.keys()))
    paired = []
    for seed in shared:
        a = e1_by_seed[seed]
        b = e2_by_seed[seed]
        paired.append(
            {
                "seed": int(seed),
                "e1_primary": float(a.primary),
                "e2_primary": float(b.primary),
                "delta_primary_e1_minus_e2": float(a.primary - b.primary),
                "e1_secondary": float(a.secondary),
                "e2_secondary": float(b.secondary),
                "delta_secondary_e1_minus_e2": float(a.secondary - b.secondary),
            }
        )

    e1_primary_vals = [r.primary for r in e1_runs]
    e2_primary_vals = [r.primary for r in e2_runs]
    e1_secondary_vals = [r.secondary for r in e1_runs]
    e2_secondary_vals = [r.secondary for r in e2_runs]

    e1_primary_mean = _mean(e1_primary_vals) if e1_primary_vals else None
    e2_primary_mean = _mean(e2_primary_vals) if e2_primary_vals else None
    e1_secondary_mean = _mean(e1_secondary_vals) if e1_secondary_vals else None
    e2_secondary_mean = _mean(e2_secondary_vals) if e2_secondary_vals else None
    e1_primary_std = _std(e1_primary_vals) if e1_primary_vals else None
    e2_primary_std = _std(e2_primary_vals) if e2_primary_vals else None

    primary_seed_wins = sum(1 for it in paired if it["delta_primary_e1_minus_e2"] > 0.0)
    secondary_seed_wins = sum(1 for it in paired if it["delta_secondary_e1_minus_e2"] > 0.0)

    checks = {
        "enough_shared_seeds": len(shared) >= expected_seeds,
        "primary_mean_better": (
            e1_primary_mean is not None
            and e2_primary_mean is not None
            and e2_primary_mean < e1_primary_mean
        ),
        "secondary_mean_better": (
            e1_secondary_mean is not None
            and e2_secondary_mean is not None
            and e2_secondary_mean < e1_secondary_mean
        ),
        "primary_seed_wins": primary_seed_wins >= min_primary_seed_wins,
        "secondary_seed_wins": secondary_seed_wins >= min_secondary_seed_wins,
        "primary_std_not_worse": (
            e1_primary_std is not None
            and e2_primary_std is not None
            and e2_primary_std <= (e1_primary_std + std_slack)
        ),
    }
    gate_pass = all(bool(v) for v in checks.values())
    decision = "enter_e3_apply" if gate_pass else "hold_pose_only"

    return {
        "decision": decision,
        "gate_pass": bool(gate_pass),
        "checks": checks,
        "expected_seeds": int(expected_seeds),
        "shared_seeds": shared,
        "min_primary_seed_wins": int(min_primary_seed_wins),
        "min_secondary_seed_wins": int(min_secondary_seed_wins),
        "std_slack": float(std_slack),
        "primary_seed_wins": int(primary_seed_wins),
        "secondary_seed_wins": int(secondary_seed_wins),
        "paired_by_seed": paired,
    }


def _fmt(v: Any, ndigits: int = 4) -> str:
    x = _safe_float(v)
    if x is None:
        return "NA"
    return f"{x:.{ndigits}f}"


def _write_md(path: Path, summary: Dict[str, Any]) -> None:
    metric = summary["metric_policy"]
    e1 = summary["E1"]
    e2 = summary["E2"]
    gate = summary["gate"]

    lines: List[str] = []
    lines.append("# Pose-only E2 stability summary")
    lines.append("")
    lines.append(
        f"- Primary metric: {metric['primary_alias']} (source key `{metric['primary_key']}`; lower is better)"
    )
    lines.append(
        f"- Secondary metric: {metric['secondary_key']} (lower is better)"
    )
    lines.append(f"- Decision: **{gate['decision']}** (gate_pass={gate['gate_pass']})")
    lines.append("")
    lines.append("| Group | n | primary mean | primary std | secondary mean | secondary std |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| E1 | {e1['n']} | {_fmt(e1['primary_mean'])} | {_fmt(e1['primary_std'])} | {_fmt(e1['secondary_mean'])} | {_fmt(e1['secondary_std'])} |"
    )
    lines.append(
        f"| E2 | {e2['n']} | {_fmt(e2['primary_mean'])} | {_fmt(e2['primary_std'])} | {_fmt(e2['secondary_mean'])} | {_fmt(e2['secondary_std'])} |"
    )
    lines.append("")
    lines.append("| seed | E1 primary | E2 primary | d(E1-E2) | E1 secondary | E2 secondary | d(E1-E2) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in gate.get("paired_by_seed", []):
        lines.append(
            f"| {row['seed']} | {_fmt(row['e1_primary'])} | {_fmt(row['e2_primary'])} | {_fmt(row['delta_primary_e1_minus_e2'])} | {_fmt(row['e1_secondary'])} | {_fmt(row['e2_secondary'])} | {_fmt(row['delta_secondary_e1_minus_e2'])} |"
        )
    lines.append("")
    lines.append("## Gate checks")
    for k, v in gate.get("checks", {}).items():
        lines.append(f"- {k}: {bool(v)}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run pose-only E2 gate (3-seed stability vs E1).")
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument("--python-bin", type=str, default="python3")
    ap.add_argument(
        "--teacher",
        type=str,
        default="validate/teacher_batches/Walk_F_teacher.json",
    )
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--e1-ckpt", action="append", default=[], help="Path to E1 checkpoint. Repeatable.")
    ap.add_argument("--e2-ckpt", action="append", default=[], help="Path to E2 checkpoint. Repeatable.")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--primary-key", type=str, default="DirectGeoDeg")
    ap.add_argument("--primary-alias", type=str, default="GeoDirectDeg")
    ap.add_argument("--secondary-key", type=str, default="DirectGeoLocalDeg")
    ap.add_argument("--expected-seeds", type=int, default=3)
    ap.add_argument("--min-primary-seed-wins", type=int, default=3)
    ap.add_argument("--min-secondary-seed-wins", type=int, default=2)
    ap.add_argument("--std-slack", type=float, default=0.02)
    ap.add_argument("--skip-freerun-if-exists", action="store_true")
    ap.add_argument("--only-summarize", action="store_true")
    ap.add_argument("--force-freerun", action="store_true")
    args = ap.parse_args()

    if not args.e1_ckpt or not args.e2_ckpt:
        raise SystemExit("[FATAL] Please provide at least one --e1-ckpt and one --e2-ckpt.")

    repo_root = Path(args.repo_root).expanduser().resolve()
    teacher = (repo_root / args.teacher).resolve() if not Path(args.teacher).is_absolute() else Path(args.teacher)
    if not teacher.is_file():
        raise SystemExit(f"[FATAL] teacher file not found: {teacher}")

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    def collect(label: str, ckpt_list: List[str]) -> List[RunRecord]:
        records: List[RunRecord] = []
        for idx, ck in enumerate(ckpt_list):
            ckpt = Path(ck).expanduser().resolve()
            if not ckpt.is_file():
                raise SystemExit(f"[FATAL] missing checkpoint: {ckpt}")
            seed = _parse_seed_id(ckpt, fallback=idx)
            seed_dir = out_root / label / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            freerun_json = seed_dir / "Walk_F_freerun_cycles.json"
            run_freerun = not args.only_summarize
            if args.skip_freerun_if_exists and freerun_json.is_file():
                run_freerun = False
            if run_freerun:
                log = _run_freerun(
                    repo_root=repo_root,
                    python_bin=args.python_bin,
                    teacher=teacher,
                    ckpt=ckpt,
                    out_dir=seed_dir,
                    rounds=args.rounds,
                    force=bool(args.force_freerun),
                )
                (seed_dir / "run_freerun.log").write_text(log)
            if not freerun_json.is_file():
                raise SystemExit(f"[FATAL] missing freerun JSON: {freerun_json}")
            parsed = _load_round_metrics(
                freerun_json,
                primary_key=args.primary_key,
                secondary_key=args.secondary_key,
            )
            parsed.label = label
            parsed.seed = int(seed)
            parsed.ckpt = ckpt
            records.append(parsed)
        records.sort(key=lambda x: x.seed)
        return records

    e1_runs = collect("E1", args.e1_ckpt)
    e2_runs = collect("E2", args.e2_ckpt)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metric_policy": {
            "primary_alias": args.primary_alias,
            "primary_key": args.primary_key,
            "secondary_key": args.secondary_key,
            "lower_is_better": True,
        },
        "E1": _summarize_group(e1_runs),
        "E2": _summarize_group(e2_runs),
    }
    summary["gate"] = _build_gate(
        e1_runs=e1_runs,
        e2_runs=e2_runs,
        expected_seeds=int(args.expected_seeds),
        min_primary_seed_wins=int(args.min_primary_seed_wins),
        min_secondary_seed_wins=int(args.min_secondary_seed_wins),
        std_slack=float(args.std_slack),
    )

    out_json = out_root / "pose_only_e2_gate_summary.json"
    out_md = out_root / "pose_only_e2_gate_summary.md"
    out_json.write_text(json.dumps(summary, indent=2))
    _write_md(out_md, summary)

    print(f"[OK] wrote: {out_json}")
    print(f"[OK] wrote: {out_md}")
    print(f"[decision] {summary['gate']['decision']} (gate_pass={summary['gate']['gate_pass']})")


if __name__ == "__main__":
    main()
