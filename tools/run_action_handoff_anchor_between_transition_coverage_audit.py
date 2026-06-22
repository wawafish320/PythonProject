#!/usr/bin/env python3
"""Rows-only transition coverage audit for anchor-between acceptance replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0"}


def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = float(k) / float(n)
    denom = 1.0 + z * z / float(n)
    center = (p + z * z / (2.0 * float(n))) / denom
    half = z * math.sqrt(p * (1.0 - p) / float(n) + z * z / (4.0 * float(n) * float(n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _partition(row: Mapping[str, Any]) -> str:
    return str(row.get("split_partition") or row.get("partition") or "all")


def _transition(row: Mapping[str, Any]) -> str:
    return str(row.get("oracle_endpoint_transition") or "unknown")


def _seed(row: Mapping[str, Any]) -> str:
    return str(row.get("contact_seed") or "")


def _unique_window_rows(rows: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = (str(row.get("clip")), int(row.get("start", 0)))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _independent_rows(rows: Iterable[Mapping[str, Any]], horizon: int) -> List[Mapping[str, Any]]:
    by_key: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in _unique_window_rows(rows):
        by_key[(str(row.get("clip")), _transition(row))].append(row)
    kept: List[Mapping[str, Any]] = []
    for key in sorted(by_key):
        last = -10**9
        for row in sorted(by_key[key], key=lambda r: int(r.get("start", 0))):
            start = int(row.get("start", 0))
            if start - last >= int(horizon):
                kept.append(row)
                last = start
    return kept


def _coverage_rows(rows: Sequence[Mapping[str, Any]], horizon: int) -> List[Dict[str, Any]]:
    transitions = sorted({_transition(row) for row in rows})
    out: List[Dict[str, Any]] = []
    for transition in transitions:
        train = [row for row in rows if _transition(row) == transition and _partition(row) == "train"]
        test = [row for row in rows if _transition(row) == transition and _partition(row) == "test"]
        train_eff = _independent_rows(train, horizon)
        test_eff = _independent_rows(test, horizon)
        out.append(
            {
                "transition": transition,
                "train_raw_n": len(_unique_window_rows(train)),
                "train_independent_n": len(train_eff),
                "test_raw_n": len(_unique_window_rows(test)),
                "test_independent_n": len(test_eff),
                "train_keys": [(str(row.get("clip")), int(row.get("start", 0))) for row in train_eff],
                "test_keys": [(str(row.get("clip")), int(row.get("start", 0))) for row in test_eff],
            }
        )
    return out


def _endpoint_rows(rows: Sequence[Mapping[str, Any]], horizon: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seed in sorted({_seed(row) for row in rows}):
        scoped = [row for row in rows if _seed(row) == seed and _partition(row) == "test"]
        if not scoped:
            continue
        for transition in sorted({_transition(row) for row in scoped}):
            raw = [row for row in scoped if _transition(row) == transition]
            eff = _independent_rows(raw, horizon)
            raw_k = sum(1 for row in _unique_window_rows(raw) if _as_bool(row.get("endpoint_bridgeability")))
            eff_k = sum(1 for row in eff if _as_bool(row.get("endpoint_bridgeability")))
            raw_n = len(_unique_window_rows(raw))
            eff_n = len(eff)
            raw_rate, raw_lo, raw_hi = _wilson(raw_k, raw_n)
            eff_rate, eff_lo, eff_hi = _wilson(eff_k, eff_n)
            out.append(
                {
                    "seed": seed,
                    "transition": transition,
                    "raw_k": raw_k,
                    "raw_n": raw_n,
                    "raw_rate": raw_rate,
                    "raw_ci95_lo": raw_lo,
                    "raw_ci95_hi": raw_hi,
                    "independent_k": eff_k,
                    "independent_n": eff_n,
                    "independent_rate": eff_rate,
                    "independent_ci95_lo": eff_lo,
                    "independent_ci95_hi": eff_hi,
                    "independent_keys": [
                        (str(row.get("clip")), int(row.get("start", 0)), _as_bool(row.get("endpoint_bridgeability")))
                        for row in eff
                    ],
                }
            )
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--min-train-independent", type=int, default=0)
    parser.add_argument("--min-test-independent", type=int, default=0)
    parser.add_argument("--fail-on-low-coverage", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.rows)
    coverage = _coverage_rows(rows, int(args.horizon))
    endpoints = _endpoint_rows(rows, int(args.horizon))
    low = [
        row
        for row in coverage
        if int(row["test_independent_n"]) > 0
        and (
            int(row["train_independent_n"]) < int(args.min_train_independent)
            or int(row["test_independent_n"]) < int(args.min_test_independent)
        )
    ]
    payload = {
        "rows": str(args.rows),
        "horizon": int(args.horizon),
        "min_train_independent": int(args.min_train_independent),
        "min_test_independent": int(args.min_test_independent),
        "coverage": coverage,
        "endpoint_by_seed": endpoints,
        "low_coverage": low,
        "pass": bool(not low),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "transition_coverage_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(args.out_dir / "transition_coverage.csv", coverage)
    _write_csv(args.out_dir / "endpoint_by_seed.csv", endpoints)
    if low and bool(args.fail_on_low_coverage):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
