#!/usr/bin/env python3
"""Score Stage7 lambda-final candidates with a single runtime-blend composite score.

Input: summary JSON from `tools/build_stage7_lambda_blend_summary.py` aggregation,
for example `debug_output/_tmp_blendaware_expbaseline_summary_20260308/summary.json`.

The score intentionally focuses on the *actual runtime output* path when
`lambda_fusion_apply=true`, i.e. the blend pose and its visual hotspots, not the
standalone direct expert.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


PROFILES: Dict[str, Dict[str, float]] = {
    "runtime_blend_balanced": {
        "blend_mean_new": 0.25,
        "blend_p95_new": 0.20,
        "blend_p99_new": 0.15,
        "blend_max_new": 0.05,
        "foot_mean_new": 0.15,
        "foot_p95_new": 0.10,
        "calf_mean_new": 0.07,
        "calf_p95_new": 0.03,
    },
    "runtime_blend_tail": {
        "blend_mean_new": 0.10,
        "blend_p95_new": 0.25,
        "blend_p99_new": 0.20,
        "blend_max_new": 0.10,
        "foot_mean_new": 0.10,
        "foot_p95_new": 0.10,
        "calf_mean_new": 0.05,
        "calf_p95_new": 0.10,
    },
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(rows: List[Dict[str, Any]], metric: str, value: float) -> float:
    vals = [float(r[metric]) for r in rows]
    lo = min(vals)
    hi = max(vals)
    if hi - lo < 1e-12:
        return 0.0
    return float((float(value) - lo) / (hi - lo))


def _score_rows(rows: List[Dict[str, Any]], weights: Dict[str, float]) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for row in rows:
        comps = {metric: _norm(rows, metric, float(row[metric])) for metric in weights.keys()}
        score = float(sum(comps[m] * float(w) for m, w in weights.items()))
        scored.append({
            "s": int(row["s"]),
            "score": score,
            "components": comps,
            "raw": {k: row[k] for k in weights.keys()},
        })
    scored.sort(key=lambda x: (float(x["score"]), int(x["s"])))
    return scored


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--profile", type=str, default="runtime_blend_balanced", choices=sorted(PROFILES.keys()))
    args = ap.parse_args()

    summary_path = Path(args.summary_json).expanduser()
    payload = _load_json(summary_path)
    rows = list(payload.get("rows", []))
    if not rows:
        raise SystemExit(f"[FATAL] no rows in {summary_path}")

    weights = PROFILES[str(args.profile)]
    scored = _score_rows(rows, weights)
    best = scored[0]

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# Stage7 runtime-blend score",
        "",
        f"summary_json={summary_path.as_posix()}",
        f"profile={args.profile}",
        "",
        "## Weights",
    ]
    for metric, weight in weights.items():
        md_lines.append(f"- {metric}: {float(weight):.4f}")
    md_lines += [
        "",
        "## Ranking",
        "",
        "| rank | s | score | " + " | ".join(weights.keys()) + " |",
        "|---|---:|---:|" + "|".join(["---:"] * len(weights)) + "|",
    ]
    for rank, row in enumerate(scored, start=1):
        comp_vals = [f"{float(row['components'][m]):.4f}" for m in weights.keys()]
        md_lines.append(f"| {rank} | {int(row['s'])} | {float(row['score']):.6f} | " + " | ".join(comp_vals) + " |")
    md_lines += [
        "",
        f"best_s={int(best['s'])}",
        f"best_score={float(best['score']):.6f}",
    ]

    (out_dir / "runtime_blend_score.md").write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    (out_dir / "runtime_blend_score.json").write_text(
        json.dumps(
            {
                "summary_json": summary_path.as_posix(),
                "profile": args.profile,
                "weights": weights,
                "ranking": scored,
                "best_s": int(best["s"]),
                "best_score": float(best["score"]),
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print((out_dir / "runtime_blend_score.md").as_posix())
    print((out_dir / "runtime_blend_score.json").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
