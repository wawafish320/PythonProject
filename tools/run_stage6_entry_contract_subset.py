#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


from run_stage6_entry_contract_matrix import (
    DEFAULT_MANIFEST,
    _load_resources,
    _materialize_family_config,
    _load_json,
    _metrics_for_eval_json,
    _discover_candidates,
    _require_paths,
    _resolve,
    _run_stage6_candidate,
    log,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--family", required=True)
    ap.add_argument("--candidates", required=True, help="Comma-separated candidate list.")
    ap.add_argument("--force-stage6", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    wanted = {item.strip() for item in str(args.candidates).split(",") if item.strip()}
    if not wanted:
        raise SystemExit("[FATAL] no candidates selected")

    resources, families = _load_resources(_resolve(args.manifest))
    family = next((item for item in families if item.family == args.family), None)
    if family is None:
        raise SystemExit(f"[FATAL] family not found in manifest: {args.family}")

    _require_paths(
        [
            resources.base_config,
            resources.stage6_config,
            resources.wrapper,
            resources.teacher,
            resources.encoder_bundle,
            resources.affine_stats,
            resources.old_stage6_exit_baseline,
        ]
    )
    resources.basetrain_out_root.mkdir(parents=True, exist_ok=True)
    resources.stage6_out_root.mkdir(parents=True, exist_ok=True)
    resources.debug_root.mkdir(parents=True, exist_ok=True)
    resources.materialized_config_root.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_json(resources.base_config)
    run_tag = resources.name.rsplit("_", 1)[-1] if "_" in resources.name else resources.name
    _, exp_dir, run_name = _materialize_family_config(resources, base_cfg, family, run_tag)
    baseline_metrics = _metrics_for_eval_json(resources.old_stage6_exit_baseline, cycle_gte=1, exclude_sic01=False)

    candidates, missing = _discover_candidates(exp_dir, run_name) if exp_dir.is_dir() else ([], ["exp_dir_missing"])
    if missing:
        log(f"[warn] discover missing for {family.family}: {', '.join(missing)}")

    selected = [candidate for candidate in candidates if candidate.candidate in wanted]
    unknown = sorted(wanted - {candidate.candidate for candidate in selected})
    if unknown:
        raise SystemExit(f"[FATAL] unknown candidates for {family.family}: {', '.join(unknown)}")

    for candidate in selected:
        candidate.family = family.family
        log(f"[subset-stage6] {family.family} / {candidate.candidate} -> {candidate.source_ckpt}")
        row = _run_stage6_candidate(
            resources,
            candidate,
            baseline_metrics,
            force_stage6=bool(args.force_stage6),
            dry_run=bool(args.dry_run),
        )
        log(
            f"[subset-done] {family.family} / {candidate.candidate} "
            f"missing={row.get('missing')} blended={row.get('blended_distance_to_old_exit')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
