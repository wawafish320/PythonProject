#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_strict_70r_trunkfull_probe as base_probe

DEFAULT_REPAIRED_REPLACE_CKPT = (
    ROOT
    / "debug_output/_tmp_strict_replace_phasez_boundary_probe_20260427_213254"
    / "baseline/checkpoints/ckpt_last_replace_phasez_baseline__tmp_strict_replace_phasez_boundary_probe_20260427_213254.pth"
)
DEFAULT_70R_DONOR_STEP0 = (
    ROOT
    / "debug_output/_tmp_strict_contract_fullchain_preflight_20260426_173158"
    / "70R_lr_probe/lr1e4_step20/checkpoints/ckpt_step_000000_WalkF_stage7_70R_lr1e4_step20_20260426_173158.pth"
)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict 70R warmstart-bridge probe from repaired replace.")
    parser.add_argument("--base-config", type=Path, default=base_probe.DEFAULT_BASE_CONFIG)
    parser.add_argument("--source-replace-ckpt", type=Path, default=DEFAULT_REPAIRED_REPLACE_CKPT)
    parser.add_argument("--direct-pose-donor-ckpt", type=Path, default=DEFAULT_70R_DONOR_STEP0)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--no-trunk-step0-group", type=Path, default=base_probe.DEFAULT_NO_TRUNK_STEP0)
    parser.add_argument("--no-trunk-step20-group", type=Path, default=base_probe.DEFAULT_NO_TRUNK_STEP20)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.run_root or (ROOT / "debug_output" / f"_tmp_strict_70R_warmstart_bridge_probe_{stamp}")).expanduser().resolve()
    run_name = args.run_name or f"70R_strict_warmstart_bridge_probe_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)

    handoff_ckpt = run_root / "handoffs/replace_to_70R_warmstart_bridge_strict_contract.pth"
    contract_report = run_root / "handoffs/replace_to_70R_warmstart_bridge_contractize.json"
    config = base_probe._make_config(
        base_config=args.base_config.expanduser(),
        run_root=run_root,
        handoff_ckpt=handoff_ckpt,
        run_name=run_name,
    )
    base_probe._contractize(
        source_ckpt=args.source_replace_ckpt.expanduser(),
        target_config=config,
        handoff_ckpt=handoff_ckpt,
        report=contract_report,
        run_root=run_root,
        tensor_donor_ckpt=args.direct_pose_donor_ckpt.expanduser(),
        transplant_prefixes=("direct_pose_",),
    )
    base_probe._train(config, run_root)

    out_dir = run_root / "checkpoints"
    step0 = out_dir / f"ckpt_step_000000_{run_name}.pth"
    step20 = out_dir / f"ckpt_step_000020_{run_name}.pth"
    if not step0.is_file() or not step20.is_file():
        raise RuntimeError(f"missing required step ckpts: step0={step0.is_file()} step20={step20.is_file()}")

    group0 = base_probe._eval_ckpt(ckpt=step0, step=0, run_root=run_root)
    group20 = base_probe._eval_ckpt(ckpt=step20, step=20, run_root=run_root)
    g0 = base_probe._group_means(group0)
    g20 = base_probe._group_means(group20)
    probe_delta = base_probe._delta(g0, g20)

    no_trunk = None
    if args.no_trunk_step0_group.is_file() and args.no_trunk_step20_group.is_file():
        nt0 = base_probe._group_means(args.no_trunk_step0_group)
        nt20 = base_probe._group_means(args.no_trunk_step20_group)
        no_trunk = {"step0": nt0, "step20": nt20, "delta": base_probe._delta(nt0, nt20)}

    train_summary = base_probe._train_log_summary(run_root / "logs/70R_train.log")
    changed = base_probe._changed_tensors(step0, step20)
    summary = {
        "run_root": str(run_root),
        "config": str(config),
        "warmstart_bridge_handoff": str(handoff_ckpt),
        "warmstart_bridge_contract_report": str(contract_report),
        "source_replace_ckpt": str(args.source_replace_ckpt.expanduser()),
        "direct_pose_donor_ckpt": str(args.direct_pose_donor_ckpt.expanduser()),
        "transplant_prefixes": ["direct_pose_"],
        "bridge_policy": {
            "kind": "migration_time_70R_warmstart_bridge",
            "strict_current_model_build": True,
            "load_context": "chain_hop",
            "bridge_semantics": "full direct_pose bundle donor alignment before 70R train",
        },
        "strict_policy": {
            "strict_current_model_build": True,
            "load_context": "chain_hop",
            "contains_chain_hop_waiver": bool(train_summary["contains_chain_hop_waiver"]),
            "contains_policy_strict_current": bool(train_summary["contains_policy_strict_current"]),
            "contains_strict_shape_validation": bool(train_summary["contains_strict_shape_validation"]),
        },
        "train": train_summary,
        "evals": {
            "step0_group_summary": str(group0),
            "step20_group_summary": str(group20),
            "step0": g0,
            "step20": g20,
            "delta_step20_minus_step0": probe_delta,
            "current_no_trunk_reference": no_trunk,
        },
        "changed_tensors": changed,
    }
    _dump_json(run_root / "probe_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
