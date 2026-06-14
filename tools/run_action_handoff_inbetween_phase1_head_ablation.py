#!/usr/bin/env python3
"""PHASE 1 goal-head capacity / injection ablation for §7.3 action handoff.

Runs a small frozen-base grid by delegating each run to the reach-aware rewire probe. The
gate is intentionally cheap and honest: only plateaued runs are eligible; if any plateaued
configuration reaches any clip (min_norm <= conv_norm_thr), PHASE 2 is not warranted.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.action_handoff_inbetween_reach import DEFAULT_CONV_NORM_THR  # noqa: E402
from train.data.action_handoff_inbetween import TURN_CLIPS  # noqa: E402


@dataclass(frozen=True)
class AblationSpec:
    name: str
    hidden: int
    depth: int
    mode: str
    targets: str


def default_grid() -> List[AblationSpec]:
    return [
        AblationSpec("small_add_s1", 256, 1, "additive", "shared_encoder.1"),
        AblationSpec("mid_add_s1", 512, 2, "additive", "shared_encoder.1"),
        AblationSpec("large_add_s1", 1024, 3, "additive", "shared_encoder.1"),
        AblationSpec("mid_film_s1", 512, 2, "film", "shared_encoder.1"),
        AblationSpec("mid_add_early_s0", 512, 2, "additive", "shared_encoder.0"),
        AblationSpec("mid_add_multi_s0_s1", 512, 2, "additive", "shared_encoder.0,shared_encoder.1"),
    ]


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None or not np.isfinite(float(v)):
        return "null"
    return f"{float(v):.{digits}f}"


def _load_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "reach_aware_rewire_probe_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing run summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(spec: AblationSpec, summary: Dict[str, Any], *, conv_norm_thr: float) -> Dict[str, Any]:
    per_clip = summary.get("per_clip", {})
    plateau_status = summary.get("lever2_hidden_pre_loss", {}).get("plateau_status", {})
    eval_plateau_status = plateau_status.get("eval_min_norm", {})
    plateau = bool(
        plateau_status.get("plateau", False)
    )
    eval_min_norm_plateau = bool(eval_plateau_status.get("plateau", plateau))
    clip_rows: Dict[str, Dict[str, Any]] = {}
    any_reached = False
    for clip in TURN_CLIPS:
        row = per_clip.get(clip, {})
        min_norm_min = float(row.get("reach_min_norm_min", float("nan")))
        reached = bool(np.isfinite(min_norm_min) and min_norm_min <= float(conv_norm_thr))
        any_reached = bool(any_reached or reached)
        clip_rows[clip] = {
            "reach_rate": float(row.get("reach_rate", float("nan"))),
            "reach_min_norm_mean": float(row.get("reach_min_norm_mean", float("nan"))),
            "reach_min_norm_min": min_norm_min,
            "reached_by_min_norm": reached,
        }
    return {
        "name": spec.name,
        "head": {"hidden": spec.hidden, "depth": spec.depth, "mode": spec.mode},
        "injection_targets": spec.targets,
        "plateau": plateau,
        "eval_min_norm_plateau": eval_min_norm_plateau,
        "eval_min_norm_plateau_status": eval_plateau_status,
        "any_clip_reached": bool(any_reached),
        "eligible_success": bool(eval_min_norm_plateau and any_reached),
        "per_clip": clip_rows,
        "l_r": clip_rows.get("Walk_L_To_R", {}),
        "run_decision": summary.get("binding_gate_decision", {}),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PHASE 1 frozen-base goal-head capacity/injection ablation.")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-runs", type=int, default=6)
    p.add_argument("--stop-on-success", action="store_true", default=True)
    p.add_argument("--no-stop-on-success", action="store_false", dest="stop_on_success")
    p.add_argument("--probe-script", type=str, default="tools/run_action_handoff_inbetween_reach_aware_rewire_probe.py")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--train-steps", type=int, default=1200)
    p.add_argument("--eval-every", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-schedule", type=str, default="step", choices=["none", "cosine", "step"])
    p.add_argument("--lr-step-size", type=int, default=400)
    p.add_argument("--lr-step-gamma", type=float, default=0.5)
    p.add_argument("--lr-floor", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--min-upgrade-train-steps", type=int, default=600)
    p.add_argument("--plateau-window", type=int, default=80)
    p.add_argument("--eval-plateau-window", type=int, default=3)
    p.add_argument("--eval-plateau-min-samples", type=int, default=6)
    p.add_argument("--plateau-rel-delta", type=float, default=0.02)
    p.add_argument("--conv-norm-thr", type=float, default=DEFAULT_CONV_NORM_THR)
    p.add_argument("--seed", type=int, default=0)
    return p


def _probe_cmd(args: argparse.Namespace, spec: AblationSpec, run_dir: Path) -> List[str]:
    return [
        str(args.python),
        str(args.probe_script),
        "--device",
        str(args.device),
        "--lr",
        str(args.lr),
        "--lr-schedule",
        str(args.lr_schedule),
        "--lr-step-size",
        str(args.lr_step_size),
        "--lr-step-gamma",
        str(args.lr_step_gamma),
        "--lr-floor",
        str(args.lr_floor),
        "--grad-clip",
        str(args.grad_clip),
        "--train-steps",
        str(args.train_steps),
        "--eval-every",
        str(args.eval_every),
        "--min-upgrade-train-steps",
        str(args.min_upgrade_train_steps),
        "--plateau-window",
        str(args.plateau_window),
        "--eval-plateau-window",
        str(args.eval_plateau_window),
        "--eval-plateau-min-samples",
        str(args.eval_plateau_min_samples),
        "--plateau-rel-delta",
        str(args.plateau_rel_delta),
        "--conv-norm-thr",
        str(args.conv_norm_thr),
        "--goal-head-hidden",
        str(spec.hidden),
        "--goal-head-depth",
        str(spec.depth),
        "--goal-head-mode",
        str(spec.mode),
        "--goal-injection-targets",
        str(spec.targets),
        "--seed",
        str(args.seed),
        "--out-dir",
        str(run_dir),
    ]


def _write_md(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# PHASE 1 Goal-Head Capacity / Injection Ablation")
    lines.append("")
    lines.append(f"- decision: **{payload['phase1_decision']}**")
    lines.append(f"- conv_norm_thr: {_fmt(payload['conv_norm_thr'], 2)}")
    lines.append(f"- runs_completed: {payload['runs_completed']}")
    lines.append("")
    lines.append("| run | head | injection | dual plateau | eval-min plateau | any reach | L_R rate mean/min | per-clip min |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for run in payload["runs"]:
        pc = run["per_clip"]
        mins = ", ".join(f"{c}:{_fmt(pc[c]['reach_min_norm_min'],2)}" for c in TURN_CLIPS)
        lr = run["l_r"]
        head = run["head"]
        lines.append(
            f"| {run['name']} | h={head['hidden']} d={head['depth']} {head['mode']} | "
            f"{run['injection_targets']} | {run['plateau']} | {run['eval_min_norm_plateau']} | "
            f"{run['any_clip_reached']} | "
            f"{_fmt(lr.get('reach_rate'),2)} {_fmt(lr.get('reach_min_norm_mean'),2)}/"
            f"{_fmt(lr.get('reach_min_norm_min'),2)} | {mins} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    for run in payload["runs"]:
        lines.append(f"- {run['artifact_dir']}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"debug_output/_tmp_action_handoff_inbetween_phase1_head_ablation_{date_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    winner: Optional[Dict[str, Any]] = None
    for spec in default_grid()[: max(1, min(int(args.max_runs), 6))]:
        run_dir = out_dir / spec.name
        cmd = _probe_cmd(args, spec, run_dir)
        log_path = run_dir / "run.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=_REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if int(proc.returncode) != 0:
            raise RuntimeError(f"ablation run failed ({spec.name}), see {log_path}")
        row = summarize_run(spec, _load_summary(run_dir), conv_norm_thr=float(args.conv_norm_thr))
        row["artifact_dir"] = str(run_dir.resolve())
        row["log_path"] = str(log_path.resolve())
        runs.append(row)
        if row["eligible_success"]:
            winner = row
            if bool(args.stop_on_success):
                break

    if winner is not None:
        decision = "frozen_base_usable_head_or_injection_was_too_weak__do_not_finetune"
        gate_reason = (
            f"{winner['name']} eval-min_norm plateaued and reached at least one clip; PHASE 2 is gated off."
        )
    else:
        all_eval_min_norm_plateau = bool(runs) and all(bool(r["eval_min_norm_plateau"]) for r in runs)
        if all_eval_min_norm_plateau:
            decision = "frozen_base_ceiling_confirmed__phase2_allowed"
            gate_reason = (
                "all completed PHASE 1 configs eval-min_norm plateaued and no clip crossed "
                "min_norm <= conv_norm_thr; train hidden loss is diagnostic only."
            )
        else:
            decision = "phase1_inconclusive__do_not_finetune"
            gate_reason = (
                "at least one completed PHASE 1 config did not eval-min_norm plateau; cannot claim "
                "frozen-base ceiling from this run alone."
            )

    payload: Dict[str, Any] = {
        "task": "Action-handoff in-betweening §7.3 PHASE 1 goal-head capacity/injection ablation",
        "phase": "PHASE 1",
        "conv_norm_thr": float(args.conv_norm_thr),
        "grid_size_limit": 6,
        "runs_completed": len(runs),
        "runs": runs,
        "winner": winner,
        "phase1_decision": decision,
        "gate_reason": gate_reason,
        "phase2_allowed": bool(decision == "frozen_base_ceiling_confirmed__phase2_allowed"),
    }
    json_path = out_dir / "phase1_head_ablation_summary.json"
    md_path = out_dir / "phase1_head_ablation_summary.md"
    _dump_json(json_path, payload)
    _write_md(md_path, payload)
    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(f"[PHASE1] {decision}: {gate_reason}")


if __name__ == "__main__":
    main()
