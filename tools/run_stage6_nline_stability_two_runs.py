#!/usr/bin/env python3
"""Run two Stage6->N-line stability probes:

Run1: freeze direct_pose_head first (hinge-only), then unfreeze.
Run2: reinitialize direct_pose_head and train from scratch.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


_ROOT = Path(__file__).resolve().parents[1]
_RUN_STAGE67 = _ROOT / "tools" / "run_stage67_transition.py"


@dataclass
class GateMetrics:
    status: str
    gate_json: str
    delta_mean_deg: float
    delta_p99_deg: float
    delta_max_deg: float
    branch_changed: Optional[bool]
    error: str


def _resolve(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (_ROOT / p)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _fmt(v: Any) -> str:
    x = _safe_float(v)
    return f"{x:+.4f}" if math.isfinite(x) else "nan"


def _run_and_tee(cmd: Sequence[str], *, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[cmd] " + " ".join(str(x) for x in cmd))
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd]\n")
        f.write(" ".join(str(x) for x in cmd) + "\n\n")
        if dry_run:
            f.write("[dry-run] command not executed.\n")
            return 0
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = int(proc.wait())
        f.write(f"\n[exit_code] {rc}\n")
        return rc


def _build_full_nline_payload(stage6_cfg: Mapping[str, Any], nline_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(stage6_cfg)
    for k in sorted(set(nline_cfg.keys()) - set(stage6_cfg.keys())):
        out[k] = nline_cfg[k]
    # Keep prior bisect/min-confirm contract so results are comparable.
    out.update(
        {
            "train_direct_pose": True,
            "train_so3_corrector": False,
            "w_direct_pose_trigger_total": 0.0,
            "w_direct_pose_trigger_twist": 0.0,
            "w_direct_pose_trigger_swing_x": 0.0,
            "w_direct_pose_trigger_swing_y": 0.0,
            "direct_pose_trigger_under_mode": "off",
            "direct_pose_trigger_under_weight": 1.0,
            "direct_pose_budget_mode": "off",
        }
    )
    return out


def _run_train(
    *,
    payload: Mapping[str, Any],
    cfg_path: Path,
    model_dir: Path,
    run_name: str,
    seed: int,
    dataset_index_mode: str,
    epochs: int,
    ckpt_in: Path,
    log_path: Path,
    dry_run: bool,
    skip_existing: bool,
) -> Dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cfg_path, payload)
    ckpt_last = model_dir / f"ckpt_last_{run_name}.pth"

    if skip_existing and ckpt_last.is_file():
        return {
            "status": "skipped_existing",
            "cfg_json": str(cfg_path),
            "ckpt": str(ckpt_last),
        }

    cmd = [
        str(sys.executable),
        "-u",
        "-m",
        "train.posttrain",
        "--config",
        str(cfg_path),
        "--out_dir",
        str(model_dir),
        "--run_name",
        str(run_name),
        "--seed",
        str(int(seed)),
        "--dataset_index_mode",
        str(dataset_index_mode),
        "--epochs",
        str(int(epochs)),
        "--ckpt_in",
        str(ckpt_in),
    ]
    rc = _run_and_tee(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        return {
            "status": f"train_cmd_exit_{rc}",
            "cfg_json": str(cfg_path),
            "ckpt": str(ckpt_last),
        }

    if dry_run:
        return {
            "status": "dry_run",
            "cfg_json": str(cfg_path),
            "ckpt": str(ckpt_last),
        }

    if not ckpt_last.is_file():
        return {
            "status": "missing_ckpt",
            "cfg_json": str(cfg_path),
            "ckpt": str(ckpt_last),
        }

    return {
        "status": "ok",
        "cfg_json": str(cfg_path),
        "ckpt": str(ckpt_last),
    }


def _run_freerun(
    *,
    arm_a_ckpt: Path,
    arm_b_ckpt: Path,
    seed: int,
    out_root: Path,
    direct_mode: str,
    log_path: Path,
    dry_run: bool,
) -> GateMetrics:
    cmd = [
        str(sys.executable),
        str(_RUN_STAGE67),
        "freerun-ab",
        "--arm-a-ckpt",
        str(arm_a_ckpt),
        "--arm-b-ckpt",
        str(arm_b_ckpt),
        "--seed",
        str(int(seed)),
        "--out-root",
        str(out_root),
        "--cycle-gte",
        "1",
        "--drop-wrap",
        "1",
        "--c2-policy",
        "ignore",
    ]
    dm = str(direct_mode).strip().lower()
    if dm:
        cmd += ["--direct-pose-fusion-direct-mode", dm]
    rc = _run_and_tee(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        return GateMetrics(
            status=f"freerun_cmd_exit_{rc}",
            gate_json="",
            delta_mean_deg=float("nan"),
            delta_p99_deg=float("nan"),
            delta_max_deg=float("nan"),
            branch_changed=None,
            error=f"freerun command exit {rc}",
        )

    gate_json = out_root / "freerun_ab_gate.json"
    if dry_run:
        return GateMetrics(
            status="dry_run",
            gate_json=str(gate_json),
            delta_mean_deg=float("nan"),
            delta_p99_deg=float("nan"),
            delta_max_deg=float("nan"),
            branch_changed=None,
            error="",
        )
    if not gate_json.is_file():
        return GateMetrics(
            status="missing_gate_json",
            gate_json=str(gate_json),
            delta_mean_deg=float("nan"),
            delta_p99_deg=float("nan"),
            delta_max_deg=float("nan"),
            branch_changed=None,
            error="missing freerun_ab_gate.json",
        )

    gate = _load_json(gate_json)
    delta = (((gate.get("freerun_global", {}) or {}).get("delta_a_minus_b", {}) or {}))
    branch = (gate.get("trigger_branch", {}) or {})
    return GateMetrics(
        status="ok",
        gate_json=str(gate_json),
        delta_mean_deg=_safe_float(delta.get("mean_deg", float("nan"))),
        delta_p99_deg=_safe_float(delta.get("p99_deg", float("nan"))),
        delta_max_deg=_safe_float(delta.get("max_deg", float("nan"))),
        branch_changed=bool(branch.get("branch_changed", False)),
        error="",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Stage6+N-line full two-path stability probes (freeze/unfreeze vs reinit).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--run-tag",
        type=str,
        default=f"stage6_nline_stability_two_runs_{datetime.now().strftime('%Y%m%d')}",
    )
    ap.add_argument(
        "--stage6-config",
        type=str,
        default="config/posttrain_WalkF_stage6_direct_cond_anchor_20260124.json",
    )
    ap.add_argument("--nline-config", type=str, default="config/exp_phase_DirectBranch_v1_d1_noreset.json")
    ap.add_argument(
        "--stage6-ckpt",
        type=str,
        default="models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage6_direct_cond_anchor_20260124.pth",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--dataset-index-mode", type=str, default="start0")
    ap.add_argument("--steps-total", type=int, default=60, help="Total steps for each run.")
    ap.add_argument("--run1-warm-steps", type=int, default=30, help="Warm stage steps before unfreezing direct head.")
    ap.add_argument("--run1-opt-warmup-steps", type=int, default=0, help="Phase2 optimizer LR warmup steps.")
    ap.add_argument("--run1-opt-warmup-start-lr", type=float, default=0.0, help="Phase2 optimizer warmup start LR.")
    ap.add_argument(
        "--direct-mode",
        type=str,
        default="absolute",
        choices=("", "absolute", "residual_rot6d", "residual_compose_stable"),
    )
    ap.add_argument("--out-root", type=str, default="")
    ap.add_argument("--model-prefix", type=str, default="")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not _RUN_STAGE67.is_file():
        raise SystemExit(f"[FATAL] missing helper: {_RUN_STAGE67}")

    stage6_cfg_path = _resolve(args.stage6_config)
    nline_cfg_path = _resolve(args.nline_config)
    stage6_ckpt = _resolve(args.stage6_ckpt)
    if not stage6_cfg_path.is_file():
        raise SystemExit(f"[FATAL] missing stage6 config: {stage6_cfg_path}")
    if not nline_cfg_path.is_file():
        raise SystemExit(f"[FATAL] missing nline config: {nline_cfg_path}")
    if (not args.dry_run) and (not stage6_ckpt.is_file()):
        raise SystemExit(f"[FATAL] missing stage6 ckpt: {stage6_ckpt}")

    stage6_cfg = _load_json(stage6_cfg_path)
    nline_cfg = _load_json(nline_cfg_path)
    if not isinstance(stage6_cfg, dict) or not isinstance(nline_cfg, dict):
        raise SystemExit("[FATAL] config json must be object")

    if int(args.steps_total) <= 1:
        raise SystemExit("[FATAL] --steps-total must be > 1")
    if int(args.run1_warm_steps) <= 0:
        raise SystemExit("[FATAL] --run1-warm-steps must be > 0")
    if int(args.run1_warm_steps) >= int(args.steps_total):
        raise SystemExit("[FATAL] --run1-warm-steps must be < --steps-total")

    out_root = _resolve(args.out_root) if str(args.out_root).strip() else (_ROOT / "debug_output" / f"_{args.run_tag}")
    model_prefix = _resolve(args.model_prefix) if str(args.model_prefix).strip() else (_ROOT / "models" / f"MLPL2_DirectBranch_v1__{args.run_tag}")
    out_root.mkdir(parents=True, exist_ok=True)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    nline_only_keys = sorted(set(nline_cfg.keys()) - set(stage6_cfg.keys()))
    full_payload = _build_full_nline_payload(stage6_cfg, nline_cfg)

    # Run1: freeze direct head (contact-meas warm stage) -> unfreeze direct head.
    run1_root = out_root / "run1_freeze_then_unfreeze"
    run1_model_dir = Path(f"{model_prefix}_run1")
    run1_root.mkdir(parents=True, exist_ok=True)
    warm_steps = int(args.run1_warm_steps)
    tune_steps = int(args.steps_total) - warm_steps
    warm_contact_weight = _safe_float(full_payload.get("contact_meas_weight", 0.0))
    if (not math.isfinite(warm_contact_weight)) or warm_contact_weight <= 0.0:
        warm_contact_weight = _safe_float(full_payload.get("w_contact_meas", 0.0))
    if (not math.isfinite(warm_contact_weight)) or warm_contact_weight <= 0.0:
        warm_contact_weight = 0.05

    run1_phase1_payload = dict(full_payload)
    run1_phase1_payload.update(
        {
            # Warm stage: keep direct head frozen, only adapt contact_meas pathway.
            "train_direct_pose": False,
            "train_contact_meas": True,
            "train_contact_td_hazard": False,
            "contact_meas_weight": float(warm_contact_weight),
            "direct_pose_reinit": False,
            "direct_pose_hinge_train_only": False,
            "direct_pose_hinge_gate_train_only": False,
            "direct_pose_alpha_train_only": False,
            "direct_pose_leg_train_only": False,
            "direct_pose_leg_gate_train_only": False,
            "steps_per_epoch": warm_steps,
        }
    )
    run1_phase1 = _run_train(
        payload=run1_phase1_payload,
        cfg_path=run1_root / "phase1_warm_runtime.json",
        model_dir=run1_model_dir,
        run_name=f"{args.run_tag}_run1_phase1_warm_seed{int(args.seed)}_e{int(args.epochs)}s{warm_steps}",
        seed=int(args.seed),
        dataset_index_mode=str(args.dataset_index_mode),
        epochs=int(args.epochs),
        ckpt_in=stage6_ckpt,
        log_path=run1_root / "phase1_train.log",
        dry_run=bool(args.dry_run),
        skip_existing=bool(args.skip_existing),
    )

    run1_phase2: Dict[str, Any]
    run1_gate: GateMetrics
    if str(run1_phase1.get("status")) in ("ok", "skipped_existing", "dry_run"):
        ckpt_after_phase1 = Path(str(run1_phase1.get("ckpt", "")))
        run1_phase2_payload = dict(full_payload)
        run1_phase2_payload.update(
            {
                "train_direct_pose": True,
                "train_contact_meas": False,
                "train_contact_td_hazard": False,
                "direct_pose_reinit": False,
                "direct_pose_hinge_train_only": False,
                "direct_pose_hinge_gate_train_only": False,
                "direct_pose_alpha_train_only": False,
                "direct_pose_leg_train_only": False,
                "direct_pose_leg_gate_train_only": False,
                "steps_per_epoch": tune_steps,
            }
        )
        if int(args.run1_opt_warmup_steps) > 0 and float(args.run1_opt_warmup_start_lr) > 0.0:
            run1_phase2_payload["opt_warmup_steps"] = int(args.run1_opt_warmup_steps)
            run1_phase2_payload["opt_warmup_start_lr"] = float(args.run1_opt_warmup_start_lr)

        run1_phase2 = _run_train(
            payload=run1_phase2_payload,
            cfg_path=run1_root / "phase2_unfreeze_runtime.json",
            model_dir=run1_model_dir,
            run_name=f"{args.run_tag}_run1_phase2_unfreeze_seed{int(args.seed)}_e{int(args.epochs)}s{tune_steps}",
            seed=int(args.seed),
            dataset_index_mode=str(args.dataset_index_mode),
            epochs=int(args.epochs),
            ckpt_in=ckpt_after_phase1,
            log_path=run1_root / "phase2_train.log",
            dry_run=bool(args.dry_run),
            skip_existing=bool(args.skip_existing),
        )

        if str(run1_phase2.get("status")) in ("ok", "skipped_existing", "dry_run"):
            run1_gate = _run_freerun(
                arm_a_ckpt=Path(str(run1_phase2.get("ckpt", ""))),
                arm_b_ckpt=stage6_ckpt,
                seed=int(args.seed),
                out_root=run1_root / f"freerun_seed{int(args.seed)}",
                direct_mode=str(args.direct_mode),
                log_path=run1_root / "freerun.log",
                dry_run=bool(args.dry_run),
            )
        else:
            run1_gate = GateMetrics(
                status="skipped_due_phase2_failure",
                gate_json="",
                delta_mean_deg=float("nan"),
                delta_p99_deg=float("nan"),
                delta_max_deg=float("nan"),
                branch_changed=None,
                error=f"phase2 status={run1_phase2.get('status')}",
            )
    else:
        run1_phase2 = {
            "status": "skipped_due_phase1_failure",
            "cfg_json": "",
            "ckpt": "",
        }
        run1_gate = GateMetrics(
            status="skipped_due_phase1_failure",
            gate_json="",
            delta_mean_deg=float("nan"),
            delta_p99_deg=float("nan"),
            delta_max_deg=float("nan"),
            branch_changed=None,
            error=f"phase1 status={run1_phase1.get('status')}",
        )

    # Run2: direct head reinit.
    run2_root = out_root / "run2_reinit_head"
    run2_model_dir = Path(f"{model_prefix}_run2")
    run2_root.mkdir(parents=True, exist_ok=True)
    run2_payload = dict(full_payload)
    run2_payload.update(
        {
            "direct_pose_reinit": True,
            "direct_pose_hinge_train_only": False,
            "direct_pose_hinge_gate_train_only": False,
            "direct_pose_alpha_train_only": False,
            "direct_pose_leg_train_only": False,
            "direct_pose_leg_gate_train_only": False,
            "steps_per_epoch": int(args.steps_total),
        }
    )
    run2_train = _run_train(
        payload=run2_payload,
        cfg_path=run2_root / "train_runtime.json",
        model_dir=run2_model_dir,
        run_name=f"{args.run_tag}_run2_reinit_seed{int(args.seed)}_e{int(args.epochs)}s{int(args.steps_total)}",
        seed=int(args.seed),
        dataset_index_mode=str(args.dataset_index_mode),
        epochs=int(args.epochs),
        ckpt_in=stage6_ckpt,
        log_path=run2_root / "train.log",
        dry_run=bool(args.dry_run),
        skip_existing=bool(args.skip_existing),
    )
    if str(run2_train.get("status")) in ("ok", "skipped_existing", "dry_run"):
        run2_gate = _run_freerun(
            arm_a_ckpt=Path(str(run2_train.get("ckpt", ""))),
            arm_b_ckpt=stage6_ckpt,
            seed=int(args.seed),
            out_root=run2_root / f"freerun_seed{int(args.seed)}",
            direct_mode=str(args.direct_mode),
            log_path=run2_root / "freerun.log",
            dry_run=bool(args.dry_run),
        )
    else:
        run2_gate = GateMetrics(
            status="skipped_due_train_failure",
            gate_json="",
            delta_mean_deg=float("nan"),
            delta_p99_deg=float("nan"),
            delta_max_deg=float("nan"),
            branch_changed=None,
            error=f"train status={run2_train.get('status')}",
        )

    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": str(args.run_tag),
        "stage6_config": str(stage6_cfg_path),
        "nline_config": str(nline_cfg_path),
        "stage6_ckpt": str(stage6_ckpt),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "dataset_index_mode": str(args.dataset_index_mode),
        "steps_total": int(args.steps_total),
        "run1_warm_steps": int(args.run1_warm_steps),
        "run1_tune_steps": int(tune_steps),
        "run1_opt_warmup_steps": int(args.run1_opt_warmup_steps),
        "run1_opt_warmup_start_lr": float(args.run1_opt_warmup_start_lr),
        "direct_mode": str(args.direct_mode),
        "nline_only_keys_count": int(len(nline_only_keys)),
        "nline_only_keys": nline_only_keys,
        "run1": {
            "phase1": run1_phase1,
            "phase2": run1_phase2,
            "gate": {
                "status": run1_gate.status,
                "gate_json": run1_gate.gate_json,
                "delta_mean_deg": run1_gate.delta_mean_deg,
                "delta_p99_deg": run1_gate.delta_p99_deg,
                "delta_max_deg": run1_gate.delta_max_deg,
                "branch_changed": run1_gate.branch_changed,
                "error": run1_gate.error,
            },
        },
        "run2": {
            "train": run2_train,
            "gate": {
                "status": run2_gate.status,
                "gate_json": run2_gate.gate_json,
                "delta_mean_deg": run2_gate.delta_mean_deg,
                "delta_p99_deg": run2_gate.delta_p99_deg,
                "delta_max_deg": run2_gate.delta_max_deg,
                "branch_changed": run2_gate.branch_changed,
                "error": run2_gate.error,
            },
        },
    }

    out_json = out_root / "two_runs_summary.json"
    _write_json(out_json, payload)

    lines: List[str] = []
    lines.append("# Stage6+N-line Stability Two-Run Summary")
    lines.append("")
    lines.append(f"- run_tag: `{args.run_tag}`")
    lines.append(f"- seed: `{int(args.seed)}`")
    lines.append(f"- epochs(per phase): `{int(args.epochs)}`")
    lines.append(f"- steps_total: `{int(args.steps_total)}`")
    lines.append(f"- run1_warm/tune steps: `{warm_steps}/{tune_steps}`")
    lines.append(f"- nline_only_keys: `{len(nline_only_keys)}`")
    lines.append("")
    lines.append("| run | status | Δmean/Δp99/Δmax (deg) | branch_changed |")
    lines.append("|---|---|---:|---:|")
    lines.append(
        f"| run1_freeze_then_unfreeze | {run1_gate.status} | "
        f"{_fmt(run1_gate.delta_mean_deg)} / {_fmt(run1_gate.delta_p99_deg)} / {_fmt(run1_gate.delta_max_deg)} | "
        f"{str(run1_gate.branch_changed).lower() if run1_gate.branch_changed is not None else 'n/a'} |"
    )
    lines.append(
        f"| run2_reinit_head | {run2_gate.status} | "
        f"{_fmt(run2_gate.delta_mean_deg)} / {_fmt(run2_gate.delta_p99_deg)} / {_fmt(run2_gate.delta_max_deg)} | "
        f"{str(run2_gate.branch_changed).lower() if run2_gate.branch_changed is not None else 'n/a'} |"
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- run1 phase1 cfg: `{run1_phase1.get('cfg_json', '')}`")
    lines.append(f"- run1 phase2 cfg: `{run1_phase2.get('cfg_json', '')}`")
    lines.append(f"- run1 gate: `{run1_gate.gate_json}`")
    lines.append(f"- run2 train cfg: `{run2_train.get('cfg_json', '')}`")
    lines.append(f"- run2 gate: `{run2_gate.gate_json}`")
    out_md = out_root / "two_runs_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
