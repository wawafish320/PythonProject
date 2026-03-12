#!/usr/bin/env python3
"""
Batch runner for loss-budget stages (R0.5 / G0 / R2).

It launches `train.posttrain` resume runs across seeds with runtime payload
patching. Stage67 legacy backend wiring has been removed; this script is
posttrain-only.

It sweeps:
  - R0.5: asymmetric under-correct trigger loss sweep (w_under list)
  - G0:    hard->soft trigger gate attribution precheck
  - R2:    branch-budget normalization (trigger/chain/guard)

Outputs:
  - debug_output/<out_dir>/summary.json
  - debug_output/<out_dir>/summary.md
  - debug_output/<out_dir>/<case>/seed*/train.log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_ROOT = Path(__file__).resolve().parents[1]
_POSTTRAIN_BACKEND = "posttrain"


def _resolve_from_root(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (_ROOT / p)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for tok in str(spec or "").split(","):
        s = tok.strip()
        if not s:
            continue
        v = int(s)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    if not out:
        raise ValueError("empty integer list")
    return out


def _parse_float_list(spec: str) -> List[float]:
    out: List[float] = []
    seen = set()
    for tok in str(spec or "").split(","):
        s = tok.strip()
        if not s:
            continue
        v = float(s)
        if not math.isfinite(v):
            continue
        key = round(v, 10)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(v))
    if not out:
        raise ValueError("empty float list")
    return out


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _fmt_float_slug(v: float) -> str:
    x = float(v)
    s = f"{x:.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def _parse_override(text: str) -> Tuple[str, str]:
    s = str(text or "").strip()
    if "=" not in s:
        raise ValueError(f"invalid override (expect key=value): {text!r}")
    k, v = s.split("=", 1)
    key = str(k).strip()
    val = str(v).strip()
    if not key:
        raise ValueError(f"invalid override key: {text!r}")
    return key, val


def _coerce_scalar(text: str) -> Any:
    s = str(text).strip()
    low = s.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        iv = int(s)
        if str(iv) == s:
            return iv
    except Exception:
        pass
    try:
        fv = float(s)
        if math.isfinite(fv):
            return fv
    except Exception:
        pass
    return s


def _apply_overrides(base: Mapping[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    out = dict(base)
    for ov in overrides:
        key, raw_val = _parse_override(ov)
        out[key] = _coerce_scalar(raw_val)
    return out


def _run_and_tee(cmd: List[str], *, cwd: Path, env: Dict[str, str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd]\n")
        f.write(" ".join(cmd) + "\n\n")
        if dry_run:
            f.write("[dry-run] command not executed.\n")
            return 0
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
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


def _write_posttrain_runtime_config(
    *,
    config_path: Path,
    base_payload: Mapping[str, Any],
    overrides: Sequence[str],
    dataset_index_mode: str,
    epochs: int,
    out_model_dir: Path,
    run_name: str,
    resume_ckpt: Optional[Path],
) -> None:
    payload = _apply_overrides(base_payload, overrides)
    payload["dataset_index_mode"] = str(dataset_index_mode)
    payload["epochs"] = int(epochs)
    payload["out_dir"] = str(out_model_dir)
    payload["run_name"] = str(run_name)
    if resume_ckpt is not None:
        payload["ckpt_in"] = str(resume_ckpt)
    _write_json(config_path, payload)


def _build_posttrain_cmd(
    *,
    python: str,
    runtime_config_json: Path,
    out_model_dir: Path,
    run_name: str,
    seed: int,
    epochs: int,
    dataset_index_mode: str,
    resume_ckpt: Optional[Path],
) -> List[str]:
    cmd = [
        python,
        "-m",
        "train.posttrain",
        "--config",
        str(runtime_config_json),
        "--out_dir",
        str(out_model_dir),
        "--run_name",
        str(run_name),
        "--seed",
        str(int(seed)),
        "--dataset_index_mode",
        str(dataset_index_mode),
        "--epochs",
        str(int(epochs)),
    ]
    if resume_ckpt is not None:
        cmd += ["--ckpt_in", str(resume_ckpt)]
    return cmd


def _pick_ckpt(run_dir: Path, run_name: str) -> Optional[Path]:
    cands = [run_dir / f"ckpt_last_{run_name}.pth"]
    for p in cands:
        if p.is_file():
            return p
    return None


def _collect_teacher_metrics_posttrain(out_dir: Path, run_name: str) -> Dict[str, Any]:
    log_path = out_dir / f"posttrain_log_{run_name}.json"
    if not log_path.is_file():
        return {}
    try:
        obj = _load_json(log_path)
    except Exception:
        return {"teacher_metrics_file": str(log_path), "teacher_metrics_source": "posttrain_log"}

    log_rows = obj.get("log", []) if isinstance(obj, dict) else []
    if not isinstance(log_rows, list) or not log_rows:
        return {"teacher_metrics_file": str(log_path), "teacher_metrics_source": "posttrain_log"}

    last: Dict[str, Any] = {}
    for ent in reversed(log_rows):
        if isinstance(ent, dict):
            last = ent
            break
    if not last:
        return {"teacher_metrics_file": str(log_path), "teacher_metrics_source": "posttrain_log"}

    keys = [
        "DirectGeoLocalDeg",
        "direct_pose_geo_deg",
        "direct_pose_trigger_total_weighted",
        "direct_pose_trigger_n",
        "direct_pose_trigger_frac",
        "direct_pose_trigger_gate_weight_mean",
        "under_correct_frac_trigger_twist",
        "under_correct_frac_trigger_twist_hard",
        "direct_pose_budget_mode_r2",
        "direct_pose_trigger_budget_applied",
        "direct_pose_budget_total_weighted",
        "direct_pose_budget_trigger_component_weighted",
        "direct_pose_budget_chain_component_weighted",
        "direct_pose_budget_guard_component_weighted",
        "direct_pose_budget_share_trigger",
        "direct_pose_budget_share_chain",
        "direct_pose_budget_share_guard",
        "direct_pose_budget_trigger_norm",
        "direct_pose_budget_chain_norm",
        "direct_pose_budget_guard_norm",
        "direct_pose_budget_trigger_ema",
        "direct_pose_budget_chain_ema",
        "direct_pose_budget_guard_ema",
        "direct_pose_budget_lambda_mode_off",
        "direct_pose_budget_lambda_mode_ema_softmax",
        "direct_pose_budget_lambda_temperature",
        "direct_pose_budget_lambda_floor",
        "direct_pose_budget_lambda_active_count",
        "direct_pose_budget_lambda_total_base",
        "direct_pose_budget_lambda_total_eff",
        "direct_pose_budget_lambda_trigger_eff",
        "direct_pose_budget_lambda_chain_eff",
        "direct_pose_budget_lambda_guard_eff",
        "direct_pose_budget_lambda_trigger_ratio",
        "direct_pose_budget_lambda_chain_ratio",
        "direct_pose_budget_lambda_guard_ratio",
        "direct_pose_budget_chain_joint_count",
        "direct_pose_budget_guard_joint_count",
        "direct_pose_budget_pcgrad_active",
        "direct_pose_budget_pcgrad_conflict_frac_trigger_guard",
        "direct_pose_budget_pcgrad_guard_drop_ratio",
        "direct_pose_budget_pcgrad_trigger_drop_ratio",
        "direct_pose_budget_pcgrad_trigger_grad_norm_pre",
        "direct_pose_budget_pcgrad_guard_grad_norm_pre",
        "direct_pose_budget_pcgrad_trigger_grad_norm_post",
        "direct_pose_budget_pcgrad_guard_grad_norm_post",
        "direct_pose_budget_pcgrad_guard_freeze_active",
        "direct_pose_trigger_budget_stopgrad",
    ]
    picked = {k: last.get(k) for k in keys if k in last}

    if "direct_pose_geo_deg" not in picked:
        alias = last.get("dir_geo", None)
        if alias is not None:
            picked["direct_pose_geo_deg"] = alias
    if "DirectGeoLocalDeg" not in picked and "direct_pose_geo_deg" in picked:
        picked["DirectGeoLocalDeg"] = picked["direct_pose_geo_deg"]

    picked["teacher_metrics_file"] = str(log_path)
    picked["teacher_metrics_source"] = "posttrain_log"
    if "epoch" in last:
        picked["teacher_metrics_epoch"] = last.get("epoch")
    if "step" in last:
        picked["teacher_metrics_step"] = last.get("step")
    return picked


def _collect_teacher_metrics(run_dir: Path, run_name: str) -> Dict[str, Any]:
    return _collect_teacher_metrics_posttrain(run_dir, run_name)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    stage: str
    overrides: Tuple[str, ...]


def _parse_cases(spec: str) -> List[str]:
    raw = [x.strip().lower() for x in str(spec or "").split(",") if x.strip()]
    if not raw:
        return ["r05", "g0"]
    out: List[str] = []
    for x in raw:
        if x == "all":
            out.extend(["r05", "g0", "r2"])
        elif x in ("r05", "g0", "r2"):
            out.append(x)
        else:
            raise ValueError(f"unsupported case token: {x!r}")
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch run R0.5/G0/R2 loss-budget stages across seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config-json", type=str, default="config/exp_phase_DirectBranch_v1_d1_noreset.json")
    ap.add_argument(
        "--resume-ckpt",
        type=str,
        default="models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage6_direct_cond_anchor_20260124.pth",
        help="Checkpoint used for resume training (set empty to train from random init).",
    )
    ap.add_argument("--out-model-dir", type=str, default="", help="Model output root. Empty -> auto timestamped path.")
    ap.add_argument("--out-dir", type=str, default="", help="Debug/log output root. Empty -> auto timestamped path.")
    ap.add_argument("--base-run-name", type=str, default="loss_budget_r05g0")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--dataset-index-mode", type=str, default="sic_balanced")
    ap.add_argument("--cases", type=str, default="r05,g0", help="Comma list from {r05,g0,r2,all}.")
    ap.add_argument("--train-config-override", action="append", default=[], help="Global payload override key=value.")

    ap.add_argument("--r05-under-weights", type=str, default="1.5,2.0,3.0")
    ap.add_argument("--r05-under-mode", type=str, default="twist_only", choices=("off", "twist_only", "twist_swing"))
    ap.add_argument("--r05-extra-override", action="append", default=[], help="Extra payload override(s) appended to each R0.5 run.")

    ap.add_argument("--g0-tau-phase", type=float, default=0.05)
    ap.add_argument("--g0-tau-contact", type=float, default=0.05)
    ap.add_argument("--g0-tau-twist-deg", type=float, default=5.0)
    ap.add_argument("--g0-extra-override", action="append", default=[], help="Extra payload override(s) appended to each G0 run.")

    ap.add_argument("--r2-under-weight", type=float, default=2.0, help="R2 baseline under-correct weight (inherits R0.5 best).")
    ap.add_argument("--r2-under-mode", type=str, default="twist_only", choices=("off", "twist_only", "twist_swing"))
    ap.add_argument("--r2-budget-lambda-trigger", type=float, default=1.0)
    ap.add_argument("--r2-budget-lambda-chain", type=float, default=0.45)
    ap.add_argument("--r2-budget-lambda-guard", type=float, default=0.45)
    ap.add_argument(
        "--r2-budget-lambda-mode",
        type=str,
        default="off",
        choices=("off", "ema_softmax"),
        help="Budget lambda mode in R2: off|ema_softmax.",
    )
    ap.add_argument(
        "--r2-budget-lambda-temperature",
        type=float,
        default=1.0,
        help="Temperature for EMA-driven lambda mode (R2).",
    )
    ap.add_argument(
        "--r2-budget-lambda-floor",
        type=float,
        default=0.15,
        help="Per-branch floor for EMA-driven lambda mode (R2).",
    )
    ap.add_argument("--r2-budget-ema-beta", type=float, default=0.95)
    ap.add_argument("--r2-budget-eps", type=float, default=1e-4)
    ap.add_argument("--r2-budget-chain-joints", type=str, default="calf_r,ball_r")
    ap.add_argument("--r2-budget-chain-frame-mode", type=str, default="trigger", choices=("trigger", "all"))
    ap.add_argument("--r2-budget-guard-frame-mode", type=str, default="non_trigger", choices=("non_trigger", "all"))
    ap.add_argument("--r2-budget-guard-exclude-joints", type=str, default="")
    ap.add_argument("--r2-extra-override", action="append", default=[], help="Extra payload override(s) appended to each R2 run.")

    ap.add_argument("--skip-existing", action="store_true", help="Skip run when ckpt_last exists.")
    ap.add_argument("--dry-run", action="store_true", help="Write commands only; do not execute.")
    args = ap.parse_args()
    backend = _POSTTRAIN_BACKEND

    cfg_path = _resolve_from_root(args.config_json)
    if not cfg_path.is_file():
        raise SystemExit(f"[FATAL] config not found: {cfg_path}")
    cfg_obj = _load_json(cfg_path)
    if not isinstance(cfg_obj, dict):
        raise SystemExit(f"[FATAL] config json must be an object: {cfg_path}")

    resume_ckpt: Optional[Path] = None
    resume_raw = str(args.resume_ckpt or "").strip()
    if resume_raw:
        resume_ckpt = _resolve_from_root(resume_raw)
        if not resume_ckpt.is_file():
            raise SystemExit(f"[FATAL] resume ckpt not found: {resume_ckpt}")

    seeds = _parse_int_list(args.seeds)
    if int(args.epochs) < 1:
        raise SystemExit("[FATAL] --epochs must be >=1")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_model_dir = (
        _resolve_from_root(args.out_model_dir)
        if str(args.out_model_dir or "").strip()
        else (_ROOT / "models" / f"MLPL2_DirectBranch_v1__pipe_{ts}_loss_budget_r05g0")
    )
    out_dir = (
        _resolve_from_root(args.out_dir)
        if str(args.out_dir or "").strip()
        else (_ROOT / "debug_output" / f"_posttrain_loss_budget_redesign_v1_{ts}" / "R0_5_G0_batch")
    )
    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = _parse_cases(args.cases)
    common_overrides = [str(x).strip() for x in (args.train_config_override or []) if str(x).strip()]
    r05_extra = [str(x).strip() for x in (args.r05_extra_override or []) if str(x).strip()]
    g0_extra = [str(x).strip() for x in (args.g0_extra_override or []) if str(x).strip()]
    r2_extra = [str(x).strip() for x in (args.r2_extra_override or []) if str(x).strip()]

    cases: List[CaseSpec] = []
    if "r05" in selected_cases:
        weights = _parse_float_list(args.r05_under_weights)
        for w in weights:
            name = f"r05_u{_fmt_float_slug(w)}"
            ov = [
                "direct_pose_trigger_gate_mode=hard",
                "w_direct_pose_trigger_total=1.0",
                f"direct_pose_trigger_under_mode={args.r05_under_mode}",
                f"direct_pose_trigger_under_weight={float(w):.8g}",
            ]
            ov.extend(r05_extra)
            cases.append(CaseSpec(name=name, stage="R0.5", overrides=tuple(ov)))
    if "g0" in selected_cases:
        ov = [
            "direct_pose_trigger_gate_mode=soft",
            "w_direct_pose_trigger_total=1.0",
            "direct_pose_trigger_under_mode=off",
            "direct_pose_trigger_under_weight=1.0",
            f"direct_pose_trigger_tau_phase={float(args.g0_tau_phase):.8g}",
            f"direct_pose_trigger_tau_contact={float(args.g0_tau_contact):.8g}",
            f"direct_pose_trigger_tau_twist_deg={float(args.g0_tau_twist_deg):.8g}",
        ]
        ov.extend(g0_extra)
        cases.append(CaseSpec(name="g0_soft", stage="G0", overrides=tuple(ov)))
    if "r2" in selected_cases:
        ov = [
            "direct_pose_trigger_gate_mode=hard",
            "direct_pose_trigger_sign_source=gt",
            "w_direct_pose_trigger_total=1.0",
            f"direct_pose_trigger_under_mode={args.r2_under_mode}",
            f"direct_pose_trigger_under_weight={float(args.r2_under_weight):.8g}",
            "direct_pose_budget_mode=r2",
            f"direct_pose_budget_lambda_trigger={float(args.r2_budget_lambda_trigger):.8g}",
            f"direct_pose_budget_lambda_chain={float(args.r2_budget_lambda_chain):.8g}",
            f"direct_pose_budget_lambda_guard={float(args.r2_budget_lambda_guard):.8g}",
            f"direct_pose_budget_lambda_mode={str(args.r2_budget_lambda_mode)}",
            f"direct_pose_budget_lambda_temperature={float(args.r2_budget_lambda_temperature):.8g}",
            f"direct_pose_budget_lambda_floor={float(args.r2_budget_lambda_floor):.8g}",
            f"direct_pose_budget_ema_beta={float(args.r2_budget_ema_beta):.8g}",
            f"direct_pose_budget_eps={float(args.r2_budget_eps):.8g}",
            f"direct_pose_budget_chain_joints={str(args.r2_budget_chain_joints)}",
            f"direct_pose_budget_chain_frame_mode={str(args.r2_budget_chain_frame_mode)}",
            f"direct_pose_budget_guard_frame_mode={str(args.r2_budget_guard_frame_mode)}",
            f"direct_pose_budget_guard_exclude_joints={str(args.r2_budget_guard_exclude_joints)}",
        ]
        ov.extend(r2_extra)
        cases.append(CaseSpec(name="r2_budget", stage="R2", overrides=tuple(ov)))

    if not cases:
        raise SystemExit("[FATAL] no cases selected.")

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(_ROOT))
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []

    for case in cases:
        case_dir = out_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)

        effective_case_cfg = _apply_overrides(cfg_obj, list(common_overrides) + list(case.overrides))
        w_tw = _safe_float(effective_case_cfg.get("w_direct_pose_trigger_twist", 0.0))
        w_sx = _safe_float(effective_case_cfg.get("w_direct_pose_trigger_swing_x", 0.0))
        w_sy = _safe_float(effective_case_cfg.get("w_direct_pose_trigger_swing_y", 0.0))
        if (w_tw + w_sx + w_sy) <= 0.0:
            warnings.append(
                f"[WARN][{case.name}] trigger axis weights sum <= 0 "
                f"(twist={w_tw}, swing_x={w_sx}, swing_y={w_sy}); run may become a no-op."
            )

        for seed in seeds:
            run_name = f"{args.base_run_name}_{case.name}_seed{int(seed)}_e{int(args.epochs)}"
            run_dir = out_model_dir
            log_path = case_dir / f"seed{int(seed)}" / "train.log"
            run_overrides = list(common_overrides) + list(case.overrides)
            runtime_cfg_path = case_dir / f"seed{int(seed)}" / f"{run_name}__posttrain_runtime.json"
            _write_posttrain_runtime_config(
                config_path=runtime_cfg_path,
                base_payload=cfg_obj,
                overrides=run_overrides,
                dataset_index_mode=str(args.dataset_index_mode),
                epochs=int(args.epochs),
                out_model_dir=out_model_dir,
                run_name=run_name,
                resume_ckpt=resume_ckpt,
            )
            cmd = _build_posttrain_cmd(
                python=sys.executable,
                runtime_config_json=runtime_cfg_path,
                out_model_dir=out_model_dir,
                run_name=run_name,
                seed=int(seed),
                epochs=int(args.epochs),
                dataset_index_mode=str(args.dataset_index_mode),
                resume_ckpt=resume_ckpt,
            )

            rec: Dict[str, Any] = {
                "case": case.name,
                "stage": case.stage,
                "seed": int(seed),
                "run_name": run_name,
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "status": "pending",
                "resume_ckpt": str(resume_ckpt) if resume_ckpt is not None else "",
                "config_overrides": run_overrides,
                "backend": backend,
                "runtime_config_json": str(runtime_cfg_path),
                "cmd": cmd,
            }

            ckpt_last = out_model_dir / f"ckpt_last_{run_name}.pth"
            if bool(args.skip_existing) and ckpt_last.is_file():
                rec["status"] = "skipped_existing"
            else:
                rc = _run_and_tee(cmd, cwd=_ROOT, env=env, log_path=log_path, dry_run=bool(args.dry_run))
                rec["status"] = "dry_run" if bool(args.dry_run) else ("ok" if rc == 0 else f"failed_exit_{rc}")
                rec["exit_code"] = int(rc)

            ckpt = _pick_ckpt(run_dir, run_name)
            rec["ckpt"] = str(ckpt) if ckpt is not None else ""
            rec["teacher_metrics"] = _collect_teacher_metrics(run_dir, run_name)
            rows.append(rec)

    status_count: Dict[str, int] = {}
    for r in rows:
        st = str(r.get("status", "unknown"))
        status_count[st] = int(status_count.get(st, 0)) + 1

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_json": str(cfg_path),
        "resume_ckpt": str(resume_ckpt) if resume_ckpt is not None else "",
        "out_model_dir": str(out_model_dir),
        "out_dir": str(out_dir),
        "seeds": seeds,
        "epochs": int(args.epochs),
        "dataset_index_mode": str(args.dataset_index_mode),
        "backend": backend,
        "cases": [{"name": c.name, "stage": c.stage, "overrides": list(c.overrides)} for c in cases],
        "common_overrides": common_overrides,
        "warnings": warnings,
        "status_count": status_count,
        "rows": rows,
    }
    summary_json = out_dir / "summary.json"
    _write_json(summary_json, summary)

    lines: List[str] = []
    lines.append("# Loss-Budget R0.5 / G0 / R2 Batch Summary")
    lines.append("")
    lines.append(f"- generated_at: `{summary['generated_at']}`")
    lines.append(f"- config_json: `{cfg_path}`")
    lines.append(f"- resume_ckpt: `{resume_ckpt if resume_ckpt is not None else 'none (random init)'}`")
    lines.append(f"- seeds: `{','.join(str(s) for s in seeds)}`")
    lines.append(f"- epochs: `{int(args.epochs)}`")
    lines.append(f"- dataset_index_mode: `{args.dataset_index_mode}`")
    lines.append(f"- backend: `{backend}`")
    lines.append(f"- out_model_dir: `{out_model_dir}`")
    lines.append(f"- out_dir: `{out_dir}`")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    for k in sorted(status_count.keys()):
        lines.append(f"- {k}: `{status_count[k]}`")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("|case|stage|seed|status|ckpt|log|")
    lines.append("|:--|:--|--:|:--|:--|:--|")
    for r in rows:
        lines.append(
            f"|{r.get('case','')}|{r.get('stage','')}|{int(r.get('seed',-1))}|{r.get('status','')}|"
            f"`{r.get('ckpt','')}`|`{r.get('log_path','')}`|"
        )
    lines.append("")
    lines.append(f"- summary_json: `{summary_json}`")
    summary_md = out_dir / "summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote summary: {summary_json}")
    print(f"[OK] wrote summary: {summary_md}")
    for w in warnings:
        print(w)


if __name__ == "__main__":
    main()
