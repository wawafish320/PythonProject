#!/usr/bin/env python3
"""
Stage67 next-iteration helper: EMA-driven dynamic-lambda sweep on cg4545_v2 baseline.

Doc reference:
  docs/changes/2026-02-20_stage67_c1_pcgrad_closeout_and_next_iteration.md
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
_RUN_LOSS = _ROOT / "tools" / "run_loss_budget_r05_g0.py"


def _resolve_from_root(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (_ROOT / p)


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _parse_float_list(csv: str) -> List[float]:
    vals: List[float] = []
    for tok in str(csv or "").split(","):
        t = tok.strip()
        if not t:
            continue
        x = _safe_float(t)
        if math.isfinite(x):
            vals.append(float(x))
    if not vals:
        raise SystemExit("[FATAL] empty float list")
    return vals


def _parse_int_list(csv: str) -> List[int]:
    vals: List[int] = []
    for tok in str(csv or "").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            vals.append(int(t))
        except Exception:
            raise SystemExit(f"[FATAL] invalid int token: {t!r}")
    if not vals:
        raise SystemExit("[FATAL] empty int list")
    return vals


def _fmt_slug(x: float) -> str:
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _run(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("[cmd] " + " ".join(str(x) for x in cmd))
    if dry_run:
        return
    rc = subprocess.call([str(x) for x in cmd], cwd=str(_ROOT))
    if int(rc) != 0:
        raise SystemExit(f"[FATAL] command failed (exit={rc})")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class SweepRow:
    temperature: float
    out_dir: Path
    model_dir: Path
    summary_json: Optional[Path]
    status_count: Dict[str, int]
    trajectories: List[Dict[str, Any]]


def _collect_trajectories(summary_path: Path) -> List[Dict[str, Any]]:
    if not summary_path.is_file():
        return []
    obj = _load_json(summary_path)
    rows = obj.get("rows", [])
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        run_name = str(row.get("run_name", "") or "")
        run_dir = Path(str(row.get("run_dir", "") or ""))
        if not run_name or not run_dir:
            continue
        log_path = run_dir / f"posttrain_log_{run_name}.json"
        if not log_path.is_file():
            continue
        try:
            log_obj = _load_json(log_path)
        except Exception:
            continue
        log_rows = log_obj.get("log", [])
        if not isinstance(log_rows, list):
            continue
        traj: List[Dict[str, Any]] = []
        for ent in log_rows:
            if not isinstance(ent, Mapping):
                continue
            ep = ent.get("epoch", None)
            try:
                epoch = int(ep)
            except Exception:
                continue
            traj.append(
                {
                    "epoch": epoch,
                    "direct_pose_budget_lambda_trigger_eff": _safe_float(ent.get("direct_pose_budget_lambda_trigger_eff")),
                    "direct_pose_budget_lambda_chain_eff": _safe_float(ent.get("direct_pose_budget_lambda_chain_eff")),
                    "direct_pose_budget_lambda_guard_eff": _safe_float(ent.get("direct_pose_budget_lambda_guard_eff")),
                    "direct_pose_budget_trigger_ema": _safe_float(ent.get("direct_pose_budget_trigger_ema")),
                    "direct_pose_budget_chain_ema": _safe_float(ent.get("direct_pose_budget_chain_ema")),
                    "direct_pose_budget_guard_ema": _safe_float(ent.get("direct_pose_budget_guard_ema")),
                    "direct_pose_budget_share_trigger": _safe_float(ent.get("direct_pose_budget_share_trigger")),
                    "direct_pose_budget_share_chain": _safe_float(ent.get("direct_pose_budget_share_chain")),
                    "direct_pose_budget_share_guard": _safe_float(ent.get("direct_pose_budget_share_guard")),
                }
            )
        out.append(
            {
                "seed": row.get("seed"),
                "run_name": run_name,
                "log_path": str(log_path),
                "trajectory": traj,
            }
        )
    return out


def _mean(vals: Sequence[float]) -> float:
    arr = [float(v) for v in vals if math.isfinite(float(v))]
    if not arr:
        return float("nan")
    return float(sum(arr) / len(arr))


def _collect_final_metrics(summary_path: Path) -> Dict[str, float]:
    if not summary_path.is_file():
        return {}
    obj = _load_json(summary_path)
    rows = obj.get("rows", [])
    if not isinstance(rows, list):
        return {}
    bucket: Dict[str, List[float]] = {}
    keys = (
        "direct_pose_budget_lambda_trigger_eff",
        "direct_pose_budget_lambda_chain_eff",
        "direct_pose_budget_lambda_guard_eff",
        "direct_pose_budget_trigger_ema",
        "direct_pose_budget_chain_ema",
        "direct_pose_budget_guard_ema",
        "direct_pose_budget_share_trigger",
        "direct_pose_budget_share_chain",
        "direct_pose_budget_share_guard",
        "direct_pose_budget_pcgrad_conflict_frac_trigger_guard",
        "direct_pose_budget_pcgrad_guard_drop_ratio",
        "direct_pose_budget_pcgrad_trigger_drop_ratio",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        tm = row.get("teacher_metrics", {})
        if not isinstance(tm, Mapping):
            continue
        for k in keys:
            v = _safe_float(tm.get(k))
            if math.isfinite(v):
                bucket.setdefault(k, []).append(float(v))
    return {k: _mean(vs) for k, vs in bucket.items()}


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage67 EMA-driven dynamic-lambda sweep helper.")
    ap.add_argument(
        "--config-json",
        type=str,
        default="config/posttrain_WalkF_stage7_n1leg_cg4545_v2_20260220.json",
    )
    ap.add_argument(
        "--resume-ckpt",
        type=str,
        default="models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage6_direct_cond_anchor_20260124.pth",
    )
    ap.add_argument("--temps", type=str, default="1,2,4", help="Comma list, e.g. 1,2,4.")
    ap.add_argument("--floor", type=float, default=0.15)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dataset-index-mode", type=str, default="start0")
    ap.add_argument("--run-tag", type=str, default=f"stage67_dynamic_lambda_sweep_{datetime.now().strftime('%Y%m%d')}")
    ap.add_argument("--out-root", type=str, default="", help="Default: debug_output/_<run-tag>")
    ap.add_argument("--model-root", type=str, default="", help="Default: models/MLPL2_DirectBranch_v1__<run-tag>")
    ap.add_argument("--base-run-name", type=str, default="loss_budget_n1_leg_cg4545_v2_dynlam")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    if not _RUN_LOSS.is_file():
        raise SystemExit(f"[FATAL] helper missing: {_RUN_LOSS}")

    temps = _parse_float_list(args.temps)
    seeds = _parse_int_list(args.seeds)
    cfg = _resolve_from_root(args.config_json)
    ckpt = _resolve_from_root(args.resume_ckpt)
    if not cfg.is_file():
        raise SystemExit(f"[FATAL] config not found: {cfg}")
    if (not args.dry_run) and (not ckpt.is_file()):
        raise SystemExit(f"[FATAL] resume ckpt not found: {ckpt}")

    out_root = (
        _resolve_from_root(args.out_root)
        if str(args.out_root).strip()
        else (_ROOT / "debug_output" / f"_{str(args.run_tag).strip()}")
    )
    model_root = (
        _resolve_from_root(args.model_root)
        if str(args.model_root).strip()
        else (_ROOT / "models" / f"MLPL2_DirectBranch_v1__{str(args.run_tag).strip()}")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    sweep_rows: List[SweepRow] = []
    for t in temps:
        t_slug = _fmt_slug(t)
        case_out = out_root / f"temp_{t_slug}"
        case_model = model_root / f"temp_{t_slug}"
        case_out.mkdir(parents=True, exist_ok=True)
        case_model.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [
            str(sys.executable),
            str(_RUN_LOSS),
            "--config-json",
            str(cfg),
            "--resume-ckpt",
            str(ckpt),
            "--out-dir",
            str(case_out / "run"),
            "--out-model-dir",
            str(case_model),
            "--cases",
            "r2",
            "--seeds",
            ",".join(str(s) for s in seeds),
            "--epochs",
            str(int(args.epochs)),
            "--dataset-index-mode",
            str(args.dataset_index_mode),
            "--base-run-name",
            f"{str(args.base_run_name)}_t{t_slug}",
            "--r2-under-mode",
            "twist_only",
            "--r2-under-weight",
            "2.0",
            "--r2-budget-lambda-trigger",
            "2.0",
            "--r2-budget-lambda-chain",
            "0.45",
            "--r2-budget-lambda-guard",
            "0.45",
            "--r2-budget-lambda-mode",
            "ema_softmax",
            "--r2-budget-lambda-temperature",
            f"{float(t):.8g}",
            "--r2-budget-lambda-floor",
            f"{float(args.floor):.8g}",
            "--r2-extra-override",
            "direct_pose_budget_pcgrad_mode=protect_trigger",
            "--r2-extra-override",
            "direct_pose_budget_pcgrad_guard_freeze_steps=0",
        ]
        if bool(args.skip_existing):
            cmd.append("--skip-existing")
        if bool(args.dry_run):
            cmd.append("--dry-run")

        _run(cmd, dry_run=bool(args.dry_run))

        summary_json = case_out / "run" / "summary.json"
        status_count: Dict[str, int] = {}
        trajectories: List[Dict[str, Any]] = []
        if summary_json.is_file() and (not args.dry_run):
            try:
                obj = _load_json(summary_json)
                status_obj = obj.get("status_count", {})
                if isinstance(status_obj, Mapping):
                    status_count = {str(k): int(v) for k, v in status_obj.items()}
                trajectories = _collect_trajectories(summary_json)
            except Exception:
                status_count = {"summary_parse_failed": 1}
        sweep_rows.append(
            SweepRow(
                temperature=float(t),
                out_dir=case_out / "run",
                model_dir=case_model,
                summary_json=(summary_json if summary_json.is_file() else None),
                status_count=status_count,
                trajectories=trajectories,
            )
        )

    summary_payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": str(args.run_tag),
        "config_json": str(cfg),
        "resume_ckpt": str(ckpt),
        "temps": [float(t) for t in temps],
        "floor": float(args.floor),
        "seeds": seeds,
        "epochs": int(args.epochs),
        "dataset_index_mode": str(args.dataset_index_mode),
        "rows": [],
    }

    for row in sweep_rows:
        metrics = _collect_final_metrics(row.summary_json) if row.summary_json is not None and row.summary_json.is_file() else {}
        summary_payload["rows"].append(
            {
                "temperature": float(row.temperature),
                "out_dir": str(row.out_dir),
                "model_dir": str(row.model_dir),
                "summary_json": str(row.summary_json) if row.summary_json is not None else "",
                "status_count": dict(row.status_count),
                "final_teacher_metrics_mean": metrics,
                "trajectory_count": len(row.trajectories),
            }
        )

    summary_json = out_root / "dynamic_lambda_sweep_summary.json"
    _write_json(summary_json, summary_payload)

    traj_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": str(args.run_tag),
        "rows": [
            {
                "temperature": float(row.temperature),
                "summary_json": str(row.summary_json) if row.summary_json is not None else "",
                "trajectories": row.trajectories,
            }
            for row in sweep_rows
        ],
    }
    traj_json = out_root / "dynamic_lambda_trajectories.json"
    _write_json(traj_json, traj_payload)

    lines: List[str] = []
    lines.append("# Stage67 Dynamic Lambda Sweep")
    lines.append("")
    lines.append(f"- run_tag: `{args.run_tag}`")
    lines.append(f"- config: `{cfg}`")
    lines.append(f"- resume_ckpt: `{ckpt}`")
    lines.append(f"- temps: `{','.join(str(x) for x in temps)}`")
    lines.append(f"- floor: `{float(args.floor):.4f}`")
    lines.append(f"- seeds: `{','.join(str(s) for s in seeds)}`")
    lines.append("")
    lines.append("| temp | status_count | summary_json |")
    lines.append("|---:|---|---|")
    for row in summary_payload["rows"]:
        lines.append(
            f"| {float(row['temperature']):.3f} | `{row['status_count']}` | `{row['summary_json']}` |"
        )
    lines.append("")
    lines.append(f"- summary_json: `{summary_json}`")
    lines.append(f"- trajectories_json: `{traj_json}`")
    md_path = out_root / "dynamic_lambda_sweep_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {summary_json}")
    print(f"[OK] wrote {traj_json}")
    print(f"[OK] wrote {md_path}")


if __name__ == "__main__":
    main()
