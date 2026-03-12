#!/usr/bin/env python3
"""Minimal confirmation runs for narrowed Stage6+N-line-only suspect keys."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


_ROOT = Path(__file__).resolve().parents[1]
_RUN_STAGE67 = _ROOT / "tools" / "run_stage67_transition.py"


_DEFAULT_CASES: Dict[str, List[str]] = {
    "amp_only": ["amp"],
    "contact_only": [
        "contact_meas_dropout",
        "contact_meas_enable",
        "contact_meas_hidden",
        "contact_phase_state_enable",
    ],
}


_COMMON_OVERRIDES: Dict[str, Any] = {
    "direct_pose_reinit": False,
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


def _resolve(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (_ROOT / p)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_and_tee(cmd: Sequence[str], *, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[cmd] " + " ".join(str(x) for x in cmd))
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd]\n")
        f.write(" ".join(str(x) for x in cmd) + "\n\n")
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


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _fmt(v: float) -> str:
    return f"{v:+.4f}" if math.isfinite(v) else "nan"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run minimal Stage6+N-line-only confirmation cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--run-tag",
        type=str,
        default=f"stage6_nline_only_min_confirm_{datetime.now().strftime('%Y%m%d')}",
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
    ap.add_argument(
        "--cases",
        type=str,
        default="amp_only,contact_only",
        help="Comma list from {amp_only,contact_only}.",
    )
    ap.add_argument(
        "--direct-mode",
        type=str,
        default="absolute",
        choices=("", "absolute", "residual_rot6d", "residual_compose_stable"),
    )
    ap.add_argument("--out-root", type=str, default="")
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
    if not stage6_ckpt.is_file():
        raise SystemExit(f"[FATAL] missing stage6 ckpt: {stage6_ckpt}")

    stage6_cfg = _load_json(stage6_cfg_path)
    nline_cfg = _load_json(nline_cfg_path)
    if not isinstance(stage6_cfg, dict) or not isinstance(nline_cfg, dict):
        raise SystemExit("[FATAL] config json must be object")

    requested = [x.strip() for x in str(args.cases).split(",") if x.strip()]
    if not requested:
        raise SystemExit("[FATAL] empty --cases")
    for c in requested:
        if c not in _DEFAULT_CASES:
            raise SystemExit(f"[FATAL] unknown case: {c}")

    out_root = _resolve(args.out_root) if str(args.out_root).strip() else (_ROOT / "debug_output" / f"_{args.run_tag}")
    out_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for case in requested:
        keys = list(_DEFAULT_CASES[case])
        case_root = out_root / case
        case_root.mkdir(parents=True, exist_ok=True)

        merged = dict(stage6_cfg)
        for k in keys:
            if k not in nline_cfg:
                raise SystemExit(f"[FATAL] key {k} not found in nline config")
            merged[k] = nline_cfg[k]
        merged.update(_COMMON_OVERRIDES)

        cfg_path = case_root / "train_runtime.json"
        _write_json(cfg_path, merged)

        model_dir = _ROOT / "models" / f"MLPL2_DirectBranch_v1__{args.run_tag}_{case}"
        model_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"{args.run_tag}_{case}_seed{int(args.seed)}_e{int(args.epochs)}"

        train_cmd = [
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
            str(int(args.seed)),
            "--dataset_index_mode",
            str(args.dataset_index_mode),
            "--epochs",
            str(int(args.epochs)),
            "--ckpt_in",
            str(stage6_ckpt),
        ]
        train_rc = _run_and_tee(train_cmd, log_path=case_root / "train.log")
        if train_rc != 0:
            results.append(
                {
                    "case": case,
                    "keys": keys,
                    "status": f"train_cmd_exit_{train_rc}",
                }
            )
            continue

        ckpt = model_dir / f"ckpt_last_{run_name}.pth"
        if not ckpt.is_file():
            results.append(
                {
                    "case": case,
                    "keys": keys,
                    "status": "missing_ckpt",
                    "ckpt": str(ckpt),
                }
            )
            continue

        freerun_out = case_root / "freerun"
        freerun_cmd = [
            str(sys.executable),
            str(_RUN_STAGE67),
            "freerun-ab",
            "--arm-a-ckpt",
            str(ckpt),
            "--arm-b-ckpt",
            str(stage6_ckpt),
            "--seed",
            str(int(args.seed)),
            "--out-root",
            str(freerun_out),
            "--cycle-gte",
            "1",
            "--drop-wrap",
            "1",
            "--c2-policy",
            "ignore",
        ]
        dm = str(args.direct_mode).strip().lower()
        if dm:
            freerun_cmd.extend(["--direct-pose-fusion-direct-mode", dm])
        freerun_rc = _run_and_tee(freerun_cmd, log_path=case_root / "freerun.log")
        if freerun_rc != 0:
            results.append(
                {
                    "case": case,
                    "keys": keys,
                    "status": f"freerun_cmd_exit_{freerun_rc}",
                    "ckpt": str(ckpt),
                }
            )
            continue

        gate_json = freerun_out / "freerun_ab_gate.json"
        if not gate_json.is_file():
            results.append(
                {
                    "case": case,
                    "keys": keys,
                    "status": "missing_gate_json",
                    "ckpt": str(ckpt),
                }
            )
            continue

        gate = _load_json(gate_json)
        delta = (((gate.get("freerun_global", {}) or {}).get("delta_a_minus_b", {}) or {}))
        branch = (gate.get("trigger_branch", {}) or {})
        results.append(
            {
                "case": case,
                "keys": keys,
                "status": "ok",
                "ckpt": str(ckpt),
                "gate_json": str(gate_json),
                "delta_mean_deg": _safe_float(delta.get("mean_deg", float("nan"))),
                "delta_p99_deg": _safe_float(delta.get("p99_deg", float("nan"))),
                "delta_max_deg": _safe_float(delta.get("max_deg", float("nan"))),
                "branch_changed": bool(branch.get("branch_changed", False)),
            }
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
        "direct_mode": str(args.direct_mode),
        "results": results,
    }

    out_json = out_root / "min_confirm_summary.json"
    out_md = out_root / "min_confirm_summary.md"
    _write_json(out_json, payload)

    lines: List[str] = []
    lines.append("# Stage6 + N-line-only Min Confirm")
    lines.append("")
    lines.append(f"- run_tag: `{args.run_tag}`")
    lines.append(f"- seed: `{int(args.seed)}`")
    lines.append(f"- epochs: `{int(args.epochs)}`")
    lines.append(f"- dataset_index_mode: `{args.dataset_index_mode}`")
    lines.append("")
    lines.append("| case | keys | Δmean/Δp99/Δmax (deg) | branch_changed | status |")
    lines.append("|---|---|---:|---:|---|")
    for r in results:
        dm = _fmt(_safe_float(r.get("delta_mean_deg", float("nan"))))
        dp = _fmt(_safe_float(r.get("delta_p99_deg", float("nan"))))
        dx = _fmt(_safe_float(r.get("delta_max_deg", float("nan"))))
        bc = str(r.get("branch_changed", "n/a")).lower()
        lines.append(
            f"| {r.get('case')} | {','.join(r.get('keys', []))} | {dm} / {dp} / {dx} | {bc} | {r.get('status')} |"
        )
    lines.append("")
    for r in results:
        if "gate_json" in r:
            lines.append(f"- `{r['case']}` gate: `{r['gate_json']}`")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
