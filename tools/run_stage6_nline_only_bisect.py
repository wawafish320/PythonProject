#!/usr/bin/env python3
"""Bisect N-line-only config keys by injecting them onto Stage6 C0 baseline."""

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
_RUN_LOSS_BUDGET = _ROOT / "tools" / "run_loss_budget_r05_g0.py"
_RUN_STAGE67 = _ROOT / "tools" / "run_stage67_transition.py"


@dataclass
class ProbeResult:
    probe: str
    keys: List[str]
    train_status: str
    train_summary_json: str
    train_ckpt: str
    freerun_gate_json: str
    delta_mean_deg: float
    delta_p99_deg: float
    delta_max_deg: float
    branch_changed: Optional[bool]
    decision: str
    error: str


def _resolve_from_root(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (_ROOT / p)


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pick_summary_row(summary_obj: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    rows = summary_obj.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return {}
    best: Dict[str, Any] = {}
    best_score = -10**9
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            s = int(r.get("seed", -1))
        except Exception:
            s = -1
        if s != int(seed):
            continue
        status = str(r.get("status", ""))
        ckpt = str(r.get("ckpt", ""))
        score = 0
        if status.startswith("ok") or status == "skipped_existing":
            score += 10
        if ckpt:
            score += 2
            if Path(ckpt).is_file():
                score += 2
        if score > best_score:
            best_score = score
            best = r
    return best


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


def _fmt(v: Any) -> str:
    x = _safe_float(v)
    if math.isfinite(x):
        return f"{x:+.4f}"
    return "nan"


def _is_jump(*, dm: float, dp: float, dx: float, branch_changed: Optional[bool], thresholds: Mapping[str, float]) -> bool:
    if not (math.isfinite(dm) and math.isfinite(dp) and math.isfinite(dx)):
        return True
    if abs(dm) > float(thresholds["mean"]):
        return True
    if abs(dp) > float(thresholds["p99"]):
        return True
    if abs(dx) > float(thresholds["max"]):
        return True
    if bool(branch_changed):
        return True
    return False


def _to_bool(v: Any) -> bool:
    return bool(v)


def _write_markdown(
    *,
    out_md: Path,
    run_tag: str,
    stage6_config: Path,
    nline_config: Path,
    stage6_ckpt: Path,
    thresholds: Mapping[str, float],
    candidate_keys: Sequence[str],
    results: Sequence[ProbeResult],
    decisions: Sequence[Mapping[str, Any]],
    final_keys: Sequence[str],
) -> None:
    lines: List[str] = []
    lines.append("# Stage6 + N-line-only Bisect Summary")
    lines.append("")
    lines.append(f"- run_tag: `{run_tag}`")
    lines.append(f"- stage6_config: `{stage6_config}`")
    lines.append(f"- nline_config: `{nline_config}`")
    lines.append(f"- stage6_ckpt(B): `{stage6_ckpt}`")
    lines.append(f"- candidate_keys: `{len(candidate_keys)}`")
    lines.append(
        f"- thresholds: `|Δmean|<={thresholds['mean']}`, `|Δp99|<={thresholds['p99']}`, `|Δmax|<={thresholds['max']}`, `branch_changed=false`"
    )
    lines.append("")

    lines.append("## Probe Results")
    lines.append("")
    lines.append("| probe | n_keys | Δmean/Δp99/Δmax (deg) | branch_changed | decision |")
    lines.append("|---|---:|---:|---:|---|")
    for r in results:
        bc = "n/a" if r.branch_changed is None else str(r.branch_changed).lower()
        lines.append(
            f"| {r.probe} | {len(r.keys)} | {_fmt(r.delta_mean_deg)} / {_fmt(r.delta_p99_deg)} / {_fmt(r.delta_max_deg)} | {bc} | {r.decision} |"
        )
    lines.append("")

    lines.append("## Round Decisions")
    lines.append("")
    if decisions:
        lines.append("| round | active_before | tested_left | left_decision | next_active |")
        lines.append("|---|---:|---:|---|---:|")
        for d in decisions:
            lines.append(
                f"| {d.get('round')} | {d.get('active_before')} | {d.get('tested_left')} | {d.get('left_decision')} | {d.get('next_active')} |"
            )
    else:
        lines.append("- no round decisions (dry-run or early stop)")
    lines.append("")

    lines.append("## Final Suspects")
    lines.append("")
    if final_keys:
        for k in final_keys:
            lines.append(f"- `{k}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Probe Artifacts")
    lines.append("")
    for r in results:
        lines.append(f"- `{r.probe}`: train=`{r.train_summary_json}`, gate=`{r.freerun_gate_json}`")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_probe(
    *,
    probe_name: str,
    keys: Sequence[str],
    stage6_cfg_obj: Mapping[str, Any],
    nline_cfg_obj: Mapping[str, Any],
    stage6_ckpt: Path,
    out_root: Path,
    model_prefix: Path,
    seed: int,
    epochs: int,
    dataset_index_mode: str,
    dry_run: bool,
    skip_existing: bool,
    thresholds: Mapping[str, float],
    direct_mode: str,
) -> ProbeResult:
    probe_root = out_root / probe_name
    probe_root.mkdir(parents=True, exist_ok=True)

    merged = dict(stage6_cfg_obj)
    for k in keys:
        merged[k] = nline_cfg_obj[k]
    merged_cfg = probe_root / "merged_config.json"
    _write_json(merged_cfg, merged)

    train_out = probe_root / "train"
    model_dir = Path(f"{model_prefix}_{probe_name}")
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    train_cmd: List[str] = [
        str(sys.executable),
        str(_RUN_LOSS_BUDGET),
        "--config-json",
        str(merged_cfg),
        "--resume-ckpt",
        str(stage6_ckpt),
        "--out-dir",
        str(train_out),
        "--out-model-dir",
        str(model_dir),
        "--cases",
        "r2",
        "--seeds",
        str(int(seed)),
        "--epochs",
        str(int(epochs)),
        "--dataset-index-mode",
        str(dataset_index_mode),
        "--base-run-name",
        str(probe_name),
        "--train-config-override",
        "direct_pose_reinit=false",
        "--train-config-override",
        "train_direct_pose=true",
        "--train-config-override",
        "train_so3_corrector=false",
        "--r2-under-mode",
        "off",
        "--r2-under-weight",
        "1.0",
        "--r2-budget-lambda-trigger",
        "0.0",
        "--r2-budget-lambda-chain",
        "0.0",
        "--r2-budget-lambda-guard",
        "0.0",
        "--r2-extra-override",
        "w_direct_pose_trigger_total=0.0",
        "--r2-extra-override",
        "w_direct_pose_trigger_twist=0.0",
        "--r2-extra-override",
        "w_direct_pose_trigger_swing_x=0.0",
        "--r2-extra-override",
        "w_direct_pose_trigger_swing_y=0.0",
        "--r2-extra-override",
        "direct_pose_trigger_under_mode=off",
        "--r2-extra-override",
        "direct_pose_trigger_under_weight=1.0",
        "--r2-extra-override",
        "direct_pose_budget_mode=off",
    ]
    if skip_existing:
        train_cmd.append("--skip-existing")
    if dry_run:
        train_cmd.append("--dry-run")

    train_log = probe_root / "launcher_train.log"
    rc = _run_and_tee(train_cmd, log_path=train_log, dry_run=dry_run)
    if rc != 0:
        return ProbeResult(
            probe=probe_name,
            keys=list(keys),
            train_status=f"train_cmd_exit_{rc}",
            train_summary_json="",
            train_ckpt="",
            freerun_gate_json="",
            delta_mean_deg=float("nan"),
            delta_p99_deg=float("nan"),
            delta_max_deg=float("nan"),
            branch_changed=None,
            decision="failed",
            error=f"train command exit {rc}",
        )
    if dry_run:
        return ProbeResult(
            probe=probe_name,
            keys=list(keys),
            train_status="dry_run",
            train_summary_json=str(train_out / "summary.json"),
            train_ckpt="",
            freerun_gate_json="",
            delta_mean_deg=float("nan"),
            delta_p99_deg=float("nan"),
            delta_max_deg=float("nan"),
            branch_changed=None,
            decision="dry_run",
            error="",
        )

    summary_json = train_out / "summary.json"
    if not summary_json.is_file():
        return ProbeResult(
            probe=probe_name,
            keys=list(keys),
            train_status="missing_train_summary",
            train_summary_json="",
            train_ckpt="",
            freerun_gate_json="",
            delta_mean_deg=float("nan"),
            delta_p99_deg=float("nan"),
            delta_max_deg=float("nan"),
            branch_changed=None,
            decision="failed",
            error="missing train summary",
        )
    summary_obj = _load_json(summary_json)
    row = _pick_summary_row(summary_obj, seed=seed)
    if not row:
        return ProbeResult(
            probe=probe_name,
            keys=list(keys),
            train_status="missing_seed_row",
            train_summary_json=str(summary_json),
            train_ckpt="",
            freerun_gate_json="",
            delta_mean_deg=float("nan"),
            delta_p99_deg=float("nan"),
            delta_max_deg=float("nan"),
            branch_changed=None,
            decision="failed",
            error="missing seed row",
        )
    status = str(row.get("status", ""))
    if not (status.startswith("ok") or status == "skipped_existing"):
        return ProbeResult(
            probe=probe_name,
            keys=list(keys),
            train_status=status,
            train_summary_json=str(summary_json),
            train_ckpt=str(row.get("ckpt", "")),
            freerun_gate_json="",
            delta_mean_deg=float("nan"),
            delta_p99_deg=float("nan"),
            delta_max_deg=float("nan"),
            branch_changed=None,
            decision="failed",
            error=f"train status={status}",
        )

    freerun_out = probe_root / f"freerun_seed{int(seed)}"
    freerun_cmd: List[str] = [
        str(sys.executable),
        str(_RUN_STAGE67),
        "freerun-ab",
        "--arm-a-summary",
        str(summary_json),
        "--arm-b-ckpt",
        str(stage6_ckpt),
        "--seed",
        str(int(seed)),
        "--out-root",
        str(freerun_out),
        "--cycle-gte",
        "1",
        "--drop-wrap",
        "1",
        "--c2-policy",
        "ignore",
    ]
    dm = str(direct_mode).strip().lower()
    if dm:
        freerun_cmd.extend(["--direct-pose-fusion-direct-mode", dm])

    freerun_log = probe_root / "launcher_freerun.log"
    rc = _run_and_tee(freerun_cmd, log_path=freerun_log, dry_run=False)
    if rc != 0:
        return ProbeResult(
            probe=probe_name,
            keys=list(keys),
            train_status=status,
            train_summary_json=str(summary_json),
            train_ckpt=str(row.get("ckpt", "")),
            freerun_gate_json="",
            delta_mean_deg=float("nan"),
            delta_p99_deg=float("nan"),
            delta_max_deg=float("nan"),
            branch_changed=None,
            decision="failed",
            error=f"freerun command exit {rc}",
        )

    gate_json = freerun_out / "freerun_ab_gate.json"
    if not gate_json.is_file():
        return ProbeResult(
            probe=probe_name,
            keys=list(keys),
            train_status=status,
            train_summary_json=str(summary_json),
            train_ckpt=str(row.get("ckpt", "")),
            freerun_gate_json="",
            delta_mean_deg=float("nan"),
            delta_p99_deg=float("nan"),
            delta_max_deg=float("nan"),
            branch_changed=None,
            decision="failed",
            error="missing freerun gate",
        )

    gate = _load_json(gate_json)
    delta = (((gate.get("freerun_global", {}) or {}).get("delta_a_minus_b", {}) or {}))
    branch = gate.get("trigger_branch", {}) or {}
    dmv = _safe_float(delta.get("mean_deg", float("nan")))
    dpv = _safe_float(delta.get("p99_deg", float("nan")))
    dxv = _safe_float(delta.get("max_deg", float("nan")))
    bchg = _to_bool(branch.get("branch_changed", False))
    fail = _is_jump(dm=dmv, dp=dpv, dx=dxv, branch_changed=bchg, thresholds=thresholds)

    return ProbeResult(
        probe=probe_name,
        keys=list(keys),
        train_status=status,
        train_summary_json=str(summary_json),
        train_ckpt=str(row.get("ckpt", "")),
        freerun_gate_json=str(gate_json),
        delta_mean_deg=dmv,
        delta_p99_deg=dpv,
        delta_max_deg=dxv,
        branch_changed=bchg,
        decision="jump_or_branch_shift" if fail else "near_zero_pass",
        error="",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bisect N-line-only key set by Stage6-base incremental injection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--run-tag",
        type=str,
        default=f"stage6_nline_only_bisect_{datetime.now().strftime('%Y%m%d')}",
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
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument(
        "--candidate-keys",
        type=str,
        default="",
        help="Optional comma list of nline-only keys. Empty -> all nline-only keys.",
    )
    ap.add_argument("--check-full", action="store_true", help="Run full nline-only injection check before bisect.")
    ap.add_argument(
        "--direct-mode",
        type=str,
        default="absolute",
        choices=("", "absolute", "residual_rot6d", "residual_compose_stable"),
        help="Eval-time direct mode override for freerun-ab.",
    )
    ap.add_argument("--threshold-mean", type=float, default=0.2)
    ap.add_argument("--threshold-p99", type=float, default=1.0)
    ap.add_argument("--threshold-max", type=float, default=2.0)
    ap.add_argument("--out-root", type=str, default="")
    ap.add_argument("--model-prefix", type=str, default="")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not _RUN_LOSS_BUDGET.is_file():
        raise SystemExit(f"[FATAL] missing helper: {_RUN_LOSS_BUDGET}")
    if not _RUN_STAGE67.is_file():
        raise SystemExit(f"[FATAL] missing helper: {_RUN_STAGE67}")

    stage6_cfg_path = _resolve_from_root(args.stage6_config)
    nline_cfg_path = _resolve_from_root(args.nline_config)
    stage6_ckpt = _resolve_from_root(args.stage6_ckpt)

    if not stage6_cfg_path.is_file():
        raise SystemExit(f"[FATAL] missing Stage6 config: {stage6_cfg_path}")
    if not nline_cfg_path.is_file():
        raise SystemExit(f"[FATAL] missing N-line config: {nline_cfg_path}")
    if not args.dry_run and not stage6_ckpt.is_file():
        raise SystemExit(f"[FATAL] missing Stage6 ckpt: {stage6_ckpt}")

    stage6_cfg = _load_json(stage6_cfg_path)
    nline_cfg = _load_json(nline_cfg_path)
    if not isinstance(stage6_cfg, dict) or not isinstance(nline_cfg, dict):
        raise SystemExit("[FATAL] config json must be object")

    only_nline = sorted(set(nline_cfg.keys()) - set(stage6_cfg.keys()))
    candidate_keys = [x.strip() for x in str(args.candidate_keys).split(",") if x.strip()]
    if candidate_keys:
        missing = [k for k in candidate_keys if k not in only_nline]
        if missing:
            raise SystemExit(f"[FATAL] candidate key(s) are not nline-only: {missing}")
    else:
        candidate_keys = list(only_nline)

    if not candidate_keys:
        raise SystemExit("[FATAL] empty candidate key set")

    out_root = (
        _resolve_from_root(args.out_root)
        if str(args.out_root).strip()
        else (_ROOT / "debug_output" / f"_{str(args.run_tag).strip()}")
    )
    model_prefix = (
        _resolve_from_root(args.model_prefix)
        if str(args.model_prefix).strip()
        else (_ROOT / "models" / f"MLPL2_DirectBranch_v1__{str(args.run_tag).strip()}")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    thresholds = {
        "mean": float(args.threshold_mean),
        "p99": float(args.threshold_p99),
        "max": float(args.threshold_max),
    }

    results: List[ProbeResult] = []
    decisions: List[Dict[str, Any]] = []

    if args.check_full:
        print(f"[INFO] full-check keys={len(candidate_keys)}")
        full_res = _build_probe(
            probe_name="check_full_nline_only",
            keys=candidate_keys,
            stage6_cfg_obj=stage6_cfg,
            nline_cfg_obj=nline_cfg,
            stage6_ckpt=stage6_ckpt,
            out_root=out_root,
            model_prefix=model_prefix,
            seed=int(args.seed),
            epochs=int(args.epochs),
            dataset_index_mode=str(args.dataset_index_mode),
            dry_run=bool(args.dry_run),
            skip_existing=bool(args.skip_existing),
            thresholds=thresholds,
            direct_mode=str(args.direct_mode),
        )
        results.append(full_res)
        if (not args.dry_run) and full_res.decision != "jump_or_branch_shift":
            print("[WARN] full-check is not failing under current setup; bisect may be uninformative.")

    active = list(candidate_keys)
    max_rounds = max(0, int(args.rounds))

    for ridx in range(1, max_rounds + 1):
        if len(active) <= 1:
            break
        split = (len(active) + 1) // 2
        left = active[:split]
        right = active[split:]
        probe_name = f"r{ridx:02d}_left_{len(left)}of{len(active)}"

        print(f"[INFO] round={ridx} active={len(active)} test_left={len(left)}")
        res = _build_probe(
            probe_name=probe_name,
            keys=left,
            stage6_cfg_obj=stage6_cfg,
            nline_cfg_obj=nline_cfg,
            stage6_ckpt=stage6_ckpt,
            out_root=out_root,
            model_prefix=model_prefix,
            seed=int(args.seed),
            epochs=int(args.epochs),
            dataset_index_mode=str(args.dataset_index_mode),
            dry_run=bool(args.dry_run),
            skip_existing=bool(args.skip_existing),
            thresholds=thresholds,
            direct_mode=str(args.direct_mode),
        )
        results.append(res)

        if args.dry_run:
            decisions.append(
                {
                    "round": ridx,
                    "active_before": len(active),
                    "tested_left": len(left),
                    "left_decision": "dry_run",
                    "next_active": len(active),
                }
            )
            continue

        if res.decision == "jump_or_branch_shift":
            next_active = left
            left_decision = "fail -> keep_left"
        elif res.decision == "near_zero_pass":
            next_active = right
            left_decision = "pass -> keep_right"
        else:
            next_active = active
            left_decision = f"stop ({res.decision})"

        decisions.append(
            {
                "round": ridx,
                "active_before": len(active),
                "tested_left": len(left),
                "left_decision": left_decision,
                "next_active": len(next_active),
            }
        )

        if next_active is active:
            break
        active = list(next_active)

    out_json = out_root / "bisect_summary.json"
    out_md = out_root / "bisect_summary.md"
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
        "check_full": bool(args.check_full),
        "thresholds": thresholds,
        "nline_only_keys": only_nline,
        "candidate_keys": candidate_keys,
        "results": [
            {
                "probe": r.probe,
                "n_keys": len(r.keys),
                "keys": r.keys,
                "train_status": r.train_status,
                "train_summary_json": r.train_summary_json,
                "train_ckpt": r.train_ckpt,
                "freerun_gate_json": r.freerun_gate_json,
                "delta_mean_deg": r.delta_mean_deg,
                "delta_p99_deg": r.delta_p99_deg,
                "delta_max_deg": r.delta_max_deg,
                "branch_changed": r.branch_changed,
                "decision": r.decision,
                "error": r.error,
            }
            for r in results
        ],
        "round_decisions": decisions,
        "final_active_keys": active,
    }
    _write_json(out_json, payload)
    _write_markdown(
        out_md=out_md,
        run_tag=str(args.run_tag),
        stage6_config=stage6_cfg_path,
        nline_config=nline_cfg_path,
        stage6_ckpt=stage6_ckpt,
        thresholds=thresholds,
        candidate_keys=candidate_keys,
        results=results,
        decisions=decisions,
        final_keys=active,
    )

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")
    if not args.dry_run:
        print(f"[DONE] final active keys ({len(active)}): {active}")


if __name__ == "__main__":
    main()
