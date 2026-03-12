#!/usr/bin/env python3
"""Step4 single-factor probes for Stage6->N-line transition (seed-level)."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


_ROOT = Path(__file__).resolve().parents[1]
_RUN_LOSS_BUDGET = _ROOT / "tools" / "run_loss_budget_r05_g0.py"
_RUN_STAGE67 = _ROOT / "tools" / "run_stage67_transition.py"


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    group: str
    note: str
    dataset_index_mode: Optional[str]
    overrides: Sequence[str]


def _resolve_from_root(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (_ROOT / p)


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _print_cmd(cmd: Sequence[str]) -> None:
    print("[cmd] " + " ".join(str(x) for x in cmd))


def _run_and_tee(cmd: Sequence[str], *, log_path: Path, dry_run: bool) -> int:
    _print_cmd(cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def _base_overrides() -> List[str]:
    # Keep this aligned with Step3 `on_residual_rot6d` probe baseline.
    return [
        "pretrain_template=models/pretrain_template.json",
        "train_direct_pose=true",
        "train_so3_corrector=false",
        "direct_pose_fusion_direct_mode=residual_rot6d",
        "direct_pose_reinit=false",
        "direct_pose_trigger_gate_mode=hard",
        "direct_pose_trigger_sign_source=gt",
        "direct_pose_trigger_under_mode=off",
        "direct_pose_trigger_under_weight=1.0",
        "w_direct_pose_trigger_total=0.0",
        "w_direct_pose_trigger_twist=0.0",
        "w_direct_pose_trigger_swing_x=0.0",
        "w_direct_pose_trigger_swing_y=0.0",
        "direct_pose_budget_mode=off",
    ]


def _probe_matrix() -> List[ProbeSpec]:
    return [
        ProbeSpec(
            name="p0_base_on_residual",
            group="P0_contract",
            note="Step3 baseline replay (`train_direct_pose=on`, `residual_rot6d`).",
            dataset_index_mode=None,
            overrides=[],
        ),
        ProbeSpec(
            name="l0_stage6_lr_pack",
            group="P1_lr_optimizer",
            note="Explicitly align Stage6-like lr/optimizer schedule knobs on N-line runtime payload.",
            dataset_index_mode=None,
            overrides=[
                "lr=0.001",
                "weight_decay=0.0",
                "tf_mode=global",
                "tf_start_epoch=0",
                "tf_end_epoch=0",
                "tf_min=1.0",
                "tf_max=1.0",
                "ss_chunk_len=1",
                "freerun_stage_schedule=",
            ],
        ),
        ProbeSpec(
            name="p0_reinit_true",
            group="P0_contract",
            note="Contract check: enable `direct_pose_reinit`.",
            dataset_index_mode=None,
            overrides=["direct_pose_reinit=true"],
        ),
        ProbeSpec(
            name="p0_encoder_bundle_align",
            group="P0_contract",
            note="Contract check: explicitly align `encoder_bundle/encoder_path` spellings.",
            dataset_index_mode=None,
            overrides=[
                "encoder_bundle=models/motion_encoder_equiv_stageA.pt",
                "encoder_path=models/motion_encoder_equiv_stageA.pt",
            ],
        ),
        ProbeSpec(
            name="d0_dataset_start0",
            group="P0_data_domain",
            note="Single-factor switch: `dataset_index_mode=sic_balanced -> start0`.",
            dataset_index_mode="start0",
            overrides=[],
        ),
        ProbeSpec(
            name="d1_seq_len_87",
            group="P0_data_domain",
            note="Single-factor switch: `seq_len=60 -> 87`.",
            dataset_index_mode=None,
            overrides=["seq_len=87"],
        ),
        ProbeSpec(
            name="d2_batch_1",
            group="P0_data_domain",
            note="Single-factor switch: `batch=16 -> 1`.",
            dataset_index_mode=None,
            overrides=["batch=1"],
        ),
        ProbeSpec(
            name="s0_tf_global",
            group="P1_schedule",
            note="Schedule probe: `tf_mode=global`.",
            dataset_index_mode=None,
            overrides=["tf_mode=global"],
        ),
        ProbeSpec(
            name="s1_ss_chunk_4",
            group="P1_schedule",
            note="Schedule probe: `ss_chunk_len=4`.",
            dataset_index_mode=None,
            overrides=["ss_chunk_len=4"],
        ),
        ProbeSpec(
            name="s2_stage_schedule_off",
            group="P1_schedule",
            note="Schedule probe: disable stage schedule (`freerun_stage_schedule=''`).",
            dataset_index_mode=None,
            overrides=["freerun_stage_schedule="],
        ),
    ]


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


def _parse_trainable_from_log(log_path: Path) -> Dict[str, Any]:
    if not log_path.is_file():
        return {"trainable_count": None, "trainable_params": []}
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    # Example:
    # [posttrain] trainable=6 params: direct_pose_head.0.weight, ...
    m = re.search(r"\[posttrain\]\s*trainable=(\d+)\s+params:\s*(.+)", txt)
    if not m:
        return {"trainable_count": None, "trainable_params": []}
    cnt = int(m.group(1))
    raw = str(m.group(2)).strip()
    params = [x.strip() for x in raw.split(",") if x.strip()]
    return {"trainable_count": cnt, "trainable_params": params}


def _runtime_snapshot(row: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    runtime_cfg = Path(str(row.get("runtime_config_json", "") or "")).expanduser()
    if not runtime_cfg.is_file():
        return out
    obj = _load_json(runtime_cfg)
    keys = [
        "train_direct_pose",
        "train_so3_corrector",
        "direct_pose_fusion_direct_mode",
        "direct_pose_reinit",
        "dataset_index_mode",
        "seq_len",
        "batch",
        "tf_mode",
        "tf_start_epoch",
        "tf_end_epoch",
        "tf_min",
        "tf_max",
        "ss_chunk_len",
        "freerun_stage_schedule",
        "encoder_bundle",
        "encoder_path",
        "ckpt_in",
    ]
    for k in keys:
        if k in obj:
            out[k] = obj[k]
    out["runtime_config_json"] = str(runtime_cfg)
    return out


def _changed_keys(overrides: Sequence[str]) -> List[str]:
    keys: List[str] = []
    for ov in overrides:
        txt = str(ov)
        if "=" not in txt:
            continue
        k = txt.split("=", 1)[0].strip()
        if k and k not in keys:
            keys.append(k)
    return keys


def _format_signed(x: Any) -> str:
    v = _safe_float(x)
    if math.isfinite(v):
        return f"{v:+.4f}"
    return "nan"


def _build_markdown(
    *,
    run_tag: str,
    out_root: Path,
    stage6_ckpt: Path,
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
    baseline_name: str,
) -> str:
    baseline = next((r for r in rows if str(r.get("probe")) == baseline_name), None)
    lines: List[str] = []
    lines.append("# Step4 Single-Factor Probe Summary")
    lines.append("")
    lines.append(f"- run_tag: `{run_tag}`")
    lines.append(f"- stage6_ckpt(B): `{stage6_ckpt}`")
    lines.append(f"- out_root: `{out_root}`")
    lines.append("- mask: `cycle>=1 + drop_wrap + exclude root`")
    lines.append("")
    lines.append(
        f"- thresholds: `|Δmean|<={thresholds['mean']}`, `|Δp99|<={thresholds['p99']}`, "
        f"`|Δmax|<={thresholds['max']}`"
    )
    if baseline is not None:
        lines.append(
            f"- baseline `{baseline_name}`: "
            f"`Δmean/Δp99/Δmax={_format_signed(baseline.get('delta_mean_deg'))}/"
            f"{_format_signed(baseline.get('delta_p99_deg'))}/{_format_signed(baseline.get('delta_max_deg'))}`"
        )
    lines.append("")
    lines.append("| probe | group | Δmean/Δp99/Δmax (deg) | vs baseline Δmean | branch_changed | trainable | verdict |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r.get('probe')} | {r.get('group')} | "
            f"{_format_signed(r.get('delta_mean_deg'))} / {_format_signed(r.get('delta_p99_deg'))} / {_format_signed(r.get('delta_max_deg'))} | "
            f"{_format_signed(r.get('delta_mean_vs_baseline'))} | "
            f"{str(r.get('branch_changed')).lower()} | "
            f"{r.get('trainable_count')} | "
            f"{r.get('user_gate')} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for r in rows:
        lines.append(f"- `{r.get('probe')}`: {r.get('note')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Step4 single-factor probes from on_residual_rot6d baseline.")
    ap.add_argument(
        "--run-tag",
        type=str,
        default=f"stage6_n1leg_v2_step4_p0_data_sched_{datetime.now().strftime('%Y%m%d')}",
    )
    ap.add_argument("--config-json", type=str, default="config/exp_phase_DirectBranch_v1_d1_noreset.json")
    ap.add_argument(
        "--stage6-ckpt",
        type=str,
        default="models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage6_direct_cond_anchor_20260124.pth",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--dataset-index-mode", type=str, default="sic_balanced")
    ap.add_argument("--out-root", type=str, default="")
    ap.add_argument("--model-prefix", type=str, default="")
    ap.add_argument(
        "--probe-names",
        type=str,
        default="",
        help="Optional comma list to run subset, e.g. p0_base_on_residual,d0_dataset_start0",
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not _RUN_LOSS_BUDGET.is_file():
        raise SystemExit(f"[FATAL] missing script: {_RUN_LOSS_BUDGET}")
    if not _RUN_STAGE67.is_file():
        raise SystemExit(f"[FATAL] missing script: {_RUN_STAGE67}")

    cfg = _resolve_from_root(args.config_json)
    stage6_ckpt = _resolve_from_root(args.stage6_ckpt)
    if not cfg.is_file():
        raise SystemExit(f"[FATAL] config missing: {cfg}")
    if not args.dry_run and not stage6_ckpt.is_file():
        raise SystemExit(f"[FATAL] stage6 ckpt missing: {stage6_ckpt}")

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

    probes = _probe_matrix()
    allow = [x.strip() for x in str(args.probe_names).split(",") if x.strip()]
    if allow:
        allow_set = set(allow)
        probes = [p for p in probes if p.name in allow_set]
        missing = [x for x in allow if x not in {p.name for p in _probe_matrix()}]
        if missing:
            raise SystemExit(f"[FATAL] unknown probe(s): {missing}")
    if not probes:
        raise SystemExit("[FATAL] no probes to run")

    thresholds = {"mean": 0.2, "p99": 1.0, "max": 2.0}
    base_ov = _base_overrides()
    summary_rows: List[Dict[str, Any]] = []

    def _append_failed_probe(
        *,
        probe: ProbeSpec,
        dataset_mode: str,
        reason: str,
        train_summary: Optional[Path] = None,
        train_row: Optional[Mapping[str, Any]] = None,
    ) -> None:
        row = dict(train_row or {})
        summary_rows.append(
            {
                "probe": probe.name,
                "group": probe.group,
                "note": probe.note,
                "dataset_index_mode": dataset_mode,
                "overrides": list(base_ov) + list(probe.overrides),
                "override_keys": _changed_keys(list(probe.overrides)),
                "train_summary_json": str(train_summary) if train_summary else "",
                "train_status": str(row.get("status", "failed")),
                "train_ckpt": str(row.get("ckpt", "")),
                "train_log_path": str(row.get("log_path", "")),
                "trainable_count": None,
                "trainable_params": [],
                "runtime": _runtime_snapshot(row) if row else {},
                "freerun_gate_json": "",
                "delta_mean_deg": float("nan"),
                "delta_p99_deg": float("nan"),
                "delta_max_deg": float("nan"),
                "branch_changed": None,
                "trigger_branch_a": "NA",
                "trigger_branch_b": "NA",
                "quickgate_decision": "not_available",
                "user_gate": f"failed:{reason}",
            }
        )

    for probe in probes:
        dataset_mode = probe.dataset_index_mode or str(args.dataset_index_mode)
        train_out = out_root / probe.name / "train"
        model_dir = Path(f"{model_prefix}_{probe.name}")
        train_launcher = train_out / "launcher.log"

        train_cmd: List[str] = [
            str(sys.executable),
            str(_RUN_LOSS_BUDGET),
            "--config-json",
            str(cfg),
            "--resume-ckpt",
            str(stage6_ckpt),
            "--out-dir",
            str(train_out),
            "--out-model-dir",
            str(model_dir),
            "--cases",
            "r2",
            "--seeds",
            str(int(args.seed)),
            "--epochs",
            str(int(args.epochs)),
            "--dataset-index-mode",
            str(dataset_mode),
            "--base-run-name",
            str(probe.name),
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
        ]
        for ov in list(base_ov) + list(probe.overrides):
            train_cmd.extend(["--train-config-override", str(ov)])
        if bool(args.skip_existing):
            train_cmd.append("--skip-existing")
        if bool(args.dry_run):
            train_cmd.append("--dry-run")

        rc = _run_and_tee(train_cmd, log_path=train_launcher, dry_run=bool(args.dry_run))
        if rc != 0:
            _append_failed_probe(probe=probe, dataset_mode=dataset_mode, reason=f"train_cmd_exit_{rc}")
            continue
        if args.dry_run:
            continue

        train_summary = train_out / "summary.json"
        if not train_summary.is_file():
            _append_failed_probe(probe=probe, dataset_mode=dataset_mode, reason="missing_train_summary")
            continue
        train_obj = _load_json(train_summary)
        row = _pick_summary_row(train_obj, seed=int(args.seed))
        if not row:
            _append_failed_probe(
                probe=probe,
                dataset_mode=dataset_mode,
                reason="missing_seed_row",
                train_summary=train_summary,
            )
            continue
        status = str(row.get("status", ""))
        if not (status.startswith("ok") or status == "skipped_existing"):
            _append_failed_probe(
                probe=probe,
                dataset_mode=dataset_mode,
                reason=f"train_status_{status}",
                train_summary=train_summary,
                train_row=row,
            )
            continue

        freerun_out = out_root / probe.name / f"freerun_seed{int(args.seed)}"
        freerun_launcher = freerun_out / "launcher.log"
        freerun_cmd: List[str] = [
            str(sys.executable),
            str(_RUN_STAGE67),
            "freerun-ab",
            "--arm-a-summary",
            str(train_summary),
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
            "--direct-pose-fusion-direct-mode",
            "residual_rot6d",
        ]
        rc = _run_and_tee(freerun_cmd, log_path=freerun_launcher, dry_run=False)
        if rc != 0:
            _append_failed_probe(
                probe=probe,
                dataset_mode=dataset_mode,
                reason=f"freerun_cmd_exit_{rc}",
                train_summary=train_summary,
                train_row=row,
            )
            continue

        gate_json = freerun_out / "freerun_ab_gate.json"
        if not gate_json.is_file():
            _append_failed_probe(
                probe=probe,
                dataset_mode=dataset_mode,
                reason="missing_freerun_gate",
                train_summary=train_summary,
                train_row=row,
            )
            continue
        gate = _load_json(gate_json)
        delta = (((gate.get("freerun_global", {}) or {}).get("delta_a_minus_b", {}) or {}))
        branch = gate.get("trigger_branch", {}) or {}
        tinfo = _parse_trainable_from_log(Path(str(row.get("log_path", ""))))
        runtime = _runtime_snapshot(row)
        dm = _safe_float(delta.get("mean_deg", float("nan")))
        dp = _safe_float(delta.get("p99_deg", float("nan")))
        dx = _safe_float(delta.get("max_deg", float("nan")))
        user_gate = bool(
            math.isfinite(dm)
            and math.isfinite(dp)
            and math.isfinite(dx)
            and abs(dm) <= float(thresholds["mean"])
            and abs(dp) <= float(thresholds["p99"])
            and abs(dx) <= float(thresholds["max"])
            and not bool(branch.get("branch_changed", True))
        )

        summary_rows.append(
            {
                "probe": probe.name,
                "group": probe.group,
                "note": probe.note,
                "dataset_index_mode": dataset_mode,
                "overrides": list(base_ov) + list(probe.overrides),
                "override_keys": _changed_keys(list(probe.overrides)),
                "train_summary_json": str(train_summary),
                "train_status": status,
                "train_ckpt": str(row.get("ckpt", "")),
                "train_log_path": str(row.get("log_path", "")),
                "trainable_count": tinfo.get("trainable_count"),
                "trainable_params": tinfo.get("trainable_params", []),
                "runtime": runtime,
                "freerun_gate_json": str(gate_json),
                "delta_mean_deg": dm,
                "delta_p99_deg": dp,
                "delta_max_deg": dx,
                "branch_changed": bool(branch.get("branch_changed", False)),
                "trigger_branch_a": str((branch.get("arm_a", {}) or {}).get("trigger_branch", "NA")),
                "trigger_branch_b": str((branch.get("arm_b", {}) or {}).get("trigger_branch", "NA")),
                "quickgate_decision": str((gate.get("decision", {}) or {}).get("quickgate_4_4_2", "unknown")),
                "user_gate": "near_zero_pass" if user_gate else "jump_or_branch_shift",
            }
        )

    if args.dry_run:
        print("[OK] dry-run completed.")
        return

    baseline_name = "p0_base_on_residual"
    baseline = next((r for r in summary_rows if str(r.get("probe")) == baseline_name), None)
    b_mean = _safe_float((baseline or {}).get("delta_mean_deg", float("nan")))
    b_p99 = _safe_float((baseline or {}).get("delta_p99_deg", float("nan")))
    b_max = _safe_float((baseline or {}).get("delta_max_deg", float("nan")))
    for r in summary_rows:
        dm = _safe_float(r.get("delta_mean_deg", float("nan")))
        dp = _safe_float(r.get("delta_p99_deg", float("nan")))
        dx = _safe_float(r.get("delta_max_deg", float("nan")))
        r["delta_mean_vs_baseline"] = dm - b_mean if math.isfinite(dm) and math.isfinite(b_mean) else float("nan")
        r["delta_p99_vs_baseline"] = dp - b_p99 if math.isfinite(dp) and math.isfinite(b_p99) else float("nan")
        r["delta_max_vs_baseline"] = dx - b_max if math.isfinite(dx) and math.isfinite(b_max) else float("nan")

    best_mean = min(
        summary_rows,
        key=lambda r: _safe_float(r.get("delta_mean_deg", float("inf"))),
    ) if summary_rows else None
    best_p99 = min(
        summary_rows,
        key=lambda r: _safe_float(r.get("delta_p99_deg", float("inf"))),
    ) if summary_rows else None
    best_max = min(
        summary_rows,
        key=lambda r: _safe_float(r.get("delta_max_deg", float("inf"))),
    ) if summary_rows else None

    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": str(args.run_tag),
        "config_json": str(cfg),
        "stage6_ckpt": str(stage6_ckpt),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "default_dataset_index_mode": str(args.dataset_index_mode),
        "thresholds": thresholds,
        "baseline_probe": baseline_name,
        "base_overrides": base_ov,
        "rows": summary_rows,
        "best": {
            "delta_mean_min_probe": (best_mean or {}).get("probe"),
            "delta_p99_min_probe": (best_p99 or {}).get("probe"),
            "delta_max_min_probe": (best_max or {}).get("probe"),
        },
    }

    out_json = out_root / "step4_single_factor_probe_summary.json"
    out_md = out_root / "step4_single_factor_probe_summary.md"
    _write_json(out_json, payload)
    out_md.write_text(
        _build_markdown(
            run_tag=str(args.run_tag),
            out_root=out_root,
            stage6_ckpt=stage6_ckpt,
            rows=summary_rows,
            thresholds=thresholds,
            baseline_name=baseline_name,
        ),
        encoding="utf-8",
    )
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
