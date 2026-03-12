#!/usr/bin/env python3
"""
Run H2a minimal matrix (adapter / residual bypass) after Phase2-A gate fallback.

Matrix (default):
  C0 = control (linear leg head, cond-only direct features)
  H1 = adapter-warm (res_mlp leg head, alpha_init>0)
  H2 = pre0 bypass (direct feat source -> cond+hidden_pre, reinit direct head)

Protocol:
  - Step7 best branch defaults (e3, lr=3e-4)
  - 3 seeds
  - fixed-table + no-table + probe triple eval
  - no-table-first gate (table is optional only)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


RE_PROBE_MEAN = re.compile(r"\[Probe cos\]\[loo_cycle\]\s+[^:]+:\s+mean=([+-]?[0-9]*\.?[0-9]+)")


@dataclass(frozen=True)
class CaseSpec:
    key: str
    desc: str
    config_overrides: Dict[str, Any]


CASES: Dict[str, CaseSpec] = {
    "C0": CaseSpec(
        key="C0",
        desc="control_linear_cond",
        config_overrides={
            "direct_pose_leg_head_variant": "linear",
            "direct_pose_leg_head_hidden": 0,
            "direct_pose_leg_head_res_alpha_init": 0.0,
            "direct_pose_feat_source": "cond",
            "direct_pose_reinit": False,
        },
    ),
    "H1": CaseSpec(
        key="H1",
        desc="adapter_warm_resmlp32",
        config_overrides={
            "direct_pose_leg_head_variant": "res_mlp",
            "direct_pose_leg_head_hidden": 32,
            "direct_pose_leg_head_res_alpha_init": 0.05,
            "direct_pose_feat_source": "cond",
            "direct_pose_reinit": False,
        },
    ),
    "H2": CaseSpec(
        key="H2",
        desc="bypass_pre0_linear",
        config_overrides={
            "direct_pose_leg_head_variant": "linear",
            "direct_pose_leg_head_hidden": 0,
            "direct_pose_leg_head_res_alpha_init": 0.0,
            "direct_pose_feat_source": "cond+hidden_pre",
            "direct_pose_reinit": True,
        },
    ),
    "H3": CaseSpec(
        key="H3",
        desc="adapter_plus_bypass",
        config_overrides={
            "direct_pose_leg_head_variant": "res_mlp",
            "direct_pose_leg_head_hidden": 32,
            "direct_pose_leg_head_res_alpha_init": 0.05,
            "direct_pose_feat_source": "cond+hidden_pre",
            "direct_pose_reinit": True,
        },
    ),
}


STEP7_CASE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "r2": {
        "direct_pose_leg_stopgrad_main": False,
        "direct_pose_leg_detach_feat": True,
    },
    "r3": {
        "direct_pose_leg_stopgrad_main": False,
        "direct_pose_leg_detach_feat": False,
    },
    "base": {},
}


def _run(cmd: List[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "." if not py else f".:{py}"
    print("[CMD]", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def _load_gate_metrics(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text())
    cmp = obj.get("compare", {}) if isinstance(obj, dict) else {}
    return {
        "sign_match_rate": float(cmp.get("sign_match_rate", obj.get("sign_match_rate", 0.0))),
        "median_abs_mu_z_diff_deg": float(
            cmp.get("median_abs_mu_z_diff_deg", obj.get("median_abs_mu_z_diff_deg", 0.0))
        ),
        "gate_pass": bool(cmp.get("gate_pass", obj.get("gate_pass", False))),
        "mismatch_sics": list(obj.get("mismatch_sics", []) or []),
    }


def _run_probe(
    *,
    repo_root: Path,
    json_path: Path,
    feature: str,
    model_kind: str,
    sics: str,
    seed: int,
) -> Dict[str, Any]:
    cmd = [
        "python",
        "-m",
        "tools.diag_legomega_linear_probe",
        "--json",
        str(json_path),
        "--bones",
        "calf_r",
        "--feature",
        feature,
        "--split",
        "loo_cycle",
        "--model",
        model_kind,
        "--sics",
        sics,
        "--use_oracle_right",
        "--seed",
        str(seed),
    ]
    cp = _run(cmd, cwd=repo_root)
    mean_val: Optional[float] = None
    for line in cp.stdout.splitlines():
        m = RE_PROBE_MEAN.search(line)
        if m:
            mean_val = float(m.group(1))
            break
    if mean_val is None:
        raise RuntimeError(f"Failed to parse probe mean from output ({feature}, {model_kind}, sics={sics}).")
    return {"mean": mean_val, "stdout": cp.stdout}


def _stat(vals: List[float]) -> Dict[str, float]:
    x = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not x:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    n = float(len(x))
    mu = float(sum(x) / len(x))
    var = float(sum((v - mu) ** 2 for v in x) / len(x))
    return {
        "n": n,
        "mean": mu,
        "std": math.sqrt(max(0.0, var)),
        "min": float(min(x)),
        "max": float(max(x)),
    }


def _aggregate_by_case(rows: List[Dict[str, Any]], case_order: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in case_order:
        rr = [r for r in rows if str(r.get("case")) == key]
        if not rr:
            continue
        fs = _stat([float(r["fixed_table"]["sign_match_rate"]) for r in rr])
        fm = _stat([float(r["fixed_table"]["median_abs_mu_z_diff_deg"]) for r in rr])
        ns = _stat([float(r["no_table"]["sign_match_rate"]) for r in rr])
        nm = _stat([float(r["no_table"]["median_abs_mu_z_diff_deg"]) for r in rr])
        fgp = _stat([1.0 if bool(r["fixed_table"].get("gate_pass", False)) else 0.0 for r in rr])
        ngp = _stat([1.0 if bool(r["no_table"].get("gate_pass", False)) else 0.0 for r in rr])
        hp = _stat([float(r["probe"]["hotspot"]["pre0_linear"]) for r in rr])
        out[key] = {
            "n": float(len(rr)),
            "fixed_sign_mean": fs["mean"],
            "fixed_sign_std": fs["std"],
            "fixed_sign_min": fs["min"],
            "fixed_med_mean": fm["mean"],
            "fixed_med_std": fm["std"],
            "fixed_med_max": fm["max"],
            "fixed_gate_pass_rate": fgp["mean"],
            "no_table_sign_mean": ns["mean"],
            "no_table_sign_std": ns["std"],
            "no_table_sign_min": ns["min"],
            "no_table_med_mean": nm["mean"],
            "no_table_med_std": nm["std"],
            "no_table_med_max": nm["max"],
            "no_table_gate_pass_rate": ngp["mean"],
            "hotspot_pre0_linear_mean": hp["mean"],
            "hotspot_pre0_linear_std": hp["std"],
        }
    return out


def _eval_case_gate(
    cur: Dict[str, float],
    *,
    no_table_sign_mean_min: float,
    no_table_sign_min_min: float,
    no_table_med_mean_max: float,
    no_table_med_max_max: float,
    no_table_gate_pass_rate_min: float,
    table_sign_gain_max: float,
    table_med_gain_max_deg: float,
) -> Dict[str, Any]:
    no_table_self = {
        "sign_mean_ok": bool(float(cur["no_table_sign_mean"]) >= float(no_table_sign_mean_min)),
        "sign_min_ok": bool(float(cur["no_table_sign_min"]) >= float(no_table_sign_min_min)),
        "med_mean_ok": bool(float(cur["no_table_med_mean"]) <= float(no_table_med_mean_max)),
        "med_max_ok": bool(float(cur["no_table_med_max"]) <= float(no_table_med_max_max)),
        "gate_pass_rate_ok": bool(float(cur["no_table_gate_pass_rate"]) >= float(no_table_gate_pass_rate_min)),
    }
    table_sign_gain = float(cur["fixed_sign_mean"]) - float(cur["no_table_sign_mean"])
    table_med_gain = float(cur["no_table_med_mean"]) - float(cur["fixed_med_mean"])
    table_dependency = {
        "sign_gain_marginal": bool(table_sign_gain <= float(table_sign_gain_max)),
        "med_gain_marginal": bool(table_med_gain <= float(table_med_gain_max_deg)),
    }
    no_table_self_sufficient = bool(all(no_table_self.values()))
    table_optional = bool(all(table_dependency.values()))
    ok = bool(no_table_self_sufficient and table_optional)
    return {
        "pass": ok,
        "no_table_self_sufficient": no_table_self_sufficient,
        "table_optional_marginal_gain_only": table_optional,
        "checks": {
            "no_table": no_table_self,
            "table_dependency": table_dependency,
        },
        "metrics": {
            "no_table_sign_mean": float(cur["no_table_sign_mean"]),
            "no_table_sign_min": float(cur["no_table_sign_min"]),
            "no_table_med_mean": float(cur["no_table_med_mean"]),
            "no_table_med_max": float(cur["no_table_med_max"]),
            "no_table_gate_pass_rate": float(cur["no_table_gate_pass_rate"]),
            "table_sign_gain": float(table_sign_gain),
            "table_med_gain_deg": float(table_med_gain),
        },
    }


def _phase2b_gate(
    case_stats: Dict[str, Dict[str, float]],
    *,
    candidate_cases: List[str],
    no_table_sign_mean_min: float,
    no_table_sign_min_min: float,
    no_table_med_mean_max: float,
    no_table_med_max_max: float,
    no_table_gate_pass_rate_min: float,
    table_sign_gain_max: float,
    table_med_gain_max_deg: float,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    passed: List[str] = []
    for k in candidate_cases:
        cur = case_stats.get(k)
        if cur is None:
            checks[k] = {"pass": False, "reason": "missing_case"}
            continue
        ck = _eval_case_gate(
            cur,
            no_table_sign_mean_min=no_table_sign_mean_min,
            no_table_sign_min_min=no_table_sign_min_min,
            no_table_med_mean_max=no_table_med_mean_max,
            no_table_med_max_max=no_table_med_max_max,
            no_table_gate_pass_rate_min=no_table_gate_pass_rate_min,
            table_sign_gain_max=table_sign_gain_max,
            table_med_gain_max_deg=table_med_gain_max_deg,
        )
        checks[k] = ck
        if bool(ck.get("pass", False)):
            passed.append(k)
    decision = "continue_h2a_scaleup" if passed else "h2a_nohit_expand_matrix"
    return {
        "decision": decision,
        "passed_cases": passed,
        "candidate_cases": candidate_cases,
        "standard": "no_table_self_sufficient_table_optional_only",
        "thresholds": {
            "no_table_sign_mean_min": float(no_table_sign_mean_min),
            "no_table_sign_min_min": float(no_table_sign_min_min),
            "no_table_med_mean_max": float(no_table_med_mean_max),
            "no_table_med_max_max": float(no_table_med_max_max),
            "no_table_gate_pass_rate_min": float(no_table_gate_pass_rate_min),
            "table_sign_gain_max": float(table_sign_gain_max),
            "table_med_gain_max_deg": float(table_med_gain_max_deg),
        },
        "checks": checks,
    }


def _write_md(summary: Dict[str, Any], out_md: Path, case_order: List[str]) -> None:
    rows = list(summary.get("rows", []) or [])
    case_stats = dict(summary.get("case_stats", {}) or {})
    gate = dict(summary.get("phase2b_gate", {}) or {})
    lines: List[str] = []
    lines.append("# H2a minimal matrix summary")
    lines.append("")
    lines.append("| case | seed | fixed sign | fixed med_abs | no-table sign | no-table med_abs | hotspot pre0 linear |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {case} | {seed} | {fs:.3f} | {fm:.6f} | {ns:.3f} | {nm:.6f} | {hp:.4f} |".format(
                case=r["case"],
                seed=r["seed"],
                fs=r["fixed_table"]["sign_match_rate"],
                fm=r["fixed_table"]["median_abs_mu_z_diff_deg"],
                ns=r["no_table"]["sign_match_rate"],
                nm=r["no_table"]["median_abs_mu_z_diff_deg"],
                hp=r["probe"]["hotspot"]["pre0_linear"],
            )
        )
    lines.append("")
    lines.append("## Aggregated (per case)")
    lines.append("")
    lines.append(
        "| case | n | fixed sign mean±std | fixed med_abs mean±std | no-table sign mean±std(min) | no-table med_abs mean±std(max) | no-table gate pass rate | hotspot pre0 linear mean±std |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in case_order:
        c = case_stats.get(key)
        if not isinstance(c, dict):
            continue
        lines.append(
            "| {k} | {n:.0f} | {fsm:.3f}±{fss:.3f} | {fmm:.6f}±{fms:.6f} | {nsm:.3f}±{nss:.3f} ({nmin:.3f}) | {nmm:.6f}±{nms:.6f} ({nmax:.6f}) | {ngp:.3f} | {hpm:.4f}±{hps:.4f} |".format(
                k=key,
                n=float(c.get("n", 0.0)),
                fsm=float(c.get("fixed_sign_mean", float("nan"))),
                fss=float(c.get("fixed_sign_std", float("nan"))),
                fmm=float(c.get("fixed_med_mean", float("nan"))),
                fms=float(c.get("fixed_med_std", float("nan"))),
                nsm=float(c.get("no_table_sign_mean", float("nan"))),
                nss=float(c.get("no_table_sign_std", float("nan"))),
                nmin=float(c.get("no_table_sign_min", float("nan"))),
                nmm=float(c.get("no_table_med_mean", float("nan"))),
                nms=float(c.get("no_table_med_std", float("nan"))),
                nmax=float(c.get("no_table_med_max", float("nan"))),
                ngp=float(c.get("no_table_gate_pass_rate", float("nan"))),
                hpm=float(c.get("hotspot_pre0_linear_mean", float("nan"))),
                hps=float(c.get("hotspot_pre0_linear_std", float("nan"))),
            )
        )
    lines.append("")
    lines.append("## H2a gate")
    lines.append("")
    lines.append(f"- decision: `{gate.get('decision', 'unknown')}`")
    lines.append(f"- passed_cases: `{gate.get('passed_cases', [])}`")
    if isinstance(gate.get("thresholds", {}), dict):
        lines.append(f"- thresholds: `{gate.get('thresholds')}`")
    checks = gate.get("checks", {}) if isinstance(gate, dict) else {}
    if isinstance(checks, dict):
        for k in gate.get("candidate_cases", []):
            ck = checks.get(k)
            if isinstance(ck, dict):
                lines.append(f"- {k}: `{ck}`")
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run H2a minimal matrix (adapter / residual bypass).")
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument(
        "--base-config",
        type=str,
        default="debug_output/h1_10p3a_20260213/posttrain_base_e3_lr3e4.json",
        help="Step7 best-branch base config (e3, lr=3e-4).",
    )
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--date-tag", type=str, default="20260213")
    ap.add_argument("--cases", type=str, default="C0,H1,H2")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--step7-case", type=str, default="r2", choices=("r2", "r3", "base"))
    ap.add_argument("--control-case", type=str, default="C0")
    ap.add_argument("--gate-no-table-sign-mean-min", type=float, default=0.8)
    ap.add_argument("--gate-no-table-sign-min-min", type=float, default=0.8)
    ap.add_argument("--gate-no-table-med-mean-max", type=float, default=1.0)
    ap.add_argument("--gate-no-table-med-max-max", type=float, default=1.0)
    ap.add_argument("--gate-no-table-pass-rate-min", type=float, default=1.0)
    ap.add_argument("--gate-table-sign-gain-max", type=float, default=0.05)
    ap.add_argument("--gate-table-med-gain-max", type=float, default=0.10)
    ap.add_argument("--skip-train-if-ckpt-exists", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    base_config = Path(args.base_config).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    teacher = repo_root / "validate/teacher_batches/Walk_F_teacher.json"
    alpha_tbl = repo_root / "debug_output/h1_10p3a_20260213/p0_hotspot_table_20260213/tables/alpha_table_calf_r_6sic.json"
    sign_tbl = repo_root / "debug_output/h1_10p3a_20260213/p0_hotspot_table_20260213/tables/sign_table_calf_r_6sic.json"

    requested_cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    requested_seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    phase1_overrides = dict(STEP7_CASE_OVERRIDES.get(str(args.step7_case), {}))

    for case_key in requested_cases:
        if case_key not in CASES:
            raise SystemExit(f"[FATAL] Unknown case: {case_key}")
    if str(args.control_case) not in requested_cases:
        raise SystemExit(f"[FATAL] control-case must be in --cases, got control={args.control_case} cases={requested_cases}")

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for case_key in requested_cases:
        case = CASES[case_key]
        for seed in requested_seeds:
            run_name = f"h1_10p3a_phase2b_{case.key.lower()}_seed{seed}_e3_lr3e4_{args.date_tag}"
            ckpt_path = repo_root / f"models/MLPL2_DirectBranch_v1/ckpt_last_{run_name}.pth"
            case_dir = out_root / f"{case.key}_seed{seed}"
            case_dir.mkdir(parents=True, exist_ok=True)

            payload = json.loads(base_config.read_text())
            payload.update(phase1_overrides)
            payload.update(case.config_overrides)
            effective_config = case_dir / "config_effective.json"
            effective_config.write_text(json.dumps(payload, indent=2))

            try:
                if (not args.skip_train_if_ckpt_exists) or (not ckpt_path.is_file()):
                    train_cmd = [
                        "python",
                        "-m",
                        "train.posttrain",
                        "--config",
                        str(effective_config),
                        "--run_name",
                        run_name,
                        "--seed",
                        str(seed),
                        "--direct_pose_loss_sics",
                        "9-14,39-42",
                        "--direct_pose_loss_sic_mode",
                        "boost",
                        "--direct_pose_loss_sic_boost",
                        "10",
                    ]
                    cp = _run(train_cmd, cwd=repo_root)
                    (case_dir / "train.log").write_text(cp.stdout)
                if not ckpt_path.is_file():
                    raise RuntimeError(f"Missing checkpoint after training: {ckpt_path}")

                # A) fixed-table eval
                fixed_root = case_dir / "fixed_table"
                fixed_freerun = fixed_root / "freerun"
                fixed_gate = fixed_root / "gate"
                fixed_root.mkdir(parents=True, exist_ok=True)
                cp = _run(
                    [
                        "python",
                        "-m",
                        "train.validate.run_freerun_cycles",
                        "--teacher",
                        str(teacher),
                        "--model",
                        str(ckpt_path),
                        "--out",
                        str(fixed_freerun),
                        "--rounds",
                        "5",
                        "--freerun_x_gt",
                        "--multicycle-reset-plan-z-on-cycle-start",
                        "--export_joint_so3_error_series",
                        "--joint_so3_error_series_branches",
                        "direct",
                        "--joint_so3_error_series_space",
                        "body",
                        "--direct_pose_leg_alpha_table_json",
                        str(alpha_tbl),
                        "--direct_pose_leg_sign_table_json",
                        str(sign_tbl),
                        "--direct_pose_leg_alpha_table_cycle_gte",
                        "1",
                        "--direct_pose_leg_sign_table_cycle_gte",
                        "1",
                        "--direct_pose_leg_alpha_table_drop_wrap",
                        "on",
                        "--direct_pose_leg_sign_table_drop_wrap",
                        "on",
                        "--force",
                    ],
                    cwd=repo_root,
                )
                (fixed_root / "freerun.log").write_text(cp.stdout)
                cp = _run(
                    [
                        "python",
                        "tools/run_h1_10p3a_gate.py",
                        "--model",
                        str(ckpt_path),
                        "--teacher",
                        str(teacher),
                        "--proxy-freerun",
                        str(fixed_freerun / "Walk_F_freerun_cycles.json"),
                        "--out-dir",
                        str(fixed_gate),
                    ],
                    cwd=repo_root,
                )
                (fixed_root / "gate.log").write_text(cp.stdout)

                # B) no-table eval + probe export
                nt_root = case_dir / "no_table"
                nt_freerun = nt_root / "freerun"
                nt_gate = nt_root / "gate"
                nt_root.mkdir(parents=True, exist_ok=True)
                cp = _run(
                    [
                        "python",
                        "-m",
                        "train.validate.run_freerun_cycles",
                        "--teacher",
                        str(teacher),
                        "--model",
                        str(ckpt_path),
                        "--out",
                        str(nt_freerun),
                        "--rounds",
                        "5",
                        "--freerun_x_gt",
                        "--multicycle-reset-plan-z-on-cycle-start",
                        "--export_joint_so3_error_series",
                        "--joint_so3_error_series_branches",
                        "direct",
                        "--joint_so3_error_series_space",
                        "body",
                        "--export_direct_leg_head_io",
                        "--export_direct_leg_omega_alpha_sweep",
                        "--direct_leg_omega_alpha_sweep_sics",
                        "9,10,11,12,13,14,25,26,27,28,39,40,41,42",
                        "--direct_leg_omega_alpha_sweep_bones",
                        "calf_r",
                        "--force",
                    ],
                    cwd=repo_root,
                )
                (nt_root / "freerun.log").write_text(cp.stdout)
                cp = _run(
                    [
                        "python",
                        "tools/run_h1_10p3a_gate.py",
                        "--model",
                        str(ckpt_path),
                        "--teacher",
                        str(teacher),
                        "--proxy-freerun",
                        str(nt_freerun / "Walk_F_freerun_cycles.json"),
                        "--out-dir",
                        str(nt_gate),
                    ],
                    cwd=repo_root,
                )
                (nt_root / "gate.log").write_text(cp.stdout)

                # C) probe diagnostics (from no-table export)
                fr_json = nt_freerun / "Walk_F_freerun_cycles.json"
                probe_hot_pre0_lin = _run_probe(
                    repo_root=repo_root,
                    json_path=fr_json,
                    feature="baseline.pre0",
                    model_kind="linear",
                    sics="9-14,39-42",
                    seed=seed,
                )
                probe_hot_pre0_mlp = _run_probe(
                    repo_root=repo_root,
                    json_path=fr_json,
                    feature="baseline.pre0",
                    model_kind="mlp",
                    sics="9-14,39-42",
                    seed=seed,
                )
                probe_hot_in_lin = _run_probe(
                    repo_root=repo_root,
                    json_path=fr_json,
                    feature="baseline.in",
                    model_kind="linear",
                    sics="9-14,39-42",
                    seed=seed,
                )

                fixed_metrics = _load_gate_metrics(fixed_gate / "h1_10p3a_gate_summary.json")
                nt_metrics = _load_gate_metrics(nt_gate / "h1_10p3a_gate_summary.json")

                row: Dict[str, Any] = {
                    "case": case.key,
                    "case_desc": case.desc,
                    "seed": int(seed),
                    "run_name": run_name,
                    "checkpoint": str(ckpt_path),
                    "overrides": dict(case.config_overrides),
                    "leg_head": {
                        "variant": payload.get("direct_pose_leg_head_variant"),
                        "hidden": payload.get("direct_pose_leg_head_hidden"),
                        "alpha_init": payload.get("direct_pose_leg_head_res_alpha_init"),
                    },
                    "direct_pose": {
                        "feat_source": payload.get("direct_pose_feat_source"),
                        "reinit": bool(payload.get("direct_pose_reinit", False)),
                    },
                    "fixed_table": fixed_metrics,
                    "no_table": nt_metrics,
                    "probe": {
                        "hotspot": {
                            "pre0_linear": probe_hot_pre0_lin["mean"],
                            "pre0_mlp": probe_hot_pre0_mlp["mean"],
                            "in_linear": probe_hot_in_lin["mean"],
                        }
                    },
                }
                rows.append(row)
                (case_dir / "run_result.json").write_text(json.dumps(row, indent=2))

            except subprocess.CalledProcessError as exc:
                fail = {
                    "case": case.key,
                    "seed": int(seed),
                    "run_name": run_name,
                    "error": f"Command failed ({exc.returncode})",
                    "output_tail": (exc.stdout or "")[-4000:],
                }
                failures.append(fail)
                (case_dir / "failure.json").write_text(json.dumps(fail, indent=2))
            except Exception as exc:  # noqa: BLE001
                fail = {
                    "case": case.key,
                    "seed": int(seed),
                    "run_name": run_name,
                    "error": str(exc),
                }
                failures.append(fail)
                (case_dir / "failure.json").write_text(json.dumps(fail, indent=2))

    case_stats = _aggregate_by_case(rows, requested_cases)
    candidate_cases = [k for k in requested_cases if k != str(args.control_case)]
    phase2b_gate = _phase2b_gate(
        case_stats,
        candidate_cases=candidate_cases,
        no_table_sign_mean_min=float(args.gate_no_table_sign_mean_min),
        no_table_sign_min_min=float(args.gate_no_table_sign_min_min),
        no_table_med_mean_max=float(args.gate_no_table_med_mean_max),
        no_table_med_max_max=float(args.gate_no_table_med_max_max),
        no_table_gate_pass_rate_min=float(args.gate_no_table_pass_rate_min),
        table_sign_gain_max=float(args.gate_table_sign_gain_max),
        table_med_gain_max_deg=float(args.gate_table_med_gain_max),
    )

    summary = {
        "rows": rows,
        "failures": failures,
        "case_stats": case_stats,
        "phase2b_gate": phase2b_gate,
        "config": {
            "base_config": str(base_config),
            "teacher": str(teacher),
            "alpha_table": str(alpha_tbl),
            "sign_table": str(sign_tbl),
            "cases": requested_cases,
            "control_case": str(args.control_case),
            "seeds": requested_seeds,
            "step7_case": str(args.step7_case),
            "step7_overrides": phase1_overrides,
            "gate_thresholds_cli": {
                "no_table_sign_mean_min": float(args.gate_no_table_sign_mean_min),
                "no_table_sign_min_min": float(args.gate_no_table_sign_min_min),
                "no_table_med_mean_max": float(args.gate_no_table_med_mean_max),
                "no_table_med_max_max": float(args.gate_no_table_med_max_max),
                "no_table_pass_rate_min": float(args.gate_no_table_pass_rate_min),
                "table_sign_gain_max": float(args.gate_table_sign_gain_max),
                "table_med_gain_max": float(args.gate_table_med_gain_max),
            },
        },
    }

    out_json = out_root / "phase2b_h2a_minimal_summary.json"
    out_md = out_root / "phase2b_h2a_minimal_summary.md"
    out_json.write_text(json.dumps(summary, indent=2))
    _write_md(summary, out_md, requested_cases)
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")
    print(f"[Done] rows={len(rows)} failures={len(failures)}")


if __name__ == "__main__":
    main()
