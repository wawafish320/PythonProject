#!/usr/bin/env python3
"""
Run Phase1 pre0 constraint/optimization ablation matrix (R0-R5) and summarize metrics.
"""

from __future__ import annotations

import argparse
import json
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
    extra_args: List[str]
    config_overrides: Dict[str, Any] | None = None


CASES: Dict[str, CaseSpec] = {
    "R0": CaseSpec(key="R0", extra_args=[]),
    "R1": CaseSpec(key="R1", extra_args=["--direct_pose_leg_detach_feat", "false"]),
    "R2": CaseSpec(key="R2", extra_args=["--direct_pose_leg_stopgrad_main", "false"]),
    "R3": CaseSpec(
        key="R3",
        extra_args=[
            "--direct_pose_leg_detach_feat",
            "false",
            "--direct_pose_leg_stopgrad_main",
            "false",
        ],
    ),
    "R4": CaseSpec(
        key="R4",
        extra_args=[],
        config_overrides={"direct_pose_leg_gate_train_only": True},
    ),
    "R5": CaseSpec(key="R5", extra_args=["--direct_pose_leg_train_only", "true"]),
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


def _write_md(summary: Dict[str, Any], out_md: Path) -> None:
    rows = summary.get("rows", [])
    lines: List[str] = []
    lines.append("# p6 pre0 constraint ablation summary")
    lines.append("")
    lines.append("| case | seed | fixed sign | fixed med_abs | no-table sign | no-table med_abs | hotspot pre0 linear | hotspot pre0 mlp | hotspot in linear | control pre0 linear | control pre0 mlp | control in linear |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {case} | {seed} | {fs:.3f} | {fm:.6f} | {ns:.3f} | {nm:.6f} | {hpl:.4f} | {hpm:.4f} | {hil:.4f} | {cpl:.4f} | {cpm:.4f} | {cil:.4f} |".format(
                case=r["case"],
                seed=r["seed"],
                fs=r["fixed_table"]["sign_match_rate"],
                fm=r["fixed_table"]["median_abs_mu_z_diff_deg"],
                ns=r["no_table"]["sign_match_rate"],
                nm=r["no_table"]["median_abs_mu_z_diff_deg"],
                hpl=r["probe"]["hotspot"]["pre0_linear"],
                hpm=r["probe"]["hotspot"]["pre0_mlp"],
                hil=r["probe"]["hotspot"]["in_linear"],
                cpl=r["probe"]["control"]["pre0_linear"],
                cpm=r["probe"]["control"]["pre0_mlp"],
                cil=r["probe"]["control"]["in_linear"],
            )
        )
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run p6 Phase1 constraint ablation matrix.")
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument("--base-config", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--date-tag", type=str, default="20260213")
    ap.add_argument("--cases", type=str, default="R0,R1,R2,R3,R4,R5")
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--skip-train-if-ckpt-exists", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    base_config = Path(args.base_config).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Fixed control resources (locked by user context).
    teacher = repo_root / "validate/teacher_batches/Walk_F_teacher.json"
    alpha_tbl = repo_root / "debug_output/h1_10p3a_20260213/p0_hotspot_table_20260213/tables/alpha_table_calf_r_6sic.json"
    sign_tbl = repo_root / "debug_output/h1_10p3a_20260213/p0_hotspot_table_20260213/tables/sign_table_calf_r_6sic.json"

    requested_cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    requested_seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for case_key in requested_cases:
        if case_key not in CASES:
            raise SystemExit(f"[FATAL] Unknown case: {case_key}")
        case = CASES[case_key]
        for seed in requested_seeds:
            run_name = f"h1_10p3a_p6_{case.key.lower()}_seed{seed}_e1s20_{args.date_tag}"
            ckpt_path = repo_root / f"models/MLPL2_DirectBranch_v1/ckpt_last_{run_name}.pth"
            case_dir = out_root / f"{case.key}_seed{seed}"
            case_dir.mkdir(parents=True, exist_ok=True)
            effective_config = base_config
            if case.config_overrides:
                payload = json.loads(base_config.read_text())
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
                    ] + list(case.extra_args)
                    cp = _run(train_cmd, cwd=repo_root)
                    (case_dir / "train.log").write_text(cp.stdout)
                if not ckpt_path.is_file():
                    raise RuntimeError(f"Missing checkpoint after training: {ckpt_path}")

                # A) fixed table_6sic eval
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

                # Probe diagnostics (from no-table freerun export).
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
                probe_ctl_pre0_lin = _run_probe(
                    repo_root=repo_root,
                    json_path=fr_json,
                    feature="baseline.pre0",
                    model_kind="linear",
                    sics="25-28",
                    seed=seed,
                )
                probe_ctl_pre0_mlp = _run_probe(
                    repo_root=repo_root,
                    json_path=fr_json,
                    feature="baseline.pre0",
                    model_kind="mlp",
                    sics="25-28",
                    seed=seed,
                )
                probe_ctl_in_lin = _run_probe(
                    repo_root=repo_root,
                    json_path=fr_json,
                    feature="baseline.in",
                    model_kind="linear",
                    sics="25-28",
                    seed=seed,
                )

                fixed_metrics = _load_gate_metrics(fixed_gate / "h1_10p3a_gate_summary.json")
                nt_metrics = _load_gate_metrics(nt_gate / "h1_10p3a_gate_summary.json")

                row: Dict[str, Any] = {
                    "case": case.key,
                    "seed": int(seed),
                    "run_name": run_name,
                    "checkpoint": str(ckpt_path),
                    "fixed_table": fixed_metrics,
                    "no_table": nt_metrics,
                    "probe": {
                        "hotspot": {
                            "pre0_linear": probe_hot_pre0_lin["mean"],
                            "pre0_mlp": probe_hot_pre0_mlp["mean"],
                            "in_linear": probe_hot_in_lin["mean"],
                        },
                        "control": {
                            "pre0_linear": probe_ctl_pre0_lin["mean"],
                            "pre0_mlp": probe_ctl_pre0_mlp["mean"],
                            "in_linear": probe_ctl_in_lin["mean"],
                        },
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

    summary = {
        "rows": rows,
        "failures": failures,
        "config": {
            "base_config": str(base_config),
            "teacher": str(teacher),
            "alpha_table": str(alpha_tbl),
            "sign_table": str(sign_tbl),
            "cases": requested_cases,
            "seeds": requested_seeds,
        },
    }

    out_json = out_root / "p6_constraint_ablation_summary.json"
    out_md = out_root / "p6_constraint_ablation_summary.md"
    out_json.write_text(json.dumps(summary, indent=2))
    _write_md(summary, out_md)
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")
    print(f"[Done] rows={len(rows)} failures={len(failures)}")


if __name__ == "__main__":
    main()
