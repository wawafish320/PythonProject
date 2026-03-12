#!/usr/bin/env python3
"""
Sweep direct_pose_hinge_sup_delta_weight_scale_deg (direction A), starting from
a fixed baseline ckpt, then run strict APPLY/NOAPPLY evaluation and compare.

This matches the "hard subset" definition you described:
  - branch=direct, bone=calf_r, axis=z
  - step_in_cycle in [49, 86] (inclusive)
  - contact: GT contact_idx=1, contact_value=0, contact_thresh=0.5 (swing)
  - report fixed-tail stats: APPLY@fixed_tail(NOAPPLY) with angle_thresh=20deg

Outputs:
  - Per-scale logs under: debug_output/_sweep_hinge_sup_scale_deg/<tag>/scale_<s>/
  - Freerun outputs under: debug_output/<tag>_{noapply,apply}_logc_r<rounds>/
  - Summary CSV/JSON under the out root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _resolve_under(root: Path, p: str) -> Path:
    pp = Path(os.path.expanduser(str(p)))
    if pp.is_absolute():
        return pp
    return (root / pp).resolve()


def _parse_csv_ints(spec: str) -> List[int]:
    s = (spec or "").strip()
    if not s:
        return []
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _cmd_str(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _run_cmd(
    *,
    cmd: Sequence[str],
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    print(f"$ {_cmd_str(cmd)}")
    if dry_run:
        log_path.write_text("[DRY_RUN] " + _cmd_str(cmd) + "\n", encoding="utf-8")
        return
    _ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        ret = p.wait()
    if ret != 0:
        raise RuntimeError(f"command failed (exit={ret}): {_cmd_str(cmd)} (see {log_path})")


def _split_md_row(line: str) -> List[str]:
    # "| a | b | c |" -> ["a", "b", "c"]
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return []
    parts = [p.strip() for p in s.strip("|").split("|")]
    return parts


def _to_float(x: str) -> Optional[float]:
    s = (x or "").strip()
    if not s or s.upper() == "NA":
        return None
    try:
        return float(s)
    except Exception:
        return None


@dataclass
class ParsedMetrics:
    fixed_tail_mean_ang: Optional[float] = None
    fixed_tail_p_gt: Optional[float] = None
    fixed_tail_n_tail: Optional[int] = None
    hinge_abs_delta_mean: Optional[float] = None


def _parse_compare_output(text: str, *, bone: str) -> ParsedMetrics:
    out = ParsedMetrics()
    lines = text.splitlines()

    # 1) Main table: APPLY@fixed_tail(NOAPPLY)
    for ln in lines:
        if f"| {bone} | APPLY@fixed_tail(NOAPPLY) |" not in ln:
            continue
        cols = _split_md_row(ln)
        # Bone | Run | n | mean_ang | P(ang>th) | n_tail | ...
        if len(cols) < 6:
            continue
        out.fixed_tail_mean_ang = _to_float(cols[3])
        out.fixed_tail_p_gt = _to_float(cols[4])
        try:
            out.fixed_tail_n_tail = int(float(cols[5])) if cols[5].strip().upper() != "NA" else None
        except Exception:
            out.fixed_tail_n_tail = None
        break

    # 2) HingeSeries table: abs(delta)_mean (APPLY-only). Parse only after marker.
    hs_start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("[HingeSeries]"):
            hs_start = i
            break
    if hs_start is not None:
        for ln in lines[hs_start:]:
            if not ln.strip().startswith("|"):
                continue
            if f"| {bone} |" not in ln:
                continue
            cols = _split_md_row(ln)
            # Bone | delta_swing_mean | delta_swing_std | delta_stance_mean | delta_stance_std | abs(delta)_mean | ...
            if len(cols) >= 6 and cols[0] == bone:
                out.hinge_abs_delta_mean = _to_float(cols[5])
                break

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep hinge sup delta weight scale (deg) and compare APPLY vs NOAPPLY.")
    ap.add_argument(
        "--config",
        type=str,
        default="config/posttrain_direct_pose_WalkF_only_hinge_calfr_z90_basefeat_rot6d_overfit_rc5_e5_spe60_deltaw2_s30.json",
    )
    ap.add_argument(
        "--ckpt-in",
        type=str,
        default="models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_direct_pose_WalkF_only_hinge_calfr_z90_basefeat_rot6d_overfit_rc5_e5_spe60_deltaw2_s10_denomfix_20260121.pth",
        help="Baseline ckpt to continue posttrain from.",
    )
    ap.add_argument("--scales", type=str, default="5,3,2,1", help="CSV ints, e.g. 5,3,2,1")
    ap.add_argument("--date", type=str, default=time.strftime("%Y%m%d"), help="Tag date, e.g. 20260121")
    ap.add_argument(
        "--run-name-tpl",
        type=str,
        default="WalkF_hinge_calfr_z90_deltaw2_s{scale}_denomfix_{date}",
        help="Python format template for --run_name and debug_output tags.",
    )

    # Training fixed args (direction A)
    ap.add_argument("--kind", type=str, default="smooth_l1")
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--max", type=float, default=10.0)

    # Freerun/eval fixed args (strict phase)
    ap.add_argument("--teacher", type=str, default="validate/teacher_batches/Walk_F_teacher.json")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--bone", type=str, default="calf_r")
    ap.add_argument("--axis", type=str, default="z")
    ap.add_argument("--hinge-max-deg", type=float, default=90.0)
    ap.add_argument("--phase-min", type=int, default=49)
    ap.add_argument("--phase-max", type=int, default=86)
    ap.add_argument("--angle-thresh", type=float, default=20.0)
    ap.add_argument("--contact-index", type=int, default=1)
    ap.add_argument("--contact-value", type=int, default=0)
    ap.add_argument("--contact-thresh", type=float, default=0.5)

    ap.add_argument(
        "--out-root",
        type=str,
        default="debug_output/_sweep_hinge_sup_scale_deg",
        help="Where to write per-scale logs and summary.",
    )
    ap.add_argument(
        "--resume",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Skip steps whose outputs already exist (ckpt/freerun/compare).",
    )
    ap.add_argument("--force", action="store_true", help="Freerun: pass --force to overwrite outputs.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands, do not execute.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    cfg_path = _resolve_under(project_root, args.config)
    cfg = _load_json(cfg_path)
    out_dir = _resolve_under(project_root, str(cfg.get("out_dir", "models/MLPL2_DirectBranch_v1")))

    base_ckpt = _resolve_under(project_root, args.ckpt_in)
    if not base_ckpt.is_file() and not bool(args.dry_run):
        raise SystemExit(f"[FATAL] ckpt_in not found: {base_ckpt}")

    scales = _parse_csv_ints(args.scales)
    if not scales:
        raise SystemExit("[FATAL] empty --scales")

    out_root = _resolve_under(project_root, args.out_root) / f"{args.date}"
    _ensure_dir(out_root)

    # Ensure PYTHONPATH contains "." for module execution.
    env = dict(os.environ)
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    meta = {
        "script": str(Path(__file__).name),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(cfg_path),
        "out_dir": str(out_dir),
        "ckpt_in": str(base_ckpt),
        "scales": scales,
        "run_name_tpl": str(args.run_name_tpl),
        "train_fixed": {"kind": str(args.kind), "power": float(args.power), "max": float(args.max)},
        "eval_fixed": {
            "teacher": str(args.teacher),
            "rounds": int(args.rounds),
            "bone": str(args.bone),
            "axis": str(args.axis),
            "hinge_max_deg": float(args.hinge_max_deg),
            "phase_min": int(args.phase_min),
            "phase_max": int(args.phase_max),
            "angle_thresh": float(args.angle_thresh),
            "contact_index": int(args.contact_index),
            "contact_value": int(args.contact_value),
            "contact_thresh": float(args.contact_thresh),
        },
    }
    _write_json(out_root / "sweep_meta.json", meta)

    rows: List[Dict[str, Any]] = []
    for scale in scales:
        tag = str(args.run_name_tpl).format(scale=int(scale), date=str(args.date))
        run_name = tag
        ckpt_out = out_dir / f"ckpt_last_{run_name}.pth"

        exp_root = out_root / f"scale_{int(scale)}"
        _ensure_dir(exp_root)

        noapply_out = _resolve_under(project_root, f"debug_output/{tag}_noapply_logc_r{int(args.rounds)}")
        apply_out = _resolve_under(project_root, f"debug_output/{tag}_apply_logc_r{int(args.rounds)}")
        compare_md = exp_root / "compare.md"

        row: Dict[str, Any] = {
            "scale_deg": int(scale),
            "tag": tag,
            "run_name": run_name,
            "ckpt_out": str(ckpt_out),
            "noapply_out": str(noapply_out),
            "apply_out": str(apply_out),
            "status": "init",
            "metrics": {},
            "error": None,
        }

        try:
            # ---- posttrain
            post_cmd = [
                "python",
                "-m",
                "train.posttrain",
                "--config",
                str(cfg_path),
                "--ckpt_in",
                str(base_ckpt),
                "--run_name",
                str(run_name),
                "--direct_pose_hinge_sup_kind",
                str(args.kind),
                "--direct_pose_hinge_sup_delta_weight_power",
                str(float(args.power)),
                "--direct_pose_hinge_sup_delta_weight_scale_deg",
                str(float(scale)),
                "--direct_pose_hinge_sup_delta_weight_max",
                str(float(args.max)),
            ]
            post_log = exp_root / "posttrain.log"
            if not bool(args.resume and ckpt_out.is_file()):
                _run_cmd(cmd=post_cmd, cwd=project_root, env=env, log_path=post_log, dry_run=bool(args.dry_run))
            else:
                post_log.write_text("[RESUME] skipped posttrain (ckpt exists)\n", encoding="utf-8")

            # ---- freerun (NOAPPLY)
            noapply_cmd = [
                "python",
                "train/validate/run_freerun_cycles.py",
                "--teacher",
                str(_resolve_under(project_root, args.teacher)),
                "--model",
                str(ckpt_out),
                "--rounds",
                str(int(args.rounds)),
                "--out",
                str(noapply_out),
                "--log_contacts",
                "--export_keybone_omega",
                "--export_keybone_omega_series",
                "--keybone_omega_series_bones",
                str(args.bone),
                "--keybone_omega_series_axis",
                str(args.axis),
            ]
            if bool(args.force):
                noapply_cmd.append("--force")
            noapply_log = exp_root / "freerun_noapply.log"
            if not bool(args.resume and noapply_out.is_dir() and list(noapply_out.glob("*_freerun_cycles.json"))):
                _run_cmd(cmd=noapply_cmd, cwd=project_root, env=env, log_path=noapply_log, dry_run=bool(args.dry_run))
            else:
                noapply_log.write_text("[RESUME] skipped freerun NOAPPLY (json exists)\n", encoding="utf-8")

            # ---- freerun (APPLY)
            apply_cmd = [
                "python",
                "train/validate/run_freerun_cycles.py",
                "--teacher",
                str(_resolve_under(project_root, args.teacher)),
                "--model",
                str(ckpt_out),
                "--rounds",
                str(int(args.rounds)),
                "--out",
                str(apply_out),
                "--log_contacts",
                "--export_keybone_omega",
                "--export_keybone_omega_series",
                "--keybone_omega_series_bones",
                str(args.bone),
                "--keybone_omega_series_axis",
                str(args.axis),
                "--export_direct_hinge_series",
                "--direct_pose_hinge_enable",
                "--direct_pose_hinge_bones",
                str(args.bone),
                "--direct_pose_hinge_axis",
                str(args.axis),
                "--direct_pose_hinge_max_deg",
                str(float(args.hinge_max_deg)),
            ]
            if bool(args.force):
                apply_cmd.append("--force")
            apply_log = exp_root / "freerun_apply.log"
            if not bool(args.resume and apply_out.is_dir() and list(apply_out.glob("*_freerun_cycles.json"))):
                _run_cmd(cmd=apply_cmd, cwd=project_root, env=env, log_path=apply_log, dry_run=bool(args.dry_run))
            else:
                apply_log.write_text("[RESUME] skipped freerun APPLY (json exists)\n", encoding="utf-8")

            # ---- compare
            compare_cmd = [
                "python",
                "tools/compare_hinge_apply_noapply.py",
                "--noapply",
                str(noapply_out),
                "--apply",
                str(apply_out),
                "--bones",
                str(args.bone),
                "--branch",
                "direct",
                "--min-cycle",
                "1",
                "--phase-min",
                str(int(args.phase_min)),
                "--phase-max",
                str(int(args.phase_max)),
                "--contact-source",
                "gt",
                "--contact-index",
                str(int(args.contact_index)),
                "--contact-value",
                str(int(args.contact_value)),
                "--contact-thresh",
                str(float(args.contact_thresh)),
                "--angle-thresh",
                str(float(args.angle_thresh)),
                "--report-hinge-series",
                "--strict-clips",
            ]
            if not bool(args.resume and compare_md.is_file()):
                print(f"$ {_cmd_str(compare_cmd)}")
                if bool(args.dry_run):
                    compare_md.write_text("[DRY_RUN] " + _cmd_str(compare_cmd) + "\n", encoding="utf-8")
                    cmp_out = ""
                else:
                    cp = subprocess.run(
                        list(compare_cmd),
                        cwd=str(project_root),
                        env=env,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    cmp_out = str(cp.stdout or "")
                    compare_md.write_text(cmp_out, encoding="utf-8")
            else:
                cmp_out = compare_md.read_text(encoding="utf-8")

            pm = _parse_compare_output(cmp_out, bone=str(args.bone))
            row["metrics"] = {
                "APPLY_fixed_tail_mean_ang": pm.fixed_tail_mean_ang,
                "APPLY_fixed_tail_P_gt": pm.fixed_tail_p_gt,
                "APPLY_fixed_tail_n_tail": pm.fixed_tail_n_tail,
                "HingeSeries_abs_delta_mean": pm.hinge_abs_delta_mean,
            }
            row["status"] = "ok" if not bool(args.dry_run) else "dry_run"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)

        rows.append(row)
        print(f"[scale={int(scale)}] status={row['status']}  tag={tag}")

    summary = {"meta": meta, "rows": rows}
    _write_json(out_root / "sweep_summary.json", summary)

    csv_path = out_root / "sweep_results.csv"
    fields = [
        "scale_deg",
        "tag",
        "status",
        "ckpt_out",
        "APPLY_fixed_tail_mean_ang",
        "APPLY_fixed_tail_P_gt",
        "APPLY_fixed_tail_n_tail",
        "HingeSeries_abs_delta_mean",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            m = r.get("metrics") or {}
            w.writerow(
                {
                    "scale_deg": r.get("scale_deg"),
                    "tag": r.get("tag"),
                    "status": r.get("status"),
                    "ckpt_out": r.get("ckpt_out"),
                    "APPLY_fixed_tail_mean_ang": m.get("APPLY_fixed_tail_mean_ang"),
                    "APPLY_fixed_tail_P_gt": m.get("APPLY_fixed_tail_P_gt"),
                    "APPLY_fixed_tail_n_tail": m.get("APPLY_fixed_tail_n_tail"),
                    "HingeSeries_abs_delta_mean": m.get("HingeSeries_abs_delta_mean"),
                }
            )

    print(f"[DONE] wrote: {csv_path}")


if __name__ == "__main__":
    main()
