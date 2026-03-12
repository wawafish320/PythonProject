#!/usr/bin/env python3
"""
Run a small A/B sweep for the SO(3) delta-rotation / angular-velocity loss and
produce the Stage7 freerun dt_frames report artifacts.

This follows:
  docs/Problems/active/2026-02-11_WalkF_stage7_phase_lag_velocity_loss.md

Pipeline per w_rot_vel:
  1) Train:   python -m train.training_MPL (config_json + config_override)
  2) Freerun: python -m train.validate.run_freerun_cycles (export_joint_so3_error_series)
  3) Report:  python tools/report_sic_hotspots_vs_gt_angvel.py (writes md/png)

Outputs:
  - debug_output/<sweep_dir>/<run_name>/
      - train.log
      - freerun.log
      - sic_vs_omega.md
      - sic_vs_omega.png
      - Walk_F_freerun_cycles.json
  - debug_output/<sweep_dir>/summary.json + summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float_tag(x: float) -> str:
    # File/dir friendly: 2.5 -> "2p5", 0.0 -> "0", -1.25 -> "m1p25"
    s = f"{float(x):.6g}"
    s = s.replace("-", "m").replace(".", "p")
    s = re.sub(r"p0$", "", s)
    return s


def _run_and_tee(cmd: List[str], *, cwd: Path, env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd]\n")
        f.write(" ".join(cmd) + "\n\n")
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
    if rc != 0:
        raise SystemExit(f"[FATAL] command failed (exit={rc}): {' '.join(cmd)} (log: {log_path})")


@dataclass
class QuickJoint:
    name: str
    align_frac: Optional[float]
    n_mu: int
    dt_median: Optional[float]
    dt_p25: Optional[float]
    dt_p75: Optional[float]
    n_dt: int


def _parse_quick_from_md(md_path: Path) -> Dict[str, QuickJoint]:
    """
    Parse the "## Quick summary" block emitted by tools/report_sic_hotspots_vs_gt_angvel.py.
    This is intentionally brittle-but-simple; if the report format changes, just re-run
    the report script and inspect the generated md.
    """
    txt = md_path.read_text(encoding="utf-8")
    lines = txt.splitlines()
    in_quick = False
    cur: Optional[str] = None
    out: Dict[str, QuickJoint] = {}

    def _ensure(name: str) -> QuickJoint:
        if name not in out:
            out[name] = QuickJoint(
                name=name,
                align_frac=None,
                n_mu=0,
                dt_median=None,
                dt_p25=None,
                dt_p75=None,
                n_dt=0,
            )
        return out[name]

    # Example lines (from the report script):
    #   - sign(mu_z * omega_z) > 0 fraction = 0.465 (N_mu=86)
    #   - dt_frames = (mu_z/omega_z) * FPS: median=1.376, IQR=[-4.539, 9.313] (N_dt=62)
    frac_re = re.compile(r"fraction\s*=\s*([0-9.+-eE]+)\s*\(N_mu=(\d+)\)")
    dt_re = re.compile(
        r"median=([0-9.+-eE]+),\s*IQR=\[([0-9.+-eE]+),\s*([0-9.+-eE]+)\]\s*\(N_dt=(\d+)\)"
    )

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("## Quick summary"):
            in_quick = True
            continue
        if not in_quick:
            continue
        if line.startswith("Interpretation note:"):
            break
        if line.startswith("- ") and line.endswith(":"):
            cur = line[2:-1].strip()
            _ensure(cur)
            continue
        if cur is None:
            continue
        if "fraction" in line and "N_mu=" in line:
            m = frac_re.search(line)
            if m:
                q = _ensure(cur)
                q.align_frac = float(m.group(1))
                q.n_mu = int(m.group(2))
            continue
        if "dt_frames" in line:
            q = _ensure(cur)
            if "NA" in line:
                q.dt_median = None
                q.dt_p25 = None
                q.dt_p75 = None
                q.n_dt = 0
            else:
                m = dt_re.search(line)
                if m:
                    q.dt_median = float(m.group(1))
                    q.dt_p25 = float(m.group(2))
                    q.dt_p75 = float(m.group(3))
                    q.n_dt = int(m.group(4))
            continue

    return out


def _compute_direct_geolocal_tail(
    freerun_json: Path,
    *,
    min_cycle: int = 1,
    exclude_wrap: bool = True,
    window_sics: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute simple tail stats for DirectGeoLocalDeg from a freerun_cycles.json.

    Mirrors doc 7.2 in:
      docs/Problems/active/2026-02-11_WalkF_stage7_phase_lag_velocity_loss.md
    """
    if window_sics is None:
        window_sics = [14, 15] + list(range(49, 56))
    obj = _load_json(freerun_json)
    steps = obj.get("metrics_per_step", [])
    vals: List[float] = []
    vals_win: List[float] = []
    for s in steps:
        cy = int(s.get("cycle", 0) or 0)
        if cy < int(min_cycle):
            continue
        if exclude_wrap and bool(s.get("wrap_boundary_step", False)):
            continue
        v = s.get("DirectGeoLocalDeg", None)
        if v is None:
            continue
        try:
            vv = float(v)
        except Exception:
            continue
        if not np.isfinite(vv):
            continue
        vals.append(vv)
        sic = s.get("step_in_cycle", None)
        sic_i = int(sic) if isinstance(sic, int) else -1
        if sic_i in set(int(x) for x in window_sics):
            vals_win.append(vv)

    arr = np.asarray(vals, dtype=np.float64)
    win = np.asarray(vals_win, dtype=np.float64)

    out: Dict[str, Any] = {
        "metric": "DirectGeoLocalDeg",
        "min_cycle": int(min_cycle),
        "exclude_wrap": bool(exclude_wrap),
        "N": int(arr.size),
        "max": float(np.max(arr)) if arr.size else float("nan"),
        "p99": float(np.percentile(arr, 99)) if arr.size else float("nan"),
        "window_sics": [int(x) for x in window_sics],
        "window_N": int(win.size),
        "window_mean": float(np.mean(win)) if win.size else float("nan"),
        "window_p90": float(np.percentile(win, 90)) if win.size else float("nan"),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run rot-vel sweep (train + freerun + dt_frames report).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--config-json",
        type=str,
        default="config/exp_phase_DirectBranch_v1_d1_noreset.json",
        help="Training config JSON (used as defaults; run_name/out can be overridden).",
    )
    ap.add_argument("--out", type=str, default=None, help="Override training --out (otherwise use config_json out).")
    ap.add_argument("--base-run-name", type=str, default=None, help="Override base run name (otherwise use config_json run_name).")
    ap.add_argument(
        "--w-rot-vel",
        type=str,
        default="0,2.5,5,10,15",
        help="Comma-separated sweep values for w_rot_vel.",
    )
    ap.add_argument("--rot-vel-log-scale", type=float, default=2.0)
    ap.add_argument("--rot-vel-omega-min-deg-s", type=float, default=30.0)
    ap.add_argument("--rot-vel-loss", type=str, default="smooth_l1", choices=("smooth_l1", "mse"))
    ap.add_argument("--aug-lr-swap-prob", type=float, default=0.0, help="Training-time L/R swap augmentation prob.")
    ap.add_argument("--w-direct-delta-sym", type=float, default=0.0, help="Soft L/R symmetry regularizer on direct_delta.")
    ap.add_argument(
        "--direct-delta-sym-mirror-xyz",
        type=str,
        default="1,1,1",
        help="Mirror xyz signs/scales for direct-delta symmetry term (comma-separated).",
    )
    ap.add_argument("--epochs", type=int, default=None, help="Optional training epochs override.")
    ap.add_argument(
        "--train-config-override",
        action="append",
        default=None,
        help="Extra KEY=VALUE overrides forwarded to train.training_MPL (repeatable).",
    )

    # Freerun + report inputs (Walk_F defaults).
    ap.add_argument("--teacher", type=str, default="validate/teacher_batches/Walk_F_teacher.json")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--npz", type=str, default="raw_data/processed_data/Walk_F.npz")
    ap.add_argument("--depth", type=int, default=3)

    # Report knobs.
    ap.add_argument("--report-branch", type=str, default="direct", choices=("inc", "direct", "blend"))
    ap.add_argument("--report-space", type=str, default="body", choices=("body", "world"))
    ap.add_argument("--report-axis", type=str, default="z", choices=("x", "y", "z"))
    ap.add_argument("--report-joints", type=str, default="calf_l,calf_r")
    ap.add_argument("--report-min-cycle", type=int, default=1)
    ap.add_argument("--report-exclude-wrap", action="store_true", default=True)
    ap.add_argument("--report-exclude-root", action="store_true", default=True)

    ap.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Output dir under debug_output. Default: rotvel_sweep_YYYYMMDD.",
    )
    ap.add_argument("--skip-train", action="store_true", help="Assume ckpt exists; skip training.")
    ap.add_argument("--skip-freerun", action="store_true", help="Assume freerun json exists; skip freerun.")
    ap.add_argument("--skip-report", action="store_true", help="Assume md/png exists; skip report.")
    args = ap.parse_args()

    cfg_path = (_ROOT / str(args.config_json)).expanduser()
    cfg = _load_json(cfg_path)
    train_out = Path(args.out or cfg.get("out") or "runs").expanduser()
    base_run = str(args.base_run_name or cfg.get("run_name") or "run")

    w_list: List[float] = []
    for part in str(args.w_rot_vel).split(","):
        part = part.strip()
        if not part:
            continue
        w_list.append(float(part))
    if not w_list:
        raise SystemExit("Empty --w-rot-vel list.")

    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
    else:
        from datetime import date

        sweep_dir = Path("debug_output") / f"rotvel_sweep_{date.today().strftime('%Y%m%d')}"
    sweep_dir = (_ROOT / sweep_dir).expanduser()
    sweep_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(_ROOT))
    # Make matplotlib deterministic + writable inside sandbox.
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

    summary: Dict[str, Any] = {
        "config_json": str(cfg_path),
        "train_out": str(train_out),
        "base_run_name": base_run,
        "rot_vel_log_scale": float(args.rot_vel_log_scale),
        "rot_vel_omega_min_deg_s": float(args.rot_vel_omega_min_deg_s),
        "rot_vel_loss": str(args.rot_vel_loss),
        "aug_lr_swap_prob": float(args.aug_lr_swap_prob),
        "w_direct_delta_sym": float(args.w_direct_delta_sym),
        "direct_delta_sym_mirror_xyz": str(args.direct_delta_sym_mirror_xyz),
        "train_config_overrides": [str(x) for x in (args.train_config_override or [])],
        "epochs_override": int(args.epochs) if args.epochs is not None else None,
        "w_rot_vel_list": [float(w) for w in w_list],
        "runs": [],
    }

    for w in w_list:
        w_tag = _fmt_float_tag(w)
        thr_tag = _fmt_float_tag(float(args.rot_vel_omega_min_deg_s))
        ls_tag = _fmt_float_tag(float(args.rot_vel_log_scale))
        loss_tag = str(args.rot_vel_loss)
        swap_tag = _fmt_float_tag(float(args.aug_lr_swap_prob))
        sym_tag = _fmt_float_tag(float(args.w_direct_delta_sym))
        mir_tag = str(args.direct_delta_sym_mirror_xyz).replace(" ", "").replace(",", "_")
        mir_tag = mir_tag.replace("-", "m").replace(".", "p")
        run_name = f"{base_run}__rotvel_w{w_tag}_lrswap_p{swap_tag}_thr{thr_tag}_ls{ls_tag}_{loss_tag}"
        if float(args.w_direct_delta_sym) != 0.0 or str(args.direct_delta_sym_mirror_xyz).replace(" ", "") != "1,1,1":
            run_name = f"{run_name}_symw{sym_tag}_symm{mir_tag}"

        run_eval_dir = sweep_dir / run_name
        run_eval_dir.mkdir(parents=True, exist_ok=True)

        # ---- Train ----------------------------------------------------------
        ckpt_best_free = train_out / run_name / f"ckpt_best_free_{run_name}.pth"
        ckpt_last = train_out / run_name / f"ckpt_last_{run_name}.pth"
        if not args.skip_train:
            cmd = [
                sys.executable,
                "-m",
                "train.training_MPL",
                "--config_json",
                str(cfg_path),
                "--out",
                str(train_out),
                "--run_name",
                run_name,
                "--config_override",
                f"w_rot_vel={float(w)}",
                "--config_override",
                f"rot_vel_log_scale={float(args.rot_vel_log_scale)}",
                "--config_override",
                f"rot_vel_omega_min_deg_s={float(args.rot_vel_omega_min_deg_s)}",
                "--config_override",
                f"rot_vel_loss={loss_tag}",
                "--config_override",
                f"aug_lr_swap_prob={float(args.aug_lr_swap_prob)}",
                "--config_override",
                f"w_direct_delta_sym={float(args.w_direct_delta_sym)}",
                "--config_override",
                f"direct_delta_sym_mirror_xyz={str(args.direct_delta_sym_mirror_xyz)}",
            ]
            if args.epochs is not None:
                cmd += ["--config_override", f"epochs={int(args.epochs)}"]
            for override in (args.train_config_override or []):
                ov = str(override).strip()
                if ov:
                    cmd += ["--config_override", ov]
            _run_and_tee(cmd, cwd=_ROOT, env=env, log_path=run_eval_dir / "train.log")

        ckpt_path = ckpt_best_free if ckpt_best_free.is_file() else ckpt_last
        if not ckpt_path.is_file():
            raise SystemExit(f"[FATAL] missing ckpt for run={run_name}: expected {ckpt_best_free} or {ckpt_last}")

        # ---- Freerun --------------------------------------------------------
        freerun_json = run_eval_dir / "Walk_F_freerun_cycles.json"
        if not args.skip_freerun:
            cmd = [
                sys.executable,
                "-m",
                "train.validate.run_freerun_cycles",
                "--model",
                str(ckpt_path),
                "--teacher",
                str((_ROOT / str(args.teacher)).expanduser()),
                "--bundle",
                str((_ROOT / str(args.bundle)).expanduser()),
                "--pretrain-template",
                str((_ROOT / str(args.pretrain_template)).expanduser()),
                "--encoder-bundle",
                str((_ROOT / str(args.encoder_bundle)).expanduser()),
                "--npz-root",
                str((_ROOT / str(args.npz_root)).expanduser()),
                "--out",
                str(run_eval_dir),
                "--depth",
                str(int(args.depth)),
                "--export_joint_so3_error_series",
            ]
            _run_and_tee(cmd, cwd=_ROOT, env=env, log_path=run_eval_dir / "freerun.log")

        if not freerun_json.is_file():
            # Fallback: some runs might write under a subdir.
            cands = sorted(run_eval_dir.glob("*_freerun_cycles.json"))
            if len(cands) == 1:
                freerun_json = cands[0]
            else:
                raise SystemExit(f"[FATAL] freerun json not found under {run_eval_dir} (found {len(cands)})")

        # ---- Report ---------------------------------------------------------
        md_path = run_eval_dir / "sic_vs_omega.md"
        fig_path = run_eval_dir / "sic_vs_omega.png"
        if not args.skip_report:
            cmd = [
                sys.executable,
                str((_ROOT / "tools" / "report_sic_hotspots_vs_gt_angvel.py")),
                "--freerun-json",
                str(freerun_json),
                "--npz",
                str((_ROOT / str(args.npz)).expanduser()),
                "--branch",
                str(args.report_branch),
                "--space",
                str(args.report_space),
                "--min-cycle",
                str(int(args.report_min_cycle)),
                "--axis",
                str(args.report_axis),
                "--joints",
                str(args.report_joints),
                "--omega-min-deg-s",
                str(float(args.rot_vel_omega_min_deg_s)),
                "--out-md",
                str(md_path),
                "--out-fig",
                str(fig_path),
            ]
            if bool(args.report_exclude_wrap):
                cmd.append("--exclude-wrap")
            if bool(args.report_exclude_root):
                cmd.append("--exclude-root")
            _run_and_tee(cmd, cwd=_ROOT, env=env, log_path=run_eval_dir / "report.log")

        quick: Dict[str, Any] = {}
        if md_path.is_file():
            try:
                q = _parse_quick_from_md(md_path)
                quick = {k: vars(v) for k, v in q.items()}
            except Exception:
                quick = {}

        tail = {}
        try:
            tail = _compute_direct_geolocal_tail(
                freerun_json,
                min_cycle=int(args.report_min_cycle),
                exclude_wrap=bool(args.report_exclude_wrap),
            )
        except Exception:
            tail = {}

        summary["runs"].append(
            {
                "run_name": run_name,
                "w_rot_vel": float(w),
                "ckpt": str(ckpt_path),
                "freerun_json": str(freerun_json),
                "report_md": str(md_path) if md_path.is_file() else None,
                "report_fig": str(fig_path) if fig_path.is_file() else None,
                "quick": quick,
                "direct_geolocal_tail": tail,
            }
        )

    out_json = sweep_dir / "summary.json"
    out_md = sweep_dir / "summary.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Write a tiny human-friendly summary table.
    lines: List[str] = []
    lines.append("# rot-vel sweep summary")
    lines.append("")
    lines.append(f"- config_json: `{cfg_path}`")
    lines.append(f"- train_out: `{train_out}`")
    lines.append(f"- base_run_name: `{base_run}`")
    lines.append(f"- rot_vel_log_scale: {float(args.rot_vel_log_scale)}")
    lines.append(f"- rot_vel_omega_min_deg_s: {float(args.rot_vel_omega_min_deg_s)}")
    lines.append(f"- rot_vel_loss: `{args.rot_vel_loss}`")
    lines.append(f"- w_direct_delta_sym: {float(args.w_direct_delta_sym)}")
    lines.append(f"- direct_delta_sym_mirror_xyz: `{args.direct_delta_sym_mirror_xyz}`")
    if args.train_config_override:
        lines.append(f"- train_config_override: `{', '.join(str(x) for x in args.train_config_override)}`")
    lines.append("")
    lines.append("|w_rot_vel|run_name|calf_l dt_med|calf_r dt_med|")
    lines.append("|---:|:---|---:|---:|")
    for r in summary["runs"]:
        q = r.get("quick") or {}
        l = q.get("calf_l", {}) if isinstance(q, dict) else {}
        rr = q.get("calf_r", {}) if isinstance(q, dict) else {}
        def _dt(x: Any) -> str:
            v = x.get("dt_median", None) if isinstance(x, dict) else None
            return "NA" if v is None else f"{float(v):.3f}"
        lines.append(
            "|{w:.3f}|`{name}`|{l}|{r}|".format(
                w=float(r.get("w_rot_vel", 0.0)),
                name=str(r.get("run_name", "")),
                l=_dt(l),
                r=_dt(rr),
            )
        )

    # Tail stats (doc 7.2): DirectGeoLocalDeg global + contact-transition window.
    lines.append("")
    lines.append("## DirectGeoLocalDeg Tail (cycle>=1; drop_wrap)")
    lines.append("")
    lines.append("|w_rot_vel|DirectGeoLocalDeg p99|max|win mean (sic 14,15,49-55)|win p90|")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in summary["runs"]:
        t = r.get("direct_geolocal_tail") or {}
        def _f(key: str) -> str:
            v = t.get(key, None)
            if v is None:
                return "NA"
            try:
                return f"{float(v):.3f}"
            except Exception:
                return "NA"
        lines.append(
            "|{w:.3f}|{p99}|{mx}|{wm}|{wp}|".format(
                w=float(r.get("w_rot_vel", 0.0)),
                p99=_f("p99"),
                mx=_f("max"),
                wm=_f("window_mean"),
                wp=_f("window_p90"),
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
