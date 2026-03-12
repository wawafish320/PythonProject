#!/usr/bin/env python3
"""
Formal Stage7 P1 sampler-only A/B:
- multi-seed (>=3)
- multi-epoch (>=3)
- keep model/loss unchanged; only dataset_index_mode differs.

Pipeline per (mode, seed):
1) train.training_MPL (resume from Stage6 ckpt)
2) run_freerun_cycles
3) report_sic_hotspots_vs_gt_angvel.py
4) diagnose_stage7_sampling_grad_closure.py

Outputs:
- debug_output/<sweep_dir>/<run_name>/
  - train.log / freerun.log / report.log / diagnose.log
  - sic_vs_omega.md / sic_vs_omega.png
  - sampling_grad_closure/sampling_grad_closure.json|md
- debug_output/<sweep_dir>/summary.json + summary.md
- debug_output/<sweep_dir>/config_<mode>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _mean_std(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([_safe_float(v) for v in vals], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


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
        cmd_txt = " ".join(cmd)
        raise SystemExit(f"[FATAL] command failed (exit={rc}): {cmd_txt} (log: {log_path})")


def _resolve_from_root(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (_ROOT / p)


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


def _parse_mode_list(spec: str) -> List[str]:
    allowed = {"sliding", "sic_balanced"}
    out: List[str] = []
    seen = set()
    for tok in str(spec or "").split(","):
        s = str(tok).strip().lower()
        if not s:
            continue
        if s not in allowed:
            raise ValueError(f"unsupported mode: {s} (allowed={sorted(allowed)})")
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    if not out:
        raise ValueError("empty mode list")
    return out


def _pick_ckpt(train_out: Path, run_name: str) -> Path:
    run_dir = train_out / run_name
    cands = [
        run_dir / f"ckpt_last_{run_name}.pth",
        run_dir / f"ckpt_best_free_{run_name}.pth",
        run_dir / f"ckpt_best_teacher_{run_name}.pth",
    ]
    for p in cands:
        if p.is_file():
            return p
    tried = ", ".join(str(x) for x in cands)
    raise FileNotFoundError(f"missing checkpoint for run={run_name}; tried: {tried}")


def _find_freerun_json(run_dir: Path, target_clip: str) -> Path:
    stem = f"{target_clip}_freerun_cycles.json"
    cands = [
        run_dir / stem,
        run_dir / stem / stem,
    ]
    for p in cands:
        if p.is_file():
            return p

    files = [p for p in sorted(run_dir.rglob(stem)) if p.is_file()]
    if files:
        files = sorted(files, key=lambda p: (len(p.parts), str(p)))
        return files[0]
    raise FileNotFoundError(f"freerun json not found under {run_dir} (target={stem})")


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

    frac_re = re.compile(r"fraction\s*=\s*([0-9.+-eE]+)\s*\(N_mu=(\d+)\)")
    dt_re = re.compile(r"median=([0-9.+-eE]+),\s*IQR=\[([0-9.+-eE]+),\s*([0-9.+-eE]+)\]\s*\(N_dt=(\d+)\)")

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


def _cv_from_counts(values: Sequence[Any]) -> float:
    arr = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return float("nan")
    return float(np.std(arr) / mean)


def _component_sic_abs_log_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    vals: List[float] = []
    for r in rows:
        lv = _safe_float(r.get("grad_log_ratio_r_over_l", float("nan")))
        if math.isfinite(lv):
            vals.append(abs(float(lv)))
    if not vals:
        return {"n": 0, "mean_abs_log": float("nan"), "p90_abs_log": float("nan")}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_abs_log": float(np.mean(arr)),
        "p90_abs_log": float(np.percentile(arr, 90)),
    }


def _extract_diag_metrics(diag_json: Path) -> Dict[str, Any]:
    payload = _load_json(diag_json)
    dist = payload.get("distribution", {}) if isinstance(payload, dict) else {}
    grad_global = (payload.get("gradient", {}) or {}).get("global", {})
    comp = payload.get("component_gradient", {}) if isinstance(payload, dict) else {}
    dual = payload.get("dual_probe_gradient", {}) if isinstance(payload, dict) else {}
    root_probe = payload.get("root_cause_probe", {}) if isinstance(payload, dict) else {}
    direct_head_sym = root_probe.get("direct_head_symmetry", {}) if isinstance(root_probe, dict) else {}

    comp_global = comp.get("global", {}) if isinstance(comp, dict) else {}
    comp_sic = comp.get("per_sic_rows", {}) if isinstance(comp, dict) else {}
    dual_global = dual.get("global", {}) if isinstance(dual, dict) else {}
    dual_sic = dual.get("per_sic_rows", {}) if isinstance(dual, dict) else {}

    comp_ratio: Dict[str, float] = {}
    comp_sic_stats: Dict[str, Dict[str, float]] = {}
    dual_ratio: Dict[str, float] = {}
    dual_sic_stats: Dict[str, Dict[str, float]] = {}
    for comp_name, comp_payload in comp_global.items():
        if not isinstance(comp_payload, dict):
            continue
        comp_ratio[str(comp_name)] = _safe_float(comp_payload.get("grad_ratio_r_over_l", float("nan")))
    for comp_name, rows in (comp_sic.items() if isinstance(comp_sic, dict) else []):
        if isinstance(rows, list):
            comp_sic_stats[str(comp_name)] = _component_sic_abs_log_stats(rows)
    for comp_name, per_probe in (dual_global.items() if isinstance(dual_global, dict) else []):
        if not isinstance(per_probe, dict):
            continue
        for probe_name, probe_payload in per_probe.items():
            if not isinstance(probe_payload, dict):
                continue
            key = f"{str(comp_name)}@{str(probe_name)}"
            dual_ratio[key] = _safe_float(probe_payload.get("grad_ratio_r_over_l", float("nan")))
    for comp_name, per_probe in (dual_sic.items() if isinstance(dual_sic, dict) else []):
        if not isinstance(per_probe, dict):
            continue
        for probe_name, rows in per_probe.items():
            if isinstance(rows, list):
                key = f"{str(comp_name)}@{str(probe_name)}"
                dual_sic_stats[key] = _component_sic_abs_log_stats(rows)

    return {
        "coverage_min": _safe_float(dist.get("coverage_frame_min", float("nan"))),
        "coverage_max": _safe_float(dist.get("coverage_frame_max", float("nan"))),
        "coverage_mean": _safe_float(dist.get("coverage_frame_mean", float("nan"))),
        "coverage_cv": _cv_from_counts(dist.get("sic_sampled_counts", [])),
        "grad_ratio_r_over_l": _safe_float(grad_global.get("grad_ratio_r_over_l", float("nan"))),
        "root_out_direct_ratio_step0": _safe_float(root_probe.get("out_direct_ratio_step0", float("nan"))),
        "root_cond_in_ratio_step0": _safe_float(root_probe.get("cond_in_ratio_step0", float("nan"))),
        "root_ratio_gap_step0": _safe_float(root_probe.get("ratio_gap_step0", float("nan"))),
        "root_direct_head_weight_rel_l2_best_sign": _safe_float(
            direct_head_sym.get("weight_rel_l2_best_sign", float("nan"))
        ),
        "root_direct_head_bias_rel_l2_best_sign": _safe_float(
            direct_head_sym.get("bias_rel_l2_best_sign", float("nan"))
        ),
        "component_grad_ratio": comp_ratio,
        "component_sic_abs_log": comp_sic_stats,
        "dual_probe_grad_ratio": dual_ratio,
        "dual_probe_sic_abs_log": dual_sic_stats,
    }


def _metric_from_row(row: Mapping[str, Any], key: str) -> float:
    if key.startswith("component_grad_ratio:"):
        name = key.split(":", 1)[1]
        return _safe_float((row.get("component_grad_ratio") or {}).get(name, float("nan")))
    if key.startswith("component_sic_abs_log:"):
        name = key.split(":", 1)[1]
        payload = (row.get("component_sic_abs_log") or {}).get(name, {})
        return _safe_float(payload.get("mean_abs_log", float("nan")))
    if key.startswith("dual_probe_grad_ratio:"):
        name = key.split(":", 1)[1]
        return _safe_float((row.get("dual_probe_grad_ratio") or {}).get(name, float("nan")))
    if key.startswith("dual_probe_sic_abs_log:"):
        name = key.split(":", 1)[1]
        payload = (row.get("dual_probe_sic_abs_log") or {}).get(name, {})
        return _safe_float(payload.get("mean_abs_log", float("nan")))
    return _safe_float(row.get(key, float("nan")))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Formal P1 sampler-only A/B (multi-seed + multi-epoch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config-json", type=str, default="config/exp_phase_DirectBranch_v1_d1_noreset.json")
    ap.add_argument(
        "--resume-ckpt",
        type=str,
        default="models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage6_direct_cond_anchor_20260124.pth",
    )
    ap.add_argument(
        "--out-model-dir",
        type=str,
        default="models/MLPL2_DirectBranch_v1__pipe_20260215_sampler_p1_formal",
    )
    ap.add_argument("--base-run-name", type=str, default="tmp_p1_sampler_formal")
    ap.add_argument("--modes", type=str, default="sliding,sic_balanced")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=None, help="Optional batch override.")
    ap.add_argument(
        "--train-config-override",
        action="append",
        default=None,
        help="Extra KEY=VALUE overrides for train.training_MPL (repeatable).",
    )

    ap.add_argument("--teacher", type=str, default="validate/teacher_batches/Walk_F_teacher.json")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--npz", type=str, default="raw_data/processed_data/Walk_F.npz")
    ap.add_argument("--target-clip", type=str, default="Walk_F")
    ap.add_argument("--depth", type=int, default=3)

    ap.add_argument("--report-branch", type=str, default="direct", choices=("inc", "direct", "blend"))
    ap.add_argument("--report-space", type=str, default="body", choices=("body", "world"))
    ap.add_argument("--report-axis", type=str, default="z", choices=("x", "y", "z"))
    ap.add_argument("--report-joints", type=str, default="calf_l,calf_r")
    ap.add_argument("--report-min-cycle", type=int, default=1)
    ap.add_argument("--report-exclude-wrap", action="store_true", default=True)
    ap.add_argument("--report-exclude-root", action="store_true", default=True)
    ap.add_argument("--omega-min-deg-s", type=float, default=30.0)

    ap.add_argument("--loss-branch", type=str, default="out", choices=("out", "out_direct"))
    ap.add_argument("--component-losses", type=str, default="rot_geo,rot_vel,direct_pose,direct_delta")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))

    ap.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Output dir under debug_output. Default: _p1_sampler_formal_YYYYMMDD_HHMMSS",
    )
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-freerun", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--skip-diagnose", action="store_true")
    args = ap.parse_args()

    cfg_path = _resolve_from_root(args.config_json)
    if not cfg_path.is_file():
        raise SystemExit(f"[FATAL] config-json not found: {cfg_path}")
    cfg = _load_json(cfg_path)

    resume_ckpt = _resolve_from_root(args.resume_ckpt)
    if not resume_ckpt.is_file():
        raise SystemExit(f"[FATAL] resume ckpt not found: {resume_ckpt}")

    modes = _parse_mode_list(args.modes)
    seeds = _parse_int_list(args.seeds)
    if int(args.epochs) < 1:
        raise SystemExit("[FATAL] --epochs must be >=1")

    train_out = _resolve_from_root(args.out_model_dir)
    if args.sweep_dir:
        sweep_dir = _resolve_from_root(args.sweep_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = _ROOT / "debug_output" / f"_p1_sampler_formal_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(_ROOT))
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

    mode_cfg_paths: Dict[str, Path] = {}
    for mode in modes:
        mode_cfg = dict(cfg)
        mode_cfg["dataset_index_mode"] = str(mode)
        cfg_out = sweep_dir / f"config_{mode}.json"
        _write_json(cfg_out, mode_cfg)
        mode_cfg_paths[mode] = cfg_out

    rows: List[Dict[str, Any]] = []

    for mode in modes:
        for seed in seeds:
            run_name = f"{args.base_run_name}_{mode}_seed{int(seed)}_e{int(args.epochs)}"
            run_dir = sweep_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            if not args.skip_train:
                train_cmd = [
                    sys.executable,
                    "-m",
                    "train.training_MPL",
                    "--config_json",
                    str(cfg_path),
                    "--out",
                    str(train_out),
                    "--run_name",
                    run_name,
                    "--resume",
                    str(resume_ckpt),
                    "--seed",
                    str(int(seed)),
                    "--config_override",
                    f"dataset_index_mode={mode}",
                    "--config_override",
                    f"epochs={int(args.epochs)}",
                ]
                if args.batch is not None:
                    train_cmd += ["--config_override", f"batch={int(args.batch)}"]
                for ov in (args.train_config_override or []):
                    txt = str(ov).strip()
                    if txt:
                        train_cmd += ["--config_override", txt]
                _run_and_tee(train_cmd, cwd=_ROOT, env=env, log_path=run_dir / "train.log")

            ckpt_path = _pick_ckpt(train_out, run_name)

            if not args.skip_freerun:
                freerun_cmd = [
                    sys.executable,
                    "-m",
                    "train.validate.run_freerun_cycles",
                    "--model",
                    str(ckpt_path),
                    "--teacher",
                    str(_resolve_from_root(args.teacher)),
                    "--bundle",
                    str(_resolve_from_root(args.bundle)),
                    "--pretrain-template",
                    str(_resolve_from_root(args.pretrain_template)),
                    "--encoder-bundle",
                    str(_resolve_from_root(args.encoder_bundle)),
                    "--npz-root",
                    str(_resolve_from_root(args.npz_root)),
                    "--out",
                    str(run_dir),
                    "--depth",
                    str(int(args.depth)),
                    "--export_joint_so3_error_series",
                ]
                _run_and_tee(freerun_cmd, cwd=_ROOT, env=env, log_path=run_dir / "freerun.log")

            freerun_json = _find_freerun_json(run_dir, args.target_clip)

            report_md = run_dir / "sic_vs_omega.md"
            report_fig = run_dir / "sic_vs_omega.png"
            if not args.skip_report:
                report_cmd = [
                    sys.executable,
                    str(_ROOT / "tools" / "report_sic_hotspots_vs_gt_angvel.py"),
                    "--freerun-json",
                    str(freerun_json),
                    "--npz",
                    str(_resolve_from_root(args.npz)),
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
                    str(float(args.omega_min_deg_s)),
                    "--out-md",
                    str(report_md),
                    "--out-fig",
                    str(report_fig),
                ]
                if bool(args.report_exclude_wrap):
                    report_cmd.append("--exclude-wrap")
                if bool(args.report_exclude_root):
                    report_cmd.append("--exclude-root")
                _run_and_tee(report_cmd, cwd=_ROOT, env=env, log_path=run_dir / "report.log")

            diag_out = run_dir / "sampling_grad_closure"
            diag_json = diag_out / "sampling_grad_closure.json"
            if not args.skip_diagnose:
                diag_cmd = [
                    sys.executable,
                    str(_ROOT / "tools" / "diagnose_stage7_sampling_grad_closure.py"),
                    "--config-json",
                    str(mode_cfg_paths[mode]),
                    "--ckpt",
                    str(ckpt_path),
                    "--target-clip",
                    str(args.target_clip),
                    "--depth",
                    str(int(args.depth)),
                    "--bundle",
                    str(_resolve_from_root(args.bundle)),
                    "--pretrain-template",
                    str(_resolve_from_root(args.pretrain_template)),
                    "--encoder-bundle",
                    str(_resolve_from_root(args.encoder_bundle)),
                    "--loss-branch",
                    str(args.loss_branch),
                    "--component-losses",
                    str(args.component_losses),
                    "--device",
                    str(args.device),
                    "--out-dir",
                    str(diag_out),
                ]
                if int(args.max_windows) > 0:
                    diag_cmd += ["--max-windows", str(int(args.max_windows))]
                _run_and_tee(diag_cmd, cwd=_ROOT, env=env, log_path=run_dir / "diagnose.log")

            if not diag_json.is_file():
                raise SystemExit(f"[FATAL] diagnose output missing: {diag_json}")

            diag_metrics = _extract_diag_metrics(diag_json)
            quick: Dict[str, QuickJoint] = {}
            if report_md.is_file():
                try:
                    quick = _parse_quick_from_md(report_md)
                except Exception:
                    quick = {}

            calf_l = quick.get("calf_l") if isinstance(quick, dict) else None
            calf_r = quick.get("calf_r") if isinstance(quick, dict) else None

            row = {
                "mode": str(mode),
                "seed": int(seed),
                "run_name": run_name,
                "run_dir": str(run_dir.resolve()),
                "ckpt": str(ckpt_path.resolve()),
                "freerun_json": str(freerun_json.resolve()),
                "report_md": str(report_md.resolve()) if report_md.is_file() else None,
                "report_fig": str(report_fig.resolve()) if report_fig.is_file() else None,
                "diagnose_json": str(diag_json.resolve()),
                "coverage_min": _safe_float(diag_metrics.get("coverage_min", float("nan"))),
                "coverage_max": _safe_float(diag_metrics.get("coverage_max", float("nan"))),
                "coverage_mean": _safe_float(diag_metrics.get("coverage_mean", float("nan"))),
                "coverage_cv": _safe_float(diag_metrics.get("coverage_cv", float("nan"))),
                "grad_ratio_r_over_l": _safe_float(diag_metrics.get("grad_ratio_r_over_l", float("nan"))),
                "root_out_direct_ratio_step0": _safe_float(
                    diag_metrics.get("root_out_direct_ratio_step0", float("nan"))
                ),
                "root_cond_in_ratio_step0": _safe_float(
                    diag_metrics.get("root_cond_in_ratio_step0", float("nan"))
                ),
                "root_ratio_gap_step0": _safe_float(diag_metrics.get("root_ratio_gap_step0", float("nan"))),
                "root_direct_head_weight_rel_l2_best_sign": _safe_float(
                    diag_metrics.get("root_direct_head_weight_rel_l2_best_sign", float("nan"))
                ),
                "root_direct_head_bias_rel_l2_best_sign": _safe_float(
                    diag_metrics.get("root_direct_head_bias_rel_l2_best_sign", float("nan"))
                ),
                "calf_l_sign_frac": _safe_float(calf_l.align_frac if calf_l else float("nan")),
                "calf_r_sign_frac": _safe_float(calf_r.align_frac if calf_r else float("nan")),
                "calf_l_dt_med": _safe_float(calf_l.dt_median if calf_l else float("nan")),
                "calf_r_dt_med": _safe_float(calf_r.dt_median if calf_r else float("nan")),
                "component_grad_ratio": dict(diag_metrics.get("component_grad_ratio", {})),
                "component_sic_abs_log": dict(diag_metrics.get("component_sic_abs_log", {})),
                "dual_probe_grad_ratio": dict(diag_metrics.get("dual_probe_grad_ratio", {})),
                "dual_probe_sic_abs_log": dict(diag_metrics.get("dual_probe_sic_abs_log", {})),
            }
            rows.append(row)

    component_names = sorted(
        {str(name) for r in rows for name in (r.get("component_grad_ratio", {}) or {}).keys()}
    )
    dual_probe_metric_names = sorted(
        {str(name) for r in rows for name in (r.get("dual_probe_grad_ratio", {}) or {}).keys()}
    )
    key_metrics = [
        "coverage_cv",
        "grad_ratio_r_over_l",
        "root_out_direct_ratio_step0",
        "root_cond_in_ratio_step0",
        "root_ratio_gap_step0",
        "root_direct_head_weight_rel_l2_best_sign",
        "root_direct_head_bias_rel_l2_best_sign",
        "calf_l_sign_frac",
        "calf_r_sign_frac",
        "calf_l_dt_med",
        "calf_r_dt_med",
    ]

    mode_stats: Dict[str, Any] = {}
    for mode in modes:
        sub = [r for r in rows if str(r.get("mode")) == str(mode)]
        stats: Dict[str, Any] = {
            "num_runs": int(len(sub)),
            "seeds": sorted(int(r.get("seed", -1)) for r in sub),
            "metrics": {},
            "component_grad_ratio": {},
            "component_sic_abs_log": {},
            "dual_probe_grad_ratio": {},
            "dual_probe_sic_abs_log": {},
            "calf_r_dt_positive_rate": float("nan"),
        }
        for mk in key_metrics:
            stats["metrics"][mk] = _mean_std([_metric_from_row(r, mk) for r in sub])
        for comp in component_names:
            stats["component_grad_ratio"][comp] = _mean_std(
                [_metric_from_row(r, f"component_grad_ratio:{comp}") for r in sub]
            )
            stats["component_sic_abs_log"][comp] = _mean_std(
                [_metric_from_row(r, f"component_sic_abs_log:{comp}") for r in sub]
            )
        for dual_key in dual_probe_metric_names:
            stats["dual_probe_grad_ratio"][dual_key] = _mean_std(
                [_metric_from_row(r, f"dual_probe_grad_ratio:{dual_key}") for r in sub]
            )
            stats["dual_probe_sic_abs_log"][dual_key] = _mean_std(
                [_metric_from_row(r, f"dual_probe_sic_abs_log:{dual_key}") for r in sub]
            )
        calf_r_vals = np.asarray([_metric_from_row(r, "calf_r_dt_med") for r in sub], dtype=np.float64)
        calf_r_vals = calf_r_vals[np.isfinite(calf_r_vals)]
        if calf_r_vals.size > 0:
            stats["calf_r_dt_positive_rate"] = float(np.mean(calf_r_vals > 0.0))
        mode_stats[mode] = stats

    rows_by_mode_seed: Dict[Tuple[str, int], Dict[str, Any]] = {
        (str(r.get("mode")), int(r.get("seed", -1))): r for r in rows
    }
    paired_metrics = key_metrics + [f"component_grad_ratio:{c}" for c in component_names] + [
        f"component_sic_abs_log:{c}" for c in component_names
    ] + [f"dual_probe_grad_ratio:{k}" for k in dual_probe_metric_names] + [
        f"dual_probe_sic_abs_log:{k}" for k in dual_probe_metric_names
    ]
    paired_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        r_sl = rows_by_mode_seed.get(("sliding", int(seed)))
        r_sb = rows_by_mode_seed.get(("sic_balanced", int(seed)))
        if r_sl is None or r_sb is None:
            continue
        rec: Dict[str, Any] = {"seed": int(seed)}
        for mk in paired_metrics:
            a = _metric_from_row(r_sb, mk)
            b = _metric_from_row(r_sl, mk)
            rec[mk] = a - b if math.isfinite(a) and math.isfinite(b) else float("nan")
        paired_rows.append(rec)

    paired_delta = {"num_pairs": int(len(paired_rows)), "metrics": {}}
    for mk in paired_metrics:
        paired_delta["metrics"][mk] = _mean_std([_safe_float(r.get(mk, float("nan"))) for r in paired_rows])

    summary: Dict[str, Any] = {
        "config_json": str(cfg_path.resolve()),
        "resume_ckpt": str(resume_ckpt.resolve()),
        "train_out": str(train_out.resolve()),
        "sweep_dir": str(sweep_dir.resolve()),
        "base_run_name": str(args.base_run_name),
        "modes": list(modes),
        "seeds": [int(s) for s in seeds],
        "epochs": int(args.epochs),
        "loss_branch": str(args.loss_branch),
        "component_losses": str(args.component_losses),
        "rows": rows,
        "component_names": component_names,
        "dual_probe_metric_names": dual_probe_metric_names,
        "mode_stats": mode_stats,
        "paired_delta_sic_balanced_minus_sliding": paired_delta,
        "paired_rows": paired_rows,
    }

    out_json = sweep_dir / "summary.json"
    out_md = sweep_dir / "summary.md"
    _write_json(out_json, summary)

    lines: List[str] = []
    lines.append("# Stage7 sampler formal P1 summary")
    lines.append("")
    lines.append(f"- config_json: `{cfg_path}`")
    lines.append(f"- resume_ckpt: `{resume_ckpt}`")
    lines.append(f"- train_out: `{train_out}`")
    lines.append(f"- modes: `{', '.join(modes)}`")
    lines.append(f"- seeds: `{', '.join(str(s) for s in seeds)}`")
    lines.append(f"- epochs: {int(args.epochs)}")
    lines.append(f"- loss_branch: `{args.loss_branch}`")
    lines.append("")

    lines.append("## Per-run table")
    lines.append("")
    lines.append("|mode|seed|coverage(min/max/cv)|grad_ratio_r_over_l|calf_l sign|calf_r sign|calf_l dt_med|calf_r dt_med|")
    lines.append("|:--|--:|:--|--:|--:|--:|--:|--:|")
    for r in sorted(rows, key=lambda x: (str(x.get("mode")), int(x.get("seed", -1)))):
        lines.append(
            "|{mode}|{seed}|{cmin:.1f}/{cmax:.1f}/{cv:.3f}|{gr:.3f}|{ls:.3f}|{rs:.3f}|{ldt:+.3f}|{rdt:+.3f}|".format(
                mode=str(r.get("mode")),
                seed=int(r.get("seed", -1)),
                cmin=_safe_float(r.get("coverage_min", float("nan"))),
                cmax=_safe_float(r.get("coverage_max", float("nan"))),
                cv=_safe_float(r.get("coverage_cv", float("nan"))),
                gr=_safe_float(r.get("grad_ratio_r_over_l", float("nan"))),
                ls=_safe_float(r.get("calf_l_sign_frac", float("nan"))),
                rs=_safe_float(r.get("calf_r_sign_frac", float("nan"))),
                ldt=_safe_float(r.get("calf_l_dt_med", float("nan"))),
                rdt=_safe_float(r.get("calf_r_dt_med", float("nan"))),
            )
        )

    lines.append("")
    lines.append("## Root-cause probe per run")
    lines.append("")
    lines.append("|mode|seed|out_direct_ratio_step0|cond_in_ratio_step0|ratio_gap_step0|head_w_rel_l2_best_sign|head_b_rel_l2_best_sign|")
    lines.append("|:--|--:|--:|--:|--:|--:|--:|")
    for r in sorted(rows, key=lambda x: (str(x.get("mode")), int(x.get("seed", -1)))):
        lines.append(
            "|{mode}|{seed}|{od:.3f}|{ci:.3f}|{gap:+.3f}|{wl2:.3f}|{bl2:.3f}|".format(
                mode=str(r.get("mode")),
                seed=int(r.get("seed", -1)),
                od=_safe_float(r.get("root_out_direct_ratio_step0", float("nan"))),
                ci=_safe_float(r.get("root_cond_in_ratio_step0", float("nan"))),
                gap=_safe_float(r.get("root_ratio_gap_step0", float("nan"))),
                wl2=_safe_float(r.get("root_direct_head_weight_rel_l2_best_sign", float("nan"))),
                bl2=_safe_float(r.get("root_direct_head_bias_rel_l2_best_sign", float("nan"))),
            )
        )

    lines.append("")
    lines.append("## Mode aggregate (mean ± std)")
    lines.append("")
    lines.append("|mode|coverage_cv|grad_ratio_r_over_l|calf_l sign|calf_r sign|calf_l dt_med|calf_r dt_med|calf_r dt>0 rate|")
    lines.append("|:--|--:|--:|--:|--:|--:|--:|--:|")

    def _fmt_mean_std(payload: Mapping[str, Any], key: str) -> str:
        d = payload.get(key, {}) if isinstance(payload, Mapping) else {}
        return f"{_safe_float(d.get('mean', float('nan'))):.3f}±{_safe_float(d.get('std', float('nan'))):.3f}"

    for mode in modes:
        ms = mode_stats.get(mode, {})
        mm = ms.get("metrics", {}) if isinstance(ms, dict) else {}
        lines.append(
            "|{mode}|{cv}|{gr}|{ls}|{rs}|{ldt}|{rdt}|{pos:.3f}|".format(
                mode=mode,
                cv=_fmt_mean_std(mm, "coverage_cv"),
                gr=_fmt_mean_std(mm, "grad_ratio_r_over_l"),
                ls=_fmt_mean_std(mm, "calf_l_sign_frac"),
                rs=_fmt_mean_std(mm, "calf_r_sign_frac"),
                ldt=_fmt_mean_std(mm, "calf_l_dt_med"),
                rdt=_fmt_mean_std(mm, "calf_r_dt_med"),
                pos=_safe_float(ms.get("calf_r_dt_positive_rate", float("nan"))),
            )
        )

    lines.append("")
    lines.append("## Root-cause probe aggregate (mean ± std)")
    lines.append("")
    lines.append(
        "|mode|out_direct_ratio_step0|cond_in_ratio_step0|ratio_gap_step0|head_w_rel_l2_best_sign|head_b_rel_l2_best_sign|"
    )
    lines.append("|:--|--:|--:|--:|--:|--:|")
    for mode in modes:
        ms = mode_stats.get(mode, {})
        mm = ms.get("metrics", {}) if isinstance(ms, dict) else {}
        lines.append(
            "|{mode}|{od}|{ci}|{gap}|{wl2}|{bl2}|".format(
                mode=mode,
                od=_fmt_mean_std(mm, "root_out_direct_ratio_step0"),
                ci=_fmt_mean_std(mm, "root_cond_in_ratio_step0"),
                gap=_fmt_mean_std(mm, "root_ratio_gap_step0"),
                wl2=_fmt_mean_std(mm, "root_direct_head_weight_rel_l2_best_sign"),
                bl2=_fmt_mean_std(mm, "root_direct_head_bias_rel_l2_best_sign"),
            )
        )

    lines.append("")
    lines.append("## Paired delta (sic_balanced - sliding)")
    lines.append("")
    lines.append("|metric|mean_delta|std|n|")
    lines.append("|:--|--:|--:|--:|")
    for mk in key_metrics:
        d = paired_delta.get("metrics", {}).get(mk, {})
        lines.append(
            "|{mk}|{mean:+.4f}|{std:.4f}|{n}|".format(
                mk=mk,
                mean=_safe_float(d.get("mean", float("nan"))),
                std=_safe_float(d.get("std", float("nan"))),
                n=int(d.get("n", 0) or 0),
            )
        )

    if component_names:
        lines.append("")
        lines.append("## Component gradients (global ratio + SIC |log ratio| mean)")
        lines.append("")
        lines.append("|mode|component|global_ratio mean±std|sic_abs_log mean±std|")
        lines.append("|:--|:--|--:|--:|")
        for mode in modes:
            ms = mode_stats.get(mode, {})
            cg = ms.get("component_grad_ratio", {}) if isinstance(ms, dict) else {}
            cs = ms.get("component_sic_abs_log", {}) if isinstance(ms, dict) else {}
            for comp in component_names:
                g = cg.get(comp, {}) if isinstance(cg, dict) else {}
                s = cs.get(comp, {}) if isinstance(cs, dict) else {}
                lines.append(
                    "|{mode}|{comp}|{gm:.3f}±{gs:.3f}|{sm:.3f}±{ss:.3f}|".format(
                        mode=mode,
                        comp=comp,
                        gm=_safe_float(g.get("mean", float("nan"))),
                        gs=_safe_float(g.get("std", float("nan"))),
                        sm=_safe_float(s.get("mean", float("nan"))),
                        ss=_safe_float(s.get("std", float("nan"))),
                    )
                )

        lines.append("")
        lines.append("### Paired component delta (sic_balanced - sliding)")
        lines.append("")
        lines.append("|metric|mean_delta|std|n|")
        lines.append("|:--|--:|--:|--:|")
        for comp in component_names:
            for mk in (f"component_grad_ratio:{comp}", f"component_sic_abs_log:{comp}"):
                d = paired_delta.get("metrics", {}).get(mk, {})
                lines.append(
                    "|{mk}|{mean:+.4f}|{std:.4f}|{n}|".format(
                        mk=mk,
                        mean=_safe_float(d.get("mean", float("nan"))),
                        std=_safe_float(d.get("std", float("nan"))),
                        n=int(d.get("n", 0) or 0),
                    )
                )

    if dual_probe_metric_names:
        lines.append("")
        lines.append("## Dual-probe gradients (component@probe)")
        lines.append("")
        lines.append("|mode|component@probe|global_ratio mean±std|sic_abs_log mean±std|")
        lines.append("|:--|:--|--:|--:|")
        for mode in modes:
            ms = mode_stats.get(mode, {})
            dg = ms.get("dual_probe_grad_ratio", {}) if isinstance(ms, dict) else {}
            ds = ms.get("dual_probe_sic_abs_log", {}) if isinstance(ms, dict) else {}
            for key in dual_probe_metric_names:
                g = dg.get(key, {}) if isinstance(dg, dict) else {}
                s = ds.get(key, {}) if isinstance(ds, dict) else {}
                lines.append(
                    "|{mode}|{key}|{gm:.3f}±{gs:.3f}|{sm:.3f}±{ss:.3f}|".format(
                        mode=mode,
                        key=key,
                        gm=_safe_float(g.get("mean", float("nan"))),
                        gs=_safe_float(g.get("std", float("nan"))),
                        sm=_safe_float(s.get("mean", float("nan"))),
                        ss=_safe_float(s.get("std", float("nan"))),
                    )
                )

        lines.append("")
        lines.append("### Paired dual-probe delta (sic_balanced - sliding)")
        lines.append("")
        lines.append("|metric|mean_delta|std|n|")
        lines.append("|:--|--:|--:|--:|")
        for key in dual_probe_metric_names:
            for mk in (f"dual_probe_grad_ratio:{key}", f"dual_probe_sic_abs_log:{key}"):
                d = paired_delta.get("metrics", {}).get(mk, {})
                lines.append(
                    "|{mk}|{mean:+.4f}|{std:.4f}|{n}|".format(
                        mk=mk,
                        mean=_safe_float(d.get("mean", float("nan"))),
                        std=_safe_float(d.get("std", float("nan"))),
                        n=int(d.get("n", 0) or 0),
                    )
                )

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for mode in modes:
        lines.append(f"- mode={mode} config: `{mode_cfg_paths[mode]}`")
    lines.append(f"- summary_json: `{out_json}`")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {out_json}")
    print(f"[OK] wrote: {out_md}")


if __name__ == "__main__":
    main()
