#!/usr/bin/env python3
"""
Sweep posttrain hyperparams (lr / rollout_cycles / steps_per_epoch) for direct_pose_hinge,
then run freerun_cycles and summarize:
  - Global R1+ metrics (GeoLocalDeg / DirectGeoLocalDeg / BlendGeoLocalDeg)
  - calf_r swing diagnostics on direct branch (contact_r==0, R1+):
      mean_ang_deg, P(ang_deg>th), mean/std omega_z@ang>th, phase amp.

This is intended to de-risk "should we extend hinge vs rollback" by providing an
automated, reproducible table across a small grid.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def _parse_csv_floats(spec: str) -> List[float]:
    s = (spec or "").strip()
    if not s:
        return []
    out: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


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


def _fmt_lr_slug(lr: float) -> str:
    # Prefer scientific notation for stable filenames (e.g., 0.001 -> 1e-3).
    s = f"{lr:.0e}"
    try:
        # Preserve 3e-4 style instead of 3E-04
        s = s.replace("E", "e").replace("e+0", "e").replace("e+","e").replace("e-0", "e-")
    except Exception:
        pass
    return s


def _clip_name_from_json_path(p: Path) -> str:
    # "<clip>_freerun_cycles.json" -> "<clip>"
    name = p.name
    suf = "_freerun_cycles.json"
    return name[: -len(suf)] if name.endswith(suf) else name


def _glob_freerun_jsons(out_dir: Path) -> List[Path]:
    return sorted(out_dir.glob("*_freerun_cycles.json"))


def _mean(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return float(sum(float(v) for v in vals) / float(len(vals)))


def _std(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    m = _mean(vals)
    if m is None:
        return None
    var = sum((float(v) - float(m)) ** 2 for v in vals) / float(len(vals))
    return float(var ** 0.5)


def _weighted_mean(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
    num = 0.0
    den = 0.0
    for w, v in pairs:
        if w <= 0.0:
            continue
        num += float(w) * float(v)
        den += float(w)
    if den <= 0.0:
        return None
    return float(num / den)


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


@dataclass
class GlobalAgg:
    weight_steps: float = 0.0
    sums: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sums is None:
            self.sums = {}

    def add(self, *, steps: float, metrics: Dict[str, Optional[float]]) -> None:
        if steps <= 0.0:
            return
        self.weight_steps += float(steps)
        for k, v in metrics.items():
            if v is None:
                continue
            self.sums[k] = self.sums.get(k, 0.0) + float(v) * float(steps)

    def mean(self) -> Dict[str, Optional[float]]:
        if self.weight_steps <= 0.0:
            return {k: None for k in self.sums.keys()}
        return {k: float(v / self.weight_steps) for k, v in self.sums.items()}


@dataclass
class CalfSwingAgg:
    th_deg: float = 20.0
    n_tot: int = 0
    sum_ang: float = 0.0
    n_gt: int = 0
    sum_omega: float = 0.0
    sum_omega2: float = 0.0
    # For phase-locked bias: per-phase omega sums/counts (only for ang>th).
    phase_sum: Dict[int, float] = None  # type: ignore[assignment]
    phase_cnt: Dict[int, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.phase_sum is None:
            self.phase_sum = {}
        if self.phase_cnt is None:
            self.phase_cnt = {}

    def add_sample(self, *, ang_deg: float) -> None:
        self.n_tot += 1
        self.sum_ang += float(ang_deg)

    def add_gt(self, *, omega_deg: float, phase: Optional[int]) -> None:
        self.n_gt += 1
        self.sum_omega += float(omega_deg)
        self.sum_omega2 += float(omega_deg) * float(omega_deg)
        if phase is not None:
            p = int(phase)
            self.phase_sum[p] = self.phase_sum.get(p, 0.0) + float(omega_deg)
            self.phase_cnt[p] = self.phase_cnt.get(p, 0) + 1

    def summary(self) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        out["th_deg"] = float(self.th_deg)
        out["n"] = float(self.n_tot)
        out["mean_ang_deg"] = (self.sum_ang / float(self.n_tot)) if self.n_tot > 0 else None
        out["p_ang_gt_th"] = (float(self.n_gt) / float(self.n_tot)) if self.n_tot > 0 else None

        if self.n_gt > 0:
            m = self.sum_omega / float(self.n_gt)
            var = (self.sum_omega2 / float(self.n_gt)) - (m * m)
            var = max(0.0, float(var))
            out["mean_omega_deg_if_gt_th"] = float(m)
            out["std_omega_deg_if_gt_th"] = float(var ** 0.5)
        else:
            out["mean_omega_deg_if_gt_th"] = None
            out["std_omega_deg_if_gt_th"] = None

        # Phase amp on omega (max mean - min mean), conditioned on ang>th.
        ph_means: Dict[int, float] = {}
        for p, s in self.phase_sum.items():
            c = int(self.phase_cnt.get(p, 0))
            if c > 0:
                ph_means[int(p)] = float(s) / float(c)
        if ph_means:
            p_max = max(ph_means, key=lambda p: ph_means[p])
            p_min = min(ph_means, key=lambda p: ph_means[p])
            out["phase_max_mean_omega_deg_if_gt_th"] = float(ph_means[p_max])
            out["phase_min_mean_omega_deg_if_gt_th"] = float(ph_means[p_min])
            out["phase_amp_mean_omega_deg_if_gt_th"] = float(ph_means[p_max] - ph_means[p_min])
            out["phase_max"] = float(p_max)
            out["phase_min"] = float(p_min)
        else:
            out["phase_max_mean_omega_deg_if_gt_th"] = None
            out["phase_min_mean_omega_deg_if_gt_th"] = None
            out["phase_amp_mean_omega_deg_if_gt_th"] = None
            out["phase_max"] = None
            out["phase_min"] = None

        return out


def _extract_global_r1p(obj: Dict[str, Any], *, keys: Sequence[str]) -> Tuple[Dict[str, Optional[float]], float]:
    rounds = obj.get("metrics_per_round", None)
    if not isinstance(rounds, list) or not rounds:
        return {k: None for k in keys}, 0.0

    pairs: Dict[str, List[Tuple[float, float]]] = {k: [] for k in keys}
    total_steps = 0.0
    for r in rounds:
        if not isinstance(r, dict):
            continue
        if int(r.get("round", 0) or 0) < 1:
            continue
        steps = float(r.get("steps", 0) or 0)
        if steps <= 0.0:
            steps = 1.0
        total_steps += steps
        for k in keys:
            v = _as_float(r.get(k, None))
            if v is not None:
                pairs[k].append((steps, v))
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        out[k] = _weighted_mean(pairs.get(k, []))
    return out, float(total_steps)


def _extract_calf_r_swing(
    obj: Dict[str, Any],
    *,
    bone: str = "calf_r",
    branch: str = "direct",
    min_cycle: int = 1,
    contact_source: str = "gt",
    contact_idx: int = 1,
    contact_value: Optional[int] = 0,
    contact_thresh: float = 0.5,
    angle_thresh: Optional[float] = None,
) -> CalfSwingAgg:
    ko = obj.get("keybone_omega", None)
    if not isinstance(ko, dict):
        return CalfSwingAgg(th_deg=float(angle_thresh or 20.0))
    series = ko.get("series", None)
    if not isinstance(series, dict):
        return CalfSwingAgg(th_deg=float(angle_thresh or ko.get("deg_thresh") or 20.0))

    th = float(angle_thresh) if angle_thresh is not None else float(ko.get("deg_thresh") or 20.0)
    th = max(0.0, th)
    agg = CalfSwingAgg(th_deg=float(th))

    sbranches = series.get("branches", None)
    if not isinstance(sbranches, dict):
        return agg
    bdat = sbranches.get(str(branch), None)
    if not isinstance(bdat, dict):
        return agg
    omega_map = bdat.get("omega_axis_deg", None)
    ang_map = bdat.get("ang_deg", None)
    if not isinstance(omega_map, dict) or not isinstance(ang_map, dict):
        return agg
    omega = omega_map.get(str(bone), None)
    ang = ang_map.get(str(bone), None)
    if not isinstance(omega, list) or not isinstance(ang, list):
        return agg

    steps = obj.get("metrics_per_step", None)
    if not isinstance(steps, list) or not steps:
        steps = None

    cycle_len = _as_int(obj.get("cycle_len", 0) or 0, 0)

    contact_key = {"gt": "ContactGTPerC", "plan": "ContactPlanPerC", "meas": "ContactMeasPerC"}.get(
        str(contact_source).strip().lower(),
        "ContactGTPerC",
    )
    want_contact: Optional[int] = None
    if contact_value is not None:
        want_contact = int(contact_value)
        if want_contact not in (0, 1):
            want_contact = None

    T = min(len(omega), len(ang), len(steps) if steps is not None else (len(ang)))
    for i in range(int(T)):
        # R1+ filter
        cy = None
        if steps is not None and i < len(steps) and isinstance(steps[i], dict):
            cy_raw = steps[i].get("cycle", None)
            if isinstance(cy_raw, int):
                cy = int(cy_raw)
        if cy is None and cycle_len > 0:
            cy = int(i // cycle_len)
        if int(cy or 0) < int(min_cycle):
            continue

        # Contact filter
        if want_contact is not None:
            if steps is None or i >= len(steps) or not isinstance(steps[i], dict):
                continue
            c = steps[i].get(contact_key, None)
            if not isinstance(c, list) or int(contact_idx) < 0 or int(contact_idx) >= len(c):
                continue
            try:
                v = float(c[int(contact_idx)])
            except Exception:
                continue
            state = 1 if v >= float(contact_thresh) else 0
            if int(state) != int(want_contact):
                continue

        try:
            a = float(ang[i])
            o = float(omega[i])
        except Exception:
            continue

        agg.add_sample(ang_deg=a)
        if a > th:
            phase: Optional[int] = None
            if steps is not None and i < len(steps) and isinstance(steps[i], dict):
                si = steps[i].get("step_in_cycle", None)
                if isinstance(si, int):
                    phase = int(si)
            if phase is None and cycle_len > 0:
                phase = int(i % cycle_len)
            agg.add_gt(omega_deg=o, phase=phase)

    return agg


def _run_cmd(*, cmd: List[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    _ensure_dir(log_path.parent)
    log_path.write_text("", encoding="utf-8")
    if dry_run:
        log_path.write_text("[DRY_RUN]\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"cmd failed (rc={proc.returncode}). See log: {log_path}\nCMD: {' '.join(cmd)}")


def _posttrain_ckpt_path(*, project_root: Path, out_dir: str, run_name: str) -> Path:
    outp = Path(out_dir)
    if not outp.is_absolute():
        outp = (project_root / outp).resolve()
    return outp / f"ckpt_last_{run_name}.pth"


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep direct_pose_hinge posttrain params and summarize freerun metrics.")
    ap.add_argument("--posttrain-config", type=str, required=True, help="Path to posttrain config JSON.")
    ap.add_argument(
        "--teacher",
        nargs="+",
        default=["validate/teacher_batches/Walk_F_teacher.json"],
        help="Teacher JSON files/dirs/globs for freerun_cycles.",
    )
    ap.add_argument("--out-root", type=str, default="debug_output/_sweep_direct_pose_hinge", help="Output root dir.")
    ap.add_argument("--tag", type=str, default=None, help="Optional subdir under --out-root.")
    ap.add_argument("--device", type=str, default="cpu", help="Device for posttrain+freerun (auto/cpu/cuda/mps).")

    ap.add_argument("--lrs", type=str, default="3e-4,1e-3,3e-3", help="Comma-separated lr list.")
    ap.add_argument("--rollout-cycles", type=str, default="1,3,5", help="Comma-separated rollout_cycles list (posttrain).")
    ap.add_argument("--steps-per-epoch", type=str, default="300,600", help="Comma-separated steps_per_epoch list (posttrain).")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs (posttrain).")
    ap.add_argument("--seed", type=int, default=None, help="Override seed (posttrain).")
    ap.add_argument(
        "--direct-pose-hinge-train-only",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Posttrain: train hinge head only (freeze base direct head).",
    )

    ap.add_argument("--freerun-rounds", type=int, default=5, help="Freerun: --rounds.")
    ap.add_argument("--freerun-phase-reset-source", type=str, default="td_hazard", help="Freerun: --phase_reset_source.")
    ap.add_argument(
        "--freerun-lambda-fusion-apply",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Freerun: --lambda_fusion_apply.",
    )
    ap.add_argument(
        "--freerun-log-contacts",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Freerun: --log_contacts.",
    )
    ap.add_argument("--freerun-force", action="store_true", help="Freerun: --force overwrite outputs.")

    ap.add_argument("--keybone-series-bones", type=str, default="calf_r", help="Freerun: --keybone_omega_series_bones.")
    ap.add_argument("--keybone-series-axis", type=str, default="z", help="Freerun: --keybone_omega_series_axis.")

    ap.add_argument(
        "--resume",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If outputs exist, skip recompute and only parse/summarize.",
    )
    ap.add_argument("--keep-going", action="store_true", help="Continue sweep even if a run fails.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands and create logs, but do not run.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = _resolve_under(project_root, args.posttrain_config)
    if not cfg_path.is_file():
        raise SystemExit(f"[FATAL] posttrain config not found: {cfg_path}")
    cfg = _load_json(cfg_path)
    if not isinstance(cfg, dict):
        raise SystemExit(f"[FATAL] invalid posttrain config JSON: {cfg_path}")

    out_root = _resolve_under(project_root, args.out_root)
    if args.tag:
        out_root = out_root / str(args.tag)
    _ensure_dir(out_root)

    # Resolve hinge args from config (used for freerun model instantiation).
    hinge_enable = bool(cfg.get("direct_pose_hinge_enable", False))
    hinge_bones = str(cfg.get("direct_pose_hinge_bones", "calf_r") or "calf_r")
    hinge_axis = str(cfg.get("direct_pose_hinge_axis", "z") or "z")
    hinge_max_deg = float(cfg.get("direct_pose_hinge_max_deg", 45.0) or 45.0)
    hinge_hidden = cfg.get("direct_pose_hinge_hidden", None)

    lrs = _parse_csv_floats(args.lrs)
    rollout_cycles = _parse_csv_ints(args.rollout_cycles)
    steps_per_epoch = _parse_csv_ints(args.steps_per_epoch)
    if not lrs or not rollout_cycles or not steps_per_epoch:
        raise SystemExit("[FATAL] empty sweep list: need --lrs, --rollout-cycles, --steps-per-epoch.")

    # Where posttrain writes ckpt.
    out_dir = str(cfg.get("out_dir", "models/MLPL2_DirectBranch_v1") or "models/MLPL2_DirectBranch_v1")
    base_run_name = str(cfg.get("run_name", "posttrain_direct_pose_hinge_sweep") or "posttrain_direct_pose_hinge_sweep")

    meta: Dict[str, Any] = {
        "script": str(Path(__file__).name),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "posttrain_config": str(cfg_path),
        "posttrain_out_dir": out_dir,
        "base_run_name": base_run_name,
        "device": str(args.device),
        "sweep": {"lrs": lrs, "rollout_cycles": rollout_cycles, "steps_per_epoch": steps_per_epoch},
        "posttrain_overrides": {"epochs": args.epochs, "seed": args.seed, "hinge_train_only": bool(args.direct_pose_hinge_train_only)},
        "freerun": {
            "teacher": list(args.teacher),
            "rounds": int(args.freerun_rounds),
            "phase_reset_source": str(args.freerun_phase_reset_source),
            "lambda_fusion_apply": bool(args.freerun_lambda_fusion_apply),
            "log_contacts": bool(args.freerun_log_contacts),
            "keybone_series_bones": str(args.keybone_series_bones),
            "keybone_series_axis": str(args.keybone_series_axis),
        },
        "hinge": {
            "enable": hinge_enable,
            "bones": hinge_bones,
            "axis": hinge_axis,
            "max_deg": hinge_max_deg,
            "hidden": hinge_hidden,
        },
    }
    _write_json(out_root / "sweep_meta.json", meta)

    rows: List[Dict[str, Any]] = []
    combos = list(itertools.product(lrs, rollout_cycles, steps_per_epoch))
    total = len(combos)
    print(f"[SweepHinge] out_root={out_root} runs={total} device={args.device} dry_run={bool(args.dry_run)}")

    for idx, (lr, rc, spe) in enumerate(combos):
        lr_slug = _fmt_lr_slug(float(lr))
        run_name = f"{base_run_name}_hingeONLY_lr{lr_slug}_rc{int(rc)}_spe{int(spe)}"
        exp_root = out_root / f"exp_{idx:03d}_{run_name}"
        freerun_out = exp_root / "freerun"
        _ensure_dir(exp_root)

        row: Dict[str, Any] = {
            "idx": int(idx),
            "run_name": run_name,
            "lr": float(lr),
            "rollout_cycles": int(rc),
            "steps_per_epoch": int(spe),
            "epochs": int(args.epochs) if args.epochs is not None else None,
            "seed": int(args.seed) if args.seed is not None else None,
            "status": "init",
            "posttrain_ckpt": None,
            "freerun_out": str(freerun_out),
            "freerun_jsons": [],
            "metrics": {},
            "error": None,
        }

        try:
            ckpt_path = _posttrain_ckpt_path(project_root=project_root, out_dir=out_dir, run_name=run_name)
            row["posttrain_ckpt"] = str(ckpt_path)

            # ---- posttrain
            post_cmd: List[str] = [
                sys.executable,
                "-m",
                "train.posttrain",
                "--config",
                str(cfg_path),
                "--run_name",
                str(run_name),
                "--lr",
                str(float(lr)),
                "--rollout_cycles",
                str(int(rc)),
                "--steps_per_epoch",
                str(int(spe)),
                "--device",
                str(args.device),
            ]
            if args.epochs is not None:
                post_cmd += ["--epochs", str(int(args.epochs))]
            if args.seed is not None:
                post_cmd += ["--seed", str(int(args.seed))]
            if bool(args.direct_pose_hinge_train_only):
                post_cmd += ["--direct_pose_hinge_train_only", "true"]
            (exp_root / "posttrain_cmd.txt").write_text(" ".join(post_cmd) + "\n", encoding="utf-8")
            if not bool(args.resume and ckpt_path.is_file()):
                _run_cmd(cmd=post_cmd, cwd=project_root, log_path=exp_root / "posttrain.log", dry_run=bool(args.dry_run))
            else:
                (exp_root / "posttrain.log").write_text("[RESUME] skipped posttrain (ckpt exists)\n", encoding="utf-8")

            if not bool(args.dry_run) and not ckpt_path.is_file():
                raise FileNotFoundError(f"posttrain ckpt not found: {ckpt_path}")

            # ---- freerun
            free_cmd: List[str] = [
                sys.executable,
                "-m",
                "train.validate.run_freerun_cycles",
                "--model",
                str(ckpt_path),
                "--teacher",
                *[str(x) for x in args.teacher],
                "--rounds",
                str(int(args.freerun_rounds)),
                "--device",
                str(args.device),
                "--phase_reset_source",
                str(args.freerun_phase_reset_source),
                "--out",
                str(freerun_out),
                "--export_keybone_omega",
                "--export_keybone_omega_series",
                "--keybone_omega_series_bones",
                str(args.keybone_series_bones),
                "--keybone_omega_series_axis",
                str(args.keybone_series_axis),
            ]
            if bool(args.freerun_lambda_fusion_apply):
                free_cmd.append("--lambda_fusion_apply")
            if bool(args.freerun_log_contacts):
                free_cmd.append("--log_contacts")
            if bool(args.freerun_force):
                free_cmd.append("--force")

            # Ensure hinge head is instantiated in freerun, otherwise hinge weights become "unexpected" and get ignored.
            if hinge_enable:
                free_cmd += [
                    "--direct_pose_hinge_enable",
                    "--direct_pose_hinge_bones",
                    str(hinge_bones),
                    "--direct_pose_hinge_axis",
                    str(hinge_axis),
                    "--direct_pose_hinge_max_deg",
                    str(float(hinge_max_deg)),
                ]
                if hinge_hidden is not None:
                    free_cmd += ["--direct_pose_hinge_hidden", str(int(hinge_hidden))]

            (exp_root / "freerun_cmd.txt").write_text(" ".join(free_cmd) + "\n", encoding="utf-8")
            have_jsons = bool(_glob_freerun_jsons(freerun_out))
            if not bool(args.resume and have_jsons):
                _run_cmd(cmd=free_cmd, cwd=project_root, log_path=exp_root / "freerun.log", dry_run=bool(args.dry_run))
            else:
                (exp_root / "freerun.log").write_text("[RESUME] skipped freerun (json exists)\n", encoding="utf-8")

            if bool(args.dry_run):
                row["status"] = "dry_run"
                rows.append(row)
                print(f"[{idx+1:03d}/{total:03d}] {run_name}  status={row['status']}")
                continue

            # ---- parse freerun outputs
            jsons = _glob_freerun_jsons(freerun_out)
            if not jsons:
                raise FileNotFoundError(f"no *_freerun_cycles.json found in {freerun_out}")
            row["freerun_jsons"] = [str(p) for p in jsons]

            keys = ("GeoLocalDeg", "DirectGeoLocalDeg", "BlendGeoLocalDeg")
            global_agg = GlobalAgg()
            calf_agg: Optional[CalfSwingAgg] = None

            clip_rows: List[Dict[str, Any]] = []
            for jp in jsons:
                obj = _load_json(jp)
                clip = str(obj.get("clip") or _clip_name_from_json_path(jp))
                g, steps_w = _extract_global_r1p(obj, keys=keys)
                global_agg.add(steps=steps_w, metrics=g)

                calf = _extract_calf_r_swing(
                    obj,
                    bone="calf_r",
                    branch="direct",
                    min_cycle=1,
                    contact_source="gt",
                    contact_idx=1,
                    contact_value=0,
                    contact_thresh=0.5,
                    angle_thresh=None,
                )
                if calf_agg is None:
                    calf_agg = calf
                else:
                    # Combine by replaying aggregates (exact).
                    # Note: thresholds are assumed identical across clips (keybone_omega_deg_thresh).
                    calf_agg.n_tot += int(calf.n_tot)
                    calf_agg.sum_ang += float(calf.sum_ang)
                    calf_agg.n_gt += int(calf.n_gt)
                    calf_agg.sum_omega += float(calf.sum_omega)
                    calf_agg.sum_omega2 += float(calf.sum_omega2)
                    for p, s in calf.phase_sum.items():
                        calf_agg.phase_sum[int(p)] = calf_agg.phase_sum.get(int(p), 0.0) + float(s)
                    for p, c in calf.phase_cnt.items():
                        calf_agg.phase_cnt[int(p)] = int(calf_agg.phase_cnt.get(int(p), 0)) + int(c)

                clip_rows.append(
                    {
                        "clip": clip,
                        "json": str(jp),
                        "global_r1p": g,
                        "calf_r_swing_direct_r1p": calf.summary(),
                    }
                )

            gmean = global_agg.mean()
            calf_sum = calf_agg.summary() if calf_agg is not None else {}
            row["metrics"] = {
                "global_r1p": gmean,
                "calf_r_swing_direct_r1p": calf_sum,
                "n_clips": int(len(jsons)),
                "clips": clip_rows,
            }
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            if not bool(args.keep_going):
                rows.append(row)
                break
        rows.append(row)

        print(f"[{idx+1:03d}/{total:03d}] {run_name}  status={row['status']}")

    # ---- write outputs
    summary = {"meta": meta, "rows": rows}
    _write_json(out_root / "sweep_summary.json", summary)

    # Flat CSV for easy sorting
    csv_path = out_root / "sweep_results.csv"
    csv_fields = [
        "idx",
        "run_name",
        "status",
        "lr",
        "rollout_cycles",
        "steps_per_epoch",
        "epochs",
        "seed",
        "posttrain_ckpt",
        "freerun_out",
        "n_clips",
        "GeoLocalDeg_r1p",
        "DirectGeoLocalDeg_r1p",
        "BlendGeoLocalDeg_r1p",
        "calf_r_swing_n",
        "calf_r_swing_mean_ang",
        "calf_r_swing_p_ang_gt_th",
        "calf_r_swing_mean_omega_if_gt_th",
        "calf_r_swing_std_omega_if_gt_th",
        "calf_r_swing_phase_amp_omega_if_gt_th",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in rows:
            m = (r.get("metrics") or {}) if isinstance(r.get("metrics"), dict) else {}
            g = (m.get("global_r1p") or {}) if isinstance(m.get("global_r1p"), dict) else {}
            calf = (m.get("calf_r_swing_direct_r1p") or {}) if isinstance(m.get("calf_r_swing_direct_r1p"), dict) else {}
            w.writerow(
                {
                    "idx": r.get("idx"),
                    "run_name": r.get("run_name"),
                    "status": r.get("status"),
                    "lr": r.get("lr"),
                    "rollout_cycles": r.get("rollout_cycles"),
                    "steps_per_epoch": r.get("steps_per_epoch"),
                    "epochs": r.get("epochs"),
                    "seed": r.get("seed"),
                    "posttrain_ckpt": r.get("posttrain_ckpt"),
                    "freerun_out": r.get("freerun_out"),
                    "n_clips": m.get("n_clips"),
                    "GeoLocalDeg_r1p": _as_float(g.get("GeoLocalDeg")),
                    "DirectGeoLocalDeg_r1p": _as_float(g.get("DirectGeoLocalDeg")),
                    "BlendGeoLocalDeg_r1p": _as_float(g.get("BlendGeoLocalDeg")),
                    "calf_r_swing_n": _as_float(calf.get("n")),
                    "calf_r_swing_mean_ang": _as_float(calf.get("mean_ang_deg")),
                    "calf_r_swing_p_ang_gt_th": _as_float(calf.get("p_ang_gt_th")),
                    "calf_r_swing_mean_omega_if_gt_th": _as_float(calf.get("mean_omega_deg_if_gt_th")),
                    "calf_r_swing_std_omega_if_gt_th": _as_float(calf.get("std_omega_deg_if_gt_th")),
                    "calf_r_swing_phase_amp_omega_if_gt_th": _as_float(calf.get("phase_amp_mean_omega_deg_if_gt_th")),
                    "error": r.get("error"),
                }
            )

    # Markdown summary (sorted by calf_r swing mean angle, then global GeoLocalDeg).
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    def _key_sort(rr: Dict[str, Any]) -> Tuple[float, float]:
        m = (rr.get("metrics") or {}) if isinstance(rr.get("metrics"), dict) else {}
        g = (m.get("global_r1p") or {}) if isinstance(m.get("global_r1p"), dict) else {}
        calf = (m.get("calf_r_swing_direct_r1p") or {}) if isinstance(m.get("calf_r_swing_direct_r1p"), dict) else {}
        a = _as_float(calf.get("mean_ang_deg"))
        gl = _as_float(g.get("GeoLocalDeg"))
        return (float(a) if a is not None else 1e9, float(gl) if gl is not None else 1e9)

    ok_rows = sorted(ok_rows, key=_key_sort)
    md_path = out_root / "sweep_summary.md"
    lines: List[str] = []
    lines.append(f"# SweepDirectPoseHinge ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    lines.append("")
    lines.append(f"- out_root: `{out_root}`")
    lines.append(f"- posttrain_config: `{cfg_path}`")
    lines.append(f"- teacher: `{', '.join(str(x) for x in args.teacher)}`")
    lines.append("")
    lines.append("| lr | rc | spe | GeoLocalDeg(R1+) | DirectGeoLocalDeg(R1+) | BlendGeoLocalDeg(R1+) | calf_r swing mean_ang | P(ang>th) | mean_omega@>th | phase_amp_omega@>th | run_name |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in ok_rows:
        m = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
        g = m.get("global_r1p") if isinstance(m.get("global_r1p"), dict) else {}
        calf = m.get("calf_r_swing_direct_r1p") if isinstance(m.get("calf_r_swing_direct_r1p"), dict) else {}
        def _fmt(x: Any, d: int = 3) -> str:
            v = _as_float(x)
            return "-" if v is None else f"{v:.{d}f}"
        def _fmt_lr(x: Any) -> str:
            v = _as_float(x)
            if v is None:
                return "-"
            try:
                return _fmt_lr_slug(float(v))
            except Exception:
                return f"{float(v):.3g}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt_lr(r.get("lr")),
                    str(r.get("rollout_cycles")),
                    str(r.get("steps_per_epoch")),
                    _fmt(g.get("GeoLocalDeg"), 4),
                    _fmt(g.get("DirectGeoLocalDeg"), 4),
                    _fmt(g.get("BlendGeoLocalDeg"), 4),
                    _fmt(calf.get("mean_ang_deg"), 2),
                    _fmt(calf.get("p_ang_gt_th"), 3),
                    _fmt(calf.get("mean_omega_deg_if_gt_th"), 2),
                    _fmt(calf.get("phase_amp_mean_omega_deg_if_gt_th"), 2),
                    str(r.get("run_name")),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {md_path}")
    print(f"[OK] wrote {out_root / 'sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
