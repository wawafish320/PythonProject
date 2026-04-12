#!/usr/bin/env python3
"""
Evaluate the practical impact of D0 "concat contacts_meas" on freerun_cycles.

This script wraps `python -m train.validate.run_freerun_cycles` to run a small ablation matrix:
  - baseline ckpt (no D0)
  - D0 ckpt
  - D0 ckpt + `--direct_pose_meas_source=zero` (simulate "no meas hint" for direct)

Then it parses the generated `*_freerun_cycles.json` and prints a compact table for:
  - Round0 step0 / first10 / first20 / mean: DirectGeoLocalDeg
  - R1+ mean: DirectGeoLocalDeg
  - (optional) BlendGeoLocalDeg when lambda_fusion_apply is enabled

Typical usage (Walk_F):
  python tools/eval_d0_concat_ablation.py --force
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Case:
    label: str
    ckpt: Path
    direct_pose_meas_source: str = "model"  # model|gt|zero


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _mean_steps(
    steps: List[Dict[str, Any]],
    key: str,
    start: int,
    count: int,
) -> Optional[float]:
    if count <= 0:
        return None
    end = min(len(steps), start + count)
    return _mean(steps[i].get(key) for i in range(start, end))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def summarize_freerun_cycles(path: Path, *, label: str) -> Dict[str, Any]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict JSON at {path}, got {type(data)}")

    steps = data.get("metrics_per_step")
    rounds = data.get("metrics_per_round")
    if not isinstance(steps, list) or not isinstance(rounds, list) or not rounds:
        raise KeyError(f"{path} missing metrics_per_step/metrics_per_round")

    r0 = rounds[0]
    start_step = int(r0.get("start_step", 0) or 0)

    r1p_direct = _mean(_safe_float(r.get("DirectGeoLocalDeg")) for r in rounds[1:])
    r1p_blend = _mean(_safe_float(r.get("BlendGeoLocalDeg")) for r in rounds[1:])

    # NOTE: Older freerun JSONs used direct_pose_meas_force_zero; treat that as direct_pose_meas_source=zero.
    direct_meas_src = str(data.get("direct_pose_meas_source", "model") or "model").strip()
    if bool(data.get("direct_pose_meas_force_zero", False)) and direct_meas_src in ("", "model"):
        direct_meas_src = "zero"

    row: Dict[str, Any] = {
        "label": str(label),
        "clip": data.get("clip", path.name.split("_freerun_cycles.json")[0]),
        "model": data.get("model"),
        "direct_pose_meas_source": direct_meas_src,
        "lambda_fusion_apply": bool(data.get("lambda_fusion_apply", False)),
        "so3_corr_apply": bool(data.get("so3_corr_apply", False)) if data.get("so3_corr_apply") is not None else None,
        "out_json": str(path),
        # Direct (GeoLocalDeg, degrees)
        "R0_step0_DirectGeoLocalDeg": _safe_float(steps[start_step].get("DirectGeoLocalDeg"))
        if start_step < len(steps)
        else None,
        "R0_first10_DirectGeoLocalDeg": _mean_steps(steps, "DirectGeoLocalDeg", start_step, 10),
        "R0_first20_DirectGeoLocalDeg": _mean_steps(steps, "DirectGeoLocalDeg", start_step, 20),
        "R0_mean_DirectGeoLocalDeg": _safe_float(r0.get("DirectGeoLocalDeg")),
        "R1p_mean_DirectGeoLocalDeg": r1p_direct,
        # Blend (only meaningful when lambda_fusion_apply)
        "R0_mean_BlendGeoLocalDeg": _safe_float(r0.get("BlendGeoLocalDeg")),
        "R1p_mean_BlendGeoLocalDeg": r1p_blend,
    }
    return row


def _find_outputs(out_dir: Path) -> List[Path]:
    return sorted(out_dir.glob("*_freerun_cycles.json"))


def _fmt(x: Any, *, width: int = 8) -> str:
    v = _safe_float(x)
    if v is None:
        return " " * (width - 1) + "-"
    s = f"{v:.2f}"
    return s.rjust(width)


def _run_one_case(args: argparse.Namespace, case: Case, out_dir: Path) -> None:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "train.validate.run_freerun_cycles",
        "--teacher",
        str(Path(args.teacher).expanduser()),
        "--model",
        str(case.ckpt),
        "--out",
        str(out_dir),
        "--rounds",
        str(int(args.rounds)),
        "--depth",
        str(int(args.depth)),
        "--time-index-mode",
        str(args.time_index_mode),
        "--device",
        str(args.device),
        "--encoder-bundle",
        str(Path(args.encoder_bundle).expanduser()),
        "--bundle",
        str(Path(args.bundle).expanduser()),
        "--pretrain-template",
        str(Path(args.pretrain_template).expanduser()),
        "--npz-root",
        str(Path(args.npz_root).expanduser()),
    ]

    if args.direct_align_inc0:
        cmd.append("--direct_align_inc0")
    if args.lambda_fusion_apply:
        cmd.append("--lambda_fusion_apply")
    if args.so3_corr_apply:
        cmd.append("--so3_corr_apply")
    if args.log_contacts:
        cmd.append("--log_contacts")

    # Direct meas ablations
    if case.direct_pose_meas_source != "model":
        cmd.extend(["--direct_pose_meas_source", str(case.direct_pose_meas_source)])

    if args.force:
        cmd.append("--force")

    print(f"[run] {case.label} -> {out_dir}")
    subprocess.run(cmd, check=True)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(description="Run & summarize D0(concat meas) ablation via freerun_cycles.")
    ap.add_argument(
        "--teacher",
        type=str,
        default="validate/teacher_batches/Walk_F_teacher.json",
        help="Teacher json path (e.g. validate/teacher_batches/Walk_F_teacher.json).",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="debug_output/eval_d0_concat_ablation",
        help="Output root dir; each case writes into a subdir.",
    )
    ap.add_argument("--rounds", type=int, default=2, help="freerun_cycles rounds.")
    ap.add_argument("--depth", type=int, default=3, help="Model depth (must match ckpt training).")
    ap.add_argument("--time_index_mode", type=str, default="cycle", choices=("auto", "global", "cycle", "none"))
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain_template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder_bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--npz_root", type=str, default="raw_data/processed_data")

    ap.add_argument("--direct_align_inc0", action="store_true", help="Enable --direct_align_inc0 in freerun_cycles.")
    ap.add_argument("--lambda_fusion_apply", action="store_true", help="Enable --lambda_fusion_apply in freerun_cycles.")
    ap.add_argument("--so3_corr_apply", action="store_true", help="Enable --so3_corr_apply in freerun_cycles.")

    ap.add_argument("--log_contacts", action="store_true", help="Enable --log_contacts in freerun_cycles.")

    ap.add_argument("--base_ckpt", type=str, default="models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1/ckpt_best_teacher_exp_phase_DirectBranch_v1.pth")
    ap.add_argument("--d0_ckpt", type=str, default="models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d0/ckpt_best_teacher_exp_phase_DirectBranch_v1_d0.pth")
    ap.add_argument("--no_ablate_meas", action="store_true", help="Do not run the (force_zero) meas ablation case.")

    ap.add_argument("--no_run", action="store_true", help="Skip running freerun_cycles; only summarize existing outputs.")
    ap.add_argument("--force", action="store_true", help="Pass --force to freerun_cycles to overwrite existing outputs.")
    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser()
    base_ckpt = Path(args.base_ckpt).expanduser()
    d0_ckpt = Path(args.d0_ckpt).expanduser()

    cases: List[Case] = [
        Case(label="base", ckpt=base_ckpt),
        Case(label="d0", ckpt=d0_ckpt),
    ]
    if not args.no_ablate_meas:
        cases.append(Case(label="d0__no_meas", ckpt=d0_ckpt, direct_pose_meas_source="zero"))

    if not args.no_run:
        for c in cases:
            if not c.ckpt.exists():
                raise FileNotFoundError(f"Missing ckpt: {c.ckpt}")
            _run_one_case(args, c, out_root / c.label)

    # Summarize
    rows: List[Dict[str, Any]] = []
    for c in cases:
        out_dir = out_root / c.label
        outs = _find_outputs(out_dir)
        if not outs:
            raise FileNotFoundError(f"No *_freerun_cycles.json under {out_dir} (did the run succeed?)")
        for p in outs:
            rows.append(summarize_freerun_cycles(p, label=c.label))

    rows.sort(key=lambda r: (str(r.get("clip")), str(r.get("label"))))
    summary_json = out_root / "summary.json"
    _save_json(summary_json, rows)

    # CSV for quick spreadsheet diffing
    fieldnames = [
        "label",
        "clip",
        "direct_pose_meas_source",
        "lambda_fusion_apply",
        "so3_corr_apply",
        "R0_step0_DirectGeoLocalDeg",
        "R0_first10_DirectGeoLocalDeg",
        "R0_first20_DirectGeoLocalDeg",
        "R0_mean_DirectGeoLocalDeg",
        "R1p_mean_DirectGeoLocalDeg",
        "R0_mean_BlendGeoLocalDeg",
        "R1p_mean_BlendGeoLocalDeg",
        "model",
        "out_json",
    ]
    summary_csv = out_root / "summary.csv"
    _write_csv(summary_csv, rows, fieldnames)

    # Pretty console table
    print("\n[summary] (DirectGeoLocalDeg, degrees)")
    print(
        f"{'case':12s} {'clip':10s}"
        f"{'R0s0':>8s} {'R0f10':>8s} {'R0f20':>8s} {'R0mean':>8s} {'R1+mean':>8s}"
        f"{'B0mean':>8s} {'B1+':>8s}"
    )
    for r in rows:
        case = str(r.get("label", ""))[:12].ljust(12)
        clip = str(r.get("clip", ""))[:10].ljust(10)
        print(
            f"{case} {clip}"
            f"{_fmt(r.get('R0_step0_DirectGeoLocalDeg'))}"
            f"{_fmt(r.get('R0_first10_DirectGeoLocalDeg'))}"
            f"{_fmt(r.get('R0_first20_DirectGeoLocalDeg'))}"
            f"{_fmt(r.get('R0_mean_DirectGeoLocalDeg'))}"
            f"{_fmt(r.get('R1p_mean_DirectGeoLocalDeg'))}"
            f"{_fmt(r.get('R0_mean_BlendGeoLocalDeg'))}"
            f"{_fmt(r.get('R1p_mean_BlendGeoLocalDeg'))}"
        )

    print(f"\n[OK] wrote {summary_json}")
    print(f"[OK] wrote {summary_csv}")


if __name__ == "__main__":
    main()
