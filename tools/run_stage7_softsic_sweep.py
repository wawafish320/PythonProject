#!/usr/bin/env python3
"""
Quick Stage7 (7.0a/7.0b) soft SIC-weight sweep with tradeoff plots.

Sweep parameter:
  - direct_pose_loss_sics = "12-15"
  - direct_pose_loss_sic_mode = "boost"
  - direct_pose_loss_sic_boost = w

For each weight w:
  1) Train 7.0a from a Stage6 ckpt
  2) Free-run eval 7.0a
  3) Train 7.0b from 7.0a ckpt
  4) Free-run eval 7.0b

Outputs:
  - sweep_rows.csv
  - tradeoff_calf_global_vs_foot1215.png
  - tradeoff_calf_sic24_vs_foot1215.png
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _run(cmd: List[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(shlex.quote(x) for x in cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _weight_tag(w: float) -> str:
    s = f"{float(w):.3f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _calc_metrics(freerun_json: Path) -> Dict[str, float]:
    obj = _load_json(freerun_json)
    per = obj["per_step_direct_geolocal_deg"]
    names = list(per["bone_names"])
    root = int(per.get("root_idx", 0))
    mat = np.asarray(per["DirectGeoLocalDeg"], dtype=np.float64)
    steps = obj["metrics_per_step"]

    sics = np.zeros((len(steps),), dtype=np.int64)
    mask = np.zeros((len(steps),), dtype=bool)
    for i, st in enumerate(steps):
        cyc = int(st.get("cycle", 0))
        sic = int(st.get("step_in_cycle", st.get("sic", i)))
        wrap = bool(st.get("wrap_boundary_step", False))
        sics[i] = sic
        if cyc >= 1 and not wrap:
            mask[i] = True

    def mean_deg(bones: List[str] | None, sic_lo: int | None, sic_hi: int | None) -> float:
        m = mask.copy()
        if sic_lo is not None and sic_hi is not None:
            m &= (sics >= int(sic_lo)) & (sics <= int(sic_hi))
        if bones is None:
            idx = [i for i in range(len(names)) if i != root]
        else:
            idx = [int(names.index(b)) for b in bones]
        if not idx or not bool(m.any()):
            return float("nan")
        sub = mat[m][:, idx]
        vals = sub[np.isfinite(sub)]
        return float(vals.mean()) if vals.size > 0 else float("nan")

    return {
        "global_mean": mean_deg(None, None, None),
        "sic12_15_foot_l_ball_l": mean_deg(["foot_l", "ball_l"], 12, 15),
        "calf_r_global": mean_deg(["calf_r"], None, None),
        "calf_r_sic2_4": mean_deg(["calf_r"], 2, 4),
    }


def _apply_soft_sic(cfg: Dict, boost: float) -> None:
    cfg["direct_pose_loss_sics"] = "12-15"
    cfg["direct_pose_loss_cycle_gte"] = 1
    cfg["direct_pose_loss_sic_mode"] = "boost"
    cfg["direct_pose_loss_sic_boost"] = float(boost)


def _plot_tradeoff(rows: List[Dict], out_png: Path, y_key: str, y_label: str) -> None:
    stage_styles = {
        "s70a": {"color": "#1f77b4", "marker": "o"},
        "s70b": {"color": "#d62728", "marker": "s"},
    }

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for stage in ("s70a", "s70b"):
        pts = [r for r in rows if r["stage"] == stage]
        pts.sort(key=lambda r: r["weight"])
        xs = [r["sic12_15_foot_l_ball_l"] for r in pts]
        ys = [r[y_key] for r in pts]
        ws = [r["weight"] for r in pts]
        st = stage_styles[stage]
        ax.plot(xs, ys, marker=st["marker"], color=st["color"], linewidth=1.5, label=stage)
        for x, y, w in zip(xs, ys, ws):
            ax.annotate(f"{w:g}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=8)

    ax.set_xlabel("SIC12-15 foot_l/ball_l mean deg (lower better)")
    ax.set_ylabel(y_label)
    ax.set_title("Soft SIC12-15 boost sweep tradeoff")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        type=str,
        default="1.0,1.5,2.0,3.0",
        help="Comma-separated direct_pose_loss_sic_boost values.",
    )
    ap.add_argument(
        "--base-s70a-config",
        type=str,
        default="config/posttrain_WalkF_stage7_direct_tail_state_weight_nohinge_splitB2_pe32h512_full_20260225_nohingeadapt1_s70a.json",
    )
    ap.add_argument(
        "--base-s70b-config",
        type=str,
        default="config/posttrain_WalkF_stage7_direct_tail_state_weight_nohinge_phasezin_splitB2_pe32h512_full_20260225_nohingeadapt1_s70b.json",
    )
    ap.add_argument(
        "--stage6-ckpt",
        type=str,
        default="models/MLPL2_DirectBranch_v1__stage67_from_stage5_20260225_rerun1/ckpt_last_WalkF_stage6_direct_cond_anchor_nohinge_pe32_h512_20260224_froms5_rerun1.pth",
    )
    ap.add_argument(
        "--model-out-dir",
        type=str,
        default="models/MLPL2_DirectBranch_v1__stage67_softsic_sweep_20260225",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default="debug_output/verify_stage7_softsic_sweep_20260225",
    )
    ap.add_argument("--s70a-epochs", type=int, default=2)
    ap.add_argument("--s70a-steps", type=int, default=30)
    ap.add_argument("--s70b-epochs", type=int, default=1)
    ap.add_argument("--s70b-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cwd = Path.cwd()
    base_a = _load_json(Path(args.base_s70a_config))
    base_b = _load_json(Path(args.base_s70b_config))

    weights: List[float] = []
    for tok in str(args.weights).split(","):
        t = tok.strip()
        if not t:
            continue
        weights.append(float(t))
    if not weights:
        raise SystemExit("[FATAL] empty weights")

    out_root = Path(args.out_root)
    model_out = Path(args.model_out_dir)
    cfg_out_dir = out_root / "configs"
    eval_root = out_root / "eval"
    out_root.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)
    cfg_out_dir.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for w in weights:
        tag = _weight_tag(w)
        print(f"\n=== Sweep weight={w} (tag={tag}) ===")

        cfg_a = deepcopy(base_a)
        run_a = f"{cfg_a.get('run_name', 's70a')}_ss12b{tag}"
        cfg_a["run_name"] = run_a
        cfg_a["out_dir"] = model_out.as_posix()
        cfg_a["ckpt_in"] = Path(args.stage6_ckpt).as_posix()
        cfg_a["epochs"] = int(args.s70a_epochs)
        cfg_a["steps_per_epoch"] = int(args.s70a_steps)
        cfg_a["seed"] = int(args.seed)
        _apply_soft_sic(cfg_a, w)

        cfg_a_path = cfg_out_dir / f"s70a_ss12b{tag}.json"
        _save_json(cfg_a_path, cfg_a)
        _run(["python", "-m", "train.posttrain", "--config", cfg_a_path.as_posix()], cwd)

        ckpt_a = model_out / f"ckpt_last_{run_a}.pth"
        if not ckpt_a.is_file():
            raise SystemExit(f"[FATAL] missing ckpt: {ckpt_a}")

        eval_a_dir = eval_root / f"w_{tag}" / "s70a"
        _run(
            [
                "python",
                "-m",
                "train.validate.run_freerun_cycles",
                "--teacher",
                "validate/teacher_batches/Walk_F_teacher.json",
                "--model",
                ckpt_a.as_posix(),
                "--bundle",
                "raw_data/processed_data/norm_template.json",
                "--pretrain-template",
                "models/pretrain_template.json",
                "--encoder-bundle",
                "models/motion_encoder_equiv_stageA.pt",
                "--npz-root",
                "raw_data/processed_data",
                "--rounds",
                "5",
                "--time-index-mode",
                "cycle",
                "--time-index-cycle-minus1",
                "--event_clock",
                "auto",
                "--lambda_fusion_apply",
                "--so3_corr_apply",
                "--direct_pose_meas_source",
                "model",
                "--direct_pose_plan_source",
                "model",
                "--contacts_meas_source",
                "model",
                "--phase_reset_source",
                "none",
                "--export_joint_geolocal",
                "--export_joint_direct_geolocal_series",
                "--out",
                eval_a_dir.as_posix(),
                "--force",
            ],
            cwd,
        )
        metrics_a = _calc_metrics(eval_a_dir / "Walk_F_freerun_cycles.json")
        rows.append({"stage": "s70a", "weight": float(w), **metrics_a})

        cfg_b = deepcopy(base_b)
        run_b = f"{cfg_b.get('run_name', 's70b')}_ss12b{tag}"
        cfg_b["run_name"] = run_b
        cfg_b["out_dir"] = model_out.as_posix()
        cfg_b["ckpt_in"] = ckpt_a.as_posix()
        cfg_b["epochs"] = int(args.s70b_epochs)
        cfg_b["steps_per_epoch"] = int(args.s70b_steps)
        cfg_b["seed"] = int(args.seed)
        _apply_soft_sic(cfg_b, w)

        cfg_b_path = cfg_out_dir / f"s70b_ss12b{tag}.json"
        _save_json(cfg_b_path, cfg_b)
        _run(["python", "-m", "train.posttrain", "--config", cfg_b_path.as_posix()], cwd)

        ckpt_b = model_out / f"ckpt_last_{run_b}.pth"
        if not ckpt_b.is_file():
            raise SystemExit(f"[FATAL] missing ckpt: {ckpt_b}")

        eval_b_dir = eval_root / f"w_{tag}" / "s70b"
        _run(
            [
                "python",
                "-m",
                "train.validate.run_freerun_cycles",
                "--teacher",
                "validate/teacher_batches/Walk_F_teacher.json",
                "--model",
                ckpt_b.as_posix(),
                "--bundle",
                "raw_data/processed_data/norm_template.json",
                "--pretrain-template",
                "models/pretrain_template.json",
                "--encoder-bundle",
                "models/motion_encoder_equiv_stageA.pt",
                "--npz-root",
                "raw_data/processed_data",
                "--rounds",
                "5",
                "--time-index-mode",
                "cycle",
                "--time-index-cycle-minus1",
                "--event_clock",
                "auto",
                "--lambda_fusion_apply",
                "--so3_corr_apply",
                "--direct_pose_meas_source",
                "model",
                "--direct_pose_plan_source",
                "model",
                "--contacts_meas_source",
                "model",
                "--phase_reset_source",
                "none",
                "--export_joint_geolocal",
                "--export_joint_direct_geolocal_series",
                "--out",
                eval_b_dir.as_posix(),
                "--force",
            ],
            cwd,
        )
        metrics_b = _calc_metrics(eval_b_dir / "Walk_F_freerun_cycles.json")
        rows.append({"stage": "s70b", "weight": float(w), **metrics_b})

    rows.sort(key=lambda r: (r["stage"], r["weight"]))
    csv_path = out_root / "sweep_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "weight",
                "global_mean",
                "sic12_15_foot_l_ball_l",
                "calf_r_global",
                "calf_r_sic2_4",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    _plot_tradeoff(
        rows,
        out_root / "tradeoff_calf_global_vs_foot1215.png",
        y_key="calf_r_global",
        y_label="calf_r global mean deg (lower better)",
    )
    _plot_tradeoff(
        rows,
        out_root / "tradeoff_calf_sic24_vs_foot1215.png",
        y_key="calf_r_sic2_4",
        y_label="calf_r SIC2-4 mean deg (lower better)",
    )

    print(f"[OK] wrote: {csv_path.as_posix()}")
    print(f"[OK] wrote: {(out_root / 'tradeoff_calf_global_vs_foot1215.png').as_posix()}")
    print(f"[OK] wrote: {(out_root / 'tradeoff_calf_sic24_vs_foot1215.png').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
