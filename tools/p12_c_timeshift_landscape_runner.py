#!/usr/bin/env python3
"""
Run C Test (Target Time-Shift Loss Landscape) for H2 checkpoints.

Protocol:
  - Evaluation aperture matches p10/p11 proxy: --freerun_x_gt + --multicycle-reset-plan-z-on-cycle-start
  - For each seed/clip:
      * run_freerun_cycles with --debug_direct_alignment (k in [-2,2])
      * export keybone_state + keybone_omega series for calf_r
      * reconstruct R_gt from (R_pred, R_err) and compute L(k)=geo(R_pred[t], R_gt[t+k])
      * compute curvature = L(+1)+L(-1)-2L(0)
      * stratify by |omega_gt| terciles (low/mid/high)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.geometry import geodesic_R, so3_exp_map


DEFAULT_CLIPS = ["Walk_F", "Walk_L_To_L", "Walk_L_To_R", "Walk_R_To_L", "Walk_R_To_R"]
K_LIST = [-2, -1, 0, 1, 2]


@dataclass(frozen=True)
class RunSpec:
    seed: int
    clip: str
    checkpoint: Path


def _run(cmd: Sequence[str], cwd: Path) -> str:
    env = os.environ.copy()
    py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "." if not py else f".:{py}"
    print("[CMD]", " ".join(str(x) for x in cmd))
    cp = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return cp.stdout


def _parse_ckpts_from_summary(summary_path: Path, case: str, seeds: Sequence[int]) -> Dict[int, Path]:
    obj = json.loads(summary_path.read_text())
    rows = obj.get("rows", [])
    out: Dict[int, Path] = {}
    for r in rows:
        if str(r.get("case")) != str(case):
            continue
        s = int(r.get("seed"))
        if s not in seeds:
            continue
        ckpt = Path(str(r.get("checkpoint"))).expanduser()
        out[s] = ckpt
    missing = [int(s) for s in seeds if int(s) not in out]
    if missing:
        raise RuntimeError(f"Missing checkpoints in summary for case={case}, seeds={missing}.")
    return out


def _mean(x: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in x if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _std(x: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in x if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0 if vals else None
    return float(np.std(np.asarray(vals, dtype=np.float64), ddof=0))


def _extract_direct_alignment_lk(payload: Dict[str, Any]) -> Dict[int, Optional[float]]:
    out: Dict[int, Optional[float]] = {k: None for k in K_LIST}
    da = payload.get("direct_alignment", {})
    ts = da.get("time_shift_noncyc", {})
    rows = ts.get("results", []) if isinstance(ts, dict) else []
    if not isinstance(rows, list):
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            k = int(r.get("shift"))
        except Exception:
            continue
        if k in out:
            v = r.get("geo_local_deg_mean", None)
            out[k] = None if (v is None) else float(v)
    return out


def _to_tensor_deg_xyz(x: Sequence[Sequence[float]]) -> torch.Tensor:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[-1] != 3:
        raise RuntimeError(f"Expected (T,3), got shape={arr.shape}")
    return torch.from_numpy(arr).float()


def _extract_keybone_series(payload: Dict[str, Any], bone: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    ks = payload.get("keybone_state", {}).get("series", {})
    ko = payload.get("keybone_omega", {}).get("series", {})
    pred = (
        ks.get("branches", {})
        .get("direct", {})
        .get("pred_rotvec_deg_xyz", {})
        .get(str(bone), None)
    )
    err = (
        ko.get("branches", {})
        .get("direct", {})
        .get("omega_deg_xyz", {})
        .get(str(bone), None)
    )
    if pred is None or err is None:
        raise RuntimeError(f"Missing keybone series for bone={bone}.")
    pred_deg = _to_tensor_deg_xyz(pred)
    err_deg = _to_tensor_deg_xyz(err)
    if pred_deg.shape != err_deg.shape:
        raise RuntimeError(f"Shape mismatch pred={tuple(pred_deg.shape)} err={tuple(err_deg.shape)}")

    steps = payload.get("metrics_per_step", [])
    if not isinstance(steps, list) or len(steps) != int(pred_deg.shape[0]):
        raise RuntimeError("metrics_per_step length mismatch with keybone series.")
    valid = np.ones((len(steps),), dtype=bool)
    for i, st in enumerate(steps):
        if isinstance(st, dict) and bool(st.get("wrap_boundary_step", False)):
            valid[i] = False
    return pred_deg, err_deg, valid


def _lk_from_reconstructed_keybone(
    *,
    pred_deg: torch.Tensor,
    err_deg: torch.Tensor,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    deg2rad = float(math.pi / 180.0)
    pred_r = pred_deg * deg2rad
    err_r = err_deg * deg2rad

    # R_err = R_pred^T @ R_gt  =>  R_gt = R_pred @ R_err
    R_pred = so3_exp_map(pred_r)  # (T,3,3)
    R_err = so3_exp_map(err_r)
    R_gt = torch.matmul(R_pred, R_err)

    T = int(R_pred.shape[0])
    vals_k: Dict[int, List[float]] = {k: [] for k in K_LIST}
    speed_all: List[float] = []
    speed_at_k0: List[float] = []
    rows_for_bins: Dict[int, List[Tuple[float, float]]] = {k: [] for k in K_LIST}

    # |omega_gt| at g index uses backward difference: geo(R_gt[g-1], R_gt[g]) in deg.
    omega_gt = np.full((T,), np.nan, dtype=np.float64)
    if T >= 2:
        w = geodesic_R(R_gt[:-1], R_gt[1:]) * (180.0 / math.pi)
        omega_gt[1:] = w.detach().cpu().numpy().astype(np.float64)

    valid = np.asarray(valid_mask, dtype=bool)
    for k in K_LIST:
        for t in range(T):
            g = t + int(k)
            if g < 0 or g >= T:
                continue
            if (not valid[t]) or (not valid[g]):
                continue
            d = geodesic_R(R_pred[t : t + 1], R_gt[g : g + 1]) * (180.0 / math.pi)
            v = float(d.item())
            vals_k[k].append(v)

            sp = float(omega_gt[g]) if math.isfinite(float(omega_gt[g])) else float("nan")
            if math.isfinite(sp):
                rows_for_bins[k].append((sp, v))
                speed_all.append(sp)
                if k == 0:
                    speed_at_k0.append(sp)

    lk = {k: _mean(vals_k[k]) for k in K_LIST}
    counts = {k: int(len(vals_k[k])) for k in K_LIST}
    curv = None
    if all(lk.get(k) is not None for k in (-1, 0, 1)):
        curv = float(lk[1] + lk[-1] - 2.0 * lk[0])
    d_p1 = None if (lk.get(1) is None or lk.get(0) is None) else float(lk[1] - lk[0])
    d_m1 = None if (lk.get(-1) is None or lk.get(0) is None) else float(lk[-1] - lk[0])
    flat_abs = None
    if d_p1 is not None and d_m1 is not None:
        flat_abs = float(0.5 * (abs(d_p1) + abs(d_m1)))

    # Terciles from k=0 target-speed distribution (fallback to all speeds if empty).
    base_speed = np.asarray(speed_at_k0 if speed_at_k0 else speed_all, dtype=np.float64)
    q1 = q2 = None
    bins_out: Dict[str, Any] = {}
    if base_speed.size >= 6:
        q1 = float(np.quantile(base_speed, 1.0 / 3.0))
        q2 = float(np.quantile(base_speed, 2.0 / 3.0))
        bin_defs = {
            "low": lambda s: s <= q1,
            "mid": lambda s: (s > q1) and (s <= q2),
            "high": lambda s: s > q2,
        }
        for bname, pred_fn in bin_defs.items():
            b_lk: Dict[int, Optional[float]] = {}
            b_n: Dict[int, int] = {}
            for k in K_LIST:
                vv = [v for s, v in rows_for_bins[k] if pred_fn(float(s))]
                b_lk[k] = _mean(vv)
                b_n[k] = int(len(vv))
            b_curv = None
            if all(b_lk.get(k) is not None for k in (-1, 0, 1)):
                b_curv = float(b_lk[1] + b_lk[-1] - 2.0 * b_lk[0])
            bins_out[bname] = {
                "Lk_deg": {str(k): b_lk[k] for k in K_LIST},
                "N": {str(k): int(b_n[k]) for k in K_LIST},
                "curvature_deg": b_curv,
                "delta_p1_deg": None if (b_lk.get(1) is None or b_lk.get(0) is None) else float(b_lk[1] - b_lk[0]),
                "delta_m1_deg": None if (b_lk.get(-1) is None or b_lk.get(0) is None) else float(b_lk[-1] - b_lk[0]),
            }

    return {
        "Lk_deg": {str(k): lk[k] for k in K_LIST},
        "N": {str(k): int(counts[k]) for k in K_LIST},
        "curvature_deg": curv,
        "delta_p1_deg": d_p1,
        "delta_m1_deg": d_m1,
        "flat_abs_deg": flat_abs,
        "omega_tercile_q1_deg": q1,
        "omega_tercile_q2_deg": q2,
        "omega_bins": bins_out,
    }


def _weighted_aggregate_bins(run_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Aggregate keybone omega bins across runs with count-weighted means per k.
    out: Dict[str, Any] = {}
    for bname in ("low", "mid", "high"):
        lk: Dict[int, Optional[float]] = {}
        nn: Dict[int, int] = {}
        for k in K_LIST:
            num = 0.0
            den = 0
            for r in run_rows:
                b = r.get("keybone", {}).get("omega_bins", {}).get(bname, {})
                v = b.get("Lk_deg", {}).get(str(k), None)
                n = b.get("N", {}).get(str(k), 0)
                if v is None:
                    continue
                try:
                    vv = float(v)
                    ni = int(n)
                except Exception:
                    continue
                if not math.isfinite(vv) or ni <= 0:
                    continue
                num += vv * ni
                den += ni
            lk[k] = (num / den) if den > 0 else None
            nn[k] = int(den)

        curv = None
        if all(lk.get(k) is not None for k in (-1, 0, 1)):
            curv = float(lk[1] + lk[-1] - 2.0 * lk[0])
        out[bname] = {
            "Lk_deg": {str(k): lk[k] for k in K_LIST},
            "N": {str(k): int(nn[k]) for k in K_LIST},
            "curvature_deg": curv,
            "delta_p1_deg": None if (lk.get(1) is None or lk.get(0) is None) else float(lk[1] - lk[0]),
            "delta_m1_deg": None if (lk.get(-1) is None or lk.get(0) is None) else float(lk[-1] - lk[0]),
        }
    return out


def _aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def _collect(path: Sequence[str]) -> List[float]:
        out_vals: List[float] = []
        for r in rows:
            cur: Any = r
            ok = True
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    ok = False
                    break
                cur = cur[p]
            if not ok or cur is None:
                continue
            try:
                v = float(cur)
            except Exception:
                continue
            if math.isfinite(v):
                out_vals.append(v)
        return out_vals

    lk_agg: Dict[str, Any] = {}
    for k in K_LIST:
        vals = _collect(["keybone", "Lk_deg", str(k)])
        lk_agg[str(k)] = {"mean": _mean(vals), "std": _std(vals), "n_runs": int(len(vals))}

    curv_vals = _collect(["keybone", "curvature_deg"])
    d1_vals = _collect(["keybone", "delta_p1_deg"])
    dm1_vals = _collect(["keybone", "delta_m1_deg"])
    flat_vals = _collect(["keybone", "flat_abs_deg"])
    da_flat_vals = _collect(["direct_alignment", "flat_abs_deg"])

    by_seed: Dict[str, Dict[str, Any]] = {}
    seeds = sorted(set(int(r.get("seed")) for r in rows))
    for s in seeds:
        rr = [r for r in rows if int(r.get("seed")) == int(s)]
        by_seed[str(s)] = {
            "n_runs": int(len(rr)),
            "L0_mean": _mean([float(r["keybone"]["Lk_deg"]["0"]) for r in rr if r["keybone"]["Lk_deg"]["0"] is not None]),
            "delta_p1_mean": _mean([float(r["keybone"]["delta_p1_deg"]) for r in rr if r["keybone"]["delta_p1_deg"] is not None]),
            "delta_m1_mean": _mean([float(r["keybone"]["delta_m1_deg"]) for r in rr if r["keybone"]["delta_m1_deg"] is not None]),
            "curvature_mean": _mean([float(r["keybone"]["curvature_deg"]) for r in rr if r["keybone"]["curvature_deg"] is not None]),
            "flat_abs_mean": _mean([float(r["keybone"]["flat_abs_deg"]) for r in rr if r["keybone"]["flat_abs_deg"] is not None]),
        }

    flat_abs_mean = _mean(flat_vals)
    curv_mean = _mean(curv_vals)
    gate_support_c = bool(
        (flat_abs_mean is not None)
        and (curv_mean is not None)
        and (flat_abs_mean <= 0.05)
        and (curv_mean <= 0.10)
    )

    return {
        "n_runs": int(len(rows)),
        "keybone_Lk_deg": lk_agg,
        "curvature_deg_mean": curv_mean,
        "curvature_deg_std": _std(curv_vals),
        "delta_p1_deg_mean": _mean(d1_vals),
        "delta_m1_deg_mean": _mean(dm1_vals),
        "flat_abs_deg_mean": flat_abs_mean,
        "flat_abs_deg_std": _std(flat_vals),
        "direct_alignment_flat_abs_deg_mean": _mean(da_flat_vals),
        "gate_support_C_flat": gate_support_c,
        "by_seed": by_seed,
        "omega_bins_weighted": _weighted_aggregate_bins(rows),
        "gate_thresholds": {"flat_abs_deg_max": 0.05, "curvature_deg_max": 0.10},
    }


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "nan"
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def _write_md(summary: Dict[str, Any], out_md: Path) -> None:
    rows = summary["rows"]
    agg = summary["aggregate"]
    lines: List[str] = []
    lines.append("# C Test Summary (Target Time-Shift Loss Landscape)")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- out_root: `{summary['config']['out_root']}`")
    lines.append(f"- case: `{summary['config']['case']}` | seeds: `{summary['config']['seeds']}`")
    lines.append(f"- clips: `{summary['config']['clips']}`")
    lines.append("- aperture: `--freerun_x_gt --multicycle-reset-plan-z-on-cycle-start`")
    lines.append("- k range: `[-2,-1,0,1,2]` (non-circular)")
    lines.append("- keybone: `calf_r`")
    lines.append("")

    lines.append("## Per Run (Keybone-Reconstructed)")
    lines.append("| seed | clip | L(-1) | L(0) | L(+1) | d(+1) | d(-1) | curvature | flat_abs |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        k = r["keybone"]["Lk_deg"]
        lines.append(
            f"| {int(r['seed'])} | {r['clip']} | {_fmt(k['-1'])} | {_fmt(k['0'])} | {_fmt(k['1'])} | "
            f"{_fmt(r['keybone']['delta_p1_deg'])} | {_fmt(r['keybone']['delta_m1_deg'])} | "
            f"{_fmt(r['keybone']['curvature_deg'])} | {_fmt(r['keybone']['flat_abs_deg'])} |"
        )
    lines.append("")

    lines.append("## Aggregate")
    lines.append(
        f"- `flat_abs_deg_mean={_fmt(agg['flat_abs_deg_mean'])}` "
        f"(thr<={_fmt(agg['gate_thresholds']['flat_abs_deg_max'])})"
    )
    lines.append(
        f"- `curvature_deg_mean={_fmt(agg['curvature_deg_mean'])}` "
        f"(thr<={_fmt(agg['gate_thresholds']['curvature_deg_max'])})"
    )
    lines.append(f"- `gate_support_C_flat={bool(agg['gate_support_C_flat'])}`")
    lines.append("")

    lines.append("## Omega-Bin Weighted (low/mid/high)")
    lines.append("| bin | L(-1) | L(0) | L(+1) | d(+1) | d(-1) | curvature |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for b in ("low", "mid", "high"):
        br = agg["omega_bins_weighted"].get(b, {})
        lk = br.get("Lk_deg", {})
        lines.append(
            f"| {b} | {_fmt(lk.get('-1'))} | {_fmt(lk.get('0'))} | {_fmt(lk.get('1'))} | "
            f"{_fmt(br.get('delta_p1_deg'))} | {_fmt(br.get('delta_m1_deg'))} | {_fmt(br.get('curvature_deg'))} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append(
        "- Keybone-reconstructed L(k) uses exported `pred_rotvec_deg_xyz` + `omega_deg_xyz` to recover `R_gt` per step."
    )
    lines.append("- `flat_abs = 0.5*(|L(+1)-L(0)| + |L(-1)-L(0)|)`.")
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run C-test timeshift landscape and summarize.")
    ap.add_argument(
        "--summary-json",
        type=str,
        default="debug_output/h1_10p3a_20260213/p10_phase2b_h2a_minimal_20260213/phase2b_h2a_minimal_summary.json",
    )
    ap.add_argument("--case", type=str, default="H2")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--clips", type=str, default=",".join(DEFAULT_CLIPS))
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--bone", type=str, default="calf_r")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    clips = [str(x).strip() for x in str(args.clips).split(",") if str(x).strip()]
    if not clips:
        raise SystemExit("[FATAL] clips is empty.")

    ckpt_by_seed = _parse_ckpts_from_summary(Path(args.summary_json).expanduser(), case=str(args.case), seeds=seeds)

    specs: List[RunSpec] = []
    for s in seeds:
        for c in clips:
            specs.append(RunSpec(seed=int(s), clip=str(c), checkpoint=ckpt_by_seed[int(s)]))

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sp in specs:
        teacher = repo / "validate" / "teacher_batches" / f"{sp.clip}_teacher.json"
        if not teacher.exists():
            failures.append({"seed": int(sp.seed), "clip": sp.clip, "error": f"missing teacher: {teacher}"})
            continue
        run_dir = out_root / f"{args.case}_seed{sp.seed}" / sp.clip
        fr_dir = run_dir / "freerun"
        fr_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python",
            "-m",
            "train.validate.run_freerun_cycles",
            "--teacher",
            str(teacher),
            "--model",
            str(sp.checkpoint),
            "--out",
            str(fr_dir),
            "--rounds",
            str(int(args.rounds)),
            "--freerun_x_gt",
            "--multicycle-reset-plan-z-on-cycle-start",
            "--debug_direct_alignment",
            "--direct_alignment_max_shift",
            "2",
            "--export_keybone_state_series",
            "--keybone_state_series_bones",
            str(args.bone),
            "--keybone_state_series_branches",
            "direct",
            "--export_keybone_omega",
            "--export_keybone_omega_series",
            "--keybone_omega_series_bones",
            str(args.bone),
            "--keybone_omega_series_axis",
            "z",
            "--force",
        ]
        try:
            stdout = _run(cmd, cwd=repo)
            (run_dir / "freerun.log").write_text(stdout)
            json_path = fr_dir / f"{sp.clip}_freerun_cycles.json"
            payload = json.loads(json_path.read_text())

            da_lk = _extract_direct_alignment_lk(payload)
            da_curv = None
            if all(da_lk.get(k) is not None for k in (-1, 0, 1)):
                da_curv = float(da_lk[1] + da_lk[-1] - 2.0 * da_lk[0])
            da_flat = None
            if da_lk.get(1) is not None and da_lk.get(0) is not None and da_lk.get(-1) is not None:
                da_flat = float(0.5 * (abs(da_lk[1] - da_lk[0]) + abs(da_lk[-1] - da_lk[0])))

            pred_deg, err_deg, valid = _extract_keybone_series(payload, bone=str(args.bone))
            keybone = _lk_from_reconstructed_keybone(pred_deg=pred_deg, err_deg=err_deg, valid_mask=valid)

            row = {
                "seed": int(sp.seed),
                "clip": str(sp.clip),
                "checkpoint": str(sp.checkpoint),
                "teacher": str(teacher),
                "json": str(json_path),
                "direct_alignment": {
                    "Lk_deg": {str(k): da_lk[k] for k in K_LIST},
                    "curvature_deg": da_curv,
                    "flat_abs_deg": da_flat,
                },
                "keybone": keybone,
            }
            rows.append(row)
            (run_dir / "c_test_result.json").write_text(json.dumps(row, indent=2))
        except subprocess.CalledProcessError as exc:
            fail = {
                "seed": int(sp.seed),
                "clip": str(sp.clip),
                "error": f"command failed ({exc.returncode})",
                "output_tail": (exc.stdout or "")[-4000:],
            }
            failures.append(fail)
            (run_dir / "failure.json").write_text(json.dumps(fail, indent=2))
        except Exception as exc:  # noqa: BLE001
            fail = {"seed": int(sp.seed), "clip": str(sp.clip), "error": str(exc)}
            failures.append(fail)
            (run_dir / "failure.json").write_text(json.dumps(fail, indent=2))

    summary = {
        "config": {
            "summary_json": str(Path(args.summary_json).expanduser()),
            "case": str(args.case),
            "seeds": [int(s) for s in seeds],
            "clips": clips,
            "rounds": int(args.rounds),
            "bone": str(args.bone),
            "out_root": str(out_root),
            "k_list": K_LIST,
            "aperture": ["--freerun_x_gt", "--multicycle-reset-plan-z-on-cycle-start"],
        },
        "rows": rows,
        "failures": failures,
        "aggregate": _aggregate(rows),
    }

    out_json = out_root / "c_test_timeshift_summary.json"
    out_md = out_root / "c_test_timeshift_summary.md"
    out_json.write_text(json.dumps(summary, indent=2))
    _write_md(summary, out_md)
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")
    if failures:
        print(f"[WARN] failures={len(failures)}")


if __name__ == "__main__":
    main()
