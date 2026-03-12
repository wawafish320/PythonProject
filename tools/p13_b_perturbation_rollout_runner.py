#!/usr/bin/env python3
"""
Run B Test (Perturbation Rollout) for H2 checkpoints.

Implementation notes:
  - Uses train.validate.run_freerun_cycles internals directly.
  - Applies a t0 perturbation on sample["motion"] (X stream) before rollout.
  - Keeps target stream fixed by default (optional --perturb-target-t0).
  - Runs true free-run (no --freerun_x_gt, no cycle reset flags).
"""

from __future__ import annotations

import argparse
import json
import math
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

from train.validate import run_freerun_cycles as fr


DEFAULT_CLIPS = ["Walk_F", "Walk_L_To_L", "Walk_L_To_R", "Walk_R_To_L", "Walk_R_To_R"]
DEFAULT_K_LIST = [-2, -1, 0, 1, 2]


@dataclass(frozen=True)
class RunSpec:
    seed: int
    clip: str
    k_shift: int
    checkpoint: Path
    teacher_json: Path


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


def _parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for tok in str(spec or "").replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def _make_runner_args(*, ckpt: Path, npz_root: Path, out_root: Path) -> argparse.Namespace:
    argv_bak = list(sys.argv)
    try:
        # parse_args() defines all defaults used by FreeRunCycleRunner.
        sys.argv = [
            "p13_b_runner",
            "--teacher",
            str(_REPO_ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"),
            "--model",
            str(ckpt),
            "--npz-root",
            str(npz_root),
            "--out",
            str(out_root),
        ]
        args = fr.parse_args()
    finally:
        sys.argv = argv_bak
    return args


def _build_base_sample(
    *,
    runner: fr.FreeRunCycleRunner,
    teacher_json: Path,
    npz_root: Path,
) -> Tuple[str, Dict[str, Any], Path]:
    data = fr._load_json(teacher_json)
    clip_name = str(data.get("clip") or teacher_json.stem.replace("_teacher", ""))
    teacher_block = data.get("teacher")
    if not isinstance(teacher_block, dict):
        raise RuntimeError(f"{teacher_json}: missing teacher payload")

    state_arr = np.asarray(teacher_block.get("state_norm"), dtype=np.float32)
    cond_arr = np.asarray(teacher_block.get("cond"), dtype=np.float32)
    if state_arr.ndim != 2 or cond_arr.ndim != 2:
        raise RuntimeError(f"{teacher_json}: invalid state/cond shape")
    t_base = int(state_arr.shape[0])

    npz_path = fr._resolve_npz_path(clip_name, data.get("source_json"), npz_root)
    ds = runner._build_dataset(npz_path, seq_len=t_base)
    runner._ensure_model_ready(ds)
    clip = ds.clips[0]
    sample = fr._build_full_cycle_sample(ds, clip, seq_len=t_base)
    return clip_name, sample, npz_path


def _clone_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        elif isinstance(v, (dict, list, tuple)):
            out[k] = json.loads(json.dumps(v))
        else:
            out[k] = v
    return out


def _perturb_sample_t0(
    *,
    sample: Dict[str, Any],
    trainer: Any,
    k_shift: int,
    mode: str,
    perturb_target_t0: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = _clone_sample(sample)
    motion = out.get("motion")
    gt_motion = out.get("gt_motion")
    if not torch.is_tensor(motion) or motion.dim() != 2:
        raise RuntimeError("sample.motion missing or invalid")
    if not torch.is_tensor(gt_motion) or gt_motion.dim() != 2:
        raise RuntimeError("sample.gt_motion missing or invalid")

    t_all = int(motion.shape[0])
    if t_all <= 1:
        raise RuntimeError("sample too short for perturbation")
    src_idx = int(k_shift) % int(t_all)
    if int(k_shift) == 0:
        return out, {"k_shift": 0, "src_index": 0, "mode": str(mode), "target_perturbed": bool(perturb_target_t0)}

    mode_l = str(mode or "full_frame").strip().lower()
    if mode_l not in ("full_frame", "rot6d_only"):
        raise RuntimeError(f"Unsupported mode={mode}")

    if mode_l == "full_frame":
        motion[0] = motion[src_idx]
    else:
        rx = getattr(trainer, "rot6d_x_slice", None) or getattr(trainer, "rot6d_slice", None)
        if not isinstance(rx, slice):
            raise RuntimeError("Cannot resolve rot6d slice for motion")
        motion[0, rx] = motion[src_idx, rx]

    if bool(perturb_target_t0):
        if mode_l == "full_frame":
            gt_motion[0] = gt_motion[src_idx]
        else:
            ry = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
            if not isinstance(ry, slice):
                raise RuntimeError("Cannot resolve rot6d slice for gt_motion")
            gt_motion[0, ry] = gt_motion[src_idx, ry]

    return out, {
        "k_shift": int(k_shift),
        "src_index": int(src_idx),
        "mode": mode_l,
        "target_perturbed": bool(perturb_target_t0),
    }


def _extract_error_curve(
    *,
    per_step: Sequence[Dict[str, Any]],
    metric_key: str,
    bone: str,
    horizon: int,
    cycle0_only: bool,
    drop_wrap: bool,
) -> List[float]:
    out: List[float] = []
    for st in per_step:
        if not isinstance(st, dict):
            continue
        if cycle0_only and int(st.get("cycle", -1)) != 0:
            continue
        if drop_wrap and bool(st.get("wrap_boundary_step", False)):
            continue

        v: Optional[float] = None
        mk = str(metric_key)
        if mk.startswith("KeyBone"):
            d = st.get(mk, None)
            if isinstance(d, dict) and (bone in d):
                try:
                    vv = float(d.get(bone))
                    if math.isfinite(vv):
                        v = vv
                except Exception:
                    v = None
        else:
            try:
                vv = float(st.get(mk, float("nan")))
                if math.isfinite(vv):
                    v = vv
            except Exception:
                v = None

        if v is None:
            continue
        out.append(float(v))
        if int(horizon) > 0 and len(out) >= int(horizon):
            break

    min_need = max(8, int(max(0, horizon) * 0.25)) if int(horizon) > 0 else 8
    if len(out) < min_need:
        raise RuntimeError(
            f"Too few valid curve points: got={len(out)}, need>={min_need}, metric={metric_key}, bone={bone}"
        )
    return out


def _mean(x: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in x if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _linear_slope(y: Sequence[float]) -> Optional[float]:
    arr = np.asarray([float(v) for v in y], dtype=np.float64)
    if arr.size < 2:
        return None
    x = np.arange(arr.size, dtype=np.float64)
    try:
        coef = np.polyfit(x, arr, deg=1)
        slope = float(coef[0])
        if math.isfinite(slope):
            return slope
    except Exception:
        pass
    return None


def _mean_curve(curves: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not curves:
        return None
    min_len = min(len(c) for c in curves)
    if min_len <= 0:
        return None
    arr = np.asarray([np.asarray(c[:min_len], dtype=np.float64) for c in curves], dtype=np.float64)
    return [float(x) for x in arr.mean(axis=0).tolist()]


def _half_life(delta_curve: Sequence[float], eps: float = 1e-6) -> Optional[int]:
    if not delta_curve:
        return None
    d0 = float(abs(delta_curve[0]))
    if d0 <= float(eps):
        return 0
    thr = 0.5 * d0
    for i, v in enumerate(delta_curve):
        if abs(float(v)) <= thr:
            return int(i)
    return None


def _aggregate(rows: Sequence[Dict[str, Any]], k_list: Sequence[int]) -> Dict[str, Any]:
    rows_by_k: Dict[int, List[Dict[str, Any]]] = {int(k): [] for k in k_list}
    for r in rows:
        kk = int(r.get("k_shift"))
        if kk in rows_by_k:
            rows_by_k[kk].append(r)

    mean_curve_by_k: Dict[int, Optional[List[float]]] = {}
    for k in k_list:
        curves = [rr.get("error_curve_deg", []) for rr in rows_by_k[int(k)] if isinstance(rr.get("error_curve_deg"), list)]
        mean_curve_by_k[int(k)] = _mean_curve(curves)

    base_curve = mean_curve_by_k.get(0)
    if base_curve is None:
        raise RuntimeError("Missing k=0 mean curve")

    per_k: Dict[str, Any] = {}
    nonzero_slopes_pos: List[float] = []
    nonzero_end_pos: List[float] = []
    divergent_flags: List[int] = []
    divergent_no_recover: List[int] = []

    for k in k_list:
        k_int = int(k)
        mc = mean_curve_by_k.get(k_int)
        if mc is None:
            per_k[str(k_int)] = {
                "n_runs": int(len(rows_by_k[k_int])),
                "curve_len": 0,
                "error_growth_rate_deg_per_step": None,
                "delta_growth_rate_deg_per_step": None,
                "e0_deg": None,
                "e_end_deg": None,
                "delta0_deg": None,
                "delta_end_deg": None,
                "delta_abs_area_deg": None,
                "recovery_half_life": None,
            }
            continue

        common_len = min(len(base_curve), len(mc))
        curve = np.asarray(mc[:common_len], dtype=np.float64)
        base = np.asarray(base_curve[:common_len], dtype=np.float64)
        delta = curve - base

        slope = _linear_slope(curve.tolist())
        slope_delta = _linear_slope(delta.tolist())
        area_abs = float(np.mean(np.abs(delta))) if delta.size > 0 else None
        rec_hl = _half_life(delta.tolist())

        rec = {
            "n_runs": int(len(rows_by_k[k_int])),
            "curve_len": int(common_len),
            "error_growth_rate_deg_per_step": slope,
            "delta_growth_rate_deg_per_step": slope_delta,
            "e0_deg": float(curve[0]) if curve.size > 0 else None,
            "e_end_deg": float(curve[-1]) if curve.size > 0 else None,
            "delta0_deg": float(delta[0]) if delta.size > 0 else None,
            "delta_end_deg": float(delta[-1]) if delta.size > 0 else None,
            "delta_abs_area_deg": area_abs,
            "recovery_half_life": rec_hl,
        }
        per_k[str(k_int)] = rec

        if k_int != 0:
            if slope_delta is not None and math.isfinite(float(slope_delta)):
                nonzero_slopes_pos.append(max(0.0, float(slope_delta)))
            d_end = rec.get("delta_end_deg")
            if d_end is not None and math.isfinite(float(d_end)):
                de = float(d_end)
                nonzero_end_pos.append(max(0.0, de))
                is_div = int(de > 0.0)
                divergent_flags.append(is_div)
                if is_div:
                    divergent_no_recover.append(1 if rec_hl is None else 0)

    asym_pairs: List[Dict[str, Any]] = []
    pos_ks = sorted({int(k) for k in k_list if int(k) > 0 and int(-k) in k_list})
    for kp in pos_ks:
        p = per_k.get(str(int(kp)), {})
        n = per_k.get(str(int(-kp)), {})
        p_area = p.get("delta_abs_area_deg")
        n_area = n.get("delta_abs_area_deg")
        p_end = p.get("delta_end_deg")
        n_end = n.get("delta_end_deg")
        area_diff = None
        if p_area is not None and n_area is not None:
            area_diff = float(p_area) - float(n_area)
        end_abs_diff = None
        if p_end is not None and n_end is not None:
            end_abs_diff = abs(float(p_end)) - abs(float(n_end))
        asym_pairs.append(
            {
                "abs_k": int(kp),
                "+k": int(kp),
                "-k": int(-kp),
                "delta_abs_area_diff_deg": area_diff,
                "delta_end_abs_diff_deg": end_abs_diff,
            }
        )

    asym_area_diffs = [float(x["delta_abs_area_diff_deg"]) for x in asym_pairs if x.get("delta_abs_area_diff_deg") is not None]
    asym_end_diffs = [float(x["delta_end_abs_diff_deg"]) for x in asym_pairs if x.get("delta_end_abs_diff_deg") is not None]

    mean_pos_slope = _mean(nonzero_slopes_pos)
    mean_pos_end = _mean(nonzero_end_pos)
    divergent_k_rate = _mean(divergent_flags)
    no_recover_rate = _mean(divergent_no_recover) if divergent_no_recover else 0.0

    gate_thresholds = {
        "mean_pos_delta_growth_rate_min": 0.002,
        "mean_pos_delta_end_min": 0.20,
        "divergent_k_rate_min": 0.50,
        "no_recovery_rate_min": 0.50,
    }
    gate_support_b = bool(
        (mean_pos_slope is not None)
        and (mean_pos_end is not None)
        and (divergent_k_rate is not None)
        and (no_recover_rate is not None)
        and (float(mean_pos_slope) >= float(gate_thresholds["mean_pos_delta_growth_rate_min"]))
        and (float(mean_pos_end) >= float(gate_thresholds["mean_pos_delta_end_min"]))
        and (float(divergent_k_rate) >= float(gate_thresholds["divergent_k_rate_min"]))
        and (float(no_recover_rate) >= float(gate_thresholds["no_recovery_rate_min"]))
    )

    return {
        "n_runs": int(len(rows)),
        "per_k": per_k,
        "mean_curve_deg": {str(int(k)): mean_curve_by_k[int(k)] for k in k_list},
        "error_growth_rate_pos_delta_mean": mean_pos_slope,
        "error_delta_end_pos_mean": mean_pos_end,
        "divergent_k_rate": divergent_k_rate,
        "no_recovery_rate": no_recover_rate,
        "asymmetric_k": {
            "pairs": asym_pairs,
            "delta_abs_area_diff_deg_mean": _mean(asym_area_diffs),
            "delta_abs_area_diff_deg_abs_mean": _mean([abs(x) for x in asym_area_diffs]) if asym_area_diffs else None,
            "delta_end_abs_diff_deg_mean": _mean(asym_end_diffs),
            "delta_end_abs_diff_deg_abs_mean": _mean([abs(x) for x in asym_end_diffs]) if asym_end_diffs else None,
        },
        "gate_thresholds": gate_thresholds,
        "gate_support_B": gate_support_b,
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
    lines.append("# B Test Summary (Perturbation Rollout)")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- out_root: `{summary['config']['out_root']}`")
    lines.append(f"- case: `{summary['config']['case']}` | seeds: `{summary['config']['seeds']}`")
    lines.append(f"- clips: `{summary['config']['clips']}`")
    lines.append(f"- k_list: `{summary['config']['k_list']}`")
    lines.append(f"- horizon: `{summary['config']['horizon']}` steps (cycle0 only)")
    lines.append(f"- metric: `{summary['config']['metric_key']}` (bone=`{summary['config']['bone']}`)")
    lines.append("- rollout mode: true free-run (no `--freerun_x_gt`, no cycle-start reset flags)")
    lines.append(
        f"- perturbation: mode=`{summary['config']['perturb_mode']}`, target_t0_perturbed={bool(summary['config']['perturb_target_t0'])}"
    )
    lines.append("")

    lines.append("## Per Run")
    lines.append("| seed | clip | k | curve_len | e0 | e_end | slope |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {int(r['seed'])} | {r['clip']} | {int(r['k_shift'])} | {int(r['curve_len'])} | "
            f"{_fmt(r['e0_deg'])} | {_fmt(r['e_end_deg'])} | {_fmt(r['error_growth_rate_deg_per_step'])} |"
        )
    lines.append("")

    lines.append("## Aggregate By k")
    lines.append("| k | n_runs | e0 | e_end | slope | delta_end | delta_slope | half_life | delta_abs_area |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in summary["config"]["k_list"]:
        p = agg["per_k"].get(str(int(k)), {})
        lines.append(
            f"| {int(k)} | {int(p.get('n_runs', 0))} | {_fmt(p.get('e0_deg'))} | {_fmt(p.get('e_end_deg'))} | "
            f"{_fmt(p.get('error_growth_rate_deg_per_step'))} | {_fmt(p.get('delta_end_deg'))} | "
            f"{_fmt(p.get('delta_growth_rate_deg_per_step'))} | {p.get('recovery_half_life', 'nan')} | {_fmt(p.get('delta_abs_area_deg'))} |"
        )
    lines.append("")

    lines.append("## Core Metrics")
    lines.append(f"- `error_growth_rate_pos_delta_mean={_fmt(agg.get('error_growth_rate_pos_delta_mean'))}`")
    lines.append(f"- `error_delta_end_pos_mean={_fmt(agg.get('error_delta_end_pos_mean'))}`")
    lines.append(f"- `divergent_k_rate={_fmt(agg.get('divergent_k_rate'))}`")
    lines.append(f"- `no_recovery_rate={_fmt(agg.get('no_recovery_rate'))}`")
    lines.append(
        f"- `asymmetric_k.delta_abs_area_diff_deg_abs_mean={_fmt(agg.get('asymmetric_k', {}).get('delta_abs_area_diff_deg_abs_mean'))}`"
    )
    lines.append(f"- `gate_support_B={bool(agg.get('gate_support_B', False))}`")
    lines.append("")

    lines.append("## Notes")
    lines.append("- `recovery_half_life` is measured on `|mean_curve(k)-mean_curve(0)|` halving time.")
    lines.append("- `asymmetric_k` compares `+k` and `-k` under equal `|k|`.")
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run B-test perturbation rollout and summarize.")
    ap.add_argument(
        "--summary-json",
        type=str,
        default="debug_output/h1_10p3a_20260213/p10_phase2b_h2a_minimal_20260213/phase2b_h2a_minimal_summary.json",
    )
    ap.add_argument("--case", type=str, default="H2")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--clips", type=str, default=",".join(DEFAULT_CLIPS))
    ap.add_argument("--k-list", type=str, default=",".join(str(k) for k in DEFAULT_K_LIST))
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--bone", type=str, default="calf_r")
    ap.add_argument("--metric-key", type=str, default="KeyBoneGeoLocalDeg")
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--perturb-mode", type=str, default="full_frame", choices=["full_frame", "rot6d_only"])
    ap.add_argument("--perturb-target-t0", action="store_true")
    ap.add_argument("--drop-wrap", action="store_true", default=True)
    ap.add_argument("--keep-wrap", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    npz_root = _REPO_ROOT / "raw_data" / "processed_data"

    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    clips = [str(x).strip() for x in str(args.clips).split(",") if str(x).strip()]
    k_list = _parse_int_list(str(args.k_list))
    if not clips:
        raise SystemExit("[FATAL] clips is empty.")
    if not k_list:
        raise SystemExit("[FATAL] k-list is empty.")
    if 0 not in set(k_list):
        raise SystemExit("[FATAL] k-list must include 0 for baseline.")

    ckpt_by_seed = _parse_ckpts_from_summary(Path(args.summary_json).expanduser(), case=str(args.case), seeds=seeds)

    specs: List[RunSpec] = []
    for s in seeds:
        ckpt = ckpt_by_seed[int(s)]
        for c in clips:
            teacher_json = _REPO_ROOT / "validate" / "teacher_batches" / f"{c}_teacher.json"
            if not teacher_json.exists():
                raise SystemExit(f"[FATAL] missing teacher file: {teacher_json}")
            for k in k_list:
                specs.append(
                    RunSpec(
                        seed=int(s),
                        clip=str(c),
                        k_shift=int(k),
                        checkpoint=ckpt,
                        teacher_json=teacher_json,
                    )
                )

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    drop_wrap = bool(args.drop_wrap) and (not bool(args.keep_wrap))

    # Build one runner per seed/checkpoint to avoid repeated model init.
    runner_by_seed: Dict[int, fr.FreeRunCycleRunner] = {}
    base_sample_cache: Dict[Tuple[int, str], Tuple[str, Dict[str, Any], Path]] = {}
    for seed in seeds:
        ckpt = ckpt_by_seed[int(seed)]
        runner_args = _make_runner_args(ckpt=ckpt, npz_root=npz_root, out_root=out_root)
        runner_by_seed[int(seed)] = fr.FreeRunCycleRunner(runner_args)

    for sp in specs:
        run_dir = out_root / f"{args.case}_seed{sp.seed}" / sp.clip / f"k_{sp.k_shift:+d}"
        fr_dir = run_dir / "freerun"
        fr_dir.mkdir(parents=True, exist_ok=True)
        out_json = fr_dir / f"{sp.clip}_freerun_cycles.json"

        runner = runner_by_seed[int(sp.seed)]
        cache_key = (int(sp.seed), str(sp.clip))
        try:
            if cache_key not in base_sample_cache:
                base_sample_cache[cache_key] = _build_base_sample(
                    runner=runner,
                    teacher_json=sp.teacher_json,
                    npz_root=npz_root,
                )
            clip_name, base_sample, npz_path = base_sample_cache[cache_key]

            sample_k, perturb_meta = _perturb_sample_t0(
                sample=base_sample,
                trainer=runner.trainer,
                k_shift=int(sp.k_shift),
                mode=str(args.perturb_mode),
                perturb_target_t0=bool(args.perturb_target_t0),
            )

            metrics_per_round, per_step, extra = fr._run_freerun_cycles(
                trainer=runner.trainer,
                sample=sample_k,
                rounds=int(args.rounds),
                device=runner.device,
                time_index_mode=str(getattr(runner.args, "time_index_mode", "auto") or "auto"),
                time_index_cycle_minus1=bool(getattr(runner.args, "time_index_cycle_minus1", False)),
                lambda_fusion_apply=bool(getattr(runner, "lambda_fusion_apply", False)),
                # Keep true free-run defaults:
                multicycle_sync_state_on_cycle_start=False,
                multicycle_reset_plan_z_on_cycle_start=False,
                multicycle_reset_pose_hist_on_cycle_start=False,
                freerun_x_gt=False,
                freerun_x_gt_except_rot6d=False,
            )

            payload = {
                "clip": str(clip_name),
                "seed": int(sp.seed),
                "k_shift": int(sp.k_shift),
                "checkpoint": str(sp.checkpoint),
                "teacher_json": str(sp.teacher_json),
                "npz_path": str(npz_path),
                "perturbation": perturb_meta,
                "metrics_per_round": metrics_per_round,
                "metrics_per_step": per_step,
                **extra,
            }
            out_json.write_text(json.dumps(payload, indent=2))

            curve = _extract_error_curve(
                per_step=per_step,
                metric_key=str(args.metric_key),
                bone=str(args.bone),
                horizon=int(args.horizon),
                cycle0_only=True,
                drop_wrap=drop_wrap,
            )
            slope = _linear_slope(curve)

            row = {
                "seed": int(sp.seed),
                "clip": str(sp.clip),
                "k_shift": int(sp.k_shift),
                "checkpoint": str(sp.checkpoint),
                "json": str(out_json),
                "curve_len": int(len(curve)),
                "e0_deg": float(curve[0]) if curve else None,
                "e_end_deg": float(curve[-1]) if curve else None,
                "error_growth_rate_deg_per_step": slope,
                "error_curve_deg": [float(v) for v in curve],
            }
            rows.append(row)
            (run_dir / "b_test_result.json").write_text(json.dumps(row, indent=2))
        except Exception as exc:  # noqa: BLE001
            fail = {
                "seed": int(sp.seed),
                "clip": str(sp.clip),
                "k_shift": int(sp.k_shift),
                "error": str(exc),
            }
            failures.append(fail)
            (run_dir / "failure.json").write_text(json.dumps(fail, indent=2))

    summary = {
        "config": {
            "summary_json": str(Path(args.summary_json).expanduser()),
            "case": str(args.case),
            "seeds": [int(s) for s in seeds],
            "clips": clips,
            "k_list": [int(k) for k in k_list],
            "rounds": int(args.rounds),
            "bone": str(args.bone),
            "metric_key": str(args.metric_key),
            "horizon": int(args.horizon),
            "perturb_mode": str(args.perturb_mode),
            "perturb_target_t0": bool(args.perturb_target_t0),
            "drop_wrap": bool(drop_wrap),
            "out_root": str(out_root),
            "rollout_mode": "true_free_run",
        },
        "rows": rows,
        "failures": failures,
        "aggregate": _aggregate(rows, k_list=k_list) if rows else {},
    }

    out_json = out_root / "b_test_perturbation_rollout_summary.json"
    out_md = out_root / "b_test_perturbation_rollout_summary.md"
    out_json.write_text(json.dumps(summary, indent=2))
    _write_md(summary, out_md)

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")
    if failures:
        print(f"[WARN] failures={len(failures)}")


if __name__ == "__main__":
    main()
