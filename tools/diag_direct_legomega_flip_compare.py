#!/usr/bin/env python3
"""
Diagnostics for Stage7 direct leg omega (SO(3)) alpha-sweep outputs.

This script supports 3 questions discussed in:
  docs/Problems/active/2026-01-26_WalkF_stage7_leg_so3_omega_corrector_contact_transition_signflip_overshoot.md

1) Baseline "L/R mixing" vs routed/shared:
   - On (step, bone) points that are flips in a chosen source (baseline or routed/shared),
     compare baseline vs routed theta_pred_deg / theta_oracle_deg / cos_pred_oracle.

2) End-effector-ish error comparison on flip steps:
   - On steps containing at least one flip point, compare per-step KeyBone* errors between baseline and routed.
   NOTE: metrics_per_step currently exports rotation errors (KeyBone*GeoDeg/GeoLocalDeg). Global position errors
         are not exported by default.

3) Baseline behavior on its own flip points:
   - On baseline strict-flip points, summarize theta_pred_deg, theta_oracle_deg, pred/oracle ratios.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _isfinite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _get_float(d: Dict[str, Any], key: str) -> float:
    """Best-effort float parsing; returns NaN for missing/None/bad values."""
    try:
        v = d.get(key, float("nan"))
        if v is None:
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _parse_csv(spec: str) -> List[str]:
    return [t.strip() for t in str(spec or "").split(",") if t.strip()]


def _stats(x: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0.0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "n": float(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
    }


def _fmt_stats(d: Dict[str, float], unit: str = "") -> str:
    u = f" {unit}" if unit else ""
    return (
        f"n={int(d['n'])} mean={d['mean']:.4f}{u} p50={d['p50']:.4f}{u} "
        f"p90={d['p90']:.4f}{u} p95={d['p95']:.4f}{u}"
    )


@dataclass(frozen=True)
class StepKey:
    cycle: int
    step_in_cycle: int
    step: int


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _alpha_steps(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    sw = obj.get("direct_leg_omega_alpha_sweep", None)
    if not isinstance(sw, dict):
        raise SystemExit("Missing direct_leg_omega_alpha_sweep in JSON.")
    steps = sw.get("steps", None)
    if not isinstance(steps, list):
        raise SystemExit("Invalid direct_leg_omega_alpha_sweep: missing steps list.")
    return steps


def _metrics_steps(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    ms = obj.get("metrics_per_step", None)
    if not isinstance(ms, list):
        raise SystemExit("Missing metrics_per_step in JSON.")
    return ms


def _step_key(st: Dict[str, Any]) -> StepKey:
    return StepKey(
        cycle=int(st.get("cycle", 0) or 0),
        step_in_cycle=int(st.get("step_in_cycle", -1) or -1),
        step=int(st.get("step", -1) or -1),
    )


def _iter_flip_points(
    steps: List[Dict[str, Any]],
    *,
    bones: Sequence[str],
    mode: str,
) -> Iterable[Tuple[int, str]]:
    """
    Yields (step_index, bone_name) for flip points.

    mode:
      - strict: cos<0 && best_alpha<0
      - cos:    cos<0
    """
    mode = str(mode or "strict").strip().lower()
    if mode not in ("strict", "cos"):
        raise ValueError(f"Unknown flip mode: {mode}")
    for i, st in enumerate(steps):
        pb = st.get("per_bone", None)
        if not isinstance(pb, dict):
            continue
        for bone in bones:
            dat = pb.get(bone, None)
            if not isinstance(dat, dict):
                continue
            cos = _get_float(dat, "cos_pred_oracle")
            if not _isfinite(cos) or cos >= 0.0:
                continue
            if mode == "cos":
                yield i, str(bone)
                continue
            best = _get_float(dat, "best_alpha")
            if _isfinite(best) and best < 0.0:
                yield i, str(bone)


def _iter_flip_steps(
    steps: List[Dict[str, Any]],
    *,
    bones: Sequence[str],
    mode: str,
) -> Iterable[int]:
    seen: set[int] = set()
    for i, _bone in _iter_flip_points(steps, bones=bones, mode=mode):
        st = steps[i]
        step = int(st.get("step", -1) or -1)
        if step not in seen:
            seen.add(step)
            yield step


def _get_keybone_err(entry: Dict[str, Any], metric_key: str, bone: str) -> Optional[float]:
    d = entry.get(metric_key, None)
    if not isinstance(d, dict):
        return None
    v = d.get(bone, None)
    if v is None:
        return None
    try:
        vv = float(v)
    except Exception:
        return None
    return vv if math.isfinite(vv) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline vs routed/shared direct leg omega alpha-sweep flips.")
    ap.add_argument("--baseline-json", type=str, required=True)
    ap.add_argument("--routed-json", type=str, required=True)
    ap.add_argument("--flip-source", type=str, default="routed", choices=("baseline", "routed"))
    ap.add_argument("--flip-mode", type=str, default="strict", choices=("strict", "cos"))
    ap.add_argument(
        "--flip-bones",
        type=str,
        default="calf_l,foot_l,ball_l,calf_r,foot_r,ball_r",
        help="Bones used to define flip points/steps.",
    )
    ap.add_argument("--print-n", type=int, default=30, help="Print top-N flip points (sorted by routed cos asc).")
    ap.add_argument("--out-csv", type=str, default="", help="Optional path to write per-point comparison CSV.")

    # Diagnostic 2: per-step (FK-ish) error comparison using metrics_per_step.
    ap.add_argument(
        "--fk-metric",
        type=str,
        default="KeyBoneDirectGeoDeg",
        help="Per-step keybone metric to compare (e.g., KeyBoneDirectGeoDeg, KeyBoneGeoDeg).",
    )
    ap.add_argument("--fk-bones", type=str, default="foot_l,foot_r,ball_l,ball_r")
    ap.add_argument("--fk-agg", type=str, default="mean", choices=("mean", "max"))
    args = ap.parse_args()

    p_base = Path(args.baseline_json).expanduser()
    p_rout = Path(args.routed_json).expanduser()
    base = _load_json(p_base)
    rout = _load_json(p_rout)

    base_steps = _alpha_steps(base)
    rout_steps = _alpha_steps(rout)
    if len(base_steps) != len(rout_steps):
        raise SystemExit(f"Alpha-sweep steps mismatch: baseline={len(base_steps)} routed={len(rout_steps)}")

    # Validate alignment; if not aligned, rebuild by StepKey intersection.
    aligned = True
    for i in range(len(base_steps)):
        if _step_key(base_steps[i]) != _step_key(rout_steps[i]):
            aligned = False
            break
    if not aligned:
        base_map = {_step_key(st): st for st in base_steps if isinstance(st, dict)}
        rout_map = {_step_key(st): st for st in rout_steps if isinstance(st, dict)}
        common = sorted(set(base_map.keys()) & set(rout_map.keys()), key=lambda k: (k.cycle, k.step_in_cycle, k.step))
        if not common:
            raise SystemExit("No overlapping alpha-sweep steps between baseline and routed JSONs.")
        base_steps = [base_map[k] for k in common]
        rout_steps = [rout_map[k] for k in common]

    flip_bones = _parse_csv(args.flip_bones)
    if not flip_bones:
        raise SystemExit("--flip-bones resolved to empty set.")

    flip_steps_src = base_steps if args.flip_source == "baseline" else rout_steps
    flip_points = list(_iter_flip_points(flip_steps_src, bones=flip_bones, mode=args.flip_mode))

    print(f"baseline_json: {p_base}")
    print(f"routed_json:   {p_rout}")
    print(f"flip_source={args.flip_source} flip_mode={args.flip_mode} flip_bones={','.join(flip_bones)}")
    print(f"flip_points={len(flip_points)} (per-(step,bone) points in alpha-sweep window)")

    # ---- Diagnostic 1: per-point comparison ----
    cnt_by_bone = Counter()
    baseline_strict_on_points = 0

    b_theta: List[float] = []
    r_theta: List[float] = []
    o_theta: List[float] = []
    b_cos: List[float] = []
    r_cos: List[float] = []
    ratio_b: List[float] = []
    ratio_r: List[float] = []

    rows: List[Dict[str, Any]] = []
    for i, bone in flip_points:
        sb = base_steps[i]
        sr = rout_steps[i]
        pb = sb.get("per_bone", None)
        pr = sr.get("per_bone", None)
        if not isinstance(pb, dict) or not isinstance(pr, dict):
            continue
        db = pb.get(bone, None)
        dr = pr.get(bone, None)
        if not isinstance(db, dict) or not isinstance(dr, dict):
            continue

        k = _step_key(sr)
        cnt_by_bone[str(bone)] += 1

        bt = _get_float(db, "theta_pred_deg")
        rt = _get_float(dr, "theta_pred_deg")
        ot = _get_float(dr, "theta_oracle_deg")
        bc = _get_float(db, "cos_pred_oracle")
        rc = _get_float(dr, "cos_pred_oracle")
        bb = _get_float(db, "best_alpha")
        rb = _get_float(dr, "best_alpha")

        if _isfinite(bt):
            b_theta.append(bt)
        if _isfinite(rt):
            r_theta.append(rt)
        if _isfinite(ot):
            o_theta.append(ot)
        if _isfinite(bc):
            b_cos.append(bc)
        if _isfinite(rc):
            r_cos.append(rc)
        if _isfinite(bt) and _isfinite(ot) and ot > 0.0:
            ratio_b.append(bt / ot)
        if _isfinite(rt) and _isfinite(ot) and ot > 0.0:
            ratio_r.append(rt / ot)

        if _isfinite(bc) and bc < 0.0 and _isfinite(bb) and bb < 0.0:
            baseline_strict_on_points += 1

        rows.append(
            {
                "cycle": k.cycle,
                "step_in_cycle": k.step_in_cycle,
                "step": k.step,
                "bone": str(bone),
                "b_theta_pred_deg": bt if _isfinite(bt) else None,
                "b_cos": bc if _isfinite(bc) else None,
                "b_best_alpha": bb if _isfinite(bb) else None,
                "r_theta_pred_deg": rt if _isfinite(rt) else None,
                "r_theta_oracle_deg": ot if _isfinite(ot) else None,
                "r_cos": rc if _isfinite(rc) else None,
                "r_best_alpha": rb if _isfinite(rb) else None,
            }
        )

    print("[Diag1] Per-point omega stats on flip points")
    if cnt_by_bone:
        print("  flip_points_by_bone:", ", ".join([f"{k}:{v}" for k, v in cnt_by_bone.most_common()]))
    print(f"  baseline_strict_flip_on_these_points: {baseline_strict_on_points}/{len(rows)}")
    print(f"  baseline theta_pred_deg: {_fmt_stats(_stats(b_theta), unit='deg')}")
    print(f"  routed   theta_pred_deg: {_fmt_stats(_stats(r_theta), unit='deg')}")
    print(f"  oracle   theta_oracle_deg: {_fmt_stats(_stats(o_theta), unit='deg')}")
    print(f"  baseline cos_pred_oracle: {_fmt_stats(_stats(b_cos))}")
    print(f"  routed   cos_pred_oracle: {_fmt_stats(_stats(r_cos))}")
    print(f"  baseline pred/oracle ratio: {_fmt_stats(_stats(ratio_b))}")
    print(f"  routed   pred/oracle ratio: {_fmt_stats(_stats(ratio_r))}")

    if rows and int(args.print_n) > 0:
        def sort_key(rr: Dict[str, Any]) -> Tuple[float, float]:
            rc = rr.get("r_cos")
            ot = rr.get("r_theta_oracle_deg")
            rc_f = float(rc) if rc is not None else 0.0
            ot_f = float(ot) if ot is not None else 0.0
            return (rc_f, -ot_f)

        rows_sorted = sorted(rows, key=sort_key)
        print(f"  top{int(args.print_n)} flip points (sorted by r_cos asc):")
        for rr in rows_sorted[: int(args.print_n)]:
            print(
                f"    step={rr['step']} sic={rr['step_in_cycle']} bone={rr['bone']} | "
                f"b_theta={rr['b_theta_pred_deg']} b_cos={rr['b_cos']} | "
                f"r_theta={rr['r_theta_pred_deg']} r_cos={rr['r_cos']} | "
                f"oracle={rr['r_theta_oracle_deg']}"
            )

    if str(args.out_csv or "").strip():
        out = Path(args.out_csv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "cycle",
            "step_in_cycle",
            "step",
            "bone",
            "b_theta_pred_deg",
            "b_cos",
            "b_best_alpha",
            "r_theta_pred_deg",
            "r_theta_oracle_deg",
            "r_cos",
            "r_best_alpha",
        ]
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for rr in rows:
                w.writerow({k: rr.get(k) for k in cols})
        print(f"  wrote csv: {out}")

    # ---- Diagnostic 2: per-step keybone errors on flip steps ----
    fk_bones = _parse_csv(args.fk_bones)
    ms_base = _metrics_steps(base)
    ms_rout = _metrics_steps(rout)
    mb = {int(e.get("step", -1) or -1): e for e in ms_base if isinstance(e, dict)}
    mr = {int(e.get("step", -1) or -1): e for e in ms_rout if isinstance(e, dict)}

    flip_steps = [s for s in _iter_flip_steps(flip_steps_src, bones=flip_bones, mode=args.flip_mode) if s in mb and s in mr]

    def agg(entry: Dict[str, Any]) -> Optional[float]:
        vals: List[float] = []
        for bname in fk_bones:
            v = _get_keybone_err(entry, str(args.fk_metric), bname)
            if v is not None:
                vals.append(v)
        if not vals:
            return None
        if str(args.fk_agg) == "max":
            return float(max(vals))
        return float(sum(vals) / len(vals))

    cmp_rows: List[Tuple[int, float, float]] = []
    for step in flip_steps:
        vb = agg(mb[step])
        vr = agg(mr[step])
        if vb is None or vr is None:
            continue
        cmp_rows.append((int(step), float(vb), float(vr)))

    print("[Diag2] Per-step keybone errors on flip steps")
    print(f"  fk_metric={args.fk_metric} fk_bones={','.join(fk_bones)} fk_agg={args.fk_agg}")
    print(f"  flip_steps={len(flip_steps)}")
    if cmp_rows:
        vb = np.asarray([x[1] for x in cmp_rows], dtype=np.float64)
        vr = np.asarray([x[2] for x in cmp_rows], dtype=np.float64)
        better = int((vb < vr - 1e-9).sum())
        worse = int((vb > vr + 1e-9).sum())
        same = int(vb.size - better - worse)
        print(f"  baseline_better/worse/same: {better}/{worse}/{same}")
        print(f"  mean baseline={float(vb.mean()):.4f} routed={float(vr.mean()):.4f}")
    else:
        print("  (no comparable per-step entries; check fk_metric/fk_bones)")

    # ---- Diagnostic 3: baseline strict-flip points magnitude behavior ----
    bones_all: List[str] = []
    bones_seen: set[str] = set()
    for st in base_steps:
        pb = st.get("per_bone", None)
        if not isinstance(pb, dict):
            continue
        for bname in pb.keys():
            bn = str(bname)
            if bn not in bones_seen:
                bones_seen.add(bn)
                bones_all.append(bn)
    bones_all = sorted(bones_all)

    base_flip_points = list(_iter_flip_points(base_steps, bones=bones_all, mode="strict"))
    b_pred: List[float] = []
    b_orc: List[float] = []
    b_ratio: List[float] = []
    for i, bone in base_flip_points:
        st = base_steps[i]
        pb = st.get("per_bone", None)
        if not isinstance(pb, dict):
            continue
        dat = pb.get(bone, None)
        if not isinstance(dat, dict):
            continue
        bt = _get_float(dat, "theta_pred_deg")
        ot = _get_float(dat, "theta_oracle_deg")
        if _isfinite(bt):
            b_pred.append(bt)
        if _isfinite(ot):
            b_orc.append(ot)
        if _isfinite(bt) and _isfinite(ot) and ot > 0.0:
            b_ratio.append(bt / ot)

    print("[Diag3] Baseline strict-flip points magnitude")
    print(f"  baseline_strict_flip_points={len(base_flip_points)}")
    print(f"  theta_pred_deg: {_fmt_stats(_stats(b_pred), unit='deg')}")
    print(f"  theta_oracle_deg: {_fmt_stats(_stats(b_orc), unit='deg')}")
    print(f"  pred/oracle ratio: {_fmt_stats(_stats(b_ratio))}")


if __name__ == "__main__":
    main()

