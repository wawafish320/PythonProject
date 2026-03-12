#!/usr/bin/env python3
"""
Analyze a contact signal (plan/meas) behavior from `run_freerun_cycles` JSON outputs.

This is meant to debug failure modes like:
  - plan collapses to always-left (never predicts right stance)
  - contacts_meas is phase-shifted vs GT contacts (often a drift symptom)
  - low amplitude (near-constant / ~0.5) so threshold-crossing events jitter
  - L/R identity confusion (swap improves stance accuracy / MSE)

Input JSON requirements:
  - Run `python -m train.validate.run_freerun_cycles --log_contacts ...`
  - The JSON must contain per-step keys:
      - ContactGTPerC
      - ContactPlanPerC (when --pred-source plan)
      - ContactMeasPerC (when --pred-source meas)

Example
-------
python tools/analyze_freerun_contact_plan.py \\
  --json debug_output/_tmp_wb_compare/v1_d1_lbnohist_v1/baseline_model/Walk_F_freerun_cycles.json \\
  --exclude-round0 \\
  --cycle-len 87
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _as_vec2(x: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(x, list) or len(x) < 2:
        return None
    try:
        return float(x[0]), float(x[1])
    except Exception:
        return None


def _shift_mse(a: np.ndarray, b: np.ndarray, *, max_shift: int) -> Tuple[int, int, float, float]:
    """
    Find shift that minimizes MSE between a[t] and b[t+shift] (over overlap).
    Returns (best_shift_signed, best_shift_mod, mse0, mse_best).
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    T = int(min(a.size, b.size))
    a = a[:T]
    b = b[:T]
    max_shift = int(max(0, max_shift))
    shifts = list(range(-max_shift, max_shift + 1))
    mse_best = None
    best_shift = 0
    for s in shifts:
        if s >= 0:
            x = a[: T - s]
            y = b[s:]
        else:
            x = a[-s:]
            y = b[: T + s]
        if x.size <= 0 or y.size <= 0:
            continue
        mse = float(((x - y) ** 2).mean())
        if mse_best is None or mse < mse_best:
            mse_best = mse
            best_shift = int(s)
    if mse_best is None:
        mse_best = float("nan")
    mse0 = float(((a - b) ** 2).mean()) if T > 0 else float("nan")
    # shift_mod: wrap into [0,cycle_len) is handled by caller (needs cycle_len).
    return int(best_shift), int(best_shift), float(mse0), float(mse_best)


def _fmt(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze contacts_plan from freerun_cycles JSON.")
    ap.add_argument("--json", type=str, required=True, help="Path to *_freerun_cycles.json")
    ap.add_argument(
        "--pred-source",
        type=str,
        default="plan",
        choices=("plan", "meas"),
        help="Which predicted contacts to analyze vs GT: plan=ContactPlanPerC, meas=ContactMeasPerC.",
    )
    ap.add_argument("--exclude-round0", action="store_true", help="Use cycles>=1 only.")
    ap.add_argument(
        "--cycle-len",
        type=int,
        default=None,
        help="Override cycle_len (defaults to JSON cycle_len). Used for per-phase stats + shift_mod.",
    )
    ap.add_argument("--max-shift", type=int, default=40, help="Max shift (frames) to search for plan-vs-GT alignment.")
    ap.add_argument(
        "--single_support_thr",
        type=float,
        default=0.7,
        help="GT contact threshold to define single-support (max>=thr and min<=1-thr).",
    )
    args = ap.parse_args()

    path = Path(args.json).expanduser()
    obj = _load_json(path)
    steps = obj.get("metrics_per_step", None)
    if not isinstance(steps, list) or not steps:
        raise SystemExit("Invalid JSON: missing metrics_per_step list.")

    cycle_len = int(args.cycle_len or obj.get("cycle_len", 0) or 0)

    pred_key = "ContactPlanPerC" if str(args.pred_source).strip().lower() == "plan" else "ContactMeasPerC"
    pred_name = "Plan" if pred_key == "ContactPlanPerC" else "Meas"

    gt: List[Tuple[float, float]] = []
    pred: List[Tuple[float, float]] = []
    step_in_cycle: List[int] = []

    for st in steps:
        cy = st.get("cycle", None)
        if bool(args.exclude_round0) and isinstance(cy, int) and cy == 0:
            continue
        g = _as_vec2(st.get("ContactGTPerC", None))
        p = _as_vec2(st.get(pred_key, None))
        if g is None or p is None:
            continue
        gt.append(g)
        pred.append(p)
        si = st.get("step_in_cycle", None)
        step_in_cycle.append(int(si) if isinstance(si, int) else -1)

    if not gt:
        raise SystemExit(f"No usable steps found (need ContactGTPerC + {pred_key}; did you run with --log_contacts?).")

    gt_a = np.asarray(gt, dtype=np.float64)  # (N,2)
    pred_a = np.asarray(pred, dtype=np.float64)  # (N,2)

    # Basic collapse stats.
    l = pred_a[:, 0]
    r = pred_a[:, 1]
    frac_r_gt_l = float((r > l).mean())
    frac_l_gt_r = float((l > r).mean())
    lr_diff = l - r
    print(f"[JSON] {path}")
    print(
        f"[Steps] N={int(gt_a.shape[0])} exclude_round0={bool(args.exclude_round0)} cycle_len={cycle_len} pred_source={args.pred_source}"
    )
    print(
        "[{}] mean(L,R)=({:.3f},{:.3f}) std(L,R)=({:.3f},{:.3f}) min(L,R)=({:.3f},{:.3f}) max(L,R)=({:.3f},{:.3f})".format(
            pred_name,
            float(l.mean()),
            float(r.mean()),
            float(l.std()),
            float(r.std()),
            float(l.min()),
            float(r.min()),
            float(l.max()),
            float(r.max()),
        )
    )
    print(
        f"[{pred_name}] P(R>L)={frac_r_gt_l:.4f} P(L>R)={frac_l_gt_r:.4f}  mean(L-R)={float(lr_diff.mean()):.3f} std(L-R)={float(lr_diff.std()):.3f}"
    )

    # Plan vs GT error stats.
    diff = pred_a - gt_a
    mse = float((diff * diff).mean())
    mae = float(np.abs(diff).mean())
    mse_per_c = diff * diff
    mse_l = float(mse_per_c[:, 0].mean())
    mse_r = float(mse_per_c[:, 1].mean())
    print(f"[{pred_name} vs GT] MSE(all)={mse:.4f}  MAE(all)={mae:.4f}  MSE(L)={mse_l:.4f} MSE(R)={mse_r:.4f}")

    # Shift search per channel (only meaningful when signals have amplitude).
    best_l, _, mse0_l, mse_best_l = _shift_mse(pred_a[:, 0], gt_a[:, 0], max_shift=int(args.max_shift))
    best_r, _, mse0_r, mse_best_r = _shift_mse(pred_a[:, 1], gt_a[:, 1], max_shift=int(args.max_shift))
    mod_l = best_l % cycle_len if cycle_len > 0 else best_l
    mod_r = best_r % cycle_len if cycle_len > 0 else best_r
    print(f"[ShiftSearch] L: best_shift={best_l} (mod {mod_l}) mse0={mse0_l:.4f} mse={mse_best_l:.4f}")
    print(f"[ShiftSearch] R: best_shift={best_r} (mod {mod_r}) mse0={mse0_r:.4f} mse={mse_best_r:.4f}")

    # Single-support stance accuracy (argmax) on GT-confident frames.
    thr = float(args.single_support_thr)
    gt_max = gt_a.max(axis=1)
    gt_min = gt_a.min(axis=1)
    mask = (gt_max >= thr) & (gt_min <= (1.0 - thr))
    pred_swap = pred_a[:, ::-1].copy()
    if mask.any():
        gt_cls = gt_a[mask].argmax(axis=1)
        pl_cls = pred_a[mask].argmax(axis=1)
        sw_cls = pred_swap[mask].argmax(axis=1)
        acc = float((gt_cls == pl_cls).mean())
        acc_sw = float((gt_cls == sw_cls).mean())
        # confusion matrix rows=GT, cols=Plan
        cm = np.zeros((2, 2), dtype=np.int64)
        cm_sw = np.zeros((2, 2), dtype=np.int64)
        for g, p in zip(gt_cls.tolist(), pl_cls.tolist()):
            cm[int(g), int(p)] += 1
        for g, p in zip(gt_cls.tolist(), sw_cls.tolist()):
            cm_sw[int(g), int(p)] += 1
        print(f"[StanceAcc] single_support_thr={thr:.2f} N={int(mask.sum())} acc={acc:.4f} cm={cm.tolist()}")
        print(f"[SwapEval] stance_acc={acc_sw:.4f} cm={cm_sw.tolist()}  (swap L/R of {pred_name.lower()})")
    else:
        print(f"[StanceAcc] no GT single-support frames under thr={thr:.2f}")

    # Swap eval (MSE/MAE): does swapping L/R better match GT? (identity confusion heuristic)
    diff_sw = pred_swap - gt_a
    mse_sw = float((diff_sw * diff_sw).mean())
    mae_sw = float(np.abs(diff_sw).mean())
    print(f"[SwapEval] mse(all)={mse_sw:.4f} mae(all)={mae_sw:.4f}  (swap L/R of {pred_name.lower()})")

    # Per-phase means (helps visualize collapse).
    if cycle_len > 0 and all(si >= 0 for si in step_in_cycle):
        si_a = np.asarray(step_in_cycle, dtype=np.int64)
        per = []
        for s in range(cycle_len):
            m = si_a == s
            if not m.any():
                per.append(None)
                continue
            g = gt_a[m].mean(axis=0)
            p = pred_a[m].mean(axis=0)
            per.append((float(p[0]), float(p[1]), float(g[0]), float(g[1])))

        # Find worst phases by |pred-gt| (L1).
        scores: List[Tuple[float, int]] = []
        for s, v in enumerate(per):
            if v is None:
                continue
            pL, pR, gL, gR = v
            l1 = abs(pL - gL) + abs(pR - gR)
            scores.append((float(l1), int(s)))
        scores.sort(reverse=True)

        print()
        print(f"[Per-Phase] show top-12 phases by L1({pred_name.lower()}-gt):")
        for l1, s in scores[:12]:
            v = per[s]
            assert v is not None
            pL, pR, gL, gR = v
            print(f"- phase {s:02d}: {pred_name.lower()}=({pL:.3f},{pR:.3f}) gt=({gL:.3f},{gR:.3f}) L1={l1:.3f}")


if __name__ == "__main__":
    main()
