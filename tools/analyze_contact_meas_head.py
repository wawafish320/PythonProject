#!/usr/bin/env python3
"""
Analyze contact_meas_head predictions from run_teacher_rollout JSON output.

This is meant for diagnosing left/right asymmetry and "near-constant ~0.5" collapse
in learned contacts_meas under teacher-forced inputs (GT pose_hist + angvel).

Example:
  python tools/analyze_contact_meas_head.py \
    --json debug_output/_tmp_teacher_debug/teacher_rollout_measdiag/Walk_F_teacher_pred.json \
    --on-th 0.8 --off-th 0.1 --top-k 20
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np


def analyze_teacher_pred_json(
    path: Path,
    *,
    on_th: float = 0.8,
    off_th: float = 0.1,
    top_k: int = 20,
) -> dict:
    """
    Analyze a `*_teacher_pred.json` produced by `train.validate.run_teacher_rollout`.

    Returns a JSON-serializable dict (means/logits converted to Python lists).
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    gt = np.asarray(data.get("aux_inputs", {}).get("contacts", []), dtype=np.float64)
    pred = np.asarray(data.get("contacts_pred", {}).get("contacts_meas", []), dtype=np.float64)
    logits_raw = data.get("contacts_pred", {}).get("contacts_meas_logits", None)
    logits = None
    if logits_raw is not None:
        logits = np.asarray(logits_raw, dtype=np.float64)

    if gt.ndim != 2 or pred.ndim != 2:
        raise ValueError(f"Expected gt/pred to be 2D arrays, got gt={gt.shape}, pred={pred.shape}.")
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch gt={gt.shape} pred={pred.shape}.")
    if logits is not None and logits.shape != gt.shape:
        logits = None  # be conservative; some older dumps may differ

    T, C = gt.shape
    if C < 2:
        raise ValueError(f"Expected contact_dim>=2, got C={C}.")

    on_th = float(on_th)
    off_th = float(off_th)

    left_support = (gt[:, 0] > on_th) & (gt[:, 1] < off_th)
    right_support = (gt[:, 1] > on_th) & (gt[:, 0] < off_th)
    double_support = (gt[:, 0] > on_th) & (gt[:, 1] > on_th)
    air = (gt[:, 0] < off_th) & (gt[:, 1] < off_th)
    other = ~(left_support | right_support | double_support | air)

    summary = {
        "clip": data.get("clip"),
        "model": data.get("model"),
        "T": int(T),
        "C": int(C),
        "thresholds": {"on": float(on_th), "off": float(off_th)},
    }

    summary["overall"] = {
        "gt_mean": [float(x) for x in gt.mean(axis=0)],
        "pred_mean": [float(x) for x in pred.mean(axis=0)],
        "mse": float(((pred - gt) ** 2).mean()),
        "bce_prob": _bce_prob(pred, gt),
    }
    if logits is not None:
        summary["overall"]["bce_logits"] = _bce_with_logits(logits, gt)

    # Per-channel AUC / corr (and cross, to detect L/R confusion)
    auc = {}
    corr = {}
    for i, name in enumerate(["L", "R"]):
        auc[f"AUC_{name}"] = _weighted_auc(pred[:, i], gt[:, i])
        corr[f"Corr_{name}"] = _pearson_corr(pred[:, i], gt[:, i])
    auc["AUC_L_vs_GT_R"] = _weighted_auc(pred[:, 0], gt[:, 1])
    auc["AUC_R_vs_GT_L"] = _weighted_auc(pred[:, 1], gt[:, 0])
    corr["Corr_L_vs_GT_R"] = _pearson_corr(pred[:, 0], gt[:, 1])
    corr["Corr_R_vs_GT_L"] = _pearson_corr(pred[:, 1], gt[:, 0])
    summary["overall"].update(auc)
    summary["overall"].update(corr)

    # Regime summaries
    regimes = [
        _summarize_regime("left_support", left_support, gt, pred, logits),
        _summarize_regime("right_support", right_support, gt, pred, logits),
        _summarize_regime("double_support", double_support, gt, pred, logits),
        _summarize_regime("air", air, gt, pred, logits),
        _summarize_regime("other", other, gt, pred, logits),
    ]
    summary["regimes"] = regimes

    # Worst left-support frames: largest (gt_L - pred_L) + (pred_R - gt_R) (i.e., underpredict L + overpredict R)
    worst = []
    if left_support.any():
        err = (gt[:, 0] - pred[:, 0]) + (pred[:, 1] - gt[:, 1])
        idxs = np.where(left_support)[0]
        order = idxs[np.argsort(err[idxs])[::-1]]
        topk = order[: max(0, int(top_k))]
        for ti in topk.tolist():
            row = {
                "ti": int(ti),
                "gt": [float(x) for x in gt[ti].tolist()],
                "pred": [float(x) for x in pred[ti].tolist()],
                "score": float(err[ti]),
            }
            if logits is not None:
                row["logits"] = [float(x) for x in logits[ti].tolist()]
            worst.append(row)
    summary["worst_left_support"] = worst
    return summary


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if not m.any():
        return float("nan")
    x = x[m]
    y = y[m]
    if x.size < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-8 or sy < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _weighted_auc(scores: np.ndarray, p_pos: np.ndarray, eps: float = 1e-9) -> float:
    """
    Soft-label AUC (pairwise ranking) with weights:
      - p_pos in [0,1] as positive mass
      - (1-p_pos) as negative mass
    """
    s = np.asarray(scores, dtype=np.float64)
    w1 = np.asarray(p_pos, dtype=np.float64)
    w0 = 1.0 - w1
    m = np.isfinite(s) & np.isfinite(w1) & np.isfinite(w0)
    if not m.any():
        return float("nan")
    s, w1, w0 = s[m], w1[m], w0[m]
    w1 = np.clip(w1, 0.0, 1.0)
    w0 = np.clip(w0, 0.0, 1.0)
    W1 = float(w1.sum())
    W0 = float(w0.sum())
    if W1 < eps or W0 < eps:
        return float("nan")
    idx = np.argsort(s, kind="mergesort")  # stable
    w1 = w1[idx]
    w0 = w0[idx]
    cum_w0 = np.cumsum(w0)
    return float((w1 * cum_w0).sum() / (W1 * W0 + eps))


def _bce_prob(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, 1.0)
    return float((-y * np.log(p) - (1.0 - y) * np.log(1.0 - p)).mean())


def _bce_with_logits(logits: np.ndarray, y: np.ndarray) -> float:
    # Stable BCEWithLogits: max(x,0) - x*y + log(1+exp(-abs(x)))
    x = np.asarray(logits, dtype=np.float64)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, 1.0)
    m = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
    return float(m.mean())


def _fmt_pair(x: np.ndarray) -> str:
    if x is None:
        return "None"
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return "[]"
    if x.size == 1:
        return f"[{x[0]:.4f}]"
    return f"[{x[0]:.4f}, {x[1]:.4f}]"


def _summarize_regime(
    name: str,
    mask: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    logits: Optional[np.ndarray],
) -> dict:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    out = {"name": name, "n": n}
    if n <= 0:
        return out

    gt_m = [float(x) for x in gt[mask].mean(axis=0)]
    pred_m = [float(x) for x in pred[mask].mean(axis=0)]
    out["gt_mean"] = gt_m
    out["pred_mean"] = pred_m

    if logits is not None:
        out["logit_mean"] = [float(x) for x in logits[mask].mean(axis=0)]

    if gt.shape[1] >= 2:
        out["p_pred_L_gt_R"] = float((pred[mask, 0] > pred[mask, 1]).mean())
        out["p_pred_R_gt_L"] = float((pred[mask, 1] > pred[mask, 0]).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True, help="Path to *_teacher_pred.json from run_teacher_rollout.")
    ap.add_argument("--on-th", type=float, default=0.8, help="Support ON threshold for regime split.")
    ap.add_argument("--off-th", type=float, default=0.1, help="Support OFF threshold for regime split.")
    ap.add_argument("--top-k", type=int, default=20, help="Print top-k worst left-support frames.")
    ap.add_argument("--out", type=str, default=None, help="Optional output JSON summary path.")
    args = ap.parse_args()

    path = Path(args.json)
    try:
        summary = analyze_teacher_pred_json(
            path,
            on_th=float(args.on_th),
            off_th=float(args.off_th),
            top_k=int(args.top_k),
        )
    except Exception as e:
        raise SystemExit(f"[FATAL] {path}: {e}")

    T = int(summary.get("T", 0) or 0)
    C = int(summary.get("C", 0) or 0)
    regimes = summary.get("regimes", [])
    if not isinstance(regimes, list):
        regimes = []
    worst = summary.get("worst_left_support", [])
    if not isinstance(worst, list):
        worst = []

    # ---- Print (human-readable) ----
    print(f"[ContactMeasDiag] json={path} clip={summary.get('clip')} T={T} C={C}")
    print(f"[Overall] gt_mean={_fmt_pair(summary['overall']['gt_mean'])} pred_mean={_fmt_pair(summary['overall']['pred_mean'])}")
    print(f"[Overall] mse={summary['overall']['mse']:.6f} bce_prob={summary['overall']['bce_prob']:.6f}"
          + (f" bce_logits={summary['overall']['bce_logits']:.6f}" if "bce_logits" in summary["overall"] else ""))
    print(
        "[Overall] "
        f"AUC_L/R={summary['overall']['AUC_L']:.3f}/{summary['overall']['AUC_R']:.3f} "
        f"(cross LvsR={summary['overall']['AUC_L_vs_GT_R']:.3f} RvsL={summary['overall']['AUC_R_vs_GT_L']:.3f})"
    )
    print(
        "[Overall] "
        f"Corr_L/R={summary['overall']['Corr_L']:.3f}/{summary['overall']['Corr_R']:.3f} "
        f"(cross LvsR={summary['overall']['Corr_L_vs_GT_R']:.3f} RvsL={summary['overall']['Corr_R_vs_GT_L']:.3f})"
    )

    for r in regimes:
        n = r["n"]
        if n <= 0:
            print(f"[{r['name']}] n=0")
            continue
        msg = f"[{r['name']}] n={n} gt_mean={_fmt_pair(r.get('gt_mean'))} pred_mean={_fmt_pair(r.get('pred_mean'))}"
        if "logit_mean" in r:
            msg += f" logit_mean={_fmt_pair(r.get('logit_mean'))}"
        if r["name"] == "left_support":
            msg += f" P(pred_L>pred_R)={r.get('p_pred_L_gt_R', float('nan')):.3f}"
        if r["name"] == "right_support":
            msg += f" P(pred_R>pred_L)={r.get('p_pred_R_gt_L', float('nan')):.3f}"
        print(msg)

    if worst:
        print(f"[WorstLeftSupport] top_k={len(worst)} (sorted by under-L + over-R)")
        for row in worst:
            ti = row["ti"]
            gt_lr = row["gt"]
            pr_lr = row["pred"]
            s = row["score"]
            extra = ""
            if "logits" in row:
                extra = f" logits=[{row['logits'][0]:+.3f},{row['logits'][1]:+.3f}]"
            print(
                f"  ti={ti:3d} gt=[{gt_lr[0]:.3f},{gt_lr[1]:.3f}] pred=[{pr_lr[0]:.3f},{pr_lr[1]:.3f}] score={s:+.3f}{extra}"
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Wrote] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
