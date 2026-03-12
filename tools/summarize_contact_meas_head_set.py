#!/usr/bin/env python3
"""
Summarize `contact_meas_head` diagnostics over a set of `*_teacher_pred.json` files.

This is meant to answer: "Is the left-support collapse systemic across the whole teacher-set,
or just a single clip (e.g. Walk_F)?".

Typical workflow:
  1) Run teacher-forced rollouts (GT pose_hist + angvel):
       python -m train.validate.run_teacher_rollout \\
         --model <ckpt.pth> --teacher validate/teacher_batches/*.json \\
         --encoder-bundle models/motion_encoder_equiv_stageA.pt --depth 3 \\
         --out debug_output/_tmp_teacher_debug/teacher_rollout_measdiag_all --force

  2) Summarize:
       python tools/summarize_contact_meas_head_set.py \\
         --pred debug_output/_tmp_teacher_debug/teacher_rollout_measdiag_all \\
         --on-th 0.8 --off-th 0.1 --out debug_output/_tmp_teacher_debug/teacher_rollout_measdiag_all
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tools.analyze_contact_meas_head import analyze_teacher_pred_json  # type: ignore
except Exception:  # pragma: no cover
    # When executed as `python tools/xxx.py`, sys.path[0] == "tools/", so import sibling directly.
    from analyze_contact_meas_head import analyze_teacher_pred_json  # type: ignore


def _expand_pred_specs(specs: Sequence[str], *, pattern: str = "*_teacher_pred.json") -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for spec in specs:
        if not spec:
            continue
        s = os.path.expanduser(str(spec))

        matches: List[Path] = []
        if any(ch in s for ch in "*?[]"):
            matches = [Path(p) for p in glob.glob(s)]
        else:
            p = Path(s)
            if p.is_dir():
                matches = sorted(p.glob(pattern))
            elif p.is_file():
                matches = [p]

        for m in matches:
            try:
                r = m.resolve()
            except Exception:
                r = m
            if r.is_file() and r not in seen:
                seen.add(r)
                out.append(r)
    return sorted(out)


def _regime_by_name(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    regimes = summary.get("regimes", None)
    if not isinstance(regimes, list):
        return out
    for r in regimes:
        if not isinstance(r, dict):
            continue
        name = r.get("name", None)
        if isinstance(name, str):
            out[name] = r
    return out


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _as_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _pair_get(vals: Any, idx: int) -> Optional[float]:
    if not isinstance(vals, (list, tuple)) or len(vals) <= idx:
        return None
    return _as_float(vals[idx])


def _fmt(x: Optional[float], *, digits: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols: List[str] = [
        "clip",
        "T",
        "mse",
        "bce_prob",
        "bce_logits",
        "AUC_L",
        "AUC_R",
        "AUC_L_vs_GT_R",
        "AUC_R_vs_GT_L",
        "Corr_L",
        "Corr_R",
        "Corr_L_vs_GT_R",
        "Corr_R_vs_GT_L",
        "left_n",
        "left_gt_L",
        "left_gt_R",
        "left_pred_L",
        "left_pred_R",
        "left_p_pred_L_gt_R",
        "right_n",
        "right_gt_L",
        "right_gt_R",
        "right_pred_L",
        "right_pred_R",
        "right_p_pred_R_gt_L",
        "worst_left_ti",
        "worst_left_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in cols})


def _write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "clip",
        "T",
        "left_n",
        "P(L>R|Lsup)",
        "Lsup_pred(L,R)",
        "right_n",
        "P(R>L|Rsup)",
        "Rsup_pred(L,R)",
        "AUC(L,R)",
        "Corr(L,R)",
        "worst_left(ti,score)",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|\n")
        for r in rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        str(r.get("clip", "")),
                        str(r.get("T", "")),
                        str(r.get("left_n", "")),
                        _fmt(_as_float(r.get("left_p_pred_L_gt_R")), digits=3),
                        f"[{_fmt(_as_float(r.get('left_pred_L')), digits=3)},{_fmt(_as_float(r.get('left_pred_R')), digits=3)}]",
                        str(r.get("right_n", "")),
                        _fmt(_as_float(r.get("right_p_pred_R_gt_L")), digits=3),
                        f"[{_fmt(_as_float(r.get('right_pred_L')), digits=3)},{_fmt(_as_float(r.get('right_pred_R')), digits=3)}]",
                        f"[{_fmt(_as_float(r.get('AUC_L')), digits=3)},{_fmt(_as_float(r.get('AUC_R')), digits=3)}]",
                        f"[{_fmt(_as_float(r.get('Corr_L')), digits=3)},{_fmt(_as_float(r.get('Corr_R')), digits=3)}]",
                        f"({r.get('worst_left_ti','-')},{_fmt(_as_float(r.get('worst_left_score')), digits=3)})",
                    ]
                )
                + " |\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize contact_meas_head diagnostics over teacher rollout outputs.")
    ap.add_argument(
        "--pred",
        nargs="+",
        required=True,
        help="Paths / dirs / globs to `*_teacher_pred.json` (dir implies '*_teacher_pred.json').",
    )
    ap.add_argument("--on-th", type=float, default=0.8, help="Support ON threshold.")
    ap.add_argument("--off-th", type=float, default=0.1, help="Support OFF threshold.")
    ap.add_argument("--top-k", type=int, default=0, help="Keep top-k worst left-support frames per clip in JSON.")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output directory (writes contact_meas_head_summary.{json,csv,md}).",
    )
    args = ap.parse_args()

    files = _expand_pred_specs(args.pred)
    if not files:
        raise SystemExit("[FATAL] --pred expanded to empty file list.")

    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for p in files:
        s = analyze_teacher_pred_json(p, on_th=float(args.on_th), off_th=float(args.off_th), top_k=int(args.top_k))
        summaries.append(s)
        reg = _regime_by_name(s)
        overall = s.get("overall", {}) if isinstance(s.get("overall", {}), dict) else {}
        left = reg.get("left_support", {})
        right = reg.get("right_support", {})

        worst = s.get("worst_left_support", [])
        worst_ti = None
        worst_score = None
        if isinstance(worst, list) and worst:
            w0 = worst[0] if isinstance(worst[0], dict) else None
            if isinstance(w0, dict):
                worst_ti = w0.get("ti", None)
                worst_score = w0.get("score", None)

        row: Dict[str, Any] = {
            "clip": s.get("clip"),
            "T": _as_int(s.get("T")),
            "mse": _as_float(overall.get("mse")),
            "bce_prob": _as_float(overall.get("bce_prob")),
            "bce_logits": _as_float(overall.get("bce_logits")),
            "AUC_L": _as_float(overall.get("AUC_L")),
            "AUC_R": _as_float(overall.get("AUC_R")),
            "AUC_L_vs_GT_R": _as_float(overall.get("AUC_L_vs_GT_R")),
            "AUC_R_vs_GT_L": _as_float(overall.get("AUC_R_vs_GT_L")),
            "Corr_L": _as_float(overall.get("Corr_L")),
            "Corr_R": _as_float(overall.get("Corr_R")),
            "Corr_L_vs_GT_R": _as_float(overall.get("Corr_L_vs_GT_R")),
            "Corr_R_vs_GT_L": _as_float(overall.get("Corr_R_vs_GT_L")),
            "left_n": _as_int(left.get("n")),
            "left_gt_L": _pair_get(left.get("gt_mean"), 0),
            "left_gt_R": _pair_get(left.get("gt_mean"), 1),
            "left_pred_L": _pair_get(left.get("pred_mean"), 0),
            "left_pred_R": _pair_get(left.get("pred_mean"), 1),
            "left_p_pred_L_gt_R": _as_float(left.get("p_pred_L_gt_R")),
            "right_n": _as_int(right.get("n")),
            "right_gt_L": _pair_get(right.get("gt_mean"), 0),
            "right_gt_R": _pair_get(right.get("gt_mean"), 1),
            "right_pred_L": _pair_get(right.get("pred_mean"), 0),
            "right_pred_R": _pair_get(right.get("pred_mean"), 1),
            "right_p_pred_R_gt_L": _as_float(right.get("p_pred_R_gt_L")),
            "worst_left_ti": worst_ti,
            "worst_left_score": worst_score,
        }
        rows.append(row)

    def _sort_key(r: Dict[str, Any]) -> Tuple[float, float]:
        # primary: left_support correctness (lower is worse); secondary: overall mse (higher is worse)
        p = r.get("left_p_pred_L_gt_R")
        p = float(p) if isinstance(p, (int, float)) and p == p else 1.0
        mse = r.get("mse")
        mse = float(mse) if isinstance(mse, (int, float)) and mse == mse else 0.0
        return (p, -mse)

    rows.sort(key=_sort_key)

    # Dataset-level aggregate (weighted by regime counts)
    tot_left_n = sum(int(r.get("left_n", 0) or 0) for r in rows)
    tot_right_n = sum(int(r.get("right_n", 0) or 0) for r in rows)
    left_correct = 0.0
    right_correct = 0.0
    for r in rows:
        ln = int(r.get("left_n", 0) or 0)
        rn = int(r.get("right_n", 0) or 0)
        lp = _as_float(r.get("left_p_pred_L_gt_R"))
        rp = _as_float(r.get("right_p_pred_R_gt_L"))
        if ln > 0 and lp is not None:
            left_correct += float(ln) * float(lp)
        if rn > 0 and rp is not None:
            right_correct += float(rn) * float(rp)

    print(f"[ContactMeasSet] files={len(files)} on_th={float(args.on_th):.3f} off_th={float(args.off_th):.3f}")
    if tot_left_n > 0:
        print(f"[Agg] left_support n={tot_left_n} weighted P(pred_L>pred_R)={left_correct / max(1.0, float(tot_left_n)):.3f}")
    else:
        print("[Agg] left_support n=0")
    if tot_right_n > 0:
        print(f"[Agg] right_support n={tot_right_n} weighted P(pred_R>pred_L)={right_correct / max(1.0, float(tot_right_n)):.3f}")
    else:
        print("[Agg] right_support n=0")

    # Compact stdout table
    header = [
        "clip",
        "T",
        "left_n",
        "P(L>R|Lsup)",
        "Lsup_pred(L,R)",
        "right_n",
        "P(R>L|Rsup)",
        "Rsup_pred(L,R)",
        "AUC(L,R)",
        "Corr(L,R)",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for r in rows:
        clip = str(r.get("clip", ""))
        print(
            "| "
            + " | ".join(
                [
                    clip,
                    str(r.get("T", "")),
                    str(r.get("left_n", "")),
                    _fmt(_as_float(r.get("left_p_pred_L_gt_R")), digits=3),
                    f"[{_fmt(_as_float(r.get('left_pred_L')), digits=3)},{_fmt(_as_float(r.get('left_pred_R')), digits=3)}]",
                    str(r.get("right_n", "")),
                    _fmt(_as_float(r.get("right_p_pred_R_gt_L")), digits=3),
                    f"[{_fmt(_as_float(r.get('right_pred_L')), digits=3)},{_fmt(_as_float(r.get('right_pred_R')), digits=3)}]",
                    f"[{_fmt(_as_float(r.get('AUC_L')), digits=3)},{_fmt(_as_float(r.get('AUC_R')), digits=3)}]",
                    f"[{_fmt(_as_float(r.get('Corr_L')), digits=3)},{_fmt(_as_float(r.get('Corr_R')), digits=3)}]",
                ]
            )
            + " |"
        )

    if args.out:
        out_dir = Path(args.out).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "contact_meas_head_summary.json"
        out_csv = out_dir / "contact_meas_head_summary.csv"
        out_md = out_dir / "contact_meas_head_summary.md"

        out_json.write_text(
            json.dumps(
                {
                    "thresholds": {"on": float(args.on_th), "off": float(args.off_th)},
                    "files": [str(p) for p in files],
                    "rows": rows,
                    "summaries": summaries if int(args.top_k) > 0 else None,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _write_csv(out_csv, rows)
        _write_md(out_md, rows)
        print(f"[Wrote] {out_json}")
        print(f"[Wrote] {out_csv}")
        print(f"[Wrote] {out_md}")


if __name__ == "__main__":
    main()
