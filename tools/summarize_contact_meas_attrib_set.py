#!/usr/bin/env python3
"""
Summarize `contact_meas_head` input attribution (bone × pose_hist block) over a set of conditions.

This script merges:
  - robust long-tail metrics from `contact_meas_lag_summary.json` (pred@GT_fall, dt<=mid)
  - attribution from `tools/attrib_contact_meas_inputs.py` outputs (`*_attrib.json`)

Example:
  python tools/summarize_contact_meas_attrib_set.py \
    --root debug_output/_tmp_teacher_debug/_batch_eventlag \
    --conds baseline keep_last1 pose_zero \
    --attrib-subdir _attrib_batch2 \
    --event falling --channel R \
    --out debug_output/_tmp_teacher_debug/_batch_eventlag/_attrib_batch2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fmt_float(x: Optional[float], *, digits: int = 3) -> str:
    return "-" if x is None else f"{float(x):.{digits}f}"


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / float(len(xs))
    my = sum(ys) / float(len(ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / (vx**0.5 * vy**0.5))


def _bone_category(name: str) -> str:
    n = str(name).lower()
    if any(k in n for k in ("foot", "ball", "toe")):
        return "foot"
    if any(k in n for k in ("calf", "thigh", "shin", "leg")):
        return "leg"
    if any(k in n for k in ("pelvis", "spine", "neck", "head", "clavicle")):
        return "spine"
    if any(
        k in n
        for k in (
            "upperarm",
            "lowerarm",
            "hand",
            "pinky",
            "ring",
            "index",
            "middle",
            "thumb",
            "foretwist",
            "armtwist",
            "shoulder",
        )
    ):
        return "arm"
    return "other"


def _expand_conds(root: Path, conds: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in conds:
        c = str(c).strip()
        if not c:
            continue
        if (root / c).is_dir():
            out.append(c)
    return out


def _find_attrib_jsons(attrib_dir: Path, *, channel: str, event: str) -> List[Path]:
    patt = str(attrib_dir / f"*_{channel}_{event}_t*_attrib.json")
    return sorted(Path(p) for p in glob.glob(patt))


def _write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in cols})


def _write_md(path: Path, rows: List[Dict[str, Any]], *, cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")


def _sum_abs(mat: Sequence[Sequence[float]]) -> float:
    s = 0.0
    for row in mat:
        for v in row:
            s += abs(float(v))
    return float(s)


def _sum_abs_vec(vec: Sequence[float]) -> float:
    return float(sum(abs(float(v)) for v in vec))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize contact_meas_head attribution set over conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--root", type=str, required=True, help="Batch root (contains <cond>/ and attribution subdir).")
    ap.add_argument("--conds", nargs="+", required=True, help="Condition directories under --root (e.g., baseline keep_last1).")
    ap.add_argument("--attrib-subdir", type=str, default="_attrib_batch2", help="Attribution subdir under --root.")
    ap.add_argument("--event", type=str, default="falling", choices=("rising", "falling"), help="Event edge type.")
    ap.add_argument("--channel", type=str, default="R", choices=("L", "R"), help="Contact channel.")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: <root>/<attrib-subdir>).")
    ap.add_argument("--topk", type=int, default=5, help="Top-K bones to report for dt=0 pose_hist block.")
    args = ap.parse_args()

    root = Path(os.path.expanduser(str(args.root))).resolve()
    attrib_subdir = str(args.attrib_subdir)
    out_dir = Path(os.path.expanduser(args.out)).resolve() if args.out else (root / attrib_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conds = _expand_conds(root, args.conds)
    if not conds:
        raise SystemExit("[FATAL] --conds expanded to empty list (no existing <root>/<cond> dirs).")

    channel = str(args.channel)
    event = str(args.event)

    rows: List[Dict[str, Any]] = []
    per_cond_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for cond in conds:
        lag_path = root / cond / "contact_meas_lag_summary.json"
        if not lag_path.is_file():
            raise SystemExit(f"[FATAL] Missing lag summary: {lag_path}")
        lag = _load_json(lag_path)
        lag_rows = lag.get("rows", [])
        if not isinstance(lag_rows, list):
            raise SystemExit(f"[FATAL] Bad lag summary format: {lag_path}")
        lag_by_clip: Dict[str, Dict[str, Any]] = {}
        for r in lag_rows:
            if isinstance(r, dict) and r.get("clip"):
                lag_by_clip[str(r["clip"])] = r

        attrib_dir = root / attrib_subdir / cond
        files = _find_attrib_jsons(attrib_dir, channel=channel, event=event)
        if not files:
            raise SystemExit(f"[FATAL] No attrib jsons matched under: {attrib_dir}")

        for p in files:
            d = _load_json(p)
            if not isinstance(d, dict):
                continue
            clip = str(d.get("clip", ""))
            if not clip:
                continue

            lag_r = lag_by_clip.get(clip, {})
            pred_at = _as_float(lag_r.get(f"{channel}_fall_pred_at_med"))
            dt_mid = _as_float(lag_r.get(f"{channel}_fall_time_to_mid_med"))

            prob_full = _as_float(d.get("prob_full"))
            prob_pose0 = _as_float(d.get("prob_pose_zero"))
            prob_ang0 = _as_float(d.get("prob_angvel_zero"))

            bone_names = d.get("bone_names", [])
            if not isinstance(bone_names, list) or not bone_names:
                bone_names = [f"bone_{i}" for i in range(int(d.get("dims", {}).get("joints", 0) or 0))]
            bone_names = [str(x) for x in bone_names]

            pose = d.get("pose", {}) if isinstance(d.get("pose"), dict) else {}
            ang = d.get("angvel", {}) if isinstance(d.get("angvel"), dict) else {}

            pose_contrib = pose.get("contrib", [])
            ang_contrib = ang.get("contrib", [])
            if not isinstance(pose_contrib, list) or not pose_contrib:
                continue
            if not isinstance(ang_contrib, list) or not ang_contrib:
                continue

            # pose_contrib: (L,J)
            pose_mat: List[List[float]] = [[float(x) for x in row] for row in pose_contrib]
            ang_vec: List[float] = [float(x) for x in ang_contrib]

            per_block = pose.get("per_block", [])
            dt0_block = None
            if isinstance(per_block, list):
                for b in per_block:
                    if isinstance(b, dict) and int(b.get("dt", -999)) == 0:
                        dt0_block = int(b.get("block", -1))
                        break
            if dt0_block is None:
                dt0_block = len(pose_mat) - 1  # fallback: newest
            dt0_block = max(0, min(int(dt0_block), len(pose_mat) - 1))

            # Per-dt sums (abs contrib).
            pose_abs_total = _sum_abs(pose_mat)
            pose_abs_by_dt: Dict[int, float] = {}
            for bi, row in enumerate(pose_mat):
                dt = bi - (len(pose_mat) - 1)
                pose_abs_by_dt[int(dt)] = float(sum(abs(float(v)) for v in row))
            pose_abs_dt0 = float(sum(abs(float(v)) for v in pose_mat[dt0_block]))

            ang_abs_total = _sum_abs_vec(ang_vec)
            ratio_pose_ang = float(pose_abs_total / max(ang_abs_total, 1e-12))

            # Foot fraction at dt=0 (pose_hist newest block).
            if channel == "R":
                foot_names = {"foot_r", "ball_r"}
            else:
                foot_names = {"foot_l", "ball_l"}
            dt0_abs_total = pose_abs_dt0
            dt0_abs_foot = 0.0
            for j, name in enumerate(bone_names):
                if name in foot_names:
                    dt0_abs_foot += abs(float(pose_mat[dt0_block][j]))
            foot_frac_dt0 = float(dt0_abs_foot / max(dt0_abs_total, 1e-12))

            # Category breakdown at dt=0.
            cat_abs_dt0: Dict[str, float] = defaultdict(float)
            for j, name in enumerate(bone_names):
                cat = _bone_category(name)
                cat_abs_dt0[cat] += abs(float(pose_mat[dt0_block][j]))
            # Normalize to fractions.
            cat_frac_dt0 = {k: float(v / max(dt0_abs_total, 1e-12)) for k, v in cat_abs_dt0.items()}

            # Top-K dt=0 pose contributors.
            top = sorted(
                [(bone_names[j], float(pose_mat[dt0_block][j])) for j in range(min(len(bone_names), len(pose_mat[dt0_block])))],
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[: int(args.topk)]
            top_str = ", ".join([f"{n}:{v:+.3f}" for n, v in top])
            top1 = top[0][0] if top else ""
            top1_is_foot = bool(top1 in foot_names)

            # Consistency check: lag pred_at vs attrib prob_full.
            pred_prob_diff = None
            if pred_at is not None and prob_full is not None:
                pred_prob_diff = float(abs(pred_at - prob_full))

            t0 = int(d.get("t", -1))

            row = {
                "cond": cond,
                "clip": clip,
                "t_edge": t0,
                "p@GT_fall": _fmt_float(pred_at, digits=3),
                "dt<=mid": _fmt_float(dt_mid, digits=1),
                "prob_full": _fmt_float(prob_full, digits=3),
                "prob_pose0": _fmt_float(prob_pose0, digits=3),
                "prob_ang0": _fmt_float(prob_ang0, digits=3),
                "pose_abs_total": _fmt_float(pose_abs_total, digits=3),
                "pose_abs_dt0": _fmt_float(pose_abs_dt0, digits=3),
                "ang_abs_total": _fmt_float(ang_abs_total, digits=3),
                "pose/ang_abs": _fmt_float(ratio_pose_ang, digits=2),
                "foot_frac_dt0": _fmt_float(foot_frac_dt0, digits=3),
                "top1_dt0": top1,
                "top1_is_foot": str(bool(top1_is_foot)),
                "top_dt0": top_str,
                "pred_prob_diff": _fmt_float(pred_prob_diff, digits=4),
                "dt-2_abs": _fmt_float(pose_abs_by_dt.get(-2), digits=3),
                "dt-1_abs": _fmt_float(pose_abs_by_dt.get(-1), digits=3),
                "dt0_abs": _fmt_float(pose_abs_by_dt.get(0), digits=3),
                "dt0_frac_foot": _fmt_float(cat_frac_dt0.get("foot"), digits=3),
                "dt0_frac_leg": _fmt_float(cat_frac_dt0.get("leg"), digits=3),
                "dt0_frac_spine": _fmt_float(cat_frac_dt0.get("spine"), digits=3),
                "dt0_frac_arm": _fmt_float(cat_frac_dt0.get("arm"), digits=3),
                "dt0_frac_other": _fmt_float(cat_frac_dt0.get("other"), digits=3),
                "attrib_json": str(p),
            }
            rows.append(row)
            per_cond_rows[cond].append(row)

    # Sort: by condition then dt<=mid desc (robust long-tail severity)
    def _sort_key(r: Dict[str, Any]) -> Tuple[str, float]:
        dt = _as_float(r.get("dt<=mid"))
        return str(r.get("cond", "")), -(dt if dt is not None else -1e9)

    rows_sorted = sorted(rows, key=_sort_key)

    cols_csv = [
        "cond",
        "clip",
        "t_edge",
        "p@GT_fall",
        "dt<=mid",
        "prob_full",
        "prob_pose0",
        "prob_ang0",
        "pose_abs_total",
        "pose_abs_dt0",
        "ang_abs_total",
        "pose/ang_abs",
        "foot_frac_dt0",
        "top1_dt0",
        "top1_is_foot",
        "dt-2_abs",
        "dt-1_abs",
        "dt0_abs",
        "dt0_frac_foot",
        "dt0_frac_leg",
        "dt0_frac_spine",
        "dt0_frac_arm",
        "dt0_frac_other",
        "pred_prob_diff",
        "attrib_json",
    ]

    out_csv = out_dir / "contact_meas_attrib_summary.csv"
    _write_csv(out_csv, rows_sorted, cols_csv)

    out_md = out_dir / "contact_meas_attrib_summary.md"
    md_cols = ["cond", "clip", "t_edge", "p@GT_fall", "dt<=mid", "pose/ang_abs", "foot_frac_dt0", "top1_dt0", "top_dt0"]
    _write_md(out_md, rows_sorted, cols=md_cols)

    # Print / write aggregate summaries (per condition).
    stats: Dict[str, Any] = {"conds": conds, "event": event, "channel": channel, "n_rows": len(rows_sorted), "per_cond": {}}
    for cond in conds:
        rs = per_cond_rows.get(cond, [])
        # Correlations on numeric fields.
        dt_vals: List[float] = []
        p_vals: List[float] = []
        pose_abs_vals: List[float] = []
        ratio_vals: List[float] = []
        foot_frac_vals: List[float] = []
        top1 = Counter()
        top1_nonfoot = 0
        for r in rs:
            dt = _as_float(r.get("dt<=mid"))
            p = _as_float(r.get("p@GT_fall"))
            pose_abs = _as_float(r.get("pose_abs_total"))
            ratio = _as_float(r.get("pose/ang_abs"))
            foot_frac = _as_float(r.get("foot_frac_dt0"))
            if dt is not None:
                dt_vals.append(float(dt))
            if p is not None:
                p_vals.append(float(p))
            if pose_abs is not None:
                pose_abs_vals.append(float(pose_abs))
            if ratio is not None:
                ratio_vals.append(float(ratio))
            if foot_frac is not None:
                foot_frac_vals.append(float(foot_frac))
            b = str(r.get("top1_dt0", ""))
            if b:
                top1[b] += 1
                if str(r.get("top1_is_foot", "False")).lower() != "true":
                    top1_nonfoot += 1

        # Pair correlations need aligned lists; build by iterating rows.
        dt_for_corr: List[float] = []
        p_for_corr: List[float] = []
        for r in rs:
            dt = _as_float(r.get("dt<=mid"))
            p = _as_float(r.get("p@GT_fall"))
            pose_abs = _as_float(r.get("pose_abs_total"))
            if dt is not None and p is not None:
                dt_for_corr.append(float(dt))
                p_for_corr.append(float(p))
        # dt vs pose_abs (aligned)
        dt_pose_x: List[float] = []
        dt_pose_y: List[float] = []
        for r in rs:
            dt = _as_float(r.get("dt<=mid"))
            pose_abs = _as_float(r.get("pose_abs_total"))
            if dt is not None and pose_abs is not None:
                dt_pose_x.append(float(dt))
                dt_pose_y.append(float(pose_abs))

        stats["per_cond"][cond] = {
            "n": len(rs),
            "dt_mid_mean": float(sum(dt_vals) / max(len(dt_vals), 1)),
            "dt_mid_max": float(max(dt_vals) if dt_vals else 0.0),
            "p_at_mean": float(sum(p_vals) / max(len(p_vals), 1)),
            "pose_ang_ratio_mean": float(sum(ratio_vals) / max(len(ratio_vals), 1)),
            "foot_frac_dt0_mean": float(sum(foot_frac_vals) / max(len(foot_frac_vals), 1)),
            "corr_dt_p": _pearson(dt_for_corr, p_for_corr),
            "corr_dt_pose_abs": _pearson(dt_pose_x, dt_pose_y),
            "top1_dt0_counts": dict(top1),
            "top1_dt0_nonfoot_frac": float(top1_nonfoot / max(len(rs), 1)),
        }

    out_json = out_dir / "contact_meas_attrib_summary_stats.json"
    out_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_md}")
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
