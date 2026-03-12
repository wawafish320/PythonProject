#!/usr/bin/env python3
"""
Build Stage7 old-vs-new summary artifacts from freerun JSON outputs.

Outputs in --out-dir:
  - summary_metrics.txt
  - calf_r_per_sic_old_vs_new.csv
  - calf_r_per_sic_old_vs_new.md
  - global_signal_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class RunData:
    path: Path
    obj: Dict[str, Any]
    names: List[str]
    root: int
    mat: np.ndarray  # (S, J)
    steps: List[Dict[str, Any]]
    sics: np.ndarray  # (S,)
    mask: np.ndarray  # (S,) => cycle>=1 and drop_wrap
    phase_req: str
    phase_app: str
    meas_src: str


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _fmt(x: float, nd: int = 6) -> str:
    if not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.{nd}f}"


def _fmt_signed(x: float, nd: int = 6) -> str:
    if not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):+.{nd}f}"


def _pct_delta(base: float, new: float) -> float:
    if (not math.isfinite(base)) or abs(base) < 1e-12:
        return float("nan")
    return float((new - base) / base * 100.0)


def _parse_sic_spec(spec: str) -> set[int]:
    out: set[int] = set()
    for tok in spec.replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        if "-" in t:
            a, b = [x.strip() for x in t.split("-", 1)]
            if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
                lo = int(a)
                hi = int(b)
                if lo > hi:
                    lo, hi = hi, lo
                out.update(range(lo, hi + 1))
            continue
        if t.lstrip("-").isdigit():
            out.add(int(t))
    return out


def _load_run(path: Path) -> RunData:
    obj = _load_json(path)
    per = obj.get("per_step_direct_geolocal_deg", None)
    if not isinstance(per, dict):
        raise SystemExit(f"[FATAL] {path}: missing per_step_direct_geolocal_deg")
    names = list(per.get("bone_names", []))
    if not names:
        raise SystemExit(f"[FATAL] {path}: missing per_step_direct_geolocal_deg.bone_names")
    root = _safe_int(per.get("root_idx", 0), 0)
    mat = np.asarray(per.get("DirectGeoLocalDeg", None), dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] != len(names):
        raise SystemExit(f"[FATAL] {path}: invalid DirectGeoLocalDeg shape={mat.shape}, expected (*,{len(names)})")
    steps = obj.get("metrics_per_step", None)
    if not isinstance(steps, list) or len(steps) != mat.shape[0]:
        raise SystemExit(f"[FATAL] {path}: invalid metrics_per_step len vs matrix rows")

    sics = np.zeros((len(steps),), dtype=np.int64)
    mask = np.zeros((len(steps),), dtype=bool)
    for i, st in enumerate(steps):
        if not isinstance(st, dict):
            continue
        cyc = _safe_int(st.get("cycle", 0), 0)
        sic = _safe_int(st.get("step_in_cycle", st.get("sic", i)), i)
        wrap = bool(st.get("wrap_boundary_step", False))
        sics[i] = int(sic)
        if cyc >= 1 and not wrap:
            mask[i] = True

    phase_req = str(obj.get("phase_reset_source") or "").strip()
    phase_app = str(obj.get("phase_reset_source_applied") or phase_req or "").strip()
    meas_src = str(obj.get("contacts_meas_source") or "").strip()

    return RunData(
        path=path,
        obj=obj,
        names=names,
        root=root,
        mat=mat,
        steps=steps,
        sics=sics,
        mask=mask,
        phase_req=phase_req,
        phase_app=phase_app,
        meas_src=meas_src,
    )


def _select_bone_indices(run: RunData, bones: Optional[Sequence[str]]) -> List[int]:
    if bones is None:
        idx = [i for i in range(len(run.names)) if i != int(run.root)]
    else:
        idx = []
        seen = set()
        for b in bones:
            if b not in run.names:
                continue
            i = int(run.names.index(b))
            if i == int(run.root) or i in seen:
                continue
            seen.add(i)
            idx.append(i)
    return idx


def _select_step_mask(run: RunData, sics: Optional[set[int]]) -> np.ndarray:
    m = run.mask.copy()
    if sics is not None:
        m &= np.isin(run.sics, np.asarray(sorted(sics), dtype=np.int64))
    return m


def _calc_stats(
    run: RunData,
    *,
    bones: Optional[Sequence[str]],
    sics: Optional[set[int]],
) -> Dict[str, Any]:
    idx = _select_bone_indices(run, bones)
    if not idx:
        return {
            "step_mask": np.zeros_like(run.mask),
            "indices": [],
            "vals": np.asarray([], dtype=np.float64),
            "steps_kept": 0,
            "samples_kept": 0,
            "mean_deg": float("nan"),
            "p50_deg": float("nan"),
            "p90_deg": float("nan"),
            "p95_deg": float("nan"),
            "p99_deg": float("nan"),
            "max_deg": float("nan"),
            "per_bone_mean": [],
        }

    step_mask = _select_step_mask(run, sics)
    if not bool(step_mask.any()):
        vals = np.asarray([], dtype=np.float64)
        per_bone = []
    else:
        sub = run.mat[step_mask][:, idx]
        vals = sub[np.isfinite(sub)]
        per_bone = []
        for col, i in enumerate(idx):
            v = sub[:, col]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            per_bone.append((run.names[i], float(v.mean())))
        per_bone.sort(key=lambda kv: kv[1], reverse=True)

    return {
        "step_mask": step_mask,
        "indices": idx,
        "vals": vals,
        "steps_kept": int(step_mask.sum()),
        "samples_kept": int(vals.size),
        "mean_deg": float(vals.mean()) if vals.size > 0 else float("nan"),
        "p50_deg": _quantile(vals, 0.50),
        "p90_deg": _quantile(vals, 0.90),
        "p95_deg": _quantile(vals, 0.95),
        "p99_deg": _quantile(vals, 0.99),
        "max_deg": _quantile(vals, 1.00),
        "per_bone_mean": per_bone,
    }


def _render_run_block(
    run: RunData,
    stats: Dict[str, Any],
    *,
    cycle_gte: int,
    drop_wrap: bool,
    sics: Optional[set[int]],
    bones: Optional[Sequence[str]],
) -> List[str]:
    lines: List[str] = []
    lines.append(f"[{run.path.as_posix()}]")
    lines.append(
        f"  phase_reset: {run.phase_app or '-'} (req={run.phase_req or '-'})  contacts_meas_source: {run.meas_src or '-'}"
    )
    sics_repr = sorted(sics) if sics is not None else None
    lines.append(f"  mask: cycle>={cycle_gte} drop_wrap={drop_wrap} sics={sics_repr} excl_root={int(run.root)}")
    lines.append(f"  bones: {list(bones) if bones is not None else 'ALL(excl root)'}")
    lines.append(f"  steps_kept: {stats['steps_kept']}")
    lines.append(f"  samples_kept (steps*bones): {stats['samples_kept']}")
    lines.append(f"  mean_deg: {stats['mean_deg']}")
    lines.append(f"  p50_deg: {stats['p50_deg']}")
    lines.append(f"  p90_deg: {stats['p90_deg']}")
    lines.append(f"  p95_deg: {stats['p95_deg']}")
    lines.append(f"  p99_deg: {stats['p99_deg']}")
    lines.append(f"  max_deg: {stats['max_deg']}")
    lines.append("  per_bone_mean_deg (top 12):")
    for b, m in stats["per_bone_mean"][:12]:
        lines.append(f"    {b}: {m:.4f}")
    return lines


def _build_summary_metrics_text(old: RunData, new: RunData) -> str:
    sections = [
        ("1", "Global (all bones excl root)", None, None),
        ("2", "SIC12-15 + bones foot_l,ball_l", ["foot_l", "ball_l"], _parse_sic_spec("12-15")),
        ("3", "calf_r global", ["calf_r"], None),
        ("4", "calf_r @ SIC2-4", ["calf_r"], _parse_sic_spec("2-4")),
        ("5", "calf_r @ SIC35-42", ["calf_r"], _parse_sic_spec("35-42")),
        ("6", "calf_r @ SIC53-63", ["calf_r"], _parse_sic_spec("53-63")),
    ]

    out: List[str] = []
    out.append("# Stage7.2 old-vs-new summary (masked DirectGeoLocalDeg)")
    out.append(f"old_json={old.path.as_posix()}")
    out.append(f"new_json={new.path.as_posix()}")
    out.append("mask_common=cycle>=1,drop_wrap")
    out.append("")
    for sec_id, sec_title, bones, sics in sections:
        out.append(f"## {sec_id}) {sec_title}")
        out.append("")
        old_stats = _calc_stats(old, bones=bones, sics=sics)
        new_stats = _calc_stats(new, bones=bones, sics=sics)
        out.extend(_render_run_block(old, old_stats, cycle_gte=1, drop_wrap=True, sics=sics, bones=bones))
        out.append("")
        out.extend(_render_run_block(new, new_stats, cycle_gte=1, drop_wrap=True, sics=sics, bones=bones))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _build_calf_per_sic(old: RunData, new: RunData, out_dir: Path) -> Dict[str, Any]:
    if "calf_r" not in old.names:
        raise SystemExit("[FATAL] calf_r not found in bone_names")
    j = int(old.names.index("calf_r"))
    m = old.mask & new.mask
    sics = sorted(set(int(x) for x in old.sics[m].tolist()) | set(int(x) for x in new.sics[m].tolist()))

    rows: List[Dict[str, Any]] = []
    for sic in sics:
        sm = m & (old.sics == int(sic))
        vb = old.mat[sm, j]
        vn = new.mat[sm, j]
        vb = vb[np.isfinite(vb)]
        vn = vn[np.isfinite(vn)]
        base = float(vb.mean()) if vb.size > 0 else float("nan")
        now = float(vn.mean()) if vn.size > 0 else float("nan")
        dlt = float(now - base) if math.isfinite(base) and math.isfinite(now) else float("nan")
        pct = _pct_delta(base, now)
        rows.append(
            {
                "sic": int(sic),
                "base_deg": base,
                "new_deg": now,
                "delta_deg": dlt,
                "delta_pct": pct,
            }
        )

    csv_path = out_dir / "calf_r_per_sic_old_vs_new.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sic", "base_deg", "new_deg", "delta_deg", "delta_pct"])
        for r in rows:
            w.writerow(
                [
                    r["sic"],
                    _fmt(r["base_deg"], 6),
                    _fmt(r["new_deg"], 6),
                    _fmt(r["delta_deg"], 6),
                    _fmt(r["delta_pct"], 3),
                ]
            )

    top_new_worst = sorted(rows, key=lambda r: (r["new_deg"] if math.isfinite(r["new_deg"]) else -np.inf), reverse=True)[:10]
    top_reg = sorted(rows, key=lambda r: (r["delta_deg"] if math.isfinite(r["delta_deg"]) else -np.inf), reverse=True)[:10]
    top_imp = sorted(rows, key=lambda r: (r["delta_deg"] if math.isfinite(r["delta_deg"]) else np.inf))[:10]

    md_lines: List[str] = []
    md_lines.append("# calf_r per-SIC old-vs-new (cycle>=1, drop_wrap)")
    md_lines.append("")
    md_lines.append(f"- old_json: {old.path.as_posix()}")
    md_lines.append(f"- new_json: {new.path.as_posix()}")
    md_lines.append("")
    md_lines.append("## Full table")
    md_lines.append("")
    md_lines.append("| sic | base_deg | new_deg | delta_deg | delta_pct |")
    md_lines.append("|---:|---:|---:|---:|---:|")
    for r in rows:
        md_lines.append(
            f"| {r['sic']} | {_fmt(r['base_deg'])} | {_fmt(r['new_deg'])} | {_fmt_signed(r['delta_deg'])} | {_fmt_signed(r['delta_pct'], 3)}% |"
        )

    def _append_rank(title: str, data: List[Dict[str, Any]]) -> None:
        md_lines.append("")
        md_lines.append(f"## {title}")
        md_lines.append("")
        md_lines.append("| sic | base_deg | new_deg | delta_deg | delta_pct |")
        md_lines.append("|---:|---:|---:|---:|---:|")
        for r in data:
            md_lines.append(
                f"| {r['sic']} | {_fmt(r['base_deg'])} | {_fmt(r['new_deg'])} | {_fmt_signed(r['delta_deg'])} | {_fmt_signed(r['delta_pct'], 3)}% |"
            )

    _append_rank("Top new worst SICs (by new_deg)", top_new_worst)
    _append_rank("Largest regressions (by delta_deg)", top_reg)
    _append_rank("Largest improvements (by delta_deg)", top_imp)

    md_path = out_dir / "calf_r_per_sic_old_vs_new.md"
    md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")

    return {
        "rows": rows,
        "top_new_worst": top_new_worst,
        "top_regressions": top_reg,
        "top_improvements": top_imp,
    }


def _build_global_signal_summary(old: RunData, new: RunData, out_dir: Path) -> Dict[str, Any]:
    if old.names != new.names or int(old.root) != int(new.root):
        raise SystemExit("[FATAL] old/new mismatch in bone_names or root_idx")

    m = old.mask & new.mask
    idx = [i for i in range(len(old.names)) if i != int(old.root)]
    if not idx or not bool(m.any()):
        raise SystemExit("[FATAL] empty masked selection for global summary")

    old_sub = old.mat[m][:, idx]
    new_sub = new.mat[m][:, idx]
    dlt_sub = new_sub - old_sub

    old_vals = old_sub[np.isfinite(old_sub)]
    new_vals = new_sub[np.isfinite(new_sub)]
    dlt_vals = dlt_sub[np.isfinite(dlt_sub)]

    mean_old = float(old_vals.mean()) if old_vals.size > 0 else float("nan")
    mean_new = float(new_vals.mean()) if new_vals.size > 0 else float("nan")
    mean_dlt = float(mean_new - mean_old) if math.isfinite(mean_old) and math.isfinite(mean_new) else float("nan")

    per_bone: List[Dict[str, Any]] = []
    for col, i in enumerate(idx):
        vb = old_sub[:, col]
        vn = new_sub[:, col]
        vb = vb[np.isfinite(vb)]
        vn = vn[np.isfinite(vn)]
        if vb.size == 0 or vn.size == 0:
            continue
        mb = float(vb.mean())
        mn = float(vn.mean())
        per_bone.append(
            {
                "bone": old.names[i],
                "base": mb,
                "new": mn,
                "delta": float(mn - mb),
            }
        )
    per_bone.sort(key=lambda r: r["delta"], reverse=True)

    regress = [r for r in per_bone if r["delta"] > 0.0]
    improve = [r for r in per_bone if r["delta"] < 0.0]

    leg8 = ["thigh_r", "calf_r", "foot_r", "ball_r", "thigh_l", "calf_l", "foot_l", "ball_l"]
    leg8_set = set(leg8)
    leg_rows = [r for r in per_bone if r["bone"] in leg8_set]
    nonleg_rows = [r for r in per_bone if r["bone"] not in leg8_set]

    def _mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([float(r[key]) for r in rows]))

    leg_old = _mean(leg_rows, "base")
    leg_new = _mean(leg_rows, "new")
    leg_dlt = float(leg_new - leg_old) if math.isfinite(leg_old) and math.isfinite(leg_new) else float("nan")

    non_old = _mean(nonleg_rows, "base")
    non_new = _mean(nonleg_rows, "new")
    non_dlt = float(non_new - non_old) if math.isfinite(non_old) and math.isfinite(non_new) else float("nan")

    improved_ratio = float(np.mean(dlt_vals < 0.0)) if dlt_vals.size > 0 else float("nan")
    worse_ratio = float(np.mean(dlt_vals > 0.0)) if dlt_vals.size > 0 else float("nan")
    median_dlt = float(np.median(dlt_vals)) if dlt_vals.size > 0 else float("nan")

    top_reg = per_bone[:10]
    top_imp = sorted(per_bone, key=lambda r: r["delta"])[:10]

    lines: List[str] = []
    lines.append("# Global signal summary (old vs new Stage7.2)")
    lines.append(f"old_json={old.path.as_posix()}")
    lines.append(f"new_json={new.path.as_posix()}")
    lines.append("mask=cycle>=1,drop_wrap")
    lines.append("")
    lines.append("[overall]")
    lines.append(f"mean_old={_fmt(mean_old)}")
    lines.append(f"mean_new={_fmt(mean_new)}")
    lines.append(f"mean_delta={_fmt_signed(mean_dlt)}")
    lines.append(f"bones_excl_root={len(per_bone)}")
    lines.append(f"bones_regress_by_mean={len(regress)}")
    lines.append(f"bones_improve_by_mean={len(improve)}")
    lines.append("")
    lines.append("[region_split]")
    lines.append(f"leg8_mean_old={_fmt(leg_old)}")
    lines.append(f"leg8_mean_new={_fmt(leg_new)}")
    lines.append(f"leg8_mean_delta={_fmt_signed(leg_dlt)}")
    lines.append(f"non_leg_mean_old={_fmt(non_old)}")
    lines.append(f"non_leg_mean_new={_fmt(non_new)}")
    lines.append(f"non_leg_mean_delta={_fmt_signed(non_dlt)}")
    lines.append("")
    lines.append("[pointwise_signal]")
    lines.append(f"points={int(dlt_vals.size)}")
    lines.append(f"improved_ratio={_fmt(improved_ratio)}")
    lines.append(f"worse_ratio={_fmt(worse_ratio)}")
    lines.append(f"median_delta={_fmt_signed(median_dlt)}")
    lines.append("")
    lines.append("[top_regressions_by_mean]")
    for r in top_reg:
        lines.append(
            f"{r['bone']}: base={_fmt(r['base'])} new={_fmt(r['new'])} delta={_fmt_signed(r['delta'])}"
        )
    lines.append("")
    lines.append("[top_improvements_by_mean]")
    for r in top_imp:
        lines.append(
            f"{r['bone']}: base={_fmt(r['base'])} new={_fmt(r['new'])} delta={_fmt_signed(r['delta'])}"
        )

    out_path = out_dir / "global_signal_summary.txt"
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return {
        "mean_old": mean_old,
        "mean_new": mean_new,
        "mean_delta": mean_dlt,
        "per_bone": per_bone,
        "leg8_mean_old": leg_old,
        "leg8_mean_new": leg_new,
        "leg8_mean_delta": leg_dlt,
        "non_leg_mean_old": non_old,
        "non_leg_mean_new": non_new,
        "non_leg_mean_delta": non_dlt,
        "improved_ratio": improved_ratio,
        "worse_ratio": worse_ratio,
        "median_delta": median_dlt,
    }


def _write_gate_json(
    out_dir: Path,
    *,
    old: RunData,
    new: RunData,
    global_summary: Dict[str, Any],
) -> None:
    def _metric(bones: Optional[Sequence[str]], sic_spec: Optional[str]) -> float:
        sics = _parse_sic_spec(sic_spec) if sic_spec else None
        old_s = _calc_stats(old, bones=bones, sics=sics)["mean_deg"]
        new_s = _calc_stats(new, bones=bones, sics=sics)["mean_deg"]
        return float(old_s), float(new_s)

    old_sic1215, new_sic1215 = _metric(["foot_l", "ball_l"], "12-15")
    old_calf_g, new_calf_g = _metric(["calf_r"], None)
    old_calf_24, new_calf_24 = _metric(["calf_r"], "2-4")

    leg_delta = float(global_summary["leg8_mean_delta"])
    nonleg_delta = float(global_summary["non_leg_mean_delta"])
    g_old = float(global_summary["mean_old"])
    g_new = float(global_summary["mean_new"])
    g_rel_pct = _pct_delta(g_old, g_new)

    gate = {
        "leg8_mean_delta": leg_delta,
        "non_leg_mean_delta": nonleg_delta,
        "global_mean_old": g_old,
        "global_mean_new": g_new,
        "global_mean_rel_delta_pct": g_rel_pct,
        "sic12_15_footl_balll_old": old_sic1215,
        "sic12_15_footl_balll_new": new_sic1215,
        "calf_r_global_old": old_calf_g,
        "calf_r_global_new": new_calf_g,
        "calf_r_sic2_4_old": old_calf_24,
        "calf_r_sic2_4_new": new_calf_24,
        "gate_keep_lower_body": bool(leg_delta <= -0.05 and new_sic1215 < old_sic1215),
        "gate_fix_non_leg": bool(nonleg_delta <= 0.015 and g_rel_pct <= 3.0),
        "gate_calf_main": bool(new_calf_g <= old_calf_g * 1.05),
        "gate_calf_aux": bool(new_calf_24 <= old_calf_24 * 1.50),
    }
    (out_dir / "gate_metrics.json").write_text(json.dumps(gate, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-json", type=str, required=True)
    ap.add_argument("--new-json", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    old = _load_run(Path(args.old_json).expanduser())
    new = _load_run(Path(args.new_json).expanduser())
    if old.names != new.names or int(old.root) != int(new.root) or old.mat.shape != new.mat.shape:
        raise SystemExit("[FATAL] old/new mismatch in names/root/shape")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_text = _build_summary_metrics_text(old, new)
    (out_dir / "summary_metrics.txt").write_text(summary_text, encoding="utf-8")
    _ = _build_calf_per_sic(old, new, out_dir)
    gsum = _build_global_signal_summary(old, new, out_dir)
    _write_gate_json(out_dir, old=old, new=new, global_summary=gsum)

    print(f"[OK] wrote: {(out_dir / 'summary_metrics.txt').as_posix()}")
    print(f"[OK] wrote: {(out_dir / 'calf_r_per_sic_old_vs_new.csv').as_posix()}")
    print(f"[OK] wrote: {(out_dir / 'calf_r_per_sic_old_vs_new.md').as_posix()}")
    print(f"[OK] wrote: {(out_dir / 'global_signal_summary.txt').as_posix()}")
    print(f"[OK] wrote: {(out_dir / 'gate_metrics.json').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
