#!/usr/bin/env python3
"""
Compare learned direct_pose_alpha distributions on spike SICs against an oracle alpha table.

Expected freerun input:
  - train.validate.run_freerun_cycles JSON with metrics_per_step entries that contain
    "DirectPoseAlpha" (enable via --export_direct_pose_alpha_series).
  - Fallback: top-level "direct_pose_alpha_series" + metrics_per_step step metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _isfinite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _as_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return float(v) if math.isfinite(v) else None


def _parse_int_csv(spec: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for tok in str(spec or "").replace(";", ",").split(","):
        t = tok.strip()
        if not t or not t.lstrip("-").isdigit():
            continue
        v = int(t)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _flag(spec: str, *, default: bool) -> bool:
    s = str(spec or "").strip().lower()
    if not s:
        return bool(default)
    if s in ("1", "true", "on", "yes", "y", "enable", "enabled"):
        return True
    if s in ("0", "false", "off", "no", "n", "disable", "disabled"):
        return False
    return bool(default)


def _canon_bone(name: Any) -> str:
    return str(name or "").strip()


def _load_oracle(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    mask = obj.get("mask", {}) if isinstance(obj, dict) else {}
    rows: List[Dict[str, Any]] = []

    # New schema: {"entries":[{"sic":..,"bone":..,"alpha":..}, ...]}
    entries = obj.get("entries", None) if isinstance(obj, dict) else None
    if isinstance(entries, list):
        for r in entries:
            if not isinstance(r, dict):
                continue
            sic = r.get("sic", None)
            bone = _canon_bone(r.get("bone", ""))
            alpha = _as_float(r.get("alpha", None))
            if isinstance(sic, int) and bone and alpha is not None and alpha > 0.0:
                rows.append({"sic": int(sic), "bone": bone, "alpha": float(alpha)})

    # Legacy schema: {"alpha_by_sic_bone": {"54":{"ball_r":4.0}}}
    if not rows and isinstance(obj, dict):
        ab = obj.get("alpha_by_sic_bone", None)
        if isinstance(ab, dict):
            for sic_k, bones in ab.items():
                try:
                    sic = int(sic_k)
                except Exception:
                    continue
                if not isinstance(bones, dict):
                    continue
                for bone, alpha in bones.items():
                    bone_c = _canon_bone(bone)
                    alpha_f = _as_float(alpha)
                    if bone_c and alpha_f is not None and alpha_f > 0.0:
                        rows.append({"sic": sic, "bone": bone_c, "alpha": float(alpha_f)})

    # De-dup by (sic,bone), keep the first.
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for r in rows:
        key = (int(r["sic"]), str(r["bone"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    uniq.sort(key=lambda x: (int(x["sic"]), str(x["bone"])))
    return uniq, (mask if isinstance(mask, dict) else {})


def _collect_alpha_rows(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = obj.get("metrics_per_step", None)
    if not isinstance(steps, list):
        raise ValueError("freerun JSON missing metrics_per_step list.")

    out: List[Dict[str, Any]] = []
    for r in steps:
        if not isinstance(r, dict):
            continue
        cyc = r.get("cycle", None)
        sic = r.get("step_in_cycle", None)
        step = r.get("step", None)
        wrap = bool(r.get("wrap_boundary_step", False))
        alpha_map = r.get("DirectPoseAlpha", None)
        if isinstance(alpha_map, dict) and alpha_map:
            clean: Dict[str, float] = {}
            for bone, v in alpha_map.items():
                b = _canon_bone(bone)
                vf = _as_float(v)
                if b and vf is not None and vf > 0.0:
                    clean[b] = float(vf)
            if clean:
                out.append(
                    {
                        "step": int(step) if isinstance(step, int) else len(out),
                        "cycle": int(cyc) if isinstance(cyc, int) else 0,
                        "sic": int(sic) if isinstance(sic, int) else -1,
                        "wrap": bool(wrap),
                        "alpha": clean,
                    }
                )

    # Fallback: reconstruct from top-level direct_pose_alpha_series if per-step map is absent.
    if out:
        return out

    series_obj = obj.get("direct_pose_alpha_series", None)
    if not isinstance(series_obj, dict):
        return out
    ser = series_obj.get("series", None)
    if not isinstance(ser, dict):
        return out
    alpha = ser.get("alpha", None)
    if not isinstance(alpha, dict):
        return out
    valid = ser.get("valid", None)
    if not isinstance(valid, list):
        valid = []

    # Keep step metadata from metrics_per_step.
    for i, r in enumerate(steps):
        if not isinstance(r, dict):
            continue
        ok = True
        if i < len(valid):
            try:
                ok = bool(int(valid[i]) != 0)
            except Exception:
                ok = False
        if not ok:
            continue
        clean: Dict[str, float] = {}
        for bone, arr in alpha.items():
            if not isinstance(arr, list) or i >= len(arr):
                continue
            vf = _as_float(arr[i])
            b = _canon_bone(bone)
            if b and vf is not None and vf > 0.0:
                clean[b] = float(vf)
        if not clean:
            continue
        cyc = r.get("cycle", None)
        sic = r.get("step_in_cycle", None)
        step = r.get("step", None)
        wrap = bool(r.get("wrap_boundary_step", False))
        out.append(
            {
                "step": int(step) if isinstance(step, int) else int(i),
                "cycle": int(cyc) if isinstance(cyc, int) else 0,
                "sic": int(sic) if isinstance(sic, int) else -1,
                "wrap": bool(wrap),
                "alpha": clean,
            }
        )
    return out


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray([float(v) for v in values if _isfinite(v)], dtype=np.float32)
    if arr.size <= 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _format(v: Any, nd: int = 4) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description="Report learned direct alpha distribution vs oracle alpha table on spike SICs.")
    ap.add_argument("--freerun-json", type=str, required=True, help="Path to *_freerun_cycles.json.")
    ap.add_argument("--oracle-table", type=str, required=True, help="Path to oracle alpha table JSON.")
    ap.add_argument("--cycle-gte", type=int, default=-1, help="Override cycle lower bound. -1 => use oracle mask/default.")
    ap.add_argument(
        "--drop-wrap",
        type=str,
        default="auto",
        choices=("auto", "on", "off"),
        help="Whether to drop wrap boundary steps. auto => use oracle mask/default on.",
    )
    ap.add_argument("--spike-sics", type=str, default="", help="Optional SIC list override (comma-separated).")
    ap.add_argument("--spike-window", type=int, default=0, help="Optional +/- window around each spike SIC (default: 0).")
    ap.add_argument("--bones", type=str, default="", help="Optional comma-separated bone subset to report.")
    ap.add_argument("--out-json", type=str, default="", help="Output JSON path.")
    ap.add_argument("--out-md", type=str, default="", help="Output markdown report path.")
    args = ap.parse_args()

    p_free = Path(args.freerun_json).expanduser().resolve()
    p_oracle = Path(args.oracle_table).expanduser().resolve()
    obj = json.loads(p_free.read_text(encoding="utf-8"))
    oracle_pairs, oracle_mask = _load_oracle(p_oracle)
    if not oracle_pairs:
        raise SystemExit("[FATAL] oracle table has no valid (sic,bone,alpha) entries.")

    rows = _collect_alpha_rows(obj)
    if not rows:
        raise SystemExit(
            "[FATAL] freerun JSON has no DirectPoseAlpha entries. Re-run freerun with --export_direct_pose_alpha_series."
        )

    if int(args.cycle_gte) >= 0:
        cycle_gte = int(args.cycle_gte)
    else:
        m = oracle_mask.get("cycle_gte", 1) if isinstance(oracle_mask, dict) else 1
        try:
            cycle_gte = max(0, int(m))
        except Exception:
            cycle_gte = 1

    if str(args.drop_wrap).strip().lower() == "auto":
        default_drop = True
        if isinstance(oracle_mask, dict) and "drop_wrap" in oracle_mask:
            default_drop = bool(oracle_mask.get("drop_wrap", True))
        drop_wrap = bool(default_drop)
    else:
        drop_wrap = _flag(str(args.drop_wrap), default=True)

    bones_filter = {_canon_bone(x) for x in str(args.bones or "").split(",") if _canon_bone(x)}
    if bones_filter:
        oracle_pairs = [r for r in oracle_pairs if str(r["bone"]) in bones_filter]
    if not oracle_pairs:
        raise SystemExit("[FATAL] No oracle entries left after --bones filtering.")

    # Spike SIC set.
    if str(args.spike_sics or "").strip():
        spike_sics = sorted(set(_parse_int_csv(args.spike_sics)))
    else:
        spike_sics = sorted(set(int(r["sic"]) for r in oracle_pairs))
    if not spike_sics:
        raise SystemExit("[FATAL] Empty spike SIC set.")

    # Optional SIC +/- window expansion (with modulo when cycle_len is known).
    cycle_len = 0
    sics_from_rows = [int(r["sic"]) for r in rows if int(r["sic"]) >= 0]
    if sics_from_rows:
        cycle_len = int(max(sics_from_rows) + 1)
    win = max(0, int(args.spike_window))
    spike_sic_set = set()
    if win <= 0:
        spike_sic_set = set(int(s) for s in spike_sics)
    else:
        for s in spike_sics:
            s_i = int(s)
            for d in range(-win, win + 1):
                v = s_i + int(d)
                if cycle_len > 0:
                    v = int(v % cycle_len)
                spike_sic_set.add(int(v))

    # Filter freerun rows once.
    filtered_rows = []
    for r in rows:
        cyc = int(r.get("cycle", 0))
        sic = int(r.get("sic", -1))
        wrap = bool(r.get("wrap", False))
        if cyc < int(cycle_gte):
            continue
        if drop_wrap and wrap:
            continue
        if sic not in spike_sic_set:
            continue
        filtered_rows.append(r)

    # Build lookup by (sic,bone).
    by_pair: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    by_bone: Dict[str, List[float]] = defaultdict(list)
    for r in filtered_rows:
        sic = int(r["sic"])
        alpha_map = r.get("alpha", {})
        if not isinstance(alpha_map, dict):
            continue
        for bone, v in alpha_map.items():
            b = _canon_bone(bone)
            vf = _as_float(v)
            if not b or vf is None or vf <= 0.0:
                continue
            by_pair[(sic, b)].append(float(vf))
            by_bone[b].append(float(vf))

    # Oracle-aligned pair report.
    pair_rows: List[Dict[str, Any]] = []
    for it in oracle_pairs:
        sic = int(it["sic"])
        bone = str(it["bone"])
        oracle_alpha = float(it["alpha"])
        vals = by_pair.get((sic, bone), [])
        st = _stats(vals)
        out = {
            "sic": sic,
            "bone": bone,
            "oracle_alpha": oracle_alpha,
            "stats": st,
        }
        if st.get("n", 0) > 0:
            mean = float(st["mean"])
            p50 = float(st["p50"])
            out["ratio_mean_vs_oracle"] = float(mean / oracle_alpha) if oracle_alpha > 0 else float("nan")
            out["ratio_p50_vs_oracle"] = float(p50 / oracle_alpha) if oracle_alpha > 0 else float("nan")
            try:
                log_ratio = np.log(np.asarray(vals, dtype=np.float32) / float(oracle_alpha))
                out["log_ratio_mean"] = float(np.mean(log_ratio))
            except Exception:
                pass
        pair_rows.append(out)

    # Bone-level summary over spike SIC rows.
    bone_rows: List[Dict[str, Any]] = []
    oracle_by_bone: Dict[str, List[float]] = defaultdict(list)
    for r in oracle_pairs:
        oracle_by_bone[str(r["bone"])].append(float(r["alpha"]))
    for bone in sorted(set(list(by_bone.keys()) + list(oracle_by_bone.keys()))):
        vals = by_bone.get(bone, [])
        st = _stats(vals)
        ent: Dict[str, Any] = {
            "bone": bone,
            "stats": st,
            "oracle_alpha_mean": float(np.mean(np.asarray(oracle_by_bone.get(bone, [float("nan")]), dtype=np.float32))),
            "oracle_pairs": int(len(oracle_by_bone.get(bone, []))),
        }
        if st.get("n", 0) > 0 and ent["oracle_pairs"] > 0 and _isfinite(ent["oracle_alpha_mean"]):
            ent["ratio_mean_vs_oracle_mean"] = float(float(st["mean"]) / float(ent["oracle_alpha_mean"]))
        bone_rows.append(ent)

    global_vals: List[float] = []
    for vals in by_bone.values():
        global_vals.extend(vals)

    report: Dict[str, Any] = {
        "freerun_json": str(p_free),
        "oracle_table": str(p_oracle),
        "filter": {
            "cycle_gte": int(cycle_gte),
            "drop_wrap": bool(drop_wrap),
            "spike_sics_input": [int(x) for x in spike_sics],
            "spike_window": int(win),
            "spike_sics_applied": [int(x) for x in sorted(spike_sic_set)],
            "bones_filter": sorted(list(bones_filter)) if bones_filter else None,
        },
        "counts": {
            "rows_total": int(len(rows)),
            "rows_filtered": int(len(filtered_rows)),
            "oracle_pairs": int(len(oracle_pairs)),
        },
        "pair_results": pair_rows,
        "bone_results": bone_rows,
        "global_stats": _stats(global_vals),
    }

    out_json = Path(args.out_json).expanduser() if str(args.out_json or "").strip() else p_free.with_name(
        p_free.stem + "_direct_pose_alpha_vs_oracle.json"
    )
    out_md = Path(args.out_md).expanduser() if str(args.out_md or "").strip() else p_free.with_name(
        p_free.stem + "_direct_pose_alpha_vs_oracle.md"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Direct Pose Alpha vs Oracle (Spike SIC)")
    lines.append("")
    lines.append(f"- freerun_json: `{p_free}`")
    lines.append(f"- oracle_table: `{p_oracle}`")
    lines.append(
        f"- filter: cycle>={cycle_gte}, drop_wrap={drop_wrap}, spike_sics={sorted(list(spike_sic_set))}, rows={len(filtered_rows)}/{len(rows)}"
    )
    lines.append("")
    lines.append("## Oracle-Aligned Pairs")
    lines.append("")
    lines.append("| sic | bone | oracle_alpha | n | learned_mean | learned_p50 | learned_p90 | ratio_mean/oracle | ratio_p50/oracle |")
    lines.append("| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in pair_rows:
        st = r.get("stats", {})
        n = int(st.get("n", 0) or 0)
        lines.append(
            "| {sic} | {bone} | {oa} | {n} | {mean} | {p50} | {p90} | {rm} | {rp} |".format(
                sic=int(r["sic"]),
                bone=str(r["bone"]),
                oa=_format(float(r["oracle_alpha"])),
                n=n,
                mean=_format(float(st["mean"])) if n > 0 else "n/a",
                p50=_format(float(st["p50"])) if n > 0 else "n/a",
                p90=_format(float(st["p90"])) if n > 0 else "n/a",
                rm=_format(float(r.get("ratio_mean_vs_oracle", float("nan")))) if n > 0 else "n/a",
                rp=_format(float(r.get("ratio_p50_vs_oracle", float("nan")))) if n > 0 else "n/a",
            )
        )
    lines.append("")
    lines.append("## Bone Summary")
    lines.append("")
    lines.append("| bone | n | learned_mean | learned_p50 | learned_p90 | oracle_alpha_mean | ratio_mean/oracle_mean |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in bone_rows:
        st = r.get("stats", {})
        n = int(st.get("n", 0) or 0)
        lines.append(
            "| {bone} | {n} | {mean} | {p50} | {p90} | {oa} | {rm} |".format(
                bone=str(r["bone"]),
                n=n,
                mean=_format(float(st["mean"])) if n > 0 else "n/a",
                p50=_format(float(st["p50"])) if n > 0 else "n/a",
                p90=_format(float(st["p90"])) if n > 0 else "n/a",
                oa=_format(float(r.get("oracle_alpha_mean", float("nan")))),
                rm=_format(float(r.get("ratio_mean_vs_oracle_mean", float("nan"))))
                if n > 0 and _isfinite(r.get("oracle_alpha_mean", float("nan")))
                else "n/a",
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] wrote JSON: {out_json}")
    print(f"[ok] wrote MD:   {out_md}")
    g = report.get("global_stats", {})
    print(
        "[summary] rows_filtered={rf}/{rt} global_n={n} global_mean={m}".format(
            rf=int(report["counts"]["rows_filtered"]),
            rt=int(report["counts"]["rows_total"]),
            n=int(g.get("n", 0) or 0),
            m=_format(float(g.get("mean", float("nan")))),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
