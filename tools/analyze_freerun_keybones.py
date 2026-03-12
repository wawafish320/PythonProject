#!/usr/bin/env python3
"""
Analyze per-keybone errors (and optional per-keybone lambda) from
`train.validate.run_freerun_cycles` JSON outputs.

This is intended for quickly answering questions like:
  - Is λ choosing the wrong branch for a joint in a certain phase?
  - How much headroom exists if we could pick min(base, direct) per-step?

Usage
-----
python tools/analyze_freerun_keybones.py \
  --json debug_output/_tmp_errbreak/v1_d1_lbnohist_v1/Walk_F_freerun_cycles.json \
  --exclude-round0

Optionally focus on certain bones:
python tools/analyze_freerun_keybones.py --json ... --bones calf_l calf_r thigh_l thigh_r
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _nanmean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return None if not vals else float(mean(vals))


def _get_keybones(steps: List[Dict[str, Any]]) -> List[str]:
    for st in steps:
        kb = st.get("KeyBoneBlendGeoLocalDeg", None)
        if isinstance(kb, dict) and kb:
            return list(kb.keys())
    return []


def _iter_selected_steps(
    steps: List[Dict[str, Any]], *, exclude_round0: bool
) -> Iterable[Dict[str, Any]]:
    for st in steps:
        cy = st.get("cycle", None)
        if exclude_round0 and isinstance(cy, int) and cy == 0:
            continue
        yield st


def _safe_get(st: Dict[str, Any], key: str) -> Optional[float]:
    v = st.get(key, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_get_kb(st: Dict[str, Any], key: str, bone: str) -> Optional[float]:
    d = st.get(key, None)
    if not isinstance(d, dict):
        return None
    v = d.get(bone, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _phase_mean_max(
    steps_sel: List[Dict[str, Any]], *, bone: str, key: str, cycle_len: int
) -> Tuple[Optional[int], Optional[float]]:
    by_phase: List[List[float]] = [[] for _ in range(max(0, int(cycle_len)))]
    for st in steps_sel:
        si = st.get("step_in_cycle", None)
        if not isinstance(si, int) or si < 0 or si >= len(by_phase):
            continue
        v = _safe_get_kb(st, key, bone)
        if v is None:
            continue
        by_phase[si].append(v)
    if not by_phase:
        return None, None
    phase_means: List[Optional[float]] = [None if not xs else float(mean(xs)) for xs in by_phase]
    best_i: Optional[int] = None
    best_v: Optional[float] = None
    for i, v in enumerate(phase_means):
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_v = v
            best_i = i
    return best_i, best_v


def _fmt_deg(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.2f}°"


def _fmt_float(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze keybone errors from freerun_cycles JSON.")
    ap.add_argument("--json", type=str, required=True, help="Path to *_freerun_cycles.json")
    ap.add_argument("--exclude-round0", action="store_true", help="Use R1-4 (or later) only.")
    ap.add_argument("--bones", nargs="*", default=None, help="Subset of keybones to print.")
    ap.add_argument(
        "--lambda-thresh",
        type=float,
        default=0.5,
        help="Threshold for reporting selection rate metrics when KeyBoneLambda(Eff) is present.",
    )
    ap.add_argument(
        "--topk-phases",
        type=int,
        default=5,
        help="For each bone, print top-k phases by (blend-direct) diff.",
    )
    args = ap.parse_args()

    path = Path(args.json).expanduser()
    obj = _load_json(path)
    steps = obj.get("metrics_per_step", None)
    if not isinstance(steps, list) or not steps:
        raise SystemExit("Invalid JSON: missing metrics_per_step list.")

    cycle_len = int(obj.get("cycle_len", 0) or 0)

    steps_sel = list(_iter_selected_steps(steps, exclude_round0=bool(args.exclude_round0)))
    if not steps_sel:
        raise SystemExit("No steps selected (maybe exclude_round0 removed everything?).")

    # ---- Global summary (mean over selected steps) ----
    base = _nanmean([_safe_get(st, "GeoLocalDeg") for st in steps_sel])
    direct = _nanmean([_safe_get(st, "DirectGeoLocalDeg") for st in steps_sel])
    blend = _nanmean([_safe_get(st, "BlendGeoLocalDeg") for st in steps_sel])
    blend_w = _nanmean([_safe_get(st, "BlendGeoLocalDegWeighted") for st in steps_sel])

    # Oracle (per-step choose min(base, direct) at the aggregated metric level).
    oracle_vals: List[Optional[float]] = []
    for st in steps_sel:
        b = _safe_get(st, "GeoLocalDeg")
        d = _safe_get(st, "DirectGeoLocalDeg")
        if b is None or d is None:
            oracle_vals.append(None)
        else:
            oracle_vals.append(min(b, d))
    oracle = _nanmean(oracle_vals)

    print(f"[JSON] {path}")
    if bool(args.exclude_round0):
        print("[Steps] exclude R0 (use cycles>=1)")
    print(f"[Mean] GeoLocalDeg(base)={_fmt_deg(base)} Direct={_fmt_deg(direct)} Blend={_fmt_deg(blend)}")
    print(f"[Mean] BlendGeoLocalDegWeighted={_fmt_deg(blend_w)} Oracle(min(base,direct))={_fmt_deg(oracle)}")
    print()

    keybones_all = _get_keybones(steps)
    if not keybones_all:
        print("[KeyBone] No KeyBone* fields found in metrics_per_step; nothing to do.")
        return

    bones: List[str]
    if args.bones:
        bones = [b for b in args.bones if b in keybones_all]
        missing = [b for b in args.bones if b not in keybones_all]
        if missing:
            print("[Warn] Unknown bones ignored:", ", ".join(missing))
    else:
        bones = keybones_all

    def kb_mean(key: str, bone: str) -> Optional[float]:
        return _nanmean([_safe_get_kb(st, key, bone) for st in steps_sel])

    def kb_oracle_mean(bone: str) -> Optional[float]:
        vals: List[Optional[float]] = []
        for st in steps_sel:
            b = _safe_get_kb(st, "KeyBoneGeoLocalDeg", bone)
            d = _safe_get_kb(st, "KeyBoneDirectGeoLocalDeg", bone)
            if b is None or d is None:
                vals.append(None)
            else:
                vals.append(min(b, d))
        return _nanmean(vals)

    def kb_direct_better_frac(bone: str) -> Optional[float]:
        tot = 0
        win = 0
        for st in steps_sel:
            b = _safe_get_kb(st, "KeyBoneGeoLocalDeg", bone)
            d = _safe_get_kb(st, "KeyBoneDirectGeoLocalDeg", bone)
            if b is None or d is None:
                continue
            tot += 1
            if d < b:
                win += 1
        return None if tot == 0 else float(win) / float(tot)

    # Optional: per-keybone lambda diagnostics (if present).
    has_kb_lam = any(isinstance(st.get("KeyBoneLambda", None), dict) for st in steps_sel)
    has_kb_lam_eff = any(isinstance(st.get("KeyBoneLambdaEff", None), dict) for st in steps_sel)
    lam_key = "KeyBoneLambdaEff" if has_kb_lam_eff else ("KeyBoneLambda" if has_kb_lam else None)

    def kb_lambda_stats(bone: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if lam_key is None:
            return None, None, None
        lam_all: List[float] = []
        lam_when_direct_better: List[float] = []
        lam_when_direct_worse: List[float] = []
        for st in steps_sel:
            lam = _safe_get_kb(st, lam_key, bone)
            b = _safe_get_kb(st, "KeyBoneGeoLocalDeg", bone)
            d = _safe_get_kb(st, "KeyBoneDirectGeoLocalDeg", bone)
            if lam is None:
                continue
            lam_all.append(lam)
            if b is None or d is None:
                continue
            if d < b:
                lam_when_direct_better.append(lam)
            else:
                lam_when_direct_worse.append(lam)
        return _nanmean(lam_all), _nanmean(lam_when_direct_better), _nanmean(lam_when_direct_worse)

    def kb_lambda_select_rate(bone: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Report how often lambda is "selecting direct" (lambda > thresh) conditioned on:
          - direct is better (d < b)
          - direct is not better (d >= b)
        """
        if lam_key is None:
            return None, None
        thr = float(args.lambda_thresh)
        tot_db = 0
        hit_db = 0
        tot_dw = 0
        hit_dw = 0
        for st in steps_sel:
            lam = _safe_get_kb(st, lam_key, bone)
            b = _safe_get_kb(st, "KeyBoneGeoLocalDeg", bone)
            d = _safe_get_kb(st, "KeyBoneDirectGeoLocalDeg", bone)
            if lam is None or b is None or d is None:
                continue
            if d < b:
                tot_db += 1
                if lam > thr:
                    hit_db += 1
            else:
                tot_dw += 1
                if lam > thr:
                    hit_dw += 1
        p_db = None if tot_db == 0 else float(hit_db) / float(tot_db)
        p_dw = None if tot_dw == 0 else float(hit_dw) / float(tot_dw)
        return p_db, p_dw

    # ---- Keybone table ----
    headers = [
        "Bone",
        "Base",
        "Direct",
        "Blend",
        "Oracle(min)",
        "Blend-Oracle",
        "P(Direct<Base)",
    ]
    if lam_key is not None:
        headers += [f"{lam_key}Mean", f"{lam_key}|DirectBetter", f"{lam_key}|DirectWorse"]
        headers += [
            f"P({lam_key}>{args.lambda_thresh:.2f}|DirectBetter)",
            f"P({lam_key}>{args.lambda_thresh:.2f}|DirectWorse)",
        ]

    rows: List[List[str]] = []
    for bone in bones:
        b = kb_mean("KeyBoneGeoLocalDeg", bone)
        d = kb_mean("KeyBoneDirectGeoLocalDeg", bone)
        bl = kb_mean("KeyBoneBlendGeoLocalDeg", bone)
        o = kb_oracle_mean(bone)
        gap = None if (bl is None or o is None) else float(bl - o)
        frac = kb_direct_better_frac(bone)
        row = [
            bone,
            _fmt_deg(b),
            _fmt_deg(d),
            _fmt_deg(bl),
            _fmt_deg(o),
            _fmt_deg(gap),
            _fmt_float(frac),
        ]
        if lam_key is not None:
            lam_mean, lam_db, lam_dw = kb_lambda_stats(bone)
            p_sel_db, p_sel_dw = kb_lambda_select_rate(bone)
            row += [
                _fmt_float(lam_mean),
                _fmt_float(lam_db),
                _fmt_float(lam_dw),
                _fmt_float(p_sel_db),
                _fmt_float(p_sel_dw),
            ]
        rows.append(row)

    # Sort rows by Blend-Oracle descending (largest headroom first).
    def _gap_key(r: List[str]) -> float:
        try:
            s = r[5]
            if s == "NA":
                return -1e9
            return float(s.replace("°", ""))
        except Exception:
            return -1e9

    rows_sorted = sorted(rows, key=_gap_key, reverse=True)

    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for r in rows_sorted:
        print("| " + " | ".join(r) + " |")

    # ---- Per-bone phase hotspots ----
    if cycle_len > 0:
        print()
        print("[Phase Hotspots] max mean phase (over selected cycles) per bone:")
        for bone in bones:
            p_b, v_b = _phase_mean_max(steps_sel, bone=bone, key="KeyBoneGeoLocalDeg", cycle_len=cycle_len)
            p_d, v_d = _phase_mean_max(steps_sel, bone=bone, key="KeyBoneDirectGeoLocalDeg", cycle_len=cycle_len)
            p_bl, v_bl = _phase_mean_max(steps_sel, bone=bone, key="KeyBoneBlendGeoLocalDeg", cycle_len=cycle_len)
            print(
                f"- {bone}: base@{p_b}={_fmt_deg(v_b)} direct@{p_d}={_fmt_deg(v_d)} blend@{p_bl}={_fmt_deg(v_bl)}"
            )

        # blend-direct worst phases per bone
        topk = max(0, int(args.topk_phases))
        if topk > 0:
            print()
            print(f"[Blend vs Direct] top-{topk} phases where (blend-direct) is largest (mean over cycles):")
            for bone in bones:
                # compute mean per phase
                by_phase: List[Dict[str, List[float]]] = [
                    {"base": [], "direct": [], "blend": []} for _ in range(int(cycle_len))
                ]
                for st in steps_sel:
                    si = st.get("step_in_cycle", None)
                    if not isinstance(si, int) or si < 0 or si >= int(cycle_len):
                        continue
                    b = _safe_get_kb(st, "KeyBoneGeoLocalDeg", bone)
                    d = _safe_get_kb(st, "KeyBoneDirectGeoLocalDeg", bone)
                    bl = _safe_get_kb(st, "KeyBoneBlendGeoLocalDeg", bone)
                    if b is not None:
                        by_phase[si]["base"].append(b)
                    if d is not None:
                        by_phase[si]["direct"].append(d)
                    if bl is not None:
                        by_phase[si]["blend"].append(bl)

                diffs: List[Tuple[int, float, float, float]] = []
                for i, dd in enumerate(by_phase):
                    if not dd["direct"] or not dd["blend"]:
                        continue
                    m_bl = float(mean(dd["blend"]))
                    m_d = float(mean(dd["direct"]))
                    diffs.append((i, m_bl - m_d, m_bl, m_d))
                diffs.sort(key=lambda x: x[1], reverse=True)
                diffs = diffs[:topk]
                if diffs:
                    s = ", ".join([f"{i}:{diff:+.2f}°(bl={bl:.1f},dir={d:.1f})" for i, diff, bl, d in diffs])
                    print(f"- {bone}: {s}")


if __name__ == "__main__":
    main()
