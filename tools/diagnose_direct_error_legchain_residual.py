#!/usr/bin/env python3
"""Leg-chain residual decomposition for DirectGeoLocalDeg error-only diagnostics.

Input JSON is produced by:
  tools/diagnose_direct_error_only_decomp.py

Goal:
- within a target phase (default: phase_left_stance), split SICs into spike/non-spike
- decompose focus-joint residuals (delta vs global) by leg chain (calf/foot/ball)
- identify which chain/joint dominates spike SICs
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _parse_csv(spec: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for tok in str(spec or "").split(","):
        s = tok.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _parse_int_csv(spec: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for tok in str(spec or "").split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            v = int(s)
        except Exception:
            raise SystemExit(f"[FATAL] invalid int token: {s}")
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _summary(vals: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray([_safe_float(v) for v in vals], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "pos": 0,
            "neg": 0,
            "zero": 0,
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25.0)),
        "p75": float(np.percentile(arr, 75.0)),
        "pos": int(np.sum(arr > 0.0)),
        "neg": int(np.sum(arr < 0.0)),
        "zero": int(np.sum(arr == 0.0)),
    }


def _cohen_d(a: Sequence[float], b: Sequence[float]) -> float:
    xa = np.asarray([_safe_float(v) for v in a], dtype=np.float64)
    xb = np.asarray([_safe_float(v) for v in b], dtype=np.float64)
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    if xa.size < 2 or xb.size < 2:
        return float("nan")
    ma, mb = float(xa.mean()), float(xb.mean())
    va, vb = float(xa.var(ddof=1)), float(xb.var(ddof=1))
    na, nb = int(xa.size), int(xb.size)
    den = float((na - 1) * va + (nb - 1) * vb)
    if den <= 1e-12:
        return float("nan")
    sp = math.sqrt(den / float(na + nb - 2))
    if sp <= 1e-12:
        return float("nan")
    return float((ma - mb) / sp)


def _group_stats(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = _summary([_safe_float((r.get("focus_joint_delta_vs_global", {}) or {}).get(k, float("nan"))) for r in rows])
    return out


def _extract_chain(rows: List[Dict[str, Any]], left_keys: List[str], right_keys: List[str]) -> Tuple[List[float], List[float], List[float], List[float]]:
    left_vals: List[float] = []
    right_vals: List[float] = []
    gap_vals: List[float] = []
    foot_bias_vals: List[float] = []
    lk0 = left_keys[1] if len(left_keys) >= 2 else left_keys[0]
    rk0 = right_keys[1] if len(right_keys) >= 2 else right_keys[0]
    for r in rows:
        fd = r.get("focus_joint_delta_vs_global", {}) or {}
        l = [_safe_float(fd.get(k, float("nan"))) for k in left_keys]
        rr = [_safe_float(fd.get(k, float("nan"))) for k in right_keys]
        if not all(math.isfinite(v) for v in l + rr):
            continue
        lmu = float(np.mean(l))
        rmu = float(np.mean(rr))
        left_vals.append(lmu)
        right_vals.append(rmu)
        gap_vals.append(rmu - lmu)
        foot_bias_vals.append(_safe_float(fd.get(rk0, float("nan"))) - _safe_float(fd.get(lk0, float("nan"))))
    return left_vals, right_vals, gap_vals, foot_bias_vals


def _build_markdown(out: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Direct Error Leg-Chain Residual Decomposition")
    lines.append("")
    lines.append(f"- source: `{out.get('source_error_json', '')}`")
    lines.append(f"- target phase: `{out.get('target_phase', '')}`")
    lines.append(f"- spike criterion: `{out.get('spike_criterion', '')}`")
    lines.append("")

    lines.append("## Split")
    lines.append("")
    lines.append(f"- phase rows: `{int(out.get('phase_row_count', 0))}`")
    lines.append(f"- spike SICs: `{out.get('spike_sics', [])}`")
    lines.append(
        f"- spike count: `{int(out.get('spike_count', 0))}`, control count: `{int(out.get('control_count', 0))}`"
    )
    lines.append("")

    chain = out.get("chain_summary", {}) if isinstance(out.get("chain_summary"), dict) else {}
    lines.append("## Chain Summary")
    lines.append("")
    lines.append("|metric|spike_mean|control_mean|delta|cohen_d|")
    lines.append("|:--|--:|--:|--:|--:|")
    for k in ["left_chain_mean", "right_chain_mean", "chain_gap_r_minus_l", "foot_bias_r_minus_l"]:
        obj = chain.get(k, {}) if isinstance(chain.get(k), dict) else {}
        lines.append(
            f"|`{k}`|{_safe_float(obj.get('spike_mean', float('nan'))):.4f}|"
            f"{_safe_float(obj.get('control_mean', float('nan'))):.4f}|"
            f"{_safe_float(obj.get('delta', float('nan'))):.4f}|"
            f"{_safe_float(obj.get('cohen_d', float('nan'))):.4f}|"
        )
    lines.append("")

    comp = out.get("component_summary", {}) if isinstance(out.get("component_summary"), dict) else {}
    lines.append("## Component Delta (spike - control)")
    lines.append("")
    lines.append("|joint|spike_mean|control_mean|delta|spike_pos/neg|")
    lines.append("|:--|--:|--:|--:|:--|")
    ranked = sorted(
        [
            (
                k,
                _safe_float((v or {}).get("spike", {}).get("mean", float("nan"))),
                _safe_float((v or {}).get("control", {}).get("mean", float("nan"))),
                _safe_float((v or {}).get("delta", float("nan"))),
                int((v or {}).get("spike", {}).get("pos", 0) or 0),
                int((v or {}).get("spike", {}).get("neg", 0) or 0),
            )
            for k, v in comp.items()
        ],
        key=lambda x: abs(x[3]),
        reverse=True,
    )
    for k, sm, cm, dv, pos, neg in ranked:
        lines.append(f"|`{k}`|{sm:.4f}|{cm:.4f}|{dv:.4f}|`{pos}/{neg}`|")
    lines.append("")

    dom = out.get("dominance", {}) if isinstance(out.get("dominance"), dict) else {}
    lines.append("## Dominance in Spike SICs")
    lines.append("")
    lines.append(f"- abs-dominant counts: `{dom.get('abs_dominant_counts', {})}`")
    lines.append(f"- positive-dominant counts: `{dom.get('pos_dominant_counts', {})}`")
    lines.append("")

    top = out.get("spike_rows", []) if isinstance(out.get("spike_rows"), list) else []
    if top:
        lines.append("## Spike SIC Detail")
        lines.append("")
        lines.append("|sic|err_focus_mean|chain_L|chain_R|gap(R-L)|foot_bias(R-L)|abs_dominant|")
        lines.append("|---:|---:|---:|---:|---:|---:|:--|")
        for r in top:
            lines.append(
                f"|{int(r.get('sic', -1))}|{_safe_float(r.get('err_focus_mean', float('nan'))):.4f}|"
                f"{_safe_float(r.get('chain_left_mean', float('nan'))):.4f}|"
                f"{_safe_float(r.get('chain_right_mean', float('nan'))):.4f}|"
                f"{_safe_float(r.get('chain_gap_r_minus_l', float('nan'))):.4f}|"
                f"{_safe_float(r.get('foot_bias_r_minus_l', float('nan'))):.4f}|"
                f"{r.get('abs_dominant_joint', 'NA')}|"
            )
        lines.append("")

    conclusion = out.get("conclusion", {}) if isinstance(out.get("conclusion"), dict) else {}
    if conclusion:
        lines.append("## Conclusion")
        lines.append("")
        lines.append(f"- trigger_chain: `{conclusion.get('trigger_chain', 'NA')}`")
        lines.append(f"- trigger_joint: `{conclusion.get('trigger_joint', 'NA')}`")
        lines.append(f"- evidence: `{conclusion.get('evidence', 'NA')}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Leg-chain residual decomposition from DirectGeoLocalDeg error-only JSON")
    ap.add_argument("--error-json", type=str, required=True)
    ap.add_argument("--target-phase", type=str, default="phase_left_stance")
    ap.add_argument("--spike-z", type=float, default=0.8)
    ap.add_argument("--spike-sics", type=str, default="")
    ap.add_argument("--left-chain", type=str, default="calf_l,foot_l,ball_l")
    ap.add_argument("--right-chain", type=str, default="calf_r,foot_r,ball_r")
    ap.add_argument("--focus-joints", type=str, default="calf_l,calf_r,foot_l,foot_r,ball_l,ball_r")
    ap.add_argument("--reference-sics", type=str, default="12,14,54,55")
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--out-md", type=str, default="")
    args = ap.parse_args()

    src = Path(args.error_json).expanduser().resolve()
    obj = json.loads(src.read_text(encoding="utf-8"))

    by_sic = obj.get("by_sic", [])
    if not isinstance(by_sic, list) or not by_sic:
        raise SystemExit("[FATAL] by_sic missing/empty")

    left_chain = _parse_csv(args.left_chain)
    right_chain = _parse_csv(args.right_chain)
    focus_joints = _parse_csv(args.focus_joints)
    ref_sics = _parse_int_csv(args.reference_sics)

    if len(left_chain) != 3 or len(right_chain) != 3:
        raise SystemExit("[FATAL] left/right chain must each contain 3 joints (calf/foot/ball)")

    phase = str(args.target_phase)
    phase_rows = [r for r in by_sic if str(r.get("phase_major", "")) == phase]
    if not phase_rows:
        raise SystemExit(f"[FATAL] no rows for phase: {phase}")

    g = obj.get("global", {}) if isinstance(obj.get("global"), dict) else {}
    g_mu = _safe_float((g.get("err_focus", {}) or {}).get("mean", float("nan")))
    g_std = _safe_float((g.get("err_focus", {}) or {}).get("std", float("nan")))

    if str(args.spike_sics).strip():
        spike_sics = _parse_int_csv(args.spike_sics)
        criterion = f"manual_sics={spike_sics}"
    else:
        if not (math.isfinite(g_mu) and math.isfinite(g_std)):
            raise SystemExit("[FATAL] invalid global mean/std for auto spike threshold")
        thr = float(g_mu + float(args.spike_z) * g_std)
        spike_sics = [int(r.get("sic")) for r in phase_rows if _safe_float((r.get("err_focus", {}) or {}).get("mean", float("nan"))) >= thr]
        criterion = f"err_focus >= global_mean + {float(args.spike_z):.3f}*global_std (thr={thr:.4f})"

    spike_sics_set = set(int(x) for x in spike_sics)
    phase_rows_sorted = sorted(phase_rows, key=lambda r: int(r.get("sic", -1)))
    spike_rows = [r for r in phase_rows_sorted if int(r.get("sic", -1)) in spike_sics_set]
    control_rows = [r for r in phase_rows_sorted if int(r.get("sic", -1)) not in spike_sics_set]

    if not spike_rows:
        raise SystemExit("[FATAL] spike set resolved to empty")

    keys = focus_joints
    spike_comp = _group_stats(spike_rows, keys)
    ctrl_comp = _group_stats(control_rows, keys)

    comp_summary: Dict[str, Any] = {}
    for k in keys:
        sm = _safe_float((spike_comp.get(k, {}) or {}).get("mean", float("nan")))
        cm = _safe_float((ctrl_comp.get(k, {}) or {}).get("mean", float("nan")))
        comp_summary[k] = {
            "spike": spike_comp.get(k, {}),
            "control": ctrl_comp.get(k, {}),
            "delta": float(sm - cm) if math.isfinite(sm) and math.isfinite(cm) else float("nan"),
            "cohen_d": _cohen_d(
                [_safe_float((r.get("focus_joint_delta_vs_global", {}) or {}).get(k, float("nan"))) for r in spike_rows],
                [_safe_float((r.get("focus_joint_delta_vs_global", {}) or {}).get(k, float("nan"))) for r in control_rows],
            ),
        }

    s_l, s_r, s_gap, s_foot = _extract_chain(spike_rows, left_chain, right_chain)
    c_l, c_r, c_gap, c_foot = _extract_chain(control_rows, left_chain, right_chain)

    chain_summary = {
        "left_chain_mean": {
            "spike_mean": _safe_float(np.mean(s_l) if s_l else float("nan")),
            "control_mean": _safe_float(np.mean(c_l) if c_l else float("nan")),
            "delta": _safe_float((np.mean(s_l) if s_l else float("nan")) - (np.mean(c_l) if c_l else float("nan"))),
            "cohen_d": _cohen_d(s_l, c_l),
        },
        "right_chain_mean": {
            "spike_mean": _safe_float(np.mean(s_r) if s_r else float("nan")),
            "control_mean": _safe_float(np.mean(c_r) if c_r else float("nan")),
            "delta": _safe_float((np.mean(s_r) if s_r else float("nan")) - (np.mean(c_r) if c_r else float("nan"))),
            "cohen_d": _cohen_d(s_r, c_r),
        },
        "chain_gap_r_minus_l": {
            "spike_mean": _safe_float(np.mean(s_gap) if s_gap else float("nan")),
            "control_mean": _safe_float(np.mean(c_gap) if c_gap else float("nan")),
            "delta": _safe_float((np.mean(s_gap) if s_gap else float("nan")) - (np.mean(c_gap) if c_gap else float("nan"))),
            "cohen_d": _cohen_d(s_gap, c_gap),
        },
        "foot_bias_r_minus_l": {
            "spike_mean": _safe_float(np.mean(s_foot) if s_foot else float("nan")),
            "control_mean": _safe_float(np.mean(c_foot) if c_foot else float("nan")),
            "delta": _safe_float((np.mean(s_foot) if s_foot else float("nan")) - (np.mean(c_foot) if c_foot else float("nan"))),
            "cohen_d": _cohen_d(s_foot, c_foot),
        },
    }

    all_joint_keys = list(dict.fromkeys(left_chain + right_chain))
    abs_dom = Counter()
    pos_dom = Counter()
    spike_detail: List[Dict[str, Any]] = []
    by_map = {int(r.get("sic", -1)): r for r in phase_rows_sorted}

    for r in spike_rows:
        sic = int(r.get("sic", -1))
        fd = r.get("focus_joint_delta_vs_global", {}) or {}
        vals = {k: _safe_float(fd.get(k, float("nan"))) for k in all_joint_keys}
        finite = {k: v for k, v in vals.items() if math.isfinite(v)}
        abs_dom_k = "NA"
        pos_dom_k = "NA"
        if finite:
            abs_dom_k = max(finite, key=lambda k: abs(finite[k]))
            pos_dom_k = max(finite, key=lambda k: finite[k])
            abs_dom[abs_dom_k] += 1
            pos_dom[pos_dom_k] += 1

        lmu = float(np.mean([vals[k] for k in left_chain if math.isfinite(vals[k])]))
        rmu = float(np.mean([vals[k] for k in right_chain if math.isfinite(vals[k])]))
        foot_bias = _safe_float(vals[right_chain[1]] - vals[left_chain[1]])

        spike_detail.append(
            {
                "sic": sic,
                "err_focus_mean": _safe_float((r.get("err_focus", {}) or {}).get("mean", float("nan"))),
                "chain_left_mean": lmu,
                "chain_right_mean": rmu,
                "chain_gap_r_minus_l": _safe_float(rmu - lmu),
                "foot_bias_r_minus_l": foot_bias,
                "abs_dominant_joint": abs_dom_k,
                "pos_dominant_joint": pos_dom_k,
            }
        )

    spike_detail.sort(key=lambda x: int(x["sic"]))

    # Reference SIC projection for quick compare
    ref_rows: List[Dict[str, Any]] = []
    for s in ref_sics:
        r = by_map.get(int(s))
        if r is None:
            ref_rows.append({"sic": int(s), "missing": True})
            continue
        fd = r.get("focus_joint_delta_vs_global", {}) or {}
        vals = {k: _safe_float(fd.get(k, float("nan"))) for k in all_joint_keys}
        lmu = float(np.mean([vals[k] for k in left_chain if math.isfinite(vals[k])]))
        rmu = float(np.mean([vals[k] for k in right_chain if math.isfinite(vals[k])]))
        ref_rows.append(
            {
                "sic": int(s),
                "phase_major": str(r.get("phase_major", "NA")),
                "err_focus_mean": _safe_float((r.get("err_focus", {}) or {}).get("mean", float("nan"))),
                "chain_left_mean": lmu,
                "chain_right_mean": rmu,
                "chain_gap_r_minus_l": _safe_float(rmu - lmu),
                "foot_bias_r_minus_l": _safe_float(vals[right_chain[1]] - vals[left_chain[1]]),
            }
        )

    trigger_joint = "NA"
    trigger_chain = "NA"
    evidence = "insufficient"
    if abs_dom:
        trigger_joint = abs_dom.most_common(1)[0][0]
        trigger_chain = "right_chain" if trigger_joint.endswith("_r") else "left_chain"
        gap_delta = _safe_float((chain_summary.get("chain_gap_r_minus_l", {}) or {}).get("delta", float("nan")))
        dom_ratio = _safe_float(abs_dom[trigger_joint] / max(1, len(spike_rows)))
        evidence = f"abs_dominant={trigger_joint} ({abs_dom[trigger_joint]}/{len(spike_rows)}={dom_ratio:.3f}), chain_gap_delta={gap_delta:.4f}"

    out = {
        "source_error_json": str(src),
        "target_phase": phase,
        "spike_criterion": criterion,
        "phase_row_count": int(len(phase_rows_sorted)),
        "spike_sics": [int(x) for x in sorted(spike_sics_set)],
        "spike_count": int(len(spike_rows)),
        "control_count": int(len(control_rows)),
        "left_chain": left_chain,
        "right_chain": right_chain,
        "component_summary": comp_summary,
        "chain_summary": chain_summary,
        "dominance": {
            "abs_dominant_counts": {k: int(v) for k, v in abs_dom.items()},
            "pos_dominant_counts": {k: int(v) for k, v in pos_dom.items()},
        },
        "spike_rows": spike_detail,
        "reference_rows": ref_rows,
        "conclusion": {
            "trigger_chain": trigger_chain,
            "trigger_joint": trigger_joint,
            "evidence": evidence,
        },
    }

    out_json = Path(args.out_json).expanduser().resolve() if str(args.out_json).strip() else src.with_name(src.stem + "_legchain_residual.json")
    out_md = Path(args.out_md).expanduser().resolve() if str(args.out_md).strip() else out_json.with_suffix(".md")

    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_build_markdown(out), encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
