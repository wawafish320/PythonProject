#!/usr/bin/env python3
"""
Segment-level analysis for `train.validate.run_freerun_cycles` JSON outputs.

Focuses on a specific joint (keybone) and a phase window (step_in_cycle range),
and optionally prints contact-plan/meas/gt statistics when available.

Examples
--------
python tools/analyze_freerun_segment.py \\
  --segment calf_l:15-19 \\
  --exclude-round0 \\
  --case "BASE=model=debug_output/.../Walk_F_freerun_cycles.json" \\
  --case "WB=contacts_wb=debug_output/.../Walk_F_freerun_cycles.json"
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


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _as_list_floats(x: Any) -> Optional[List[float]]:
    if not isinstance(x, list) or not x:
        return None
    out: List[float] = []
    for v in x:
        fv = _as_float(v)
        if fv is None:
            return None
        out.append(fv)
    return out


def _get_step_in_cycle(st: Dict[str, Any]) -> Optional[int]:
    si = st.get("step_in_cycle", None)
    return int(si) if isinstance(si, int) else None


def _iter_segment(
    steps: List[Dict[str, Any]],
    *,
    exclude_round0: bool,
    step_lo: int,
    step_hi: int,
) -> Iterable[Dict[str, Any]]:
    for st in steps:
        cy = st.get("cycle", None)
        if exclude_round0 and isinstance(cy, int) and cy == 0:
            continue
        si = _get_step_in_cycle(st)
        if si is None:
            continue
        if int(step_lo) <= int(si) <= int(step_hi):
            yield st


def _kb(st: Dict[str, Any], key: str, bone: str) -> Optional[float]:
    d = st.get(key, None)
    if not isinstance(d, dict):
        return None
    return _as_float(d.get(bone, None))


def _fmt_deg(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.2f}°"


def _fmt_float(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.4f}"


def _fmt_list(xs: Optional[List[float]]) -> str:
    if not xs:
        return "NA"
    return "[" + ", ".join(f"{v:.3f}" for v in xs) + "]"


def _parse_segment(spec: str) -> Tuple[str, int, int]:
    # "calf_l:15-19"
    if ":" not in spec:
        raise ValueError(f"Invalid --segment spec (expected bone:lo-hi): {spec}")
    bone, rng = spec.split(":", 1)
    bone = bone.strip()
    if "-" not in rng:
        raise ValueError(f"Invalid --segment range (expected lo-hi): {spec}")
    lo_s, hi_s = rng.split("-", 1)
    lo = int(lo_s.strip())
    hi = int(hi_s.strip())
    if lo > hi:
        lo, hi = hi, lo
    return bone, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a phase window for a keybone from freerun_cycles JSON.")
    ap.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case spec: 'Label=path/to/*_freerun_cycles.json' (repeatable).",
    )
    ap.add_argument("--segment", type=str, required=True, help="Segment spec: bone:lo-hi (step_in_cycle range).")
    ap.add_argument("--exclude-round0", action="store_true", help="Use cycles>=1 only.")
    args = ap.parse_args()

    if not args.case:
        raise SystemExit("Need at least one --case Label=Path.")

    bone, lo, hi = _parse_segment(str(args.segment))

    cases: List[Tuple[str, Path]] = []
    for spec in args.case:
        if "=" not in str(spec):
            raise SystemExit(f"Invalid --case spec (expected Label=Path): {spec}")
        label, path = str(spec).split("=", 1)
        label = label.strip()
        p = Path(path).expanduser()
        if not p.is_file():
            raise SystemExit(f"--case file not found: {p}")
        cases.append((label, p))

    print(f"[Segment] bone={bone} step_in_cycle=[{lo},{hi}] exclude_round0={bool(args.exclude_round0)}")
    print()

    headers = [
        "Case",
        "N",
        "BaseMean",
        "DirectMean",
        "BlendMean",
        "P(Direct<Base)",
        "KeyBoneLambdaEffMean",
        "LambdaEffMean",
        "ContactGTPerC",
        "ContactMeasPerC",
        "ContactPlanPerC",
        "ContactErrAbsPerC",
        "ContactMeasGtAbsMean",
        "ContactPlanGtAbsMean",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")

    for label, path in cases:
        obj = _load_json(path)
        steps = obj.get("metrics_per_step", None)
        if not isinstance(steps, list) or not steps:
            raise SystemExit(f"Invalid JSON: missing metrics_per_step list: {path}")

        seg = list(_iter_segment(steps, exclude_round0=bool(args.exclude_round0), step_lo=lo, step_hi=hi))
        n = len(seg)
        if n == 0:
            row = [label, "0"] + ["NA"] * (len(headers) - 2)
            print("| " + " | ".join(row) + " |")
            continue

        base = _nanmean([_kb(st, "KeyBoneGeoLocalDeg", bone) for st in seg])
        direct = _nanmean([_kb(st, "KeyBoneDirectGeoLocalDeg", bone) for st in seg])
        blend = _nanmean([_kb(st, "KeyBoneBlendGeoLocalDeg", bone) for st in seg])

        wins = 0
        tot = 0
        for st in seg:
            b = _kb(st, "KeyBoneGeoLocalDeg", bone)
            d = _kb(st, "KeyBoneDirectGeoLocalDeg", bone)
            if b is None or d is None:
                continue
            tot += 1
            if d < b:
                wins += 1
        p_db = None if tot == 0 else float(wins) / float(tot)

        kb_lam_eff = _nanmean([_kb(st, "KeyBoneLambdaEff", bone) for st in seg])
        lam_eff_mean = _nanmean([_as_float(st.get("LambdaEffMean", None)) for st in seg])

        # Contacts (only present when --log_contacts is enabled in freerun_cycles).
        c_gt = _as_list_floats(seg[0].get("ContactGTPerC", None))
        c_meas = _as_list_floats(seg[0].get("ContactMeasPerC", None))
        c_plan = _as_list_floats(seg[0].get("ContactPlanPerC", None))
        c_err = _as_list_floats(seg[0].get("ContactErrAbsPerC", None))

        # Prefer segment-mean if available.
        def mean_list(key: str) -> Optional[List[float]]:
            acc: List[List[float]] = []
            for st in seg:
                v = _as_list_floats(st.get(key, None))
                if v is None:
                    continue
                acc.append(v)
            if not acc:
                return None
            dim = len(acc[0])
            if any(len(a) != dim for a in acc):
                return None
            out: List[float] = []
            for j in range(dim):
                out.append(float(mean(a[j] for a in acc)))
            return out

        c_gt_m = mean_list("ContactGTPerC") or c_gt
        c_meas_m = mean_list("ContactMeasPerC") or c_meas
        c_plan_m = mean_list("ContactPlanPerC") or c_plan
        c_err_m = mean_list("ContactErrAbsPerC") or c_err
        meas_gt_abs_mean = _nanmean([_as_float(st.get("ContactMeasGtAbsMean", None)) for st in seg])
        plan_gt_abs_mean = _nanmean([_as_float(st.get("ContactPlanGtAbsMean", None)) for st in seg])

        row = [
            label,
            str(n),
            _fmt_deg(base),
            _fmt_deg(direct),
            _fmt_deg(blend),
            _fmt_float(p_db),
            _fmt_float(kb_lam_eff),
            _fmt_float(lam_eff_mean),
            _fmt_list(c_gt_m),
            _fmt_list(c_meas_m),
            _fmt_list(c_plan_m),
            _fmt_list(c_err_m),
            _fmt_float(meas_gt_abs_mean),
            _fmt_float(plan_gt_abs_mean),
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
