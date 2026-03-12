#!/usr/bin/env python3
"""
Analyze per-step relationship between:
  - NOAPPLY keybone_omega.series (signed omega component on the configured axis; typically z)
  - APPLY direct_hinge_series (delta_eff / delta_raw / gate)

This is meant to localize cases like:
  "omega_z(noapply) is large but delta(apply) is small"
within a hard subset (cycle/contact filters) and optionally a phase window (step_in_cycle).

Example:
  python tools/analyze_hinge_omega_delta_phase.py \\
    --noapply debug_output/_verify_step1_axisoracle_sup1_noapply_5clip_logc_r5_td_d3 \\
    --apply   debug_output/_verify_step1_axisoracle_sup1_apply_5clip_logc_r5_td_hingeon_d3 \\
    --bones calf_r --branch direct --min-cycle 1 \\
    --contact-source gt --contact-index 1 --contact-value 0 \\
    --phase-range 15 23 \\
    --contact-thresh 0.5 1e-4 \\
    --err-thresh 20 --delta-small 1 \\
    --out-csv /tmp/hinge_omega_delta.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clip_from_path(p: Path) -> str:
    name = p.name
    suf = "_freerun_cycles.json"
    if name.endswith(suf):
        return name[: -len(suf)]
    return name


def _glob_inputs(specs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for s in specs:
        ss = str(s).strip()
        if not ss:
            continue
        p = Path(ss).expanduser()
        if p.is_dir():
            out += sorted(p.glob("*_freerun_cycles.json"))
            continue
        if p.is_file():
            out.append(p)
            continue
        for m in sorted(glob.glob(ss)):
            mp = Path(m).expanduser()
            if mp.is_file():
                out.append(mp)
    # Dedupe while preserving order.
    seen = set()
    uniq: List[Path] = []
    for p in out:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        uniq.append(p)
    return uniq


def _resolve_contact_key(src: str) -> str:
    s = str(src).strip().lower()
    return {"gt": "ContactGTPerC", "plan": "ContactPlanPerC", "meas": "ContactMeasPerC"}.get(s, "ContactGTPerC")


def _mean(xs: Sequence[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return None
    return float(sum(vals) / float(len(vals)))


def _pearson_corr(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
    if len(pairs) < 2:
        return None
    xs = [float(a) for a, _ in pairs if a is not None and math.isfinite(float(a))]
    ys = [float(b) for _, b in pairs if b is not None and math.isfinite(float(b))]
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    mx = _mean(xs)
    my = _mean(ys)
    if mx is None or my is None:
        return None
    vx = sum((x - mx) ** 2 for x in xs) / float(len(xs))
    vy = sum((y - my) ** 2 for y in ys) / float(len(ys))
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / float(len(xs))
    return float(cov / ((vx**0.5) * (vy**0.5)))


def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "NA"
    try:
        if not math.isfinite(float(x)):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


def _phase_of_step(i: int, *, steps: Optional[List[Dict[str, Any]]], cycle_len: int) -> Optional[int]:
    if steps is not None and i < len(steps):
        si = steps[i].get("step_in_cycle", None)
        if isinstance(si, int):
            return int(si)
    if int(cycle_len) > 0:
        return int(i % int(cycle_len))
    return None


def _cycle_of_step(i: int, *, steps: Optional[List[Dict[str, Any]]], cycle_len: int) -> Optional[int]:
    if steps is not None and i < len(steps):
        cy = steps[i].get("cycle", None)
        if isinstance(cy, int):
            return int(cy)
    if int(cycle_len) > 0:
        return int(i // int(cycle_len))
    return None


def _iter_selected_indices(
    *,
    steps: Optional[List[Dict[str, Any]]],
    cycle_len: int,
    min_cycle: int,
    contact_key: str,
    contact_idx: Optional[int],
    contact_value: Optional[int],
    contact_thresh: float,
    phase_min: Optional[int],
    phase_max: Optional[int],
    max_len: int,
) -> Iterable[int]:
    for i in range(int(max_len)):
        cy = _cycle_of_step(i, steps=steps, cycle_len=int(cycle_len))
        if int(cy or 0) < int(min_cycle):
            continue

        if phase_min is not None or phase_max is not None:
            ph = _phase_of_step(i, steps=steps, cycle_len=int(cycle_len))
            if ph is None:
                continue
            if phase_min is not None and int(ph) < int(phase_min):
                continue
            if phase_max is not None and int(ph) > int(phase_max):
                continue

        if contact_value is not None:
            if steps is None or i >= len(steps):
                continue
            rec = steps[i]
            c = rec.get(contact_key, None)
            if not isinstance(c, list) or contact_idx is None or contact_idx < 0 or contact_idx >= len(c):
                continue
            try:
                v = float(c[int(contact_idx)])
            except Exception:
                continue
            state = 1 if v >= float(contact_thresh) else 0
            if int(state) != int(contact_value):
                continue

        yield int(i)


def _extract_keybone_omega_axis_series(
    obj: Dict[str, Any], *, branch: str, bone: str
) -> Tuple[List[float], List[float], Optional[List[Dict[str, Any]]], int]:
    ko = obj.get("keybone_omega", None)
    if not isinstance(ko, dict):
        raise ValueError("missing keybone_omega (run with --export_keybone_omega)")
    series = ko.get("series", None)
    if not isinstance(series, dict):
        raise ValueError("missing keybone_omega.series (run with --export_keybone_omega_series)")
    sbranches = series.get("branches", None)
    if not isinstance(sbranches, dict):
        raise ValueError("invalid keybone_omega.series: missing branches")
    bdat = sbranches.get(str(branch), None)
    if not isinstance(bdat, dict):
        raise ValueError(f"invalid keybone_omega.series: missing branches.{branch}")
    omega_map = bdat.get("omega_axis_deg", None)
    ang_map = bdat.get("ang_deg", None)
    if not isinstance(omega_map, dict) or not isinstance(ang_map, dict):
        raise ValueError(f"invalid keybone_omega.series.branches.{branch}: missing omega_axis_deg/ang_deg")
    omega = omega_map.get(str(bone), None)
    ang = ang_map.get(str(bone), None)
    if not isinstance(omega, list) or not isinstance(ang, list):
        raise ValueError(f"missing series for bone={bone!r} under branch={branch!r}")

    steps = obj.get("metrics_per_step", None)
    steps = steps if isinstance(steps, list) and steps else None
    cycle_len = int(obj.get("cycle_len", 0) or 0)
    return omega, ang, steps, cycle_len


def _extract_direct_hinge_series(
    obj: Dict[str, Any],
    *,
    bone: str,
) -> Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]:
    """
    Returns:
      (delta_eff_deg, delta_raw_deg, gate, valid, valid_raw, valid_gate)
    """
    dh = obj.get("direct_hinge_series", None)
    if not isinstance(dh, dict):
        raise ValueError("missing direct_hinge_series (re-run APPLY with --export_direct_hinge_series)")
    series = dh.get("series", None)
    if not isinstance(series, dict):
        raise ValueError("invalid direct_hinge_series: missing series dict")
    delta_deg = series.get("delta_deg", None)
    delta_raw_deg = series.get("delta_raw_deg", None)
    gate = series.get("gate", None)
    valid = series.get("valid", None)
    valid_raw = series.get("valid_raw", None)
    valid_gate = series.get("valid_gate", None)
    if not isinstance(delta_deg, dict) or not isinstance(delta_raw_deg, dict) or not isinstance(gate, dict):
        raise ValueError("invalid direct_hinge_series.series: missing delta_deg/delta_raw_deg/gate dicts")
    if not isinstance(valid, list) or not isinstance(valid_raw, list) or not isinstance(valid_gate, list):
        raise ValueError("invalid direct_hinge_series.series: missing valid/valid_raw/valid_gate lists")
    if str(bone) not in delta_deg or str(bone) not in delta_raw_deg or str(bone) not in gate:
        raise ValueError(f"direct_hinge_series missing bone={bone!r} (available={list(delta_deg.keys())})")
    d = delta_deg[str(bone)]
    dr = delta_raw_deg[str(bone)]
    g = gate[str(bone)]
    if not isinstance(d, list) or not isinstance(dr, list) or not isinstance(g, list):
        raise ValueError(f"direct_hinge_series series for bone={bone!r} is not a list")
    return (
        [float(x) if x is not None else 0.0 for x in d],
        [float(x) if x is not None else 0.0 for x in dr],
        [float(x) if x is not None else 0.0 for x in g],
        [1 if int(x) == 1 else 0 for x in valid],
        [1 if int(x) == 1 else 0 for x in valid_raw],
        [1 if int(x) == 1 else 0 for x in valid_gate],
    )


@dataclass
class Row:
    clip: str
    bone: str
    branch: str
    idx: int
    cycle: Optional[int]
    step_in_cycle: Optional[int]
    contact: Optional[float]
    omega_no_deg: float
    ang_no_deg: float
    omega_ap_deg: Optional[float]
    ang_ap_deg: Optional[float]
    delta_eff_deg: Optional[float]
    delta_raw_deg: Optional[float]
    gate: Optional[float]
    valid_delta: int
    valid_raw: int
    valid_gate: int


def _write_csv(path: Path, rows: Sequence[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "clip",
                "bone",
                "branch",
                "idx",
                "cycle",
                "step_in_cycle",
                "contact",
                "omega_no_deg",
                "ang_no_deg",
                "omega_ap_deg",
                "ang_ap_deg",
                "delta_eff_deg",
                "delta_raw_deg",
                "gate",
                "valid_delta",
                "valid_raw",
                "valid_gate",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.clip,
                    r.bone,
                    r.branch,
                    int(r.idx),
                    "" if r.cycle is None else int(r.cycle),
                    "" if r.step_in_cycle is None else int(r.step_in_cycle),
                    "" if r.contact is None else float(r.contact),
                    float(r.omega_no_deg),
                    float(r.ang_no_deg),
                    "" if r.omega_ap_deg is None else float(r.omega_ap_deg),
                    "" if r.ang_ap_deg is None else float(r.ang_ap_deg),
                    "" if r.delta_eff_deg is None else float(r.delta_eff_deg),
                    "" if r.delta_raw_deg is None else float(r.delta_raw_deg),
                    "" if r.gate is None else float(r.gate),
                    int(r.valid_delta),
                    int(r.valid_raw),
                    int(r.valid_gate),
                ]
            )


def _summarize_rows(rows: Sequence[Row], *, err_thresh: float, delta_small: float) -> Dict[str, Any]:
    omega_no = [r.omega_no_deg for r in rows]
    omega_ap = [r.omega_ap_deg for r in rows if r.omega_ap_deg is not None]

    delta_eff = [r.delta_eff_deg for r in rows if r.delta_eff_deg is not None]
    delta_raw = [r.delta_raw_deg for r in rows if r.delta_raw_deg is not None]
    gate = [r.gate for r in rows if r.gate is not None]

    mean_abs_eff = _mean([abs(float(x)) for x in delta_eff if x is not None]) if delta_eff else None
    mean_abs_raw = _mean([abs(float(x)) for x in delta_raw if x is not None]) if delta_raw else None
    ratio_eff_over_raw = None
    if mean_abs_eff is not None and mean_abs_raw is not None and float(mean_abs_raw) > 1e-9:
        ratio_eff_over_raw = float(mean_abs_eff) / float(mean_abs_raw)

    corr_eff = _pearson_corr([(r.omega_no_deg, r.delta_eff_deg) for r in rows if r.delta_eff_deg is not None])
    corr_raw = _pearson_corr([(r.omega_no_deg, r.delta_raw_deg) for r in rows if r.delta_raw_deg is not None])
    corr_abs = _pearson_corr(
        [(abs(r.omega_no_deg), abs(r.delta_eff_deg)) for r in rows if r.delta_eff_deg is not None]
    )

    high = [r for r in rows if float(r.ang_no_deg) > float(err_thresh)]
    high_valid = [r for r in high if r.delta_eff_deg is not None]
    p_small = None
    if high_valid:
        p_small = float(sum(1 for r in high_valid if abs(float(r.delta_eff_deg)) < float(delta_small))) / float(
            len(high_valid)
        )

    return {
        "n": int(len(rows)),
        "n_high": int(len(high)),
        "mean_abs_omega_no": _mean([abs(x) for x in omega_no]),
        "mean_abs_omega_ap": _mean([abs(x) for x in omega_ap]) if omega_ap else None,
        "mean_abs_delta_eff": mean_abs_eff,
        "mean_abs_delta_raw": mean_abs_raw,
        "ratio_eff_over_raw": ratio_eff_over_raw,
        "mean_gate": _mean([float(g) for g in gate]) if gate else None,
        "corr(omega_no,delta_eff)": corr_eff,
        "corr(omega_no,delta_raw)": corr_raw,
        "corr(|omega_no|,|delta_eff|)": corr_abs,
        "p_high(|delta_eff|<delta_small)": p_small,
    }


def _group_by_phase(rows: Sequence[Row]) -> Dict[int, List[Row]]:
    out: Dict[int, List[Row]] = {}
    for r in rows:
        if r.step_in_cycle is None:
            continue
        out.setdefault(int(r.step_in_cycle), []).append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze NOAPPLY omega vs APPLY hinge delta by phase.")
    ap.add_argument("--noapply", nargs="+", required=True, help="NOAPPLY input(s): dir/file/glob.")
    ap.add_argument("--apply", nargs="+", required=True, help="APPLY input(s): dir/file/glob.")
    ap.add_argument("--bones", nargs="+", default=["calf_r"], help="Bone names to analyze.")
    ap.add_argument("--branch", type=str, default="direct", choices=("inc", "direct", "blend"))
    ap.add_argument("--min-cycle", type=int, default=1, help="Include only steps with cycle >= this value.")
    ap.add_argument("--contact-source", type=str, default="gt", choices=("gt", "plan", "meas"))
    ap.add_argument("--contact-side", type=str, default=None, choices=("l", "r"))
    ap.add_argument("--contact-index", type=int, default=None)
    ap.add_argument("--contact-value", type=int, default=0, help="Filter binarized contact to 0(swing) or 1(stance).")
    ap.add_argument(
        "--contact-thresh",
        nargs="+",
        type=float,
        default=[0.5, 1e-4],
        help="Contact thresholds to report (repeatable). Default: 0.5 1e-4",
    )
    ap.add_argument("--phase-range", nargs=2, type=int, default=None, metavar=("PHASE_MIN", "PHASE_MAX"))
    ap.add_argument("--err-thresh", type=float, default=20.0, help="High-error threshold on ang_no_deg (deg).")
    ap.add_argument("--delta-small", type=float, default=1.0, help="Small-delta threshold in deg.")
    ap.add_argument("--out-csv", type=str, default=None, help="Optional CSV path to dump selected rows.")
    ap.add_argument("--strict-clips", action="store_true", help="Error if NOAPPLY/APPLY clip sets differ.")
    args = ap.parse_args()

    noapply_paths = _glob_inputs(args.noapply)
    apply_paths = _glob_inputs(args.apply)
    if not noapply_paths:
        raise SystemExit("NOAPPLY inputs resolved to no files.")
    if not apply_paths:
        raise SystemExit("APPLY inputs resolved to no files.")

    def _index_by_clip(paths: Sequence[Path]) -> Dict[str, Path]:
        m: Dict[str, Path] = {}
        for p in paths:
            try:
                obj = _load_json(p)
                clip = str(obj.get("clip") or _clip_from_path(p))
            except Exception:
                clip = _clip_from_path(p)
            m[clip] = p
        return m

    noapply_by_clip = _index_by_clip(noapply_paths)
    apply_by_clip = _index_by_clip(apply_paths)
    clips_no = set(noapply_by_clip.keys())
    clips_ap = set(apply_by_clip.keys())
    common = sorted(clips_no & clips_ap)
    if bool(args.strict_clips) and (clips_no != clips_ap):
        miss_a = sorted(clips_no - clips_ap)
        miss_b = sorted(clips_ap - clips_no)
        raise SystemExit(f"Clip mismatch. Missing in APPLY: {miss_a} ; missing in NOAPPLY: {miss_b}")
    if not common:
        raise SystemExit("No common clips between NOAPPLY/APPLY inputs.")

    contact_idx: Optional[int] = None
    if args.contact_index is not None:
        contact_idx = int(args.contact_index)
    elif args.contact_side is not None:
        contact_idx = 0 if str(args.contact_side).strip().lower() == "l" else 1
    elif args.contact_value is not None:
        # Default to right contact when filtering is requested without specifying channel.
        contact_idx = 1

    contact_value = args.contact_value
    if contact_value is not None and int(contact_value) not in (0, 1):
        raise SystemExit("--contact-value must be 0 or 1.")
    contact_value = int(contact_value) if contact_value is not None else None

    phase_min: Optional[int] = None
    phase_max: Optional[int] = None
    if args.phase_range is not None:
        phase_min = int(args.phase_range[0])
        phase_max = int(args.phase_range[1])
        if phase_min > phase_max:
            raise SystemExit("--phase-range expects PHASE_MIN <= PHASE_MAX")

    bones = [str(b) for b in (args.bones or []) if str(b).strip()]
    if not bones:
        raise SystemExit("Empty --bones.")

    contact_key = _resolve_contact_key(args.contact_source)

    print("[AnalyzeHingeOmegaDeltaPhase]")
    print(f"- branch={args.branch} bones={','.join(bones)} min_cycle={int(args.min_cycle)}")
    print(
        f"- contact={contact_key}"
        + (f"[{contact_idx}]" if contact_idx is not None else "")
        + (f" -> {contact_value}" if contact_value is not None else " (no filter)")
    )
    if phase_min is not None or phase_max is not None:
        print(f"- phase_range={phase_min}..{phase_max}")
    print(f"- clips_common={len(common)} ({', '.join(common)})")
    print()

    all_rows: List[Row] = []

    for ct in args.contact_thresh:
        ct = float(ct)
        print(f"[ContactThresh] {ct:g}")

        rows_by_bone: Dict[str, List[Row]] = {b: [] for b in bones}

        for clip in common:
            no_obj = _load_json(noapply_by_clip[clip])
            ap_obj = _load_json(apply_by_clip[clip])

            for bone in bones:
                omega_no, ang_no, steps_no, cycle_len = _extract_keybone_omega_axis_series(
                    no_obj, branch=str(args.branch), bone=str(bone)
                )
                omega_ap: Optional[List[float]] = None
                ang_ap: Optional[List[float]] = None
                try:
                    omega_ap, ang_ap, _, _ = _extract_keybone_omega_axis_series(
                        ap_obj, branch=str(args.branch), bone=str(bone)
                    )
                except Exception:
                    omega_ap = None
                    ang_ap = None

                delta_eff, delta_raw, gate, valid, valid_raw, valid_gate = _extract_direct_hinge_series(
                    ap_obj, bone=str(bone)
                )

                steps_len = len(steps_no) if steps_no is not None else len(omega_no)
                max_len = min(
                    len(omega_no),
                    len(ang_no),
                    steps_len,
                    len(delta_eff),
                    len(delta_raw),
                    len(gate),
                    len(valid),
                    len(valid_raw),
                    len(valid_gate),
                )
                if omega_ap is not None:
                    max_len = min(max_len, len(omega_ap))
                if ang_ap is not None:
                    max_len = min(max_len, len(ang_ap))

                sel = list(
                    _iter_selected_indices(
                        steps=steps_no,
                        cycle_len=int(cycle_len),
                        min_cycle=int(args.min_cycle),
                        contact_key=str(contact_key),
                        contact_idx=contact_idx,
                        contact_value=contact_value,
                        contact_thresh=float(ct),
                        phase_min=phase_min,
                        phase_max=phase_max,
                        max_len=int(max_len),
                    )
                )
                if not sel:
                    continue

                for i in sel:
                    rec = steps_no[i] if steps_no is not None and i < len(steps_no) else {}
                    cy = _cycle_of_step(i, steps=steps_no, cycle_len=int(cycle_len))
                    ph = _phase_of_step(i, steps=steps_no, cycle_len=int(cycle_len))

                    cval: Optional[float] = None
                    if contact_value is not None and contact_idx is not None and isinstance(rec, dict):
                        c = rec.get(contact_key, None)
                        if isinstance(c, list) and 0 <= int(contact_idx) < len(c):
                            try:
                                cval = float(c[int(contact_idx)])
                            except Exception:
                                cval = None

                    de = float(delta_eff[i]) if int(valid[i]) == 1 else None
                    dr = float(delta_raw[i]) if int(valid_raw[i]) == 1 else None
                    gg = float(gate[i]) if int(valid_gate[i]) == 1 else None

                    row = Row(
                        clip=str(clip),
                        bone=str(bone),
                        branch=str(args.branch),
                        idx=int(i),
                        cycle=cy,
                        step_in_cycle=ph,
                        contact=cval,
                        omega_no_deg=float(omega_no[i]),
                        ang_no_deg=float(ang_no[i]),
                        omega_ap_deg=float(omega_ap[i]) if omega_ap is not None else None,
                        ang_ap_deg=float(ang_ap[i]) if ang_ap is not None else None,
                        delta_eff_deg=de,
                        delta_raw_deg=dr,
                        gate=gg,
                        valid_delta=int(valid[i]),
                        valid_raw=int(valid_raw[i]),
                        valid_gate=int(valid_gate[i]),
                    )
                    rows_by_bone[str(bone)].append(row)

        for bone in bones:
            rows = rows_by_bone.get(str(bone), [])
            if not rows:
                print(f"[Bone] {bone}: no rows (after filters).")
                continue

            s = _summarize_rows(rows, err_thresh=float(args.err_thresh), delta_small=float(args.delta_small))
            print(f"[Bone] {bone}")
            print(f"- n={s['n']} n_high(ang_no>{args.err_thresh:g})={s['n_high']}")
            print(
                "- "
                + ", ".join(
                    [
                        f"mean|omega_no|={_fmt(s['mean_abs_omega_no'], 3)}",
                        f"mean|omega_ap|={_fmt(s['mean_abs_omega_ap'], 3)}",
                        f"mean|delta_eff|={_fmt(s['mean_abs_delta_eff'], 3)}",
                        f"mean|delta_raw|={_fmt(s['mean_abs_delta_raw'], 3)}",
                        f"|eff|/|raw|={_fmt(s['ratio_eff_over_raw'], 3)}",
                        f"mean_gate={_fmt(s['mean_gate'], 3)}",
                    ]
                )
            )
            print(
                "- "
                + ", ".join(
                    [
                        f"corr(omega_no,delta_eff)={_fmt(s['corr(omega_no,delta_eff)'], 3)}",
                        f"corr(omega_no,delta_raw)={_fmt(s['corr(omega_no,delta_raw)'], 3)}",
                        f"corr(|omega_no|,|delta_eff|)={_fmt(s['corr(|omega_no|,|delta_eff|)'], 3)}",
                        f"p_high(|delta_eff|<{args.delta_small:g})={_fmt(s['p_high(|delta_eff|<delta_small)'], 3)}",
                    ]
                )
            )

            ph_bins = _group_by_phase(rows)
            if ph_bins:
                ph_list = sorted(ph_bins.keys())
                if phase_min is not None or phase_max is not None:
                    ph_list = [
                        p
                        for p in ph_list
                        if (phase_min is None or p >= phase_min) and (phase_max is None or p <= phase_max)
                    ]

                print("[PhaseBins]")
                hdr = [
                    "phase",
                    "n",
                    "mean|omega_no|",
                    "mean|omega_ap|",
                    "mean|delta_eff|",
                    "mean|delta_raw|",
                    "mean_gate",
                    f"p_high(ang_no>{args.err_thresh:g},|delta_eff|<{args.delta_small:g})",
                ]
                print("| " + " | ".join(hdr) + " |")
                print("|" + "|".join(["---"] + ["---:"] * (len(hdr) - 1)) + "|")
                for ph in ph_list:
                    rs = ph_bins[ph]
                    high = [
                        r
                        for r in rs
                        if float(r.ang_no_deg) > float(args.err_thresh) and r.delta_eff_deg is not None
                    ]
                    p_small = None
                    if high:
                        p_small = float(
                            sum(1 for r in high if abs(float(r.delta_eff_deg)) < float(args.delta_small))
                        ) / float(len(high))

                    print(
                        "| "
                        + " | ".join(
                            [
                                str(int(ph)),
                                str(len(rs)),
                                _fmt(_mean([abs(r.omega_no_deg) for r in rs]), 3),
                                _fmt(_mean([abs(r.omega_ap_deg) for r in rs if r.omega_ap_deg is not None]), 3),
                                _fmt(
                                    _mean([abs(float(r.delta_eff_deg)) for r in rs if r.delta_eff_deg is not None]),
                                    3,
                                ),
                                _fmt(
                                    _mean([abs(float(r.delta_raw_deg)) for r in rs if r.delta_raw_deg is not None]),
                                    3,
                                ),
                                _fmt(_mean([float(r.gate) for r in rs if r.gate is not None]), 3),
                                _fmt(p_small, 3),
                            ]
                        )
                        + " |"
                    )

            print()

            all_rows.extend(rows)

        print()

    if args.out_csv is not None:
        out = Path(str(args.out_csv)).expanduser()
        _write_csv(out, all_rows)
        print(f"[CSV] wrote {out} (rows={len(all_rows)})")


if __name__ == "__main__":
    main()

