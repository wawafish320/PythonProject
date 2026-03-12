#!/usr/bin/env python3
"""
Compare hinge APPLY vs NOAPPLY runs using keybone_omega.series exported by:
  python -m train.validate.run_freerun_cycles --export_keybone_omega --export_keybone_omega_series

Why
----
When you filter by a strict contact threshold (e.g. contact<1e-4) and/or compute
tail stats (ang_deg > th), APPLY can change the tail subset size (n_tail), which
can make omega_tail_mean "look worse" due to selection effects.

This tool reports:
  - per-run stats (NOAPPLY/APPLY): n, mean_ang, P(ang>th), n_tail, omega_tail_mean/std
  - fixed-tail omega: APPLY omega mean/std on the NOAPPLY-defined tail mask
  - (optional, APPLY-only) hinge delta/gate diagnostics when --export_direct_hinge_series was enabled

Typical usage (5 clips in each directory):
  python tools/compare_hinge_apply_noapply.py \\
    --noapply debug_output/_verify_hinge_*_noapply_logc \\
    --apply   debug_output/_verify_hinge_*_apply_logc \\
    --bones calf_r \\
    --branch direct \\
    --min-cycle 1 \\
    --phase-min 49 --phase-max 86 \\
    --contact-source gt --contact-index 1 --contact-value 0 \\
    --contact-thresh 0.5 1e-4
"""

from __future__ import annotations

import argparse
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
        # Fallback: treat as glob.
        for m in sorted(glob.glob(ss)):
            mp = Path(m).expanduser()
            if mp.is_file():
                out.append(mp)
    # Dedupe while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in out:
        if str(p) in seen:
            continue
        seen.add(str(p))
        uniq.append(p)
    return uniq


def _mean(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(float(x) for x in xs) / float(len(xs)))


def _pstdev(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    m = _mean(xs)
    if m is None:
        return None
    var = sum((float(x) - float(m)) ** 2 for x in xs) / float(len(xs))
    return float(max(0.0, var) ** 0.5)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _step_in_cycle(rec: Optional[Dict[str, Any]], *, step: int, cycle_len: int) -> Optional[int]:
    """
    Return step_in_cycle for a given global step.

    Prefers exported metrics_per_step[*].step_in_cycle if available; falls back to
    step % cycle_len when cycle_len is known.
    """
    if rec is not None:
        sic_raw = rec.get("step_in_cycle", None)
        if isinstance(sic_raw, int):
            return int(sic_raw)
        if sic_raw is not None:
            try:
                return int(sic_raw)
            except Exception:
                pass
    if int(cycle_len) > 0:
        return int(int(step) % int(cycle_len))
    return None


@dataclass
class TailStats:
    th_deg: float
    n: int = 0
    sum_ang: float = 0.0
    n_tail: int = 0
    omega_tail: List[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.omega_tail is None:
            self.omega_tail = []

    def add(self, *, ang_deg: float, omega_deg: float) -> None:
        self.n += 1
        self.sum_ang += float(ang_deg)
        if float(ang_deg) > float(self.th_deg):
            self.n_tail += 1
            self.omega_tail.append(float(omega_deg))

    def merge_(self, other: "TailStats") -> None:
        if int(other.n) <= 0:
            return
        self.n += int(other.n)
        self.sum_ang += float(other.sum_ang)
        self.n_tail += int(other.n_tail)
        self.omega_tail += list(other.omega_tail)

    def summary(self) -> Dict[str, Optional[float]]:
        mean_ang = (self.sum_ang / float(self.n)) if self.n > 0 else None
        p_tail = (float(self.n_tail) / float(self.n)) if self.n > 0 else None
        omega_m = _mean(self.omega_tail)
        omega_s = _pstdev(self.omega_tail)
        return {
            "th_deg": float(self.th_deg),
            "n": float(self.n),
            "mean_ang_deg": mean_ang,
            "p_ang_gt_th": p_tail,
            "n_tail": float(self.n_tail),
            "omega_tail_mean_deg": omega_m,
            "omega_tail_std_deg": omega_s,
        }


@dataclass
class HingeSeriesAgg:
    n_swing: int = 0
    n_stance: int = 0
    delta_swing_deg: List[float] = None  # type: ignore[assignment]
    delta_stance_deg: List[float] = None  # type: ignore[assignment]
    abs_delta_deg: List[float] = None  # type: ignore[assignment]
    abs_delta_raw_deg: List[float] = None  # type: ignore[assignment]
    gate_swing: List[float] = None  # type: ignore[assignment]
    gate_stance: List[float] = None  # type: ignore[assignment]
    corr_gate_pairs: List[Tuple[float, float]] = None  # type: ignore[assignment]
    corr_gate_pairs_other: List[Tuple[float, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.delta_swing_deg = [] if self.delta_swing_deg is None else self.delta_swing_deg
        self.delta_stance_deg = [] if self.delta_stance_deg is None else self.delta_stance_deg
        self.abs_delta_deg = [] if self.abs_delta_deg is None else self.abs_delta_deg
        self.abs_delta_raw_deg = [] if self.abs_delta_raw_deg is None else self.abs_delta_raw_deg
        self.gate_swing = [] if self.gate_swing is None else self.gate_swing
        self.gate_stance = [] if self.gate_stance is None else self.gate_stance
        self.corr_gate_pairs = [] if self.corr_gate_pairs is None else self.corr_gate_pairs
        self.corr_gate_pairs_other = [] if self.corr_gate_pairs_other is None else self.corr_gate_pairs_other

    def merge_(self, other: "HingeSeriesAgg") -> None:
        self.n_swing += int(other.n_swing)
        self.n_stance += int(other.n_stance)
        self.delta_swing_deg += list(other.delta_swing_deg)
        self.delta_stance_deg += list(other.delta_stance_deg)
        self.abs_delta_deg += list(other.abs_delta_deg)
        self.abs_delta_raw_deg += list(other.abs_delta_raw_deg)
        self.gate_swing += list(other.gate_swing)
        self.gate_stance += list(other.gate_stance)
        self.corr_gate_pairs += list(other.corr_gate_pairs)
        self.corr_gate_pairs_other += list(other.corr_gate_pairs_other)


def _pearson_corr(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
    xs = [float(a) for a, _ in pairs]
    ys = [float(b) for _, b in pairs]
    if len(xs) < 2:
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


def _resolve_contact_key(src: str) -> str:
    s = str(src).strip().lower()
    return {"gt": "ContactGTPerC", "plan": "ContactPlanPerC", "meas": "ContactMeasPerC"}.get(s, "ContactGTPerC")


def _extract_clip_series(
    obj: Dict[str, Any], *, branch: str, bone: str
) -> Tuple[List[float], List[float], List[Dict[str, Any]], int]:
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
    if not isinstance(steps, list) or not steps:
        raise ValueError("missing metrics_per_step (needed for cycle/contact filters)")

    cycle_len = _as_int(obj.get("cycle_len", 0) or 0, 0)
    return omega, ang, steps, int(cycle_len)


def _iter_selected_indices(
    *,
    steps: List[Dict[str, Any]],
    cycle_len: int,
    min_cycle: int,
    phase_min: Optional[int],
    phase_max: Optional[int],
    contact_key: str,
    contact_idx: Optional[int],
    contact_value: Optional[int],
    contact_thresh: float,
    max_len: int,
) -> Iterable[int]:
    for i in range(int(max_len)):
        # cycle filter
        cy = None
        rec = steps[i] if i < len(steps) and isinstance(steps[i], dict) else None
        if rec is not None:
            cy_raw = rec.get("cycle", None)
            if isinstance(cy_raw, int):
                cy = int(cy_raw)
        if cy is None and int(cycle_len) > 0:
            cy = int(i // int(cycle_len))
        if int(cy or 0) < int(min_cycle):
            continue

        # phase/step-in-cycle filter
        if phase_min is not None or phase_max is not None:
            sic = _step_in_cycle(rec, step=i, cycle_len=cycle_len)
            if sic is None:
                continue
            if phase_min is not None and int(sic) < int(phase_min):
                continue
            if phase_max is not None and int(sic) > int(phase_max):
                continue

        # contact filter
        if contact_value is not None:
            if rec is None:
                continue
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


def _extract_direct_hinge_series(
    obj: Dict[str, Any],
    *,
    bone: str,
) -> Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]:
    """
    Extract per-step hinge series (mean-over-batch) for a single bone.

    Returns:
      (delta_deg, delta_raw_deg, gate, valid, valid_raw, valid_gate)
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


def _agg_hinge_series(
    paths: Sequence[Path],
    *,
    bones: Sequence[str],
    min_cycle: int,
    phase_min: Optional[int],
    phase_max: Optional[int],
    contact_source: str,
    contact_idx: Optional[int],
    contact_thresh: float,
) -> Dict[str, HingeSeriesAgg]:
    """
    Aggregate hinge series diagnostics over multiple clips (APPLY runs only).
    """
    contact_key = _resolve_contact_key(contact_source)
    out: Dict[str, HingeSeriesAgg] = {b: HingeSeriesAgg() for b in bones}

    other_idx: Optional[int] = None
    if contact_idx in (0, 1):
        other_idx = 1 - int(contact_idx)

    for p in paths:
        obj = _load_json(p)
        steps = obj.get("metrics_per_step", None)
        if not isinstance(steps, list) or not steps:
            continue
        cycle_len = _as_int(obj.get("cycle_len", 0) or 0, 0)

        for bone in bones:
            try:
                delta_deg, delta_raw_deg, gate, valid, valid_raw, valid_gate = _extract_direct_hinge_series(obj, bone=bone)
            except Exception:
                continue

            T = min(len(delta_deg), len(delta_raw_deg), len(gate), len(valid), len(valid_raw), len(valid_gate), len(steps))
            if T <= 0:
                continue

            agg = HingeSeriesAgg()
            for i in range(int(T)):
                rec = steps[i] if isinstance(steps[i], dict) else None
                if rec is None:
                    continue

                cy = None
                cy_raw = rec.get("cycle", None)
                if isinstance(cy_raw, int):
                    cy = int(cy_raw)
                if cy is None and int(cycle_len) > 0:
                    cy = int(i // int(cycle_len))
                if int(cy or 0) < int(min_cycle):
                    continue

                if phase_min is not None or phase_max is not None:
                    sic = _step_in_cycle(rec, step=i, cycle_len=cycle_len)
                    if sic is None:
                        continue
                    if phase_min is not None and int(sic) < int(phase_min):
                        continue
                    if phase_max is not None and int(sic) > int(phase_max):
                        continue

                c = rec.get(contact_key, None)
                if not isinstance(c, list) or contact_idx is None or contact_idx < 0 or contact_idx >= len(c):
                    continue
                try:
                    cr = float(c[int(contact_idx)])
                except Exception:
                    continue
                is_swing = bool(cr < float(contact_thresh))

                # delta (effective)
                if int(valid[i]) == 1:
                    dv = float(delta_deg[i])
                    if math.isfinite(dv):
                        if is_swing:
                            agg.n_swing += 1
                            agg.delta_swing_deg.append(dv)
                        else:
                            agg.n_stance += 1
                            agg.delta_stance_deg.append(dv)
                        agg.abs_delta_deg.append(abs(dv))

                # delta_raw
                if int(valid_raw[i]) == 1:
                    dr = float(delta_raw_deg[i])
                    if math.isfinite(dr):
                        agg.abs_delta_raw_deg.append(abs(dr))

                # gate
                if int(valid_gate[i]) == 1:
                    gv = float(gate[i])
                    if math.isfinite(gv):
                        if is_swing:
                            agg.gate_swing.append(gv)
                        else:
                            agg.gate_stance.append(gv)
                        agg.corr_gate_pairs.append((gv, float(1.0 - cr)))
                        if other_idx is not None and other_idx < len(c):
                            try:
                                co = float(c[int(other_idx)])
                            except Exception:
                                co = None
                            if co is not None and math.isfinite(float(co)):
                                agg.corr_gate_pairs_other.append((gv, float(1.0 - float(co))))

            out[str(bone)].merge_(agg)

    return out


def _agg_run(
    paths: Sequence[Path],
    *,
    bones: Sequence[str],
    branch: str,
    min_cycle: int,
    phase_min: Optional[int],
    phase_max: Optional[int],
    contact_source: str,
    contact_idx: Optional[int],
    contact_value: Optional[int],
    contact_thresh: float,
    angle_thresh_deg: Optional[float],
) -> Tuple[Dict[str, TailStats], Dict[str, Dict[str, List[int]]], float, List[str]]:
    """
    Returns:
      - stats_by_bone: TailStats for each bone (aggregated across clips)
      - tail_idx_by_clip_bone: per-clip NOAPPLY tail indices (subset of selected indices) for fixed-tail eval
      - th_deg: the effective angle threshold (deg)
      - used_clips: clip names included
    """
    if not paths:
        raise ValueError("empty run inputs")

    contact_key = _resolve_contact_key(contact_source)

    # Determine the default tail threshold from the first JSON, unless overridden.
    first = _load_json(paths[0])
    ko = first.get("keybone_omega", None)
    if not isinstance(ko, dict):
        raise ValueError(f"{paths[0]}: missing keybone_omega")
    th_default = float(ko.get("deg_thresh") or 0.0)
    th_deg = float(angle_thresh_deg) if angle_thresh_deg is not None else float(th_default)

    stats: Dict[str, TailStats] = {b: TailStats(th_deg=float(th_deg)) for b in bones}
    tail_idx_by_clip_bone: Dict[str, Dict[str, List[int]]] = {}
    used_clips: List[str] = []

    for p in paths:
        obj = _load_json(p)
        clip = str(obj.get("clip") or _clip_from_path(p))
        # NOTE: we store indices per-clip so fixed-tail can align apply/noapply by clip name.
        tail_idx_by_clip_bone.setdefault(clip, {})

        for bone in bones:
            omega, ang, steps, cycle_len = _extract_clip_series(obj, branch=branch, bone=bone)
            max_len = min(len(omega), len(ang), len(steps))

            sel = list(
                _iter_selected_indices(
                    steps=steps,
                    cycle_len=cycle_len,
                    min_cycle=min_cycle,
                    phase_min=phase_min,
                    phase_max=phase_max,
                    contact_key=contact_key,
                    contact_idx=contact_idx,
                    contact_value=contact_value,
                    contact_thresh=contact_thresh,
                    max_len=max_len,
                )
            )
            if not sel:
                tail_idx_by_clip_bone[clip][bone] = []
                continue

            # Per-clip bone stats
            clip_stats = TailStats(th_deg=float(th_deg))
            tail_idx: List[int] = []
            for i in sel:
                try:
                    a = float(ang[i])
                    o = float(omega[i])
                except Exception:
                    continue
                clip_stats.add(ang_deg=a, omega_deg=o)
                if a > float(th_deg):
                    tail_idx.append(int(i))

            stats[bone].merge_(clip_stats)
            tail_idx_by_clip_bone[clip][bone] = tail_idx

        used_clips.append(clip)

    return stats, tail_idx_by_clip_bone, float(th_deg), used_clips


def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "NA"
    try:
        if not math.isfinite(float(x)):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare hinge APPLY vs NOAPPLY with fixed-tail omega stats.")
    ap.add_argument("--noapply", nargs="+", required=True, help="NOAPPLY input(s): dir/file/glob (space-separated).")
    ap.add_argument("--apply", nargs="+", required=True, help="APPLY input(s): dir/file/glob (space-separated).")
    ap.add_argument("--bones", nargs="+", default=["calf_r"], help="Bone names to analyze (must be exported).")
    ap.add_argument("--branch", type=str, default="direct", choices=("inc", "direct", "blend"))
    ap.add_argument("--min-cycle", type=int, default=1, help="Include only steps with cycle >= this value.")
    ap.add_argument(
        "--phase-min",
        type=int,
        default=None,
        help="Include only steps with step_in_cycle >= this value (inclusive).",
    )
    ap.add_argument(
        "--phase-max",
        type=int,
        default=None,
        help="Include only steps with step_in_cycle <= this value (inclusive).",
    )
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
    ap.add_argument(
        "--angle-thresh",
        type=float,
        default=None,
        help="Tail threshold (deg) for ang_deg > th; default: keybone_omega.deg_thresh",
    )
    ap.add_argument(
        "--report-hinge-series",
        action="store_true",
        help="Also report APPLY hinge delta/gate diagnostics from direct_hinge_series (requires --export_direct_hinge_series).",
    )
    ap.add_argument(
        "--strict-clips",
        action="store_true",
        help="Error if NOAPPLY/APPLY do not have identical clip sets (default: use intersection).",
    )
    args = ap.parse_args()

    if args.phase_min is not None and args.phase_max is not None and int(args.phase_min) > int(args.phase_max):
        raise SystemExit("--phase-min must be <= --phase-max.")

    noapply_paths = _glob_inputs(args.noapply)
    apply_paths = _glob_inputs(args.apply)
    if not noapply_paths:
        raise SystemExit("NOAPPLY inputs resolved to no files.")
    if not apply_paths:
        raise SystemExit("APPLY inputs resolved to no files.")

    # Map by clip for fixed-tail alignment.
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

    # Resolve contact index.
    contact_idx: Optional[int] = None
    if args.contact_index is not None:
        contact_idx = int(args.contact_index)
    elif args.contact_side is not None:
        contact_idx = 0 if str(args.contact_side).strip().lower() == "l" else 1
    elif args.contact_value is not None:
        # Match existing convention: default to right foot when filtering without specifying channel.
        contact_idx = 1

    contact_value = args.contact_value
    if contact_value is not None and int(contact_value) not in (0, 1):
        raise SystemExit("--contact-value must be 0 or 1.")
    contact_value = int(contact_value) if contact_value is not None else None

    bones = [str(b) for b in (args.bones or []) if str(b).strip()]
    if not bones:
        raise SystemExit("Empty --bones.")

    print("[CompareHingeApplyNoapply]")
    print(f"- branch={args.branch} bones={','.join(bones)} min_cycle={int(args.min_cycle)}")
    if args.phase_min is not None or args.phase_max is not None:
        print(f"- step_in_cycle_range=[{args.phase_min},{args.phase_max}] (inclusive)")
    print(
        f"- contact={_resolve_contact_key(args.contact_source)}"
        + (f"[{contact_idx}]" if contact_idx is not None else "")
        + (f" -> {contact_value}" if contact_value is not None else " (no filter)")
    )
    print(f"- clips_common={len(common)} ({', '.join(common)})")
    print()

    for ct in args.contact_thresh:
        ct = float(ct)
        print(f"[ContactThresh] {ct:g}")

        no_stats, no_tail_idx, th_deg, _ = _agg_run(
            [noapply_by_clip[c] for c in common],
            bones=bones,
            branch=str(args.branch),
            min_cycle=int(args.min_cycle),
            phase_min=args.phase_min,
            phase_max=args.phase_max,
            contact_source=str(args.contact_source),
            contact_idx=contact_idx,
            contact_value=contact_value,
            contact_thresh=ct,
            angle_thresh_deg=float(args.angle_thresh) if args.angle_thresh is not None else None,
        )
        ap_stats, _, th_deg2, _ = _agg_run(
            [apply_by_clip[c] for c in common],
            bones=bones,
            branch=str(args.branch),
            min_cycle=int(args.min_cycle),
            phase_min=args.phase_min,
            phase_max=args.phase_max,
            contact_source=str(args.contact_source),
            contact_idx=contact_idx,
            contact_value=contact_value,
            contact_thresh=ct,
            angle_thresh_deg=float(args.angle_thresh) if args.angle_thresh is not None else None,
        )
        # Use NOAPPLY threshold as the "source of truth" for fixed-tail.
        th_use = float(th_deg)
        if abs(float(th_deg2) - float(th_use)) > 1e-6:
            print(f"[Warn] angle_thresh mismatch: noapply={th_deg} apply={th_deg2}; using noapply={th_use}")

        headers = [
            "Bone",
            "Run",
            "n",
            f"mean_ang",
            f"P(ang>{th_use:g})",
            "n_tail",
            "omega_tail_mean",
            "omega_tail_std",
        ]
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")

        for bone in bones:
            # NOAPPLY row
            s = no_stats.get(bone)
            ss = s.summary() if s is not None else {}
            print(
                "| "
                + " | ".join(
                    [
                        bone,
                        "NOAPPLY",
                        _fmt(ss.get("n"), 0),
                        _fmt(ss.get("mean_ang_deg"), 3),
                        _fmt(ss.get("p_ang_gt_th"), 4),
                        _fmt(ss.get("n_tail"), 0),
                        _fmt(ss.get("omega_tail_mean_deg"), 3),
                        _fmt(ss.get("omega_tail_std_deg"), 3),
                    ]
                )
                + " |"
            )

            # APPLY row
            s = ap_stats.get(bone)
            ss = s.summary() if s is not None else {}
            print(
                "| "
                + " | ".join(
                    [
                        bone,
                        "APPLY",
                        _fmt(ss.get("n"), 0),
                        _fmt(ss.get("mean_ang_deg"), 3),
                        _fmt(ss.get("p_ang_gt_th"), 4),
                        _fmt(ss.get("n_tail"), 0),
                        _fmt(ss.get("omega_tail_mean_deg"), 3),
                        _fmt(ss.get("omega_tail_std_deg"), 3),
                    ]
                )
                + " |"
            )

            # APPLY fixed-tail row: use NOAPPLY tail indices.
            omega_fixed: List[float] = []
            ang_fixed: List[float] = []
            n_fixed = 0
            for clip in common:
                tail_idx = no_tail_idx.get(clip, {}).get(bone, [])
                if not tail_idx:
                    continue
                ap_obj = _load_json(apply_by_clip[clip])
                omega_ap, ang_ap, steps_ap, _ = _extract_clip_series(ap_obj, branch=str(args.branch), bone=bone)
                T = min(len(omega_ap), len(ang_ap), len(steps_ap))
                for i in tail_idx:
                    if int(i) < 0 or int(i) >= int(T):
                        continue
                    try:
                        omega_fixed.append(float(omega_ap[int(i)]))
                        ang_fixed.append(float(ang_ap[int(i)]))
                        n_fixed += 1
                    except Exception:
                        continue
            mf = _mean(omega_fixed)
            sf = _pstdev(omega_fixed)
            ma = _mean(ang_fixed)
            n_tail_fixed = sum(1 for a in ang_fixed if math.isfinite(float(a)) and float(a) > float(th_use))
            pf = (float(n_tail_fixed) / float(n_fixed)) if n_fixed > 0 else None
            print(
                "| "
                + " | ".join(
                    [
                        bone,
                        "APPLY@fixed_tail(NOAPPLY)",
                        _fmt(float(n_fixed), 0),
                        _fmt(ma, 3),
                        _fmt(pf, 4),
                        _fmt(float(n_tail_fixed), 0),
                        _fmt(mf, 3),
                        _fmt(sf, 3),
                    ]
                )
                + " |"
            )

        if bool(args.report_hinge_series):
            # Hinge delta/gate diagnostics are APPLY-only.
            hinge_agg = _agg_hinge_series(
                [apply_by_clip[c] for c in common],
                bones=bones,
                min_cycle=int(args.min_cycle),
                phase_min=args.phase_min,
                phase_max=args.phase_max,
                contact_source=str(args.contact_source),
                contact_idx=contact_idx,
                contact_thresh=ct,
            )
            hs_headers = [
                "Bone",
                "delta_swing_mean",
                "delta_swing_std",
                "delta_stance_mean",
                "delta_stance_std",
                "abs(delta)_mean",
                "abs(delta_raw)_mean",
                "|eff|/|raw|",
                "gate_swing_mean",
                "gate_stance_mean",
                "corr(gate,1-contact_idx)",
                "corr(gate,1-contact_other)",
            ]
            print(f"[HingeSeries] (APPLY only) contact_thresh={ct:g}")
            print("| " + " | ".join(hs_headers) + " |")
            print("|" + "|".join(["---"] + ["---:"] * (len(hs_headers) - 1)) + "|")

            for bone in bones:
                agg = hinge_agg.get(str(bone), None)
                if agg is None:
                    print("| " + " | ".join([str(bone)] + ["NA"] * (len(hs_headers) - 1)) + " |")
                    continue

                dsw_m = _mean(agg.delta_swing_deg)
                dsw_s = _pstdev(agg.delta_swing_deg)
                dst_m = _mean(agg.delta_stance_deg)
                dst_s = _pstdev(agg.delta_stance_deg)
                abs_eff = _mean(agg.abs_delta_deg)
                abs_raw = _mean(agg.abs_delta_raw_deg)
                ratio = None
                if abs_eff is not None and abs_raw is not None and float(abs_raw) > 1e-9:
                    ratio = float(abs_eff) / float(abs_raw)

                gsw_m = _mean(agg.gate_swing) if agg.gate_swing else None
                gst_m = _mean(agg.gate_stance) if agg.gate_stance else None
                corr_main = _pearson_corr(agg.corr_gate_pairs) if agg.corr_gate_pairs else None
                corr_other = _pearson_corr(agg.corr_gate_pairs_other) if agg.corr_gate_pairs_other else None

                print(
                    "| "
                    + " | ".join(
                        [
                            str(bone),
                            _fmt(dsw_m, 3),
                            _fmt(dsw_s, 3),
                            _fmt(dst_m, 3),
                            _fmt(dst_s, 3),
                            _fmt(abs_eff, 3),
                            _fmt(abs_raw, 3),
                            _fmt(ratio, 3),
                            _fmt(gsw_m, 3),
                            _fmt(gst_m, 3),
                            _fmt(corr_main, 3),
                            _fmt(corr_other, 3),
                        ]
                    )
                    + " |"
                )

            print()

        print()


if __name__ == "__main__":
    main()
