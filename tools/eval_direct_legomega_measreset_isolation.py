#!/usr/bin/env python3
"""
Stage7 direct leg-omega (SO(3)) flip isolation for freerun_cycles.

Goal: split flip(59/288) into:
  (A) closed-loop contacts_meas + phase reset wiring, vs.
  (B) phase reset event extraction itself (reset source).

This script runs a small, fixed ablation matrix (free-run; no --freerun_x_gt):
  A0: contacts_meas_source=model   phase_reset_source=contacts_meas
  A1: contacts_meas_source=gt      phase_reset_source=contacts_meas
  A2: contacts_meas_source=pretrain_contact phase_reset_source=contacts_meas
  A3: contacts_meas_source=zero    phase_reset_source=contacts_meas
  B1: contacts_meas_source=model   phase_reset_source=ttc_gt
  B3: contacts_meas_source=gt      phase_reset_source=ttc_gt

Notes:
  - B0 == A0, B2 == A1 (aliases; not re-run by default).
  - Evaluation window is enforced via direct_leg_omega_alpha_sweep args and summary filters:
      cycle>=1, drop_wrap, sics=14,15,49-55, bones=thigh/calf/foot/ball (L/R) => 288 points.
  - Always enables:
      --export_direct_leg_omega_alpha_sweep --export_plan_state_series --log_contacts
  - After each run:
      (1) summarize flip/overshoot (same definition as tools/summarize_direct_legomega_alpha_sweep.py),
      (2) export per-step contact(plan/meas/err)+plan/phase state CSV for alignment.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _parse_int_ranges(spec: str) -> List[int]:
    out: Set[int] = set()
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            lo = int(a.strip())
            hi = int(b.strip())
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(tok))
    return sorted(out)


def _parse_bones(spec: str) -> Set[str]:
    return {b.strip() for b in str(spec or "").split(",") if b.strip()}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict JSON at {path}, got {type(obj)}")
    return obj


def _isfinite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _get_float(d: Dict[str, Any], key: str) -> float:
    try:
        v = d.get(key, float("nan"))
        if v is None:
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _find_freerun_json(out_dir: Path) -> Path:
    cands = sorted(out_dir.glob("*_freerun_cycles.json"))
    if not cands:
        raise FileNotFoundError(f"No *_freerun_cycles.json found in {out_dir}")
    if len(cands) > 1:
        for p in cands:
            if p.name.startswith("Walk_F_"):
                return p
    return cands[0]


def summarize_alpha_sweep(
    obj: Dict[str, Any],
    *,
    cycle_gte: int,
    drop_wrap: bool,
    sics: Set[int],
    bones: Set[str],
) -> Dict[str, Any]:
    sw = obj.get("direct_leg_omega_alpha_sweep")
    if not isinstance(sw, dict):
        raise KeyError("Missing direct_leg_omega_alpha_sweep (re-run with --export_direct_leg_omega_alpha_sweep).")
    steps = sw.get("steps")
    if not isinstance(steps, list):
        raise TypeError("direct_leg_omega_alpha_sweep.steps is not a list")

    total = 0
    flip = 0
    overshoot = 0
    flip_by_sic: Dict[int, int] = {}
    overs_by_sic: Dict[int, int] = {}
    flip_by_bone: Dict[str, int] = {}
    overs_by_bone: Dict[str, int] = {}

    for st in steps:
        if not isinstance(st, dict):
            continue
        cyc = int(st.get("cycle", 0) or 0)
        if cyc < int(cycle_gte):
            continue
        if drop_wrap and bool(st.get("wrap_boundary_step", False)):
            continue
        sic = int(st.get("step_in_cycle", -1) or -1)
        if sic not in sics:
            continue
        pb = st.get("per_bone")
        if not isinstance(pb, dict):
            continue

        for bone, dat in pb.items():
            bone_s = str(bone)
            if bone_s not in bones:
                continue
            if not isinstance(dat, dict):
                continue
            cos = _get_float(dat, "cos_pred_oracle")
            best = _get_float(dat, "best_alpha")
            ratio = _get_float(dat, "norm_ratio_pred_over_oracle")
            total += 1

            if _isfinite(cos) and _isfinite(best) and cos < 0.0 and best < 0.0:
                flip += 1
                flip_by_sic[sic] = int(flip_by_sic.get(sic, 0)) + 1
                flip_by_bone[bone_s] = int(flip_by_bone.get(bone_s, 0)) + 1
            if _isfinite(cos) and _isfinite(best) and _isfinite(ratio) and cos > 0.0 and ratio > 1.0 and (0.0 < best < 1.0):
                overshoot += 1
                overs_by_sic[sic] = int(overs_by_sic.get(sic, 0)) + 1
                overs_by_bone[bone_s] = int(overs_by_bone.get(bone_s, 0)) + 1

    return {
        "total": int(total),
        "flip": int(flip),
        "overshoot": int(overshoot),
        "flip_by_sic": dict(sorted(flip_by_sic.items())),
        "overshoot_by_sic": dict(sorted(overs_by_sic.items())),
        "flip_by_bone": dict(sorted(flip_by_bone.items(), key=lambda kv: (-kv[1], kv[0]))),
        "overshoot_by_bone": dict(sorted(overs_by_bone.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


@dataclass(frozen=True)
class Case:
    label: str
    contacts_meas_source: str
    phase_reset_source: str
    alias_of: Optional[str] = None  # label of canonical case (not re-run)

    @property
    def name(self) -> str:
        # Keep it readable and filesystem-safe.
        return f"{self.label}__meas_{self.contacts_meas_source}__reset_{self.phase_reset_source}"


def _run_freerun_cycles(
    *,
    teacher: Path,
    model: Path,
    bundle: Path,
    out_dir: Path,
    rounds: int,
    time_index_mode: str,
    sics_expanded: Sequence[int],
    bones_csv: str,
    contacts_meas_source: str,
    phase_reset_source: str,
    force: bool,
    dry_run: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        "-m",
        "train.validate.run_freerun_cycles",
        "--teacher",
        str(teacher),
        "--model",
        str(model),
        "--bundle",
        str(bundle),
        "--rounds",
        str(int(rounds)),
        "--time-index-mode",
        str(time_index_mode),
        "--out",
        str(out_dir),
        "--export_direct_leg_omega_alpha_sweep",
        "--export_plan_state_series",
        "--log_contacts",
        "--direct_leg_omega_alpha_sweep_sics",
        ",".join(str(x) for x in sics_expanded),
        "--direct_leg_omega_alpha_sweep_bones",
        str(bones_csv),
        "--contacts_meas_source",
        str(contacts_meas_source),
        "--phase_reset_source",
        str(phase_reset_source),
    ]
    if force:
        cmd.append("--force")

    print(f"[run] {out_dir.name}: contacts_meas_source={contacts_meas_source} phase_reset_source={phase_reset_source}")
    if dry_run:
        print(" ".join(cmd))
        return _find_freerun_json(out_dir) if any(out_dir.glob("*_freerun_cycles.json")) else out_dir / "Walk_F_freerun_cycles.json"

    subprocess.check_call(cmd)
    return _find_freerun_json(out_dir)


def _export_series(*, out_json: Path, out_csv: Path, cycle_gte: int, drop_wrap: bool) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "tools/export_contact_planmeas_state_series.py",
        "--json",
        str(out_json),
        "--out",
        str(out_csv),
        "--cycle-gte",
        str(int(cycle_gte)),
        "--only-alpha-sweep-steps",
    ]
    if drop_wrap:
        cmd.append("--drop-wrap")
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run & summarize Stage7 legomega flip isolation (meas/reset ablations).")
    ap.add_argument(
        "--teacher",
        type=str,
        default="validate/teacher_batches/Walk_F_teacher.json",
        help="Teacher json path (default: validate/teacher_batches/Walk_F_teacher.json).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage7_legomega_routedshared_warm_20260126.pth",
        help="Checkpoint path (default: the Stage7 legomega routed/shared warm ckpt).",
    )
    ap.add_argument(
        "--bundle",
        type=str,
        default="raw_data/processed_data/norm_template.json",
        help="Bundle json path (default: raw_data/processed_data/norm_template.json).",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default="debug_output/_diag_legomega_measreset_isolation_20260130",
        help="Output root dir (one subdir per case).",
    )
    ap.add_argument("--rounds", type=int, default=5, help="freerun_cycles rounds (default: 5).")
    ap.add_argument("--time-index-mode", type=str, default="auto", choices=("auto", "global", "cycle", "none"))

    ap.add_argument("--cycle-gte", type=int, default=1, help="Summary/export filter: cycle>=K (default: 1).")
    ap.add_argument("--drop-wrap", action="store_true", help="Summary/export filter: drop wrap-boundary steps.")
    ap.add_argument("--sics", type=str, default="14,15,49-55", help="Summary filter: step_in_cycle set (ranges ok).")
    ap.add_argument(
        "--bones",
        type=str,
        default="thigh_r,calf_r,foot_r,ball_r,thigh_l,calf_l,foot_l,ball_l",
        help="Summary filter: bones CSV (must match direct_leg_omega_alpha_sweep keys).",
    )

    ap.add_argument(
        "--cases",
        type=str,
        default="A0,A1,A2,A3,B0,B1,B2,B3",
        help="CSV subset of {A0,A1,A2,A3,B0,B1,B2,B3} (default: all).",
    )
    ap.add_argument("--force", action="store_true", help="Pass --force to freerun_cycles (overwrite outputs).")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    ap.add_argument("--no-export-series", action="store_true", help="Skip exporting the per-step series CSV.")
    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    teacher = Path(args.teacher).expanduser()
    model = Path(args.model).expanduser()
    bundle = Path(args.bundle).expanduser()

    sics_expanded = _parse_int_ranges(args.sics)
    sics_set = set(sics_expanded)
    bones_set = _parse_bones(args.bones)

    # Canonical run configs (we avoid re-running B0/B2 since they alias A0/A1).
    canonical: Dict[str, Case] = {
        "A0": Case("A0", "model", "contacts_meas"),
        "A1": Case("A1", "gt", "contacts_meas"),
        "A2": Case("A2", "pretrain_contact", "contacts_meas"),
        "A3": Case("A3", "zero", "contacts_meas"),
        "B1": Case("B1", "model", "ttc_gt"),
        "B3": Case("B3", "gt", "ttc_gt"),
    }
    aliases: Dict[str, Case] = {
        "B0": Case("B0", "model", "contacts_meas", alias_of="A0"),
        "B2": Case("B2", "gt", "contacts_meas", alias_of="A1"),
    }
    all_cases: Dict[str, Case] = {**canonical, **aliases}

    want = [t.strip() for t in str(args.cases or "").split(",") if t.strip()]
    for lab in want:
        if lab not in all_cases:
            raise SystemExit(f"Unknown case '{lab}'. Allowed: {sorted(all_cases.keys())}")

    # Run canonical cases that are requested, then summarize all requested cases (including aliases).
    rows: List[Dict[str, Any]] = []
    out_json_by_label: Dict[str, Path] = {}

    for lab, cs in canonical.items():
        if lab not in want:
            continue
        out_dir = out_root / cs.name
        out_json = None
        if not bool(args.force) and any(out_dir.glob("*_freerun_cycles.json")):
            out_json = _find_freerun_json(out_dir)
            print(f"[reuse] {lab} -> {out_json}")
        else:
            out_json = _run_freerun_cycles(
                teacher=teacher,
                model=model,
                bundle=bundle,
                out_dir=out_dir,
                rounds=int(args.rounds),
                time_index_mode=str(args.time_index_mode),
                sics_expanded=sics_expanded,
                bones_csv=str(args.bones),
                contacts_meas_source=cs.contacts_meas_source,
                phase_reset_source=cs.phase_reset_source,
                force=bool(args.force),
                dry_run=bool(args.dry_run),
            )

        out_json_by_label[lab] = out_json

        if not bool(args.no_export_series) and not bool(args.dry_run):
            series_csv = out_dir / "contacts_planmeas_planstate_series.csv"
            _export_series(out_json=out_json, out_csv=series_csv, cycle_gte=int(args.cycle_gte), drop_wrap=bool(args.drop_wrap))

        if bool(args.dry_run):
            continue

        obj = _load_json(out_json)
        st = summarize_alpha_sweep(
            obj,
            cycle_gte=int(args.cycle_gte),
            drop_wrap=bool(args.drop_wrap),
            sics=sics_set,
            bones=bones_set,
        )
        rows.append(
            {
                "case": lab,
                "contacts_meas_source": cs.contacts_meas_source,
                "phase_reset_source": cs.phase_reset_source,
                "out_json": str(out_json),
                **st,
            }
        )
        print(f"[ok] {lab}: flip={st['flip']}/{st['total']} overshoot={st['overshoot']}/{st['total']}")

    # Add alias rows (no extra runs).
    for lab, cs in aliases.items():
        if lab not in want:
            continue
        if cs.alias_of is None or cs.alias_of not in out_json_by_label:
            # Alias was requested but canonical wasn't; force a run by treating it as canonical.
            raise SystemExit(f"Requested alias {lab} but missing canonical {cs.alias_of}; include it in --cases.")
        out_json = out_json_by_label[cs.alias_of]
        if bool(args.dry_run):
            continue
        obj = _load_json(out_json)
        st = summarize_alpha_sweep(
            obj,
            cycle_gte=int(args.cycle_gte),
            drop_wrap=bool(args.drop_wrap),
            sics=sics_set,
            bones=bones_set,
        )
        rows.append(
            {
                "case": lab,
                "alias_of": cs.alias_of,
                "contacts_meas_source": cs.contacts_meas_source,
                "phase_reset_source": cs.phase_reset_source,
                "out_json": str(out_json),
                **st,
            }
        )
        print(f"[alias] {lab} == {cs.alias_of}: flip={st['flip']}/{st['total']} overshoot={st['overshoot']}/{st['total']}")

    if bool(args.dry_run):
        return

    # Sort in a stable order (A0..A3,B0..B3).
    order = {k: i for i, k in enumerate(["A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3"])}
    rows.sort(key=lambda r: order.get(str(r.get("case", "")), 999))

    print("\n| case | contacts_meas_source | phase_reset_source | flip/total | overshoot/total | out_json |")
    print("|---|---|---|---:|---:|---|")
    for r in rows:
        total = int(r.get("total", 0))
        flip = int(r.get("flip", 0))
        over = int(r.get("overshoot", 0))
        frac_f = f"{flip}/{total}" if total > 0 else "n/a"
        frac_o = f"{over}/{total}" if total > 0 else "n/a"
        out_json = str(r.get("out_json", ""))
        print(
            f"| {r.get('case')} | {r.get('contacts_meas_source')} | {r.get('phase_reset_source')} | "
            f"{frac_f} | {frac_o} | {out_json} |"
        )

    summary_path = out_root / "summary_legomega_measreset_isolation.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
