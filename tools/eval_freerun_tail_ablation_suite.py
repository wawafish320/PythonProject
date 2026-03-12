#!/usr/bin/env python3
"""
Run a minimal `freerun_cycles` ablation suite to localize tail / seam error sources.

Goal
----
With 4-6 runs, quickly answer which lever likely gives the biggest win next:
  1) lambda fusion / gate behavior,
  2) multi-cycle seam state mismatch,
  3) direct head quality (and its dependence on phase/contact hints).

Default cases (5)
-----------------
1) base:
   - --so3_corr_apply --lambda_fusion_apply
2) no_lambda (incremental-only rollout update):
   - --so3_corr_apply
3) seam_sync_state (debug seam carry mismatch upper bound):
   - base + --multicycle_sync_state_on_cycle_start
4) direct_hint_gt (direct head upper bound given teacher contacts):
   - base + --direct_pose_meas_source gt --direct_pose_plan_source gt
5) direct_hint_zero (direct head w/o contacts hint):
   - base + --direct_pose_meas_source zero --direct_pose_plan_source zero

Notes
-----
- We summarize tail statistics over cycles 1-4 (steady state).
- We also summarize the worst cyclic window over step_in_cycle means (seam indicator).
- The "oracle" summary (only for base) computes per-step min(GeoLocalDeg, DirectGeoLocalDeg)
  on the same rollout to estimate how much room a better gate could have.

Example
-------
python tools/eval_freerun_tail_ablation_suite.py \\
  --teacher validate/teacher_batches/Walk_F_teacher.json \\
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_lambda_cycles2_after_direct_pose.pth \\
  --out-root debug_output/ablation_suite_d1_depth3/Walk_F \\
  --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Case:
    name: str
    args: Tuple[str, ...]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict JSON at {path}, got {type(obj)}")
    return obj


def _find_outputs(out_dir: Path) -> List[Path]:
    return sorted(out_dir.glob("*_freerun_cycles.json"))


def _slug(s: str) -> str:
    out = []
    for ch in str(s).strip():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch in (" ", ".", ":", "/", "\\", "|", "+"):
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "case"


def _quantile(a: np.ndarray, q: float) -> float:
    # numpy quantile is stable/fast; keep float64 for reproducibility.
    return float(np.quantile(a.astype(np.float64), q))


def _stats(a: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(a), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": _quantile(arr, 0.95),
        "p99": _quantile(arr, 0.99),
        "max": float(arr.max()),
    }


def _fmt_deg(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.2f}°"


def _fmt_float(x: float, nd: int = 3) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def _filter_steps(
    steps: List[Dict[str, Any]],
    *,
    cycles: Iterable[int],
) -> List[Dict[str, Any]]:
    cset = {int(c) for c in cycles}
    out: List[Dict[str, Any]] = []
    for rec in steps:
        try:
            if int(rec.get("cycle", -999)) in cset:
                out.append(rec)
        except Exception:
            continue
    return out


def _per_step_values(steps: List[Dict[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for rec in steps:
        v = rec.get(key, None)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    return vals


def _per_step_oracle_min(
    steps: List[Dict[str, Any]],
    *,
    inc_key: str,
    direct_key: str,
) -> List[float]:
    out: List[float] = []
    for rec in steps:
        inc = rec.get(inc_key, None)
        direct = rec.get(direct_key, None)
        if inc is None or direct is None:
            continue
        try:
            out.append(min(float(inc), float(direct)))
        except Exception:
            continue
    return out


def _ti_stats(
    steps: List[Dict[str, Any]],
    *,
    key: str,
    cycle_len: int,
) -> List[Dict[str, Any]]:
    acc: Dict[int, List[float]] = {}
    for rec in steps:
        ti = rec.get("step_in_cycle", None)
        v = rec.get(key, None)
        if ti is None or v is None:
            continue
        try:
            ti_i = int(ti)
            if cycle_len > 0:
                ti_i = ti_i % int(cycle_len)
            acc.setdefault(ti_i, []).append(float(v))
        except Exception:
            continue

    out: List[Dict[str, Any]] = []
    for ti_i, vals in acc.items():
        st = _stats(vals)
        out.append({"ti": int(ti_i), **st})
    out.sort(key=lambda d: float(d.get("mean", float("nan"))), reverse=True)
    return out


def _worst_window_by_ti_mean(
    ti_stats: List[Dict[str, Any]],
    *,
    cycle_len: int,
    window: int,
) -> Optional[Tuple[float, int, int]]:
    if cycle_len <= 0 or window <= 0:
        return None
    mean_by: Dict[int, float] = {int(d["ti"]): float(d["mean"]) for d in ti_stats if np.isfinite(float(d["mean"]))}
    series = [mean_by.get(i, float("nan")) for i in range(int(cycle_len))]
    series2 = series + series

    best: Optional[Tuple[float, int, int]] = None
    for start in range(int(cycle_len)):
        seg = series2[start : start + int(window)]
        if any(np.isnan(x) for x in seg):
            continue
        avg = float(np.mean(seg))
        end = int((start + int(window) - 1) % int(cycle_len))
        if best is None or avg > best[0]:
            best = (avg, int(start), int(end))
    return best


def summarize_freerun_cycles_tail(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    steps = data.get("metrics_per_step")
    if not isinstance(steps, list):
        raise KeyError(f"{path} missing metrics_per_step list")

    cycle_len = int(data.get("cycle_len", 0) or 0)
    if cycle_len <= 0:
        # fallback: infer from max step_in_cycle (best effort)
        try:
            cycle_len = 1 + max(int(s.get("step_in_cycle", 0) or 0) for s in steps if isinstance(s, dict))
        except Exception:
            cycle_len = 0

    steady_steps = _filter_steps(steps, cycles=(1, 2, 3, 4))

    direct_vals = _per_step_values(steady_steps, "DirectGeoLocalDeg")
    blend_vals = _per_step_values(steady_steps, "BlendGeoLocalDeg")
    inc_vals = _per_step_values(steady_steps, "GeoLocalDeg")
    oracle_vals = _per_step_oracle_min(steady_steps, inc_key="GeoLocalDeg", direct_key="DirectGeoLocalDeg")

    direct = _stats(direct_vals)
    blend = _stats(blend_vals)
    inc = _stats(inc_vals)
    oracle = _stats(oracle_vals)

    direct_ti = _ti_stats(steady_steps, key="DirectGeoLocalDeg", cycle_len=cycle_len)
    blend_ti = _ti_stats(steady_steps, key="BlendGeoLocalDeg", cycle_len=cycle_len)

    direct_w5 = _worst_window_by_ti_mean(direct_ti, cycle_len=cycle_len, window=5)
    blend_w5 = _worst_window_by_ti_mean(blend_ti, cycle_len=cycle_len, window=5)

    def _tail_ratio(st: Dict[str, float]) -> float:
        med = float(st.get("median", float("nan")))
        mx = float(st.get("max", float("nan")))
        if not np.isfinite(med) or med <= 0 or not np.isfinite(mx):
            return float("nan")
        return float(mx / med)

    out: Dict[str, Any] = {
        "clip": data.get("clip", path.name.split("_freerun_cycles.json")[0]),
        "out_json": str(path),
        "model": data.get("model"),
        "cycle_len": int(cycle_len),
        "rounds": int(data.get("rounds", 0) or 0),
        "lambda_fusion_apply": bool(data.get("lambda_fusion_apply", False)),
        "so3_corr_apply": bool(data.get("so3_corr_apply", False)),
        "direct_pose_meas_source": data.get("direct_pose_meas_source", "model"),
        "direct_pose_plan_source": data.get("direct_pose_plan_source", "model"),
        "multicycle_sync_state_on_cycle_start": bool(data.get("multicycle_sync_state_on_cycle_start", False)),
        "multicycle_reset_plan_z_on_cycle_start": bool(data.get("multicycle_reset_plan_z_on_cycle_start", False)),
        # steady-state summary
        "steady_direct": direct,
        "steady_blend": blend,
        "steady_inc": inc,
        "steady_oracle_min_inc_direct": oracle,
        "steady_direct_tail_ratio": _tail_ratio(direct),
        "steady_blend_tail_ratio": _tail_ratio(blend),
        "steady_inc_tail_ratio": _tail_ratio(inc),
        "steady_oracle_tail_ratio": _tail_ratio(oracle),
        # worst cyclic window over ti-means (w=5)
        "direct_worst5": {"avg": direct_w5[0], "start": direct_w5[1], "end": direct_w5[2]} if direct_w5 else None,
        "blend_worst5": {"avg": blend_w5[0], "start": blend_w5[1], "end": blend_w5[2]} if blend_w5 else None,
        # top offenders
        "direct_top_mean_ti": int(direct_ti[0]["ti"]) if direct_ti else None,
        "direct_top_mean": float(direct_ti[0]["mean"]) if direct_ti else None,
        "direct_top_max_ti": int(max(direct_ti, key=lambda d: float(d.get("max", float("-inf"))))["ti"]) if direct_ti else None,
        "direct_top_max": float(max(direct_ti, key=lambda d: float(d.get("max", float("-inf"))))["max"]) if direct_ti else None,
        "blend_top_mean_ti": int(blend_ti[0]["ti"]) if blend_ti else None,
        "blend_top_mean": float(blend_ti[0]["mean"]) if blend_ti else None,
        "blend_top_max_ti": int(max(blend_ti, key=lambda d: float(d.get("max", float("-inf"))))["ti"]) if blend_ti else None,
        "blend_top_max": float(max(blend_ti, key=lambda d: float(d.get("max", float("-inf"))))["max"]) if blend_ti else None,
    }
    return out


def _run_freerun_cycles(
    *,
    teacher: Sequence[str],
    model: str,
    out_dir: Path,
    rounds: int,
    force: bool,
    extra_args: Sequence[str],
) -> None:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "train.validate.run_freerun_cycles",
        "--teacher",
        *list(teacher),
        "--model",
        str(model),
        "--out",
        str(out_dir),
        "--rounds",
        str(int(rounds)),
    ]
    if force:
        cmd.append("--force")
    cmd.extend(list(extra_args))
    subprocess.check_call(cmd)


def _cases(include_plan_z_reset: bool) -> List[Case]:
    base = ("--so3_corr_apply", "--lambda_fusion_apply")
    out = [
        Case("base", base),
        Case("no_lambda", ("--so3_corr_apply",)),
        Case("seam_sync_state", base + ("--multicycle-sync-state-on-cycle-start",)),
        Case("direct_hint_gt", base + ("--direct_pose_meas_source", "gt", "--direct_pose_plan_source", "gt")),
        Case("direct_hint_zero", base + ("--direct_pose_meas_source", "zero", "--direct_pose_plan_source", "zero")),
    ]
    if include_plan_z_reset:
        out.append(Case("seam_reset_plan_z", base + ("--multicycle-reset-plan-z-on-cycle-start",)))
    return out


def _print_case_summary_table(rows: List[Dict[str, Any]]) -> None:
    print("| Case | Direct median | Direct max | Direct worst5(avg@ti) | Blend median | Blend max | Blend worst5(avg@ti) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        d = r["steady_direct"]
        b = r["steady_blend"]
        dw5 = r.get("direct_worst5")
        bw5 = r.get("blend_worst5")
        dw5_s = "n/a"
        if isinstance(dw5, dict) and np.isfinite(float(dw5.get("avg", float("nan")))):
            dw5_s = f"{_fmt_deg(float(dw5['avg']))} ({int(dw5['start'])}→{int(dw5['end'])})"
        bw5_s = "n/a"
        if isinstance(bw5, dict) and np.isfinite(float(bw5.get("avg", float("nan")))):
            bw5_s = f"{_fmt_deg(float(bw5['avg']))} ({int(bw5['start'])}→{int(bw5['end'])})"
        print(
            f"| {r['case']} | {_fmt_deg(float(d['median']))} | {_fmt_deg(float(d['max']))} | {dw5_s} | "
            f"{_fmt_deg(float(b['median']))} | {_fmt_deg(float(b['max']))} | {bw5_s} |"
        )


def _diagnose(rows_by_case: Dict[str, Dict[str, Any]]) -> None:
    # Heuristic, intentionally simple: compare a few deltas against base.
    base = rows_by_case.get("base")
    if not base:
        return

    def _get(case: str, metric: str, field: str) -> Optional[float]:
        r = rows_by_case.get(case)
        if not r:
            return None
        st = r.get(metric)
        if isinstance(st, dict):
            v = st.get(field)
            try:
                return float(v)
            except Exception:
                return None
        return None

    def _get_w5(case: str, which: str) -> Optional[float]:
        r = rows_by_case.get(case)
        if not r:
            return None
        w = r.get(which)
        if isinstance(w, dict):
            try:
                return float(w.get("avg"))
            except Exception:
                return None
        return None

    # Compare no_lambda vs base (gate effect)
    base_blend_med = _get("base", "steady_blend", "median")
    base_blend_max = _get("base", "steady_blend", "max")
    base_blend_w5 = _get_w5("base", "blend_worst5")

    nl_blend_med = _get("no_lambda", "steady_blend", "median")
    nl_blend_max = _get("no_lambda", "steady_blend", "max")
    nl_blend_w5 = _get_w5("no_lambda", "blend_worst5")

    sync_blend_w5 = _get_w5("seam_sync_state", "blend_worst5")
    gt_direct_med = _get("direct_hint_gt", "steady_direct", "median")
    gt_direct_max = _get("direct_hint_gt", "steady_direct", "max")
    zero_direct_med = _get("direct_hint_zero", "steady_direct", "median")

    # Oracle gap (how much better could an ideal chooser be on this rollout)
    oracle_blend_med = _get("base", "steady_oracle_min_inc_direct", "median")
    oracle_blend_max = _get("base", "steady_oracle_min_inc_direct", "max")

    print("\n**Diagnosis Hints (heuristic)**")
    if base_blend_w5 is not None and nl_blend_w5 is not None:
        delta = float(nl_blend_w5 - base_blend_w5)
        if delta < -0.3:
            print(f"- `lambda_fusion_apply` hurts seam-window: no_lambda better by {abs(delta):.2f}° ⇒ prioritize λ/gate training.")
        elif delta > 0.3:
            print(f"- `lambda_fusion_apply` helps seam-window: base better by {abs(delta):.2f}° ⇒ keep λ, focus elsewhere.")
        else:
            print(f"- `lambda_fusion_apply` seam-window difference small (Δ {delta:+.2f}°).")

    if base_blend_med is not None and nl_blend_med is not None:
        print(f"- Blend steady median: base={_fmt_deg(base_blend_med)} vs no_lambda={_fmt_deg(nl_blend_med)}.")
    if base_blend_max is not None and nl_blend_max is not None:
        print(f"- Blend steady max: base={_fmt_deg(base_blend_max)} vs no_lambda={_fmt_deg(nl_blend_max)}.")

    if base_blend_w5 is not None and sync_blend_w5 is not None:
        delta = float(sync_blend_w5 - base_blend_w5)
        if delta < -0.5:
            print(f"- Seam state sync collapses seam-window by {abs(delta):.2f}° ⇒ seam carry mismatch is a dominant bottleneck.")
        else:
            print(f"- Seam state sync changes seam-window by {delta:+.2f}° ⇒ seam may be intrinsic (not only carry mismatch).")

    if base is not None and gt_direct_med is not None and gt_direct_max is not None:
        base_direct_med = _get("base", "steady_direct", "median")
        base_direct_max = _get("base", "steady_direct", "max")
        if base_direct_med is not None and base_direct_max is not None:
            dmed = float(gt_direct_med - base_direct_med)
            dmax = float(gt_direct_max - base_direct_max)
            if dmed < -0.3 or dmax < -0.7:
                print(f"- Direct head improves with GT contacts (Δmed {dmed:+.2f}°, Δmax {dmax:+.2f}°) ⇒ contacts/phase-hint is limiting.")
            else:
                print(f"- Direct head barely changes with GT contacts (Δmed {dmed:+.2f}°, Δmax {dmax:+.2f}°) ⇒ direct head itself is limiting.")

    if zero_direct_med is not None:
        base_direct_med = _get("base", "steady_direct", "median")
        if base_direct_med is not None:
            dmed = float(zero_direct_med - base_direct_med)
            if dmed > 0.3:
                print(f"- Direct head worsens when contacts hints are zero (Δmed {dmed:+.2f}°) ⇒ direct relies on contacts.")
            else:
                print(f"- Direct head similar with zero contacts (Δmed {dmed:+.2f}°) ⇒ contacts hints not critical.")

    if oracle_blend_med is not None and oracle_blend_max is not None and base_blend_med is not None and base_blend_max is not None:
        gap_med = float(base_blend_med - oracle_blend_med)
        gap_max = float(base_blend_max - oracle_blend_max)
        if gap_med > 0.3 or gap_max > 0.7:
            print(f"- Oracle(min(inc,direct)) is much better than Blend (gap med {gap_med:.2f}°, gap max {gap_max:.2f}°) ⇒ λ/gate has clear headroom.")
        else:
            print(f"- Oracle(min(inc,direct)) close to Blend ⇒ improving gate alone may not move much.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a minimal freerun_cycles tail/seam ablation suite.")
    ap.add_argument(
        "--teacher",
        nargs="+",
        default=["validate/teacher_batches/Walk_F_teacher.json"],
        help="Teacher JSON file(s) / directory / glob(s) (passed to freerun_cycles).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_lambda_cycles2_after_direct_pose.pth",
        help="Checkpoint path (.pth) for freerun_cycles.",
    )
    ap.add_argument("--out-root", type=str, default="debug_output/ablation_suite_tail", help="Root directory for case outputs.")
    ap.add_argument("--rounds", type=int, default=5, help="freerun_cycles rounds.")
    ap.add_argument("--no-run", action="store_true", help="Skip running; only summarize existing case outputs.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing freerun outputs (passed to freerun_cycles).")
    ap.add_argument(
        "--run-arg",
        action="append",
        default=[],
        help="Extra arg token(s) forwarded to freerun_cycles for all cases (repeatable).",
    )
    ap.add_argument("--include-plan-z-reset", action="store_true", help="Add 6th case: seam_reset_plan_z.")
    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    cases = _cases(bool(args.include_plan_z_reset))
    rows: List[Dict[str, Any]] = []
    rows_by_case: Dict[str, Dict[str, Any]] = {}

    for c in cases:
        case_dir = out_root / _slug(c.name)
        if not args.no_run:
            _run_freerun_cycles(
                teacher=args.teacher,
                model=args.model,
                out_dir=case_dir,
                rounds=int(args.rounds),
                force=bool(args.force),
                extra_args=tuple(c.args) + tuple(args.run_arg),
            )

        outs = _find_outputs(case_dir)
        if not outs:
            raise FileNotFoundError(f"No *_freerun_cycles.json under {case_dir} (did the run succeed?)")
        if len(outs) != 1:
            raise RuntimeError(
                f"Expected exactly 1 *_freerun_cycles.json under {case_dir} for this suite, found {len(outs)}. "
                "If you passed multiple teachers/clips, run per-clip or extend the summarizer."
            )
        summary = summarize_freerun_cycles_tail(outs[0])
        summary["case"] = c.name
        rows.append(summary)
        rows_by_case[c.name] = summary

    print("\n**Outputs**")
    print("| Case | out_json | model |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {r['case']} | `{r['out_json']}` | `{r.get('model')}` |")

    print("\n**Tail Summary (cycles1-4)**")
    _print_case_summary_table(rows)

    _diagnose(rows_by_case)

    # Persist a machine-readable summary for later analysis.
    out_summary = out_root / "summary_tail_ablation.json"
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] wrote `{out_summary}`")


if __name__ == "__main__":
    main()
