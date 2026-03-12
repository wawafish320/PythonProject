#!/usr/bin/env python3
"""
Stage6/Stage7 transition helper for:
  docs/changes/2026-02-16_stage6_stage7_transition_keep_anchors_replace_oracle_with_loss_budget.md

It wraps `tools/run_loss_budget_r05_g0.py` into two practical entrypoints:
1) quickgate: L1/L2 fast A/B gate (Arm A=with L1/L2, Arm B=skip L1/L2)
2) mainline: N0/N1-leg/N1b/N2/N3 node templates (`N1` kept as trigger-only control)

Loss-budget backend wiring is posttrain-only (legacy backend removed).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_ROOT = Path(__file__).resolve().parents[1]
_RUN_LOSS_BUDGET = _ROOT / "tools" / "run_loss_budget_r05_g0.py"
_C1_GATE_TOL_BY_MODE = {
    "selector_fix_tau7": 7.0,
    "strict0": 0.0,
}


def _resolve_from_root(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (_ROOT / p)


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _print_cmd(cmd: Sequence[str]) -> None:
    print("[cmd] " + " ".join(str(x) for x in cmd))


def _run(cmd: Sequence[str], *, dry_run: bool) -> None:
    _print_cmd(cmd)
    if dry_run:
        return
    rc = subprocess.call([str(x) for x in cmd], cwd=str(_ROOT))
    if int(rc) != 0:
        raise SystemExit(f"[FATAL] command failed (exit={rc})")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_row(summary_obj: Mapping[str, Any]) -> Dict[str, Any]:
    rows = summary_obj.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return {}
    preferred: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        status = str(r.get("status", ""))
        if status.startswith("ok") or status == "skipped_existing":
            preferred.append(r)
    if preferred:
        return preferred[0]
    first = rows[0]
    return first if isinstance(first, dict) else {}


def _teacher_metric(row: Mapping[str, Any], key: str) -> float:
    m = row.get("teacher_metrics", {})
    if not isinstance(m, Mapping):
        return float("nan")
    return _safe_float(m.get(key, float("nan")))


def _build_quickgate_report(
    *,
    arm_a_summary: Path,
    arm_b_summary: Path,
    out_root: Path,
    run_tag: str,
) -> None:
    if not arm_a_summary.is_file() or not arm_b_summary.is_file():
        print("[WARN] quickgate summary missing; skip teacher report")
        return

    a = _load_json(arm_a_summary)
    b = _load_json(arm_b_summary)
    row_a = _pick_row(a)
    row_b = _pick_row(b)

    metrics = [
        "direct_pose_geo_deg",
        "under_correct_frac_trigger_twist",
        "direct_pose_trigger_frac",
        "direct_pose_trigger_total_weighted",
    ]

    metric_table: Dict[str, Dict[str, Any]] = {}
    for key in metrics:
        va = _teacher_metric(row_a, key)
        vb = _teacher_metric(row_b, key)
        metric_table[key] = {
            "arm_a": va,
            "arm_b": vb,
            "delta_a_minus_b": va - vb if math.isfinite(va) and math.isfinite(vb) else float("nan"),
        }

    geo_a = metric_table["direct_pose_geo_deg"]["arm_a"]
    geo_b = metric_table["direct_pose_geo_deg"]["arm_b"]
    under_a = metric_table["under_correct_frac_trigger_twist"]["arm_a"]
    under_b = metric_table["under_correct_frac_trigger_twist"]["arm_b"]

    teacher_geo_not_worse: Optional[bool]
    teacher_under_not_worse: Optional[bool]
    if math.isfinite(geo_a) and math.isfinite(geo_b):
        teacher_geo_not_worse = bool(geo_a <= geo_b)
    else:
        teacher_geo_not_worse = None
    if math.isfinite(under_a) and math.isfinite(under_b):
        teacher_under_not_worse = bool(under_a <= under_b)
    else:
        teacher_under_not_worse = None
    if teacher_geo_not_worse is None or teacher_under_not_worse is None:
        teacher_quick_pass: Optional[bool] = None
    else:
        teacher_quick_pass = bool(teacher_geo_not_worse and teacher_under_not_worse)

    payload: Dict[str, Any] = {
        "run_tag": run_tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "arm_a_summary": str(arm_a_summary),
        "arm_b_summary": str(arm_b_summary),
        "arm_a_status": str(row_a.get("status", "unknown")),
        "arm_b_status": str(row_b.get("status", "unknown")),
        "teacher_metrics": metric_table,
        "teacher_gate": {
            "direct_pose_geo_not_worse": teacher_geo_not_worse,
            "under_correct_not_worse": teacher_under_not_worse,
            "teacher_quick_pass": teacher_quick_pass,
            "needs_freerun_ab": True,
            "note": "Doc 4.4 requires trigger_branch delta_err/new-max checks from freerun A/B.",
        },
    }

    out_json = out_root / "quickgate_teacher_read.json"
    _write_json(out_json, payload)

    lines: List[str] = []
    lines.append("# Stage6/7 L1/L2 Quickgate (Teacher Quick Read)")
    lines.append("")
    lines.append(f"- run_tag: `{run_tag}`")
    lines.append(f"- arm_a_summary: `{arm_a_summary}`")
    lines.append(f"- arm_b_summary: `{arm_b_summary}`")
    lines.append(f"- arm_a_status: `{payload['arm_a_status']}`")
    lines.append(f"- arm_b_status: `{payload['arm_b_status']}`")
    lines.append("")
    lines.append("| metric | armA_withL12 | armB_skipL12 | delta(A-B) |")
    lines.append("|:--|--:|--:|--:|")
    for key in metrics:
        row = metric_table[key]
        lines.append(
            f"| {key} | {row['arm_a']:.6f} | {row['arm_b']:.6f} | {row['delta_a_minus_b']:+.6f} |"
            if math.isfinite(_safe_float(row["arm_a"]))
            and math.isfinite(_safe_float(row["arm_b"]))
            and math.isfinite(_safe_float(row["delta_a_minus_b"]))
            else f"| {key} | {row['arm_a']} | {row['arm_b']} | {row['delta_a_minus_b']} |"
        )
    lines.append("")
    lines.append("## Teacher Gate")
    lines.append("")
    lines.append(f"- direct_pose_geo_not_worse: `{teacher_geo_not_worse}`")
    lines.append(f"- under_correct_not_worse: `{teacher_under_not_worse}`")
    lines.append(f"- teacher_quick_pass: `{payload['teacher_gate']['teacher_quick_pass']}`")
    lines.append("- needs_freerun_ab: `true`")
    lines.append("")
    lines.append("Follow-up (doc 4.4.2): run freerun A/B and check trigger_branch delta_err + worst-point migration.")
    out_md = out_root / "quickgate_teacher_read.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


def _write_quickgate_stub_summary(
    *,
    summary_path: Path,
    ckpt: Path,
    seed: int,
    run_name: str,
    note: str,
) -> None:
    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status_count": {"raw_ckpt": 1},
        "rows": [
            {
                "case": "raw_ckpt",
                "stage": "L1/L2 quickgate",
                "seed": int(seed),
                "run_name": str(run_name),
                "run_dir": "",
                "log_path": "",
                "status": "raw_ckpt",
                "resume_ckpt": str(ckpt),
                "ckpt": str(ckpt),
                "exit_code": 0,
                "note": str(note),
                "config_overrides": [],
                "teacher_metrics": {},
            }
        ],
    }
    _write_json(summary_path, payload)


def _add_common_loss_budget_args(
    cmd: List[str],
    *,
    config_json: Path,
    resume_ckpt: Path,
    out_dir: Path,
    out_model_dir: Path,
    cases: str,
    seeds: str,
    epochs: int,
    dataset_index_mode: str,
    base_run_name: str,
    train_overrides: Sequence[str],
    skip_existing: bool,
    dry_run: bool,
) -> List[str]:
    cmd.extend(
        [
            str(sys.executable),
            str(_RUN_LOSS_BUDGET),
            "--config-json",
            str(config_json),
            "--resume-ckpt",
            str(resume_ckpt),
            "--out-dir",
            str(out_dir),
            "--out-model-dir",
            str(out_model_dir),
            "--cases",
            str(cases),
            "--seeds",
            str(seeds),
            "--epochs",
            str(int(epochs)),
            "--dataset-index-mode",
            str(dataset_index_mode),
            "--base-run-name",
            str(base_run_name),
        ]
    )
    for ov in train_overrides:
        txt = str(ov).strip()
        if txt:
            cmd.extend(["--train-config-override", txt])
    if skip_existing:
        cmd.append("--skip-existing")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _pick_summary_row_by_seed(summary_obj: Mapping[str, Any], seed: Optional[int]) -> Dict[str, Any]:
    rows = summary_obj.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return {}

    candidates: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if seed is not None:
            s_val = r.get("seed", None)
            try:
                s_int = int(s_val) if s_val is not None else None
            except Exception:
                s_int = None
            if s_int != int(seed):
                continue
        candidates.append(r)
    if not candidates:
        candidates = [r for r in rows if isinstance(r, dict)]
    if not candidates:
        return {}

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for r in candidates:
        st = str(r.get("status", ""))
        ckpt = str(r.get("ckpt", "")).strip()
        score = 0
        if st.startswith("ok") or st == "skipped_existing":
            score += 10
        if ckpt:
            score += 3
            if Path(ckpt).is_file():
                score += 3
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _resolve_ckpt_from_summary(summary_path: Path, seed: Optional[int]) -> Tuple[Path, Dict[str, Any]]:
    if not summary_path.is_file():
        raise SystemExit(f"[FATAL] summary not found: {summary_path}")
    obj = _load_json(summary_path)
    row = _pick_summary_row_by_seed(obj, seed)
    ckpt_raw = str(row.get("ckpt", "")).strip() if isinstance(row, dict) else ""
    if not ckpt_raw:
        raise SystemExit(f"[FATAL] ckpt missing in summary row: {summary_path}")
    ckpt = Path(ckpt_raw).expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"[FATAL] ckpt path from summary not found: {ckpt}")
    return ckpt, row


def _finite_values(vals: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for v in vals:
        x = _safe_float(v)
        if math.isfinite(x):
            out.append(float(x))
    return out


def _quantile(vals: Sequence[Any], q: float) -> float:
    arr = sorted(_finite_values(vals))
    if not arr:
        return float("nan")
    qq = float(max(0.0, min(1.0, q)))
    idx = int(round(qq * (len(arr) - 1)))
    idx = max(0, min(len(arr) - 1, idx))
    return float(arr[idx])


def _mean(vals: Sequence[Any]) -> float:
    arr = _finite_values(vals)
    if not arr:
        return float("nan")
    return float(sum(arr) / len(arr))


def _build_step_mask(steps: Sequence[Mapping[str, Any]], *, cycle_gte: int, drop_wrap: bool) -> List[bool]:
    mask: List[bool] = []
    for ent in steps:
        cyc = int(ent.get("cycle", 0) or 0)
        if cyc < int(cycle_gte):
            mask.append(False)
            continue
        if drop_wrap and bool(ent.get("wrap_boundary_step", False)):
            mask.append(False)
            continue
        mask.append(True)
    return mask


def _extract_direct_series(obj: Mapping[str, Any]) -> Tuple[List[str], int, List[List[float]]]:
    per = obj.get("per_step_direct_geolocal_deg", {})
    if not isinstance(per, Mapping):
        raise SystemExit("[FATAL] missing per_step_direct_geolocal_deg (need --export_joint_direct_geolocal_series)")
    names = per.get("bone_names", [])
    mat = per.get("DirectGeoLocalDeg", [])
    if not isinstance(names, list) or not names:
        raise SystemExit("[FATAL] invalid per_step_direct_geolocal_deg.bone_names")
    if not isinstance(mat, list) or not mat:
        raise SystemExit("[FATAL] invalid per_step_direct_geolocal_deg.DirectGeoLocalDeg")
    try:
        root_idx = int(per.get("root_idx", 0) or 0)
    except Exception:
        root_idx = 0
    out_mat: List[List[float]] = []
    for row in mat:
        if not isinstance(row, list):
            out_mat.append([])
            continue
        out_mat.append([_safe_float(x) for x in row])
    return [str(x) for x in names], int(root_idx), out_mat


def _extract_rotvec_series(obj: Mapping[str, Any]) -> Optional[Tuple[List[str], List[List[List[float]]]]]:
    so3 = obj.get("per_step_joint_so3_error", {})
    if not isinstance(so3, Mapping):
        return None
    names = so3.get("bone_names", [])
    branches = so3.get("branches", {})
    if not isinstance(names, list) or not names or not isinstance(branches, Mapping):
        return None
    direct = branches.get("direct", {})
    if not isinstance(direct, Mapping):
        return None
    body = direct.get("body", {})
    if not isinstance(body, Mapping):
        return None
    rotvec = body.get("rotvec_deg_xyz", None)
    if not isinstance(rotvec, list) or not rotvec:
        return None
    out: List[List[List[float]]] = []
    for step_row in rotvec:
        if not isinstance(step_row, list):
            out.append([])
            continue
        jr: List[List[float]] = []
        for joint_row in step_row:
            if not isinstance(joint_row, list) or len(joint_row) < 3:
                jr.append([float("nan"), float("nan"), float("nan")])
            else:
                jr.append([_safe_float(joint_row[0]), _safe_float(joint_row[1]), _safe_float(joint_row[2])])
        out.append(jr)
    return [str(x) for x in names], out


def _safe_contact_pair(step_ent: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    for key in ("ContactGTPerC", "ContactMeasPerC", "ContactPlanPerC"):
        v = step_ent.get(key, None)
        if isinstance(v, list) and len(v) >= 2:
            l = _safe_float(v[0])
            r = _safe_float(v[1])
            if math.isfinite(l) and math.isfinite(r):
                return float(l), float(r)
    return None


def _phase_label(l: float, r: float, *, stance_thr: float, flight_thr: float, dom_margin: float) -> str:
    if not (math.isfinite(l) and math.isfinite(r)):
        return "phase_invalid"
    if l >= stance_thr and r >= stance_thr:
        return "phase_double_support"
    if l <= flight_thr and r <= flight_thr:
        return "phase_flight"
    if (l - r) > dom_margin:
        return "phase_left_stance"
    if (r - l) > dom_margin:
        return "phase_right_stance"
    return "phase_transition"


def _masked_flat_and_max(
    *,
    steps: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    root_idx: int,
    mat: Sequence[Sequence[float]],
    mask: Sequence[bool],
) -> Dict[str, Any]:
    vals: List[float] = []
    max_val = float("nan")
    max_meta: Dict[str, Any] = {}
    n_steps = min(len(steps), len(mat), len(mask))
    for i in range(n_steps):
        if not bool(mask[i]):
            continue
        row = mat[i]
        if not isinstance(row, Sequence):
            continue
        for j, v in enumerate(row):
            if int(j) == int(root_idx):
                continue
            x = _safe_float(v)
            if not math.isfinite(x):
                continue
            vals.append(x)
            if not math.isfinite(max_val) or x > max_val:
                max_val = float(x)
                ent = steps[i]
                max_meta = {
                    "value_deg": float(x),
                    "joint": str(names[j]) if j < len(names) else f"joint_{j}",
                    "joint_idx": int(j),
                    "step": int(ent.get("step", i) or i),
                    "cycle": int(ent.get("cycle", 0) or 0),
                    "step_in_cycle": int(ent.get("step_in_cycle", 0) or 0),
                }
    return {
        "n_samples": int(len(vals)),
        "mean_deg": _mean(vals),
        "p99_deg": _quantile(vals, 0.99),
        "max_deg": float(max_val) if math.isfinite(max_val) else float("nan"),
        "max_point": max_meta,
    }


def _top_worstpoint_regressions(
    *,
    base_steps: Sequence[Mapping[str, Any]],
    base_names: Sequence[str],
    base_root_idx: int,
    base_mat: Sequence[Sequence[float]],
    new_steps: Sequence[Mapping[str, Any]],
    new_names: Sequence[str],
    new_root_idx: int,
    new_mat: Sequence[Sequence[float]],
    mask: Sequence[bool],
    topn: int,
) -> List[Dict[str, Any]]:
    _ = base_root_idx  # keep explicit naming; root idx should match names order.
    rows: List[Dict[str, Any]] = []
    n_steps = min(len(base_steps), len(new_steps), len(base_mat), len(new_mat), len(mask))
    for i in range(n_steps):
        if not bool(mask[i]):
            continue
        br = base_mat[i]
        nr = new_mat[i]
        if not isinstance(br, Sequence) or not isinstance(nr, Sequence):
            continue
        n_joint = min(len(br), len(nr), len(base_names), len(new_names))
        for j in range(n_joint):
            if int(j) == int(base_root_idx) or int(j) == int(new_root_idx):
                continue
            b = _safe_float(br[j])
            n = _safe_float(nr[j])
            if not (math.isfinite(b) and math.isfinite(n)):
                continue
            dlt = float(n - b)
            if dlt <= 0.0:
                continue
            ent = new_steps[i]
            rows.append(
                {
                    "delta_deg": dlt,
                    "new_deg": float(n),
                    "base_deg": float(b),
                    "joint": str(new_names[j]),
                    "joint_idx": int(j),
                    "step": int(ent.get("step", i) or i),
                    "cycle": int(ent.get("cycle", 0) or 0),
                    "step_in_cycle": int(ent.get("step_in_cycle", 0) or 0),
                }
            )
    rows.sort(key=lambda x: float(x.get("delta_deg", float("nan"))), reverse=True)
    return rows[: max(1, int(topn))]


def _compute_trigger_branch_diag(
    *,
    obj: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    mat: Sequence[Sequence[float]],
    mask: Sequence[bool],
    target_joint: str,
    spike_sics: Sequence[int],
    gate_min_n: int,
    contact_stance_thr: float,
    contact_flight_thr: float,
    contact_dom_margin: float,
) -> Dict[str, Any]:
    rot = _extract_rotvec_series(obj)
    if rot is None:
        return {
            "status": "missing_so3_series",
            "note": "Need --export_joint_so3_error_series with direct/body.",
            "trigger_branch": "NA",
            "trigger_branch_delta_err_deg": float("nan"),
            "rows": [],
        }
    rot_names, rotvec = rot
    idx_map = {str(n): int(i) for i, n in enumerate(names)}
    rot_idx_map = {str(n): int(i) for i, n in enumerate(rot_names)}
    if str(target_joint) not in idx_map or str(target_joint) not in rot_idx_map:
        return {
            "status": "target_joint_missing",
            "note": f"target joint {target_joint} not found in series.",
            "trigger_branch": "NA",
            "trigger_branch_delta_err_deg": float("nan"),
            "rows": [],
        }
    j_err = idx_map[str(target_joint)]
    j_rot = rot_idx_map[str(target_joint)]

    recs: List[Dict[str, Any]] = []
    n_steps = min(len(steps), len(mat), len(mask), len(rotvec))
    for i in range(n_steps):
        if not bool(mask[i]):
            continue
        row_err = mat[i]
        row_rot = rotvec[i]
        if j_err >= len(row_err) or j_rot >= len(row_rot):
            continue
        err = _safe_float(row_err[j_err])
        rv = row_rot[j_rot]
        rvz = _safe_float(rv[2] if isinstance(rv, Sequence) and len(rv) >= 3 else float("nan"))
        if not (math.isfinite(err) and math.isfinite(rvz)):
            continue
        ent = steps[i]
        cp = _safe_contact_pair(ent)
        if cp is None:
            continue
        l, r = cp
        ph = _phase_label(
            float(l),
            float(r),
            stance_thr=float(contact_stance_thr),
            flight_thr=float(contact_flight_thr),
            dom_margin=float(contact_dom_margin),
        )
        recs.append(
            {
                "sic": int(ent.get("step_in_cycle", 0) or 0),
                "phase": ph,
                "err_deg": float(err),
                "rvz": float(rvz),
                "right_contact": float(r),
            }
        )

    if not recs:
        return {
            "status": "empty_records",
            "note": "No usable masked records for trigger branch diagnosis.",
            "trigger_branch": "NA",
            "trigger_branch_delta_err_deg": float("nan"),
            "rows": [],
        }

    spike_set = set(int(x) for x in spike_sics)
    phase_name = "phase_left_stance"
    rows: List[Dict[str, Any]] = []
    for sign_key in ("twist_neg", "twist_pos", "twist_zero"):
        for gate_key in ("r_contact_low", "r_contact_mid", "r_contact_high"):
            spike_vals: List[float] = []
            ctrl_vals: List[float] = []
            for r in recs:
                if str(r["phase"]) != phase_name:
                    continue
                rvz = float(r["rvz"])
                rc = float(r["right_contact"])
                if sign_key == "twist_neg" and not (rvz < 0.0):
                    continue
                if sign_key == "twist_pos" and not (rvz > 0.0):
                    continue
                if sign_key == "twist_zero" and not (rvz == 0.0):
                    continue
                if gate_key == "r_contact_low" and not (rc <= float(contact_flight_thr)):
                    continue
                if gate_key == "r_contact_mid" and not (float(contact_flight_thr) < rc < float(contact_stance_thr)):
                    continue
                if gate_key == "r_contact_high" and not (rc >= float(contact_stance_thr)):
                    continue
                if int(r["sic"]) in spike_set:
                    spike_vals.append(float(r["err_deg"]))
                else:
                    ctrl_vals.append(float(r["err_deg"]))

            s_mean = _mean(spike_vals)
            c_mean = _mean(ctrl_vals)
            dlt = float(s_mean - c_mean) if math.isfinite(s_mean) and math.isfinite(c_mean) else float("nan")
            rows.append(
                {
                    "branch": f"{sign_key}__{gate_key}",
                    "twist_sign": sign_key,
                    "contact_gate": gate_key,
                    "n_spike": int(len(spike_vals)),
                    "n_control": int(len(ctrl_vals)),
                    "delta_err_deg": dlt,
                }
            )

    rows.sort(
        key=lambda r: abs(_safe_float(r.get("delta_err_deg", float("nan"))))
        if math.isfinite(_safe_float(r.get("delta_err_deg", float("nan"))))
        else -1.0,
        reverse=True,
    )

    robust = [
        r
        for r in rows
        if int(r.get("n_spike", 0)) >= int(gate_min_n)
        and int(r.get("n_control", 0)) >= int(gate_min_n)
        and math.isfinite(_safe_float(r.get("delta_err_deg", float("nan"))))
    ]
    top = robust[0] if robust else (rows[0] if rows else {})
    trigger_branch = str(top.get("branch", "NA")) if isinstance(top, Mapping) else "NA"
    trigger_delta = _safe_float(top.get("delta_err_deg", float("nan"))) if isinstance(top, Mapping) else float("nan")

    low_contact_robust = [r for r in robust if str(r.get("contact_gate", "")) == "r_contact_low"]
    trigger_low_contact = str(low_contact_robust[0].get("branch", "NA")) if low_contact_robust else "NA"

    return {
        "status": "ok",
        "phase": phase_name,
        "target_joint": str(target_joint),
        "trigger_branch": trigger_branch,
        "trigger_branch_low_contact": trigger_low_contact,
        "trigger_branch_delta_err_deg": trigger_delta,
        "rows": rows,
        "robust_rows": robust,
    }


def _compare_le(a: float, b: float, tol: float) -> Optional[bool]:
    aa = _safe_float(a)
    bb = _safe_float(b)
    tt = _safe_float(tol)
    if not (math.isfinite(aa) and math.isfinite(bb) and math.isfinite(tt)):
        return None
    return bool(aa <= bb + tt)


def _decision_from_criteria(criteria_vals: Sequence[Optional[bool]]) -> str:
    if any(x is False for x in criteria_vals):
        return "fail"
    if all(x is True for x in criteria_vals):
        return "pass"
    return "inconclusive"


def _first_robust_delta(branch_obj: Mapping[str, Any]) -> float:
    robust_rows = branch_obj.get("robust_rows", [])
    if not isinstance(robust_rows, list) or not robust_rows:
        return float("nan")
    first = robust_rows[0]
    if not isinstance(first, Mapping):
        return float("nan")
    return _safe_float(first.get("delta_err_deg", float("nan")))


def _pick_c1_delta(branch_obj: Mapping[str, Any], policy: str) -> Tuple[float, str]:
    selected_delta = _safe_float(branch_obj.get("trigger_branch_delta_err_deg", float("nan")))
    robust_delta = _first_robust_delta(branch_obj)
    p = str(policy).strip().lower()
    if p == "robust_only":
        if math.isfinite(robust_delta):
            return robust_delta, "robust_rows"
        return float("nan"), "insufficient_robust_rows"
    return selected_delta, "selected_branch"


def _resolve_c1_trigger_tol(args: argparse.Namespace) -> Tuple[float, str, str]:
    mode = str(getattr(args, "c1_gate_mode", "selector_fix_tau7")).strip().lower()
    if mode not in _C1_GATE_TOL_BY_MODE:
        mode = "selector_fix_tau7"
    override = getattr(args, "trigger_delta_tol", None)
    if override is not None:
        tol = _safe_float(override)
        if math.isfinite(tol):
            return float(tol), mode, "cli_override"
    return float(_C1_GATE_TOL_BY_MODE[mode]), mode, f"mode_default:{mode}"


def _run_quickgate(args: argparse.Namespace) -> None:
    if not _RUN_LOSS_BUDGET.is_file():
        raise SystemExit(f"[FATAL] missing helper script: {_RUN_LOSS_BUDGET}")

    cfg = _resolve_from_root(args.config_json)
    stage6_ckpt = _resolve_from_root(args.stage6_ckpt)
    l2_ckpt = _resolve_from_root(args.l2_ckpt)
    if not cfg.is_file():
        raise SystemExit(f"[FATAL] missing config json: {cfg}")
    if not args.dry_run:
        if not stage6_ckpt.is_file():
            raise SystemExit(f"[FATAL] missing stage6 ckpt: {stage6_ckpt}")
        if not l2_ckpt.is_file():
            raise SystemExit(f"[FATAL] missing l2 ckpt: {l2_ckpt}")

    run_tag = str(args.run_tag).strip()
    if not run_tag:
        run_tag = f"stage67_l12_quickgate_{datetime.now().strftime('%Y%m%d')}"

    out_root = _resolve_from_root(args.out_root) if str(args.out_root).strip() else (_ROOT / "debug_output" / f"_{run_tag}")
    model_root = (
        _resolve_from_root(args.model_root)
        if str(args.model_root).strip()
        else (_ROOT / "models" / f"MLPL2_DirectBranch_v1__{run_tag}")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    model_root.parent.mkdir(parents=True, exist_ok=True)

    common_ov = [
        f"w_direct_pose_trigger_twist={float(args.w_direct_pose_trigger_twist):.8g}",
        f"w_direct_pose_trigger_swing_x={float(args.w_direct_pose_trigger_swing_x):.8g}",
    ]
    for x in args.train_config_override or []:
        txt = str(x).strip()
        if txt:
            common_ov.append(txt)
    # Lock N-line direct-pose semantics to absolute to avoid default drift.
    common_ov.append("direct_pose_fusion_direct_mode=absolute")

    arm_a_out = out_root / "armA_withL12"
    arm_b_out = out_root / "armB_skipL12"
    arm_a_model = Path(f"{model_root}_armA")
    arm_b_model = Path(f"{model_root}_armB")

    mode = str(args.quickgate_mode).strip().lower()
    if mode == "raw_ckpt_ab":
        if args.dry_run:
            print("[INFO] quickgate raw_ckpt_ab (dry-run): skip training, keep ArmA/ArmB ckpt as-is.")
            return
        note = (
            "Raw-ckpt quickgate mode: no extra R0.5 training. "
            "Use freerun-ab to compare ArmA/ArmB directly under fixed eval mask."
        )
        arm_a_out.mkdir(parents=True, exist_ok=True)
        arm_b_out.mkdir(parents=True, exist_ok=True)
        _write_quickgate_stub_summary(
            summary_path=arm_a_out / "summary.json",
            ckpt=l2_ckpt,
            seed=int(args.seed),
            run_name="raw_ckpt_armA_withL12",
            note=note,
        )
        _write_quickgate_stub_summary(
            summary_path=arm_b_out / "summary.json",
            ckpt=stage6_ckpt,
            seed=int(args.seed),
            run_name="raw_ckpt_armB_skipL12",
            note=note,
        )
        _build_quickgate_report(
            arm_a_summary=arm_a_out / "summary.json",
            arm_b_summary=arm_b_out / "summary.json",
            out_root=out_root,
            run_tag=run_tag,
        )
        print("[INFO] raw_ckpt_ab summaries ready; run `freerun-ab` for gate decision.")
        return

    cmd_a: List[str] = []
    _add_common_loss_budget_args(
        cmd_a,
        config_json=cfg,
        resume_ckpt=l2_ckpt,
        out_dir=arm_a_out,
        out_model_dir=arm_a_model,
        cases="r05",
        seeds=str(int(args.seed)),
        epochs=int(args.epochs),
        dataset_index_mode=str(args.dataset_index_mode),
        base_run_name="r05_quick_armA_withL12",
        train_overrides=common_ov,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )
    cmd_a.extend([
        "--r05-under-mode",
        str(args.r05_under_mode),
        "--r05-under-weights",
        str(float(args.under_weight)),
    ])

    cmd_b: List[str] = []
    _add_common_loss_budget_args(
        cmd_b,
        config_json=cfg,
        resume_ckpt=stage6_ckpt,
        out_dir=arm_b_out,
        out_model_dir=arm_b_model,
        cases="r05",
        seeds=str(int(args.seed)),
        epochs=int(args.epochs),
        dataset_index_mode=str(args.dataset_index_mode),
        base_run_name="r05_quick_armB_skipL12",
        train_overrides=common_ov,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )
    cmd_b.extend([
        "--r05-under-mode",
        str(args.r05_under_mode),
        "--r05-under-weights",
        str(float(args.under_weight)),
    ])

    print("[INFO] quickgate arm A (with L1/L2)")
    _run(cmd_a, dry_run=bool(args.dry_run))
    print("[INFO] quickgate arm B (skip L1/L2)")
    _run(cmd_b, dry_run=bool(args.dry_run))

    if not args.dry_run:
        _build_quickgate_report(
            arm_a_summary=arm_a_out / "summary.json",
            arm_b_summary=arm_b_out / "summary.json",
            out_root=out_root,
            run_tag=run_tag,
        )


def _run_freerun_ab(args: argparse.Namespace) -> None:
    quick_root = (
        _resolve_from_root(args.quickgate_root)
        if str(args.quickgate_root).strip()
        else (_ROOT / "debug_output" / f"_{str(args.run_tag).strip()}")
    )
    out_root = (
        _resolve_from_root(args.out_root)
        if str(args.out_root).strip()
        else (quick_root / "freerun_ab")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    seed: Optional[int] = int(args.seed) if args.seed is not None else None
    arm_a_row: Dict[str, Any] = {}
    arm_b_row: Dict[str, Any] = {}

    if str(args.arm_a_ckpt).strip():
        ckpt_a = _resolve_from_root(args.arm_a_ckpt)
        if not args.dry_run and not ckpt_a.is_file():
            raise SystemExit(f"[FATAL] arm A ckpt not found: {ckpt_a}")
    else:
        sum_a = (
            _resolve_from_root(args.arm_a_summary)
            if str(args.arm_a_summary).strip()
            else (quick_root / "armA_withL12" / "summary.json")
        )
        ckpt_a, arm_a_row = _resolve_ckpt_from_summary(sum_a, seed=seed)

    if str(args.arm_b_ckpt).strip():
        ckpt_b = _resolve_from_root(args.arm_b_ckpt)
        if not args.dry_run and not ckpt_b.is_file():
            raise SystemExit(f"[FATAL] arm B ckpt not found: {ckpt_b}")
    else:
        sum_b = (
            _resolve_from_root(args.arm_b_summary)
            if str(args.arm_b_summary).strip()
            else (quick_root / "armB_skipL12" / "summary.json")
        )
        ckpt_b, arm_b_row = _resolve_ckpt_from_summary(sum_b, seed=seed)

    teacher = _resolve_from_root(args.teacher)
    bundle = _resolve_from_root(args.bundle)
    pretrain_template = _resolve_from_root(args.pretrain_template)
    encoder_bundle = _resolve_from_root(args.encoder_bundle)
    if not teacher.is_file():
        raise SystemExit(f"[FATAL] teacher not found: {teacher}")
    if not bundle.is_file():
        raise SystemExit(f"[FATAL] bundle not found: {bundle}")
    if not pretrain_template.is_file():
        raise SystemExit(f"[FATAL] pretrain template not found: {pretrain_template}")
    if not encoder_bundle.is_file():
        raise SystemExit(f"[FATAL] encoder bundle not found: {encoder_bundle}")

    arm_a_eval = out_root / "armA_withL12" / "C1_none"
    arm_b_eval = out_root / "armB_skipL12" / "C1_none"
    arm_a_eval.mkdir(parents=True, exist_ok=True)
    arm_b_eval.mkdir(parents=True, exist_ok=True)

    def _freerun_cmd(model_ckpt: Path, out_dir: Path) -> List[str]:
        cmd = [
            str(sys.executable),
            "-m",
            "train.validate.run_freerun_cycles",
            "--teacher",
            str(teacher),
            "--model",
            str(model_ckpt),
            "--bundle",
            str(bundle),
            "--pretrain-template",
            str(pretrain_template),
            "--encoder-bundle",
            str(encoder_bundle),
            "--out",
            str(out_dir),
            "--rounds",
            str(int(args.rounds)),
            "--depth",
            str(int(args.depth)),
            "--time-index-mode",
            str(args.time_index_mode),
            "--phase_reset_source",
            str(args.phase_reset_source),
            "--phase_reset_source_strict",
            str(args.phase_reset_source_strict),
            "--so3_corr_apply",
            "--log_contacts",
            "--export_joint_direct_geolocal_series",
            "--export_joint_so3_error_series",
            "--joint_so3_error_series_branches",
            "direct",
            "--joint_so3_error_series_space",
            "body",
            "--force",
            "--direct_pose_meas_source",
            str(getattr(args, "direct_pose_meas_source", "model")),
            "--contacts_meas_source",
            str(getattr(args, "contacts_meas_source", "model")),
        ]
        if bool(getattr(args, "lambda_fusion_apply", True)):
            cmd.append("--lambda_fusion_apply")
        mode = str(getattr(args, "direct_pose_fusion_direct_mode", "absolute") or "").strip().lower()
        if mode not in ("absolute", "residual_rot6d", "residual_compose_stable"):
            mode = "absolute"
        cmd.extend(["--direct_pose_fusion_direct_mode", mode])
        return cmd

    print("[INFO] freerun arm A (with L1/L2)")
    _run(_freerun_cmd(ckpt_a, arm_a_eval), dry_run=bool(args.dry_run))
    print("[INFO] freerun arm B (skip L1/L2)")
    _run(_freerun_cmd(ckpt_b, arm_b_eval), dry_run=bool(args.dry_run))

    if args.dry_run:
        return

    json_a = arm_a_eval / "Walk_F_freerun_cycles.json"
    json_b = arm_b_eval / "Walk_F_freerun_cycles.json"
    if not json_a.is_file() or not json_b.is_file():
        raise SystemExit(f"[FATAL] missing freerun json(s): A={json_a.is_file()} B={json_b.is_file()}")

    obj_a = _load_json(json_a)
    obj_b = _load_json(json_b)
    steps_a = obj_a.get("metrics_per_step", [])
    steps_b = obj_b.get("metrics_per_step", [])
    if not isinstance(steps_a, list) or not isinstance(steps_b, list) or not steps_a or not steps_b:
        raise SystemExit("[FATAL] invalid metrics_per_step in freerun outputs")

    names_a, root_a, mat_a = _extract_direct_series(obj_a)
    names_b, root_b, mat_b = _extract_direct_series(obj_b)

    mask_a = _build_step_mask(steps_a, cycle_gte=int(args.cycle_gte), drop_wrap=bool(args.drop_wrap))
    mask_b = _build_step_mask(steps_b, cycle_gte=int(args.cycle_gte), drop_wrap=bool(args.drop_wrap))
    mask_ab = [bool(a and b) for a, b in zip(mask_a[: min(len(mask_a), len(mask_b))], mask_b[: min(len(mask_a), len(mask_b))])]

    stats_a = _masked_flat_and_max(steps=steps_a, names=names_a, root_idx=root_a, mat=mat_a, mask=mask_a)
    stats_b = _masked_flat_and_max(steps=steps_b, names=names_b, root_idx=root_b, mat=mat_b, mask=mask_b)

    spike_sics = [int(x) for x in str(args.spike_sics).split(",") if str(x).strip()]
    branch_a = _compute_trigger_branch_diag(
        obj=obj_a,
        steps=steps_a,
        names=names_a,
        mat=mat_a,
        mask=mask_a,
        target_joint=str(args.target_joint),
        spike_sics=spike_sics,
        gate_min_n=int(args.gate_min_n),
        contact_stance_thr=float(args.contact_stance_thr),
        contact_flight_thr=float(args.contact_flight_thr),
        contact_dom_margin=float(args.contact_dom_margin),
    )
    branch_b = _compute_trigger_branch_diag(
        obj=obj_b,
        steps=steps_b,
        names=names_b,
        mat=mat_b,
        mask=mask_b,
        target_joint=str(args.target_joint),
        spike_sics=spike_sics,
        gate_min_n=int(args.gate_min_n),
        contact_stance_thr=float(args.contact_stance_thr),
        contact_flight_thr=float(args.contact_flight_thr),
        contact_dom_margin=float(args.contact_dom_margin),
    )

    top_reg = _top_worstpoint_regressions(
        base_steps=steps_b,
        base_names=names_b,
        base_root_idx=root_b,
        base_mat=mat_b,
        new_steps=steps_a,
        new_names=names_a,
        new_root_idx=root_a,
        new_mat=mat_a,
        mask=mask_ab,
        topn=int(args.topn),
    )

    c1_policy = str(args.c1_policy).strip().lower()
    trigger_delta_tol, c1_gate_mode, trigger_delta_tol_source = _resolve_c1_trigger_tol(args)
    selected_trig_a = _safe_float(branch_a.get("trigger_branch_delta_err_deg", float("nan")))
    selected_trig_b = _safe_float(branch_b.get("trigger_branch_delta_err_deg", float("nan")))
    robust_trig_a = _first_robust_delta(branch_a)
    robust_trig_b = _first_robust_delta(branch_b)
    trig_a, trig_a_source = _pick_c1_delta(branch_a, c1_policy)
    trig_b, trig_b_source = _pick_c1_delta(branch_b, c1_policy)
    under_a = _safe_float(
        ((arm_a_row.get("teacher_metrics", {}) if isinstance(arm_a_row, Mapping) else {}) or {}).get(
            "under_correct_frac_trigger_twist", float("nan")
        )
    )
    under_b = _safe_float(
        ((arm_b_row.get("teacher_metrics", {}) if isinstance(arm_b_row, Mapping) else {}) or {}).get(
            "under_correct_frac_trigger_twist", float("nan")
        )
    )
    share_trigger_a = _safe_float(
        ((arm_a_row.get("teacher_metrics", {}) if isinstance(arm_a_row, Mapping) else {}) or {}).get(
            "direct_pose_budget_share_trigger", float("nan")
        )
    )
    share_trigger_b = _safe_float(
        ((arm_b_row.get("teacher_metrics", {}) if isinstance(arm_b_row, Mapping) else {}) or {}).get(
            "direct_pose_budget_share_trigger", float("nan")
        )
    )
    share_trigger_min = float(args.share_trigger_min)
    share_trigger_max = float(args.share_trigger_max)
    if share_trigger_min > share_trigger_max:
        share_trigger_min, share_trigger_max = share_trigger_max, share_trigger_min
    share_trigger_a_in_range = (
        bool(share_trigger_min <= share_trigger_a <= share_trigger_max) if math.isfinite(share_trigger_a) else None
    )
    share_trigger_b_in_range = (
        bool(share_trigger_min <= share_trigger_b <= share_trigger_max) if math.isfinite(share_trigger_b) else None
    )
    arm_a_status = str(arm_a_row.get("status", "unknown")) if isinstance(arm_a_row, Mapping) else "unknown"
    arm_b_status = str(arm_b_row.get("status", "unknown")) if isinstance(arm_b_row, Mapping) else "unknown"
    quickgate_mode_inferred = (
        "raw_ckpt_ab" if arm_a_status == "raw_ckpt" and arm_b_status == "raw_ckpt" else "train_r05"
    )
    max_a = _safe_float(stats_a.get("max_deg", float("nan")))
    max_b = _safe_float(stats_b.get("max_deg", float("nan")))
    phase_a = str(obj_a.get("phase_reset_source_applied") or obj_a.get("phase_reset_source") or "").strip()
    phase_b = str(obj_b.get("phase_reset_source_applied") or obj_b.get("phase_reset_source") or "").strip()
    c0_phase_none = bool(phase_a == "none" and phase_b == "none")

    c1_branch_not_worse = _compare_le(trig_a, trig_b, float(trigger_delta_tol))
    c2_under_not_worse_raw = _compare_le(under_a, under_b, float(args.under_tol))
    c2_policy = str(args.c2_policy).strip().lower()
    if c2_policy == "ignore" and c2_under_not_worse_raw is None:
        c2_under_not_worse = True
    else:
        c2_under_not_worse = c2_under_not_worse_raw
    c3_max_not_worse = _compare_le(max_a, max_b, float(args.max_tol))

    criteria_legacy = [c0_phase_none, c1_branch_not_worse, c2_under_not_worse, c3_max_not_worse]
    decision_legacy = _decision_from_criteria(criteria_legacy)
    criteria_hard_gate_v2 = [c0_phase_none, c2_under_not_worse, c3_max_not_worse]
    decision_hard_gate_v2 = _decision_from_criteria(criteria_hard_gate_v2)
    branch_changed = str(branch_a.get("trigger_branch", "NA")) != str(branch_b.get("trigger_branch", "NA"))
    robust_rows_a = branch_a.get("robust_rows", [])
    robust_rows_b = branch_b.get("robust_rows", [])
    robust_rows_a_n = len(robust_rows_a) if isinstance(robust_rows_a, list) else 0
    robust_rows_b_n = len(robust_rows_b) if isinstance(robust_rows_b, list) else 0

    train_r05_stoploss_triggered_legacy = bool(
        quickgate_mode_inferred == "train_r05"
        and decision_legacy == "fail"
        and (c1_branch_not_worse is False or c3_max_not_worse is False)
    )
    train_r05_stoploss_triggered = bool(
        quickgate_mode_inferred == "train_r05"
        and decision_hard_gate_v2 == "fail"
        and (c3_max_not_worse is False)
    )
    if train_r05_stoploss_triggered:
        recommended_next = "stop_train_r05_sweeps_then_switch_to_stage6_n1_leg"
    elif decision_hard_gate_v2 == "pass":
        recommended_next = "optionally_resume_from_l2_then_enter_n_line"
    elif decision_hard_gate_v2 == "fail":
        recommended_next = "keep_l12_optional_and_use_stage6_n1_leg_default"
    else:
        recommended_next = "inconclusive_collect_more_evidence"

    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "quickgate_root": str(quick_root),
        "eval_out_root": str(out_root),
        "inputs": {
            "teacher": str(teacher),
            "bundle": str(bundle),
            "pretrain_template": str(pretrain_template),
            "encoder_bundle": str(encoder_bundle),
            "arm_a_ckpt": str(ckpt_a),
            "arm_b_ckpt": str(ckpt_b),
            "arm_a_json": str(json_a),
            "arm_b_json": str(json_b),
        },
        "mask": {
            "cycle_gte": int(args.cycle_gte),
            "drop_wrap": bool(args.drop_wrap),
            "exclude_root": True,
        },
        "phase_reset_source_applied": {
            "arm_a": phase_a,
            "arm_b": phase_b,
        },
        "teacher_metrics": {
            "arm_a_under_correct_frac_trigger_twist": under_a,
            "arm_b_under_correct_frac_trigger_twist": under_b,
            "arm_a_direct_pose_budget_share_trigger": share_trigger_a,
            "arm_b_direct_pose_budget_share_trigger": share_trigger_b,
            "arm_a_status": arm_a_status,
            "arm_b_status": arm_b_status,
        },
        "freerun_global": {
            "arm_a": stats_a,
            "arm_b": stats_b,
            "delta_a_minus_b": {
                "mean_deg": _safe_float(stats_a.get("mean_deg", float("nan"))) - _safe_float(stats_b.get("mean_deg", float("nan"))),
                "p99_deg": _safe_float(stats_a.get("p99_deg", float("nan"))) - _safe_float(stats_b.get("p99_deg", float("nan"))),
                "max_deg": _safe_float(stats_a.get("max_deg", float("nan"))) - _safe_float(stats_b.get("max_deg", float("nan"))),
            },
        },
        "trigger_branch": {
            "arm_a": branch_a,
            "arm_b": branch_b,
            "delta_err_deg_a_minus_b": trig_a - trig_b if math.isfinite(trig_a) and math.isfinite(trig_b) else float("nan"),
            "branch_changed": branch_changed,
            "robust_rows_count": {
                "arm_a": int(robust_rows_a_n),
                "arm_b": int(robust_rows_b_n),
            },
            "c1_delta_inputs": {
                "policy": c1_policy,
                "arm_a_selected_delta_err_deg": selected_trig_a,
                "arm_b_selected_delta_err_deg": selected_trig_b,
                "arm_a_robust_delta_err_deg": robust_trig_a,
                "arm_b_robust_delta_err_deg": robust_trig_b,
                "arm_a_gate_delta_err_deg": trig_a,
                "arm_b_gate_delta_err_deg": trig_b,
                "arm_a_gate_delta_source": trig_a_source,
                "arm_b_gate_delta_source": trig_b_source,
            },
        },
        "worstpoints_regression_topn": top_reg,
        "criteria": {
            "c0_phase_reset_source_applied_none": c0_phase_none,
            "c1_trigger_branch_delta_not_worse": c1_branch_not_worse,
            "c1_policy": c1_policy,
            "c1_gate_mode": c1_gate_mode,
            "c2_under_correct_not_worse_teacher": c2_under_not_worse,
            "c2_under_correct_not_worse_teacher_raw": c2_under_not_worse_raw,
            "c2_policy": c2_policy,
            "c3_new_max_not_worse": c3_max_not_worse,
            "trigger_delta_tol": float(trigger_delta_tol),
            "trigger_delta_tol_source": trigger_delta_tol_source,
            "under_tol": float(args.under_tol),
            "max_tol": float(args.max_tol),
        },
        "decision": {
            "quickgate_4_4_2": decision_legacy,
            "quickgate_4_4_2_rule": "pass iff c0/c1/c2/c3 are all True; fail if any False; else inconclusive.",
            "hard_gate_v2": decision_hard_gate_v2,
            "hard_gate_v2_rule": (
                "pass iff c0/c2/c3 are all True; fail if any False; else inconclusive. "
                "c1/branch_changed/robust_rows are monitoring-only."
            ),
            # Backward-compatible alias (legacy); prefer hard_gate_v2 in new tooling.
            "rule": "pass iff c0/c1/c2/c3 are all True; fail if any False; else inconclusive.",
            "quickgate_mode_inferred": quickgate_mode_inferred,
            "monitoring_only": {
                "c1_trigger_branch_delta_not_worse": c1_branch_not_worse,
                "c1_gate_mode": c1_gate_mode,
                "trigger_delta_tol": float(trigger_delta_tol),
                "trigger_delta_tol_source": trigger_delta_tol_source,
                "branch_changed": branch_changed,
                "robust_rows_count_arm_a": int(robust_rows_a_n),
                "robust_rows_count_arm_b": int(robust_rows_b_n),
                "share_trigger_range": [share_trigger_min, share_trigger_max],
                "arm_a_share_trigger": share_trigger_a,
                "arm_b_share_trigger": share_trigger_b,
                "arm_a_share_trigger_in_range": share_trigger_a_in_range,
                "arm_b_share_trigger_in_range": share_trigger_b_in_range,
            },
            "train_r05_stoploss_triggered": train_r05_stoploss_triggered,
            "train_r05_stoploss_triggered_legacy": train_r05_stoploss_triggered_legacy,
            "recommended_next": recommended_next,
        },
    }

    out_json = out_root / "freerun_ab_gate.json"
    _write_json(out_json, payload)

    lines: List[str] = []
    lines.append("# Stage6/7 Quickgate Freerun A/B Gate (Doc 4.4.2)")
    lines.append("")
    lines.append(f"- armA_withL12 ckpt: `{ckpt_a}`")
    lines.append(f"- armB_skipL12 ckpt: `{ckpt_b}`")
    lines.append(f"- armA json: `{json_a}`")
    lines.append(f"- armB json: `{json_b}`")
    lines.append("")
    lines.append("## Global DirectGeoLocalDeg (masked)")
    lines.append("")
    lines.append("| metric | armA_withL12 | armB_skipL12 | delta(A-B) |")
    lines.append("|:--|--:|--:|--:|")
    lines.append(
        f"| mean_deg | {_safe_float(stats_a.get('mean_deg', float('nan'))):.6f} | {_safe_float(stats_b.get('mean_deg', float('nan'))):.6f} | "
        f"{(_safe_float(stats_a.get('mean_deg', float('nan'))) - _safe_float(stats_b.get('mean_deg', float('nan')))):+.6f} |"
    )
    lines.append(
        f"| p99_deg | {_safe_float(stats_a.get('p99_deg', float('nan'))):.6f} | {_safe_float(stats_b.get('p99_deg', float('nan'))):.6f} | "
        f"{(_safe_float(stats_a.get('p99_deg', float('nan'))) - _safe_float(stats_b.get('p99_deg', float('nan')))):+.6f} |"
    )
    lines.append(
        f"| max_deg | {_safe_float(stats_a.get('max_deg', float('nan'))):.6f} | {_safe_float(stats_b.get('max_deg', float('nan'))):.6f} | "
        f"{(_safe_float(stats_a.get('max_deg', float('nan'))) - _safe_float(stats_b.get('max_deg', float('nan')))):+.6f} |"
    )
    lines.append("")
    lines.append("## Trigger Branch")
    lines.append("")
    lines.append(
        f"- armA trigger_branch: `{branch_a.get('trigger_branch', 'NA')}` "
        f"(delta_err_deg={_safe_float(branch_a.get('trigger_branch_delta_err_deg', float('nan'))):.6f})"
    )
    lines.append(
        f"- armB trigger_branch: `{branch_b.get('trigger_branch', 'NA')}` "
        f"(delta_err_deg={_safe_float(branch_b.get('trigger_branch_delta_err_deg', float('nan'))):.6f})"
    )
    lines.append(
        f"- teacher under_correct_frac_trigger_twist: armA={under_a:.6f} armB={under_b:.6f}"
        if math.isfinite(under_a) and math.isfinite(under_b)
        else "- teacher under_correct_frac_trigger_twist: unavailable"
    )
    lines.append(
        f"- teacher share_trigger: armA={share_trigger_a:.6f} armB={share_trigger_b:.6f} "
        f"(target range [{share_trigger_min:.2f}, {share_trigger_max:.2f}])"
        if math.isfinite(share_trigger_a) and math.isfinite(share_trigger_b)
        else "- teacher share_trigger: unavailable"
    )
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"- inferred quickgate mode: `{quickgate_mode_inferred}`")
    lines.append(f"- c0 phase_reset_source_applied==none: `{c0_phase_none}` (A={phase_a}, B={phase_b})")
    lines.append(f"- c1 policy: `{c1_policy}`")
    lines.append(f"- c1 gate mode: `{c1_gate_mode}`")
    lines.append(f"- c1 trigger_delta_tol(deg): `{float(trigger_delta_tol):.4f}`")
    lines.append(f"- c1 trigger_delta_tol source: `{trigger_delta_tol_source}`")
    lines.append(
        f"- c1 gate delta source: armA=`{trig_a_source}` armB=`{trig_b_source}` "
        f"(A={trig_a:.6f}, B={trig_b:.6f})"
        if math.isfinite(trig_a) and math.isfinite(trig_b)
        else f"- c1 gate delta source: armA=`{trig_a_source}` armB=`{trig_b_source}` (A={trig_a}, B={trig_b})"
    )
    lines.append(f"- c1 trigger_branch delta not worse: `{c1_branch_not_worse}`")
    if c2_policy == "ignore" and c2_under_not_worse_raw is None:
        lines.append(
            "- c2 under-correct (teacher) not worse: "
            f"`{c2_under_not_worse_raw}` (policy=ignore -> effective `{c2_under_not_worse}`)"
        )
    else:
        lines.append(f"- c2 under-correct (teacher) not worse: `{c2_under_not_worse}`")
    lines.append(f"- c3 new max not worse: `{c3_max_not_worse}`")
    lines.append(f"- decision legacy quickgate_4_4_2 (c0+c1+c2+c3): `{decision_legacy}`")
    lines.append(f"- decision hard_gate_v2 (c0+c2+c3): `{decision_hard_gate_v2}`")
    lines.append(
        f"- monitoring only: c1=`{c1_branch_not_worse}`, branch_changed=`{branch_changed}`, "
        f"robust_rows(A/B)=`{robust_rows_a_n}/{robust_rows_b_n}`, "
        f"share_trigger_in_range(A/B)=`{share_trigger_a_in_range}/{share_trigger_b_in_range}`"
    )
    lines.append(f"- train_r05 c1/c3 stoploss triggered: `{train_r05_stoploss_triggered}`")
    lines.append(f"- train_r05 c1/c3 stoploss triggered (legacy): `{train_r05_stoploss_triggered_legacy}`")
    lines.append(f"- recommended_next: `{recommended_next}`")
    lines.append("")
    lines.append(f"- machine summary: `{out_json}`")
    out_md = out_root / "freerun_ab_gate.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


def _resolve_resume_ckpt(args: argparse.Namespace) -> Path:
    mode = str(args.resume_from).strip().lower()
    if mode == "stage6":
        return _resolve_from_root(args.stage6_ckpt)
    if mode == "l2":
        return _resolve_from_root(args.l2_ckpt)
    if mode == "custom":
        raw = str(args.resume_ckpt or "").strip()
        if not raw:
            raise SystemExit("[FATAL] --resume-from custom requires --resume-ckpt")
        return _resolve_from_root(raw)
    raise SystemExit(f"[FATAL] unsupported --resume-from: {mode}")


def _node_defaults(node: str) -> Dict[str, Any]:
    if node == "n0":
        return {
            "cases": "r05",
            "seeds": "0,1,2",
            "epochs": 3,
            "base_run_name": "loss_budget_r0",
            "extra": ["--r05-under-mode", "off", "--r05-under-weights", "1.0"],
        }
    if node == "n1":
        return {
            "cases": "r05",
            "seeds": "0,1,2",
            "epochs": 3,
            "base_run_name": "loss_budget_r1",
            "extra": ["--r05-under-mode", "twist_only", "--r05-under-weights", "1.5,2.0,3.0"],
        }
    if node in ("n1leg", "n1-leg", "n1_leg"):
        return {
            "cases": "r2",
            "seeds": "0,1,2",
            "epochs": 3,
            "base_run_name": "loss_budget_n1_leg",
            "extra": [
                "--r2-under-mode",
                "twist_only",
                "--r2-under-weight",
                "2.0",
                "--r2-budget-lambda-trigger",
                "1.0",
                "--r2-budget-lambda-chain",
                "0.6",
                "--r2-budget-lambda-guard",
                "0.35",
                "--r2-budget-chain-joints",
                "thigh_r,calf_r,ball_r",
                "--r2-budget-chain-frame-mode",
                "trigger",
                "--r2-budget-guard-frame-mode",
                "non_trigger",
            ],
        }
    if node == "n1b":
        return {
            "cases": "g0",
            "seeds": "0,1,2",
            "epochs": 1,
            "base_run_name": "loss_budget_g0",
            "extra": [
                "--g0-tau-phase",
                "0.05",
                "--g0-tau-contact",
                "0.05",
                "--g0-tau-twist-deg",
                "5.0",
            ],
        }
    if node == "n2":
        return {
            "cases": "r2",
            "seeds": "0,1,2",
            "epochs": 3,
            "base_run_name": "loss_budget_r2",
            "extra": [
                "--r2-under-mode",
                "twist_only",
                "--r2-under-weight",
                "2.0",
                "--r2-budget-lambda-trigger",
                "1.0",
                "--r2-budget-lambda-chain",
                "0.45",
                "--r2-budget-lambda-guard",
                "0.45",
                "--r2-budget-chain-joints",
                "thigh_r,calf_r,ball_r",
                "--r2-budget-chain-frame-mode",
                "trigger",
                "--r2-budget-guard-frame-mode",
                "non_trigger",
            ],
        }
    if node == "n3":
        return {
            "cases": "r2",
            "seeds": "0,1,2",
            "epochs": 3,
            "base_run_name": "loss_budget_r3",
            "extra": [
                "--r2-under-mode",
                "twist_only",
                "--r2-under-weight",
                "2.0",
                "--r2-budget-lambda-trigger",
                "1.0",
                "--r2-budget-lambda-chain",
                "0.3",
                "--r2-budget-lambda-guard",
                "0.45",
                "--r2-budget-chain-joints",
                "thigh_r,calf_r,ball_r",
                "--r2-budget-chain-frame-mode",
                "all",
                "--r2-budget-guard-frame-mode",
                "non_trigger",
            ],
        }
    raise SystemExit(f"[FATAL] unsupported node: {node}")


def _run_mainline(args: argparse.Namespace) -> None:
    if not _RUN_LOSS_BUDGET.is_file():
        raise SystemExit(f"[FATAL] missing helper script: {_RUN_LOSS_BUDGET}")

    node = str(args.node).strip().lower()
    defaults = _node_defaults(node)

    cfg = _resolve_from_root(args.config_json)
    if not cfg.is_file():
        raise SystemExit(f"[FATAL] missing config json: {cfg}")

    resume_ckpt = _resolve_resume_ckpt(args)
    if not args.dry_run and not resume_ckpt.is_file():
        raise SystemExit(f"[FATAL] missing resume ckpt: {resume_ckpt}")

    run_tag = str(args.run_tag).strip()
    if not run_tag:
        run_tag = f"posttrain_stage67_transition_{datetime.now().strftime('%Y%m%d')}"

    out_dir = (
        _resolve_from_root(args.out_dir)
        if str(args.out_dir).strip()
        else (_ROOT / "debug_output" / f"_{run_tag}" / node)
    )
    out_model_dir = (
        _resolve_from_root(args.out_model_dir)
        if str(args.out_model_dir).strip()
        else (_ROOT / "models" / f"MLPL2_DirectBranch_v1__{run_tag}_{node}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model_dir.mkdir(parents=True, exist_ok=True)

    seeds = str(args.seeds).strip() if str(args.seeds).strip() else str(defaults["seeds"])
    epochs = int(args.epochs) if args.epochs is not None else int(defaults["epochs"])

    train_overrides = [
        f"w_direct_pose_trigger_twist={float(args.w_direct_pose_trigger_twist):.8g}",
        f"w_direct_pose_trigger_swing_x={float(args.w_direct_pose_trigger_swing_x):.8g}",
    ]
    for x in args.train_config_override or []:
        txt = str(x).strip()
        if txt:
            train_overrides.append(txt)
    # Lock N-line direct-pose semantics to absolute to avoid default drift.
    train_overrides.append("direct_pose_fusion_direct_mode=absolute")

    cmd: List[str] = []
    _add_common_loss_budget_args(
        cmd,
        config_json=cfg,
        resume_ckpt=resume_ckpt,
        out_dir=out_dir,
        out_model_dir=out_model_dir,
        cases=str(defaults["cases"]),
        seeds=seeds,
        epochs=epochs,
        dataset_index_mode=str(args.dataset_index_mode),
        base_run_name=str(defaults["base_run_name"]),
        train_overrides=train_overrides,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )
    cmd.extend(list(defaults["extra"]))

    print(f"[INFO] run node={node} resume_from={args.resume_from} resume_ckpt={resume_ckpt}")
    _run(cmd, dry_run=bool(args.dry_run))
    if not args.dry_run:
        print(f"[OK] expected summary: {out_dir / 'summary.json'}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Stage6/Stage7 transition helper (quickgate + N-line templates).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    default_cfg = "config/exp_phase_DirectBranch_v1_d1_noreset.json"
    default_stage6 = "models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage6_direct_cond_anchor_20260124.pth"
    default_l2 = "models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage7_legomega_baseline_alignproj_min0p5_20260128.pth"

    ap_q = sub.add_parser("quickgate", help="Run L1/L2 fast gate A/B (doc 4.4).")
    ap_q.add_argument("--config-json", type=str, default=default_cfg)
    ap_q.add_argument("--stage6-ckpt", type=str, default=default_stage6)
    ap_q.add_argument("--l2-ckpt", type=str, default=default_l2)
    ap_q.add_argument("--run-tag", type=str, default=f"stage67_l12_quickgate_{datetime.now().strftime('%Y%m%d')}")
    ap_q.add_argument("--out-root", type=str, default="", help="Output debug root. Empty -> debug_output/_<run_tag>")
    ap_q.add_argument("--model-root", type=str, default="", help="Output model root prefix. Empty -> models/MLPL2_DirectBranch_v1__<run_tag>")
    ap_q.add_argument("--seed", type=int, default=0)
    ap_q.add_argument("--epochs", type=int, default=1)
    ap_q.add_argument("--dataset-index-mode", type=str, default="sic_balanced")
    ap_q.add_argument("--r05-under-mode", type=str, default="twist_only", choices=("off", "twist_only", "twist_swing"))
    ap_q.add_argument("--under-weight", type=float, default=2.0)
    ap_q.add_argument("--w-direct-pose-trigger-twist", type=float, default=0.5)
    ap_q.add_argument("--w-direct-pose-trigger-swing-x", type=float, default=0.15)
    ap_q.add_argument(
        "--quickgate-mode",
        type=str,
        default="raw_ckpt_ab",
        choices=("train_r05", "raw_ckpt_ab"),
        help=(
            "raw_ckpt_ab: recommended contract-check (default); "
            "train_r05: optional fine-tune quickgate, stop-loss to Stage6->N1-leg if c1/c3 keeps failing."
        ),
    )
    ap_q.add_argument("--train-config-override", action="append", default=[])
    ap_q.add_argument("--skip-existing", action="store_true")
    ap_q.add_argument("--dry-run", action="store_true")

    ap_n = sub.add_parser(
        "mainline",
        help="Run N0/N1-leg/N1b/N2/N3 templates (N1 retained as trigger-only control).",
    )
    ap_n.add_argument(
        "--node",
        type=str,
        required=True,
        choices=("n0", "n1leg", "n1-leg", "n1_leg", "n1b", "n2", "n3", "n1"),
    )
    ap_n.add_argument("--config-json", type=str, default=default_cfg)
    ap_n.add_argument("--resume-from", type=str, default="stage6", choices=("stage6", "l2", "custom"))
    ap_n.add_argument("--resume-ckpt", type=str, default="", help="Used when --resume-from custom")
    ap_n.add_argument("--stage6-ckpt", type=str, default=default_stage6)
    ap_n.add_argument("--l2-ckpt", type=str, default=default_l2)
    ap_n.add_argument("--run-tag", type=str, default=f"posttrain_stage67_transition_{datetime.now().strftime('%Y%m%d')}")
    ap_n.add_argument("--out-dir", type=str, default="", help="Debug output dir. Empty -> debug_output/_<run_tag>/<node>")
    ap_n.add_argument("--out-model-dir", type=str, default="", help="Model output dir. Empty -> models/MLPL2_DirectBranch_v1__<run_tag>_<node>")
    ap_n.add_argument("--dataset-index-mode", type=str, default="sic_balanced")
    ap_n.add_argument("--seeds", type=str, default="", help="Override seeds list, e.g. 0,1,2")
    ap_n.add_argument("--epochs", type=int, default=None, help="Override epochs")
    ap_n.add_argument("--w-direct-pose-trigger-twist", type=float, default=0.5)
    ap_n.add_argument("--w-direct-pose-trigger-swing-x", type=float, default=0.15)
    ap_n.add_argument("--train-config-override", action="append", default=[])
    ap_n.add_argument("--skip-existing", action="store_true")
    ap_n.add_argument("--dry-run", action="store_true")

    ap_f = sub.add_parser("freerun-ab", help="Run freerun A/B and evaluate doc 4.4.2 gates.")
    ap_f.add_argument("--run-tag", type=str, default=f"stage67_l12_quickgate_{datetime.now().strftime('%Y%m%d')}")
    ap_f.add_argument("--quickgate-root", type=str, default="", help="Root with armA/armB quickgate summaries.")
    ap_f.add_argument("--out-root", type=str, default="", help="Output dir. Empty -> <quickgate-root>/freerun_ab")
    ap_f.add_argument("--arm-a-summary", type=str, default="", help="ArmA summary.json override.")
    ap_f.add_argument("--arm-b-summary", type=str, default="", help="ArmB summary.json override.")
    ap_f.add_argument("--arm-a-ckpt", type=str, default="", help="ArmA ckpt override (skip summary lookup).")
    ap_f.add_argument("--arm-b-ckpt", type=str, default="", help="ArmB ckpt override (skip summary lookup).")
    ap_f.add_argument("--seed", type=int, default=0, help="Seed row to pick from summary.")
    ap_f.add_argument("--teacher", type=str, default="validate/teacher_batches/Walk_F_teacher.json")
    ap_f.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap_f.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap_f.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap_f.add_argument("--rounds", type=int, default=5)
    ap_f.add_argument("--depth", type=int, default=3)
    ap_f.add_argument("--time-index-mode", type=str, default="auto", choices=("auto", "global", "cycle", "none"))
    ap_f.add_argument("--phase-reset-source", type=str, default="none", choices=("contacts_meas", "ttc_gt", "td_hazard", "none"))
    ap_f.add_argument("--phase-reset-source-strict", type=str, default="on", choices=("on", "off"))
    ap_f.add_argument("--lambda-fusion-apply", dest="lambda_fusion_apply", action="store_true")
    ap_f.add_argument("--no-lambda-fusion-apply", dest="lambda_fusion_apply", action="store_false")
    ap_f.set_defaults(lambda_fusion_apply=True)
    ap_f.add_argument(
        "--direct-pose-meas-source",
        type=str,
        default="model",
        choices=("model", "whitebox", "gt", "softgt", "zero"),
        help="Pass-through to freerun eval: override direct head contacts_meas source.",
    )
    ap_f.add_argument(
        "--contacts-meas-source",
        type=str,
        default="model",
        choices=("model", "whitebox", "gt", "zero"),
        help="Pass-through to freerun eval: override runtime contacts_meas source for model/event-clock.",
    )
    ap_f.add_argument(
        "--direct-pose-fusion-direct-mode",
        type=str,
        default="absolute",
        choices=("", "absolute", "residual_rot6d", "residual_compose_stable"),
        help="Override direct output interpretation in lambda fusion for freerun eval.",
    )
    ap_f.add_argument("--cycle-gte", type=int, default=1)
    ap_f.add_argument("--drop-wrap", type=int, default=1, choices=(0, 1))
    ap_f.add_argument("--topn", type=int, default=25)
    ap_f.add_argument("--target-joint", type=str, default="foot_r")
    ap_f.add_argument("--spike-sics", type=str, default="47,48,49,50,51,59,60,61,65,66,67,69,76,77,78,79,80")
    ap_f.add_argument("--gate-min-n", type=int, default=30)
    ap_f.add_argument("--contact-stance-thr", type=float, default=0.55)
    ap_f.add_argument("--contact-flight-thr", type=float, default=0.20)
    ap_f.add_argument("--contact-dom-margin", type=float, default=0.05)
    ap_f.add_argument(
        "--c1-policy",
        type=str,
        default="robust_only",
        choices=("selected", "robust_only"),
        help="How to source trigger-branch delta for c1 gate (default robust_only).",
    )
    ap_f.add_argument(
        "--c1-gate-mode",
        type=str,
        default="selector_fix_tau7",
        choices=("selector_fix_tau7", "strict0"),
        help=(
            "selector_fix_tau7: Stage67 selector/gate fix preset (use c1 tol=7deg). "
            "strict0: legacy strict c1 gate (tol=0deg)."
        ),
    )
    ap_f.add_argument(
        "--trigger-delta-tol",
        type=float,
        default=None,
        help="Override c1 trigger-branch tolerance in deg. If unset, follow --c1-gate-mode preset.",
    )
    ap_f.add_argument("--under-tol", type=float, default=0.0)
    ap_f.add_argument("--max-tol", type=float, default=0.0)
    ap_f.add_argument(
        "--share-trigger-min",
        type=float,
        default=0.15,
        help="Monitoring range lower bound for teacher direct_pose_budget_share_trigger.",
    )
    ap_f.add_argument(
        "--share-trigger-max",
        type=float,
        default=0.85,
        help="Monitoring range upper bound for teacher direct_pose_budget_share_trigger.",
    )
    ap_f.add_argument(
        "--c2-policy",
        type=str,
        default="require",
        choices=("require", "ignore"),
        help="How to handle missing teacher c2 under-correct metric in gate decision.",
    )
    ap_f.add_argument("--dry-run", action="store_true")

    return ap


def main() -> None:
    args = _build_parser().parse_args()
    if str(args.mode) == "quickgate":
        _run_quickgate(args)
        return
    if str(args.mode) == "mainline":
        _run_mainline(args)
        return
    if str(args.mode) == "freerun-ab":
        _run_freerun_ab(args)
        return
    raise SystemExit(f"[FATAL] unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
