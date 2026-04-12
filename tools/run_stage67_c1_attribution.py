#!/usr/bin/env python3
"""
Stage67 next-iteration helper: C1 attribution probes (P0~P4, eval-only).

Doc reference:
  docs/changes/2026-02-20_stage67_c1_pcgrad_closeout_and_next_iteration.md
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_ROOT = Path(__file__).resolve().parents[1]
_STAGE67 = _ROOT / "tools" / "run_stage67_transition.py"
_DEFAULT_SPIKE_SICS = "47,48,49,50,51,59,60,61,65,66,67,69,76,77,78,79,80"


def _resolve_from_root(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (_ROOT / p)


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _mean(vals: Sequence[Any]) -> float:
    arr = [float(v) for v in (_safe_float(x) for x in vals) if math.isfinite(float(v))]
    if not arr:
        return float("nan")
    return float(sum(arr) / len(arr))


def _quantile(vals: Sequence[Any], q: float) -> float:
    arr = sorted(float(v) for v in (_safe_float(x) for x in vals) if math.isfinite(float(v)))
    if not arr:
        return float("nan")
    qq = max(0.0, min(1.0, float(q)))
    idx = int(round(qq * (len(arr) - 1)))
    idx = max(0, min(len(arr) - 1, idx))
    return float(arr[idx])


def _parse_int_list(csv: str) -> List[int]:
    vals: List[int] = []
    for tok in str(csv or "").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            vals.append(int(t))
        except Exception:
            raise SystemExit(f"[FATAL] invalid int token: {t!r}")
    if not vals:
        raise SystemExit("[FATAL] empty int list")
    return vals


def _parse_float_list(csv: str) -> List[float]:
    vals: List[float] = []
    for tok in str(csv or "").split(","):
        t = tok.strip()
        if not t:
            continue
        x = _safe_float(t)
        if math.isfinite(x):
            vals.append(float(x))
    if not vals:
        raise SystemExit("[FATAL] empty float list")
    return vals


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("[cmd] " + " ".join(str(x) for x in cmd))
    if dry_run:
        return
    rc = subprocess.call([str(x) for x in cmd], cwd=str(_ROOT))
    if int(rc) != 0:
        raise SystemExit(f"[FATAL] command failed (exit={rc})")


def _build_step_mask(steps: Sequence[Mapping[str, Any]], *, cycle_gte: int, drop_wrap: bool) -> List[bool]:
    out: List[bool] = []
    for ent in steps:
        cyc = int(ent.get("cycle", 0) or 0)
        if cyc < int(cycle_gte):
            out.append(False)
            continue
        if bool(drop_wrap) and bool(ent.get("wrap_boundary_step", False)):
            out.append(False)
            continue
        out.append(True)
    return out


def _extract_direct_series(obj: Mapping[str, Any]) -> Tuple[List[str], int, List[List[float]]]:
    per = obj.get("per_step_direct_geolocal_deg", {})
    if not isinstance(per, Mapping):
        raise SystemExit("[FATAL] missing per_step_direct_geolocal_deg")
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
            "trigger_branch": "NA",
            "trigger_branch_delta_err_deg": float("nan"),
            "rows": [],
            "robust_rows": [],
        }
    rot_names, rotvec = rot
    idx_map = {str(n): int(i) for i, n in enumerate(names)}
    rot_idx_map = {str(n): int(i) for i, n in enumerate(rot_names)}
    if str(target_joint) not in idx_map or str(target_joint) not in rot_idx_map:
        return {
            "status": "target_joint_missing",
            "trigger_branch": "NA",
            "trigger_branch_delta_err_deg": float("nan"),
            "rows": [],
            "robust_rows": [],
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
        recs.append(
            {
                "sic": int(ent.get("step_in_cycle", 0) or 0),
                "phase": _phase_label(
                    float(l),
                    float(r),
                    stance_thr=float(contact_stance_thr),
                    flight_thr=float(contact_flight_thr),
                    dom_margin=float(contact_dom_margin),
                ),
                "err_deg": float(err),
                "rvz": float(rvz),
                "right_contact": float(r),
            }
        )

    if not recs:
        return {
            "status": "empty_records",
            "trigger_branch": "NA",
            "trigger_branch_delta_err_deg": float("nan"),
            "rows": [],
            "robust_rows": [],
        }

    spike_set = set(int(x) for x in spike_sics)
    rows: List[Dict[str, Any]] = []
    for sign_key in ("twist_neg", "twist_pos", "twist_zero"):
        for gate_key in ("r_contact_low", "r_contact_mid", "r_contact_high"):
            spike_vals: List[float] = []
            ctrl_vals: List[float] = []
            for r in recs:
                if str(r["phase"]) != "phase_left_stance":
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
    return {
        "status": "ok",
        "trigger_branch": str(top.get("branch", "NA")) if isinstance(top, Mapping) else "NA",
        "trigger_branch_delta_err_deg": _safe_float(top.get("delta_err_deg", float("nan"))) if isinstance(top, Mapping) else float("nan"),
        "rows": rows,
        "robust_rows": robust,
    }


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


def _compare_le(a: float, b: float, tol: float) -> Optional[bool]:
    aa = _safe_float(a)
    bb = _safe_float(b)
    tt = _safe_float(tol)
    if not (math.isfinite(aa) and math.isfinite(bb) and math.isfinite(tt)):
        return None
    return bool(aa <= bb + tt)


def _collect_target_joint_delta_rows(
    *,
    obj_a: Mapping[str, Any],
    obj_b: Mapping[str, Any],
    target_joint: str,
    cycle_gte: int,
    drop_wrap: bool,
) -> List[Dict[str, Any]]:
    steps_a = obj_a.get("metrics_per_step", [])
    steps_b = obj_b.get("metrics_per_step", [])
    if not isinstance(steps_a, list) or not isinstance(steps_b, list):
        return []
    names_a, _, mat_a = _extract_direct_series(obj_a)
    names_b, _, mat_b = _extract_direct_series(obj_b)
    idx_a_map = {str(n): i for i, n in enumerate(names_a)}
    idx_b_map = {str(n): i for i, n in enumerate(names_b)}
    if str(target_joint) not in idx_a_map or str(target_joint) not in idx_b_map:
        return []
    j_a = int(idx_a_map[str(target_joint)])
    j_b = int(idx_b_map[str(target_joint)])
    mask_a = _build_step_mask(steps_a, cycle_gte=cycle_gte, drop_wrap=drop_wrap)
    mask_b = _build_step_mask(steps_b, cycle_gte=cycle_gte, drop_wrap=drop_wrap)
    n = min(len(steps_a), len(steps_b), len(mat_a), len(mat_b), len(mask_a), len(mask_b))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        if not (bool(mask_a[i]) and bool(mask_b[i])):
            continue
        row_a = mat_a[i]
        row_b = mat_b[i]
        if j_a >= len(row_a) or j_b >= len(row_b):
            continue
        ea = _safe_float(row_a[j_a])
        eb = _safe_float(row_b[j_b])
        if not (math.isfinite(ea) and math.isfinite(eb)):
            continue
        step_a = steps_a[i]
        cp = _safe_contact_pair(step_a)
        out.append(
            {
                "step": int(step_a.get("step", i) or i),
                "cycle": int(step_a.get("cycle", 0) or 0),
                "step_in_cycle": int(step_a.get("step_in_cycle", 0) or 0),
                "err_a_deg": float(ea),
                "err_b_deg": float(eb),
                "delta_err_deg": float(ea - eb),
                "right_contact": float(cp[1]) if cp is not None else float("nan"),
            }
        )
    return out


def _summarize_distribution(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    vals = [_safe_float(r.get("delta_err_deg", float("nan"))) for r in rows]
    vals_f = [float(v) for v in vals if math.isfinite(v)]
    if not vals_f:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "frac_positive": float("nan"),
            "frac_gt_3deg": float("nan"),
            "frac_gt_7deg": float("nan"),
            "frac_gt_10deg": float("nan"),
        }
    n = len(vals_f)
    frac_pos = float(sum(1 for v in vals_f if v > 0.0) / n)
    return {
        "n": int(n),
        "mean": _mean(vals_f),
        "median": _quantile(vals_f, 0.5),
        "p75": _quantile(vals_f, 0.75),
        "p90": _quantile(vals_f, 0.90),
        "p95": _quantile(vals_f, 0.95),
        "p99": _quantile(vals_f, 0.99),
        "min": float(min(vals_f)),
        "max": float(max(vals_f)),
        "frac_positive": frac_pos,
        "frac_gt_3deg": float(sum(1 for v in vals_f if v > 3.0) / n),
        "frac_gt_7deg": float(sum(1 for v in vals_f if v > 7.0) / n),
        "frac_gt_10deg": float(sum(1 for v in vals_f if v > 10.0) / n),
    }


def _topk_sics_by_abs_mean(rows: Sequence[Mapping[str, Any]], k: int) -> List[int]:
    bucket: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        sic = int(r.get("step_in_cycle", -1) or -1)
        dlt = _safe_float(r.get("delta_err_deg", float("nan")))
        if sic < 0 or not math.isfinite(dlt):
            continue
        bucket[sic].append(abs(float(dlt)))
    scored: List[Tuple[float, int]] = []
    for sic, vals in bucket.items():
        m = _mean(vals)
        if math.isfinite(m):
            scored.append((float(m), int(sic)))
    scored.sort(reverse=True)
    return [int(sic) for _, sic in scored[: max(1, int(k))]]


def _evaluate_c1_probe(
    *,
    obj_a: Mapping[str, Any],
    obj_b: Mapping[str, Any],
    target_joint: str,
    spike_sics: Sequence[int],
    gate_min_n: int,
    trigger_delta_tol: float,
    c1_policy: str,
    cycle_gte: int,
    drop_wrap: bool,
    contact_stance_thr: float,
    contact_flight_thr: float,
    contact_dom_margin: float,
) -> Dict[str, Any]:
    steps_a = obj_a.get("metrics_per_step", [])
    steps_b = obj_b.get("metrics_per_step", [])
    if not isinstance(steps_a, list) or not isinstance(steps_b, list) or not steps_a or not steps_b:
        return {"c1": None, "error": "invalid_metrics_per_step"}

    names_a, _, mat_a = _extract_direct_series(obj_a)
    names_b, _, mat_b = _extract_direct_series(obj_b)
    mask_a = _build_step_mask(steps_a, cycle_gte=cycle_gte, drop_wrap=drop_wrap)
    mask_b = _build_step_mask(steps_b, cycle_gte=cycle_gte, drop_wrap=drop_wrap)

    branch_a = _compute_trigger_branch_diag(
        obj=obj_a,
        steps=steps_a,
        names=names_a,
        mat=mat_a,
        mask=mask_a,
        target_joint=target_joint,
        spike_sics=spike_sics,
        gate_min_n=gate_min_n,
        contact_stance_thr=contact_stance_thr,
        contact_flight_thr=contact_flight_thr,
        contact_dom_margin=contact_dom_margin,
    )
    branch_b = _compute_trigger_branch_diag(
        obj=obj_b,
        steps=steps_b,
        names=names_b,
        mat=mat_b,
        mask=mask_b,
        target_joint=target_joint,
        spike_sics=spike_sics,
        gate_min_n=gate_min_n,
        contact_stance_thr=contact_stance_thr,
        contact_flight_thr=contact_flight_thr,
        contact_dom_margin=contact_dom_margin,
    )
    trig_a, trig_a_source = _pick_c1_delta(branch_a, c1_policy)
    trig_b, trig_b_source = _pick_c1_delta(branch_b, c1_policy)
    c1 = _compare_le(trig_a, trig_b, trigger_delta_tol)
    robust_a_n = len(branch_a.get("robust_rows", [])) if isinstance(branch_a.get("robust_rows", []), list) else 0
    robust_b_n = len(branch_b.get("robust_rows", [])) if isinstance(branch_b.get("robust_rows", []), list) else 0
    return {
        "c1": c1,
        "trigger_delta_tol_deg": float(trigger_delta_tol),
        "arm_a_delta_err_deg": trig_a,
        "arm_b_delta_err_deg": trig_b,
        "arm_a_delta_source": trig_a_source,
        "arm_b_delta_source": trig_b_source,
        "arm_a_trigger_branch": str(branch_a.get("trigger_branch", "NA")),
        "arm_b_trigger_branch": str(branch_b.get("trigger_branch", "NA")),
        "branch_changed": str(branch_a.get("trigger_branch", "NA")) != str(branch_b.get("trigger_branch", "NA")),
        "robust_rows_count_arm_a": int(robust_a_n),
        "robust_rows_count_arm_b": int(robust_b_n),
        "spike_sics": [int(x) for x in spike_sics],
        "gate_min_n": int(gate_min_n),
    }


def _c1_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    t = 0
    f = 0
    na = 0
    for r in rows:
        v = r.get("c1", None)
        if v is True:
            t += 1
        elif v is False:
            f += 1
        else:
            na += 1
    n = t + f + na
    return {
        "true": int(t),
        "false": int(f),
        "na": int(na),
        "n": int(n),
        "true_ge_2_of_3": bool(t >= 2) if n >= 3 else False,
    }


def _classify_p4(dist: Mapping[str, Any]) -> str:
    med = _safe_float(dist.get("median", float("nan")))
    p95 = _safe_float(dist.get("p95", float("nan")))
    frac_pos = _safe_float(dist.get("frac_positive", float("nan")))
    if math.isfinite(med) and math.isfinite(frac_pos) and med > 0.25 and frac_pos > 0.60:
        return "right_shift"
    if math.isfinite(med) and math.isfinite(p95) and abs(med) <= 0.25 and p95 > 3.0:
        return "tail_only"
    return "mixed_or_flat"


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Stage67 C1 attribution probes (P0~P4).")
    ap.add_argument("--run-tag", type=str, default=f"stage67_c1_attribution_{datetime.now().strftime('%Y%m%d')}")
    ap.add_argument("--out-root", type=str, default="", help="Default: debug_output/_<run-tag>")
    ap.add_argument("--arm-a-ckpts", type=str, required=True, help="Comma list for ArmA ckpts (B e3, 3 seeds).")
    ap.add_argument(
        "--arm-b-ckpt",
        type=str,
        default="models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage6_direct_cond_anchor_20260124.pth",
    )
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--teacher", type=str, default="validate/teacher_batches/Walk_F_teacher.json")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--time-index-mode", type=str, default="auto", choices=("auto", "global", "cycle", "none"))
    ap.add_argument("--phase-reset-source", type=str, default="none", choices=("contacts_meas", "ttc_gt", "td_hazard", "none"))
    ap.add_argument("--phase-reset-source-strict", type=str, default="on", choices=("on", "off"))
    ap.add_argument("--lambda-fusion-apply", dest="lambda_fusion_apply", action="store_true")
    ap.add_argument("--no-lambda-fusion-apply", dest="lambda_fusion_apply", action="store_false")
    ap.set_defaults(lambda_fusion_apply=True)
    ap.add_argument("--direct-pose-meas-source", type=str, default="model", choices=("model", "gt", "softgt", "zero"))
    ap.add_argument("--contacts-meas-source", type=str, default="model", choices=("model", "gt", "zero"))
    ap.add_argument(
        "--direct-pose-fusion-direct-mode",
        type=str,
        default="absolute",
        choices=("absolute", "residual_rot6d", "residual_compose_stable"),
    )
    ap.add_argument("--cycle-gte", type=int, default=1)
    ap.add_argument("--drop-wrap", type=int, default=1, choices=(0, 1))
    ap.add_argument("--target-joint", type=str, default="foot_r")
    ap.add_argument("--spike-sics", type=str, default=_DEFAULT_SPIKE_SICS)
    ap.add_argument("--gate-min-n", type=int, default=30)
    ap.add_argument("--contact-stance-thr", type=float, default=0.55)
    ap.add_argument("--contact-flight-thr", type=float, default=0.20)
    ap.add_argument("--contact-dom-margin", type=float, default=0.05)
    ap.add_argument("--c1-policy", type=str, default="robust_only", choices=("selected", "robust_only"))
    ap.add_argument("--p1-taus", type=str, default="3,7,10")
    ap.add_argument("--p2-topk", type=int, default=17)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    if not _STAGE67.is_file():
        raise SystemExit(f"[FATAL] missing helper: {_STAGE67}")

    seeds = _parse_int_list(args.seeds)
    p1_taus = _parse_float_list(args.p1_taus)
    spike_sics_default = _parse_int_list(args.spike_sics)

    arm_a_ckpt_raw = [x.strip() for x in str(args.arm_a_ckpts).split(",") if x.strip()]
    if not arm_a_ckpt_raw:
        raise SystemExit("[FATAL] --arm-a-ckpts is empty")
    arm_a_ckpts = [_resolve_from_root(x) for x in arm_a_ckpt_raw]
    arm_b_ckpt = _resolve_from_root(args.arm_b_ckpt)

    if not args.dry_run:
        for p in arm_a_ckpts:
            if not p.is_file():
                raise SystemExit(f"[FATAL] arm_a ckpt not found: {p}")
        if not arm_b_ckpt.is_file():
            raise SystemExit(f"[FATAL] arm_b ckpt not found: {arm_b_ckpt}")

    if len(arm_a_ckpts) == 1:
        arm_a_by_seed = {int(s): arm_a_ckpts[0] for s in seeds}
    elif len(arm_a_ckpts) == len(seeds):
        arm_a_by_seed = {int(s): arm_a_ckpts[i] for i, s in enumerate(seeds)}
    else:
        raise SystemExit(
            f"[FATAL] arm_a ckpt count ({len(arm_a_ckpts)}) must be 1 or match seeds ({len(seeds)})."
        )

    out_root = (
        _resolve_from_root(args.out_root)
        if str(args.out_root).strip()
        else (_ROOT / "debug_output" / f"_{str(args.run_tag).strip()}")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    commands: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    p4_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        arm_a_ckpt = arm_a_by_seed[int(seed)]
        seed_root = out_root / f"seed{int(seed)}"
        p0_root = seed_root / "p0_control"
        gate_json = p0_root / "freerun_ab_gate.json"

        cmd: List[str] = [
            str(sys.executable),
            str(_STAGE67),
            "freerun-ab",
            "--out-root",
            str(p0_root),
            "--arm-a-ckpt",
            str(arm_a_ckpt),
            "--arm-b-ckpt",
            str(arm_b_ckpt),
            "--teacher",
            str(args.teacher),
            "--bundle",
            str(args.bundle),
            "--pretrain-template",
            str(args.pretrain_template),
            "--encoder-bundle",
            str(args.encoder_bundle),
            "--rounds",
            str(int(args.rounds)),
            "--depth",
            str(int(args.depth)),
            "--time-index-mode",
            str(args.time_index_mode),
            "--phase-reset-source",
            str(args.phase_reset_source),
            "--phase-reset-source-strict",
            str(args.phase_reset_source_strict),
            "--direct-pose-meas-source",
            str(args.direct_pose_meas_source),
            "--contacts-meas-source",
            str(args.contacts_meas_source),
            "--direct-pose-fusion-direct-mode",
            str(args.direct_pose_fusion_direct_mode),
            "--cycle-gte",
            str(int(args.cycle_gte)),
            "--drop-wrap",
            str(int(args.drop_wrap)),
            "--target-joint",
            str(args.target_joint),
            "--spike-sics",
            str(args.spike_sics),
            "--gate-min-n",
            str(int(args.gate_min_n)),
            "--contact-stance-thr",
            str(float(args.contact_stance_thr)),
            "--contact-flight-thr",
            str(float(args.contact_flight_thr)),
            "--contact-dom-margin",
            str(float(args.contact_dom_margin)),
            "--c1-policy",
            str(args.c1_policy),
            "--c2-policy",
            "ignore",
            "--trigger-delta-tol",
            "0.0",
        ]
        if bool(args.lambda_fusion_apply):
            cmd.append("--lambda-fusion-apply")
        else:
            cmd.append("--no-lambda-fusion-apply")
        if bool(args.dry_run):
            cmd.append("--dry-run")
        commands.append({"seed": int(seed), "probe": "P0_control", "cmd": cmd})

        if not (bool(args.skip_existing) and gate_json.is_file()):
            _run(cmd, dry_run=bool(args.dry_run))

        if args.dry_run:
            continue
        if not gate_json.is_file():
            raise SystemExit(f"[FATAL] missing gate json for seed{seed}: {gate_json}")

        gate_obj = _load_json(gate_json)
        inputs = gate_obj.get("inputs", {})
        if not isinstance(inputs, Mapping):
            raise SystemExit(f"[FATAL] invalid inputs in gate json: {gate_json}")
        arm_a_json = _resolve_from_root(str(inputs.get("arm_a_json", "")))
        arm_b_json = _resolve_from_root(str(inputs.get("arm_b_json", "")))
        if not arm_a_json.is_file() or not arm_b_json.is_file():
            raise SystemExit(f"[FATAL] missing freerun jsons: {arm_a_json} {arm_b_json}")
        obj_a = _load_json(arm_a_json)
        obj_b = _load_json(arm_b_json)

        rows_delta = _collect_target_joint_delta_rows(
            obj_a=obj_a,
            obj_b=obj_b,
            target_joint=str(args.target_joint),
            cycle_gte=int(args.cycle_gte),
            drop_wrap=bool(args.drop_wrap),
        )
        all_sics = sorted({int(r.get("step_in_cycle", -1) or -1) for r in rows_delta if int(r.get("step_in_cycle", -1) or -1) >= 0})
        topk_sics = _topk_sics_by_abs_mean(rows_delta, k=max(1, int(args.p2_topk)))
        if not topk_sics:
            topk_sics = list(spike_sics_default)
        if not all_sics:
            all_sics = list(spike_sics_default)

        probe_defs: List[Tuple[str, List[int], int, float]] = []
        probe_defs.append(("P0_control", list(spike_sics_default), int(args.gate_min_n), 0.0))
        probe_defs.append(("P2_default", list(spike_sics_default), int(args.gate_min_n), 0.0))
        probe_defs.append(("P2_all_sic", list(all_sics), int(args.gate_min_n), 0.0))
        probe_defs.append(("P2_topk_by_delta", list(topk_sics), int(args.gate_min_n), 0.0))
        probe_defs.append(("P3_robust_half_n", list(spike_sics_default), max(1, int(args.gate_min_n) // 2), 0.0))
        for tau in p1_taus:
            probe_defs.append((f"P1_tau_{str(tau).replace('.', 'p')}deg", list(spike_sics_default), int(args.gate_min_n), float(tau)))

        for probe_id, sics, gate_min_n, tol in probe_defs:
            ev = _evaluate_c1_probe(
                obj_a=obj_a,
                obj_b=obj_b,
                target_joint=str(args.target_joint),
                spike_sics=sics,
                gate_min_n=int(gate_min_n),
                trigger_delta_tol=float(tol),
                c1_policy=str(args.c1_policy),
                cycle_gte=int(args.cycle_gte),
                drop_wrap=bool(args.drop_wrap),
                contact_stance_thr=float(args.contact_stance_thr),
                contact_flight_thr=float(args.contact_flight_thr),
                contact_dom_margin=float(args.contact_dom_margin),
            )
            probe_rows.append(
                {
                    "seed": int(seed),
                    "probe": probe_id,
                    **ev,
                }
            )

        sic_default = set(int(x) for x in spike_sics_default)
        sic_topk = set(int(x) for x in topk_sics)
        rows_default = [r for r in rows_delta if int(r.get("step_in_cycle", -1) or -1) in sic_default]
        rows_topk = [r for r in rows_delta if int(r.get("step_in_cycle", -1) or -1) in sic_topk]
        dist_all = _summarize_distribution(rows_delta)
        dist_default = _summarize_distribution(rows_default)
        dist_topk = _summarize_distribution(rows_topk)
        p4_rows.append(
            {
                "seed": int(seed),
                "arm_a_json": str(arm_a_json),
                "arm_b_json": str(arm_b_json),
                "all_masked": dist_all,
                "default_sic": dist_default,
                "topk_sic": dist_topk,
                "all_sics_count": len(all_sics),
                "topk_sics": [int(x) for x in topk_sics],
                "default_sics": [int(x) for x in spike_sics_default],
                "p4_class": _classify_p4(dist_all),
            }
        )

    if args.dry_run:
        cmd_payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_tag": str(args.run_tag),
            "commands": commands,
        }
        cmd_json = out_root / "c1_attribution_command_manifest.json"
        _write_json(cmd_json, cmd_payload)
        print(f"[OK] wrote {cmd_json}")
        return

    by_probe: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in probe_rows:
        by_probe[str(r.get("probe", "NA"))].append(r)
    probe_agg: List[Dict[str, Any]] = []
    for probe, rows in sorted(by_probe.items()):
        counts = _c1_counts(rows)
        probe_agg.append(
            {
                "probe": probe,
                "counts": counts,
                "rows": rows,
            }
        )

    p4_mode_count: Dict[str, int] = defaultdict(int)
    for r in p4_rows:
        p4_mode_count[str(r.get("p4_class", "mixed_or_flat"))] += 1

    any_probe_ge2 = any(
        bool(ent.get("counts", {}).get("true_ge_2_of_3", False))
        for ent in probe_agg
        if str(ent.get("probe", "")).startswith("P")
    )
    recommended_action = (
        "prefer_selector_or_gate_fix"
        if any_probe_ge2
        else "all_probes_still_0of3_then_escalate_to_structural_treatment_n2_n3"
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": str(args.run_tag),
        "inputs": {
            "seeds": seeds,
            "arm_a_ckpts": {str(k): str(v) for k, v in arm_a_by_seed.items()},
            "arm_b_ckpt": str(arm_b_ckpt),
            "spike_sics_default": [int(x) for x in spike_sics_default],
            "gate_min_n": int(args.gate_min_n),
            "p1_taus": [float(x) for x in p1_taus],
            "p2_topk": int(args.p2_topk),
            "c1_policy": str(args.c1_policy),
            "mask": {
                "cycle_gte": int(args.cycle_gte),
                "drop_wrap": bool(args.drop_wrap),
                "exclude_root": True,
            },
        },
        "probes": {
            "rows": probe_rows,
            "aggregate": probe_agg,
        },
        "p4_distribution": {
            "rows": p4_rows,
            "mode_count": dict(p4_mode_count),
        },
        "decision": {
            "any_probe_true_ge_2_of_3": bool(any_probe_ge2),
            "recommended_action": recommended_action,
        },
    }
    out_json = out_root / "c1_attribution_summary.json"
    _write_json(out_json, summary)

    lines: List[str] = []
    lines.append("# Stage67 C1 Attribution (P0~P4)")
    lines.append("")
    lines.append(f"- run_tag: `{args.run_tag}`")
    lines.append(f"- arm_b_ckpt: `{arm_b_ckpt}`")
    lines.append(f"- seeds: `{','.join(str(s) for s in seeds)}`")
    lines.append(f"- c1_policy: `{args.c1_policy}`")
    lines.append("")
    lines.append("| probe | c1 true | c1 false | c1 na | ge2/3 |")
    lines.append("|---|---:|---:|---:|---|")
    for ent in probe_agg:
        p = str(ent.get("probe", "NA"))
        c = ent.get("counts", {})
        lines.append(
            f"| {p} | {int(c.get('true', 0))} | {int(c.get('false', 0))} | {int(c.get('na', 0))} | "
            f"{bool(c.get('true_ge_2_of_3', False))} |"
        )
    lines.append("")
    lines.append("## P4 Distribution")
    lines.append("")
    lines.append(f"- class_count: `{dict(p4_mode_count)}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- any_probe_true_ge_2_of_3: `{bool(any_probe_ge2)}`")
    lines.append(f"- recommended_action: `{recommended_action}`")
    lines.append("")
    lines.append(f"- summary_json: `{out_json}`")
    out_md = out_root / "c1_attribution_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
