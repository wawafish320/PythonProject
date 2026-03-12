#!/usr/bin/env python3
"""Foot-chain fine-grained diagnostics: twist/swing + d/dt + contact gating.

This script is intentionally error-only (no Jacobian). It reconstructs rows from
freerun diag batches and analyzes DirectGeoLocalDeg residual behavior for one
joint (default: foot_r), focusing on spike SICs inside a target phase.

Coordinate options:
- raw_parent: use raw SO(3) log-map components in parent frame
- twist_swing_local: convert residual rotvec into joint-local coordinates
  (using GT or pred reference), where:
    x/y => swing components, z => twist component
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import diagnose_cond_rho_c5_signsplit as signsplit
import diagnose_cond_rho_delta as base
import diagnose_jac_rl_walkf_phase_bins as phasebins
from train.geometry import rot6d_to_matrix, so3_log_map


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


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
            raise SystemExit(f"[FATAL] invalid integer token: {s}")
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _parse_lower_csv(spec: str) -> List[str]:
    out: List[str] = []
    for tok in str(spec or "").split(","):
        s = tok.strip().lower()
        if s:
            out.append(s)
    return out


def _summary(vals: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def _residual_rotvec_components(
    *,
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    coord_mode: str,
    twist_ref: str,
) -> Dict[str, torch.Tensor]:
    """Build residual rotvec components for diagnostics.

    Returns:
        dict with:
            phi_parent_deg: (B,T,3) raw residual rotvec in parent frame
            comp_deg: (B,T,3) diagnostic components
                     - raw_parent: same as phi_parent_deg
                     - twist_swing_local: joint-local components
                       (x/y swing, z twist)
    """
    if R_pred.shape != R_gt.shape or R_pred.shape[-2:] != (3, 3):
        raise SystemExit(f"[FATAL] invalid residual rot input shape: pred={tuple(R_pred.shape)} gt={tuple(R_gt.shape)}")

    # Residual rotation in parent frame.
    R_rel = torch.matmul(R_pred, R_gt.transpose(-1, -2))
    phi_parent = so3_log_map(R_rel) * (180.0 / math.pi)

    mode = str(coord_mode or "twist_swing_local").strip().lower()
    if mode == "raw_parent":
        return {"phi_parent_deg": phi_parent, "comp_deg": phi_parent}

    if mode != "twist_swing_local":
        raise SystemExit(f"[FATAL] unsupported coord_mode: {coord_mode}")

    ref = str(twist_ref or "gt").strip().lower()
    if ref == "gt":
        R_ref = R_gt
    elif ref == "pred":
        R_ref = R_pred
    else:
        raise SystemExit(f"[FATAL] unsupported twist_ref: {twist_ref}")

    # Express residual rotvec in joint-local coordinates.
    phi_local = torch.matmul(R_ref.transpose(-1, -2), phi_parent.unsqueeze(-1)).squeeze(-1)
    return {"phi_parent_deg": phi_parent, "comp_deg": phi_local}


def _phase_label(
    l: float,
    r: float,
    *,
    stance_thr: float,
    flight_thr: float,
    dominance_margin: float,
) -> str:
    if not (math.isfinite(l) and math.isfinite(r)):
        return "phase_invalid"
    if l >= stance_thr and r >= stance_thr:
        return "phase_double_support"
    if l <= flight_thr and r <= flight_thr:
        return "phase_flight"
    if (l - r) > dominance_margin:
        return "phase_left_stance"
    if (r - l) > dominance_margin:
        return "phase_right_stance"
    return "phase_transition"


def _subset_axis_stats(rvx: np.ndarray, rvy: np.ndarray, rvz: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    m = mask.astype(bool, copy=False)
    x = rvx[m]
    y = rvy[m]
    z = rvz[m]
    dom = Counter()
    for a, b, c in zip(x, y, z):
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
            continue
        idx = int(np.argmax(np.abs([a, b, c])))
        dom[["x", "y", "z"][idx]] += 1
    out = {
        "signed": {
            "x": _summary(x),
            "y": _summary(y),
            "z": _summary(z),
        },
        "abs": {
            "x": _summary(np.abs(x)),
            "y": _summary(np.abs(y)),
            "z": _summary(np.abs(z)),
        },
        "pos_rate": {
            "x": float(np.mean(x > 0.0)) if x.size > 0 else float("nan"),
            "y": float(np.mean(y > 0.0)) if y.size > 0 else float("nan"),
            "z": float(np.mean(z > 0.0)) if z.size > 0 else float("nan"),
        },
        "dominant_abs_axis": {k: int(v) for k, v in dom.items()},
    }
    return out


def _subset_contact_gating(
    *,
    err_deg: np.ndarray,
    rvz: np.ndarray,
    d_err: np.ndarray,
    right_c: np.ndarray,
    mask: np.ndarray,
    stance_thr: float,
    flight_thr: float,
) -> Dict[str, Any]:
    m = mask.astype(bool, copy=False)
    rc = right_c[m]
    e = err_deg[m]
    z = rvz[m]
    de = d_err[m]

    bins = {
        "r_contact_low": rc <= float(flight_thr),
        "r_contact_mid": (rc > float(flight_thr)) & (rc < float(stance_thr)),
        "r_contact_high": rc >= float(stance_thr),
    }

    out: Dict[str, Any] = {}
    for k, bm in bins.items():
        if not np.any(bm):
            out[k] = {
                "n": 0,
                "err_deg": _summary(np.asarray([], dtype=np.float64)),
                "rvz_signed": _summary(np.asarray([], dtype=np.float64)),
                "rvz_abs": _summary(np.asarray([], dtype=np.float64)),
                "d_err": _summary(np.asarray([], dtype=np.float64)),
            }
            continue
        out[k] = {
            "n": int(np.sum(bm)),
            "err_deg": _summary(e[bm]),
            "rvz_signed": _summary(z[bm]),
            "rvz_abs": _summary(np.abs(z[bm])),
            "d_err": _summary(de[bm]),
        }
    return out


def _subset_summary(
    *,
    name: str,
    mask: np.ndarray,
    err_deg: np.ndarray,
    rvx: np.ndarray,
    rvy: np.ndarray,
    rvz: np.ndarray,
    d_err: np.ndarray,
    d_rvx: np.ndarray,
    d_rvy: np.ndarray,
    d_rvz: np.ndarray,
    right_c: np.ndarray,
    speed: np.ndarray,
    raw_rvx: np.ndarray,
    raw_rvy: np.ndarray,
    raw_rvz: np.ndarray,
    stance_thr: float,
    flight_thr: float,
) -> Dict[str, Any]:
    m = mask.astype(bool, copy=False)
    swing_norm = np.sqrt(np.square(rvx[m]) + np.square(rvy[m]))
    twist_abs = np.abs(rvz[m])
    return {
        "name": name,
        "n_rows": int(np.sum(m)),
        "err_deg": _summary(err_deg[m]),
        "axis": _subset_axis_stats(rvx, rvy, rvz, m),
        "swing_norm": _summary(swing_norm),
        "twist_abs": _summary(twist_abs),
        "raw_parent_axis": _subset_axis_stats(raw_rvx, raw_rvy, raw_rvz, m),
        "d_err": _summary(d_err[m]),
        "d_axis": {
            "x": _summary(d_rvx[m]),
            "y": _summary(d_rvy[m]),
            "z": _summary(d_rvz[m]),
        },
        "speed": _summary(speed[m]),
        "right_contact": _summary(right_c[m]),
        "contact_gating": _subset_contact_gating(
            err_deg=err_deg,
            rvz=rvz,
            d_err=d_err,
            right_c=right_c,
            mask=m,
            stance_thr=stance_thr,
            flight_thr=flight_thr,
        ),
    }


def _branch_summary(
    *,
    mask: np.ndarray,
    err_deg: np.ndarray,
    rvx: np.ndarray,
    rvy: np.ndarray,
    rvz: np.ndarray,
    d_err: np.ndarray,
    right_c: np.ndarray,
) -> Dict[str, Any]:
    m = mask.astype(bool, copy=False)
    return {
        "n": int(np.sum(m)),
        "err_deg": _summary(err_deg[m]),
        "abs_rvx": _summary(np.abs(rvx[m])),
        "abs_rvy": _summary(np.abs(rvy[m])),
        "abs_rvz": _summary(np.abs(rvz[m])),
        "rvz_signed": _summary(rvz[m]),
        "d_err": _summary(d_err[m]),
        "right_contact": _summary(right_c[m]),
    }


def _to_markdown(out: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Foot_r Twist/Swing + Derivative + Contact Gating")
    lines.append("")
    lines.append(f"- source rho json: `{out.get('source_rho_json', '')}`")
    lines.append(f"- model: `{out.get('model', '')}`")
    lines.append(f"- target joint: `{out.get('target_joint', '')}`")
    lines.append(f"- target phase: `{out.get('target_phase', '')}`")
    lines.append(f"- spike SICs: `{out.get('spike_sics', [])}`")
    coord = out.get("coordinate_system", {}) if isinstance(out.get("coordinate_system"), dict) else {}
    if coord:
        lines.append(
            f"- coord_mode: `{coord.get('mode', 'NA')}`, twist_ref: `{coord.get('twist_ref', 'NA')}` "
            f"(x=`{coord.get('x', 'NA')}`, y=`{coord.get('y', 'NA')}`, z=`{coord.get('z', 'NA')}`)"
        )
    lines.append("")

    split = out.get("split", {}) if isinstance(out.get("split"), dict) else {}
    lines.append("## Split")
    lines.append("")
    lines.append(
        f"- total rows: `{int(split.get('n_total', 0))}`, phase rows: `{int(split.get('n_phase', 0))}`, "
        f"spike rows: `{int(split.get('n_spike', 0))}`, control rows: `{int(split.get('n_control', 0))}`"
    )
    trig_n = _safe_float(out.get("direct_pose_trigger_n", float("nan")))
    trig_n_txt = str(int(trig_n)) if math.isfinite(trig_n) else "NA"
    lines.append(
        f"- trigger rows: `{trig_n_txt}`, "
        f"under_correct_frac_trigger_twist: `{_safe_float(out.get('under_correct_frac_trigger_twist', float('nan'))):.4f}`"
    )
    lines.append("")

    cmpo = out.get("comparison", {}) if isinstance(out.get("comparison"), dict) else {}
    lines.append("## Spike vs Control")
    lines.append("")
    lines.append("|metric|spike|control|delta|")
    lines.append("|:--|--:|--:|--:|")
    for k in [
        "err_deg_mean",
        "swing_norm_mean",
        "twist_abs_mean",
        "abs_rvx_mean",
        "abs_rvy_mean",
        "abs_rvz_mean",
        "d_err_mean",
        "d_rvx_mean",
        "d_rvy_mean",
        "d_rvz_mean",
        "right_contact_mean",
    ]:
        row = cmpo.get(k, {}) if isinstance(cmpo.get(k), dict) else {}
        lines.append(
            f"|`{k}`|{_safe_float(row.get('spike', float('nan'))):.4f}|"
            f"{_safe_float(row.get('control', float('nan'))):.4f}|"
            f"{_safe_float(row.get('delta', float('nan'))):.4f}|"
        )
    lines.append("")

    gating = out.get("gating_delta", {}) if isinstance(out.get("gating_delta"), dict) else {}
    if gating:
        lines.append("## Contact Gating Delta (spike - control)")
        lines.append("")
        lines.append("|gate|delta_err_deg|delta_twist_z_signed|delta_twist_z_abs|delta_d_err|")
        lines.append("|:--|--:|--:|--:|--:|")
        for g in ["r_contact_low", "r_contact_mid", "r_contact_high"]:
            row = gating.get(g, {}) if isinstance(gating.get(g), dict) else {}
            lines.append(
                f"|`{g}`|{_safe_float(row.get('delta_err_deg', float('nan'))):.4f}|"
                f"{_safe_float(row.get('delta_rvz_signed', float('nan'))):.4f}|"
                f"{_safe_float(row.get('delta_abs_rvz', float('nan'))):.4f}|"
                f"{_safe_float(row.get('delta_d_err', float('nan'))):.4f}|"
            )
        lines.append("")

    branch = (
        out.get("twist_sign_contact_branch", {})
        if isinstance(out.get("twist_sign_contact_branch"), dict)
        else {}
    )
    branch_rows = branch.get("rows", []) if isinstance(branch.get("rows"), list) else []
    if branch_rows:
        lines.append("## Twist-Sign x Contact Branch Delta (spike - control)")
        lines.append("")
        lines.append("|branch|n_spike|n_control|delta_err_deg|delta_abs_swing_x|delta_abs_swing_y|delta_abs_twist_z|delta_d_err|")
        lines.append("|:--|--:|--:|--:|--:|--:|--:|--:|")
        for r in branch_rows:
            lines.append(
                f"|`{r.get('branch', '')}`|{int(r.get('n_spike', 0))}|{int(r.get('n_control', 0))}|"
                f"{_safe_float(r.get('delta_err_deg', float('nan'))):.4f}|"
                f"{_safe_float(r.get('delta_abs_rvx', float('nan'))):.4f}|"
                f"{_safe_float(r.get('delta_abs_rvy', float('nan'))):.4f}|"
                f"{_safe_float(r.get('delta_abs_rvz', float('nan'))):.4f}|"
                f"{_safe_float(r.get('delta_d_err', float('nan'))):.4f}|"
            )
        lines.append("")

    top = out.get("spike_sic_table", []) if isinstance(out.get("spike_sic_table"), list) else []
    if top:
        lines.append("## Spike SIC Detail")
        lines.append("")
        lines.append("|sic|rows|err_deg|swing_x|swing_y|twist_z|d_err|right_contact|")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in top:
            lines.append(
                f"|{int(r.get('sic', -1))}|{int(r.get('n_rows', 0))}|{_safe_float(r.get('err_deg_mean', float('nan'))):.4f}|"
                f"{_safe_float(r.get('rvx_mean', float('nan'))):.4f}|{_safe_float(r.get('rvy_mean', float('nan'))):.4f}|"
                f"{_safe_float(r.get('rvz_mean', float('nan'))):.4f}|{_safe_float(r.get('d_err_mean', float('nan'))):.4f}|"
                f"{_safe_float(r.get('right_contact_mean', float('nan'))):.4f}|"
            )
        lines.append("")

    concl = out.get("conclusion", {}) if isinstance(out.get("conclusion"), dict) else {}
    if concl:
        lines.append("## Conclusion")
        lines.append("")
        lines.append(f"- trigger_axis: `{concl.get('trigger_axis', 'NA')}`")
        lines.append(f"- trigger_gate: `{concl.get('trigger_gate', 'NA')}`")
        lines.append(f"- trigger_branch: `{concl.get('trigger_branch', 'NA')}`")
        lines.append(f"- trigger_branch_low_contact: `{concl.get('trigger_branch_low_contact', 'NA')}`")
        lines.append(f"- note: `{concl.get('note', 'NA')}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Foot_r twist/swing + derivative + contact gating diagnostics")
    ap.add_argument("--rho-json", type=str, required=True)
    ap.add_argument("--teacher", type=str, default="validate/teacher_batches/Walk_F_teacher.json")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--data-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--device", type=str, default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--context-len", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--steps", type=str, default="all")
    ap.add_argument("--target-joint", type=str, default="foot_r")
    ap.add_argument("--target-phase", type=str, default="phase_left_stance")
    ap.add_argument(
        "--coord-mode",
        type=str,
        default="twist_swing_local",
        choices=("twist_swing_local", "raw_parent"),
        help="Residual coordinate mode for x/y/z components.",
    )
    ap.add_argument(
        "--twist-ref",
        type=str,
        default="gt",
        choices=("gt", "pred"),
        help="Reference orientation used when coord-mode=twist_swing_local.",
    )
    ap.add_argument(
        "--spike-sics",
        type=str,
        default="47,48,49,50,51,59,60,61,65,66,67,69,76,77,78,79,80",
    )
    ap.add_argument("--contact-stance-thr", type=float, default=0.55)
    ap.add_argument("--contact-flight-thr", type=float, default=0.20)
    ap.add_argument("--contact-dom-margin", type=float, default=0.05)
    ap.add_argument("--gate-min-n", type=int, default=30, help="Min rows per group for robust gate selection")
    ap.add_argument("--clip-ids", type=str, default="")
    ap.add_argument("--clip-name-contains", type=str, default="Walk_F")
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--out-md", type=str, default="")
    args = ap.parse_args()

    rho_path = Path(args.rho_json).expanduser().resolve()
    rho_payload = json.loads(rho_path.read_text(encoding="utf-8"))
    model_path = str(rho_payload.get("model", "")).strip()
    diag_pts = [str(x) for x in (rho_payload.get("diag_pts", []) or [])]
    if not model_path:
        raise SystemExit("[FATAL] model missing in rho-json")
    if not diag_pts:
        raise SystemExit("[FATAL] diag_pts missing in rho-json")

    runner_args = argparse.Namespace(**vars(args))
    runner_args.model = model_path
    runner, ds_one, _ = base._load_runner(runner_args)
    model = runner.model
    trainer = runner.trainer
    if model is None or trainer is None:
        raise SystemExit("[FATAL] failed to load model/trainer")

    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        raise SystemExit("[FATAL] trainer rot6d slice missing")

    dy = int(getattr(ds_one, "Y", np.zeros((1, 1), dtype=np.float32)).shape[-1])
    st = int(rot_slice.start or 0)
    ed = int(rot_slice.stop or dy)
    if ed <= st or ((ed - st) % 6) != 0:
        raise SystemExit(f"[FATAL] invalid rot6d slice [{st}:{ed}]")
    joint_count = (ed - st) // 6

    bone_names = list(getattr(ds_one, "bone_names", []) or [])[: int(joint_count)]
    name_to_idx = {str(n): i for i, n in enumerate(bone_names)}
    target_joint = str(args.target_joint)
    if target_joint not in name_to_idx:
        raise SystemExit(f"[FATAL] target joint not found: {target_joint}")
    jidx = int(name_to_idx[target_joint])

    spike_sics = _parse_int_csv(args.spike_sics)
    spike_set = set(spike_sics)

    clip_ids_filter = set(_parse_int_csv(str(args.clip_ids or "")))
    clip_name_filter = _parse_lower_csv(str(args.clip_name_contains or ""))
    apply_clip_filter = bool(clip_ids_filter or clip_name_filter)
    clip_path_by_id: Dict[int, str] = {}
    if apply_clip_filter:
        clip_path_by_id = phasebins._build_clip_path_by_id(
            seq_len=int(args.seq_len),
            data_root=Path(args.data_root).expanduser().resolve(),
            bundle=Path(args.bundle).expanduser().resolve(),
            pretrain_template=Path(args.pretrain_template).expanduser().resolve(),
        )

    def _keep_clip(cid: int) -> bool:
        if not apply_clip_filter:
            return True
        if clip_ids_filter and cid not in clip_ids_filter:
            return False
        if clip_name_filter:
            p = str(clip_path_by_id.get(int(cid), "")).lower()
            if not p:
                return False
            if not any(tok in p for tok in clip_name_filter):
                return False
        return True

    sic_list: List[int] = []
    phase_list: List[str] = []
    err_list: List[float] = []
    rvx_list: List[float] = []
    rvy_list: List[float] = []
    rvz_list: List[float] = []
    raw_rvx_list: List[float] = []
    raw_rvy_list: List[float] = []
    raw_rvz_list: List[float] = []
    d_err_list: List[float] = []
    d_rvx_list: List[float] = []
    d_rvy_list: List[float] = []
    d_rvz_list: List[float] = []
    right_c_list: List[float] = []
    speed_list: List[float] = []
    twist_pred_list: List[float] = []
    twist_gt_list: List[float] = []

    for dp in diag_pts:
        diag_pt = Path(dp).expanduser().resolve()
        if not diag_pt.is_file():
            raise SystemExit(f"[FATAL] diag pt missing: {diag_pt}")

        clip_ids = phasebins._diag_clip_ids(diag_pt)
        keep_rows = [i for i, cid in enumerate(clip_ids) if _keep_clip(int(cid))]
        if apply_clip_filter and not keep_rows:
            continue

        batch = base._rebuild_batch_from_diag(
            diag_pt=diag_pt,
            seq_len=int(args.seq_len),
            data_root=Path(args.data_root).expanduser().resolve(),
            bundle=Path(args.bundle).expanduser(),
            pretrain_template=Path(args.pretrain_template).expanduser(),
            device=runner.device,
        )
        if apply_clip_filter and len(keep_rows) < len(clip_ids):
            batch = phasebins._slice_batch_rows(batch, keep_rows)

        state = batch.get("motion")
        cond = batch.get("cond_in")
        gt_motion = batch.get("gt_motion")
        contacts = batch.get("contacts")
        start = batch.get("start")
        clip_len = batch.get("clip_len")
        angvel = batch.get("angvel")
        pose_hist = batch.get("pose_hist")

        if not (torch.is_tensor(state) and torch.is_tensor(cond) and torch.is_tensor(gt_motion)):
            raise SystemExit("[FATAL] missing motion/cond_in/gt_motion")
        if not torch.is_tensor(start):
            raise SystemExit("[FATAL] missing start")
        if not torch.is_tensor(clip_len):
            clip_len = torch.full_like(start, int(state.shape[1]))

        model.eval()
        with torch.no_grad():
            use_learned_meas = bool(getattr(model, "contact_meas_enable", False)) and getattr(model, "contact_meas_head", None) is not None
            contacts_in = None if use_learned_meas else contacts
            ret = model(
                state,
                cond=cond,
                contacts=contacts_in,
                angvel=angvel,
                pose_history=pose_hist,
                plan_z=None,
                time_index=None,
            )
            out_direct = ret.get("out_direct") if isinstance(ret, dict) else None
            if not (torch.is_tensor(out_direct) and out_direct.dim() == 3):
                raise SystemExit("[FATAL] invalid out_direct")

            B = int(out_direct.shape[0])
            T = int(out_direct.shape[1])
            steps = base._parse_steps(str(args.steps or "all"), T)
            if not steps:
                raise SystemExit("[FATAL] no valid steps")

            pred6 = out_direct[..., rot_slice].reshape(B, T, joint_count, 6)
            gt6 = gt_motion[..., rot_slice].reshape(B, T, joint_count, 6)
            pred_R = rot6d_to_matrix(pred6)
            gt_R = rot6d_to_matrix(gt6)

            Rp = pred_R[:, :, jidx]  # (B,T,3,3)
            Rg = gt_R[:, :, jidx]
            comp = _residual_rotvec_components(
                R_pred=Rp,
                R_gt=Rg,
                coord_mode=str(args.coord_mode),
                twist_ref=str(args.twist_ref),
            )
            phi_parent = comp["phi_parent_deg"]  # raw parent-frame residual axis-angle
            phi = comp["comp_deg"]  # diagnostic components (x/y/z)
            err = torch.norm(phi, dim=-1)
            pred_local_deg = so3_log_map(Rp) * (180.0 / math.pi)
            gt_local_deg = so3_log_map(Rg) * (180.0 / math.pi)
            twist_pred_deg = pred_local_deg[..., 2]
            twist_gt_deg = gt_local_deg[..., 2]

            d_phi = torch.full_like(phi, float("nan"))
            d_err = torch.full_like(err, float("nan"))
            if T > 1:
                d_phi[:, :-1, :] = phi[:, 1:, :] - phi[:, :-1, :]
                d_err[:, :-1] = err[:, 1:] - err[:, :-1]
                # Use backward difference for the tail step to avoid systematic NaN at t=T-1.
                d_phi[:, -1, :] = phi[:, -1, :] - phi[:, -2, :]
                d_err[:, -1] = err[:, -1] - err[:, -2]

            cond_raw = signsplit._cond_raw_for_label(batch, label_time="t")
            if cond_raw.dim() != 3:
                raise SystemExit("[FATAL] cond_raw invalid")
            cdim = int(cond_raw.shape[-1])
            adim = int(cdim - 3)
            if adim < 0:
                raise SystemExit(f"[FATAL] invalid cond dim: {cdim}")
            speed = cond_raw[..., adim + 2]

            if not (torch.is_tensor(contacts) and contacts.dim() == 3 and int(contacts.shape[-1]) >= 2):
                raise SystemExit("[FATAL] contacts missing for phase/gating")

            contacts_np = contacts[..., :2].detach().cpu().numpy().astype(np.float64, copy=False)
            phi_np = phi.detach().cpu().numpy().astype(np.float64, copy=False)
            phi_parent_np = phi_parent.detach().cpu().numpy().astype(np.float64, copy=False)
            err_np = err.detach().cpu().numpy().astype(np.float64, copy=False)
            d_phi_np = d_phi.detach().cpu().numpy().astype(np.float64, copy=False)
            d_err_np = d_err.detach().cpu().numpy().astype(np.float64, copy=False)
            speed_np = speed.detach().cpu().numpy().astype(np.float64, copy=False)
            twist_pred_np = twist_pred_deg.detach().cpu().numpy().astype(np.float64, copy=False)
            twist_gt_np = twist_gt_deg.detach().cpu().numpy().astype(np.float64, copy=False)

            start_np = start.view(-1).detach().cpu().numpy().astype(np.int64, copy=False)
            clip_len_np = clip_len.view(-1).detach().cpu().numpy().astype(np.int64, copy=False)

            for b in range(B):
                clip_n = int(clip_len_np[b]) if int(clip_len_np[b]) > 0 else int(T)
                for t in steps:
                    ti = int(t)
                    sic_abs = int((int(start_np[b]) + ti) % max(1, clip_n))
                    cl = float(contacts_np[b, ti, 0])
                    cr = float(contacts_np[b, ti, 1])
                    phase = _phase_label(
                        cl,
                        cr,
                        stance_thr=float(args.contact_stance_thr),
                        flight_thr=float(args.contact_flight_thr),
                        dominance_margin=float(args.contact_dom_margin),
                    )

                    sic_list.append(int(sic_abs))
                    phase_list.append(phase)
                    err_list.append(float(err_np[b, ti]))
                    rvx_list.append(float(phi_np[b, ti, 0]))
                    rvy_list.append(float(phi_np[b, ti, 1]))
                    rvz_list.append(float(phi_np[b, ti, 2]))
                    raw_rvx_list.append(float(phi_parent_np[b, ti, 0]))
                    raw_rvy_list.append(float(phi_parent_np[b, ti, 1]))
                    raw_rvz_list.append(float(phi_parent_np[b, ti, 2]))
                    d_err_list.append(float(d_err_np[b, ti]))
                    d_rvx_list.append(float(d_phi_np[b, ti, 0]))
                    d_rvy_list.append(float(d_phi_np[b, ti, 1]))
                    d_rvz_list.append(float(d_phi_np[b, ti, 2]))
                    right_c_list.append(float(cr))
                    speed_list.append(float(speed_np[b, ti]))
                    twist_pred_list.append(float(twist_pred_np[b, ti]))
                    twist_gt_list.append(float(twist_gt_np[b, ti]))

    if not sic_list:
        raise SystemExit("[FATAL] empty aggregation")

    sic = np.asarray(sic_list, dtype=np.int64)
    phase = np.asarray(phase_list, dtype=object)
    err_deg = np.asarray(err_list, dtype=np.float64)
    rvx = np.asarray(rvx_list, dtype=np.float64)
    rvy = np.asarray(rvy_list, dtype=np.float64)
    rvz = np.asarray(rvz_list, dtype=np.float64)
    raw_rvx = np.asarray(raw_rvx_list, dtype=np.float64)
    raw_rvy = np.asarray(raw_rvy_list, dtype=np.float64)
    raw_rvz = np.asarray(raw_rvz_list, dtype=np.float64)
    d_err = np.asarray(d_err_list, dtype=np.float64)
    d_rvx = np.asarray(d_rvx_list, dtype=np.float64)
    d_rvy = np.asarray(d_rvy_list, dtype=np.float64)
    d_rvz = np.asarray(d_rvz_list, dtype=np.float64)
    right_c = np.asarray(right_c_list, dtype=np.float64)
    speed = np.asarray(speed_list, dtype=np.float64)
    twist_pred = np.asarray(twist_pred_list, dtype=np.float64)
    twist_gt = np.asarray(twist_gt_list, dtype=np.float64)

    target_phase = str(args.target_phase)
    m_phase = phase == target_phase
    m_spike = m_phase & np.isin(sic, np.asarray(sorted(list(spike_set)), dtype=np.int64))
    m_control = m_phase & (~np.isin(sic, np.asarray(sorted(list(spike_set)), dtype=np.int64)))
    i_trigger = m_phase & (right_c <= float(args.contact_flight_thr)) & np.isfinite(twist_gt) & (twist_gt > 0.0)
    sign_match = np.sign(twist_pred) == np.sign(twist_gt)
    is_under = (np.abs(twist_pred) < np.abs(twist_gt)) & sign_match
    is_over = (np.abs(twist_pred) > np.abs(twist_gt)) & sign_match
    trigger_n = int(np.sum(i_trigger))
    if trigger_n > 0:
        under_correct_frac_trigger_twist = float(np.mean(is_under[i_trigger]))
        over_correct_frac_trigger_twist = float(np.mean(is_over[i_trigger]))
    else:
        under_correct_frac_trigger_twist = None
        over_correct_frac_trigger_twist = None
    direct_pose_trigger_gate_weight_mean = float(np.mean(i_trigger.astype(np.float64))) if err_deg.size > 0 else None
    gate_masks = {
        "r_contact_low": right_c <= float(args.contact_flight_thr),
        "r_contact_mid": (right_c > float(args.contact_flight_thr)) & (right_c < float(args.contact_stance_thr)),
        "r_contact_high": right_c >= float(args.contact_stance_thr),
    }
    twist_sign_masks = {
        "twist_neg": rvz < 0.0,
        "twist_pos": rvz > 0.0,
        "twist_zero": rvz == 0.0,
    }

    spike = _subset_summary(
        name="spike",
        mask=m_spike,
        err_deg=err_deg,
        rvx=rvx,
        rvy=rvy,
        rvz=rvz,
        d_err=d_err,
        d_rvx=d_rvx,
        d_rvy=d_rvy,
        d_rvz=d_rvz,
        right_c=right_c,
        speed=speed,
        raw_rvx=raw_rvx,
        raw_rvy=raw_rvy,
        raw_rvz=raw_rvz,
        stance_thr=float(args.contact_stance_thr),
        flight_thr=float(args.contact_flight_thr),
    )
    control = _subset_summary(
        name="control",
        mask=m_control,
        err_deg=err_deg,
        rvx=rvx,
        rvy=rvy,
        rvz=rvz,
        d_err=d_err,
        d_rvx=d_rvx,
        d_rvy=d_rvy,
        d_rvz=d_rvz,
        right_c=right_c,
        speed=speed,
        raw_rvx=raw_rvx,
        raw_rvy=raw_rvy,
        raw_rvz=raw_rvz,
        stance_thr=float(args.contact_stance_thr),
        flight_thr=float(args.contact_flight_thr),
    )

    def _delta_pair(sv: float, cv: float) -> Dict[str, float]:
        s = _safe_float(sv)
        c = _safe_float(cv)
        return {
            "spike": s,
            "control": c,
            "delta": float(s - c) if math.isfinite(s) and math.isfinite(c) else float("nan"),
        }

    cmpo = {
        "err_deg_mean": _delta_pair((spike.get("err_deg", {}) or {}).get("mean"), (control.get("err_deg", {}) or {}).get("mean")),
        "swing_norm_mean": _delta_pair((spike.get("swing_norm", {}) or {}).get("mean"), (control.get("swing_norm", {}) or {}).get("mean")),
        "twist_abs_mean": _delta_pair((spike.get("twist_abs", {}) or {}).get("mean"), (control.get("twist_abs", {}) or {}).get("mean")),
        "abs_rvx_mean": _delta_pair((spike.get("axis", {}).get("abs", {}).get("x", {}) or {}).get("mean"), (control.get("axis", {}).get("abs", {}).get("x", {}) or {}).get("mean")),
        "abs_rvy_mean": _delta_pair((spike.get("axis", {}).get("abs", {}).get("y", {}) or {}).get("mean"), (control.get("axis", {}).get("abs", {}).get("y", {}) or {}).get("mean")),
        "abs_rvz_mean": _delta_pair((spike.get("axis", {}).get("abs", {}).get("z", {}) or {}).get("mean"), (control.get("axis", {}).get("abs", {}).get("z", {}) or {}).get("mean")),
        "d_err_mean": _delta_pair((spike.get("d_err", {}) or {}).get("mean"), (control.get("d_err", {}) or {}).get("mean")),
        "d_rvx_mean": _delta_pair((spike.get("d_axis", {}).get("x", {}) or {}).get("mean"), (control.get("d_axis", {}).get("x", {}) or {}).get("mean")),
        "d_rvy_mean": _delta_pair((spike.get("d_axis", {}).get("y", {}) or {}).get("mean"), (control.get("d_axis", {}).get("y", {}) or {}).get("mean")),
        "d_rvz_mean": _delta_pair((spike.get("d_axis", {}).get("z", {}) or {}).get("mean"), (control.get("d_axis", {}).get("z", {}) or {}).get("mean")),
        "right_contact_mean": _delta_pair((spike.get("right_contact", {}) or {}).get("mean"), (control.get("right_contact", {}) or {}).get("mean")),
    }

    gating_delta: Dict[str, Any] = {}
    for gk in ["r_contact_low", "r_contact_mid", "r_contact_high"]:
        gs = (spike.get("contact_gating", {}) or {}).get(gk, {})
        gc = (control.get("contact_gating", {}) or {}).get(gk, {})
        s_err = _safe_float((gs.get("err_deg", {}) or {}).get("mean", float("nan")))
        c_err = _safe_float((gc.get("err_deg", {}) or {}).get("mean", float("nan")))
        s_rvz_signed = _safe_float((gs.get("rvz_signed", {}) or {}).get("mean", float("nan")))
        c_rvz_signed = _safe_float((gc.get("rvz_signed", {}) or {}).get("mean", float("nan")))
        s_rvz_abs = _safe_float((gs.get("rvz_abs", {}) or {}).get("mean", float("nan")))
        c_rvz_abs = _safe_float((gc.get("rvz_abs", {}) or {}).get("mean", float("nan")))
        s_de = _safe_float((gs.get("d_err", {}) or {}).get("mean", float("nan")))
        c_de = _safe_float((gc.get("d_err", {}) or {}).get("mean", float("nan")))
        gating_delta[gk] = {
            "n_spike": int(gs.get("n", 0) or 0),
            "n_control": int(gc.get("n", 0) or 0),
            "delta_err_deg": float(s_err - c_err) if math.isfinite(s_err) and math.isfinite(c_err) else float("nan"),
            "delta_rvz_signed": float(s_rvz_signed - c_rvz_signed) if math.isfinite(s_rvz_signed) and math.isfinite(c_rvz_signed) else float("nan"),
            "delta_abs_rvz": float(s_rvz_abs - c_rvz_abs) if math.isfinite(s_rvz_abs) and math.isfinite(c_rvz_abs) else float("nan"),
            "delta_d_err": float(s_de - c_de) if math.isfinite(s_de) and math.isfinite(c_de) else float("nan"),
        }

    branch_rows: List[Dict[str, Any]] = []
    branch_detail: Dict[str, Any] = {}
    for sign_key in ["twist_neg", "twist_pos", "twist_zero"]:
        sm = twist_sign_masks[sign_key]
        for gate_key_i in ["r_contact_low", "r_contact_mid", "r_contact_high"]:
            gm = gate_masks[gate_key_i]
            br = f"{sign_key}__{gate_key_i}"
            ms = m_spike & sm & gm
            mc = m_control & sm & gm
            ssum = _branch_summary(
                mask=ms,
                err_deg=err_deg,
                rvx=rvx,
                rvy=rvy,
                rvz=rvz,
                d_err=d_err,
                right_c=right_c,
            )
            csum = _branch_summary(
                mask=mc,
                err_deg=err_deg,
                rvx=rvx,
                rvy=rvy,
                rvz=rvz,
                d_err=d_err,
                right_c=right_c,
            )

            s_err = _safe_float((ssum.get("err_deg", {}) or {}).get("mean", float("nan")))
            c_err = _safe_float((csum.get("err_deg", {}) or {}).get("mean", float("nan")))
            s_ax = _safe_float((ssum.get("abs_rvx", {}) or {}).get("mean", float("nan")))
            c_ax = _safe_float((csum.get("abs_rvx", {}) or {}).get("mean", float("nan")))
            s_ay = _safe_float((ssum.get("abs_rvy", {}) or {}).get("mean", float("nan")))
            c_ay = _safe_float((csum.get("abs_rvy", {}) or {}).get("mean", float("nan")))
            s_az = _safe_float((ssum.get("abs_rvz", {}) or {}).get("mean", float("nan")))
            c_az = _safe_float((csum.get("abs_rvz", {}) or {}).get("mean", float("nan")))
            s_de = _safe_float((ssum.get("d_err", {}) or {}).get("mean", float("nan")))
            c_de = _safe_float((csum.get("d_err", {}) or {}).get("mean", float("nan")))

            row = {
                "branch": br,
                "twist_sign": sign_key,
                "contact_gate": gate_key_i,
                "n_spike": int(ssum.get("n", 0) or 0),
                "n_control": int(csum.get("n", 0) or 0),
                "delta_err_deg": float(s_err - c_err) if math.isfinite(s_err) and math.isfinite(c_err) else float("nan"),
                "delta_abs_rvx": float(s_ax - c_ax) if math.isfinite(s_ax) and math.isfinite(c_ax) else float("nan"),
                "delta_abs_rvy": float(s_ay - c_ay) if math.isfinite(s_ay) and math.isfinite(c_ay) else float("nan"),
                "delta_abs_rvz": float(s_az - c_az) if math.isfinite(s_az) and math.isfinite(c_az) else float("nan"),
                "delta_d_err": float(s_de - c_de) if math.isfinite(s_de) and math.isfinite(c_de) else float("nan"),
            }
            branch_rows.append(row)
            branch_detail[br] = {"spike": ssum, "control": csum, "delta": row}

    def _abs_finite_delta(row: Dict[str, Any]) -> float:
        v = _safe_float(row.get("delta_err_deg", float("nan")))
        return abs(v) if math.isfinite(v) else -1.0

    branch_rows = sorted(branch_rows, key=_abs_finite_delta, reverse=True)

    # SIC detail table for spikes
    spike_rows: List[Dict[str, Any]] = []
    for s in sorted(list(spike_set)):
        ms = m_spike & (sic == int(s))
        if not np.any(ms):
            continue
        spike_rows.append(
            {
                "sic": int(s),
                "n_rows": int(np.sum(ms)),
                "err_deg_mean": _safe_float(np.mean(err_deg[ms])),
                "rvx_mean": _safe_float(np.mean(rvx[ms])),
                "rvy_mean": _safe_float(np.mean(rvy[ms])),
                "rvz_mean": _safe_float(np.mean(rvz[ms])),
                "d_err_mean": _safe_float(np.mean(d_err[ms])),
                "right_contact_mean": _safe_float(np.mean(right_c[ms])),
            }
        )

    # infer trigger axis and gating
    abs_axis_delta = {
        "x": _safe_float((cmpo.get("abs_rvx_mean", {}) or {}).get("delta", float("nan"))),
        "y": _safe_float((cmpo.get("abs_rvy_mean", {}) or {}).get("delta", float("nan"))),
        "z": _safe_float((cmpo.get("abs_rvz_mean", {}) or {}).get("delta", float("nan"))),
    }
    trigger_axis = max(abs_axis_delta, key=lambda k: abs(_safe_float(abs_axis_delta[k])))

    gate_candidates: List[str] = []
    for g in ["r_contact_low", "r_contact_mid", "r_contact_high"]:
        row = gating_delta.get(g, {}) or {}
        if int(row.get("n_spike", 0) or 0) < int(args.gate_min_n):
            continue
        if int(row.get("n_control", 0) or 0) < int(args.gate_min_n):
            continue
        if not math.isfinite(_safe_float(row.get("delta_err_deg", float("nan")))):
            continue
        gate_candidates.append(g)
    if gate_candidates:
        gate_key = max(
            gate_candidates,
            key=lambda g: abs(_safe_float((gating_delta.get(g, {}) or {}).get("delta_err_deg", float("nan")))),
        )
    else:
        gate_key = max(
            ["r_contact_low", "r_contact_mid", "r_contact_high"],
            key=lambda g: abs(_safe_float((gating_delta.get(g, {}) or {}).get("delta_err_deg", float("nan")))),
        )

    robust_branch_rows = [
        r
        for r in branch_rows
        if int(r.get("n_spike", 0)) >= int(args.gate_min_n)
        and int(r.get("n_control", 0)) >= int(args.gate_min_n)
        and math.isfinite(_safe_float(r.get("delta_err_deg", float("nan"))))
    ]
    if robust_branch_rows:
        trigger_branch = str(robust_branch_rows[0].get("branch", "NA"))
    elif branch_rows:
        trigger_branch = str(branch_rows[0].get("branch", "NA"))
    else:
        trigger_branch = "NA"

    low_contact_rows = [
        r
        for r in branch_rows
        if str(r.get("contact_gate", "")) == "r_contact_low"
        and int(r.get("n_spike", 0)) >= int(args.gate_min_n)
        and int(r.get("n_control", 0)) >= int(args.gate_min_n)
        and math.isfinite(_safe_float(r.get("delta_err_deg", float("nan"))))
    ]
    trigger_branch_low_contact = str(low_contact_rows[0].get("branch", "NA")) if low_contact_rows else "NA"

    note = (
        f"abs-axis delta: x={abs_axis_delta['x']:.4f}, y={abs_axis_delta['y']:.4f}, z={abs_axis_delta['z']:.4f}; "
        f"coord_mode={str(args.coord_mode)}, twist_ref={str(args.twist_ref)}; "
        f"max gate delta_err_deg at {gate_key} (gate_min_n={int(args.gate_min_n)}); "
        f"trigger_branch={trigger_branch}, low_contact_branch={trigger_branch_low_contact}"
    )

    out = {
        "source_rho_json": str(rho_path),
        "model": model_path,
        "diag_pts": [str(Path(x).expanduser().resolve()) for x in diag_pts],
        "target_joint": target_joint,
        "target_phase": target_phase,
        "spike_sics": [int(x) for x in sorted(list(spike_set))],
        "contact_phase_config": {
            "stance_thr": float(args.contact_stance_thr),
            "flight_thr": float(args.contact_flight_thr),
            "dominance_margin": float(args.contact_dom_margin),
        },
        "coordinate_system": {
            "mode": str(args.coord_mode),
            "twist_ref": str(args.twist_ref),
            "x": "swing_x_local_deg" if str(args.coord_mode) == "twist_swing_local" else "raw_parent_x_deg",
            "y": "swing_y_local_deg" if str(args.coord_mode) == "twist_swing_local" else "raw_parent_y_deg",
            "z": "twist_z_local_deg" if str(args.coord_mode) == "twist_swing_local" else "raw_parent_z_deg",
            "raw_parent_reference": "phi_parent_deg = log(R_pred * R_gt^T) * rad2deg",
        },
        "split": {
            "n_total": int(err_deg.size),
            "n_phase": int(np.sum(m_phase)),
            "n_spike": int(np.sum(m_spike)),
            "n_control": int(np.sum(m_control)),
        },
        "under_correct_frac_trigger_twist": under_correct_frac_trigger_twist,
        "under_correct_frac_trigger_twist_hard": under_correct_frac_trigger_twist,
        "over_correct_frac_trigger_twist": over_correct_frac_trigger_twist,
        "direct_pose_trigger_gate_weight_mean": direct_pose_trigger_gate_weight_mean,
        "direct_pose_trigger_n": float(trigger_n),
        "direct_pose_trigger_frac": float(trigger_n / max(1, int(err_deg.size))),
        "spike": spike,
        "control": control,
        "comparison": cmpo,
        "gating_delta": gating_delta,
        "twist_sign_contact_branch": {
            "gate_min_n": int(args.gate_min_n),
            "rows": branch_rows,
            "detail": branch_detail,
        },
        # Keep backward-compatible alias for existing downstream readers.
        "rvz_sign_contact_branch": {
            "gate_min_n": int(args.gate_min_n),
            "rows": branch_rows,
            "detail": branch_detail,
        },
        "spike_sic_table": spike_rows,
        "conclusion": {
            "trigger_axis": trigger_axis,
            "trigger_gate": gate_key,
            "trigger_branch": trigger_branch,
            "trigger_branch_low_contact": trigger_branch_low_contact,
            "note": note,
        },
    }

    out_json = (
        Path(args.out_json).expanduser().resolve()
        if str(args.out_json).strip()
        else rho_path.with_name(rho_path.stem + "_footr_twist_swing_gate.json")
    )
    out_md = Path(args.out_md).expanduser().resolve() if str(args.out_md).strip() else out_json.with_suffix(".md")

    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(out), encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
