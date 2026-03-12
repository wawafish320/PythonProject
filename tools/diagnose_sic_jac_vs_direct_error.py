#!/usr/bin/env python3
"""Diagnose SIC-level relation between jac_RL and direct geolocal error.

The script reconstructs batches from diag .pt files (via clip_id/start), runs the
model once per batch, then aggregates both metrics by absolute SIC:

- jac_RL = ||d y_R / d x|| / ||d y_L / d x||
  where x is the first linear input of direct_pose_head.
- direct geolocal error (deg) from out_direct vs gt_motion rot6d.

Aggregation is done on the same model and the same rows, so jac/error are directly
comparable.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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

import diagnose_cond_rho_delta as base
import diagnose_jac_rl_walkf_phase_bins as phasebins
from train.geometry import geodesic_R, rot6d_to_matrix


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    if not (math.isfinite(num) and math.isfinite(den)):
        return float("nan")
    if abs(den) <= eps:
        return float("nan")
    return float(num / den)


def _parse_int_csv(spec: str) -> Set[int]:
    out: Set[int] = set()
    for tok in str(spec or "").split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            out.add(int(s))
        except Exception:
            raise SystemExit(f"[FATAL] invalid integer token: {s}")
    return out


def _parse_lower_csv(spec: str) -> List[str]:
    out: List[str] = []
    for tok in str(spec or "").split(","):
        s = tok.strip().lower()
        if s:
            out.append(s)
    return out


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks with tie handling."""
    n = int(x.size)
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        xi = x[order[i]]
        while j < n and x[order[j]] == xi:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return float("nan")
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if den <= 1e-12:
        return float("nan")
    return float((x * y).sum() / den)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return float("nan")
    return _pearson(_rankdata_average(x), _rankdata_average(y))


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


@dataclass
class SicAccumulator:
    rows: int = 0
    batches: Set[str] = field(default_factory=set)
    left_sq_sum: float = 0.0
    right_sq_sum: float = 0.0
    jac_row_values: List[float] = field(default_factory=list)
    phase_counts: Counter = field(default_factory=Counter)
    # Direct error metrics (deg)
    err_all_joints: List[float] = field(default_factory=list)
    err_focus_mean: List[float] = field(default_factory=list)
    err_left_focus_mean: List[float] = field(default_factory=list)
    err_right_focus_mean: List[float] = field(default_factory=list)
    err_focus_rl_ratio: List[float] = field(default_factory=list)
    err_by_joint: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))


def _summarize_vals(vals: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray([_safe_float(v) for v in vals], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
    }


def _to_markdown(out: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# SIC jac_RL vs DirectGeoLocalDeg")
    lines.append("")
    lines.append(f"- source rho json: `{out.get('source_rho_json', '')}`")
    lines.append(f"- model: `{out.get('model', '')}`")
    lines.append(f"- diag batches: `{len(out.get('diag_pts', []))}`")
    lines.append(f"- steps: `{out.get('steps_used', [])}`")
    lines.append(f"- clip filter: `{out.get('clip_filter', {})}`")
    lines.append(f"- focus joints: `{out.get('focus_joints', [])}`")
    lines.append("")

    hotspots = out.get("hotspots", []) if isinstance(out.get("hotspots"), list) else []
    rows = out.get("by_sic", []) if isinstance(out.get("by_sic"), list) else []
    if hotspots and rows:
        by = {int(r.get("sic", -1)): r for r in rows}
        lines.append("## Hotspots")
        lines.append("")
        lines.append("|sic|phase_major|jac_RL|rows|batches|err_focus_mean|err_focus_R/L|")
        lines.append("|---:|:--|---:|---:|---:|---:|---:|")
        for s in hotspots:
            r = by.get(int(s))
            if r is None:
                lines.append(f"|{int(s)}|NA|nan|0|0|nan|nan|")
                continue
            lines.append(
                f"|{int(s)}|{r.get('phase_major', 'NA')}|{_safe_float(r.get('jac_rl', float('nan'))):.4f}|"
                f"{int(r.get('n_rows', 0))}|{int(r.get('n_batches', 0))}|"
                f"{_safe_float((r.get('err_focus_mean', {}) or {}).get('mean', float('nan'))):.4f}|"
                f"{_safe_float((r.get('err_focus_rl_ratio', {}) or {}).get('mean', float('nan'))):.4f}|"
            )
        lines.append("")

    corr = out.get("sic_level_correlation", {}) if isinstance(out.get("sic_level_correlation"), dict) else {}
    if corr:
        lines.append("## SIC-Level Correlation")
        lines.append("")
        lines.append("|metric|n_sic|pearson|spearman|")
        lines.append("|:--|--:|--:|--:|")
        for k, v in corr.items():
            if not isinstance(v, dict):
                continue
            lines.append(
                f"|`{k}`|{int(v.get('n_sic', 0))}|{_safe_float(v.get('pearson', float('nan'))):.4f}|"
                f"{_safe_float(v.get('spearman', float('nan'))):.4f}|"
            )
        lines.append("")

    lines.append("## Top SIC by direct focus mean")
    lines.append("")
    lines.append("|sic|phase_major|jac_RL|err_focus_mean|err_calf_l|err_calf_r|")
    lines.append("|---:|:--|---:|---:|---:|---:|")
    ranked = sorted(
        rows,
        key=lambda r: abs(_safe_float((r.get("err_focus_mean", {}) or {}).get("mean", float("nan")))),
        reverse=True,
    )
    for r in ranked[:12]:
        byj = r.get("err_by_joint", {}) if isinstance(r.get("err_by_joint"), dict) else {}
        lines.append(
            f"|{int(r.get('sic', -1))}|{r.get('phase_major', 'NA')}|{_safe_float(r.get('jac_rl', float('nan'))):.4f}|"
            f"{_safe_float((r.get('err_focus_mean', {}) or {}).get('mean', float('nan'))):.4f}|"
            f"{_safe_float((byj.get('calf_l', {}) or {}).get('mean', float('nan'))):.4f}|"
            f"{_safe_float((byj.get('calf_r', {}) or {}).get('mean', float('nan'))):.4f}|"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose SIC jac_RL vs direct geolocal error on identical rows.")
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
    ap.add_argument("--steps", type=str, default="", help="Step list, e.g. 0,1,2 or all.")
    ap.add_argument("--left-bones", type=str, default="thigh_l,calf_l,foot_l,ball_l")
    ap.add_argument("--right-bones", type=str, default="thigh_r,calf_r,foot_r,ball_r")
    ap.add_argument("--focus-joints", type=str, default="calf_l,calf_r,foot_l,foot_r,ball_l,ball_r")
    ap.add_argument("--hotspots", type=str, default="12,14,54,55")
    ap.add_argument("--contact-stance-thr", type=float, default=0.55)
    ap.add_argument("--contact-flight-thr", type=float, default=0.20)
    ap.add_argument("--contact-dom-margin", type=float, default=0.05)
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

    left_bones = base._parse_csv(args.left_bones)
    right_bones = base._parse_csv(args.right_bones)
    focus_joints = base._parse_csv(args.focus_joints)
    hotspots = sorted(list(_parse_int_csv(args.hotspots)))

    if len(left_bones) != len(right_bones):
        raise SystemExit("[FATAL] left/right bones length mismatch")
    miss = [b for b in (left_bones + right_bones + focus_joints) if b not in name_to_idx]
    if miss:
        raise SystemExit(f"[FATAL] unresolved joints/bones: {miss}")

    left_slices = [base._joint_rot6d_slice(rot_slice, name_to_idx[b]) for b in left_bones]
    right_slices = [base._joint_rot6d_slice(rot_slice, name_to_idx[b]) for b in right_bones]
    left_joint_idx = [name_to_idx[b] for b in left_bones]
    right_joint_idx = [name_to_idx[b] for b in right_bones]
    focus_joint_idx = [name_to_idx[b] for b in focus_joints]

    steps_spec = str(args.steps or "").strip()
    if not steps_spec:
        js_steps = rho_payload.get("steps", [])
        if isinstance(js_steps, list) and js_steps:
            steps_spec = ",".join(str(int(x)) for x in js_steps)
        else:
            steps_spec = "0,1"

    clip_ids_filter = _parse_int_csv(str(args.clip_ids or ""))
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

    by_sic: Dict[int, SicAccumulator] = defaultdict(SicAccumulator)

    first_linear = base._find_first_linear(getattr(model, "direct_pose_head", None))
    if first_linear is None:
        raise SystemExit("[FATAL] cannot find first linear in direct_pose_head")

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
        angvel = batch.get("angvel")
        pose_hist = batch.get("pose_hist")
        start = batch.get("start")
        clip_len = batch.get("clip_len")

        if not (torch.is_tensor(state) and torch.is_tensor(cond) and torch.is_tensor(gt_motion)):
            raise SystemExit("[FATAL] missing motion/cond_in/gt_motion")
        if not torch.is_tensor(start):
            raise SystemExit("[FATAL] missing start in reconstructed batch")
        if not torch.is_tensor(clip_len):
            clip_len = torch.full_like(start, int(state.shape[1]))

        capture: Dict[str, torch.Tensor] = {}

        def _pre_hook(mod: torch.nn.Module, inputs):
            if not inputs:
                return inputs
            x = inputs[0]
            if not torch.is_tensor(x):
                return inputs
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            capture["x"] = x
            if len(inputs) == 1:
                return (x,)
            return (x, *inputs[1:])

        model.eval()
        model.zero_grad(set_to_none=True)
        hook = first_linear.register_forward_pre_hook(_pre_hook)
        try:
            with torch.enable_grad():
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

                x = capture.get("x")
                if not torch.is_tensor(x):
                    raise SystemExit("[FATAL] failed to capture direct head input")

                B = int(out_direct.shape[0])
                T = int(out_direct.shape[1])
                n_rows = int(x.shape[0])
                if B <= 0 or n_rows <= 0 or (n_rows % B) != 0:
                    raise SystemExit("[FATAL] invalid x rows vs batch size")
                t_rows = int(n_rows // B)

                steps = base._parse_steps(steps_spec, int(out_direct.shape[1]))
                if not steps:
                    raise SystemExit("[FATAL] no valid steps")
                steps = [int(t) for t in steps if 0 <= int(t) < t_rows]
                if not steps:
                    raise SystemExit("[FATAL] no valid steps after x-time alignment")

                pred6 = out_direct[..., rot_slice].reshape(B, T, joint_count, 6)
                gt6 = gt_motion[..., rot_slice].reshape(B, T, joint_count, 6)
                pred_R = rot6d_to_matrix(pred6)
                gt_R = rot6d_to_matrix(gt6)
                geo_deg = geodesic_R(pred_R, gt_R) * (180.0 / math.pi)  # (B,T,J)
                geo_np = geo_deg.detach().cpu().numpy().astype(np.float64, copy=False)

                contacts_np: Optional[np.ndarray] = None
                if torch.is_tensor(contacts) and contacts.dim() == 3 and int(contacts.shape[-1]) >= 2:
                    contacts_np = contacts[..., :2].detach().cpu().numpy().astype(np.float64, copy=False)

                start_np = start.view(-1).detach().cpu().numpy().astype(np.int64, copy=False)
                clip_len_np = clip_len.view(-1).detach().cpu().numpy().astype(np.int64, copy=False)

                for i, t in enumerate(steps):
                    y_t = out_direct[:, t, :]
                    loss_left = sum(y_t[:, sl].sum() for sl in left_slices)
                    loss_right = sum(y_t[:, sl].sum() for sl in right_slices)
                    g_left = torch.autograd.grad(loss_left, x, retain_graph=True, create_graph=False, allow_unused=False)[0]
                    g_right = torch.autograd.grad(
                        loss_right,
                        x,
                        retain_graph=(i < len(steps) - 1),
                        create_graph=False,
                        allow_unused=False,
                    )[0]

                    for b in range(B):
                        clip_n = int(clip_len_np[b]) if int(clip_len_np[b]) > 0 else int(t_rows)
                        sic_abs = int((int(start_np[b]) + int(t)) % max(1, clip_n))
                        rec = by_sic[sic_abs]
                        rec.rows += 1
                        rec.batches.add(str(diag_pt))

                        ridx = b * t_rows + int(t)
                        gl = float(g_left[ridx].norm().detach().cpu())
                        gr = float(g_right[ridx].norm().detach().cpu())
                        if math.isfinite(gl) and math.isfinite(gr):
                            rec.left_sq_sum += gl * gl
                            rec.right_sq_sum += gr * gr
                            rec.jac_row_values.append(_safe_ratio(gr, gl))

                        if contacts_np is not None and b < contacts_np.shape[0] and t < contacts_np.shape[1]:
                            cl = float(contacts_np[b, t, 0])
                            cr = float(contacts_np[b, t, 1])
                            rec.phase_counts[_phase_label(
                                cl,
                                cr,
                                stance_thr=float(args.contact_stance_thr),
                                flight_thr=float(args.contact_flight_thr),
                                dominance_margin=float(args.contact_dom_margin),
                            )] += 1

                        row_geo = geo_np[b, t, :]
                        rec.err_all_joints.append(float(np.mean(row_geo)))

                        focus_vals = row_geo[focus_joint_idx]
                        left_vals = row_geo[left_joint_idx]
                        right_vals = row_geo[right_joint_idx]
                        rec.err_focus_mean.append(float(np.mean(focus_vals)))
                        rec.err_left_focus_mean.append(float(np.mean(left_vals)))
                        rec.err_right_focus_mean.append(float(np.mean(right_vals)))
                        rec.err_focus_rl_ratio.append(_safe_ratio(float(np.mean(right_vals)), float(np.mean(left_vals))))

                        for jn, ji in zip(focus_joints, focus_joint_idx):
                            rec.err_by_joint[str(jn)].append(float(row_geo[int(ji)]))
        finally:
            hook.remove()
            model.zero_grad(set_to_none=True)

    if not by_sic:
        raise SystemExit("[FATAL] empty SIC aggregation")

    by_sic_rows: List[Dict[str, Any]] = []
    for sic in sorted(by_sic.keys()):
        rec = by_sic[sic]
        jac = _safe_ratio(math.sqrt(max(rec.right_sq_sum, 0.0)), math.sqrt(max(rec.left_sq_sum, 0.0)))
        phase_major = "NA"
        if rec.phase_counts:
            phase_major = sorted(rec.phase_counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]

        row = {
            "sic": int(sic),
            "n_rows": int(rec.rows),
            "n_batches": int(len(rec.batches)),
            "jac_rl": float(jac),
            "jac_row": _summarize_vals(rec.jac_row_values),
            "phase_major": phase_major,
            "phase_counts": {k: int(v) for k, v in sorted(rec.phase_counts.items())},
            "err_all_joints": _summarize_vals(rec.err_all_joints),
            "err_focus_mean": _summarize_vals(rec.err_focus_mean),
            "err_left_focus_mean": _summarize_vals(rec.err_left_focus_mean),
            "err_right_focus_mean": _summarize_vals(rec.err_right_focus_mean),
            "err_focus_rl_ratio": _summarize_vals(rec.err_focus_rl_ratio),
            "err_by_joint": {jn: _summarize_vals(vals) for jn, vals in sorted(rec.err_by_joint.items())},
        }
        by_sic_rows.append(row)

    def _corr(metric_getter) -> Dict[str, Any]:
        x_vals: List[float] = []
        y_vals: List[float] = []
        for r in by_sic_rows:
            x = _safe_float(r.get("jac_rl", float("nan")))
            y = _safe_float(metric_getter(r))
            if math.isfinite(x) and math.isfinite(y):
                x_vals.append(x)
                y_vals.append(y)
        x_arr = np.asarray(x_vals, dtype=np.float64)
        y_arr = np.asarray(y_vals, dtype=np.float64)
        return {
            "n_sic": int(x_arr.size),
            "pearson": _pearson(x_arr, y_arr),
            "spearman": _spearman(x_arr, y_arr),
        }

    corr: Dict[str, Any] = {
        "err_all_joints_mean": _corr(lambda r: (r.get("err_all_joints", {}) or {}).get("mean", float("nan"))),
        "err_focus_mean": _corr(lambda r: (r.get("err_focus_mean", {}) or {}).get("mean", float("nan"))),
        "err_left_focus_mean": _corr(lambda r: (r.get("err_left_focus_mean", {}) or {}).get("mean", float("nan"))),
        "err_right_focus_mean": _corr(lambda r: (r.get("err_right_focus_mean", {}) or {}).get("mean", float("nan"))),
        "err_focus_rl_ratio": _corr(lambda r: (r.get("err_focus_rl_ratio", {}) or {}).get("mean", float("nan"))),
    }
    for jn in focus_joints:
        corr[f"err_joint_{jn}"] = _corr(
            lambda r, joint_name=jn: ((r.get("err_by_joint", {}) or {}).get(str(joint_name), {}) or {}).get("mean", float("nan"))
        )

    out = {
        "source_rho_json": str(rho_path),
        "model": model_path,
        "diag_pts": [str(Path(x).expanduser().resolve()) for x in diag_pts],
        "steps_used": [int(x) for x in (base._parse_steps(steps_spec, int(args.seq_len)) or [])],
        "clip_filter": {
            "enabled": bool(apply_clip_filter),
            "clip_ids": sorted(list(clip_ids_filter)),
            "clip_name_contains": list(clip_name_filter),
        },
        "hotspots": [int(x) for x in hotspots],
        "left_bones": list(left_bones),
        "right_bones": list(right_bones),
        "focus_joints": list(focus_joints),
        "contact_phase_config": {
            "stance_thr": float(args.contact_stance_thr),
            "flight_thr": float(args.contact_flight_thr),
            "dominance_margin": float(args.contact_dom_margin),
        },
        "by_sic": by_sic_rows,
        "sic_level_correlation": corr,
    }

    out_json = Path(args.out_json).expanduser().resolve() if str(args.out_json).strip() else rho_path.with_name(
        rho_path.stem + "_sic_jac_vs_direct_error.json"
    )
    out_md = Path(args.out_md).expanduser().resolve() if str(args.out_md).strip() else out_json.with_suffix(".md")

    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(out), encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
