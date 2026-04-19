#!/usr/bin/env python3
"""Diagnose jac_RL_all in Walk_F by |dir_body_y| and contact-phase bins.

This script is a Walk_F-focused follow-up of diagnose_jac_rl_by_dir_sign.py.
It filters diag rows to target clips (typically Walk_F), then reports:

    jac_RL_all = ||d y_R / d x|| / ||d y_L / d x||

aggregated by:
1) |dir_body_y| quantile bins
2) contact-phase bins (left/right/double/flight/transition)
3) cross bins: (|dir_body_y| bin) x (contact-phase bin)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

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
from train.configuration.norm_spec import merge_norm_spec
from train.data.dataset import MotionEventDataset


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


def _sign_test_two_sided(pos_n: int, neg_n: int) -> float:
    n = int(pos_n) + int(neg_n)
    if n <= 0:
        return float("nan")
    k = int(pos_n)
    cdf_lo = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    sf_hi = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * min(cdf_lo, sf_hi)))


def _parse_int_csv(spec: str) -> Set[int]:
    out: Set[int] = set()
    for tok in str(spec or "").split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            out.add(int(s))
        except Exception:
            raise SystemExit(f"[FATAL] invalid integer token in --clip-ids: {s}")
    return out


def _parse_lower_csv(spec: str) -> List[str]:
    out: List[str] = []
    for tok in str(spec or "").split(","):
        s = tok.strip().lower()
        if s:
            out.append(s)
    return out


def _parse_quantiles(spec: str) -> List[float]:
    toks = [t.strip() for t in str(spec or "").split(",") if t.strip()]
    if len(toks) < 2:
        raise SystemExit("[FATAL] --abs-dir-quantiles requires at least 2 values.")
    vals: List[float] = []
    for t in toks:
        try:
            v = float(t)
        except Exception:
            raise SystemExit(f"[FATAL] invalid quantile token: {t}")
        if not (0.0 <= v <= 1.0):
            raise SystemExit(f"[FATAL] quantile out of range [0,1]: {v}")
        vals.append(v)
    vals = sorted(vals)
    if vals[0] > 0.0:
        vals = [0.0] + vals
    if vals[-1] < 1.0:
        vals = vals + [1.0]
    uniq: List[float] = []
    for v in vals:
        if not uniq or abs(v - uniq[-1]) > 1e-9:
            uniq.append(v)
    if len(uniq) < 2:
        raise SystemExit("[FATAL] failed to build valid quantile boundaries.")
    return uniq


def _diag_clip_ids(diag_pt: Path) -> List[int]:
    payload = torch.load(str(diag_pt), map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"[FATAL] invalid diag pt payload: {diag_pt}")
    clip_id = payload.get("clip_id")
    if not torch.is_tensor(clip_id):
        raise SystemExit(f"[FATAL] diag pt missing clip_id: {diag_pt}")
    return [int(x) for x in clip_id.view(-1).tolist()]


def _slice_batch_rows(batch: Dict[str, torch.Tensor], keep_rows: List[int]) -> Dict[str, torch.Tensor]:
    if not batch:
        return batch
    any_tensor = next((v for v in batch.values() if torch.is_tensor(v) and v.dim() >= 1), None)
    if any_tensor is None:
        return batch
    bsz = int(any_tensor.shape[0])
    if not keep_rows:
        raise SystemExit("[FATAL] empty keep_rows for batch slicing")
    idx = torch.as_tensor(keep_rows, dtype=torch.long, device=any_tensor.device)
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() >= 1 and int(v.shape[0]) == bsz:
            out[k] = v.index_select(0, idx)
        else:
            out[k] = v
    return out


def _build_clip_path_by_id(
    *,
    seq_len: int,
    data_root: Path,
    bundle: Path,
    pretrain_template: Path,
) -> Dict[int, str]:
    norm_spec = merge_norm_spec(bundle.resolve(), pretrain_template.resolve(), pretrain_keys=None, strict=True)
    ds = MotionEventDataset(
        data_dir=str(data_root.resolve()),
        seq_len=int(seq_len),
        paths=None,
        pose_hist_len=int(norm_spec.get("pose_hist_len", 0) or 0),
        norm_spec=norm_spec,
        index_mode="sliding",
    )
    paths = list(getattr(ds, "paths", []) or [])
    return {int(i): str(p) for i, p in enumerate(paths)}


def _contacts_for_label(batch: Dict[str, torch.Tensor], label_time: str) -> np.ndarray:
    contacts = batch.get("contacts")
    if not (torch.is_tensor(contacts) and contacts.dim() == 3 and int(contacts.shape[-1]) >= 2):
        raise SystemExit("[FATAL] missing/invalid contacts for contact-phase split.")
    c = contacts[..., :2]
    if str(label_time).lower() == "t1":
        c_shift = c.clone()
        if int(c.shape[1]) > 1:
            c_shift[:, :-1, :] = c[:, 1:, :]
        c = c_shift
    return c.detach().cpu().numpy().reshape(-1, 2).astype(np.float64, copy=False)


def _build_abs_masks(
    abs_dir: np.ndarray,
    quantiles: List[float],
    *,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    finite = np.isfinite(abs_dir)
    if valid_mask is not None:
        if valid_mask.shape != abs_dir.shape:
            raise SystemExit("[FATAL] abs-mask valid_mask shape mismatch.")
        finite = finite & valid_mask.astype(bool, copy=False)
    vals = abs_dir[finite]
    if vals.size <= 0:
        raise SystemExit("[FATAL] empty finite |dir_body_y| values.")
    q_edges = [float(np.quantile(vals, q)) for q in quantiles]
    masks: Dict[str, np.ndarray] = {}
    defs: List[Dict[str, Any]] = []
    eps = 1e-12
    for i in range(len(q_edges) - 1):
        lo = float(q_edges[i])
        hi = float(q_edges[i + 1])
        if hi <= lo + eps:
            continue
        if i == len(q_edges) - 2:
            m = finite & (abs_dir >= lo) & (abs_dir <= hi + eps)
        else:
            m = finite & (abs_dir >= lo) & (abs_dir < hi)
        q0 = int(round(100.0 * quantiles[i]))
        q1 = int(round(100.0 * quantiles[i + 1]))
        key = f"abs_q{q0:02d}_{q1:02d}"
        masks[key] = m
        defs.append(
            {
                "key": key,
                "q_lo": float(quantiles[i]),
                "q_hi": float(quantiles[i + 1]),
                "val_lo": lo,
                "val_hi": hi,
                "n": int(m.sum()),
            }
        )
    if not masks:
        raise SystemExit("[FATAL] all |dir_body_y| bins collapsed.")
    return masks, defs


def _build_phase_masks(
    contacts_2: np.ndarray,
    *,
    stance_thr: float,
    flight_thr: float,
    dominance_margin: float,
) -> Dict[str, np.ndarray]:
    if contacts_2.ndim != 2 or int(contacts_2.shape[1]) < 2:
        raise SystemExit("[FATAL] invalid contacts matrix for phase masks.")
    l = contacts_2[:, 0]
    r = contacts_2[:, 1]
    valid = np.isfinite(l) & np.isfinite(r)
    double_support = valid & (l >= stance_thr) & (r >= stance_thr)
    left_stance = valid & (~double_support) & ((l - r) > dominance_margin)
    right_stance = valid & (~double_support) & ((r - l) > dominance_margin)
    flight = valid & (l <= flight_thr) & (r <= flight_thr)
    transition = valid & (~double_support) & (~left_stance) & (~right_stance) & (~flight)
    return {
        "phase_left_stance": left_stance,
        "phase_right_stance": right_stance,
        "phase_double_support": double_support,
        "phase_flight": flight,
        "phase_transition": transition,
    }


def _summarize_subset_rows(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = np.asarray([
        _safe_float(r.get("subsets", {}).get(key, {}).get("jac_rl_all", float("nan")))
        for r in rows
    ], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n_rows = np.asarray([
        _safe_float(r.get("subsets", {}).get(key, {}).get("n", 0))
        for r in rows
    ], dtype=np.float64)
    n_rows = n_rows[np.isfinite(n_rows)]
    return {
        "jac_rl_all_mean": float(vals.mean()) if vals.size > 0 else float("nan"),
        "jac_rl_all_std": float(vals.std()) if vals.size > 0 else float("nan"),
        "n_steps_effective": int(vals.size),
        "n_rows_mean": float(n_rows.mean()) if n_rows.size > 0 else float("nan"),
    }


def _measure_batch(
    *,
    model: torch.nn.Module,
    trainer: Any,
    batch: Dict[str, torch.Tensor],
    steps_spec: str,
    left_slices: List[slice],
    right_slices: List[slice],
    pelvis_orientation_source: str,
    label_time: str,
    abs_quantiles: List[float],
    contact_stance_thr: float,
    contact_flight_thr: float,
    contact_dom_margin: float,
) -> Dict[str, Any]:
    first_linear = signsplit._find_first_linear(getattr(model, "direct_pose_head", None))
    state = batch.get("motion")
    cond = batch.get("cond_in")
    contacts = batch.get("contacts")
    angvel = batch.get("angvel")
    pose_hist = batch.get("pose_hist")
    if not (torch.is_tensor(state) and torch.is_tensor(cond)):
        raise SystemExit("[FATAL] missing motion/cond_in")

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

    rows: List[Dict[str, Any]] = []
    sign_info: Dict[str, Any] = {}
    subset_keys: List[str] = []
    abs_defs: List[Dict[str, Any]] = []
    phase_keys: List[str] = []
    cross_keys: List[str] = []
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

            steps = base._parse_steps(steps_spec, int(out_direct.shape[1]))
            if not steps:
                raise SystemExit("[FATAL] no valid steps")

            bsz = int(out_direct.shape[0])
            n_rows = int(x.shape[0])
            if bsz <= 0 or n_rows <= 0:
                raise SystemExit("[FATAL] invalid batch/x size for time alignment.")
            if (n_rows % bsz) != 0:
                raise SystemExit(f"[FATAL] x rows ({n_rows}) not divisible by batch size ({bsz}).")
            t_rows = int(n_rows // bsz)
            step_arr = np.asarray([int(t) for t in steps], dtype=np.int64)
            if np.any(step_arr < 0) or np.any(step_arr >= t_rows):
                raise SystemExit(f"[FATAL] requested steps {steps} out of x-time range [0,{t_rows}).")
            row_time = (np.arange(n_rows, dtype=np.int64) % t_rows)
            step_any_mask_np = np.isin(row_time, step_arr)

            sign_feature, _, sign_info = signsplit._resolve_sign_feature(
                trainer=trainer,
                batch=batch,
                ret=ret if isinstance(ret, dict) else {},
                sign_source="dir_body_y",
                pelvis_orientation_source=pelvis_orientation_source,
                label_time=label_time,
            )
            if int(sign_feature.shape[0]) != int(x.shape[0]):
                raise SystemExit(
                    f"[FATAL] sign feature rows ({sign_feature.shape[0]}) != x rows ({int(x.shape[0])})"
                )

            abs_dir = np.abs(sign_feature.astype(np.float64, copy=False))
            abs_masks_np, abs_defs = _build_abs_masks(
                abs_dir,
                abs_quantiles,
                valid_mask=step_any_mask_np,
            )

            contacts_np = _contacts_for_label(batch, label_time=label_time)
            if int(contacts_np.shape[0]) != int(x.shape[0]):
                raise SystemExit(
                    f"[FATAL] contacts rows ({contacts_np.shape[0]}) != x rows ({int(x.shape[0])})"
                )
            phase_masks_np = _build_phase_masks(
                contacts_np,
                stance_thr=float(contact_stance_thr),
                flight_thr=float(contact_flight_thr),
                dominance_margin=float(contact_dom_margin),
            )
            phase_masks_np = {k: (v & step_any_mask_np) for k, v in phase_masks_np.items()}
            phase_keys = list(phase_masks_np.keys())

            masks_np: Dict[str, np.ndarray] = {"all": step_any_mask_np.copy()}
            masks_np.update(abs_masks_np)
            masks_np.update(phase_masks_np)
            for ak, am in abs_masks_np.items():
                for pk, pm in phase_masks_np.items():
                    ck = f"{ak}__{pk}"
                    masks_np[ck] = am & pm
                    cross_keys.append(ck)
            subset_keys = list(masks_np.keys())

            masks = {k: torch.from_numpy(v.astype(np.bool_)).to(x.device) for k, v in masks_np.items()}
            row_time_t = torch.from_numpy(row_time).to(device=x.device)

            for i, t in enumerate(steps):
                y_t = out_direct[:, int(t), :]
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

                rec: Dict[str, Any] = {"step": int(t), "subsets": {}}
                step_mask_t = row_time_t.eq(int(t))
                for name, m in masks.items():
                    m_step = m & step_mask_t
                    n_eff = int(m_step.sum().detach().cpu())
                    if n_eff <= 0:
                        rec["subsets"][name] = {"n": 0, "jac_rl_all": float("nan")}
                        continue
                    l_all = float(g_left[m_step].norm().detach().cpu())
                    r_all = float(g_right[m_step].norm().detach().cpu())
                    rec["subsets"][name] = {
                        "n": n_eff,
                        "jac_rl_all": _safe_ratio(r_all, l_all),
                        "left_norm_all": l_all,
                        "right_norm_all": r_all,
                    }
                rows.append(rec)
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)

    agg: Dict[str, Any] = {}
    for k in subset_keys:
        agg[k] = _summarize_subset_rows(rows, k)
    return {
        "sign_info": sign_info,
        "subset_keys": subset_keys,
        "abs_bin_defs": abs_defs,
        "phase_keys": phase_keys,
        "cross_keys": cross_keys,
        "steps": [int(r.get("step", -1)) for r in rows],
        "per_step": rows,
        "aggregate": agg,
        "direct_head_input_shape": list(capture.get("x").shape) if torch.is_tensor(capture.get("x")) else [],
    }


def _to_markdown(out: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Jacobian RL by |dir_body_y| and Contact Phase")
    lines.append("")
    lines.append(f"- source rho json: `{out.get('source_rho_json', '')}`")
    lines.append(f"- model: `{out.get('model', '')}`")
    lines.append(f"- diag batches: `{len(out.get('diag_pts', []))}`")
    lines.append(f"- steps: `{out.get('steps', [])}`")
    lines.append(f"- pelvis orientation source: `{out.get('pelvis_orientation_source', '')}`")
    lines.append(f"- label time: `{out.get('label_time', '')}`")
    clip_filter = out.get("clip_filter", {}) if isinstance(out.get("clip_filter"), dict) else {}
    lines.append(f"- clip filter enabled: `{bool(clip_filter.get('enabled', False))}`")
    if bool(clip_filter.get("enabled", False)):
        lines.append(f"- clip ids: `{clip_filter.get('clip_ids', [])}`")
        lines.append(f"- clip name contains: `{clip_filter.get('clip_name_contains', [])}`")
    lines.append("")

    agg = out.get("aggregate", {}) if isinstance(out.get("aggregate"), dict) else {}
    lines.append("## Overall")
    lines.append("")
    lines.append("|subset|jac_RL_all mean|std|`>1` n|`<1` n|p(two-sided)|")
    lines.append("|:--|--:|--:|--:|--:|--:|")
    for key in ["all"]:
        obj = agg.get(key, {}) if isinstance(agg.get(key), dict) else {}
        lines.append(
            f"|`{key}`|{_safe_float(obj.get('jac_rl_all_mean', float('nan'))):.6f}|"
            f"{_safe_float(obj.get('jac_rl_all_std', float('nan'))):.6f}|"
            f"{int(obj.get('gt1_n', 0) or 0)}|{int(obj.get('lt1_n', 0) or 0)}|"
            f"{_safe_float(obj.get('sign_test_p_two_sided', float('nan'))):.4f}|"
        )
    lines.append("")

    phase_keys = list(out.get("phase_keys", []) or [])
    if phase_keys:
        lines.append("## Contact Phase")
        lines.append("")
        lines.append("|subset|jac_RL_all mean|std|n_batches_eff|n_rows_mean|`>1` n|`<1` n|")
        lines.append("|:--|--:|--:|--:|--:|--:|--:|")
        for key in phase_keys:
            obj = agg.get(key, {}) if isinstance(agg.get(key), dict) else {}
            lines.append(
                f"|`{key}`|{_safe_float(obj.get('jac_rl_all_mean', float('nan'))):.6f}|"
                f"{_safe_float(obj.get('jac_rl_all_std', float('nan'))):.6f}|"
                f"{int(obj.get('n_batches_effective', 0) or 0)}|{_safe_float(obj.get('n_rows_mean', float('nan'))):.1f}|"
                f"{int(obj.get('gt1_n', 0) or 0)}|{int(obj.get('lt1_n', 0) or 0)}|"
            )
        lines.append("")

    abs_defs = list(out.get("abs_bin_defs", []) or [])
    if abs_defs:
        lines.append("## |dir_body_y| Quantile Bins")
        lines.append("")
        lines.append("|subset|q-range|value range|jac_RL_all mean|std|n_rows_raw|")
        lines.append("|:--|--:|--:|--:|--:|--:|")
        for d in abs_defs:
            key = str(d.get("key", ""))
            obj = agg.get(key, {}) if isinstance(agg.get(key), dict) else {}
            lines.append(
                f"|`{key}`|{_safe_float(d.get('q_lo', float('nan'))):.2f}-{_safe_float(d.get('q_hi', float('nan'))):.2f}|"
                f"{_safe_float(d.get('val_lo', float('nan'))):.4f}-{_safe_float(d.get('val_hi', float('nan'))):.4f}|"
                f"{_safe_float(obj.get('jac_rl_all_mean', float('nan'))):.6f}|"
                f"{_safe_float(obj.get('jac_rl_all_std', float('nan'))):.6f}|"
                f"{int(d.get('n', 0) or 0)}|"
            )
        lines.append("")

    cross_keys = list(out.get("cross_keys", []) or [])
    if cross_keys:
        ranked: List[Tuple[str, float]] = []
        for k in cross_keys:
            m = _safe_float(agg.get(k, {}).get("jac_rl_all_mean", float("nan")))
            if math.isfinite(m):
                ranked.append((k, abs(m - 1.0)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        lines.append("## Top Cross Bins by |mean-1|")
        lines.append("")
        lines.append("|subset|jac_RL_all mean|std|n_batches_eff|n_rows_mean|`>1` n|`<1` n|")
        lines.append("|:--|--:|--:|--:|--:|--:|--:|")
        for k, _ in ranked[:12]:
            obj = agg.get(k, {}) if isinstance(agg.get(k), dict) else {}
            lines.append(
                f"|`{k}`|{_safe_float(obj.get('jac_rl_all_mean', float('nan'))):.6f}|"
                f"{_safe_float(obj.get('jac_rl_all_std', float('nan'))):.6f}|"
                f"{int(obj.get('n_batches_effective', 0) or 0)}|{_safe_float(obj.get('n_rows_mean', float('nan'))):.1f}|"
                f"{int(obj.get('gt1_n', 0) or 0)}|{int(obj.get('lt1_n', 0) or 0)}|"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk_F jac_RL diagnostics by |dir_body_y| and contact phase bins.")
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
    ap.add_argument("--steps", type=str, default="", help="Override steps; default uses rho-json steps")
    ap.add_argument("--left-bones", type=str, default="thigh_l,calf_l,foot_l,ball_l")
    ap.add_argument("--right-bones", type=str, default="thigh_r,calf_r,foot_r,ball_r")
    ap.add_argument("--pelvis-orientation-source", type=str, default="gt", choices=("gt", "pred"))
    ap.add_argument("--label-time", type=str, default="t", choices=("t", "t1"))
    ap.add_argument("--abs-dir-quantiles", type=str, default="0,0.33,0.66,1.0")
    ap.add_argument("--contact-stance-thr", type=float, default=0.55)
    ap.add_argument("--contact-flight-thr", type=float, default=0.20)
    ap.add_argument("--contact-dom-margin", type=float, default=0.05)
    ap.add_argument("--clip-ids", type=str, default="", help="Optional clip-id filter (comma-separated).")
    ap.add_argument(
        "--clip-name-contains",
        type=str,
        default="Walk_F",
        help="Optional lowercase-substring filter on clip npz path (comma-separated).",
    )
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
    if len(left_bones) != len(right_bones):
        raise SystemExit("[FATAL] left/right bones length mismatch")
    miss = [b for b in (left_bones + right_bones) if b not in name_to_idx]
    if miss:
        raise SystemExit(f"[FATAL] unresolved bones: {miss}")
    left_slices = [base._joint_rot6d_slice(rot_slice, name_to_idx[b]) for b in left_bones]
    right_slices = [base._joint_rot6d_slice(rot_slice, name_to_idx[b]) for b in right_bones]

    steps_spec = str(args.steps or "").strip()
    if not steps_spec:
        js_steps = rho_payload.get("steps", [])
        if isinstance(js_steps, list) and js_steps:
            steps_spec = ",".join(str(int(x)) for x in js_steps)
        else:
            steps_spec = "0,1"

    abs_quantiles = _parse_quantiles(str(args.abs_dir_quantiles))
    clip_ids_filter = _parse_int_csv(str(args.clip_ids or ""))
    clip_name_filter = _parse_lower_csv(str(args.clip_name_contains or ""))
    apply_clip_filter = bool(clip_ids_filter or clip_name_filter)
    clip_path_by_id: Dict[int, str] = {}
    if apply_clip_filter:
        clip_path_by_id = _build_clip_path_by_id(
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

    per_batch: List[Dict[str, Any]] = []
    expected_subset_keys: Optional[List[str]] = None
    expected_phase_keys: Optional[List[str]] = None
    expected_cross_keys: Optional[List[str]] = None
    expected_abs_defs: Optional[List[Dict[str, Any]]] = None
    skipped_batches: List[Dict[str, Any]] = []
    for dp in diag_pts:
        diag_pt = Path(dp).expanduser().resolve()
        if not diag_pt.is_file():
            raise SystemExit(f"[FATAL] diag pt missing: {diag_pt}")
        clip_ids = _diag_clip_ids(diag_pt)
        keep_rows = [i for i, cid in enumerate(clip_ids) if _keep_clip(int(cid))]
        if apply_clip_filter and not keep_rows:
            skipped_batches.append(
                {
                    "diag_pt": str(diag_pt),
                    "reason": "no rows matched clip filter",
                    "rows_total": int(len(clip_ids)),
                }
            )
            continue
        batch = base._rebuild_batch_from_diag(
            diag_pt=diag_pt,
            seq_len=int(args.seq_len),
            data_root=Path(args.data_root).expanduser().resolve(),
            bundle=Path(args.bundle).expanduser(),
            pretrain_template=Path(args.pretrain_template).expanduser(),
            device=runner.device,
        )
        rows_total = int(len(clip_ids))
        rows_kept = int(len(keep_rows)) if apply_clip_filter else rows_total
        if apply_clip_filter and rows_kept < rows_total:
            batch = _slice_batch_rows(batch, keep_rows)

        ret = _measure_batch(
            model=model,
            trainer=trainer,
            batch=batch,
            steps_spec=steps_spec,
            left_slices=left_slices,
            right_slices=right_slices,
            pelvis_orientation_source=str(args.pelvis_orientation_source),
            label_time=str(args.label_time),
            abs_quantiles=abs_quantiles,
            contact_stance_thr=float(args.contact_stance_thr),
            contact_flight_thr=float(args.contact_flight_thr),
            contact_dom_margin=float(args.contact_dom_margin),
        )
        subset_keys = list(ret.get("subset_keys", []) or [])
        phase_keys = list(ret.get("phase_keys", []) or [])
        cross_keys = list(ret.get("cross_keys", []) or [])
        abs_defs = list(ret.get("abs_bin_defs", []) or [])
        if not subset_keys:
            raise SystemExit("[FATAL] empty subset keys")
        if expected_subset_keys is None:
            expected_subset_keys = list(subset_keys)
            expected_phase_keys = list(phase_keys)
            expected_cross_keys = list(cross_keys)
            expected_abs_defs = list(abs_defs)
        else:
            if subset_keys != expected_subset_keys:
                raise SystemExit("[FATAL] inconsistent subset keys across batches")
            if phase_keys != (expected_phase_keys or []):
                raise SystemExit("[FATAL] inconsistent phase keys across batches")
            if cross_keys != (expected_cross_keys or []):
                raise SystemExit("[FATAL] inconsistent cross keys across batches")

        per_batch.append(
            {
                "diag_pt": str(diag_pt),
                "rows_total": rows_total,
                "rows_kept": rows_kept,
                **ret,
            }
        )

    if not per_batch:
        raise SystemExit("[FATAL] no diag batch left after clip filtering")

    subset_keys = list(expected_subset_keys or [])
    agg: Dict[str, Any] = {}
    for k in subset_keys:
        arr = np.asarray([
            _safe_float(r.get("aggregate", {}).get(k, {}).get("jac_rl_all_mean", float("nan")))
            for r in per_batch
        ], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        gt1_n = int(np.sum(arr > 1.0))
        lt1_n = int(np.sum(arr < 1.0))
        row_m = np.asarray([
            _safe_float(r.get("aggregate", {}).get(k, {}).get("n_rows_mean", float("nan")))
            for r in per_batch
        ], dtype=np.float64)
        row_m = row_m[np.isfinite(row_m)]
        agg[k] = {
            "jac_rl_all_mean": float(arr.mean()) if arr.size > 0 else float("nan"),
            "jac_rl_all_std": float(arr.std()) if arr.size > 0 else float("nan"),
            "n_batches_effective": int(arr.size),
            "n_rows_mean": float(row_m.mean()) if row_m.size > 0 else float("nan"),
            "gt1_n": gt1_n,
            "lt1_n": lt1_n,
            "sign_test_p_two_sided": _sign_test_two_sided(gt1_n, lt1_n),
        }

    out = {
        "source_rho_json": str(rho_path),
        "model": model_path,
        "diag_pts": [str(Path(x).expanduser().resolve()) for x in diag_pts],
        "steps": [int(x) for x in base._parse_steps(steps_spec, 1000)],
        "pelvis_orientation_source": str(args.pelvis_orientation_source),
        "label_time": str(args.label_time),
        "abs_dir_quantiles": list(abs_quantiles),
        "contact_phase_config": {
            "stance_thr": float(args.contact_stance_thr),
            "flight_thr": float(args.contact_flight_thr),
            "dominance_margin": float(args.contact_dom_margin),
        },
        "clip_filter": {
            "enabled": bool(apply_clip_filter),
            "clip_ids": sorted(int(x) for x in clip_ids_filter),
            "clip_name_contains": list(clip_name_filter),
            "skipped_batches": skipped_batches,
        },
        "subset_keys": subset_keys,
        "phase_keys": list(expected_phase_keys or []),
        "cross_keys": list(expected_cross_keys or []),
        "abs_bin_defs": list(expected_abs_defs or []),
        "aggregate": agg,
        "per_batch": per_batch,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if str(args.out_json).strip():
        p = Path(args.out_json).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Saved] {p}")

    if str(args.out_md).strip():
        p = Path(args.out_md).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_to_markdown(out), encoding="utf-8")
        print(f"[Saved] {p}")


if __name__ == "__main__":
    main()
