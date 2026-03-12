#!/usr/bin/env python3
"""Diagnose DirectGeoLocalDeg spikes with error-only decomposition.

This script intentionally excludes Jacobian metrics. It reconstructs batches from
freerun diag `.pt` files and decomposes DirectGeoLocalDeg by:

1) absolute SIC
2) contact phase
3) |dir_body_y| quantile bins
4) speed quantile bins

It also builds a simple additive explanation for per-SIC error means:
  err_focus(sic) ~= global + phase_effect + absdir_effect + speed_effect
and reports residuals for hotspot SICs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
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


def _parse_quantiles(spec: str) -> List[float]:
    toks = [t.strip() for t in str(spec or "").split(",") if t.strip()]
    if len(toks) < 2:
        raise SystemExit("[FATAL] quantiles require at least 2 values")
    vals: List[float] = []
    for t in toks:
        try:
            v = float(t)
        except Exception:
            raise SystemExit(f"[FATAL] invalid quantile token: {t}")
        if not (0.0 <= v <= 1.0):
            raise SystemExit(f"[FATAL] quantile out of [0,1]: {v}")
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
        raise SystemExit("[FATAL] quantile boundaries collapsed")
    return uniq


def _rankdata_average(x: np.ndarray) -> np.ndarray:
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


def _summarize(arr: np.ndarray) -> Dict[str, Any]:
    vals = np.asarray(arr, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90.0)),
        "p95": float(np.percentile(vals, 95.0)),
    }


def _build_quantile_bins(values: np.ndarray, quantiles: List[float], prefix: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not np.any(finite):
        idx = np.full(values.shape, -1, dtype=np.int64)
        return idx, []

    q_edges = [float(np.quantile(values[finite], q)) for q in quantiles]
    idx = np.full(values.shape, -1, dtype=np.int64)
    defs: List[Dict[str, Any]] = []

    bin_id = 0
    eps = 1e-12
    for i in range(len(q_edges) - 1):
        lo = float(q_edges[i])
        hi = float(q_edges[i + 1])
        if hi <= lo + eps:
            continue
        if i == len(q_edges) - 2:
            m = finite & (values >= lo) & (values <= hi + eps)
        else:
            m = finite & (values >= lo) & (values < hi)
        if not np.any(m):
            continue
        idx[m] = int(bin_id)
        q0 = int(round(100.0 * quantiles[i]))
        q1 = int(round(100.0 * quantiles[i + 1]))
        defs.append(
            {
                "id": int(bin_id),
                "key": f"{prefix}_q{q0:02d}_{q1:02d}",
                "q_lo": float(quantiles[i]),
                "q_hi": float(quantiles[i + 1]),
                "val_lo": lo,
                "val_hi": hi,
                "n": int(m.sum()),
            }
        )
        bin_id += 1
    return idx, defs


def _summarize_group(
    *,
    name: str,
    mask: np.ndarray,
    err_focus: np.ndarray,
    err_left: np.ndarray,
    err_right: np.ndarray,
    abs_dir: np.ndarray,
    speed: np.ndarray,
) -> Dict[str, Any]:
    m = mask.astype(bool, copy=False)
    n = int(np.sum(m))
    if n <= 0:
        return {
            "name": name,
            "n_rows": 0,
            "err_focus": _summarize(np.asarray([], dtype=np.float64)),
            "err_left": _summarize(np.asarray([], dtype=np.float64)),
            "err_right": _summarize(np.asarray([], dtype=np.float64)),
            "err_rl_ratio": _summarize(np.asarray([], dtype=np.float64)),
            "abs_dir_body_y": _summarize(np.asarray([], dtype=np.float64)),
            "speed": _summarize(np.asarray([], dtype=np.float64)),
        }
    l = err_left[m]
    r = err_right[m]
    ratio = np.asarray([_safe_ratio(float(rv), float(lv)) for rv, lv in zip(r, l)], dtype=np.float64)
    return {
        "name": name,
        "n_rows": n,
        "err_focus": _summarize(err_focus[m]),
        "err_left": _summarize(l),
        "err_right": _summarize(r),
        "err_rl_ratio": _summarize(ratio),
        "abs_dir_body_y": _summarize(abs_dir[m]),
        "speed": _summarize(speed[m]),
    }


def _to_markdown(out: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# DirectGeoLocalDeg Error-Only Decomposition")
    lines.append("")
    lines.append(f"- source rho json: `{out.get('source_rho_json', '')}`")
    lines.append(f"- model: `{out.get('model', '')}`")
    lines.append(f"- diag batches: `{len(out.get('diag_pts', []))}`")
    lines.append(f"- steps: `{out.get('steps_used', [])}`")
    lines.append(f"- clip filter: `{out.get('clip_filter', {})}`")
    lines.append(f"- focus joints: `{out.get('focus_joints', [])}`")
    lines.append("")

    glob = out.get("global", {}) if isinstance(out.get("global"), dict) else {}
    lines.append("## Global")
    lines.append("")
    lines.append(
        f"- rows: `{int(glob.get('n_rows', 0))}`, SIC covered: `{int(glob.get('n_sic', 0))}`, "
        f"focus mean: `{_safe_float((glob.get('err_focus', {}) or {}).get('mean', float('nan'))):.4f}`"
    )
    add = out.get("additive_model", {}) if isinstance(out.get("additive_model"), dict) else {}
    lines.append(
        f"- additive R2 (phase + |dir_body_y| + speed): `{_safe_float(add.get('r2', float('nan'))):.4f}`"
    )
    lines.append(
        f"- corr(err_focus, abs_dir_body_y): pearson `{_safe_float((glob.get('corr', {}) or {}).get('err_vs_absdir', {}).get('pearson', float('nan'))):.4f}`, "
        f"spearman `{_safe_float((glob.get('corr', {}) or {}).get('err_vs_absdir', {}).get('spearman', float('nan'))):.4f}`"
    )
    lines.append(
        f"- corr(err_focus, speed): pearson `{_safe_float((glob.get('corr', {}) or {}).get('err_vs_speed', {}).get('pearson', float('nan'))):.4f}`, "
        f"spearman `{_safe_float((glob.get('corr', {}) or {}).get('err_vs_speed', {}).get('spearman', float('nan'))):.4f}`"
    )
    lines.append("")

    hotspots = out.get("hotspots", []) if isinstance(out.get("hotspots"), list) else []
    if hotspots:
        lines.append("## Hotspots")
        lines.append("")
        lines.append("|sic|phase_major|abs_bin_major|speed_bin_major|err_focus_mean|err_R/L|residual_after_additive|focus_cancel|top_joint|")
        lines.append("|---:|:--|:--|:--|---:|---:|---:|---:|:--|")
        for r in hotspots:
            lines.append(
                f"|{int(r.get('sic', -1))}|{r.get('phase_major', 'NA')}|{r.get('abs_bin_major', 'NA')}|"
                f"{r.get('speed_bin_major', 'NA')}|"
                f"{_safe_float((r.get('err_focus', {}) or {}).get('mean', float('nan'))):.4f}|"
                f"{_safe_float((r.get('err_rl_ratio', {}) or {}).get('mean', float('nan'))):.4f}|"
                f"{_safe_float(r.get('residual_after_additive', float('nan'))):.4f}|"
                f"{_safe_float(r.get('focus_delta_cancel', float('nan'))):.4f}|"
                f"{r.get('top_joint', 'NA')}|"
            )
        lines.append("")

    phase_rows = out.get("by_phase", []) if isinstance(out.get("by_phase"), list) else []
    if phase_rows:
        lines.append("## By Phase")
        lines.append("")
        lines.append("|phase|n_rows|err_focus_mean|err_R/L|")
        lines.append("|:--|--:|--:|--:|")
        for r in phase_rows:
            lines.append(
                f"|{r.get('name', 'NA')}|{int(r.get('n_rows', 0))}|"
                f"{_safe_float((r.get('err_focus', {}) or {}).get('mean', float('nan'))):.4f}|"
                f"{_safe_float((r.get('err_rl_ratio', {}) or {}).get('mean', float('nan'))):.4f}|"
            )
        lines.append("")

    abs_rows = out.get("by_abs_dir_bin", []) if isinstance(out.get("by_abs_dir_bin"), list) else []
    if abs_rows:
        lines.append("## By |dir_body_y| Bin")
        lines.append("")
        lines.append("|bin|value range|n_rows|err_focus_mean|err_R/L|")
        lines.append("|:--|:--|--:|--:|--:|")
        abs_defs = {str(d.get("key", "")): d for d in (out.get("abs_dir_bin_defs", []) or [])}
        for r in abs_rows:
            d = abs_defs.get(str(r.get("name", "")), {})
            lines.append(
                f"|{r.get('name', 'NA')}|{_safe_float(d.get('val_lo', float('nan'))):.4f}-{_safe_float(d.get('val_hi', float('nan'))):.4f}|"
                f"{int(r.get('n_rows', 0))}|"
                f"{_safe_float((r.get('err_focus', {}) or {}).get('mean', float('nan'))):.4f}|"
                f"{_safe_float((r.get('err_rl_ratio', {}) or {}).get('mean', float('nan'))):.4f}|"
            )
        lines.append("")

    speed_rows = out.get("by_speed_bin", []) if isinstance(out.get("by_speed_bin"), list) else []
    if speed_rows:
        lines.append("## By Speed Bin")
        lines.append("")
        lines.append("|bin|value range|n_rows|err_focus_mean|err_R/L|")
        lines.append("|:--|:--|--:|--:|--:|")
        speed_defs = {str(d.get("key", "")): d for d in (out.get("speed_bin_defs", []) or [])}
        for r in speed_rows:
            d = speed_defs.get(str(r.get("name", "")), {})
            lines.append(
                f"|{r.get('name', 'NA')}|{_safe_float(d.get('val_lo', float('nan'))):.4f}-{_safe_float(d.get('val_hi', float('nan'))):.4f}|"
                f"{int(r.get('n_rows', 0))}|"
                f"{_safe_float((r.get('err_focus', {}) or {}).get('mean', float('nan'))):.4f}|"
                f"{_safe_float((r.get('err_rl_ratio', {}) or {}).get('mean', float('nan'))):.4f}|"
            )
        lines.append("")

    top_cross = out.get("top_cross_phase_abs", []) if isinstance(out.get("top_cross_phase_abs"), list) else []
    if top_cross:
        lines.append("## Top Cross Bins (phase x |dir_body_y|)")
        lines.append("")
        lines.append("|cross_bin|n_rows|err_focus_mean|err_R/L|")
        lines.append("|:--|--:|--:|--:|")
        for r in top_cross[:12]:
            lines.append(
                f"|{r.get('name', 'NA')}|{int(r.get('n_rows', 0))}|"
                f"{_safe_float((r.get('err_focus', {}) or {}).get('mean', float('nan'))):.4f}|"
                f"{_safe_float((r.get('err_rl_ratio', {}) or {}).get('mean', float('nan'))):.4f}|"
            )
        lines.append("")

    top_sic = out.get("top_sic_by_focus", []) if isinstance(out.get("top_sic_by_focus"), list) else []
    if top_sic:
        lines.append("## Top SIC by focus error")
        lines.append("")
        lines.append("|sic|phase_major|err_focus_mean|err_R/L|top_joint|")
        lines.append("|---:|:--|---:|---:|:--|")
        for r in top_sic[:12]:
            lines.append(
                f"|{int(r.get('sic', -1))}|{r.get('phase_major', 'NA')}|"
                f"{_safe_float((r.get('err_focus', {}) or {}).get('mean', float('nan'))):.4f}|"
                f"{_safe_float((r.get('err_rl_ratio', {}) or {}).get('mean', float('nan'))):.4f}|"
                f"{r.get('top_joint', 'NA')}|"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="DirectGeoLocalDeg spike localization (error-only decomposition).")
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
    ap.add_argument("--steps", type=str, default="", help="Step list, e.g. 0,1,2 or all")
    ap.add_argument("--left-bones", type=str, default="thigh_l,calf_l,foot_l,ball_l")
    ap.add_argument("--right-bones", type=str, default="thigh_r,calf_r,foot_r,ball_r")
    ap.add_argument("--focus-joints", type=str, default="calf_l,calf_r,foot_l,foot_r,ball_l,ball_r")
    ap.add_argument("--hotspots", type=str, default="12,14,54,55")
    ap.add_argument("--pelvis-orientation-source", type=str, default="gt", choices=("gt", "pred"))
    ap.add_argument("--label-time", type=str, default="t", choices=("t", "t1"))
    ap.add_argument("--abs-dir-quantiles", type=str, default="0,0.33,0.66,1.0")
    ap.add_argument("--speed-quantiles", type=str, default="0,0.33,0.66,1.0")
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

    left_joint_idx = [name_to_idx[b] for b in left_bones]
    right_joint_idx = [name_to_idx[b] for b in right_bones]
    focus_joint_idx = [name_to_idx[b] for b in focus_joints]

    steps_spec = str(args.steps or "").strip()
    if not steps_spec:
        js_steps = rho_payload.get("steps", [])
        if isinstance(js_steps, list) and js_steps:
            steps_spec = ",".join(str(int(x)) for x in js_steps)
        else:
            steps_spec = "all"

    abs_quantiles = _parse_quantiles(str(args.abs_dir_quantiles))
    speed_quantiles = _parse_quantiles(str(args.speed_quantiles))

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

    sic_list: List[int] = []
    phase_list: List[str] = []
    abs_dir_list: List[float] = []
    speed_list: List[float] = []
    err_by_joint_rows: List[np.ndarray] = []
    batch_tag_list: List[str] = []

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
            raise SystemExit("[FATAL] missing start in reconstructed batch")
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

            steps = base._parse_steps(steps_spec, T)
            if not steps:
                raise SystemExit("[FATAL] no valid steps")

            pred6 = out_direct[..., rot_slice].reshape(B, T, joint_count, 6)
            gt6 = gt_motion[..., rot_slice].reshape(B, T, joint_count, 6)
            pred_R = rot6d_to_matrix(pred6)
            gt_R = rot6d_to_matrix(gt6)
            geo_deg = geodesic_R(pred_R, gt_R) * (180.0 / math.pi)
            geo_np = geo_deg.detach().cpu().numpy().astype(np.float64, copy=False)

            cond_raw = signsplit._cond_raw_for_label(batch, label_time=str(args.label_time).lower())
            if cond_raw.dim() != 3 or int(cond_raw.shape[0]) != B or int(cond_raw.shape[1]) != T:
                raise SystemExit("[FATAL] cond raw shape mismatch")
            cond_raw_np = cond_raw.detach().cpu().numpy().astype(np.float64, copy=False)
            cdim = int(cond_raw_np.shape[-1])
            action_dim = int(cdim - 3)
            if action_dim < 0:
                raise SystemExit(f"[FATAL] invalid cond dim: {cdim}")
            dir_x = cond_raw_np[..., action_dim + 0]
            dir_y = cond_raw_np[..., action_dim + 1]
            speed = cond_raw_np[..., action_dim + 2]

            yaw, _ = signsplit._resolve_yaw_for_sign(
                trainer=trainer,
                batch=batch,
                ret=ret if isinstance(ret, dict) else {},
                label_time=str(args.label_time).lower(),
                pelvis_orientation_source=str(args.pelvis_orientation_source).lower(),
            )
            yaw_np = yaw.detach().cpu().numpy().astype(np.float64, copy=False)
            abs_dir = np.abs(-dir_x * np.sin(yaw_np) + dir_y * np.cos(yaw_np))

            contacts_np: Optional[np.ndarray] = None
            if torch.is_tensor(contacts) and contacts.dim() == 3 and int(contacts.shape[-1]) >= 2:
                contacts_np = contacts[..., :2].detach().cpu().numpy().astype(np.float64, copy=False)

            start_np = start.view(-1).detach().cpu().numpy().astype(np.int64, copy=False)
            clip_len_np = clip_len.view(-1).detach().cpu().numpy().astype(np.int64, copy=False)

            for b in range(B):
                clip_n = int(clip_len_np[b]) if int(clip_len_np[b]) > 0 else int(T)
                for t in steps:
                    t_i = int(t)
                    sic_abs = int((int(start_np[b]) + t_i) % max(1, clip_n))

                    row_geo = geo_np[b, t_i, :]
                    err_by_joint_rows.append(row_geo.astype(np.float32, copy=False))
                    sic_list.append(int(sic_abs))
                    abs_dir_list.append(float(abs_dir[b, t_i]))
                    speed_list.append(float(speed[b, t_i]))
                    batch_tag_list.append(str(diag_pt))

                    if contacts_np is not None:
                        cl = float(contacts_np[b, t_i, 0])
                        cr = float(contacts_np[b, t_i, 1])
                        phase = _phase_label(
                            cl,
                            cr,
                            stance_thr=float(args.contact_stance_thr),
                            flight_thr=float(args.contact_flight_thr),
                            dominance_margin=float(args.contact_dom_margin),
                        )
                    else:
                        phase = "phase_unknown"
                    phase_list.append(phase)

    if not err_by_joint_rows:
        raise SystemExit("[FATAL] empty aggregation after filtering")

    err_all = np.stack(err_by_joint_rows, axis=0).astype(np.float64, copy=False)
    sic_arr = np.asarray(sic_list, dtype=np.int64)
    phase_arr = np.asarray(phase_list, dtype=object)
    abs_dir_arr = np.asarray(abs_dir_list, dtype=np.float64)
    speed_arr = np.asarray(speed_list, dtype=np.float64)
    batch_arr = np.asarray(batch_tag_list, dtype=object)

    err_focus = err_all[:, focus_joint_idx].mean(axis=1)
    err_left = err_all[:, left_joint_idx].mean(axis=1)
    err_right = err_all[:, right_joint_idx].mean(axis=1)

    abs_bin_idx, abs_defs = _build_quantile_bins(abs_dir_arr, abs_quantiles, "abs")
    speed_bin_idx, speed_defs = _build_quantile_bins(speed_arr, speed_quantiles, "speed")

    abs_bin_name_by_id = {int(d["id"]): str(d["key"]) for d in abs_defs}
    speed_bin_name_by_id = {int(d["id"]): str(d["key"]) for d in speed_defs}

    abs_bin_name_arr = np.asarray(
        [abs_bin_name_by_id.get(int(i), "abs_invalid") if int(i) >= 0 else "abs_invalid" for i in abs_bin_idx],
        dtype=object,
    )
    speed_bin_name_arr = np.asarray(
        [speed_bin_name_by_id.get(int(i), "speed_invalid") if int(i) >= 0 else "speed_invalid" for i in speed_bin_idx],
        dtype=object,
    )

    global_joint_mean = err_all.mean(axis=0)
    global_stats = _summarize_group(
        name="global",
        mask=np.ones(err_focus.shape, dtype=bool),
        err_focus=err_focus,
        err_left=err_left,
        err_right=err_right,
        abs_dir=abs_dir_arr,
        speed=speed_arr,
    )

    corr_abs_m = np.isfinite(err_focus) & np.isfinite(abs_dir_arr)
    corr_speed_m = np.isfinite(err_focus) & np.isfinite(speed_arr)
    corr = {
        "err_vs_absdir": {
            "n": int(np.sum(corr_abs_m)),
            "pearson": _pearson(err_focus[corr_abs_m], abs_dir_arr[corr_abs_m]) if np.sum(corr_abs_m) > 1 else float("nan"),
            "spearman": _spearman(err_focus[corr_abs_m], abs_dir_arr[corr_abs_m]) if np.sum(corr_abs_m) > 1 else float("nan"),
        },
        "err_vs_speed": {
            "n": int(np.sum(corr_speed_m)),
            "pearson": _pearson(err_focus[corr_speed_m], speed_arr[corr_speed_m]) if np.sum(corr_speed_m) > 1 else float("nan"),
            "spearman": _spearman(err_focus[corr_speed_m], speed_arr[corr_speed_m]) if np.sum(corr_speed_m) > 1 else float("nan"),
        },
    }

    phase_keys = sorted(set(str(x) for x in phase_arr.tolist()))
    by_phase: List[Dict[str, Any]] = []
    for key in phase_keys:
        m = phase_arr == key
        by_phase.append(
            _summarize_group(
                name=key,
                mask=m,
                err_focus=err_focus,
                err_left=err_left,
                err_right=err_right,
                abs_dir=abs_dir_arr,
                speed=speed_arr,
            )
        )

    by_abs: List[Dict[str, Any]] = []
    for d in abs_defs:
        key = str(d["key"])
        m = abs_bin_name_arr == key
        by_abs.append(
            _summarize_group(
                name=key,
                mask=m,
                err_focus=err_focus,
                err_left=err_left,
                err_right=err_right,
                abs_dir=abs_dir_arr,
                speed=speed_arr,
            )
        )

    by_speed: List[Dict[str, Any]] = []
    for d in speed_defs:
        key = str(d["key"])
        m = speed_bin_name_arr == key
        by_speed.append(
            _summarize_group(
                name=key,
                mask=m,
                err_focus=err_focus,
                err_left=err_left,
                err_right=err_right,
                abs_dir=abs_dir_arr,
                speed=speed_arr,
            )
        )

    cross_phase_abs: List[Dict[str, Any]] = []
    for ph in phase_keys:
        for d in abs_defs:
            key = str(d["key"])
            m = (phase_arr == ph) & (abs_bin_name_arr == key)
            if not np.any(m):
                continue
            name = f"{key}__{ph}"
            cross_phase_abs.append(
                _summarize_group(
                    name=name,
                    mask=m,
                    err_focus=err_focus,
                    err_left=err_left,
                    err_right=err_right,
                    abs_dir=abs_dir_arr,
                    speed=speed_arr,
                )
            )
    cross_phase_abs.sort(
        key=lambda r: _safe_float((r.get("err_focus", {}) or {}).get("mean", float("nan")),),
        reverse=True,
    )

    # Additive decomposition (row-level)
    global_mu = _safe_float((global_stats.get("err_focus", {}) or {}).get("mean", float("nan")))
    phase_eff: Dict[str, float] = {}
    for r in by_phase:
        mu = _safe_float((r.get("err_focus", {}) or {}).get("mean", float("nan"))
)
        phase_eff[str(r.get("name", ""))] = float(mu - global_mu) if math.isfinite(mu) and math.isfinite(global_mu) else float("nan")
    abs_eff: Dict[str, float] = {}
    for r in by_abs:
        mu = _safe_float((r.get("err_focus", {}) or {}).get("mean", float("nan"))
)
        abs_eff[str(r.get("name", ""))] = float(mu - global_mu) if math.isfinite(mu) and math.isfinite(global_mu) else float("nan")
    speed_eff: Dict[str, float] = {}
    for r in by_speed:
        mu = _safe_float((r.get("err_focus", {}) or {}).get("mean", float("nan"))
)
        speed_eff[str(r.get("name", ""))] = float(mu - global_mu) if math.isfinite(mu) and math.isfinite(global_mu) else float("nan")

    pred = np.full(err_focus.shape, float("nan"), dtype=np.float64)
    for i in range(err_focus.shape[0]):
        ph = str(phase_arr[i])
        ab = str(abs_bin_name_arr[i])
        sp = str(speed_bin_name_arr[i])
        if not math.isfinite(global_mu):
            continue
        e0 = _safe_float(phase_eff.get(ph, float("nan")))
        e1 = _safe_float(abs_eff.get(ab, float("nan")))
        e2 = _safe_float(speed_eff.get(sp, float("nan")))
        if not (math.isfinite(e0) and math.isfinite(e1) and math.isfinite(e2)):
            continue
        pred[i] = float(global_mu + e0 + e1 + e2)

    finite_pred = np.isfinite(pred) & np.isfinite(err_focus)
    if np.any(finite_pred):
        y = err_focus[finite_pred]
        y_hat = pred[finite_pred]
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    else:
        r2 = float("nan")

    # SIC aggregation
    by_sic: List[Dict[str, Any]] = []
    sic_vals = sorted(set(int(x) for x in sic_arr.tolist()))
    for sic in sic_vals:
        m = sic_arr == int(sic)
        if not np.any(m):
            continue
        phase_counts = Counter(str(x) for x in phase_arr[m].tolist())
        abs_counts = Counter(str(x) for x in abs_bin_name_arr[m].tolist())
        speed_counts = Counter(str(x) for x in speed_bin_name_arr[m].tolist())
        phase_major = phase_counts.most_common(1)[0][0] if phase_counts else "NA"
        abs_major = abs_counts.most_common(1)[0][0] if abs_counts else "NA"
        speed_major = speed_counts.most_common(1)[0][0] if speed_counts else "NA"

        rows_batch = set(str(x) for x in batch_arr[m].tolist())
        joint_mean = err_all[m].mean(axis=0)
        top_idx = int(np.argmax(joint_mean)) if joint_mean.size > 0 else -1
        top_joint = str(bone_names[top_idx]) if 0 <= top_idx < len(bone_names) else "NA"

        obs = _safe_float(np.mean(err_focus[m]))
        phase_c = _safe_float(phase_eff.get(phase_major, float("nan")))
        abs_c = _safe_float(abs_eff.get(abs_major, float("nan")))
        speed_c = _safe_float(speed_eff.get(speed_major, float("nan")))
        pred_sic = float("nan")
        if math.isfinite(global_mu) and math.isfinite(phase_c) and math.isfinite(abs_c) and math.isfinite(speed_c):
            pred_sic = float(global_mu + phase_c + abs_c + speed_c)
        residual = obs - pred_sic if math.isfinite(obs) and math.isfinite(pred_sic) else float("nan")

        focus_joint_delta = {jn: float(joint_mean[name_to_idx[jn]] - global_joint_mean[name_to_idx[jn]]) for jn in focus_joints}
        focus_joint_rank = sorted(focus_joint_delta.items(), key=lambda kv: kv[1], reverse=True)
        focus_delta_vals = np.asarray([float(v) for _, v in focus_joint_rank], dtype=np.float64)
        focus_delta_abs_mean = float(np.mean(np.abs(focus_delta_vals))) if focus_delta_vals.size > 0 else float("nan")
        focus_delta_mean = float(np.mean(focus_delta_vals)) if focus_delta_vals.size > 0 else float("nan")
        focus_delta_cancel = (
            float(focus_delta_abs_mean - abs(focus_delta_mean))
            if math.isfinite(focus_delta_abs_mean) and math.isfinite(focus_delta_mean)
            else float("nan")
        )

        by_sic.append(
            {
                "sic": int(sic),
                "n_rows": int(np.sum(m)),
                "n_batches": int(len(rows_batch)),
                "phase_major": phase_major,
                "phase_counts": {k: int(v) for k, v in sorted(phase_counts.items())},
                "abs_bin_major": abs_major,
                "abs_bin_counts": {k: int(v) for k, v in sorted(abs_counts.items())},
                "speed_bin_major": speed_major,
                "speed_bin_counts": {k: int(v) for k, v in sorted(speed_counts.items())},
                "err_focus": _summarize(err_focus[m]),
                "err_left": _summarize(err_left[m]),
                "err_right": _summarize(err_right[m]),
                "err_rl_ratio": _summarize(np.asarray([_safe_ratio(float(r), float(l)) for r, l in zip(err_right[m], err_left[m])], dtype=np.float64)),
                "abs_dir_body_y": _summarize(abs_dir_arr[m]),
                "speed": _summarize(speed_arr[m]),
                "top_joint": top_joint,
                "top_joint_mean": float(joint_mean[top_idx]) if top_idx >= 0 else float("nan"),
                "residual_after_additive": float(residual),
                "additive_components": {
                    "global": float(global_mu),
                    "phase": float(phase_c),
                    "abs_dir": float(abs_c),
                    "speed": float(speed_c),
                    "pred": float(pred_sic),
                },
                "focus_joint_delta_vs_global": {k: float(v) for k, v in focus_joint_rank},
                "focus_delta_abs_mean": focus_delta_abs_mean,
                "focus_delta_mean": focus_delta_mean,
                "focus_delta_cancel": focus_delta_cancel,
            }
        )

    by_sic.sort(key=lambda r: int(r.get("sic", -1)))
    by_sic_map = {int(r["sic"]): r for r in by_sic}

    hotspots_rows: List[Dict[str, Any]] = []
    for s in hotspots:
        row = by_sic_map.get(int(s))
        if row is None:
            hotspots_rows.append({"sic": int(s), "missing": True})
        else:
            hotspots_rows.append(row)

    top_sic = sorted(
        by_sic,
        key=lambda r: _safe_float((r.get("err_focus", {}) or {}).get("mean", float("nan"))),
        reverse=True,
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
        "focus_joints": list(focus_joints),
        "left_bones": list(left_bones),
        "right_bones": list(right_bones),
        "contact_phase_config": {
            "stance_thr": float(args.contact_stance_thr),
            "flight_thr": float(args.contact_flight_thr),
            "dominance_margin": float(args.contact_dom_margin),
        },
        "abs_dir_bin_defs": abs_defs,
        "speed_bin_defs": speed_defs,
        "global": {
            "n_rows": int(err_focus.shape[0]),
            "n_sic": int(len(by_sic)),
            "err_focus": global_stats.get("err_focus", {}),
            "err_left": global_stats.get("err_left", {}),
            "err_right": global_stats.get("err_right", {}),
            "err_rl_ratio": global_stats.get("err_rl_ratio", {}),
            "abs_dir_body_y": global_stats.get("abs_dir_body_y", {}),
            "speed": global_stats.get("speed", {}),
            "corr": corr,
            "global_joint_mean": {str(n): float(global_joint_mean[i]) for i, n in enumerate(bone_names)},
        },
        "additive_model": {
            "formula": "err_focus ~= global + phase_effect + abs_dir_bin_effect + speed_bin_effect",
            "r2": float(r2),
            "phase_effect": {k: float(v) for k, v in phase_eff.items()},
            "abs_dir_effect": {k: float(v) for k, v in abs_eff.items()},
            "speed_effect": {k: float(v) for k, v in speed_eff.items()},
            "n_rows_used": int(np.sum(np.isfinite(pred) & np.isfinite(err_focus))),
        },
        "by_phase": by_phase,
        "by_abs_dir_bin": by_abs,
        "by_speed_bin": by_speed,
        "top_cross_phase_abs": cross_phase_abs,
        "by_sic": by_sic,
        "hotspots": hotspots_rows,
        "top_sic_by_focus": top_sic[:20],
    }

    out_json = Path(args.out_json).expanduser().resolve() if str(args.out_json).strip() else rho_path.with_name(
        rho_path.stem + "_direct_error_only_decomp.json"
    )
    out_md = Path(args.out_md).expanduser().resolve() if str(args.out_md).strip() else out_json.with_suffix(".md")

    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(out), encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
