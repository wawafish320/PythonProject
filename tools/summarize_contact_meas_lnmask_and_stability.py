#!/usr/bin/env python3
"""
LN-consistent bone ablation + top-bones stability stats for contact_meas_head.

Motivation
----------
`contact_meas_head` begins with a LayerNorm over the *concatenated* feature vector:

  x = [pose_hist_norm (L*J*6), angvel_norm (J*3)]  ->  LayerNorm(966)  ->  MLP  ->  logits(2)

Naively zeroing pose dims changes LayerNorm mean/variance, which "re-calibrates" the
remaining dims (LN coupling). This script measures bone-group causal influence in a
LayerNorm-consistent way by masking AFTER LayerNorm:

  y = LN(x) ; y[mask] = LN.bias[mask] ; logits = MLP_tail(y)

This keeps LN statistics identical to baseline, while removing information carried
by selected dims.

It also summarizes the stability of the top-K *positive* (push-to-contact) bones in
the newest pose_hist block (dt=0) across clips/conditions, via frequency + Jaccard.

Inputs
------
- Teacher-rollout JSONs: <root>/<cond>/*_teacher_pred.json
- Attribution JSONs (optional but recommended): <root>/<attrib-subdir>/<cond>/*_attrib.json

Outputs
-------
- <out>/contact_meas_lnmask_ablation.csv / .md
- <out>/contact_meas_topbones_stability.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.analyze_contact_meas_lag import _edge_times, _schmitt_state  # type: ignore  # noqa: E402
from tools.attrib_contact_meas_inputs import (  # type: ignore  # noqa: E402
    _apply_angvel_ablation,
    _apply_pose_hist_ablation,
    _load_contact_meas_head,
    _load_json,
    _load_bone_names,
    _reconstruct_pose_hist_buffer_norm,
    _span_to_slice,
)


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fmt(x: Optional[float], *, digits: int = 3) -> str:
    return "-" if x is None else f"{float(x):.{digits}f}"


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / float(len(xs))
    my = sum(ys) / float(len(ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / (vx**0.5 * vy**0.5))


def _write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in cols})


def _write_md(path: Path, rows: List[Dict[str, Any]], *, cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")


def _bone_category(name: str) -> str:
    n = str(name).lower()
    if any(k in n for k in ("foot", "ball", "toe")):
        return "foot"
    if any(k in n for k in ("calf", "thigh", "shin", "leg")):
        return "leg"
    if any(k in n for k in ("pelvis", "spine", "neck", "head", "clavicle", "bip")):
        return "spine"
    if any(
        k in n
        for k in (
            "upperarm",
            "lowerarm",
            "hand",
            "pinky",
            "ring",
            "index",
            "middle",
            "thumb",
            "foretwist",
            "armtwist",
            "shoulder",
            "wrist",
            "finger",
        )
    ):
        return "arm"
    return "other"


def _is_upper_body(name: str) -> bool:
    return _bone_category(name) in ("arm", "spine")


def _is_side_foot(name: str, *, channel: str) -> bool:
    n = str(name).lower()
    if _bone_category(n) != "foot":
        return False
    if channel.upper() == "R":
        return n.endswith("_r") or n.endswith("r") or "_r_" in n
    return n.endswith("_l") or n.endswith("l") or "_l_" in n


def _infer_pose_hist_len(pose_hist_dim: int, *, rot_y_slice: slice) -> int:
    pose_dim = int(rot_y_slice.stop - rot_y_slice.start)
    if pose_dim <= 0:
        return 0
    if pose_hist_dim % pose_dim != 0:
        return 0
    return int(pose_hist_dim // pose_dim)


def _build_ln_mask_indices(
    *,
    bone_names: Sequence[str],
    pose_hist_len: int,
    channel: str,
    group: str,
    pose_hist_dim: int,
    angvel_dim: int,
) -> np.ndarray:
    group = str(group).lower().strip()
    channel = str(channel).upper().strip()
    J = int(len(bone_names))
    if pose_hist_len <= 0:
        return np.zeros((0,), dtype=np.int64)
    if pose_hist_dim != pose_hist_len * J * 6:
        raise ValueError(f"pose_hist_dim mismatch: got {pose_hist_dim}, expected {pose_hist_len}*{J}*6={pose_hist_len*J*6}")
    if angvel_dim != J * 3:
        raise ValueError(f"angvel_dim mismatch: got {angvel_dim}, expected J*3={J*3}")

    # Special masks (not bone-scoped) for "only X" analyses.
    if group in ("pose_all", "angvel_only", "only_angvel"):
        return np.arange(0, pose_hist_dim, dtype=np.int64)
    if group in ("angvel_all", "pose_only", "only_pose"):
        return np.arange(pose_hist_dim, pose_hist_dim + angvel_dim, dtype=np.int64)
    if group in ("only_angvel_lower_body", "angvel_lower_body_only"):
        # Mask all pose dims, and mask angvel dims for bones NOT in lower-body (leg+foot).
        keep = {j for j, n in enumerate(bone_names) if _bone_category(n) in ("leg", "foot")}
        idx = list(range(0, pose_hist_dim))
        base = int(pose_hist_dim)
        for j in range(J):
            if j in keep:
                continue
            idx.extend(range(base + j * 3, base + j * 3 + 3))
        return np.asarray(sorted(set(idx)), dtype=np.int64)
    if group in ("only_angvel_foot", "angvel_foot_only"):
        # Mask all pose dims, and mask angvel dims for bones NOT in foot category (foot/ball/toe).
        keep = {j for j, n in enumerate(bone_names) if _bone_category(n) == "foot"}
        idx = list(range(0, pose_hist_dim))
        base = int(pose_hist_dim)
        for j in range(J):
            if j in keep:
                continue
            idx.extend(range(base + j * 3, base + j * 3 + 3))
        return np.asarray(sorted(set(idx)), dtype=np.int64)

    bone_sel: List[int] = []
    for j, name in enumerate(bone_names):
        cat = _bone_category(name)
        if group == "upper_body":
            if _is_upper_body(name):
                bone_sel.append(j)
        elif group == "arm":
            if cat == "arm":
                bone_sel.append(j)
        elif group == "spine":
            if cat == "spine":
                bone_sel.append(j)
        elif group == "leg":
            if cat == "leg":
                bone_sel.append(j)
        elif group == "lower_body":
            if cat in ("leg", "foot"):
                bone_sel.append(j)
        elif group == "foot":
            if _is_side_foot(name, channel=channel):
                bone_sel.append(j)
        elif group == "foot_any":
            if cat == "foot":
                bone_sel.append(j)
        else:
            raise ValueError(f"unknown group={group}")

    idx: List[int] = []
    for b in range(int(pose_hist_len)):
        base = b * (J * 6)
        for j in bone_sel:
            idx.extend(range(base + j * 6, base + j * 6 + 6))
    ang_base = int(pose_hist_dim)
    for j in bone_sel:
        idx.extend(range(ang_base + j * 3, ang_base + j * 3 + 3))
    return np.asarray(sorted(set(idx)), dtype=np.int64)


def _infer_probs(
    head: torch.nn.Module,
    x_seq: np.ndarray,
    *,
    mask_idx: Optional[np.ndarray] = None,
    mask_mode: str = "post_ln_bias",
) -> np.ndarray:
    """
    Args:
        head: nn.Sequential(LayerNorm, Linear, ReLU, Dropout, Linear)
        x_seq: (T,D) float32
        mask_idx: indices in feature dim to mask (applied after LN)
        mask_mode: currently supports 'post_ln_bias' only
    Returns:
        probs: (T,2) float64
    """
    x_t = torch.from_numpy(np.asarray(x_seq, dtype=np.float32))
    with torch.no_grad():
        if mask_idx is None or mask_idx.size == 0:
            logits = head(x_t)
        else:
            if str(mask_mode) != "post_ln_bias":
                raise ValueError(f"unsupported mask_mode={mask_mode}")
            ln = head[0]
            tail = head[1:]
            y = ln(x_t)
            if not isinstance(ln, torch.nn.LayerNorm):
                raise RuntimeError(f"head[0] is not LayerNorm, got {type(ln)}")
            bias = ln.bias.view(1, -1)
            y[:, mask_idx] = bias[:, mask_idx]
            logits = tail(y)
        probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy()


def _time_to_threshold_single(
    x: np.ndarray,
    edges: np.ndarray,
    *,
    thr: float,
    max_steps: int,
    direction: str,
) -> Tuple[List[float], List[Optional[int]]]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1)
    T = int(x.size)
    pred_at: List[float] = []
    dts: List[Optional[int]] = []
    if edges.size == 0 or T <= 0:
        return pred_at, dts
    thr = float(thr)
    for t0 in edges.tolist():
        t0 = int(t0)
        if t0 < 0 or t0 >= T:
            continue
        pred_at.append(float(x[t0]))
        end = min(T, t0 + int(max_steps) + 1)
        seg = x[t0:end]
        if seg.size == 0:
            dts.append(None)
            continue
        if direction == "ge":
            idx = np.where(seg >= thr)[0]
        else:
            idx = np.where(seg <= thr)[0]
        dts.append(int(idx[0]) if idx.size else None)
    return pred_at, dts


def _median_optional_int(values: Sequence[Optional[int]]) -> Optional[float]:
    clean = [int(v) for v in values if v is not None]
    if not clean:
        return None
    return float(np.median(np.asarray(clean, dtype=np.float64)))


def _median_float(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _event_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    channel: str,
    event: str,
    on_th: float,
    off_th: float,
    mid_th: float,
    window: int,
) -> Dict[str, Optional[float]]:
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ch = 0 if str(channel).upper() == "L" else 1
    s_gt = _schmitt_state(gt[:, ch], on_th=float(on_th), off_th=float(off_th))
    rise, fall = _edge_times(s_gt)
    edges = rise if str(event) == "rising" else fall

    if str(event) == "rising":
        pred_at, dts = _time_to_threshold_single(pred[:, ch], edges, thr=float(mid_th), max_steps=int(window), direction="ge")
    else:
        pred_at, dts = _time_to_threshold_single(pred[:, ch], edges, thr=float(mid_th), max_steps=int(window), direction="le")
    return {
        "pred_at_med": _median_float(pred_at),
        "dt_med": _median_optional_int(dts),
        "n_edges": float(int(edges.size)),
    }


def _reconstruct_x_seq(
    pred_json: Path,
    *,
    bundle_path: Path,
    pretrain_template: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[str]]:
    """
    Returns:
      x_seq: (T, D=pose_hist_dim+angvel_dim)
      gt:    (T, 2) contact GT (aux_inputs.contacts)
      meta:  dims/layout/ablation info
      bone_names: length J
    """
    d = _load_json(pred_json)
    aux = d.get("aux_inputs", {}) if isinstance(d.get("aux_inputs"), dict) else {}
    gt = np.asarray(aux.get("contacts", []), dtype=np.float32)
    if gt.ndim != 2 or gt.shape[1] < 2:
        raise ValueError("aux_inputs.contacts missing or malformed")
    T = int(gt.shape[0])

    teacher = d.get("teacher", {}) if isinstance(d.get("teacher"), dict) else {}
    state_norm = np.asarray(teacher.get("state_norm", []), dtype=np.float32)
    gt_target_norm = np.asarray(teacher.get("target_norm", []), dtype=np.float32)
    if state_norm.ndim != 2 or gt_target_norm.ndim != 2:
        raise ValueError("teacher.state_norm/target_norm missing or malformed")

    layouts = d.get("layouts", {}) if isinstance(d.get("layouts"), dict) else {}
    st_layout = layouts.get("state", {}) if isinstance(layouts.get("state"), dict) else {}
    out_layout = layouts.get("output", {}) if isinstance(layouts.get("output"), dict) else {}
    ang_x_slice = _span_to_slice(st_layout.get("BoneAngularVelocities"))
    rot_y_slice = _span_to_slice(out_layout.get("BoneRotations6D"))
    if rot_y_slice is None:
        raise ValueError("output layout missing BoneRotations6D")

    pose_hist_seq = np.asarray(aux.get("pose_hist_norm", []), dtype=np.float32)
    angvel_seq = np.asarray(aux.get("angvel_norm", []), dtype=np.float32)
    pose_hist_dim = int(pose_hist_seq.shape[1]) if pose_hist_seq.ndim == 2 else int(d.get("dims", {}).get("pose_hist", 0) or 0)
    angvel_dim = int(angvel_seq.shape[1]) if angvel_seq.ndim == 2 else int(d.get("dims", {}).get("angvel", 0) or 0)
    if pose_hist_dim <= 0 or angvel_dim <= 0:
        raise ValueError(f"invalid pose_hist_dim={pose_hist_dim}, angvel_dim={angvel_dim}")

    ab = d.get("ablation", {}) if isinstance(d.get("ablation"), dict) else {}
    pose_hist_source = str(ab.get("pose_hist_source", "seq") or "seq").lower().strip()
    angvel_source = str(ab.get("angvel_source", "seq") or "seq").lower().strip()

    if angvel_source == "state":
        if ang_x_slice is None:
            raise ValueError("state layout missing BoneAngularVelocities for angvel_source=state")
        angvel = state_norm[:, ang_x_slice]
    else:
        angvel = angvel_seq
    if angvel.ndim != 2 or angvel.shape[0] != T or angvel.shape[1] != angvel_dim:
        raise ValueError(f"angvel shape mismatch {angvel.shape} expected (T,{angvel_dim})")

    if pose_hist_source == "seq":
        pose_hist = pose_hist_seq
        pose_hist_len = _infer_pose_hist_len(pose_hist_dim, rot_y_slice=rot_y_slice)
    else:
        pretrain_tpl = _load_json(pretrain_template)
        bundle = _load_json(bundle_path)
        pose_hist_len = int(pretrain_tpl.get("pose_hist_len", 0) or 0)
        if pose_hist_len <= 0:
            raise ValueError("pretrain_template missing pose_hist_len")
        pose_hist = _reconstruct_pose_hist_buffer_norm(
            gt_target_norm=gt_target_norm,
            rot_y_slice=rot_y_slice,
            pose_hist_len=pose_hist_len,
            pose_hist_dim=pose_hist_dim,
            bundle_mu_y=np.asarray(bundle.get("MuY", []), dtype=np.float32),
            bundle_std_y=np.asarray(bundle.get("StdY", []), dtype=np.float32),
            pretrain_template=pretrain_tpl,
        )
    if pose_hist.ndim != 2 or pose_hist.shape[0] != T or pose_hist.shape[1] != pose_hist_dim:
        raise ValueError(f"pose_hist shape mismatch {pose_hist.shape} expected (T,{pose_hist_dim})")

    # Apply forward-time ablations (match Trainer._rollout_sequence).
    pose_hist_len_eff = int(_load_json(pretrain_template).get("pose_hist_len", 0) or 0)
    if pose_hist_len_eff <= 0:
        pose_hist_len_eff = pose_hist_len
    if pose_hist_len_eff <= 0:
        pose_hist_len_eff = _infer_pose_hist_len(pose_hist_dim, rot_y_slice=rot_y_slice)

    pose_hist = _apply_pose_hist_ablation(
        pose_hist,
        pose_hist_len=int(pose_hist_len_eff),
        mode=str(ab.get("pose_hist_ablation", "none")),
        keep_last=int(ab.get("pose_hist_keep_last", 1) or 1),
    )
    angvel = _apply_angvel_ablation(angvel, mode=str(ab.get("angvel_ablation", "none")))

    x_seq = np.concatenate([pose_hist, angvel], axis=-1).astype(np.float32, copy=False)

    # Bone names for masking / reporting.
    source_json = d.get("source_json")
    source_json_path = Path(source_json).expanduser().resolve() if isinstance(source_json, str) else None
    bone_names = _load_bone_names(source_json_path) if source_json_path else None
    if bone_names is None:
        # Try to infer from pose dims.
        pose_dim = int(rot_y_slice.stop - rot_y_slice.start)
        J = int(pose_dim // 6) if pose_dim % 6 == 0 else 0
        bone_names = [f"bone_{i}" for i in range(J)]

    meta: Dict[str, Any] = {
        "clip": str(d.get("clip", pred_json.stem.replace("_teacher_pred", ""))),
        "model": str(d.get("model", "")),
        "pose_hist_source": pose_hist_source,
        "angvel_source": angvel_source,
        "pose_hist_dim": int(pose_hist_dim),
        "angvel_dim": int(angvel_dim),
        "pose_hist_len": int(pose_hist_len_eff),
        "rot_y_slice": {"start": int(rot_y_slice.start), "stop": int(rot_y_slice.stop)},
        "ablation": dict(ab),
    }
    return x_seq, gt, meta, list(bone_names)


def _find_pred_jsons(root: Path, cond: str) -> List[Path]:
    patt = str(root / cond / "*_teacher_pred.json")
    return sorted(Path(p) for p in glob.glob(patt))


def _find_attrib_jsons(root: Path, attrib_subdir: str, cond: str, *, channel: str, event: str) -> List[Path]:
    patt = str(root / attrib_subdir / cond / f"*_{channel}_{event}_t*_attrib.json")
    return sorted(Path(p) for p in glob.glob(patt))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return float(inter / max(union, 1))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LN-consistent (post-LayerNorm) bone ablation + top-bones stability stats for contact_meas_head.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--root", type=str, required=True, help="Batch root (contains <cond>/ and <attrib-subdir>/).")
    ap.add_argument("--conds", nargs="+", required=True, help="Condition dirs under --root.")
    ap.add_argument("--attrib-subdir", type=str, default="_attrib_batch2", help="Attribution subdir under --root.")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json", help="Normalization bundle (MuY/StdY).")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json", help="Pretrain template for pose_hist buffer reconstruction.")
    ap.add_argument("--event", type=str, default="falling", choices=("rising", "falling"), help="Which GT edge to evaluate.")
    ap.add_argument("--channel", type=str, default="R", choices=("L", "R"), help="Which contact channel to evaluate.")
    ap.add_argument("--on-th", type=float, default=0.8, help="GT Schmitt ON threshold.")
    ap.add_argument("--off-th", type=float, default=0.1, help="GT Schmitt OFF threshold.")
    ap.add_argument("--mid-th", type=float, default=0.55, help="Mid threshold for time-to metrics.")
    ap.add_argument("--time-window", type=int, default=30, help="Max steps for time-to metrics.")
    ap.add_argument("--topk", type=int, default=8, help="Top-K *positive* bones at dt=0 to track for stability.")
    ap.add_argument(
        "--scan-groups",
        nargs="+",
        default=[
            "upper_body",
            "arm",
            "spine",
            "leg",
            "foot",
            "foot_any",
            "lower_body",
            # LN-consistent "input redesign" probes:
            "pose_all",  # == angvel-only (keep all angvel)
            "only_angvel_lower_body",
            "only_angvel_foot",
            "angvel_all",  # == pose-only
        ],
        help="Extra LN-consistent (post-LN) mask groups to scan (pose_hist(all blocks)+angvel for selected bones).",
    )
    ap.add_argument("--long-tail-dt", type=float, default=15.0, help="Baseline dt>=this treated as long-tail subset for stats.")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: <root>/<attrib-subdir>).")
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root)).resolve()
    attrib_subdir = str(args.attrib_subdir)
    out_dir = Path(os.path.expanduser(args.out)).resolve() if args.out else (root / attrib_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = Path(os.path.expanduser(args.bundle)).resolve()
    pretrain_tpl_path = Path(os.path.expanduser(args.pretrain_template)).resolve()

    conds = [c for c in (str(x).strip() for x in args.conds) if c and (root / c).is_dir()]
    if not conds:
        raise SystemExit("[FATAL] --conds expanded to empty list (no existing dirs).")

    channel = str(args.channel).upper()
    event = str(args.event)
    topk = int(max(0, int(args.topk)))

    # Load attrib jsons per cond (optional; used for stability stats).
    attrib_by_cond_clip: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for cond in conds:
        files = _find_attrib_jsons(root, attrib_subdir, cond, channel=channel, event=event)
        for p in files:
            d = _load_json(p)
            if isinstance(d, dict) and d.get("clip"):
                attrib_by_cond_clip[cond][str(d["clip"])] = d

    # Cache heads by ckpt path.
    head_cache: Dict[str, torch.nn.Module] = {}

    rows: List[Dict[str, Any]] = []
    # For stability stats:
    top_pos_sets: Dict[str, Dict[str, Set[str]]] = defaultdict(dict)  # clip -> cond -> set(bones)
    per_cond_bone_freq: Dict[str, Counter] = defaultdict(Counter)
    per_cond_corr_acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for cond in conds:
        pred_files = _find_pred_jsons(root, cond)
        if not pred_files:
            raise SystemExit(f"[FATAL] No *_teacher_pred.json under {root/cond}")

        # Lag summary for reference dt/p (optional but useful for sanity).
        lag_path = root / cond / "contact_meas_lag_summary.json"
        lag_rows = {}
        if lag_path.is_file():
            lag = _load_json(lag_path)
            for r in lag.get("rows", []) if isinstance(lag, dict) else []:
                if isinstance(r, dict) and r.get("clip"):
                    lag_rows[str(r["clip"])] = r

        for pred_json in pred_files:
            x_seq, gt, meta, bone_names = _reconstruct_x_seq(pred_json, bundle_path=bundle_path, pretrain_template=pretrain_tpl_path)
            clip = str(meta.get("clip", pred_json.stem.replace("_teacher_pred", "")))

            ckpt_path = str(meta.get("model", "")).strip()
            if not ckpt_path:
                raise SystemExit(f"[FATAL] missing model path in {pred_json}")
            if ckpt_path not in head_cache:
                head_cache[ckpt_path] = _load_contact_meas_head(Path(ckpt_path).expanduser().resolve())
            head = head_cache[ckpt_path]

            pose_hist_dim = int(meta["pose_hist_dim"])
            angvel_dim = int(meta["angvel_dim"])
            pose_hist_len = int(meta["pose_hist_len"])

            # Baseline and LN-consistent masks.
            probs_full = _infer_probs(head, x_seq)
            mask_upper = _build_ln_mask_indices(
                bone_names=bone_names,
                pose_hist_len=pose_hist_len,
                channel=channel,
                group="upper_body",
                pose_hist_dim=pose_hist_dim,
                angvel_dim=angvel_dim,
            )
            mask_foot = _build_ln_mask_indices(
                bone_names=bone_names,
                pose_hist_len=pose_hist_len,
                channel=channel,
                group="foot",
                pose_hist_dim=pose_hist_dim,
                angvel_dim=angvel_dim,
            )
            probs_upper = _infer_probs(head, x_seq, mask_idx=mask_upper, mask_mode="post_ln_bias")
            probs_foot = _infer_probs(head, x_seq, mask_idx=mask_foot, mask_mode="post_ln_bias")

            m_full = _event_metrics(gt, probs_full, channel=channel, event=event, on_th=float(args.on_th), off_th=float(args.off_th), mid_th=float(args.mid_th), window=int(args.time_window))
            m_upper = _event_metrics(gt, probs_upper, channel=channel, event=event, on_th=float(args.on_th), off_th=float(args.off_th), mid_th=float(args.mid_th), window=int(args.time_window))
            m_foot = _event_metrics(gt, probs_foot, channel=channel, event=event, on_th=float(args.on_th), off_th=float(args.off_th), mid_th=float(args.mid_th), window=int(args.time_window))

            # Pull lag summary medians for sanity (should match full metrics).
            lag_r = lag_rows.get(clip, {})
            lag_p = _as_float(lag_r.get(f"{channel}_fall_pred_at_med")) if event == "falling" else _as_float(lag_r.get(f"{channel}_rise_pred_at_med"))
            lag_dt = _as_float(lag_r.get(f"{channel}_fall_time_to_mid_med")) if event == "falling" else _as_float(lag_r.get(f"{channel}_rise_time_to_mid_med"))

            # Stability stats from attrib json (dt=0, positive bones).
            attrib = attrib_by_cond_clip.get(cond, {}).get(clip)
            top_pos_list: List[str] = []
            top_pos_set: Set[str] = set()
            upper_pos_frac_dt0 = None
            dt0_pos_frac_all = None
            if isinstance(attrib, dict):
                pose = attrib.get("pose", {}) if isinstance(attrib.get("pose"), dict) else {}
                pose_contrib = pose.get("contrib", [])
                if isinstance(pose_contrib, list) and pose_contrib:
                    pose_mat = np.asarray(pose_contrib, dtype=np.float64)  # (L,J)
                    if pose_mat.ndim == 2 and pose_mat.shape[1] == len(bone_names):
                        dt0_block = pose_mat.shape[0] - 1
                        row = pose_mat[dt0_block]
                        pos = [(bone_names[j], float(row[j])) for j in range(len(bone_names)) if float(row[j]) > 0.0]
                        pos_sorted = sorted(pos, key=lambda kv: kv[1], reverse=True)[:topk]
                        top_pos_list = [n for n, _ in pos_sorted]
                        top_pos_set = set(top_pos_list)
                        for n in top_pos_list:
                            per_cond_bone_freq[cond][n] += 1
                        top_pos_sets[clip][cond] = top_pos_set

                        dt0_pos_total = float(sum(v for _, v in pos))
                        dt0_pos_upper = float(sum(v for n, v in pos if _is_upper_body(n)))
                        upper_pos_frac_dt0 = float(dt0_pos_upper / max(dt0_pos_total, 1e-12))

                        # Concentration of positive contrib in dt=0 block.
                        total_pos_all = float(np.maximum(pose_mat, 0.0).sum())
                        dt0_pos_frac_all = float(dt0_pos_total / max(total_pos_all, 1e-12))

                        # Corr accumulators (dt vs upper-body evidence).
                        dt_mid = _as_float(lag_r.get(f"{channel}_fall_time_to_mid_med")) if event == "falling" else None
                        if dt_mid is not None:
                            per_cond_corr_acc[cond]["dt_mid"].append(float(dt_mid))
                            per_cond_corr_acc[cond]["upper_pos_frac_dt0"].append(float(upper_pos_frac_dt0))
                            per_cond_corr_acc[cond]["dt0_pos_frac_all"].append(float(dt0_pos_frac_all))

            row_out = {
                "cond": cond,
                "clip": clip,
                "p_med": _fmt(m_full.get("pred_at_med")),
                "dt_med": _fmt(m_full.get("dt_med"), digits=1),
                "p_upper_mask": _fmt(m_upper.get("pred_at_med")),
                "dt_upper_mask": _fmt(m_upper.get("dt_med"), digits=1),
                "p_foot_mask": _fmt(m_foot.get("pred_at_med")),
                "dt_foot_mask": _fmt(m_foot.get("dt_med"), digits=1),
                "dp_upper": _fmt((m_upper.get("pred_at_med") - m_full.get("pred_at_med")) if (m_upper.get("pred_at_med") is not None and m_full.get("pred_at_med") is not None) else None),
                "dp_foot": _fmt((m_foot.get("pred_at_med") - m_full.get("pred_at_med")) if (m_foot.get("pred_at_med") is not None and m_full.get("pred_at_med") is not None) else None),
                "ddt_upper": _fmt((m_upper.get("dt_med") - m_full.get("dt_med")) if (m_upper.get("dt_med") is not None and m_full.get("dt_med") is not None) else None, digits=1),
                "ddt_foot": _fmt((m_foot.get("dt_med") - m_full.get("dt_med")) if (m_foot.get("dt_med") is not None and m_full.get("dt_med") is not None) else None, digits=1),
                "lag_p_med": _fmt(lag_p),
                "lag_dt_med": _fmt(lag_dt, digits=1),
                "top_pos_dt0": ", ".join(top_pos_list),
                "upper_pos_frac_dt0": _fmt(upper_pos_frac_dt0),
                "dt0_pos_frac_all": _fmt(dt0_pos_frac_all),
                "pred_json": str(pred_json),
            }
            rows.append(row_out)

    # Sort rows for readability: cond then dt desc.
    def _sort_key(r: Dict[str, Any]) -> Tuple[str, float]:
        dt = _as_float(r.get("dt_med"))
        return str(r.get("cond", "")), -(dt if dt is not None else -1e9)

    rows_sorted = sorted(rows, key=_sort_key)

    cols_csv = [
        "cond",
        "clip",
        "p_med",
        "dt_med",
        "p_upper_mask",
        "dt_upper_mask",
        "p_foot_mask",
        "dt_foot_mask",
        "dp_upper",
        "dp_foot",
        "ddt_upper",
        "ddt_foot",
        "upper_pos_frac_dt0",
        "dt0_pos_frac_all",
        "top_pos_dt0",
        "lag_p_med",
        "lag_dt_med",
        "pred_json",
    ]

    out_csv = out_dir / "contact_meas_lnmask_ablation.csv"
    _write_csv(out_csv, rows_sorted, cols_csv)

    out_md = out_dir / "contact_meas_lnmask_ablation.md"
    md_cols = [
        "cond",
        "clip",
        "p_med",
        "dt_med",
        "p_upper_mask",
        "dt_upper_mask",
        "p_foot_mask",
        "dt_foot_mask",
        "dp_upper",
        "dp_foot",
        "upper_pos_frac_dt0",
        "top_pos_dt0",
    ]
    _write_md(out_md, rows_sorted, cols=md_cols)

    # Stability stats JSON.
    cond_pairs = [(a, b) for i, a in enumerate(conds) for b in conds[i + 1 :]]
    jaccard_by_clip: Dict[str, Dict[str, float]] = {}
    for clip, by_cond in top_pos_sets.items():
        one: Dict[str, float] = {}
        for a, b in cond_pairs:
            one[f"{a}__vs__{b}"] = _jaccard(by_cond.get(a, set()), by_cond.get(b, set()))
        jaccard_by_clip[clip] = one

    jaccard_mean: Dict[str, float] = {}
    for a, b in cond_pairs:
        vals = [jaccard_by_clip[c].get(f"{a}__vs__{b}", 0.0) for c in jaccard_by_clip.keys()]
        jaccard_mean[f"{a}__vs__{b}"] = float(sum(vals) / max(len(vals), 1))

    per_cond_stats: Dict[str, Any] = {}
    for cond in conds:
        freq = per_cond_bone_freq.get(cond, Counter())
        top_freq = dict(freq.most_common(15))

        dt_mid = per_cond_corr_acc[cond].get("dt_mid", [])
        upper_frac = per_cond_corr_acc[cond].get("upper_pos_frac_dt0", [])
        dt0_frac = per_cond_corr_acc[cond].get("dt0_pos_frac_all", [])
        per_cond_stats[cond] = {
            "n": int(len(dt_mid)),
            "top_pos_dt0_freq": top_freq,
            "corr_dt_mid__upper_pos_frac_dt0": _pearson(dt_mid, upper_frac) if len(dt_mid) == len(upper_frac) else None,
            "corr_dt_mid__dt0_pos_frac_all": _pearson(dt_mid, dt0_frac) if len(dt_mid) == len(dt0_frac) else None,
        }

    out_stats = {
        "root": str(root),
        "conds": conds,
        "event": event,
        "channel": channel,
        "topk_pos_dt0": topk,
        "mask_mode": "post_ln_bias",
        "groups": ["upper_body", "foot"],
        "jaccard_by_clip": jaccard_by_clip,
        "jaccard_mean": jaccard_mean,
        "per_cond": per_cond_stats,
    }
    out_json = out_dir / "contact_meas_topbones_stability.json"
    out_json.write_text(json.dumps(out_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # ===== Group scan (post-LN masks; LN-consistent) =====
    scan_groups = [str(g).strip() for g in (args.scan_groups or []) if str(g).strip()]
    scan_groups = [g for g in scan_groups if g not in ("", "none")]
    if scan_groups:
        scan_rows: List[Dict[str, Any]] = []
        for cond in conds:
            pred_files = _find_pred_jsons(root, cond)
            for pred_json in pred_files:
                x_seq, gt, meta, bone_names = _reconstruct_x_seq(
                    pred_json,
                    bundle_path=bundle_path,
                    pretrain_template=pretrain_tpl_path,
                )
                clip = str(meta.get("clip", pred_json.stem.replace("_teacher_pred", "")))

                ckpt_path = str(meta.get("model", "")).strip()
                if not ckpt_path:
                    continue
                if ckpt_path not in head_cache:
                    head_cache[ckpt_path] = _load_contact_meas_head(Path(ckpt_path).expanduser().resolve())
                head = head_cache[ckpt_path]

                probs_full = _infer_probs(head, x_seq)
                m_full = _event_metrics(
                    gt,
                    probs_full,
                    channel=channel,
                    event=event,
                    on_th=float(args.on_th),
                    off_th=float(args.off_th),
                    mid_th=float(args.mid_th),
                    window=int(args.time_window),
                )
                p0 = m_full.get("pred_at_med")
                dt0 = m_full.get("dt_med")

                pose_hist_dim = int(meta["pose_hist_dim"])
                angvel_dim = int(meta["angvel_dim"])
                pose_hist_len = int(meta["pose_hist_len"])
                dims_per_bone = int(pose_hist_len * 6 + 3) if pose_hist_len > 0 else 0

                for g in scan_groups:
                    mask_idx = _build_ln_mask_indices(
                        bone_names=bone_names,
                        pose_hist_len=pose_hist_len,
                        channel=channel,
                        group=g,
                        pose_hist_dim=pose_hist_dim,
                        angvel_dim=angvel_dim,
                    )
                    probs_g = _infer_probs(head, x_seq, mask_idx=mask_idx, mask_mode="post_ln_bias")
                    m_g = _event_metrics(
                        gt,
                        probs_g,
                        channel=channel,
                        event=event,
                        on_th=float(args.on_th),
                        off_th=float(args.off_th),
                        mid_th=float(args.mid_th),
                        window=int(args.time_window),
                    )
                    p1 = m_g.get("pred_at_med")
                    dt1 = m_g.get("dt_med")
                    dp = (p1 - p0) if (p0 is not None and p1 is not None) else None
                    ddt = (dt1 - dt0) if (dt0 is not None and dt1 is not None) else None

                    n_dims = int(mask_idx.size)
                    n_bones = (int(n_dims // dims_per_bone) if dims_per_bone > 0 else None)

                    scan_rows.append(
                        {
                            "cond": cond,
                            "clip": clip,
                            "group": g,
                            "p": p0,
                            "dt": dt0,
                            "p_mask": p1,
                            "dt_mask": dt1,
                            "dp": dp,
                            "ddt": ddt,
                            "n_bones": n_bones,
                            "n_dims": n_dims,
                            "pred_json": str(pred_json),
                        }
                    )

        scan_rows_sorted = sorted(
            scan_rows,
            key=lambda rr: (
                str(rr.get("cond", "")),
                str(rr.get("group", "")),
                -(float(rr["dt"]) if rr.get("dt") is not None else -1e9),
            ),
        )
        scan_csv = out_dir / "contact_meas_lnmask_group_scan.csv"
        scan_cols = ["cond", "clip", "group", "p", "dt", "p_mask", "dt_mask", "dp", "ddt", "n_bones", "n_dims", "pred_json"]
        _write_csv(scan_csv, scan_rows_sorted, scan_cols)

        scan_md = out_dir / "contact_meas_lnmask_group_scan.md"
        scan_md_cols = ["cond", "clip", "group", "p", "dt", "p_mask", "dt_mask", "dp", "ddt", "n_bones"]
        scan_rows_md: List[Dict[str, Any]] = []
        for rr in scan_rows_sorted:
            scan_rows_md.append(
                {
                    **rr,
                    "p": _fmt(rr.get("p")),
                    "dt": _fmt(rr.get("dt"), digits=1),
                    "p_mask": _fmt(rr.get("p_mask")),
                    "dt_mask": _fmt(rr.get("dt_mask"), digits=1),
                    "dp": _fmt(rr.get("dp")),
                    "ddt": _fmt(rr.get("ddt"), digits=1),
                }
            )
        _write_md(scan_md, scan_rows_md, cols=scan_md_cols)

        # Aggregate stats
        by_cg: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for rr in scan_rows:
            by_cg[(str(rr.get("cond", "")), str(rr.get("group", "")))].append(rr)

        scan_stats: Dict[str, Any] = {
            "event": event,
            "channel": channel,
            "mid_th": float(args.mid_th),
            "time_window": int(args.time_window),
            "long_tail_dt": float(args.long_tail_dt),
            "scan_groups": scan_groups,
            "per_cond_group": {},
            "worst_regressions": [],
            "best_improvements_by_case": [],
        }

        # worst regressions (by ddt then dp)
        reg = [rr for rr in scan_rows if rr.get("ddt") is not None and float(rr["ddt"]) > 0.0]
        reg = sorted(reg, key=lambda rr: (float(rr["ddt"]), float(rr["dp"]) if rr.get("dp") is not None else 0.0), reverse=True)[:20]
        scan_stats["worst_regressions"] = [
            {"cond": rr["cond"], "clip": rr["clip"], "group": rr["group"], "ddt": rr["ddt"], "dp": rr["dp"]} for rr in reg
        ]

        # best improvements per (cond, clip): min ddt and min dp
        by_case: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for rr in scan_rows:
            by_case[(str(rr.get("cond", "")), str(rr.get("clip", "")))].append(rr)
        for (cond, clip), rs in by_case.items():
            rs_ddt = [rr for rr in rs if rr.get("ddt") is not None]
            rs_dp = [rr for rr in rs if rr.get("dp") is not None]
            best_ddt = min(rs_ddt, key=lambda rr: float(rr["ddt"])) if rs_ddt else None
            best_dp = min(rs_dp, key=lambda rr: float(rr["dp"])) if rs_dp else None
            scan_stats["best_improvements_by_case"].append(
                {
                    "cond": cond,
                    "clip": clip,
                    "baseline_dt": (rs[0].get("dt") if rs else None),
                    "baseline_p": (rs[0].get("p") if rs else None),
                    "best_ddt_group": (best_ddt.get("group") if best_ddt else None),
                    "best_ddt": (best_ddt.get("ddt") if best_ddt else None),
                    "best_dp_group": (best_dp.get("group") if best_dp else None),
                    "best_dp": (best_dp.get("dp") if best_dp else None),
                }
            )

        for (cond, g), rs in by_cg.items():
            dt_list: List[float] = []
            dp_list: List[float] = []
            ddt_list: List[float] = []
            dp_long: List[float] = []
            ddt_long: List[float] = []
            for rr in rs:
                dt0 = rr.get("dt")
                dp = rr.get("dp")
                ddt = rr.get("ddt")
                if dt0 is not None and dp is not None:
                    dt_list.append(float(dt0))
                    dp_list.append(float(dp))
                if ddt is not None:
                    ddt_list.append(float(ddt))
                if dt0 is not None and float(dt0) >= float(args.long_tail_dt):
                    if dp is not None:
                        dp_long.append(float(dp))
                    if ddt is not None:
                        ddt_long.append(float(ddt))

            scan_stats["per_cond_group"][f"{cond}::{g}"] = {
                "n": int(len(rs)),
                "dp_mean": (float(np.mean(dp_list)) if dp_list else None),
                "ddt_mean": (float(np.mean(ddt_list)) if ddt_list else None),
                "dp_mean_long_tail": (float(np.mean(dp_long)) if dp_long else None),
                "ddt_mean_long_tail": (float(np.mean(ddt_long)) if ddt_long else None),
                "corr_dt__neg_dp": (_pearson(dt_list, [-x for x in dp_list]) if len(dt_list) == len(dp_list) and len(dt_list) >= 2 else None),
            }

        scan_stats_json = out_dir / "contact_meas_lnmask_group_scan_stats.json"
        scan_stats_json.write_text(json.dumps(scan_stats, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[OK] wrote {scan_csv}")
        print(f"[OK] wrote {scan_md}")
        print(f"[OK] wrote {scan_stats_json}")

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_md}")
    print(f"[OK] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
