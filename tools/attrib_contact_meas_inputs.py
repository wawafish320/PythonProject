#!/usr/bin/env python3
"""
Input attribution for contact_meas_head at GT event edges.

This script is meant to "pin down" which joints / which pose_hist blocks push the
contact_meas logits at a GT rising/falling edge, using simple first-order attribution:

  - grad_norm: || d logit / d x || (per bone / per block)
  - contrib:   sum( (d logit / d x) * x )  (signed; local Taylor along input direction)

It supports reconstructing the *actual* rollout inputs for:
  - angvel_source: {state, seq}
  - pose_hist_source: {buffer, seq}

For pose_hist_source=buffer, we reconstruct the Trainer rolling buffer from
teacher.target_norm (GT Y_norm) + bundle MuY/StdY, then re-apply the pose-history
tanh+zscore normalizer using models/pretrain_template.json.

Example (Walk_F, GT falling, channel R):
  python tools/attrib_contact_meas_inputs.py \
    --pred-json debug_output/_tmp_teacher_debug/_batch_eventlag/baseline/Walk_F_teacher_pred.json \
    --event falling --channel R --out debug_output/_tmp_teacher_debug/_batch_eventlag/_attrib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn

try:
    from tools.analyze_contact_meas_lag import _edge_times, _schmitt_state  # type: ignore
except Exception:  # pragma: no cover
    from analyze_contact_meas_lag import _edge_times, _schmitt_state  # type: ignore


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _span_to_slice(span_obj) -> Optional[slice]:
    if span_obj is None:
        return None
    if isinstance(span_obj, (list, tuple)) and len(span_obj) == 2:
        start = int(span_obj[0])
        length = int(span_obj[1])
        return slice(start, start + length)
    if isinstance(span_obj, dict):
        start = int(span_obj.get("start", 0) or 0)
        size = int(span_obj.get("size", 0) or 0)
        return slice(start, start + size) if size > 0 else None
    return None


def _pick_event_index(
    gt_contacts: np.ndarray,
    *,
    ch: int,
    on_th: float,
    off_th: float,
    event: str,
    which: int,
) -> Optional[int]:
    s = _schmitt_state(gt_contacts[:, ch], on_th=float(on_th), off_th=float(off_th))
    rise, fall = _edge_times(s)
    edges = rise if event == "rising" else fall
    if edges.size == 0:
        return None
    which = int(which)
    which = max(0, min(which, int(edges.size) - 1))
    return int(edges[which])


def _load_contact_meas_head(ckpt_path: Path) -> nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    if not isinstance(sd, dict):
        raise SystemExit(f"[FATAL] Unsupported checkpoint format: {type(sd)}")

    keys = [k for k in sd.keys() if str(k).startswith("contact_meas_head.")]
    if not keys:
        raise SystemExit("[FATAL] checkpoint has no contact_meas_head.* weights.")

    in_dim = int(sd["contact_meas_head.0.weight"].shape[0])
    hid = int(sd["contact_meas_head.1.weight"].shape[0])
    out_dim = int(sd["contact_meas_head.4.weight"].shape[0])

    head = nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hid),
        nn.ReLU(),
        nn.Dropout(0.0),
        nn.Linear(hid, out_dim),
    )
    sub = {str(k).replace("contact_meas_head.", ""): v for k, v in sd.items() if str(k).startswith("contact_meas_head.")}
    head.load_state_dict(sub, strict=True)
    head.eval()
    return head


def _reconstruct_pose_hist_buffer_norm(
    *,
    gt_target_norm: np.ndarray,
    rot_y_slice: slice,
    pose_hist_len: int,
    pose_hist_dim: int,
    bundle_mu_y: np.ndarray,
    bundle_std_y: np.ndarray,
    pretrain_template: Dict[str, object],
) -> np.ndarray:
    gt_target_norm = np.asarray(gt_target_norm, dtype=np.float32)
    if gt_target_norm.ndim != 2:
        raise ValueError(f"gt_target_norm must be (T,D); got {gt_target_norm.shape}")
    T, Dy = int(gt_target_norm.shape[0]), int(gt_target_norm.shape[1])

    mu_y = np.asarray(bundle_mu_y, dtype=np.float32).reshape(1, -1)
    std_y = np.asarray(bundle_std_y, dtype=np.float32).reshape(1, -1)
    if mu_y.shape[1] < Dy or std_y.shape[1] < Dy:
        raise ValueError(f"bundle MuY/StdY smaller than Dy={Dy}: mu={mu_y.shape} std={std_y.shape}")
    mu_y = mu_y[:, :Dy]
    std_y = std_y[:, :Dy]
    y_raw = gt_target_norm * std_y + mu_y  # (T,Dy)  == y_out_features

    rot_raw = y_raw[:, rot_y_slice]
    pose_hist_len = int(pose_hist_len)
    pose_hist_dim = int(pose_hist_dim)
    if pose_hist_len <= 0 or pose_hist_dim <= 0:
        return np.zeros((T, 0), dtype=np.float32)
    if pose_hist_dim % pose_hist_len != 0:
        raise ValueError(f"pose_hist_dim={pose_hist_dim} not divisible by pose_hist_len={pose_hist_len}")
    pose_dim = pose_hist_dim // pose_hist_len
    if rot_raw.shape[1] != pose_dim:
        raise ValueError(f"rot_raw dim={rot_raw.shape[1]} != pose_dim={pose_dim} (pose_hist_dim/L)")

    # Rebuild rolling buffer: blocks are [older ... newer], ending at time t (i.e., include current y_raw[t]).
    offsets = np.arange(-pose_hist_len + 1, 1, dtype=np.int64)  # [-L+1 ... 0]
    idx = np.arange(T, dtype=np.int64)[:, None] + offsets[None, :]
    np.clip(idx, 0, T - 1, out=idx)
    hist = rot_raw[idx]  # (T, L, pose_dim)
    buf_raw = hist.reshape(T, pose_hist_dim)

    scales = np.asarray(pretrain_template.get("tanh_scales_pose_hist", []), dtype=np.float32)
    mu = np.asarray(pretrain_template.get("MuPoseHist", []), dtype=np.float32)
    std = np.asarray(pretrain_template.get("StdPoseHist", []), dtype=np.float32)
    if scales.size != pose_hist_dim or mu.size != pose_hist_dim or std.size != pose_hist_dim:
        raise ValueError(
            "pretrain_template pose_hist stats size mismatch: "
            f"scales={scales.size} mu={mu.size} std={std.size} vs pose_hist_dim={pose_hist_dim}"
        )

    z = np.tanh(buf_raw / scales.reshape(1, -1))
    z = (z - mu.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1e-6)
    return z.astype(np.float32, copy=False)


def _apply_pose_hist_ablation(
    pose_hist: np.ndarray,
    *,
    pose_hist_len: int,
    mode: str,
    keep_last: int,
) -> np.ndarray:
    pose_hist = np.asarray(pose_hist, dtype=np.float32)
    mode = str(mode or "none").lower().strip()
    if pose_hist.ndim != 2 or pose_hist.size == 0:
        return pose_hist
    if mode in ("", "none"):
        return pose_hist
    if mode == "zero":
        return np.zeros_like(pose_hist)
    L = int(pose_hist_len)
    if L <= 0:
        return pose_hist
    D = int(pose_hist.shape[1])
    if D % L != 0:
        return pose_hist
    pose_dim = D // L
    hist = pose_hist.reshape(pose_hist.shape[0], L, pose_dim).copy()
    if mode == "keep_last":
        k = max(1, min(L, int(keep_last)))
        if L - k > 0:
            hist[:, : L - k, :] = 0.0
        return hist.reshape(pose_hist.shape[0], D)
    if mode == "replicate_last":
        src = hist[:, -1:, :].copy()
        hist[:, :, :] = src
        return hist.reshape(pose_hist.shape[0], D)
    if mode == "replicate_oldest":
        src = hist[:, :1, :].copy()
        hist[:, :, :] = src
        return hist.reshape(pose_hist.shape[0], D)
    return pose_hist


def _apply_angvel_ablation(angvel: np.ndarray, *, mode: str) -> np.ndarray:
    angvel = np.asarray(angvel, dtype=np.float32)
    mode = str(mode or "none").lower().strip()
    if angvel.ndim != 2 or angvel.size == 0:
        return angvel
    if mode in ("", "none"):
        return angvel
    if mode == "zero":
        return np.zeros_like(angvel)
    return angvel


def _load_bone_names(source_json: Path) -> Optional[List[str]]:
    try:
        d = _load_json(source_json)
        sk = d.get("meta", {}).get("skeleton", {})
        names = sk.get("bone_names", []) if isinstance(sk, dict) else []
        if not isinstance(names, list) or not names:
            return None
        return [str(x) for x in names]
    except Exception:
        return None


def _topk(items: List[Dict[str, object]], *, key: str, k: int) -> List[Dict[str, object]]:
    k = int(max(0, k))
    if k <= 0:
        return []
    return sorted(items, key=lambda d: float(abs(float(d.get(key, 0.0) or 0.0))), reverse=True)[:k]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bone×block input attribution for contact_meas_head at GT edge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pred-json", type=str, required=True, help="*_teacher_pred.json produced by run_teacher_rollout.py")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: pred-json parent / _attrib).")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json", help="Normalization bundle (MuY/StdY).")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json", help="Pretrain template (pose_hist tanh/zscore stats).")
    ap.add_argument("--event", type=str, default="falling", choices=("rising", "falling"), help="Which GT edge to use.")
    ap.add_argument("--channel", type=str, default="R", choices=("L", "R"), help="Which contact channel to attribute.")
    ap.add_argument("--on-th", type=float, default=0.8, help="GT Schmitt ON threshold.")
    ap.add_argument("--off-th", type=float, default=0.1, help="GT Schmitt OFF threshold.")
    ap.add_argument("--edge-which", type=int, default=0, help="Which edge instance to use if multiple exist (0=first).")
    ap.add_argument("--t", type=int, default=-1, help="Override time index (>=0). If <0, uses the selected GT edge.")
    ap.add_argument(
        "--pose-hist-source",
        type=str,
        default="auto",
        choices=("auto", "buffer", "seq"),
        help="Pose history source for attribution (auto uses pred-json ablation.pose_hist_source).",
    )
    ap.add_argument(
        "--angvel-source",
        type=str,
        default="auto",
        choices=("auto", "state", "seq"),
        help="Angvel source for attribution (auto uses pred-json ablation.angvel_source).",
    )
    ap.add_argument("--topk", type=int, default=8, help="Top-K bones to print per block (by |contrib|).")
    args = ap.parse_args()

    pred_path = Path(args.pred_json).expanduser().resolve()
    d = _load_json(pred_path)
    out_dir = Path(args.out).expanduser().resolve() if args.out else (pred_path.parent / "_attrib").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clip = str(d.get("clip", pred_path.stem.replace("_teacher_pred", "")))
    source_json = d.get("source_json")
    source_json_path = Path(source_json).expanduser().resolve() if isinstance(source_json, str) else None
    bone_names = _load_bone_names(source_json_path) if source_json_path else None

    gt = np.asarray(d.get("aux_inputs", {}).get("contacts", []), dtype=np.float32)
    if gt.ndim != 2 or gt.shape[1] < 2:
        raise SystemExit("[FATAL] aux_inputs.contacts missing or malformed.")
    T = int(gt.shape[0])
    ch = 0 if args.channel == "L" else 1

    # Pick time index.
    t0 = int(args.t)
    if t0 < 0:
        t0 = _pick_event_index(gt, ch=ch, on_th=float(args.on_th), off_th=float(args.off_th), event=str(args.event), which=int(args.edge_which))  # type: ignore[assignment]
        if t0 is None:
            raise SystemExit("[FATAL] No GT edges found for the requested event/channel.")
        t0 = int(t0)
    if not (0 <= t0 < T):
        raise SystemExit(f"[FATAL] invalid t={t0} for T={T}.")

    ab = d.get("ablation", {}) if isinstance(d.get("ablation"), dict) else {}
    pose_hist_source = str(args.pose_hist_source).lower()
    if pose_hist_source == "auto":
        pose_hist_source = str(ab.get("pose_hist_source", "seq") or "seq").lower().strip()
    angvel_source = str(args.angvel_source).lower()
    if angvel_source == "auto":
        angvel_source = str(ab.get("angvel_source", "seq") or "seq").lower().strip()

    # Inputs from pred-json.
    teacher = d.get("teacher", {}) if isinstance(d.get("teacher"), dict) else {}
    state_norm = np.asarray(teacher.get("state_norm", []), dtype=np.float32)
    cond = np.asarray(teacher.get("cond", []), dtype=np.float32)
    gt_target_norm = np.asarray(teacher.get("target_norm", []), dtype=np.float32)
    if state_norm.ndim != 2 or cond.ndim != 2:
        raise SystemExit("[FATAL] teacher.state_norm / teacher.cond missing or malformed.")
    if gt_target_norm.ndim != 2:
        raise SystemExit("[FATAL] teacher.target_norm missing or malformed.")

    layouts = d.get("layouts", {}) if isinstance(d.get("layouts"), dict) else {}
    st_layout = layouts.get("state", {}) if isinstance(layouts.get("state"), dict) else {}
    out_layout = layouts.get("output", {}) if isinstance(layouts.get("output"), dict) else {}
    ang_x_slice = _span_to_slice(st_layout.get("BoneAngularVelocities"))
    rot_y_slice = _span_to_slice(out_layout.get("BoneRotations6D"))
    if rot_y_slice is None:
        raise SystemExit("[FATAL] output layout missing BoneRotations6D slice.")

    aux = d.get("aux_inputs", {}) if isinstance(d.get("aux_inputs"), dict) else {}
    pose_hist_seq = np.asarray(aux.get("pose_hist_norm", []), dtype=np.float32)
    angvel_seq = np.asarray(aux.get("angvel_norm", []), dtype=np.float32)

    pose_hist_dim = int(pose_hist_seq.shape[1]) if pose_hist_seq.ndim == 2 else int(d.get("dims", {}).get("pose_hist", 0) or 0)
    angvel_dim = int(angvel_seq.shape[1]) if angvel_seq.ndim == 2 else int(d.get("dims", {}).get("angvel", 0) or 0)
    if pose_hist_dim <= 0 or angvel_dim <= 0:
        raise SystemExit(f"[FATAL] invalid pose_hist_dim={pose_hist_dim}, angvel_dim={angvel_dim}.")

    # Angvel input.
    if angvel_source == "state":
        if ang_x_slice is None:
            raise SystemExit("[FATAL] state layout missing BoneAngularVelocities slice for angvel_source=state.")
        angvel = state_norm[:, ang_x_slice]
    else:
        angvel = angvel_seq
    if angvel.ndim != 2 or angvel.shape[0] != T or angvel.shape[1] != angvel_dim:
        raise SystemExit(f"[FATAL] angvel shape mismatch: got {angvel.shape}, expected (T,{angvel_dim})")

    # Pose history input.
    if pose_hist_source == "seq":
        pose_hist = pose_hist_seq
    else:
        pretrain_tpl = _load_json(Path(args.pretrain_template).expanduser().resolve())
        bundle = _load_json(Path(args.bundle).expanduser().resolve())
        pose_hist_len = int(pretrain_tpl.get("pose_hist_len", 0) or 0)
        if pose_hist_len <= 0:
            raise SystemExit("[FATAL] pretrain_template missing pose_hist_len for pose_hist_source=buffer.")
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
        raise SystemExit(f"[FATAL] pose_hist shape mismatch: got {pose_hist.shape}, expected (T,{pose_hist_dim})")

    # Apply forward-time ablations (match Trainer._rollout_sequence).
    pose_hist_len_eff = None
    try:
        pose_hist_len_eff = int(_load_json(Path(args.pretrain_template).expanduser().resolve()).get("pose_hist_len", 0) or 0)
    except Exception:
        pose_hist_len_eff = None
    if not pose_hist_len_eff:
        pose_dim = int(rot_y_slice.stop - rot_y_slice.start)
        pose_hist_len_eff = int(pose_hist_dim // max(1, pose_dim)) if pose_dim > 0 else 0

    pose_hist = _apply_pose_hist_ablation(
        pose_hist,
        pose_hist_len=int(pose_hist_len_eff),
        mode=str(ab.get("pose_hist_ablation", "none")),
        keep_last=int(ab.get("pose_hist_keep_last", 1) or 1),
    )
    angvel = _apply_angvel_ablation(angvel, mode=str(ab.get("angvel_ablation", "none")))

    # Load head + compute grads.
    ckpt_path = Path(str(d.get("model", ""))).expanduser().resolve()
    head = _load_contact_meas_head(ckpt_path)

    x_full = np.concatenate([pose_hist[t0], angvel[t0]], axis=-1).astype(np.float32, copy=False)
    if x_full.shape[0] <= 0:
        raise SystemExit("[FATAL] empty input feature vector.")

    def _infer(x_vec: np.ndarray) -> Tuple[float, float, np.ndarray]:
        x_t = torch.from_numpy(np.asarray(x_vec, dtype=np.float32)).view(1, -1)
        x_t.requires_grad_(True)
        logits = head(x_t)
        if logits.shape[-1] < 2:
            raise RuntimeError(f"logits shape {tuple(logits.shape)} invalid")
        logit = logits[0, ch]
        prob = torch.sigmoid(logit)
        logit.backward()
        grad = x_t.grad.detach().cpu().numpy().reshape(-1)
        return float(logit.detach().cpu().item()), float(prob.detach().cpu().item()), grad

    logit_full, prob_full, grad_full = _infer(x_full)
    pose_part = x_full[:pose_hist_dim]
    ang_part = x_full[pose_hist_dim:]

    # Counterfactuals (sanity; matches earlier manual checks).
    x_pose0 = x_full.copy()
    x_pose0[:pose_hist_dim] = 0.0
    _, prob_pose0, _ = _infer(x_pose0)
    x_ang0 = x_full.copy()
    x_ang0[pose_hist_dim:] = 0.0
    _, prob_ang0, _ = _infer(x_ang0)

    # Attribution: reshape into (L,J,6) and (J,3).
    pose_hist_len = 0
    # Prefer pretrain_template pose_hist_len when available (buffer), else infer from dims in pred-json.
    if pose_hist_source == "buffer":
        try:
            pose_hist_len = int(_load_json(Path(args.pretrain_template).expanduser().resolve()).get("pose_hist_len", 0) or 0)
        except Exception:
            pose_hist_len = 0
    if pose_hist_len <= 0:
        # Infer from pose_hist_dim and Dy rot slice (J*6).
        pose_dim = int(rot_y_slice.stop - rot_y_slice.start)
        pose_hist_len = int(pose_hist_dim // max(1, pose_dim)) if pose_dim > 0 else 0
    if pose_hist_len <= 0 or pose_hist_dim % pose_hist_len != 0:
        raise SystemExit(f"[FATAL] cannot infer pose_hist_len from pose_hist_dim={pose_hist_dim} and rot_y_slice={rot_y_slice}.")
    pose_dim = pose_hist_dim // pose_hist_len
    if pose_dim % 6 != 0:
        raise SystemExit(f"[FATAL] pose_dim={pose_dim} not divisible by 6.")
    J = pose_dim // 6
    if angvel_dim != J * 3:
        raise SystemExit(f"[FATAL] angvel_dim={angvel_dim} != J*3={J*3} (J={J}).")

    pose_x = pose_part.reshape(pose_hist_len, J, 6)
    pose_g = grad_full[:pose_hist_dim].reshape(pose_hist_len, J, 6)
    ang_x = ang_part.reshape(J, 3)
    ang_g = grad_full[pose_hist_dim:].reshape(J, 3)

    pose_grad_norm = np.linalg.norm(pose_g, axis=-1)  # (L,J)
    pose_input_norm = np.linalg.norm(pose_x, axis=-1)
    pose_contrib = np.sum(pose_g * pose_x, axis=-1)  # (L,J)

    ang_grad_norm = np.linalg.norm(ang_g, axis=-1)  # (J,)
    ang_input_norm = np.linalg.norm(ang_x, axis=-1)
    ang_contrib = np.sum(ang_g * ang_x, axis=-1)  # (J,)

    # Bone naming.
    if bone_names is None or len(bone_names) != J:
        bone_names = [f"bone_{i}" for i in range(J)]

    # Build per-block top-k.
    blocks: List[Dict[str, object]] = []
    for bi in range(pose_hist_len):
        items: List[Dict[str, object]] = []
        for j in range(J):
            items.append(
                {
                    "bone_index": int(j),
                    "bone": bone_names[j],
                    "grad_norm": float(pose_grad_norm[bi, j]),
                    "input_norm": float(pose_input_norm[bi, j]),
                    "contrib": float(pose_contrib[bi, j]),
                }
            )
        blocks.append(
            {
                "block": int(bi),
                "dt": int(bi - (pose_hist_len - 1)),  # newest block => dt=0
                "top": _topk(items, key="contrib", k=int(args.topk)),
                "sum_abs_contrib": float(np.sum(np.abs(pose_contrib[bi]))),
            }
        )

    ang_items: List[Dict[str, object]] = []
    for j in range(J):
        ang_items.append(
            {
                "bone_index": int(j),
                "bone": bone_names[j],
                "grad_norm": float(ang_grad_norm[j]),
                "input_norm": float(ang_input_norm[j]),
                "contrib": float(ang_contrib[j]),
            }
        )

    payload = {
        "clip": clip,
        "pred_json": str(pred_path),
        "ckpt": str(ckpt_path),
        "source_json": str(source_json_path) if source_json_path else None,
        "event": str(args.event),
        "channel": str(args.channel),
        "t": int(t0),
        "gt_at_t": float(gt[t0, ch]),
        "ablation": dict(ab),
        "sources_used": {
            "pose_hist_source": pose_hist_source,
            "angvel_source": angvel_source,
        },
        "logit_full": float(logit_full),
        "prob_full": float(prob_full),
        "prob_pose_zero": float(prob_pose0),
        "prob_angvel_zero": float(prob_ang0),
        "dims": {
            "pose_hist_len": int(pose_hist_len),
            "joints": int(J),
            "pose_hist_dim": int(pose_hist_dim),
            "angvel_dim": int(angvel_dim),
        },
        "pose": {
            "grad_norm": pose_grad_norm.tolist(),
            "contrib": pose_contrib.tolist(),
            "per_block": blocks,
        },
        "angvel": {
            "grad_norm": ang_grad_norm.tolist(),
            "contrib": ang_contrib.tolist(),
            "top": _topk(ang_items, key="contrib", k=int(args.topk)),
            "sum_abs_contrib": float(np.sum(np.abs(ang_contrib))),
        },
        "bone_names": bone_names,
    }

    out_json = out_dir / f"{clip}_{args.channel}_{args.event}_t{t0}_attrib.json"
    out_csv = out_dir / f"{clip}_{args.channel}_{args.event}_t{t0}_attrib_pose.csv"
    out_csv_ang = out_dir / f"{clip}_{args.channel}_{args.event}_t{t0}_attrib_angvel.csv"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV: pose block × bone
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("block,dt,bone_index,bone,grad_norm,input_norm,contrib\n")
        for bi in range(pose_hist_len):
            dt = bi - (pose_hist_len - 1)
            for j in range(J):
                f.write(
                    f"{bi},{dt},{j},{bone_names[j]},"
                    f"{pose_grad_norm[bi,j]:.8g},{pose_input_norm[bi,j]:.8g},{pose_contrib[bi,j]:.8g}\n"
                )

    # CSV: angvel bone
    with out_csv_ang.open("w", encoding="utf-8") as f:
        f.write("bone_index,bone,grad_norm,input_norm,contrib\n")
        for j in range(J):
            f.write(f"{j},{bone_names[j]},{ang_grad_norm[j]:.8g},{ang_input_norm[j]:.8g},{ang_contrib[j]:.8g}\n")

    # Console summary (compact).
    print(f"[OK] {clip} {args.channel} {args.event} @ t={t0}")
    print(f"  GT={gt[t0, ch]:.4f} logit={logit_full:.4f} prob={prob_full:.4f}")
    print(f"  prob(pose=0)={prob_pose0:.4f} prob(angvel=0)={prob_ang0:.4f}")
    for b in blocks:
        top = b.get("top", [])
        dt = b.get("dt")
        if top:
            best = top[0]
            print(f"  pose_hist block dt={dt}: top {best.get('bone')} contrib={best.get('contrib'):.4g} grad_norm={best.get('grad_norm'):.4g}")
    if payload["angvel"]["top"]:
        best = payload["angvel"]["top"][0]
        print(f"  angvel: top {best.get('bone')} contrib={best.get('contrib'):.4g} grad_norm={best.get('grad_norm'):.4g}")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_csv_ang}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
