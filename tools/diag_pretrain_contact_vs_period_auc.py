#!/usr/bin/env python3
"""
Compare pretrain frozen contact_head vs soft_period[:2] as contact predictors.

This diagnostic uses the same pretrain bundle and input preprocessing stack:
  contact_seq + normalized angvel + normalized pose_history

Outputs:
  - summary.json
  - summary.md
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from train.geometry import angvel_vec_from_R_seq, reproject_rot6d, rot6d_to_matrix
from train.io import load_soft_contacts_from_json, npz_scalar_to_str
from train.models import MotionEncoder, PeriodHead
from train.normalizers import VectorTanhNormalizer, _make_angnorm_from_spec
from train.pretrain_mpl_min import (
    StepHead,
    _get_fps_from_npz_or_json,
    _get_rot_slice_from_layout,
    _parse_output_layout_from_npz,
)


@dataclass
class ClipData:
    name: str
    source_json: str
    inputs: np.ndarray  # [T-1, D]
    labels: np.ndarray  # [T-1, 2], soft labels in [0,1]


def _weighted_auc(scores: np.ndarray, pos_w: np.ndarray, eps: float = 1e-9) -> float:
    """
    Weighted ROC-AUC with tie handling.
    pos_w can be soft labels in [0,1], neg_w=1-pos_w.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    w1 = np.asarray(pos_w, dtype=np.float64).reshape(-1)
    w0 = 1.0 - w1
    m = np.isfinite(s) & np.isfinite(w1) & np.isfinite(w0)
    s, w1, w0 = s[m], w1[m], w0[m]
    if s.size == 0:
        return float("nan")

    w1 = np.clip(w1, 0.0, 1.0)
    w0 = np.clip(w0, 0.0, 1.0)
    W1 = float(w1.sum())
    W0 = float(w0.sum())
    if W1 < eps or W0 < eps:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    s = s[order]
    w1 = w1[order]
    w0 = w0[order]

    auc_num = 0.0
    cum_neg = 0.0
    n = s.shape[0]
    i = 0
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        g_pos = float(w1[i:j].sum())
        g_neg = float(w0[i:j].sum())
        auc_num += g_pos * (cum_neg + 0.5 * g_neg)
        cum_neg += g_neg
        i = j

    return float(auc_num / (W1 * W0 + eps))


def _build_pose_hist_norm(norm_spec: Dict, pose_hist_len: int, j: int) -> VectorTanhNormalizer | None:
    pose_hist_dim = int(pose_hist_len * j * 6)
    scales = norm_spec.get("tanh_scales_pose_hist", [])
    if pose_hist_len <= 0 or len(scales) != pose_hist_dim:
        return None
    mu = norm_spec.get("MuPoseHist", [])
    std = norm_spec.get("StdPoseHist", [])
    return VectorTanhNormalizer(
        np.asarray(scales, dtype=np.float32),
        np.asarray(mu, dtype=np.float32) if mu else None,
        np.asarray(std, dtype=np.float32) if std else None,
    )


def _build_clip_data(
    npz_path: Path,
    *,
    ang_norm,
    pose_hist_norm: VectorTanhNormalizer | None,
    pose_hist_len: int,
) -> ClipData:
    with np.load(npz_path, allow_pickle=True) as z:
        if "y_out_features" not in z:
            raise RuntimeError(f"{npz_path.name} missing y_out_features")
        y_full = np.asarray(z["y_out_features"], dtype=np.float32)
        layout = _parse_output_layout_from_npz(z)
        dy = int(y_full.shape[1])
        rot_st, rot_sz = _get_rot_slice_from_layout(layout, dy)
        if rot_sz <= 0 or rot_sz % 6 != 0:
            raise RuntimeError(f"{npz_path.name}: invalid rot6d slice (Dy={dy})")
        y_rot = y_full[:, rot_st : rot_st + rot_sz]

        if "source_json" not in z:
            raise RuntimeError(f"{npz_path.name} missing source_json")
        source_json = npz_scalar_to_str(z["source_json"])
        fps = float(_get_fps_from_npz_or_json(z, source_json))

    soft_contacts = load_soft_contacts_from_json(source_json).astype(np.float32, copy=False)
    t_eff = int(min(y_rot.shape[0], soft_contacts.shape[0]))
    if t_eff <= 1:
        raise RuntimeError(f"{npz_path.name}: too short after alignment (T={t_eff})")

    y_rot = y_rot[:t_eff]
    soft_contacts = soft_contacts[:t_eff]

    y_t = torch.from_numpy(y_rot).to(torch.float32)
    y_t = reproject_rot6d(y_t.unsqueeze(0))[0]
    j = int(y_t.shape[1] // 6)
    r = rot6d_to_matrix(y_t.view(1, t_eff, j, 6))[0]
    w = angvel_vec_from_R_seq(r.unsqueeze(0), fps)[0]
    ang = w.reshape(t_eff - 1, j * 3).cpu().numpy().astype(np.float32, copy=False)

    labels = soft_contacts[1:].astype(np.float32, copy=False)
    if labels.shape[1] < 2:
        raise RuntimeError(f"{npz_path.name}: expected 2 contact channels, got {labels.shape[1]}")
    labels = labels[:, :2]

    pose_seq = y_t.cpu().numpy().astype(np.float32, copy=False)
    if pose_hist_len > 0:
        hist = []
        for t_idx in range(t_eff - 1):
            frames = []
            for h in range(pose_hist_len, 0, -1):
                src_idx = t_idx + 1 - h
                if src_idx < 0:
                    src_idx = 0
                frames.append(pose_seq[src_idx])
            hist.append(np.concatenate(frames, axis=0))
        pose_hist = np.stack(hist, axis=0).astype(np.float32, copy=False)
    else:
        pose_hist = np.zeros((t_eff - 1, 0), dtype=np.float32)

    ang_normed = ang_norm.transform(ang).astype(np.float32, copy=False)
    pose_hist_normed = (
        pose_hist_norm.transform(pose_hist).astype(np.float32, copy=False)
        if pose_hist_norm is not None
        else pose_hist
    )

    inputs = np.concatenate([labels, ang_normed, pose_hist_normed], axis=1).astype(np.float32, copy=False)
    return ClipData(name=npz_path.name, source_json=source_json, inputs=inputs, labels=labels)


def _device_from_arg(device_arg: str) -> torch.device:
    d = str(device_arg).strip().lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=str, default="models/motion_encoder_equiv.pt.best.pt")
    ap.add_argument("--norm_spec", type=str, default="models/pretrain_template.json")
    ap.add_argument("--in_glob", type=str, default="raw_data/processed_data/*.npz")
    ap.add_argument("--contact_threshold", type=float, default=0.5)
    ap.add_argument(
        "--encoder_input_mode",
        type=str,
        default="full",
        choices=["full", "period_projected", "no_contact", "no_contact_period_projected"],
        help=(
            "full: use full encoder input; "
            "period_projected: zero-out angvel channels (pretrain period-branch style); "
            "no_contact: zero-out input contact channels; "
            "no_contact_period_projected: zero-out both contact and angvel channels."
        ),
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max_clips", type=int, default=0, help="0 means all clips")
    ap.add_argument("--out_dir", type=str, default="debug_output/_tmp_pretrain_contact_auc_20260304")
    args = ap.parse_args()

    device = _device_from_arg(args.device)

    bundle_path = Path(args.bundle).expanduser().resolve()
    norm_path = Path(args.norm_spec).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(str(bundle_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"bundle must be dict, got {type(payload).__name__}")
    for key in ("encoder", "period_head", "contact_head"):
        if key not in payload:
            raise KeyError(f"bundle missing key: {key}")

    with norm_path.open("r", encoding="utf-8") as f:
        norm_spec = json.load(f)
    if not isinstance(norm_spec, dict):
        raise RuntimeError("norm_spec must be a dict")

    meta = dict(payload.get("meta", {}))
    enc_state = payload["encoder"]
    period_state = payload["period_head"]
    contact_state = payload["contact_head"]

    w0 = enc_state.get("mlp.0.weight")
    if w0 is None:
        raise RuntimeError("bundle encoder missing mlp.0.weight; cannot infer dimensions")
    input_dim = int(meta.get("input_dim", w0.shape[1]))
    hidden_dim = int(meta.get("hidden_dim", w0.shape[0]))
    z_dim = int(meta.get("z_dim", 0))
    mlp_layers = int(meta.get("mlp_layers", 3))
    mlp_dropout = float(meta.get("mlp_dropout", 0.0))

    period_dim = int(period_state["fc.weight"].shape[0])
    contact_dim = int(contact_state["fc.weight"].shape[0])
    if contact_dim < 2:
        raise RuntimeError(f"contact_head output dim={contact_dim}, expected >=2")

    encoder = MotionEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        z_dim=z_dim,
        num_layers=mlp_layers,
        dropout=mlp_dropout,
    )
    encoder.load_state_dict(enc_state)
    encoder.eval().to(device)

    period_head = PeriodHead(hidden_dim=hidden_dim, out_dim=period_dim)
    period_head.load_state_dict(period_state)
    period_head.eval().to(device)

    contact_head = StepHead(hidden_dim=hidden_dim, K=contact_dim)
    contact_head.load_state_dict(contact_state)
    contact_head.eval().to(device)

    j = int(norm_spec.get("J", 0) or 0)
    if j <= 0:
        raise RuntimeError("norm_spec missing valid J")
    ang_norm = _make_angnorm_from_spec(norm_spec, J_times_3=j * 3, require_zscore=False)
    pose_hist_len = int(norm_spec.get("pose_hist_len", 0) or 0)
    pose_hist_norm = _build_pose_hist_norm(norm_spec, pose_hist_len, j)
    ang_dim = int(j * 3)

    npz_files = sorted(Path(p).resolve() for p in glob.glob(args.in_glob))
    if not npz_files:
        raise RuntimeError(f"no npz matched: {args.in_glob}")
    if int(args.max_clips) > 0:
        npz_files = npz_files[: int(args.max_clips)]

    clip_rows: List[Dict] = []
    all_labels: List[np.ndarray] = []
    all_contact_prob: List[np.ndarray] = []
    all_period_prob: List[np.ndarray] = []

    for npz_path in npz_files:
        clip = _build_clip_data(
            npz_path,
            ang_norm=ang_norm,
            pose_hist_norm=pose_hist_norm,
            pose_hist_len=pose_hist_len,
        )
        if clip.inputs.shape[-1] != input_dim:
            raise RuntimeError(
                f"{npz_path.name}: input dim mismatch {clip.inputs.shape[-1]} vs bundle {input_dim}"
            )

        x_np = clip.inputs
        mode = str(args.encoder_input_mode).strip().lower()
        if mode in ("no_contact", "no_contact_period_projected"):
            x_np = x_np.copy()
            x_np[:, :2] = 0.0
        if mode in ("period_projected", "no_contact_period_projected"):
            x_proj = np.zeros_like(x_np, dtype=np.float32)
            x_proj[:, :2] = x_np[:, :2]
            pose_start = 2 + ang_dim
            if pose_start < x_np.shape[1]:
                x_proj[:, pose_start:] = x_np[:, pose_start:]
            x_np = x_proj
        x = torch.from_numpy(x_np).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            h = encoder(x, return_summary=False)  # [1,T,H]
            logits = contact_head(h)
            contact_prob = torch.sigmoid(logits)[0, :, :2].detach().cpu().numpy().astype(np.float32, copy=False)
            soft_period = torch.tanh(period_head(h))[0, :, :2].detach().cpu().numpy().astype(np.float32, copy=False)
            period_prob = np.clip((soft_period + 1.0) * 0.5, 0.0, 1.0).astype(np.float32, copy=False)

        y = clip.labels.astype(np.float32, copy=False)
        y_hard = (y >= float(args.contact_threshold)).astype(np.float32, copy=False)

        row = {
            "clip": clip.name,
            "frames": int(y.shape[0]),
            "contact_auc_soft_l": _weighted_auc(contact_prob[:, 0], y[:, 0]),
            "contact_auc_soft_r": _weighted_auc(contact_prob[:, 1], y[:, 1]),
            "period_auc_soft_l": _weighted_auc(period_prob[:, 0], y[:, 0]),
            "period_auc_soft_r": _weighted_auc(period_prob[:, 1], y[:, 1]),
            "contact_auc_hard_l": _weighted_auc(contact_prob[:, 0], y_hard[:, 0]),
            "contact_auc_hard_r": _weighted_auc(contact_prob[:, 1], y_hard[:, 1]),
            "period_auc_hard_l": _weighted_auc(period_prob[:, 0], y_hard[:, 0]),
            "period_auc_hard_r": _weighted_auc(period_prob[:, 1], y_hard[:, 1]),
        }
        row["delta_auc_soft_l"] = row["contact_auc_soft_l"] - row["period_auc_soft_l"]
        row["delta_auc_soft_r"] = row["contact_auc_soft_r"] - row["period_auc_soft_r"]
        row["delta_auc_hard_l"] = row["contact_auc_hard_l"] - row["period_auc_hard_l"]
        row["delta_auc_hard_r"] = row["contact_auc_hard_r"] - row["period_auc_hard_r"]
        clip_rows.append(row)

        all_labels.append(y)
        all_contact_prob.append(contact_prob)
        all_period_prob.append(period_prob)

    y_all = np.concatenate(all_labels, axis=0)
    y_all_hard = (y_all >= float(args.contact_threshold)).astype(np.float32, copy=False)
    cp_all = np.concatenate(all_contact_prob, axis=0)
    pp_all = np.concatenate(all_period_prob, axis=0)

    overall = {
        "frames_total": int(y_all.shape[0]),
        "clips_total": int(len(clip_rows)),
        "contact_auc_soft_l": _weighted_auc(cp_all[:, 0], y_all[:, 0]),
        "contact_auc_soft_r": _weighted_auc(cp_all[:, 1], y_all[:, 1]),
        "period_auc_soft_l": _weighted_auc(pp_all[:, 0], y_all[:, 0]),
        "period_auc_soft_r": _weighted_auc(pp_all[:, 1], y_all[:, 1]),
        "contact_auc_hard_l": _weighted_auc(cp_all[:, 0], y_all_hard[:, 0]),
        "contact_auc_hard_r": _weighted_auc(cp_all[:, 1], y_all_hard[:, 1]),
        "period_auc_hard_l": _weighted_auc(pp_all[:, 0], y_all_hard[:, 0]),
        "period_auc_hard_r": _weighted_auc(pp_all[:, 1], y_all_hard[:, 1]),
    }
    overall["delta_auc_soft_l"] = overall["contact_auc_soft_l"] - overall["period_auc_soft_l"]
    overall["delta_auc_soft_r"] = overall["contact_auc_soft_r"] - overall["period_auc_soft_r"]
    overall["delta_auc_hard_l"] = overall["contact_auc_hard_l"] - overall["period_auc_hard_l"]
    overall["delta_auc_hard_r"] = overall["contact_auc_hard_r"] - overall["period_auc_hard_r"]
    overall["contact_auc_soft_mean"] = 0.5 * (
        overall["contact_auc_soft_l"] + overall["contact_auc_soft_r"]
    )
    overall["period_auc_soft_mean"] = 0.5 * (
        overall["period_auc_soft_l"] + overall["period_auc_soft_r"]
    )
    overall["delta_auc_soft_mean"] = overall["contact_auc_soft_mean"] - overall["period_auc_soft_mean"]
    overall["contact_auc_hard_mean"] = 0.5 * (
        overall["contact_auc_hard_l"] + overall["contact_auc_hard_r"]
    )
    overall["period_auc_hard_mean"] = 0.5 * (
        overall["period_auc_hard_l"] + overall["period_auc_hard_r"]
    )
    overall["delta_auc_hard_mean"] = overall["contact_auc_hard_mean"] - overall["period_auc_hard_mean"]

    result = {
        "bundle": str(bundle_path),
        "norm_spec": str(norm_path),
        "in_glob": str(args.in_glob),
        "contact_threshold": float(args.contact_threshold),
        "encoder_input_mode": str(args.encoder_input_mode),
        "device": str(device),
        "bundle_meta": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "z_dim": z_dim,
            "period_dim": period_dim,
            "contact_dim": contact_dim,
        },
        "overall": overall,
        "per_clip": clip_rows,
    }

    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# pretrain contact_head vs soft_period[:2] AUC",
        "",
        "## Setup",
        f"- bundle: `{bundle_path}`",
        f"- norm_spec: `{norm_path}`",
        f"- in_glob: `{args.in_glob}`",
        f"- clips: {overall['clips_total']} | frames: {overall['frames_total']}",
        f"- threshold(hard): {float(args.contact_threshold):.3f}",
        f"- encoder_input_mode: `{args.encoder_input_mode}`",
        "",
        "## Overall",
        f"- soft AUC mean (L/R avg): contact={overall['contact_auc_soft_mean']:.6f} | period={overall['period_auc_soft_mean']:.6f} | delta={overall['delta_auc_soft_mean']:+.6f}",
        f"- soft AUC L: contact={overall['contact_auc_soft_l']:.6f} | period={overall['period_auc_soft_l']:.6f} | delta={overall['delta_auc_soft_l']:+.6f}",
        f"- soft AUC R: contact={overall['contact_auc_soft_r']:.6f} | period={overall['period_auc_soft_r']:.6f} | delta={overall['delta_auc_soft_r']:+.6f}",
        f"- hard AUC mean (L/R avg): contact={overall['contact_auc_hard_mean']:.6f} | period={overall['period_auc_hard_mean']:.6f} | delta={overall['delta_auc_hard_mean']:+.6f}",
        f"- hard AUC L: contact={overall['contact_auc_hard_l']:.6f} | period={overall['period_auc_hard_l']:.6f} | delta={overall['delta_auc_hard_l']:+.6f}",
        f"- hard AUC R: contact={overall['contact_auc_hard_r']:.6f} | period={overall['period_auc_hard_r']:.6f} | delta={overall['delta_auc_hard_r']:+.6f}",
        "",
        "## Per-clip soft AUC (delta = contact - period)",
        "",
        "| clip | frames | L delta | R delta |",
        "|---|---:|---:|---:|",
    ]
    for row in clip_rows:
        md_lines.append(
            f"| {row['clip']} | {row['frames']} | {row['delta_auc_soft_l']:+.6f} | {row['delta_auc_soft_r']:+.6f} |"
        )
    summary_md = out_dir / "summary.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[done] wrote: {summary_json}")
    print(f"[done] wrote: {summary_md}")
    print(
        "[overall] soft AUC mean contact={:.6f} period={:.6f} delta={:+.6f}".format(
            overall["contact_auc_soft_mean"],
            overall["period_auc_soft_mean"],
            overall["delta_auc_soft_mean"],
        )
    )


if __name__ == "__main__":
    _main()
