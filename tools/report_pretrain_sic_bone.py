#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from train.geometry import geodesic_R, reproject_rot6d, rot6d_to_matrix
from train.data.io import npz_scalar_to_str
from train.models import MotionEncoder, PeriodHead
from train.data.normalizers import VectorTanhNormalizer, _make_angnorm_from_spec
from train.pretrain_mpl_min import (
    InputProjectors,
    InputSlices,
    StepHead,
    _get_fps_from_npz_or_json,
    _get_rot_slice_from_layout,
    _parse_output_layout_from_npz,
)
from train.utils import build_mlp


@dataclass
class ClipEval:
    clip: str
    npz_path: Path
    source_json: str
    columns: Tuple[str, str]
    bone_names: List[str]
    root_idx: int
    sic: np.ndarray
    geo_deg: np.ndarray


def _device_from_arg(device_arg: str) -> torch.device:
    d = str(device_arg).strip().lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _expand_npz_specs(specs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for spec in specs:
        s = str(spec or "").strip()
        if not s:
            continue
        matches: Iterable[str]
        if any(ch in s for ch in "*?[]"):
            matches = sorted(glob.glob(s))
        else:
            matches = [s]
        for item in matches:
            p = Path(item).expanduser().resolve()
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out)


def _build_pose_hist_norm(norm_spec: Dict[str, Any], pose_hist_len: int, joint_count: int) -> Optional[VectorTanhNormalizer]:
    pose_hist_dim = int(pose_hist_len * joint_count * 6)
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


def _build_pose_target_norm(norm_spec: Dict[str, Any], pose_dim: int) -> Optional[VectorTanhNormalizer]:
    scales = norm_spec.get("tanh_scales_pose_target", [])
    if len(scales) != int(pose_dim):
        return None
    mu = norm_spec.get("MuPoseTarget", [])
    std = norm_spec.get("StdPoseTarget", [])
    return VectorTanhNormalizer(
        np.asarray(scales, dtype=np.float32),
        np.asarray(mu, dtype=np.float32) if mu else None,
        np.asarray(std, dtype=np.float32) if std else None,
    )


def _read_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"expected dict JSON root: {path}")
    return obj


def _resolve_columns(source_json: Dict[str, Any]) -> Tuple[str, str]:
    candidates: List[Any] = []
    meta = source_json.get("meta")
    if isinstance(meta, dict):
        rot6d_spec = meta.get("rot6d_spec")
        if isinstance(rot6d_spec, dict):
            candidates.append(rot6d_spec.get("columns"))
    rot6d_spec_root = source_json.get("rot6d_spec")
    if isinstance(rot6d_spec_root, dict):
        candidates.append(rot6d_spec_root.get("columns"))
    for cols in candidates:
        if isinstance(cols, (list, tuple)) and len(cols) >= 2:
            a = str(cols[0]).strip().upper()
            b = str(cols[1]).strip().upper()
            if a in ("X", "Y", "Z") and b in ("X", "Y", "Z") and a != b:
                return a, b
    return ("X", "Z")


def _resolve_bone_names(source_json: Dict[str, Any], joint_count: int) -> Tuple[List[str], int]:
    names: List[str] = []
    root_idx = 0

    meta = source_json.get("meta")
    skel = meta.get("skeleton") if isinstance(meta, dict) else None
    if isinstance(skel, dict):
        raw_names = skel.get("bone_names")
        if isinstance(raw_names, list):
            names = [str(x) for x in raw_names]
        root_name = str(skel.get("root_bone") or "").strip()
        if root_name and root_name in names:
            root_idx = int(names.index(root_name))

    if len(names) != int(joint_count):
        names = [f"bone_{i:02d}" for i in range(int(joint_count))]
        root_idx = 0
    return names, root_idx


def _make_decoder_from_state(state: Dict[str, torch.Tensor], period_dim: int, out_dim: int) -> nn.Sequential:
    w0 = state.get("0.weight")
    if w0 is None:
        raise RuntimeError("decoder state missing 0.weight")
    hidden_dim = int(w0.shape[0])
    dec = build_mlp(
        int(period_dim),
        hidden_dim,
        num_layers=2,
        activation=nn.GELU,
        final_dim=int(out_dim),
    )
    dec.load_state_dict(state)
    return dec


def _build_models(
    payload: Dict[str, Any],
    *,
    device: torch.device,
) -> Tuple[MotionEncoder, PeriodHead, StepHead, nn.Sequential]:
    meta = dict(payload.get("meta", {}))
    enc_state = payload["encoder"]
    period_state = payload["period_head"]
    contact_state = payload["contact_head"]
    pose_state = payload["decoder_pose"]

    w0 = enc_state.get("mlp.0.weight")
    if w0 is None:
        raise RuntimeError("bundle encoder missing mlp.0.weight")

    input_dim = int(meta.get("input_dim", w0.shape[1]))
    hidden_dim = int(meta.get("hidden_dim", w0.shape[0]))
    z_dim = int(meta.get("z_dim", 0))
    mlp_layers = int(meta.get("mlp_layers", 3))
    mlp_dropout = float(meta.get("mlp_dropout", 0.0))
    period_dim = int(period_state["fc.weight"].shape[0])
    contact_dim = int(contact_state["fc.weight"].shape[0])
    pose_dim = int(pose_state["4.weight"].shape[0] if "4.weight" in pose_state else pose_state["2.weight"].shape[0])

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

    decoder_pose = _make_decoder_from_state(pose_state, period_dim=period_dim, out_dim=pose_dim)
    decoder_pose.eval().to(device)
    return encoder, period_head, contact_head, decoder_pose


def _build_clip_inputs(
    npz_path: Path,
    *,
    ang_norm: Any,
    pose_hist_norm: Optional[VectorTanhNormalizer],
    pose_hist_len: int,
) -> Dict[str, Any]:
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

    source_obj = _read_json(Path(source_json).expanduser().resolve())
    frames = source_obj.get("Frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{source_json}: missing Frames")
    soft_contacts = []
    for frame in frames:
        foot = frame.get("FootEvidence", {}) if isinstance(frame, dict) else {}
        l = float((foot.get("L") or {}).get("Contact", 0.0)) if isinstance(foot.get("L"), dict) else 0.0
        r = float((foot.get("R") or {}).get("Contact", 0.0)) if isinstance(foot.get("R"), dict) else 0.0
        soft_contacts.append([l, r])
    soft_contacts_arr = np.asarray(soft_contacts, dtype=np.float32)

    t_eff = int(min(y_rot.shape[0], soft_contacts_arr.shape[0]))
    if t_eff <= 1:
        raise RuntimeError(f"{npz_path.name}: too short after alignment (T={t_eff})")

    y_rot = y_rot[:t_eff]
    soft_contacts_arr = soft_contacts_arr[:t_eff]

    y_t = torch.from_numpy(y_rot).to(torch.float32)
    y_t = reproject_rot6d(y_t.unsqueeze(0))[0]
    joint_count = int(y_t.shape[1] // 6)

    pose_seq = y_t.cpu().numpy().astype(np.float32, copy=False)
    pose_target_raw = pose_seq[1:]

    r = rot6d_to_matrix(y_t.view(1, t_eff, joint_count, 6))[0]
    from train.geometry import angvel_vec_from_R_seq

    w = angvel_vec_from_R_seq(r.unsqueeze(0), fps)[0]
    ang = w.reshape(t_eff - 1, joint_count * 3).cpu().numpy().astype(np.float32, copy=False)

    if pose_hist_len > 0:
        hist = []
        for t_idx in range(t_eff - 1):
            frames_hist = []
            for h in range(pose_hist_len, 0, -1):
                src_idx = t_idx + 1 - h
                if src_idx < 0:
                    src_idx = 0
                frames_hist.append(pose_seq[src_idx])
            hist.append(np.concatenate(frames_hist, axis=0))
        pose_hist = np.stack(hist, axis=0).astype(np.float32, copy=False)
    else:
        pose_hist = np.zeros((t_eff - 1, 0), dtype=np.float32)

    ang_normed = ang_norm.transform(ang).astype(np.float32, copy=False)
    pose_hist_normed = (
        pose_hist_norm.transform(pose_hist).astype(np.float32, copy=False)
        if pose_hist_norm is not None
        else pose_hist
    )
    inputs = np.concatenate([soft_contacts_arr[1:], ang_normed, pose_hist_normed], axis=1).astype(np.float32, copy=False)

    columns = _resolve_columns(source_obj)
    bone_names, root_idx = _resolve_bone_names(source_obj, joint_count)
    return {
        "clip": npz_path.stem,
        "npz_path": npz_path,
        "source_json": source_json,
        "inputs": inputs,
        "pose_target_raw": pose_target_raw,
        "columns": columns,
        "bone_names": bone_names,
        "root_idx": int(root_idx),
    }


def _evaluate_clip(
    clip_data: Dict[str, Any],
    *,
    encoder: MotionEncoder,
    period_head: PeriodHead,
    decoder_pose: nn.Sequential,
    pose_target_norm: Optional[VectorTanhNormalizer],
    projectors: InputProjectors,
    device: torch.device,
) -> ClipEval:
    inputs = clip_data["inputs"]
    pose_target_raw = clip_data["pose_target_raw"]
    bone_names = list(clip_data["bone_names"])
    columns = tuple(clip_data["columns"])
    root_idx = int(clip_data["root_idx"])

    x = torch.from_numpy(inputs).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        period_inputs = projectors.period(x)
        _, h_period = encoder(period_inputs, return_summary=True)
        soft_period = torch.tanh(period_head(h_period))
        pose_pred_norm = decoder_pose(soft_period)[0].detach().cpu().numpy().astype(np.float32, copy=False)

    pose_pred_raw = pose_target_norm.inverse(pose_pred_norm) if pose_target_norm is not None else pose_pred_norm
    pose_pred_t = torch.from_numpy(np.asarray(pose_pred_raw, dtype=np.float32))
    pose_tgt_t = torch.from_numpy(np.asarray(pose_target_raw, dtype=np.float32))

    joint_count = int(len(bone_names))
    pred6 = reproject_rot6d(pose_pred_t.view(-1, joint_count, 6))
    tgt6 = reproject_rot6d(pose_tgt_t.view(-1, joint_count, 6))
    rp = rot6d_to_matrix(pred6, columns=columns)
    rg = rot6d_to_matrix(tgt6, columns=columns)
    geo_deg = (geodesic_R(rp, rg) * (180.0 / math.pi)).cpu().numpy().astype(np.float64, copy=False)
    sic = np.arange(geo_deg.shape[0], dtype=np.int64)

    return ClipEval(
        clip=str(clip_data["clip"]),
        npz_path=Path(clip_data["npz_path"]).resolve(),
        source_json=str(clip_data["source_json"]),
        columns=columns,
        bone_names=bone_names,
        root_idx=root_idx,
        sic=sic,
        geo_deg=geo_deg,
    )


def _aggregate(clips: Sequence[ClipEval]) -> Dict[str, Any]:
    if not clips:
        raise RuntimeError("no clip results")

    bone_names = list(clips[0].bone_names)
    root_idx = int(clips[0].root_idx)
    for clip in clips[1:]:
        if clip.bone_names != bone_names:
            raise RuntimeError(f"bone_names mismatch: {clip.clip}")
        if int(clip.root_idx) != int(root_idx):
            raise RuntimeError(f"root_idx mismatch: {clip.clip}")

    pair_values: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    per_sic_values: Dict[int, List[float]] = defaultdict(list)
    per_bone_values: Dict[str, List[float]] = defaultdict(list)
    all_vals_ex_root: List[float] = []

    max_sic = -1
    for clip in clips:
        for i, sic in enumerate(clip.sic.tolist()):
            max_sic = max(max_sic, int(sic))
            for j, bone in enumerate(bone_names):
                val = float(clip.geo_deg[i, j])
                if not math.isfinite(val):
                    continue
                pair_values[(int(sic), str(bone))].append(val)
                if int(j) != int(root_idx):
                    per_sic_values[int(sic)].append(val)
                    per_bone_values[str(bone)].append(val)
                    all_vals_ex_root.append(val)

    sics = list(range(max_sic + 1))
    bones = bone_names[:]
    mean_mat = np.full((len(sics), len(bones)), np.nan, dtype=np.float64)
    p50_mat = np.full_like(mean_mat, np.nan)
    p90_mat = np.full_like(mean_mat, np.nan)
    cnt_mat = np.zeros((len(sics), len(bones)), dtype=np.int64)
    top_pairs: List[Dict[str, Any]] = []

    for i, sic in enumerate(sics):
        for j, bone in enumerate(bones):
            vals = pair_values.get((int(sic), str(bone)), [])
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            mean_mat[i, j] = float(np.nanmean(arr))
            p50_mat[i, j] = float(np.nanpercentile(arr, 50))
            p90_mat[i, j] = float(np.nanpercentile(arr, 90))
            cnt_mat[i, j] = int(arr.size)
            if int(j) != int(root_idx):
                top_pairs.append(
                    {
                        "sic": int(sic),
                        "bone": str(bone),
                        "mean_deg": float(mean_mat[i, j]),
                        "p50_deg": float(p50_mat[i, j]),
                        "p90_deg": float(p90_mat[i, j]),
                        "count": int(cnt_mat[i, j]),
                    }
                )

    top_pairs.sort(key=lambda r: (-float(r["mean_deg"]), int(r["sic"]), str(r["bone"])))

    per_bone_mean = [
        {
            "bone": str(bone),
            "mean_deg": float(np.nanmean(np.asarray(vals, dtype=np.float64))) if vals else float("nan"),
            "count": int(len(vals)),
        }
        for bone, vals in per_bone_values.items()
    ]
    per_bone_mean.sort(key=lambda r: (-float(r["mean_deg"]), str(r["bone"])))

    per_sic_mean = [
        {
            "sic": int(sic),
            "mean_deg_ex_root": float(np.nanmean(np.asarray(vals, dtype=np.float64))) if vals else float("nan"),
            "count": int(len(vals)),
        }
        for sic, vals in sorted(per_sic_values.items(), key=lambda kv: int(kv[0]))
    ]
    per_sic_mean.sort(key=lambda r: int(r["sic"]))

    return {
        "bone_names": bones,
        "root_idx": int(root_idx),
        "root_bone": str(bones[root_idx]) if 0 <= root_idx < len(bones) else "root",
        "sics": sics,
        "mean_mat": mean_mat,
        "p50_mat": p50_mat,
        "p90_mat": p90_mat,
        "cnt_mat": cnt_mat,
        "top_pairs": top_pairs,
        "per_bone_mean": per_bone_mean,
        "per_sic_mean": per_sic_mean,
        "overall_mean_deg_ex_root": float(np.nanmean(np.asarray(all_vals_ex_root, dtype=np.float64))) if all_vals_ex_root else float("nan"),
    }


def _write_matrix_csv(path: Path, *, sics: Sequence[int], bones: Sequence[str], mat: np.ndarray, digits: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sic", *bones])
        for i, sic in enumerate(sics):
            row = [int(sic)]
            for j in range(len(bones)):
                v = float(mat[i, j])
                row.append("" if not math.isfinite(v) else f"{v:.{digits}f}")
            w.writerow(row)


def _write_count_csv(path: Path, *, sics: Sequence[int], bones: Sequence[str], mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sic", *bones])
        for i, sic in enumerate(sics):
            w.writerow([int(sic), *[int(mat[i, j]) for j in range(len(bones))]])


def _write_rows_csv(path: Path, *, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def main() -> int:
    ap = argparse.ArgumentParser(description="Report pretrain bundle per-(sic,bone) local rotation error.")
    ap.add_argument("--bundle", type=str, required=True)
    ap.add_argument("--norm-spec", type=str, default="models/pretrain_template.json")
    ap.add_argument("--npz", type=str, action="append", default=None, help="NPZ path or glob. Repeatable.")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    bundle_path = Path(args.bundle).expanduser().resolve()
    norm_path = Path(args.norm_spec).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_specs = args.npz or ["raw_data/processed_data/Walk_F.npz"]
    npz_paths = _expand_npz_specs(npz_specs)
    if not npz_paths:
        raise SystemExit(f"[FATAL] no npz matched: {npz_specs}")

    device = _device_from_arg(args.device)
    payload = torch.load(str(bundle_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"[FATAL] bundle must be dict, got {type(payload).__name__}")
    for key in ("encoder", "period_head", "contact_head", "decoder_pose"):
        if key not in payload:
            raise SystemExit(f"[FATAL] bundle missing key: {key}")

    norm_spec = _read_json(norm_path)
    joint_count = int(norm_spec.get("J", 0) or 0)
    if joint_count <= 0:
        raise SystemExit("[FATAL] norm spec missing valid J")

    ang_norm = _make_angnorm_from_spec(norm_spec, J_times_3=joint_count * 3, require_zscore=False)
    pose_hist_len = int(norm_spec.get("pose_hist_len", 0) or 0)
    pose_hist_norm = _build_pose_hist_norm(norm_spec, pose_hist_len, joint_count)

    encoder, period_head, _contact_head, decoder_pose = _build_models(payload, device=device)

    meta = dict(payload.get("meta", {}))
    input_dim = int(meta.get("input_dim", 0) or 0)
    ang_dim = int(meta.get("ang_dim", joint_count * 3))
    pose_dim = int(meta.get("pose_dim", joint_count * 6))
    pose_hist_dim = int(max(0, input_dim - 2 - ang_dim))
    if pose_hist_dim != int(pose_hist_len * joint_count * 6):
        raise SystemExit(
            f"[FATAL] pose_hist_dim mismatch bundle={pose_hist_dim} vs norm_spec={pose_hist_len * joint_count * 6}"
        )

    layout = InputSlices(
        contact=slice(0, 2),
        ang=slice(2, 2 + ang_dim),
        pose_hist=(slice(2 + ang_dim, input_dim) if pose_hist_dim > 0 else None),
    )
    projectors = InputProjectors(
        layout,
        period_include_ang_sign=False,
        period_use_ang_features=True,
        angnorm=ang_norm,
        amp_linear=True,
    )
    pose_target_norm = _build_pose_target_norm(norm_spec, pose_dim)

    clip_results: List[ClipEval] = []
    for npz_path in npz_paths:
        clip_data = _build_clip_inputs(
            npz_path,
            ang_norm=ang_norm,
            pose_hist_norm=pose_hist_norm,
            pose_hist_len=pose_hist_len,
        )
        if int(clip_data["inputs"].shape[1]) != int(input_dim):
            raise SystemExit(
                f"[FATAL] {npz_path.name}: input dim mismatch {clip_data['inputs'].shape[1]} vs bundle {input_dim}"
            )
        clip_results.append(
            _evaluate_clip(
                clip_data,
                encoder=encoder,
                period_head=period_head,
                decoder_pose=decoder_pose,
                pose_target_norm=pose_target_norm,
                projectors=projectors,
                device=device,
            )
        )

    agg = _aggregate(clip_results)
    top_k = max(0, int(args.top_k))
    top_pairs = agg["top_pairs"][:top_k]

    summary = {
        "bundle": str(bundle_path),
        "norm_spec": str(norm_path),
        "device": str(device),
        "clips": [
            {
                "clip": c.clip,
                "npz_path": str(c.npz_path),
                "source_json": str(c.source_json),
                "steps": int(c.geo_deg.shape[0]),
                "joint_count": int(c.geo_deg.shape[1]),
                "columns": list(c.columns),
            }
            for c in clip_results
        ],
        "bundle_meta": meta,
        "root_idx": int(agg["root_idx"]),
        "root_bone": str(agg["root_bone"]),
        "overall_mean_deg_ex_root": float(agg["overall_mean_deg_ex_root"]),
        "top_pairs": top_pairs,
        "per_bone_mean_top20": agg["per_bone_mean"][:20],
        "per_sic_mean_top20": sorted(agg["per_sic_mean"], key=lambda r: (-float(r["mean_deg_ex_root"]), int(r["sic"])))[:20],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_matrix_csv(out_dir / "per_sic_bone_mean_deg.csv", sics=agg["sics"], bones=agg["bone_names"], mat=agg["mean_mat"])
    _write_matrix_csv(out_dir / "per_sic_bone_p50_deg.csv", sics=agg["sics"], bones=agg["bone_names"], mat=agg["p50_mat"])
    _write_matrix_csv(out_dir / "per_sic_bone_p90_deg.csv", sics=agg["sics"], bones=agg["bone_names"], mat=agg["p90_mat"])
    _write_count_csv(out_dir / "per_sic_bone_count.csv", sics=agg["sics"], bones=agg["bone_names"], mat=agg["cnt_mat"])
    _write_rows_csv(
        out_dir / "top_pairs.csv",
        rows=top_pairs,
        fieldnames=["sic", "bone", "mean_deg", "p50_deg", "p90_deg", "count"],
    )
    _write_rows_csv(
        out_dir / "per_bone_mean_deg.csv",
        rows=agg["per_bone_mean"],
        fieldnames=["bone", "mean_deg", "count"],
    )
    _write_rows_csv(
        out_dir / "per_sic_mean_deg.csv",
        rows=agg["per_sic_mean"],
        fieldnames=["sic", "mean_deg_ex_root", "count"],
    )

    md_lines = [
        "# pretrain per-(sic,bone) report",
        "",
        "## Setup",
        f"- bundle: `{bundle_path}`",
        f"- norm_spec: `{norm_path}`",
        f"- clips: {len(clip_results)}",
        f"- overall_mean_deg_ex_root: {float(agg['overall_mean_deg_ex_root']):.6f}",
        f"- root_bone: `{agg['root_bone']}` (idx={int(agg['root_idx'])})",
        "",
        "## Clips",
    ]
    for c in clip_results:
        md_lines.append(
            f"- `{c.clip}`: steps={int(c.geo_deg.shape[0])} joints={int(c.geo_deg.shape[1])} columns={list(c.columns)}"
        )
    md_lines.extend(
        [
            "",
            "## Top Pairs",
            "",
            "| sic | bone | mean_deg | p50_deg | p90_deg | count |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in top_pairs:
        md_lines.append(
            f"| {int(row['sic'])} | {row['bone']} | {float(row['mean_deg']):.6f} | "
            f"{float(row['p50_deg']):.6f} | {float(row['p90_deg']):.6f} | {int(row['count'])} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {out_dir / 'summary.json'}")
    print(f"[OK] wrote: {out_dir / 'per_sic_bone_mean_deg.csv'}")
    print(f"[OK] wrote: {out_dir / 'top_pairs.csv'}")
    print(f"[Diag] overall_mean_deg_ex_root = {float(agg['overall_mean_deg_ex_root']):.6f}")
    if top_pairs:
        head = top_pairs[0]
        print(
            f"[Diag] top_pair = sic {int(head['sic'])} | bone {head['bone']} | mean_deg {float(head['mean_deg']):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
