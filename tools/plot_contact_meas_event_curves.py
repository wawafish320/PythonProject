#!/usr/bin/env python3
"""
Plot event-aligned contact_meas curves (pred vs GT) around GT rising/falling edges.

This is meant to "pin down" the shape of the hysteresis / long-tail issue:
  - pred value at GT_fall
  - how quickly it drops below a mid threshold (default 0.55)

Typical usage (matches the repo's debug_output layout):
  python tools/plot_contact_meas_event_curves.py \
    --root debug_output/_tmp_teacher_debug/_batch_eventlag \
    --conds baseline keep_last1 pose_zero \
    --out debug_output/_tmp_teacher_debug/_batch_eventlag/_event_curves \
    --event falling --channel R --pre 10 --post 30

You can optionally overlay raw FootEvidence scalars (from source_json) and/or angvel magnitude
for a specific bone around the same event:
  python tools/plot_contact_meas_event_curves.py \
    --root debug_output/_tmp_teacher_debug/_batch_eventlag \
    --conds baseline keep_last1 pose_zero \
    --out debug_output/_tmp_teacher_debug/_batch_eventlag/_event_curves_plus \
    --event falling --channel R --pre 10 --post 30 \
    --overlay foot_vz foot_height angvel_mag --bone foot_r
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from tools.analyze_contact_meas_lag import _edge_times, _schmitt_state  # type: ignore
except Exception:  # pragma: no cover
    # When executed as `python tools/xxx.py`, sys.path[0] == "tools/", so import sibling directly.
    from analyze_contact_meas_lag import _edge_times, _schmitt_state  # type: ignore


def _ensure_mpl(out_dir: Path) -> None:
    # Avoid ~/.matplotlib permission issues on some setups.
    cache_dir = out_dir / "_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))

def _load_raw_frames(path: Path) -> Optional[List[dict]]:
    try:
        d = _load_json(path)
    except Exception:
        return None
    frames = d.get("Frames")
    if not isinstance(frames, list):
        return None
    return frames  # type: ignore[return-value]


def _discover_conditions(root: Path, *, conds: Optional[Sequence[str]]) -> List[str]:
    if conds:
        return [str(c) for c in conds]
    out: List[str] = []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            if list(p.glob("*_teacher_pred.json")):
                out.append(p.name)
    return out


def _build_clip_index(root: Path, conds: Sequence[str]) -> Dict[str, Dict[str, Path]]:
    """
    Returns: clip -> {cond: path}
    """
    out: Dict[str, Dict[str, Path]] = {}
    for cond in conds:
        d = root / cond
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*_teacher_pred.json")):
            clip = p.name.replace("_teacher_pred.json", "")
            out.setdefault(clip, {})[cond] = p
    return out


def _pick_event_indices(gt_contacts: np.ndarray, *, ch: int, on_th: float, off_th: float, event: str) -> np.ndarray:
    s = _schmitt_state(gt_contacts[:, ch], on_th=float(on_th), off_th=float(off_th))
    rise, fall = _edge_times(s)
    if event == "rising":
        return rise
    return fall


def _extract_window(x: np.ndarray, t0: int, *, pre: int, post: int) -> Tuple[np.ndarray, np.ndarray]:
    T = int(x.shape[0])
    s = max(0, int(t0) - int(pre))
    e = min(T, int(t0) + int(post) + 1)
    ts = np.arange(s - int(t0), e - int(t0), dtype=np.int64)
    return ts, x[s:e]


def _time_to_le(x: np.ndarray, t0: int, *, thr: float, post: int) -> Optional[int]:
    T = int(x.shape[0])
    t0 = int(t0)
    if t0 < 0 or t0 >= T:
        return None
    end = min(T, t0 + int(post) + 1)
    seg = x[t0:end]
    idx = np.where(seg <= float(thr))[0]
    return int(idx[0]) if idx.size else None


def _bone_index_from_source(source_json: Path, bone: str) -> Optional[int]:
    try:
        d = _load_json(source_json)
        sk = d.get("meta", {}).get("skeleton", {})
        names = sk.get("bone_names", [])
        if not isinstance(names, list) or not names:
            return None
        bone_l = str(bone).lower()
        for i, n in enumerate(names):
            if str(n).lower() == bone_l:
                return int(i)
        for i, n in enumerate(names):
            if bone_l in str(n).lower():
                return int(i)
        return None
    except Exception:
        return None


def _extract_angvel_mag(d: Dict[str, object], *, bone_index: int) -> Optional[np.ndarray]:
    try:
        ab = d.get("ablation", {}) if isinstance(d.get("ablation"), dict) else {}
        src = str(ab.get("angvel_source", "state") or "state").lower().strip()
        ang = None
        if src == "seq":
            ang = np.asarray(d.get("aux_inputs", {}).get("angvel_norm", []), dtype=np.float64)
        else:
            st = np.asarray(d.get("teacher", {}).get("state_norm", []), dtype=np.float64)
            sl = d.get("layouts", {}).get("state", {}).get("BoneAngularVelocities", {})
            if not isinstance(sl, dict):
                return None
            st_i = int(sl.get("start", 0) or 0)
            sz_i = int(sl.get("size", 0) or 0)
            if st.ndim != 2 or sz_i <= 0:
                return None
            ang = st[:, st_i : st_i + sz_i]
        if ang is None or ang.ndim != 2 or ang.shape[1] < (bone_index + 1) * 3:
            return None
        v = ang[:, bone_index * 3 : (bone_index + 1) * 3]
        return np.linalg.norm(v, axis=1)
    except Exception:
        return None


def _extract_pose_fk(
    source_json: Path,
    *,
    bone: str,
    T: int,
    fps: float,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Pose(rot6d) -> FK to derive foot kinematics.

    Note:
      - FootEvidence.{height/vz/vxy} in raw_data tends to match `ball_{L/R}` better than `foot_{L/R}`.
        If you want a tighter overlay with FootEvidence, try `--bone ball_r` (or ball_l).
      - FootEvidence.vz_mps appears to be |vz| (non-negative). We follow that convention.
    """
    try:
        d = _load_json(source_json)
        frames = d.get("Frames")
        if not isinstance(frames, list) or not frames:
            return None
        sk = d.get("meta", {}).get("skeleton", {})
        if not isinstance(sk, dict):
            return None

        names = sk.get("bone_names", [])
        parents = sk.get("parents", [])
        offsets = sk.get("ref_local_offsets_m", [])
        if not isinstance(names, list) or not isinstance(parents, list) or not isinstance(offsets, list):
            return None
        if not names or not parents or not offsets:
            return None

        bone_i = None
        bone_l = str(bone).lower()
        for i, n in enumerate(names):
            if str(n).lower() == bone_l:
                bone_i = int(i)
                break
        if bone_i is None:
            for i, n in enumerate(names):
                if bone_l in str(n).lower():
                    bone_i = int(i)
                    break
        if bone_i is None:
            return None

        # Lazy import: torch isn't needed for the default plotting path.
        import torch

        from train.geometry import fk_positions_from_rot6d

        T = int(min(int(T), len(frames)))
        J = int(len(names))

        rot6d = np.asarray([fr.get("BoneRotations", []) for fr in frames[:T]], dtype=np.float32)
        if rot6d.ndim != 3 or rot6d.shape[1] != J or rot6d.shape[2] != 6:
            return None

        offsets_t = torch.tensor(offsets, dtype=torch.float32)
        rot_t = torch.from_numpy(rot6d)  # (T, J, 6)
        pos_local = fk_positions_from_rot6d(rot_t, parents, offsets_t, root_pos=None).detach().cpu().numpy()  # (T, J, 3)

        # Apply root yaw + translation to get world positions.
        yaw = np.asarray([fr.get("RootYaw", 0.0) for fr in frames[:T]], dtype=np.float32).reshape(T)
        root_pos = np.asarray([fr.get("RootPosition", [0.0, 0.0, 0.0]) for fr in frames[:T]], dtype=np.float32)
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            root_pos = np.zeros((T, 3), dtype=np.float32)

        c = np.cos(yaw)
        s = np.sin(yaw)
        Rz = np.stack(
            [
                np.stack([c, -s, np.zeros_like(c)], axis=-1),
                np.stack([s, c, np.zeros_like(c)], axis=-1),
                np.stack([np.zeros_like(c), np.zeros_like(c), np.ones_like(c)], axis=-1),
            ],
            axis=-2,
        )  # (T, 3, 3)
        pos_world = np.einsum("tij,tbj->tbi", Rz, pos_local) + root_pos[:, None, :]  # (T, J, 3)

        p = pos_world[:, bone_i, :]  # (T, 3)
        h = p[:, 2].astype(np.float64)
        v = np.zeros_like(p, dtype=np.float64)
        v[1:] = (p[1:].astype(np.float64) - p[:-1].astype(np.float64)) * float(fps)
        vxy = np.linalg.norm(v[:, :2], axis=-1)
        vz = np.abs(v[:, 2])

        return {
            "fk_height": h,
            "fk_vxy": vxy,
            "fk_vz": vz,
        }
    except Exception:
        return None


def _extract_posehist_fk(
    d_pred: Dict[str, object],
    source_json: Path,
    *,
    bone: str,
    hist_idx: int,
    fps: float,
    pretrain_template: Path,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Derive FK kinematics from the model input `aux_inputs.pose_hist_norm` by inverting the
    pose-history tanh+zscore normalizer (VectorTanhNormalizer).
    """
    try:
        aux = d_pred.get("aux_inputs", {})
        if not isinstance(aux, dict):
            return None
        pose_hist_norm = np.asarray(aux.get("pose_hist_norm", []), dtype=np.float32)
        if pose_hist_norm.ndim != 2 or pose_hist_norm.shape[0] == 0:
            return None
        T = int(pose_hist_norm.shape[0])
        D = int(pose_hist_norm.shape[1])
        if D <= 0:
            return None

        tpl = _load_json(pretrain_template)
        scales = np.asarray(tpl.get("tanh_scales_pose_hist", []), dtype=np.float32)
        mu = np.asarray(tpl.get("MuPoseHist", []), dtype=np.float32)
        std = np.asarray(tpl.get("StdPoseHist", []), dtype=np.float32)
        if scales.size != D or mu.size != D or std.size != D:
            return None

        # Lazy torch import.
        import torch

        from train.geometry import fk_positions_from_rot6d
        from train.data.normalizers import VectorTanhNormalizerTorch

        norm_t = torch.from_numpy(pose_hist_norm)
        vt = VectorTanhNormalizerTorch(
            torch.from_numpy(scales),
            torch.from_numpy(mu),
            torch.from_numpy(std),
        )
        pose_hist_raw = vt.inverse(norm_t).detach().cpu().numpy()  # (T, D)

        # Infer hist_len and J from the source skeleton.
        src = _load_json(source_json)
        sk = src.get("meta", {}).get("skeleton", {})
        if not isinstance(sk, dict):
            return None
        names = sk.get("bone_names", [])
        parents = sk.get("parents", [])
        offsets = sk.get("ref_local_offsets_m", [])
        if not isinstance(names, list) or not isinstance(parents, list) or not isinstance(offsets, list):
            return None
        J = int(len(names))
        if J <= 0 or (D % (J * 6)) != 0:
            return None
        L = int(D // (J * 6))
        if L <= 0:
            return None
        idx = int(hist_idx)
        if idx < 0:
            idx = L + idx
        idx = max(0, min(L - 1, idx))
        dt = idx - (L - 1)

        hist = pose_hist_raw.reshape(T, L, J, 6)
        rot6d = hist[:, idx, :, :]  # (T, J, 6)

        offsets_t = torch.tensor(offsets, dtype=torch.float32)
        pos_local = fk_positions_from_rot6d(torch.from_numpy(rot6d), parents, offsets_t, root_pos=None).detach().cpu().numpy()

        frames = src.get("Frames")
        if not isinstance(frames, list) or not frames:
            return None
        yaw_all = np.asarray([fr.get("RootYaw", 0.0) for fr in frames], dtype=np.float32).reshape(-1)
        root_pos_all = np.asarray([fr.get("RootPosition", [0.0, 0.0, 0.0]) for fr in frames], dtype=np.float32)
        if root_pos_all.ndim != 2 or root_pos_all.shape[1] != 3:
            root_pos_all = np.zeros((yaw_all.shape[0], 3), dtype=np.float32)
        idxs = np.clip(np.arange(T, dtype=np.int64) + int(dt), 0, yaw_all.shape[0] - 1)
        yaw = yaw_all[idxs]
        root_pos = root_pos_all[idxs]
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            root_pos = np.zeros((T, 3), dtype=np.float32)
        c = np.cos(yaw)
        s = np.sin(yaw)
        Rz = np.stack(
            [
                np.stack([c, -s, np.zeros_like(c)], axis=-1),
                np.stack([s, c, np.zeros_like(c)], axis=-1),
                np.stack([np.zeros_like(c), np.zeros_like(c), np.ones_like(c)], axis=-1),
            ],
            axis=-2,
        )  # (T, 3, 3)
        pos_world = np.einsum("tij,tbj->tbi", Rz, pos_local) + root_pos[:, None, :]

        bone_i = _bone_index_from_source(source_json, bone)
        if bone_i is None:
            return None
        p = pos_world[:, int(bone_i), :].astype(np.float64)  # (T, 3)
        h = p[:, 2]
        v = np.zeros_like(p)
        v[1:] = (p[1:] - p[:-1]) * float(fps)
        vxy = np.linalg.norm(v[:, :2], axis=-1)
        vz = np.abs(v[:, 2])
        return {
            "posehist_fk_height": h,
            "posehist_fk_vxy": vxy,
            "posehist_fk_vz": vz,
        }
    except Exception:
        return None


def _extract_foot_evidence(frames: List[dict], *, side: str, key: str, T: int) -> Optional[np.ndarray]:
    side = str(side).upper()
    if side not in ("L", "R"):
        return None
    out: List[float] = []
    for fr in frames[:T]:
        try:
            fe = fr.get("FootEvidence", {})
            sd = fe.get(side, {}) if isinstance(fe, dict) else {}
            out.append(float(sd.get(key)))
        except Exception:
            out.append(float("nan"))
    if not out:
        return None
    return np.asarray(out, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot event-aligned contact_meas curves (pred vs GT).")
    ap.add_argument("--root", type=str, required=True, help="Root dir that contains condition subdirs.")
    ap.add_argument("--conds", nargs="*", default=None, help="Optional condition names (subdirs) to include.")
    ap.add_argument("--out", type=str, required=True, help="Output directory for PNGs.")
    ap.add_argument("--channel", type=str, default="R", choices=("L", "R"), help="Which contact channel to plot.")
    ap.add_argument("--event", type=str, default="falling", choices=("rising", "falling"), help="Which GT edge to align on.")
    ap.add_argument("--pre", type=int, default=10, help="Frames before GT edge to include.")
    ap.add_argument("--post", type=int, default=30, help="Frames after GT edge to include.")
    ap.add_argument("--on-th", type=float, default=0.8, help="GT Schmitt ON threshold.")
    ap.add_argument("--off-th", type=float, default=0.1, help="GT Schmitt OFF threshold.")
    ap.add_argument("--mid-th", type=float, default=0.55, help="Mid threshold used for annotation.")
    ap.add_argument(
        "--overlay",
        nargs="*",
        default=[],
        choices=(
            "foot_vz",
            "foot_vxy",
            "foot_height",
            "foot_dist",
            "fk_vz",
            "fk_vxy",
            "fk_height",
            "posehist_fk_vz",
            "posehist_fk_vxy",
            "posehist_fk_height",
            "angvel_mag",
        ),
        help="Optional extra curves plotted in a second panel.",
    )
    ap.add_argument(
        "--bone",
        type=str,
        default=None,
        help="Bone name for --overlay angvel_mag (defaults to foot_l/foot_r based on --channel).",
    )
    ap.add_argument(
        "--posehist-idx",
        type=int,
        default=-1,
        help="Which pose_hist frame to convert to FK (0=oldest, -1=newest).",
    )
    ap.add_argument(
        "--pretrain-template",
        type=str,
        default="models/pretrain_template.json",
        help="Template JSON that contains pose-history tanh/zscore stats (tanh_scales_pose_hist, MuPoseHist, StdPoseHist).",
    )
    ap.add_argument("--max-clips", type=int, default=0, help="Optional limit (0 = no limit).")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_mpl(out_dir)

    conds = _discover_conditions(root, conds=args.conds)
    if not conds:
        raise SystemExit("[FATAL] No condition subdirs found under --root.")

    clip_index = _build_clip_index(root, conds)
    clips = sorted(clip_index.keys())
    if args.max_clips and int(args.max_clips) > 0:
        clips = clips[: int(args.max_clips)]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    ch = 0 if args.channel == "L" else 1
    event = str(args.event)
    side = "L" if ch == 0 else "R"
    on_th = float(args.on_th)
    off_th = float(args.off_th)
    mid_th = float(args.mid_th)
    pre = int(args.pre)
    post = int(args.post)
    overlays = [str(x) for x in (args.overlay or [])]
    bone_name = str(args.bone) if args.bone else ("foot_l" if ch == 0 else "foot_r")
    posehist_idx = int(args.posehist_idx)
    pretrain_template = Path(args.pretrain_template).expanduser().resolve()

    wrote = 0
    for clip in clips:
        cond_to_path = clip_index.get(clip, {})
        if not cond_to_path:
            continue
        # Use the first available condition as GT source (GT should match across conditions).
        first_path = next(iter(cond_to_path.values()))
        data0 = _load_json(first_path)
        gt = np.asarray(data0.get("aux_inputs", {}).get("contacts", []), dtype=np.float64)
        if gt.ndim != 2 or gt.shape[1] <= ch:
            continue
        T = int(gt.shape[0])
        gt_edges = _pick_event_indices(gt, ch=ch, on_th=on_th, off_th=off_th, event=event)
        if gt_edges.size == 0:
            continue
        # Plot first GT edge for compactness.
        t0 = int(gt_edges[0])

        if overlays:
            fig, (ax, ax2) = plt.subplots(
                2,
                1,
                figsize=(10, 6.5),
                gridspec_kw={"height_ratios": [2.5, 1.0]},
                sharex=True,
            )
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax2 = None

        t_gt, y_gt = _extract_window(gt[:, ch], t0, pre=pre, post=post)
        ax.plot(t_gt, y_gt, color="black", linewidth=2.0, label="GT (soft)")

        for cond in conds:
            p = cond_to_path.get(cond)
            if p is None or not p.is_file():
                continue
            d = _load_json(p)
            pred = np.asarray(d.get("contacts_pred", {}).get("contacts_meas", []), dtype=np.float64)
            if pred.ndim != 2 or pred.shape[1] <= ch:
                continue
            t_pr, y_pr = _extract_window(pred[:, ch], t0, pre=pre, post=post)
            pred_at = float(pred[t0, ch]) if 0 <= t0 < pred.shape[0] else float("nan")
            dt_mid = _time_to_le(pred[:, ch], t0, thr=mid_th, post=post)
            label = f"{cond} (p@0={pred_at:.3f}, dt<={mid_th:.2f}={dt_mid})"
            ax.plot(t_pr, y_pr, linewidth=1.8, label=label)

            if ax2 is not None and "angvel_mag" in overlays:
                src_json = d.get("source_json")
                if isinstance(src_json, str):
                    bone_i = _bone_index_from_source(Path(src_json), bone_name)
                else:
                    bone_i = None
                if bone_i is not None:
                    mag = _extract_angvel_mag(d, bone_index=bone_i)
                    if mag is not None and mag.size >= T:
                        t_m, y_m = _extract_window(mag[:T], t0, pre=pre, post=post)
                        ax2.plot(t_m, y_m, linewidth=1.4, label=f"angvel_mag:{cond}:{bone_name}")

        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.axhline(on_th, color="gray", linestyle=":", linewidth=1.0)
        ax.axhline(mid_th, color="gray", linestyle=":", linewidth=1.0)
        ax.axhline(off_th, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlim(-pre, post)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("frames relative to GT edge")
        ax.set_ylabel(f"contact_{args.channel}")
        ax.set_title(f"{clip} | {args.channel} | GT {event} @ t={t0}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        if ax2 is not None:
            src_json = data0.get("source_json")
            frames = _load_raw_frames(Path(src_json)) if isinstance(src_json, str) else None
            fk = (
                _extract_pose_fk(Path(src_json), bone=bone_name, T=T, fps=float(data0.get("fps", 60.0)))
                if isinstance(src_json, str) and any(ov.startswith("fk_") for ov in overlays)
                else None
            )
            posehist_fk = (
                _extract_posehist_fk(
                    data0,
                    Path(src_json),
                    bone=bone_name,
                    hist_idx=posehist_idx,
                    fps=float(data0.get("fps", 60.0)),
                    pretrain_template=pretrain_template,
                )
                if isinstance(src_json, str) and any(ov.startswith("posehist_fk_") for ov in overlays)
                else None
            )
            if frames is not None:
                key_map = {
                    "foot_vz": "vz_mps",
                    "foot_vxy": "vxy_mps",
                    "foot_height": "foot_height_world_m",
                    "foot_dist": "dist_to_ground_m",
                }
                for ov in overlays:
                    if ov not in key_map:
                        continue
                    arr = _extract_foot_evidence(frames, side=side, key=key_map[ov], T=T)
                    if arr is None:
                        continue
                    t_e, y_e = _extract_window(arr, t0, pre=pre, post=post)
                    ax2.plot(t_e, y_e, linewidth=1.6, label=f"{ov}")
            if fk is not None:
                fk_map = {
                    "fk_height": "fk_height",
                    "fk_vxy": "fk_vxy",
                    "fk_vz": "fk_vz",
                }
                for ov in overlays:
                    if ov not in fk_map:
                        continue
                    arr = fk.get(fk_map[ov])
                    if arr is None or not isinstance(arr, np.ndarray) or arr.size < T:
                        continue
                    t_f, y_f = _extract_window(arr[:T], t0, pre=pre, post=post)
                    ax2.plot(t_f, y_f, linewidth=1.6, label=f"{ov}:{bone_name}")
            if posehist_fk is not None:
                ph_map = {
                    "posehist_fk_height": "posehist_fk_height",
                    "posehist_fk_vxy": "posehist_fk_vxy",
                    "posehist_fk_vz": "posehist_fk_vz",
                }
                for ov in overlays:
                    if ov not in ph_map:
                        continue
                    arr = posehist_fk.get(ph_map[ov])
                    if arr is None or not isinstance(arr, np.ndarray) or arr.size < T:
                        continue
                    t_f, y_f = _extract_window(arr[:T], t0, pre=pre, post=post)
                    ax2.plot(t_f, y_f, linewidth=1.6, label=f"{ov}:{bone_name}:idx{posehist_idx}")
            ax2.axvline(0, color="gray", linestyle="--", linewidth=1.0)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylabel("overlay")
            ax2.legend(loc="best", fontsize=8)
            ax2.set_xlabel("frames relative to GT edge")

        fig.tight_layout()

        suffix = "_plus" if overlays else ""
        out_path = out_dir / f"{clip}_{args.channel}_{event}{suffix}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        wrote += 1

    print(f"[OK] wrote {wrote} plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
