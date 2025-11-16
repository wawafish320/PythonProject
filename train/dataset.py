from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import (
    rot6d_to_matrix,
    reproject_rot6d,
    angvel_vec_from_R_seq,
    wrap_to_pi_np as _wrap_to_pi_np,
)
from .io_utils import (
    load_soft_contacts_from_json as _load_soft_contacts_from_json,
    direction_yaw_from_array as _direction_yaw_from_array,
    velocity_yaw_from_array as _velocity_yaw_from_array,
    speed_from_X_layout as _speed_from_X_layout,
    npz_scalar_to_str,
)
from .layout_utils import normalize_layout as _normalize_layout
from .normalizers import VectorTanhNormalizer
from torch.utils.data._utils.collate import default_collate as _default_collate


def _fix_firstdim_any(v, L: int):
    """
    Recursively pad/truncate so that the FIRST dimension equals L for tensors/ndarrays,
    and apply the same fix inside dict/list/tuple. Scalars are kept untouched.
    NOTE: numpy arrays with non-numeric dtype (e.g., strings '<U..' or object) are converted to lists
    to avoid PyTorch default_collate errors.
    """

    if isinstance(v, torch.Tensor):
        if v.dim() >= 1:
            n = v.shape[0]
            if n == L:
                return v
            if n > L:
                return v[:L]
            pad = v[-1:].expand(L - n, *v.shape[1:])
            return torch.cat([v, pad], dim=0)
        return v
    if isinstance(v, np.ndarray):
        if v.dtype.kind in ("U", "S", "O"):
            return v.tolist()
        if v.ndim >= 1:
            n = v.shape[0]
            if n == L:
                return v
            if n > L:
                return v[:L]
            pad = np.repeat(v[-1:, ...], L - n, axis=0)
            return np.concatenate([v, pad], axis=0)
        return v
    if isinstance(v, dict):
        return {k: _fix_firstdim_any(vv, L) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        seq = [_fix_firstdim_any(x, L) for x in v]
        return type(v)(seq)
    return v


def make_fixedlen_collate(seq_len: int):
    DROP_KEYS = {"json_path", "npz_path", "foot_json_path", "source_json", "json_candidates"}

    def _collate(batch):
        fixed = []
        for sample in batch:
            s = _fix_firstdim_any(sample, seq_len)

            for k in list(s.keys()):
                if k in DROP_KEYS:
                    s.pop(k, None)

            for k, v in list(s.items()):
                if isinstance(v, np.ndarray) and v.dtype.kind in ("U", "S", "O"):
                    s[k] = v.tolist()

            fixed.append(s)
        return _default_collate(fixed)

    return _collate

class MotionAugmentation:
    """
    训练时的数据增强：
      - 时间扭曲（等长重采样，反射边界，保持可微）
      - 加性高斯噪声

    """

    def __init__(self, noise_std: float=0.0, time_warp_prob: float=0.0):
        self.noise_std = float(noise_std)
        self.time_warp_prob = float(time_warp_prob)

    @torch.no_grad()
    def _time_warp(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        B, T, D = x.shape
        if T <= 1 or scale <= 0:
            return x
        device = x.device
        t = torch.arange(T, device=device, dtype=torch.float32)
        c = (T - 1) / 2.0
        src = (t - c) / scale + c
        period = 2.0 * (T - 1)
        s = torch.remainder(src, period)
        src_reflect = torch.where(s <= T - 1, s, period - s)
        i0 = torch.clamp(src_reflect.floor().to(torch.long), 0, T - 1)
        i1 = torch.clamp(i0 + 1, 0, T - 1)
        a = (src_reflect - i0.to(src_reflect.dtype)).view(1, T, 1)
        idx0 = i0.view(1, T, 1).repeat(B, 1, D)
        idx1 = i1.view(1, T, 1).repeat(B, 1, D)
        x0 = torch.gather(x, 1, idx0)
        x1 = torch.gather(x, 1, idx1)
        y = torch.lerp(x0, x1, a)
        return y


@dataclass
class ClipData:
    npz_path: str
    X: np.ndarray
    Y: np.ndarray
    C: np.ndarray
    meta: dict[str, Any]
    state_layout_norm: Dict[str, Tuple[int, int]]
    output_layout_norm: Dict[str, Tuple[int, int]]
    contacts: Optional[np.ndarray] = None
    angvel_norm: Optional[np.ndarray] = None
    angvel_raw: Optional[np.ndarray] = None
    pose_hist_norm: Optional[np.ndarray] = None
    bone_rot6d: Optional[np.ndarray] = None


def _infer_forward_axis_from_clip(
    clip: Mapping[str, Any],
    *,
    up_axis: int = 2,
) -> Optional[tuple[int, float]]:
    rot6d = clip.get("bone_rot6d")
    if rot6d is None:
        return None
    try:
        rot6d_arr = np.asarray(rot6d, dtype=np.float32)
    except Exception:
        return None
    if rot6d_arr.ndim != 3 or rot6d_arr.shape[2] != 6 or rot6d_arr.shape[0] < 2:
        return None

    import torch

    root_rot6 = torch.as_tensor(rot6d_arr[:, 0], dtype=torch.float32)
    try:
        root_m = rot6d_to_matrix(reproject_rot6d(root_rot6).view(-1, 1, 6)).view(-1, 3, 3)
    except Exception:
        return None

    up_axis = int(np.clip(up_axis, 0, 2))
    planar_axes = [ax for ax in (0, 1, 2) if ax != up_axis]
    if len(planar_axes) != 2:
        planar_axes = [0, 1]
    ax0, ax1 = planar_axes
    root_vel = clip.get("root_vel")
    yaw_vel = _velocity_yaw_from_array(root_vel)

    best_axis = None
    best_offset = 0.0
    best_score = float("inf")

    if yaw_vel is not None and yaw_vel.size >= 2:
        for axis in (0, 1, 2):
            vec = root_m[:, :, axis]
            planar = vec[:, [ax0, ax1]].cpu().numpy()
            yaw_series = np.arctan2(planar[:, 1], planar[:, 0])
            L = min(len(yaw_series), len(yaw_vel))
            if L <= 1:
                continue
            diff = _wrap_to_pi_np(yaw_series[:L] - yaw_vel[:L])
            score = float(np.nanmedian(np.abs(diff)))
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_axis = axis
                best_offset = float(np.nanmedian(diff))
        if best_axis is not None:
            return best_axis, best_score, best_offset

    target_lists = []
    root_yaw = clip.get("root_yaw")
    if root_yaw is not None:
        try:
            yaw_gt = np.asarray(root_yaw, dtype=np.float32).reshape(-1)
            if yaw_gt.size >= 2:
                target_lists.append(yaw_gt)
        except Exception:
            pass

    for key in ("cond_tgt_raw", "cond_tgt", "cond_in", "traj_dir"):
        yaw_cmd = _direction_yaw_from_array(clip.get(key))
        if yaw_cmd is not None and yaw_cmd.size >= 2:
            target_lists.append(yaw_cmd)

    if not target_lists:
        return None

    best_axis = None
    best_offset = 0.0
    best_score = float("inf")
    for axis in (0, 1, 2):
        vec = root_m[:, :, axis]
        planar = vec[:, [ax0, ax1]].cpu().numpy()
        yaw_series = np.arctan2(planar[:, 1], planar[:, 0])
        errs = []
        offsets = []
        for tgt in target_lists:
            L = min(len(yaw_series), len(tgt))
            if L <= 1:
                continue
            diff = _wrap_to_pi_np(yaw_series[:L] - tgt[:L])
            med = float(np.nanmedian(np.abs(diff)))
            if np.isfinite(med):
                errs.append(med)
                offsets.append(float(np.nanmedian(diff)))
        if errs:
            score = float(np.nanmedian(errs))
            if score < best_score:
                best_score = score
                best_axis = axis
                best_offset = float(np.nanmedian(offsets)) if offsets else 0.0
    if best_axis is not None:
        return best_axis, best_score, best_offset
    return None


class MotionEventDataset(Dataset):
    """
    无状态数据集：
      - 为每帧提供 contacts / angvel / pose_history（与预训练保持一致）
      - cond 序列仍支持归一化等功能
    """

    def __init__(self, data_dir: str, seq_len: int, skeleton_file: None = None, paths: None = None,
                 pose_hist_len: int = 0,
                 norm_spec: Optional[dict] = None,
                 period_dim: int = 0):
        self.norm_stats_inherited = None
        self.paths = sorted(paths) if paths is not None else sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        if not self.paths:
            self.paths = sorted(glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True))
        # [PATCH-B2] drop legacy summary npz that isn't a clip
        self.paths = [p for p in self.paths if os.path.basename(p) != 'normalized_dataset.npz']
        if not self.paths:
            raise FileNotFoundError(f'No .npz files found under {data_dir} (tried *.npz and **/*.npz)')
        self.seq_len = int(seq_len)
        self.clips: list[ClipData] = []
        self.index = []
        self.pose_hist_len = max(0, int(pose_hist_len))
        self.period_dim = int(period_dim)
        self.contact_dim = 2
        self.angvel_norm = None
        self.pose_hist_norm = None
        self.mu_x = np.asarray(norm_spec.get("MuX"), dtype=np.float32) if norm_spec and norm_spec.get("MuX") is not None else None
        self.std_x = np.asarray(norm_spec.get("StdX"), dtype=np.float32) if norm_spec and norm_spec.get("StdX") is not None else None
        if norm_spec:
            ang_scales = norm_spec.get("tanh_scales_angvel")
            if ang_scales is not None:
                self.angvel_norm = VectorTanhNormalizer(
                    np.asarray(ang_scales, dtype=np.float32),
                    np.asarray(norm_spec.get("MuAngVel"), dtype=np.float32) if norm_spec.get("MuAngVel") is not None else None,
                    np.asarray(norm_spec.get("StdAngVel"), dtype=np.float32) if norm_spec.get("StdAngVel") is not None else None,
                )
            pose_scales = norm_spec.get("tanh_scales_pose_hist")
            if pose_scales is not None and len(pose_scales) > 0:
                self.pose_hist_len = max(self.pose_hist_len, int(norm_spec.get("pose_hist_len", self.pose_hist_len)))
                self.pose_hist_norm = VectorTanhNormalizer(
                    np.asarray(pose_scales, dtype=np.float32),
                    np.asarray(norm_spec.get("MuPoseHist"), dtype=np.float32) if norm_spec.get("MuPoseHist") else None,
                    np.asarray(norm_spec.get("StdPoseHist"), dtype=np.float32) if norm_spec.get("StdPoseHist") else None,
                )
        expected_dx = expected_dy = expected_dc = None
        self.up_axis = int((norm_spec or {}).get('up_axis', 2))
        axis_score_sum = np.zeros(3, dtype=np.float64)
        axis_score_cnt = np.zeros(3, dtype=np.int64)
        axis_offset_sum = np.zeros(3, dtype=np.float64)
        axis_offset_cnt = np.zeros(3, dtype=np.int64)
        expected_state_layout: Dict[str, Tuple[int, int]] | None = None
        expected_output_layout: Dict[str, Tuple[int, int]] | None = None
        for p in self.paths:
            try:
                clip = dict(np.load(p, allow_pickle=True, mmap_mode='r'))
                # 优先使用合并后的“分组自适应归一化”大包；无则回退到原始分片
                X = clip.get('X_norm')
                Y = clip.get('Y_norm')
                C = clip.get('cond_in')
                if X is None or Y is None:
                    raise ValueError(f'{p}: missing X_norm/Y_norm; regenerated dataset required.')
                if C is None:
                    raise ValueError(f'{p}: missing cond_in')
                    continue
                X = np.asarray(X, dtype=np.float32).copy()
                Y = np.asarray(Y, dtype=np.float32).copy()
                C = np.asarray(C, dtype=np.float32).copy()
                if X.shape[0] != Y.shape[0]:
                    raise ValueError(f'{p}: X/Y length mismatch {X.shape[0]} vs {Y.shape[0]}')
                if C.shape[0] < X.shape[0]:
                    raise ValueError(f'{p}: cond length {C.shape[0]} shorter than frames {X.shape[0]}')

                s_layout = clip.get('state_layout')
                o_layout = clip.get('output_layout')
                if s_layout is None and 'state_layout_json' in clip:
                    try:
                        s_layout = json.loads(str(clip['state_layout_json']))
                    except Exception:
                        s_layout = None
                if o_layout is None and 'output_layout_json' in clip:
                    try:
                        o_layout = json.loads(str(clip['output_layout_json']))
                    except Exception:
                        o_layout = None

                meta_raw = clip.get('meta_json', None)
                if meta_raw is None:
                    raise ValueError(f'{p}: missing meta_json (regenerate dataset with updated converter).')
                try:
                    if hasattr(meta_raw, 'item'):
                        meta_raw = meta_raw.item()
                    if isinstance(meta_raw, (bytes, bytearray)):
                        meta_raw = meta_raw.decode('utf-8', 'ignore')
                    if not isinstance(meta_raw, str):
                        raise TypeError
                    meta_dict = json.loads(meta_raw)
                except Exception as _meta_err:
                    raise ValueError(f'{p}: failed to parse meta_json ({_meta_err}).') from _meta_err

                fps_val = clip.get('fps', meta_dict.get('fps', 60.0))
                bone_names_val = meta_dict.get('bone_names', clip.get('bone_names', []))
                rot6d_spec_val = meta_dict.get('rot6d_spec', {})
                asset_to_ue_val = meta_dict.get('asset_to_ue', {})
                units_val = meta_dict.get('units', 'meters')
                traj_meta_val = meta_dict.get('trajectory', {})

                if not isinstance(s_layout, dict) or not s_layout:
                    raise ValueError(f'{p}: state_layout missing or empty.')
                if not isinstance(o_layout, dict) or not o_layout:
                    raise ValueError(f'{p}: output_layout missing or empty.')
                if bone_names_val is None:
                    raise ValueError(f'{p}: bone_names missing in meta_json.')
                if units_val is None:
                    raise ValueError(f'{p}: units missing in meta_json.')
                if asset_to_ue_val is None:
                    raise ValueError(f'{p}: asset_to_ue missing in meta_json.')
                if traj_meta_val is None:
                    raise ValueError(f'{p}: trajectory missing in meta_json.')

                meta = dict(meta_dict)
                meta['state_layout'] = s_layout
                meta['output_layout'] = o_layout
                meta['fps'] = float(fps_val)
                meta['bone_names'] = list(bone_names_val)
                meta['rot6d_spec'] = rot6d_spec_val
                meta['asset_to_ue'] = asset_to_ue_val
                meta['units'] = units_val
                meta['trajectory'] = traj_meta_val

                if not isinstance(meta['state_layout'], dict) or not meta['state_layout']:
                    raise ValueError(f'{p}: missing state_layout in bundle/meta.')
                if not isinstance(meta['output_layout'], dict) or not meta['output_layout']:
                    raise ValueError(f'{p}: missing output_layout in bundle/meta.')

                Dx_now, Dy_now, Dc_now = int(X.shape[1]), int(Y.shape[1]), int(C.shape[1])
                state_norm = _normalize_layout(meta['state_layout'], Dx_now)
                output_norm = _normalize_layout(meta['output_layout'], Dy_now)
                state_norm = {k: (int(st), int(sz)) for k, (st, sz) in state_norm.items()}
                output_norm = {k: (int(st), int(sz)) for k, (st, sz) in output_norm.items()}
                if expected_dx is None:
                    expected_dx, expected_dy, expected_dc = Dx_now, Dy_now, Dc_now
                    expected_state_layout = dict(state_norm)
                    expected_output_layout = dict(output_norm)
                else:
                    if (Dx_now, Dy_now, Dc_now) != (expected_dx, expected_dy, expected_dc):
                        raise ValueError(
                            f'{p}: feature dims {(Dx_now, Dy_now, Dc_now)}'
                            f' differ from first clip {(expected_dx, expected_dy, expected_dc)}'
                        )
                    if state_norm != expected_state_layout:
                        raise ValueError(f'{p}: state_layout differs from first clip.')
                    if output_norm != expected_output_layout:
                        raise ValueError(f'{p}: output_layout differs from first clip.')

                T = int(X.shape[0])
                contacts = None
                angvel_norm = None
                angvel_raw = None
                pose_hist_norm = None
                bone_rot6d_raw = clip.get('bone_rot6d')
                if bone_rot6d_raw is not None:
                    bone_rot6d_raw = np.asarray(bone_rot6d_raw, dtype=np.float32)
                    if bone_rot6d_raw.shape[0] < T and bone_rot6d_raw.shape[0] > 0:
                        pad = np.repeat(bone_rot6d_raw[-1:], T - bone_rot6d_raw.shape[0], axis=0)
                        bone_rot6d_raw = np.concatenate([bone_rot6d_raw, pad], axis=0)
                    if bone_rot6d_raw.shape[0] > T:
                        bone_rot6d_raw = bone_rot6d_raw[:T]
                src = clip.get('source_json')
                src_json = npz_scalar_to_str(src) if src is not None else None
                if src_json:
                    if not os.path.isabs(src_json):
                        src_json = os.path.join(os.path.dirname(p), src_json)
                    try:
                        sc_full = _load_soft_contacts_from_json(src_json)
                        if sc_full.shape[0] < T and sc_full.shape[0] > 0:
                            pad = np.repeat(sc_full[-1:], T - sc_full.shape[0], axis=0)
                            sc_full = np.concatenate([sc_full, pad], axis=0)
                        if sc_full.shape[0] > T:
                            sc_full = sc_full[:T]
                        if sc_full.shape[0] == T:
                            contacts = sc_full.astype(np.float32, copy=False)
                        else:
                            print(f"[Dataset] soft_contact length mismatch for {p}: {sc_full.shape[0]} vs {T}")
                    except Exception as _sc_err:
                        print(f"[Dataset] soft_contact unavailable for {p}: {_sc_err}")

                rot_seq = clip.get('y_out_features')
                if rot_seq is not None:
                    rot_seq = np.asarray(rot_seq, dtype=np.float32)
                    if rot_seq.shape[0] != T:
                        rot_seq = rot_seq[:T]
                else:
                    rot_seq = None

                if rot_seq is not None and rot_seq.size > 0:
                    try:
                        J = rot_seq.shape[1] // 6
                        y_t = torch.from_numpy(rot_seq).to(torch.float32)
                        y_t = reproject_rot6d(y_t.unsqueeze(0))[0]
                        R = rot6d_to_matrix(y_t.view(1, T, J, 6))[0]
                        fps = float(meta['fps'])
                        w = angvel_vec_from_R_seq(R.unsqueeze(0), fps)[0].reshape(-1, J * 3).cpu().numpy()
                        if w.shape[0] < T:
                            pad = np.repeat(w[-1:], T - w.shape[0], axis=0)
                            w = np.concatenate([w, pad], axis=0)
                        elif w.shape[0] > T:
                            w = w[:T]
                        angvel_raw = w.astype(np.float32, copy=False)
                        angvel_norm = self.angvel_norm.transform(w) if self.angvel_norm is not None else angvel_raw

                        if self.pose_hist_len > 0:
                            hist_offsets = np.arange(self.pose_hist_len, 0, -1, dtype=np.int64)
                            frame_ids = np.arange(T, dtype=np.int64)[:, None] - hist_offsets[None, :]
                            np.clip(frame_ids, 0, T - 1, out=frame_ids)
                            pose_hist_raw = rot_seq[frame_ids].reshape(T, -1)
                            pose_hist_norm = self.pose_hist_norm.transform(pose_hist_raw) if self.pose_hist_norm is not None else pose_hist_raw.astype(np.float32, copy=False)
                        else:
                            pose_hist_norm = np.zeros((T, 0), dtype=np.float32)
                    except Exception as feat_err:
                        print(f"[Dataset] feature compute failed for {p}: {feat_err}")
                        angvel_norm = None
                        angvel_raw = None
                        pose_hist_norm = None

                clip_obj = ClipData(
                    npz_path=p,
                    X=X,
                    Y=Y,
                    C=C,
                    meta=meta,
                    state_layout_norm=state_norm,
                    output_layout_norm=output_norm,
                    contacts=contacts,
                    angvel_norm=angvel_norm,
                    angvel_raw=angvel_raw,
                    pose_hist_norm=pose_hist_norm,
                    bone_rot6d=bone_rot6d_raw,
                )
                self.clips.append(clip_obj)
                axis_info = _infer_forward_axis_from_clip(
                    {
                        'bone_rot6d': clip.get('bone_rot6d'),
                        'root_yaw': clip.get('root_yaw'),
                        'root_vel': clip.get('root_vel'),
                        'cond_tgt_raw': clip.get('cond_tgt_raw'),
                        'cond_tgt': clip.get('cond_tgt'),
                        'cond_in': clip.get('cond_in'),
                        'traj_dir': clip.get('traj_dir'),
                    },
                    up_axis=self.up_axis,
                )
                if axis_info is not None:
                    axis_idx, score, offset = axis_info
                    axis_score_sum[axis_idx] += float(score)
                    axis_score_cnt[axis_idx] += 1
                    if offset is not None and np.isfinite(offset):
                        axis_offset_sum[axis_idx] += float(offset)
                        axis_offset_cnt[axis_idx] += 1
                cid = len(self.clips) - 1
                if T >= self.seq_len:
                    for s in range(0, T - self.seq_len + 1):
                        self.index.append((cid, s))
            except Exception as e:
                print(f'[Dataset] Warning: skip {p}: {e}')
        if not self.clips:
            raise ValueError('No valid clips were loaded from the dataset.')
        first = self.clips[0]
        self.Dx = int(first.X.shape[1])
        self.Dy = int(first.Y.shape[1])
        self.Dc = int(first.C.shape[1])
        self.fps = float(first.meta['fps'])
        self.bone_names = first.meta['bone_names']
        skeleton_meta = first.meta.get('skeleton', {})
        self.parents = list(skeleton_meta.get('parents', [])) if isinstance(skeleton_meta, dict) else []
        offsets = skeleton_meta.get('ref_local_offsets_m') if isinstance(skeleton_meta, dict) else None
        self.bone_offsets = np.asarray(offsets, dtype=np.float32) if offsets is not None else None
        self.contact_dim = int(first.contacts.shape[1]) if first.contacts is not None else 0
        self.angvel_dim = int(first.angvel_norm.shape[1]) if first.angvel_norm is not None else 0
        self.pose_hist_dim = int(first.pose_hist_norm.shape[1]) if first.pose_hist_norm is not None else 0
        self.encoder_input_dim = self.contact_dim + self.angvel_dim + self.pose_hist_dim
        forward_axis = None
        forward_axis_offset = 0.0
        if axis_score_cnt.sum() > 0:
            avg = np.full(3, np.inf, dtype=np.float64)
            valid = axis_score_cnt > 0
            avg[valid] = axis_score_sum[valid] / np.maximum(axis_score_cnt[valid], 1)
            forward_axis = int(np.argmin(avg))
            self._forward_axis_scores = avg
            if axis_offset_cnt[forward_axis] > 0:
                forward_axis_offset = axis_offset_sum[forward_axis] / max(1, axis_offset_cnt[forward_axis])
        self.forward_axis = forward_axis
        self.forward_axis_offset = float(forward_axis_offset)
        if forward_axis is not None:
            score_deg = float(np.degrees(self._forward_axis_scores[forward_axis]))
            print(f'[ForwardAxis] inferred axis={forward_axis} (median Δyaw={score_deg:.2f}° offset={np.degrees(forward_axis_offset):.2f}°)')
        else:
            print('[ForwardAxis][WARN] unable to infer forward axis; fallback to default later.')

        def _get_layout(m, key):
            """Strict layout getter: only 'state_layout' and 'output_layout' are accepted."""
            if key not in ('state_layout', 'output_layout'):
                return {}
            li = (m or {}).get(key)
            if li is not None and hasattr(li, 'item'):
                li = li.item()
            return li if isinstance(li, dict) else {}

        self.state_layout = _get_layout(first.meta, 'state_layout')
        self.output_layout = _get_layout(first.meta, 'output_layout')
        if not self.state_layout:
            raise ValueError(f"{first.npz_path}: state_layout missing from meta/bundle.")
        if not self.output_layout:
            raise ValueError(f"{first.npz_path}: output_layout missing from meta/bundle.")

        self.state_layout_norm = dict(first.state_layout_norm)
        self.output_layout_norm = dict(first.output_layout_norm)

        traj_meta = first.meta.get('trajectory', {}) or {}
        self.traj_elem_dim = int(traj_meta.get('elem_dim', 2))
        self.traj_plane_axes = list(traj_meta.get('plane_axes', [0, 1]))
        # 现有：
        axes_meta = first.meta.get('axes')
        if not isinstance(axes_meta, dict):
            raise ValueError('meta.axes missing; cannot determine up-axis.')
        axes_lower = {str(k).lower(): str(v).lower() for k, v in axes_meta.items()}
        inv_axes = {v: k for k, v in axes_lower.items()}
        if 'up' not in inv_axes:
            raise ValueError('meta.axes must specify which axis is "up".')
        up_axis_letter = inv_axes['up']
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if up_axis_letter not in axis_map:
            raise ValueError(f'Unsupported up-axis letter: {up_axis_letter}')
        self._up_axis = axis_map[up_axis_letter]
        print(f'[Spec] SRC up-axis = {self._up_axis} (0:X,1:Y,2:Z)')

        # 新增（基于 JSON 模版）：
        _hand = first.meta.get('handedness') or (first.meta.get('trajectory', {}) or {}).get('handedness')
        if not isinstance(_hand, str):
            _hand = 'right'
        _hand_l = str(_hand).lower()
        self._hand_sign = -1.0 if _hand_l.startswith('left') else 1.0
        print(f'[Spec] handedness(from JSON) = {_hand_l}  hand_sign={self._hand_sign:+.0f}')

        stats_meta = (first.meta.get('motion_stats') or {})
        ang_meta = None
        if isinstance(stats_meta, dict):
            ang_meta = stats_meta.get('bone_angular_velocity') or stats_meta.get('BoneAngularVelocities')
        if isinstance(ang_meta, dict):
            try:
                self.angvel_norm_mode = str(ang_meta.get('mode', 'standardize')).lower()
            except Exception:
                self.angvel_norm_mode = 'standardize'
            mean_vals = ang_meta.get('mean', [])
            std_vals = ang_meta.get('std', [])
            self.angvel_mu = np.asarray(mean_vals, dtype=np.float32) if len(mean_vals) else None
            self.angvel_std = np.asarray(std_vals, dtype=np.float32) if len(std_vals) else None
            if self.angvel_std is not None:
                self.angvel_std = np.clip(self.angvel_std, 1e-6, None)
            if self.angvel_mu is None or self.angvel_std is None or self.angvel_mu.size != self.angvel_std.size:
                self.angvel_norm_mode = None
                self.angvel_mu = None
                self.angvel_std = None
        else:
            self.angvel_norm_mode = None
            self.angvel_mu = None
            self.angvel_std = None

        self.yaw_aug_deg = 0.0
        self.is_train = True
        # 统一默认：对 C 做窗口后归一化（post-transform, per-window）
        self.normalize_c = True
        self.c_norm_scope = 'window'  # 固定为 window，避免外部再传错配置

        Cs = [clip.C for clip in self.clips if clip.C is not None and clip.C.shape[1] > 0]
        if Cs:
            Ccat = np.concatenate(Cs, axis=0).astype(np.float32, copy=False)
            self.C_mu, self.C_std = self._robust_mean_std(Ccat)
        else:
            self.C_mu, self.C_std = (None, None)

        self._inject_root_yaw_from_rot6d()

    @staticmethod
    def _window_with_edge_pad(arr: np.ndarray, start: int, length: int) -> np.ndarray:
        end = start + length
        if end <= arr.shape[0]:
            return arr[start:end]
        available = arr[start:]
        if available.shape[0] == 0:
            anchor = arr[-1:, :].copy()
            available = anchor
        needed = length - available.shape[0]
        if needed <= 0:
            return available
        pad_src = available[-1:, :].copy()
        pad = np.repeat(pad_src, needed, axis=0)
        return np.concatenate([available, pad], axis=0)

    @staticmethod
    def _robust_mean_std(arr):
        q1 = np.percentile(arr, 25, axis=0)
        q3 = np.percentile(arr, 75, axis=0)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        mask = (arr >= lo) & (arr <= hi)
        safe = np.where(mask, arr, np.nan)
        mu = np.nanmean(safe, axis=0)
        std = np.nanstd(safe, axis=0)
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        std = np.nan_to_num(std, nan=1e-06, posinf=1e-06, neginf=1e-06).astype(np.float32, copy=False)
        std = np.clip(std, 1e-06, None)
        return (mu.reshape(1, -1), std.reshape(1, -1))

    def _inject_root_yaw_from_rot6d(self):
        if not self.clips:
            return
        if self.mu_x is None or self.std_x is None:
            return
        root_entry = self.clips[0].state_layout_norm.get('RootYaw')
        if root_entry is None:
            return
        st, ln = int(root_entry[0]), int(root_entry[1])
        if ln <= 0:
            return
        end = st + ln
        if end > self.mu_x.shape[0] or end > self.std_x.shape[0]:
            return
        mu_slice = self.mu_x[st:end].reshape(1, -1)
        std_slice = np.clip(self.std_x[st:end].reshape(1, -1), 1e-6, None)
        forward_axis = int(self.forward_axis) if self.forward_axis is not None else 0
        offset = float(getattr(self, 'forward_axis_offset', 0.0) or 0.0)
        up_axis = int(getattr(self, '_up_axis', 2))
        for clip in self.clips:
            rot = clip.bone_rot6d
            if rot is None or rot.size == 0:
                continue
            yaw_raw = self._compute_root_yaw_from_rot6d(rot, forward_axis, up_axis, offset)
            if yaw_raw is None or yaw_raw.size == 0:
                continue
            T = clip.X.shape[0]
            if yaw_raw.shape[0] != T:
                if yaw_raw.shape[0] < T:
                    pad = np.repeat(yaw_raw[-1:], T - yaw_raw.shape[0], axis=0)
                    yaw_raw = np.concatenate([yaw_raw, pad], axis=0)
                else:
                    yaw_raw = yaw_raw[:T]
            yaw_norm = (yaw_raw.reshape(-1, 1) - mu_slice) / std_slice
            clip.X[:, st:end] = yaw_norm.astype(np.float32, copy=False)
            clip.bone_rot6d = None

    @staticmethod
    def _compute_root_yaw_from_rot6d(rot6d_arr: np.ndarray, forward_axis: int, up_axis: int, offset: float) -> Optional[np.ndarray]:
        if rot6d_arr.ndim == 3 and rot6d_arr.shape[-1] == 6:
            root = rot6d_arr[:, 0, :]
        elif rot6d_arr.ndim == 2 and rot6d_arr.shape[-1] == 6:
            root = rot6d_arr
        else:
            return None
        try:
            root_t = torch.from_numpy(root.astype(np.float32, copy=False)).view(-1, 1, 6)
        except Exception:
            return None
        with torch.no_grad():
            try:
                rep = reproject_rot6d(root_t)
                R = rot6d_to_matrix(rep).view(-1, 3, 3)
            except Exception:
                return None
        forward_axis = int(max(0, min(2, forward_axis)))
        up_axis = int(max(0, min(2, up_axis)))
        planar_axes = [ax for ax in (0, 1, 2) if ax != up_axis]
        if len(planar_axes) != 2:
            planar_axes = [0, 1]
        ax0, ax1 = planar_axes
        forward_vec = R[:, :, forward_axis]
        yaw = torch.atan2(forward_vec[:, ax1], forward_vec[:, ax0])
        if offset:
            yaw = yaw - float(offset)
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        return yaw.cpu().numpy()

    @staticmethod
    def _rot2_(xy, c, s):
        x, y = (xy[..., 0].copy(), xy[..., 1].copy())
        xy[..., 0] = c * x - s * y
        xy[..., 1] = s * x + c * y

    @staticmethod
    def _rot3_about_up_(arr_xyz, up_idx, c, s):
        if up_idx == 2:
            i1, i2 = (0, 1)
        elif up_idx == 0:
            i1, i2 = (1, 2)
        else:
            i1, i2 = (2, 0)
        x = arr_xyz[..., i1].copy()
        y = arr_xyz[..., i2].copy()
        arr_xyz[..., i1] = c * x - s * y
        arr_xyz[..., i2] = s * x + c * y

    def _apply_yaw_inplace(self, arr_TD, layout, deg):
        if abs(deg) < 1e-06:
            return
        t = np.deg2rad(deg)
        hs = float(getattr(self, '_hand_sign', 1.0))
        c, s = (np.cos(t), np.sin(t) * hs)

        T, D = arr_TD.shape
        if 'BonePositions' in layout:
            a, b = layout['BonePositions']
            v = arr_TD[:, a:b].reshape(T, -1, 3)
            self._rot3_about_up_(v, self._up_axis, c, s)
            arr_TD[:, a:b] = v.reshape(T, -1)
        if 'BoneVelocities' in layout:
            a, b = layout['BoneVelocities']
            v = arr_TD[:, a:b].reshape(T, -1, 3)
            self._rot3_about_up_(v, self._up_axis, c, s)
            arr_TD[:, a:b] = v.reshape(T, -1)
        if 'BoneRotations6D' in layout:
            a, b = layout['BoneRotations6D']
            vec = arr_TD[:, a:b].reshape(T, -1, 2, 3)
            self._rot3_about_up_(vec[..., 0, :], self._up_axis, c, s)
            self._rot3_about_up_(vec[..., 1, :], self._up_axis, c, s)
            arr_TD[:, a:b] = vec.reshape(T, -1)
        horiz = sorted(list({0, 1, 2} - {self._up_axis}))
        plane_axes = list(self.traj_plane_axes)
        if 'TrajectoryPos' in layout:
            a, b = layout['TrajectoryPos']
            dim = int(self.traj_elem_dim)
            if dim == 3:
                v = arr_TD[:, a:b].reshape(T, -1, 3)
                self._rot3_about_up_(v, self._up_axis, c, s)
                arr_TD[:, a:b] = v.reshape(T, -1)
            elif dim == 2 and sorted(plane_axes) == horiz:
                v = arr_TD[:, a:b].reshape(T, -1, 2)
                self._rot2_(v, c, s)
                arr_TD[:, a:b] = v.reshape(T, -1)
        if 'TrajectoryDir' in layout and int(self.traj_elem_dim) == 2 and (sorted(plane_axes) == horiz):
            a, b = layout['TrajectoryDir']
            v = arr_TD[:, a:b].reshape(T, -1, 2)
            self._rot2_(v, c, s)
            n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-08
            v /= n
            arr_TD[:, a:b] = v.reshape(T, -1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        clip_id, s = self.index[idx]
        s = int(s)
        e = s + int(self.seq_len)
        clip = self.clips[clip_id]

        # Y 已在转换阶段对齐到 “下一帧”，这里不要再 +1
        Xv = clip.X[s:e]
        Yv = clip.Y[s:e]
        C_full = clip.C
        C_in_win = C_full[s:e]
        C_tgt_win = self._window_with_edge_pad(C_full, s + 1, self.seq_len)
        # 双保险
        assert len(Xv) == len(Yv), f"X/Y length mismatch: {len(Xv)} vs {len(Yv)} at s={s}, e={e}"

        need_aug = self.is_train and getattr(self, 'yaw_aug_deg', 0.0) > 0.0
        X = Xv.copy() if need_aug else Xv
        Y = Yv.copy() if need_aug else Yv
        C_in = C_in_win.copy() if need_aug else C_in_win
        C_tgt = C_tgt_win.copy() if need_aug else C_tgt_win
        C_tgt_raw = C_tgt.copy()

        if self.is_train and self.yaw_aug_deg > 0.0:
            deg = float(np.random.uniform(-self.yaw_aug_deg, self.yaw_aug_deg))
            self._apply_yaw_inplace(X, self.state_layout, deg)
            self._apply_yaw_inplace(Y, self.output_layout, deg)
            if C_in.shape[1] >= 2:
                xy = C_in[:, -2:].copy()
                t = np.deg2rad(deg)
                hs = float(getattr(self, '_hand_sign', 1.0))
                c_t, s_t = (np.cos(t), np.sin(t) * hs)

                x, y = (xy[:, 0].copy(), xy[:, 1].copy())
                xy[:, 0] = c_t * x - s_t * y
                xy[:, 1] = s_t * x + c_t * y
                n = np.linalg.norm(xy, axis=-1, keepdims=True) + 1e-08
                C_in[:, -2:] = xy / n
            if C_tgt.shape[1] >= 2:
                xy2 = C_tgt[:, -2:].copy()
                t = np.deg2rad(deg)
                hs = float(getattr(self, '_hand_sign', 1.0))
                c_t, s_t = (np.cos(t), np.sin(t) * hs)
                x2, y2 = (xy2[:, 0].copy(), xy2[:, 1].copy())
                xy2[:, 0] = c_t * x2 - s_t * y2
                xy2[:, 1] = s_t * x2 + c_t * y2
                n2 = np.linalg.norm(xy2, axis=-1, keepdims=True) + 1e-08
                C_tgt[:, -2:] = xy2 / n2

        if self.normalize_c and (C_in.shape[1] > 0):
            # 窗口后归一：对“当前窗口、且已完成全部确定性变换后的 C”做鲁棒均/方
            mu, std = self._robust_mean_std(C_in)
            # 极端情况下做一次兜底（比如某些通道常数）：

            try:
                std = np.clip(np.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6), 1e-6, None)
                mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                # 兜底到全局统计量（仍然做 clip）
                mu = np.nan_to_num(self.C_mu, nan=0.0, posinf=0.0, neginf=0.0) if (self.C_mu is not None) else 0.0
                std = np.nan_to_num(self.C_std, nan=1e-6, posinf=1e-6, neginf=1e-6) if (
                            self.C_std is not None) else 1e-6
                std = np.clip(std, 1e-6, None)
            C_in = (C_in - mu) / std
            C_tgt = (C_tgt - mu) / std
            np.nan_to_num(C_in, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(C_tgt, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(C_in, -6.0, 6.0, out=C_in)
            np.clip(C_tgt, -6.0, 6.0, out=C_tgt)

        sample = {'motion': torch.from_numpy(X).float(), 'gt_motion': torch.from_numpy(Y).float()}
        # 边界一致性所需的索引信息（在 collate 中保持为 [B] 标量）
        try:
            sample['clip_id'] = torch.tensor(int(clip_id), dtype=torch.int64)
            sample['start']   = torch.tensor(int(s), dtype=torch.int64)
        except Exception:
            sample['clip_id'] = torch.tensor(0, dtype=torch.int64)
            sample['start']   = torch.tensor(int(s), dtype=torch.int64)

        if C_in.shape[1] > 0:
            sample['cond_in'] = torch.from_numpy(C_in.astype(np.float32, copy=False)).float()
            sample['cond_tgt'] = torch.from_numpy(C_tgt.astype(np.float32, copy=False)).float()
            sample['cond_tgt_raw'] = torch.from_numpy(C_tgt_raw.astype(np.float32, copy=False)).float()
        if clip.contacts is not None:
            sample['contacts'] = torch.from_numpy(clip.contacts[s:e].astype(np.float32, copy=False)).float()
        else:
            sample['contacts'] = torch.zeros((self.seq_len, self.contact_dim), dtype=torch.float32)
        if clip.angvel_norm is not None:
            sample['angvel'] = torch.from_numpy(clip.angvel_norm[s:e].astype(np.float32, copy=False)).float()
        else:
            sample['angvel'] = torch.zeros((self.seq_len, self.angvel_dim), dtype=torch.float32)
        if getattr(clip, 'angvel_raw', None) is not None:
            sample['angvel_raw'] = torch.from_numpy(clip.angvel_raw[s:e].astype(np.float32, copy=False)).float()
        if clip.pose_hist_norm is not None:
            sample['pose_hist'] = torch.from_numpy(clip.pose_hist_norm[s:e].astype(np.float32, copy=False)).float()
        else:
            sample['pose_hist'] = torch.zeros((self.seq_len, self.pose_hist_dim), dtype=torch.float32)
        return sample

    def set_cond_norm_stats(self, mu, std, inherit=False):
        if isinstance(mu, torch.Tensor):
            mu = mu.detach().cpu().numpy()
        if isinstance(std, torch.Tensor):
            std = std.detach().cpu().numpy()
        self.C_mu = np.array(mu, dtype=np.float32).reshape(1, -1).copy()
        self.C_std = np.array(std, dtype=np.float32).reshape(1, -1).copy()
        self.norm_stats_inherited = bool(inherit)
