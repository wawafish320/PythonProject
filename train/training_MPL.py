from __future__ import annotations


# ===== Common Helpers (extracted) =====

# ========== [Unified Geometry Utilities] ==========
import math as _math
import sys
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict, Mapping, Sequence, Callable, List, Tuple

from .adaptive import (
    AdaptiveLossManager,
    build_adaptive_loss,
    AdaptiveHyperparamScheduler,
)
from .eval_utils import FreeRunSettings, evaluate_teacher, evaluate_freerun
from .geometry import (
    rot6d_to_matrix,
    matrix_to_rot6d,
    compose_rot6d_delta,
    infer_rot6d_delta_from_abs,
    axis_angle_to_matrix,
    geodesic_R,
    so3_log_map,
    angvel_vec_from_R_seq,
    reproject_rot6d,
    root_relative_matrices,
    _root_relative_matrices,
    _matrix_log_map,
    normalize_rot6d_delta,
    _rot6d_identity_like,
    wrap_to_pi_np as _wrap_to_pi_np,
    gram_schmidt_renorm_np,
)
from .layout import (
    parse_layout_entry,
    normalize_layout as _normalize_layout,
    layout_span as _layout_span,
    LayoutCenter,
    DataNormalizer,
    apply_layout_center,
)
from .dataset import (
    MotionAugmentation,
    MotionEventDataset,
    ClipData,
    make_fixedlen_collate,
    _infer_forward_axis_from_clip,
)
from .diagnostics import _maybe_optimize_dataset_index, _norm_debug_once, _parse_stage_schedule
from .io import (
    load_soft_contacts_from_json as _load_soft_contacts_from_json,
    direction_yaw_from_array as _direction_yaw_from_array,
    velocity_yaw_from_array as _velocity_yaw_from_array,
    speed_from_X_layout as _speed_from_X_layout,
    npz_scalar_to_str as _npz_scalar_to_str,
)
try:
    from .utils import (
        build_mlp,
        safe_set_slice,
        expand_paths_from_specs,
        get_flag_value_from_argv,
        get_flag_values_from_argv,
        validate_and_fix_model_,
        sanity_check_model_dims,
        set_global_args,
        get_global_arg,
    )
except ImportError:  # pragma: no cover - fallback for script execution
    from utils import (
        build_mlp,
        safe_set_slice,
        expand_paths_from_specs,
        get_flag_value_from_argv,
        get_flag_values_from_argv,
        validate_and_fix_model_,
        sanity_check_model_dims,
        set_global_args,
        get_global_arg,
    )
from .normalizers import VectorTanhNormalizerTorch

try:
    from .configuration import StageMetricAdjuster, load_val_metrics
except ImportError:  # pragma: no cover - fallback for script execution
    from train.configuration import StageMetricAdjuster, load_val_metrics


import torch.nn as nn

try:
    from .models import MotionEncoder, PeriodHead, EventMotionModel, MotionJointLoss
except ImportError:  # pragma: no cover - fallback for script execution
    from models import MotionEncoder, PeriodHead, EventMotionModel, MotionJointLoss


_arg = get_global_arg


def __apply_layout_center(ds_train, trainer):
    bundle_path = _arg('bundle_json', None)
    apply_layout_center(ds_train, trainer, bundle_path)



import os, json, math, glob, time, argparse

from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:
    print('Warning: tqdm not found. For a progress bar, run: pip install tqdm')

    def tqdm(iterable, *GLOBAL_ARGS, **kwargs):
        return iterable

class Trainer:
    def _parent_relative_matrices(self, R):
        fn = getattr(self.loss_fn, '_parent_relative_matrices', None)
        if callable(fn):
            try:
                return fn(R)
            except Exception:
                return R
        return R

    def _fk_positions(self, R):
        fn = getattr(self.loss_fn, '_fk_positions', None)
        if callable(fn):
            try:
                return fn(R)
            except Exception:
                return None
        return None

    def _joint_weights(self, ref_tensor, joint_count):
        fn = getattr(self.loss_fn, '_joint_weight_vector', None)
        if callable(fn):
            try:
                return fn(ref_tensor.device, ref_tensor.dtype, joint_count)
            except Exception:
                pass
        import torch
        return torch.ones(joint_count, device=ref_tensor.device, dtype=ref_tensor.dtype)

    def _rollout_sequence(self, state_seq, cond_seq=None, cond_raw_seq=None, contacts_seq=None, angvel_seq=None, pose_hist_seq=None, *, gt_seq=None, cond_norm_mu=None, cond_norm_std=None, mode='mixed', tf_ratio=1.0):
        self._require_normalizer("Trainer._rollout_sequence")
        import torch
        assert state_seq.dim() == 3, "state_seq expects [B,T,Dx]"
        B, T, _ = state_seq.shape
        mode = str(mode or 'mixed')
        valid_modes = {'teacher', 'free', 'mixed', 'train_free'}
        if mode not in valid_modes:
            raise ValueError(f"_rollout_sequence mode must be one of {valid_modes}, got {mode}")
        self._diag_roll_mode = mode
        allow_grad = mode == 'train_free'
        mix_state_mode = str(getattr(self, 'mixed_state_mode', 'rot6d')).lower()
        mix_full_state = bool(getattr(self, 'mix_full_state', mix_state_mode == 'full'))

        motion = state_seq[:, 0]  # X in Z-domain at t=0
        try:
            motion_raw_local = self.normalizer.denorm_x(motion)
        except Exception as exc:
            self._raise_norm_error("normalizer.denorm_x 在 roll-out 初始化时失败", exc)

        outs = []
        delta_preds = []
        period_preds = []
        hidden_preds = []
        last_attn = None
        y_raw_local = None
        Dy = None
        if gt_seq is not None and gt_seq.dim() == 3:
            y0 = gt_seq[:, 0]
            Dy = y0.shape[-1]
            y_raw_local = self._denorm(y0)
        if Dy is None:
            Dy = int(getattr(self, 'Dy', 0) or (gt_seq.shape[-1] if gt_seq is not None else 0))

        rot6d_slice = getattr(self.train_loader, "rot6d_x_slice", None) if hasattr(self, "train_loader") else None
        if rot6d_slice is None:
            rot6d_slice = getattr(self, 'rot6d_x_slice', None) or getattr(self, 'rot6d_slice', None)
        if not isinstance(rot6d_slice, slice):
            rot6d_slice = slice(0, motion.size(-1))

        if y_raw_local is None and motion_raw_local is not None and Dy:
            slice_len = rot6d_slice.stop - rot6d_slice.start
            if slice_len != Dy:
                self._raise_norm_error("rollout 初始化 rot6d_x_slice 与 Dy 不匹配。")
            y_raw_local = motion_raw_local[..., rot6d_slice].clone()
        if y_raw_local is None and Dy:
            J = Dy // 6
            if J > 0:
                zeros = state_seq.new_zeros((B, J, 6))
                ident = _rot6d_identity_like(zeros).view(B, Dy)
                y_raw_local = ident

        has_time_dim = {
            'cond': callable(getattr(cond_seq, 'dim', None)) and cond_seq.dim() == 3,
            'cond_raw': callable(getattr(cond_raw_seq, 'dim', None)) and getattr(cond_raw_seq, 'dim', lambda: 0)() == 3,
            'contacts': callable(getattr(contacts_seq, 'dim', None)) and contacts_seq.dim() == 3,
            'angvel': callable(getattr(angvel_seq, 'dim', None)) and angvel_seq.dim() == 3,
            'pose_hist': callable(getattr(pose_hist_seq, 'dim', None)) and pose_hist_seq.dim() == 3,
        }

        amp_enabled = bool(getattr(self, 'use_amp', False))
        rot6d_y_slice = getattr(self, 'rot6d_y_slice', None) or rot6d_slice
        pose_hist_len = int(getattr(self, 'pose_hist_len', 0) or 0)
        pose_hist_dim = int(getattr(self, 'pose_hist_dim', 0) or 0)
        pose_hist_stride = pose_hist_dim // pose_hist_len if pose_hist_len > 0 else 0
        pose_hist_enabled = (
            pose_hist_len > 0
            and pose_hist_dim > 0
            and pose_hist_stride > 0
        )
        scales = mu = std = None
        pose_hist_buffer_norm = None
        pose_hist_buffer_raw = None
        if pose_hist_enabled:
            scales, mu, std = self._pose_hist_params(state_seq)
            if scales is None:
                pose_hist_enabled = False
            else:
                if has_time_dim['pose_hist']:
                    initial_norm = pose_hist_seq[:, 0].to(device=state_seq.device, dtype=state_seq.dtype)
                elif isinstance(pose_hist_seq, torch.Tensor) and pose_hist_seq.numel() > 0:
                    initial_norm = pose_hist_seq.to(device=state_seq.device, dtype=state_seq.dtype)
                else:
                    initial_norm = None
                with torch.no_grad():
                    if initial_norm is not None:
                        pose_hist_buffer_norm = initial_norm
                        pose_hist_buffer_raw = self._pose_hist_inverse_vec(initial_norm, scales, mu, std)
                    else:
                        base_rot = y_raw_local[..., rot6d_y_slice]
                        pose_hist_buffer_raw = (
                            base_rot.unsqueeze(1)
                            .repeat(1, pose_hist_len, 1)
                            .reshape(B, pose_hist_dim)
                        )
                        pose_hist_buffer_norm = self._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)

        cond_norm_mu = self._prepare_cond_stat(cond_norm_mu, state_seq) if cond_norm_mu is not None else None
        cond_norm_std = self._prepare_cond_stat(cond_norm_std, state_seq) if cond_norm_std is not None else None

        # 相对化重投影配置
        enable_reprojection = bool(getattr(self, 'enable_cond_reprojection', True))
        reprojection_applied_count = 0

        for t in range(T):
            self._diag_roll_step = int(t)
            cond_input = cond_seq[:, t] if has_time_dim['cond'] else cond_seq
            contacts_t = contacts_seq[:, t] if has_time_dim['contacts'] else contacts_seq
            angvel_t = angvel_seq[:, t] if has_time_dim['angvel'] else angvel_seq
            if pose_hist_enabled:
                pose_hist_t = pose_hist_buffer_norm
            else:
                pose_hist_t = pose_hist_seq[:, t] if has_time_dim['pose_hist'] else pose_hist_seq
            cond_raw_t = None
            if cond_raw_seq is not None:
                if has_time_dim['cond_raw']:
                    idx = min(cond_raw_seq.shape[1] - 1, max(0, t + 1))
                    cond_raw_t = cond_raw_seq[:, idx]
                elif torch.is_tensor(cond_raw_seq):
                    cond_raw_t = cond_raw_seq
                else:
                    cond_raw_t = cond_raw_seq

            cond_raw_for_env = cond_raw_t
            cond_raw_for_model = cond_raw_t

            # === 相对化重投影：当使用模型预测时，将目标方向转换到模型的局部坐标系 ===
            if enable_reprojection and t > 0 and mode in ('free', 'train_free', 'mixed') and cond_raw_t is not None:
                try:
                    # 获取 GT 的根骨朝向
                    gt_yaw = None
                    if gt_seq is not None and has_time_dim.get('cond_raw'):
                        gt_idx = min(gt_seq.shape[1] - 1, t)
                        gt_raw = self._denorm(gt_seq[:, gt_idx])
                        gt_yaw = self._infer_root_yaw_from_rot6d(gt_raw)
                    elif state_seq is not None:
                        # 从当前输入状态推断
                        state_raw = self.normalizer.denorm_x(state_seq[:, t], prev_raw=motion_raw_local)
                        gt_yaw = self._infer_root_yaw_from_rot6d(state_raw)

                    # 获取模型预测的根骨朝向
                    pred_yaw = None
                    if y_raw_local is not None:
                        pred_yaw = self._infer_root_yaw_from_rot6d(y_raw_local)

                    # 执行重投影
                    if gt_yaw is not None and pred_yaw is not None:
                        cond_raw_t_reprojected = self._reproject_cond_to_local_frame(
                            cond_raw_t, gt_yaw, pred_yaw
                        )
                        if cond_raw_t_reprojected is not None:
                            cond_raw_for_model = cond_raw_t_reprojected
                            reprojection_applied_count += 1
                except Exception as e:
                    # 重投影失败时，回退到原始 cond（静默失败，不中断训练）
                    if getattr(self, '_reprojection_warn_once', True):
                        print(f"[Warning] Cond reprojection failed at step {t}: {e}")
                        self._reprojection_warn_once = False

            if cond_raw_for_model is not None:
                cond_override = self._normalize_cond_from_raw(cond_raw_for_model, cond_norm_mu, cond_norm_std)
                if cond_override is not None:
                    cond_input = cond_override

            if cond_input is None and cond_seq is not None:
                cond_input = cond_seq[:, t] if has_time_dim['cond'] else cond_seq

            with self._amp_context(amp_enabled):
                ret = self.model(
                    motion,
                    cond_input,
                    contacts=contacts_t,
                    angvel=angvel_t,
                    pose_history=pose_hist_t,
                )

            if not isinstance(ret, dict):
                raise RuntimeError("Model forward must return a dict with at least 'out'.")
            delta_out = ret.get('delta', ret.get('out', None))
            period_pred = ret.get('period_pred', None)
            last_attn = ret.get('attn', last_attn)

            if delta_out is None:
                raise RuntimeError("Model forward must return 'delta' tensor.")

            delta_preds.append(delta_out)
            hidden_step = ret.get('h_final')
            if hidden_step is not None:
                if hidden_step.dim() == 1:
                    hidden_step = hidden_step.unsqueeze(0).unsqueeze(0)
                elif hidden_step.dim() == 2:
                    hidden_step = hidden_step.unsqueeze(1)
                elif hidden_step.dim() >= 3 and hidden_step.size(1) != 1:
                    hidden_step = hidden_step[:, -1:, ...]
                hidden_preds.append(hidden_step)

            prev_raw_snapshot = y_raw_local.clone() if y_raw_local is not None else None
            y_raw = self._compose_delta_to_raw(y_raw_local, delta_out)
            if y_raw is None:
                self._raise_norm_error("compose_delta_to_raw 返回 None，缺少上一帧 RAW 数据。")

            if allow_grad:
                y_raw_local = y_raw.clone()
            else:
                y_raw_local = y_raw.detach()
            y_norm = self._norm_y(y_raw)

            debug_stats = {}
            try:
                debug_stats['delta_norm_abs_mean'] = float(delta_out.abs().mean().item())
            except Exception:
                debug_stats['delta_norm_abs_mean'] = None
            try:
                if prev_raw_snapshot is not None:
                    delta_raw = y_raw - prev_raw_snapshot
                    debug_stats['delta_raw_abs_mean'] = float(delta_raw.abs().mean().item())
                else:
                    debug_stats['delta_raw_abs_mean'] = None
            except Exception:
                debug_stats['delta_raw_abs_mean'] = None
            debug_stats['rot6d_geo_deg'] = None

            outs.append(y_norm)
            if period_pred is not None:
                period_preds.append(period_pred)
            debug_stats['rot6d_geo_deg'] = None
            if gt_seq is not None and gt_seq.dim() == 3:
                try:
                    gt_frame = self._denorm(gt_seq[:, min(t + 1, gt_seq.shape[1] - 1)])
                    rot_slice = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
                    if isinstance(rot_slice, slice):
                        pred_block = y_raw_local[:, rot_slice].reshape(y_raw_local.shape[0], -1, 6)
                        gt_block = gt_frame[:, rot_slice].reshape(gt_frame.shape[0], -1, 6)
                        pred_m = rot6d_to_matrix(reproject_rot6d(pred_block))
                        gt_m = rot6d_to_matrix(reproject_rot6d(gt_block))
                        geo_diff = geodesic_R(pred_m, gt_m) * (180.0 / math.pi)
                        debug_stats['rot6d_geo_deg'] = float(geo_diff.mean().item())
                except Exception:
                    debug_stats['rot6d_geo_deg'] = None
            self._last_step_debug_stats = debug_stats

            if t < T - 1:
                if mode == 'teacher':
                    # Next input is GT (Z-domain); sync RAW for potential later use
                    motion = state_seq[:, t + 1]
                    try:
                        motion_raw_local = self.normalizer.denorm_x(motion, prev_raw=motion_raw_local)
                    except Exception as exc:
                        self._raise_norm_error("normalizer.denorm_x 在 teacher forcing 同步时失败", exc)

                elif mode in ('free', 'train_free'):
                    if motion_raw_local is None:
                        self._raise_norm_error("free-run 模式需要 DataNormalizer 提供 RAW 状态写回。")
                    next_raw = self._apply_free_carry(motion_raw_local, y_raw, cond_next_raw=cond_raw_for_env)
                    if mode == 'free':
                        next_raw = next_raw.detach()
                    else:
                        next_raw = next_raw.clone()
                    motion_raw_local = next_raw
                    motion = self._diag_norm_x(motion_raw_local)
                # For teacher mode, allow ground-truth raw to override for next delta composition if available
                if mode == 'teacher' and gt_seq is not None:
                    y_next = self._denorm(gt_seq[:, t + 1])
                    y_raw_local = y_next.detach()

                elif mode == 'mixed':  # scheduled sampling (rot6d-only legacy vs full-state)
                    if motion_raw_local is None:
                        self._raise_norm_error("mixed 模式需要 DataNormalizer 提供 RAW 状态写回。")
                    free_raw = self._apply_free_carry(motion_raw_local, y_raw, cond_next_raw=cond_raw_for_env).detach()
                    free_z = self._diag_norm_x(free_raw)
                    gt_next   = state_seq[:, t + 1]
                    sel = (torch.rand(B, device=self.device) < float(tf_ratio)).float().unsqueeze(-1)
                    if sel.dtype != gt_next.dtype:
                        sel = sel.to(gt_next.dtype)
                    if mix_full_state:
                        motion = sel * gt_next + (1.0 - sel) * free_z
                    else:
                        sx = rot6d_slice
                        motion_next = gt_next.clone()
                        motion_next[..., sx] = sel * gt_next[..., sx] + (1.0 - sel) * free_z[..., sx]
                        motion = motion_next
                    # resync RAW for next step
                    try:
                        motion_raw_local = self.normalizer.denorm_x(motion, prev_raw=motion_raw_local)
                    except Exception as exc:
                        self._raise_norm_error("normalizer.denorm_x 在 mixed 模式同步时失败", exc)
                if pose_hist_enabled and pose_hist_stride > 0:
                    with torch.no_grad():
                        pose_hist_buffer_raw = torch.roll(pose_hist_buffer_raw, shifts=-pose_hist_stride, dims=-1)
                        pose_hist_buffer_raw[..., -pose_hist_stride:] = y_raw_local[..., rot6d_y_slice]
                        pose_hist_buffer_norm = self._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)

        y = torch.stack(outs, dim=1)
        preds = {'out': y, 'delta': torch.stack(delta_preds, dim=1)}
        if hidden_preds:
            hidden_seq = torch.cat(hidden_preds, dim=1)
            preds['hidden_seq'] = hidden_seq
        if period_preds:
            preds['period_pred'] = torch.stack(
                [p if p.dim() == 2 else p.squeeze(1) for p in period_preds], dim=1
            )

        # 诊断：报告重投影应用情况
        if enable_reprojection and reprojection_applied_count > 0:
            diag_limit = int(getattr(self, '_reprojection_diag_limit', 3))
            if not hasattr(self, '_reprojection_diag_count'):
                self._reprojection_diag_count = 0
            if self._reprojection_diag_count < diag_limit:
                epoch = getattr(self, 'cur_epoch', -1)
                print(f"[CondReprojection] Epoch {epoch}, Mode '{mode}': Applied reprojection to {reprojection_applied_count}/{T} steps")
                self._reprojection_diag_count += 1

        self._diag_roll_mode = None
        self._diag_roll_step = -1
        return preds, last_attn

    def _maybe_apply_teacher_noise(self, state_seq: torch.Tensor) -> torch.Tensor:
        noise_deg = float(getattr(self, 'teacher_rot_noise_deg', 0.0) or 0.0)
        noise_prob = float(getattr(self, 'teacher_rot_noise_prob', 0.0) or 0.0)
        if noise_deg <= 1e-6 or noise_prob <= 0.0:
            return state_seq
        rot_slice = getattr(self, 'rot6d_x_slice', None)
        if not isinstance(rot_slice, slice):
            return state_seq
        rot_chunk = state_seq[..., rot_slice]
        B, T, D = rot_chunk.shape
        if D % 6 != 0 or T < 2:
            return state_seq
        J = D // 6
        device = state_seq.device
        rotJ = rot_chunk.view(B, T, J, 6)
        R = rot6d_to_matrix(rotJ)
        B, T, J = R.shape[:3]
        mask_shape = R.shape[:-2]
        mask = (torch.rand(mask_shape, device=device) < float(noise_prob))
        if not mask.any():
            return state_seq
        max_rad = float(noise_deg) * (_math.pi / 180.0)
        angles = torch.empty(B, T, J, device=device, dtype=state_seq.dtype).uniform_(-max_rad, max_rad)
        axes = torch.randn(B, T, J, 3, device=device, dtype=state_seq.dtype)
        delta_R = axis_angle_to_matrix(axes, angles)
        R_noisy = torch.matmul(delta_R, R)
        mask_mat = mask.unsqueeze(-1).unsqueeze(-1)
        R = torch.where(mask_mat, R_noisy, R)
        rot_noisy = matrix_to_rot6d(R).view(B, T, J * 6)
        out = state_seq.clone()
        out[..., rot_slice] = rot_noisy
        return out

    def _freerun_loss_window(self, state_seq, gt_seq, cond_seq, cond_raw_seq, contacts_seq,
                             angvel_seq, pose_hist_seq, batch, *, start: int, length: int,
                             train_mode: bool = False, return_preds: bool = False,
                             cond_norm_mu: Optional[torch.Tensor] = None,
                             cond_norm_std: Optional[torch.Tensor] = None):
        if state_seq is None or gt_seq is None:
            return None
        T = state_seq.shape[1]
        start = max(0, int(start))
        stop = min(T, start + int(length))
        if stop - start < 2:
            return None

        def _slice_tensor(tensor):
            if tensor is None:
                return None
            if hasattr(tensor, 'dim') and tensor.dim() == 3 and tensor.size(1) >= stop:
                return tensor[:, start:stop]
            return tensor

        state_sub = state_seq[:, start:stop]
        gt_sub = gt_seq[:, start:stop]
        cond_sub = _slice_tensor(cond_seq)
        cond_raw_sub = _slice_tensor(cond_raw_seq)
        contacts_sub = _slice_tensor(contacts_seq)
        angvel_sub = _slice_tensor(angvel_seq)
        pose_hist_sub = _slice_tensor(pose_hist_seq)

        mode = 'train_free' if train_mode else 'free'
        preds_free, attn_free = self._rollout_sequence(
            state_sub,
            cond_sub,
            cond_raw_sub,
            contacts_seq=contacts_sub,
            angvel_seq=angvel_sub,
            pose_hist_seq=pose_hist_sub,
            gt_seq=gt_sub,
            mode=mode,
            tf_ratio=0.0,
            cond_norm_mu=cond_norm_mu,
            cond_norm_std=cond_norm_std,
        )
        preds_free = self._ensure_rot6d_delta(preds_free)
        with self._amp_context(self.use_amp):
            out = self.loss_fn(preds_free, gt_sub, attn_weights=attn_free, batch=batch)
        if isinstance(out, tuple):
            free_loss, stats = out
        else:
            free_loss, stats = out, {}
        if return_preds:
            return free_loss, stats or {}, preds_free, gt_sub
        return free_loss, stats or {}, None, None

    def _short_freerun_loss(self, state_seq, gt_seq, cond_seq, cond_raw_seq, contacts_seq,
                             angvel_seq, pose_hist_seq, batch, cond_norm_mu=None, cond_norm_std=None):
        horizon = int(getattr(self, 'freerun_horizon', 0) or 0)
        weight = float(getattr(self, 'freerun_weight', 0.0))
        if horizon <= 0 or weight <= 0.0:
            return None
        T = state_seq.shape[1]
        window = min(int(horizon) + 1, T)
        if window < 2:
            return None
        max_start = T - window
        if max_start > 0:
            start = int(torch.randint(0, max_start + 1, (1,), device=state_seq.device).item())
        else:
            start = 0
        payload = self._freerun_loss_window(
            state_seq, gt_seq, cond_seq, cond_raw_seq, contacts_seq, angvel_seq, pose_hist_seq,
            batch, start=start, length=window, train_mode=True,
            cond_norm_mu=cond_norm_mu, cond_norm_std=cond_norm_std,
        )
        if payload is None:
            return None
        free_loss, stats, _, _ = payload
        return free_loss, stats

    def _ensure_rot6d_delta(self, preds_dict):
        import torch
        if not isinstance(preds_dict, dict):
            return preds_dict
        if torch.is_tensor(preds_dict.get('delta')):
            return preds_dict
        rot_sl = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
        pred_out = preds_dict.get('out')
        if pred_out is None or not isinstance(rot_sl, slice):
            return preds_dict
        try:
            pred_out_raw = self._denorm(pred_out)
        except Exception as exc:
            self._raise_norm_error("_ensure_rot6d_delta 反归一化预测失败", exc)
            return preds_dict
        rot_delta_abs = infer_rot6d_delta_from_abs(pred_out_raw[..., rot_sl])
        if rot_delta_abs is None:
            return preds_dict
        rot_width = rot_sl.stop - rot_sl.start
        if rot_width <= 0 or rot_width % 6 != 0:
            return preds_dict
        J = rot_width // 6
        identity = _rot6d_identity_like(rot_delta_abs.view(*rot_delta_abs.shape[:-1], J, 6)).view_as(rot_delta_abs)
        rot_delta_residual = rot_delta_abs - identity
        std = getattr(self, 'std_y', None)
        if std is None:
            std = getattr(self, 'StdY', None)
        if std is not None:
            std_t = self._cached_norm_param('std_y', std, pred_out)
            if std_t is not None:
                std_slice = std_t[..., rot_sl]
                while std_slice.dim() < rot_delta_residual.dim():
                    std_slice = std_slice.unsqueeze(0)
                rot_delta_residual = rot_delta_residual / std_slice.clamp_min(1e-6)
        delta_full = torch.zeros_like(pred_out)
        delta_full[..., rot_sl] = rot_delta_residual
        preds_out = dict(preds_dict)
        preds_out['delta'] = delta_full
        preds_out['_delta_fallback'] = True
        return preds_out

    def _current_freerun_weight(self) -> float:
        base = float(getattr(self, 'freerun_weight', 0.0) or 0.0)
        if base <= 0.0:
            return 0.0
        mode = str(getattr(self, 'freerun_weight_mode', 'epoch_linear') or 'epoch_linear').lower()
        init = float(getattr(self, 'freerun_weight_init', 0.0) or 0.0)
        init = max(0.0, min(init, base))
        if mode == 'epoch_linear':
            ramp = max(1, int(getattr(self, 'freerun_weight_ramp_epochs', 1) or 1))
            epoch = max(1, int(getattr(self, 'cur_epoch', 1) or 1))
            factor = min(1.0, max(0.0, epoch / ramp))
            return init + (base - init) * factor
        return base

    def _current_freerun_horizon(self) -> int:
        max_h = int(getattr(self, 'freerun_horizon', 0) or 0)
        if max_h <= 0:
            return 0
        min_h = int(getattr(self, 'freerun_horizon_min', 6) or 6)
        min_h = max(1, min(min_h, max_h))
        init_h = getattr(self, 'freerun_init_horizon', None)
        if init_h is None or init_h <= 0:
            init_h = min(max_h, max(min_h, 8))
        init_h = max(min_h, min(int(init_h), max_h))
        ramp_epochs = max(1, int(getattr(self, 'freerun_horizon_ramp_epochs', 4) or 4))
        epoch = max(1, int(getattr(self, 'cur_epoch', 1) or 1))
        progress = min(1.0, max(0.0, (epoch - 1) / ramp_epochs))
        upper = init_h + int(round((max_h - init_h) * progress))
        return max(min_h, min(upper, max_h))

    def _should_log_freerun_gradients(self, batch_idx: int) -> bool:
        if not bool(getattr(self, 'freerun_grad_log', False)):
            return False
        interval = max(1, int(getattr(self, 'freerun_grad_log_interval', 50) or 50))
        return (batch_idx % interval) == 0

    def _collect_freerun_gradients(self, free_loss, preds_dict, effective_h: int):
        import torch
        if preds_dict is None:
            return None
        predY = preds_dict.get('out') if isinstance(preds_dict, dict) else None
        if predY is None or not predY.requires_grad:
            return None
        grad = torch.autograd.grad(
            free_loss,
            predY,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            return None
        grad = grad.view(grad.shape[0], grad.shape[1], -1)
        per_step = grad.norm(dim=-1).mean(dim=0)
        if per_step.numel() == 0:
            return None
        horizon_idx = max(0, min(int(effective_h), per_step.shape[0] - 1))
        step0 = per_step[0].detach()
        steph = per_step[horizon_idx].detach()
        ratio = float((steph / (step0 + 1e-8)).item())
        return {
            'per_step': per_step.detach().cpu(),
            'step0': float(step0.cpu()),
            'steph': float(steph.cpu()),
            'ratio': ratio,
            'horizon': int(effective_h),
        }

    def _maybe_print_grad_monitor(self, grad_info, epoch: int, batch_idx: int) -> None:
        if not isinstance(grad_info, dict):
            return
        ratio = float(grad_info.get('ratio', float('nan')))
        step0 = float(grad_info.get('step0', float('nan')))
        steph = float(grad_info.get('steph', float('nan')))
        horizon = int(grad_info.get('horizon', -1))
        print(
            f"[FreeGrad][ep {int(epoch):03d}][bi {int(batch_idx):04d}] "
            f"step0={step0:.3e} step{horizon}={steph:.3e} ratio={ratio:.3e}"
        )
        threshold = float(getattr(self, 'freerun_grad_ratio_alert', 1e-2) or 1e-2)
        if ratio < threshold:
            print(
                f"[FreeGrad][warn] ratio<{threshold:.2e}; "
                "考虑缩短 horizon 或引入 skip-connection/latent consistency。"
            )

    def _history_drift_debug(
        self,
        state_seq,
        gt_seq,
        cond_seq,
        cond_raw_seq,
        contacts_seq,
        angvel_seq,
        pose_hist_seq,
        *,
        epoch: int,
        batch_idx: int,
        cond_norm_mu=None,
        cond_norm_std=None,
    ) -> None:
        steps = int(getattr(self, 'history_debug_steps', 0) or 0)
        if steps <= 1:
            return
        steps = min(steps, state_seq.shape[1])
        if steps <= 1:
            return
        with torch.no_grad():
            preds_free, _ = self._rollout_sequence(
                state_seq[:, :steps],
                cond_seq[:, :steps] if isinstance(cond_seq, torch.Tensor) and cond_seq.dim() == 3 else cond_seq,
                cond_raw_seq[:, :steps] if isinstance(cond_raw_seq, torch.Tensor) and cond_raw_seq.dim() == 3 else cond_raw_seq,
                contacts_seq=contacts_seq[:, :steps] if isinstance(contacts_seq, torch.Tensor) and contacts_seq.dim() == 3 else contacts_seq,
                angvel_seq=angvel_seq[:, :steps] if isinstance(angvel_seq, torch.Tensor) and angvel_seq.dim() == 3 else angvel_seq,
                pose_hist_seq=pose_hist_seq[:, :steps] if isinstance(pose_hist_seq, torch.Tensor) and pose_hist_seq.dim() == 3 else pose_hist_seq,
                gt_seq=gt_seq[:, :steps],
                mode='train_free',
                tf_ratio=0.0,
                cond_norm_mu=cond_norm_mu,
                cond_norm_std=cond_norm_std,
            )
        pred_out = preds_free.get('out') if isinstance(preds_free, dict) else None
        if pred_out is None:
            return
        try:
            gt_raw = self._denorm(gt_seq[:, :steps])
            pred_raw = self._denorm(pred_out)
        except Exception as exc:
            print(f"[HistDrift][warn] denorm failed: {exc}")
            return
        rot_slice = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
        root_idx = int(getattr(self, 'eval_root_idx', getattr(self, 'root_idx', 0)))
        stats = {}
        if isinstance(rot_slice, slice):
            rot_width = rot_slice.stop - rot_slice.start
            if rot_width > 0 and rot_width % 6 == 0:
                B = gt_raw.shape[0]
                J = rot_width // 6
                gt_flat = gt_raw[:, :steps, rot_slice].reshape(B * steps, J, 6)
                pred_flat = pred_raw[:, :steps, rot_slice].reshape(B * steps, J, 6)
                gt_m = rot6d_to_matrix(reproject_rot6d(gt_flat)).view(B, steps, J, 3, 3)
                pred_m = rot6d_to_matrix(reproject_rot6d(pred_flat)).view(B, steps, J, 3, 3)
                geo = geodesic_R(pred_m, gt_m) * (180.0 / _math.pi)
                geo_step = geo.mean(dim=(0, 2)).detach().cpu().tolist()
                stats['rot_geo_mean_deg'] = float(geo.mean().item())
                stats['rot_geo_step_deg'] = geo_step
                pred_root = _root_relative_matrices(pred_m, root_idx)
                gt_root = _root_relative_matrices(gt_m, root_idx)
                joint_weights = self._joint_weights(pred_root, J)
                weights_sum = joint_weights.sum().clamp_min(1e-6)
                w = joint_weights.view(1, 1, -1)
                geo_local_rad = geodesic_R(pred_root, gt_root)
                geo_local = geo_local_rad * (180.0 / _math.pi)
                geo_local_mean = (geo_local * w).sum() / (weights_sum * geo_local.shape[0] * geo_local.shape[1])
                stats['rot_local_mean_deg'] = float(geo_local_mean.item())
                stats['rot_local_step_deg'] = ((geo_local * w).sum(dim=-1) / weights_sum).mean(dim=0).detach().cpu().tolist()
                stats['_geo_local_rad'] = geo_local_rad.detach()
                fk_pred = self._fk_positions(pred_root)
                fk_gt = self._fk_positions(gt_root)
                if fk_pred is not None and fk_gt is not None:
                    fk_err = (fk_pred - fk_gt).norm(dim=-1)
                    fk_cm = fk_err * 100.0
                    stats['fk_pos_cm'] = float(((fk_cm * w).sum() / (weights_sum * fk_cm.shape[0] * fk_cm.shape[1])).item())
                    stats['fk_pos_step_cm'] = (((fk_cm * w).sum(dim=-1) / weights_sum).mean(dim=0)).detach().cpu().tolist()
        geo_local_tensor_rad = stats.get('_geo_local_rad')
        limb_summary = {}
        collect_fn = getattr(self.loss_fn, '_collect_limb_geo_stats', None)
        if geo_local_tensor_rad is not None and callable(collect_fn):
            try:
                limb_summary = collect_fn(geo_local_tensor_rad)
            except Exception as exc:
                print(f"[HistDrift][ERR] limb summary failed: {exc}")
        try:
            if not stats:
                return
            geo_val = stats.get('rot_geo_mean_deg', float('nan'))
            ang_val = stats.get('rot_local_mean_deg', float('nan'))
            extra = ""
            if limb_summary:
                limb_raw = limb_summary.get('rot_geo_limb_deg', float('nan'))
                limb_weighted = limb_summary.get('rot_geo_limb_over_torso', float('nan'))
                if _math.isfinite(limb_raw):
                    extra += f" limb={limb_raw:.2f}°"
                if _math.isfinite(limb_weighted):
                    extra += f" limb/torso={limb_weighted:.2f}"
            fk_val = stats.get('fk_pos_cm')
            fk_extra = ""
            if isinstance(fk_val, (float, int)) and _math.isfinite(fk_val):
                fk_extra = f" fk_cm={fk_val:.2f}"
            local_val = stats.get('rot_local_mean_deg', float('nan'))
            local_extra = ""
            if isinstance(local_val, (float, int)) and _math.isfinite(local_val):
                local_extra = f" local={local_val:.2f}°"
            print(
                "[HistDrift]"
                f"[ep {int(epoch):03d}]"
                f"[bi {int(batch_idx):04d}] "
                f"rot_geo={geo_val:.2f}° ang_dir={ang_val:.2f}° steps={steps}{extra}{local_extra}{fk_extra}"
            )
            geo_curve = stats.get('rot_geo_step_deg')
            local_curve = stats.get('rot_local_step_deg')
            fk_curve = stats.get('fk_pos_step_cm')
            geo_local_tensor_rad = stats.get('_geo_local_rad')
            if isinstance(geo_curve, list):
                for idx, val in enumerate(geo_curve, start=1):
                    ang_val_step = local_curve[idx - 1] if isinstance(local_curve, list) and idx - 1 < len(local_curve) else float('nan')
                    local_val_step = ang_val_step
                    fk_val_step = fk_curve[idx - 1] if isinstance(fk_curve, list) and idx - 1 < len(fk_curve) else float('nan')
                    summary_txt = ""
                    if isinstance(geo_local_tensor_rad, torch.Tensor) and geo_local_tensor_rad.shape[1] >= idx and callable(collect_fn):
                        try:
                            step_tensor = geo_local_tensor_rad[:, idx - 1:idx]
                            limb_step = collect_fn(step_tensor)
                        except Exception as exc:
                            print(f"[HistDrift][ERR] limb step summary failed (step={idx}): {exc}")
                            limb_step = None
                        if limb_step:
                            limb_deg = limb_step.get('rot_geo_limb_deg', float('nan'))
                            torso_deg = limb_step.get('rot_geo_torso_deg', float('nan'))
                            if _math.isfinite(limb_deg):
                                summary_txt += f" limb={limb_deg:.2f}°"
                            if _math.isfinite(torso_deg):
                                summary_txt += f" torso={torso_deg:.2f}°"
                    if not _math.isnan(ang_val_step):
                        extra_txt = ""
                        if not (_math.isnan(local_val_step) or local_val_step in (float('inf'), float('-inf'))):
                            extra_txt += f" local={local_val_step:.2f}°"
                        if not (_math.isnan(fk_val_step) or fk_val_step in (float('inf'), float('-inf'))):
                            extra_txt += f" fk_cm={fk_val_step:.2f}"
                        print(
                            "[HistDrift]"
                            f"[ep {int(epoch):03d}]"
                            f"[bi {int(batch_idx):04d}]"
                            f"[step {idx:02d}] rot_geo={val:.2f}° ang_dir={ang_val_step:.2f}°{extra_txt or ''}{summary_txt}"
                        )
        finally:
            stats.pop('_geo_local_rad', None)

    def _joint_group_masks(self, J: int, bone_names: Optional[Sequence[str]] = None):
        masks = {}
        if bone_names:
            torso_idx = []
            prox_idx = []
            dist_idx = []
            for idx, name in enumerate(bone_names):
                lname = str(name).lower()
                if any(key in lname for key in ('spine', 'pelvis', 'root', 'torso', 'chest', 'neck')):
                    torso_idx.append(idx)
                elif any(key in lname for key in ('upperarm', 'thigh', 'clavicle', 'shoulder', 'hip')):
                    prox_idx.append(idx)
                else:
                    dist_idx.append(idx)
        else:
            torso_count = min(5, J)
            prox_count = min(5, max(0, J - torso_count))
            torso_idx = list(range(torso_count))
            prox_idx = list(range(torso_count, torso_count + prox_count))
            dist_idx = list(range(torso_count + prox_count, J))
        def _mask(idxs):
            mask = torch.zeros(J, dtype=torch.bool, device=self.device)
            if idxs:
                valid = [i for i in idxs if 0 <= i < J]
                if valid:
                    mask[valid] = True
            return mask
        masks['torso'] = _mask(torso_idx)
        masks['proximal'] = _mask(prox_idx)
        masks['distal'] = _mask(dist_idx)
        return masks

    def _summarize_angvel_dir(
        self,
        pred_w: Optional[torch.Tensor],
        gt_w: Optional[torch.Tensor],
        *,
        bone_names: Optional[Sequence[str]] = None,
        magnitude_threshold: float = 0.1,
        smooth_window: int = 3,
    ) -> dict:
        if pred_w is None or gt_w is None:
            return {}
        if pred_w.numel() == 0 or gt_w.numel() == 0:
            return {}
        B, T, J, _ = pred_w.shape
        eps = 1e-6
        dot = (pred_w * gt_w).sum(dim=-1)
        norm = pred_w.norm(dim=-1) * gt_w.norm(dim=-1)
        cos = torch.clamp(dot / (norm + eps), -1.0 + 1e-7, 1.0 - 1e-7)
        angle_deg = torch.acos(cos) * (180.0 / _math.pi)
        raw = float(angle_deg.mean().item())
        mag = gt_w.norm(dim=-1)
        weight = (mag > magnitude_threshold).float()
        weighted = float((angle_deg * weight).sum().item() / (weight.sum().item() + eps))
        smooth = weighted
        if smooth_window >= 3 and T >= smooth_window:
            pad = smooth_window // 2
            pred_flat = pred_w.reshape(B, T, J * 3).transpose(1, 2)
            gt_flat = gt_w.reshape(B, T, J * 3).transpose(1, 2)
            pred_s = F.avg_pool1d(pred_flat, kernel_size=smooth_window, stride=1, padding=pad).transpose(1, 2).reshape(B, T, J, 3)
            gt_s = F.avg_pool1d(gt_flat, kernel_size=smooth_window, stride=1, padding=pad).transpose(1, 2).reshape(B, T, J, 3)
            dot_s = (pred_s * gt_s).sum(dim=-1)
            norm_s = pred_s.norm(dim=-1) * gt_s.norm(dim=-1)
            cos_s = torch.clamp(dot_s / (norm_s + eps), -1.0 + 1e-7, 1.0 - 1e-7)
            angle_s = torch.acos(cos_s) * (180.0 / _math.pi)
            smooth = float((angle_s * weight).sum().item() / (weight.sum().item() + eps))
        masks = self._joint_group_masks(J, bone_names)
        group_vals = {}
        for key, mask in masks.items():
            if mask.any():
                mask_f = mask.view(1, 1, J)
                grp_weight = weight * mask_f
                denom = grp_weight.sum().item()
                if denom > 0:
                    grp_val = float((angle_deg * grp_weight).sum().item() / (denom + eps))
                else:
                    grp_val = float('nan')
            else:
                grp_val = float('nan')
            group_vals[key] = grp_val
        return {
            'raw': raw,
            'weighted': weighted,
            'smooth': smooth,
            'torso': group_vals.get('torso', float('nan')),
            'proximal': group_vals.get('proximal', float('nan')),
            'distal': group_vals.get('distal', float('nan')),
        }

    def _apply_stage_schedule(self, epoch: int):
        schedule = getattr(self, 'freerun_stage_schedule', None)
        overrides: Dict[str, Any] = {}
        if not schedule:
            return overrides

        if not hasattr(self, '_stage_active_idx') or self._stage_active_idx is None:
            idx = 0
            for i, stage in enumerate(schedule):
                try:
                    st = int(stage.get('start', 1))
                    ed = int(stage.get('end', st))
                except Exception:
                    continue
                if st <= epoch <= ed:
                    idx = i
                    break
                if epoch >= st:
                    idx = i
            self._activate_stage(idx, epoch)

        if getattr(self, '_stage_pending_advance', False):
            self._advance_stage(epoch)

        def _assign(key: str, value: Any) -> bool:
            target = self
            attr_name = key
            prefix = None
            if '.' in key:
                prefix, attr_name = key.split('.', 1)
                if prefix in ('loss', 'loss_fn'):
                    target = getattr(self, 'loss_fn', None)
                elif prefix in ('opt', 'optimizer'):
                    target = getattr(self, 'optimizer', None)
                elif prefix in ('trainer', 'self'):
                    target = self
                else:
                    target = getattr(self, prefix, None)
            elif not hasattr(target, attr_name):
                loss_candidate = getattr(self, 'loss_fn', None)
                if loss_candidate is not None and hasattr(loss_candidate, attr_name):
                    target = loss_candidate
                    prefix = 'loss'
                else:
                    target = None
            if target is None or not hasattr(target, attr_name):
                return False
            current = getattr(target, attr_name)
            coerced = value
            if current is not None:
                if isinstance(current, bool):
                    coerced = bool(value)
                elif isinstance(current, int) and not isinstance(current, bool):
                    try:
                        coerced = int(round(float(value)))
                    except Exception:
                        coerced = current
                elif isinstance(current, float):
                    try:
                        coerced = float(value)
                    except Exception:
                        coerced = current
            setattr(target, attr_name, coerced)
            key_name = key if prefix else attr_name
            overrides[key_name] = coerced
            return True

        selected = self._current_stage()
        if selected is None:
            return overrides

        while epoch > selected.get('end', epoch) and (self._stage_active_idx or 0) < len(schedule) - 1:
            self._advance_stage(epoch)
            selected = self._current_stage()
            if selected is None:
                return overrides

        params = selected.get('params') or {}
        for key, value in params.items():
            _assign(key, value)

        label = selected.get('label')
        stage_tag = f"{selected.get('start', '?')}-{selected.get('end', '?')}"
        if label:
            stage_tag += f" {label}"
        if overrides:
            summary = ', '.join(f"{k}={overrides[k]}" for k in sorted(overrides))
        else:
            summary = 'no overrides'
        print(f"[StageSched][ep {epoch:03d}] stage={stage_tag} | {summary}")
        return overrides

    # === Adaptive metric-driven tuning helpers ===
    def _get_current_stage(self, config: Mapping[str, Any]):
        schedule = config.get("freerun_stage_schedule", []) if isinstance(config, Mapping) else []
        cur_ep = int(getattr(self, 'cur_epoch', -1))
        for stage in schedule:
            rng = stage.get("range")
            if rng:
                start = int(rng[0])
                end = int(rng[-1] if len(rng) > 1 else rng[0])
            else:
                start = int(stage.get("start", 1))
                end = int(stage.get("end", start))
            if start <= cur_ep <= end:
                return stage
        return None

    def _apply_config_changes(self, config: Mapping[str, Any]):
        stage = self._get_current_stage(config)
        if not stage:
            return
        trainer_cfg = stage.get("trainer", {})
        loss_cfg = stage.get("loss", {})

        if "freerun_weight" in trainer_cfg:
            self.freerun_weight = float(trainer_cfg["freerun_weight"])
        if "freerun_horizon" in trainer_cfg:
            self.freerun_horizon = int(trainer_cfg["freerun_horizon"])
        if "eval_horizon" in trainer_cfg:
            if hasattr(self, 'eval_settings'):
                self.eval_settings.horizon = int(trainer_cfg["eval_horizon"])

        if hasattr(self, 'loss_fn') and self.loss_fn is not None:
            if "w_fk_pos" in loss_cfg:
                self.loss_fn.w_fk_pos = float(loss_cfg["w_fk_pos"])
            if "w_rot_local" in loss_cfg:
                self.loss_fn.w_rot_local = float(loss_cfg["w_rot_local"])
            if "w_yaw" in loss_cfg:
                self.loss_fn.w_yaw = float(loss_cfg["w_yaw"])

        # Handle loss_groups (e.g., "core" group with w_rot_delta_root)
        loss_groups = stage.get("loss_groups", {})
        if hasattr(self, 'loss_fn') and self.loss_fn is not None:
            for group_name, group_weights in loss_groups.items():
                if isinstance(group_weights, dict):
                    for weight_name, weight_value in group_weights.items():
                        if hasattr(self.loss_fn, weight_name):
                            setattr(self.loss_fn, weight_name, float(weight_value))

        # 同步在线调度器的边界/当前值，避免后续被旧参数拉回去
        if hasattr(self, 'hyperparam_scheduler') and self.hyperparam_scheduler is not None:
            sched_params = self.hyperparam_scheduler.params
            if "freerun_horizon" in trainer_cfg:
                sched_params["freerun_horizon"] = int(trainer_cfg["freerun_horizon"])
            if hasattr(self, 'freerun_horizon_min'):
                sched_params["freerun_min"] = int(getattr(self, 'freerun_horizon_min'))
            if "freerun_horizon" in trainer_cfg:
                sched_params["freerun_max"] = int(
                    max(
                        trainer_cfg.get("freerun_horizon", 0),
                        sched_params.get("freerun_max", 0),
                        getattr(self, 'freerun_horizon_min', 0),
                    )
                )
            if hasattr(self, 'teacher_forcing_ratio'):
                sched_params["teacher_forcing_ratio"] = float(getattr(self, 'teacher_forcing_ratio'))

    def _save_adjusted_config(self, epoch: int):
        out_dir = getattr(self, 'out_dir', None)
        cfg = getattr(self, 'full_config', None)
        if not out_dir or cfg is None:
            return
        try:
            out_path = Path(out_dir) / f"config_adjusted_ep{int(epoch):03d}.json"
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[AdaptiveTuning][WARN] failed to save adjusted config: {exc}")

    def _activate_stage(self, idx: int, epoch: int) -> None:
        schedule = getattr(self, 'freerun_stage_schedule', None)
        if not schedule:
            return
        idx = max(0, min(idx, len(schedule) - 1))
        self._stage_active_idx = idx
        self._stage_epoch_entered = epoch
        self._stage_goal_history = {}
        stage = schedule[idx]
        stage.pop('_goal_state', None)

    def _advance_stage(self, epoch: int) -> None:
        schedule = getattr(self, 'freerun_stage_schedule', None)
        if not schedule:
            self._stage_pending_advance = False
            return
        idx = (self._stage_active_idx or 0) + 1
        if idx >= len(schedule):
            self._stage_pending_advance = False
            return
        self._activate_stage(idx, epoch)
        self._stage_pending_advance = False

    def _current_stage(self) -> Optional[Dict[str, Any]]:
        schedule = getattr(self, 'freerun_stage_schedule', None)
        idx = getattr(self, '_stage_active_idx', None)
        if schedule and idx is not None and 0 <= idx < len(schedule):
            return schedule[idx]
        return None

    def _maybe_finish_stage(self, epoch: int, metrics: Dict[str, Any], *, tag: str) -> None:
        stage = self._current_stage()
        if stage is None:
            return
        goal = stage.get('goal')
        if not goal:
            return
        tags = goal.get('tags') or ['valfree']
        if tag not in tags:
            return
        window = int(goal.get('window', 3) or 3)
        if not isinstance(self._stage_goal_history, dict):
            self._stage_goal_history = {}
        history = self._stage_goal_history.get(tag)
        if history is None or history.maxlen != window:
            history = deque(maxlen=window)
            self._stage_goal_history[tag] = history
        history.append(dict(metrics))
        min_epochs = int(goal.get('min_epochs', 0) or 0)
        elapsed = 0
        if self._stage_epoch_entered is not None:
            elapsed = epoch - self._stage_epoch_entered + 1
        if elapsed < min_epochs:
            return
        if len(history) < window:
            return
        metrics_cfg = goal.get('metrics') or {}
        if not metrics_cfg:
            return
        for metric_name, cfg in metrics_cfg.items():
            values = [self._extract_metric_from_record(rec, metric_name) for rec in history]
            values = [v for v in values if v is not None]
            if not values:
                return
            avg_val = sum(values) / len(values)
            if not self._metric_within_goal(avg_val, cfg):
                return
        goal_state = stage.get('_goal_state')
        if not isinstance(goal_state, dict):
            goal_state = {}
            stage['_goal_state'] = goal_state
        if goal_state.get('met'):
            return
        goal_state['met'] = True
        goal_state['epoch'] = epoch
        label = stage.get('label') or f"{stage.get('start', '?')}-{stage.get('end', '?')}"
        print(f"[StageGoal] stage={label} met at epoch {epoch:03d}; scheduling advance")
        self._stage_pending_advance = True

    def _extract_metric_from_record(self, record: Mapping[str, Any], name: str) -> Optional[float]:
        target: Any = record
        for part in str(name).split('/'):
            if isinstance(target, Mapping) and part in target:
                target = target[part]
            else:
                return None
        try:
            return float(target)
        except Exception:
            return None

    def _metric_within_goal(self, value: float, cfg: Mapping[str, Any]) -> bool:
        ref = float(cfg.get('ref', 0.0) or 0.0)
        hi = cfg.get('hi')
        lo = cfg.get('lo')
        if hi is None and cfg.get('hi_ratio') is not None:
            hi = ref * float(cfg['hi_ratio'])
        if lo is None and cfg.get('lo_ratio') is not None:
            lo = ref * float(cfg['lo_ratio'])
        if hi is not None and value > float(hi):
            return False
        if lo is not None and value < float(lo):
            return False
        return True
    def _log_freerun_vs_teacher_stats(self, epoch: int, batch_idx: int, free_stats: Optional[dict]) -> None:
        if not bool(getattr(self, 'freerun_grad_log', False)):
            return
        teacher_stats = getattr(self, '_last_teacher_stats', None)
        if not isinstance(teacher_stats, dict):
            return
        if not isinstance(free_stats, dict):
            return
        keys = ('rot_geo', 'rot_delta', 'angvel_dir')
        parts = []
        for key in keys:
            t = teacher_stats.get(key)
            f = free_stats.get(key)
            if t is None or f is None:
                continue
            denom = abs(t) + 1e-6
            ratio = f / denom
            parts.append(f"{key}=T{t:.3f}/F{f:.3f}(x{ratio:.2f})")
        if parts:
            joined = " ".join(parts)
            print(f"[FreeLossCmp][ep {epoch:03d}][bi {batch_idx:04d}] {joined}")

    def compute_freerun_loss(
        self,
        state_seq,
        gt_seq,
        cond_seq,
        cond_raw_seq,
        contacts_seq,
        angvel_seq,
        pose_hist_seq,
        batch,
        cond_norm_mu=None,
        cond_norm_std=None,
        *,
        batch_idx: int,
        log_grad: bool = False,
    ):
        import torch
        weight = self._current_freerun_weight()
        need_training = (weight > 0.0) or log_grad
        if not need_training:
            return None
        if state_seq is None or gt_seq is None:
            return None
        T = state_seq.shape[1]
        if T < 2:
            return None
        horizon_cap = min(self._current_freerun_horizon(), T - 1)
        if horizon_cap < 1:
            return None
        min_h = int(getattr(self, 'freerun_horizon_min', 6) or 6)
        min_h = max(1, min(min_h, horizon_cap))
        lower = min(min_h, horizon_cap)
        upper = max(lower, horizon_cap)
        if upper > lower:
            effective_h = int(torch.randint(lower, upper + 1, (1,), device=state_seq.device).item())
        else:
            effective_h = lower
        window = effective_h + 1
        if window < 2:
            return None
        max_start = max(0, T - window)
        if max_start > 0:
            start = int(torch.randint(0, max_start + 1, (1,), device=state_seq.device).item())
        else:
            start = 0
        payload = self._freerun_loss_window(
            state_seq,
            gt_seq,
            cond_seq,
            cond_raw_seq,
            contacts_seq,
            angvel_seq,
            pose_hist_seq,
            batch,
            start=start,
            length=window,
            train_mode=True,
            return_preds=log_grad,
            cond_norm_mu=cond_norm_mu,
            cond_norm_std=cond_norm_std,
        )
        if payload is None:
            return None
        free_loss, stats, preds_free, _ = payload
        grad_monitor = None
        if log_grad:
            grad_monitor = self._collect_freerun_gradients(free_loss, preds_free, effective_h)
        self._freerun_active_horizon = effective_h
        return free_loss, stats or {}, grad_monitor, effective_h

    def test_gradient_connection(self, loader):
        if getattr(self, '_grad_connection_checked', False):
            return
        if not bool(getattr(self, 'enable_grad_connection_test', True)):
            self._grad_connection_checked = True
            return
        import torch
        sample_batch = None
        it = iter(loader)
        try:
            sample_batch = next(it)
        except StopIteration:
            print("[GradConn] skipped: empty loader.")
            self._grad_connection_checked = True
            return
        x_cand = self._pick_first(sample_batch, ('motion','X','x_in_features'))
        y_cand = self._pick_first(sample_batch, ('gt_motion','Y','y_out_features','y_out_seq'))
        if x_cand is None or y_cand is None:
            print("[GradConn] skipped: batch missing motion/gt.")
            self._grad_connection_checked = True
            return
        state_seq = x_cand.to(self.device).float()
        gt_seq = y_cand.to(self.device).float()
        window = min(int(getattr(self, 'grad_conn_window', 8) or 8), state_seq.shape[1])
        if window < 2:
            print("[GradConn] skipped: window < 2.")
            self._grad_connection_checked = True
            return
        state_seq = state_seq[:, :window]
        gt_seq = gt_seq[:, :window]

        def _slice_optional(key):
            val = sample_batch.get(key) if isinstance(sample_batch, dict) else None
            if val is None:
                return None
            tensor = val.to(self.device).float()
            if tensor.dim() == 3 and tensor.size(1) >= window:
                return tensor[:, :window]
            return tensor

        cond_seq = _slice_optional('cond_in')
        cond_raw_seq = _slice_optional('cond_tgt_raw')
        contacts_seq = _slice_optional('contacts')
        angvel_seq = _slice_optional('angvel')
        pose_hist_seq = _slice_optional('pose_hist')
        cond_norm_mu = sample_batch.get('cond_norm_mu') if isinstance(sample_batch, dict) else None
        cond_norm_std = sample_batch.get('cond_norm_std') if isinstance(sample_batch, dict) else None
        if cond_norm_mu is not None:
            cond_norm_mu = cond_norm_mu.to(self.device).float()
        if cond_norm_std is not None:
            cond_norm_std = cond_norm_std.to(self.device).float()

        use_anomaly = bool(getattr(self, 'grad_conn_detect_anomaly', True))
        import contextlib
        anomaly_ctx = torch.autograd.set_detect_anomaly if use_anomaly else contextlib.nullcontext
        with anomaly_ctx(True if use_anomaly else False):
            preds, attn = self._rollout_sequence(
                state_seq,
                cond_seq,
                cond_raw_seq,
                contacts_seq=contacts_seq,
                angvel_seq=angvel_seq,
                pose_hist_seq=pose_hist_seq,
                gt_seq=gt_seq,
                cond_norm_mu=cond_norm_mu,
                cond_norm_std=cond_norm_std,
                mode='train_free',
                tf_ratio=0.0,
            )
            with self._amp_context(self.use_amp):
                out = self.loss_fn(preds, gt_seq, attn_weights=attn, batch=sample_batch)
            loss = out[0] if isinstance(out, tuple) else out
            self.optimizer.zero_grad(set_to_none=True)
            try:
                loss.backward()
            except RuntimeError as exc:
                raise RuntimeError("[GradConn] backward failed; 检查 train_free 梯度链路。") from exc
        # loss/backward completed inside context at this point
        grad_hits = sum(
            1 for p in self.model.parameters()
            if p.grad is not None and torch.isfinite(p.grad).any()
        )
        if grad_hits == 0:
            raise RuntimeError("[GradConn] backward produced no gradients; 可能仍有 detach().")
        self.optimizer.zero_grad(set_to_none=True)
        self._grad_connection_checked = True
        print(f"[GradConn] ok: window={window} grad_hits={grad_hits}.")

    def _maybe_apply_adaptive_loss(self, loss, stats):
        module = getattr(self, 'adaptive_loss_module', None)
        if module is None:
            return loss, stats
        payload_fn = getattr(self.loss_fn, 'adaptive_loss_payload', None)
        if not callable(payload_fn):
            return loss, stats
        payload = payload_fn()
        if not payload:
            return loss, stats
        raw_losses = payload.get('losses') or {}
        total_weight = float(payload.get('total_weight', 0.0))
        if total_weight <= 0:
            return loss, stats
        filtered = {
            name: raw_losses[name]
            for name in module.loss_names
            if name in raw_losses and raw_losses[name] is not None
        }
        if not filtered:
            return loss, stats
        core_loss = payload.get('core_loss') or loss
        weighted_loss, rel_weights = module(
            filtered,
            model=self.model,
            epoch=getattr(self, 'cur_epoch', 0),
        )
        adapted = core_loss + weighted_loss * total_weight
        if not isinstance(stats, dict):
            stats = {} if stats is None else dict(stats)
        stats = dict(stats)
        stats['adaptive_loss/total_weight'] = float(total_weight)
        try:
            stats['adaptive_loss/base'] = float(core_loss.detach().cpu())
        except Exception:
            pass
        for name, rel in rel_weights.items():
            stats[f'adaptive_loss/weight/{name}'] = float(rel * total_weight)
        return adapted, stats

    def _step_hyperparam_scheduler(self, loss_tensor, grad_norm_value):
        scheduler = getattr(self, 'hyperparam_scheduler', None)
        if scheduler is None:
            return
        try:
            loss_val = float(loss_tensor.detach().cpu())
        except Exception:
            loss_val = float('nan')
        scheduler.step(loss_val, float(grad_norm_value))
        params = scheduler.get_params()
        if 'freerun_horizon' in params:
            self.freerun_horizon = int(params['freerun_horizon'])
        if 'teacher_forcing_ratio' in params:
            self.teacher_forcing_ratio = float(params['teacher_forcing_ratio'])
    def __init__(self, model, loss_fn, lr=0.0001, grad_clip=0.0, weight_decay=0.01, tf_warmup_steps=0, tf_total_steps=0, augmentor=None, use_amp=None, accum_steps=1, *, pin_memory=False, args=None):
        import torch
        self.model = model
        self.loss_fn = loss_fn
        # Make MuY/StdY available on Trainer for _denorm()
        self.mu_y = getattr(loss_fn, 'mu_y', None)
        self.std_y = getattr(loss_fn, 'std_y', None)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"[LR-DBG:init] arg_lr={lr:.2e} opt_pg0={self.optimizer.param_groups[0]['lr']:.2e}")
        # Autoreg tuning knobs
        self.use_freerun_state_sync = True
        self.history_debug_steps: int = 0
        self.freerun_stage_schedule = []
        self._stage_active_idx: Optional[int] = None
        self._stage_epoch_entered: Optional[int] = None
        self._stage_goal_history: Dict[str, deque] = {}
        self._stage_pending_advance: bool = False
        self.args = args
        self.enable_adaptive = bool(getattr(args, 'adaptive_loss_tuning', False)) if args is not None else bool(_arg('adaptive_loss_tuning', False))
        self.adjuster = None
        self.full_config = None
        self._adaptive_config_path: Optional[Path] = None
        if self.enable_adaptive:
            cfg_path = getattr(args, 'config_path', None) if args is not None else _arg('config_path', None) or _arg('config_json', None)
            if cfg_path:
                cfg_file = Path(cfg_path).expanduser()
                self._adaptive_config_path = cfg_file
                if cfg_file.is_file():
                    try:
                        with open(cfg_file, 'r', encoding='utf-8') as f:
                            self.full_config = json.load(f)
                        self.adjuster = StageMetricAdjuster(self.full_config)
                        print(f"[AdaptiveTuning] loaded config from {cfg_file}")
                    except Exception as exc:
                        print(f"[AdaptiveTuning][WARN] failed to load config {cfg_file}: {exc}")
                else:
                    print(f"[AdaptiveTuning][WARN] config_path not found: {cfg_file}")
            else:
                print("[AdaptiveTuning][WARN] adaptive tuning enabled but config_path is missing; disable or provide path.")

        self.grad_clip = float(grad_clip)
        self.tf_warmup_steps = int(tf_warmup_steps)
        self.tf_total_steps = int(tf_total_steps)
        self.augmentor = augmentor
        self.device = next(model.parameters()).device
        if use_amp is None:
            self.use_amp = getattr(self.device, 'type', '') in ('cuda', 'mps')
        else:
            self.use_amp = bool(use_amp)
        self.accum_steps = int(accum_steps)
        if getattr(self.device, 'type', None) == 'mps' and getattr(self, 'use_amp', False):
            print('[AMP] MPS backend detected; using torch.autocast(mps, fp16).')
        dev_type = getattr(self.device, 'type', 'cpu')
        if dev_type == 'mps':
            self._amp_context = lambda enabled: torch.autocast(device_type='mps', dtype=torch.float16, enabled=enabled)
        elif dev_type == 'cuda':
            self._amp_context = lambda enabled: torch.amp.autocast('cuda', enabled=enabled)
        else:
            import contextlib as _ctx
            self._amp_context = lambda enabled: _ctx.nullcontext()
        self._non_blocking = bool(pin_memory and dev_type != 'cpu')
        self._x_layout = {}
        self._y_layout = {}
        self.fps = 60.0
        self.y_to_x_map = []
        self.MuY = None
        self.StdY = None
        self._norm_cache = {}
        self._norm_template_path: Optional[str] = None
        self._bundle_json_path: Optional[str] = None
        # Pose history (explicit buffer) metadata injected from dataset/spec.
        self.pose_hist_len: int = 0
        self.pose_hist_dim: int = 0
        self.pose_hist_scales: Optional[torch.Tensor] = None
        self.pose_hist_mu: Optional[torch.Tensor] = None
        self.pose_hist_std: Optional[torch.Tensor] = None
        self.nan_grad_reports: int = 0
        self.nan_grad_report_limit: int = 5
        self.diag_input_stats: bool = False
        # yaw 诊断相关：仅在需要时打印有限次数的告警，避免刷屏
        self.yaw_diag_deg_threshold: float = 45.0
        self._yaw_diag_limit: int = 5
        self._yaw_diag_hits: int = 0
        self._diag_roll_mode: Optional[str] = None
        self._diag_roll_step: int = -1
        self._diag_roll_epoch: int = 0
        self.freerun_weight_mode: str = 'epoch_linear'
        self.freerun_weight_ramp_epochs: int = 1
        self.freerun_horizon_min: int = 6
        self.freerun_init_horizon: int = 8
        self.freerun_horizon_ramp_epochs: int = 4
        self.freerun_weight_init: float = 0.0
        self.freerun_weight_init: float = 0.0
        self.freerun_grad_log: bool = False
        self.freerun_grad_log_interval: int = 50
        self.freerun_grad_ratio_alert: float = 1e-2
        self._freerun_active_horizon: int = 0
        self._last_teacher_stats: Optional[dict[str, float]] = None
        self.enable_grad_connection_test: bool = True
        self._grad_connection_checked: bool = False
        self.grad_conn_window: int = 8
        self.grad_conn_detect_anomaly: bool = True
        self._carry_debug_buffer: list[dict[str, float]] = []
        self.adaptive_loss_module = None
        self.hyperparam_scheduler: Optional[AdaptiveHyperparamScheduler] = None
        self.teacher_forcing_ratio: float = 1.0
        # ---- Metrics buffering for in-process consumers ----
        self.metric_history: list[dict[str, Any]] = []
        self.metric_history_maxlen: int = 256
        self.latest_metrics: dict[str, dict[str, Any]] = {}
        self._metric_callbacks: list[Callable[[dict[str, Any]], None]] = []

    def _diag_norm_x(self, x_raw, mu_x=None, std_x=None):
        # 仅使用 DataNormalizer；缺失即视为致命错误
        self._require_normalizer("Trainer._diag_norm_x")
        try:
            return self.normalizer.norm(x_raw)
        except Exception as exc:
            self._raise_norm_error("normalizer.norm 在 _diag_norm_x 中失败", exc)

    def _pick_first(self, batch, keys):
        if batch is None:
            return None
        if isinstance(batch, dict):
            for k in keys:
                if k in batch and batch[k] is not None:
                    return batch[k]
        return None

    def _format_template_hint(self, prefix: str) -> str:
        hints: list[str] = []
        norm_tpl = getattr(self, '_norm_template_path', None)
        bundle = getattr(self, '_bundle_json_path', None)
        if isinstance(norm_tpl, str) and norm_tpl:
            hints.append(f"norm_template={norm_tpl}")
        if isinstance(bundle, str) and bundle:
            hints.append(f"bundle_json={bundle}")
        if hints:
            return f"{prefix} ({', '.join(hints)})"
        return prefix

    def _require_normalizer(self, context: str) -> None:
        if not hasattr(self, 'normalizer') or self.normalizer is None:
            raise RuntimeError(self._format_template_hint(f"[FATAL] {context} 需要已注入的 DataNormalizer。"))

    def _raise_norm_error(self, context: str, exc: Optional[Exception] = None) -> None:
        msg = self._format_template_hint(f"[FATAL] {context}")
        raise RuntimeError(msg) from exc

    @torch.no_grad()
    def eval_epoch(self, loader, mode='teacher'):
        self.model.eval()
        return evaluate_teacher(self, loader, mode=mode)

    def fit(self, train_loader, epochs=10, log_every=50, out_dir=None, patience=10, run_name='run'):
        import torch, os
        self.model.train()
        self.train_loader = train_loader
        device_type = getattr(self.device, 'type', 'cpu')
        scaler = torch.amp.GradScaler('cuda' if device_type=='cuda' else 'cpu', enabled=(getattr(self, 'use_amp', False) and device_type in ('cuda', 'mps')))
        accum_steps = int(getattr(self, 'accum_steps', 1) or 1)
        best_val, best_ckpt = float('inf'), None
        best_valfree = float('inf')
        history = {'train':[], 'val':[]}
        tf_mode = getattr(self, 'tf_mode', 'epoch_linear')
        tf_start = int(getattr(self, 'tf_start_epoch', 0))
        tf_end   = int(getattr(self, 'tf_end_epoch', 0))
        tf_max_base   = float(getattr(self, 'tf_max', 1.0))
        tf_min_base   = float(getattr(self, 'tf_min', 0.0))

        try:
            self.test_gradient_connection(train_loader)
        except Exception as _grad_exc:
            print(f"[GradConn] failed during warm-up: {_grad_exc}")
            raise

        for ep in range(1, int(epochs)+1):

            # record epoch for schedulers
            try:
                self.cur_epoch = int(ep)
                self.current_epoch = int(ep)
                self.total_epochs = int(epochs)
            except Exception:
                pass
            # 复位 yaw 诊断状态，保证每个 epoch 打印次数受限
            self._diag_roll_epoch = int(ep)
            self._yaw_diag_hits = 0
            self._diag_roll_step = -1
            self._diag_roll_mode = None
            epoch_sums = {}
            epoch_cnt = 0
            if ep == 1:
                print(f"[LR-DBG:fit-epoch{ep:03d}-start] pg0={self.optimizer.param_groups[0]['lr']:.2e}")

            stage_overrides = self._apply_stage_schedule(ep)
            tf_max_epoch = float(stage_overrides.get('tf_max', tf_max_base))
            tf_min_epoch = float(stage_overrides.get('tf_min', tf_min_base))

            if tf_mode == 'epoch_linear' and tf_end > tf_start:
                if ep <= tf_start:
                    tf_ratio = tf_max_epoch
                elif ep >= tf_end:
                    tf_ratio = tf_min_epoch
                else:
                    r = (ep - tf_start) / max(1, (tf_end - tf_start))
                    tf_ratio = tf_max_epoch + (tf_min_epoch - tf_max_epoch) * r
            else:
                tf_ratio = tf_max_epoch
            self.teacher_forcing_ratio = float(tf_ratio)
            self._last_tf_ratio = float(tf_ratio)
            sched = getattr(self, 'hyperparam_scheduler', None)
            if sched is not None:
                sched.params['teacher_forcing_ratio'] = float(tf_ratio)
                if 'freerun_horizon' in sched.params:
                    sched.params['freerun_horizon'] = int(getattr(self, 'freerun_horizon', sched.params['freerun_horizon']))
                if 'freerun_min' in sched.params and hasattr(self, 'freerun_horizon_min'):
                    sched.params['freerun_min'] = int(getattr(self, 'freerun_horizon_min'))
                if 'freerun_max' in sched.params:
                    sched.params['freerun_max'] = int(max(
                        sched.params.get('freerun_max', 0),
                        getattr(self, 'freerun_horizon', 0),
                        getattr(self, 'freerun_horizon_min', 0),
                    ))
            running, cnt = 0.0, 0
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)


            for bi, batch in enumerate(train_loader, start=1):
                x_cand = self._pick_first(batch, ('motion','X','x_in_features'))
                y_cand = self._pick_first(batch, ('gt_motion','Y','y_out_features','y_out_seq'))
                if x_cand is None or y_cand is None:
                    continue
                # 位置：Trainer.train(...) 里
                def _to_device(maybe_tensor):
                    if maybe_tensor is None:
                        return None
                    try:
                        tensor = maybe_tensor.to(self.device, non_blocking=self._non_blocking)
                        return tensor if tensor.dtype == torch.float32 else tensor.float()
                    except Exception:
                        return None

                state_seq = _to_device(x_cand)
                gt_seq = _to_device(y_cand)
                if state_seq is None or gt_seq is None:
                    continue
                state_seq = self._maybe_apply_teacher_noise(state_seq)

                cond_seq = _to_device(batch.get('cond_in')) if isinstance(batch, dict) else None
                cond_raw_seq = _to_device(batch.get('cond_tgt_raw')) if isinstance(batch, dict) else None
                contacts_seq = _to_device(batch.get('contacts')) if isinstance(batch, dict) else None
                angvel_seq = _to_device(batch.get('angvel')) if isinstance(batch, dict) else None
                angvel_raw_seq = _to_device(batch.get('angvel_raw')) if isinstance(batch, dict) else None
                pose_hist_seq = _to_device(batch.get('pose_hist')) if isinstance(batch, dict) else None
                cond_norm_mu = _to_device(batch.get('cond_norm_mu')) if isinstance(batch, dict) else None
                cond_norm_std = _to_device(batch.get('cond_norm_std')) if isinstance(batch, dict) else None

                # === 插入开始：一次性打印训练端 X(z) 的 RMS，验证不是 0 ===
                current_tf_ratio = float(getattr(self, 'teacher_forcing_ratio', tf_ratio))
                tf_ratio = current_tf_ratio
                preds_dict, last_attn = self._rollout_sequence(
                    state_seq,
                    cond_seq,
                    cond_raw_seq,
                    contacts_seq=contacts_seq,
                    angvel_seq=angvel_seq,
                    pose_hist_seq=pose_hist_seq,
                    gt_seq=gt_seq,
                    cond_norm_mu=cond_norm_mu,
                    cond_norm_std=cond_norm_std,
                    mode='mixed',
                    tf_ratio=current_tf_ratio,
                )

                stats = {}
                with self._amp_context(self.use_amp):
                    out = self.loss_fn(preds_dict, gt_seq, attn_weights=last_attn, batch=batch)
                if isinstance(out, tuple):
                    loss, stats = out
                else:
                    loss, stats = out, {}
                if not isinstance(stats, dict):
                    stats = {} if stats is None else dict(stats)

                loss, stats = self._maybe_apply_adaptive_loss(loss, stats)

                log_grad = self._should_log_freerun_gradients(bi)
                freerun_payload = self.compute_freerun_loss(
                    state_seq,
                    gt_seq,
                    cond_seq,
                    cond_raw_seq,
                    contacts_seq,
                    angvel_seq,
                    pose_hist_seq,
                    batch,
                    cond_norm_mu=cond_norm_mu,
                    cond_norm_std=cond_norm_std,
                    batch_idx=bi,
                    log_grad=log_grad,
                )
                if freerun_payload is not None:
                    free_loss, free_stats, grad_monitor, eff_h = freerun_payload
                    weight = self._current_freerun_weight()
                    if weight > 0.0:
                        loss = loss + weight * free_loss
                    stats['freerun_loss'] = float(free_loss.detach().cpu())
                    stats['freerun/weight'] = float(weight)
                    stats['freerun/horizon'] = float(eff_h)
                    if isinstance(free_stats, dict):
                        for fk, fv in free_stats.items():
                            try:
                                stats[f'freerun/{fk}'] = fv
                            except Exception:
                                pass
                    if grad_monitor is not None:
                        stats['freerun/grad_ratio'] = float(grad_monitor.get('ratio', float('nan')))
                        self._maybe_print_grad_monitor(grad_monitor, ep, bi)
                    self._log_freerun_vs_teacher_stats(ep, bi, free_stats)

                if getattr(self, 'history_debug_steps', 0) > 1 and bi == 1:
                    try:
                        self._history_drift_debug(
                            state_seq,
                            gt_seq,
                            cond_seq,
                            cond_raw_seq,
                            contacts_seq,
                            angvel_seq,
                            pose_hist_seq,
                            epoch=ep,
                            batch_idx=bi,
                            cond_norm_mu=cond_norm_mu,
                            cond_norm_std=cond_norm_std,
                        )
                    except Exception as exc:
                        print(f"[HistDrift][warn] debug failed: {exc}")

                try:
                    if isinstance(stats, dict):
                        for _k, _v in stats.items():
                            try:
                                if hasattr(_v, 'detach'):
                                    val = float(_v.detach().cpu())
                                else:
                                    val = float(_v)
                                epoch_sums[_k] = epoch_sums.get(_k, 0.0) + val
                            except Exception:
                                pass
                    epoch_cnt += 1
                except Exception:
                    pass
                self._last_teacher_stats = stats if isinstance(stats, dict) else None

                scaler.scale(loss / accum_steps).backward()

                if (bi + 1) % accum_steps == 0:
                    scaler.unscale_(self.optimizer)

                    _any_bad_grad = False
                    _bad_names = []
                    for _name, _p in self.model.named_parameters():
                        if _p.grad is None:
                            continue
                        if not torch.isfinite(_p.grad).all():
                            _any_bad_grad = True
                            if len(_bad_names) < 3:
                                _bad_names.append(_name)
                            _p.grad = torch.nan_to_num(_p.grad, nan=0.0, posinf=0.0, neginf=0.0)

                    if _any_bad_grad:
                        try:
                            loss_val = float(loss.detach().cpu())
                        except Exception:
                            loss_val = float('nan')
                        self._dump_nan_grad_report(ep, bi, batch, state_seq, gt_seq, preds_dict, loss_val, stats)
                        if log_every:
                            print(f"[Guard][Grad] non-finite grads on {', '.join(_bad_names)} ... skip optimizer.step()")
                        scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    gn = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=float(getattr(self, 'grad_clip', 1.0))
                    )
                    self._step_hyperparam_scheduler(loss, float(gn))
                    tf_ratio = float(getattr(self, 'teacher_forcing_ratio', tf_ratio))
                    self._last_tf_ratio = float(tf_ratio)
                    if log_every and (bi % int(log_every or 50) == 0):
                        lr0 = float(self.optimizer.param_groups[0].get('lr', 0.0))
                        print(f"[Grad] ep={ep:03d} bi={bi:04d} gn={float(gn):.3e} lr={lr0:.2e}")

                    scaler.step(self.optimizer)
                    if log_every and ep == 1 and bi == 1:
                        print(f"[LR-DBG:after-opt-step] pg0={self.optimizer.param_groups[0]['lr']:.2e}")
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    _param_finite = True
                    with torch.no_grad():
                        for _n, _p in self.model.named_parameters():
                            if not torch.isfinite(_p).all():
                                _param_finite = False
                                break
                    if not _param_finite:
                        if log_every:
                            print("[Guard][Param] non-finite parameters after step; try sanitize via validate_and_fix_model_")
                        try:
                            validate_and_fix_model_(self.model, reinit_on_nonfinite=True)
                        except Exception as _san_e:
                            print("[Guard][Param] sanitize failed:", _san_e)
                            raise

                    _sched = getattr(self, 'lr_scheduler', None)
                    if _sched is not None:
                        try:
                            _sched.step()
                        except Exception:
                            pass

                running += float(loss.detach().cpu()); cnt += 1
                if log_every and (bi % int(log_every) == 0):
                    print("[Train][ep %03d][%04d/%d] loss=%.4f tf=%.3f" % (ep, bi, len(train_loader), running/max(1,cnt), float(tf_ratio)))
            avg_train = running / max(1, cnt)
            history['train'].append(avg_train)
            print("[Train][ep %03d] loss=%.4f" % (ep, avg_train))

            # --- 阶段化评估与日志输出 ---
            is_teacher_phase = float(getattr(self, '_last_tf_ratio', 1.0)) >= 0.999
            _metrics = None
            metrics_for_json = None
            metrics_tag = None
            teacher_metrics_cached = None
            try:
                import math as _math

                def _run_valfree_eval(log_prefix: str = "ValFree"):
                    if getattr(self, 'val_mode', 'none') != 'online' or bool(getattr(self, 'no_monitor', False)):
                        return None
                    vloader = self.train_loader
                    _mon_batches = int(getattr(self, 'monitor_batches', 8) or 8)
                    free_metrics = dict(self.validate_autoreg_online(vloader, max_batches=_mon_batches))
                    free_metrics.setdefault('phase', 'freerun')
                    free_metrics['tf_ratio'] = float(getattr(self, '_last_tf_ratio', 1.0))
                    _extra = ""
                    _kgeo = free_metrics.get('KeyBone/GeoDegMean', float('nan'))
                    _klocal = free_metrics.get('KeyBone/GeoLocalDegMean', float('nan'))
                    yaw_cmd = free_metrics.get('CondYawVsPredDeg', float('nan'))
                    free_ang_dir = free_metrics.get('AngVelDirDeg', float('nan'))
                    if _math.isfinite(_kgeo):
                        _extra += f" | LimbGeoDeg={_kgeo:.3f}°"
                    if _math.isfinite(_klocal):
                        _extra += f" | LimbGeoLocalDeg={_klocal:.3f}°"
                    if _math.isfinite(yaw_cmd):
                        _extra += f" | YawCmdDiff={yaw_cmd:.2f}°"
                    if _math.isfinite(free_ang_dir):
                        _extra += f" | AngVelDirDeg={free_ang_dir:.2f}"
                    print(
                        f"[{log_prefix}@ep {ep:03d}] "
                        f"MSEnormY={free_metrics.get('MSEnormY', float('nan')):.6f} | "
                        f"GeoDeg={free_metrics.get('GeoDeg', float('nan')):.3f}° | "
                        f"YawAbsDeg={free_metrics.get('YawAbsDeg', float('nan')):.3f} | "
                        f"RootVelMAE={free_metrics.get('RootVelMAE', float('nan')):.5f} | "
                        f"AngVelMAE={free_metrics.get('AngVelMAE', float('nan')):.5f} rad/s | "
                        f"AngMagRel={free_metrics.get('AngVelMagRel', float('nan')):.3f}" + _extra
                    )
                    return free_metrics

                if is_teacher_phase:
                    teacher_metrics = dict(self.eval_epoch(self.train_loader, mode='teacher') or {})
                    teacher_metrics.setdefault('phase', 'teacher')
                    teacher_metrics['tf_ratio'] = float(getattr(self, '_last_tf_ratio', 1.0))
                    metrics_for_json = teacher_metrics
                    metrics_tag = 'teacher'
                    loss_val = teacher_metrics.get('loss', float('nan'))
                    geo_deg = teacher_metrics.get('GeoDeg', float('nan'))
                    ang_mae = teacher_metrics.get('AngVelMAE', float('nan'))
                    ang_rel = teacher_metrics.get('AngVelMagRel', float('nan'))
                    mse_y = teacher_metrics.get('MSEnormY', float('nan'))
                    print(
                        f"[ValTeacher@ep {ep:03d}] "
                        f"loss={loss_val:.6f} | "
                        f"MSEnormY={mse_y:.6f} | "
                        f"GeoDeg={geo_deg:.3f}° | "
                        f"AngVelMAE={ang_mae:.5f} rad/s | "
                        f"AngMagRel={ang_rel:.3f}"
                    )
                else:
                    free_metrics = _run_valfree_eval()
                    if free_metrics is not None:
                        metrics_for_json = free_metrics
                        metrics_tag = 'valfree'
                        _metrics = free_metrics
                    teacher_metrics_cached = dict(self.eval_epoch(self.train_loader, mode='teacher') or {})
                    teacher_metrics_cached.setdefault('phase', 'teacher')
                    teacher_metrics_cached['tf_ratio'] = float(getattr(self, '_last_tf_ratio', 1.0))
                    if metrics_for_json is not None:
                        _gap_extra = ""
                        _kgeo = metrics_for_json.get('KeyBone/GeoDegMean', float('nan'))
                        _klocal = metrics_for_json.get('KeyBone/GeoLocalDegMean', float('nan'))
                        yaw_cmd = metrics_for_json.get('CondYawVsPredDeg', float('nan'))
                        free_ang_dir = metrics_for_json.get('AngVelDirDeg', float('nan'))
                        if _math.isfinite(_kgeo):
                            _gap_extra += f" | LimbGeoDeg={_kgeo:.3f}°"
                        if _math.isfinite(_klocal):
                            _gap_extra += f" | LimbGeoLocalDeg={_klocal:.3f}°"
                        if _math.isfinite(yaw_cmd):
                            _gap_extra += f" | YawCmdDiff={yaw_cmd:.2f}°"
                        if _math.isfinite(free_ang_dir):
                            _gap_extra += f" | AngVelDirDeg={free_ang_dir:.2f}"
                        print(
                            f"[Gap@ep {ep:03d}] "
                            f"teach_loss={teacher_metrics_cached.get('loss', float('nan')):.6f} | "
                            f"GeoDeg={metrics_for_json.get('GeoDeg', float('nan')):.3f}° | "
                            f"AngVelMAE={metrics_for_json.get('AngVelMAE', float('nan')):.5f} | "
                            f"MSEnormY={metrics_for_json.get('MSEnormY', float('nan')):.6f}" + _gap_extra
                        )

                forced_valfree_metrics = None
                if getattr(self, 'force_valfree_eval', False):
                    need_force = metrics_tag != 'valfree'
                    if need_force:
                        forced_valfree_metrics = _run_valfree_eval("ValFreeForced")
                        if forced_valfree_metrics is not None:
                            _metrics = forced_valfree_metrics
            except Exception as _e:
                phase_label = 'ValTeacher' if is_teacher_phase else 'ValFree'
                import traceback
                traceback.print_exc()
                print(f"[{phase_label}@ep {ep:03d}] skipped due to error: {_e}")

            if metrics_for_json is not None and metrics_tag is not None:
                self._record_epoch_metrics(metrics_for_json, tag=metrics_tag, epoch=ep)
                if metrics_tag == 'valfree':
                    self._save_val_metrics(ep, metrics_for_json)
                else:
                    self._dump_metrics_json(metrics_for_json, tag=metrics_tag, epoch=ep)
                self._maybe_finish_stage(ep, metrics_for_json, tag=str(metrics_tag))
            if forced_valfree_metrics is not None:
                self._record_epoch_metrics(forced_valfree_metrics, tag='valfree', epoch=ep)
                self._save_val_metrics(ep, forced_valfree_metrics)
                self._dump_metrics_json(forced_valfree_metrics, tag='valfree', epoch=ep)
                self._maybe_finish_stage(ep, forced_valfree_metrics, tag='valfree')
            if (not is_teacher_phase) and teacher_metrics_cached is not None:
                self._record_epoch_metrics(teacher_metrics_cached, tag='teacher', epoch=ep)
                self._maybe_finish_stage(ep, teacher_metrics_cached, tag='teacher')
                self._dump_metrics_json(teacher_metrics_cached, tag='teacher', epoch=ep)

            # --- 依据在线评估的 MSEnormY 记录最佳模型 ---
            if _metrics is not None:
                current = float(_metrics.get('MSEnormY', float('inf')))
                if current < best_valfree - 1e-9:
                    best_valfree = current
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)
                        best_ckpt = os.path.join(out_dir, 'ckpt_best_' + str(run_name) + '.pth')
                        torch.save({'model': self.model.state_dict()}, best_ckpt)

            # --- Adaptive metric-driven tuning (post-epoch) ---
            if getattr(self, 'adjuster', None) is not None and ep >= 3:
                try:
                    metrics_dir = Path(getattr(self, 'out_dir', out_dir) or out_dir) / 'metrics'
                    val_history = load_val_metrics(metrics_dir)
                    changes = self.adjuster.apply(val_history)
                    if changes:
                        print(f"[AdaptiveTuning] Epoch {ep} adjustments:")
                        for k, v in changes.items():
                            print(f"  {k}: {v}")
                        self._apply_config_changes(self.full_config)
                        self._save_adjusted_config(ep)
                except Exception as _adapt_e:
                    print(f"[WARN] Adaptive tuning failed @ep{ep}: {_adapt_e}")
        if out_dir:

            import os, torch

            os.makedirs(out_dir, exist_ok=True)

            last_ckpt = os.path.join(out_dir, 'ckpt_last_' + str(run_name) + '.pth')

            torch.save({'model': self.model.state_dict()}, last_ckpt)

        return best_ckpt, history


    def _sl_from_layout(self, layout, key):
        if not isinstance(layout, dict) or key not in layout:
            return None
        st, ln = int(layout[key][0]), int(layout[key][1])
        return slice(st, st+ln) if ln > 0 else None

    # ===== autoregressive online validation (UE-shaped) =====
    @torch.no_grad()
    def validate_autoreg_online(self, loader, max_batches=8):
        settings = getattr(self, 'eval_settings', FreeRunSettings())
        if max_batches is not None:
            max_batches = int(max_batches)
        effective = FreeRunSettings(
            warmup_steps=settings.warmup_steps,
            horizon=settings.horizon,
            max_batches=max_batches if max_batches is not None else settings.max_batches,
        )
        return evaluate_freerun(self, loader, effective)

    def _denorm(self, y):
        # 仅使用 DataNormalizer；缺失或异常直接终止
        self._require_normalizer("Trainer._denorm")
        try:
            return self.normalizer.denorm(y)
        except Exception as exc:
            self._raise_norm_error("normalizer.denorm 在 _denorm 中失败", exc)

    def _cached_norm_param(self, key: str, value, ref_tensor):
        import torch
        if value is None:
            return None
        cache = self._norm_cache.setdefault(key, {})
        device = ref_tensor.device
        dtype = ref_tensor.dtype
        cache_key = (device, dtype)
        tensor = cache.get(cache_key)
        if tensor is None:
            if torch.is_tensor(value):
                tensor = value.to(device=device, dtype=dtype)
            else:
                tensor = torch.as_tensor(value, device=device, dtype=dtype)
            cache[cache_key] = tensor
        return tensor

    def _norm_y(self, y_raw):
        self._require_normalizer("Trainer._norm_y")
        try:
            return self.normalizer.norm_y(y_raw)
        except AttributeError as exc:
            self._raise_norm_error("DataNormalizer 缺少 norm_y 方法", exc)
        except Exception as exc:
            self._raise_norm_error("normalizer.norm_y 失败", exc)

    def _compose_delta_to_raw(self, y_prev_raw, delta_norm):
        import torch
        if y_prev_raw is None:
            self._raise_norm_error("compose_delta_to_raw 需要上一帧 RAW，但收到 None。")
        if delta_norm is None:
            self._raise_norm_error("compose_delta_to_raw 收到 None delta。")
        if delta_norm.shape[-1] != y_prev_raw.shape[-1]:
            self._raise_norm_error("compose_delta_to_raw 维度不匹配。")
        std = getattr(self, 'StdY', None) or getattr(self, 'std_y', None)
        delta_raw = delta_norm
        if std is not None:
            try:
                std_t = self._cached_norm_param('std_y', std, delta_norm)
                if std_t is not None:
                    while std_t.dim() < delta_norm.dim():
                        std_t = std_t.unsqueeze(0)
                    delta_raw = delta_norm * std_t.clamp_min(1e-6)
            except Exception:
                pass
        try:
            return compose_rot6d_delta(y_prev_raw, delta_raw)
        except Exception as e:
            self._raise_norm_error("compose_rot6d_delta 失败", e)

    def _reproject_cond_to_local_frame(self, cond_raw, yaw_gt, yaw_pred):
        """
        将条件信息（目标方向/速度）重投影到模型预测的局部坐标系。

        参数:
            cond_raw: [B, cond_dim] 原始条件，格式: [..., dir_x, dir_y, speed]
            yaw_gt: [B] 或 [B, 1] GT的根骨朝向（世界坐标系）
            yaw_pred: [B] 或 [B, 1] 模型预测的根骨朝向（世界坐标系）

        返回:
            重投影后的 cond_raw，方向分量旋转到模型的局部坐标系
        """
        if cond_raw is None:
            return None

        import torch
        device = cond_raw.device
        dtype = cond_raw.dtype

        # 确保 yaw 是 [B] 形状
        if yaw_gt.dim() > 1:
            yaw_gt = yaw_gt.squeeze(-1)
        if yaw_pred.dim() > 1:
            yaw_pred = yaw_pred.squeeze(-1)

        # 计算朝向偏差：Δyaw = yaw_pred - yaw_gt
        delta_yaw = yaw_pred - yaw_gt
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))  # 归一化到 [-π, π]

        # 解析 cond_raw: [...action_dims, dir_x, dir_y, speed]
        cond_dim = cond_raw.shape[-1]
        if cond_dim < 3:
            return cond_raw  # 无法重投影，返回原值

        action_dim = cond_dim - 3
        cond_reprojected = cond_raw.clone()

        # 提取方向分量
        dir_world = cond_raw[..., action_dim:action_dim + 2]  # [B, 2]

        # 将方向旋转 -Δyaw，转换到模型的局部坐标系
        # 旋转矩阵: [[cos(-θ), -sin(-θ)], [sin(-θ), cos(-θ)]]
        cos_delta = torch.cos(-delta_yaw)
        sin_delta = torch.sin(-delta_yaw)

        dir_local_x = dir_world[..., 0] * cos_delta - dir_world[..., 1] * sin_delta
        dir_local_y = dir_world[..., 0] * sin_delta + dir_world[..., 1] * cos_delta

        # 写回重投影后的方向
        cond_reprojected[..., action_dim] = dir_local_x
        cond_reprojected[..., action_dim + 1] = dir_local_y

        # 速度保持不变（标量，与朝向无关）

        return cond_reprojected

    def _apply_free_carry(self, x_prev, y_denorm, cond_next_raw=None):
        """
        将模型预测的 Y(raw) 写回下一帧的 X(raw)，并根据 cond 信息更新根部位置/速度。
        """
        x_next = x_prev.clone()
        import torch, math
        device = x_prev.device
        dtype = x_prev.dtype

        # --- 1) 写回骨骼旋转 ---
        rx = getattr(self, 'rot6d_x_slice', None) or getattr(self, 'rot6d_slice', None)
        ry = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
        if not (isinstance(rx, slice) and isinstance(ry, slice)):
            self._raise_norm_error("_apply_free_carry 缺少 rot6d 切片")
        if (rx.stop - rx.start) != (ry.stop - ry.start):
            self._raise_norm_error("_apply_free_carry rot6d 区间长度不一致")
        x_next[..., rx] = y_denorm[..., ry]

        # 预解析 cond 原始信息：动作维度 + dir(2) + speed(1) —— 必须存在
        yaw_sl = getattr(self, 'yaw_x_slice', None)
        if cond_next_raw is None:
            self._raise_norm_error("_apply_free_carry 缺少 cond_next_raw（应包含方向与速度信息）")
        cond_raw = torch.as_tensor(cond_next_raw, device=device, dtype=dtype)
        if cond_raw.dim() == 1:
            cond_raw = cond_raw.unsqueeze(0)
        if cond_raw.shape[0] != x_prev.shape[0]:
            cond_raw = cond_raw.expand(x_prev.shape[0], -1)
        cond_dim = cond_raw.shape[-1]
        if cond_dim < 3:
            self._raise_norm_error("_apply_free_carry cond_next_raw 最少需要 [dir_x, dir_y, speed]")
        action_dim = max(0, cond_dim - 3)
        cond_dir = cond_raw[..., action_dim:action_dim + 2]
        cond_speed = cond_raw[..., action_dim + 2]

        if cond_dir.shape[-1] < 2:
            self._raise_norm_error("_apply_free_carry cond_next_raw 缺少二维方向")

        cond_dir_world = cond_dir
        offset = float(getattr(self, 'yaw_forward_axis_offset', 0.0) or 0.0)
        dir_norm = cond_dir_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dir_unit_world = cond_dir_world / dir_norm
        yaw_cmd_world = torch.atan2(dir_unit_world[..., 1], dir_unit_world[..., 0])
        yaw_cmd_vals = torch.atan2(torch.sin(yaw_cmd_world - offset), torch.cos(yaw_cmd_world - offset))

        # --- 2) 根部朝向（yaw） ---
        if not isinstance(yaw_sl, slice):
            self._raise_norm_error("_apply_free_carry 缺少 RootYaw 切片")
        yaw_pred_rot6d = self._infer_root_yaw_from_rot6d(y_denorm)
        if yaw_pred_rot6d is None:
            self._raise_norm_error("_apply_free_carry 无法从 rot6d 推断 RootYaw")
        yaw_pred_rot6d = torch.atan2(torch.sin(yaw_pred_rot6d), torch.cos(yaw_pred_rot6d))
        if yaw_pred_rot6d.dim() == 1:
            yaw_pred_rot6d = yaw_pred_rot6d.unsqueeze(-1)

        yaw_write = yaw_cmd_vals
        if yaw_write.dim() == 1:
            yaw_write = yaw_write.unsqueeze(-1)
        yaw_write = yaw_write.to(device=device, dtype=dtype)
        x_next[..., yaw_sl] = yaw_write
        yaw_vals = yaw_pred_rot6d  # 保留模型自身的 rot6d 朝向用于诊断

        # --- 3) 衍生角速度 ---
        av_sl = getattr(self, 'angvel_x_slice', None)
        if isinstance(av_sl, slice):
            J = (rx.stop - rx.start) // 6
            if J <= 0:
                self._raise_norm_error("_apply_free_carry rot6d 切片无有效关节")
            prev6 = x_prev[..., rx].reshape(x_prev.shape[0], J, 6)
            curr6 = x_next[..., rx].reshape(x_prev.shape[0], J, 6)
            Rp = rot6d_to_matrix(prev6)
            Rc = rot6d_to_matrix(curr6)
            Rseq = torch.stack([Rp, Rc], dim=1)
            fps = float(getattr(self, 'bone_hz', 60.0) or 60.0)
            w = angvel_vec_from_R_seq(Rseq, fps=fps)[:, -1]
            x_next[..., av_sl] = w.reshape(x_prev.shape[0], J * 3)

        # --- 4) 根部速度/位置 ---
        rootvel_sl = getattr(self, 'rootvel_x_slice', None)
        rootpos_sl = getattr(self, 'rootpos_x_slice', None)
        if not isinstance(rootvel_sl, slice):
            self._raise_norm_error("_apply_free_carry 缺少 RootVelocity 切片")
        if cond_speed is None:
            self._raise_norm_error("_apply_free_carry cond_next_raw 缺少速度分量")
        vel_world = dir_unit_world * cond_speed.unsqueeze(-1)
        vel_world = vel_world[..., : (rootvel_sl.stop - rootvel_sl.start)]
        x_next[..., rootvel_sl] = vel_world

        if not isinstance(rootpos_sl, slice):
            self._raise_norm_error("_apply_free_carry 缺少 RootPosition 切片")
        dt = 1.0 / max(float(getattr(self, 'bone_hz', 60.0) or 60.0), 1e-6)
        pos = x_prev[..., rootpos_sl].clone()
        step = vel_world[..., :min(2, vel_world.shape[-1])] * dt
        pos[..., :step.shape[-1]] = pos[..., :step.shape[-1]] + step
        x_next[..., rootpos_sl] = pos

        # ===== yaw 诊断：当预测朝向与指令差距过大时打印提示 =====
        diag_limit = int(getattr(self, '_yaw_diag_limit', 0) or 0)
        if diag_limit > 0 and getattr(self, '_yaw_diag_hits', 0) < diag_limit:
            try:
                yaw_pred = yaw_vals.squeeze(-1)
                if torch.is_tensor(yaw_pred) and yaw_pred.numel() > 0:
                    yaw_cmd = yaw_cmd_vals
                    yaw_diff = torch.atan2(torch.sin(yaw_pred - yaw_cmd), torch.cos(yaw_pred - yaw_cmd))
                    diff_deg = yaw_diff.abs() * (180.0 / math.pi)
                    max_deg = float(diff_deg.max().detach().cpu())
                    if max_deg >= float(getattr(self, 'yaw_diag_deg_threshold', 45.0) or 45.0):
                        idx0 = 0
                        yaw_pred_0 = float(yaw_pred.reshape(-1)[idx0].detach().cpu())
                        yaw_cmd_0 = float(yaw_cmd.reshape(-1)[idx0].detach().cpu())
                        speed_flat = cond_speed.reshape(-1) if torch.is_tensor(cond_speed) else None
                        speed_0 = float(speed_flat[idx0].detach().cpu()) if speed_flat is not None and speed_flat.numel() > 0 else float('nan')
                        mode = getattr(self, '_diag_roll_mode', '?')
                        step_idx = getattr(self, '_diag_roll_step', -1)
                        epoch = getattr(self, '_diag_roll_epoch', getattr(self, 'cur_epoch', -1))
                        print(
                            "[YawDiag]"
                            f"[ep {int(epoch):03d}]"
                            f"[{mode}]"
                            f" step={int(step_idx):03d}"
                            f" diff_max={max_deg:.1f}°"
                            f" pred0={yaw_pred_0:.3f}rad"
                            f" cmd0={yaw_cmd_0:.3f}rad"
                            f" speed0={speed_0:.3f}"
                        )
                        self._yaw_diag_hits = int(getattr(self, '_yaw_diag_hits', 0)) + 1
            except Exception:
                pass

        debug_steps = int(getattr(self, 'freerun_debug_steps', 0) or 0)
        if debug_steps > 0:
            buffer = getattr(self, '_carry_debug_buffer', None)
            if buffer is not None and len(buffer) < debug_steps:
                try:
                    sample = 0
                    rad2deg = 180.0 / math.pi
                    yaw_cmd_sample = float(yaw_cmd_vals.reshape(-1)[sample].detach().cpu() * rad2deg)
                    yaw_pred_sample = float(yaw_vals.reshape(-1)[sample].detach().cpu() * rad2deg)
                    yaw_after_carry_sample = float(yaw_write.reshape(-1)[sample].detach().cpu() * rad2deg)
                    yaw_diff_deg = math.degrees(
                        float(
                            torch.atan2(
                                torch.sin(yaw_vals.reshape(-1)[sample] - yaw_cmd_vals.reshape(-1)[sample]),
                                torch.cos(yaw_vals.reshape(-1)[sample] - yaw_cmd_vals.reshape(-1)[sample]),
                            ).item()
                        )
                    )
                    rootvel_slice = x_next[sample, rootvel_sl].detach().cpu().tolist()
                    cond_speed_sample = float(cond_speed.reshape(-1)[sample].detach().cpu())
                    ortho_err = float('nan')
                    rot_width = rx.stop - rx.start
                    if rot_width % 6 == 0 and rot_width > 0:
                        J = rot_width // 6
                        curr6 = x_next[sample : sample + 1, rx].reshape(1, J, 6)
                        R = rot6d_to_matrix(curr6)
                        eye = torch.eye(3, device=R.device, dtype=R.dtype)
                        diff = torch.matmul(R.transpose(-1, -2), R) - eye
                        ortho_err = float(diff.abs().mean().detach().cpu())
                    delta_norm_debug = None
                    delta_raw_debug = None
                    rot6d_geo_debug = None
                    stats = getattr(self, '_last_step_debug_stats', None)
                    if isinstance(stats, dict):
                        delta_norm_debug = stats.get('delta_norm_abs_mean')
                        delta_raw_debug = stats.get('delta_raw_abs_mean')
                        rot6d_geo_debug = stats.get('rot6d_geo_deg')
                    buffer.append(
                        {
                            "yaw_cmd_deg": yaw_cmd_sample,
                            "yaw_after_carry_deg": yaw_after_carry_sample,
                            "yaw_rot6d_deg": yaw_pred_sample,
                            "yaw_cmd_diff_deg": yaw_diff_deg,
                            "root_vel_pred": rootvel_slice,
                            "cond_speed": cond_speed_sample,
                            "rot6d_ortho_err": ortho_err,
                            "delta_norm_abs_mean": delta_norm_debug,
                            "delta_raw_abs_mean": delta_raw_debug,
                            "rot6d_geo_deg": rot6d_geo_debug,
                        }
                    )
                except Exception:
                    buffer.append(
                        {
                            "yaw_cmd_deg": float('nan'),
                            "yaw_after_carry_deg": float('nan'),
                            "yaw_rot6d_deg": float('nan'),
                            "yaw_cmd_diff_deg": float('nan'),
                            "root_vel_pred": [],
                            "cond_speed": float('nan'),
                            "rot6d_ortho_err": float('nan'),
                        }
                    )

        return x_next

    def _prepare_cond_stat(self, stat: Optional[torch.Tensor], ref_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if stat is None:
            return None
        import torch
        if not torch.is_tensor(stat):
            stat_t = torch.as_tensor(stat, device=ref_tensor.device, dtype=ref_tensor.dtype)
        else:
            stat_t = stat.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        if stat_t.dim() >= 3:
            stat_t = stat_t.view(stat_t.shape[0], -1)
        if stat_t.dim() == 1:
            stat_t = stat_t.unsqueeze(0)
        if stat_t.size(0) == 1 and ref_tensor.size(0) > 1:
            stat_t = stat_t.expand(ref_tensor.size(0), -1).contiguous()
        return stat_t

    def _normalize_cond_from_raw(
        self,
        cond_raw: Optional[torch.Tensor],
        cond_mu: Optional[torch.Tensor],
        cond_std: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        import torch
        if cond_raw is None or cond_mu is None or cond_std is None:
            return None
        if cond_mu.dim() == 3:
            cond_mu = cond_mu.squeeze(1)
        if cond_std.dim() == 3:
            cond_std = cond_std.squeeze(1)
        if cond_mu.shape != cond_raw.shape:
            # broadcast along batch if mu/std have single row
            if cond_mu.size(0) == 1 and cond_raw.size(0) > 1:
                cond_mu = cond_mu.expand(cond_raw.size(0), -1)
            if cond_std.size(0) == 1 and cond_raw.size(0) > 1:
                cond_std = cond_std.expand(cond_raw.size(0), -1)
        std = cond_std.clamp_min(1e-6)
        cond_norm = (cond_raw - cond_mu) / std
        clamp_val = float(getattr(self, 'cond_norm_clip', 6.0) or 0.0)
        if clamp_val > 0:
            cond_norm = cond_norm.clamp(-clamp_val, clamp_val)
        return cond_norm

    def _pose_hist_params(self, ref: torch.Tensor) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns pose-history normalizer tensors aligned with the reference tensor's device/dtype.
        """
        if self.pose_hist_scales is None or self.pose_hist_dim <= 0:
            return None, None, None
        device = ref.device
        dtype = ref.dtype
        scales = self.pose_hist_scales.to(device=device, dtype=dtype)
        mu = self.pose_hist_mu.to(device=device, dtype=dtype) if self.pose_hist_mu is not None else None
        std = self.pose_hist_std.to(device=device, dtype=dtype) if self.pose_hist_std is not None else None
        return scales, mu, std

    @staticmethod
    def _pose_hist_transform_vec(raw_flat: torch.Tensor, scales: Optional[torch.Tensor], mu: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply VectorTanhNormalizer-style transform on flattened pose-history raw values.
        Reuses shared torch normalizer to keep behavior aligned with pretrain.
        """
        if scales is None or raw_flat.numel() == 0:
            return raw_flat
        norm = VectorTanhNormalizerTorch(scales, mu, std)
        # ensure buffers on same device/dtype as input
        norm = norm.to(device=raw_flat.device, dtype=raw_flat.dtype)
        return norm(raw_flat)

    @staticmethod
    def _pose_hist_inverse_vec(norm_flat: torch.Tensor, scales: Optional[torch.Tensor], mu: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Inverse of VectorTanhNormalizer.transform on flattened pose-history vectors.
        """
        if scales is None or norm_flat.numel() == 0:
            return norm_flat
        norm = VectorTanhNormalizerTorch(scales, mu, std)
        norm = norm.to(device=norm_flat.device, dtype=norm_flat.dtype)
        return norm.inverse(norm_flat)

    def _infer_root_yaw_from_rot6d(self, y_denorm: "torch.Tensor"):
        import torch
        ry = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
        if not isinstance(ry, slice):
            return None
        try:
            rot_flat = y_denorm[..., ry]
        except Exception:
            return None
        if rot_flat.numel() == 0:
            return None
        J = (ry.stop - ry.start) // 6
        if J <= 0:
            return None
        try:
            rot6d = reproject_rot6d(rot_flat).view(rot_flat.shape[0], J, 6)
            R = rot6d_to_matrix(rot6d)
        except Exception:
            return None
        root_idx = int(getattr(self, 'eval_root_idx', 0))
        root_idx = max(0, min(J - 1, root_idx))
        root_R = R[:, root_idx]
        up_axis = int(getattr(self, 'eval_up_axis', getattr(self, '_up_axis', 2)))
        up_axis = max(0, min(2, up_axis))
        forward_axis = int(getattr(self, 'yaw_forward_axis', 2))
        forward_axis = max(0, min(2, forward_axis))
        forward_vec = root_R[..., forward_axis]
        planar_axes = [ax for ax in (0, 1, 2) if ax != up_axis]
        if len(planar_axes) != 2:
            return None
        ax0, ax1 = planar_axes
        yaw = torch.atan2(forward_vec[..., ax1], forward_vec[..., ax0])
        offset = float(getattr(self, 'yaw_forward_axis_offset', 0.0))
        if offset != 0.0:
            yaw = yaw - offset
        return torch.atan2(torch.sin(yaw), torch.cos(yaw))

    def _train_augment_if_needed(self, state_seq, gt_seq, cond_seq=None):
        """仅训练阶段使用的时序/噪声增强。"""
        import torch
        aug = getattr(self, 'augmentor', None)
        if aug is None:
            return state_seq, gt_seq, cond_seq

        prob = float(getattr(aug, 'time_warp_prob', 0.0) or 0.0)
        if prob > 0.0 and torch.rand(1, device=state_seq.device).item() < prob:
            scale = float(torch.empty(1, device=state_seq.device).uniform_(0.85, 1.15).item())
            state_seq = aug._time_warp(state_seq, scale)
            gt_seq = aug._time_warp(gt_seq, scale)
            if (cond_seq is not None) and (cond_seq.dim() == 3):
                cond_seq = aug._time_warp(cond_seq, scale)

        std = float(getattr(aug, 'noise_std', 0.0) or 0.0)
        if std > 0.0:
            def _n(sl):
                if isinstance(sl, slice):
                    state_seq[:, :, sl] = state_seq[:, :, sl] + torch.randn_like(state_seq[:, :, sl]) * std

            _n(getattr(self, 'rot6d_x_slice', None))
            _n(getattr(self, 'rootvel_x_slice', None))
            _n(getattr(self, 'angvel_x_slice', None))
            _n(getattr(self, 'yaw_x_slice', None))

        return state_seq, gt_seq, cond_seq

    def _metrics_json_safe(self, value):
        import math
        import torch
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore

        if isinstance(value, dict):
            return {str(k): self._metrics_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._metrics_json_safe(v) for v in value]
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                try:
                    return self._metrics_json_safe(value.item())
                except Exception:
                    pass
            return [self._metrics_json_safe(v) for v in value.detach().cpu().tolist()]
        if np is not None and isinstance(value, np.ndarray):  # type: ignore[arg-type]
            return [self._metrics_json_safe(v) for v in value.tolist()]
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        try:
            if np is not None and isinstance(value, np.generic):  # type: ignore[arg-type]
                return self._metrics_json_safe(float(value))
        except Exception:
            pass
        if hasattr(value, 'item') and not isinstance(value, (int, bool, str)):
            try:
                return self._metrics_json_safe(value.item())
            except Exception:
                pass
        if isinstance(value, (int, str, bool)) or value is None:
            return value
        return str(value)

    def register_metric_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """注册监听器，在每次记录指标时得到通知（运行在同一进程内）。"""
        if not callable(callback):
            return
        if callback not in self._metric_callbacks:
            self._metric_callbacks.append(callback)

    def unregister_metric_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        if not callable(callback):
            return
        try:
            self._metric_callbacks.remove(callback)
        except ValueError:
            pass

    def get_metric_history(self, tag: Optional[str] = None, last: Optional[int] = None) -> list[dict[str, Any]]:
        """返回内存中的指标快照，用于训练过程内的策略决策。"""
        records = self.metric_history
        if tag is not None:
            records = [rec for rec in records if rec.get('tag') == tag]
        if last is not None and last > 0:
            records = records[-last:]
        return [dict(rec) for rec in records]

    def latest_epoch_metrics(self, tag: Optional[str] = None) -> Optional[dict[str, Any]]:
        """获取最近一次写入的指标（可按 tag 过滤）。"""
        if tag is not None:
            record = self.latest_metrics.get(str(tag))
            return None if record is None else dict(record)
        if not self.metric_history:
            return None
        return dict(self.metric_history[-1])

    def _record_epoch_metrics(self, metrics: Dict[str, Any], *, tag: str, epoch: int) -> None:
        if metrics is None:
            return
        payload: dict[str, Any] = {
            'epoch': int(epoch),
            'tag': str(tag),
            'metrics': self._metrics_json_safe(metrics),
        }
        tf_ratio = getattr(self, '_last_tf_ratio', None)
        if tf_ratio is not None:
            try:
                payload['tf_ratio'] = float(tf_ratio)
            except Exception:
                payload['tf_ratio'] = tf_ratio
        maxlen = max(1, int(getattr(self, 'metric_history_maxlen', 256) or 256))
        self.metric_history.append(payload)
        if len(self.metric_history) > maxlen:
            self.metric_history.pop(0)
        self.latest_metrics[str(tag)] = payload
        for callback in list(self._metric_callbacks):
            try:
                callback(payload)
            except Exception as exc:
                print(f"[MetricsCallback][WARN] {callback} raised: {exc}")

    def _dump_metrics_json(self, metrics: Dict[str, Any], *, tag: str, epoch: int) -> None:
        out_dir = getattr(self, 'out_dir', None)
        if not out_dir:
            return
        try:
            metrics_dir = os.path.join(out_dir, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            payload = {
                'epoch': int(epoch),
                'tag': str(tag),
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
                'metrics': self._metrics_json_safe(metrics),
            }
            json_path = os.path.join(metrics_dir, f'{tag}_ep{int(epoch):03d}.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[MetricsWrite][WARN] failed to write {tag} metrics @ep{epoch}: {exc}")

    def _save_val_metrics(self, epoch: int, metrics: Mapping[str, Any]) -> Optional[Path]:
        out_dir = getattr(self, 'out_dir', None)
        if not out_dir:
            return None
        try:
            self._dump_metrics_json(dict(metrics), tag='valfree', epoch=epoch)
        except Exception:
            pass
        metrics_dir = Path(out_dir) / 'metrics'
        json_path = metrics_dir / f'valfree_ep{int(epoch):03d}.json'
        return json_path if json_path.exists() else None

    @torch.no_grad()
    def _diagnose_free_run(
        self,
        batch,
        predY,
        gtY,
        predsX,
        period_seq_pred,
        motion_seq,
        y_seq,
        contacts_seq,
        angvel_seq,
        pose_hist_seq,
        angvel_raw_seq=None,
    ):
        diag = _diagnose_free_run_impl(
            self,
            batch,
            predY,
            gtY,
            predsX,
            period_seq_pred,
            motion_seq,
            y_seq,
            contacts_seq,
            angvel_seq,
            pose_hist_seq,
            angvel_raw_seq=angvel_raw_seq,
        )
        if diag is None:
            clip = None
            start = None
            if isinstance(batch, dict):
                clip = batch.get('clip_id')
                start = batch.get('start')
            msg = f"_diagnose_free_run returned None (clip={clip}, start={start})"
            print(f"[FreeRunDiag][WARN] {msg}")
        return diag


    def _dump_nan_grad_report(self, epoch, batch_idx, batch, state_seq, gt_seq, preds_dict, loss_value, stats):
        out_dir = getattr(self, 'out_dir', None)
        if not out_dir:
            return
        limit = int(getattr(self, 'nan_grad_report_limit', 0) or 0)
        if self.nan_grad_reports >= limit:
            return
        import os, json

        def _tensor_stats(tensor):
            if tensor is None:
                return None
            try:
                t = tensor.detach()
                if t.numel() == 0:
                    return {'shape': list(t.shape), 'numel': 0}
                t = t.to(dtype=torch.float32, device='cpu')
                return {
                    'shape': list(t.shape),
                    'numel': int(t.numel()),
                    'min': float(t.min().item()),
                    'max': float(t.max().item()),
                    'mean': float(t.mean().item()),
                    'std': float(t.std().item()),
                }
            except Exception as exc:
                return {'error': str(exc)}

        try:
            os.makedirs(os.path.join(out_dir, 'nan_grad'), exist_ok=True)
            payload = {
                'epoch': int(epoch),
                'batch_idx': int(batch_idx),
                'tf_ratio': float(getattr(self, '_last_tf_ratio', 1.0)),
                'loss': float(loss_value),
                'loss_parts': dict(stats) if isinstance(stats, dict) else {},
                'state_stats': _tensor_stats(state_seq),
                'gt_stats': _tensor_stats(gt_seq),
                'pred_out_stats': _tensor_stats(preds_dict.get('out') if isinstance(preds_dict, dict) else None),
                'pred_delta_stats': _tensor_stats(preds_dict.get('delta') if isinstance(preds_dict, dict) else None),
                'batch_meta': {},
            }
            if isinstance(batch, dict):
                clip_id = batch.get('clip_id')
                start = batch.get('start')
                if clip_id is not None:
                    try:
                        payload['batch_meta']['clip_id'] = int(clip_id if isinstance(clip_id, int) else clip_id.item())
                    except Exception:
                        pass
                if start is not None:
                    try:
                        payload['batch_meta']['start'] = int(start if isinstance(start, int) else start.item())
                    except Exception:
                        pass
            fname = os.path.join(out_dir, 'nan_grad', f'ep{int(epoch):03d}_b{int(batch_idx):05d}.json')
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.nan_grad_reports += 1
            print(f"[GradNan] dumped diagnostic to {fname}")
        except Exception as exc:
            print(f"[GradNan][WARN] failed to dump diagnostic: {exc}")


def _diagnose_free_run_impl(
    self,
    batch,
    predY,
    gtY,
    predsX,
    period_seq_pred,
    motion_seq,
    y_seq,
    contacts_seq,
    angvel_seq,
    pose_hist_seq,
    angvel_raw_seq=None,
):
    import math
    import torch
    self._require_normalizer("_diagnose_free_run_impl")

    result: Dict[str, Any] = {}
    diag_scope = getattr(self, '_diag_scope', 'free_run')

    def _record_metric(name: str, value):
        result[name] = value
        if diag_scope == 'free_run':
            result[f'FreeRun/{name}'] = value
        elif diag_scope == 'single_step':
            result[f'SingleStep/{name}'] = value

    rot6d_y = self._sl_from_layout(getattr(self, '_y_layout', {}), 'BoneRotations6D')
    yaw_x = self._sl_from_layout(getattr(self, '_x_layout', {}), 'RootYaw')
    rv_x = self._sl_from_layout(getattr(self, '_x_layout', {}), 'RootVelocity')
    rot6d_x = self._sl_from_layout(getattr(self, '_x_layout', {}), 'BoneRotations6D')

    eval_align_root = bool(getattr(self, 'eval_align_root0', True))
    root_idx = int(getattr(self, 'eval_root_idx', 0))
    up_axis = int(getattr(self, 'eval_up_axis', getattr(self, '_up_axis', 2)))
    fps_eval = float(getattr(self, 'bone_hz', 60.0))
    contact_threshold = float(getattr(self, 'foot_contact_threshold', 1.5))
    mag_rel_beta = float(getattr(self, 'eval_angvel_beta', 0.25) or 0.25)
    mag_rel_threshold = float(getattr(self, 'eval_angvel_mag_threshold', 0.10) or 0.10)
    geo = None
    deg = 180.0 / math.pi

    bone_names_src = getattr(self, '_bone_names', None)
    if not bone_names_src:
        bundle_meta = getattr(self, '_bundle_meta', None)
        if isinstance(bundle_meta, dict):
            bone_names_src = bundle_meta.get('bone_names') or bundle_meta.get('skeleton', {}).get('bone_names')
    bone_names = [str(b) for b in bone_names_src] if isinstance(bone_names_src, (list, tuple)) else []

    def _mean_curve(tensor: torch.Tensor, reduce_dims) -> torch.Tensor:
        cur = tensor
        for dim in sorted(reduce_dims, reverse=True):
            cur = cur.mean(dim=dim)
        return cur

    predX_tensor = torch.stack(predsX, dim=1) if predsX else None
    gtX_raw_full = None
    if motion_seq is not None:
        try:
            flat_motion = motion_seq.reshape(-1, motion_seq.shape[-1])
            gtX_raw_full = self.normalizer.denorm_x(flat_motion).view_as(motion_seq)
        except Exception as exc:
            self._raise_norm_error("normalizer.denorm_x 在诊断阶段还原 GT X 时失败", exc)

    if predX_tensor is not None:
        flat_pred = predX_tensor.reshape(-1, predX_tensor.shape[-1])
        try:
            predX_raw = self.normalizer.denorm_x(flat_pred).view_as(predX_tensor)
        except Exception as exc:
            self._raise_norm_error("normalizer.denorm_x 在诊断阶段还原预测 X 时失败", exc)
        if motion_seq is not None:
            if gtX_raw_full is None:
                self._raise_norm_error("诊断阶段缺少 GT RAW 序列。")
            gtX_raw = gtX_raw_full[:, :predX_tensor.shape[1]]
        else:
            gtX_raw = None
    else:
        predX_raw = None
        gtX_raw = None

    if isinstance(yaw_x, slice) and predX_raw is not None and gtX_raw is not None:
        dyaw = torch.atan2(
            torch.sin(predX_raw[..., yaw_x] - gtX_raw[..., yaw_x]),
            torch.cos(predX_raw[..., yaw_x] - gtX_raw[..., yaw_x]),
        )
        _record_metric('YawAbsDeg', float(dyaw.abs().mean() * (180.0 / math.pi)))
    if isinstance(rv_x, slice) and predX_raw is not None and gtX_raw is not None:
        _record_metric('RootVelMAE', float((predX_raw[..., rv_x] - gtX_raw[..., rv_x]).abs().mean().item()))
        if getattr(self, 'diag_input_stats', False):
            diff = (predX_raw[..., rv_x] - gtX_raw[..., rv_x]).abs()
            result['RootVelMAE_std'] = float(diff.std().item())

    if isinstance(rot6d_x, slice) and predX_raw is not None and gtX_raw is not None:
        try:
            Bx, Tx, Dx = predX_raw.shape
            px = predX_raw[..., rot6d_x]
            gx = gtX_raw[..., rot6d_x]
            if Dx > 0 and px.shape[-1] % 6 == 0:
                Jx = px.shape[-1] // 6
                px6 = reproject_rot6d(px.reshape(-1, px.shape[-1]))
                gx6 = reproject_rot6d(gx.reshape(-1, gx.shape[-1]))
                Rp = rot6d_to_matrix(px6.view(-1, Jx, 6)).view(Bx, Tx, Jx, 3, 3)
                Rg = rot6d_to_matrix(gx6.view(-1, Jx, 6)).view(Bx, Tx, Jx, 3, 3)
                geo_in = geodesic_R(Rp, Rg)
                _record_metric('InputRotGeoDeg', float((geo_in.mean() * deg).item()))
                result['FreeRun/InputRotGeoDeg'] = result['InputRotGeoDeg']
                if getattr(self, 'diag_input_stats', False):
                    result['InputRotGeoDeg_max'] = float((geo_in.max() * deg).item())
                    result['InputRotGeoDeg_std'] = float((geo_in.std() * deg).item())
                try:
                    geo_in_deg = geo_in * deg
                    mean_curve = _mean_curve(geo_in_deg.mean(dim=-1), reduce_dims=(0,))
                    max_curve = geo_in_deg.max(dim=-1).values.max(dim=0).values
                    result['InputRotGeoDegCurve'] = mean_curve.detach().cpu().tolist()
                    result['InputRotGeoDegCurveMax'] = max_curve.detach().cpu().tolist()
                except Exception:
                    pass
        except Exception:
            pass

    # Cond vs motion diagnostics
    cond_raw_seq = None
    if isinstance(batch, dict):
        cond_raw_seq = batch.get("cond_tgt_raw")
        if cond_raw_seq is None:
            cond_raw_seq = batch.get("cond_in")
    if torch.is_tensor(cond_raw_seq):
        cond_raw_seq = cond_raw_seq.float()
        if cond_raw_seq.dim() == 2:
            cond_raw_seq = cond_raw_seq.unsqueeze(0)
        if cond_raw_seq.dim() == 3 and predX_raw is not None:
            B = predX_raw.shape[0]
            if cond_raw_seq.shape[0] == B:
                start_idx = 1
                horizon = predX_raw.shape[1]
                if cond_raw_seq.shape[1] >= start_idx + horizon:
                    cond_slice = cond_raw_seq[:, start_idx:start_idx + horizon]
                else:
                    cond_slice = cond_raw_seq[:, -horizon:]
                cond_dim = cond_slice.shape[-1]
                if cond_dim >= 2:
                    if cond_dim >= 3:
                        dir_slice = cond_slice[..., cond_dim - 3:cond_dim - 1]
                        speed_slice = cond_slice[..., -1]
                    else:
                        dir_slice = cond_slice[..., -2:]
                        speed_slice = dir_slice.norm(dim=-1)
                    L = min(cond_slice.shape[1], predX_raw.shape[1])
                    if L > 0:
                        device = predX_raw.device
                        dir_slice = dir_slice[:, :L].to(device)
                        speed_slice = speed_slice[:, :L].to(device)
                        dir_norm = dir_slice.norm(dim=-1).clamp_min(1e-6)
                        dir_unit = dir_slice / dir_norm.unsqueeze(-1)
                        yaw_cmd_world = torch.atan2(dir_unit[..., 1], dir_unit[..., 0])
                        offset = float(getattr(self, 'yaw_forward_axis_offset', 0.0) or 0.0)
                        yaw_cmd = torch.atan2(
                            torch.sin(yaw_cmd_world - offset),
                            torch.cos(yaw_cmd_world - offset),
                        )
                        if isinstance(yaw_x, slice):
                            yaw_pred = predX_raw[:, :L, yaw_x].squeeze(-1)
                            yaw_diff_pred = torch.atan2(torch.sin(yaw_pred - yaw_cmd), torch.cos(yaw_pred - yaw_cmd)).abs()
                            yaw_diff_pred_deg = yaw_diff_pred * deg
                            _record_metric('CondYawVsPredDeg', float(yaw_diff_pred_deg.mean().item()))
                            result['FreeRun/CondYawVsPredDeg'] = result['CondYawVsPredDeg']
                            try:
                                result['CondYawVsPredCurveDeg'] = yaw_diff_pred_deg.mean(dim=0).detach().cpu().tolist()
                            except Exception:
                                pass
                            if gtX_raw is not None and gtX_raw.shape[1] >= start_idx + L:
                                yaw_gt = gtX_raw[:, start_idx:start_idx + L, yaw_x].squeeze(-1)
                                yaw_diff_gt = torch.atan2(torch.sin(yaw_gt - yaw_cmd), torch.cos(yaw_gt - yaw_cmd)).abs()
                                yaw_diff_gt_deg = yaw_diff_gt * deg
                                _record_metric('CondYawVsGTDeg', float(yaw_diff_gt_deg.mean().item()))
                                result['FreeRun/CondYawVsGTDeg'] = result['CondYawVsGTDeg']
                                try:
                                    result['CondYawVsGTCurveDeg'] = yaw_diff_gt_deg.mean(dim=0).detach().cpu().tolist()
                                except Exception:
                                    pass
                        if isinstance(rv_x, slice):
                            cond_vel = dir_unit * speed_slice.unsqueeze(-1)
                            vel_pred = predX_raw[:, :L, rv_x]
                            _record_metric('CondVelVsPredMAE', float((vel_pred - cond_vel).abs().mean().item()))
                            result['FreeRun/CondVelVsPredMAE'] = result['CondVelVsPredMAE']
                            if gtX_raw is not None and gtX_raw.shape[1] >= start_idx + L:
                                vel_gt = gtX_raw[:, start_idx:start_idx + L, rv_x]
                                _record_metric('CondVelVsGTMAE', float((vel_gt - cond_vel).abs().mean().item()))
                                result['FreeRun/CondVelVsGTMAE'] = result['CondVelVsGTMAE']

    if period_seq_pred:
        try:
            norm_period = []
            for p in period_seq_pred:
                if p.dim() == 3 and p.size(1) == 1:
                    norm_period.append(p.squeeze(1))
                elif p.dim() == 2:
                    norm_period.append(p)
                else:
                    norm_period.append(p.reshape(p.shape[0], -1))
            if norm_period:
                period_tensor = torch.stack(norm_period, dim=1)
                result['period_abs_mean'] = float(period_tensor.abs().mean().item())
                result['period_abs_std'] = float(period_tensor.abs().std().item())
        except Exception:
            pass

    w_pred = w_gt = None
    angvel_slice = getattr(self, 'angvel_x_slice', None)
    if isinstance(rot6d_y, slice):
        predY_raw = self._denorm(predY)
        gtY_raw = self._denorm(gtY)
        py = predY_raw[..., rot6d_y]
        gy = gtY_raw[..., rot6d_y]
        if py.shape[-1] % 6 == 0:
            J = py.shape[-1] // 6
            py6 = reproject_rot6d(py).view(py.shape[0], py.shape[1], J, 6)
            gy6 = reproject_rot6d(gy).view(gy.shape[0], gy.shape[1], J, 6)
            Rp = rot6d_to_matrix(py6)
            Rg = rot6d_to_matrix(gy6)
            if eval_align_root and Rp.shape[1] > 0 and 0 <= root_idx < J:
                Rpr0 = Rp[:, 0, root_idx]
                Rgr0 = Rg[:, 0, root_idx]
                R_align = Rgr0 @ Rpr0.transpose(-1, -2)
                Rp = (R_align.view(Rp.shape[0], 1, 1, 3, 3).expand_as(Rp)) @ Rp
            geo = geodesic_R(Rp, Rg)
            _record_metric('GeoDeg', float((geo.mean() * deg).item()))
            result['SingleStep/GeoDeg'] = result['GeoDeg']
            try:
                geo_deg = geo * deg
                geo_curve = _mean_curve(geo_deg.mean(dim=-1), reduce_dims=(0,))
                geo_curve_max = geo_deg.max(dim=-1).values.max(dim=0).values
                result['GeoDegCurve'] = geo_curve.detach().cpu().tolist()
                result['GeoDegCurveMax'] = geo_curve_max.detach().cpu().tolist()
            except Exception:
                pass
            geo_local = None
            try:
                Rp_root = _root_relative_matrices(Rp, root_idx)
                Rg_root = _root_relative_matrices(Rg, root_idx)
                geo_local = geodesic_R(Rp_root, Rg_root) * deg
                joint_weights = self._joint_weights(Rp_root, J)
                weights_sum = joint_weights.sum().clamp_min(1e-6)
                w = joint_weights.view(1, 1, -1)
                geo_local_mean = (geo_local * w).sum() / (weights_sum * geo_local.shape[0] * geo_local.shape[1])
                _record_metric('GeoLocalDeg', float(geo_local_mean.item()))
                result['SingleStep/GeoLocalDeg'] = result['GeoLocalDeg']
                step_vals = ((geo_local * w).sum(dim=-1) / weights_sum).mean(dim=0)
                result['GeoLocalDegCurve'] = step_vals.detach().cpu().tolist()
                max_vals = geo_local.max(dim=-1).values.max(dim=0).values
                result['GeoLocalDegCurveMax'] = max_vals.detach().cpu().tolist()
            except Exception as exc:
                raise RuntimeError(
                    "GeoLocalDeg diagnostics require valid skeleton FK; ensure bundle/meta provide parents & offsets."
                ) from exc
            try:
                Rp_parent = self._parent_relative_matrices(Rp_root)
                Rg_parent = self._parent_relative_matrices(Rg_root)
                w_pred = angvel_vec_from_R_seq(Rp_parent, fps_eval)
                w_gt = angvel_vec_from_R_seq(Rg_parent, fps_eval)
                _record_metric('AngVelMAE', float((w_pred - w_gt).abs().mean().item()))
                mag_p = w_pred.norm(dim=-1)
                mag_g = w_gt.norm(dim=-1)
                mag_avg = 0.5 * (mag_p + mag_g)
                maskA = (mag_avg > mag_rel_threshold)
                mag_rel = (mag_p - mag_g).abs() / (mag_avg + mag_rel_beta)
                ang_mag_rel = (mag_rel * maskA).sum(dim=(0, 1)) / maskA.sum(dim=(0, 1)).clamp_min(1)
                _record_metric('AngVelMagRel', float(torch.nanmedian(ang_mag_rel).item()))
                try:
                    dot_full = (w_pred * w_gt).sum(dim=-1)
                    norm_full = mag_p * mag_g
                    ang_full = torch.zeros_like(dot_full)
                    valid_full = norm_full > 1e-6
                    if valid_full.any():
                        ang_full[valid_full] = torch.acos(torch.clamp(dot_full[valid_full] / norm_full[valid_full], -1.0, 1.0))
                    ang_full_deg = ang_full * deg
                    ang_curve = ang_full_deg.mean(dim=(0, 2))
                    ang_curve_max = ang_full_deg.max(dim=2).values.max(dim=0).values
                    result['AngVelDirDegCurve'] = ang_curve.detach().cpu().tolist()
                    result['AngVelDirDegCurveMax'] = ang_curve_max.detach().cpu().tolist()
                    if diag_scope == 'single_step':
                        result['SingleStep/AngVelDirDegCurve'] = result['AngVelDirDegCurve']
                        result['SingleStep/AngVelDirDegCurveMax'] = result['AngVelDirDegCurveMax']
                    try:
                        summary = self._summarize_angvel_dir(w_pred, w_gt, bone_names=bone_names)
                    except Exception as _angvel_exc:
                        print(f"[Diag][WARN] _summarize_angvel_dir failed: {_angvel_exc}")
                        import traceback
                        traceback.print_exc()
                        summary = {}
                    if summary:
                        _record_metric('AngVelDirDegRaw', summary.get('raw', float('nan')))
                        _record_metric('AngVelDirDegWeighted', summary.get('weighted', float('nan')))
                        _record_metric('AngVelDirDegSmooth', summary.get('smooth', float('nan')))
                        _record_metric('AngVelDirDegTorso', summary.get('torso', float('nan')))
                        _record_metric('AngVelDirDegProximal', summary.get('proximal', float('nan')))
                        _record_metric('AngVelDirDegDistal', summary.get('distal', float('nan')))
                except Exception:
                    pass
            except Exception:
                pass

    if bone_names:
        if w_pred is not None and w_gt is not None:
            for j_idx, name in enumerate(bone_names):
                if j_idx >= w_pred.shape[2]:
                    continue
                w_pred_b = w_pred[..., j_idx, :]
                w_gt_b = w_gt[..., j_idx, :]
                mag_mae = float((w_pred_b.norm(dim=-1) - w_gt_b.norm(dim=-1)).abs().mean().item())
                ang_mae = float((w_pred_b - w_gt_b).abs().mean().item())
                result[f'Bone/{name}/AngVelMagMAE'] = mag_mae
                result[f'Bone/{name}/AngVelMAE'] = ang_mae
        key_bone_names = getattr(self, 'eval_key_bones', None)
        if not key_bone_names:
            key_bone_names = [
                'upperarm_l', 'lowerarm_l', 'hand_l',
                'upperarm_r', 'lowerarm_r', 'hand_r',
                'thigh_l', 'calf_l', 'foot_l',
                'thigh_r', 'calf_r', 'foot_r',
            ]
        idx_map = {name: idx for idx, name in enumerate(bone_names)}
        key_indices = [idx_map[name] for name in key_bone_names if name in idx_map]
        key_geo_vals: list[float] = []
        key_geo_local_vals: list[float] = []
        key_ang_mae_vals: list[float] = []
        key_ang_mag_mae_vals: list[float] = []
        key_ang_mag_rel_vals: list[float] = []
        key_ang_dir_vals: list[float] = []
        keybone_details: Dict[str, Dict[str, float]] = {}
        geo_local_tensor = geo_local if torch.is_tensor(geo_local) else None
        if geo_local_tensor is None:
            raise RuntimeError(
                "GeoLocalDeg metrics unavailable; ensure FK + geodesic computation succeeded before KeyBone diagnostics."
            )
        for name in key_bone_names:
            if name not in idx_map:
                continue
            j_idx = idx_map[name]
            prefix = f'KeyBone/{name}'
            if geo is not None and geo.shape[-1] > j_idx:
                geo_val = float((geo[..., j_idx].mean() * (180.0 / math.pi)).item())
            else:
                geo_val = float('nan')
            result[f'{prefix}/GeoDeg'] = geo_val
            if math.isfinite(geo_val):
                key_geo_vals.append(geo_val)

            geo_local_val = float('nan')
            if geo_local_tensor is not None and geo_local_tensor.shape[-1] > j_idx:
                geo_local_val = float(geo_local_tensor[..., j_idx].mean().item())
            result[f'{prefix}/GeoLocalDeg'] = geo_local_val
            if math.isfinite(geo_local_val):
                key_geo_local_vals.append(geo_local_val)

            if w_pred is not None and w_gt is not None and w_pred.shape[2] > j_idx:
                w_pred_b = w_pred[..., j_idx, :]
                w_gt_b = w_gt[..., j_idx, :]
                ang_mae = float((w_pred_b - w_gt_b).abs().mean().item())
                result[f'{prefix}/AngVelMAE'] = ang_mae
                if math.isfinite(ang_mae):
                    key_ang_mae_vals.append(ang_mae)

                mag_p = w_pred_b.norm(dim=-1)
                mag_g = w_gt_b.norm(dim=-1)
                mag_avg = 0.5 * (mag_p + mag_g)
                mag_rel = (mag_p - mag_g).abs() / (mag_avg + mag_rel_beta)
                mag_mae = float((mag_p - mag_g).abs().mean().item())
                result[f'{prefix}/AngVelMagMAE'] = mag_mae
                if math.isfinite(mag_mae):
                    key_ang_mag_mae_vals.append(mag_mae)

                valid_mag = mag_avg > mag_rel_threshold
                mag_rel_val = float(torch.median(mag_rel[valid_mag]).item()) if valid_mag.any() else float('nan')
                result[f'{prefix}/AngVelMagRel'] = mag_rel_val
                if math.isfinite(mag_rel_val):
                    key_ang_mag_rel_vals.append(mag_rel_val)

                dir_val = geo_local_val
                if not math.isfinite(dir_val):
                    raise RuntimeError(
                        f"GeoLocalDeg for key bone '{name}' is NaN; ensure FK skeleton matches outputs."
                    )
                result[f'{prefix}/AngVelDirDeg'] = dir_val
                key_ang_dir_vals.append(dir_val)
                keybone_details[name] = {
                    'GeoDeg': geo_val,
                    'GeoLocalDeg': geo_local_val,
                    'AngVelMAE': ang_mae,
                    'AngVelMagMAE': mag_mae,
                    'AngVelMagRel': mag_rel_val,
                    'AngVelDirDeg': dir_val,
                }
            else:
                result[f'{prefix}/AngVelMAE'] = float('nan')
                result[f'{prefix}/AngVelMagMAE'] = float('nan')
                result[f'{prefix}/AngVelMagRel'] = float('nan')
                dir_val = geo_local_val
                if not math.isfinite(dir_val):
                    raise RuntimeError(
                        f"GeoLocalDeg for key bone '{name}' is NaN; ensure FK skeleton matches outputs."
                    )
                result[f'{prefix}/AngVelDirDeg'] = dir_val
                key_ang_dir_vals.append(dir_val)
                keybone_details[name] = {
                    'GeoDeg': geo_val,
                    'GeoLocalDeg': geo_local_val,
                    'AngVelMAE': float('nan'),
                    'AngVelMagMAE': float('nan'),
                    'AngVelMagRel': float('nan'),
                    'AngVelDirDeg': dir_val,
                }

        summary = {}
        if key_geo_vals:
            summary['GeoDegMean'] = float(sum(key_geo_vals) / len(key_geo_vals))
            _record_metric('KeyBone/GeoDegMean', summary['GeoDegMean'])
        if key_ang_mae_vals:
            summary['AngVelMAE'] = float(sum(key_ang_mae_vals) / len(key_ang_mae_vals))
            _record_metric('KeyBone/AngVelMAE', summary['AngVelMAE'])
        if key_ang_mag_mae_vals:
            summary['AngVelMagMAE'] = float(sum(key_ang_mag_mae_vals) / len(key_ang_mag_mae_vals))
            _record_metric('KeyBone/AngVelMagMAE', summary['AngVelMagMAE'])
        if key_ang_mag_rel_vals:
            summary['AngVelMagRel'] = float(sum(key_ang_mag_rel_vals) / len(key_ang_mag_rel_vals))
            _record_metric('KeyBone/AngVelMagRel', summary['AngVelMagRel'])
        if not key_geo_local_vals:
            raise RuntimeError("KeyBone GeoLocalDegMean is empty; diagnostics require valid limb geodesic values.")
        summary['GeoLocalDegMean'] = float(sum(key_geo_local_vals) / len(key_geo_local_vals))
        _record_metric('KeyBone/GeoLocalDegMean', summary['GeoLocalDegMean'])
        if key_ang_dir_vals:
            summary['AngVelDirDeg'] = float(sum(key_ang_dir_vals) / len(key_ang_dir_vals))
            _record_metric('KeyBone/AngVelDirDeg', summary['AngVelDirDeg'])
        if key_indices:
            kb_curve = geo_local_tensor[:, :, key_indices].mean(dim=(0, 2))
            result['KeyBone/AngVelDirDegCurve'] = kb_curve.detach().cpu().tolist()
        if keybone_details:
            _record_metric('KeyBoneDetails', keybone_details)
        if summary:
            _record_metric('KeyBoneSummary', summary)

    if w_gt is not None and gtX_raw_full is not None and isinstance(angvel_slice, slice):
        try:
            angvel_data = gtX_raw_full[:, :w_gt.shape[1]+1, angvel_slice]
            J_ang = (angvel_slice.stop - angvel_slice.start) // 3
            if J_ang == w_gt.shape[2]:
                angvel_data = angvel_data[:, 1:w_gt.shape[1]+1].reshape(w_gt.shape[0], w_gt.shape[1], J_ang, 3)
                diff_gt = (w_gt - angvel_data).abs()
                result['AngVelGTReconMAE'] = float(diff_gt.mean().item())
                dot_gt = (w_gt * angvel_data).sum(dim=-1)
                norm_gt = w_gt.norm(dim=-1) * angvel_data.norm(dim=-1)
                mask_gt = norm_gt > 1e-6
                if mask_gt.any():
                    ang_dir = torch.acos(torch.clamp(dot_gt[mask_gt] / norm_gt[mask_gt], -1.0, 1.0)) * (180.0 / math.pi)
                    result['AngVelGTReconDirDeg'] = float(ang_dir.mean().item())
        except Exception:
            pass

    if w_pred is not None:
        contact_pred = (w_pred.norm(dim=-1) < contact_threshold).float()
        result['FootContact'] = float(contact_pred.mean().item())

    return result

def train_entry():
    global GLOBAL_ARGS
    import argparse, warnings, os, glob, time, math, json, ast
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    # ---- Slice helpers (inserted) ----
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        '--config_json',
        type=str,
        default=None,
        help='JSON 配置文件路径。键名需与 CLI 参数一致，并作为默认值参与解析。',
    )

    config_args, remaining_argv = config_parser.parse_known_args()

    META_KEYS = {'dataset_profile', 'strategy_meta'}

    def _load_config_defaults(config_path: Optional[str], parser: argparse.ArgumentParser) -> Dict[str, Any]:
        if not config_path:
            return {}
        cfg_path = os.path.expanduser(config_path)
        if not os.path.isfile(cfg_path):
            parser.error(f"[config_json] 文件不存在: {cfg_path}")
        with open(cfg_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if not isinstance(payload, Mapping):
            parser.error(f"[config_json] 根对象必须是 JSON dict，当前类型 {type(payload).__name__}")
        valid_dests = {action.dest for action in parser._actions if action.dest and action.dest != 'help'}
        unknown_keys = sorted(k for k in payload.keys() if k not in valid_dests and k not in META_KEYS)
        if unknown_keys:
            parser.error(f"[config_json] 存在未识别字段: {', '.join(unknown_keys)}")
        print(f"[config_json] Loaded defaults from {cfg_path} ({len(payload)} keys)")
        return dict(payload)

    def _apply_config_overrides(namespace: argparse.Namespace, overrides: Optional[Sequence[str]], parser: argparse.ArgumentParser) -> None:
        if not overrides:
            return

        def _parse_literal(raw: str):
            txt = raw.strip()
            if not txt:
                return txt
            try:
                return ast.literal_eval(txt)
            except Exception:
                lowered = txt.lower()
                if lowered == 'none':
                    return None
                return txt

        applied: Dict[str, Any] = {}
        for entry in overrides:
            if not entry:
                continue
            if '=' not in entry:
                parser.error(f"[config_override] 期望 KEY=VALUE，实际收到: {entry}")
            key, value_expr = entry.split('=', 1)
            key = key.strip()
            if not key:
                parser.error('[config_override] 键名不能为空')
            if not hasattr(namespace, key):
                parser.error(f"[config_override] 未知键名: {key}")
            new_value = _parse_literal(value_expr)
            setattr(namespace, key, new_value)
            applied[key] = new_value
        if applied:
            formatted = ', '.join(f"{k}={applied[k]}" for k in sorted(applied))
            print(f"[config_override] Applied: {formatted}")

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[config_parser])
    p.add_argument('--val_mode', type=str, default='online', choices=['online','none'])
    p.add_argument(
        '--encoder_path',
        type=str,
        default='models/motion_encoder_equiv.pt',
        help='预训练 MotionEncoder bundle 路径（.pt，比如第二阶段导出的 motion_encoder_equiv.pt）',
    )
    p.add_argument('--norm_template', type=str, default='raw_data/processed_data/norm_template.json', help='数据归一化模板路径')
    p.add_argument('--pretrain_template', type=str, default='models/pretrain_template.json', help='预训练编码器模板（含角速度统计）')
    p.add_argument('--no_monitor', action='store_true', default=False)
    p.add_argument('--data', type=str, required=True, help='数据目录（含 *.npz）')
    p.add_argument('--out', type=str, default='./runs', help='输出目录根路径')
    p.add_argument('--run_name', type=str, default=None, help='子目录名；未给则用时间戳')
    p.add_argument('--config_override', action='append', default=None, metavar='KEY=VALUE',
                   help='在解析后覆写配置值，可重复，例如 --config_override lr=5e-5')
    p.add_argument('--train_files', type=str, default='', help='逗号分隔的路径/通配/或 @list.txt')
    p.add_argument('--diag_topk', type=int, default=8, help='free-run 评估时打印 X_norm 的 |z| Top-K')
    p.add_argument('--diag_thr', type=float, default=8.0, help='|z| 阈值，统计 X_norm 爆炸比例')
    p.add_argument("--bundle_json", type=str, default=None, help='UE 导出的运行时 bundle（可含 MuY/StdY、feature_layout、MuC_other/StdC_other 等）', required=True)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--min_lr_ratio', type=float, default=0.05)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--tf_mode', type=str, default='epoch_linear', choices=['global', 'epoch_linear'])
    p.add_argument('--tf_warmup_epochs', type=int, default=3)
    p.add_argument('--tf_start_epoch', type=int, default=0)
    p.add_argument('--tf_end_epoch', type=int, default=10)
    p.add_argument('--tf_max', type=float, default=1.0)
    p.add_argument('--tf_min', type=float, default=0.1)
    p.add_argument('--mixed_state_mode', type=str, default='full', choices=['rot6d', 'full'],
                   help='控制 scheduled sampling 混合的特征范围：仅 rot6d 或整条 state。')
    p.add_argument('--freerun_horizon', type=int, default=0,
                   help='>0 时，在每个 batch 内追加该长度的自由滚动序列并复用原 loss。')
    p.add_argument('--freerun_weight', type=float, default=0.1,
                   help='短 horizon 自由滚动 loss 的权重。')
    p.add_argument('--freerun_weight_init', type=float, default=None,
                   help='自由滚动 loss 的初始权重（未指定则按最终权重的 20%% 推断，并在 ramp_epochs 内过渡）。')
    p.add_argument('--freerun_horizon_min', type=int, default=6,
                   help='自由滚动窗口的最小 horizon（默认 6，对应 100ms 左右）。')
    p.add_argument('--freerun_init_horizon', type=int, default=None,
                   help='训练早期的初始 horizon，上限不超过 --freerun_horizon。未指定时自动取约 70%% 的最终 horizon。')
    p.add_argument('--freerun_horizon_ramp_epochs', type=int, default=5,
                   help='多少个 epoch 内将 freerun horizon 从初始值平滑提升到 --freerun_horizon。')
    p.add_argument('--freerun_weight_mode', type=str, default='epoch_linear', choices=['constant', 'epoch_linear'],
                   help='自由滚动 loss 的权重调度方式：常量或按 epoch 线性增权。')
    p.add_argument('--freerun_weight_ramp_epochs', type=int, default=5,
                   help='当 weight_mode=epoch_linear 时，需要多少个 epoch 将权重升至 freerun_weight。')
    p.add_argument('--freerun_grad_log', action='store_true',
                   help='启用 freerun 梯度日志，定期打印 step0 vs stepH 的 grad norm。')
    p.add_argument('--freerun_grad_log_interval', type=int, default=50,
                   help='启用梯度日志时，每隔多少个 batch 采样一次。')
    p.add_argument('--freerun_grad_ratio_alert', type=float, default=0.01,
                   help='若 stepH/step0 的梯度范数比低于该阈值则打印告警。')
    p.add_argument('--freerun_debug_steps', type=int, default=0,
                   help='>0 时，在 freerun 评估中打印前 N 个自回归步的 yaw/速度诊断')
    p.add_argument('--history_debug_steps', type=int, default=0,
                   help='>1 时，在训练批次中额外运行 train_free rollout 诊断历史漂移步数')
    p.add_argument('--history_adaptive_export_frames', type=int, default=0,
                   help='>0 时启用 adaptive history 模块，并指定推理期固定历史帧数')
    p.add_argument('--history_adaptive_max_frames', type=int, default=None,
                   help='训练期允许的最大历史帧数（默认使用 norm_template 中的 pose_hist_len）')
    p.add_argument('--history_adaptive_hidden', type=int, default=256,
                   help='adaptive history 内部隐藏维度')
    p.add_argument('--history_adaptive_heads', type=int, default=2,
                   help='adaptive history 注意力头数')
    p.add_argument('--history_adaptive_train_variable', action='store_true',
                   help='训练时随机截断历史长度，提升部署鲁棒性')
    p.add_argument('--freerun_stage_schedule', type=str, default=None,
                   help='分阶段调度（freerun/tf/损失等）的 JSON/字符串配置。')
    p.add_argument('--adaptive_loss_method', type=str, default='none', choices=['none', 'gradnorm', 'uncertainty', 'dwa'],
                   help='在线损失权重策略（none/gradnorm/uncertainty/dwa）。')
    p.add_argument('--adaptive_loss_alpha', type=float, default=1.5,
                   help='GradNorm 等策略的调节超参。')
    p.add_argument('--adaptive_loss_temperature', type=float, default=2.0,
                   help='DWA 策略温度，默认 2.0。')
    p.add_argument('--adaptive_loss_terms', type=str, default='fk_pos,rot_local,rot_delta,rot_delta_root,rot_ortho',
                   help='需要自适应权重的 loss 名称，逗号分隔。')
    p.add_argument('--adaptive_loss_tuning', action='store_true',
                   help='启用基于验证指标的自适应损失权重调整（StageMetricAdjuster）。')
    p.add_argument('--config_path', type=str, default=None,
                   help='完整配置 JSON 路径（包含阶段调度配置），用于自适应调整。')
    p.add_argument('--adaptive_scheduler', action='store_true',
                   help='启用在线超参调度器（freerun horizon / tf 比例）。')
    p.add_argument('--adaptive_sched_loss_spike', type=float, default=1.5,
                   help='判定 loss spike 的倍数阈值。')
    p.add_argument('--adaptive_sched_convergence', type=float, default=0.02,
                   help='判定收敛的相对标准差阈值。')
    p.add_argument('--adaptive_sched_adjustment', type=float, default=0.1,
                   help='调度器每次调整的相对幅度。')
    p.add_argument('--adaptive_sched_interval', type=int, default=50,
                   help='调度器检查周期（batch 数）。')
    p.add_argument('--teacher_rot_noise_deg', type=float, default=0.0,
                   help='Teacher 阶段对上一帧 rot6d 注入的最大扰动角度（度）。0 = 不扰动。')
    p.add_argument('--teacher_rot_noise_prob', type=float, default=0.0,
                   help='每帧被注入 rot6d 扰动的概率。')
    p.add_argument('--tf_warmup_steps', type=int, default=5000)
    p.add_argument('--tf_total_steps', type=int, default=200000)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--context_len', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--amp', action='store_true', help='启用自动混合精度 (torch.autocast)')
    p.add_argument('--w_rot_ortho', type=float, default=0.001)
    p.add_argument('--w_rot_delta', type=float, default=1.0)
    p.add_argument('--w_rot_delta_root', type=float, default=0.0)
    p.add_argument('--w_fk_pos', type=float, default=0.0,
                   help='FK 末端位置损失权重（0 表示禁用）。')
    p.add_argument('--w_rot_local', type=float, default=0.0,
                   help='父子关节局部 geodesic 约束权重（0=关闭）。')
    p.add_argument('--w_yaw', type=float, default=0.0,
                   help='Root yaw (水平朝向) geodesic 损失权重（0=关闭）。')
    p.add_argument('--seq_len', type=int, default=120)
    p.add_argument('--yaw_aug_deg', type=float, default=0.0)
    p.add_argument('--normalize_c', action='store_true')
    p.add_argument('--aug_noise_std', type=float, default=0.0)
    p.add_argument('--aug_time_warp_prob', type=float, default=0.0)
    # TensorBoard 相关逻辑已移除，避免冗余参数
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--foot_contact_threshold', type=float, default=1.5, help='角速度阈值（rad/s），低于该值视为脚接触')
    p.add_argument('--monitor_batches', type=int, default=2, help='每个 epoch 在线指标采样的批次数')
    p.add_argument('--force_valfree_eval', action='store_true', default=False,
                   help='即使当前为纯 teacher 阶段，也强制执行一次 freerun 验证并写出 valfree 指标')
    p.add_argument('--eval_horizon', type=int, default=None,
                   help='在线 freerun 验证时的 horizon（帧数）；未指定则遍历整段序列')
    p.add_argument('--eval_warmup', type=int, default=0,
                   help='在线 freerun 验证前的 teacher forcing 帧数（warmup steps）')
    p.add_argument('--yaw_forward_axis', type=int, default=None, help='若提供，则覆盖数据推断的根骨前向轴(0/1/2)')
    p.add_argument('--yaw_forward_offset', type=float, default=None, help='额外指定 yaw 前向轴偏移（单位：度，优先于数据推断）')
    p.add_argument('--eval_angvel_dir_percentile', type=float, default=0.75, help='KeyBone 角速度方向指标仅统计大于该分位数的帧 (0~1)')
    p.add_argument('--diag_input_stats', action='store_true', help='启用输入特征统计（Teacher vs Free-run）')
    p.add_argument('--freerun_debug_path', type=str, default=None, help='若提供，则将首个 freerun batch 的诊断数据保存至该路径')
    p.add_argument('--no_grad_conn_test', action='store_true', help='跳过训练前的梯度连通性自检')

    required_actions = []
    for action in p._actions:
        if getattr(action, 'required', False):
            required_actions.append(action)
            action.required = False

    config_defaults = _load_config_defaults(config_args.config_json, p)
    namespace = argparse.Namespace(**config_defaults)
    namespace.config_json = config_args.config_json
    GLOBAL_ARGS = p.parse_args(remaining_argv, namespace=namespace)
    set_global_args(GLOBAL_ARGS)
    _apply_config_overrides(GLOBAL_ARGS, getattr(GLOBAL_ARGS, 'config_override', None), p)
    GLOBAL_ARGS.config_override = None

    missing_required = [act for act in required_actions if getattr(GLOBAL_ARGS, act.dest, None) is None]
    if missing_required:
        missing_opts = []
        for act in missing_required:
            if act.option_strings:
                missing_opts.append(act.option_strings[-1])
            else:
                missing_opts.append(act.dest)
        p.error(f"missing required arguments: {', '.join(missing_opts)}")

    train_paths = expand_paths_from_specs(_arg('train_files', ''))
    if not train_paths:
        if GLOBAL_ARGS.data and os.path.isdir(GLOBAL_ARGS.data):
            train_paths = sorted(glob.glob(os.path.join(GLOBAL_ARGS.data, '*.npz')))
        else:
            raise FileNotFoundError('No training files. Provide --train_files or --data with .npz')
    run_name = _arg('run_name') or time.strftime('%Y%m%d-%H%M%S')
    out_dir = Path(_arg('out', './runs')).expanduser() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def _load_json(path_str: str, label: str):
        if not path_str:
            return None
        path = Path(path_str).expanduser()
        if not path.is_file():
            print(f"[Spec][WARN] {label} not found at {path}")
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[Spec] Loaded {label}: {path}")
            return data
        except Exception as err:
            print(f"[Spec][WARN] failed to read {label} {path}: {err}")
            return None

    norm_template_arg = _arg('norm_template')
    norm_template_path = Path(norm_template_arg).expanduser() if norm_template_arg else None
    norm_spec = _load_json(norm_template_arg, 'norm_template')
    if norm_spec is None:
        raise SystemExit(f"[FATAL] norm_template 缺失或无效，请确认路径：{norm_template_path}")
    pretrain_template_arg = _arg('pretrain_template')
    pretrain_spec = _load_json(pretrain_template_arg, 'pretrain_template')
    if pretrain_spec is not None:
        for key in ('MuAngVel', 'StdAngVel', 'tanh_scales_angvel', 'pose_hist_len', 'tanh_scales_pose_hist', 'MuPoseHist', 'StdPoseHist'):
            if key in pretrain_spec and pretrain_spec[key] is not None:
                norm_spec[key] = pretrain_spec[key]
    pose_hist_len = 0
    if norm_spec is not None:
        try:
            pose_hist_len = int(norm_spec.get('pose_hist_len', 0) or 0)
        except Exception:
            pose_hist_len = 0

    ds_train = MotionEventDataset(
        GLOBAL_ARGS.data,
        seq_len=GLOBAL_ARGS.seq_len,
        paths=train_paths,
        pose_hist_len=pose_hist_len,
        norm_spec=norm_spec,
    )
    ds_train = _maybe_optimize_dataset_index(ds_train, GLOBAL_ARGS)
    ds_train.is_train = True
    ds_train.yaw_aug_deg = float(_arg('yaw_aug_deg', 0.0))
    ds_train.normalize_c = bool(_arg('normalize_c', False))
    if not hasattr(ds_train, 'state_layout'):
        ds_train.state_layout = getattr(ds_train, 'state_layout', {}) or {}
    pin = device.type == 'cuda'
    lkw = dict(num_workers=_arg('num_workers', 0), pin_memory=pin, persistent_workers=_arg('num_workers', 0) > 0, **{'prefetch_factor': 2} if _arg('num_workers', 0) > 0 else {})
    lkw['collate_fn'] = make_fixedlen_collate(_arg('seq_len', 120))
    train_loader = DataLoader(ds_train, batch_size=_arg('batch', 32), shuffle=True, drop_last=True, **lkw)
    Dx, Dy, Dc = (int(ds_train.Dx), int(ds_train.Dy), int(ds_train.Dc))
    L = int(_arg('depth', 2))
    H = int(_arg('width', 512))
    K = int(_arg('context_len', 16))
    print(f'[Export][Dims] Dx={Dx}, Dy={Dy}, Dc={Dc} | L={L}, H={H}, K={K}')

    pose_hist_dim_raw = int(getattr(ds_train, 'pose_hist_dim', 0) or 0)
    pose_hist_len_raw = int(getattr(ds_train, 'pose_hist_len', 0) or 0)
    history_export_frames = int(_arg('history_adaptive_export_frames', 0) or 0)
    history_frame_dim = (
        pose_hist_dim_raw // pose_hist_len_raw
        if pose_hist_len_raw > 0 and pose_hist_dim_raw % pose_hist_len_raw == 0
        else 0
    )
    pose_hist_dim_model = pose_hist_dim_raw
    if history_export_frames > 0 and history_frame_dim > 0:
        pose_hist_dim_model = history_export_frames * history_frame_dim

    model = EventMotionModel(
        in_state_dim=ds_train.Dx,
        out_motion_dim=ds_train.Dy,
        cond_dim=ds_train.Dc,
        period_dim=getattr(ds_train, 'period_dim', 0),
        hidden_dim=_arg('width', 512),
        num_layers=_arg('depth', 2),
        num_heads=_arg('num_heads', 4),
        dropout=_arg('dropout', 0.1),
        context_len=_arg('context_len', 16),
        contact_dim=getattr(ds_train, 'contact_dim', 0),
        angvel_dim=getattr(ds_train, 'angvel_dim', 0),
        pose_hist_dim=pose_hist_dim_model,
    ).to(device)
    if history_export_frames > 0:
        if pose_hist_dim_raw <= 0 or pose_hist_len_raw <= 0:
            print("[AdaptiveHistory][WARN] pose history not available; adaptive history disabled.")
        elif pose_hist_dim_raw % pose_hist_len_raw != 0:
            print("[AdaptiveHistory][WARN] pose history dim不整除帧数，跳过 adaptive history。")
        else:
            max_frames = _arg('history_adaptive_max_frames', None)
            if max_frames is None:
                max_frames = pose_hist_len_raw
            try:
                from .history import AdaptiveHistoryModule
            except ImportError:  # pragma: no cover
                from history import AdaptiveHistoryModule

            module_device = torch.device('cpu') if device.type == 'mps' else device
            history_module = AdaptiveHistoryModule(
                pose_dim=history_frame_dim,
                hidden_dim=int(_arg('history_adaptive_hidden', H)),
                num_history_frames=history_export_frames,
                max_history_frames=int(max_frames),
                cond_dim=0,
                num_heads=int(_arg('history_adaptive_heads', 2) or 2),
                train_variable_history=bool(_arg('history_adaptive_train_variable', False)),
            ).to(module_device)
            model.enable_adaptive_history(history_module, pose_hist_len=pose_hist_len_raw)

    validate_and_fix_model_(model, Dx, Dc)
    validate_and_fix_model_(model)

    # Attach frozen MotionEncoder (optional, used for soft period hints)
    encoder_path_cfg = _arg('encoder_path', '')
    resolved_bundle = None
    if encoder_path_cfg:
        base_candidate = Path(encoder_path_cfg).expanduser()
        search_roots = [
            base_candidate,
            Path(__file__).resolve().parent / base_candidate,
            Path(__file__).resolve().parent.parent / base_candidate,
        ]
        data_root = Path(GLOBAL_ARGS.data).expanduser() if getattr(GLOBAL_ARGS, 'data', None) else None
        if data_root is not None:
            search_roots.append(data_root.parent / base_candidate)
        for cand in search_roots:
            if cand is not None and cand.is_file():
                resolved_bundle = cand
                break
        if resolved_bundle is None:
            print(f"[MPL][WARN] MotionEncoder bundle not found (tried {encoder_path_cfg})")
        else:
            try:
                bundle = torch.load(str(resolved_bundle), map_location='cpu')
                model.attach_motion_encoder(bundle)
                print(f"[MPL] Attached MotionEncoder bundle: {resolved_bundle}")
                try:
                    ds_train.period_dim = getattr(model, 'period_dim', getattr(ds_train, 'period_dim', 0))
                except Exception:
                    pass
            except Exception as err:
                resolved_bundle = None
                print(f"[MPL][WARN] failed to attach MotionEncoder bundle: {err}")

    with torch.no_grad():
        _l0 = model.shared_encoder[0]
        if not torch.isfinite(_l0.weight).all() or (_l0.bias is not None and (not torch.isfinite(_l0.bias).all())):
            print('[Guard] first-linear became non-finite post-sanitize, reinitializing')
            torch.nn.init.kaiming_uniform_(_l0.weight, a=math.sqrt(5))
            if _l0.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(_l0.weight)
                bound = 1.0 / math.sqrt(max(fan_in, 1))
                torch.nn.init.uniform_(_l0.bias, -bound, bound)
            assert torch.isfinite(_l0.weight).all() and (_l0.bias is None or torch.isfinite(_l0.bias).all())
    with torch.no_grad():
        lin0 = model.shared_encoder[0]
        assert torch.isfinite(lin0.weight).all() and (lin0.bias is None or torch.isfinite(lin0.bias).all()), '[PostCheck] shared_encoder.0 still not finite'
    try:
        model._pasa_fps = float(getattr(ds_train, 'fps', 60.0))
    except Exception:
        pass
    fps_data = float(getattr(ds_train, 'fps', 60.0) or 60.0)
    w_rot_delta = float(_arg('w_rot_delta', 1.0))
    w_fk_pos = float(_arg('w_fk_pos', 0.0) or 0.0)
    w_rot_local = float(_arg('w_rot_local', 0.0) or 0.0)
    w_yaw = float(_arg('w_yaw', 0.0) or 0.0)

    loss_fn = MotionJointLoss(
        output_layout=ds_train.output_layout,
        fps=fps_data,
        rot6d_spec=getattr(ds_train, 'rot6d_spec', {}),
        w_rot_delta=w_rot_delta,
        w_rot_delta_root=_arg('w_rot_delta_root', 0.0),
        w_rot_ortho=_arg('w_rot_ortho', 0.001),
        meta=None,
        w_fk_pos=w_fk_pos,
        w_rot_local=w_rot_local,
        w_yaw=w_yaw,
    )
    if getattr(ds_train, 'bone_names', None):
        try:
            loss_fn.set_bone_names(ds_train.bone_names)
        except Exception:
            pass
    if getattr(ds_train, 'parents', None):
        try:
            loss_fn.set_skeleton(ds_train.parents, getattr(ds_train, 'bone_offsets', None))
        except Exception as exc:
            print(f"[Loss][WARN] set_skeleton failed: {exc}")
    bundle_json_arg = _arg('bundle_json')
    bundle_json_path = str(Path(bundle_json_arg).expanduser()) if bundle_json_arg else None
    loss_fn.template_hint = str(norm_template_path) if norm_template_path else None
    loss_fn.bundle_hint = bundle_json_path

    print(
        f"[LossWeights] "
        f"w_rot_delta={loss_fn.w_rot_delta} "
        f"w_rot_delta_root={loss_fn.w_rot_delta_root} "
        f"w_rot_ortho={loss_fn.w_rot_ortho} "
        f"w_fk_pos={loss_fn.w_fk_pos} "
        f"w_rot_local={loss_fn.w_rot_local} "
        f"w_yaw={loss_fn.w_yaw}"
    )

    loss_fn.dt_traj = 1.0 / max(1e-6, fps_data)
    loss_fn.dt_bone = 1.0 / max(1e-6, fps_data)
    print(f"[Dt] dt_traj={loss_fn.dt_traj:.6f}s | dt_bone={loss_fn.dt_bone:.6f}s (dataset fps={fps_data})")

    if hasattr(loss_fn, 'rot6d_eps'):
        loss_fn.rot6d_eps = 1e-6
    augmentor = MotionAugmentation(noise_std=_arg('aug_noise_std', 0.0), time_warp_prob=_arg('aug_time_warp_prob', 0.0))
    trainer = Trainer(model=model, loss_fn=loss_fn, lr=_arg('lr', 0.0001), grad_clip=_arg('grad_clip', 0.0), weight_decay=_arg('weight_decay', 0.01), tf_warmup_steps=_arg('tf_warmup_steps', 5000), tf_total_steps=_arg('tf_total_steps', 200000), augmentor=augmentor, use_amp=_arg('amp', False), accum_steps=_arg('accum_steps', 1), pin_memory=pin, args=GLOBAL_ARGS)
    trainer._norm_template_path = str(norm_template_path) if norm_template_path else None
    trainer._bundle_json_path = bundle_json_path
    trainer.out_dir = str(out_dir)
    __apply_layout_center(ds_train, trainer)
    trainer.pose_hist_len = int(getattr(ds_train, 'pose_hist_len', 0) or 0)
    trainer.pose_hist_dim = int(getattr(ds_train, 'pose_hist_dim', 0) or 0)
    _pose_norm = getattr(ds_train, 'pose_hist_norm', None)
    if _pose_norm is not None:
        trainer.pose_hist_scales = torch.as_tensor(_pose_norm.scales, dtype=torch.float32)
        trainer.pose_hist_mu = torch.as_tensor(_pose_norm.mu, dtype=torch.float32) if getattr(_pose_norm, 'mu', None) is not None else None
        trainer.pose_hist_std = torch.as_tensor(_pose_norm.std, dtype=torch.float32) if getattr(_pose_norm, 'std', None) is not None else None
    else:
        trainer.pose_hist_scales = None
        trainer.pose_hist_mu = None
        trainer.pose_hist_std = None
    loss_fn.mu_y = getattr(trainer, "mu_y", None)
    loss_fn.std_y = getattr(trainer, "std_y", None)
    if getattr(trainer, '_bundle_meta', None):
        try:
            loss_fn.meta = dict(trainer._bundle_meta)
        except Exception:
            pass

    trainer.foot_contact_threshold = float(_arg('foot_contact_threshold'))
    # 一次性归一化数值诊断
    _norm_debug_once(trainer, train_loader, thr=float(_arg('diag_thr')), topk=int(_arg('diag_topk')), print_to_console=False)
    trainer.bone_hz = fps_data


    safe_set_slice(trainer, 'yaw_x_slice', parse_layout_entry(trainer._x_layout.get('RootYaw'), 'RootYaw'))
    safe_set_slice(trainer, 'rootvel_x_slice', parse_layout_entry(trainer._x_layout.get('RootVelocity'), 'RootVelocity'))
    safe_set_slice(trainer, 'angvel_x_slice', parse_layout_entry(trainer._x_layout.get('BoneAngularVelocities'), 'BoneAngularVelocities'))

    # 诊断参数（也可用命令行 --diag_topk/--diag_thr 覆盖）
    trainer.diag_topk = int(_arg('diag_topk', 8) or 8)
    trainer.diag_thr = float(_arg('diag_thr', 8.0) or 8.0)
    import math as _math_local
    forward_axis_override = _arg('yaw_forward_axis', None)
    if forward_axis_override is not None:
        trainer.yaw_forward_axis = int(forward_axis_override)
    elif getattr(ds_train, 'forward_axis', None) is not None:
        trainer.yaw_forward_axis = int(ds_train.forward_axis)
    else:
        trainer.yaw_forward_axis = int(getattr(trainer, 'yaw_forward_axis', 2))
    offset_override = _arg('yaw_forward_offset', None)
    if offset_override is not None:
        trainer.yaw_forward_axis_offset = float(_math_local.radians(float(offset_override)))
    else:
        trainer.yaw_forward_axis_offset = float(getattr(ds_train, 'forward_axis_offset', 0.0) or 0.0)
    trainer.eval_angvel_dir_percentile = float(_arg('eval_angvel_dir_percentile'))
    trainer.diag_input_stats = bool(_arg('diag_input_stats'))

    # === validation/monitor switches ===
    trainer.val_mode = _arg('val_mode', 'online')
    trainer.no_monitor = bool(_arg('no_monitor', False))
    trainer.monitor_batches = int(_arg('monitor_batches', 8) or 8)
    trainer.force_valfree_eval = bool(_arg('force_valfree_eval', False))
    trainer.eval_settings = FreeRunSettings(
        warmup_steps=int(_arg('eval_warmup', 0) or 0),
        horizon=_arg('eval_horizon', None),
        max_batches=trainer.monitor_batches,
    )
    trainer.tf_mode = _arg('tf_mode', 'epoch_linear')
    trainer.tf_warmup_epochs = _arg('tf_warmup_epochs', 3)
    trainer.tf_start_epoch = _arg('tf_start_epoch', 0)
    trainer.tf_end_epoch = _arg('tf_end_epoch', 10)
    trainer.tf_max = _arg('tf_max', 1.0)
    trainer.tf_min = _arg('tf_min', 0.1)
    trainer.mixed_state_mode = str(_arg('mixed_state_mode', 'full')).lower()
    trainer.mix_full_state = trainer.mixed_state_mode != 'rot6d'
    trainer.freerun_horizon = int(_arg('freerun_horizon', 0) or 0)
    trainer.freerun_weight = float(_arg('freerun_weight', 0.1))
    trainer.freerun_horizon_min = int(_arg('freerun_horizon_min', 6) or 6)
    trainer.freerun_debug_steps = int(_arg('freerun_debug_steps', 0) or 0)
    trainer.history_debug_steps = int(_arg('history_debug_steps', 0) or 0)
    trainer.freerun_stage_schedule = _parse_stage_schedule(_arg('freerun_stage_schedule', None))
    _init_h_arg = _arg('freerun_init_horizon', None)
    if _init_h_arg is None:
        if trainer.freerun_horizon > 0:
            heur_h = max(
                trainer.freerun_horizon_min,
                min(
                    trainer.freerun_horizon,
                    max(
                        trainer.freerun_horizon_min,
                        int(round(max(trainer.freerun_horizon * 0.7, trainer.freerun_horizon_min)))
                    ),
                ),
            )
        else:
            heur_h = trainer.freerun_horizon_min
        trainer.freerun_init_horizon = int(heur_h)
    else:
        trainer.freerun_init_horizon = int(_init_h_arg)
    _weight_init_arg = _arg('freerun_weight_init', None)
    if _weight_init_arg is None:
        inferred = trainer.freerun_weight * 0.2
        trainer.freerun_weight_init = float(max(0.0, min(trainer.freerun_weight, inferred)))
    else:
        trainer.freerun_weight_init = float(_weight_init_arg)
    if trainer.freerun_weight <= 0.0 or trainer.freerun_weight_init > trainer.freerun_weight:
        trainer.freerun_weight_init = max(0.0, min(trainer.freerun_weight, trainer.freerun_weight_init))
    trainer.freerun_horizon_ramp_epochs = int(_arg('freerun_horizon_ramp_epochs', 5) or 5)
    trainer.freerun_weight_mode = str(_arg('freerun_weight_mode', 'epoch_linear') or 'epoch_linear').lower()
    trainer.freerun_weight_ramp_epochs = int(_arg('freerun_weight_ramp_epochs', 5) or 5)
    trainer.freerun_grad_log = bool(_arg('freerun_grad_log', False))
    trainer.freerun_grad_log_interval = int(_arg('freerun_grad_log_interval', 50) or 50)
    trainer.freerun_grad_ratio_alert = float(_arg('freerun_grad_ratio_alert', 0.01) or 0.01)
    adaptive_loss_method = str(_arg('adaptive_loss_method', 'none') or 'none').lower()
    adaptive_loss_terms = [
        term.strip()
        for term in str(_arg('adaptive_loss_terms', 'fk_pos,rot_local,rot_delta,rot_delta_root,rot_ortho') or '').split(',')
        if term.strip()
    ]
    if not adaptive_loss_terms:
        adaptive_loss_terms = None  # 运行时自动根据 loss payload 决定
    sched_params = None
    if _arg('adaptive_scheduler', False):
        sched_params = dict(
            freerun_horizon=trainer.freerun_horizon,
            freerun_min=trainer.freerun_horizon_min,
            freerun_max=int(_arg('freerun_horizon_max', trainer.freerun_horizon or 0) or max(trainer.freerun_horizon or 0, 0)),
            teacher_forcing_ratio=float(_arg('tf_max', 1.0)),
            loss_spike_threshold=float(_arg('adaptive_sched_loss_spike', 1.5)),
            convergence_threshold=float(_arg('adaptive_sched_convergence', 0.02)),
            adjustment_rate=float(_arg('adaptive_sched_adjustment', 0.1)),
            check_interval=int(_arg('adaptive_sched_interval', 50) or 50),
        )
    adaptive_manager = AdaptiveLossManager(
        adaptive_loss_terms,
        adaptive_loss_method,
        loss_alpha=float(_arg('adaptive_loss_alpha', 1.5)),
        loss_temperature=float(_arg('adaptive_loss_temperature', 2.0)),
        scheduler_params=sched_params,
    )
    trainer.adaptive_loss_module = adaptive_manager.loss_module
    trainer.hyperparam_scheduler = adaptive_manager.scheduler
    trainer.teacher_rot_noise_deg = float(_arg('teacher_rot_noise_deg', 0.0))
    trainer.teacher_rot_noise_prob = float(_arg('teacher_rot_noise_prob', 0.0))
    if _arg('adaptive_scheduler', False):
        scheduler_init = {
            'freerun_horizon': int(trainer.freerun_horizon or trainer.freerun_init_horizon),
            'freerun_min': int(trainer.freerun_horizon_min),
            'freerun_max': int(max(trainer.freerun_horizon, trainer.freerun_init_horizon, trainer.freerun_horizon_min)),
            'teacher_forcing_ratio': float(_arg('tf_max', 1.0)),
        }
        trainer.hyperparam_scheduler = AdaptiveHyperparamScheduler(
            scheduler_init,
            loss_spike_threshold=float(_arg('adaptive_sched_loss_spike', 1.5)),
            convergence_threshold=float(_arg('adaptive_sched_convergence', 0.02)),
            adjustment_rate=float(_arg('adaptive_sched_adjustment', 0.1)),
            check_interval=int(_arg('adaptive_sched_interval', 50) or 50),
        )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, _arg('epochs', 300) * steps_per_epoch)
    effective_warmup = min(_arg('warmup_steps', 1000), int(total_steps * 0.1))
    base_lr = float(_arg('lr', 0.0001))
    min_lr = base_lr * float(max(1e-06, _arg('min_lr_ratio', 0.05)))

    def lr_lambda(step):
        # 关键：避免构造时把 LR 压到 1e-8 * base_lr
        if step <= 0:
            return 1.0
        if step < effective_warmup:
            return step / float(max(1, effective_warmup))
        t = (step - effective_warmup) / float(max(1, total_steps - effective_warmup))
        t = min(1.0, max(0.0, t))
        return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * 0.5 * (1.0 + math.cos(math.pi * t))


    lam0 = (max(1e-08, 0.0 / float(max(1, effective_warmup))) if effective_warmup > 0 else 1.0)

    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=lr_lambda)
    trainer.freerun_debug_path = _arg('freerun_debug_path', None)
    trainer.enable_grad_connection_test = not bool(_arg('no_grad_conn_test', False))
    trainer._current_run_name = run_name

    best_ckpt, history = trainer.fit(
        train_loader,
        epochs=_arg('epochs', 300),
        log_every=_arg('log_every', 50),
        out_dir=str(out_dir),
        patience=_arg('patience', 20),
        run_name=run_name
    )

    try:

        vloader = train_loader
        _mon_batches = int(_arg('monitor_batches', 8) or 8)
        _metrics = trainer.validate_autoreg_online(vloader, max_batches=_mon_batches)
        print(f"[ValFree@last] MSEnormY={_metrics['MSEnormY']:.6f} "
              f"GeoDeg={_metrics['GeoDeg']:.3f} "
              f"YawAbsDeg={_metrics['YawAbsDeg']:.3f} "
              f"RootVelMAE={_metrics['RootVelMAE']:.5f}"
              f"AngVelMAE={_metrics.get('AngVelMAE', float('nan')):.5f} rad/s")
    except Exception as _e:
        print(f"[ValFree] skipped due to error: {_e}")
        import traceback
        traceback.print_exc()
        # 可选：若有 best_ckpt 就加载（保持你原有逻辑）
        try:
            if best_ckpt:
                ckpt_path = Path(best_ckpt).expanduser()
                if ckpt_path.is_file():
                    ckpt = torch.load(str(ckpt_path), map_location=device)
                    missing, unexpected = model.load_state_dict(ckpt['model'], strict=True)
                    assert not missing and (
                        not unexpected), f'state_dict mismatch: missing={missing}, unexpected={unexpected}'
                    print(f'[Load] best checkpoint loaded: {ckpt_path}')
                else:
                    print(f'[WARN] checkpoint not found: {ckpt_path}')
        except Exception as __e:
            print(f'[Load][WARN] failed to load best ckpt: {__e}')
    finally:
        # === 关键改动：无论上面成功/失败，这里都尝试导出 ONNX ===
        print('[Export][ENTER] preparing to export ONNX...')
        try:
            import os, traceback

            model_to_export = model.eval().cpu()

            onnx_path = os.path.join(str(out_dir), f'{run_name}_step_stateful_nophase.onnx')

            # 维度探测仅用于日志 & sanity；失败也不中断真正导出
            try:
                _b = next(iter(train_loader))
                Dx = int(_b['motion'].shape[-1])
                Dy = int(_b['gt_motion'].shape[-1])
                Dc = int(_b['cond_in'].shape[-1]) if 'cond_in' in _b else 0
                print(f'[Export][ProbeDims] Dx={Dx} Dy={Dy} Dc={Dc}')
                try:
                    sanity_check_model_dims(model_to_export, Dx, Dy, Dc)
                    print('[Export][Sanity] input dims check OK')
                except Exception as ee:
                    print('[Export][Sanity][WARN]', ee)
            except Exception as ee:
                print('[Export][ProbeDims][WARN] cannot read a batch for dim probe:', ee)

            os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
            export_onnx_step_stateful_nophase(
                model_to_export,
                train_loader,
                onnx_path,
                opset=18,
                dynamic_batch=False,
            )
        except Exception as e:
            print('[Export][ERROR]', e)
            traceback.print_exc()
@torch.no_grad()
def export_onnx_step_stateful_nophase(model: torch.nn.Module, loader, onnx_path: str, opset: int = 18, dynamic_batch: bool = False):
    """
    单步（无隐式状态）ONNX 导出：
      输入:  state[B,Dx], cond[B,Dc], contacts[B,C], angvel[B,A], pose_hist[B,P]
      输出:  motion_pred[B,Dy]

    训练与推理均使用显式历史缓冲，对应 UE 中的 PoseHistoryBuffer。
    """
    import os, torch

    if loader is None:
        raise ValueError('loader is None；需要 DataLoader 来获取示例形状。')

    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)):
        if batch and isinstance(batch[0], dict):
            batch = batch[0]
        else:
            tmp = {}
            if len(batch) > 0 and isinstance(batch[0], torch.Tensor):
                tmp['motion'] = batch[0]
            if len(batch) > 1 and isinstance(batch[1], torch.Tensor):
                tmp['gt_motion'] = batch[1]
            if len(batch) > 2 and isinstance(batch[2], torch.Tensor):
                tmp['cond_in'] = batch[2]
            batch = tmp
    if not isinstance(batch, dict):
        raise TypeError("DataLoader 必须返回 dict 才能导出 ONNX。")

    def _pick(*keys):
        for k in keys:
            v = batch.get(k)
            if v is not None:
                return v
        return None

    state_seq = _pick('motion', 'X', 'x_in_features')
    if state_seq is None:
        raise KeyError("Batch 缺少输入 X：需要 'motion' 或同义键。")

    try:
        shape_dbg = {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, 'shape')}
        print('[Export][BatchShapes]', shape_dbg)
    except Exception:
        pass

    cond_seq = _pick('cond_in', 'C', 'conditions')
    contacts_seq = _pick('contacts', 'soft_contact', 'contacts_in')
    angvel_seq = _pick('angvel', 'angular_velocity', 'angvel_in')
    pose_hist_seq = _pick('pose_hist', 'pose_history')

    state_seq = state_seq.to(torch.float32)
    if state_seq.dim() == 3:
        _, _, Dx = state_seq.shape
        state0 = state_seq[:1, 0, :].contiguous()
    elif state_seq.dim() == 2:
        Dx = state_seq.shape[-1]
        state0 = state_seq[:1, :].contiguous()
    else:
        raise ValueError(f'Unexpected X shape: {tuple(state_seq.shape)}')

    def _frame_or_zero(tensor, dim, dtype):
        if tensor is None:
            return torch.zeros((1, dim), dtype=dtype) if dim > 0 else torch.zeros((1, 0), dtype=dtype)
        tensor = tensor.to(dtype)
        if tensor.dim() == 3:
            return tensor[:1, 0, :dim].contiguous()
        if tensor.dim() == 2:
            return tensor[:1, :dim].contiguous()
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)[:, :dim].contiguous()
        raise ValueError(f'Unexpected tensor shape: {tuple(tensor.shape)}')

    cond_dim = int(getattr(model, 'cond_dim', cond_seq.shape[-1] if isinstance(cond_seq, torch.Tensor) else 0))
    contact_dim = int(getattr(model, 'contact_dim', contacts_seq.shape[-1] if isinstance(contacts_seq, torch.Tensor) else 0))
    angvel_dim = int(getattr(model, 'angvel_dim', angvel_seq.shape[-1] if isinstance(angvel_seq, torch.Tensor) else 0))
    pose_hist_dim = int(getattr(model, 'pose_hist_dim', pose_hist_seq.shape[-1] if isinstance(pose_hist_seq, torch.Tensor) else 0))

    cond0 = _frame_or_zero(cond_seq, cond_dim, torch.float32)
    contacts0 = _frame_or_zero(contacts_seq, contact_dim, torch.float32)
    angvel0 = _frame_or_zero(angvel_seq, angvel_dim, torch.float32)
    pose_hist0 = _frame_or_zero(pose_hist_seq, pose_hist_dim, torch.float32)

    device = torch.device('cpu')
    model = model.to(device).eval()

    class _StatelessWrapper(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, state, cond, contacts, angvel, pose_hist):
            cond_in = cond if cond.shape[-1] > 0 else None
            contacts_in = contacts if contacts.shape[-1] > 0 else None
            angvel_in = angvel if angvel.shape[-1] > 0 else None
            pose_hist_in = pose_hist if pose_hist.shape[-1] > 0 else None
            out = self.core(
                state,
                cond_in,
                contacts=contacts_in,
                angvel=angvel_in,
                pose_history=pose_hist_in,
            )
            if isinstance(out, dict):
                pred = out.get('out')
                if pred is None:
                    raise RuntimeError("Model dict output missing 'out'.")
                return pred
            return out

    wrapper = _StatelessWrapper(model).cpu().eval()
    sample_out = wrapper(state0, cond0, contacts0, angvel0, pose_hist0)
    Dy = int(sample_out.shape[-1])

    inputs = (state0, cond0, contacts0, angvel0, pose_hist0)
    input_names = ['state', 'cond', 'contacts', 'angvel', 'pose_hist']
    output_names = ['motion_pred']
    dynamic_axes = {name: {0: 'B'} for name in input_names + output_names} if dynamic_batch else None

    os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        f=onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f'[Export][OK] saved: {onnx_path} | Dx={Dx} Dy={Dy} Dc={cond_dim} C={contact_dim} A={angvel_dim} P={pose_hist_dim}')

def main():
    """
    包装器主函数：
    1. 首先执行预分析并生成 bundle.json。
    2. 然后调用真正的训练函数 train_entry()。
    """
    argv0 = sys.argv[:]
    is_export_only = '--arpg_export_only' in argv0
    if is_export_only:
        rest_argv = [arg for arg in argv0 if arg != '--arpg_export_only']
    else:
        rest_argv = argv0
    out_dir_arg = get_flag_value_from_argv(rest_argv, '--out') or get_flag_value_from_argv(rest_argv, '-o')
    run_name_arg = get_flag_value_from_argv(rest_argv, '--run_name')
    out_dir = out_dir_arg or './runs'
    run_name = run_name_arg or time.strftime('%Y%m%d-%H%M%S')
    train_files_flag = get_flag_values_from_argv(rest_argv, '--train_files')
    data_dir_flag = get_flag_value_from_argv(rest_argv, '--data')
    train_files = expand_paths_from_specs(train_files_flag)
    if not train_files and data_dir_flag and os.path.isdir(os.path.expanduser(data_dir_flag)):
        train_files = expand_paths_from_specs([data_dir_flag])


    from types import SimpleNamespace
    global GLOBAL_ARGS
    GLOBAL_ARGS = SimpleNamespace(out=out_dir, run_name=run_name, allow_val_on_train='--allow_val_on_train' in rest_argv, val_ratio=float(get_flag_value_from_argv(rest_argv, '--val_ratio') or 0))
    set_global_args(GLOBAL_ARGS)
    print(f'[ARPG-PATCH] 参数准备完毕，即将进入训练入口: train_entry()')
    sys.argv = rest_argv
    try:
        train_entry()
    finally:
        sys.argv = argv0
if __name__ == '__main__':
    main()
