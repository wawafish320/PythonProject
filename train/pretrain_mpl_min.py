
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练面向软脚接触的 MotionEncoder（方案 A，逐帧 MLP 编码，仅读取 Y 通道）。
归一化策略沿用主训练的逻辑（tanh + 分位数尺度），但完全不读取 X 侧统计量，仅消费
norm_template.json 中与角速度、姿态历史相关的字段。

小提示：幅值的「绝对刻度」建议放到主训练 / UE 侧再加一个可训练的一阶标定层（Affine1D）来修正：
直接将预训练得到的 `amp_pred` 经过 `γ·amp_pred + β` 的轻量线性调整即可。
这样既能保持当前模型已经学到的排序/节律信息，又避免一次性 OLS 校准可能出现的符号反转。

- 输入： y_out_features（Rot6D） → reproject → rot6d_to_matrix → angvel_vec_from_R_seq(dt=1/FPS)；
        同时拼接软接触分数与最近若干帧 pose_history，一起喂给逐帧 MLP 编码器。
- 归一化： 必须使用 tanh(x / tanh_scales)，若模板提供 μ/σ 则额外做 z-score。
- 目标：  重建当前帧姿态/角速度 & 软接触 BCE，并对齐 sin/cos 软周期提示。
- 训练：  软周期 latent 由 period_head 输出，解码器负责重建，loss 组合见脚本参数。
- 保存：  导出 MotionEncoder + period/contact 头 + pose/ang 解码器与元数据，供主训练直接加载。

脚本属于严格模式：缺失关键键或路径会直接抛出错误。
"""

import os, glob, json, random, contextlib
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- exact imports from your project (no fallback) ----
from train.geometry import reproject_rot6d, rot6d_to_matrix, angvel_vec_from_R_seq  # noqa: E402
from train.utils import build_mlp  # noqa: E402
from train.models import MotionEncoder, PeriodHead  # noqa: E402
from train.normalizers import (
    VectorTanhNormalizer,
    AngvelNormalizer,
    _make_angnorm_from_spec,
)  # noqa: E402
from train.io import npz_scalar_to_str, load_soft_contacts_from_json as _load_soft_contacts_from_json  # noqa: E402


# ----------------------------- small utils -----------------------------
def _get_fps_from_npz_or_json(z, json_path: Optional[str]) -> int:
    if "FPS" in z:
        fps = int(np.array(z["FPS"]).item())
        if fps > 0:
            return fps
    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fps = int(data.get("FPS", 0))
        if fps > 0:
            return fps
    raise RuntimeError(f"FPS not found in npz nor JSON (json={json_path})")


# --------------------------- Dataset ----------------------------



    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        逆 transform：
          if zscore: X = X*std + mu
          W_raw = atanh(X) * s_eff
        说明：假设 transform 的最后一步是 tanh 压缩。
        """
        assert (
                X.ndim == 2 and X.shape[1] == self.s_eff.size
        ), f"X shape {tuple(X.shape)} not compatible with J*3={self.s_eff.size}."

        Y = X.astype(np.float32)
        # 可选 z-score 还原
        if getattr(self, "mu", None) is not None and getattr(self, "std", None) is not None:
            Y = Y * self.std + self.mu
        # 数值安全：tanh 值域 (-1,1)
        Y = np.clip(Y, -0.999999, 0.999999)
        W_raw = np.arctanh(Y) * self.s_eff
        return W_raw.astype(np.float32)



def _build_angvel_norm_spec(in_glob: str, save_path: str, pose_hist_len: int = 3) -> dict:
    """
    扫描 Rot6D -> 角速度 -> 在 tanh(w/s) 域统计 μ/σ；
    - 每 clip 内 2.5/97.5 分位，clip 间“中位数”聚合 -> s_eff
    - 缺关键键/维度/样本，直接抛错（零回退）
    - 返回内存 dict；并将同样内容写入 save_path（供主训练复用）
    """
    import numpy as np, json, glob, os, torch
    scale_floor = 1e-3
    pose_hist_len = max(0, int(pose_hist_len))
    files = sorted(glob.glob(in_glob))
    if not files:
        raise RuntimeError(f"No npz matched: {in_glob}")
    with np.load(files[0], allow_pickle=True) as z0:
        if "y_out_features" not in z0:
            raise RuntimeError(f"{os.path.basename(files[0])} missing y_out_features")
        Y0 = np.asarray(z0["y_out_features"], dtype=np.float32)
        Dy = int(Y0.shape[1])
        if Dy % 6 != 0:
            raise RuntimeError(f"y_out_features dim {Dy} is not multiple of 6")
        J = Dy // 6

    lo_list, hi_list, W_all = [], [], []
    pose_lo_list, pose_hi_list, pose_all = [], [], []
    pose_cur_lo, pose_cur_hi, pose_cur_all = [], [], []
    for p in files:
        with np.load(p, allow_pickle=True) as z:
            if "y_out_features" not in z:
                raise RuntimeError(f"{os.path.basename(p)} missing y_out_features")
            Y = np.asarray(z["y_out_features"], dtype=np.float32)
            json_path = npz_scalar_to_str(z["source_json"]) if "source_json" in z else None
            fps = _get_fps_from_npz_or_json(z, json_path)
        T = int(Y.shape[0])
        if T < 2:
            raise RuntimeError(f"{os.path.basename(p)} too short: T={T}")
        y_t = torch.from_numpy(Y).to(torch.float32)
        y_t = reproject_rot6d(y_t.unsqueeze(0))[0]
        R = rot6d_to_matrix(y_t.view(1, T, J, 6))[0]
        w = angvel_vec_from_R_seq(R.unsqueeze(0), fps)[0].reshape(-1, J*3).cpu().numpy().astype(np.float32)
        lo_list.append(np.percentile(w, 2.5, axis=0))
        hi_list.append(np.percentile(w, 97.5, axis=0))
        W_all.append(w)

        pose_seq = y_t.cpu().numpy().astype(np.float32)
        if pose_seq.shape[1] != J * 6:
            raise RuntimeError(f"{os.path.basename(p)}: pose dim {pose_seq.shape[1]} != J*6 {J*6}")

        if pose_hist_len > 0:
            hist_vecs = []
            for t in range(1, T):
                frames = []
                for h in range(pose_hist_len, 0, -1):
                    idx = t - h
                    if idx < 0:
                        idx = 0
                    frames.append(pose_seq[idx])
                if frames:
                    hist_vecs.append(np.concatenate(frames, axis=0))
            if hist_vecs:
                hist_arr = np.stack(hist_vecs, axis=0)
                pose_lo_list.append(np.percentile(hist_arr, 2.5, axis=0))
                pose_hi_list.append(np.percentile(hist_arr, 97.5, axis=0))
                pose_all.append(hist_arr)

        pose_curr = pose_seq[1:]
        if pose_curr.size > 0:
            pose_cur_lo.append(np.percentile(pose_curr, 2.5, axis=0))
            pose_cur_hi.append(np.percentile(pose_curr, 97.5, axis=0))
            pose_cur_all.append(pose_curr)

    lo_med = np.median(np.stack(lo_list, axis=0), axis=0)
    hi_med = np.median(np.stack(hi_list, axis=0), axis=0)
    s = np.maximum(np.abs(lo_med), np.abs(hi_med)).astype(np.float32)
    s = np.clip(s, scale_floor, None)

    W = np.concatenate(W_all, axis=0).astype(np.float32)
    X = np.tanh(W / s)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32); sd = np.clip(sd, 1e-6, None)

    pose_hist_dim = pose_hist_len * J * 6
    if pose_all:
        pose_hist_concat = np.concatenate(pose_all, axis=0).astype(np.float32)
        pose_lo_med = np.median(np.stack(pose_lo_list, axis=0), axis=0)
        pose_hi_med = np.median(np.stack(pose_hi_list, axis=0), axis=0)
        s_pose = np.maximum(np.abs(pose_lo_med), np.abs(pose_hi_med)).astype(np.float32)
        s_pose = np.clip(s_pose, scale_floor, None)
        pose_tanh = np.tanh(pose_hist_concat / s_pose)
        pose_mu = pose_tanh.mean(axis=0).astype(np.float32)
        pose_sd = pose_tanh.std(axis=0).astype(np.float32)
        pose_sd = np.clip(pose_sd, 1e-6, None)
    elif pose_hist_dim > 0:
        s_pose = np.ones((pose_hist_dim,), dtype=np.float32)
        pose_mu = np.zeros_like(s_pose)
        pose_sd = np.ones_like(s_pose)
    else:
        s_pose = np.array([], dtype=np.float32)
        pose_mu = np.array([], dtype=np.float32)
        pose_sd = np.array([], dtype=np.float32)

    pose_cur_dim = J * 6
    if pose_cur_all:
        pose_curr_concat = np.concatenate(pose_cur_all, axis=0).astype(np.float32)
        pose_cur_lo_med = np.median(np.stack(pose_cur_lo, axis=0), axis=0)
        pose_cur_hi_med = np.median(np.stack(pose_cur_hi, axis=0), axis=0)
        s_pose_cur = np.maximum(np.abs(pose_cur_lo_med), np.abs(pose_cur_hi_med)).astype(np.float32)
        s_pose_cur = np.clip(s_pose_cur, scale_floor, None)
        pose_cur_tanh = np.tanh(pose_curr_concat / s_pose_cur)
        pose_cur_mu = pose_cur_tanh.mean(axis=0).astype(np.float32)
        pose_cur_sd = pose_cur_tanh.std(axis=0).astype(np.float32)
        pose_cur_sd = np.clip(pose_cur_sd, 1e-6, None)
    else:
        s_pose_cur = np.ones((pose_cur_dim,), dtype=np.float32)
        pose_cur_mu = np.zeros_like(s_pose_cur)
        pose_cur_sd = np.ones_like(s_pose_cur)

    spec = {
        "J": int(J),
        "angvel_dim": int(J*3),
        "angvel_span": [0, int(J*3)],
        "tanh_scales_angvel": s.tolist(),
        "MuAngVel": mu.tolist(),
        "StdAngVel": sd.tolist(),
        "pose_hist_len": int(pose_hist_len),
        "pose_hist_dim": int(pose_hist_dim),
        "tanh_scales_pose_hist": s_pose.tolist() if pose_hist_dim > 0 else [],
        "MuPoseHist": pose_mu.tolist() if pose_hist_dim > 0 else [],
        "StdPoseHist": pose_sd.tolist() if pose_hist_dim > 0 else [],
        "pose_target_dim": int(pose_cur_dim),
        "tanh_scales_pose_target": s_pose_cur.tolist(),
        "MuPoseTarget": pose_cur_mu.tolist(),
        "StdPoseTarget": pose_cur_sd.tolist(),
        "meta": {"built_in": "pretrain(self-build-spec)", "quantiles": [2.5, 97.5]},
    }
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    print(f"[SelfBuild] wrote pretrain template: {save_path} | J={J} dim={J*3}")
    return spec


def _compute_soft_period_vectors(soft_contacts: np.ndarray, threshold: float = 0.50) -> np.ndarray:
    """
    依据左右脚软接触得分估算“软周期”向量：
      - 识别 soft_contact_score 从低 → 高 的上升沿（阈值 threshold）
      - 以上升沿为周期起点，线性插值得到相位 θ ∈ [0, 2π)
      - 输出 [sin(θ_L), cos(θ_L), sin(θ_R), cos(θ_R)]
    若缺乏足够上升沿，则退化为随时间均匀递增的相位。
    """
    T = int(soft_contacts.shape[0])
    if T <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    phases = np.zeros((T, 2), dtype=np.float32)
    two_pi = 2.0 * np.pi
    for foot in range(2):
        scores = soft_contacts[:, foot]
        high = scores > threshold
        rising = np.where(np.logical_and(high[1:], np.logical_not(high[:-1])))[0] + 1  # 上升沿索引
        if rising.size == 0:
            phases[:, foot] = np.linspace(0.0, two_pi, T, endpoint=False, dtype=np.float32)
            continue

        phase = np.zeros(T, dtype=np.float32)
        for i, start in enumerate(rising):
            end = rising[i + 1] if i + 1 < rising.size else T
            seg_len = max(1, end - start)
            ramp = np.linspace(0.0, two_pi, seg_len, endpoint=False, dtype=np.float32)
            phase[start:end] = ramp
        first = rising[0]
        if first > 0:
            ramp = np.linspace(-two_pi, 0.0, first, endpoint=False, dtype=np.float32)
            phase[:first] = ramp
        phases[:, foot] = phase

    sincos = np.concatenate([np.sin(phases), np.cos(phases)], axis=1)
    return sincos.astype(np.float32)


class YAngvelContactsDataset(Dataset):
    """
    严格Y侧：
      y_out_features (J*6 rot6d) → reproject_rot6d → rot6d_to_matrix
      → angvel_vec_from_R_seq(..., fps) → [T-1, 3J] → 按模板(angvel专用)归一化

    目标：
      来自 source_json 的软接触 (K=2)，对齐到 Frames[1:] → [T-1, 2]

    说明：
      - 在 __init__ 中预先过滤 T_eff-1 < T_w 的短片段（T_eff = min(T_npz, T_json)）。
      - 与主训练保持一致：angvel 计算传入 fps，而不是 dt。
      - 归一化仅使用模板里的 angvel 专用参数（tanh_scales_angvel 或 s_eff_angvel，可选 zscore）。
    """

    def __init__(self, in_glob: str, T_w: int,
                 require_zscore: bool = False,
                 norm_spec: dict | None = None,
                 p_event: float = 0.30,
                 event_thresh: float = 0.50,
                 event_min_gap: int = 6,
                 event_pre: int = -1):
        self.files = sorted(glob.glob(in_glob))
        if not self.files:
            raise RuntimeError(f"No npz matched: {in_glob}")
        self.T_w = int(T_w)

        self._soft_contact_cache: dict[str, np.ndarray] = {}

        # sampling mix params (Plan-A)
        self.p_event = float(p_event)
        self.event_thresh = float(event_thresh)
        self.event_min_gap = int(event_min_gap)
        self.event_pre = int(event_pre)

        # strict: consume in-memory spec only (no JSON read)
        if norm_spec is None:
            raise RuntimeError("norm_spec must be provided (self-build, no JSON)")
        self.tpl_path = "<in-memory>"
        self.period_threshold = float(event_thresh)
        self.period_dim = 4

        # probe J from first npz
        with np.load(self.files[0], allow_pickle=True) as z0:
            if "y_out_features" not in z0:
                raise RuntimeError(f"{os.path.basename(self.files[0])} missing y_out_features")
            Dy = int(np.asarray(z0["y_out_features"]).shape[1])
            if Dy % 6 != 0:
                raise RuntimeError(f"y_out_features length {Dy} is not multiple of 6")
            self.J = Dy // 6

        # angvel 专用归一化（读取 tanh_scales_angvel 或 s_eff_angvel；可选 zscore）
        self.norm = _make_angnorm_from_spec(norm_spec, J_times_3=self.J * 3, require_zscore=require_zscore)

        self.pose_hist_len = int(norm_spec.get("pose_hist_len", 0) or 0)
        pose_hist_scales = norm_spec.get("tanh_scales_pose_hist", [])
        pose_hist_mu = norm_spec.get("MuPoseHist", [])
        pose_hist_std = norm_spec.get("StdPoseHist", [])
        if self.pose_hist_len > 0 and len(pose_hist_scales) == self.pose_hist_len * self.J * 6:
            self.pose_hist_dim = int(norm_spec.get("pose_hist_dim", self.pose_hist_len * self.J * 6) or (self.pose_hist_len * self.J * 6))
            self.pose_hist_norm = VectorTanhNormalizer(
                np.asarray(pose_hist_scales, dtype=np.float32),
                np.asarray(pose_hist_mu, dtype=np.float32) if pose_hist_mu else None,
                np.asarray(pose_hist_std, dtype=np.float32) if pose_hist_std else None,
            )
        else:
            self.pose_hist_dim = 0
            self.pose_hist_norm = None

        pose_target_scales = norm_spec.get("tanh_scales_pose_target", [])
        pose_target_mu = norm_spec.get("MuPoseTarget", [])
        pose_target_std = norm_spec.get("StdPoseTarget", [])
        if pose_target_scales:
            self.pose_target_dim = int(norm_spec.get("pose_target_dim", self.J * 6) or (self.J * 6))
            self.pose_target_norm = VectorTanhNormalizer(
                np.asarray(pose_target_scales, dtype=np.float32),
                np.asarray(pose_target_mu, dtype=np.float32) if pose_target_mu else None,
                np.asarray(pose_target_std, dtype=np.float32) if pose_target_std else None,
            )
        else:
            self.pose_target_dim = self.J * 6
            self.pose_target_norm = None

        self.contact_dim = 2
        self.input_dim = self.contact_dim + self.J * 3 + self.pose_hist_dim

        # === 预过滤：仅保留 T_eff-1 >= T_w 的样本 ===
        self._valid_idx: list[int] = []
        self._skipped: list[tuple[str, int]] = []

        for i, p in enumerate(self.files):
            try:
                with np.load(p, allow_pickle=True) as z:
                    if "y_out_features" not in z:
                        # 缺关键键也视为不可用
                        self._skipped.append((os.path.basename(p), -1))
                        continue
                    Y_arr = np.asarray(z["y_out_features"], dtype=np.float32)
                    T_npz, Dy_npz = Y_arr.shape
                    if Dy_npz != self.J * 6:
                        raise RuntimeError(f"{os.path.basename(p)}: y_out_features dim {Dy_npz} != J*6 {self.J*6}")

                    if "source_json" not in z:
                        self._skipped.append((os.path.basename(p), -1))
                        continue
                    json_path = npz_scalar_to_str(z["source_json"])

                    # 为了严格对齐，使用 npz 和 json 的最短长度
                    if json_path not in self._soft_contact_cache:
                        self._soft_contact_cache[json_path] = _load_soft_contacts_from_json(json_path)
                    tgt_full = self._soft_contact_cache[json_path]  # [T_json, 2]
                    T_json = int(tgt_full.shape[0])

                    T_eff = min(T_npz, T_json)
                    eff = T_eff - 1
                    if eff >= self.T_w:
                        self._valid_idx.append(i)
                    else:
                        self._skipped.append((os.path.basename(p), eff))
            except Exception:
                self._skipped.append((os.path.basename(p), -1))

        if self._skipped:
            msg = ", ".join([f"{n}(T-1={e})" for n, e in self._skipped])
            print(f"[ds] skipped {len(self._skipped)} short/invalid clips: {msg}")
        if not self._valid_idx:
            raise RuntimeError(f"No clip has effective length >= T_w={self.T_w}")

    def __len__(self):
        return len(self._valid_idx)

    def __getitem__(self, idx: int):
        # 只从可用样本池中取
        p = self.files[self._valid_idx[idx]]
        with np.load(p, allow_pickle=True) as z:
            if "y_out_features" not in z:
                raise RuntimeError(f"{os.path.basename(p)} missing y_out_features")
            Y = np.asarray(z["y_out_features"], dtype=np.float32)  # [T_npz, J*6]
            T_npz, Dy = Y.shape
            if Dy != self.J * 6:
                raise RuntimeError(f"{os.path.basename(p)}: Dy mismatch {Dy} vs J*6 {self.J*6}")

            if "source_json" not in z:
                raise RuntimeError(f"{os.path.basename(p)} missing source_json")
            json_path = npz_scalar_to_str(z["source_json"])

            # 统一从 npz/json 获取 fps（影响角速度单位，非窗口长度判定）
            fps = float(_get_fps_from_npz_or_json(z, json_path))

            # targets from JSON (strict, cached)
            tgt_full = self._soft_contact_cache.get(json_path)
            if tgt_full is None:
                tgt_full = _load_soft_contacts_from_json(json_path)
                self._soft_contact_cache[json_path] = tgt_full  # cache lazily if unseen

        # 对齐长度（严格）
        T = min(T_npz, int(tgt_full.shape[0]))
        if T <= 1:
            raise RuntimeError(f"{os.path.basename(p)} too short after alignment: T={T}")

        if T != T_npz:
            Y = Y[:T]
        tgt_full = tgt_full[:T]

        # Rot6D -> R -> angvel （与主训练一致：传 fps）
        y_t = torch.from_numpy(Y).to(torch.float32)          # [T, J*6]
        y_t = reproject_rot6d(y_t.unsqueeze(0))[0]           # [T, J*6]
        J = self.J
        R = rot6d_to_matrix(y_t.view(1, T, J, 6))[0]         # [T, J, 3, 3]
        w = angvel_vec_from_R_seq(R.unsqueeze(0), fps)[0]    # [T-1, J, 3]
        ang = w.reshape(T - 1, J * 3).cpu().numpy().astype(np.float32)  # [T-1, 3J]

        # 目标与差分对齐到 Frames[1:]
        contact_seq = tgt_full[1:].astype(np.float32)        # [T-1, 2]
        period_full = _compute_soft_period_vectors(tgt_full, threshold=self.period_threshold)
        period = period_full[1:].astype(np.float32)          # [T-1, 4]

        pose_seq = y_t.cpu().numpy().astype(np.float32)      # [T, J*6]
        pose_target_raw = pose_seq[1:]                       # [T-1, J*6]
        if pose_target_raw.shape[0] != ang.shape[0]:
            raise RuntimeError(f"{os.path.basename(p)}: pose/ang time mismatch {pose_target_raw.shape[0]} vs {ang.shape[0]}")

        if self.pose_hist_len > 0:
            pose_hist = []
            for t_idx in range(ang.shape[0]):
                frames = []
                for h in range(self.pose_hist_len, 0, -1):
                    src_idx = t_idx + 1 - h
                    if src_idx < 0:
                        src_idx = 0
                    frames.append(pose_seq[src_idx])
                pose_hist.append(np.concatenate(frames, axis=0))
            pose_hist = np.stack(pose_hist, axis=0)
        else:
            pose_hist = np.zeros((ang.shape[0], 0), dtype=np.float32)

        # 归一化输入 / 目标
        ang_norm = self.norm.transform(ang)
        pose_hist_norm = self.pose_hist_norm.transform(pose_hist) if self.pose_hist_norm else pose_hist.astype(np.float32, copy=False)
        pose_target_norm = self.pose_target_norm.transform(pose_target_raw) if self.pose_target_norm else pose_target_raw.astype(np.float32, copy=False)

        inputs_full = np.concatenate(
            [
                contact_seq,
                ang_norm,
                pose_hist_norm,
            ],
            axis=1,
        )

        # 取窗（此时已保证 T-1 >= T_w）
        Tw = self.T_w
        Tm1 = T - 1
        use_event = (np.random.random() < self.p_event)
        if use_event:
            L = (contact_seq[:, 0] > self.event_thresh); R = (contact_seq[:, 1] > self.event_thresh)
            events = event_indices_from_LR(L, R, min_gap=self.event_min_gap)
            starts = window_starts_from_events(Tm1, Tw, events, pre=self.event_pre)
            s = int(np.random.choice(starts)) if len(starts) > 0 else np.random.randint(0, Tm1 - Tw + 1)
        else:
            s = np.random.randint(0, Tm1 - Tw + 1)
                # Safeguard: clamp start to valid range to guarantee window length Tw
        if s < 0:
            s = 0
        max_start = max(0, Tm1 - Tw)
        if s > max_start:
            s = max_start
        e = s + Tw

        sample = {
            "inputs": torch.from_numpy(inputs_full[s:e].astype(np.float32, copy=False)),
            "contact_target": torch.from_numpy(contact_seq[s:e].astype(np.float32, copy=False)),
            "period_hint": torch.from_numpy(period[s:e].astype(np.float32, copy=False)),
            "angvel_target": torch.from_numpy(ang_norm[s:e].astype(np.float32, copy=False)),
            "angvel_target_raw": torch.from_numpy(ang[s:e].astype(np.float32, copy=False)),
            "pose_target": torch.from_numpy(pose_target_norm[s:e].astype(np.float32, copy=False)),
            "pose_target_raw": torch.from_numpy(pose_target_raw[s:e].astype(np.float32, copy=False)),
        }
        return sample


def _split_sample(sample):
    if isinstance(sample, (list, tuple)):
        if len(sample) == 3:
            return sample[0], sample[1], sample[2]
        if len(sample) == 2:
            return sample[0], sample[1], None
    if isinstance(sample, dict):
        if "inputs" in sample and "contact_target" in sample:
            return (
                sample.get("inputs"),
                sample.get("contact_target"),
                sample.get("period_hint"),
            )
    raise RuntimeError(
        f"Unsupported sample structure: {type(sample)}; expected dict/list with 'inputs'/'contact_target'/'period_hint'."
    )




# ---------------------------- Model ----------------------------
class StepHead(nn.Module):
    def __init__(self, hidden_dim, K, bidirectional=False):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, K)
    def forward(self, h):
        return self.fc(h)                 # [B, T, K]

class AuxHead(nn.Module):
    def __init__(self, z_dim, K):
        super().__init__()
        self.fc = nn.Linear(z_dim, K)
    def forward(self, z):
        return self.fc(z)                 # [B, K]

class PeriodHead(nn.Module):
    def __init__(self, hidden_dim, out_dim, bidirectional=False):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, out_dim)
    def forward(self, h):
        return self.fc(h)                 # [B, T, out_dim]



def compute_pos_weight(tgt_btK: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    p = tgt_btK.mean(dim=(0,1))  # [K]
    pos_w = (1.0 - p) / (p.clamp_min(eps))
    return pos_w


def _collect_metrics(acc, metrics: dict):
    for k, v in metrics.items():
        if v is None:
            continue
        acc[k].append(float(v))


def _summarize_metrics(acc) -> dict:
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def _format_metrics(tag: str, epoch: int, metrics: dict) -> str:
    if not metrics:
        return f"{tag}[ep {epoch}] (no data)"
    parts = [f"{k}={metrics[k]:.4f}" for k in sorted(metrics.keys())]
    return f"{tag}[ep {epoch}] " + " | ".join(parts)


# ===================== [Pretrain Plan-A Isolation Layer] =====================

def _print_planA_banner():
    owned = ["StepHead(PT-only)", "AuxHead(PT-only)",
             "hazard_losses_soft_only(PT-only)",
             "event_indices_from_LR(PT-only)",
             "window_starts_from_events(PT-only)"]
    can_remove_in_main = [
        "主训练中的事件头/损失/事件采样（若主训练不产出事件）"
    ]
    print("\n" + "="*78)
    print("[Pretrain-PLAN-A] 预训练完全解耦启用")
    print("[Pretrain-PLAN-A] 由预训练脚本负责（PT owns）:", ", ".join(owned))
    print("[Pretrain-PLAN-A] 主训练可移除（若不消费事件）:", "; ".join(can_remove_in_main))
    print("[Pretrain-PLAN-A] 如需在主训练侧使用，可在主训练中包装一层调用本脚本等价实现。")
    print("="*78 + "\n")

def hazard_losses_soft_only(logits_step: torch.Tensor,
                            tgt: torch.Tensor,
                            win: int = 2,
                            smooth_w: float = 0.08,
                            edge_guard: int = 1,
                            pos_weight = None):
    """预训练专用 hazard 损失（轻量）:
    - 主项: 逐帧 BCE 与 K=2 软证据对齐
    - 平滑: 对 σ(logits) 做时间维 TV-L2
    """
    bce = F.binary_cross_entropy_with_logits(logits_step, tgt, reduction="mean", pos_weight=pos_weight)
    probs = torch.sigmoid(logits_step)
    tv = probs.new_tensor(0.0)
    if probs.dim() == 3 and probs.size(1) > 1:
        # 时间差分；可选忽略首尾 edge_guard 帧
        p = probs
        if edge_guard > 0 and p.size(1) > 2*edge_guard:
            p = p[:, edge_guard:-edge_guard, :]
        if p.size(1) > 1:
            tv = (p[:, 1:, :] - p[:, :-1, :]).pow(2).mean()
    loss = bce + smooth_w * tv
    stats = {"bce": float(bce.detach().cpu()), "tv": float(tv.detach().cpu()), "smooth_w": float(smooth_w)}
    return loss, stats

def event_indices_from_LR(L, R, min_gap: int = 6):
    """
    接受 torch.Tensor 或 numpy.ndarray；返回事件帧索引（min_gap 去抖）
    """

    try:
        L_np = L.detach().cpu().numpy() if hasattr(L, 'detach') else L
    except Exception:
        L_np = L
    try:
        R_np = R.detach().cpu().numpy() if hasattr(R, 'detach') else R
    except Exception:
        R_np = R

    Lb = np.asarray(L_np, dtype=bool)
    Rb = np.asarray(R_np, dtype=bool)

    idx = np.nonzero(Lb | Rb)[0].astype(int).tolist()
    out, last = [], -10**9
    for i in idx:
        if i - last >= min_gap:
            out.append(int(i))
            last = i
    return out


def window_starts_from_events(Tm1: int, Tw: int, events, pre: int = -1):
    """根据事件位置给出窗口起点候选，clamp 到合法范围（确保切片长度恰为 Tw）"""
    starts = []
    hi = max(0, Tm1 - Tw)  # inclusive upper bound for start
    for e in events:
        s = max(0, min(hi, (e + pre)))
        starts.append(int(s))
    return sorted(set(starts))

# =================== [End Pretrain Plan-A Isolation Layer] ===================



# ---------------------------- Train ----------------------------


class LinearNextProbe(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim, bias=True)
    def forward(self, x):  # x: [N, D]
        return self.fc(x)

@torch.no_grad()
def _peek_hidden_dim(encoder, dl, device) -> int:
    for batch in dl:
        if not isinstance(batch, dict) or "inputs" not in batch:
            continue
        inputs = batch["inputs"].to(device).float()
        _, h = encoder(inputs, return_summary=True)  # [B, T, H]
        return int(h.size(-1))
    raise RuntimeError("Empty dataloader when peeking hidden dim.")



def run_linear_next_angvel_probe(encoder, dl, device, J: int, norm_obj, *,
                                 max_batches: int = 200, epochs: int = 3, lr: float = 1e-3):
    """
    冻结 encoder，仅训练线性层预测下一步原始角速度:
      输入  : h_t   (来自 encoder 的每步隐状态)
      目标  : w_{t+1} (ds 输出的 norm 后角速度逆变换得到的原始角速度)
      评估  : 验证集 MSE vs 常数均值基线 & 复制上一帧基线
    备注  : 为避免小数据集下无验证集的问题，这里先累计所有样本，再做 80/20 拆分。
    """
    encoder.eval()
    H = _peek_hidden_dim(encoder, dl, device)
    probe = LinearNextProbe(H, J * 3).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    feats_all, yraw_all, yprev_all = [], [], []
    n_batches = 0

    with torch.no_grad():
        for batch in dl:
            if not isinstance(batch, dict) or "inputs" not in batch or "angvel_target" not in batch:
                continue
            inputs = batch["inputs"].to(device).float()                # [B, T, D_in]
            ang_norm = batch["angvel_target"].to(device).float()       # [B, T, 3J]
            if "angvel_target_raw" in batch and batch["angvel_target_raw"] is not None:
                ang_raw = batch["angvel_target_raw"].to(device).float()
            else:
                ang_raw = torch.from_numpy(
                    norm_obj.inverse_transform(ang_norm.cpu().numpy().reshape(-1, J * 3))
                ).to(device).view(ang_norm.size(0), ang_norm.size(1), J * 3)

            if inputs.size(1) < 2:
                continue

            _, h = encoder(inputs, return_summary=True)                # [B, T, H]
            X = h[:, :-1, :].reshape(-1, H)
            Y_target = ang_raw[:, 1:, :].reshape(-1, J * 3)
            Y_prev   = ang_raw[:, :-1, :].reshape(-1, J * 3)

            feats_all.append(X)
            yraw_all.append(Y_target)
            yprev_all.append(Y_prev)
            n_batches += 1
            if n_batches >= max_batches:
                break

    if not feats_all:
        raise RuntimeError("[Probe-Next] dataloader empty; cannot run probe.")

    Xall = torch.cat(feats_all, 0)
    Yall = torch.cat(yraw_all, 0)
    Yprev_all = torch.cat(yprev_all, 0)
    n = Xall.size(0)
    n_val = max(1, int(0.2 * n))
    split = n - n_val
    Xtr, Ytr = Xall[:split], Yall[:split]
    Xva, Yva, Yprev_va = Xall[split:], Yall[split:], Yprev_all[split:]

    for ep in range(1, epochs + 1):
        probe.train()
        opt.zero_grad(set_to_none=True)
        pred = probe(Xtr)
        loss = torch.mean((pred - Ytr) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        print(f"[Probe-Next] ep{ep} train_MSE={float(loss):.6f}")

    # 验证：与常数均值 & 复制上一帧基线对比
    probe.eval()
    with torch.no_grad():
        pred_va = probe(Xva)
        mse_probe = torch.mean((pred_va - Yva) ** 2).item()
        const = Ytr.mean(dim=0, keepdim=True)
        mse_const = torch.mean((const.expand_as(Yva) - Yva) ** 2).item()
        mse_copy  = torch.mean((Yprev_va - Yva) ** 2).item()
        rel_const = (mse_const - mse_probe) / max(mse_const, 1e-8)
        rel_copy  = (mse_copy  - mse_probe) / max(mse_copy , 1e-8)
        print(f"[Probe-Next] valid_MSE(probe)={mse_probe:.6f} | baseline(constant-mean)={mse_const:.6f} "
              f"| baseline(copy-last)={mse_copy:.6f} | rel_improve_vs_const={rel_const:.3%} | rel_improve_vs_copy={rel_copy:.3%}")



def train_linear_next_probe_all(encoder, dl, device, J: int, norm_obj, *, epochs: int = 3, lr: float = 1e-3, max_batches: int = 200):
    """
    不做验证拆分，使用全部样本训练线性 probe（h_t → w_{t+1} 原始角速度）。
    返回已训练好的 probe 以及最终 train_MSE。
    """
    encoder.eval()
    H = _peek_hidden_dim(encoder, dl, device)
    probe = LinearNextProbe(H, J * 3).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    feats_all, yraw_all = [], []
    n_batches = 0
    with torch.no_grad():
        for batch in dl:
            if not isinstance(batch, dict) or "inputs" not in batch or "angvel_target" not in batch:
                continue
            inputs = batch["inputs"].to(device).float()
            ang_norm = batch["angvel_target"].to(device).float()
            if "angvel_target_raw" in batch and batch["angvel_target_raw"] is not None:
                ang_raw = batch["angvel_target_raw"].to(device).float()
            else:
                ang_raw = torch.from_numpy(
                    norm_obj.inverse_transform(ang_norm.cpu().numpy().reshape(-1, J * 3))
                ).to(device).view(ang_norm.size(0), ang_norm.size(1), J * 3)

            if inputs.size(1) < 2:
                continue

            _, h = encoder(inputs, return_summary=True)  # [B, T, H]
            X = h[:, :-1, :].reshape(-1, H)
            Y_target = ang_raw[:, 1:, :].reshape(-1, J * 3)
            feats_all.append(X)
            yraw_all.append(Y_target)
            n_batches += 1
            if n_batches >= max_batches:
                break

    if not feats_all:
        raise RuntimeError("[Probe-Next-All] dataloader empty; cannot train probe.")
    Xall = torch.cat(feats_all, 0); Yall = torch.cat(yraw_all, 0)

    last_mse = None
    for ep in range(1, epochs + 1):
        probe.train()
        opt.zero_grad(set_to_none=True)
        pred = probe(Xall)
        loss = torch.mean((pred - Yall) ** 2)
        last_mse = float(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        print(f"[Probe-Next-All] ep{ep} train_MSE={last_mse:.6f}")
    return probe, last_mse


def rollout_autoreg_probe(encoder, probe, ds, device, J: int, norm_obj, *, warmup: int = 20, horizon: int = 20, nseq: int = 3):
    """
    自回归检验：
      - 选 nseq 个样本窗口（来自 ds[i]）
      - 前 warmup 步用 GT（teacher forcing）计算 h_t
      - 之后 horizon 步：每步用 probe 预测 raw w_{t+1} → transform 成 norm → 作为下一输入，闭环滚动；
        同时与 GT raw w 比较，打印每步 MSE，并汇总平均。
    """
    import numpy as np
    encoder.eval(); probe.eval()

    nseq = min(nseq, len(ds))
    all_step_mse = []  # [horizon] 累计

    for i in range(nseq):
        Yin_norm, _, _ = _split_sample(ds[i])                  # [Tw, 3J] (norm)
        Tw = Yin_norm.shape[0]
        if Tw <= 2:
            continue
        # 将 GT norm → raw，便于对比
        Yraw = norm_obj.inverse_transform(Yin_norm.numpy())  # [Tw, 3J]

        # 实际使用的 warmup/horizon（防越界）
        W = min(warmup, Tw - 2)
        H = min(horizon, Tw - 1 - W)
        if H <= 0:
            continue

        # 先用 GT warmup 计算到 h_W
        cur = Yin_norm[:W].numpy()           # [W, 3J] norm
        step_mse = []
        for k in range(H):
            # 1) 计算当前 h_t
            cur_t = torch.from_numpy(cur).to(device).float().unsqueeze(0)  # [1, t, 3J]
            with torch.no_grad():
                _, h = encoder(cur_t, return_summary=True)  # [1, t, H]
                h_last = h[:, -1, :]             # [1, H]
                # 2) 预测下一步 raw 角速度
                pred_raw = probe(h_last)         # [1, 3J]
                # 3) 与 GT raw 比较
                gt_raw = torch.from_numpy(Yraw[W + k: W + k + 1]).to(device).float()  # [1, 3J]
                mse = torch.mean((pred_raw - gt_raw) ** 2).item()
                step_mse.append(mse)
                # 4) 把预测的 raw → norm，作为下一输入
                pred_norm = norm_obj.transform(pred_raw.detach().cpu().numpy())  # [1, 3J]
            # 5) 闭环推进
            cur = np.concatenate([cur, pred_norm.astype(np.float32)], axis=0)   # time 维追加

        all_step_mse.append(step_mse)
        print(f"[Rollout] seq#{i} W={W} H={H} | per-step MSE (first 5): " +
              ", ".join(f"{v:.4f}" for v in step_mse[:5]) + (" ..." if len(step_mse) > 5 else ""))

    if all_step_mse:
        A = np.array([np.array(x, dtype=np.float64) for x in all_step_mse], dtype=object)
        # pad to same length for mean over horizon
        maxH = max(len(x) for x in all_step_mse)
        mat = np.full((len(all_step_mse), maxH), np.nan, dtype=np.float64)
        for r, row in enumerate(all_step_mse):
            mat[r, :len(row)] = row
        per_step_mean = np.nanmean(mat, axis=0)  # [maxH]
        print("[Rollout] mean MSE per step:", ", ".join(f"{v:.4f}" for v in per_step_mean[:10]) + (" ..." if maxH > 10 else ""))
        print(f"[Rollout] overall mean MSE={np.nanmean(mat):.6f}")
    else:
        print("[Rollout] no valid sequences (window too short).")


def _load_joint_names_from_tpl(tpl_path: str, J: int):
    """
    从 norm_template.json 里尽力读出骨骼名字序列（长度应为 J）。
    兼容几种可能字段：["skeleton"] / ["meta"]["skeleton"] / ["rot6d_spec"]["joints"] 等。
    失败则返回 None。
    """
    try:
        import json
        with open(tpl_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)
        candidates = []
        if isinstance(tpl.get("skeleton"), list):
            candidates = tpl["skeleton"]
        elif isinstance(tpl.get("meta"), dict):
            m = tpl["meta"]
            if isinstance(m.get("skeleton"), list):
                candidates = m["skeleton"]
            elif isinstance(m.get("rot6d_spec"), dict):
                rs = m["rot6d_spec"]
                for key in ["joints", "names", "order"]:
                    if isinstance(rs.get(key), list):
                        candidates = rs[key]; break
        elif isinstance(tpl.get("rot6d_spec"), dict):
            rs = tpl["rot6d_spec"]
            for key in ["joints", "names", "order"]:
                if isinstance(rs.get(key), list):
                    candidates = rs[key]; break
        if isinstance(candidates, list) and len(candidates) == J:
            return [str(x) for x in candidates]
    except Exception:
        pass
    return None


def _load_joint_names_from_npz(npz_path: str, J: int):
    """
    优先从 npz 直接读取骨骼名，避免依赖外部 json/template。
    支持的键：'bone_names'、'joint_names'、'joints'（object/string array）。
    """
    try:
        import numpy as _np
        with _np.load(npz_path, allow_pickle=True) as z:
            for key in ["bone_names", "joint_names", "joints", "names"]:
                if key in z:
                    arr = z[key]
                    try:
                        if hasattr(arr, "tolist"):
                            names = arr.tolist()
                        else:
                            names = list(arr)
                        names = [str(x) for x in names]
                        if len(names) == J:
                            return names
                    except Exception:
                        pass
    except Exception:
        pass
    return None
def _load_joint_names_from_source_json(npz_path: str, J: int):
    """
    如果模板里取不到，则读取该 npz 的 source_json，并尝试同样的字段。
    """
    try:
        import numpy as _np, json, os
        with _np.load(npz_path, allow_pickle=True) as z:
            if "source_json" not in z:
                return None
            sj = z["source_json"]
            if hasattr(sj, "item"):
                sj = sj.item()
            if isinstance(sj, bytes):
                sj = sj.decode("utf-8")
            src = str(sj)
        if not os.path.exists(src):
            return None
        with open(src, "r", encoding="utf-8") as f:
            doc = json.load(f)
        candidates = []
        if isinstance(doc.get("skeleton"), list):
            candidates = doc["skeleton"]
        elif isinstance(doc.get("meta"), dict):
            m = doc["meta"]
            if isinstance(m.get("skeleton"), list):
                candidates = m["skeleton"]
            elif isinstance(m.get("rot6d_spec"), dict):
                rs = m["rot6d_spec"]
                for key in ["joints", "names", "order"]:
                    if isinstance(rs.get(key), list):
                        candidates = rs[key]; break
        elif isinstance(doc.get("rot6d_spec"), dict):
            rs = doc["rot6d_spec"]
            for key in ["joints", "names", "order"]:
                if isinstance(rs.get(key), list):
                    candidates = rs[key]; break
        if isinstance(candidates, list) and len(candidates) == J:
            return [str(x) for x in candidates]
    except Exception:
        pass
    return None
def _find_joint_index(names, patterns):
    """
    在 names 中按多个关键字匹配（大小写不敏感）；返回首个命中的索引，否则 -1。
    patterns: 如 ["ball","_l"] 或 ["pelvis"] 等。
    """
    if not names:
        return -1
    low = [n.lower() for n in names]
    for i, n in enumerate(low):
        ok = True
        for kw in patterns:
            if kw.lower() not in n:
                ok = False; break
        if ok:
            return i
    return -1

def _xcorr_max(a, b):
    """
    归一化互相关，返回 (max_corr, lag_idx)，lag>0 表示 b 滞后 a。
    """
    import numpy as np
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    c = np.correlate(a, b, mode="full")
    c = c / (len(a))  # 标准化幅值（近似）
    lags = np.arange(-len(a)+1, len(a))
    k = int(np.nanargmax(np.abs(c)))
    return float(c[k]), int(lags[k])



def motion_cues_diagnostics(encoder, head_step, ds, device):
    """
    软分段运动诊断：不丢中间态
      - Soft-LCVR: mean_off(|ω|)/mean_on(|ω|)，权重 off=1-p，on=p
      - Soft-Corr: corr(1-p(contact), |ω|_腿段聚合)
      - Soft-AUC : 速度 score 区分 off/on 的加权 AUC（阈值无关）
      - Sway     : s(t)=pL-pR 与骨盆角速度互相关峰值与滞后
      - 主频匹配  : s(t) 与骨盆角速度主频相对差
    依赖已有工具函数：_load_joint_names_*、_find_joint_index、_xcorr_max、ds.norm.inverse_transform
    """
    import numpy as np, torch

    encoder.eval(); head_step.eval()
    J = ds.J

    # 读取关节名
    names = _load_joint_names_from_npz(ds.files[0], J) \
            or _load_joint_names_from_tpl(ds.tpl_path, J) \
            or _load_joint_names_from_source_json(ds.files[0], J)

    def _pick_idx(pats):
        if not names:
            return -1
        for p in pats:
            i = _find_joint_index(names, p)
            if i >= 0:
                return i
        return -1

    idx_ball_l = _pick_idx([["ball","_l"], ["foot","_l"]])
    idx_ball_r = _pick_idx([["ball","_r"], ["foot","_r"]])
    idx_thigh_l = _pick_idx([["thigh","_l"], ["upperleg","_l"], ["femur","_l"]])
    idx_calf_l  = _pick_idx([["calf","_l"],  ["lowerleg","_l"], ["shin","_l"]])
    idx_thigh_r = _pick_idx([["thigh","_r"], ["upperleg","_r"], ["femur","_r"]])
    idx_calf_r  = _pick_idx([["calf","_r"],  ["lowerleg","_r"], ["shin","_r"]])
    idxPel = _pick_idx([["pelvis"], ["hips"], ["root"]])

    print(f"[MotionDiag] joints: "
          f"L(ball/thigh/calf)={idx_ball_l if idx_ball_l>=0 else '?'}({names[idx_ball_l] if names and idx_ball_l>=0 else '?'})/"
          f"{idx_thigh_l if idx_thigh_l>=0 else '?'}({names[idx_thigh_l] if names and idx_thigh_l>=0 else '?'})/"
          f"{idx_calf_l if idx_calf_l>=0 else '?'}({names[idx_calf_l] if names and idx_calf_l>=0 else '?'}) ; "
          f"R(ball/thigh/calf)={idx_ball_r if idx_ball_r>=0 else '?'}({names[idx_ball_r] if names and idx_ball_r>=0 else '?'})/"
          f"{idx_thigh_r if idx_thigh_r>=0 else '?'}({names[idx_thigh_r] if names and idx_thigh_r>=0 else '?'})/"
          f"{idx_calf_r if idx_calf_r>=0 else '?'}({names[idx_calf_r] if names and idx_calf_r>=0 else '?'}) ; "
          f"Pelvis={idxPel if idxPel>=0 else '?'}({names[idxPel] if names and idxPel>=0 else '?'})")

    def _nanmean_w(x, w, eps=1e-9):
        x = np.asarray(x, np.float64); w = np.asarray(w, np.float64)
        m = np.isfinite(x) & np.isfinite(w)
        if not m.any(): return float("nan")
        W = w[m].sum()
        if W < eps: return float("nan")
        return float((x[m] * w[m]).sum() / (W + eps))

    def _soft_lcvr(speed, p, eps=1e-9):
        w_on  = np.clip(p, 0.0, 1.0)
        w_off = 1.0 - w_on
        mu_on  = _nanmean_w(speed, w_on, eps)
        mu_off = _nanmean_w(speed, w_off, eps)
        if not np.isfinite(mu_on) or not np.isfinite(mu_off): return float("nan")
        return float(mu_off / (mu_on + eps))

    def _soft_corr(x, y):
        x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
        if np.std(x) < 1e-8 or np.std(y) < 1e-8: return float("nan")
        return float(np.corrcoef(x, y)[0,1])

    def _weighted_auc(scores, p_pos, eps=1e-9):
        s = np.asarray(scores, np.float64)
        w1 = np.asarray(p_pos,  np.float64)
        w0 = 1.0 - w1
        m = np.isfinite(s) & np.isfinite(w1) & np.isfinite(w0)
        s, w1, w0 = s[m], w1[m], w0[m]
        W1, W0 = w1.sum(), w0.sum()
        if W1 < eps or W0 < eps: return float("nan")
        idx = np.argsort(s)
        s, w1, w0 = s[idx], w1[idx], w0[idx]
        cum_w0 = np.cumsum(w0)
        return float((w1 * cum_w0).sum() / (W1 * W0 + eps))

    def _main_freq(x):
        x = np.asarray(x, np.float64)
        x = x - np.nanmean(x)
        spec = np.fft.rfft(np.nan_to_num(x, nan=0.0))
        mag = np.abs(spec)
        if mag.size <= 1: return 0
        return int(np.argmax(mag[1:])) + 1

    nseq = min(5, len(ds))
    stats = []

    for i in range(nseq):
        Yin_norm, _, _ = _split_sample(ds[i])                    # [T,3J] (normed)
        T = Yin_norm.shape[0]
        Yin = Yin_norm.unsqueeze(0).to(device).float()

        with torch.no_grad():
            _, h = encoder(Yin, return_summary=True)  # [1,T,H]
            logits = head_step(h)              # [1,T,2]
            probs  = torch.sigmoid(logits)[0].cpu().numpy()  # [T,2]

        Yraw = ds.norm.inverse_transform(Yin_norm.numpy())   # [T,3J]
        W = Yraw.reshape(T, J, 3)
        Wmag = np.linalg.norm(W, axis=-1)     # [T,J]

        idxs_L = [k for k in (idx_thigh_l, idx_calf_l) if k >= 0] or ([idx_ball_l] if idx_ball_l>=0 else [])
        idxs_R = [k for k in (idx_thigh_r, idx_calf_r) if k >= 0] or ([idx_ball_r] if idx_ball_r>=0 else [])
        speed_L = Wmag[:, idxs_L].mean(axis=1) if idxs_L else np.full((T,), np.nan)
        speed_R = Wmag[:, idxs_R].mean(axis=1) if idxs_R else np.full((T,), np.nan)

        pL, pR = probs[:,0], probs[:,1]
        LCVR_L = _soft_lcvr(speed_L, pL)
        LCVR_R = _soft_lcvr(speed_R, pR)
        CVC_L  = _soft_corr(1.0 - pL, speed_L)
        CVC_R  = _soft_corr(1.0 - pR, speed_R)
        AUC_L  = _weighted_auc(speed_L, 1.0 - pL)
        AUC_R  = _weighted_auc(speed_R, 1.0 - pR)

        s = pL - pR
        Pel = W[:, idxPel, :] if idxPel >= 0 else W.mean(axis=1)[:, :3]

        best_axis = 0; best_corr = float("nan"); best_lag = 0
        for ax in range(3):
            corr, lag = _xcorr_max(s, Pel[:, ax])
            if (not np.isnan(corr)) and (abs(corr) > abs(best_corr) if not np.isnan(best_corr) else True):
                best_corr, best_lag, best_axis = float(corr), int(lag), ax

        f_s = _main_freq(s); f_p = _main_freq(Pel[:, best_axis])
        f_rel = float(abs(f_s - f_p) / max(f_s, 1)) if max(f_s, f_p) > 0 else float("nan")

        eff_on_L  = float(np.clip(pL, 0, 1).sum())
        eff_off_L = float(np.clip(1 - pL, 0, 1).sum())
        eff_on_R  = float(np.clip(pR, 0, 1).sum())
        eff_off_R = float(np.clip(1 - pR, 0, 1).sum())
        print(f"[MotionDiag] seq#{i} eff-weights L(on/off)={eff_on_L:.1f}/{eff_off_L:.1f} "
              f"R(on/off)={eff_on_R:.1f}/{eff_off_R:.1f}")

        stats.append(dict(
            LCVR_L=LCVR_L, LCVR_R=LCVR_R,
            CVC_L=CVC_L,   CVC_R=CVC_R,
            AUC_L=AUC_L,   AUC_R=AUC_R,
            sway_axis=best_axis, sway_corr=best_corr, sway_lag=best_lag,
            freq_rel_diff=f_rel
        ))

    if not stats:
        print("[MotionDiag] no samples."); 
        return

    def _nanmean(xs):
        arr = np.array([x for x in xs], dtype=float)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    print("[MotionDiag][per-seq]:")
    for i, s in enumerate(stats):
        print(f"  #{i} Soft-LCVR(L/R)={s['LCVR_L']:.2f}/{s['LCVR_R']:.2f} | "
              f"Soft-Corr(L/R)={s['CVC_L']:.2f}/{s['CVC_R']:.2f} | "
              f"AUC(L/R)={s['AUC_L']:.2f}/{s['AUC_R']:.2f} | "
              f"sway(axis={s['sway_axis']}) corr={s['sway_corr']:.2f} lag={s['sway_lag']} | "
              f"f_rel={s['freq_rel_diff']:.2f}")

    print(f"[MotionDiag][mean] Soft-LCVR(L/R)={_nanmean([s['LCVR_L'] for s in stats]):.2f}/"
          f"{_nanmean([s['LCVR_R'] for s in stats]):.2f} | "
          f"Soft-Corr(L/R)={_nanmean([s['CVC_L'] for s in stats]):.2f}/"
          f"{_nanmean([s['CVC_R'] for s in stats]):.2f} | "
          f"AUC(L/R)={_nanmean([s['AUC_L'] for s in stats]):.2f}/"
          f"{_nanmean([s['AUC_R'] for s in stats]):.2f} | "
          f"sway_corr={_nanmean([abs(s['sway_corr']) for s in stats]):.2f} | "
          f"|f_rel|={_nanmean([s['freq_rel_diff'] for s in stats]):.2f}")

    idx_ball_r = _pick_idx([["ball","_r"], ["foot","_r"]])
    idx_thigh_l = _pick_idx([["thigh","_l"], ["upperleg","_l"], ["femur","_l"]])
    idx_calf_l  = _pick_idx([["calf","_l"],  ["lowerleg","_l"], ["shin","_l"]])
    idx_thigh_r = _pick_idx([["thigh","_r"], ["upperleg","_r"], ["femur","_r"]])
    idx_calf_r  = _pick_idx([["calf","_r"],  ["lowerleg","_r"], ["shin","_r"]])
    idxPel = _pick_idx([["pelvis"], ["hips"], ["root"]])

    print(
        "[MotionDiag] joints:",
        f"L(ball/thigh/calf)={idx_ball_l if idx_ball_l>=0 else '?'}({names[idx_ball_l] if names and idx_ball_l>=0 else '?'})/"
        f"{idx_thigh_l if idx_thigh_l>=0 else '?'}({names[idx_thigh_l] if names and idx_thigh_l>=0 else '?'})/"
        f"{idx_calf_l if idx_calf_l>=0 else '?'}({names[idx_calf_l] if names and idx_calf_l>=0 else '?'}) ;",
        f"R(ball/thigh/calf)={idx_ball_r if idx_ball_r>=0 else '?'}({names[idx_ball_r] if names and idx_ball_r>=0 else '?'})/"
        f"{idx_thigh_r if idx_thigh_r>=0 else '?'}({names[idx_thigh_r] if names and idx_thigh_r>=0 else '?'})/"
        f"{idx_calf_r if idx_calf_r>=0 else '?'}({names[idx_calf_r] if names and idx_calf_r>=0 else '?'}) ;",
        f"Pelvis={idxPel if idxPel>=0 else '?'}({names[idxPel] if names and idxPel>=0 else '?'})",
    )

    # === helpers（函数内局部，避免全局污染） ===
    def _nanmean_w(x, w, eps=1e-9):
        x = np.asarray(x, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(w)
        if not m.any(): return float("nan")
        w_sum = w[m].sum()
        if w_sum < eps: return float("nan")
        return float((x[m] * w[m]).sum() / (w_sum + eps))

    def _soft_lcvr(speed, p, eps=1e-9):
        # mean_off / mean_on, 软权重：off=1-p, on=p
        w_on  = np.clip(p, 0.0, 1.0).astype(np.float64)
        w_off = (1.0 - w_on)
        mu_on  = _nanmean_w(speed, w_on, eps)
        mu_off = _nanmean_w(speed, w_off, eps)
        if not np.isfinite(mu_on) or not np.isfinite(mu_off): return float("nan")
        return float(mu_off / (mu_on + eps))

    def _soft_corr(x, y, w=None, eps=1e-9):
        # 加权皮尔逊（w=None 则普通皮尔逊）
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if w is None:
            if x.std() < 1e-8 or y.std() < 1e-8: return float("nan")
            return float(np.corrcoef(x, y)[0,1])
        w = np.asarray(w, dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
        if not m.any(): return float("nan")
        w = w[m]; x = x[m]; y = y[m]
        W = w.sum()
        if W < eps: return float("nan")
        mx = (w*x).sum()/W; my = (w*y).sum()/W
        vx = (w*(x-mx)**2).sum()/W; vy = (w*(y-my)**2).sum()/W
        if vx < 1e-12 or vy < 1e-12: return float("nan")
        cov = (w*(x-mx)*(y-my)).sum()/W
        return float(cov / np.sqrt(vx*vy))

    def _weighted_auc(scores, p_off, eps=1e-9):
        """
        用速度 scores 区分 off(权重=p_off) vs on(权重=1-p_off) 的加权AUC（阈值无关）。
        AUC = P(score_off > score_on)，带权重。
        """
        s = np.asarray(scores, dtype=np.float64)
        w1 = np.clip(np.asarray(p_off, dtype=np.float64), 0.0, 1.0)      # class=off 的权重
        w0 = 1.0 - w1                                                    # class=on  的权重
        m = np.isfinite(s) & np.isfinite(w1) & np.isfinite(w0)
        s, w1, w0 = s[m], w1[m], w0[m]
        W1, W0 = w1.sum(), w0.sum()
        if W1 < eps or W0 < eps: return float("nan")
        idx = np.argsort(s)  # 升序
        s, w1, w0 = s[idx], w1[idx], w0[idx]
        cum_w0 = np.cumsum(w0)
        auc_num = float((w1 * cum_w0).sum())
        auc = auc_num / (W1 * W0 + eps)
        return float(auc)

    def _main_freq(x):
        x = x - x.mean()
        spec = np.fft.rfft(x)
        mag = np.abs(spec)
        if mag.size <= 1: return 0
        k = int(np.argmax(mag[1:])) + 1
        return k

    # === 采样若干窗口 ===
    nseq = min(5, len(ds))
    stats = []

    for i in range(nseq):
        Yin_norm, _, _ = _split_sample(ds[i])                               # [T, 3J]
        T = Yin_norm.shape[0]
        Yin = Yin_norm.unsqueeze(0).to(device).float()    # [1,T,3J]

        with torch.no_grad():
            _, h = encoder(Yin, return_summary=True)      # [1,T,H]
            logits = head_step(h)                         # [1,T,2]
            probs  = torch.sigmoid(logits)[0].cpu().numpy()  # [T,2]

        # 还原原始角速度
        Yraw = ds.norm.inverse_transform(Yin_norm.numpy())   # [T,3J]
        W = Yraw.reshape(T, J, 3)
        # 防 NaN
        if not np.isfinite(W).all():
            W = np.nan_to_num(W, nan=0.0)
        Wmag = np.linalg.norm(W, axis=-1)                    # [T,J]

        # 腿段聚合：优先 thigh+calf，缺失则回退 ball
        idxs_L = [k for k in (idx_thigh_l, idx_calf_l) if k >= 0]
        idxs_R = [k for k in (idx_thigh_r, idx_calf_r) if k >= 0]
        if not idxs_L and idx_ball_l >= 0: idxs_L = [idx_ball_l]
        if not idxs_R and idx_ball_r >= 0: idxs_R = [idx_ball_r]

        if idxs_L:
            speed_L = Wmag[:, idxs_L].mean(axis=1)          # [T]
        else:
            speed_L = np.full((T,), np.nan, dtype=np.float64)
        if idxs_R:
            speed_R = Wmag[:, idxs_R].mean(axis=1)
        else:
            speed_R = np.full((T,), np.nan, dtype=np.float64)

        pL, pR = probs[:,0], probs[:,1]
        # Soft 指标（不中断、不丢帧）
        LCVR_L = _soft_lcvr(speed_L, pL)
        LCVR_R = _soft_lcvr(speed_R, pR)
        # 速度-接触相关（与 1-p 相关，越大表示“非接触速度更大”）
        CVC_L = _soft_corr(1.0 - pL, speed_L)
        CVC_R = _soft_corr(1.0 - pR, speed_R)
        # 加权 AUC：score=速度，正类=off(1-p)
        AUC_L = _weighted_auc(speed_L, 1.0 - pL)
        AUC_R = _weighted_auc(speed_R, 1.0 - pR)

        # 左右晃动：与交替信号相关（仍用 s=pL-pR，并不需要阈值）
        s = pL - pR
        if idxPel >= 0:
            Pel = W[:, idxPel, :]   # [T,3]
        else:
            Pel = W.mean(axis=1)[:, :3]  # 兜底

        best_axis = 0; best_corr = float("nan"); best_lag = 0
        for ax in range(3):
            corr, lag = _xcorr_max(s, Pel[:, ax])
            if (not np.isnan(corr)) and (abs(corr) > abs(best_corr) if not np.isnan(best_corr) else True):
                best_corr, best_lag, best_axis = float(corr), int(lag), ax

        f_s = _main_freq(s)
        f_p = _main_freq(Pel[:, best_axis])
        f_rel = float(abs(f_s - f_p) / max(f_s, 1)) if max(f_s, f_p) > 0 else float("nan")

        # 打印一些统计，确认不会出现“全空集”
        eff_on_L  = float(np.clip(pL, 0, 1).sum())
        eff_off_L = float(np.clip(1 - pL, 0, 1).sum())
        eff_on_R  = float(np.clip(pR, 0, 1).sum())
        eff_off_R = float(np.clip(1 - pR, 0, 1).sum())
        print(f"[MotionDiag] seq#{i} eff-weights L(on/off)={eff_on_L:.1f}/{eff_off_L:.1f} "
              f"R(on/off)={eff_on_R:.1f}/{eff_off_R:.1f}")

        stats.append(dict(
            LCVR_L=LCVR_L, LCVR_R=LCVR_R,
            CVC_L=CVC_L,   CVC_R=CVC_R,
            AUC_L=AUC_L,   AUC_R=AUC_R,
            sway_axis=best_axis, sway_corr=best_corr, sway_lag=best_lag,
            freq_rel_diff=f_rel
        ))

    # === 汇总打印 ===
    if not stats:
        print("[MotionDiag] no samples.")
        return

    def _nanmean(xs):
        arr = np.array([x for x in xs], dtype=float)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    print("[MotionDiag][per-seq]:")
    for i, s in enumerate(stats):
        print(
            f"  #{i} Soft-LCVR(L/R)={s['LCVR_L']:.2f}/{s['LCVR_R']:.2f} | "
            f"Soft-Corr(L/R)={s['CVC_L']:.2f}/{s['CVC_R']:.2f} | "
            f"AUC(L/R)={s['AUC_L']:.2f}/{s['AUC_R']:.2f} | "
            f"sway(axis={s['sway_axis']}) corr={s['sway_corr']:.2f} lag={s['sway_lag']} | "
            f"f_rel={s['freq_rel_diff']:.2f}"
        )

    print(
        f"[MotionDiag][mean] Soft-LCVR(L/R)={_nanmean([s['LCVR_L'] for s in stats]):.2f}/"
        f"{_nanmean([s['LCVR_R'] for s in stats]):.2f} | "
        f"Soft-Corr(L/R)={_nanmean([s['CVC_L'] for s in stats]):.2f}/"
        f"{_nanmean([s['CVC_R'] for s in stats]):.2f} | "
        f"AUC(L/R)={_nanmean([s['AUC_L'] for s in stats]):.2f}/"
        f"{_nanmean([s['AUC_R'] for s in stats]):.2f} | "
        f"sway_corr={_nanmean([abs(s['sway_corr']) for s in stats]):.2f} | "
        f"|f_rel|={_nanmean([s['freq_rel_diff'] for s in stats]):.2f}"
    )


def diag_norm_roundtrip(ds, n_samples: int = 2):
    """
    从 ds 里取若干窗口：Y(norm) → inverse_transform → transform，检查误差
    """
    import numpy as np
    ok = True
    for i in range(min(n_samples, len(ds))):
        sample = ds[i]
        ang_norm = sample.get("angvel_target")
        if ang_norm is None:
            raise RuntimeError("Dataset sample missing 'angvel_target'; diagnostics require normalized angvel.")
        if isinstance(ang_norm, torch.Tensor):
            Yin_np = ang_norm.detach().cpu().numpy()
        else:
            Yin_np = np.asarray(ang_norm, dtype=np.float32)
        Yin_inv = ds.norm.inverse_transform(Yin_np)    # 回到原始角速度
        Yin_rt  = ds.norm.transform(Yin_inv)           # 再变换回来
        err = np.abs(Yin_rt - Yin_np)
        print(f"[Diag-Norm] roundtrip_err: mean={err.mean():.4g} | p99={np.percentile(err,99):.4g} | max={err.max():.4g}")
        ok = ok and (np.percentile(err, 99.5) < 1e-3)
    if ok:
        print("[Diag-Norm] round-trip OK.")



def run_probes_and_motion_diag(enc, head_step, ds, device):
    """
    统一收口：运行软分段运动诊断 + 两个 probe。
    通过参数传入 enc/head_step/ds/device，避免未解析引用与缩进问题。
    """
    import json
    # 先跑软分段 MotionDiag（如果可用）
    try:
        motion_cues_diagnostics(enc, head_step, ds, device)
    except Exception as e:
        print("[MotionDiag] failed:", repr(e))

    # 解析左右腿关节索引
    try:
        names = _load_joint_names_from_npz(ds.files[0], ds.J) \
                or _load_joint_names_from_tpl(ds.tpl_path, ds.J) \
                or _load_joint_names_from_source_json(ds.files[0], ds.J)

        def _pick_idx(pats):
            if not names: 
                return -1
            for p in pats:
                i = _find_joint_index(names, p)
                if i >= 0:
                    return i
            return -1

        idx_ball_l = _pick_idx([["ball","_l"], ["foot","_l"]])
        idx_ball_r = _pick_idx([["ball","_r"], ["foot","_r"]])
        idx_thigh_l = _pick_idx([["thigh","_l"], ["upperleg","_l"], ["femur","_l"]])
        idx_calf_l  = _pick_idx([["calf","_l"],  ["lowerleg","_l"], ["shin","_l"]])
        idx_thigh_r = _pick_idx([["thigh","_r"], ["upperleg","_r"], ["femur","_r"]])
        idx_calf_r  = _pick_idx([["calf","_r"],  ["lowerleg","_r"], ["shin","_r"]])

        left_idxs  = [k for k in (idx_thigh_l, idx_calf_l) if k >= 0] or ([idx_ball_l] if idx_ball_l>=0 else [])
        right_idxs = [k for k in (idx_thigh_r, idx_calf_r) if k >= 0] or ([idx_ball_r] if idx_ball_r>=0 else [])

        if left_idxs and right_idxs:
            try:
                probe_contact_decode_from_latent(encoder=enc, ds=ds, device=device,
                                                 left_idxs=left_idxs, right_idxs=right_idxs)
            except Exception as e:
                print("[Probes][contact_decode] failed:", repr(e))
            try:
                probe_speed_nextstep_gain_from_latent(encoder=enc, ds=ds, device=device,
                                                      left_idxs=left_idxs, right_idxs=right_idxs)
            except Exception as e:
                print("[Probes][next_speed] failed:", repr(e))
        else:
            print("[Probes] skip: cannot resolve left/right leg indices.")
    except Exception as e:
        print("[Probes] failed:", repr(e))


@dataclass
class InputSlices:
    contact: slice
    ang: slice
    pose_hist: Optional[slice]


class InputProjectors:
    """Ensure each branch observes only its intended subset of inputs."""
    def __init__(
        self,
        layout: InputSlices,
        *,
        period_include_ang_sign: bool = False,
        period_use_ang_features: bool = False,
        angnorm: Optional[AngvelNormalizer] = None,
        amp_linear: bool = False,
    ):
        self.layout = layout
        self.period_include_ang_sign = bool(period_include_ang_sign)
        self.period_use_ang_features = bool(period_use_ang_features)
        self.angnorm = angnorm
        self.amp_linear = bool(amp_linear)

        if angnorm is not None:
            # Cache tensors for fast denorm/norm toggling; move to device lazily in forward.
            self._ang_s_eff = torch.tensor(angnorm.scales, dtype=torch.float32)
            self._ang_mu = None if angnorm.mu is None else torch.tensor(angnorm.mu, dtype=torch.float32)
            self._ang_std = None if angnorm.std is None else torch.tensor(angnorm.std, dtype=torch.float32)
        else:
            self._ang_s_eff = None
            self._ang_mu = None
            self._ang_std = None

    def period(self, inputs: torch.Tensor) -> torch.Tensor:
        proj = torch.zeros_like(inputs)
        proj[..., self.layout.contact] = inputs[..., self.layout.contact]
        if self.period_include_ang_sign:
            proj[..., self.layout.ang] = torch.sign(inputs[..., self.layout.ang])
        elif self.period_use_ang_features:
            proj[..., self.layout.ang] = inputs[..., self.layout.ang]
        if self.layout.pose_hist is not None:
            proj[..., self.layout.pose_hist] = inputs[..., self.layout.pose_hist]
        return proj

    @torch.no_grad()
    def _ang_denorm_torch(self, Y: torch.Tensor) -> torch.Tensor:
        """Inverse of dataset normalization: tanh → raw (optionally undo z-score)."""
        if self._ang_s_eff is None:
            return Y

        s = self._ang_s_eff.to(Y.device)
        X = Y.clone()
        if self._ang_mu is not None and self._ang_std is not None:
            mu = self._ang_mu.to(Y.device)
            std = self._ang_std.to(Y.device)
            X = X * std + mu
        X = X.clamp_(-0.999999, 0.999999)
        return X.atanh_() * s

    @torch.no_grad()
    def _ang_norm_torch(self, W: torch.Tensor) -> torch.Tensor:
        if self._ang_s_eff is None:
            return W

        s = self._ang_s_eff.to(W.device)
        X = W / s
        X = X.tanh_()
        if self._ang_mu is not None and self._ang_std is not None:
            mu = self._ang_mu.to(W.device)
            std = self._ang_std.to(W.device)
            X = (X - mu) / std
        return X

    def amp(
        self,
        inputs: torch.Tensor,
        *,
        scales: Optional[torch.Tensor] = None,
        shuffle_time: bool = False,
        isolate_for_equiv: bool = False,
    ) -> torch.Tensor:
        """
        Prepare amp-specific input channels: angular velocity only, applying
        denorm → scale → renorm so equivariance loss can observe true magnitude shifts.
        """
        if self.angnorm is None:
            raise RuntimeError("InputProjectors.amp expected angnorm; dataset should supply AngvelNormalizer.")

        proj = torch.zeros_like(inputs)
        ang_slice = self.layout.ang
        ang_normed = inputs[..., ang_slice]

        ang_raw = self._ang_denorm_torch(ang_normed)
        if scales is not None:
            ang_raw = ang_raw * scales.view(-1, 1, 1)
        use_linear = self.amp_linear or isolate_for_equiv
        if use_linear:
            s_eff = self._ang_s_eff.to(ang_raw.device) if self._ang_s_eff is not None else 1.0
            proj[..., ang_slice] = ang_raw / s_eff
        else:
            ang_scaled_norm = self._ang_norm_torch(ang_raw)
            proj[..., ang_slice] = ang_scaled_norm
        return proj

    def decode(
        self,
        soft_period: torch.Tensor,
        amp_scalar: Optional[torch.Tensor] = None,
        *,
        use_amp: bool = False,
        film: bool = False,
    ) -> torch.Tensor:
        if not use_amp or amp_scalar is None:
            return soft_period
        amp = amp_scalar.unsqueeze(-1)
        if film:
            return soft_period * (1.0 + amp)
        raise NotImplementedError("Decoder amp fusion via concatenation not supported in this script.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", type=str, required=True)
    ap.add_argument("--T_w", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--encoder_layers", type=int, default=3,
                    help="逐帧 MLP 编码器的层数。")
    ap.add_argument("--encoder_dropout", type=float, default=0.1,
                    help="逐帧 MLP 编码器的 dropout。")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pose_hist_len", type=int, default=3,
                    help="输入时纳入的历史姿态帧数（只含过去帧，不含当前帧）。")
    ap.add_argument("--period_dim", type=int, default=32,
                    help="软周期高维向量的维度。")
    ap.add_argument("--w_pose", type=float, default=1.0,
                    help="姿态重建损失的权重。")
    ap.add_argument("--w_contact", type=float, default=0.25,
                    help="接触 BCE 辅助损失权重。")
    ap.add_argument("--w_period_hint", type=float, default=0.1,
                    help="软周期前 4 维对齐 sin/cos 提示的权重。")
    ap.add_argument("--w_amp_rank", type=float, default=0.6,
                    help="相对幅度排序对比损失权重。")
    ap.add_argument("--w_amp_rel", type=float, default=0.05,
                    help="相对幅度 L1 约束权重（仅做稳定器）。")
    ap.add_argument("--w_amp_equiv", type=float, default=0.15,
                    help="幅度缩放等变正则权重。")
    ap.add_argument("--w_period_inv", type=float, default=0.2,
                    help="缩放增强时 soft_period 保持不变的正则权重。")
    ap.add_argument("--w_phase_energy", type=float, default=0.0,
                    help="软周期能量底线正则权重。")
    ap.add_argument("--phase_energy_eps", type=float, default=0.05,
                    help="软周期能量下限（若 w_phase_energy>0 生效）。")
    ap.add_argument("--amp_tau", type=float, default=0.25,
                    help="排序对比损失的温度系数 τ。")
    ap.add_argument("--amp_pairs", type=int, default=6,
                    help="每个样本用于排序对比的帧对数量。")
    ap.add_argument("--amp_margin", type=float, default=0.05,
                    help="排序对比的最小幅度差阈值；低于该阈值的帧对将跳过。")
    ap.add_argument("--amp_scale_min", type=float, default=0.7,
                    help="幅度等变正则使用的最小缩放因子。")
    ap.add_argument("--amp_scale_max", type=float, default=1.3,
                    help="幅度等变正则使用的最大缩放因子。")
    ap.add_argument("--amp_detach", action="store_true",
                    help="幅度头读取的隐状态不回传梯度（保护节律支路）。")
    ap.add_argument("--amp_warmup_epochs", type=int, default=0,
                    help="幅度损失暖启动的 epoch 数；>0 时线性放大到 1。")
    ap.add_argument("--amp_share_encoder", action="store_true",
                    help="让幅度/周期/接触共享同一编码器（默认为 False，即幅度支路使用独立编码器）。")
    ap.add_argument("--amp_linear_equiv", dest="amp_linear_equiv", action="store_true", default=True,
                    help="在幅度等变训练时使用线性归一化（不再重新施加 tanh），让缩放扰动更易被感知（默认开启）。")
    ap.add_argument("--amp_linear_equiv_off", dest="amp_linear_equiv", action="store_false",
                    help="关闭幅度等变线性归一化，恢复旧的 tanh 归一化。")
    ap.add_argument("--resume", type=str, default=None,
                    help="从此前保存的 motion_encoder *.pt 中恢复权重（仅加载模型参数，不恢复优化器）。")
    ap.add_argument("--out_best", type=str, default=None,
                    help="当 scale_slope 创新高时，额外保存最佳 checkpoint 的路径（默认在 --out 基础上追加 .best.pt）。")
    ap.add_argument("--period_use_ang_sign", action="store_true",
                    help="在节律支路中附加角速度的符号（保持幅度不曝光）。")
    ap.add_argument("--decode_use_amp", action="store_true",
                    help="允许在解码器输入中消费幅度标量（默认关闭，仅调试）。")
    ap.add_argument("--decode_film_amp", action="store_true",
                    help="若 decode_use_amp，则使用 (1+amp) 的 FiLM 缩放 soft_period。")
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--mixed_precision", action="store_true",
                    help="在 CUDA 上启用 torch.cuda.amp，减少显存/加速（需 GPU 支持）")
    ap.add_argument("--require_zscore", action="store_true",
                    help="Require MuAngVel/StdAngVel in template. If absent, raise instead of skipping z-score.")
    ap.add_argument("--out", type=str, default="motion_encoder.pt")
    ap.add_argument("--contact_threshold", type=float, default=0.5,
                    help="用于日志统计的二值阈值")
    ap.add_argument("--log_every", type=int, default=0,
                    help="训练过程中每隔多少 step 打印一次 batch 级别的快速统计（0 关闭）")
    ap.add_argument("--seed", type=int, default=2024,
                    help="随机种子（影响分割、采样、DataLoader shuffle）")
    ap.add_argument("--run_legacy_diag", action="store_true",
                    help="训练结束后额外运行旧版诊断流程（依赖隐式输入形态，默认关闭）")

    ap.add_argument('--p_event', type=float, default=0.30,
                        help='事件窗采样占比（0=纯均匀，1=全事件），建议 0.30')
    ap.add_argument('--event_thresh', type=float, default=0.50,
                        help='软证据阈值（L/R > 阈值 视为事件）')
    ap.add_argument('--event_min_gap', type=int, default=6,
                        help='事件间最小间隔（去抖）')
    ap.add_argument('--event_pre', type=int, default=-1,
                        help='事件窗起点相对事件帧的偏移（通常 -1）')

    args = ap.parse_args()

    if args.decode_use_amp and not args.decode_film_amp:
        print("[warn] --decode_use_amp currently supports仅 FiLM fusion; enabling --decode_film_amp.")
        args.decode_film_amp = True

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    _print_planA_banner()
    out_dir = os.path.dirname(os.path.abspath(args.out)) or os.getcwd()
    tpl_out = os.path.join(out_dir, "pretrain_template.json")
    _spec = _build_angvel_norm_spec(args.in_glob, save_path=tpl_out, pose_hist_len=args.pose_hist_len)
    base_ds = YAngvelContactsDataset(
        args.in_glob,
        T_w=args.T_w,
        require_zscore=args.require_zscore,
        norm_spec=_spec,
        p_event=args.p_event,
        event_thresh=args.event_thresh,
        event_min_gap=args.event_min_gap,
        event_pre=args.event_pre,
    )
    print(f"[ds] N={len(base_ds)} clips | J={base_ds.J} | input_dim={base_ds.input_dim} | pose_dim={base_ds.pose_target_dim} | K={args.K} | tpl={base_ds.tpl_path}")
    ds = base_ds  # 兼容后续诊断流程沿用原变量名

    train_ds = base_ds
    print("[split] 使用全量数据进行训练（无验证集）。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mixed_precision and device.type != "cuda":
        print("[warn] --mixed_precision specified but CUDA unavailable; falling back to FP32.")
    pin_memory = (device.type == "cuda")
    persistent_workers = bool(args.workers > 0)

    def _seed_worker(worker_id: int):
        worker_seed = (args.seed + worker_id) % (2**32 - 1)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_gen = torch.Generator()
    train_gen.manual_seed(args.seed)
    dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=_seed_worker if args.workers > 0 else None,
        generator=train_gen,
    )

    in_dim = base_ds.input_dim
    pose_dim = base_ds.pose_target_dim
    ang_dim = base_ds.J * 3
    period_latent_dim = int(args.period_dim)

    enc = MotionEncoder(
        input_dim=in_dim,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        num_layers=args.encoder_layers,
        dropout=args.encoder_dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    if args.amp_share_encoder:
        enc_amp = enc
    else:
        enc_amp = MotionEncoder(
            input_dim=in_dim,
            hidden_dim=args.hidden_dim,
            z_dim=args.z_dim,
            num_layers=args.encoder_layers,
            dropout=args.encoder_dropout,
            bidirectional=args.bidirectional,
        ).to(device)
    contact_head = StepHead(args.hidden_dim, args.K, bidirectional=args.bidirectional).to(device)
    period_head = PeriodHead(args.hidden_dim, period_latent_dim, bidirectional=args.bidirectional).to(device)
    amp_head = nn.Linear(args.hidden_dim, 1).to(device)

    dec_hidden = max(args.hidden_dim // 2, period_latent_dim)

    def _make_decoder(out_dim: int) -> nn.Sequential:
        return build_mlp(
            period_latent_dim,
            dec_hidden,
            num_layers=2,
            activation=nn.GELU,
            final_dim=out_dim,
        ).to(device)

    decoder_pose = _make_decoder(pose_dim)
    decoder_ang = _make_decoder(ang_dim)

    def _collect_trainable_params() -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        params.extend(enc.parameters())
        if enc_amp is not enc:
            params.extend(enc_amp.parameters())
        params.extend(contact_head.parameters())
        params.extend(period_head.parameters())
        params.extend(decoder_pose.parameters())
        params.extend(decoder_ang.parameters())
        params.extend(amp_head.parameters())
        return [p for p in params if p.requires_grad]

    opt = torch.optim.AdamW(_collect_trainable_params(), lr=args.lr)

    if args.resume:
        print(f"[resume] loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        def _safe_load(module: nn.Module, key: str) -> None:
            state = ckpt.get(key)
            if state is None:
                print(f"[resume] warning: key '{key}' not found in checkpoint.")
                return
            missing, unexpected = module.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[resume] warning: load {key} missing={missing} unexpected={unexpected}")

        _safe_load(enc, "encoder")
        if enc_amp is not enc:
            if "encoder_amp" in ckpt:
                _safe_load(enc_amp, "encoder_amp")
            else:
                _safe_load(enc_amp, "encoder")
        else:
            if "encoder_amp" in ckpt:
                _safe_load(enc, "encoder_amp")

        _safe_load(contact_head, "contact_head")
        _safe_load(period_head, "period_head")
        _safe_load(amp_head, "amp_head")
        _safe_load(decoder_pose, "decoder_pose")
        _safe_load(decoder_ang, "decoder_ang")
        print("[resume] checkpoint weights loaded.")
        opt = torch.optim.AdamW(_collect_trainable_params(), lr=args.lr)

    use_amp = bool(args.mixed_precision and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    thresh = float(args.contact_threshold)
    total_steps = max(1, len(dl))
    ang_norm_obj = train_ds.norm
    if not hasattr(ang_norm_obj, "s_eff") or ang_norm_obj.s_eff is None:
        raise AttributeError("AngvelNormalizer expected to expose 's_eff' (effective tanh scales)")

    amp_pairs = max(1, int(args.amp_pairs))
    contact_dim = getattr(train_ds, "contact_dim", 2)
    ang_feature_dim = train_ds.J * 3
    pose_hist_dim = getattr(train_ds, "pose_hist_dim", 0)
    contact_slice = slice(0, contact_dim)
    ang_slice = slice(contact_dim, contact_dim + ang_feature_dim)
    pose_hist_slice = None
    if pose_hist_dim > 0:
        pose_hist_slice = slice(contact_dim + ang_feature_dim, contact_dim + ang_feature_dim + pose_hist_dim)
    layout = InputSlices(contact=contact_slice, ang=ang_slice, pose_hist=pose_hist_slice)
    projectors = InputProjectors(
        layout,
        period_include_ang_sign=args.period_use_ang_sign,
        period_use_ang_features=bool(args.w_period_inv > 0),
        angnorm=train_ds.norm,
        amp_linear=args.amp_linear_equiv,
    )
    eps = 1e-6
    scale_min = float(args.amp_scale_min)
    scale_max = float(args.amp_scale_max)

    def _gather_state() -> dict:
        return {
            "encoder": enc.state_dict(),
            "encoder_amp": enc.state_dict() if enc_amp is enc else enc_amp.state_dict(),
            "period_head": period_head.state_dict(),
            "contact_head": contact_head.state_dict(),
            "amp_head": amp_head.state_dict(),
            "decoder_pose": decoder_pose.state_dict(),
            "decoder_ang": decoder_ang.state_dict(),
            "meta": {
                "input_dim": in_dim,
                "pose_dim": pose_dim,
                "ang_dim": ang_dim,
                "period_dim": period_latent_dim,
                "bidirectional": bool(args.bidirectional),
                "hidden_dim": args.hidden_dim,
                "z_dim": args.z_dim,
                "mlp_layers": args.encoder_layers,
                "mlp_dropout": args.encoder_dropout,
                "amp_share_encoder": bool(args.amp_share_encoder),
                "amp_linear_equiv": bool(args.amp_linear_equiv),
            },
        }

    def _save_checkpoint(path: Optional[str]) -> None:
        if not path:
            return
        try:
            torch.save(_gather_state(), path)
            print(f"[checkpoint] saved -> {path}")
        except Exception as err:
            print(f"[checkpoint] warning: failed to save {path}: {err}")

    best_metric = float("-inf")
    best_path = args.out_best
    if best_path is None or best_path == "":
        best_path = f"{args.out}.best.pt"

    def _capture_rng_states() -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        cpu_state = torch.get_rng_state()
        cuda_state = None
        if device.type == "cuda":
            cuda_state = torch.cuda.get_rng_state(device)
        return cpu_state, cuda_state

    def _restore_rng_states(cpu_state: torch.Tensor, cuda_state: Optional[torch.Tensor]) -> None:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device=device)

    def _relative_amp_targets(raw_ang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = raw_ang.shape
        mag = raw_ang.view(B, T, train_ds.J, 3).norm(dim=-1).mean(dim=-1)
        median = mag.median(dim=1, keepdim=True).values
        p10 = torch.quantile(mag, 0.10, dim=1, keepdim=True)
        base = torch.maximum(median, p10 + eps)
        rel = torch.log((mag / (base + eps)) + eps)
        return rel, mag

    margin = float(args.amp_margin)

    def _rank_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, float, int]:
        losses: list[torch.Tensor] = []
        correct = 0.0
        total = 0
        device_local = pred.device
        B, T = pred.shape
        if amp_pairs <= 0:
            return pred.new_tensor(0.0), 0.0, 0
        for b in range(B):
            idx = torch.arange(T, device=device_local)
            if idx.numel() < 2:
                continue
            pairs_sampled = 0
            attempts = 0
            while pairs_sampled < amp_pairs and attempts < amp_pairs * 6:
                i = idx[torch.randint(0, idx.numel(), (1,), device=device_local)]
                j = idx[torch.randint(0, idx.numel(), (1,), device=device_local)]
                attempts += 1
                if i == j:
                    continue
                diff_tgt = (target[b, i] - target[b, j]).item()
                if abs(diff_tgt) < margin:
                    continue
                label = 1.0 if diff_tgt > 0 else -1.0
                diff_pred = (pred[b, i] - pred[b, j])
                losses.append(torch.log1p(torch.exp(-label * diff_pred / args.amp_tau)))
                correct += float((diff_pred.detach().item() * label) > 0)
                total += 1
                pairs_sampled += 1
        if not losses:
            return pred.new_tensor(0.0), 0.0, 0
        return torch.stack(losses).mean(), correct / total, total

    for ep in range(1, args.epochs+1):
        enc.train()
        if enc_amp is not enc:
            enc_amp.train()
        contact_head.train()
        period_head.train()
        amp_head.train()
        decoder_pose.train()
        decoder_ang.train()
        train_meter = defaultdict(list)
        last_metrics = {}
        if args.amp_warmup_epochs and args.amp_warmup_epochs > 0:
            amp_warm = min(1.0, ep / float(args.amp_warmup_epochs))
        else:
            amp_warm = 1.0
        rank_correct_sum = 0.0
        rank_pair_sum = 0
        period_drift_sum = 0.0
        for step, batch in enumerate(dl, 1):
            inputs = batch["inputs"].to(device).float()
            contact_tgt = batch["contact_target"].to(device).float()
            pose_tgt = batch["pose_target"].to(device).float()
            ang_tgt_norm = batch["angvel_target"].to(device).float()
            ang_raw = batch.get("angvel_target_raw")
            if ang_raw is None:
                raise RuntimeError("Dataset must provide 'angvel_target_raw' for relative amplitude training.")
            ang_raw = ang_raw.to(device).float()
            period_hint = batch.get("period_hint")
            if period_hint is not None:
                period_hint = period_hint.to(device).float()

            opt.zero_grad(set_to_none=True)

            autocast_ctx = contextlib.nullcontext()
            if device.type == "cuda":
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)

            period_cpu_state, period_cuda_state = _capture_rng_states()
            with autocast_ctx:
                period_inputs = projectors.period(inputs)
                _, h_period = enc(period_inputs, return_summary=True)
                contact_logits = contact_head(h_period)
                soft_period = torch.tanh(period_head(h_period))

            amp_inputs = projectors.amp(
                inputs,
                shuffle_time=False,
                isolate_for_equiv=args.amp_linear_equiv,
            )
            amp_cpu_state, amp_cuda_state = _capture_rng_states()
            with autocast_ctx:
                # 等变/排序需要与基准帧对齐，这里禁用时间打乱
                _, h_amp_full = enc_amp(amp_inputs, return_summary=True)
            h_amp = h_amp_full.detach() if args.amp_detach else h_amp_full
            amp_pred = amp_head(h_amp).squeeze(-1)

            decode_features = projectors.decode(
                soft_period,
                amp_pred,
                use_amp=args.decode_use_amp,
                film=args.decode_film_amp,
            )
            with autocast_ctx:
                pose_pred = decoder_pose(decode_features)
                ang_pred_norm = decoder_ang(decode_features)

            loss_pose = F.mse_loss(pose_pred, pose_tgt)
            loss_contact = F.binary_cross_entropy_with_logits(contact_logits, contact_tgt)
            loss_hint = torch.zeros((), device=pose_pred.device)
            if period_hint is not None and period_hint.numel() > 0:
                hint_dim = period_hint.shape[-1]
                if soft_period.shape[-1] >= hint_dim:
                    loss_hint = F.mse_loss(soft_period[..., :hint_dim], period_hint)

            amp_target_rel, ang_mag = _relative_amp_targets(ang_raw)
            amp_pred_center = amp_pred - amp_pred.mean(dim=1, keepdim=True)
            amp_target_center = amp_target_rel - amp_target_rel.mean(dim=1, keepdim=True)
            loss_amp_rel = torch.mean(torch.abs(amp_pred_center - amp_target_center))
            loss_rank, rank_acc_batch, rank_pairs = _rank_loss_fn(amp_pred, amp_target_rel)

            loss_phase_energy = torch.zeros((), device=pose_pred.device)
            if args.w_phase_energy > 0:
                energy = soft_period.norm(dim=-1)
                penalty = F.relu(args.phase_energy_eps - energy)
                loss_phase_energy = penalty.mean()

            loss_equiv = torch.zeros((), device=pose_pred.device)
            loss_period_inv = torch.zeros((), device=pose_pred.device)
            slope_value = None
            period_drift_val = 0.0
            if args.w_amp_equiv > 0 or args.w_period_inv > 0:
                log_min = torch.log(torch.tensor(scale_min, device=device))
                log_max = torch.log(torch.tensor(scale_max, device=device))
                log_s = torch.empty(inputs.size(0), device=device).uniform_(float(log_min), float(log_max))
                scales = torch.exp(log_s)
                amp_inputs_scaled = projectors.amp(
                    inputs,
                    scales=scales,
                    shuffle_time=False,
                    isolate_for_equiv=args.amp_linear_equiv,
                )
                _restore_rng_states(amp_cpu_state, amp_cuda_state)
                with autocast_ctx:
                    _, h_amp_scaled_full = enc_amp(amp_inputs_scaled, return_summary=True)
                h_amp_scaled = h_amp_scaled_full.detach() if args.amp_detach else h_amp_scaled_full
                amp_scaled = amp_head(h_amp_scaled).squeeze(-1)
                inputs_scaled = inputs.clone()
                inputs_scaled[..., projectors.layout.ang] = amp_inputs_scaled[..., projectors.layout.ang]
                period_inputs_scaled = projectors.period(inputs_scaled)
                _restore_rng_states(period_cpu_state, period_cuda_state)
                with autocast_ctx:
                    _, h_period_scaled = enc(period_inputs_scaled, return_summary=True)
                soft_period_scaled = torch.tanh(period_head(h_period_scaled))

                log_s = log_s.view(-1, 1)
                log_s_expanded = log_s.expand_as(amp_pred)
                if args.w_amp_equiv > 0:
                    diff_amp = amp_scaled - amp_pred
                    loss_equiv = F.mse_loss(diff_amp, log_s_expanded)
                    slope_batch = (diff_amp.detach()) / (log_s_expanded + eps)
                    slope_value = float(slope_batch.mean().cpu())
                if args.w_period_inv > 0:
                    loss_period_inv = F.mse_loss(soft_period_scaled, soft_period.detach())
                    period_drift_val = float(torch.mean(torch.abs(soft_period_scaled.detach() - soft_period.detach())).cpu())
            loss_total = (
                args.w_pose * loss_pose
                + args.w_contact * loss_contact
                + args.w_period_hint * loss_hint
                + amp_warm * (
                    args.w_amp_rank * loss_rank
                    + args.w_amp_rel * loss_amp_rel
                    + args.w_amp_equiv * loss_equiv
                )
                + args.w_period_inv * loss_period_inv
                + args.w_phase_energy * loss_phase_energy
            )

            if use_amp:
                scaler.scale(loss_total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                if enc_amp is not enc:
                    torch.nn.utils.clip_grad_norm_(enc_amp.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(contact_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(period_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(amp_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(decoder_pose.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(decoder_ang.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                if enc_amp is not enc:
                    torch.nn.utils.clip_grad_norm_(enc_amp.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(contact_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(period_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(amp_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(decoder_pose.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(decoder_ang.parameters(), 1.0)
                opt.step()

            with torch.no_grad():
                probs = torch.sigmoid(contact_logits.detach())
                bin_pred = (probs > thresh).float()
                bin_tgt = (contact_tgt > thresh).float()
                acc = (bin_pred == bin_tgt).float().mean().item()
                mae_pose = torch.mean(torch.abs(pose_pred.detach() - pose_tgt)).item()
                amp_rel_mae = float(loss_amp_rel.detach().cpu())
                batch_metrics = {
                    "loss": float(loss_total.detach().cpu()),
                    "loss_pose": float(loss_pose.detach().cpu()),
                    "loss_contact": float(loss_contact.detach().cpu()),
                    "loss_hint": float(loss_hint.detach().cpu()),
                    "loss_amp_rank": float(loss_rank.detach().cpu()),
                    "loss_amp_rel": amp_rel_mae,
                    "loss_amp_equiv": float(loss_equiv.detach().cpu()),
                    "loss_period_inv": float(loss_period_inv.detach().cpu()),
                    "loss_phase_energy": float(loss_phase_energy.detach().cpu()) if args.w_phase_energy > 0 else None,
                    "amp_warm": amp_warm,
                    "contact_acc": acc,
                    "pose_mae": mae_pose,
                    "rank_acc": rank_acc_batch if rank_pairs > 0 else None,
                    "scale_slope": slope_value,
                    "period_drift": period_drift_val,
                    "pred_on_rate": float(probs.mean().detach().cpu()),
                    "tgt_on_rate": float(contact_tgt.mean().detach().cpu()),
                }
                _collect_metrics(train_meter, batch_metrics)
                last_metrics = batch_metrics

            rank_correct_sum += rank_acc_batch * rank_pairs
            rank_pair_sum += rank_pairs

            if args.log_every and (step % args.log_every == 0 or step == 1):
                m = last_metrics
                extras = []
                if m.get("rank_acc") is not None:
                    extras.append(f"rank_acc={m['rank_acc']:.3f}")
                if m.get("scale_slope") is not None:
                    extras.append(f"scale_slope={m['scale_slope']:.3f}")
                extras.append(f"amp_warm={amp_warm:.2f}")
                dbg_msg = (
                    f"[Train][ep {ep} step {step}/{total_steps}] "
                    f"loss={m['loss']:.4f} | pose={m['loss_pose']:.4f} "
                    f"| amp_rank={m['loss_amp_rank']:.4f} | amp_rel={m['loss_amp_rel']:.4f} "
                    f"| contact={m['loss_contact']:.4f} | hint={m['loss_hint']:.4f} "
                    f"| contact_acc={m['contact_acc']:.4f} | pred_on={m['pred_on_rate']:.4f}"
                )
                if extras:
                    dbg_msg += " | " + " | ".join(extras)
                print(dbg_msg)

        train_summary = _summarize_metrics(train_meter)
        if rank_pair_sum > 0:
            train_summary["rank_acc"] = train_summary.get("rank_acc", 0.0)
        print(_format_metrics("[Train]", ep, train_summary))

        current_slope = train_summary.get("scale_slope")
        if current_slope is not None and current_slope > best_metric:
            best_metric = current_slope
            if best_path:
                _save_checkpoint(best_path)
                print(f"[best] scale_slope improved to {current_slope:.4f}")

    enc.eval()
    if enc_amp is not enc:
        enc_amp.eval()
    contact_head.eval()
    period_head.eval()
    amp_head.eval()
    decoder_pose.eval()
    decoder_ang.eval()

    torch.save(_gather_state(), args.out)
    print(f"[OK] Saved MotionEncoder bundle -> {args.out} (D_in={in_dim}, period_dim={period_latent_dim}, bi={args.bidirectional})")

    # === 评估并打印汇总指标（无梯度） ===
    prev_train_flag = getattr(ds, "is_train", True)
    prev_yaw_aug = getattr(ds, "yaw_aug_deg", 0.0)
    prev_norm_c = getattr(ds, "normalize_c", True)
    ds.is_train = False
    ds.yaw_aug_deg = 0.0
    ds.normalize_c = prev_norm_c
    eval_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    _run_pretrain_eval_report(
        encoder=enc,
        encoder_amp=enc_amp,
        period_head=period_head,
        contact_head=contact_head,
        amp_head=amp_head,
                decoder_pose=decoder_pose,
        decoder_ang=decoder_ang,
        loader=eval_loader,
        device=device,
        norm_obj=ds.norm,
        pose_norm_obj=getattr(ds, 'pose_target_norm', None),
        contact_threshold=float(args.contact_threshold),
        dataset=ds,
        amp_pairs=amp_pairs,
        scale_min=scale_min,
        scale_max=scale_max,
        legacy_diag=args.run_legacy_diag,
        projectors=projectors,
        decode_use_amp=args.decode_use_amp,
        decode_film_amp=args.decode_film_amp,
    )

    if args.run_legacy_diag:
        try:
            diag_norm_roundtrip(ds, n_samples=3)
            probe, _ = train_linear_next_probe_all(
                encoder=enc, dl=dl, device=device, J=ds.J, norm_obj=ds.norm,
                epochs=3, lr=1e-3, max_batches=200
            )
            rollout_autoreg_probe(
                encoder=enc, probe=probe, ds=ds, device=device, J=ds.J, norm_obj=ds.norm,
                warmup=min(20, args.T_w//2), horizon=min(20, max(1, args.T_w//3)), nseq=3
            )
            motion_cues_diagnostics(encoder=enc, head_step=contact_head, ds=ds, device=device)
            run_probes_and_motion_diag(enc, contact_head, ds, device)
        except Exception as diag_err:
            print(f"[diag] skipped due to: {diag_err}")

    ds.is_train = prev_train_flag
    ds.yaw_aug_deg = prev_yaw_aug
    ds.normalize_c = prev_norm_c


def _run_pretrain_eval_report(
    encoder,
    encoder_amp,
    period_head,
    contact_head,
    amp_head,
        decoder_pose,
    decoder_ang,
    loader,
    device,
    norm_obj,
    pose_norm_obj,
    contact_threshold: float,
    dataset,
    amp_pairs: int,
    scale_min: float,
    scale_max: float,
    legacy_diag: bool,
    projectors: InputProjectors,
    decode_use_amp: bool,
    decode_film_amp: bool,
):
    import numpy as np
    import torch.nn.functional as F
    from collections import defaultdict

    encoder.eval()
    amp_encoder = encoder_amp if encoder_amp is not None else encoder
    amp_encoder.eval()
    period_head.eval()
    contact_head.eval()
    amp_head.eval()
    decoder_pose.eval()
    decoder_ang.eval()

    totals = defaultdict(float)
    eps = 1e-8
    contact_dim = getattr(dataset, "contact_dim", 2)
    ang_feature_dim = dataset.J * 3
    ang_slice = slice(contact_dim, contact_dim + ang_feature_dim)
    pose_hist_dim_eval = getattr(dataset, "pose_hist_dim", 0)

    amp_pred_list = []
    amp_target_list = []
    rank_correct = 0.0
    rank_total = 0
    scale_slopes = []
    period_drifts = []
    swing_pred_vals = []
    swing_gt_vals = []
    stance_pred_vals = []
    stance_gt_vals = []

    def _relative_amp_targets_eval(raw_ang_tensor: torch.Tensor):
        B, T, _ = raw_ang_tensor.shape
        mag = raw_ang_tensor.view(B, T, dataset.J, 3).norm(dim=-1).mean(dim=-1)
        median = mag.median(dim=1, keepdim=True).values
        p10 = torch.quantile(mag, 0.10, dim=1, keepdim=True)
        base = torch.maximum(median, p10 + eps)
        rel = torch.log((mag / (base + eps)) + eps)
        return rel, mag

    for batch in loader:
        if not isinstance(batch, dict):
            continue
        if any(k not in batch for k in ("inputs", "contact_target", "angvel_target", "pose_target", "angvel_target_raw")):
            continue

        inputs = batch["inputs"].to(device).float()
        contact_tgt = batch["contact_target"].to(device).float()
        ang_tgt_norm = batch["angvel_target"].to(device).float()
        pose_tgt = batch["pose_target"].to(device).float()
        ang_raw = batch["angvel_target_raw"].to(device).float()
        period_hint = batch.get("period_hint")
        if period_hint is not None:
            period_hint = period_hint.to(device).float()

        with torch.no_grad():
            period_inputs = projectors.period(inputs)
            _, h_period = encoder(period_inputs, return_summary=True)
            contact_logits = contact_head(h_period)
            soft_period = torch.tanh(period_head(h_period))

            amp_inputs = projectors.amp(inputs, isolate_for_equiv=projectors.amp_linear)
            _, h_amp = amp_encoder(amp_inputs, return_summary=True)
            amp_pred = amp_head(h_amp).squeeze(-1)

            decode_features = projectors.decode(
                soft_period,
                amp_pred,
                use_amp=decode_use_amp,
                film=decode_film_amp,
            )
            pose_pred = decoder_pose(decode_features)
            ang_pred_norm = decoder_ang(decode_features)

        B, T = inputs.shape[:2]
        totals["frames"] += B * T
        totals["pose_mse"] += F.mse_loss(pose_pred, pose_tgt, reduction="sum").item()
        totals["pose_mae"] += torch.abs(pose_pred - pose_tgt).sum().item()
        totals["pose_elem"] += pose_pred.numel()
        totals["ang_mse"] += F.mse_loss(ang_pred_norm, ang_tgt_norm, reduction="sum").item()
        totals["ang_mae"] += torch.abs(ang_pred_norm - ang_tgt_norm).sum().item()
        totals["ang_elem"] += ang_pred_norm.numel()

        probs = torch.sigmoid(contact_logits)
        bin_pred = (probs > contact_threshold).float()
        bin_tgt = (contact_tgt > contact_threshold).float()
        totals["contact_correct"] += (bin_pred == bin_tgt).float().sum().item()
        totals["contact_total"] += float(bin_tgt.numel())
        totals["pred_on_sum"] += probs.sum().item()
        totals["tgt_on_sum"] += contact_tgt.sum().item()

        if period_hint is not None and period_hint.numel() > 0:
            hint_dim = min(soft_period.shape[-1], period_hint.shape[-1])
            diff = soft_period[..., :hint_dim] - period_hint[..., :hint_dim]
            totals["period_hint_mse"] += torch.sum(diff * diff).item()
            totals["period_hint_elem"] += float(hint_dim * B * T)

        totals["period_abs_sum"] += torch.abs(soft_period).sum().item()
        totals["period_sq_sum"] += torch.square(soft_period).sum().item()
        totals["period_elem"] += float(soft_period.numel())

        amp_target_rel, ang_mag = _relative_amp_targets_eval(ang_raw)
        amp_pred_np = amp_pred.detach().cpu().numpy()
        amp_target_np = amp_target_rel.detach().cpu().numpy()
        amp_pred_list.append(amp_pred_np)
        amp_target_list.append(amp_target_np)

        rank_pairs = max(1, amp_pairs)
        for b in range(amp_pred_np.shape[0]):
            seq_len = amp_pred_np.shape[1]
            if seq_len < 2:
                continue
            samples = min(rank_pairs, seq_len * (seq_len - 1) // 2)
            attempts = 0
            drawn = 0
            while drawn < samples and attempts < samples * 6:
                i = np.random.randint(0, seq_len)
                j = np.random.randint(0, seq_len)
                attempts += 1
                if i == j:
                    continue
                diff_target = amp_target_np[b, i] - amp_target_np[b, j]
                if diff_target == 0.0:
                    continue
                diff_pred = amp_pred_np[b, i] - amp_pred_np[b, j]
                rank_correct += float(diff_pred * diff_target > 0)
                rank_total += 1
                drawn += 1

        if scale_max > scale_min:
            log_min = torch.log(torch.tensor(scale_min, device=device))
            log_max = torch.log(torch.tensor(scale_max, device=device))
            log_s = torch.empty(inputs.size(0), device=device).uniform_(float(log_min), float(log_max))
            scales = torch.exp(log_s)
        else:
            scales = torch.full((inputs.size(0),), scale_min, device=device)
            log_s = torch.log(scales + eps)
        amp_inputs_scaled = projectors.amp(
            inputs,
            scales=scales,
            shuffle_time=False,
            isolate_for_equiv=projectors.amp_linear,
        )
        log_s = log_s.view(-1, 1)
        with torch.no_grad():
            _, h_amp_scaled = amp_encoder(amp_inputs_scaled, return_summary=True)
            amp_scaled = amp_head(h_amp_scaled).squeeze(-1)
        log_s_expanded = log_s.expand_as(amp_pred)
        slope_tensor = ((amp_scaled - amp_pred) / (log_s_expanded + eps)).detach()
        slope_batch = slope_tensor.mean(dim=1).cpu().numpy()
        scale_slopes.append(slope_batch)
        soft_period_scaled = soft_period  # invariant by construction
        period_drifts.append(torch.mean(torch.abs(soft_period_scaled - soft_period)).item())

        if norm_obj is not None:
            ang_pred_raw_np = norm_obj.inverse_transform(ang_pred_norm.detach().cpu().numpy().reshape(-1, ang_pred_norm.shape[-1]))
            ang_pred_raw = torch.from_numpy(ang_pred_raw_np).to(device).view_as(ang_raw)
            totals["ang_raw_mae"] += torch.abs(ang_pred_raw - ang_raw).sum().item()
            totals["ang_raw_elem"] += float(ang_raw.numel())

        if pose_norm_obj is not None and "pose_target_raw" in batch:
            pose_target_raw = batch["pose_target_raw"].to(device).float()
            pose_pred_np = pose_pred.detach().cpu().numpy().reshape(-1, pose_pred.shape[-1])
            pose_pred_raw_np = pose_norm_obj.inverse_transform(pose_pred_np)
            pose_pred_raw = torch.from_numpy(pose_pred_raw_np).to(device).view_as(pose_target_raw)
            totals["pose_raw_mae"] += torch.abs(pose_pred_raw - pose_target_raw).sum().item()
            totals["pose_raw_elem"] += float(pose_target_raw.numel())

        contact_np = contact_tgt.detach().cpu().numpy()
        pred_ratio = np.exp(amp_pred_np)
        target_ratio = np.exp(amp_target_np)
        contact_mean = contact_np.mean(axis=-1)
        stance_mask = contact_mean > contact_threshold
        swing_mask = ~stance_mask
        if swing_mask.any():
            swing_pred_vals.append(pred_ratio[swing_mask])
            swing_gt_vals.append(target_ratio[swing_mask])
        if stance_mask.any():
            stance_pred_vals.append(pred_ratio[stance_mask])
            stance_gt_vals.append(target_ratio[stance_mask])

    def _safe_ratio(num, den):
        return float(num / den) if den > 0 else float("nan")

    pose_mse = _safe_ratio(totals["pose_mse"], totals["pose_elem"])
    pose_mae = _safe_ratio(totals["pose_mae"], totals["pose_elem"])
    ang_mse = _safe_ratio(totals["ang_mse"], totals["ang_elem"])
    ang_mae = _safe_ratio(totals["ang_mae"], totals["ang_elem"])
    contact_acc = _safe_ratio(totals["contact_correct"], totals["contact_total"])
    pred_on_rate = _safe_ratio(totals["pred_on_sum"], totals["contact_total"])
    tgt_on_rate = _safe_ratio(totals["tgt_on_sum"], totals["contact_total"])
    period_hint_mse = _safe_ratio(totals["period_hint_mse"], totals["period_hint_elem"])
    period_abs_mean = _safe_ratio(totals["period_abs_sum"], totals["period_elem"])
    period_rms = float(np.sqrt(_safe_ratio(totals["period_sq_sum"], totals["period_elem"]))) if totals["period_elem"] > 0 else float("nan")

    print(
        "[Eval] frames={:.0f} | pose_mse={:.4f} | pose_mae={:.4f} | "
        "ang_mse={:.4f} | ang_mae={:.4f} | contact_acc={:.4f} | "
        "pred_on={:.4f} | tgt_on={:.4f} | period_hint_mse={:.4f} | "
        "period_abs_mean={:.4f} | period_rms={:.4f}".format(
            totals["frames"],
            pose_mse,
            pose_mae,
            ang_mse,
            ang_mae,
            contact_acc,
            pred_on_rate,
            tgt_on_rate,
            period_hint_mse,
            period_abs_mean,
            period_rms,
        )
    )
    if legacy_diag and totals.get("pose_raw_elem", 0.0) > 0:
        print("[Eval] pose_raw_mae={:.4f}".format(_safe_ratio(totals["pose_raw_mae"], totals["pose_raw_elem"])))
    if legacy_diag and totals.get("ang_raw_elem", 0.0) > 0:
        print("[Eval] ang_raw_mae={:.4f}".format(_safe_ratio(totals["ang_raw_mae"], totals["ang_raw_elem"])))

    if amp_pred_list:
        amp_pred_all = np.concatenate([arr.reshape(-1) for arr in amp_pred_list], axis=0)
        amp_target_all = np.concatenate([arr.reshape(-1) for arr in amp_target_list], axis=0)
        rel_mae = float(np.mean(np.abs(amp_pred_all - amp_target_all)))
        rel_rmse = float(np.sqrt(np.mean((amp_pred_all - amp_target_all) ** 2)))
        if np.var(amp_pred_all) < eps or np.var(amp_target_all) < eps:
            rel_corr = float("nan")
        else:
            rel_corr = float(np.corrcoef(amp_pred_all, amp_target_all)[0, 1])
        rank_acc = rank_correct / rank_total if rank_total > 0 else float("nan")
        slope_mean = float(np.nanmean(np.concatenate(scale_slopes))) if scale_slopes else float("nan")
        period_drift = float(np.nanmean(period_drifts)) if period_drifts else float("nan")

        def _safe_mean(arr_list):
            if not arr_list:
                return float("nan")
            arr = np.concatenate(arr_list)
            return float(np.nanmean(arr)) if arr.size else float("nan")

        swing_pred_mean = _safe_mean(swing_pred_vals)
        swing_gt_mean = _safe_mean(swing_gt_vals)
        stance_pred_mean = _safe_mean(stance_pred_vals)
        stance_gt_mean = _safe_mean(stance_gt_vals)

        print(
            "[EvalExt] amp_rank_acc={:.3f} | rel_corr={:.3f} | rel_NMAE={:.3f} | "
            "rel_NRMSE={:.3f} | scale_slope={:.3f} | period_drift={:.4f}".format(
                rank_acc,
                rel_corr,
                rel_mae,
                rel_rmse,
                slope_mean,
                period_drift,
            )
        )
        print(
            "[EvalExt] swing_gt={:.3f} pred={:.3f} | stance_gt={:.3f} pred={:.3f}".format(
                swing_gt_mean,
                swing_pred_mean,
                stance_gt_mean,
                stance_pred_mean,
            )
        )

def probe_contact_decode_from_latent(encoder, ds, device, *, left_idxs, right_idxs, epochs=3, lr=1e-3):


    encoder.eval()
    H = None
    feats, labs = [], []

    nseq = min(20, len(ds))
    for i in range(nseq):
        Yin_norm, tgt, period = _split_sample(ds[i])
        T = Yin_norm.shape[0]
        with torch.no_grad():
            _, h_seq = encoder(Yin_norm.unsqueeze(0).to(device).float(), return_summary=True)
            h = h_seq[0]  # [T,H]
        if H is None: H = h.shape[-1]
        feats.append(h.detach().cpu())
        labs.append(torch.from_numpy(np.asarray(tgt, dtype=np.float32)))
    if not feats:
        print("[Probe-ContactDecode] no samples."); return {}
    X = torch.cat(feats, 0);  Y = torch.cat(labs, 0)
    X = X.to(device); Y = Y.to(device)

    head = nn.Linear(H, 2).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = head(X)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, Y)
        loss.backward(); opt.step()

    with torch.no_grad():
        P = torch.sigmoid(head(X)).cpu().numpy()
        Ynp = Y.cpu().numpy()

    Yin_speed, Y_gt = [], []
    J = ds.J
    for i in range(nseq):
        Yin_norm, tgt, period = _split_sample(ds[i])
        Yraw = ds.norm.inverse_transform(Yin_norm.numpy())
        Wmag = np.linalg.norm(Yraw.reshape(-1, J, 3), axis=-1)
        def _agg(idx_list):
            idxs = [k for k in idx_list if k >= 0]
            return Wmag[:, idxs].mean(axis=1) if idxs else np.full((Wmag.shape[0],), np.nan)
        sL = _agg(left_idxs); sR = _agg(right_idxs)
        s = np.stack([sL, sR], 1)
        Yin_speed.append(s); Y_gt.append(np.asarray(tgt, np.float32))
    S = np.concatenate(Yin_speed, 0); Ygt = np.concatenate(Y_gt, 0)

    def _linreg_score(x, y):
        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)
        A = torch.stack([x, torch.ones_like(x)], -1)
        try:
            w = torch.linalg.lstsq(A, y).solution
        except:
            w, _ = torch.lstsq(y.unsqueeze(-1), A); w = w.squeeze(-1)
        pred = (A @ w).squeeze(-1)
        return pred.detach().cpu().numpy()

    def _weighted_auc(scores, y_pos):
        s = np.asarray(scores, np.float64)
        w1 = np.asarray(y_pos,  np.float64)
        w0 = 1.0 - w1
        m = np.isfinite(s) & np.isfinite(w1)
        s, w1, w0 = s[m], w1[m], w0[m]
        W1, W0 = w1.sum(), w0.sum()
        if W1 < 1e-6 or W0 < 1e-6: return float("nan")
        idx = np.argsort(s); s, w1, w0 = s[idx], w1[idx], w0[idx]
        cum_w0 = np.cumsum(w0)
        auc = float((w1 * cum_w0).sum()) / float(W1 * W0 + 1e-9)
        return auc

    auc_lat_L = _weighted_auc(P[:,0], Ynp[:,0]); auc_lat_R = _weighted_auc(P[:,1], Ynp[:,1])
    pred_sL = _linreg_score(S[:,0], Ygt[:,0]);   pred_sR = _linreg_score(S[:,1], Ygt[:,1])
    auc_spd_L = _weighted_auc(pred_sL, Ygt[:,0]); auc_spd_R = _weighted_auc(pred_sR, Ygt[:,1])

    print(f"[Probe-ContactDecode] AUC latent(L/R) = {auc_lat_L:.3f}/{auc_lat_R:.3f} | speed-baseline(L/R) = {auc_spd_L:.3f}/{auc_spd_R:.3f}")
    return dict(auc_lat_L=auc_lat_L, auc_lat_R=auc_lat_R, auc_spd_L=auc_spd_L, auc_spd_R=auc_spd_R)




def probe_speed_nextstep_gain_from_latent(encoder, ds, device, *, left_idxs, right_idxs, epochs=3, lr=1e-3):
    encoder.eval()
    H = None
    X_lat, y_next = [], []
    X_spd, X_copy = [], []

    J = ds.J
    def _agg_speed(Wmag, idx_list):
        idxs = [k for k in idx_list if k >= 0]
        return Wmag[:, idxs].mean(axis=1) if idxs else np.full((Wmag.shape[0],), np.nan)

    nseq = min(20, len(ds))
    for i in range(nseq):
        Yin_norm, _, _ = _split_sample(ds[i])
        T = Yin_norm.shape[0]
        with torch.no_grad():
            _, h_seq = encoder(Yin_norm.unsqueeze(0).to(device).float(), return_summary=True)
            h = h_seq[0]  # [T,H]
        if H is None: H = h.shape[-1]

        Yraw = ds.norm.inverse_transform(Yin_norm.numpy())
        Wmag = np.linalg.norm(Yraw.reshape(T, J, 3), axis=-1)
        sL = _agg_speed(Wmag, left_idxs); sR = _agg_speed(Wmag, right_idxs)
        spd = np.nanmean(np.stack([sL, sR], 1), 1)

        if T < 3:
            continue
        h_t = h[:-1].detach().cpu()
        y_t1 = torch.from_numpy(spd[1:].astype(np.float32))
        spd_t = torch.from_numpy(spd[:-1].astype(np.float32))
        spd_tm1 = torch.from_numpy(np.r_[spd[:1], spd[:-1]].astype(np.float32)[:-1])

        X_lat.append(h_t)
        y_next.append(y_t1)
        X_spd.append(torch.stack([spd_t, spd_tm1], -1))
        X_copy.append(spd_t)

    if not X_lat:
        print("[Probe-NextSpeed] no samples."); return {}

    X_lat = torch.cat(X_lat, 0).to(device)
    y_next = torch.cat(y_next, 0).to(device)
    X_spd = torch.cat(X_spd, 0).to(device)
    X_copy = torch.cat(X_copy, 0).to(device)

    head = nn.Linear(H, 1).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = head(X_lat).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y_next)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred_lat = head(X_lat).squeeze(-1)

    head_b = nn.Linear(2, 1).to(device)
    opt_b  = torch.optim.Adam(head_b.parameters(), lr=lr)
    for _ in range(epochs):
        opt_b.zero_grad()
        pred = head_b(X_spd).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y_next)
        loss.backward(); opt_b.step()
    with torch.no_grad():
        pred_spd = head_b(X_spd).squeeze(-1)

    pred_copy = X_copy

    def _r2(y, yhat):
        y = y.detach().cpu().numpy(); yhat = yhat.detach().cpu().numpy()
        var = np.var(y)
        return float(1.0 - np.mean((y-yhat)**2) / (var + 1e-9))
    r2_lat  = _r2(y_next, pred_lat)
    r2_spd  = _r2(y_next, pred_spd)
    r2_copy = _r2(y_next, pred_copy)

    print(f"[Probe-NextSpeed] R2 latent={r2_lat:.3f} | speed-lin={r2_spd:.3f} | copy={r2_copy:.3f}")
    return dict(r2_lat=r2_lat, r2_spd=r2_spd, r2_copy=r2_copy)

if __name__ == "__main__":
    main()
