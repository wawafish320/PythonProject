from __future__ import annotations


# ===== Common Helpers (extracted) =====

# ========== [Unified Geometry Utilities] ==========
import torch
from typing import Any, Optional, Dict, Mapping, Sequence, Callable

from .eval_utils import FreeRunSettings, evaluate_teacher, evaluate_freerun


# ===== MPL Expert (embedded) =====
import torch.nn as nn
class MotionEncoder(nn.Module):
    """
    Stateless per-frame encoder that mirrors the Plan-A pretraining MLP.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        z_dim: int = 0,
        num_layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.z_dim = int(z_dim)

        layers: list[nn.Module] = []
        d_in = int(input_dim)
        for i in range(max(1, int(num_layers))):
            d_out = self.hidden_dim
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.GELU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            d_in = d_out
        self.mlp = nn.Sequential(*layers)
        self.summary_head = nn.Linear(self.hidden_dim, self.z_dim) if self.z_dim > 0 else None

    def forward(self, x: torch.Tensor, return_summary: Optional[bool] = None):
        """
        x: [B,T,D] or [T,D]; returns per-frame hidden states [B,T,H].
        When return_summary=True (or summary_head exists) also returns a pooled summary.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        enc = self.mlp(flat).reshape(B, T, self.hidden_dim)

        need_summary = return_summary if return_summary is not None else (self.summary_head is not None)
        if not need_summary:
            return enc

        summary_vec = enc.mean(dim=1)
        if self.summary_head is not None:
            summary_vec = self.summary_head(summary_vec)
        return summary_vec, enc


class PeriodHead(nn.Module):
    """
    Lightweight linear head used during pretraining to predict soft period hints.
    """
    def __init__(self, hidden_dim: int, out_dim: int, bidirectional: bool = False):
        super().__init__()
        self.fc = nn.Linear(int(hidden_dim), int(out_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)
# ===== /MPL Expert =====


def rot6d_to_matrix(xJ6: torch.Tensor, *, columns=("X","Z")) -> torch.Tensor:
    """
    xJ6: (..., J, 6)
      columns=("X","Z") 表示提供的是 X 列和 Z 列（列即轴，columns_are_axes），派生列按左手系规则补齐：
        remaining = second_column × first_column
      例如 X、Z 提供，则 Y = Z × X
    返回: (..., J, 3, 3)  列即轴
    """
    assert xJ6.shape[-1] == 6
    # 取两列
    a = xJ6[..., 0:3].clone()   # first column
    b = xJ6[..., 3:6].clone()   # second column

    # 归一化 + 正交化
    def _norm(v):
        return v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    ax1 = columns[0]  # "X" or "Y" or "Z"
    ax2 = columns[1]

    r = {}
    r[ax1] = _norm(a)
    # 让第二列与第一列正交
    b = b - (r[ax1] * b).sum(dim=-1, keepdim=True) * r[ax1]
    r[ax2] = _norm(b)

    # 派生列（左手）：remaining = second × first
    remaining = [ax for ax in ("X","Y","Z") if ax not in (ax1, ax2)][0]
    def _cross(u, v):  # torch.cross(u, v)
        return torch.cross(u, v, dim=-1)
    r[remaining] = _norm(_cross(r[ax2], r[ax1]))

    # 组装列为轴
    RX = r["X"].unsqueeze(-1)
    RY = r["Y"].unsqueeze(-1)
    RZ = r["Z"].unsqueeze(-1)
    R = torch.cat([RX, RY, RZ], dim=-1)  # (..., J, 3, 3)

    # 行列式修正：若 det<0，仅翻转“派生列”
    orig_shape = R.shape[:-2]  # (..., J)
    det = torch.linalg.det(R.reshape(-1, 3, 3)).reshape(orig_shape)
    neg = (det < 0.0).unsqueeze(-1)  # (..., J, 1)

    col_idx = {"X": 0, "Y": 1, "Z": 2}[remaining]
    R = R.clone()
    # R[..., :, col_idx] 选择的是“该列”的 3 个分量（行方向切片）
    R[..., :, col_idx] = torch.where(neg, -R[..., :, col_idx], R[..., :, col_idx])

    return R


def matrix_to_rot6d(R: torch.Tensor, *, columns=("X", "Z")) -> torch.Tensor:
    """
    逆变换：从 (..., J, 3, 3) 的旋转矩阵提取指定的两列，拼接成 (..., J, 6) 的 6D 表示。
    """
    if R.shape[-1] != 3 or R.shape[-2] != 3:
        raise ValueError("[matrix_to_rot6d] expects (..., 3, 3) matrices.")
    axis_idx = {"X": 0, "Y": 1, "Z": 2}
    col_a = axis_idx[columns[0]]
    col_b = axis_idx[columns[1]]
    a = R[..., :, col_a]
    b = R[..., :, col_b]
    return torch.cat([a, b], dim=-1)


def _rot6d_identity_like(residual: torch.Tensor, *, columns=("X", "Z")) -> torch.Tensor:
    """
    residual: (..., J, 6)
    返回与 residual 同形状的 6D 单位旋转表示，用于将 Δrot6d 残差偏移到单位附近。
    """
    axis_vec = {
        "X": (1.0, 0.0, 0.0),
        "Y": (0.0, 1.0, 0.0),
        "Z": (0.0, 0.0, 1.0),
    }
    if columns[0] not in axis_vec or columns[1] not in axis_vec:
        raise ValueError(f"[rot6d_identity] unsupported columns {columns}")
    first = residual.new_tensor(axis_vec[columns[0]])
    second = residual.new_tensor(axis_vec[columns[1]])
    base = residual.new_zeros(residual.shape)
    base[..., 0:3] = first
    base[..., 3:6] = second
    return base


def normalize_rot6d_delta(delta_flat: torch.Tensor, *, columns=("X", "Z")) -> torch.Tensor:
    """
    将模型输出的 Δrot6d（残差形式）转换为有效的 6D 旋转（接近单位旋转）。
    返回形状 (..., J, 6)。
    """
    if delta_flat.shape[-1] % 6 != 0:
        raise ValueError(f"[normalize_rot6d_delta] last dim {delta_flat.shape[-1]} not divisible by 6")
    orig = delta_flat.shape
    J = orig[-1] // 6
    residual = delta_flat.view(*orig[:-1], J, 6)
    delta_with_identity = residual + _rot6d_identity_like(residual, columns=columns)
    delta_flat = delta_with_identity.view(*orig[:-1], 6 * J)
    delta_proj = reproject_rot6d(delta_flat)
    return delta_proj.view(*orig[:-1], J, 6)


def compose_rot6d_delta(prev_rot6d: torch.Tensor, delta_rot6d: torch.Tensor, *, columns=("X", "Z")) -> torch.Tensor:
    """
    给定上一帧的绝对 6D 旋转向量 prev_rot6d 和模型预测的增量 Δrot6d，
    先将两者正交化生成矩阵，再执行 ΔR @ R_{t-1}，返回新的绝对 6D 旋转。
    """
    if prev_rot6d.shape != delta_rot6d.shape:
        raise ValueError(f"[compose_rot6d_delta] shape mismatch: prev={prev_rot6d.shape}, delta={delta_rot6d.shape}")
    if prev_rot6d.shape[-1] % 6 != 0:
        raise ValueError(f"[compose_rot6d_delta] last dim must be multiple of 6, got {prev_rot6d.shape[-1]}")
    orig_shape = prev_rot6d.shape
    J = orig_shape[-1] // 6
    prev = reproject_rot6d(prev_rot6d).view(*orig_shape[:-1], J, 6)
    delta = normalize_rot6d_delta(delta_rot6d, columns=columns)
    R_prev = rot6d_to_matrix(prev, columns=columns)
    R_delta = rot6d_to_matrix(delta, columns=columns)
    R_next = torch.matmul(R_delta, R_prev)
    return matrix_to_rot6d(R_next, columns=columns).view(orig_shape)


def infer_rot6d_delta_from_abs(rot6d_seq: torch.Tensor, *, columns=("X", "Z")) -> Optional[torch.Tensor]:
    """
    从绝对 rot6d 序列推导相邻帧 delta（同形状，首帧为单位旋转）。
    rot6d_seq: (..., T, J*6)
    返回: (..., T, J*6) delta 序列
    """
    if rot6d_seq is None:
        return None
    if rot6d_seq.dim() < 2:
        return None
    D = rot6d_seq.shape[-1]
    if D % 6 != 0:
        return None
    T = rot6d_seq.shape[-2]
    if T < 2:
        return None
    J = D // 6
    lead = rot6d_seq.shape[:-2]
    # 先重新投影到有效 rot6d，再展平成 (Bflat, T, J, 6)
    reproj = reproject_rot6d(rot6d_seq).contiguous()
    flat = reproj.reshape(-1, T, J, 6)
    mats = rot6d_to_matrix(flat, columns=columns)
    dR = torch.matmul(mats[:, 1:], mats[:, :-1].transpose(-1, -2))
    delta_body = matrix_to_rot6d(dR, columns=columns).reshape(flat.shape[0], T - 1, J * 6)
    delta_seq = rot6d_seq.new_zeros((flat.shape[0], T, J * 6))
    base = _rot6d_identity_like(flat[:, :1], columns=columns).reshape(flat.shape[0], 1, J * 6)
    delta_seq[:, :1] = base
    delta_seq[:, 1:] = delta_body
    return delta_seq.reshape(*lead, T, D)


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' rotation formula. axis: (...,3), angle: (...) in radians.
    Returns (...,3,3).
    """
    axis = F.normalize(axis, dim=-1, eps=1e-8)
    x, y, z = axis.unbind(dim=-1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one_minus = 1.0 - cos

    row0 = torch.stack([
        cos + x * x * one_minus,
        x * y * one_minus - z * sin,
        x * z * one_minus + y * sin,
    ], dim=-1)
    row1 = torch.stack([
        y * x * one_minus + z * sin,
        cos + y * y * one_minus,
        y * z * one_minus - x * sin,
    ], dim=-1)
    row2 = torch.stack([
        z * x * one_minus - y * sin,
        z * y * one_minus + x * sin,
        cos + z * z * one_minus,
    ], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

def geodesic_R(R_pred: torch.Tensor, R_gt: torch.Tensor, *, reduce=None) -> torch.Tensor:
    """
    Geodesic angular distance (radians) between two rotation matrices (...,J,3,3).
    Returns tensor shape (...,J) unless reduced.
    """
    assert R_pred.shape[-2:] == (3,3) and R_gt.shape[-2:] == (3,3)
    Rt = torch.matmul(R_pred.transpose(-1, -2), R_gt)  # (...,J,3,3)
    # Clamp trace for numerical stability
    trace = Rt[..., 0,0] + Rt[..., 1,1] + Rt[..., 2,2]
    cos = (trace - 1.) * 0.5
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    ang = torch.acos(cos)  # radians
    if reduce == "mean":
        return ang.mean()
    if reduce == "sum":
        return ang.sum()
    return ang

def _matrix_log_map(R: torch.Tensor) -> torch.Tensor:
    """Log map from SO(3) to so(3) as a 3D rotation vector."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    skew = R - R.transpose(-1, -2)
    vec = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1) * 0.5
    denom = (2.0 * sin_theta).unsqueeze(-1)
    factor = theta.unsqueeze(-1) / denom.clamp_min(1e-6)
    small = (sin_theta.abs() < 1e-4).unsqueeze(-1)
    approx = 0.5 * vec
    exact = vec * factor
    return torch.where(small, approx, exact)

def _root_relative_matrices(R: torch.Tensor, root_idx: int) -> torch.Tensor:
    """Express joint rotations in the root joint frame."""
    if root_idx < 0 or root_idx >= R.shape[-3]:
        return R
    R_root = R[..., root_idx, :, :]
    R_root_T = R_root.transpose(-1, -2).unsqueeze(-3)
    return torch.matmul(R_root_T, R)
# ========== [End Utilities] ==========


# ========== [End Loader] ==========
import math as _math

# === Unified layout entry parser (single source of truth) ===
from typing import Any, Optional

def parse_layout_entry(
    entry_value: Any,
    entry_name: Optional[str] = None,
    total_dim: Optional[int] = None
) -> Optional[slice]:
    """
    将布局条目解析为 slice，采用“严格 SIZE 语义”：
      - dict: {'start': s, 'size': k} / {'start': s, 'dim': k} / {'start': s, 'end': e}
      - list/tuple: [s, k]  ->  slice(s, s+k)   # 永远把第二项当 size
    不做任何启发式，不根据名称或 total_dim 改变语义。
    total_dim 仅用于（可选）越界检查（若提供）。
    """
    if entry_value is None:
        return None
    if isinstance(entry_value, slice):
        return entry_value
    # dict 形式
    if isinstance(entry_value, dict):
        if 'start' not in entry_value:
            return None
        s = int(entry_value['start'])
        if 'end' in entry_value:
            e = int(entry_value['end'])
            if (total_dim is not None) and not (0 <= s <= e <= total_dim):
                # 仅检查，不更改行为
                pass
            return slice(s, e)
        # 支持 size/dim 同义
        k = entry_value.get('size', entry_value.get('dim', None))
        if k is None:
            return None
        k = int(k)
        if (total_dim is not None) and not (0 <= s <= s+k <= total_dim):
            # 仅检查，不更改行为
            pass
        return slice(s, s + max(0, k))
    # list/tuple 形式：严格把第二项当 size
    if isinstance(entry_value, (list, tuple)) and len(entry_value) >= 2:
        s = int(entry_value[0])
        k = int(entry_value[1])
        if (total_dim is not None) and not (0 <= s <= s+k <= total_dim):
            # 仅检查，不更改行为
            pass
        return slice(s, s + max(0, k))
    return None



def _normalize_layout(raw_layout: Dict[str, Any], D: int) -> Dict[str, Tuple[int, int]]:
    """Normalize diverse span specs (start/size/end, tuple/list) -> (start, size).
    Strict checks: 0 <= start < end <= D.
    """
    norm = {}
    for name, meta in (raw_layout or {}).items():
        st = ed = sz = None
        if isinstance(meta, dict):
            if 'start' in meta and 'end' in meta:
                st = int(meta['start']); ed = int(meta['end']); sz = ed - st
            elif 'start' in meta and 'size' in meta:
                st = int(meta['start']); sz = int(meta['size']); ed = st + sz
        elif isinstance(meta, (list, tuple)) and len(meta) >= 2:
            # Heuristic: treat as (start,size); if looks like (start,end) with end>start and D known, still valid
            st = int(meta[0]); second = int(meta[1])
            if D is not None and second > st and second <= D:
                # ambiguous; prefer (start,size) since project uses [start,size], but accept (start,end) too
                # If interpreted as size and exceeds bound, switch to end
                as_size_ok = (st + second) <= D
                if as_size_ok:
                    sz = second; ed = st + sz
                else:
                    ed = second; sz = ed - st
            else:
                sz = second; ed = st + sz
        if st is None or (sz is None and ed is None):
            raise AssertionError(f"[FATAL] layout[{name}] cannot be normalized: {meta}")
        if ed is None: ed = st + sz
        if sz is None: sz = ed - st
        if not (0 <= st < ed <= D):
            raise AssertionError(f"[FATAL] layout[{name}] must be [start,end) with 0<=start<end<=D, got ({st},{ed}) and D={D}")
        norm[name] = (int(st), int(sz))
    return norm


def _layout_span(layout: Dict[str, Any], key: str) -> Optional[Tuple[int,int]]:
    sl = parse_layout_entry(layout.get(key), key)
    if sl is None:
        return None
    return (int(sl.start), int(sl.stop))


# ===== End Helpers =====
# =========================
# [STRICT] LayoutCenter (start,end) semantics only
# =========================
class LayoutCenter:
    """
    Single source of truth for:
      - meta.state_layout / meta.output_layout  (strictly [start,end))
      - y_to_x_map
      - MuX/StdX, MuY/StdY
      - optional: tanh_scales_* , fps, bone_names, rot6d_spec
    No fallback or guessing. Fail fast on missing or out-of-range.
    """
    def __init__(self, bundle_path: str):
        import json, numpy as np
        with open(bundle_path, "r", encoding="utf-8") as f:
            b = json.load(f)
        self.bundle = b
        self.np = np
        self.meta = b.get("meta", {}) or {}
        # required stats
        self.mu_x = np.asarray(b["MuX"], dtype=np.float32)
        self.std_x = np.asarray(b["StdX"], dtype=np.float32)
        self.mu_y = np.asarray(b["MuY"], dtype=np.float32)
        self.std_y = np.asarray(b["StdY"], dtype=np.float32)
        # raw layouts (strictly interpreted as [start,end))
        self.state_layout_raw  = dict(self.meta.get("state_layout", {}))
        self.output_layout_raw = dict(self.meta.get("output_layout", {}))
        # normalized to [start,size]
        self.state_layout  = None
        self.output_layout = None

        # optional
        self.y_to_x_map = list(self.meta.get("y_to_x_map", []))
        self.tanh_scales_rootvel = self.bundle.get("tanh_scales_rootvel", None)
        self.tanh_scales_angvel  = self.bundle.get("tanh_scales_angvel", None)
        self.fps = float(self.meta.get("fps", 60.0))
        self.bone_names = list(self.meta.get("bone_names", []))
        self.rot6d_spec = dict(self.meta.get("rot6d_spec", {}))
    def strict_validate(self, Dx: int, Dy: int):
        # 1) stats dimension
        assert self.mu_x.size == Dx and self.std_x.size == Dx, \
            f"[FATAL] MuX/StdX length ({self.mu_x.size}/{self.std_x.size}) != Dx({Dx})"
        assert self.mu_y.size == Dy and self.std_y.size == Dy, \
            f"[FATAL] MuY/StdY length ({self.mu_y.size}/{self.std_y.size}) != Dy({Dy})"
        # 2) normalize to [start,size] using strict [start,end) semantics
        self.state_layout  = _normalize_layout(self.state_layout_raw,  Dx)
        self.output_layout = _normalize_layout(self.output_layout_raw, Dy)
        # 3) required keys
        need_x = ("RootYaw","RootVelocity","BoneRotations6D","BoneAngularVelocities")
        need_y = ("BoneRotations6D",)
        for k in need_x:
            assert k in self.state_layout, f"[FATAL] meta.state_layout missing key: {k}"
        for k in need_y:
            assert k in self.output_layout, f"[FATAL] meta.output_layout missing key: {k}"
        # 4) ranges already validated above, but keep assertions
        for name, (st, sz) in self.state_layout.items():
            assert 0 <= st < Dx and 0 < sz <= Dx-st, f"[FATAL] state_layout[{name}] OOR: start={st} size={sz} Dx={Dx}"
        for name, (st, sz) in self.output_layout.items():
            assert 0 <= st < Dy and 0 < sz <= Dy-st, f"[FATAL] output_layout[{name}] OOR: start={st} size={sz} Dy={Dy}"

    def materialize_y_to_x_map(self):
        if self.y_to_x_map:
            return self.y_to_x_map
        y2x = []
        common = sorted(set(self.state_layout.keys()) & set(self.output_layout.keys()))
        for name in common:
            xs, xz = self.state_layout[name]
            ys, yz = self.output_layout[name]
            k = min(int(xz), int(yz))
            if k > 0:
                y2x.append({"name": name, "x_start": int(xs), "x_size": k, "y_start": int(ys), "y_size": k})
        return y2x

    def apply_to_dataset(self, ds):
        ds.state_layout  = dict(self.state_layout)
        ds.output_layout = dict(self.output_layout)
        if not hasattr(ds, "fps") or ds.fps is None:
            ds.fps = self.fps

    def apply_to_trainer(self, trainer):
        import torch, numpy as np
        trainer._x_layout = dict(self.state_layout)
        trainer._y_layout = dict(self.output_layout)
        trainer.y_to_x_map = self.materialize_y_to_x_map()
        # stats
        trainer.mu_x  = torch.tensor(np.asarray(self.mu_x).reshape(1, -1), dtype=torch.float32)
        trainer.std_x = torch.tensor(np.asarray(self.std_x).reshape(1, -1), dtype=torch.float32)
        trainer.mu_y  = torch.tensor(np.asarray(self.mu_y).reshape(1, -1), dtype=torch.float32)
        trainer.std_y = torch.tensor(np.asarray(self.std_y).reshape(1, -1), dtype=torch.float32)
        # tanh scales passthrough
        trainer.tanh_scales_rootvel = self.tanh_scales_rootvel
        trainer.tanh_scales_angvel  = self.tanh_scales_angvel
        # named slices
        trainer.yaw_slice       = parse_layout_entry(trainer._y_layout.get('RootYaw'), 'RootYaw')
        trainer.rootvel_slice   = parse_layout_entry(trainer._y_layout.get('RootVelocity'), 'RootVelocity')
        trainer.angvel_slice    = parse_layout_entry(trainer._y_layout.get('BoneAngularVelocities'), 'BoneAngularVelocities')
        trainer.yaw_x_slice     = parse_layout_entry(trainer._x_layout.get('RootYaw'), 'RootYaw')
        trainer.rootvel_x_slice = parse_layout_entry(trainer._x_layout.get('RootVelocity'), 'RootVelocity')
        trainer.angvel_x_slice  = parse_layout_entry(trainer._x_layout.get('BoneAngularVelocities'), 'BoneAngularVelocities')
        trainer.rootpos_x_slice = parse_layout_entry(trainer._x_layout.get('RootPosition'), 'RootPosition')
        # 👉 新增：明确注入 Rot6D 的 X/Y 切片（供 Y→X 写回用）
        trainer.rot6d_x_slice = parse_layout_entry(trainer._x_layout.get('BoneRotations6D'), 'BoneRotations6D')
        trainer.rot6d_y_slice = parse_layout_entry(trainer._y_layout.get('BoneRotations6D'), 'BoneRotations6D')

        # Forward axis hint for yaw reconstruction
        axis_map = {'X': 0, 'Y': 1, 'Z': 2}
        columns = []
        if isinstance(self.rot6d_spec, dict):
            cols_val = self.rot6d_spec.get('columns')
            if isinstance(cols_val, (list, tuple)):
                columns = [str(c).upper() for c in cols_val]
        forward_hint = columns[1] if len(columns) > 1 else (columns[0] if columns else 'Z')
        trainer.yaw_forward_axis = axis_map.get(forward_hint, getattr(trainer, 'yaw_forward_axis', 2))


def __apply_layout_center(ds_train, trainer):
    bundle_path = _arg('bundle_json', None) if '_arg' in globals() else None
    if not bundle_path:
        raise SystemExit("[FATAL] 需要 --bundle_json 指定集中化的归一化与布局模板（不再支持猜测/回退）")

    # 1) 加载并校验
    _center = LayoutCenter(bundle_path)
    _center.strict_validate(int(ds_train.Dx), int(ds_train.Dy))

    # 落到数据集（fps/布局等）
    _center.apply_to_dataset(ds_train)
    # 2) 先把布局/切片/统计等“落到 trainer 身上”
    _center.apply_to_trainer(trainer)

    # 显式设置 Rot6D 的 X/Y 切片，供 Y→X 写回使用（避免依赖隐式回退）
    trainer.rot6d_x_slice = parse_layout_entry(trainer._x_layout.get('BoneRotations6D'), 'BoneRotations6D')
    trainer.rot6d_y_slice = parse_layout_entry(trainer._y_layout.get('BoneRotations6D'), 'BoneRotations6D')

    # 保存 bundle meta（验证用；不影响训练）
    try:
        trainer._bundle_meta = dict(_center.meta)
    except Exception:
        trainer._bundle_meta = {}

    # 骨骼名（若数据集提供）
    if getattr(trainer, '_bone_names', None) is None and getattr(ds_train, 'bone_names', None):
        try:
            trainer._bone_names = list(ds_train.bone_names)
        except Exception:
            pass

    # 3) 注入 normalizer（关键：放在 apply_to_trainer 之后，保证切片已存在）
    trainer.normalizer = DataNormalizer(
        mu_x=_center.mu_x, std_x=_center.std_x,
        mu_y=_center.mu_y, std_y=_center.std_y,
        y_to_x_map=_center.materialize_y_to_x_map(),
        yaw_x_slice=trainer.yaw_x_slice,   yaw_y_slice=trainer.yaw_slice,
        rootvel_x_slice=trainer.rootvel_x_slice, rootvel_y_slice=trainer.rootvel_slice,
        angvel_x_slice=trainer.angvel_x_slice,   angvel_y_slice=trainer.angvel_slice,
        tanh_scales_rootvel=_center.tanh_scales_rootvel,
        tanh_scales_angvel=_center.tanh_scales_angvel,
        angvel_mode=getattr(ds_train, 'angvel_norm_mode', None),
        angvel_mu=getattr(ds_train, 'angvel_mu', None),
        angvel_std=getattr(ds_train, 'angvel_std', None),
    )

    # 4) 诊断输出（保持你原来的格式）
    xlay = getattr(trainer, "_x_layout", getattr(ds_train, "state_layout", None))
    ylay = getattr(trainer, "_y_layout", getattr(ds_train, "output_layout", None))

    yaw    = parse_layout_entry(ylay.get('RootYaw'), 'RootYaw')
    rootv  = parse_layout_entry(ylay.get('RootVelocity'), 'RootVelocity')
    angv   = parse_layout_entry(ylay.get('BoneAngularVelocities'), 'BoneAngularVelocities')

    yaw_x   = parse_layout_entry(xlay.get('RootYaw'), 'RootYaw')
    rootv_x = parse_layout_entry(xlay.get('RootVelocity'), 'RootVelocity')
    angv_x  = parse_layout_entry(xlay.get('BoneAngularVelocities'), 'BoneAngularVelocities')

    rot6d_x_span = _layout_span(xlay, 'BoneRotations6D')
    rot6d_y_span = _layout_span(ylay, 'BoneRotations6D')

    mx = getattr(trainer, 'mu_x', None); sx = getattr(trainer, 'std_x', None)
    my = getattr(trainer, 'mu_y', None); sy = getattr(trainer, 'std_y', None)
    len_mx = int(mx.numel()) if hasattr(mx, 'numel') else (len(mx) if mx is not None else 0)
    len_sx = int(sx.numel()) if hasattr(sx, 'numel') else (len(sx) if sx is not None else 0)
    len_my = int(my.numel()) if hasattr(my, 'numel') else (len(my) if my is not None else 0)
    len_sy = int(sy.numel()) if hasattr(sy, 'numel') else (len(sy) if sy is not None else 0)

from torch import nn

# ==== ARPG-PATCH: angular velocity helpers (eval-only) ====
import torch
import math as _math

def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """
    SO(3) log map. R: [...,3,3] -> axis-angle vector phi [...,3] (radians * axis)
    """
    tr = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) * 0.5
    tr = torch.clamp(tr, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(tr)
    W = R - R.transpose(-1, -2)
    vee = torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], dim=-1) * 0.5
    sin_theta = torch.sin(theta)
    k = theta / torch.clamp(2.0 * sin_theta, min=1e-6)
    k = k.unsqueeze(-1)
    phi = k * vee
    small = (theta < 1e-5).unsqueeze(-1)
    phi = torch.where(small, 0.5 * vee, phi)
    return phi

def angvel_vec_from_R_seq(R_seq: torch.Tensor, fps: float) -> torch.Tensor:
    """
    R_seq: [B,T,J,3,3] GS'ed rotation matrices
    Returns omega: [B,T-1,J,3] in rad/s
    """
    dR = torch.matmul(R_seq[:, 1:], R_seq[:, :-1].transpose(-1, -2))
    phi = so3_log_map(dR)
    omega = phi * float(fps)
    return omega

import torch
import torch.nn.functional as F
import numpy as np


def validate_and_fix_model_(m: nn.Module, Dx: int | None = None, Dc: int | None = None, *, reinit_on_nonfinite: bool = True) -> None:
    """
    Single entrypoint for production sanity:
    - Optionally checks first Linear's in_features against Dx+Dc when both are provided.
    - Scans parameters; if any non-finite, re-initializes that leaf module once.
    - Asserts all params are finite at the end.
    Side effects: may modify `m` in-place.
    """
    # 1) optional in_features check on the first Linear we encounter
    if Dx is not None and Dc is not None:
        first_linear_in = None
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                first_linear_in = mod.in_features
                break
        if first_linear_in is not None and first_linear_in != (Dx + Dc):
            raise RuntimeError(f"First Linear in_features={first_linear_in} != Dx+Dc={Dx+Dc}")

    # 2) re-init leaf modules that contain non-finite params
    def _reinit_module_(mod: nn.Module) -> None:
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_uniform_(mod.weight, a=_math.sqrt(5))
            if getattr(mod, 'bias', None) is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(mod.weight)
                bound = 1.0 / (_math.sqrt(max(fan_in, 1)))
                nn.init.uniform_(mod.bias, -bound, bound)
        elif hasattr(mod, 'reset_parameters'):
            try:
                mod.reset_parameters()  # type: ignore[attr-defined]
            except Exception:
                pass


    with torch.no_grad():
        for name, mod in m.named_modules():
            has_bad = False
            for pname, p in mod.named_parameters(recurse=False):
                if not torch.isfinite(p).all():
                    has_bad = True
                    break
            if has_bad and reinit_on_nonfinite:
                _reinit_module_(mod)

        # final assert
        for name, mod in m.named_modules():
            for pname, p in mod.named_parameters(recurse=False):
                if not torch.isfinite(p).all():
                    raise RuntimeError(f"param still non-finite after reinit: {name}.{pname}")


import sys
def _fix_firstdim_any(v, L: int):
    """
    Recursively pad/truncate so that the FIRST dimension equals L for tensors/ndarrays,
    and apply the same fix inside dict/list/tuple. Scalars are kept untouched.
    NOTE: numpy arrays with non-numeric dtype (e.g., strings '<U..' or object) are converted to lists
    to avoid PyTorch default_collate errors.
    """

    # PyTorch tensor
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
    # Numpy ndarray
    if isinstance(v, np.ndarray):
        # If dtype is non-numeric (strings, object), DO NOT collate as ndarray -> convert to list
        if v.dtype.kind in ('U', 'S', 'O'):
            return v.tolist()
        # numeric arrays: pad/truncate by first dimension
        if v.ndim >= 1:
            n = v.shape[0]
            if n == L:
                return v
            if n > L:
                return v[:L]
            pad = np.repeat(v[-1:, ...], L - n, axis=0)
            return np.concatenate([v, pad], axis=0)
        return v
    # Dict: recurse
    if isinstance(v, dict):
        return {k: _fix_firstdim_any(vv, L) for k, vv in v.items()}
    # Sequence: recurse
    if isinstance(v, (list, tuple)):
        seq = [_fix_firstdim_any(x, L) for x in v]
        return type(v)(seq)
    # Other scalars/strings: return as-is
    return v





def make_fixedlen_collate(seq_len: int):
    from torch.utils.data._utils.collate import default_collate as _default_collate


    DROP_KEYS = {'json_path', 'npz_path', 'foot_json_path', 'source_json', 'json_candidates'}

    def _collate(batch):
        fixed = []
        for sample in batch:
            s = _fix_firstdim_any(sample, seq_len)

            for k in list(s.keys()):
                if k in DROP_KEYS:
                    s.pop(k, None)

            for k, v in list(s.items()):
                if isinstance(v, np.ndarray) and v.dtype.kind in ('U', 'S', 'O'):
                    s[k] = v.tolist()

            fixed.append(s)
        return _default_collate(fixed)

    return _collate




def _arg(name, default=None):
    g = globals().get('GLOBAL_ARGS', None)
    try:
        return getattr(g, name)
    except Exception:
        return default



class _CondFiLM(nn.Module):

    def __init__(self, cond_dim: int, hidden_dim: int, film_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, film_dim * 2)

    def forward(self, cond: torch.Tensor):
        h = torch.nn.functional.gelu(self.fc1(cond))
        g, b = self.fc2(h).chunk(2, dim=-1)
        g = 1.0 + 0.5 * torch.tanh(g)
        b = 0.5 * torch.tanh(b)
        return (g, b)

import os, json, math, glob, time, argparse
from dataclasses import dataclass

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, Optional

try:
    from .cli_utils import (
        expand_paths_from_specs,
        get_flag_value_from_argv,
        get_flag_values_from_argv,
    )
except ImportError:
    from cli_utils import (
        expand_paths_from_specs,
        get_flag_value_from_argv,
        get_flag_values_from_argv,
    )
try:
    from tqdm import tqdm
except ImportError:
    print('Warning: tqdm not found. For a progress bar, run: pip install tqdm')

    def tqdm(iterable, *GLOBAL_ARGS, **kwargs):
        return iterable

def reproject_rot6d(flat_6d: torch.Tensor) -> torch.Tensor:
    orig = flat_6d.shape                    # (..., D)
    D = orig[-1]
    if D % 6 != 0:
        raise ValueError(f"[reproject_rot6d] last dim={D} not divisible by 6")
    J = D // 6

    x = flat_6d.view(*orig[:-1], J, 6)      # (..., J, 6)
    b1 = F.normalize(x[..., :3], dim=-1, eps=1e-8)
    b2 = x[..., 3:]
    b2 = b2 - (b1 * b2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)

    y = torch.cat([b1, b2], dim=-1)        # (..., J, 6)
    return y.view(*orig[:-1], 6 * J)        # 还原到 (..., D)



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


def _npz_scalar_to_str(v) -> Optional[str]:
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    return v if isinstance(v, str) and v else None


def _load_soft_contacts_from_json(json_path: str) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("Frames")
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError(f"{json_path}: JSON missing/empty Frames.")
    sc = []
    for i, fr in enumerate(frames):
        fe = (fr.get("FootEvidence") or {})
        L = (fe.get("L") or {})
        R = (fe.get("R") or {})
        if "soft_contact_score" not in L or "soft_contact_score" not in R:
            raise ValueError(f"{json_path}: frame {i} missing soft_contact_score.")
        sc.append([float(L["soft_contact_score"]), float(R["soft_contact_score"])])
    return np.asarray(sc, dtype=np.float32)


def _wrap_to_pi_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _direction_yaw_from_array(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    try:
        a = np.asarray(arr)
    except Exception:
        return None
    if a.ndim != 2 or a.shape[1] < 2:
        return None
    # Prefer最后三维: [dir_x, dir_y, speed]
    dir_xy = None
    speed = None
    if a.shape[1] >= 3:
        cand = a[:, -3:-1]
        norms = np.linalg.norm(cand, axis=-1)
        if np.nanmedian(norms) > 1e-4:
            dir_xy = cand
            speed = a[:, -1]
    if dir_xy is None:
        cand = a[:, -2:]
        norms = np.linalg.norm(cand, axis=-1)
        if np.nanmedian(norms) > 1e-4:
            dir_xy = cand
    if dir_xy is None:
        return None
    yaw = np.arctan2(dir_xy[..., 1], dir_xy[..., 0])
    if speed is not None:
        try:
            mask = np.asarray(speed) > 1e-3
            if mask.shape == yaw.shape and mask.any():
                yaw = yaw[mask]
        except Exception:
            pass
    return yaw if yaw.size > 0 else None


def _velocity_yaw_from_array(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=np.float32)
    except Exception:
        return None
    if a.ndim != 2 or a.shape[1] < 2:
        return None
    xy = a[:, :2]
    speed = np.linalg.norm(xy, axis=1)
    mask = speed > 1e-4
    if not mask.any():
        return None
    yaw = np.arctan2(xy[:, 1], xy[:, 0])
    return yaw[mask]


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


class VectorTanhNormalizer:
    def __init__(self, scales: np.ndarray, mu: Optional[np.ndarray]=None, std: Optional[np.ndarray]=None):
        scales = np.asarray(scales, dtype=np.float32)
        if scales.ndim != 1:
            raise ValueError(f"scales must be 1-D, got {scales.shape}")
        self.scales = np.clip(scales, 1e-6, None)
        if mu is not None:
            mu = np.asarray(mu, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            if mu.shape != self.scales.shape or std.shape != self.scales.shape:
                raise ValueError("mu/std shape mismatch with scales.")
            self.mu = mu
            self.std = np.clip(std, 1e-6, None)
        else:
            self.mu, self.std = None, None

    def transform(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr.astype(np.float32, copy=False)
        X = np.tanh(arr / self.scales)
        if self.mu is not None and self.std is not None:
            X = (X - self.mu) / self.std
        return X.astype(np.float32, copy=False)


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
                src_json = _npz_scalar_to_str(clip.get('source_json'))
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

class EventMotionModel(nn.Module):
    """
    无状态动作生成模型：通过显式传入的历史缓冲而非隐式 hidden_state 建模。
    """

    def __init__(
        self,
        in_state_dim: int,
        out_motion_dim: int,
        cond_dim: int = 0,
        period_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        context_len: int = 32,
        use_layer_norm: bool = True,
        *,
        contact_dim: int = 0,
        angvel_dim: int = 0,
        pose_hist_dim: int = 0,
    ):
        super().__init__()
        self.in_state_dim = int(in_state_dim)
        self.out_motion_dim = int(out_motion_dim)
        self.cond_dim = int(cond_dim)
        self.period_dim = int(period_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.context_len = int(context_len)
        self.contact_dim = max(0, int(contact_dim))
        self.angvel_dim = max(0, int(angvel_dim))
        self.pose_hist_dim = max(0, int(pose_hist_dim))
        self.encoder_input_dim = self.contact_dim + self.angvel_dim + self.pose_hist_dim

        input_dim = self.in_state_dim + self.cond_dim
        enc_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        self.shared_encoder = nn.Sequential(*enc_layers)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        self._pasa_heads = max(1, int(num_heads))
        if hidden_dim % self._pasa_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {self._pasa_heads}.")
        self._pasa_dhead = hidden_dim // self._pasa_heads
        self._pasa_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._pasa_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._pasa_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._pasa_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._pasa_lnq = nn.LayerNorm(hidden_dim)
        self._pasa_film = _CondFiLM(cond_dim=self.cond_dim, hidden_dim=128, film_dim=hidden_dim)
        self.coupling_norm = nn.LayerNorm(hidden_dim)
        self.input_clip = 16.0

        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_motion_dim),
        )
        self.period_encoder = nn.Linear(self.period_dim, hidden_dim) if self.period_dim > 0 else None

        self.frozen_encoder: Optional[MotionEncoder] = None
        self.frozen_period_head: Optional[PeriodHead] = None
        self._encoder_meta: dict[str, Any] = {}
        self._frozen_hidden_dim: Optional[int] = None
        self.latent_bridge: Optional[nn.Module] = None

    def _target_device(self) -> torch.device:
        try:
            return next(self.motion_head.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def forward(
        self,
        state: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        contacts: Optional[torch.Tensor] = None,
        angvel: Optional[torch.Tensor] = None,
        pose_history: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        is_single = state.ndim == 2
        if is_single:
            state = state.unsqueeze(1)
            if cond is not None and cond.ndim == 2:
                cond = cond.unsqueeze(1)
            if contacts is not None and contacts.ndim == 2:
                contacts = contacts.unsqueeze(1)
            if angvel is not None and angvel.ndim == 2:
                angvel = angvel.unsqueeze(1)
            if pose_history is not None and pose_history.ndim == 2:
                pose_history = pose_history.unsqueeze(1)

        if cond is None and self.cond_dim > 0:
            cond = torch.zeros(state.shape[:-1] + (self.cond_dim,), device=state.device, dtype=state.dtype)
        if cond is not None and cond.ndim == 2 and state.ndim == 3:
            cond = cond.unsqueeze(1)

        device = state.device
        dtype = state.dtype
        if contacts is None and self.contact_dim > 0:
            contacts = torch.zeros(state.shape[:-1] + (self.contact_dim,), device=device, dtype=dtype)
        if angvel is None and self.angvel_dim > 0:
            angvel = torch.zeros(state.shape[:-1] + (self.angvel_dim,), device=device, dtype=dtype)
        if pose_history is None and self.pose_hist_dim > 0:
            pose_history = torch.zeros(state.shape[:-1] + (self.pose_hist_dim,), device=device, dtype=dtype)

        encoder_feats = []
        if contacts is not None and contacts.size(-1) > 0:
            encoder_feats.append(contacts)
        if angvel is not None and angvel.size(-1) > 0:
            encoder_feats.append(angvel)
        if pose_history is not None and pose_history.size(-1) > 0:
            encoder_feats.append(pose_history)
        encoder_input = torch.cat(encoder_feats, dim=-1) if encoder_feats else None

        x_inputs = [state]
        if cond is not None:
            x_inputs.append(cond)
        x = torch.cat(x_inputs, dim=-1)
        if not torch.isfinite(x).all():
            bad_mask = (~torch.isfinite(x))
            _nz = bad_mask.nonzero(as_tuple=False)
            bad = _nz[0].tolist() if _nz.numel() > 0 else []
            s_ok = bool(torch.isfinite(state).all())
            c_ok = True
            if cond is not None and cond.is_floating_point():
                c_ok = bool(torch.isfinite(cond).all())
            raise RuntimeError(f'[Guard] x to shared_encoder has non-finite at {bad} | state_finite={s_ok} cond_finite={c_ok}')
        _lin0 = self.shared_encoder[0]
        _w_ok = bool(torch.isfinite(_lin0.weight).all())
        _b_ok = _lin0.bias is None or bool(torch.isfinite(_lin0.bias).all())
        if not _w_ok or not _b_ok:
            raise RuntimeError('[Guard] shared_encoder.0 has non-finite params; refusing to auto-reinit in forward')
        x = torch.nan_to_num(x, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
        clip_val = float(getattr(self, 'input_clip', 16.0) or 16.0)
        x = x.clamp(-clip_val, clip_val)
        with torch.no_grad():
            for _idx, _mod in enumerate(self.shared_encoder):
                if isinstance(_mod, torch.nn.Linear):
                    _ok_w = bool(torch.isfinite(_mod.weight).all())
                    _ok_b = _mod.bias is None or bool(torch.isfinite(_mod.bias).all())
                    if not _ok_w or not _ok_b:
                        raise RuntimeError(f"[Guard] shared_encoder.{_idx} has non-finite params; refusing to auto-reinit in forward")

        lin0 = self.shared_encoder[0]
        act1 = self.shared_encoder[1]
        z0 = lin0(x)
        y1 = act1(z0)

        enc_hidden = None
        if (
            encoder_input is not None
            and self.frozen_encoder is not None
            and encoder_input.size(-1) == self.encoder_input_dim
        ):
            enc_hidden = self.frozen_encoder(encoder_input, return_summary=False)
            if isinstance(enc_hidden, tuple):
                enc_hidden = enc_hidden[-1]

        soft_period = None
        if enc_hidden is not None and self.frozen_period_head is not None:
            soft_period = torch.tanh(self.frozen_period_head(enc_hidden))
        if self.period_dim > 0 and self.period_encoder is not None and soft_period is not None:
            period_emb = self.period_encoder(soft_period)
            y1 = y1 + period_emb

        h = self.shared_encoder[2:](y1)
        h_temporal = h + self.residual_proj(x)
        h_temporal = torch.nan_to_num(h_temporal, nan=0.0, posinf=1000000.0, neginf=-1000000.0).clamp(-100.0, 100.0)

        B, Tq, H = h_temporal.shape
        Dh = self._pasa_dhead
        scale = 1.0 / _math.sqrt(max(1, Dh))
        cond_for_film = cond
        if cond_for_film is not None and cond_for_film.ndim == 3 and cond_for_film.size(1) > 1:
            cond_for_film = cond_for_film[:, -1]
        if cond_for_film is None or cond_for_film.ndim != 2:
            cond_in = torch.zeros(B, self.cond_dim, device=device, dtype=dtype)
        else:
            cond_in = cond_for_film
        g, b = self._pasa_film(cond_in)
        q_in = self._pasa_lnq(h_temporal)
        Q = self._pasa_q(q_in).view(B, Tq, self._pasa_heads, Dh).transpose(1, 2)
        K = self._pasa_k(h_temporal).view(B, Tq, self._pasa_heads, Dh).permute(0, 2, 1, 3)
        V = self._pasa_v(h_temporal).view(B, Tq, self._pasa_heads, Dh).permute(0, 2, 1, 3)
        attn = torch.softmax(Q * scale @ K.transpose(-1, -2), dim=-1)
        ctx = (attn @ V).transpose(1, 2).contiguous().view(B, Tq, -1)
        attn_out = self._pasa_o(ctx)
        h_final = self.coupling_norm((h_temporal + attn_out) * (1 + g).unsqueeze(1) + b.unsqueeze(1))

        hidden_out = h_final
        out = self.motion_head(h_final)
        if is_single:
            out = out.squeeze(1)
            hidden_out = hidden_out.squeeze(1)
            if soft_period is not None:
                soft_period = soft_period.squeeze(1)
        setattr(self, '_last_hidden_seq', hidden_out.detach())
        result = {
            'out': out,
            'delta': out,
            'attn': attn.mean(dim=1),
        }
        result['h_final'] = hidden_out
        if soft_period is not None:
            result['period_pred'] = soft_period
        return result

    def attach_motion_encoder(self, bundle, *, map_location: str | torch.device = 'cpu'):
        """
        加载并冻结预训练的 MotionEncoder + PeriodHead。
        """
        if isinstance(bundle, (str, os.PathLike)):
            payload = torch.load(bundle, map_location=map_location)
        else:
            payload = bundle
        if not isinstance(payload, dict):
            raise TypeError("MotionEncoder bundle must be a dict or path to a dict.")

        encoder_state = payload.get('encoder', None)
        if encoder_state is None:
            raise KeyError("Bundle missing 'encoder' state_dict.")
        meta = dict(payload.get('meta', {}))

        weight0 = encoder_state.get('mlp.0.weight')
        if weight0 is None:
            for key, val in encoder_state.items():
                if key.endswith('weight') and val.ndim == 2:
                    weight0 = val
                    break
        if weight0 is None:
            raise ValueError("Unable to infer MotionEncoder dimensions from state_dict.")
        input_dim = int(meta.get('input_dim', weight0.shape[1]))
        hidden_dim = int(meta.get('hidden_dim', weight0.shape[0]))
        z_dim = int(meta.get('z_dim', 0))
        num_layers = int(meta.get('mlp_layers', 3))
        dropout = float(meta.get('mlp_dropout', 0.0))
        encoder = MotionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bool(meta.get('bidirectional', False)),
        )
        encoder.load_state_dict(encoder_state)
        encoder.eval()
        encoder.requires_grad_(False)

        period_state = payload.get('period_head', None)
        if period_state is None:
            raise KeyError("Bundle missing 'period_head' state_dict.")
        period_dim = int(period_state['fc.weight'].shape[0])
        period_head = PeriodHead(hidden_dim, period_dim, bidirectional=bool(meta.get('bidirectional', False)))
        period_head.load_state_dict(period_state)
        period_head.eval()
        period_head.requires_grad_(False)

        if self.encoder_input_dim and self.encoder_input_dim != input_dim:
            raise ValueError(f"Encoder input dim mismatch: dataset={self.encoder_input_dim} vs bundle={input_dim}")
        self.encoder_input_dim = input_dim

        device = self._target_device()
        self.frozen_encoder = encoder.to(device)
        self.frozen_period_head = period_head.to(device)
        self._frozen_hidden_dim = hidden_dim
        self._encoder_meta = meta

        if self.period_dim != period_dim or self.period_encoder is None:
            self.period_dim = period_dim
            self.period_encoder = nn.Linear(self.period_dim, self.hidden_dim).to(device)

        if self.hidden_dim != hidden_dim:
            self.latent_bridge = nn.Linear(self.hidden_dim, hidden_dim).to(device)
        else:
            self.latent_bridge = nn.Identity()

        return meta

class MotionJointLoss(nn.Module):
    def __init__(
        self,
        w_attn_reg: float = 0.01,
        output_layout: Dict[str, Any] = None,
        fps: float = 60.0,
        rot6d_spec: Dict[str, Any] = None,
        w_rot_geo: float = 0.0,
        w_rot_ortho: float = 0.0,
        ignore_motion_groups: str = '',
        w_rot_delta: float = 1.0,
        w_rot_delta_root: float = 0.0,
        w_rot_log: float = 0.0,
        w_cond_yaw: float = 0.0,
        cond_yaw_min_speed: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
        w_fk_pos: float = 0.0,
        w_rot_local: float = 0.0,
    ):
        super().__init__()
        self.meta = dict(meta) if isinstance(meta, dict) else {}
        self.w_attn_reg = float(w_attn_reg)
        self.w_rot_geo = float(w_rot_geo)
        self.w_rot_ortho = float(w_rot_ortho)
        self.w_rot_delta = float(w_rot_delta)
        self.w_rot_delta_root = float(w_rot_delta_root)
        self.w_rot_log = float(w_rot_log)
        self.w_cond_yaw = float(w_cond_yaw)
        self.cond_yaw_min_speed = float(cond_yaw_min_speed)
        self.w_fk_pos = float(w_fk_pos)
        self.w_rot_local = float(w_rot_local)
        self.angvel_eps = 1e-6
        self.fps = float(fps)
        self.output_layout = output_layout or {}
        self.rot6d_spec = rot6d_spec or {}
        self._rot6d_columns = self._resolve_rot6d_columns(self.rot6d_spec)
        layout = self.output_layout or {}
        inner = layout.get('slices') if isinstance(layout.get('slices'), dict) else layout
        total_dim_hint = next((int(inner[k]) for k in ('output_dim','D','dim','size','total_dim') if isinstance(inner.get(k), int)), None)
        self.group_slices = {name: sl for name, sl in ((n, parse_layout_entry(v, n, total_dim_hint)) for n, v in inner.items()) if isinstance(name, str) and isinstance(sl, slice)}
        self.ignore_groups = [g.strip() for g in (ignore_motion_groups or '').split(',') if g.strip()]
        self.attn_lambda_local = getattr(self, 'attn_lambda_local', 0.02)
        self.attn_lambda_entropy = getattr(self, 'attn_lambda_entropy', 0.0)
        self._warned_bad_rot6d = False
        self.template_hint: Optional[str] = None
        self.bundle_hint: Optional[str] = None
        self._joint_weight_cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self.root_idx = 0
        self.bone_names: list[str] = []
        self.limb_monitor_names: list[str] = [
            'upperarm_l', 'lowerarm_l', 'hand_l',
            'upperarm_r', 'lowerarm_r', 'hand_r',
            'thigh_l', 'calf_l', 'foot_l',
            'thigh_r', 'calf_r', 'foot_r',
        ]
        self._limb_mask_cache: Optional[torch.Tensor] = None
        self._torso_mask_cache: Optional[torch.Tensor] = None
        self._limb_mask_joint_count: Optional[int] = None
        skeleton = self.meta.get('skeleton') if isinstance(self.meta, dict) else None
        self.parents: list[int] = []
        self._parents_tensor: Optional[torch.Tensor] = None
        self.bone_offsets: Optional[torch.Tensor] = None
        if isinstance(skeleton, dict):
            parents = skeleton.get('parents')
            if isinstance(parents, (list, tuple)):
                self.parents = [int(p) for p in parents]
                try:
                    self.root_idx = max(0, self.parents.index(-1))
                except ValueError:
                    self.root_idx = 0
            offsets = skeleton.get('ref_local_offsets_m')
            if isinstance(offsets, (list, tuple)):
                try:
                    self.bone_offsets = torch.as_tensor(offsets, dtype=torch.float32)
                except Exception:
                    self.bone_offsets = None
        self.has_fk = bool(self.parents and self.bone_offsets is not None)

    def _format_template_hint(self, prefix: str) -> str:
        hints: list[str] = []
        if isinstance(self.template_hint, str) and self.template_hint:
            hints.append(f"norm_template={self.template_hint}")
        if isinstance(self.bundle_hint, str) and self.bundle_hint:
            hints.append(f"bundle_json={self.bundle_hint}")
        if hints:
            return f"{prefix} ({', '.join(hints)})"
        return prefix

    @staticmethod
    def _resolve_rot6d_columns(spec: Optional[Dict[str, Any]]) -> tuple[str, str]:
        if isinstance(spec, dict):
            cols = spec.get('columns')
            if isinstance(cols, (list, tuple)) and len(cols) >= 2:
                a = str(cols[0]).strip().upper()
                b = str(cols[1]).strip().upper()
                if a in ("X", "Y", "Z") and b in ("X", "Y", "Z") and a != b:
                    return a, b
        return ("X", "Z")

    def set_bone_names(self, names: Optional[Sequence[str]]) -> None:
        self.bone_names = [str(n) for n in (names or [])]
        self._limb_mask_cache = None
        self._torso_mask_cache = None
        self._limb_mask_joint_count = None
        # reset fk caches when bone count changes
        self._parents_tensor = None

    def set_skeleton(self, parents: Optional[Sequence[int]], offsets: Optional[Sequence[Sequence[float]]]) -> None:
        if parents is not None:
            self.parents = [int(p) for p in parents]
            self._parents_tensor = None
        if offsets is not None:
            try:
                self.bone_offsets = torch.as_tensor(offsets, dtype=torch.float32)
            except Exception:
                self.bone_offsets = None
        self.has_fk = bool(self.parents and self.bone_offsets is not None)

    def _resolve_limb_masks(self, joint_count: int, device) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        import torch
        if joint_count <= 0:
            return None
        names = self.bone_names
        if not names or joint_count > len(names):
            return None
        monitor = getattr(self, 'limb_monitor_names', None) or []
        if not monitor:
            return None
        idx_map = {name: idx for idx, name in enumerate(names[:joint_count])}
        limb_indices = [idx_map[name] for name in monitor if name in idx_map]
        if not limb_indices:
            return None
        if self._limb_mask_joint_count != joint_count or self._limb_mask_cache is None:
            mask = torch.zeros(joint_count, dtype=torch.bool)
            mask[limb_indices] = True
            self._limb_mask_cache = mask
            self._torso_mask_cache = (~mask).clone()
            self._limb_mask_joint_count = joint_count
        limb_mask = self._limb_mask_cache.to(device=device)
        torso_mask = self._torso_mask_cache.to(device=device)
        if not torso_mask.any():
            torso_mask = (~limb_mask).clone()
            if torso_mask.numel() == 0 or not torso_mask.any():
                return None
        return limb_mask, torso_mask

    def _parent_relative_matrices(self, R: torch.Tensor) -> torch.Tensor:
        import torch
        parents = getattr(self, 'parents', None)
        if not parents:
            return R
        if not torch.is_tensor(R):
            return R
        J = R.shape[-3]
        if len(parents) < J:
            return R
        parents_tensor = getattr(self, '_parents_tensor', None)
        if parents_tensor is None or parents_tensor.device != R.device or parents_tensor.numel() < J:
            parents_tensor = torch.as_tensor(parents[:J], device=R.device, dtype=torch.long)
            self._parents_tensor = parents_tensor
        else:
            parents_tensor = parents_tensor[:J]
        R_rel = torch.empty_like(R)
        for j in range(J):
            p = int(parents_tensor[j].item())
            if p < 0 or p >= J:
                R_rel[..., j, :, :] = R[..., j, :, :]
                continue
            parent = R[..., p, :, :]
            child = R[..., j, :, :]
            R_rel[..., j, :, :] = torch.matmul(parent.transpose(-1, -2), child)
        return R_rel

    def _root_relative(self, R: torch.Tensor) -> torch.Tensor:
        root_idx = int(getattr(self, 'root_idx', 0))
        return _root_relative_matrices(R, root_idx)

    def _joint_weight_vector(self, device, dtype, joint_count: int) -> torch.Tensor:
        key = (str(device), str(dtype), int(joint_count))
        cache = getattr(self, '_joint_weight_cache', None)
        if cache is None:
            cache = {}
            self._joint_weight_cache = cache
        if key in cache:
            return cache[key]
        import torch
        weights = torch.ones(joint_count, device=device, dtype=dtype)
        cache[key] = weights
        return weights

    def _fk_positions(self, R_seq: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if R_seq is None:
            return None
        if not getattr(self, 'has_fk', False):
            return None
        import torch
        J = R_seq.shape[-3]
        if len(self.parents) < J or self.bone_offsets.shape[0] < J:
            return None
        device = R_seq.device
        dtype = R_seq.dtype
        parents_tensor = getattr(self, '_parents_tensor', None)
        if parents_tensor is None or parents_tensor.device != device or parents_tensor.numel() < J:
            parents_tensor = torch.as_tensor(self.parents[:J], device=device, dtype=torch.long)
            self._parents_tensor = parents_tensor
        else:
            parents_tensor = parents_tensor[:J]
        offsets = self.bone_offsets.to(device=device, dtype=dtype)
        B, T = R_seq.shape[:2]
        world_rot = torch.empty_like(R_seq)
        world_pos = torch.zeros(B, T, J, 3, device=device, dtype=dtype)
        for j in range(J):
            parent = int(parents_tensor[j].item())
            rot_j = R_seq[..., j, :, :]
            if parent < 0 or parent >= J:
                world_rot[..., j, :, :] = rot_j
                continue
            parent_rot = world_rot[..., parent, :, :].clone()
            world_rot[..., j, :, :] = torch.matmul(parent_rot, rot_j)
            offset = offsets[j].view(1, 1, 3, 1)
            delta = torch.matmul(parent_rot, offset).squeeze(-1)
            parent_pos = world_pos[..., parent, :].clone()
            world_pos[..., j, :] = parent_pos + delta
        return world_pos

    def _collect_limb_geo_stats(self, geo_tensor: torch.Tensor) -> Dict[str, float]:
        import torch, math
        if geo_tensor is None or geo_tensor.numel() == 0:
            return {}
        J = geo_tensor.shape[-1]
        masks = self._resolve_limb_masks(J, geo_tensor.device)
        if not masks:
            return {}
        limb_mask, torso_mask = masks
        flat = geo_tensor.reshape(-1, J)
        joint_mean = torch.nanmean(flat, dim=0)
        stats: Dict[str, float] = {}
        deg = 180.0 / math.pi
        limb_val = torso_val = None
        if limb_mask.any():
            limb_val = joint_mean[limb_mask].mean()
            stats['rot_geo_limb_deg'] = float((limb_val * deg).detach().cpu())
            stats['rot_geo_limb_count'] = int(limb_mask.sum().item())
        if torso_mask.any():
            torso_val = joint_mean[torso_mask].mean()
            stats['rot_geo_torso_deg'] = float((torso_val * deg).detach().cpu())
            stats['rot_geo_torso_count'] = int(torso_mask.sum().item())
        if limb_val is not None and torso_val is not None:
            ratio = limb_val / torso_val.clamp_min(1e-6)
            stats['rot_geo_limb_over_torso'] = float(ratio.detach().cpu())
        return stats
    def _forward_base_inner(self, pred_motion: torch.Tensor, gt_motion: torch.Tensor, attn_weights=None) -> tuple[torch.Tensor, dict[str, float]]:
        """
        参数:
            pred_motion: [B,T,D] or [T,D] or [B,D]
            gt_motion:   同形状
            attn_weights: None 或 [B,H,T,T]/[L,B,H,T,T] 或 list/tuple/dict 的任意嵌套
        返回:
            loss 标量, 分项 dict (float)
        """
        Z = lambda v: gt_motion.new_tensor(float(v))
        pm, gm = (pred_motion, gt_motion)
        assert pm.shape == gm.shape, f'pred/gt shape mismatch: {pm.shape} vs {gm.shape}'

        # === 其他辅助项 ===
        if attn_weights is not None:
            l_attn = self.compute_attention_regularization(attn_weights, geomask=None)
        else:
            l_attn = gm.new_zeros(())
        geo_details = None
        if self.w_rot_geo > 0:
            geo_payload = self.compute_rot6d_geo_loss(pm, gm, return_per_joint=True)
            if isinstance(geo_payload, tuple):
                l_geo, geo_details = geo_payload
            else:
                l_geo = geo_payload
        else:
            l_geo = Z(0.0)
        l_delta = Z(0.0)
        l_ortho = Z(0.0)
        loss = self.w_attn_reg * l_attn + self.w_rot_geo * l_geo
        stats = {
            'attn': float(l_attn.detach().cpu()),
            'rot_geo': float(l_geo.detach().cpu()),
            'rot_delta': 0.0,
            'rot_ortho': 0.0,
            'rot_ortho_raw': 0.0,
        }
        if geo_details is not None:
            limb_stats = self._collect_limb_geo_stats(geo_details.detach())
            if limb_stats:
                stats.update(limb_stats)
        return loss, stats

    def _slice_if_exists(self, name: str, X: torch.Tensor) -> Optional[torch.Tensor]:
        """
        从 self.group_slices 中获取预先解析好的 slice，并应用于张量。
        """
        sl = self.group_slices.get(name)

        # 因为 self.group_slices 只包含 slice 对象，所以只需做一次类型检查即可。
        if isinstance(sl, slice):
            return X[..., sl]

        return None

    @staticmethod
    def _build_ignore_mask(D: int, group_slices: Dict[str, slice], ignore_groups: list, device) -> torch.Tensor:
        """
        返回一个布尔 mask，True=参与计算，False=忽略。
        """
        mask = torch.ones(D, dtype=torch.bool, device=device)
        for g in ignore_groups:
            sl = group_slices.get(g, None)
            if sl is not None:
                mask[sl] = False
        return mask

    def compute_attention_regularization(self, attn_weights, geomask=None):
        """
        返回一个标量 loss：
        - 支持 Tensor: [B,H,T,T] 或 [L,B,H,T,T]
        - 支持 list/tuple/dict: 递归展开后逐个累加
        - geomask: None 或可广播到 [..., T, T] 的掩码（1=允许区域，0=不鼓励区域）
        """
        if attn_weights is None:
            if geomask is not None and torch.is_tensor(geomask):
                return torch.zeros((), device=geomask.device, dtype=geomask.dtype)
            return torch.tensor(0.0)

        def _flatten_items(x):
            if torch.is_tensor(x):
                return [x]
            if isinstance(x, (list, tuple)):
                items = []
                for y in x:
                    items.extend(_flatten_items(y))
                return items
            if isinstance(x, dict):
                items = []
                for y in x.values():
                    items.extend(_flatten_items(y))
                return items
            return []
        items = _flatten_items(attn_weights)
        if len(items) == 0:
            return torch.tensor(0.0, device=geomask.device if geomask is not None and torch.is_tensor(geomask) else None)
        loss_total = None
        count = 0
        for A in items:
            if A.dim() == 5:
                L, B, H, T, _ = A.shape
                A = A.reshape(L * B * H, T, T)
            elif A.dim() == 4:
                B, H, T, _ = A.shape
                A = A.reshape(B * H, T, T)
            elif A.dim() == 3:
                T = A.shape[-1]
            else:
                continue
            A = A.float()
            A = A / A.sum(-1, keepdim=True).clamp_min(1e-06)
            device = A.device
            if geomask is not None:
                try:
                    M = 1.0 - geomask
                    loss_local = (A * M).mean()
                except Exception:
                    gm = geomask
                    if torch.is_tensor(gm):
                        if gm.dim() == 2:
                            gm = gm.view(1, T, T)
                        elif gm.dim() == 4:
                            gm = gm.mean(0)
                        M = 1.0 - gm
                        loss_local = (A * M).mean()
                    else:
                        idx = torch.arange(T, device=device)
                        Dmat = (idx[None, :] - idx[:, None]).abs().float()
                        Dmat = Dmat / Dmat.max().clamp_min(1.0)
                        loss_local = (A * Dmat).mean()
            else:
                idx = torch.arange(T, device=device)
                Dmat = (idx[None, :] - idx[:, None]).abs().float()
                Dmat = Dmat / Dmat.max().clamp_min(1.0)
                loss_local = (A * Dmat).mean()
            Aeps = A.clamp_min(1e-06)
            entropy = -(Aeps * Aeps.log()).sum(-1).mean()
            loss_attn = self.attn_lambda_local * loss_local + self.attn_lambda_entropy * -entropy
            loss_total = loss_attn if loss_total is None else loss_total + loss_attn
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=items[0].device)
        return loss_total / count

    def _maybe_get_rot6d(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        """
        若存在 "BoneRotations6D" 切片，则返回该切片；否则 None。
        """
        rot = self._slice_if_exists('BoneRotations6D', X)
        return rot

    def compute_rot6d_geo_loss(self, pred: torch.Tensor, gt: torch.Tensor, *, return_per_joint: bool = False):
        Z = lambda v: gt.new_tensor(float(v))
        sl = self.group_slices.get('BoneRotations6D', None)

        # 1) 只取 rot6d 的扁平切片 (…, D)，不要先 reshape 到 (J,6)
        pr = self._maybe_get_rot6d(pred)  # (…, D) or None
        gr = self._maybe_get_rot6d(gt)  # (…, D) or None
        if pr is None or gr is None:
            return Z(0.0)

        D = pr.shape[-1]
        if D % 6 != 0:
            if not self._warned_bad_rot6d:
                self._warned_bad_rot6d = True
                print(
                    f"[Loss][WARN] BoneRotations6D slice dim={D} (not multiple of 6). slice={sl}, total_pred_D={pred.shape[-1]}. Skip rot6d_geo this run.")
            return Z(0.0)
        J = D // 6

        # 2) 训练端反归一化：在扁平 (…, D) 上做 raw = z*StdY + MuY
        try:
            sl_b = self.group_slices.get('BoneRotations6D', None)
            if isinstance(sl_b, slice) and getattr(self, "mu_y", None) is not None and getattr(self, "std_y",
                                                                                               None) is not None:
                st = int(sl_b.start);
                ln = int(sl_b.stop - sl_b.start)
                if ln == D:  # 只有当这段就是完整 rot6d 段时才生效
                    mu = torch.as_tensor(self.mu_y, device=pr.device, dtype=pr.dtype)[..., st:st + ln]
                    sd = torch.as_tensor(self.std_y, device=pr.device, dtype=pr.dtype)[..., st:st + ln].clamp(min=1e-6)
                    while mu.dim() < pr.dim():
                        mu = mu.unsqueeze(0);
                        sd = sd.unsqueeze(0)
                    pr = pr * sd + mu
                    gr = gr * sd + mu
                    if not hasattr(self, "_train_denorm_hit"):
                        print("[GeoLoss] TRAIN denorm(Y.rot6d) applied on flat D.")
                        self._train_denorm_hit = True

        except Exception:
            pass

        # 3) 先在扁平 (…, D) 上做 reproject，再 reshape 到 (…, J, 6)
        pr = reproject_rot6d(pr)  # (…, D)
        gr = reproject_rot6d(gr)  # (…, D)
        pr = pr.view(*pr.shape[:-1], J, 6)  # (…, J, 6)
        gr = gr.view(*gr.shape[:-1], J, 6)  # (…, J, 6)

        # 4) geodesic
        Rp = rot6d_to_matrix(pr)
        Rg = rot6d_to_matrix(gr)
        RtR = torch.matmul(Rp.transpose(-1, -2), Rg)
        tr = RtR[..., 0, 0] + RtR[..., 1, 1] + RtR[..., 2, 2]
        cos = (tr - 1.0) * 0.5
        cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.arccos(cos)
        if return_per_joint:
            return theta.mean(), theta
        return theta.mean()


    def compute_rot6d_ortho_loss(self, pred: torch.Tensor) -> torch.Tensor:

        """Ortho penalty on **raw 6D** (pre-GS):
        encourage unit-length columns and mutual orthogonality.
        This must NOT use rot6d_to_matrix (which orthonormalizes and would yield ~0 loss).
        """
        Z = lambda v: pred.new_tensor(float(v))
        pr = self._maybe_get_rot6d(pred)  # (..., D) or None
        if pr is None:
            return Z(0.0)
        D = pr.shape[-1]
        if D % 6 != 0:
            if not getattr(self, '_warned_bad_rot6d_ortho', False):
                self._warned_bad_rot6d_ortho = True
                print(f"[Loss][WARN] BoneRotations6D slice dim={D} not multiple of 6. Skip rot6d_ortho.")
            return Z(0.0)
        J = D // 6
        a6 = pr.view(*pr.shape[:-1], J, 6)  # (..., J, 6) raw 6D (no GS)
        v1 = a6[..., 0:3]
        v2 = a6[..., 3:6]
        len_p = (v1.norm(dim=-1) - 1.0).pow(2) + (v2.norm(dim=-1) - 1.0).pow(2)
        ortho_p = (v1.mul(v2).sum(dim=-1)).pow(2)
        return (len_p + ortho_p).mean()

    def _rot6d_matrices(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        rot6d = self._maybe_get_rot6d(X)
        if rot6d is None:
            return None
        D = rot6d.shape[-1]
        if D % 6 != 0:
            return None
        J = D // 6

        try:
            sl = self.group_slices.get('BoneRotations6D', None)
            if (
                isinstance(sl, slice)
                and getattr(self, 'mu_y', None) is not None
                and getattr(self, 'std_y', None) is not None
                and (sl.stop - sl.start) == D
            ):
                mu = torch.as_tensor(self.mu_y, device=rot6d.device, dtype=rot6d.dtype)[..., sl]
                std = torch.as_tensor(self.std_y, device=rot6d.device, dtype=rot6d.dtype)[..., sl].clamp(min=1e-6)
                while mu.dim() < rot6d.dim():
                    mu = mu.unsqueeze(0)
                    std = std.unsqueeze(0)
                rot6d = rot6d * std + mu
        except Exception:
            pass

        rot6d = torch.nan_to_num(rot6d, nan=0.0, posinf=1.0, neginf=-1.0)
        rot6d = reproject_rot6d(rot6d)
        rot6d = rot6d.view(*rot6d.shape[:-1], J, 6)
        return rot6d_to_matrix(rot6d)

    def _angvel_hz(self) -> float:
        hz = float(getattr(self, 'bone_hz', 0.0) or 0.0)
        if hz <= 0.0:
            dt = float(getattr(self, 'dt_bone', 0.0) or 0.0)
            if dt > 0.0:
                hz = 1.0 / max(dt, 1e-6)
        if hz <= 0.0:
            hz = float(getattr(self, 'fps', 60.0) or 60.0)
        return max(hz, 1e-6)

    def _angular_velocity_from_mats(self, R_seq: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if R_seq is None or R_seq.dim() < 5:
            return None
        T = R_seq.shape[-4]
        if T < 2:
            return None
        lead = R_seq.shape[:-4]
        B = int(_math.prod(lead)) if lead else 1
        J = R_seq.shape[-3]
        mats = R_seq.reshape(B, T, J, 3, 3)
        dR = torch.matmul(mats[:, 1:], mats[:, :-1].transpose(-1, -2))
        vec = _matrix_log_map(dR)
        hz = self._angvel_hz()
        omega = vec * hz
        return omega.reshape(*lead, T - 1, J, 3)

    def _angular_velocity_from_delta(self, delta: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if delta is None:
            return None
        rot_delta = self._maybe_get_rot6d(delta)
        if rot_delta is None:
            return None
        D = rot_delta.shape[-1]
        if D % 6 != 0:
            return None
        delta_proj = normalize_rot6d_delta(rot_delta, columns=self._rot6d_columns)
        dR = rot6d_to_matrix(delta_proj, columns=self._rot6d_columns)
        if dR.dim() < 5:
            return None
        lead = dR.shape[:-4]
        B = int(_math.prod(lead)) if lead else 1
        T = dR.shape[-4]
        J = dR.shape[-3]
        mats = dR.reshape(B, T, J, 3, 3)
        vec = _matrix_log_map(mats)
        hz = self._angvel_hz()
        omega = vec * hz
        return omega.reshape(*lead, T, J, 3)

    def _align_angvel_pair(self, pred: torch.Tensor, gt: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if pred.shape[:-3] != gt.shape[:-3]:
            return None
        Tp = pred.shape[-3]
        Tg = gt.shape[-3]
        if Tp == 0 or Tg == 0:
            return None
        if Tp == Tg:
            return pred, gt
        if Tp == Tg + 1:
            pred = pred[..., 1:, :, :]
        elif Tg == Tp + 1:
            gt = gt[..., 1:, :, :]
        else:
            L = min(Tp, Tg)
            if L <= 0:
                return None
            pred = pred[..., :L, :, :]
            gt = gt[..., :L, :, :]
        if pred.shape[-3] == 0:
            return None
        return pred, gt

    def _angular_direction_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> Optional[torch.Tensor]:
        eps = self.angvel_eps
        norm_p = pred.norm(dim=-1)
        norm_g = gt.norm(dim=-1)
        mask = (norm_p > eps) & (norm_g > eps)
        if not torch.any(mask):
            return None
        denom = (norm_p * norm_g).clamp_min(eps)
        cos = ((pred * gt).sum(dim=-1) / denom).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        ang = torch.acos(cos)
        return ang[mask].mean()

    def _broadcast_param_slice(self, arr, sl: Optional[slice], ref_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if arr is None or not isinstance(sl, slice):
            return None
        tensor = torch.as_tensor(arr, device=ref_tensor.device, dtype=ref_tensor.dtype)
        sliced = tensor[..., sl]
        if sliced.numel() == 0:
            return None
        width = sliced.numel()
        view_shape = [1] * (ref_tensor.dim() - 1) + [width]
        return sliced.reshape(*view_shape)

    def _denorm_yaw_slice(self, yaw_norm: torch.Tensor, sl: slice) -> Optional[torch.Tensor]:
        mu = self._broadcast_param_slice(getattr(self, 'mu_y', None), sl, yaw_norm)
        std = self._broadcast_param_slice(getattr(self, 'std_y', None), sl, yaw_norm)
        if mu is None or std is None:
            return None
        return yaw_norm * std + mu

    def _compute_cond_yaw_loss(self, pred_motion: torch.Tensor, batch: Optional[dict]) -> Optional[torch.Tensor]:
        if self.w_cond_yaw <= 0:
            return None
        if batch is None or not isinstance(batch, dict):
            return None
        if pred_motion.dim() < 3:
            return None
        yaw_sl = self.group_slices.get('RootYaw') or self.group_slices.get('Yaw')
        if not isinstance(yaw_sl, slice):
            return None
        cond_raw = batch.get('cond_tgt_raw')
        if cond_raw is None:
            cond_raw = batch.get('cond_in')
        if cond_raw is None:
            return None
        if not torch.is_tensor(cond_raw):
            cond_raw = torch.as_tensor(cond_raw)
        cond_raw = cond_raw.to(device=pred_motion.device, dtype=pred_motion.dtype)
        if cond_raw.dim() == 2:
            cond_raw = cond_raw.unsqueeze(0)
        if cond_raw.dim() != 3:
            return None
        yaw_norm = pred_motion[..., yaw_sl]
        if yaw_norm.numel() == 0:
            return None
        Bp = yaw_norm.shape[0]
        if cond_raw.shape[0] != Bp:
            if cond_raw.shape[0] == 1:
                cond_raw = cond_raw.expand(Bp, -1, -1)
            else:
                return None
        Tp = yaw_norm.shape[1]
        Tc = cond_raw.shape[1]
        L = min(Tp, Tc)
        if L <= 0:
            return None
        yaw_norm = yaw_norm[:, :L]
        cond_slice = cond_raw[:, :L]
        cond_dim = cond_slice.shape[-1]
        if cond_dim < 2:
            return None
        if cond_dim >= 3:
            dir_slice = cond_slice[..., cond_dim - 3:cond_dim - 1]
            speed_slice = cond_slice[..., -1]
        else:
            dir_slice = cond_slice[..., -2:]
            speed_slice = dir_slice.norm(dim=-1)
        dir_norm = dir_slice.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dir_unit = dir_slice / dir_norm
        yaw_cmd = torch.atan2(dir_unit[..., 1], dir_unit[..., 0])
        yaw_pred_raw = self._denorm_yaw_slice(yaw_norm, yaw_sl)
        if yaw_pred_raw is None:
            return None
        yaw_cmd = yaw_cmd.unsqueeze(-1)
        yaw_diff = torch.atan2(
            torch.sin(yaw_pred_raw - yaw_cmd),
            torch.cos(yaw_pred_raw - yaw_cmd),
        ).abs()
        yaw_diff = yaw_diff.mean(dim=-1)
        speed_abs = speed_slice.abs()
        min_speed = max(0.0, float(getattr(self, 'cond_yaw_min_speed', 0.0) or 0.0))
        if min_speed > 0.0:
            mask = speed_abs >= min_speed
        else:
            mask = None
        if mask is not None and mask.any():
            loss = yaw_diff[mask].mean()
        else:
            loss = yaw_diff.mean()
        return loss

    def _prepare_angvel_payload(
        self,
        pred_motion: torch.Tensor,
        gt_motion: torch.Tensor,
        delta_motion: Optional[torch.Tensor],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        omega_gt = self._angular_velocity_from_mats(self._rot6d_matrices(gt_motion))
        if omega_gt is None:
            return None
        omega_pred = self._angular_velocity_from_delta(delta_motion)
        if omega_pred is None:
            omega_pred = self._angular_velocity_from_mats(self._rot6d_matrices(pred_motion))
        if omega_pred is None:
            return None
        aligned = self._align_angvel_pair(omega_pred, omega_gt)
        if aligned is None:
            return None
        return aligned

    def compute_rot6d_delta_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        Z = lambda v: pred.new_tensor(float(v))
        rot_delta = self._maybe_get_rot6d(pred)
        if rot_delta is None:
            return Z(0.0)
        D = rot_delta.shape[-1]
        if D % 6 != 0:
            return Z(0.0)
        J = D // 6

        delta_proj = normalize_rot6d_delta(rot_delta, columns=("X", "Z"))
        dRp = rot6d_to_matrix(delta_proj, columns=("X", "Z"))
        if dRp.dim() < 5:
            return Z(0.0)

        Rg = self._rot6d_matrices(gt)
        if Rg is None or Rg.dim() < 5:
            return Z(0.0)

        T_pred = dRp.shape[-4]
        T_gt = Rg.shape[-4]
        if T_gt < 2 or T_pred < 1:
            return Z(0.0)

        lead_pred = dRp.shape[:-4]
        lead_gt = Rg.shape[:-4]
        Bp = int(_math.prod(lead_pred)) if lead_pred else 1
        Bg = int(_math.prod(lead_gt)) if lead_gt else 1
        dRp = dRp.reshape(Bp, T_pred, J, 3, 3)
        Rg = Rg.reshape(Bg, T_gt, J, 3, 3)

        dRg = torch.matmul(Rg[:, 1:], Rg[:, :-1].transpose(-1, -2))

        if dRp.shape[1] == dRg.shape[1] + 1:
            dRp = dRp[:, 1:]
        elif dRp.shape[1] != dRg.shape[1]:
            L = min(dRp.shape[1], dRg.shape[1])
            dRp = dRp[:, :L]
            dRg = dRg[:, :L]

        if dRp.shape[1] == 0:
            return Z(0.0)

        theta = geodesic_R(dRp, dRg)
        theta = torch.nan_to_num(theta, nan=0.0, posinf=_math.pi, neginf=0.0)
        return theta.mean()

    def compute_rot6d_delta_root_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        Z = lambda v: pred.new_tensor(float(v))
        Rp = self._rot6d_matrices(pred)
        Rg = self._rot6d_matrices(gt)
        if Rp is None or Rg is None:
            return Z(0.0)
        if Rp.dim() < 4:
            return Z(0.0)
        root_idx = int(getattr(self, 'root_idx', 0))
        Bp = Rp.shape[:-3]
        Bg = Rg.shape[:-3]
        T = Rp.shape[-3]
        J = Rp.shape[-2]
        Rp = Rp.reshape(-1, T, J, 3, 3)
        Rg = Rg.reshape(-1, T, J, 3, 3)
        Rp = _root_relative_matrices(Rp, root_idx)
        Rg = _root_relative_matrices(Rg, root_idx)
        dRp = torch.matmul(Rp[:, 1:], Rp[:, :-1].transpose(-1, -2))
        dRg = torch.matmul(Rg[:, 1:], Rg[:, :-1].transpose(-1, -2))
        theta = geodesic_R(dRp, dRg)
        theta = torch.nan_to_num(theta, nan=0.0, posinf=_math.pi, neginf=0.0)
        return theta.mean()

    def compute_rot6d_log_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        Z = lambda v: pred.new_tensor(float(v))
        Rp = self._rot6d_matrices(pred)
        Rg = self._rot6d_matrices(gt)
        if Rp is None or Rg is None:
            return Z(0.0)
        if Rp.dim() < 4:
            return Z(0.0)
        T = Rp.shape[-3]
        J = Rp.shape[-2]
        Rp = Rp.reshape(-1, T, J, 3, 3)
        Rg = Rg.reshape(-1, T, J, 3, 3)
        dRp = torch.matmul(Rp[:, 1:], Rp[:, :-1].transpose(-1, -2))
        dRg = torch.matmul(Rg[:, 1:], Rg[:, :-1].transpose(-1, -2))
        log_p = _matrix_log_map(dRp)
        log_g = _matrix_log_map(dRg)
        return torch.nn.functional.smooth_l1_loss(log_p, log_g)


    def forward(self, pred_motion, gt_motion, attn_weights=None, batch=None):
        # 统一拿出模型输出（可能是 dict 或 tensor）
        delta_fallback = False
        if isinstance(pred_motion, dict):
            delta_fallback = bool(pred_motion.get('_delta_fallback', False))
        pm = pred_motion.get('out') if isinstance(pred_motion, dict) else pred_motion
        gm = gt_motion
        delta_pm = pred_motion.get('delta') if isinstance(pred_motion, dict) else None

        # _forward_base_inner 已包含核心动作损失与统计
        base_out = self._forward_base_inner(pm, gt_motion, attn_weights=attn_weights)  # type: ignore
        if isinstance(base_out, tuple):
            loss, stats = base_out
        else:
            loss, stats = base_out, {}

        if isinstance(stats, dict):
            stats = dict(stats)
        else:
            stats = {}

        if self.w_rot_delta > 0 and delta_pm is not None:
            l_delta = self.compute_rot6d_delta_loss(delta_pm, gt_motion)
            loss = loss + self.w_rot_delta * l_delta
            stats['rot_delta'] = float(l_delta.detach().cpu())
        else:
            stats.setdefault('rot_delta', 0.0)

        if self.w_rot_delta_root > 0 and delta_pm is not None:
            l_delta_root = self.compute_rot6d_delta_root_loss(delta_pm, gt_motion)
            loss = loss + self.w_rot_delta_root * l_delta_root
            stats['rot_delta_root'] = float(l_delta_root.detach().cpu())
        else:
            stats.setdefault('rot_delta_root', 0.0)

        if self.w_rot_log > 0 and delta_pm is not None and not delta_fallback:
            l_rot_log = self.compute_rot6d_log_loss(delta_pm, gt_motion)
            loss = loss + self.w_rot_log * l_rot_log
            stats['rot_log'] = float(l_rot_log.detach().cpu())
        else:
            stats.setdefault('rot_log', 0.0)

        if self.w_rot_ortho > 0 and not delta_fallback:
            target_for_ortho = delta_pm if delta_pm is not None else pm
            l_ortho = self.compute_rot6d_ortho_loss(target_for_ortho)
            loss = loss + self.w_rot_ortho * l_ortho
            stats['rot_ortho'] = float((self.w_rot_ortho * l_ortho).detach().cpu())
            stats['rot_ortho_raw'] = float(l_ortho.detach().cpu())
        else:
            stats.setdefault('rot_ortho', 0.0)
            stats.setdefault('rot_ortho_raw', 0.0)

        if delta_fallback and self.w_rot_ortho > 0 and delta_pm is not None:
            # 即使跳过 rot_ortho，也在 stats 中记录原生值方便诊断
            try:
                l_ortho = self.compute_rot6d_ortho_loss(delta_pm)
                stats['rot_ortho_fallback'] = float(l_ortho.detach().cpu())
            except Exception:
                stats['rot_ortho_fallback'] = float('nan')


        cond_yaw_loss = self._compute_cond_yaw_loss(pm, batch)
        if cond_yaw_loss is not None:
            loss = loss + self.w_cond_yaw * cond_yaw_loss
            stats['cond_yaw'] = float(cond_yaw_loss.detach().cpu())
        else:
            stats.setdefault('cond_yaw', 0.0)

        Rp_world = Rg_world = None
        Rp_root = Rg_root = None
        if self.w_fk_pos > 0.0 or self.w_rot_local > 0.0:
            Rp_world = self._rot6d_matrices(pm)
            Rg_world = self._rot6d_matrices(gm)
            if Rp_world is not None and Rg_world is not None:
                Rp_root = self._root_relative(Rp_world)
                Rg_root = self._root_relative(Rg_world)

        if self.w_fk_pos > 0.0 and self.has_fk:
            if Rp_root is not None and Rg_root is not None:
                pos_pred = self._fk_positions(Rp_root)
                pos_gt = self._fk_positions(Rg_root)
                if pos_pred is not None and pos_gt is not None:
                    fk_res = F.smooth_l1_loss(pos_pred, pos_gt, reduction='none').mean(dim=-1)
                    weights = self._joint_weight_vector(pos_pred.device, pos_pred.dtype, pos_pred.shape[-2])
                    w = weights.view(1, 1, -1)
                    fk_loss = (fk_res * w).mean()
                    loss = loss + self.w_fk_pos * fk_loss
                    stats['fk_pos'] = float(fk_loss.detach().cpu())
        else:
            stats.setdefault('fk_pos', 0.0)

        if self.w_rot_local > 0.0:
            if Rp_root is not None and Rg_root is not None:
                Rp_local = self._parent_relative_matrices(Rp_root)
                Rg_local = self._parent_relative_matrices(Rg_root)
                geo_local = geodesic_R(Rp_local, Rg_local)
                weights = self._joint_weight_vector(Rp_local.device, Rp_local.dtype, Rp_local.shape[-3])
                w = weights.view(1, 1, -1)
                local_loss = (geo_local * w).mean()
                loss = loss + self.w_rot_local * local_loss
                stats['rot_local_deg'] = float((local_loss * (180.0 / math.pi)).detach().cpu())
        else:
            stats.setdefault('rot_local_deg', 0.0)

        return loss, stats

class DataNormalizer:
    """封装数据规格与(反)归一化逻辑。"""
    def __init__(self, *,
                 mu_x=None, std_x=None, mu_y=None, std_y=None,
                 y_to_x_map=None,
                 yaw_x_slice=None, yaw_y_slice=None,
                 rootvel_x_slice=None, rootvel_y_slice=None,
                 angvel_x_slice=None, angvel_y_slice=None,
                 tanh_scales_rootvel=None, tanh_scales_angvel=None,
                 traj_dir_slice=None,
                 angvel_mode=None, angvel_mu=None, angvel_std=None):
        import numpy as np
        self.mu_x = None if mu_x is None else np.asarray(mu_x, dtype=np.float32)
        self.std_x = None if std_x is None else np.asarray(std_x, dtype=np.float32)
        self.mu_y = None if mu_y is None else np.asarray(mu_y, dtype=np.float32)
        self.std_y = None if std_y is None else np.asarray(std_y, dtype=np.float32)
        self.y_to_x_map = y_to_x_map or []
        self.yaw_x_slice      = parse_layout_entry(yaw_x_slice,      'RootYaw')
        self.yaw_y_slice      = parse_layout_entry(yaw_y_slice,      'RootYaw')
        self.rootvel_x_slice  = parse_layout_entry(rootvel_x_slice,  'RootVelocity')
        self.rootvel_y_slice  = parse_layout_entry(rootvel_y_slice,  'RootVelocity')
        self.angvel_x_slice   = parse_layout_entry(angvel_x_slice,   'BoneAngularVelocities')
        self.angvel_y_slice   = parse_layout_entry(angvel_y_slice,   'BoneAngularVelocities')
        self.traj_dir_slice   = parse_layout_entry(traj_dir_slice,   'TrajectoryDir')

        self.tanh_scales_rootvel = None if tanh_scales_rootvel is None else np.asarray(tanh_scales_rootvel, dtype=np.float32)
        self.tanh_scales_angvel  = None if tanh_scales_angvel  is None else np.asarray(tanh_scales_angvel,  dtype=np.float32)
        self.angvel_mode = (angvel_mode or '').strip().lower() if isinstance(angvel_mode, str) else None
        self.angvel_mu = None if angvel_mu is None else np.asarray(angvel_mu, dtype=np.float32)
        self.angvel_std = None if angvel_std is None else np.asarray(angvel_std, dtype=np.float32)
        if self.angvel_std is not None:
            self.angvel_std = np.clip(self.angvel_std, 1e-6, None)

        # torch tensor cache: key = (name, device, dtype)
        self._tensor_cache: dict[tuple[str, str, str], "torch.Tensor"] = {}

        def _slice_width(sl):
            return int(sl.stop - sl.start) if isinstance(sl, slice) else 0

        root_widths = (_slice_width(self.rootvel_x_slice), _slice_width(self.rootvel_y_slice))
        max_root_width = max(root_widths)
        if max_root_width:
            if self.tanh_scales_rootvel is None:
                raise ValueError("DataNormalizer requires tanh_scales_rootvel when RootVelocity is present.")
            if self.tanh_scales_rootvel.size != max_root_width:
                raise ValueError(f"tanh_scales_rootvel length {self.tanh_scales_rootvel.size} "
                                 f"!= RootVelocity width {max_root_width}.")

        ang_widths = (_slice_width(self.angvel_x_slice), _slice_width(self.angvel_y_slice))
        max_ang_width = max(ang_widths)
        if max_ang_width:
            if self.angvel_mode == 'standardize':
                if self.angvel_mu is None or self.angvel_std is None:
                    raise ValueError("angvel_mode=standardize requires MuAngVel and StdAngVel.")
                if self.angvel_mu.size != max_ang_width or self.angvel_std.size != max_ang_width:
                    raise ValueError(f"MuAngVel/StdAngVel length mismatch ({self.angvel_mu.size}/{self.angvel_std.size}) "
                                     f"vs BoneAngularVelocities width {max_ang_width}.")
            else:
                self.angvel_mode = 'tanh'
                if self.tanh_scales_angvel is None:
                    raise ValueError("tanh_scales_angvel missing for BoneAngularVelocities slice.")
                if self.tanh_scales_angvel.size != max_ang_width:
                    raise ValueError(f"tanh_scales_angvel length {self.tanh_scales_angvel.size} "
                                     f"!= BoneAngularVelocities width {max_ang_width}.")
        else:
            if self.angvel_mode == 'standardize' and (self.angvel_mu is None or self.angvel_std is None):
                raise ValueError("angvel_mode=standardize declared but BoneAngularVelocities slice is absent.")
            if self.angvel_mode not in (None, 'standardize'):
                self.angvel_mode = 'tanh'
        if self.angvel_mode is None:
            self.angvel_mode = 'tanh'

    def _match_tensor(self, key: str, arr, ref_tensor):
        import torch
        if arr is None:
            return None
        cache_key = (key, str(ref_tensor.device), str(ref_tensor.dtype))
        tensor = self._tensor_cache.get(cache_key)
        if tensor is None:
            tensor = torch.as_tensor(arr, device=ref_tensor.device, dtype=ref_tensor.dtype)
            self._tensor_cache[cache_key] = tensor
        while tensor.dim() < ref_tensor.dim():
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _atanh_safe_t(x, torch):
        x = torch.clamp(x, -0.999999, 0.999999)
        return torch.atanh(x) if hasattr(torch, "atanh") else 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def norm(self, x_raw_t):
        import torch
        x_raw = x_raw_t

        x_proc = x_raw

        if isinstance(self.rootvel_x_slice, slice):
            if self.tanh_scales_rootvel is None:
                raise RuntimeError("tanh_scales_rootvel missing; cannot normalize RootVelocity.")
            x_proc = x_proc.clone()
            sc = self._match_tensor('tanh_rootvel', self.tanh_scales_rootvel, x_raw).clamp_min(1e-6)
            x_proc[..., self.rootvel_x_slice] = torch.tanh(
                x_raw[..., self.rootvel_x_slice] / sc
            )

        if isinstance(self.angvel_x_slice, slice):
            x_proc = x_proc.clone()
            width = self.angvel_x_slice.stop - self.angvel_x_slice.start
            if (
                self.angvel_mode == 'standardize'
                and self.angvel_mu is not None
                and self.angvel_std is not None
                and width == self.angvel_mu.size
            ):
                mu = self._match_tensor('angvel_mu_x', self.angvel_mu, x_raw)
                std = self._match_tensor('angvel_std_x', self.angvel_std, x_raw).clamp_min(1e-6)
                x_proc[..., self.angvel_x_slice] = (
                    x_raw[..., self.angvel_x_slice] - mu
                ) / std
            elif self.angvel_mode == 'tanh':
                if self.tanh_scales_angvel is None:
                    raise RuntimeError("tanh_scales_angvel missing; cannot normalize BoneAngularVelocities.")
                sc = self._match_tensor('tanh_angvel', self.tanh_scales_angvel, x_raw).clamp_min(1e-6)
                x_proc[..., self.angvel_x_slice] = torch.tanh(
                    x_raw[..., self.angvel_x_slice] / sc
                )
            else:
                raise RuntimeError(f"Unsupported angvel_mode during norm(): {self.angvel_mode}")

        if isinstance(self.yaw_x_slice, slice):
            x_proc = x_proc.clone()
            x_proc[..., self.yaw_x_slice] = x_raw[..., self.yaw_x_slice].abs()

        if (self.mu_x is not None) and (self.std_x is not None):
            mu = self._match_tensor('mu_x', self.mu_x, x_raw)
            sd = self._match_tensor('std_x', self.std_x, x_raw).clamp_min(1e-3)
            z = (x_proc - mu) / sd
        else:
            z = x_proc
        return z

    def denorm_x(self, xz_t, prev_raw=None):
        """把 X 的 Z 域还原为 RAW：做 μ/σ 逆 + 分组逆变换；Yaw 因 abs 丢符号，若给了 prev_raw 就延用上一帧符号。"""
        import torch

        xz = xz_t

        if (self.mu_x is not None) and (self.std_x is not None):
            mu = self._match_tensor('mu_x', self.mu_x, xz)
            sd = self._match_tensor('std_x', self.std_x, xz).clamp_min(1e-6)
            x_pre = xz * sd + mu
        else:
            x_pre = xz.clone()

        x_raw = x_pre

        # 分组逆变换（与 norm 对称）
        if isinstance(self.rootvel_x_slice, slice):
            if self.tanh_scales_rootvel is None:
                raise RuntimeError("tanh_scales_rootvel missing; cannot denormalize RootVelocity.")
            x_raw = x_raw.clone()
            sc = self._match_tensor('tanh_rootvel', self.tanh_scales_rootvel, xz)
            x_raw[..., self.rootvel_x_slice] = self._atanh_safe_t(
                x_pre[..., self.rootvel_x_slice], torch
            ) * sc

        if isinstance(self.angvel_x_slice, slice):
            x_raw = x_raw.clone()
            width = self.angvel_x_slice.stop - self.angvel_x_slice.start
            if (
                self.angvel_mode == 'standardize'
                and self.angvel_mu is not None
                and self.angvel_std is not None
                and width == self.angvel_mu.size
            ):
                mu = self._match_tensor('angvel_mu_x', self.angvel_mu, xz)
                std = self._match_tensor('angvel_std_x', self.angvel_std, xz).clamp_min(1e-6)
                x_raw[..., self.angvel_x_slice] = x_pre[..., self.angvel_x_slice] * std + mu
            elif self.angvel_mode == 'tanh':
                if self.tanh_scales_angvel is None:
                    raise RuntimeError("tanh_scales_angvel missing; cannot denormalize BoneAngularVelocities.")
                sc = self._match_tensor('tanh_angvel', self.tanh_scales_angvel, xz)
                x_raw[..., self.angvel_x_slice] = self._atanh_safe_t(
                    x_pre[..., self.angvel_x_slice], torch
                ) * sc
            else:
                raise RuntimeError(f"Unsupported angvel_mode during denorm_x(): {self.angvel_mode}")

        # yaw: 使用上一帧 RAW 的符号（若可用）
        if isinstance(self.yaw_x_slice, slice) and (prev_raw is not None):
            s = self.yaw_x_slice
            prev = torch.as_tensor(prev_raw, device=xz.device, dtype=xz.dtype)
            sign = torch.sign(prev[..., s]).clamp(min=-1.0, max=1.0)
            x_raw = x_raw.clone()
            x_raw[..., s] = torch.abs(x_raw[..., s]) * sign

        return x_raw



    def denorm(self, y_t):
        import torch, math
        y_pre = y_t
        if self.std_y is not None and self.mu_y is not None:
            std = self._match_tensor('std_y', self.std_y, y_t).clamp_min(1e-6)
            mu  = self._match_tensor('mu_y', self.mu_y, y_t)
            y_pre = y_pre * std + mu
        else:
            y_pre = y_t.clone()

        y = y_pre.clone()

        if isinstance(self.yaw_y_slice, slice):
            yaw_clamped = torch.clamp(y_pre[..., self.yaw_y_slice], -1.0, 1.0)
            y[..., self.yaw_y_slice] = yaw_clamped * math.pi

        if isinstance(self.rootvel_y_slice, slice):
            if self.tanh_scales_rootvel is None:
                raise RuntimeError("tanh_scales_rootvel missing; cannot denormalize RootVelocity.")
            sc = self._match_tensor('tanh_rootvel', self.tanh_scales_rootvel, y_pre)
            y[..., self.rootvel_y_slice] = self._atanh_safe_t(
                y_pre[..., self.rootvel_y_slice], torch
            ) * sc

        if isinstance(self.angvel_y_slice, slice):
            width = self.angvel_y_slice.stop - self.angvel_y_slice.start
            if (
                self.angvel_mode == 'standardize'
                and self.angvel_mu is not None
                and self.angvel_std is not None
                and width == self.angvel_mu.size
            ):
                mu = self._match_tensor('angvel_mu_y', self.angvel_mu, y_pre)
                std = self._match_tensor('angvel_std_y', self.angvel_std, y_pre).clamp_min(1e-6)
                y[..., self.angvel_y_slice] = y_pre[..., self.angvel_y_slice] * std + mu
            elif self.angvel_mode == 'tanh':
                if self.tanh_scales_angvel is None:
                    raise RuntimeError("tanh_scales_angvel missing; cannot denormalize BoneAngularVelocities.")
                sc = self._match_tensor('tanh_angvel', self.tanh_scales_angvel, y_pre)
                y[..., self.angvel_y_slice] = self._atanh_safe_t(
                    y_pre[..., self.angvel_y_slice], torch
                ) * sc
            else:
                raise RuntimeError(f"Unsupported angvel_mode during denorm(): {self.angvel_mode}")
        try:
            sl = self.traj_dir_slice
            if isinstance(sl, slice):
                a, b = sl.start, sl.stop - sl.start
                dim = 3 if b % 3 == 0 else 2 if b % 2 == 0 else 0
                if dim > 0:
                    blk = y[..., a:a+b].view(*y.shape[:-1], b // dim, dim)
                    y[..., a:a+b] = torch.nn.functional.normalize(blk, dim=-1).reshape_as(y[..., a:a+b])
        except Exception as _err:
            print(f"[Norm-ERR] trajectory direction normalization failed: {_err}")
            pass
        return y

    def norm_y(self, y_raw_t):
        import torch, math
        y_raw = y_raw_t

        y_pre = y_raw

        if isinstance(self.yaw_y_slice, slice):
            y_pre = y_pre.clone()
            y_pre[..., self.yaw_y_slice] = torch.clamp(
                y_raw[..., self.yaw_y_slice] / math.pi,
                -0.999999,
                0.999999,
            )

        if isinstance(self.rootvel_y_slice, slice):
            if self.tanh_scales_rootvel is None:
                raise RuntimeError("tanh_scales_rootvel missing; cannot normalize RootVelocity.")
            y_pre = y_pre.clone()
            sc = self._match_tensor('tanh_rootvel', self.tanh_scales_rootvel, y_raw).clamp_min(1e-6)
            y_pre[..., self.rootvel_y_slice] = torch.tanh(
                y_raw[..., self.rootvel_y_slice] / sc
            )

        if isinstance(self.angvel_y_slice, slice):
            width = self.angvel_y_slice.stop - self.angvel_y_slice.start
            y_pre = y_pre.clone()
            if (
                self.angvel_mode == 'standardize'
                and self.angvel_mu is not None
                and self.angvel_std is not None
                and width == self.angvel_mu.size
            ):
                mu = self._match_tensor('angvel_mu_y', self.angvel_mu, y_raw)
                std = self._match_tensor('angvel_std_y', self.angvel_std, y_raw).clamp_min(1e-6)
                y_pre[..., self.angvel_y_slice] = (
                    y_raw[..., self.angvel_y_slice] - mu
                ) / std
            elif self.angvel_mode == 'tanh':
                if self.tanh_scales_angvel is None:
                    raise RuntimeError("tanh_scales_angvel missing; cannot normalize BoneAngularVelocities.")
                sc = self._match_tensor('tanh_angvel', self.tanh_scales_angvel, y_raw).clamp_min(1e-6)
                y_pre[..., self.angvel_y_slice] = torch.tanh(
                    y_raw[..., self.angvel_y_slice] / sc
                )
            else:
                raise RuntimeError(f"Unsupported angvel_mode during norm_y(): {self.angvel_mode}")

        try:
            if isinstance(self.traj_dir_slice, slice):
                a = self.traj_dir_slice.start
                b = self.traj_dir_slice.stop - self.traj_dir_slice.start
                dim = 3 if b % 3 == 0 else 2 if b % 2 == 0 else 0
                if dim > 0:
                    blk = y_raw[..., a:a + b].view(*y_raw.shape[:-1], b // dim, dim)
                    blk = torch.nn.functional.normalize(blk, dim=-1, eps=1e-9)
                    y_pre = y_pre.clone()
                    y_pre[..., a:a + b] = blk.reshape_as(y_raw[..., a:a + b])
        except Exception as _err:
            print(f"[Norm-ERR] trajectory direction norm_y failed: {_err}")

        if self.mu_y is not None and self.std_y is not None:
            mu = self._match_tensor('mu_y', self.mu_y, y_raw)
            std = self._match_tensor('std_y', self.std_y, y_raw).clamp_min(1e-6)
            return (y_pre - mu) / std
        return y_pre

    @classmethod
    def from_bundle(cls, bundle: dict):
        get = bundle.get
        def key2slice(d, key):
            if d is None: return None
            meta = d.get(key) if isinstance(d, dict) else None
            if meta is None: return None
            if isinstance(meta, dict):
                st, sz = int(meta.get('start', 0)), int(meta.get('size', 0))
                return slice(st, st+sz) if sz > 0 else None
            if isinstance(meta, (list, tuple)) and len(meta) >= 2:
                st, sz = int(meta[0]), int(meta[1])
                return slice(st, st+sz)
            return None
        s_layout = get('state_layout') or {}
        o_layout = get('output_layout') or {}
        return cls(
            mu_x = get('MuX'), std_x = get('StdX'),
            mu_y = get('MuY'), std_y = get('StdY'),
            y_to_x_map = get('y_to_x_map', []),
            yaw_x_slice     = key2slice(s_layout, 'RootYaw') or key2slice(s_layout, 'Yaw'),
            yaw_y_slice     = key2slice(o_layout, 'RootYaw') or key2slice(o_layout, 'Yaw'),
            rootvel_x_slice = key2slice(s_layout, 'RootVelocity'),
            rootvel_y_slice = key2slice(o_layout, 'RootVelocity'),
            angvel_x_slice  = key2slice(s_layout, 'BoneAngularVelocities'),
            angvel_y_slice  = key2slice(o_layout, 'BoneAngularVelocities'),
            tanh_scales_rootvel = get('tanh_scales_rootvel'),
            tanh_scales_angvel  = get('tanh_scales_angvel'),
            traj_dir_slice = key2slice(o_layout, 'TrajectoryDir'),
        )


def _parse_stage_schedule(spec: Optional[Any]):
    """Parse stage schedule definitions from CLI strings or structured JSON."""

    def _coerce_value(key: str, val: Any) -> Any:
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, bool) or val is None:
            return val
        if isinstance(val, str):
            txt = val.strip()
            if not txt:
                return txt
            lowered = txt.lower()
            if lowered in ('true', 'false'):
                return lowered == 'true'
            if lowered == 'none':
                return None
            try:
                if key.endswith(('steps', 'horizon', 'epoch', 'epochs')):
                    return int(float(txt))
                return float(txt)
            except ValueError:
                try:
                    return int(txt)
                except ValueError:
                    return txt
        return val

    def _append_stage(stages: list, start: int, end: int, params: Dict[str, Any], label: Optional[str] = None):
        if start is None or end is None:
            return
        stage = {'start': int(start), 'end': int(end), 'params': dict(params)}
        if label:
            stage['label'] = str(label)
        stages.append(stage)

    def _parse_string(spec_str: str):
        out = []
        for entry in spec_str.split(';'):
            chunk = entry.strip()
            if not chunk or ':' not in chunk:
                continue
            range_part, params_part = chunk.split(':', 1)
            label = None
            if '@' in range_part:
                range_part, label = [seg.strip() for seg in range_part.split('@', 1)]
            range_part = range_part.strip()
            if '-' in range_part:
                start_s, end_s = range_part.split('-', 1)
                start = int(start_s.strip())
                end = int(end_s.strip())
            else:
                start = end = int(range_part.strip())
            params = {}
            for token in params_part.split(','):
                token = token.strip()
                if not token or '=' not in token:
                    continue
                key, val = token.split('=', 1)
                key = key.strip()
                val = val.strip()
                params[key] = _coerce_value(key, val)
            _append_stage(out, start, end, params, label)
        return out

    def _normalize_range(entry: Mapping[str, Any]):
        start = entry.get('start')
        end = entry.get('end')
        if start is None and end is None:
            rng = entry.get('range') or entry.get('epochs')
            if isinstance(rng, str):
                part = rng.strip()
                if '-' in part:
                    s, e = part.split('-', 1)
                    return int(s.strip()), int(e.strip())
                return int(part), int(part)
            if isinstance(rng, Sequence) and rng:
                if len(rng) == 1:
                    val = int(rng[0])
                    return val, val
                return int(rng[0]), int(rng[-1])
        if start is None and end is not None:
            start = end
        if end is None and start is not None:
            end = start
        if start is None:
            return None, None
        return int(start), int(end)

    def _merge_params(entry: Mapping[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if not isinstance(entry, Mapping):
            return params
        base = entry.get('params') if isinstance(entry.get('params'), Mapping) else {}
        for key, val in base.items():
            params[key] = val

        def _ingest(source: Optional[Mapping[str, Any]], prefix: Optional[str] = None):
            if not isinstance(source, Mapping):
                return
            for k, v in source.items():
                name = f"{prefix}.{k}" if prefix else k
                params[name] = v

        _ingest(entry.get('trainer'))
        _ingest(entry.get('loss'), prefix='loss')
        tf_cfg = entry.get('tf')
        if isinstance(tf_cfg, Mapping):
            if 'max' in tf_cfg:
                params['tf_max'] = tf_cfg['max']
            if 'min' in tf_cfg:
                params['tf_min'] = tf_cfg['min']

        reserved = {'start', 'end', 'range', 'epochs', 'params', 'trainer', 'loss', 'tf', 'label', 'name', 'updates'}
        for key, val in entry.items():
            if key in reserved:
                continue
            params[key] = val

        updates = entry.get('updates')
        if isinstance(updates, Sequence) and not isinstance(updates, (str, bytes)):
            for item in updates:
                if isinstance(item, Mapping):
                    target = item.get('key') or item.get('name') or item.get('param')
                    value = item.get('value')
                    if target:
                        params[target] = value
        # coerce
        return {k: _coerce_value(k, v) for k, v in params.items()}

    if not spec:
        return []
    if isinstance(spec, str):
        return _parse_string(spec)
    stages: list = []
    entries: Sequence[Any]
    if isinstance(spec, Mapping):
        entries = [spec]
    elif isinstance(spec, Sequence):
        entries = list(spec)
    else:
        return []
    for entry in entries:
        if isinstance(entry, str):
            stages.extend(_parse_string(entry))
            continue
        if not isinstance(entry, Mapping):
            continue
        start, end = _normalize_range(entry)
        label = entry.get('label') or entry.get('name')
        params = _merge_params(entry)
        _append_stage(stages, start, end, params, label)
    return stages


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

    def _rollout_sequence(self, state_seq, cond_seq=None, cond_raw_seq=None, contacts_seq=None, angvel_seq=None, pose_hist_seq=None, *, gt_seq=None, mode='mixed', tf_ratio=1.0):
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

        for t in range(T):
            self._diag_roll_step = int(t)
            cond_t = cond_seq[:, t] if has_time_dim['cond'] else cond_seq
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

            with self._amp_context(amp_enabled):
                ret = self.model(
                    motion,
                    cond_t,
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
                    next_raw = self._apply_free_carry(motion_raw_local, y_raw, cond_next_raw=cond_raw_t)
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
                    free_raw = self._apply_free_carry(motion_raw_local, y_raw, cond_next_raw=cond_raw_t).detach()
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
        mask = (torch.rand(B, T, J, 1, device=device) < float(noise_prob))
        if not mask.any():
            return state_seq
        rotJ = rot_chunk.view(B, T, J, 6)
        R = rot6d_to_matrix(rotJ)
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
                             train_mode: bool = False, return_preds: bool = False):
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
        )
        preds_free = self._ensure_rot6d_delta(preds_free)
        with self._amp_context(self.use_amp):
            out = self.loss_fn(preds_free, gt_sub, attn_weights=attn_free, batch=batch)
        if isinstance(out, tuple):
            free_loss, stats = out
        else:
            free_loss, stats = out, {}
        latent_payload = self._latent_consistency_penalty(
            preds_free,
            contacts_sub,
            angvel_sub,
            pose_hist_sub,
            step_index=-1,
            tag='freerun/',
        )
        if latent_payload is not None:
            lat_loss, lat_stats = latent_payload
            free_loss = free_loss + lat_loss
            if isinstance(stats, dict):
                stats.update(lat_stats)
            else:
                stats = dict(lat_stats)
        if return_preds:
            return free_loss, stats or {}, preds_free, gt_sub
        return free_loss, stats or {}, None, None

    def _latent_encoder_features(
        self,
        contacts_seq,
        angvel_seq,
        pose_hist_seq,
        *,
        step_index: int = -1,
    ):
        import torch

        def _pick_feature(tensor):
            if tensor is None or not torch.is_tensor(tensor):
                return None
            if tensor.dim() == 3 and tensor.size(1) > 0:
                idx = step_index
                if idx < 0:
                    idx = tensor.size(1) + idx
                idx = max(0, min(tensor.size(1) - 1, idx))
                return tensor[:, idx]
            if tensor.dim() == 2:
                return tensor
            return None

        feats = []
        for candidate in (_pick_feature(contacts_seq), _pick_feature(angvel_seq), _pick_feature(pose_hist_seq)):
            if candidate is not None:
                feats.append(candidate)
        if not feats:
            return None
        encoder_dim = int(getattr(self.model, 'encoder_input_dim', 0) or 0)
        cat = torch.cat(feats, dim=-1)
        if encoder_dim <= 0 or cat.size(-1) != encoder_dim:
            if not self._latent_encoder_warned:
                print(
                    "[LatentCons][warn] encoder input dim mismatch或未启用，"
                    f"expected={encoder_dim} got={cat.size(-1)}；跳过 latent consistency。"
                )
                self._latent_encoder_warned = True
            return None
        return cat

    def _latent_consistency_penalty(
        self,
        preds_dict,
        contacts_seq,
        angvel_seq,
        pose_hist_seq,
        *,
        step_index: int = -1,
        tag: str = '',
    ):
        import torch
        weight = float(getattr(self, 'w_latent_consistency', 0.0) or 0.0)
        if weight <= 0.0:
            return None
        model = getattr(self, 'model', None)
        encoder = getattr(model, 'frozen_encoder', None) if model is not None else None
        period_head = getattr(model, 'frozen_period_head', None) if model is not None else None
        if encoder is None or period_head is None:
            return None
        if not isinstance(preds_dict, dict):
            return None
        hidden_seq = preds_dict.get('hidden_seq')
        if hidden_seq is None:
            return None
        if hidden_seq.dim() == 2:
            hidden_seq = hidden_seq.unsqueeze(1)
        if hidden_seq.size(1) == 0:
            return None
        idx = step_index
        if idx < 0:
            idx = hidden_seq.size(1) + idx
        idx = max(0, min(hidden_seq.size(1) - 1, idx))
        hidden_final = hidden_seq[:, idx]
        bridge = getattr(model, 'latent_bridge', None)
        if bridge is not None:
            hidden_final = bridge(hidden_final)
        head_fc = getattr(period_head, 'fc', None)
        expected_dim = getattr(head_fc, 'in_features', hidden_final.size(-1))
        if hidden_final.size(-1) != expected_dim:
            if not self._latent_consistency_dim_warned:
                print(
                    "[LatentCons][warn] 模型 hidden_dim 映射后仍与 period head 输入不一致；"
                    "跳过 latent consistency (hidden_dim=%d, period_in=%d)."
                    % (hidden_final.size(-1), int(expected_dim))
                )
                self._latent_consistency_dim_warned = True
            return None
        encoder_input = self._latent_encoder_features(
            contacts_seq,
            angvel_seq,
            pose_hist_seq,
            step_index=idx,
        )
        if encoder_input is None:
            return None
        with torch.no_grad():
            gt_hidden = encoder(encoder_input, return_summary=False)
            if isinstance(gt_hidden, tuple):
                gt_hidden = gt_hidden[-1]
            if gt_hidden.dim() == 3:
                gt_hidden = gt_hidden[:, -1]
            soft_gt = torch.tanh(period_head(gt_hidden)).detach()
        soft_pred = torch.tanh(period_head(hidden_final))
        loss_raw = F.l1_loss(soft_pred, soft_gt)
        loss = weight * loss_raw
        prefix = f"{tag}latent_consistency" if tag else "latent_consistency"
        stats = {
            f"{prefix}/raw": float(loss_raw.detach().cpu()),
            f"{prefix}/weight": float(weight),
        }
        return loss, stats

    def _short_freerun_loss(self, state_seq, gt_seq, cond_seq, cond_raw_seq, contacts_seq,
                             angvel_seq, pose_hist_seq, batch):
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
            batch, start=start, length=window, train_mode=True
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

    def _compute_lookahead_loss(
        self,
        state_seq,
        gt_seq,
        cond_seq,
        cond_raw_seq,
        contacts_seq,
        angvel_seq,
        pose_hist_seq,
        batch,
    ):
        weight = float(getattr(self, 'lookahead_weight', 0.0) or 0.0)
        steps = int(getattr(self, 'lookahead_steps', 0) or 0)
        if weight <= 0.0 or steps <= 1:
            return None
        steps = min(steps, state_seq.shape[1])
        if steps <= 1:
            return None
        def _slice(tensor):
            if tensor is None:
                return None
            if torch.is_tensor(tensor) and tensor.dim() == 3 and tensor.size(1) >= steps:
                return tensor[:, :steps]
            return tensor
        state_sub = state_seq[:, :steps]
        gt_sub = gt_seq[:, :steps]
        cond_sub = _slice(cond_seq)
        cond_raw_sub = _slice(cond_raw_seq)
        contacts_sub = _slice(contacts_seq)
        angvel_sub = _slice(angvel_seq)
        pose_hist_sub = _slice(pose_hist_seq)
        preds_la, attn_la = self._rollout_sequence(
            state_sub,
            cond_sub,
            cond_raw_sub,
            contacts_seq=contacts_sub,
            angvel_seq=angvel_sub,
            pose_hist_seq=pose_hist_sub,
            gt_seq=gt_sub,
            mode='train_free',
            tf_ratio=0.0,
        )
        preds_la = self._ensure_rot6d_delta(preds_la)
        with self._amp_context(self.use_amp):
            out = self.loss_fn(preds_la, gt_sub, attn_weights=attn_la, batch=batch)
        if isinstance(out, tuple):
            la_loss, la_stats = out
        else:
            la_loss, la_stats = out, {}
        latent_payload = self._latent_consistency_penalty(
            preds_la,
            contacts_sub,
            angvel_sub,
            pose_hist_sub,
            step_index=-1,
            tag='lookahead/',
        )
        if latent_payload is not None:
            lat_loss, lat_stats = latent_payload
            la_loss = la_loss + lat_loss
            if isinstance(la_stats, dict):
                la_stats.update(lat_stats)
            else:
                la_stats = dict(lat_stats)
        return la_loss, la_stats or {}, steps

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
        schedule = getattr(self, 'lookahead_stage_schedule', None)
        overrides: Dict[str, Any] = {}
        if not schedule:
            return overrides

        def _coerce_like(current, new_val):
            if new_val is None:
                return None
            if isinstance(current, bool):
                if isinstance(new_val, str):
                    return new_val.strip().lower() not in ('0', 'false', 'no', 'off')
                return bool(new_val)
            if isinstance(current, int) and not isinstance(current, bool):
                try:
                    return int(round(float(new_val)))
                except Exception:
                    return current
            if isinstance(current, float):
                try:
                    return float(new_val)
                except Exception:
                    return current
            return new_val

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
            coerced = _coerce_like(current, value) if current is not None else value
            setattr(target, attr_name, coerced)
            key_name = key if prefix else attr_name
            overrides[key_name] = coerced
            return True

        selected = None
        for stage in schedule:
            try:
                st = int(stage.get('start'))
                ed = int(stage.get('end'))
            except Exception:
                continue
            if st <= epoch <= ed:
                params = stage.get('params') or {}
                selected = {'start': st, 'end': ed, 'params': params, 'label': stage.get('label')}
                for key, value in params.items():
                    _assign(key, value)
                break

        if selected:
            label = f" {selected['label']}" if selected.get('label') else ''
            if overrides:
                summary = ', '.join(f"{k}={overrides[k]}" for k in sorted(overrides))
            else:
                summary = 'no overrides'
            print(
                "[StageSched]"
                f"[ep {epoch:03d}] stage={selected['start']}-{selected['end']}{label} | "
                f"{summary}"
            )
        return overrides
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
    def __init__(self, model, loss_fn, lr=0.0001, grad_clip=0.0, weight_decay=0.01, tf_warmup_steps=0, tf_total_steps=0, augmentor=None, use_amp=None, accum_steps=1, *, pin_memory=False):
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
        self.lookahead_steps: int = 0
        self.lookahead_weight: float = 0.0
        self.lookahead_stage_schedule = []

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
        self.w_latent_consistency: float = 0.0
        self._latent_consistency_dim_warned: bool = False
        self._latent_encoder_warned: bool = False
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
                if ep <= tf_start: tf_ratio = tf_max_epoch
                elif ep >= tf_end: tf_ratio = tf_min_epoch
                else:
                    r = (ep - tf_start) / max(1, (tf_end - tf_start))
                    tf_ratio = tf_max_epoch + (tf_min_epoch - tf_max_epoch) * r
            else:
                tf_ratio = tf_max_epoch
            self._last_tf_ratio = float(tf_ratio)
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

                # === 插入开始：一次性打印训练端 X(z) 的 RMS，验证不是 0 ===
                preds_dict, last_attn = self._rollout_sequence(
                    state_seq,
                    cond_seq,
                    cond_raw_seq,
                    contacts_seq=contacts_seq,
                    angvel_seq=angvel_seq,
                    pose_hist_seq=pose_hist_seq,
                    gt_seq=gt_seq,
                    mode='mixed',
                    tf_ratio=tf_ratio,
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

                lookahead_payload = self._compute_lookahead_loss(
                    state_seq,
                    gt_seq,
                    cond_seq,
                    cond_raw_seq,
                    contacts_seq,
                    angvel_seq,
                    pose_hist_seq,
                    batch,
                )
                if lookahead_payload is not None:
                    la_loss, la_stats, la_steps = lookahead_payload
                    la_weight = float(getattr(self, 'lookahead_weight', 0.0) or 0.0)
                    if la_weight > 0.0:
                        loss = loss + la_weight * la_loss
                    stats['lookahead_loss'] = float(la_loss.detach().cpu())
                    stats['lookahead/weight'] = float(la_weight)
                    stats['lookahead/steps'] = float(la_steps)
                    if isinstance(la_stats, dict):
                        for fk, fv in la_stats.items():
                            stats[f'lookahead/{fk}'] = fv

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
                    if getattr(self, 'val_mode', 'none') == 'online' and not bool(getattr(self, 'no_monitor', False)):
                        vloader = self.train_loader
                        _mon_batches = int(getattr(self, 'monitor_batches', 8) or 8)
                        free_metrics = dict(self.validate_autoreg_online(vloader, max_batches=_mon_batches))
                        free_metrics.setdefault('phase', 'freerun')
                        free_metrics['tf_ratio'] = float(getattr(self, '_last_tf_ratio', 1.0))
                        metrics_for_json = free_metrics
                        metrics_tag = 'valfree'
                        _metrics = free_metrics
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
                            f"[ValFree@ep {ep:03d}] "
                            f"MSEnormY={free_metrics.get('MSEnormY', float('nan')):.6f} | "
                            f"GeoDeg={free_metrics.get('GeoDeg', float('nan')):.3f}° | "
                            f"YawAbsDeg={free_metrics.get('YawAbsDeg', float('nan')):.3f} | "
                            f"RootVelMAE={free_metrics.get('RootVelMAE', float('nan')):.5f} | "
                            f"AngVelMAE={free_metrics.get('AngVelMAE', float('nan')):.5f} rad/s | "
                            f"AngMagRel={free_metrics.get('AngVelMagRel', float('nan')):.3f}" + _extra
                        )
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
            except Exception as _e:
                phase_label = 'ValTeacher' if is_teacher_phase else 'ValFree'
                print(f"[{phase_label}@ep {ep:03d}] skipped due to error: {_e}")

            if metrics_for_json is not None and metrics_tag is not None:
                self._record_epoch_metrics(metrics_for_json, tag=metrics_tag, epoch=ep)
                self._dump_metrics_json(metrics_for_json, tag=metrics_tag, epoch=ep)
            if (not is_teacher_phase) and teacher_metrics_cached is not None:
                self._record_epoch_metrics(teacher_metrics_cached, tag='teacher', epoch=ep)
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
        """
        if scales is None or raw_flat.numel() == 0:
            return raw_flat
        z = torch.tanh(raw_flat / scales.clamp_min(1e-6))
        if mu is not None and std is not None:
            z = (z - mu) / std.clamp_min(1e-6)
        return z

    @staticmethod
    def _pose_hist_inverse_vec(norm_flat: torch.Tensor, scales: Optional[torch.Tensor], mu: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Inverse of VectorTanhNormalizer.transform on flattened pose-history vectors.
        """
        if scales is None or norm_flat.numel() == 0:
            return norm_flat
        z = norm_flat
        if mu is not None and std is not None:
            z = z * std.clamp_min(1e-6) + mu
        eps = 1.0 - 1e-6
        z = z.clamp(min=-eps, max=eps)
        if hasattr(torch, "atanh"):
            raw = torch.atanh(z) * scales
        else:
            raw = 0.5 * (torch.log1p(z) - torch.log1p(-z)) * scales
        return raw

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
        return _diagnose_free_run_impl(
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
def _speed_from_X_layout(X, state_layout):
    """
    返回每帧“速度标量”（z 域）用于 IndexOpt 的 min_speed 过滤。
    兼容 state_layout 形态：
      - {'RootVelocity': {'start': 3, 'size': 2}}
      - {'RootVelocity': (3, 5)}   # [start, end)
      - {'RootVelocity': (3, 2)}   # [start, size]
      - {'RootVelocity': [3, 2]}   # 同上
      - 数值为 str/np 标量时自动 int()
    """
    import numpy as np

    try:
        if not isinstance(state_layout, dict):
            return None

        # ✅ 优先使用你已有的解析工具，避免重复实现
        try:
            sl = parse_layout_entry(state_layout.get('RootVelocity'), 'RootVelocity')
        except Exception:
            sl = None

        a = b = None
        if isinstance(sl, slice):
            a = int(sl.start or 0)
            b = int(sl.stop)
        else:
            rv = state_layout.get('RootVelocity')
            if rv is None:
                return None

            # dict 形式: {'start': x, 'size': y}
            if isinstance(rv, dict):
                a = int(rv.get('start', 0))
                sz = int(rv.get('size', 0))
                b = a + sz
            else:
                # 序列形式: (start, size) 或 (start, end)
                if hasattr(rv, '__iter__'):
                    r0 = int(rv[0])
                    r1 = int(rv[1])
                    # 判断 r1 是 size 还是 end（RootVelocity 通常 2 或 3 维）
                    if r1 in (1, 2, 3):
                        a, b = r0, r0 + r1
                    else:
                        a, b = r0, r1
                else:
                    return None

        if b is None or b <= a:
            return None

        v = X[:, a:b]
        spd = np.linalg.norm(v[:, :2], axis=1) if v.shape[1] >= 2 else np.abs(v[:, 0])
        return spd.astype(np.float32)
    except Exception:
        # 紧急兜底：过滤失效，但不会中断训练
        return None


def _maybe_optimize_dataset_index(ds, args):
    """
    Rebuild ds.index using a stride strategy (optionally filtered by root speed).
    Safe to call for both train/val.
    """
    try:
        stride = max(1, int(getattr(args, 'index_stride', 1)))
        min_speed = float(getattr(args, 'min_speed', 0.0))
    except Exception:
        stride, min_speed = (1, 0.0)
    new_index = []
    for cid, clip in enumerate(ds.clips):
        X = clip.X
        T = int(X.shape[0])
        L = ds.seq_len
        starts = list(range(0, max(0, T - L + 1), stride))
        if min_speed > 0.0:
            spd = _speed_from_X_layout(X, clip.state_layout_norm)
            if spd is not None and spd.shape[0] >= T:
                keep = []
                for s in starts:
                    e = s + L
                    seg_spd = spd[s:e]
                    if seg_spd.mean() >= min_speed or seg_spd.max() >= min_speed * 0.8:
                        keep.append(s)
                starts = keep if keep else starts
        for s in starts:
            new_index.append((cid, int(s)))
    if new_index:
        ds.index = new_index
        print(f'[IndexOpt] stride={stride} min_speed={min_speed} -> windows={len(ds.index)}')
    else:
        print('[IndexOpt] No windows built; keep original.')
    return ds




# === [ARPG PATCH] One-shot normalization diagnostics (safe & self-contained) ===
def _norm_debug_once(trainer, train_loader, thr=8.0, topk=8, print_to_console=True, writer=None, tag_prefix="NormDiag"):
    import numpy as np
    try:
        import torch  # noqa: F401
    except Exception:
        torch = None

    def _to_np(x):
        if x is None:
            return None
        try:
            import torch as _t
            if isinstance(x, _t.Tensor):
                return x.detach().cpu().float().numpy()
        except Exception as _err:
            print(f"[Norm-ERR] failed to convert tensor to numpy during diagnostics: {_err}")
            pass
        try:
            return np.asarray(x)
        except Exception:
            return None

    try:
        batch = next(iter(train_loader))
    except Exception as e:
        if print_to_console:
            print("[NormDiag] cannot fetch batch:", e)
        return

    xz = yz = None
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 1: xz = _to_np(batch[0])
        if len(batch) >= 2: yz = _to_np(batch[1])
    elif isinstance(batch, dict):
        vals = [v for v in batch.values() if _to_np(v) is not None]
        if len(vals) >= 1: xz = _to_np(vals[0])
        if len(vals) >= 2: yz = _to_np(vals[1])
    else:
        xz = _to_np(batch)

    mu_x = _to_np(getattr(trainer, "mu_x", None))
    se_x = _to_np(getattr(trainer, "std_x", None))
    mu_y = _to_np(getattr(trainer, "mu_y", None))
    se_y = _to_np(getattr(trainer, "std_y", None))

    yaw_x     = getattr(trainer, "yaw_x_slice", None)
    rootvel_x = getattr(trainer, "rootvel_x_slice", None)
    angvel_x  = getattr(trainer, "angvel_x_slice", None)
    yaw_y     = getattr(trainer, "yaw_slice", None)
    rootvel_y = getattr(trainer, "rootvel_slice", None)
    angvel_y  = getattr(trainer, "angvel_slice", None)

    def _slice_from_layout(layout, key):
        v = None if layout is None else layout.get(key)
        if v is None: return None
        try:
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                s, l = int(v[0]), int(v[1])
                return slice(s, s + l)
        except Exception:
            return None
        return None

    rot6d_x = _slice_from_layout(getattr(trainer, "_x_layout", None), "BoneRotations6D")
    rot6d_y = _slice_from_layout(getattr(trainer, "_y_layout", None), "BoneRotations6D")

    def _sel(z, sl):
        if z is None or sl is None: return None
        try:
            return z[..., sl]
        except Exception:
            return None

    def _to_2d_last(z):
        if z is None: return None
        z = _to_np(z)
        if z is None: return None
        if z.ndim == 0: return None
        if z.ndim == 1: return z[None, :]
        if z.ndim >= 2: return z.reshape(-1, z.shape[-1])
        return None

    def _z_stats(z, name, thr):
        z2 = _to_2d_last(z)
        if z2 is None or z2.size == 0: return None
        absz = np.abs(z2)
        pct = float((absz > thr).mean() * 100.0)
        return dict(name=name,
                    absmax=float(np.nanmax(absz)),
                    p99=float(np.nanpercentile(absz, 99.0)),
                    p999=float(np.nanpercentile(absz, 99.9)),
                    pct_over_thr=pct,
                    mean=float(np.nanmean(z2)),
                    std=float(np.nanstd(z2)))

    def _raw_stats(z, mu, se, name):
        z2 = _to_2d_last(z)
        if z2 is None or mu is None or se is None: return None
        mu = _to_np(mu); se = _to_np(se)
        if mu is None or se is None: return None
        if z2.shape[-1] != mu.shape[-1] or mu.shape[-1] != se.shape[-1]:
            return None
        raw = z2 * se + mu
        return dict(name=name,
                    min=float(np.nanmin(raw)),
                    p1=float(np.nanpercentile(raw, 1.0)),
                    p50=float(np.nanpercentile(raw, 50.0)),
                    p99=float(np.nanpercentile(raw, 99.0)),
                    max=float(np.nanmax(raw)))

    def _roundtrip(z, mu, se):
        z2 = _to_2d_last(z); mu = _to_np(mu); se = _to_np(se)
        if z2 is None or mu is None or se is None: return None
        if z2.shape[-1] != mu.shape[-1] or mu.shape[-1] != se.shape[-1]: return None
        raw = z2 * se + mu
        z3 = (raw - mu) / se
        return float(np.nanmax(np.abs(z3 - z2)))

    if print_to_console:
        def _sh(x):
            try: return tuple(np.asarray(x).shape)
            except Exception: return None
        print("[NormDiag] shapes: Xz", _sh(xz), "Yz", _sh(yz),
              "muX", None if mu_x is None else len(mu_x),
              "seX", None if se_x is None else len(se_x),
              "muY", None if mu_y is None else len(mu_y),
              "seY", None if se_y is None else len(se_y))

    rows_z = [
        _z_stats(_sel(xz, yaw_x),     "X.yaw(z)", thr),
        _z_stats(_sel(xz, rootvel_x), "X.rootvel(z)", thr),
        _z_stats(_sel(xz, angvel_x),  "X.angvel(z)", thr),
        _z_stats(_sel(xz, rot6d_x),   "X.rot6d(z)", thr),
        _z_stats(_sel(yz, yaw_y),     "Y.yaw(z)", thr),
        _z_stats(_sel(yz, rootvel_y), "Y.rootvel(z)", thr),
        _z_stats(_sel(yz, angvel_y),  "Y.angvel(z)", thr),
        _z_stats(_sel(yz, rot6d_y),   "Y.rot6d(z)", thr),
    ]

    rows_raw = [
        _raw_stats(_sel(xz, yaw_x),     mu_x[yaw_x]     if (mu_x is not None and yaw_x     is not None) else None,
                                       se_x[yaw_x]     if (se_x is not None and yaw_x     is not None) else None, "X.yaw(raw)"),
        _raw_stats(_sel(xz, rootvel_x), mu_x[rootvel_x] if (mu_x is not None and rootvel_x is not None) else None,
                                       se_x[rootvel_x] if (se_x is not None and rootvel_x is not None) else None, "X.rootvel(raw)"),
        _raw_stats(_sel(xz, angvel_x),  mu_x[angvel_x]  if (mu_x is not None and angvel_x  is not None) else None,
                                       se_x[angvel_x]  if (se_x is not None and angvel_x  is not None) else None, "X.angvel(raw)"),
        _raw_stats(_sel(xz, rot6d_x),   mu_x[rot6d_x]   if (mu_x is not None and rot6d_x   is not None) else None,
                                       se_x[rot6d_x]   if (se_x is not None and rot6d_x   is not None) else None, "X.rot6d(raw)"),
        _raw_stats(_sel(yz, rot6d_y),   mu_y[rot6d_y]   if (mu_y is not None and rot6d_y   is not None) else None,
                                       se_y[rot6d_y]   if (se_y is not None and rot6d_y   is not None) else None, "Y.rot6d(raw)"),
    ]

    err_x = _roundtrip(xz, mu_x, se_x)
    err_y = _roundtrip(yz, mu_y, se_y)

    top_dims = None
    X2 = _to_2d_last(xz)
    if X2 is not None and X2.size:
        Z = np.abs(X2)
        p = np.nanpercentile(Z, 99.0, axis=0)
        order = np.argsort(-p)
        k = min(int(topk), p.shape[0])
        idx = order[:k]
        top_dims = [(int(i), float(p[i])) for i in idx]

    if print_to_console:
        print(f"[NormDiag] roundtrip_err: X={err_x}  Y={err_y}  thr={thr}")
        for r in rows_z:
            if r is not None:
                print("[NormDiag] {name}: absmax={absmax:.3f} p99={p99:.3f} p999={p999:.3f} "
                      "pct>|thr|={pct_over_thr:.2f}% mean={mean:.3f} std={std:.3f}".format(**r))
        for r in rows_raw:
            if r is not None:
                print("[NormDiag] {name}: min={min:.4f} p1={p1:.4f} p50={p50:.4f} p99={p99:.4f} max={max:.4f}".format(**r))
        if top_dims is not None:
            print(f"[NormDiag] X top-{topk} dims by p99(|z|):", top_dims)

    if writer is not None and hasattr(writer, "add_histogram"):
        try:
            import torch as _t
            def _h(arr, tag):
                if arr is None: return
                writer.add_histogram(f"{tag_prefix}/{tag}", _t.as_tensor(arr), 0)
            _h(_sel(xz, yaw_x), "X_yaw")
            _h(_sel(xz, rootvel_x), "X_rootvel")
            _h(_sel(xz, angvel_x), "X_angvel")
            _h(_sel(xz, rot6d_x), "X_rot6d")
            _h(_sel(yz, rot6d_y), "Y_rot6d")
        except Exception as e:
            if print_to_console:
                print("[NormDiag] histogram failed:", e)


def train_entry():
    global GLOBAL_ARGS
    import argparse, warnings, os, glob, time, math, json, ast
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    # ---- Slice helpers (inserted) ----
    def _safe_set_slice(obj, attr, maybe_slice):
        """Assign only if maybe_slice is a slice; avoid overwriting valid slices with None."""
        if isinstance(maybe_slice, slice):
            setattr(obj, attr, maybe_slice)

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
                   help='自由滚动 loss 的初始权重（未指定则按最终权重的 20% 推断，并在 ramp_epochs 内过渡）。')
    p.add_argument('--freerun_horizon_min', type=int, default=6,
                   help='自由滚动窗口的最小 horizon（默认 6，对应 100ms 左右）。')
    p.add_argument('--freerun_init_horizon', type=int, default=None,
                   help='训练早期的初始 horizon，上限不超过 --freerun_horizon。未指定时自动取约 70% 的最终 horizon。')
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
    p.add_argument('--lookahead_steps', type=int, default=0,
                   help='>1 时，启用 train_free lookahead loss 的窗口长度')
    p.add_argument('--lookahead_weight', type=float, default=0.0,
                   help='train_free lookahead loss 的权重')
    p.add_argument('--lookahead_stage_schedule', type=str, default=None,
                   help='按阶段调整 lookahead/freerun/tf 的日程表，格式如 "1-3:lookahead_steps=3,lookahead_weight=0.3;4-6:lookahead_steps=6,lookahead_weight=0.4"')
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
    p.add_argument('--w_rot_geo', type=float, default=0.01)
    p.add_argument('--w_rot_delta', type=float, default=1.0)
    p.add_argument('--w_rot_delta_root', type=float, default=0.0)
    p.add_argument('--w_rot_log', type=float, default=0.0)
    p.add_argument('--w_cond_yaw', type=float, default=None,
                   help='yaw 指令对齐损失权重（默认 0.1 * w_rot_delta，<=0 关闭）。')
    p.add_argument('--w_fk_pos', type=float, default=0.0,
                   help='FK 末端位置损失权重（0 表示禁用）。')
    p.add_argument('--w_rot_local', type=float, default=0.0,
                   help='父子关节局部 geodesic 约束权重（0=关闭）。')
    p.add_argument('--w_latent_consistency', type=float, default=0.0,
                   help='Latent consistency loss 权重，用于约束 free-run / lookahead 隐状态落在预训练流形内。')
    p.add_argument('--cond_yaw_min_speed', type=float, default=0.1,
                   help='仅对速度大于该阈值 (m/s) 的帧应用 yaw 指令损失。')
    p.add_argument('--seq_len', type=int, default=120)
    p.add_argument('--yaw_aug_deg', type=float, default=0.0)
    p.add_argument('--normalize_c', action='store_true')
    p.add_argument('--aug_noise_std', type=float, default=0.0)
    p.add_argument('--aug_time_warp_prob', type=float, default=0.0)
    # TensorBoard 相关逻辑已移除，避免冗余参数
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--foot_contact_threshold', type=float, default=1.5, help='角速度阈值（rad/s），低于该值视为脚接触')
    p.add_argument('--monitor_batches', type=int, default=2, help='每个 epoch 在线指标采样的批次数')
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

    def _arg(name, default=None):
        return getattr(GLOBAL_ARGS, name, default)


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
        pose_hist_dim=getattr(ds_train, 'pose_hist_dim', 0),
    ).to(device)
    validate_and_fix_model_(model, Dx, Dc)
    validate_and_fix_model_(model)
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
    else:
        print("[MPL][WARN] encoder_path not provided; proceeding without frozen encoder.")

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
    w_cond_yaw = _arg('w_cond_yaw', None)
    if w_cond_yaw is None:
        w_cond_yaw = max(0.0, 0.1 * w_rot_delta)
    cond_yaw_min_speed = max(0.0, float(_arg('cond_yaw_min_speed', 0.1)))
    w_fk_pos = float(_arg('w_fk_pos', 0.0) or 0.0)
    w_rot_local = float(_arg('w_rot_local', 0.0) or 0.0)
    loss_fn = MotionJointLoss(
        output_layout=ds_train.output_layout,
        fps=fps_data,
        rot6d_spec=getattr(ds_train, 'rot6d_spec', {}),
        w_rot_geo=_arg('w_rot_geo', 0.01),
        w_rot_delta=w_rot_delta,
        w_rot_ortho=_arg('w_rot_ortho', 0.001),
        w_cond_yaw=w_cond_yaw,
        cond_yaw_min_speed=cond_yaw_min_speed,
        meta=None,
        w_fk_pos=w_fk_pos,
        w_rot_local=w_rot_local,
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
        f"w_rot_geo={loss_fn.w_rot_geo} "
        f"w_rot_delta={loss_fn.w_rot_delta} "
        f"w_rot_ortho={loss_fn.w_rot_ortho} "
        f"w_cond_yaw={loss_fn.w_cond_yaw} "
        f"w_fk_pos={loss_fn.w_fk_pos} "
        f"w_rot_local={loss_fn.w_rot_local}"
    )

    loss_fn.dt_traj = 1.0 / max(1e-6, fps_data)
    loss_fn.dt_bone = 1.0 / max(1e-6, fps_data)
    print(f"[Dt] dt_traj={loss_fn.dt_traj:.6f}s | dt_bone={loss_fn.dt_bone:.6f}s (dataset fps={fps_data})")

    if hasattr(loss_fn, 'rot6d_eps'):
        loss_fn.rot6d_eps = 1e-6
    augmentor = MotionAugmentation(noise_std=_arg('aug_noise_std', 0.0), time_warp_prob=_arg('aug_time_warp_prob', 0.0))
    trainer = Trainer(model=model, loss_fn=loss_fn, lr=_arg('lr', 0.0001), grad_clip=_arg('grad_clip', 0.0), weight_decay=_arg('weight_decay', 0.01), tf_warmup_steps=_arg('tf_warmup_steps', 5000), tf_total_steps=_arg('tf_total_steps', 200000), augmentor=augmentor, use_amp=_arg('amp', False), accum_steps=_arg('accum_steps', 1), pin_memory=pin)
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


    _safe_set_slice(trainer, 'yaw_x_slice', parse_layout_entry(trainer._x_layout.get('RootYaw'), 'RootYaw'))
    _safe_set_slice(trainer, 'rootvel_x_slice', parse_layout_entry(trainer._x_layout.get('RootVelocity'), 'RootVelocity'))
    _safe_set_slice(trainer, 'angvel_x_slice', parse_layout_entry(trainer._x_layout.get('BoneAngularVelocities'), 'BoneAngularVelocities'))

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
    trainer.lookahead_steps = int(_arg('lookahead_steps', 0) or 0)
    trainer.lookahead_weight = float(_arg('lookahead_weight', 0.0) or 0.0)
    trainer.lookahead_stage_schedule = _parse_stage_schedule(_arg('lookahead_stage_schedule', None))
    trainer.w_latent_consistency = float(_arg('w_latent_consistency', 0.0) or 0.0)
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
    trainer.teacher_rot_noise_deg = float(_arg('teacher_rot_noise_deg', 0.0))
    trainer.teacher_rot_noise_prob = float(_arg('teacher_rot_noise_prob', 0.0))
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


def _first_linear_in_features(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    return None

def sanity_check_model_dims(model, Dx, Dy, Dc):
    nin = _first_linear_in_features(model)
    if nin is not None and nin != Dx + Dc:
        raise RuntimeError(f'[Guard] 模型第一层 in_features={nin}，但应为 Dx+Dc={Dx + Dc}；很可能构建时把 in_dim 设成 Dy+Dc={Dy + Dc} 了。')

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
    print(f'[ARPG-PATCH] 参数准备完毕，即将进入训练入口: train_entry()')
    sys.argv = rest_argv
    try:
        train_entry()
    finally:
        sys.argv = argv0
if __name__ == '__main__':
    main()
