"""
几何与旋转数学工具库

提供旋转表示转换、测地线计算、角速度计算等功能。
用于运动学骨骼动画中的旋转处理。

从 training_MPL.py 重构而来，作为独立工具模块。
"""
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np


# ============================================
# 旋转表示转换 (6D Rotation)
# ============================================

def rot6d_to_matrix(xJ6: torch.Tensor, *, columns=("X", "Z")) -> torch.Tensor:
    """
    将 6D 旋转表示转换为 3x3 旋转矩阵

    Args:
        xJ6: (..., J, 6) - 6D 旋转表示 (提供两列)
        columns: 指定提供的列 (默认 X 和 Z)
                 第三列通过叉积派生: remaining = second × first

    Returns:
        (..., J, 3, 3) - 旋转矩阵 (列即轴)
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
    remaining = [ax for ax in ("X", "Y", "Z") if ax not in (ax1, ax2)][0]
    def _cross(u, v):  # torch.cross(u, v)
        return torch.cross(u, v, dim=-1)
    r[remaining] = _norm(_cross(r[ax2], r[ax1]))

    # 组装列为轴
    RX = r["X"].unsqueeze(-1)
    RY = r["Y"].unsqueeze(-1)
    RZ = r["Z"].unsqueeze(-1)
    R = torch.cat([RX, RY, RZ], dim=-1)  # (..., J, 3, 3)

    # 行列式修正：若 det<0，仅翻转"派生列"
    orig_shape = R.shape[:-2]  # (..., J)
    det = torch.linalg.det(R.reshape(-1, 3, 3)).reshape(orig_shape)
    neg = (det < 0.0).unsqueeze(-1)  # (..., J, 1)

    col_idx = {"X": 0, "Y": 1, "Z": 2}[remaining]
    R = R.clone()
    # R[..., :, col_idx] 选择的是"该列"的 3 个分量（行方向切片）
    R[..., :, col_idx] = torch.where(neg, -R[..., :, col_idx], R[..., :, col_idx])

    return R


def matrix_to_rot6d(R: torch.Tensor, *, columns=("X", "Z")) -> torch.Tensor:
    """
    逆变换：从旋转矩阵提取 6D 表示

    Args:
        R: (..., J, 3, 3) - 旋转矩阵
        columns: 要提取的两列轴

    Returns:
        (..., J, 6) - 6D 旋转表示
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
    生成与输入同形状的 6D 单位旋转表示

    Args:
        residual: (..., J, 6) - 参考形状
        columns: 轴列定义

    Returns:
        (..., J, 6) - 单位旋转 (I 矩阵对应的 6D 表示)
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
    将模型输出的 Δrot6d 残差转换为有效的 6D 旋转 (接近单位旋转)

    Args:
        delta_flat: (..., J*6) - 残差形式的旋转增量
        columns: 轴列定义

    Returns:
        (..., J, 6) - 归一化后的 6D 旋转增量
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
    合成 6D 旋转增量：R_next = ΔR @ R_prev

    Args:
        prev_rot6d: (..., J*6) - 上一帧的绝对 6D 旋转
        delta_rot6d: (..., J*6) - 模型预测的增量 Δrot6d
        columns: 轴列定义

    Returns:
        (..., J*6) - 新的绝对 6D 旋转
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
    从绝对 rot6d 序列推导相邻帧 delta

    Args:
        rot6d_seq: (..., T, J*6) - 绝对旋转序列
        columns: 轴列定义

    Returns:
        (..., T, J*6) - delta 序列 (首帧为单位旋转)
        None - 如果输入无效
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


def reproject_rot6d(flat_6d: torch.Tensor) -> torch.Tensor:
    """
    Gram-Schmidt 正交化 6D 旋转表示

    确保两列单位且正交，用于修正数值误差。

    Args:
        flat_6d: (..., J*6) - 可能有误差的 6D 旋转表示

    Returns:
        (..., J*6) - 正交化后的 6D 旋转表示
    """
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


# ============================================
# 轴角 (Axis-Angle) 转换
# ============================================

def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues 旋转公式：轴角 -> 旋转矩阵

    Args:
        axis: (..., 3) - 旋转轴 (单位向量)
        angle: (...) - 旋转角 (弧度)

    Returns:
        (..., 3, 3) - 旋转矩阵
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


# ============================================
# 测地线距离 (Geodesic Distance)
# ============================================

def geodesic_R(R_pred: torch.Tensor, R_gt: torch.Tensor, *, reduce=None) -> torch.Tensor:
    """
    计算 SO(3) 流形上的测地线角距离 (弧度)

    Args:
        R_pred: (..., J, 3, 3) - 预测的旋转矩阵
        R_gt: (..., J, 3, 3) - 真值旋转矩阵
        reduce: None | "mean" | "sum" - 归约方式

    Returns:
        (..., J) - 每个关节的角距离 (弧度), 或归约后的标量
    """
    assert R_pred.shape[-2:] == (3, 3) and R_gt.shape[-2:] == (3, 3)
    Rt = torch.matmul(R_pred.transpose(-1, -2), R_gt)  # (..., J, 3, 3)
    # Clamp trace for numerical stability
    trace = Rt[..., 0, 0] + Rt[..., 1, 1] + Rt[..., 2, 2]
    cos = (trace - 1.) * 0.5
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    ang = torch.acos(cos)  # radians
    if reduce == "mean":
        return ang.mean()
    if reduce == "sum":
        return ang.sum()
    return ang


def _matrix_log_map(R: torch.Tensor) -> torch.Tensor:
    """
    Log map from SO(3) to so(3) as a 3D rotation vector

    Args:
        R: (..., 3, 3) - 旋转矩阵

    Returns:
        (..., 3) - 旋转向量 (axis * angle)
    """
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


def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """
    SO(3) log map: 旋转矩阵 -> 轴角向量

    Args:
        R: [..., 3, 3] - 旋转矩阵

    Returns:
        [..., 3] - 轴角向量 phi = axis * angle (弧度 * 轴)
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


# ============================================
# 角速度计算 (Angular Velocity)
# ============================================

def angvel_vec_from_R_seq(R_seq: torch.Tensor, fps: float) -> torch.Tensor:
    """
    从旋转序列计算角速度向量

    Args:
        R_seq: [B, T, J, 3, 3] - 旋转矩阵序列
        fps: float - 采样帧率

    Returns:
        omega: [B, T-1, J, 3] - 角速度 (rad/s)
    """
    dR = torch.matmul(R_seq[:, 1:], R_seq[:, :-1].transpose(-1, -2))
    phi = so3_log_map(dR)
    omega = phi * float(fps)
    return omega


# ============================================
# 骨骼层级工具 (Skeleton Hierarchy)
# ============================================

def root_relative_matrices(R: torch.Tensor, root_idx: int) -> torch.Tensor:
    """
    将关节旋转表达为相对于根关节的局部坐标系

    Args:
        R: (..., J, 3, 3) - 世界坐标系下的关节旋转
        root_idx: int - 根关节索引

    Returns:
        (..., J, 3, 3) - 根关节坐标系下的关节旋转
    """
    if root_idx < 0 or root_idx >= R.shape[-3]:
        return R
    R_root = R[..., root_idx, :, :]
    R_root_T = R_root.transpose(-1, -2).unsqueeze(-3)
    return torch.matmul(R_root_T, R)


# 别名保持向后兼容
_root_relative_matrices = root_relative_matrices


# ============================================
# Numpy 版本工具 (用于数据预处理)
# ============================================

def wrap_to_pi_np(x: np.ndarray) -> np.ndarray:
    """
    将角度包裹到 [-π, π) 范围

    Args:
        x: np.ndarray - 角度 (弧度)

    Returns:
        np.ndarray - 包裹后的角度
    """
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def gram_schmidt_renorm_np(rot6d: np.ndarray) -> np.ndarray:
    """
    Numpy 版本的 Gram-Schmidt 正交化

    Args:
        rot6d: (..., 6) - 6D 旋转表示，列A = X, 列B = Z

    Returns:
        (..., 6) - 正交化后的 6D 旋转表示
    """
    if rot6d.shape[-1] != 6:
        raise ValueError(f"Expected last dim=6, got {rot6d.shape[-1]}")

    a = rot6d[..., :3].copy()
    b = rot6d[..., 3:].copy()

    # 归一化第一列
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    a = a / np.maximum(a_norm, 1e-8)

    # 正交化第二列
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    b = b / np.maximum(b_norm, 1e-8)

    return np.concatenate([a, b], axis=-1)
