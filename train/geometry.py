"""
几何与旋转数学工具库

提供旋转表示转换、测地线计算、角速度计算等功能。
用于运动学骨骼动画中的旋转处理。

从 training_MPL.py 重构而来，作为独立工具模块。
本模块是 train/ 子系统中几何原子能力的唯一来源（single source of truth）。
"""
import math as _math
from typing import Optional, Sequence, Tuple
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


def compose_rot6d_delta(
    prev_rot6d: torch.Tensor,
    delta_rot6d: torch.Tensor,
    *,
    columns=("X", "Z"),
    reproject_result: bool = True,
) -> torch.Tensor:
    """
    合成 6D 旋转增量，允许在末尾携带非旋转通道（例如速度等附加输出）。

    - 对前 rot_len = floor(D/6)*6 维做旋转合成：R_next = ΔR @ R_prev
    - 对尾部残余通道（若有）执行残差相加：tail_next = tail_prev + tail_delta

    Args:
        prev_rot6d: (..., D)
        delta_rot6d: (..., D)
        columns: 6D 表示中使用的两列轴
        reproject_result: 是否在合成后将 R_next 投影回 SO(3)

    Returns:
        (..., D)
    """
    if prev_rot6d.shape != delta_rot6d.shape:
        raise ValueError(f"[compose_rot6d_delta] shape mismatch: prev={prev_rot6d.shape}, delta={delta_rot6d.shape}")
    D = prev_rot6d.shape[-1]
    rot_len = (D // 6) * 6
    if rot_len <= 0:
        raise ValueError(f"[compose_rot6d_delta] cannot infer rot6d part from dim={D}")

    # 旋转部分
    prev_rot = prev_rot6d[..., :rot_len]
    delta_rot = delta_rot6d[..., :rot_len]
    orig_shape = prev_rot.shape
    J = rot_len // 6
    prev = reproject_rot6d(prev_rot).view(*orig_shape[:-1], J, 6)
    delta = normalize_rot6d_delta(delta_rot, columns=columns)
    R_prev = rot6d_to_matrix(prev, columns=columns)
    R_delta = rot6d_to_matrix(delta, columns=columns)
    R_next = torch.matmul(R_delta, R_prev)
    if reproject_result:
        # 使用 SVD 将接近正交的矩阵精确投影回 SO(3)，抑制自回归链上的数值漂移
        R_next = orthogonalize_rotation_matrix(R_next)
    rot_next = matrix_to_rot6d(R_next, columns=columns).view(*orig_shape[:-1], rot_len)

    # 额外通道残差相加
    if rot_len == D:
        return rot_next.view(*prev_rot6d.shape)
    tail_prev = prev_rot6d[..., rot_len:]
    tail_delta = delta_rot6d[..., rot_len:]
    tail_next = tail_prev + tail_delta
    return torch.cat([rot_next, tail_next], dim=-1)


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


def orthogonalize_rotation_matrix(R: torch.Tensor) -> torch.Tensor:
    """
    使用 SVD 将接近正交的矩阵精确投影回 SO(3) 流形 (det=+1)。

    对于数值上接近旋转矩阵的 R，找到 Frobenius 范数下最近的正交矩阵：
        R ≈ U S V^T  ->  R_ortho = U D V^T,  D = diag(1, 1, sign(det(UV^T)))

    Args:
        R: (..., 3, 3) - 近似旋转矩阵

    Returns:
        (..., 3, 3) - 严格的旋转矩阵 (R^T R = I, det(R) = 1)
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"[orthogonalize_rotation_matrix] expects (..., 3, 3), got {R.shape}")

    orig_shape = R.shape
    R_flat = R.reshape(-1, 3, 3)  # (N, 3, 3)

    # SVD: R = U S V^T
    # 注意: 在 MPS 后端上，torch.linalg.svd 会自动回退到 CPU；这里统一使用 U.device
    U, _, Vh = torch.linalg.svd(R_flat)  # Vh is V^T
    device = U.device
    dtype = U.dtype

    # 最近的正交矩阵为 U D V^T，其中 D 的对角为 (1, 1, sign(det(UV^T)))
    UVt = torch.matmul(U, Vh)
    # 为避免在 MPS 上触发 det 的反向传播（需要未实现的 LU 求解），
    # 我们将 det 视为常数: 在 no_grad 上下文中计算其符号。
    with torch.no_grad():
        det = torch.linalg.det(UVt)  # (N,)
        sign = det.sign()

    # 构造 D: (N, 3, 3)，初始为单位矩阵，将最后一个奇异值乘以 det 的符号
    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    D = eye.repeat(U.shape[0], 1, 1)
    D[:, 2, 2] = sign.to(dtype=dtype)

    R_ortho = torch.matmul(torch.matmul(U, D), Vh)
    # 若 SVD 在 CPU 上运行，确保返回张量在与输入相同的设备上
    return R_ortho.to(R.device).reshape(orig_shape)


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


def so3_exp_map(omega: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """
    Exponential map on SO(3): omega (axis-angle vector) -> rotation matrix.

    Uses stable Rodrigues coefficients with Taylor expansions around theta=0, so
    gradients are well-defined when omega≈0 (unlike axis/angle normalization).

    Args:
        omega: (..., 3)
        eps: small-angle threshold for Taylor approximation.

    Returns:
        (..., 3, 3)
    """
    if omega.shape[-1] != 3:
        raise ValueError(f"[so3_exp_map] expects (...,3), got {omega.shape}")
    wx, wy, wz = omega.unbind(dim=-1)
    zeros = torch.zeros_like(wx)
    K = torch.stack(
        [
            torch.stack([zeros, -wz, wy], dim=-1),
            torch.stack([wz, zeros, -wx], dim=-1),
            torch.stack([-wy, wx, zeros], dim=-1),
        ],
        dim=-2,
    )  # (...,3,3)

    theta2 = (omega * omega).sum(dim=-1, keepdim=True)  # (...,1) non-negative in theory

    # A = sin(theta)/theta, B = (1-cos(theta))/theta^2
    # Taylor around theta=0:
    #   A ≈ 1 - t2/6 + t4/120
    #   B ≈ 1/2 - t2/24 + t4/720
    t2 = theta2
    t4 = t2 * t2
    A_taylor = 1.0 - t2 / 6.0 + t4 / 120.0
    B_taylor = 0.5 - t2 / 24.0 + t4 / 720.0

    # Use sqrt(theta2 + eps) to avoid NaN/Inf gradients at theta2≈0 (0 * inf -> NaN).
    # This mirrors the "sqrt(x + eps)" trick used elsewhere in this repo for stable gradients.
    sqrt_eps = 1e-8
    theta = torch.sqrt(theta2 + sqrt_eps)  # (...,1)
    A = torch.sin(theta) / theta
    B = (1.0 - torch.cos(theta)) / (theta2 + sqrt_eps)

    use_taylor = (theta2 < (eps * eps)).to(dtype=omega.dtype)
    A = A * (1.0 - use_taylor) + A_taylor * use_taylor
    B = B * (1.0 - use_taylor) + B_taylor * use_taylor

    # Broadcast A,B to (...,1,1)
    A = A.unsqueeze(-1)
    B = B.unsqueeze(-1)

    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(K.shape)
    K2 = torch.matmul(K, K)
    return eye + A * K + B * K2


def _apply_so3_correction_to_delta_raw(
    delta_raw: torch.Tensor,
    *,
    rot_slice: slice,
    omega_hat: Optional[torch.Tensor],
    gate_val: float,
    max_deg: float,
    columns: Sequence[str],
    omega_detach: bool,
) -> torch.Tensor:
    if omega_hat is None or float(gate_val) <= 1e-6:
        return delta_raw
    joint_count = int((rot_slice.stop - rot_slice.start) // 6)
    omega = (
        omega_hat[:, 0]
        if torch.is_tensor(omega_hat) and omega_hat.dim() == 4 and int(omega_hat.size(1)) == 1
        else omega_hat
    )
    if (not torch.is_tensor(omega)) or omega.shape[0] != delta_raw.shape[0] or omega.shape[-2:] != (joint_count, 3):
        return delta_raw
    omega_eff = (omega.detach() if bool(omega_detach) else omega).to(device=delta_raw.device, dtype=delta_raw.dtype)
    omega_eff = omega_eff * float(gate_val)
    if float(max_deg) > 0.0:
        max_rad = float(max_deg) * (_math.pi / 180.0)
        omega_eff = omega_eff * (
            max_rad / omega_eff.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        ).clamp_max(1.0)
    cols = tuple(columns) if isinstance(columns, (list, tuple)) and len(columns) >= 2 else ("X", "Z")
    delta6_proj = normalize_rot6d_delta(delta_raw[..., rot_slice], columns=cols)
    R_delta = rot6d_to_matrix(delta6_proj, columns=cols)
    R_used = torch.matmul(so3_exp_map(omega_eff), R_delta)
    delta6_used_abs = matrix_to_rot6d(R_used, columns=cols)
    delta6_used = delta6_used_abs - _rot6d_identity_like(delta6_used_abs, columns=cols)
    corrected = delta_raw.clone()
    corrected[..., rot_slice] = delta6_used.reshape(delta_raw.shape[0], joint_count * 6)
    return corrected


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
    trace = Rt[..., 0, 0] + Rt[..., 1, 1] + Rt[..., 2, 2]
    cos = (trace - 1.0) * 0.5
    cos = cos.clamp(-1.0, 1.0)
    skew = Rt - Rt.transpose(-1, -2)
    vec = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1) * 0.5
    sin = vec.norm(dim=-1)
    ang = torch.atan2(sin, cos)  # radians in [0, pi]
    if reduce == "mean":
        return ang.mean()
    if reduce == "sum":
        return ang.sum()
    return ang


def geodesic_R_safe(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    *,
    reduce=None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    数值更稳健的 SO(3) 测地线角距离。

    与 `geodesic_R` 的区别：
    - `sin` 项使用 `sqrt(sum(vec^2) + eps)`，在 `vec≈0` 时避免 `0*inf` 类梯度问题。

    Args:
        R_pred: (..., J, 3, 3) - 预测旋转矩阵
        R_gt: (..., J, 3, 3) - 真值旋转矩阵
        reduce: None | "mean" | "sum"
        eps: float - sqrt 内部的稳定项

    Returns:
        (..., J) 或归约后的标量，单位弧度
    """
    assert R_pred.shape[-2:] == (3, 3) and R_gt.shape[-2:] == (3, 3)
    Rt = torch.matmul(R_pred.transpose(-1, -2), R_gt)  # (..., J, 3, 3)
    trace = Rt[..., 0, 0] + Rt[..., 1, 1] + Rt[..., 2, 2]
    cos = (trace - 1.0) * 0.5
    cos = cos.clamp(-1.0, 1.0)
    skew = Rt - Rt.transpose(-1, -2)
    vec = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1) * 0.5
    sin = torch.sqrt((vec * vec).sum(dim=-1) + float(eps))
    ang = torch.atan2(sin, cos)
    if reduce == "mean":
        return ang.mean()
    if reduce == "sum":
        return ang.sum()
    return ang


def _matrix_log_map(R: torch.Tensor) -> torch.Tensor:
    """
    Log map from SO(3) to so(3) as a standard axis-angle / rotvec.

    Args:
        R: (..., 3, 3) - 旋转矩阵

    Returns:
        (..., 3) - 旋转向量 (axis * angle)
    """
    return so3_log_map(R)


def _axis_from_rotation_near_pi(R: torch.Tensor) -> torch.Tensor:
    """
    Recover a stable rotation axis when theta≈pi.

    For theta≈pi, vee(0.5 * (R-R^T)) = sin(theta) * axis becomes numerically tiny, so
    we extract the axis from the symmetric part of (R + I) / 2 instead.
    """
    eps = 1e-8
    xx = ((R[..., 0, 0] + 1.0) * 0.5).clamp_min(0.0)
    yy = ((R[..., 1, 1] + 1.0) * 0.5).clamp_min(0.0)
    zz = ((R[..., 2, 2] + 1.0) * 0.5).clamp_min(0.0)
    xy = (R[..., 0, 1] + R[..., 1, 0]) * 0.25
    xz = (R[..., 0, 2] + R[..., 2, 0]) * 0.25
    yz = (R[..., 1, 2] + R[..., 2, 1]) * 0.25

    idx = torch.stack([xx, yy, zz], dim=-1).argmax(dim=-1)
    ax = torch.zeros_like(xx)
    ay = torch.zeros_like(yy)
    az = torch.zeros_like(zz)

    mask_x = idx == 0
    if bool(mask_x.any().detach().cpu().item()):
        x = torch.sqrt(xx[mask_x].clamp_min(eps))
        ax[mask_x] = x
        ay[mask_x] = xy[mask_x] / x.clamp_min(eps)
        az[mask_x] = xz[mask_x] / x.clamp_min(eps)

    mask_y = idx == 1
    if bool(mask_y.any().detach().cpu().item()):
        y = torch.sqrt(yy[mask_y].clamp_min(eps))
        ay[mask_y] = y
        ax[mask_y] = xy[mask_y] / y.clamp_min(eps)
        az[mask_y] = yz[mask_y] / y.clamp_min(eps)

    mask_z = idx == 2
    if bool(mask_z.any().detach().cpu().item()):
        z = torch.sqrt(zz[mask_z].clamp_min(eps))
        az[mask_z] = z
        ax[mask_z] = xz[mask_z] / z.clamp_min(eps)
        ay[mask_z] = yz[mask_z] / z.clamp_min(eps)

    axis = torch.stack([ax, ay, az], dim=-1)
    norm = axis.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(axis)
    fallback[..., 0] = 1.0
    return torch.where(norm > 0.0, axis / norm.clamp_min(eps), fallback)


def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """
    SO(3) log map: 旋转矩阵 -> 标准 axis-angle / rotvec.

    Args:
        R: [..., 3, 3] - 旋转矩阵

    Returns:
        [..., 3] - rotvec = axis * angle (弧度)
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"[so3_log_map] expects (...,3,3), got {R.shape}")

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    W = R - R.transpose(-1, -2)
    vee = torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], dim=-1) * 0.5
    sin_theta = vee.norm(dim=-1)
    theta = torch.atan2(sin_theta, cos_theta)

    small = theta < 1e-6
    near_pi = (cos_theta < (-1.0 + 1e-4)) & (~small)

    scale = theta / sin_theta.clamp_min(1e-8)
    phi = vee * scale.unsqueeze(-1)
    phi = torch.where(small.unsqueeze(-1), vee, phi)

    if bool(near_pi.any().detach().cpu().item()):
        axis_pi = _axis_from_rotation_near_pi(R)
        phi_pi = axis_pi * theta.unsqueeze(-1)
        phi = torch.where(near_pi.unsqueeze(-1), phi_pi, phi)
    return phi


def _coerce_required_columns(columns: Sequence[str], *, caller: str) -> Tuple[str, str]:
    if not isinstance(columns, (list, tuple)) or len(columns) < 2:
        raise ValueError(f"[{caller}] columns must be an explicit sequence with at least two axes.")
    cols = (str(columns[0]), str(columns[1]))
    valid = {"X", "Y", "Z"}
    if cols[0] not in valid or cols[1] not in valid or cols[0] == cols[1]:
        raise ValueError(f"[{caller}] invalid columns={columns!r}; expected two distinct axes from X/Y/Z.")
    return cols


def _require_bounded_rot_slice(rot_slice: slice, *, caller: str) -> Tuple[int, int, int]:
    if not isinstance(rot_slice, slice) or rot_slice.start is None or rot_slice.stop is None:
        raise ValueError(f"[{caller}] rot_slice must be a bounded slice.")
    start = int(rot_slice.start)
    stop = int(rot_slice.stop)
    rot_len = stop - start
    if start < 0 or stop <= start or rot_len % 6 != 0:
        raise ValueError(f"[{caller}] invalid rot_slice={rot_slice}; length must be a positive multiple of 6.")
    return start, stop, rot_len


def compose_delta_raw_to_next(
    y_prev_raw: torch.Tensor,
    delta_raw: torch.Tensor,
    *,
    rot_slice: slice,
    columns: Sequence[str],
    omega_hat: Optional[torch.Tensor] = None,
    gate_val: float = 0.0,
    max_deg: float = 0.0,
    omega_detach: bool = True,
    reproject: bool = False,
) -> torch.Tensor:
    """
    Compose a RAW residual step through the shared geometry path.

    This is intentionally a thin orchestrator: SO(3) correction is delegated to
    `_apply_so3_correction_to_delta_raw`, and rot6d composition is delegated to
    `compose_rot6d_delta`. The canonical default keeps the current rollout
    behavior (`reproject=False`).
    """
    caller = "compose_delta_raw_to_next"
    cols = _coerce_required_columns(columns, caller=caller)
    _, _, rot_len = _require_bounded_rot_slice(rot_slice, caller=caller)
    if not torch.is_tensor(y_prev_raw) or not torch.is_tensor(delta_raw):
        raise TypeError(f"[{caller}] y_prev_raw and delta_raw must be tensors.")
    if y_prev_raw.shape != delta_raw.shape:
        raise ValueError(f"[{caller}] shape mismatch: y_prev_raw={y_prev_raw.shape}, delta_raw={delta_raw.shape}.")
    if y_prev_raw.dim() != 2:
        raise ValueError(f"[{caller}] expected 2D RAW tensors shaped (B,D), got {y_prev_raw.shape}.")
    if rot_slice.stop > y_prev_raw.shape[-1]:
        raise ValueError(f"[{caller}] rot_slice={rot_slice} exceeds RAW dim={y_prev_raw.shape[-1]}.")
    if rot_len <= 0:
        raise ValueError(f"[{caller}] rot_slice has no rot6d content.")

    delta_used = _apply_so3_correction_to_delta_raw(
        delta_raw,
        rot_slice=rot_slice,
        omega_hat=omega_hat,
        gate_val=float(gate_val or 0.0),
        max_deg=float(max_deg or 0.0),
        columns=cols,
        omega_detach=bool(omega_detach),
    )
    rot_next = compose_rot6d_delta(
        y_prev_raw[..., rot_slice],
        delta_used[..., rot_slice],
        columns=cols,
        reproject_result=bool(reproject),
    )
    y_next = y_prev_raw + delta_used
    y_next = y_next.clone()
    y_next[..., rot_slice] = rot_next
    return y_next


def so3_geodesic_blend(
    R_inc: torch.Tensor,
    R_dir: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """
    Blend two SO(3) pose tensors with canonical lambda shape (B,J).
    """
    caller = "so3_geodesic_blend"
    if not torch.is_tensor(R_inc) or not torch.is_tensor(R_dir) or not torch.is_tensor(lam):
        raise TypeError(f"[{caller}] R_inc, R_dir, and lam must be tensors.")
    if R_inc.shape != R_dir.shape:
        raise ValueError(f"[{caller}] R shape mismatch: R_inc={R_inc.shape}, R_dir={R_dir.shape}.")
    if R_inc.dim() != 4 or R_inc.shape[-2:] != (3, 3):
        raise ValueError(f"[{caller}] expected R tensors shaped (B,J,3,3), got {R_inc.shape}.")
    if lam.dim() != 2:
        raise ValueError(f"[{caller}] lam must have canonical shape (B,J), got {lam.shape}.")
    if lam.shape[0] != R_inc.shape[0] or lam.shape[1] != R_inc.shape[1]:
        raise ValueError(f"[{caller}] lam shape {lam.shape} is incompatible with R shape {R_inc.shape}.")

    lam = lam.to(device=R_inc.device, dtype=R_inc.dtype)
    R_res = torch.matmul(R_dir, R_inc.transpose(-1, -2))
    omega = so3_log_map(R_res)
    return torch.matmul(so3_exp_map(omega * lam.unsqueeze(-1)), R_inc)


def blend_rot6d_raw_with_lambda(
    y_inc_raw: torch.Tensor,
    direct_raw: torch.Tensor,
    lam: torch.Tensor,
    *,
    rot_slice: slice,
    columns: Sequence[str],
) -> torch.Tensor:
    """
    Replace the rot6d slice in RAW pose via SO(3) geodesic blending.

    `lam` is intentionally canonical-only: callers must pass shape (B,J).
    Shape normalization and denormalization stay in higher-level wrappers.
    """
    caller = "blend_rot6d_raw_with_lambda"
    cols = _coerce_required_columns(columns, caller=caller)
    _, _, rot_len = _require_bounded_rot_slice(rot_slice, caller=caller)
    if not torch.is_tensor(y_inc_raw) or not torch.is_tensor(direct_raw):
        raise TypeError(f"[{caller}] y_inc_raw and direct_raw must be tensors.")
    if y_inc_raw.shape != direct_raw.shape:
        raise ValueError(f"[{caller}] RAW shape mismatch: y_inc_raw={y_inc_raw.shape}, direct_raw={direct_raw.shape}.")
    if y_inc_raw.dim() != 2:
        raise ValueError(f"[{caller}] expected RAW tensors shaped (B,D), got {y_inc_raw.shape}.")
    if rot_slice.stop > y_inc_raw.shape[-1]:
        raise ValueError(f"[{caller}] rot_slice={rot_slice} exceeds RAW dim={y_inc_raw.shape[-1]}.")
    joint_count = rot_len // 6
    if lam.dim() != 2 or lam.shape[0] != y_inc_raw.shape[0] or lam.shape[1] != joint_count:
        raise ValueError(f"[{caller}] lam must have shape (B,J)=({y_inc_raw.shape[0]},{joint_count}), got {lam.shape}.")

    inc6 = reproject_rot6d(y_inc_raw[..., rot_slice]).view(y_inc_raw.shape[0], joint_count, 6)
    dir6 = reproject_rot6d(direct_raw[..., rot_slice]).view(y_inc_raw.shape[0], joint_count, 6)
    R_inc = rot6d_to_matrix(inc6, columns=cols)
    R_dir = rot6d_to_matrix(dir6, columns=cols)
    R_blend = so3_geodesic_blend(R_inc, R_dir, lam)
    rot_blend6 = matrix_to_rot6d(R_blend, columns=cols).reshape(y_inc_raw.shape[0], rot_len)
    y_blend = y_inc_raw.clone()
    y_blend[..., rot_slice] = rot_blend6
    return y_blend


def apply_direct_leg_so3_to_raw(
    direct_raw: torch.Tensor,
    *,
    rot_slice: slice,
    columns: Sequence[str],
    omega_leg: torch.Tensor,
    leg_idx: torch.Tensor,
    exclude_idx: Optional[int] = None,
    stopgrad_base: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply direct-leg SO(3) deltas to a RAW pose tensor.

    This pure helper performs only geometry: omega normalization, optional index
    exclusion, base rotation extraction, Exp(omega) application, and scatter back
    to the rot6d slice. Accumulators, supervision, denorm/norm, and Trainer state
    remain outside this contract.
    """
    caller = "apply_direct_leg_so3_to_raw"
    cols = _coerce_required_columns(columns, caller=caller)
    _, _, rot_len = _require_bounded_rot_slice(rot_slice, caller=caller)
    if not torch.is_tensor(direct_raw) or not torch.is_tensor(omega_leg) or not torch.is_tensor(leg_idx):
        raise TypeError(f"[{caller}] direct_raw, omega_leg, and leg_idx must be tensors.")
    if direct_raw.dim() != 2:
        raise ValueError(f"[{caller}] expected direct_raw shaped (B,D), got {direct_raw.shape}.")
    if rot_slice.stop > direct_raw.shape[-1]:
        raise ValueError(f"[{caller}] rot_slice={rot_slice} exceeds RAW dim={direct_raw.shape[-1]}.")
    if leg_idx.dim() != 1:
        raise ValueError(f"[{caller}] leg_idx must be a 1D tensor, got {leg_idx.shape}.")

    omega = omega_leg
    if omega.dim() == 4 and omega.size(1) == 1:
        omega = omega[:, 0]
    elif omega.dim() != 3:
        raise ValueError(f"[{caller}] omega_leg must have shape (B,K,3) or (B,1,K,3), got {omega_leg.shape}.")
    if omega.dim() != 3 or omega.shape[0] != direct_raw.shape[0] or omega.shape[-1] != 3:
        raise ValueError(f"[{caller}] omega_leg must align with direct_raw batch and have last dim 3, got {omega_leg.shape}.")
    if omega.shape[1] != leg_idx.numel():
        raise ValueError(f"[{caller}] omega K={omega.shape[1]} must match leg_idx length={leg_idx.numel()}.")

    joint_count = rot_len // 6
    idx_use = leg_idx.to(device=direct_raw.device, dtype=torch.long)
    if idx_use.numel() > 0 and (int(idx_use.min().item()) < 0 or int(idx_use.max().item()) >= joint_count):
        raise ValueError(f"[{caller}] leg_idx values must be within [0,{joint_count}).")
    omega = omega.to(device=direct_raw.device, dtype=direct_raw.dtype)

    keep_mask = None
    if exclude_idx is not None and idx_use.numel() > 0:
        exclude_val = int(exclude_idx)
        if bool((idx_use == exclude_val).any().detach().cpu().item()):
            keep_mask = idx_use != exclude_val
            idx_use = idx_use[keep_mask]
            omega = omega[:, keep_mask, :]

    base6 = reproject_rot6d(direct_raw[..., rot_slice]).view(direct_raw.shape[0], joint_count, 6)
    R_base = rot6d_to_matrix(base6, columns=cols)
    R_leg_base_used = R_base[:, idx_use, :, :]
    if bool(stopgrad_base):
        R_leg_base_used = R_leg_base_used.detach()
    if int(idx_use.numel()) <= 0:
        return direct_raw, R_leg_base_used, idx_use, keep_mask

    R_leg = torch.matmul(so3_exp_map(omega), R_leg_base_used)
    R_final = R_base.clone()
    R_final[:, idx_use, :, :] = R_leg
    rot6_final = matrix_to_rot6d(R_final, columns=cols).view(direct_raw.shape[0], rot_len)
    direct_raw_next = direct_raw.clone()
    direct_raw_next[..., rot_slice] = rot6_final
    return direct_raw_next, R_leg_base_used, idx_use, keep_mask


# ============================================
# 角速度计算 (Angular Velocity)
# ============================================

def angvel_vec_from_R_seq(R_seq: torch.Tensor, fps: float) -> torch.Tensor:
    """
    从旋转序列计算角速度向量

    Args:
        R_seq: (..., T, J, 3, 3) - 旋转矩阵序列
        fps: float - 采样帧率

    Returns:
        omega: (..., T-1, J, 3) - 角速度 (rad/s)
    """
    if R_seq.shape[-2:] != (3, 3):
        raise ValueError(f"[angvel_vec_from_R_seq] expects (..., 3, 3), got {tuple(R_seq.shape)}")
    if R_seq.dim() < 5:
        raise ValueError(f"[angvel_vec_from_R_seq] expects >=5D tensor (..., T, J, 3, 3), got {R_seq.dim()}D")
    if R_seq.shape[-4] < 2:
        raise ValueError(f"[angvel_vec_from_R_seq] expects T>=2, got T={R_seq.shape[-4]}")
    dR = torch.matmul(R_seq[..., 1:, :, :, :], R_seq[..., :-1, :, :, :].transpose(-1, -2))
    return angvel_vec_from_delta_R(dR, fps=float(fps))


def angvel_vec_from_delta_R(delta_R: torch.Tensor, fps: float) -> torch.Tensor:
    """
    从相邻帧旋转增量计算角速度向量。

    Args:
        delta_R: (..., T, J, 3, 3) 或 (..., J, 3, 3) 的相对旋转
        fps: float - 采样帧率

    Returns:
        omega: (..., T, J, 3) 或 (..., J, 3) - 标准 rotvec 角速度 (rad/s)
    """
    if delta_R.shape[-2:] != (3, 3):
        raise ValueError(f"[angvel_vec_from_delta_R] expects (..., 3, 3), got {tuple(delta_R.shape)}")
    phi = so3_log_map(delta_R)
    return phi * float(fps)


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


def parent_relative_matrices(R: torch.Tensor, parents: Optional[Sequence[int] | torch.Tensor]) -> torch.Tensor:
    """
    将世界旋转转换为 parent-relative 旋转。

    Args:
        R: (..., J, 3, 3) - 世界坐标系下的关节旋转
        parents: Sequence[int] 或 Tensor[J]，root 关节 parent 置为 -1

    Returns:
        (..., J, 3, 3) - 相对父关节旋转
    """
    if R.shape[-2:] != (3, 3):
        return R
    if parents is None:
        return R
    J = int(R.shape[-3])
    if J <= 0:
        return R
    if torch.is_tensor(parents):
        p = parents.to(device=R.device, dtype=torch.long).reshape(-1)
    else:
        try:
            p = torch.as_tensor(list(parents), device=R.device, dtype=torch.long).reshape(-1)
        except Exception:
            return R
    if int(p.numel()) < J:
        return R
    p = p[:J]
    safe_p = p.clamp(min=0, max=max(J - 1, 0))
    parent_R = R.index_select(-3, safe_p)
    rel = torch.matmul(parent_R.transpose(-1, -2), R)
    keep_world = ((p < 0) | (p >= J)).view(*((1,) * (R.dim() - 3)), J, 1, 1)
    return torch.where(keep_world, R, rel)


# 别名保持向后兼容
_root_relative_matrices = root_relative_matrices
_parent_relative_matrices = parent_relative_matrices


def wrap_to_pi_torch(x: torch.Tensor, *, neg_pi_to_pos_pi: bool = False) -> torch.Tensor:
    """
    将角度包裹到 [-π, π) 范围，语义对齐 wrap_to_pi_np。
    """
    dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    pi = torch.tensor(torch.pi, device=x.device, dtype=dtype)
    y = torch.remainder(x.to(dtype=dtype) + pi, 2.0 * pi) - pi
    if neg_pi_to_pos_pi:
        y = torch.where(y == -pi, pi, y)
    return y


def _root_yaw_from_matrix_torch(
    root_R: torch.Tensor,
    *,
    forward_axis: int,
    up_axis: int,
    offset: float = 0.0,
) -> Optional[torch.Tensor]:
    if not torch.is_tensor(root_R) or root_R.numel() == 0:
        return None
    if root_R.shape[-2:] != (3, 3):
        return None
    forward_axis = int(max(0, min(2, forward_axis)))
    up_axis = int(max(0, min(2, up_axis)))
    planar_axes = [ax for ax in (0, 1, 2) if ax != up_axis]
    if len(planar_axes) != 2:
        planar_axes = [0, 1]
    ax0, ax1 = planar_axes
    forward_vec = root_R[..., :, forward_axis]
    yaw = torch.atan2(forward_vec[..., ax1], forward_vec[..., ax0])
    if offset:
        yaw = yaw - float(offset)
    return wrap_to_pi_torch(yaw)


def root_yaw_from_rot6d_torch(
    rot6d: torch.Tensor,
    *,
    forward_axis: int,
    up_axis: int,
    offset: float = 0.0,
    root_idx: int = 0,
    reproject: bool = True,
) -> Optional[torch.Tensor]:
    if not torch.is_tensor(rot6d) or rot6d.numel() == 0:
        return None
    if rot6d.shape[-1] != 6:
        return None

    original_ndim = int(rot6d.ndim)
    if original_ndim == 1:
        rot6d_root = rot6d.view(1, 1, 6)
    elif original_ndim == 2:
        rot6d_root = rot6d.view(*rot6d.shape[:-1], 1, 6)
    else:
        joint_count = int(rot6d.shape[-2])
        if joint_count <= 0:
            return None
        root_idx = max(0, min(joint_count - 1, int(root_idx)))
        rot6d_root = rot6d[..., root_idx : root_idx + 1, :]

    try:
        rot6d_used = reproject_rot6d(rot6d_root) if reproject else rot6d_root
        root_R = rot6d_to_matrix(rot6d_used)[..., 0, :, :]
    except Exception:
        return None

    yaw = _root_yaw_from_matrix_torch(
        root_R,
        forward_axis=forward_axis,
        up_axis=up_axis,
        offset=offset,
    )
    if yaw is None:
        return None
    if original_ndim == 1:
        return yaw.reshape(())
    return yaw


def root_yaw_from_raw_rot6d(
    raw: Optional[torch.Tensor],
    *,
    rot_slice: slice,
    root_idx: int,
    up_axis: int,
    forward_axis: int,
    offset: float = 0.0,
    reproject: bool = True,
) -> Optional[torch.Tensor]:
    if not torch.is_tensor(raw) or raw.numel() == 0:
        return None
    if raw.ndim != 2:
        return None
    if not isinstance(rot_slice, slice):
        return None

    start = rot_slice.start
    stop = rot_slice.stop
    step = rot_slice.step
    if not isinstance(start, int) or not isinstance(stop, int):
        return None
    if step not in (None, 1):
        return None
    if start < 0 or stop <= start:
        return None
    if int(raw.shape[-1]) < stop:
        return None

    try:
        rot_flat = raw[..., rot_slice]
    except Exception:
        return None
    if not torch.is_tensor(rot_flat) or rot_flat.numel() == 0:
        return None

    rot_len = int(rot_flat.shape[-1])
    if rot_len != (stop - start) or rot_len % 6 != 0:
        return None

    joint_count = rot_len // 6
    if joint_count <= 0:
        return None

    try:
        root_idx = int(root_idx)
        up_axis = int(up_axis)
        forward_axis = int(forward_axis)
        offset = float(offset)
    except Exception:
        return None
    if root_idx < 0 or root_idx >= joint_count:
        return None

    try:
        rot6d = rot_flat.reshape(int(raw.shape[0]), joint_count, 6)
    except Exception:
        return None

    return root_yaw_from_rot6d_torch(
        rot6d,
        forward_axis=forward_axis,
        up_axis=up_axis,
        offset=offset,
        root_idx=root_idx,
        reproject=bool(reproject),
    )


def reproject_cond_to_local_frame(
    cond_raw: Optional[torch.Tensor],
    yaw_gt: torch.Tensor,
    yaw_pred: torch.Tensor,
) -> Optional[torch.Tensor]:
    if cond_raw is None:
        return None
    if not torch.is_tensor(cond_raw):
        raise TypeError("cond_raw must be a torch.Tensor")

    cond_dim = int(cond_raw.shape[-1])
    if cond_dim < 3:
        raise ValueError("cond_raw must contain [..., dir_x, dir_y, speed]")

    if yaw_gt.dim() > 1:
        yaw_gt = yaw_gt.squeeze(-1)
    if yaw_pred.dim() > 1:
        yaw_pred = yaw_pred.squeeze(-1)

    delta_yaw = wrap_to_pi_torch(yaw_pred - yaw_gt)
    action_dim = cond_dim - 3
    cond_reprojected = cond_raw.clone()
    dir_world = cond_raw[..., action_dim : action_dim + 2]

    cos_delta = torch.cos(-delta_yaw)
    sin_delta = torch.sin(-delta_yaw)
    dir_local_x = dir_world[..., 0] * cos_delta - dir_world[..., 1] * sin_delta
    dir_local_y = dir_world[..., 0] * sin_delta + dir_world[..., 1] * cos_delta

    cond_reprojected[..., action_dim] = dir_local_x
    cond_reprojected[..., action_dim + 1] = dir_local_y
    return cond_reprojected


def root_yaw_from_rot6d_np(
    rot6d: np.ndarray,
    *,
    forward_axis: int,
    up_axis: int,
    offset: float = 0.0,
    root_idx: int = 0,
) -> Optional[np.ndarray]:
    if not isinstance(rot6d, np.ndarray) or rot6d.size == 0:
        return None
    if rot6d.shape[-1] != 6:
        return None
    try:
        rot6d_t = torch.from_numpy(rot6d.astype(np.float32, copy=False))
    except Exception:
        return None
    with torch.no_grad():
        yaw = root_yaw_from_rot6d_torch(
            rot6d_t,
            forward_axis=forward_axis,
            up_axis=up_axis,
            offset=offset,
            root_idx=root_idx,
            reproject=True,
        )
    if yaw is None:
        return None
    try:
        return yaw.detach().cpu().numpy()
    except Exception:
        return None


# ============================================
# Numpy 版本工具 (用于数据预处理)
# ============================================

def wrap_to_pi_np(x: np.ndarray, *, neg_pi_to_pos_pi: bool = False) -> np.ndarray:
    """
    将角度包裹到 [-π, π) 范围

    Args:
        x: np.ndarray - 角度 (弧度)
        neg_pi_to_pos_pi: 是否将边界点 -π 显式映射到 +π

    Returns:
        np.ndarray - 包裹后的角度
    """
    y = (x + np.pi) % (2.0 * np.pi) - np.pi
    if neg_pi_to_pos_pi:
        y = np.where(y == -np.pi, np.pi, y)
    return y


def wrap_to_pi_np_negpi_to_pospi(x: np.ndarray) -> np.ndarray:
    """
    显式边界特化：[-π, π) 基础上，将 -π 映射为 +π。
    """
    return wrap_to_pi_np(x, neg_pi_to_pos_pi=True)


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


def rot6d_to_matrix_np(
    xJ6: np.ndarray,
    *,
    columns: Tuple[str, str] = ("X", "Z"),
    eps: float = 1e-8,
    canonical_axis_order: bool = True,
) -> np.ndarray:
    """
    Numpy 版本 6D -> 旋转矩阵。

    Args:
        xJ6: (..., 6) - 6D 旋转表示（提供两列轴向量）
        columns: 两列对应的轴名，例如 ("X","Z")
        eps: 归一化稳定项
        canonical_axis_order:
            - True: 输出列顺序固定为 [X, Y, Z]（与 torch 版对齐）。
            - False: 输出列顺序为 [first, second, first×second]（兼容旧 convert 实现）。

    Returns:
        (..., 3, 3) - 旋转矩阵
    """
    if xJ6.shape[-1] != 6:
        raise ValueError(f"rot6d_to_matrix_np expects (..., 6); got {xJ6.shape}")
    if len(columns) != 2 or columns[0] == columns[1]:
        raise ValueError(f"rot6d_to_matrix_np expects two distinct axis names; got {columns}")

    axis_idx = {"X": 0, "Y": 1, "Z": 2}
    if columns[0] not in axis_idx or columns[1] not in axis_idx:
        raise ValueError(f"rot6d_to_matrix_np unsupported columns {columns}")

    a = np.array(xJ6[..., 0:3], copy=True)
    b = np.array(xJ6[..., 3:6], copy=True)

    def _norm(v: np.ndarray) -> np.ndarray:
        return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), eps, None)

    a = _norm(a)
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b = _norm(b)

    if not canonical_axis_order:
        c = _norm(np.cross(a, b, axis=-1))
        return np.stack([a, b, c], axis=-1)

    ax1, ax2 = columns
    r = {ax1: a, ax2: b}
    remaining = [ax for ax in ("X", "Y", "Z") if ax not in (ax1, ax2)][0]
    r[remaining] = _norm(np.cross(r[ax2], r[ax1], axis=-1))
    R = np.stack([r["X"], r["Y"], r["Z"]], axis=-1)

    # 与 torch 版对齐：若 det<0，仅翻转派生列
    det = np.linalg.det(R.reshape(-1, 3, 3)).reshape(R.shape[:-2])
    neg = det < 0.0
    if np.any(neg):
        R = R.copy()
        col_idx = axis_idx[remaining]
        R[..., :, col_idx] = np.where(neg[..., None], -R[..., :, col_idx], R[..., :, col_idx])
    return R


def so3_angle_from_matrix_np(R: np.ndarray, *, degrees: bool = False) -> np.ndarray:
    """
    Numpy 版本：从旋转矩阵计算旋转角（相对单位旋转的 geodesic angle）。

    Args:
        R: (..., 3, 3) - 旋转矩阵
        degrees: 是否返回角度制

    Returns:
        (...) - 旋转角
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"so3_angle_from_matrix_np expects (..., 3, 3); got {R.shape}")
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    ang = np.arccos(cos)
    if degrees:
        ang = np.degrees(ang)
    return ang


def rot6d_to_angle_deg_np(
    xJ6: np.ndarray,
    *,
    columns: Tuple[str, str] = ("X", "Z"),
    eps: float = 1e-8,
    canonical_axis_order: bool = True,
) -> np.ndarray:
    """
    Numpy 版本：6D 旋转表示 -> 旋转角（度）。

    Args:
        xJ6: (..., 6) - 6D 旋转表示
        columns: 两列对应的轴名
        eps: 归一化稳定项
        canonical_axis_order: 参见 `rot6d_to_matrix_np`

    Returns:
        (...) - 角度制旋转角
    """
    R = rot6d_to_matrix_np(
        xJ6,
        columns=columns,
        eps=eps,
        canonical_axis_order=canonical_axis_order,
    )
    return so3_angle_from_matrix_np(R, degrees=True)


# ============================================
# Forward kinematics (positions)
# ============================================

def fk_positions_from_rot6d(
    rot6d: torch.Tensor,
    parents: Sequence[int] | torch.Tensor,
    offsets: torch.Tensor,
    root_pos: Optional[torch.Tensor] = None,
    *,
    columns: Tuple[str, str] = ("X", "Z"),
) -> torch.Tensor:
    """
    Compute global joint positions from local joint rotations (6D) and fixed local offsets.

    Assumptions:
        - `parents[j]` gives the parent joint index of joint j (use -1 for root).
        - `offsets[j]` is the child-in-parent local offset (meters).
        - Root joint position is `root_pos + offsets[root]` (if root_pos is None, uses zeros).

    Args:
        rot6d: (..., J, 6) local rotations.
        parents: (J,) list/tensor of parent indices.
        offsets: (J, 3) local offsets in parent frame.
        root_pos: (..., 3) root translation (same leading dims as rot6d without the joint dim).
        columns: 6D columns convention for `rot6d_to_matrix`.

    Returns:
        pos: (..., J, 3) global joint positions.
    """
    if rot6d.shape[-1] != 6:
        raise ValueError(f"fk_positions_from_rot6d expects (..., J, 6); got rot6d.shape={tuple(rot6d.shape)}")
    if offsets.dim() != 2 or offsets.shape[-1] != 3:
        raise ValueError(f"offsets must be (J, 3); got offsets.shape={tuple(offsets.shape)}")
    J = int(rot6d.shape[-2])
    if J <= 0:
        raise ValueError("fk_positions_from_rot6d: J must be > 0")
    if offsets.shape[0] < J:
        raise ValueError(f"offsets has {offsets.shape[0]} joints, but rot6d needs {J}")

    device = rot6d.device
    dtype = rot6d.dtype
    offsets_t = offsets[:J].to(device=device, dtype=dtype)

    if torch.is_tensor(parents):
        parents_list = [int(x) for x in parents.reshape(-1).tolist()[:J]]
    else:
        parents_list = [int(x) for x in list(parents)[:J]]
    if len(parents_list) < J:
        raise ValueError(f"parents has {len(parents_list)} entries, but rot6d needs {J}")

    lead_shape = rot6d.shape[:-2]
    flat = rot6d.reshape(-1, J, 6)
    R_local = rot6d_to_matrix(flat, columns=columns)  # (N, J, 3, 3)

    if root_pos is None:
        root_flat = rot6d.new_zeros((flat.shape[0], 3))
    else:
        if root_pos.shape[-1] != 3:
            raise ValueError(f"root_pos must be (..., 3); got root_pos.shape={tuple(root_pos.shape)}")
        root_flat = root_pos.to(device=device, dtype=dtype).reshape(-1, 3)
        if root_flat.shape[0] != flat.shape[0]:
            raise ValueError(
                f"root_pos leading dims {tuple(root_pos.shape[:-1])} incompatible with rot6d leading dims {lead_shape}"
            )

    gR = rot6d.new_empty((flat.shape[0], J, 3, 3))
    gP = rot6d.new_empty((flat.shape[0], J, 3))
    for j in range(J):
        p = parents_list[j]
        if p < 0 or p >= J:
            gR[:, j] = R_local[:, j]
            gP[:, j] = root_flat + offsets_t[j]
        else:
            gR[:, j] = gR[:, p] @ R_local[:, j]
            gP[:, j] = gP[:, p] + (gR[:, p] @ offsets_t[j].view(1, 3, 1)).squeeze(-1)

    return gP.reshape(*lead_shape, J, 3)
