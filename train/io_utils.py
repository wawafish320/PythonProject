"""
数据加载与格式转换工具

处理 JSON、NPZ 等格式，提供统一的数据加载接口。
包含运动数据的预处理和分析工具。

从 training_MPL.py 重构而来。
"""
import json
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path


# ============================================
# JSON 加载
# ============================================

def load_soft_contacts_from_json(json_path: str) -> np.ndarray:
    """
    从 JSON 文件加载软接触标注 (Soft Contact Scores)

    Args:
        json_path: JSON 文件路径

    Returns:
        np.ndarray: [T, 2] - 每帧左右脚的软接触分数

    Raises:
        ValueError: 如果 JSON 格式不正确

    Example JSON 格式:
        {
            "Frames": [
                {"FootEvidence": {"L": {"soft_contact_score": 0.8}, "R": {"soft_contact_score": 0.2}}},
                ...
            ]
        }
    """
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


def npz_scalar_to_str(v) -> str:
    """Unwrap numpy scalar/bytes to Python str for paths stored in npz."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    if not isinstance(v, str):
        raise ValueError(f"Expected string, got {type(v)}: {repr(v)}")
    return v


# ============================================
# 角度与方向处理
# ============================================

from train.geometry import wrap_to_pi_np  # noqa: E402


def direction_yaw_from_array(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    从方向向量数组中提取 yaw 角度

    优先从最后三维提取: [dir_x, dir_y, speed]
    如果不可用则从最后两维提取: [dir_x, dir_y]

    Args:
        arr: [T, D] - 可能包含方向信息的数组

    Returns:
        yaw 角度数组 (弧度) 或 None (如果无法提取)

    Example:
        >>> arr = np.array([[1, 0, 0.5], [0, 1, 0.8]])  # [dir_x, dir_y, speed]
        >>> direction_yaw_from_array(arr)
        array([0., 1.5707963])  # yaw 角度
    """
    if arr is None:
        return None
    try:
        a = np.asarray(arr)
    except Exception:
        return None

    if a.ndim != 2 or a.shape[1] < 2:
        return None

    # Prefer 最后三维: [dir_x, dir_y, speed]
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

    # 如果有速度信息，过滤低速帧
    if speed is not None:
        try:
            mask = np.asarray(speed) > 1e-3
            if mask.shape == yaw.shape and mask.any():
                yaw = yaw[mask]
        except Exception:
            pass

    return yaw if yaw.size > 0 else None


def velocity_yaw_from_array(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    从速度向量数组中提取 yaw 角度

    Args:
        arr: [T, D] - 包含 XY 速度的数组 (至少前两列是速度)

    Returns:
        yaw 角度数组 (弧度) 或 None (如果无法提取)

    Example:
        >>> arr = np.array([[0.5, 0], [0, 0.8]])  # [vel_x, vel_y, ...]
        >>> velocity_yaw_from_array(arr)
        array([0., 1.5707963])  # yaw 角度
    """
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


# ============================================
# 状态布局工具
# ============================================

def speed_from_X_layout(X: np.ndarray, state_layout: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    从状态向量中提取速度标量 (用于数据过滤)

    Args:
        X: [T, D] - 状态向量序列
        state_layout: 状态布局字典，需包含 'RootVelocity' 键

    Returns:
        [T] - 速度标量数组 (速度的 L2 范数)
        None - 如果提取失败

    支持的布局格式:
        - {'RootVelocity': {'start': 3, 'size': 2}}
        - {'RootVelocity': (3, 5)}   # [start, end)
        - {'RootVelocity': [3, 2]}   # [start, size]

    Example:
        >>> X = np.random.randn(100, 20)
        >>> layout = {'RootVelocity': {'start': 3, 'size': 2}}
        >>> speed = speed_from_X_layout(X, layout)
        >>> speed.shape
        (100,)
    """
    try:
        if not isinstance(state_layout, dict):
            return None

        # 优先使用 parse_layout_entry 工具
        try:
            from .layout_utils import parse_layout_entry
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
        # 计算速度的 L2 范数
        spd = np.linalg.norm(v[:, :2], axis=1) if v.shape[1] >= 2 else np.abs(v[:, 0])
        return spd.astype(np.float32)

    except Exception:
        # 紧急兜底：过滤失效，但不会中断训练
        return None


# ============================================
# 别名保持向后兼容
# ============================================

_load_soft_contacts_from_json = load_soft_contacts_from_json
_wrap_to_pi_np = wrap_to_pi_np
_direction_yaw_from_array = direction_yaw_from_array
_velocity_yaw_from_array = velocity_yaw_from_array
_speed_from_X_layout = speed_from_X_layout
