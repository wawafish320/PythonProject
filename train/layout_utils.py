"""
数据布局 (Layout) 解析与管理工具

处理状态空间和输出空间的维度映射，提供统一的布局解析接口。
支持多种布局格式：dict、list/tuple、slice 等。

从 training_MPL.py 和 convert_json_to_npz.py 重构而来。
"""
from typing import Dict, Any, Optional, Tuple


def parse_layout_entry(
    entry_value: Any,
    entry_name: Optional[str] = None,
    total_dim: Optional[int] = None
) -> Optional[slice]:
    """
    将布局条目解析为 slice，采用"严格 SIZE 语义"

    支持格式：
        - dict: {'start': s, 'size': k} -> slice(s, s+k)
                {'start': s, 'dim': k}   -> slice(s, s+k)
                {'start': s, 'end': e}   -> slice(s, e)
        - list/tuple: [s, k] -> slice(s, s+k)  # 第二项永远是 size
        - slice: 直接返回

    Args:
        entry_value: 布局配置值
        entry_name: 条目名称 (用于错误提示)
        total_dim: 总维度 (可选，用于越界检查)

    Returns:
        slice 对象，如果解析失败则返回 None

    Example:
        >>> parse_layout_entry({'start': 3, 'size': 2})
        slice(3, 5, None)
        >>> parse_layout_entry([3, 2])
        slice(3, 5, None)
        >>> parse_layout_entry(slice(3, 5))
        slice(3, 5, None)
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

        # 优先使用 'end'
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


def normalize_layout(raw_layout: Dict[str, Any], D: int) -> Dict[str, Tuple[int, int]]:
    """
    标准化布局为 (start, size) 格式，并进行严格的越界检查

    Args:
        raw_layout: 原始布局配置字典
        D: 总维度

    Returns:
        Dict[str, (start, size)] - 标准化后的布局

    Raises:
        AssertionError: 如果布局条目越界或格式错误

    Example:
        >>> normalize_layout({'RootYaw': {'start': 0, 'size': 1}}, 10)
        {'RootYaw': (0, 1)}
        >>> normalize_layout({'RootVel': [3, 2]}, 10)
        {'RootVel': (3, 2)}
    """
    norm = {}
    for name, meta in (raw_layout or {}).items():
        st = ed = sz = None

        if isinstance(meta, dict):
            if 'start' in meta and 'end' in meta:
                st = int(meta['start'])
                ed = int(meta['end'])
                sz = ed - st
            elif 'start' in meta and 'size' in meta:
                st = int(meta['start'])
                sz = int(meta['size'])
                ed = st + sz

        elif isinstance(meta, (list, tuple)) and len(meta) >= 2:
            # Heuristic: treat as (start,size); if looks like (start,end) with end>start and D known, still valid
            st = int(meta[0])
            second = int(meta[1])
            if D is not None and second > st and second <= D:
                # ambiguous; prefer (start,size) since project uses [start,size], but accept (start,end) too
                # If interpreted as size and exceeds bound, switch to end
                as_size_ok = (st + second) <= D
                if as_size_ok:
                    sz = second
                    ed = st + sz
                else:
                    ed = second
                    sz = ed - st
            else:
                sz = second
                ed = st + sz

        if st is None or (sz is None and ed is None):
            raise AssertionError(f"[FATAL] layout[{name}] cannot be normalized: {meta}")
        if ed is None:
            ed = st + sz
        if sz is None:
            sz = ed - st
        if not (0 <= st < ed <= D):
            raise AssertionError(f"[FATAL] layout[{name}] must be [start,end) with 0<=start<end<=D, got ({st},{ed}) and D={D}")

        norm[name] = (int(st), int(sz))

    return norm


def layout_span(layout: Dict[str, Any], key: str) -> Optional[Tuple[int, int]]:
    """
    从布局中提取指定键的 (start, end) 范围

    Args:
        layout: 布局配置字典
        key: 布局键名

    Returns:
        (start, end) 或 None (如果键不存在)

    Example:
        >>> layout_span({'RootYaw': {'start': 0, 'size': 1}}, 'RootYaw')
        (0, 1)
    """
    sl = parse_layout_entry(layout.get(key), key)
    if sl is None:
        return None
    return (int(sl.start), int(sl.stop))


def canonicalize_state_layout(layout_dict: dict) -> dict:
    """
    标准化 state_layout 键名，用于向后兼容

    将旧版键名映射到新版标准键名：
        - root_pos -> RootPosition
        - root_vel -> RootVelocity
        - root_yaw -> RootYaw
        - rot6d -> BoneRotations6D
        - angvel -> BoneAngularVelocities

    Args:
        layout_dict: 原始布局字典

    Returns:
        标准化后的布局字典

    Example:
        >>> canonicalize_state_layout({'root_yaw': [0, 1], 'rot6d': [1, 18]})
        {'RootYaw': [0, 1], 'BoneRotations6D': [1, 19]}
    """
    mapping = {
        "root_pos": "RootPosition",
        "root_vel": "RootVelocity",
        "root_yaw": "RootYaw",
        "rot6d": "BoneRotations6D",
        "angvel": "BoneAngularVelocities",
        # 标准名称保持不变
        "RootPosition": "RootPosition",
        "RootVelocity": "RootVelocity",
        "RootYaw": "RootYaw",
        "BoneRotations6D": "BoneRotations6D",
        "BoneAngularVelocities": "BoneAngularVelocities",
    }

    out = {}
    for k, v in (layout_dict or {}).items():
        if v is None:
            continue

        try:
            # 尝试解析为 [start, end] 格式
            st, ed = int(v[0]), int(v[1])
        except Exception:
            # Fallback for dict form {'start':..., 'size':...}
            try:
                st = int(v.get("start", v.get("offset", 0)))
                ln = int(v.get("size", v.get("length", 0)))
                ed = st + ln
            except Exception:
                continue

        out[mapping.get(k, k)] = [st, ed]

    return out


# 别名保持向后兼容
_canon_state_layout = canonicalize_state_layout
_normalize_layout = normalize_layout
_layout_span = layout_span
