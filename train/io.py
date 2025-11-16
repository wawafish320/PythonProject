from __future__ import annotations

"""
Unified IO helpers for loading metadata, derived arrays, and dataset-specific signals.

This module merges the legacy io_utils.py and layout_io.py helpers so callers
have a single import surface: `from train.io import ...`.
"""

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .layout import normalize_layout, canonicalize_state_layout

__all__ = [
    "load_soft_contacts_from_json",
    "direction_yaw_from_array",
    "velocity_yaw_from_array",
    "speed_from_X_layout",
    "npz_scalar_to_str",
    "load_layouts_from_meta",
]


def load_soft_contacts_from_json(json_path: str) -> np.ndarray:
    """
    从 JSON 文件加载软接触标注 (Soft Contact Scores)
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


def direction_yaw_from_array(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    从方向向量数组中提取 yaw 角度
    """
    if arr is None:
        return None
    try:
        a = np.asarray(arr)
    except Exception:
        return None

    if a.ndim != 2 or a.shape[1] < 2:
        return None

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


def velocity_yaw_from_array(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    从速度向量数组中提取 yaw 角度
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


def speed_from_X_layout(X: np.ndarray, state_layout: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    从状态向量中提取速度标量 (用于数据过滤)
    """
    try:
        if not isinstance(state_layout, dict):
            return None

        try:
            from .layout import parse_layout_entry

            sl = parse_layout_entry(state_layout.get("RootVelocity"), "RootVelocity")
        except Exception:
            sl = None

        a = b = None
        if isinstance(sl, slice):
            a = int(sl.start or 0)
            b = int(sl.stop)
        else:
            rv = state_layout.get("RootVelocity")
            if rv is None:
                return None

            if isinstance(rv, dict):
                a = int(rv.get("start", 0))
                sz = int(rv.get("size", 0))
                b = a + sz
            else:
                if hasattr(rv, "__iter__"):
                    r0 = int(rv[0])
                    r1 = int(rv[1])
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
        return None


def load_layouts_from_meta(meta_json) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
    """
    Extract and normalize state/output layouts from meta json object.
    Falls back to JSON strings if meta_json contains serialized layouts.
    """
    if meta_json is None:
        raise ValueError("meta_json is required to load layouts")

    if hasattr(meta_json, "item"):
        try:
            meta_json = meta_json.item()
        except Exception:
            pass

    if isinstance(meta_json, (bytes, bytearray)):
        try:
            meta_json = meta_json.decode("utf-8")
        except Exception:
            pass

    if isinstance(meta_json, str):
        meta = json.loads(meta_json)
    elif isinstance(meta_json, dict):
        meta = meta_json
    else:
        raise ValueError(f"Unsupported meta_json type: {type(meta_json)}")

    state_raw = meta.get("state_layout") or meta.get("StateLayout")
    out_raw = meta.get("output_layout") or meta.get("OutputLayout")

    if state_raw is None or out_raw is None:
        raise ValueError("meta_json missing state_layout or output_layout")

    state_raw = canonicalize_state_layout(state_raw)
    out_raw = canonicalize_state_layout(out_raw)

    Dx = int(meta.get("Dx", 0)) if meta.get("Dx") is not None else None
    Dy = int(meta.get("Dy", 0)) if meta.get("Dy") is not None else None
    state_layout = (
        normalize_layout(state_raw, Dx) if Dx is not None else normalize_layout(state_raw, sum(v[1] for v in state_raw.values()))
    )
    out_layout = (
        normalize_layout(out_raw, Dy) if Dy is not None else normalize_layout(out_raw, sum(v[1] for v in out_raw.values()))
    )
    return state_layout, out_layout
