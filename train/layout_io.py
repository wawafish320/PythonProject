"""Layout loading helpers shared across dataset and other tools."""

from __future__ import annotations

import json
from typing import Dict, Any, Tuple

from .layout_utils import normalize_layout, canonicalize_state_layout


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

    # canonicalize then normalize
    state_raw = canonicalize_state_layout(state_raw)
    out_raw = canonicalize_state_layout(out_raw)

    Dx = int(meta.get("Dx", 0)) if meta.get("Dx") is not None else None
    Dy = int(meta.get("Dy", 0)) if meta.get("Dy") is not None else None
    state_layout = normalize_layout(state_raw, Dx) if Dx is not None else normalize_layout(state_raw, sum(v[1] for v in state_raw.values()))
    out_layout = normalize_layout(out_raw, Dy) if Dy is not None else normalize_layout(out_raw, sum(v[1] for v in out_raw.values()))
    return state_layout, out_layout

