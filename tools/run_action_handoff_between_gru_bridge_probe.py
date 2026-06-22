#!/usr/bin/env python3
"""Debug-only GRU bridge probe for context/goal/contact conditioning.

This tool trains small from-scratch CPU/GPU probes over existing action-handoff
windows. It does not import production trainers, mutate checkpoints, or edit
production model code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    CONTEXT_LEN_C,
    EGO_VEL_SLICE,
    FPS,
    LOCKED_CLIPS,
    POSE_DIM,
    POSE_SLICE,
    RAW_COND_DIR_SLICE,
    STATE_DIM,
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
    load_clip_states,
)


ANGVEL_DIM = 138
DEFAULT_HORIZON = 16
MATCHED_TARGETS = ("Walk_L_To_L", "Walk_R_To_L", "Walk_R_To_R")
UNMATCHED_TARGET = "Walk_L_To_R"
DEFAULT_NPZ_ROOT = Path("raw_data/processed_data")
DEFAULT_Z_FEATURES = Path("debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz")
DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_between_gru_bridge_probe_20260606")
EPS = 1.0e-8

ROOT_ROT6D_SLICE = slice(POSE_SLICE.start, POSE_SLICE.start + 6)
NONROOT_ROT6D_SLICE = slice(POSE_SLICE.start + 6, POSE_SLICE.stop)
BONE0_ANGVEL_SLICE = slice(0, 3)
NONROOT_ANGVEL_SLICE = slice(3, ANGVEL_DIM)
ROOT_MOTION_SLICE = slice(EGO_VEL_SLICE.start, YAW_RATE_SLICE.stop)
CONTACT_LABEL_THRESHOLD = 0.5
CONTACT_SUPPORT_LABELS = ("flight_or_unknown", "right", "left", "dual")
GOAL_CONTACT_MODES = ("none", "target_support_end")


@dataclass(frozen=True)
class ClipData:
    name: str
    state281: np.ndarray
    root_pos: np.ndarray
    root_vel: np.ndarray
    bone_angvel: np.ndarray
    cond_dir: np.ndarray
    contact01: np.ndarray


@dataclass(frozen=True)
class BridgeItem:
    clip: str
    start: int
    end: int
    ctx_state: np.ndarray
    ctx_aux: np.ndarray
    seq: Mapping[str, np.ndarray]
    goal_lowdim: np.ndarray


@dataclass(frozen=True)
class SplitDef:
    name: str
    kind: str
    train_idx: Tuple[int, ...]
    test_idx: Tuple[int, ...]
    note: str


@dataclass(frozen=True)
class VariantSpec:
    name: str
    use_goal: bool
    contact_mode: str
    role: str
    runtime_status: str


@dataclass(frozen=True)
class ContactLossConfig:
    contact_step_weight: float = 0.50
    contact_predict_mse_weight: float = 2.0
    contact_predict_bce_weight: float = 0.25
    contact_state_weight: float = 0.50
    contact_endpoint_support_weight: float = 0.0
    contact_support_threshold01: Tuple[float, float] = (CONTACT_LABEL_THRESHOLD, CONTACT_LABEL_THRESHOLD)


DEFAULT_CONTACT_LOSS_CONFIG = ContactLossConfig()


VARIANT_SPECS: Tuple[VariantSpec, ...] = (
    VariantSpec(
        name="ctx_only",
        use_goal=False,
        contact_mode="none",
        role="context-only learned control; checks whether target intent is ignored",
        runtime_status="runtime-safe negative control",
    ),
    VariantSpec(
        name="no_contact",
        use_goal=True,
        contact_mode="none",
        role="context plus goal without contact cycle conditioning",
        runtime_status="runtime-safe learned baseline",
    ),
    VariantSpec(
        name="predicted_contact",
        use_goal=True,
        contact_mode="predicted",
        role="main learned contact-plan bridge",
        runtime_status="runtime candidate: contact plan generated from context plus goal intent",
    ),
    VariantSpec(
        name="oracle_contact_upper_bound",
        use_goal=True,
        contact_mode="oracle",
        role="upper bound with target soft contact injected",
        runtime_status="oracle upper-bound only; not a deployable input contract",
    ),
    VariantSpec(
        name="shifted_or_random_contact_control",
        use_goal=True,
        contact_mode="negative",
        role="wrong-cycle contact control; should degrade if contact is used",
        runtime_status="debug negative control",
    ),
)


def _fmt(v: Any, digits: int = 6) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "null"
    if not math.isfinite(x):
        return "null"
    return f"{x:.{digits}f}"


def _finite_float32(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)


def _mse_np(a: np.ndarray, b: np.ndarray) -> float:
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    if da.size == 0 or db.size == 0:
        return 0.0
    d = da - db
    return float(np.mean(d * d))


def _contact_threshold01_from_raw(scaler: ContactScaler, raw_threshold: float) -> Tuple[float, float]:
    raw = np.full((2,), float(raw_threshold), dtype=np.float32)
    denom = np.maximum(np.asarray(scaler.max_v, dtype=np.float32) - np.asarray(scaler.min_v, dtype=np.float32), 1.0e-6)
    threshold = np.clip((raw - np.asarray(scaler.min_v, dtype=np.float32)) / denom, 0.0, 1.0)
    return (float(threshold[0]), float(threshold[1]))


def _support_class_np(contact_frame: np.ndarray, threshold01: Tuple[float, float]) -> int:
    c = np.asarray(contact_frame, dtype=np.float32).reshape(-1)
    thr = np.asarray(threshold01, dtype=np.float32).reshape(2)
    right = bool(c.size > 0 and c[0] > float(thr[0]))
    left = bool(c.size > 1 and c[1] > float(thr[1]))
    return int(right) + 2 * int(left)


def _support_label_from_class(cls: int) -> str:
    idx = max(0, min(int(cls), len(CONTACT_SUPPORT_LABELS) - 1))
    return CONTACT_SUPPORT_LABELS[idx]


def _support_onehot_np(contact_frame: np.ndarray, threshold01: Tuple[float, float]) -> np.ndarray:
    onehot = np.zeros((len(CONTACT_SUPPORT_LABELS),), dtype=np.float32)
    onehot[_support_class_np(contact_frame, threshold01)] = 1.0
    return onehot


def _threshold_t(contact: torch.Tensor, threshold01: Tuple[float, float]) -> torch.Tensor:
    return torch.as_tensor(threshold01, dtype=contact.dtype, device=contact.device).reshape(1, 1, 2)


def _endpoint_support_classes_t(contact: torch.Tensor, threshold01: Tuple[float, float]) -> torch.Tensor:
    if contact.ndim != 3 or int(contact.shape[-1]) != 2:
        raise ValueError(f"expected contact [B,H,2], got shape={tuple(contact.shape)}")
    endpoint = torch.stack([contact[:, 0], contact[:, -1]], dim=1)
    bits = (endpoint > _threshold_t(contact, threshold01)).to(torch.long)
    return bits[:, :, 0] + 2 * bits[:, :, 1]


def _endpoint_support_logits_t(
    contact: torch.Tensor,
    threshold01: Tuple[float, float],
    *,
    temperature: float = 0.08,
) -> torch.Tensor:
    if contact.ndim != 3 or int(contact.shape[-1]) != 2:
        raise ValueError(f"expected contact [B,H,2], got shape={tuple(contact.shape)}")
    endpoint = torch.stack([contact[:, 0], contact[:, -1]], dim=1)
    threshold = _threshold_t(contact, threshold01)
    right = (endpoint[..., 0] - threshold[..., 0]) / max(float(temperature), 1.0e-4)
    left = (endpoint[..., 1] - threshold[..., 1]) / max(float(temperature), 1.0e-4)
    return torch.stack(
        [
            -right - left,
            right - left,
            -right + left,
            right + left,
        ],
        dim=-1,
    )


def _endpoint_support_loss_t(
    pred_contact: Optional[torch.Tensor],
    true_contact: torch.Tensor,
    threshold01: Tuple[float, float],
) -> torch.Tensor:
    if pred_contact is None:
        return true_contact.new_tensor(0.0)
    logits = _endpoint_support_logits_t(pred_contact, threshold01)
    target = _endpoint_support_classes_t(true_contact, threshold01)
    return F.cross_entropy(logits.reshape(-1, len(CONTACT_SUPPORT_LABELS)), target.reshape(-1))


def _endpoint_support_metrics_np(
    pred_contact: np.ndarray,
    true_contact: np.ndarray,
    threshold01: Tuple[float, float],
) -> Dict[str, Any]:
    pred = np.asarray(pred_contact, dtype=np.float32).reshape(-1, 2)
    true = np.asarray(true_contact, dtype=np.float32).reshape(-1, 2)
    if pred.shape[0] == 0 or true.shape[0] == 0:
        return {
            "endpoint_support_start_match": False,
            "endpoint_support_end_match": False,
            "endpoint_support_both_match": False,
            "endpoint_support_pred_start": "empty",
            "endpoint_support_true_start": "empty",
            "endpoint_support_pred_end": "empty",
            "endpoint_support_true_end": "empty",
        }
    pred_start = _support_class_np(pred[0], threshold01)
    pred_end = _support_class_np(pred[-1], threshold01)
    true_start = _support_class_np(true[0], threshold01)
    true_end = _support_class_np(true[-1], threshold01)
    start_ok = bool(pred_start == true_start)
    end_ok = bool(pred_end == true_end)
    return {
        "endpoint_support_start_match": start_ok,
        "endpoint_support_end_match": end_ok,
        "endpoint_support_both_match": bool(start_ok and end_ok),
        "endpoint_support_pred_start": _support_label_from_class(pred_start),
        "endpoint_support_true_start": _support_label_from_class(true_start),
        "endpoint_support_pred_end": _support_label_from_class(pred_end),
        "endpoint_support_true_end": _support_label_from_class(true_end),
    }


def _pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n < 2:
        return 0.0
    x = x[:n]
    y = y[:n]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if den <= EPS:
        return 0.0
    return float(np.sum(x * y) / den)


def _safe_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals: List[float] = []
    for row in rows:
        try:
            v = float(row.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, sort_keys=True)
                out[key] = value
            writer.writerow(out)


@dataclass(frozen=True)
class ContactScaler:
    min_v: np.ndarray
    max_v: np.ndarray

    def to01(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        denom = np.maximum(self.max_v - self.min_v, 1.0e-6).astype(np.float32)
        return np.clip((arr - self.min_v) / denom, 0.0, 1.0).astype(np.float32, copy=False)

    def stats(self) -> Dict[str, Any]:
        return {
            "source_min": [float(v) for v in self.min_v.reshape(-1).tolist()],
            "source_max": [float(v) for v in self.max_v.reshape(-1).tolist()],
            "mapped_range": [0.0, 1.0],
        }


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    constant_count: int

    @classmethod
    def fit(cls, x: np.ndarray, *, passthrough_slice: Optional[slice] = None) -> "Standardizer":
        arr = np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
        mean = np.mean(arr, axis=0).astype(np.float32)
        std = np.std(arr, axis=0).astype(np.float32)
        keep = std > 1.0e-6
        std = np.where(keep, std, 1.0).astype(np.float32)
        if passthrough_slice is not None:
            mean[passthrough_slice] = 0.0
            std[passthrough_slice] = 1.0
            keep[passthrough_slice] = True
        return cls(mean=mean, std=std, constant_count=int(np.sum(~keep)))

    def transform_t(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def inverse_t(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * std + mean

    def transform_np(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        return ((arr - self.mean) / self.std).astype(np.float32, copy=False)


def _load_probe_clips(npz_root: Path, z_features: Path) -> Tuple[Dict[str, ClipData], ContactScaler]:
    states = load_clip_states(z_features, npz_root, clips=LOCKED_CLIPS)
    all_contact = np.concatenate([np.asarray(states[name], dtype=np.float32)[:, CONTACT_SLICE] for name in LOCKED_CLIPS], axis=0)
    scaler = ContactScaler(
        min_v=np.min(all_contact, axis=0).astype(np.float32),
        max_v=np.max(all_contact, axis=0).astype(np.float32),
    )
    out: Dict[str, ClipData] = {}
    for name in LOCKED_CLIPS:
        raw_path = npz_root / f"{name}.npz"
        if not raw_path.exists():
            raise FileNotFoundError(f"processed npz not found: {raw_path}")
        with np.load(raw_path, allow_pickle=True) as z:
            state = np.asarray(states[name], dtype=np.float32)
            n = int(state.shape[0])
            root_pos = np.asarray(z["root_pos"], dtype=np.float32).reshape(-1, 3)[:n]
            root_vel = np.asarray(z["root_vel"], dtype=np.float32).reshape(-1, 2)[:n]
            bone_angvel = np.asarray(z["bone_ang_vel"], dtype=np.float32).reshape(-1, ANGVEL_DIM)[:n]
            cond_in = np.asarray(z["cond_in"], dtype=np.float32)[:n]
        n = min(n, root_pos.shape[0], root_vel.shape[0], bone_angvel.shape[0], cond_in.shape[0])
        if n < 2:
            raise RuntimeError(f"{name}: aligned frame count too small ({n})")
        state = state[:n].copy()
        contact01 = scaler.to01(state[:, CONTACT_SLICE])
        state[:, CONTACT_SLICE] = contact01
        out[name] = ClipData(
            name=name,
            state281=_finite_float32(state),
            root_pos=_finite_float32(root_pos[:n]),
            root_vel=_finite_float32(root_vel[:n]),
            bone_angvel=_finite_float32(bone_angvel[:n]),
            cond_dir=_finite_float32(cond_in[:n, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]]),
            contact01=_finite_float32(contact01),
        )
    return out, scaler


def _window(arr: np.ndarray, start: int, length: int, *, wrap: bool) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    t = int(x.shape[0])
    s = int(start)
    n = int(length)
    if wrap:
        idx = (np.arange(s, s + n, dtype=np.int64) % t).astype(np.int64)
        return x[idx].copy()
    return x[s : s + n].copy()


def _context_window(arr: np.ndarray, start: int, context_len: int, *, wrap: bool) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    c = int(context_len)
    s = int(start)
    if wrap:
        idx = (np.arange(s - c, s, dtype=np.int64) % x.shape[0]).astype(np.int64)
        return x[idx].copy()
    lo = max(0, s - c)
    ctx = x[lo:s].copy()
    if ctx.shape[0] >= c:
        return ctx[-c:].copy()
    pad_src = x[max(0, min(s, x.shape[0] - 1))].reshape(1, -1)
    pad = np.repeat(pad_src, c - ctx.shape[0], axis=0)
    return np.concatenate([pad, ctx], axis=0).astype(np.float32, copy=False)


def _make_sequence(clip: ClipData, start: int, horizon: int) -> Dict[str, np.ndarray]:
    s = int(start)
    h = int(horizon)
    return {
        "state281": _window(clip.state281, s, h, wrap=False),
        "root_pos": _window(clip.root_pos, s, h, wrap=False),
        "root_vel": _window(clip.root_vel, s, h, wrap=False),
        "bone_angvel": _window(clip.bone_angvel, s, h, wrap=False),
        "cond_dir": _window(clip.cond_dir, s, h, wrap=False),
        "contact": _window(clip.contact01, s, h, wrap=False),
    }


def _goal_lowdim(
    seq: Mapping[str, np.ndarray],
    *,
    goal_contact_mode: str = "none",
    contact_support_threshold01: Tuple[float, float] = (CONTACT_LABEL_THRESHOLD, CONTACT_LABEL_THRESHOLD),
) -> np.ndarray:
    state = np.asarray(seq["state281"], dtype=np.float32).reshape(-1, STATE_DIM)
    root_pos = np.asarray(seq["root_pos"], dtype=np.float32).reshape(-1, 3)
    cond_dir = np.asarray(seq["cond_dir"], dtype=np.float32).reshape(-1, 2)
    root_disp = root_pos[-1] - root_pos[0]
    endpoint_root = state[-1, ROOT_MOTION_SLICE].reshape(-1)
    cond_yaw = np.concatenate([cond_dir, state[:, YAW_RATE_SLICE]], axis=1).reshape(-1)
    parts = [root_disp, endpoint_root, cond_yaw]
    mode = str(goal_contact_mode)
    if mode == "target_support_end":
        contact = np.asarray(seq["contact"], dtype=np.float32).reshape(-1, 2)
        parts.append(_support_onehot_np(contact[-1], contact_support_threshold01))
    elif mode != "none":
        raise ValueError(f"unsupported goal_contact_mode={mode!r}; expected one of {GOAL_CONTACT_MODES}")
    return _finite_float32(np.concatenate(parts, axis=0))


def _root_intent_eval_features(state: np.ndarray, root_pos: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=np.float32).reshape(-1, STATE_DIM)
    root = np.asarray(root_pos, dtype=np.float32).reshape(-1, 3)
    root_disp = root[-1] - root[0]
    endpoint_root = s[-1, ROOT_MOTION_SLICE].reshape(-1)
    yaw_traj = s[:, YAW_RATE_SLICE].reshape(-1)
    return _finite_float32(np.concatenate([root_disp, endpoint_root, yaw_traj], axis=0))


def _root_intent_component_metrics(
    *,
    pred_state: np.ndarray,
    pred_root_pos: np.ndarray,
    true_state: np.ndarray,
    true_root_pos: np.ndarray,
) -> Dict[str, float]:
    ps = np.asarray(pred_state, dtype=np.float32).reshape(-1, STATE_DIM)
    ts = np.asarray(true_state, dtype=np.float32).reshape(-1, STATE_DIM)
    pr = np.asarray(pred_root_pos, dtype=np.float32).reshape(-1, 3)
    tr = np.asarray(true_root_pos, dtype=np.float32).reshape(-1, 3)
    return {
        "root_disp_mse": _mse_np(pr[-1] - pr[0], tr[-1] - tr[0]),
        "endpoint_ego_vel_mse": _mse_np(ps[-1, EGO_VEL_SLICE], ts[-1, EGO_VEL_SLICE]),
        "endpoint_yaw_rate_mse": _mse_np(ps[-1, YAW_RATE_SLICE], ts[-1, YAW_RATE_SLICE]),
        "yaw_traj_mse": _mse_np(ps[:, YAW_RATE_SLICE], ts[:, YAW_RATE_SLICE]),
    }


def _build_items(
    clips: Mapping[str, ClipData],
    *,
    horizon: int,
    context_len: int,
    stride: int,
    goal_contact_mode: str = "none",
    goal_support_threshold01: Tuple[float, float] = (CONTACT_LABEL_THRESHOLD, CONTACT_LABEL_THRESHOLD),
) -> List[BridgeItem]:
    items: List[BridgeItem] = []
    for name in TURN_CLIPS:
        clip = clips[name]
        max_start = int(clip.state281.shape[0]) - int(horizon)
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, max(1, int(stride))):
            seq = _make_sequence(clip, start, int(horizon))
            items.append(
                BridgeItem(
                    clip=name,
                    start=int(start),
                    end=int(start + horizon - 1),
                    ctx_state=_context_window(clip.state281, start, int(context_len), wrap=(name == WALK_F)),
                    ctx_aux=_context_window(clip.bone_angvel, start, int(context_len), wrap=(name == WALK_F)),
                    seq=seq,
                    goal_lowdim=_goal_lowdim(
                        seq,
                        goal_contact_mode=str(goal_contact_mode),
                        contact_support_threshold01=goal_support_threshold01,
                    ),
                )
            )
    return items


def _build_split(
    items: Sequence[BridgeItem],
    *,
    train_fraction: float,
    block_gap: int,
) -> SplitDef:
    by_clip: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        by_clip[item.clip].append(int(idx))
    train_idx: List[int] = []
    test_idx: List[int] = []
    notes: List[str] = []
    for clip in MATCHED_TARGETS:
        idxs = sorted(by_clip.get(clip, []), key=lambda i: items[i].start)
        n = len(idxs)
        if n < 2:
            notes.append(f"{clip}: n={n} skipped")
            continue
        train_n = max(1, min(n - 1, int(math.floor(float(train_fraction) * n))))
        test_start = train_n + max(0, int(block_gap))
        if test_start >= n:
            test_start = train_n
        train_idx.extend(idxs[:train_n])
        test_idx.extend(idxs[test_start:])
        notes.append(f"{clip}: train={train_n} gap={max(0, test_start - train_n)} test={n - test_start}")
    return SplitDef(
        name="contiguous_block",
        kind="contiguous_block",
        train_idx=tuple(train_idx),
        test_idx=tuple(test_idx),
        note="; ".join(notes),
    )


def _stack_items(items: Sequence[BridgeItem], idxs: Sequence[int], key: str) -> np.ndarray:
    return np.stack([np.asarray(items[int(i)].seq[key], dtype=np.float32) for i in idxs], axis=0).astype(np.float32)


def _stack_attr(items: Sequence[BridgeItem], idxs: Sequence[int], attr: str) -> np.ndarray:
    return np.stack([np.asarray(getattr(items[int(i)], attr), dtype=np.float32) for i in idxs], axis=0).astype(np.float32)


def _fit_normalizers(items: Sequence[BridgeItem], idxs: Sequence[int]) -> Tuple[Standardizer, Standardizer, Standardizer]:
    state_frames = np.concatenate(
        [
            _stack_attr(items, idxs, "ctx_state").reshape(-1, STATE_DIM),
            _stack_items(items, idxs, "state281").reshape(-1, STATE_DIM),
        ],
        axis=0,
    )
    aux_frames = np.concatenate(
        [
            _stack_attr(items, idxs, "ctx_aux").reshape(-1, ANGVEL_DIM),
            _stack_items(items, idxs, "bone_angvel").reshape(-1, ANGVEL_DIM),
        ],
        axis=0,
    )
    goal = _stack_attr(items, idxs, "goal_lowdim").reshape(len(idxs), -1)
    return (
        Standardizer.fit(state_frames, passthrough_slice=CONTACT_SLICE),
        Standardizer.fit(aux_frames),
        Standardizer.fit(goal),
    )


def _batch_from_items(
    *,
    items: Sequence[BridgeItem],
    idxs: Sequence[int],
    state_norm: Standardizer,
    aux_norm: Standardizer,
    goal_norm: Standardizer,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    ctx_state = torch.as_tensor(_stack_attr(items, idxs, "ctx_state"), dtype=torch.float32, device=device)
    gt_state = torch.as_tensor(_stack_items(items, idxs, "state281"), dtype=torch.float32, device=device)
    ctx_aux = torch.as_tensor(_stack_attr(items, idxs, "ctx_aux"), dtype=torch.float32, device=device)
    gt_aux = torch.as_tensor(_stack_items(items, idxs, "bone_angvel"), dtype=torch.float32, device=device)
    goal = torch.as_tensor(_stack_attr(items, idxs, "goal_lowdim"), dtype=torch.float32, device=device)
    root_pos = torch.as_tensor(_stack_items(items, idxs, "root_pos"), dtype=torch.float32, device=device)
    root_vel = torch.as_tensor(_stack_items(items, idxs, "root_vel"), dtype=torch.float32, device=device)
    cond_dir = torch.as_tensor(_stack_items(items, idxs, "cond_dir"), dtype=torch.float32, device=device)
    contact = gt_state[:, :, CONTACT_SLICE]
    return {
        "ctx_state_raw": ctx_state,
        "gt_state_raw": gt_state,
        "ctx_aux_raw": ctx_aux,
        "gt_aux_raw": gt_aux,
        "ctx_state_n": state_norm.transform_t(ctx_state),
        "gt_state_n": state_norm.transform_t(gt_state),
        "ctx_aux_n": aux_norm.transform_t(ctx_aux),
        "gt_aux_n": aux_norm.transform_t(gt_aux),
        "goal_n": goal_norm.transform_t(goal),
        "goal_raw": goal,
        "gt_contact": contact,
        "ctx_contact": ctx_state[:, :, CONTACT_SLICE],
        "root_pos": root_pos,
        "root_vel": root_vel,
        "cond_dir": cond_dir,
    }


class ContactPlanGRU(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        h = int(latent_dim)
        self.hist_gru = nn.GRU(2, h, batch_first=True)
        self.init = nn.Sequential(nn.Linear(3 * h, h), nn.Tanh())
        self.cell = nn.GRUCell(2 + 2 * h, h)
        self.head = nn.Linear(h, 2)

    def forward(self, ctx_contact: torch.Tensor, z_ctx: torch.Tensor, z_goal: torch.Tensor, horizon: int) -> torch.Tensor:
        _, h_n = self.hist_gru(ctx_contact)
        hist = h_n[-1]
        h = self.init(torch.cat([hist, z_ctx, z_goal], dim=-1))
        prev = ctx_contact[:, -1]
        outs: List[torch.Tensor] = []
        for _ in range(int(horizon)):
            h = self.cell(torch.cat([prev, z_ctx, z_goal], dim=-1), h)
            cur = torch.sigmoid(self.head(h))
            outs.append(cur)
            prev = cur
        return torch.stack(outs, dim=1)


class GRUBridgeProbe(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        goal_dim: int,
        aux_dim: int,
        latent_dim: int,
        contact_embed_dim: int,
    ) -> None:
        super().__init__()
        h = int(latent_dim)
        ce = int(contact_embed_dim)
        self.ctx_gru = nn.GRU(int(state_dim), h, batch_first=True)
        self.goal_mlp = nn.Sequential(nn.Linear(int(goal_dim), h), nn.GELU(), nn.Linear(h, h))
        self.contact_plan = ContactPlanGRU(h)
        self.contact_embed = nn.Sequential(nn.Linear(2, ce), nn.GELU(), nn.Linear(ce, ce))
        self.dec_init = nn.Sequential(nn.Linear(2 * h, h), nn.Tanh())
        self.dec_cell = nn.GRUCell(int(state_dim) + 2 * h + ce, h)
        self.root_head = nn.Linear(h, 3)
        self.pose_root_head = nn.Linear(h, 6)
        self.pose_local_head = nn.Linear(h, POSE_DIM - 6)
        self.aux_root_head = nn.Linear(h, 3)
        self.aux_local_head = nn.Linear(h, int(aux_dim) - 3)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        spec: VariantSpec,
        *,
        negative_contact: str,
        contact_shift: int,
        retain_usage: bool = False,
    ) -> Dict[str, Any]:
        ctx_state = batch["ctx_state_n"]
        ctx_contact = batch["ctx_contact"]
        goal_n = batch["goal_n"]
        gt_contact = batch["gt_contact"]
        horizon = int(gt_contact.shape[1])
        _, h_n = self.ctx_gru(ctx_state)
        z_ctx = h_n[-1]
        if spec.use_goal:
            z_goal = self.goal_mlp(goal_n)
        else:
            z_goal = torch.zeros_like(z_ctx)

        predicted_contact: Optional[torch.Tensor] = None
        if spec.contact_mode in {"predicted", "oracle", "negative"}:
            predicted_contact = self.contact_plan(ctx_contact, z_ctx, z_goal, horizon)

        if spec.contact_mode == "predicted":
            contact_used = predicted_contact
        elif spec.contact_mode == "oracle":
            contact_used = gt_contact
        elif spec.contact_mode == "negative":
            if negative_contact == "random":
                g = torch.Generator(device=gt_contact.device)
                g.manual_seed(20260606)
                contact_used = torch.rand(gt_contact.shape, dtype=gt_contact.dtype, device=gt_contact.device, generator=g)
            else:
                shift = max(1, min(abs(int(contact_shift)), horizon - 1 if horizon > 1 else 1))
                contact_used = torch.roll(gt_contact, shifts=shift, dims=1)
        else:
            contact_used = torch.zeros_like(gt_contact)

        assert contact_used is not None
        prev_state = ctx_state[:, -1]
        prev_aux = batch["ctx_aux_n"][:, -1]
        h = self.dec_init(torch.cat([z_ctx, z_goal], dim=-1))
        states: List[torch.Tensor] = []
        auxes: List[torch.Tensor] = []
        for t in range(horizon):
            contact_t = contact_used[:, t]
            if spec.contact_mode == "none":
                contact_feat = torch.zeros((ctx_state.shape[0], self.contact_embed[-1].out_features), dtype=ctx_state.dtype, device=ctx_state.device)
            else:
                contact_feat = self.contact_embed(contact_t)
            dec_in = torch.cat([prev_state, z_ctx, z_goal, contact_feat], dim=-1)
            h = self.dec_cell(dec_in, h)
            cur = prev_state.clone()
            cur[:, ROOT_ROT6D_SLICE] = prev_state[:, ROOT_ROT6D_SLICE] + self.pose_root_head(h)
            cur[:, NONROOT_ROT6D_SLICE] = prev_state[:, NONROOT_ROT6D_SLICE] + self.pose_local_head(h)
            cur[:, ROOT_MOTION_SLICE] = prev_state[:, ROOT_MOTION_SLICE] + self.root_head(h)
            cur[:, CONTACT_SLICE] = contact_t
            aux = prev_aux.clone()
            aux[:, BONE0_ANGVEL_SLICE] = prev_aux[:, BONE0_ANGVEL_SLICE] + self.aux_root_head(h)
            aux[:, NONROOT_ANGVEL_SLICE] = prev_aux[:, NONROOT_ANGVEL_SLICE] + self.aux_local_head(h)
            states.append(cur)
            auxes.append(aux)
            prev_state = cur
            prev_aux = aux

        usage_tensors: Dict[str, Optional[torch.Tensor]] = {
            "z_ctx": z_ctx,
            "z_goal": z_goal if z_goal.requires_grad else None,
            "contact_used": contact_used if contact_used.requires_grad else None,
            "predicted_contact": predicted_contact if predicted_contact is not None and predicted_contact.requires_grad else None,
        }
        if retain_usage:
            for tensor in usage_tensors.values():
                if tensor is not None:
                    tensor.retain_grad()
        return {
            "state_n": torch.stack(states, dim=1),
            "aux_n": torch.stack(auxes, dim=1),
            "contact_used": contact_used,
            "predicted_contact": predicted_contact,
            "usage_tensors": usage_tensors,
        }


def _world_root_vel_from_ego_torch(ego_vel: torch.Tensor, cond_dir: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(cond_dir, dim=-1, keepdim=True).clamp_min(EPS)
    fwd = cond_dir / norm
    lat = torch.stack([-fwd[..., 1], fwd[..., 0]], dim=-1)
    return ego_vel[..., 0:1] * fwd + ego_vel[..., 1:2] * lat


def _integrate_root_pos_torch(root_vel: torch.Tensor, start_root: torch.Tensor) -> torch.Tensor:
    b, h, _ = root_vel.shape
    out = root_vel.new_zeros((b, h, 3))
    out[:, 0, :] = start_root
    if h > 1:
        steps = root_vel[:, :-1] / float(FPS)
        xy = torch.cumsum(steps, dim=1)
        out[:, 1:, :2] = start_root[:, None, :2] + xy
        out[:, 1:, 2] = start_root[:, None, 2]
    return out


def _world_root_vel_from_ego_np(ego_vel: np.ndarray, cond_dir: np.ndarray) -> np.ndarray:
    ego = np.asarray(ego_vel, dtype=np.float32).reshape(-1, 2)
    cmd = np.asarray(cond_dir, dtype=np.float32).reshape(-1, 2)
    norm = np.maximum(np.linalg.norm(cmd, axis=1, keepdims=True), EPS)
    fwd = cmd / norm
    lat = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
    return (ego[:, 0:1] * fwd + ego[:, 1:2] * lat).astype(np.float32)


def _integrate_root_pos_np(root_vel: np.ndarray, start_root: np.ndarray) -> np.ndarray:
    vel = np.asarray(root_vel, dtype=np.float32).reshape(-1, 2)
    root0 = np.asarray(start_root, dtype=np.float32).reshape(3)
    out = np.zeros((vel.shape[0], 3), dtype=np.float32)
    out[0] = root0
    for i in range(1, vel.shape[0]):
        out[i, :2] = out[i - 1, :2] + vel[i - 1] / float(FPS)
        out[i, 2] = root0[2]
    return out


def _directlocal(state: np.ndarray, aux: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    a = np.asarray(aux, dtype=np.float32).reshape(-1)
    return np.concatenate([s[NONROOT_ROT6D_SLICE], a[NONROOT_ANGVEL_SLICE]], axis=0)


def _loss_terms(
    *,
    out: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    state_norm: Standardizer,
    aux_norm: Standardizer,
    spec: VariantSpec,
    contact_loss: ContactLossConfig = DEFAULT_CONTACT_LOSS_CONFIG,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_state = state_norm.inverse_t(out["state_n"])
    pred_aux = aux_norm.inverse_t(out["aux_n"])
    true_state = batch["gt_state_raw"]
    true_aux = batch["gt_aux_raw"]
    true_contact = batch["gt_contact"]
    pred_root_vel = _world_root_vel_from_ego_torch(pred_state[:, :, EGO_VEL_SLICE], batch["cond_dir"])
    pred_root_pos = _integrate_root_pos_torch(pred_root_vel, batch["root_pos"][:, 0])

    root_pos = F.mse_loss(pred_root_pos, batch["root_pos"])
    root_vel = F.mse_loss(pred_root_vel, batch["root_vel"])
    ego = F.mse_loss(pred_state[:, :, EGO_VEL_SLICE], true_state[:, :, EGO_VEL_SLICE])
    yaw = F.mse_loss(pred_state[:, :, YAW_RATE_SLICE], true_state[:, :, YAW_RATE_SLICE])
    pose_root = F.mse_loss(pred_state[:, :, ROOT_ROT6D_SLICE], true_state[:, :, ROOT_ROT6D_SLICE])
    pose_nonroot = F.mse_loss(pred_state[:, :, NONROOT_ROT6D_SLICE], true_state[:, :, NONROOT_ROT6D_SLICE])
    aux_root = F.mse_loss(pred_aux[:, :, BONE0_ANGVEL_SLICE], true_aux[:, :, BONE0_ANGVEL_SLICE])
    aux_nonroot = F.mse_loss(pred_aux[:, :, NONROOT_ANGVEL_SLICE], true_aux[:, :, NONROOT_ANGVEL_SLICE])
    contact_state = F.mse_loss(out["contact_used"], true_contact)
    if pred_state.shape[1] > 1:
        pose_step = F.mse_loss(
            pred_state[:, 1:, POSE_SLICE] - pred_state[:, :-1, POSE_SLICE],
            true_state[:, 1:, POSE_SLICE] - true_state[:, :-1, POSE_SLICE],
        )
        aux_step = F.mse_loss(
            pred_aux[:, 1:] - pred_aux[:, :-1],
            true_aux[:, 1:] - true_aux[:, :-1],
        )
        contact_step = F.mse_loss(
            out["contact_used"][:, 1:] - out["contact_used"][:, :-1],
            true_contact[:, 1:] - true_contact[:, :-1],
        )
    else:
        pose_step = pred_state.new_tensor(0.0)
        aux_step = pred_state.new_tensor(0.0)
        contact_step = pred_state.new_tensor(0.0)
    seam_c1 = F.mse_loss(
        pred_state[:, 0] - batch["ctx_state_raw"][:, -1],
        true_state[:, 0] - batch["ctx_state_raw"][:, -1],
    )
    pred_contact = out.get("predicted_contact")
    if pred_contact is not None:
        contact_predict = F.mse_loss(pred_contact, true_contact)
        contact_bce = F.binary_cross_entropy(pred_contact.clamp(1.0e-6, 1.0 - 1.0e-6), true_contact.clamp(0.0, 1.0))
        contact_endpoint_support = _endpoint_support_loss_t(
            pred_contact,
            true_contact,
            contact_loss.contact_support_threshold01,
        )
    else:
        contact_predict = pred_state.new_tensor(0.0)
        contact_bce = pred_state.new_tensor(0.0)
        contact_endpoint_support = pred_state.new_tensor(0.0)

    loss = (
        12.0 * root_pos
        + 3.0 * root_vel
        + 3.0 * ego
        + 2.0 * yaw
        + 1.0 * pose_root
        + 1.0 * pose_nonroot
        + 0.25 * aux_root
        + 0.25 * aux_nonroot
        + 0.75 * pose_step
        + 0.10 * aux_step
        + 0.50 * seam_c1
        + float(contact_loss.contact_step_weight) * contact_step
        + (
            float(contact_loss.contact_predict_mse_weight) * contact_predict
            + float(contact_loss.contact_predict_bce_weight) * contact_bce
            + float(contact_loss.contact_endpoint_support_weight) * contact_endpoint_support
            if spec.contact_mode in {"predicted", "oracle", "negative"}
            else 0.0
        )
        + (float(contact_loss.contact_state_weight) * contact_state if spec.contact_mode == "predicted" else 0.0)
    )
    terms = {
        "loss": float(loss.detach().cpu().item()),
        "root_pos_mse": float(root_pos.detach().cpu().item()),
        "root_vel_mse": float(root_vel.detach().cpu().item()),
        "ego_vel_mse": float(ego.detach().cpu().item()),
        "yaw_rate_mse": float(yaw.detach().cpu().item()),
        "pose_root_rot6d_mse": float(pose_root.detach().cpu().item()),
        "pose_nonroot_rot6d_mse": float(pose_nonroot.detach().cpu().item()),
        "bone_angvel_root_mse": float(aux_root.detach().cpu().item()),
        "bone_angvel_nonroot_mse": float(aux_nonroot.detach().cpu().item()),
        "contact_plan_mse": float(contact_state.detach().cpu().item()),
        "contact_predict_mse": float(contact_predict.detach().cpu().item()),
        "contact_predict_bce": float(contact_bce.detach().cpu().item()),
        "contact_endpoint_support_ce": float(contact_endpoint_support.detach().cpu().item()),
        "pose_step_mse": float(pose_step.detach().cpu().item()),
        "bone_angvel_step_mse": float(aux_step.detach().cpu().item()),
        "contact_cycle_delta_mse": float(contact_step.detach().cpu().item()),
        "seam_c1_state_mse": float(seam_c1.detach().cpu().item()),
    }
    return loss, terms


def _module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    count = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.sum(g * g).cpu().item())
        count += int(g.numel())
    if count <= 0:
        return 0.0
    return float(math.sqrt(total / count))


def _tensor_l2_mean(tensor: Optional[torch.Tensor]) -> float:
    if tensor is None:
        return 0.0
    x = tensor.detach()
    if x.numel() <= 0:
        return 0.0
    return float(torch.sqrt(torch.mean(x * x)).cpu().item())


def _tensor_grad_l2_mean(tensor: Optional[torch.Tensor]) -> float:
    if tensor is None or tensor.grad is None:
        return 0.0
    g = tensor.grad.detach()
    if g.numel() <= 0:
        return 0.0
    return float(torch.sqrt(torch.mean(g * g)).cpu().item())


def _train_variant(
    *,
    spec: VariantSpec,
    train_batch: Mapping[str, torch.Tensor],
    state_norm: Standardizer,
    aux_norm: Standardizer,
    goal_dim: int,
    latent_dim: int,
    contact_embed_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    negative_contact: str,
    contact_shift: int,
    contact_loss: ContactLossConfig = DEFAULT_CONTACT_LOSS_CONFIG,
) -> Tuple[GRUBridgeProbe, Dict[str, float], Dict[str, Any]]:
    torch.manual_seed(int(seed))
    model = GRUBridgeProbe(
        state_dim=STATE_DIM,
        goal_dim=int(goal_dim),
        aux_dim=ANGVEL_DIM,
        latent_dim=int(latent_dim),
        contact_embed_dim=int(contact_embed_dim),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    final_terms: Dict[str, float] = {}
    for _epoch in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(train_batch, spec, negative_contact=negative_contact, contact_shift=int(contact_shift))
        loss, final_terms = _loss_terms(
            out=out,
            batch=train_batch,
            state_norm=state_norm,
            aux_norm=aux_norm,
            spec=spec,
            contact_loss=contact_loss,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"{spec.name}: non-finite loss")
        loss.backward()
        opt.step()

    model.train()
    opt.zero_grad(set_to_none=True)
    usage_out = model(
        train_batch,
        spec,
        negative_contact=negative_contact,
        contact_shift=int(contact_shift),
        retain_usage=True,
    )
    usage_loss, usage_terms = _loss_terms(
        out=usage_out,
        batch=train_batch,
        state_norm=state_norm,
        aux_norm=aux_norm,
        spec=spec,
        contact_loss=contact_loss,
    )
    usage_loss.backward()
    usage_tensors = usage_out["usage_tensors"]
    grad_usage = {
        "variant": spec.name,
        "usage_loss": float(usage_loss.detach().cpu().item()),
        "z_ctx_l2_mean": _tensor_l2_mean(usage_tensors.get("z_ctx")),
        "z_ctx_grad_l2_mean": _tensor_grad_l2_mean(usage_tensors.get("z_ctx")),
        "z_goal_l2_mean": _tensor_l2_mean(usage_tensors.get("z_goal")),
        "z_goal_grad_l2_mean": _tensor_grad_l2_mean(usage_tensors.get("z_goal")),
        "contact_used_l2_mean": _tensor_l2_mean(usage_tensors.get("contact_used")),
        "contact_used_grad_l2_mean": _tensor_grad_l2_mean(usage_tensors.get("contact_used")),
        "predicted_contact_l2_mean": _tensor_l2_mean(usage_tensors.get("predicted_contact")),
        "predicted_contact_grad_l2_mean": _tensor_grad_l2_mean(usage_tensors.get("predicted_contact")),
        "ctx_gru_grad_l2_mean": _module_grad_norm(model.ctx_gru),
        "goal_mlp_grad_l2_mean": _module_grad_norm(model.goal_mlp),
        "contact_plan_grad_l2_mean": _module_grad_norm(model.contact_plan),
        "contact_embed_grad_l2_mean": _module_grad_norm(model.contact_embed),
        "decoder_cell_grad_l2_mean": _module_grad_norm(model.dec_cell),
        "root_head_grad_l2_mean": _module_grad_norm(model.root_head),
        "pose_root_head_grad_l2_mean": _module_grad_norm(model.pose_root_head),
        "pose_local_head_grad_l2_mean": _module_grad_norm(model.pose_local_head),
        "aux_root_head_grad_l2_mean": _module_grad_norm(model.aux_root_head),
        "aux_local_head_grad_l2_mean": _module_grad_norm(model.aux_local_head),
        "contact_step_weight": float(contact_loss.contact_step_weight),
        "contact_predict_mse_weight": float(contact_loss.contact_predict_mse_weight),
        "contact_predict_bce_weight": float(contact_loss.contact_predict_bce_weight),
        "contact_state_weight": float(contact_loss.contact_state_weight),
        "contact_endpoint_support_weight": float(contact_loss.contact_endpoint_support_weight),
        **{f"final_{k}": v for k, v in usage_terms.items()},
    }
    opt.zero_grad(set_to_none=True)
    return model, final_terms, grad_usage


def _predict_model(
    *,
    model: GRUBridgeProbe,
    spec: VariantSpec,
    batch: Mapping[str, torch.Tensor],
    state_norm: Standardizer,
    aux_norm: Standardizer,
    negative_contact: str,
    contact_shift: int,
) -> Dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        out = model(batch, spec, negative_contact=negative_contact, contact_shift=int(contact_shift))
        state = state_norm.inverse_t(out["state_n"])
        aux = aux_norm.inverse_t(out["aux_n"])
        pred_contact = out.get("predicted_contact")
        return {
            "state": state.detach().cpu().numpy().astype(np.float32),
            "aux": aux.detach().cpu().numpy().astype(np.float32),
            "contact_used": out["contact_used"].detach().cpu().numpy().astype(np.float32),
            "predicted_contact": (
                pred_contact.detach().cpu().numpy().astype(np.float32)
                if pred_contact is not None
                else np.zeros_like(out["contact_used"].detach().cpu().numpy().astype(np.float32))
            ),
        }


def _baseline_ctx_last_hold(items: Sequence[BridgeItem], idxs: Sequence[int], horizon: int) -> Dict[str, np.ndarray]:
    states: List[np.ndarray] = []
    auxes: List[np.ndarray] = []
    contacts: List[np.ndarray] = []
    for item_idx in idxs:
        item = items[int(item_idx)]
        state = np.repeat(item.ctx_state[-1:].astype(np.float32), int(horizon), axis=0)
        aux = np.repeat(item.ctx_aux[-1:].astype(np.float32), int(horizon), axis=0)
        states.append(state)
        auxes.append(aux)
        contacts.append(state[:, CONTACT_SLICE])
    return {
        "state": np.stack(states, axis=0).astype(np.float32),
        "aux": np.stack(auxes, axis=0).astype(np.float32),
        "contact_used": np.stack(contacts, axis=0).astype(np.float32),
        "predicted_contact": np.stack(contacts, axis=0).astype(np.float32),
    }


def _baseline_root_linear_pose_hold(items: Sequence[BridgeItem], idxs: Sequence[int], horizon: int) -> Dict[str, np.ndarray]:
    states: List[np.ndarray] = []
    auxes: List[np.ndarray] = []
    contacts: List[np.ndarray] = []
    h = int(horizon)
    for item_idx in idxs:
        item = items[int(item_idx)]
        true_state = np.asarray(item.seq["state281"], dtype=np.float32).reshape(h, STATE_DIM)
        root_pos = np.asarray(item.seq["root_pos"], dtype=np.float32).reshape(h, 3)
        cond_dir = np.asarray(item.seq["cond_dir"], dtype=np.float32).reshape(h, 2)
        state = np.repeat(item.ctx_state[-1:].astype(np.float32), h, axis=0)
        aux = np.repeat(item.ctx_aux[-1:].astype(np.float32), h, axis=0)
        disp_xy = root_pos[-1, :2] - root_pos[0, :2]
        world_vel = disp_xy * float(FPS) / float(max(1, h - 1))
        fwd = cond_dir / np.maximum(np.linalg.norm(cond_dir, axis=1, keepdims=True), EPS)
        lat = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
        ego = np.stack([np.sum(world_vel.reshape(1, 2) * fwd, axis=1), np.sum(world_vel.reshape(1, 2) * lat, axis=1)], axis=1)
        state[:, EGO_VEL_SLICE] = ego.astype(np.float32)
        state[:, YAW_RATE_SLICE] = np.linspace(
            float(item.ctx_state[-1, YAW_RATE_SLICE][0]),
            float(true_state[-1, YAW_RATE_SLICE][0]),
            h,
            dtype=np.float32,
        ).reshape(h, 1)
        states.append(state)
        auxes.append(aux)
        contacts.append(state[:, CONTACT_SLICE])
    return {
        "state": np.stack(states, axis=0).astype(np.float32),
        "aux": np.stack(auxes, axis=0).astype(np.float32),
        "contact_used": np.stack(contacts, axis=0).astype(np.float32),
        "predicted_contact": np.stack(contacts, axis=0).astype(np.float32),
    }


def _contact_cycle_metrics(pred_contact: np.ndarray, true_contact: np.ndarray) -> Dict[str, float]:
    p = np.asarray(pred_contact, dtype=np.float32).reshape(-1, 2)
    t = np.asarray(true_contact, dtype=np.float32).reshape(-1, 2)
    if p.shape[0] > 1 and t.shape[0] > 1:
        delta_mse = _mse_np(np.diff(p, axis=0), np.diff(t, axis=0))
    else:
        delta_mse = 0.0
    balance_corr = _pearson_np(p[:, 0] - p[:, 1], t[:, 0] - t[:, 1])
    total_corr = _pearson_np(np.sum(p, axis=1), np.sum(t, axis=1))
    return {
        "contact_cycle_delta_mse": delta_mse,
        "contact_balance_corr": balance_corr,
        "contact_total_corr": total_corr,
        "contact_phase_consistency": 0.5 * (balance_corr + total_corr),
    }


def _metric_rows_for_predictions(
    *,
    variant: str,
    model_kind: str,
    partition: str,
    items: Sequence[BridgeItem],
    idxs: Sequence[int],
    pred: Mapping[str, np.ndarray],
    contact_support_threshold01: Tuple[float, float] = DEFAULT_CONTACT_LOSS_CONFIG.contact_support_threshold01,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pred_state_all = np.asarray(pred["state"], dtype=np.float32)
    pred_aux_all = np.asarray(pred["aux"], dtype=np.float32)
    contact_used_all = np.asarray(pred["contact_used"], dtype=np.float32)
    predicted_contact_all = np.asarray(pred.get("predicted_contact", contact_used_all), dtype=np.float32)
    for local_i, item_idx in enumerate(idxs):
        item = items[int(item_idx)]
        true_state = np.asarray(item.seq["state281"], dtype=np.float32).reshape(pred_state_all.shape[1], STATE_DIM)
        true_aux = np.asarray(item.seq["bone_angvel"], dtype=np.float32).reshape(pred_aux_all.shape[1], ANGVEL_DIM)
        true_contact = true_state[:, CONTACT_SLICE]
        pred_state = pred_state_all[local_i].copy()
        pred_aux = pred_aux_all[local_i].copy()
        contact_used = np.clip(contact_used_all[local_i], 0.0, 1.0)
        predicted_contact = np.clip(predicted_contact_all[local_i], 0.0, 1.0)
        pred_state[:, CONTACT_SLICE] = contact_used
        cond_dir = np.asarray(item.seq["cond_dir"], dtype=np.float32)
        pred_root_vel = _world_root_vel_from_ego_np(pred_state[:, EGO_VEL_SLICE], cond_dir)
        pred_root_pos = _integrate_root_pos_np(pred_root_vel, np.asarray(item.seq["root_pos"], dtype=np.float32)[0])
        true_root_pos = np.asarray(item.seq["root_pos"], dtype=np.float32)
        pred_intent = _root_intent_eval_features(pred_state, pred_root_pos)
        true_intent = _root_intent_eval_features(true_state, true_root_pos)
        c = _contact_cycle_metrics(contact_used, true_contact)
        bce = -np.mean(
            true_contact * np.log(np.clip(contact_used, 1.0e-6, 1.0 - 1.0e-6))
            + (1.0 - true_contact) * np.log(np.clip(1.0 - contact_used, 1.0e-6, 1.0 - 1.0e-6))
        )
        finite_parts = [pred_state.reshape(-1), pred_aux.reshape(-1), contact_used.reshape(-1)]
        finite = np.concatenate(finite_parts, axis=0)
        entry_pred_delta = pred_state[0] - np.asarray(item.ctx_state[-1], dtype=np.float32)
        entry_true_delta = true_state[0] - np.asarray(item.ctx_state[-1], dtype=np.float32)
        entry_pred_local = _directlocal(pred_state[0], pred_aux[0]) - _directlocal(item.ctx_state[-1], item.ctx_aux[-1])
        entry_true_local = _directlocal(true_state[0], true_aux[0]) - _directlocal(item.ctx_state[-1], item.ctx_aux[-1])
        if pred_state.shape[0] > 1:
            exit_pred_delta = pred_state[-1] - pred_state[-2]
            exit_true_delta = true_state[-1] - true_state[-2]
            exit_pred_local = _directlocal(pred_state[-1], pred_aux[-1]) - _directlocal(pred_state[-2], pred_aux[-2])
            exit_true_local = _directlocal(true_state[-1], true_aux[-1]) - _directlocal(true_state[-2], true_aux[-2])
        else:
            exit_pred_delta = entry_pred_delta
            exit_true_delta = entry_true_delta
            exit_pred_local = entry_pred_local
            exit_true_local = entry_true_local
        row: Dict[str, Any] = {
            "variant": variant,
            "model_kind": model_kind,
            "partition": partition,
            "clip": item.clip,
            "start": int(item.start),
            "end": int(item.end),
            "finite_fraction": float(np.mean(np.isfinite(finite))) if finite.size else 1.0,
            "state_mse": _mse_np(pred_state, true_state),
            "root_intent_mse": _mse_np(pred_intent, true_intent),
            "root_pos_mse": _mse_np(pred_root_pos, true_root_pos),
            "root_vel_mse": _mse_np(pred_root_vel, item.seq["root_vel"]),
            "ego_vel_mse": _mse_np(pred_state[:, EGO_VEL_SLICE], true_state[:, EGO_VEL_SLICE]),
            "yaw_rate_mse": _mse_np(pred_state[:, YAW_RATE_SLICE], true_state[:, YAW_RATE_SLICE]),
            "pose_root_rot6d_mse": _mse_np(pred_state[:, ROOT_ROT6D_SLICE], true_state[:, ROOT_ROT6D_SLICE]),
            "pose_nonroot_rot6d_mse": _mse_np(pred_state[:, NONROOT_ROT6D_SLICE], true_state[:, NONROOT_ROT6D_SLICE]),
            "bone_angvel_root_mse": _mse_np(pred_aux[:, BONE0_ANGVEL_SLICE], true_aux[:, BONE0_ANGVEL_SLICE]),
            "bone_angvel_nonroot_mse": _mse_np(pred_aux[:, NONROOT_ANGVEL_SLICE], true_aux[:, NONROOT_ANGVEL_SLICE]),
            "contact_plan_mse": _mse_np(contact_used, true_contact),
            "contact_plan_bce": float(bce) if math.isfinite(float(bce)) else 0.0,
            "predicted_contact_mse": _mse_np(predicted_contact, true_contact),
            "seam_c0_state_mse": _mse_np(pred_state[0], true_state[0]),
            "seam_c1_state_mse": _mse_np(entry_pred_delta, entry_true_delta),
            "seam_c0_directlocal_mse": _mse_np(_directlocal(pred_state[0], pred_aux[0]), _directlocal(true_state[0], true_aux[0])),
            "seam_c1_directlocal_mse": _mse_np(entry_pred_local, entry_true_local),
            "seam_exit_c0_state_mse": _mse_np(pred_state[-1], true_state[-1]),
            "seam_exit_c1_state_mse": _mse_np(exit_pred_delta, exit_true_delta),
            "seam_exit_c0_directlocal_mse": _mse_np(_directlocal(pred_state[-1], pred_aux[-1]), _directlocal(true_state[-1], true_aux[-1])),
            "seam_exit_c1_directlocal_mse": _mse_np(exit_pred_local, exit_true_local),
        }
        row.update(c)
        row.update(_endpoint_support_metrics_np(contact_used, true_contact, contact_support_threshold01))
        row.update(
            _root_intent_component_metrics(
                pred_state=pred_state,
                pred_root_pos=pred_root_pos,
                true_state=true_state,
                true_root_pos=true_root_pos,
            )
        )
        rows.append(row)
    return rows


def _root_pos_invariant_row(
    *,
    variant: str,
    model_kind: str,
    partition: str,
    batch: Mapping[str, torch.Tensor],
    pred: Mapping[str, np.ndarray],
    metric_rows: Sequence[Mapping[str, Any]],
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    pred_state = torch.as_tensor(np.asarray(pred["state"], dtype=np.float32), dtype=torch.float32, device=batch["cond_dir"].device)
    root_vel = _world_root_vel_from_ego_torch(pred_state[:, :, EGO_VEL_SLICE], batch["cond_dir"])
    root_pos = _integrate_root_pos_torch(root_vel, batch["root_pos"][:, 0])
    loss_style = float(F.mse_loss(root_pos, batch["root_pos"]).detach().cpu().item())
    eval_style = _safe_mean(metric_rows, "root_pos_mse")
    abs_delta = abs(loss_style - eval_style)
    tol = float(atol) + float(rtol) * max(abs(loss_style), abs(eval_style), 1.0)
    ok = bool(abs_delta <= tol)
    row = {
        "variant": variant,
        "model_kind": model_kind,
        "partition": partition,
        "loss_style_root_pos_mse": loss_style,
        "eval_style_root_pos_mse": eval_style,
        "abs_delta": abs_delta,
        "tolerance": tol,
        "ok": ok,
    }
    if not ok:
        raise AssertionError(
            f"root_pos invariant failed for {variant}/{partition}: "
            f"loss_style={loss_style:.12g} eval_style={eval_style:.12g} "
            f"abs_delta={abs_delta:.12g} tol={tol:.12g}"
        )
    return row


PRIMARY_METRICS = (
    "root_intent_mse",
    "root_pos_mse",
    "root_vel_mse",
    "root_disp_mse",
    "endpoint_ego_vel_mse",
    "endpoint_yaw_rate_mse",
    "yaw_traj_mse",
    "ego_vel_mse",
    "yaw_rate_mse",
    "contact_plan_mse",
    "contact_cycle_delta_mse",
    "contact_phase_consistency",
    "endpoint_support_start_match",
    "endpoint_support_end_match",
    "endpoint_support_both_match",
    "pose_root_rot6d_mse",
    "pose_nonroot_rot6d_mse",
    "bone_angvel_root_mse",
    "bone_angvel_nonroot_mse",
    "seam_c0_state_mse",
    "seam_c1_state_mse",
    "seam_c0_directlocal_mse",
    "seam_c1_directlocal_mse",
)


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["model_kind"]), str(row["partition"]))].append(row)
    out: List[Dict[str, Any]] = []
    for (variant, model_kind, partition), part_rows in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "variant": variant,
            "model_kind": model_kind,
            "partition": partition,
            "n": int(len(part_rows)),
        }
        for key in ("finite_fraction", "state_mse", *PRIMARY_METRICS, "contact_plan_bce", "predicted_contact_mse"):
            summary[key] = _safe_mean(part_rows, key)
        out.append(summary)
    return out


def _add_delta_columns(summary_rows: List[Dict[str, Any]]) -> None:
    by_name = {
        (str(r["variant"]), str(r["partition"])): r
        for r in summary_rows
        if str(r.get("partition")) == "test"
    }
    refs = (
        "ctx_only",
        "no_contact",
        "shifted_or_random_contact_control",
        "ctx_last_hold",
        "root_linear_to_goal_pose_hold",
    )
    for row in summary_rows:
        if str(row.get("partition")) != "test":
            continue
        for ref in refs:
            ref_row = by_name.get((ref, "test"))
            if ref_row is None or ref == row.get("variant"):
                continue
            for metric in (
                "root_intent_mse",
                "contact_plan_mse",
                "contact_cycle_delta_mse",
                "pose_nonroot_rot6d_mse",
                "seam_c1_directlocal_mse",
            ):
                row[f"delta_{metric}_vs_{ref}"] = float(row.get(metric, 0.0)) - float(ref_row.get(metric, 0.0))


def _make_baseline_rows(
    *,
    items: Sequence[BridgeItem],
    split: SplitDef,
    horizon: int,
    contact_support_threshold01: Tuple[float, float] = DEFAULT_CONTACT_LOSS_CONFIG.contact_support_threshold01,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, fn in (
        ("ctx_last_hold", _baseline_ctx_last_hold),
        ("root_linear_to_goal_pose_hold", _baseline_root_linear_pose_hold),
    ):
        for partition, idxs in (("train", split.train_idx), ("test", split.test_idx)):
            pred = fn(items, idxs, int(horizon))
            rows.extend(
                _metric_rows_for_predictions(
                    variant=name,
                    model_kind="dumb_baseline",
                    partition=partition,
                    items=items,
                    idxs=idxs,
                    pred=pred,
                    contact_support_threshold01=contact_support_threshold01,
                )
            )
    return rows


def _write_summary_md(
    path: Path,
    *,
    payload: Mapping[str, Any],
    variant_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Between GRU Bridge Probe",
        "",
        "Debug-only train-fit. Production trainer, checkpoints, and model classes are unchanged.",
        "",
        "## Tensor Contract",
        "",
        f"- ctx_state: `[B,{payload['flags']['context_len']},281] float32 {payload['flags']['device']}`",
        f"- goal_intent: `[B,{payload['schema']['goal_dim']}] float32 {payload['flags']['device']}`",
        f"- soft_contact: `[B,{payload['flags']['horizon']},2] float32 {payload['flags']['device']}`, min/max mapped to `[0,1]`",
        f"- state output: `[B,{payload['flags']['horizon']},281] float32 {payload['flags']['device']}`",
        f"- bone_angvel output: `[B,{payload['flags']['horizon']},138] float32 {payload['flags']['device']}`",
        "",
        "## Dataset",
        "",
        f"- matched windows: `{payload['dataset']['matched_window_count']}` from `{', '.join(payload['dataset']['matched_targets'])}`",
        f"- excluded unmatched diagnostic target: `{payload['dataset']['unmatched_target']}` (`{payload['dataset']['unmatched_window_count']}` windows)",
        f"- split: `{payload['split']['note']}`",
        "",
        "## Learned Variants (Test)",
        "",
        "| variant | n | root intent | contact mse | pose local | seam C1 local | contact phase |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variant_rows:
        if row.get("partition") != "test":
            continue
        lines.append(
            f"| {row['variant']} | {row['n']} | {_fmt(row['root_intent_mse'], 8)} | "
            f"{_fmt(row['contact_plan_mse'], 8)} | {_fmt(row['pose_nonroot_rot6d_mse'], 8)} | "
            f"{_fmt(row['seam_c1_directlocal_mse'], 8)} | {_fmt(row['contact_phase_consistency'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## Dumb Baselines (Test)",
            "",
            "| baseline | n | root intent | root pos | pose local | contact mse |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in baseline_rows:
        if row.get("partition") != "test":
            continue
        lines.append(
            f"| {row['variant']} | {row['n']} | {_fmt(row['root_intent_mse'], 8)} | "
            f"{_fmt(row['root_pos_mse'], 8)} | {_fmt(row['pose_nonroot_rot6d_mse'], 8)} | "
            f"{_fmt(row['contact_plan_mse'], 8)} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- summary json: `{payload['artifacts']['summary_json']}`",
            f"- variants csv: `{payload['artifacts']['variants_csv']}`",
            f"- baselines csv: `{payload['artifacts']['baselines_csv']}`",
            f"- grad usage csv: `{payload['artifacts']['grad_usage_csv']}`",
            f"- per-window csv: `{payload['artifacts']['per_window_csv']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug-only GRU bridge probe for between conditioning")
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--contact-embed-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--seed", type=int, default=20260606)
    p.add_argument("--train-fraction", type=float, default=0.60)
    p.add_argument("--block-gap", type=int, default=8)
    p.add_argument("--torch-num-threads", type=int, default=8)
    p.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    p.add_argument("--negative-contact", choices=("shifted", "random"), default="random")
    p.add_argument("--contact-shift", type=int, default=5)
    p.add_argument(
        "--goal-contact-mode",
        choices=GOAL_CONTACT_MODES,
        default="none",
        help="Debug-only goal repair: append target endpoint support to the low-dim goal when enabled.",
    )
    p.add_argument("--contact-step-weight", type=float, default=DEFAULT_CONTACT_LOSS_CONFIG.contact_step_weight)
    p.add_argument("--contact-predict-mse-weight", type=float, default=DEFAULT_CONTACT_LOSS_CONFIG.contact_predict_mse_weight)
    p.add_argument("--contact-predict-bce-weight", type=float, default=DEFAULT_CONTACT_LOSS_CONFIG.contact_predict_bce_weight)
    p.add_argument("--contact-state-weight", type=float, default=DEFAULT_CONTACT_LOSS_CONFIG.contact_state_weight)
    p.add_argument(
        "--contact-endpoint-support-weight",
        type=float,
        default=DEFAULT_CONTACT_LOSS_CONFIG.contact_endpoint_support_weight,
        help="Differentiable CE on predicted contact support labels at frame 0 and H-1.",
    )
    p.add_argument("--root-invariant-atol", type=float, default=1.0e-8)
    p.add_argument("--root-invariant-rtol", type=float, default=1.0e-5)
    return p.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    if name == "mps" and not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        raise RuntimeError("mps requested but unavailable")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.torch_num_threads))
    device = _resolve_device(str(args.device))
    clips, contact_scaler = _load_probe_clips(args.npz_root, args.z_features)
    contact_support_threshold01 = _contact_threshold01_from_raw(contact_scaler, CONTACT_LABEL_THRESHOLD)
    contact_loss = ContactLossConfig(
        contact_step_weight=float(args.contact_step_weight),
        contact_predict_mse_weight=float(args.contact_predict_mse_weight),
        contact_predict_bce_weight=float(args.contact_predict_bce_weight),
        contact_state_weight=float(args.contact_state_weight),
        contact_endpoint_support_weight=float(args.contact_endpoint_support_weight),
        contact_support_threshold01=contact_support_threshold01,
    )
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        stride=int(args.stride),
        goal_contact_mode=str(args.goal_contact_mode),
        goal_support_threshold01=contact_loss.contact_support_threshold01,
    )
    main_items = [item for item in all_items if item.clip in MATCHED_TARGETS]
    unmatched_items = [item for item in all_items if item.clip == UNMATCHED_TARGET]
    if not main_items:
        raise RuntimeError("no matched bridge items available")
    split = _build_split(main_items, train_fraction=float(args.train_fraction), block_gap=int(args.block_gap))
    if not split.train_idx or not split.test_idx:
        raise RuntimeError(f"empty split: train={len(split.train_idx)} test={len(split.test_idx)}")

    state_norm, aux_norm, goal_norm = _fit_normalizers(main_items, split.train_idx)
    train_batch = _batch_from_items(
        items=main_items,
        idxs=split.train_idx,
        state_norm=state_norm,
        aux_norm=aux_norm,
        goal_norm=goal_norm,
        device=device,
    )
    test_batch = _batch_from_items(
        items=main_items,
        idxs=split.test_idx,
        state_norm=state_norm,
        aux_norm=aux_norm,
        goal_norm=goal_norm,
        device=device,
    )
    goal_dim = int(train_batch["goal_n"].shape[-1])

    per_window_rows: List[Dict[str, Any]] = []
    baseline_rows = _make_baseline_rows(
        items=main_items,
        split=split,
        horizon=int(args.horizon),
        contact_support_threshold01=contact_loss.contact_support_threshold01,
    )
    per_window_rows.extend(baseline_rows)
    root_invariant_rows: List[Dict[str, Any]] = []
    grad_rows: List[Dict[str, Any]] = []
    variant_meta_rows: List[Dict[str, Any]] = []

    for spec in VARIANT_SPECS:
        model, final_terms, grad_usage = _train_variant(
            spec=spec,
            train_batch=train_batch,
            state_norm=state_norm,
            aux_norm=aux_norm,
            goal_dim=goal_dim,
            latent_dim=int(args.latent_dim),
            contact_embed_dim=int(args.contact_embed_dim),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            seed=int(args.seed) + len(grad_rows),
            device=device,
            negative_contact=str(args.negative_contact),
            contact_shift=int(args.contact_shift),
            contact_loss=contact_loss,
        )
        grad_rows.append(grad_usage)
        params = int(sum(p.numel() for p in model.parameters()))
        variant_meta_rows.append(
            {
                "variant": spec.name,
                "model_kind": "learned_gru_bridge",
                "use_goal": bool(spec.use_goal),
                "contact_mode": spec.contact_mode,
                "role": spec.role,
                "runtime_status": spec.runtime_status,
                "parameter_count": params,
                "epochs": int(args.epochs),
                "contact_step_weight": float(contact_loss.contact_step_weight),
                "contact_predict_mse_weight": float(contact_loss.contact_predict_mse_weight),
                "contact_predict_bce_weight": float(contact_loss.contact_predict_bce_weight),
                "contact_state_weight": float(contact_loss.contact_state_weight),
                "contact_endpoint_support_weight": float(contact_loss.contact_endpoint_support_weight),
                "contact_support_threshold01": [float(v) for v in contact_loss.contact_support_threshold01],
                **{f"final_train_{k}": v for k, v in final_terms.items()},
            }
        )
        for partition, batch, idxs in (
            ("train", train_batch, split.train_idx),
            ("test", test_batch, split.test_idx),
        ):
            pred = _predict_model(
                model=model,
                spec=spec,
                batch=batch,
                state_norm=state_norm,
                aux_norm=aux_norm,
                negative_contact=str(args.negative_contact),
                contact_shift=int(args.contact_shift),
            )
            rows = _metric_rows_for_predictions(
                variant=spec.name,
                model_kind="learned_gru_bridge",
                partition=partition,
                items=main_items,
                idxs=idxs,
                pred=pred,
                contact_support_threshold01=contact_loss.contact_support_threshold01,
            )
            root_invariant_rows.append(
                _root_pos_invariant_row(
                    variant=spec.name,
                    model_kind="learned_gru_bridge",
                    partition=partition,
                    batch=batch,
                    pred=pred,
                    metric_rows=rows,
                    atol=float(args.root_invariant_atol),
                    rtol=float(args.root_invariant_rtol),
                )
            )
            per_window_rows.extend(rows)

    summary_rows = _summary_rows(per_window_rows)
    _add_delta_columns(summary_rows)
    learned_summary = [r for r in summary_rows if r.get("model_kind") == "learned_gru_bridge"]
    baseline_summary = [r for r in summary_rows if r.get("model_kind") == "dumb_baseline"]
    meta_by_variant = {r["variant"]: r for r in variant_meta_rows}
    variant_csv_rows: List[Dict[str, Any]] = []
    for row in learned_summary:
        merged = dict(meta_by_variant.get(row["variant"], {}))
        merged.update(row)
        variant_csv_rows.append(merged)

    finite_rate = _safe_mean(per_window_rows, "finite_fraction")
    payload: Dict[str, Any] = {
        "task": "between_gru_bridge_probe",
        "status": "debug_train_fit",
        "flags": {
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "stride": int(args.stride),
            "epochs": int(args.epochs),
            "latent_dim": int(args.latent_dim),
            "contact_embed_dim": int(args.contact_embed_dim),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "device": str(device),
            "negative_contact": str(args.negative_contact),
            "contact_shift": int(args.contact_shift),
            "goal_contact_mode": str(args.goal_contact_mode),
            "contact_step_weight": float(contact_loss.contact_step_weight),
            "contact_predict_mse_weight": float(contact_loss.contact_predict_mse_weight),
            "contact_predict_bce_weight": float(contact_loss.contact_predict_bce_weight),
            "contact_state_weight": float(contact_loss.contact_state_weight),
            "contact_endpoint_support_weight": float(contact_loss.contact_endpoint_support_weight),
            "contact_support_threshold01": [float(v) for v in contact_loss.contact_support_threshold01],
            "root_invariant_atol": float(args.root_invariant_atol),
            "root_invariant_rtol": float(args.root_invariant_rtol),
        },
        "schema": {
            "ctx_state": {"shape": [len(split.train_idx), int(args.context_len), STATE_DIM], "dtype": "float32", "device": str(device)},
            "goal_dim": goal_dim,
            "goal_contact_mode": str(args.goal_contact_mode),
            "soft_contact": {"shape": [len(split.train_idx), int(args.horizon), 2], "dtype": "float32", "device": str(device), "range": [0.0, 1.0]},
            "state_output": {"shape": [len(split.train_idx), int(args.horizon), STATE_DIM], "dtype": "float32", "device": str(device)},
            "bone_angvel_output": {"shape": [len(split.train_idx), int(args.horizon), ANGVEL_DIM], "dtype": "float32", "device": str(device)},
        },
        "dataset": {
            "matched_targets": list(MATCHED_TARGETS),
            "matched_window_count": int(len(main_items)),
            "unmatched_target": UNMATCHED_TARGET,
            "unmatched_window_count": int(len(unmatched_items)),
            "clip_lengths": {name: int(clip.state281.shape[0]) for name, clip in clips.items()},
            "contact_scaler": contact_scaler.stats(),
        },
        "split": {
            "name": split.name,
            "kind": split.kind,
            "train_n": int(len(split.train_idx)),
            "test_n": int(len(split.test_idx)),
            "note": split.note,
        },
        "normalizers": {
            "state_constant_count": int(state_norm.constant_count),
            "aux_constant_count": int(aux_norm.constant_count),
            "goal_constant_count": int(goal_norm.constant_count),
        },
        "variant_summaries": variant_csv_rows,
        "baseline_summaries": baseline_summary,
        "grad_usage": grad_rows,
        "root_invariant": root_invariant_rows,
        "finite_fraction_mean": finite_rate,
        "artifacts": {
            "summary_json": str(args.out_dir / "summary.json"),
            "summary_md": str(args.out_dir / "summary.md"),
            "variants_csv": str(args.out_dir / "variants.csv"),
            "baselines_csv": str(args.out_dir / "baselines.csv"),
            "grad_usage_csv": str(args.out_dir / "grad_usage.csv"),
            "root_invariant_csv": str(args.out_dir / "root_invariant.csv"),
            "per_window_csv": str(args.out_dir / "per_window.csv"),
        },
    }

    _write_csv(args.out_dir / "per_window.csv", per_window_rows)
    _write_csv(args.out_dir / "variants.csv", variant_csv_rows)
    _write_csv(args.out_dir / "baselines.csv", baseline_summary)
    _write_csv(args.out_dir / "grad_usage.csv", grad_rows)
    _write_csv(args.out_dir / "root_invariant.csv", root_invariant_rows)
    _write_json(args.out_dir / "summary.json", payload)
    _write_summary_md(
        args.out_dir / "summary.md",
        payload=payload,
        variant_rows=variant_csv_rows,
        baseline_rows=baseline_summary,
    )
    print(f"[OK] wrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
