#!/usr/bin/env python3
"""Debug-only anchor-between reconstructed acceptance replay.

Builds no-learned-residual anchor windows:
  state281 pose/contact = context-last hold by default
  root                  = linear-to-goal through state ego velocity
  yaw_rate              = linear start-to-end yaw-rate
  bone_angvel aux       = context-last hold

The replay uses the existing reconstructed-domain acceptance helpers. Support-side
is reported as read-only audit and is not part of the hard go/no-go gate here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    CONTEXT_LEN_C,
    EGO_VEL_SLICE,
    POSE_SLICE,
    STATE_DIM,
    WALK_F,
    YAW_RATE_SLICE,
)
from tools.run_action_handoff_command_demotion_replay import (  # noqa: E402
    _command_compatibility,
    _support_side_core,
)
from tools.run_action_handoff_between_anchor_residual_probe import (  # noqa: E402
    DEFAULT_CONTACT_LOSS_CONFIG as RESIDUAL_DEFAULT_CONTACT_LOSS_CONFIG,
    VARIANT_SPECS as RESIDUAL_VARIANT_SPECS,
    ContactLossConfig as ResidualContactLossConfig,
    _batch_from_items as _residual_batch_from_items,
    _build_items as _residual_build_items,
    _build_split as _residual_build_split,
    _contact_threshold01_from_raw as _residual_contact_threshold01_from_raw,
    _fit_normalizers as _residual_fit_normalizers,
    _load_probe_clips as _residual_load_probe_clips,
    _predict_anchor_model as _residual_predict_anchor_model,
    _resolve_device as _residual_resolve_device,
    _train_variant as _residual_train_variant,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    ANGVEL_DIM,
    CONTACT_THRESHOLD,
    DEFAULT_NPZ_ROOT,
    DEFAULT_Z_FEATURES,
    _dump_json,
    _dump_md,
    _fmt,
    _load_clips,
    _load_skeleton_meta,
)
from tools.run_action_handoff_oracle_schedule_trajectory_decoder_smoke import (  # noqa: E402
    DEFAULT_HORIZON,
    _build_items,
    _calibrate_reconstructed_baseline_bands,
    _calibrate_reconstructed_support_side_bands,
    _evaluate_seq_common,
    _reconstructed_gt_seq,
    _world_root_vel_from_ego,
)
from tools.run_action_handoff_support_contract_tightening_probe import _support_contract  # noqa: E402
from tools.run_action_handoff_support_schedule_predictive_baseline import MATCHED_TARGETS  # noqa: E402


DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_anchor_between_acceptance_replay_20260606")
FPS = 60.0
EPS = 1.0e-8
HARD_GATE_FAMILIES = ("regime_reached", "rate_budget", "support_honesty", "pose_continuity", "endpoint_bridgeability")
PREDICTED_CONTACT_VARIANT = "predicted_contact"


def _jsonify(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonify(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonify(v.tolist())
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    return v


def _safe_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals: List[float] = []
    for row in rows:
        try:
            v = float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def _safe_p95(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals: List[float] = []
    for row in rows:
        try:
            v = float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.percentile(np.asarray(vals, dtype=np.float64), 95.0)) if vals else 0.0


def _optional_mean(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    return _safe_mean(rows, key) if any(key in row for row in rows) else ""


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    if da.size == 0 or db.size == 0:
        return 0.0
    d = da - db
    return float(np.mean(d * d))


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


def _anchor_state_aux(
    item: Any,
    *,
    context_len: int,
    contact_mode: str,
    contact_override_raw: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    h = int(np.asarray(item.seq["state281"]).shape[0])
    ctx_state = np.asarray(item.ctx, dtype=np.float32).reshape(int(context_len), STATE_DIM)
    ctx_aux = _context_window(
        np.asarray(item.seq.get("_clip_bone_angvel"), dtype=np.float32),
        int(item.start),
        int(context_len),
        wrap=(str(item.clip) == WALK_F),
    )
    true_state = np.asarray(item.seq["state281"], dtype=np.float32).reshape(h, STATE_DIM)
    true_root = np.asarray(item.seq["root_pos"], dtype=np.float32).reshape(h, 3)
    cond_dir = np.asarray(item.seq["cond_dir"], dtype=np.float32).reshape(h, 2)

    state = np.repeat(ctx_state[-1:].copy(), h, axis=0).astype(np.float32)
    aux = np.repeat(ctx_aux[-1:].copy(), h, axis=0).astype(np.float32)

    disp_xy = true_root[-1, :2] - true_root[0, :2]
    world_vel = disp_xy * float(FPS) / float(max(1, h - 1))
    norm = np.maximum(np.linalg.norm(cond_dir, axis=1, keepdims=True), EPS)
    fwd = cond_dir / norm
    lat = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
    ego = np.stack([np.sum(world_vel.reshape(1, 2) * fwd, axis=1), np.sum(world_vel.reshape(1, 2) * lat, axis=1)], axis=1)
    state[:, EGO_VEL_SLICE] = ego.astype(np.float32)
    state[:, YAW_RATE_SLICE] = np.linspace(
        float(ctx_state[-1, YAW_RATE_SLICE][0]),
        float(true_state[-1, YAW_RATE_SLICE][0]),
        h,
        dtype=np.float32,
    ).reshape(h, 1)
    if contact_override_raw is not None:
        override = np.asarray(contact_override_raw, dtype=np.float32).reshape(h, 2)
        state[:, CONTACT_SLICE] = override
    elif str(contact_mode) == "oracle":
        state[:, CONTACT_SLICE] = true_state[:, CONTACT_SLICE]

    info = {
        "anchor_pose": "ctx_last_hold",
        "anchor_contact": (
            "predicted_contact_plan_gru_only"
            if contact_override_raw is not None
            else ("oracle_contact_passthrough" if str(contact_mode) == "oracle" else "ctx_last_hold")
        ),
        "anchor_root": "linear_to_goal_via_constant_world_velocity_projected_to_ego",
        "anchor_yaw": "linear_ctx_last_to_target_endpoint_yaw_rate",
        "anchor_aux": "ctx_last_hold_bone_angvel",
    }
    return state.astype(np.float32), aux.astype(np.float32), info


def _seq_from_anchor(item: Any, state: np.ndarray, aux: np.ndarray) -> Dict[str, np.ndarray]:
    s = np.asarray(state, dtype=np.float32).reshape(-1, STATE_DIM)
    cond_dir = np.asarray(item.seq["cond_dir"], dtype=np.float32).reshape(s.shape[0], 2)
    root_vel = _world_root_vel_from_ego(s[:, EGO_VEL_SLICE], cond_dir, command_align_root_vel=False)
    root = np.zeros((s.shape[0], 3), dtype=np.float32)
    root[0] = np.asarray(item.seq["root_pos"], dtype=np.float32).reshape(-1, 3)[0]
    if s.shape[0] > 1:
        root[1:, :2] = root[0:1, :2] + np.cumsum(root_vel[:-1] / float(FPS), axis=0)
        root[1:, 2] = root[0, 2]
    return {
        "rot6d": s[:, POSE_SLICE].astype(np.float32, copy=False),
        "root_pos": root,
        "root_vel": root_vel.astype(np.float32, copy=False),
        "bone_angvel": np.asarray(aux, dtype=np.float32).reshape(s.shape[0], ANGVEL_DIM),
        "cond_dir": cond_dir,
        "contact": s[:, CONTACT_SLICE].astype(np.float32, copy=False),
        "yaw_rate": s[:, YAW_RATE_SLICE].reshape(-1).astype(np.float32, copy=False),
    }


def _attach_clip_aux(items: Sequence[Any], clips: Mapping[str, Any]) -> None:
    for item in items:
        item.seq["_clip_bone_angvel"] = np.asarray(clips[str(item.clip)].bone_angvel, dtype=np.float32)


def _hard_gate(row: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    failed = [key for key in HARD_GATE_FAMILIES if not bool(row.get(key, False))]
    return bool(not failed), failed


def _item_key(item: Any) -> Tuple[str, int, int]:
    return (str(item.clip), int(item.start), int(item.end))


def _endpoint_transition(item: Any) -> Tuple[str, str, str, str]:
    labels = [str(x) for x in item.support_contract.get("normalized_label_sequence", [])]
    start = labels[0] if labels else "empty"
    end = labels[-1] if labels else "empty"
    transition = f"{start}->{end}"
    symmetry = "symmetric" if start == end else "transition"
    return start, end, transition, symmetry


def _limit_windows_per_target(items: Sequence[Any], max_windows_per_target: int) -> List[Any]:
    if int(max_windows_per_target) <= 0:
        return list(items)
    kept: List[Any] = []
    counts: Dict[str, int] = defaultdict(int)
    for item in items:
        clip = str(item.clip)
        if counts[clip] >= int(max_windows_per_target):
            continue
        kept.append(item)
        counts[clip] += 1
    return kept


def _split_partition_by_key(items: Sequence[Any], *, train_fraction: float, block_gap: int) -> Dict[Tuple[str, int, int], str]:
    split = _residual_build_split(items, train_fraction=float(train_fraction), block_gap=int(block_gap))
    out: Dict[Tuple[str, int, int], str] = {}
    for idx in split.train_idx:
        out[_item_key(items[int(idx)])] = "train"
    for idx in split.test_idx:
        out[_item_key(items[int(idx)])] = "test"
    for idx, item in enumerate(items):
        out.setdefault(_item_key(item), "gap")
    return out


def _regime_balanced_partition_by_key(items: Sequence[Any], *, train_fraction: float) -> Dict[Tuple[str, int, int], str]:
    by_stratum: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
    for item in items:
        _start, _end, transition, _symmetry = _endpoint_transition(item)
        by_stratum[(str(item.clip), transition)].append(item)

    out: Dict[Tuple[str, int, int], str] = {}
    test_fraction = max(0.0, min(1.0, 1.0 - float(train_fraction)))
    for _stratum, vals in sorted(by_stratum.items()):
        vals = sorted(vals, key=lambda x: int(x.start))
        n = len(vals)
        if n <= 1:
            for item in vals:
                out[_item_key(item)] = "train"
            continue
        test_n = int(round(test_fraction * n))
        test_n = max(1, min(n - 1, test_n))
        positions = np.linspace(0, n - 1, test_n, dtype=np.float64)
        test_pos = {int(round(float(pos))) for pos in positions}
        cursor = 0
        while len(test_pos) < test_n and cursor < n:
            test_pos.add(cursor)
            cursor += 1
        for pos, item in enumerate(vals):
            out[_item_key(item)] = "test" if pos in test_pos else "train"
    return out


def _regime_block_gap_partition_by_key(
    items: Sequence[Any],
    *,
    train_fraction: float,
    block_gap: int,
    horizon: int,
) -> Dict[Tuple[str, int, int], str]:
    by_clip: Dict[str, List[Any]] = defaultdict(list)
    for item in items:
        by_clip[str(item.clip)].append(item)

    gap = max(int(block_gap), int(horizon))
    out: Dict[Tuple[str, int, int], str] = {}
    for _clip, vals in sorted(by_clip.items()):
        vals = sorted(vals, key=lambda x: int(x.start))
        n = len(vals)
        if n < 3:
            for item in vals:
                out[_item_key(item)] = "train"
            continue
        train_n = int(math.floor(max(0.0, min(1.0, float(train_fraction))) * n))
        train_n = max(1, min(n - 2, train_n))
        if train_n + gap >= n:
            train_n = max(1, n - gap - 1)
        if train_n + gap >= n:
            gap_n = max(0, n - train_n - 1)
        else:
            gap_n = gap
        test_start = min(n - 1, train_n + gap_n)
        for pos, item in enumerate(vals):
            if pos < train_n:
                out[_item_key(item)] = "train"
            elif pos < test_start:
                out[_item_key(item)] = "gap"
            else:
                out[_item_key(item)] = "test"
    return out


def _horizon_dedup_balanced_partition_by_key(
    items: Sequence[Any],
    *,
    train_fraction: float,
    horizon: int,
) -> Dict[Tuple[str, int, int], str]:
    kept_by_clip: Dict[str, List[Any]] = defaultdict(list)
    for clip in sorted({str(item.clip) for item in items}):
        last = -10**9
        for item in sorted([x for x in items if str(x.clip) == clip], key=lambda x: int(x.start)):
            if int(item.start) - last >= int(horizon):
                kept_by_clip[clip].append(item)
                last = int(item.start)

    test_keys = set()
    test_fraction = max(0.0, min(1.0, 1.0 - float(train_fraction)))
    for clip, vals in sorted(kept_by_clip.items()):
        n = len(vals)
        if n <= 1:
            continue
        test_n = int(round(test_fraction * n))
        test_n = max(1, min(n - 1, test_n))
        by_sym: Dict[str, List[int]] = defaultdict(list)
        for pos, item in enumerate(vals):
            _start, _end, _transition, symmetry = _endpoint_transition(item)
            by_sym[symmetry].append(pos)
        chosen: List[int] = []
        for symmetry in ("transition", "symmetric"):
            if len(chosen) >= test_n:
                break
            positions = by_sym.get(symmetry, [])
            if positions:
                chosen.append(positions[len(positions) // 2])
        cursor = 0
        while len(chosen) < test_n and cursor < n:
            if cursor not in chosen:
                chosen.append(cursor)
            cursor += 1
        for pos in chosen[:test_n]:
            test_keys.add(_item_key(vals[int(pos)]))

    kept_keys = {_item_key(item) for vals in kept_by_clip.values() for item in vals}
    out: Dict[Tuple[str, int, int], str] = {}
    for item in items:
        key = _item_key(item)
        if key in test_keys:
            out[key] = "test"
        elif key in kept_keys:
            out[key] = "train"
        else:
            out[key] = "gap"
    return out


def _partition_by_key(
    items: Sequence[Any],
    *,
    split_mode: str,
    train_fraction: float,
    block_gap: int,
    horizon: int,
) -> Dict[Tuple[str, int, int], str]:
    if str(split_mode) == "contiguous":
        return _split_partition_by_key(items, train_fraction=float(train_fraction), block_gap=int(block_gap))
    if str(split_mode) == "regime_balanced":
        return _regime_balanced_partition_by_key(items, train_fraction=float(train_fraction))
    if str(split_mode) == "regime_block_gap":
        return _regime_block_gap_partition_by_key(
            items,
            train_fraction=float(train_fraction),
            block_gap=int(block_gap),
            horizon=int(horizon),
        )
    if str(split_mode) == "horizon_dedup_balanced":
        return _horizon_dedup_balanced_partition_by_key(
            items,
            train_fraction=float(train_fraction),
            horizon=int(horizon),
        )
    raise ValueError(f"unknown split_mode={split_mode!r}")


def _contact01_to_raw(contact01: np.ndarray, scaler: Any) -> np.ndarray:
    c = np.clip(np.asarray(contact01, dtype=np.float32), 0.0, 1.0)
    min_v = np.asarray(scaler.min_v, dtype=np.float32).reshape(1, 1, 2)
    max_v = np.asarray(scaler.max_v, dtype=np.float32).reshape(1, 1, 2)
    return (c * np.maximum(max_v - min_v, 1.0e-6) + min_v).astype(np.float32, copy=False)


def _train_predicted_contact_by_key(
    args: argparse.Namespace,
    eval_items: Sequence[Any],
    partition_by_key: Mapping[Tuple[str, int, int], str],
    *,
    seed: int,
) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    torch.manual_seed(int(seed))
    device = _residual_resolve_device(str(args.device))
    clips, scaler = _residual_load_probe_clips(Path(args.npz_root), Path(args.z_features))
    contact_support_threshold01 = _residual_contact_threshold01_from_raw(scaler, CONTACT_THRESHOLD)
    contact_loss = ResidualContactLossConfig(
        contact_step_weight=float(args.contact_step_weight),
        contact_predict_mse_weight=float(args.contact_predict_mse_weight),
        contact_predict_bce_weight=float(args.contact_predict_bce_weight),
        contact_state_weight=float(args.contact_state_weight),
        contact_endpoint_support_weight=float(args.contact_endpoint_support_weight),
        contact_support_threshold01=contact_support_threshold01,
    )
    residual_items = _residual_build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        stride=int(args.stride),
    )
    residual_items = [item for item in residual_items if item.clip in MATCHED_TARGETS]
    residual_items = _limit_windows_per_target(residual_items, int(args.max_windows_per_target))
    eval_keys = {_item_key(item) for item in eval_items}
    residual_keys = {_item_key(item) for item in residual_items}
    missing = sorted(eval_keys - residual_keys)
    if missing:
        raise RuntimeError(f"predicted contact item mismatch; missing keys: {missing[:5]}")

    train_idx = tuple(i for i, item in enumerate(residual_items) if str(partition_by_key.get(_item_key(item), "")) == "train")
    test_idx = tuple(i for i, item in enumerate(residual_items) if str(partition_by_key.get(_item_key(item), "")) == "test")
    gap_idx = tuple(i for i, item in enumerate(residual_items) if str(partition_by_key.get(_item_key(item), "")) == "gap")
    if not train_idx or not test_idx:
        raise RuntimeError(f"empty predicted-contact split: train={len(train_idx)} test={len(test_idx)}")
    state_norm, aux_norm, goal_norm = _residual_fit_normalizers(residual_items, train_idx)
    train_batch = _residual_batch_from_items(
        items=residual_items,
        idxs=train_idx,
        state_norm=state_norm,
        aux_norm=aux_norm,
        goal_norm=goal_norm,
        device=device,
    )
    goal_dim = int(train_batch["goal_n"].shape[-1])
    spec_by_name = {spec.name: spec for spec in RESIDUAL_VARIANT_SPECS}
    spec = spec_by_name[PREDICTED_CONTACT_VARIANT]
    spec_i = list(RESIDUAL_VARIANT_SPECS).index(spec)
    model, final_terms, _grad = _residual_train_variant(
        spec=spec,
        train_batch=train_batch,
        state_norm=state_norm,
        aux_norm=aux_norm,
        goal_dim=goal_dim,
        latent_dim=int(args.latent_dim),
        contact_embed_dim=int(args.contact_embed_dim),
        frame_hidden_dim=int(args.frame_hidden_dim),
        dropout=float(args.dropout),
        residual_l2_weight=float(args.residual_l2_weight),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(seed) + spec_i,
        device=device,
        negative_contact="random",
        contact_loss=contact_loss,
    )
    all_idx = tuple(range(len(residual_items)))
    all_batch = _residual_batch_from_items(
        items=residual_items,
        idxs=all_idx,
        state_norm=state_norm,
        aux_norm=aux_norm,
        goal_norm=goal_norm,
        device=device,
    )
    pred = _residual_predict_anchor_model(
        model=model,
        spec=spec,
        batch=all_batch,
        state_norm=state_norm,
        aux_norm=aux_norm,
        negative_contact="random",
    )
    pred01_all = np.asarray(pred["predicted_contact"], dtype=np.float32).reshape(len(all_idx), int(args.horizon), 2)
    pred_raw_all = _contact01_to_raw(pred01_all, scaler)
    out: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    train_set = {int(i) for i in train_idx}
    test_set = {int(i) for i in test_idx}
    gap_set = {int(i) for i in gap_idx}
    for local_i, item_i in enumerate(all_idx):
        item = residual_items[int(item_i)]
        true01 = np.asarray(item.seq["contact"], dtype=np.float32).reshape(int(args.horizon), 2)
        true_raw = _contact01_to_raw(true01.reshape(1, int(args.horizon), 2), scaler)[0]
        partition = "train" if int(item_i) in train_set else ("test" if int(item_i) in test_set else ("gap" if int(item_i) in gap_set else "unused"))
        out[_item_key(item)] = {
            "contact_raw": pred_raw_all[local_i],
            "contact01": pred01_all[local_i],
            "true_contact01": true01,
            "true_contact_raw_from_scaler": true_raw,
            "partition": partition,
            "predicted_contact_mse_01": _mse(pred01_all[local_i], true01),
            "predicted_contact_mse_raw": _mse(pred_raw_all[local_i], true_raw),
            "contact_seed": int(seed),
            "train_split_n": int(len(train_idx)),
            "test_split_n": int(len(test_idx)),
            "gap_split_n": int(len(gap_idx)),
            "final_train_contact_plan_mse": float(final_terms.get("contact_plan_mse", 0.0)),
            "final_train_contact_predict_mse": float(final_terms.get("contact_predict_mse", 0.0)),
            "final_train_contact_predict_bce": float(final_terms.get("contact_predict_bce", 0.0)),
            "final_train_contact_endpoint_support_ce": float(final_terms.get("contact_endpoint_support_ce", 0.0)),
            "contact_step_weight": float(contact_loss.contact_step_weight),
            "contact_predict_mse_weight": float(contact_loss.contact_predict_mse_weight),
            "contact_predict_bce_weight": float(contact_loss.contact_predict_bce_weight),
            "contact_state_weight": float(contact_loss.contact_state_weight),
            "contact_endpoint_support_weight": float(contact_loss.contact_endpoint_support_weight),
            "contact_support_threshold01": [float(v) for v in contact_loss.contact_support_threshold01],
        }
    return out


def _flatten_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = row.get("metrics", {}) or {}
    out = dict(row)
    out.pop("metrics", None)
    out.pop("thresholds", None)
    out.pop("endpoint_details", None)
    out.pop("support_side_failures", None)
    out.pop("support_side_core_failures", None)
    out.pop("support_side_full_failures", None)
    for key, val in metrics.items():
        out[str(key)] = val
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list, tuple)) else row.get(key) for key in keys})


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    failed = Counter()
    for row in rows:
        for fam in str(row.get("hard_failed_family") or "").split(","):
            if fam:
                failed[fam] += 1

    def rate(key: str) -> float:
        return float(np.mean([bool(r.get(key, False)) for r in rows]))

    return {
        "n": int(len(rows)),
        "hard_acceptance_pass_rate": rate("hard_acceptance_pass"),
        "regime_reached_pass_rate": rate("regime_reached"),
        "rate_budget_pass_rate": rate("rate_budget"),
        "support_honesty_pass_rate": rate("support_honesty"),
        "pose_continuity_pass_rate": rate("pose_continuity"),
        "endpoint_bridgeability_pass_rate": rate("endpoint_bridgeability"),
        "command_compatibility_pass_rate": rate("command_compatibility"),
        "support_side_core_pass_rate": rate("support_side_core"),
        "support_side_full_legacy_pass_rate": rate("support_side_full_legacy"),
        "acceptance_proxy_with_support_side_pass_rate": rate("acceptance_proxy_pass"),
        "yaw_traj_mse_mean": _safe_mean(rows, "yaw_traj_mse"),
        "root_pos_mse_mean": _safe_mean(rows, "root_pos_mse"),
        "pose_rot6d_mse_mean": _safe_mean(rows, "pose_rot6d_mse"),
        "bone_angvel_mse_mean": _safe_mean(rows, "bone_angvel_mse"),
        "pose_step_l2_p95_mean": _safe_mean(rows, "pose_step_l2_p95"),
        "rootvel_step_l2_p95_mean": _safe_mean(rows, "rootvel_step_l2_p95"),
        "yaw_rate_step_abs_p95_mean": _safe_mean(rows, "yaw_rate_step_abs_p95"),
        "foot_slip_p95_to_band_ratio_mean": _safe_mean(rows, "foot_slip_p95_to_band_ratio"),
        "root_path_error_p95_m_mean": _safe_mean(rows, "root_path_error_p95_m"),
        "root_path_error_p95_m_p95": _safe_p95(rows, "root_path_error_p95_m"),
        "predicted_contact_mse_01_mean": _optional_mean(rows, "predicted_contact_mse_01"),
        "predicted_contact_mse_raw_mean": _optional_mean(rows, "predicted_contact_mse_raw"),
        "failed_family_counts": dict(failed),
    }


def _summaries_by(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    return {value: _summarize([r for r in rows if str(r.get(key)) == value]) for value in sorted(set(str(r.get(key)) for r in rows))}


def _split_leakage_summary(rows: Sequence[Mapping[str, Any]], *, horizon: int) -> Dict[str, Any]:
    starts_by_part_clip: Dict[Tuple[str, str], set] = defaultdict(set)
    strata_by_part: Dict[str, set] = defaultdict(set)
    for row in rows:
        clip = str(row.get("clip") or row.get("target") or "")
        if not clip:
            continue
        try:
            start = int(row.get("start"))
        except (TypeError, ValueError):
            continue
        part = str(row.get("split_partition") or row.get("partition") or "")
        starts_by_part_clip[(part, clip)].add(start)
        if part:
            strata_by_part[part].add(str(row.get("split_stratum") or row.get("oracle_endpoint_transition") or ""))

    total_test = 0
    leaky = 0
    dist_hist = Counter()
    effective_by_clip: Dict[str, Dict[str, int]] = {}
    clips = sorted({clip for part, clip in starts_by_part_clip if part == "test"})
    for clip in clips:
        train_starts = sorted(starts_by_part_clip.get(("train", clip), set()))
        test_starts = sorted(starts_by_part_clip.get(("test", clip), set()))
        total_test += len(test_starts)
        kept: List[int] = []
        last = -10**9
        for start in test_starts:
            if start - last >= int(horizon):
                kept.append(start)
                last = start
            d = min((abs(start - train_start) for train_start in train_starts), default=10**9)
            dist_hist[int(d)] += 1
            if d < int(horizon):
                leaky += 1
        effective_by_clip[clip] = {
            "raw_test_n": int(len(test_starts)),
            "effective_independent_test_n": int(len(kept)),
        }
    return {
        "horizon": int(horizon),
        "test_window_n": int(total_test),
        "test_train_neighbor_within_horizon_n": int(leaky),
        "test_train_neighbor_within_horizon_rate": float(leaky / total_test) if total_test else 0.0,
        "nearest_train_distance_hist": {str(k): int(v) for k, v in sorted(dist_hist.items())},
        "effective_independent_test_by_clip": effective_by_clip,
        "effective_independent_test_n": int(sum(v["effective_independent_test_n"] for v in effective_by_clip.values())),
        "test_stratum_count": int(len(strata_by_part.get("test", set()))),
    }


def _write_md(path: Path, payload: Mapping[str, Any]) -> None:
    verdict = payload["verdict"]
    lines = [
        "# Anchor-Between Acceptance Replay",
        "",
        "Debug-only replay. Learned residual is disabled; support_side is read-only audit.",
        "",
        "## Verdict",
        "",
        f"- decision: `{verdict['decision']}`",
        f"- hard pass rate: `{_fmt(verdict['hard_acceptance_pass_rate'])}`",
        f"- support-side core audit pass rate: `{_fmt(verdict['support_side_core_pass_rate'])}`",
        f"- support-side full legacy audit pass rate: `{_fmt(verdict['support_side_full_legacy_pass_rate'])}`",
        "",
            "## Summary By Partition",
        "",
        "| partition | n | hard pass | endpoint | regime | contact mse raw | contact mse 01 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for part, rec in sorted(payload.get("summary_by_partition", {}).items()):
        lines.append(
            f"| {part} | {int(rec.get('n', 0))} | {_fmt(rec.get('hard_acceptance_pass_rate'))} | "
            f"{_fmt(rec.get('endpoint_bridgeability_pass_rate'))} | {_fmt(rec.get('regime_reached_pass_rate'))} | "
            f"{_fmt(rec.get('predicted_contact_mse_raw_mean'), 8)} | {_fmt(rec.get('predicted_contact_mse_01_mean'), 8)} |"
        )
    leak = payload.get("split_leakage", {}) or {}
    lines.extend(
        [
            "",
            "## Split Leakage",
            "",
            f"- test windows with train neighbor inside horizon: `{leak.get('test_train_neighbor_within_horizon_n', 0)}/{leak.get('test_window_n', 0)}`",
            f"- effective independent test windows: `{leak.get('effective_independent_test_n', 0)}`",
            "",
            "## Summary By Symmetry",
            "",
            "| symmetry | n | hard pass | endpoint | contact mse 01 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for symmetry, rec in sorted(payload.get("summary_by_transition_symmetry", {}).items()):
        lines.append(
            f"| {symmetry} | {int(rec.get('n', 0))} | {_fmt(rec.get('hard_acceptance_pass_rate'))} | "
            f"{_fmt(rec.get('endpoint_bridgeability_pass_rate'))} | {_fmt(rec.get('predicted_contact_mse_01_mean'), 8)} |"
        )
    lines.extend(
        [
            "",
        "## Summary By Target",
        "",
        "| target | n | hard pass | regime | rate | support honesty | pose continuity | endpoint | support core audit | root mse | yaw mse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target, rec in sorted(payload["summary_by_target"].items()):
        lines.append(
            f"| {target} | {int(rec.get('n', 0))} | {_fmt(rec.get('hard_acceptance_pass_rate'))} | "
            f"{_fmt(rec.get('regime_reached_pass_rate'))} | {_fmt(rec.get('rate_budget_pass_rate'))} | "
            f"{_fmt(rec.get('support_honesty_pass_rate'))} | {_fmt(rec.get('pose_continuity_pass_rate'))} | "
            f"{_fmt(rec.get('endpoint_bridgeability_pass_rate'))} | {_fmt(rec.get('support_side_core_pass_rate'))} | "
            f"{_fmt(rec.get('root_pos_mse_mean'), 8)} | {_fmt(rec.get('yaw_traj_mse_mean'), 8)} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            f"- summary json: `{payload['artifacts']['summary_json']}`",
            f"- rows csv: `{payload['artifacts']['rows_csv']}`",
            f"- summary md: `{payload['artifacts']['summary_md']}`",
        ]
    )
    _dump_md(path, lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(int(args.torch_num_threads))
    clips = _load_clips(Path(args.npz_root), Path(args.z_features))
    skeleton = _load_skeleton_meta(Path(args.npz_root), WALK_F)
    items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    _attach_clip_aux(items, clips)
    main_items = _limit_windows_per_target([item for item in items if item.clip in MATCHED_TARGETS], int(args.max_windows_per_target))
    if not main_items:
        raise RuntimeError("no matched target windows")
    split_partition = _partition_by_key(
        main_items,
        split_mode=str(args.split_mode),
        train_fraction=float(args.train_fraction),
        block_gap=int(args.block_gap),
        horizon=int(args.horizon),
    )

    baseline_bands = _calibrate_reconstructed_baseline_bands(
        items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
    )
    support_bands = _calibrate_reconstructed_support_side_bands(
        items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
    )
    # Reuse the command band calibration from the demotion replay. It is a read-only
    # command audit here; hard gate is continuity/arrival/seam.
    from tools.run_action_handoff_command_demotion_replay import _calibrate_command_bands  # noqa: WPS433

    command_bands = _calibrate_command_bands(clips, horizon=int(args.horizon), quantile=float(args.command_quantile))

    contact_seed_values: List[Optional[int]]
    predicted_contact_by_seed: Dict[Optional[int], Dict[Tuple[str, int, int], Dict[str, Any]]] = {}
    if str(args.contact_mode) == "predicted_probe":
        base_seed = int(args.contact_seed_base) if args.contact_seed_base is not None else int(args.seed)
        contact_seed_values = [base_seed + i for i in range(max(1, int(args.contact_seeds)))]
        for seed in contact_seed_values:
            predicted_contact_by_seed[int(seed)] = _train_predicted_contact_by_key(
                args,
                main_items,
                split_partition,
                seed=int(seed),
            )
    else:
        contact_seed_values = [None]

    rows: List[Dict[str, Any]] = []
    for contact_seed in contact_seed_values:
        predicted_contact_by_key = predicted_contact_by_seed.get(contact_seed, {})
        for item in main_items:
            predicted_info = predicted_contact_by_key.get(_item_key(item), {})
            contact_override = predicted_info.get("contact_raw")
            state, aux, anchor_info = _anchor_state_aux(
                item,
                context_len=int(args.context_len),
                contact_mode=str(args.contact_mode),
                contact_override_raw=contact_override if contact_override is not None else None,
            )
            seq = _seq_from_anchor(item, state, aux)
            row = _evaluate_seq_common(
                variant="anchor_between_no_residual",
                split="anchor_between_acceptance_replay",
                split_kind="debug_anchor_no_residual",
                partition="all",
                item=item,
                seq=seq,
                baseline_bands=baseline_bands,
                support_bands=support_bands,
                skeleton=skeleton,
                min_run_frames=int(args.min_run_frames),
                endpoint_note=f"anchor contact mode={args.contact_mode}; endpoint proxy uses reconstructed helper first/last support",
                oracle_contact_passthrough=False,
                command_align_root_vel=False,
                calibration_domain="reconstructed_state281",
            )
            labels = [str(x) for x in _support_contract(seq["contact"], min_run_frames=int(args.min_run_frames))["normalized_label_sequence"]]
            side = _support_side_core(seq, labels, skeleton, support_bands[item.clip]["feature_bands"])
            cmd = _command_compatibility(seq, command_bands.get(item.clip))
            true_seq = _reconstructed_gt_seq(item, oracle_contact_passthrough=True, command_align_root_vel=False)
            true_state = np.asarray(item.seq["state281"], dtype=np.float32).reshape(int(args.horizon), STATE_DIM)
            true_aux = np.asarray(item.seq["bone_angvel"], dtype=np.float32).reshape(int(args.horizon), ANGVEL_DIM)
            root_err = np.linalg.norm(np.asarray(seq["root_pos"], dtype=np.float64) - np.asarray(true_seq["root_pos"], dtype=np.float64), axis=1)
            hard_pass, hard_failed = _hard_gate(row)
            row.update(side)
            row.update(cmd)
            oracle_start, oracle_end, transition, symmetry = _endpoint_transition(item)
            row["case"] = "anchor_between:no_residual"
            row["contact_mode"] = str(args.contact_mode)
            row["contact_seed"] = "" if contact_seed is None else int(contact_seed)
            row["oracle_endpoint_start"] = oracle_start
            row["oracle_endpoint_end"] = oracle_end
            row["oracle_endpoint_transition"] = transition
            row["oracle_transition_symmetry"] = symmetry
            row["split_stratum"] = f"{item.clip}|{transition}"
            row["split_partition"] = str(predicted_info.get("partition") or split_partition.get(_item_key(item), "all"))
            row["hard_acceptance_pass"] = bool(hard_pass)
            row["hard_failed_family"] = ",".join(hard_failed)
            row["support_side_read_only_audit"] = True
            row["support_side_not_in_hard_gate"] = True
            row["anchor_info"] = anchor_info
            row["root_path_error_p95_m"] = float(np.percentile(root_err, 95.0)) if root_err.size else 0.0
            row["root_path_error_max_m"] = float(np.max(root_err)) if root_err.size else 0.0
            row["root_pos_mse"] = _mse(seq["root_pos"], true_seq["root_pos"])
            row["root_disp_mse"] = _mse(seq["root_pos"][-1] - seq["root_pos"][0], true_seq["root_pos"][-1] - true_seq["root_pos"][0])
            row["yaw_traj_mse"] = _mse(state[:, YAW_RATE_SLICE], true_state[:, YAW_RATE_SLICE])
            row["pose_rot6d_mse"] = _mse(state[:, POSE_SLICE], true_state[:, POSE_SLICE])
            row["bone_angvel_mse"] = _mse(aux, true_aux)
            if predicted_info:
                row["predicted_contact_mse_01"] = float(predicted_info.get("predicted_contact_mse_01", 0.0))
                row["predicted_contact_mse_raw"] = float(predicted_info.get("predicted_contact_mse_raw", 0.0))
                row["contact_train_split_n"] = int(predicted_info.get("train_split_n", 0))
                row["contact_test_split_n"] = int(predicted_info.get("test_split_n", 0))
                row["contact_gap_split_n"] = int(predicted_info.get("gap_split_n", 0))
                row["final_train_contact_plan_mse"] = float(predicted_info.get("final_train_contact_plan_mse", 0.0))
                row["final_train_contact_predict_mse"] = float(predicted_info.get("final_train_contact_predict_mse", 0.0))
                row["final_train_contact_predict_bce"] = float(predicted_info.get("final_train_contact_predict_bce", 0.0))
                row["final_train_contact_endpoint_support_ce"] = float(
                    predicted_info.get("final_train_contact_endpoint_support_ce", 0.0)
                )
                row["contact_step_weight"] = float(predicted_info.get("contact_step_weight", args.contact_step_weight))
                row["contact_predict_mse_weight"] = float(
                    predicted_info.get("contact_predict_mse_weight", args.contact_predict_mse_weight)
                )
                row["contact_predict_bce_weight"] = float(
                    predicted_info.get("contact_predict_bce_weight", args.contact_predict_bce_weight)
                )
                row["contact_state_weight"] = float(predicted_info.get("contact_state_weight", args.contact_state_weight))
                row["contact_endpoint_support_weight"] = float(
                    predicted_info.get("contact_endpoint_support_weight", args.contact_endpoint_support_weight)
                )
                threshold01 = predicted_info.get("contact_support_threshold01", [])
                row["contact_support_threshold01"] = threshold01
            rows.append(row)

    flat_rows = [_flatten_row(r) for r in rows]
    summary = _summarize(flat_rows)
    by_target = _summaries_by(flat_rows, "target")
    by_partition = _summaries_by(flat_rows, "split_partition")
    by_transition = _summaries_by(flat_rows, "oracle_endpoint_transition")
    by_symmetry = _summaries_by(flat_rows, "oracle_transition_symmetry")
    by_seed = _summaries_by(flat_rows, "contact_seed")
    split_leakage = _split_leakage_summary(flat_rows, horizon=int(args.horizon))
    hard_pass = float(summary.get("hard_acceptance_pass_rate", 0.0) or 0.0)
    decision = "go_anchor_between_no_residual" if hard_pass >= float(args.pass_rate_threshold) else "no_go_anchor_between_no_residual"

    payload = {
        "task": "anchor_between_acceptance_replay",
        "scope": "debug-only no-learned-residual anchor-between replay; support_side read-only audit",
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "stride": int(args.stride),
            "min_run_frames": int(args.min_run_frames),
            "max_windows_per_target": int(args.max_windows_per_target),
            "contact_mode": str(args.contact_mode),
            "split_mode": str(args.split_mode),
            "reconstructed_baseline_quantile": float(args.reconstructed_baseline_quantile),
            "command_quantile": float(args.command_quantile),
            "pass_rate_threshold": float(args.pass_rate_threshold),
            "train_fraction": float(args.train_fraction),
            "block_gap": int(args.block_gap),
            "epochs": int(args.epochs),
            "latent_dim": int(args.latent_dim),
            "contact_embed_dim": int(args.contact_embed_dim),
            "frame_hidden_dim": int(args.frame_hidden_dim),
            "dropout": float(args.dropout),
            "residual_l2_weight": float(args.residual_l2_weight),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "contact_step_weight": float(args.contact_step_weight),
            "contact_predict_mse_weight": float(args.contact_predict_mse_weight),
            "contact_predict_bce_weight": float(args.contact_predict_bce_weight),
            "contact_state_weight": float(args.contact_state_weight),
            "contact_endpoint_support_weight": float(args.contact_endpoint_support_weight),
            "contact_support_threshold_raw": float(CONTACT_THRESHOLD),
            "seed": int(args.seed),
            "contact_seeds": int(args.contact_seeds),
            "contact_seed_base": int(args.contact_seed_base) if args.contact_seed_base is not None else int(args.seed),
            "contact_seed_values": [int(x) for x in contact_seed_values if x is not None],
            "torch_num_threads": int(args.torch_num_threads),
            "dtype": "float32",
            "device": "cpu",
        },
        "schema": {
            "state281": {"shape": [int(args.horizon), STATE_DIM], "dtype": "float32", "device": "cpu"},
            "bone_angvel_aux": {"shape": [int(args.horizon), ANGVEL_DIM], "dtype": "float32", "device": "cpu"},
            "seq_rot6d": {"shape": [int(args.horizon), 276], "dtype": "float32", "device": "cpu"},
            "seq_root_pos": {"shape": [int(args.horizon), 3], "dtype": "float32", "device": "cpu"},
            "seq_root_vel": {"shape": [int(args.horizon), 2], "dtype": "float32", "device": "cpu"},
            "seq_contact": {"shape": [int(args.horizon), 2], "dtype": "float32", "device": "cpu"},
            "seq_yaw_rate": {"shape": [int(args.horizon)], "dtype": "float32", "device": "cpu"},
            "predicted_probe_contact": {
                "shape": [int(args.horizon), 2],
                "dtype": "float32",
                "device": "cpu",
                "range": "raw contact scale after inverse min/max from probe contact01",
            },
        },
        "matched_targets": list(MATCHED_TARGETS),
        "hard_gate_families": list(HARD_GATE_FAMILIES),
        "read_only_audits": ["support_side_core", "support_side_full_legacy", "command_compatibility"],
        "summary": summary,
        "summary_by_target": by_target,
        "summary_by_partition": by_partition,
        "summary_by_endpoint_transition": by_transition,
        "summary_by_transition_symmetry": by_symmetry,
        "summary_by_contact_seed": by_seed,
        "split_leakage": split_leakage,
        "verdict": {
            "decision": decision,
            "hard_acceptance_pass_rate": hard_pass,
            "pass_rate_threshold": float(args.pass_rate_threshold),
            "support_side_core_pass_rate": float(summary.get("support_side_core_pass_rate", 0.0) or 0.0),
            "support_side_full_legacy_pass_rate": float(summary.get("support_side_full_legacy_pass_rate", 0.0) or 0.0),
            "failed_family_counts": summary.get("failed_family_counts", {}),
        },
        "hard_constraint_confirmations": {
            "trained_new_model": bool(str(args.contact_mode) == "predicted_probe"),
            "trained_new_debug_contact_probe": bool(str(args.contact_mode) == "predicted_probe"),
            "trained_debug_contact_probe_seed_count": int(len([x for x in contact_seed_values if x is not None])),
            "trained_new_production_model": False,
            "learned_residual_used": False,
            "edited_production_runtime_trainer_gate": False,
            "mutated_checkpoint": False,
            "support_side_hard_gate": False,
        },
        "rows": rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.out_dir / "anchor_between_acceptance_summary.json"
    rows_csv = args.out_dir / "anchor_between_acceptance_rows.csv"
    summary_md = args.out_dir / "anchor_between_acceptance_summary.md"
    payload["artifacts"] = {
        "summary_json": str(summary_json),
        "rows_csv": str(rows_csv),
        "summary_md": str(summary_md),
    }
    _dump_json(summary_json, payload)
    _write_csv(rows_csv, flat_rows)
    _write_md(summary_md, payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--max-windows-per-target", type=int, default=0)
    p.add_argument("--contact-mode", choices=("ctx_hold", "oracle", "predicted_probe"), default="ctx_hold")
    p.add_argument(
        "--split-mode",
        choices=("contiguous", "regime_balanced", "regime_block_gap", "horizon_dedup_balanced"),
        default="contiguous",
    )
    p.add_argument("--reconstructed-baseline-quantile", type=float, default=100.0)
    p.add_argument("--command-quantile", type=float, default=100.0)
    p.add_argument("--pass-rate-threshold", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--contact-embed-dim", type=int, default=16)
    p.add_argument("--frame-hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--residual-l2-weight", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=3.0e-4)
    p.add_argument("--contact-step-weight", type=float, default=RESIDUAL_DEFAULT_CONTACT_LOSS_CONFIG.contact_step_weight)
    p.add_argument("--contact-predict-mse-weight", type=float, default=RESIDUAL_DEFAULT_CONTACT_LOSS_CONFIG.contact_predict_mse_weight)
    p.add_argument("--contact-predict-bce-weight", type=float, default=RESIDUAL_DEFAULT_CONTACT_LOSS_CONFIG.contact_predict_bce_weight)
    p.add_argument("--contact-state-weight", type=float, default=RESIDUAL_DEFAULT_CONTACT_LOSS_CONFIG.contact_state_weight)
    p.add_argument(
        "--contact-endpoint-support-weight",
        type=float,
        default=RESIDUAL_DEFAULT_CONTACT_LOSS_CONFIG.contact_endpoint_support_weight,
        help="Differentiable CE on predicted contact support labels at frame 0 and H-1.",
    )
    p.add_argument("--seed", type=int, default=20260606)
    p.add_argument("--contact-seeds", type=int, default=1)
    p.add_argument("--contact-seed-base", type=int, default=None)
    p.add_argument("--train-fraction", type=float, default=0.60)
    p.add_argument("--block-gap", type=int, default=8)
    p.add_argument("--torch-num-threads", type=int, default=8)
    p.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    return p.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(f"[OK] wrote {payload['artifacts']['summary_md']}")
    print(f"[OK] wrote {payload['artifacts']['summary_json']}")


if __name__ == "__main__":
    main()
