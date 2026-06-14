#!/usr/bin/env python3
"""Read-only/no-FK preflight for a context-conditioned soft-contact feature bridge.

No training, no checkpoint mutation, no production trainer/runtime/gate change.

This materializes the proposed feature groups for `between`:
  - ctx_state: previous clip context [C,281]
  - soft_contact_cycle: continuous contact/cycle features [H,8]
  - target_lowdim: root/heading/arrival intent, not full future pose
  - hard_support_tokens_debug: thresholded support tokens for ablation only

This probe intentionally does not use FK in the learning objective or success
metrics. Existing FK-derived contract metrics can be audited separately, but they
are not the feature-bridge design target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    CONTEXT_LEN_C,
    EGO_VEL_SLICE,
    STATE_DIM,
    YAW_RATE_SLICE,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    ANGVEL_DIM,
    DEFAULT_NPZ_ROOT,
    DEFAULT_Z_FEATURES,
    _dump_json,
    _dump_md,
    _fmt,
    _load_clips,
)
from tools.run_action_handoff_oracle_schedule_trajectory_decoder_smoke import (  # noqa: E402
    TinyDeterministicDecoder,
    _build_items,
    _fit_standardizer,
    _integrate_root_pos_torch,
    _loss_metrics,
    _one_hot_labels,
    _predict_raw,
    _reshape_state_aux,
    _run_phase_features,
    _seq_from_prediction,
    _support_labels,
    _support_stats,
    _world_root_vel_from_ego_torch,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import (  # noqa: E402
    DEFAULT_HORIZON,
)
from tools.run_action_handoff_support_schedule_predictive_baseline import (  # noqa: E402
    MATCHED_TARGETS,
    _build_splits,
    UNMATCHED_TARGET,
)


DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_between_feature_bridge_preflight_20260605")
OCP = True
CARV = False
RECON_QUANTILE = 100.0


FEATURE_GROUP_META: Dict[str, Dict[str, str]] = {
    "ctx_state": {
        "shape": "[C,281]",
        "role": "causal previous-clip context; observed input, encoded by model",
        "runtime_status": "available",
    },
    "soft_contact_cycle": {
        "shape": "[H,8]",
        "role": "continuous soft contact/cycle physics signal",
        "runtime_status": "oracle/debug in this preflight; production needs schedule source",
    },
    "target_lowdim": {
        "shape": "[6 + H*3]",
        "role": "low-dimensional root/heading/arrival intent",
        "runtime_status": "candidate runtime cue",
    },
    "hard_support_tokens_debug": {
        "shape": "[H,4] one-hot + [H,2] run phase + [6] stats",
        "role": "thresholded support-token ablation only",
        "runtime_status": "debug-only hard-label comparison",
    },
    "endpoint_prefix_debug": {
        "shape": "[279]",
        "role": "old smoke endpoint prefix; useful reference but future-pose leakage-prone",
        "runtime_status": "debug-only reference, not first-round target",
    },
}

VARIANTS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    (
        "ctx_target_lowdim",
        ("ctx_state", "target_lowdim"),
        "target intent without explicit future contact/cycle",
    ),
    (
        "ctx_soft_cycle",
        ("ctx_state", "soft_contact_cycle"),
        "soft contact/cycle without target intent",
    ),
    (
        "ctx_target_soft_cycle",
        ("ctx_state", "target_lowdim", "soft_contact_cycle"),
        "main feature-bridge hypothesis",
    ),
    (
        "ctx_target_hard_support_debug",
        ("ctx_state", "target_lowdim", "hard_support_tokens_debug"),
        "debug-only hard support-token comparison",
    ),
)


def _target_from_item(item: Any) -> np.ndarray:
    state = np.asarray(item.seq["state281"], dtype=np.float32).reshape(-1)
    aux = np.asarray(item.seq["bone_angvel"], dtype=np.float32).reshape(-1)
    return _finite_float32(np.concatenate([state, aux], axis=0))


def _finite_float32(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)


def _pad_first_zero(delta: np.ndarray, rows: int) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float32).reshape(max(0, rows - 1), -1)
    out = np.zeros((rows, d.shape[1] if d.ndim == 2 else 0), dtype=np.float32)
    if rows > 1 and d.size:
        out[1:] = d
    return out


def _soft_contact_cycle_features(contact: np.ndarray) -> np.ndarray:
    c = np.asarray(contact, dtype=np.float32).reshape(-1, 2)
    h = c.shape[0]
    dc = _pad_first_zero(np.diff(c, axis=0), h)
    anchor = np.zeros((h, 2), dtype=np.float32)
    if h > 1:
        anchor[1:] = c[:-1] * c[1:]
    balance = (c[:, 0:1] - c[:, 1:2]).astype(np.float32, copy=False)
    total = (c[:, 0:1] + c[:, 1:2]).astype(np.float32, copy=False)
    return _finite_float32(np.concatenate([c, dc, anchor, balance, total], axis=1))


def _target_lowdim_features(seq: Mapping[str, np.ndarray]) -> np.ndarray:
    state = np.asarray(seq["state281"], dtype=np.float32).reshape(-1, STATE_DIM)
    root_pos = np.asarray(seq["root_pos"], dtype=np.float32).reshape(-1, 3)
    cond_dir = np.asarray(seq["cond_dir"], dtype=np.float32).reshape(-1, 2)
    yaw_rate = np.asarray(seq["yaw_rate"], dtype=np.float32).reshape(-1, 1)
    root_disp = (root_pos[-1] - root_pos[0]).astype(np.float32, copy=False)
    endpoint_root = np.concatenate(
        [
            state[-1, EGO_VEL_SLICE].reshape(-1),
            state[-1, YAW_RATE_SLICE].reshape(-1),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    cond_yaw = np.concatenate([cond_dir, yaw_rate], axis=1).reshape(-1)
    return _finite_float32(np.concatenate([root_disp, endpoint_root, cond_yaw], axis=0))


def _hard_support_token_features(item: Any) -> np.ndarray:
    labels = _support_labels(np.asarray(item.seq["contact"], dtype=np.float32))
    parts = [
        _one_hot_labels(labels).reshape(-1),
        _run_phase_features(labels).reshape(-1),
        _support_stats(labels).reshape(-1),
    ]
    return _finite_float32(np.concatenate(parts, axis=0))


def _feature_groups_for_item(item: Any) -> Dict[str, np.ndarray]:
    seq = item.seq
    return {
        "ctx_state": _finite_float32(np.asarray(item.ctx, dtype=np.float32).reshape(-1)),
        "soft_contact_cycle": _soft_contact_cycle_features(seq["contact"]).reshape(-1),
        "target_lowdim": _target_lowdim_features(seq).reshape(-1),
        "hard_support_tokens_debug": _hard_support_token_features(item).reshape(-1),
        "endpoint_prefix_debug": _finite_float32(
            np.asarray(seq["state281"], dtype=np.float32)[-1, : CONTACT_SLICE.start].reshape(-1)
        ),
    }


def _numeric_summary(arrays: Sequence[np.ndarray]) -> Dict[str, Any]:
    if not arrays:
        return {
            "flat_dim": 0,
            "finite_fraction": 0.0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    flat = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in arrays], axis=0)
    finite = np.isfinite(flat)
    vals = flat[finite]
    return {
        "flat_dim": int(np.asarray(arrays[0]).reshape(-1).shape[0]),
        "finite_fraction": float(np.mean(finite)) if finite.size else 1.0,
        "mean": float(np.mean(vals)) if vals.size else None,
        "std": float(np.std(vals)) if vals.size else None,
        "min": float(np.min(vals)) if vals.size else None,
        "max": float(np.max(vals)) if vals.size else None,
    }


def _feature_group_rows(items: Sequence[Any]) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, np.ndarray]]]:
    cache: Dict[int, Dict[str, np.ndarray]] = {}
    by_group: Dict[str, List[np.ndarray]] = {k: [] for k in FEATURE_GROUP_META}
    for idx, item in enumerate(items):
        groups = _feature_groups_for_item(item)
        cache[idx] = groups
        for key, value in groups.items():
            by_group[key].append(value)

    rows: List[Dict[str, Any]] = []
    for key, meta in FEATURE_GROUP_META.items():
        stats = _numeric_summary(by_group.get(key, []))
        rows.append(
            {
                "feature_group": key,
                "shape_contract": meta["shape"],
                "flat_dim_per_item": stats["flat_dim"],
                "dtype": "float32",
                "device": "cpu",
                "finite_fraction": stats["finite_fraction"],
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "runtime_status": meta["runtime_status"],
                "role": meta["role"],
            }
        )
    return rows, cache


def _variant_rows(cache: Mapping[int, Mapping[str, np.ndarray]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, groups, purpose in VARIANTS:
        dims = {g: int(np.asarray(next(iter(cache.values()))[g]).reshape(-1).shape[0]) for g in groups} if cache else {}
        total_dim = int(sum(dims.values()))
        finite_parts: List[np.ndarray] = []
        for per_item in cache.values():
            finite_parts.extend(np.asarray(per_item[g], dtype=np.float32).reshape(-1) for g in groups)
        flat = np.concatenate(finite_parts, axis=0) if finite_parts else np.zeros((0,), dtype=np.float32)
        rows.append(
            {
                "variant": name,
                "groups": ",".join(groups),
                "flat_input_dim_per_item": total_dim,
                "group_dims_json": json.dumps(dims, sort_keys=True),
                "finite_fraction": float(np.mean(np.isfinite(flat))) if flat.size else 1.0,
                "purpose": purpose,
            }
        )
    return rows


def _features_for_variant(
    items: Sequence[Any],
    idxs: Sequence[int],
    *,
    variant_groups: Sequence[str],
    cache: Mapping[int, Mapping[str, np.ndarray]],
) -> np.ndarray:
    rows = []
    for item_idx in idxs:
        groups = cache[int(item_idx)]
        rows.append(np.concatenate([np.asarray(groups[g], dtype=np.float32).reshape(-1) for g in variant_groups], axis=0))
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _targets_for_items(items: Sequence[Any], idxs: Sequence[int]) -> np.ndarray:
    return np.stack([_target_from_item(items[int(i)]) for i in idxs], axis=0).astype(np.float32, copy=False)


def _stack_seq(items: Sequence[Any], idxs: Sequence[int], key: str) -> np.ndarray:
    return np.stack([np.asarray(items[int(i)].seq[key], dtype=np.float32) for i in idxs], axis=0).astype(
        np.float32,
        copy=False,
    )


def _train_variant(
    *,
    variant: str,
    groups: Sequence[str],
    items: Sequence[Any],
    cache: Mapping[int, Mapping[str, np.ndarray]],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    horizon: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    train_x_raw = _features_for_variant(items, train_idx, variant_groups=groups, cache=cache)
    test_x_raw = _features_for_variant(items, test_idx, variant_groups=groups, cache=cache)
    train_y_raw = _targets_for_items(items, train_idx)
    test_y_raw = _targets_for_items(items, test_idx)
    x_scaler = _fit_standardizer(train_x_raw)
    y_scaler = _fit_standardizer(train_y_raw)
    train_x = x_scaler.transform(train_x_raw)
    train_y = y_scaler.transform(train_y_raw)

    torch.manual_seed(int(seed))
    model = TinyDeterministicDecoder(train_x.shape[1], int(hidden_dim), train_y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    xtr = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    y_mean = torch.as_tensor(y_scaler.mean, dtype=torch.float32, device=device)
    y_std = torch.as_tensor(y_scaler.std, dtype=torch.float32, device=device)
    true_raw = torch.as_tensor(train_y_raw, dtype=torch.float32, device=device)
    true_root_pos = torch.as_tensor(_stack_seq(items, train_idx, "root_pos"), dtype=torch.float32, device=device)
    true_root_vel = torch.as_tensor(_stack_seq(items, train_idx, "root_vel"), dtype=torch.float32, device=device)
    true_cond_dir = torch.as_tensor(_stack_seq(items, train_idx, "cond_dir"), dtype=torch.float32, device=device)
    true_contact = torch.as_tensor(_stack_seq(items, train_idx, "contact"), dtype=torch.float32, device=device)

    final_terms: Dict[str, float] = {}
    for _epoch in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred_std = model(xtr)
        pred_raw = pred_std * y_std + y_mean
        state_w = int(horizon) * STATE_DIM
        pred_state = pred_raw[:, :state_w].reshape(-1, int(horizon), STATE_DIM)
        pred_aux = pred_raw[:, state_w:].reshape(-1, int(horizon), ANGVEL_DIM)
        true_state = true_raw[:, :state_w].reshape(-1, int(horizon), STATE_DIM)
        true_aux = true_raw[:, state_w:].reshape(-1, int(horizon), ANGVEL_DIM)
        pred_state = pred_state.clone()
        pred_state[:, :, CONTACT_SLICE] = true_contact
        pred_root_vel = _world_root_vel_from_ego_torch(
            pred_state[:, :, EGO_VEL_SLICE],
            true_cond_dir,
            command_align_root_vel=CARV,
        )
        pred_root_pos = _integrate_root_pos_torch(pred_root_vel, true_root_pos[:, 0])
        state_raw = F.mse_loss(pred_state, true_state)
        pose = F.mse_loss(pred_state[:, :, : CONTACT_SLICE.start], true_state[:, :, : CONTACT_SLICE.start])
        aux = F.mse_loss(pred_aux, true_aux)
        root_vel = F.mse_loss(pred_root_vel, true_root_vel)
        root_pos = F.mse_loss(pred_root_pos, true_root_pos)
        pose_step = F.mse_loss(
            pred_state[:, 1:, : CONTACT_SLICE.start] - pred_state[:, :-1, : CONTACT_SLICE.start],
            true_state[:, 1:, : CONTACT_SLICE.start] - true_state[:, :-1, : CONTACT_SLICE.start],
        )
        flat = F.mse_loss(pred_std, ytr)
        loss = (
            state_raw
            + pose
            + 0.5 * aux
            + 4.0 * root_vel
            + 8.0 * root_pos
            + 8.0 * pose_step
            + 0.05 * flat
        )
        loss.backward()
        opt.step()
        final_terms = {
            "loss": float(loss.detach().cpu().item()),
            "state_raw_mse": float(state_raw.detach().cpu().item()),
            "pose_prefix_mse": float(pose.detach().cpu().item()),
            "aux_bone_angvel_mse": float(aux.detach().cpu().item()),
            "root_vel_mse": float(root_vel.detach().cpu().item()),
            "root_pos_mse": float(root_pos.detach().cpu().item()),
            "pose_step_mse": float(pose_step.detach().cpu().item()),
            "flat_standardized_mse": float(flat.detach().cpu().item()),
        }

    train_pred_raw = _predict_raw(model, train_x_raw, x_scaler, y_scaler, device)
    test_pred_raw = _predict_raw(model, test_x_raw, x_scaler, y_scaler, device)
    train_state, _ = _reshape_state_aux(train_pred_raw, int(horizon))
    test_state, _ = _reshape_state_aux(test_pred_raw, int(horizon))
    for local_i, item_idx in enumerate(train_idx):
        train_state[local_i, :, CONTACT_SLICE] = np.asarray(items[int(item_idx)].seq["contact"], dtype=np.float32)
    for local_i, item_idx in enumerate(test_idx):
        test_state[local_i, :, CONTACT_SLICE] = np.asarray(items[int(item_idx)].seq["contact"], dtype=np.float32)
    params = int(sum(p.numel() for p in model.parameters()))
    return {
        "variant": variant,
        "groups": list(groups),
        "train_idx": [int(x) for x in train_idx],
        "test_idx": [int(x) for x in test_idx],
        "train_pred_raw": train_pred_raw,
        "test_pred_raw": test_pred_raw,
        "train_y_raw": train_y_raw,
        "test_y_raw": test_y_raw,
        "train_loss_metrics": _loss_metrics(train_pred_raw, train_y_raw, int(horizon)),
        "test_loss_metrics": _loss_metrics(test_pred_raw, test_y_raw, int(horizon)),
        "input_dim": int(train_x_raw.shape[1]),
        "output_dim": int(train_y_raw.shape[1]),
        "parameter_count": params,
        "x_constant_features_train": int(x_scaler.constant_count),
        "y_constant_outputs_train": int(y_scaler.constant_count),
        "final_train_terms": final_terms,
    }


def _mse_np(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(d * d)) if d.size else 0.0


def _step_mse_np(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape[0] < 2 or b.shape[0] < 2:
        return 0.0
    return _mse_np(np.diff(a, axis=0), np.diff(b, axis=0))


def _evaluate_variant_predictions_no_fk(
    *,
    result: Mapping[str, Any],
    items: Sequence[Any],
    horizon: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for partition, pred_key, idx_key in (
        ("train", "train_pred_raw", "train_idx"),
        ("test", "test_pred_raw", "test_idx"),
    ):
        pred_state, pred_aux = _reshape_state_aux(np.asarray(result[pred_key], dtype=np.float32), int(horizon))
        for local_i, item_idx in enumerate(result[idx_key]):
            item = items[int(item_idx)]
            seq = _seq_from_prediction(
                item,
                pred_state[local_i],
                pred_aux[local_i],
                oracle_contact_passthrough=OCP,
                command_align_root_vel=CARV,
            )
            seq = dict(seq)
            seq["state281"] = pred_state[local_i].astype(np.float32, copy=False)
            true_state = np.asarray(item.seq["state281"], dtype=np.float32).reshape(int(horizon), STATE_DIM)
            true_aux = np.asarray(item.seq["bone_angvel"], dtype=np.float32).reshape(int(horizon), ANGVEL_DIM)
            true_target = _target_lowdim_features(item.seq)
            pred_target = _target_lowdim_features(seq)
            row = {
                "variant": str(result["variant"]),
                "split": "contiguous_block",
                "split_kind": "debug_train_fit_no_fk",
                "partition": partition,
                "clip": item.clip,
                "start": int(item.start),
                "end": int(item.end),
                "state_mse": _mse_np(pred_state[local_i], true_state),
                "pose_rot6d_mse": _mse_np(pred_state[local_i, :, : EGO_VEL_SLICE.start], true_state[:, : EGO_VEL_SLICE.start]),
                "ego_vel_mse": _mse_np(pred_state[local_i, :, EGO_VEL_SLICE], true_state[:, EGO_VEL_SLICE]),
                "yaw_rate_mse": _mse_np(pred_state[local_i, :, YAW_RATE_SLICE], true_state[:, YAW_RATE_SLICE]),
                "contact_mse": _mse_np(pred_state[local_i, :, CONTACT_SLICE], true_state[:, CONTACT_SLICE]),
                "bone_angvel_aux_mse": _mse_np(pred_aux[local_i], true_aux),
                "pose_step_mse": _step_mse_np(
                    pred_state[local_i, :, : EGO_VEL_SLICE.start],
                    true_state[:, : EGO_VEL_SLICE.start],
                ),
                "root_intent_mse": _mse_np(pred_target, true_target),
                "root_pos_mse": _mse_np(seq["root_pos"], item.seq["root_pos"]),
                "root_vel_mse": _mse_np(seq["root_vel"], item.seq["root_vel"]),
                "input_dim": int(result["input_dim"]),
                "parameter_count": int(result["parameter_count"]),
            }
            rows.append(row)
    return rows


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([bool(r.get(key, False)) for r in rows])) if rows else 0.0


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            v = float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def _feature_target_guard(items: Sequence[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        target = _target_from_item(item)
        lowdim = _target_lowdim_features(item.seq)
        rows.append(
            {
                "clip": item.clip,
                "start": int(item.start),
                "end": int(item.end),
                "target_dim": int(target.reshape(-1).shape[0]),
                "target_lowdim": int(lowdim.reshape(-1).shape[0]),
                "target_finite": bool(np.all(np.isfinite(target))),
                "target_lowdim_finite": bool(np.all(np.isfinite(lowdim))),
            }
        )
    summary = {
        "n": int(len(rows)),
        "target_finite_rate": _rate(rows, "target_finite"),
        "target_lowdim_finite_rate": _rate(rows, "target_lowdim_finite"),
        "target_dim": int(rows[0]["target_dim"]) if rows else 0,
        "target_lowdim_dim": int(rows[0]["target_lowdim"]) if rows else 0,
    }
    return rows, summary


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
            out = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                out[key] = value
            writer.writerow(out)


def _write_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    guard = payload["feature_target_guard"]
    train_fit_ran = bool(payload.get("decision", {}).get("debug_train_fit_ran", False))
    lines = [
        "# Between Feature-Bridge Preflight",
        "",
        (
            "Debug train-fit ablation included. No checkpoint mutation, no production runtime/trainer/gate change."
            if train_fit_ran
            else "Read-only preflight. No training, no checkpoint mutation, no production runtime/trainer/gate change."
        ),
        "",
        "## Reframe",
        "",
        "`between` is treated as `previous context + soft contact/cycle + target intent -> bridge features -> eval state`.",
        "`state281` remains the assembled evaluation interface, not the primary modeling space.",
        "",
        "## Dataset / Feature Guard",
        "",
        f"- matched windows: `{payload['dataset']['matched_window_count']}` from `{payload['dataset']['matched_targets']}`",
        f"- excluded unmatched diagnostic target: `{payload['dataset']['unmatched_target']}`",
        f"- target finite / target-lowdim finite: `{_fmt(guard['target_finite_rate'], 3)}` / `{_fmt(guard['target_lowdim_finite_rate'], 3)}`",
        f"- target dims: raw `{guard['target_dim']}`, lowdim `{guard['target_lowdim_dim']}`",
        "",
        "## Feature Groups",
        "",
        "| group | shape | dim | finite | runtime status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in payload["feature_groups"]:
        lines.append(
            f"| {row['feature_group']} | {row['shape_contract']} | "
            f"{row['flat_dim_per_item']} | {_fmt(row['finite_fraction'], 3)} | "
            f"{row['runtime_status']} |"
        )
    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "| variant | groups | dim | finite | purpose |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in payload["variants"]:
        lines.append(
            f"| {row['variant']} | {row['groups']} | {row['flat_input_dim_per_item']} | "
            f"{_fmt(row['finite_fraction'], 3)} | {row['purpose']} |"
        )
    if payload.get("train_fit_summaries"):
        lines.extend(
            [
                "",
                "## Debug Train-Fit",
                "",
                "| variant | partition | n | state mse | pose mse | root intent mse | root pos mse | root vel mse |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload["train_fit_summaries"]:
            lines.append(
                f"| {row['variant']} | {row['partition']} | {row['n']} | "
                f"{_fmt(row['state_mse'], 8)} | "
                f"{_fmt(row['pose_rot6d_mse'], 8)} | "
                f"{_fmt(row['root_intent_mse'], 8)} | "
                f"{_fmt(row['root_pos_mse'], 8)} | "
                f"{_fmt(row['root_vel_mse'], 8)} |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- preflight clean: `{str(payload['decision']['preflight_clean']).lower()}`",
            f"- next step: `{payload['decision']['next_step']}`",
            "",
            "## Artifacts",
            "",
            f"- summary json: `{payload['artifacts']['summary_json']}`",
            f"- feature groups csv: `{payload['artifacts']['feature_groups_csv']}`",
            f"- variants csv: `{payload['artifacts']['variants_csv']}`",
            f"- feature target guard rows csv: `{payload['artifacts']['feature_target_guard_rows_csv']}`",
        ]
    )
    if payload["artifacts"].get("train_fit_rows_csv"):
        lines.append(f"- train-fit rows csv: `{payload['artifacts']['train_fit_rows_csv']}`")
    if payload["artifacts"].get("train_fit_summary_csv"):
        lines.append(f"- train-fit summary csv: `{payload['artifacts']['train_fit_summary_csv']}`")
    _dump_md(path, lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only between feature-bridge preflight")
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument(
        "--run-train-fit",
        action="store_true",
        help="debug-only train-fit for the four feature ablations; default is read-only preflight",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--block-gap", type=int, default=8)
    p.add_argument("--split-low-n-threshold", type=int, default=20)
    p.add_argument("--torch-num-threads", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clips = _load_clips(args.npz_root, args.z_features)
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    main_items = [item for item in all_items if item.clip in MATCHED_TARGETS]
    unmatched_items = [item for item in all_items if item.clip == UNMATCHED_TARGET]
    if not main_items:
        raise RuntimeError("no matched feature-bridge items available")

    feature_rows, feature_cache = _feature_group_rows(main_items)
    variant_rows = _variant_rows(feature_cache)
    guard_rows, guard_summary = _feature_target_guard(main_items)

    train_fit_rows: List[Dict[str, Any]] = []
    train_fit_summaries: List[Dict[str, Any]] = []
    if bool(args.run_train_fit):
        torch.set_num_threads(int(args.torch_num_threads))
        splits = _build_splits(
            main_items,
            train_fraction=float(args.train_fraction),
            block_gap=int(args.block_gap),
            seed=int(args.seed),
            low_n_threshold=int(args.split_low_n_threshold),
            include_random=False,
        )
        split = next((s for s in splits if str(s.name) == "contiguous_block"), splits[0])
        device = torch.device("cpu")
        for variant_name, groups, _purpose in VARIANTS:
            result = _train_variant(
                variant=variant_name,
                groups=groups,
                items=main_items,
                cache=feature_cache,
                train_idx=split.train_idx,
                test_idx=split.test_idx,
                horizon=int(args.horizon),
                hidden_dim=int(args.hidden_dim),
                epochs=int(args.epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                seed=int(args.seed),
                device=device,
            )
            rows = _evaluate_variant_predictions_no_fk(
                result=result,
                items=main_items,
                horizon=int(args.horizon),
            )
            for row in rows:
                row["train_fit_epochs"] = int(args.epochs)
                row["state_mse"] = (
                    float(result["train_loss_metrics"]["state_mse"])
                    if row.get("partition") == "train"
                    else float(result["test_loss_metrics"]["state_mse"])
                )
            train_fit_rows.extend(rows)
            for partition in ("train", "test"):
                part_rows = [r for r in rows if str(r.get("partition")) == partition]
                summary = {
                    "variant": variant_name,
                    "partition": partition,
                    "n": int(len(part_rows)),
                    "input_dim": int(result["input_dim"]),
                    "output_dim": int(result["output_dim"]),
                    "parameter_count": int(result["parameter_count"]),
                    "state_mse": _mean(part_rows, "state_mse"),
                    "pose_rot6d_mse": _mean(part_rows, "pose_rot6d_mse"),
                    "ego_vel_mse": _mean(part_rows, "ego_vel_mse"),
                    "yaw_rate_mse": _mean(part_rows, "yaw_rate_mse"),
                    "bone_angvel_aux_mse": _mean(part_rows, "bone_angvel_aux_mse"),
                    "pose_step_mse": _mean(part_rows, "pose_step_mse"),
                    "root_intent_mse": _mean(part_rows, "root_intent_mse"),
                    "root_pos_mse": _mean(part_rows, "root_pos_mse"),
                    "root_vel_mse": _mean(part_rows, "root_vel_mse"),
                }
                train_fit_summaries.append(summary)

    finite_ok = all(float(row["finite_fraction"]) == 1.0 for row in feature_rows)
    target_ok = bool(
        abs(float(guard_summary["target_finite_rate"]) - 1.0) <= 1e-12
        and abs(float(guard_summary["target_lowdim_finite_rate"]) - 1.0) <= 1e-12
    )
    preflight_clean = bool(finite_ok and target_ok)
    payload: Dict[str, Any] = {
        "task": "between_feature_bridge_preflight",
        "status": "debug_train_fit_ablation_no_checkpoint" if bool(args.run_train_fit) else "read_only_preflight_no_training",
        "flags": {
            "oracle_contact_passthrough": OCP,
            "command_align_root_vel": CARV,
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "stride": int(args.stride),
            "min_run_frames": int(args.min_run_frames),
        },
        "schema": {
            "state281": {
                "shape": [int(args.horizon), STATE_DIM],
                "dtype": "float32",
                "device": "cpu",
                "role": "assembled evaluation/reconstruction interface only",
            },
            "bone_angvel_witness": {
                "shape": [int(args.horizon), ANGVEL_DIM],
                "dtype": "float32",
                "device": "cpu",
                "role": "optional aux/witness, not a state281 field",
            },
        },
        "dataset": {
            "matched_targets": list(MATCHED_TARGETS),
            "matched_window_count": int(len(main_items)),
            "unmatched_target": UNMATCHED_TARGET,
            "unmatched_window_count": int(len(unmatched_items)),
        },
        "feature_groups": feature_rows,
        "variants": variant_rows,
        "feature_target_guard": guard_summary,
        "train_fit_summaries": train_fit_summaries,
        "decision": {
            "preflight_clean": preflight_clean,
            "finite_feature_groups": finite_ok,
            "feature_targets_finite": target_ok,
            "debug_train_fit_ran": bool(args.run_train_fit),
            "next_step": (
                "run a debug train-fit for variants A/B/C/D"
                if preflight_clean and not bool(args.run_train_fit)
                else "compare no-FK feature metrics and redesign split output/latent target"
                if preflight_clean
                else "fix feature contract or feature target guard before train-fit"
            ),
        },
        "artifacts": {
            "summary_json": str(args.out_dir / "summary.json"),
            "summary_md": str(args.out_dir / "summary.md"),
            "feature_groups_csv": str(args.out_dir / "feature_groups.csv"),
            "variants_csv": str(args.out_dir / "variants.csv"),
            "feature_target_guard_rows_csv": str(args.out_dir / "feature_target_guard_rows.csv"),
            "train_fit_rows_csv": str(args.out_dir / "train_fit_rows.csv") if bool(args.run_train_fit) else "",
            "train_fit_summary_csv": str(args.out_dir / "train_fit_summary.csv") if bool(args.run_train_fit) else "",
        },
    }

    _write_csv(args.out_dir / "feature_groups.csv", feature_rows)
    _write_csv(args.out_dir / "variants.csv", variant_rows)
    _write_csv(args.out_dir / "feature_target_guard_rows.csv", guard_rows)
    if bool(args.run_train_fit):
        _write_csv(args.out_dir / "train_fit_rows.csv", train_fit_rows)
        _write_csv(args.out_dir / "train_fit_summary.csv", train_fit_summaries)
    _dump_json(args.out_dir / "summary.json", payload)
    _write_summary_md(args.out_dir / "summary.md", payload)
    print(f"[OK] wrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
