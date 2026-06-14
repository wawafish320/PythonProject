#!/usr/bin/env python3
"""Debug-only anchored residual bridge probe.

This probe keeps the prior GRU context/contact encoders and variant controls, but
changes the decoder parameterization:
  pose/local = ctx-last hold + parallel residual
  root       = linear-to-goal anchor + parallel residual

No production trainer, checkpoint, or model class is touched.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.run_action_handoff_between_gru_bridge_probe import (  # noqa: E402
    ANGVEL_DIM,
    BONE0_ANGVEL_SLICE,
    CONTACT_SLICE,
    CONTACT_LABEL_THRESHOLD,
    ContactLossConfig,
    DEFAULT_CONTACT_LOSS_CONFIG,
    DEFAULT_HORIZON,
    DEFAULT_NPZ_ROOT,
    DEFAULT_Z_FEATURES,
    EPS,
    FPS,
    MATCHED_TARGETS,
    NONROOT_ANGVEL_SLICE,
    NONROOT_ROT6D_SLICE,
    POSE_DIM,
    ROOT_MOTION_SLICE,
    ROOT_ROT6D_SLICE,
    STATE_DIM,
    UNMATCHED_TARGET,
    VARIANT_SPECS,
    VariantSpec,
    _add_delta_columns,
    _batch_from_items,
    _build_items,
    _build_split,
    _contact_threshold01_from_raw,
    _fit_normalizers,
    _load_probe_clips,
    _loss_terms,
    _make_baseline_rows,
    _metric_rows_for_predictions,
    _mse_np,
    _module_grad_norm,
    _root_pos_invariant_row,
    _resolve_device,
    _safe_mean,
    _summary_rows,
    _tensor_grad_l2_mean,
    _tensor_l2_mean,
    _write_csv,
    _write_json,
)


DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_between_anchor_residual_probe_20260606")
DEFAULT_REG_LADDER_OUT_DIR = Path("debug_output/_tmp_action_handoff_between_anchor_residual_reg_ladder_20260606")
DEFAULT_TRAIN_CURVE_OUT_DIR = Path("debug_output/_tmp_action_handoff_between_anchor_residual_train_curve_20260606")


def _anchor_state_aux_raw(batch: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    ctx_last = batch["ctx_state_raw"][:, -1]
    aux_last = batch["ctx_aux_raw"][:, -1]
    goal = batch["goal_raw"]
    cond_dir = batch["cond_dir"]
    _, h, _ = cond_dir.shape

    state = ctx_last[:, None, :].repeat(1, h, 1).clone()
    aux = aux_last[:, None, :].repeat(1, h, 1).clone()

    root_disp_xy = goal[:, :2]
    world_vel = root_disp_xy * float(FPS) / float(max(1, h - 1))
    norm = torch.linalg.norm(cond_dir, dim=-1, keepdim=True).clamp_min(EPS)
    fwd = cond_dir / norm
    lat = torch.stack([-fwd[..., 1], fwd[..., 0]], dim=-1)
    ego = torch.stack(
        [
            torch.sum(world_vel[:, None, :] * fwd, dim=-1),
            torch.sum(world_vel[:, None, :] * lat, dim=-1),
        ],
        dim=-1,
    )
    state[:, :, ROOT_MOTION_SLICE.start : ROOT_MOTION_SLICE.start + 2] = ego

    start_yaw = ctx_last[:, ROOT_MOTION_SLICE.stop - 1 : ROOT_MOTION_SLICE.stop]
    end_yaw = goal[:, 5:6]
    alpha = torch.linspace(0.0, 1.0, h, dtype=state.dtype, device=state.device).reshape(1, h, 1)
    yaw = start_yaw[:, None, :] * (1.0 - alpha) + end_yaw[:, None, :] * alpha
    state[:, :, ROOT_MOTION_SLICE.stop - 1 : ROOT_MOTION_SLICE.stop] = yaw
    return state, aux


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


class AnchoredResidualBridgeProbe(nn.Module):
    def __init__(
        self,
        *,
        goal_dim: int,
        latent_dim: int,
        contact_embed_dim: int,
        frame_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        h = int(latent_dim)
        ce = int(contact_embed_dim)
        fh = int(frame_hidden_dim)
        self.ctx_gru = nn.GRU(STATE_DIM, h, batch_first=True)
        self.goal_mlp = nn.Sequential(nn.Linear(int(goal_dim), h), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(h, h))
        self.contact_plan = ContactPlanGRU(h)
        self.contact_embed = nn.Sequential(nn.Linear(2, ce), nn.GELU(), nn.Linear(ce, ce))
        self.frame_mlp = nn.Sequential(
            nn.Linear(2 * h + ce + 3, fh),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(fh, fh),
            nn.GELU(),
        )
        self.root_head = nn.Linear(fh, 3)
        self.pose_root_head = nn.Linear(fh, 6)
        self.pose_local_head = nn.Linear(fh, POSE_DIM - 6)
        self.aux_root_head = nn.Linear(fh, 3)
        self.aux_local_head = nn.Linear(fh, ANGVEL_DIM - 3)
        self._init_small_residual_heads()

    def _init_small_residual_heads(self) -> None:
        for head in (
            self.root_head,
            self.pose_root_head,
            self.pose_local_head,
            self.aux_root_head,
            self.aux_local_head,
        ):
            nn.init.normal_(head.weight, mean=0.0, std=1.0e-4)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        spec: VariantSpec,
        *,
        state_norm: Any,
        aux_norm: Any,
        negative_contact: str,
        retain_usage: bool = False,
    ) -> Dict[str, Any]:
        ctx_state = batch["ctx_state_n"]
        ctx_contact = batch["ctx_contact"]
        goal_n = batch["goal_n"]
        gt_contact = batch["gt_contact"]
        b, horizon, _ = gt_contact.shape
        _, h_n = self.ctx_gru(ctx_state)
        z_ctx = h_n[-1]
        z_goal = self.goal_mlp(goal_n) if spec.use_goal else torch.zeros_like(z_ctx)

        predicted_contact: Optional[torch.Tensor] = None
        if spec.contact_mode in {"predicted", "oracle", "negative"}:
            predicted_contact = self.contact_plan(ctx_contact, z_ctx, z_goal, horizon)

        if spec.contact_mode == "predicted":
            contact_used = predicted_contact
        elif spec.contact_mode == "oracle":
            contact_used = gt_contact
        elif spec.contact_mode == "negative":
            g = torch.Generator(device=gt_contact.device)
            g.manual_seed(20260606)
            contact_used = torch.rand(gt_contact.shape, dtype=gt_contact.dtype, device=gt_contact.device, generator=g)
        else:
            contact_used = ctx_contact[:, -1:, :].repeat(1, horizon, 1)

        assert contact_used is not None
        anchor_state_raw, anchor_aux_raw = _anchor_state_aux_raw(batch)
        anchor_state_n = state_norm.transform_t(anchor_state_raw)
        anchor_aux_n = aux_norm.transform_t(anchor_aux_raw)

        phase = torch.linspace(0.0, 1.0, horizon, dtype=ctx_state.dtype, device=ctx_state.device).reshape(1, horizon, 1)
        phase = phase.repeat(b, 1, 1)
        phase_feat = torch.cat(
            [
                phase,
                torch.sin(2.0 * math.pi * phase),
                torch.cos(2.0 * math.pi * phase),
            ],
            dim=-1,
        )
        contact_feat = self.contact_embed(contact_used)
        z_ctx_seq = z_ctx[:, None, :].repeat(1, horizon, 1)
        z_goal_seq = z_goal[:, None, :].repeat(1, horizon, 1)
        frame = self.frame_mlp(torch.cat([z_ctx_seq, z_goal_seq, contact_feat, phase_feat], dim=-1))

        state = anchor_state_n.clone()
        state[:, :, ROOT_ROT6D_SLICE] = anchor_state_n[:, :, ROOT_ROT6D_SLICE] + self.pose_root_head(frame)
        state[:, :, NONROOT_ROT6D_SLICE] = anchor_state_n[:, :, NONROOT_ROT6D_SLICE] + self.pose_local_head(frame)
        state[:, :, ROOT_MOTION_SLICE] = anchor_state_n[:, :, ROOT_MOTION_SLICE] + self.root_head(frame)
        state[:, :, CONTACT_SLICE] = contact_used

        aux = anchor_aux_n.clone()
        aux[:, :, BONE0_ANGVEL_SLICE] = anchor_aux_n[:, :, BONE0_ANGVEL_SLICE] + self.aux_root_head(frame)
        aux[:, :, NONROOT_ANGVEL_SLICE] = anchor_aux_n[:, :, NONROOT_ANGVEL_SLICE] + self.aux_local_head(frame)

        residual_l2 = torch.mean((state - anchor_state_n) ** 2) + torch.mean((aux - anchor_aux_n) ** 2)
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
            "state_n": state,
            "aux_n": aux,
            "contact_used": contact_used,
            "predicted_contact": predicted_contact,
            "anchor_state_n": anchor_state_n,
            "anchor_aux_n": anchor_aux_n,
            "residual_l2": residual_l2,
            "usage_tensors": usage_tensors,
        }


def _train_variant(
    *,
    spec: VariantSpec,
    train_batch: Mapping[str, torch.Tensor],
    state_norm: Any,
    aux_norm: Any,
    goal_dim: int,
    latent_dim: int,
    contact_embed_dim: int,
    frame_hidden_dim: int,
    dropout: float,
    residual_l2_weight: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    negative_contact: str,
    contact_loss: ContactLossConfig = DEFAULT_CONTACT_LOSS_CONFIG,
) -> Tuple[AnchoredResidualBridgeProbe, Dict[str, float], Dict[str, Any]]:
    torch.manual_seed(int(seed))
    model = AnchoredResidualBridgeProbe(
        goal_dim=int(goal_dim),
        latent_dim=int(latent_dim),
        contact_embed_dim=int(contact_embed_dim),
        frame_hidden_dim=int(frame_hidden_dim),
        dropout=float(dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    final_terms: Dict[str, float] = {}
    for _epoch in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(
            train_batch,
            spec,
            state_norm=state_norm,
            aux_norm=aux_norm,
            negative_contact=str(negative_contact),
        )
        loss, final_terms = _loss_terms(
            out=out,
            batch=train_batch,
            state_norm=state_norm,
            aux_norm=aux_norm,
            spec=spec,
            contact_loss=contact_loss,
        )
        loss = loss + float(residual_l2_weight) * out["residual_l2"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"{spec.name}: non-finite loss")
        loss.backward()
        opt.step()

    model.train()
    opt.zero_grad(set_to_none=True)
    usage_out = model(
        train_batch,
        spec,
        state_norm=state_norm,
        aux_norm=aux_norm,
        negative_contact=str(negative_contact),
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
    usage_loss = usage_loss + float(residual_l2_weight) * usage_out["residual_l2"]
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
        "frame_mlp_grad_l2_mean": _module_grad_norm(model.frame_mlp),
        "root_head_grad_l2_mean": _module_grad_norm(model.root_head),
        "pose_root_head_grad_l2_mean": _module_grad_norm(model.pose_root_head),
        "pose_local_head_grad_l2_mean": _module_grad_norm(model.pose_local_head),
        "aux_root_head_grad_l2_mean": _module_grad_norm(model.aux_root_head),
        "aux_local_head_grad_l2_mean": _module_grad_norm(model.aux_local_head),
        "residual_l2": float(usage_out["residual_l2"].detach().cpu().item()),
        "contact_step_weight": float(contact_loss.contact_step_weight),
        "contact_predict_mse_weight": float(contact_loss.contact_predict_mse_weight),
        "contact_predict_bce_weight": float(contact_loss.contact_predict_bce_weight),
        "contact_state_weight": float(contact_loss.contact_state_weight),
        "contact_endpoint_support_weight": float(contact_loss.contact_endpoint_support_weight),
        **{f"final_{k}": v for k, v in usage_terms.items()},
    }
    opt.zero_grad(set_to_none=True)
    return model, final_terms, grad_usage


def _predict_anchor_model(
    *,
    model: AnchoredResidualBridgeProbe,
    spec: VariantSpec,
    batch: Mapping[str, torch.Tensor],
    state_norm: Any,
    aux_norm: Any,
    negative_contact: str,
) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        out = model(
            batch,
            spec,
            state_norm=state_norm,
            aux_norm=aux_norm,
            negative_contact=str(negative_contact),
        )
        state = state_norm.inverse_t(out["state_n"])
        aux = aux_norm.inverse_t(out["aux_n"])
        pred_contact = out.get("predicted_contact")
        return {
            "state": state.detach().cpu().numpy().astype("float32"),
            "aux": aux.detach().cpu().numpy().astype("float32"),
            "contact_used": out["contact_used"].detach().cpu().numpy().astype("float32"),
            "predicted_contact": (
                pred_contact.detach().cpu().numpy().astype("float32")
                if pred_contact is not None
                else out["contact_used"].detach().cpu().numpy().astype("float32")
            ),
        }


# Seeds vary the model init / contact-plan RNG only; data, split, and normalizers
# are held fixed so the spread is purely training-run variance, not data variance.
YAW_SWEEP_FOCUS = ("no_contact", "predicted_contact", "oracle_contact_upper_bound")
YAW_SWEEP_METRICS = ("yaw_traj_mse", "contact_plan_mse", "endpoint_yaw_rate_mse", "root_disp_mse")
REG_LADDER_METRICS = (
    "yaw_traj_mse",
    "root_pos_mse",
    "root_disp_mse",
    "pose_nonroot_rot6d_mse",
    "contact_plan_mse",
    "endpoint_yaw_rate_mse",
)
TRAIN_CURVE_METRICS = (
    "yaw_traj_mse",
    "root_pos_mse",
    "root_disp_mse",
    "pose_nonroot_rot6d_mse",
    "contact_plan_mse",
)


def _mean_std_min_max(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float]:
    vals: List[float] = []
    for row in rows:
        try:
            v = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _summary_value(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    model_kind: str,
    partition: str,
    key: str,
) -> Optional[float]:
    for row in rows:
        if (
            str(row.get("variant")) == str(variant)
            and str(row.get("model_kind")) == str(model_kind)
            and str(row.get("partition")) == str(partition)
        ):
            try:
                value = float(row.get(key, float("nan")))
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
    return None


def _subset_train_indices(train_idx: Sequence[int], fraction: float) -> Tuple[int, ...]:
    idxs = tuple(int(i) for i in train_idx)
    if not idxs:
        return tuple()
    n = max(1, int(round(float(fraction) * len(idxs))))
    n = min(n, len(idxs))
    if n >= len(idxs):
        return idxs
    positions = np.linspace(0, len(idxs) - 1, n, dtype=np.float64)
    out: List[int] = []
    seen = set()
    for pos in positions:
        i = int(round(float(pos)))
        i = max(0, min(len(idxs) - 1, i))
        if i not in seen:
            out.append(idxs[i])
            seen.add(i)
    cursor = 0
    while len(out) < n and cursor < len(idxs):
        if cursor not in seen:
            out.append(idxs[cursor])
            seen.add(cursor)
        cursor += 1
    return tuple(out[:n])


def _yaw_scale_baseline_rows(
    *,
    main_items: Sequence[Any],
    split: Any,
    baseline_summary: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    train_yaws: List[np.ndarray] = []
    for item_i in split.train_idx:
        state = np.asarray(main_items[int(item_i)].seq["state281"], dtype=np.float32).reshape(-1, STATE_DIM)
        train_yaws.append(state[:, ROOT_MOTION_SLICE.stop - 1 : ROOT_MOTION_SLICE.stop])
    if not train_yaws:
        return []
    train_yaw = np.stack(train_yaws, axis=0).astype(np.float32)
    scalar_mean = np.mean(train_yaw, axis=(0, 1), keepdims=True).astype(np.float32)
    step_mean = np.mean(train_yaw, axis=0, keepdims=True).astype(np.float32)

    true_test: List[np.ndarray] = []
    for item_i in split.test_idx:
        state = np.asarray(main_items[int(item_i)].seq["state281"], dtype=np.float32).reshape(-1, STATE_DIM)
        true_test.append(state[:, ROOT_MOTION_SLICE.stop - 1 : ROOT_MOTION_SLICE.stop])
    true = np.stack(true_test, axis=0).astype(np.float32)
    scalar_pred = np.repeat(scalar_mean, true.shape[0] * true.shape[1], axis=0).reshape(true.shape)
    step_pred = np.repeat(step_mean, true.shape[0], axis=0).reshape(true.shape)
    anchor_floor = _summary_value(
        baseline_summary,
        variant="root_linear_to_goal_pose_hold",
        model_kind="dumb_baseline",
        partition="test",
        key="yaw_traj_mse",
    )
    rows = [
        {
            "baseline": "train_scalar_mean_yaw_rate",
            "partition": "test",
            "yaw_traj_mse": _mse_np(scalar_pred, true),
            "anchor_floor_yaw_traj_mse": float(anchor_floor) if anchor_floor is not None else "",
        },
        {
            "baseline": "train_per_step_mean_yaw_rate",
            "partition": "test",
            "yaw_traj_mse": _mse_np(step_pred, true),
            "anchor_floor_yaw_traj_mse": float(anchor_floor) if anchor_floor is not None else "",
        },
    ]
    if anchor_floor is not None:
        for row in rows:
            row["anchor_over_baseline_ratio"] = float(anchor_floor) / max(float(row["yaw_traj_mse"]), EPS)
            row["baseline_minus_anchor"] = float(row["yaw_traj_mse"]) - float(anchor_floor)
    return rows


def _reg_ladder_summarize_partition_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    anchor_floor_yaw_traj: Optional[float],
    pose_guard_floor: Optional[float],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[float, float, int, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = (
            float(row["residual_l2_weight"]),
            float(row["dropout"]),
            int(row["frame_hidden_dim"]),
            str(row["variant"]),
            str(row["partition"]),
        )
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []
    for (residual_l2_weight, dropout, frame_hidden_dim, variant, partition), rows in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "residual_l2_weight": residual_l2_weight,
            "dropout": dropout,
            "frame_hidden_dim": frame_hidden_dim,
            "variant": variant,
            "partition": partition,
            "n_seeds": int(len(rows)),
            "anchor_floor_yaw_traj_mse": float(anchor_floor_yaw_traj) if anchor_floor_yaw_traj is not None else "",
            "ctx_last_hold_pose_nonroot_rot6d_mse": float(pose_guard_floor) if pose_guard_floor is not None else "",
            "crossed_anchor": (
                bool(_mean_std_min_max(rows, "yaw_traj_mse")["mean"] <= float(anchor_floor_yaw_traj))
                if partition == "test" and anchor_floor_yaw_traj is not None
                else ""
            ),
            "pose_guard_ok": (
                bool(_mean_std_min_max(rows, "pose_nonroot_rot6d_mse")["mean"] <= float(pose_guard_floor))
                if partition == "test" and pose_guard_floor is not None
                else ""
            ),
            "root_pos_guard_ok_1e4": (
                bool(_mean_std_min_max(rows, "root_pos_mse")["mean"] <= 1.0e-4)
                if partition == "test"
                else ""
            ),
            "root_invariant_all_ok": bool(all(bool(r.get("root_invariant_ok", False)) for r in rows)),
            "root_invariant_abs_delta_max": float(max(float(r.get("root_invariant_abs_delta", 0.0)) for r in rows)),
        }
        for metric in REG_LADDER_METRICS:
            stats = _mean_std_min_max(rows, metric)
            for name, value in stats.items():
                summary[f"{metric}_{name}"] = value
        out.append(summary)
    return out


def _reg_ladder_make_summary_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    anchor_floor_yaw_traj: Optional[float],
    pose_guard_floor: Optional[float],
) -> List[Dict[str, Any]]:
    by_seed: Dict[Tuple[float, float, int, str, int], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in raw_rows:
        key = (
            float(row["residual_l2_weight"]),
            float(row["dropout"]),
            int(row["frame_hidden_dim"]),
            str(row["variant"]),
            int(row["seed"]),
        )
        by_seed[key][str(row["partition"])] = row

    grouped: Dict[Tuple[float, float, int, str], List[Dict[str, float]]] = defaultdict(list)
    invariant_ok: Dict[Tuple[float, float, int, str], List[bool]] = defaultdict(list)
    invariant_abs_delta: Dict[Tuple[float, float, int, str], List[float]] = defaultdict(list)
    for (residual_l2_weight, dropout, frame_hidden_dim, variant, _seed), parts in by_seed.items():
        train = parts.get("train")
        test = parts.get("test")
        if train is None or test is None:
            continue
        key = (residual_l2_weight, dropout, frame_hidden_dim, variant)
        grouped[key].append(
            {
                "train_yaw_traj_mse": float(train.get("yaw_traj_mse", 0.0)),
                "test_yaw_traj_mse": float(test.get("yaw_traj_mse", 0.0)),
                "yaw_gap_mse": float(test.get("yaw_traj_mse", 0.0)) - float(train.get("yaw_traj_mse", 0.0)),
                "test_pose_nonroot_rot6d_mse": float(test.get("pose_nonroot_rot6d_mse", 0.0)),
                "test_root_pos_mse": float(test.get("root_pos_mse", 0.0)),
                "test_root_disp_mse": float(test.get("root_disp_mse", 0.0)),
                "test_contact_plan_mse": float(test.get("contact_plan_mse", 0.0)),
                "train_contact_plan_mse": float(train.get("contact_plan_mse", 0.0)),
            }
        )
        invariant_ok[key].extend([bool(train.get("root_invariant_ok", False)), bool(test.get("root_invariant_ok", False))])
        invariant_abs_delta[key].extend(
            [float(train.get("root_invariant_abs_delta", 0.0)), float(test.get("root_invariant_abs_delta", 0.0))]
        )

    out: List[Dict[str, Any]] = []
    for (residual_l2_weight, dropout, frame_hidden_dim, variant), rows in sorted(grouped.items()):
        test_yaw = _mean_std_min_max(rows, "test_yaw_traj_mse")
        test_pose = _mean_std_min_max(rows, "test_pose_nonroot_rot6d_mse")
        test_root = _mean_std_min_max(rows, "test_root_pos_mse")
        row: Dict[str, Any] = {
            "residual_l2_weight": residual_l2_weight,
            "dropout": dropout,
            "frame_hidden_dim": frame_hidden_dim,
            "variant": variant,
            "n_seeds": int(len(rows)),
            "anchor_floor_yaw_traj_mse": float(anchor_floor_yaw_traj) if anchor_floor_yaw_traj is not None else "",
            "ctx_last_hold_pose_nonroot_rot6d_mse": float(pose_guard_floor) if pose_guard_floor is not None else "",
            "crossed_anchor": (
                bool(test_yaw["mean"] <= float(anchor_floor_yaw_traj))
                if anchor_floor_yaw_traj is not None
                else False
            ),
            "pose_guard_ok": (
                bool(test_pose["mean"] <= float(pose_guard_floor))
                if pose_guard_floor is not None
                else False
            ),
            "root_pos_guard_ok_1e4": bool(test_root["mean"] <= 1.0e-4),
            "root_invariant_all_ok": bool(all(invariant_ok.get((residual_l2_weight, dropout, frame_hidden_dim, variant), []))),
            "root_invariant_abs_delta_max": (
                float(max(invariant_abs_delta[(residual_l2_weight, dropout, frame_hidden_dim, variant)]))
                if invariant_abs_delta.get((residual_l2_weight, dropout, frame_hidden_dim, variant))
                else 0.0
            ),
        }
        for key in (
            "train_yaw_traj_mse",
            "test_yaw_traj_mse",
            "yaw_gap_mse",
            "test_pose_nonroot_rot6d_mse",
            "test_root_pos_mse",
            "test_root_disp_mse",
            "test_contact_plan_mse",
            "train_contact_plan_mse",
        ):
            stats = _mean_std_min_max(rows, key)
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
            row[f"{key}_min"] = stats["min"]
            row[f"{key}_max"] = stats["max"]
        out.append(row)
    return out


def _run_reg_ladder(
    *,
    main_items: Sequence[Any],
    split: Any,
    train_batch: Mapping[str, torch.Tensor],
    test_batch: Mapping[str, torch.Tensor],
    state_norm: Any,
    aux_norm: Any,
    goal_dim: int,
    device: torch.device,
    args: argparse.Namespace,
    baseline_summary: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if int(args.yaw_seeds) < 1:
        raise RuntimeError("--reg-ladder requires --yaw-seeds >= 1")

    base_seed = int(args.yaw_seed_base) if args.yaw_seed_base is not None else int(args.seed)
    sweep_seeds = [base_seed + i for i in range(int(args.yaw_seeds))]
    anchor_floor = _summary_value(
        baseline_summary,
        variant="root_linear_to_goal_pose_hold",
        model_kind="dumb_baseline",
        partition="test",
        key="yaw_traj_mse",
    )
    pose_guard_floor = _summary_value(
        baseline_summary,
        variant="ctx_last_hold",
        model_kind="dumb_baseline",
        partition="test",
        key="pose_nonroot_rot6d_mse",
    )
    spec_by_name = {s.name: s for s in VARIANT_SPECS}
    focus = [spec_by_name[name] for name in YAW_SWEEP_FOCUS if name in spec_by_name]

    raw_rows: List[Dict[str, Any]] = []
    grid_i = 0
    for residual_l2_weight in [float(v) for v in args.reg_ladder_residual_l2_weights]:
        for dropout in [float(v) for v in args.reg_ladder_dropouts]:
            for frame_hidden_dim in [int(v) for v in args.reg_ladder_frame_hidden_dims]:
                grid_i += 1
                for seed in sweep_seeds:
                    for spec in focus:
                        spec_i = VARIANT_SPECS.index(spec)
                        print(
                            "[reg-ladder] "
                            f"grid={grid_i} residual_l2={residual_l2_weight:g} "
                            f"dropout={dropout:g} frame_hidden={frame_hidden_dim} "
                            f"seed={seed} variant={spec.name}",
                            flush=True,
                        )
                        model, _final_terms, _grad = _train_variant(
                            spec=spec,
                            train_batch=train_batch,
                            state_norm=state_norm,
                            aux_norm=aux_norm,
                            goal_dim=int(goal_dim),
                            latent_dim=int(args.latent_dim),
                            contact_embed_dim=int(args.contact_embed_dim),
                            frame_hidden_dim=int(frame_hidden_dim),
                            dropout=float(dropout),
                            residual_l2_weight=float(residual_l2_weight),
                            epochs=int(args.epochs),
                            lr=float(args.lr),
                            weight_decay=float(args.weight_decay),
                            seed=int(seed) + spec_i,
                            device=device,
                            negative_contact=str(args.negative_contact),
                            contact_loss=args.contact_loss,
                        )
                        for partition, batch, idxs in (
                            ("train", train_batch, split.train_idx),
                            ("test", test_batch, split.test_idx),
                        ):
                            pred = _predict_anchor_model(
                                model=model,
                                spec=spec,
                                batch=batch,
                                state_norm=state_norm,
                                aux_norm=aux_norm,
                                negative_contact=str(args.negative_contact),
                            )
                            rows = _metric_rows_for_predictions(
                                variant=spec.name,
                                model_kind="anchored_parallel_residual_reg_ladder",
                                partition=partition,
                                items=main_items,
                                idxs=idxs,
                                pred=pred,
                                contact_support_threshold01=args.contact_loss.contact_support_threshold01,
                            )
                            invariant = _root_pos_invariant_row(
                                variant=spec.name,
                                model_kind="anchored_parallel_residual_reg_ladder",
                                partition=partition,
                                batch=batch,
                                pred=pred,
                                metric_rows=rows,
                                atol=float(args.root_invariant_atol),
                                rtol=float(args.root_invariant_rtol),
                            )
                            raw_rows.append(
                                {
                                    "grid_i": int(grid_i),
                                    "residual_l2_weight": float(residual_l2_weight),
                                    "dropout": float(dropout),
                                    "frame_hidden_dim": int(frame_hidden_dim),
                                    "seed": int(seed),
                                    "variant": spec.name,
                                    "partition": partition,
                                    "n": int(len(rows)),
                                    "root_invariant_ok": bool(invariant["ok"]),
                                    "root_invariant_abs_delta": float(invariant["abs_delta"]),
                                    **{metric: _safe_mean(rows, metric) for metric in REG_LADDER_METRICS},
                                }
                            )

    partition_rows = _reg_ladder_summarize_partition_rows(
        raw_rows,
        anchor_floor_yaw_traj=anchor_floor,
        pose_guard_floor=pose_guard_floor,
    )
    summary_rows = _reg_ladder_make_summary_rows(
        raw_rows,
        anchor_floor_yaw_traj=anchor_floor,
        pose_guard_floor=pose_guard_floor,
    )
    return {
        "seeds": sweep_seeds,
        "anchor_floor_yaw_traj_mse": anchor_floor,
        "ctx_last_hold_pose_nonroot_rot6d_mse": pose_guard_floor,
        "raw": raw_rows,
        "partition_rows": partition_rows,
        "summary_rows": summary_rows,
    }


def _train_curve_make_summary_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    anchor_floor_yaw_traj: Optional[float],
    pose_guard_floor: Optional[float],
) -> List[Dict[str, Any]]:
    by_seed: Dict[Tuple[float, str, int], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in raw_rows:
        key = (float(row["train_subset_fraction"]), str(row["variant"]), int(row["seed"]))
        by_seed[key][str(row["partition"])] = row

    grouped: Dict[Tuple[float, str], List[Dict[str, float]]] = defaultdict(list)
    invariant_ok: Dict[Tuple[float, str], List[bool]] = defaultdict(list)
    invariant_abs_delta: Dict[Tuple[float, str], List[float]] = defaultdict(list)
    train_n_by_key: Dict[Tuple[float, str], int] = {}
    for (fraction, variant, _seed), parts in by_seed.items():
        train = parts.get("train")
        test = parts.get("test")
        if train is None or test is None:
            continue
        key = (fraction, variant)
        train_n_by_key[key] = int(train.get("train_subset_n", 0))
        grouped[key].append(
            {
                "train_yaw_traj_mse": float(train.get("yaw_traj_mse", 0.0)),
                "test_yaw_traj_mse": float(test.get("yaw_traj_mse", 0.0)),
                "yaw_gap_mse": float(test.get("yaw_traj_mse", 0.0)) - float(train.get("yaw_traj_mse", 0.0)),
                "test_pose_nonroot_rot6d_mse": float(test.get("pose_nonroot_rot6d_mse", 0.0)),
                "test_root_pos_mse": float(test.get("root_pos_mse", 0.0)),
                "test_root_disp_mse": float(test.get("root_disp_mse", 0.0)),
                "test_contact_plan_mse": float(test.get("contact_plan_mse", 0.0)),
            }
        )
        invariant_ok[key].extend([bool(train.get("root_invariant_ok", False)), bool(test.get("root_invariant_ok", False))])
        invariant_abs_delta[key].extend(
            [float(train.get("root_invariant_abs_delta", 0.0)), float(test.get("root_invariant_abs_delta", 0.0))]
        )

    out: List[Dict[str, Any]] = []
    for (fraction, variant), rows in sorted(grouped.items()):
        test_yaw = _mean_std_min_max(rows, "test_yaw_traj_mse")
        test_pose = _mean_std_min_max(rows, "test_pose_nonroot_rot6d_mse")
        test_root = _mean_std_min_max(rows, "test_root_pos_mse")
        row: Dict[str, Any] = {
            "train_subset_fraction": float(fraction),
            "train_subset_n": int(train_n_by_key.get((fraction, variant), 0)),
            "variant": variant,
            "n_seeds": int(len(rows)),
            "anchor_floor_yaw_traj_mse": float(anchor_floor_yaw_traj) if anchor_floor_yaw_traj is not None else "",
            "ctx_last_hold_pose_nonroot_rot6d_mse": float(pose_guard_floor) if pose_guard_floor is not None else "",
            "crossed_anchor": (
                bool(test_yaw["mean"] <= float(anchor_floor_yaw_traj))
                if anchor_floor_yaw_traj is not None
                else False
            ),
            "pose_guard_ok": (
                bool(test_pose["mean"] <= float(pose_guard_floor))
                if pose_guard_floor is not None
                else False
            ),
            "root_pos_guard_ok_1e4": bool(test_root["mean"] <= 1.0e-4),
            "root_invariant_all_ok": bool(all(invariant_ok.get((fraction, variant), []))),
            "root_invariant_abs_delta_max": (
                float(max(invariant_abs_delta[(fraction, variant)]))
                if invariant_abs_delta.get((fraction, variant))
                else 0.0
            ),
        }
        for key in (
            "train_yaw_traj_mse",
            "test_yaw_traj_mse",
            "yaw_gap_mse",
            "test_pose_nonroot_rot6d_mse",
            "test_root_pos_mse",
            "test_root_disp_mse",
            "test_contact_plan_mse",
        ):
            stats = _mean_std_min_max(rows, key)
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
            row[f"{key}_min"] = stats["min"]
            row[f"{key}_max"] = stats["max"]
        out.append(row)

    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in out:
        by_variant[str(row["variant"])].append(row)
    for rows in by_variant.values():
        rows.sort(key=lambda r: float(r["train_subset_fraction"]))
        for i, row in enumerate(rows):
            if i == 0:
                row["test_yaw_delta_from_prev_fraction"] = ""
                row["test_yaw_slope_from_prev_fraction"] = ""
                continue
            prev = rows[i - 1]
            delta = float(row["test_yaw_traj_mse_mean"]) - float(prev["test_yaw_traj_mse_mean"])
            denom = float(row["train_subset_fraction"]) - float(prev["train_subset_fraction"])
            row["test_yaw_delta_from_prev_fraction"] = delta
            row["test_yaw_slope_from_prev_fraction"] = delta / denom if abs(denom) > EPS else 0.0
    return out


def _train_curve_summarize_partition_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    anchor_floor_yaw_traj: Optional[float],
    pose_guard_floor: Optional[float],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[float, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(float(row["train_subset_fraction"]), str(row["variant"]), str(row["partition"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (fraction, variant, partition), rows in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "train_subset_fraction": float(fraction),
            "train_subset_n": int(rows[0].get("train_subset_n", 0)) if rows else 0,
            "variant": variant,
            "partition": partition,
            "n_seeds": int(len(rows)),
            "anchor_floor_yaw_traj_mse": float(anchor_floor_yaw_traj) if anchor_floor_yaw_traj is not None else "",
            "ctx_last_hold_pose_nonroot_rot6d_mse": float(pose_guard_floor) if pose_guard_floor is not None else "",
            "crossed_anchor": (
                bool(_mean_std_min_max(rows, "yaw_traj_mse")["mean"] <= float(anchor_floor_yaw_traj))
                if partition == "test" and anchor_floor_yaw_traj is not None
                else ""
            ),
            "pose_guard_ok": (
                bool(_mean_std_min_max(rows, "pose_nonroot_rot6d_mse")["mean"] <= float(pose_guard_floor))
                if partition == "test" and pose_guard_floor is not None
                else ""
            ),
            "root_pos_guard_ok_1e4": (
                bool(_mean_std_min_max(rows, "root_pos_mse")["mean"] <= 1.0e-4)
                if partition == "test"
                else ""
            ),
            "root_invariant_all_ok": bool(all(bool(r.get("root_invariant_ok", False)) for r in rows)),
            "root_invariant_abs_delta_max": float(max(float(r.get("root_invariant_abs_delta", 0.0)) for r in rows)),
        }
        for metric in TRAIN_CURVE_METRICS:
            stats = _mean_std_min_max(rows, metric)
            for name, value in stats.items():
                summary[f"{metric}_{name}"] = value
        out.append(summary)
    return out


def _run_train_curve(
    *,
    main_items: Sequence[Any],
    split: Any,
    test_batch: Mapping[str, torch.Tensor],
    state_norm: Any,
    aux_norm: Any,
    goal_norm: Any,
    goal_dim: int,
    device: torch.device,
    args: argparse.Namespace,
    baseline_summary: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if int(args.yaw_seeds) < 1:
        raise RuntimeError("--train-curve requires --yaw-seeds >= 1")

    base_seed = int(args.yaw_seed_base) if args.yaw_seed_base is not None else int(args.seed)
    sweep_seeds = [base_seed + i for i in range(int(args.yaw_seeds))]
    anchor_floor = _summary_value(
        baseline_summary,
        variant="root_linear_to_goal_pose_hold",
        model_kind="dumb_baseline",
        partition="test",
        key="yaw_traj_mse",
    )
    pose_guard_floor = _summary_value(
        baseline_summary,
        variant="ctx_last_hold",
        model_kind="dumb_baseline",
        partition="test",
        key="pose_nonroot_rot6d_mse",
    )
    spec_by_name = {s.name: s for s in VARIANT_SPECS}
    focus = [spec_by_name[name] for name in YAW_SWEEP_FOCUS if name in spec_by_name]

    raw_rows: List[Dict[str, Any]] = []
    for fraction in [float(v) for v in args.train_curve_fractions]:
        subset_idx = _subset_train_indices(split.train_idx, fraction)
        if not subset_idx:
            raise RuntimeError(f"empty train subset for fraction={fraction}")
        subset_batch = _batch_from_items(
            items=main_items,
            idxs=subset_idx,
            state_norm=state_norm,
            aux_norm=aux_norm,
            goal_norm=goal_norm,
            device=device,
        )
        for seed in sweep_seeds:
            for spec in focus:
                spec_i = VARIANT_SPECS.index(spec)
                print(
                    "[train-curve] "
                    f"fraction={fraction:g} train_n={len(subset_idx)} seed={seed} variant={spec.name}",
                    flush=True,
                )
                model, _final_terms, _grad = _train_variant(
                    spec=spec,
                    train_batch=subset_batch,
                    state_norm=state_norm,
                    aux_norm=aux_norm,
                    goal_dim=int(goal_dim),
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
                    negative_contact=str(args.negative_contact),
                    contact_loss=args.contact_loss,
                )
                for partition, batch, idxs in (
                    ("train", subset_batch, subset_idx),
                    ("test", test_batch, split.test_idx),
                ):
                    pred = _predict_anchor_model(
                        model=model,
                        spec=spec,
                        batch=batch,
                        state_norm=state_norm,
                        aux_norm=aux_norm,
                        negative_contact=str(args.negative_contact),
                    )
                    rows = _metric_rows_for_predictions(
                        variant=spec.name,
                        model_kind="anchored_parallel_residual_train_curve",
                        partition=partition,
                        items=main_items,
                        idxs=idxs,
                        pred=pred,
                        contact_support_threshold01=args.contact_loss.contact_support_threshold01,
                    )
                    invariant = _root_pos_invariant_row(
                        variant=spec.name,
                        model_kind="anchored_parallel_residual_train_curve",
                        partition=partition,
                        batch=batch,
                        pred=pred,
                        metric_rows=rows,
                        atol=float(args.root_invariant_atol),
                        rtol=float(args.root_invariant_rtol),
                    )
                    raw_rows.append(
                        {
                            "train_subset_fraction": float(fraction),
                            "train_subset_n": int(len(subset_idx)),
                            "seed": int(seed),
                            "variant": spec.name,
                            "partition": partition,
                            "n": int(len(rows)),
                            "root_invariant_ok": bool(invariant["ok"]),
                            "root_invariant_abs_delta": float(invariant["abs_delta"]),
                            **{metric: _safe_mean(rows, metric) for metric in TRAIN_CURVE_METRICS},
                        }
                    )

    partition_rows = _train_curve_summarize_partition_rows(
        raw_rows,
        anchor_floor_yaw_traj=anchor_floor,
        pose_guard_floor=pose_guard_floor,
    )
    summary_rows = _train_curve_make_summary_rows(
        raw_rows,
        anchor_floor_yaw_traj=anchor_floor,
        pose_guard_floor=pose_guard_floor,
    )
    scale_rows = _yaw_scale_baseline_rows(main_items=main_items, split=split, baseline_summary=baseline_summary)
    return {
        "seeds": sweep_seeds,
        "anchor_floor_yaw_traj_mse": anchor_floor,
        "ctx_last_hold_pose_nonroot_rot6d_mse": pose_guard_floor,
        "raw": raw_rows,
        "partition_rows": partition_rows,
        "summary_rows": summary_rows,
        "yaw_scale_baselines": scale_rows,
    }


def _yaw_seed_sweep(
    *,
    seeds: Sequence[int],
    main_items: Sequence[Any],
    split: Any,
    train_batch: Mapping[str, torch.Tensor],
    test_batch: Mapping[str, torch.Tensor],
    state_norm: Any,
    aux_norm: Any,
    goal_dim: int,
    device: torch.device,
    args: argparse.Namespace,
    anchor_floor_yaw_traj: Optional[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Re-train the goal/contact focus variants across seeds to put an error bar on
    the yaw_traj / contact_plan numbers (debug-only; no model/trainer/gate change)."""
    spec_by_name = {s.name: s for s in VARIANT_SPECS}
    focus = [spec_by_name[name] for name in YAW_SWEEP_FOCUS if name in spec_by_name]
    raw_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for spec in focus:
            spec_i = VARIANT_SPECS.index(spec)
            model, _final_terms, _grad = _train_variant(
                spec=spec,
                train_batch=train_batch,
                state_norm=state_norm,
                aux_norm=aux_norm,
                goal_dim=int(goal_dim),
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
                negative_contact=str(args.negative_contact),
                contact_loss=args.contact_loss,
            )
            for partition, batch, idxs in (
                ("train", train_batch, split.train_idx),
                ("test", test_batch, split.test_idx),
            ):
                pred = _predict_anchor_model(
                    model=model,
                    spec=spec,
                    batch=batch,
                    state_norm=state_norm,
                    aux_norm=aux_norm,
                    negative_contact=str(args.negative_contact),
                )
                rows = _metric_rows_for_predictions(
                    variant=spec.name,
                    model_kind="anchored_parallel_residual",
                    partition=partition,
                    items=main_items,
                    idxs=idxs,
                    pred=pred,
                    contact_support_threshold01=args.contact_loss.contact_support_threshold01,
                )
                raw_rows.append(
                    {
                        "seed": int(seed),
                        "variant": spec.name,
                        "partition": partition,
                        "n": int(len(rows)),
                        **{metric: _safe_mean(rows, metric) for metric in YAW_SWEEP_METRICS},
                    }
                )

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["variant"]), str(row["partition"]))].append(row)
    summary_rows: List[Dict[str, Any]] = []
    for (variant, partition), seed_rows in sorted(grouped.items()):
        summary: Dict[str, Any] = {"variant": variant, "partition": partition, "n_seeds": int(len(seed_rows))}
        for metric in YAW_SWEEP_METRICS:
            vals = [float(r[metric]) for r in seed_rows if math.isfinite(float(r.get(metric, float("nan"))))]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                summary[f"{metric}_mean"] = float(arr.mean())
                summary[f"{metric}_std"] = float(arr.std(ddof=1)) if len(vals) > 1 else 0.0
                summary[f"{metric}_min"] = float(arr.min())
                summary[f"{metric}_max"] = float(arr.max())
            else:
                summary[f"{metric}_mean"] = 0.0
                summary[f"{metric}_std"] = 0.0
                summary[f"{metric}_min"] = 0.0
                summary[f"{metric}_max"] = 0.0
        summary_rows.append(summary)

    if anchor_floor_yaw_traj is not None:
        summary_rows.append(
            {
                "variant": "root_linear_to_goal_pose_hold__anchor_floor",
                "partition": "test",
                "n_seeds": 0,
                "yaw_traj_mse_mean": float(anchor_floor_yaw_traj),
                "yaw_traj_mse_std": 0.0,
                "yaw_traj_mse_min": float(anchor_floor_yaw_traj),
                "yaw_traj_mse_max": float(anchor_floor_yaw_traj),
            }
        )
    return raw_rows, summary_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug-only anchored residual bridge probe")
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--reg-ladder",
        action="store_true",
        help="Run debug-only residual regularization ladder over focus variants/seeds.",
    )
    p.add_argument(
        "--train-curve",
        action="store_true",
        help="Run debug-only train-subset learning curve over focus variants/seeds.",
    )
    p.add_argument(
        "--reg-ladder-residual-l2-weights",
        type=float,
        nargs="+",
        default=(0.05, 0.2, 0.5),
        help="Residual L2 weights for --reg-ladder.",
    )
    p.add_argument(
        "--reg-ladder-dropouts",
        type=float,
        nargs="+",
        default=(0.10, 0.30),
        help="Dropout values for --reg-ladder.",
    )
    p.add_argument(
        "--reg-ladder-frame-hidden-dims",
        type=int,
        nargs="+",
        default=(128, 32),
        help="Frame hidden dimensions for --reg-ladder; pass 128 for the 6-cell first round.",
    )
    p.add_argument(
        "--train-curve-fractions",
        type=float,
        nargs="+",
        default=(0.25, 0.50, 1.0),
        help="Fractions of the fixed full training split to use for --train-curve.",
    )
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--contact-embed-dim", type=int, default=16)
    p.add_argument("--frame-hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--residual-l2-weight", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=3.0e-4)
    p.add_argument("--seed", type=int, default=20260606)
    p.add_argument("--train-fraction", type=float, default=0.60)
    p.add_argument("--block-gap", type=int, default=8)
    p.add_argument("--torch-num-threads", type=int, default=8)
    p.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    p.add_argument("--negative-contact", choices=("random",), default="random")
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
    p.add_argument(
        "--yaw-seeds",
        type=int,
        default=3,
        help="Number of seeds for the yaw_traj/contact_plan error-bar sweep (0 disables).",
    )
    p.add_argument(
        "--yaw-seed-base",
        type=int,
        default=None,
        help="Base seed for the sweep; defaults to --seed.",
    )
    args = p.parse_args()
    if bool(args.reg_ladder) and bool(args.train_curve):
        raise SystemExit("--reg-ladder and --train-curve are mutually exclusive")
    out_dir_was_explicit = any(arg == "--out-dir" or arg.startswith("--out-dir=") for arg in sys.argv[1:])
    if bool(args.reg_ladder) and not out_dir_was_explicit:
        args.out_dir = DEFAULT_REG_LADDER_OUT_DIR
    if bool(args.train_curve) and not out_dir_was_explicit:
        args.out_dir = DEFAULT_TRAIN_CURVE_OUT_DIR
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.torch_num_threads))
    device = _resolve_device(str(args.device))
    clips, contact_scaler = _load_probe_clips(args.npz_root, args.z_features)
    args.contact_loss = ContactLossConfig(
        contact_step_weight=float(args.contact_step_weight),
        contact_predict_mse_weight=float(args.contact_predict_mse_weight),
        contact_predict_bce_weight=float(args.contact_predict_bce_weight),
        contact_state_weight=float(args.contact_state_weight),
        contact_endpoint_support_weight=float(args.contact_endpoint_support_weight),
        contact_support_threshold01=_contact_threshold01_from_raw(contact_scaler, CONTACT_LABEL_THRESHOLD),
    )
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        stride=int(args.stride),
    )
    main_items = [item for item in all_items if item.clip in MATCHED_TARGETS]
    unmatched_items = [item for item in all_items if item.clip == UNMATCHED_TARGET]
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
        contact_support_threshold01=args.contact_loss.contact_support_threshold01,
    )
    per_window_rows.extend(baseline_rows)
    baseline_summary = _summary_rows(baseline_rows)

    if bool(args.reg_ladder):
        reg_ladder = _run_reg_ladder(
            main_items=main_items,
            split=split,
            train_batch=train_batch,
            test_batch=test_batch,
            state_norm=state_norm,
            aux_norm=aux_norm,
            goal_dim=goal_dim,
            device=device,
            args=args,
            baseline_summary=baseline_summary,
        )
        payload: Dict[str, Any] = {
            "task": "between_anchor_residual_reg_ladder",
            "status": "debug_reg_ladder",
            "flags": {
                "horizon": int(args.horizon),
                "context_len": int(args.context_len),
                "epochs": int(args.epochs),
                "latent_dim": int(args.latent_dim),
                "contact_embed_dim": int(args.contact_embed_dim),
                "reg_ladder_residual_l2_weights": [float(v) for v in args.reg_ladder_residual_l2_weights],
                "reg_ladder_dropouts": [float(v) for v in args.reg_ladder_dropouts],
                "reg_ladder_frame_hidden_dims": [int(v) for v in args.reg_ladder_frame_hidden_dims],
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "seed": int(args.seed),
                "yaw_seeds": int(args.yaw_seeds),
                "yaw_seed_base": int(args.yaw_seed_base) if args.yaw_seed_base is not None else int(args.seed),
                "device": str(device),
                "contact_step_weight": float(args.contact_loss.contact_step_weight),
                "contact_predict_mse_weight": float(args.contact_loss.contact_predict_mse_weight),
                "contact_predict_bce_weight": float(args.contact_loss.contact_predict_bce_weight),
                "contact_state_weight": float(args.contact_loss.contact_state_weight),
                "contact_endpoint_support_weight": float(args.contact_loss.contact_endpoint_support_weight),
                "root_invariant_atol": float(args.root_invariant_atol),
                "root_invariant_rtol": float(args.root_invariant_rtol),
            },
            "schema": {
                "ctx_state": {
                    "shape": [len(split.train_idx), int(args.context_len), STATE_DIM],
                    "dtype": "float32",
                    "device": str(device),
                },
                "goal_dim": goal_dim,
                "soft_contact": {
                    "shape": [len(split.train_idx), int(args.horizon), 2],
                    "dtype": "float32",
                    "device": str(device),
                    "range": [0.0, 1.0],
                },
                "state_output": {
                    "shape": [len(split.train_idx), int(args.horizon), STATE_DIM],
                    "dtype": "float32",
                    "device": str(device),
                },
                "bone_angvel_output": {
                    "shape": [len(split.train_idx), int(args.horizon), ANGVEL_DIM],
                    "dtype": "float32",
                    "device": str(device),
                },
            },
            "dataset": {
                "matched_targets": list(MATCHED_TARGETS),
                "matched_window_count": int(len(main_items)),
                "unmatched_target": UNMATCHED_TARGET,
                "unmatched_window_count": int(len(unmatched_items)),
                "contact_scaler": contact_scaler.stats(),
            },
            "split": {
                "name": split.name,
                "kind": split.kind,
                "train_n": int(len(split.train_idx)),
                "test_n": int(len(split.test_idx)),
                "note": split.note,
            },
            "baseline_summaries": baseline_summary,
            "reg_ladder": reg_ladder,
            "artifacts": {
                "summary_json": str(args.out_dir / "summary.json"),
                "baselines_csv": str(args.out_dir / "baselines.csv"),
                "reg_ladder_csv": str(args.out_dir / "reg_ladder.csv"),
                "reg_ladder_summary_csv": str(args.out_dir / "reg_ladder_summary.csv"),
                "reg_ladder_seed_rows_csv": str(args.out_dir / "reg_ladder_seed_rows.csv"),
            },
        }
        _write_csv(args.out_dir / "baselines.csv", baseline_summary)
        _write_csv(args.out_dir / "reg_ladder.csv", reg_ladder["partition_rows"])
        _write_csv(args.out_dir / "reg_ladder_summary.csv", reg_ladder["summary_rows"])
        _write_csv(args.out_dir / "reg_ladder_seed_rows.csv", reg_ladder["raw"])
        _write_json(args.out_dir / "summary.json", payload)
        print(f"[OK] wrote {args.out_dir / 'summary.json'}")
        return

    if bool(args.train_curve):
        train_curve = _run_train_curve(
            main_items=main_items,
            split=split,
            test_batch=test_batch,
            state_norm=state_norm,
            aux_norm=aux_norm,
            goal_norm=goal_norm,
            goal_dim=goal_dim,
            device=device,
            args=args,
            baseline_summary=baseline_summary,
        )
        payload = {
            "task": "between_anchor_residual_train_curve",
            "status": "debug_train_subset_learning_curve",
            "flags": {
                "horizon": int(args.horizon),
                "context_len": int(args.context_len),
                "epochs": int(args.epochs),
                "latent_dim": int(args.latent_dim),
                "contact_embed_dim": int(args.contact_embed_dim),
                "frame_hidden_dim": int(args.frame_hidden_dim),
                "dropout": float(args.dropout),
                "residual_l2_weight": float(args.residual_l2_weight),
                "train_curve_fractions": [float(v) for v in args.train_curve_fractions],
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "seed": int(args.seed),
                "yaw_seeds": int(args.yaw_seeds),
                "yaw_seed_base": int(args.yaw_seed_base) if args.yaw_seed_base is not None else int(args.seed),
                "device": str(device),
                "contact_step_weight": float(args.contact_loss.contact_step_weight),
                "contact_predict_mse_weight": float(args.contact_loss.contact_predict_mse_weight),
                "contact_predict_bce_weight": float(args.contact_loss.contact_predict_bce_weight),
                "contact_state_weight": float(args.contact_loss.contact_state_weight),
                "contact_endpoint_support_weight": float(args.contact_loss.contact_endpoint_support_weight),
                "root_invariant_atol": float(args.root_invariant_atol),
                "root_invariant_rtol": float(args.root_invariant_rtol),
            },
            "schema": {
                "full_train_ctx_state": {
                    "shape": [len(split.train_idx), int(args.context_len), STATE_DIM],
                    "dtype": "float32",
                    "device": str(device),
                },
                "test_ctx_state": {
                    "shape": [len(split.test_idx), int(args.context_len), STATE_DIM],
                    "dtype": "float32",
                    "device": str(device),
                },
                "state_output": {
                    "shape": ["train_subset_n_or_test_n", int(args.horizon), STATE_DIM],
                    "dtype": "float32",
                    "device": str(device),
                },
                "bone_angvel_output": {
                    "shape": ["train_subset_n_or_test_n", int(args.horizon), ANGVEL_DIM],
                    "dtype": "float32",
                    "device": str(device),
                },
                "soft_contact": {
                    "shape": ["train_subset_n_or_test_n", int(args.horizon), 2],
                    "dtype": "float32",
                    "device": str(device),
                    "range": [0.0, 1.0],
                },
            },
            "dataset": {
                "matched_targets": list(MATCHED_TARGETS),
                "matched_window_count": int(len(main_items)),
                "unmatched_target": UNMATCHED_TARGET,
                "unmatched_window_count": int(len(unmatched_items)),
                "contact_scaler": contact_scaler.stats(),
            },
            "split": {
                "name": split.name,
                "kind": split.kind,
                "full_train_n": int(len(split.train_idx)),
                "test_n": int(len(split.test_idx)),
                "note": split.note,
                "normalizer_fit": "full_train_split",
                "test_set": "fixed_full_split_test",
            },
            "baseline_summaries": baseline_summary,
            "train_curve": train_curve,
            "artifacts": {
                "summary_json": str(args.out_dir / "summary.json"),
                "baselines_csv": str(args.out_dir / "baselines.csv"),
                "train_curve_csv": str(args.out_dir / "train_curve.csv"),
                "train_curve_summary_csv": str(args.out_dir / "train_curve_summary.csv"),
                "train_curve_seed_rows_csv": str(args.out_dir / "train_curve_seed_rows.csv"),
                "yaw_scale_baselines_csv": str(args.out_dir / "yaw_scale_baselines.csv"),
            },
        }
        _write_csv(args.out_dir / "baselines.csv", baseline_summary)
        _write_csv(args.out_dir / "train_curve.csv", train_curve["partition_rows"])
        _write_csv(args.out_dir / "train_curve_summary.csv", train_curve["summary_rows"])
        _write_csv(args.out_dir / "train_curve_seed_rows.csv", train_curve["raw"])
        _write_csv(args.out_dir / "yaw_scale_baselines.csv", train_curve["yaw_scale_baselines"])
        _write_json(args.out_dir / "summary.json", payload)
        print(f"[OK] wrote {args.out_dir / 'summary.json'}")
        return

    root_invariant_rows: List[Dict[str, Any]] = []
    grad_rows: List[Dict[str, Any]] = []
    variant_meta_rows: List[Dict[str, Any]] = []

    for spec_i, spec in enumerate(VARIANT_SPECS):
        model, final_terms, grad_usage = _train_variant(
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
            seed=int(args.seed) + spec_i,
            device=device,
            negative_contact=str(args.negative_contact),
            contact_loss=args.contact_loss,
        )
        grad_rows.append(grad_usage)
        params = int(sum(p.numel() for p in model.parameters()))
        variant_meta_rows.append(
            {
                "variant": spec.name,
                "model_kind": "anchored_parallel_residual",
                "use_goal": bool(spec.use_goal),
                "contact_mode": spec.contact_mode,
                "role": spec.role,
                "runtime_status": spec.runtime_status,
                "parameter_count": params,
                "epochs": int(args.epochs),
                "contact_step_weight": float(args.contact_loss.contact_step_weight),
                "contact_predict_mse_weight": float(args.contact_loss.contact_predict_mse_weight),
                "contact_predict_bce_weight": float(args.contact_loss.contact_predict_bce_weight),
                "contact_state_weight": float(args.contact_loss.contact_state_weight),
                "contact_endpoint_support_weight": float(args.contact_loss.contact_endpoint_support_weight),
                **{f"final_train_{k}": v for k, v in final_terms.items()},
            }
        )
        for partition, batch, idxs in (
            ("train", train_batch, split.train_idx),
            ("test", test_batch, split.test_idx),
        ):
            pred = _predict_anchor_model(
                model=model,
                spec=spec,
                batch=batch,
                state_norm=state_norm,
                aux_norm=aux_norm,
                negative_contact=str(args.negative_contact),
            )
            rows = _metric_rows_for_predictions(
                variant=spec.name,
                model_kind="anchored_parallel_residual",
                partition=partition,
                items=main_items,
                idxs=idxs,
                pred=pred,
                contact_support_threshold01=args.contact_loss.contact_support_threshold01,
            )
            root_invariant_rows.append(
                _root_pos_invariant_row(
                    variant=spec.name,
                    model_kind="anchored_parallel_residual",
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
    learned_summary = [r for r in summary_rows if r.get("model_kind") == "anchored_parallel_residual"]
    baseline_summary = [r for r in summary_rows if r.get("model_kind") == "dumb_baseline"]
    meta_by_variant = {r["variant"]: r for r in variant_meta_rows}
    variant_csv_rows: List[Dict[str, Any]] = []
    for row in learned_summary:
        merged = dict(meta_by_variant.get(row["variant"], {}))
        merged.update(row)
        variant_csv_rows.append(merged)

    payload: Dict[str, Any] = {
        "task": "between_anchor_residual_probe",
        "status": "debug_train_fit",
        "flags": {
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "epochs": int(args.epochs),
            "latent_dim": int(args.latent_dim),
            "contact_embed_dim": int(args.contact_embed_dim),
            "frame_hidden_dim": int(args.frame_hidden_dim),
            "dropout": float(args.dropout),
            "residual_l2_weight": float(args.residual_l2_weight),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "device": str(device),
            "contact_step_weight": float(args.contact_loss.contact_step_weight),
            "contact_predict_mse_weight": float(args.contact_loss.contact_predict_mse_weight),
            "contact_predict_bce_weight": float(args.contact_loss.contact_predict_bce_weight),
            "contact_state_weight": float(args.contact_loss.contact_state_weight),
            "contact_endpoint_support_weight": float(args.contact_loss.contact_endpoint_support_weight),
            "root_invariant_atol": float(args.root_invariant_atol),
            "root_invariant_rtol": float(args.root_invariant_rtol),
        },
        "schema": {
            "ctx_state": {"shape": [len(split.train_idx), int(args.context_len), STATE_DIM], "dtype": "float32", "device": str(device)},
            "goal_dim": goal_dim,
            "soft_contact": {"shape": [len(split.train_idx), int(args.horizon), 2], "dtype": "float32", "device": str(device), "range": [0.0, 1.0]},
            "state_output": {"shape": [len(split.train_idx), int(args.horizon), STATE_DIM], "dtype": "float32", "device": str(device)},
            "bone_angvel_output": {"shape": [len(split.train_idx), int(args.horizon), ANGVEL_DIM], "dtype": "float32", "device": str(device)},
        },
        "dataset": {
            "matched_targets": list(MATCHED_TARGETS),
            "matched_window_count": int(len(main_items)),
            "unmatched_target": UNMATCHED_TARGET,
            "unmatched_window_count": int(len(unmatched_items)),
            "contact_scaler": contact_scaler.stats(),
        },
        "split": {
            "name": split.name,
            "kind": split.kind,
            "train_n": int(len(split.train_idx)),
            "test_n": int(len(split.test_idx)),
            "note": split.note,
        },
        "variant_summaries": variant_csv_rows,
        "baseline_summaries": baseline_summary,
        "grad_usage": grad_rows,
        "root_invariant": root_invariant_rows,
        "artifacts": {
            "summary_json": str(args.out_dir / "summary.json"),
            "variants_csv": str(args.out_dir / "variants.csv"),
            "baselines_csv": str(args.out_dir / "baselines.csv"),
            "grad_usage_csv": str(args.out_dir / "grad_usage.csv"),
            "root_invariant_csv": str(args.out_dir / "root_invariant.csv"),
            "per_window_csv": str(args.out_dir / "per_window.csv"),
        },
    }

    if int(args.yaw_seeds) >= 1:
        base_seed = int(args.yaw_seed_base) if args.yaw_seed_base is not None else int(args.seed)
        sweep_seeds = [base_seed + i for i in range(int(args.yaw_seeds))]
        anchor_floor = next(
            (
                float(r.get("yaw_traj_mse", 0.0))
                for r in baseline_summary
                if str(r.get("variant")) == "root_linear_to_goal_pose_hold" and str(r.get("partition")) == "test"
            ),
            None,
        )
        yaw_sweep_raw, yaw_sweep_summary = _yaw_seed_sweep(
            seeds=sweep_seeds,
            main_items=main_items,
            split=split,
            train_batch=train_batch,
            test_batch=test_batch,
            state_norm=state_norm,
            aux_norm=aux_norm,
            goal_dim=goal_dim,
            device=device,
            args=args,
            anchor_floor_yaw_traj=anchor_floor,
        )
        payload["yaw_seed_sweep"] = {"seeds": sweep_seeds, "raw": yaw_sweep_raw, "summary": yaw_sweep_summary}
        payload["artifacts"]["yaw_seed_sweep_csv"] = str(args.out_dir / "yaw_seed_sweep.csv")
        payload["artifacts"]["yaw_seed_sweep_summary_csv"] = str(args.out_dir / "yaw_seed_sweep_summary.csv")
        _write_csv(args.out_dir / "yaw_seed_sweep.csv", yaw_sweep_raw)
        _write_csv(args.out_dir / "yaw_seed_sweep_summary.csv", yaw_sweep_summary)

    _write_csv(args.out_dir / "per_window.csv", per_window_rows)
    _write_csv(args.out_dir / "variants.csv", variant_csv_rows)
    _write_csv(args.out_dir / "baselines.csv", baseline_summary)
    _write_csv(args.out_dir / "grad_usage.csv", grad_rows)
    _write_csv(args.out_dir / "root_invariant.csv", root_invariant_rows)
    _write_json(args.out_dir / "summary.json", payload)
    print(f"[OK] wrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
