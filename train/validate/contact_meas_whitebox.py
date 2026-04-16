from __future__ import annotations

import math as _math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from train.geometry import fk_positions_from_rot6d, reproject_rot6d
from train.layout import resolve_layout_slice


_STATE_UPDATE_UNSET = object()


@dataclass(frozen=True)
class ContactMeasRuntime:
    x_raw: torch.Tensor
    contact_dim: int
    rot_slice: slice
    rot_flat: torch.Tensor
    joint_count: int
    parents: Any
    offsets: torch.Tensor
    root_pos: torch.Tensor
    up_axis: int


def _resolve_contact_meas_runtime(trainer: Any, x_raw) -> Optional[ContactMeasRuntime]:
    if x_raw is None or (not torch.is_tensor(x_raw)):
        return None
    if x_raw.dim() == 3 and x_raw.size(1) == 1:
        x_raw = x_raw[:, 0]
    if x_raw.dim() != 2:
        return None

    model = getattr(trainer, "model", None)
    contact_dim = int(getattr(model, "contact_dim", 0) or 0) if model is not None else 0
    if contact_dim <= 0:
        return None

    x_layout = getattr(trainer, "_x_layout", None) or {}
    rot_slice = resolve_layout_slice(x_layout, "BoneRotations6D", positive_only=True)
    if not isinstance(rot_slice, slice):
        return None
    rot_flat = x_raw[..., rot_slice]
    if rot_flat.numel() == 0 or (rot_flat.shape[-1] % 6 != 0):
        return None
    joint_count = int(rot_flat.shape[-1] // 6)

    loss_fn = getattr(trainer, "loss_fn", None)
    parents = getattr(loss_fn, "parents", None)
    offsets = getattr(loss_fn, "bone_offsets", None)
    if not parents or offsets is None:
        return None
    if offsets.shape[0] < joint_count:
        return None

    root_slice = resolve_layout_slice(x_layout, "RootPosition", positive_only=True)
    if isinstance(root_slice, slice) and (root_slice.stop - root_slice.start) == 3:
        root_pos = x_raw[..., root_slice]
    else:
        root_pos = x_raw.new_zeros((x_raw.shape[0], 3))

    up_axis = int(getattr(trainer, "eval_up_axis", getattr(trainer, "_up_axis", 2)))
    up_axis = 2 if up_axis not in (0, 1, 2) else up_axis
    return ContactMeasRuntime(
        x_raw=x_raw,
        contact_dim=contact_dim,
        rot_slice=rot_slice,
        rot_flat=rot_flat,
        joint_count=joint_count,
        parents=parents,
        offsets=offsets,
        root_pos=root_pos,
        up_axis=up_axis,
    )


def _resolve_contact_meas_cfg(trainer: Any) -> dict[str, Any]:
    cfg = getattr(trainer, "_contact_meas_cfg", None)
    if not isinstance(cfg, dict):
        loss_fn = getattr(trainer, "loss_fn", None)
        meta = getattr(loss_fn, "meta", None)
        foot_evidence = (meta.get("foot_evidence") if isinstance(meta, dict) else {}) or {}
        sweep = (foot_evidence.get("sweep") if isinstance(foot_evidence, dict) else {}) or {}
        spec = (foot_evidence.get("soft_score_spec") if isinstance(foot_evidence, dict) else {}) or {}

        def _finite_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
            if not isinstance(mapping, Mapping):
                return default
            try:
                value = float(mapping.get(key, default))
            except (TypeError, ValueError):
                value = default
            return default if not _math.isfinite(value) else value

        radius_cm = _finite_float(sweep, "sphere_radius_cm", 0.0)
        up_offset_cm = _finite_float(sweep, "up_offset_cm", 0.0)
        down_distance_cm = _finite_float(sweep, "down_distance_cm", 0.0)
        cfg = {
            "radius_m": max(0.0, radius_cm) / 100.0,
            "up_offset_m": max(0.0, up_offset_cm) / 100.0,
            "down_distance_m": max(0.0, down_distance_cm) / 100.0,
            "dist0_cm": max(0.0, _finite_float(spec, "dist0_cm", 0.5)),
            "alpha_dist": max(1e-6, _finite_float(spec, "alpha_dist", 2.0)),
            "vz0_cmps": max(1e-6, _finite_float(spec, "vz0_cmps", 40.0)),
            "alpha_vz": max(1e-6, _finite_float(spec, "alpha_vz", 0.5)),
            "vxy0_cmps": max(1e-6, _finite_float(spec, "vxy0_cmps", 96.0)),
            "alpha_vxy": max(1e-6, _finite_float(spec, "alpha_vxy", 0.2)),
            "gate_by_hit": bool(spec.get("gate_by_hit", True)) if isinstance(spec, dict) else True,
            "min_score": 1e-4,
            "max_score": 0.9,
            "scale": 0.92,
        }
        trainer._contact_meas_cfg = cfg

    gate_override = getattr(trainer, "contact_meas_gate_by_hit_override", None)
    if gate_override is not None:
        cfg["gate_by_hit"] = bool(gate_override)
    return cfg


def _resolve_contact_meas_bone_names(trainer: Any, joint_count: int) -> list[str]:
    loss_fn = getattr(trainer, "loss_fn", None)
    bone_names_src = getattr(loss_fn, "bone_names", None) or getattr(trainer, "_bone_names", None)
    if not bone_names_src:
        meta = getattr(loss_fn, "meta", None)
        if isinstance(meta, dict):
            bone_names_src = meta.get("bone_names") or meta.get("skeleton", {}).get("bone_names")
    bone_names = [str(name) for name in bone_names_src] if isinstance(bone_names_src, (list, tuple)) else []
    return bone_names[:joint_count] if joint_count > 0 else bone_names


def _resolve_contact_meas_foot_indices(
    trainer: Any,
    *,
    bone_names: Sequence[str],
    joint_count: int,
    contact_dim: int,
) -> Optional[list[int]]:
    foot_indices = getattr(trainer, "_contact_meas_foot_idxs", None)
    if not isinstance(foot_indices, (list, tuple)) or len(foot_indices) != contact_dim:
        foot_indices = None
    if foot_indices is not None:
        return [int(idx) for idx in foot_indices]

    resolved: list[int] = []
    name_to_idx = {name: idx for idx, name in enumerate(bone_names[:joint_count])} if bone_names else {}
    loss_fn = getattr(trainer, "loss_fn", None)
    meta = getattr(loss_fn, "meta", None)
    if isinstance(meta, dict):
        markers = meta.get("foot_evidence", {}).get("markers")
        if isinstance(markers, str):
            for name in [item.strip() for item in markers.split(",") if item.strip()]:
                idx = name_to_idx.get(name)
                if isinstance(idx, int):
                    resolved.append(int(idx))
    for name in ("ball_l", "ball_r", "foot_l", "foot_r"):
        if len(resolved) >= contact_dim:
            break
        idx = name_to_idx.get(name)
        if isinstance(idx, int) and 0 <= idx < joint_count and idx not in resolved:
            resolved.append(int(idx))
    if len(resolved) != contact_dim:
        return None
    trainer._contact_meas_foot_idxs = list(resolved)
    return resolved


def _compute_contact_meas_ground_z(
    trainer: Any,
    *,
    ground_z_now: torch.Tensor,
    prev_foot_pos,
    ground_z_prev,
    ground_z_hist,
) -> tuple[torch.Tensor, Any, Any]:
    mode = str(getattr(trainer, "contact_meas_ground_z_mode", "window") or "window").strip().lower()
    if mode not in ("ema", "window", "slew"):
        mode = "window"
    hist_update = _STATE_UPDATE_UNSET
    prev_ok = torch.is_tensor(ground_z_prev) and ground_z_prev.shape == ground_z_now.shape and prev_foot_pos is not None
    if not prev_ok:
        if mode == "window":
            win = max(1, int(getattr(trainer, "contact_meas_ground_z_window", 5) or 5))
            hist_update = ground_z_now.unsqueeze(-1).repeat(1, win).detach() if win > 1 else None
        return ground_z_now, ground_z_prev, hist_update

    ground_z_prev = ground_z_prev.to(device=ground_z_now.device, dtype=ground_z_now.dtype)
    ground_z_cand = ground_z_now
    if mode == "ema":
        beta = float(getattr(trainer, "contact_meas_ground_z_beta", 0.05) or 0.05)
        if (not _math.isfinite(beta)) or beta <= 0.0:
            beta = 0.05
        ground_z_cand = ground_z_prev + min(1.0, beta) * (ground_z_now - ground_z_prev)
    elif mode == "window":
        win = max(1, int(getattr(trainer, "contact_meas_ground_z_window", 5) or 5))
        q = float(getattr(trainer, "contact_meas_ground_z_quantile", 0.2) or 0.2)
        if (not _math.isfinite(q)) or q < 0.0:
            q = 0.0
        q = min(1.0, q)
        hist = ground_z_hist
        if (not torch.is_tensor(hist)) or hist.shape != (ground_z_now.shape[0], win):
            hist = ground_z_prev.unsqueeze(-1).repeat(1, win)
        else:
            hist = hist.to(device=ground_z_now.device, dtype=ground_z_now.dtype)
        if win > 1:
            hist = torch.roll(hist, shifts=-1, dims=-1)
            hist[..., -1] = ground_z_now
            hist_update = hist.detach()
        try:
            vals = hist.sort(dim=-1).values
            q_idx = int(_math.ceil(q * float(win - 1)))
            ground_z_cand = vals[..., max(0, min(win - 1, q_idx))]
        except (RuntimeError, TypeError, ValueError):
            ground_z_cand = ground_z_prev

    max_down = getattr(trainer, "contact_meas_ground_z_max_down_m", None)
    max_up = getattr(trainer, "contact_meas_ground_z_max_up_m", None)
    try:
        max_down_m = float(max_down) if max_down is not None else 0.0
    except (TypeError, ValueError):
        max_down_m = 0.0
    try:
        max_up_m = float(max_up) if max_up is not None else 0.0
    except (TypeError, ValueError):
        max_up_m = 0.0
    if mode == "slew" and max_down_m <= 0.0 and max_up_m <= 0.0:
        max_down_m, max_up_m = 0.01, 0.002
    if max_down_m > 0.0 or max_up_m > 0.0:
        delta = (ground_z_cand - ground_z_prev).clamp(-max(0.0, max_down_m), max(0.0, max_up_m))
        return ground_z_prev + delta, ground_z_prev, hist_update
    return ground_z_cand, ground_z_prev, hist_update


def _compute_contact_meas_whitebox_state(
    trainer: Any,
    runtime: ContactMeasRuntime,
    foot_idxs: Sequence[int],
    prev_foot_pos,
    cfg: Mapping[str, Any],
    *,
    prev_root_pos,
    ground_z_prev,
    ground_z_hist,
) -> SimpleNamespace:
    loss_fn = getattr(trainer, "loss_fn", None)
    cols = getattr(loss_fn, "_rot6d_columns", ("X", "Z"))
    rot_proj = reproject_rot6d(runtime.rot_flat).view(runtime.x_raw.shape[0], runtime.joint_count, 6)
    pos = fk_positions_from_rot6d(
        rot_proj,
        runtime.parents,
        runtime.offsets,
        root_pos=runtime.root_pos,
        columns=cols,
    )
    foot_pos = pos[:, torch.as_tensor(foot_idxs, device=pos.device, dtype=torch.long)]
    fps = float(getattr(trainer, "fps", 60.0) or 60.0)
    has_prev = torch.is_tensor(prev_foot_pos) and prev_foot_pos.shape == foot_pos.shape
    vel = torch.zeros_like(foot_pos)
    if has_prev:
        vel = (foot_pos - prev_foot_pos.to(device=foot_pos.device, dtype=foot_pos.dtype)) * fps
    root_vel = torch.zeros_like(runtime.root_pos)
    if has_prev and torch.is_tensor(prev_root_pos) and prev_root_pos.shape == runtime.root_pos.shape:
        root_vel = (runtime.root_pos - prev_root_pos.to(device=runtime.root_pos.device, dtype=runtime.root_pos.dtype)) * fps

    planar_axes = [axis for axis in range(3) if axis != runtime.up_axis]
    vel_xy = vel[..., planar_axes]
    root_vel_xy = root_vel[..., planar_axes]
    vxy_abs_mps = vel_xy.norm(dim=-1)
    vxy_rel_mps = (vel_xy - root_vel_xy.unsqueeze(-2)).norm(dim=-1)
    vz_mps = vel[..., runtime.up_axis].abs()

    vxy_mode = getattr(trainer, "contact_meas_vxy_mode", None)
    if vxy_mode is None and isinstance(cfg, Mapping):
        vxy_mode = cfg.get("vxy_mode")
    vxy_mode = str(vxy_mode or "abs").strip().lower()
    use_root_rel = vxy_mode in ("root", "root_rel", "root-relative", "rel", "relative")
    vxy_mps_used = vxy_rel_mps if use_root_rel else vxy_abs_mps
    vxy_mode = "root_rel" if use_root_rel else "abs"

    vz_cmps = vz_mps * 100.0
    vxy_abs_cmps = vxy_abs_mps * 100.0
    vxy_rel_cmps = vxy_rel_mps * 100.0
    vxy_cmps = vxy_mps_used * 100.0
    root_vxy_cmps = root_vel_xy.norm(dim=-1) * 100.0

    vz0_cmps = max(1e-6, float(cfg.get("vz0_cmps", 40.0) or 40.0))
    alpha_vz = float(cfg.get("alpha_vz", 0.5) or 0.5)
    if (not _math.isfinite(alpha_vz)) or alpha_vz <= 0.0:
        alpha_vz = 0.5
    vz_score = torch.exp(-torch.relu(vz_cmps - vz0_cmps) / max(1e-6, alpha_vz * vz0_cmps))

    vxy0_cmps = max(1e-6, float(cfg.get("vxy0_cmps", 96.0) or 96.0))
    alpha_vxy = float(cfg.get("alpha_vxy", 0.2) or 0.2)
    if (not _math.isfinite(alpha_vxy)) or alpha_vxy <= 0.0:
        alpha_vxy = 0.2
    vxy_score = torch.exp(-torch.relu(vxy_cmps - vxy0_cmps) / max(1e-6, alpha_vxy * vxy0_cmps))

    radius_m = float(cfg.get("radius_m", 0.0) or 0.0)
    bottom_z = foot_pos[..., runtime.up_axis] - radius_m
    stance_idx = (vxy_score * vz_score).detach().argmax(dim=-1)
    ground_z_now = bottom_z.gather(-1, stance_idx.unsqueeze(-1)).squeeze(-1)
    ground_z, ground_z_prev_tensor, ground_z_hist_update = _compute_contact_meas_ground_z(
        trainer,
        ground_z_now=ground_z_now,
        prev_foot_pos=prev_foot_pos,
        ground_z_prev=ground_z_prev,
        ground_z_hist=ground_z_hist,
    )
    dist_to_ground_m = (bottom_z - ground_z.unsqueeze(-1)).clamp_min(0.0)

    start_z = sweep_target_z = hit_flag = None
    if bool(cfg.get("gate_by_hit", True)):
        up_off = float(cfg.get("up_offset_m", 0.0) or 0.0)
        down_dist = float(cfg.get("down_distance_m", 0.0) or 0.0)
        start_z = foot_pos[..., runtime.up_axis] + up_off
        sweep_target_z = ground_z.unsqueeze(-1) + radius_m
        hit_flag = (start_z >= sweep_target_z) & ((start_z - down_dist) <= sweep_target_z)

    dist_cm = dist_to_ground_m * 100.0
    dist0_cm = max(1e-6, float(cfg.get("dist0_cm", 0.5) or 0.5))
    alpha_dist = float(cfg.get("alpha_dist", 2.0) or 2.0)
    if (not _math.isfinite(alpha_dist)) or alpha_dist <= 0.0:
        alpha_dist = 2.0
    dist_raw = torch.sigmoid((alpha_dist * (dist0_cm - dist_cm)) / dist0_cm)
    dist_score = (dist_raw / max(1e-6, 1.0 / (1.0 + _math.exp(-alpha_dist)))).clamp(0.0, 1.0)

    contacts_meas = dist_score * vz_score * vxy_score
    scale = float(cfg.get("scale", 1.0) or 1.0)
    if _math.isfinite(scale) and scale > 0.0:
        contacts_meas = contacts_meas * scale
    if hit_flag is not None:
        contacts_meas = contacts_meas * hit_flag.to(dtype=contacts_meas.dtype)
    contacts_meas = contacts_meas.clamp(0.0, float(cfg.get("max_score", 1.0) or 1.0))
    min_score = float(cfg.get("min_score", 0.0) or 0.0)
    if min_score > 0.0:
        contacts_meas = (
            torch.where(hit_flag, contacts_meas.clamp_min(min_score), contacts_meas)
            if hit_flag is not None
            else contacts_meas.clamp_min(min_score)
        )

    return SimpleNamespace(
        contacts_meas=contacts_meas,
        foot_pos=foot_pos,
        root_pos=runtime.root_pos,
        ground_z=ground_z.detach(),
        ground_z_hist=ground_z_hist_update,
        debug_ctx={
            "up_axis": runtime.up_axis,
            "contact_dim": runtime.contact_dim,
            "vxy_mode": vxy_mode,
            "ground_z_now": ground_z_now,
            "ground_z": ground_z,
            "ground_z_prev": ground_z_prev_tensor,
            "foot_pos_z": foot_pos[..., runtime.up_axis],
            "bottom_z": bottom_z,
            "dist_cm": dist_cm,
            "vz_cmps": vz_cmps,
            "root_vxy_cmps": root_vxy_cmps,
            "vxy_cmps": vxy_cmps,
            "vxy_abs_cmps": vxy_abs_cmps,
            "vxy_rel_cmps": vxy_rel_cmps,
            "start_z": start_z,
            "sweep_target_z": sweep_target_z,
            "hit_flag": hit_flag,
            "dist_score": dist_score,
            "vz_score": vz_score,
            "vxy_score": vxy_score,
        },
    )


def _build_contact_meas_whitebox_debug(
    *,
    cfg: Mapping[str, Any],
    bone_names: Sequence[str],
    foot_idxs: Sequence[int],
    state: SimpleNamespace,
) -> Dict[str, Any]:
    def _mean_list(x: Optional[torch.Tensor]) -> Optional[list[float]]:
        if x is None or not torch.is_tensor(x):
            return None
        if x.ndim == 1:
            return [float(x.detach().mean().item())]
        if x.ndim == 2:
            return x.detach().mean(dim=0).cpu().tolist()
        return x.detach().reshape(x.shape[0], -1).mean(dim=0).cpu().tolist()

    wb_cfg: Optional[Dict[str, Any]] = {}
    for key, value in (cfg or {}).items():
        if isinstance(value, bool):
            wb_cfg[str(key)] = bool(value)
            continue
        try:
            wb_cfg[str(key)] = float(value)
        except (TypeError, ValueError):
            wb_cfg = None
            break

    foot_names = [bone_names[int(idx)] if int(idx) < len(bone_names) else str(int(idx)) for idx in foot_idxs] if bone_names else None
    debug = state.debug_ctx
    hit_flag = debug["hit_flag"]
    return {
        "UpAxis": int(debug["up_axis"]),
        "Batch": int(state.foot_pos.shape[0]),
        "ContactDim": int(debug["contact_dim"]),
        "FootIdxs": [int(idx) for idx in foot_idxs],
        "FootNames": foot_names,
        "VxyMode": str(debug["vxy_mode"]),
        "GroundZSelect": "stance",
        "Cfg": wb_cfg,
        "GroundZNowMean": float(debug["ground_z_now"].detach().mean().item()),
        "GroundZMean": float(debug["ground_z"].detach().mean().item()),
        "GroundZPrevMean": float(debug["ground_z_prev"].detach().mean().item()) if torch.is_tensor(debug["ground_z_prev"]) else None,
        "FootPosZMean": _mean_list(debug["foot_pos_z"]),
        "BottomZMean": _mean_list(debug["bottom_z"]),
        "DistCmMean": _mean_list(debug["dist_cm"]),
        "VzCmpsMean": _mean_list(debug["vz_cmps"]),
        "RootVxyCmpsMean": _mean_list(debug["root_vxy_cmps"]),
        "VxyCmpsMean": _mean_list(debug["vxy_cmps"]),
        "VxyAbsCmpsMean": _mean_list(debug["vxy_abs_cmps"]),
        "VxyRelCmpsMean": _mean_list(debug["vxy_rel_cmps"]),
        "StartZMean": _mean_list(debug["start_z"]),
        "SweepTargetZMean": _mean_list(debug["sweep_target_z"]),
        "HitRate": _mean_list(hit_flag.to(dtype=state.contacts_meas.dtype)) if hit_flag is not None else None,
        "DistScoreMean": _mean_list(debug["dist_score"]),
        "VzScoreMean": _mean_list(debug["vz_score"]),
        "VxyScoreMean": _mean_list(debug["vxy_score"]),
        "MeasMean": _mean_list(state.contacts_meas),
    }


def compute_contact_meas_whitebox(trainer: Any, x_raw, prev_foot_pos=None):
    log_wb = bool(getattr(trainer, "log_contacts_whitebox", False))
    runtime = _resolve_contact_meas_runtime(trainer, x_raw)
    if runtime is None:
        if log_wb:
            trainer._contact_meas_whitebox_debug = None
        return None, prev_foot_pos

    bone_names = _resolve_contact_meas_bone_names(trainer, runtime.joint_count)
    foot_idxs = _resolve_contact_meas_foot_indices(
        trainer,
        bone_names=bone_names,
        joint_count=runtime.joint_count,
        contact_dim=runtime.contact_dim,
    )
    if foot_idxs is None:
        if log_wb:
            trainer._contact_meas_whitebox_debug = None
        return None, prev_foot_pos

    cfg = _resolve_contact_meas_cfg(trainer)
    state = _compute_contact_meas_whitebox_state(
        trainer,
        runtime,
        foot_idxs,
        prev_foot_pos,
        cfg,
        prev_root_pos=getattr(trainer, "_contact_meas_prev_root_pos", None),
        ground_z_prev=getattr(trainer, "_contact_meas_ground_z", None),
        ground_z_hist=getattr(trainer, "_contact_meas_ground_z_hist", _STATE_UPDATE_UNSET),
    )
    trainer._contact_meas_prev_root_pos = state.root_pos.detach() if torch.is_tensor(state.root_pos) else state.root_pos
    trainer._contact_meas_ground_z = state.ground_z
    if state.ground_z_hist is not _STATE_UPDATE_UNSET:
        trainer._contact_meas_ground_z_hist = state.ground_z_hist
    if log_wb:
        try:
            trainer._contact_meas_whitebox_debug = _build_contact_meas_whitebox_debug(
                cfg=cfg,
                bone_names=bone_names,
                foot_idxs=foot_idxs,
                state=state,
            )
        except (RuntimeError, TypeError, ValueError, AttributeError, KeyError, IndexError):
            trainer._contact_meas_whitebox_debug = None
    return state.contacts_meas, state.foot_pos.detach()
