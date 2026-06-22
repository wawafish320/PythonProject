#!/usr/bin/env python3
"""Matched positive seam neuron audit for action-handoff middle-state diagnostics.

This is the positive counterpart to MM-like cut stress tests. It builds a seam where
the last Walk_F frame is matched to a grounded target onset, re-anchors target root
position, rebuilds pose history from the spliced rotation sequence, and then records
internal model activations under teacher forcing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import train.validate.run_freerun_cycles as freerun  # noqa: E402
from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    EGO_VEL_SLICE,
    GROUND_CONTACT_THR,
    GROUND_POSE_THR,
    POSE_SLICE,
    POSE_TOPK,
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
    full_state_align,
    load_clip_states,
)
from train.data.normalizers import VectorTanhNormalizer  # noqa: E402
from train.geometry import angvel_vec_from_R_seq, reproject_rot6d, rot6d_to_matrix  # noqa: E402
from tools.run_action_handoff_inbetween_b1_cond_baseline_probe import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_CKPT,
    DEFAULT_ENCODER_BUNDLE,
    DEFAULT_NPZ_ROOT,
    DEFAULT_PRETRAIN_TEMPLATE,
    DEFAULT_Z_FEATURES,
    _make_runner_args,
)


ROOT_POS_KEY = "RootPosition"
ROOT_VEL_KEY = "RootVelocity"
ROT6D_KEY = "BoneRotations6D"
ANGVEL_KEY = "BoneAngularVelocities"


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _fmt(v: Any, digits: int = 4) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "null"
    if not math.isfinite(x):
        return "null"
    return f"{x:.{digits}f}"


def _parse_clips(raw: str) -> List[str]:
    clips = [tok.strip() for tok in str(raw or "").replace(";", ",").split(",") if tok.strip()]
    valid = set(TURN_CLIPS)
    bad = [c for c in clips if c not in valid]
    if bad:
        raise ValueError(f"unsupported target clip(s): {bad}; expected subset of {sorted(valid)}")
    return clips or list(TURN_CLIPS)


def _layout_slice(layout: Mapping[str, Any], key: str, *, fallback: Optional[slice] = None) -> slice:
    raw = layout.get(key)
    if isinstance(raw, slice):
        return raw
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        st, sz = int(raw[0]), int(raw[1])
        return slice(st, st + sz)
    if isinstance(raw, Mapping):
        st = int(raw.get("start", 0))
        if "size" in raw:
            return slice(st, st + int(raw["size"]))
        if "end" in raw:
            return slice(st, int(raw["end"]))
    if fallback is not None:
        return fallback
    raise KeyError(f"missing layout slice for {key!r}")


def _npz_scalar_to_json(v: Any) -> Dict[str, Any]:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    if isinstance(v, str):
        return json.loads(v)
    if isinstance(v, Mapping):
        return dict(v)
    raise TypeError(f"cannot parse JSON scalar of type {type(v).__name__}")


def _load_npz_raw(npz_root: Path, clip: str) -> Dict[str, Any]:
    path = npz_root / f"{clip}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"npz not found: {path}")
    with np.load(path, allow_pickle=True) as z:
        state_layout = _npz_scalar_to_json(z["state_layout_json"])
        output_layout = _npz_scalar_to_json(z["output_layout_json"])
        return {
            "path": str(path),
            "x_raw": np.asarray(z["x_in_features"], dtype=np.float32),
            "y_raw": np.asarray(z["y_out_features"], dtype=np.float32),
            "root_pos": np.asarray(z["root_pos"], dtype=np.float32),
            "root_vel": np.asarray(z["root_vel"], dtype=np.float32),
            "bone_rot6d": np.asarray(z["bone_rot6d"], dtype=np.float32),
            "state_layout": state_layout,
            "output_layout": output_layout,
        }


def _load_clip(runner: Any, npz_root: Path, clip_name: str, seq_len: int) -> Any:
    ds = runner._build_dataset(npz_root / f"{clip_name}.npz", seq_len=int(seq_len))
    runner._ensure_model_ready(ds)
    return ds.clips[0]


def _candidate_onset(
    hub_state: np.ndarray,
    target_state: np.ndarray,
    *,
    onset_scan: int,
    pose_topk: int,
    contact_thr: float,
    pose_thr: float,
) -> Optional[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scan_max = min(int(onset_scan), int(target_state.shape[0]) - 1)
    for onset in range(max(0, scan_max) + 1):
        align = full_state_align(
            hub_state,
            target_state[onset],
            topk=int(pose_topk),
            contact_thr=float(contact_thr),
            pose_thr=float(pose_thr),
        )
        score = float(align.full_state_pose_d / max(float(pose_thr), 1e-8))
        score += float(align.full_state_contact_d / max(float(contact_thr), 1e-8))
        rows.append(
            {
                "onset": int(onset),
                "phi": int(align.full_state_phi),
                "pose_d": float(align.full_state_pose_d),
                "contact_d": float(align.full_state_contact_d),
                "pose_only_phi": int(align.pose_only_phi),
                "pose_only_pose_d": float(align.pose_only_pose_d),
                "pose_only_contact_d": float(align.pose_only_contact_d),
                "pose_topk_frames": [int(x) for x in align.pose_topk_frames],
                "groundable": bool(align.groundable),
                "score": score,
            }
        )
    good = [r for r in rows if r["groundable"]]
    if not good:
        return {"selected": None, "scan_rows": rows}
    best = min(good, key=lambda r: (float(r["score"]), int(r["onset"])))
    return {"selected": best, "scan_rows": rows}


def _unwrapped_walk_positions(phi: int, pre: int, post: int, n: int) -> np.ndarray:
    start = int(phi) - int(pre) + 1
    return np.arange(start, start + int(pre) + int(post), dtype=np.int64)


def _target_positions(onset: int, length: int) -> np.ndarray:
    return np.arange(int(onset), int(onset) + int(length), dtype=np.int64)


def _root_cycle_delta(raw: Mapping[str, Any], clip_len: int) -> np.ndarray:
    root_pos = np.asarray(raw["root_pos"], dtype=np.float32)
    if root_pos.shape[0] > clip_len:
        return root_pos[int(clip_len)] - root_pos[0]
    x_raw = np.asarray(raw["x_raw"], dtype=np.float32)
    return x_raw[-1, 0:3] - x_raw[0, 0:3]


def _gather_clip_rows(arr: Optional[np.ndarray], idx: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return np.zeros((int(idx.shape[0]), 0), dtype=np.float32)
    if arr is None:
        return np.zeros((int(idx.shape[0]), int(width)), dtype=np.float32)
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    if a.shape[1] < width:
        pad = np.zeros((a.shape[0], int(width) - int(a.shape[1])), dtype=np.float32)
        a = np.concatenate([a, pad], axis=1)
    return np.ascontiguousarray(a[idx, : int(width)]).astype(np.float32, copy=False)


def _walk_rootpos(raw: Mapping[str, Any], positions: np.ndarray, clip_len: int) -> np.ndarray:
    x_raw = np.asarray(raw["x_raw"], dtype=np.float32)
    idx = np.mod(positions, int(clip_len)).astype(np.int64)
    cycles = np.floor_divide(positions, int(clip_len)).astype(np.float32)
    delta = _root_cycle_delta(raw, clip_len).reshape(1, 3)
    return x_raw[idx, 0:3] + cycles[:, None] * delta


def _target_rootpos(raw: Mapping[str, Any], positions: np.ndarray) -> np.ndarray:
    x_raw = np.asarray(raw["x_raw"], dtype=np.float32)
    return x_raw[positions.astype(np.int64), 0:3].copy()


def _normalize_rootpos_into_x(
    x_norm: np.ndarray,
    rootpos_raw: np.ndarray,
    *,
    mu_x: np.ndarray,
    std_x: np.ndarray,
    rootpos_sl: slice,
) -> np.ndarray:
    out = np.asarray(x_norm, dtype=np.float32).copy()
    out[:, rootpos_sl] = (rootpos_raw.astype(np.float32) - mu_x[rootpos_sl]) / std_x[rootpos_sl]
    return out.astype(np.float32, copy=False)


def _rebuild_pose_history(
    rot_seq: np.ndarray,
    *,
    pose_hist_dim: int,
    norm_spec: Mapping[str, Any],
) -> np.ndarray:
    if int(pose_hist_dim) <= 0:
        return np.zeros((int(rot_seq.shape[0]), 0), dtype=np.float32)
    pose_dim = int(rot_seq.shape[1])
    if pose_dim <= 0 or int(pose_hist_dim) % pose_dim != 0:
        raise ValueError(f"pose_hist_dim={pose_hist_dim} is not divisible by pose_dim={pose_dim}")
    hist_len = int(pose_hist_dim) // pose_dim
    t = int(rot_seq.shape[0])
    offsets = np.arange(hist_len, 0, -1, dtype=np.int64)
    frame_ids = np.arange(t, dtype=np.int64)[:, None] - offsets[None, :]
    np.clip(frame_ids, 0, t - 1, out=frame_ids)
    raw = rot_seq[frame_ids].reshape(t, -1).astype(np.float32, copy=False)
    scales = norm_spec.get("tanh_scales_pose_hist")
    if scales is not None and len(scales) == raw.shape[1]:
        return VectorTanhNormalizer(
            np.asarray(scales, dtype=np.float32),
            np.asarray(norm_spec.get("MuPoseHist"), dtype=np.float32) if norm_spec.get("MuPoseHist") is not None else None,
            np.asarray(norm_spec.get("StdPoseHist"), dtype=np.float32) if norm_spec.get("StdPoseHist") is not None else None,
        ).transform(raw)
    return raw.astype(np.float32, copy=False)


def _rebuild_angvel(rot_seq: np.ndarray, *, norm_spec: Mapping[str, Any], fallback_width: int) -> np.ndarray:
    if int(fallback_width) <= 0:
        return np.zeros((int(rot_seq.shape[0]), 0), dtype=np.float32)
    t = int(rot_seq.shape[0])
    try:
        joints = int(rot_seq.shape[1]) // 6
        y = torch.as_tensor(rot_seq, dtype=torch.float32)
        y = reproject_rot6d(y.unsqueeze(0))[0]
        R = rot6d_to_matrix(y.view(1, t, joints, 6))[0]
        w = angvel_vec_from_R_seq(R.unsqueeze(0), 60.0)[0].reshape(t, joints * 3).cpu().numpy()
        scales = norm_spec.get("tanh_scales_angvel")
        if scales is not None and len(scales) == w.shape[1]:
            return VectorTanhNormalizer(
                np.asarray(scales, dtype=np.float32),
                np.asarray(norm_spec.get("MuAngVel"), dtype=np.float32) if norm_spec.get("MuAngVel") is not None else None,
                np.asarray(norm_spec.get("StdAngVel"), dtype=np.float32) if norm_spec.get("StdAngVel") is not None else None,
            ).transform(w)
        return w.astype(np.float32, copy=False)
    except Exception:
        return np.zeros((t, int(fallback_width)), dtype=np.float32)


def _make_sample(
    *,
    case: str,
    walk_clip: Any,
    target_clip: Any,
    walk_raw: Mapping[str, Any],
    target_raw: Mapping[str, Any],
    phi: int,
    onset: int,
    pre: int,
    post: int,
    dims: Mapping[str, int],
    slices: Mapping[str, slice],
    mu_x: np.ndarray,
    std_x: np.ndarray,
    norm_spec: Mapping[str, Any],
    cond_ramp_frames: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    total = int(pre) + int(post)
    rootpos_sl = slices["rootpos_x"]
    rootvel_sl = slices["rootvel_x"]
    rot_x_sl = slices["rot_x"]
    rot_y_sl = slices["rot_y"]

    walk_len = int(walk_clip.X.shape[0])
    target_len = int(target_clip.X.shape[0])

    matched_cases = {
        "matched_positive",
        "matched_positive_xhist",
        "matched_positive_walkcond_xhist",
        "matched_positive_rampcond_xhist",
    }
    if case in matched_cases:
        walk_pos = _unwrapped_walk_positions(int(phi), int(pre), 0, walk_len)
        walk_all_pos = _unwrapped_walk_positions(int(phi), int(pre), int(post), walk_len)
        target_pos = _target_positions(int(onset), int(post))
        if int(target_pos[-1]) >= target_len:
            raise ValueError(f"{case}: target positions exceed clip length ({int(target_pos[-1])} >= {target_len})")
        walk_idx = np.mod(walk_pos, walk_len).astype(np.int64)
        walk_all_idx = np.mod(walk_all_pos, walk_len).astype(np.int64)
        target_idx = target_pos.astype(np.int64)

        x_pre = _gather_clip_rows(walk_clip.X, walk_idx, dims["state"])
        x_post = _gather_clip_rows(target_clip.X, target_idx, dims["state"])
        walk_root = _walk_rootpos(walk_raw, walk_pos, walk_len)
        target_root = _target_rootpos(target_raw, target_pos)
        reanchor_delta = walk_root[-1] - target_root[0]
        target_root = target_root + reanchor_delta.reshape(1, 3)
        x_pre = _normalize_rootpos_into_x(x_pre, walk_root, mu_x=mu_x, std_x=std_x, rootpos_sl=rootpos_sl)
        x_post = _normalize_rootpos_into_x(x_post, target_root, mu_x=mu_x, std_x=std_x, rootpos_sl=rootpos_sl)

        hard_cond = np.concatenate(
            [
                _gather_clip_rows(walk_clip.C, walk_idx, dims["cond"]),
                _gather_clip_rows(target_clip.C, target_idx, dims["cond"]),
            ],
            axis=0,
        )
        walk_cond = _gather_clip_rows(walk_clip.C, walk_all_idx, dims["cond"])
        cond_variant = "hard_target_cond"
        if case == "matched_positive_walkcond_xhist":
            cond = walk_cond
            cond_variant = "walk_continuous_cond"
        elif case == "matched_positive_rampcond_xhist":
            cond = hard_cond.copy()
            ramp = int(max(1, min(int(cond_ramp_frames), int(post))))
            start_cond = walk_cond[int(pre) - 1]
            for i in range(ramp):
                alpha = float(i + 1) / float(ramp + 1)
                cond[int(pre) + i] = (1.0 - alpha) * start_cond + alpha * hard_cond[int(pre) + i]
            cond_variant = f"ramp_target_cond_{ramp}f"
        else:
            cond = hard_cond
        contacts = np.concatenate(
            [
                _gather_clip_rows(getattr(walk_clip, "contacts", None), walk_idx, dims["contact"]),
                _gather_clip_rows(getattr(target_clip, "contacts", None), target_idx, dims["contact"]),
            ],
            axis=0,
        )
        rot_seq = np.concatenate(
            [
                _gather_clip_rows(walk_clip.Y[:, rot_y_sl], walk_idx, rot_y_sl.stop - rot_y_sl.start),
                _gather_clip_rows(target_clip.Y[:, rot_y_sl], target_idx, rot_y_sl.stop - rot_y_sl.start),
            ],
            axis=0,
        )
        rootpos_raw = np.concatenate([walk_root, target_root], axis=0)
        source = ["walk"] * int(pre) + ["target"] * int(post)
        meta = {
            "case": case,
            "description": (
                "Walk_F matched phase pre + target grounded onset post, with target root re-anchored. "
                + (
                    "pose_history/angvel rebuilt from X/current rot6d to isolate history side-channel."
                    if case.endswith("_xhist")
                    else "pose_history/angvel follow dataset-style Y/history rot6d."
                )
            ),
            "cond_variant": cond_variant,
            "walk_positions_unwrapped": [int(x) for x in walk_pos.tolist()],
            "walk_all_positions_unwrapped": [int(x) for x in walk_all_pos.tolist()],
            "target_positions": [int(x) for x in target_pos.tolist()],
            "walk_indices": [int(x) for x in walk_idx.tolist()],
            "walk_all_indices": [int(x) for x in walk_all_idx.tolist()],
            "target_indices": [int(x) for x in target_idx.tolist()],
            "reanchor_delta_xyz": [float(x) for x in reanchor_delta.tolist()],
        }
    elif case == "walk_continuous":
        walk_pos = _unwrapped_walk_positions(int(phi), int(pre), int(post), walk_len)
        walk_idx = np.mod(walk_pos, walk_len).astype(np.int64)
        rootpos_raw = _walk_rootpos(walk_raw, walk_pos, walk_len)
        x = _gather_clip_rows(walk_clip.X, walk_idx, dims["state"])
        x = _normalize_rootpos_into_x(x, rootpos_raw, mu_x=mu_x, std_x=std_x, rootpos_sl=rootpos_sl)
        cond = _gather_clip_rows(walk_clip.C, walk_idx, dims["cond"])
        contacts = _gather_clip_rows(getattr(walk_clip, "contacts", None), walk_idx, dims["contact"])
        rot_seq = _gather_clip_rows(walk_clip.Y[:, rot_y_sl], walk_idx, rot_y_sl.stop - rot_y_sl.start)
        x_pre, x_post = x[:total], np.zeros((0, dims["state"]), dtype=np.float32)
        source = ["walk"] * total
        meta = {
            "case": case,
            "description": "Continuous unwrapped Walk_F baseline through the same phi.",
            "walk_positions_unwrapped": [int(x) for x in walk_pos.tolist()],
            "walk_indices": [int(x) for x in walk_idx.tolist()],
        }
    elif case == "target_continuous":
        target_pos = _target_positions(int(onset), total)
        if int(target_pos[-1]) >= target_len:
            raise ValueError(f"{case}: target positions exceed clip length ({int(target_pos[-1])} >= {target_len})")
        target_idx = target_pos.astype(np.int64)
        rootpos_raw = _target_rootpos(target_raw, target_pos)
        x = _gather_clip_rows(target_clip.X, target_idx, dims["state"])
        x = _normalize_rootpos_into_x(x, rootpos_raw, mu_x=mu_x, std_x=std_x, rootpos_sl=rootpos_sl)
        cond = _gather_clip_rows(target_clip.C, target_idx, dims["cond"])
        contacts = _gather_clip_rows(getattr(target_clip, "contacts", None), target_idx, dims["contact"])
        rot_seq = _gather_clip_rows(target_clip.Y[:, rot_y_sl], target_idx, rot_y_sl.stop - rot_y_sl.start)
        x_pre, x_post = x[:total], np.zeros((0, dims["state"]), dtype=np.float32)
        source = ["target"] * total
        meta = {
            "case": case,
            "description": "Continuous target clip baseline from selected onset.",
            "target_positions": [int(x) for x in target_pos.tolist()],
            "target_indices": [int(x) for x in target_idx.tolist()],
        }
    else:
        raise ValueError(f"unsupported case={case!r}")

    if case in matched_cases:
        state = np.concatenate([x_pre, x_post], axis=0)
    else:
        state = x_pre

    x_rot_raw_for_history = state[:, rot_x_sl] * std_x[rot_x_sl] + mu_x[rot_x_sl]
    history_rot_seq = x_rot_raw_for_history if case.endswith("_xhist") else rot_seq
    pose_hist = _rebuild_pose_history(history_rot_seq, pose_hist_dim=int(dims["pose_hist"]), norm_spec=norm_spec)
    angvel = _rebuild_angvel(history_rot_seq, norm_spec=norm_spec, fallback_width=int(dims["angvel"]))
    if int(angvel.shape[1]) != int(dims["angvel"]):
        angvel = _gather_clip_rows(None, np.arange(total), int(dims["angvel"]))
    if int(pose_hist.shape[1]) != int(dims["pose_hist"]):
        pose_hist = _gather_clip_rows(None, np.arange(total), int(dims["pose_hist"]))

    sample = {
        "state": torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).float(),
        "cond": torch.from_numpy(np.ascontiguousarray(cond)).unsqueeze(0).float(),
        "contacts": torch.from_numpy(np.ascontiguousarray(contacts)).unsqueeze(0).float(),
        "angvel": torch.from_numpy(np.ascontiguousarray(angvel)).unsqueeze(0).float(),
        "pose_history": torch.from_numpy(np.ascontiguousarray(pose_hist)).unsqueeze(0).float(),
    }

    root_jump = rootpos_raw[int(pre)] - rootpos_raw[int(pre) - 1] if total > int(pre) else np.zeros(3, dtype=np.float32)
    x_rot_raw = state[:, rot_x_sl] * std_x[rot_x_sl] + mu_x[rot_x_sl]
    x_rootvel_norm = state[:, rootvel_sl]
    meta.update(
        {
            "source": source,
            "rootpos_jump_at_cut_xyz": [float(x) for x in root_jump.tolist()],
            "rootpos_jump_at_cut_l2": float(np.linalg.norm(root_jump.astype(np.float64))),
            "history_source": "x_current_rot6d" if case.endswith("_xhist") else "y_history_rot6d",
            "x_rot6d_step_l2_at_cut": float(
                np.linalg.norm((x_rot_raw[int(pre)] - x_rot_raw[int(pre) - 1]).astype(np.float64))
                / math.sqrt(max(1, x_rot_raw.shape[1]))
            ) if total > int(pre) else None,
            "x_rootvel_norm_step_l2_at_cut": float(
                np.linalg.norm((x_rootvel_norm[int(pre)] - x_rootvel_norm[int(pre) - 1]).astype(np.float64))
            ) if total > int(pre) else None,
            "history_rot6d_step_l2_at_cut": float(
                np.linalg.norm((history_rot_seq[int(pre)] - history_rot_seq[int(pre) - 1]).astype(np.float64))
                / math.sqrt(max(1, history_rot_seq.shape[1]))
            ) if total > int(pre) else None,
            "y_history_rot6d_step_l2_at_cut": float(
                np.linalg.norm((rot_seq[int(pre)] - rot_seq[int(pre) - 1]).astype(np.float64))
                / math.sqrt(max(1, rot_seq.shape[1]))
            ) if total > int(pre) else None,
            "contact_step_l2_at_cut": float(
                np.linalg.norm((contacts[int(pre)] - contacts[int(pre) - 1]).astype(np.float64))
            ) if total > int(pre) and contacts.shape[1] > 0 else None,
            "cond_step_l2_at_cut": float(
                np.linalg.norm((cond[int(pre)] - cond[int(pre) - 1]).astype(np.float64))
            ) if total > int(pre) and cond.shape[1] > 0 else None,
            "input_contract": {
                "state": {"shape": [int(x) for x in sample["state"].shape], "dtype": "float32", "device": "cpu"},
                "cond": {"shape": [int(x) for x in sample["cond"].shape], "dtype": "float32", "device": "cpu"},
                "contacts": {"shape": [int(x) for x in sample["contacts"].shape], "dtype": "float32", "device": "cpu"},
                "angvel": {"shape": [int(x) for x in sample["angvel"].shape], "dtype": "float32", "device": "cpu"},
                "pose_history": {
                    "shape": [int(x) for x in sample["pose_history"].shape],
                    "dtype": "float32",
                    "device": "cpu",
                },
            },
        }
    )
    return sample, meta


def _as_series(t: torch.Tensor, *, T: int) -> Optional[np.ndarray]:
    if not torch.is_tensor(t):
        return None
    x = t.detach().float().cpu()
    if x.ndim == 0:
        return np.full((int(T), 1), float(x.item()), dtype=np.float32)
    if x.ndim >= 2 and int(x.shape[0]) == 1 and int(x.shape[1]) == int(T):
        return x[0].reshape(int(T), -1).numpy()
    if x.ndim >= 1 and int(x.shape[0]) == int(T):
        return x.reshape(int(T), -1).numpy()
    if int(x.numel()) % int(T) == 0:
        return x.reshape(int(T), -1).numpy()
    return None


def _summarize_series(arr: np.ndarray, *, cut: int, topk: int) -> Dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1)
    norms = np.linalg.norm(a, axis=1)
    delta = np.full((a.shape[0],), np.nan, dtype=np.float64)
    if a.shape[0] >= 2:
        delta[1:] = np.linalg.norm(np.diff(a, axis=0), axis=1)
    cos_prev = None
    if 0 < int(cut) < a.shape[0]:
        u = a[int(cut) - 1]
        v = a[int(cut)]
        den = float(np.linalg.norm(u) * np.linalg.norm(v))
        cos_prev = float(np.dot(u, v) / den) if den > 1e-12 else None
    pre_vals = delta[max(1, int(cut) - 4): int(cut)]
    post_vals = delta[int(cut): min(a.shape[0], int(cut) + 4)]
    cut_delta = float(delta[int(cut)]) if 0 <= int(cut) < a.shape[0] and np.isfinite(delta[int(cut)]) else None
    pre_mean = float(np.nanmean(pre_vals)) if pre_vals.size and np.isfinite(pre_vals).any() else None
    post_mean = float(np.nanmean(post_vals)) if post_vals.size and np.isfinite(post_vals).any() else None
    top_dims: List[Dict[str, float | int]] = []
    if 0 < int(cut) < a.shape[0] and a.shape[1] > 0:
        d = a[int(cut)] - a[int(cut) - 1]
        order = np.argsort(-np.abs(d))[: max(0, int(topk))]
        top_dims = [{"dim": int(i), "delta": float(d[i]), "abs_delta": float(abs(d[i]))} for i in order]
    return {
        "shape": [int(x) for x in a.shape],
        "norm_at_cut": float(norms[int(cut)]) if 0 <= int(cut) < a.shape[0] else None,
        "delta_norm_at_cut": cut_delta,
        "pre4_delta_mean": pre_mean,
        "post4_delta_mean": post_mean,
        "cut_over_pre4": (float(cut_delta / pre_mean) if cut_delta is not None and pre_mean and pre_mean > 1e-12 else None),
        "cos_prev_at_cut": cos_prev,
        "top_delta_dims_at_cut": top_dims,
    }


def _case_forward_audit(
    model: Any,
    sample: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    cut_step: int,
    topk_dims: int,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    state = sample["state"].to(device=device)
    cond = sample["cond"].to(device=device)
    contacts = sample["contacts"].to(device=device)
    angvel = sample["angvel"].to(device=device)
    pose_history = sample["pose_history"].to(device=device)
    _, T, _ = state.shape

    raw_captures: Dict[str, List[torch.Tensor]] = {}
    handles = []

    def add_capture(name: str, module: Any, *, pre: bool = False) -> None:
        if module is None:
            return

        def _save(_m: Any, inputs: Tuple[Any, ...], output: Any = None) -> None:
            val = inputs[0] if pre else output
            if torch.is_tensor(val):
                raw_captures.setdefault(name, []).append(val.detach().float().cpu())

        if pre:
            handles.append(module.register_forward_pre_hook(_save))
        else:
            handles.append(module.register_forward_hook(_save))

    add_capture("hidden_pre_pasa_lnq_input", getattr(model, "_pasa_lnq", None), pre=True)
    add_capture("contact_plan_cell_raw", getattr(model, "contact_plan_cell", None))
    add_capture("contact_plan_head_input", getattr(model, "contact_plan_head", None), pre=True)
    add_capture("direct_pose_head_output", getattr(model, "direct_pose_head", None))
    if getattr(model, "shared_encoder", None) is not None:
        for i, module in enumerate(model.shared_encoder):
            add_capture(f"shared_encoder.{i}", module)
    if getattr(model, "direct_pose_head", None) is not None:
        for i, module in enumerate(model.direct_pose_head):
            add_capture(f"direct_pose_head.{i}", module)

    try:
        if hasattr(model, "set_eval_runtime_controls"):
            model.set_eval_runtime_controls(debug_contact_plan_logits_decomp=True)
        with torch.no_grad():
            result = model(
                state,
                cond,
                contacts=contacts,
                angvel=angvel,
                pose_history=pose_history,
                time_index=torch.arange(T, device=device, dtype=state.dtype).view(1, T),
                rollout_step=torch.arange(T, device=device, dtype=state.dtype).view(1, T),
            )
    finally:
        for h in handles:
            h.remove()
        if hasattr(model, "_reset_eval_runtime_controls"):
            model._reset_eval_runtime_controls()

    if not isinstance(result, dict):
        raise RuntimeError("EventMotionModel.forward did not return dict")

    series: Dict[str, np.ndarray] = {}
    for name, vals in raw_captures.items():
        if not vals:
            continue
        if name == "contact_plan_head_input" and len(vals) == int(T) * 2:
            stacked = torch.stack([v.reshape(-1, v.shape[-1]).mean(dim=0) for v in vals[1::2]], dim=0)
            series["contact_plan_z_corrected"] = stacked.numpy()
            raw = torch.stack([v.reshape(-1, v.shape[-1]).mean(dim=0) for v in vals[0::2]], dim=0)
            series["contact_plan_z_raw_head_input"] = raw.numpy()
            continue
        if name == "contact_plan_cell_raw":
            stacked = torch.stack([v.reshape(-1, v.shape[-1]).mean(dim=0) for v in vals], dim=0)
            series[name] = stacked.numpy()
            continue
        arr = _as_series(vals[-1], T=int(T))
        if arr is not None:
            series[name] = arr.astype(np.float32, copy=False)

    for key, value in result.items():
        if torch.is_tensor(value):
            arr = _as_series(value, T=int(T))
            if arr is not None:
                series[f"out::{key}"] = arr.astype(np.float32, copy=False)

    summaries = {
        name: _summarize_series(arr, cut=int(cut_step), topk=int(topk_dims))
        for name, arr in sorted(series.items())
    }
    output_contract = {
        k: {"shape": [int(x) for x in v.shape], "dtype": str(v.dtype).replace("torch.", ""), "device": str(v.device)}
        for k, v in sorted(result.items())
        if torch.is_tensor(v)
    }
    return {
        "summary_by_signal": summaries,
        "output_contract": output_contract,
    }, series


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Matched positive seam neuron audit.")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--bundle", type=str, default=DEFAULT_BUNDLE)
    p.add_argument("--pretrain-template", type=str, default=DEFAULT_PRETRAIN_TEMPLATE)
    p.add_argument("--encoder-bundle", type=str, default=DEFAULT_ENCODER_BUNDLE)
    p.add_argument("--npz-root", type=str, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=str, default=DEFAULT_Z_FEATURES)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--pre-frames", type=int, default=16)
    p.add_argument("--post-frames", type=int, default=24)
    p.add_argument("--onset-scan", type=int, default=8)
    p.add_argument("--target-clips", type=str, default=",".join(TURN_CLIPS))
    p.add_argument(
        "--cases",
        type=str,
        default=(
            "matched_positive,matched_positive_xhist,"
            "matched_positive_walkcond_xhist,matched_positive_rampcond_xhist,"
            "walk_continuous,target_continuous"
        ),
    )
    p.add_argument("--cond-ramp-frames", type=int, default=8)
    p.add_argument("--pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--ground-pose-thr", type=float, default=GROUND_POSE_THR)
    p.add_argument("--topk-dims", type=int, default=8)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    npz_root = Path(args.npz_root)
    z_features = Path(args.z_features)
    target_clips = _parse_clips(args.target_clips)
    cases = [c.strip() for c in str(args.cases).replace(";", ",").split(",") if c.strip()]
    valid_cases = {
        "matched_positive",
        "matched_positive_xhist",
        "matched_positive_walkcond_xhist",
        "matched_positive_rampcond_xhist",
        "walk_continuous",
        "target_continuous",
    }
    bad_cases = [c for c in cases if c not in valid_cases]
    if bad_cases:
        raise ValueError(f"unsupported cases: {bad_cases}; expected subset of {sorted(valid_cases)}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"debug_output/_tmp_action_handoff_matched_seam_neuron_audit_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = freerun.FreeRunCycleRunner(_make_runner_args(args))
    seq_len = max(64, int(args.pre_frames) + int(args.post_frames))
    walk_clip = _load_clip(runner, npz_root, WALK_F, seq_len=seq_len)
    runner.model.eval()
    model = runner.model

    norm_spec = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    mu_x = np.asarray(norm_spec["MuX"], dtype=np.float32)
    std_x = np.asarray(norm_spec["StdX"], dtype=np.float32)
    std_x = np.where(np.abs(std_x) > 1e-8, std_x, 1.0).astype(np.float32)

    walk_raw = _load_npz_raw(npz_root, WALK_F)
    rootpos_sl = _layout_slice(walk_clip.state_layout_norm, ROOT_POS_KEY, fallback=slice(0, 3))
    rootvel_sl = _layout_slice(walk_clip.state_layout_norm, ROOT_VEL_KEY, fallback=slice(3, 5))
    rot_x_sl = _layout_slice(walk_clip.state_layout_norm, ROT6D_KEY, fallback=slice(5, 281))
    rot_y_sl = _layout_slice(walk_clip.output_layout_norm, ROT6D_KEY, fallback=slice(0, 276))

    dims = {
        "state": int(getattr(model, "in_state_dim", walk_clip.X.shape[1])),
        "cond": int(getattr(model, "cond_dim", walk_clip.C.shape[1])),
        "contact": int(getattr(model, "contact_dim", 0) or 0),
        "angvel": int(getattr(model, "angvel_dim", 0) or 0),
        "pose_hist": int(getattr(model, "pose_hist_dim", 0) or 0),
    }
    slices = {"rootpos_x": rootpos_sl, "rootvel_x": rootvel_sl, "rot_x": rot_x_sl, "rot_y": rot_y_sl}

    states_281 = load_clip_states(z_features, npz_root)
    hub_state = states_281[WALK_F]

    payload: Dict[str, Any] = {
        "task": "action_handoff_matched_positive_seam_neuron_audit",
        "scope": "read-only teacher-forced neuron/activation audit; no training and no checkpoint mutation",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "npz_root": str(npz_root.resolve()),
        "z_features": str(z_features.resolve()),
        "config": {
            "pre_frames": int(args.pre_frames),
            "post_frames": int(args.post_frames),
            "cut_step": int(args.pre_frames),
            "onset_scan": int(args.onset_scan),
            "target_clips": target_clips,
            "cases": cases,
            "device": str(args.device),
        },
        "dims": dims,
        "slices": {k: [int(v.start or 0), int(v.stop or 0)] for k, v in slices.items()},
        "model_flags": {
            "contact_plan_enable": bool(getattr(model, "contact_plan_enable", False)),
            "contact_plan_hidden": int(getattr(model, "contact_plan_hidden", 0) or 0),
            "contact_plan_inject": str(getattr(model, "contact_plan_inject", "")),
            "use_event_clock": bool(getattr(model, "use_event_clock", False)),
            "direct_pose_enable": bool(getattr(model, "direct_pose_enable", False)),
            "direct_pose_feat_source": str(getattr(model, "direct_pose_feat_source", "")),
            "direct_pose_split_enable": bool(getattr(model, "direct_pose_split_enable", False)),
            "lambda_fusion_enable": bool(getattr(model, "lambda_fusion_enable", False)),
            "so3_delta_corrector": bool(getattr(model, "so3_delta_corrector", None) is not None),
        },
        "targets": {},
    }
    activation_npz: Dict[str, np.ndarray] = {}

    for target in target_clips:
        target_state = states_281[target]
        cand = _candidate_onset(
            hub_state,
            target_state,
            onset_scan=int(args.onset_scan),
            pose_topk=int(args.pose_topk),
            contact_thr=float(args.ground_contact_thr),
            pose_thr=float(args.ground_pose_thr),
        )
        target_payload: Dict[str, Any] = {"alignment": cand, "cases": {}}
        selected = cand.get("selected") if isinstance(cand, dict) else None
        if not selected:
            target_payload["skip_reason"] = "no groundable matched onset in scan window"
            payload["targets"][target] = target_payload
            continue

        phi = int(selected["phi"])
        onset = int(selected["onset"])
        target_clip = _load_clip(runner, npz_root, target, seq_len=seq_len)
        target_raw = _load_npz_raw(npz_root, target)

        for case in cases:
            sample, meta = _make_sample(
                case=case,
                walk_clip=walk_clip,
                target_clip=target_clip,
                walk_raw=walk_raw,
                target_raw=target_raw,
                phi=phi,
                onset=onset,
                pre=int(args.pre_frames),
                post=int(args.post_frames),
                dims=dims,
                slices=slices,
                mu_x=mu_x,
                std_x=std_x,
                norm_spec=norm_spec,
                cond_ramp_frames=int(args.cond_ramp_frames),
            )
            audit, series = _case_forward_audit(
                model,
                sample,
                device=runner.device,
                cut_step=int(args.pre_frames),
                topk_dims=int(args.topk_dims),
            )
            target_payload["cases"][case] = {**meta, **audit}
            for name, arr in series.items():
                safe_name = name.replace("::", "__").replace(".", "_")
                activation_npz[f"{target}__{case}__{safe_name}"] = arr.astype(np.float32, copy=False)

        payload["targets"][target] = target_payload

    json_path = out_dir / "matched_seam_neuron_audit_summary.json"
    npz_path = out_dir / "matched_seam_activations.npz"
    md_path = out_dir / "matched_seam_neuron_audit_summary.md"
    _dump_json(json_path, payload)
    if activation_npz:
        np.savez_compressed(npz_path, **activation_npz)

    lines: List[str] = []
    lines.append("# Matched Positive Seam Neuron Audit")
    lines.append("")
    lines.append("Read-only teacher-forced audit over matched grounded seams; no training and no checkpoint mutation.")
    lines.append("")
    lines.append(f"- checkpoint: `{Path(args.checkpoint).name}`")
    lines.append(f"- cut: pre/post `{int(args.pre_frames)}/{int(args.post_frames)}`")
    lines.append(f"- cases: `{', '.join(cases)}`")
    lines.append(f"- activations npz: `{npz_path.resolve()}`")
    lines.append("")
    for target in target_clips:
        t = payload["targets"].get(target, {})
        lines.append(f"## Target `{target}`")
        if t.get("skip_reason"):
            lines.append(f"- skipped: {t['skip_reason']}")
            lines.append("")
            continue
        sel = t["alignment"]["selected"]
        lines.append(
            f"- selected onset `{sel['onset']}` matched to Walk_F phi `{sel['phi']}` "
            f"(pose_d={_fmt(sel['pose_d'])}, contact_d={_fmt(sel['contact_d'])})"
        )
        lines.append("| case | root jump@cut | X rot6d step@cut | hist rot6d step@cut | contact step@cut | cond step@cut | hidden_pre cut/pre | h_final cut/pre | plan_z cut/pre | contacts_plan cut/pre |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for case in cases:
            c = t["cases"].get(case, {})
            s = c.get("summary_by_signal", {})
            hidden = s.get("hidden_pre_pasa_lnq_input", {})
            hfinal = s.get("out::h_final", {})
            plan = s.get("contact_plan_z_corrected", {})
            cplan = s.get("out::contacts_plan", {})
            lines.append(
                f"| {case} | "
                f"{_fmt(c.get('rootpos_jump_at_cut_l2'))} | "
                f"{_fmt(c.get('x_rot6d_step_l2_at_cut'))} | "
                f"{_fmt(c.get('history_rot6d_step_l2_at_cut'))} | "
                f"{_fmt(c.get('contact_step_l2_at_cut'))} | "
                f"{_fmt(c.get('cond_step_l2_at_cut'))} | "
                f"{_fmt(hidden.get('cut_over_pre4'))} | "
                f"{_fmt(hfinal.get('cut_over_pre4'))} | "
                f"{_fmt(plan.get('cut_over_pre4'))} | "
                f"{_fmt(cplan.get('cut_over_pre4'))} |"
            )
        lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- `{json_path.resolve()}`")
    if activation_npz:
        lines.append(f"- `{npz_path.resolve()}`")
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    if activation_npz:
        print(f"[ok] wrote: {npz_path}")
    print(f"[ok] wrote: {md_path}")
    for target in target_clips:
        t = payload["targets"].get(target, {})
        if t.get("skip_reason"):
            print(f"[target {target}] skipped: {t['skip_reason']}")
            continue
        sel = t["alignment"]["selected"]
        row_bits = [f"match onset={sel['onset']} phi={sel['phi']} pose_d={_fmt(sel['pose_d'])} contact_d={_fmt(sel['contact_d'])}"]
        for case in cases:
            s = t["cases"][case]["summary_by_signal"]
            hidden = s.get("hidden_pre_pasa_lnq_input", {})
            hfinal = s.get("out::h_final", {})
            row_bits.append(f"{case}: hidden_pre_ratio={_fmt(hidden.get('cut_over_pre4'))}, h_final_ratio={_fmt(hfinal.get('cut_over_pre4'))}")
        print(f"[target {target}] " + " | ".join(row_bits))


if __name__ == "__main__":
    main()
