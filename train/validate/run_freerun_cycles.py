#!/usr/bin/env python3
"""
Free‑run multi‑cycle diagnostics on a trained MPL checkpoint.

Goal:
    - Take one or more teacher clips (validate/teacher_batches/*.json).
    - Build the same MotionEventDataset / Trainer stack as training.
    - Let the model free‑run autoregressively for N cycles on a single clip
      *without resetting between cycles*.
    - For each cycle, dump a JSON payload with diagnostics similar to the
      existing teacher/free‑run summaries (MSEnormY, GeoDeg, etc.).

Usage example (single clip):
    python -m train.validate.run_freerun_cycles \\
        --model models/MLPL2_uncertainty_adapt8_v6/exp_phase_MLPL2_uncertainty_adapt8_v6/ckpt_best_exp_phase_MLPL2_uncertainty_adapt8_v6.pth \\
        --teacher validate/teacher_batches/Walk_F_teacher.json \\
        --bundle raw_data/processed_data/norm_template.json \\
        --pretrain-template models/pretrain_template.json \\
        --npz-root raw_data/processed_data \\
        --rounds 5 \\
        --out debug_output/freerun_cycles

Notes:
    - This script intentionally mirrors the teacher rollout tooling
      (train/validate/run_teacher_rollout.py) but switches to true free‑run.
    - Currently only PyTorch checkpoints are supported (no ONNX).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch  # ensure torch is bound before any inner scope uses it

from train.training_MPL import MotionEventDataset, Trainer, geodesic_R, validate_and_fix_model_
from train.geometry import matrix_to_rot6d, reproject_rot6d, rot6d_to_matrix, so3_exp_map, so3_log_map
from train.history import (
    PoseHistState,
    advance_pose_hist_state_with_tail,
    init_pose_hist_state,
    pose_hist_inverse_vec,
    pose_hist_transform_vec,
    resolve_pose_hist_input,
)
from train.models import EventMotionModel, MotionJointLoss
from train.layout import LayoutCenter, DataNormalizer
from train.geometry import compose_rot6d_delta
from train.rotvec_semantics import require_standard_rotvec_spec
from train.ttc import ttc_to_next_event_np


# ---- Helpers (mostly mirrored from run_teacher_rollout) ---------------------


def _expand_specs(specs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for spec in specs:
        if not spec:
            continue
        path = Path(spec).expanduser()
        matches: List[Path] = []
        if any(ch in spec for ch in "*?[]"):
            matches = sorted(Path(".").glob(spec))
        elif path.is_dir():
            matches = sorted(path.glob("*.json"))
        elif path.is_file():
            matches = [path]
        if not matches and path.parent.exists() and any(ch in path.name for ch in "*?[]"):
            matches = sorted(path.parent.glob(path.name))
        for candidate in matches:
            resolved = candidate.resolve()
            if resolved not in seen and resolved.is_file():
                seen.add(resolved)
                out.append(resolved)
    return sorted(out)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_norm_spec(bundle_path: Path, pretrain_path: Optional[Path]) -> Dict[str, Any]:
    with bundle_path.open("r", encoding="utf-8") as f:
        base = json.load(f)
    require_standard_rotvec_spec(base, context=f"bundle {bundle_path}")
    spec = dict(base)
    if pretrain_path and pretrain_path.is_file():
        with pretrain_path.open("r", encoding="utf-8") as f:
            pre = json.load(f)
        require_standard_rotvec_spec(pre, context=f"pretrain_template {pretrain_path}")
        for key in (
            "MuAngVel",
            "StdAngVel",
            "tanh_scales_angvel",
            "pose_hist_len",
            "pose_hist_dim",
            "tanh_scales_pose_hist",
            "MuPoseHist",
            "StdPoseHist",
        ):
            if key in pre and pre[key] is not None:
                spec[key] = pre[key]
    return spec


def _resolve_direct_pose_leg_idx_tensor(model: Any, *, device: torch.device) -> Optional[torch.Tensor]:
    idx = getattr(model, "direct_pose_leg_joint_idx_tensor", None)
    if torch.is_tensor(idx):
        try:
            return idx.to(device=device, dtype=torch.long).reshape(-1)
        except Exception:
            pass
    raw = getattr(model, "direct_pose_leg_joint_idx", None)
    if isinstance(raw, (list, tuple)) and raw:
        try:
            return torch.as_tensor([int(v) for v in raw], device=device, dtype=torch.long).reshape(-1)
        except Exception:
            return None
    return None


def _select_pose_hist_initial_norm(
    pose_hist_seq: Optional[torch.Tensor],
    *,
    step: int,
    batch_size: int,
    pose_hist_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if pose_hist_seq is not None and pose_hist_seq.dim() == 3 and int(pose_hist_seq.size(1)) > int(step):
        return pose_hist_seq[:, step]
    if pose_hist_seq is not None and pose_hist_seq.dim() == 2:
        return pose_hist_seq
    return torch.zeros((batch_size, pose_hist_dim), device=device, dtype=dtype)


def _init_eval_pose_hist_state(
    trainer: Trainer,
    *,
    ref_tensor: torch.Tensor,
    pose_hist_seq: Optional[torch.Tensor],
    step: int,
    device: torch.device,
    dtype: torch.dtype,
) -> PoseHistState:
    pose_hist_len = int(getattr(trainer, "pose_hist_len", 0) or 0)
    pose_hist_dim = int(getattr(trainer, "pose_hist_dim", 0) or 0)
    initial_norm = _select_pose_hist_initial_norm(
        pose_hist_seq,
        step=step,
        batch_size=int(ref_tensor.shape[0]),
        pose_hist_dim=pose_hist_dim,
        device=device,
        dtype=dtype,
    )
    return init_pose_hist_state(
        ref_tensor=ref_tensor,
        pose_hist_seq=initial_norm,
        y_prev_raw=None,
        rot_slice=None,
        pose_hist_len=pose_hist_len,
        pose_hist_dim=pose_hist_dim,
        params_fn=trainer._pose_hist_params,
    )


def _resolve_eval_pose_hist_input(
    *,
    state: PoseHistState,
    pose_hist_seq: Optional[torch.Tensor],
    idx: int,
    source: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    source_norm = str(source or "buffer").strip().lower()
    if source_norm == "zero":
        return torch.zeros((batch_size, int(state.dim)), device=device, dtype=dtype)
    if source_norm == "seq":
        resolved = resolve_pose_hist_input(
            state=PoseHistState(
                enabled=False,
                length=state.length,
                dim=state.dim,
                stride=state.stride,
            ),
            pose_hist_seq=pose_hist_seq,
            idx=idx,
        )
    else:
        resolved = resolve_pose_hist_input(
            state=state,
            pose_hist_seq=None,
            idx=idx,
        )
    if resolved is None and int(state.dim) > 0:
        return torch.zeros((batch_size, int(state.dim)), device=device, dtype=dtype)
    return resolved


def _compose_pose_hist_hybrid_rot_write(
    current_rot_write: Optional[torch.Tensor],
    donor_rot_write: Optional[torch.Tensor],
    *,
    leg_joint_idx: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if (not torch.is_tensor(current_rot_write)) or (not torch.is_tensor(donor_rot_write)):
        return None
    if current_rot_write.shape != donor_rot_write.shape:
        return None
    if current_rot_write.ndim < 2:
        return None
    rot_dim = int(current_rot_write.shape[-1])
    if rot_dim <= 0 or (rot_dim % 6) != 0:
        return None
    if not torch.is_tensor(leg_joint_idx) or int(leg_joint_idx.numel()) <= 0:
        return donor_rot_write
    joint_count = int(rot_dim // 6)
    idx = leg_joint_idx.to(device=current_rot_write.device, dtype=torch.long).reshape(-1)
    keep = (idx >= 0) & (idx < joint_count)
    if not bool(keep.any().detach().cpu().item()):
        return donor_rot_write
    idx = idx[keep]
    cur = current_rot_write.reshape(*current_rot_write.shape[:-1], joint_count, 6)
    donor = donor_rot_write.reshape(*donor_rot_write.shape[:-1], joint_count, 6).clone()
    donor[..., idx, :] = cur[..., idx, :]
    return donor.reshape(*current_rot_write.shape[:-1], rot_dim)


class _PretrainContactTinyAnchor(torch.nn.Module):
    """Tiny recurrent calibrator for pretrain_contact probabilities."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.gru = torch.nn.GRUCell(self.input_dim, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward_step(self, x: torch.Tensor, h: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            h = torch.zeros((int(x.shape[0]), self.hidden_dim), device=x.device, dtype=x.dtype)
        h2 = self.gru(x, h)
        logits = self.out(h2)
        return logits, h2


def _resolve_npz_path(clip_name: str, source_json: Optional[str], npz_root: Path) -> Path:
    candidates: List[Path] = []
    if npz_root:
        candidates.append(npz_root / f"{clip_name}.npz")
    if source_json:
        src_path = Path(source_json)
        if not src_path.is_absolute():
            src_path = (Path.cwd() / src_path).resolve()
        candidates.append(src_path.with_suffix(".npz"))
        if "processed_data" not in src_path.parts:
            candidates.append(src_path.parent / "processed_data" / f"{clip_name}.npz")
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"Processed NPZ for clip '{clip_name}' not found. Tried: "
        + ", ".join(str(c) for c in candidates)
    )

def _min_length(*arrays: Optional[np.ndarray]) -> int:
    lengths = [arr.shape[0] for arr in arrays if isinstance(arr, np.ndarray) and arr.shape[0] > 0]
    if not lengths:
        raise ValueError("No valid arrays to determine sequence length.")
    return min(lengths)


def _parse_int_list_spec(spec: Any, *, n: int, default: int = 0) -> List[int]:
    """
    Parse an int or comma-separated int list into a list of length n.

    - Empty/invalid => [default]*n
    - Single value => broadcast
    - Short list => pad with last
    - Long list => truncate
    """
    n = max(0, int(n))
    if n <= 0:
        return []
    s = str(spec or "").strip()
    if not s:
        return [int(default)] * n

    vals: List[int] = []
    try:
        if "," in s:
            for tok in s.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    vals.append(int(tok))
                except Exception:
                    pass
        else:
            try:
                vals.append(int(s))
            except Exception:
                vals = []
    except Exception:
        vals = []

    if not vals:
        vals = [int(default)]
    if len(vals) == 1 and n > 1:
        vals = vals * n
    if len(vals) < n:
        vals = vals + [int(vals[-1])] * (n - len(vals))
    if len(vals) > n:
        vals = vals[:n]
    return [int(v) for v in vals]


def _parse_float_list_spec(
    spec: Any,
    *,
    n: int,
    default: float = 0.0,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
) -> List[float]:
    """
    Parse a float or comma-separated float list into a list of length n.

    - Empty/invalid => [default]*n
    - Single value => broadcast
    - Short list => pad with last
    - Long list => truncate

    clamp_min/clamp_max apply after padding/truncation when provided.
    """
    n = max(0, int(n))
    if n <= 0:
        return []
    s = str(spec or "").strip()
    if not s:
        out = [float(default)] * n
    else:
        vals: List[float] = []
        try:
            if "," in s:
                for tok in s.split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    try:
                        vals.append(float(tok))
                    except Exception:
                        pass
            else:
                try:
                    vals.append(float(s))
                except Exception:
                    vals = []
        except Exception:
            vals = []

        if not vals:
            vals = [float(default)]
        if len(vals) == 1 and n > 1:
            vals = vals * n
        if len(vals) < n:
            vals = vals + [float(vals[-1])] * (n - len(vals))
        if len(vals) > n:
            vals = vals[:n]
        out = [float(v) for v in vals]

    if clamp_min is not None:
        out = [max(float(clamp_min), float(v)) for v in out]
    if clamp_max is not None:
        out = [min(float(clamp_max), float(v)) for v in out]
    return out


def _ttc_from_events_cyclic_np(
    events: np.ndarray,
    *,
    ttc_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cyclic TTC (time-to-next-event) computed from a precomputed event table.

    Args:
        events: (T,C) bool. Should contain at least one True per channel to be valid.
        ttc_max: optional cap in frames (None disables).

    Returns:
        ttc: (T,C) float32
        valid: (T,C) bool
    """
    ev = np.asarray(events, dtype=bool)
    if ev.ndim != 2:
        raise ValueError(f"events must be 2D (T,C), got shape={ev.shape}")
    T, C = int(ev.shape[0]), int(ev.shape[1])
    if T <= 0 or C <= 0:
        z_ttc = np.zeros((max(0, T), max(0, C)), dtype=np.float32)
        z_valid = np.zeros_like(z_ttc, dtype=bool)
        return z_ttc, z_valid

    next_idx = np.full((T, C), -1, dtype=np.int64)
    t_idx = np.arange(T, dtype=np.int64)
    for ch in range(C):
        idx = np.where(ev[:, ch])[0].astype(np.int64)
        if idx.size == 0:
            continue
        first = int(idx.min())
        nxt = -1
        for t in range(T - 1, -1, -1):
            if bool(ev[t, ch]):
                nxt = int(t)
            next_idx[t, ch] = int(nxt)
        # Wrap-around for positions after the last event.
        for t in range(T):
            if next_idx[t, ch] < 0:
                next_idx[t, ch] = first + T

    valid = next_idx >= 0
    ttc = (next_idx - t_idx[:, None]).astype(np.float32)
    ttc[~valid] = 0.0

    if ttc_max is not None:
        ttc_max_i = int(ttc_max)
        if ttc_max_i >= 0:
            np.clip(ttc, 0.0, float(ttc_max_i), out=ttc)
    return ttc, valid


def _apply_direct_leg_so3_correction_norm(
    trainer: Trainer,
    model: Any,
    direct_norm: torch.Tensor,
    omega_leg: torch.Tensor,
    *,
    columns: Tuple[str, str] = ("X", "Z"),
    omega_scale: float = 1.0,
    omega_sign: float = 1.0,
    apply_side: str = "left",
) -> torch.Tensor:
    """
    Apply leg-specific SO(3) residual to the *direct* branch output.

    The model outputs:
      - out_direct: normalized absolute Y
      - direct_leg_omega: axis-angle omega in rad for selected joints

    This function performs: denorm -> compose on SO(3) -> renorm.
    """
    if direct_norm is None or (not torch.is_tensor(direct_norm)):
        return direct_norm
    if omega_leg is None or (not torch.is_tensor(omega_leg)):
        return direct_norm
    try:
        direct_raw = trainer._denorm(direct_norm)
    except Exception:
        return direct_norm
    if direct_raw is None or (not torch.is_tensor(direct_raw)) or direct_raw.shape != direct_norm.shape:
        return direct_norm

    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        rot_slice = slice(0, direct_raw.shape[-1])
    rot_len = int(rot_slice.stop - rot_slice.start)
    if rot_len <= 0 or (rot_len % 6) != 0:
        return direct_norm
    J = int(rot_len // 6)

    leg_idx = getattr(model, "direct_pose_leg_joint_idx_tensor", None)
    if not torch.is_tensor(leg_idx) or int(leg_idx.numel()) <= 0:
        return direct_norm
    idx_use = leg_idx.to(device=direct_raw.device)

    # Normalize omega shape to (B,K,3).
    if omega_leg.dim() == 4 and omega_leg.size(1) == 1:
        omega_leg = omega_leg[:, 0]
    if omega_leg.dim() != 3 or omega_leg.shape[0] != direct_raw.shape[0] or omega_leg.shape[-1] != 3:
        return direct_norm
    if int(omega_leg.shape[1]) != int(idx_use.numel()):
        return direct_norm

    # Exclude root if present (keep omega aligned).
    try:
        root_idx = int(getattr(getattr(trainer, "loss_fn", None), "root_idx", 0) or 0)
    except Exception:
        root_idx = 0
    if 0 <= int(root_idx) < J and bool((idx_use == int(root_idx)).any().detach().cpu().item()):
        keep = (idx_use != int(root_idx))
        if bool(keep.any().detach().cpu().item()):
            idx_use = idx_use[keep]
            omega_leg = omega_leg[:, keep, :]
    if int(idx_use.numel()) <= 0:
        return direct_norm

    try:
        base6 = reproject_rot6d(direct_raw[..., rot_slice]).view(direct_raw.shape[0], J, 6)
        R_base = rot6d_to_matrix(base6, columns=columns)  # (B,J,3,3)
        R_leg_base = R_base[:, idx_use, :, :]
        if bool(getattr(model, "direct_pose_leg_stopgrad_main", False)):
            R_leg_base = R_leg_base.detach()
        try:
            s = float(omega_scale)
        except Exception:
            s = 1.0
        try:
            sg = float(omega_sign)
        except Exception:
            sg = 1.0
        if (abs(s - 1.0) > 1e-12) or (abs(sg - 1.0) > 1e-12):
            omega_leg = omega_leg * (s * sg)
        R_delta = so3_exp_map(omega_leg)  # (B,K,3,3)
        side = str(apply_side or "left").strip().lower()
        if side in ("right", "post", "postmul", "r"):
            R_leg = torch.matmul(R_leg_base, R_delta)
        else:
            R_leg = torch.matmul(R_delta, R_leg_base)
        R_final = R_base.clone()
        R_final[:, idx_use, :, :] = R_leg
        rot6_final = matrix_to_rot6d(R_final, columns=columns).view(direct_raw.shape[0], rot_len)
        direct_raw = direct_raw.clone()
        direct_raw[..., rot_slice] = rot6_final
    except Exception:
        return direct_norm
    try:
        return trainer._norm_y(direct_raw)
    except Exception:
        return direct_norm


# ---- Runner -----------------------------------------------------------------


class FreeRunCycleRunner:
    """
    Lightweight wrapper that reuses the TeacherRollout stack but runs free‑run.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        if not args.model:
            raise SystemExit("[FATAL] --model must be specified (ONNX not supported here).")
        self.device = self._resolve_device(args.device)
        self.bundle_path = Path(args.bundle).expanduser().resolve()
        self.bundle = LayoutCenter(str(self.bundle_path))
        pretrain_path = Path(args.pretrain_template).expanduser()
        self.norm_spec = _merge_norm_spec(self.bundle_path, pretrain_path if pretrain_path.is_file() else None)
        self.pose_hist_len = int(self.norm_spec.get("pose_hist_len", 0) or 0)

        ckpt = torch.load(Path(args.model).expanduser(), map_location="cpu")
        self._ckpt_posttrain_cfg = None
        try:
            if isinstance(ckpt, dict):
                cfg = ckpt.get("posttrain_cfg", None)
                if isinstance(cfg, dict):
                    self._ckpt_posttrain_cfg = dict(cfg)
        except Exception:
            self._ckpt_posttrain_cfg = None
        raw_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # Drop frozen_encoder / frozen_period_head weights that are not part of the runtime model
        # to avoid noisy mismatch warnings during load_state_dict(strict=False).
        self.state_dict = {}
        skipped = 0
        for k, v in raw_state.items():
            if (
                k.startswith("frozen_encoder.")
                or k.startswith("frozen_period_head.")
                or k.startswith("contact_plan_input_proj.")
            ):
                skipped += 1
                continue
            self.state_dict[k] = v
        if skipped > 0:
            print(f"[FreeRun][INFO] stripped {skipped} frozen encoder keys from checkpoint for runtime load.")

        self.width = self._infer_width()
        self.period_dim = self._infer_period_dim()
        self.encoder_bundle_path = Path(args.encoder_bundle).expanduser() if args.encoder_bundle else None

        self.model: Optional[EventMotionModel] = None
        self.loss_fn: Optional[MotionJointLoss] = None
        self.trainer: Optional[Trainer] = None
        self.contact_dim: Optional[int] = None
        self.angvel_dim: Optional[int] = None
        self.pose_hist_dim: Optional[int] = None
        self.angvel_meta: Dict[str, Any] = {
            "mode": None,
            "mu": None,
            "std": None,
        }
        self.so3_corr_apply = bool(getattr(args, "so3_corr_apply", False))
        self.so3_corr_max_deg = float(getattr(args, "so3_corr_max_deg", 20.0) or 20.0)
        gate_force_raw = getattr(args, "so3_corr_gate_force", None)
        self.so3_corr_gate_force = None
        if gate_force_raw is not None:
            s = str(gate_force_raw).strip().lower()
            if s in ("", "none", "null"):
                self.so3_corr_gate_force = None
            else:
                self.so3_corr_gate_force = float(gate_force_raw)
        self.so3_corr_gate_from_contacts_err = bool(getattr(args, "so3_corr_gate_from_contacts_err", False))
        self.so3_corr_gate_from_contacts_err_mode = str(getattr(args, "so3_corr_gate_from_contacts_err_mode", "scale") or "scale").lower()
        self.so3_corr_gate_err_k = float(getattr(args, "so3_corr_gate_err_k", 1.0) or 1.0)
        self.so3_corr_gate_err_bias = float(getattr(args, "so3_corr_gate_err_bias", 0.0) or 0.0)
        self.so3_corr_gate_err_max = float(getattr(args, "so3_corr_gate_err_max", 1.0) or 1.0)
        self.so3_corr_gate_err_ref_steps = int(getattr(args, "so3_corr_gate_err_ref_steps", 8) or 8)
        self.so3_corr_gate_err_margin = float(getattr(args, "so3_corr_gate_err_margin", 0.0) or 0.0)
        self.so3_corr_gate_err_use_ref = bool(getattr(args, "so3_corr_gate_err_use_ref", False))
        self.so3_corr_gate_scale_max = float(getattr(args, "so3_corr_gate_scale_max", 2.0) or 2.0)
        self.log_contacts_whitebox = bool(getattr(args, "log_contacts_whitebox", False))
        self.log_contacts_whitebox_first_steps = max(
            0, int(getattr(args, "log_contacts_whitebox_first_steps", 4) or 4)
        )
        self.log_contact_plan_logits_decomp = bool(getattr(args, "log_contact_plan_logits_decomp", False))
        # Exporting contact_meas swap/IO diagnostics requires per-step contact logging.
        self.export_contact_meas_head_swap = bool(getattr(args, "export_contact_meas_head_swap", False))
        self.log_contacts = bool(
            getattr(args, "log_contacts", False)
            or self.so3_corr_gate_from_contacts_err
            or self.log_contacts_whitebox
            or self.log_contact_plan_logits_decomp
            or self.export_contact_meas_head_swap
        )
        try:
            self.contact_plan_inject_scale = float(getattr(args, "contact_plan_inject_scale", 1.0))
        except Exception:
            self.contact_plan_inject_scale = 1.0
        try:
            self.contact_plan_time_bias_scale = float(getattr(args, "contact_plan_time_bias_scale", 1.0))
        except Exception:
            self.contact_plan_time_bias_scale = 1.0
        gate_raw = str(getattr(args, "contact_meas_gate_by_hit", "auto") or "auto").strip().lower()
        if gate_raw in ("true", "1", "yes", "y"):
            self.contact_meas_gate_by_hit_override = True
        elif gate_raw in ("false", "0", "no", "n"):
            self.contact_meas_gate_by_hit_override = False
        else:
            self.contact_meas_gate_by_hit_override = None
        self.contact_meas_vxy_mode = str(getattr(args, "contact_meas_vxy_mode", "abs") or "abs").strip().lower()
        self.contact_meas_ground_z_mode = str(getattr(args, "contact_meas_ground_z_mode", "window") or "window").strip().lower()
        self.contact_meas_ground_z_beta = float(getattr(args, "contact_meas_ground_z_beta", 0.05) or 0.05)
        self.contact_meas_ground_z_window = int(getattr(args, "contact_meas_ground_z_window", 5) or 5)
        self.contact_meas_ground_z_quantile = float(getattr(args, "contact_meas_ground_z_quantile", 0.2) or 0.2)
        self.contact_meas_ground_z_slew_up_cm = float(getattr(args, "contact_meas_ground_z_slew_up_cm", 0.0) or 0.0)
        self.contact_meas_ground_z_slew_down_cm = float(getattr(args, "contact_meas_ground_z_slew_down_cm", 0.0) or 0.0)
        # Optional: force contact_plan init behavior for ablations (even if ckpt lacks init_head weights).
        self.contact_plan_init_mode_override = getattr(args, "contact_plan_init_mode", None)
        if self.contact_plan_init_mode_override is not None:
            mode = str(self.contact_plan_init_mode_override).strip().lower()
            if mode in ("learnable_obs", "obs+learnable"):
                mode = "learnable+obs"
            self.contact_plan_init_mode_override = mode
        self.contact_plan_init_hidden_override = getattr(args, "contact_plan_init_hidden", None)
        self.contact_plan_init_dropout_override = getattr(args, "contact_plan_init_dropout", None)
        self.direct_pose_meas_source = str(getattr(args, "direct_pose_meas_source", "model") or "model").strip().lower()
        self.direct_pose_meas_warmup_steps = max(0, int(getattr(args, "direct_pose_meas_warmup_steps", 0) or 0))
        self.direct_pose_plan_source = str(getattr(args, "direct_pose_plan_source", "model") or "model").strip().lower()
        # Optional: distribution-matched "soft-GT" hint mapping for direct plan/meas overrides.
        self.direct_pose_softgt_stats_spec = getattr(args, "direct_pose_softgt_stats", None)
        self.direct_pose_softgt_stats = None
        try:
            raw = self.direct_pose_softgt_stats_spec
            if raw is not None:
                s = str(raw).strip()
                if s:
                    p = Path(s).expanduser()
                    if p.is_file():
                        payload = _load_json(p)
                    else:
                        payload = json.loads(s)
                    if isinstance(payload, dict) and payload:
                        self.direct_pose_softgt_stats = payload
        except Exception:
            self.direct_pose_softgt_stats = None
        self.direct_pose_hinge_enable = bool(getattr(args, "direct_pose_hinge_enable", False))
        self.direct_pose_hinge_bones = str(getattr(args, "direct_pose_hinge_bones", "calf_r") or "calf_r")
        self.direct_pose_hinge_axis = str(getattr(args, "direct_pose_hinge_axis", "z") or "z").strip().lower()
        # Diagnostics: replace the model's hinge head output with an axis-oracle delta computed from GT,
        # then apply it via the normal hinge correction path (to verify apply/target consistency).
        self.direct_pose_hinge_oracle_delta = bool(getattr(args, "direct_pose_hinge_oracle_delta", False))
        try:
            self.direct_pose_hinge_max_deg = float(getattr(args, "direct_pose_hinge_max_deg", 45.0) or 45.0)
        except Exception:
            self.direct_pose_hinge_max_deg = 45.0
        try:
            self.direct_pose_hinge_hidden = int(getattr(args, "direct_pose_hinge_hidden", 0) or 0)
        except Exception:
            self.direct_pose_hinge_hidden = 0
        # Cross-leg feature ablations (evaluation-time only).
        self.direct_pose_leg_cross_leg_ablate = str(
            getattr(args, "direct_pose_leg_cross_leg_ablate", "none") or "none"
        ).strip().lower()
        self.direct_pose_leg_side_plan_other_ablate = str(
            getattr(args, "direct_pose_leg_side_plan_other_ablate", "none") or "none"
        ).strip().lower()
        # Ablation: override the runtime contacts_meas used by contacts_err / Event-Clock (and thus λ diagnostics).
        # This is a *runtime* switch only (no weight changes).
        self.contacts_meas_source = str(getattr(args, "contacts_meas_source", "model") or "model").strip().lower()
        try:
            self.contacts_meas_pretrain_clamp = float(getattr(args, "contacts_meas_pretrain_clamp", 1.0) or 0.0)
        except Exception:
            self.contacts_meas_pretrain_clamp = 1.0
        if not np.isfinite(float(self.contacts_meas_pretrain_clamp)):
            self.contacts_meas_pretrain_clamp = 1.0
        self.contacts_meas_pretrain_clamp = float(max(0.0, float(self.contacts_meas_pretrain_clamp)))
        # Optional: logit-space affine calibration for pretrain_contact source.
        # Payload schema (JSON file or inline JSON): {"scale":[...], "bias":[...], "eps":1e-4}
        self.contacts_meas_pretrain_affine_stats_spec = getattr(args, "contacts_meas_pretrain_affine_stats", None)
        self.contacts_meas_pretrain_affine = None
        try:
            raw = self.contacts_meas_pretrain_affine_stats_spec
            if raw is not None:
                s = str(raw).strip()
                if s:
                    p = Path(s).expanduser()
                    if p.is_file():
                        payload = _load_json(p)
                    else:
                        payload = json.loads(s)
                    cfg = payload.get("pretrain_contact_affine") if isinstance(payload, dict) else None
                    if not isinstance(cfg, dict):
                        cfg = payload if isinstance(payload, dict) else None
                    if isinstance(cfg, dict):
                        scale = cfg.get("scale", None)
                        bias = cfg.get("bias", None)
                        if isinstance(scale, (list, tuple)) and isinstance(bias, (list, tuple)):
                            if len(scale) == len(bias) and len(scale) > 0:
                                try:
                                    sc = [float(x) for x in scale]
                                    bs = [float(x) for x in bias]
                                    eps = float(cfg.get("eps", 1e-4) or 1e-4)
                                    if not np.isfinite(eps):
                                        eps = 1e-4
                                    eps = float(min(1e-2, max(1e-8, eps)))
                                    self.contacts_meas_pretrain_affine = {
                                        "scale": sc,
                                        "bias": bs,
                                        "eps": eps,
                                    }
                                except Exception:
                                    self.contacts_meas_pretrain_affine = None
        except Exception:
            self.contacts_meas_pretrain_affine = None
        # Optional: tiny recurrent anchor for pretrain_contact source.
        # Checkpoint schema (torch.save dict): {"kind":"pretrain_contact_anchor","config":...,"state_dict":...}
        self.contacts_meas_pretrain_anchor_ckpt_spec = getattr(args, "contacts_meas_pretrain_anchor_ckpt", None)
        self.contacts_meas_pretrain_anchor = None
        self.contacts_meas_pretrain_anchor_config = None
        try:
            raw = self.contacts_meas_pretrain_anchor_ckpt_spec
            if raw is not None:
                s = str(raw).strip()
                if s:
                    p = Path(s).expanduser()
                    payload = torch.load(p, map_location="cpu")
                    if isinstance(payload, dict):
                        cfg = payload.get("config", None)
                        st = payload.get("state_dict", None)
                        if isinstance(cfg, dict) and isinstance(st, dict):
                            in_dim = int(cfg.get("input_dim", 4) or 4)
                            hid = int(cfg.get("hidden_dim", 16) or 16)
                            out_dim = int(cfg.get("output_dim", 2) or 2)
                            mdl = _PretrainContactTinyAnchor(input_dim=in_dim, hidden_dim=hid, output_dim=out_dim)
                            mdl.load_state_dict(st, strict=True)
                            mdl.eval()
                            self.contacts_meas_pretrain_anchor = mdl
                            self.contacts_meas_pretrain_anchor_config = {
                                "input_dim": int(in_dim),
                                "hidden_dim": int(hid),
                                "output_dim": int(out_dim),
                                "delta_scale": float(cfg.get("delta_scale", 1.0) or 1.0),
                            }
        except Exception:
            self.contacts_meas_pretrain_anchor = None
            self.contacts_meas_pretrain_anchor_config = None
        # Debug-only: post-process learned contact_meas_head output (when contacts_meas_source=model).
        try:
            s = float(getattr(args, "contacts_meas_model_logit_scale", 1.0) or 1.0)
        except Exception:
            s = 1.0
        if not np.isfinite(float(s)):
            s = 1.0
        self.contacts_meas_model_logit_scale = float(max(1e-6, float(s)))
        self.contacts_meas_model_onehot = bool(getattr(args, "contacts_meas_model_onehot", False))
        # More reasonable diagnostic: only enforce one-hot in (likely) single-support regions.
        # Detect "double support" as >=2 channels whose prob exceeds a threshold, and skip one-hot there.
        self.contacts_meas_model_onehot_conditional = bool(getattr(args, "contacts_meas_model_onehot_conditional", False))
        try:
            thr = float(getattr(args, "contacts_meas_model_onehot_ds_thr", 0.5) or 0.5)
        except Exception:
            thr = 0.5
        if not np.isfinite(float(thr)):
            thr = 0.5
        self.contacts_meas_model_onehot_ds_thr = float(max(0.0, min(1.0, float(thr))))
        # Debug-only: override contacts_meas with GT only on selected step_in_cycle (sic), while keeping
        # contacts_meas_source=model elsewhere. This is for causal localization of rare tail spikes.
        self.contacts_meas_gt_override_sics = str(getattr(args, "contacts_meas_gt_override_sics", "") or "").strip()
        try:
            self.contacts_meas_gt_override_cycle_gte = int(getattr(args, "contacts_meas_gt_override_cycle_gte", 1) or 1)
        except Exception:
            self.contacts_meas_gt_override_cycle_gte = 1
        self.contacts_meas_gt_override_cycle_gte = max(0, int(self.contacts_meas_gt_override_cycle_gte))
        self.contacts_meas_gt_override_drop_wrap = str(
            getattr(args, "contacts_meas_gt_override_drop_wrap", "on") or "on"
        ).strip().lower()
        # Phase reset / clock anchor source (contact crossing vs TTC countdown).
        self.phase_reset_source = str(getattr(args, "phase_reset_source", "contacts_meas") or "contacts_meas").strip().lower()
        # If enabled, abort when the requested phase_reset_source cannot be applied and would fall back to contacts_meas.
        # This prevents accidental apples-to-oranges acceptance due to silent fallback when a head is missing in the ckpt.
        self.phase_reset_source_strict = str(getattr(args, "phase_reset_source_strict", "off") or "off").strip().lower()
        self.ttc_event_kind = str(getattr(args, "ttc_event_kind", "touchdown") or "touchdown").strip().lower()
        self.ttc_max = getattr(args, "ttc_max", None)
        # Debug-only: shift TTC_gt event frames (within each cycle) before using them as phase reset anchors.
        self.ttc_gt_event_shift = str(getattr(args, "ttc_gt_event_shift", "") or "").strip()
        self.ttc_apply_phase_reset_to_phase_z = str(
            getattr(args, "ttc_apply_phase_reset_to_phase_z", "on") or "on"
        ).strip().lower()
        if self.phase_reset_source in ("hazard", "tdhazard", "td_hazard", "tdhaz"):
            raise SystemExit(
                "[FATAL] phase_reset_source=td_hazard has been retired from the current mainline. "
                "Use --phase_reset_source contacts_meas, ttc_gt, or none."
            )
        self.lambda_fusion_apply = bool(getattr(args, "lambda_fusion_apply", False))
        # Stage2: deterministic r_t (shared with posttrain) for λ modulation.
        def _cfg_get(name: str, default: Any) -> Any:
            v = getattr(args, name, None)
            if v is not None:
                return v
            if isinstance(self._ckpt_posttrain_cfg, dict) and name in self._ckpt_posttrain_cfg:
                return self._ckpt_posttrain_cfg.get(name)
            return default

        self.lambda_reliability_mode = str(_cfg_get("lambda_reliability_mode", "none") or "none")
        self.lambda_reliability_warmup_steps = int(_cfg_get("lambda_reliability_warmup_steps", 0) or 0)
        self.lambda_reliability_contact_err_max = float(_cfg_get("lambda_reliability_contact_err_max", 1.0) or 1.0)
        self.lambda_reliability_warmup_joint_scales = None
        try:
            raw_scales = _cfg_get("lambda_reliability_warmup_joint_scales", None)
            if raw_scales is not None:
                payload = raw_scales
                if isinstance(payload, str):
                    s = payload.strip()
                    if s:
                        try:
                            p = Path(s).expanduser()
                            if p.is_file():
                                payload = _load_json(p)
                            else:
                                payload = json.loads(s)
                        except Exception:
                            payload = None
                if isinstance(payload, dict):
                    payload = payload.get("scales", payload.get("values", None))
                if isinstance(payload, (list, tuple)) and payload:
                    self.lambda_reliability_warmup_joint_scales = [float(x) for x in payload]
        except Exception:
            self.lambda_reliability_warmup_joint_scales = None
        self.normalizer: Optional[DataNormalizer] = None
        self._pose_hist_hybrid_donor_runner: Optional["FreeRunCycleRunner"] = None
        self._pose_hist_hybrid_donor_ckpt_path: Optional[Path] = None

    @staticmethod
    def _resolve_device(pref: str) -> torch.device:
        pref = pref.lower()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pref == "mps":
            return torch.device("mps" if has_mps else "cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if has_mps:
            return torch.device("mps")
        return torch.device("cpu")

    def _infer_width(self) -> int:
        key = "shared_encoder.0.weight"
        if key not in self.state_dict:
            raise KeyError(f"Checkpoint missing key '{key}' to infer hidden width.")
        return int(self.state_dict[key].shape[0])

    def _infer_period_dim(self) -> int:
        key = "period_encoder.weight"
        if key in self.state_dict:
            return int(self.state_dict[key].shape[1])
        return 0

    def _build_dataset(self, npz_path: Path, seq_len: int) -> MotionEventDataset:
        ds = MotionEventDataset(
            data_dir=str(npz_path.parent),
            seq_len=max(2, int(seq_len)),
            paths=[str(npz_path)],
            pose_hist_len=self.pose_hist_len,
            norm_spec=self.norm_spec,
        )
        if not ds.clips:
            raise RuntimeError(f"No clips loaded from {npz_path}")
        return ds

    def _ensure_model_ready(self, ds: MotionEventDataset) -> None:
        Dx, Dy, Dc = int(ds.Dx), int(ds.Dy), int(ds.Dc)
        self.contact_dim = int(getattr(ds, "contact_dim", 0))
        self.angvel_dim = int(getattr(ds, "angvel_dim", 0))
        self.pose_hist_dim = int(getattr(ds, "pose_hist_dim", 0))
        self.bundle.strict_validate(Dx, Dy)
        if self.model is not None:
            return

        contact_plan_enable = any(
            str(k).startswith("contact_plan_cell.") or str(k).startswith("contact_plan_head.")
            for k in self.state_dict.keys()
        )
        contact_plan_hidden = 64
        try:
            w_hh = self.state_dict.get("contact_plan_cell.weight_hh", None)
            if torch.is_tensor(w_hh) and w_hh.ndim == 2:
                contact_plan_hidden = int(w_hh.shape[1])
        except Exception:
            pass
        contact_plan_time_pe_dim = 0
        try:
            w_time = self.state_dict.get("contact_plan_time_head.weight", None)
            if torch.is_tensor(w_time) and w_time.ndim == 2:
                contact_plan_time_pe_dim = int(w_time.shape[1])
        except Exception:
            contact_plan_time_pe_dim = 0
        # Infer obs-conditioned contact plan init head (plan_z0 = init_z + init_head(obs0)).
        contact_plan_init_mode = "learnable"
        contact_plan_init_hidden = 128
        contact_plan_init_dropout = 0.0
        try:
            init_has_weights = any(str(k).startswith("contact_plan_init_head.") for k in self.state_dict.keys())
            if init_has_weights:
                contact_plan_init_mode = "learnable+obs"
                w_init = self.state_dict.get("contact_plan_init_head.1.weight", None)
                if torch.is_tensor(w_init) and w_init.ndim == 2:
                    contact_plan_init_hidden = int(w_init.shape[0])
        except Exception:
            contact_plan_init_mode = "learnable"
        # Allow overriding init mode for ablations (create init_head even if ckpt doesn't have weights).
        if self.contact_plan_init_mode_override is not None:
            contact_plan_init_mode = str(self.contact_plan_init_mode_override)
        if self.contact_plan_init_hidden_override is not None:
            try:
                contact_plan_init_hidden = int(self.contact_plan_init_hidden_override)
            except Exception:
                pass
        if self.contact_plan_init_dropout_override is not None:
            try:
                contact_plan_init_dropout = float(self.contact_plan_init_dropout_override)
            except Exception:
                contact_plan_init_dropout = 0.0
        # Infer trunk injection mode from checkpoint shared_encoder input dim.
        contact_plan_inject = "none"
        try:
            w0 = self.state_dict.get("shared_encoder.0.weight", None)
            if torch.is_tensor(w0) and w0.ndim == 2:
                nin = int(w0.shape[1])
                base_in = int(Dx + Dc)
                extra = int(max(0, nin - base_in))
                if extra > 0:
                    # If extra matches contact_dim, assume contacts injection; otherwise plan_z injection.
                    if int(self.contact_dim) > 0 and extra == int(self.contact_dim):
                        contact_plan_inject = "contacts"
                    else:
                        contact_plan_inject = "plan_z"
                        # Ensure plan hidden matches injected dim (prefer actual injected size).
                        if extra != int(contact_plan_hidden):
                            contact_plan_hidden = int(extra)
        except Exception:
            contact_plan_inject = "none"

        # Infer Event-Clock v3 (contact_plan residual correction) from checkpoint weights.
        event_clock_has_weights = any(
            str(k).startswith("event_clock_gate.") or str(k).startswith("event_clock_corrector.")
            for k in self.state_dict.keys()
        )
        event_clock_mode = str(getattr(self.args, "event_clock", "auto") or "auto").strip().lower()
        use_event_clock = bool(event_clock_has_weights)
        if event_clock_mode == "on":
            use_event_clock = True
        elif event_clock_mode == "off":
            use_event_clock = False
        event_clock_hidden_dim = 64
        event_clock_gate_hidden_dim = 32
        try:
            w_ec = self.state_dict.get("event_clock_corrector.correction_head.0.weight", None)
            if torch.is_tensor(w_ec) and w_ec.ndim == 2:
                event_clock_hidden_dim = int(w_ec.shape[0])
        except Exception:
            pass
        try:
            w_gate = self.state_dict.get("event_clock_gate.confidence_head.0.weight", None)
            if torch.is_tensor(w_gate) and w_gate.ndim == 2:
                event_clock_gate_hidden_dim = int(w_gate.shape[0])
        except Exception:
            pass
        if getattr(self.args, "event_clock_hidden_dim", None) is not None:
            try:
                event_clock_hidden_dim = int(self.args.event_clock_hidden_dim)
            except Exception:
                pass
        if getattr(self.args, "event_clock_gate_hidden_dim", None) is not None:
            try:
                event_clock_gate_hidden_dim = int(self.args.event_clock_gate_hidden_dim)
            except Exception:
                pass

        # Infer direct pose head (cond + contacts_plan -> absolute pose).
        # If we don't instantiate this head, load_state_dict(strict=False) will warn about unexpected keys
        # and the runtime model won't expose `out_direct`.
        direct_pose_enable = False
        direct_pose_hidden = 256
        direct_pose_meas_mode = "concat"
        direct_pose_feat_source = "cond"
        direct_pose_time_pe_dim = 0
        direct_pose_use_phase_z = False
        direct_pose_phase_z_mode = "concat"
        direct_pose_split_enable = False
        direct_pose_arm_split_enable = False
        direct_pose_arm_bones = None
        direct_pose_nonleg_proj_dim = 0
        try:
            if isinstance(self._ckpt_posttrain_cfg, dict):
                direct_pose_use_phase_z = bool(self._ckpt_posttrain_cfg.get("direct_pose_use_phase_z", False))
                v = self._ckpt_posttrain_cfg.get("direct_pose_phase_z_mode", None)
                if v is not None:
                    direct_pose_phase_z_mode = str(v).strip().lower() or "concat"
                direct_pose_split_enable = bool(self._ckpt_posttrain_cfg.get("direct_pose_split_enable", False))
                direct_pose_arm_split_enable = bool(self._ckpt_posttrain_cfg.get("direct_pose_arm_split_enable", False))
                direct_pose_arm_bones = self._ckpt_posttrain_cfg.get("direct_pose_arm_bones", None)
                try:
                    direct_pose_nonleg_proj_dim = int(self._ckpt_posttrain_cfg.get("direct_pose_nonleg_proj_dim", 0) or 0)
                except Exception:
                    direct_pose_nonleg_proj_dim = 0
        except Exception:
            direct_pose_use_phase_z = False
            direct_pose_phase_z_mode = "concat"
            direct_pose_split_enable = False
            direct_pose_arm_split_enable = False
            direct_pose_arm_bones = None
            direct_pose_nonleg_proj_dim = 0
        try:
            direct_has_weights = any(str(k).startswith("direct_pose_head.") for k in self.state_dict.keys())
            split_has_weights_nonleg = bool(
                any(str(k).startswith("direct_pose_out_leg.") for k in self.state_dict.keys())
                and any(str(k).startswith("direct_pose_out_nonleg.") for k in self.state_dict.keys())
            )
            split_has_weights_arm = bool(
                any(str(k).startswith("direct_pose_out_leg.") for k in self.state_dict.keys())
                and any(str(k).startswith("direct_pose_out_arm.") for k in self.state_dict.keys())
                and any(str(k).startswith("direct_pose_out_else.") for k in self.state_dict.keys())
            )
            split_has_weights = bool(split_has_weights_nonleg or split_has_weights_arm)
            direct_has_weights = bool(direct_has_weights or split_has_weights)
            if direct_has_weights and int(Dy) > 0 and int(Dc) > 0 and int(self.contact_dim) > 0:
                w_in = self.state_dict.get("direct_pose_head.0.weight", None)
                w_out = self.state_dict.get("direct_pose_head.6.weight", None)
                w_out_leg = self.state_dict.get("direct_pose_out_leg.weight", None)
                w_out_nonleg = self.state_dict.get("direct_pose_out_nonleg.weight", None)
                w_out_arm = self.state_dict.get("direct_pose_out_arm.weight", None)
                w_out_else = self.state_dict.get("direct_pose_out_else.weight", None)
                if torch.is_tensor(w_in) and w_in.ndim == 2:
                    in_dim = int(w_in.shape[1])
                    hid = int(w_in.shape[0])
                    out_dim = None
                    if torch.is_tensor(w_out) and w_out.ndim == 2:
                        out_dim = int(w_out.shape[0])
                        direct_pose_split_enable = False
                        direct_pose_arm_split_enable = False
                    elif (
                        torch.is_tensor(w_out_leg)
                        and w_out_leg.ndim == 2
                        and torch.is_tensor(w_out_nonleg)
                        and w_out_nonleg.ndim == 2
                    ):
                        out_dim = int(w_out_leg.shape[0] + w_out_nonleg.shape[0])
                        direct_pose_split_enable = True
                        direct_pose_arm_split_enable = False
                        # Split readout must share the same hidden trunk output dim.
                        if int(w_out_leg.shape[1]) > 0:
                            hid = int(w_out_leg.shape[1])
                        try:
                            nonleg_in_dim = int(w_out_nonleg.shape[1])
                            if nonleg_in_dim > 0 and int(nonleg_in_dim) != int(hid):
                                direct_pose_nonleg_proj_dim = int(nonleg_in_dim)
                        except Exception:
                            pass
                    elif (
                        torch.is_tensor(w_out_leg)
                        and w_out_leg.ndim == 2
                        and torch.is_tensor(w_out_arm)
                        and w_out_arm.ndim == 2
                        and torch.is_tensor(w_out_else)
                        and w_out_else.ndim == 2
                    ):
                        out_dim = int(w_out_leg.shape[0] + w_out_arm.shape[0] + w_out_else.shape[0])
                        direct_pose_split_enable = True
                        direct_pose_arm_split_enable = True
                        # Split readout must share the same hidden trunk output dim.
                        if int(w_out_leg.shape[1]) > 0:
                            hid = int(w_out_leg.shape[1])
                        try:
                            arm_in_dim = int(w_out_arm.shape[1])
                            if arm_in_dim > 0 and int(arm_in_dim) != int(hid):
                                direct_pose_nonleg_proj_dim = int(arm_in_dim)
                        except Exception:
                            pass
                    if out_dim is None:
                        raise SystemExit("[FATAL] direct_pose_head weights found but output readout weights are missing.")
                    try:
                        w_proj = self.state_dict.get("direct_pose_nonleg_proj.0.weight", None)
                        w_proj_arm = self.state_dict.get("direct_pose_arm_proj.0.weight", None)
                        w_proj_else = self.state_dict.get("direct_pose_else_proj.0.weight", None)
                        if torch.is_tensor(w_proj) and w_proj.ndim == 2 and int(w_proj.shape[0]) > 0:
                            direct_pose_nonleg_proj_dim = int(w_proj.shape[0])
                        elif torch.is_tensor(w_proj_arm) and w_proj_arm.ndim == 2 and int(w_proj_arm.shape[0]) > 0:
                            direct_pose_nonleg_proj_dim = int(w_proj_arm.shape[0])
                        elif torch.is_tensor(w_proj_else) and w_proj_else.ndim == 2 and int(w_proj_else.shape[0]) > 0:
                            direct_pose_nonleg_proj_dim = int(w_proj_else.shape[0])
                    except Exception:
                        pass
                    expected_out = int(Dy)
                    expected_out_modes = int(Dy) * 2
                    base_candidates = [
                        (int(Dc), "cond"),
                        (int(self.width), "hidden"),
                        (int(Dc + self.width), "cond+hidden"),
                    ]
                    Cc = int(self.contact_dim)

                    if out_dim == expected_out:
                        # concat mode: input = base + plan + meas (+ time_pe)
                        for base_dim, src in base_candidates:
                            phase_dim = int(2 * Cc) if bool(direct_pose_use_phase_z) else 0
                            if str(direct_pose_phase_z_mode or "concat").strip().lower() == "replace_contacts":
                                # input = base + time_pe + phase_z (no plan/meas)
                                tdim = int(in_dim - base_dim - phase_dim)
                            else:
                                tdim = int(in_dim - base_dim - (2 * Cc) - phase_dim)
                            if tdim >= 0 and tdim % 2 == 0:
                                direct_pose_enable = True
                                direct_pose_hidden = hid
                                direct_pose_meas_mode = "concat"
                                direct_pose_feat_source = src
                                direct_pose_time_pe_dim = int(tdim)
                                break
                    elif out_dim == expected_out_modes:
                        # mode_select: input = base + plan (+ time_pe)
                        for base_dim, src in base_candidates:
                            phase_dim = int(2 * Cc) if bool(direct_pose_use_phase_z) else 0
                            if str(direct_pose_phase_z_mode or "concat").strip().lower() == "replace_contacts":
                                raise SystemExit(
                                    "[FATAL] direct_pose_phase_z_mode='replace_contacts' is not supported for direct_pose_meas_mode='mode_select'."
                                )
                            tdim = int(in_dim - base_dim - Cc - phase_dim)
                            if tdim >= 0 and tdim % 2 == 0:
                                direct_pose_enable = True
                                direct_pose_hidden = hid
                                direct_pose_meas_mode = "mode_select"
                                direct_pose_feat_source = src
                                direct_pose_time_pe_dim = int(tdim)
                                break
                    else:
                        raise SystemExit(
                            f"[FATAL] Unrecognized direct_pose_head out_dim={out_dim} (expected {expected_out} or {expected_out_modes})."
                        )

                    if not direct_pose_enable:
                        raise SystemExit(
                            f"[FATAL] Unrecognized direct_pose_head shape: in_dim={in_dim} out_dim={out_dim} "
                            f"(cond_dim={Dc}, hidden_dim={self.width}, contact_dim={self.contact_dim})."
                        )
        except Exception:
            direct_pose_enable = False
            direct_pose_feat_source = "cond"
            direct_pose_time_pe_dim = 0
            direct_pose_use_phase_z = False
            direct_pose_split_enable = False
            direct_pose_nonleg_proj_dim = 0

        # Prefer checkpoint posttrain_cfg when present (cannot infer hidden_pre from tensor shapes).
        if isinstance(self._ckpt_posttrain_cfg, dict):
            try:
                v = self._ckpt_posttrain_cfg.get("direct_pose_feat_source", None)
                if v is not None:
                    s = str(v).strip().lower()
                    if s not in ("", "auto"):
                        if s in ("h", "h_final", "hidden_only", "post", "final"):
                            s = "hidden"
                        if s in ("h_pre", "h_temporal", "hidden_pre", "pre", "temporal", "mid"):
                            s = "hidden_pre"
                        if s in ("cond_hidden", "hidden_cond", "concat", "cond+hidden", "hidden+cond"):
                            s = "cond+hidden"
                        if s in ("cond+hidden_pre", "cond_hidden_pre", "hidden_pre+cond", "cond+pre", "pre+cond"):
                            s = "cond+hidden_pre"
                        if s in ("cond", "hidden", "hidden_pre", "cond+hidden", "cond+hidden_pre"):
                            direct_pose_feat_source = s
            except Exception:
                pass

        # Infer lambda fusion head (Stage2): must match ckpt shapes to avoid size mismatch errors.
        lambda_has_weights = any(str(k).startswith("lambda_fusion_head.") for k in self.state_dict.keys())
        lambda_fusion_enable = bool(lambda_has_weights)
        lambda_fusion_mode = "per_joint"
        lambda_fusion_hidden = 128
        lambda_fusion_use_rollout_step = False
        try:
            if lambda_has_weights:
                w_in = self.state_dict.get("lambda_fusion_head.1.weight", None)
                w_out = self.state_dict.get("lambda_fusion_head.4.weight", None)
                if torch.is_tensor(w_in) and w_in.ndim == 2:
                    lambda_fusion_hidden = int(w_in.shape[0])
                    base_in = int(self.width + (self.contact_dim if contact_plan_enable else 0))
                    in_features = int(w_in.shape[1])
                    if in_features == base_in + 1:
                        lambda_fusion_use_rollout_step = True
                    elif in_features == base_in:
                        lambda_fusion_use_rollout_step = False
                if torch.is_tensor(w_out) and w_out.ndim == 2:
                    out_dim = int(w_out.shape[0])
                    lambda_fusion_mode = "global" if out_dim == 1 else "per_joint"
        except Exception:
            lambda_fusion_enable = False

        # Important: EventMotionModel's period_dim can be mutated later by attach_motion_encoder().
        # Some checkpoints are trained with period_dim=0 at init (so Event-Clock ignores period_feat),
        # then period_dim is set by the attached encoder bundle (creating period_encoder weights).
        # To faithfully reconstruct such ckpts, infer the Event-Clock period_feat_dim from its weight shapes
        # and use it as the model init period_dim (then attach encoder bundle BEFORE loading weights).
        period_dim_ckpt = int(getattr(ds, "period_dim", 0) or self.period_dim)
        event_clock_period_feat_dim = None
        try:
            w0_ec = self.state_dict.get("event_clock_gate.confidence_head.0.weight", None)
            if torch.is_tensor(w0_ec) and w0_ec.ndim == 2:
                base = int(self.contact_dim) * 2 + 1
                event_clock_period_feat_dim = max(0, int(w0_ec.shape[1]) - base)
        except Exception:
            event_clock_period_feat_dim = None
        period_dim_init = int(period_dim_ckpt)
        try:
            if (
                bool(event_clock_has_weights)
                and event_clock_period_feat_dim is not None
                and int(event_clock_period_feat_dim) != int(period_dim_ckpt)
            ):
                period_dim_init = int(event_clock_period_feat_dim)
        except Exception:
            period_dim_init = int(period_dim_ckpt)
        if period_dim_init != int(period_dim_ckpt) and bool(event_clock_has_weights):
            if not (self.encoder_bundle_path and self.encoder_bundle_path.is_file()):
                print(
                    f"[FreeRun][WARN] ckpt period_dim={int(period_dim_ckpt)} but Event-Clock was initialized with "
                    f"period_feat_dim={int(period_dim_init)}; no encoder_bundle provided so period_encoder weights may be dropped. "
                    "Pass --encoder-bundle to fully reconstruct the model."
                )
            else:
                print(
                    f"[FreeRun][INFO] ckpt period_dim={int(period_dim_ckpt)} but Event-Clock period_feat_dim={int(period_dim_init)}; "
                    "initializing model with Event-Clock-compatible period_dim then attaching encoder bundle before loading weights."
                )

        # ---- Infer contact phase state (prev_phase_vec) ----
        phase_state_enable = False
        phase_state_hidden = 64
        try:
            phase_state_enable = any(
                k == "contact_phase_state_init"
                or k.startswith("contact_phase_state_delta_head.")
                for k in self.state_dict.keys()
            )
            w_h = self.state_dict.get("contact_phase_state_delta_head.1.weight", None)
            if torch.is_tensor(w_h) and w_h.ndim == 2 and int(w_h.shape[0]) > 0:
                phase_state_hidden = int(w_h.shape[0])
            w_out = self.state_dict.get("contact_phase_state_delta_head.3.weight", None)
            if torch.is_tensor(w_out) and w_out.ndim == 2 and int(w_out.shape[1]) > 0:
                phase_state_hidden = int(w_out.shape[1])
        except Exception:
            phase_state_enable = False
            phase_state_hidden = 64

        # Phase reset / clock anchor source:
        # - default ("contacts_meas"): phase resets are triggered by contacts_meas threshold crossing inside the model.
        # - TTC: we drive resets externally (run_freerun_cycles) and disable the internal threshold-crossing reset.
        phase_reset_source_cfg = str(getattr(self, "phase_reset_source", "contacts_meas") or "contacts_meas").strip().lower()
        if phase_reset_source_cfg in ("ttc", "ttcgt"):
            phase_reset_source_cfg = "ttc_gt"
        if phase_reset_source_cfg in ("hazard", "tdhazard", "td_hazard", "tdhaz"):
            raise SystemExit(
                "[FATAL] phase_reset_source=td_hazard has been retired from the current mainline. "
                "Use --phase_reset_source contacts_meas, ttc_gt, or none."
            )
        if phase_reset_source_cfg in ("none", "off", "disable", "disabled", "noreset", "no_reset"):
            phase_reset_source_cfg = "none"
        phase_reset_source_applied = str(phase_reset_source_cfg)
        try:
            strict = str(getattr(self, "phase_reset_source_strict", "off") or "off").strip().lower()
        except Exception:
            strict = "off"
        if strict in ("on", "1", "true", "yes") and phase_reset_source_applied != phase_reset_source_cfg:
            raise SystemExit(
                f"[FATAL] phase_reset_source={phase_reset_source_cfg} requested but applied={phase_reset_source_applied}. "
                "This would change acceptance semantics (likely a missing head in the checkpoint). "
                "Use a checkpoint with the required head or set --phase_reset_source_strict off."
            )
        try:
            setattr(self, "phase_reset_source_applied", str(phase_reset_source_applied))
        except Exception:
            pass

        phase_event_kind = "touchdown"
        if phase_reset_source_applied in ("ttc_gt", "none"):
            phase_event_kind = "none"
        try:
            phase_min_interval = int(getattr(self.args, "contact_phase_state_event_min_interval", 0) or 0)
        except Exception:
            phase_min_interval = 0
        if phase_event_kind == "none":
            phase_min_interval = 0

        # ---- Direct hinge extra config (feat source / contact gating) ----
        # These options are stored in posttrain checkpoints (posttrain_cfg) but historically weren't
        # reconstructed here, which can cause size-mismatch (hinge_feat_source) or behavior mismatch (gating).
        direct_pose_hinge_feat_source = None
        direct_pose_hinge_base_feat = "none"
        direct_pose_hinge_clean = False
        direct_pose_hinge_eps_max_deg = None
        direct_pose_hinge_eps_max_scale = 0.5
        direct_pose_hinge_eps_hidden = None
        direct_pose_hinge_eps_dropout = 0.0
        direct_pose_hinge_eps_source = "hidden"
        direct_pose_hinge_gate_mode = "none"
        direct_pose_hinge_gate_source = "plan"
        direct_pose_hinge_gate_power = 1.0
        direct_pose_hinge_hidden = int(self.direct_pose_hinge_hidden)

        if bool(getattr(self, "direct_pose_hinge_enable", False)):
            # Prefer checkpoint posttrain_cfg when present.
            if isinstance(self._ckpt_posttrain_cfg, dict):
                try:
                    v = self._ckpt_posttrain_cfg.get("direct_pose_hinge_feat_source", None)
                    if v is not None:
                        s = str(v).strip().lower()
                        direct_pose_hinge_feat_source = None if s in ("", "auto") else s
                except Exception:
                    pass
                try:
                    v = self._ckpt_posttrain_cfg.get("direct_pose_hinge_base_feat", None)
                    if v is not None:
                        s = str(v).strip().lower()
                        if s in ("rot6d", "rot_6d", "base_rot6d", "y_rot6d", "y_direct_rot6d"):
                            direct_pose_hinge_base_feat = "rot6d"
                        elif s in ("", "auto", "off", "disable", "disabled", "0", "none", "null"):
                            direct_pose_hinge_base_feat = "none"
                except Exception:
                    pass
                try:
                    direct_pose_hinge_clean = bool(self._ckpt_posttrain_cfg.get("direct_pose_hinge_clean", False))
                except Exception:
                    direct_pose_hinge_clean = False
                try:
                    v = float(self._ckpt_posttrain_cfg.get("direct_pose_hinge_eps_max_deg", 0.0) or 0.0)
                    direct_pose_hinge_eps_max_deg = None if (not math.isfinite(v) or v <= 0.0) else float(v)
                except Exception:
                    direct_pose_hinge_eps_max_deg = None
                # NOTE: allow 0.0 to explicitly disable eps(hidden) in clean hinge mode.
                try:
                    _raw = self._ckpt_posttrain_cfg.get("direct_pose_hinge_eps_max_scale", 0.5)
                    if _raw is None:
                        _raw = 0.5
                    v = float(_raw)
                    if (not math.isfinite(v)) or v < 0.0:
                        v = 0.5
                    direct_pose_hinge_eps_max_scale = float(v)
                except Exception:
                    direct_pose_hinge_eps_max_scale = 0.5
                try:
                    v = self._ckpt_posttrain_cfg.get("direct_pose_hinge_eps_hidden", None)
                    direct_pose_hinge_eps_hidden = int(v) if v is not None else None
                except Exception:
                    direct_pose_hinge_eps_hidden = None
                try:
                    v = float(self._ckpt_posttrain_cfg.get("direct_pose_hinge_eps_dropout", 0.0) or 0.0)
                    if (not math.isfinite(v)) or v < 0.0:
                        v = 0.0
                    direct_pose_hinge_eps_dropout = float(max(0.0, min(1.0, v)))
                except Exception:
                    direct_pose_hinge_eps_dropout = 0.0
                try:
                    v = self._ckpt_posttrain_cfg.get("direct_pose_hinge_eps_source", None)
                    if v is not None:
                        s = str(v).strip().lower()
                        if s in ("h_pre", "h_temporal", "pre", "temporal", "mid", "hidden_pre"):
                            direct_pose_hinge_eps_source = "hidden_pre"
                        elif s in ("h_final", "post", "final", "hidden"):
                            direct_pose_hinge_eps_source = "hidden"
                except Exception:
                    direct_pose_hinge_eps_source = "hidden"
                try:
                    direct_pose_hinge_gate_mode = str(
                        self._ckpt_posttrain_cfg.get("direct_pose_hinge_gate_mode", direct_pose_hinge_gate_mode)
                        or direct_pose_hinge_gate_mode
                    ).strip().lower()
                except Exception:
                    pass
                try:
                    direct_pose_hinge_gate_source = str(
                        self._ckpt_posttrain_cfg.get("direct_pose_hinge_gate_source", direct_pose_hinge_gate_source)
                        or direct_pose_hinge_gate_source
                    ).strip().lower()
                except Exception:
                    pass
                try:
                    direct_pose_hinge_gate_power = float(
                        self._ckpt_posttrain_cfg.get("direct_pose_hinge_gate_power", direct_pose_hinge_gate_power)
                        or direct_pose_hinge_gate_power
                    )
                except Exception:
                    pass

            # If not explicitly set, infer clean/legacy hinge mode from weights.
            if not bool(direct_pose_hinge_clean):
                try:
                    if (
                        ("direct_pose_hinge_nonhidden_head.0.weight" in self.state_dict)
                        or ("direct_pose_hinge_eps_head.1.weight" in self.state_dict)
                    ):
                        direct_pose_hinge_clean = True
                except Exception:
                    pass

            # Infer hinge hidden sizes from weights (needed when ckpt used default/None).
            if bool(direct_pose_hinge_clean):
                try:
                    w0 = self.state_dict.get("direct_pose_hinge_nonhidden_head.0.weight", None)
                    if torch.is_tensor(w0) and w0.ndim == 2 and int(w0.shape[0]) > 0:
                        direct_pose_hinge_hidden = int(w0.shape[0])
                except Exception:
                    pass
                if direct_pose_hinge_eps_hidden is None:
                    try:
                        w_eps = self.state_dict.get("direct_pose_hinge_eps_head.1.weight", None)
                        if torch.is_tensor(w_eps) and w_eps.ndim == 2 and int(w_eps.shape[0]) > 0:
                            direct_pose_hinge_eps_hidden = int(w_eps.shape[0])
                    except Exception:
                        pass
            else:
                try:
                    w0 = self.state_dict.get("direct_pose_hinge_head.0.weight", None)
                    if torch.is_tensor(w0) and w0.ndim == 2 and int(w0.shape[0]) > 0:
                        direct_pose_hinge_hidden = int(w0.shape[0])
                except Exception:
                    pass

            # If feat source isn't in checkpoint config, infer it from hinge input dim.
            if direct_pose_hinge_feat_source is None:
                try:
                    w0 = self.state_dict.get("direct_pose_hinge_head.0.weight", None)
                    if torch.is_tensor(w0) and w0.ndim == 2:
                        in_dim = int(w0.shape[1])
                        # If base_feat isn't in config, infer it from hinge_in_dim by checking whether
                        # subtracting 6*hinge_out yields a plausible feat_source base dim.
                        if str(direct_pose_hinge_base_feat).strip().lower() in ("", "auto"):
                            direct_pose_hinge_base_feat = "none"
                        inferred_base_feat = None
                        try:
                            w_out = self.state_dict.get("direct_pose_hinge_head.2.weight", None)
                            hinge_out = int(w_out.shape[0]) if torch.is_tensor(w_out) and w_out.ndim == 2 else 0
                        except Exception:
                            hinge_out = 0
                        want_meas = str(direct_pose_meas_mode).strip().lower() == "concat"
                        base_dim_raw = int(
                            in_dim
                            - int(self.contact_dim)
                            - (int(self.contact_dim) if want_meas else 0)
                            - int(direct_pose_time_pe_dim)
                        )
                        base_dim = int(base_dim_raw)
                        if inferred_base_feat is None and hinge_out > 0:
                            base_dim_rot6d = int(base_dim_raw - (6 * hinge_out))
                            if base_dim_rot6d in (0, int(Dc), int(self.width), int(Dc + self.width)):
                                inferred_base_feat = "rot6d"
                                base_dim = int(base_dim_rot6d)
                        if inferred_base_feat is not None:
                            direct_pose_hinge_base_feat = str(inferred_base_feat)
                        if base_dim == 0:
                            direct_pose_hinge_feat_source = "none"
                        elif base_dim == int(Dc):
                            direct_pose_hinge_feat_source = "cond"
                        elif base_dim == int(self.width):
                            direct_pose_hinge_feat_source = "hidden"
                        elif base_dim == int(Dc + self.width):
                            direct_pose_hinge_feat_source = "cond+hidden"
                except Exception:
                    pass

        # ---- Direct leg residual head config (optional; stored in posttrain_cfg) ----
        # Needed to reconstruct checkpoints that include direct_pose_leg_head / joint_idx buffer.
        direct_pose_leg_enable = False
        direct_pose_leg_bones = None
        direct_pose_leg_mode = "rot6d_add"
        direct_pose_leg_stopgrad_main = False
        direct_pose_leg_detach_feat = False
        direct_pose_leg_max_deg = 0.0
        direct_pose_leg_side_routing = False
        direct_pose_leg_contact_order = "lr"
        direct_pose_leg_side_embed_dim = 0
        direct_pose_leg_side_plan_other = False
        direct_pose_leg_side_phase_other = False
        direct_pose_leg_side_phase_rel = False
        direct_pose_leg_side_cue = "none"
        direct_pose_leg_side_cue_tau = 30.0
        direct_pose_leg_side_sign_gate = False
        direct_pose_leg_side_rank1 = False
        direct_pose_leg_gate_mode = "none"
        direct_pose_leg_gate_power = 1.0
        direct_pose_leg_scale_log_clip = 4.0
        direct_pose_leg_scale_clamp_k = 0.0
        try:
            if isinstance(self._ckpt_posttrain_cfg, dict):
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_enable", None)
                if v is not None:
                    direct_pose_leg_enable = bool(v)
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_bones", None)
                if v is not None:
                    direct_pose_leg_bones = str(v)
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_mode", None)
                if v is not None:
                    s = str(v).strip().lower()
                    direct_pose_leg_mode = "so3" if s in ("so3", "omega", "compose", "so3_compose") else "rot6d_add"
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_stopgrad_main", None)
                if v is not None:
                    direct_pose_leg_stopgrad_main = bool(v)
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_detach_feat", None)
                if v is not None:
                    direct_pose_leg_detach_feat = bool(v)
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_max_deg", None)
                if v is not None:
                    try:
                        direct_pose_leg_max_deg = float(v)
                    except Exception:
                        direct_pose_leg_max_deg = 0.0
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_gate_mode", None)
                if v is not None:
                    s = str(v).strip().lower()
                    if s in (
                        "signed_scale",
                        "signedscale",
                        "signed",
                        "signmag",
                        "sign_mag",
                        "signmagscale",
                        "signedmag",
                        "sscale",
                    ):
                        raise SystemExit(
                            "[FATAL] ckpt posttrain_cfg uses direct_pose_leg_gate_mode='signed_scale', "
                            "which is removed in current train/eval main chain. "
                            "Migrate to direct_pose_leg_gate_mode='scale' (or 'learned')."
                        )
                    if s in ("learned", "on", "true", "1", "yes", "y"):
                        direct_pose_leg_gate_mode = "learned"
                    elif s in ("scale", "mag", "magnitude", "logmag", "log_mag", "exp", "alpha"):
                        direct_pose_leg_gate_mode = "scale"
                    elif s in ("", "auto", "none", "off", "false", "0", "no", "n", "disable", "disabled"):
                        direct_pose_leg_gate_mode = "none"
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_gate_power", None)
                if v is not None:
                    try:
                        direct_pose_leg_gate_power = float(v)
                    except Exception:
                        direct_pose_leg_gate_power = 1.0
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_scale_log_clip", None)
                if v is not None:
                    try:
                        direct_pose_leg_scale_log_clip = float(v)
                    except Exception:
                        direct_pose_leg_scale_log_clip = 4.0
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_scale_clamp_k", None)
                if v is not None:
                    try:
                        direct_pose_leg_scale_clamp_k = float(v)
                    except Exception:
                        direct_pose_leg_scale_clamp_k = 0.0
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_side_routing", None)
                if v is not None:
                    direct_pose_leg_side_routing = bool(v)
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_contact_order", None)
                if v is not None:
                    direct_pose_leg_contact_order = str(v)
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_side_embed_dim", None)
                if v is not None:
                    try:
                        direct_pose_leg_side_embed_dim = int(v)
                    except Exception:
                        direct_pose_leg_side_embed_dim = 0
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_side_plan_other", None)
                if v is not None:
                    direct_pose_leg_side_plan_other = bool(v)
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_side_phase_other", None)
                if v is not None:
                    direct_pose_leg_side_phase_other = bool(v)
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_side_phase_rel", None)
                if v is not None:
                    direct_pose_leg_side_phase_rel = bool(v)
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_side_cue", None)
                if v is not None:
                    direct_pose_leg_side_cue = str(v)
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_side_cue_tau", None)
                if v is not None:
                    try:
                        direct_pose_leg_side_cue_tau = float(v)
                    except Exception:
                        direct_pose_leg_side_cue_tau = 30.0
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_side_sign_gate", None)
                if v is not None:
                    direct_pose_leg_side_sign_gate = bool(v)
                v = self._ckpt_posttrain_cfg.get("direct_pose_leg_side_rank1", None)
                if v is not None:
                    direct_pose_leg_side_rank1 = bool(v)
        except Exception:
            pass
        # If weights exist, ensure the head is instantiated even when cfg key wasn't present.
        try:
            if any(
                str(k).startswith("direct_pose_leg_head.")
                or str(k).startswith("direct_pose_leg_head_shared.")
                or str(k).startswith("direct_pose_leg_side_embed.")
                or str(k).startswith("direct_pose_leg_side_sign_gate_head.")
                for k in self.state_dict.keys()
            ):
                direct_pose_leg_enable = True
            if any(
                str(k).startswith("direct_pose_leg_gate_head.")
                or str(k).startswith("direct_pose_leg_gate_head_shared.")
                for k in self.state_dict.keys()
            ):
                # Gate/scale head only exists when leg SO(3) head is enabled.
                direct_pose_leg_enable = True
                if str(direct_pose_leg_gate_mode).strip().lower() in ("", "none", "off", "false", "0"):
                    # Backward-compatible default when ckpt lacks explicit mode.
                    direct_pose_leg_gate_mode = "learned"
            if any(str(k).startswith("direct_pose_leg_head_shared.") for k in self.state_dict.keys()):
                direct_pose_leg_side_routing = True
            if any(str(k).startswith("direct_pose_leg_gate_head_shared.") for k in self.state_dict.keys()):
                direct_pose_leg_side_routing = True
            if any(str(k).startswith("direct_pose_leg_side_sign_gate_head.") for k in self.state_dict.keys()):
                direct_pose_leg_side_routing = True
                direct_pose_leg_side_sign_gate = True
            if not bool(direct_pose_leg_side_rank1):
                # Heuristic auto-detect for older ckpts without posttrain_cfg:
                # rank1 head uses out_dim=(3 + K_side), which is typically not divisible by 3.
                w = self.state_dict.get("direct_pose_leg_head_shared.6.weight", None)
                if torch.is_tensor(w) and w.ndim == 2 and int(w.shape[0]) > 0 and (int(w.shape[0]) % 3) != 0:
                    direct_pose_leg_side_rank1 = True
        except Exception:
            pass

        model_kwargs = dict(
            in_state_dim=Dx,
            out_motion_dim=Dy,
            cond_dim=Dc,
            period_dim=int(period_dim_init),
            hidden_dim=self.width,
            num_layers=self.args.depth,
            num_heads=self.args.num_heads,
            dropout=self.args.dropout,
            context_len=self.args.context_len,
            contact_dim=self.contact_dim,
            angvel_dim=self.angvel_dim,
            pose_hist_dim=self.pose_hist_dim,
            state_layout=getattr(ds, "state_layout", None),
            bone_names=getattr(ds, "bone_names", None),
            output_layout=getattr(ds, "output_layout", None),
            contact_plan_enable=bool(contact_plan_enable or contact_plan_inject != "none" or direct_pose_enable),
            contact_plan_hidden=int(contact_plan_hidden),
            contact_plan_inject=str(contact_plan_inject),
            contact_plan_inject_detach=True,
            contact_plan_time_pe_dim=int(contact_plan_time_pe_dim),
            contact_plan_init_mode=str(contact_plan_init_mode),
            contact_plan_init_hidden=int(contact_plan_init_hidden),
            contact_plan_init_dropout=float(contact_plan_init_dropout),
            contact_phase_state_enable=bool(phase_state_enable),
            contact_phase_state_init_mode="obs",
            contact_phase_state_hidden=int(phase_state_hidden),
            contact_phase_state_delta_max=0.5,
            contact_phase_state_delta_init=(6.283185307179586 / 80.0),
            contact_phase_state_event_kind=str(phase_event_kind),
            contact_phase_state_event_thr=float(getattr(self.args, "contact_phase_state_event_thr", 0.5) or 0.5),
            contact_phase_state_event_hyst=float(getattr(self.args, "contact_phase_state_event_hyst", 0.0) or 0.0),
            contact_phase_state_event_min_interval=int(phase_min_interval),
            phase_reset_source=str(phase_reset_source_applied),
            use_event_clock=bool(use_event_clock),
            event_clock_max_delta=float(getattr(self.args, "event_clock_max_delta", 0.5) or 0.5),
            event_clock_hidden_dim=int(event_clock_hidden_dim),
            event_clock_gate_hidden_dim=int(event_clock_gate_hidden_dim),
            direct_pose_enable=bool(direct_pose_enable),
            direct_pose_hidden=int(direct_pose_hidden),
            direct_pose_dropout=0.0,
            direct_pose_detach_plan=True,
            direct_pose_meas_mode=str(direct_pose_meas_mode),
            direct_pose_meas_drop_prob=0.0,
            direct_pose_meas_noise_std=0.0,
            direct_pose_plan_drop_prob=0.0,
            direct_pose_feat_source=str(direct_pose_feat_source),
            direct_pose_time_pe_dim=int(direct_pose_time_pe_dim),
            direct_pose_use_phase_z=bool(direct_pose_use_phase_z),
            direct_pose_phase_z_mode=str(direct_pose_phase_z_mode),
            direct_pose_split_enable=bool(direct_pose_split_enable),
            direct_pose_nonleg_proj_dim=int(max(0, int(direct_pose_nonleg_proj_dim or 0))),
            direct_pose_arm_split_enable=bool(direct_pose_arm_split_enable),
            direct_pose_arm_bones=direct_pose_arm_bones,
            direct_pose_leg_enable=bool(direct_pose_leg_enable),
            direct_pose_leg_bones=direct_pose_leg_bones,
            direct_pose_leg_mode=str(direct_pose_leg_mode),
            direct_pose_leg_stopgrad_main=bool(direct_pose_leg_stopgrad_main),
            direct_pose_leg_detach_feat=bool(direct_pose_leg_detach_feat),
            direct_pose_leg_max_deg=float(direct_pose_leg_max_deg),
            direct_pose_leg_gate_mode=str(direct_pose_leg_gate_mode),
            direct_pose_leg_gate_power=float(direct_pose_leg_gate_power),
            direct_pose_leg_scale_log_clip=float(direct_pose_leg_scale_log_clip),
            direct_pose_leg_scale_clamp_k=float(direct_pose_leg_scale_clamp_k),
            direct_pose_leg_side_routing=bool(direct_pose_leg_side_routing),
            direct_pose_leg_contact_order=str(direct_pose_leg_contact_order),
            direct_pose_leg_side_embed_dim=int(direct_pose_leg_side_embed_dim),
            direct_pose_leg_side_plan_other=bool(direct_pose_leg_side_plan_other),
            direct_pose_leg_side_phase_other=bool(direct_pose_leg_side_phase_other),
            direct_pose_leg_side_phase_rel=bool(direct_pose_leg_side_phase_rel),
            direct_pose_leg_side_cue=str(direct_pose_leg_side_cue),
            direct_pose_leg_side_cue_tau=float(direct_pose_leg_side_cue_tau),
            direct_pose_leg_side_sign_gate=bool(direct_pose_leg_side_sign_gate),
            direct_pose_leg_side_rank1=bool(direct_pose_leg_side_rank1),
            direct_pose_hinge_enable=bool(self.direct_pose_hinge_enable),
            direct_pose_hinge_bones=str(self.direct_pose_hinge_bones),
            direct_pose_hinge_axis=str(self.direct_pose_hinge_axis),
            direct_pose_hinge_max_deg=float(self.direct_pose_hinge_max_deg),
            direct_pose_hinge_hidden=int(direct_pose_hinge_hidden),
            direct_pose_hinge_feat_source=direct_pose_hinge_feat_source,
            direct_pose_hinge_base_feat=str(direct_pose_hinge_base_feat),
            direct_pose_hinge_clean=bool(direct_pose_hinge_clean),
            direct_pose_hinge_eps_max_deg=direct_pose_hinge_eps_max_deg,
            direct_pose_hinge_eps_max_scale=float(direct_pose_hinge_eps_max_scale),
            direct_pose_hinge_eps_hidden=direct_pose_hinge_eps_hidden,
            direct_pose_hinge_eps_dropout=float(direct_pose_hinge_eps_dropout),
            direct_pose_hinge_eps_source=str(direct_pose_hinge_eps_source),
            direct_pose_hinge_gate_mode=str(direct_pose_hinge_gate_mode),
            direct_pose_hinge_gate_source=str(direct_pose_hinge_gate_source),
            direct_pose_hinge_gate_power=float(direct_pose_hinge_gate_power),
            lambda_fusion_enable=bool(lambda_fusion_enable),
            lambda_fusion_mode=str(lambda_fusion_mode),
            lambda_fusion_hidden=int(lambda_fusion_hidden),
            lambda_fusion_dropout=0.0,
            lambda_fusion_detach_err=True,
            lambda_fusion_logit_init=-2.0,
            lambda_fusion_use_rollout_step=bool(lambda_fusion_use_rollout_step),
        )
        try:
            model = EventMotionModel(**model_kwargs).to(self.device)
        except TypeError as exc:
            # Compat path: keep freerun robust when EventMotionModel signature
            # removes/renames options that still exist in older checkpoints/configs.
            if "unexpected keyword argument" not in str(exc):
                raise
            import inspect

            allowed = set(inspect.signature(EventMotionModel.__init__).parameters.keys())
            allowed.discard("self")
            dropped = sorted(k for k in model_kwargs.keys() if k not in allowed)
            if dropped:
                preview = ", ".join(dropped[:12])
                more = len(dropped) - min(len(dropped), 12)
                if more > 0:
                    preview = f"{preview}, ... (+{more})"
                print(
                    "[FreeRun][WARN] dropping unsupported EventMotionModel kwargs: "
                    f"{preview}"
                )
            filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed}
            model = EventMotionModel(**filtered_kwargs).to(self.device)
        # Validate basic shapes then load weights (allow extra frozen encoder keys).
        validate_and_fix_model_(model, Dx, Dc)
        # Attach frozen motion encoder BEFORE loading weights (period_dim/period_encoder may be created here).
        if self.encoder_bundle_path and self.encoder_bundle_path.is_file():
            model.attach_motion_encoder(torch.load(str(self.encoder_bundle_path), map_location="cpu"))
        # If the user overrides hinge bones vs the checkpoint, hinge head tensors may:
        #   (1) have incompatible shapes (K changes due to multi-bone hinge), and/or
        #   (2) become semantically wrong even if shapes match (same K but different bone order).
        # In that case, drop direct_pose_hinge_* tensors and rely on safe-zero init / oracle deltas.
        try:
            def _norm_bones(v: Any) -> List[str]:
                if v is None:
                    return []
                if isinstance(v, (list, tuple)):
                    items = [str(x).strip() for x in v]
                else:
                    items = [s.strip() for s in str(v).split(",") if s.strip()]
                return [x for x in items if x]

            ckpt_bones_raw = None
            if isinstance(getattr(self, "_ckpt_posttrain_cfg", None), dict):
                ckpt_bones_raw = self._ckpt_posttrain_cfg.get("direct_pose_hinge_bones", None)
            req_bones = _norm_bones(getattr(self, "direct_pose_hinge_bones", None))
            ckpt_bones = _norm_bones(ckpt_bones_raw)

            hinge_prefixes = (
                "direct_pose_hinge_head.",
                "direct_pose_hinge_nonhidden_head.",
                "direct_pose_hinge_eps_head.",
                "direct_pose_hinge_gate_head.",
                "direct_pose_hinge_gate_head_clean.",
            )

            removed: List[str] = []
            if bool(getattr(self, "direct_pose_hinge_enable", False)) and req_bones and ckpt_bones and (req_bones != ckpt_bones):
                for k in list(self.state_dict.keys()):
                    if any(str(k).startswith(p) for p in hinge_prefixes):
                        removed.append(str(k))
                        self.state_dict.pop(k, None)
                if removed:
                    print(
                        f"[FreeRun][INFO] direct_pose_hinge_bones override: ckpt={ckpt_bones} req={req_bones}; "
                        f"dropped {len(removed)} direct_pose_hinge_* tensors (will re-init hinge heads)."
                    )

            # Fallback: even when ckpt doesn't store bones metadata, guard against size mismatches.
            model_sd = model.state_dict()
            removed_shape: List[str] = []
            for k in list(self.state_dict.keys()):
                if not any(str(k).startswith(p) for p in hinge_prefixes):
                    continue
                v = self.state_dict.get(k, None)
                vv = model_sd.get(k, None)
                if torch.is_tensor(v) and torch.is_tensor(vv) and tuple(v.shape) != tuple(vv.shape):
                    removed_shape.append(str(k))
                    self.state_dict.pop(k, None)
            if removed_shape:
                print(
                    f"[FreeRun][INFO] dropped {len(removed_shape)} direct_pose_hinge_* tensors due to shape mismatch "
                    "(likely hinge_bones K changed)."
                )
        except Exception:
            pass
        missing, unexpected = model.load_state_dict(self.state_dict, strict=False)
        if missing or unexpected:
            print(f"[FreeRun][WARN] state_dict mismatch: missing={missing}, unexpected={unexpected}")
        # Common pitfall: the checkpoint contains hinge head weights but the model was instantiated with hinge disabled,
        # so those weights get dropped as "unexpected" and hinge silently becomes a no-op.
        try:
            if (
                (not bool(getattr(self, "direct_pose_hinge_enable", False)))
                and isinstance(unexpected, list)
                and any("direct_pose_hinge" in str(k) for k in unexpected)
            ):
                print(
                    "[FreeRun][WARN] ckpt contains direct_pose_hinge_* weights but hinge is disabled, so they will be ignored. "
                    "If you intended to use hinge, pass --direct_pose_hinge_enable (and matching hinge args). "
                    "If this is an ablation, you can ignore this warning."
                )
        except Exception:
            pass
        # Eval-time ablations for diagnosing "baseline direction capability".
        try:
            setattr(model, "direct_pose_leg_cross_leg_ablate", str(getattr(self, "direct_pose_leg_cross_leg_ablate", "none") or "none"))
        except Exception:
            pass
        try:
            setattr(
                model,
                "direct_pose_leg_side_plan_other_ablate",
                str(getattr(self, "direct_pose_leg_side_plan_other_ablate", "none") or "none"),
            )
        except Exception:
            pass
        try:
            setattr(model, "contact_plan_inject_scale", float(getattr(self, "contact_plan_inject_scale", 1.0)))
        except Exception:
            pass
        try:
            setattr(
                model,
                "contact_plan_time_bias_scale",
                float(getattr(self, "contact_plan_time_bias_scale", 1.0)),
            )
        except Exception:
            pass
        if bool(getattr(self, "log_contact_plan_logits_decomp", False)):
            try:
                setattr(model, "debug_contact_plan_logits_decomp", True)
            except Exception:
                pass
        model.eval()

        loss_fn = MotionJointLoss(
            output_layout=self.bundle.output_layout,
            fps=self.bundle.fps,
            rot6d_spec=self.bundle.rot6d_spec,
            meta=self.bundle.meta,
        )
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            lr=1e-4,
            grad_clip=0.0,
            weight_decay=0.0,
            tf_warmup_steps=0,
            tf_total_steps=0,
            augmentor=None,
            use_amp=False,
            accum_steps=1,
            pin_memory=False,
        )
        try:
            hinge_idx = getattr(model, "direct_pose_hinge_joint_idx", None)
            if hinge_idx:
                trainer.direct_pose_hinge_joint_idx = list(hinge_idx)
                trainer.direct_pose_hinge_axis = str(getattr(model, "direct_pose_hinge_axis", "Z") or "Z")
                trainer.direct_pose_hinge_max_rad = getattr(model, "direct_pose_hinge_max_rad", None)
                loss_fn.direct_pose_hinge_joint_idx = list(hinge_idx)
                loss_fn.direct_pose_hinge_axis = str(getattr(model, "direct_pose_hinge_axis", "Z") or "Z")
                loss_fn.direct_pose_hinge_max_rad = getattr(model, "direct_pose_hinge_max_rad", None)
        except Exception:
            pass
        trainer.so3_corr_apply = bool(self.so3_corr_apply)
        trainer.so3_corr_max_deg = float(self.so3_corr_max_deg)
        trainer.so3_corr_gate_force = self.so3_corr_gate_force
        trainer.so3_corr_gate_from_contacts_err = self.so3_corr_gate_from_contacts_err
        trainer.so3_corr_gate_from_contacts_err_mode = self.so3_corr_gate_from_contacts_err_mode
        trainer.so3_corr_gate_err_k = self.so3_corr_gate_err_k
        trainer.so3_corr_gate_err_bias = self.so3_corr_gate_err_bias
        trainer.so3_corr_gate_err_max = self.so3_corr_gate_err_max
        trainer.so3_corr_gate_err_ref_steps = self.so3_corr_gate_err_ref_steps
        trainer.so3_corr_gate_err_margin = self.so3_corr_gate_err_margin
        trainer.so3_corr_gate_err_use_ref = self.so3_corr_gate_err_use_ref
        trainer.so3_corr_gate_scale_max = self.so3_corr_gate_scale_max
        trainer.log_contacts = self.log_contacts
        trainer.log_contacts_whitebox = bool(getattr(self, "log_contacts_whitebox", False))
        trainer.log_contacts_whitebox_first_steps = int(getattr(self, "log_contacts_whitebox_first_steps", 0) or 0)
        trainer.contact_meas_gate_by_hit_override = getattr(self, "contact_meas_gate_by_hit_override", None)
        trainer.contact_meas_vxy_mode = str(getattr(self, "contact_meas_vxy_mode", "abs") or "abs")
        trainer.contact_meas_ground_z_mode = str(getattr(self, "contact_meas_ground_z_mode", "window") or "window")
        trainer.contact_meas_ground_z_beta = float(getattr(self, "contact_meas_ground_z_beta", 0.05) or 0.05)
        trainer.contact_meas_ground_z_window = int(getattr(self, "contact_meas_ground_z_window", 5) or 5)
        trainer.contact_meas_ground_z_quantile = float(getattr(self, "contact_meas_ground_z_quantile", 0.2) or 0.2)
        # Slew in meters per step (set 0 to disable). Applied after the chosen mode.
        try:
            up_cm = float(getattr(self, "contact_meas_ground_z_slew_up_cm", 0.0) or 0.0)
        except Exception:
            up_cm = 0.0
        try:
            down_cm = float(getattr(self, "contact_meas_ground_z_slew_down_cm", 0.0) or 0.0)
        except Exception:
            down_cm = 0.0
        trainer.contact_meas_ground_z_max_up_m = max(0.0, up_cm) / 100.0
        trainer.contact_meas_ground_z_max_down_m = max(0.0, down_cm) / 100.0
        trainer.lambda_fusion_apply = bool(self.lambda_fusion_apply)
        trainer.lambda_reliability_mode = str(getattr(self, "lambda_reliability_mode", "none") or "none")
        trainer.lambda_reliability_warmup_steps = int(getattr(self, "lambda_reliability_warmup_steps", 0) or 0)
        trainer.lambda_reliability_contact_err_max = float(getattr(self, "lambda_reliability_contact_err_max", 1.0) or 1.0)
        trainer.lambda_reliability_warmup_joint_scales = getattr(self, "lambda_reliability_warmup_joint_scales", None)
        trainer.direct_pose_meas_source = str(getattr(self, "direct_pose_meas_source", "model") or "model")
        trainer.direct_pose_meas_warmup_steps = int(getattr(self, "direct_pose_meas_warmup_steps", 0) or 0)
        trainer.direct_pose_plan_source = str(getattr(self, "direct_pose_plan_source", "model") or "model")
        trainer.direct_pose_softgt_stats = getattr(self, "direct_pose_softgt_stats", None)
        trainer.direct_pose_softgt_stats_spec = getattr(self, "direct_pose_softgt_stats_spec", None)
        trainer.contacts_meas_source = str(getattr(self, "contacts_meas_source", "model") or "model")
        trainer.contacts_meas_pretrain_clamp = float(getattr(self, "contacts_meas_pretrain_clamp", 1.0) or 0.0)
        trainer.contacts_meas_pretrain_affine_stats_spec = getattr(self, "contacts_meas_pretrain_affine_stats_spec", None)
        trainer.contacts_meas_pretrain_affine = getattr(self, "contacts_meas_pretrain_affine", None)
        trainer.contacts_meas_pretrain_anchor_ckpt_spec = getattr(self, "contacts_meas_pretrain_anchor_ckpt_spec", None)
        trainer.contacts_meas_pretrain_anchor = getattr(self, "contacts_meas_pretrain_anchor", None)
        trainer.contacts_meas_pretrain_anchor_config = getattr(self, "contacts_meas_pretrain_anchor_config", None)
        # Debug-only: learned contacts_meas post-process knobs (applied inside the free-run loop).
        try:
            trainer.contacts_meas_model_logit_scale = float(getattr(self, "contacts_meas_model_logit_scale", 1.0) or 1.0)
        except Exception:
            trainer.contacts_meas_model_logit_scale = 1.0
        trainer.contacts_meas_model_onehot = bool(getattr(self, "contacts_meas_model_onehot", False))
        trainer.contacts_meas_model_onehot_conditional = bool(getattr(self, "contacts_meas_model_onehot_conditional", False))
        try:
            thr = float(getattr(self, "contacts_meas_model_onehot_ds_thr", 0.5) or 0.5)
        except Exception:
            thr = 0.5
        if not np.isfinite(float(thr)):
            thr = 0.5
        trainer.contacts_meas_model_onehot_ds_thr = float(max(0.0, min(1.0, float(thr))))
        trainer.contacts_meas_gt_override_sics = str(getattr(self, "contacts_meas_gt_override_sics", "") or "").strip()
        try:
            trainer.contacts_meas_gt_override_cycle_gte = int(getattr(self, "contacts_meas_gt_override_cycle_gte", 1) or 1)
        except Exception:
            trainer.contacts_meas_gt_override_cycle_gte = 1
        trainer.contacts_meas_gt_override_cycle_gte = max(0, int(trainer.contacts_meas_gt_override_cycle_gte))
        trainer.contacts_meas_gt_override_drop_wrap = str(
            getattr(self, "contacts_meas_gt_override_drop_wrap", "on") or "on"
        ).strip().lower()
        trainer.phase_reset_source = str(
            getattr(self, "phase_reset_source_applied", None)
            or getattr(self, "phase_reset_source", "contacts_meas")
            or "contacts_meas"
        )
        trainer.ttc_event_kind = str(getattr(self, "ttc_event_kind", "touchdown") or "touchdown")
        trainer.ttc_max = getattr(self, "ttc_max", None)
        trainer.ttc_gt_event_shift = str(getattr(self, "ttc_gt_event_shift", "") or "").strip()
        trainer.ttc_apply_phase_reset_to_phase_z = str(getattr(self, "ttc_apply_phase_reset_to_phase_z", "on") or "on")
        # Inject bundle‑derived slices & normalizer
        self.bundle.apply_to_dataset(ds)
        self.bundle.apply_to_trainer(trainer)
        trainer._bundle_meta = dict(self.bundle.meta)
        try:
            trainer.fps = float(getattr(self.bundle, "fps", None) or getattr(ds, "fps", 60.0) or 60.0)
            trainer.bone_hz = float(trainer.fps)
        except Exception:
            pass
        trainer.pose_hist_len = int(getattr(ds, "pose_hist_len", 0) or 0)
        trainer.pose_hist_dim = int(getattr(ds, "pose_hist_dim", 0) or 0)
        pose_norm = getattr(ds, "pose_hist_norm", None)
        if pose_norm is not None:
            trainer.pose_hist_scales = torch.as_tensor(
                getattr(pose_norm, "scales", None), dtype=torch.float32
            )
            trainer.pose_hist_mu = (
                torch.as_tensor(pose_norm.mu, dtype=torch.float32) if getattr(pose_norm, "mu", None) is not None else None
            )
            trainer.pose_hist_std = (
                torch.as_tensor(pose_norm.std, dtype=torch.float32) if getattr(pose_norm, "std", None) is not None else None
            )
        else:
            trainer.pose_hist_scales = None
            trainer.pose_hist_mu = None
            trainer.pose_hist_std = None
        self.angvel_meta = {
            "mode": getattr(ds, "angvel_norm_mode", None),
            "mu": getattr(ds, "angvel_mu", None),
            "std": getattr(ds, "angvel_std", None),
        }
        trainer.normalizer = DataNormalizer(
            mu_x=self.bundle.mu_x,
            std_x=self.bundle.std_x,
            mu_y=self.bundle.mu_y,
            std_y=self.bundle.std_y,
            y_to_x_map=self.bundle.materialize_y_to_x_map(),
            yaw_x_slice=trainer.yaw_x_slice,
            yaw_y_slice=trainer.yaw_slice,
            rootvel_x_slice=trainer.rootvel_x_slice,
            rootvel_y_slice=trainer.rootvel_slice,
            angvel_x_slice=trainer.angvel_x_slice,
            angvel_y_slice=trainer.angvel_slice,
            tanh_scales_rootvel=self.bundle.tanh_scales_rootvel,
            tanh_scales_angvel=self.bundle.tanh_scales_angvel,
            angvel_mode=self.angvel_meta["mode"],
            angvel_mu=self.angvel_meta["mu"],
            angvel_std=self.angvel_meta["std"],
        )
        self.model = model
        self.loss_fn = loss_fn
        self.trainer = trainer
        self.normalizer = trainer.normalizer

    # ------------------------------------------------------------------ #
    #   Core per‑clip multi‑cycle free‑run logic
    # ------------------------------------------------------------------ #

    def _get_pose_hist_hybrid_donor_runner(self, ds: MotionEventDataset) -> Optional["FreeRunCycleRunner"]:
        if not bool(getattr(self.args, "pose_hist_hybrid_boundary_carry", False)):
            return None
        pose_hist_source_eff = str(getattr(self.args, "pose_hist_source", "buffer") or "buffer").strip().lower()
        pose_hist_update_eff = str(getattr(self.args, "pose_hist_update_source", "pred") or "pred").strip().lower()
        if pose_hist_source_eff not in ("", "buffer") or pose_hist_update_eff != "pred":
            return None

        donor_raw = str(getattr(self.args, "pose_hist_hybrid_donor_ckpt", "") or "").strip()
        if not donor_raw:
            raise ValueError(
                "pose_hist_hybrid_boundary_carry requires --pose_hist_hybrid_donor_ckpt when pose_hist_source=buffer and pose_hist_update_source=pred."
            )
        donor_path = Path(donor_raw).expanduser().resolve()
        if not donor_path.is_file():
            raise FileNotFoundError(f"pose_hist hybrid donor checkpoint not found: {donor_path}")

        donor_runner = self._pose_hist_hybrid_donor_runner
        if donor_runner is None or self._pose_hist_hybrid_donor_ckpt_path != donor_path:
            donor_args = argparse.Namespace(**vars(self.args))
            donor_args.model = str(donor_path)
            donor_args.pose_hist_hybrid_boundary_carry = False
            donor_args.pose_hist_hybrid_donor_ckpt = None
            donor_runner = FreeRunCycleRunner(donor_args)
            self._pose_hist_hybrid_donor_runner = donor_runner
            self._pose_hist_hybrid_donor_ckpt_path = donor_path

        donor_runner._ensure_model_ready(ds)
        return donor_runner

    def run_clip(self, teacher_path: Path, out_dir: Path, npz_root: Path, rounds: int) -> Optional[Path]:
        """
        Run N free‑run cycles on a single teacher clip and write per‑cycle JSON.
        """
        data = _load_json(teacher_path)
        clip_name = str(data.get("clip") or teacher_path.stem.replace("_teacher", ""))
        teacher_block = data.get("teacher")
        if not isinstance(teacher_block, dict):
            raise ValueError(f"{teacher_path}: missing 'teacher' payload.")

        # Use teacher JSON as reference for nominal cycle length
        state_arr = np.asarray(teacher_block.get("state_norm"), dtype=np.float32)
        cond_arr = np.asarray(teacher_block.get("cond"), dtype=np.float32)
        if state_arr.ndim != 2 or cond_arr.ndim != 2:
            raise ValueError(f"{teacher_path}: invalid state/cond shapes.")
        T_base = int(state_arr.shape[0])

        npz_path = _resolve_npz_path(clip_name, data.get("source_json"), npz_root)

        # Build dataset with full‑cycle seq_len so __getitem__ would cover one full window.
        ds = self._build_dataset(npz_path, seq_len=T_base)
        self._ensure_model_ready(ds)
        donor_runner = self._get_pose_hist_hybrid_donor_runner(ds)
        donor_trainer = donor_runner.trainer if donor_runner is not None else None
        clip = ds.clips[0]

        # Construct a single "full‑cycle" sample equivalent to MotionEventDataset.__getitem__ at s=0.
        base_sample = _build_full_cycle_sample(ds, clip, seq_len=T_base)

        # Run free‑run for N cycles without reset.
        metrics_per_round, per_step, extra = _run_freerun_cycles(
            trainer=self.trainer,
            sample=base_sample,
            rounds=rounds,
            device=self.device,
            donor_trainer=donor_trainer,
            time_index_mode=str(getattr(self.args, "time_index_mode", "auto") or "auto"),
            time_index_cycle_minus1=bool(getattr(self.args, "time_index_cycle_minus1", False)),
            lambda_fusion_apply=bool(self.lambda_fusion_apply),
            export_joint_geolocal=bool(getattr(self.args, "export_joint_geolocal", False)),
            export_joint_direct_geolocal_series=bool(
                getattr(self.args, "export_joint_direct_geolocal_series", False)
            ),
            export_joint_so3_error_series=bool(getattr(self.args, "export_joint_so3_error_series", False)),
            joint_so3_error_series_branches=str(
                getattr(self.args, "joint_so3_error_series_branches", "direct") or "direct"
            ),
            joint_so3_error_series_space=str(
                getattr(self.args, "joint_so3_error_series_space", "body") or "body"
            ),
            export_keybone_pos_err=bool(getattr(self.args, "export_keybone_pos_err", False)),
            export_keybone_omega=bool(getattr(self.args, "export_keybone_omega", False)),
            keybone_omega_deg_thresh=float(getattr(self.args, "keybone_omega_deg_thresh", 20.0) or 20.0),
            export_keybone_omega_series=bool(getattr(self.args, "export_keybone_omega_series", False)),
            keybone_omega_series_bones=str(
                getattr(self.args, "keybone_omega_series_bones", "calf_l,calf_r,lowerarm_l") or ""
            ),
            keybone_omega_series_axis=str(getattr(self.args, "keybone_omega_series_axis", "z") or "z"),
            export_plan_state_series=bool(getattr(self.args, "export_plan_state_series", False)),
            export_contact_meas_head_swap=bool(getattr(self.args, "export_contact_meas_head_swap", False)),
            export_direct_hinge_series=bool(getattr(self.args, "export_direct_hinge_series", False)),
            export_direct_leg_omega_series=bool(getattr(self.args, "export_direct_leg_omega_series", False)),
            export_direct_leg_head_io=bool(getattr(self.args, "export_direct_leg_head_io", False)),
            export_direct_nonleg_probe=bool(getattr(self.args, "export_direct_nonleg_probe", False)),
            direct_nonleg_probe_bones=str(
                getattr(
                    self.args,
                    "direct_nonleg_probe_bones",
                    "upperarm_l,lowerarm_l,hand_l,pinky_01_l",
                )
                or "upperarm_l,lowerarm_l,hand_l,pinky_01_l"
            ),
            direct_nonleg_probe_sics=str(getattr(self.args, "direct_nonleg_probe_sics", "") or ""),
            export_direct_arm_probe=bool(getattr(self.args, "export_direct_arm_probe", False)),
            direct_arm_probe_bones=str(
                getattr(
                    self.args,
                    "direct_arm_probe_bones",
                    "clavicle_l,clavicle_r,upperarm_l,upperarm_r,RUpArmTwist_l_01,RUpArmTwist_r_01,lowerarm_l,lowerarm_r,hand_l,hand_r,spine_01",
                )
                or "clavicle_l,clavicle_r,upperarm_l,upperarm_r,RUpArmTwist_l_01,RUpArmTwist_r_01,lowerarm_l,lowerarm_r,hand_l,hand_r,spine_01"
            ),
            direct_arm_probe_sics=str(getattr(self.args, "direct_arm_probe_sics", "") or ""),
            export_direct_leg_omega_alpha_sweep=bool(getattr(self.args, "export_direct_leg_omega_alpha_sweep", False)),
            direct_leg_omega_alpha_sweep_alphas=str(getattr(self.args, "direct_leg_omega_alpha_sweep_alphas", "0,0.25,0.5,1,-1") or ""),
            direct_leg_omega_alpha_sweep_steps=str(getattr(self.args, "direct_leg_omega_alpha_sweep_steps", "") or ""),
            direct_leg_omega_alpha_sweep_sics=str(getattr(self.args, "direct_leg_omega_alpha_sweep_sics", "") or ""),
            direct_leg_omega_alpha_sweep_sic_range=str(getattr(self.args, "direct_leg_omega_alpha_sweep_sic_range", "") or ""),
            direct_leg_omega_alpha_sweep_bones=str(getattr(self.args, "direct_leg_omega_alpha_sweep_bones", "leg") or "leg"),
            export_direct_leg_omega_grad=bool(getattr(self.args, "export_direct_leg_omega_grad", False)),
            direct_leg_omega_grad_sics=str(getattr(self.args, "direct_leg_omega_grad_sics", "12,14") or "12,14"),
            direct_leg_omega_grad_cycle_gte=int(getattr(self.args, "direct_leg_omega_grad_cycle_gte", 1) or 1),
            direct_leg_omega_grad_drop_wrap=str(getattr(self.args, "direct_leg_omega_grad_drop_wrap", "on") or "on"),
            direct_leg_omega_grad_bones=str(getattr(self.args, "direct_leg_omega_grad_bones", "leg") or "leg"),
            direct_pose_leg_noapply=bool(getattr(self.args, "direct_pose_leg_noapply", False)),
            direct_pose_leg_apply_scale=float(getattr(self.args, "direct_pose_leg_apply_scale", 1.0) or 1.0),
            direct_pose_leg_apply_sign=float(getattr(self.args, "direct_pose_leg_apply_sign", 1.0) or 1.0),
            direct_pose_leg_apply_side=str(getattr(self.args, "direct_pose_leg_apply_side", "left") or "left"),
            direct_pose_leg_contact_gate=bool(getattr(self.args, "direct_pose_leg_contact_gate", False)),
            direct_pose_leg_contact_gate_mode=str(getattr(self.args, "direct_pose_leg_contact_gate_mode", "delta") or "delta"),
            direct_pose_leg_contact_gate_order=str(getattr(self.args, "direct_pose_leg_contact_gate_order", "rl") or "rl"),
            direct_pose_leg_contact_gate_signal=str(getattr(self.args, "direct_pose_leg_contact_gate_signal", "logit") or "logit"),
            direct_pose_leg_contact_gate_k=float(getattr(self.args, "direct_pose_leg_contact_gate_k", 20.0) or 20.0),
            direct_pose_leg_contact_gate_min=float(getattr(self.args, "direct_pose_leg_contact_gate_min", 0.0) or 0.0),
            direct_pose_leg_contact_gate_phase_window_deg=float(getattr(self.args, "direct_pose_leg_contact_gate_phase_window_deg", 30.0) or 30.0),
            direct_pose_leg_contact_gate_joints=str(getattr(self.args, "direct_pose_leg_contact_gate_joints", "all") or "all"),
            direct_pose_leg_contact_flip=bool(getattr(self.args, "direct_pose_leg_contact_flip", False)),
            direct_pose_leg_contact_flip_order=str(getattr(self.args, "direct_pose_leg_contact_flip_order", "rl") or "rl"),
            direct_pose_leg_contact_flip_phase_window_deg=float(getattr(self.args, "direct_pose_leg_contact_flip_phase_window_deg", 30.0) or 30.0),
            direct_pose_leg_contact_flip_delta_thr=float(getattr(self.args, "direct_pose_leg_contact_flip_delta_thr", 0.0) or 0.0),
            direct_pose_leg_contact_flip_joints=str(getattr(self.args, "direct_pose_leg_contact_flip_joints", "foot_r,foot_l") or "foot_r,foot_l"),
            direct_pose_hinge_oracle_delta=bool(getattr(self.args, "direct_pose_hinge_oracle_delta", False)),
            export_keybone_state_series=bool(getattr(self.args, "export_keybone_state_series", False)),
            keybone_state_series_bones=str(
                getattr(self.args, "keybone_state_series_bones", "calf_l,calf_r,lowerarm_l") or ""
            ),
            keybone_state_series_branches=str(getattr(self.args, "keybone_state_series_branches", "inc,direct,blend") or ""),
            direct_align_inc0=bool(getattr(self.args, "direct_align_inc0", False)),
            multicycle_sync_state_on_cycle_start=bool(getattr(self.args, "multicycle_sync_state_on_cycle_start", False)),
            multicycle_reset_plan_z_on_cycle_start=bool(getattr(self.args, "multicycle_reset_plan_z_on_cycle_start", False)),
            multicycle_reset_pose_hist_on_cycle_start=bool(
                getattr(self.args, "multicycle_reset_pose_hist_on_cycle_start", False)
            ),
            freerun_x_gt_except_rot6d=bool(getattr(self.args, "freerun_x_gt_except_rot6d", False)),
            freerun_x_gt=bool(getattr(self.args, "freerun_x_gt", False)),
            pose_hist_hybrid_boundary_carry=bool(getattr(self.args, "pose_hist_hybrid_boundary_carry", False)),
            pose_hist_source=str(getattr(self.args, "pose_hist_source", "buffer") or "buffer"),
            pose_hist_update_source=str(getattr(self.args, "pose_hist_update_source", "pred") or "pred"),
            cond_reprojection=str(getattr(self.args, "cond_reprojection", "auto") or "auto"),
            analyze_phase_shift=bool(getattr(self.args, "analyze_phase_shift", False)),
            phase_shift_max=getattr(self.args, "phase_shift_max", None),
            debug_so3_corr=bool(getattr(self.args, "debug_so3_corr", False)),
            debug_rot_gain=bool(getattr(self.args, "debug_rot_gain", False)),
            rot_gain_joints=str(getattr(self.args, "rot_gain_joints", "calf_l") or "calf_l"),
            rot_gain_deg=float(getattr(self.args, "rot_gain_deg", 0.5) or 0.5),
            rot_gain_axis=str(getattr(self.args, "rot_gain_axis", "z") or "z"),
            debug_direct_alignment=bool(getattr(self.args, "debug_direct_alignment", False)),
            direct_alignment_max_shift=int(getattr(self.args, "direct_alignment_max_shift", 2) or 2),
            direct_alignment_joints=str(
                getattr(self.args, "direct_alignment_joints", "upperarm_l,lowerarm_l,hand_l,upperarm_r,lowerarm_r,hand_r")
                or ""
            ),
            direct_alignment_include_round0=bool(getattr(self.args, "direct_alignment_include_round0", False)),
        )
        if bool(getattr(self.args, "direct_align_inc0", False)):
            try:
                def _mean_key(seg, key: str):
                    vals = []
                    for rec in seg:
                        v = rec.get(key)
                        if v is None:
                            continue
                        try:
                            vals.append(float(v))
                        except Exception:
                            continue
                    if not vals:
                        return None
                    return float(sum(vals) / len(vals))

                def _fmt(v):
                    return f"{v:6.2f}" if v is not None else "  nan "

                # Print a compact summary to quickly judge if direct is mostly a phase/anchor offset.
                print("[Diag][AlignInc0] DirectGeoLocalDeg vs AlignInc0 (deg):")
                for r in metrics_per_round:
                    rr = r.get("round")
                    if rr is None:
                        continue
                    start = int(r.get("start_step", 0) or 0)
                    end = int(r.get("end_step", -1) or -1)
                    seg = []
                    for rec in per_step:
                        if not isinstance(rec, dict):
                            continue
                        step_val = rec.get("step", None)
                        if step_val is None:
                            continue
                        try:
                            step_i = int(step_val)
                        except Exception:
                            continue
                        if start <= step_i <= end:
                            seg.append(rec)
                    k = min(10, len(seg))
                    early = seg[:k] if k > 0 else []

                    inc_m = r.get("GeoLocalDeg")
                    d_m = r.get("DirectGeoLocalDeg")
                    da_m = r.get("DirectGeoLocalDegAlignInc0")
                    inc_e = _mean_key(early, "GeoLocalDeg")
                    d_e = _mean_key(early, "DirectGeoLocalDeg")
                    da_e = _mean_key(early, "DirectGeoLocalDegAlignInc0")
                    gain = None
                    try:
                        if d_m is not None and da_m is not None:
                            gain = float(d_m) - float(da_m)
                    except Exception:
                        gain = None

                    print(
                        f"  Round{int(rr)} mean  inc={_fmt(inc_m)} direct={_fmt(d_m)} align={_fmt(da_m)}  gain={_fmt(gain)}"
                    )
                    if k > 0:
                        print(
                            f"          first{k:02d} inc={_fmt(inc_e)} direct={_fmt(d_e)} align={_fmt(da_e)}"
                        )
            except Exception:
                pass

        payload = {
            "clip": clip_name,
            "source_json": data.get("source_json"),
            "teacher_json": str(teacher_path.resolve()),
            "bundle": str(Path(self.args.bundle).expanduser().resolve()) if getattr(self.args, "bundle", None) else None,
            "pretrain_template": str(Path(self.args.pretrain_template).expanduser().resolve())
            if getattr(self.args, "pretrain_template", None)
            else None,
            "encoder_bundle": str(Path(self.args.encoder_bundle).expanduser().resolve())
            if getattr(self.args, "encoder_bundle", None)
            else None,
            "fps": data.get("fps", getattr(ds, "fps", 60.0)),
            "cycle_len": int(T_base),
            "rounds": rounds,
            "time_index_mode": str(getattr(self.args, "time_index_mode", "auto") or "auto"),
            "time_index_cycle_minus1": bool(getattr(self.args, "time_index_cycle_minus1", False)),
            "time_index_cycle_len": int(T_base - 1) if bool(getattr(self.args, "time_index_cycle_minus1", False)) else int(T_base),
            "multicycle_sync_state_on_cycle_start": bool(getattr(self.args, "multicycle_sync_state_on_cycle_start", False)),
            "multicycle_reset_plan_z_on_cycle_start": bool(getattr(self.args, "multicycle_reset_plan_z_on_cycle_start", False)),
            "multicycle_reset_pose_hist_on_cycle_start": bool(
                getattr(self.args, "multicycle_reset_pose_hist_on_cycle_start", False)
            ),
            "freerun_x_gt_except_rot6d": bool(getattr(self.args, "freerun_x_gt_except_rot6d", False)),
            "freerun_x_gt": bool(getattr(self.args, "freerun_x_gt", False)),
            "pose_hist_hybrid_boundary_carry": bool(getattr(self.args, "pose_hist_hybrid_boundary_carry", False)),
            "pose_hist_hybrid_donor_ckpt": str(Path(getattr(self.args, "pose_hist_hybrid_donor_ckpt")).expanduser().resolve())
            if getattr(self.args, "pose_hist_hybrid_donor_ckpt", None)
            else None,
            "pose_hist_source": str(getattr(self.args, "pose_hist_source", "buffer") or "buffer"),
            "pose_hist_update_source": str(getattr(self.args, "pose_hist_update_source", "pred") or "pred"),
            "contact_plan_init_mode": str(getattr(self.model, "contact_plan_init_mode", None) or "unknown"),
            "contact_plan_init_hidden": int(getattr(self.model, "contact_plan_init_hidden", 0) or 0),
            "contact_plan_init_dropout": float(getattr(self.model, "_contact_plan_init_dropout", 0.0) or 0.0),
            "contact_plan_init_mode_override": getattr(self, "contact_plan_init_mode_override", None),
            "contact_plan_inject_scale": float(getattr(self, "contact_plan_inject_scale", 1.0)),
            "contact_plan_time_bias_scale": float(getattr(self, "contact_plan_time_bias_scale", 1.0)),
            "log_contact_plan_logits_decomp": bool(getattr(self, "log_contact_plan_logits_decomp", False)),
            "event_clock": str(getattr(self.args, "event_clock", "auto") or "auto"),
            "lambda_fusion_apply": bool(self.lambda_fusion_apply),
            "so3_corr_apply": bool(getattr(self.trainer, "so3_corr_apply", False)) if self.trainer is not None else False,
            "debug_so3_corr": bool(getattr(self.args, "debug_so3_corr", False)),
            "debug_rot_gain": bool(getattr(self.args, "debug_rot_gain", False)),
            "rot_gain_joints": str(getattr(self.args, "rot_gain_joints", "calf_l") or "calf_l"),
            "rot_gain_deg": float(getattr(self.args, "rot_gain_deg", 0.5) or 0.5),
            "rot_gain_axis": str(getattr(self.args, "rot_gain_axis", "z") or "z"),
            "direct_align_inc0": bool(getattr(self.args, "direct_align_inc0", False)),
            "direct_pose_meas_source": str(getattr(self, "direct_pose_meas_source", "model") or "model"),
            "direct_pose_meas_warmup_steps": int(getattr(self, "direct_pose_meas_warmup_steps", 0) or 0),
            "direct_pose_plan_source": str(getattr(self, "direct_pose_plan_source", "model") or "model"),
            "direct_pose_softgt_stats_spec": getattr(self, "direct_pose_softgt_stats_spec", None),
            "direct_pose_softgt_stats": getattr(self, "direct_pose_softgt_stats", None),
            "direct_pose_hinge_enable": bool(getattr(self, "direct_pose_hinge_enable", False)),
            "direct_pose_hinge_bones": str(getattr(self, "direct_pose_hinge_bones", "") or ""),
            "direct_pose_hinge_axis": str(getattr(self, "direct_pose_hinge_axis", "z") or "z"),
            "direct_pose_hinge_max_deg": float(getattr(self, "direct_pose_hinge_max_deg", 45.0) or 45.0),
            "direct_pose_hinge_hidden": int(getattr(self, "direct_pose_hinge_hidden", 0) or 0),
            "direct_pose_hinge_oracle_delta": bool(getattr(self, "direct_pose_hinge_oracle_delta", False)),
            "direct_pose_leg_noapply": bool(getattr(self.args, "direct_pose_leg_noapply", False)),
            "direct_pose_leg_cross_leg_ablate": str(getattr(self, "direct_pose_leg_cross_leg_ablate", "none") or "none"),
            "direct_pose_leg_side_plan_other_ablate": str(
                getattr(self, "direct_pose_leg_side_plan_other_ablate", "none") or "none"
            ),
            "phase_z_ablate": str(getattr(self.args, "phase_z_ablate", "none") or "none"),
            "contacts_meas_source": str(getattr(self, "contacts_meas_source", "model") or "model"),
            "contacts_meas_pretrain_clamp": float(getattr(self, "contacts_meas_pretrain_clamp", 1.0) or 0.0),
            "contacts_meas_pretrain_affine_stats_spec": getattr(self, "contacts_meas_pretrain_affine_stats_spec", None),
            "contacts_meas_pretrain_affine": getattr(self, "contacts_meas_pretrain_affine", None),
            "contacts_meas_pretrain_anchor_ckpt_spec": getattr(self, "contacts_meas_pretrain_anchor_ckpt_spec", None),
            "contacts_meas_pretrain_anchor_config": getattr(self, "contacts_meas_pretrain_anchor_config", None),
            "contacts_meas_model_logit_scale": float(getattr(self, "contacts_meas_model_logit_scale", 1.0) or 1.0),
            "contacts_meas_model_onehot": bool(getattr(self, "contacts_meas_model_onehot", False)),
            "contacts_meas_model_onehot_conditional": bool(getattr(self, "contacts_meas_model_onehot_conditional", False)),
            "contacts_meas_model_onehot_ds_thr": float(getattr(self, "contacts_meas_model_onehot_ds_thr", 0.5) or 0.5),
            "contacts_meas_gt_override_sics": str(getattr(self, "contacts_meas_gt_override_sics", "") or "").strip(),
            "contacts_meas_gt_override_cycle_gte": int(getattr(self, "contacts_meas_gt_override_cycle_gte", 1) or 1),
            "contacts_meas_gt_override_drop_wrap": str(
                getattr(self, "contacts_meas_gt_override_drop_wrap", "on") or "on"
            ).strip().lower(),
            "phase_reset_source": str(getattr(self, "phase_reset_source", "contacts_meas") or "contacts_meas"),
            "phase_reset_source_strict": str(getattr(self, "phase_reset_source_strict", "off") or "off"),
            "phase_reset_source_applied": str(
                getattr(self, "phase_reset_source_applied", None)
                or getattr(self, "phase_reset_source", "contacts_meas")
                or "contacts_meas"
            ),
            "ttc_event_kind": str(getattr(self, "ttc_event_kind", "touchdown") or "touchdown"),
            "ttc_max": int(getattr(self, "ttc_max", 0)) if getattr(self, "ttc_max", None) is not None else None,
            "ttc_gt_event_shift": str(getattr(self, "ttc_gt_event_shift", "") or "").strip(),
            "lambda_reliability_mode": str(getattr(self, "lambda_reliability_mode", "none") or "none"),
            "lambda_reliability_warmup_steps": int(getattr(self, "lambda_reliability_warmup_steps", 0) or 0),
            "lambda_reliability_contact_err_max": float(getattr(self, "lambda_reliability_contact_err_max", 1.0) or 1.0),
            "lambda_reliability_warmup_joint_scales": getattr(self, "lambda_reliability_warmup_joint_scales", None),
            "contact_meas_gate_by_hit": getattr(self, "contact_meas_gate_by_hit_override", None),
            "contact_meas_vxy_mode": str(getattr(self, "contact_meas_vxy_mode", "abs") or "abs"),
            "contact_meas_ground_z_mode": str(getattr(self, "contact_meas_ground_z_mode", "window") or "window"),
            "contact_meas_ground_z_beta": float(getattr(self, "contact_meas_ground_z_beta", 0.05) or 0.05),
            "contact_meas_ground_z_window": int(getattr(self, "contact_meas_ground_z_window", 5) or 5),
            "contact_meas_ground_z_quantile": float(getattr(self, "contact_meas_ground_z_quantile", 0.2) or 0.2),
            "contact_meas_ground_z_slew_up_cm": float(getattr(self, "contact_meas_ground_z_slew_up_cm", 0.0) or 0.0),
            "contact_meas_ground_z_slew_down_cm": float(getattr(self, "contact_meas_ground_z_slew_down_cm", 0.0) or 0.0),
            "model": str(Path(self.args.model).expanduser().resolve()),
            "metrics_per_round": metrics_per_round,
            "metrics_per_step": per_step,
        }
        if isinstance(extra, dict) and extra:
            payload.update(extra)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{clip_name}_freerun_cycles.json"
        if out_path.exists() and not self.args.force:
            raise FileExistsError(f"{out_path} exists (use --force to overwrite)")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] {clip_name}: wrote {out_path}")
        return out_path


def _build_full_cycle_sample(ds: MotionEventDataset, clip, seq_len: int) -> Dict[str, torch.Tensor]:
    """
    Construct a single sample with length = seq_len, equivalent to
    MotionEventDataset.__getitem__ at s=0, but without random yaw augmentation.
    """
    import numpy as np  # local alias

    T = int(seq_len)
    s = 0
    e = s + T

    Xv = clip.X[s:e]
    Yv = clip.Y[s:e]
    C_full = clip.C
    C_in_win = C_full[s:e]
    C_tgt_win = ds._window_with_edge_pad(C_full, s + 1, T)

    X = Xv.copy()
    Y = Yv.copy()
    C_in = C_in_win.copy()
    C_tgt = C_tgt_win.copy()
    C_tgt_raw = C_tgt.copy()

    cond_norm_mu = None
    cond_norm_std = None
    if ds.normalize_c and C_in.shape[1] > 0:
        mu, std = ds._robust_mean_std(C_in)
        try:
            std = np.clip(np.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6), 1e-6, None)
            mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            mu = np.nan_to_num(ds.C_mu, nan=0.0, posinf=0.0, neginf=0.0) if (ds.C_mu is not None) else 0.0
            std = np.nan_to_num(ds.C_std, nan=1e-6, posinf=1e-6, neginf=1e-6) if (ds.C_std is not None) else 1e-6
            std = np.clip(std, 1e-6, None)
        cond_norm_mu = mu.astype(np.float32, copy=False).reshape(-1)
        cond_norm_std = std.astype(np.float32, copy=False).reshape(-1)
        C_in = (C_in - mu) / std
        C_tgt = (C_tgt - mu) / std
        np.nan_to_num(C_in, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(C_tgt, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(C_in, -6.0, 6.0, out=C_in)
        np.clip(C_tgt, -6.0, 6.0, out=C_tgt)

    sample: Dict[str, torch.Tensor] = {
        "motion": torch.from_numpy(X).float(),
        "gt_motion": torch.from_numpy(Y).float(),
        "clip_id": torch.tensor(0, dtype=torch.int64),
        "start": torch.tensor(int(s), dtype=torch.int64),
    }
    if C_in.shape[1] > 0:
        sample["cond_in"] = torch.from_numpy(C_in.astype(np.float32, copy=False)).float()
        sample["cond_tgt"] = torch.from_numpy(C_tgt.astype(np.float32, copy=False)).float()
        sample["cond_tgt_raw"] = torch.from_numpy(C_tgt_raw.astype(np.float32, copy=False)).float()
        if cond_norm_mu is not None and cond_norm_mu.size == C_in.shape[1]:
            sample["cond_norm_mu"] = torch.from_numpy(cond_norm_mu).float()
            sample["cond_norm_std"] = torch.from_numpy(cond_norm_std).float()

    if clip.contacts is not None:
        sample["contacts"] = torch.from_numpy(clip.contacts[s:e].astype(np.float32, copy=False)).float()
    else:
        sample["contacts"] = torch.zeros((seq_len, ds.contact_dim), dtype=torch.float32)

    if clip.angvel_norm is not None:
        sample["angvel"] = torch.from_numpy(clip.angvel_norm[s:e].astype(np.float32, copy=False)).float()
    else:
        sample["angvel"] = torch.zeros((seq_len, ds.angvel_dim), dtype=torch.float32)

    if getattr(clip, "angvel_raw", None) is not None:
        sample["angvel_raw"] = torch.from_numpy(clip.angvel_raw[s:e].astype(np.float32, copy=False)).float()

    if clip.pose_hist_norm is not None:
        sample["pose_hist"] = torch.from_numpy(clip.pose_hist_norm[s:e].astype(np.float32, copy=False)).float()
    else:
        sample["pose_hist"] = torch.zeros((seq_len, ds.pose_hist_dim), dtype=torch.float32)

    return sample


def _run_freerun_cycles(
    trainer: Trainer,
    sample: Dict[str, torch.Tensor],
    rounds: int,
    device: torch.device,
    *,
    donor_trainer: Optional[Trainer] = None,
    time_index_mode: str = "auto",
    time_index_cycle_minus1: bool = False,
    lambda_fusion_apply: bool = False,
    export_joint_geolocal: bool = False,
    export_joint_direct_geolocal_series: bool = False,
    export_joint_so3_error_series: bool = False,
    joint_so3_error_series_branches: str = "direct",
    joint_so3_error_series_space: str = "body",
    export_keybone_pos_err: bool = False,
    export_keybone_omega: bool = False,
    keybone_omega_deg_thresh: float = 20.0,
    export_keybone_omega_series: bool = False,
    keybone_omega_series_bones: str = "calf_l,calf_r,lowerarm_l",
    keybone_omega_series_axis: str = "z",
    export_plan_state_series: bool = False,
    export_contact_meas_head_swap: bool = False,
    export_direct_hinge_series: bool = False,
    export_direct_leg_omega_series: bool = False,
    export_direct_leg_head_io: bool = False,
    export_direct_nonleg_probe: bool = False,
    direct_nonleg_probe_bones: str = "upperarm_l,lowerarm_l,hand_l,pinky_01_l",
    direct_nonleg_probe_sics: str = "",
    export_direct_arm_probe: bool = False,
    direct_arm_probe_bones: str = "clavicle_l,clavicle_r,upperarm_l,upperarm_r,RUpArmTwist_l_01,RUpArmTwist_r_01,lowerarm_l,lowerarm_r,hand_l,hand_r,spine_01",
    direct_arm_probe_sics: str = "",
    export_direct_leg_omega_alpha_sweep: bool = False,
    direct_leg_omega_alpha_sweep_alphas: str = "0,0.25,0.5,1,-1",
    direct_leg_omega_alpha_sweep_steps: str = "",
    direct_leg_omega_alpha_sweep_sics: str = "",
    direct_leg_omega_alpha_sweep_sic_range: str = "",
    direct_leg_omega_alpha_sweep_bones: str = "leg",
    export_direct_leg_omega_grad: bool = False,
    direct_leg_omega_grad_sics: str = "12,14",
    direct_leg_omega_grad_cycle_gte: int = 1,
    direct_leg_omega_grad_drop_wrap: str = "on",
    direct_leg_omega_grad_bones: str = "leg",
    direct_pose_leg_noapply: bool = False,
    direct_pose_leg_apply_scale: float = 1.0,
    direct_pose_leg_apply_sign: float = 1.0,
    direct_pose_leg_apply_side: str = "left",
    direct_pose_leg_contact_gate: bool = False,
    direct_pose_leg_contact_gate_mode: str = "delta",
    direct_pose_leg_contact_gate_order: str = "rl",
    direct_pose_leg_contact_gate_signal: str = "logit",
    direct_pose_leg_contact_gate_k: float = 20.0,
    direct_pose_leg_contact_gate_min: float = 0.0,
    direct_pose_leg_contact_gate_phase_window_deg: float = 30.0,
    direct_pose_leg_contact_gate_joints: str = "all",
    direct_pose_leg_contact_flip: bool = False,
    direct_pose_leg_contact_flip_order: str = "rl",
    direct_pose_leg_contact_flip_phase_window_deg: float = 30.0,
    direct_pose_leg_contact_flip_delta_thr: float = 0.0,
    direct_pose_leg_contact_flip_joints: str = "foot_r,foot_l",
    direct_pose_hinge_oracle_delta: bool = False,
    export_keybone_state_series: bool = False,
    keybone_state_series_bones: str = "calf_l,calf_r,lowerarm_l",
    keybone_state_series_branches: str = "inc,direct,blend",
    direct_align_inc0: bool = False,
    multicycle_sync_state_on_cycle_start: bool = False,
    multicycle_reset_plan_z_on_cycle_start: bool = False,
    multicycle_reset_pose_hist_on_cycle_start: bool = False,
    freerun_x_gt_except_rot6d: bool = False,
    freerun_x_gt: bool = False,
    pose_hist_hybrid_boundary_carry: bool = False,
    pose_hist_source: str = "buffer",
    pose_hist_update_source: str = "pred",
    cond_reprojection: str = "auto",
    analyze_phase_shift: bool = False,
    phase_shift_max: Optional[int] = None,
    debug_so3_corr: bool = False,
    debug_rot_gain: bool = False,
    rot_gain_joints: str = "calf_l",
    rot_gain_deg: float = 0.5,
    rot_gain_axis: str = "z",
    debug_direct_alignment: bool = False,
    direct_alignment_max_shift: int = 2,
    direct_alignment_joints: str = "upperarm_l,lowerarm_l,hand_l,upperarm_r,lowerarm_r,hand_r",
    direct_alignment_include_round0: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Core free‑run loop: autoregress over `rounds * T` steps without reset,
    then compute per‑round diagnostics.

    Note:
        GeoDeg (global/world) will be affected by root drift and thus can
        "couple" root deviation into all joint errors if rotations are global.
        We additionally report:
            - RootGeoDeg: root joint geodesic error (deg)
            - GeoLocalDeg: pose geodesic error excluding root joint (deg), decoupled from motion (unweighted mean)
            - GeoLocalDegWeighted: pose geodesic error excluding root joint (deg) with Trainer joint weights
            - RootPosErr / RootVelMAE (if X has RootPosition/RootVelocity)

        Training/online diagnostics may apply a constant "root0 alignment" (align predicted
        root at the first step to GT root, then measure drift). This script reports both:
            - GeoDeg / RootGeoDeg: raw (no alignment)
            - GeoDegAligned0 / RootGeoDegAligned0: constant aligned at step 0

        When direct_align_inc0=True, we additionally report a diagnostic that applies a
        per-joint constant bias computed at step0:
            R_bias[j] = R_inc0[j] @ R_dir0[j]^T
            R_dir_align_inc0[t,j] = R_bias[j] @ R_dir[t,j]
        This helps verify whether the direct head's early errors are mostly phase/anchor
        offsets (constant bias) versus genuinely worse dynamics.
    """
    import math  # keep local binding: this function contains nested `import math` blocks

    if rounds <= 0:
        raise ValueError("rounds must be > 0")

    # Move base sequences to device and tile along time.
    state_seq_base = sample["motion"].unsqueeze(0).to(device)  # [1, T, Dx]
    gt_seq_base = sample["gt_motion"].unsqueeze(0).to(device)  # [1, T, Dy]
    T_cycle = state_seq_base.shape[1]
    T_total = T_cycle * rounds
    # Post-train rollout uses steps=(T-1) for time_index in cycle mode (idx=t%steps),
    # while this script historically used cycle_len=T. Provide an ablation to align them.
    time_index_cycle_len = int(T_cycle)
    if bool(time_index_cycle_minus1) and int(T_cycle) > 1:
        time_index_cycle_len = max(1, int(T_cycle) - 1)

    def _maybe_to_device(key: str) -> Optional[torch.Tensor]:
        t = sample.get(key)
        if t is None:
            return None
        return t.to(device).unsqueeze(0)

    cond_seq = _maybe_to_device("cond_in")      # [1, T, Dc]
    cond_seq_raw = _maybe_to_device("cond_tgt_raw")
    contacts_seq = _maybe_to_device("contacts")
    angvel_seq = _maybe_to_device("angvel")
    pose_hist_seq = _maybe_to_device("pose_hist")

    cond_norm_mu = sample.get("cond_norm_mu")
    cond_norm_std = sample.get("cond_norm_std")
    if cond_norm_mu is not None:
        cond_norm_mu = cond_norm_mu.to(device)
    if cond_norm_std is not None:
        cond_norm_std = cond_norm_std.to(device)

    # Tile along time dimension to obtain a long run: [1, T_total, D]
    def _tile_time(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return t.repeat(1, rounds, 1)

    state_seq = state_seq_base.repeat(1, rounds, 1)
    gt_seq = gt_seq_base.repeat(1, rounds, 1)
    cond_seq = _tile_time(cond_seq)
    cond_seq_raw = _tile_time(cond_seq_raw)
    contacts_seq = _tile_time(contacts_seq)
    angvel_seq = _tile_time(angvel_seq)
    pose_hist_seq = _tile_time(pose_hist_seq)

    if cond_norm_mu is not None:
        cond_norm_mu = trainer._prepare_cond_stat(cond_norm_mu, state_seq)
    if cond_norm_std is not None:
        cond_norm_std = trainer._prepare_cond_stat(cond_norm_std, state_seq)

    # Pose-history buffer (mirror Trainer._rollout_sequence behavior)
    pose_hist_len = int(getattr(trainer, "pose_hist_len", 0) or 0)
    pose_hist_dim = int(getattr(trainer, "pose_hist_dim", 0) or 0)
    pose_hist_stride = pose_hist_dim // pose_hist_len if pose_hist_len > 0 else 0
    pose_hist_enabled = pose_hist_len > 0 and pose_hist_dim > 0 and pose_hist_stride > 0
    pose_hist_source_raw = str(pose_hist_source or "buffer").strip().lower()
    if pose_hist_source_raw in ("seq", "gt", "teacher"):
        pose_hist_source = "seq"
    elif pose_hist_source_raw in ("zero", "zeros", "off", "none", "null"):
        pose_hist_source = "zero"
    else:
        pose_hist_source = "buffer"
    pose_hist_update_source_raw = str(pose_hist_update_source or "pred").strip().lower()
    if pose_hist_update_source_raw in ("gt", "teacher", "seq"):
        pose_hist_update_source = "gt"
    elif pose_hist_update_source_raw in ("zero", "zeros", "off", "none", "null"):
        pose_hist_update_source = "zero"
    elif pose_hist_update_source_raw in ("freeze", "hold", "stop"):
        pose_hist_update_source = "freeze"
    else:
        pose_hist_update_source = "pred"
    pose_hist_state = PoseHistState(
        enabled=False,
        length=pose_hist_len,
        dim=pose_hist_dim,
        stride=pose_hist_stride,
    )
    if pose_hist_enabled:
        pose_hist_state = _init_eval_pose_hist_state(
            trainer,
            ref_tensor=state_seq,
            pose_hist_seq=pose_hist_seq,
            step=0,
            device=device,
            dtype=state_seq.dtype,
        )
        pose_hist_enabled = pose_hist_state.enabled
        pose_hist_stride = int(pose_hist_state.stride)
    scales = pose_hist_state.scales
    mu = pose_hist_state.mu
    std = pose_hist_state.std

    B, T, Dx = state_seq.shape
    assert T == T_total, "Internal error: tiled length mismatch."
    if T < 2:
        raise ValueError("Sequence too short for free‑run (need at least 2 frames).")

    time_index_mode = str(time_index_mode or "auto").strip().lower()

    warmup = 0
    start_t = warmup
    end_t = T - 1  # last usable index for t+1

    model = trainer.model
    # ---- Seam closure diagnostics (teacher-only discontinuity) ---------------
    # Compare the first vs last frame within ONE cycle (t=0 vs t=T_cycle-1).
    seam_closure: Dict[str, Any] = {
        "cycle_len": int(T_cycle),
        "rounds": int(rounds),
    }
    try:
        def _pair_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
            diff = (a - b).detach()
            diff = torch.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=-1e6)
            return {
                "abs_mean": float(diff.abs().mean().item()),
                "rmse": float((diff.pow(2).mean().sqrt()).item()),
                "max_abs": float(diff.abs().max().item()) if diff.numel() > 0 else 0.0,
            }

        if int(T_cycle) >= 2:
            seam_closure["x_norm"] = _pair_stats(state_seq[:, 0], state_seq[:, int(T_cycle) - 1])
            if torch.is_tensor(cond_seq) and cond_seq.dim() == 3 and int(cond_seq.size(1)) >= int(T_cycle):
                seam_closure["cond_in_norm"] = _pair_stats(cond_seq[:, 0], cond_seq[:, int(T_cycle) - 1])
            if torch.is_tensor(cond_seq_raw) and cond_seq_raw.dim() == 3 and int(cond_seq_raw.size(1)) >= int(T_cycle):
                seam_closure["cond_tgt_raw"] = _pair_stats(cond_seq_raw[:, 0], cond_seq_raw[:, int(T_cycle) - 1])
            if torch.is_tensor(contacts_seq) and contacts_seq.dim() == 3 and int(contacts_seq.size(1)) >= int(T_cycle):
                seam_closure["contacts"] = _pair_stats(contacts_seq[:, 0], contacts_seq[:, int(T_cycle) - 1])
            if torch.is_tensor(angvel_seq) and angvel_seq.dim() == 3 and int(angvel_seq.size(1)) >= int(T_cycle):
                seam_closure["angvel_norm"] = _pair_stats(angvel_seq[:, 0], angvel_seq[:, int(T_cycle) - 1])
            if torch.is_tensor(pose_hist_seq) and pose_hist_seq.dim() == 3 and int(pose_hist_seq.size(1)) >= int(T_cycle):
                seam_closure["pose_hist_norm"] = _pair_stats(pose_hist_seq[:, 0], pose_hist_seq[:, int(T_cycle) - 1])

            # X-space physical discontinuity (requires bundle stats + inverse-tanh).
            if getattr(trainer, "normalizer", None) is not None:
                try:
                    x0_raw = trainer.normalizer.denorm_x(state_seq[:, 0])
                    x1_raw = trainer.normalizer.denorm_x(state_seq[:, int(T_cycle) - 1])
                    rootpos_sl = getattr(trainer, "rootpos_x_slice", None)
                    rootvel_sl = getattr(trainer, "rootvel_x_slice", None)
                    angvel_sl = getattr(trainer, "angvel_x_slice", None)
                    if isinstance(rootpos_sl, slice):
                        seam_closure["root_pos_raw"] = _pair_stats(x0_raw[..., rootpos_sl], x1_raw[..., rootpos_sl])
                    if isinstance(rootvel_sl, slice):
                        seam_closure["root_vel_raw"] = _pair_stats(x0_raw[..., rootvel_sl], x1_raw[..., rootvel_sl])
                    if isinstance(angvel_sl, slice):
                        seam_closure["angvel_raw"] = _pair_stats(x0_raw[..., angvel_sl], x1_raw[..., angvel_sl])
                except Exception:
                    pass
    except Exception:
        seam_closure = {"cycle_len": int(T_cycle), "rounds": int(rounds)}

    # Cond reprojection mode (debug/ablation): match Trainer._rollout_sequence behavior if requested.
    cond_reprojection = str(cond_reprojection or "on").strip().lower()
    if cond_reprojection not in ("on", "off", "auto"):
        cond_reprojection = "on"

    predsY: List[torch.Tensor] = []  # incremental (Δ) absolute pose (y_norm), not necessarily used for update
    predsY_blend: List[torch.Tensor] = []  # blended absolute pose (y_norm)
    predsY_direct: List[torch.Tensor] = []
    predsX: List[torch.Tensor] = []
    contacts_log: List[Optional[Dict[str, Any]]] = []
    time_index_log: List[Optional[int]] = []
    lambda_log: List[Optional[torch.Tensor]] = []  # (B,J) on CPU
    lambda_eff_log: List[Optional[torch.Tensor]] = []  # (B,J) on CPU (after r_t)
    lambda_rel_log: List[Optional[torch.Tensor]] = []  # (B,) on CPU (r_t)
    event_clock_lambda_corr_log: List[Optional[torch.Tensor]] = []  # (B,1) on CPU
    so3_debug_log: List[Optional[Dict[str, Any]]] = []
    rot_gain_debug_log: List[Optional[Dict[str, Any]]] = []
    plan_z_in_log: List[Optional[List[float]]] = []
    phase_z_in_log: List[Optional[List[float]]] = []
    phase_event_age_in_log: List[Optional[List[float]]] = []
    direct_hinge_step_log: List[Optional[List[float]]] = []
    direct_hinge_raw_step_log: List[Optional[List[float]]] = []
    direct_hinge_base_raw_step_log: List[Optional[List[float]]] = []
    direct_hinge_eps_raw_step_log: List[Optional[List[float]]] = []
    direct_hinge_gate_step_log: List[Optional[List[float]]] = []
    # direct_leg_omega: (B,K,3) axis-angle residual in rad (already tanh-bounded by max_deg in the model).
    # We store mean-over-batch per step for offline phase-locked spike inspection.
    direct_leg_omega_step_log: List[Optional[List[List[float]]]] = []
    # Optional leg gate/scale diagnostics from direct branch (per-step mean over batch).
    direct_leg_scale_step_log: List[Optional[List[float]]] = []
    direct_leg_scale_log_step_log: List[Optional[List[float]]] = []
    direct_leg_scale_log_raw_step_log: List[Optional[List[float]]] = []
    # Debug-only: keep pre-leg-apply direct output and raw omega tensors (per step) so we can run
    # alpha-sweeps/oracle-alignment on the *same* rollout stream without re-running the model.
    direct_pre_leg_norm_step_log: List[Optional[torch.Tensor]] = []
    direct_leg_omega_tensor_step_log: List[Optional[torch.Tensor]] = []

    # Debug-only: export direct leg head first-layer IO (input vector + first Linear pre-activation).
    # We intentionally reuse the alpha-sweep step selector (direct_leg_omega_alpha_sweep_{steps,sics,sic_range})
    # so the payload stays small and aligned to the oracle-alignment analysis.
    direct_leg_head_io_enabled = bool(export_direct_leg_head_io)
    direct_leg_head_io: Dict[int, Dict[str, Any]] = {}
    _leg_head_io_cur_t: int = -1
    _leg_head_io_side_call: int = 0
    _leg_head_io_handles: List[Any] = []
    if direct_leg_head_io_enabled:
        # Parse selector specs (same semantics as the alpha-sweep exporter).
        step_sel: Set[int] = set()
        try:
            for tok in str(direct_leg_omega_alpha_sweep_steps or "").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if tok.lstrip("-").isdigit():
                    step_sel.add(int(tok))
        except Exception:
            step_sel = set()

        sic_sel: Set[int] = set()
        try:
            for tok in str(direct_leg_omega_alpha_sweep_sics or "").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if tok.lstrip("-").isdigit():
                    sic_sel.add(int(tok))
        except Exception:
            sic_sel = set()

        sic_lo = sic_hi = None
        try:
            sr = str(direct_leg_omega_alpha_sweep_sic_range or "").strip()
            if sr:
                sep = "-" if "-" in sr else ":"
                parts = [p.strip() for p in sr.split(sep) if p.strip()]
                if len(parts) == 2 and parts[0].lstrip("-").isdigit() and parts[1].lstrip("-").isdigit():
                    sic_lo = int(parts[0])
                    sic_hi = int(parts[1])
                    if sic_lo > sic_hi:
                        sic_lo, sic_hi = sic_hi, sic_lo
        except Exception:
            sic_lo = sic_hi = None

        def _want_io_t(tt: int) -> bool:
            if tt < 0:
                return False
            # Match alpha-sweep mask: cycle>=1 + drop_wrap.
            if int(T_cycle) > 0:
                cyc = int(tt // int(T_cycle))
                sic = int(tt % int(T_cycle))
                if cyc < 1:
                    return False
                if sic == int(T_cycle) - 1:
                    return False
            # Explicit step list.
            if step_sel:
                return int(tt) in step_sel
            # step_in_cycle selectors.
            if sic_sel or (sic_lo is not None and sic_hi is not None):
                if int(T_cycle) <= 0:
                    return False
                sic = int(tt % int(T_cycle))
                if sic_sel and sic in sic_sel:
                    return True
                if (sic_lo is not None and sic_hi is not None) and (sic_lo <= sic <= sic_hi):
                    return True
                return False
            # No selector => no export (avoid huge payloads).
            return False

        def _mean_vec(x: Any) -> Optional[List[float]]:
            if not torch.is_tensor(x):
                return None
            v = x.detach()
            if v.ndim == 2:
                v = v.mean(dim=0)
            elif v.ndim > 2:
                v = v.reshape(-1)
            try:
                return [float(t) for t in v.cpu().tolist()]
            except Exception:
                return None

        def _hook(kind: str):
            def _fn(_mod: Any, _inputs: Tuple[Any, ...], _output: Any) -> None:
                nonlocal _leg_head_io_side_call
                tt = int(_leg_head_io_cur_t)
                if not _want_io_t(tt):
                    return
                if not _inputs:
                    return
                x_in = _mean_vec(_inputs[0])
                y_out = _mean_vec(_output)
                if x_in is None or y_out is None:
                    return
                cyc = int(tt // int(T_cycle)) if int(T_cycle) > 0 else 0
                sic = int(tt % int(T_cycle)) if int(T_cycle) > 0 else tt
                ent = direct_leg_head_io.get(tt)
                if ent is None:
                    ent = {"step": tt, "cycle": cyc, "step_in_cycle": sic}
                    direct_leg_head_io[tt] = ent
                if kind == "baseline":
                    ent["baseline"] = {"in": x_in, "pre0": y_out}
                else:
                    side = "r" if _leg_head_io_side_call == 0 else "l"
                    _leg_head_io_side_call += 1
                    block = ent.get("shared")
                    if not isinstance(block, dict):
                        block = {}
                        ent["shared"] = block
                    block[side] = {"in": x_in, "pre0": y_out}

            return _fn

        # Register forward hooks on the first Linear layer of the leg head(s).
        try:
            import torch.nn as nn  # local import to avoid polluting module namespace

            def _first_linear(m: Any) -> Optional[nn.Linear]:
                if not isinstance(m, nn.Module):
                    return None
                for mm in m.modules():
                    if isinstance(mm, nn.Linear):
                        return mm
                return None

            fc0_base = _first_linear(getattr(model, "direct_pose_leg_head", None))
            fc0_shared = _first_linear(getattr(model, "direct_pose_leg_head_shared", None))
            if fc0_base is not None:
                _leg_head_io_handles.append(fc0_base.register_forward_hook(_hook("baseline")))
            if fc0_shared is not None:
                _leg_head_io_handles.append(fc0_shared.register_forward_hook(_hook("shared")))
        except Exception:
            _leg_head_io_handles = []
            direct_leg_head_io_enabled = False
            direct_leg_head_io = {}

    # Debug-only: direct non-leg feature probe export.
    # Goal: inspect whether non-leg branch features carry enough information for selected upper-limb bones.
    # We record:
    #   - pre_proj_in: input to direct_pose_nonleg_proj first Linear (i.e., shared hidden before non-leg projection)
    #   - proj_pre0: first Linear pre-activation of direct_pose_nonleg_proj
    #   - out_in: input to direct_pose_out_nonleg (post-proj feature actually used by non-leg readout)
    # plus per-step GT/direct rot6d targets for selected bones.
    direct_nonleg_probe_enabled = bool(export_direct_nonleg_probe)
    direct_nonleg_probe_steps: Dict[int, Dict[str, Any]] = {}
    direct_nonleg_probe_handles: List[Any] = []
    _direct_nonleg_probe_cur_t: int = -1
    _direct_nonleg_probe_active: bool = False
    direct_nonleg_probe_sic_sel: Set[int] = set()
    direct_nonleg_probe_bone_names_full: List[str] = []
    direct_nonleg_probe_bones_req: List[str] = []
    direct_nonleg_probe_all_nonleg: bool = False

    # Resolve selected bones and indices (J-space).
    direct_nonleg_probe_bone_names_sel: List[str] = []
    direct_nonleg_probe_joint_idx_sel: List[int] = []
    direct_nonleg_probe_leg_set: Set[str] = set()
    if direct_nonleg_probe_enabled:
        try:
            _bn = getattr(getattr(trainer, "loss_fn", None), "bone_names", None)
            if _bn is None:
                _bn = getattr(trainer, "_bone_names", None)
            if _bn is None:
                _meta = getattr(trainer, "bundle_meta", None)
                if isinstance(_meta, dict):
                    _bn = _meta.get("bone_names") or _meta.get("skeleton", {}).get("bone_names")
            direct_nonleg_probe_bone_names_full = [str(b) for b in _bn] if isinstance(_bn, (list, tuple)) else []
        except Exception:
            direct_nonleg_probe_bone_names_full = []
        try:
            for n in list(getattr(model, "direct_pose_leg_joint_names", None) or []):
                direct_nonleg_probe_leg_set.add(str(n))
        except Exception:
            direct_nonleg_probe_leg_set = set()
        try:
            req = [t.strip() for t in str(direct_nonleg_probe_bones or "").split(",") if t.strip()]
        except Exception:
            req = []
        if not req:
            req = ["upperarm_l", "lowerarm_l", "hand_l", "pinky_01_l"]
        direct_nonleg_probe_bones_req = [str(x) for x in req]
        direct_nonleg_probe_all_nonleg = bool(
            len(req) == 1 and req[0].strip().lower() in ("all_nonleg", "nonleg", "all")
        )
        if direct_nonleg_probe_all_nonleg:
            for j, nm in enumerate(direct_nonleg_probe_bone_names_full):
                nn = str(nm)
                if int(j) == int(root_idx):
                    continue
                if nn in direct_nonleg_probe_leg_set:
                    continue
                direct_nonleg_probe_bone_names_sel.append(nn)
                direct_nonleg_probe_joint_idx_sel.append(int(j))
        else:
            name_to_idx = {str(n): int(i) for i, n in enumerate(direct_nonleg_probe_bone_names_full)}
            for nm in req:
                if nm in name_to_idx:
                    j = int(name_to_idx[nm])
                    if int(j) == int(root_idx):
                        continue
                    direct_nonleg_probe_bone_names_sel.append(str(nm))
                    direct_nonleg_probe_joint_idx_sel.append(int(j))

        # Optional SIC selector.
        try:
            for tok in str(direct_nonleg_probe_sics or "").split(","):
                tok = tok.strip()
                if tok and tok.lstrip("-").isdigit():
                    direct_nonleg_probe_sic_sel.add(int(tok))
        except Exception:
            direct_nonleg_probe_sic_sel = set()

        def _want_nonleg_probe_t(tt: int) -> bool:
            if tt < 0:
                return False
            if int(T_cycle) > 0:
                cyc = int(tt // int(T_cycle))
                sic = int(tt % int(T_cycle))
                if cyc < 1:
                    return False
                if sic == int(T_cycle) - 1:
                    return False
                if direct_nonleg_probe_sic_sel and (sic not in direct_nonleg_probe_sic_sel):
                    return False
            return True

        def _nonleg_probe_mean_vec(x: Any) -> Optional[List[float]]:
            if not torch.is_tensor(x):
                return None
            v = x.detach()
            if v.ndim == 2:
                v = v.mean(dim=0)
            elif v.ndim > 2:
                v = v.reshape(-1)
            try:
                return [float(t) for t in v.cpu().tolist()]
            except Exception:
                return None

        def _nonleg_probe_ent(tt: int) -> Dict[str, Any]:
            cyc = int(tt // int(T_cycle)) if int(T_cycle) > 0 else 0
            sic = int(tt % int(T_cycle)) if int(T_cycle) > 0 else int(tt)
            ent = direct_nonleg_probe_steps.get(int(tt))
            if ent is None:
                ent = {
                    "step": int(tt),
                    "cycle": int(cyc),
                    "step_in_cycle": int(sic),
                    "features": {},
                    "targets": {},
                }
                direct_nonleg_probe_steps[int(tt)] = ent
            return ent

        def _hook_nonleg_proj(_mod: Any, _inputs: Tuple[Any, ...], _output: Any) -> None:
            if (not _direct_nonleg_probe_active) or (not direct_nonleg_probe_enabled):
                return
            tt = int(_direct_nonleg_probe_cur_t)
            if not _want_nonleg_probe_t(tt):
                return
            if not _inputs:
                return
            xin = _nonleg_probe_mean_vec(_inputs[0])
            ypre = _nonleg_probe_mean_vec(_output)
            ent = _nonleg_probe_ent(tt)
            if xin is not None:
                ent["features"]["pre_proj_in"] = xin
            if ypre is not None:
                ent["features"]["proj_pre0"] = ypre

        def _hook_nonleg_out(_mod: Any, _inputs: Tuple[Any, ...], _output: Any) -> None:
            if (not _direct_nonleg_probe_active) or (not direct_nonleg_probe_enabled):
                return
            tt = int(_direct_nonleg_probe_cur_t)
            if not _want_nonleg_probe_t(tt):
                return
            if not _inputs:
                return
            xin = _nonleg_probe_mean_vec(_inputs[0])
            if xin is None:
                return
            ent = _nonleg_probe_ent(tt)
            ent["features"]["out_in"] = xin

        try:
            import torch.nn as nn

            nonleg_proj = getattr(model, "direct_pose_nonleg_proj", None)
            if isinstance(nonleg_proj, nn.Sequential) and len(nonleg_proj) > 0 and isinstance(nonleg_proj[0], nn.Linear):
                direct_nonleg_probe_handles.append(nonleg_proj[0].register_forward_hook(_hook_nonleg_proj))
            out_nonleg = getattr(model, "direct_pose_out_nonleg", None)
            if isinstance(out_nonleg, nn.Linear):
                direct_nonleg_probe_handles.append(out_nonleg.register_forward_hook(_hook_nonleg_out))
        except Exception:
            direct_nonleg_probe_handles = []
            direct_nonleg_probe_enabled = False
            direct_nonleg_probe_steps = {}

    # Debug-only: direct arm-split feature probe export.
    # Goal: inspect arm branch representation drift under arm-split heads.
    # We record:
    #   - direct_in: input to direct_pose_head first Linear (flattened direct conditioning)
    #   - direct_phase: trailing phase_z slice inside direct_in (when present)
    #   - trunk_hidden: output of direct_pose_head shared trunk
    #   - proj_pre0: first Linear pre-activation of direct_pose_arm_proj
    #   - out_in: input to direct_pose_out_arm
    #   - arm_out: output of direct_pose_out_arm before scatter-back
    # plus per-step GT/direct rot6d targets for selected bones.
    direct_arm_probe_enabled = bool(export_direct_arm_probe)
    direct_arm_probe_steps: Dict[int, Dict[str, Any]] = {}
    direct_arm_probe_handles: List[Any] = []
    _direct_arm_probe_cur_t: int = -1
    _direct_arm_probe_active: bool = False
    direct_arm_probe_sic_sel: Set[int] = set()
    direct_arm_probe_bone_names_full: List[str] = []
    direct_arm_probe_bones_req: List[str] = []
    direct_arm_probe_all: bool = False
    direct_arm_probe_bone_names_sel: List[str] = []
    direct_arm_probe_joint_idx_sel: List[int] = []
    if direct_arm_probe_enabled:
        try:
            _bn = getattr(getattr(trainer, "loss_fn", None), "bone_names", None)
            if _bn is None:
                _bn = getattr(trainer, "_bone_names", None)
            if _bn is None:
                _meta = getattr(trainer, "bundle_meta", None)
                if isinstance(_meta, dict):
                    _bn = _meta.get("bone_names") or _meta.get("skeleton", {}).get("bone_names")
            direct_arm_probe_bone_names_full = [str(b) for b in _bn] if isinstance(_bn, (list, tuple)) else []
        except Exception:
            direct_arm_probe_bone_names_full = []
        try:
            req = [t.strip() for t in str(direct_arm_probe_bones or "").split(",") if t.strip()]
        except Exception:
            req = []
        if not req:
            req = [
                "clavicle_l",
                "clavicle_r",
                "upperarm_l",
                "upperarm_r",
                "RUpArmTwist_l_01",
                "RUpArmTwist_r_01",
                "lowerarm_l",
                "lowerarm_r",
                "hand_l",
                "hand_r",
                "spine_01",
            ]
        direct_arm_probe_bones_req = [str(x) for x in req]
        direct_arm_probe_all = bool(len(req) == 1 and req[0].strip().lower() in ("all", "all_arm", "arm"))
        if direct_arm_probe_all:
            for j, nm in enumerate(direct_arm_probe_bone_names_full):
                if int(j) == int(root_idx):
                    continue
                direct_arm_probe_bone_names_sel.append(str(nm))
                direct_arm_probe_joint_idx_sel.append(int(j))
        else:
            name_to_idx = {str(n): int(i) for i, n in enumerate(direct_arm_probe_bone_names_full)}
            for nm in req:
                if nm in name_to_idx:
                    j = int(name_to_idx[nm])
                    if int(j) == int(root_idx):
                        continue
                    direct_arm_probe_bone_names_sel.append(str(nm))
                    direct_arm_probe_joint_idx_sel.append(int(j))

        try:
            for tok in str(direct_arm_probe_sics or "").split(","):
                tok = tok.strip()
                if tok and tok.lstrip("-").isdigit():
                    direct_arm_probe_sic_sel.add(int(tok))
        except Exception:
            direct_arm_probe_sic_sel = set()

        def _want_arm_probe_t(tt: int) -> bool:
            if tt < 0:
                return False
            if int(T_cycle) > 0:
                cyc = int(tt // int(T_cycle))
                sic = int(tt % int(T_cycle))
                if cyc < 1:
                    return False
                if sic == int(T_cycle) - 1:
                    return False
                if direct_arm_probe_sic_sel and (sic not in direct_arm_probe_sic_sel):
                    return False
            return True

        def _arm_probe_mean_vec(x: Any) -> Optional[List[float]]:
            if not torch.is_tensor(x):
                return None
            v = x.detach()
            if v.ndim == 2:
                v = v.mean(dim=0)
            elif v.ndim > 2:
                v = v.reshape(-1)
            try:
                return [float(t) for t in v.cpu().tolist()]
            except Exception:
                return None

        def _arm_probe_ent(tt: int) -> Dict[str, Any]:
            cyc = int(tt // int(T_cycle)) if int(T_cycle) > 0 else 0
            sic = int(tt % int(T_cycle)) if int(T_cycle) > 0 else int(tt)
            ent = direct_arm_probe_steps.get(int(tt))
            if ent is None:
                ent = {
                    "step": int(tt),
                    "cycle": int(cyc),
                    "step_in_cycle": int(sic),
                    "features": {},
                    "targets": {},
                }
                direct_arm_probe_steps[int(tt)] = ent
            return ent

        def _hook_arm_trunk_in(_mod: Any, _inputs: Tuple[Any, ...], _output: Any) -> None:
            if (not _direct_arm_probe_active) or (not direct_arm_probe_enabled):
                return
            tt = int(_direct_arm_probe_cur_t)
            if not _want_arm_probe_t(tt):
                return
            if not _inputs:
                return
            xin_t = _inputs[0] if torch.is_tensor(_inputs[0]) else None
            xin = _arm_probe_mean_vec(xin_t)
            ent = _arm_probe_ent(tt)
            if xin is not None:
                ent["features"]["direct_in"] = xin
            phase_dim = int(getattr(model, "_direct_pose_phase_dim", 0) or 0)
            if torch.is_tensor(xin_t) and phase_dim > 0 and int(xin_t.shape[-1]) >= phase_dim:
                phase_v = _arm_probe_mean_vec(xin_t[..., -phase_dim:])
                if phase_v is not None:
                    ent["features"]["direct_phase"] = phase_v

        def _hook_arm_trunk_out(_mod: Any, _inputs: Tuple[Any, ...], _output: Any) -> None:
            if (not _direct_arm_probe_active) or (not direct_arm_probe_enabled):
                return
            tt = int(_direct_arm_probe_cur_t)
            if not _want_arm_probe_t(tt):
                return
            y = _arm_probe_mean_vec(_output)
            if y is None:
                return
            ent = _arm_probe_ent(tt)
            ent["features"]["trunk_hidden"] = y

        def _hook_arm_proj(_mod: Any, _inputs: Tuple[Any, ...], _output: Any) -> None:
            if (not _direct_arm_probe_active) or (not direct_arm_probe_enabled):
                return
            tt = int(_direct_arm_probe_cur_t)
            if not _want_arm_probe_t(tt):
                return
            ypre = _arm_probe_mean_vec(_output)
            if ypre is None:
                return
            ent = _arm_probe_ent(tt)
            ent["features"]["proj_pre0"] = ypre

        def _hook_arm_out(_mod: Any, _inputs: Tuple[Any, ...], _output: Any) -> None:
            if (not _direct_arm_probe_active) or (not direct_arm_probe_enabled):
                return
            tt = int(_direct_arm_probe_cur_t)
            if not _want_arm_probe_t(tt):
                return
            ent = _arm_probe_ent(tt)
            if _inputs:
                xin = _arm_probe_mean_vec(_inputs[0])
                if xin is not None:
                    ent["features"]["out_in"] = xin
            yout = _arm_probe_mean_vec(_output)
            if yout is not None:
                ent["features"]["arm_out"] = yout

        try:
            import torch.nn as nn

            trunk = getattr(model, "direct_pose_head", None)
            if isinstance(trunk, nn.Sequential) and len(trunk) > 0 and isinstance(trunk[0], nn.Linear):
                direct_arm_probe_handles.append(trunk[0].register_forward_hook(_hook_arm_trunk_in))
                direct_arm_probe_handles.append(trunk.register_forward_hook(_hook_arm_trunk_out))
            arm_proj = getattr(model, "direct_pose_arm_proj", None)
            if isinstance(arm_proj, nn.Sequential) and len(arm_proj) > 0 and isinstance(arm_proj[0], nn.Linear):
                direct_arm_probe_handles.append(arm_proj[0].register_forward_hook(_hook_arm_proj))
            out_arm = getattr(model, "direct_pose_out_arm", None)
            if isinstance(out_arm, nn.Linear):
                direct_arm_probe_handles.append(out_arm.register_forward_hook(_hook_arm_out))
            if not direct_arm_probe_handles:
                direct_arm_probe_enabled = False
        except Exception:
            direct_arm_probe_handles = []
            direct_arm_probe_enabled = False
            direct_arm_probe_steps = {}

    # Debug-only: contact-plan gating diagnostics for direct_leg_omega apply.
    # We gate per side (right/left) using the plan transition magnitude:
    #   delta_c = |contacts_plan[t] - contacts_plan[t-1]|
    #   g = gmin + (1-gmin) * exp(-k * delta_c)
    # Then apply per-joint scaling based on joint name suffix (_r / _l).
    direct_leg_omega_plan_gate_step_log: List[Optional[Dict[str, Any]]] = []
    direct_leg_omega_plan_prev: Optional[torch.Tensor] = None
    direct_leg_omega_gate_mask_r: Optional[torch.Tensor] = None  # (1,K,1)
    direct_leg_omega_gate_mask_l: Optional[torch.Tensor] = None  # (1,K,1)
    direct_leg_omega_gate_apply_mask: Optional[torch.Tensor] = None  # (1,K,1) where 1=apply gate, 0=force g=1
    gate_ch_r = 0
    gate_ch_l = 1
    gate_mode = str(direct_pose_leg_contact_gate_mode or "delta").strip().lower()
    if gate_mode not in ("delta", "phase"):
        gate_mode = "delta"
    gate_order = str(direct_pose_leg_contact_gate_order or "rl").strip().lower()
    gate_signal = str(direct_pose_leg_contact_gate_signal or "logit").strip().lower()
    if gate_signal not in ("prob", "logit"):
        gate_signal = "logit"
    if gate_order in ("lr", "l,r", "l r"):
        gate_ch_l, gate_ch_r = 0, 1
    else:
        gate_ch_r, gate_ch_l = 0, 1
    if bool(direct_pose_leg_contact_gate):
        try:
            leg_names_all = list(getattr(model, "direct_pose_leg_joint_names", None) or [])
        except Exception:
            leg_names_all = []
        if leg_names_all:
            try:
                names_l = [str(n).lower() for n in leg_names_all]
                m_r = [1.0 if n.endswith(("_r", "right")) else 0.0 for n in names_l]
                m_l = [1.0 if n.endswith(("_l", "left")) else 0.0 for n in names_l]
                spec = str(direct_pose_leg_contact_gate_joints or "all").strip().lower()
                if spec in ("", "all", "*"):
                    m_apply = [1.0 for _ in names_l]
                elif spec in ("distal", "feet", "foot", "toes", "toe"):
                    m_apply = [
                        1.0 if any(tok in n for tok in ("foot", "ball", "toe")) else 0.0 for n in names_l
                    ]
                else:
                    want = {t.strip().lower() for t in spec.replace(";", ",").split(",") if t.strip()}
                    m_apply = [1.0 if n in want else 0.0 for n in names_l]
                direct_leg_omega_gate_mask_r = torch.tensor(m_r, device=device, dtype=state_seq.dtype).view(1, -1, 1)
                direct_leg_omega_gate_mask_l = torch.tensor(m_l, device=device, dtype=state_seq.dtype).view(1, -1, 1)
                direct_leg_omega_gate_apply_mask = torch.tensor(m_apply, device=device, dtype=state_seq.dtype).view(1, -1, 1)
            except Exception:
                direct_leg_omega_gate_mask_r = None
                direct_leg_omega_gate_mask_l = None
                direct_leg_omega_gate_apply_mask = None

    # Debug-only: conditional sign flip for direct_leg_omega inside a phase window (per-side),
    # typically to test "best_alpha<0" only around contact transitions.
    direct_leg_omega_flip_step_log: List[Optional[Dict[str, Any]]] = []
    direct_leg_omega_flip_mask_r: Optional[torch.Tensor] = None  # (1,K,1)
    direct_leg_omega_flip_mask_l: Optional[torch.Tensor] = None  # (1,K,1)
    direct_leg_omega_flip_apply_mask: Optional[torch.Tensor] = None  # (1,K,1)
    direct_leg_omega_flip_plan_prev: Optional[torch.Tensor] = None  # (B,C) contacts_plan_logits
    flip_ch_r = 0
    flip_ch_l = 1
    flip_order = str(direct_pose_leg_contact_flip_order or "rl").strip().lower()
    if flip_order in ("lr", "l,r", "l r"):
        flip_ch_l, flip_ch_r = 0, 1
    else:
        flip_ch_r, flip_ch_l = 0, 1
    if bool(direct_pose_leg_contact_flip):
        try:
            leg_names_all = list(getattr(model, "direct_pose_leg_joint_names", None) or [])
        except Exception:
            leg_names_all = []
        if leg_names_all:
            try:
                names_l = [str(n).lower() for n in leg_names_all]
                m_r = [1.0 if n.endswith(("_r", "right")) else 0.0 for n in names_l]
                m_l = [1.0 if n.endswith(("_l", "left")) else 0.0 for n in names_l]
                spec = str(direct_pose_leg_contact_flip_joints or "foot_r,foot_l").strip().lower()
                if spec in ("", "all", "*"):
                    m_apply = [1.0 for _ in names_l]
                elif spec in ("distal", "feet", "foot", "toes", "toe"):
                    m_apply = [1.0 if any(tok in n for tok in ("foot", "ball", "toe")) else 0.0 for n in names_l]
                else:
                    want = {t.strip().lower() for t in spec.replace(";", ",").split(",") if t.strip()}
                    m_apply = [1.0 if n in want else 0.0 for n in names_l]
                direct_leg_omega_flip_mask_r = torch.tensor(m_r, device=device, dtype=state_seq.dtype).view(1, -1, 1)
                direct_leg_omega_flip_mask_l = torch.tensor(m_l, device=device, dtype=state_seq.dtype).view(1, -1, 1)
                direct_leg_omega_flip_apply_mask = torch.tensor(m_apply, device=device, dtype=state_seq.dtype).view(1, -1, 1)
            except Exception:
                direct_leg_omega_flip_mask_r = None
                direct_leg_omega_flip_mask_l = None
                direct_leg_omega_flip_apply_mask = None

    # Initialize motion & raw state
    motion = state_seq[:, start_t]  # [B, Dx]
    motion_raw = None
    if getattr(trainer, "normalizer", None) is not None:
        try:
            motion_raw = trainer.normalizer.denorm_x(motion)
        except Exception:
            motion_raw = None

    y_raw_prev = None
    try:
        y_raw_prev = trainer._denorm(gt_seq[:, start_t])
    except Exception:
        y_raw_prev = None
    if y_raw_prev is None and motion_raw is not None:
        rot6d_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
        if isinstance(rot6d_slice, slice):
            slice_len = rot6d_slice.stop - rot6d_slice.start
            if slice_len == gt_seq.shape[-1]:
                try:
                    y_raw_prev = motion_raw[:, rot6d_slice].clone()
                except Exception:
                    y_raw_prev = None

    # Rolling GT raw for diagnostics
    gt_motion_raw = motion_raw.clone() if motion_raw is not None else None

    # Main autoregressive loop
    plan_enable = bool(getattr(trainer.model, "contact_plan_enable", False)) if getattr(trainer, "model", None) is not None else False
    # NOTE: let the model decide the initial plan_z when plan_z is None.
    # This allows using a learnable contact_plan_init_z (or falling back to zeros).
    plan_z = None
    phase_z = None
    phase_event_age = None
    meas_prev_logits = None
    meas_prev_prob = None

    # Ensure phase_event_age is stateful when exporting plan_state_series.
    # EventMotionModel only maintains phase_event_age internally when either:
    #   - contact_phase_state_event_min_interval > 0, or
    #   - phase_event_age is provided (external state), or
    #   - it's used as a cue (e.g. leg_side_cue_mode == 'phase_event_age').
    #
    # For evaluation, we want phase_event_age_in==0 to be a reliable proxy for reset-applied frames even
    # when min_interval==0. The model does *not* track phase_event_age in that case unless it's provided,
    # so we seed it with zeros here. When min_interval>0, the model will initialize and track age internally
    # (starting at min_interval so the first event can fire immediately), so we should NOT override it.
    if bool(export_plan_state_series) and bool(plan_enable):
        try:
            min_interval0 = int(getattr(model, "contact_phase_state_event_min_interval", 0) or 0)
        except Exception:
            min_interval0 = 0
        if int(min_interval0) <= 0:
            try:
                Cc0 = int(getattr(model, "contact_dim", 0) or 0)
            except Exception:
                Cc0 = 0
            if int(Cc0) > 0:
                try:
                    phase_event_age = torch.zeros(
                        (int(motion.shape[0]), int(Cc0)), device=motion.device, dtype=motion.dtype
                    )
                except Exception:
                    phase_event_age = None

    # Optional: external phase reset / clock anchor from TTC countdown.
    phase_reset_source = str(getattr(trainer, "phase_reset_source", "contacts_meas") or "contacts_meas").strip().lower()
    ttc_event_kind = str(getattr(trainer, "ttc_event_kind", "touchdown") or "touchdown").strip().lower()
    ttc_max = getattr(trainer, "ttc_max", None)
    ttc_apply_phase_reset_to_phase_z = str(getattr(trainer, "ttc_apply_phase_reset_to_phase_z", "on") or "on").strip().lower()
    ttc_apply_phase_reset_to_phase_z = ttc_apply_phase_reset_to_phase_z not in ("off", "false", "0", "disable", "disabled")
    ttc_gt_full: Optional[torch.Tensor] = None        # (B,T,C) float
    ttc_gt_valid_full: Optional[torch.Tensor] = None  # (B,T,C) bool
    ttc_gt_event_full: Optional[torch.Tensor] = None  # (B,T,C) bool
    if phase_reset_source in ("ttc_gt", "ttc") and contacts_seq is None:
        print("[FreeRun][WARN] phase_reset_source uses TTC but teacher contacts are missing; falling back to contacts_meas.")
        phase_reset_source = "contacts_meas"
    if phase_reset_source in ("ttc_gt", "ttc"):
        try:
            if torch.is_tensor(contacts_seq) and contacts_seq.dim() == 3 and int(contacts_seq.shape[1]) > 0:
                thr = float(getattr(model, "contact_phase_state_event_thr", 0.5) or 0.5)
                hyst = float(getattr(model, "contact_phase_state_event_hyst", 0.0) or 0.0)
                # Compute TTC in cyclic (per-cycle) mode and select a single stable event per foot
                # to avoid multi-crossing jitter in soft_contact_score within one cycle.
                contacts_np_full = contacts_seq[0].detach().cpu().numpy()
                T_total_np = int(contacts_np_full.shape[0])
                T_cycle_np = int(max(1, int(T_cycle)))
                kind_eff = "touchdown" if ttc_event_kind in ("", "none") else ttc_event_kind  # avoid accidental disable
                if T_total_np >= T_cycle_np and (T_total_np % T_cycle_np) == 0:
                    ttc_list = []
                    valid_list = []
                    ev_list = []
                    for i in range(T_total_np // T_cycle_np):
                        seg = contacts_np_full[i * T_cycle_np:(i + 1) * T_cycle_np]
                        t_i, v_i, e_i = ttc_to_next_event_np(
                            seg,
                            thr=thr,
                            kind=kind_eff,
                            hyst=hyst,
                            ttc_max=ttc_max,
                            cyclic=True,
                            select="longest_run",
                        )
                        ttc_list.append(t_i)
                        valid_list.append(v_i)
                        ev_list.append(e_i)
                    ttc_np = np.concatenate(ttc_list, axis=0) if ttc_list else np.zeros_like(contacts_np_full, dtype=np.float32)
                    valid_np = np.concatenate(valid_list, axis=0) if valid_list else np.zeros_like(contacts_np_full, dtype=bool)
                    events_np = np.concatenate(ev_list, axis=0) if ev_list else np.zeros_like(contacts_np_full, dtype=bool)
                else:
                    ttc_np, valid_np, events_np = ttc_to_next_event_np(
                        contacts_np_full,
                        thr=thr,
                        kind=kind_eff,
                        hyst=hyst,
                        ttc_max=ttc_max,
                        cyclic=True,
                        select="longest_run",
                    )

                # Debug-only: shift TTC_gt event timestamps within each cycle.
                # This is useful to emulate a "strong contact" anchor (e.g., onset+5 frames) without changing the
                # underlying teacher contacts stream.
                try:
                    shift_raw = str(getattr(trainer, "ttc_gt_event_shift", "") or "").strip()
                except Exception:
                    shift_raw = ""
                if shift_raw:
                    try:
                        shifts = _parse_int_list_spec(shift_raw, n=int(contacts_np_full.shape[1]), default=0)
                    except Exception:
                        shifts = []
                    if shifts and any(int(s) != 0 for s in shifts):
                        try:
                            ev_in = np.asarray(events_np, dtype=bool)
                            T_total_s = int(ev_in.shape[0])
                            T_cycle_s = int(max(1, int(T_cycle_np)))
                            Cc_s = int(ev_in.shape[1])
                            if T_total_s <= 0 or Cc_s <= 0:
                                raise ValueError("empty events array")

                            if T_total_s >= T_cycle_s and (T_total_s % T_cycle_s) == 0:
                                nseg = int(T_total_s // T_cycle_s)
                                ev_out = np.zeros_like(ev_in, dtype=bool)
                                ttc_list2: List[np.ndarray] = []
                                valid_list2: List[np.ndarray] = []
                                for i in range(nseg):
                                    seg = ev_in[i * T_cycle_s:(i + 1) * T_cycle_s]
                                    seg_out = np.zeros_like(seg, dtype=bool)
                                    for ch, sh in enumerate(shifts):
                                        sh = int(sh)
                                        if sh == 0:
                                            seg_out[:, ch] = seg[:, ch]
                                            continue
                                        idx = np.where(seg[:, ch])[0].astype(np.int64)
                                        if idx.size == 0:
                                            continue
                                        seg_out[(idx + sh) % T_cycle_s, ch] = True
                                    ev_out[i * T_cycle_s:(i + 1) * T_cycle_s] = seg_out
                                    t_i, v_i = _ttc_from_events_cyclic_np(seg_out, ttc_max=ttc_max)
                                    ttc_list2.append(t_i)
                                    valid_list2.append(v_i)
                                events_np = ev_out
                                ttc_np = (
                                    np.concatenate(ttc_list2, axis=0)
                                    if ttc_list2
                                    else np.zeros_like(contacts_np_full, dtype=np.float32)
                                )
                                valid_np = (
                                    np.concatenate(valid_list2, axis=0)
                                    if valid_list2
                                    else np.zeros_like(contacts_np_full, dtype=bool)
                                )
                            else:
                                # Fallback: treat the full sequence as a single cycle.
                                seg_out = np.zeros_like(ev_in, dtype=bool)
                                for ch, sh in enumerate(shifts):
                                    sh = int(sh)
                                    if sh == 0:
                                        seg_out[:, ch] = ev_in[:, ch]
                                        continue
                                    idx = np.where(ev_in[:, ch])[0].astype(np.int64)
                                    if idx.size == 0:
                                        continue
                                    seg_out[(idx + sh) % T_total_s, ch] = True
                                events_np = seg_out
                                ttc_np, valid_np = _ttc_from_events_cyclic_np(seg_out, ttc_max=ttc_max)
                        except Exception as _shift_err:
                            print(f"[FreeRun][WARN] failed to apply --ttc_gt_event_shift={shift_raw!r}: {_shift_err}")
                ttc_gt_full = torch.from_numpy(ttc_np).to(device=device, dtype=state_seq.dtype).unsqueeze(0)
                ttc_gt_valid_full = torch.from_numpy(valid_np).to(device=device).unsqueeze(0)
                ttc_gt_event_full = torch.from_numpy(events_np).to(device=device).unsqueeze(0)
        except Exception as _ttc_err:
            print(f"[FreeRun][WARN] failed to compute TTC_gt from teacher contacts: {_ttc_err}")
            ttc_gt_full = None
            ttc_gt_valid_full = None
            ttc_gt_event_full = None
            phase_reset_source = "contacts_meas"

    # Closed-loop gate from contacts_err needs a stable reference level; we estimate it online
    # from the first few steps (gate=0) to avoid step-0 transients from plan_z initialization.
    ref_err_steps = int(getattr(trainer, "so3_corr_gate_err_ref_steps", 8) or 8)
    ref_err_steps = max(0, ref_err_steps)
    ref_err_margin = float(getattr(trainer, "so3_corr_gate_err_margin", 0.0) or 0.0)
    use_ref = bool(getattr(trainer, "so3_corr_gate_err_use_ref", False))
    ref_err_sum = 0.0
    ref_err_count = 0
    ref_err_value: Optional[float] = None
    prev_foot_pos_meas = None

    # Rotation slice/J for lambda fusion shape checks.
    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        rot_slice = slice(0, gt_seq.shape[-1])
    rot_len = int(rot_slice.stop - rot_slice.start)
    J = (rot_len // 6) if (rot_len > 0 and (rot_len % 6) == 0) else 0
    cols = ("X", "Z")
    try:
        columns = getattr(getattr(trainer, "loss_fn", None), "_rot6d_columns", ("X", "Z"))
        if isinstance(columns, (list, tuple)) and len(columns) >= 2:
            a = str(columns[0]).strip().upper()
            b = str(columns[1]).strip().upper()
            if a in ("X", "Y", "Z") and b in ("X", "Y", "Z") and a != b:
                cols = (a, b)
    except Exception:
        cols = ("X", "Z")

    pose_hist_hybrid_boundary_carry = bool(pose_hist_hybrid_boundary_carry)
    pose_hist_hybrid_enabled = False
    pose_hist_hybrid_leg_idx: Optional[torch.Tensor] = None
    donor_state: Optional[Dict[str, Any]] = None
    if (
        pose_hist_hybrid_boundary_carry
        and pose_hist_enabled
        and pose_hist_source == "buffer"
        and pose_hist_update_source == "pred"
    ):
        if donor_trainer is None:
            raise ValueError("pose_hist hybrid boundary carry requested but donor_trainer is missing.")
        phase_reset_is_none = str(phase_reset_source or "none").strip().lower() in (
            "",
            "none",
            "off",
            "false",
            "0",
            "disable",
            "disabled",
        )
        if not phase_reset_is_none:
            raise ValueError(
                f"pose_hist hybrid boundary carry prototype only supports phase_reset_source=none; got {phase_reset_source!r}."
            )
        contacts_meas_source_base = str(getattr(trainer, "contacts_meas_source", "model") or "model").strip().lower()
        if contacts_meas_source_base not in (
            "zero",
            "ignore",
            "none",
            "pretrain_contact",
            "pretrain",
            "frozen_contact",
            "gt",
            "teacher",
        ):
            raise ValueError(
                "pose_hist hybrid boundary carry prototype only supports external contacts_meas_source "
                f"(got {contacts_meas_source_base!r})."
            )
        donor_model = getattr(donor_trainer, "model", None)
        if donor_model is None:
            raise ValueError("pose_hist hybrid donor trainer is missing model.")
        donor_rot_slice = getattr(donor_trainer, "rot6d_y_slice", None) or getattr(donor_trainer, "rot6d_slice", None)
        if not isinstance(donor_rot_slice, slice):
            donor_rot_slice = slice(0, gt_seq.shape[-1])
        donor_rot_len = int(donor_rot_slice.stop - donor_rot_slice.start)
        if int(donor_rot_len) != int(rot_len) or int(donor_rot_len) != int(pose_hist_stride):
            raise ValueError(
                "pose_hist hybrid donor rot6d layout mismatch: "
                f"current rot_len={rot_len}, donor rot_len={donor_rot_len}, pose_hist_stride={pose_hist_stride}."
            )
        pose_hist_hybrid_leg_idx = _resolve_direct_pose_leg_idx_tensor(model, device=device)
        donor_leg_idx = _resolve_direct_pose_leg_idx_tensor(donor_model, device=device)
        if not torch.is_tensor(pose_hist_hybrid_leg_idx) or int(pose_hist_hybrid_leg_idx.numel()) <= 0:
            raise ValueError("pose_hist hybrid boundary carry requires current model direct_pose_leg_joint_idx.")
        if not torch.is_tensor(donor_leg_idx) or int(donor_leg_idx.numel()) <= 0:
            raise ValueError("pose_hist hybrid boundary carry requires donor model direct_pose_leg_joint_idx.")
        if (
            pose_hist_hybrid_leg_idx.shape != donor_leg_idx.shape
            or not torch.equal(pose_hist_hybrid_leg_idx.detach().cpu(), donor_leg_idx.detach().cpu())
        ):
            raise ValueError("pose_hist hybrid donor leg joint set does not match current model leg joint set.")

        donor_pose_hist_len = int(getattr(donor_trainer, "pose_hist_len", 0) or 0)
        donor_pose_hist_dim = int(getattr(donor_trainer, "pose_hist_dim", 0) or 0)
        donor_pose_hist_stride = donor_pose_hist_dim // donor_pose_hist_len if donor_pose_hist_len > 0 else 0
        donor_pose_hist_enabled = donor_pose_hist_len > 0 and donor_pose_hist_dim > 0 and donor_pose_hist_stride > 0
        if (not donor_pose_hist_enabled) or int(donor_pose_hist_stride) != int(pose_hist_stride):
            raise ValueError(
                "pose_hist hybrid donor pose_history layout mismatch: "
                f"current stride={pose_hist_stride}, donor stride={donor_pose_hist_stride}."
            )

        donor_motion = state_seq[:, start_t]
        donor_motion_raw = None
        if getattr(donor_trainer, "normalizer", None) is not None:
            try:
                donor_motion_raw = donor_trainer.normalizer.denorm_x(donor_motion)
            except Exception:
                donor_motion_raw = None
        donor_y_raw_prev = None
        try:
            donor_y_raw_prev = donor_trainer._denorm(gt_seq[:, start_t])
        except Exception:
            donor_y_raw_prev = None
        if donor_y_raw_prev is None and donor_motion_raw is not None:
            slice_len = int(donor_rot_slice.stop - donor_rot_slice.start)
            if slice_len == gt_seq.shape[-1]:
                try:
                    donor_y_raw_prev = donor_motion_raw[:, donor_rot_slice].clone()
                except Exception:
                    donor_y_raw_prev = None
        donor_plan_enable = bool(getattr(donor_model, "contact_plan_enable", False))
        donor_phase_event_age = None
        if bool(export_plan_state_series) and bool(donor_plan_enable):
            try:
                donor_min_interval0 = int(getattr(donor_model, "contact_phase_state_event_min_interval", 0) or 0)
            except Exception:
                donor_min_interval0 = 0
            if int(donor_min_interval0) <= 0:
                try:
                    donor_contact_dim0 = int(getattr(donor_model, "contact_dim", 0) or 0)
                except Exception:
                    donor_contact_dim0 = 0
                if int(donor_contact_dim0) > 0:
                    try:
                        donor_phase_event_age = torch.zeros(
                            (int(donor_motion.shape[0]), int(donor_contact_dim0)),
                            device=donor_motion.device,
                            dtype=donor_motion.dtype,
                        )
                    except Exception:
                        donor_phase_event_age = None

        donor_state = {
            "trainer": donor_trainer,
            "model": donor_model,
            "motion": donor_motion,
            "motion_raw": donor_motion_raw,
            "gt_motion_raw": donor_motion_raw.clone() if torch.is_tensor(donor_motion_raw) else None,
            "y_raw_prev": donor_y_raw_prev,
            "plan_enable": donor_plan_enable,
            "plan_z": None,
            "phase_z": None,
            "phase_event_age": donor_phase_event_age,
            "meas_prev_logits": None,
            "meas_prev_prob": None,
            "pose_hist_state": _init_eval_pose_hist_state(
                donor_trainer,
                ref_tensor=state_seq,
                pose_hist_seq=pose_hist_seq,
                step=0,
                device=device,
                dtype=state_seq.dtype,
            ),
            "rot_slice": donor_rot_slice,
        }
        if not bool(donor_state["pose_hist_state"].enabled):
            raise ValueError("pose_hist hybrid donor missing pose_history normalization stats.")
        pose_hist_hybrid_enabled = True

    def _advance_pose_hist_hybrid_donor_step(
        *,
        step_t: int,
        is_cycle_start_step: bool,
        gt_motion_next_shared: torch.Tensor,
        cond_raw_step_shared: Optional[torch.Tensor],
        contacts_in_step: Optional[torch.Tensor],
        time_index_step: Optional[int],
        rollout_step_step: Optional[torch.Tensor],
        direct_meas_override_step: Any,
        direct_plan_override_step: Any,
        gate_override_step: Any,
        amp_ctx: Any,
    ) -> Optional[torch.Tensor]:
        nonlocal donor_state
        if not pose_hist_hybrid_enabled or donor_state is None:
            return None

        dtr = donor_state["trainer"]
        dmodel = donor_state["model"]
        donor_motion = donor_state["motion"]
        donor_motion_raw = donor_state["motion_raw"]
        donor_y_raw_prev = donor_state["y_raw_prev"]
        donor_pose_hist_state = donor_state["pose_hist_state"]

        if (
            is_cycle_start_step
            and bool(multicycle_reset_pose_hist_on_cycle_start)
            and bool(donor_pose_hist_state.enabled)
        ):
            donor_pose_hist_state = _init_eval_pose_hist_state(
                dtr,
                ref_tensor=state_seq,
                pose_hist_seq=pose_hist_seq,
                step=step_t,
                device=device,
                dtype=state_seq.dtype,
            )
            donor_state["pose_hist_state"] = donor_pose_hist_state
        if is_cycle_start_step and bool(multicycle_sync_state_on_cycle_start):
            try:
                donor_motion = state_seq[:, step_t].detach()
            except Exception:
                pass
            if getattr(dtr, "normalizer", None) is not None:
                try:
                    donor_motion_raw = dtr.normalizer.denorm_x(donor_motion)
                except Exception:
                    donor_motion_raw = None
            try:
                donor_y_raw_prev = dtr._denorm(gt_seq[:, step_t])
            except Exception:
                donor_y_raw_prev = None
            if bool(donor_pose_hist_state.enabled):
                donor_pose_hist_state = _init_eval_pose_hist_state(
                    dtr,
                    ref_tensor=state_seq,
                    pose_hist_seq=pose_hist_seq,
                    step=step_t,
                    device=device,
                    dtype=state_seq.dtype,
                )
                donor_state["pose_hist_state"] = donor_pose_hist_state
        if is_cycle_start_step and bool(multicycle_reset_plan_z_on_cycle_start) and bool(donor_state["plan_enable"]):
            donor_state["plan_z"] = None
            donor_state["phase_z"] = None
            donor_state["phase_event_age"] = None

        donor_gt_motion_raw = donor_state["gt_motion_raw"]
        if donor_gt_motion_raw is not None:
            try:
                donor_gt_motion_raw = dtr.normalizer.denorm_x(gt_motion_next_shared, prev_raw=donor_gt_motion_raw)
            except Exception:
                donor_gt_motion_raw = None
        donor_state["gt_motion_raw"] = donor_gt_motion_raw

        if getattr(dtr, "use_freerun_state_sync", False) and isinstance(getattr(dtr, "angvel_x_slice", None), slice):
            donor_angvel_t = donor_motion[..., dtr.angvel_x_slice].detach()
        else:
            donor_angvel_t = angvel_seq[:, step_t] if (angvel_seq is not None and angvel_seq.dim() == 3) else angvel_seq

        donor_pose_hist_t = resolve_pose_hist_input(
            state=donor_pose_hist_state,
            pose_hist_seq=None,
            idx=step_t,
        )
        if donor_pose_hist_t is None:
            donor_pose_hist_t = torch.zeros(
                (B, int(donor_pose_hist_state.dim)),
                device=device,
                dtype=state_seq.dtype,
            )

        donor_cond_input = cond_seq[:, step_t] if (cond_seq is not None and cond_seq.dim() == 3) else cond_seq
        donor_cond_raw_for_model = cond_raw_step_shared
        enable_reproj_donor = (cond_reprojection != "off")
        if cond_reprojection == "auto":
            try:
                enable_reproj_donor = bool(getattr(dtr, "enable_cond_reprojection", True))
                donor_yaw_strategy = str(getattr(dtr, "freerun_yaw_strategy", "trajectory") or "trajectory")
                if donor_yaw_strategy == "trajectory":
                    enable_reproj_donor = False
            except Exception:
                enable_reproj_donor = False
        if enable_reproj_donor and donor_cond_raw_for_model is not None and step_t > 0:
            donor_yaw_gt = None
            if gt_seq is not None and gt_seq.dim() == 3:
                try:
                    donor_gt_idx = min(gt_seq.shape[1] - 1, step_t)
                    donor_gt_raw_frame = dtr._denorm(gt_seq[:, donor_gt_idx])
                    donor_yaw_gt = dtr._infer_root_yaw_from_rot6d(donor_gt_raw_frame)
                except Exception:
                    donor_yaw_gt = None
            donor_pred_yaw = dtr._infer_root_yaw_from_rot6d(donor_y_raw_prev) if donor_y_raw_prev is not None else None
            if donor_yaw_gt is not None and donor_pred_yaw is not None:
                try:
                    donor_reproj = dtr._reproject_cond_to_local_frame(
                        donor_cond_raw_for_model,
                        donor_yaw_gt,
                        donor_pred_yaw,
                    )
                except Exception:
                    donor_reproj = None
                if donor_reproj is not None:
                    donor_cond_raw_for_model = donor_reproj
        if donor_cond_raw_for_model is not None:
            donor_cond_override = dtr._normalize_cond_from_raw(donor_cond_raw_for_model, cond_norm_mu, cond_norm_std)
            if donor_cond_override is not None:
                donor_cond_input = donor_cond_override

        try:
            setattr(dmodel, "direct_pose_meas_override", direct_meas_override_step)
        except Exception:
            pass
        try:
            setattr(dmodel, "direct_pose_plan_override", direct_plan_override_step)
        except Exception:
            pass

        donor_meas_prev_in = donor_state["meas_prev_prob"]
        with amp_ctx:
            donor_ret = dmodel(
                donor_motion,
                donor_cond_input,
                contacts=contacts_in_step,
                angvel=donor_angvel_t,
                pose_history=donor_pose_hist_t,
                plan_z=donor_state["plan_z"],
                phase_z=donor_state["phase_z"],
                phase_event_age=donor_state["phase_event_age"],
                meas_logits_prev=donor_meas_prev_in,
                time_index=time_index_step,
                rollout_step=rollout_step_step,
            )
        if not isinstance(donor_ret, dict):
            raise RuntimeError("pose_hist hybrid donor forward must return dict.")
        donor_out = donor_ret.get("out", None)
        if donor_out is None:
            raise RuntimeError("pose_hist hybrid donor forward missing 'out'.")

        if bool(donor_state["plan_enable"]):
            try:
                donor_z_next = donor_ret.get("plan_z_next", None)
                if donor_z_next is not None:
                    donor_state["plan_z"] = donor_z_next.detach()
                donor_p_next = donor_ret.get("phase_z_next", None)
                if donor_p_next is not None:
                    donor_state["phase_z"] = donor_p_next.detach()
                donor_a_next = donor_ret.get("phase_event_age_next", None)
                if donor_a_next is not None:
                    donor_state["phase_event_age"] = donor_a_next.detach()
            except Exception:
                pass
        try:
            donor_meas_logits_step = donor_ret.get("contacts_meas_logits", None)
            if torch.is_tensor(donor_meas_logits_step):
                donor_state["meas_prev_logits"] = donor_meas_logits_step.detach()
            donor_meas_prob_step = donor_ret.get("contacts_meas", None)
            if torch.is_tensor(donor_meas_prob_step):
                donor_state["meas_prev_prob"] = donor_meas_prob_step.detach()
        except Exception:
            pass

        if donor_y_raw_prev is not None:
            try:
                donor_so3_gate = gate_override_step if gate_override_step is not None else getattr(dtr, "so3_corr_gate_force", None)
                donor_y_inc_raw = dtr._compose_delta_to_raw(
                    donor_y_raw_prev,
                    donor_out,
                    omega_hat=donor_ret.get("omega_hat", None) if bool(getattr(dtr, "so3_corr_apply", False)) else None,
                    so3_gate=donor_so3_gate,
                    so3_max_deg=getattr(dtr, "so3_corr_max_deg", None),
                    omega_detach=True,
                )
            except Exception:
                donor_y_inc_raw = dtr._denorm(donor_out)
        else:
            donor_y_inc_raw = dtr._denorm(donor_out)

        donor_y_blend_raw = donor_y_inc_raw
        if bool(lambda_fusion_apply) and donor_y_inc_raw is not None and torch.is_tensor(donor_y_inc_raw):
            donor_lam_step = None
            donor_lam_eff_step = None
            try:
                donor_lam = donor_ret.get("lambda_fusion", None)
                if torch.is_tensor(donor_lam):
                    if donor_lam.dim() == 3 and donor_lam.size(1) == 1:
                        donor_lam = donor_lam[:, 0]
                    if donor_lam.dim() == 1:
                        if donor_lam.shape[0] == donor_motion.shape[0]:
                            donor_lam = donor_lam.unsqueeze(-1)
                        elif donor_motion.shape[0] == 1 and J > 0 and donor_lam.shape[0] == J:
                            donor_lam = donor_lam.unsqueeze(0)
                    if donor_lam.dim() == 2 and donor_lam.shape[0] == donor_motion.shape[0]:
                        if donor_lam.shape[-1] == 1 and J > 0:
                            donor_lam = donor_lam.expand(donor_lam.shape[0], J)
                        if J > 0 and donor_lam.shape[-1] == J:
                            donor_lam_step = donor_lam.clamp(0.0, 1.0)
                            donor_lam_eff_step = donor_lam_step
                            try:
                                donor_lam_eff_step, _ = dtr._lambda_fusion_apply_reliability(
                                    donor_lam_step,
                                    step_idx=int(step_t - start_t),
                                    total_steps=int(max(1, int(end_t - start_t))),
                                    rollout_step=rollout_step_step,
                                    ret=donor_ret,
                                )
                            except Exception:
                                donor_lam_eff_step = donor_lam_step
            except Exception:
                donor_lam_step = None
                donor_lam_eff_step = None
            donor_direct_norm_step = None
            try:
                donor_direct_out = donor_ret.get("out_direct", None)
                if torch.is_tensor(donor_direct_out):
                    if donor_direct_out.dim() == 3 and donor_direct_out.size(1) == 1:
                        donor_direct_out = donor_direct_out[:, 0]
                    if donor_direct_out.dim() == 2 and donor_direct_out.shape[0] == donor_motion.shape[0]:
                        donor_direct_norm_step = donor_direct_out
            except Exception:
                donor_direct_norm_step = None
            donor_lam_for_blend = donor_lam_eff_step if torch.is_tensor(donor_lam_eff_step) else donor_lam_step
            if torch.is_tensor(donor_direct_norm_step) and torch.is_tensor(donor_lam_for_blend):
                try:
                    donor_y_blend_raw = dtr._apply_lambda_fusion_to_raw(
                        donor_y_inc_raw,
                        direct_norm=donor_direct_norm_step,
                        lambda_fusion=donor_lam_for_blend,
                    )
                except Exception:
                    donor_y_blend_raw = donor_y_inc_raw

        donor_y_used_raw = donor_y_blend_raw if bool(lambda_fusion_apply) else donor_y_inc_raw
        donor_state["y_raw_prev"] = donor_y_used_raw.detach() if torch.is_tensor(donor_y_used_raw) else None

        if donor_motion_raw is not None:
            donor_motion_raw = dtr._apply_free_carry(donor_motion_raw, donor_y_used_raw, cond_next_raw=cond_raw_step_shared).detach()
            donor_motion = dtr._diag_norm_x(donor_motion_raw)
            if bool(freerun_x_gt) and torch.is_tensor(donor_gt_motion_raw) and donor_gt_motion_raw.shape == donor_motion_raw.shape:
                try:
                    donor_motion_raw = donor_gt_motion_raw.detach()
                    donor_motion = dtr._diag_norm_x(donor_motion_raw)
                except Exception:
                    pass
            elif bool(freerun_x_gt_except_rot6d) and torch.is_tensor(donor_gt_motion_raw) and donor_gt_motion_raw.shape == donor_motion_raw.shape:
                donor_rx = getattr(dtr, "rot6d_x_slice", None) or getattr(dtr, "rot6d_slice", None)
                if isinstance(donor_rx, slice):
                    try:
                        donor_hybrid_x = donor_gt_motion_raw.detach().clone()
                        donor_hybrid_x[..., donor_rx] = donor_motion_raw[..., donor_rx]
                        donor_motion_raw = donor_hybrid_x
                        donor_motion = dtr._diag_norm_x(donor_motion_raw)
                    except Exception:
                        pass
        else:
            donor_motion = dtr._apply_free_carry(donor_motion, donor_y_used_raw, cond_next_raw=None).detach()
        donor_state["motion"] = donor_motion
        donor_state["motion_raw"] = donor_motion_raw

        if bool(donor_pose_hist_state.enabled) and int(donor_pose_hist_state.stride) > 0:
            donor_rot_write = None
            if torch.is_tensor(donor_y_used_raw):
                donor_rot_slice = donor_state.get("rot_slice")
                if isinstance(donor_rot_slice, slice):
                    donor_rot_write = donor_y_used_raw[..., donor_rot_slice]
            donor_pose_hist_state = advance_pose_hist_state_with_tail(
                donor_pose_hist_state,
                rot_tail_raw=donor_rot_write,
            )
            donor_state["pose_hist_state"] = donor_pose_hist_state
        return donor_y_used_raw

    # Optional: autograd gradient diagnostics for the leg-omega head (DirectGeoLocalDeg loss).
    export_direct_leg_omega_grad = bool(export_direct_leg_omega_grad)
    direct_leg_omega_grad_steps: List[Dict[str, Any]] = []
    direct_leg_omega_grad_meta: Dict[str, Any] = {}
    direct_leg_omega_grad_sics_set: Optional[set[int]] = None
    direct_leg_omega_grad_bone_k: Optional[List[int]] = None  # subset of K (leg-head output joints) to report
    direct_leg_omega_grad_bone_names: Optional[List[str]] = None
    direct_leg_omega_grad_bone_joint_idx: Optional[List[int]] = None  # corresponding full-body joint indices
    direct_leg_omega_grad_leg_named_params: List[Tuple[str, torch.Tensor]] = []
    direct_leg_omega_grad_leg_params: List[torch.Tensor] = []
    grad_drop_wrap = True
    grad_cycle_gte = 1
    direct_leg_omega_grad_joint_mask: Optional[torch.Tensor] = None
    deg_factor = 180.0 / float(np.pi)
    if export_direct_leg_omega_grad:
        # Parse sics (supports "a,b,c" and "a-b" / "a:b").
        def _parse_int_set(spec: str) -> Optional[set[int]]:
            s = str(spec or "").strip()
            if not s:
                return None
            out: set[int] = set()
            for tok in s.replace(";", ",").split(","):
                t = tok.strip()
                if not t:
                    continue
                if "-" in t or ":" in t:
                    sep = "-" if "-" in t else ":"
                    a, b = [x.strip() for x in t.split(sep, 1)]
                    if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
                        try:
                            lo = int(a)
                            hi = int(b)
                            if lo > hi:
                                lo, hi = hi, lo
                            for v in range(lo, hi + 1):
                                out.add(int(v))
                        except Exception:
                            pass
                    continue
                if t.lstrip("-").isdigit():
                    try:
                        out.add(int(t))
                    except Exception:
                        pass
            return out if out else None

        def _flag_on(spec: str) -> bool:
            s = str(spec or "").strip().lower()
            return s not in ("", "0", "false", "off", "disable", "disabled", "none", "null")

        direct_leg_omega_grad_sics_set = _parse_int_set(direct_leg_omega_grad_sics)
        try:
            grad_cycle_gte = int(direct_leg_omega_grad_cycle_gte)
        except Exception:
            grad_cycle_gte = 1
        grad_cycle_gte = max(0, int(grad_cycle_gte))
        grad_drop_wrap = _flag_on(direct_leg_omega_grad_drop_wrap)

        # Joint mask (exclude root) matches DirectGeoLocalDeg definition.
        root_idx_eval = int(getattr(trainer, "eval_root_idx", 0) or 0)
        if int(J) > 0:
            root_idx_eval = max(0, min(int(J) - 1, int(root_idx_eval)))
            direct_leg_omega_grad_joint_mask = torch.ones(int(J), device=device, dtype=torch.bool)
            direct_leg_omega_grad_joint_mask[int(root_idx_eval)] = False

        # Collect leg-head parameters (named_parameters uses the module attribute names, e.g. direct_pose_leg_head.*).
        try:
            for name, p in model.named_parameters():
                if (p is not None) and torch.is_tensor(p) and bool(getattr(p, "requires_grad", False)):
                    if "direct_pose_leg" in str(name):
                        direct_leg_omega_grad_leg_named_params.append((str(name), p))
                        direct_leg_omega_grad_leg_params.append(p)
        except Exception:
            direct_leg_omega_grad_leg_named_params = []
            direct_leg_omega_grad_leg_params = []

        # Report subset of leg bones (defaults to all legs).
        leg_joint_names = list(getattr(model, "direct_pose_leg_joint_names", []) or [])
        leg_joint_idx = list(getattr(model, "direct_pose_leg_joint_idx", []) or [])
        if int(len(leg_joint_names)) != int(len(leg_joint_idx)):
            leg_joint_names = [str(i) for i in leg_joint_idx]
        bones_spec = str(direct_leg_omega_grad_bones or "leg").strip()
        bones_spec_l = bones_spec.lower()
        bones_keep: Optional[set[str]] = None
        if bones_spec_l not in ("", "all", "leg"):
            bones_keep = {tok.strip() for tok in bones_spec.split(",") if tok.strip()}
            if not bones_keep:
                bones_keep = None

        keep_k: List[int] = []
        keep_names: List[str] = []
        keep_joint_idx: List[int] = []
        for k, (nm, j_idx) in enumerate(zip(leg_joint_names, leg_joint_idx)):
            if bones_keep is not None and str(nm) not in bones_keep:
                continue
            keep_k.append(int(k))
            keep_names.append(str(nm))
            try:
                keep_joint_idx.append(int(j_idx))
            except Exception:
                keep_joint_idx.append(int(-1))

        direct_leg_omega_grad_bone_k = keep_k
        direct_leg_omega_grad_bone_names = keep_names
        direct_leg_omega_grad_bone_joint_idx = keep_joint_idx
        direct_leg_omega_grad_meta = {
            "enabled": True,
            "loss": "DirectGeoLocalDeg",
            "units": {"loss": "deg", "domega": "deg_per_rad", "theta_grad_norm": "unitless"},
            "mask": {
                "cycle_gte": int(grad_cycle_gte),
                "drop_wrap": bool(grad_drop_wrap),
                "sics": sorted(list(direct_leg_omega_grad_sics_set)) if direct_leg_omega_grad_sics_set is not None else None,
            },
            "bones": direct_leg_omega_grad_bone_names,
            "joint_idx": direct_leg_omega_grad_bone_joint_idx,
            "params": {"count": int(len(direct_leg_omega_grad_leg_params))},
            "note": (
                "Gradients are computed per selected step as autograd d(loss)/d(theta_legomega) and d(loss)/d(omega_pred).\n"
                "- loss is DirectGeoLocalDeg: mean geodesic distance (deg) over all non-root joints.\n"
                "- d(omega_pred) is reported per leg joint (||dL/domega_xyz||, mean over batch).\n"
                "- theta_legomega includes all parameters whose name contains 'direct_pose_leg'."
            ),
        }

    # Optional: finite-difference probe for local closed-loop gain (contractiveness).
    debug_rot_gain = bool(debug_rot_gain)
    rot_gain_axis = str(rot_gain_axis or "z").strip().lower()
    if rot_gain_axis not in ("x", "y", "z"):
        rot_gain_axis = "z"
    try:
        rot_gain_deg = float(rot_gain_deg)
    except Exception:
        rot_gain_deg = 0.5
    rot_gain_joint_indices: List[int] = []
    rot_gain_joint_names: List[str] = []
    rot_gain_bone_names: List[str] = []
    rot_gain_root_idx = int(getattr(trainer, "eval_root_idx", 0) or 0)
    if debug_rot_gain and int(J) > 0:
        try:
            loss_fn = getattr(trainer, "loss_fn", None)
            bone_names = getattr(loss_fn, "bone_names", None) if loss_fn is not None else None
            if not bone_names:
                bone_names = getattr(trainer, "_bone_names", None)
            if not bone_names:
                bundle_meta = getattr(trainer, "_bundle_meta", None)
                if isinstance(bundle_meta, dict):
                    bone_names = bundle_meta.get("bone_names") or bundle_meta.get("skeleton", {}).get("bone_names")
            rot_gain_bone_names = [str(b) for b in bone_names] if isinstance(bone_names, (list, tuple)) else []
            if len(rot_gain_bone_names) < int(J):
                rot_gain_bone_names = rot_gain_bone_names + [
                    f"joint_{i}" for i in range(len(rot_gain_bone_names), int(J))
                ]
            rot_gain_bone_names = rot_gain_bone_names[: int(J)]
            idx_map = {name: idx for idx, name in enumerate(rot_gain_bone_names)}

            spec = str(rot_gain_joints or "").strip()
            spec_l = spec.lower()
            if spec_l in ("all", "*"):
                rot_gain_joint_indices = [i for i in range(int(J)) if i != int(rot_gain_root_idx)]
            elif spec_l in ("keybones", "key_bones"):
                key_bone_names = getattr(loss_fn, "eval_key_bones", None) if loss_fn is not None else None
                if not key_bone_names:
                    key_bone_names = [
                        "pelvis",
                        "upperarm_l",
                        "lowerarm_l",
                        "hand_l",
                        "upperarm_r",
                        "lowerarm_r",
                        "hand_r",
                        "thigh_l",
                        "calf_l",
                        "foot_l",
                        "thigh_r",
                        "calf_r",
                        "foot_r",
                    ]
                for name in key_bone_names:
                    if name in idx_map:
                        rot_gain_joint_indices.append(int(idx_map[name]))
            else:
                for tok in [t.strip() for t in spec.split(",") if t.strip()]:
                    idx = None
                    try:
                        idx = int(tok)
                    except Exception:
                        idx = idx_map.get(tok, None)
                    if idx is None:
                        continue
                    if 0 <= int(idx) < int(J):
                        rot_gain_joint_indices.append(int(idx))

            seen = set()
            rot_gain_joint_indices = [i for i in rot_gain_joint_indices if not (i in seen or seen.add(i))]
            rot_gain_joint_names = [rot_gain_bone_names[i] for i in rot_gain_joint_indices if 0 <= int(i) < len(rot_gain_bone_names)]
            if not rot_gain_joint_indices:
                debug_rot_gain = False
                print("[FreeRun][WARN] --debug_rot_gain set but --rot_gain_joints resolved to empty; disabling.")
        except Exception:
            debug_rot_gain = False

    # Optional: distribution-matched "soft-GT" hint mapping for direct plan/meas overrides.
    # This is strictly a runtime ablation for the *direct* head only (does not change contacts_err / Event-Clock / λ).
    softgt_cfg = getattr(trainer, "direct_pose_softgt_stats", None)
    softgt_cfg_plan = softgt_cfg.get("plan") if isinstance(softgt_cfg, dict) else None
    softgt_cfg_meas = softgt_cfg.get("meas") if isinstance(softgt_cfg, dict) else None
    softgt_clamp = None
    try:
        if isinstance(softgt_cfg, dict) and softgt_cfg.get("clamp", None) is not None:
            softgt_clamp = float(softgt_cfg.get("clamp"))
    except Exception:
        softgt_clamp = None
    if softgt_clamp is None or (not np.isfinite(softgt_clamp)):
        softgt_clamp = 1e-4
    softgt_clamp = float(max(1e-6, min(0.499, softgt_clamp)))
    _softgt_tensor_cache: Dict[Tuple[str, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    # Debug-only: override contacts_meas with GT on selected step_in_cycle (sic), while keeping model meas elsewhere.
    cm_gt_override_sics_spec = str(getattr(trainer, "contacts_meas_gt_override_sics", "") or "").strip()
    try:
        cm_gt_override_cycle_gte = int(getattr(trainer, "contacts_meas_gt_override_cycle_gte", 1) or 1)
    except Exception:
        cm_gt_override_cycle_gte = 1
    cm_gt_override_cycle_gte = max(0, int(cm_gt_override_cycle_gte))
    s_drop_cm = str(getattr(trainer, "contacts_meas_gt_override_drop_wrap", "on") or "on").strip().lower()
    cm_gt_override_drop_wrap = s_drop_cm not in ("off", "false", "0", "disable", "disabled")

    cm_gt_override_sics: Optional[set[int]] = None
    if cm_gt_override_sics_spec:
        tmp: set[int] = set()
        for tok in cm_gt_override_sics_spec.replace(";", ",").split(","):
            s = tok.strip()
            if not s:
                continue
            if "-" in s:
                a, b = [x.strip() for x in s.split("-", 1)]
                if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
                    try:
                        lo = int(a)
                        hi = int(b)
                        if lo > hi:
                            lo, hi = hi, lo
                        for v in range(lo, hi + 1):
                            tmp.add(int(v))
                    except Exception:
                        pass
                continue
            if s.lstrip("-").isdigit():
                try:
                    tmp.add(int(s))
                except Exception:
                    pass
        cm_gt_override_sics = tmp if tmp else None

    def _softgt_map(gt_contacts_t: Optional[torch.Tensor], *, kind: str) -> Optional[torch.Tensor]:
        if not torch.is_tensor(gt_contacts_t):
            return None
        cfg = softgt_cfg_plan if kind == "plan" else softgt_cfg_meas if kind == "meas" else None
        if not isinstance(cfg, dict):
            return gt_contacts_t
        scale = cfg.get("scale", None)
        bias = cfg.get("bias", None)
        if not isinstance(scale, (list, tuple)) or not isinstance(bias, (list, tuple)):
            return gt_contacts_t
        C = int(gt_contacts_t.shape[-1])
        if int(len(scale)) != C or int(len(bias)) != C:
            return gt_contacts_t
        try:
            # logit(p) = log(p) - log(1-p)
            p = gt_contacts_t.to(dtype=gt_contacts_t.dtype).clamp(softgt_clamp, 1.0 - softgt_clamp)
            logit = torch.log(p) - torch.log1p(-p)
        except Exception:
            return gt_contacts_t
        key = (str(kind), int(C), str(gt_contacts_t.dtype))
        if key not in _softgt_tensor_cache:
            try:
                s = torch.tensor([float(x) for x in scale], device=gt_contacts_t.device, dtype=gt_contacts_t.dtype).view(1, C)
                b = torch.tensor([float(x) for x in bias], device=gt_contacts_t.device, dtype=gt_contacts_t.dtype).view(1, C)
                _softgt_tensor_cache[key] = (s, b)
            except Exception:
                return gt_contacts_t
        s, b = _softgt_tensor_cache[key]
        return torch.sigmoid(b + s * logit)

    def _predict_pretrain_contacts_from_frozen(
        motion_t: Optional[torch.Tensor],
        pose_hist_t: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], bool]:
        """
        Predict contact probs from frozen pretrain encoder/contact head.

        Important:
        - input contact channels are zeroed to avoid trivial leakage/copy.
        - this is a runtime diagnostic source, independent from contact_meas_head.
        """
        if not torch.is_tensor(motion_t):
            return None, False
        enc = getattr(model, "frozen_encoder", None)
        head = getattr(model, "frozen_contact_head", None)
        if enc is None or head is None:
            return None, False
        try:
            affine_applied = False
            cdim = int(getattr(model, "contact_dim", 0) or 0)
            in_dim = int(getattr(model, "encoder_input_dim", 0) or 0)
            try:
                pre_clamp = float(getattr(trainer, "contacts_meas_pretrain_clamp", 1.0) or 0.0)
            except Exception:
                pre_clamp = 1.0
            if cdim <= 0 or in_dim <= 0:
                return None, False
            B_loc = int(motion_t.shape[0])
            dev = motion_t.device
            dtp = motion_t.dtype

            c_seed = torch.zeros((B_loc, cdim), device=dev, dtype=dtp)

            ang_t = None
            av_sl = getattr(model, "_contact_meas_state_angvel_slice", None)
            if isinstance(av_sl, slice):
                try:
                    ang_t = motion_t[..., av_sl]
                except Exception:
                    ang_t = None
            if not torch.is_tensor(ang_t):
                ang_t = torch.zeros((B_loc, 0), device=dev, dtype=dtp)
            elif ang_t.ndim != 2:
                ang_t = ang_t.reshape(B_loc, -1)

            if torch.is_tensor(pose_hist_t):
                ph_t = pose_hist_t.to(device=dev, dtype=dtp)
                if ph_t.ndim == 3 and ph_t.size(1) == 1:
                    ph_t = ph_t[:, 0]
                elif ph_t.ndim != 2:
                    ph_t = ph_t.reshape(B_loc, -1)
            else:
                ph_t = torch.zeros((B_loc, 0), device=dev, dtype=dtp)

            enc_in = torch.cat([c_seed, ang_t, ph_t], dim=-1)
            if int(enc_in.shape[-1]) != int(in_dim):
                if int(enc_in.shape[-1]) > int(in_dim):
                    enc_in = enc_in[..., : int(in_dim)]
                else:
                    enc_in = torch.nn.functional.pad(enc_in, (0, int(in_dim) - int(enc_in.shape[-1])))
            if np.isfinite(float(pre_clamp)) and float(pre_clamp) > 0.0:
                enc_in = enc_in.clamp(-float(pre_clamp), float(pre_clamp))

            with torch.no_grad():
                h = enc(enc_in.unsqueeze(1), return_summary=False)
                logits = head(h)
                if torch.is_tensor(logits) and logits.ndim == 3 and logits.size(1) == 1:
                    logits = logits[:, 0]
                if (not torch.is_tensor(logits)) or logits.ndim != 2:
                    return None, False
                affine_cfg = getattr(trainer, "contacts_meas_pretrain_affine", None)
                if isinstance(affine_cfg, dict):
                    scale = affine_cfg.get("scale", None)
                    bias = affine_cfg.get("bias", None)
                    try:
                        eps = float(affine_cfg.get("eps", 1e-4) or 1e-4)
                    except Exception:
                        eps = 1e-4
                    if not np.isfinite(float(eps)):
                        eps = 1e-4
                    eps = float(min(1e-2, max(1e-8, eps)))
                    C = int(logits.shape[-1])
                    if isinstance(scale, (list, tuple)) and isinstance(bias, (list, tuple)):
                        if int(len(scale)) == C and int(len(bias)) == C:
                            s = torch.tensor([float(x) for x in scale], device=dev, dtype=logits.dtype).view(1, C)
                            b = torch.tensor([float(x) for x in bias], device=dev, dtype=logits.dtype).view(1, C)
                            # Calibrate in logit space: p' = sigmoid(b + s * logit(p)).
                            p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
                            logit_p = torch.log(p) - torch.log1p(-p)
                            logits = b + s * logit_p
                            affine_applied = True
                probs = torch.sigmoid(logits)
                if int(probs.shape[-1]) != int(cdim):
                    if int(probs.shape[-1]) > int(cdim):
                        probs = probs[..., : int(cdim)]
                    else:
                        probs = torch.nn.functional.pad(probs, (0, int(cdim) - int(probs.shape[-1])))
                return probs, bool(affine_applied)
        except Exception:
            return None, False

    def _apply_pretrain_contact_anchor(
        prob_t: Optional[torch.Tensor],
        h_t: Optional[torch.Tensor],
        prev_prob_t: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        if not torch.is_tensor(prob_t) or prob_t.ndim != 2:
            return prob_t, h_t, prev_prob_t, False
        mdl = getattr(trainer, "contacts_meas_pretrain_anchor", None)
        cfg = getattr(trainer, "contacts_meas_pretrain_anchor_config", None)
        if mdl is None or (not isinstance(cfg, dict)):
            return prob_t, h_t, prev_prob_t, False
        try:
            cur = prob_t
            if torch.is_tensor(prev_prob_t) and prev_prob_t.shape == cur.shape:
                d = (cur - prev_prob_t).to(device=cur.device, dtype=cur.dtype)
            else:
                d = torch.zeros_like(cur)
            try:
                delta_scale = float(cfg.get("delta_scale", 1.0) or 1.0)
            except Exception:
                delta_scale = 1.0
            if not np.isfinite(float(delta_scale)):
                delta_scale = 1.0
            x = torch.cat([cur, d * float(delta_scale)], dim=-1)
            in_dim = int(cfg.get("input_dim", int(x.shape[-1])) or int(x.shape[-1]))
            if int(x.shape[-1]) != int(in_dim):
                if int(x.shape[-1]) > int(in_dim):
                    x = x[..., : int(in_dim)]
                else:
                    x = torch.nn.functional.pad(x, (0, int(in_dim) - int(x.shape[-1])))

            mdl = mdl.to(device=cur.device, dtype=cur.dtype)
            h_in = None
            if torch.is_tensor(h_t) and h_t.ndim == 2 and int(h_t.shape[0]) == int(cur.shape[0]):
                h_in = h_t.to(device=cur.device, dtype=cur.dtype)
            with torch.no_grad():
                logits, h_next = mdl.forward_step(x, h_in)
                if (not torch.is_tensor(logits)) or logits.ndim != 2:
                    return prob_t, h_t, cur.detach(), False
                out = torch.sigmoid(logits)
                cdim = int(cur.shape[-1])
                if int(out.shape[-1]) != cdim:
                    if int(out.shape[-1]) > cdim:
                        out = out[..., :cdim]
                    else:
                        out = torch.nn.functional.pad(out, (0, cdim - int(out.shape[-1])))
                return out, (h_next.detach() if torch.is_tensor(h_next) else None), cur.detach(), True
        except Exception:
            return prob_t, h_t, prob_t.detach() if torch.is_tensor(prob_t) else prev_prob_t, False

    pretrain_anchor_h = None
    pretrain_anchor_prev_prob = None

    for t in range(start_t, end_t):
        # ---- Multi-cycle ablations -------------------------------------------------
        # The tiled multi-cycle sequence introduces a wrap boundary transition between cycles:
        # cond/contacts jump back to frame0 while the autoregressive state carries on.
        # For debugging phase drift, optionally "sync" the autoregressive state to teacher at each cycle start.
        is_cycle_start = bool((rounds > 1) and (int(T_cycle) > 0) and (int(t) > int(start_t)) and ((int(t) % int(T_cycle)) == 0))
        # Ablation: reset pose_hist buffer at each cycle start to avoid cross-cycle carry.
        # This is intentionally scoped to pose_hist_source=buffer to isolate seam mismatch from buffer carry.
        if (
            is_cycle_start
            and bool(multicycle_reset_pose_hist_on_cycle_start)
            and pose_hist_enabled
            and pose_hist_source == "buffer"
            and pose_hist_stride > 0
            and scales is not None
        ):
            try:
                pose_hist_state = _init_eval_pose_hist_state(
                    trainer,
                    ref_tensor=state_seq,
                    pose_hist_seq=pose_hist_seq,
                    step=t,
                    device=device,
                    dtype=state_seq.dtype,
                )
                pose_hist_enabled = pose_hist_state.enabled
                pose_hist_stride = int(pose_hist_state.stride)
                scales = pose_hist_state.scales
                mu = pose_hist_state.mu
                std = pose_hist_state.std
            except Exception:
                pass
        if is_cycle_start and bool(multicycle_sync_state_on_cycle_start):
            try:
                motion = state_seq[:, t].detach()
            except Exception:
                pass
            if getattr(trainer, "normalizer", None) is not None:
                try:
                    motion_raw = trainer.normalizer.denorm_x(motion)
                except Exception:
                    motion_raw = None
            try:
                y_raw_prev = trainer._denorm(gt_seq[:, t])
            except Exception:
                y_raw_prev = None
            if pose_hist_enabled and pose_hist_stride > 0 and scales is not None:
                try:
                    pose_hist_state = _init_eval_pose_hist_state(
                        trainer,
                        ref_tensor=state_seq,
                        pose_hist_seq=pose_hist_seq,
                        step=t,
                        device=device,
                        dtype=state_seq.dtype,
                    )
                    pose_hist_enabled = pose_hist_state.enabled
                    pose_hist_stride = int(pose_hist_state.stride)
                    scales = pose_hist_state.scales
                    mu = pose_hist_state.mu
                    std = pose_hist_state.std
                except Exception:
                    pass
            # White-box contact meas caches per-foot positions across steps; reset it when we teleport state.
            prev_foot_pos_meas = None
        if is_cycle_start and bool(multicycle_reset_plan_z_on_cycle_start) and bool(plan_enable):
            plan_z = None
            phase_z = None
            phase_event_age = None
        # Per-step TTC signals (used only when phase_reset_source uses TTC).
        ttc_gt_step: Optional[torch.Tensor] = None        # (B,C)
        ttc_gt_valid_step: Optional[torch.Tensor] = None  # (B,C) bool
        ttc_state_step: Optional[torch.Tensor] = None     # (B,C)
        ttc_event_step: Optional[torch.Tensor] = None     # (B,C) bool
        try:
            if phase_reset_source in ("ttc_gt", "ttc") and ttc_gt_full is not None and ttc_gt_full.dim() == 3:
                idx_t = min(int(ttc_gt_full.shape[1]) - 1, int(t))
                ttc_gt_step = ttc_gt_full[:, idx_t]
                ttc_state_step = ttc_gt_step
                if ttc_gt_valid_full is not None and ttc_gt_valid_full.dim() == 3:
                    ttc_gt_valid_step = ttc_gt_valid_full[:, idx_t]
                if ttc_gt_event_full is not None and ttc_gt_event_full.dim() == 3:
                    ttc_event_step = ttc_gt_event_full[:, idx_t]
                elif ttc_gt_valid_step is not None and torch.is_tensor(ttc_gt_step):
                    # Fallback: touchdown is defined as TTC==0 (no extra inference knobs).
                    ttc_event_step = (ttc_gt_step <= 0.0) & ttc_gt_valid_step
        except Exception:
            ttc_gt_step = None
            ttc_gt_valid_step = None
            ttc_state_step = None
            ttc_event_step = None

        cond_input = cond_seq[:, t] if (cond_seq is not None and cond_seq.dim() == 3) else cond_seq
        if getattr(trainer, "use_freerun_state_sync", False) and isinstance(getattr(trainer, "angvel_x_slice", None), slice):
            angvel_t = motion[..., trainer.angvel_x_slice].detach()
        else:
            angvel_t = angvel_seq[:, t] if (angvel_seq is not None and angvel_seq.dim() == 3) else angvel_seq
        if pose_hist_enabled:
            pose_hist_t = _resolve_eval_pose_hist_input(
                state=pose_hist_state,
                pose_hist_seq=pose_hist_seq,
                idx=t,
                source=pose_hist_source,
                batch_size=B,
                device=device,
                dtype=state_seq.dtype,
            )
        else:
            pose_hist_t = resolve_pose_hist_input(
                state=PoseHistState(
                    enabled=False,
                    length=pose_hist_len,
                    dim=pose_hist_dim,
                    stride=pose_hist_stride,
                ),
                pose_hist_seq=pose_hist_seq,
                idx=t,
            )
        gt_motion_next = state_seq[:, t + 1]
        if gt_motion_raw is not None:
            try:
                gt_motion_raw = trainer.normalizer.denorm_x(gt_motion_next, prev_raw=gt_motion_raw)
            except Exception:
                gt_motion_raw = None

        cond_raw_step = None
        if cond_seq_raw is not None:
            if cond_seq_raw.dim() == 3:
                idx = min(cond_seq_raw.shape[1] - 1, t + 1)
                cond_raw_step = cond_seq_raw[:, idx]
            else:
                cond_raw_step = cond_seq_raw

        cond_raw_for_model = cond_raw_step
        # Align condition to model's local yaw (mirrors Trainer._rollout_sequence free-run path)
        enable_reproj = (cond_reprojection != "off")
        if cond_reprojection == "auto":
            try:
                enable_reproj = bool(getattr(trainer, "enable_cond_reprojection", True))
                yaw_strategy = str(getattr(trainer, "freerun_yaw_strategy", "trajectory") or "trajectory")
                if yaw_strategy == "trajectory":
                    enable_reproj = False
            except Exception:
                enable_reproj = False
        if enable_reproj and cond_raw_for_model is not None and t > 0:
            yaw_gt = None
            if gt_seq is not None and gt_seq.dim() == 3:
                gt_idx = min(gt_seq.shape[1] - 1, t)
                gt_raw_frame = trainer._denorm(gt_seq[:, gt_idx])
                yaw_gt = trainer._infer_root_yaw_from_rot6d(gt_raw_frame)
            pred_yaw = trainer._infer_root_yaw_from_rot6d(y_raw_prev) if y_raw_prev is not None else None
            if yaw_gt is not None and pred_yaw is not None:
                reproj = trainer._reproject_cond_to_local_frame(cond_raw_for_model, yaw_gt, pred_yaw)
                if reproj is not None:
                    cond_raw_for_model = reproj

        if cond_raw_for_model is not None:
            cond_override = trainer._normalize_cond_from_raw(cond_raw_for_model, cond_norm_mu, cond_norm_std)
            if cond_override is not None:
                cond_input = cond_override

        dev_type = getattr(device, "type", "cpu")
        if dev_type == "mps":
            amp_ctx = torch.autocast(device_type="mps", dtype=torch.float16, enabled=getattr(trainer, "use_amp", False))
        elif dev_type == "cuda":
            amp_ctx = torch.amp.autocast("cuda", enabled=getattr(trainer, "use_amp", False))
        else:
            from contextlib import nullcontext

            amp_ctx = nullcontext()

        if time_index_mode == "none":
            time_index_t = None
        elif time_index_mode == "cycle" or (time_index_mode == "auto" and rounds > 1):
            # NOTE: Use per-cycle phase index, NOT the global step index, to avoid
            # cross-cycle drift when time_index_cycle_len != T_cycle (e.g. --time-index-cycle-minus1).
            if int(T_cycle) > 0:
                t_in_cycle = int(t % int(T_cycle))
            else:
                t_in_cycle = int(t)
            if bool(time_index_cycle_minus1) and int(time_index_cycle_len) == (int(T_cycle) - 1):
                # Map the wrap-boundary frame (t_in_cycle==T_cycle-1) to the last transition index
                # instead of 0 to avoid a hard time-PE reset at the seam.
                if t_in_cycle >= int(time_index_cycle_len):
                    t_in_cycle = int(time_index_cycle_len) - 1
            time_index_t = int(t_in_cycle)
        else:
            # global / auto(single-round): keep increasing global step index
            time_index_t = int(t)
        time_index_log.append(int(time_index_t) if time_index_t is not None else None)

        rollout_step_t = None
        try:
            denom = int(end_t - start_t - 1)
            if denom > 0:
                step_norm = float(int(t - start_t)) / float(denom)
            else:
                step_norm = 0.0
            rollout_step_t = torch.full((motion.shape[0], 1, 1), step_norm, device=device, dtype=motion.dtype)
        except Exception:
            rollout_step_t = None

        # Optional: override which contacts signal the *direct* head uses as phase hint.
        # This does NOT affect contacts_err/Event-Clock/λ unless you also override --contacts_meas_source.
        direct_meas_source_eff = str(getattr(trainer, "direct_pose_meas_source", "model") or "model").strip().lower()
        direct_meas_warmup = int(getattr(trainer, "direct_pose_meas_warmup_steps", 0) or 0)
        step_idx = int(t - start_t)
        if direct_meas_warmup > 0 and step_idx >= direct_meas_warmup:
            direct_meas_source_eff = "zero"

        # Optional: override which contacts signal the model uses as contacts_meas (affects contacts_err/Event-Clock/λ).
        use_learned_meas = bool(getattr(model, "contact_meas_enable", False)) and getattr(model, "contact_meas_head", None) is not None
        contacts_meas_source_cfg = str(getattr(trainer, "contacts_meas_source", "model") or "model").strip().lower()
        contacts_meas_source_applied = str(contacts_meas_source_cfg)

        # Compute whitebox contacts only when needed:
        # - direct head explicitly requests whitebox,
        # - plan init_mode is obs-based (t==0 only),
        # - whitebox debug logging enabled.
        init_mode = str(getattr(model, "contact_plan_init_mode", "learnable") or "learnable").strip().lower()
        log_wb = bool(getattr(trainer, "log_contacts_whitebox", False))
        if log_wb:
            try:
                setattr(trainer, "_contact_meas_whitebox_debug", None)
            except Exception:
                pass

        need_wb = (contacts_meas_source_cfg in ("whitebox", "wb")) or (
            bool(plan_enable)
            and (
                (direct_meas_source_eff in ("whitebox", "wb"))
                or (init_mode in ("obs", "learnable+obs") and plan_z is None and step_idx == 0)
                or log_wb
            )
        )
        contacts_wb_t = None
        if need_wb:
            try:
                contacts_wb_t, prev_foot_pos_meas = trainer._contact_meas_whitebox(motion_raw, prev_foot_pos_meas)
            except Exception:
                contacts_wb_t = None

        contacts_in_t = None
        if contacts_meas_source_cfg in ("zero", "ignore", "none"):
            try:
                cdim = int(getattr(model, "contact_dim", 0) or 0)
            except Exception:
                cdim = 0
            if cdim > 0:
                contacts_in_t = motion.new_zeros((motion.shape[0], cdim))
        elif contacts_meas_source_cfg in ("whitebox", "wb"):
            contacts_in_t = contacts_wb_t
            if contacts_in_t is None:
                contacts_meas_source_applied = "whitebox_missing"
        elif contacts_meas_source_cfg in ("pretrain_contact", "pretrain", "frozen_contact"):
            contacts_in_t, pretrain_affine_applied = _predict_pretrain_contacts_from_frozen(motion, pose_hist_t)
            if contacts_in_t is None:
                contacts_meas_source_applied = "pretrain_contact_missing"
                pretrain_anchor_h = None
                pretrain_anchor_prev_prob = None
            elif bool(pretrain_affine_applied):
                contacts_meas_source_applied = "pretrain_contact_affine"
            if contacts_in_t is not None:
                contacts_in_t, pretrain_anchor_h, pretrain_anchor_prev_prob, pretrain_anchor_applied = _apply_pretrain_contact_anchor(
                    contacts_in_t, pretrain_anchor_h, pretrain_anchor_prev_prob
                )
                if bool(pretrain_anchor_applied):
                    if bool(pretrain_affine_applied):
                        contacts_meas_source_applied = "pretrain_contact_anchor_affine"
                    else:
                        contacts_meas_source_applied = "pretrain_contact_anchor"
        elif contacts_meas_source_cfg in ("gt", "teacher"):
            try:
                if torch.is_tensor(contacts_seq) and contacts_seq.dim() == 3 and contacts_seq.shape[0] == motion.shape[0]:
                    idx0 = min(int(contacts_seq.shape[1]) - 1, int(t))
                    contacts_in_t = contacts_seq[:, idx0]
                else:
                    contacts_meas_source_applied = "gt_missing"
            except Exception:
                contacts_in_t = None
                contacts_meas_source_applied = "gt_missing"
        else:
            # Default: model-produced contacts_meas; keep legacy whitebox injection only for:
            # - no learned meas head (fallback)
            # - plan_z0 init when init_mode is obs-based (t==0 only)
            if plan_enable:
                if not use_learned_meas:
                    contacts_in_t = contacts_wb_t
                    contacts_meas_source_applied = "whitebox_fallback"
                elif init_mode in ("obs", "learnable+obs") and plan_z is None and step_idx == 0:
                    contacts_in_t = contacts_wb_t
                    if contacts_in_t is not None:
                        contacts_meas_source_applied = "whitebox_init"

        # Debug-only: override contacts_meas with teacher contacts only on selected sic.
        if (
            cm_gt_override_sics is not None
            and contacts_in_t is None
            and contacts_meas_source_cfg in ("model", "")
            and rounds > 1
            and int(T_cycle) > 0
            and torch.is_tensor(contacts_seq)
            and contacts_seq.dim() == 3
            and int(contacts_seq.shape[0]) == int(motion.shape[0])
        ):
            cyc_i = int(t // int(T_cycle))
            sic_i = int(t % int(T_cycle))
            wrap_i = bool(sic_i == (int(T_cycle) - 1))
            if (cyc_i >= int(cm_gt_override_cycle_gte)) and (not bool(cm_gt_override_drop_wrap) or not wrap_i) and (
                sic_i in cm_gt_override_sics
            ):
                idx0 = min(int(contacts_seq.shape[1]) - 1, int(t))
                try:
                    contacts_in_t = contacts_seq[:, idx0]
                    contacts_meas_source_applied = f"gt_override_sic{int(sic_i)}"
                except Exception:
                    contacts_in_t = None

        # Optional: post-process learned contact_meas_head output and feed it back as an override.
        # This is a debug-only knob to test whether sharper / single-support contacts_meas can
        # reduce sparse heavy-tail spikes without any retraining.
        try:
            cm_scale = float(getattr(trainer, "contacts_meas_model_logit_scale", 1.0) or 1.0)
        except Exception:
            cm_scale = 1.0
        cm_onehot = bool(getattr(trainer, "contacts_meas_model_onehot", False))
        cm_onehot_cond = bool(getattr(trainer, "contacts_meas_model_onehot_conditional", False))
        try:
            cm_onehot_ds_thr = float(getattr(trainer, "contacts_meas_model_onehot_ds_thr", 0.5) or 0.5)
        except Exception:
            cm_onehot_ds_thr = 0.5
        if not np.isfinite(float(cm_onehot_ds_thr)):
            cm_onehot_ds_thr = 0.5
        cm_onehot_ds_thr = float(max(0.0, min(1.0, float(cm_onehot_ds_thr))))
        cm_onehot_eff = bool(cm_onehot or cm_onehot_cond)
        if (
            contacts_in_t is None
            and use_learned_meas
            and contacts_meas_source_cfg in ("model", "")
            and (cm_onehot_eff or abs(float(cm_scale) - 1.0) > 1e-9)
        ):
            try:
                head = getattr(model, "contact_meas_head", None)
                rot_sl = getattr(model, "_contact_meas_state_rot_slice", None)
                av_sl = getattr(model, "_contact_meas_state_angvel_slice", None)
                idx = getattr(model, "_contact_meas_lower_joint_idx", None)
                if head is None:
                    raise RuntimeError("contact_meas_head missing.")
                if not isinstance(rot_sl, slice) or not isinstance(av_sl, slice):
                    raise RuntimeError("contact_meas_head v1 slices not initialized.")
                if idx is None or (not torch.is_tensor(idx)) or int(idx.numel()) <= 0:
                    raise RuntimeError("contact_meas_head v1 lower-body idx not initialized.")

                st = motion
                if st.ndim != 2:
                    st = st.reshape(st.shape[0], -1)
                st = st.unsqueeze(1)  # (B,1,Dx)
                pose_all = st[..., rot_sl]  # (B,1,J*6)
                w_all = st[..., av_sl]      # (B,1,J*3)
                Jp = int(pose_all.shape[-1] // 6)
                Jw = int(w_all.shape[-1] // 3)
                if Jp <= 0 or Jw <= 0:
                    raise RuntimeError("Invalid contact_meas_head input dims (J<=0).")

                idx_dev = idx.to(device=pose_all.device)
                pose_lower = (
                    pose_all.view(pose_all.shape[0], pose_all.shape[1], Jp, 6)
                    .index_select(2, idx_dev)
                    .reshape(pose_all.shape[0], pose_all.shape[1], -1)
                )
                w_lower = (
                    w_all.view(w_all.shape[0], w_all.shape[1], Jw, 3)
                    .index_select(2, idx_dev)
                    .reshape(w_all.shape[0], w_all.shape[1], -1)
                )

                with torch.no_grad():
                    logits = head(pose_lower, w_lower).squeeze(1)  # (B,C)
                if not np.isfinite(float(cm_scale)):
                    cm_scale = 1.0
                logits = logits * float(cm_scale)
                probs = torch.sigmoid(logits)

                if cm_onehot_eff and torch.is_tensor(probs) and probs.ndim == 2 and int(probs.shape[-1]) > 0:
                    # Winner-take-all across channels (diagnostic; enforces single support).
                    arg = probs.argmax(dim=-1, keepdim=True)
                    oh = torch.zeros_like(probs)
                    oh.scatter_(1, arg, 1.0)
                    if cm_onehot_cond:
                        # Treat "double support" as >=2 channels above threshold, and keep the soft probs there.
                        if int(probs.shape[-1]) >= 2 and float(cm_onehot_ds_thr) > 0.0:
                            is_ds = (probs > float(cm_onehot_ds_thr)).sum(dim=-1, keepdim=True) >= 2
                        else:
                            is_ds = torch.zeros((int(probs.shape[0]), 1), dtype=torch.bool, device=probs.device)
                        probs = torch.where(is_ds, probs, oh)
                    else:
                        probs = oh

                contacts_in_t = probs
                tag = []
                if abs(float(cm_scale) - 1.0) > 1e-9:
                    tag.append(f"scale{float(cm_scale):g}")
                if cm_onehot_eff:
                    if cm_onehot_cond:
                        tag.append(f"condonehot_ds{float(cm_onehot_ds_thr):g}")
                    else:
                        tag.append("onehot")
                contacts_meas_source_applied = "model_post_" + "_".join(tag) if tag else "model_post"
            except Exception:
                # Keep baseline path unchanged on any failure.
                pass

        direct_meas_override = None
        if direct_meas_source_eff in ("zero", "ignore", "none"):
            direct_meas_override = "ignore"
        elif direct_meas_source_eff in ("whitebox", "wb"):
            direct_meas_override = contacts_wb_t
        elif direct_meas_source_eff in ("gt", "teacher"):
            try:
                if torch.is_tensor(contacts_seq) and contacts_seq.dim() == 3 and contacts_seq.shape[0] == motion.shape[0]:
                    idx0 = min(int(contacts_seq.shape[1]) - 1, int(t))
                    direct_meas_override = contacts_seq[:, idx0]
            except Exception:
                direct_meas_override = None
        elif direct_meas_source_eff in ("softgt", "soft_gt", "soft-gt"):
            try:
                if torch.is_tensor(contacts_seq) and contacts_seq.dim() == 3 and contacts_seq.shape[0] == motion.shape[0]:
                    idx0 = min(int(contacts_seq.shape[1]) - 1, int(t))
                    gt_c = contacts_seq[:, idx0]
                    mapped = _softgt_map(gt_c, kind="meas")
                    direct_meas_override = mapped if torch.is_tensor(mapped) else gt_c
            except Exception:
                direct_meas_override = None
        else:
            direct_meas_override = None
        try:
            setattr(model, "direct_pose_meas_override", direct_meas_override)
        except Exception:
            pass

        # Optional: override which contacts *plan* the direct head uses (ablation: direct upper bound).
        # This does NOT affect contacts_plan/contacts_err/lambda (only direct hint).
        direct_plan_source_eff = str(getattr(trainer, "direct_pose_plan_source", "model") or "model").strip().lower()
        direct_plan_override = None
        if direct_plan_source_eff in ("zero", "ignore", "none"):
            direct_plan_override = "ignore"
        elif direct_plan_source_eff in ("gt", "teacher"):
            try:
                if torch.is_tensor(contacts_seq) and contacts_seq.dim() == 3 and contacts_seq.shape[0] == motion.shape[0]:
                    idx0 = min(int(contacts_seq.shape[1]) - 1, int(t))
                    direct_plan_override = contacts_seq[:, idx0]
            except Exception:
                direct_plan_override = None
        elif direct_plan_source_eff in ("softgt", "soft_gt", "soft-gt"):
            try:
                if torch.is_tensor(contacts_seq) and contacts_seq.dim() == 3 and contacts_seq.shape[0] == motion.shape[0]:
                    idx0 = min(int(contacts_seq.shape[1]) - 1, int(t))
                    gt_c = contacts_seq[:, idx0]
                    mapped = _softgt_map(gt_c, kind="plan")
                    direct_plan_override = mapped if torch.is_tensor(mapped) else gt_c
            except Exception:
                direct_plan_override = None
        else:
            direct_plan_override = None
        try:
            setattr(model, "direct_pose_plan_override", direct_plan_override)
        except Exception:
            pass

        # Event-Clock driver state uses the previous probability signal from external contacts.
        meas_prev_in = meas_prev_prob

        # Snapshot per-step inputs/states for optional finite-difference probes (keep baseline path unchanged).
        plan_z_in = plan_z.detach() if torch.is_tensor(plan_z) else plan_z
        phase_z_in = phase_z.detach() if torch.is_tensor(phase_z) else phase_z
        phase_event_age_in = phase_event_age.detach() if torch.is_tensor(phase_event_age) else phase_event_age
        meas_prev_in_in = meas_prev_in.detach() if torch.is_tensor(meas_prev_in) else meas_prev_in
        motion_raw_in = motion_raw.detach() if torch.is_tensor(motion_raw) else None
        y_raw_prev_in = y_raw_prev.detach() if torch.is_tensor(y_raw_prev) else None
        pose_hist_buffer_raw_in = (
            pose_hist_state.buffer_raw.detach()
            if torch.is_tensor(pose_hist_state.buffer_raw)
            else None
        )

        if bool(export_plan_state_series):
            def _mean_vec(x: Any) -> Optional[List[float]]:
                if not torch.is_tensor(x):
                    return None
                v = x.detach()
                # Accept (B,D), (D,), or other shapes (flatten).
                if v.ndim == 2:
                    v = v.mean(dim=0)
                elif v.ndim > 2:
                    v = v.reshape(-1)
                try:
                    return [float(t) for t in v.cpu().tolist()]
                except Exception:
                    return None

            plan_z_in_log.append(_mean_vec(plan_z_in))
            phase_z_in_log.append(_mean_vec(phase_z_in))
            phase_event_age_in_log.append(_mean_vec(phase_event_age_in))

        # Optional: cross-leg ablation on phase_z (evaluation-time only).
        phase_z_eff = phase_z
        try:
            ab = str(getattr(args, "phase_z_ablate", "none") or "none").strip().lower()
        except Exception:
            ab = "none"
        if ab not in ("", "none", "off", "disable", "disabled") and torch.is_tensor(phase_z):
            try:
                C = int(getattr(model, "contact_dim", 0) or 0)
            except Exception:
                C = 0
            try:
                pz = phase_z.to(device=device, dtype=dtype)
                if pz.ndim == 3 and pz.size(1) == 1:
                    pz = pz[:, 0]
                # phase_z is expected to be (B, 2*C): [sinφ0,cosφ0,sinφ1,cosφ1,...]
                if pz.ndim == 2 and C > 0 and int(pz.shape[-1]) == int(2 * C):
                    if C >= 2:
                        pz = pz.clone()
                        if ab == "zero_ch0":
                            pz[..., 0:2] = 0.0
                        elif ab == "zero_ch1":
                            pz[..., 2:4] = 0.0
                        elif ab == "swap_ch01":
                            a = pz[..., 0:2].clone()
                            pz[..., 0:2] = pz[..., 2:4]
                            pz[..., 2:4] = a
                    phase_z_eff = pz
            except Exception:
                phase_z_eff = phase_z

        if bool(direct_leg_head_io_enabled):
            _leg_head_io_cur_t = int(t)
            _leg_head_io_side_call = 0
        if bool(direct_nonleg_probe_enabled):
            _direct_nonleg_probe_cur_t = int(t)
            _direct_nonleg_probe_active = True
        if bool(direct_arm_probe_enabled):
            _direct_arm_probe_cur_t = int(t)
            _direct_arm_probe_active = True

        with amp_ctx:
            ret = model(
                motion,
                cond_input,
                contacts=contacts_in_t,
                angvel=angvel_t,
                pose_history=pose_hist_t,
                plan_z=plan_z,
                phase_z=phase_z_eff,
                phase_event_age=phase_event_age,
                meas_logits_prev=meas_prev_in,
                time_index=time_index_t,
                rollout_step=rollout_step_t,
            )
        if bool(direct_nonleg_probe_enabled):
            _direct_nonleg_probe_active = False
        if bool(direct_arm_probe_enabled):
            _direct_arm_probe_active = False

        if not isinstance(ret, dict):
            raise RuntimeError("Model forward must return a dict with at least 'out'.")
        out = ret.get("out")
        if out is None:
            break

        # Optional: direct pose head output (absolute y_norm; does NOT use y_{t-1}).
        direct_norm_step = None
        try:
            direct_out = ret.get("out_direct", None)
            if torch.is_tensor(direct_out):
                if direct_out.dim() == 3 and direct_out.size(1) == 1:
                    direct_out = direct_out[:, 0]
                if direct_out.dim() == 2 and direct_out.shape[0] == motion.shape[0]:
                    direct_norm_step = direct_out
        except Exception:
            direct_norm_step = None
        # Optional: leg-specific residual omega (axis-angle, rad) for the direct branch.
        direct_leg_omega_step = None
        direct_leg_scale_step = None
        direct_leg_scale_log_step = None
        direct_leg_scale_log_raw_step = None
        try:
            omega_out = ret.get("direct_leg_omega", None)
            if torch.is_tensor(omega_out):
                if omega_out.dim() == 4 and omega_out.size(1) == 1:
                    omega_out = omega_out[:, 0]
                if omega_out.dim() == 3 and omega_out.shape[0] == motion.shape[0]:
                    direct_leg_omega_step = omega_out
        except Exception:
            direct_leg_omega_step = None
        try:
            scale_out = ret.get("direct_leg_scale", None)
            if torch.is_tensor(scale_out):
                if scale_out.dim() == 3 and scale_out.size(1) == 1:
                    scale_out = scale_out[:, 0]
                if scale_out.dim() == 2 and scale_out.shape[0] == motion.shape[0]:
                    direct_leg_scale_step = scale_out
        except Exception:
            direct_leg_scale_step = None
        try:
            scale_out = ret.get("direct_leg_scale_log", None)
            if torch.is_tensor(scale_out):
                if scale_out.dim() == 3 and scale_out.size(1) == 1:
                    scale_out = scale_out[:, 0]
                if scale_out.dim() == 2 and scale_out.shape[0] == motion.shape[0]:
                    direct_leg_scale_log_step = scale_out
        except Exception:
            direct_leg_scale_log_step = None
        try:
            scale_out = ret.get("direct_leg_scale_log_raw", None)
            if torch.is_tensor(scale_out):
                if scale_out.dim() == 3 and scale_out.size(1) == 1:
                    scale_out = scale_out[:, 0]
                if scale_out.dim() == 2 and scale_out.shape[0] == motion.shape[0]:
                    direct_leg_scale_log_raw_step = scale_out
        except Exception:
            direct_leg_scale_log_raw_step = None
        direct_hinge_step = None
        direct_hinge_raw_step = None
        direct_hinge_base_raw_step = None
        direct_hinge_eps_raw_step = None
        direct_hinge_gate_step = None
        try:
            hinge_out = ret.get("direct_hinge_delta", None)
            if torch.is_tensor(hinge_out):
                if hinge_out.dim() == 3 and hinge_out.size(1) == 1:
                    hinge_out = hinge_out[:, 0]
                if hinge_out.dim() == 2 and hinge_out.shape[0] == motion.shape[0]:
                    direct_hinge_step = hinge_out
        except Exception:
            direct_hinge_step = None
        try:
            hinge_out = ret.get("direct_hinge_delta_raw", None)
            if torch.is_tensor(hinge_out):
                if hinge_out.dim() == 3 and hinge_out.size(1) == 1:
                    hinge_out = hinge_out[:, 0]
                if hinge_out.dim() == 2 and hinge_out.shape[0] == motion.shape[0]:
                    direct_hinge_raw_step = hinge_out
        except Exception:
            direct_hinge_raw_step = None
        try:
            hinge_out = ret.get("direct_hinge_delta_base_raw", None)
            if torch.is_tensor(hinge_out):
                if hinge_out.dim() == 3 and hinge_out.size(1) == 1:
                    hinge_out = hinge_out[:, 0]
                if hinge_out.dim() == 2 and hinge_out.shape[0] == motion.shape[0]:
                    direct_hinge_base_raw_step = hinge_out
        except Exception:
            direct_hinge_base_raw_step = None
        try:
            hinge_out = ret.get("direct_hinge_delta_eps_raw", None)
            if torch.is_tensor(hinge_out):
                if hinge_out.dim() == 3 and hinge_out.size(1) == 1:
                    hinge_out = hinge_out[:, 0]
                if hinge_out.dim() == 2 and hinge_out.shape[0] == motion.shape[0]:
                    direct_hinge_eps_raw_step = hinge_out
        except Exception:
            direct_hinge_eps_raw_step = None
        try:
            gate_out = ret.get("direct_hinge_gate", None)
            if torch.is_tensor(gate_out):
                if gate_out.dim() == 3 and gate_out.size(1) == 1:
                    gate_out = gate_out[:, 0]
                if gate_out.dim() == 2 and gate_out.shape[0] == motion.shape[0]:
                    direct_hinge_gate_step = gate_out
        except Exception:
            direct_hinge_gate_step = None

        if bool(export_direct_leg_omega_series):
            if torch.is_tensor(direct_leg_omega_step):
                try:
                    v = direct_leg_omega_step.detach()
                    # Expect (B,K,3); mean over batch.
                    if v.ndim == 4 and int(v.shape[1]) == 1:
                        v = v[:, 0]
                    if v.ndim == 3 and int(v.shape[0]) > 0 and int(v.shape[-1]) == 3:
                        v = v.mean(dim=0)
                    if v.ndim == 2 and int(v.shape[-1]) == 3:
                        direct_leg_omega_step_log.append(
                            [[float(x) for x in row] for row in v.cpu().tolist()]
                        )
                    else:
                        direct_leg_omega_step_log.append(None)
                except Exception:
                    direct_leg_omega_step_log.append(None)
            else:
                direct_leg_omega_step_log.append(None)

            if torch.is_tensor(direct_leg_scale_step):
                try:
                    v = direct_leg_scale_step.detach()
                    if v.ndim == 3 and int(v.shape[1]) == 1:
                        v = v[:, 0]
                    if v.ndim == 2 and int(v.shape[0]) > 0:
                        v = v.mean(dim=0)
                    if v.ndim == 1:
                        direct_leg_scale_step_log.append([float(x) for x in v.cpu().tolist()])
                    else:
                        direct_leg_scale_step_log.append(None)
                except Exception:
                    direct_leg_scale_step_log.append(None)
            else:
                direct_leg_scale_step_log.append(None)

            if torch.is_tensor(direct_leg_scale_log_step):
                try:
                    v = direct_leg_scale_log_step.detach()
                    if v.ndim == 3 and int(v.shape[1]) == 1:
                        v = v[:, 0]
                    if v.ndim == 2 and int(v.shape[0]) > 0:
                        v = v.mean(dim=0)
                    if v.ndim == 1:
                        direct_leg_scale_log_step_log.append([float(x) for x in v.cpu().tolist()])
                    else:
                        direct_leg_scale_log_step_log.append(None)
                except Exception:
                    direct_leg_scale_log_step_log.append(None)
            else:
                direct_leg_scale_log_step_log.append(None)

            if torch.is_tensor(direct_leg_scale_log_raw_step):
                try:
                    v = direct_leg_scale_log_raw_step.detach()
                    if v.ndim == 3 and int(v.shape[1]) == 1:
                        v = v[:, 0]
                    if v.ndim == 2 and int(v.shape[0]) > 0:
                        v = v.mean(dim=0)
                    if v.ndim == 1:
                        direct_leg_scale_log_raw_step_log.append([float(x) for x in v.cpu().tolist()])
                    else:
                        direct_leg_scale_log_raw_step_log.append(None)
                except Exception:
                    direct_leg_scale_log_raw_step_log.append(None)
            else:
                direct_leg_scale_log_raw_step_log.append(None)

        # Diagnostics: compute an axis-oracle hinge delta from GT and use it instead of the hinge head output.
        # This intentionally uses GT (information leak) and is ONLY for debugging apply/target consistency.
        if bool(direct_pose_hinge_oracle_delta) and torch.is_tensor(direct_norm_step):
            try:
                import math

                hinge_idx = getattr(trainer, "direct_pose_hinge_joint_idx", None)
                if isinstance(hinge_idx, (list, tuple)) and hinge_idx:
                    hinge_idx = [int(i) for i in hinge_idx]
                else:
                    hinge_idx = []

                rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
                if isinstance(rot_slice, slice) and hinge_idx and gt_seq is not None and gt_seq.dim() == 3:
                    with torch.no_grad():
                        gt_norm_step = gt_seq[:, t]
                        direct_raw_base = trainer._denorm(direct_norm_step)
                        gt_raw_step = trainer._denorm(gt_norm_step)

                        rot_len = int(rot_slice.stop - rot_slice.start)
                        if rot_len > 0 and (rot_len % 6) == 0:
                            J_full = int(rot_len // 6)
                            if max(hinge_idx) < int(J_full):
                                base6 = reproject_rot6d(direct_raw_base[..., rot_slice]).view(B, J_full, 6)
                                gt6 = reproject_rot6d(gt_raw_step[..., rot_slice]).view(B, J_full, 6)
                                R_base = rot6d_to_matrix(base6, columns=cols)
                                R_gt = rot6d_to_matrix(gt6, columns=cols)
                                # R_gt ≈ R_base @ R_err  =>  R_err = R_base^T @ R_gt
                                R_err = torch.matmul(R_base.transpose(-1, -2), R_gt)
                                R_h = R_err[:, hinge_idx]  # (B,K,3,3)

                                axis = str(getattr(trainer, "direct_pose_hinge_axis", "Z") or "Z").strip().upper()
                                axis_i = {"X": 0, "Y": 1, "Z": 2}.get(axis, 2)
                                if int(axis_i) == 0:  # X
                                    delta_tgt = torch.atan2(
                                        R_h[..., 2, 1] - R_h[..., 1, 2],
                                        R_h[..., 1, 1] + R_h[..., 2, 2],
                                    )
                                elif int(axis_i) == 1:  # Y
                                    delta_tgt = torch.atan2(
                                        R_h[..., 0, 2] - R_h[..., 2, 0],
                                        R_h[..., 0, 0] + R_h[..., 2, 2],
                                    )
                                else:  # Z
                                    delta_tgt = torch.atan2(
                                        R_h[..., 1, 0] - R_h[..., 0, 1],
                                        R_h[..., 0, 0] + R_h[..., 1, 1],
                                    )

                                max_rad = getattr(trainer, "direct_pose_hinge_max_rad", None)
                                try:
                                    max_rad = float(max_rad) if max_rad is not None else None
                                except Exception:
                                    max_rad = None
                                if max_rad is not None and max_rad > 0.0 and math.isfinite(max_rad):
                                    delta_tgt = delta_tgt.clamp(-max_rad, max_rad)

                                direct_hinge_step = delta_tgt
            except Exception:
                # Keep baseline behavior on any failure.
                pass
        if bool(export_direct_hinge_series):
            if torch.is_tensor(direct_hinge_step):
                try:
                    v = direct_hinge_step.detach()
                    if v.ndim == 2:
                        v = v.mean(dim=0)
                    elif v.ndim > 2:
                        v = v.reshape(-1)
                    direct_hinge_step_log.append([float(x) for x in v.cpu().tolist()])
                except Exception:
                    direct_hinge_step_log.append(None)
            else:
                direct_hinge_step_log.append(None)
            if torch.is_tensor(direct_hinge_raw_step):
                try:
                    v = direct_hinge_raw_step.detach()
                    if v.ndim == 2:
                        v = v.mean(dim=0)
                    elif v.ndim > 2:
                        v = v.reshape(-1)
                    direct_hinge_raw_step_log.append([float(x) for x in v.cpu().tolist()])
                except Exception:
                    direct_hinge_raw_step_log.append(None)
            else:
                direct_hinge_raw_step_log.append(None)
            if torch.is_tensor(direct_hinge_base_raw_step):
                try:
                    v = direct_hinge_base_raw_step.detach()
                    if v.ndim == 2:
                        v = v.mean(dim=0)
                    elif v.ndim > 2:
                        v = v.reshape(-1)
                    direct_hinge_base_raw_step_log.append([float(x) for x in v.cpu().tolist()])
                except Exception:
                    direct_hinge_base_raw_step_log.append(None)
            else:
                direct_hinge_base_raw_step_log.append(None)
            if torch.is_tensor(direct_hinge_eps_raw_step):
                try:
                    v = direct_hinge_eps_raw_step.detach()
                    if v.ndim == 2:
                        v = v.mean(dim=0)
                    elif v.ndim > 2:
                        v = v.reshape(-1)
                    direct_hinge_eps_raw_step_log.append([float(x) for x in v.cpu().tolist()])
                except Exception:
                    direct_hinge_eps_raw_step_log.append(None)
            else:
                direct_hinge_eps_raw_step_log.append(None)
            if torch.is_tensor(direct_hinge_gate_step):
                try:
                    v = direct_hinge_gate_step.detach()
                    if v.ndim == 2:
                        v = v.mean(dim=0)
                    elif v.ndim > 2:
                        v = v.reshape(-1)
                    direct_hinge_gate_step_log.append([float(x) for x in v.cpu().tolist()])
                except Exception:
                    direct_hinge_gate_step_log.append(None)
            else:
                direct_hinge_gate_step_log.append(None)
        if torch.is_tensor(direct_norm_step) and torch.is_tensor(direct_hinge_step):
            try:
                direct_norm_step = trainer._apply_direct_hinge_correction_norm(direct_norm_step, direct_hinge_step)
            except Exception:
                pass

        if bool(direct_nonleg_probe_enabled):
            try:
                tt = int(t)
                if (not direct_nonleg_probe_joint_idx_sel) and direct_nonleg_probe_bones_req:
                    cand_names: List[str] = []
                    try:
                        if "rot_gain_bone_names" in locals() and isinstance(rot_gain_bone_names, (list, tuple)):
                            cand_names = [str(x) for x in rot_gain_bone_names]
                    except Exception:
                        cand_names = []
                    if not cand_names:
                        cand_names = list(direct_nonleg_probe_bone_names_full)
                    if not cand_names:
                        try:
                            _bn = getattr(trainer, "_bone_names", None)
                            if isinstance(_bn, (list, tuple)):
                                cand_names = [str(x) for x in _bn]
                        except Exception:
                            cand_names = []
                    if cand_names:
                        if direct_nonleg_probe_all_nonleg:
                            for j, nm in enumerate(cand_names):
                                if int(j) == int(root_idx):
                                    continue
                                if str(nm) in direct_nonleg_probe_leg_set:
                                    continue
                                direct_nonleg_probe_bone_names_sel.append(str(nm))
                                direct_nonleg_probe_joint_idx_sel.append(int(j))
                        else:
                            name_to_idx = {str(n): int(i) for i, n in enumerate(cand_names)}
                            for nm in direct_nonleg_probe_bones_req:
                                if nm in name_to_idx:
                                    j = int(name_to_idx[nm])
                                    if int(j) == int(root_idx):
                                        continue
                                    direct_nonleg_probe_bone_names_sel.append(str(nm))
                                    direct_nonleg_probe_joint_idx_sel.append(int(j))

                if _want_nonleg_probe_t(tt) and torch.is_tensor(direct_norm_step):
                    if isinstance(rot_slice, slice):
                        gt_norm_step = gt_seq[:, tt]
                        gt_raw_step = trainer._denorm(gt_norm_step)
                        direct_raw_step = trainer._denorm(direct_norm_step)

                        B_probe = int(gt_raw_step.shape[0])
                        gt6 = reproject_rot6d(gt_raw_step[..., rot_slice].reshape(B_probe, int(J), 6))
                        dr6 = reproject_rot6d(direct_raw_step[..., rot_slice].reshape(B_probe, int(J), 6))
                        ent = _nonleg_probe_ent(tt)
                        ent_targets = ent.get("targets")
                        if not isinstance(ent_targets, dict):
                            ent_targets = {}
                            ent["targets"] = ent_targets
                        ent_targets["gt_rot6d_all"] = [
                            float(x) for x in gt6.reshape(B_probe, -1).mean(dim=0).detach().cpu().tolist()
                        ]
                        ent_targets["direct_rot6d_all"] = [
                            float(x) for x in dr6.reshape(B_probe, -1).mean(dim=0).detach().cpu().tolist()
                        ]
                        if direct_nonleg_probe_joint_idx_sel:
                            sel = torch.as_tensor(
                                direct_nonleg_probe_joint_idx_sel,
                                device=gt6.device,
                                dtype=torch.long,
                            )
                            gt_sel = gt6.index_select(1, sel).reshape(B_probe, -1).mean(dim=0)
                            dr_sel = dr6.index_select(1, sel).reshape(B_probe, -1).mean(dim=0)
                            ent_targets["gt_rot6d"] = [float(x) for x in gt_sel.detach().cpu().tolist()]
                            ent_targets["direct_rot6d"] = [float(x) for x in dr_sel.detach().cpu().tolist()]
            except Exception:
                pass

        if bool(direct_arm_probe_enabled):
            try:
                tt = int(t)
                if _want_arm_probe_t(tt) and torch.is_tensor(direct_norm_step):
                    if isinstance(rot_slice, slice):
                        gt_norm_step = gt_seq[:, tt]
                        gt_raw_step = trainer._denorm(gt_norm_step)
                        direct_raw_step = trainer._denorm(direct_norm_step)

                        B_probe = int(gt_raw_step.shape[0])
                        gt6 = reproject_rot6d(gt_raw_step[..., rot_slice].reshape(B_probe, int(J), 6))
                        dr6 = reproject_rot6d(direct_raw_step[..., rot_slice].reshape(B_probe, int(J), 6))
                        ent = _arm_probe_ent(tt)
                        ent_targets = ent.get("targets")
                        if not isinstance(ent_targets, dict):
                            ent_targets = {}
                            ent["targets"] = ent_targets
                        ent_targets["gt_rot6d_all"] = [
                            float(x) for x in gt6.reshape(B_probe, -1).mean(dim=0).detach().cpu().tolist()
                        ]
                        ent_targets["direct_rot6d_all"] = [
                            float(x) for x in dr6.reshape(B_probe, -1).mean(dim=0).detach().cpu().tolist()
                        ]
                        if direct_arm_probe_joint_idx_sel:
                            sel = torch.as_tensor(
                                direct_arm_probe_joint_idx_sel,
                                device=gt6.device,
                                dtype=torch.long,
                            )
                            gt_sel = gt6.index_select(1, sel).reshape(B_probe, -1).mean(dim=0)
                            dr_sel = dr6.index_select(1, sel).reshape(B_probe, -1).mean(dim=0)
                            ent_targets["gt_rot6d"] = [float(x) for x in gt_sel.detach().cpu().tolist()]
                            ent_targets["direct_rot6d"] = [float(x) for x in dr_sel.detach().cpu().tolist()]
            except Exception:
                pass
        # Snapshot the direct output (after hinge, before leg omega) for debug-only alpha sweep / oracle alignment.
        if bool(export_direct_leg_omega_alpha_sweep):
            try:
                direct_pre_leg_norm_step_log.append(direct_norm_step.detach() if torch.is_tensor(direct_norm_step) else None)
            except Exception:
                direct_pre_leg_norm_step_log.append(None)
            try:
                direct_leg_omega_tensor_step_log.append(direct_leg_omega_step.detach() if torch.is_tensor(direct_leg_omega_step) else None)
            except Exception:
                direct_leg_omega_tensor_step_log.append(None)

        # Debug-only: contact-plan gating for direct_leg_omega apply (per-side).
        omega_leg_apply = direct_leg_omega_step
        gate_entry = None
        # Debug-only: phase-window conditional sign flip (per-side) for direct_leg_omega.
        flip_entry = None
        if bool(direct_pose_leg_contact_flip):
            try:
                if (
                    torch.is_tensor(omega_leg_apply)
                    and torch.is_tensor(direct_leg_omega_flip_mask_r)
                    and torch.is_tensor(direct_leg_omega_flip_mask_l)
                    and torch.is_tensor(direct_leg_omega_flip_apply_mask)
                ):
                    p_next = ret.get("phase_z_next", None)
                    v = None  # (B,C,2) with last dim=[sin,cos]
                    if torch.is_tensor(p_next):
                        z = p_next
                        try:
                            Cc = int(getattr(model, "contact_dim", 0) or 0)
                        except Exception:
                            Cc = 0
                        if z.ndim == 3 and int(z.size(1)) == 1:
                            z = z[:, 0]
                        if z.ndim == 1 and int(z.numel()) == int(Cc) * 2:
                            z = z.reshape(1, int(Cc) * 2)
                        if Cc > 0:
                            if z.ndim == 2 and int(z.shape[-1]) == int(Cc) * 2:
                                v = z.reshape(int(z.shape[0]), int(Cc), 2)
                            elif z.ndim == 3 and int(z.shape[-2]) == int(Cc) and int(z.shape[-1]) == 2:
                                v = z
                    if v is not None and int(v.shape[0]) == motion.shape[0] and int(v.shape[1]) >= 2:
                        ang = torch.atan2(v[..., 0], v[..., 1])  # (B,C) in [-pi,pi]
                        abs_ang = ang.abs()
                        w_deg = float(direct_pose_leg_contact_flip_phase_window_deg)
                        w_deg = float(max(1e-3, min(180.0, w_deg)))
                        w_rad = w_deg * (float(np.pi) / 180.0)
                        flip_r = abs_ang[:, flip_ch_r] <= w_rad
                        flip_l = abs_ang[:, flip_ch_l] <= w_rad
                        # Optional: also require a contact-plan logits transition spike to reduce false flips.
                        delta_thr = float(direct_pose_leg_contact_flip_delta_thr)
                        delta_thr = float(max(0.0, delta_thr))
                        delta_per_c = None
                        if delta_thr > 0.0:
                            try:
                                logits_cur = ret.get("contacts_plan_logits", None)
                                if torch.is_tensor(logits_cur):
                                    if logits_cur.dim() == 3 and logits_cur.size(1) == 1:
                                        logits_cur = logits_cur[:, 0]
                                    if logits_cur.dim() == 1:
                                        logits_cur = logits_cur.view(logits_cur.shape[0], 1)
                                    if logits_cur.dim() == 2 and logits_cur.shape[0] == motion.shape[0] and logits_cur.shape[1] >= 2:
                                        prev = direct_leg_omega_flip_plan_prev
                                        if torch.is_tensor(prev) and prev.shape == logits_cur.shape:
                                            dlog = (logits_cur - prev).abs()
                                        else:
                                            dlog = torch.zeros_like(logits_cur)
                                        flip_r = flip_r & (dlog[:, flip_ch_r] > delta_thr)
                                        flip_l = flip_l & (dlog[:, flip_ch_l] > delta_thr)
                                        delta_per_c = [
                                            float(dlog[:, flip_ch_r].mean().item()),
                                            float(dlog[:, flip_ch_l].mean().item()),
                                        ]
                                        direct_leg_omega_flip_plan_prev = logits_cur.detach()
                            except Exception:
                                delta_per_c = None
                        # Per-side sign: -1 inside window, +1 otherwise.
                        sr = torch.where(flip_r, -torch.ones_like(abs_ang[:, flip_ch_r]), torch.ones_like(abs_ang[:, flip_ch_r]))
                        sl = torch.where(flip_l, -torch.ones_like(abs_ang[:, flip_ch_l]), torch.ones_like(abs_ang[:, flip_ch_l]))
                        sr = sr.to(dtype=omega_leg_apply.dtype).view(-1, 1, 1)
                        sl = sl.to(dtype=omega_leg_apply.dtype).view(-1, 1, 1)
                        other = (1.0 - direct_leg_omega_flip_mask_r - direct_leg_omega_flip_mask_l).clamp(0.0, 1.0)
                        sign_joint = direct_leg_omega_flip_mask_r * sr + direct_leg_omega_flip_mask_l * sl + other
                        # Only flip selected joints; others are forced to +1.
                        sign_joint = direct_leg_omega_flip_apply_mask * sign_joint + (1.0 - direct_leg_omega_flip_apply_mask)
                        if omega_leg_apply.dim() == 4 and omega_leg_apply.size(1) == 1:
                            omega_leg_apply = omega_leg_apply * sign_joint.unsqueeze(1)
                        elif omega_leg_apply.dim() == 3:
                            omega_leg_apply = omega_leg_apply * sign_joint

                        flip_entry = {
                            "DirectLegOmegaFlipSource": "phase_z_next",
                            "DirectLegOmegaFlipOrder": str(flip_order),
                            "DirectLegOmegaFlipWindowDeg": float(w_deg),
                            "DirectLegOmegaFlipDeltaThr": float(delta_thr),
                            "DirectLegOmegaFlipDeltaLogitPerC": delta_per_c,
                            "DirectLegOmegaFlipJoints": str(direct_pose_leg_contact_flip_joints or "foot_r,foot_l"),
                            "DirectLegOmegaFlipApplyFrac": float(direct_leg_omega_flip_apply_mask.mean().item())
                            if torch.is_tensor(direct_leg_omega_flip_apply_mask)
                            else None,
                            "DirectLegOmegaFlipCondPerC": [
                                float(flip_r.to(dtype=omega_leg_apply.dtype).mean().item()),
                                float(flip_l.to(dtype=omega_leg_apply.dtype).mean().item()),
                            ],
                            "DirectLegOmegaFlipPhaseAbsDegPerC": [
                                float(abs_ang[:, flip_ch_r].mean().item() * (180.0 / float(np.pi))),
                                float(abs_ang[:, flip_ch_l].mean().item() * (180.0 / float(np.pi))),
                            ],
                        }
            except Exception:
                flip_entry = None
            direct_leg_omega_flip_step_log.append(flip_entry)
        if bool(direct_pose_leg_contact_gate):
            try:
                gmin = float(direct_pose_leg_contact_gate_min)
                gmin = float(max(0.0, min(1.0, gmin)))

                if gate_mode == "phase":
                    # Gate by phase angle proximity to 0 using phase_z_next (sin/cos per contact channel).
                    p_next = ret.get("phase_z_next", None)
                    v = None  # (B,C,2) with last dim=[sin,cos]
                    if torch.is_tensor(p_next):
                        z = p_next
                        try:
                            Cc = int(getattr(model, "contact_dim", 0) or 0)
                        except Exception:
                            Cc = 0
                        if z.ndim == 3 and int(z.size(1)) == 1:
                            z = z[:, 0]
                        if z.ndim == 1 and int(z.numel()) == int(Cc) * 2:
                            z = z.reshape(1, int(Cc) * 2)
                        if Cc > 0:
                            if z.ndim == 2 and int(z.shape[-1]) == int(Cc) * 2:
                                v = z.reshape(int(z.shape[0]), int(Cc), 2)
                            elif z.ndim == 3 and int(z.shape[-2]) == int(Cc) and int(z.shape[-1]) == 2:
                                v = z
                    if v is not None and int(v.shape[0]) == motion.shape[0] and int(v.shape[1]) >= 2:
                        ang = torch.atan2(v[..., 0], v[..., 1])  # (B,C) in [-pi,pi]
                        abs_ang = ang.abs()
                        w_deg = float(direct_pose_leg_contact_gate_phase_window_deg)
                        w_deg = float(max(1e-3, min(180.0, w_deg)))
                        w_rad = w_deg * (float(np.pi) / 180.0)
                        g_raw = (abs_ang / w_rad).clamp(0.0, 1.0)
                        g_r = (gmin + (1.0 - gmin) * g_raw[:, gate_ch_r]).clamp(0.0, 1.0)
                        g_l = (gmin + (1.0 - gmin) * g_raw[:, gate_ch_l]).clamp(0.0, 1.0)

                        if (
                            torch.is_tensor(omega_leg_apply)
                            and torch.is_tensor(direct_leg_omega_gate_mask_r)
                            and torch.is_tensor(direct_leg_omega_gate_mask_l)
                        ):
                            gr = g_r.view(-1, 1, 1)
                            gl = g_l.view(-1, 1, 1)
                            other = (1.0 - direct_leg_omega_gate_mask_r - direct_leg_omega_gate_mask_l).clamp(0.0, 1.0)
                            g_joint = direct_leg_omega_gate_mask_r * gr + direct_leg_omega_gate_mask_l * gl + other
                            if torch.is_tensor(direct_leg_omega_gate_apply_mask):
                                g_joint = direct_leg_omega_gate_apply_mask * g_joint + (1.0 - direct_leg_omega_gate_apply_mask)
                            if omega_leg_apply.dim() == 4 and omega_leg_apply.size(1) == 1:
                                omega_leg_apply = omega_leg_apply * g_joint.unsqueeze(1)
                            elif omega_leg_apply.dim() == 3:
                                omega_leg_apply = omega_leg_apply * g_joint

                        gate_entry = {
                            "DirectLegOmegaGateSource": "contacts_plan",
                            "DirectLegOmegaGateMode": str(gate_mode),
                            "DirectLegOmegaGateOrder": str(gate_order),
                            "DirectLegOmegaGateJoints": str(direct_pose_leg_contact_gate_joints or "all"),
                            "DirectLegOmegaGateApplyFrac": float(direct_leg_omega_gate_apply_mask.mean().item())
                            if torch.is_tensor(direct_leg_omega_gate_apply_mask)
                            else None,
                            "DirectLegOmegaGatePhaseWindowDeg": float(w_deg),
                            "DirectLegOmegaGateMin": float(gmin),
                            "DirectLegOmegaGatePhaseAbsDegPerC": [
                                float(abs_ang[:, gate_ch_r].mean().item() * (180.0 / float(np.pi))),
                                float(abs_ang[:, gate_ch_l].mean().item() * (180.0 / float(np.pi))),
                            ],
                            "DirectLegOmegaGateGPerC": [
                                float(g_r.mean().item()),
                                float(g_l.mean().item()),
                            ],
                        }
                else:
                    # Delta-gating on contacts_plan (prob or logits).
                    plan_key = "contacts_plan_logits" if gate_signal == "logit" else "contacts_plan"
                    plan_cur = ret.get(plan_key, None)
                    if torch.is_tensor(plan_cur):
                        if plan_cur.dim() == 3 and plan_cur.size(1) == 1:
                            plan_cur = plan_cur[:, 0]
                        if plan_cur.dim() == 1:
                            plan_cur = plan_cur.view(plan_cur.shape[0], 1)
                        if plan_cur.dim() == 2 and plan_cur.shape[0] == motion.shape[0] and plan_cur.shape[1] >= 2:
                            plan_prev = direct_leg_omega_plan_prev
                            if torch.is_tensor(plan_prev):
                                if plan_prev.dim() == 3 and plan_prev.size(1) == 1:
                                    plan_prev = plan_prev[:, 0]
                                if plan_prev.dim() == 1:
                                    plan_prev = plan_prev.view(plan_prev.shape[0], 1)
                            if torch.is_tensor(plan_prev) and plan_prev.shape == plan_cur.shape:
                                delta = (plan_cur - plan_prev).abs()
                            else:
                                delta = torch.zeros_like(plan_cur)

                            k = float(direct_pose_leg_contact_gate_k)
                            # g = gmin + (1-gmin)*exp(-k*|dc|)  (dc is per-foot contact-plan delta)
                            g_r = (gmin + (1.0 - gmin) * torch.exp(-k * delta[:, gate_ch_r])).clamp(0.0, 1.0)
                            g_l = (gmin + (1.0 - gmin) * torch.exp(-k * delta[:, gate_ch_l])).clamp(0.0, 1.0)

                            if (
                                torch.is_tensor(omega_leg_apply)
                                and torch.is_tensor(direct_leg_omega_gate_mask_r)
                                and torch.is_tensor(direct_leg_omega_gate_mask_l)
                            ):
                                gr = g_r.view(-1, 1, 1)
                                gl = g_l.view(-1, 1, 1)
                                other = (1.0 - direct_leg_omega_gate_mask_r - direct_leg_omega_gate_mask_l).clamp(0.0, 1.0)
                                g_joint = direct_leg_omega_gate_mask_r * gr + direct_leg_omega_gate_mask_l * gl + other
                                if torch.is_tensor(direct_leg_omega_gate_apply_mask):
                                    g_joint = direct_leg_omega_gate_apply_mask * g_joint + (1.0 - direct_leg_omega_gate_apply_mask)
                                if omega_leg_apply.dim() == 4 and omega_leg_apply.size(1) == 1:
                                    omega_leg_apply = omega_leg_apply * g_joint.unsqueeze(1)
                                elif omega_leg_apply.dim() == 3:
                                    omega_leg_apply = omega_leg_apply * g_joint

                            gate_entry = {
                                "DirectLegOmegaGateSource": "contacts_plan",
                                "DirectLegOmegaGateMode": str(gate_mode),
                                "DirectLegOmegaGateOrder": str(gate_order),
                                "DirectLegOmegaGateJoints": str(direct_pose_leg_contact_gate_joints or "all"),
                                "DirectLegOmegaGateApplyFrac": float(direct_leg_omega_gate_apply_mask.mean().item())
                                if torch.is_tensor(direct_leg_omega_gate_apply_mask)
                                else None,
                                "DirectLegOmegaGateSignal": str(gate_signal),
                                "DirectLegOmegaGateK": float(k),
                                "DirectLegOmegaGateMin": float(gmin),
                                "DirectLegOmegaGateDeltaPerC": [
                                    float(delta[:, gate_ch_r].mean().item()),
                                    float(delta[:, gate_ch_l].mean().item()),
                                ],
                                "DirectLegOmegaGateGPerC": [
                                    float(g_r.mean().item()),
                                    float(g_l.mean().item()),
                                ],
                            }
                            direct_leg_omega_plan_prev = plan_cur.detach()
            except Exception:
                gate_entry = None
            direct_leg_omega_plan_gate_step_log.append(gate_entry)

        if (not bool(direct_pose_leg_noapply)) and torch.is_tensor(direct_norm_step) and torch.is_tensor(direct_leg_omega_step):
            try:
                leg_mode = str(getattr(model, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add").lower().strip()
                if leg_mode == "so3":
                    direct_norm_step = _apply_direct_leg_so3_correction_norm(
                        trainer,
                        model,
                        direct_norm_step,
                        omega_leg_apply,
                        columns=cols,
                        omega_scale=float(direct_pose_leg_apply_scale),
                        omega_sign=float(direct_pose_leg_apply_sign),
                        apply_side=str(direct_pose_leg_apply_side),
                    )
            except Exception:
                pass

        # Debug-only: autograd gradient norms for direct_leg_omega on selected steps.
        if bool(export_direct_leg_omega_grad):
            try:
                # Step selector (mask aligned to evaluation protocol).
                try:
                    T_cycle_i = int(T_cycle)
                except Exception:
                    T_cycle_i = 0
                if int(T_cycle_i) > 0:
                    cyc_i = int(t // int(T_cycle_i))
                    sic_i = int(t % int(T_cycle_i))
                else:
                    cyc_i = 0
                    sic_i = int(t)
                wrap_i = bool((rounds > 1) and (int(T_cycle_i) > 0) and (int(sic_i) == (int(T_cycle_i) - 1)))
                if (int(cyc_i) >= int(grad_cycle_gte)) and (not bool(grad_drop_wrap) or not wrap_i):
                    if direct_leg_omega_grad_sics_set is None or int(sic_i) in direct_leg_omega_grad_sics_set:
                        # Require direct head + omega head to be present.
                        if (
                            torch.is_tensor(direct_norm_step)
                            and torch.is_tensor(direct_leg_omega_step)
                            and bool(getattr(direct_leg_omega_step, "requires_grad", False))
                            and torch.is_tensor(gt_seq)
                            and gt_seq.dim() == 3
                            and int(t) < int(gt_seq.size(1))
                            and isinstance(rot_slice, slice)
                            and int(J) > 0
                            and torch.is_tensor(direct_leg_omega_grad_joint_mask)
                        ):
                            gt_norm_step = gt_seq[:, int(t)]

                            # DirectGeoLocalDeg loss (deg): mean over non-root joints.
                            direct_raw = trainer._denorm(direct_norm_step)
                            gt_raw_step = trainer._denorm(gt_norm_step)
                            rot_len = int(rot_slice.stop - rot_slice.start)
                            if rot_len > 0 and (rot_len % 6) == 0:
                                d6 = reproject_rot6d(direct_raw[..., rot_slice]).view(int(direct_raw.shape[0]), int(J), 6)
                                g6 = reproject_rot6d(gt_raw_step[..., rot_slice]).view(int(gt_raw_step.shape[0]), int(J), 6)
                                Rd = rot6d_to_matrix(d6, columns=cols)
                                Rg = rot6d_to_matrix(g6, columns=cols)
                                dloc = geodesic_R(Rd, Rg) * float(deg_factor)  # (B,J) in deg

                                jm = direct_leg_omega_grad_joint_mask
                                loss = dloc[:, jm].mean() if bool(jm.any().detach().cpu().item()) else dloc.mean()

                                # Compute gradients w.r.t leg-head params and omega output (no side effects on .grad).
                                inputs: List[torch.Tensor] = list(direct_leg_omega_grad_leg_params) + [direct_leg_omega_step]
                                grads = torch.autograd.grad(
                                    loss,
                                    inputs,
                                    allow_unused=True,
                                    retain_graph=False,
                                    create_graph=False,
                                )

                                # Global leg-head param grad norm + per-prefix split.
                                theta_sq = 0.0
                                missing = 0
                                prefix_sq: Dict[str, float] = {}
                                for (pname, _), g in zip(direct_leg_omega_grad_leg_named_params, grads[:-1]):
                                    if g is None:
                                        missing += 1
                                        continue
                                    gg = g.detach()
                                    if not torch.isfinite(gg).all():
                                        gg = torch.nan_to_num(gg, nan=0.0, posinf=0.0, neginf=0.0)
                                    s = float(gg.float().pow(2).sum().item())
                                    theta_sq += s
                                    pref = str(pname).split(".", 1)[0]
                                    prefix_sq[pref] = prefix_sq.get(pref, 0.0) + s
                                theta_norm = float(math.sqrt(max(0.0, theta_sq)))
                                theta_norm_by_prefix = {k: float(math.sqrt(max(0.0, v))) for k, v in prefix_sq.items()}

                                # Omega grad norms per leg joint (mean over batch).
                                g_omega = grads[-1]
                                omega_grad_norm_by_k: Optional[List[float]] = None
                                if g_omega is not None and torch.is_tensor(g_omega):
                                    go = g_omega.detach()
                                    if go.ndim == 4 and int(go.size(1)) == 1:
                                        go = go[:, 0]
                                    if go.ndim == 3 and int(go.shape[-1]) == 3:
                                        if not torch.isfinite(go).all():
                                            go = torch.nan_to_num(go, nan=0.0, posinf=0.0, neginf=0.0)
                                        omega_grad_norm_by_k = (
                                            go.norm(dim=-1).mean(dim=0).to(dtype=torch.float32).cpu().tolist()
                                        )  # (K,)

                                # Per-bone report (subset of leg joints).
                                per_bone: Dict[str, Any] = {}
                                if (
                                    direct_leg_omega_grad_bone_k is not None
                                    and direct_leg_omega_grad_bone_names is not None
                                    and direct_leg_omega_grad_bone_joint_idx is not None
                                ):
                                    for kk, nm, j_idx in zip(
                                        direct_leg_omega_grad_bone_k,
                                        direct_leg_omega_grad_bone_names,
                                        direct_leg_omega_grad_bone_joint_idx,
                                    ):
                                        ent: Dict[str, Any] = {}
                                        if 0 <= int(j_idx) < int(dloc.shape[-1]):
                                            ent["dloc_deg"] = float(dloc[:, int(j_idx)].mean().detach().cpu().item())
                                        else:
                                            ent["dloc_deg"] = None
                                        if omega_grad_norm_by_k is not None and 0 <= int(kk) < int(len(omega_grad_norm_by_k)):
                                            ent["domega_grad_norm"] = float(omega_grad_norm_by_k[int(kk)])
                                        else:
                                            ent["domega_grad_norm"] = None
                                        per_bone[str(nm)] = ent

                                direct_leg_omega_grad_steps.append(
                                    {
                                        "t": int(t),
                                        "cycle": int(cyc_i),
                                        "sic": int(sic_i),
                                        "wrap": bool(wrap_i),
                                        "loss_deg": float(loss.detach().cpu().item()),
                                        "theta_grad_norm": float(theta_norm),
                                        "theta_grad_norm_by_prefix": theta_norm_by_prefix,
                                        "theta_grad_missing_params": int(missing),
                                        "per_bone": per_bone,
                                    }
                                )
            except Exception:
                # Never let diagnostics break rollout.
                pass

        ec_lambda_corr_step = None
        try:
            ec_lam = ret.get("event_clock_lambda_corr", None)
            if torch.is_tensor(ec_lam):
                if ec_lam.dim() == 3 and ec_lam.size(1) == 1:
                    ec_lam = ec_lam[:, 0]
                if ec_lam.dim() == 1 and ec_lam.shape[0] == motion.shape[0]:
                    ec_lam = ec_lam.unsqueeze(-1)
                if ec_lam.dim() == 2 and ec_lam.shape[0] == motion.shape[0] and ec_lam.shape[-1] == 1:
                    ec_lambda_corr_step = ec_lam.clamp(0.0, 1.0)
        except Exception:
            ec_lambda_corr_step = None
        try:
            event_clock_lambda_corr_log.append(
                ec_lambda_corr_step.detach().cpu() if torch.is_tensor(ec_lambda_corr_step) else None
            )
        except Exception:
            event_clock_lambda_corr_log.append(None)

        # Optional: lambda fusion gate (Stage2), normalized to (B,J) for stats / blending.
        lam_step = None
        lam_eff_step = None
        lam_rel_step = None
        lam_stats = None
        try:
            lam = ret.get("lambda_fusion", None)
            if torch.is_tensor(lam):
                if lam.dim() == 3 and lam.size(1) == 1:
                    lam = lam[:, 0]
                if lam.dim() == 1:
                    if lam.shape[0] == motion.shape[0]:
                        lam = lam.unsqueeze(-1)
                    elif motion.shape[0] == 1 and J > 0 and lam.shape[0] == J:
                        lam = lam.unsqueeze(0)
                if lam.dim() == 2 and lam.shape[0] == motion.shape[0]:
                    if lam.shape[-1] == 1 and J > 0:
                        lam = lam.expand(lam.shape[0], J)
                    if J > 0 and lam.shape[-1] == J:
                        lam_step = lam.clamp(0.0, 1.0)
                        lam_cpu = lam_step.detach().cpu()
                        lambda_log.append(lam_cpu)
                        lam_stats = (float(lam_cpu.mean().item()), float(lam_cpu.std(unbiased=False).item()))
                        # Shared reliability r_t -> lambda_eff for actual on-manifold blend.
                        lam_eff_step = lam_step
                        try:
                            lam_eff_step, lam_rel_step = trainer._lambda_fusion_apply_reliability(
                                lam_step,
                                step_idx=int(t - start_t),
                                total_steps=int(max(1, int(end_t - start_t))),
                                rollout_step=rollout_step_t,
                                ret=ret,
                            )
                        except Exception:
                            lam_eff_step, lam_rel_step = lam_step, None
                        try:
                            lambda_eff_log.append(lam_eff_step.detach().cpu() if torch.is_tensor(lam_eff_step) else None)
                        except Exception:
                            lambda_eff_log.append(None)
                        try:
                            lambda_rel_log.append(lam_rel_step.detach().cpu() if torch.is_tensor(lam_rel_step) else None)
                        except Exception:
                            lambda_rel_log.append(None)
        except Exception:
            lam_step = None
            lam_eff_step = None
            lam_rel_step = None
            lam_stats = None
        if lam_step is None:
            lambda_log.append(None)
            lambda_eff_log.append(None)
            lambda_rel_log.append(None)

        gate_override = None
        contact_entry: Optional[Dict[str, Any]] = None
        if bool(getattr(trainer, "log_contacts", False)):
            try:
                # GT soft contacts (from dataset / teacher), aligned to the tiled timeline.
                gt_contacts = None
                gt_contacts_next = None
                if torch.is_tensor(contacts_seq) and contacts_seq.dim() == 3 and contacts_seq.shape[0] == motion.shape[0]:
                    idx0 = min(int(contacts_seq.shape[1]) - 1, int(t))
                    idx1 = min(int(contacts_seq.shape[1]) - 1, int(t) + 1)
                    gt_contacts = contacts_seq[:, idx0]
                    gt_contacts_next = contacts_seq[:, idx1]

                plan = ret.get("contacts_plan", None)
                meas = ret.get("contacts_meas", None)
                err = ret.get("contacts_err", None)
                plan_logits = ret.get("contacts_plan_logits", None)
                plan_logits_base = ret.get("contacts_plan_logits_base", None)
                plan_logits_phase = ret.get("contacts_plan_logits_phase", None)
                plan_logits_time = ret.get("contacts_plan_logits_time", None)
                plan_logits_raw = ret.get("contacts_plan_logits_raw", None)
                meas_logits = ret.get("contacts_meas_logits", None)
                plan_per_c = None
                meas_per_c = None
                err_per_c = None
                plan_logits_mean = None
                plan_logits_std = None
                plan_logits_per_c = None
                plan_logits_base_per_c = None
                plan_logits_phase_per_c = None
                plan_logits_time_per_c = None
                plan_logits_raw_per_c = None
                meas_logits_mean = None
                meas_logits_std = None
                meas_logits_per_c = None
                angvel_mean = None
                angvel_abs_mean = None
                angvel_std = None
                pose_hist_mean = None
                pose_hist_abs_mean = None
                pose_hist_std = None
                plan_lr_absdiff_mean = None
                plan_lr_diff_std = None
                meas_lr_absdiff_mean = None
                meas_lr_diff_std = None
                gt_lr_absdiff_mean = None
                gt_lr_diff_std = None
                gt_next_lr_absdiff_mean = None
                gt_next_lr_diff_std = None
                if torch.is_tensor(plan) and plan.ndim == 2:
                    plan_mean = float(plan.mean().item())
                    plan_abs_mean = float(plan.abs().mean().item())
                    try:
                        plan_per_c = plan.mean(dim=0).detach().cpu().tolist()
                    except Exception:
                        plan_per_c = None
                    try:
                        if plan.shape[-1] >= 2:
                            d = plan[:, 0] - plan[:, 1]
                            plan_lr_absdiff_mean = float(d.abs().mean().item())
                            plan_lr_diff_std = float(d.std(unbiased=False).item())
                    except Exception:
                        plan_lr_absdiff_mean = None
                        plan_lr_diff_std = None
                else:
                    plan_mean = None
                    plan_abs_mean = None
                if torch.is_tensor(meas) and meas.ndim == 2:
                    meas_mean = float(meas.mean().item())
                    meas_abs_mean = float(meas.abs().mean().item())
                    try:
                        meas_per_c = meas.mean(dim=0).detach().cpu().tolist()
                    except Exception:
                        meas_per_c = None
                    try:
                        if meas.shape[-1] >= 2:
                            d = meas[:, 0] - meas[:, 1]
                            meas_lr_absdiff_mean = float(d.abs().mean().item())
                            meas_lr_diff_std = float(d.std(unbiased=False).item())
                    except Exception:
                        meas_lr_absdiff_mean = None
                        meas_lr_diff_std = None
                else:
                    meas_mean = None
                    meas_abs_mean = None
                if torch.is_tensor(plan_logits) and plan_logits.ndim == 2:
                    try:
                        plan_logits_mean = float(plan_logits.mean().item())
                        plan_logits_std = float(plan_logits.std(unbiased=False).item())
                        plan_logits_per_c = plan_logits.mean(dim=0).detach().cpu().tolist()
                    except Exception:
                        plan_logits_mean = None
                        plan_logits_std = None
                        plan_logits_per_c = None
                if torch.is_tensor(plan_logits_base) and plan_logits_base.ndim == 2:
                    try:
                        plan_logits_base_per_c = plan_logits_base.mean(dim=0).detach().cpu().tolist()
                    except Exception:
                        plan_logits_base_per_c = None
                if torch.is_tensor(plan_logits_phase) and plan_logits_phase.ndim == 2:
                    try:
                        plan_logits_phase_per_c = plan_logits_phase.mean(dim=0).detach().cpu().tolist()
                    except Exception:
                        plan_logits_phase_per_c = None
                if torch.is_tensor(plan_logits_time) and plan_logits_time.ndim == 2:
                    try:
                        plan_logits_time_per_c = plan_logits_time.mean(dim=0).detach().cpu().tolist()
                    except Exception:
                        plan_logits_time_per_c = None
                if torch.is_tensor(plan_logits_raw) and plan_logits_raw.ndim == 2:
                    try:
                        plan_logits_raw_per_c = plan_logits_raw.mean(dim=0).detach().cpu().tolist()
                    except Exception:
                        plan_logits_raw_per_c = None
                if torch.is_tensor(meas_logits) and meas_logits.ndim == 2:
                    try:
                        meas_logits_mean = float(meas_logits.mean().item())
                        meas_logits_std = float(meas_logits.std(unbiased=False).item())
                        meas_logits_per_c = meas_logits.mean(dim=0).detach().cpu().tolist()
                    except Exception:
                        meas_logits_mean = None
                        meas_logits_std = None
                        meas_logits_per_c = None
                try:
                    av = angvel_t
                    if torch.is_tensor(av):
                        if av.ndim == 3 and av.size(1) == 1:
                            av = av[:, 0]
                        if av.ndim == 2:
                            angvel_mean = float(av.mean().item())
                            angvel_abs_mean = float(av.abs().mean().item())
                            angvel_std = float(av.std(unbiased=False).item())
                except Exception:
                    angvel_mean = None
                    angvel_abs_mean = None
                    angvel_std = None
                try:
                    ph = pose_hist_t
                    if torch.is_tensor(ph):
                        if ph.ndim == 3 and ph.size(1) == 1:
                            ph = ph[:, 0]
                        if ph.ndim == 2:
                            pose_hist_mean = float(ph.mean().item())
                            pose_hist_abs_mean = float(ph.abs().mean().item())
                            pose_hist_std = float(ph.std(unbiased=False).item())
                except Exception:
                    pose_hist_mean = None
                    pose_hist_abs_mean = None
                    pose_hist_std = None
                err_abs_mean = None
                if torch.is_tensor(err) and err.ndim == 2:
                    err_abs = err.abs()
                    err_abs_mean = float(err_abs.mean().item())
                    # Also keep per-channel mean abs error (small C, debug-friendly).
                    err_abs_per_c = err_abs.mean(dim=0).detach().cpu().tolist()
                    try:
                        err_per_c = err.mean(dim=0).detach().cpu().tolist()
                    except Exception:
                        err_per_c = None
                else:
                    err_abs_per_c = None
                gt_mean = None
                gt_abs_mean = None
                gt_per_c = None
                gt_next_mean = None
                gt_next_abs_mean = None
                gt_next_per_c = None
                if torch.is_tensor(gt_contacts) and gt_contacts.ndim == 2:
                    gt_mean = float(gt_contacts.mean().item())
                    gt_abs_mean = float(gt_contacts.abs().mean().item())
                    try:
                        gt_per_c = gt_contacts.mean(dim=0).detach().cpu().tolist()
                    except Exception:
                        gt_per_c = None
                    try:
                        if gt_contacts.shape[-1] >= 2:
                            d = gt_contacts[:, 0] - gt_contacts[:, 1]
                            gt_lr_absdiff_mean = float(d.abs().mean().item())
                            gt_lr_diff_std = float(d.std(unbiased=False).item())
                    except Exception:
                        gt_lr_absdiff_mean = None
                        gt_lr_diff_std = None
                if torch.is_tensor(gt_contacts_next) and gt_contacts_next.ndim == 2:
                    gt_next_mean = float(gt_contacts_next.mean().item())
                    gt_next_abs_mean = float(gt_contacts_next.abs().mean().item())
                    try:
                        gt_next_per_c = gt_contacts_next.mean(dim=0).detach().cpu().tolist()
                    except Exception:
                        gt_next_per_c = None
                    try:
                        if gt_contacts_next.shape[-1] >= 2:
                            d = gt_contacts_next[:, 0] - gt_contacts_next[:, 1]
                            gt_next_lr_absdiff_mean = float(d.abs().mean().item())
                            gt_next_lr_diff_std = float(d.std(unbiased=False).item())
                    except Exception:
                        gt_next_lr_absdiff_mean = None
                        gt_next_lr_diff_std = None

                # Errors against GT contacts (debugging alignment / meas head quality).
                plan_gt_abs_mean = None
                meas_gt_abs_mean = None
                plan_gt_abs_per_c = None
                meas_gt_abs_per_c = None
                if torch.is_tensor(gt_contacts) and gt_contacts.ndim == 2:
                    if torch.is_tensor(plan) and plan.ndim == 2 and plan.shape == gt_contacts.shape:
                        diff = (plan - gt_contacts).abs()
                        plan_gt_abs_mean = float(diff.mean().item())
                        plan_gt_abs_per_c = diff.mean(dim=0).detach().cpu().tolist()
                    if torch.is_tensor(meas) and meas.ndim == 2 and meas.shape == gt_contacts.shape:
                        diff = (meas - gt_contacts).abs()
                        meas_gt_abs_mean = float(diff.mean().item())
                        meas_gt_abs_per_c = diff.mean(dim=0).detach().cpu().tolist()
                contact_entry = {
                    "ContactGTMean": gt_mean,
                    "ContactGTAbsMean": gt_abs_mean,
                    "ContactGTPerC": gt_per_c,
                    "ContactGTNextMean": gt_next_mean,
                    "ContactGTNextAbsMean": gt_next_abs_mean,
                    "ContactGTNextPerC": gt_next_per_c,
                    "ContactsMeasSource": str(contacts_meas_source_cfg),
                    "ContactsMeasSourceApplied": str(contacts_meas_source_applied),
                    "ContactPlanMean": plan_mean,
                    "ContactPlanAbsMean": plan_abs_mean,
                    "ContactPlanPerC": plan_per_c,
                    "ContactMeasMean": meas_mean,
                    "ContactMeasAbsMean": meas_abs_mean,
                    "ContactMeasPerC": meas_per_c,
                    "ContactPlanLogitsMean": plan_logits_mean,
                    "ContactPlanLogitsStd": plan_logits_std,
                    "ContactPlanLogitsPerC": plan_logits_per_c,
                    "ContactPlanLogitsBasePerC": plan_logits_base_per_c,
                    "ContactPlanLogitsPhasePerC": plan_logits_phase_per_c,
                    "ContactPlanLogitsTimePerC": plan_logits_time_per_c,
                    "ContactPlanLogitsRawPerC": plan_logits_raw_per_c,
                    "ContactMeasLogitsMean": meas_logits_mean,
                    "ContactMeasLogitsStd": meas_logits_std,
                    "ContactMeasLogitsPerC": meas_logits_per_c,
                    "AngvelMean": angvel_mean,
                    "AngvelAbsMean": angvel_abs_mean,
                    "AngvelStd": angvel_std,
                    "PoseHistMean": pose_hist_mean,
                    "PoseHistAbsMean": pose_hist_abs_mean,
                    "PoseHistStd": pose_hist_std,
                    "ContactPlanLRAbsDiffMean": plan_lr_absdiff_mean,
                    "ContactPlanLRDiffStd": plan_lr_diff_std,
                    "ContactMeasLRAbsDiffMean": meas_lr_absdiff_mean,
                    "ContactMeasLRDiffStd": meas_lr_diff_std,
                    "ContactGTLRAbsDiffMean": gt_lr_absdiff_mean,
                    "ContactGTLRDiffStd": gt_lr_diff_std,
                    "ContactGTNextLRAbsDiffMean": gt_next_lr_absdiff_mean,
                    "ContactGTNextLRDiffStd": gt_next_lr_diff_std,
                    "ContactErrAbsMean": err_abs_mean,
                    "ContactErrAbsPerC": err_abs_per_c,
                    "ContactErrPerC": err_per_c,
                    "ContactPlanGtAbsMean": plan_gt_abs_mean,
                    "ContactMeasGtAbsMean": meas_gt_abs_mean,
                    "ContactPlanGtAbsPerC": plan_gt_abs_per_c,
                    "ContactMeasGtAbsPerC": meas_gt_abs_per_c,
                }
                # Optional: contact_meas_head input-swap diagnostics (pose drift vs angvel drift).
                # This recomputes meas logits on:
                #   (pose_pred, angvel_pred) == current rollout state,
                #   (pose_pred, angvel_gt)   == "fix angvel",
                #   (pose_gt, angvel_pred)   == "fix pose",
                #   (pose_gt, angvel_gt)     == pure teacher-forced state.
                if bool(export_contact_meas_head_swap):
                    try:
                        raise RuntimeError("contact_meas_head swap diagnostics have been retired with the internal meas head.")
                        rot_sl = getattr(model, "_contact_meas_state_rot_slice", None)
                        av_sl = getattr(model, "_contact_meas_state_angvel_slice", None)
                        idx = getattr(model, "_contact_meas_lower_joint_idx", None)
                        if not isinstance(rot_sl, slice) or not isinstance(av_sl, slice):
                            raise RuntimeError("contact_meas_head v1 slices not initialized.")
                        if idx is None or (not torch.is_tensor(idx)) or int(idx.numel()) <= 0:
                            raise RuntimeError("contact_meas_head v1 lower-body idx not initialized.")

                        def _extract_lb(state_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                            st = state_2d
                            if st.ndim != 2:
                                st = st.reshape(st.shape[0], -1)
                            st = st.unsqueeze(1)  # (B,1,Dx)
                            pose_all = st[..., rot_sl]  # (B,1,J*6)
                            w_all = st[..., av_sl]      # (B,1,J*3)
                            Jp = int(pose_all.shape[-1] // 6)
                            Jw = int(w_all.shape[-1] // 3)
                            if Jp <= 0 or Jw <= 0:
                                raise RuntimeError("Invalid contact_meas_head input dims (J<=0).")
                            idx_dev = idx.to(device=pose_all.device)
                            pose_lower = (
                                pose_all.view(pose_all.shape[0], pose_all.shape[1], Jp, 6)
                                .index_select(2, idx_dev)
                                .reshape(pose_all.shape[0], pose_all.shape[1], -1)
                            )
                            w_lower = (
                                w_all.view(w_all.shape[0], w_all.shape[1], Jw, 3)
                                .index_select(2, idx_dev)
                                .reshape(w_all.shape[0], w_all.shape[1], -1)
                            )
                            return pose_lower, w_lower

                        pose_p, w_p = _extract_lb(motion)
                        pose_g, w_g = _extract_lb(state_seq[:, t])

                        # Drift norms (mean over batch).
                        Bc = int(pose_p.shape[0])
                        pose_l2 = float((pose_p - pose_g).reshape(Bc, -1).norm(dim=-1).mean().item())
                        w_l2 = float((w_p - w_g).reshape(Bc, -1).norm(dim=-1).mean().item())

                        with torch.no_grad():
                            head = getattr(model, "contact_meas_head")
                            log_pp = head(pose_p, w_p).squeeze(1)  # (B,C)
                            log_pg = head(pose_p, w_g).squeeze(1)
                            log_gp = head(pose_g, w_p).squeeze(1)
                            log_gg = head(pose_g, w_g).squeeze(1)

                        contact_entry["ContactMeasHeadSwapPoseL2"] = float(pose_l2)
                        contact_entry["ContactMeasHeadSwapAngvelL2"] = float(w_l2)
                        contact_entry["ContactMeasHeadSwapLogitsPPPerC"] = log_pp.mean(dim=0).detach().cpu().tolist()
                        contact_entry["ContactMeasHeadSwapLogitsPGPerC"] = log_pg.mean(dim=0).detach().cpu().tolist()
                        contact_entry["ContactMeasHeadSwapLogitsGPPerC"] = log_gp.mean(dim=0).detach().cpu().tolist()
                        contact_entry["ContactMeasHeadSwapLogitsGGPerC"] = log_gg.mean(dim=0).detach().cpu().tolist()

                        # Sanity: compare to model-returned meas logits (should match PP when meas head is used).
                        try:
                            if torch.is_tensor(meas_logits) and meas_logits.ndim == 2 and meas_logits.shape == log_pp.shape:
                                max_abs = float((meas_logits.detach() - log_pp.detach()).abs().max().item())
                                contact_entry["ContactMeasHeadSwapMaxAbsDiffToModelLogits"] = max_abs
                        except Exception:
                            pass
                    except Exception as e:
                        # Keep the rest of contact logging intact even if swap export fails.
                        try:
                            contact_entry["ContactMeasHeadSwapError"] = str(e)
                        except Exception:
                            pass
                # TTC / clock-anchor diagnostics
                try:
                    contact_entry["PhaseResetSource"] = str(phase_reset_source)
                    contact_entry["TTCEventKind"] = str(ttc_event_kind)
                    contact_entry["TTCGTPerC"] = (
                        ttc_gt_step.mean(dim=0).detach().cpu().tolist() if torch.is_tensor(ttc_gt_step) else None
                    )
                    contact_entry["TTCGTValidPerC"] = (
                        ttc_gt_valid_step.to(dtype=torch.float32).mean(dim=0).detach().cpu().tolist()
                        if torch.is_tensor(ttc_gt_valid_step)
                        else None
                    )
                    contact_entry["TTCStatePerC"] = (
                        ttc_state_step.mean(dim=0).detach().cpu().tolist() if torch.is_tensor(ttc_state_step) else None
                    )
                    contact_entry["TTCEventPerC"] = (
                        ttc_event_step.to(dtype=torch.float32).mean(dim=0).detach().cpu().tolist()
                        if torch.is_tensor(ttc_event_step)
                        else None
                    )
                except Exception:
                    pass
                # Debug: record what meas the model was explicitly fed (override only; otherwise None).
                try:
                    meas_override_per_c = None
                    if torch.is_tensor(contacts_in_t):
                        ov = contacts_in_t
                        if ov.ndim == 3 and ov.size(1) == 1:
                            ov = ov[:, 0]
                        if ov.ndim == 2:
                            meas_override_per_c = ov.mean(dim=0).detach().cpu().tolist()
                    contact_entry["ContactsMeasOverridePerC"] = meas_override_per_c
                except Exception:
                    pass
                # Debug: record what meas the direct head actually saw (override only; otherwise it's "model").
                try:
                    direct_meas_per_c = None
                    if isinstance(direct_meas_override, str):
                        if direct_meas_override.strip().lower() in ("ignore", "zero", "none"):
                            C = None
                            for ref in (plan, meas, gt_contacts):
                                if torch.is_tensor(ref) and ref.ndim == 2:
                                    C = int(ref.shape[-1])
                                    break
                            if C is not None and C > 0:
                                direct_meas_per_c = [0.0 for _ in range(C)]
                    elif torch.is_tensor(direct_meas_override):
                        ov = direct_meas_override
                        if ov.ndim == 3 and ov.size(1) == 1:
                            ov = ov[:, 0]
                        if ov.ndim == 2:
                            direct_meas_per_c = ov.mean(dim=0).detach().cpu().tolist()
                    contact_entry["DirectMeasSource"] = str(direct_meas_source_eff)
                    contact_entry["DirectMeasOverridePerC"] = direct_meas_per_c
                except Exception:
                    pass
                # Debug: record what plan the direct head actually saw (override only; otherwise it's "model").
                try:
                    direct_plan_per_c = None
                    if isinstance(direct_plan_override, str):
                        if direct_plan_override.strip().lower() in ("ignore", "zero", "none"):
                            C = None
                            for ref in (plan, meas, gt_contacts):
                                if torch.is_tensor(ref) and ref.ndim == 2:
                                    C = int(ref.shape[-1])
                                    break
                            if C is not None and C > 0:
                                direct_plan_per_c = [0.0 for _ in range(C)]
                    elif torch.is_tensor(direct_plan_override):
                        ov = direct_plan_override
                        if ov.ndim == 3 and ov.size(1) == 1:
                            ov = ov[:, 0]
                        if ov.ndim == 2:
                            direct_plan_per_c = ov.mean(dim=0).detach().cpu().tolist()
                    contact_entry["DirectPlanSource"] = str(direct_plan_source_eff)
                    contact_entry["DirectPlanOverridePerC"] = direct_plan_per_c
                except Exception:
                    pass
                if (
                    bool(getattr(trainer, "so3_corr_apply", False))
                    and bool(getattr(trainer, "so3_corr_gate_from_contacts_err", False))
                    and bool(getattr(model, "contact_plan_enable", False))
                    and (err_abs_mean is not None)
                ):
                    k = float(getattr(trainer, "so3_corr_gate_err_k", 1.0) or 1.0)
                    bias = float(getattr(trainer, "so3_corr_gate_err_bias", 0.0) or 0.0)
                    mx = float(getattr(trainer, "so3_corr_gate_err_max", 1.0) or 1.0)
                    warmup_active = bool(ref_err_steps > 0 and ref_err_count < ref_err_steps)
                    if warmup_active:
                        # Collect reference error for the first few steps (gate forced to 0.0).
                        if use_ref:
                            ref_err_sum += float(err_abs_mean)
                        ref_err_count += 1
                        gate_override = 0.0
                        contact_entry["So3GateWarmup"] = True
                        contact_entry["So3GateOverride"] = float(gate_override)
                    else:
                        eff = float(err_abs_mean)
                        if use_ref:
                            if ref_err_value is None:
                                denom = max(1, int(ref_err_count))
                                ref_err_value = float(ref_err_sum / denom) if denom > 0 else float(err_abs_mean)
                            eff = eff - float(ref_err_value)
                        eff = max(0.0, eff - float(ref_err_margin))
                        mode = str(getattr(trainer, "so3_corr_gate_from_contacts_err_mode", "scale") or "scale").lower()
                        # base gate: forced > learned > 0
                        base_gate = getattr(trainer, "so3_corr_gate_force", None)
                        if base_gate is None:
                            try:
                                logit = getattr(model, "so3_corr_gate_logit", None)
                                if torch.is_tensor(logit):
                                    base_gate = float(torch.sigmoid(logit.detach()).item())
                            except Exception:
                                base_gate = None
                        base_gate = float(base_gate or 0.0)
                        if mode == "override":
                            gate_override = max(0.0, min(mx, k * (eff - bias)))
                        else:
                            scale_max = float(getattr(trainer, "so3_corr_gate_scale_max", 2.0) or 2.0)
                            scale = max(0.0, min(scale_max, 1.0 + k * (eff - bias)))
                            gate_override = base_gate * scale
                            gate_override = max(0.0, min(mx, gate_override))
                        contact_entry["So3GateWarmup"] = False
                        contact_entry["So3GateErrRef"] = float(ref_err_value) if ref_err_value is not None else None
                        contact_entry["So3GateErrEff"] = float(eff)
                        contact_entry["So3GateBase"] = float(base_gate)
                        contact_entry["So3GateMode"] = str(mode)
                        contact_entry["So3GateScale"] = float(scale) if mode != "override" else None
                        contact_entry["So3GateOverride"] = float(gate_override)
            except Exception:
                contact_entry = None
        elif bool(plan_enable):
            # Phase-only logging: keep a tiny dict so phase state diagnostics are exported even when
            # --log_contacts is off.
            contact_entry = {}

        # Optional: attach detailed white-box intermediates for debugging discrete collapses.
        if contact_entry is not None and bool(getattr(trainer, "log_contacts_whitebox", False)):
            try:
                first_steps = int(getattr(trainer, "log_contacts_whitebox_first_steps", 4) or 4)
                step_idx = int(t - start_t)
                want_wb = step_idx < max(0, first_steps)
                if not want_wb:
                    meas_abs = contact_entry.get("ContactMeasAbsMean", None)
                    gt_abs = contact_entry.get("ContactGTAbsMean", None)
                    meas_gt_abs = contact_entry.get("ContactMeasGtAbsMean", None)
                    if (meas_abs is not None) and (gt_abs is not None) and float(meas_abs) < 0.05 and float(gt_abs) > 0.2:
                        want_wb = True
                    elif meas_gt_abs is not None and float(meas_gt_abs) > 0.35:
                        want_wb = True
                if want_wb:
                    wb = getattr(trainer, "_contact_meas_whitebox_debug", None)
                    if isinstance(wb, dict) and wb:
                        contact_entry["ContactMeasWhitebox"] = wb
            except Exception:
                pass

        if plan_enable:
            try:
                z_next = ret.get("plan_z_next", None)
                if z_next is not None:
                    plan_z = z_next.detach()
                    if contact_entry is not None:
                        try:
                            contact_entry["PlanZNorm"] = float(plan_z.norm().item())
                        except Exception:
                            contact_entry["PlanZNorm"] = None

                p_next = ret.get("phase_z_next", None)
                if p_next is not None:
                    phase_z = p_next.detach()
                    if contact_entry is not None:
                        try:
                            contact_entry["PhaseZNorm"] = float(phase_z.norm().item())
                        except Exception:
                            contact_entry["PhaseZNorm"] = None

                        # Extra phase debug (small, useful): per-foot sin/cos and angle.
                        # NOTE: accept multiple tensor layouts; always emit keys to avoid silent drops.
                        contact_entry["PhaseZShape"] = (
                            list(phase_z.shape) if torch.is_tensor(phase_z) else str(type(phase_z))
                        )
                        contact_entry["PhaseBinN"] = None
                        contact_entry["PhaseZSinCosPerC"] = None
                        contact_entry["PhaseAngleRadPerC"] = None
                        contact_entry["PhaseAngleDegPerC"] = None
                        contact_entry["PhaseZPerCNorm"] = None
                        contact_entry["PhaseBinPerC"] = None

                        # Also export the *input* phase state used at this step (pre-update) for cross-clip comparability.
                        # This is the exact phase state that was fed into the model on this step.
                        contact_entry["PhaseZInSinCosPerC"] = None
                        contact_entry["PhaseAngleInRadPerC"] = None
                        contact_entry["PhaseAngleInDegPerC"] = None
                        contact_entry["PhaseZInPerCNorm"] = None
                        contact_entry["PhaseBinInPerC"] = None
                        # Touchdown-anchored phase (derived from PhaseAngleInRadPerC + ContactGTPerC touchdown events).
                        contact_entry["PhaseAngleInTdRadPerC"] = None
                        contact_entry["PhaseAngleInTdDegPerC"] = None
                        contact_entry["PhaseBinInTdPerC"] = None

                        try:
                            Cc = int(getattr(model, "contact_dim", 0) or 0)
                        except Exception:
                            Cc = 0

                        if Cc > 0 and torch.is_tensor(phase_z):
                            try:
                                z = phase_z
                                v = None  # (B,C,2) with last dim = [sin, cos]
                                if z.ndim == 1 and int(z.numel()) == int(Cc) * 2:
                                    z = z.reshape(1, int(Cc) * 2)
                                if z.ndim == 2 and int(z.shape[-1]) == int(Cc) * 2:
                                    v = z.reshape(int(z.shape[0]), int(Cc), 2)
                                elif z.ndim == 2 and int(z.shape[0]) == int(Cc) and int(z.shape[1]) == 2:
                                    v = z.reshape(1, int(Cc), 2)
                                elif z.ndim == 3 and int(z.shape[-2]) == int(Cc) and int(z.shape[-1]) == 2:
                                    v = z
                                elif z.ndim == 3 and int(z.shape[1]) == 1 and int(z.shape[-1]) == int(Cc) * 2:
                                    v = z[:, 0, :].reshape(int(z.shape[0]), int(Cc), 2)
                                elif (
                                    z.ndim == 4
                                    and int(z.shape[1]) == 1
                                    and int(z.shape[2]) == int(Cc)
                                    and int(z.shape[3]) == 2
                                ):
                                    v = z[:, 0, :, :]

                                if v is not None:
                                    v_mean = v.mean(dim=0)  # (C,2)
                                    # Default bin count for diagnostics (kept fixed; phase-bin table support was reverted).
                                    n_bins = 96
                                    contact_entry["PhaseBinN"] = int(n_bins)
                                    contact_entry["PhaseZSinCosPerC"] = v_mean.detach().cpu().tolist()
                                    # angle in [-pi,pi], where anchor [sin=0,cos=1] -> 0
                                    ang = torch.atan2(v_mean[..., 0], v_mean[..., 1])
                                    contact_entry["PhaseAngleRadPerC"] = ang.detach().cpu().tolist()
                                    contact_entry["PhaseAngleDegPerC"] = (
                                        (ang * (180.0 / float(np.pi))).detach().cpu().tolist()
                                    )
                                    contact_entry["PhaseZPerCNorm"] = v_mean.norm(dim=-1).detach().cpu().tolist()
                                    if int(n_bins) > 0:
                                        try:
                                            twopi = float(2.0 * math.pi)
                                            ang01 = torch.remainder(ang, twopi) / twopi
                                            b = torch.floor(ang01 * float(n_bins)).to(torch.long).clamp(0, int(n_bins) - 1)
                                            contact_entry["PhaseBinPerC"] = b.detach().cpu().tolist()
                                        except Exception:
                                            contact_entry["PhaseBinPerC"] = None
                            except Exception:
                                pass

                        # Pre-update phase input (use the exact tensor fed into the model, i.e. phase_z_eff).
                        if Cc > 0 and torch.is_tensor(phase_z_eff):
                            try:
                                v_in = None
                                z_in = phase_z_eff
                                if z_in.ndim == 1 and int(z_in.numel()) == int(Cc) * 2:
                                    z_in = z_in.reshape(1, int(Cc) * 2)
                                if z_in.ndim == 2 and int(z_in.shape[-1]) == int(Cc) * 2:
                                    v_in = z_in.reshape(int(z_in.shape[0]), int(Cc), 2)
                                elif z_in.ndim == 3 and int(z_in.shape[0]) == int(Cc) and int(z_in.shape[1]) == 2:
                                    v_in = z_in.reshape(1, int(Cc), 2)
                                elif z_in.ndim == 3 and int(z_in.shape[-2]) == int(Cc) and int(z_in.shape[-1]) == 2:
                                    v_in = z_in
                                elif z_in.ndim == 3 and int(z_in.shape[1]) == 1 and int(z_in.shape[-1]) == int(Cc) * 2:
                                    v_in = z_in[:, 0, :].reshape(int(z_in.shape[0]), int(Cc), 2)
                                elif (
                                    z_in.ndim == 4
                                    and int(z_in.shape[1]) == 1
                                    and int(z_in.shape[2]) == int(Cc)
                                    and int(z_in.shape[3]) == 2
                                ):
                                    v_in = z_in[:, 0, :, :]
                                if v_in is not None:
                                    v_in_mean = v_in.mean(dim=0)  # (C,2)
                                    contact_entry["PhaseZInSinCosPerC"] = v_in_mean.detach().cpu().tolist()
                                    ang_in = torch.atan2(v_in_mean[..., 0], v_in_mean[..., 1])
                                    contact_entry["PhaseAngleInRadPerC"] = ang_in.detach().cpu().tolist()
                                    contact_entry["PhaseAngleInDegPerC"] = (
                                        (ang_in * (180.0 / float(np.pi))).detach().cpu().tolist()
                                    )
                                    contact_entry["PhaseZInPerCNorm"] = v_in_mean.norm(dim=-1).detach().cpu().tolist()
                                    n_bins = contact_entry.get("PhaseBinN", None)
                                    if isinstance(n_bins, int) and int(n_bins) > 0:
                                        twopi = float(2.0 * math.pi)
                                        ang01 = torch.remainder(ang_in, twopi) / twopi
                                        b = (
                                            torch.floor(ang01 * float(n_bins))
                                            .to(torch.long)
                                            .clamp(0, int(n_bins) - 1)
                                        )
                                        contact_entry["PhaseBinInPerC"] = b.detach().cpu().tolist()
                            except Exception:
                                pass

                a_next = ret.get("phase_event_age_next", None)
                if a_next is not None:
                    phase_event_age = a_next.detach()
                    if contact_entry is not None and torch.is_tensor(phase_event_age) and phase_event_age.ndim == 2:
                        try:
                            contact_entry["PhaseEventAgePerC"] = phase_event_age.mean(dim=0).detach().cpu().tolist()
                            contact_entry["PhaseEventAgeMean"] = float(phase_event_age.mean().item())
                        except Exception:
                            contact_entry["PhaseEventAgePerC"] = None
                            contact_entry["PhaseEventAgeMean"] = None
            except Exception:
                pass

        # External phase reset from TTC anchors (avoids contact threshold crossing jitter).
        ev_src = None
        if phase_reset_source in ("ttc_gt", "ttc"):
            ev_src = ttc_event_step
        if (
            bool(plan_enable)
            and bool(ttc_apply_phase_reset_to_phase_z)
            and phase_reset_source in ("ttc_gt", "ttc")
            and torch.is_tensor(phase_z)
            and torch.is_tensor(ev_src)
        ):
            try:
                Cc = int(getattr(model, "contact_dim", 0) or 0)
            except Exception:
                Cc = 0
            if Cc > 0:
                ev = ev_src
                try:
                    if ev.ndim == 3 and ev.size(1) == 1:
                        ev = ev[:, 0]
                    if ev.ndim == 1:
                        ev = ev.view(1, -1)
                    if ev.ndim != 2:
                        ev = ev.reshape(phase_z.shape[0], -1)
                    if ev.shape[0] == 1 and phase_z.shape[0] > 1:
                        ev = ev.expand(phase_z.shape[0], -1)
                    if int(ev.shape[-1]) != int(Cc):
                        if int(ev.shape[-1]) > int(Cc):
                            ev = ev[..., :Cc]
                        else:
                            pad = int(Cc) - int(ev.shape[-1])
                            ev = torch.cat([ev, ev.new_zeros(ev.shape[0], pad)], dim=-1)
                except Exception:
                    ev = None

                if torch.is_tensor(ev) and phase_z.numel() == int(phase_z.shape[0]) * int(Cc) * 2:
                    try:
                        # phase_z: (B, 2C) -> (B,C,2)
                        phase = phase_z.view(phase_z.shape[0], Cc, 2)
                        anchor = phase.new_zeros((phase.shape[0], Cc, 2))
                        anchor[..., 1] = 1.0
                        m = ev.to(dtype=phase.dtype).unsqueeze(-1)
                        phase = phase * (1.0 - m) + anchor * m
                        phase_z = phase.reshape(phase_z.shape[0], -1)
                        if contact_entry is not None:
                            try:
                                contact_entry["PhaseZNorm"] = float(phase_z.norm().item())
                            except Exception:
                                contact_entry["PhaseZNorm"] = None
                    except Exception:
                        pass

                    # Track/update phase_event_age externally (frames since last reset).
                    try:
                        if not torch.is_tensor(phase_event_age):
                            phase_event_age = torch.zeros((phase_z.shape[0], Cc), device=phase_z.device, dtype=phase_z.dtype)
                        age = phase_event_age
                        if age.ndim == 3 and age.size(1) == 1:
                            age = age[:, 0]
                        if age.ndim == 1:
                            age = age.view(1, -1)
                        if age.ndim != 2:
                            age = age.reshape(phase_z.shape[0], -1)
                        if age.shape[0] == 1 and phase_z.shape[0] > 1:
                            age = age.expand(phase_z.shape[0], -1)
                        if int(age.shape[-1]) != int(Cc):
                            if int(age.shape[-1]) > int(Cc):
                                age = age[..., :Cc]
                            else:
                                pad = int(Cc) - int(age.shape[-1])
                                age = torch.cat([age, age.new_zeros(age.shape[0], pad)], dim=-1)
                        phase_event_age = torch.where(ev, torch.zeros_like(age), age + 1.0)
                    except Exception:
                        pass

        # Update Event-Clock prev meas buffers (used to build Δmeas at next step when T=1).
        try:
            meas_logits_step = ret.get("contacts_meas_logits", None)
            if torch.is_tensor(meas_logits_step):
                meas_prev_logits = meas_logits_step.detach()
            meas_prob_step = ret.get("contacts_meas", None)
            if torch.is_tensor(meas_prob_step):
                meas_prev_prob = meas_prob_step.detach()
        except Exception:
            pass

        delta_norm = out
        if y_raw_prev is not None:
            try:
                so3_gate = gate_override if gate_override is not None else getattr(trainer, "so3_corr_gate_force", None)
                y_inc_raw = trainer._compose_delta_to_raw(
                    y_raw_prev,
                    delta_norm,
                    omega_hat=ret.get("omega_hat", None) if bool(getattr(trainer, "so3_corr_apply", False)) else None,
                    so3_gate=so3_gate,
                    so3_max_deg=getattr(trainer, "so3_corr_max_deg", None),
                    omega_detach=True,
                )
            except Exception:
                y_inc_raw = trainer._denorm(delta_norm)
        else:
            y_inc_raw = trainer._denorm(delta_norm)

        # Optional: finite-difference probe for local closed-loop gain (one-step amplification).
        rot_gain_dbg: Optional[Dict[str, Any]] = None
        if (
            bool(debug_rot_gain)
            and y_raw_prev_in is not None
            and torch.is_tensor(y_inc_raw)
            and torch.is_tensor(motion_raw_in)
            and int(J) > 0
            and rot_gain_joint_indices
        ):
            try:
                import math

                from train.geometry import matrix_to_rot6d, so3_exp_map

                B_dbg = int(y_raw_prev_in.shape[0])
                if B_dbg > 0:
                    # Current pose rot6d (raw) -> rotation matrices.
                    prev_flat = y_raw_prev_in[..., rot_slice].reshape(B_dbg, int(J) * 6)
                    prev_flat = reproject_rot6d(prev_flat)
                    prev6 = prev_flat.reshape(B_dbg, int(J), 6)
                    R_prev = rot6d_to_matrix(prev6, columns=cols)

                    # Apply perturbation: R_prev_pert = Exp(omega) @ R_prev.
                    axis_idx = {"x": 0, "y": 1, "z": 2}[str(rot_gain_axis)]
                    omega = torch.zeros((B_dbg, int(J), 3), device=R_prev.device, dtype=R_prev.dtype)
                    omega[:, rot_gain_joint_indices, axis_idx] = float(rot_gain_deg) * (math.pi / 180.0)
                    R_delta = so3_exp_map(omega)
                    R_prev_pert = torch.matmul(R_delta, R_prev)
                    prev6_pert = matrix_to_rot6d(R_prev_pert, columns=cols)
                    prev_flat_pert = prev6_pert.reshape(B_dbg, int(J) * 6)

                    # Perturb raw prev pose + X-state rot6d slice to keep inputs consistent.
                    y_prev_pert = y_raw_prev_in.clone()
                    y_prev_pert[..., rot_slice] = prev_flat_pert.reshape_as(y_prev_pert[..., rot_slice])

                    motion_raw_pert = motion_raw_in.clone()
                    rx = getattr(trainer, "rot6d_x_slice", None) or getattr(trainer, "rot6d_slice", None)
                    if isinstance(rx, slice):
                        rx_len = int(rx.stop - rx.start)
                        if rx_len == int(J) * 6:
                            motion_raw_pert[..., rx] = prev_flat_pert
                    motion_pert = trainer._diag_norm_x(motion_raw_pert)

                    # Optional: also perturb pose_history's most recent block (buffer mode only).
                    pose_hist_t_pert = pose_hist_t
                    if (
                        pose_hist_enabled
                        and pose_hist_source == "buffer"
                        and torch.is_tensor(pose_hist_buffer_raw_in)
                        and scales is not None
                        and int(pose_hist_stride) == int(J) * 6
                    ):
                        try:
                            buf_raw_pert = pose_hist_buffer_raw_in.clone()
                            buf_raw_pert[..., -int(pose_hist_stride):] = prev_flat_pert
                            pose_hist_t_pert = pose_hist_transform_vec(buf_raw_pert, scales, mu, std)
                        except Exception:
                            pose_hist_t_pert = pose_hist_t

                    # Run one extra forward pass with perturbed inputs, holding latent states constant.
                    with amp_ctx:
                        ret_pert = model(
                            motion_pert,
                            cond_input,
                            contacts=contacts_in_t,
                            angvel=angvel_t,
                            pose_history=pose_hist_t_pert,
                            plan_z=plan_z_in,
                            phase_z=phase_z_in,
                            phase_event_age=phase_event_age_in,
                            meas_logits_prev=meas_prev_in_in,
                            time_index=time_index_t,
                            rollout_step=rollout_step_t,
                        )
                    out_pert = ret_pert.get("out") if isinstance(ret_pert, dict) else None
                    y_next_pert = None
                    if out_pert is not None:
                        delta_norm_pert = out_pert
                        if y_prev_pert is not None:
                            try:
                                so3_gate = gate_override if gate_override is not None else getattr(trainer, "so3_corr_gate_force", None)
                                y_next_pert = trainer._compose_delta_to_raw(
                                    y_prev_pert,
                                    delta_norm_pert,
                                    omega_hat=ret_pert.get("omega_hat", None)
                                    if bool(getattr(trainer, "so3_corr_apply", False))
                                    else None,
                                    so3_gate=so3_gate,
                                    so3_max_deg=getattr(trainer, "so3_corr_max_deg", None),
                                    omega_detach=True,
                                )
                            except Exception:
                                y_next_pert = trainer._denorm(delta_norm_pert)
                        else:
                            y_next_pert = trainer._denorm(delta_norm_pert)

                    if torch.is_tensor(y_next_pert):
                        # Next pose rotation matrices for baseline vs perturbed.
                        next_flat = reproject_rot6d(y_inc_raw[..., rot_slice].reshape(B_dbg, int(J) * 6))
                        next6 = next_flat.reshape(B_dbg, int(J), 6)
                        R_next = rot6d_to_matrix(next6, columns=cols)

                        next_flat_pert = reproject_rot6d(y_next_pert[..., rot_slice].reshape(B_dbg, int(J) * 6))
                        next6_pert = next_flat_pert.reshape(B_dbg, int(J), 6)
                        R_next_pert = rot6d_to_matrix(next6_pert, columns=cols)

                        deg = 180.0 / math.pi
                        d_prev = geodesic_R(R_prev_pert, R_prev) * deg  # (B,J)
                        d_next = geodesic_R(R_next_pert, R_next) * deg  # (B,J)
                        d_prev_m = d_prev.mean(dim=0)
                        d_next_m = d_next.mean(dim=0)

                        # Selected joints (mean).
                        sel = torch.as_tensor(rot_gain_joint_indices, device=d_prev_m.device, dtype=torch.long)
                        d0_sel = float(d_prev_m.index_select(0, sel).mean().item()) if int(sel.numel()) > 0 else None
                        d1_sel = float(d_next_m.index_select(0, sel).mean().item()) if int(sel.numel()) > 0 else None
                        gain_sel = None
                        if d0_sel is not None and d1_sel is not None and float(d0_sel) > 1e-6:
                            gain_sel = float(d1_sel) / float(d0_sel)

                        # Ex-root max (captures spread/coupling).
                        mask = torch.ones(int(J), device=d_prev_m.device, dtype=torch.bool)
                        if 0 <= int(rot_gain_root_idx) < int(J):
                            mask[int(rot_gain_root_idx)] = False
                        d0_max = float(d_prev_m[mask].max().item()) if bool(mask.any().item()) else float(d_prev_m.max().item())
                        d1_max = float(d_next_m[mask].max().item()) if bool(mask.any().item()) else float(d_next_m.max().item())
                        gain_max = float(d1_max / max(1e-6, d0_max))

                        # Which joint becomes max at next step (ex-root).
                        max_joint = None
                        max_joint_idx = None
                        try:
                            if bool(mask.any().item()):
                                idxs = mask.nonzero(as_tuple=False).view(-1)
                                j = int(idxs[d_next_m[mask].argmax()].item())
                            else:
                                j = int(d_next_m.argmax().item())
                            max_joint_idx = j
                            if rot_gain_bone_names and 0 <= int(j) < len(rot_gain_bone_names):
                                max_joint = str(rot_gain_bone_names[int(j)])
                        except Exception:
                            max_joint = None
                            max_joint_idx = None

                        rot_gain_dbg = {
                            "RotGainDbgD0SelDegMean": d0_sel,
                            "RotGainDbgD1SelDegMean": d1_sel,
                            "RotGainDbgGainSel": gain_sel,
                            "RotGainDbgD0MaxDegExRoot": d0_max,
                            "RotGainDbgD1MaxDegExRoot": d1_max,
                            "RotGainDbgGainMaxExRoot": gain_max,
                            "RotGainDbgNextMaxJoint": max_joint,
                            "RotGainDbgNextMaxJointIdx": max_joint_idx,
                        }
            except Exception:
                rot_gain_dbg = None

        # Optional: SO(3) corrector diagnostics (omega_hat alignment + 1-step effect).
        so3_dbg: Optional[Dict[str, Any]] = None
        if bool(debug_so3_corr) and y_raw_prev is not None and J > 0:
            try:
                import math

                from train.geometry import so3_log_map

                omega_hat = ret.get("omega_hat", None)
                if torch.is_tensor(omega_hat) and omega_hat.numel() > 0:
                    if omega_hat.dim() == 4 and omega_hat.size(1) == 1:
                        omega_hat = omega_hat[:, 0]
                    if omega_hat.dim() == 3 and omega_hat.shape[-2:] == (J, 3):
                        # Resolve gate used during compose.
                        gate_used = gate_override if gate_override is not None else getattr(trainer, "so3_corr_gate_force", None)
                        if gate_used is None:
                            logit = getattr(model, "so3_corr_gate_logit", None)
                            if torch.is_tensor(logit):
                                gate_used = float(torch.sigmoid(logit.detach()).item())
                            else:
                                gate_used = 0.0
                        gate_used = float(gate_used or 0.0)

                        max_deg = getattr(trainer, "so3_corr_max_deg", None)
                        if max_deg is None:
                            max_deg = float(getattr(trainer, "so3_corr_max_deg", 20.0) or 20.0)
                        max_deg = float(max_deg or 0.0)
                        max_rad = (max_deg * (math.pi / 180.0)) if max_deg > 0.0 else None

                        omega_hat_det = omega_hat.detach().to(device=y_raw_prev.device, dtype=y_raw_prev.dtype)
                        omega_eff = omega_hat_det * gate_used
                        clipped_frac = 0.0
                        if max_rad is not None:
                            n = omega_eff.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                            s = (max_rad / n).clamp_max(1.0)
                            omega_eff = omega_eff * s
                            try:
                                clipped_frac = float((s < 1.0).to(dtype=omega_eff.dtype).mean().item())
                            except Exception:
                                clipped_frac = 0.0

                        # Compose a "no-corr" and a "corr" next pose for 1-step comparison.
                        try:
                            y_inc_raw_nocorr = trainer._compose_delta_to_raw(y_raw_prev, delta_norm, omega_hat=None)
                        except Exception:
                            y_inc_raw_nocorr = y_inc_raw
                        try:
                            y_inc_raw_corr = trainer._compose_delta_to_raw(
                                y_raw_prev,
                                delta_norm,
                                omega_hat=omega_hat,
                                so3_gate=float(gate_used),
                                so3_max_deg=float(max_deg),
                                omega_detach=True,
                            )
                        except Exception:
                            y_inc_raw_corr = y_inc_raw

                        gt_raw_step = None
                        try:
                            gt_raw_step = trainer._denorm(gt_seq[:, t])
                        except Exception:
                            gt_raw_step = None

                        if torch.is_tensor(gt_raw_step):
                            B_dbg = int(gt_raw_step.shape[0])
                            pred6 = reproject_rot6d(y_inc_raw_nocorr[..., rot_slice].reshape(B_dbg, J, 6))
                            pred6_corr = reproject_rot6d(y_inc_raw_corr[..., rot_slice].reshape(B_dbg, J, 6))
                            gt6 = reproject_rot6d(gt_raw_step[..., rot_slice].reshape(B_dbg, J, 6))

                            R_pred = rot6d_to_matrix(pred6, columns=cols)
                            R_pred_corr = rot6d_to_matrix(pred6_corr, columns=cols)
                            R_gt = rot6d_to_matrix(gt6, columns=cols)

                            # Target correction to fix predicted absolute pose:
                            #   R_gt ≈ Exp(omega_target) @ R_pred
                            R_err_world = torch.matmul(R_gt, R_pred.transpose(-1, -2))
                            omega_tgt_world = so3_log_map(R_err_world)  # (B,J,3)
                            R_err_body = torch.matmul(R_pred.transpose(-1, -2), R_gt)
                            omega_tgt_body = so3_log_map(R_err_body)

                            def _mean_cos(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
                                try:
                                    na = a.norm(dim=-1)
                                    nb = b.norm(dim=-1)
                                    mask = (na > 1e-6) & (nb > 1e-6)
                                    if not bool(mask.any().item()):
                                        return None
                                    cos = (a * b).sum(dim=-1) / (na * nb).clamp_min(1e-6)
                                    return float(cos[mask].mean().item())
                                except Exception:
                                    return None

                            omega_body_to_world = torch.matmul(R_pred, omega_eff.unsqueeze(-1)).squeeze(-1)

                            cos_hat_world = _mean_cos(omega_hat_det, omega_tgt_world)
                            cos_eff_world = _mean_cos(omega_eff, omega_tgt_world)
                            cos_eff_body = _mean_cos(omega_eff, omega_tgt_body)
                            cos_eff_world_if_body = _mean_cos(omega_body_to_world, omega_tgt_world)

                            deg = 180.0 / math.pi
                            omega_hat_deg = float((omega_hat_det.norm(dim=-1) * deg).mean().item())
                            omega_eff_deg = float((omega_eff.norm(dim=-1) * deg).mean().item())
                            omega_tgt_deg = float((omega_tgt_world.norm(dim=-1) * deg).mean().item())

                            geo = geodesic_R(R_pred, R_gt) * deg
                            geo_corr = geodesic_R(R_pred_corr, R_gt) * deg
                            if J > 1:
                                geo_loc = float(geo[..., 1:].mean().item())
                                geo_loc_corr = float(geo_corr[..., 1:].mean().item())
                            else:
                                geo_loc = float(geo.mean().item())
                                geo_loc_corr = float(geo_corr.mean().item())

                            so3_dbg = {
                                "So3DbgGate": float(gate_used),
                                "So3DbgOmegaHatDegMean": omega_hat_deg,
                                "So3DbgOmegaEffDegMean": omega_eff_deg,
                                "So3DbgOmegaTargetWorldDegMean": omega_tgt_deg,
                                "So3DbgCosHatWorld": cos_hat_world,
                                "So3DbgCosEffWorld": cos_eff_world,
                                "So3DbgCosEffBody": cos_eff_body,
                                "So3DbgCosEffWorldIfBody": cos_eff_world_if_body,
                                "So3DbgGeoLocalNoCorr": geo_loc,
                                "So3DbgGeoLocalWithCorr": geo_loc_corr,
                                "So3DbgGeoLocalDelta": float(geo_loc_corr - geo_loc),
                                "So3DbgClippedFrac": float(clipped_frac),
                            }
            except Exception:
                so3_dbg = None
        so3_debug_log.append(so3_dbg)
        rot_gain_debug_log.append(rot_gain_dbg)

        # Stage2: on-manifold blend (incremental -> direct) for rollout update + metrics.
        y_blend_raw = y_inc_raw
        if bool(lambda_fusion_apply) and y_inc_raw is not None and torch.is_tensor(y_inc_raw):
            lam_for_blend = lam_eff_step if torch.is_tensor(lam_eff_step) else lam_step
            if torch.is_tensor(direct_norm_step) and torch.is_tensor(lam_for_blend):
                try:
                    y_blend_raw = trainer._apply_lambda_fusion_to_raw(
                        y_inc_raw,
                        direct_norm=direct_norm_step,
                        lambda_fusion=lam_for_blend,
                    )
                except Exception:
                    y_blend_raw = y_inc_raw

        # Store incremental and blend predictions (normalized absolute Y).
        try:
            y_inc_norm = trainer._norm_y(y_inc_raw)
        except Exception:
            y_inc_norm = delta_norm
        predsY.append(y_inc_norm.detach())

        try:
            y_blend_norm = trainer._norm_y(y_blend_raw)
        except Exception:
            y_blend_norm = y_inc_norm
        predsY_blend.append(y_blend_norm.detach())

        # Choose the actual rollout state update.
        y_used_raw = y_blend_raw if bool(lambda_fusion_apply) else y_inc_raw
        y_raw_prev = y_used_raw.detach() if torch.is_tensor(y_used_raw) else None

        if torch.is_tensor(direct_norm_step):
            predsY_direct.append(direct_norm_step.detach())
        else:
            predsY_direct.append(y_inc_norm.detach())

        if motion_raw is not None:
            motion_raw = trainer._apply_free_carry(motion_raw, y_used_raw, cond_next_raw=cond_raw_step).detach()
            motion = trainer._diag_norm_x(motion_raw)
            if bool(freerun_x_gt) and torch.is_tensor(gt_motion_raw) and gt_motion_raw.shape == motion_raw.shape:
                try:
                    motion_raw = gt_motion_raw.detach()
                    motion = trainer._diag_norm_x(motion_raw)
                except Exception:
                    pass
            elif bool(freerun_x_gt_except_rot6d) and torch.is_tensor(gt_motion_raw) and gt_motion_raw.shape == motion_raw.shape:
                rx = getattr(trainer, "rot6d_x_slice", None) or getattr(trainer, "rot6d_slice", None)
                if isinstance(rx, slice):
                    try:
                        hybrid = gt_motion_raw.detach().clone()
                        hybrid[..., rx] = motion_raw[..., rx]
                        motion_raw = hybrid
                        motion = trainer._diag_norm_x(motion_raw)
                    except Exception:
                        pass
        else:
            motion = trainer._apply_free_carry(motion, y_used_raw, cond_next_raw=None).detach()

        predsX.append(motion)

        # Align contact logging with predsY/predsX timeline.
        contacts_log.append(contact_entry)

        donor_y_used_raw = _advance_pose_hist_hybrid_donor_step(
            step_t=int(t),
            is_cycle_start_step=bool(is_cycle_start),
            gt_motion_next_shared=gt_motion_next,
            cond_raw_step_shared=cond_raw_step,
            contacts_in_step=contacts_in_t,
            time_index_step=time_index_t,
            rollout_step_step=rollout_step_t,
            direct_meas_override_step=direct_meas_override,
            direct_plan_override_step=direct_plan_override,
            gate_override_step=gate_override,
            amp_ctx=amp_ctx,
        )

        if pose_hist_enabled and pose_hist_stride > 0 and pose_hist_source == "buffer" and pose_hist_update_source != "freeze":
            rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
            rot_write = None
            if pose_hist_update_source == "gt":
                try:
                    gt_raw_step = trainer._denorm(gt_seq[:, t])
                except Exception:
                    gt_raw_step = None
                if torch.is_tensor(gt_raw_step) and isinstance(rot_slice, slice):
                    rot_write = gt_raw_step[..., rot_slice]
            elif pose_hist_update_source == "zero":
                rot_write = torch.zeros(
                    (B, pose_hist_stride),
                    device=device,
                    dtype=state_seq.dtype,
                )
            else:
                if torch.is_tensor(y_used_raw) and isinstance(rot_slice, slice):
                    rot_write = y_used_raw[..., rot_slice]
                    is_cycle_boundary = bool(int(T_cycle) > 0 and (int(t) % int(T_cycle)) == (int(T_cycle) - 1))
                    if pose_hist_hybrid_enabled and is_cycle_boundary:
                        donor_rot_write = None
                        if torch.is_tensor(donor_y_used_raw):
                            donor_rot_slice = donor_state.get("rot_slice") if isinstance(donor_state, dict) else None
                            if isinstance(donor_rot_slice, slice):
                                donor_rot_write = donor_y_used_raw[..., donor_rot_slice]
                        hybrid_rot_write = _compose_pose_hist_hybrid_rot_write(
                            rot_write,
                            donor_rot_write,
                            leg_joint_idx=pose_hist_hybrid_leg_idx,
                        )
                        if torch.is_tensor(hybrid_rot_write):
                            rot_write = hybrid_rot_write
            pose_hist_state = advance_pose_hist_state_with_tail(
                pose_hist_state,
                rot_tail_raw=rot_write,
            )

    if not predsY:
        raise RuntimeError("No predictions produced during free‑run.")

    # Align predictions and GT (match train/eval_utils.evaluate_freerun):
    # Dataset Y is already aligned to the "next frame" during conversion
    # (see MotionEventDataset.__getitem__: "Y 已在转换阶段对齐到 下一帧"),
    # and free-run evaluation compares predY[t] vs gtY[t] starting at start_t.
    predY_full = torch.stack(predsY, dim=1)  # [B, free_steps_raw, Dy]
    predY_blend_full = torch.stack(predsY_blend, dim=1) if predsY_blend else None  # [B, free_steps_raw, Dy]
    predY_direct_full = torch.stack(predsY_direct, dim=1) if predsY_direct else None  # [B, free_steps_raw, Dy]
    free_steps_raw = predY_full.shape[1]
    max_aligned = max(0, min(free_steps_raw, T_total - start_t))
    if max_aligned <= 0:
        raise RuntimeError("Not enough frames for aligned free-run evaluation.")
    predY = predY_full[:, :max_aligned]
    predY_blend = predY_blend_full[:, :max_aligned] if predY_blend_full is not None else None
    predY_direct = predY_direct_full[:, :max_aligned] if predY_direct_full is not None else None
    free_steps = max_aligned
    gt_start = start_t
    gt_end = gt_start + free_steps
    gtY = gt_seq[:, gt_start:gt_end]

    # Align predicted X (state) to GT for root drift metrics.
    predX_full = torch.stack(predsX, dim=1) if predsX else None  # [B, free_steps_raw, Dx] (next-state sequence)
    predX = predX_full[:, :max_aligned] if predX_full is not None else None  # [B, free_steps, Dx]
    # predsX[t] is the carried state after predicting predY[t], so align it to GT state at t+1.
    gtX = state_seq[:, gt_start + 1:gt_end + 1]  # [B, free_steps, Dx]

    contact_steps = contacts_log[:max_aligned] if contacts_log else []
    time_index_steps = time_index_log[:max_aligned] if time_index_log else []
    lambda_steps = lambda_log[:max_aligned] if lambda_log else []
    lambda_eff_steps = lambda_eff_log[:max_aligned] if lambda_eff_log else []
    lambda_rel_steps = lambda_rel_log[:max_aligned] if lambda_rel_log else []
    event_clock_lambda_corr_steps = event_clock_lambda_corr_log[:max_aligned] if event_clock_lambda_corr_log else []
    so3_debug_steps = so3_debug_log[:max_aligned] if so3_debug_log else []
    rot_gain_debug_steps = rot_gain_debug_log[:max_aligned] if rot_gain_debug_log else []

    predX_raw = None
    gtX_raw = None
    root_pos_err: Optional[torch.Tensor] = None  # [free_steps]
    root_vel_mae: Optional[torch.Tensor] = None  # [free_steps]
    if predX is not None and getattr(trainer, "normalizer", None) is not None:
        try:
            with torch.no_grad():
                predX_raw = trainer.normalizer.denorm_x(predX.reshape(-1, predX.shape[-1])).view_as(predX)
                gtX_raw = trainer.normalizer.denorm_x(gtX.reshape(-1, gtX.shape[-1])).view_as(gtX)
            rootpos_sl = getattr(trainer, "rootpos_x_slice", None)
            if isinstance(rootpos_sl, slice):
                diff = predX_raw[..., rootpos_sl] - gtX_raw[..., rootpos_sl]
                root_pos_err = torch.norm(diff, dim=-1).mean(dim=0)  # [free_steps]
            rootvel_sl = getattr(trainer, "rootvel_x_slice", None)
            if isinstance(rootvel_sl, slice):
                diff = predX_raw[..., rootvel_sl] - gtX_raw[..., rootvel_sl]
                root_vel_mae = diff.abs().mean(dim=-1).mean(dim=0)  # [free_steps]
        except Exception:
            predX_raw = None
            gtX_raw = None
            root_pos_err = None
            root_vel_mae = None

    # ---- Per‑round metrics ---------------------------------------------------
    # Shared slices for rotations (reuse previously inferred)
    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        rot_slice = slice(0, predY.shape[-1])
    width = rot_slice.stop - rot_slice.start
    deg_factor = 180.0 / float(np.pi)
    metrics_per_round: List[Dict[str, Any]] = []

    # Denorm entire run once for GeoDeg
    with torch.no_grad():
        pred_raw_full = trainer._denorm(predY.reshape(1, free_steps, -1))
        pred_blend_raw_full = (
            trainer._denorm(predY_blend.reshape(1, free_steps, -1))
            if predY_blend is not None
            else None
        )
        gt_raw_full = trainer._denorm(gtY.reshape(1, free_steps, -1))
        pred_direct_raw_full = (
            trainer._denorm(predY_direct.reshape(1, free_steps, -1))
            if predY_direct is not None
            else None
        )

    per_step: List[Dict[str, Any]] = []
    extra: Dict[str, Any] = {}
    if isinstance(seam_closure, dict) and seam_closure:
        extra["seam_closure"] = seam_closure
    extra["cond_reprojection"] = str(cond_reprojection)
    extra["analyze_phase_shift"] = bool(analyze_phase_shift)
    extra["phase_shift_max"] = int(phase_shift_max) if phase_shift_max is not None else None
    extra["pose_hist_hybrid_boundary_carry_requested"] = bool(pose_hist_hybrid_boundary_carry)
    extra["pose_hist_hybrid_boundary_carry_enabled"] = bool(pose_hist_hybrid_enabled)
    extra["debug_direct_alignment"] = bool(debug_direct_alignment)
    extra["direct_alignment_max_shift"] = int(direct_alignment_max_shift)
    extra["direct_alignment_joints"] = str(direct_alignment_joints or "")
    extra["direct_alignment_include_round0"] = bool(direct_alignment_include_round0)
    if bool(export_direct_leg_omega_grad) and (direct_leg_omega_grad_meta or direct_leg_omega_grad_steps):
        payload = dict(direct_leg_omega_grad_meta) if isinstance(direct_leg_omega_grad_meta, dict) else {"enabled": True}
        payload["steps"] = direct_leg_omega_grad_steps
        # Small summary for quick scanning (mean/p99/max).
        try:
            if direct_leg_omega_grad_steps:
                theta = np.asarray(
                    [float(s.get("theta_grad_norm", 0.0) or 0.0) for s in direct_leg_omega_grad_steps], dtype=np.float32
                )
                loss_arr = np.asarray(
                    [float(s.get("loss_deg", 0.0) or 0.0) for s in direct_leg_omega_grad_steps], dtype=np.float32
                )
                summary: Dict[str, Any] = {
                    "steps": int(len(direct_leg_omega_grad_steps)),
                    "theta_grad_norm": {
                        "mean": float(theta.mean()),
                        "p99": float(np.quantile(theta, 0.99)),
                        "max": float(theta.max()),
                    },
                    "loss_deg": {
                        "mean": float(loss_arr.mean()),
                        "p99": float(np.quantile(loss_arr, 0.99)),
                        "max": float(loss_arr.max()),
                    },
                }
                bones = payload.get("bones", None)
                if isinstance(bones, list) and bones:
                    domega_by_bone: Dict[str, Any] = {}
                    for b in bones:
                        vals: List[float] = []
                        for s in direct_leg_omega_grad_steps:
                            pb = s.get("per_bone", {})
                            if isinstance(pb, dict):
                                ent = pb.get(str(b), None)
                                if isinstance(ent, dict) and ent.get("domega_grad_norm", None) is not None:
                                    try:
                                        vals.append(float(ent["domega_grad_norm"]))
                                    except Exception:
                                        pass
                        if vals:
                            v = np.asarray(vals, dtype=np.float32)
                            domega_by_bone[str(b)] = {
                                "mean": float(v.mean()),
                                "p99": float(np.quantile(v, 0.99)),
                                "max": float(v.max()),
                            }
                        else:
                            domega_by_bone[str(b)] = None
                    summary["domega_grad_norm_by_bone"] = domega_by_bone
                payload["summary"] = summary
        except Exception:
            pass
        extra["direct_leg_omega_grad"] = payload
    # Optional: export model internal plan/phase state series (inputs to the next-step predictor).
    if bool(export_plan_state_series):
        try:
            Cc = int(getattr(model, "contact_dim", 0) or 0)
        except Exception:
            Cc = 0
        try:
            plan_dim = int(getattr(model, "contact_plan_hidden", 0) or 0)
        except Exception:
            plan_dim = 0

        def _infer_dim(log: List[Optional[List[float]]], fallback: int) -> int:
            for v in log:
                if isinstance(v, list) and v:
                    return int(len(v))
            return int(fallback)

        plan_dim = _infer_dim(plan_z_in_log, plan_dim)
        phase_dim = _infer_dim(phase_z_in_log, max(0, int(Cc) * 2))
        age_dim = _infer_dim(phase_event_age_in_log, max(0, int(Cc)))

        def _pack_series(log: List[Optional[List[float]]], dim: int) -> Dict[str, Any]:
            out: List[List[float]] = []
            valid: List[int] = []
            for t in range(int(free_steps)):
                v = log[t] if t < len(log) else None
                if isinstance(v, list) and int(len(v)) == int(dim):
                    out.append([float(x) for x in v])
                    valid.append(1)
                else:
                    out.append([0.0 for _ in range(int(dim))] if int(dim) > 0 else [])
                    valid.append(0)
            return {"data": out, "valid": valid, "dim": int(dim)}

        extra["plan_state_series"] = {
            "enabled": bool(plan_enable),
            "contact_dim": int(Cc),
            "plan_z_dim": int(plan_dim),
            "phase_z_dim": int(phase_dim),
            "phase_event_age_dim": int(age_dim),
            "units": {"plan_z_in": "unitless", "phase_z_in": "unitless", "phase_event_age_in": "frames"},
            "note": (
                "Per-step model inputs for contact_plan internal state.\n"
                "- plan_z_in: contact_plan GRU hidden state fed into the model at that step.\n"
                "- phase_z_in: phase sin/cos state fed into the model at that step (layout depends on model; typically 2*contact_dim).\n"
                "- phase_event_age_in: frames since last accepted phase reset event per contact channel.\n"
                "Data is mean-over-batch. Missing values are zero-filled with a corresponding valid mask."
            ),
            "series": {
                "plan_z_in": _pack_series(plan_z_in_log, int(plan_dim)),
                "phase_z_in": _pack_series(phase_z_in_log, int(phase_dim)),
                "phase_event_age_in": _pack_series(phase_event_age_in_log, int(age_dim)),
            },
        }

    # Optional: export per-step direct hinge head output delta (rad/deg).
    if bool(export_direct_hinge_series):
        try:
            hinge_names = list(getattr(model, "direct_pose_hinge_joint_names", None) or [])
        except Exception:
            hinge_names = []
        try:
            hinge_idx = [int(i) for i in (getattr(model, "direct_pose_hinge_joint_idx", None) or [])]
        except Exception:
            hinge_idx = []

        H = 0
        if hinge_names:
            H = int(len(hinge_names))
        else:
            for v in direct_hinge_step_log:
                if isinstance(v, list) and v:
                    H = int(len(v))
                    break
        if H > 0:
            if not hinge_names or len(hinge_names) != H:
                hinge_names = [f"hinge_{i}" for i in range(H)]

            deg = 180.0 / float(np.pi)
            delta_rad: Dict[str, List[float]] = {str(n): [] for n in hinge_names}
            delta_deg: Dict[str, List[float]] = {str(n): [] for n in hinge_names}
            delta_raw_rad: Dict[str, List[float]] = {str(n): [] for n in hinge_names}
            delta_raw_deg: Dict[str, List[float]] = {str(n): [] for n in hinge_names}
            gate_series: Dict[str, List[float]] = {str(n): [] for n in hinge_names}
            have_base_raw = any(
                isinstance(v, list) and int(len(v)) == int(H) for v in direct_hinge_base_raw_step_log
            )
            have_eps_raw = any(
                isinstance(v, list) and int(len(v)) == int(H) for v in direct_hinge_eps_raw_step_log
            )
            delta_base_raw_rad: Dict[str, List[float]] = {str(n): [] for n in hinge_names} if have_base_raw else {}
            delta_base_raw_deg: Dict[str, List[float]] = {str(n): [] for n in hinge_names} if have_base_raw else {}
            delta_eps_raw_rad: Dict[str, List[float]] = {str(n): [] for n in hinge_names} if have_eps_raw else {}
            delta_eps_raw_deg: Dict[str, List[float]] = {str(n): [] for n in hinge_names} if have_eps_raw else {}
            valid: List[int] = []
            valid_raw: List[int] = []
            valid_gate: List[int] = []
            valid_base_raw: List[int] = []
            valid_eps_raw: List[int] = []
            for t in range(int(free_steps)):
                v = direct_hinge_step_log[t] if t < len(direct_hinge_step_log) else None
                ok = isinstance(v, list) and int(len(v)) == int(H)
                valid.append(1 if ok else 0)
                v_raw = direct_hinge_raw_step_log[t] if t < len(direct_hinge_raw_step_log) else None
                ok_raw = isinstance(v_raw, list) and int(len(v_raw)) == int(H)
                valid_raw.append(1 if ok_raw else 0)
                v_base = direct_hinge_base_raw_step_log[t] if t < len(direct_hinge_base_raw_step_log) else None
                ok_base = isinstance(v_base, list) and int(len(v_base)) == int(H)
                if have_base_raw:
                    valid_base_raw.append(1 if ok_base else 0)
                v_eps = direct_hinge_eps_raw_step_log[t] if t < len(direct_hinge_eps_raw_step_log) else None
                ok_eps = isinstance(v_eps, list) and int(len(v_eps)) == int(H)
                if have_eps_raw:
                    valid_eps_raw.append(1 if ok_eps else 0)
                v_gate = direct_hinge_gate_step_log[t] if t < len(direct_hinge_gate_step_log) else None
                ok_gate = isinstance(v_gate, list) and int(len(v_gate)) == int(H)
                valid_gate.append(1 if ok_gate else 0)
                for i, name in enumerate(hinge_names):
                    x = float(v[i]) if ok else 0.0
                    delta_rad[str(name)].append(float(x))
                    delta_deg[str(name)].append(float(x * deg))
                    xr = float(v_raw[i]) if ok_raw else 0.0
                    delta_raw_rad[str(name)].append(float(xr))
                    delta_raw_deg[str(name)].append(float(xr * deg))
                    if have_base_raw:
                        xb = float(v_base[i]) if ok_base else 0.0
                        delta_base_raw_rad[str(name)].append(float(xb))
                        delta_base_raw_deg[str(name)].append(float(xb * deg))
                    if have_eps_raw:
                        xe = float(v_eps[i]) if ok_eps else 0.0
                        delta_eps_raw_rad[str(name)].append(float(xe))
                        delta_eps_raw_deg[str(name)].append(float(xe * deg))
                    g = float(v_gate[i]) if ok_gate else 0.0
                    gate_series[str(name)].append(float(g))

            series: Dict[str, Any] = {
                "delta_rad": delta_rad,
                "delta_deg": delta_deg,
                "valid": valid,
                "delta_raw_rad": delta_raw_rad,
                "delta_raw_deg": delta_raw_deg,
                "valid_raw": valid_raw,
                "gate": gate_series,
                "valid_gate": valid_gate,
            }
            if have_base_raw:
                series["delta_base_raw_rad"] = delta_base_raw_rad
                series["delta_base_raw_deg"] = delta_base_raw_deg
                series["valid_base_raw"] = valid_base_raw
            if have_eps_raw:
                series["delta_eps_raw_rad"] = delta_eps_raw_rad
                series["delta_eps_raw_deg"] = delta_eps_raw_deg
                series["valid_eps_raw"] = valid_eps_raw

            extra["direct_hinge_series"] = {
                "axis": str(getattr(model, "direct_pose_hinge_axis", "Z") or "Z"),
                "max_deg": float(getattr(model, "direct_pose_hinge_max_deg", 0.0) or 0.0),
                "joint_idx": hinge_idx,
                "bones": [str(n) for n in hinge_names],
                "units": {"delta_rad": "rad", "delta_deg": "deg", "gate": "unitless"},
                "note": (
                    "direct_hinge_delta is the model's predicted 1D correction δ on the configured hinge axis (joint-local).\n"
                    "- delta_rad/deg: effective delta applied to direct (may include contact-based gating).\n"
                    "- delta_raw_rad/deg: raw hinge head output before gating (if available).\n"
                    "- gate: per-hinge gating factor in [0,1] (if available).\n"
                    "All series are mean-over-batch per step. Missing values are zero-filled with valid masks."
                ),
                "series": series,
            }

    # Optional: export per-step direct leg SO(3) residual omega (axis-angle), plus ||omega|| diagnostics.
    if bool(export_direct_leg_omega_series):
        try:
            leg_names = list(getattr(model, "direct_pose_leg_joint_names", None) or [])
        except Exception:
            leg_names = []
        try:
            leg_idx = [int(i) for i in (getattr(model, "direct_pose_leg_joint_idx", None) or [])]
        except Exception:
            leg_idx = []

        K = 0
        if leg_names:
            K = int(len(leg_names))
        else:
            for v in direct_leg_omega_step_log:
                if isinstance(v, list) and v and isinstance(v[0], list) and len(v[0]) == 3:
                    K = int(len(v))
                    break
        if K <= 0:
            for v in direct_leg_scale_step_log:
                if isinstance(v, list) and v:
                    K = int(len(v))
                    break
        if K > 0:
            if not leg_names or len(leg_names) != K:
                leg_names = [f"leg_{i}" for i in range(K)]

            deg = 180.0 / float(np.pi)
            max_rad = 0.0
            try:
                max_rad = float(getattr(model, "direct_pose_leg_max_rad", 0.0) or 0.0)
            except Exception:
                max_rad = 0.0
            max_deg = float(max_rad * deg) if max_rad > 0.0 else 0.0

            omega_xyz_rad: Dict[str, List[List[float]]] = {str(n): [] for n in leg_names}
            theta_deg: Dict[str, List[float]] = {str(n): [] for n in leg_names}
            theta_over_max: Dict[str, List[float]] = {str(n): [] for n in leg_names} if max_rad > 0.0 else {}
            valid: List[int] = []
            scale: Dict[str, List[float]] = {str(n): [] for n in leg_names}
            scale_log: Dict[str, List[float]] = {str(n): [] for n in leg_names}
            scale_log_raw: Dict[str, List[float]] = {str(n): [] for n in leg_names}
            valid_scale: List[int] = []
            valid_scale_log: List[int] = []
            valid_scale_log_raw: List[int] = []

            for t in range(int(free_steps)):
                v = direct_leg_omega_step_log[t] if t < len(direct_leg_omega_step_log) else None
                ok = (
                    isinstance(v, list)
                    and int(len(v)) == int(K)
                    and all(isinstance(row, list) and int(len(row)) == 3 for row in v)
                )
                valid.append(1 if ok else 0)
                v_scale = direct_leg_scale_step_log[t] if t < len(direct_leg_scale_step_log) else None
                ok_scale = isinstance(v_scale, list) and int(len(v_scale)) == int(K)
                valid_scale.append(1 if ok_scale else 0)
                v_scale_log = direct_leg_scale_log_step_log[t] if t < len(direct_leg_scale_log_step_log) else None
                ok_scale_log = isinstance(v_scale_log, list) and int(len(v_scale_log)) == int(K)
                valid_scale_log.append(1 if ok_scale_log else 0)
                v_scale_log_raw = (
                    direct_leg_scale_log_raw_step_log[t] if t < len(direct_leg_scale_log_raw_step_log) else None
                )
                ok_scale_log_raw = isinstance(v_scale_log_raw, list) and int(len(v_scale_log_raw)) == int(K)
                valid_scale_log_raw.append(1 if ok_scale_log_raw else 0)
                for i, name in enumerate(leg_names):
                    row = v[i] if ok else [0.0, 0.0, 0.0]
                    x = float(row[0]) if len(row) > 0 else 0.0
                    y = float(row[1]) if len(row) > 1 else 0.0
                    z = float(row[2]) if len(row) > 2 else 0.0
                    omega_xyz_rad[str(name)].append([x, y, z])
                    th = float((x * x + y * y + z * z) ** 0.5)
                    theta_deg[str(name)].append(float(th * deg))
                    if max_rad > 0.0:
                        theta_over_max[str(name)].append(float(th / max_rad))
                    scale[str(name)].append(float(v_scale[i]) if ok_scale else 0.0)
                    scale_log[str(name)].append(float(v_scale_log[i]) if ok_scale_log else 0.0)
                    scale_log_raw[str(name)].append(float(v_scale_log_raw[i]) if ok_scale_log_raw else 0.0)

            # Small summary to quickly spot saturation / spikes.
            sat_thr_ratio = 0.98
            stats: Dict[str, Any] = {}
            for name in leg_names:
                arr = np.asarray(theta_deg[str(name)], dtype=np.float32)
                vm = np.asarray(valid, dtype=bool)
                vals = arr[vm] if arr.size == vm.size else arr
                if vals.size == 0:
                    stats[str(name)] = None
                    continue
                ent: Dict[str, Any] = {
                    "mean_deg": float(vals.mean()),
                    "p99_deg": float(np.quantile(vals, 0.99)),
                    "max_deg": float(vals.max()),
                }
                if max_deg > 0.0 and str(name) in theta_over_max:
                    r = np.asarray(theta_over_max[str(name)], dtype=np.float32)
                    rv = r[vm] if r.size == vm.size else r
                    ent["sat_thr_ratio"] = float(sat_thr_ratio)
                    ent["sat_rate"] = float((rv >= float(sat_thr_ratio)).mean()) if rv.size > 0 else None
                s = np.asarray(scale[str(name)], dtype=np.float32)
                sv = s[np.asarray(valid_scale, dtype=bool)] if s.size == len(valid_scale) else s
                if sv.size > 0:
                    ent["scale_mean"] = float(sv.mean())
                    ent["scale_p99"] = float(np.quantile(sv, 0.99))
                    ent["scale_max"] = float(sv.max())
                stats[str(name)] = ent

            extra["direct_leg_omega_series"] = {
                "enabled": bool(getattr(model, "direct_pose_leg_enable", False)),
                "mode": str(getattr(model, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add"),
                "max_deg": float(max_deg),
                "joint_idx": leg_idx,
                "bones": [str(n) for n in leg_names],
                "units": {
                    "omega_xyz_rad": "rad",
                    "theta_deg": "deg",
                    "theta_over_max": "unitless",
                    "scale": "unitless",
                    "scale_log": "log-scale",
                    "scale_log_raw": "log-scale (pre-clip/pre-clamp)",
                },
                "note": (
                    "direct_leg_omega is the model's predicted axis-angle residual for selected leg joints.\n"
                    "IMPORTANT: the model already applies a smooth tanh-based magnitude bound using direct_pose_leg_max_deg.\n"
                    "Series are mean-over-batch per step; missing values are zero-filled with a valid mask.\n"
                    "When leg gate mode is scale, this export also includes scale diagnostics:\n"
                    "- scale: effective multiplicative scale applied to omega.\n"
                    "- scale_log: effective log-scale after internal clipping/clamp.\n"
                    "- scale_log_raw: raw log-scale head output before clipping/clamp."
                ),
                "series": {
                    "omega_xyz_rad": omega_xyz_rad,
                    "theta_deg": theta_deg,
                    "theta_over_max": theta_over_max,
                    "valid": valid,
                    "scale": scale,
                    "scale_log": scale_log,
                    "scale_log_raw": scale_log_raw,
                    "valid_scale": valid_scale,
                    "valid_scale_log": valid_scale_log,
                    "valid_scale_log_raw": valid_scale_log_raw,
                },
                "stats": stats,
            }

    # Optional: export direct leg head first-layer IO (input vector + first Linear pre-activation).
    if bool(direct_leg_head_io_enabled) and isinstance(direct_leg_head_io, dict) and direct_leg_head_io:
        try:
            steps_sorted = [direct_leg_head_io[t] for t in sorted(direct_leg_head_io.keys())]
            extra["direct_leg_head_io"] = {
                "enabled": True,
                "steps_spec": {
                    "steps": str(direct_leg_omega_alpha_sweep_steps or ""),
                    "sics": str(direct_leg_omega_alpha_sweep_sics or ""),
                    "sic_range": str(direct_leg_omega_alpha_sweep_sic_range or ""),
                },
                "mask": {"cycle_gte": 1, "drop_wrap": True},
                "note": (
                    "Captured via forward hooks on the first Linear layer of the leg head(s).\n"
                    "- baseline: direct_pose_leg_head.0 (or first Linear) IO.\n"
                    "- shared: direct_pose_leg_head_shared.0 (or first Linear) IO; 'r' is the first call, 'l' is the second.\n"
                    "Vectors are mean-over-batch."
                ),
                "steps": steps_sorted,
            }
        except Exception:
            pass

    # Optional: export direct non-leg probe bundle.
    if bool(direct_nonleg_probe_enabled) and isinstance(direct_nonleg_probe_steps, dict) and direct_nonleg_probe_steps:
        try:
            steps_sorted = [direct_nonleg_probe_steps[t] for t in sorted(direct_nonleg_probe_steps.keys())]
            extra["direct_nonleg_probe"] = {
                "enabled": True,
                "bones": [str(x) for x in direct_nonleg_probe_bone_names_sel],
                "joint_idx": [int(x) for x in direct_nonleg_probe_joint_idx_sel],
                "features": ["pre_proj_in", "proj_pre0", "out_in"],
                "target": "rot6d_gt_vs_direct",
                "mask": {
                    "cycle_gte": 1,
                    "drop_wrap": True,
                    "sics": str(direct_nonleg_probe_sics or ""),
                },
                "note": (
                    "Captured via forward hooks for the non-leg branch.\n"
                    "- pre_proj_in: input to direct_pose_nonleg_proj first Linear (shared hidden).\n"
                    "- proj_pre0: first Linear pre-activation of direct_pose_nonleg_proj.\n"
                    "- out_in: input to direct_pose_out_nonleg (post-proj feature).\n"
                    "Targets are mean-over-batch rot6d vectors for selected bones: gt_rot6d vs direct_rot6d."
                ),
                "steps": steps_sorted,
            }
        except Exception:
            pass

    # Optional: export direct arm probe bundle.
    if bool(direct_arm_probe_enabled) and isinstance(direct_arm_probe_steps, dict) and direct_arm_probe_steps:
        try:
            steps_sorted = [direct_arm_probe_steps[t] for t in sorted(direct_arm_probe_steps.keys())]
            extra["direct_arm_probe"] = {
                "enabled": True,
                "bones": [str(x) for x in direct_arm_probe_bone_names_sel],
                "joint_idx": [int(x) for x in direct_arm_probe_joint_idx_sel],
                "features": ["direct_in", "direct_phase", "trunk_hidden", "proj_pre0", "out_in", "arm_out"],
                "target": "rot6d_gt_vs_direct",
                "mask": {
                    "cycle_gte": 1,
                    "drop_wrap": True,
                    "sics": str(direct_arm_probe_sics or ""),
                },
                "note": (
                    "Captured via forward hooks for the arm-split branch.\n"
                    "- direct_in: input to direct_pose_head first Linear (flattened direct conditioning).\n"
                    "- direct_phase: trailing phase_z slice inside direct_in when phase_z is enabled.\n"
                    "- trunk_hidden: output of direct_pose_head shared trunk.\n"
                    "- proj_pre0: first Linear pre-activation of direct_pose_arm_proj.\n"
                    "- out_in: input to direct_pose_out_arm.\n"
                    "- arm_out: output of direct_pose_out_arm before scatter-back.\n"
                    "Targets are mean-over-batch rot6d vectors for selected bones: gt_rot6d vs direct_rot6d."
                ),
                "steps": steps_sorted,
            }
        except Exception:
            pass

    # Remove any debug hooks to avoid leaking across clips.
    if _leg_head_io_handles:
        for h in list(_leg_head_io_handles):
            try:
                h.remove()
            except Exception:
                pass
        _leg_head_io_handles = []
    if direct_nonleg_probe_handles:
        for h in list(direct_nonleg_probe_handles):
            try:
                h.remove()
            except Exception:
                pass
        direct_nonleg_probe_handles = []
    if direct_arm_probe_handles:
        for h in list(direct_arm_probe_handles):
            try:
                h.remove()
            except Exception:
                pass
        direct_arm_probe_handles = []

    # Debug-only: alpha-sweep and oracle alignment for direct_leg_omega.
    # - Uses the *pre-leg-apply* direct output captured during rollout, so the sweep is on a fixed rollout stream.
    # - Also computes omega_oracle from GT: omega_oracle = so3_log_map(R_gt @ R_base^T),
    #   then logs cos(pred, oracle) and ||pred||/||oracle||.
    if bool(export_direct_leg_omega_alpha_sweep):
        try:
            import math
            import train.geometry as _geo

            # Parse alphas.
            alphas: List[float] = []
            for tok in str(direct_leg_omega_alpha_sweep_alphas or "").replace(";", ",").split(","):
                t = tok.strip()
                if not t:
                    continue
                try:
                    v = float(t)
                except Exception:
                    continue
                if math.isfinite(v):
                    alphas.append(float(v))
            if not alphas:
                alphas = [0.0, 0.25, 0.5, 1.0, -1.0]

            # Resolve leg names/indices (k-aligned).
            try:
                leg_names_all = list(getattr(model, "direct_pose_leg_joint_names", None) or [])
            except Exception:
                leg_names_all = []
            try:
                leg_idx_all = [int(i) for i in (getattr(model, "direct_pose_leg_joint_idx", None) or [])]
            except Exception:
                leg_idx_all = []

            K_all = int(len(leg_names_all)) if leg_names_all else int(len(leg_idx_all))

            # If model doesn't expose leg metadata, infer K from the first valid omega tensor.
            if K_all <= 0:
                for v in direct_leg_omega_tensor_step_log:
                    if torch.is_tensor(v) and v.dim() >= 3 and int(v.shape[-1]) == 3:
                        try:
                            vv = v[:, 0] if (v.dim() == 4 and v.size(1) == 1) else v
                            if vv.dim() == 3:
                                K_all = int(vv.shape[1])
                                break
                        except Exception:
                            continue
            if K_all <= 0:
                K_all = 0

            if (not leg_names_all) and K_all > 0:
                leg_names_all = [f"leg_{i}" for i in range(int(K_all))]
            if (not leg_idx_all) and K_all > 0:
                leg_idx_all = list(range(int(K_all)))
            if leg_names_all and len(leg_idx_all) != len(leg_names_all):
                leg_idx_all = leg_idx_all[: len(leg_names_all)]
                K_all = int(len(leg_idx_all))

            # Parse bone selection -> sel_k indices.
            want_raw = str(direct_leg_omega_alpha_sweep_bones or "leg").strip()
            want = [s.strip() for s in want_raw.split(",") if s.strip()]
            want_l = {s.lower() for s in want}
            if not want or (want_l & {"leg", "all"}):
                sel_k = list(range(int(K_all)))
            else:
                name_to_k = {str(n): int(i) for i, n in enumerate(leg_names_all)}
                name_to_k_l = {str(n).lower(): int(i) for i, n in enumerate(leg_names_all)}
                sel_k = []
                for b in want:
                    kk = name_to_k.get(b, None)
                    if kk is None:
                        kk = name_to_k_l.get(str(b).lower(), None)
                    if kk is None:
                        continue
                    sel_k.append(int(kk))
                sel_k = sorted(set(sel_k))

            # Parse step selection.
            T_cycle_i = int(T_cycle) if (rounds > 1 and int(T_cycle) > 0) else 0
            t_sel: set[int] = set()
            for tok in str(direct_leg_omega_alpha_sweep_steps or "").replace(";", ",").split(","):
                s = tok.strip()
                if not s:
                    continue
                # "cycle:sic" shortcut.
                if ":" in s and T_cycle_i > 0:
                    parts = [p.strip() for p in s.split(":") if p.strip()]
                    if len(parts) == 2 and parts[0].lstrip("-").isdigit() and parts[1].lstrip("-").isdigit():
                        try:
                            cyc = int(parts[0])
                            sic = int(parts[1])
                            t_sel.add(int(cyc) * int(T_cycle_i) + int(sic))
                            continue
                        except Exception:
                            pass
                if s.lstrip("-").isdigit():
                    try:
                        t_sel.add(int(s))
                    except Exception:
                        pass

            sic_sel: set[int] = set()
            for tok in str(direct_leg_omega_alpha_sweep_sics or "").replace(";", ",").split(","):
                s = tok.strip()
                if not s:
                    continue
                if s.lstrip("-").isdigit():
                    try:
                        sic_sel.add(int(s))
                    except Exception:
                        pass

            sic_lo = sic_hi = None
            sr = str(direct_leg_omega_alpha_sweep_sic_range or "").strip()
            if sr and any(ch in sr for ch in "-:"):
                sep = "-" if "-" in sr else ":"
                parts = [p.strip() for p in sr.split(sep) if p.strip()]
                if len(parts) == 2 and parts[0].lstrip("-").isdigit() and parts[1].lstrip("-").isdigit():
                    try:
                        sic_lo = int(parts[0])
                        sic_hi = int(parts[1])
                        if sic_lo > sic_hi:
                            sic_lo, sic_hi = sic_hi, sic_lo
                    except Exception:
                        sic_lo = sic_hi = None

            if T_cycle_i > 0 and (sic_sel or (sic_lo is not None and sic_hi is not None)):
                for tt in range(int(free_steps)):
                    cyc = int(tt // int(T_cycle_i))
                    if cyc < 1:
                        continue
                    sic = int(tt % int(T_cycle_i))
                    if sic == int(T_cycle_i) - 1:
                        continue  # drop wrap
                    if sic_sel and sic not in sic_sel:
                        continue
                    if (sic_lo is not None and sic_hi is not None) and not (sic_lo <= sic <= sic_hi):
                        continue
                    t_sel.add(int(tt))

            # Apply standard mask and bounds.
            t_list: List[int] = []
            for tt in sorted(t_sel):
                if tt < 0 or tt >= int(free_steps):
                    continue
                if T_cycle_i > 0:
                    cyc = int(tt // int(T_cycle_i))
                    sic = int(tt % int(T_cycle_i))
                    if cyc < 1:
                        continue
                    if sic == int(T_cycle_i) - 1:
                        continue
                t_list.append(int(tt))

            # Prepare rot slice.
            rot_slice_dbg = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
            if not isinstance(rot_slice_dbg, slice):
                rot_slice_dbg = slice(0, int(gt_raw_full.shape[-1]))
            rot_len = int(rot_slice_dbg.stop - rot_slice_dbg.start)
            if rot_len <= 0 or (rot_len % 6) != 0:
                raise ValueError("Invalid rot6d slice for leg omega alpha sweep.")
            J_dbg = int(rot_len // 6)

            # Root index (exclude if accidentally included).
            try:
                root_idx = int(getattr(getattr(trainer, "loss_fn", None), "root_idx", 0) or 0)
            except Exception:
                root_idx = 0

            out_steps: List[Dict[str, Any]] = []
            oracle_err_any: Optional[str] = None
            oracle_right_err_any: Optional[str] = None
            sweep_err_any: Optional[str] = None
            sweep_right_err_any: Optional[str] = None
            for tt in t_list:
                base_norm = direct_pre_leg_norm_step_log[tt] if tt < len(direct_pre_leg_norm_step_log) else None
                omega_pred = direct_leg_omega_tensor_step_log[tt] if tt < len(direct_leg_omega_tensor_step_log) else None
                if (not torch.is_tensor(base_norm)) or (not torch.is_tensor(omega_pred)):
                    continue

                try:
                    gt_raw_step = gt_raw_full[:, tt]
                except Exception:
                    continue

                # Ensure base/gt/omega are on the same device; otherwise matmul/geodesic will fail under try/except
                # and we'd export all-None alpha-sweep results.
                try:
                    dev = base_norm.device
                    # Use fp32 to avoid matmul dtype mismatch (fp16 vs fp32) and improve numerical stability.
                    base_norm = base_norm.to(device=dev, dtype=torch.float32)
                    omega_pred = omega_pred.to(device=dev, dtype=torch.float32)
                    gt_raw_step = gt_raw_step.to(device=dev, dtype=torch.float32)
                except Exception:
                    pass

                # omega_pred: (B,K,3)
                if omega_pred.dim() == 4 and omega_pred.size(1) == 1:
                    omega_pred = omega_pred[:, 0]
                if omega_pred.dim() != 3 or int(omega_pred.shape[-1]) != 3:
                    continue

                # Denorm the pre-leg direct output.
                try:
                    direct_raw_base = trainer._denorm(base_norm)
                except Exception:
                    continue
                if (not torch.is_tensor(direct_raw_base)) or (not torch.is_tensor(gt_raw_step)):
                    continue
                if direct_raw_base.shape != gt_raw_step.shape:
                    continue
                B_dbg = int(direct_raw_base.shape[0])

                # Convert base/gt to rotation matrices.
                try:
                    base6 = reproject_rot6d(direct_raw_base[..., rot_slice_dbg].reshape(B_dbg, J_dbg, 6))
                    gt6 = reproject_rot6d(gt_raw_step[..., rot_slice_dbg].reshape(B_dbg, J_dbg, 6))
                    R_base = rot6d_to_matrix(base6, columns=cols)  # (B,J,3,3)
                    R_gt = rot6d_to_matrix(gt6, columns=cols)
                except Exception:
                    continue

                # Select valid (k,j) pairs.
                sel_k_use = [int(k) for k in sel_k if 0 <= int(k) < int(K_all) and int(k) < int(omega_pred.shape[1])]
                idx_use = [int(leg_idx_all[int(k)]) for k in sel_k_use]
                keep = [
                    (k, j)
                    for k, j in zip(sel_k_use, idx_use)
                    if 0 <= int(j) < int(J_dbg) and not (0 <= int(root_idx) < int(J_dbg) and int(j) == int(root_idx))
                ]
                if not keep:
                    continue
                sel_k_use = [int(k) for k, _ in keep]
                idx_use = [int(j) for _, j in keep]

                try:
                    R_leg_base = R_base[:, idx_use]  # (B,K,3,3)
                    R_leg_gt = R_gt[:, idx_use]
                    omega_sel = omega_pred[:, sel_k_use, :]  # (B,K,3)
                except Exception:
                    continue

                # Oracle omega (standard axis-angle):
                #   left : omega_oracle_left  = log(R_gt @ R_base^T)      (matches left-mul apply: Exp(ω) @ R_base)
                #   right: omega_oracle_right = log(R_base^T @ R_gt)      (matches right-mul apply: R_base @ Exp(ω))
                omega_oracle = None
                omega_oracle_right = None
                try:
                    R_delta_oracle = torch.matmul(R_leg_gt, R_leg_base.transpose(-1, -2))
                    omega_oracle = _geo.so3_log_map(R_delta_oracle)  # (B,K,3)
                except Exception as e:
                    omega_oracle = None
                    if oracle_err_any is None:
                        oracle_err_any = repr(e)
                try:
                    R_delta_oracle_right = torch.matmul(R_leg_base.transpose(-1, -2), R_leg_gt)
                    omega_oracle_right = _geo.so3_log_map(R_delta_oracle_right)  # (B,K,3)
                except Exception as e:
                    omega_oracle_right = None
                    if oracle_right_err_any is None:
                        oracle_right_err_any = repr(e)

                # Precompute norms for oracle alignment.
                n_pred = omega_sel.norm(dim=-1)  # (B,K)
                if torch.is_tensor(omega_oracle):
                    n_or = omega_oracle.norm(dim=-1)
                    dot = (omega_sel * omega_oracle).sum(dim=-1)
                    denom = (n_pred * n_or).clamp_min(1e-8)
                    cos_po = (dot / denom).clamp(-1.0, 1.0)
                    cos_mean = cos_po.mean(dim=0)  # (K,)
                    ratio_mean = (n_pred / (n_or + 1e-8)).mean(dim=0)  # (K,)
                    theta_or_deg = (n_or.mean(dim=0) * (180.0 / math.pi))  # (K,)
                else:
                    cos_mean = None
                    ratio_mean = None
                    theta_or_deg = None
                if torch.is_tensor(omega_oracle_right):
                    n_or_r = omega_oracle_right.norm(dim=-1)
                    dot_r = (omega_sel * omega_oracle_right).sum(dim=-1)
                    denom_r = (n_pred * n_or_r).clamp_min(1e-8)
                    cos_po_r = (dot_r / denom_r).clamp(-1.0, 1.0)
                    cos_mean_r = cos_po_r.mean(dim=0)  # (K,)
                    ratio_mean_r = (n_pred / (n_or_r + 1e-8)).mean(dim=0)  # (K,)
                    theta_or_deg_r = (n_or_r.mean(dim=0) * (180.0 / math.pi))  # (K,)
                else:
                    cos_mean_r = None
                    ratio_mean_r = None
                    theta_or_deg_r = None

                theta_pred_deg = n_pred.mean(dim=0) * (180.0 / math.pi)  # (K,)

                # Alpha sweep errors per joint.
                try:
                    errs_per_alpha = []  # left-mul: Exp(a*ω) @ R_base
                    errs_per_alpha_right = []  # right-mul: R_base @ Exp(a*ω)
                    for a in alphas:
                        R_delta = _geo.so3_exp_map(omega_sel * float(a))  # (B,K,3,3)
                        R_leg_a = torch.matmul(R_delta, R_leg_base)
                        R_leg_a_r = torch.matmul(R_leg_base, R_delta)
                        e = geodesic_R(R_leg_a, R_leg_gt) * (180.0 / math.pi)  # (B,K)
                        e_r = geodesic_R(R_leg_a_r, R_leg_gt) * (180.0 / math.pi)
                        errs_per_alpha.append(e.mean(dim=0))  # (K,)
                        errs_per_alpha_right.append(e_r.mean(dim=0))
                except Exception as e:
                    errs_per_alpha = []
                    errs_per_alpha_right = []
                    if sweep_err_any is None:
                        sweep_err_any = repr(e)
                    if sweep_right_err_any is None:
                        sweep_right_err_any = repr(e)

                # Candidate indices (for A/B/C/D diagnostics) if present in alphas.
                try:
                    idx_a_pos = next((i for i, a in enumerate(alphas) if abs(float(a) - 1.0) < 1e-12), None)
                except Exception:
                    idx_a_pos = None
                try:
                    idx_a_neg = next((i for i, a in enumerate(alphas) if abs(float(a) + 1.0) < 1e-12), None)
                except Exception:
                    idx_a_neg = None

                per_bone: Dict[str, Any] = {}
                for ii, (k_idx, j_idx) in enumerate(zip(sel_k_use, idx_use)):
                    name = leg_names_all[int(k_idx)] if 0 <= int(k_idx) < len(leg_names_all) else f"leg_{int(k_idx)}"
                    alpha_errs: List[Optional[float]] = []  # left
                    alpha_errs_r: List[Optional[float]] = []  # right
                    for ea in errs_per_alpha:
                        try:
                            alpha_errs.append(float(ea[ii].item()))
                        except Exception:
                            alpha_errs.append(None)
                    for ea in errs_per_alpha_right:
                        try:
                            alpha_errs_r.append(float(ea[ii].item()))
                        except Exception:
                            alpha_errs_r.append(None)
                    if len(alpha_errs) != len(alphas):
                        alpha_errs = [None for _ in alphas]
                    if len(alpha_errs_r) != len(alphas):
                        alpha_errs_r = [None for _ in alphas]
                    best_alpha = best_err = None
                    best_alpha_r = best_err_r = None
                    try:
                        pairs = [(float(e), float(a)) for e, a in zip(alpha_errs, alphas) if e is not None]
                        if pairs:
                            best_err, best_alpha = min(pairs, key=lambda x: x[0])
                    except Exception:
                        best_alpha = best_err = None
                    try:
                        pairs_r = [(float(e), float(a)) for e, a in zip(alpha_errs_r, alphas) if e is not None]
                        if pairs_r:
                            best_err_r, best_alpha_r = min(pairs_r, key=lambda x: x[0])
                    except Exception:
                        best_alpha_r = best_err_r = None

                    # A/B/C/D candidates: compare left/right × +/-omega at alpha=1.
                    candidates: Dict[str, Optional[float]] = {}
                    if idx_a_pos is not None and 0 <= int(idx_a_pos) < len(alphas):
                        candidates["A_left_pos"] = alpha_errs[int(idx_a_pos)]
                        candidates["C_right_pos"] = alpha_errs_r[int(idx_a_pos)]
                    if idx_a_neg is not None and 0 <= int(idx_a_neg) < len(alphas):
                        candidates["B_left_neg"] = alpha_errs[int(idx_a_neg)]
                        candidates["D_right_neg"] = alpha_errs_r[int(idx_a_neg)]
                    best_cand = None
                    try:
                        cand_items = [(k, v) for k, v in candidates.items() if v is not None]
                        if cand_items:
                            best_cand = min(cand_items, key=lambda kv: float(kv[1]))[0]
                    except Exception:
                        best_cand = None

                    per_bone[str(name)] = {
                        "joint_idx": int(j_idx),
                        # Export mean omega vectors (rad) for axis-level diagnostics.
                        # omega_oracle already uses the standard axis-angle semantics expected by
                        # so3_exp_map(omega_pred).
                        "omega_pred_xyz_rad": (
                            [float(v.item()) for v in omega_sel[:, ii, :].mean(dim=0)]
                            if torch.is_tensor(omega_sel) and omega_sel.dim() == 3 and omega_sel.shape[1] > ii
                            else None
                        ),
                        "omega_oracle_xyz_rad": (
                            [float(v.item()) for v in omega_oracle[:, ii, :].mean(dim=0)]
                            if torch.is_tensor(omega_oracle) and omega_oracle.dim() == 3 and omega_oracle.shape[1] > ii
                            else None
                        ),
                        "omega_oracle_right_xyz_rad": (
                            [float(v.item()) for v in omega_oracle_right[:, ii, :].mean(dim=0)]
                            if torch.is_tensor(omega_oracle_right)
                            and omega_oracle_right.dim() == 3
                            and omega_oracle_right.shape[1] > ii
                            else None
                        ),
                        "theta_pred_deg": float(theta_pred_deg[ii].item()) if theta_pred_deg.numel() > ii else None,
                        "theta_oracle_deg": float(theta_or_deg[ii].item()) if torch.is_tensor(theta_or_deg) and theta_or_deg.numel() > ii else None,
                        "cos_pred_oracle": float(cos_mean[ii].item()) if torch.is_tensor(cos_mean) and cos_mean.numel() > ii else None,
                        "norm_ratio_pred_over_oracle": float(ratio_mean[ii].item())
                        if torch.is_tensor(ratio_mean) and ratio_mean.numel() > ii
                        else None,
                        "theta_oracle_right_deg": float(theta_or_deg_r[ii].item())
                        if torch.is_tensor(theta_or_deg_r) and theta_or_deg_r.numel() > ii
                        else None,
                        "cos_pred_oracle_right": float(cos_mean_r[ii].item())
                        if torch.is_tensor(cos_mean_r) and cos_mean_r.numel() > ii
                        else None,
                        "norm_ratio_pred_over_oracle_right": float(ratio_mean_r[ii].item())
                        if torch.is_tensor(ratio_mean_r) and ratio_mean_r.numel() > ii
                        else None,
                        "alpha_geolocal_deg": alpha_errs,
                        "alpha_geolocal_deg_right": alpha_errs_r,
                        "best_alpha": best_alpha,
                        "best_geolocal_deg": best_err,
                        "best_alpha_right": best_alpha_r,
                        "best_geolocal_deg_right": best_err_r,
                        "candidates_abcd": candidates,
                        "best_candidate_abcd": best_cand,
                    }

                out_steps.append(
                    {
                        "step": int(tt),
                        "cycle": int(tt // int(T_cycle_i)) if int(T_cycle_i) > 0 else None,
                        "step_in_cycle": int(tt % int(T_cycle_i)) if int(T_cycle_i) > 0 else None,
                        "wrap_boundary_step": bool(int(T_cycle_i) > 0 and (int(tt) % int(T_cycle_i)) == (int(T_cycle_i) - 1)),
                        "per_bone": per_bone,
                    }
                )

            extra["direct_leg_omega_alpha_sweep"] = {
                "enabled": True,
                "alphas": [float(a) for a in alphas],
                "bones": [str(leg_names_all[int(k)]) if 0 <= int(k) < len(leg_names_all) else f"leg_{int(k)}" for k in sel_k],
                "steps_spec": {
                    "steps": str(direct_leg_omega_alpha_sweep_steps or ""),
                    "sics": str(direct_leg_omega_alpha_sweep_sics or ""),
                    "sic_range": str(direct_leg_omega_alpha_sweep_sic_range or ""),
                    "bones": str(direct_leg_omega_alpha_sweep_bones or ""),
                },
                "mask": {"cycle_gte": 1, "drop_wrap": True},
                "debug": {
                    "oracle_err_any": oracle_err_any,
                    "oracle_right_err_any": oracle_right_err_any,
                    "sweep_err_any": sweep_err_any,
                    "sweep_right_err_any": sweep_right_err_any,
                },
                "note": (
                    "alpha_geolocal_deg[b][i] is DirectGeoLocalDeg for that bone under exp(alpha[i]*omega_pred) @ R_base.\n"
                    "alpha_geolocal_deg_right[b][i] is DirectGeoLocalDeg for that bone under R_base @ exp(alpha[i]*omega_pred).\n"
                    "R_base is the pre-leg-apply direct output (after hinge, before leg omega) on the same rollout stream.\n"
                    "omega_pred_xyz_rad / omega_oracle_xyz_rad / omega_oracle_right_xyz_rad are mean-over-batch omega vectors (rad).\n"
                    "omega_oracle is derived from GT:\n"
                    "  - left : omega_oracle  = so3_log_map(R_gt @ R_base^T)\n"
                    "  - right: omega_oracle_right = so3_log_map(R_base^T @ R_gt)"
                ),
                "steps": out_steps,
            }
        except Exception:
            pass

    # Multi-cycle cycle-start mismatch diagnostics: pose_history buffer vs teacher pose_hist.
    # This isolates "external phase reset vs internal history carry" at wrap seams.
    try:
        cycle_start_mismatch: List[Dict[str, Any]] = []
        if rounds > 1 and int(T_cycle) > 0 and pose_hist_enabled and pose_hist_stride > 0:
            if (
                torch.is_tensor(pose_hist_seq)
                and pose_hist_seq.dim() == 3
                and torch.is_tensor(scales)
                and torch.is_tensor(pred_raw_full)
                and pred_raw_full.dim() == 3
            ):
                # Match the actual rollout state update stream.
                y_used_raw_full = (
                    pred_blend_raw_full
                    if bool(lambda_fusion_apply) and pred_blend_raw_full is not None
                    else pred_raw_full
                )
                rot6d_y_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
                if not isinstance(rot6d_y_slice, slice):
                    rot6d_y_slice = slice(0, y_used_raw_full.shape[-1])

                buf_norm = pose_hist_seq[:, 0]
                buf_raw = pose_hist_inverse_vec(buf_norm, scales, mu, std)
                for t in range(int(free_steps)):
                    if int(t) > 0 and (int(t) % int(T_cycle)) == 0:
                        k = int(t) // int(T_cycle)
                        teacher_norm = pose_hist_seq[:, t] if t < int(pose_hist_seq.size(1)) else None
                        if torch.is_tensor(teacher_norm):
                            diff = (buf_norm - teacher_norm).detach()
                            diff = torch.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=-1e6)
                            entry = {
                                "cycle": int(k),
                                "step": int(t),
                                "pose_hist_norm_abs_mean": float(diff.abs().mean().item()),
                                "pose_hist_norm_rmse": float((diff.pow(2).mean().sqrt()).item()),
                            }
                            # State mismatch (predX vs teacher) is captured at the wrap boundary step (t-1).
                            if root_pos_err is not None and int(t) - 1 < int(root_pos_err.numel()):
                                entry["root_pos_err_at_wrap_m"] = float(root_pos_err[int(t) - 1].item())
                            if root_vel_mae is not None and int(t) - 1 < int(root_vel_mae.numel()):
                                entry["root_vel_mae_at_wrap"] = float(root_vel_mae[int(t) - 1].item())
                            cycle_start_mismatch.append(entry)

                    # Advance buffer using the rollout-updated pose for this step.
                    buf_raw = torch.roll(buf_raw, shifts=-pose_hist_stride, dims=-1)
                    buf_raw[..., -pose_hist_stride:] = y_used_raw_full[:, t, rot6d_y_slice]
                    buf_norm = pose_hist_transform_vec(buf_raw, scales, mu, std)

        if cycle_start_mismatch:
            extra["multicycle_cycle_start_mismatch"] = cycle_start_mismatch
    except Exception:
        pass

    # Optional: per-bone geodesic error for key bones (same set as training diag)
    loss_fn = getattr(trainer, "loss_fn", None)
    bone_names = getattr(loss_fn, "bone_names", None) if loss_fn is not None else None
    if not bone_names:
        bone_names = getattr(trainer, "_bone_names", None)
    if not bone_names:
        bundle_meta = getattr(trainer, "_bundle_meta", None)
        if isinstance(bundle_meta, dict):
            bone_names = bundle_meta.get("bone_names") or bundle_meta.get("skeleton", {}).get("bone_names")
    bone_names = [str(b) for b in bone_names] if isinstance(bone_names, (list, tuple)) else []
    if not bone_names:
        key_bone_names: List[str] = []
        key_indices: List[int] = []
    else:
        key_bone_names_full = list(getattr(loss_fn, "eval_key_bones", None) or [
            "pelvis",
            "upperarm_l", "lowerarm_l", "hand_l",
            "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l",
            "thigh_r", "calf_r", "foot_r",
        ])
        # Extra distal foot markers for phase-locked spike / hinge diagnostics (no-op if absent).
        for extra_bone in ("ball_l", "ball_r", "toe_l", "toe_r"):
            if extra_bone not in key_bone_names_full:
                key_bone_names_full.append(extra_bone)
        idx_map = {name: idx for idx, name in enumerate(bone_names)}
        key_pairs = [(name, idx_map[name]) for name in key_bone_names_full if name in idx_map]
        key_bone_names = [name for name, _ in key_pairs]
        key_indices = [idx for _, idx in key_pairs]

    # Optional: FK position errors for key bones (computed from X state: predX_raw vs gtX_raw).
    # This aligns with RootPosErr semantics (predX is next-state aligned to GT at t+1).
    pos_err_full_world: Optional[torch.Tensor] = None    # (free_steps, J) in meters, world frame (includes root translation drift)
    pos_err_full_rootrel: Optional[torch.Tensor] = None  # (free_steps, J) in meters, root-translation removed
    if bool(export_keybone_pos_err) and predX_raw is not None and gtX_raw is not None:
        try:
            try:
                from train.geometry import fk_positions_from_rot6d
            except ImportError:  # pragma: no cover
                from geometry import fk_positions_from_rot6d

            rot_x_sl = getattr(trainer, "rot6d_x_slice", None)
            root_x_sl = getattr(trainer, "rootpos_x_slice", None)
            parents = getattr(loss_fn, "parents", None)
            offsets = getattr(loss_fn, "bone_offsets", None)
            cols = getattr(loss_fn, "_rot6d_columns", ("X", "Z"))

            if (
                isinstance(rot_x_sl, slice)
                and torch.is_tensor(offsets)
                and parents is not None
                and (rot_x_sl.stop - rot_x_sl.start) > 0
                and ((rot_x_sl.stop - rot_x_sl.start) % 6) == 0
            ):
                Jx = int((rot_x_sl.stop - rot_x_sl.start) // 6)
                if Jx > 0:
                    pr6 = predX_raw[..., rot_x_sl].view(predX_raw.shape[0], free_steps, Jx, 6)
                    gt6 = gtX_raw[..., rot_x_sl].view(gtX_raw.shape[0], free_steps, Jx, 6)
                    pr6 = reproject_rot6d(pr6)
                    gt6 = reproject_rot6d(gt6)

                    if isinstance(root_x_sl, slice) and (root_x_sl.stop - root_x_sl.start) == 3:
                        root_p = predX_raw[..., root_x_sl]
                        root_g = gtX_raw[..., root_x_sl]
                    else:
                        root_p = pr6.new_zeros((pr6.shape[0], pr6.shape[1], 3))
                        root_g = gt6.new_zeros((gt6.shape[0], gt6.shape[1], 3))

                    pos_p = fk_positions_from_rot6d(pr6, parents, offsets, root_pos=root_p, columns=cols)
                    pos_g = fk_positions_from_rot6d(gt6, parents, offsets, root_pos=root_g, columns=cols)
                    # World-frame L2 error (includes root translation drift; can jump at cycle wraps).
                    pos_err_full_world = torch.norm(pos_p - pos_g, dim=-1).mean(dim=0)  # (T,J)

                    # Root-relative (remove root translation): compare (p - p_root) to isolate pose FK error.
                    try:
                        root_idx = int(getattr(loss_fn, "root_idx", 0) or 0)
                    except Exception:
                        root_idx = 0
                    if 0 <= int(root_idx) < int(pos_p.shape[-2]):
                        p0 = pos_p[..., int(root_idx): int(root_idx) + 1, :]  # (B,T,1,3)
                        g0 = pos_g[..., int(root_idx): int(root_idx) + 1, :]
                        pos_err_full_rootrel = torch.norm((pos_p - p0) - (pos_g - g0), dim=-1).mean(dim=0)  # (T,J)
        except Exception:
            pos_err_full_world = None
            pos_err_full_rootrel = None
    # Per-step metrics across the whole free-run (helps locate drift frame)
    if width > 0 and width % 6 == 0:
        J = width // 6
        root_idx = int(getattr(trainer, "eval_root_idx", 0) or 0)
        root_idx = max(0, min(J - 1, root_idx))
        joint_mask = torch.ones(J, device=device, dtype=torch.bool)
        joint_mask[root_idx] = False
        pr6_full = pred_raw_full[..., rot_slice].view(1, free_steps, J, 6)
        gt6_full = gt_raw_full[..., rot_slice].view(1, free_steps, J, 6)
        pr6_full = reproject_rot6d(pr6_full)
        gt6_full = reproject_rot6d(gt6_full)
        Rp_full = rot6d_to_matrix(pr6_full)  # [1, free_steps, J, 3, 3]
        Rg_full = rot6d_to_matrix(gt6_full)
        geo_full = geodesic_R(Rp_full, Rg_full) * deg_factor  # [1, free_steps, J]
        # Simple numeric diff on reprojected 6D values (debug-only; unitless).
        rot6d_l2_full = torch.norm(pr6_full - gt6_full, dim=-1)  # [1, free_steps, J]
        rot6d_local_l2_full = rot6d_l2_full
        geo_full_aligned0 = None
        # Constant root0 alignment (mirrors Trainer._diagnose_free_run_impl behavior for GeoDeg).
        # This suppresses the initial global frame mismatch and focuses on drift.
        if free_steps > 0:
            try:
                Rpr0 = Rp_full[:, 0, root_idx]  # [1,3,3]
                Rgr0 = Rg_full[:, 0, root_idx]
                R_align = torch.matmul(Rgr0, Rpr0.transpose(-1, -2))  # [1,3,3]
                Rp_aligned = torch.matmul(
                    R_align.view(1, 1, 1, 3, 3).expand_as(Rp_full),
                    Rp_full,
                )
                geo_full_aligned0 = geodesic_R(Rp_aligned, Rg_full) * deg_factor
            except Exception:
                geo_full_aligned0 = None
        # Pose-only geodesic per joint. "GeoLocal*" aggregates this while excluding the root joint,
        # so the metric reflects BoneRotations6D pose quality and is not dominated by root/motion.
        geo_local_full = geo_full  # [1, free_steps, J]

        # Blend diagnostics: absolute pose after SO(3) fusion (used for rollout update if enabled).
        geo_blend_full = None
        geo_blend_full_aligned0 = None
        geo_blend_local_full = None
        rot6d_blend_l2_full = None
        rot6d_blend_local_l2_full = None
        if pred_blend_raw_full is not None:
            try:
                b6_full = pred_blend_raw_full[..., rot_slice].view(1, free_steps, J, 6)
                b6_full = reproject_rot6d(b6_full)
                Rb_full = rot6d_to_matrix(b6_full)  # [1, free_steps, J, 3, 3]
                geo_blend_full = geodesic_R(Rb_full, Rg_full) * deg_factor
                rot6d_blend_l2_full = torch.norm(b6_full - gt6_full, dim=-1)
                rot6d_blend_local_l2_full = rot6d_blend_l2_full
                if free_steps > 0:
                    try:
                        Rbr0 = Rb_full[:, 0, root_idx]
                        Rgr0 = Rg_full[:, 0, root_idx]
                        R_align_b = torch.matmul(Rgr0, Rbr0.transpose(-1, -2))
                        Rb_aligned = torch.matmul(
                            R_align_b.view(1, 1, 1, 3, 3).expand_as(Rb_full),
                            Rb_full,
                        )
                        geo_blend_full_aligned0 = geodesic_R(Rb_aligned, Rg_full) * deg_factor
                    except Exception:
                        geo_blend_full_aligned0 = None
                geo_blend_local_full = geo_blend_full
            except Exception:
                geo_blend_full = None
                geo_blend_full_aligned0 = None
                geo_blend_local_full = None
                rot6d_blend_l2_full = None
                rot6d_blend_local_l2_full = None

        # Direct head diagnostics (if available): absolute pose prediction that does NOT use y_{t-1}.
        geo_direct_full = None
        geo_direct_full_aligned0 = None
        geo_direct_local_full = None
        rot6d_direct_l2_full = None
        rot6d_direct_local_l2_full = None
        geo_direct_full_align_inc0 = None
        geo_direct_local_full_align_inc0 = None
        Rd_full = None
        if pred_direct_raw_full is not None:
            try:
                d6_full = pred_direct_raw_full[..., rot_slice].view(1, free_steps, J, 6)
                d6_full = reproject_rot6d(d6_full)
                Rd_full = rot6d_to_matrix(d6_full)  # [1, free_steps, J, 3, 3]
                geo_direct_full = geodesic_R(Rd_full, Rg_full) * deg_factor
                rot6d_direct_l2_full = torch.norm(d6_full - gt6_full, dim=-1)
                rot6d_direct_local_l2_full = rot6d_direct_l2_full
                if free_steps > 0:
                    try:
                        Rdr0 = Rd_full[:, 0, root_idx]
                        Rgr0 = Rg_full[:, 0, root_idx]
                        R_align_d = torch.matmul(Rgr0, Rdr0.transpose(-1, -2))
                        Rd_aligned = torch.matmul(
                            R_align_d.view(1, 1, 1, 3, 3).expand_as(Rd_full),
                            Rd_full,
                        )
                        geo_direct_full_aligned0 = geodesic_R(Rd_aligned, Rg_full) * deg_factor
                    except Exception:
                        geo_direct_full_aligned0 = None
                geo_direct_local_full = geo_direct_full
                if bool(direct_align_inc0) and free_steps > 0:
                    try:
                        # Per-joint constant bias at step0: R_bias[j] = R_inc0[j] @ R_dir0[j]^T
                        R_bias = torch.matmul(Rp_full[:, 0], Rd_full[:, 0].transpose(-1, -2))  # [B,J,3,3]
                        Rd_inc0_aligned = torch.matmul(R_bias.unsqueeze(1), Rd_full)  # [B,T,J,3,3]
                        geo_direct_full_align_inc0 = geodesic_R(Rd_inc0_aligned, Rg_full) * deg_factor
                        geo_direct_local_full_align_inc0 = geo_direct_full_align_inc0
                    except Exception:
                        geo_direct_full_align_inc0 = None
                        geo_direct_local_full_align_inc0 = None
            except Exception:
                geo_direct_full = None
                geo_direct_full_aligned0 = None
                geo_direct_local_full = None
                rot6d_direct_l2_full = None
                rot6d_direct_local_l2_full = None
                geo_direct_full_align_inc0 = None
                geo_direct_local_full_align_inc0 = None

        # Training diagnostics use joint weights (unified/hierarchy weights) for GeoLocalDeg.
        # Keep both:
        #   - GeoLocalDeg: unweighted mean over all joints (debug-friendly)
        #   - GeoLocalDegWeighted: weighted mean matching Trainer._diagnose_free_run_impl
        joint_weights = None
        weights_sum = None
        w_joint = None
        try:
            joint_weights = trainer._joint_weights(Rp_full, J)  # [J]
            if 0 <= root_idx < joint_weights.numel():
                joint_weights = joint_weights.clone()
                joint_weights[root_idx] = 0.0
            weights_sum = joint_weights.sum().clamp_min(1e-6)
            w_joint = joint_weights.view(1, 1, -1)  # [1,1,J]
        except Exception:
            joint_weights = None
            weights_sum = None
            w_joint = None

        # Optional: FK position errors for key bones computed from pose outputs (Y) in a root-relative frame.
        # This is useful to tell whether joint-space sign flips actually matter for end-effectors (foot/ball).
        # NOTE: This is intentionally separate from KeyBonePosErr* which is derived from X (state) to align with RootPosErr.
        pos_err_direct_rootrel_y: Optional[torch.Tensor] = None  # (free_steps, J) in meters
        pos_err_blend_rootrel_y: Optional[torch.Tensor] = None   # (free_steps, J) in meters
        if bool(export_keybone_pos_err):
            try:
                try:
                    from train.geometry import fk_positions_from_rot6d
                except ImportError:  # pragma: no cover
                    from geometry import fk_positions_from_rot6d

                parents = getattr(loss_fn, "parents", None)
                offsets = getattr(loss_fn, "bone_offsets", None)
                cols = getattr(loss_fn, "_rot6d_columns", ("X", "Z"))
                if parents is not None and torch.is_tensor(offsets):
                    # Root-relative positions (root translation removed): compare FK with root at origin.
                    pos_gt = fk_positions_from_rot6d(gt6_full, parents, offsets, root_pos=None, columns=cols)

                    if pred_direct_raw_full is not None:
                        d6_fk = pred_direct_raw_full[..., rot_slice].view(1, free_steps, J, 6)
                        d6_fk = reproject_rot6d(d6_fk)
                        pos_d = fk_positions_from_rot6d(d6_fk, parents, offsets, root_pos=None, columns=cols)
                        pos_err_direct_rootrel_y = torch.norm(pos_d - pos_gt, dim=-1).mean(dim=0)  # (T,J)

                    if pred_blend_raw_full is not None:
                        b6_fk = pred_blend_raw_full[..., rot_slice].view(1, free_steps, J, 6)
                        b6_fk = reproject_rot6d(b6_fk)
                        pos_b = fk_positions_from_rot6d(b6_fk, parents, offsets, root_pos=None, columns=cols)
                        pos_err_blend_rootrel_y = torch.norm(pos_b - pos_gt, dim=-1).mean(dim=0)  # (T,J)
            except Exception:
                pos_err_direct_rootrel_y = None
                pos_err_blend_rootrel_y = None

        # ---- Phase-shift analysis (per-cycle circular shift) -----------------
        # Diagnose "phase drift" by finding the best circular time shift per cycle that aligns:
        #   - contacts_plan / contacts_meas to GT contacts
        #   - direct pose (Rd_full) to GT pose (Rg_full) by minimizing GeoLocalDeg
        if bool(analyze_phase_shift):
            try:
                L = int(T_cycle)
                if L > 1 and int(free_steps) >= 2:
                    full_cycles = int(int(free_steps) // L) if L > 0 else 0
                    max_shift = None
                    if phase_shift_max is not None:
                        try:
                            ms = int(phase_shift_max)
                        except Exception:
                            ms = None
                        if ms is not None:
                            ms = max(0, min(int(ms), L - 1))
                            max_shift = ms

                    def _extract_contact_series(key: str) -> Optional[torch.Tensor]:
                        if not contact_steps:
                            return None
                        out = []
                        for rec in contact_steps[: int(free_steps)]:
                            if not isinstance(rec, dict):
                                return None
                            v = rec.get(key, None)
                            if not isinstance(v, (list, tuple)):
                                return None
                            try:
                                out.append([float(x) for x in v])
                            except Exception:
                                return None
                        if not out:
                            return None
                        try:
                            return torch.as_tensor(out, dtype=torch.float32)
                        except Exception:
                            return None

                    gt_c = _extract_contact_series("ContactGTPerC")
                    plan_c = _extract_contact_series("ContactPlanPerC")
                    meas_c = _extract_contact_series("ContactMeasPerC")

                    def _best_shift_mse(pred_seg: torch.Tensor, gt_seg: torch.Tensor) -> Dict[str, Any]:
                        # pred_seg/gt_seg: (L,C)
                        idx = torch.arange(int(pred_seg.shape[0]))
                        if max_shift is None:
                            candidates = list(range(0, int(pred_seg.shape[0])))
                        else:
                            candidates = list(range(-int(max_shift), int(max_shift) + 1))
                        best_s = 0
                        best_mse = None
                        mse0 = None
                        for s in candidates:
                            idxp = (idx + int(s)) % int(pred_seg.shape[0])
                            mse = float((pred_seg[idxp] - gt_seg).pow(2).mean().item())
                            if int(s) == 0:
                                mse0 = mse
                            if best_mse is None or mse < best_mse:
                                best_mse = mse
                                best_s = int(s)
                        if max_shift is None:
                            # normalize to signed shortest representation
                            mod = int(best_s) % int(pred_seg.shape[0])
                            signed = mod if mod <= (int(pred_seg.shape[0]) // 2) else (mod - int(pred_seg.shape[0]))
                        else:
                            signed = int(best_s)
                            mod = int(best_s) % int(pred_seg.shape[0])
                        return {
                            "shift": int(signed),
                            "shift_mod": int(mod),
                            "mse": float(best_mse) if best_mse is not None else None,
                            "mse0": float(mse0) if mse0 is not None else None,
                        }

                    def _direct_mean_geo_local_for_shift(cycle_start: int, shift: int) -> Optional[float]:
                        if Rd_full is None or geo_direct_local_full is None:
                            return None
                        if not isinstance(joint_mask, torch.Tensor) or not joint_mask.any():
                            return None
                        idx0 = torch.arange(L, device=Rd_full.device, dtype=torch.long)
                        idx_gt = idx0 + int(cycle_start)
                        idx_pred = ((idx0 + (int(shift) % L)) % L) + int(cycle_start)
                        geo = geodesic_R(Rd_full[:, idx_pred], Rg_full[:, idx_gt]) * deg_factor  # (1,L,J)
                        return float(geo[..., joint_mask].mean().item())

                    def _best_shift_direct_geo_local(cycle_start: int) -> Dict[str, Any]:
                        if Rd_full is None or geo_direct_local_full is None:
                            return {}
                        idx = torch.arange(L, device=Rd_full.device, dtype=torch.long)
                        if max_shift is None:
                            candidates = list(range(0, L))
                        else:
                            candidates = list(range(-int(max_shift), int(max_shift) + 1))
                        base = _direct_mean_geo_local_for_shift(cycle_start, 0)
                        best_s = 0
                        best_v = None
                        for s in candidates:
                            v = _direct_mean_geo_local_for_shift(cycle_start, int(s))
                            if v is None:
                                continue
                            if best_v is None or float(v) < best_v:
                                best_v = float(v)
                                best_s = int(s)
                        if max_shift is None:
                            mod = int(best_s) % L
                            signed = mod if mod <= (L // 2) else (mod - L)
                        else:
                            signed = int(best_s)
                            mod = int(best_s) % L
                        return {
                            "shift": int(signed),
                            "shift_mod": int(mod),
                            "geo_local_deg_mean0": float(base) if base is not None else None,
                            "geo_local_deg_mean": float(best_v) if best_v is not None else None,
                        }

                    phase_cycles: List[Dict[str, Any]] = []
                    for cy in range(int(full_cycles)):
                        start = int(cy) * int(L)
                        end = start + int(L)
                        if end > int(free_steps):
                            break
                        entry: Dict[str, Any] = {
                            "cycle": int(cy),
                            "start_step": int(start),
                            "end_step": int(end - 1),
                        }

                        # Contacts shifts (if logged)
                        if gt_c is not None:
                            gt_seg = gt_c[start:end]
                            if plan_c is not None:
                                entry["contact_plan"] = _best_shift_mse(plan_c[start:end], gt_seg)
                            if meas_c is not None:
                                entry["contact_meas"] = _best_shift_mse(meas_c[start:end], gt_seg)

                        # Direct pose shift (if available)
                        if Rd_full is not None and geo_direct_local_full is not None:
                            entry["direct_pose"] = _best_shift_direct_geo_local(start)
                            # Evaluate direct error at the contact-derived shifts (if present).
                            try:
                                plan_shift = entry.get("contact_plan", {}).get("shift", None)
                                if plan_shift is not None:
                                    entry["direct_geo_local_deg_mean_at_plan_shift"] = _direct_mean_geo_local_for_shift(
                                        start, int(plan_shift)
                                    )
                            except Exception:
                                pass
                            try:
                                meas_shift = entry.get("contact_meas", {}).get("shift", None)
                                if meas_shift is not None:
                                    entry["direct_geo_local_deg_mean_at_meas_shift"] = _direct_mean_geo_local_for_shift(
                                        start, int(meas_shift)
                                    )
                            except Exception:
                                pass

                            # Track the problematic phase region (step_in_cycle=8) for quick inspection.
                            step8 = start + 8
                            if 0 <= step8 < int(free_steps):
                                try:
                                    entry["direct_geo_local_deg_step8"] = float(
                                        geo_direct_local_full[:, step8, joint_mask].mean().item()
                                    )
                                except Exception:
                                    entry["direct_geo_local_deg_step8"] = None

                        phase_cycles.append(entry)

                    extra["phase_shift"] = {
                        "cycle_len": int(L),
                        "full_cycles": int(full_cycles),
                        "max_shift": int(max_shift) if max_shift is not None else None,
                        "cycles": phase_cycles,
                    }
            except Exception:
                pass

        # ---- Direct alignment debug (non-circular time shift + joint confusion) ----
        # Motivation:
        #   - Off-by-one: direct head might be predicting pose for t+1 while we compare to t.
        #   - Bone mapping: a stable L/R swap or joint order mismatch can masquerade as a "direct floor",
        #     especially on upper limbs.
        if bool(debug_direct_alignment) and Rd_full is not None and Rg_full is not None and joint_mask is not None:
            try:
                L = int(T_cycle) if (rounds > 1 and int(T_cycle) > 0) else int(free_steps)
                if L <= 1:
                    raise ValueError(f"invalid cycle_len={L}")
                full_cycles = int(int(free_steps) // int(L)) if int(L) > 0 else 0
                include0 = bool(direct_alignment_include_round0) or (int(full_cycles) <= 1)
                start_cycle = 0 if include0 else 1

                # 1) Non-circular shift sweep: mean DirectGeoLocalDeg over cycles for shifts in [-k, k].
                try:
                    k = int(direct_alignment_max_shift)
                except Exception:
                    k = 2
                k = max(0, min(int(k), int(L) - 1))

                def _mean_geo_local_noncyc_shift(shift: int) -> Optional[float]:
                    if int(shift) == 0:
                        # reuse already computed tensor
                        try:
                            return float(geo_direct_local_full[:, : int(full_cycles) * int(L), joint_mask].mean().item())
                        except Exception:
                            pass
                    vals = []
                    for cy in range(int(start_cycle), int(full_cycles)):
                        cs = int(cy) * int(L)
                        ce = cs + int(L)
                        if ce > int(free_steps):
                            break
                        s = int(shift)
                        if s >= 0:
                            ps = cs
                            pe = ce - s
                            gs = cs + s
                            ge = ce
                        else:
                            ps = cs - s
                            pe = ce
                            gs = cs
                            ge = ce + s
                        if pe <= ps or ge <= gs:
                            continue
                        geo = geodesic_R(Rd_full[:, ps:pe], Rg_full[:, gs:ge]) * deg_factor  # (1,T,J)
                        vals.append(float(geo[..., joint_mask].mean().item()))
                    if not vals:
                        return None
                    return float(sum(vals) / max(1, len(vals)))

                shift_rows: List[Dict[str, Any]] = []
                best_s = 0
                best_v = None
                for s in range(-int(k), int(k) + 1):
                    v = _mean_geo_local_noncyc_shift(int(s))
                    shift_rows.append({"shift": int(s), "geo_local_deg_mean": v})
                    if v is None:
                        continue
                    if best_v is None or float(v) < best_v:
                        best_v = float(v)
                        best_s = int(s)

                # 2) Joint confusion matrix on a selected subset (default: arms).
                #    Compute mean geo per (pred_joint_i, gt_joint_j) and surface the best GT match per pred.
                subset: List[int] = []
                subset_names: List[str] = []
                try:
                    names = list(bone_names) if isinstance(bone_names, (list, tuple)) else []
                    if len(names) < int(J):
                        names = names + [f"joint_{i}" for i in range(len(names), int(J))]
                    names = names[: int(J)]
                except Exception:
                    names = [f"joint_{i}" for i in range(int(J))]
                idx_map = {str(n): int(i) for i, n in enumerate(names)}

                spec = str(direct_alignment_joints or "").strip()
                spec_l = spec.lower()
                if spec_l in ("all", "*"):
                    subset = list(range(int(J)))
                elif spec_l in ("keybones", "key_bones"):
                    subset = [int(i) for i in key_indices] if key_indices else []
                elif spec_l in ("arms", "arm", "upper", "upperbody", "upper_body"):
                    want = ["upperarm_l", "lowerarm_l", "hand_l", "upperarm_r", "lowerarm_r", "hand_r"]
                    subset = [idx_map[n] for n in want if n in idx_map]
                else:
                    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
                        idx = None
                        try:
                            idx = int(tok)
                        except Exception:
                            idx = idx_map.get(tok, None)
                        if idx is None:
                            continue
                        if 0 <= int(idx) < int(J):
                            subset.append(int(idx))

                seen = set()
                subset = [i for i in subset if not (i in seen or seen.add(i))]
                subset_names = [names[i] for i in subset if 0 <= int(i) < len(names)]

                confusion = None
                best_gt = None
                if subset:
                    mat: List[List[Optional[float]]] = []
                    best_rows: List[Dict[str, Any]] = []
                    for i_pred, name_pred in zip(subset, subset_names):
                        row: List[Optional[float]] = []
                        best_j = None
                        best_name = None
                        best_val = None
                        for j_gt, name_gt in zip(subset, subset_names):
                            vals = []
                            for cy in range(int(start_cycle), int(full_cycles)):
                                cs = int(cy) * int(L)
                                ce = cs + int(L)
                                if ce > int(free_steps):
                                    break
                                geo = geodesic_R(Rd_full[:, cs:ce, int(i_pred)], Rg_full[:, cs:ce, int(j_gt)]) * deg_factor
                                vals.append(float(geo.mean().item()))
                            v = float(sum(vals) / max(1, len(vals))) if vals else None
                            row.append(v)
                            if v is not None and (best_val is None or float(v) < best_val):
                                best_val = float(v)
                                best_j = int(j_gt)
                                best_name = str(name_gt)
                        mat.append(row)
                        best_rows.append(
                            {
                                "pred_joint": str(name_pred),
                                "pred_idx": int(i_pred),
                                "best_gt_joint": str(best_name) if best_j is not None else None,
                                "best_gt_idx": int(best_j) if best_j is not None else None,
                                "best_geo_deg_mean": float(best_val) if best_val is not None else None,
                            }
                        )
                    confusion = {
                        "subset": [{"name": str(n), "idx": int(i)} for n, i in zip(subset_names, subset)],
                        "mean_geo_deg": mat,
                    }
                    best_gt = best_rows

                extra["direct_alignment"] = {
                    "cycle_len": int(L),
                    "full_cycles": int(full_cycles),
                    "include_round0": bool(include0),
                    "max_shift": int(k),
                    "time_shift_noncyc": {
                        "results": shift_rows,
                        "best": {"shift": int(best_s), "geo_local_deg_mean": float(best_v) if best_v is not None else None},
                    },
                    "joint_confusion": confusion,
                    "best_gt_for_pred": best_gt,
                }
            except Exception:
                pass
    else:
        root_idx = 0
        joint_mask = None
        geo_full = None
        geo_full_aligned0 = None
        geo_local_full = None
        rot6d_l2_full = None
        rot6d_local_l2_full = None
        geo_blend_full = None
        geo_blend_full_aligned0 = None
        geo_blend_local_full = None
        rot6d_blend_l2_full = None
        rot6d_blend_local_l2_full = None
        geo_direct_full = None
        geo_direct_full_aligned0 = None
        geo_direct_local_full = None
        rot6d_direct_l2_full = None
        rot6d_direct_local_l2_full = None
        geo_direct_full_align_inc0 = None
        geo_direct_local_full_align_inc0 = None
        joint_weights = None
        weights_sum = None
        w_joint = None

    # Optional: export per-joint GeoLocal stats and recommend warmup joint scales.
    # This is meant to support per-joint warmup scaling ablations without hand-tuning.
    if bool(export_joint_geolocal) and geo_local_full is not None:
        try:
            # Build joint name list aligned to BoneRotations6D joint count.
            names = list(bone_names) if isinstance(bone_names, (list, tuple)) else []
            if len(names) < int(J):
                names = names + [f"joint_{i}" for i in range(len(names), int(J))]
            names = names[: int(J)]

            inc = geo_local_full[0]  # (T,J)
            steps_total = int(inc.shape[0])
            k = int(getattr(trainer, "lambda_reliability_warmup_steps", 0) or 0)
            if k <= 0:
                k = min(10, steps_total)
            k = max(2, min(int(k), steps_total))

            # Match `--exclude-round0` semantics for per-joint stats when rounds>1:
            # mask steps where cycle>=1 based on the nominal cycle length (T_cycle).
            mask_r1p = None
            try:
                L = int(T_cycle) if (rounds > 1 and int(T_cycle) > 0) else 0
                if L > 0 and int(steps_total) > L:
                    idx = torch.arange(int(steps_total), device=inc.device)
                    mask_r1p = (idx // int(L)) >= 1
            except Exception:
                mask_r1p = None

            inc_mean = inc.mean(dim=0)
            inc_start = inc[0]
            inc_end = inc[-1]
            inc_early_mean = inc[:k].mean(dim=0)
            inc_late_mean = inc[-k:].mean(dim=0) if steps_total >= k else inc_early_mean
            inc_drift_delta = inc[k - 1] - inc[0]

            dloc = geo_direct_local_full[0] if geo_direct_local_full is not None else None
            dloc_align_inc0 = geo_direct_local_full_align_inc0[0] if geo_direct_local_full_align_inc0 is not None else None
            bloc = geo_blend_local_full[0] if geo_blend_local_full is not None else None

            per_joint = {
                "bone_names": names,
                "root_idx": int(root_idx),
                "steps": steps_total,
                "analysis_steps": int(k),
                "GeoLocalDegMean": inc_mean.detach().cpu().tolist(),
                "GeoLocalDegStart": inc_start.detach().cpu().tolist(),
                "GeoLocalDegEnd": inc_end.detach().cpu().tolist(),
                "GeoLocalDegEarlyMean": inc_early_mean.detach().cpu().tolist(),
                "GeoLocalDegLateMean": inc_late_mean.detach().cpu().tolist(),
                "GeoLocalDegDriftDelta": inc_drift_delta.detach().cpu().tolist(),
            }
            if mask_r1p is not None:
                try:
                    inc_r1p = inc[mask_r1p]
                    if inc_r1p.numel() > 0:
                        per_joint["GeoLocalDegMean_R1p"] = inc_r1p.mean(dim=0).detach().cpu().tolist()
                except Exception:
                    pass
            if dloc is not None:
                try:
                    per_joint["DirectGeoLocalDegMean"] = dloc.mean(dim=0).detach().cpu().tolist()
                    per_joint["DirectGeoLocalDegEarlyMean"] = dloc[:k].mean(dim=0).detach().cpu().tolist()
                    per_joint["DirectGeoLocalDegLateMean"] = (
                        dloc[-k:].mean(dim=0) if steps_total >= k else dloc[:k].mean(dim=0)
                    ).detach().cpu().tolist()
                    if mask_r1p is not None:
                        try:
                            dloc_r1p = dloc[mask_r1p]
                            if dloc_r1p.numel() > 0:
                                per_joint["DirectGeoLocalDegMean_R1p"] = dloc_r1p.mean(dim=0).detach().cpu().tolist()
                        except Exception:
                            pass
                except Exception:
                    pass
            if dloc_align_inc0 is not None:
                try:
                    per_joint["DirectGeoLocalDegAlignInc0Mean"] = dloc_align_inc0.mean(dim=0).detach().cpu().tolist()
                    per_joint["DirectGeoLocalDegAlignInc0EarlyMean"] = dloc_align_inc0[:k].mean(dim=0).detach().cpu().tolist()
                    per_joint["DirectGeoLocalDegAlignInc0LateMean"] = (
                        dloc_align_inc0[-k:].mean(dim=0) if steps_total >= k else dloc_align_inc0[:k].mean(dim=0)
                    ).detach().cpu().tolist()
                    if mask_r1p is not None:
                        try:
                            da_r1p = dloc_align_inc0[mask_r1p]
                            if da_r1p.numel() > 0:
                                per_joint["DirectGeoLocalDegAlignInc0Mean_R1p"] = da_r1p.mean(dim=0).detach().cpu().tolist()
                        except Exception:
                            pass
                except Exception:
                    pass
            if bloc is not None:
                try:
                    per_joint["BlendGeoLocalDegMean"] = bloc.mean(dim=0).detach().cpu().tolist()
                    per_joint["BlendGeoLocalDegEarlyMean"] = bloc[:k].mean(dim=0).detach().cpu().tolist()
                    per_joint["BlendGeoLocalDegLateMean"] = (
                        bloc[-k:].mean(dim=0) if steps_total >= k else bloc[:k].mean(dim=0)
                    ).detach().cpu().tolist()
                    if mask_r1p is not None:
                        try:
                            bloc_r1p = bloc[mask_r1p]
                            if bloc_r1p.numel() > 0:
                                per_joint["BlendGeoLocalDegMean_R1p"] = bloc_r1p.mean(dim=0).detach().cpu().tolist()
                        except Exception:
                            pass
                except Exception:
                    pass

            # Heuristic: per-joint warmup scale based on early drift delta.
            # - larger drift => scale > 1 (warmup faster for that joint)
            # - smaller drift => scale < 1 (warmup slower, but will still reach 1 for long rollouts)
            try:
                alpha = 0.5  # sqrt compresses extreme ratios
                min_scale = 0.25
                max_scale = 4.0
                eps = 1e-8

                score = inc_drift_delta.detach().clamp(min=0.0)  # (J,)
                if joint_mask is not None and joint_mask.any():
                    denom = float(score[joint_mask].mean().item())
                else:
                    denom = float(score.mean().item())
                if denom <= eps:
                    scales = torch.ones_like(score)
                else:
                    scales = (score / (denom + eps)).clamp(min=eps).pow(alpha)
                # Keep root neutral by default.
                if 0 <= int(root_idx) < int(scales.numel()):
                    scales[int(root_idx)] = 1.0
                scales = scales.clamp(min_scale, max_scale)
                if joint_mask is not None and joint_mask.any():
                    m = scales[joint_mask].mean()
                    if torch.is_tensor(m) and float(m.item()) > eps:
                        scales = scales.clone()
                        scales[joint_mask] = (scales[joint_mask] / m).clamp(min_scale, max_scale)
                scales_out = scales.detach().cpu().tolist()

                extra["lambda_reliability_warmup_joint_scales_suggested"] = scales_out
                extra["lambda_reliability_warmup_joint_scales_suggested_meta"] = {
                    "method": "inc_geolocal_drift_delta_sqrt_norm",
                    "alpha": float(alpha),
                    "min_scale": float(min_scale),
                    "max_scale": float(max_scale),
                    "analysis_steps": int(k),
                    "root_idx": int(root_idx),
                }
                print("[FreeRun][Suggest] lambda_reliability_warmup_joint_scales =", json.dumps(scales_out))
            except Exception:
                pass

            extra["per_joint_geolocal"] = per_joint
        except Exception:
            pass

    # Optional: export per-step per-joint DirectGeoLocalDeg series (T x J).
    # This is useful to inspect phase-locked residual spikes beyond the keybone subset.
    if bool(export_joint_direct_geolocal_series) and geo_direct_local_full is not None:
        try:
            names = list(bone_names) if isinstance(bone_names, (list, tuple)) else []
            if len(names) < int(J):
                names = names + [f"joint_{i}" for i in range(len(names), int(J))]
            names = names[: int(J)]

            dloc = geo_direct_local_full[0]  # (T,J) deg
            if 0 <= int(root_idx) < int(dloc.shape[1]):
                dloc = dloc.clone()
                dloc[:, int(root_idx)] = 0.0

            extra["per_step_direct_geolocal_deg"] = {
                "bone_names": names,
                "root_idx": int(root_idx),
                "steps": int(dloc.shape[0]),
                "joints": int(dloc.shape[1]),
                "DirectGeoLocalDeg": dloc.detach().cpu().tolist(),
            }
        except Exception:
            pass

    # Optional: export per-step per-joint SO(3) log-map error vectors (T x J x 3).
    # This is the vector form of the scalar geodesic, useful for:
    #   - bias vs variance decomposition (mean/std of epsilon_{t,j})
    #   - axis dominance analysis (is the error mostly twist-like on a fixed local axis?)
    #   - random-walk style drift/diffusion modeling on the Lie algebra.
    if bool(export_joint_so3_error_series):
        try:
            import math

            from train.geometry import so3_log_map

            # Resolve branches to export.
            want = {s.strip().lower() for s in str(joint_so3_error_series_branches or "").split(",") if s.strip()}
            if not want:
                want = {"direct"}
            want = {s for s in want if s in ("inc", "direct", "blend")}

            # Resolve error space: body/world/both.
            space = str(joint_so3_error_series_space or "body").strip().lower()
            if space not in ("body", "world", "both"):
                space = "body"

            names = list(bone_names) if isinstance(bone_names, (list, tuple)) else []
            if len(names) < int(J):
                names = names + [f"joint_{i}" for i in range(len(names), int(J))]
            names = names[: int(J)]

            deg = 180.0 / math.pi

            def _export_space(R_pred: torch.Tensor, *, kind: str) -> Optional[Dict[str, Any]]:
                if R_pred is None or (not torch.is_tensor(R_pred)):
                    return None
                if R_pred.shape != Rg_full.shape:
                    return None
                if kind == "world":
                    R_err = torch.matmul(Rg_full, R_pred.transpose(-1, -2))  # (1,T,J,3,3)
                else:
                    # body / joint-local
                    R_err = torch.matmul(R_pred.transpose(-1, -2), Rg_full)  # (1,T,J,3,3)

                w = so3_log_map(R_err)  # (1,T,J,3) in radians
                w_deg = (w * float(deg)).detach().cpu()[0]  # (T,J,3) in degrees
                ang_deg = w_deg.norm(dim=-1)  # (T,J) in degrees

                if 0 <= int(root_idx) < int(w_deg.shape[1]):
                    w_deg[:, int(root_idx), :] = 0.0
                    ang_deg[:, int(root_idx)] = 0.0

                return {
                    "rotvec_deg_xyz": w_deg.tolist(),
                    "ang_deg": ang_deg.tolist(),
                }

            # Map branch -> rotation matrices.
            br_map = {
                "inc": Rp_full,
                "direct": Rd_full,
                "blend": (Rb_full if "Rb_full" in locals() else None),
            }

            out_br: Dict[str, Any] = {}
            for br in ("inc", "direct", "blend"):
                if br not in want:
                    continue
                R_pred = br_map.get(br, None)
                if R_pred is None:
                    continue
                ent: Dict[str, Any] = {}
                if space in ("body", "both"):
                    b = _export_space(R_pred, kind="body")
                    if b is not None:
                        ent["body"] = b
                if space in ("world", "both"):
                    w = _export_space(R_pred, kind="world")
                    if w is not None:
                        ent["world"] = w
                if ent:
                    out_br[str(br)] = ent

            if out_br:
                extra["per_step_joint_so3_error"] = {
                    "bone_names": names,
                    "root_idx": int(root_idx),
                    "steps": int(free_steps),
                    "joints": int(J),
                    "spaces": (["body"] if space == "body" else ["world"] if space == "world" else ["body", "world"]),
                    "err_R_body": "R_pred^T @ R_gt",
                    "err_R_world": "R_gt @ R_pred^T",
                    "units": {"rotvec_deg_xyz": "deg", "ang_deg": "deg"},
                    "note": (
                        "per_step_joint_so3_error provides vector-form SO(3) errors epsilon_{t,j} for offline analysis.\n"
                        "- body: epsilon = so3_log_map(R_pred^T @ R_gt) converted to degrees, axis in joint-local (pred) frame.\n"
                        "- world: epsilon = so3_log_map(R_gt @ R_pred^T) converted to degrees, axis in world frame.\n"
                        "Recommended mask (for Stage7-style analysis): cycle>=1 + drop wrap_boundary_step + exclude root."
                    ),
                    "branches": out_br,
                }
        except Exception:
            pass

    # Optional: export predicted local joint rotations for selected bones (state features).
    # This is intended to test whether joint-local pose state (beyond plan/meas/phase) explains residual δ* variation.
    if bool(export_keybone_state_series):
        try:
            import math

            from train.geometry import so3_log_map

            deg = 180.0 / math.pi
            want_br = {s.strip().lower() for s in str(keybone_state_series_branches or "").split(",") if s.strip()}
            if not want_br:
                want_br = {"inc", "direct", "blend"}

            # Resolve bone selection.
            spec_raw = str(keybone_state_series_bones or "").strip()
            spec = spec_raw.lower()
            idx_map = {str(n): int(i) for i, n in enumerate(bone_names or [])}

            sel_idx: List[int] = []
            sel_names: List[str] = []

            def _push_joint(j: int) -> None:
                if j < 0:
                    return
                if int(j) == int(root_idx):
                    return
                if j >= int(J):
                    return
                name = bone_names[j] if 0 <= j < len(bone_names) else f"joint_{j}"
                if name not in sel_names:
                    sel_names.append(str(name))
                    sel_idx.append(int(j))

            if spec in ("", "none", "null", "off"):
                pass
            elif spec in ("all", "joints"):
                for j in range(int(J)):
                    _push_joint(int(j))
            elif spec in ("keybones", "keybone"):
                for j in key_indices:
                    _push_joint(int(j))
            else:
                for tok in spec_raw.split(","):
                    t = tok.strip()
                    if not t:
                        continue
                    tl = t.lower()
                    if tl in ("all", "joints"):
                        for j in range(int(J)):
                            _push_joint(int(j))
                        continue
                    if tl in ("keybones", "keybone"):
                        for j in key_indices:
                            _push_joint(int(j))
                        continue
                    j = None
                    if t.isdigit() or (t.startswith("-") and t[1:].isdigit()):
                        try:
                            j = int(t)
                        except Exception:
                            j = None
                    else:
                        j = idx_map.get(t, None)
                        if j is None:
                            j = idx_map.get(tl, None)
                    if j is None:
                        continue
                    _push_joint(int(j))

            if sel_idx:
                series_cfg: Dict[str, Any] = {
                    "bones": list(sel_names),
                    "bone_indices": [int(j) for j in sel_idx],
                    "units": {"pred_rotvec_deg_xyz": "deg", "pred_ang_deg": "deg"},
                    "note": (
                        "pred_rotvec_deg_xyz[t] is so3_log_map(R_pred) converted to degrees (joint-local rotation).\n"
                        "pred_ang_deg[t] = ||pred_rotvec_deg_xyz[t]||.\n"
                        "This export is meant for offline feature probing (e.g., predicting oracle hinge delta*)."
                    ),
                    "branches": {},
                }

                br_map = {"inc": Rp_full, "direct": Rd_full, "blend": (Rb_full if "Rb_full" in locals() else None)}
                for br in ("inc", "direct", "blend"):
                    if br not in want_br:
                        continue
                    R_any = br_map.get(br, None)
                    if R_any is None:
                        continue
                    try:
                        R_sel = R_any[:, :, sel_idx]  # (1,T,K,3,3)
                        w = so3_log_map(R_sel)  # (1,T,K,3) axis-angle (rad)
                        w_deg = (w * float(deg)).detach().cpu()[0]  # (T,K,3)
                        ang_deg = w_deg.norm(dim=-1)  # (T,K)

                        rot_out: Dict[str, List[List[float]]] = {}
                        ang_out: Dict[str, List[float]] = {}
                        for k, name in enumerate(sel_names):
                            rot_out[str(name)] = [[float(a), float(b), float(c)] for a, b, c in w_deg[:, k, :].tolist()]
                            ang_out[str(name)] = [float(x) for x in ang_deg[:, k].tolist()]
                        series_cfg["branches"][str(br)] = {"pred_rotvec_deg_xyz": rot_out, "pred_ang_deg": ang_out}
                    except Exception:
                        continue

                extra["keybone_state"] = {"series": series_cfg}
        except Exception:
            pass

    # Optional: export SO(3) log-map direction diagnostics for key bones.
    # This helps distinguish "twist-like" errors (axis-consistent) vs "swing-like" errors (axis varies),
    # which is hard to see from the scalar geodesic angle alone.
    if bool(export_keybone_omega):
        try:
            import math

            from train.geometry import so3_log_map

            L = int(T_cycle) if int(T_cycle) > 0 else 0
            deg = 180.0 / math.pi
            th = float(keybone_omega_deg_thresh)
            th = max(0.0, th)
            export_series = bool(export_keybone_omega_series)
            series_axis = str(keybone_omega_series_axis or "z").strip().lower()
            if series_axis not in ("x", "y", "z"):
                series_axis = "z"
            axis_idx = {"x": 0, "y": 1, "z": 2}[series_axis]

            keybone_omega: Dict[str, Any] = {
                "err_R": "R_pred^T @ R_gt",
                "deg_thresh": float(th),
                "bone_names": list(key_bone_names) if key_bone_names else [],
                "note": (
                    "Axis stats are computed from the SO(3) log-map vector omega (axis*angle). "
                    "We report abs(axis) alignment to canonical x/y/z axes in the joint's local frame."
                ),
                "branches": {},
            }

            if key_indices and (Rg_full is not None) and (geo_local_full is not None):
                idx = torch.tensor(list(key_indices), device=Rg_full.device, dtype=torch.long)
                Rg_k = Rg_full[:, :, idx]  # (1,T,K,3,3)

                # Optional: per-step signed omega component series for a subset of keybones.
                # This is mainly for diagnosing hinge-like errors (e.g., knee/elbow twist around local z).
                series_cfg: Optional[Dict[str, Any]] = None
                kk_series: List[int] = []
                series_bones: List[str] = []
                if export_series and key_bone_names:
                    spec_raw = str(keybone_omega_series_bones or "").strip()
                    spec = spec_raw.lower()
                    name_to_k = {str(n): int(i) for i, n in enumerate(key_bone_names)}

                    def _push_bone(k: int) -> None:
                        if k < 0 or k >= len(key_indices):
                            return
                        j = int(key_indices[k])
                        if int(j) == int(root_idx):
                            return
                        b = str(key_bone_names[k])
                        if b not in series_bones:
                            series_bones.append(b)
                            kk_series.append(int(k))

                    if spec in ("", "none", "null", "off"):
                        pass
                    elif spec in ("all", "keybones", "keybone"):
                        for k in range(len(key_indices)):
                            _push_bone(k)
                    else:
                        for tok in spec_raw.split(","):
                            t = tok.strip()
                            if not t:
                                continue
                            tl = t.lower()
                            if tl in ("all", "keybones", "keybone"):
                                for k in range(len(key_indices)):
                                    _push_bone(k)
                                continue
                            k = name_to_k.get(t, None)
                            if k is None:
                                k = name_to_k.get(tl, None)
                            if k is None:
                                continue
                            _push_bone(int(k))

                    if series_bones:
                        series_cfg = {
                            "axis": str(series_axis),
                            "units": {"omega_axis_deg": "deg", "omega_deg_xyz": "deg", "ang_deg": "deg"},
                            "bones": list(series_bones),
                            "note": (
                                "omega_axis_deg[t] is the selected component (x/y/z) of "
                                "omega=so3_log_map(R_pred^T@R_gt) converted to degrees (joint local frame). "
                                "omega_deg_xyz[t] is the full omega vector [wx,wy,wz] in degrees (joint local frame). "
                                "ang_deg[t]=||omega|| in degrees."
                            ),
                            "branches": {},
                        }
                        keybone_omega["series"] = series_cfg

                def _branch_stats(Rp_any: Optional[torch.Tensor], branch_name: str) -> None:
                    if Rp_any is None:
                        return
                    try:
                        Rp_k = Rp_any[:, :, idx]  # (1,T,K,3,3)
                        R_err = torch.matmul(Rp_k.transpose(-1, -2), Rg_k)  # (1,T,K,3,3)
                        w = so3_log_map(R_err)  # (1,T,K,3)
                        w = w[0]  # (T,K,3)
                        ang = w.norm(dim=-1)  # (T,K)
                        ang_deg = ang * float(deg)
                        axis = w / ang.unsqueeze(-1).clamp_min(1e-9)
                        abs_axis = axis.abs()  # (T,K,3)

                        if series_cfg is not None and kk_series:
                            try:
                                w_deg = (w * float(deg)).detach().cpu()  # (T,K,3)
                                ang_deg_cpu = ang_deg.detach().cpu()  # (T,K)
                                omega_comp = w_deg[:, :, int(axis_idx)]  # (T,K)
                                omega_out: Dict[str, List[float]] = {}
                                omega_xyz_out: Dict[str, List[List[float]]] = {}
                                ang_out: Dict[str, List[float]] = {}
                                for k, bname in zip(kk_series, series_bones):
                                    omega_out[str(bname)] = [float(x) for x in omega_comp[:, int(k)].tolist()]
                                    omega_xyz_out[str(bname)] = [
                                        [float(a), float(b), float(c)] for a, b, c in w_deg[:, int(k), :].tolist()
                                    ]
                                    ang_out[str(bname)] = [float(x) for x in ang_deg_cpu[:, int(k)].tolist()]
                                series_cfg["branches"][str(branch_name)] = {
                                    "omega_axis_deg": omega_out,
                                    "omega_deg_xyz": omega_xyz_out,
                                    "ang_deg": ang_out,
                                }
                            except Exception:
                                pass

                        T = int(ang_deg.shape[0])
                        K = int(ang_deg.shape[1])
                        rows: List[Dict[str, Any]] = []
                        for kk in range(K):
                            bname = str(key_bone_names[kk]) if kk < len(key_bone_names) else f"bone_{kk}"
                            j_idx = int(key_indices[kk]) if kk < len(key_indices) else -1

                            # Keep root neutral to match KeyBoneGeoLocalDeg convention.
                            if int(j_idx) == int(root_idx):
                                rows.append(
                                    {
                                        "bone": bname,
                                        "joint_idx": int(j_idx),
                                        "is_root": True,
                                        "n": int(T),
                                        "mean_deg": 0.0,
                                        "std_deg": 0.0,
                                        "n_deg_gt_thresh": 0,
                                        "p_deg_gt_thresh": 0.0,
                                        "mean_abs_axis_xyz_if_gt_thresh": None,
                                        "dominant_axis_frac_xyz_if_gt_thresh": None,
                                        "phase_max_mean_deg": None,
                                        "phase_min_mean_deg": None,
                                        "phase_amp_mean_deg": None,
                                    }
                                )
                                continue

                            a = ang_deg[:, kk]  # (T,)
                            mean_deg = float(a.mean().item()) if T > 0 else 0.0
                            std_deg = float(a.std(unbiased=False).item()) if T > 1 else 0.0
                            mask = a > float(th)
                            n_sel = int(mask.sum().item()) if T > 0 else 0
                            p_sel = float(n_sel) / float(T) if T > 0 else 0.0

                            mean_abs = None
                            dom_frac = None
                            if n_sel > 0:
                                aa = abs_axis[:, kk][mask]  # (n_sel,3)
                                mean_abs = [float(x) for x in aa.mean(dim=0).detach().cpu().tolist()]
                                dom = aa.argmax(dim=-1)  # (n_sel,)
                                dom_frac = [
                                    float((dom == 0).to(dtype=aa.dtype).mean().item()),
                                    float((dom == 1).to(dtype=aa.dtype).mean().item()),
                                    float((dom == 2).to(dtype=aa.dtype).mean().item()),
                                ]

                            # Per-phase mean angle for spotting stance/swing hotspots.
                            ph_max = None
                            ph_min = None
                            ph_amp = None
                            if L > 0 and T > 0:
                                by_phase = [[] for _ in range(L)]
                                for tt in range(T):
                                    p = int(tt % L)
                                    v = float(a[tt].item())
                                    if math.isfinite(v):
                                        by_phase[p].append(v)
                                phase_means = [float(sum(xs) / len(xs)) if xs else float("nan") for xs in by_phase]
                                try:
                                    p_max = int(np.nanargmax(np.asarray(phase_means, dtype=np.float64)))
                                    p_min = int(np.nanargmin(np.asarray(phase_means, dtype=np.float64)))
                                    v_max = float(phase_means[p_max])
                                    v_min = float(phase_means[p_min])
                                    if math.isfinite(v_max) and math.isfinite(v_min):
                                        ph_max = {"phase": int(p_max), "mean_deg": float(v_max)}
                                        ph_min = {"phase": int(p_min), "mean_deg": float(v_min)}
                                        ph_amp = float(v_max - v_min)
                                except Exception:
                                    ph_max = None
                                    ph_min = None
                                    ph_amp = None

                            rows.append(
                                {
                                    "bone": bname,
                                    "joint_idx": int(j_idx),
                                    "is_root": False,
                                    "n": int(T),
                                    "mean_deg": float(mean_deg),
                                    "std_deg": float(std_deg),
                                    "n_deg_gt_thresh": int(n_sel),
                                    "p_deg_gt_thresh": float(p_sel),
                                    "mean_abs_axis_xyz_if_gt_thresh": mean_abs,
                                    "dominant_axis_frac_xyz_if_gt_thresh": dom_frac,
                                    "phase_max_mean_deg": ph_max,
                                    "phase_min_mean_deg": ph_min,
                                    "phase_amp_mean_deg": ph_amp,
                                }
                            )

                        keybone_omega["branches"][str(branch_name)] = rows
                    except Exception:
                        pass

                # NOTE: Rp_full/Rd_full are constructed earlier from denormed rot6d and match the GeoLocal metrics.
                _branch_stats(Rp_full, "inc")
                _branch_stats(Rd_full, "direct")
                try:
                    _branch_stats(Rb_full if "Rb_full" in locals() else None, "blend")
                except Exception:
                    pass

            extra["keybone_omega"] = keybone_omega
        except Exception:
            pass

    for t in range(free_steps):
        geo_t = None
        geo_local_t = None
        geo_local_weighted_t = None
        rot6d_local_l2_t = None
        rot6d_local_l2_weighted_t = None
        root_geo_t = None
        geo_aligned0_t = None
        root_geo_aligned0_t = None
        keybone_geo: Dict[str, float] = {}
        keybone_geo_local: Dict[str, float] = {}
        keybone_direct_geo: Dict[str, float] = {}
        keybone_direct_geo_local: Dict[str, float] = {}
        keybone_blend_geo: Dict[str, float] = {}
        keybone_blend_geo_local: Dict[str, float] = {}
        keybone_lambda: Dict[str, float] = {}
        keybone_lambda_eff: Dict[str, float] = {}
        keybone_lambda_rel: Dict[str, float] = {}
        keybone_pos_err_world: Dict[str, float] = {}
        keybone_pos_err_rootrel: Dict[str, float] = {}
        keybone_direct_pos_err_rootrel: Dict[str, float] = {}
        keybone_blend_pos_err_rootrel: Dict[str, float] = {}

        if geo_full is not None:
            # Mean over all joints
            geo_t = float(geo_full[:, t].mean().item())
            root_geo_t = float(geo_full[:, t, root_idx].mean().item())
            # Per-key-bone geodesic errors
            if key_indices:
                per_joint = geo_full[0, t]  # [J]
                for name, j_idx in zip(key_bone_names, key_indices):
                    if 0 <= j_idx < per_joint.numel():
                        keybone_geo[name] = float(per_joint[j_idx].item())

        if geo_full_aligned0 is not None:
            geo_aligned0_t = float(geo_full_aligned0[:, t].mean().item())
            root_geo_aligned0_t = float(geo_full_aligned0[:, t, root_idx].mean().item())

        if geo_local_full is not None:
            if joint_mask is not None and joint_mask.any():
                geo_local_t = float(geo_local_full[:, t, joint_mask].mean().item())
            else:
                geo_local_t = 0.0
            if key_indices:
                per_joint_local = geo_local_full[0, t]
                for name, j_idx in zip(key_bone_names, key_indices):
                    if 0 <= j_idx < per_joint_local.numel():
                        keybone_geo_local[name] = 0.0 if j_idx == root_idx else float(per_joint_local[j_idx].item())
            if w_joint is not None and weights_sum is not None:
                # Weighted GeoLocalDeg (matches Trainer)
                geo_local_weighted_t = float(
                    ((geo_local_full[:, t] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                )

        if rot6d_local_l2_full is not None:
            if joint_mask is not None and joint_mask.any():
                rot6d_local_l2_t = float(rot6d_local_l2_full[:, t, joint_mask].mean().item())
            else:
                rot6d_local_l2_t = 0.0
            if w_joint is not None and weights_sum is not None:
                rot6d_local_l2_weighted_t = float(
                    ((rot6d_local_l2_full[:, t] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                )

        # Optional: FK position error per key bone (meters), computed from X state.
        # - World: includes root translation drift (can jump at cycle wraps)
        # - RootRel: remove root translation, isolate pose FK error
        if key_indices:
            if pos_err_full_world is not None:
                try:
                    per_joint_pos = pos_err_full_world[t]  # (J,)
                    for name, j_idx in zip(key_bone_names, key_indices):
                        if 0 <= j_idx < per_joint_pos.numel():
                            keybone_pos_err_world[name] = float(per_joint_pos[j_idx].item())
                except Exception:
                    pass
            if pos_err_full_rootrel is not None:
                try:
                    per_joint_pos = pos_err_full_rootrel[t]  # (J,)
                    for name, j_idx in zip(key_bone_names, key_indices):
                        if 0 <= j_idx < per_joint_pos.numel():
                            keybone_pos_err_rootrel[name] = float(per_joint_pos[j_idx].item())
                except Exception:
                    pass
            # Root-relative FK position errors computed from pose outputs (Y).
            if pos_err_direct_rootrel_y is not None:
                try:
                    per_joint_pos = pos_err_direct_rootrel_y[t]  # (J,)
                    for name, j_idx in zip(key_bone_names, key_indices):
                        if 0 <= j_idx < per_joint_pos.numel():
                            keybone_direct_pos_err_rootrel[name] = float(per_joint_pos[j_idx].item())
                except Exception:
                    pass
            if pos_err_blend_rootrel_y is not None:
                try:
                    per_joint_pos = pos_err_blend_rootrel_y[t]  # (J,)
                    for name, j_idx in zip(key_bone_names, key_indices):
                        if 0 <= j_idx < per_joint_pos.numel():
                            keybone_blend_pos_err_rootrel[name] = float(per_joint_pos[j_idx].item())
                except Exception:
                    pass

        direct_geo_t = None
        direct_geo_aligned0_t = None
        direct_geo_local_t = None
        direct_geo_local_weighted_t = None
        direct_rot6d_local_l2_t = None
        direct_rot6d_local_l2_weighted_t = None
        direct_root_geo_t = None
        direct_root_geo_aligned0_t = None

        if geo_direct_full is not None:
            direct_geo_t = float(geo_direct_full[:, t].mean().item())
            direct_root_geo_t = float(geo_direct_full[:, t, root_idx].mean().item())
            if key_indices:
                per_joint = geo_direct_full[0, t]  # [J]
                for name, j_idx in zip(key_bone_names, key_indices):
                    if 0 <= j_idx < per_joint.numel():
                        keybone_direct_geo[name] = float(per_joint[j_idx].item())

        if geo_direct_full_aligned0 is not None:
            direct_geo_aligned0_t = float(geo_direct_full_aligned0[:, t].mean().item())
            direct_root_geo_aligned0_t = float(geo_direct_full_aligned0[:, t, root_idx].mean().item())

        if geo_direct_local_full is not None:
            if joint_mask is not None and joint_mask.any():
                direct_geo_local_t = float(geo_direct_local_full[:, t, joint_mask].mean().item())
            else:
                direct_geo_local_t = 0.0
            if key_indices:
                per_joint_local = geo_direct_local_full[0, t]
                for name, j_idx in zip(key_bone_names, key_indices):
                    if 0 <= j_idx < per_joint_local.numel():
                        keybone_direct_geo_local[name] = 0.0 if j_idx == root_idx else float(per_joint_local[j_idx].item())
            if w_joint is not None and weights_sum is not None:
                direct_geo_local_weighted_t = float(
                    ((geo_direct_local_full[:, t] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                )

        if rot6d_direct_local_l2_full is not None:
            if joint_mask is not None and joint_mask.any():
                direct_rot6d_local_l2_t = float(rot6d_direct_local_l2_full[:, t, joint_mask].mean().item())
            else:
                direct_rot6d_local_l2_t = 0.0
            if w_joint is not None and weights_sum is not None:
                direct_rot6d_local_l2_weighted_t = float(
                    ((rot6d_direct_local_l2_full[:, t] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                )

        direct_geo_align_inc0_t = None
        direct_geo_local_align_inc0_t = None
        direct_geo_local_weighted_align_inc0_t = None
        direct_root_geo_align_inc0_t = None
        if geo_direct_full_align_inc0 is not None:
            direct_geo_align_inc0_t = float(geo_direct_full_align_inc0[:, t].mean().item())
            direct_root_geo_align_inc0_t = float(geo_direct_full_align_inc0[:, t, root_idx].mean().item())
        if geo_direct_local_full_align_inc0 is not None:
            if joint_mask is not None and joint_mask.any():
                direct_geo_local_align_inc0_t = float(geo_direct_local_full_align_inc0[:, t, joint_mask].mean().item())
            else:
                direct_geo_local_align_inc0_t = 0.0
            if w_joint is not None and weights_sum is not None:
                direct_geo_local_weighted_align_inc0_t = float(
                    ((geo_direct_local_full_align_inc0[:, t] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                )

        blend_geo_t = None
        blend_geo_aligned0_t = None
        blend_geo_local_t = None
        blend_geo_local_weighted_t = None
        blend_rot6d_local_l2_t = None
        blend_rot6d_local_l2_weighted_t = None
        blend_root_geo_t = None
        blend_root_geo_aligned0_t = None
        if geo_blend_full is not None:
            blend_geo_t = float(geo_blend_full[:, t].mean().item())
            blend_root_geo_t = float(geo_blend_full[:, t, root_idx].mean().item())
            if key_indices:
                per_joint = geo_blend_full[0, t]  # [J]
                for name, j_idx in zip(key_bone_names, key_indices):
                    if 0 <= j_idx < per_joint.numel():
                        keybone_blend_geo[name] = float(per_joint[j_idx].item())
        if geo_blend_full_aligned0 is not None:
            blend_geo_aligned0_t = float(geo_blend_full_aligned0[:, t].mean().item())
            blend_root_geo_aligned0_t = float(geo_blend_full_aligned0[:, t, root_idx].mean().item())
        if geo_blend_local_full is not None:
            if joint_mask is not None and joint_mask.any():
                blend_geo_local_t = float(geo_blend_local_full[:, t, joint_mask].mean().item())
            else:
                blend_geo_local_t = 0.0
            if key_indices:
                per_joint_local = geo_blend_local_full[0, t]
                for name, j_idx in zip(key_bone_names, key_indices):
                    if 0 <= j_idx < per_joint_local.numel():
                        keybone_blend_geo_local[name] = 0.0 if j_idx == root_idx else float(per_joint_local[j_idx].item())
            if w_joint is not None and weights_sum is not None:
                blend_geo_local_weighted_t = float(
                    ((geo_blend_local_full[:, t] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                )

        if rot6d_blend_local_l2_full is not None:
            if joint_mask is not None and joint_mask.any():
                blend_rot6d_local_l2_t = float(rot6d_blend_local_l2_full[:, t, joint_mask].mean().item())
            else:
                blend_rot6d_local_l2_t = 0.0
            if w_joint is not None and weights_sum is not None:
                blend_rot6d_local_l2_weighted_t = float(
                    ((rot6d_blend_local_l2_full[:, t] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                )

        lam_mean_t = lam_std_t = None
        if lambda_steps:
            lam_t = lambda_steps[t] if t < len(lambda_steps) else None
            if torch.is_tensor(lam_t):
                try:
                    lam_mean_t = float(lam_t.mean().item())
                    lam_std_t = float(lam_t.std(unbiased=False).item())
                except Exception:
                    lam_mean_t = lam_std_t = None
                if key_indices:
                    try:
                        lam0 = lam_t[0] if lam_t.dim() == 2 else lam_t
                        for name, j_idx in zip(key_bone_names, key_indices):
                            if 0 <= j_idx < lam0.numel():
                                keybone_lambda[name] = float(lam0[j_idx].item())
                    except Exception:
                        pass

        lam_eff_mean_t = lam_eff_std_t = None
        if lambda_eff_steps:
            lam_t = lambda_eff_steps[t] if t < len(lambda_eff_steps) else None
            if torch.is_tensor(lam_t):
                try:
                    lam_eff_mean_t = float(lam_t.mean().item())
                    lam_eff_std_t = float(lam_t.std(unbiased=False).item())
                except Exception:
                    lam_eff_mean_t = lam_eff_std_t = None
                if key_indices:
                    try:
                        lam0 = lam_t[0] if lam_t.dim() == 2 else lam_t
                        for name, j_idx in zip(key_bone_names, key_indices):
                            if 0 <= j_idx < lam0.numel():
                                keybone_lambda_eff[name] = float(lam0[j_idx].item())
                    except Exception:
                        pass

        lam_rel_mean_t = None
        if lambda_rel_steps:
            rel_t = lambda_rel_steps[t] if t < len(lambda_rel_steps) else None
            if torch.is_tensor(rel_t):
                try:
                    lam_rel_mean_t = float(rel_t.mean().item())
                except Exception:
                    lam_rel_mean_t = None
                if key_indices:
                    try:
                        if rel_t.dim() == 1:
                            rel0 = rel_t[0] if rel_t.numel() > 0 else None
                            if rel0 is not None:
                                v = float(rel0.item())
                                for name in key_bone_names:
                                    keybone_lambda_rel[name] = v
                        elif rel_t.dim() == 2:
                            rel0 = rel_t[0]
                            for name, j_idx in zip(key_bone_names, key_indices):
                                if 0 <= j_idx < rel0.numel():
                                    keybone_lambda_rel[name] = float(rel0[j_idx].item())
                    except Exception:
                        pass

        ec_lam_corr_mean_t = ec_lam_corr_std_t = None
        if event_clock_lambda_corr_steps:
            ec_t = event_clock_lambda_corr_steps[t] if t < len(event_clock_lambda_corr_steps) else None
            if torch.is_tensor(ec_t):
                try:
                    ec_lam_corr_mean_t = float(ec_t.mean().item())
                    ec_lam_corr_std_t = float(ec_t.std(unbiased=False).item())
                except Exception:
                    ec_lam_corr_mean_t = ec_lam_corr_std_t = None

        entry: Dict[str, Any] = {
            "step": int(t),
            "cycle": int(t // int(T_cycle)) if (rounds > 1 and int(T_cycle) > 0) else (0 if rounds > 1 else None),
            "step_in_cycle": int(t % int(T_cycle)) if (rounds > 1 and int(T_cycle) > 0) else None,
            "wrap_boundary_step": bool((rounds > 1) and (int(T_cycle) > 0) and ((int(t) % int(T_cycle)) == (int(T_cycle) - 1))),
            "time_index": int(time_index_steps[t])
            if (time_index_steps and t < len(time_index_steps) and time_index_steps[t] is not None)
            else None,
            "GeoDeg": geo_t,
            "GeoDegAligned0": geo_aligned0_t,
            "GeoLocalDeg": geo_local_t,
            "GeoLocalDegWeighted": geo_local_weighted_t,
            "Rot6dLocalL2": rot6d_local_l2_t,
            "Rot6dLocalL2Weighted": rot6d_local_l2_weighted_t,
            "RootGeoDeg": root_geo_t,
            "RootGeoDegAligned0": root_geo_aligned0_t,
            "BlendGeoDeg": blend_geo_t,
            "BlendGeoDegAligned0": blend_geo_aligned0_t,
            "BlendGeoLocalDeg": blend_geo_local_t,
            "BlendGeoLocalDegWeighted": blend_geo_local_weighted_t,
            "BlendRot6dLocalL2": blend_rot6d_local_l2_t,
            "BlendRot6dLocalL2Weighted": blend_rot6d_local_l2_weighted_t,
            "BlendRootGeoDeg": blend_root_geo_t,
            "BlendRootGeoDegAligned0": blend_root_geo_aligned0_t,
            "DirectGeoDeg": direct_geo_t,
            "DirectGeoDegAligned0": direct_geo_aligned0_t,
            "DirectGeoLocalDeg": direct_geo_local_t,
            "DirectGeoLocalDegWeighted": direct_geo_local_weighted_t,
            "DirectRot6dLocalL2": direct_rot6d_local_l2_t,
            "DirectRot6dLocalL2Weighted": direct_rot6d_local_l2_weighted_t,
            "DirectRootGeoDeg": direct_root_geo_t,
            "DirectRootGeoDegAligned0": direct_root_geo_aligned0_t,
            "DirectGeoDegAlignInc0": direct_geo_align_inc0_t,
            "DirectGeoLocalDegAlignInc0": direct_geo_local_align_inc0_t,
            "DirectGeoLocalDegWeightedAlignInc0": direct_geo_local_weighted_align_inc0_t,
            "DirectRootGeoDegAlignInc0": direct_root_geo_align_inc0_t,
            "LambdaMean": lam_mean_t,
            "LambdaStd": lam_std_t,
            "LambdaEffMean": lam_eff_mean_t,
            "LambdaEffStd": lam_eff_std_t,
            "LambdaRelMean": lam_rel_mean_t,
            "EventClockLambdaCorrMean": ec_lam_corr_mean_t,
            "EventClockLambdaCorrStd": ec_lam_corr_std_t,
            "RootPosErr": float(root_pos_err[t].item()) if root_pos_err is not None else None,
            "RootVelMAE": float(root_vel_mae[t].item()) if root_vel_mae is not None else None,
        }
        if so3_debug_steps:
            s3 = so3_debug_steps[t] if t < len(so3_debug_steps) else None
            if isinstance(s3, dict):
                for k, v in s3.items():
                    entry[k] = v
        if rot_gain_debug_steps:
            rg = rot_gain_debug_steps[t] if t < len(rot_gain_debug_steps) else None
            if isinstance(rg, dict):
                for k, v in rg.items():
                    entry[k] = v
        if direct_leg_omega_plan_gate_step_log:
            g = direct_leg_omega_plan_gate_step_log[t] if t < len(direct_leg_omega_plan_gate_step_log) else None
            if isinstance(g, dict):
                for k, v in g.items():
                    entry[k] = v
        if direct_leg_omega_flip_step_log:
            f = direct_leg_omega_flip_step_log[t] if t < len(direct_leg_omega_flip_step_log) else None
            if isinstance(f, dict):
                for k, v in f.items():
                    entry[k] = v
        if contact_steps:
            c = contact_steps[t] if t < len(contact_steps) else None
            if isinstance(c, dict):
                # Keep keys flat for easy plotting.
                for ck in (
                    "ContactGTMean",
                    "ContactGTAbsMean",
                    "ContactGTPerC",
                    "ContactGTNextMean",
                    "ContactGTNextAbsMean",
                    "ContactGTNextPerC",
                    "ContactsMeasSource",
                    "ContactsMeasSourceApplied",
                    "ContactsMeasOverridePerC",
                    "ContactPlanMean",
                    "ContactPlanAbsMean",
                    "ContactPlanPerC",
                    "ContactPlanLogitsMean",
                    "ContactPlanLogitsStd",
                    "ContactPlanLogitsPerC",
                    "ContactPlanLogitsBasePerC",
                    "ContactPlanLogitsTimePerC",
                    "ContactPlanLogitsRawPerC",
                    "ContactMeasMean",
                    "ContactMeasAbsMean",
                    "ContactMeasPerC",
                    "ContactMeasLogitsMean",
                    "ContactMeasLogitsStd",
                    "ContactMeasLogitsPerC",
                    "AngvelMean",
                    "AngvelAbsMean",
                    "AngvelStd",
                    "PoseHistMean",
                    "PoseHistAbsMean",
	                    "PoseHistStd",
	                    "PlanZNorm",
	                    "PhaseZNorm",
	                    "PhaseZShape",
	                    "PhaseBinN",
	                    "PhaseZInSinCosPerC",
	                    "PhaseAngleInRadPerC",
	                    "PhaseAngleInDegPerC",
	                    "PhaseZInPerCNorm",
	                    "PhaseBinInPerC",
	                    # Touchdown-anchored phase (derived from PhaseAngleInRadPerC + ContactGTPerC touchdown events).
	                    "PhaseAngleInTdRadPerC",
	                    "PhaseAngleInTdDegPerC",
	                    "PhaseBinInTdPerC",
	                    "PhaseZSinCosPerC",
	                    "PhaseAngleRadPerC",
	                    "PhaseAngleDegPerC",
	                    "PhaseZPerCNorm",
	                    "PhaseBinPerC",
	                    "PhaseEventAgePerC",
                    "PhaseEventAgeMean",
                    "PhaseResetSource",
                    "TTCEventKind",
                    "TTCGTPerC",
                    "TTCGTValidPerC",
                    "TTCStatePerC",
                    "TTCEventPerC",
                    "ContactPlanLRAbsDiffMean",
                    "ContactPlanLRDiffStd",
                    "ContactMeasLRAbsDiffMean",
                    "ContactMeasLRDiffStd",
                    "ContactGTLRAbsDiffMean",
                    "ContactGTLRDiffStd",
                    "ContactGTNextLRAbsDiffMean",
                    "ContactGTNextLRDiffStd",
                    "ContactErrAbsMean",
                    "ContactPlanGtAbsMean",
                    "ContactMeasGtAbsMean",
                    "DirectMeasSource",
                    "DirectMeasOverridePerC",
                    "DirectPlanSource",
                    "DirectPlanOverridePerC",
                    "So3GateWarmup",
                    "So3GateErrRef",
                    "So3GateErrEff",
                    "So3GateBase",
                    "So3GateMode",
                    "So3GateScale",
                    "So3GateOverride",
                    "ContactMeasHeadSwapPoseL2",
                    "ContactMeasHeadSwapAngvelL2",
                    "ContactMeasHeadSwapLogitsPPPerC",
                    "ContactMeasHeadSwapLogitsPGPerC",
                    "ContactMeasHeadSwapLogitsGPPerC",
                    "ContactMeasHeadSwapLogitsGGPerC",
                    "ContactMeasHeadSwapMaxAbsDiffToModelLogits",
                    "ContactMeasHeadSwapError",
                ):
                    if ck in c:
                        entry[ck] = c.get(ck)
                if "ContactErrAbsPerC" in c:
                    entry["ContactErrAbsPerC"] = c.get("ContactErrAbsPerC")
                if "ContactErrPerC" in c:
                    entry["ContactErrPerC"] = c.get("ContactErrPerC")
                if "ContactPlanGtAbsPerC" in c:
                    entry["ContactPlanGtAbsPerC"] = c.get("ContactPlanGtAbsPerC")
                if "ContactMeasGtAbsPerC" in c:
                    entry["ContactMeasGtAbsPerC"] = c.get("ContactMeasGtAbsPerC")
                if "ContactMeasWhitebox" in c:
                    # Nested dict (debug only): keep it grouped instead of flattening dozens of keys.
                    entry["ContactMeasWhitebox"] = c.get("ContactMeasWhitebox")
        if keybone_geo:
            entry["KeyBoneGeoDeg"] = keybone_geo
        if keybone_geo_local:
            entry["KeyBoneGeoLocalDeg"] = keybone_geo_local
        if keybone_pos_err_world:
            entry["KeyBonePosErrWorld"] = keybone_pos_err_world
        if keybone_pos_err_rootrel:
            entry["KeyBonePosErrRootRel"] = keybone_pos_err_rootrel
        if keybone_direct_pos_err_rootrel:
            entry["KeyBoneDirectPosErrRootRel"] = keybone_direct_pos_err_rootrel
        if keybone_blend_pos_err_rootrel:
            entry["KeyBoneBlendPosErrRootRel"] = keybone_blend_pos_err_rootrel
        if keybone_direct_geo:
            entry["KeyBoneDirectGeoDeg"] = keybone_direct_geo
        if keybone_direct_geo_local:
            entry["KeyBoneDirectGeoLocalDeg"] = keybone_direct_geo_local
        if keybone_blend_geo:
            entry["KeyBoneBlendGeoDeg"] = keybone_blend_geo
        if keybone_blend_geo_local:
            entry["KeyBoneBlendGeoLocalDeg"] = keybone_blend_geo_local
        if keybone_lambda:
            entry["KeyBoneLambda"] = keybone_lambda
        if keybone_lambda_eff:
            entry["KeyBoneLambdaEff"] = keybone_lambda_eff
        if keybone_lambda_rel:
            entry["KeyBoneLambdaRel"] = keybone_lambda_rel
        per_step.append(entry)

    def _nanmean(xs: List[Optional[float]]) -> Optional[float]:
        vals = [float(x) for x in xs if x is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def _nanstd(xs: List[Optional[float]]) -> Optional[float]:
        vals = [float(x) for x in xs if x is not None]
        if len(vals) < 2:
            return 0.0 if len(vals) == 1 else None
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        return float(var ** 0.5)

    for r in range(rounds):
        # Intra-cycle transitions only: each round covers exactly (T_cycle-1) steps,
        # i.e., transitions within frames [r*T_cycle .. (r+1)*T_cycle-1], dropping the wrap boundary.
        t0 = r * T_cycle
        t1 = min((r + 1) * T_cycle - 1, free_steps)
        if t1 <= t0:
            continue

        geo_deg_val: Optional[float] = None
        geo_deg_start: Optional[float] = None
        geo_deg_end: Optional[float] = None
        geo_local_deg_val: Optional[float] = None
        geo_local_deg_start: Optional[float] = None
        geo_local_deg_end: Optional[float] = None
        geo_local_deg_weighted_val: Optional[float] = None
        geo_local_deg_weighted_start: Optional[float] = None
        geo_local_deg_weighted_end: Optional[float] = None
        rot6d_local_l2_val: Optional[float] = None
        rot6d_local_l2_start: Optional[float] = None
        rot6d_local_l2_end: Optional[float] = None
        rot6d_local_l2_weighted_val: Optional[float] = None
        rot6d_local_l2_weighted_start: Optional[float] = None
        rot6d_local_l2_weighted_end: Optional[float] = None
        root_geo_deg_val: Optional[float] = None
        root_geo_deg_start: Optional[float] = None
        root_geo_deg_end: Optional[float] = None
        root_geo_deg_aligned0_val: Optional[float] = None
        root_geo_deg_aligned0_start: Optional[float] = None
        root_geo_deg_aligned0_end: Optional[float] = None
        keybone_geo_mean: Optional[float] = None
        keybone_geo_local_mean: Optional[float] = None
        keybone_blend_geo_mean: Optional[float] = None
        keybone_blend_geo_local_mean: Optional[float] = None
        root_pos_err_mean: Optional[float] = None
        root_pos_err_start: Optional[float] = None
        root_pos_err_end: Optional[float] = None
        root_vel_mae_mean: Optional[float] = None
        root_vel_mae_start: Optional[float] = None
        root_vel_mae_end: Optional[float] = None

        if geo_full is not None:
            geo_seg = geo_full[:, t0:t1]  # [B, Tr, J]
            geo_deg_val = float(geo_seg.mean().item())
            if geo_seg.shape[1] > 0:
                geo_deg_start = float(geo_seg[:, 0].mean().item())
                geo_deg_end = float(geo_seg[:, -1].mean().item())
            root_geo_deg_val = float(geo_seg[..., root_idx].mean().item())
            if geo_seg.shape[1] > 0:
                root_geo_deg_start = float(geo_seg[:, 0, root_idx].mean().item())
                root_geo_deg_end = float(geo_seg[:, -1, root_idx].mean().item())
            if key_indices:
                kb = geo_seg[..., key_indices]
                keybone_geo_mean = float(kb.mean().item())
        geo_deg_aligned0_val: Optional[float] = None
        geo_deg_aligned0_start: Optional[float] = None
        geo_deg_aligned0_end: Optional[float] = None
        if geo_full_aligned0 is not None:
            geo_align_seg = geo_full_aligned0[:, t0:t1]
            geo_deg_aligned0_val = float(geo_align_seg.mean().item())
            if geo_align_seg.shape[1] > 0:
                geo_deg_aligned0_start = float(geo_align_seg[:, 0].mean().item())
                geo_deg_aligned0_end = float(geo_align_seg[:, -1].mean().item())
            root_geo_deg_aligned0_val = float(geo_align_seg[..., root_idx].mean().item())
            if geo_align_seg.shape[1] > 0:
                root_geo_deg_aligned0_start = float(geo_align_seg[:, 0, root_idx].mean().item())
                root_geo_deg_aligned0_end = float(geo_align_seg[:, -1, root_idx].mean().item())

        if geo_local_full is not None:
            geo_local_seg = geo_local_full[:, t0:t1]
            if joint_mask is not None and joint_mask.any():
                geo_local_deg_val = float(geo_local_seg[..., joint_mask].mean().item())
            else:
                geo_local_deg_val = 0.0
            if geo_local_seg.shape[1] > 0:
                if joint_mask is not None and joint_mask.any():
                    geo_local_deg_start = float(geo_local_seg[:, 0, joint_mask].mean().item())
                    geo_local_deg_end = float(geo_local_seg[:, -1, joint_mask].mean().item())
                else:
                    geo_local_deg_start = 0.0
                    geo_local_deg_end = 0.0
            if w_joint is not None and weights_sum is not None:
                geo_local_deg_weighted_val = float(
                    (geo_local_seg * w_joint).sum().item()
                    / (weights_sum.item() * geo_local_seg.shape[0] * geo_local_seg.shape[1])
                )
                if geo_local_seg.shape[1] > 0:
                    geo_local_deg_weighted_start = float(
                        ((geo_local_seg[:, 0] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                    )
                    geo_local_deg_weighted_end = float(
                        ((geo_local_seg[:, -1] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                    )
            if key_indices:
                key_no_root = [i for i in key_indices if i != root_idx]
                if key_no_root:
                    kb_local = geo_local_seg[..., key_no_root]
                    keybone_geo_local_mean = float(kb_local.mean().item())

        if geo_blend_full is not None:
            try:
                bseg = geo_blend_full[:, t0:t1]
                if key_indices:
                    kb = bseg[..., key_indices]
                    keybone_blend_geo_mean = float(kb.mean().item())
            except Exception:
                pass

        if geo_blend_local_full is not None:
            try:
                bseg = geo_blend_local_full[:, t0:t1]
                if key_indices:
                    key_no_root = [i for i in key_indices if i != root_idx]
                    if key_no_root:
                        kb_local = bseg[..., key_no_root]
                        keybone_blend_geo_local_mean = float(kb_local.mean().item())
            except Exception:
                pass

        if rot6d_local_l2_full is not None:
            try:
                l2_seg = rot6d_local_l2_full[:, t0:t1]
                if joint_mask is not None and joint_mask.any():
                    rot6d_local_l2_val = float(l2_seg[..., joint_mask].mean().item())
                else:
                    rot6d_local_l2_val = 0.0
                if l2_seg.shape[1] > 0:
                    if joint_mask is not None and joint_mask.any():
                        rot6d_local_l2_start = float(l2_seg[:, 0, joint_mask].mean().item())
                        rot6d_local_l2_end = float(l2_seg[:, -1, joint_mask].mean().item())
                    else:
                        rot6d_local_l2_start = 0.0
                        rot6d_local_l2_end = 0.0
                if w_joint is not None and weights_sum is not None:
                    rot6d_local_l2_weighted_val = float(
                        (l2_seg * w_joint).sum().item()
                        / (weights_sum.item() * l2_seg.shape[0] * l2_seg.shape[1])
                    )
                    if l2_seg.shape[1] > 0:
                        rot6d_local_l2_weighted_start = float(
                            ((l2_seg[:, 0] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                        )
                        rot6d_local_l2_weighted_end = float(
                            ((l2_seg[:, -1] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                        )
            except Exception:
                pass

        if root_pos_err is not None:
            seg = root_pos_err[t0:t1]
            root_pos_err_mean = float(seg.mean().item())
            root_pos_err_start = float(seg[0].item()) if seg.numel() > 0 else None
            root_pos_err_end = float(seg[-1].item()) if seg.numel() > 0 else None

        if root_vel_mae is not None:
            seg = root_vel_mae[t0:t1]
            root_vel_mae_mean = float(seg.mean().item())
            root_vel_mae_start = float(seg[0].item()) if seg.numel() > 0 else None
            root_vel_mae_end = float(seg[-1].item()) if seg.numel() > 0 else None

        round_entry: Dict[str, Any] = {
            "round": int(r),
            "start_step": int(t0),
            "end_step": int(t1 - 1),
            "steps": int(t1 - t0),
            "GeoDeg": geo_deg_val,
            "GeoDegStart": geo_deg_start,
            "GeoDegEnd": geo_deg_end,
            "GeoDegAligned0": geo_deg_aligned0_val,
            "GeoDegAligned0Start": geo_deg_aligned0_start,
            "GeoDegAligned0End": geo_deg_aligned0_end,
            "GeoLocalDeg": geo_local_deg_val,
            "GeoLocalDegStart": geo_local_deg_start,
            "GeoLocalDegEnd": geo_local_deg_end,
            "GeoLocalDegWeighted": geo_local_deg_weighted_val,
            "GeoLocalDegWeightedStart": geo_local_deg_weighted_start,
            "GeoLocalDegWeightedEnd": geo_local_deg_weighted_end,
            "Rot6dLocalL2": rot6d_local_l2_val,
            "Rot6dLocalL2Start": rot6d_local_l2_start,
            "Rot6dLocalL2End": rot6d_local_l2_end,
            "Rot6dLocalL2Weighted": rot6d_local_l2_weighted_val,
            "Rot6dLocalL2WeightedStart": rot6d_local_l2_weighted_start,
            "Rot6dLocalL2WeightedEnd": rot6d_local_l2_weighted_end,
            "RootGeoDeg": root_geo_deg_val,
            "RootGeoDegStart": root_geo_deg_start,
            "RootGeoDegEnd": root_geo_deg_end,
            "RootGeoDegAligned0": root_geo_deg_aligned0_val,
            "RootGeoDegAligned0Start": root_geo_deg_aligned0_start,
            "RootGeoDegAligned0End": root_geo_deg_aligned0_end,
            "RootPosErrMean": root_pos_err_mean,
            "RootPosErrStart": root_pos_err_start,
            "RootPosErrEnd": root_pos_err_end,
            "RootVelMAEMean": root_vel_mae_mean,
            "RootVelMAEStart": root_vel_mae_start,
            "RootVelMAEEnd": root_vel_mae_end,
        }
        if geo_blend_full is not None:
            try:
                bseg = geo_blend_full[:, t0:t1]
                round_entry["BlendGeoDeg"] = float(bseg.mean().item())
                round_entry["BlendRootGeoDeg"] = float(bseg[..., root_idx].mean().item())
                if bseg.shape[1] > 0:
                    round_entry["BlendGeoDegStart"] = float(bseg[:, 0].mean().item())
                    round_entry["BlendGeoDegEnd"] = float(bseg[:, -1].mean().item())
                    round_entry["BlendRootGeoDegStart"] = float(bseg[:, 0, root_idx].mean().item())
                    round_entry["BlendRootGeoDegEnd"] = float(bseg[:, -1, root_idx].mean().item())
            except Exception:
                pass
        if geo_blend_full_aligned0 is not None:
            try:
                bseg = geo_blend_full_aligned0[:, t0:t1]
                round_entry["BlendGeoDegAligned0"] = float(bseg.mean().item())
                round_entry["BlendRootGeoDegAligned0"] = float(bseg[..., root_idx].mean().item())
                if bseg.shape[1] > 0:
                    round_entry["BlendGeoDegAligned0Start"] = float(bseg[:, 0].mean().item())
                    round_entry["BlendGeoDegAligned0End"] = float(bseg[:, -1].mean().item())
                    round_entry["BlendRootGeoDegAligned0Start"] = float(bseg[:, 0, root_idx].mean().item())
                    round_entry["BlendRootGeoDegAligned0End"] = float(bseg[:, -1, root_idx].mean().item())
            except Exception:
                pass
        if geo_blend_local_full is not None:
            try:
                bloc = geo_blend_local_full[:, t0:t1]
                if joint_mask is not None and joint_mask.any():
                    round_entry["BlendGeoLocalDeg"] = float(bloc[..., joint_mask].mean().item())
                else:
                    round_entry["BlendGeoLocalDeg"] = 0.0
                if bloc.shape[1] > 0:
                    if joint_mask is not None and joint_mask.any():
                        round_entry["BlendGeoLocalDegStart"] = float(bloc[:, 0, joint_mask].mean().item())
                        round_entry["BlendGeoLocalDegEnd"] = float(bloc[:, -1, joint_mask].mean().item())
                    else:
                        round_entry["BlendGeoLocalDegStart"] = 0.0
                        round_entry["BlendGeoLocalDegEnd"] = 0.0
                if w_joint is not None and weights_sum is not None:
                    round_entry["BlendGeoLocalDegWeighted"] = float(
                        (bloc * w_joint).sum().item()
                        / (weights_sum.item() * bloc.shape[0] * bloc.shape[1])
                    )
                    if bloc.shape[1] > 0:
                        round_entry["BlendGeoLocalDegWeightedStart"] = float(
                            ((bloc[:, 0] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                        )
                        round_entry["BlendGeoLocalDegWeightedEnd"] = float(
                            ((bloc[:, -1] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                        )
            except Exception:
                pass
        if rot6d_blend_local_l2_full is not None:
            try:
                l2 = rot6d_blend_local_l2_full[:, t0:t1]
                if joint_mask is not None and joint_mask.any():
                    round_entry["BlendRot6dLocalL2"] = float(l2[..., joint_mask].mean().item())
                else:
                    round_entry["BlendRot6dLocalL2"] = 0.0
                if l2.shape[1] > 0:
                    if joint_mask is not None and joint_mask.any():
                        round_entry["BlendRot6dLocalL2Start"] = float(l2[:, 0, joint_mask].mean().item())
                        round_entry["BlendRot6dLocalL2End"] = float(l2[:, -1, joint_mask].mean().item())
                    else:
                        round_entry["BlendRot6dLocalL2Start"] = 0.0
                        round_entry["BlendRot6dLocalL2End"] = 0.0
                if w_joint is not None and weights_sum is not None:
                    round_entry["BlendRot6dLocalL2Weighted"] = float(
                        (l2 * w_joint).sum().item()
                        / (weights_sum.item() * l2.shape[0] * l2.shape[1])
                    )
                    if l2.shape[1] > 0:
                        round_entry["BlendRot6dLocalL2WeightedStart"] = float(
                            ((l2[:, 0] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                        )
                        round_entry["BlendRot6dLocalL2WeightedEnd"] = float(
                            ((l2[:, -1] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                        )
            except Exception:
                pass
        if geo_direct_full is not None:
            try:
                dseg = geo_direct_full[:, t0:t1]
                round_entry["DirectGeoDeg"] = float(dseg.mean().item())
                round_entry["DirectRootGeoDeg"] = float(dseg[..., root_idx].mean().item())
                if dseg.shape[1] > 0:
                    round_entry["DirectGeoDegStart"] = float(dseg[:, 0].mean().item())
                    round_entry["DirectGeoDegEnd"] = float(dseg[:, -1].mean().item())
                    round_entry["DirectRootGeoDegStart"] = float(dseg[:, 0, root_idx].mean().item())
                    round_entry["DirectRootGeoDegEnd"] = float(dseg[:, -1, root_idx].mean().item())
            except Exception:
                pass
        if geo_direct_full_aligned0 is not None:
            try:
                dseg = geo_direct_full_aligned0[:, t0:t1]
                round_entry["DirectGeoDegAligned0"] = float(dseg.mean().item())
                round_entry["DirectRootGeoDegAligned0"] = float(dseg[..., root_idx].mean().item())
                if dseg.shape[1] > 0:
                    round_entry["DirectGeoDegAligned0Start"] = float(dseg[:, 0].mean().item())
                    round_entry["DirectGeoDegAligned0End"] = float(dseg[:, -1].mean().item())
                    round_entry["DirectRootGeoDegAligned0Start"] = float(dseg[:, 0, root_idx].mean().item())
                    round_entry["DirectRootGeoDegAligned0End"] = float(dseg[:, -1, root_idx].mean().item())
            except Exception:
                pass
        if geo_direct_local_full is not None:
            try:
                dloc = geo_direct_local_full[:, t0:t1]
                if joint_mask is not None and joint_mask.any():
                    round_entry["DirectGeoLocalDeg"] = float(dloc[..., joint_mask].mean().item())
                else:
                    round_entry["DirectGeoLocalDeg"] = 0.0
                if dloc.shape[1] > 0:
                    if joint_mask is not None and joint_mask.any():
                        round_entry["DirectGeoLocalDegStart"] = float(dloc[:, 0, joint_mask].mean().item())
                        round_entry["DirectGeoLocalDegEnd"] = float(dloc[:, -1, joint_mask].mean().item())
                    else:
                        round_entry["DirectGeoLocalDegStart"] = 0.0
                        round_entry["DirectGeoLocalDegEnd"] = 0.0
                if w_joint is not None and weights_sum is not None:
                    round_entry["DirectGeoLocalDegWeighted"] = float(
                        (dloc * w_joint).sum().item()
                        / (weights_sum.item() * dloc.shape[0] * dloc.shape[1])
                    )
                    if dloc.shape[1] > 0:
                        round_entry["DirectGeoLocalDegWeightedStart"] = float(
                            ((dloc[:, 0] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                        )
                        round_entry["DirectGeoLocalDegWeightedEnd"] = float(
                            ((dloc[:, -1] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                        )
            except Exception:
                pass
        if rot6d_direct_local_l2_full is not None:
            try:
                l2 = rot6d_direct_local_l2_full[:, t0:t1]
                if joint_mask is not None and joint_mask.any():
                    round_entry["DirectRot6dLocalL2"] = float(l2[..., joint_mask].mean().item())
                else:
                    round_entry["DirectRot6dLocalL2"] = 0.0
                if l2.shape[1] > 0:
                    if joint_mask is not None and joint_mask.any():
                        round_entry["DirectRot6dLocalL2Start"] = float(l2[:, 0, joint_mask].mean().item())
                        round_entry["DirectRot6dLocalL2End"] = float(l2[:, -1, joint_mask].mean().item())
                    else:
                        round_entry["DirectRot6dLocalL2Start"] = 0.0
                        round_entry["DirectRot6dLocalL2End"] = 0.0
                if w_joint is not None and weights_sum is not None:
                    round_entry["DirectRot6dLocalL2Weighted"] = float(
                        (l2 * w_joint).sum().item()
                        / (weights_sum.item() * l2.shape[0] * l2.shape[1])
                    )
                    if l2.shape[1] > 0:
                        round_entry["DirectRot6dLocalL2WeightedStart"] = float(
                            ((l2[:, 0] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                        )
                        round_entry["DirectRot6dLocalL2WeightedEnd"] = float(
                            ((l2[:, -1] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                        )
            except Exception:
                pass
        if geo_direct_full_align_inc0 is not None:
            try:
                dseg = geo_direct_full_align_inc0[:, t0:t1]
                round_entry["DirectGeoDegAlignInc0"] = float(dseg.mean().item())
                round_entry["DirectRootGeoDegAlignInc0"] = float(dseg[..., root_idx].mean().item())
                if dseg.shape[1] > 0:
                    round_entry["DirectGeoDegAlignInc0Start"] = float(dseg[:, 0].mean().item())
                    round_entry["DirectGeoDegAlignInc0End"] = float(dseg[:, -1].mean().item())
                    round_entry["DirectRootGeoDegAlignInc0Start"] = float(dseg[:, 0, root_idx].mean().item())
                    round_entry["DirectRootGeoDegAlignInc0End"] = float(dseg[:, -1, root_idx].mean().item())
            except Exception:
                pass
        if geo_direct_local_full_align_inc0 is not None:
            try:
                dloc = geo_direct_local_full_align_inc0[:, t0:t1]
                if joint_mask is not None and joint_mask.any():
                    round_entry["DirectGeoLocalDegAlignInc0"] = float(dloc[..., joint_mask].mean().item())
                else:
                    round_entry["DirectGeoLocalDegAlignInc0"] = 0.0
                if dloc.shape[1] > 0:
                    if joint_mask is not None and joint_mask.any():
                        round_entry["DirectGeoLocalDegAlignInc0Start"] = float(dloc[:, 0, joint_mask].mean().item())
                        round_entry["DirectGeoLocalDegAlignInc0End"] = float(dloc[:, -1, joint_mask].mean().item())
                    else:
                        round_entry["DirectGeoLocalDegAlignInc0Start"] = 0.0
                        round_entry["DirectGeoLocalDegAlignInc0End"] = 0.0
                if w_joint is not None and weights_sum is not None:
                    round_entry["DirectGeoLocalDegWeightedAlignInc0"] = float(
                        (dloc * w_joint).sum().item()
                        / (weights_sum.item() * dloc.shape[0] * dloc.shape[1])
                    )
                    if dloc.shape[1] > 0:
                        round_entry["DirectGeoLocalDegWeightedAlignInc0Start"] = float(
                            ((dloc[:, 0] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                        )
                        round_entry["DirectGeoLocalDegWeightedAlignInc0End"] = float(
                            ((dloc[:, -1] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                        )
            except Exception:
                pass
        if contact_steps:
            seg = contact_steps[t0:t1]
            if seg:
                round_entry["ContactPlanGtAbsMean"] = _nanmean([c.get("ContactPlanGtAbsMean") if isinstance(c, dict) else None for c in seg])
                round_entry["ContactMeasGtAbsMean"] = _nanmean([c.get("ContactMeasGtAbsMean") if isinstance(c, dict) else None for c in seg])
                round_entry["ContactErrAbsMean"] = _nanmean([c.get("ContactErrAbsMean") if isinstance(c, dict) else None for c in seg])
                plan_mean_seq = [c.get("ContactPlanMean") if isinstance(c, dict) else None for c in seg]
                round_entry["ContactPlanMeanStd"] = _nanstd(plan_mean_seq)
        if lambda_steps:
            try:
                lam_seg = [c.reshape(-1).numpy() for c in lambda_steps[t0:t1] if torch.is_tensor(c)]
                if lam_seg:
                    flat = np.concatenate(lam_seg, axis=0)
                    round_entry["LambdaMean"] = float(np.mean(flat))
                    round_entry["LambdaStd"] = float(np.std(flat))
            except Exception:
                pass
        if lambda_eff_steps:
            try:
                lam_seg = [c.reshape(-1).numpy() for c in lambda_eff_steps[t0:t1] if torch.is_tensor(c)]
                if lam_seg:
                    flat = np.concatenate(lam_seg, axis=0)
                    round_entry["LambdaEffMean"] = float(np.mean(flat))
                    round_entry["LambdaEffStd"] = float(np.std(flat))
            except Exception:
                pass
        if lambda_rel_steps:
            try:
                rel_seg = [c.reshape(-1).numpy() for c in lambda_rel_steps[t0:t1] if torch.is_tensor(c)]
                if rel_seg:
                    flat = np.concatenate(rel_seg, axis=0)
                    round_entry["LambdaRelMean"] = float(np.mean(flat))
            except Exception:
                pass
        if keybone_geo_mean is not None:
            round_entry["KeyBoneGeoDegMean"] = keybone_geo_mean
        if keybone_geo_local_mean is not None:
            round_entry["KeyBoneGeoLocalDegMean"] = keybone_geo_local_mean
        if keybone_blend_geo_mean is not None:
            round_entry["KeyBoneBlendGeoDegMean"] = keybone_blend_geo_mean
        if keybone_blend_geo_local_mean is not None:
            round_entry["KeyBoneBlendGeoLocalDegMean"] = keybone_blend_geo_local_mean
        metrics_per_round.append(round_entry)

    if bool(debug_so3_corr) and so3_debug_steps:
        try:
            keys = (
                "So3DbgGate",
                "So3DbgOmegaHatDegMean",
                "So3DbgOmegaEffDegMean",
                "So3DbgOmegaTargetWorldDegMean",
                "So3DbgCosHatWorld",
                "So3DbgCosEffWorld",
                "So3DbgCosEffBody",
                "So3DbgCosEffWorldIfBody",
                "So3DbgGeoLocalDelta",
                "So3DbgClippedFrac",
            )
            summary: Dict[str, Optional[float]] = {}
            for k in keys:
                vals: List[float] = []
                for rec in so3_debug_steps:
                    if not isinstance(rec, dict):
                        continue
                    v = rec.get(k, None)
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if not np.isfinite(fv):
                        continue
                    vals.append(fv)
                summary[k + "Mean"] = float(sum(vals) / len(vals)) if vals else None
            extra["so3_debug"] = summary
        except Exception:
            pass

    if bool(debug_rot_gain) and rot_gain_debug_steps:
        try:
            def _finite_vals(key: str) -> List[float]:
                vals: List[float] = []
                for rec in rot_gain_debug_steps:
                    if not isinstance(rec, dict):
                        continue
                    v = rec.get(key, None)
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if not np.isfinite(fv):
                        continue
                    vals.append(fv)
                return vals

            gain_sel = _finite_vals("RotGainDbgGainSel")
            gain_max = _finite_vals("RotGainDbgGainMaxExRoot")
            d0_sel = _finite_vals("RotGainDbgD0SelDegMean")
            d1_sel = _finite_vals("RotGainDbgD1SelDegMean")

            extra["rot_gain_debug"] = {
                "rot_gain_deg": float(rot_gain_deg),
                "rot_gain_axis": str(rot_gain_axis),
                "rot_gain_joints": list(rot_gain_joint_names) if rot_gain_joint_names else list(rot_gain_joint_indices),
                "RotGainDbgGainSelMean": float(np.mean(gain_sel)) if gain_sel else None,
                "RotGainDbgGainSelMax": float(np.max(gain_sel)) if gain_sel else None,
                "RotGainDbgGainMaxExRootMean": float(np.mean(gain_max)) if gain_max else None,
                "RotGainDbgGainMaxExRootMax": float(np.max(gain_max)) if gain_max else None,
                "RotGainDbgD0SelDegMeanMean": float(np.mean(d0_sel)) if d0_sel else None,
                "RotGainDbgD1SelDegMeanMean": float(np.mean(d1_sel)) if d1_sel else None,
            }
        except Exception:
            pass

    return metrics_per_round, per_step, extra


# ---- CLI --------------------------------------------------------------------


def _failfast_removed_stage74_freerun_cli_args(argv: Sequence[str]) -> None:
    removed_flags = (
        "--direct_pose_leg_alpha_table_json",
        "--direct_pose_leg_alpha_table_cycle_gte",
        "--direct_pose_leg_alpha_table_drop_wrap",
        "--direct_pose_leg_sign_table_json",
        "--direct_pose_leg_sign_table_cycle_gte",
        "--direct_pose_leg_sign_table_drop_wrap",
    )
    found: List[str] = []
    for tok in argv:
        key = str(tok).split("=", 1)[0].strip()
        if key in removed_flags:
            found.append(key)
    if found:
        uniq = sorted(set(found))
        raise SystemExit(
            "[FATAL] Removed Stage7.3/7.4 CLI args detected in run_freerun_cycles: "
            + ", ".join(uniq)
            + ". These alpha/sign table apply paths are archived. "
              "Use main-chain diagnostics (e.g., direct_leg_omega_alpha_sweep or contact-gate/flip probes)."
        )


def parse_args() -> argparse.Namespace:
    _failfast_removed_stage74_freerun_cli_args(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description="Run multi‑cycle free‑run diagnostics on teacher batches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--teacher",
        nargs="+",
        required=True,
        help="Teacher JSON files, directories, or glob patterns (e.g., validate/teacher_batches/*.json).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to checkpoint (.pth) that contains {'model': state_dict}.",
    )
    parser.add_argument(
        "--bundle",
        type=str,
        default="raw_data/processed_data/norm_template.json",
        help="Normalization bundle (same schema as norm_template.json).",
    )
    parser.add_argument(
        "--pretrain-template",
        type=str,
        default="models/pretrain_template.json",
        help="Optional template that carries angvel / pose history stats (merged into bundle spec).",
    )
    parser.add_argument(
        "--encoder-bundle",
        type=str,
        default="models/motion_encoder_equiv_stageA.pt",
        help="Frozen motion encoder bundle (.pt) if your checkpoint expects it.",
    )
    parser.add_argument(
        "--npz-root",
        type=str,
        default="raw_data/processed_data",
        help="Directory that holds processed *.npz clips generated via convert_json_to_npz.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="debug_output/freerun_cycles",
        help="Directory to store per‑clip multi‑cycle JSON diagnostics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Computation device preference.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Attention head count used during training (must divide hidden width).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability for shared encoder / motion head.",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=16,
        help="Context length hyperparameter (only stored for completeness).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help=(
            "Encoder depth used during training (must match checkpoint). "
            "depth<=2 uses the original MLP encoder; depth>2 enables a residual encoder "
            "(2-layer stem + (depth-2) residual blocks)."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of full animation cycles to free‑run without reset.",
    )
    parser.add_argument(
        "--time-index-mode",
        type=str,
        default="auto",
        choices=("auto", "global", "cycle", "none"),
        help=(
            "How to feed time_index into EventMotionModel (used by contact_plan time-PE). "
            "'global' uses the global step t; 'cycle' uses (t %% cycle_len) to stay in-range under multi-cycle; "
            "'auto' uses 'cycle' when rounds>1 else 'global'; 'none' disables time_index."
        ),
    )
    parser.add_argument(
        "--time-index-cycle-minus1",
        action="store_true",
        help=(
            "Ablation: when time_index_mode is cycle/auto and rounds>1, use cycle_len=(T_cycle-1) for time_index "
            "(aligns with posttrain rollout_steps=T-1 semantics: idx=t%%(T-1))."
        ),
    )
    parser.add_argument(
        "--cond-reprojection",
        type=str,
        default="auto",
        choices=("on", "off", "auto"),
        help=(
            "Condition reprojection (yaw-based) before normalizing cond_tgt_raw: "
            "'on' always applies reprojection; 'off' disables it; "
            "'auto' follows Trainer.enable_cond_reprojection and disables reprojection when freerun_yaw_strategy='trajectory'."
        ),
    )
    parser.add_argument(
        "--analyze-phase-shift",
        action="store_true",
        help=(
            "Compute per-cycle best circular shifts that align contacts_plan/meas and the direct head (if available) "
            "to GT, and write the summary into the output JSON as 'phase_shift'."
        ),
    )
    parser.add_argument(
        "--phase-shift-max",
        type=int,
        default=None,
        help=(
            "If set, restrict phase-shift search to signed shifts in [-K, K] (with wrap-around). "
            "Default searches the full cycle."
        ),
    )
    parser.add_argument(
        "--multicycle-sync-state-on-cycle-start",
        action="store_true",
        help=(
            "Ablation: when rounds>1, overwrite the autoregressive rollout state with teacher state at each cycle start "
            "(t%%T_cycle==0) to avoid wrap-boundary carry. This is NOT a pure free-run across cycles."
        ),
    )
    parser.add_argument(
        "--multicycle-reset-plan-z-on-cycle-start",
        action="store_true",
        help=(
            "Ablation: when rounds>1, set plan_z=None at each cycle start so the contact_plan GRU state re-inits. "
            "Useful to test whether plan_z drift is the main source of phase collapse."
        ),
    )
    parser.add_argument(
        "--multicycle-reset-pose-hist-on-cycle-start",
        action="store_true",
        help=(
            "Ablation: when rounds>1 and pose_hist_source=buffer, overwrite the rolling pose_history buffer at each "
            "cycle start (t%%T_cycle==0) using the dataset pose_hist_norm at that step (teacher history). "
            "This breaks cross-cycle pose_hist carry while preserving within-cycle updates (unless pose_hist_update_source "
            "is freeze/gt/zero). Not a pure free-run across cycles (injects teacher pose_hist at boundaries)."
        ),
    )
    parser.add_argument(
        "--freerun_x_gt_except_rot6d",
        action="store_true",
        help=(
            "Ablation: after each autoregressive update, overwrite the carried state X with teacher GT for all components "
            "except the BoneRotations6D slice. This keeps pose free-run while removing drift from root/state channels "
            "(helps diagnose whether errors come from pose vs X-side root inputs)."
        ),
    )
    parser.add_argument(
        "--freerun_x_gt",
        action="store_true",
        help=(
            "Ablation: after each autoregressive update, overwrite the carried state X with teacher GT for all components "
            "(including BoneRotations6D). This turns the rollout into an X-side teacher-forced stream (useful for TF vs FR "
            "comparisons without changing exporters)."
        ),
    )
    parser.add_argument(
        "--pose_hist_source",
        type=str,
        default="buffer",
        choices=("buffer", "seq", "zero"),
        help=(
            "Ablation: select pose_history input source when pose_hist is enabled in the checkpoint. "
            "'buffer' uses the rolling buffer built from rollout y_used_raw (default; matches deployment); "
            "'seq' uses the dataset pose_hist_norm (teacher history) to break the self-feedback loop; "
            "'zero' feeds zeros."
        ),
    )
    parser.add_argument(
        "--pose_hist_update_source",
        type=str,
        default="pred",
        choices=("pred", "gt", "zero", "freeze"),
        help=(
            "When --pose_hist_source=buffer, control how the rolling pose_history buffer is updated each step: "
            "'pred' inserts rollout y_used_raw rot6d (default); "
            "'gt' inserts teacher GT rot6d; "
            "'zero' inserts zeros; "
            "'freeze' keeps the initial buffer fixed."
        ),
    )
    parser.add_argument(
        "--pose_hist_hybrid_boundary_carry",
        action="store_true",
        help=(
            "Prototype: when pose_hist_source=buffer and pose_hist_update_source=pred, replace wrap-boundary pose_hist "
            "carry with a hybrid rot6d write: current leg joints + frozen donor non-leg joints. "
            "This stays eval-only and only affects the pose_hist boundary writer."
        ),
    )
    parser.add_argument(
        "--pose_hist_hybrid_donor_ckpt",
        type=str,
        default=None,
        help=(
            "Frozen donor checkpoint used by --pose_hist_hybrid_boundary_carry. The donor runs in parallel with its "
            "own motion/pose_hist/plan state and contributes non-leg rot6d only at wrap boundaries."
        ),
    )
    parser.add_argument(
        "--contact_plan_init_mode",
        type=str,
        default=None,
        choices=("zeros", "learnable", "obs", "learnable+obs"),
        help=(
            "Override contact_plan_init_mode used by EventMotionModel when contact_plan is enabled. "
            "This is mainly for ablations / bootstrapping older checkpoints that don't carry init_head weights."
        ),
    )
    parser.add_argument(
        "--contact_plan_init_hidden",
        type=int,
        default=None,
        help="Hidden dim for contact_plan_init_head when --contact_plan_init_mode is obs/learnable+obs.",
    )
    parser.add_argument(
        "--contact_plan_init_dropout",
        type=float,
        default=None,
        help="Dropout for contact_plan_init_head when --contact_plan_init_mode is obs/learnable+obs.",
    )
    parser.add_argument(
        "--contact_phase_state_event_thr",
        type=float,
        default=0.5,
        help=(
            "Threshold (0..1) for contacts_meas crossing events used as clock anchors when --phase_reset_source=contacts_meas, "
            "and also used when deriving TTC_gt events from teacher contacts."
        ),
    )
    parser.add_argument(
        "--contact_phase_state_event_hyst",
        type=float,
        default=0.0,
        help=(
            "Debounce phase reset events to suppress double-trigger: "
            "touchdown/liftoff requires prev_meas further than thr±hyst (0 disables)."
        ),
    )
    parser.add_argument(
        "--contact_phase_state_event_min_interval",
        type=int,
        default=0,
        help="Per-foot min interval (frames) between accepted phase reset events (0 disables).",
    )
    parser.add_argument(
        "--phase_reset_source",
        type=str,
        default="contacts_meas",
        choices=("contacts_meas", "ttc_gt", "none"),
        help=(
            "Phase reset / clock anchor source for contact_phase_state: "
            "'contacts_meas'=threshold crossing on contacts_meas inside the model (default); "
            "'ttc_gt'=use TTC computed from teacher GT contacts and drive resets externally; "
            "'none'=disable phase reset events (no-reset)."
        ),
    )
    parser.add_argument(
        "--phase_reset_source_strict",
        type=str,
        default="off",
        choices=("off", "on"),
        help=(
            "If 'on', abort when the requested --phase_reset_source cannot be applied and would fall back to contacts_meas."
        ),
    )
    parser.add_argument(
        "--ttc_event_kind",
        type=str,
        default="touchdown",
        choices=("touchdown", "liftoff", "both"),
        help="Event kind for TTC countdown (used when --phase_reset_source is ttc_*).",
    )
    parser.add_argument(
        "--ttc_max",
        type=int,
        default=None,
        help="Optional TTC cap (frames) when computing TTC_gt from teacher contacts (None disables).",
    )
    parser.add_argument(
        "--ttc_gt_event_shift",
        type=str,
        default="",
        help=(
            "Debug-only: shift TTC_gt event frames (within each cycle) before using them as phase-reset anchors. "
            "Format: int or comma-separated per-contact shifts (e.g. '5,0'). "
            "Positive delays events; negative advances events; wrap-around is applied within each cycle."
        ),
    )
    parser.add_argument(
        "--ttc_apply_phase_reset_to_phase_z",
        type=str,
        default="on",
        choices=("on", "off"),
        help=(
            "When --phase_reset_source is ttc_* and plan_enable is true, also reset phase_z to the anchor "
            "[sin=0,cos=1] at TTC events. This couples phase_z to the TTC event stream; set to 'off' to disable "
            "the external phase_z reset (ablation)."
        ),
    )
    parser.add_argument(
        "--event_clock",
        type=str,
        default="auto",
        choices=("auto", "on", "off"),
        help="Enable Event-Clock v3 inside contact_plan loop (auto=enable when ckpt has weights).",
    )
    parser.add_argument(
        "--contact_plan_time_bias_scale",
        type=float,
        default=1.0,
        help=(
            "Scale the contact_plan time-PE bias (contact_plan_time_head) before adding to plan logits. "
            "k=0 disables time-PE; k>1 amplifies. Note: when Event-Clock is on, the time term is further scaled by lambda_corr."
        ),
    )
    parser.add_argument(
        "--contact_plan_inject_scale",
        type=float,
        default=1.0,
        help=(
            "Scale injected contact_plan feature into shared_encoder input (only affects contact_plan_inject!=none). "
            "k=0 disables injection; k>1 amplifies."
        ),
    )
    parser.add_argument(
        "--log_contact_plan_logits_decomp",
        action="store_true",
        help=(
            "When --log_contacts is enabled, also log ContactPlanLogits{Base,Phase,Time,Raw}PerC per step "
            "(requires the patched EventMotionModel that returns contacts_plan_logits_{base,phase,time,raw})."
        ),
    )
    parser.add_argument(
        "--event_clock_max_delta",
        type=float,
        default=0.5,
        help="Event-Clock Δz clip amplitude (0 disables clip).",
    )
    parser.add_argument(
        "--event_clock_hidden_dim",
        type=int,
        default=None,
        help="Override Event-Clock corrector hidden dim (if not inferred from ckpt).",
    )
    parser.add_argument(
        "--event_clock_gate_hidden_dim",
        type=int,
        default=None,
        help="Override Event-Clock gate hidden dim (if not inferred from ckpt).",
    )
    parser.add_argument(
        "--so3_corr_apply",
        action="store_true",
        help="Apply SO(3) corrector during compose (uses model omega_hat).",
    )
    parser.add_argument(
        "--debug_so3_corr",
        action="store_true",
        help=(
            "Log SO(3) corrector diagnostics into metrics_per_step (So3Dbg*): "
            "omega_hat vs target correction alignment and 1-step GeoLocalDeg delta."
        ),
    )
    parser.add_argument(
        "--debug_rot_gain",
        action="store_true",
        help=(
            "Finite-difference probe: inject a small rot6d perturbation at each step and log "
            "one-step amplification (RotGainDbg*). This estimates local closed-loop gain/contractiveness."
        ),
    )
    parser.add_argument(
        "--rot_gain_joints",
        type=str,
        default="calf_l",
        help="Comma-separated bone names or indices to perturb (e.g. 'calf_l,calf_r' or '9,12'). Supports 'keybones' or 'all'.",
    )
    parser.add_argument(
        "--rot_gain_deg",
        type=float,
        default=0.5,
        help="Perturbation angle in degrees (applied as left-multiply ΔR @ R_prev).",
    )
    parser.add_argument(
        "--rot_gain_axis",
        type=str,
        default="z",
        choices=("x", "y", "z"),
        help="Axis for the perturbation in the chosen rotation convention.",
    )
    parser.add_argument(
        "--lambda_fusion_apply",
        action="store_true",
        help="Apply Stage2 lambda fusion during rollout update (requires out_direct + lambda_fusion).",
    )
    parser.add_argument(
        "--lambda_reliability_mode",
        type=str,
        default=None,
        help="Override deterministic r_t mode for λ (none|warmup|contacts_err|warmup+contacts_err). If omitted and ckpt has posttrain_cfg, uses that.",
    )
    parser.add_argument(
        "--lambda_reliability_warmup_steps",
        type=int,
        default=None,
        help="Warmup steps K for r_t ramp 0->1 when mode includes warmup (override).",
    )
    parser.add_argument(
        "--lambda_reliability_contact_err_max",
        type=float,
        default=None,
        help="contacts_err_abs_mean scale for r_t=clamp(1-err/max,0,1) when mode includes contacts_err (override).",
    )
    parser.add_argument(
        "--lambda_reliability_warmup_joint_scales",
        type=str,
        default=None,
        help="Optional per-joint warmup scales: JSON list (e.g. '[1,1,2,...]') or a JSON file path containing list/scales. If omitted and ckpt has posttrain_cfg, uses that.",
    )
    parser.add_argument(
        "--export_joint_geolocal",
        action="store_true",
        help="Export per-joint GeoLocal stats and suggest lambda_reliability_warmup_joint_scales (written into output JSON and printed).",
    )
    parser.add_argument(
        "--export_joint_direct_geolocal_series",
        action="store_true",
        help=(
            "Export per-step per-joint DirectGeoLocalDeg series (T x J) into output JSON under 'per_step_direct_geolocal_deg'. "
            "Useful for inspecting phase-locked spikes across all joints."
        ),
    )
    parser.add_argument(
        "--export_joint_so3_error_series",
        action="store_true",
        help=(
            "Export per-step per-joint SO(3) log-map error vectors epsilon(t,j) into output JSON under "
            "'per_step_joint_so3_error'. This is the vector form of the geodesic angle and is useful for "
            "bias/variance/random-walk decomposition (axis dominance, phase-locked bias, drift/diffusion)."
        ),
    )
    parser.add_argument(
        "--joint_so3_error_series_branches",
        type=str,
        default="direct",
        help="Comma-separated branches to export for per_step_joint_so3_error (subset of inc,direct,blend). Default: direct.",
    )
    parser.add_argument(
        "--joint_so3_error_series_space",
        type=str,
        default="body",
        choices=("body", "world", "both"),
        help=(
            "Which error frame to export for per_step_joint_so3_error: "
            "body uses R_pred^T@R_gt (joint-local); world uses R_gt@R_pred^T. Default: body."
        ),
    )
    parser.add_argument(
        "--export_keybone_pos_err",
        action="store_true",
        help=(
            "Export per-step keybone FK position errors (meters) into metrics_per_step. "
            "State (X)-based: 'KeyBonePosErrWorld'/'KeyBonePosErrRootRel' from predX_raw vs gtX_raw "
            "(aligns with RootPosErr, i.e. next-state). "
            "Pose (Y)-based root-relative: 'KeyBoneDirectPosErrRootRel'/'KeyBoneBlendPosErrRootRel' "
            "(isolates pose/FK error; useful for checking whether joint-space flips matter for foot/ball)."
        ),
    )
    parser.add_argument(
        "--export_keybone_omega",
        action="store_true",
        help=(
            "Export keybone SO(3) log-map diagnostics (axis-angle direction stats) into output JSON under 'keybone_omega'. "
            "Useful to tell whether large GeoLocal errors are mostly 'twist-like' (axis-consistent) vs 'swing-like'."
        ),
    )
    parser.add_argument(
        "--keybone_omega_deg_thresh",
        type=float,
        default=20.0,
        help="Threshold (deg) for conditioning keybone omega axis statistics (default: 20).",
    )
    parser.add_argument(
        "--export_keybone_omega_series",
        action="store_true",
        help=(
            "Also export per-step signed omega component series into keybone_omega.series "
            "(useful to diagnose phase-locked hinge/twist bias vs variance)."
        ),
    )
    parser.add_argument(
        "--keybone_omega_series_bones",
        type=str,
        default="calf_l,calf_r,lowerarm_l",
        help=(
            "Comma-separated subset of keybones to export omega series for (default: calf_l,calf_r,lowerarm_l). "
            "Use 'keybones' to export all keybones."
        ),
    )
    parser.add_argument(
        "--keybone_omega_series_axis",
        type=str,
        default="z",
        choices=("x", "y", "z"),
        help="Which local-axis component of omega to export for the series (default: z).",
    )
    parser.add_argument(
        "--export_plan_state_series",
        action="store_true",
        help=(
            "Export contact_plan internal state inputs per step into output JSON under 'plan_state_series': "
            "plan_z_in (GRU hidden), phase_z_in (sin/cos phase state), and phase_event_age_in. "
            "Useful for diagnosing missing conditioning signals for phase-locked residuals."
        ),
    )
    parser.add_argument(
        "--export_contact_meas_head_swap",
        action="store_true",
        help=(
            "Export contact_meas_head input-swap diagnostics into metrics_per_step (requires contact_meas_head): "
            "recompute meas logits on four (pose,angvel) combinations: (pred,pred),(pred,gt),(gt,pred),(gt,gt). "
            "This helps isolate whether meas collapse is driven by pose drift vs angvel drift under free-run."
        ),
    )
    parser.add_argument(
        "--export_direct_hinge_series",
        action="store_true",
        help=(
            "Export per-step direct hinge head output (direct_hinge_delta) into output JSON under 'direct_hinge_series'. "
            "This is the model's predicted δ (rad/deg) before any oracle fitting."
        ),
    )
    parser.add_argument(
        "--export_direct_leg_omega_series",
        action="store_true",
        help=(
            "Export per-step direct leg residual omega (direct_leg_omega, axis-angle in rad) into output JSON under "
            "'direct_leg_omega_series' (includes ||omega|| in deg and saturation stats vs direct_pose_leg_max_deg)."
        ),
    )
    parser.add_argument(
        "--export_direct_leg_head_io",
        action="store_true",
        help=(
            "Debug-only: export direct leg head first-layer IO (input vector + first Linear pre-activation) into output JSON "
            "under 'direct_leg_head_io'. This reuses the alpha-sweep step selector (direct_leg_omega_alpha_sweep_{steps,sics,sic_range}) "
            "to keep the payload small."
        ),
    )
    parser.add_argument(
        "--export_direct_nonleg_probe",
        action="store_true",
        help=(
            "Debug-only: export non-leg probe bundle into output JSON under 'direct_nonleg_probe'. "
            "Includes non-leg branch features (pre_proj_in/proj_pre0/out_in) and selected-bone rot6d targets."
        ),
    )
    parser.add_argument(
        "--direct_nonleg_probe_bones",
        type=str,
        default="upperarm_l,lowerarm_l,hand_l,pinky_01_l",
        help=(
            "Comma-separated bones for --export_direct_nonleg_probe targets "
            "(default: upperarm_l,lowerarm_l,hand_l,pinky_01_l; use 'all_nonleg' for all non-leg joints)."
        ),
    )
    parser.add_argument(
        "--direct_nonleg_probe_sics",
        type=str,
        default="",
        help=(
            "Optional comma-separated step_in_cycle filter for --export_direct_nonleg_probe "
            "(default empty => all valid SICs with cycle>=1 and drop_wrap)."
        ),
    )
    parser.add_argument(
        "--export_direct_arm_probe",
        action="store_true",
        help=(
            "Debug-only: export arm-split probe bundle into output JSON under 'direct_arm_probe'. "
            "Includes direct_in/direct_phase/trunk_hidden/proj_pre0/out_in/arm_out and selected-bone rot6d targets."
        ),
    )
    parser.add_argument(
        "--direct_arm_probe_bones",
        type=str,
        default="clavicle_l,clavicle_r,upperarm_l,upperarm_r,RUpArmTwist_l_01,RUpArmTwist_r_01,lowerarm_l,lowerarm_r,hand_l,hand_r,spine_01",
        help=(
            "Comma-separated bones for --export_direct_arm_probe targets "
            "(default: clavicle_l,clavicle_r,upperarm_l,upperarm_r,RUpArmTwist_l_01,RUpArmTwist_r_01,lowerarm_l,lowerarm_r,hand_l,hand_r,spine_01; use 'all_arm' to keep all selected joints)."
        ),
    )
    parser.add_argument(
        "--direct_arm_probe_sics",
        type=str,
        default="",
        help=(
            "Optional comma-separated step_in_cycle filter for --export_direct_arm_probe "
            "(default empty => all valid SICs with cycle>=1 and drop_wrap)."
        ),
    )
    parser.add_argument(
        "--export_direct_leg_omega_alpha_sweep",
        action="store_true",
        help=(
            "Debug-only: export a small alpha-sweep + oracle-alignment report for direct_leg_omega into output JSON under "
            "'direct_leg_omega_alpha_sweep'. This evaluates DirectGeoLocalDeg under exp(alpha*omega_pred) @ R_base on the "
            "same rollout stream (R_base is the pre-leg-apply direct output)."
        ),
    )
    parser.add_argument(
        "--direct_leg_omega_alpha_sweep_alphas",
        type=str,
        default="0,0.25,0.5,1,-1",
        help="Comma-separated alpha list for exp(alpha*omega) sweep (default: 0,0.25,0.5,1,-1).",
    )
    parser.add_argument(
        "--direct_leg_omega_alpha_sweep_steps",
        type=str,
        default="",
        help=(
            "Comma-separated absolute step indices to export, optionally using 'cycle:sic' tokens (e.g. '3:51,4:51,228'). "
            "Mask cycle>=1 and drop wrap is always applied."
        ),
    )
    parser.add_argument(
        "--direct_leg_omega_alpha_sweep_sics",
        type=str,
        default="",
        help="Comma-separated step_in_cycle (sic) list to include (expanded across cycles>=1, drop wrap).",
    )
    parser.add_argument(
        "--direct_leg_omega_alpha_sweep_sic_range",
        type=str,
        default="",
        help="Inclusive sic range 'a-b' (or 'a:b') to include (expanded across cycles>=1, drop wrap).",
    )
    parser.add_argument(
        "--direct_leg_omega_alpha_sweep_bones",
        type=str,
        default="leg",
        help="Comma-separated bone names to include (default: 'leg' => all direct_pose_leg_bones).",
    )
    parser.add_argument(
        "--export_direct_leg_omega_grad",
        action="store_true",
        help=(
            "Debug-only: compute autograd gradient norms for the direct leg-omega head on selected steps, using "
            "DirectGeoLocalDeg (SO(3) geodesic, deg) as the scalar loss. Results are written into output JSON "
            "under 'direct_leg_omega_grad'."
        ),
    )
    parser.add_argument(
        "--direct_leg_omega_grad_sics",
        type=str,
        default="12,14",
        help="Comma-separated step_in_cycle (sic) list to include for --export_direct_leg_omega_grad (default: 12,14).",
    )
    parser.add_argument(
        "--direct_leg_omega_grad_cycle_gte",
        type=int,
        default=1,
        help="Only include cycles >= this value for --export_direct_leg_omega_grad (default: 1).",
    )
    parser.add_argument(
        "--direct_leg_omega_grad_drop_wrap",
        type=str,
        default="on",
        help="Whether to drop wrap-boundary steps for --export_direct_leg_omega_grad (on/off; default: on).",
    )
    parser.add_argument(
        "--direct_leg_omega_grad_bones",
        type=str,
        default="leg",
        help="Comma-separated bone names to include for --export_direct_leg_omega_grad (default: 'leg' => all direct_pose_leg_bones).",
    )
    parser.add_argument(
        "--direct_pose_leg_noapply",
        action="store_true",
        help=(
            "Debug-only: do NOT apply direct_leg_omega to out_direct (i.e., evaluate the direct head output alone). "
            "This changes both direct metrics and (when --lambda_fusion_apply is set) the rollout update stream."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_apply_scale",
        type=float,
        default=1.0,
        help=(
            "Debug-only: scale direct_leg_omega before apply (omega_eff = scale * omega). "
            "Useful for amplitude/overshoot ablations (e.g. 0.25/0.5). Default: 1.0."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_apply_sign",
        type=float,
        default=1.0,
        help=(
            "Debug-only: multiply direct_leg_omega by a sign before apply (omega_eff = sign * omega). "
            "Use -1 to test axis/sign flip hypotheses. Default: +1."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_apply_side",
        type=str,
        default="left",
        choices=["left", "right"],
        help=(
            "Debug-only: choose SO(3) composition side for direct_leg_omega. "
            "left: Exp(omega) @ R_base (default); right: R_base @ Exp(omega)."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_contact_gate",
        action="store_true",
        help=(
            "Debug-only: gate direct_leg_omega apply by contacts_plan transition strength (per-side). "
            "We compute per-foot delta |c_t - c_{t-1}| from contacts_plan and scale omega by "
            "g = gmin + (1-gmin)*exp(-k*delta). Default: off."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_contact_gate_mode",
        type=str,
        default="delta",
        choices=["delta", "phase"],
        help=(
            "Gating mode: "
            "delta=use |contacts_plan[t]-contacts_plan[t-1]| (or logits delta); "
            "phase=use |phase_angle| proximity to 0 (touchdown anchor) from phase_z_next. Default: delta."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_contact_gate_phase_window_deg",
        type=float,
        default=30.0,
        help=(
            "For gate_mode=phase: window (deg) around phase angle 0 where omega is suppressed. "
            "g_raw = clamp(|angle|/window, 0, 1). Default: 30.0."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_contact_gate_joints",
        type=str,
        default="all",
        help=(
            "Which leg joints to gate (based on model.direct_pose_leg_joint_names). "
            "all=gate all leg joints; distal=only joints whose name contains foot/ball/toe; "
            "or provide comma-separated exact joint names (e.g. 'foot_r,ball_r,foot_l,ball_l'). "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_contact_gate_order",
        type=str,
        default="rl",
        choices=["rl", "lr"],
        help=(
            "Mapping from contacts_plan channels to (right,left). "
            "rl: contacts_plan[0]=right, [1]=left (default). "
            "lr: contacts_plan[0]=left,  [1]=right."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_contact_gate_signal",
        type=str,
        default="logit",
        choices=["prob", "logit"],
        help=(
            "Which contacts_plan signal to use for delta gating: "
            "prob=contacts_plan in [0,1]; logit=contacts_plan_logits (more sensitive). Default: logit."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_contact_gate_k",
        type=float,
        default=20.0,
        help="Gating sharpness k for exp(-k*|dc|). Larger => stronger suppression on transitions. Default: 20.0.",
    )
    parser.add_argument(
        "--direct_pose_leg_contact_gate_min",
        type=float,
        default=0.0,
        help="Minimum gating floor gmin in [0,1]. g = gmin + (1-gmin)*exp(-k*|dc|). Default: 0.0.",
    )
    parser.add_argument(
        "--direct_pose_leg_contact_flip",
        action="store_true",
        help=(
            "Debug-only: conditional sign flip for direct_leg_omega inside a phase window (per-side), "
            "using phase_z_next. Intended to test the 'best_alpha < 0' hypothesis only in contact-transition windows."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_contact_flip_order",
        type=str,
        default="rl",
        choices=["rl", "lr"],
        help=(
            "Mapping from contact channels to (right,left) for flip window tests. "
            "rl: ch0=right,ch1=left (default); lr: ch0=left,ch1=right."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_contact_flip_phase_window_deg",
        type=float,
        default=30.0,
        help="Phase window (deg) around angle 0 where omega is sign-flipped for the selected joints. Default: 30.0.",
    )
    parser.add_argument(
        "--direct_pose_leg_contact_flip_delta_thr",
        type=float,
        default=0.0,
        help=(
            "Optional extra condition: only flip when |Δcontacts_plan_logits| > thr for that foot "
            "(computed from consecutive steps). 0 disables. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_contact_flip_joints",
        type=str,
        default="foot_r,foot_l",
        help=(
            "Which leg joints to flip (based on model.direct_pose_leg_joint_names). "
            "Supports all/distal/or comma-separated exact joint names. Default: foot_r,foot_l."
        ),
    )
    parser.add_argument(
        "--export_keybone_state_series",
        action="store_true",
        help=(
            "Export per-step predicted local joint rotations for selected bones/branches into output JSON under 'keybone_state'. "
            "Each rotation is represented by rotvec_deg_xyz = so3_log_map(R_pred) converted to degrees (joint local). "
            "This helps test whether adding joint-local pose state would explain residual δ* variation."
        ),
    )
    parser.add_argument(
        "--keybone_state_series_bones",
        type=str,
        default="calf_l,calf_r,lowerarm_l",
        help=(
            "Comma-separated bones to export keybone_state series for (default: calf_l,calf_r,lowerarm_l). "
            "Use 'keybones' to export the standard eval_key_bones set; use 'all' to export all joints except root."
        ),
    )
    parser.add_argument(
        "--keybone_state_series_branches",
        type=str,
        default="inc,direct,blend",
        help="Comma-separated branches to export in keybone_state (subset of inc,direct,blend). Default: inc,direct,blend.",
    )
    parser.add_argument(
        "--debug_direct_alignment",
        action="store_true",
        help=(
            "Diagnostics: check direct head alignment by (1) non-circular per-cycle time shift sweep (±k) "
            "and (2) direct joint confusion matrix on a chosen subset (detect L/R swap / joint order mismatch). "
            "Writes results into output JSON under 'direct_alignment'."
        ),
    )
    parser.add_argument(
        "--direct_alignment_max_shift",
        type=int,
        default=2,
        help="Max |shift| for non-circular time shift sweep (default: 2).",
    )
    parser.add_argument(
        "--direct_alignment_joints",
        type=str,
        default="upperarm_l,lowerarm_l,hand_l,upperarm_r,lowerarm_r,hand_r",
        help="Comma-separated bone names/indices for joint confusion matrix (supports 'arms'|'keybones'|'all').",
    )
    parser.add_argument(
        "--direct_alignment_include_round0",
        action="store_true",
        help="Include round0/cycle0 in alignment averaging (default: exclude when rounds>1).",
    )
    parser.add_argument(
        "--direct_align_inc0",
        action="store_true",
        help=(
            "Diagnostics only: align the direct head's per-joint rotations by a constant bias computed at step0 "
            "(R_bias[j]=R_inc0[j]@R_dir0[j]^T) and report Direct*AlignInc0 metrics. "
            "Useful to verify whether direct early errors are mainly phase/anchor offsets."
        ),
    )
    parser.add_argument(
        "--direct_pose_meas_source",
        type=str,
        default="model",
        choices=("model", "whitebox", "gt", "softgt", "zero"),
        help=(
            "Override the *direct* head's contacts_meas source: "
            "'model'=as-is; 'whitebox'=use whitebox contacts_in_t; 'gt'=use teacher soft contacts; "
            "'softgt'=use teacher contacts mapped into the model's typical soft range via --direct_pose_softgt_stats; "
            "'zero'=ignore. Does not change contacts_err/lambda (only direct hint)."
        ),
    )
    parser.add_argument(
        "--contacts_meas_source",
        type=str,
        default="model",
        choices=("model", "whitebox", "gt", "zero", "pretrain_contact"),
        help=(
            "Override the model's runtime contacts_meas source used by contacts_err / Event-Clock (and thus λ stats): "
            "'model'=use learned contact_meas_head; 'whitebox'=use runtime whitebox contacts; "
            "'gt'=use teacher soft contacts; 'zero'=all zeros; "
            "'pretrain_contact'=use frozen pretrain contact_head from --encoder-bundle "
            "(input contact channels are zeroed to avoid leakage). "
            "This affects contacts_err, Event-Clock signals (delta_meas/lr_diff), and any downstream closed-loop stats."
        ),
    )
    parser.add_argument(
        "--contacts_meas_pretrain_clamp",
        type=float,
        default=1.0,
        help=(
            "When --contacts_meas_source=pretrain_contact, clamp the frozen pretrain encoder input to [-k,+k]. "
            "Default 1.0 to reduce OOD saturation under freerun drift; set 0 to disable."
        ),
    )
    parser.add_argument(
        "--contacts_meas_pretrain_affine_stats",
        type=str,
        default=None,
        help=(
            "Optional pretrain-contact calibration spec (JSON path or inline JSON), applied only when "
            "--contacts_meas_source=pretrain_contact. Expected schema: {\"scale\":[...],\"bias\":[...],\"eps\":1e-4}. "
            "Runtime mapping uses logit-space affine: p' = sigmoid(b + s * logit(p))."
        ),
    )
    parser.add_argument(
        "--contacts_meas_pretrain_anchor_ckpt",
        type=str,
        default=None,
        help=(
            "Optional tiny-GRU anchor checkpoint for pretrain_contact source (torch .pt with config+state_dict). "
            "Applied after optional pretrain affine calibration."
        ),
    )
    parser.add_argument(
        "--contacts_meas_model_logit_scale",
        type=float,
        default=1.0,
        help=(
            "When --contacts_meas_source=model, scale contact_meas_head logits before sigmoid. "
            ">=1 sharpens (higher confidence), <1 softens. Debug-only (no retraining)."
        ),
    )
    parser.add_argument(
        "--contacts_meas_model_onehot",
        action="store_true",
        help=(
            "When --contacts_meas_source=model, convert the learned contacts_meas probs into hard one-hot across channels "
            "(winner-take-all). Diagnostic only; enforces single support."
        ),
    )
    parser.add_argument(
        "--contacts_meas_model_onehot_conditional",
        action="store_true",
        help=(
            "When --contacts_meas_source=model, apply winner-take-all only outside (likely) double support. "
            "Double support is detected as >=2 channels whose prob exceeds --contacts_meas_model_onehot_ds_thr "
            "(useful for L/R feet contacts)."
        ),
    )
    parser.add_argument(
        "--contacts_meas_model_onehot_ds_thr",
        type=float,
        default=0.5,
        help=(
            "Double-support detection threshold for --contacts_meas_model_onehot_conditional (prob-space, after "
            "--contacts_meas_model_logit_scale and sigmoid). Default: 0.5."
        ),
    )
    parser.add_argument(
        "--contacts_meas_gt_override_sics",
        type=str,
        default="",
        help=(
            "Debug-only: when --contacts_meas_source=model, override contacts_meas with teacher contacts only on "
            "selected step_in_cycle (sic). Supports comma lists and ranges like '12,14' or '10-15'."
        ),
    )
    parser.add_argument(
        "--contacts_meas_gt_override_cycle_gte",
        type=int,
        default=1,
        help="Debug-only: apply --contacts_meas_gt_override_sics only for cycles >= N (default: 1).",
    )
    parser.add_argument(
        "--contacts_meas_gt_override_drop_wrap",
        type=str,
        default="on",
        help="Debug-only: if 'on', do not apply GT override on wrap_boundary_step (default: on).",
    )
    parser.add_argument(
        "--direct_pose_meas_warmup_steps",
        type=int,
        default=0,
        help="If >0, only use --direct_pose_meas_source for the first K steps (global), then force 'zero' for the rest.",
    )
    parser.add_argument(
        "--direct_pose_plan_source",
        type=str,
        default="model",
        choices=("model", "gt", "softgt", "zero"),
        help=(
            "Override the *direct* head's contacts_plan source: "
            "'model'=as-is; 'gt'=use teacher soft contacts; "
            "'softgt'=use teacher contacts mapped into the model's typical soft range via --direct_pose_softgt_stats; "
            "'zero'=all zeros. Does not change contacts_plan/contacts_err/lambda (only direct hint)."
        ),
    )
    parser.add_argument(
        "--phase_z_ablate",
        type=str,
        default="none",
        choices=("none", "zero_ch0", "zero_ch1", "swap_ch01"),
        help=(
            "Ablation on the cached phase state passed into EventMotionModel.forward (phase_z). "
            "This affects any conditioning paths that consume phase_z (e.g. direct_pose_use_phase_z), "
            "and is intended for cross-leg context diagnosis.\n"
            "Assumes phase_z layout is [sinφ_ch0, cosφ_ch0, sinφ_ch1, cosφ_ch1, ...] (dim=2*contact_dim).\n"
            "none=disable; zero_ch0/zero_ch1 zero the (sin,cos) pair for that contact channel; "
            "swap_ch01 swaps ch0<->ch1 (requires contact_dim>=2)."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_cross_leg_ablate",
        type=str,
        default="none",
        choices=("none", "zero", "roll_batch", "roll_time"),
        help=(
            "Ablation (eval-only) for the *non-routed* direct_pose_leg_head (legacy/mixed L/R):\n"
            "Runs the leg head twice: (1) for right joints, it ablates the left-channel contact features; "
            "(2) for left joints, it ablates the right-channel contact features; then merges outputs back by bone side.\n"
            "Ablated features include contacts_plan, contacts_meas (when concatenated), and phase_z (sin/cos pairs) "
            "inside the leg-head input only (direct_out is unchanged).\n"
            "none=disable; zero=zero the opposite channel; roll_batch/roll_time roll the opposite channel to decorrelate."
        ),
    )
    parser.add_argument(
        "--direct_pose_leg_side_plan_other_ablate",
        type=str,
        default="none",
        choices=("none", "zero", "roll_batch", "roll_time"),
        help=(
            "Ablation (eval-only) for the *routed/shared* leg head when direct_pose_leg_side_plan_other=True.\n"
            "This edits only the plan_other scalar appended per side.\n"
            "none=disable; zero=zero plan_other; roll_batch/roll_time roll plan_other to decorrelate."
        ),
    )
    parser.add_argument(
        "--direct_pose_softgt_stats",
        type=str,
        default=None,
        help=(
            "Optional JSON payload or JSON file path for direct soft-GT hint mapping. "
            "Used when --direct_pose_plan_source=softgt and/or --direct_pose_meas_source=softgt.\n"
            "Expected format (logit-affine per channel):\n"
            "{\n"
            '  \"clamp\": 1e-4,\n'
            '  \"plan\": {\"scale\": [..], \"bias\": [..]},\n'
            '  \"meas\": {\"scale\": [..], \"bias\": [..]}\n'
            "}\n"
            "Mapping: p_soft = sigmoid(bias + scale * logit(clamp(p_gt)))."
        ),
    )
    parser.add_argument(
        "--direct_pose_hinge_enable",
        action="store_true",
        help="Enable hinge-style 1D correction for direct head (joint-local axis twist).",
    )
    parser.add_argument(
        "--direct_pose_hinge_bones",
        type=str,
        default="calf_r",
        help="Comma-separated bone names/indices for hinge correction (default: calf_r).",
    )
    parser.add_argument(
        "--direct_pose_hinge_axis",
        type=str,
        default="z",
        choices=("x", "y", "z"),
        help="Local axis for hinge correction (default: z).",
    )
    parser.add_argument(
        "--direct_pose_hinge_max_deg",
        type=float,
        default=45.0,
        help="Max hinge correction magnitude in degrees (tanh-scaled).",
    )
    parser.add_argument(
        "--direct_pose_hinge_hidden",
        type=int,
        default=0,
        help="Hidden dim for hinge head (0=auto).",
    )
    parser.add_argument(
        "--direct_pose_hinge_oracle_delta",
        action="store_true",
        help=(
            "Diagnostics: override hinge head output with an axis-oracle delta computed from GT at each step, "
            "then apply it via the normal hinge correction path (useful to verify apply/target consistency)."
        ),
    )
    parser.add_argument(
        "--so3_corr_max_deg",
        type=float,
        default=20.0,
        help="Max correction angle per step (deg) when --so3_corr_apply is set.",
    )
    parser.add_argument(
        "--so3_corr_gate_force",
        type=str,
        default=None,
        help="Force SO(3) gate scalar (float), or 'null'/'none' to use learned gate.",
    )
    parser.add_argument(
        "--so3_corr_gate_from_contacts_err",
        action="store_true",
        help="Override SO(3) gate per-step using |contacts_plan-contacts_meas| (requires both heads).",
    )
    parser.add_argument(
        "--so3_corr_gate_from_contacts_err_mode",
        type=str,
        default="scale",
        choices=("scale", "override"),
        help="How to apply contact error to SO(3) gate: 'scale' multiplies learned/forced gate; 'override' replaces it.",
    )
    parser.add_argument(
        "--so3_corr_gate_err_k",
        type=float,
        default=1.0,
        help="Gate = clamp(k*(ContactErrAbsMean-bias), 0, max) when --so3_corr_gate_from_contacts_err is set.",
    )
    parser.add_argument(
        "--so3_corr_gate_err_bias",
        type=float,
        default=0.0,
        help="Gate bias term for --so3_corr_gate_from_contacts_err.",
    )
    parser.add_argument(
        "--so3_corr_gate_err_max",
        type=float,
        default=1.0,
        help="Gate max clamp for --so3_corr_gate_from_contacts_err.",
    )
    parser.add_argument(
        "--so3_corr_gate_err_ref_steps",
        type=int,
        default=8,
        help="Number of initial steps used to estimate reference ContactErrAbsMean (gate forced to 0).",
    )
    parser.add_argument(
        "--so3_corr_gate_err_margin",
        type=float,
        default=0.0,
        help="Extra margin subtracted from (err-ref) before computing gate.",
    )
    parser.add_argument(
        "--so3_corr_gate_err_use_ref",
        action="store_true",
        help="Use (ContactErrAbsMean - ref) as gate signal (ref estimated over ref_steps). Default uses absolute err.",
    )
    parser.add_argument(
        "--so3_corr_gate_scale_max",
        type=float,
        default=2.0,
        help="Max multiplicative scale when --so3_corr_gate_from_contacts_err_mode=scale.",
    )
    parser.add_argument(
        "--log_contacts",
        action="store_true",
        help="Log contacts_plan/meas/err stats into metrics_per_step JSON (auto-enabled by --so3_corr_gate_from_contacts_err).",
    )
    parser.add_argument(
        "--log_contacts_whitebox",
        action="store_true",
        help=(
            "Attach per-foot white-box intermediates (dist/vel/sweep) into metrics_per_step JSON. "
            "Useful for debugging step-level collapses (hit_flag flip / ground_z drift)."
        ),
    )
    parser.add_argument(
        "--log_contacts_whitebox_first_steps",
        type=int,
        default=4,
        help="Always attach white-box debug payload for the first N free-run steps (still logs suspected collapse steps beyond N).",
    )
    parser.add_argument(
        "--contact_meas_gate_by_hit",
        type=str,
        default="auto",
        choices=("auto", "true", "false"),
        help="Override white-box gate_by_hit (auto uses bundle/meta). Set to 'false' for the first ablation pass.",
    )
    parser.add_argument(
        "--contact_meas_vxy_mode",
        type=str,
        default="abs",
        choices=("abs", "root_rel"),
        help="White-box vxy gate: abs uses ||v_foot_xy||, root_rel uses ||v_foot_xy - v_root_xy|| (more robust under translation).",
    )
    parser.add_argument(
        "--contact_meas_ground_z_mode",
        type=str,
        default="window",
        choices=("ema", "window", "slew"),
        help="White-box ground_z update mode (P2).",
    )
    parser.add_argument(
        "--contact_meas_ground_z_beta",
        type=float,
        default=0.05,
        help="EMA beta for --contact_meas_ground_z_mode=ema (higher adapts faster).",
    )
    parser.add_argument(
        "--contact_meas_ground_z_window",
        type=int,
        default=5,
        help="Window length for --contact_meas_ground_z_mode=window.",
    )
    parser.add_argument(
        "--contact_meas_ground_z_quantile",
        type=float,
        default=0.2,
        help="Low-quantile (0..1) over the window for --contact_meas_ground_z_mode=window. q=0.2 ignores single downward spikes when window=5.",
    )
    parser.add_argument(
        "--contact_meas_ground_z_slew_up_cm",
        type=float,
        default=0.0,
        help="Max upward change (cm per step) applied to ground_z after the chosen mode (0 disables).",
    )
    parser.add_argument(
        "--contact_meas_ground_z_slew_down_cm",
        type=float,
        default=0.0,
        help="Max downward change (cm per step) applied to ground_z after the chosen mode (0 disables).",
    )
    parser.add_argument(
        "--analyze_phase_shift",
        action="store_true",
        help=(
            "Analyze per-cycle phase shift between contacts_plan/meas and GT (MSE), and direct pose (GeoLocalDeg) if available. "
            "Writes a 'phase_shift' section into the output JSON."
        ),
    )
    parser.add_argument(
        "--phase_shift_max",
        type=int,
        default=None,
        help=(
            "Max signed shift to search when --analyze_phase_shift is set. "
            "If omitted, searches all shifts in [0, cycle_len)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_files = _expand_specs(args.teacher)
    if not teacher_files:
        raise SystemExit("[FATAL] No teacher JSON files matched the provided specs.")

    runner = FreeRunCycleRunner(args)
    out_dir = Path(args.out).expanduser().resolve()
    npz_root = Path(args.npz_root).expanduser().resolve()

    success = 0
    failures: List[str] = []
    for teacher_path in teacher_files:
        try:
            runner.run_clip(teacher_path, out_dir, npz_root, rounds=args.rounds)
            success += 1
        except Exception as exc:
            failures.append(f"{teacher_path}: {exc}")
            print(f"[ERR] {teacher_path}: {exc}")

    print(f"[Done] clips={success} ok / {len(failures)} failed")
    if failures:
        print("Failed clips:")
        for msg in failures:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()
