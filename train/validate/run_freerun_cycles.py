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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch  # ensure torch is bound before any inner scope uses it

from train.training_MPL import MotionEventDataset, Trainer, geodesic_R, validate_and_fix_model_
from train.geometry import rot6d_to_matrix, reproject_rot6d
from train.models import EventMotionModel, MotionJointLoss
from train.layout import LayoutCenter, DataNormalizer
from train.geometry import compose_rot6d_delta


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
    spec = dict(base)
    if pretrain_path and pretrain_path.is_file():
        with pretrain_path.open("r", encoding="utf-8") as f:
            pre = json.load(f)
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
            if k.startswith("frozen_encoder.") or k.startswith("frozen_period_head."):
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
        self.log_contacts = bool(
            getattr(args, "log_contacts", False) or self.so3_corr_gate_from_contacts_err or self.log_contacts_whitebox
        )
        gate_raw = str(getattr(args, "contact_meas_gate_by_hit", "auto") or "auto").strip().lower()
        if gate_raw in ("true", "1", "yes", "y"):
            self.contact_meas_gate_by_hit_override = True
        elif gate_raw in ("false", "0", "no", "n"):
            self.contact_meas_gate_by_hit_override = False
        else:
            self.contact_meas_gate_by_hit_override = None
        self.contact_meas_vxy_mode = str(getattr(args, "contact_meas_vxy_mode", "abs") or "abs").strip().lower()
        self.contact_meas_ground_z_select = str(getattr(args, "contact_meas_ground_z_select", "min") or "min").strip().lower()
        self.contact_meas_ground_z_mode = str(getattr(args, "contact_meas_ground_z_mode", "min") or "min").strip().lower()
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
        self.direct_pose_meas_force_zero = bool(getattr(args, "direct_pose_meas_force_zero", False))
        self.direct_pose_meas_source = str(getattr(args, "direct_pose_meas_source", "model") or "model").strip().lower()
        self.direct_pose_meas_warmup_steps = max(0, int(getattr(args, "direct_pose_meas_warmup_steps", 0) or 0))
        self.direct_pose_plan_source = str(getattr(args, "direct_pose_plan_source", "model") or "model").strip().lower()
        # Backward-compatible alias: --direct_pose_meas_force_zero ~= --direct_pose_meas_source=zero (unless explicitly overridden).
        if self.direct_pose_meas_force_zero and self.direct_pose_meas_source in ("", "model"):
            self.direct_pose_meas_source = "zero"
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
        # Infer contact-plan head output dim from checkpoint shapes (sigmoid heads only).
        contact_plan_head_mode = "sigmoid"
        try:
            w_head = self.state_dict.get("contact_plan_head.4.weight", None)
            if torch.is_tensor(w_head) and w_head.ndim == 2:
                out_dim = int(w_head.shape[0])
                if out_dim == 4 and int(self.contact_dim or 0) == 2:
                    raise SystemExit("[FATAL] joint4 contact_plan_head_mode is no longer supported.")
        except SystemExit:
            raise
        except Exception:
            pass

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

        contact_meas_enable = any(str(k).startswith("contact_meas_head.") for k in self.state_dict.keys())
        contact_meas_hidden = 64
        try:
            w1 = self.state_dict.get("contact_meas_head.1.weight", None)
            if torch.is_tensor(w1) and w1.ndim == 2:
                contact_meas_hidden = int(w1.shape[0])
        except Exception:
            pass

        # Infer direct pose head (cond + contacts_plan -> absolute pose).
        # If we don't instantiate this head, load_state_dict(strict=False) will warn about unexpected keys
        # and the runtime model won't expose `out_direct`.
        direct_pose_enable = False
        direct_pose_hidden = 256
        direct_pose_meas_mode = "none"
        try:
            direct_has_weights = any(str(k).startswith("direct_pose_head.") for k in self.state_dict.keys())
            if direct_has_weights and int(Dy) > 0 and int(Dc) > 0 and int(self.contact_dim) > 0:
                w_in = self.state_dict.get("direct_pose_head.0.weight", None)
                w_out = self.state_dict.get("direct_pose_head.6.weight", None)
                if torch.is_tensor(w_in) and w_in.ndim == 2 and torch.is_tensor(w_out) and w_out.ndim == 2:
                    in_dim = int(w_in.shape[1])
                    hid = int(w_in.shape[0])
                    out_dim = int(w_out.shape[0])
                    expected_in = int(Dc) + int(self.contact_dim)
                    expected_in_concat = int(Dc) + int(self.contact_dim) * 2
                    expected_out = int(Dy)
                    expected_out_modes = int(Dy) * 2
                    if hid > 0 and out_dim in (expected_out, expected_out_modes) and in_dim in (expected_in, expected_in_concat):
                        direct_pose_enable = True
                        direct_pose_hidden = hid
                        if in_dim == expected_in_concat and out_dim == expected_out:
                            direct_pose_meas_mode = "concat"
                        elif in_dim == expected_in and out_dim == expected_out_modes:
                            direct_pose_meas_mode = "mode_select"
                        else:
                            direct_pose_meas_mode = "none"
        except Exception:
            direct_pose_enable = False

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

        model = EventMotionModel(
            in_state_dim=Dx,
            out_motion_dim=Dy,
            cond_dim=Dc,
            period_dim=int(getattr(ds, "period_dim", 0) or self.period_dim),
            hidden_dim=self.width,
            num_layers=self.args.depth,
            num_heads=self.args.num_heads,
            dropout=self.args.dropout,
            context_len=self.args.context_len,
            contact_dim=self.contact_dim,
            angvel_dim=self.angvel_dim,
            pose_hist_dim=self.pose_hist_dim,
            bone_names=getattr(ds, "bone_names", None),
            output_layout=getattr(ds, "output_layout", None),
            contact_plan_enable=bool(contact_plan_enable or contact_plan_inject != "none" or direct_pose_enable),
            contact_plan_hidden=int(contact_plan_hidden),
            contact_plan_inject=str(contact_plan_inject),
            contact_plan_inject_detach=True,
            contact_plan_head_mode=str(contact_plan_head_mode),
            contact_plan_time_pe_dim=int(contact_plan_time_pe_dim),
            contact_plan_init_mode=str(contact_plan_init_mode),
            contact_plan_init_hidden=int(contact_plan_init_hidden),
            contact_plan_init_dropout=float(contact_plan_init_dropout),
            direct_pose_enable=bool(direct_pose_enable),
            direct_pose_hidden=int(direct_pose_hidden),
            direct_pose_dropout=0.0,
            direct_pose_detach_plan=True,
            direct_pose_meas_mode=str(direct_pose_meas_mode),
            direct_pose_meas_drop_prob=0.0,
            direct_pose_meas_noise_std=0.0,
            direct_pose_plan_drop_prob=0.0,
            lambda_fusion_enable=bool(lambda_fusion_enable),
            lambda_fusion_mode=str(lambda_fusion_mode),
            lambda_fusion_hidden=int(lambda_fusion_hidden),
            lambda_fusion_dropout=0.0,
            lambda_fusion_detach_err=True,
            lambda_fusion_logit_init=-2.0,
            lambda_fusion_use_rollout_step=bool(lambda_fusion_use_rollout_step),
            contact_meas_enable=bool(contact_meas_enable),
            contact_meas_hidden=int(contact_meas_hidden),
            contact_meas_dropout=0.0,
            # If a learned meas head exists, don't override it with external contacts/whitebox.
            contacts_as_meas_override=bool(contact_plan_enable or contact_plan_inject != "none") and (not bool(contact_meas_enable)),
        ).to(self.device)
        # Validate basic shapes then load weights (allow extra frozen encoder keys).
        validate_and_fix_model_(model, Dx, Dc)
        missing, unexpected = model.load_state_dict(self.state_dict, strict=False)
        if missing or unexpected:
            print(f"[FreeRun][WARN] state_dict mismatch: missing={missing}, unexpected={unexpected}")
        if bool(getattr(self, "direct_pose_meas_force_zero", False)):
            setattr(model, "direct_pose_meas_force_zero", True)
        # Optional frozen motion encoder
        if self.encoder_bundle_path and self.encoder_bundle_path.is_file():
            model.attach_motion_encoder(torch.load(str(self.encoder_bundle_path), map_location="cpu"))
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
        trainer.contact_meas_ground_z_select = str(getattr(self, "contact_meas_ground_z_select", "min") or "min")
        trainer.contact_meas_ground_z_mode = str(getattr(self, "contact_meas_ground_z_mode", "min") or "min")
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
        clip = ds.clips[0]

        # Construct a single "full‑cycle" sample equivalent to MotionEventDataset.__getitem__ at s=0.
        base_sample = _build_full_cycle_sample(ds, clip, seq_len=T_base)

        # Run free‑run for N cycles without reset.
        metrics_per_round, per_step, extra = _run_freerun_cycles(
            trainer=self.trainer,
            sample=base_sample,
            rounds=rounds,
            device=self.device,
            time_index_mode=str(getattr(self.args, "time_index_mode", "auto") or "auto"),
            round_seg_mode=str(getattr(self.args, "round_seg_mode", "intra") or "intra"),
            lambda_fusion_apply=bool(self.lambda_fusion_apply),
            export_joint_geolocal=bool(getattr(self.args, "export_joint_geolocal", False)),
            direct_align_inc0=bool(getattr(self.args, "direct_align_inc0", False)),
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
            "fps": data.get("fps", getattr(ds, "fps", 60.0)),
            "cycle_len": int(T_base),
            "rounds": rounds,
            "time_index_mode": str(getattr(self.args, "time_index_mode", "auto") or "auto"),
            "round_seg_mode": str(getattr(self.args, "round_seg_mode", "intra") or "intra"),
            "contact_plan_init_mode": str(getattr(self.model, "contact_plan_init_mode", None) or "unknown"),
            "contact_plan_init_hidden": int(getattr(self.model, "contact_plan_init_hidden", 0) or 0),
            "contact_plan_init_dropout": float(getattr(self.model, "_contact_plan_init_dropout", 0.0) or 0.0),
            "contact_plan_head_mode": str(getattr(self.model, "contact_plan_head_mode", None) or "unknown"),
            "contact_plan_init_mode_override": getattr(self, "contact_plan_init_mode_override", None),
            "lambda_fusion_apply": bool(self.lambda_fusion_apply),
            "so3_corr_apply": bool(getattr(self.trainer, "so3_corr_apply", False)) if self.trainer is not None else False,
            "direct_align_inc0": bool(getattr(self.args, "direct_align_inc0", False)),
            "direct_pose_meas_force_zero": bool(getattr(self, "direct_pose_meas_force_zero", False)),
            "direct_pose_meas_source": str(getattr(self, "direct_pose_meas_source", "model") or "model"),
            "direct_pose_meas_warmup_steps": int(getattr(self, "direct_pose_meas_warmup_steps", 0) or 0),
            "direct_pose_plan_source": str(getattr(self, "direct_pose_plan_source", "model") or "model"),
            "lambda_reliability_mode": str(getattr(self, "lambda_reliability_mode", "none") or "none"),
            "lambda_reliability_warmup_steps": int(getattr(self, "lambda_reliability_warmup_steps", 0) or 0),
            "lambda_reliability_contact_err_max": float(getattr(self, "lambda_reliability_contact_err_max", 1.0) or 1.0),
            "lambda_reliability_warmup_joint_scales": getattr(self, "lambda_reliability_warmup_joint_scales", None),
            "contact_meas_gate_by_hit": getattr(self, "contact_meas_gate_by_hit_override", None),
            "contact_meas_vxy_mode": str(getattr(self, "contact_meas_vxy_mode", "abs") or "abs"),
            "contact_meas_ground_z_select": str(getattr(self, "contact_meas_ground_z_select", "min") or "min"),
            "contact_meas_ground_z_mode": str(getattr(self, "contact_meas_ground_z_mode", "min") or "min"),
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
    time_index_mode: str = "auto",
    round_seg_mode: str = "intra",
    lambda_fusion_apply: bool = False,
    export_joint_geolocal: bool = False,
    direct_align_inc0: bool = False,
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
    if rounds <= 0:
        raise ValueError("rounds must be > 0")

    # Move base sequences to device and tile along time.
    state_seq_base = sample["motion"].unsqueeze(0).to(device)  # [1, T, Dx]
    gt_seq_base = sample["gt_motion"].unsqueeze(0).to(device)  # [1, T, Dy]
    T_cycle = state_seq_base.shape[1]
    T_total = T_cycle * rounds

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
    pose_hist_buffer_norm = None
    pose_hist_buffer_raw = None
    scales = mu = std = None
    if pose_hist_enabled:
        scales, mu, std = trainer._pose_hist_params(state_seq)
        if scales is None:
            pose_hist_enabled = False
        else:
            if pose_hist_seq is not None and pose_hist_seq.dim() == 3 and pose_hist_seq.size(1) > 0:
                initial_norm = pose_hist_seq[:, 0]
            elif pose_hist_seq is not None:
                initial_norm = pose_hist_seq
            else:
                initial_norm = torch.zeros((B, pose_hist_dim), device=device, dtype=state_seq.dtype)
            pose_hist_buffer_norm = initial_norm
            pose_hist_buffer_raw = trainer._pose_hist_inverse_vec(initial_norm, scales, mu, std)

    B, T, Dx = state_seq.shape
    assert T == T_total, "Internal error: tiled length mismatch."
    if T < 2:
        raise ValueError("Sequence too short for free‑run (need at least 2 frames).")

    time_index_mode = str(time_index_mode or "auto").strip().lower()
    round_seg_mode = str(round_seg_mode or "intra").strip().lower()

    warmup = 0
    start_t = warmup
    end_t = T - 1  # last usable index for t+1

    model = trainer.model
    predsY: List[torch.Tensor] = []  # incremental (Δ) absolute pose (y_norm), not necessarily used for update
    predsY_blend: List[torch.Tensor] = []  # blended absolute pose (y_norm)
    predsY_direct: List[torch.Tensor] = []
    predsX: List[torch.Tensor] = []
    contacts_log: List[Optional[Dict[str, Any]]] = []
    time_index_log: List[Optional[int]] = []
    lambda_log: List[Optional[torch.Tensor]] = []  # (B,J) on CPU
    lambda_eff_log: List[Optional[torch.Tensor]] = []  # (B,J) on CPU (after r_t)
    lambda_rel_log: List[Optional[torch.Tensor]] = []  # (B,) on CPU (r_t)

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

    for t in range(start_t, end_t):
        cond_input = cond_seq[:, t] if (cond_seq is not None and cond_seq.dim() == 3) else cond_seq
        if getattr(trainer, "use_freerun_state_sync", False) and isinstance(getattr(trainer, "angvel_x_slice", None), slice):
            angvel_t = motion[..., trainer.angvel_x_slice].detach()
        else:
            angvel_t = angvel_seq[:, t] if (angvel_seq is not None and angvel_seq.dim() == 3) else angvel_seq
        if pose_hist_enabled:
            pose_hist_t = pose_hist_buffer_norm
        else:
            pose_hist_t = pose_hist_seq[:, t] if (pose_hist_seq is not None and pose_hist_seq.dim() == 3) else pose_hist_seq
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
        if cond_raw_for_model is not None and t > 0:
            try:
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
            except Exception:
                pass

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
            time_index_t = int(t % T_cycle)
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
        # This does NOT affect contacts_err/lambda (those always use model-produced contacts_meas).
        direct_meas_source_eff = str(getattr(trainer, "direct_pose_meas_source", "model") or "model").strip().lower()
        direct_meas_warmup = int(getattr(trainer, "direct_pose_meas_warmup_steps", 0) or 0)
        step_idx = int(t - start_t)
        if direct_meas_warmup > 0 and step_idx >= direct_meas_warmup:
            direct_meas_source_eff = "zero"

        # Compute whitebox contacts only when needed:
        # - model has no learned meas head (fallback),
        # - direct head explicitly requests whitebox,
        # - plan init_mode is obs-based (t==0 only),
        # - whitebox debug logging enabled.
        use_learned_meas = bool(getattr(model, "contact_meas_enable", False))
        init_mode = str(getattr(model, "contact_plan_init_mode", "learnable") or "learnable").strip().lower()
        log_wb = bool(getattr(trainer, "log_contacts_whitebox", False))
        if log_wb:
            try:
                setattr(trainer, "_contact_meas_whitebox_debug", None)
            except Exception:
                pass

        need_wb = bool(plan_enable) and (
            (not use_learned_meas)
            or (direct_meas_source_eff in ("whitebox", "wb"))
            or (init_mode in ("obs", "learnable+obs") and plan_z is None and step_idx == 0)
            or log_wb
        )
        contacts_wb_t = None
        if need_wb:
            try:
                contacts_wb_t, prev_foot_pos_meas = trainer._contact_meas_whitebox(motion_raw, prev_foot_pos_meas)
            except Exception:
                contacts_wb_t = None

        contacts_in_t = None
        if plan_enable:
            if not use_learned_meas:
                contacts_in_t = contacts_wb_t
            elif init_mode in ("obs", "learnable+obs") and plan_z is None and step_idx == 0:
                # Feed contacts only for plan_z0 init (won't override the learned meas head).
                contacts_in_t = contacts_wb_t

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
        else:
            direct_plan_override = None
        try:
            setattr(model, "direct_pose_plan_override", direct_plan_override)
        except Exception:
            pass

        with amp_ctx:
            ret = model(
                motion,
                cond_input,
                contacts=contacts_in_t,
                angvel=angvel_t,
                pose_history=pose_hist_t,
                plan_z=plan_z,
                time_index=time_index_t,
                rollout_step=rollout_step_t,
            )

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
                meas_logits = ret.get("contacts_meas_logits", None)
                plan_per_c = None
                meas_per_c = None
                err_per_c = None
                plan_logits_mean = None
                plan_logits_std = None
                plan_logits_per_c = None
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
                    "ContactPlanMean": plan_mean,
                    "ContactPlanAbsMean": plan_abs_mean,
                    "ContactPlanPerC": plan_per_c,
                    "ContactMeasMean": meas_mean,
                    "ContactMeasAbsMean": meas_abs_mean,
                    "ContactMeasPerC": meas_per_c,
                    "ContactPlanLogitsMean": plan_logits_mean,
                    "ContactPlanLogitsStd": plan_logits_std,
                    "ContactPlanLogitsPerC": plan_logits_per_c,
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
                    and bool(getattr(model, "contact_meas_enable", False))
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
        predsY.append(y_inc_norm)

        try:
            y_blend_norm = trainer._norm_y(y_blend_raw)
        except Exception:
            y_blend_norm = y_inc_norm
        predsY_blend.append(y_blend_norm)

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
        else:
            motion = trainer._apply_free_carry(motion, y_used_raw, cond_next_raw=None).detach()

        predsX.append(motion)

        # Align contact logging with predsY/predsX timeline.
        contacts_log.append(contact_entry)

        if pose_hist_enabled and pose_hist_stride > 0:
            with torch.no_grad():
                pose_hist_buffer_raw = torch.roll(pose_hist_buffer_raw, shifts=-pose_hist_stride, dims=-1)
                rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
                if isinstance(rot_slice, slice):
                    pose_hist_buffer_raw[..., -pose_hist_stride:] = y_used_raw[..., rot_slice]
                pose_hist_buffer_norm = trainer._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)

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
        key_bone_names = getattr(loss_fn, "eval_key_bones", None) or [
            "pelvis",
            "upperarm_l", "lowerarm_l", "hand_l",
            "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l",
            "thigh_r", "calf_r", "foot_r",
        ]
        idx_map = {name: idx for idx, name in enumerate(bone_names)}
        key_indices = [idx_map[name] for name in key_bone_names if name in idx_map]
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
        if pred_blend_raw_full is not None:
            try:
                b6_full = pred_blend_raw_full[..., rot_slice].view(1, free_steps, J, 6)
                b6_full = reproject_rot6d(b6_full)
                Rb_full = rot6d_to_matrix(b6_full)  # [1, free_steps, J, 3, 3]
                geo_blend_full = geodesic_R(Rb_full, Rg_full) * deg_factor
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

        # Direct head diagnostics (if available): absolute pose prediction that does NOT use y_{t-1}.
        geo_direct_full = None
        geo_direct_full_aligned0 = None
        geo_direct_local_full = None
        geo_direct_full_align_inc0 = None
        geo_direct_local_full_align_inc0 = None
        if pred_direct_raw_full is not None:
            try:
                d6_full = pred_direct_raw_full[..., rot_slice].view(1, free_steps, J, 6)
                d6_full = reproject_rot6d(d6_full)
                Rd_full = rot6d_to_matrix(d6_full)  # [1, free_steps, J, 3, 3]
                geo_direct_full = geodesic_R(Rd_full, Rg_full) * deg_factor
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
    else:
        root_idx = 0
        joint_mask = None
        geo_full = None
        geo_full_aligned0 = None
        geo_local_full = None
        geo_blend_full = None
        geo_blend_full_aligned0 = None
        geo_blend_local_full = None
        geo_direct_full = None
        geo_direct_full_aligned0 = None
        geo_direct_local_full = None
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
            if dloc is not None:
                try:
                    per_joint["DirectGeoLocalDegMean"] = dloc.mean(dim=0).detach().cpu().tolist()
                    per_joint["DirectGeoLocalDegEarlyMean"] = dloc[:k].mean(dim=0).detach().cpu().tolist()
                    per_joint["DirectGeoLocalDegLateMean"] = (
                        dloc[-k:].mean(dim=0) if steps_total >= k else dloc[:k].mean(dim=0)
                    ).detach().cpu().tolist()
                except Exception:
                    pass
            if dloc_align_inc0 is not None:
                try:
                    per_joint["DirectGeoLocalDegAlignInc0Mean"] = dloc_align_inc0.mean(dim=0).detach().cpu().tolist()
                    per_joint["DirectGeoLocalDegAlignInc0EarlyMean"] = dloc_align_inc0[:k].mean(dim=0).detach().cpu().tolist()
                    per_joint["DirectGeoLocalDegAlignInc0LateMean"] = (
                        dloc_align_inc0[-k:].mean(dim=0) if steps_total >= k else dloc_align_inc0[:k].mean(dim=0)
                    ).detach().cpu().tolist()
                except Exception:
                    pass
            if bloc is not None:
                try:
                    per_joint["BlendGeoLocalDegMean"] = bloc.mean(dim=0).detach().cpu().tolist()
                    per_joint["BlendGeoLocalDegEarlyMean"] = bloc[:k].mean(dim=0).detach().cpu().tolist()
                    per_joint["BlendGeoLocalDegLateMean"] = (
                        bloc[-k:].mean(dim=0) if steps_total >= k else bloc[:k].mean(dim=0)
                    ).detach().cpu().tolist()
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

    for t in range(free_steps):
        geo_t = None
        geo_local_t = None
        geo_local_weighted_t = None
        root_geo_t = None
        geo_aligned0_t = None
        root_geo_aligned0_t = None
        keybone_geo: Dict[str, float] = {}
        keybone_geo_local: Dict[str, float] = {}
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
            if geo_local_full is not None and w_joint is not None and weights_sum is not None:
                # Weighted GeoLocalDeg (matches Trainer)
                geo_local_weighted_t = float(
                    ((geo_local_full[:, t] * w_joint).sum(dim=-1) / weights_sum).mean().item()
                )

        direct_geo_t = None
        direct_geo_aligned0_t = None
        direct_geo_local_t = None
        direct_geo_local_weighted_t = None
        direct_root_geo_t = None
        direct_root_geo_aligned0_t = None
        if geo_direct_full is not None:
            direct_geo_t = float(geo_direct_full[:, t].mean().item())
            direct_root_geo_t = float(geo_direct_full[:, t, root_idx].mean().item())
        if geo_direct_full_aligned0 is not None:
            direct_geo_aligned0_t = float(geo_direct_full_aligned0[:, t].mean().item())
            direct_root_geo_aligned0_t = float(geo_direct_full_aligned0[:, t, root_idx].mean().item())
        if geo_direct_local_full is not None:
            if joint_mask is not None and joint_mask.any():
                direct_geo_local_t = float(geo_direct_local_full[:, t, joint_mask].mean().item())
            else:
                direct_geo_local_t = 0.0
            if w_joint is not None and weights_sum is not None:
                direct_geo_local_weighted_t = float(
                    ((geo_direct_local_full[:, t] * w_joint).sum(dim=-1) / weights_sum).mean().item()
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
        blend_root_geo_t = None
        blend_root_geo_aligned0_t = None
        if geo_blend_full is not None:
            blend_geo_t = float(geo_blend_full[:, t].mean().item())
            blend_root_geo_t = float(geo_blend_full[:, t, root_idx].mean().item())
        if geo_blend_full_aligned0 is not None:
            blend_geo_aligned0_t = float(geo_blend_full_aligned0[:, t].mean().item())
            blend_root_geo_aligned0_t = float(geo_blend_full_aligned0[:, t, root_idx].mean().item())
        if geo_blend_local_full is not None:
            if joint_mask is not None and joint_mask.any():
                blend_geo_local_t = float(geo_blend_local_full[:, t, joint_mask].mean().item())
            else:
                blend_geo_local_t = 0.0
            if w_joint is not None and weights_sum is not None:
                blend_geo_local_weighted_t = float(
                    ((geo_blend_local_full[:, t] * w_joint).sum(dim=-1) / weights_sum).mean().item()
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

        lam_eff_mean_t = lam_eff_std_t = None
        if lambda_eff_steps:
            lam_t = lambda_eff_steps[t] if t < len(lambda_eff_steps) else None
            if torch.is_tensor(lam_t):
                try:
                    lam_eff_mean_t = float(lam_t.mean().item())
                    lam_eff_std_t = float(lam_t.std(unbiased=False).item())
                except Exception:
                    lam_eff_mean_t = lam_eff_std_t = None

        lam_rel_mean_t = None
        if lambda_rel_steps:
            rel_t = lambda_rel_steps[t] if t < len(lambda_rel_steps) else None
            if torch.is_tensor(rel_t):
                try:
                    lam_rel_mean_t = float(rel_t.mean().item())
                except Exception:
                    lam_rel_mean_t = None
        entry: Dict[str, Any] = {
            "step": int(t),
            "time_index": int(time_index_steps[t]) if (time_index_steps and t < len(time_index_steps) and time_index_steps[t] is not None) else None,
            "GeoDeg": geo_t,
            "GeoDegAligned0": geo_aligned0_t,
            "GeoLocalDeg": geo_local_t,
            "GeoLocalDegWeighted": geo_local_weighted_t,
            "RootGeoDeg": root_geo_t,
            "RootGeoDegAligned0": root_geo_aligned0_t,
            "BlendGeoDeg": blend_geo_t,
            "BlendGeoDegAligned0": blend_geo_aligned0_t,
            "BlendGeoLocalDeg": blend_geo_local_t,
            "BlendGeoLocalDegWeighted": blend_geo_local_weighted_t,
            "BlendRootGeoDeg": blend_root_geo_t,
            "BlendRootGeoDegAligned0": blend_root_geo_aligned0_t,
            "DirectGeoDeg": direct_geo_t,
            "DirectGeoDegAligned0": direct_geo_aligned0_t,
            "DirectGeoLocalDeg": direct_geo_local_t,
            "DirectGeoLocalDegWeighted": direct_geo_local_weighted_t,
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
            "RootPosErr": float(root_pos_err[t].item()) if root_pos_err is not None else None,
            "RootVelMAE": float(root_vel_mae[t].item()) if root_vel_mae is not None else None,
        }
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
                    "ContactPlanMean",
                    "ContactPlanAbsMean",
                    "ContactPlanPerC",
                    "ContactPlanLogitsMean",
                    "ContactPlanLogitsStd",
                    "ContactPlanLogitsPerC",
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
        if round_seg_mode == "legacy":
            # Legacy behavior: treat each round as T_cycle steps starting at r*T_cycle.
            # This includes the boundary transition at t=(r+1)*T_cycle-1 (wrap into next round),
            # and the last round may be short by 1 due to the (T_total-1) evaluation horizon.
            t0 = r * T_cycle
            t1 = min((r + 1) * T_cycle, free_steps)
        else:
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
        root_geo_deg_val: Optional[float] = None
        root_geo_deg_start: Optional[float] = None
        root_geo_deg_end: Optional[float] = None
        root_geo_deg_aligned0_val: Optional[float] = None
        root_geo_deg_aligned0_start: Optional[float] = None
        root_geo_deg_aligned0_end: Optional[float] = None
        keybone_geo_mean: Optional[float] = None
        keybone_geo_local_mean: Optional[float] = None
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
            "round_seg_mode": str(round_seg_mode),
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
        metrics_per_round.append(round_entry)

    return metrics_per_round, per_step, extra


# ---- CLI --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
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
        default="models/motion_encoder_equiv.pt",
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
        "--round-seg-mode",
        type=str,
        default="intra",
        choices=("legacy", "intra"),
        help=(
            "How to segment the (T_total-1) transition steps into rounds. "
            "'intra' reports per-round metrics over within-cycle transitions only (T_cycle-1) and drops wrap boundaries; "
            "'legacy' keeps the old [r*T_cycle:(r+1)*T_cycle] slicing (Round0 may include one boundary step)."
        ),
    )
    parser.add_argument(
        "--so3_corr_apply",
        action="store_true",
        help="Apply SO(3) corrector during compose (uses model omega_hat).",
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
        "--direct_align_inc0",
        action="store_true",
        help=(
            "Diagnostics only: align the direct head's per-joint rotations by a constant bias computed at step0 "
            "(R_bias[j]=R_inc0[j]@R_dir0[j]^T) and report Direct*AlignInc0 metrics. "
            "Useful to verify whether direct early errors are mainly phase/anchor offsets."
        ),
    )
    parser.add_argument(
        "--direct_pose_meas_force_zero",
        action="store_true",
        help="Ablation: force direct head to ignore contacts_meas (concat->zeros, mode_select->uniform).",
    )
    parser.add_argument(
        "--direct_pose_meas_source",
        type=str,
        default="model",
        choices=("model", "whitebox", "gt", "zero"),
        help=(
            "Override the *direct* head's contacts_meas source: "
            "'model'=as-is; 'whitebox'=use whitebox contacts_in_t; 'gt'=use teacher soft contacts; 'zero'=ignore. "
            "Does not change contacts_err/lambda (only direct hint)."
        ),
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
        choices=("model", "gt", "zero"),
        help=(
            "Override the *direct* head's contacts_plan source: "
            "'model'=as-is; 'gt'=use teacher soft contacts; 'zero'=all zeros. "
            "Does not change contacts_plan/contacts_err/lambda (only direct hint)."
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
        "--contact_meas_ground_z_select",
        type=str,
        default="min",
        choices=("min", "stance"),
        help="How to compute ground_z_now from feet: min=legacy min(bottom_z), stance=choose most stance-like foot by low speeds.",
    )
    parser.add_argument(
        "--contact_meas_ground_z_mode",
        type=str,
        default="min",
        choices=("min", "ema", "window", "slew"),
        help="White-box ground_z update mode (P2). Default 'min' matches the legacy monotonic-min behavior.",
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
