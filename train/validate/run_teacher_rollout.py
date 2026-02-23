#!/usr/bin/env python3
"""
Roll out a trained MPL model on pre-exported teacher batches and write predictions to JSON.

The script feeds ground-truth state+condition pairs (x+c) from validate/teacher_batches/*.json
into a specified checkpoint, captures the normalized (and optional denormalized) Y outputs, and
stores them alongside simple diagnostics so UE-side inference can replay the model output.

Example:
    python train/validate/run_teacher_rollout.py \
        --model models/MLPNoDryRun/exp_phase_MLP/ckpt_best_exp_phase_MLP.pth \
        --teacher validate/teacher_batches/Walk_F_teacher.json \
        --bundle raw_data/processed_data/norm_template.json \
        --pretrain-template models/pretrain_template.json \
        --npz-root raw_data/processed_data \
        --out validate/teacher_predictions
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train.training_MPL import MotionEventDataset, Trainer, validate_and_fix_model_, geodesic_R
from train.geometry import rot6d_to_matrix, matrix_to_rot6d, reproject_rot6d, normalize_rot6d_delta
from train.models import EventMotionModel, MotionJointLoss
from train.layout import LayoutCenter, DataNormalizer
from train.geometry import compose_rot6d_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run teacher-forced rollouts for UE teacher batches using a trained MPL checkpoint.",
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
        default=None,
        help="Path to checkpoint (.pth) that contains {'model': state_dict}. Required unless --onnx-model is provided.",
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
        "--onnx-model",
        type=str,
        default=None,
        help="If set, use this ONNX model for inference instead of a PyTorch checkpoint.",
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
        default="validate/teacher_predictions",
        help="Directory to store rollout JSON files.",
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
        "--direct_pose_meas_force_zero",
        action="store_true",
        help="Ablation: force direct head to ignore contacts_meas (concat->zeros, mode_select->uniform).",
    )
    parser.add_argument(
        "--angvel_source",
        type=str,
        default="state",
        choices=("state", "seq"),
        help=(
            "Which angvel signal to feed into the model. "
            "'state' uses angvel from the X-state slice (Trainer.use_freerun_state_sync=True). "
            "'seq' uses the precomputed angvel_norm from the dataset (enables --angvel_ablation)."
        ),
    )
    parser.add_argument(
        "--pose_hist_source",
        type=str,
        default="buffer",
        choices=("buffer", "seq"),
        help=(
            "Which pose_hist signal to feed into the model. "
            "'buffer' uses Trainer's internal rolling history (built from y_raw each step; matches deployment). "
            "'seq' uses the precomputed pose_hist_norm from the dataset (lets you ablate/sweep history blocks)."
        ),
    )
    parser.add_argument(
        "--pose_hist_ablation",
        type=str,
        default="none",
        choices=("none", "zero", "keep_last", "replicate_last", "replicate_oldest"),
        help=(
            "Ablation on pose_hist input (per-step, before passing into the model). "
            "'keep_last' keeps only the most recent history block (zeros older ones). "
            "'replicate_*' copies one block into all blocks to keep scale similar."
        ),
    )
    parser.add_argument(
        "--pose_hist_keep_last",
        type=int,
        default=1,
        help="When --pose_hist_ablation=keep_last, keep the last K history blocks (K in [1, pose_hist_len]).",
    )
    parser.add_argument(
        "--pose_hist_time_shift",
        type=int,
        default=0,
        help=(
            "Time-shift pose_hist along the sequence axis before rollout. "
            "Positive = use earlier frames (delay input), Negative = use later frames (lookahead)."
        ),
    )
    parser.add_argument(
        "--angvel_ablation",
        type=str,
        default="none",
        choices=("none", "zero"),
        help="Ablation on angvel input (per-step, before passing into the model).",
    )
    parser.add_argument(
        "--with-denorm",
        action="store_true",
        help="Include denormalized predictions (rot6d raw) in the output JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prediction JSON files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output (only warnings/errors).",
    )
    return parser.parse_args()


def expand_specs(specs: Sequence[str]) -> List[Path]:
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


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_norm_spec(bundle_path: Path, pretrain_path: Optional[Path]) -> Dict[str, object]:
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


def resolve_npz_path(clip_name: str, source_json: Optional[str], npz_root: Path) -> Path:
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


def _shift_time_axis(x: np.ndarray, shift: int) -> np.ndarray:
    """Shift along time axis with edge padding. new[t] = old[clip(t - shift)]."""
    if x is None or not isinstance(x, np.ndarray):
        return x
    shift = int(shift)
    if shift == 0:
        return x
    T = int(x.shape[0])
    if T <= 0:
        return x
    idx = np.arange(T, dtype=np.int64) - shift
    idx = np.clip(idx, 0, T - 1)
    return x[idx]


def _ablate_pose_hist(
    pose_hist: np.ndarray,
    *,
    pose_hist_len: int,
    mode: str,
    keep_last: int,
) -> np.ndarray:
    """Ablate flattened pose_hist = [older ... newer] blocks per timestep."""
    if pose_hist is None or not isinstance(pose_hist, np.ndarray):
        return pose_hist
    if pose_hist.size == 0:
        return pose_hist
    mode = str(mode or "none").lower().strip()
    if mode in ("none", ""):
        return pose_hist
    if mode == "zero":
        return np.zeros_like(pose_hist)

    L = int(pose_hist_len)
    if L <= 0:
        return pose_hist
    D = int(pose_hist.shape[1])
    if D % L != 0:
        return pose_hist
    pose_dim = D // L
    hist = pose_hist.reshape(pose_hist.shape[0], L, pose_dim).copy()

    if mode == "keep_last":
        k = int(keep_last)
        k = max(1, min(L, k))
        if L - k > 0:
            hist[:, : L - k, :] = 0.0
        return hist.reshape(pose_hist.shape[0], D)

    if mode == "replicate_last":
        src = hist[:, -1:, :].copy()
        hist[:, :, :] = src
        return hist.reshape(pose_hist.shape[0], D)

    if mode == "replicate_oldest":
        src = hist[:, :1, :].copy()
        hist[:, :, :] = src
        return hist.reshape(pose_hist.shape[0], D)

    return pose_hist


class TeacherRolloutRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        if not args.model and not args.onnx_model:
            raise SystemExit("[FATAL] --model or --onnx-model must be specified.")
        self.use_onnx = bool(args.onnx_model)
        self.device = self._resolve_device(args.device if not self.use_onnx else "cpu")
        self.bundle_path = Path(args.bundle).expanduser().resolve()
        self.bundle = LayoutCenter(str(self.bundle_path))
        pretrain_path = Path(args.pretrain_template).expanduser()
        self.norm_spec = merge_norm_spec(self.bundle_path, pretrain_path if pretrain_path.is_file() else None)
        self.pose_hist_len = int(self.norm_spec.get("pose_hist_len", 0) or 0)
        self.ckpt = None
        self.state_dict = None
        self.onnx_path = Path(args.onnx_model).expanduser().resolve() if self.use_onnx else None
        self.ort_session = None
        self.ort_input_map: dict[str, str] = {}
        self.ort_output_name: Optional[str] = None
        if not self.use_onnx:
            self.ckpt = torch.load(Path(args.model).expanduser(), map_location="cpu")
            raw_state = self.ckpt["model"] if isinstance(self.ckpt, dict) and "model" in self.ckpt else self.ckpt
            # Drop frozen_encoder / frozen_period_head weights that are not part of the runtime model to avoid
            # mismatch errors. These are attached separately via --encoder-bundle for evaluation/export.
            self.state_dict = {}
            skipped = 0
            for k, v in raw_state.items():
                if (
                    str(k).startswith("frozen_encoder.")
                    or str(k).startswith("frozen_period_head.")
                    or str(k).startswith("contact_plan_input_proj.")
                ):
                    skipped += 1
                    continue
                self.state_dict[k] = v
            if skipped > 0:
                print(f"[TeacherRollout][INFO] stripped {skipped} frozen encoder keys from checkpoint for runtime load.")
        self.width = self._infer_width() if not self.use_onnx else None
        self.period_dim = self._infer_period_dim() if not self.use_onnx else 0
        self.encoder_bundle_path = Path(args.encoder_bundle).expanduser() if args.encoder_bundle else None
        self.model: Optional[EventMotionModel] = None
        self.loss_fn: Optional[MotionJointLoss] = None
        self.trainer: Optional[Trainer] = None
        self.contact_dim: Optional[int] = None
        self.angvel_dim: Optional[int] = None
        self.pose_hist_dim: Optional[int] = None
        self.dataset_pose_norm = None
        self.angvel_meta = {
            "mode": None,
            "mu": None,
            "std": None,
        }
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

    def _build_dataset(self, npz_path: Path) -> Tuple[MotionEventDataset, object]:
        ds = MotionEventDataset(
            data_dir=str(npz_path.parent),
            seq_len=max(2, self.pose_hist_len + 1),
            paths=[str(npz_path)],
            pose_hist_len=self.pose_hist_len,
            norm_spec=self.norm_spec,
        )
        if not ds.clips:
            raise RuntimeError(f"No clips loaded from {npz_path}")
        clip = ds.clips[0]
        return ds, clip

    def _ensure_model_ready(self, ds: MotionEventDataset) -> None:
        Dx, Dy, Dc = int(ds.Dx), int(ds.Dy), int(ds.Dc)
        self.contact_dim = int(getattr(ds, "contact_dim", 0))
        self.angvel_dim = int(getattr(ds, "angvel_dim", 0))
        self.pose_hist_dim = int(getattr(ds, "pose_hist_dim", 0))
        self.bundle.strict_validate(Dx, Dy)
        if self.use_onnx:
            if self.normalizer is None:
                self.normalizer = DataNormalizer(
                    mu_x=self.bundle.mu_x,
                    std_x=self.bundle.std_x,
                    mu_y=self.bundle.mu_y,
                    std_y=self.bundle.std_y,
                    y_to_x_map=self.bundle.materialize_y_to_x_map(),
                    yaw_x_slice=None,
                    yaw_y_slice=None,
                    rootvel_x_slice=self.bundle.state_layout.get("RootVelocity"),
                    rootvel_y_slice=self.bundle.output_layout.get("RootVelocity"),
                    angvel_x_slice=self.bundle.state_layout.get("BoneAngularVelocities"),
                    angvel_y_slice=self.bundle.output_layout.get("BoneAngularVelocities"),
                    tanh_scales_rootvel=self.bundle.tanh_scales_rootvel,
                    tanh_scales_angvel=self.bundle.tanh_scales_angvel,
                    angvel_mode=getattr(ds, "angvel_norm_mode", None),
                    angvel_mu=getattr(ds, "angvel_mu", None),
                    angvel_std=getattr(ds, "angvel_std", None),
                )
            if self.ort_session is None:
                self._init_onnx_session()
            return

        if self.model is not None:
            return

        # ---- Infer optional heads from checkpoint (plan / meas / direct pose) ----
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
        try:
            init_has_weights = any(str(k).startswith("contact_plan_init_head.") for k in self.state_dict.keys())
            if init_has_weights:
                contact_plan_init_mode = "learnable+obs"
                w_init = self.state_dict.get("contact_plan_init_head.1.weight", None)
                if torch.is_tensor(w_init) and w_init.ndim == 2:
                    contact_plan_init_hidden = int(w_init.shape[0])
        except Exception:
            contact_plan_init_mode = "learnable"
        # Infer trunk injection mode from checkpoint shared_encoder input dim.
        contact_plan_inject = "none"
        try:
            w0 = self.state_dict.get("shared_encoder.0.weight", None)
            if torch.is_tensor(w0) and w0.ndim == 2:
                nin = int(w0.shape[1])
                base_in = int(Dx + Dc)
                extra = int(max(0, nin - base_in))
                if extra > 0:
                    if int(self.contact_dim or 0) > 0 and extra == int(self.contact_dim):
                        contact_plan_inject = "contacts"
                    else:
                        contact_plan_inject = "plan_z"
                        # Ensure plan hidden matches injected dim (prefer actual injected size).
                        if extra != int(contact_plan_hidden):
                            contact_plan_hidden = int(extra)
        except Exception:
            contact_plan_inject = "none"

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

        contact_meas_enable = any(str(k).startswith("contact_meas_head.") for k in self.state_dict.keys())
        contact_meas_hidden = 64
        if contact_meas_enable:
            w0 = self.state_dict.get("contact_meas_head.mlp.0.weight", None)
            if not (torch.is_tensor(w0) and w0.ndim == 2):
                raise SystemExit(
                    "[FATAL] This repo now only supports contact_meas_head v1 (lowerbody_nohist_v1). "
                    "The provided checkpoint seems to contain a legacy contact_meas_head; please retrain."
                )
            contact_meas_hidden = int(w0.shape[0])

        direct_pose_enable = False
        direct_pose_hidden = 256
        direct_pose_meas_mode = "concat"
        direct_pose_feat_source = "cond"
        direct_pose_time_pe_dim = 0
        try:
            direct_has_weights = any(str(k).startswith("direct_pose_head.") for k in self.state_dict.keys())
            if direct_has_weights and int(Dy) > 0 and int(Dc) > 0 and int(self.contact_dim or 0) > 0:
                w_in = self.state_dict.get("direct_pose_head.0.weight", None)
                w_out = self.state_dict.get("direct_pose_head.6.weight", None)
                if torch.is_tensor(w_in) and w_in.ndim == 2 and torch.is_tensor(w_out) and w_out.ndim == 2:
                    in_dim = int(w_in.shape[1])
                    hid = int(w_in.shape[0])
                    out_dim = int(w_out.shape[0])
                    expected_out = int(Dy)
                    expected_out_modes = int(Dy) * 2
                    base_candidates = [
                        (int(Dc), "cond"),
                        (int(self.width), "hidden"),
                        (int(Dc + self.width), "cond+hidden"),
                    ]
                    Cc = int(self.contact_dim or 0)

                    if out_dim == expected_out:
                        for base_dim, src in base_candidates:
                            tdim = int(in_dim - base_dim - (2 * Cc))
                            if tdim >= 0 and tdim % 2 == 0:
                                direct_pose_enable = True
                                direct_pose_hidden = hid
                                direct_pose_meas_mode = "concat"
                                direct_pose_feat_source = src
                                direct_pose_time_pe_dim = int(tdim)
                                break
                    elif out_dim == expected_out_modes:
                        for base_dim, src in base_candidates:
                            tdim = int(in_dim - base_dim - Cc)
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
            state_layout=getattr(ds, "state_layout", None),
            bone_names=getattr(ds, "bone_names", None),
            output_layout=getattr(ds, "output_layout", None),
            contact_plan_enable=bool(contact_plan_enable or contact_plan_inject != "none" or direct_pose_enable),
            contact_plan_hidden=int(contact_plan_hidden),
            contact_plan_dropout=0.0,
            contact_plan_inject=str(contact_plan_inject),
            contact_plan_inject_detach=True,
            contact_plan_time_pe_dim=int(contact_plan_time_pe_dim),
            contact_plan_init_mode=str(contact_plan_init_mode),
            contact_plan_init_hidden=int(contact_plan_init_hidden),
            contact_plan_init_dropout=0.0,
            contact_phase_state_enable=bool(phase_state_enable),
            contact_phase_state_init_mode="obs",
            contact_phase_state_hidden=int(phase_state_hidden),
            contact_phase_state_delta_max=0.5,
            contact_phase_state_delta_init=(6.283185307179586 / 80.0),
            contact_phase_state_event_kind="touchdown",
            contact_phase_state_event_thr=0.5,
            contact_phase_state_event_hyst=0.0,
            contact_phase_state_event_min_interval=0,
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
            contact_meas_enable=bool(contact_meas_enable),
            contact_meas_hidden=int(contact_meas_hidden),
            contact_meas_dropout=0.0,
        ).to(self.device)
        validate_and_fix_model_(model, Dx, Dc)
        missing, unexpected = model.load_state_dict(self.state_dict, strict=False)
        if missing or unexpected:
            print(f"[TeacherRollout][WARN] state_dict mismatch: missing={missing}, unexpected={unexpected}")
        if bool(getattr(self.args, "direct_pose_meas_force_zero", False)):
            setattr(model, "direct_pose_meas_force_zero", True)
        # Attach frozen motion encoder bundle if提供
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
        self.bundle.apply_to_dataset(ds)
        self.bundle.apply_to_trainer(trainer)
        trainer._bundle_meta = dict(self.bundle.meta)
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


    def _init_onnx_session(self) -> None:
        if self.onnx_path is None:
            raise SystemExit("[FATAL] --onnx-model not provided.")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit("[FATAL] onnxruntime is required for --onnx-model") from exc

        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        inputs = session.get_inputs()
        if not inputs:
            raise SystemExit("[FATAL] ONNX model has no inputs.")
        # Expected signature from `export_onnx_step_stateful_nophase`:
        #   state, cond, contacts, angvel, pose_hist, (optional) plan_z, (optional) phase_z,
        #   (optional) td_hazard_acc
        ort_input_map: dict[str, str] = {}
        for inp in inputs:
            name_l = inp.name.lower()
            if "state" in name_l and "state" not in ort_input_map:
                ort_input_map["state"] = inp.name
            elif "cond" in name_l and "cond" not in ort_input_map:
                ort_input_map["cond"] = inp.name
            elif "contact" in name_l and "contacts" not in ort_input_map:
                ort_input_map["contacts"] = inp.name
            elif ("angvel" in name_l or "ang" in name_l) and "angvel" not in ort_input_map:
                ort_input_map["angvel"] = inp.name
            elif "pose" in name_l and "pose_hist" not in ort_input_map:
                ort_input_map["pose_hist"] = inp.name
            elif "plan" in name_l and "plan_z" not in ort_input_map:
                ort_input_map["plan_z"] = inp.name
            elif "phase" in name_l and "phase_z" not in ort_input_map:
                ort_input_map["phase_z"] = inp.name
            elif ("hazard" in name_l or "td_hazard" in name_l) and "td_hazard_acc" not in ort_input_map:
                # Exported only when phase_reset_source=td_hazard; keep optional for backward compat.
                ort_input_map["td_hazard_acc"] = inp.name

        required = ("state", "cond", "contacts", "angvel", "pose_hist")
        missing = [k for k in required if k not in ort_input_map]
        if missing:
            available = ", ".join([i.name for i in inputs])
            raise SystemExit(f"[FATAL] ONNX missing inputs {missing}. Available: {available}")
        self.ort_input_map = ort_input_map

        # Best-effort plan_z dim inference (needed to keep plan state across steps).
        self.ort_plan_dim = None
        if "plan_z" in ort_input_map:
            inp_obj = next((i for i in inputs if i.name == ort_input_map["plan_z"]), None)
            if inp_obj is not None and getattr(inp_obj, "shape", None):
                try:
                    last = inp_obj.shape[-1]
                    if isinstance(last, int) and last > 0:
                        self.ort_plan_dim = int(last)
                except Exception:
                    self.ort_plan_dim = None
        # Best-effort phase_z dim inference (needed to keep phase state across steps).
        self.ort_phase_dim = None
        if "phase_z" in ort_input_map:
            inp_obj = next((i for i in inputs if i.name == ort_input_map["phase_z"]), None)
            if inp_obj is not None and getattr(inp_obj, "shape", None):
                try:
                    last = inp_obj.shape[-1]
                    if isinstance(last, int) and last > 0:
                        self.ort_phase_dim = int(last)
                except Exception:
                    self.ort_phase_dim = None
        # Best-effort td_hazard_acc dim inference (exported only when phase_reset_source=td_hazard).
        self.ort_td_hazard_acc_dim = None
        if "td_hazard_acc" in ort_input_map:
            inp_obj = next((i for i in inputs if i.name == ort_input_map["td_hazard_acc"]), None)
            if inp_obj is not None and getattr(inp_obj, "shape", None):
                try:
                    last = inp_obj.shape[-1]
                    if isinstance(last, int) and last > 0:
                        self.ort_td_hazard_acc_dim = int(last)
                except Exception:
                    self.ort_td_hazard_acc_dim = None
        outputs = session.get_outputs()
        if not outputs:
            raise SystemExit("[FATAL] ONNX model has no outputs.")
        out_motion = None
        out_plan = None
        out_phase = None
        out_td_hazard_acc = None
        for out in outputs:
            name_l = out.name.lower()
            if out_motion is None and ("motion" in name_l or "pred" in name_l):
                out_motion = out.name
            if out_plan is None and ("plan_z" in name_l or (("plan" in name_l) and ("phase" not in name_l))):
                out_plan = out.name
            if out_phase is None and ("phase_z" in name_l or ("phase" in name_l and "z_next" in name_l)):
                out_phase = out.name
            if out_td_hazard_acc is None and ("td_hazard_acc" in name_l or "hazard_acc" in name_l):
                out_td_hazard_acc = out.name
        self.ort_output_name = out_motion or outputs[0].name
        self.ort_plan_output_name = out_plan
        self.ort_phase_output_name = out_phase
        self.ort_td_hazard_acc_output_name = out_td_hazard_acc
        self.ort_session = session

    def run_clip(self, teacher_path: Path, out_dir: Path, npz_root: Path, quiet: bool = False) -> Optional[Path]:
        data = load_json(teacher_path)
        clip_name = str(data.get("clip") or teacher_path.stem.replace("_teacher", ""))
        teacher_block = data.get("teacher")
        if not isinstance(teacher_block, dict):
            raise ValueError(f"{teacher_path}: missing 'teacher' payload.")
        state_arr = np.asarray(teacher_block.get("state_norm"), dtype=np.float32)
        cond_arr = np.asarray(teacher_block.get("cond"), dtype=np.float32)
        if state_arr.ndim != 2 or cond_arr.ndim != 2:
            raise ValueError(f"{teacher_path}: invalid state/cond shapes.")
        npz_path = resolve_npz_path(clip_name, data.get("source_json"), npz_root)
        ds, clip = self._build_dataset(npz_path)
        self._ensure_model_ready(ds)

        contacts = clip.contacts if clip.contacts is not None else np.zeros((state_arr.shape[0], self.contact_dim or 0), dtype=np.float32)
        angvel = clip.angvel_norm if clip.angvel_norm is not None else np.zeros((state_arr.shape[0], self.angvel_dim or 0), dtype=np.float32)
        pose_hist = clip.pose_hist_norm if clip.pose_hist_norm is not None else np.zeros((state_arr.shape[0], self.pose_hist_dim or 0), dtype=np.float32)
        gt_norm = clip.Y
        usable_len = _min_length(state_arr, cond_arr, contacts, angvel, pose_hist, gt_norm)
        if usable_len < state_arr.shape[0]:
            print(f"[WARN] {clip_name}: trimming teacher sequence from {state_arr.shape[0]} to {usable_len} frames.")
        state_arr = state_arr[:usable_len]
        cond_arr = cond_arr[:usable_len]
        contacts = contacts[:usable_len]
        angvel = angvel[:usable_len]
        pose_hist = pose_hist[:usable_len]
        gt_norm = gt_norm[:usable_len]

        # ---- Optional input ablations (for debugging history/lag) ----
        pose_shift = int(getattr(self.args, "pose_hist_time_shift", 0) or 0)
        if pose_shift != 0 and pose_hist.shape[1] > 0:
            pose_hist = _shift_time_axis(pose_hist, shift=pose_shift)
        pose_mode = str(getattr(self.args, "pose_hist_ablation", "none") or "none").lower().strip()
        if pose_mode not in ("", "none") and pose_hist.shape[1] > 0:
            pose_hist = _ablate_pose_hist(
                pose_hist,
                pose_hist_len=int(self.pose_hist_len),
                mode=pose_mode,
                keep_last=int(getattr(self.args, "pose_hist_keep_last", 1) or 1),
            )
        ang_mode = str(getattr(self.args, "angvel_ablation", "none") or "none").lower().strip()
        if ang_mode == "zero" and angvel.shape[1] > 0:
            angvel = np.zeros_like(angvel)
        teacher_block["state_norm"] = state_arr.tolist()
        teacher_block["cond"] = cond_arr.tolist()
        if isinstance(teacher_block.get("target_norm"), list):
            teacher_block["target_norm"] = teacher_block["target_norm"][:usable_len]

        if self.use_onnx:
            pred_norm = self._run_onnx_rollout(state_arr, cond_arr, contacts, angvel, pose_hist, gt_norm)
            contacts_meas_pred = None
            contacts_meas_logits_pred = None
        else:
            state_t = torch.from_numpy(state_arr).unsqueeze(0).to(self.device)
            cond_t = torch.from_numpy(cond_arr).unsqueeze(0).to(self.device)
            contacts_t = (
                torch.from_numpy(contacts).unsqueeze(0).to(self.device) if contacts.shape[1] > 0 else None
            )
            angvel_t = (
                torch.from_numpy(angvel).unsqueeze(0).to(self.device) if angvel.shape[1] > 0 else None
            )
            pose_hist_t = (
                torch.from_numpy(pose_hist).unsqueeze(0).to(self.device) if pose_hist.shape[1] > 0 else None
            )
            gt_t = torch.from_numpy(gt_norm).unsqueeze(0).to(self.device)

            # Allow overriding how teacher rollout sources angvel / pose_hist inside Trainer._rollout_sequence.
            # This is purely for debugging; default behavior stays consistent with deployment.
            if str(getattr(self.args, "angvel_source", "state") or "state").lower().strip() == "seq":
                try:
                    setattr(self.trainer, "use_freerun_state_sync", False)
                except Exception:
                    pass
            if str(getattr(self.args, "pose_hist_source", "buffer") or "buffer").lower().strip() == "seq":
                try:
                    setattr(self.trainer, "force_pose_hist_seq", True)
                except Exception:
                    pass

            # Forward-time ablations applied inside Trainer._rollout_sequence (works for both state/buffer and seq sources).
            try:
                setattr(self.trainer, "rollout_angvel_ablation", str(getattr(self.args, "angvel_ablation", "none")))
                setattr(self.trainer, "rollout_pose_hist_ablation", str(getattr(self.args, "pose_hist_ablation", "none")))
                setattr(self.trainer, "rollout_pose_hist_keep_last", int(getattr(self.args, "pose_hist_keep_last", 1) or 1))
            except Exception:
                pass

            self.model.eval()
            with torch.no_grad():
                preds, _ = self.trainer._rollout_sequence(
                    state_t,
                    cond_t,
                    contacts_seq=contacts_t,
                    angvel_seq=angvel_t,
                    pose_hist_seq=pose_hist_t,
                    gt_seq=gt_t,
                    mode="mixed",
                    tf_ratio=1.0,
                )
            pred_norm = preds["out"][0].cpu().numpy()
            contacts_meas_pred = None
            contacts_meas_logits_pred = None
            try:
                cm = preds.get("contacts_meas", None)
                if torch.is_tensor(cm):
                    if cm.dim() == 3 and cm.size(0) > 0:
                        contacts_meas_pred = cm[0].detach().cpu().numpy()
                    elif cm.dim() == 2:
                        contacts_meas_pred = cm.detach().cpu().numpy()
            except Exception:
                contacts_meas_pred = None
            try:
                cml = preds.get("contacts_meas_logits", None)
                if torch.is_tensor(cml):
                    if cml.dim() == 3 and cml.size(0) > 0:
                        contacts_meas_logits_pred = cml[0].detach().cpu().numpy()
                    elif cml.dim() == 2:
                        contacts_meas_logits_pred = cml.detach().cpu().numpy()
            except Exception:
                contacts_meas_logits_pred = None

        mse_norm = float(np.mean((pred_norm - gt_norm) ** 2))
        pred_raw = gt_raw = None
        geo_deg = None
        if self.args.with_denorm:
            if self.use_onnx:
                pred_raw_t = self.normalizer.denorm(torch.from_numpy(pred_norm).unsqueeze(0))
                gt_raw_t = self.normalizer.denorm(torch.from_numpy(gt_norm).unsqueeze(0))
                pred_raw = pred_raw_t.cpu().numpy()[0]
                gt_raw = gt_raw_t.cpu().numpy()[0]
            else:
                with torch.no_grad():
                    pred_raw_tensor = self.trainer._denorm(torch.from_numpy(pred_norm).unsqueeze(0).to(self.device))
                    gt_raw_tensor = self.trainer._denorm(torch.from_numpy(gt_norm).unsqueeze(0).to(self.device))
                pred_raw = pred_raw_tensor.cpu().numpy()[0]
                gt_raw = gt_raw_tensor.cpu().numpy()[0]
            geo_deg = self._compute_geo_deg(pred_raw, gt_raw)

        payload = {
            "clip": clip_name,
            "model": str(self.onnx_path if self.use_onnx else Path(self.args.model).resolve()),
            "teacher_json": str(teacher_path.resolve()),
            "source_json": data.get("source_json"),
            "fps": data.get("fps", getattr(ds, "fps", 60.0)),
            "num_pairs": int(usable_len),
            "dims": {
                "Dx": int(state_arr.shape[1]),
                "Dy": int(pred_norm.shape[1]),
                "Dc": int(cond_arr.shape[1]),
                "contacts": int(contacts.shape[1]),
                "angvel": int(angvel.shape[1]),
                "pose_hist": int(pose_hist.shape[1]),
            },
            "layouts": data.get("layouts", {}),
            "teacher": teacher_block,
            "aux_inputs": {
                "contacts": contacts.tolist() if contacts.shape[1] > 0 else [],
                "angvel_norm": angvel.tolist() if angvel.shape[1] > 0 else [],
                "pose_hist_norm": pose_hist.tolist() if pose_hist.shape[1] > 0 else [],
            },
            "prediction": {
                "y_norm": pred_norm.tolist(),
                "y_raw": pred_raw.tolist() if pred_raw is not None else None,
            },
            "contacts_pred": {
                "contacts_meas": contacts_meas_pred.tolist() if contacts_meas_pred is not None else None,
                "contacts_meas_logits": contacts_meas_logits_pred.tolist() if contacts_meas_logits_pred is not None else None,
            },
            "diagnostics": {
                "MSEnormY": mse_norm,
                "GeoDeg": geo_deg,
            },
            "ablation": {
                "direct_pose_meas_force_zero": bool(getattr(self.args, "direct_pose_meas_force_zero", False)),
                "angvel_source": str(getattr(self.args, "angvel_source", "state")),
                "pose_hist_source": str(getattr(self.args, "pose_hist_source", "buffer")),
                "pose_hist_ablation": str(getattr(self.args, "pose_hist_ablation", "none")),
                "pose_hist_keep_last": int(getattr(self.args, "pose_hist_keep_last", 1) or 1),
                "pose_hist_time_shift": int(getattr(self.args, "pose_hist_time_shift", 0) or 0),
                "angvel_ablation": str(getattr(self.args, "angvel_ablation", "none")),
            },
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{clip_name}_teacher_pred.json"
        if out_path.exists() and not self.args.force:
            raise FileExistsError(f"{out_path} exists (use --force to overwrite)")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        if not quiet:
            print(f"[OK] {clip_name}: wrote {out_path} (frames={usable_len}, mse={mse_norm:.6f})")
        return out_path

    def _compute_geo_deg(self, pred_raw: np.ndarray, gt_raw: np.ndarray) -> Optional[float]:
        if pred_raw is None or gt_raw is None:
            return None
        rot_slice = None
        if self.trainer is not None:
            rot_slice = getattr(self.trainer, "rot6d_y_slice", None) or getattr(self.trainer, "rot6d_slice", None)
        if rot_slice is None:
            span = self.bundle.output_layout.get("BoneRotations6D")
            if span:
                rot_slice = slice(int(span[0]), int(span[0] + span[1]))
        if not isinstance(rot_slice, slice):
            return None
        try:
            width = rot_slice.stop - rot_slice.start
            if width % 6 != 0:
                return None
            joints = width // 6
            pred = (
                torch.from_numpy(pred_raw[:, rot_slice])
                .view(pred_raw.shape[0], joints, 6)
                .unsqueeze(0)
                .to(self.device)
            )
            gt = (
                torch.from_numpy(gt_raw[:, rot_slice])
                .view(gt_raw.shape[0], joints, 6)
                .unsqueeze(0)
                .to(self.device)
            )
            pred_m = rot6d_to_matrix(reproject_rot6d(pred)).squeeze(0)
            gt_m = rot6d_to_matrix(reproject_rot6d(gt)).squeeze(0)
            deg = geodesic_R(pred_m, gt_m) * (180.0 / math.pi)
            return float(deg.mean().item())
        except Exception:
            return None

    def _run_onnx_rollout(
        self,
        state_arr: np.ndarray,
        cond_arr: np.ndarray,
        contacts: np.ndarray,
        angvel: np.ndarray,
        pose_hist: np.ndarray,
        gt_norm: Optional[np.ndarray],
    ) -> np.ndarray:
        if self.ort_session is None:
            raise SystemExit("[FATAL] ONNX session not initialized.")
        if self.normalizer is None:
            raise SystemExit("[FATAL] DataNormalizer missing for ONNX rollout.")
        T = state_arr.shape[0]
        outputs: List[np.ndarray] = []
        import torch

        def _span_to_slice(span_obj):
            if span_obj is None:
                return None
            if isinstance(span_obj, (list, tuple)) and len(span_obj) == 2:
                start = int(span_obj[0])
                length = int(span_obj[1])
                return slice(start, start + length)
            if isinstance(span_obj, dict):
                start = int(span_obj.get("start", 0))
                size = int(span_obj.get("size", 0))
                return slice(start, start + size) if size > 0 else None
            return None

        rot_x_slice = _span_to_slice(self.bundle.state_layout.get("BoneRotations6D") if self.bundle else None)
        # 输出端的 rot6d 仅用于诊断；ONNX 输出本身工作在 Y 空间，因此这里只需要 X 端切片
        if not isinstance(rot_x_slice, slice):
            raise SystemExit("[FATAL] BoneRotations6D slice missing in bundle layouts; cannot denorm ONNX outputs.")

        # Teacher-forcing: 以 GT Y_raw 作为上一帧基准，在 Y 空间合成 delta
        if gt_norm is None or gt_norm.shape[0] == 0:
            raise SystemExit("[FATAL] gt_norm is required for ONNX teacher rollout.")
        gt0 = torch.from_numpy(gt_norm[:1]).to(torch.float32)
        y_prev_raw = self.normalizer.denorm(gt0)  # [1, Dy_raw]

        std_y = None
        if getattr(self.normalizer, "std_y", None) is not None:
            std_y = torch.as_tensor(self.normalizer.std_y, dtype=torch.float32).view(1, -1)

        plan_z = None
        if "plan_z" in self.ort_input_map:
            plan_dim = getattr(self, "ort_plan_dim", None)
            if not isinstance(plan_dim, int) or plan_dim <= 0:
                raise SystemExit("[FATAL] Unable to infer plan_z dim from ONNX input shapes.")
            plan_z = np.zeros((1, plan_dim), dtype=np.float32)
        phase_z = None
        if "phase_z" in self.ort_input_map:
            phase_dim = getattr(self, "ort_phase_dim", None)
            if not isinstance(phase_dim, int) or phase_dim <= 0:
                raise SystemExit("[FATAL] Unable to infer phase_z dim from ONNX input shapes.")
            phase_z = np.zeros((1, phase_dim), dtype=np.float32)
        td_hazard_acc = None
        if "td_hazard_acc" in self.ort_input_map:
            hz_dim = getattr(self, "ort_td_hazard_acc_dim", None)
            if not isinstance(hz_dim, int) or hz_dim <= 0:
                raise SystemExit("[FATAL] Unable to infer td_hazard_acc dim from ONNX input shapes.")
            td_hazard_acc = np.zeros((1, hz_dim), dtype=np.float32)

        for t in range(T):
            feeds: dict[str, np.ndarray] = {
                self.ort_input_map["state"]: state_arr[t : t + 1],
                self.ort_input_map["cond"]: cond_arr[t : t + 1],
                self.ort_input_map["contacts"]: contacts[t : t + 1],
                self.ort_input_map["angvel"]: angvel[t : t + 1],
                self.ort_input_map["pose_hist"]: pose_hist[t : t + 1],
            }
            out_names = [self.ort_output_name]
            want_plan_out = False
            want_phase_out = False
            want_hz_out = False
            if plan_z is not None:
                feeds[self.ort_input_map["plan_z"]] = plan_z
                if getattr(self, "ort_plan_output_name", None):
                    want_plan_out = True
                    out_names.append(self.ort_plan_output_name)
            if phase_z is not None:
                feeds[self.ort_input_map["phase_z"]] = phase_z
                if getattr(self, "ort_phase_output_name", None):
                    want_phase_out = True
                    out_names.append(self.ort_phase_output_name)
            if td_hazard_acc is not None:
                feeds[self.ort_input_map["td_hazard_acc"]] = td_hazard_acc
                if getattr(self, "ort_td_hazard_acc_output_name", None):
                    want_hz_out = True
                    out_names.append(self.ort_td_hazard_acc_output_name)
            outs = self.ort_session.run(out_names, feeds)
            y = outs[0]
            out_k = 1
            if want_plan_out and len(outs) > out_k:
                try:
                    plan_z = np.asarray(outs[out_k], dtype=np.float32)
                except Exception:
                    pass
                out_k += 1
            if want_phase_out and len(outs) > out_k:
                try:
                    phase_z = np.asarray(outs[out_k], dtype=np.float32)
                except Exception:
                    pass
                out_k += 1
            if want_hz_out and len(outs) > out_k:
                try:
                    td_hazard_acc = np.asarray(outs[out_k], dtype=np.float32)
                except Exception:
                    pass
            delta_norm = torch.as_tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32)  # [1, Dy]

            # ΔY_norm -> ΔY_raw
            if std_y is not None:
                delta_raw = delta_norm * std_y.clamp_min(1e-6)
            else:
                delta_raw = delta_norm

            # 在 Y 空间合成：前 rot6d 部分用 compose_rot6d_delta，尾部（如 RootVel）做残差相加
            D = int(delta_raw.shape[-1])
            rot_len = (D // 6) * 6
            if rot_len <= 0:
                raise SystemExit(f"[FATAL] invalid Y-dim for rot6d composition: {D}")

            y_prev = y_prev_raw
            prev_rot = y_prev[..., :rot_len]
            delta_rot = delta_raw[..., :rot_len]
            # 正规化 delta_rot 并转换为矩阵
            J = rot_len // 6
            prev = reproject_rot6d(prev_rot).view(1, J, 6)
            delta = normalize_rot6d_delta(delta_rot, columns=("X", "Z"))
            R_prev = rot6d_to_matrix(prev, columns=("X", "Z"))
            R_delta = rot6d_to_matrix(delta, columns=("X", "Z"))
            R_next = torch.matmul(R_delta, R_prev)
            rot_next = matrix_to_rot6d(R_next, columns=("X", "Z")).view(1, rot_len)

            if rot_len == D:
                y_raw = rot_next
            else:
                tail_prev = y_prev[..., rot_len:]
                tail_delta = delta_raw[..., rot_len:]
                tail_next = tail_prev + tail_delta
                y_raw = torch.cat([rot_next, tail_next], dim=-1)

            y_norm = self.normalizer.norm_y(y_raw)
            outputs.append(y_norm.squeeze(0).cpu().numpy())
            # Teacher 模式：每步都以 GT 下一帧作为基准，避免累积误差干扰对齐检查
            if (t + 1) < gt_norm.shape[0]:
                gt_next = torch.from_numpy(gt_norm[t + 1 : t + 2]).to(torch.float32)
                y_prev_raw = self.normalizer.denorm(gt_next)
            else:
                y_prev_raw = y_raw.detach()

        return np.stack(outputs, axis=0)


def main() -> None:
    args = parse_args()
    teacher_files = expand_specs(args.teacher)
    if not teacher_files:
        raise SystemExit("[FATAL] No teacher JSON files matched the provided specs.")
    runner = TeacherRolloutRunner(args)
    out_dir = Path(args.out).expanduser().resolve()
    npz_root = Path(args.npz_root).expanduser().resolve()
    success = 0
    failures: List[str] = []
    for teacher_path in teacher_files:
        try:
            runner.run_clip(teacher_path, out_dir, npz_root, quiet=args.quiet)
            success += 1
        except Exception as exc:
            failures.append(f"{teacher_path}: {exc}")
            print(f"[ERR] {teacher_path}: {exc}")
    print(f"[Done] rollouts={success} ok / {len(failures)} failed")
    if failures:
        print("Failed clips:")
        for msg in failures:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()
