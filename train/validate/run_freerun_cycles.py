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
        ).to(self.device)
        # Validate basic shapes then load weights (allow extra frozen encoder keys).
        validate_and_fix_model_(model, Dx, Dc)
        missing, unexpected = model.load_state_dict(self.state_dict, strict=False)
        if missing or unexpected:
            print(f"[FreeRun][WARN] state_dict mismatch: missing={missing}, unexpected={unexpected}")
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
        # Inject bundle‑derived slices & normalizer
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
        metrics_per_round, per_step = _run_freerun_cycles(
            trainer=self.trainer,
            sample=base_sample,
            rounds=rounds,
            device=self.device,
        )

        payload = {
            "clip": clip_name,
            "source_json": data.get("source_json"),
            "teacher_json": str(teacher_path.resolve()),
            "fps": data.get("fps", getattr(ds, "fps", 60.0)),
            "cycle_len": int(T_base),
            "rounds": rounds,
            "model": str(Path(self.args.model).expanduser().resolve()),
            "metrics_per_round": metrics_per_round,
            "metrics_per_step": per_step,
        }

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
) -> List[Dict[str, Any]]:
    """
    Core free‑run loop: autoregress over `rounds * T` steps without reset,
    then compute per‑round diagnostics (MSEnormY, GeoDeg).
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

    warmup = 0
    start_t = warmup
    end_t = T - 1  # last usable index for t+1

    model = trainer.model
    predsY: List[torch.Tensor] = []
    predsX: List[torch.Tensor] = []

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
    for t in range(start_t, end_t):
        cond_input = cond_seq[:, t] if (cond_seq is not None and cond_seq.dim() == 3) else cond_seq
        contacts_t = contacts_seq[:, t] if (contacts_seq is not None and contacts_seq.dim() == 3) else contacts_seq
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

        with amp_ctx:
            ret = model(
                motion,
                cond_input,
                contacts=contacts_t,
                angvel=angvel_t,
                pose_history=pose_hist_t,
            )

        if not isinstance(ret, dict):
            raise RuntimeError("Model forward must return a dict with at least 'out'.")
        out = ret.get("out")
        if out is None:
            break

        delta_norm = out
        if y_raw_prev is not None:
            try:
                y_raw = trainer._compose_delta_to_raw(y_raw_prev, delta_norm)
            except Exception:
                y_raw = trainer._denorm(delta_norm)
        else:
            y_raw = trainer._denorm(delta_norm)

        y_raw_prev = y_raw.detach()

        try:
            y_norm = trainer._norm_y(y_raw)
        except Exception:
            y_norm = delta_norm

        predsY.append(y_norm)

        if motion_raw is not None:
            motion_raw = trainer._apply_free_carry(motion_raw, y_raw, cond_next_raw=cond_raw_step).detach()
            motion = trainer._diag_norm_x(motion_raw)
        else:
            motion = trainer._apply_free_carry(motion, y_raw, cond_next_raw=None).detach()

        predsX.append(motion)

        if pose_hist_enabled and pose_hist_stride > 0:
            with torch.no_grad():
                pose_hist_buffer_raw = torch.roll(pose_hist_buffer_raw, shifts=-pose_hist_stride, dims=-1)
                rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
                if isinstance(rot_slice, slice):
                    pose_hist_buffer_raw[..., -pose_hist_stride:] = y_raw[..., rot_slice]
                pose_hist_buffer_norm = trainer._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)

    if not predsY:
        raise RuntimeError("No predictions produced during free‑run.")

    # Align predictions and GT strictly by one-step look-ahead:
    #   at iteration t we predicted frame (t+1) based on history up to t.
    # So we ignore the very last GT frame and compare:
    #   predY[:, i]  vs  gt_seq[:, start_t+1+i]
    predY_full = torch.stack(predsY, dim=1)  # [B, free_steps_raw, Dy]
    free_steps_raw = predY_full.shape[1]
    max_aligned = max(0, min(free_steps_raw, T_total - (start_t + 1)))
    if max_aligned <= 0:
        raise RuntimeError("Not enough frames for aligned free-run evaluation.")
    predY = predY_full[:, :max_aligned]
    free_steps = max_aligned
    gt_start = start_t + 1
    gt_end = gt_start + free_steps
    gtY = gt_seq[:, gt_start:gt_end]

    # ---- Per‑round metrics ---------------------------------------------------
    # Shared slices for rotations
    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        rot_slice = slice(0, predY.shape[-1])
    width = rot_slice.stop - rot_slice.start
    deg_factor = 180.0 / float(np.pi)
    metrics_per_round: List[Dict[str, Any]] = []

    # Denorm entire run once for GeoDeg
    with torch.no_grad():
        pred_raw_full = trainer._denorm(predY.reshape(1, free_steps, -1))
        gt_raw_full = trainer._denorm(gtY.reshape(1, free_steps, -1))

    per_step: List[Dict[str, Any]] = []

    # Optional: per-bone geodesic error for key bones (same set as training diag)
    loss_fn = getattr(trainer, "loss_fn", None)
    bone_names = getattr(loss_fn, "bone_names", []) if loss_fn is not None else []
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
        pr6_full = pred_raw_full[..., rot_slice].view(1, free_steps, J, 6)
        gt6_full = gt_raw_full[..., rot_slice].view(1, free_steps, J, 6)
        pr6_full = reproject_rot6d(pr6_full)
        gt6_full = reproject_rot6d(gt6_full)
        Rp_full = rot6d_to_matrix(pr6_full)  # [1, free_steps, J, 3, 3]
        Rg_full = rot6d_to_matrix(gt6_full)
        geo_full = geodesic_R(Rp_full, Rg_full) * deg_factor  # [1, free_steps, J]
    else:
        geo_full = None

    for t in range(free_steps):
        geo_t = None
        keybone_geo: Dict[str, float] = {}
        if geo_full is not None:
            # Mean over all joints
            geo_t = float(geo_full[:, t].mean().item())
            # Per-key-bone geodesic errors
            if key_indices:
                per_joint = geo_full[0, t]  # [J]
                for name, j_idx in zip(key_bone_names, key_indices):
                    if 0 <= j_idx < per_joint.numel():
                        keybone_geo[name] = float(per_joint[j_idx].item())
        entry: Dict[str, Any] = {
            "step": int(t),
            "GeoDeg": geo_t,
        }
        if keybone_geo:
            entry["KeyBoneGeoDeg"] = keybone_geo
        per_step.append(entry)

    for r in range(rounds):
        t0 = r * T_cycle
        t1 = min((r + 1) * T_cycle, free_steps)
        if t1 <= t0:
            continue

        pred_r = predY[:, t0:t1]  # [1, Tr, Dy]
        gt_r = gtY[:, t0:t1]

        # GeoDeg for this round (all joints)
        pr_raw = pred_raw_full[:, t0:t1, :]
        gt_raw = gt_raw_full[:, t0:t1, :]
        width = rot_slice.stop - rot_slice.start
        geo_deg_val: Optional[float] = None
        keybone_geo_mean: Optional[float] = None
        if width > 0 and width % 6 == 0:
            J = width // 6
            pr6 = pr_raw[..., rot_slice]
            gt6 = gt_raw[..., rot_slice]
            pr6 = reproject_rot6d(pr6.view(-1, J, 6))
            gt6 = reproject_rot6d(gt6.view(-1, J, 6))
            Rp = rot6d_to_matrix(pr6).view(1, -1, J, 3, 3)
            Rg = rot6d_to_matrix(gt6).view(1, -1, J, 3, 3)
            geo = geodesic_R(Rp, Rg) * deg_factor  # [1, Tr, J]
            geo_deg_val = float(geo.mean().item())
            if key_indices:
                kb = geo[..., key_indices]  # [1, Tr, K]
                keybone_geo_mean = float(kb.mean().item())

        round_entry: Dict[str, Any] = {
            "round": int(r),
            "start_step": int(t0),
            "end_step": int(t1 - 1),
            "steps": int(t1 - t0),
            "GeoDeg": geo_deg_val,
        }
        if keybone_geo_mean is not None:
            round_entry["KeyBoneGeoDegMean"] = keybone_geo_mean
        metrics_per_round.append(round_entry)

    return metrics_per_round, per_step


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
        help="Nominal model depth (kept for compatibility; EventMotionModel uses two linear blocks).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of full animation cycles to free‑run without reset.",
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
