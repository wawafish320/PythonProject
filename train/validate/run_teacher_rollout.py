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

from train.training_MPL import (
    EventMotionModel,
    MotionEventDataset,
    LayoutCenter,
    DataNormalizer,
    MotionJointLoss,
    Trainer,
    validate_and_fix_model_,
    rot6d_to_matrix,
    reproject_rot6d,
    geodesic_R,
)


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
        help="Frozen motion encoder bundle (.pt) used during training (needed when checkpoint expects it).",
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
        help="Nominal model depth (kept for compatibility; EventMotionModel uses two linear blocks).",
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
            self.state_dict = self.ckpt["model"] if isinstance(self.ckpt, dict) and "model" in self.ckpt else self.ckpt
        self.width = self._infer_width() if not self.use_onnx else None
        self.period_dim = self._infer_period_dim() if not self.use_onnx else 0
        self.encoder_bundle_path = Path(args.encoder_bundle).expanduser()
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
                    yaw_x_slice=self.bundle.state_layout.get("RootYaw"),
                    yaw_y_slice=self.bundle.output_layout.get("RootYaw"),
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
        self._attach_encoder_if_available(model)
        validate_and_fix_model_(model, Dx, Dc)
        missing, unexpected = model.load_state_dict(self.state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"State dict mismatch (missing={missing}, unexpected={unexpected})")
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

    def _attach_encoder_if_available(self, model: EventMotionModel) -> None:
        bundle_path = getattr(self, "encoder_bundle_path", None)
        if not bundle_path or not bundle_path.is_file():
            return
        try:
            payload = torch.load(str(bundle_path), map_location="cpu")
            model.attach_motion_encoder(payload)
            if not self.args.quiet:
                print(f"[Spec] Attached motion encoder bundle: {bundle_path}")
        except Exception as exc:
            raise RuntimeError(f"Failed to attach motion encoder bundle {bundle_path}: {exc}") from exc

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
        if len(inputs) != 5:
            raise SystemExit(f"[FATAL] Expected 5 ONNX inputs (state/cond/contacts/angvel/pose_hist); got {len(inputs)}")
        canonical = ["state", "cond", "contact", "ang", "pose"]
        mapping: dict[str, str] = {}
        for inp in inputs:
            name_l = inp.name.lower()
            for key in canonical:
                if key not in mapping and key in name_l:
                    mapping[key] = inp.name
                    break
        if len(mapping) < 5:
            ordered = [inp.name for inp in inputs]
            mapping = dict(zip(["state", "cond", "contact", "ang", "pose"], ordered))
        self.ort_input_map = {
            "state": mapping["state"],
            "cond": mapping["cond"],
            "contacts": mapping["contact"],
            "angvel": mapping["ang"],
            "pose_hist": mapping["pose"],
        }
        outputs = session.get_outputs()
        if not outputs:
            raise SystemExit("[FATAL] ONNX model has no outputs.")
        self.ort_output_name = outputs[0].name
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
        teacher_block["state_norm"] = state_arr.tolist()
        teacher_block["cond"] = cond_arr.tolist()
        if isinstance(teacher_block.get("target_norm"), list):
            teacher_block["target_norm"] = teacher_block["target_norm"][:usable_len]

        if self.use_onnx:
            pred_norm = self._run_onnx_rollout(state_arr, cond_arr, contacts, angvel, pose_hist)
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

            self.model.eval()
            with torch.no_grad():
                preds, _ = self.trainer._rollout_sequence(
                    state_t,
                    cond_t,
                    contacts_seq=contacts_t,
                    angvel_seq=angvel_t,
                    pose_hist_seq=pose_hist_t,
                    gt_seq=gt_t,
                    mode="teacher",
                    tf_ratio=1.0,
                )
            pred_norm = preds["out"][0].cpu().numpy()

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
            "diagnostics": {
                "MSEnormY": mse_norm,
                "GeoDeg": geo_deg,
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
    ) -> np.ndarray:
        if self.ort_session is None:
            raise SystemExit("[FATAL] ONNX session not initialized.")
        T = state_arr.shape[0]
        outputs: List[np.ndarray] = []
        for t in range(T):
            feeds = {
                self.ort_input_map["state"]: state_arr[t : t + 1],
                self.ort_input_map["cond"]: cond_arr[t : t + 1],
                self.ort_input_map["contacts"]: contacts[t : t + 1],
                self.ort_input_map["angvel"]: angvel[t : t + 1],
                self.ort_input_map["pose_hist"]: pose_hist[t : t + 1],
            }
            y = self.ort_session.run([self.ort_output_name], feeds)[0]
            outputs.append(np.asarray(y, dtype=np.float32)[0])
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
