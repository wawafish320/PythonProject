#!/usr/bin/env python3
"""Real-data rollout smoke for ``Trainer._rollout_sequence``."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train.validate.run_teacher_rollout import (
    TeacherRolloutRunner,
    _ablate_pose_hist,
    _min_length,
    _shift_time_axis,
    load_json,
    resolve_npz_path,
)


DEFAULT_TEACHER = "validate/teacher_batches/Walk_F_teacher.json"
DEFAULT_BUNDLE = "raw_data/processed_data/norm_template.json"
DEFAULT_PRETRAIN_TEMPLATE = "models/pretrain_template.json"
DEFAULT_ENCODER_BUNDLE = "models/motion_encoder_equiv.pt"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"
DEFAULT_OUT_DIR = "validate/rollout_smoke"
DEFAULT_MODEL_CANDIDATES = (
    "models/MLPL2_DirectBranch_v1__nophase_to_stage7_20260303/ckpt_last_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain_fromnophase_20260303.pth",
    "models/MLPL2_DirectBranch_v1__nophase_to_stage7_20260303/ckpt_last_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat_fromnophase_20260303.pth",
    "models/MLPL2_DirectBranch_v1__stage6_n1leg_v2_20260217_step3x_smoke_full_directpose/ckpt_last_nline_full_n1leg_directpose_smoke_r2_budget_seed0_e1.pth",
)

REQUIRED_KEYS = ("out", "delta")
OPTIONAL_KEYS = (
    "hidden_seq",
    "period_pred",
    "contacts_plan",
    "contacts_plan_logits",
    "out_direct",
    "contacts_meas",
    "contacts_err",
    "event_clock_lambda_logit",
    "event_clock_dynamic_prior",
    "event_clock_delta_z",
)


@dataclass(frozen=True)
class SmokeCase:
    name: str
    mode: str
    tf_ratio: float


def _resolve_default_model(user_model: Optional[str]) -> Path:
    if user_model:
        path = Path(user_model).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        return path
    for candidate in DEFAULT_MODEL_CANDIDATES:
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No default rollout smoke checkpoint found. Pass --model explicitly."
    )


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run a real-data rollout smoke against Trainer._rollout_sequence.")
    ap.add_argument("--teacher", default=DEFAULT_TEACHER)
    ap.add_argument("--model", default=None)
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--pretrain-template", default=DEFAULT_PRETRAIN_TEMPLATE)
    ap.add_argument("--encoder-bundle", default=DEFAULT_ENCODER_BUNDLE)
    ap.add_argument("--npz-root", default=DEFAULT_NPZ_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--context-len", type=int, default=16)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--mixed-tf-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-mixed", action="store_true")
    ap.add_argument("--skip-train-free", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap


def _make_runner_args(args: argparse.Namespace, model_path: Path) -> argparse.Namespace:
    return SimpleNamespace(
        teacher=[str(args.teacher)],
        model=str(model_path),
        bundle=str(Path(args.bundle).expanduser().resolve()),
        pretrain_template=str(Path(args.pretrain_template).expanduser().resolve()),
        onnx_model=None,
        encoder_bundle=str(Path(args.encoder_bundle).expanduser().resolve()),
        npz_root=str(Path(args.npz_root).expanduser().resolve()),
        out=str(Path(args.out).expanduser().resolve()),
        device=str(args.device),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        context_len=int(args.context_len),
        depth=int(args.depth),
        force=True,
        with_denorm=True,
        quiet=bool(args.quiet),
        angvel_source="state",
        pose_hist_source="buffer",
        pose_hist_ablation="none",
        pose_hist_keep_last=1,
        pose_hist_time_shift=0,
        angvel_ablation="none",
    )


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _prepare_rollout_inputs(
    runner: TeacherRolloutRunner,
    *,
    teacher_path: Path,
    npz_root: Path,
    max_frames: int,
) -> Dict[str, Any]:
    data = load_json(teacher_path)
    clip_name = str(data.get("clip") or teacher_path.stem.replace("_teacher", ""))
    teacher_block = data.get("teacher")
    if not isinstance(teacher_block, dict):
        raise ValueError(f"{teacher_path}: missing 'teacher' payload.")

    state_arr = np.asarray(teacher_block.get("state_norm"), dtype=np.float32)
    cond_raw_arr = np.asarray(teacher_block.get("cond"), dtype=np.float32)
    if state_arr.ndim != 2 or cond_raw_arr.ndim != 2:
        raise ValueError(f"{teacher_path}: invalid state/cond shapes.")

    npz_path = resolve_npz_path(clip_name, data.get("source_json"), npz_root)
    ds, clip = runner._build_dataset(npz_path)
    runner._ensure_model_ready(ds)

    contacts = clip.contacts if clip.contacts is not None else np.zeros((state_arr.shape[0], runner.contact_dim or 0), dtype=np.float32)
    angvel = clip.angvel_norm if clip.angvel_norm is not None else np.zeros((state_arr.shape[0], runner.angvel_dim or 0), dtype=np.float32)
    pose_hist = clip.pose_hist_norm if clip.pose_hist_norm is not None else np.zeros((state_arr.shape[0], runner.pose_hist_dim or 0), dtype=np.float32)
    gt_norm = clip.Y

    usable_len = _min_length(state_arr, cond_raw_arr, contacts, angvel, pose_hist, gt_norm)
    max_frames = max(2, int(max_frames or usable_len))
    usable_len = min(int(usable_len), max_frames)

    state_arr = state_arr[:usable_len]
    cond_raw_arr = cond_raw_arr[:usable_len]
    contacts = contacts[:usable_len]
    angvel = angvel[:usable_len]
    pose_hist = pose_hist[:usable_len]
    gt_norm = gt_norm[:usable_len]

    cond_arr = cond_raw_arr.copy()
    cond_norm_mu = None
    cond_norm_std = None
    if cond_arr.shape[1] > 0 and bool(getattr(ds, "normalize_c", True)):
        try:
            mu, std = ds._robust_mean_std(cond_raw_arr)
            std = np.clip(np.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6), 1e-6, None)
            mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            mu = np.nan_to_num(getattr(ds, "C_mu", None), nan=0.0, posinf=0.0, neginf=0.0)
            std = np.nan_to_num(getattr(ds, "C_std", None), nan=1e-6, posinf=1e-6, neginf=1e-6)
            std = np.clip(std, 1e-6, None)
        cond_norm_mu = np.asarray(mu, dtype=np.float32).reshape(-1)
        cond_norm_std = np.asarray(std, dtype=np.float32).reshape(-1)
        cond_arr = (cond_raw_arr - cond_norm_mu) / cond_norm_std
        np.nan_to_num(cond_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(cond_arr, -6.0, 6.0, out=cond_arr)

    pose_shift = int(getattr(runner.args, "pose_hist_time_shift", 0) or 0)
    if pose_shift != 0 and pose_hist.shape[1] > 0:
        pose_hist = _shift_time_axis(pose_hist, shift=pose_shift)
    pose_mode = str(getattr(runner.args, "pose_hist_ablation", "none") or "none").lower().strip()
    if pose_mode not in ("", "none") and pose_hist.shape[1] > 0:
        pose_hist = _ablate_pose_hist(
            pose_hist,
            pose_hist_len=int(runner.pose_hist_len),
            mode=pose_mode,
            keep_last=int(getattr(runner.args, "pose_hist_keep_last", 1) or 1),
        )
    ang_mode = str(getattr(runner.args, "angvel_ablation", "none") or "none").lower().strip()
    if ang_mode == "zero" and angvel.shape[1] > 0:
        angvel = np.zeros_like(angvel)

    state_t = torch.from_numpy(state_arr).unsqueeze(0).to(runner.device)
    cond_t = torch.from_numpy(cond_arr).unsqueeze(0).to(runner.device)
    cond_raw_t = torch.from_numpy(cond_raw_arr).unsqueeze(0).to(runner.device)
    contacts_t = torch.from_numpy(contacts).unsqueeze(0).to(runner.device) if contacts.shape[1] > 0 else None
    angvel_t = torch.from_numpy(angvel).unsqueeze(0).to(runner.device) if angvel.shape[1] > 0 else None
    pose_hist_t = torch.from_numpy(pose_hist).unsqueeze(0).to(runner.device) if pose_hist.shape[1] > 0 else None
    gt_t = torch.from_numpy(gt_norm).unsqueeze(0).to(runner.device)
    cond_norm_mu_t = torch.from_numpy(cond_norm_mu).to(runner.device) if cond_norm_mu is not None else None
    cond_norm_std_t = torch.from_numpy(cond_norm_std).to(runner.device) if cond_norm_std is not None else None

    return {
        "clip_name": clip_name,
        "teacher_json": str(teacher_path.resolve()),
        "source_json": data.get("source_json"),
        "npz_path": str(npz_path),
        "fps": data.get("fps", getattr(ds, "fps", 60.0)),
        "state_arr": state_arr,
        "cond_arr": cond_arr,
        "cond_raw_arr": cond_raw_arr,
        "contacts": contacts,
        "angvel": angvel,
        "pose_hist": pose_hist,
        "gt_norm": gt_norm,
        "state_t": state_t,
        "cond_t": cond_t,
        "cond_raw_t": cond_raw_t,
        "contacts_t": contacts_t,
        "angvel_t": angvel_t,
        "pose_hist_t": pose_hist_t,
        "gt_t": gt_t,
        "cond_norm_mu_t": cond_norm_mu_t,
        "cond_norm_std_t": cond_norm_std_t,
        "dims": {
            "Dx": int(state_arr.shape[1]),
            "Dy": int(gt_norm.shape[1]),
            "Dc": int(cond_arr.shape[1]),
            "contacts": int(contacts.shape[1]),
            "angvel": int(angvel.shape[1]),
            "pose_hist": int(pose_hist.shape[1]),
        },
    }


def _reset_trainer_rollout_state(trainer: Any) -> None:
    for name in (
        "_last_step_debug_stats",
    ):
        if hasattr(trainer, name):
            delattr(trainer, name)
    trainer._diag_roll_mode = None
    trainer._diag_roll_step = -1


def _validate_pred_tensor(name: str, tensor: torch.Tensor, total_steps: int) -> Dict[str, Any]:
    if not torch.is_tensor(tensor):
        raise TypeError(f"preds[{name!r}] is not a tensor")
    if tensor.dim() < 2:
        raise ValueError(f"preds[{name!r}] must have batch dimension, got shape {tuple(tensor.shape)}")
    if int(tensor.shape[0]) != 1:
        raise ValueError(f"preds[{name!r}] batch mismatch: expected 1, got {tuple(tensor.shape)}")
    if tensor.dim() >= 3 and int(tensor.shape[1]) != int(total_steps):
        raise ValueError(
            f"preds[{name!r}] time mismatch: expected {total_steps}, got {tuple(tensor.shape)}"
        )
    finite = bool(torch.isfinite(tensor).all().item())
    if not finite:
        raise ValueError(f"preds[{name!r}] contains non-finite values")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "finite": finite,
    }


def _summarize_case(
    runner: TeacherRolloutRunner,
    inputs: Dict[str, Any],
    case: SmokeCase,
) -> Dict[str, Any]:
    trainer = runner.trainer
    _reset_trainer_rollout_state(trainer)
    runner.model.eval()

    with torch.no_grad():
        preds, last_attn = trainer._rollout_sequence(
            inputs["state_t"],
            inputs["cond_t"],
            cond_raw_seq=inputs["cond_raw_t"],
            contacts_seq=inputs["contacts_t"],
            angvel_seq=inputs["angvel_t"],
            pose_hist_seq=inputs["pose_hist_t"],
            gt_seq=inputs["gt_t"],
            cond_norm_mu=inputs["cond_norm_mu_t"],
            cond_norm_std=inputs["cond_norm_std_t"],
            mode=case.mode,
            tf_ratio=float(case.tf_ratio),
        )

    total_steps = int(inputs["state_t"].shape[1])
    key_set = sorted(preds.keys())
    for key in REQUIRED_KEYS:
        if key not in preds:
            raise KeyError(f"Missing required pred key: {key}")

    tensor_summaries: Dict[str, Any] = {}
    for key, value in preds.items():
        if torch.is_tensor(value):
            tensor_summaries[key] = _validate_pred_tensor(key, value, total_steps)

    out_t = preds["out"]
    delta_t = preds["delta"]
    if tuple(out_t.shape) != tuple(inputs["gt_t"].shape):
        raise ValueError(
            f"preds['out'] shape mismatch: expected {tuple(inputs['gt_t'].shape)}, got {tuple(out_t.shape)}"
        )
    if tuple(delta_t.shape) != tuple(inputs["gt_t"].shape):
        raise ValueError(
            f"preds['delta'] shape mismatch: expected {tuple(inputs['gt_t'].shape)}, got {tuple(delta_t.shape)}"
        )

    pred_norm = out_t[0].detach().cpu().numpy()
    gt_norm = inputs["gt_norm"]
    mse_norm = float(np.mean((pred_norm - gt_norm) ** 2))

    with torch.no_grad():
        pred_raw_t = trainer._denorm(out_t)
        gt_raw_t = trainer._denorm(inputs["gt_t"])
    pred_raw = pred_raw_t[0].detach().cpu().numpy()
    gt_raw = gt_raw_t[0].detach().cpu().numpy()
    geo_deg = runner._compute_geo_deg(pred_raw, gt_raw)

    optional_present = sorted([key for key in OPTIONAL_KEYS if key in preds])
    return {
        "mode": case.mode,
        "tf_ratio": float(case.tf_ratio),
        "keys": key_set,
        "optional_keys_present": optional_present,
        "last_attn_shape": list(last_attn.shape) if torch.is_tensor(last_attn) else None,
        "metrics": {
            "MSEnormY": mse_norm,
            "GeoDeg": None if geo_deg is None else float(geo_deg),
        },
        "tensor_summaries": tensor_summaries,
    }


def main() -> None:
    args = _build_parser().parse_args()
    model_path = _resolve_default_model(args.model)
    teacher_path = Path(args.teacher).expanduser().resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Teacher batch not found: {teacher_path}")

    _set_seed(int(args.seed))
    runner = TeacherRolloutRunner(_make_runner_args(args, model_path))
    inputs = _prepare_rollout_inputs(
        runner,
        teacher_path=teacher_path,
        npz_root=Path(args.npz_root).expanduser().resolve(),
        max_frames=int(args.max_frames),
    )

    cases = [SmokeCase(name="teacher", mode="mixed", tf_ratio=1.0)]
    if not bool(args.skip_mixed):
        cases.append(SmokeCase(name="mixed", mode="mixed", tf_ratio=float(args.mixed_tf_ratio)))
    if not bool(args.skip_train_free):
        cases.append(SmokeCase(name="train_free", mode="train_free", tf_ratio=0.0))

    results: Dict[str, Any] = {}
    for idx, case in enumerate(cases):
        _set_seed(int(args.seed) + idx)
        case_summary = _summarize_case(runner, inputs, case)
        results[case.name] = case_summary
        if not args.quiet:
            metrics = case_summary["metrics"]
            print(
                f"[rollout-smoke] {case.name:>10s} mode={case.mode:<10s} tf={case.tf_ratio:.3f} "
                f"mse={metrics['MSEnormY']:.6f} geo={metrics['GeoDeg']}"
            )

    summary = {
        "clip": inputs["clip_name"],
        "teacher_json": inputs["teacher_json"],
        "source_json": inputs["source_json"],
        "npz_path": inputs["npz_path"],
        "model": str(model_path),
        "device": str(runner.device),
        "frames": int(inputs["state_t"].shape[1]),
        "dims": inputs["dims"],
        "cases": results,
    }

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{inputs['clip_name']}_rollout_smoke.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print(f"[rollout-smoke][PASS] wrote {out_path}")


if __name__ == "__main__":
    main()
