#!/usr/bin/env python3
"""
Probe direct-head Jacobian asymmetry for one cond channel on a reconstructed batch.

Use case:
- Load a checkpoint.
- Rebuild a batch from (clip_id, start) pairs saved in a freerun diag .pt artifact.
- Measure ||dL_left/dx|| and ||dL_right/dx|| where x is direct head input.
- Report per-step channel-specific norms for cond channel c (default c=6, speed).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.dataset import MotionEventDataset
from train.posttrain_common import _merge_norm_spec
from train.validate.run_freerun_cycles import FreeRunCycleRunner, _load_json, _resolve_npz_path


def _parse_steps(spec: str, total_steps: int) -> List[int]:
    txt = str(spec or "").strip().lower()
    if txt in ("", "all", "*"):
        return list(range(int(total_steps)))
    out: List[int] = []
    seen = set()
    for tok in txt.split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            v = int(s)
        except Exception:
            continue
        if 0 <= v < int(total_steps) and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _parse_csv(spec: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for tok in str(spec or "").split(","):
        s = tok.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    if not (math.isfinite(num) and math.isfinite(den)):
        return float("nan")
    if abs(den) <= eps:
        return float("nan")
    return float(num / den)


def _find_first_linear(module: torch.nn.Module) -> Optional[torch.nn.Linear]:
    if module is None:
        return None
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            return m
    return None


def _joint_rot6d_slice(rot_slice: slice, joint_idx: int) -> slice:
    st = int(rot_slice.start or 0) + 6 * int(joint_idx)
    ed = st + 6
    return slice(st, ed)


def _load_runner(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    teacher_path = Path(args.teacher).expanduser().resolve()
    if not teacher_path.is_file():
        raise SystemExit(f"[FATAL] teacher not found: {teacher_path}")
    teacher_data = _load_json(teacher_path)
    teacher_block = teacher_data.get("teacher")
    if not isinstance(teacher_block, dict):
        raise SystemExit(f"[FATAL] invalid teacher payload: {teacher_path}")
    state_arr = np.asarray(teacher_block.get("state_norm"), dtype=np.float32)
    if state_arr.ndim != 2:
        raise SystemExit(f"[FATAL] invalid teacher state_norm shape: {state_arr.shape}")

    clip_name = str(teacher_data.get("clip") or teacher_path.stem.replace("_teacher", ""))
    npz_path = _resolve_npz_path(
        clip_name,
        teacher_data.get("source_json"),
        Path(args.npz_root).expanduser().resolve(),
    )

    runner_args = argparse.Namespace(
        model=str(Path(args.model).expanduser().resolve()),
        device=str(args.device),
        bundle=str(Path(args.bundle).expanduser()),
        pretrain_template=str(Path(args.pretrain_template).expanduser()),
        encoder_bundle=str(Path(args.encoder_bundle).expanduser()),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        context_len=int(args.context_len),
        depth=int(args.depth),
        so3_corr_apply=False,
        so3_corr_max_deg=20.0,
        lambda_fusion_apply=False,
    )
    runner = FreeRunCycleRunner(runner_args)
    seq_len = int(state_arr.shape[0])
    ds_one = runner._build_dataset(npz_path, seq_len=seq_len)
    runner._ensure_model_ready(ds_one)
    return runner, ds_one, npz_path


def _rebuild_batch_from_diag(
    *,
    diag_pt: Path,
    seq_len: int,
    data_root: Path,
    bundle: Path,
    pretrain_template: Path,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    payload = torch.load(str(diag_pt), map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"[FATAL] invalid diag pt payload: {diag_pt}")
    clip_id = payload.get("clip_id")
    start = payload.get("start")
    if not (torch.is_tensor(clip_id) and torch.is_tensor(start)):
        raise SystemExit(f"[FATAL] diag pt missing clip_id/start tensors: {diag_pt}")
    clip_ids = [int(x) for x in clip_id.view(-1).tolist()]
    starts = [int(x) for x in start.view(-1).tolist()]
    if len(clip_ids) != len(starts) or not clip_ids:
        raise SystemExit(f"[FATAL] invalid clip/start lengths in {diag_pt}")

    norm_spec = _merge_norm_spec(bundle.resolve(), pretrain_template.resolve())
    ds = MotionEventDataset(
        data_dir=str(data_root.resolve()),
        seq_len=int(seq_len),
        paths=None,
        pose_hist_len=int(norm_spec.get("pose_hist_len", 0) or 0),
        norm_spec=norm_spec,
        index_mode="sliding",
    )
    ds.is_train = False
    pair_to_idx = {(int(cid), int(st)): int(i) for i, (cid, st) in enumerate(ds.index)}
    samples: List[Dict[str, torch.Tensor]] = []
    missing: List[Tuple[int, int]] = []
    for cid, st in zip(clip_ids, starts):
        idx = pair_to_idx.get((cid, st))
        if idx is None:
            missing.append((cid, st))
            continue
        samples.append(ds[idx])
    if missing:
        raise SystemExit(f"[FATAL] missing (clip_id,start) pairs: {missing[:8]}")
    if not samples:
        raise SystemExit("[FATAL] no samples reconstructed from diag pt")

    out: Dict[str, torch.Tensor] = {}
    keys = set()
    for s in samples:
        keys.update(list(s.keys()))
    for k in sorted(keys):
        vals = [s[k] for s in samples if k in s]
        if len(vals) != len(samples):
            continue
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0).to(device)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe direct head jacobian for one cond channel on a reconstructed batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--diag-pt", type=str, required=True, help="freerun_diag_epXXX_..._bYY.pt with clip_id/start.")
    ap.add_argument("--teacher", type=str, default="validate/teacher_batches/Walk_F_teacher.json")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--data-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--device", type=str, default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--context-len", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--steps", type=str, default="0,1")
    ap.add_argument("--left-bones", type=str, default="thigh_l,calf_l,foot_l,ball_l")
    ap.add_argument("--right-bones", type=str, default="thigh_r,calf_r,foot_r,ball_r")
    ap.add_argument("--cond-channel", type=int, default=6, help="cond channel index in direct-head input x.")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    runner, ds_one, npz_path = _load_runner(args)
    model = runner.model
    trainer = runner.trainer
    if model is None or trainer is None:
        raise SystemExit("[FATAL] failed to load model/trainer.")
    if getattr(model, "direct_pose_head", None) is None:
        raise SystemExit("[FATAL] model has no direct_pose_head.")
    first_linear = _find_first_linear(model.direct_pose_head)
    if first_linear is None:
        raise SystemExit("[FATAL] cannot find first linear in direct_pose_head.")

    batch = _rebuild_batch_from_diag(
        diag_pt=Path(args.diag_pt).expanduser().resolve(),
        seq_len=int(args.seq_len),
        data_root=Path(args.data_root).expanduser().resolve(),
        bundle=Path(args.bundle).expanduser(),
        pretrain_template=Path(args.pretrain_template).expanduser(),
        device=runner.device,
    )

    state = batch.get("motion")
    cond = batch.get("cond_in")
    contacts = batch.get("contacts")
    angvel = batch.get("angvel")
    pose_hist = batch.get("pose_hist")
    if not torch.is_tensor(state):
        raise SystemExit("[FATAL] reconstructed batch missing motion.")
    if not torch.is_tensor(cond):
        raise SystemExit("[FATAL] reconstructed batch missing cond_in.")

    cond_dim = int(cond.shape[-1]) if cond.dim() >= 1 else 0
    cond_ch = int(args.cond_channel)
    if not (0 <= cond_ch < cond_dim):
        raise SystemExit(f"[FATAL] cond-channel out of range: {cond_ch} not in [0, {cond_dim})")

    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        raise SystemExit("[FATAL] trainer rot6d slice missing.")
    dy = int(getattr(ds_one, "Y", np.zeros((1, 1), dtype=np.float32)).shape[-1])
    st = int(rot_slice.start or 0)
    ed = int(rot_slice.stop or dy)
    if ed <= st or ((ed - st) % 6) != 0:
        raise SystemExit(f"[FATAL] invalid rot6d slice [{st}:{ed}]")
    joint_count = (ed - st) // 6
    bone_names = list(getattr(ds_one, "bone_names", []) or [])[: int(joint_count)]
    if len(bone_names) < int(joint_count):
        raise SystemExit(f"[FATAL] insufficient bone names: {len(bone_names)}/{joint_count}")
    name_to_idx = {str(n): i for i, n in enumerate(bone_names)}

    left_bones = _parse_csv(args.left_bones)
    right_bones = _parse_csv(args.right_bones)
    if len(left_bones) != len(right_bones):
        raise SystemExit("[FATAL] left-bones and right-bones length mismatch.")
    missing = [b for b in (left_bones + right_bones) if b not in name_to_idx]
    if missing:
        raise SystemExit(f"[FATAL] unresolved bones: {missing}")
    left_slices = [_joint_rot6d_slice(rot_slice, name_to_idx[b]) for b in left_bones]
    right_slices = [_joint_rot6d_slice(rot_slice, name_to_idx[b]) for b in right_bones]

    capture: Dict[str, torch.Tensor] = {}

    def _pre_hook(mod: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]):
        if not inputs:
            return inputs
        x = inputs[0]
        if not torch.is_tensor(x):
            return inputs
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        capture["x"] = x
        if len(inputs) == 1:
            return (x,)
        return (x, *inputs[1:])

    model.eval()
    model.zero_grad(set_to_none=True)
    hook = first_linear.register_forward_pre_hook(_pre_hook)
    rows: List[Dict[str, Any]] = []
    try:
        with torch.enable_grad():
            use_learned_meas = bool(getattr(model, "contact_meas_enable", False)) and getattr(model, "contact_meas_head", None) is not None
            contacts_in = None if use_learned_meas else contacts
            ret = model(
                state,
                cond=cond,
                contacts=contacts_in,
                angvel=angvel,
                pose_history=pose_hist,
                plan_z=None,
                time_index=None,
            )
            if not isinstance(ret, dict) or "out_direct" not in ret:
                raise SystemExit("[FATAL] forward missing out_direct.")
            out_direct = ret["out_direct"]
            if not torch.is_tensor(out_direct) or out_direct.dim() != 3:
                raise SystemExit("[FATAL] invalid out_direct shape.")
            x = capture.get("x")
            if not torch.is_tensor(x):
                raise SystemExit("[FATAL] failed to capture direct head input.")

            steps = _parse_steps(args.steps, int(out_direct.shape[1]))
            if not steps:
                raise SystemExit("[FATAL] no valid steps.")

            for i, t in enumerate(steps):
                y_t = out_direct[:, int(t), :]
                loss_left = sum(y_t[:, sl].sum() for sl in left_slices)
                loss_right = sum(y_t[:, sl].sum() for sl in right_slices)
                g_left = torch.autograd.grad(loss_left, x, retain_graph=True, create_graph=False, allow_unused=False)[0]
                g_right = torch.autograd.grad(
                    loss_right,
                    x,
                    retain_graph=(i < len(steps) - 1),
                    create_graph=False,
                    allow_unused=False,
                )[0]

                l_all = float(g_left.norm().detach().cpu())
                r_all = float(g_right.norm().detach().cpu())
                l_ch = float(g_left[:, int(cond_ch)].norm().detach().cpu())
                r_ch = float(g_right[:, int(cond_ch)].norm().detach().cpu())
                rows.append(
                    {
                        "step": int(t),
                        "grad_norm_left_all": l_all,
                        "grad_norm_right_all": r_all,
                        "grad_ratio_r_over_l_all": _safe_ratio(r_all, l_all),
                        "grad_norm_left_cond_ch": l_ch,
                        "grad_norm_right_cond_ch": r_ch,
                        "grad_ratio_r_over_l_cond_ch": _safe_ratio(r_ch, l_ch),
                    }
                )
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)

    arr_all = np.asarray([_safe_ratio(float(r["grad_norm_right_all"]), float(r["grad_norm_left_all"])) for r in rows], dtype=np.float64)
    arr_ch = np.asarray([_safe_ratio(float(r["grad_norm_right_cond_ch"]), float(r["grad_norm_left_cond_ch"])) for r in rows], dtype=np.float64)
    arr_all = arr_all[np.isfinite(arr_all)]
    arr_ch = arr_ch[np.isfinite(arr_ch)]
    agg = {
        "ratio_r_over_l_all_mean": float(arr_all.mean()) if arr_all.size > 0 else float("nan"),
        "ratio_r_over_l_cond_ch_mean": float(arr_ch.mean()) if arr_ch.size > 0 else float("nan"),
        "ratio_r_over_l_all_n": int(arr_all.size),
        "ratio_r_over_l_cond_ch_n": int(arr_ch.size),
    }

    out: Dict[str, Any] = {
        "model": str(Path(args.model).expanduser().resolve()),
        "diag_pt": str(Path(args.diag_pt).expanduser().resolve()),
        "npz": str(npz_path),
        "probe": {
            "steps": [int(r["step"]) for r in rows],
            "left_bones": left_bones,
            "right_bones": right_bones,
            "cond_channel": int(cond_ch),
            "cond_dim": int(cond_dim),
            "direct_head_input_shape": list(capture.get("x").shape) if torch.is_tensor(capture.get("x")) else [],
        },
        "per_step": rows,
        "aggregate": agg,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if str(args.out or "").strip():
        p = Path(args.out).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Saved] {p}")


if __name__ == "__main__":
    main()
