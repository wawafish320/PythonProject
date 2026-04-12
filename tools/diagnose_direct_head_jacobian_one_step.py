#!/usr/bin/env python3
"""
Single-batch Jacobian flip diagnosis:
1) measure Jacobian R/L on one batch
2) apply exactly one SGD update on the same batch
3) measure Jacobian R/L again on the same batch

This isolates whether "first gradient step changes working point and flips Jacobian".
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.validate.run_freerun_cycles import (
    FreeRunCycleRunner,
    _build_full_cycle_sample,
    _load_json,
    _resolve_npz_path,
)


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
        if "-" in s:
            a, b = s.split("-", 1)
            try:
                lo = int(a.strip())
                hi = int(b.strip())
            except Exception:
                continue
            if hi < lo:
                lo, hi = hi, lo
            for v in range(lo, hi + 1):
                if 0 <= v < int(total_steps) and v not in seen:
                    seen.add(v)
                    out.append(v)
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
        if not s:
            continue
        if s in seen:
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


def _as_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


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


def _resolve_bone_names(loss_fn: Any, joint_count: int) -> List[str]:
    candidates: List[Sequence[str]] = []
    for key in ("bone_names", "joint_names", "_bone_names"):
        v = getattr(loss_fn, key, None)
        if isinstance(v, (list, tuple)) and len(v) >= int(joint_count):
            candidates.append(v)
    meta = getattr(loss_fn, "meta", None)
    if isinstance(meta, dict):
        v = meta.get("bone_names")
        if isinstance(v, (list, tuple)) and len(v) >= int(joint_count):
            candidates.append(v)
    if candidates:
        return [str(x) for x in list(candidates[0])[: int(joint_count)]]
    return [f"joint_{i}" for i in range(int(joint_count))]


def _agg_rows(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals = np.asarray([_as_float(r.get(key, float("nan"))) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std()),
    }


def _to_batched_device(sample: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sample.items():
        if not torch.is_tensor(v):
            continue
        t = v.to(device)
        if t.dim() == 0:
            out[k] = t.reshape(1)
        else:
            out[k] = t.unsqueeze(0)
    return out


def _measure_jacobian(
    *,
    model: torch.nn.Module,
    trainer: Any,
    state: torch.Tensor,
    cond: Optional[torch.Tensor],
    contacts: Optional[torch.Tensor],
    angvel: Optional[torch.Tensor],
    pose_hist: Optional[torch.Tensor],
    steps: List[int],
    left_slices: List[slice],
    right_slices: List[slice],
) -> Dict[str, Any]:
    if getattr(model, "direct_pose_head", None) is None:
        raise SystemExit("[FATAL] model has no direct_pose_head.")
    first_linear = _find_first_linear(model.direct_pose_head)
    if first_linear is None:
        raise SystemExit("[FATAL] cannot find first Linear layer in direct_pose_head.")

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

    model.zero_grad(set_to_none=True)
    model.eval()
    hook = first_linear.register_forward_pre_hook(_pre_hook)
    rows: List[Dict[str, Any]] = []
    try:
        with torch.enable_grad():
            # Keep same policy as existing jac probe.
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
                raise SystemExit("[FATAL] model forward missing out_direct.")
            out_direct = ret["out_direct"]
            if not torch.is_tensor(out_direct) or out_direct.dim() != 3:
                raise SystemExit(f"[FATAL] out_direct shape invalid: {tuple(getattr(out_direct, 'shape', ())) }")
            x = capture.get("x")
            if not torch.is_tensor(x):
                raise SystemExit("[FATAL] failed to capture direct head input tensor.")

            total_t = int(out_direct.shape[1])
            for i, t in enumerate(steps):
                if int(t) >= total_t:
                    continue
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

                gl_all = float(g_left.norm().detach().cpu())
                gr_all = float(g_right.norm().detach().cpu())
                ratio_all = _safe_ratio(gr_all, gl_all)

                if g_left.dim() >= 3 and int(t) < int(g_left.shape[1]):
                    gl_local = float(g_left[:, int(t), :].norm().detach().cpu())
                    gr_local = float(g_right[:, int(t), :].norm().detach().cpu())
                else:
                    gl_local = gl_all
                    gr_local = gr_all
                ratio_local = _safe_ratio(gr_local, gl_local)

                rows.append(
                    {
                        "step": int(t),
                        "grad_norm_left_all": gl_all,
                        "grad_norm_right_all": gr_all,
                        "grad_ratio_r_over_l_all": ratio_all,
                        "grad_norm_left_local_t": gl_local,
                        "grad_norm_right_local_t": gr_local,
                        "grad_ratio_r_over_l_local_t": ratio_local,
                    }
                )
    finally:
        hook.remove()

    return {
        "per_step": rows,
        "aggregate": {
            "ratio_r_over_l_all": _agg_rows(rows, "grad_ratio_r_over_l_all"),
            "ratio_r_over_l_local_t": _agg_rows(rows, "grad_ratio_r_over_l_local_t"),
        },
        "direct_head_input_shape": list(capture["x"].shape) if torch.is_tensor(capture.get("x")) else [],
    }


def _global_grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
    sq = 0.0
    for p in params:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        v = float(g.detach().norm().cpu())
        sq += v * v
    return float(math.sqrt(max(sq, 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="One-step same-batch Jacobian flip diagnostic (pre-jac -> 1-step SGD -> post-jac).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--teacher", type=str, default="validate/teacher_batches/Walk_F_teacher.json")
    ap.add_argument("--config-json", type=str, default="config/exp_phase_DirectBranch_v1_d1_noreset.json")
    ap.add_argument(
        "--model",
        type=str,
        default="models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage6_direct_cond_anchor_20260124.pth",
    )
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--context-len", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--steps", type=str, default="0,1,2,10,20")
    ap.add_argument("--left-bones", type=str, default="thigh_l,calf_l,foot_l,ball_l")
    ap.add_argument("--right-bones", type=str, default="thigh_r,calf_r,foot_r,ball_r")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sgd-lr", type=float, default=1e-3)
    ap.add_argument("--sgd-momentum", type=float, default=0.0)
    ap.add_argument("--sgd-weight-decay", type=float, default=1e-2)
    ap.add_argument("--sgd-nesterov", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="<=0 disables clipping")
    ap.add_argument("--out", type=str, default="", help="Optional JSON output path.")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    teacher_path = Path(args.teacher).expanduser().resolve()
    if not teacher_path.is_file():
        raise SystemExit(f"[FATAL] teacher not found: {teacher_path}")
    teacher_data = _load_json(teacher_path)
    teacher_block = teacher_data.get("teacher")
    if not isinstance(teacher_block, dict):
        raise SystemExit(f"[FATAL] teacher payload invalid: {teacher_path}")
    state_arr = np.asarray(teacher_block.get("state_norm"), dtype=np.float32)
    if state_arr.ndim != 2:
        raise SystemExit(f"[FATAL] invalid state_norm shape in teacher: {state_arr.shape}")

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
    ds = runner._build_dataset(npz_path, seq_len=seq_len)
    runner._ensure_model_ready(ds)
    model = runner.model
    trainer = runner.trainer
    if model is None or trainer is None:
        raise SystemExit("[FATAL] failed to initialize model/trainer.")

    cfg_path = Path(args.config_json).expanduser().resolve()
    cfg: Dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cfg = dict(raw)
        except Exception:
            cfg = {}
    else:
        print(f"[WARN] config-json not found, using minimal in-script defaults: {cfg_path}")

    def _cfg(name: str, default: Any) -> Any:
        return cfg.get(name, default)

    if cfg:
        legacy_loss_keys = (
            "ignore_motion_groups",
            "bone_prior_stds",
            "use_hierarchy_weights",
            "hierarchy_mode",
            "hierarchy_alpha",
            "max_weight_ratio",
            "weight_gamma",
            "bone_prior_mode",
            "bone_prior_samples",
        )
        legacy_hits = [k for k in legacy_loss_keys if k in cfg]
        if legacy_hits:
            joined = ", ".join(sorted(legacy_hits))
            raise SystemExit(
                "[FATAL] config_json contains removed MotionJointLoss keys: "
                f"{joined}. Please remove them and keep unified_* knobs only."
            )

    # Align key loss weights with train entry (enough for non-zero 1-step update on same batch).
    lf = trainer.loss_fn
    lf.w_rot_ortho = float(_cfg("w_rot_ortho", getattr(lf, "w_rot_ortho", 0.001)))
    lf.w_rot_local = float(_cfg("w_rot_local", getattr(lf, "w_rot_local", 0.2)))
    lf.w_rot_vel = float(_cfg("w_rot_vel", getattr(lf, "w_rot_vel", 0.0)))
    lf.rot_vel_log_scale = float(_cfg("rot_vel_log_scale", getattr(lf, "rot_vel_log_scale", 1.0)))
    lf.rot_vel_omega_min_deg_s = float(_cfg("rot_vel_omega_min_deg_s", getattr(lf, "rot_vel_omega_min_deg_s", 0.0)))
    lf.rot_vel_loss = str(_cfg("rot_vel_loss", getattr(lf, "rot_vel_loss", "smooth_l1")) or "smooth_l1")
    lf.w_root_vel = float(_cfg("w_root_vel", getattr(lf, "w_root_vel", 0.0)))
    lf.w_root_speed = float(_cfg("w_root_speed", getattr(lf, "w_root_speed", 0.0)))
    lf.w_contact_plan = float(_cfg("w_contact_plan", getattr(lf, "w_contact_plan", 0.0)))
    lf.w_contact_meas = float(_cfg("w_contact_meas", getattr(lf, "w_contact_meas", 0.0)))
    lf.w_contact_td_hazard_bce = float(
        _cfg("w_contact_td_hazard_bce", getattr(lf, "w_contact_td_hazard_bce", 0.0))
    )
    lf.w_contact_td_hazard_mass = float(
        _cfg("w_contact_td_hazard_mass", getattr(lf, "w_contact_td_hazard_mass", 0.0))
    )
    lf.w_contact_td_hazard_unimodal = float(
        _cfg("w_contact_td_hazard_unimodal", getattr(lf, "w_contact_td_hazard_unimodal", 0.0))
    )
    lf.w_direct_pose = float(_cfg("w_direct_pose", getattr(lf, "w_direct_pose", 0.2)))
    lf.direct_pose_side_weight_left = float(
        _cfg("direct_pose_side_weight_left", getattr(lf, "direct_pose_side_weight_left", 1.0))
    )
    lf.direct_pose_side_weight_right = float(
        _cfg("direct_pose_side_weight_right", getattr(lf, "direct_pose_side_weight_right", 1.0))
    )
    lf.w_direct_delta = float(_cfg("w_direct_delta", getattr(lf, "w_direct_delta", 0.0)))
    lf.w_direct_delta_sym = float(_cfg("w_direct_delta_sym", getattr(lf, "w_direct_delta_sym", 0.0)))
    lf.rot_local_tail_weight = float(_cfg("rot_local_tail_weight", getattr(lf, "rot_local_tail_weight", 0.0)))
    lf.rot_local_tail_k = int(_cfg("rot_local_tail_k", getattr(lf, "rot_local_tail_k", 0)))
    lf.rot_local_tail_scope = str(_cfg("rot_local_tail_scope", getattr(lf, "rot_local_tail_scope", "all")) or "all")
    lf.rot_local_tail_select = str(
        _cfg("rot_local_tail_select", getattr(lf, "rot_local_tail_select", "batch")) or "batch"
    )
    lf.rot_local_tail_ema_beta = float(_cfg("rot_local_tail_ema_beta", getattr(lf, "rot_local_tail_ema_beta", 0.9)))
    lf.rot_local_tail_reduce = str(_cfg("rot_local_tail_reduce", getattr(lf, "rot_local_tail_reduce", "flat")) or "flat")
    lf.rot_local_tail_uniform_mix = float(
        _cfg("rot_local_tail_uniform_mix", getattr(lf, "rot_local_tail_uniform_mix", 0.4))
    )
    lf.rot_local_tail_rank_mix = float(
        _cfg("rot_local_tail_rank_mix", getattr(lf, "rot_local_tail_rank_mix", 0.6))
    )
    lf.unified_downstream_power = float(
        _cfg("unified_downstream_power", getattr(lf, "unified_downstream_power", 0.6))
    )
    lf.unified_self_scale = float(_cfg("unified_self_scale", getattr(lf, "unified_self_scale", 1.5)))
    lf.unified_min_weight = float(_cfg("unified_min_weight", getattr(lf, "unified_min_weight", 0.05)))
    try:
        if getattr(ds, "bone_names", None):
            lf.set_bone_names(ds.bone_names)
    except Exception:
        pass
    try:
        if getattr(ds, "parents", None):
            lf.set_skeleton(ds.parents, getattr(ds, "bone_offsets", None))
    except Exception:
        pass
    print(
        "[LossCfg] "
        f"w_rot_local={lf.w_rot_local} w_direct_pose={lf.w_direct_pose} "
        f"w_contact_plan={lf.w_contact_plan} w_contact_meas={lf.w_contact_meas}"
    )

    clip = ds.clips[0]
    sample = _build_full_cycle_sample(ds, clip, seq_len=seq_len)
    batch = _to_batched_device(sample, runner.device)

    state = batch["motion"]
    gt = batch["gt_motion"]
    cond = batch.get("cond_in")
    cond_raw = batch.get("cond_tgt_raw")
    contacts = batch.get("contacts")
    angvel = batch.get("angvel")
    pose_hist = batch.get("pose_hist")
    cond_norm_mu = batch.get("cond_norm_mu")
    cond_norm_std = batch.get("cond_norm_std")
    time_base = batch.get("start")

    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        raise SystemExit("[FATAL] trainer rot6d slice is missing.")

    dy = int(getattr(ds, "Y", np.zeros((1, 1), dtype=np.float32)).shape[-1])
    st = int(rot_slice.start or 0)
    ed = int(rot_slice.stop or dy)
    if ed <= st or ((ed - st) % 6) != 0:
        raise SystemExit(f"[FATAL] invalid rot6d slice: [{st}:{ed}] over Dy={dy}")
    joint_count = (ed - st) // 6
    bone_names = _resolve_bone_names(trainer.loss_fn, joint_count)
    name_to_idx = {str(n): i for i, n in enumerate(bone_names)}

    left_bones = _parse_csv(args.left_bones)
    right_bones = _parse_csv(args.right_bones)
    if len(left_bones) != len(right_bones):
        raise SystemExit("[FATAL] left-bones and right-bones must have equal length.")
    missing = [b for b in (left_bones + right_bones) if b not in name_to_idx]
    if missing:
        raise SystemExit(f"[FATAL] bones not found in model skeleton: {missing}")
    left_slices = [_joint_rot6d_slice(rot_slice, name_to_idx[b]) for b in left_bones]
    right_slices = [_joint_rot6d_slice(rot_slice, name_to_idx[b]) for b in right_bones]

    total_t = int(state.shape[1])
    steps = _parse_steps(args.steps, total_t)
    if not steps:
        raise SystemExit("[FATAL] no valid steps after parsing --steps.")

    # (1) Jacobian before update
    before = _measure_jacobian(
        model=model,
        trainer=trainer,
        state=state,
        cond=cond,
        contacts=contacts,
        angvel=angvel,
        pose_hist=pose_hist,
        steps=steps,
        left_slices=left_slices,
        right_slices=right_slices,
    )

    # (2) One SGD update on the same batch (teacher-forcing style rollout)
    trainable_params = [p for p in model.parameters() if bool(getattr(p, "requires_grad", False))]
    opt = torch.optim.SGD(
        trainable_params,
        lr=float(args.sgd_lr),
        momentum=max(float(args.sgd_momentum), 0.0),
        weight_decay=float(args.sgd_weight_decay),
        nesterov=bool(args.sgd_nesterov and float(args.sgd_momentum) > 0.0),
    )
    model.train()
    opt.zero_grad(set_to_none=True)
    preds_dict, last_attn = trainer._rollout_sequence(
        state,
        cond,
        cond_raw,
        contacts_seq=contacts,
        angvel_seq=angvel,
        pose_hist_seq=pose_hist,
        gt_seq=gt,
        cond_norm_mu=cond_norm_mu,
        cond_norm_std=cond_norm_std,
        mode="mixed",
        tf_ratio=1.0,
        time_base=time_base,
    )
    out = trainer.loss_fn(preds_dict, gt, attn_weights=last_attn, batch=batch)
    if isinstance(out, tuple):
        loss, stats = out
    else:
        loss, stats = out, {}
    loss_val = float(loss.detach().cpu())
    loss.backward()
    grad_norm_before_clip = _global_grad_norm(trainable_params)
    if float(args.grad_clip) > 0.0:
        torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))
    grad_norm_after_clip = _global_grad_norm(trainable_params)
    opt.step()
    opt.zero_grad(set_to_none=True)

    # (3) Jacobian after update (same batch)
    after = _measure_jacobian(
        model=model,
        trainer=trainer,
        state=state,
        cond=cond,
        contacts=contacts,
        angvel=angvel,
        pose_hist=pose_hist,
        steps=steps,
        left_slices=left_slices,
        right_slices=right_slices,
    )

    before_by_step = {int(r["step"]): r for r in before["per_step"]}
    after_by_step = {int(r["step"]): r for r in after["per_step"]}
    delta_rows: List[Dict[str, Any]] = []
    for s in steps:
        if s not in before_by_step or s not in after_by_step:
            continue
        b = before_by_step[s]
        a = after_by_step[s]
        delta_rows.append(
            {
                "step": int(s),
                "before_ratio_r_over_l_all": _as_float(b.get("grad_ratio_r_over_l_all", float("nan"))),
                "after_ratio_r_over_l_all": _as_float(a.get("grad_ratio_r_over_l_all", float("nan"))),
                "delta_ratio_r_over_l_all": _as_float(a.get("grad_ratio_r_over_l_all", float("nan")))
                - _as_float(b.get("grad_ratio_r_over_l_all", float("nan"))),
            }
        )

    print("Same-batch Jacobian one-step flip diagnostic")
    print(f"model: {Path(args.model).expanduser().resolve()}")
    print(f"teacher: {teacher_path}")
    print(f"steps: {steps}")
    print(
        f"sgd: lr={float(args.sgd_lr):.2e}, momentum={float(args.sgd_momentum):.3f}, "
        f"wd={float(args.sgd_weight_decay):.2e}, grad_clip={float(args.grad_clip):.3f}"
    )
    print("-" * 84)
    print("step | jac_before(R/L) | jac_after(R/L) | delta")
    for r in delta_rows:
        print(
            f"{int(r['step']):4d} | "
            f"{float(r['before_ratio_r_over_l_all']):15.6f} | "
            f"{float(r['after_ratio_r_over_l_all']):14.6f} | "
            f"{float(r['delta_ratio_r_over_l_all']):+8.6f}"
        )
    b_mean = _as_float(before["aggregate"]["ratio_r_over_l_all"].get("mean", float("nan")))
    a_mean = _as_float(after["aggregate"]["ratio_r_over_l_all"].get("mean", float("nan")))
    print("-" * 84)
    print(f"agg mean R/L(all): before={b_mean:.6f} after={a_mean:.6f} delta={a_mean - b_mean:+.6f}")
    print(
        f"update: loss={loss_val:.6f} grad_norm(before_clip)={grad_norm_before_clip:.6f} "
        f"grad_norm(after_clip)={grad_norm_after_clip:.6f}"
    )

    payload: Dict[str, Any] = {
        "model": str(Path(args.model).expanduser().resolve()),
        "teacher": str(teacher_path),
        "npz": str(npz_path),
        "probe": {
            "steps": steps,
            "left_bones": left_bones,
            "right_bones": right_bones,
            "left_rot6d_dims": int(6 * len(left_bones)),
            "right_rot6d_dims": int(6 * len(right_bones)),
            "rot6d_slice": [int(rot_slice.start or 0), int(rot_slice.stop or 0)],
            "direct_head_input_shape": before.get("direct_head_input_shape", []),
        },
        "update": {
            "optimizer": "sgd",
            "sgd_lr": float(args.sgd_lr),
            "sgd_momentum": float(args.sgd_momentum),
            "sgd_weight_decay": float(args.sgd_weight_decay),
            "sgd_nesterov": bool(args.sgd_nesterov and float(args.sgd_momentum) > 0.0),
            "grad_clip": float(args.grad_clip),
            "loss": loss_val,
            "grad_norm_before_clip": grad_norm_before_clip,
            "grad_norm_after_clip": grad_norm_after_clip,
            "loss_stats": stats if isinstance(stats, dict) else {},
        },
        "loss_config_applied": {
            "config_json": str(cfg_path) if cfg else "",
            "w_rot_ortho": float(lf.w_rot_ortho),
            "w_rot_local": float(lf.w_rot_local),
            "w_rot_vel": float(lf.w_rot_vel),
            "w_root_vel": float(lf.w_root_vel),
            "w_root_speed": float(lf.w_root_speed),
            "w_contact_plan": float(lf.w_contact_plan),
            "w_contact_meas": float(lf.w_contact_meas),
            "w_direct_pose": float(lf.w_direct_pose),
            "direct_pose_side_weight_left": float(lf.direct_pose_side_weight_left),
            "direct_pose_side_weight_right": float(lf.direct_pose_side_weight_right),
            "w_direct_delta": float(lf.w_direct_delta),
            "w_direct_delta_sym": float(lf.w_direct_delta_sym),
        },
        "before_update": {
            "per_step": before["per_step"],
            "aggregate": before["aggregate"],
        },
        "after_update": {
            "per_step": after["per_step"],
            "aggregate": after["aggregate"],
        },
        "delta": {
            "per_step_ratio_r_over_l_all": delta_rows,
            "aggregate_ratio_r_over_l_all_mean": {
                "before": b_mean,
                "after": a_mean,
                "delta": a_mean - b_mean,
            },
        },
    }

    out_path = str(args.out or "").strip()
    if out_path:
        p = Path(out_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Saved] {p}")


if __name__ == "__main__":
    main()
