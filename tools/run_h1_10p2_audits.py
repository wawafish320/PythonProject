#!/usr/bin/env python3
"""
Run H1 route audits requested in docs/Problems/active/2026-02-11_WalkF_stage7_phase_lag_velocity_loss.md
Section 10.2 (A/B/C), using a teacher one-step lens.

Outputs:
  - <out_dir>/h1_10p2_summary.json
  - <out_dir>/h1_10p2_summary.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from train.geometry import geodesic_R, rot6d_to_matrix, so3_log_map
from train.validate.run_freerun_cycles import (
    FreeRunCycleRunner,
    _build_full_cycle_sample,
    _load_json,
    _resolve_npz_path,
)


def _parse_index_spec(spec: str, *, upper: Optional[int] = None) -> List[int]:
    text = str(spec or "").strip().lower()
    if text in ("", "none", "null"):
        return []
    out: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            try:
                lo = int(a.strip())
                hi = int(b.strip())
            except Exception:
                continue
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(token))
            except Exception:
                continue
    out = sorted(set(out))
    if upper is not None:
        out = [x for x in out if 0 <= int(x) < int(upper)]
    return out


def _expand_specs(specs: Sequence[str]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for raw in specs:
        if not raw:
            continue
        spec = str(raw)
        p = Path(spec).expanduser()
        matches: List[Path] = []
        if any(ch in spec for ch in "*?[]"):
            matches = sorted(Path(".").glob(spec))
        elif p.is_dir():
            matches = sorted(p.glob("*.json"))
        elif p.is_file():
            matches = [p]
        for m in matches:
            r = str(m.resolve())
            if r in seen:
                continue
            seen.add(r)
            out.append(Path(r))
    return sorted(out)


def _infer_rot_slice(runner: FreeRunCycleRunner, out_dim: int) -> slice:
    trainer = getattr(runner, "trainer", None)
    sl = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if isinstance(sl, slice):
        return sl
    return slice(0, int(out_dim))


def _to_device(x: Any, device: torch.device) -> Optional[torch.Tensor]:
    if not torch.is_tensor(x):
        return None
    return x.to(device)


def _build_runner(args: argparse.Namespace) -> FreeRunCycleRunner:
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
    return FreeRunCycleRunner(runner_args)


def _load_teacher_sample(
    runner: FreeRunCycleRunner,
    *,
    teacher_path: Path,
    npz_root: Path,
) -> Dict[str, Any]:
    payload = _load_json(teacher_path)
    clip_name = str(payload.get("clip") or teacher_path.stem.replace("_teacher", ""))
    teacher_blk = payload.get("teacher")
    if not isinstance(teacher_blk, dict):
        raise RuntimeError(f"{teacher_path}: missing 'teacher' block.")
    state_arr = np.asarray(teacher_blk.get("state_norm"), dtype=np.float32)
    if state_arr.ndim != 2 or state_arr.shape[0] < 2:
        raise RuntimeError(f"{teacher_path}: invalid state_norm shape={state_arr.shape}.")
    seq_len = int(state_arr.shape[0])
    npz_path = _resolve_npz_path(clip_name, payload.get("source_json"), npz_root)
    ds = runner._build_dataset(npz_path, seq_len=seq_len)
    runner._ensure_model_ready(ds)
    clip = ds.clips[0]
    sample = _build_full_cycle_sample(ds, clip, seq_len=seq_len)
    bone_names = list(getattr(ds, "bone_names", []) or [])
    return {
        "clip": clip_name,
        "teacher_path": str(teacher_path),
        "npz_path": str(npz_path),
        "seq_len": seq_len,
        "sample": sample,
        "bone_names": bone_names,
    }


def _forward_direct_error(
    runner: FreeRunCycleRunner,
    sample: Dict[str, Any],
    *,
    require_grad: bool,
) -> Dict[str, Any]:
    trainer = getattr(runner, "trainer", None)
    if trainer is None or runner.model is None:
        raise RuntimeError("Runner model/trainer is not initialized.")
    model = runner.model.to(runner.device)
    model.eval()

    state = _to_device(sample.get("motion"), runner.device)
    gt = _to_device(sample.get("gt_motion"), runner.device)
    cond = _to_device(sample.get("cond_in"), runner.device)
    angvel = _to_device(sample.get("angvel"), runner.device)
    pose_hist = _to_device(sample.get("pose_hist"), runner.device)
    contacts = _to_device(sample.get("contacts"), runner.device)

    if state is None or gt is None:
        raise RuntimeError("sample missing motion/gt_motion.")
    state = state.unsqueeze(0)
    gt = gt.unsqueeze(0)
    cond = cond.unsqueeze(0) if cond is not None else None
    angvel = angvel.unsqueeze(0) if angvel is not None else None
    pose_hist = pose_hist.unsqueeze(0) if pose_hist is not None else None
    contacts = contacts.unsqueeze(0) if contacts is not None else None

    # Keep model's learned contact_meas path when available.
    use_learned_meas = bool(getattr(model, "contact_meas_enable", False)) and getattr(model, "contact_meas_head", None) is not None
    if use_learned_meas:
        contacts = None

    cm = torch.enable_grad() if require_grad else torch.no_grad()
    with cm:
        ret = model(
            state,
            cond=cond,
            contacts=contacts,
            angvel=angvel,
            pose_history=pose_hist,
            plan_z=None,
            time_index=None,
        )
        if not isinstance(ret, dict) or "out_direct" not in ret:
            raise RuntimeError("model forward missing out_direct.")
        out_direct = ret["out_direct"]  # (1,T,Dy), normalized
        if out_direct.ndim != 3:
            raise RuntimeError(f"out_direct shape {tuple(out_direct.shape)} != (B,T,Dy).")
        out_direct_raw = trainer._denorm(out_direct)
        gt_raw = trainer._denorm(gt)

        rot_slice = _infer_rot_slice(runner, int(out_direct_raw.shape[-1]))
        r0 = int(rot_slice.start or 0)
        r1 = int(rot_slice.stop or out_direct_raw.shape[-1])
        rot_len = int(max(0, r1 - r0))
        if rot_len <= 0 or (rot_len % 6) != 0:
            raise RuntimeError(f"invalid rot slice [{r0}:{r1}] over Dy={out_direct_raw.shape[-1]}.")
        J = rot_len // 6

        pred6 = out_direct_raw[..., r0:r1].reshape(1, int(out_direct_raw.shape[1]), J, 6)
        gt6 = gt_raw[..., r0:r1].reshape(1, int(gt_raw.shape[1]), J, 6)

        Rp = rot6d_to_matrix(pred6)
        Rg = rot6d_to_matrix(gt6)
        Rerr = torch.matmul(Rp.transpose(-1, -2), Rg)
        rotvec_rad = so3_log_map(Rerr)
        ang_rad = geodesic_R(Rp, Rg)

    root_idx = int(getattr(trainer, "eval_root_idx", 0) or 0)
    return {
        "out_direct": out_direct,
        "out_direct_raw": out_direct_raw,
        "gt": gt,
        "gt_raw": gt_raw,
        "pred6": pred6,
        "gt6": gt6,
        "rotvec_rad": rotvec_rad,
        "ang_rad": ang_rad,
        "rot_slice": (r0, r1),
        "joint_count": J,
        "time_steps": int(out_direct.shape[1]),
        "root_idx": root_idx,
    }


def _joint_index(bone_names: Sequence[str], name: str) -> int:
    target = str(name).strip()
    for i, b in enumerate(bone_names):
        if str(b) == target:
            return int(i)
    raise KeyError(f"Joint '{name}' not found.")


def _stats_1d(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p99": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90.0)),
        "p99": float(np.percentile(x, 99.0)),
    }


def _decode_sensitivity_audit(
    *,
    clip_data: Dict[str, Any],
    forward: Dict[str, Any],
    hotspot_idx: Sequence[int],
    hotspot_joint: str,
) -> Dict[str, Any]:
    bone_names = clip_data["bone_names"]
    j = _joint_index(bone_names, hotspot_joint)
    T = int(forward["time_steps"])
    hs = sorted(set(int(i) for i in hotspot_idx if 0 <= int(i) < T))
    hs_mask = np.zeros((T,), dtype=bool)
    hs_mask[hs] = True
    non_mask = ~hs_mask

    pred6 = forward["pred6"].detach().cpu().numpy()[0, :, j, :]  # (T,6)
    gt6 = forward["gt6"].detach().cpu().numpy()[0, :, j, :]      # (T,6)
    err6 = pred6 - gt6
    raw_l2 = np.linalg.norm(err6, axis=-1)

    rotvec_deg = (
        forward["rotvec_rad"].detach().cpu().numpy()[0, :, j, :] * (180.0 / math.pi)
    )
    rot_norm_deg = np.linalg.norm(rotvec_deg, axis=-1)
    amp = rot_norm_deg / (raw_l2 + 1e-8)

    a1 = pred6[:, 0:3]
    a2 = pred6[:, 3:6]
    g1 = gt6[:, 0:3]
    g2 = gt6[:, 3:6]

    def _gs(v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n1 = np.linalg.norm(v1, axis=-1)
        n2 = np.linalg.norm(v2, axis=-1)
        den = (n1 * n2) + 1e-8
        cos = np.sum(v1 * v2, axis=-1) / den
        cross = np.cross(v1, v2, axis=-1)
        sin = np.linalg.norm(cross, axis=-1) / den
        return cos, sin

    cos_p, sin_p = _gs(a1, a2)
    cos_g, sin_g = _gs(g1, g2)

    def _pack(mask: np.ndarray) -> Dict[str, Any]:
        d = err6[mask]
        return {
            "count": int(mask.sum()),
            "raw6d_err_l2": _stats_1d(raw_l2[mask]),
            "rotvec_err_deg": _stats_1d(rot_norm_deg[mask]),
            "decode_gain_deg_per_raw_l2": _stats_1d(amp[mask]),
            "raw6d_err_mean_abs_per_dim": [float(x) for x in np.mean(np.abs(d), axis=0)] if d.size else [float("nan")] * 6,
            "raw6d_err_mean_signed_per_dim": [float(x) for x in np.mean(d, axis=0)] if d.size else [float("nan")] * 6,
            "pred_cos_a1_a2": _stats_1d(cos_p[mask]),
            "pred_sin_a1xa2": _stats_1d(sin_p[mask]),
            "gt_cos_a1_a2": _stats_1d(cos_g[mask]),
            "gt_sin_a1xa2": _stats_1d(sin_g[mask]),
            "pred_near_collinear_frac_abs_cos_gt_0p95": float(np.mean(np.abs(cos_p[mask]) > 0.95)) if mask.any() else float("nan"),
        }

    hot = _pack(hs_mask)
    non = _pack(non_mask)

    gain_hot = float(hot["decode_gain_deg_per_raw_l2"]["median"])
    gain_non = float(non["decode_gain_deg_per_raw_l2"]["median"])
    raw_hot = float(hot["raw6d_err_l2"]["median"])
    raw_non = float(non["raw6d_err_l2"]["median"])
    col_hot = float(hot["pred_near_collinear_frac_abs_cos_gt_0p95"])
    col_non = float(non["pred_near_collinear_frac_abs_cos_gt_0p95"])

    inference = "prediction_bias_likely"
    if np.isfinite(gain_hot) and np.isfinite(gain_non) and np.isfinite(raw_hot) and np.isfinite(raw_non):
        gain_ratio = gain_hot / max(gain_non, 1e-8)
        raw_ratio = raw_hot / max(raw_non, 1e-8)
        if gain_ratio >= 1.25 and raw_ratio <= 1.10 and col_hot >= (col_non + 0.10):
            inference = "decode_amplification_likely"

    return {
        "clip": clip_data["clip"],
        "joint": hotspot_joint,
        "hotspot_sics": [int(x) for x in hs],
        "hotspot": hot,
        "non_hotspot": non,
        "inference": inference,
    }


def _cross_clip_bias_audit(
    *,
    clip_runs: Sequence[Dict[str, Any]],
    hotspot_idx: Sequence[int],
    joints: Sequence[str],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for run in clip_runs:
        clip = str(run["clip"])
        bone_names = run["bone_names"]
        rotvec_deg = run["rotvec_deg"]  # (T,J,3)
        T = int(rotvec_deg.shape[0])
        hs = sorted(set(int(i) for i in hotspot_idx if 0 <= int(i) < T))
        hs_mask = np.zeros((T,), dtype=bool)
        hs_mask[hs] = True
        for jn in joints:
            if jn not in bone_names:
                continue
            j = bone_names.index(jn)
            all_xyz = np.mean(rotvec_deg[:, j, :], axis=0)
            hs_xyz = np.mean(rotvec_deg[hs_mask, j, :], axis=0) if hs_mask.any() else np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            rows.append(
                {
                    "clip": clip,
                    "joint": jn,
                    "mu_all_xyz_deg": [float(x) for x in all_xyz],
                    "mu_all_norm_deg": float(np.linalg.norm(all_xyz)),
                    "mu_all_z_deg": float(all_xyz[2]),
                    "mu_hot_xyz_deg": [float(x) for x in hs_xyz],
                    "mu_hot_norm_deg": float(np.linalg.norm(hs_xyz)),
                    "mu_hot_z_deg": float(hs_xyz[2]),
                }
            )

    # Clip-level summary for calf_r hotspot z sign consistency.
    calf_rows = [r for r in rows if r["joint"] == "calf_r" and np.isfinite(float(r["mu_hot_z_deg"]))]
    neg = sum(1 for r in calf_rows if float(r["mu_hot_z_deg"]) < 0.0)
    pos = sum(1 for r in calf_rows if float(r["mu_hot_z_deg"]) > 0.0)
    med_abs = float(np.median([abs(float(r["mu_hot_z_deg"])) for r in calf_rows])) if calf_rows else float("nan")
    generality = "insufficient"
    if calf_rows:
        if neg >= max(2, int(math.ceil(0.6 * len(calf_rows)))):
            generality = "cross_clip_consistent_negative_bias"
        elif pos >= max(2, int(math.ceil(0.6 * len(calf_rows)))):
            generality = "cross_clip_consistent_positive_bias"
        else:
            generality = "clip_specific_or_mixed"

    return {
        "hotspot_sics": [int(x) for x in hotspot_idx],
        "rows": rows,
        "calf_r_hotspot_sign_summary": {
            "num_clips": int(len(calf_rows)),
            "num_negative": int(neg),
            "num_positive": int(pos),
            "median_abs_mu_hot_z_deg": med_abs,
            "inference": generality,
        },
    }


def _find_last_linear(module: torch.nn.Module) -> torch.nn.Linear:
    for m in reversed(list(module.modules())):
        if isinstance(m, torch.nn.Linear):
            return m
    raise RuntimeError("No Linear layer found in direct_pose_head.")


def _gradient_budget_audit(
    *,
    runner: FreeRunCycleRunner,
    clip_data: Dict[str, Any],
    forward_ng: Dict[str, Any],
    hotspot_idx: Sequence[int],
    transition_idx: Sequence[int],
    hotspot_joint: str,
) -> Dict[str, Any]:
    trainer = runner.trainer
    model = runner.model
    if trainer is None or model is None:
        raise RuntimeError("Runner model/trainer is not initialized.")
    model = model.to(runner.device)
    model.eval()

    sample = clip_data["sample"]
    bone_names = clip_data["bone_names"]
    j_hot = _joint_index(bone_names, hotspot_joint)
    T = int(forward_ng["time_steps"])
    J = int(forward_ng["joint_count"])
    root_idx = int(forward_ng["root_idx"])
    hs = sorted(set(int(i) for i in hotspot_idx if 0 <= int(i) < T))
    tr = sorted(set(int(i) for i in transition_idx if 0 <= int(i) < T))

    ang_rad_base = forward_ng["ang_rad"].detach().cpu().numpy()[0]  # (T,J)
    keep = np.ones((T, J), dtype=bool)
    if 0 <= root_idx < J:
        keep[:, root_idx] = False
    total_count = int(keep.sum())
    total_loss = float(ang_rad_base[keep].sum())

    hs_joint = np.zeros((T, J), dtype=bool)
    hs_joint[hs, j_hot] = True
    hs_joint &= keep

    nonhs_joint = np.zeros((T, J), dtype=bool)
    nonhs_joint[:, j_hot] = True
    nonhs_joint[hs, j_hot] = False
    nonhs_joint &= keep

    hs_allj = np.zeros((T, J), dtype=bool)
    hs_allj[hs, :] = True
    hs_allj &= keep

    tr_allj = np.zeros((T, J), dtype=bool)
    tr_allj[tr, :] = True
    tr_allj &= keep

    sets = {
        "all_non_root": keep,
        "hotspot_joint": hs_joint,
        "non_hotspot_joint": nonhs_joint,
        "hotspot_all_joints": hs_allj,
        "transition_all_joints": tr_allj,
    }

    state = _to_device(sample.get("motion"), runner.device).unsqueeze(0)
    gt = _to_device(sample.get("gt_motion"), runner.device).unsqueeze(0)
    cond = _to_device(sample.get("cond_in"), runner.device)
    cond = cond.unsqueeze(0) if cond is not None else None
    angvel = _to_device(sample.get("angvel"), runner.device)
    angvel = angvel.unsqueeze(0) if angvel is not None else None
    pose_hist = _to_device(sample.get("pose_hist"), runner.device)
    pose_hist = pose_hist.unsqueeze(0) if pose_hist is not None else None
    contacts = _to_device(sample.get("contacts"), runner.device)
    contacts = contacts.unsqueeze(0) if contacts is not None else None
    use_learned_meas = bool(getattr(model, "contact_meas_enable", False)) and getattr(model, "contact_meas_head", None) is not None
    if use_learned_meas:
        contacts = None

    r0, r1 = forward_ng["rot_slice"]
    direct_head = model.direct_pose_head
    if direct_head is None:
        raise RuntimeError("model.direct_pose_head is None.")
    last_linear = _find_last_linear(direct_head)

    rows: List[Dict[str, Any]] = []
    for name, mask_np in sets.items():
        n = int(mask_np.sum())
        if n <= 0:
            rows.append(
                {
                    "set": name,
                    "count": 0,
                    "count_share": 0.0,
                    "loss_share_vs_total": 0.0,
                    "mean_ang_deg": float("nan"),
                    "grad_last_weight_l2": float("nan"),
                    "grad_last_bias_l2": float("nan"),
                    "grad_direct_head_l2": float("nan"),
                    "grad_last_weight_l2_per_sample": float("nan"),
                }
            )
            continue

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            ret = model(
                state,
                cond=cond,
                contacts=contacts,
                angvel=angvel,
                pose_history=pose_hist,
                plan_z=None,
                time_index=None,
            )
            out_direct = ret["out_direct"]
            out_direct_raw = trainer._denorm(out_direct)
            gt_raw = trainer._denorm(gt)
            pred6 = out_direct_raw[..., r0:r1].reshape(1, T, J, 6)
            gt6 = gt_raw[..., r0:r1].reshape(1, T, J, 6)
            Rp = rot6d_to_matrix(pred6)
            Rg = rot6d_to_matrix(gt6)
            ang = geodesic_R(Rp, Rg)[0]  # (T,J), rad
            mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(device=ang.device, dtype=ang.dtype)
            loss = (ang * mask_t).sum() / mask_t.sum().clamp_min(1.0)
            loss.backward()

        gw = last_linear.weight.grad.detach()
        gb = last_linear.bias.grad.detach() if last_linear.bias is not None and last_linear.bias.grad is not None else None
        dh_sq = 0.0
        for p in direct_head.parameters():
            if p.grad is None:
                continue
            dh_sq += float(torch.sum(p.grad.detach() ** 2).item())
        grad_direct_l2 = math.sqrt(max(dh_sq, 0.0))
        grad_w = float(torch.norm(gw).item())
        grad_b = float(torch.norm(gb).item()) if gb is not None else 0.0
        loss_part = float(ang_rad_base[mask_np].sum())
        rows.append(
            {
                "set": name,
                "count": n,
                "count_share": float(n / max(total_count, 1)),
                "loss_share_vs_total": float(loss_part / max(total_loss, 1e-8)),
                "mean_ang_deg": float(np.mean(ang_rad_base[mask_np]) * (180.0 / math.pi)),
                "grad_last_weight_l2": grad_w,
                "grad_last_bias_l2": grad_b,
                "grad_direct_head_l2": grad_direct_l2,
                "grad_last_weight_l2_per_sample": float(grad_w / max(n, 1)),
            }
        )

    # Per-sic loss concentration for hotspot joint.
    sic_rows: List[Dict[str, Any]] = []
    for s in range(T):
        if not keep[s, j_hot]:
            continue
        mean_deg = float(ang_rad_base[s, j_hot] * (180.0 / math.pi))
        sic_rows.append(
            {
                "sic": int(s),
                "joint": hotspot_joint,
                "mean_ang_deg": mean_deg,
                "is_hotspot": bool(s in hs),
                "is_transition": bool(s in tr),
            }
        )
    sic_rows = sorted(sic_rows, key=lambda x: abs(float(x["mean_ang_deg"])), reverse=True)

    return {
        "clip": clip_data["clip"],
        "joint": hotspot_joint,
        "hotspot_sics": [int(x) for x in hs],
        "transition_sics": [int(x) for x in tr],
        "rows": rows,
        "hotspot_joint_sic_rank_by_mean_ang_deg": sic_rows[:20],
    }


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _write_md(
    out_path: Path,
    *,
    a: Dict[str, Any],
    b: Dict[str, Any],
    c: Dict[str, Any],
    model_path: str,
    teacher_specs: Sequence[str],
) -> None:
    lines: List[str] = []
    lines.append("# H1 10.2 Audit Run (Teacher One-Step Lens)")
    lines.append("")
    lines.append(f"- model: `{model_path}`")
    lines.append(f"- teacher specs: `{', '.join(teacher_specs)}`")
    lines.append("")
    lines.append("## 10.2.A rot6d->SO(3) decode sensitivity (Walk_F)")
    lines.append("")
    lines.append(f"- clip={a.get('clip')} joint={a.get('joint')} hotspot={a.get('hotspot_sics')}")
    lines.append(f"- inference={a.get('inference')}")
    ah = a.get("hotspot", {})
    an = a.get("non_hotspot", {})
    lines.append(
        f"- raw6d_l2 median: hotspot={ah.get('raw6d_err_l2', {}).get('median')} "
        f"vs non={an.get('raw6d_err_l2', {}).get('median')}"
    )
    lines.append(
        f"- decode_gain median: hotspot={ah.get('decode_gain_deg_per_raw_l2', {}).get('median')} "
        f"vs non={an.get('decode_gain_deg_per_raw_l2', {}).get('median')}"
    )
    lines.append(
        f"- near_collinear_frac(|cos|>0.95): hotspot={ah.get('pred_near_collinear_frac_abs_cos_gt_0p95')} "
        f"vs non={an.get('pred_near_collinear_frac_abs_cos_gt_0p95')}"
    )
    lines.append("")
    lines.append("## 10.2.B cross-clip one-step bias generalization")
    lines.append("")
    s = b.get("calf_r_hotspot_sign_summary", {})
    lines.append(
        f"- calf_r hotspot z-sign: negative={s.get('num_negative')}, positive={s.get('num_positive')}, "
        f"num_clips={s.get('num_clips')}, inference={s.get('inference')}"
    )
    lines.append("")
    lines.append("| clip | joint | mu_hot_z_deg | mu_hot_norm_deg | mu_all_z_deg |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in b.get("rows", []):
        if row.get("joint") not in ("calf_r", "thigh_r", "foot_r"):
            continue
        lines.append(
            f"| {row.get('clip')} | {row.get('joint')} | "
            f"{float(row.get('mu_hot_z_deg')):.3f} | "
            f"{float(row.get('mu_hot_norm_deg')):.3f} | "
            f"{float(row.get('mu_all_z_deg')):.3f} |"
        )
    lines.append("")
    lines.append("## 10.2.C hotspot optimization budget / gradient audit")
    lines.append("")
    lines.append("| set | count_share | loss_share | mean_ang_deg | grad_last_w_l2 | grad_last_w_l2_per_sample |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in c.get("rows", []):
        lines.append(
            f"| {row.get('set')} | "
            f"{float(row.get('count_share')):.4f} | "
            f"{float(row.get('loss_share_vs_total')):.4f} | "
            f"{float(row.get('mean_ang_deg')):.4f} | "
            f"{float(row.get('grad_last_weight_l2')):.6f} | "
            f"{float(row.get('grad_last_weight_l2_per_sample')):.8f} |"
        )
    lines.append("")
    lines.append("### calf_r SIC rank by mean angle (top 10)")
    lines.append("")
    lines.append("| rank | sic | mean_ang_deg | hotspot | transition |")
    lines.append("|---:|---:|---:|---:|---:|")
    for k, row in enumerate(c.get("hotspot_joint_sic_rank_by_mean_ang_deg", [])[:10], start=1):
        lines.append(
            f"| {k} | {int(row.get('sic'))} | {float(row.get('mean_ang_deg')):.4f} | "
            f"{str(bool(row.get('is_hotspot')))} | {str(bool(row.get('is_transition')))} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run H1 10.2 (A/B/C) audits with teacher one-step outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument(
        "--teacher",
        nargs="+",
        default=["validate/teacher_batches/*.json"],
        help="Teacher JSON files, dirs, or globs for cross-clip audit.",
    )
    ap.add_argument("--walkf-teacher", type=str, default="validate/teacher_batches/Walk_F_teacher.json")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--context-len", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--hotspot-joint", type=str, default="calf_r")
    ap.add_argument("--hotspot-sics", type=str, default="9-14,39-42")
    ap.add_argument("--transition-sics", type=str, default="14,15,49-55")
    ap.add_argument(
        "--crossclip-joints",
        type=str,
        default="calf_r,calf_l,thigh_r,thigh_l,foot_r,foot_l,ball_r,ball_l",
    )
    ap.add_argument("--out-dir", type=str, default="debug_output/h1_10p2_20260213")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    npz_root = Path(args.npz_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = _build_runner(args)

    teacher_paths = _expand_specs(args.teacher)
    if not teacher_paths:
        raise SystemExit("No teacher JSON resolved from --teacher specs.")

    hotspot_sics = _parse_index_spec(args.hotspot_sics)
    transition_sics = _parse_index_spec(args.transition_sics)
    cross_joints = [s.strip() for s in str(args.crossclip_joints).split(",") if s.strip()]

    clip_runs: List[Dict[str, Any]] = []
    walkf_audit_data: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
    walkf_teacher_abs = str(Path(args.walkf_teacher).expanduser().resolve())

    for tp in teacher_paths:
        clip_data = _load_teacher_sample(runner, teacher_path=tp, npz_root=npz_root)
        fwd = _forward_direct_error(runner, clip_data["sample"], require_grad=False)
        rotvec_deg = fwd["rotvec_rad"].detach().cpu().numpy()[0] * (180.0 / math.pi)
        clip_runs.append(
            {
                "clip": clip_data["clip"],
                "teacher_path": clip_data["teacher_path"],
                "bone_names": clip_data["bone_names"],
                "rotvec_deg": rotvec_deg,
            }
        )
        if str(Path(tp).resolve()) == walkf_teacher_abs:
            walkf_audit_data = (clip_data, fwd)

    if walkf_audit_data is None:
        # Fallback: load walkf explicitly if not present in --teacher specs.
        wtp = Path(args.walkf_teacher).expanduser().resolve()
        clip_data = _load_teacher_sample(runner, teacher_path=wtp, npz_root=npz_root)
        fwd = _forward_direct_error(runner, clip_data["sample"], require_grad=False)
        walkf_audit_data = (clip_data, fwd)

    walkf_clip, walkf_fwd_ng = walkf_audit_data

    audit_a = _decode_sensitivity_audit(
        clip_data=walkf_clip,
        forward=walkf_fwd_ng,
        hotspot_idx=hotspot_sics,
        hotspot_joint=str(args.hotspot_joint),
    )
    audit_b = _cross_clip_bias_audit(
        clip_runs=clip_runs,
        hotspot_idx=hotspot_sics,
        joints=cross_joints,
    )
    audit_c = _gradient_budget_audit(
        runner=runner,
        clip_data=walkf_clip,
        forward_ng=walkf_fwd_ng,
        hotspot_idx=hotspot_sics,
        transition_idx=transition_sics,
        hotspot_joint=str(args.hotspot_joint),
    )

    out_json = {
        "config": {
            "model": str(Path(args.model).expanduser().resolve()),
            "teacher_specs": [str(x) for x in args.teacher],
            "walkf_teacher": walkf_teacher_abs,
            "hotspot_joint": str(args.hotspot_joint),
            "hotspot_sics": hotspot_sics,
            "transition_sics": transition_sics,
            "crossclip_joints": cross_joints,
        },
        "10_2_A_decode_sensitivity": audit_a,
        "10_2_B_cross_clip_bias": audit_b,
        "10_2_C_budget_grad": audit_c,
    }

    out_json_path = out_dir / "h1_10p2_summary.json"
    out_md_path = out_dir / "h1_10p2_summary.md"
    out_json_path.write_text(json.dumps(_to_jsonable(out_json), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(
        out_md_path,
        a=audit_a,
        b=audit_b,
        c=audit_c,
        model_path=str(Path(args.model).expanduser().resolve()),
        teacher_specs=[str(x) for x in args.teacher],
    )
    print(f"[OK] wrote {out_json_path}")
    print(f"[OK] wrote {out_md_path}")


if __name__ == "__main__":
    main()
