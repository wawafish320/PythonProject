#!/usr/bin/env python3
"""Diagnose per-cond-channel rho shift under c6(speed)=0 intervention.

Definition:
- rho_i = ||d y_R / d c_i|| / ||d y_L / d c_i||
- Delta rho_i = rho_i(base) - rho_i(c6=0)

This script reconstructs one or multiple batches from freerun diag .pt artifacts,
measures rho_i on direct-head input jacobians, and reports channel-wise deltas.
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

from train.data.dataset import MotionEventDataset
from train.configuration.norm_spec import merge_norm_spec
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


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _parse_diag_pts(spec: str) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for tok in str(spec or "").split(","):
        s = tok.strip()
        if not s:
            continue
        p = Path(s).expanduser().resolve()
        if str(p) in seen:
            continue
        seen.add(str(p))
        out.append(p)
    return out


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


def _rebuild_sample_cyclic(
    *,
    ds: MotionEventDataset,
    clip_id: int,
    start: int,
    seq_len: int,
) -> Dict[str, torch.Tensor]:
    if not (0 <= int(clip_id) < len(ds.clips)):
        raise SystemExit(f"[FATAL] clip_id out of range in diag batch: {clip_id}")
    clip = ds.clips[int(clip_id)]
    clip_t = int(getattr(clip, "X", np.zeros((0,), dtype=np.float32)).shape[0])
    if clip_t <= 0:
        raise SystemExit(f"[FATAL] empty clip for clip_id={clip_id}")

    st = int(start) % max(1, clip_t)
    idx_win = (np.arange(int(seq_len), dtype=np.int64) + st) % max(1, clip_t)

    x_win = np.asarray(clip.X[idx_win], dtype=np.float32)
    y_win = np.asarray(clip.Y[idx_win], dtype=np.float32)
    c_in = np.asarray(clip.C[idx_win], dtype=np.float32)
    c_tgt = np.asarray(clip.C[(idx_win + 1) % max(1, clip_t)], dtype=np.float32)
    c_tgt_raw = c_tgt.copy()

    cond_norm_mu = None
    cond_norm_std = None
    if bool(getattr(ds, "normalize_c", True)) and c_in.shape[1] > 0:
        mu, std = ds._robust_mean_std(c_in)
        try:
            std = np.clip(np.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6), 1e-6, None)
            mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            mu = np.nan_to_num(getattr(ds, "C_mu", None), nan=0.0, posinf=0.0, neginf=0.0)
            std = np.nan_to_num(getattr(ds, "C_std", None), nan=1e-6, posinf=1e-6, neginf=1e-6)
            std = np.clip(std, 1e-6, None)
        cond_norm_mu = np.asarray(mu, dtype=np.float32).reshape(-1)
        cond_norm_std = np.asarray(std, dtype=np.float32).reshape(-1)
        c_in = (c_in - mu) / std
        c_tgt = (c_tgt - mu) / std
        np.nan_to_num(c_in, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(c_tgt, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(c_in, -6.0, 6.0, out=c_in)
        np.clip(c_tgt, -6.0, 6.0, out=c_tgt)

    out: Dict[str, torch.Tensor] = {
        "motion": torch.from_numpy(x_win).float(),
        "gt_motion": torch.from_numpy(y_win).float(),
        "clip_id": torch.tensor(int(clip_id), dtype=torch.int64),
        "start": torch.tensor(int(st), dtype=torch.int64),
        "clip_len": torch.tensor(int(clip_t), dtype=torch.int64),
    }
    if c_in.shape[1] > 0:
        out["cond_in"] = torch.from_numpy(c_in).float()
        out["cond_tgt"] = torch.from_numpy(c_tgt).float()
        out["cond_tgt_raw"] = torch.from_numpy(c_tgt_raw).float()
        if cond_norm_mu is not None and cond_norm_std is not None and cond_norm_mu.size == c_in.shape[1]:
            out["cond_norm_mu"] = torch.from_numpy(cond_norm_mu).float()
            out["cond_norm_std"] = torch.from_numpy(cond_norm_std).float()

    if getattr(clip, "contacts", None) is not None:
        out["contacts"] = torch.from_numpy(np.asarray(clip.contacts[idx_win], dtype=np.float32)).float()
    else:
        out["contacts"] = torch.zeros((int(seq_len), int(getattr(ds, "contact_dim", 0))), dtype=torch.float32)
    if getattr(clip, "angvel_norm", None) is not None:
        out["angvel"] = torch.from_numpy(np.asarray(clip.angvel_norm[idx_win], dtype=np.float32)).float()
    else:
        out["angvel"] = torch.zeros((int(seq_len), int(getattr(ds, "angvel_dim", 0))), dtype=torch.float32)
    if getattr(clip, "angvel_raw", None) is not None:
        out["angvel_raw"] = torch.from_numpy(np.asarray(clip.angvel_raw[idx_win], dtype=np.float32)).float()
    if getattr(clip, "pose_hist_norm", None) is not None:
        out["pose_hist"] = torch.from_numpy(np.asarray(clip.pose_hist_norm[idx_win], dtype=np.float32)).float()
    else:
        out["pose_hist"] = torch.zeros((int(seq_len), int(getattr(ds, "pose_hist_dim", 0))), dtype=torch.float32)
    return out


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

    norm_spec = merge_norm_spec(bundle.resolve(), pretrain_template.resolve(), pretrain_keys=None, strict=True)
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
    missing = [(cid, st) for cid, st in zip(clip_ids, starts) if pair_to_idx.get((cid, st)) is None]
    samples: List[Dict[str, torch.Tensor]] = []
    if missing:
        print(
            f"[WARN] missing (clip_id,start) pairs under sliding index; "
            f"fallback to cyclic reconstruction for this batch. missing_head={missing[:8]}"
        )
        for cid, st in zip(clip_ids, starts):
            samples.append(
                _rebuild_sample_cyclic(
                    ds=ds,
                    clip_id=int(cid),
                    start=int(st),
                    seq_len=int(seq_len),
                )
            )
    else:
        for cid, st in zip(clip_ids, starts):
            idx = pair_to_idx.get((cid, st))
            if idx is None:
                continue
            samples.append(ds[idx])
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


def _measure_rho(
    *,
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cond_channels: List[int],
    speed_channel: int,
    zero_speed: bool,
    steps_spec: str,
    left_slices: List[slice],
    right_slices: List[slice],
) -> Dict[str, Any]:
    first_linear = _find_first_linear(getattr(model, "direct_pose_head", None))
    if first_linear is None:
        raise SystemExit("[FATAL] cannot find first linear in direct_pose_head.")

    state = batch.get("motion")
    cond = batch.get("cond_in")
    contacts = batch.get("contacts")
    angvel = batch.get("angvel")
    pose_hist = batch.get("pose_hist")
    if not torch.is_tensor(state):
        raise SystemExit("[FATAL] reconstructed batch missing motion.")
    if not torch.is_tensor(cond):
        raise SystemExit("[FATAL] reconstructed batch missing cond_in.")

    cond_in = cond.clone()
    if bool(zero_speed):
        cond_in[..., int(speed_channel)] = 0.0

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
                cond=cond_in,
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

            steps = _parse_steps(steps_spec, int(out_direct.shape[1]))
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

                row: Dict[str, Any] = {
                    "step": int(t),
                    "grad_ratio_r_over_l_all": _safe_ratio(
                        float(g_right.norm().detach().cpu()),
                        float(g_left.norm().detach().cpu()),
                    ),
                    "channels": {},
                }
                ch_payload = row["channels"]
                for ch in cond_channels:
                    l_ch = float(g_left[:, int(ch)].norm().detach().cpu())
                    r_ch = float(g_right[:, int(ch)].norm().detach().cpu())
                    ch_payload[str(int(ch))] = {
                        "left": l_ch,
                        "right": r_ch,
                        "rho": _safe_ratio(r_ch, l_ch),
                    }
                rows.append(row)
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)

    agg: Dict[str, Any] = {}
    for ch in cond_channels:
        vals = np.asarray(
            [_safe_float(r.get("channels", {}).get(str(int(ch)), {}).get("rho", float("nan"))) for r in rows],
            dtype=np.float64,
        )
        vals = vals[np.isfinite(vals)]
        agg[str(int(ch))] = {
            "n": int(vals.size),
            "rho_mean": float(vals.mean()) if vals.size > 0 else float("nan"),
            "rho_std": float(vals.std()) if vals.size > 0 else float("nan"),
        }

    arr_all = np.asarray([_safe_float(r.get("grad_ratio_r_over_l_all", float("nan"))) for r in rows], dtype=np.float64)
    arr_all = arr_all[np.isfinite(arr_all)]

    return {
        "zero_speed": bool(zero_speed),
        "steps": [int(r["step"]) for r in rows],
        "per_step": rows,
        "aggregate": {
            "ratio_r_over_l_all_mean": float(arr_all.mean()) if arr_all.size > 0 else float("nan"),
            "ratio_r_over_l_all_std": float(arr_all.std()) if arr_all.size > 0 else float("nan"),
            "ratio_r_over_l_all_n": int(arr_all.size),
            "rho_by_channel": agg,
            "direct_head_input_shape": list(capture.get("x").shape) if torch.is_tensor(capture.get("x")) else [],
        },
    }


def _channel_label(ch: int, speed_channel: int) -> str:
    if ch == int(speed_channel):
        return f"c{ch}(speed)"
    if ch in (4, 5):
        return f"c{ch}(dir)"
    if ch in (0, 1, 2, 3):
        return f"c{ch}(action)"
    return f"c{ch}"


def _build_markdown(summary: Dict[str, Any]) -> str:
    speed_ch = int(summary.get("speed_channel", 6))
    channels = [int(x) for x in summary.get("channels", [])]
    lines: List[str] = []
    lines.append("# Cond Rho Delta Diagnosis")
    lines.append("")
    lines.append(f"- model: `{summary.get('model', '')}`")
    lines.append(f"- diag batches: `{len(summary.get('diag_pts', []))}`")
    lines.append(f"- speed channel: `c{speed_ch}`")
    lines.append(f"- steps: `{summary.get('steps', [])}`")
    lines.append("")

    rows = summary.get("aggregate_by_channel", {}) if isinstance(summary.get("aggregate_by_channel"), dict) else {}
    if rows:
        lines.append("## Channel-wise rho and Delta rho")
        lines.append("")
        lines.append("|channel|rho(base)|rho(c6=0)|Delta rho|n_batches|")
        lines.append("|:--|--:|--:|--:|--:|")
        for ch in channels:
            obj = rows.get(str(int(ch)), {}) if isinstance(rows.get(str(int(ch))), dict) else {}
            rb = _safe_float(obj.get("rho_base_mean", float("nan")))
            rz = _safe_float(obj.get("rho_c6_zero_mean", float("nan")))
            dr = _safe_float(obj.get("delta_rho_mean", float("nan")))
            nb = int(obj.get("n_batches", 0) or 0)
            lines.append(f"|`{_channel_label(ch, speed_ch)}`|{rb:.6f}|{rz:.6f}|{dr:+.6f}|{nb}|")
        lines.append("")

    rank = summary.get("delta_rank_desc", []) if isinstance(summary.get("delta_rank_desc"), list) else []
    if rank:
        lines.append("## Delta ranking (desc)")
        lines.append("")
        for item in rank:
            ch = int(item.get("channel", -1))
            dr = _safe_float(item.get("delta_rho_mean", float("nan")))
            lines.append(f"- `{_channel_label(ch, speed_ch)}`: `{dr:+.6f}`")
        lines.append("")

    sign_rows = summary.get("sign_stability_by_channel", {}) if isinstance(summary.get("sign_stability_by_channel"), dict) else {}
    if sign_rows:
        lines.append("## Sign stability")
        lines.append("")
        lines.append("|channel|pos|neg|zero|neg_rate|pos_rate|sign_test_p_two_sided|")
        lines.append("|:--|--:|--:|--:|--:|--:|--:|")
        for ch in channels:
            k = str(int(ch))
            obj = sign_rows.get(k, {}) if isinstance(sign_rows.get(k), dict) else {}
            pos = int(obj.get("pos_n", 0) or 0)
            neg = int(obj.get("neg_n", 0) or 0)
            zero = int(obj.get("zero_n", 0) or 0)
            n_eff = int(obj.get("n_effective", 0) or 0)
            nr = _safe_float(obj.get("neg_rate", float("nan")))
            pr = _safe_float(obj.get("pos_rate", float("nan")))
            pv = _safe_float(obj.get("sign_test_p_two_sided", float("nan")))
            lines.append(
                f"|`{_channel_label(ch, speed_ch)}`|{pos}|{neg}|{zero}|"
                f"{nr*100:.1f}%|{pr*100:.1f}%|{pv:.4f}|"
            )
        lines.append("")

    grp = summary.get("group_summary", {}) if isinstance(summary.get("group_summary"), dict) else {}
    if grp:
        lines.append("## Group summary")
        lines.append("")
        for k in ("action", "dir"):
            g = grp.get(k, {}) if isinstance(grp.get(k), dict) else {}
            lines.append(
                f"- {k}: rho(base)={_safe_float(g.get('rho_base_mean', float('nan'))):.6f}, "
                f"rho(c6=0)={_safe_float(g.get('rho_c6_zero_mean', float('nan'))):.6f}, "
                f"Delta={_safe_float(g.get('delta_rho_mean', float('nan'))):+.6f}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnose rho_i(base) vs rho_i(c6=0) and Delta rho_i on reconstructed batches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--diag-pts", type=str, required=True, help="Comma-separated freerun_diag*.pt paths")
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
    ap.add_argument("--speed-channel", type=int, default=6)
    ap.add_argument("--channels", type=str, default="all_non_speed", help="all, all_non_speed, or comma-separated ints")
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--out-md", type=str, default="")
    args = ap.parse_args()

    runner, ds_one, npz_path = _load_runner(args)
    model = runner.model
    trainer = runner.trainer
    if model is None or trainer is None:
        raise SystemExit("[FATAL] failed to load model/trainer.")
    if getattr(model, "direct_pose_head", None) is None:
        raise SystemExit("[FATAL] model has no direct_pose_head.")

    diag_pts = _parse_diag_pts(args.diag_pts)
    if not diag_pts:
        raise SystemExit("[FATAL] empty --diag-pts")
    for p in diag_pts:
        if not p.is_file():
            raise SystemExit(f"[FATAL] diag pt not found: {p}")

    left_bones = _parse_csv(args.left_bones)
    right_bones = _parse_csv(args.right_bones)
    if len(left_bones) != len(right_bones):
        raise SystemExit("[FATAL] left-bones and right-bones length mismatch.")

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

    missing = [b for b in (left_bones + right_bones) if b not in name_to_idx]
    if missing:
        raise SystemExit(f"[FATAL] unresolved bones: {missing}")
    left_slices = [_joint_rot6d_slice(rot_slice, name_to_idx[b]) for b in left_bones]
    right_slices = [_joint_rot6d_slice(rot_slice, name_to_idx[b]) for b in right_bones]

    per_batch: List[Dict[str, Any]] = []
    selected_channels: Optional[List[int]] = None
    all_steps: List[int] = []

    for diag_pt in diag_pts:
        batch = _rebuild_batch_from_diag(
            diag_pt=diag_pt,
            seq_len=int(args.seq_len),
            data_root=Path(args.data_root).expanduser().resolve(),
            bundle=Path(args.bundle).expanduser(),
            pretrain_template=Path(args.pretrain_template).expanduser(),
            device=runner.device,
        )
        cond = batch.get("cond_in")
        if not torch.is_tensor(cond):
            raise SystemExit(f"[FATAL] batch missing cond_in: {diag_pt}")
        cond_dim = int(cond.shape[-1])
        speed_ch = int(args.speed_channel)
        if not (0 <= speed_ch < cond_dim):
            raise SystemExit(f"[FATAL] speed channel out of range: {speed_ch} not in [0,{cond_dim})")

        if selected_channels is None:
            spec = str(args.channels or "").strip().lower()
            if spec in ("", "all_non_speed"):
                selected_channels = [i for i in range(cond_dim) if i != speed_ch]
            elif spec == "all":
                selected_channels = list(range(cond_dim))
            else:
                tmp: List[int] = []
                seen = set()
                for tok in str(args.channels).split(","):
                    s = tok.strip()
                    if not s:
                        continue
                    try:
                        ch = int(s)
                    except Exception:
                        continue
                    if 0 <= ch < cond_dim and ch not in seen:
                        seen.add(ch)
                        tmp.append(ch)
                selected_channels = tmp
            if not selected_channels:
                raise SystemExit("[FATAL] no valid channels selected")

        base = _measure_rho(
            model=model,
            batch=batch,
            cond_channels=selected_channels,
            speed_channel=speed_ch,
            zero_speed=False,
            steps_spec=str(args.steps),
            left_slices=left_slices,
            right_slices=right_slices,
        )
        zero = _measure_rho(
            model=model,
            batch=batch,
            cond_channels=selected_channels,
            speed_channel=speed_ch,
            zero_speed=True,
            steps_spec=str(args.steps),
            left_slices=left_slices,
            right_slices=right_slices,
        )

        all_steps = base.get("steps", []) if isinstance(base.get("steps"), list) else all_steps
        delta: Dict[str, Any] = {}
        for ch in selected_channels:
            k = str(int(ch))
            rb = _safe_float(base.get("aggregate", {}).get("rho_by_channel", {}).get(k, {}).get("rho_mean", float("nan")))
            rz = _safe_float(zero.get("aggregate", {}).get("rho_by_channel", {}).get(k, {}).get("rho_mean", float("nan")))
            delta[k] = {
                "rho_base": rb,
                "rho_c6_zero": rz,
                "delta_rho": rb - rz if (math.isfinite(rb) and math.isfinite(rz)) else float("nan"),
            }

        per_batch.append(
            {
                "diag_pt": str(diag_pt),
                "cond_dim": int(cond_dim),
                "speed_channel": int(speed_ch),
                "base": base,
                "c6_zero": zero,
                "delta": delta,
            }
        )

    if selected_channels is None:
        raise SystemExit("[FATAL] channel selection failed")

    agg_by_channel: Dict[str, Any] = {}
    for ch in selected_channels:
        k = str(int(ch))
        rb = np.asarray([_safe_float(b.get("delta", {}).get(k, {}).get("rho_base", float("nan"))) for b in per_batch], dtype=np.float64)
        rz = np.asarray([_safe_float(b.get("delta", {}).get(k, {}).get("rho_c6_zero", float("nan"))) for b in per_batch], dtype=np.float64)
        dr = np.asarray([_safe_float(b.get("delta", {}).get(k, {}).get("delta_rho", float("nan"))) for b in per_batch], dtype=np.float64)
        rb = rb[np.isfinite(rb)]
        rz = rz[np.isfinite(rz)]
        dr = dr[np.isfinite(dr)]
        agg_by_channel[k] = {
            "rho_base_mean": float(rb.mean()) if rb.size > 0 else float("nan"),
            "rho_c6_zero_mean": float(rz.mean()) if rz.size > 0 else float("nan"),
            "delta_rho_mean": float(dr.mean()) if dr.size > 0 else float("nan"),
            "delta_rho_std": float(dr.std()) if dr.size > 0 else float("nan"),
            "n_batches": int(dr.size),
        }

    rank_rows: List[Dict[str, Any]] = []
    for ch in selected_channels:
        k = str(int(ch))
        rank_rows.append(
            {
                "channel": int(ch),
                "label": _channel_label(int(ch), int(args.speed_channel)),
                "delta_rho_mean": _safe_float(agg_by_channel.get(k, {}).get("delta_rho_mean", float("nan")),),
            }
        )
    rank_rows = [r for r in rank_rows if math.isfinite(_safe_float(r.get("delta_rho_mean", float("nan"))))]
    rank_rows.sort(key=lambda r: float(r["delta_rho_mean"]), reverse=True)

    def _sign_test_two_sided(pos_n: int, neg_n: int) -> float:
        n = int(pos_n) + int(neg_n)
        if n <= 0:
            return float("nan")
        k = int(pos_n)
        # exact two-sided sign-test p-value under H0: p(pos)=0.5
        cdf_lo = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
        sf_hi = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
        p = 2.0 * min(cdf_lo, sf_hi)
        return float(min(1.0, max(0.0, p)))

    sign_stability: Dict[str, Any] = {}
    for ch in selected_channels:
        k = str(int(ch))
        vals = [
            _safe_float(b.get("delta", {}).get(k, {}).get("delta_rho", float("nan")))
            for b in per_batch
        ]
        vals = [v for v in vals if math.isfinite(v)]
        pos_n = int(sum(v > 0 for v in vals))
        neg_n = int(sum(v < 0 for v in vals))
        zero_n = int(sum(abs(v) <= 1e-12 for v in vals))
        n_eff = pos_n + neg_n
        sign_stability[k] = {
            "pos_n": pos_n,
            "neg_n": neg_n,
            "zero_n": zero_n,
            "n_total": int(len(vals)),
            "n_effective": int(n_eff),
            "pos_rate": float(pos_n / n_eff) if n_eff > 0 else float("nan"),
            "neg_rate": float(neg_n / n_eff) if n_eff > 0 else float("nan"),
            "sign_test_p_two_sided": _sign_test_two_sided(pos_n, neg_n),
        }

    def _group_stats(group_channels: List[int]) -> Dict[str, float]:
        rb = np.asarray([_safe_float(agg_by_channel.get(str(int(ch)), {}).get("rho_base_mean", float("nan"))) for ch in group_channels], dtype=np.float64)
        rz = np.asarray([_safe_float(agg_by_channel.get(str(int(ch)), {}).get("rho_c6_zero_mean", float("nan"))) for ch in group_channels], dtype=np.float64)
        dr = np.asarray([_safe_float(agg_by_channel.get(str(int(ch)), {}).get("delta_rho_mean", float("nan"))) for ch in group_channels], dtype=np.float64)
        rb = rb[np.isfinite(rb)]
        rz = rz[np.isfinite(rz)]
        dr = dr[np.isfinite(dr)]
        return {
            "rho_base_mean": float(rb.mean()) if rb.size > 0 else float("nan"),
            "rho_c6_zero_mean": float(rz.mean()) if rz.size > 0 else float("nan"),
            "delta_rho_mean": float(dr.mean()) if dr.size > 0 else float("nan"),
            "n_channels": int(dr.size),
        }

    group_summary = {
        "action": _group_stats([ch for ch in selected_channels if ch in (0, 1, 2, 3)]),
        "dir": _group_stats([ch for ch in selected_channels if ch in (4, 5)]),
    }

    out: Dict[str, Any] = {
        "model": str(Path(args.model).expanduser().resolve()),
        "npz": str(npz_path),
        "diag_pts": [str(p) for p in diag_pts],
        "steps": all_steps,
        "speed_channel": int(args.speed_channel),
        "channels": [int(ch) for ch in selected_channels],
        "aggregate_by_channel": agg_by_channel,
        "delta_rank_desc": rank_rows,
        "sign_stability_by_channel": sign_stability,
        "group_summary": group_summary,
        "per_batch": per_batch,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if str(args.out_json).strip():
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Saved] {out_json}")

    if str(args.out_md).strip():
        out_md = Path(args.out_md).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_build_markdown(out), encoding="utf-8")
        print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
