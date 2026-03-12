#!/usr/bin/env python3
"""
Diagnostic 7: Input Similarity vs Delta* Variance (KNN)
======================================================

Hypothesis:
If the same (or very similar) input x maps to a wide oracle delta* distribution, a regressor will
predict the conditional mean (regression-to-the-mean) -> systematic under-predict of large |delta*|.

We quantify conditional ambiguity by:
  - selecting a hard subset mask from a NOAPPLY freerun export
  - computing delta* (axis-oracle) per step
  - for each selected step i, compute std(delta* of its K nearest neighbors) in feature space

We report this for two "input" feature spaces:
  A) model motion/state input (`motion` tensor passed into EventMotionModel.forward)
  B) direct_pose_hinge_head input (`hinge_flat` tensor passed into direct_pose_hinge_head)

Default: disable hinge-apply during feature-capture freerun (to better match NOAPPLY trajectory),
while still running the hinge head to extract `hinge_flat`.

Example (matches your compare_hinge_apply_noapply hard subset):
  python tools/diag7_knn_input_vs_delta.py \\
    --noapply debug_output/.../Walk_F_freerun_cycles.json \\
    --direct_pose_hinge_bones calf_r --direct_pose_hinge_axis z --direct_pose_hinge_max_deg 90 \\
    --min_cycle 1 --phase_min 49 --phase_max 86 \\
    --contact_source gt --contact_index 1 --contact_value 0 --contact_thresh 0.5 \\
    --angle_thresh 20 \\
    --k 10 20 40
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from train.geometry import reproject_rot6d, rot6d_to_matrix
from train.validate.run_freerun_cycles import (
    FreeRunCycleRunner,
    _build_full_cycle_sample,
    _resolve_npz_path,
    _run_freerun_cycles,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_contact_key(src: str) -> str:
    s = str(src or "gt").strip().lower()
    if s in ("gt", "ground_truth", "groundtruth"):
        return "ContactGTPerC"
    if s in ("plan", "contacts_plan", "contact_plan"):
        return "ContactPlanPerC"
    if s in ("meas", "contacts_meas", "contact_meas"):
        return "ContactMeasPerC"
    raise ValueError(f"Unknown --contact_source={src!r} (expected gt/plan/meas)")


def _extract_ang_no_deg(noapply: Dict[str, Any], *, bone: str, branch: str) -> List[float]:
    ko = noapply.get("keybone_omega")
    if not isinstance(ko, dict):
        raise ValueError(
            "NOAPPLY JSON missing keybone_omega; rerun freerun with --export_keybone_omega --export_keybone_omega_series."
        )
    series = ko.get("series")
    if not isinstance(series, dict):
        raise ValueError("NOAPPLY JSON missing keybone_omega.series; rerun with --export_keybone_omega_series.")
    branches = series.get("branches")
    if not isinstance(branches, dict):
        raise ValueError("invalid keybone_omega.series: missing branches")
    bdat = branches.get(str(branch))
    if not isinstance(bdat, dict):
        raise ValueError(f"invalid keybone_omega.series: missing branches.{branch}")
    ang_map = bdat.get("ang_deg")
    if not isinstance(ang_map, dict):
        raise ValueError(f"invalid keybone_omega.series.branches.{branch}: missing ang_deg")
    ang = ang_map.get(str(bone))
    if not isinstance(ang, list):
        raise ValueError(f"missing ang series for bone={bone!r} under branch={branch!r}")
    return [float(x) for x in ang]


def _build_hard_mask(
    noapply: Dict[str, Any],
    *,
    bone: str,
    branch: str,
    min_cycle: int,
    phase_min: int,
    phase_max: int,
    contact_source: str,
    contact_index: int,
    contact_value: int,
    contact_thresh: float,
    angle_thresh: float,
) -> np.ndarray:
    steps = noapply.get("metrics_per_step")
    if not isinstance(steps, list) or not steps:
        raise ValueError("NOAPPLY JSON missing metrics_per_step")
    ang_no = _extract_ang_no_deg(noapply, bone=bone, branch=branch)
    if len(ang_no) < len(steps):
        raise ValueError(f"NOAPPLY ang series shorter than metrics_per_step: {len(ang_no)} vs {len(steps)}")

    contact_key = _resolve_contact_key(contact_source)
    mask = np.zeros((len(steps),), dtype=bool)
    for i, rec in enumerate(steps):
        if not isinstance(rec, dict):
            continue
        cy = int(rec.get("cycle", 0) or 0)
        ph = int(rec.get("step_in_cycle", 0) or 0)
        if cy < int(min_cycle):
            continue
        if ph < int(phase_min) or ph > int(phase_max):
            continue

        c = rec.get(contact_key)
        if not (isinstance(c, list) and 0 <= int(contact_index) < len(c)):
            continue
        cv = float(c[int(contact_index)])
        thr = float(contact_thresh)
        if int(contact_value) == 1:
            ok_contact = cv >= thr
        else:
            ok_contact = cv < thr
        if not ok_contact:
            continue

        if float(ang_no[i]) <= float(angle_thresh):
            continue

        mask[i] = True
    return mask


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return (x - mu) / sd


def _knn_std(
    feats: np.ndarray,
    y: np.ndarray,
    *,
    ks: Sequence[int],
    metric: str,
) -> Dict[int, Dict[str, Any]]:
    feats = np.asarray(feats, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    assert feats.shape[0] == y.shape[0]
    n = feats.shape[0]
    if n <= 1:
        return {int(k): {"n": int(n)} for k in ks}

    if metric == "cos":
        fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        dmat = 1.0 - (fn @ fn.T)
    else:
        # squared L2 (monotonic for neighbor selection)
        s = np.sum(feats * feats, axis=1, keepdims=True)
        dmat = s + s.T - 2.0 * (feats @ feats.T)
        dmat = np.maximum(dmat, 0.0)

    out: Dict[int, Dict[str, Any]] = {}
    for k in ks:
        k = int(k)
        kk = min(max(1, k), n - 1)
        stds = np.zeros((n,), dtype=np.float64)
        stds_abs = np.zeros((n,), dtype=np.float64)
        for i in range(n):
            d = dmat[i].copy()
            d[i] = float("inf")
            nn = np.argpartition(d, kk)[:kk]
            vals = y[nn]
            stds[i] = float(vals.std())
            stds_abs[i] = float(np.abs(vals).std())
        out[k] = {
            "k_eff": int(kk),
            "knn_std_deg": stds,
            "knn_std_abs_deg": stds_abs,
            "std_mean": float(stds.mean()),
            "std_p50": float(np.percentile(stds, 50)),
            "std_p90": float(np.percentile(stds, 90)),
            "std_abs_mean": float(stds_abs.mean()),
            "std_abs_p50": float(np.percentile(stds_abs, 50)),
            "std_abs_p90": float(np.percentile(stds_abs, 90)),
        }
    return out


def _compute_axis_oracle_delta_deg(
    *,
    trainer,
    direct_norm_all: np.ndarray,  # (T,Dy)
    gt_norm_all: np.ndarray,  # (T,Dy)
    hinge_joint_idx: List[int],
    axis: str,
) -> np.ndarray:
    axis = str(axis).strip().upper()
    axis_i = {"X": 0, "Y": 1, "Z": 2}.get(axis, 2)

    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        raise RuntimeError("trainer has no rot6d_y_slice/rot6d_slice")

    cols = getattr(getattr(trainer, "loss_fn", None), "_rot6d_columns", ("X", "Z"))
    cols = tuple(cols) if isinstance(cols, (list, tuple)) and len(cols) >= 2 else ("X", "Z")

    max_rad = getattr(trainer, "direct_pose_hinge_max_rad", None)
    try:
        max_rad = float(max_rad) if max_rad is not None else None
    except Exception:
        max_rad = None

    T = int(direct_norm_all.shape[0])
    rad2deg = 180.0 / math.pi

    device = trainer.device
    dtype = next(trainer.model.parameters()).dtype if getattr(trainer, "model", None) is not None else torch.float32

    direct_norm_t = torch.from_numpy(np.asarray(direct_norm_all, dtype=np.float32)).to(device=device, dtype=dtype)
    gt_norm_t = torch.from_numpy(np.asarray(gt_norm_all, dtype=np.float32)).to(device=device, dtype=dtype)
    with torch.no_grad():
        direct_raw = trainer._denorm(direct_norm_t)
        gt_raw = trainer._denorm(gt_norm_t)

        rot_len = int(rot_slice.stop - rot_slice.start)
        if rot_len <= 0 or (rot_len % 6) != 0:
            raise RuntimeError("invalid rot6d slice length")
        J = int(rot_len // 6)
        if not hinge_joint_idx or max(hinge_joint_idx) >= J:
            raise RuntimeError(f"invalid hinge_joint_idx={hinge_joint_idx} for J={J}")

        base6 = reproject_rot6d(direct_raw[..., rot_slice]).view(T, J, 6)
        gt6 = reproject_rot6d(gt_raw[..., rot_slice]).view(T, J, 6)
        R_base = rot6d_to_matrix(base6, columns=cols)
        R_gt = rot6d_to_matrix(gt6, columns=cols)
        R_err = torch.matmul(R_base.transpose(-1, -2), R_gt)  # (T,J,3,3)
        R_h = R_err[:, hinge_joint_idx]  # (T,K,3,3)

        if axis_i == 0:  # X
            delta = torch.atan2(R_h[..., 2, 1] - R_h[..., 1, 2], R_h[..., 1, 1] + R_h[..., 2, 2])
        elif axis_i == 1:  # Y
            delta = torch.atan2(R_h[..., 0, 2] - R_h[..., 2, 0], R_h[..., 0, 0] + R_h[..., 2, 2])
        else:  # Z
            delta = torch.atan2(R_h[..., 1, 0] - R_h[..., 0, 1], R_h[..., 0, 0] + R_h[..., 1, 1])

        if max_rad is not None and max_rad > 0.0 and math.isfinite(max_rad):
            delta = delta.clamp(-max_rad, max_rad)

        delta = delta.mean(dim=-1)  # (T,)
        out_deg = delta.detach().cpu().to(torch.float32).numpy() * rad2deg

    return out_deg.astype(np.float64)


def _tensor_to_1d_cpu(x: torch.Tensor) -> np.ndarray:
    v = x.detach()
    if v.ndim == 3 and v.size(1) == 1:
        v = v[:, 0]
    if v.ndim == 2:
        v = v.mean(dim=0)
    elif v.ndim > 2:
        v = v.reshape(-1)
    return v.to(device="cpu", dtype=torch.float32).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noapply", type=str, required=True, help="NOAPPLY freerun_cycles.json (defines the hard subset mask).")
    ap.add_argument("--model", type=str, default=None, help="Checkpoint path (default: model from NOAPPLY JSON).")
    ap.add_argument("--teacher", type=str, default=None, help="Teacher JSON path (default: teacher_json from NOAPPLY).")
    ap.add_argument("--bundle", type=str, default=None, help="norm_template.json (default: bundle from NOAPPLY).")
    ap.add_argument("--pretrain_template", type=str, default=None, help="pretrain_template.json (default: from NOAPPLY).")
    ap.add_argument("--encoder_bundle", type=str, default=None, help="motion encoder bundle (default: from NOAPPLY).")
    ap.add_argument("--npz_root", type=str, default="raw_data/processed_data", help="Root dir for processed NPZ files.")
    ap.add_argument("--device", type=str, default="auto", help="cpu|cuda|mps|auto")
    ap.add_argument("--rounds", type=int, default=None, help="Rounds (default: rounds from NOAPPLY).")
    ap.add_argument("--time_index_mode", type=str, default=None, help="none|global|cycle|auto (default: from NOAPPLY).")
    ap.add_argument(
        "--time_index_cycle_minus1",
        action="store_true",
        help="Match posttrain cycle_len=T-1 when time_index_mode=cycle/auto.",
    )

    # Hinge config for feature extraction run (match your APPLY settings)
    ap.add_argument("--direct_pose_hinge_bones", type=str, default="calf_r")
    ap.add_argument("--direct_pose_hinge_axis", type=str, default="z")
    ap.add_argument("--direct_pose_hinge_max_deg", type=float, default=90.0)
    ap.add_argument(
        "--hinge_apply",
        action="store_true",
        help="Enable hinge-apply during feature capture (default: disabled via trainer monkeypatch).",
    )

    # Hard subset definition (must match compare_hinge_apply_noapply.py usage)
    ap.add_argument("--bone", type=str, default="calf_r")
    ap.add_argument("--branch", type=str, default="direct")
    ap.add_argument("--min_cycle", type=int, default=1)
    ap.add_argument("--phase_min", type=int, default=49)
    ap.add_argument("--phase_max", type=int, default=86)
    ap.add_argument("--contact_source", type=str, default="gt", choices=("gt", "plan", "meas"))
    ap.add_argument("--contact_index", type=int, default=1)
    ap.add_argument("--contact_value", type=int, default=0, choices=(0, 1))
    ap.add_argument("--contact_thresh", type=float, default=0.5)
    ap.add_argument("--angle_thresh", type=float, default=20.0)

    ap.add_argument("--k", type=int, nargs="+", default=[10], help="K values for KNN std computation.")
    ap.add_argument("--metric", type=str, default="l2", choices=("l2", "cos"), help="Distance metric for KNN.")
    ap.add_argument("--save_npz", type=str, default=None, help="Optional output NPZ path to save features/delta/mask.")

    args = ap.parse_args()

    noapply_path = Path(args.noapply).expanduser().resolve()
    noapply = _load_json(noapply_path)

    model_path = Path(args.model or noapply.get("model") or "").expanduser().resolve()
    teacher_path = Path(args.teacher or noapply.get("teacher_json") or "").expanduser().resolve()
    bundle_path = Path(args.bundle or noapply.get("bundle") or "").expanduser().resolve()
    pretrain_path = Path(args.pretrain_template or noapply.get("pretrain_template") or "").expanduser().resolve()
    encoder_bundle = args.encoder_bundle or noapply.get("encoder_bundle")
    encoder_bundle_path = Path(encoder_bundle).expanduser().resolve() if encoder_bundle else None

    if not model_path.is_file():
        raise SystemExit(f"[FATAL] model not found: {model_path}")
    if not teacher_path.is_file():
        raise SystemExit(f"[FATAL] teacher not found: {teacher_path}")
    if not bundle_path.is_file():
        raise SystemExit(f"[FATAL] bundle not found: {bundle_path}")
    if not pretrain_path.is_file():
        raise SystemExit(f"[FATAL] pretrain_template not found: {pretrain_path}")
    if encoder_bundle_path is not None and not encoder_bundle_path.is_file():
        print(f"[WARN] encoder_bundle missing: {encoder_bundle_path} (continue without)")
        encoder_bundle_path = None

    rounds = int(args.rounds if args.rounds is not None else int(noapply.get("rounds", 1) or 1))
    time_index_mode = str(args.time_index_mode or noapply.get("time_index_mode") or "auto")
    time_index_cycle_minus1 = bool(args.time_index_cycle_minus1 or bool(noapply.get("time_index_cycle_minus1", False)))

    hard_mask = _build_hard_mask(
        noapply,
        bone=str(args.bone),
        branch=str(args.branch),
        min_cycle=int(args.min_cycle),
        phase_min=int(args.phase_min),
        phase_max=int(args.phase_max),
        contact_source=str(args.contact_source),
        contact_index=int(args.contact_index),
        contact_value=int(args.contact_value),
        contact_thresh=float(args.contact_thresh),
        angle_thresh=float(args.angle_thresh),
    )
    idx_sel = np.nonzero(hard_mask)[0].astype(int)
    print(f"[Diag7] hard subset: N={len(idx_sel)} / T={len(hard_mask)}")
    if len(idx_sel) <= 1:
        raise SystemExit("[FATAL] hard subset too small for KNN diagnostic.")

    # --- Build a FreeRunCycleRunner (reuse runtime config; override hinge on) ---
    ns = argparse.Namespace()
    # Required
    ns.model = str(model_path)
    ns.device = str(args.device)
    ns.bundle = str(bundle_path)
    ns.pretrain_template = str(pretrain_path)
    ns.encoder_bundle = str(encoder_bundle_path) if encoder_bundle_path is not None else None
    # Model hyperparams (must match ckpt; prefer ckpt.posttrain_cfg over defaults).
    ckpt_cfg: Dict[str, Any] = {}
    try:
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and isinstance(ckpt.get("posttrain_cfg"), dict):
            ckpt_cfg = dict(ckpt["posttrain_cfg"])
    except Exception as e:
        print(f"[WARN] failed to read ckpt posttrain_cfg from {model_path}: {e} (fallback to defaults)")

    ns.depth = int(ckpt_cfg.get("depth", noapply.get("depth", 2) or 2) or 2)
    ns.num_heads = int(ckpt_cfg.get("num_heads", noapply.get("num_heads", 4) or 4) or 4)
    ns.dropout = float(ckpt_cfg.get("dropout", noapply.get("dropout", 0.1) or 0.1) or 0.1)
    ns.context_len = int(ckpt_cfg.get("context_len", noapply.get("context_len", 16) or 16) or 16)
    # Match runtime toggles from NOAPPLY (when present)
    for key in (
        "event_clock",
        "event_clock_max_delta",
        "event_clock_hidden_dim",
        "event_clock_gate_hidden_dim",
        "contact_plan_inject_scale",
        "contact_plan_time_bias_scale",
        "contact_meas_gate_by_hit",
        "contact_meas_vxy_mode",
        "contact_meas_ground_z_mode",
        "contact_meas_ground_z_beta",
        "contact_meas_ground_z_window",
        "contact_meas_ground_z_quantile",
        "contact_meas_ground_z_slew_up_cm",
        "contact_meas_ground_z_slew_down_cm",
        "contact_plan_init_mode",
        "contact_plan_init_hidden",
        "contact_plan_init_dropout",
        "direct_pose_meas_force_zero",
        "direct_pose_meas_source",
        "direct_pose_meas_warmup_steps",
        "direct_pose_plan_source",
        "contacts_meas_source",
        "phase_reset_source",
        "ttc_event_kind",
        "ttc_max",
        "ttc_update_alpha",
        "so3_corr_apply",
        "so3_corr_max_deg",
        "so3_corr_gate_force",
        "so3_corr_gate_from_contacts_err",
        "so3_corr_gate_from_contacts_err_mode",
        "so3_corr_gate_err_k",
        "so3_corr_gate_err_bias",
        "so3_corr_gate_err_max",
        "so3_corr_gate_err_ref_steps",
        "so3_corr_gate_err_margin",
        "so3_corr_gate_err_use_ref",
        "so3_corr_gate_scale_max",
        "contact_phase_state_event_min_interval",
    ):
        if key in noapply:
            setattr(ns, key, noapply.get(key))

    # Hinge enabled for feature capture
    ns.direct_pose_hinge_enable = True
    ns.direct_pose_hinge_bones = str(args.direct_pose_hinge_bones)
    ns.direct_pose_hinge_axis = str(args.direct_pose_hinge_axis)
    ns.direct_pose_hinge_max_deg = float(args.direct_pose_hinge_max_deg)
    ns.direct_pose_hinge_hidden = int(noapply.get("direct_pose_hinge_hidden", 0) or 0)
    ns.direct_pose_hinge_oracle_delta = False

    runner = FreeRunCycleRunner(ns)
    teacher = _load_json(teacher_path)
    clip_name = str(teacher.get("clip") or teacher_path.stem.replace("_teacher", ""))
    npz_root = Path(args.npz_root).expanduser().resolve()
    npz_path = _resolve_npz_path(clip_name, teacher.get("source_json"), npz_root)

    ds = runner._build_dataset(npz_path, seq_len=int(noapply.get("cycle_len", 0) or 87))
    runner._ensure_model_ready(ds)
    assert runner.model is not None and runner.trainer is not None

    # Optionally disable hinge apply (keep hinge head running to expose hinge_flat).
    orig_apply = None
    if not bool(args.hinge_apply):
        if hasattr(runner.trainer, "_apply_direct_hinge_correction_norm"):
            orig_apply = runner.trainer._apply_direct_hinge_correction_norm

            def _no_apply(self, y_norm, delta):  # noqa: ANN001
                return y_norm

            runner.trainer._apply_direct_hinge_correction_norm = types.MethodType(_no_apply, runner.trainer)

    clip = ds.clips[0]
    T_base = int(clip.X.shape[0] - 1) if hasattr(clip, "X") else int(noapply.get("cycle_len", 87))
    base_sample = _build_full_cycle_sample(ds, clip, seq_len=T_base)

    # --- Capture features via hooks ---
    state_feats: List[np.ndarray] = []
    direct_norms: List[np.ndarray] = []
    hinge_in_feats: List[np.ndarray] = []

    def _model_pre_hook(_mod, inputs):
        if not inputs:
            return
        x = inputs[0]
        if torch.is_tensor(x):
            state_feats.append(_tensor_to_1d_cpu(x))

    def _direct_head_hook(_mod, _inp, out):
        if torch.is_tensor(out):
            direct_norms.append(_tensor_to_1d_cpu(out))

    def _hinge_pre_hook(_mod, inputs):
        if not inputs:
            return
        x = inputs[0]
        if torch.is_tensor(x):
            hinge_in_feats.append(_tensor_to_1d_cpu(x))

    h_pre = runner.model.register_forward_pre_hook(_model_pre_hook)
    h_dir = runner.model.direct_pose_head.register_forward_hook(_direct_head_hook) if getattr(runner.model, "direct_pose_head", None) else None
    h_hinge = (
        runner.model.direct_pose_hinge_head.register_forward_pre_hook(_hinge_pre_hook)
        if getattr(runner.model, "direct_pose_hinge_head", None)
        else None
    )

    # Run freerun once (we only need per-step bookkeeping + hooks)
    _metrics_per_round, per_step, _extra = _run_freerun_cycles(
        trainer=runner.trainer,
        sample=base_sample,
        rounds=rounds,
        device=runner.device,
        time_index_mode=time_index_mode,
        time_index_cycle_minus1=time_index_cycle_minus1,
        lambda_fusion_apply=False,
        export_joint_geolocal=False,
        export_keybone_omega=False,
        export_keybone_omega_series=False,
        export_plan_state_series=False,
        export_direct_hinge_series=False,
        direct_pose_hinge_oracle_delta=False,
        export_keybone_state_series=False,
        direct_align_inc0=bool(noapply.get("direct_align_inc0", False)),
        multicycle_sync_state_on_cycle_start=bool(noapply.get("multicycle_sync_state_on_cycle_start", False)),
        multicycle_reset_plan_z_on_cycle_start=bool(noapply.get("multicycle_reset_plan_z_on_cycle_start", False)),
        freerun_x_gt_except_rot6d=bool(noapply.get("freerun_x_gt_except_rot6d", False)),
        pose_hist_source=str(noapply.get("pose_hist_source", "buffer") or "buffer"),
        pose_hist_update_source=str(noapply.get("pose_hist_update_source", "pred") or "pred"),
        cond_reprojection=str(noapply.get("cond_reprojection", "auto") or "auto"),
        analyze_phase_shift=False,
        phase_shift_max=None,
        debug_so3_corr=False,
        debug_rot_gain=False,
        debug_direct_alignment=False,
    )

    # Remove hooks
    h_pre.remove()
    if h_dir is not None:
        h_dir.remove()
    if h_hinge is not None:
        h_hinge.remove()
    if orig_apply is not None:
        runner.trainer._apply_direct_hinge_correction_norm = orig_apply

    T_run = len(per_step)
    if len(state_feats) != T_run:
        print(f"[WARN] state_feats len mismatch: {len(state_feats)} vs steps {T_run}")
    if len(direct_norms) != T_run:
        print(f"[WARN] direct_norms len mismatch: {len(direct_norms)} vs steps {T_run}")
    if len(hinge_in_feats) != T_run:
        print(f"[WARN] hinge_in_feats len mismatch: {len(hinge_in_feats)} vs steps {T_run}")

    T = min(T_run, len(hard_mask), len(state_feats), len(direct_norms), len(hinge_in_feats))
    if T <= 1:
        raise SystemExit("[FATAL] insufficient steps captured for diagnostic.")
    if T != T_run or T != len(hard_mask):
        print(f"[WARN] truncating to T={T} (run={T_run}, mask={len(hard_mask)})")

    state_all = np.stack(state_feats[:T], axis=0)
    direct_all = np.stack(direct_norms[:T], axis=0)
    hinge_in_all = np.stack(hinge_in_feats[:T], axis=0)
    hard_mask = hard_mask[:T]
    idx_sel = idx_sel[idx_sel < T]
    if len(idx_sel) <= 1:
        raise SystemExit("[FATAL] hard subset too small after truncation.")

    # Build GT tiled (normalized Y) aligned to *this run's* step_in_cycle (more robust than NOAPPLY steps).
    gt_base = base_sample.get("gt_motion")
    if not torch.is_tensor(gt_base):
        raise SystemExit("[FATAL] base_sample missing gt_motion")
    gt_base_np = gt_base.detach().cpu().to(torch.float32).numpy()  # (T_cycle,Dy)
    cycle_len = int(gt_base_np.shape[0])
    gt_all = np.zeros((T, gt_base_np.shape[1]), dtype=np.float32)
    cycle_all = np.zeros((T,), dtype=np.int32)
    step_in_cycle_all = np.zeros((T,), dtype=np.int32)
    for t in range(T):
        rec = per_step[t] if t < len(per_step) and isinstance(per_step[t], dict) else {}
        sic = rec.get("step_in_cycle", None)
        if sic is None:
            sic = int(t % cycle_len)
        try:
            sic = int(sic)
        except Exception:
            sic = int(t % cycle_len)
        sic = max(0, min(cycle_len - 1, int(sic)))
        gt_all[t] = gt_base_np[sic]
        step_in_cycle_all[t] = int(sic)

        cy = rec.get("cycle", None)
        if cy is None:
            cy = int(t // cycle_len) if cycle_len > 0 else 0
        try:
            cy = int(cy)
        except Exception:
            cy = int(t // cycle_len) if cycle_len > 0 else 0
        cycle_all[t] = int(cy)

    hinge_idx = getattr(runner.trainer, "direct_pose_hinge_joint_idx", None)
    if not (isinstance(hinge_idx, list) and hinge_idx):
        hinge_idx = getattr(runner.model, "direct_pose_hinge_joint_idx", None)
    if not (isinstance(hinge_idx, list) and hinge_idx):
        raise SystemExit("[FATAL] hinge_joint_idx not available; is the ckpt configured with direct_pose_hinge_bones?")
    hinge_idx = [int(x) for x in hinge_idx]

    delta_tgt_deg = _compute_axis_oracle_delta_deg(
        trainer=runner.trainer,
        direct_norm_all=direct_all,
        gt_norm_all=gt_all,
        hinge_joint_idx=hinge_idx,
        axis=str(args.direct_pose_hinge_axis),
    )

    # Slice hard subset arrays
    y_sel = delta_tgt_deg[idx_sel]
    x_state_sel = state_all[idx_sel]
    x_hinge_sel = hinge_in_all[idx_sel]

    # Standardize features before distance (for l2); for cos we still normalize in _knn_std.
    if str(args.metric) == "l2":
        x_state_sel = _zscore(x_state_sel)
        x_hinge_sel = _zscore(x_hinge_sel)

    ks = [int(k) for k in args.k]
    metric = str(args.metric)
    res_state = _knn_std(x_state_sel, y_sel, ks=ks, metric=metric)
    res_hinge = _knn_std(x_hinge_sel, y_sel, ks=ks, metric=metric)

    def _print_space(name: str, res: Dict[int, Dict[str, Any]]) -> None:
        print(f"\n[Diag7] Space={name}")
        for k in ks:
            rr = res.get(int(k))
            if not isinstance(rr, dict) or "knn_std_deg" not in rr:
                print(f"  k={k}: (insufficient samples)")
                continue
            stds = rr["knn_std_deg"]
            stds_abs = rr["knn_std_abs_deg"]
            frac10 = float(np.mean(stds > 10.0))
            frac3 = float(np.mean(stds < 3.0))
            frac10_abs = float(np.mean(stds_abs > 10.0))
            frac3_abs = float(np.mean(stds_abs < 3.0))
            print(
                f"  k={k} (k_eff={rr['k_eff']}): "
                f"std(mean/p50/p90)={rr['std_mean']:.3f}/{rr['std_p50']:.3f}/{rr['std_p90']:.3f} "
                f"P(std>10)={frac10:.3f} P(std<3)={frac3:.3f} | "
                f"std_abs(mean/p50/p90)={rr['std_abs_mean']:.3f}/{rr['std_abs_p50']:.3f}/{rr['std_abs_p90']:.3f} "
                f"P(std_abs>10)={frac10_abs:.3f} P(std_abs<3)={frac3_abs:.3f}"
            )

    print(
        "[Diag7] delta_tgt_deg (hard subset): "
        f"mean_abs={float(np.mean(np.abs(y_sel))):.3f} "
        f"p50_abs={float(np.percentile(np.abs(y_sel), 50)):.3f} "
        f"p90_abs={float(np.percentile(np.abs(y_sel), 90)):.3f}"
    )
    _print_space("state(motion)", res_state)
    _print_space("hinge_input(hinge_flat)", res_hinge)

    if args.save_npz:
        outp = Path(args.save_npz).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            outp,
            idx_sel=idx_sel.astype(np.int32),
            hard_mask=hard_mask.astype(np.uint8),
            delta_tgt_deg=delta_tgt_deg.astype(np.float32),
            state_feats=state_all.astype(np.float32),
            hinge_in_feats=hinge_in_all.astype(np.float32),
            cycle=cycle_all.astype(np.int32),
            step_in_cycle=step_in_cycle_all.astype(np.int32),
        )
        print(f"[Diag7] wrote {outp}")


if __name__ == "__main__":
    main()
