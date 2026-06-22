#!/usr/bin/env python3
"""Action-handoff regime bridge causal/probe audit.

Read-only, no base-model training.  This tool answers the next cheap code-level
questions for the matched seam:

* A: counterfactual per-channel input overwrite before the shared trunk;
* B: low-complexity regime-representation mapping probes;
* C: realized-motion direct/readout ablation;
* D: velocity-jump budget at the matched seam;
* E: lambda-fusion runtime behavior audit.

The probe deliberately keeps representation metrics separate from realized-motion
metrics.  High rep->rep R2 is reported only as learnability evidence, not as a
handoff success criterion.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import train.validate.run_freerun_cycles as freerun  # noqa: E402
from train import rollout_kernel as _rollout_kernel  # noqa: E402
from train.action_handoff_inbetween_cond_probe import rollout_to_egocentric  # noqa: E402
from train.action_handoff_inbetween_goal_injection import context_window_indices  # noqa: E402
from train.action_handoff_inbetween_model import GateThresholds, StateNormalizer, evaluate_rollout_state_space  # noqa: E402
from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    EGO_VEL_SLICE,
    FPS,
    GROUND_CONTACT_THR,
    GROUND_POSE_THR,
    POSE_SLICE,
    POSE_TOPK,
    RAW_COND_DIR_SLICE,
    SEAM_LEN_K,
    STATE_DIM,
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
    full_state_align,
    load_clip_states,
)
from train.geometry import fk_positions_from_rot6d  # noqa: E402
from tools.run_action_handoff_inbetween_b1_cond_baseline_probe import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_CKPT,
    DEFAULT_ENCODER_BUNDLE,
    DEFAULT_NPZ_ROOT,
    DEFAULT_PRETRAIN_TEMPLATE,
    DEFAULT_Z_FEATURES,
    _make_runner_args,
)
from tools.run_action_handoff_inbetween_reach_aware_rewire_probe import (  # noqa: E402
    _append_window,
    _next_pose_hist_norm,
    _phase_take,
)
from tools.run_action_handoff_inbetween_reach_honesty_probe import _last_contact_from_ret  # noqa: E402
from tools.run_action_handoff_matched_seam_neuron_audit import (  # noqa: E402
    ANGVEL_KEY,
    ROOT_POS_KEY,
    ROOT_VEL_KEY,
    ROT6D_KEY,
    _candidate_onset,
    _layout_slice,
    _load_clip,
    _load_npz_raw,
    _make_sample,
    _summarize_series,
)


ROOT_VEL_OUT_SLICE = slice(276, 278)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _fmt(v: Any, digits: int = 4) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "null"
    if not math.isfinite(x):
        return "null"
    return f"{x:.{digits}f}"


def _mean_finite(vals: Sequence[Any]) -> Optional[float]:
    xs: List[float] = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            xs.append(f)
    return float(np.mean(xs)) if xs else None


def _parse_clips(raw: str) -> List[str]:
    clips = [tok.strip() for tok in str(raw or "").replace(";", ",").split(",") if tok.strip()]
    valid = set(TURN_CLIPS)
    bad = [c for c in clips if c not in valid]
    if bad:
        raise ValueError(f"unsupported target clip(s): {bad}; expected subset of {sorted(valid)}")
    return clips or list(TURN_CLIPS)


def _clone_sample(sample: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {str(k): v.clone() if torch.is_tensor(v) else v for k, v in sample.items()}


def _replace_post(dst: torch.Tensor, src: torch.Tensor, sl: Optional[slice], cut: int) -> torch.Tensor:
    out = dst.clone()
    if sl is None:
        out[:, int(cut) :] = src[:, int(cut) :]
    else:
        out[:, int(cut) :, sl] = src[:, int(cut) :, sl]
    return out


def _as_series(t: torch.Tensor, *, T: int) -> Optional[np.ndarray]:
    if not torch.is_tensor(t):
        return None
    x = t.detach().float().cpu()
    if x.ndim == 0:
        return np.full((int(T), 1), float(x.item()), dtype=np.float32)
    if x.ndim >= 2 and int(x.shape[0]) == 1 and int(x.shape[1]) == int(T):
        return x[0].reshape(int(T), -1).numpy()
    if x.ndim >= 1 and int(x.shape[0]) == int(T):
        return x.reshape(int(T), -1).numpy()
    if int(x.numel()) % int(T) == 0:
        return x.reshape(int(T), -1).numpy()
    return None


@dataclass
class ForwardProbeResult:
    summary_by_signal: Dict[str, Any]
    arrays: Dict[str, np.ndarray]
    output_contract: Dict[str, Any]
    trunk_input_dim: int
    plan_feat_slice: Optional[Tuple[int, int]]


def _forward_probe(
    model: Any,
    sample: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    cut_step: int,
    topk_dims: int,
    plan_feat_reference: Optional[np.ndarray] = None,
    state_dim: int,
    cond_dim: int,
) -> ForwardProbeResult:
    """Forward one teacher-forced sample and optionally overwrite plan_feat in x.

    Inputs are rank-3 float tensors shaped:
      state [1,T,Dx], cond [1,T,Dc], contacts [1,T,C],
      angvel [1,T,138], pose_history [1,T,828] on CPU before transfer.
    """

    state = sample["state"].to(device=device)
    cond = sample["cond"].to(device=device)
    contacts = sample["contacts"].to(device=device)
    angvel = sample["angvel"].to(device=device)
    pose_history = sample["pose_history"].to(device=device)
    _, T, _ = state.shape

    captures: Dict[str, List[torch.Tensor]] = {}
    handles = []
    plan_slice: Optional[Tuple[int, int]] = None

    def _capture(name: str, tensor: torch.Tensor) -> None:
        captures.setdefault(name, []).append(tensor.detach().float().cpu())

    lin0 = getattr(model, "shared_encoder", [None])[0] if getattr(model, "shared_encoder", None) is not None else None

    if lin0 is not None:

        def _lin0_pre(_m: Any, inputs: Tuple[Any, ...]) -> Optional[Tuple[torch.Tensor]]:
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            x = inputs[0]
            x_used = x
            nonlocal plan_slice
            p0 = int(state_dim) + int(cond_dim)
            p1 = int(x.shape[-1])
            if p1 > p0:
                plan_slice = (p0, p1)
            if plan_feat_reference is not None and plan_slice is not None:
                ref = torch.as_tensor(plan_feat_reference, device=x.device, dtype=x.dtype)
                if ref.shape == x.shape:
                    x_used = x.clone()
                    x_used[:, int(cut_step) :, p0:p1] = ref[:, int(cut_step) :, p0:p1]
            _capture("shared_encoder_0_input", x_used)
            if x_used is not x:
                return (x_used,)
            return None

        handles.append(lin0.register_forward_pre_hook(_lin0_pre))
        handles.append(lin0.register_forward_hook(lambda _m, _inp, out: _capture("shared_encoder_0", out)))

    pasa_lnq = getattr(model, "_pasa_lnq", None)
    if pasa_lnq is not None:
        handles.append(
            pasa_lnq.register_forward_pre_hook(
                lambda _m, inputs: _capture("hidden_pre_pasa_lnq_input", inputs[0])
                if inputs and torch.is_tensor(inputs[0])
                else None
            )
        )

    try:
        if hasattr(model, "set_eval_runtime_controls"):
            model.set_eval_runtime_controls(debug_contact_plan_logits_decomp=True)
        with torch.no_grad():
            result = model(
                state,
                cond,
                contacts=contacts,
                angvel=angvel,
                pose_history=pose_history,
                time_index=torch.arange(T, device=device, dtype=state.dtype).view(1, T),
                rollout_step=torch.arange(T, device=device, dtype=state.dtype).view(1, T),
            )
    finally:
        for h in handles:
            h.remove()
        if hasattr(model, "_reset_eval_runtime_controls"):
            model._reset_eval_runtime_controls()

    if not isinstance(result, dict):
        raise RuntimeError("EventMotionModel.forward did not return dict")

    arrays: Dict[str, np.ndarray] = {}
    for name, vals in captures.items():
        if not vals:
            continue
        arr = _as_series(vals[-1], T=int(T))
        if arr is not None:
            arrays[name] = arr.astype(np.float32, copy=False)

    for key in ("h_final", "out", "out_direct", "contacts_plan", "lambda_fusion", "lambda_fusion_logits"):
        value = result.get(key)
        if not torch.is_tensor(value):
            continue
        arr = _as_series(value, T=int(T))
        if arr is not None:
            arrays[f"out__{key}"] = arr.astype(np.float32, copy=False)

    summaries = {
        name: _summarize_series(arr, cut=int(cut_step), topk=int(topk_dims))
        for name, arr in sorted(arrays.items())
    }
    output_contract = {
        k: {"shape": [int(x) for x in v.shape], "dtype": str(v.dtype).replace("torch.", ""), "device": str(v.device)}
        for k, v in sorted(result.items())
        if torch.is_tensor(v)
    }
    trunk_dim = int(arrays.get("shared_encoder_0_input", np.zeros((0, 0))).shape[-1])
    return ForwardProbeResult(
        summary_by_signal=summaries,
        arrays=arrays,
        output_contract=output_contract,
        trunk_input_dim=trunk_dim,
        plan_feat_slice=plan_slice,
    )


def _collapse_fraction(base: Optional[float], variant: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if base is None or variant is None or baseline is None:
        return None
    if not (math.isfinite(base) and math.isfinite(variant) and math.isfinite(baseline)):
        return None
    denom = base - baseline
    if abs(denom) <= 1e-12:
        return None
    return float((base - variant) / denom)


def _build_variant_samples(
    matched: Mapping[str, torch.Tensor],
    walk: Mapping[str, torch.Tensor],
    *,
    cut: int,
    rootpos_sl: slice,
    rootvel_sl: slice,
    rot_x_sl: slice,
    state_angvel_sl: slice,
) -> Dict[str, Dict[str, torch.Tensor]]:
    variants: Dict[str, Dict[str, torch.Tensor]] = {"matched_base": _clone_sample(matched)}

    s = _clone_sample(matched)
    s["state"] = _replace_post(s["state"], walk["state"], rootpos_sl, cut)
    variants["rootpos_to_walk"] = s

    s = _clone_sample(matched)
    s["state"] = _replace_post(s["state"], walk["state"], rootvel_sl, cut)
    variants["rootvel_to_walk"] = s

    s = _clone_sample(matched)
    s["state"] = _replace_post(s["state"], walk["state"], rootpos_sl, cut)
    s["state"] = _replace_post(s["state"], walk["state"], rootvel_sl, cut)
    variants["rootpos_rootvel_to_walk"] = s

    s = _clone_sample(matched)
    s["angvel"] = _replace_post(s["angvel"], walk["angvel"], None, cut)
    variants["side_angvel_to_walk"] = s

    s = _clone_sample(matched)
    s["state"] = _replace_post(s["state"], walk["state"], state_angvel_sl, cut)
    variants["x_state_angvel_to_walk"] = s

    s = _clone_sample(matched)
    s["state"] = _replace_post(s["state"], walk["state"], state_angvel_sl, cut)
    s["angvel"] = _replace_post(s["angvel"], walk["angvel"], None, cut)
    variants["x_state_angvel_plus_side_angvel_to_walk"] = s

    s = _clone_sample(matched)
    s["pose_history"] = _replace_post(s["pose_history"], walk["pose_history"], None, cut)
    variants["pose_history_to_walk"] = s

    s = _clone_sample(matched)
    s["state"] = _replace_post(s["state"], walk["state"], rot_x_sl, cut)
    variants["xrot6d_to_walk"] = s

    s = _clone_sample(matched)
    s["state"] = _replace_post(s["state"], walk["state"], rot_x_sl, cut)
    s["pose_history"] = _replace_post(s["pose_history"], walk["pose_history"], None, cut)
    variants["history_pose_to_walk"] = s

    s = _clone_sample(matched)
    s["contacts"] = _replace_post(s["contacts"], walk["contacts"], None, cut)
    variants["contacts_to_walk_negative_control"] = s

    s = _clone_sample(matched)
    s["state"] = _replace_post(s["state"], walk["state"], rootvel_sl, cut)
    s["state"] = _replace_post(s["state"], walk["state"], state_angvel_sl, cut)
    variants["rootvel_angvel_to_walk"] = s

    s = _clone_sample(matched)
    s["state"] = _replace_post(s["state"], walk["state"], rootvel_sl, cut)
    s["state"] = _replace_post(s["state"], walk["state"], state_angvel_sl, cut)
    s["pose_history"] = _replace_post(s["pose_history"], walk["pose_history"], None, cut)
    variants["rootvel_angvel_posehist_to_walk"] = s

    s = _clone_sample(matched)
    s["state"] = _replace_post(s["state"], walk["state"], None, cut)
    variants["state_all_to_walk"] = s

    s = _clone_sample(matched)
    state_all = _replace_post(s["state"], walk["state"], None, cut)
    state_all[:, int(cut) :, rot_x_sl] = matched["state"][:, int(cut) :, rot_x_sl]
    s["state"] = state_all
    variants["state_except_xrot6d_to_walk"] = s

    return variants


def _load_hidden_pre(z_features: Path, clips: Sequence[str]) -> Dict[str, np.ndarray]:
    with np.load(z_features, allow_pickle=True) as z:
        out: Dict[str, np.ndarray] = {}
        for clip in clips:
            key = f"{clip}__hidden_pre"
            if key not in z.files:
                raise KeyError(f"missing {key} in {z_features}")
            out[clip] = np.asarray(z[key], dtype=np.float32)
    return out


def _one_hot(index: int, n: int) -> np.ndarray:
    out = np.zeros((int(n),), dtype=np.float32)
    out[int(index)] = 1.0
    return out


def _make_mapping_pairs(
    *,
    states: Mapping[str, np.ndarray],
    hidden: Mapping[str, np.ndarray],
    target_clips: Sequence[str],
    pose_thr: float,
    contact_thr: float,
    pose_topk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    walk_state = np.asarray(states[WALK_F], dtype=np.float32)
    walk_h = np.asarray(hidden[WALK_F], dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    xs_h: List[np.ndarray] = []
    xs_state: List[np.ndarray] = []
    ys_h: List[np.ndarray] = []
    clip_to_idx = {clip: i for i, clip in enumerate(target_clips)}
    for clip in target_clips:
        t_state = np.asarray(states[clip], dtype=np.float32)
        t_h = np.asarray(hidden[clip], dtype=np.float32)
        n = int(min(t_state.shape[0], t_h.shape[0]))
        for t in range(n):
            align = full_state_align(
                walk_state,
                t_state[t],
                topk=int(pose_topk),
                contact_thr=float(contact_thr),
                pose_thr=float(pose_thr),
            )
            if align.full_state_pose_d > float(pose_thr) or align.full_state_contact_d > float(contact_thr):
                continue
            phi = int(min(align.full_state_phi, walk_h.shape[0] - 1))
            oh = _one_hot(clip_to_idx[clip], len(target_clips))
            xs_h.append(np.concatenate([walk_h[phi], oh], axis=0))
            xs_state.append(np.concatenate([walk_state[phi], oh], axis=0))
            ys_h.append(t_h[t])
            rows.append(
                {
                    "clip": clip,
                    "target_frame": int(t),
                    "walk_phi": int(phi),
                    "pose_d": float(align.full_state_pose_d),
                    "contact_d": float(align.full_state_contact_d),
                }
            )
    if not rows:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            rows,
        )
    return (
        np.stack(xs_h, axis=0).astype(np.float32),
        np.stack(xs_state, axis=0).astype(np.float32),
        np.stack(ys_h, axis=0).astype(np.float32),
        rows,
    )


def _split_indices(rows: Sequence[Mapping[str, Any]], *, seed: int, test_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    by_clip: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        by_clip.setdefault(str(row["clip"]), []).append(i)
    train: List[int] = []
    test: List[int] = []
    for _, idxs in sorted(by_clip.items()):
        arr = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(arr)
        n_test = max(1, int(round(float(test_frac) * int(arr.size)))) if int(arr.size) > 1 else 0
        test.extend([int(x) for x in arr[:n_test]])
        train.extend([int(x) for x in arr[n_test:]])
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def _standardize_train_test(
    X: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xtr_raw = np.asarray(X[train_idx], dtype=np.float64)
    Xte_raw = np.asarray(X[test_idx], dtype=np.float64)
    mu = Xtr_raw.mean(axis=0, keepdims=True)
    std = Xtr_raw.std(axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return (Xtr_raw - mu) / std, (Xte_raw - mu) / std, mu, std


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    sse = float(np.sum((yt - yp) ** 2))
    mu = np.mean(yt, axis=0, keepdims=True)
    sst = float(np.sum((yt - mu) ** 2))
    if sst <= 1e-12:
        return float("nan")
    return float(1.0 - sse / sst)


def _ridge_fit_eval(
    X: np.ndarray,
    Y: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alphas: Sequence[float],
) -> Dict[str, Any]:
    if int(train_idx.size) < 2 or int(test_idx.size) < 1:
        return {"status": "insufficient_pairs", "n_train": int(train_idx.size), "n_test": int(test_idx.size)}
    Xtr, Xte, _, _ = _standardize_train_test(X, train_idx, test_idx)
    Ytr = np.asarray(Y[train_idx], dtype=np.float64)
    Yte = np.asarray(Y[test_idx], dtype=np.float64)
    Xtr_i = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1), dtype=np.float64)], axis=1)
    Xte_i = np.concatenate([Xte, np.ones((Xte.shape[0], 1), dtype=np.float64)], axis=1)
    best: Optional[Dict[str, Any]] = None
    d = int(Xtr_i.shape[1])
    eye = np.eye(d, dtype=np.float64)
    eye[-1, -1] = 0.0
    for alpha in alphas:
        a = float(alpha)
        try:
            W = np.linalg.solve(Xtr_i.T @ Xtr_i + a * eye, Xtr_i.T @ Ytr)
        except np.linalg.LinAlgError:
            W = np.linalg.pinv(Xtr_i.T @ Xtr_i + a * eye) @ Xtr_i.T @ Ytr
        pred = Xte_i @ W
        r2 = _r2_score(Yte, pred)
        rec = {
            "alpha": a,
            "r2": float(r2),
            "rmse": float(np.sqrt(np.mean((Yte - pred) ** 2))),
            "mae": float(np.mean(np.abs(Yte - pred))),
        }
        if best is None or (math.isfinite(r2) and r2 > float(best["r2"])):
            best = rec
    assert best is not None
    per_clip: Dict[str, Dict[str, Any]] = {}
    # Refit best alpha for per-clip rows.
    a = float(best["alpha"])
    try:
        W = np.linalg.solve(Xtr_i.T @ Xtr_i + a * eye, Xtr_i.T @ Ytr)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(Xtr_i.T @ Xtr_i + a * eye) @ Xtr_i.T @ Ytr
    pred = Xte_i @ W
    for clip in sorted({str(rows[int(i)]["clip"]) for i in test_idx}):
        mask = np.asarray([str(rows[int(i)]["clip"]) == clip for i in test_idx], dtype=bool)
        per_clip[clip] = {
            "n_test": int(np.sum(mask)),
            "r2": _r2_score(Yte[mask], pred[mask]) if np.sum(mask) >= 2 else None,
            "rmse": float(np.sqrt(np.mean((Yte[mask] - pred[mask]) ** 2))) if np.sum(mask) else None,
        }
    return {
        "status": "ok",
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "input_dim": int(X.shape[1]),
        "output_dim": int(Y.shape[1]),
        "best": best,
        "per_clip_test": per_clip,
    }


def _mlp_fit_eval(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int,
    hidden: int,
    lr: float,
    seed: int,
) -> Dict[str, Any]:
    if int(epochs) <= 0:
        return {"status": "skipped", "reason": "mlp_epochs<=0"}
    if int(train_idx.size) < 4 or int(test_idx.size) < 1:
        return {"status": "insufficient_pairs", "n_train": int(train_idx.size), "n_test": int(test_idx.size)}
    torch.manual_seed(int(seed))
    Xtr, Xte, _, _ = _standardize_train_test(X, train_idx, test_idx)
    Ytr_raw = np.asarray(Y[train_idx], dtype=np.float64)
    Yte_raw = np.asarray(Y[test_idx], dtype=np.float64)
    ymu = Ytr_raw.mean(axis=0, keepdims=True)
    ystd = Ytr_raw.std(axis=0, keepdims=True)
    ystd = np.where(ystd > 1e-8, ystd, 1.0)
    Ytr = (Ytr_raw - ymu) / ystd

    device = torch.device("cpu")
    xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(Ytr, dtype=torch.float32, device=device)
    model = torch.nn.Sequential(
        torch.nn.Linear(int(X.shape[1]), int(hidden)),
        torch.nn.GELU(),
        torch.nn.Linear(int(hidden), int(hidden)),
        torch.nn.GELU(),
        torch.nn.Linear(int(hidden), int(Y.shape[1])),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        pred = model(xt)
        loss = torch.mean((pred - yt) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_std = model(torch.as_tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)
    pred = pred_std * ystd + ymu
    return {
        "status": "ok",
        "epochs": int(epochs),
        "hidden": int(hidden),
        "r2": _r2_score(Yte_raw, pred),
        "rmse": float(np.sqrt(np.mean((Yte_raw - pred) ** 2))),
        "mae": float(np.mean(np.abs(Yte_raw - pred))),
    }


def _velocity_world_from_ego(root_vel_ego: np.ndarray, cond_dir: np.ndarray) -> np.ndarray:
    rv = np.asarray(root_vel_ego, dtype=np.float64)
    cd = np.asarray(cond_dir, dtype=np.float64)
    fdir = cd / np.maximum(np.linalg.norm(cd, axis=1, keepdims=True), 1e-8)
    side = np.stack([-fdir[:, 1], fdir[:, 0]], axis=1)
    return rv[:, :1] * fdir + rv[:, 1:2] * side


def _integrate_root_pos(root_vel_ego: np.ndarray, cond_dir: np.ndarray, *, fps: float) -> np.ndarray:
    world_v = _velocity_world_from_ego(root_vel_ego, cond_dir)
    root = np.zeros((int(world_v.shape[0]), 3), dtype=np.float32)
    if int(world_v.shape[0]) > 1:
        steps = world_v[:-1] / float(fps)
        root[1:, 0:2] = np.cumsum(steps, axis=0).astype(np.float32)
    return root


def _foot_slip_summary(
    trainer: Any,
    y_raw: torch.Tensor,
    contacts: torch.Tensor,
    cond_dir: torch.Tensor,
    *,
    fps: float,
    skeleton_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    loss_fn = getattr(trainer, "loss_fn", None)
    bone_names = [str(x) for x in (getattr(loss_fn, "bone_names", None) or [])]
    parents = getattr(loss_fn, "parents", None)
    offsets = getattr(loss_fn, "bone_offsets", None)
    if (not bone_names or parents is None or not torch.is_tensor(offsets)) and isinstance(skeleton_meta, Mapping):
        try:
            bone_names = [str(x) for x in skeleton_meta.get("bone_names", [])]
            parents = [int(x) for x in skeleton_meta.get("parents", [])]
            offsets = torch.as_tensor(skeleton_meta.get("ref_local_offsets_m", []), dtype=torch.float32)
        except Exception:
            bone_names, parents, offsets = [], None, None
    if not bone_names or parents is None or not torch.is_tensor(offsets):
        return {"status": "unavailable", "reason": "missing skeleton metadata"}

    def _idx(side: str) -> Optional[int]:
        for name in (f"ball_{side}", f"toe_{side}", f"foot_{side}"):
            if name in bone_names:
                return int(bone_names.index(name))
        return None

    idx_l = _idx("l")
    idx_r = _idx("r")
    if idx_l is None and idx_r is None:
        return {"status": "unavailable", "reason": "no foot/ball/toe bones in skeleton"}

    y_np = y_raw.detach().cpu().float().numpy()
    c_np = contacts.detach().cpu().float().numpy()
    cd_np = cond_dir.detach().cpu().float().numpy()
    if y_np.ndim != 2 or y_np.shape[0] < 2:
        return {"status": "unavailable", "reason": "rollout too short"}
    rot = torch.as_tensor(y_np[:, :276].reshape(y_np.shape[0], -1, 6), dtype=torch.float32)
    root = torch.as_tensor(_integrate_root_pos(y_np[:, ROOT_VEL_OUT_SLICE], cd_np, fps=float(fps)), dtype=torch.float32)
    try:
        pos = fk_positions_from_rot6d(rot, parents, offsets.detach().cpu().float(), root_pos=root).detach().cpu().numpy()
    except Exception as exc:
        return {"status": "unavailable", "reason": f"fk failed: {type(exc).__name__}: {exc}"}

    def _side(ch_idx: int, joint_idx: Optional[int]) -> Dict[str, Any]:
        if joint_idx is None or c_np.shape[1] <= ch_idx:
            return {"n_dual_contact": 0, "mean_mps": None, "p95_mps": None, "max_mps": None}
        mask = (c_np[:-1, ch_idx] > 0.5) & (c_np[1:, ch_idx] > 0.5)
        speed = np.linalg.norm(pos[1:, joint_idx] - pos[:-1, joint_idx], axis=1) * float(fps)
        vals = speed[mask]
        if vals.size == 0:
            return {"n_dual_contact": 0, "mean_mps": None, "p95_mps": None, "max_mps": None}
        return {
            "n_dual_contact": int(vals.size),
            "mean_mps": float(np.mean(vals)),
            "p95_mps": float(np.percentile(vals, 95)),
            "max_mps": float(np.max(vals)),
        }

    # Contact order in the current freerun CLI defaults to rl:
    # contacts[0]=right, contacts[1]=left.
    right = _side(0, idx_r)
    left = _side(1, idx_l)
    vals = [v for side in (right, left) for k, v in side.items() if k.endswith("_mps") and v is not None]
    return {
        "status": "ok",
        "skeleton_source": "trainer_loss_fn" if getattr(loss_fn, "bone_offsets", None) is not None else "npz_meta_json",
        "contact_order": "rl",
        "right": right,
        "left": left,
        "mean_mps_over_sides": float(np.mean(vals)) if vals else None,
        "max_mps_over_sides": float(np.max(vals)) if vals else None,
    }


def _load_skeleton_meta(npz_root: Path, clip: str = WALK_F) -> Optional[Dict[str, Any]]:
    try:
        with np.load(npz_root / f"{clip}.npz", allow_pickle=True) as z:
            raw = z["meta_json"]
            if hasattr(raw, "item"):
                raw = raw.item()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "ignore")
            data = json.loads(raw) if isinstance(raw, str) else raw
        skel = data.get("skeleton") if isinstance(data, Mapping) else None
        return dict(skel) if isinstance(skel, Mapping) else None
    except Exception:
        return None


def _build_full_sample(runner: Any, npz_root: Path, clip: str) -> Dict[str, torch.Tensor]:
    ds = runner._build_dataset(npz_root / f"{clip}.npz", seq_len=128)
    runner._ensure_model_ready(ds)
    c = ds.clips[0]
    return freerun._build_full_cycle_sample(ds, c, seq_len=int(c.X.shape[0]))


def _take_last_step(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if not torch.is_tensor(t):
        return None
    if t.dim() == 3:
        return t[:, -1]
    if t.dim() == 2:
        return t
    if t.dim() == 1:
        return t.unsqueeze(0)
    return None


def _run_readout_rollout(
    runner: Any,
    walk_sample: Mapping[str, torch.Tensor],
    target_sample: Mapping[str, torch.Tensor],
    *,
    phi: int,
    onset: int,
    horizon: int,
    context_len: int,
    mode: str,
) -> Dict[str, Any]:
    model = runner.model
    trainer = runner.trainer
    device = runner.device
    walk_T = int(walk_sample["motion"].shape[0])
    target_T = int(target_sample["motion"].shape[0])
    C = int(context_len)
    walk_idx0 = context_window_indices(int(phi), C, walk_T, mode="wrap")
    target_idx0 = np.clip(int(onset) - C + 1 + np.arange(C, dtype=np.int64), 0, max(0, target_T - 1))

    motion_hist = _phase_take(walk_sample, "motion", walk_idx0, device)
    cond_hist = _phase_take(target_sample, "cond_in", target_idx0, device)
    contacts_hist = _phase_take(walk_sample, "contacts", walk_idx0, device)
    pose_hist = _phase_take(walk_sample, "pose_hist", walk_idx0, device)

    free_carry_cfg = _rollout_kernel.resolve_free_carry_runtime_config(trainer)
    if bool(getattr(trainer, "use_freerun_state_sync", False)) and isinstance(free_carry_cfg.angvel_x_slice, slice):
        angvel_hist = motion_hist[:, free_carry_cfg.angvel_x_slice]
    else:
        angvel_hist = _phase_take(walk_sample, "angvel", walk_idx0, device)

    y_raw_prev = trainer._denorm(walk_sample["gt_motion"][int(phi) % walk_T].to(device=device).unsqueeze(0))
    motion_raw_last = trainer.normalizer.denorm_x(motion_hist[-1:].detach())

    raw_steps: List[torch.Tensor] = []
    inc_steps: List[torch.Tensor] = []
    direct_steps: List[torch.Tensor] = []
    contact_steps: List[torch.Tensor] = []
    cond_dir_steps: List[torch.Tensor] = []
    lambda_means: List[float] = []
    used_source: List[str] = []

    with torch.no_grad():
        for step in range(int(horizon)):
            ret = model(
                motion_hist.unsqueeze(0),
                cond_hist.unsqueeze(0),
                contacts=contacts_hist.unsqueeze(0),
                angvel=angvel_hist.unsqueeze(0),
                pose_history=pose_hist.unsqueeze(0),
            )
            out = ret["out"]
            delta_norm = out[:, -1] if out.dim() == 3 else out
            try:
                y_inc_raw = trainer._compose_delta_to_raw(y_raw_prev, delta_norm)
            except Exception:
                y_inc_raw = trainer._denorm(delta_norm)

            direct_norm = _take_last_step(ret.get("out_direct"))
            direct_raw = trainer._denorm(direct_norm) if torch.is_tensor(direct_norm) else None
            lam_step = _take_last_step(ret.get("lambda_fusion"))
            if torch.is_tensor(lam_step):
                lambda_means.append(float(lam_step.detach().float().mean().item()))

            y_used_raw = y_inc_raw
            source = "main_incremental"
            if str(mode) == "lambda_model":
                if torch.is_tensor(direct_norm) and torch.is_tensor(lam_step):
                    y_used_raw = trainer._apply_lambda_fusion_to_raw(
                        y_inc_raw,
                        direct_norm=direct_norm,
                        lambda_fusion=lam_step,
                    )
                    source = "lambda_model_rot6d_blend"
            elif str(mode) == "lambda_force1":
                if torch.is_tensor(direct_norm):
                    rot_slice = getattr(trainer, "rot6d_y_slice", slice(0, 276))
                    J = int((rot_slice.stop - rot_slice.start) // 6)
                    lam_ones = torch.ones((y_inc_raw.shape[0], J), device=y_inc_raw.device, dtype=y_inc_raw.dtype)
                    y_used_raw = trainer._apply_lambda_fusion_to_raw(
                        y_inc_raw,
                        direct_norm=direct_norm,
                        lambda_fusion=lam_ones,
                    )
                    source = "lambda_force1_rot6d_blend"
            elif str(mode) == "direct_full":
                if torch.is_tensor(direct_raw):
                    y_used_raw = direct_raw
                    source = "direct_full_raw_denorm"
            elif str(mode) != "main":
                raise ValueError(f"unsupported readout mode: {mode}")

            target_idx = int(min(max(0, int(onset) + step), target_T - 1))
            next_idx = int(min(max(0, int(onset) + step + 1), target_T - 1))
            raw_steps.append(y_used_raw[0].detach())
            inc_steps.append(y_inc_raw[0].detach())
            if torch.is_tensor(direct_raw):
                direct_steps.append(direct_raw[0].detach())
            cond_dir_steps.append(
                target_sample["cond_tgt_raw"][target_idx, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]]
                .to(device=device)
                .detach()
            )
            contact_step, _ = _last_contact_from_ret(ret, contacts_hist[-1])
            contact_steps.append(contact_step.detach())
            used_source.append(source)

            cond_next_raw = target_sample["cond_tgt_raw"][next_idx].to(device=device).unsqueeze(0)
            motion_raw_next = _rollout_kernel.apply_free_carry_raw(
                x_prev=motion_raw_last.detach(),
                y_next_raw=y_used_raw.detach(),
                cond_next_raw=cond_next_raw,
                rot6d_x_slice=free_carry_cfg.rot6d_x_slice,
                rot6d_y_slice=free_carry_cfg.rot6d_y_slice,
                angvel_x_slice=free_carry_cfg.angvel_x_slice,
                rootvel_x_slice=free_carry_cfg.rootvel_x_slice,
                rootpos_x_slice=free_carry_cfg.rootpos_x_slice,
                bone_hz=free_carry_cfg.bone_hz,
                columns=free_carry_cfg.columns,
            ).detach()
            motion_next = trainer._diag_norm_x(motion_raw_next)[0].detach()
            pose_hist_next = _next_pose_hist_norm(
                trainer,
                pose_hist[-1],
                y_used_raw,
                rot_slice=free_carry_cfg.rot6d_y_slice
                if isinstance(free_carry_cfg.rot6d_y_slice, slice)
                else slice(0, y_used_raw.shape[-1]),
            )
            motion_hist = _append_window(motion_hist, motion_next)
            cond_hist = _append_window(cond_hist, target_sample["cond_in"][next_idx].to(device))
            contacts_hist = _append_window(contacts_hist, contact_step)
            pose_hist = _append_window(pose_hist, pose_hist_next)
            if bool(getattr(trainer, "use_freerun_state_sync", False)) and isinstance(free_carry_cfg.angvel_x_slice, slice):
                angvel_hist = motion_hist[:, free_carry_cfg.angvel_x_slice]
            else:
                angvel_hist = _append_window(angvel_hist, target_sample["angvel"][next_idx].to(device))
            y_raw_prev = y_used_raw.detach()
            motion_raw_last = motion_raw_next

    y_raw = torch.stack(raw_steps, dim=0)
    contacts = torch.stack(contact_steps, dim=0).clamp(0.0, 1.0)
    cond_dir = torch.stack(cond_dir_steps, dim=0)
    return {
        "mode": str(mode),
        "used_source_counts": {s: int(used_source.count(s)) for s in sorted(set(used_source))},
        "y_raw": y_raw,
        "contacts": contacts,
        "cond_dir": cond_dir,
        "lambda_mean": _mean_finite(lambda_means),
    }


def _velocity_baseline_thresholds(
    states: Mapping[str, np.ndarray],
    raw_angvel: Mapping[str, np.ndarray],
    clips: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for clip in clips:
        st = np.asarray(states[clip], dtype=np.float64)
        ego = np.linalg.norm(np.diff(st[:, EGO_VEL_SLICE], axis=0), axis=1) if st.shape[0] > 1 else np.zeros(0)
        yaw = np.abs(np.diff(st[:, YAW_RATE_SLICE].reshape(-1), axis=0)) if st.shape[0] > 1 else np.zeros(0)
        av = np.asarray(raw_angvel[clip], dtype=np.float64).reshape(raw_angvel[clip].shape[0], -1)
        avd = np.linalg.norm(np.diff(av, axis=0), axis=1) / math.sqrt(max(1, av.shape[1])) if av.shape[0] > 1 else np.zeros(0)
        out[clip] = {
            "ego_vel_delta_p95": float(np.percentile(ego, 95)) if ego.size else None,
            "yaw_rate_delta_p95": float(np.percentile(yaw, 95)) if yaw.size else None,
            "angvel_delta_rms_p95": float(np.percentile(avd, 95)) if avd.size else None,
        }
    return out


def _frames_needed(delta: float, step_p95: Optional[float]) -> Optional[int]:
    if step_p95 is None or not math.isfinite(float(step_p95)) or float(step_p95) <= 1e-12:
        return None
    return int(max(1, math.ceil(float(delta) / float(step_p95))))


def _load_raw_angvel(npz_root: Path, clips: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for clip in clips:
        with np.load(npz_root / f"{clip}.npz", allow_pickle=True) as z:
            out[clip] = np.asarray(z["bone_ang_vel"], dtype=np.float32)
    return out


def _seam_velocity_budget(
    *,
    states: Mapping[str, np.ndarray],
    raw_angvel: Mapping[str, np.ndarray],
    baselines: Mapping[str, Any],
    target: str,
    phi: int,
    onset: int,
) -> Dict[str, Any]:
    walk_state = np.asarray(states[WALK_F], dtype=np.float64)
    target_state = np.asarray(states[target], dtype=np.float64)
    w = walk_state[int(phi) % int(walk_state.shape[0])]
    q = target_state[int(onset)]
    ego_delta = q[EGO_VEL_SLICE] - w[EGO_VEL_SLICE]
    yaw_delta = float(q[YAW_RATE_SLICE][0] - w[YAW_RATE_SLICE][0])
    w_ang = np.asarray(raw_angvel[WALK_F], dtype=np.float64)[int(phi) % int(raw_angvel[WALK_F].shape[0])]
    t_ang = np.asarray(raw_angvel[target], dtype=np.float64)[int(onset)]
    ang_delta = t_ang - w_ang
    per_joint = np.linalg.norm(ang_delta.reshape(-1, 3), axis=1)
    base_walk = baselines.get(WALK_F, {})
    base_target = baselines.get(target, {})
    ego_l2 = float(np.linalg.norm(ego_delta))
    ang_rms = float(np.linalg.norm(ang_delta.reshape(-1)) / math.sqrt(max(1, ang_delta.size)))
    frames = {
        "ego_vs_walk95": _frames_needed(ego_l2, base_walk.get("ego_vel_delta_p95")),
        "ego_vs_target95": _frames_needed(ego_l2, base_target.get("ego_vel_delta_p95")),
        "yaw_vs_walk95": _frames_needed(abs(yaw_delta), base_walk.get("yaw_rate_delta_p95")),
        "yaw_vs_target95": _frames_needed(abs(yaw_delta), base_target.get("yaw_rate_delta_p95")),
        "angvel_vs_walk95": _frames_needed(ang_rms, base_walk.get("angvel_delta_rms_p95")),
        "angvel_vs_target95": _frames_needed(ang_rms, base_target.get("angvel_delta_rms_p95")),
    }
    top = np.argsort(-per_joint)[:5]
    return {
        "phi": int(phi),
        "onset": int(onset),
        "ego_vel_delta": [float(x) for x in ego_delta.tolist()],
        "ego_vel_delta_l2": ego_l2,
        "ego_vel_delta_heading_rad": float(math.atan2(float(ego_delta[1]), float(ego_delta[0]))),
        "yaw_rate_delta_rad_s": yaw_delta,
        "yaw_rate_delta_deg_s": float(yaw_delta * 180.0 / math.pi),
        "bone_angvel_delta_rms_rad_s": ang_rms,
        "bone_angvel_delta_mean_joint_rad_s": float(np.mean(per_joint)),
        "bone_angvel_delta_max_joint_rad_s": float(np.max(per_joint)),
        "top5_angvel_delta_joints": [
            {"joint_index": int(i), "delta_rad_s": float(per_joint[int(i)])}
            for i in top.tolist()
        ],
        "frames_needed_at_continuous_p95": frames,
    }


def _status_from_ablation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    candidates = []
    for row in rows:
        name = str(row.get("variant"))
        if name in {"matched_base", "walk_continuous", "target_continuous"}:
            continue
        cf = row.get("hidden_pre_collapse_fraction_to_walk")
        if cf is None:
            continue
        candidates.append((name, float(cf), row.get("hidden_pre_cut_over_pre4")))
    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0] if candidates else None
    status = "unmeasured"
    if best is not None:
        status = "established_primary_driver" if best[1] >= 0.7 else "suspect_not_collapsed"
    return {
        "status": status,
        "best_variant": best[0] if best else None,
        "best_hidden_pre_collapse_fraction_to_walk": best[1] if best else None,
        "ranking": [
            {"variant": name, "hidden_pre_collapse_fraction_to_walk": frac, "hidden_pre_cut_over_pre4": ratio}
            for name, frac, ratio in candidates
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Action-handoff regime bridge causal/probe audit.")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--bundle", type=str, default=DEFAULT_BUNDLE)
    p.add_argument("--pretrain-template", type=str, default=DEFAULT_PRETRAIN_TEMPLATE)
    p.add_argument("--encoder-bundle", type=str, default=DEFAULT_ENCODER_BUNDLE)
    p.add_argument("--npz-root", type=Path, default=Path(DEFAULT_NPZ_ROOT))
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--pre-frames", type=int, default=16)
    p.add_argument("--post-frames", type=int, default=24)
    p.add_argument("--onset-scan", type=int, default=8)
    p.add_argument("--target-clips", type=str, default=",".join(TURN_CLIPS))
    p.add_argument("--pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--ground-pose-thr", type=float, default=GROUND_POSE_THR)
    p.add_argument("--mapping-pose-thr", type=float, default=0.08)
    p.add_argument("--mapping-contact-thr", type=float, default=0.30)
    p.add_argument("--mapping-test-frac", type=float, default=0.25)
    p.add_argument("--mlp-epochs", type=int, default=250)
    p.add_argument("--mlp-hidden", type=int, default=128)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--skip-c", action="store_true", help="Skip realized-motion direct/readout rollout probe.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    npz_root = Path(args.npz_root)
    z_features = Path(args.z_features)
    target_clips = _parse_clips(args.target_clips)
    if not z_features.exists():
        raise FileNotFoundError(f"z-features not found: {z_features}")
    if not npz_root.exists():
        raise FileNotFoundError(f"npz root not found: {npz_root}")
    if not Path(args.checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_regime_bridge_probe_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = freerun.FreeRunCycleRunner(_make_runner_args(args))
    seq_len = max(64, int(args.pre_frames) + int(args.post_frames))
    walk_clip = _load_clip(runner, npz_root, WALK_F, seq_len=seq_len)
    runner.model.eval()
    model = runner.model

    norm_spec = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    mu_x = np.asarray(norm_spec["MuX"], dtype=np.float32)
    std_x = np.asarray(norm_spec["StdX"], dtype=np.float32)
    std_x = np.where(np.abs(std_x) > 1e-8, std_x, 1.0).astype(np.float32)

    walk_raw = _load_npz_raw(npz_root, WALK_F)
    rootpos_sl = _layout_slice(walk_clip.state_layout_norm, ROOT_POS_KEY, fallback=slice(0, 3))
    rootvel_sl = _layout_slice(walk_clip.state_layout_norm, ROOT_VEL_KEY, fallback=slice(3, 5))
    rot_x_sl = _layout_slice(walk_clip.state_layout_norm, ROT6D_KEY, fallback=slice(5, 281))
    state_angvel_sl = _layout_slice(walk_clip.state_layout_norm, ANGVEL_KEY, fallback=slice(281, 419))
    rot_y_sl = _layout_slice(walk_clip.output_layout_norm, ROT6D_KEY, fallback=slice(0, 276))
    _ = rootpos_sl
    dims = {
        "state": int(getattr(model, "in_state_dim", walk_clip.X.shape[1])),
        "cond": int(getattr(model, "cond_dim", walk_clip.C.shape[1])),
        "contact": int(getattr(model, "contact_dim", 0) or 0),
        "angvel": int(getattr(model, "angvel_dim", 0) or 0),
        "pose_hist": int(getattr(model, "pose_hist_dim", 0) or 0),
    }
    slices = {"rootpos_x": rootpos_sl, "rootvel_x": rootvel_sl, "rot_x": rot_x_sl, "rot_y": rot_y_sl}

    states_281 = load_clip_states(z_features, npz_root)
    hidden = _load_hidden_pre(z_features, [WALK_F, *target_clips])
    raw_angvel = _load_raw_angvel(npz_root, [WALK_F, *target_clips])
    skeleton_meta = _load_skeleton_meta(npz_root, WALK_F)

    payload: Dict[str, Any] = {
        "task": "action_handoff_regime_bridge_probe",
        "scope": "read-only causal ablation + cheap mapping + realized-motion direct/readout probe; no base training",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "z_features": str(z_features.resolve()),
        "npz_root": str(npz_root.resolve()),
        "config": {
            "pre_frames": int(args.pre_frames),
            "post_frames": int(args.post_frames),
            "cut_step": int(args.pre_frames),
            "target_clips": list(target_clips),
            "device": str(args.device),
            "mapping_pose_thr": float(args.mapping_pose_thr),
            "mapping_contact_thr": float(args.mapping_contact_thr),
            "mlp_epochs": int(args.mlp_epochs),
        },
        "code_refs": {
            "trunk_concat": "train/models.py:5020",
            "direct_feat_source": "train/models.py:5175",
            "direct_readout_concat": "train/models.py:5260",
            "eval_lambda_runtime": "train/eval_utils.py:385",
            "lambda_raw_blend": "train/training_MPL.py:855",
            "so3_lambda_direction": "train/geometry.py:702",
        },
        "input_contract": {
            "normalized_state_layout": {
                "rootpos_x": [int(rootpos_sl.start or 0), int(rootpos_sl.stop or 0)],
                "rootvel_x": [int(rootvel_sl.start or 0), int(rootvel_sl.stop or 0)],
                "rot6d_x": [int(rot_x_sl.start or 0), int(rot_x_sl.stop or 0)],
                "bone_angvel_x": [int(state_angvel_sl.start or 0), int(state_angvel_sl.stop or 0)],
            },
            "matched_sample_tensors": {
                "state": {"shape": [1, int(args.pre_frames) + int(args.post_frames), dims["state"]], "dtype": "float32"},
                "cond": {"shape": [1, int(args.pre_frames) + int(args.post_frames), dims["cond"]], "dtype": "float32"},
                "contacts": {"shape": [1, int(args.pre_frames) + int(args.post_frames), dims["contact"]], "dtype": "float32"},
                "angvel": {"shape": [1, int(args.pre_frames) + int(args.post_frames), dims["angvel"]], "dtype": "float32"},
                "pose_history": {"shape": [1, int(args.pre_frames) + int(args.post_frames), dims["pose_hist"]], "dtype": "float32"},
                "device_before_forward": "cpu",
            },
            "state_281": {"shape": ["T", STATE_DIM], "dtype": "float32", "device": "cpu"},
        },
        "model_flags": {
            "contact_plan_enable": bool(getattr(model, "contact_plan_enable", False)),
            "contact_plan_inject": str(getattr(model, "contact_plan_inject", "")),
            "direct_pose_enable": bool(getattr(model, "direct_pose_enable", False)),
            "direct_pose_feat_source": str(getattr(model, "direct_pose_feat_source", "")),
            "lambda_fusion_enable": bool(getattr(model, "lambda_fusion_enable", False)),
            "lambda_fusion_mode": str(getattr(model, "lambda_fusion_mode", "")),
        },
        "skeleton_meta_loaded_for_foot_slip": bool(skeleton_meta is not None),
        "A_channel_ablation": {"per_target": {}, "aggregate": {}},
        "B_regime_mapping": {},
        "C_direct_motion_honesty": {"per_target": {}, "aggregate": {}},
        "D_velocity_budget": {"per_target": {}, "baseline_p95": {}},
        "E_lambda_runtime": {},
        "tri_state": {},
    }

    # ---------------------------------------------------------------- A/D selected matched seams
    selected_meta: Dict[str, Dict[str, Any]] = {}
    ablation_rows_all: List[Dict[str, Any]] = []
    for target in target_clips:
        cand = _candidate_onset(
            states_281[WALK_F],
            states_281[target],
            onset_scan=int(args.onset_scan),
            pose_topk=int(args.pose_topk),
            contact_thr=float(args.ground_contact_thr),
            pose_thr=float(args.ground_pose_thr),
        )
        selected = cand.get("selected") if isinstance(cand, dict) else None
        t_payload: Dict[str, Any] = {"alignment": cand, "variants": []}
        if not selected:
            t_payload["skip_reason"] = "no groundable matched onset in scan window"
            payload["A_channel_ablation"]["per_target"][target] = t_payload
            continue

        phi = int(selected["phi"])
        onset = int(selected["onset"])
        selected_meta[target] = {"phi": phi, "onset": onset, "alignment": selected}
        target_clip = _load_clip(runner, npz_root, target, seq_len=seq_len)
        target_raw = _load_npz_raw(npz_root, target)

        matched_sample, matched_meta = _make_sample(
            case="matched_positive_xhist",
            walk_clip=walk_clip,
            target_clip=target_clip,
            walk_raw=walk_raw,
            target_raw=target_raw,
            phi=phi,
            onset=onset,
            pre=int(args.pre_frames),
            post=int(args.post_frames),
            dims=dims,
            slices=slices,
            mu_x=mu_x,
            std_x=std_x,
            norm_spec=norm_spec,
            cond_ramp_frames=8,
        )
        walk_sample, _ = _make_sample(
            case="walk_continuous",
            walk_clip=walk_clip,
            target_clip=target_clip,
            walk_raw=walk_raw,
            target_raw=target_raw,
            phi=phi,
            onset=onset,
            pre=int(args.pre_frames),
            post=int(args.post_frames),
            dims=dims,
            slices=slices,
            mu_x=mu_x,
            std_x=std_x,
            norm_spec=norm_spec,
            cond_ramp_frames=8,
        )
        target_sample_tf, _ = _make_sample(
            case="target_continuous",
            walk_clip=walk_clip,
            target_clip=target_clip,
            walk_raw=walk_raw,
            target_raw=target_raw,
            phi=phi,
            onset=onset,
            pre=int(args.pre_frames),
            post=int(args.post_frames),
            dims=dims,
            slices=slices,
            mu_x=mu_x,
            std_x=std_x,
            norm_spec=norm_spec,
            cond_ramp_frames=8,
        )

        variants = _build_variant_samples(
            matched_sample,
            walk_sample,
            cut=int(args.pre_frames),
            rootpos_sl=rootpos_sl,
            rootvel_sl=rootvel_sl,
            rot_x_sl=rot_x_sl,
            state_angvel_sl=state_angvel_sl,
        )
        walk_probe = _forward_probe(
            model,
            walk_sample,
            device=runner.device,
            cut_step=int(args.pre_frames),
            topk_dims=8,
            state_dim=dims["state"],
            cond_dim=dims["cond"],
        )
        target_probe = _forward_probe(
            model,
            target_sample_tf,
            device=runner.device,
            cut_step=int(args.pre_frames),
            topk_dims=8,
            state_dim=dims["state"],
            cond_dim=dims["cond"],
        )
        probes: Dict[str, ForwardProbeResult] = {
            "walk_continuous": walk_probe,
            "target_continuous": target_probe,
        }
        for name, sample in variants.items():
            probes[name] = _forward_probe(
                model,
                sample,
                device=runner.device,
                cut_step=int(args.pre_frames),
                topk_dims=8,
                state_dim=dims["state"],
                cond_dim=dims["cond"],
            )
        if walk_probe.arrays.get("shared_encoder_0_input") is not None:
            probes["plan_feat_to_walk"] = _forward_probe(
                model,
                matched_sample,
                device=runner.device,
                cut_step=int(args.pre_frames),
                topk_dims=8,
                plan_feat_reference=walk_probe.arrays["shared_encoder_0_input"][None, ...],
                state_dim=dims["state"],
                cond_dim=dims["cond"],
            )

        base_hidden = probes["matched_base"].summary_by_signal.get("hidden_pre_pasa_lnq_input", {}).get("cut_over_pre4")
        walk_hidden = probes["walk_continuous"].summary_by_signal.get("hidden_pre_pasa_lnq_input", {}).get("cut_over_pre4")
        base_shared = probes["matched_base"].summary_by_signal.get("shared_encoder_0", {}).get("cut_over_pre4")
        walk_shared = probes["walk_continuous"].summary_by_signal.get("shared_encoder_0", {}).get("cut_over_pre4")
        for name, pr in probes.items():
            hidden_ratio = pr.summary_by_signal.get("hidden_pre_pasa_lnq_input", {}).get("cut_over_pre4")
            shared_ratio = pr.summary_by_signal.get("shared_encoder_0", {}).get("cut_over_pre4")
            hfinal_ratio = pr.summary_by_signal.get("out__h_final", {}).get("cut_over_pre4")
            out_ratio = pr.summary_by_signal.get("out__out", {}).get("cut_over_pre4")
            row = {
                "target": target,
                "variant": name,
                "shared_encoder0_cut_over_pre4": shared_ratio,
                "hidden_pre_cut_over_pre4": hidden_ratio,
                "h_final_cut_over_pre4": hfinal_ratio,
                "out_cut_over_pre4": out_ratio,
                "shared_encoder0_collapse_fraction_to_walk": _collapse_fraction(base_shared, shared_ratio, walk_shared),
                "hidden_pre_collapse_fraction_to_walk": _collapse_fraction(base_hidden, hidden_ratio, walk_hidden),
                "plan_feat_slice_in_trunk_x": list(pr.plan_feat_slice) if pr.plan_feat_slice is not None else None,
            }
            t_payload["variants"].append(row)
            ablation_rows_all.append(row)
        t_payload["matched_meta"] = {
            "phi": phi,
            "onset": onset,
            "pose_d": float(selected["pose_d"]),
            "contact_d": float(selected["contact_d"]),
            "rootvel_norm_step_l2_at_cut": matched_meta.get("x_rootvel_norm_step_l2_at_cut"),
            "history_rot6d_step_l2_at_cut": matched_meta.get("history_rot6d_step_l2_at_cut"),
            "contact_step_l2_at_cut": matched_meta.get("contact_step_l2_at_cut"),
            "cond_step_l2_at_cut": matched_meta.get("cond_step_l2_at_cut"),
        }
        payload["A_channel_ablation"]["per_target"][target] = t_payload

    payload["A_channel_ablation"]["aggregate"] = {
        "n_rows": int(len(ablation_rows_all)),
        "mean_by_variant": {},
        "status": _status_from_ablation(ablation_rows_all),
    }
    for variant in sorted({str(r["variant"]) for r in ablation_rows_all}):
        rows = [r for r in ablation_rows_all if str(r["variant"]) == variant]
        payload["A_channel_ablation"]["aggregate"]["mean_by_variant"][variant] = {
            "n": int(len(rows)),
            "shared_encoder0_cut_over_pre4_mean": _mean_finite([r.get("shared_encoder0_cut_over_pre4") for r in rows]),
            "hidden_pre_cut_over_pre4_mean": _mean_finite([r.get("hidden_pre_cut_over_pre4") for r in rows]),
            "hidden_pre_collapse_fraction_to_walk_mean": _mean_finite(
                [r.get("hidden_pre_collapse_fraction_to_walk") for r in rows]
            ),
        }

    # ---------------------------------------------------------------- B mapping probes
    Xh, Xs, Yh, map_rows = _make_mapping_pairs(
        states=states_281,
        hidden=hidden,
        target_clips=target_clips,
        pose_thr=float(args.mapping_pose_thr),
        contact_thr=float(args.mapping_contact_thr),
        pose_topk=int(args.pose_topk),
    )
    train_idx, test_idx = _split_indices(map_rows, seed=int(args.seed), test_frac=float(args.mapping_test_frac))
    alphas = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    mapping: Dict[str, Any] = {
        "pair_definition": (
            "for each target frame, pick Walk_F full_state_align frame; keep pairs with "
            f"pose_d<={float(args.mapping_pose_thr)} and contact_d<={float(args.mapping_contact_thr)}"
        ),
        "n_pairs": int(len(map_rows)),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "by_target_counts": {
            clip: int(sum(1 for r in map_rows if str(r["clip"]) == clip))
            for clip in target_clips
        },
        "hidden_pre_rep_mapping": {},
        "physical_state_mapping": {},
        "honesty_note": "rep->rep R2 only measures low-complexity learnability; it is not a realized-motion success metric.",
    }
    if len(map_rows) > 0:
        mapping["hidden_pre_rep_mapping"]["ridge"] = _ridge_fit_eval(
            Xh,
            Yh,
            map_rows,
            train_idx=train_idx,
            test_idx=test_idx,
            alphas=alphas,
        )
        mapping["hidden_pre_rep_mapping"]["mlp"] = _mlp_fit_eval(
            Xh,
            Yh,
            train_idx=train_idx,
            test_idx=test_idx,
            epochs=int(args.mlp_epochs),
            hidden=int(args.mlp_hidden),
            lr=float(args.mlp_lr),
            seed=int(args.seed),
        )
        Y_state_all = []
        for r in map_rows:
            Y_state_all.append(states_281[str(r["clip"])][int(r["target_frame"])])
        Y_state = np.stack(Y_state_all, axis=0).astype(np.float32)
        physical_groups = {
            "ego_vel": EGO_VEL_SLICE,
            "yaw_rate": YAW_RATE_SLICE,
            "contact": CONTACT_SLICE,
            "pose": POSE_SLICE,
            "state_all": slice(0, STATE_DIM),
        }
        for name, sl in physical_groups.items():
            mapping["physical_state_mapping"][name] = {
                "ridge": _ridge_fit_eval(
                    Xs,
                    Y_state[:, sl],
                    map_rows,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    alphas=alphas,
                )
            }
    payload["B_regime_mapping"] = mapping

    # ---------------------------------------------------------------- C direct/readout realized motion
    if not bool(args.skip_c):
        walk_full_sample = _build_full_sample(runner, npz_root, WALK_F)
        normalizer = StateNormalizer(states_281)
        thr = GateThresholds()
        c_rows: List[Dict[str, Any]] = []
        for target, meta in selected_meta.items():
            target_full_sample = _build_full_sample(runner, npz_root, target)
            goal_seam = states_281[target][int(meta["onset"]) : int(meta["onset"]) + SEAM_LEN_K]
            if int(goal_seam.shape[0]) < SEAM_LEN_K:
                payload["C_direct_motion_honesty"]["per_target"][target] = {"skip_reason": "goal seam too short"}
                continue
            per_mode: Dict[str, Any] = {}
            for mode in ("main", "lambda_model", "lambda_force1", "direct_full"):
                roll = _run_readout_rollout(
                    runner,
                    walk_full_sample,
                    target_full_sample,
                    phi=int(meta["phi"]),
                    onset=int(meta["onset"]),
                    horizon=int(args.post_frames),
                    context_len=int(args.context_len),
                    mode=mode,
                )
                y_raw = roll["y_raw"]
                contacts = roll["contacts"]
                cond_dir = roll["cond_dir"]
                roll_state = rollout_to_egocentric(
                    y_raw[:, :276].detach().cpu().numpy(),
                    y_raw[:, ROOT_VEL_OUT_SLICE].detach().cpu().numpy(),
                    cond_dir.detach().cpu().numpy(),
                    contacts.detach().cpu().numpy(),
                    fps=FPS,
                )
                state_eval = evaluate_rollout_state_space(
                    roll_state,
                    goal_seam,
                    normalizer.std,
                    thr,
                )
                foot = _foot_slip_summary(
                    runner.trainer,
                    y_raw,
                    contacts,
                    cond_dir,
                    fps=FPS,
                    skeleton_meta=skeleton_meta,
                )
                rec = {
                    "mode": mode,
                    "used_source_counts": roll["used_source_counts"],
                    "lambda_mean": roll["lambda_mean"],
                    "state_space": state_eval,
                    "foot_slip": foot,
                }
                per_mode[mode] = rec
                c_rows.append({"target": target, **rec})
            payload["C_direct_motion_honesty"]["per_target"][target] = {
                "phi": int(meta["phi"]),
                "onset": int(meta["onset"]),
                "goal_seam_len": int(goal_seam.shape[0]),
                "modes": per_mode,
            }
        payload["C_direct_motion_honesty"]["aggregate"] = {
            "mode_summary": {},
            "honesty_note": (
                "This is a matched-seam AR rollout with readout override; success/failure is judged on "
                "realized state pop and FK foot slip, not on hidden_pre similarity."
            ),
        }
        for mode in sorted({str(r["mode"]) for r in c_rows}):
            rows = [r for r in c_rows if str(r["mode"]) == mode]
            payload["C_direct_motion_honesty"]["aggregate"]["mode_summary"][mode] = {
                "n": int(len(rows)),
                "pop_safe_rate": _mean_finite([1.0 if bool(r["state_space"].get("pop_safe")) else 0.0 for r in rows]),
                "clip_resumable_rate": _mean_finite(
                    [1.0 if bool(r["state_space"].get("clip_resumable")) else 0.0 for r in rows]
                ),
                "mean_pop": _mean_finite([r["state_space"].get("pop") for r in rows]),
                "mean_best_pose_d": _mean_finite([r["state_space"].get("best_pose_d") for r in rows]),
                "foot_slip_mean_mps": _mean_finite(
                    [
                        r.get("foot_slip", {}).get("mean_mps_over_sides")
                        for r in rows
                        if isinstance(r.get("foot_slip"), Mapping)
                    ]
                ),
                "foot_slip_max_mps": _mean_finite(
                    [
                        r.get("foot_slip", {}).get("max_mps_over_sides")
                        for r in rows
                        if isinstance(r.get("foot_slip"), Mapping)
                    ]
                ),
            }

    # ---------------------------------------------------------------- D velocity budget
    baselines = _velocity_baseline_thresholds(states_281, raw_angvel, [WALK_F, *target_clips])
    payload["D_velocity_budget"]["baseline_p95"] = baselines
    for target, meta in selected_meta.items():
        payload["D_velocity_budget"]["per_target"][target] = _seam_velocity_budget(
            states=states_281,
            raw_angvel=raw_angvel,
            baselines=baselines,
            target=target,
            phi=int(meta["phi"]),
            onset=int(meta["onset"]),
        )
    payload["D_velocity_budget"]["realized_motion_acceptance_definition"] = {
        "do_not_use": "hidden_pre/turn-rep matching alone",
        "required": [
            "realized yaw follows commanded turn direction (yaw_corr>0 and heading MAE vs target decreases)",
            "pop_safe on egocentric state seam",
            "FK foot-slip under dual-contact stays within baseline-calibrated band",
            "positive and negative controls: continuous target/walk and direct/readout ablations",
        ],
    }

    # ---------------------------------------------------------------- E lambda runtime
    lambda_rows = []
    for row in ablation_rows_all:
        if row["variant"] != "matched_base":
            continue
        t = row["target"]
        tp = payload["A_channel_ablation"]["per_target"].get(t, {})
        variants = {r["variant"]: r for r in tp.get("variants", [])}
        lambda_rows.append(
            {
                "target": t,
                "matched_hidden_pre_cut_over_pre4": variants.get("matched_base", {}).get("hidden_pre_cut_over_pre4"),
            }
        )
    # The actual lambda series is already summarized in A probes; aggregate from matched_base if present.
    lam_cut_ratios = []
    lam_means = []
    for target, t_payload in payload["A_channel_ablation"]["per_target"].items():
        if t_payload.get("skip_reason"):
            continue
        # Rerun-free: use the saved per-target variants list for cut ratios; lambda mean is unavailable there.
        # C's lambda_model rows give the runtime mean when C is enabled.
        for r in t_payload.get("variants", []):
            if r.get("variant") == "matched_base":
                pass
        c_modes = payload["C_direct_motion_honesty"]["per_target"].get(target, {}).get("modes", {})
        if isinstance(c_modes, Mapping) and isinstance(c_modes.get("lambda_model"), Mapping):
            lm = c_modes["lambda_model"].get("lambda_mean")
            if lm is not None:
                lam_means.append(lm)
    payload["E_lambda_runtime"] = {
        "code_behavior": {
            "eval_application": "lambda_fusion is applied in eval only when trainer.lambda_fusion_apply is true; see train/eval_utils.py:385.",
            "blend_scope": "trainer._apply_lambda_fusion_to_raw blends only the rot6d slice with direct_norm; root velocity/contact are not replaced; see train/training_MPL.py:855.",
            "lambda_direction": "SO(3) blend uses omega*lam, so lam=1 moves to direct pose; see train/geometry.py:702.",
        },
        "measured_runtime_lambda_mean_from_C": _mean_finite(lam_means),
        "measured_runtime_lambda_n": int(len(lam_means)),
        "interpretation": (
            "If mean remains near the prior high value and C shows similar pop/foot behavior, current lambda_fusion is not a seam-aware gate."
        ),
    }

    # ---------------------------------------------------------------- tri-state summary
    a_status = payload["A_channel_ablation"]["aggregate"]["status"]
    b_r2 = (
        payload["B_regime_mapping"]
        .get("hidden_pre_rep_mapping", {})
        .get("ridge", {})
        .get("best", {})
        .get("r2")
    )
    c_direct = (
        payload["C_direct_motion_honesty"]
        .get("aggregate", {})
        .get("mode_summary", {})
        .get("direct_full", {})
    )
    direct_pop = c_direct.get("pop_safe_rate") if isinstance(c_direct, Mapping) else None
    direct_slip = c_direct.get("foot_slip_mean_mps") if isinstance(c_direct, Mapping) else None
    payload["tri_state"] = {
        "A_trunk_driver": a_status,
        "B_rep_mapping_low_complexity": {
            "status": "established_probe_only" if b_r2 is not None and math.isfinite(float(b_r2)) and float(b_r2) >= 0.5 else "suspect_or_low_r2",
            "ridge_hidden_pre_r2": b_r2,
            "not_a_motion_success": True,
        },
        "C_direct_motion_source": {
            "status": (
                "usable_motion_source_candidate"
                if direct_pop is not None and float(direct_pop) >= 0.5 and (direct_slip is None or float(direct_slip) < 0.25)
                else "rep_stability_not_enough_or_unverified"
            ),
            "direct_full_pop_safe_rate": direct_pop,
            "direct_full_foot_slip_mean_mps": direct_slip,
        },
        "D_motion_acceptance": {
            "status": "defined_not_yet_bridge_trained",
            "criterion": "realized yaw + pop_safe + FK foot-slip + controls",
        },
        "E_lambda_runtime": {
            "status": "established_code_path",
            "mean_lambda_from_C": payload["E_lambda_runtime"].get("measured_runtime_lambda_mean_from_C"),
        },
    }

    json_path = out_dir / "regime_bridge_probe_summary.json"
    md_path = out_dir / "regime_bridge_probe_summary.md"
    _dump_json(json_path, payload)

    lines: List[str] = []
    lines.append("# Action-Handoff Regime Bridge Probe")
    lines.append("")
    lines.append("Read-only probe. No base-model training, no checkpoint mutation.")
    lines.append("")
    lines.append("## A. Channel Causal Ablation")
    lines.append("")
    lines.append("| variant | n | hidden_pre cut/pre mean | hidden collapse to Walk mean | shared0 cut/pre mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for variant, row in payload["A_channel_ablation"]["aggregate"]["mean_by_variant"].items():
        lines.append(
            f"| {variant} | {row['n']} | {_fmt(row['hidden_pre_cut_over_pre4_mean'])} | "
            f"{_fmt(row['hidden_pre_collapse_fraction_to_walk_mean'])} | {_fmt(row['shared_encoder0_cut_over_pre4_mean'])} |"
        )
    lines.append("")
    st = payload["A_channel_ablation"]["aggregate"]["status"]
    lines.append(
        f"- A status: `{st['status']}`, best=`{st['best_variant']}`, "
        f"collapse={_fmt(st['best_hidden_pre_collapse_fraction_to_walk'])}"
    )
    lines.append("")
    lines.append("## B. Regime Mapping Learnability")
    lines.append("")
    ridge = payload["B_regime_mapping"].get("hidden_pre_rep_mapping", {}).get("ridge", {})
    mlp = payload["B_regime_mapping"].get("hidden_pre_rep_mapping", {}).get("mlp", {})
    lines.append(
        f"- pairs: `{payload['B_regime_mapping'].get('n_pairs')}` "
        f"(train `{payload['B_regime_mapping'].get('n_train')}`, test `{payload['B_regime_mapping'].get('n_test')}`)"
    )
    lines.append(
        f"- hidden_pre ridge R2: `{_fmt(ridge.get('best', {}).get('r2'))}`; "
        f"MLP R2: `{_fmt(mlp.get('r2'))}`"
    )
    for group, grow in payload["B_regime_mapping"].get("physical_state_mapping", {}).items():
        r2 = grow.get("ridge", {}).get("best", {}).get("r2")
        lines.append(f"- physical `{group}` ridge R2: `{_fmt(r2)}`")
    lines.append("")
    lines.append("> Rep mapping R2 is only a low-complexity probe. It is not a handoff success metric.")
    lines.append("")
    lines.append("## C. Direct Motion Honesty")
    lines.append("")
    lines.append("| mode | n | pop_safe | mean_pop | best_pose_d | foot_slip_mean_mps | foot_slip_max_mps |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mode, row in payload["C_direct_motion_honesty"].get("aggregate", {}).get("mode_summary", {}).items():
        lines.append(
            f"| {mode} | {row['n']} | {_fmt(row.get('pop_safe_rate'))} | {_fmt(row.get('mean_pop'))} | "
            f"{_fmt(row.get('mean_best_pose_d'))} | {_fmt(row.get('foot_slip_mean_mps'))} | "
            f"{_fmt(row.get('foot_slip_max_mps'))} |"
        )
    lines.append("")
    lines.append("## D. Velocity Budget")
    lines.append("")
    lines.append("| target | ego Δ L2 | yaw Δ deg/s | angvel Δ rms rad/s | max needed frames |")
    lines.append("|---|---:|---:|---:|---:|")
    for target, row in payload["D_velocity_budget"]["per_target"].items():
        frames = [v for v in row["frames_needed_at_continuous_p95"].values() if v is not None]
        lines.append(
            f"| {target} | {_fmt(row['ego_vel_delta_l2'])} | {_fmt(row['yaw_rate_delta_deg_s'])} | "
            f"{_fmt(row['bone_angvel_delta_rms_rad_s'])} | {max(frames) if frames else 'null'} |"
        )
    lines.append("")
    lines.append("## E. Lambda Runtime")
    lines.append("")
    lines.append(f"- runtime lambda mean from C: `{_fmt(payload['E_lambda_runtime'].get('measured_runtime_lambda_mean_from_C'))}`")
    lines.append("- eval applies lambda only under `lambda_fusion_apply`; blend changes rot6d only, not root/contact.")
    lines.append("")
    lines.append("## Tri-State")
    lines.append("")
    for key, row in payload["tri_state"].items():
        lines.append(f"- `{key}`: `{row.get('status')}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- `{json_path.resolve()}`")
    lines.append(f"- `{md_path.resolve()}`")
    _dump_md(md_path, lines)

    print(f"[ok] wrote {json_path}")
    print(f"[ok] wrote {md_path}")
    print(
        "[A] best="
        f"{payload['A_channel_ablation']['aggregate']['status']['best_variant']} "
        f"collapse={_fmt(payload['A_channel_ablation']['aggregate']['status']['best_hidden_pre_collapse_fraction_to_walk'])}"
    )
    print(
        "[B] hidden_pre ridge R2="
        f"{_fmt(ridge.get('best', {}).get('r2'))} mlp R2={_fmt(mlp.get('r2'))}"
    )


if __name__ == "__main__":
    main()
