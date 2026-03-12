#!/usr/bin/env python3
"""
Leave-one-out sweep for Experiment 2 (delta* predictability).

This is a convenience wrapper around tools/analyze_hinge_oracle_predictability.py that:
  - caches SampleSet per clip (oracle delta* computation is the expensive part)
  - runs leave-one-out: train on N-1 clips, test on held-out clip
  - reports micro-averaged metrics across holdouts

Supports:
  - ridge (linear): same as Experiment 2 baseline
  - mlp (nonlinear): small MLP regressor to test whether the residual is underfitting

Default behavior (if you already generated the base_direct freerun JSONs):
  python tools/sweep_hinge_oracle_predictability_l1o.py
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _load_pred_module() -> Any:
    mod_path = Path(__file__).resolve().parent / "analyze_hinge_oracle_predictability.py"
    spec = importlib.util.spec_from_file_location("hinge_pred", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    # Needed for dataclasses typing in py>=3.12 (dataclass looks up sys.modules[__module__]).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_clip_arg(s: str) -> Tuple[str, Path]:
    if "=" not in s:
        raise ValueError(f"--clip must be NAME=PATH, got {s!r}")
    name, path = s.split("=", 1)
    name = name.strip()
    path = Path(path.strip()).expanduser()
    if not name:
        raise ValueError(f"Invalid clip name in {s!r}")
    if not path.is_file():
        raise ValueError(f"Clip JSON not found: {path}")
    return name, path


def _fmt(x: float, prec: int = 3) -> str:
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return "NA"


def _parse_csv_floats(spec: str) -> List[float]:
    out: List[float] = []
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _parse_csv_ints(spec: str) -> List[int]:
    out: List[int] = []
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X.std(axis=0, dtype=np.float64).astype(np.float32)
    # Avoid division by ~0 for constant columns (e.g., bias term).
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
    return mean, std


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=np.float32) - mean) / std


class _MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, *, hidden: int, depth: int, dropout: float) -> None:
        super().__init__()
        in_dim = int(in_dim)
        hidden = int(hidden)
        depth = int(depth)
        dropout = float(dropout)

        if in_dim <= 0:
            raise ValueError(f"in_dim must be >0, got {in_dim}")
        if hidden <= 0:
            raise ValueError(f"hidden must be >0, got {hidden}")
        if depth < 0:
            raise ValueError(f"depth must be >=0, got {depth}")

        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _huber_loss(pred: torch.Tensor, target: torch.Tensor, *, beta: float, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Huber loss in degrees.
      - pred/target: (...,)
      - beta: transition point (deg)
      - weight: optional per-sample weights (...,)
    """
    beta = float(beta)
    if not (beta > 0.0):
        loss = (pred - target).abs()
    else:
        diff = pred - target
        abs_diff = diff.abs()
        loss = torch.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def _fit_mlp(
    *,
    X: np.ndarray,
    y: np.ndarray,
    ang_deg: np.ndarray,
    angle_thresh: float,
    train_tail_only: bool,
    tail_weight: float,
    seed: int,
    device: torch.device,
    hidden: int,
    depth: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    steps: int,
    batch_size: int,
    val_frac: float,
    eval_every: int,
    patience: int,
    huber_beta: float,
    grad_clip: float,
) -> Tuple[_MLPRegressor, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    ang_deg = np.asarray(ang_deg, dtype=np.float32).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if y.shape[0] != X.shape[0] or ang_deg.shape[0] != X.shape[0]:
        raise ValueError(f"X/y/ang length mismatch: {X.shape} vs {y.shape} vs {ang_deg.shape}")

    # Optional: focus on tail-only frames (ang>th).
    if bool(train_tail_only):
        m = ang_deg > float(angle_thresh)
        if not bool(np.any(m)):
            raise ValueError("train_tail_only set but no tail frames found.")
        X = X[m]
        y = y[m]
        ang_deg = ang_deg[m]

    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(X.shape[0])
    n_val = int(round(float(val_frac) * float(X.shape[0])))
    n_val = max(1, min(n_val, X.shape[0] - 1)) if X.shape[0] >= 2 else 0
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:] if n_val > 0 else idx

    X_tr = X[tr_idx]
    y_tr = y[tr_idx]
    ang_tr = ang_deg[tr_idx]

    X_val = X[val_idx] if n_val > 0 else None
    y_val = y[val_idx] if n_val > 0 else None

    mean, std = _standardize_fit(X_tr)
    X_tr = _standardize_apply(X_tr, mean, std)
    if X_val is not None:
        X_val = _standardize_apply(X_val, mean, std)

    # Per-sample weight: upweight tail frames.
    w_tr: Optional[np.ndarray] = None
    if float(tail_weight) > 0.0:
        w_tr = np.ones_like(y_tr, dtype=np.float32)
        w_tr += float(tail_weight) * (ang_tr > float(angle_thresh)).astype(np.float32)

    torch.manual_seed(int(seed))
    model = _MLPRegressor(X.shape[1], hidden=int(hidden), depth=int(depth), dropout=float(dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    w_tr_t = torch.tensor(w_tr, dtype=torch.float32, device=device) if w_tr is not None else None

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device) if X_val is not None else None
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device) if y_val is not None else None

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad = 0

    steps = int(steps)
    batch_size = int(batch_size)
    eval_every = max(1, int(eval_every))
    patience = max(1, int(patience))
    n_tr = int(X_tr_t.shape[0])
    batch_size = max(1, min(batch_size, n_tr))

    with torch.enable_grad():
        perm = torch.randperm(n_tr, device=device)
        ptr = 0
        for step in range(1, steps + 1):
            # Sample without replacement by cycling through random permutations.
            if ptr + batch_size > n_tr:
                perm = torch.randperm(n_tr, device=device)
                ptr = 0
            bi = perm[ptr : ptr + batch_size]
            ptr += batch_size
            xb = X_tr_t[bi]
            yb = y_tr_t[bi]
            wb = w_tr_t[bi] if w_tr_t is not None else None

            pred = model(xb)
            loss = _huber_loss(pred, yb, beta=float(huber_beta), weight=wb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            opt.step()

            if X_val_t is None or y_val_t is None:
                continue
            if step % eval_every != 0:
                continue

            with torch.no_grad():
                model.eval()
                val_pred = model(X_val_t)
                val_mae = (val_pred - y_val_t).abs().mean().item()
                model.train()

            if val_mae < best_val - 1e-6:
                best_val = float(val_mae)
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    model.load_state_dict(best_state)
    model.eval()
    return model, mean, std


def _mlp_predict(model: _MLPRegressor, X: np.ndarray, *, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    Xn = _standardize_apply(X, mean, std)
    with torch.no_grad():
        pred = model(torch.tensor(Xn, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
    return pred.astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Leave-one-out sweep for hinge oracle predictability.")
    ap.add_argument(
        "--clip",
        action="append",
        default=None,
        help="Clip mapping NAME=PATH to freerun_cycles JSON. Can be repeated.",
    )
    ap.add_argument("--bone", type=str, default="calf_r")
    ap.add_argument("--branch", type=str, default="direct", choices=("inc", "direct", "blend"))
    ap.add_argument("--axis", type=str, default="z", choices=("x", "y", "z"))
    ap.add_argument("--max-deg", type=float, default=45.0)
    ap.add_argument("--grid-step-deg", type=float, default=0.25)
    ap.add_argument("--angle-thresh", type=float, default=20.0)
    ap.add_argument("--min-cycle", type=int, default=1)
    ap.add_argument("--contact-source", type=str, default="gt", choices=("gt", "plan", "meas"))
    ap.add_argument("--contact-side", type=str, default="r", choices=("l", "r"))
    ap.add_argument("--contact-value", type=int, default=0, choices=(0, 1))
    ap.add_argument("--contact-thresh", type=float, default=0.5)
    ap.add_argument("--model", type=str, default="ridge", choices=("ridge", "mlp"))
    ap.add_argument(
        "--feature-sets",
        type=str,
        default="const,phase,cond,cond+phase,plan+meas+phase,cond+plan+meas+phase",
        help="Comma list of feature sets.",
    )
    ap.add_argument(
        "--ridges",
        type=str,
        default="1.0",
        help="Comma list of ridge alphas to sweep (default: 1.0).",
    )
    ap.add_argument("--train-tail-only", action="store_true", help="Train only on frames with ang_deg>angle_thresh.")
    ap.add_argument(
        "--phase-harmonics",
        type=str,
        default="1",
        help="Comma list of phase harmonics to sweep (default: 1).",
    )
    # MLP regressor knobs (only used when --model=mlp).
    ap.add_argument("--mlp-hidden", type=str, default="64", help="Comma list of hidden sizes (default: 64).")
    ap.add_argument("--mlp-depth", type=str, default="2", help="Comma list of depths (#hidden layers, default: 2).")
    ap.add_argument("--mlp-dropout", type=str, default="0.0", help="Comma list of dropouts (default: 0.0).")
    ap.add_argument("--mlp-lr", type=str, default="1e-3", help="Comma list of learning rates (default: 1e-3).")
    ap.add_argument("--mlp-weight-decay", type=str, default="1e-4", help="Comma list of weight decays (default: 1e-4).")
    ap.add_argument("--mlp-steps", type=int, default=2000, help="Training steps per fold (default: 2000).")
    ap.add_argument("--mlp-batch-size", type=int, default=1024, help="Batch size (default: 1024).")
    ap.add_argument("--mlp-val-frac", type=float, default=0.1, help="Validation fraction per fold (default: 0.1).")
    ap.add_argument("--mlp-eval-every", type=int, default=100, help="Eval frequency (steps) for early stop (default: 100).")
    ap.add_argument("--mlp-patience", type=int, default=20, help="Early-stop patience (evals, default: 20).")
    ap.add_argument("--mlp-huber-beta", type=float, default=5.0, help="Huber beta in degrees (default: 5.0).")
    ap.add_argument("--mlp-tail-weight", type=float, default=0.0, help="Extra weight for tail frames in training (default: 0.0).")
    ap.add_argument("--mlp-grad-clip", type=float, default=5.0, help="Grad norm clip (default: 5.0, <=0 disables).")
    ap.add_argument("--mlp-seed", type=int, default=0, help="Seed for MLP folds (default: 0).")
    ap.add_argument("--mlp-device", type=str, default="cpu", help="Device for MLP training (default: cpu).")
    ap.add_argument("--topk", type=int, default=10, help="Top-K configs to print (default: 10).")
    args = ap.parse_args()

    mod = _load_pred_module()

    # Default clips (current workspace).
    if not args.clip:
        defaults = [
            ("Walk_F", Path("debug_output/_oracle_hinge_upperbound/Walk_F_exp007_base_direct/Walk_F_freerun_cycles.json")),
            ("Walk_L_To_L", Path("debug_output/_oracle_hinge_upperbound/Walk_L_To_L_base_direct/Walk_L_To_L_freerun_cycles.json")),
            ("Walk_L_To_R", Path("debug_output/_oracle_hinge_upperbound/Walk_L_To_R_base_direct/Walk_L_To_R_freerun_cycles.json")),
            ("Walk_R_To_L", Path("debug_output/_oracle_hinge_upperbound/Walk_R_To_L_base_direct/Walk_R_To_L_freerun_cycles.json")),
            ("Walk_R_To_R", Path("debug_output/_oracle_hinge_upperbound/Walk_R_To_R_base_direct/Walk_R_To_R_freerun_cycles.json")),
        ]
        clips: Dict[str, Path] = {n: p for n, p in defaults if p.is_file()}
        if len(clips) < 2:
            raise SystemExit("No --clip provided and default clip paths not found (need >=2 clips).")
    else:
        clips = dict(_parse_clip_arg(s) for s in args.clip)

    feature_sets = [s.strip() for s in str(args.feature_sets).split(",") if s.strip()]
    phase_harmonics = [int(s) for s in str(args.phase_harmonics).split(",") if s.strip()]
    model_kind = str(args.model).strip().lower()
    ridges = _parse_csv_floats(str(args.ridges)) if model_kind == "ridge" else []

    common = dict(
        bone=str(args.bone),
        branch=str(args.branch),
        axis=str(args.axis),
        max_deg=float(args.max_deg),
        grid_step_deg=float(args.grid_step_deg),
        min_cycle=int(args.min_cycle),
        contact_source=str(args.contact_source),
        contact_side=str(args.contact_side),
        contact_value=int(args.contact_value),
        contact_thresh=float(args.contact_thresh),
    )

    # Cache SampleSet per clip once (oracle delta* computation is the expensive part).
    samples: Dict[str, Any] = {}
    for name, p in clips.items():
        samples[name] = mod._build_samples(p, **common)

    th = float(args.angle_thresh)

    # Cache X per clip per config for faster sweeps.
    X_cache: Dict[Tuple[str, str, int], np.ndarray] = {}

    def _get_X(clip_name: str, feature_set: str, phase_h: int) -> np.ndarray:
        k = (str(clip_name), str(feature_set), int(phase_h))
        if k not in X_cache:
            X_cache[k] = np.asarray(
                mod._make_features(samples[clip_name], feature_set=str(feature_set), phase_harmonics=int(phase_h)),
                dtype=np.float32,
            )
        return X_cache[k]

    def _eval_delta_hat(sample_set: Any, delta_hat: np.ndarray) -> Dict[str, float]:
        y = np.asarray(sample_set.delta_star_deg, dtype=np.float32).reshape(-1)
        ang0 = np.asarray(sample_set.ang_deg, dtype=np.float32).reshape(-1)
        delta_hat = np.clip(delta_hat, -common["max_deg"], common["max_deg"])
        ang1 = mod._apply_delta(np.asarray(sample_set.w_deg_xyz, dtype=np.float32), delta_hat, axis=common["axis"])
        tail = ang0 > th
        n = int(ang0.shape[0])
        tail_n = int(np.sum(tail))
        return {
            "N": float(n),
            "tail_N": float(tail_n),
            "ang_mean": float(np.mean(ang0)),
            "ang_mean_after": float(np.mean(ang1)),
            "p_gt_th": float(np.mean(ang0 > th)),
            "p_gt_th_after": float(np.mean(ang1 > th)),
            "delta_mae": float(mod._mae(delta_hat, y)),
            "delta_mae_tail": float(mod._mae(delta_hat[tail], y[tail])) if tail_n > 0 else float("nan"),
        }

    # Sweep configs and compute micro-average across leave-one-out.
    names = list(samples.keys())
    cfg_rows: List[Dict[str, Any]] = []

    if model_kind == "ridge":
        for fs in feature_sets:
            for ph in phase_harmonics:
                for ridge in ridges:
                    acc = {"N": 0.0, "tail_N": 0.0, "ang0": 0.0, "ang1": 0.0, "p0": 0.0, "p1": 0.0, "d": 0.0, "dt": 0.0}
                    ok = True
                    for holdout in names:
                        train_names = [n for n in names if n != holdout]
                        test_set = samples[holdout]
                        try:
                            X_train = np.concatenate([_get_X(n, fs, int(ph)) for n in train_names], axis=0)
                            y_train = np.concatenate(
                                [np.asarray(samples[n].delta_star_deg, dtype=np.float32).reshape(-1) for n in train_names], axis=0
                            )
                            ang_train = np.concatenate([np.asarray(samples[n].ang_deg, dtype=np.float32).reshape(-1) for n in train_names], axis=0)
                            if bool(args.train_tail_only):
                                m = ang_train > th
                                if not bool(np.any(m)):
                                    raise ValueError("train_tail_only set but no tail frames found.")
                                X_train = X_train[m]
                                y_train = y_train[m]

                            w = mod._ridge_solve(X_train.astype(np.float32), y_train.astype(np.float32), alpha=float(ridge))
                            X_test = _get_X(holdout, fs, int(ph))
                            delta_hat = (X_test @ w).astype(np.float32)
                            m = _eval_delta_hat(test_set, delta_hat)
                        except Exception:
                            ok = False
                            break

                        n = m["N"]
                        tn = m["tail_N"]
                        acc["N"] += n
                        acc["tail_N"] += tn
                        acc["ang0"] += n * m["ang_mean"]
                        acc["ang1"] += n * m["ang_mean_after"]
                        acc["p0"] += n * m["p_gt_th"]
                        acc["p1"] += n * m["p_gt_th_after"]
                        acc["d"] += n * m["delta_mae"]
                        if np.isfinite(m["delta_mae_tail"]):
                            acc["dt"] += tn * m["delta_mae_tail"]

                    if not ok or acc["N"] <= 0:
                        continue

                    N = acc["N"]
                    tailN = acc["tail_N"]
                    cfg_rows.append(
                        {
                            "feature_set": fs,
                            "phase_h": int(ph),
                            "ridge": float(ridge),
                            "mean": acc["ang0"] / N,
                            "mean_after": acc["ang1"] / N,
                            "p_gt_th": acc["p0"] / N,
                            "p_gt_th_after": acc["p1"] / N,
                            "delta_mae": acc["d"] / N,
                            "delta_mae_tail": (acc["dt"] / tailN) if tailN > 0 else float("nan"),
                            "N": int(N),
                        }
                    )

    elif model_kind == "mlp":
        hiddens = _parse_csv_ints(str(args.mlp_hidden)) or [64]
        depths = _parse_csv_ints(str(args.mlp_depth)) or [2]
        dropouts = _parse_csv_floats(str(args.mlp_dropout)) or [0.0]
        lrs = _parse_csv_floats(str(args.mlp_lr)) or [1e-3]
        wds = _parse_csv_floats(str(args.mlp_weight_decay)) or [1e-4]

        dev = torch.device(str(args.mlp_device))

        for fs in feature_sets:
            for ph in phase_harmonics:
                for hidden in hiddens:
                    for depth in depths:
                        for dropout in dropouts:
                            for lr in lrs:
                                for wd in wds:
                                    acc = {"N": 0.0, "tail_N": 0.0, "ang0": 0.0, "ang1": 0.0, "p0": 0.0, "p1": 0.0, "d": 0.0, "dt": 0.0}
                                    ok = True
                                    for holdout in names:
                                        train_names = [n for n in names if n != holdout]
                                        test_set = samples[holdout]
                                        try:
                                            X_train = np.concatenate([_get_X(n, fs, int(ph)) for n in train_names], axis=0)
                                            y_train = np.concatenate(
                                                [np.asarray(samples[n].delta_star_deg, dtype=np.float32).reshape(-1) for n in train_names], axis=0
                                            )
                                            ang_train = np.concatenate(
                                                [np.asarray(samples[n].ang_deg, dtype=np.float32).reshape(-1) for n in train_names], axis=0
                                            )

                                            model, mean, std = _fit_mlp(
                                                X=X_train,
                                                y=y_train,
                                                ang_deg=ang_train,
                                                angle_thresh=th,
                                                train_tail_only=bool(args.train_tail_only),
                                                tail_weight=float(args.mlp_tail_weight),
                                                seed=int(args.mlp_seed),
                                                device=dev,
                                                hidden=int(hidden),
                                                depth=int(depth),
                                                dropout=float(dropout),
                                                lr=float(lr),
                                                weight_decay=float(wd),
                                                steps=int(args.mlp_steps),
                                                batch_size=int(args.mlp_batch_size),
                                                val_frac=float(args.mlp_val_frac),
                                                eval_every=int(args.mlp_eval_every),
                                                patience=int(args.mlp_patience),
                                                huber_beta=float(args.mlp_huber_beta),
                                                grad_clip=float(args.mlp_grad_clip),
                                            )

                                            X_test = _get_X(holdout, fs, int(ph))
                                            delta_hat = _mlp_predict(model, X_test, mean=mean, std=std, device=dev)
                                            m = _eval_delta_hat(test_set, delta_hat)
                                        except Exception:
                                            ok = False
                                            break

                                        n = m["N"]
                                        tn = m["tail_N"]
                                        acc["N"] += n
                                        acc["tail_N"] += tn
                                        acc["ang0"] += n * m["ang_mean"]
                                        acc["ang1"] += n * m["ang_mean_after"]
                                        acc["p0"] += n * m["p_gt_th"]
                                        acc["p1"] += n * m["p_gt_th_after"]
                                        acc["d"] += n * m["delta_mae"]
                                        if np.isfinite(m["delta_mae_tail"]):
                                            acc["dt"] += tn * m["delta_mae_tail"]

                                    if not ok or acc["N"] <= 0:
                                        continue

                                    N = acc["N"]
                                    tailN = acc["tail_N"]
                                    cfg_rows.append(
                                        {
                                            "feature_set": fs,
                                            "phase_h": int(ph),
                                            "hidden": int(hidden),
                                            "depth": int(depth),
                                            "dropout": float(dropout),
                                            "lr": float(lr),
                                            "wd": float(wd),
                                            "mean": acc["ang0"] / N,
                                            "mean_after": acc["ang1"] / N,
                                            "p_gt_th": acc["p0"] / N,
                                            "p_gt_th_after": acc["p1"] / N,
                                            "delta_mae": acc["d"] / N,
                                            "delta_mae_tail": (acc["dt"] / tailN) if tailN > 0 else float("nan"),
                                            "N": int(N),
                                        }
                                    )

    else:
        raise SystemExit(f"Unknown --model: {args.model!r}")

    if not cfg_rows:
        raise SystemExit("No valid configs (check feature sets / clip availability).")

    # Sort: minimize tail prob, then mean_after
    cfg_rows.sort(key=lambda r: (r["p_gt_th_after"], r["mean_after"]))

    extra = ""
    if model_kind == "mlp":
        extra = (
            f" mlp_steps={args.mlp_steps} bs={args.mlp_batch_size} val_frac={args.mlp_val_frac} "
            f"eval_every={args.mlp_eval_every} patience={args.mlp_patience} huber_beta={args.mlp_huber_beta} "
            f"tail_weight={args.mlp_tail_weight} device={args.mlp_device} seed={args.mlp_seed}"
        )
    print(
        f"[Config] model={args.model} bone={args.bone} branch={args.branch} axis={args.axis} "
        f"min_cycle={args.min_cycle} contact={args.contact_source}:{args.contact_side}=={args.contact_value} "
        f"th={args.angle_thresh} clips={len(names)} train_tail_only={bool(args.train_tail_only)}{extra}"
    )
    print()

    print(f"=== Top-{int(args.topk)} configs (micro-avg LOO) ===")
    if model_kind == "ridge":
        print("fs | ph | ridge | mean->after | P(>th)->after | dMAE(all) | dMAE(tail) | N")
        for r in cfg_rows[: int(args.topk)]:
            print(
                f"{r['feature_set']:<20s} | {r['phase_h']:2d} | {r['ridge']:<6g} | "
                f"{_fmt(r['mean'],2)}->{_fmt(r['mean_after'],2)} | { _fmt(r['p_gt_th'],3)}->{_fmt(r['p_gt_th_after'],3)} | "
                f"{_fmt(r['delta_mae'],2)} | { _fmt(r['delta_mae_tail'],2)} | {r['N']}"
            )
    else:
        print("fs | ph | h | d | drop | lr | wd | mean->after | P(>th)->after | dMAE(all) | dMAE(tail) | N")
        for r in cfg_rows[: int(args.topk)]:
            print(
                f"{r['feature_set']:<20s} | {r['phase_h']:2d} | {r['hidden']:3d} | {r['depth']:1d} | {r['dropout']:<4g} | "
                f"{r['lr']:<7g} | {r['wd']:<6g} | "
                f"{_fmt(r['mean'],2)}->{_fmt(r['mean_after'],2)} | { _fmt(r['p_gt_th'],3)}->{_fmt(r['p_gt_th_after'],3)} | "
                f"{_fmt(r['delta_mae'],2)} | { _fmt(r['delta_mae_tail'],2)} | {r['N']}"
            )

    best_per: Dict[str, Dict[str, Any]] = {}
    for r in cfg_rows:
        fs = r["feature_set"]
        if fs not in best_per:
            best_per[fs] = r

    print("\n=== Best per feature_set (micro-avg LOO) ===")
    if model_kind == "ridge":
        print("fs | ph | ridge | mean->after | P(>th)->after | dMAE(all) | dMAE(tail) | N")
        for fs in feature_sets:
            r = best_per.get(fs)
            if not r:
                continue
            print(
                f"{r['feature_set']:<20s} | {r['phase_h']:2d} | {r['ridge']:<6g} | "
                f"{_fmt(r['mean'],2)}->{_fmt(r['mean_after'],2)} | { _fmt(r['p_gt_th'],3)}->{_fmt(r['p_gt_th_after'],3)} | "
                f"{_fmt(r['delta_mae'],2)} | { _fmt(r['delta_mae_tail'],2)} | {r['N']}"
            )
    else:
        print("fs | ph | h | d | drop | lr | wd | mean->after | P(>th)->after | dMAE(all) | dMAE(tail) | N")
        for fs in feature_sets:
            r = best_per.get(fs)
            if not r:
                continue
            print(
                f"{r['feature_set']:<20s} | {r['phase_h']:2d} | {r['hidden']:3d} | {r['depth']:1d} | {r['dropout']:<4g} | "
                f"{r['lr']:<7g} | {r['wd']:<6g} | "
                f"{_fmt(r['mean'],2)}->{_fmt(r['mean_after'],2)} | { _fmt(r['p_gt_th'],3)}->{_fmt(r['p_gt_th_after'],3)} | "
                f"{_fmt(r['delta_mae'],2)} | { _fmt(r['delta_mae_tail'],2)} | {r['N']}"
            )


if __name__ == "__main__":
    main()
