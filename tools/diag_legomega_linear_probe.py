#!/usr/bin/env python3
"""
Linear/MLP probe for direct_leg_omega: can the (frozen) leg-head input features
predict the oracle omega direction/magnitude at phase-locked steps?

This is a lightweight offline diagnostic:
  1) Run run_freerun_cycles with:
       --export_direct_leg_head_io
       --export_direct_leg_omega_alpha_sweep
     and a small selector (e.g. --direct_leg_omega_alpha_sweep_sics 8,12,14,54,55).
  2) This script reads the resulting JSON, aligns feature vectors (leg head fc0 input)
     with oracle omega targets (from alpha-sweep), trains a tiny probe, and reports
     cos(pred, oracle).

Note: This probe is *not* training the full model. It only tests representational
      sufficiency of the fixed features presented to the leg-omega head.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def _parse_csv_list(s: str) -> List[str]:
    return [t.strip() for t in str(s or "").split(",") if t.strip()]


def _parse_int_set(spec: str) -> Optional[set[int]]:
    s = str(spec or "").strip()
    if not s:
        return None
    out: set[int] = set()
    for tok in s.replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        if "-" in t or ":" in t:
            sep = "-" if "-" in t else ":"
            a, b = [x.strip() for x in t.split(sep, 1)]
            if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
                try:
                    lo = int(a)
                    hi = int(b)
                    if lo > hi:
                        lo, hi = hi, lo
                    for v in range(lo, hi + 1):
                        out.add(int(v))
                except Exception:
                    pass
            continue
        if t.lstrip("-").isdigit():
            try:
                out.add(int(t))
            except Exception:
                pass
    return out if out else None


def _safe_unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _cos_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    denom = np.maximum(na * nb, eps)
    return (a * b).sum(axis=-1) / denom


@dataclass
class Sample:
    step: int
    cycle: int
    sic: int
    x: np.ndarray  # (D,)
    y: np.ndarray  # (3*B,)
    baseline_cos_by_bone: Dict[str, float]


def _extract_head_io_features(
    payload: Dict[str, Any],
    *,
    feature: str,
    shared_side: str,
) -> Dict[int, Tuple[int, int, np.ndarray]]:
    """
    Returns: step -> (cycle, sic, x[D])
    """
    block = payload.get("direct_leg_head_io", None)
    if not isinstance(block, dict):
        raise ValueError("Missing 'direct_leg_head_io' in JSON. Run run_freerun_cycles with --export_direct_leg_head_io.")

    steps = block.get("steps", None)
    if not isinstance(steps, list) or not steps:
        raise ValueError("'direct_leg_head_io.steps' missing/empty.")

    feat_kind, feat_key = feature.split(".", 1) if "." in feature else ("baseline", feature)
    if feat_kind not in ("baseline", "shared"):
        raise ValueError(f"--feature must start with baseline.* or shared.*; got {feature!r}")
    if feat_key not in ("in", "pre0"):
        raise ValueError(f"--feature must end with .in or .pre0; got {feature!r}")

    out: Dict[int, Tuple[int, int, np.ndarray]] = {}
    side = str(shared_side or "l").strip().lower()
    if side not in ("l", "r"):
        side = "l"

    for ent in steps:
        if not isinstance(ent, dict):
            continue
        t = ent.get("step", None)
        cyc = ent.get("cycle", None)
        sic = ent.get("step_in_cycle", None)
        if not isinstance(t, int) or not isinstance(cyc, int) or not isinstance(sic, int):
            continue

        vec: Optional[Sequence[float]] = None
        if feat_kind == "baseline":
            base = ent.get("baseline", None)
            if isinstance(base, dict):
                vec = base.get(feat_key, None)
        else:
            sh = ent.get("shared", None)
            if isinstance(sh, dict) and side in sh and isinstance(sh[side], dict):
                vec = sh[side].get(feat_key, None)

        if not isinstance(vec, list) or not vec:
            continue
        x = np.asarray(vec, dtype=np.float32)
        if x.ndim != 1 or x.size <= 0:
            continue
        out[int(t)] = (int(cyc), int(sic), x)

    if not out:
        raise ValueError("No usable head-io features found (check selector and --feature).")
    return out


def _extract_oracle_targets(
    payload: Dict[str, Any],
    *,
    bones: Sequence[str],
    use_oracle_right: bool,
) -> Dict[int, Dict[str, Any]]:
    block = payload.get("direct_leg_omega_alpha_sweep", None)
    if not isinstance(block, dict):
        raise ValueError(
            "Missing 'direct_leg_omega_alpha_sweep' in JSON. Run run_freerun_cycles with --export_direct_leg_omega_alpha_sweep."
        )
    steps = block.get("steps", None)
    if not isinstance(steps, list) or not steps:
        raise ValueError("'direct_leg_omega_alpha_sweep.steps' missing/empty.")

    bones_set = {str(b) for b in bones}
    tgt_key = "omega_oracle_right_xyz_rad" if use_oracle_right else "omega_oracle_xyz_rad"
    out: Dict[int, Dict[str, Any]] = {}
    for ent in steps:
        if not isinstance(ent, dict):
            continue
        t = ent.get("step", None)
        cyc = ent.get("cycle", None)
        sic = ent.get("step_in_cycle", None)
        if not isinstance(t, int) or not isinstance(cyc, int) or not isinstance(sic, int):
            continue
        pb = ent.get("per_bone", None)
        if not isinstance(pb, dict) or not pb:
            continue

        per_bone: Dict[str, Dict[str, Any]] = {}
        ok = True
        for b in bones_set:
            bb = pb.get(b, None)
            if not isinstance(bb, dict):
                ok = False
                break
            v = bb.get(tgt_key, None)
            if not (isinstance(v, list) and len(v) == 3):
                ok = False
                break
            per_bone[b] = bb
        if not ok:
            continue
        out[int(t)] = {"cycle": int(cyc), "sic": int(sic), "per_bone": per_bone}

    if not out:
        raise ValueError("No usable alpha-sweep oracle targets found for the requested bones.")
    return out


def _build_dataset(
    *,
    payload: Dict[str, Any],
    bones: List[str],
    use_oracle_right: bool,
    feature: str,
    shared_side: str,
    sics_filter: Optional[set[int]],
    cycles_filter: Optional[set[int]],
) -> List[Sample]:
    feat_by_step = _extract_head_io_features(payload, feature=feature, shared_side=shared_side)
    tgt_by_step = _extract_oracle_targets(payload, bones=bones, use_oracle_right=use_oracle_right)

    steps = sorted(set(feat_by_step.keys()) & set(tgt_by_step.keys()))
    if not steps:
        raise ValueError("No overlapping steps between head_io and alpha_sweep.")

    tgt_key = "omega_oracle_right_xyz_rad" if use_oracle_right else "omega_oracle_xyz_rad"

    samples: List[Sample] = []
    for t in steps:
        cyc_f, sic_f, x = feat_by_step[t]
        tgt_ent = tgt_by_step[t]
        cyc = int(tgt_ent["cycle"])
        sic = int(tgt_ent["sic"])
        if (cyc_f != cyc) or (sic_f != sic):
            # Should not happen; indicates misalignment in JSON.
            continue
        if sics_filter is not None and int(sic) not in sics_filter:
            continue
        if cycles_filter is not None and int(cyc) not in cycles_filter:
            continue

        pb = tgt_ent["per_bone"]
        y_chunks: List[np.ndarray] = []
        base_cos: Dict[str, float] = {}
        for b in bones:
            bb = pb.get(b, None)
            if not isinstance(bb, dict):
                continue
            v = bb.get(tgt_key, None)
            if not (isinstance(v, list) and len(v) == 3):
                continue
            y_chunks.append(np.asarray(v, dtype=np.float32))
            # baseline cos from alpha-sweep (pred vs oracle, left apply)
            cos_key = "cos_pred_oracle_right" if use_oracle_right else "cos_pred_oracle"
            try:
                base_cos[b] = float(bb.get(cos_key))
            except Exception:
                base_cos[b] = float("nan")

        if len(y_chunks) != len(bones):
            continue
        y = np.concatenate(y_chunks, axis=0)  # (3*B,)
        samples.append(Sample(step=int(t), cycle=int(cyc), sic=int(sic), x=x, y=y, baseline_cos_by_bone=base_cos))

    if not samples:
        raise ValueError("No samples after filtering.")
    return samples


def _fit_probe(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    model_kind: str,
    hidden: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> nn.Module:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    d_in = int(X_train.shape[1])
    d_out = int(Y_train.shape[1])

    kind = str(model_kind or "linear").strip().lower()
    if kind == "linear":
        net: nn.Module = nn.Linear(d_in, d_out)
    elif kind == "mlp":
        h = max(1, int(hidden))
        net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(inplace=False), nn.Linear(h, d_out))
    else:
        raise ValueError(f"Unknown --model {model_kind!r}; choose linear|mlp.")

    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss(reduction="mean")

    x_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(Y_train).float()

    net.train()
    for _ in range(int(max(1, epochs))):
        opt.zero_grad(set_to_none=True)
        pred = net(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
    return net


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return mu.astype(np.float32), std.astype(np.float32)


def _standardize_apply(X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mu) / std).astype(np.float32)


def _eval_cos(
    Y_pred: np.ndarray,
    Y_true: np.ndarray,
    bones: Sequence[str],
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    B = int(len(bones))
    out: Dict[str, np.ndarray] = {}
    for bi, b in enumerate(bones):
        a = Y_pred[:, bi * 3 : bi * 3 + 3]
        g = Y_true[:, bi * 3 : bi * 3 + 3]
        out[str(b)] = _cos_np(a, g, eps=eps)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True, help="Path to Walk_F_freerun_cycles.json containing head_io + alpha_sweep.")
    ap.add_argument("--bones", type=str, default="foot_l", help="Comma-separated bones to probe (default: foot_l).")
    ap.add_argument(
        "--use_oracle_right",
        action="store_true",
        help="Use omega_oracle_right_xyz_rad as target (for right-multiply apply). Default uses omega_oracle_xyz_rad.",
    )
    ap.add_argument(
        "--feature",
        type=str,
        default="baseline.in",
        help="Feature to use from direct_leg_head_io: baseline.in|baseline.pre0|shared.in|shared.pre0 (shared uses --shared_side).",
    )
    ap.add_argument(
        "--shared_side",
        type=str,
        default="l",
        choices=("l", "r"),
        help="When --feature starts with shared.*, choose which side call to use (l/r).",
    )
    ap.add_argument("--sics", type=str, default="", help="Optional sic filter (e.g. '12,14' or '8-14').")
    ap.add_argument("--cycles", type=str, default="", help="Optional cycle filter (e.g. '1-4' or '4').")
    ap.add_argument(
        "--split",
        type=str,
        default="loo_cycle",
        choices=("loo_cycle", "train_all"),
        help="Evaluation split: leave-one-cycle-out (default) or train_all (fit+report on all samples).",
    )
    ap.add_argument("--model", type=str, default="linear", choices=("linear", "mlp"), help="Probe type.")
    ap.add_argument("--hidden", type=int, default=128, help="Hidden dim for --model mlp.")
    ap.add_argument("--epochs", type=int, default=2000, help="Training epochs (small dataset => cheap).")
    ap.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = ap.parse_args()

    path = Path(args.json).expanduser()
    payload = json.loads(path.read_text())

    bones = _parse_csv_list(args.bones)
    if not bones:
        raise SystemExit("[FATAL] --bones is empty.")

    sics_filter = _parse_int_set(args.sics)
    cycles_filter = _parse_int_set(args.cycles)

    samples = _build_dataset(
        payload=payload,
        bones=bones,
        use_oracle_right=bool(args.use_oracle_right),
        feature=str(args.feature),
        shared_side=str(args.shared_side),
        sics_filter=sics_filter,
        cycles_filter=cycles_filter,
    )

    X = np.stack([s.x for s in samples], axis=0)  # (N,D)
    Y = np.stack([s.y for s in samples], axis=0)  # (N,3B)
    cycles = np.asarray([s.cycle for s in samples], dtype=np.int32)
    sics = np.asarray([s.sic for s in samples], dtype=np.int32)
    steps = np.asarray([s.step for s in samples], dtype=np.int32)

    # Baseline stats (from alpha-sweep cos_pred_oracle).
    base_cos = {b: np.asarray([s.baseline_cos_by_bone.get(b, float("nan")) for s in samples], dtype=np.float32) for b in bones}

    print(f"[Data] N={X.shape[0]} D={X.shape[1]} bones={bones} feature={args.feature}")

    def _stat(v: np.ndarray) -> str:
        v = v[np.isfinite(v)]
        if v.size == 0:
            return "None"
        return f"mean={v.mean():.4f} p50={np.quantile(v,0.5):.4f} p10={np.quantile(v,0.1):.4f} min={v.min():.4f}"

    for b in bones:
        print(f"[Baseline cos] {b}: {_stat(base_cos[b])}")

    split = str(args.split)
    if split == "train_all":
        mu, std = _standardize_fit(X)
        Xs = _standardize_apply(X, mu, std)
        net = _fit_probe(
            Xs,
            Y,
            model_kind=str(args.model),
            hidden=int(args.hidden),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            epochs=int(args.epochs),
            seed=int(args.seed),
        )
        net.eval()
        with torch.no_grad():
            Yp = net(torch.from_numpy(Xs).float()).cpu().numpy()
        cos_by_bone = _eval_cos(Yp, Y, bones)
        for b in bones:
            print(f"[Probe cos][train_all] {b}: {_stat(cos_by_bone[b])}")
        return

    # Leave-one-cycle-out evaluation.
    uniq_cycles = sorted(set(int(c) for c in cycles.tolist()))
    if len(uniq_cycles) < 2:
        raise SystemExit(f"[FATAL] Need >=2 cycles for loo_cycle split; got cycles={uniq_cycles}.")

    all_test_cos: Dict[str, List[float]] = {b: [] for b in bones}
    all_test_meta: List[Tuple[int, int, int]] = []  # (step, cycle, sic)

    for test_cyc in uniq_cycles:
        test_mask = cycles == int(test_cyc)
        train_mask = ~test_mask
        if int(train_mask.sum()) < 2 or int(test_mask.sum()) < 1:
            continue

        Xtr = X[train_mask]
        Ytr = Y[train_mask]
        Xte = X[test_mask]
        Yte = Y[test_mask]

        mu, std = _standardize_fit(Xtr)
        Xtr_s = _standardize_apply(Xtr, mu, std)
        Xte_s = _standardize_apply(Xte, mu, std)

        net = _fit_probe(
            Xtr_s,
            Ytr,
            model_kind=str(args.model),
            hidden=int(args.hidden),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            epochs=int(args.epochs),
            seed=int(args.seed) + int(test_cyc),
        )
        net.eval()
        with torch.no_grad():
            Yp = net(torch.from_numpy(Xte_s).float()).cpu().numpy()

        cos_by_bone = _eval_cos(Yp, Yte, bones)
        for b in bones:
            all_test_cos[b].extend([float(v) for v in cos_by_bone[b].tolist()])
        for st, cyc, sic in zip(steps[test_mask].tolist(), cycles[test_mask].tolist(), sics[test_mask].tolist()):
            all_test_meta.append((int(st), int(cyc), int(sic)))

    # Aggregate over all held-out samples.
    for b in bones:
        v = np.asarray(all_test_cos[b], dtype=np.float32)
        print(f"[Probe cos][loo_cycle] {b}: {_stat(v)}")

    # Extra: per-sic aggregates (useful for the foot_l@sic14 question).
    if all_test_meta:
        meta = np.asarray(all_test_meta, dtype=np.int32)  # (M,3)
        sic_all = meta[:, 2]
        for b in bones:
            v = np.asarray(all_test_cos[b], dtype=np.float32)
            for sic in sorted(set(int(x) for x in sic_all.tolist())):
                m = sic_all == int(sic)
                if int(m.sum()) <= 0:
                    continue
                vv = v[m]
                print(f"[Probe cos][loo_cycle] {b} @ sic={sic}: {_stat(vv)}")


if __name__ == "__main__":  # pragma: no cover
    main()
