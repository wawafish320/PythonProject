#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class Sample:
    clip: str
    cycle: int
    sic: int
    x: np.ndarray
    y: np.ndarray


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
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                for v in range(lo, hi + 1):
                    out.add(int(v))
            continue
        if t.lstrip("-").isdigit():
            out.add(int(t))
    return out if out else None


def _infer_dims(ckpt_path: Path) -> Dict[str, int]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint structure: {ckpt_path}")

    if "shared_encoder.0.weight" not in state:
        raise RuntimeError("Missing shared_encoder.0.weight; cannot infer hidden_dim.")
    hidden_dim = int(state["shared_encoder.0.weight"].shape[0])

    if "contact_plan_cell.weight_ih" in state:
        cond_dim = int(state["contact_plan_cell.weight_ih"].shape[1])
    else:
        raise RuntimeError("Missing contact_plan_cell.weight_ih; cannot infer cond_dim.")

    if "direct_pose_leg_head.0.weight" in state:
        leg_in_dim = int(state["direct_pose_leg_head.0.weight"].shape[1])
    elif "direct_pose_head.0.weight" in state:
        leg_in_dim = int(state["direct_pose_head.0.weight"].shape[1])
    else:
        raise RuntimeError("Missing direct_pose_head/direct_pose_leg_head weights; cannot infer input dim.")

    chp_dim = int(cond_dim + hidden_dim)
    if chp_dim > leg_in_dim:
        raise RuntimeError(
            f"Invalid slices: cond_dim+hidden_dim={chp_dim} > leg_in_dim={leg_in_dim} (ckpt={ckpt_path})."
        )

    return {
        "cond_dim": cond_dim,
        "hidden_dim": hidden_dim,
        "cond_hidden_pre_dim": chp_dim,
        "leg_in_dim": leg_in_dim,
    }


def _extract_samples(
    json_path: Path,
    *,
    bone: str,
    use_oracle_right: bool,
    sics_filter: Optional[set[int]],
) -> List[Sample]:
    payload = json.loads(json_path.read_text())
    clip = str(payload.get("clip") or json_path.stem.replace("_freerun_cycles", ""))

    io_block = payload.get("direct_leg_head_io", {})
    io_steps = io_block.get("steps", []) if isinstance(io_block, dict) else []
    if not isinstance(io_steps, list) or not io_steps:
        raise RuntimeError(f"Missing direct_leg_head_io.steps in {json_path}")

    tgt_block = payload.get("direct_leg_omega_alpha_sweep", {})
    tgt_steps = tgt_block.get("steps", []) if isinstance(tgt_block, dict) else []
    if not isinstance(tgt_steps, list) or not tgt_steps:
        raise RuntimeError(f"Missing direct_leg_omega_alpha_sweep.steps in {json_path}")

    feat_by_step: Dict[int, Tuple[int, int, np.ndarray]] = {}
    for ent in io_steps:
        if not isinstance(ent, dict):
            continue
        t = ent.get("step")
        cyc = ent.get("cycle")
        sic = ent.get("step_in_cycle")
        if not isinstance(t, int) or not isinstance(cyc, int) or not isinstance(sic, int):
            continue
        base = ent.get("baseline", {})
        if not isinstance(base, dict):
            continue
        x = base.get("in")
        if not isinstance(x, list) or len(x) <= 0:
            continue
        feat_by_step[int(t)] = (int(cyc), int(sic), np.asarray(x, dtype=np.float32))

    tgt_key = "omega_oracle_right_xyz_rad" if use_oracle_right else "omega_oracle_xyz_rad"
    tgt_by_step: Dict[int, np.ndarray] = {}
    for ent in tgt_steps:
        if not isinstance(ent, dict):
            continue
        t = ent.get("step")
        if not isinstance(t, int):
            continue
        pb = ent.get("per_bone", {})
        if not isinstance(pb, dict):
            continue
        b = pb.get(str(bone), {})
        if not isinstance(b, dict):
            continue
        y = b.get(tgt_key)
        if isinstance(y, list) and len(y) == 3:
            tgt_by_step[int(t)] = np.asarray(y, dtype=np.float32)

    out: List[Sample] = []
    for t in sorted(set(feat_by_step.keys()) & set(tgt_by_step.keys())):
        cyc, sic, x = feat_by_step[t]
        if sics_filter is not None and int(sic) not in sics_filter:
            continue
        out.append(Sample(clip=clip, cycle=int(cyc), sic=int(sic), x=x, y=tgt_by_step[t]))
    return out


def _fit_linear(X: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    std = np.maximum(X.std(axis=0, keepdims=True), 1e-6)
    Xs = (X - mu) / std
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1), dtype=Xs.dtype)], axis=1)
    w, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    return {"mu": mu, "std": std, "w": w}


def _predict_linear(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    Xs = (X - model["mu"]) / model["std"]
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1), dtype=Xs.dtype)], axis=1)
    return Xb @ model["w"]


def _cosine_mean(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    n1 = np.linalg.norm(y_pred, axis=1)
    n2 = np.linalg.norm(y_true, axis=1)
    denom = np.maximum(n1 * n2, eps)
    cos = (y_pred * y_true).sum(axis=1) / denom
    return float(np.mean(cos))


def _sign_rate(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> Tuple[float, int]:
    zp = y_pred[:, 2::3]
    zt = y_true[:, 2::3]
    mask = (np.abs(zp) > eps) & (np.abs(zt) > eps)
    if int(mask.sum()) <= 0:
        return float("nan"), 0
    ok = np.sign(zp[mask]) == np.sign(zt[mask])
    return float(np.mean(ok)), int(mask.sum())


def _r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = y_true.mean(axis=0, keepdims=True)
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    sign, sign_n = _sign_rate(y_pred, y_true)
    return {
        "r2": _r2(y_pred, y_true),
        "cos": _cosine_mean(y_pred, y_true),
        "sign": sign,
        "sign_count": int(sign_n),
        "n": int(y_true.shape[0]),
    }


def _eval_train_all(X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
    model = _fit_linear(X, Y)
    yp = _predict_linear(model, X)
    return _metrics(yp, Y)


def _eval_loo_clip(X: np.ndarray, Y: np.ndarray, clips: Sequence[str]) -> Dict[str, Any]:
    clips_arr = np.asarray(clips)
    uniq = sorted(set(str(c) for c in clips_arr.tolist()))
    fold_preds: List[np.ndarray] = []
    fold_true: List[np.ndarray] = []
    per_clip: Dict[str, Dict[str, Any]] = {}

    for c in uniq:
        te = clips_arr == str(c)
        tr = ~te
        if int(tr.sum()) < 4 or int(te.sum()) < 1:
            continue
        model = _fit_linear(X[tr], Y[tr])
        yp = _predict_linear(model, X[te])
        fold_preds.append(yp)
        fold_true.append(Y[te])
        per_clip[str(c)] = _metrics(yp, Y[te])

    if not fold_preds:
        raise RuntimeError("No valid LOO-clip folds.")

    yp_all = np.concatenate(fold_preds, axis=0)
    yt_all = np.concatenate(fold_true, axis=0)
    out = _metrics(yp_all, yt_all)
    out["per_clip"] = per_clip
    return out


def _apply_control(
    X_full: np.ndarray,
    *,
    cond_dim: int,
    hidden_dim: int,
    mode: str,
    seed: int,
) -> np.ndarray:
    x = np.asarray(X_full, dtype=np.float32).copy()
    rng = np.random.default_rng(int(seed))
    cond_sl = slice(0, int(cond_dim))
    hid_sl = slice(int(cond_dim), int(cond_dim + hidden_dim))

    if mode == "none":
        return x
    if mode == "cond_shuffle":
        p = rng.permutation(x.shape[0])
        x[:, cond_sl] = x[p, cond_sl]
        return x
    if mode == "hidden_pre_shuffle":
        p = rng.permutation(x.shape[0])
        x[:, hid_sl] = x[p, hid_sl]
        return x
    if mode == "cond_drop":
        x[:, cond_sl] = 0.0
        return x
    if mode == "hidden_pre_drop":
        x[:, hid_sl] = 0.0
        return x
    raise ValueError(f"Unknown control mode: {mode}")


def _fmt(v: Any) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        x = float(v)
    except Exception:
        return str(v)
    if np.isnan(x):
        return "nan"
    return f"{x:.4f}"


def _write_markdown(summary: Dict[str, Any], out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# D0 Cond Audit Summary")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- checkpoint: `{summary['config']['checkpoint']}`")
    lines.append(f"- json_files: {len(summary['config']['json_files'])}")
    lines.append(f"- bone: `{summary['config']['bone']}`")
    lines.append(f"- sics: `{summary['config']['sics']}`")
    lines.append(
        f"- dims: cond={summary['dims']['cond_dim']} hidden_pre={summary['dims']['hidden_dim']} "
        f"cond+hidden_pre={summary['dims']['cond_hidden_pre_dim']} leg_in={summary['dims']['leg_in_dim']}"
    )
    lines.append("")

    lines.append("## Samples")
    lines.append("| clip | n |")
    lines.append("|---|---:|")
    for k, v in summary["sample_count_by_clip"].items():
        lines.append(f"| {k} | {int(v)} |")
    lines.append("")

    lines.append("## Main Probe (Linear)")
    lines.append("| feature | split | R2 | cos | sign | sign_n | n |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for feat, rows in summary["main"].items():
        for split in ("train_all", "loo_clip"):
            m = rows[split]
            lines.append(
                f"| {feat} | {split} | {_fmt(m['r2'])} | {_fmt(m['cos'])} | {_fmt(m['sign'])} | {int(m['sign_count'])} | {int(m['n'])} |"
            )
    lines.append("")

    lines.append("## Controls on cond+hidden_pre")
    lines.append("| control | split | R2 | cos | sign |")
    lines.append("|---|---|---:|---:|---:|")
    for ctl, rows in summary["controls"].items():
        for split in ("train_all", "loo_clip"):
            m = rows[split]
            lines.append(f"| {ctl} | {split} | {_fmt(m['r2'])} | {_fmt(m['cos'])} | {_fmt(m['sign'])} |")
    lines.append("")

    lines.append("## Quick Read")
    full = summary["controls"]["none"]["loo_clip"]
    csh = summary["controls"]["cond_shuffle"]["loo_clip"]
    hsh = summary["controls"]["hidden_pre_shuffle"]["loo_clip"]
    cdp = summary["controls"]["cond_drop"]["loo_clip"]
    hdp = summary["controls"]["hidden_pre_drop"]["loo_clip"]
    lines.append(
        "- LOO cos drop vs full (none): "
        f"cond_shuffle={_fmt(full['cos'] - csh['cos'])}, "
        f"hidden_shuffle={_fmt(full['cos'] - hsh['cos'])}, "
        f"cond_drop={_fmt(full['cos'] - cdp['cos'])}, "
        f"hidden_drop={_fmt(full['cos'] - hdp['cos'])}."
    )
    lines.append(
        "- LOO sign drop vs full (none): "
        f"cond_shuffle={_fmt(full['sign'] - csh['sign'])}, "
        f"hidden_shuffle={_fmt(full['sign'] - hsh['sign'])}, "
        f"cond_drop={_fmt(full['sign'] - cdp['sign'])}, "
        f"hidden_drop={_fmt(full['sign'] - hdp['sign'])}."
    )

    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="D0 cond-content audit from run_freerun_cycles exports.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--json-files", type=str, required=True, help="Comma-separated freerun JSON paths.")
    ap.add_argument("--bone", type=str, default="calf_r")
    ap.add_argument("--sics", type=str, default="9-14,39-42")
    ap.add_argument("--use-oracle-right", action="store_true")
    ap.add_argument("--out-json", type=str, required=True)
    ap.add_argument("--out-md", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).expanduser()
    json_files = [Path(x.strip()).expanduser() for x in str(args.json_files).split(",") if x.strip()]
    if not json_files:
        raise SystemExit("[FATAL] --json-files is empty.")
    for p in json_files:
        if not p.exists():
            raise SystemExit(f"[FATAL] json file not found: {p}")

    dims = _infer_dims(ckpt)
    sics_filter = _parse_int_set(str(args.sics))

    samples: List[Sample] = []
    for p in json_files:
        samples.extend(
            _extract_samples(
                p,
                bone=str(args.bone),
                use_oracle_right=bool(args.use_oracle_right),
                sics_filter=sics_filter,
            )
        )
    if not samples:
        raise SystemExit("[FATAL] no samples extracted.")

    x_dim = int(samples[0].x.shape[0])
    y_dim = int(samples[0].y.shape[0])
    if any(int(s.x.shape[0]) != x_dim for s in samples):
        raise SystemExit("[FATAL] inconsistent feature dims across samples.")
    if any(int(s.y.shape[0]) != y_dim for s in samples):
        raise SystemExit("[FATAL] inconsistent target dims across samples.")

    X_full = np.stack([s.x for s in samples], axis=0).astype(np.float32)
    Y = np.stack([s.y for s in samples], axis=0).astype(np.float32)
    clips = [s.clip for s in samples]

    cond_dim = int(dims["cond_dim"])
    hid_dim = int(dims["hidden_dim"])
    chp_dim = int(dims["cond_hidden_pre_dim"])

    feat_map: Dict[str, np.ndarray] = {
        "cond": X_full[:, :cond_dim],
        "hidden_pre": X_full[:, cond_dim : cond_dim + hid_dim],
        "cond+hidden_pre": X_full[:, :chp_dim],
    }

    main_summary: Dict[str, Any] = {}
    for name, X in feat_map.items():
        main_summary[name] = {
            "train_all": _eval_train_all(X, Y),
            "loo_clip": _eval_loo_clip(X, Y, clips),
        }

    control_modes = ["none", "cond_shuffle", "hidden_pre_shuffle", "cond_drop", "hidden_pre_drop"]
    ctl_summary: Dict[str, Any] = {}
    for i, mode in enumerate(control_modes):
        Xc = _apply_control(
            X_full[:, :chp_dim],
            cond_dim=cond_dim,
            hidden_dim=hid_dim,
            mode=mode,
            seed=int(args.seed) + i,
        )
        ctl_summary[mode] = {
            "train_all": _eval_train_all(Xc, Y),
            "loo_clip": _eval_loo_clip(Xc, Y, clips),
        }

    sample_count_by_clip: Dict[str, int] = {}
    for c in clips:
        sample_count_by_clip[str(c)] = int(sample_count_by_clip.get(str(c), 0) + 1)

    summary = {
        "config": {
            "checkpoint": str(ckpt),
            "json_files": [str(p) for p in json_files],
            "bone": str(args.bone),
            "sics": str(args.sics),
            "use_oracle_right": bool(args.use_oracle_right),
            "seed": int(args.seed),
        },
        "dims": dims,
        "sample_count_by_clip": sample_count_by_clip,
        "main": main_summary,
        "controls": ctl_summary,
    }

    out_json = Path(args.out_json).expanduser()
    out_md = Path(args.out_md).expanduser()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    _write_markdown(summary, out_md)

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
