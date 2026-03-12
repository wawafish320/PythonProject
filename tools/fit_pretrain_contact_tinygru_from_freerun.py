#!/usr/bin/env python3
"""
Fit a tiny GRU anchor for pretrain_contact from freerun JSONs.

Output checkpoint can be consumed by:
  train.validate.run_freerun_cycles --contacts_meas_pretrain_anchor_ckpt <ckpt.pt>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _expand_json_specs(specs: Sequence[str], *, pattern: str = "*_freerun_cycles.json") -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for spec in specs:
        if not spec:
            continue
        s = os.path.expanduser(str(spec))
        matches: List[Path] = []
        if any(ch in s for ch in "*?[]"):
            matches = [Path(p) for p in glob.glob(s)]
        else:
            p = Path(s)
            if p.is_dir():
                matches = sorted(p.glob(pattern))
            elif p.is_file():
                matches = [p]
        for m in matches:
            try:
                r = m.resolve()
            except Exception:
                r = m
            if r.is_file() and r not in seen:
                seen.add(r)
                out.append(r)
    return sorted(out)


def _as_float_list(x: Any) -> Optional[List[float]]:
    if not isinstance(x, list) or not x:
        return None
    out: List[float] = []
    for v in x:
        try:
            fv = float(v)
        except Exception:
            return None
        if not math.isfinite(fv):
            return None
        out.append(fv)
    return out


@dataclass
class SequenceSample:
    clip: str
    meas: np.ndarray  # (T,C)
    target: np.ndarray  # (T,C)
    gt: np.ndarray  # (T,C)
    plan: np.ndarray  # (T,C)


class TinyContactAnchor(torch.nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.gru = torch.nn.GRUCell(self.input_dim, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward_step(self, x: torch.Tensor, h: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            h = torch.zeros((int(x.shape[0]), self.hidden_dim), device=x.device, dtype=x.dtype)
        h2 = self.gru(x, h)
        logits = self.out(h2)
        return logits, h2


def _load_sequences(
    paths: Sequence[Path],
    *,
    round_gte: int,
    source_prefix: str,
    target_mode: str,
    mix_alpha: float,
) -> List[SequenceSample]:
    out: List[SequenceSample] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        steps = data.get("metrics_per_step", None)
        if not isinstance(steps, list):
            continue
        cycle_len = int(data.get("cycle_len", 0) or 0)
        step_floor = int(round_gte * cycle_len) if cycle_len > 0 else 0
        meas_rows: List[List[float]] = []
        tgt_rows: List[List[float]] = []
        gt_rows: List[List[float]] = []
        plan_rows: List[List[float]] = []
        for idx, st in enumerate(steps):
            if not isinstance(st, dict):
                continue
            step = int(st.get("step", idx) or idx)
            if step < step_floor:
                continue
            if source_prefix:
                sap = str(st.get("ContactsMeasSourceApplied", "") or "")
                if not sap.startswith(source_prefix):
                    continue
            meas = _as_float_list(st.get("ContactMeasPerC"))
            gt = _as_float_list(st.get("ContactGTPerC"))
            plan = _as_float_list(st.get("ContactPlanPerC"))
            if meas is None or gt is None or plan is None:
                continue
            if len(meas) != len(gt) or len(meas) != len(plan) or len(meas) <= 0:
                continue
            if target_mode == "gt":
                tgt = gt
            elif target_mode == "plan":
                tgt = plan
            else:
                tgt = [(1.0 - float(mix_alpha)) * float(g) + float(mix_alpha) * float(pl) for g, pl in zip(gt, plan)]
            meas_rows.append([float(x) for x in meas])
            tgt_rows.append([float(x) for x in tgt])
            gt_rows.append([float(x) for x in gt])
            plan_rows.append([float(x) for x in plan])
        if not meas_rows:
            continue
        arr_meas = np.asarray(meas_rows, dtype=np.float32)
        arr_tgt = np.asarray(tgt_rows, dtype=np.float32)
        arr_gt = np.asarray(gt_rows, dtype=np.float32)
        arr_plan = np.asarray(plan_rows, dtype=np.float32)
        if arr_meas.ndim != 2 or arr_tgt.ndim != 2 or arr_meas.shape != arr_tgt.shape:
            continue
        clip = str(data.get("clip") or p.stem.replace("_freerun_cycles", ""))
        out.append(SequenceSample(clip=clip, meas=arr_meas, target=arr_tgt, gt=arr_gt, plan=arr_plan))
    return out


def _mean(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    arr = np.asarray(list(xs), dtype=np.float64)
    if arr.size <= 0:
        return None
    return float(arr.mean())


def _fmt(x: Optional[float], nd: int = 6) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit tiny GRU pretrain_contact anchor from freerun JSONs.")
    ap.add_argument("--json", nargs="+", required=True, help="Paths/dirs/globs to *_freerun_cycles.json.")
    ap.add_argument("--round-gte", type=int, default=1, help="Use steps with round >= K (via step>=K*cycle_len).")
    ap.add_argument(
        "--require-source-prefix",
        type=str,
        default="pretrain_contact",
        help="Keep steps whose ContactsMeasSourceApplied starts with this prefix.",
    )
    ap.add_argument("--target", type=str, default="mix", choices=("gt", "plan", "mix"), help="Supervision target.")
    ap.add_argument("--mix-alpha", type=float, default=0.8, help="Target mix alpha when --target=mix.")
    ap.add_argument("--hidden-dim", type=int, default=16, help="GRU hidden size.")
    ap.add_argument("--delta-scale", type=float, default=1.0, help="Scale for input delta term.")
    ap.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    ap.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    ap.add_argument("--w-bce", type=float, default=1.0, help="BCE loss weight.")
    ap.add_argument("--w-smooth", type=float, default=0.05, help="Temporal smoothness weight.")
    ap.add_argument("--w-consistency", type=float, default=0.02, help="Consistency-to-input weight.")
    ap.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda", "mps"), help="Training device.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument(
        "--out-ckpt",
        type=str,
        default="debug_output/_tmp_pretrain_contact_tinygru_fit/anchor_ckpt.pt",
        help="Output checkpoint path.",
    )
    ap.add_argument(
        "--out-md",
        type=str,
        default="debug_output/_tmp_pretrain_contact_tinygru_fit/summary.md",
        help="Output markdown summary path.",
    )
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    mix_alpha = float(args.mix_alpha)
    if not math.isfinite(mix_alpha):
        mix_alpha = 0.8
    mix_alpha = float(max(0.0, min(1.0, mix_alpha)))

    paths = _expand_json_specs(args.json)
    if not paths:
        raise SystemExit("[FATAL] No JSON matched --json.")

    seqs = _load_sequences(
        paths,
        round_gte=max(0, int(args.round_gte)),
        source_prefix=str(args.require_source_prefix or "").strip(),
        target_mode=str(args.target or "mix").strip().lower(),
        mix_alpha=mix_alpha,
    )
    if not seqs:
        raise SystemExit("[FATAL] No valid sequences after filtering.")

    C = int(seqs[0].meas.shape[1])
    if C <= 0:
        raise SystemExit("[FATAL] Invalid contact dim.")
    for s in seqs:
        if int(s.meas.shape[1]) != C:
            raise SystemExit("[FATAL] Mixed contact dims across inputs are not supported.")

    dev = torch.device(str(args.device))
    model = TinyContactAnchor(input_dim=2 * C, hidden_dim=int(args.hidden_dim), output_dim=C).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    bce_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    delta_scale = float(args.delta_scale)
    if not math.isfinite(delta_scale):
        delta_scale = 1.0
    w_bce = float(args.w_bce)
    w_smooth = float(args.w_smooth)
    w_cons = float(args.w_consistency)

    hist: List[Dict[str, float]] = []
    for ep in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_smooth = 0.0
        total_cons = 0.0
        n_seq = 0
        for s in seqs:
            meas = torch.from_numpy(s.meas).to(device=dev, dtype=torch.float32)
            tgt = torch.from_numpy(s.target).to(device=dev, dtype=torch.float32)
            h = None
            prev_in = None
            prev_prob = None
            bce_acc = []
            smooth_acc = []
            cons_acc = []
            for t in range(int(meas.shape[0])):
                cur = meas[t : t + 1]
                if prev_in is None:
                    delta = torch.zeros_like(cur)
                else:
                    delta = cur - prev_in
                x = torch.cat([cur, delta * float(delta_scale)], dim=-1)
                logits, h = model.forward_step(x, h)
                prob = torch.sigmoid(logits)
                bce_acc.append(bce_fn(logits, tgt[t : t + 1]))
                if prev_prob is not None:
                    smooth_acc.append((prob - prev_prob).abs().mean())
                cons_acc.append((prob - cur).abs().mean())
                prev_in = cur
                prev_prob = prob
            if not bce_acc:
                continue
            bce = torch.stack(bce_acc).mean()
            smooth = torch.stack(smooth_acc).mean() if smooth_acc else torch.zeros((), device=dev)
            cons = torch.stack(cons_acc).mean() if cons_acc else torch.zeros((), device=dev)
            loss = float(w_bce) * bce + float(w_smooth) * smooth + float(w_cons) * cons
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.detach().item())
            total_bce += float(bce.detach().item())
            total_smooth += float(smooth.detach().item())
            total_cons += float(cons.detach().item())
            n_seq += 1
        if n_seq <= 0:
            raise SystemExit("[FATAL] No trainable sequence samples.")
        rec = {
            "epoch": float(ep),
            "loss": float(total_loss / n_seq),
            "bce": float(total_bce / n_seq),
            "smooth": float(total_smooth / n_seq),
            "consistency": float(total_cons / n_seq),
        }
        hist.append(rec)
        if ep == 1 or ep % 20 == 0 or ep == int(args.epochs):
            print(
                f"[ep {ep:04d}] loss={rec['loss']:.6f} bce={rec['bce']:.6f} "
                f"smooth={rec['smooth']:.6f} cons={rec['consistency']:.6f}"
            )

    model.eval()
    err_before_target: List[float] = []
    err_after_target: List[float] = []
    err_before_gt: List[float] = []
    err_after_gt: List[float] = []
    err_before_plan: List[float] = []
    err_after_plan: List[float] = []
    with torch.no_grad():
        for s in seqs:
            meas = torch.from_numpy(s.meas).to(device=dev, dtype=torch.float32)
            tgt = torch.from_numpy(s.target).to(device=dev, dtype=torch.float32)
            gt = torch.from_numpy(s.gt).to(device=dev, dtype=torch.float32)
            plan = torch.from_numpy(s.plan).to(device=dev, dtype=torch.float32)
            h = None
            prev_in = None
            preds: List[torch.Tensor] = []
            for t in range(int(meas.shape[0])):
                cur = meas[t : t + 1]
                if prev_in is None:
                    delta = torch.zeros_like(cur)
                else:
                    delta = cur - prev_in
                x = torch.cat([cur, delta * float(delta_scale)], dim=-1)
                logits, h = model.forward_step(x, h)
                preds.append(torch.sigmoid(logits))
                prev_in = cur
            pred = torch.cat(preds, dim=0) if preds else torch.zeros_like(meas)
            err_before_target.extend((meas - tgt).abs().flatten().cpu().tolist())
            err_after_target.extend((pred - tgt).abs().flatten().cpu().tolist())
            err_before_gt.extend((meas - gt).abs().flatten().cpu().tolist())
            err_after_gt.extend((pred - gt).abs().flatten().cpu().tolist())
            err_before_plan.extend((meas - plan).abs().flatten().cpu().tolist())
            err_after_plan.extend((pred - plan).abs().flatten().cpu().tolist())

    out_ckpt = Path(args.out_ckpt).expanduser()
    out_md = Path(args.out_md).expanduser()
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "input_dim": int(2 * C),
        "hidden_dim": int(args.hidden_dim),
        "output_dim": int(C),
        "delta_scale": float(delta_scale),
    }
    fit_meta = {
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_json_total": int(len(paths)),
        "num_sequences_used": int(len(seqs)),
        "round_gte": int(max(0, int(args.round_gte))),
        "require_source_prefix": str(args.require_source_prefix or ""),
        "target": str(args.target),
        "mix_alpha": float(mix_alpha),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "w_bce": float(w_bce),
        "w_smooth": float(w_smooth),
        "w_consistency": float(w_cons),
        "device": str(dev),
        "seed": int(args.seed),
        "final_loss": hist[-1] if hist else None,
        "metrics": {
            "abs_err_target_before": _mean(err_before_target),
            "abs_err_target_after": _mean(err_after_target),
            "abs_err_target_delta": (
                (_mean(err_after_target) - _mean(err_before_target))
                if _mean(err_before_target) is not None and _mean(err_after_target) is not None
                else None
            ),
            "abs_err_gt_before": _mean(err_before_gt),
            "abs_err_gt_after": _mean(err_after_gt),
            "abs_err_gt_delta": (
                (_mean(err_after_gt) - _mean(err_before_gt))
                if _mean(err_before_gt) is not None and _mean(err_after_gt) is not None
                else None
            ),
            "abs_err_plan_before": _mean(err_before_plan),
            "abs_err_plan_after": _mean(err_after_plan),
            "abs_err_plan_delta": (
                (_mean(err_after_plan) - _mean(err_before_plan))
                if _mean(err_before_plan) is not None and _mean(err_after_plan) is not None
                else None
            ),
        },
    }
    payload = {
        "kind": "pretrain_contact_anchor",
        "config": config,
        "state_dict": model.state_dict(),
        "fit": fit_meta,
        "history": hist,
    }
    torch.save(payload, out_ckpt)

    m = fit_meta["metrics"]
    lines: List[str] = []
    lines.append("# tiny GRU pretrain_contact anchor fit")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- json specs resolved: {len(paths)}")
    lines.append(f"- sequences used: {len(seqs)}")
    lines.append(f"- round_gte: {fit_meta['round_gte']}")
    lines.append(f"- source prefix: `{fit_meta['require_source_prefix']}`")
    lines.append(f"- target: `{fit_meta['target']}` (mix_alpha={fit_meta['mix_alpha']})")
    lines.append(f"- model: input_dim={config['input_dim']} hidden_dim={config['hidden_dim']} output_dim={config['output_dim']}")
    lines.append(f"- delta_scale: {config['delta_scale']}")
    lines.append(f"- loss weights: bce={w_bce} smooth={w_smooth} consistency={w_cons}")
    lines.append("")
    lines.append("## Metrics")
    lines.append(f"- abs_err_target before: {_fmt(m.get('abs_err_target_before'))}")
    lines.append(f"- abs_err_target after: {_fmt(m.get('abs_err_target_after'))}")
    lines.append(f"- abs_err_target delta: {_fmt(m.get('abs_err_target_delta'))}")
    lines.append(f"- abs_err_gt before: {_fmt(m.get('abs_err_gt_before'))}")
    lines.append(f"- abs_err_gt after: {_fmt(m.get('abs_err_gt_after'))}")
    lines.append(f"- abs_err_gt delta: {_fmt(m.get('abs_err_gt_delta'))}")
    lines.append(f"- abs_err_plan before: {_fmt(m.get('abs_err_plan_before'))}")
    lines.append(f"- abs_err_plan after: {_fmt(m.get('abs_err_plan_after'))}")
    lines.append(f"- abs_err_plan delta: {_fmt(m.get('abs_err_plan_delta'))}")
    lines.append("")
    lines.append("## Output")
    lines.append(f"- ckpt: `{out_ckpt}`")
    lines.append(f"- summary: `{out_md}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {out_ckpt}")
    print(f"[done] wrote {out_md}")
    print(
        "[overall] abs_err_target "
        f"before={_fmt(m.get('abs_err_target_before'))} "
        f"after={_fmt(m.get('abs_err_target_after'))} "
        f"delta={_fmt(m.get('abs_err_target_delta'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

