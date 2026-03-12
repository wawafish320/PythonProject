#!/usr/bin/env python3
"""
Create a checkpoint that *already* contains `direct_pose_leg_head_shared.*` tensors, initialized from
the legacy `direct_pose_leg_head.*` weights.

Why:
  - `train/posttrain.py` will auto warm-start the shared head when it sees `direct_pose_leg_head.*`
    but no `direct_pose_leg_head_shared.*`.
  - By injecting shared-head tensors into the input ckpt, we can bypass that warm-start and test
    alternative init strategies (e.g. no R/L averaging) without touching training code.

This script assumes the common Stage7 layout:
  - legacy head input ends with phase_z_in: [sin(c0),cos(c0), sin(c1),cos(c1)] (last 4 dims)
  - shared head input is: [... base ..., plan_side, meas_side, sin_side, cos_side]  (same total dim)

Example:
  python tools/make_legomega_shared_init_ckpt.py \
    --ckpt-in  models/.../ckpt_last_legacy.pth \
    --ckpt-out models/.../ckpt_init_shared_copyR.pth \
    --config   config/posttrain_WalkF_stage7_legomega_routedshared_warm_20260126.json \
    --mode     copy_r
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s or "").split(",") if x.strip()]


def _infer_side_positions(names: List[str]) -> Tuple[List[int], List[int]]:
    pos_r: List[int] = []
    pos_l: List[int] = []
    for i, n in enumerate(names):
        s = str(n).lower()
        if s.endswith("_r") or "right" in s:
            pos_r.append(int(i))
        elif s.endswith("_l") or "left" in s:
            pos_l.append(int(i))
    return pos_r, pos_l


def _align_left_to_right_order(names: List[str], pos_r: List[int], pos_l: List[int]) -> List[int]:
    names_lc = [str(n).lower() for n in names]
    pos_l_by_name: Dict[str, int] = {}
    for p in pos_l:
        if 0 <= int(p) < len(names_lc):
            pos_l_by_name[names_lc[int(p)]] = int(p)
    aligned: List[int] = []
    for p in pos_r:
        pn = names_lc[int(p)] if 0 <= int(p) < len(names_lc) else ""
        cand = pn.replace("_r", "_l").replace("right", "left")
        if cand in pos_l_by_name:
            aligned.append(int(pos_l_by_name[cand]))
    if len(aligned) != len(pos_r):
        return list(pos_l)
    return aligned


def main() -> None:
    ap = argparse.ArgumentParser(description="Inject direct_pose_leg_head_shared weights into a ckpt (init from legacy).")
    ap.add_argument("--ckpt-in", type=str, required=True)
    ap.add_argument("--ckpt-out", type=str, required=True)
    ap.add_argument("--config", type=str, required=True, help="Posttrain config JSON (for bones + contact order).")
    ap.add_argument(
        "--mode",
        type=str,
        default="avg",
        choices=("avg", "copy_r", "copy_l"),
        help="How to merge R/L when building shared weights (default: avg).",
    )
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    bones = _split_csv(cfg.get("direct_pose_leg_bones", ""))
    if not bones:
        raise SystemExit("Config has empty direct_pose_leg_bones; cannot infer joint ordering.")

    contact_order = str(cfg.get("direct_pose_leg_contact_order", "lr") or "lr").lower().strip()
    if contact_order == "lr":
        ch_l, ch_r = 0, 1
    elif contact_order == "rl":
        ch_r, ch_l = 0, 1
    else:
        raise SystemExit(f"Unsupported direct_pose_leg_contact_order={contact_order!r}; expected 'lr' or 'rl'.")

    ckpt_in = Path(args.ckpt_in)
    ckpt = torch.load(str(ckpt_in), map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise SystemExit("Unexpected checkpoint format: expected dict with key 'model'.")
    sd = ckpt["model"]
    if not isinstance(sd, (dict, OrderedDict)):
        raise SystemExit("Unexpected checkpoint format: ckpt['model'] is not a state_dict mapping.")

    # Legacy leg head tensors (sequential: Linear(0), Linear(3), Linear(6)).
    k_w0 = "direct_pose_leg_head.0.weight"
    k_b0 = "direct_pose_leg_head.0.bias"
    k_w1 = "direct_pose_leg_head.3.weight"
    k_b1 = "direct_pose_leg_head.3.bias"
    k_w2 = "direct_pose_leg_head.6.weight"
    k_b2 = "direct_pose_leg_head.6.bias"
    for k in (k_w0, k_b0, k_w1, k_b1, k_w2, k_b2):
        if k not in sd:
            raise SystemExit(f"Missing legacy leg head tensor: {k}")

    W0 = sd[k_w0].detach().clone()
    b0 = sd[k_b0].detach().clone()
    W1 = sd[k_w1].detach().clone()
    b1 = sd[k_b1].detach().clone()
    W2 = sd[k_w2].detach().clone()
    b2 = sd[k_b2].detach().clone()

    if W0.ndim != 2 or W2.ndim != 2:
        raise SystemExit("Unexpected tensor rank for legacy weights.")
    hid = int(W0.shape[0])
    legacy_in = int(W0.shape[1])
    if legacy_in < 4:
        raise SystemExit(f"Legacy in_features={legacy_in} < 4; cannot locate phase columns.")

    # Determine offsets (matches train/posttrain.py warm-start logic).
    phase_off_legacy = legacy_in - 4
    phase_off_shared = legacy_in - 2
    plan_off_shared = phase_off_shared - 2
    if plan_off_shared < 0:
        raise SystemExit("Computed negative plan_off_shared; unexpected input layout.")

    # Infer side joint positions.
    pos_r, pos_l = _infer_side_positions(bones)
    if not pos_r or not pos_l or len(pos_r) != len(pos_l):
        raise SystemExit(
            f"Expected symmetric _r/_l bones; got pos_r={pos_r} pos_l={pos_l} (bones={bones})."
        )
    K = int(len(bones))
    K_side = int(len(pos_r))
    if int(W2.shape[0]) != 3 * K:
        raise SystemExit(f"Legacy out_features mismatch: got {int(W2.shape[0])}, expected {3*K}.")

    aligned_pos_l = _align_left_to_right_order(bones, pos_r, pos_l)
    if len(aligned_pos_l) != K_side:
        raise SystemExit("Failed to align left joints to right order (unexpected).")

    # Build W0 for each side: copy base cols, map phase cols, keep plan/meas as zeros.
    W0_r = torch.zeros_like(W0)
    W0_l = torch.zeros_like(W0)
    W0_r[:, :phase_off_legacy] = W0[:, :phase_off_legacy]
    W0_l[:, :phase_off_legacy] = W0[:, :phase_off_legacy]

    idx_sin_r = phase_off_legacy + 2 * int(ch_r) + 0
    idx_cos_r = phase_off_legacy + 2 * int(ch_r) + 1
    idx_sin_l = phase_off_legacy + 2 * int(ch_l) + 0
    idx_cos_l = phase_off_legacy + 2 * int(ch_l) + 1
    W0_r[:, phase_off_shared + 0] = W0[:, idx_sin_r]
    W0_r[:, phase_off_shared + 1] = W0[:, idx_cos_r]
    W0_l[:, phase_off_shared + 0] = W0[:, idx_sin_l]
    W0_l[:, phase_off_shared + 1] = W0[:, idx_cos_l]

    if args.mode == "avg":
        W0_shared = 0.5 * (W0_r + W0_l)
    elif args.mode == "copy_r":
        W0_shared = W0_r
    else:  # copy_l
        W0_shared = W0_l

    # Second layer: shared trunk (copy as-is).
    W1_shared = W1
    b1_shared = b1

    # Third layer: extract per-side joint rows, then merge.
    W2_r = torch.zeros((3 * K_side, hid), device=W2.device, dtype=W2.dtype)
    W2_l = torch.zeros((3 * K_side, hid), device=W2.device, dtype=W2.dtype)
    b2_r = torch.zeros((3 * K_side,), device=b2.device, dtype=b2.dtype)
    b2_l = torch.zeros((3 * K_side,), device=b2.device, dtype=b2.dtype)

    for i, p in enumerate(pos_r):
        src = slice(3 * int(p), 3 * int(p) + 3)
        dst = slice(3 * int(i), 3 * int(i) + 3)
        W2_r[dst, :] = W2[src, :]
        b2_r[dst] = b2[src]
    for i, p in enumerate(aligned_pos_l):
        src = slice(3 * int(p), 3 * int(p) + 3)
        dst = slice(3 * int(i), 3 * int(i) + 3)
        W2_l[dst, :] = W2[src, :]
        b2_l[dst] = b2[src]

    if args.mode == "avg":
        W2_shared = 0.5 * (W2_r + W2_l)
        b2_shared = 0.5 * (b2_r + b2_l)
    elif args.mode == "copy_r":
        W2_shared = W2_r
        b2_shared = b2_r
    else:  # copy_l
        W2_shared = W2_l
        b2_shared = b2_l

    # Inject tensors. Shared head is also a 3-linear Sequential at indices 0/3/6.
    sd = OrderedDict(sd)  # preserve deterministic key order on save
    sd["direct_pose_leg_head_shared.0.weight"] = W0_shared
    sd["direct_pose_leg_head_shared.0.bias"] = b0
    sd["direct_pose_leg_head_shared.3.weight"] = W1_shared
    sd["direct_pose_leg_head_shared.3.bias"] = b1_shared
    sd["direct_pose_leg_head_shared.6.weight"] = W2_shared
    sd["direct_pose_leg_head_shared.6.bias"] = b2_shared
    ckpt["model"] = sd

    # Lightweight provenance.
    ptcfg = ckpt.get("posttrain_cfg", None)
    if not isinstance(ptcfg, dict):
        ptcfg = {}
    ptcfg = dict(ptcfg)
    ptcfg["direct_pose_leg_side_routing"] = True
    ptcfg["direct_pose_leg_contact_order"] = contact_order
    ptcfg["_init_direct_pose_leg_head_shared_from_legacy_mode"] = str(args.mode)
    ckpt["posttrain_cfg"] = ptcfg

    ckpt_out = Path(args.ckpt_out)
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(ckpt_out))

    print(
        f"[ok] wrote {ckpt_out} (mode={args.mode}, contact_order={contact_order}, K={K}, K_side={K_side}, "
        f"legacy_in={legacy_in}, hid={hid})"
    )


if __name__ == "__main__":
    main()

