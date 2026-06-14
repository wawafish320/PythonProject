#!/usr/bin/env python3
from __future__ import annotations

"""Minimal AR/free-rollout commanded-yaw wiring probe for MinimalGoalAR.

Scope:
  - Diagnose ONLY AR/free-rollout per-step commanded yaw wiring.
  - No base training, no checkpoint dependency, no upstream latent injection.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.action_handoff_inbetween_model import MinimalGoalAR, ModelConfig  # noqa: E402
from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    EGO_VEL_SLICE,
    POSE_SLICE,
    STATE_DIM,
    YAW_RATE_SLICE,
)

VERDICT_CONNECTED_AND_READ = "AR_YAW_WIRING_CONNECTED_AND_READ"
VERDICT_CONNECTED_BUT_IGNORED = "AR_YAW_WIRING_CONNECTED_BUT_IGNORED"
VERDICT_NOT_CONNECTED = "AR_YAW_WIRING_NOT_CONNECTED"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal AR commanded-yaw wiring probe for MinimalGoalAR.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--seam-len", type=int, default=6)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--yaw-overwrite-tol", type=float, default=1e-7)
    p.add_argument("--body-read-eps", type=float, default=1e-5)
    return p


def _shape_dtype_device(x: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": [int(v) for v in x.shape],
        "dtype": str(x.dtype),
        "device": str(x.device),
    }


def _body_delta(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    return {
        "body_delta_pose_mean": float(torch.mean(torch.abs(a[..., POSE_SLICE] - b[..., POSE_SLICE])).item()),
        "body_delta_ego_mean": float(torch.mean(torch.abs(a[..., EGO_VEL_SLICE] - b[..., EGO_VEL_SLICE])).item()),
        "body_delta_contact_mean": float(
            torch.mean(torch.abs(a[..., CONTACT_SLICE] - b[..., CONTACT_SLICE])).item()
        ),
    }


def _max_body_delta(*items: Dict[str, float]) -> Dict[str, float]:
    keys = ("body_delta_pose_mean", "body_delta_ego_mean", "body_delta_contact_mean")
    return {k: float(max(float(d[k]) for d in items)) for k in keys}


def _decide_verdict(
    *,
    finite_ok: bool,
    yaw_overwrite_max_abs: float,
    yaw_tol: float,
    max_body_delta: Dict[str, float],
    body_read_eps: float,
) -> str:
    if not finite_ok:
        return VERDICT_INCONCLUSIVE
    if not np.isfinite(yaw_overwrite_max_abs):
        return VERDICT_INCONCLUSIVE
    if float(yaw_overwrite_max_abs) > float(yaw_tol):
        return VERDICT_NOT_CONNECTED
    read = any(float(max_body_delta[k]) > float(body_read_eps) for k in max_body_delta)
    if read:
        return VERDICT_CONNECTED_AND_READ
    return VERDICT_CONNECTED_BUT_IGNORED


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_ar_commanded_yaw_wiring_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    b = int(args.batch)
    c = int(args.context_len)
    k = int(args.seam_len)
    h = int(args.horizon)
    d = int(STATE_DIM)

    model = MinimalGoalAR(ModelConfig(state_dim=d, seam_len=k, hidden=int(args.hidden)))
    model.eval()

    ctx = torch.randn(b, c, d, dtype=torch.float32)
    goal = torch.randn(b, k, d, dtype=torch.float32)

    t = torch.linspace(-1.0, 1.0, h, dtype=torch.float32).view(1, h, 1)
    scale = torch.linspace(0.5, 1.1, b, dtype=torch.float32).view(b, 1, 1)
    cmd_target = t.repeat(b, 1, 1) * scale
    cmd_zero = torch.zeros_like(cmd_target)
    cmd_neg = -cmd_target

    with torch.no_grad():
        out_plain = model.rollout_free(ctx, goal, horizon=h)
        out_target = model.rollout_free_commanded_yaw(ctx, goal, cmd_target)
        out_zero = model.rollout_free_commanded_yaw(ctx, goal, cmd_zero)
        out_neg = model.rollout_free_commanded_yaw(ctx, goal, cmd_neg)

    finite_ok = bool(
        torch.isfinite(out_plain).all()
        and torch.isfinite(out_target).all()
        and torch.isfinite(out_zero).all()
        and torch.isfinite(out_neg).all()
    )

    yaw_overwrite_max_abs = float(
        torch.max(torch.abs(out_target[..., YAW_RATE_SLICE] - cmd_target)).item()
    )
    plain_yaw_cmd_target_mae = float(
        torch.mean(torch.abs(out_plain[..., YAW_RATE_SLICE] - cmd_target)).item()
    )
    target_vs_zero = _body_delta(out_target, out_zero)
    target_vs_neg = _body_delta(out_target, out_neg)
    max_body_delta = _max_body_delta(target_vs_zero, target_vs_neg)

    verdict = _decide_verdict(
        finite_ok=finite_ok,
        yaw_overwrite_max_abs=yaw_overwrite_max_abs,
        yaw_tol=float(args.yaw_overwrite_tol),
        max_body_delta=max_body_delta,
        body_read_eps=float(args.body_read_eps),
    )

    summary: Dict[str, Any] = {
        "task": "MinimalGoalAR AR/free-rollout commanded-yaw wiring probe",
        "scope": "AR/free-rollout only; masked-conditioned path is out-of-scope",
        "config": {
            "seed": int(args.seed),
            "batch": b,
            "context_len": c,
            "seam_len": k,
            "horizon": h,
            "state_dim": d,
            "hidden": int(args.hidden),
            "yaw_overwrite_tol": float(args.yaw_overwrite_tol),
            "body_read_eps": float(args.body_read_eps),
        },
        "shapes_dtypes_devices": {
            "ctx": _shape_dtype_device(ctx),
            "goal": _shape_dtype_device(goal),
            "cmd_yaw": _shape_dtype_device(cmd_target),
            "out": _shape_dtype_device(out_target),
        },
        "checks": {
            "finite_ok": finite_ok,
            "yaw_overwrite_max_abs": yaw_overwrite_max_abs,
            "yaw_overwrite_connected": bool(yaw_overwrite_max_abs <= float(args.yaw_overwrite_tol)),
            "plain_rollout_yaw_cmd_target_mae": plain_yaw_cmd_target_mae,
        },
        "body_sensitivity": {
            "target_vs_zero": target_vs_zero,
            "target_vs_neg_target": target_vs_neg,
            "max_body_delta": max_body_delta,
        },
        "verdict": verdict,
    }

    json_path = out_dir / "ar_commanded_yaw_wiring_probe_summary.json"
    md_path = out_dir / "ar_commanded_yaw_wiring_probe_summary.md"
    _dump_json(json_path, summary)

    lines: list[str] = []
    lines.append("# AR Commanded-Yaw Wiring Probe (MinimalGoalAR)")
    lines.append("")
    lines.append("- scope: AR/free-rollout only; masked-conditioned path is out-of-scope")
    lines.append(
        f"- shapes: ctx={summary['shapes_dtypes_devices']['ctx']['shape']}, "
        f"goal={summary['shapes_dtypes_devices']['goal']['shape']}, "
        f"cmd_yaw={summary['shapes_dtypes_devices']['cmd_yaw']['shape']}, "
        f"out={summary['shapes_dtypes_devices']['out']['shape']}"
    )
    lines.append(
        f"- dtypes/devices: ctx={summary['shapes_dtypes_devices']['ctx']['dtype']}@{summary['shapes_dtypes_devices']['ctx']['device']}, "
        f"goal={summary['shapes_dtypes_devices']['goal']['dtype']}@{summary['shapes_dtypes_devices']['goal']['device']}, "
        f"cmd_yaw={summary['shapes_dtypes_devices']['cmd_yaw']['dtype']}@{summary['shapes_dtypes_devices']['cmd_yaw']['device']}, "
        f"out={summary['shapes_dtypes_devices']['out']['dtype']}@{summary['shapes_dtypes_devices']['out']['device']}"
    )
    lines.append(
        f"- yaw overwrite: max_abs={yaw_overwrite_max_abs:.6e} "
        f"(tol={float(args.yaw_overwrite_tol):.2e}, pass={summary['checks']['yaw_overwrite_connected']})"
    )
    lines.append(
        f"- plain rollout yaw-vs-cmd MAE (diagnostic): {plain_yaw_cmd_target_mae:.6e}"
    )
    lines.append(
        "- body sensitivity target_vs_zero: "
        f"pose={target_vs_zero['body_delta_pose_mean']:.6e}, "
        f"ego={target_vs_zero['body_delta_ego_mean']:.6e}, "
        f"contact={target_vs_zero['body_delta_contact_mean']:.6e}"
    )
    lines.append(
        "- body sensitivity target_vs_neg_target: "
        f"pose={target_vs_neg['body_delta_pose_mean']:.6e}, "
        f"ego={target_vs_neg['body_delta_ego_mean']:.6e}, "
        f"contact={target_vs_neg['body_delta_contact_mean']:.6e}"
    )
    lines.append(f"- verdict: **{verdict}**")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {json_path.resolve()}")
    lines.append(f"- {md_path.resolve()}")
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(
        f"[probe] verdict={verdict} yaw_max_abs={yaw_overwrite_max_abs:.3e} "
        f"plain_yaw_mae={plain_yaw_cmd_target_mae:.3e}"
    )


if __name__ == "__main__":
    main()
