#!/usr/bin/env python3
"""R1 smoke for direct 1+x residual fusion (head=0 safety check)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _q(vals: List[float], q: float) -> float:
    arr = sorted(v for v in vals if math.isfinite(v))
    if not arr:
        return float("nan")
    i = int(round(max(0.0, min(1.0, float(q))) * (len(arr) - 1)))
    return float(arr[i])


def _masked_stats(cycles_json: Path, key: str) -> Dict[str, float]:
    obj = json.loads(cycles_json.read_text())
    vals: List[float] = []
    for st in obj.get("metrics_per_step", []):
        cyc = int(st.get("cycle", 0) or 0)
        if cyc < 1:
            continue
        if bool(st.get("wrap_boundary_step", False)):
            continue
        v = float(st.get(key, float("nan")))
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return {"n": 0.0, "mean_deg": float("nan"), "p99_deg": float("nan"), "max_deg": float("nan")}
    return {
        "n": float(len(vals)),
        "mean_deg": float(sum(vals) / len(vals)),
        "p99_deg": _q(vals, 0.99),
        "max_deg": float(max(vals)),
    }


def _run(cmd: List[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _avg(vals: List[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _make_zero_head_ckpt(src_ckpt: Path, dst_ckpt: Path) -> Dict[str, int]:
    obj = torch.load(src_ckpt, map_location="cpu")
    model_sd = obj.get("model", None)
    if not isinstance(model_sd, dict):
        raise RuntimeError(f"checkpoint has no model state dict: {src_ckpt}")
    changed = 0
    for k, v in list(model_sd.items()):
        if not (isinstance(k, str) and k.startswith("direct_pose_head.")):
            continue
        if torch.is_tensor(v):
            model_sd[k] = torch.zeros_like(v)
            changed += 1
    if changed == 0:
        raise RuntimeError(f"no direct_pose_head.* tensors found in {src_ckpt}")
    obj["model"] = model_sd
    cfg = obj.get("posttrain_cfg", {})
    if isinstance(cfg, dict):
        cfg = dict(cfg)
        cfg["direct_pose_fusion_direct_mode"] = "residual_rot6d"
        obj["posttrain_cfg"] = cfg
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, dst_ckpt)
    return {"changed_tensors": int(changed)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run R1 residual smoke (zero-head safety check).")
    ap.add_argument("--run-tag", type=str, default="stage6_n1leg_v2_20260217")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()

    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    if not seeds:
        raise SystemExit("[FATAL] --seeds is empty")

    root = Path(".").resolve()
    base_out = root / "debug_output" / f"_{args.run_tag}" / "step4_r1_residual_smoke"
    base_out.mkdir(parents=True, exist_ok=True)
    src_model_dir = root / "models" / f"MLPL2_DirectBranch_v1__{args.run_tag}_step3z_fixa_bridge_directpose"
    dst_model_dir = root / "models" / f"MLPL2_DirectBranch_v1__{args.run_tag}_step4_r1_zerohead_bridge"
    dst_model_dir.mkdir(parents=True, exist_ok=True)

    teacher = root / "validate" / "teacher_batches" / "Walk_F_teacher.json"
    bundle = root / "raw_data" / "processed_data" / "norm_template.json"
    pretrain_template = root / "models" / "pretrain_template.json"
    encoder_bundle = root / "models" / "motion_encoder_equiv_stageA.pt"

    conds: List[Tuple[str, str, bool]] = [
        ("absolute_on", "absolute", True),
        ("absolute_off", "absolute", False),
        ("residual_on", "residual_rot6d", True),
        ("residual_off", "residual_rot6d", False),
    ]

    summary: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": args.run_tag,
        "mask": {"cycle_gte": 1, "drop_wrap": True},
        "source_model_dir": str(src_model_dir),
        "zero_head_model_dir": str(dst_model_dir),
        "rows": [],
    }

    for seed in seeds:
        src_ckpt = src_model_dir / f"ckpt_last_nline_bridge_directpose_fixa_r2_budget_seed{seed}_e1.pth"
        if not src_ckpt.is_file():
            raise SystemExit(f"[FATAL] missing source ckpt: {src_ckpt}")
        dst_ckpt = dst_model_dir / f"ckpt_last_nline_bridge_directpose_r1_zerohead_seed{seed}_e1.pth"
        ckpt_meta = _make_zero_head_ckpt(src_ckpt, dst_ckpt)

        seed_row: Dict[str, object] = {"seed": int(seed), "zero_head": ckpt_meta, "conds": {}}
        for cond_name, mode, lam in conds:
            out_dir = base_out / f"seed{seed}" / cond_name
            cmd = [
                sys.executable,
                "-m",
                "train.validate.run_freerun_cycles",
                "--teacher",
                str(teacher),
                "--model",
                str(dst_ckpt),
                "--bundle",
                str(bundle),
                "--pretrain-template",
                str(pretrain_template),
                "--encoder-bundle",
                str(encoder_bundle),
                "--out",
                str(out_dir),
                "--rounds",
                "5",
                "--depth",
                "3",
                "--time-index-mode",
                "auto",
                "--phase_reset_source",
                "none",
                "--phase_reset_source_strict",
                "on",
                "--so3_corr_apply",
                "--direct_pose_fusion_direct_mode",
                str(mode),
                "--log_contacts",
                "--export_joint_direct_geolocal_series",
                "--export_joint_so3_error_series",
                "--joint_so3_error_series_branches",
                "direct",
                "--joint_so3_error_series_space",
                "body",
                "--force",
            ]
            if lam:
                cmd.append("--lambda_fusion_apply")
            _run(cmd)
            out_json = out_dir / "Walk_F_freerun_cycles.json"
            geo = _masked_stats(out_json, "GeoLocalDeg")
            blend = _masked_stats(out_json, "BlendGeoLocalDeg")
            run_obj = json.loads(out_json.read_text())
            seed_row["conds"][cond_name] = {
                "mode": str(mode),
                "lambda_fusion_apply": bool(lam),
                "out_json": str(out_json),
                "runtime_mode": run_obj.get("direct_pose_fusion_direct_mode", None),
                "runtime_lambda": bool(run_obj.get("lambda_fusion_apply", False)),
                "geo": geo,
                "blend": blend,
            }

        c = seed_row["conds"]
        d_abs = {
            "geo": {
                "mean": float(c["absolute_on"]["geo"]["mean_deg"] - c["absolute_off"]["geo"]["mean_deg"]),
                "p99": float(c["absolute_on"]["geo"]["p99_deg"] - c["absolute_off"]["geo"]["p99_deg"]),
                "max": float(c["absolute_on"]["geo"]["max_deg"] - c["absolute_off"]["geo"]["max_deg"]),
            },
            "blend": {
                "mean": float(c["absolute_on"]["blend"]["mean_deg"] - c["absolute_off"]["blend"]["mean_deg"]),
                "p99": float(c["absolute_on"]["blend"]["p99_deg"] - c["absolute_off"]["blend"]["p99_deg"]),
                "max": float(c["absolute_on"]["blend"]["max_deg"] - c["absolute_off"]["blend"]["max_deg"]),
            },
        }
        d_res = {
            "geo": {
                "mean": float(c["residual_on"]["geo"]["mean_deg"] - c["residual_off"]["geo"]["mean_deg"]),
                "p99": float(c["residual_on"]["geo"]["p99_deg"] - c["residual_off"]["geo"]["p99_deg"]),
                "max": float(c["residual_on"]["geo"]["max_deg"] - c["residual_off"]["geo"]["max_deg"]),
            },
            "blend": {
                "mean": float(c["residual_on"]["blend"]["mean_deg"] - c["residual_off"]["blend"]["mean_deg"]),
                "p99": float(c["residual_on"]["blend"]["p99_deg"] - c["residual_off"]["blend"]["p99_deg"]),
                "max": float(c["residual_on"]["blend"]["max_deg"] - c["residual_off"]["blend"]["max_deg"]),
            },
        }
        seed_row["delta_on_minus_off"] = {"absolute": d_abs, "residual": d_res}
        summary["rows"].append(seed_row)

    rows = summary["rows"]
    abs_geo_max = [float(r["delta_on_minus_off"]["absolute"]["geo"]["max"]) for r in rows]
    abs_blend_max = [float(r["delta_on_minus_off"]["absolute"]["blend"]["max"]) for r in rows]
    res_geo_max = [float(r["delta_on_minus_off"]["residual"]["geo"]["max"]) for r in rows]
    res_blend_max = [float(r["delta_on_minus_off"]["residual"]["blend"]["max"]) for r in rows]

    residual_zero_3of3 = all(abs(v) < 1e-9 for v in res_geo_max + res_blend_max)
    absolute_nonzero_3of3 = all((abs(v) > 1e-6) for v in abs_geo_max + abs_blend_max)

    summary["aggregate"] = {
        "absolute_on_minus_off_geo_max_avg": _avg(abs_geo_max),
        "absolute_on_minus_off_blend_max_avg": _avg(abs_blend_max),
        "residual_on_minus_off_geo_max_avg": _avg(res_geo_max),
        "residual_on_minus_off_blend_max_avg": _avg(res_blend_max),
        "g1_residual_safe_zero_3of3": bool(residual_zero_3of3),
        "g2_absolute_nonzero_3of3": bool(absolute_nonzero_3of3),
    }
    summary["decision"] = "PASS_R1_SMOKE" if (residual_zero_3of3 and absolute_nonzero_3of3) else "HOLD_R1_SMOKE"

    out_json = base_out / "step4_r1_residual_smoke_summary.json"
    out_md = base_out / "step4_r1_residual_smoke_summary.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    lines: List[str] = []
    lines.append("# Step4 R1 Residual Smoke Summary")
    lines.append("")
    lines.append("- Goal: `head=0` 下验证 residual direct (`residual_rot6d`) 在 `lambda on` 不伤 incremental baseline。")
    lines.append("- Protocol: zero `direct_pose_head.*`, compare `lambda on-off` under two modes (`absolute` vs `residual_rot6d`).")
    lines.append("")
    lines.append("| seed | absolute ΔGeo(max) | absolute ΔBlend(max) | residual ΔGeo(max) | residual ΔBlend(max) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in rows:
        da = r["delta_on_minus_off"]["absolute"]
        dr = r["delta_on_minus_off"]["residual"]
        lines.append(
            f"| {r['seed']} | {da['geo']['max']:+.6f} | {da['blend']['max']:+.6f} | "
            f"{dr['geo']['max']:+.6f} | {dr['blend']['max']:+.6f} |"
        )
    lines.append("")
    agg = summary["aggregate"]
    lines.append(f"- absolute on-off avg ΔGeo(max): {agg['absolute_on_minus_off_geo_max_avg']:+.6f}")
    lines.append(f"- absolute on-off avg ΔBlend(max): {agg['absolute_on_minus_off_blend_max_avg']:+.6f}")
    lines.append(f"- residual on-off avg ΔGeo(max): {agg['residual_on_minus_off_geo_max_avg']:+.6f}")
    lines.append(f"- residual on-off avg ΔBlend(max): {agg['residual_on_minus_off_blend_max_avg']:+.6f}")
    lines.append("")
    lines.append("| Gate | Rule | Result |")
    lines.append("|---|---|---|")
    lines.append(
        f"| g1_residual_safe_zero_3of3 | residual mode 下 `lambda on-off == 0` (Geo/Blend max, 3/3) | {agg['g1_residual_safe_zero_3of3']} |"
    )
    lines.append(
        f"| g2_absolute_nonzero_3of3 | absolute mode 下 `lambda on-off != 0` (Geo/Blend max, 3/3) | {agg['g2_absolute_nonzero_3of3']} |"
    )
    lines.append("")
    lines.append(f"- decision: **{summary['decision']}**")
    out_md.write_text("\n".join(lines) + "\n")

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
