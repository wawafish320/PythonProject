#!/usr/bin/env python3
"""R3.0 smoke: compose-stable residual + head=0/no-op + seed2 replay."""

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


def _find_seed_ckpt(model_dir: Path, seed: int) -> Path:
    cands = sorted(model_dir.glob(f"*seed{seed}_e1.pth"))
    if not cands:
        raise FileNotFoundError(f"no seed ckpt found under {model_dir} for seed={seed}")
    if len(cands) > 1:
        names = ", ".join(p.name for p in cands)
        raise RuntimeError(f"multiple seed ckpts found under {model_dir} for seed={seed}: {names}")
    return cands[0]


def _make_zero_head_ckpt(src_ckpt: Path, dst_ckpt: Path, mode: str) -> Dict[str, int]:
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
        cfg["direct_pose_fusion_direct_mode"] = str(mode)
        obj["posttrain_cfg"] = cfg
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, dst_ckpt)
    return {"changed_tensors": int(changed)}


def _run_freerun(
    *,
    model_ckpt: Path,
    out_dir: Path,
    teacher: Path,
    bundle: Path,
    pretrain_template: Path,
    encoder_bundle: Path,
    mode: str,
    lambda_fusion_apply: bool,
    rounds: int,
    depth: int,
) -> Tuple[Path, Dict[str, object]]:
    cmd = [
        sys.executable,
        "-m",
        "train.validate.run_freerun_cycles",
        "--teacher",
        str(teacher),
        "--model",
        str(model_ckpt),
        "--bundle",
        str(bundle),
        "--pretrain-template",
        str(pretrain_template),
        "--encoder-bundle",
        str(encoder_bundle),
        "--out",
        str(out_dir),
        "--rounds",
        str(int(rounds)),
        "--depth",
        str(int(depth)),
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
    if bool(lambda_fusion_apply):
        cmd.append("--lambda_fusion_apply")
    _run(cmd)
    out_json = out_dir / "Walk_F_freerun_cycles.json"
    run_obj = json.loads(out_json.read_text())
    return out_json, run_obj


def _delta_stats(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {
        "mean": float(a["mean_deg"] - b["mean_deg"]),
        "p99": float(a["p99_deg"] - b["p99_deg"]),
        "max": float(a["max_deg"] - b["max_deg"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run stage6 R3.0 compose-stable residual smoke.")
    ap.add_argument("--run-tag", type=str, default="stage6_n1leg_v2_20260217")
    ap.add_argument("--no-op-seeds", type=str, default="0,1,2")
    ap.add_argument("--replay-seed", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    no_op_seeds = [int(x) for x in str(args.no_op_seeds).split(",") if str(x).strip()]
    if not no_op_seeds:
        raise SystemExit("[FATAL] --no-op-seeds is empty")

    root = Path(".").resolve()
    teacher = root / "validate" / "teacher_batches" / "Walk_F_teacher.json"
    bundle = root / "raw_data" / "processed_data" / "norm_template.json"
    pretrain_template = root / "models" / "pretrain_template.json"
    encoder_bundle = root / "models" / "motion_encoder_equiv_stageA.pt"

    bridge_model_dir = root / "models" / f"MLPL2_DirectBranch_v1__{args.run_tag}_step4_r2_residual_bridge_directpose"
    full_model_dir = root / "models" / f"MLPL2_DirectBranch_v1__{args.run_tag}_step4_r2_residual_full_directpose"
    zero_head_model_dir = root / "models" / f"MLPL2_DirectBranch_v1__{args.run_tag}_step4_r3_0_zerohead_compose_stable"

    base_out = root / "debug_output" / f"_{args.run_tag}" / "step4_r3_0_compose_stable_smoke"
    no_op_out = base_out / "no_op"
    replay_out = base_out / "seed2_replay"
    no_op_out.mkdir(parents=True, exist_ok=True)
    replay_out.mkdir(parents=True, exist_ok=True)
    zero_head_model_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": args.run_tag,
        "mask": {"cycle_gte": 1, "drop_wrap": True},
        "compose_mode": "residual_compose_stable",
        "no_op": {"rows": []},
        "seed2_replay": {},
    }

    # A) head=0/no-op smoke (compose-stable mode, lambda on/off).
    for seed in no_op_seeds:
        src_ckpt = _find_seed_ckpt(bridge_model_dir, seed)
        dst_ckpt = zero_head_model_dir / f"ckpt_last_nline_bridge_directpose_r3_0_zerohead_seed{seed}_e1.pth"
        zero_meta = _make_zero_head_ckpt(src_ckpt, dst_ckpt, mode="residual_compose_stable")

        seed_row: Dict[str, object] = {"seed": int(seed), "zero_head": zero_meta, "runs": {}}
        for lam_name, lam_flag in (("on", True), ("off", False)):
            out_dir = no_op_out / f"seed{seed}" / f"compose_stable_{lam_name}"
            out_json, run_obj = _run_freerun(
                model_ckpt=dst_ckpt,
                out_dir=out_dir,
                teacher=teacher,
                bundle=bundle,
                pretrain_template=pretrain_template,
                encoder_bundle=encoder_bundle,
                mode="residual_compose_stable",
                lambda_fusion_apply=lam_flag,
                rounds=int(args.rounds),
                depth=int(args.depth),
            )
            seed_row["runs"][lam_name] = {
                "out_json": str(out_json),
                "runtime_mode": run_obj.get("direct_pose_fusion_direct_mode", None),
                "runtime_lambda": bool(run_obj.get("lambda_fusion_apply", False)),
                "geo": _masked_stats(out_json, "GeoLocalDeg"),
                "blend": _masked_stats(out_json, "BlendGeoLocalDeg"),
            }
        on_obj = seed_row["runs"]["on"]
        off_obj = seed_row["runs"]["off"]
        seed_row["delta_on_minus_off"] = {
            "geo": _delta_stats(on_obj["geo"], off_obj["geo"]),
            "blend": _delta_stats(on_obj["blend"], off_obj["blend"]),
        }
        summary["no_op"]["rows"].append(seed_row)

    no_op_rows = summary["no_op"]["rows"]
    no_op_geo_max = [float(r["delta_on_minus_off"]["geo"]["max"]) for r in no_op_rows]
    no_op_blend_max = [float(r["delta_on_minus_off"]["blend"]["max"]) for r in no_op_rows]
    no_op_exact_zero = all(abs(v) < 1e-9 for v in (no_op_geo_max + no_op_blend_max))
    no_op_runtime_mode_ok = all(
        str(r["runs"]["on"].get("runtime_mode")) == "residual_compose_stable"
        and str(r["runs"]["off"].get("runtime_mode")) == "residual_compose_stable"
        for r in no_op_rows
    )
    summary["no_op"]["aggregate"] = {
        "geo_max_delta_deg": no_op_geo_max,
        "blend_max_delta_deg": no_op_blend_max,
        "g_no_op_exact_zero": bool(no_op_exact_zero),
        "g_no_op_runtime_mode_ok": bool(no_op_runtime_mode_ok),
    }

    # B) seed2 replay (full vs bridge), compare legacy residual vs compose-stable residual.
    seed = int(args.replay_seed)
    replay_rows: Dict[str, object] = {}
    for tag, mode in (
        ("legacy_residual_rot6d", "residual_rot6d"),
        ("compose_stable_residual", "residual_compose_stable"),
    ):
        full_ckpt = _find_seed_ckpt(full_model_dir, seed)
        bridge_ckpt = _find_seed_ckpt(bridge_model_dir, seed)
        mode_out = replay_out / f"seed{seed}" / tag
        full_json, full_obj = _run_freerun(
            model_ckpt=full_ckpt,
            out_dir=mode_out / "full",
            teacher=teacher,
            bundle=bundle,
            pretrain_template=pretrain_template,
            encoder_bundle=encoder_bundle,
            mode=mode,
            lambda_fusion_apply=True,
            rounds=int(args.rounds),
            depth=int(args.depth),
        )
        bridge_json, bridge_obj = _run_freerun(
            model_ckpt=bridge_ckpt,
            out_dir=mode_out / "bridge",
            teacher=teacher,
            bundle=bundle,
            pretrain_template=pretrain_template,
            encoder_bundle=encoder_bundle,
            mode=mode,
            lambda_fusion_apply=True,
            rounds=int(args.rounds),
            depth=int(args.depth),
        )
        full_metrics = {
            "direct": _masked_stats(full_json, "DirectGeoLocalDeg"),
            "rollout_geo": _masked_stats(full_json, "GeoLocalDeg"),
            "rollout_blend": _masked_stats(full_json, "BlendGeoLocalDeg"),
        }
        bridge_metrics = {
            "direct": _masked_stats(bridge_json, "DirectGeoLocalDeg"),
            "rollout_geo": _masked_stats(bridge_json, "GeoLocalDeg"),
            "rollout_blend": _masked_stats(bridge_json, "BlendGeoLocalDeg"),
        }
        replay_rows[tag] = {
            "mode": mode,
            "full": {
                "out_json": str(full_json),
                "runtime_mode": full_obj.get("direct_pose_fusion_direct_mode", None),
                "runtime_lambda": bool(full_obj.get("lambda_fusion_apply", False)),
                "metrics": full_metrics,
            },
            "bridge": {
                "out_json": str(bridge_json),
                "runtime_mode": bridge_obj.get("direct_pose_fusion_direct_mode", None),
                "runtime_lambda": bool(bridge_obj.get("lambda_fusion_apply", False)),
                "metrics": bridge_metrics,
            },
            "delta_full_minus_bridge": {
                "direct": _delta_stats(full_metrics["direct"], bridge_metrics["direct"]),
                "rollout_geo": _delta_stats(full_metrics["rollout_geo"], bridge_metrics["rollout_geo"]),
                "rollout_blend": _delta_stats(full_metrics["rollout_blend"], bridge_metrics["rollout_blend"]),
            },
        }

    legacy_delta = replay_rows["legacy_residual_rot6d"]["delta_full_minus_bridge"]
    stable_delta = replay_rows["compose_stable_residual"]["delta_full_minus_bridge"]
    replay_runtime_mode_ok = all(
        str(replay_rows[k]["full"]["runtime_mode"]) == replay_rows[k]["mode"]
        and str(replay_rows[k]["bridge"]["runtime_mode"]) == replay_rows[k]["mode"]
        for k in replay_rows
    )
    replay_lambda_ok = all(
        bool(replay_rows[k]["full"]["runtime_lambda"]) and bool(replay_rows[k]["bridge"]["runtime_lambda"])
        for k in replay_rows
    )
    summary["seed2_replay"] = {
        "seed": int(seed),
        "rows": replay_rows,
        "stable_minus_legacy_delta": {
            "direct_max": float(stable_delta["direct"]["max"] - legacy_delta["direct"]["max"]),
            "rollout_geo_max": float(stable_delta["rollout_geo"]["max"] - legacy_delta["rollout_geo"]["max"]),
            "rollout_blend_max": float(stable_delta["rollout_blend"]["max"] - legacy_delta["rollout_blend"]["max"]),
            "rollout_blend_p99": float(stable_delta["rollout_blend"]["p99"] - legacy_delta["rollout_blend"]["p99"]),
        },
        "g_seed2_runtime_mode_ok": bool(replay_runtime_mode_ok),
        "g_seed2_lambda_on_ok": bool(replay_lambda_ok),
    }

    gates = {
        "g1_no_op_exact_zero": bool(no_op_exact_zero),
        "g2_no_op_runtime_mode_ok": bool(no_op_runtime_mode_ok),
        "g3_seed2_runtime_mode_ok": bool(replay_runtime_mode_ok),
        "g4_seed2_lambda_on_ok": bool(replay_lambda_ok),
    }
    summary["gates"] = gates
    summary["decision"] = (
        "PASS_R3_0_COMPOSE_STABLE_SMOKE"
        if all(bool(v) for v in gates.values())
        else "HOLD_R3_0_COMPOSE_STABLE_SMOKE"
    )

    out_json = base_out / "step4_r3_0_compose_stable_smoke_summary.json"
    out_md = base_out / "step4_r3_0_compose_stable_smoke_summary.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    lines: List[str] = []
    lines.append("# Step4 R3.0 Compose-Stable Residual Smoke Summary")
    lines.append("")
    lines.append("- Goal A: `head=0` 下 `residual_compose_stable` 验证 no-op（`lambda on-off` 应为 exact-zero）。")
    lines.append("- Goal B: seed2 replay 对照 `legacy residual_rot6d` vs `compose_stable residual`。")
    lines.append("")
    lines.append("## A) head=0 / no-op")
    lines.append("")
    lines.append("| seed | compose-stable ΔGeo(max) | compose-stable ΔBlend(max) |")
    lines.append("|---:|---:|---:|")
    for r in no_op_rows:
        d = r["delta_on_minus_off"]
        lines.append(f"| {r['seed']} | {d['geo']['max']:+.6f} | {d['blend']['max']:+.6f} |")
    lines.append("")
    lines.append(f"- g1_no_op_exact_zero: {gates['g1_no_op_exact_zero']}")
    lines.append(f"- g2_no_op_runtime_mode_ok: {gates['g2_no_op_runtime_mode_ok']}")
    lines.append("")
    lines.append("## B) seed2 replay (full - bridge)")
    lines.append("")
    lines.append("| mode | direct Δmax | rollout Geo Δmax | rollout Blend Δp99 / Δmax |")
    lines.append("|---|---:|---:|---:|")
    for k in ("legacy_residual_rot6d", "compose_stable_residual"):
        d = replay_rows[k]["delta_full_minus_bridge"]
        lines.append(
            f"| {replay_rows[k]['mode']} | {d['direct']['max']:+.6f} | {d['rollout_geo']['max']:+.6f} | "
            f"{d['rollout_blend']['p99']:+.6f} / {d['rollout_blend']['max']:+.6f} |"
        )
    dd = summary["seed2_replay"]["stable_minus_legacy_delta"]
    lines.append("")
    lines.append(
        f"- stable-minus-legacy: direct Δmax {dd['direct_max']:+.6f}, "
        f"rollout_geo Δmax {dd['rollout_geo_max']:+.6f}, "
        f"rollout_blend Δp99/Δmax {dd['rollout_blend_p99']:+.6f}/{dd['rollout_blend_max']:+.6f}"
    )
    lines.append(f"- g3_seed2_runtime_mode_ok: {gates['g3_seed2_runtime_mode_ok']}")
    lines.append(f"- g4_seed2_lambda_on_ok: {gates['g4_seed2_lambda_on_ok']}")
    lines.append("")
    lines.append("| Gate | Rule | Result |")
    lines.append("|---|---|---|")
    lines.append("| g1_no_op_exact_zero | no-op `lambda on-off` (Geo/Blend max) == 0 for all no-op seeds | " + str(gates["g1_no_op_exact_zero"]) + " |")
    lines.append("| g2_no_op_runtime_mode_ok | no-op runs runtime mode == residual_compose_stable | " + str(gates["g2_no_op_runtime_mode_ok"]) + " |")
    lines.append("| g3_seed2_runtime_mode_ok | seed2 replay runtime mode matches requested mode for both arms | " + str(gates["g3_seed2_runtime_mode_ok"]) + " |")
    lines.append("| g4_seed2_lambda_on_ok | seed2 replay runtime lambda_fusion_apply=true for both arms | " + str(gates["g4_seed2_lambda_on_ok"]) + " |")
    lines.append("")
    lines.append(f"- decision: **{summary['decision']}**")
    out_md.write_text("\n".join(lines) + "\n")

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
