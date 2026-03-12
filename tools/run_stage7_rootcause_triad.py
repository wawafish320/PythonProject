#!/usr/bin/env python3
"""
Stage7 root-cause triad runner (seed-parallel experiment scaffold).

For each seed, run three probes:
1) stage6_step0: diagnose on Stage6 ckpt (no training)
2) resume_eN:    resume Stage6 ckpt, train N epochs, then diagnose
3) random_eN:    random init, train N epochs, then diagnose

Outputs:
- debug_output/<out_dir>/seed*/{stage6_step0,resume_eN,random_eN}/sampling_grad_closure.json
- debug_output/<out_dir>/summary.json + summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]


def _resolve_from_root(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (_ROOT / p)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for tok in str(spec or "").split(","):
        s = tok.strip()
        if not s:
            continue
        v = int(s)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    if not out:
        raise ValueError("empty integer list")
    return out


def _run_and_tee(cmd: List[str], *, cwd: Path, env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd]\n")
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = int(proc.wait())
    if rc != 0:
        raise SystemExit(f"[FATAL] command failed (exit={rc}): {' '.join(cmd)} (log={log_path})")


def _pick_ckpt(out_dir: Path, run_name: str) -> Path:
    run_dir = out_dir / run_name
    cands = [
        run_dir / f"ckpt_last_{run_name}.pth",
        run_dir / f"ckpt_best_free_{run_name}.pth",
        run_dir / f"ckpt_best_teacher_{run_name}.pth",
    ]
    for p in cands:
        if p.is_file():
            return p
    raise FileNotFoundError(f"missing checkpoint for run={run_name}; tried={cands}")


def _mean_std_ci(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([_safe_float(v) for v in vals], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if arr.size >= 2:
        sem = float(std / math.sqrt(float(arr.size)))
        ci = 1.96 * sem
        ci_low, ci_high = float(mean - ci), float(mean + ci)
    else:
        ci_low, ci_high = mean, mean
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def _extract_root_metrics(diag_json: Path) -> Dict[str, Any]:
    payload = _load_json(diag_json)
    root = payload.get("root_cause_probe", {}) if isinstance(payload, dict) else {}
    sym = root.get("direct_head_symmetry", {}) if isinstance(root, dict) else {}
    cond_side = root.get("cond_in_side_stats", {}) if isinstance(root, dict) else {}
    dyn = root.get("dynamics_probe_step0", {}) if isinstance(root, dict) else {}
    dyn_ratio = dyn.get("probe_ratio_r_over_l", {}) if isinstance(dyn, dict) else {}
    return {
        "out_direct_ratio_step0": _safe_float(root.get("out_direct_ratio_step0", float("nan"))),
        "cond_in_ratio_step0": _safe_float(root.get("cond_in_ratio_step0", float("nan"))),
        "ratio_gap_step0": _safe_float(root.get("ratio_gap_step0", float("nan"))),
        "direct_pose_loss_ratio_step0": _safe_float(root.get("direct_pose_loss_ratio_step0", float("nan"))),
        "direct_pose_loss_log_ratio_step0": _safe_float(root.get("direct_pose_loss_log_ratio_step0", float("nan"))),
        "cond_in_side_mean_abs_z": _safe_float(cond_side.get("mean_abs_z", float("nan"))),
        "cond_in_side_max_abs_z": _safe_float(cond_side.get("max_abs_z", float("nan"))),
        "cond_in_side_status": str(cond_side.get("status", "unknown")),
        "head_weight_rel_l2_best_sign": _safe_float(sym.get("weight_rel_l2_best_sign", float("nan"))),
        "head_bias_rel_l2_best_sign": _safe_float(sym.get("bias_rel_l2_best_sign", float("nan"))),
        "head_weight_norm_ratio_r_over_l": _safe_float(sym.get("weight_norm_ratio_r_over_l", float("nan"))),
        "head_status": str(sym.get("status", "unknown")),
        "head_layer": str(sym.get("layer", "")),
        "dyn_ratio_shared_encoder_pre_act": _safe_float(dyn_ratio.get("shared_encoder_pre_act", float("nan"))),
        "dyn_ratio_temporal_pre_pasa": _safe_float(dyn_ratio.get("temporal_pre_pasa", float("nan"))),
        "dyn_ratio_direct_head_pre_out": _safe_float(dyn_ratio.get("direct_head_pre_out", float("nan"))),
        "dyn_amp_log_shared_to_temporal": _safe_float(dyn.get("amp_log_shared_to_temporal", float("nan"))),
        "dyn_amp_log_temporal_to_direct_head_pre_out": _safe_float(
            dyn.get("amp_log_temporal_to_direct_head_pre_out", float("nan"))
        ),
        "dyn_amp_log_direct_head_pre_out_to_out_direct": _safe_float(
            dyn.get("amp_log_direct_head_pre_out_to_out_direct", float("nan"))
        ),
        "dyn_amp_log_shared_to_out_direct": _safe_float(dyn.get("amp_log_shared_to_out_direct", float("nan"))),
    }


def _extract_jacobian_metrics(jac_json: Path) -> Dict[str, Any]:
    if not jac_json.is_file():
        return {
            "jac_lr_ratio_all_mean": float("nan"),
            "jac_lr_ratio_all_median": float("nan"),
            "jac_lr_ratio_all_std": float("nan"),
            "jac_lr_ratio_local_mean": float("nan"),
            "jac_lr_ratio_local_median": float("nan"),
            "jac_lr_ratio_local_std": float("nan"),
        }
    payload = _load_json(jac_json)
    agg = payload.get("aggregate", {}) if isinstance(payload, dict) else {}
    all_obj = agg.get("ratio_r_over_l_all", {}) if isinstance(agg, dict) else {}
    loc_obj = agg.get("ratio_r_over_l_local_t", {}) if isinstance(agg, dict) else {}
    return {
        "jac_lr_ratio_all_mean": _safe_float(all_obj.get("mean", float("nan"))),
        "jac_lr_ratio_all_median": _safe_float(all_obj.get("median", float("nan"))),
        "jac_lr_ratio_all_std": _safe_float(all_obj.get("std", float("nan"))),
        "jac_lr_ratio_local_mean": _safe_float(loc_obj.get("mean", float("nan"))),
        "jac_lr_ratio_local_median": _safe_float(loc_obj.get("median", float("nan"))),
        "jac_lr_ratio_local_std": _safe_float(loc_obj.get("std", float("nan"))),
    }


def _diag_cmd(
    *,
    python: str,
    config_json: Path,
    ckpt: Path,
    target_clip: str,
    depth: int,
    bundle: Path,
    pretrain_template: Path,
    encoder_bundle: Path,
    loss_branch: str,
    component_losses: str,
    device: str,
    out_dir: Path,
    max_windows: int,
) -> List[str]:
    cmd = [
        python,
        str(_ROOT / "tools" / "diagnose_stage7_sampling_grad_closure.py"),
        "--config-json",
        str(config_json),
        "--ckpt",
        str(ckpt),
        "--target-clip",
        str(target_clip),
        "--depth",
        str(int(depth)),
        "--bundle",
        str(bundle),
        "--pretrain-template",
        str(pretrain_template),
        "--encoder-bundle",
        str(encoder_bundle),
        "--loss-branch",
        str(loss_branch),
        "--component-losses",
        str(component_losses),
        "--device",
        str(device),
        "--out-dir",
        str(out_dir),
    ]
    if int(max_windows) > 0:
        cmd += ["--max-windows", str(int(max_windows))]
    return cmd


def _jac_probe_cmd(
    *,
    python: str,
    teacher: Path,
    ckpt: Path,
    bundle: Path,
    pretrain_template: Path,
    encoder_bundle: Path,
    npz_root: Path,
    depth: int,
    device: str,
    steps: str,
    left_bones: str,
    right_bones: str,
    out_json: Path,
) -> List[str]:
    return [
        python,
        str(_ROOT / "tools" / "diagnose_direct_head_lr_jacobian.py"),
        "--teacher",
        str(teacher),
        "--model",
        str(ckpt),
        "--bundle",
        str(bundle),
        "--pretrain-template",
        str(pretrain_template),
        "--encoder-bundle",
        str(encoder_bundle),
        "--npz-root",
        str(npz_root),
        "--depth",
        str(int(depth)),
        "--device",
        str(device),
        "--steps",
        str(steps),
        "--left-bones",
        str(left_bones),
        "--right-bones",
        str(right_bones),
        "--out",
        str(out_json),
    ]


def _train_cmd(
    *,
    python: str,
    config_json: Path,
    out_model_dir: Path,
    run_name: str,
    seed: int,
    epochs: int,
    dataset_index_mode: str,
    resume_ckpt: Optional[Path],
    extra_overrides: Sequence[str],
) -> List[str]:
    cmd = [
        python,
        "-m",
        "train.training_MPL",
        "--config_json",
        str(config_json),
        "--out",
        str(out_model_dir),
        "--run_name",
        str(run_name),
        "--seed",
        str(int(seed)),
        "--config_override",
        f"dataset_index_mode={dataset_index_mode}",
        "--config_override",
        f"epochs={int(epochs)}",
    ]
    if resume_ckpt is not None:
        cmd += ["--resume", str(resume_ckpt)]
    for ov in extra_overrides:
        txt = str(ov).strip()
        if txt:
            cmd += ["--config_override", txt]
    return cmd


@dataclass
class VerdictThresholds:
    out_ratio_high: float
    delta_resume_vs_step0: float
    head_w_rel_l2_high: float


def _verdict(agg: Mapping[str, Any], thr: VerdictThresholds, *, resume_arm: str) -> Dict[str, Any]:
    stage6 = agg.get("stage6_step0", {})
    resume = agg.get(str(resume_arm), {})
    paired = agg.get("paired", {})

    def _m(obj: Any, key: str) -> float:
        if not isinstance(obj, Mapping):
            return float("nan")
        v = obj.get(key, {})
        if isinstance(v, Mapping):
            return _safe_float(v.get("mean", float("nan")))
        return float("nan")

    stage6_out = _m(stage6, "out_direct_ratio_step0")
    stage6_head = _m(stage6, "head_weight_rel_l2_best_sign")
    delta_resume = _m(paired.get("resume_minus_stage6", {}), "out_direct_ratio_step0")
    resume_out = _m(resume, "out_direct_ratio_step0")

    flags = {
        "inheritance_bias_present": bool(math.isfinite(stage6_out) and stage6_out >= float(thr.out_ratio_high)),
        "head_asymmetry_present": bool(math.isfinite(stage6_head) and stage6_head >= float(thr.head_w_rel_l2_high)),
        "dynamics_amplification_present": bool(
            math.isfinite(delta_resume) and delta_resume >= float(thr.delta_resume_vs_step0)
        ),
        "resume_bias_high": bool(math.isfinite(resume_out) and resume_out >= float(thr.out_ratio_high)),
    }

    if flags["inheritance_bias_present"] and flags["head_asymmetry_present"] and flags["dynamics_amplification_present"]:
        primary = "mixed: inherited output-path asymmetry + training dynamics amplification"
    elif flags["inheritance_bias_present"] and flags["head_asymmetry_present"]:
        primary = "inheritance-dominant with direct-head parameter asymmetry"
    elif flags["dynamics_amplification_present"]:
        primary = "dynamics-dominant amplification"
    elif flags["head_asymmetry_present"]:
        primary = "parameter-asymmetry-dominant"
    else:
        primary = "inconclusive"

    return {"primary": primary, "flags": flags}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Stage7 root-cause triad experiments (step0 / resume_eN / random_eN) across seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config-json", type=str, default="config/exp_phase_DirectBranch_v1_d1_noreset.json")
    ap.add_argument(
        "--resume-ckpt",
        type=str,
        default="models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage6_direct_cond_anchor_20260124.pth",
    )
    ap.add_argument("--out-model-dir", type=str, default="models/MLPL2_DirectBranch_v1__pipe_20260215_rootcause_triad")
    ap.add_argument("--base-run-name", type=str, default="rootcause_triad")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--dataset-index-mode", type=str, default="sliding")
    ap.add_argument("--train-config-override", action="append", default=[])

    ap.add_argument("--target-clip", type=str, default="Walk_F")
    ap.add_argument(
        "--teacher",
        type=str,
        default="",
        help='Optional teacher JSON; default resolves to validate/teacher_batches/<target_clip>_teacher.json',
    )
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--loss-branch", type=str, default="out", choices=("out", "out_direct"))
    ap.add_argument("--component-losses", type=str, default="rot_geo,rot_vel,direct_pose,direct_delta")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--jac-steps", type=str, default="0,1,2,10,20")
    ap.add_argument("--jac-left-bones", type=str, default="thigh_l,calf_l,foot_l,ball_l")
    ap.add_argument("--jac-right-bones", type=str, default="thigh_r,calf_r,foot_r,ball_r")
    ap.add_argument("--skip-jacobian-probe", action="store_true")

    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-step0", action="store_true")
    ap.add_argument("--skip-resume", action="store_true")
    ap.add_argument("--skip-random", action="store_true")
    ap.add_argument("--out-dir", type=str, default=None)

    ap.add_argument("--thr-out-ratio-high", type=float, default=1.30)
    ap.add_argument("--thr-delta-resume-vs-step0", type=float, default=0.15)
    ap.add_argument("--thr-head-w-rel-l2-high", type=float, default=0.50)
    args = ap.parse_args()

    cfg_path = _resolve_from_root(args.config_json)
    if not cfg_path.is_file():
        raise SystemExit(f"[FATAL] config not found: {cfg_path}")
    resume_ckpt = _resolve_from_root(args.resume_ckpt)
    if not resume_ckpt.is_file():
        raise SystemExit(f"[FATAL] resume ckpt not found: {resume_ckpt}")

    seeds = _parse_int_list(args.seeds)
    if int(args.epochs) < 1:
        raise SystemExit("[FATAL] --epochs must be >=1")

    out_model_dir = _resolve_from_root(args.out_model_dir)
    if args.out_dir:
        out_dir = _resolve_from_root(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _ROOT / "debug_output" / f"_p1_rootcause_triad_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_json(cfg_path)
    cfg_effective = dict(cfg)
    cfg_effective["dataset_index_mode"] = str(args.dataset_index_mode)
    cfg_effective_path = out_dir / "config_effective.json"
    _write_json(cfg_effective_path, cfg_effective)

    bundle = _resolve_from_root(args.bundle)
    pretrain_template = _resolve_from_root(args.pretrain_template)
    encoder_bundle = _resolve_from_root(args.encoder_bundle)
    npz_root = _resolve_from_root(args.npz_root)
    if args.teacher:
        teacher = _resolve_from_root(args.teacher)
    else:
        teacher = _resolve_from_root(f"validate/teacher_batches/{args.target_clip}_teacher.json")
    if not teacher.is_file():
        raise SystemExit(f"[FATAL] teacher json not found: {teacher}")

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(_ROOT))
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

    resume_arm = f"resume_e{int(args.epochs)}"
    random_arm = f"random_e{int(args.epochs)}"

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = out_dir / f"seed{int(seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        rec: Dict[str, Any] = {
            "seed": int(seed),
            "arms": {},
        }

        if not bool(args.skip_step0):
            arm_dir = seed_dir / "stage6_step0"
            cmd = _diag_cmd(
                python=sys.executable,
                config_json=cfg_effective_path,
                ckpt=resume_ckpt,
                target_clip=str(args.target_clip),
                depth=int(args.depth),
                bundle=bundle,
                pretrain_template=pretrain_template,
                encoder_bundle=encoder_bundle,
                loss_branch=str(args.loss_branch),
                component_losses=str(args.component_losses),
                device=str(args.device),
                out_dir=arm_dir,
                max_windows=int(args.max_windows),
            )
            _run_and_tee(cmd, cwd=_ROOT, env=env, log_path=seed_dir / "stage6_step0.log")
            diag_json = arm_dir / "sampling_grad_closure.json"
            jac_json = arm_dir / "direct_head_lr_jacobian.json"
            if not bool(args.skip_jacobian_probe):
                cmd_jac = _jac_probe_cmd(
                    python=sys.executable,
                    teacher=teacher,
                    ckpt=resume_ckpt,
                    bundle=bundle,
                    pretrain_template=pretrain_template,
                    encoder_bundle=encoder_bundle,
                    npz_root=npz_root,
                    depth=int(args.depth),
                    device=str(args.device),
                    steps=str(args.jac_steps),
                    left_bones=str(args.jac_left_bones),
                    right_bones=str(args.jac_right_bones),
                    out_json=jac_json,
                )
                _run_and_tee(cmd_jac, cwd=_ROOT, env=env, log_path=seed_dir / "stage6_step0_jac.log")
            metrics = _extract_root_metrics(diag_json)
            metrics.update(_extract_jacobian_metrics(jac_json))
            rec["arms"]["stage6_step0"] = {
                "diag_json": str(diag_json.resolve()),
                "jac_json": str(jac_json.resolve()) if jac_json.is_file() else "",
                "metrics": metrics,
            }

        if not bool(args.skip_resume):
            run_name = f"{args.base_run_name}_resume_seed{int(seed)}_e{int(args.epochs)}"
            if not bool(args.skip_train):
                cmd_train = _train_cmd(
                    python=sys.executable,
                    config_json=cfg_path,
                    out_model_dir=out_model_dir,
                    run_name=run_name,
                    seed=int(seed),
                    epochs=int(args.epochs),
                    dataset_index_mode=str(args.dataset_index_mode),
                    resume_ckpt=resume_ckpt,
                    extra_overrides=list(args.train_config_override or []),
                )
                _run_and_tee(cmd_train, cwd=_ROOT, env=env, log_path=seed_dir / "resume_train.log")
            ckpt = _pick_ckpt(out_model_dir, run_name)
            arm_dir = seed_dir / resume_arm
            cmd_diag = _diag_cmd(
                python=sys.executable,
                config_json=cfg_effective_path,
                ckpt=ckpt,
                target_clip=str(args.target_clip),
                depth=int(args.depth),
                bundle=bundle,
                pretrain_template=pretrain_template,
                encoder_bundle=encoder_bundle,
                loss_branch=str(args.loss_branch),
                component_losses=str(args.component_losses),
                device=str(args.device),
                out_dir=arm_dir,
                max_windows=int(args.max_windows),
            )
            _run_and_tee(cmd_diag, cwd=_ROOT, env=env, log_path=seed_dir / "resume_diag.log")
            diag_json = arm_dir / "sampling_grad_closure.json"
            jac_json = arm_dir / "direct_head_lr_jacobian.json"
            if not bool(args.skip_jacobian_probe):
                cmd_jac = _jac_probe_cmd(
                    python=sys.executable,
                    teacher=teacher,
                    ckpt=ckpt,
                    bundle=bundle,
                    pretrain_template=pretrain_template,
                    encoder_bundle=encoder_bundle,
                    npz_root=npz_root,
                    depth=int(args.depth),
                    device=str(args.device),
                    steps=str(args.jac_steps),
                    left_bones=str(args.jac_left_bones),
                    right_bones=str(args.jac_right_bones),
                    out_json=jac_json,
                )
                _run_and_tee(cmd_jac, cwd=_ROOT, env=env, log_path=seed_dir / "resume_jac.log")
            metrics = _extract_root_metrics(diag_json)
            metrics.update(_extract_jacobian_metrics(jac_json))
            rec["arms"][resume_arm] = {
                "run_name": run_name,
                "ckpt": str(ckpt.resolve()),
                "diag_json": str(diag_json.resolve()),
                "jac_json": str(jac_json.resolve()) if jac_json.is_file() else "",
                "metrics": metrics,
            }

        if not bool(args.skip_random):
            run_name = f"{args.base_run_name}_random_seed{int(seed)}_e{int(args.epochs)}"
            if not bool(args.skip_train):
                cmd_train = _train_cmd(
                    python=sys.executable,
                    config_json=cfg_path,
                    out_model_dir=out_model_dir,
                    run_name=run_name,
                    seed=int(seed),
                    epochs=int(args.epochs),
                    dataset_index_mode=str(args.dataset_index_mode),
                    resume_ckpt=None,
                    extra_overrides=list(args.train_config_override or []),
                )
                _run_and_tee(cmd_train, cwd=_ROOT, env=env, log_path=seed_dir / "random_train.log")
            ckpt = _pick_ckpt(out_model_dir, run_name)
            arm_dir = seed_dir / random_arm
            cmd_diag = _diag_cmd(
                python=sys.executable,
                config_json=cfg_effective_path,
                ckpt=ckpt,
                target_clip=str(args.target_clip),
                depth=int(args.depth),
                bundle=bundle,
                pretrain_template=pretrain_template,
                encoder_bundle=encoder_bundle,
                loss_branch=str(args.loss_branch),
                component_losses=str(args.component_losses),
                device=str(args.device),
                out_dir=arm_dir,
                max_windows=int(args.max_windows),
            )
            _run_and_tee(cmd_diag, cwd=_ROOT, env=env, log_path=seed_dir / "random_diag.log")
            diag_json = arm_dir / "sampling_grad_closure.json"
            jac_json = arm_dir / "direct_head_lr_jacobian.json"
            if not bool(args.skip_jacobian_probe):
                cmd_jac = _jac_probe_cmd(
                    python=sys.executable,
                    teacher=teacher,
                    ckpt=ckpt,
                    bundle=bundle,
                    pretrain_template=pretrain_template,
                    encoder_bundle=encoder_bundle,
                    npz_root=npz_root,
                    depth=int(args.depth),
                    device=str(args.device),
                    steps=str(args.jac_steps),
                    left_bones=str(args.jac_left_bones),
                    right_bones=str(args.jac_right_bones),
                    out_json=jac_json,
                )
                _run_and_tee(cmd_jac, cwd=_ROOT, env=env, log_path=seed_dir / "random_jac.log")
            metrics = _extract_root_metrics(diag_json)
            metrics.update(_extract_jacobian_metrics(jac_json))
            rec["arms"][random_arm] = {
                "run_name": run_name,
                "ckpt": str(ckpt.resolve()),
                "diag_json": str(diag_json.resolve()),
                "jac_json": str(jac_json.resolve()) if jac_json.is_file() else "",
                "metrics": metrics,
            }

        rows.append(rec)

    metric_keys_core = [
        "out_direct_ratio_step0",
        "cond_in_ratio_step0",
        "ratio_gap_step0",
        "direct_pose_loss_ratio_step0",
        "direct_pose_loss_log_ratio_step0",
        "cond_in_side_mean_abs_z",
        "cond_in_side_max_abs_z",
        "head_weight_rel_l2_best_sign",
        "head_bias_rel_l2_best_sign",
        "head_weight_norm_ratio_r_over_l",
    ]
    metric_keys_dynamics = [
        "dyn_ratio_shared_encoder_pre_act",
        "dyn_ratio_temporal_pre_pasa",
        "dyn_ratio_direct_head_pre_out",
        "dyn_amp_log_shared_to_temporal",
        "dyn_amp_log_temporal_to_direct_head_pre_out",
        "dyn_amp_log_direct_head_pre_out_to_out_direct",
        "dyn_amp_log_shared_to_out_direct",
    ]
    metric_keys_jacobian = [
        "jac_lr_ratio_all_mean",
        "jac_lr_ratio_all_median",
        "jac_lr_ratio_all_std",
        "jac_lr_ratio_local_mean",
        "jac_lr_ratio_local_median",
        "jac_lr_ratio_local_std",
    ]
    metric_keys = list(metric_keys_core) + list(metric_keys_dynamics) + list(metric_keys_jacobian)
    arms = ["stage6_step0", resume_arm, random_arm]
    agg: Dict[str, Any] = {}
    for arm in arms:
        sub = [r.get("arms", {}).get(arm, {}).get("metrics", {}) for r in rows]
        arm_stats: Dict[str, Any] = {}
        for mk in metric_keys:
            arm_stats[mk] = _mean_std_ci([_safe_float((x or {}).get(mk, float("nan"))) for x in sub])
        agg[arm] = arm_stats

    paired: Dict[str, Any] = {}
    paired_rows: List[Dict[str, Any]] = []
    for r in rows:
        seed = int(r.get("seed", -1))
        arms_payload = r.get("arms", {})
        m0 = (arms_payload.get("stage6_step0", {}) or {}).get("metrics", {})
        mr = (arms_payload.get(resume_arm, {}) or {}).get("metrics", {})
        mn = (arms_payload.get(random_arm, {}) or {}).get("metrics", {})
        row = {"seed": seed}
        for mk in metric_keys:
            v0 = _safe_float((m0 or {}).get(mk, float("nan")))
            vr = _safe_float((mr or {}).get(mk, float("nan")))
            vn = _safe_float((mn or {}).get(mk, float("nan")))
            row[f"resume_minus_stage6:{mk}"] = vr - v0 if math.isfinite(vr) and math.isfinite(v0) else float("nan")
            row[f"random_minus_stage6:{mk}"] = vn - v0 if math.isfinite(vn) and math.isfinite(v0) else float("nan")
            row[f"resume_minus_random:{mk}"] = vr - vn if math.isfinite(vr) and math.isfinite(vn) else float("nan")
        paired_rows.append(row)

    for comp_name in ("resume_minus_stage6", "random_minus_stage6", "resume_minus_random"):
        stats: Dict[str, Any] = {}
        for mk in metric_keys:
            key = f"{comp_name}:{mk}"
            stats[mk] = _mean_std_ci([_safe_float(pr.get(key, float("nan"))) for pr in paired_rows])
        paired[comp_name] = stats
    agg["paired"] = paired

    verdict = _verdict(
        agg=agg,
        thr=VerdictThresholds(
            out_ratio_high=float(args.thr_out_ratio_high),
            delta_resume_vs_step0=float(args.thr_delta_resume_vs_step0),
            head_w_rel_l2_high=float(args.thr_head_w_rel_l2_high),
        ),
        resume_arm=f"resume_e{int(args.epochs)}",
    )

    summary: Dict[str, Any] = {
        "config_json": str(cfg_path.resolve()),
        "config_effective": str(cfg_effective_path.resolve()),
        "resume_ckpt": str(resume_ckpt.resolve()),
        "out_model_dir": str(out_model_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "seeds": [int(s) for s in seeds],
        "epochs": int(args.epochs),
        "dataset_index_mode": str(args.dataset_index_mode),
        "train_config_override": [str(x) for x in (args.train_config_override or [])],
        "target_clip": str(args.target_clip),
        "teacher": str(teacher.resolve()),
        "npz_root": str(npz_root.resolve()),
        "loss_branch": str(args.loss_branch),
        "component_losses": str(args.component_losses),
        "jac_steps": str(args.jac_steps),
        "jac_left_bones": str(args.jac_left_bones),
        "jac_right_bones": str(args.jac_right_bones),
        "jacobian_probe_enabled": not bool(args.skip_jacobian_probe),
        "rows": rows,
        "metric_keys": metric_keys,
        "metric_keys_core": metric_keys_core,
        "metric_keys_dynamics": metric_keys_dynamics,
        "metric_keys_jacobian": metric_keys_jacobian,
        "aggregate": agg,
        "paired_rows": paired_rows,
        "thresholds": {
            "out_ratio_high": float(args.thr_out_ratio_high),
            "delta_resume_vs_step0": float(args.thr_delta_resume_vs_step0),
            "head_w_rel_l2_high": float(args.thr_head_w_rel_l2_high),
        },
        "verdict": verdict,
    }

    out_json = out_dir / "summary.json"
    out_md = out_dir / "summary.md"
    _write_json(out_json, summary)

    def _fmt_mss(obj: Mapping[str, Any], key: str) -> str:
        d = obj.get(key, {}) if isinstance(obj, Mapping) else {}
        return f"{_safe_float(d.get('mean', float('nan'))):.3f}±{_safe_float(d.get('std', float('nan'))):.3f}"

    def _fmt_ci(obj: Mapping[str, Any], key: str) -> str:
        d = obj.get(key, {}) if isinstance(obj, Mapping) else {}
        return f"[{_safe_float(d.get('ci95_low', float('nan'))):.3f},{_safe_float(d.get('ci95_high', float('nan'))):.3f}]"

    lines: List[str] = []
    lines.append("# Stage7 root-cause triad summary")
    lines.append("")
    lines.append(f"- config_json: `{cfg_path}`")
    lines.append(f"- resume_ckpt: `{resume_ckpt}`")
    lines.append(f"- seeds: `{','.join(str(s) for s in seeds)}`")
    lines.append(f"- epochs: {int(args.epochs)}")
    lines.append(f"- dataset_index_mode: `{args.dataset_index_mode}`")
    if args.train_config_override:
        lines.append(f"- train_config_override: `{', '.join(str(x) for x in args.train_config_override)}`")
    lines.append(f"- target_clip: `{args.target_clip}`")
    lines.append(f"- teacher: `{teacher}`")
    lines.append(f"- jacobian_probe: `{not bool(args.skip_jacobian_probe)}`")
    lines.append(f"- jac_steps: `{args.jac_steps}`")
    lines.append(f"- jac_left_bones: `{args.jac_left_bones}`")
    lines.append(f"- jac_right_bones: `{args.jac_right_bones}`")
    lines.append("")

    lines.append("## Per-seed core metrics")
    lines.append("")
    lines.append(
        "|seed|arm|out_direct_ratio|cond_in_ratio|gap|direct_pose_loss_ratio|direct_pose_loss_log_ratio|cond_in_side_mean|cond_in_side_max|head_w_rel_l2_best_sign|head_b_rel_l2_best_sign|"
    )
    lines.append("|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        seed = int(r.get("seed", -1))
        for arm in arms:
            m = (r.get("arms", {}).get(arm, {}) or {}).get("metrics", {})
            if not isinstance(m, Mapping) or not m:
                continue
            lines.append(
                "|{seed}|{arm}|{od:.3f}|{ci:.3f}|{gap:+.3f}|{dlr:.3f}|{dll:+.3f}|{csm:.3f}|{csx:.3f}|{wl2:.3f}|{bl2:.3f}|".format(
                    seed=seed,
                    arm=arm,
                    od=_safe_float(m.get("out_direct_ratio_step0", float("nan"))),
                    ci=_safe_float(m.get("cond_in_ratio_step0", float("nan"))),
                    gap=_safe_float(m.get("ratio_gap_step0", float("nan"))),
                    dlr=_safe_float(m.get("direct_pose_loss_ratio_step0", float("nan"))),
                    dll=_safe_float(m.get("direct_pose_loss_log_ratio_step0", float("nan"))),
                    csm=_safe_float(m.get("cond_in_side_mean_abs_z", float("nan"))),
                    csx=_safe_float(m.get("cond_in_side_max_abs_z", float("nan"))),
                    wl2=_safe_float(m.get("head_weight_rel_l2_best_sign", float("nan"))),
                    bl2=_safe_float(m.get("head_bias_rel_l2_best_sign", float("nan"))),
                )
            )

    lines.append("")
    lines.append("## Per-seed dynamics-path metrics")
    lines.append("")
    lines.append(
        "|seed|arm|shared_ratio|temporal_ratio|direct_pre_ratio|amp_log(shared->temporal)|"
        "amp_log(temporal->direct_pre)|amp_log(direct_pre->out_direct)|amp_log(shared->out_direct)|"
    )
    lines.append("|--:|:--|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        seed = int(r.get("seed", -1))
        for arm in arms:
            m = (r.get("arms", {}).get(arm, {}) or {}).get("metrics", {})
            if not isinstance(m, Mapping) or not m:
                continue
            lines.append(
                "|{seed}|{arm}|{sr:.3f}|{tr:.3f}|{dr:.3f}|{a_st:+.3f}|{a_td:+.3f}|{a_do:+.3f}|{a_so:+.3f}|".format(
                    seed=seed,
                    arm=arm,
                    sr=_safe_float(m.get("dyn_ratio_shared_encoder_pre_act", float("nan"))),
                    tr=_safe_float(m.get("dyn_ratio_temporal_pre_pasa", float("nan"))),
                    dr=_safe_float(m.get("dyn_ratio_direct_head_pre_out", float("nan"))),
                    a_st=_safe_float(m.get("dyn_amp_log_shared_to_temporal", float("nan"))),
                    a_td=_safe_float(m.get("dyn_amp_log_temporal_to_direct_head_pre_out", float("nan"))),
                    a_do=_safe_float(m.get("dyn_amp_log_direct_head_pre_out_to_out_direct", float("nan"))),
                    a_so=_safe_float(m.get("dyn_amp_log_shared_to_out_direct", float("nan"))),
                )
            )

    lines.append("")
    lines.append("## Aggregate by arm (mean +- std, 95% CI)")
    lines.append("")
    lines.append("|arm|out_direct_ratio|cond_in_ratio|gap|direct_pose_loss_ratio|direct_pose_loss_log_ratio|cond_in_side_mean|cond_in_side_max|head_w_rel_l2_best_sign|")
    lines.append("|:--|:--|:--|:--|:--|:--|:--|:--|:--|")
    for arm in arms:
        st = agg.get(arm, {})
        lines.append(
            "|{arm}|{od} {od_ci}|{ci} {ci_ci}|{gap} {gap_ci}|{dlr} {dlr_ci}|{dll} {dll_ci}|{csm} {csm_ci}|{csx} {csx_ci}|{wl2} {wl2_ci}|".format(
                arm=arm,
                od=_fmt_mss(st, "out_direct_ratio_step0"),
                od_ci=_fmt_ci(st, "out_direct_ratio_step0"),
                ci=_fmt_mss(st, "cond_in_ratio_step0"),
                ci_ci=_fmt_ci(st, "cond_in_ratio_step0"),
                gap=_fmt_mss(st, "ratio_gap_step0"),
                gap_ci=_fmt_ci(st, "ratio_gap_step0"),
                dlr=_fmt_mss(st, "direct_pose_loss_ratio_step0"),
                dlr_ci=_fmt_ci(st, "direct_pose_loss_ratio_step0"),
                dll=_fmt_mss(st, "direct_pose_loss_log_ratio_step0"),
                dll_ci=_fmt_ci(st, "direct_pose_loss_log_ratio_step0"),
                csm=_fmt_mss(st, "cond_in_side_mean_abs_z"),
                csm_ci=_fmt_ci(st, "cond_in_side_mean_abs_z"),
                csx=_fmt_mss(st, "cond_in_side_max_abs_z"),
                csx_ci=_fmt_ci(st, "cond_in_side_max_abs_z"),
                wl2=_fmt_mss(st, "head_weight_rel_l2_best_sign"),
                wl2_ci=_fmt_ci(st, "head_weight_rel_l2_best_sign"),
            )
        )

    lines.append("")
    lines.append("## Per-seed Jacobian LR metrics")
    lines.append("")
    lines.append("|seed|arm|jac_RL_all_mean|jac_RL_all_median|jac_RL_all_std|jac_RL_local_mean|jac_RL_local_median|jac_RL_local_std|")
    lines.append("|--:|:--|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        seed = int(r.get("seed", -1))
        for arm in arms:
            m = (r.get("arms", {}).get(arm, {}) or {}).get("metrics", {})
            if not isinstance(m, Mapping) or not m:
                continue
            lines.append(
                "|{seed}|{arm}|{a_mean:.3f}|{a_med:.3f}|{a_std:.3f}|{l_mean:.3f}|{l_med:.3f}|{l_std:.3f}|".format(
                    seed=seed,
                    arm=arm,
                    a_mean=_safe_float(m.get("jac_lr_ratio_all_mean", float("nan"))),
                    a_med=_safe_float(m.get("jac_lr_ratio_all_median", float("nan"))),
                    a_std=_safe_float(m.get("jac_lr_ratio_all_std", float("nan"))),
                    l_mean=_safe_float(m.get("jac_lr_ratio_local_mean", float("nan"))),
                    l_med=_safe_float(m.get("jac_lr_ratio_local_median", float("nan"))),
                    l_std=_safe_float(m.get("jac_lr_ratio_local_std", float("nan"))),
                )
            )

    lines.append("")
    lines.append("## Aggregate Jacobian LR metrics (mean +- std, 95% CI)")
    lines.append("")
    lines.append("|arm|jac_RL_all_mean|jac_RL_all_median|jac_RL_all_std|jac_RL_local_mean|jac_RL_local_median|jac_RL_local_std|")
    lines.append("|:--|:--|:--|:--|:--|:--|:--|")
    for arm in arms:
        st = agg.get(arm, {})
        lines.append(
            "|{arm}|{a_mean} {a_mean_ci}|{a_med} {a_med_ci}|{a_std} {a_std_ci}|{l_mean} {l_mean_ci}|{l_med} {l_med_ci}|{l_std} {l_std_ci}|".format(
                arm=arm,
                a_mean=_fmt_mss(st, "jac_lr_ratio_all_mean"),
                a_mean_ci=_fmt_ci(st, "jac_lr_ratio_all_mean"),
                a_med=_fmt_mss(st, "jac_lr_ratio_all_median"),
                a_med_ci=_fmt_ci(st, "jac_lr_ratio_all_median"),
                a_std=_fmt_mss(st, "jac_lr_ratio_all_std"),
                a_std_ci=_fmt_ci(st, "jac_lr_ratio_all_std"),
                l_mean=_fmt_mss(st, "jac_lr_ratio_local_mean"),
                l_mean_ci=_fmt_ci(st, "jac_lr_ratio_local_mean"),
                l_med=_fmt_mss(st, "jac_lr_ratio_local_median"),
                l_med_ci=_fmt_ci(st, "jac_lr_ratio_local_median"),
                l_std=_fmt_mss(st, "jac_lr_ratio_local_std"),
                l_std_ci=_fmt_ci(st, "jac_lr_ratio_local_std"),
            )
        )

    lines.append("")
    lines.append("## Aggregate dynamics metrics (mean +- std, 95% CI)")
    lines.append("")
    lines.append(
        "|arm|shared_ratio|temporal_ratio|direct_pre_ratio|amp_log(shared->temporal)|"
        "amp_log(temporal->direct_pre)|amp_log(direct_pre->out_direct)|amp_log(shared->out_direct)|"
    )
    lines.append("|:--|:--|:--|:--|:--|:--|:--|:--|")
    for arm in arms:
        st = agg.get(arm, {})
        lines.append(
            "|{arm}|{sr} {sr_ci}|{tr} {tr_ci}|{dr} {dr_ci}|{a_st} {a_st_ci}|{a_td} {a_td_ci}|{a_do} {a_do_ci}|{a_so} {a_so_ci}|".format(
                arm=arm,
                sr=_fmt_mss(st, "dyn_ratio_shared_encoder_pre_act"),
                sr_ci=_fmt_ci(st, "dyn_ratio_shared_encoder_pre_act"),
                tr=_fmt_mss(st, "dyn_ratio_temporal_pre_pasa"),
                tr_ci=_fmt_ci(st, "dyn_ratio_temporal_pre_pasa"),
                dr=_fmt_mss(st, "dyn_ratio_direct_head_pre_out"),
                dr_ci=_fmt_ci(st, "dyn_ratio_direct_head_pre_out"),
                a_st=_fmt_mss(st, "dyn_amp_log_shared_to_temporal"),
                a_st_ci=_fmt_ci(st, "dyn_amp_log_shared_to_temporal"),
                a_td=_fmt_mss(st, "dyn_amp_log_temporal_to_direct_head_pre_out"),
                a_td_ci=_fmt_ci(st, "dyn_amp_log_temporal_to_direct_head_pre_out"),
                a_do=_fmt_mss(st, "dyn_amp_log_direct_head_pre_out_to_out_direct"),
                a_do_ci=_fmt_ci(st, "dyn_amp_log_direct_head_pre_out_to_out_direct"),
                a_so=_fmt_mss(st, "dyn_amp_log_shared_to_out_direct"),
                a_so_ci=_fmt_ci(st, "dyn_amp_log_shared_to_out_direct"),
            )
        )

    lines.append("")
    lines.append("## Paired deltas (mean +- std, 95% CI)")
    lines.append("")
    lines.append("|delta|out_direct_ratio|cond_in_ratio|gap|direct_pose_loss_ratio|direct_pose_loss_log_ratio|cond_in_side_mean|cond_in_side_max|head_w_rel_l2_best_sign|")
    lines.append("|:--|:--|:--|:--|:--|:--|:--|:--|:--|")
    for comp in ("resume_minus_stage6", "random_minus_stage6", "resume_minus_random"):
        st = paired.get(comp, {})
        lines.append(
            "|{comp}|{od} {od_ci}|{ci} {ci_ci}|{gap} {gap_ci}|{dlr} {dlr_ci}|{dll} {dll_ci}|{csm} {csm_ci}|{csx} {csx_ci}|{wl2} {wl2_ci}|".format(
                comp=comp,
                od=_fmt_mss(st, "out_direct_ratio_step0"),
                od_ci=_fmt_ci(st, "out_direct_ratio_step0"),
                ci=_fmt_mss(st, "cond_in_ratio_step0"),
                ci_ci=_fmt_ci(st, "cond_in_ratio_step0"),
                gap=_fmt_mss(st, "ratio_gap_step0"),
                gap_ci=_fmt_ci(st, "ratio_gap_step0"),
                dlr=_fmt_mss(st, "direct_pose_loss_ratio_step0"),
                dlr_ci=_fmt_ci(st, "direct_pose_loss_ratio_step0"),
                dll=_fmt_mss(st, "direct_pose_loss_log_ratio_step0"),
                dll_ci=_fmt_ci(st, "direct_pose_loss_log_ratio_step0"),
                csm=_fmt_mss(st, "cond_in_side_mean_abs_z"),
                csm_ci=_fmt_ci(st, "cond_in_side_mean_abs_z"),
                csx=_fmt_mss(st, "cond_in_side_max_abs_z"),
                csx_ci=_fmt_ci(st, "cond_in_side_max_abs_z"),
                wl2=_fmt_mss(st, "head_weight_rel_l2_best_sign"),
                wl2_ci=_fmt_ci(st, "head_weight_rel_l2_best_sign"),
            )
        )

    lines.append("")
    lines.append("## Paired dynamics deltas (mean +- std, 95% CI)")
    lines.append("")
    lines.append(
        "|delta|shared_ratio|temporal_ratio|direct_pre_ratio|amp_log(shared->temporal)|"
        "amp_log(temporal->direct_pre)|amp_log(direct_pre->out_direct)|amp_log(shared->out_direct)|"
    )
    lines.append("|:--|:--|:--|:--|:--|:--|:--|:--|")
    for comp in ("resume_minus_stage6", "random_minus_stage6", "resume_minus_random"):
        st = paired.get(comp, {})
        lines.append(
            "|{comp}|{sr} {sr_ci}|{tr} {tr_ci}|{dr} {dr_ci}|{a_st} {a_st_ci}|{a_td} {a_td_ci}|{a_do} {a_do_ci}|{a_so} {a_so_ci}|".format(
                comp=comp,
                sr=_fmt_mss(st, "dyn_ratio_shared_encoder_pre_act"),
                sr_ci=_fmt_ci(st, "dyn_ratio_shared_encoder_pre_act"),
                tr=_fmt_mss(st, "dyn_ratio_temporal_pre_pasa"),
                tr_ci=_fmt_ci(st, "dyn_ratio_temporal_pre_pasa"),
                dr=_fmt_mss(st, "dyn_ratio_direct_head_pre_out"),
                dr_ci=_fmt_ci(st, "dyn_ratio_direct_head_pre_out"),
                a_st=_fmt_mss(st, "dyn_amp_log_shared_to_temporal"),
                a_st_ci=_fmt_ci(st, "dyn_amp_log_shared_to_temporal"),
                a_td=_fmt_mss(st, "dyn_amp_log_temporal_to_direct_head_pre_out"),
                a_td_ci=_fmt_ci(st, "dyn_amp_log_temporal_to_direct_head_pre_out"),
                a_do=_fmt_mss(st, "dyn_amp_log_direct_head_pre_out_to_out_direct"),
                a_do_ci=_fmt_ci(st, "dyn_amp_log_direct_head_pre_out_to_out_direct"),
                a_so=_fmt_mss(st, "dyn_amp_log_shared_to_out_direct"),
                a_so_ci=_fmt_ci(st, "dyn_amp_log_shared_to_out_direct"),
            )
        )

    lines.append("")
    lines.append("## Paired Jacobian LR deltas (mean +- std, 95% CI)")
    lines.append("")
    lines.append("|delta|jac_RL_all_mean|jac_RL_all_median|jac_RL_all_std|jac_RL_local_mean|jac_RL_local_median|jac_RL_local_std|")
    lines.append("|:--|:--|:--|:--|:--|:--|:--|")
    for comp in ("resume_minus_stage6", "random_minus_stage6", "resume_minus_random"):
        st = paired.get(comp, {})
        lines.append(
            "|{comp}|{a_mean} {a_mean_ci}|{a_med} {a_med_ci}|{a_std} {a_std_ci}|{l_mean} {l_mean_ci}|{l_med} {l_med_ci}|{l_std} {l_std_ci}|".format(
                comp=comp,
                a_mean=_fmt_mss(st, "jac_lr_ratio_all_mean"),
                a_mean_ci=_fmt_ci(st, "jac_lr_ratio_all_mean"),
                a_med=_fmt_mss(st, "jac_lr_ratio_all_median"),
                a_med_ci=_fmt_ci(st, "jac_lr_ratio_all_median"),
                a_std=_fmt_mss(st, "jac_lr_ratio_all_std"),
                a_std_ci=_fmt_ci(st, "jac_lr_ratio_all_std"),
                l_mean=_fmt_mss(st, "jac_lr_ratio_local_mean"),
                l_mean_ci=_fmt_ci(st, "jac_lr_ratio_local_mean"),
                l_med=_fmt_mss(st, "jac_lr_ratio_local_median"),
                l_med_ci=_fmt_ci(st, "jac_lr_ratio_local_median"),
                l_std=_fmt_mss(st, "jac_lr_ratio_local_std"),
                l_std_ci=_fmt_ci(st, "jac_lr_ratio_local_std"),
            )
        )

    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- primary: **{verdict.get('primary', 'n/a')}**")
    flags = verdict.get("flags", {}) if isinstance(verdict.get("flags", {}), Mapping) else {}
    for k in sorted(flags.keys()):
        lines.append(f"- {k}: {bool(flags[k])}")

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- summary_json: `{out_json}`")
    lines.append(f"- summary_md: `{out_md}`")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {out_json}")
    print(f"[OK] wrote: {out_md}")


if __name__ == "__main__":
    main()
