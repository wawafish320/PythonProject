#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

try:
    from run_cp015_oldplan_downstream_chain import ROOT, load_json, safe_float, write_json
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import ROOT, load_json, safe_float, write_json

from train.posttrain import (
    _build_dataset_and_loader,
    _build_model_and_trainer,
    _build_posttrain_model_from_ckpt,
    _build_rollout_mode_kwargs,
    _cfg_from_payload,
    _freeze_all,
    _lambda_fusion_loss_rollout,
    _resolve_device,
    _resolve_train_mode,
    _select_trainable_params,
    _set_seed,
    _unfreeze_for_train_mode,
)


RUN_TAG = "20260402_arm_efficiency_audit"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_efficiency_audit_{RUN_TAG}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_replace_efficiency_audit_{RUN_TAG}"
CONFIG_ROOT = OUT_ROOT / "configs"
LOG_FILE = OUT_ROOT / "lane.log"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
STATUS_JSON = OUT_ROOT / "status.json"
CPU_EXEC = ROOT / "debug_output" / "_tmp_phasecd_min_ablation_20260330" / "cpu_nomps_exec.py"
ENCODER_BUNDLE = ROOT / "models" / "motion_encoder_equiv.pt.best.pt"
AFFINE_STATS = ROOT / "debug_output" / "_tmp_phaseb_affine_20260304" / "affine_fit_mix08" / "affine_stats.json"

BASELINE_70A_CKPT = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "70a"
    / "ckpt_last_WalkF_stage7_70a_fromfresh_20260317.pth"
)
BASELINE_70A_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "eval_model_source"
    / "70a_group_summary.json"
)
BASELINE_WARMSTART_CKPT = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "warmstart"
    / "ckpt_last_70a_replace_zerophase_20260317.pth"
)
BASELINE_WARMSTART_REPORT = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "warmstart"
    / "replace_zerophase_report.json"
)
BASELINE_REPLACE_CKPT = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "70b_replace_lowdrift"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth"
)
BASELINE_REPLACE_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "eval_model_source"
    / "new70b_replace_lowdrift_group_summary.json"
)
BASELINE_REPLACE_LOG = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "70b_replace_lowdrift"
    / "posttrain_log_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.json"
)
BASELINE_REPLACE_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "configs"
    / "posttrain_70b_replace_lowdrift_fromfresh_20260317.json"
)

TAIL_70A_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth"
)
TAIL_70A_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "eval_model_source_group_summary.json"
)
TAIL_WARMSTART_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_from_70a_20260402"
    / "warmstart"
    / "ckpt_last_cp015_tailk7_70a_replace_zerophase_20260402.pth"
)
TAIL_CURRENT_REPLACE_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_schedule_ablation_20260402"
    / "eval_model_source"
    / "e2x60_group_summary.json"
)
TAIL_CURRENT_REPLACE_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_schedule_ablation_20260402"
    / "e2x60"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_e2x60_lr5e5_from_cp015_tailk7_70a_20260402.pth"
)
TAIL_CURRENT_REPLACE_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_schedule_ablation_20260402"
    / "e2x60"
    / "posttrain_log_WalkF_stage7_70b_replace_lowdrift_e2x60_lr5e5_from_cp015_tailk7_70a_20260402.json"
)
TAIL_BEST3WAY_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_3way_objective_ablation_20260402_3way_objective"
    / "eval_model_source"
    / "e2x60_3way_arm125_group_summary.json"
)
TAIL_BEST3WAY_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_3way_objective_ablation_20260402_3way_objective"
    / "e2x60_3way_arm125"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_e2x60_3way_arm125_lr5e5_from_cp015_tailk7_70a_20260402_3way_objective.pth"
)
TAIL_BEST3WAY_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_3way_objective_ablation_20260402_3way_objective"
    / "e2x60_3way_arm125"
    / "posttrain_log_WalkF_stage7_70b_replace_lowdrift_e2x60_3way_arm125_lr5e5_from_cp015_tailk7_70a_20260402_3way_objective.json"
)
PROBE_TEMPLATE_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_3way_objective_ablation_20260402_3way_objective"
    / "configs"
    / "posttrain_70b_replace_lowdrift_e2x60_3way_arm125_lr5e5_from_cp015_tailk7_70a_20260402_3way_objective.json"
)

GROUPS: Tuple[str, ...] = ("all_ex_root", "leg", "arm", "else")
PCTS: Tuple[str, ...] = ("mean", "p50", "p90", "p95")
SNAPSHOT_STEPS: Tuple[int, ...] = (0, 1, 5, 20, 60)
MODULE_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "shared_trunk": ("direct_pose_head",),
    "arm_branch": ("direct_pose_arm_proj", "direct_pose_out_arm"),
    "else_branch": ("direct_pose_else_proj", "direct_pose_out_else"),
    "leg_readout": ("direct_pose_out_leg",),
    "leg_branch": ("direct_pose_leg_head",),
}
HOOK_MODULES: Mapping[str, str] = {
    "direct_pose_head": "shared_trunk_out",
    "direct_pose_arm_proj": "arm_proj_out",
    "direct_pose_out_arm": "arm_readout_out",
    "direct_pose_else_proj": "else_proj_out",
    "direct_pose_out_else": "else_readout_out",
}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fmt(x: Any, digits: int = 6) -> str:
    v = safe_float(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def assert_exists(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required artifact(s):\n" + "\n".join(missing))


def run_cpu(cmd: Sequence[str], *, log_file: Path, commands: List[str]) -> None:
    cmd_list = [str(x) for x in cmd]
    cmd_str = " ".join(cmd_list)
    commands.append(cmd_str)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + cmd_str + "\n")
        fh.flush()
        log(f"RUN {cmd_str}")
        proc = subprocess.Popen(
            cmd_list,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        code = int(proc.wait())
        fh.write(f"[exit_code] {code}\n")
        fh.flush()
    if code != 0:
        raise SystemExit(code)


def load_group_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    obj = load_json(path)
    groups = obj.get("groups", {})
    out: Dict[str, Dict[str, float]] = {}
    for group_name in GROUPS:
        payload = groups.get(group_name, {}) if isinstance(groups, dict) else {}
        out[group_name] = {pct: safe_float(payload.get(pct)) for pct in PCTS}
    return out


def state_and_cfg(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "model" not in obj:
        raise RuntimeError(f"unexpected checkpoint format: {path}")
    state = obj["model"]
    cfg = obj.get("posttrain_cfg", {})
    if not isinstance(state, dict):
        raise RuntimeError(f"unexpected state_dict in {path}")
    return state, cfg if isinstance(cfg, dict) else {}


def key_matches_prefix(key: str, prefixes: Sequence[str]) -> bool:
    return any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)


def state_vector(state: Mapping[str, Any], prefixes: Sequence[str]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for key in sorted(state.keys()):
        value = state[key]
        if not key_matches_prefix(str(key), prefixes):
            continue
        if not torch.is_tensor(value) or not torch.is_floating_point(value):
            continue
        chunks.append(value.detach().cpu().float().reshape(-1))
    if not chunks:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def vector_stats(vec: torch.Tensor) -> Dict[str, float]:
    if vec.numel() <= 0:
        return {
            "numel": 0.0,
            "l2": float("nan"),
            "mean_abs": float("nan"),
            "max_abs": float("nan"),
            "std": float("nan"),
        }
    return {
        "numel": float(vec.numel()),
        "l2": float(vec.norm().item()),
        "mean_abs": float(vec.abs().mean().item()),
        "max_abs": float(vec.abs().max().item()),
        "std": float(vec.std(unbiased=False).item()) if vec.numel() > 1 else 0.0,
    }


def cosine_between(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() <= 0 or b.numel() <= 0 or a.numel() != b.numel():
        return float("nan")
    an = float(a.norm().item())
    bn = float(b.norm().item())
    if an <= 0.0 or bn <= 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / max(1e-12, an * bn))


def delta_stats(src: Mapping[str, Any], dst: Mapping[str, Any], prefixes: Sequence[str]) -> Dict[str, float]:
    a = state_vector(src, prefixes)
    b = state_vector(dst, prefixes)
    if a.numel() <= 0 or b.numel() <= 0 or a.numel() != b.numel():
        return {"delta_l2": float("nan"), "delta_rel": float("nan"), "cosine": float("nan")}
    delta = b - a
    base_norm = float(a.norm().item())
    delta_norm = float(delta.norm().item())
    return {
        "delta_l2": delta_norm,
        "delta_rel": float(delta_norm / max(1e-12, base_norm)),
        "cosine": cosine_between(a, b),
    }


def changed_tensor_keys(src: Mapping[str, Any], dst: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key in sorted(set(src.keys()) & set(dst.keys())):
        a = src[key]
        b = dst[key]
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            continue
        if not torch.is_floating_point(a) or not torch.is_floating_point(b):
            continue
        if a.shape != b.shape:
            out.append({"key": key, "shape_src": list(a.shape), "shape_dst": list(b.shape), "shape_changed": True})
            continue
        diff = (a.detach().cpu().float() - b.detach().cpu().float()).abs()
        max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
        if max_abs > 0.0:
            out.append(
                {
                    "key": key,
                    "shape": list(a.shape),
                    "delta_l2": float(diff.norm().item()),
                    "max_abs": max_abs,
                }
            )
    return out


def column_diff_summary(src: Mapping[str, Any], dst: Mapping[str, Any], key: str) -> Dict[str, Any]:
    a = src.get(key)
    b = dst.get(key)
    if not (torch.is_tensor(a) and torch.is_tensor(b)) or a.shape != b.shape or a.ndim != 2:
        return {"key": key, "changed_cols": [], "changed_col_count": 0}
    diff = (b.detach().cpu().float() - a.detach().cpu().float()).abs().amax(dim=0)
    changed = [
        {"col": int(idx), "max_abs": float(val)}
        for idx, val in enumerate(diff.tolist())
        if float(val) > 0.0
    ]
    return {"key": key, "changed_cols": changed, "changed_col_count": len(changed)}


def build_static_audit() -> Dict[str, Any]:
    baseline_70a_state, baseline_70a_cfg = state_and_cfg(BASELINE_70A_CKPT)
    baseline_ws_state, baseline_ws_cfg = state_and_cfg(BASELINE_WARMSTART_CKPT)
    baseline_replace_state, _ = state_and_cfg(BASELINE_REPLACE_CKPT)
    tail_70a_state, tail_70a_cfg = state_and_cfg(TAIL_70A_CKPT)
    tail_ws_state, tail_ws_cfg = state_and_cfg(TAIL_WARMSTART_CKPT)
    tail_replace_state, _ = state_and_cfg(TAIL_CURRENT_REPLACE_CKPT)
    tail_3way_state, _ = state_and_cfg(TAIL_BEST3WAY_CKPT)

    by_group: Dict[str, Any] = {}
    for group_name, prefixes in MODULE_GROUPS.items():
        base70 = state_vector(baseline_70a_state, prefixes)
        tail70 = state_vector(tail_70a_state, prefixes)
        basews = state_vector(baseline_ws_state, prefixes)
        tailws = state_vector(tail_ws_state, prefixes)
        by_group[group_name] = {
            "baseline_70a": vector_stats(base70),
            "tailk7_70a": vector_stats(tail70),
            "donor_cosine_baseline_vs_tailk7": cosine_between(base70, tail70),
            "baseline_warmstart": vector_stats(basews),
            "tailk7_warmstart": vector_stats(tailws),
            "warmstart_cosine_baseline_vs_tailk7": cosine_between(basews, tailws),
            "baseline_70a_to_warmstart": delta_stats(baseline_70a_state, baseline_ws_state, prefixes),
            "tailk7_70a_to_warmstart": delta_stats(tail_70a_state, tail_ws_state, prefixes),
            "baseline_warmstart_to_replace": delta_stats(baseline_ws_state, baseline_replace_state, prefixes),
            "tailk7_warmstart_to_replace_e2x60": delta_stats(tail_ws_state, tail_replace_state, prefixes),
            "tailk7_warmstart_to_3way_arm125": delta_stats(tail_ws_state, tail_3way_state, prefixes),
        }

    baseline_warmstart_changes = changed_tensor_keys(baseline_70a_state, baseline_ws_state)
    tail_warmstart_changes = changed_tensor_keys(tail_70a_state, tail_ws_state)
    return {
        "posttrain_cfg": {
            "baseline_70a": {
                "direct_pose_feat_source": baseline_70a_cfg.get("direct_pose_feat_source"),
                "direct_pose_time_pe_dim": baseline_70a_cfg.get("direct_pose_time_pe_dim"),
                "direct_pose_split_enable": baseline_70a_cfg.get("direct_pose_split_enable"),
                "direct_pose_arm_split_enable": baseline_70a_cfg.get("direct_pose_arm_split_enable"),
            },
            "baseline_warmstart": {
                "direct_pose_feat_source": baseline_ws_cfg.get("direct_pose_feat_source"),
                "direct_pose_time_pe_dim": baseline_ws_cfg.get("direct_pose_time_pe_dim"),
                "direct_pose_split_enable": baseline_ws_cfg.get("direct_pose_split_enable"),
                "direct_pose_arm_split_enable": baseline_ws_cfg.get("direct_pose_arm_split_enable"),
            },
            "tailk7_70a": {
                "direct_pose_feat_source": tail_70a_cfg.get("direct_pose_feat_source"),
                "direct_pose_time_pe_dim": tail_70a_cfg.get("direct_pose_time_pe_dim"),
                "direct_pose_split_enable": tail_70a_cfg.get("direct_pose_split_enable"),
                "direct_pose_arm_split_enable": tail_70a_cfg.get("direct_pose_arm_split_enable"),
            },
            "tailk7_warmstart": {
                "direct_pose_feat_source": tail_ws_cfg.get("direct_pose_feat_source"),
                "direct_pose_time_pe_dim": tail_ws_cfg.get("direct_pose_time_pe_dim"),
                "direct_pose_split_enable": tail_ws_cfg.get("direct_pose_split_enable"),
                "direct_pose_arm_split_enable": tail_ws_cfg.get("direct_pose_arm_split_enable"),
            },
        },
        "baseline_warmstart_report": load_json(BASELINE_WARMSTART_REPORT),
        "baseline_warmstart_changed_keys": baseline_warmstart_changes,
        "tailk7_warmstart_changed_keys": tail_warmstart_changes,
        "baseline_phase_adapt_columns": [
            column_diff_summary(baseline_70a_state, baseline_ws_state, "direct_pose_head.0.weight"),
            column_diff_summary(baseline_70a_state, baseline_ws_state, "direct_pose_leg_head.0.weight"),
        ],
        "by_group": by_group,
    }


def make_probe_config(case_name: str, ckpt_in: Path) -> Tuple[Path, Path, str]:
    payload = load_json(PROBE_TEMPLATE_CONFIG)
    run_name = f"WalkF_stage7_70b_replace_effprobe_{case_name}_{RUN_TAG}"
    out_dir = MODEL_ROOT / case_name
    cfg_json = CONFIG_ROOT / f"posttrain_{case_name}_{RUN_TAG}.json"
    payload.update(
        {
            "ckpt_in": str(ckpt_in),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "epochs": 1,
            "steps_per_epoch": 60,
            "save_step_ckpts": "0,1,5,20,60",
            "rollout_random_offset": False,
            "direct_pose_grad_monitor_enable": True,
            "seed": 0,
        }
    )
    write_json(cfg_json, payload)
    return cfg_json, out_dir, run_name


def run_probe_train(case_name: str, cfg_json: Path, out_dir: Path, run_name: str, commands: List[str]) -> Path:
    ckpt_last = out_dir / f"ckpt_last_{run_name}.pth"
    if ckpt_last.is_file():
        return ckpt_last
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_payload = load_json(cfg_json)
    run_cpu(
        [
            sys.executable,
            str(CPU_EXEC),
            "-m",
            "train.posttrain",
            "--config",
            str(cfg_json),
            "--ckpt_in",
            str(cfg_payload["ckpt_in"]),
            "--out_dir",
            str(out_dir),
            "--run_name",
            run_name,
            "--posttrain_contacts_source",
            "pretrain_contact",
            "--posttrain_contacts_pretrain_clamp",
            "1.0",
            "--encoder_bundle",
            str(cfg_payload.get("encoder_bundle", ENCODER_BUNDLE)),
            "--posttrain_contacts_pretrain_affine_stats",
            str(cfg_payload.get("posttrain_contacts_pretrain_affine_stats", AFFINE_STATS)),
        ],
        log_file=LOG_FILE,
        commands=commands,
    )
    return ckpt_last


def snapshot_ckpt(case_out_dir: Path, run_name: str, step: int) -> Path:
    if step == 60:
        step_ckpt = case_out_dir / f"ckpt_step_{step:06d}_{run_name}.pth"
        if step_ckpt.is_file():
            return step_ckpt
    return case_out_dir / f"ckpt_step_{step:06d}_{run_name}.pth"


def run_eval_and_summary(
    *,
    case_name: str,
    step: int,
    ckpt: Path,
    commands: List[str],
) -> Tuple[Path, Path]:
    eval_dir = OUT_ROOT / "eval_model_source" / case_name / f"step_{step:03d}"
    eval_json = eval_dir / "Walk_F_freerun_cycles.json"
    summary_json = OUT_ROOT / "eval_model_source" / case_name / f"step_{step:03d}_group_summary.json"
    if not eval_json.is_file():
        eval_dir.mkdir(parents=True, exist_ok=True)
        run_cpu(
            [
                sys.executable,
                str(CPU_EXEC),
                "-m",
                "train.validate.run_freerun_cycles",
                "--teacher",
                "validate/teacher_batches/Walk_F_teacher.json",
                "--model",
                str(ckpt),
                "--rounds",
                "5",
                "--depth",
                "3",
                "--time-index-mode",
                "cycle",
                "--event_clock",
                "auto",
                "--phase_reset_source",
                "none",
                "--contacts_meas_source",
                "model",
                "--lambda_fusion_apply",
                "--log_contacts",
                "--export_direct_arm_probe",
                "--export_joint_direct_geolocal_series",
                "--out",
                str(eval_dir),
                "--force",
            ],
            log_file=LOG_FILE,
            commands=commands,
        )
    if not summary_json.is_file():
        run_cpu(
            [
                sys.executable,
                str(CPU_EXEC),
                "tools/phasea_group_summary.py",
                str(eval_json),
                "--cycle_gte",
                "1",
                "--drop_wrap",
                "--out",
                str(summary_json),
            ],
            log_file=LOG_FILE,
            commands=commands,
        )
    return eval_json, summary_json


def tensor_norms_by_prefix(model: torch.nn.Module, prefixes: Sequence[str]) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not key_matches_prefix(name, prefixes):
            continue
        g = param.grad.detach()
        total += float(g.pow(2).sum().item())
    return float(math.sqrt(total)) if total > 0.0 else 0.0


def activation_summary(model: torch.nn.Module, batch: Mapping[str, Any], *, rollout_common_kwargs: Dict[str, Any], rollout_mode_kwargs: Dict[str, Any]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    hooks = []
    modules = dict(model.named_modules())

    def register(name: str, label: str) -> None:
        module = modules.get(name)
        if module is None:
            return

        def _hook(_mod: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
            if not torch.is_tensor(output):
                return
            x = output.detach().cpu().float()
            stats[f"{label}_mean_abs"] = float(x.abs().mean().item())
            stats[f"{label}_std"] = float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0

        hooks.append(module.register_forward_hook(_hook))

    for module_name, label in HOOK_MODULES.items():
        register(module_name, label)
    try:
        with torch.no_grad():
            _lambda_fusion_loss_rollout(batch=batch, **rollout_common_kwargs, **rollout_mode_kwargs)
    finally:
        for hook in hooks:
            hook.remove()
    return stats


def build_rollout_context(cfg_json: Path, ckpt_in: Path) -> Dict[str, Any]:
    payload = load_json(cfg_json)
    payload["ckpt_in"] = str(ckpt_in)
    cfg = _cfg_from_payload(payload)
    _set_seed(int(cfg.seed))
    device = _resolve_device(cfg.device)
    norm_spec, ds, _ = _build_dataset_and_loader(cfg)
    model, *_meta = _build_posttrain_model_from_ckpt(cfg=cfg, ds=ds, device=device)
    trainer = _build_model_and_trainer(cfg=cfg, ds=ds, model=model, norm_spec=norm_spec)
    train_mode = _resolve_train_mode(cfg)
    _freeze_all(model)
    _unfreeze_for_train_mode(model, cfg, train_mode)
    model.train()
    _params, _names = _select_trainable_params(model)
    loader = DataLoader(ds, batch_size=int(cfg.batch), shuffle=False, drop_last=True, num_workers=0)
    batch = next(iter(loader))
    columns = ("X", "Z")
    rollout_common_kwargs: Dict[str, Any] = {
        "trainer": trainer,
        "model": model,
        "columns": columns,
        "rollout_steps": cfg.rollout_steps,
        "rollout_cycles": cfg.rollout_cycles,
        "include_boundary": cfg.rollout_include_boundary,
        "boundary_weight": cfg.lambda_boundary_weight,
        "random_offset": cfg.rollout_random_offset,
        "time_index_mode": cfg.time_index_mode,
        "time_weight_max": cfg.lambda_time_weight_max,
        "time_weight_mode": cfg.lambda_time_weight_mode,
        "detach_rollout_state": cfg.detach_rollout_state,
        "contact_meas_weight": cfg.contact_meas_weight,
    }
    rollout_mode_kwargs = _build_rollout_mode_kwargs(cfg, train_mode)
    return {
        "cfg": cfg,
        "trainer": trainer,
        "model": model,
        "batch": batch,
        "rollout_common_kwargs": rollout_common_kwargs,
        "rollout_mode_kwargs": rollout_mode_kwargs,
    }


def gradient_audit_for_snapshot(
    *,
    cfg_json: Path,
    ckpt_in: Path,
    log_row: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    ctx = build_rollout_context(cfg_json, ckpt_in)
    trainer = ctx["trainer"]
    model = ctx["model"]
    batch = ctx["batch"]
    rollout_common_kwargs = ctx["rollout_common_kwargs"]
    rollout_mode_kwargs = ctx["rollout_mode_kwargs"]
    if log_row is not None and safe_float(log_row.get("dir_group_norm_used")) > 0.0:
        if safe_float(log_row.get("dir_group_norm_3way_active")) > 0.0:
            trainer._direct_pose_group_norm_ema = {
                "leg": torch.tensor(float(log_row.get("dir_group_norm_leg_ema", 0.0)), dtype=torch.float32),
                "arm": torch.tensor(float(log_row.get("dir_group_norm_arm_ema", 0.0)), dtype=torch.float32),
                "else": torch.tensor(float(log_row.get("dir_group_norm_else_ema", 0.0)), dtype=torch.float32),
            }
        else:
            trainer._direct_pose_group_norm_ema = {
                "leg": torch.tensor(float(log_row.get("dir_group_norm_leg_ema", 0.0)), dtype=torch.float32),
                "nonleg": torch.tensor(float(log_row.get("dir_group_norm_nonleg_ema", 0.0)), dtype=torch.float32),
            }
    model.zero_grad(set_to_none=True)
    loss, stats, _aux = _lambda_fusion_loss_rollout(
        batch=batch,
        **rollout_common_kwargs,
        **rollout_mode_kwargs,
    )
    loss.backward()
    activations = activation_summary(
        model,
        batch,
        rollout_common_kwargs=rollout_common_kwargs,
        rollout_mode_kwargs=rollout_mode_kwargs,
    )
    return {
        "loss": float(loss.detach().cpu()),
        "stats": {key: safe_float(value) for key, value in stats.items()},
        "grad_norms": {
            "shared_trunk": tensor_norms_by_prefix(model, MODULE_GROUPS["shared_trunk"]),
            "arm_branch": tensor_norms_by_prefix(model, MODULE_GROUPS["arm_branch"]),
            "else_branch": tensor_norms_by_prefix(model, MODULE_GROUPS["else_branch"]),
            "leg_branch": tensor_norms_by_prefix(model, MODULE_GROUPS["leg_branch"]),
            "leg_readout": tensor_norms_by_prefix(model, MODULE_GROUPS["leg_readout"]),
        },
        "activations": activations,
    }


def log_row_by_step(log_path: Path) -> Dict[int, Dict[str, Any]]:
    obj = load_json(log_path)
    rows = obj.get("log", []) if isinstance(obj, dict) else []
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        step = int(safe_float(row.get("step")))
        out[step] = row
    return out


def collect_probe_case(
    *,
    case_name: str,
    ckpt_in: Path,
    commands: List[str],
) -> Dict[str, Any]:
    cfg_json, out_dir, run_name = make_probe_config(case_name, ckpt_in)
    ckpt_last = run_probe_train(case_name, cfg_json, out_dir, run_name, commands)
    log_path = out_dir / f"posttrain_log_{run_name}.json"
    rows_by_step = log_row_by_step(log_path)
    start_state, _ = state_and_cfg(snapshot_ckpt(out_dir, run_name, 0))
    snapshots: Dict[str, Any] = {}

    for step in SNAPSHOT_STEPS:
        ckpt = snapshot_ckpt(out_dir, run_name, step) if step < 60 else snapshot_ckpt(out_dir, run_name, 60)
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing snapshot checkpoint: {ckpt}")
        eval_json, summary_json = run_eval_and_summary(case_name=case_name, step=step, ckpt=ckpt, commands=commands)
        cur_state, _ = state_and_cfg(ckpt)
        grad_audit = gradient_audit_for_snapshot(cfg_json=cfg_json, ckpt_in=ckpt, log_row=rows_by_step.get(step))
        snapshots[str(step)] = {
            "ckpt": str(ckpt),
            "eval_json": str(eval_json),
            "group_summary_json": str(summary_json),
            "metrics": load_group_metrics(summary_json),
            "log_row": rows_by_step.get(step),
            "grad_audit": grad_audit,
            "delta_from_step0": {
                group_name: delta_stats(start_state, cur_state, prefixes)
                for group_name, prefixes in MODULE_GROUPS.items()
            },
        }

    return {
        "cfg_json": str(cfg_json),
        "ckpt_in": str(ckpt_in),
        "out_dir": str(out_dir),
        "run_name": run_name,
        "train_log_json": str(log_path),
        "ckpt_last": str(ckpt_last),
        "snapshots": snapshots,
    }


def overall_metric_table() -> List[Dict[str, Any]]:
    refs = [
        ("baseline_70a", BASELINE_70A_GROUP),
        ("tailk7_70a", TAIL_70A_GROUP),
        ("baseline_replace", BASELINE_REPLACE_GROUP),
        ("tailk7_current_replace_e2x60", TAIL_CURRENT_REPLACE_GROUP),
        ("tailk7_best_3way_arm125", TAIL_BEST3WAY_GROUP),
    ]
    rows = []
    for name, path in refs:
        rows.append({"name": name, "metrics": load_group_metrics(path), "source": str(path)})
    return rows


def probe_comparison(case_a: Mapping[str, Any], case_b: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for step in SNAPSHOT_STEPS:
        a = case_a["snapshots"][str(step)]
        b = case_b["snapshots"][str(step)]
        out[str(step)] = {
            "arm_train_dir_base": {
                "baseline_entry": safe_float(a["grad_audit"]["stats"].get("dir_arm_base")),
                "tailk7_entry": safe_float(b["grad_audit"]["stats"].get("dir_arm_base")),
                "delta_tail_minus_base": safe_float(b["grad_audit"]["stats"].get("dir_arm_base"))
                - safe_float(a["grad_audit"]["stats"].get("dir_arm_base")),
            },
            "arm_eval": {
                pct: {
                    "baseline_entry": safe_float(a["metrics"]["arm"].get(pct)),
                    "tailk7_entry": safe_float(b["metrics"]["arm"].get(pct)),
                    "delta_tail_minus_base": safe_float(b["metrics"]["arm"].get(pct))
                    - safe_float(a["metrics"]["arm"].get(pct)),
                }
                for pct in ("mean", "p90", "p95")
            },
            "grad_norms": {
                key: {
                    "baseline_entry": safe_float(a["grad_audit"]["grad_norms"].get(key)),
                    "tailk7_entry": safe_float(b["grad_audit"]["grad_norms"].get(key)),
                    "delta_tail_minus_base": safe_float(b["grad_audit"]["grad_norms"].get(key))
                    - safe_float(a["grad_audit"]["grad_norms"].get(key)),
                }
                for key in ("shared_trunk", "arm_branch", "else_branch")
            },
            "param_delta": {
                key: {
                    "baseline_entry": safe_float(a["delta_from_step0"][key].get("delta_l2")),
                    "tailk7_entry": safe_float(b["delta_from_step0"][key].get("delta_l2")),
                    "delta_tail_minus_base": safe_float(b["delta_from_step0"][key].get("delta_l2"))
                    - safe_float(a["delta_from_step0"][key].get("delta_l2")),
                }
                for key in ("shared_trunk", "arm_branch", "else_branch")
            },
        }
    return out


def summarize_answers(
    *,
    static_audit: Mapping[str, Any],
    probe_baseline: Mapping[str, Any],
    probe_tail: Mapping[str, Any],
) -> Dict[str, Any]:
    base0 = probe_baseline["snapshots"]["0"]
    base60 = probe_baseline["snapshots"]["60"]
    tail0 = probe_tail["snapshots"]["0"]
    tail60 = probe_tail["snapshots"]["60"]

    base_train_drop = safe_float(base0["grad_audit"]["stats"].get("dir_arm_base")) - safe_float(base60["grad_audit"]["stats"].get("dir_arm_base"))
    tail_train_drop = safe_float(tail0["grad_audit"]["stats"].get("dir_arm_base")) - safe_float(tail60["grad_audit"]["stats"].get("dir_arm_base"))
    base_eval_drop = safe_float(base0["metrics"]["arm"].get("p95")) - safe_float(base60["metrics"]["arm"].get("p95"))
    tail_eval_drop = safe_float(tail0["metrics"]["arm"].get("p95")) - safe_float(tail60["metrics"]["arm"].get("p95"))

    baseline_warmstart_changed = len(static_audit.get("baseline_warmstart_changed_keys", []))
    tail_warmstart_changed = len(static_audit.get("tailk7_warmstart_changed_keys", []))
    arm_delta_base = safe_float(base60["delta_from_step0"]["arm_branch"].get("delta_l2"))
    arm_delta_tail = safe_float(tail60["delta_from_step0"]["arm_branch"].get("delta_l2"))
    trunk_delta_base = safe_float(base60["delta_from_step0"]["shared_trunk"].get("delta_l2"))
    trunk_delta_tail = safe_float(tail60["delta_from_step0"]["shared_trunk"].get("delta_l2"))

    train_side_hard = tail_train_drop <= max(1e-12, 0.75 * base_train_drop)
    eval_mismatch = tail_train_drop > 0.0 and tail_eval_drop <= max(1e-12, 0.5 * base_eval_drop)

    root_cause = []
    if baseline_warmstart_changed > 0 and tail_warmstart_changed == 0:
        root_cause.append("warmstart_entry_semantics_diverged")
    if train_side_hard or arm_delta_tail < arm_delta_base:
        root_cause.append("entry_basin_or_optimizability")
    if eval_mismatch:
        root_cause.append("train_eval_mismatch")

    if not root_cause:
        root_cause.append("evidence_mixed")

    next_step = (
        "stop polishing replace and move upstream to donor-state design"
        if ("entry_basin_or_optimizability" in root_cause or "warmstart_entry_semantics_diverged" in root_cause)
        else "keep replace investigation local"
    )

    return {
        "tailk7_arm_efficiency_significantly_lower_than_baseline": bool(
            tail_eval_drop < max(1e-12, 0.75 * base_eval_drop)
        ),
        "controlled_probe_arm_train_drop_p95": {
            "baseline_entry": {"train_dir_arm_drop": base_train_drop, "eval_arm_p95_drop": base_eval_drop},
            "tailk7_entry": {"train_dir_arm_drop": tail_train_drop, "eval_arm_p95_drop": tail_eval_drop},
        },
        "baseline_vs_tail_step60_param_delta": {
            "arm_branch_delta_l2": {"baseline_entry": arm_delta_base, "tailk7_entry": arm_delta_tail},
            "shared_trunk_delta_l2": {"baseline_entry": trunk_delta_base, "tailk7_entry": trunk_delta_tail},
        },
        "root_cause_tags": root_cause,
        "recommendation": next_step,
        "evidence_notes": {
            "baseline_warmstart_changed_keys": baseline_warmstart_changed,
            "tailk7_warmstart_changed_keys": tail_warmstart_changed,
        },
    }


def table_row(name: str, metrics: Mapping[str, Mapping[str, float]]) -> str:
    def quad(group: str) -> str:
        return " / ".join(fmt(metrics[group].get(pct)) for pct in PCTS)

    return f"| {name} | {quad('all_ex_root')} | {quad('leg')} | {quad('arm')} | {quad('else')} |"


def write_summary_md(
    *,
    static_audit: Mapping[str, Any],
    overall_rows: Sequence[Mapping[str, Any]],
    probe_baseline: Mapping[str, Any],
    probe_tail: Mapping[str, Any],
    answers: Mapping[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# cp015 tailk7 replace efficiency audit")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append(
        f"- baseline historical warmstart changed `{len(static_audit.get('baseline_warmstart_changed_keys', []))}` tensor keys; "
        f"tailk7 warmstart changed `{len(static_audit.get('tailk7_warmstart_changed_keys', []))}` tensor keys."
    )
    root_tags = ", ".join(str(x) for x in answers.get("root_cause_tags", []))
    lines.append(f"- root-cause tags: `{root_tags}`")
    lines.append(
        f"- recommendation: `{answers.get('recommendation')}`"
    )
    lines.append("")
    lines.append("## Main Table")
    lines.append("")
    lines.append("| case | all_ex_root mean/p50/p90/p95 | leg mean/p50/p90/p95 | arm mean/p50/p90/p95 | else mean/p50/p90/p95 |")
    lines.append("|---|---|---|---|---|")
    for row in overall_rows:
        lines.append(table_row(str(row["name"]), row["metrics"]))
    lines.append("")
    lines.append("## Controlled Probe")
    lines.append("")
    lines.append("| step | baseline train arm | tailk7 train arm | baseline arm p95 | tailk7 arm p95 | baseline arm-branch Δ | tailk7 arm-branch Δ |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for step in SNAPSHOT_STEPS:
        base = probe_baseline["snapshots"][str(step)]
        tail = probe_tail["snapshots"][str(step)]
        lines.append(
            f"| {step} | "
            f"{fmt(base['grad_audit']['stats'].get('dir_arm_base'))} | "
            f"{fmt(tail['grad_audit']['stats'].get('dir_arm_base'))} | "
            f"{fmt(base['metrics']['arm'].get('p95'))} | "
            f"{fmt(tail['metrics']['arm'].get('p95'))} | "
            f"{fmt(base['delta_from_step0']['arm_branch'].get('delta_l2'))} | "
            f"{fmt(tail['delta_from_step0']['arm_branch'].get('delta_l2'))} |"
        )
    lines.append("")
    lines.append("## Warmstart Delta")
    lines.append("")
    for item in static_audit.get("baseline_phase_adapt_columns", []):
        lines.append(
            f"- {item['key']}: changed_cols={','.join(str(x['col']) for x in item.get('changed_cols', [])) or 'none'}"
        )
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    assert_exists(
        [
            BASELINE_70A_CKPT,
            BASELINE_70A_GROUP,
            BASELINE_WARMSTART_CKPT,
            BASELINE_WARMSTART_REPORT,
            BASELINE_REPLACE_CKPT,
            BASELINE_REPLACE_GROUP,
            BASELINE_REPLACE_LOG,
            TAIL_70A_CKPT,
            TAIL_70A_GROUP,
            TAIL_WARMSTART_CKPT,
            TAIL_CURRENT_REPLACE_CKPT,
            TAIL_CURRENT_REPLACE_GROUP,
            TAIL_CURRENT_REPLACE_LOG,
            TAIL_BEST3WAY_CKPT,
            TAIL_BEST3WAY_GROUP,
            TAIL_BEST3WAY_LOG,
            PROBE_TEMPLATE_CONFIG,
            CPU_EXEC,
        ]
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    commands: List[str] = []
    static_audit = build_static_audit()
    overall_rows = overall_metric_table()

    probe_baseline = collect_probe_case(
        case_name="baseline_entry",
        ckpt_in=BASELINE_WARMSTART_CKPT,
        commands=commands,
    )
    probe_tail = collect_probe_case(
        case_name="tailk7_entry",
        ckpt_in=TAIL_WARMSTART_CKPT,
        commands=commands,
    )

    comparison = probe_comparison(probe_baseline, probe_tail)
    answers = summarize_answers(
        static_audit=static_audit,
        probe_baseline=probe_baseline,
        probe_tail=probe_tail,
    )

    summary = {
        "run_tag": RUN_TAG,
        "out_root": str(OUT_ROOT),
        "model_root": str(MODEL_ROOT),
        "commands": commands,
        "references": {
            "baseline_70a_ckpt": str(BASELINE_70A_CKPT),
            "baseline_warmstart_ckpt": str(BASELINE_WARMSTART_CKPT),
            "baseline_replace_ckpt": str(BASELINE_REPLACE_CKPT),
            "tailk7_70a_ckpt": str(TAIL_70A_CKPT),
            "tailk7_warmstart_ckpt": str(TAIL_WARMSTART_CKPT),
            "tailk7_current_replace_ckpt": str(TAIL_CURRENT_REPLACE_CKPT),
            "tailk7_best3way_ckpt": str(TAIL_BEST3WAY_CKPT),
        },
        "static_audit": static_audit,
        "overall_rows": overall_rows,
        "controlled_probe": {
            "baseline_entry": probe_baseline,
            "tailk7_entry": probe_tail,
            "comparison": comparison,
        },
        "answers": answers,
    }
    write_json(SUMMARY_JSON, summary)
    write_summary_md(
        static_audit=static_audit,
        overall_rows=overall_rows,
        probe_baseline=probe_baseline,
        probe_tail=probe_tail,
        answers=answers,
    )
    write_json(
        STATUS_JSON,
        {
            "ok": True,
            "summary_json": str(SUMMARY_JSON),
            "summary_md": str(SUMMARY_MD),
        },
    )
    log(f"WROTE {SUMMARY_JSON}")
    log(f"WROTE {SUMMARY_MD}")


if __name__ == "__main__":
    main()
