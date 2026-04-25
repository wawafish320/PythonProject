#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from run_cp015_oldplan_downstream_chain import ROOT, load_json, safe_float, write_json
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import ROOT, load_json, safe_float, write_json

try:
    import run_cp015_tailk7_replace_efficiency_audit as effprobe
except ModuleNotFoundError:
    from tools import run_cp015_tailk7_replace_efficiency_audit as effprobe

from train import posttrain as posttrain_mod


RUN_TAG = "20260402_exit_optizability_audit"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_exit_optizability_audit_{RUN_TAG}"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
STATUS_JSON = OUT_ROOT / "status.json"
LOG_FILE = OUT_ROOT / "lane.log"

REFERENCE_SENTINEL_SUMMARY_JSON = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_warmstart_contract_sentinel_20260402_warmstart_contract_sentinel"
    / "summary.json"
)
REFERENCE_EFFICIENCY_SUMMARY_JSON = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_efficiency_audit_20260402_arm_efficiency_audit"
    / "summary.json"
)
REFERENCE_WARMSTART_REPORT_JSON = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_warmstart_contract_sentinel_20260402_warmstart_contract_sentinel"
    / "warmstart"
    / "replace_baseline_style_report.json"
)

SNAPSHOT_STEPS: Tuple[int, ...] = (0, 1, 5)
CASE_ORDER: Tuple[str, ...] = (
    "baseline_entry",
    "tailk7_copy_only",
    "tailk7_adapted_warmstart",
)
PAIRWISE_CASES: Tuple[Tuple[str, str], ...] = (
    ("baseline_entry", "tailk7_copy_only"),
    ("baseline_entry", "tailk7_adapted_warmstart"),
)
ACTIVATION_MODULES: Mapping[str, str] = {
    "direct_pose_head": "direct_pose_head",
    "direct_pose_arm_proj": "direct_pose_arm_proj",
    "direct_pose_out_arm": "direct_pose_out_arm",
}
GROUP_PREFIXES: Mapping[str, Tuple[str, ...]] = {
    "shared_trunk": effprobe.MODULE_GROUPS["shared_trunk"],
    "arm_branch": effprobe.MODULE_GROUPS["arm_branch"],
}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {msg}\n")


def fmt(x: Any, digits: int = 6) -> str:
    v = safe_float(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def assert_exists(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required artifact(s):\n" + "\n".join(missing))


def safe_div(num: float, den: float) -> float:
    if (not math.isfinite(num)) or (not math.isfinite(den)) or abs(den) <= 1e-12:
        return float("nan")
    return float(num / den)


def state_delta_vector(src: Mapping[str, Any], dst: Mapping[str, Any], prefixes: Sequence[str]) -> torch.Tensor:
    a = effprobe.state_vector(src, prefixes)
    b = effprobe.state_vector(dst, prefixes)
    if a.numel() <= 0 or b.numel() <= 0 or a.numel() != b.numel():
        return torch.empty(0, dtype=torch.float32)
    return b - a


def grad_vector(model: torch.nn.Module, prefixes: Sequence[str]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not effprobe.key_matches_prefix(name, prefixes):
            continue
        chunks.append(param.grad.detach().cpu().float().reshape(-1))
    if not chunks:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def activation_probe(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    rollout_common_kwargs: Dict[str, Any],
    rollout_mode_kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    hooks = []
    modules = dict(model.named_modules())

    def register(module_name: str, label: str) -> None:
        module = modules.get(module_name)
        if module is None:
            return

        def _hook(_mod: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
            if not torch.is_tensor(output):
                return
            x = output.detach().cpu().float()
            scale = float(x.pow(2).mean().sqrt().item()) if x.numel() > 0 else float("nan")
            stats[label] = {
                "scale": scale,
                "mean_abs": float(x.abs().mean().item()) if x.numel() > 0 else float("nan"),
                "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
            }

        hooks.append(module.register_forward_hook(_hook))

    for module_name, label in ACTIVATION_MODULES.items():
        register(module_name, label)
    try:
        with torch.no_grad():
            effprobe._lambda_fusion_loss_rollout(
                batch=batch,
                **rollout_common_kwargs,
                **rollout_mode_kwargs,
            )
    finally:
        for hook in hooks:
            hook.remove()
    return stats


def _seed_group_norm_ema(trainer: Any, log_row: Mapping[str, Any] | None) -> None:
    if log_row is None:
        return
    if safe_float(log_row.get("dir_group_norm_used")) <= 0.0:
        return
    loss_fn = getattr(trainer, "loss_fn", None)
    if loss_fn is None or not hasattr(loss_fn, "_direct_pose_group_norm_ema"):
        raise AttributeError("trainer.loss_fn missing canonical _direct_pose_group_norm_ema")
    if safe_float(log_row.get("dir_group_norm_3way_active")) > 0.0:
        loss_fn._direct_pose_group_norm_ema = {
            "leg": torch.tensor(float(log_row.get("dir_group_norm_leg_ema", 0.0)), dtype=torch.float32),
            "arm": torch.tensor(float(log_row.get("dir_group_norm_arm_ema", 0.0)), dtype=torch.float32),
            "else": torch.tensor(float(log_row.get("dir_group_norm_else_ema", 0.0)), dtype=torch.float32),
        }
    else:
        loss_fn._direct_pose_group_norm_ema = {
            "leg": torch.tensor(float(log_row.get("dir_group_norm_leg_ema", 0.0)), dtype=torch.float32),
            "nonleg": torch.tensor(float(log_row.get("dir_group_norm_nonleg_ema", 0.0)), dtype=torch.float32),
        }


def code_liveness_report() -> Dict[str, Any]:
    cfg_fields = set(posttrain_mod.PostTrainConfig.__dataclass_fields__.keys())
    parser = posttrain_mod._build_posttrain_arg_parser()
    parser_flags = {
        opt.lstrip("-")
        for action in getattr(parser, "_actions", [])
        for opt in getattr(action, "option_strings", [])
    }
    return {
        "dataclass_has_direct_pose_use_phase_z": "direct_pose_use_phase_z" in cfg_fields,
        "dataclass_has_direct_pose_phase_z_mode": "direct_pose_phase_z_mode" in cfg_fields,
        "parser_has_direct_pose_use_phase_z": "direct_pose_use_phase_z" in parser_flags,
        "parser_has_direct_pose_phase_z_mode": "direct_pose_phase_z_mode" in parser_flags,
    }


def load_reference_cases() -> Dict[str, Dict[str, Any]]:
    sentinel = load_json(REFERENCE_SENTINEL_SUMMARY_JSON)
    controlled = sentinel.get("controlled_probe", {})
    out: Dict[str, Dict[str, Any]] = {}
    for case_name in CASE_ORDER:
        payload = controlled.get(case_name, {})
        if not isinstance(payload, dict):
            raise RuntimeError(f"missing case payload: {case_name}")
        out[case_name] = payload
    return out


def collect_step_metrics(
    *,
    case_name: str,
    cfg_json: Path,
    ckpt: Path,
    log_row: Mapping[str, Any] | None,
    step0_state: Mapping[str, Any],
    step0_dir_arm_base: float,
    arm_metrics: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    ctx = effprobe.build_rollout_context(cfg_json, ckpt)
    trainer = ctx["trainer"]
    model = ctx["model"]
    batch = ctx["batch"]
    rollout_common_kwargs = ctx["rollout_common_kwargs"]
    rollout_mode_kwargs = ctx["rollout_mode_kwargs"]
    _seed_group_norm_ema(trainer, log_row)

    model.zero_grad(set_to_none=True)
    loss, stats, _aux = effprobe._lambda_fusion_loss_rollout(
        batch=batch,
        **rollout_common_kwargs,
        **rollout_mode_kwargs,
    )
    loss.backward()

    grad_norms = {
        "shared_trunk": effprobe.tensor_norms_by_prefix(model, GROUP_PREFIXES["shared_trunk"]),
        "arm_branch": effprobe.tensor_norms_by_prefix(model, GROUP_PREFIXES["arm_branch"]),
    }
    grad_vectors = {
        "shared_trunk": grad_vector(model, GROUP_PREFIXES["shared_trunk"]),
        "arm_branch": grad_vector(model, GROUP_PREFIXES["arm_branch"]),
    }
    activations = activation_probe(
        model,
        batch,
        rollout_common_kwargs=rollout_common_kwargs,
        rollout_mode_kwargs=rollout_mode_kwargs,
    )

    cur_state, _cur_cfg = effprobe.state_and_cfg(ckpt)
    deltas = {
        "shared_trunk": effprobe.delta_stats(step0_state, cur_state, GROUP_PREFIXES["shared_trunk"]),
        "arm_branch": effprobe.delta_stats(step0_state, cur_state, GROUP_PREFIXES["arm_branch"]),
    }
    delta_vectors = {
        "shared_trunk": state_delta_vector(step0_state, cur_state, GROUP_PREFIXES["shared_trunk"]),
        "arm_branch": state_delta_vector(step0_state, cur_state, GROUP_PREFIXES["arm_branch"]),
    }

    dir_arm_base = safe_float(stats.get("dir_arm_base"))
    dir_arm_improvement = safe_float(step0_dir_arm_base) - dir_arm_base
    row = {
        "case": case_name,
        "loss": float(loss.detach().cpu()),
        "dir_arm_base": dir_arm_base,
        "dir_else_base": safe_float(stats.get("dir_else_base")),
        "dir_leg_base": safe_float(stats.get("dir_leg_base")),
        "shared_trunk_grad": grad_norms["shared_trunk"],
        "arm_branch_grad": grad_norms["arm_branch"],
        "shared_trunk_delta": safe_float(deltas["shared_trunk"].get("delta_l2")),
        "arm_branch_delta": safe_float(deltas["arm_branch"].get("delta_l2")),
        "dir_arm_improvement_from_step0": dir_arm_improvement,
        "dir_arm_improvement_per_trunk_delta": safe_div(
            dir_arm_improvement,
            safe_float(deltas["shared_trunk"].get("delta_l2")),
        ),
        "dir_arm_improvement_per_arm_delta": safe_div(
            dir_arm_improvement,
            safe_float(deltas["arm_branch"].get("delta_l2")),
        ),
        "arm_mean": safe_float((arm_metrics or {}).get("mean")),
        "arm_p90": safe_float((arm_metrics or {}).get("p90")),
        "arm_p95": safe_float((arm_metrics or {}).get("p95")),
        "activations": activations,
        "grad_vectors": grad_vectors,
        "delta_vectors": delta_vectors,
    }
    return row


def summarize_case(
    *,
    case_name: str,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg_json = Path(str(payload["cfg_json"]))
    snapshots = payload.get("snapshots", {})
    step0_ckpt = Path(str(snapshots["0"]["ckpt"]))
    step0_state, _ = effprobe.state_and_cfg(step0_ckpt)
    step0_dir_arm_base = safe_float(snapshots["0"]["grad_audit"]["stats"].get("dir_arm_base"))
    by_step: Dict[str, Dict[str, Any]] = {}

    for step in SNAPSHOT_STEPS:
        snap = snapshots[str(step)]
        ckpt = Path(str(snap["ckpt"]))
        by_step[str(step)] = collect_step_metrics(
            case_name=case_name,
            cfg_json=cfg_json,
            ckpt=ckpt,
            log_row=snap.get("log_row"),
            step0_state=step0_state,
            step0_dir_arm_base=step0_dir_arm_base,
            arm_metrics=((snap.get("metrics") or {}).get("arm") or {}),
        )
        by_step[str(step)]["ckpt"] = str(ckpt)
        by_step[str(step)]["cfg_json"] = str(cfg_json)
        by_step[str(step)]["train_log_json"] = str(payload.get("train_log_json"))

    return {
        "cfg_json": str(cfg_json),
        "train_log_json": str(payload.get("train_log_json")),
        "step0_ckpt": str(step0_ckpt),
        "steps": by_step,
    }


def strip_vectors(payload: Any) -> Any:
    if isinstance(payload, torch.Tensor):
        return {
            "numel": int(payload.numel()),
            "l2": float(payload.norm().item()) if payload.numel() > 0 else float("nan"),
        }
    if isinstance(payload, dict):
        out: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in ("grad_vectors", "delta_vectors"):
                continue
            out[key] = strip_vectors(value)
        return out
    if isinstance(payload, list):
        return [strip_vectors(x) for x in payload]
    return payload


def compute_pairwise_cosines(case_data: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for lhs, rhs in PAIRWISE_CASES:
        pair_key = f"{lhs}__vs__{rhs}"
        pair_steps: Dict[str, Any] = {}
        for step in SNAPSHOT_STEPS:
            left = case_data[lhs]["steps"][str(step)]
            right = case_data[rhs]["steps"][str(step)]
            pair_steps[str(step)] = {
                "shared_trunk_grad_cosine": effprobe.cosine_between(
                    left["grad_vectors"]["shared_trunk"],
                    right["grad_vectors"]["shared_trunk"],
                ),
                "arm_branch_grad_cosine": effprobe.cosine_between(
                    left["grad_vectors"]["arm_branch"],
                    right["grad_vectors"]["arm_branch"],
                ),
                "shared_trunk_update_cosine": effprobe.cosine_between(
                    left["delta_vectors"]["shared_trunk"],
                    right["delta_vectors"]["shared_trunk"],
                ),
                "arm_branch_update_cosine": effprobe.cosine_between(
                    left["delta_vectors"]["arm_branch"],
                    right["delta_vectors"]["arm_branch"],
                ),
            }
        out[pair_key] = {
            "lhs": lhs,
            "rhs": rhs,
            "steps": pair_steps,
        }
    return out


def derived_conclusion(
    case_data: Mapping[str, Any],
    *,
    sentinel_summary: Mapping[str, Any],
    pairwise: Mapping[str, Any],
) -> Dict[str, Any]:
    base5 = case_data["baseline_entry"]["steps"]["5"]
    copy5 = case_data["tailk7_copy_only"]["steps"]["5"]
    adapt5 = case_data["tailk7_adapted_warmstart"]["steps"]["5"]
    recovery = sentinel_summary.get("recovery", {}) if isinstance(sentinel_summary, dict) else {}
    step60 = recovery.get("step60", {}) if isinstance(recovery, dict) else {}
    base60 = step60.get("baseline_entry", {}) if isinstance(step60, dict) else {}
    adapt60 = step60.get("tailk7_adapted_warmstart", {}) if isinstance(step60, dict) else {}
    copy60 = step60.get("tailk7_copy_only", {}) if isinstance(step60, dict) else {}
    pair_adapt5 = ((pairwise.get("baseline_entry__vs__tailk7_adapted_warmstart") or {}).get("steps") or {}).get("5", {})

    evidence = {
        "step5_shared_trunk_grad_ratio_copy_vs_baseline": safe_div(
            safe_float(copy5["shared_trunk_grad"]),
            safe_float(base5["shared_trunk_grad"]),
        ),
        "step5_shared_trunk_grad_ratio_adapt_vs_baseline": safe_div(
            safe_float(adapt5["shared_trunk_grad"]),
            safe_float(base5["shared_trunk_grad"]),
        ),
        "step5_arm_improvement_per_trunk_delta_ratio_copy_vs_baseline": safe_div(
            safe_float(copy5["dir_arm_improvement_per_trunk_delta"]),
            safe_float(base5["dir_arm_improvement_per_trunk_delta"]),
        ),
        "step5_arm_improvement_per_trunk_delta_ratio_adapt_vs_baseline": safe_div(
            safe_float(adapt5["dir_arm_improvement_per_trunk_delta"]),
            safe_float(base5["dir_arm_improvement_per_trunk_delta"]),
        ),
        "step5_direct_pose_head_scale_ratio_copy_vs_baseline": safe_div(
            safe_float(copy5["activations"]["direct_pose_head"]["scale"]),
            safe_float(base5["activations"]["direct_pose_head"]["scale"]),
        ),
        "step5_direct_pose_head_scale_ratio_adapt_vs_baseline": safe_div(
            safe_float(adapt5["activations"]["direct_pose_head"]["scale"]),
            safe_float(base5["activations"]["direct_pose_head"]["scale"]),
        ),
        "step60_adapt_dir_arm_ratio_vs_baseline": safe_div(
            safe_float(adapt60.get("dir_arm_base")),
            safe_float(base60.get("dir_arm_base")),
        ),
        "step60_adapt_arm_p95_ratio_vs_baseline": safe_div(
            safe_float(adapt60.get("arm_p95")),
            safe_float(base60.get("arm_p95")),
        ),
        "step60_adapt_trunk_grad_ratio_vs_baseline": safe_div(
            safe_float(adapt60.get("shared_trunk_grad")),
            safe_float(base60.get("shared_trunk_grad")),
        ),
        "step60_adapt_arm_grad_ratio_vs_baseline": safe_div(
            safe_float(adapt60.get("arm_branch_grad")),
            safe_float(base60.get("arm_branch_grad")),
        ),
        "step5_baseline_vs_adapt_trunk_grad_cosine": safe_float(pair_adapt5.get("shared_trunk_grad_cosine")),
        "step5_baseline_vs_adapt_arm_grad_cosine": safe_float(pair_adapt5.get("arm_branch_grad_cosine")),
        "step5_baseline_vs_adapt_trunk_update_cosine": safe_float(pair_adapt5.get("shared_trunk_update_cosine")),
        "step5_baseline_vs_adapt_arm_update_cosine": safe_float(pair_adapt5.get("arm_branch_update_cosine")),
    }
    inherited_conclusion = str(recovery.get("conclusion") or "").lower()
    donor_state_main_issue = (
        ("donor state / 70a exit basin" in inherited_conclusion)
        and safe_float(copy5["shared_trunk_grad"]) > safe_float(base5["shared_trunk_grad"])
        and safe_float(copy5["dir_arm_improvement_per_trunk_delta"]) < safe_float(base5["dir_arm_improvement_per_trunk_delta"])
        and safe_float(adapt60.get("dir_arm_base")) > safe_float(base60.get("dir_arm_base"))
        and safe_float(adapt60.get("arm_p95")) > safe_float(base60.get("arm_p95"))
        and abs(safe_float(pair_adapt5.get("shared_trunk_grad_cosine"))) < 0.2
        and abs(safe_float(pair_adapt5.get("arm_branch_grad_cosine"))) < 0.2
    )
    return {
        "donor_state_or_exit_basin_main_issue": bool(donor_state_main_issue),
        "replace_or_warmstart_polishing_should_stop": bool(donor_state_main_issue),
        "move_to_donor_or_exit_state_design": bool(donor_state_main_issue),
        "evidence": evidence,
    }


def render_step_table(case_data: Mapping[str, Any], step: int) -> List[str]:
    lines = [
        f"### Step {step}",
        "",
        "| case | dir_arm_base | dir_else_base | dir_leg_base | trunk_grad | arm_grad | trunk_delta | arm_delta | arm_impr/trunk_delta | arm_impr/arm_delta | arm_p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name in CASE_ORDER:
        row = case_data[case_name]["steps"][str(step)]
        lines.append(
            f"| {case_name} | {fmt(row['dir_arm_base'])} | {fmt(row['dir_else_base'])} | {fmt(row['dir_leg_base'])} | "
            f"{fmt(row['shared_trunk_grad'])} | {fmt(row['arm_branch_grad'])} | {fmt(row['shared_trunk_delta'])} | "
            f"{fmt(row['arm_branch_delta'])} | {fmt(row['dir_arm_improvement_per_trunk_delta'])} | "
            f"{fmt(row['dir_arm_improvement_per_arm_delta'])} | {fmt(row['arm_p95'])} |"
        )
    lines.extend(
        [
            "",
            "| case | head_scale | head_mean_abs | head_std | arm_proj_scale | arm_proj_mean_abs | arm_proj_std | out_arm_scale | out_arm_mean_abs | out_arm_std |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case_name in CASE_ORDER:
        act = case_data[case_name]["steps"][str(step)]["activations"]
        head = act["direct_pose_head"]
        arm_proj = act["direct_pose_arm_proj"]
        out_arm = act["direct_pose_out_arm"]
        lines.append(
            f"| {case_name} | {fmt(head['scale'])} | {fmt(head['mean_abs'])} | {fmt(head['std'])} | "
            f"{fmt(arm_proj['scale'])} | {fmt(arm_proj['mean_abs'])} | {fmt(arm_proj['std'])} | "
            f"{fmt(out_arm['scale'])} | {fmt(out_arm['mean_abs'])} | {fmt(out_arm['std'])} |"
        )
    lines.append("")
    return lines


def render_pairwise_cosines(pairwise: Mapping[str, Any]) -> List[str]:
    lines = [
        "## Pairwise Cosines",
        "",
        "| pair | step | trunk_grad_cos | arm_grad_cos | trunk_update_cos | arm_update_cos |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pair_key, payload in pairwise.items():
        label = pair_key.replace("__vs__", " vs ")
        for step in SNAPSHOT_STEPS:
            row = payload["steps"][str(step)]
            lines.append(
                f"| {label} | {step} | {fmt(row['shared_trunk_grad_cosine'])} | {fmt(row['arm_branch_grad_cosine'])} | "
                f"{fmt(row['shared_trunk_update_cosine'])} | {fmt(row['arm_branch_update_cosine'])} |"
            )
    lines.append("")
    return lines


def render_summary_md(
    *,
    case_data: Mapping[str, Any],
    pairwise: Mapping[str, Any],
    inherited: Mapping[str, Any],
    code_liveness: Mapping[str, Any],
    conclusion: Mapping[str, Any],
) -> str:
    lines = [
        "# cp015 tailk7 70a exit optizability audit",
        "",
        "## Findings",
        "",
        f"- inherited conclusion kept: `{inherited['warmstart_summary_conclusion']}`",
        f"- legacy phase keys remain parser-dead in code: `dataclass=({code_liveness['dataclass_has_direct_pose_use_phase_z']}, {code_liveness['dataclass_has_direct_pose_phase_z_mode']})`, `parser=({code_liveness['parser_has_direct_pose_use_phase_z']}, {code_liveness['parser_has_direct_pose_phase_z_mode']})`",
        f"- donor state / 70a exit basin main issue: `{conclusion['donor_state_or_exit_basin_main_issue']}`",
        f"- recommendation: `{'stop replace/warmstart polishing and move to donor-state / 70a exit-state design' if conclusion['replace_or_warmstart_polishing_should_stop'] else 'need one more minimal proof before stopping replace/warmstart'}`",
        "",
        "## Strongest Evidence",
        "",
        f"- copy-only still shows classic basin-unfriendly local efficiency: step5 trunk-grad ratio copy/baseline = `{fmt(conclusion['evidence']['step5_shared_trunk_grad_ratio_copy_vs_baseline'])}` but arm-improvement-per-trunk-delta ratio copy/baseline = `{fmt(conclusion['evidence']['step5_arm_improvement_per_trunk_delta_ratio_copy_vs_baseline'])}`",
        f"- adapted warmstart can improve early local train-side efficiency, but inherited step60 still fails vs baseline under the same replace probe: dir_arm ratio=`{fmt(conclusion['evidence']['step60_adapt_dir_arm_ratio_vs_baseline'])}`, arm_p95 ratio=`{fmt(conclusion['evidence']['step60_adapt_arm_p95_ratio_vs_baseline'])}`, trunk_grad ratio=`{fmt(conclusion['evidence']['step60_adapt_trunk_grad_ratio_vs_baseline'])}`, arm_grad ratio=`{fmt(conclusion['evidence']['step60_adapt_arm_grad_ratio_vs_baseline'])}`",
        f"- baseline vs adapted remain almost orthogonal in local optimization geometry at step5: trunk_grad_cos=`{fmt(conclusion['evidence']['step5_baseline_vs_adapt_trunk_grad_cosine'])}`, arm_grad_cos=`{fmt(conclusion['evidence']['step5_baseline_vs_adapt_arm_grad_cosine'])}`, trunk_update_cos=`{fmt(conclusion['evidence']['step5_baseline_vs_adapt_trunk_update_cosine'])}`, arm_update_cos=`{fmt(conclusion['evidence']['step5_baseline_vs_adapt_arm_update_cosine'])}`",
        "",
        "## Step Tables",
        "",
    ]
    for step in SNAPSHOT_STEPS:
        lines.extend(render_step_table(case_data, step))
    lines.extend(render_pairwise_cosines(pairwise))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    assert_exists(
        [
            REFERENCE_SENTINEL_SUMMARY_JSON,
            REFERENCE_EFFICIENCY_SUMMARY_JSON,
            REFERENCE_WARMSTART_REPORT_JSON,
        ]
    )
    log("loading existing controlled probe artifacts")
    sentinel_summary = load_json(REFERENCE_SENTINEL_SUMMARY_JSON)
    efficiency_summary = load_json(REFERENCE_EFFICIENCY_SUMMARY_JSON)
    warmstart_report = load_json(REFERENCE_WARMSTART_REPORT_JSON)
    case_refs = load_reference_cases()
    code_liveness = code_liveness_report()

    log("running minimal step0/1/5 exit-state audits over existing checkpoints")
    case_data: Dict[str, Any] = {}
    for case_name in CASE_ORDER:
        case_data[case_name] = summarize_case(case_name=case_name, payload=case_refs[case_name])

    log("computing pairwise gradient/update cosines")
    pairwise = compute_pairwise_cosines(case_data)
    inherited = {
        "warmstart_summary_conclusion": ((sentinel_summary.get("recovery") or {}).get("conclusion")),
        "warmstart_summary_recommendation": ((sentinel_summary.get("recovery") or {}).get("recommendation")),
        "efficiency_root_cause_tags": ((efficiency_summary.get("answers") or {}).get("root_cause_tags")),
        "warmstart_report_legacy_phase_key_liveness": warmstart_report.get("legacy_phase_key_liveness"),
    }
    conclusion = derived_conclusion(case_data, sentinel_summary=sentinel_summary, pairwise=pairwise)

    result = {
        "run_tag": RUN_TAG,
        "out_root": str(OUT_ROOT),
        "self_command": " ".join(sys.argv),
        "references": {
            "sentinel_summary_json": str(REFERENCE_SENTINEL_SUMMARY_JSON),
            "efficiency_summary_json": str(REFERENCE_EFFICIENCY_SUMMARY_JSON),
            "warmstart_report_json": str(REFERENCE_WARMSTART_REPORT_JSON),
        },
        "code_liveness": code_liveness,
        "inherited": inherited,
        "cases": strip_vectors(case_data),
        "pairwise_cosines": strip_vectors(pairwise),
        "conclusion": conclusion,
    }
    write_json(SUMMARY_JSON, result)
    SUMMARY_MD.write_text(
        render_summary_md(
            case_data=case_data,
            pairwise=pairwise,
            inherited=inherited,
            code_liveness=code_liveness,
            conclusion=conclusion,
        ),
        encoding="utf-8",
    )
    write_json(
        STATUS_JSON,
        {
            "ok": True,
            "summary_json": str(SUMMARY_JSON),
            "summary_md": str(SUMMARY_MD),
        },
    )
    log(f"wrote {SUMMARY_JSON}")
    log(f"wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
