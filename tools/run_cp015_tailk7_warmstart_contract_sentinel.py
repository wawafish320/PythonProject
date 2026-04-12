#!/usr/bin/env python3
from __future__ import annotations

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
    from train import posttrain as posttrain_mod
except Exception:
    posttrain_mod = None

try:
    import run_cp015_tailk7_replace_efficiency_audit as effprobe
except ModuleNotFoundError:
    from tools import run_cp015_tailk7_replace_efficiency_audit as effprobe


RUN_TAG = "20260402_warmstart_contract_sentinel"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_warmstart_contract_sentinel_{RUN_TAG}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_warmstart_contract_sentinel_{RUN_TAG}"
CONFIG_ROOT = OUT_ROOT / "configs"
LOG_FILE = OUT_ROOT / "lane.log"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
STATUS_JSON = OUT_ROOT / "status.json"

REFERENCE_AUDIT_SUMMARY_JSON = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_efficiency_audit_20260402_arm_efficiency_audit"
    / "summary.json"
)
BASELINE_WARMSTART_REPORT_JSON = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "warmstart"
    / "replace_zerophase_report.json"
)
BASELINE_70A_CKPT = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "70a"
    / "ckpt_last_WalkF_stage7_70a_fromfresh_20260317.pth"
)
BASELINE_WARMSTART_CKPT = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "warmstart"
    / "ckpt_last_70a_replace_zerophase_20260317.pth"
)
TAILK7_DONOR_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth"
)

ADAPTED_WARMSTART_CKPT = (
    MODEL_ROOT
    / "warmstart"
    / f"ckpt_last_cp015_tailk7_70a_replace_baseline_style_{RUN_TAG}.pth"
)
ADAPTED_WARMSTART_REPORT_JSON = OUT_ROOT / "warmstart" / "replace_baseline_style_report.json"

TARGET_KEYS: Tuple[str, ...] = (
    "direct_pose_head.0.weight",
    "direct_pose_leg_head.0.weight",
)
TARGET_COLS: Tuple[int, ...] = (39, 40, 41, 42)
CASE_NAME = "tailk7_adapted_warmstart"
CASE_LABELS: Tuple[str, ...] = (
    "baseline_entry",
    "tailk7_copy_only",
    "tailk7_adapted_warmstart",
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def assert_exists(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required artifact(s):\n" + "\n".join(missing))


def legacy_phase_flag_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "template_config_json": str(effprobe.PROBE_TEMPLATE_CONFIG),
        "raw_has_direct_pose_use_phase_z": False,
        "raw_has_direct_pose_phase_z_mode": False,
        "parsed_check_available": False,
        "parsed_has_direct_pose_use_phase_z": False,
        "parsed_has_direct_pose_phase_z_mode": False,
    }
    payload = load_json(effprobe.PROBE_TEMPLATE_CONFIG)
    report["raw_has_direct_pose_use_phase_z"] = bool("direct_pose_use_phase_z" in payload)
    report["raw_has_direct_pose_phase_z_mode"] = bool("direct_pose_phase_z_mode" in payload)
    if posttrain_mod is None:
        return report
    cfg = posttrain_mod._cfg_from_payload(payload)
    report["parsed_check_available"] = True
    report["parsed_has_direct_pose_use_phase_z"] = bool(hasattr(cfg, "direct_pose_use_phase_z"))
    report["parsed_has_direct_pose_phase_z_mode"] = bool(hasattr(cfg, "direct_pose_phase_z_mode"))
    return report


def strict_key_match(changed: Sequence[Mapping[str, Any]]) -> bool:
    changed_keys = tuple(sorted(str(item.get("key")) for item in changed))
    return changed_keys == tuple(sorted(TARGET_KEYS))


def strict_col_match(col_summaries: Sequence[Mapping[str, Any]]) -> bool:
    for item in col_summaries:
        cols = tuple(sorted(int(col["col"]) for col in item.get("changed_cols", [])))
        if cols != TARGET_COLS:
            return False
    return True


def make_probe_config(case_name: str, ckpt_in: Path) -> Tuple[Path, Path, str]:
    payload = load_json(effprobe.PROBE_TEMPLATE_CONFIG)
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
    effprobe.run_cpu(
        [
            sys.executable,
            str(effprobe.CPU_EXEC),
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
            str(cfg_payload.get("encoder_bundle", effprobe.ENCODER_BUNDLE)),
            "--posttrain_contacts_pretrain_affine_stats",
            str(cfg_payload.get("posttrain_contacts_pretrain_affine_stats", effprobe.AFFINE_STATS)),
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


def run_eval_and_summary(*, case_name: str, step: int, ckpt: Path, commands: List[str]) -> Tuple[Path, Path]:
    eval_dir = OUT_ROOT / "eval_model_source" / case_name / f"step_{step:03d}"
    eval_json = eval_dir / "Walk_F_freerun_cycles.json"
    summary_json = OUT_ROOT / "eval_model_source" / case_name / f"step_{step:03d}_group_summary.json"
    if not eval_json.is_file():
        eval_dir.mkdir(parents=True, exist_ok=True)
        effprobe.run_cpu(
            [
                sys.executable,
                str(effprobe.CPU_EXEC),
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
        effprobe.run_cpu(
            [
                sys.executable,
                str(effprobe.CPU_EXEC),
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


def collect_probe_case(*, case_name: str, ckpt_in: Path, commands: List[str]) -> Dict[str, Any]:
    cfg_json, out_dir, run_name = make_probe_config(case_name, ckpt_in)
    ckpt_last = run_probe_train(case_name, cfg_json, out_dir, run_name, commands)
    log_path = out_dir / f"posttrain_log_{run_name}.json"
    rows_by_step = effprobe.log_row_by_step(log_path)
    start_state, _ = effprobe.state_and_cfg(snapshot_ckpt(out_dir, run_name, 0))
    snapshots: Dict[str, Any] = {}

    for step in effprobe.SNAPSHOT_STEPS:
        ckpt = snapshot_ckpt(out_dir, run_name, step)
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing snapshot checkpoint: {ckpt}")
        eval_json, summary_json = run_eval_and_summary(case_name=case_name, step=step, ckpt=ckpt, commands=commands)
        cur_state, _ = effprobe.state_and_cfg(ckpt)
        grad_audit = effprobe.gradient_audit_for_snapshot(cfg_json=cfg_json, ckpt_in=ckpt, log_row=rows_by_step.get(step))
        snapshots[str(step)] = {
            "ckpt": str(ckpt),
            "eval_json": str(eval_json),
            "group_summary_json": str(summary_json),
            "metrics": effprobe.load_group_metrics(summary_json),
            "log_row": rows_by_step.get(step),
            "grad_audit": grad_audit,
            "delta_from_step0": {
                group_name: effprobe.delta_stats(start_state, cur_state, prefixes)
                for group_name, prefixes in effprobe.MODULE_GROUPS.items()
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


def create_adapted_warmstart() -> Dict[str, Any]:
    if ADAPTED_WARMSTART_CKPT.is_file() and ADAPTED_WARMSTART_REPORT_JSON.is_file():
        return load_json(ADAPTED_WARMSTART_REPORT_JSON)

    baseline_report = load_json(BASELINE_WARMSTART_REPORT_JSON)
    baseline_70a_state, _ = effprobe.state_and_cfg(BASELINE_70A_CKPT)
    baseline_ws_state, _ = effprobe.state_and_cfg(BASELINE_WARMSTART_CKPT)
    donor_obj = torch.load(TAILK7_DONOR_CKPT, map_location="cpu")
    if not isinstance(donor_obj, dict) or "model" not in donor_obj:
        raise RuntimeError(f"unexpected checkpoint format: {TAILK7_DONOR_CKPT}")
    donor_state = donor_obj["model"]
    if not isinstance(donor_state, dict):
        raise RuntimeError(f"unexpected state_dict format: {TAILK7_DONOR_CKPT}")

    template_changed_keys = effprobe.changed_tensor_keys(baseline_70a_state, baseline_ws_state)
    template_col_summaries = [
        effprobe.column_diff_summary(baseline_70a_state, baseline_ws_state, key)
        for key in TARGET_KEYS
    ]
    template_strict_only_keys = strict_key_match(template_changed_keys)
    template_strict_only_cols = strict_col_match(template_col_summaries)
    if (not template_strict_only_keys) or (not template_strict_only_cols):
        raise RuntimeError(
            "baseline warmstart tensor adaptation is not strictly limited to the expected two keys / four cols; "
            "inspect the report instead of silently assuming the contract."
        )

    adapted_model = dict(donor_state)
    applied: List[Dict[str, Any]] = []
    for key in TARGET_KEYS:
        donor_tensor = donor_state.get(key)
        base0_tensor = baseline_70a_state.get(key)
        basews_tensor = baseline_ws_state.get(key)
        if not (torch.is_tensor(donor_tensor) and torch.is_tensor(base0_tensor) and torch.is_tensor(basews_tensor)):
            raise RuntimeError(f"missing tensor for key={key}")
        if donor_tensor.shape != base0_tensor.shape or donor_tensor.shape != basews_tensor.shape:
            raise RuntimeError(
                f"shape mismatch for key={key}: donor={tuple(donor_tensor.shape)} "
                f"baseline70a={tuple(base0_tensor.shape)} baselinews={tuple(basews_tensor.shape)}"
            )
        delta = basews_tensor.detach().cpu().float() - base0_tensor.detach().cpu().float()
        donor_float = donor_tensor.detach().cpu().float()
        adapted_float = donor_float + delta
        adapted_model[key] = adapted_float.to(dtype=donor_tensor.dtype)

        applied_cols = effprobe.column_diff_summary({key: donor_float}, {key: adapted_float}, key)
        applied.append(
            {
                "key": key,
                "shape": list(donor_tensor.shape),
                "delta_l2": float((adapted_float - donor_float).norm().item()),
                "max_abs": float((adapted_float - donor_float).abs().max().item()),
                "changed_cols": applied_cols.get("changed_cols", []),
                "changed_col_count": int(applied_cols.get("changed_col_count", 0)),
            }
        )

    out_obj = dict(donor_obj)
    out_obj["model"] = adapted_model
    ADAPTED_WARMSTART_CKPT.parent.mkdir(parents=True, exist_ok=True)
    ADAPTED_WARMSTART_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, ADAPTED_WARMSTART_CKPT)

    changed_vs_donor = effprobe.changed_tensor_keys(donor_state, adapted_model)
    col_vs_donor = [
        effprobe.column_diff_summary(donor_state, adapted_model, key)
        for key in TARGET_KEYS
    ]
    report = {
        "source_ckpt": str(TAILK7_DONOR_CKPT),
        "output_ckpt": str(ADAPTED_WARMSTART_CKPT),
        "baseline_template_source_ckpt": str(BASELINE_70A_CKPT),
        "baseline_template_warmstart_ckpt": str(BASELINE_WARMSTART_CKPT),
        "baseline_template_report_json": str(BASELINE_WARMSTART_REPORT_JSON),
        "baseline_template_report": baseline_report,
        "legacy_phase_key_liveness": legacy_phase_flag_report(),
        "applied_strategy": {
            "type": "add_exact_baseline_warmstart_delta",
            "description": (
                "Reconstruct baseline-style warmstart adaptation as "
                "(baseline_warmstart - baseline_70a) pasted onto the same tensor keys/cols in the tailk7 donor."
            ),
            "checkpoint_posttrain_cfg_left_untouched": True,
        },
        "baseline_template_changed_tensor_keys": template_changed_keys,
        "baseline_template_col_summaries": template_col_summaries,
        "baseline_template_strict_only_two_keys": bool(template_strict_only_keys),
        "baseline_template_strict_only_cols_39_42": bool(template_strict_only_cols),
        "applied_tensor_deltas": applied,
        "changed_tensor_keys_vs_donor": changed_vs_donor,
        "column_diffs_vs_donor": col_vs_donor,
        "strict_only_two_tensor_keys_changed": bool(strict_key_match(changed_vs_donor)),
        "strict_only_cols_39_42_changed": bool(strict_col_match(col_vs_donor)),
    }
    write_json(ADAPTED_WARMSTART_REPORT_JSON, report)
    return report


def load_reference_cases() -> Dict[str, Any]:
    ref = load_json(REFERENCE_AUDIT_SUMMARY_JSON)
    return {
        "baseline_entry": ref["controlled_probe"]["baseline_entry"],
        "tailk7_copy_only": ref["controlled_probe"]["tailk7_entry"],
        "reference_summary_json": str(REFERENCE_AUDIT_SUMMARY_JSON),
    }


def probe_row(case: Mapping[str, Any], step: int) -> Dict[str, float]:
    snap = case["snapshots"][str(step)]
    grad_audit = snap.get("grad_audit", {})
    stats = grad_audit.get("stats", {})
    grad_norms = grad_audit.get("grad_norms", {})
    metrics = snap.get("metrics", {})
    delta_from_step0 = snap.get("delta_from_step0", {})
    return {
        "dir_arm_base": safe_float(stats.get("dir_arm_base")),
        "arm_mean": safe_float(metrics.get("arm", {}).get("mean")),
        "arm_p90": safe_float(metrics.get("arm", {}).get("p90")),
        "arm_p95": safe_float(metrics.get("arm", {}).get("p95")),
        "shared_trunk_grad": safe_float(grad_norms.get("shared_trunk")),
        "arm_branch_grad": safe_float(grad_norms.get("arm_branch")),
        "shared_trunk_delta": safe_float(delta_from_step0.get("shared_trunk", {}).get("delta_l2")),
        "arm_branch_delta": safe_float(delta_from_step0.get("arm_branch", {}).get("delta_l2")),
    }


def build_comparison_rows(cases: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step in effprobe.SNAPSHOT_STEPS:
        row: Dict[str, Any] = {"step": int(step)}
        for case_name in CASE_LABELS:
            row[case_name] = probe_row(cases[case_name], int(step))
        rows.append(row)
    return rows


def step60_recovery(cases: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    base0 = probe_row(cases["baseline_entry"], 0)
    base60 = probe_row(cases["baseline_entry"], 60)
    copy0 = probe_row(cases["tailk7_copy_only"], 0)
    copy60 = probe_row(cases["tailk7_copy_only"], 60)
    adapt0 = probe_row(cases["tailk7_adapted_warmstart"], 0)
    adapt60 = probe_row(cases["tailk7_adapted_warmstart"], 60)

    def gap_closed(copy_v: float, adapt_v: float, base_v: float) -> float:
        denom = copy_v - base_v
        if abs(denom) <= 1e-12:
            return float("nan")
        return float((copy_v - adapt_v) / denom)

    train_drop_base = base0["dir_arm_base"] - base60["dir_arm_base"]
    train_drop_copy = copy0["dir_arm_base"] - copy60["dir_arm_base"]
    train_drop_adapt = adapt0["dir_arm_base"] - adapt60["dir_arm_base"]

    p95_gap_closed = gap_closed(copy60["arm_p95"], adapt60["arm_p95"], base60["arm_p95"])
    p90_gap_closed = gap_closed(copy60["arm_p90"], adapt60["arm_p90"], base60["arm_p90"])
    mean_gap_closed = gap_closed(copy60["arm_mean"], adapt60["arm_mean"], base60["arm_mean"])
    train_gap_closed = gap_closed(train_drop_copy, train_drop_adapt, train_drop_base)

    significant_recovery = bool(
        safe_float(p95_gap_closed) >= 0.30
        and safe_float(train_gap_closed) >= 0.30
        and adapt60["arm_p95"] < copy60["arm_p95"]
    )
    if significant_recovery:
        conclusion = "warmstart contract is a main contributing factor"
        recommendation = "keep warmstart/replace on the table"
    else:
        conclusion = "the problem mainly remains in donor state / 70a exit basin, not the warmstart phase adaptation itself"
        recommendation = "stop polishing replace and move upstream to donor-state design"

    return {
        "step60": {
            "baseline_entry": base60,
            "tailk7_copy_only": copy60,
            "tailk7_adapted_warmstart": adapt60,
        },
        "step0": {
            "baseline_entry": base0,
            "tailk7_copy_only": copy0,
            "tailk7_adapted_warmstart": adapt0,
        },
        "train_dir_arm_drop": {
            "baseline_entry": train_drop_base,
            "tailk7_copy_only": train_drop_copy,
            "tailk7_adapted_warmstart": train_drop_adapt,
        },
        "gap_closed_fraction": {
            "train_dir_arm_drop": train_gap_closed,
            "arm_mean_step60": mean_gap_closed,
            "arm_p90_step60": p90_gap_closed,
            "arm_p95_step60": p95_gap_closed,
        },
        "significant_recovery": significant_recovery,
        "conclusion": conclusion,
        "recommendation": recommendation,
    }


def write_summary_md(
    *,
    warmstart_report: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
    recovery: Mapping[str, Any],
    adapted_case: Mapping[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# cp015 tailk7 warmstart contract sentinel")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append(
        f"- baseline historical warmstart tensor diff is strict: "
        f"`keys_only={warmstart_report.get('baseline_template_strict_only_two_keys')}` / "
        f"`cols_only_39_42={warmstart_report.get('baseline_template_strict_only_cols_39_42')}`"
    )
    legacy = warmstart_report.get("legacy_phase_key_liveness", {})
    lines.append(
        f"- legacy phase keys are raw-config only and parser-dead: "
        f"`raw=({legacy.get('raw_has_direct_pose_use_phase_z')}, {legacy.get('raw_has_direct_pose_phase_z_mode')})`, "
        f"`parsed=({legacy.get('parsed_has_direct_pose_use_phase_z')}, {legacy.get('parsed_has_direct_pose_phase_z_mode')})`"
    )
    lines.append(f"- conclusion: `{recovery.get('conclusion')}`")
    lines.append(f"- recommendation: `{recovery.get('recommendation')}`")
    lines.append("")
    lines.append("## Warmstart")
    lines.append("")
    lines.append(f"- adapted warmstart ckpt: `{ADAPTED_WARMSTART_CKPT}`")
    for item in warmstart_report.get("applied_tensor_deltas", []):
        cols = ",".join(str(col["col"]) for col in item.get("changed_cols", []))
        lines.append(
            f"- {item['key']}: cols={cols or 'none'} delta_l2={item['delta_l2']:.6f} max_abs={item['max_abs']:.6f}"
        )
    lines.append("")
    lines.append("## Controlled Probe")
    lines.append("")
    lines.append(
        "| step | baseline dir_arm_base | copy dir_arm_base | adapted dir_arm_base | "
        "baseline arm p95 | copy arm p95 | adapted arm p95 | "
        "baseline trunk grad | copy trunk grad | adapted trunk grad | "
        "baseline arm grad | copy arm grad | adapted arm grad | "
        "baseline trunk delta | copy trunk delta | adapted trunk delta | "
        "baseline arm delta | copy arm delta | adapted arm delta |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in comparison_rows:
        base = row["baseline_entry"]
        copy = row["tailk7_copy_only"]
        adapt = row["tailk7_adapted_warmstart"]
        lines.append(
            f"| {row['step']} | "
            f"{base['dir_arm_base']:.6f} | {copy['dir_arm_base']:.6f} | {adapt['dir_arm_base']:.6f} | "
            f"{base['arm_p95']:.6f} | {copy['arm_p95']:.6f} | {adapt['arm_p95']:.6f} | "
            f"{base['shared_trunk_grad']:.6f} | {copy['shared_trunk_grad']:.6f} | {adapt['shared_trunk_grad']:.6f} | "
            f"{base['arm_branch_grad']:.6f} | {copy['arm_branch_grad']:.6f} | {adapt['arm_branch_grad']:.6f} | "
            f"{base['shared_trunk_delta']:.6f} | {copy['shared_trunk_delta']:.6f} | {adapt['shared_trunk_delta']:.6f} | "
            f"{base['arm_branch_delta']:.6f} | {copy['arm_branch_delta']:.6f} | {adapt['arm_branch_delta']:.6f} |"
        )
    lines.append("")
    lines.append("## Step60 Arm Recovery")
    lines.append("")
    lines.append(
        f"- train dir_arm drop gap-closed: `{safe_float(recovery['gap_closed_fraction']['train_dir_arm_drop']):.6f}`"
    )
    lines.append(
        f"- arm mean gap-closed: `{safe_float(recovery['gap_closed_fraction']['arm_mean_step60']):.6f}`"
    )
    lines.append(
        f"- arm p90 gap-closed: `{safe_float(recovery['gap_closed_fraction']['arm_p90_step60']):.6f}`"
    )
    lines.append(
        f"- arm p95 gap-closed: `{safe_float(recovery['gap_closed_fraction']['arm_p95_step60']):.6f}`"
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- probe config: `{adapted_case['cfg_json']}`")
    lines.append(f"- train log: `{adapted_case['train_log_json']}`")
    lines.append(f"- final summary json: `{SUMMARY_JSON}`")
    lines.append(f"- final summary md: `{SUMMARY_MD}`")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    assert_exists(
        [
            REFERENCE_AUDIT_SUMMARY_JSON,
            BASELINE_WARMSTART_REPORT_JSON,
            BASELINE_70A_CKPT,
            BASELINE_WARMSTART_CKPT,
            TAILK7_DONOR_CKPT,
            effprobe.PROBE_TEMPLATE_CONFIG,
            effprobe.CPU_EXEC,
        ]
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    commands: List[str] = []
    warmstart_report = create_adapted_warmstart()
    reference_cases = load_reference_cases()
    adapted_case = collect_probe_case(case_name=CASE_NAME, ckpt_in=ADAPTED_WARMSTART_CKPT, commands=commands)

    cases = {
        "baseline_entry": reference_cases["baseline_entry"],
        "tailk7_copy_only": reference_cases["tailk7_copy_only"],
        "tailk7_adapted_warmstart": adapted_case,
    }
    comparison_rows = build_comparison_rows(cases)
    recovery = step60_recovery(cases)

    adapted_snapshots = adapted_case["snapshots"]
    artifacts = {
        "new_script": str(Path(__file__).resolve()),
        "lane_log": str(LOG_FILE),
        "warmstart_ckpt": str(ADAPTED_WARMSTART_CKPT),
        "warmstart_report_json": str(ADAPTED_WARMSTART_REPORT_JSON),
        "probe_config_json": str(adapted_case["cfg_json"]),
        "ckpt_last": str(adapted_case["ckpt_last"]),
        "train_log_json": str(adapted_case["train_log_json"]),
        "step_ckpts": [adapted_snapshots[str(step)]["ckpt"] for step in effprobe.SNAPSHOT_STEPS],
        "eval_jsons": [adapted_snapshots[str(step)]["eval_json"] for step in effprobe.SNAPSHOT_STEPS],
        "group_summary_jsons": [adapted_snapshots[str(step)]["group_summary_json"] for step in effprobe.SNAPSHOT_STEPS],
        "summary_json": str(SUMMARY_JSON),
        "summary_md": str(SUMMARY_MD),
        "status_json": str(STATUS_JSON),
    }
    summary = {
        "run_tag": RUN_TAG,
        "out_root": str(OUT_ROOT),
        "model_root": str(MODEL_ROOT),
        "self_command": f"{sys.executable} {Path(__file__).resolve()}",
        "commands": commands,
        "artifacts": artifacts,
        "references": {
            "reference_summary_json": reference_cases["reference_summary_json"],
            "baseline_entry_case": "controlled_probe.baseline_entry",
            "tailk7_copy_only_case": "controlled_probe.tailk7_entry",
        },
        "warmstart": warmstart_report,
        "controlled_probe": {
            "baseline_entry": reference_cases["baseline_entry"],
            "tailk7_copy_only": reference_cases["tailk7_copy_only"],
            "tailk7_adapted_warmstart": adapted_case,
        },
        "comparison_rows": comparison_rows,
        "recovery": recovery,
    }
    write_json(SUMMARY_JSON, summary)
    write_summary_md(
        warmstart_report=warmstart_report,
        comparison_rows=comparison_rows,
        recovery=recovery,
        adapted_case=adapted_case,
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
