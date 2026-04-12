#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
RUN_DATE = "20260314"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_stage6_plantransplant_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_stage6_plantransplant_{RUN_DATE}"
SURGERY_ROOT = MODEL_ROOT / "surgery_ckpts"
STAGE6_MODEL_ROOT = MODEL_ROOT / "stage6"
BASELINE_COMPARE_JSON = ROOT / "debug_output" / "_tmp_stage6_basetrain_compare_20260313" / "compare_summary.json"
STAGE6_CONFIG = ROOT / "config" / "posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json"
TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
ENCODER_BUNDLE = ROOT / "models" / "motion_encoder_equiv.pt.best.pt"
AFFINE_STATS = ROOT / "debug_output" / "_tmp_phaseb_affine_20260304" / "affine_fit_mix08" / "affine_stats.json"
PRETRAIN_CLAMP = "1.0"

PLAN_PREFIXES: Tuple[str, ...] = (
    "contact_plan_cell.",
    "contact_plan_head.",
    "contact_plan_time_head.",
    "contact_plan_phase_head.",
    "contact_plan_init_head.",
    "event_clock_gate.",
    "event_clock_corrector.",
)
PLAN_EXACT_KEYS: Tuple[str, ...] = (
    "contact_plan_init_z",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    backbone_case: str
    planstack_case: str
    run_name: str

    @property
    def is_transplant(self) -> bool:
        return self.backbone_case != self.planstack_case


BASELINE_CASES: Tuple[CaseSpec, ...] = (
    CaseSpec(
        name="old_bestfree",
        backbone_case="old_bestfree",
        planstack_case="old_bestfree",
        run_name="old_bestfree_stage6_plantransplant_20260314",
    ),
    CaseSpec(
        name="cp015_bestfree",
        backbone_case="cp015_bestfree",
        planstack_case="cp015_bestfree",
        run_name="cp015_bestfree_stage6_plantransplant_20260314",
    ),
)

TRANSPLANT_CASES: Tuple[CaseSpec, ...] = (
    CaseSpec(
        name="cp015_with_old_planstack",
        backbone_case="cp015_bestfree",
        planstack_case="old_bestfree",
        run_name="cp015_with_old_planstack_stage6_plantransplant_20260314",
    ),
    CaseSpec(
        name="old_with_cp015_planstack",
        backbone_case="old_bestfree",
        planstack_case="cp015_bestfree",
        run_name="old_with_cp015_planstack_stage6_plantransplant_20260314",
    ),
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def is_finite(x: Any) -> bool:
    return math.isfinite(safe_float(x))


def mean(values: Iterable[Any]) -> float:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def fmt(x: Any, digits: int = 6) -> str:
    v = safe_float(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def diff(cur: Any, ref: Any) -> float:
    a = safe_float(cur)
    b = safe_float(ref)
    if not math.isfinite(a) or not math.isfinite(b):
        return float("nan")
    return float(a - b)


def improvement(baseline: Any, cur: Any) -> float:
    base = safe_float(baseline)
    now = safe_float(cur)
    if not math.isfinite(base) or not math.isfinite(now):
        return float("nan")
    return float(base - now)


def ratio(num: Any, den: Any) -> float:
    a = safe_float(num)
    b = safe_float(den)
    if not math.isfinite(a) or not math.isfinite(b) or b == 0.0:
        return float("nan")
    return float(a / b)


def delta_block(cur: Mapping[str, Any], ref: Mapping[str, Any]) -> Dict[str, float]:
    keys = sorted(set(cur.keys()) | set(ref.keys()))
    return {key: diff(cur.get(key), ref.get(key)) for key in keys}


def enrich_init_section(section: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(section)
    if not is_finite(payload.get("leg_over_nonleg")):
        payload["leg_over_nonleg"] = ratio(payload.get("dir_leg_base"), payload.get("dir_nonleg_base"))
    if not is_finite(payload.get("grad_arm_over_else")):
        payload["grad_arm_over_else"] = ratio(
            payload.get("direct_grad_norm_out_arm"),
            payload.get("direct_grad_norm_out_else"),
        )
    return payload


def enrich_stage6_init(payload: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    out["step1"] = enrich_init_section(payload.get("step1", {}))
    out["head20_mean"] = enrich_init_section(payload.get("head20_mean", {}))
    return out


def run_cmd(cmd: Sequence[str], *, log_file: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(str(x) for x in cmd) + "\n")
        fh.flush()
        log("RUN " + " ".join(str(x) for x in cmd))
        proc = subprocess.Popen(
            [str(x) for x in cmd],
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


def extract_stage6_init(log_json: Path, out_json: Path) -> Dict[str, Any]:
    obj = load_json(log_json)
    rows = obj.get("log", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"missing log rows in {log_json}")

    def row_payload(row: Mapping[str, Any]) -> Dict[str, float]:
        dir_leg_base = safe_float(row.get("dir_leg_base"))
        dir_nonleg_base = safe_float(row.get("dir_nonleg_base"))
        grad_arm = safe_float(row.get("direct_grad_norm_out_arm"))
        grad_else = safe_float(row.get("direct_grad_norm_out_else"))
        return {
            "step": safe_float(row.get("step")),
            "dir_leg_base": dir_leg_base,
            "dir_nonleg_base": dir_nonleg_base,
            "leg_over_nonleg": ratio(dir_leg_base, dir_nonleg_base),
            "arm_over_else": safe_float(row.get("arm_over_else")),
            "direct_grad_norm_out_arm": grad_arm,
            "direct_grad_norm_out_else": grad_else,
            "grad_arm_over_else": ratio(grad_arm, grad_else),
        }

    step1 = row_payload(rows[0])
    head = [row_payload(row) for row in rows[: min(20, len(rows))]]
    head20 = {
        "dir_leg_base": mean(r["dir_leg_base"] for r in head),
        "dir_nonleg_base": mean(r["dir_nonleg_base"] for r in head),
        "leg_over_nonleg": mean(r["leg_over_nonleg"] for r in head),
        "arm_over_else": mean(r["arm_over_else"] for r in head),
        "direct_grad_norm_out_arm": mean(r["direct_grad_norm_out_arm"] for r in head),
        "direct_grad_norm_out_else": mean(r["direct_grad_norm_out_else"] for r in head),
        "grad_arm_over_else": mean(r["grad_arm_over_else"] for r in head),
    }
    payload = {
        "source": str(log_json),
        "rows": int(len(rows)),
        "head_count": int(len(head)),
        "step1": step1,
        "head20_mean": head20,
    }
    write_json(out_json, payload)
    return payload


def load_group_summary(path: Path) -> Dict[str, Any]:
    return load_json(path)


def group_metrics(path: Path) -> Dict[str, float]:
    groups = load_group_summary(path).get("groups", {})
    return {
        "all_ex_root": safe_float(groups.get("all_ex_root", {}).get("mean")),
        "leg": safe_float(groups.get("leg", {}).get("mean")),
        "nonleg": safe_float(groups.get("nonleg", {}).get("mean")),
        "arm": safe_float(groups.get("arm", {}).get("mean")),
        "else": safe_float(groups.get("else", {}).get("mean")),
    }


def lane_paths(spec: CaseSpec) -> Dict[str, Path]:
    lane_root = OUT_ROOT / spec.name
    surgery_dir = SURGERY_ROOT / spec.name
    stage6_model_dir = STAGE6_MODEL_ROOT / spec.name
    run_name = spec.run_name
    return {
        "lane_root": lane_root,
        "lane_log": lane_root / "lane.log",
        "surgery_dir": surgery_dir,
        "surgery_ckpt": surgery_dir / f"{spec.name}.pth",
        "surgery_report": lane_root / "transplant_report.json",
        "stage6_model_dir": stage6_model_dir,
        "stage6_log_json": stage6_model_dir / f"posttrain_log_{run_name}.json",
        "stage6_ckpt": stage6_model_dir / f"ckpt_last_{run_name}.pth",
        "stage6_init_json": lane_root / "posttrain_stage6_init_stats.json",
        "stage6_eval_dir": lane_root / "stage6_freerun",
        "stage6_eval_json": lane_root / "stage6_freerun" / "Walk_F_freerun_cycles.json",
        "stage6_group_json": lane_root / "stage6_group_summary.json",
        "status_json": lane_root / "status.json",
    }


def resolve_model_state(ckpt: Any, path: Path) -> Dict[str, Any]:
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"{path}: expected dict checkpoint, got {type(ckpt).__name__}")
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"{path}: expected model state dict, got {type(state).__name__}")
    return state


def tensor_stats(value: Any) -> Dict[str, Any]:
    if not torch.is_tensor(value):
        return {
            "shape": None,
            "dtype": type(value).__name__,
            "mean": None,
            "std": None,
            "norm": None,
        }
    data = value.detach().float().reshape(-1)
    if data.numel() == 0:
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "mean": float("nan"),
            "std": float("nan"),
            "norm": float("nan"),
        }
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "norm": float(data.norm().item()),
    }


def collect_plan_keys(state_dict: Mapping[str, Any], *, label: str) -> Tuple[List[str], Dict[str, List[str]]]:
    prefix_map: Dict[str, List[str]] = {}
    ordered: List[str] = []
    for prefix in PLAN_PREFIXES:
        matches = sorted(k for k in state_dict.keys() if str(k).startswith(prefix))
        if not matches:
            raise RuntimeError(f"{label}: missing plan-stack prefix `{prefix}`")
        prefix_map[prefix] = matches
        ordered.extend(matches)
    for key in PLAN_EXACT_KEYS:
        if key not in state_dict:
            raise RuntimeError(f"{label}: missing exact plan-stack key `{key}`")
        ordered.append(key)
    seen = set()
    deduped: List[str] = []
    for key in ordered:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped, prefix_map


def build_transplant_report(
    *,
    donor_name: str,
    donor_state: Mapping[str, Any],
    recipient_name: str,
    recipient_state: Mapping[str, Any],
) -> Dict[str, Any]:
    donor_keys, donor_prefix_map = collect_plan_keys(donor_state, label=donor_name)
    recipient_keys, recipient_prefix_map = collect_plan_keys(recipient_state, label=recipient_name)
    if donor_keys != recipient_keys:
        raise RuntimeError(
            "plan-stack key mismatch between donor and recipient:\n"
            f"donor-only={sorted(set(donor_keys) - set(recipient_keys))}\n"
            f"recipient-only={sorted(set(recipient_keys) - set(donor_keys))}"
        )

    transplant_keys = donor_keys
    per_key: List[Dict[str, Any]] = []
    changed_count = 0
    for key in transplant_keys:
        donor_value = donor_state[key]
        recipient_value = recipient_state[key]
        donor_shape = tuple(donor_value.shape) if torch.is_tensor(donor_value) else None
        recipient_shape = tuple(recipient_value.shape) if torch.is_tensor(recipient_value) else None
        if donor_shape != recipient_shape:
            raise RuntimeError(
                f"shape mismatch for `{key}`: donor={donor_shape} recipient={recipient_shape}"
            )
        if torch.is_tensor(donor_value) != torch.is_tensor(recipient_value):
            raise RuntimeError(f"type mismatch for `{key}` between donor and recipient")
        same_before = bool(torch.equal(donor_value, recipient_value)) if torch.is_tensor(donor_value) else donor_value == recipient_value
        if not same_before:
            changed_count += 1
        per_key.append(
            {
                "key": key,
                "prefix_group": next((prefix for prefix in PLAN_PREFIXES if key.startswith(prefix)), "exact"),
                "same_before": same_before,
                "donor": tensor_stats(donor_value),
                "recipient_before": tensor_stats(recipient_value),
            }
        )
    return {
        "donor": donor_name,
        "recipient": recipient_name,
        "key_count": int(len(transplant_keys)),
        "changed_key_count": int(changed_count),
        "unchanged_key_count": int(len(transplant_keys) - changed_count),
        "prefix_counts": {prefix: len(keys) for prefix, keys in recipient_prefix_map.items()},
        "exact_keys": list(PLAN_EXACT_KEYS),
        "keys": transplant_keys,
        "per_key": per_key,
        "donor_prefix_map": donor_prefix_map,
        "recipient_prefix_map": recipient_prefix_map,
    }


def ensure_transplant_ckpt(spec: CaseSpec, baseline_rows: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not spec.is_transplant:
        return {
            "skipped": True,
            "reason": "baseline_reused",
        }

    paths = lane_paths(spec)
    if paths["surgery_ckpt"].is_file() and paths["surgery_report"].is_file():
        return load_json(paths["surgery_report"])

    recipient_row = baseline_rows[spec.backbone_case]
    donor_row = baseline_rows[spec.planstack_case]
    recipient_ckpt_path = Path(str(recipient_row["ckpt"]))
    donor_ckpt_path = Path(str(donor_row["ckpt"]))
    if not recipient_ckpt_path.is_file():
        raise RuntimeError(f"missing recipient ckpt: {recipient_ckpt_path}")
    if not donor_ckpt_path.is_file():
        raise RuntimeError(f"missing donor ckpt: {donor_ckpt_path}")

    log(f"build transplant {spec.name}: backbone={spec.backbone_case} donor_planstack={spec.planstack_case}")
    recipient_ckpt = torch.load(recipient_ckpt_path, map_location="cpu")
    donor_ckpt = torch.load(donor_ckpt_path, map_location="cpu")
    recipient_state = resolve_model_state(recipient_ckpt, recipient_ckpt_path)
    donor_state = resolve_model_state(donor_ckpt, donor_ckpt_path)

    report = build_transplant_report(
        donor_name=spec.planstack_case,
        donor_state=donor_state,
        recipient_name=spec.backbone_case,
        recipient_state=recipient_state,
    )

    recipient_out = copy.deepcopy(recipient_ckpt)
    recipient_out_state = resolve_model_state(recipient_out, recipient_ckpt_path)
    for key in report["keys"]:
        value = donor_state[key]
        recipient_out_state[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)

    paths["surgery_dir"].mkdir(parents=True, exist_ok=True)
    torch.save(recipient_out, paths["surgery_ckpt"])

    reloaded = torch.load(paths["surgery_ckpt"], map_location="cpu")
    reloaded_state = resolve_model_state(reloaded, paths["surgery_ckpt"])
    same_after = 0
    for key in report["keys"]:
        donor_value = donor_state[key]
        reload_value = reloaded_state[key]
        if torch.is_tensor(donor_value):
            if tuple(donor_value.shape) != tuple(reload_value.shape):
                raise RuntimeError(f"reload shape mismatch for `{key}`")
            if torch.equal(donor_value, reload_value):
                same_after += 1
        else:
            if donor_value == reload_value:
                same_after += 1

    report.update(
        {
            "backbone_case": spec.backbone_case,
            "planstack_case": spec.planstack_case,
            "recipient_ckpt": str(recipient_ckpt_path),
            "donor_ckpt": str(donor_ckpt_path),
            "surgery_ckpt": str(paths["surgery_ckpt"]),
            "verify_same_after_count": int(same_after),
            "verified_all_after": bool(same_after == len(report["keys"])),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if not bool(report["verified_all_after"]):
        raise RuntimeError(f"post-save verification failed for {spec.name}")
    write_json(paths["surgery_report"], report)
    return report


def ensure_stage6(spec: CaseSpec, baseline_rows: Mapping[str, Dict[str, Any]]) -> None:
    if not spec.is_transplant:
        return
    paths = lane_paths(spec)
    if paths["stage6_ckpt"].is_file() and paths["stage6_log_json"].is_file():
        if not paths["stage6_init_json"].is_file():
            extract_stage6_init(paths["stage6_log_json"], paths["stage6_init_json"])
        return

    ckpt_in = paths["surgery_ckpt"]
    if not ckpt_in.is_file():
        raise RuntimeError(f"missing surgery ckpt for {spec.name}: {ckpt_in}")

    paths["stage6_model_dir"].mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            sys.executable,
            "-m",
            "train.posttrain",
            "--config",
            str(STAGE6_CONFIG),
            "--ckpt_in",
            str(ckpt_in),
            "--out_dir",
            str(paths["stage6_model_dir"]),
            "--run_name",
            spec.run_name,
            "--posttrain_contacts_source",
            "pretrain_contact",
            "--posttrain_contacts_pretrain_clamp",
            PRETRAIN_CLAMP,
            "--encoder_bundle",
            str(ENCODER_BUNDLE),
            "--posttrain_contacts_pretrain_affine_stats",
            str(AFFINE_STATS),
        ],
        log_file=paths["lane_log"],
    )
    extract_stage6_init(paths["stage6_log_json"], paths["stage6_init_json"])


def ensure_stage6_eval(spec: CaseSpec) -> None:
    if not spec.is_transplant:
        return
    paths = lane_paths(spec)
    if paths["stage6_group_json"].is_file():
        return
    run_cmd(
        [
            sys.executable,
            "-m",
            "train.validate.run_freerun_cycles",
            "--teacher",
            str(TEACHER),
            "--model",
            str(paths["stage6_ckpt"]),
            "--rounds",
            "5",
            "--depth",
            "3",
            "--time-index-mode",
            "cycle",
            "--phase_reset_source",
            "none",
            "--contacts_meas_source",
            "pretrain_contact",
            "--contacts_meas_pretrain_clamp",
            PRETRAIN_CLAMP,
            "--contacts_meas_pretrain_affine_stats",
            str(AFFINE_STATS),
            "--encoder-bundle",
            str(ENCODER_BUNDLE),
            "--export_joint_direct_geolocal_series",
            "--out",
            str(paths["stage6_eval_dir"]),
            "--force",
        ],
        log_file=paths["lane_log"],
    )
    run_cmd(
        [
            sys.executable,
            str(ROOT / "tools" / "phasea_group_summary.py"),
            str(paths["stage6_eval_json"]),
            "--cycle_gte",
            "1",
            "--drop_wrap",
            "--out",
            str(paths["stage6_group_json"]),
        ],
        log_file=paths["lane_log"],
    )


def load_baseline_rows() -> Dict[str, Dict[str, Any]]:
    payload = load_json(BASELINE_COMPARE_JSON)
    lanes = payload.get("lanes", [])
    if not isinstance(lanes, list):
        raise RuntimeError(f"invalid lanes in {BASELINE_COMPARE_JSON}")
    rows = {str(row["name"]): row for row in lanes if isinstance(row, dict) and "name" in row}
    required = {spec.name for spec in BASELINE_CASES}
    missing = sorted(required - set(rows.keys()))
    if missing:
        raise RuntimeError(f"baseline compare missing lanes: {missing}")
    return rows


def build_case_entry(
    spec: CaseSpec,
    *,
    baseline_rows: Mapping[str, Dict[str, Any]],
    baseline_entries: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    source_baseline_name = spec.backbone_case
    old_baseline_name = "old_bestfree"
    source_baseline_entry = baseline_entries[source_baseline_name]
    old_baseline_entry = baseline_entries[old_baseline_name]

    if not spec.is_transplant:
        row = baseline_rows[spec.name]
        stage6_init_path = Path(str(row["paths"]["stage6_init_stats"]))
        stage6_group_path = Path(str(row["paths"]["stage6_group_summary"]))
        stage6_init = enrich_stage6_init(load_json(stage6_init_path))
        stage6_exit = group_metrics(stage6_group_path)
        entry = {
            "name": spec.name,
            "case_type": "baseline_reused",
            "backbone_case": spec.backbone_case,
            "planstack_case": spec.planstack_case,
            "source_baseline": spec.name,
            "stage6_exit": stage6_exit,
            "stage6_init": stage6_init,
            "delta_vs_source_baseline": {k: 0.0 for k in stage6_exit.keys()},
            "improvement_vs_source_baseline": {k: 0.0 for k in stage6_exit.keys()},
            "delta_vs_old_bestfree": {
                key: diff(stage6_exit.get(key), old_baseline_entry["stage6_exit"].get(key))
                for key in stage6_exit.keys()
            },
            "init_delta_vs_source_baseline": {
                "step1": {k: 0.0 for k in stage6_init["step1"].keys()},
                "head20_mean": {k: 0.0 for k in stage6_init["head20_mean"].keys()},
            },
            "gap_to_old_bestfree": {
                key: diff(stage6_exit.get(key), old_baseline_entry["stage6_exit"].get(key))
                for key in stage6_exit.keys()
            },
            "paths": {
                "stage6_init_stats": str(stage6_init_path),
                "stage6_group_summary": str(stage6_group_path),
                "stage6_ckpt": str(row["paths"]["stage6_ckpt"]),
                "baseline_compare_row": str(BASELINE_COMPARE_JSON),
            },
        }
        return entry

    paths = lane_paths(spec)
    stage6_init = enrich_stage6_init(load_json(paths["stage6_init_json"]))
    stage6_exit = group_metrics(paths["stage6_group_json"])
    surgery_report = load_json(paths["surgery_report"])
    entry = {
        "name": spec.name,
        "case_type": "planstack_transplant",
        "backbone_case": spec.backbone_case,
        "planstack_case": spec.planstack_case,
        "source_baseline": source_baseline_name,
        "stage6_exit": stage6_exit,
        "stage6_init": stage6_init,
        "delta_vs_source_baseline": {
            key: diff(stage6_exit.get(key), source_baseline_entry["stage6_exit"].get(key))
            for key in stage6_exit.keys()
        },
        "improvement_vs_source_baseline": {
            key: improvement(source_baseline_entry["stage6_exit"].get(key), stage6_exit.get(key))
            for key in stage6_exit.keys()
        },
        "delta_vs_old_bestfree": {
            key: diff(stage6_exit.get(key), old_baseline_entry["stage6_exit"].get(key))
            for key in stage6_exit.keys()
        },
        "init_delta_vs_source_baseline": {
            "step1": delta_block(stage6_init["step1"], source_baseline_entry["stage6_init"]["step1"]),
            "head20_mean": delta_block(stage6_init["head20_mean"], source_baseline_entry["stage6_init"]["head20_mean"]),
        },
        "gap_to_old_bestfree": {
            key: diff(stage6_exit.get(key), old_baseline_entry["stage6_exit"].get(key))
            for key in stage6_exit.keys()
        },
        "paths": {
            "lane_root": str(paths["lane_root"]),
            "stage6_init_stats": str(paths["stage6_init_json"]),
            "stage6_group_summary": str(paths["stage6_group_json"]),
            "stage6_ckpt": str(paths["stage6_ckpt"]),
            "surgery_report": str(paths["surgery_report"]),
            "surgery_ckpt": str(paths["surgery_ckpt"]),
        },
        "transplant_report": {
            "key_count": surgery_report.get("key_count"),
            "changed_key_count": surgery_report.get("changed_key_count"),
            "verified_all_after": surgery_report.get("verified_all_after"),
        },
    }
    return entry


def build_summary(
    baseline_rows: Mapping[str, Dict[str, Any]],
    baseline_entries: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    order = [spec.name for spec in BASELINE_CASES] + [spec.name for spec in TRANSPLANT_CASES]
    spec_map = {spec.name: spec for spec in BASELINE_CASES + TRANSPLANT_CASES}
    cases = [
        build_case_entry(
            spec_map[name],
            baseline_rows=baseline_rows,
            baseline_entries=baseline_entries,
        )
        for name in order
    ]
    case_map = {case["name"]: case for case in cases}

    cp015_base = case_map["cp015_bestfree"]["stage6_exit"]
    old_base = case_map["old_bestfree"]["stage6_exit"]
    cp015_old_plan = case_map["cp015_with_old_planstack"]["stage6_exit"]
    old_cp015_plan = case_map["old_with_cp015_planstack"]["stage6_exit"]

    baseline_gap = diff(cp015_base["all_ex_root"], old_base["all_ex_root"])
    gap_after_transplant = diff(cp015_old_plan["all_ex_root"], old_base["all_ex_root"])
    gap_closed = improvement(cp015_base["all_ex_root"], cp015_old_plan["all_ex_root"])
    gap_closed_ratio = float(gap_closed / baseline_gap) if math.isfinite(gap_closed) and math.isfinite(baseline_gap) and baseline_gap != 0.0 else float("nan")

    answers = {
        "q1_cp015_with_old_planstack_improvement_vs_cp015_bestfree": {
            key: improvement(cp015_base.get(key), cp015_old_plan.get(key))
            for key in ("all_ex_root", "leg", "nonleg")
        },
        "q2_cp015_with_old_planstack_vs_old_bestfree": {
            "all_ex_root_gap_to_old_bestfree": diff(cp015_old_plan["all_ex_root"], old_base["all_ex_root"]),
            "leg_gap_to_old_bestfree": diff(cp015_old_plan["leg"], old_base["leg"]),
            "nonleg_gap_to_old_bestfree": diff(cp015_old_plan["nonleg"], old_base["nonleg"]),
            "beats_old_bestfree_all_ex_root": bool(safe_float(cp015_old_plan["all_ex_root"]) <= safe_float(old_base["all_ex_root"])),
            "baseline_gap_closed_ratio": gap_closed_ratio,
        },
        "q3_old_with_cp015_planstack_vs_old_bestfree": {
            key: diff(old_cp015_plan.get(key), old_base.get(key))
            for key in ("all_ex_root", "leg", "nonleg")
        },
        "q4_hypothesis_signal": {
            "baseline_cp015_minus_old_all_ex_root": baseline_gap,
            "cp015_old_plan_gap_to_old_all_ex_root": gap_after_transplant,
            "old_cp015_plan_minus_old_all_ex_root": diff(old_cp015_plan["all_ex_root"], old_base["all_ex_root"]),
        },
    }

    return {
        "run_date": RUN_DATE,
        "policy": {
            "baseline_compare_summary": str(BASELINE_COMPARE_JSON),
            "stage6_config": str(STAGE6_CONFIG),
            "teacher": str(TEACHER),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "validate_rounds": 5,
            "validate_depth": 3,
            "validate_time_index_mode": "cycle",
            "validate_phase_reset_source": "none",
            "validate_contacts_meas_source": "pretrain_contact",
            "validate_contacts_meas_pretrain_clamp": PRETRAIN_CLAMP,
            "group_summary_args": ["--cycle_gte", "1", "--drop_wrap"],
            "plan_prefixes": list(PLAN_PREFIXES),
            "plan_exact_keys": list(PLAN_EXACT_KEYS),
        },
        "cases": cases,
        "answers": answers,
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    cases = summary["cases"]
    case_map = {case["name"]: case for case in cases}
    lines: List[str] = []
    lines.append("# Stage6 frozen plan-stack transplant")
    lines.append("")
    lines.append(f"- run_date: {summary['run_date']}")
    lines.append(f"- baseline compare reused: `{summary['policy']['baseline_compare_summary']}`")
    lines.append(f"- stage6 config: `{summary['policy']['stage6_config']}`")
    lines.append(f"- teacher: `{summary['policy']['teacher']}`")
    lines.append(f"- encoder bundle: `{summary['policy']['encoder_bundle']}`")
    lines.append(f"- affine stats: `{summary['policy']['affine_stats']}`")
    lines.append("")
    lines.append("## Stage6 exit")
    lines.append("")
    lines.append("| case | type | backbone | plan-stack | all_ex_root | leg | nonleg | delta_vs_source all_ex_root |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for case in cases:
        s = case["stage6_exit"]
        d = case["delta_vs_source_baseline"]
        lines.append(
            f"| {case['name']} | {case['case_type']} | {case['backbone_case']} | {case['planstack_case']} | "
            f"{fmt(s['all_ex_root'])} | {fmt(s['leg'])} | {fmt(s['nonleg'])} | {fmt(d['all_ex_root'])} |"
        )
    lines.append("")
    lines.append("## Stage6 init")
    lines.append("")
    lines.append("| case | step1 dir_leg_base | step1 dir_nonleg_base | step1 leg/nonleg | head20 dir_leg_base | head20 dir_nonleg_base | head20 leg/nonleg | head20 grad arm/else |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for case in cases:
        step1 = case["stage6_init"]["step1"]
        head20 = case["stage6_init"]["head20_mean"]
        lines.append(
            f"| {case['name']} | {fmt(step1['dir_leg_base'])} | {fmt(step1['dir_nonleg_base'])} | {fmt(step1['leg_over_nonleg'])} | "
            f"{fmt(head20['dir_leg_base'])} | {fmt(head20['dir_nonleg_base'])} | {fmt(head20['leg_over_nonleg'])} | {fmt(head20['grad_arm_over_else'])} |"
        )
    lines.append("")
    lines.append("## Init delta vs source baseline")
    lines.append("")
    lines.append("| case | step1 leg/nonleg delta | head20 leg/nonleg delta | head20 grad arm/else delta |")
    lines.append("|---|---:|---:|---:|")
    for case in cases:
        d = case["init_delta_vs_source_baseline"]
        lines.append(
            f"| {case['name']} | {fmt(d['step1'].get('leg_over_nonleg'))} | "
            f"{fmt(d['head20_mean'].get('leg_over_nonleg'))} | {fmt(d['head20_mean'].get('grad_arm_over_else'))} |"
        )
    lines.append("")
    lines.append("## Transplant verification")
    lines.append("")
    lines.append("| case | transplanted keys | changed keys | verified_after_save |")
    lines.append("|---|---:|---:|---:|")
    for name in ("cp015_with_old_planstack", "old_with_cp015_planstack"):
        case = case_map[name]
        report = load_json(Path(case["paths"]["surgery_report"]))
        lines.append(
            f"| {name} | {report.get('key_count')} | {report.get('changed_key_count')} | {str(bool(report.get('verified_all_after'))).lower()} |"
        )
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    q1 = summary["answers"]["q1_cp015_with_old_planstack_improvement_vs_cp015_bestfree"]
    q2 = summary["answers"]["q2_cp015_with_old_planstack_vs_old_bestfree"]
    q3 = summary["answers"]["q3_old_with_cp015_planstack_vs_old_bestfree"]
    lines.append(
        f"1. `cp015_with_old_planstack` vs `cp015_bestfree`: "
        f"all_ex_root improve {fmt(q1['all_ex_root'])}, leg improve {fmt(q1['leg'])}, nonleg improve {fmt(q1['nonleg'])}."
    )
    lines.append(
        f"2. `cp015_with_old_planstack` all_ex_root = {fmt(case_map['cp015_with_old_planstack']['stage6_exit']['all_ex_root'])}; "
        f"`old_bestfree` = {fmt(case_map['old_bestfree']['stage6_exit']['all_ex_root'])}; "
        f"gap = {fmt(q2['all_ex_root_gap_to_old_bestfree'])}; "
        f"beats_old = `{str(bool(q2['beats_old_bestfree_all_ex_root'])).lower()}`; "
        f"closed_ratio = {fmt(q2['baseline_gap_closed_ratio'])}."
    )
    lines.append(
        f"3. `old_with_cp015_planstack` vs `old_bestfree`: "
        f"all_ex_root delta {fmt(q3['all_ex_root'])}, leg delta {fmt(q3['leg'])}, nonleg delta {fmt(q3['nonleg'])}."
    )
    lines.append(
        "4. Hypothesis signal should be judged from the two swap directions together; "
        "see exact deltas in `summary.json`."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [
        BASELINE_COMPARE_JSON,
        STAGE6_CONFIG,
        TEACHER,
        ENCODER_BUNDLE,
        AFFINE_STATS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    baseline_rows = load_baseline_rows()
    baseline_entries: Dict[str, Dict[str, Any]] = {}
    for spec in BASELINE_CASES:
        row = baseline_rows[spec.name]
        stage6_init_path = Path(str(row["paths"]["stage6_init_stats"]))
        stage6_group_path = Path(str(row["paths"]["stage6_group_summary"]))
        if not stage6_init_path.is_file():
            raise RuntimeError(f"missing baseline stage6 init stats: {stage6_init_path}")
        if not stage6_group_path.is_file():
            raise RuntimeError(f"missing baseline stage6 group summary: {stage6_group_path}")
        baseline_entries[spec.name] = {
            "stage6_exit": group_metrics(stage6_group_path),
            "stage6_init": enrich_stage6_init(load_json(stage6_init_path)),
        }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT_ROOT / "manifest.json",
        {
            "run_date": RUN_DATE,
            "root": str(ROOT),
            "baseline_compare_summary": str(BASELINE_COMPARE_JSON),
            "stage6_config": str(STAGE6_CONFIG),
            "teacher": str(TEACHER),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "baseline_cases": [spec.__dict__ for spec in BASELINE_CASES],
            "transplant_cases": [spec.__dict__ for spec in TRANSPLANT_CASES],
            "plan_prefixes": list(PLAN_PREFIXES),
            "plan_exact_keys": list(PLAN_EXACT_KEYS),
        },
    )

    for idx, spec in enumerate(TRANSPLANT_CASES, start=1):
        log(f"=== [{idx}/{len(TRANSPLANT_CASES)}] {spec.name} ===")
        report = ensure_transplant_ckpt(spec, baseline_rows)
        ensure_stage6(spec, baseline_rows)
        ensure_stage6_eval(spec)
        write_json(
            lane_paths(spec)["status_json"],
            {
                "case": spec.__dict__,
                "transplant_report": report,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    summary = build_summary(baseline_rows, baseline_entries)
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
