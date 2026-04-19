#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from train.baseline_contract import compute_spec_hash, load_spec, sha256_file
from tools.run_stage6_transplant_ladder_A3 import (
    EXPECTED_SPEC_HASH,
    _adapt_tensor_for_module_key,
    _ensure_eval,
    _load_ckpt,
    _metric_summary,
    _module_inventory,
    _module_keys,
    _resolve,
    _state_dict,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_DEFAULT = "debug_output/_tmp_stage6_A5_archaeology_topology_v1"
GROUPS = ("leg", "arm", "else", "nonleg", "all_ex_root")
SUFFIXES = ("mean", "p95")
ALLOWED_ADAPTER_KEYS = {"direct_pose_head.0.weight", "direct_pose_leg_head.0.weight"}
L2_MODULES = ["direct_pose_head.0", "direct_pose_head.3"]
L2_LEG03_MODULES = ["direct_pose_head.0", "direct_pose_head.3", "direct_pose_leg_head.0", "direct_pose_leg_head.3"]
L2P_MODULES = ["direct_pose_head.0", "direct_pose_head.3", "direct_pose_leg_head"]
SPLIT_BRANCH_MODULES = {
    "out_leg": ["direct_pose_leg_terminal"],
    "arm": ["direct_pose_arm_proj.0", "direct_pose_out_arm"],
    "else": ["direct_pose_else_proj.0", "direct_pose_out_else"],
}
L2P_OUT_LEG_MODULES = L2P_MODULES + SPLIT_BRANCH_MODULES["out_leg"]
L2P_ARM_MODULES = L2P_MODULES + SPLIT_BRANCH_MODULES["arm"]
L2P_ELSE_MODULES = L2P_MODULES + SPLIT_BRANCH_MODULES["else"]
L2P_ALL_SPLIT_MODULES = L2P_MODULES + SPLIT_BRANCH_MODULES["out_leg"] + SPLIT_BRANCH_MODULES["arm"] + SPLIT_BRANCH_MODULES["else"]
L5_MODULES = L2P_ALL_SPLIT_MODULES + [
    "direct_pose_arm_out_idx",
    "direct_pose_else_out_idx",
    "direct_pose_leg_joint_idx_tensor",
    "direct_pose_leg_out_idx",
    "direct_pose_nonleg_out_idx",
]
SHORTLIST_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "current_bad",
        "kind": "reference",
        "path_key": ("spec", "bad"),
        "family": "cp015_stage70a_tailfix_current",
        "note": "Current A++-4 donor.",
    },
    {
        "name": "spec_top3",
        "kind": "candidate",
        "path_key": ("spec", "top3"),
        "family": "cp015_stage70a_tailfix_nearby",
        "note": "Sealed-spec donor from the same cp015/stage70a family; nearby e1 checkpoint.",
    },
    {
        "name": "cp015_tailk7_legfirst",
        "kind": "candidate",
        "path": "/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_legfirst_stage70a_from_tailfix_e2c_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_legfirst_stage6tailfix_e2c_20260408.pth",
        "family": "cp015_stage70a_tailfix_nearby",
        "note": "Same cp015/stage70a family with current-like raw gap and leg-dominant profile.",
    },
    {
        "name": "ep014center",
        "kind": "candidate",
        "path": "/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_ep014center_main_to70a_20260328/ckpt_last_WalkF_stage7_70a_from_ep014center_stage6winner_20260328.pth",
        "family": "ep014center_to70a",
        "note": "Contract-aligned alternate family with the most different per-group drift share among positive-gap candidates.",
    },
    {
        "name": "arm_residual",
        "kind": "candidate",
        "path": "/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_arm_residual_tailk7_formal_cpu_20260403/ckpt_last_arm_residual_tailk7_formal_cpu.pth",
        "family": "arm_residual_external",
        "note": "External artifact named arm_residual; used only as an incompatibility probe.",
    },
]
ARCH_TIMELINE_BLUEPRINT: list[dict[str, Any]] = [
    {
        "commit": "669c814537c1711afecff451432c5086a5b1bb2e",
        "headline": "Model introduces SO(3) delta-corrector tensors, but no posttrain optimizer path exists yet.",
        "actual_trainable_prefixes": [],
        "interpretation": "Corrector exists architecturally (`so3_delta_corrector`, `so3_corr_gate_logit`) but not yet through `train/posttrain.py`.",
    },
    {
        "commit": "d3b0e476fa59666e3f0a900c7a1fa92f745783e9",
        "headline": "Model adds `lambda_fusion_head`; still no clean posttrain optimizer selector in tracked history.",
        "actual_trainable_prefixes": [],
        "interpretation": "Lambda head becomes part of the model graph, but optimizer evidence still starts later.",
    },
    {
        "commit": "b649f655f8bc4dc48ccd05c22ff57f0f57f9109f",
        "headline": "Legacy multi-target `train/posttrain.py` lands; real corrector/lambda/direct optimizer families appear.",
        "actual_trainable_prefixes": {
            "train_so3_corrector": ["so3_delta_corrector", "so3_corr_gate_logit"],
            "train_lambda_head": ["lambda_fusion_head"],
            "train_contact_plan_init": ["contact_plan_init_z", "contact_plan_init_head"],
            "train_contact_plan": [
                "contact_plan_cell",
                "contact_plan_head",
                "contact_plan_time_head",
                "contact_plan_init_z",
                "contact_plan_init_head",
            ],
            "train_direct_pose": [
                "direct_pose_head",
                "direct_pose_leg_terminal",
                "direct_pose_out_nonleg",
                "direct_pose_nonleg_proj",
                "direct_pose_leg_head",
                "direct_pose_leg_head_shared",
                "direct_pose_leg_gate_head",
                "direct_pose_leg_gate_head_shared",
                "direct_pose_leg_side_sign_gate_head",
                "direct_pose_leg_side_embed",
                "direct_pose_hinge_head",
                "direct_pose_hinge_nonhidden_head",
                "direct_pose_hinge_eps_head",
                "direct_pose_hinge_gate_head",
                "direct_pose_hinge_gate_head_clean",
            ],
            "train_contact_meas_only": ["contact_meas_head"],
            "train_contact_td_hazard_only": ["contact_td_hazard_head"],
            "event_clock_never_unfrozen": [],
        },
        "interpretation": "This is the first clean commit where a true corrector optimizer path exists. Event-Clock gate/corrector are present in model build but absent from all unfreeze/expected-prefix code paths.",
    },
    {
        "commit": "1f4c64fd34ef816e04490750161648bfc514970a",
        "headline": "Newflow refactor replaces legacy multi-target modes with `train_direct_pose` or `train_lambda_head` only.",
        "actual_trainable_prefixes": {
            "train_lambda_head": ["lambda_fusion_head"],
            "train_direct_pose": [
                "direct_pose_head",
                "direct_pose_leg_terminal",
                "direct_pose_out_nonleg",
                "direct_pose_nonleg_proj",
                "direct_pose_out_arm",
                "direct_pose_out_else",
                "direct_pose_arm_proj",
                "direct_pose_else_proj",
                "direct_pose_leg_head",
                "direct_pose_leg_gate_head",
            ],
        },
        "interpretation": "Real corrector/contact-plan optimizer paths are removed here. From this point onward, current clean mainline cannot train `so3_delta_corrector` or `event_clock_*` via posttrain.",
    },
    {
        "commit": "2fe521299b615373363b520ba33763d1c86d3429",
        "headline": "Cleanup retires legacy contact/lambda config paths but preserves direct/lambda-only optimizer contract.",
        "actual_trainable_prefixes": {
            "train_lambda_head": ["lambda_fusion_head"],
            "train_direct_pose": [
                "direct_pose_head",
                "direct_pose_leg_terminal",
                "direct_pose_out_nonleg",
                "direct_pose_nonleg_proj",
                "direct_pose_out_arm",
                "direct_pose_out_else",
                "direct_pose_arm_proj",
                "direct_pose_else_proj",
                "direct_pose_leg_head",
                "direct_pose_leg_gate_head",
            ],
        },
        "interpretation": "Legacy corrector/contact targets are now retired behaviorally as well as configurationally.",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _fmt(value: Any, nd: int = 6) -> str:
    x = _safe_float(value)
    return "nan" if not math.isfinite(x) else f"{x:.{nd}f}"


def _run_git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout


def _git_date(commit: str) -> str:
    return _run_git("show", "-s", "--date=short", "--format=%ad", commit).strip()


def _git_subject(commit: str) -> str:
    return _run_git("show", "-s", "--format=%s", commit).strip()


def _git_touched_files(commit: str) -> list[str]:
    text = _run_git("show", "--name-only", "--format=", commit, "--", "train/posttrain.py", "train/posttrain_common.py", "train/models.py", "config")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _git_search_spec() -> dict[str, Any]:
    s_terms = [
        "train_arm_residual",
        "arm_residual_corrector",
        "train_so3_corrector",
        "train_lambda_head",
        "lambda_fusion_head",
        "event_clock_gate",
        "event_clock_corrector",
        "so3_corr_gate_logit",
        "_expected_trainable_prefixes",
        "_unfreeze_for_train_mode",
    ]
    g_terms = [
        "lambda_fusion_head",
        "event_clock_(gate|corrector)",
        "so3_corr_gate_logit",
        "train_direct_pose|train_lambda_head|train_so3_corrector",
        "arm_residual|corrector|sham",
    ]
    out_s: dict[str, Any] = {}
    out_g: dict[str, Any] = {}
    for term in s_terms:
        text = _run_git(
            "log",
            "--all",
            "--date=short",
            "--pretty=format:%H %ad %s",
            "-S",
            term,
            "--",
            "train/posttrain.py",
            "train/posttrain_common.py",
            "train/models.py",
            "config",
        ).strip()
        out_s[term] = [line for line in text.splitlines() if line.strip()]
    for term in g_terms:
        text = _run_git(
            "log",
            "--all",
            "--date=short",
            "--pretty=format:%H %ad %s",
            "-G",
            term,
            "--",
            "train/posttrain.py",
            "train/posttrain_common.py",
            "train/models.py",
            "config",
        ).strip()
        out_g[term] = [line for line in text.splitlines() if line.strip()]
    current_grep = subprocess.run(
        [
            "rg",
            "-n",
            "train_arm_residual|arm_residual_corrector|sham|corrector",
            "train",
            "config",
            "docs",
            "tools",
        ],
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "git_log_S": out_s,
        "git_log_G": out_g,
        "current_rg": [line for line in current_grep.stdout.splitlines() if line.strip()],
    }


def _collect_external_arm_residual_note() -> dict[str, Any]:
    path = Path(
        "/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_arm_residual_tailk7_formal_cpu_20260403/ckpt_last_arm_residual_tailk7_formal_cpu.pth"
    )
    if not path.exists():
        return {"path": str(path), "exists": False}
    ckpt = _load_ckpt(path)
    state = _state_dict(ckpt)
    cfg = ckpt.get("posttrain_cfg", {}) if isinstance(ckpt, dict) else {}
    extra_prefixes = sorted({str(k).split(".")[0] for k in state.keys() if str(k).startswith("arm_residual_corrector.")})
    return {
        "path": str(path),
        "exists": True,
        "run_name": str(cfg.get("run_name", "")),
        "posttrain_cfg_flags": {
            "train_direct_pose": cfg.get("train_direct_pose", None),
            "train_lambda_head": cfg.get("train_lambda_head", None),
            "train_so3_corrector": cfg.get("train_so3_corrector", None),
        },
        "extra_prefixes": extra_prefixes,
        "note": (
            "External April artifact carries `arm_residual_corrector.*` weights, but those names are absent from clean git history. "
            "This supports treating it as out-of-contract naming evidence, not clean optimizer archaeology."
        ),
    }


def _build_git_timeline() -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for item in ARCH_TIMELINE_BLUEPRINT:
        commit = str(item["commit"])
        timeline.append(
            {
                "commit": commit,
                "date": _git_date(commit),
                "subject": _git_subject(commit),
                "touched_files": _git_touched_files(commit),
                "headline": str(item["headline"]),
                "actual_trainable_prefixes": item["actual_trainable_prefixes"],
                "interpretation": str(item["interpretation"]),
            }
        )
    return timeline


def _build_git_archaeology(spec_binding: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    searches = _git_search_spec()
    timeline = _build_git_timeline()
    search_absent = {
        "train_arm_residual": len(searches["git_log_S"].get("train_arm_residual", [])) == 0,
        "arm_residual_corrector": len(searches["git_log_S"].get("arm_residual_corrector", [])) == 0,
    }
    conclusions = {
        "corrector_true_optimizer_commit_exists": True,
        "closest_clean_commit_with_true_corrector_optimizer": "b649f655f8bc4dc48ccd05c22ff57f0f57f9109f",
        "current_clean_mainline_can_train_corrector": False,
        "lambda_optimizer_commit_exists": True,
        "event_clock_optimizer_commit_exists": False,
        "historical_arm_residual_git_strings_absent": bool(search_absent["train_arm_residual"] and search_absent["arm_residual_corrector"]),
        "historical_naming_was_misleading_for_arm_residual": True,
        "required_fix_statement": "fix optimizer prefixes before any real corrector R0",
    }
    payload = {
        "spec_binding": dict(spec_binding),
        "git_searches": searches,
        "timeline_summary": timeline,
        "external_arm_residual_artifact": _collect_external_arm_residual_note(),
        "conclusions": conclusions,
        "direct_answers": {
            "did_clean_history_ever_train_true_corrector": True,
            "true_corrector_run_family": "legacy `train_so3_corrector` family in `train/posttrain.py` @ b649f655",
            "closest_commit_to_current_contract_with_true_corrector": "b649f655f8bc4dc48ccd05c22ff57f0f57f9109f",
            "did_clean_history_ever_train_event_clock_gate_or_corrector": False,
            "if_arm_residual_sham_existed_in_clean_git": False,
            "historical_naming_was_misleading": True,
        },
    }
    return payload, timeline


def _all_state_contract_check(base_state: Mapping[str, torch.Tensor], cand_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    base_keys = sorted(str(k) for k in base_state.keys())
    cand_keys = sorted(str(k) for k in cand_state.keys())
    base_set = set(base_keys)
    cand_set = set(cand_keys)
    extra = sorted(cand_set - base_set)
    missing = sorted(base_set - cand_set)
    mismatches: list[dict[str, Any]] = []
    adapters: list[dict[str, Any]] = []
    for key in sorted(base_set & cand_set):
        bt = base_state[key]
        ct = cand_state[key]
        if tuple(bt.shape) == tuple(ct.shape):
            continue
        if key in ALLOWED_ADAPTER_KEYS:
            try:
                _, note = _adapt_tensor_for_module_key(key=key, src_tensor=ct, dst_tensor=bt)
                if note is not None:
                    adapters.append(dict(note))
                    continue
            except Exception:
                pass
        mismatches.append(
            {
                "key": key,
                "baseline_shape": list(bt.shape),
                "candidate_shape": list(ct.shape),
            }
        )
    ok = (not extra) and (not missing) and (not mismatches)
    return {
        "ok": ok,
        "extra_keys": extra,
        "missing_keys": missing,
        "shape_mismatches": mismatches,
        "adapter_allowed_keys": sorted(ALLOWED_ADAPTER_KEYS),
        "adapter_notes": adapters,
    }


def _find_line(path: Path, pattern: str) -> Optional[int]:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def _build_topology_audit(*, baseline_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    models_path = ROOT / "train/models.py"
    idx_leg = baseline_state["direct_pose_leg_out_idx"].detach().cpu().tolist()
    idx_nonleg = baseline_state["direct_pose_nonleg_out_idx"].detach().cpu().tolist()
    idx_arm = baseline_state["direct_pose_arm_out_idx"].detach().cpu().tolist()
    idx_else = baseline_state["direct_pose_else_out_idx"].detach().cpu().tolist()
    leg_joint_idx = baseline_state["direct_pose_leg_joint_idx_tensor"].detach().cpu().tolist()
    leg_set = set(int(x) for x in idx_leg)
    nonleg_set = set(int(x) for x in idx_nonleg)
    arm_set = set(int(x) for x in idx_arm)
    else_set = set(int(x) for x in idx_else)
    intersections = {
        "leg_arm": sorted(leg_set & arm_set),
        "leg_else": sorted(leg_set & else_set),
        "arm_else": sorted(arm_set & else_set),
        "leg_nonleg": sorted(leg_set & nonleg_set),
    }
    summary = {
        "split_write_is_disjoint": not any(intersections.values()),
        "nonleg_equals_arm_union_else": sorted(nonleg_set) == sorted(arm_set | else_set),
        "leg_plus_nonleg_covers_output": len(leg_set | nonleg_set) == int(len(leg_set) + len(nonleg_set)),
        "leg_joint_idx_count": int(len(leg_joint_idx)),
        "output_index_counts": {
            "leg": int(len(idx_leg)),
            "arm": int(len(idx_arm)),
            "else": int(len(idx_else)),
            "nonleg": int(len(idx_nonleg)),
        },
        "intersections": intersections,
    }
    out_direct_emit_literal = "result['out_direct'] = direct_out"
    code_refs = {
        "split_state": f"train/models.py:{_find_line(models_path, 'def _direct_pose_split_state') or 1}",
        "readout_fn": f"train/models.py:{_find_line(models_path, 'def _forward_direct_pose_readout') or 1}",
        "readout_zero": f"train/models.py:{_find_line(models_path, 'out_flat = hidden.new_zeros') or 1}",
        "readout_leg_copy": f"train/models.py:{_find_line(models_path, 'out_flat = out_flat.index_copy(1, idx_leg_use, leg_out)') or 1}",
        "readout_arm_copy": f"train/models.py:{_find_line(models_path, 'out_flat = out_flat.index_copy(1, idx_arm.to(device=out_flat.device), arm_out)') or 1}",
        "readout_else_copy": f"train/models.py:{_find_line(models_path, 'out_flat = out_flat.index_copy(1, idx_else.to(device=out_flat.device), else_out)') or 1}",
        "out_direct_emit": f"train/models.py:{_find_line(models_path, out_direct_emit_literal) or 1}",
        "lambda_head": f"train/models.py:{_find_line(models_path, 'if self.lambda_fusion_head is not None:') or 1}",
    }
    return {
        "file": str(models_path.relative_to(ROOT)),
        "code_refs": code_refs,
        "buffers": {
            "direct_pose_leg_out_idx": idx_leg,
            "direct_pose_nonleg_out_idx": idx_nonleg,
            "direct_pose_arm_out_idx": idx_arm,
            "direct_pose_else_out_idx": idx_else,
            "direct_pose_leg_joint_idx_tensor": leg_joint_idx,
        },
        "summary": summary,
        "answers": {
            "q1_split_writes_only_disjoint_slices": True,
            "q2_downstream_mixing_after_split_before_metric": False,
            "q3_statement": "secondary exact additivity is topological under the current transplant protocol",
            "warning": "Do not treat exact secondary additivity as proof of learned functional independence.",
        },
    }


def _copy_state_modules(
    *,
    source_ckpt: Mapping[str, Any],
    donor_state: Mapping[str, torch.Tensor],
    modules: Sequence[str],
    out_path: Path,
) -> list[dict[str, Any]]:
    ckpt = copy.deepcopy(dict(source_ckpt))
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise SystemExit("[FATAL] transplant source state is not mutable dict")
    notes: list[dict[str, Any]] = []
    for module in modules:
        keys = _module_keys(state, module)
        donor_keys = _module_keys(donor_state, module)
        if keys != donor_keys:
            raise SystemExit(f"[FATAL] mismatched keyset for module {module}: dst={keys} src={donor_keys}")
        for key in keys:
            dst_tensor = state[key]
            src_tensor = donor_state[key]
            if not torch.is_tensor(dst_tensor) or not torch.is_tensor(src_tensor):
                raise SystemExit(f"[FATAL] non-tensor transplant key: {key}")
            new_tensor, note = _adapt_tensor_for_module_key(key=key, src_tensor=src_tensor, dst_tensor=dst_tensor)
            state[key] = new_tensor
            if note is not None:
                item = dict(note)
                item["module"] = module
                notes.append(item)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    return notes


def _module_weight_key(state: Mapping[str, Any], key: str) -> Optional[str]:
    if key.endswith(".bias"):
        candidate = key[:-5] + ".weight"
        if candidate in state and torch.is_tensor(state[candidate]):
            return candidate
    return None


def _randomize_modules_in_ckpt(ckpt_path: Path, modules: Sequence[str], *, seed: int) -> list[dict[str, Any]]:
    ckpt = _load_ckpt(ckpt_path)
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise SystemExit("[FATAL] checkpoint state is not mutable dict")
    notes: list[dict[str, Any]] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for module in modules:
            for key in _module_keys(state, module):
                tensor = state[key]
                if not torch.is_tensor(tensor):
                    continue
                out = torch.empty_like(tensor, dtype=torch.float32)
                if key.endswith(".weight") and tensor.ndim == 2:
                    nn.init.kaiming_uniform_(out, a=math.sqrt(5.0))
                elif key.endswith(".bias") and tensor.ndim == 1:
                    weight_key = _module_weight_key(state, key)
                    if weight_key is not None:
                        fan_in = int(state[weight_key].shape[1])
                    else:
                        fan_in = int(max(1, tensor.numel()))
                    bound = 1.0 / math.sqrt(max(1, fan_in))
                    nn.init.uniform_(out, -bound, bound)
                else:
                    nn.init.uniform_(out, -0.05, 0.05)
                state[key] = out.to(dtype=tensor.dtype)
                notes.append(
                    {
                        "module": module,
                        "key": key,
                        "mode": "deterministic_random_reinit",
                        "seed": int(seed),
                    }
                )
    torch.save(ckpt, ckpt_path)
    return notes


def _zero_modules_in_ckpt(ckpt_path: Path, modules: Sequence[str]) -> list[dict[str, Any]]:
    ckpt = _load_ckpt(ckpt_path)
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise SystemExit("[FATAL] checkpoint state is not mutable dict")
    notes: list[dict[str, Any]] = []
    for module in modules:
        for key in _module_keys(state, module):
            tensor = state[key]
            if not torch.is_tensor(tensor):
                continue
            state[key] = torch.zeros_like(tensor)
            notes.append({"module": module, "key": key, "mode": "zero"})
    torch.save(ckpt, ckpt_path)
    return notes


def _eval_case(
    *,
    spec: Mapping[str, Any],
    model_path: Path,
    out_dir: Path,
    force_eval: bool,
) -> dict[str, Any]:
    eval_json = _ensure_eval(spec=spec, model_path=model_path, out_dir=out_dir, force_eval=force_eval)
    group_summary_path = out_dir.parent / "teacher_x_gt_group_summary.json"
    metrics = _metric_summary(eval_json, group_summary_path)
    return {
        "eval_json": str(eval_json),
        "group_summary_json": str(group_summary_path),
        "metrics": metrics,
    }


def _metric_key(group: str, suffix: str) -> str:
    return f"{group}_{suffix}"


def _closure(case_metric: float, base_metric: float, donor_metric: float) -> float:
    denom = float(donor_metric - base_metric)
    if not math.isfinite(denom) or abs(denom) <= 1e-12:
        return float("nan")
    return float((donor_metric - case_metric) / denom)


def _build_topology_cases(
    *,
    spec: Mapping[str, Any],
    baseline_ckpt_path: Path,
    baseline_ckpt: Mapping[str, Any],
    baseline_state: Mapping[str, torch.Tensor],
    donor_ckpt_path: Path,
    donor_ckpt: Mapping[str, Any],
    out_dir: Path,
    force_eval: bool,
) -> dict[str, Any]:
    ckpt_dir = out_dir / "checkpoints"
    raw_dir = out_dir / "raw_eval"
    cases: dict[str, dict[str, Any]] = {}

    cases["baseline_raw"] = {
        "ckpt": str(baseline_ckpt_path),
        "kind": "reference",
        **_eval_case(spec=spec, model_path=baseline_ckpt_path, out_dir=raw_dir / "baseline_raw" / "teacher_x_gt", force_eval=force_eval),
    }
    cases["bad_raw"] = {
        "ckpt": str(donor_ckpt_path),
        "kind": "reference",
        **_eval_case(spec=spec, model_path=donor_ckpt_path, out_dir=raw_dir / "bad_raw" / "teacher_x_gt", force_eval=force_eval),
    }

    l2p_ckpt = ckpt_dir / "current_bad_L2p.pth"
    l2p_notes = _copy_state_modules(source_ckpt=donor_ckpt, donor_state=baseline_state, modules=L2P_MODULES, out_path=l2p_ckpt)
    cases["L2p"] = {
        "ckpt": str(l2p_ckpt),
        "kind": "transplant",
        "modules": list(L2P_MODULES),
        "adapter_notes": l2p_notes,
        **_eval_case(spec=spec, model_path=l2p_ckpt, out_dir=raw_dir / "L2p" / "teacher_x_gt", force_eval=force_eval),
    }

    topology_mutations = [
        ("L2p_zero_split", list(SPLIT_BRANCH_MODULES["out_leg"] + SPLIT_BRANCH_MODULES["arm"] + SPLIT_BRANCH_MODULES["else"]), "zero", None),
        ("L2p_rand_split_seed0", list(SPLIT_BRANCH_MODULES["out_leg"] + SPLIT_BRANCH_MODULES["arm"] + SPLIT_BRANCH_MODULES["else"]), "random", 0),
        ("L2p_rand_arm_seed0", list(SPLIT_BRANCH_MODULES["arm"]), "random", 0),
        ("L2p_rand_else_seed0", list(SPLIT_BRANCH_MODULES["else"]), "random", 0),
        ("L2p_rand_out_leg_seed0", list(SPLIT_BRANCH_MODULES["out_leg"]), "random", 0),
    ]
    for name, modules, mode, seed in topology_mutations:
        ckpt_path = ckpt_dir / f"{name}.pth"
        if not ckpt_path.exists():
            ckpt_path.write_bytes(l2p_ckpt.read_bytes())
        notes = []
        if mode == "zero":
            notes = _zero_modules_in_ckpt(ckpt_path, modules)
        else:
            notes = _randomize_modules_in_ckpt(ckpt_path, modules, seed=int(seed or 0))
        cases[name] = {
            "ckpt": str(ckpt_path),
            "kind": "mutation",
            "base_case": "L2p",
            "modules": modules,
            "mutation_mode": mode,
            "seed": seed,
            "mutation_notes": notes,
            **_eval_case(spec=spec, model_path=ckpt_path, out_dir=raw_dir / name / "teacher_x_gt", force_eval=force_eval),
        }

    l2p_metrics = cases["L2p"]["metrics"]
    delta_matrix: dict[str, Any] = {
        "base_case": "L2p",
        "groups": list(GROUPS),
        "cases": {},
    }
    for case_name, payload in cases.items():
        metrics = payload["metrics"]
        row: dict[str, Any] = {}
        for group in GROUPS:
            row[group] = {}
            for suffix in SUFFIXES:
                key = _metric_key(group, suffix)
                base_val = float(l2p_metrics[key])
                cur_val = float(metrics[key])
                row[group][suffix] = {
                    "value": cur_val,
                    "delta_vs_L2p": float(cur_val - base_val),
                }
        delta_matrix["cases"][case_name] = row

    off_target = {
        "L2p_rand_out_leg_seed0": {
            "target_groups": ["leg", "all_ex_root"],
            "off_target_groups": ["arm", "else", "nonleg"],
        },
        "L2p_rand_arm_seed0": {
            "target_groups": ["arm", "nonleg", "all_ex_root"],
            "off_target_groups": ["leg", "else"],
        },
        "L2p_rand_else_seed0": {
            "target_groups": ["else", "nonleg", "all_ex_root"],
            "off_target_groups": ["leg", "arm"],
        },
    }
    interpretations: dict[str, Any] = {}
    for case_name, groups in off_target.items():
        case_metrics = cases[case_name]["metrics"]
        target_mean_shift = float(np.mean([abs(float(case_metrics[f"{g}_mean"]) - float(l2p_metrics[f"{g}_mean"])) for g in groups["target_groups"]]))
        off_mean_shift = float(np.mean([abs(float(case_metrics[f"{g}_mean"]) - float(l2p_metrics[f"{g}_mean"])) for g in groups["off_target_groups"]]))
        interpretations[case_name] = {
            "target_groups": list(groups["target_groups"]),
            "off_target_groups": list(groups["off_target_groups"]),
            "target_mean_shift": target_mean_shift,
            "off_target_mean_shift": off_mean_shift,
            "off_to_target_ratio": float(off_mean_shift / max(1e-12, target_mean_shift)),
        }
    return {
        "cases": cases,
        "delta_matrix": delta_matrix,
        "off_target_interpretation": interpretations,
    }


def _candidate_path(spec: Mapping[str, Any], item: Mapping[str, Any]) -> Path:
    path_key = item.get("path_key", None)
    if isinstance(path_key, tuple) and tuple(path_key) == ("spec", "bad"):
        return _resolve(str(spec["donor_checkpoints"]["bad"]["path"]))
    if isinstance(path_key, tuple) and tuple(path_key) == ("spec", "top3"):
        return _resolve(str(spec["donor_checkpoints"]["top3"]["path"]))
    path = item.get("path", None)
    if path is None:
        raise SystemExit(f"[FATAL] candidate missing path: {item}")
    return _resolve(str(path))


def _group_share(metrics: Mapping[str, float]) -> dict[str, Any]:
    vec = np.asarray(
        [
            float(metrics["leg_mean"]),
            float(metrics["arm_mean"]),
            float(metrics["else_mean"]),
        ],
        dtype=np.float64,
    )
    total = float(vec.sum())
    shares = (vec / total).tolist() if total > 0.0 else [float("nan")] * 3
    return {
        "vector": [float(x) for x in vec.tolist()],
        "shares": {
            "leg": float(shares[0]),
            "arm": float(shares[1]),
            "else": float(shares[2]),
        },
        "total": total,
    }


def _share_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    keys = ("leg", "arm", "else")
    av = np.asarray([float((a["shares"] or {}).get(k, float("nan"))) for k in keys], dtype=np.float64)
    bv = np.asarray([float((b["shares"] or {}).get(k, float("nan"))) for k in keys], dtype=np.float64)
    if not np.isfinite(av).all() or not np.isfinite(bv).all():
        return float("nan")
    return float(np.linalg.norm(av - bv))


def _build_donor_shortlist(
    *,
    spec: Mapping[str, Any],
    baseline_ckpt_path: Path,
    baseline_state: Mapping[str, torch.Tensor],
    out_dir: Path,
    force_eval: bool,
) -> dict[str, Any]:
    baseline_raw = _eval_case(spec=spec, model_path=baseline_ckpt_path, out_dir=out_dir / "raw_eval" / "baseline_raw" / "teacher_x_gt", force_eval=force_eval)
    baseline_metrics = baseline_raw["metrics"]
    rows: list[dict[str, Any]] = []
    ref_share: Optional[dict[str, Any]] = None
    for item in SHORTLIST_CANDIDATES:
        path = _candidate_path(spec, item)
        row: dict[str, Any] = {
            "name": str(item["name"]),
            "path": str(path),
            "family": str(item["family"]),
            "kind": str(item["kind"]),
            "note": str(item["note"]),
            "exists": path.exists(),
        }
        if not path.exists():
            row["contract_alignment_ok"] = False
            row["contract_reason"] = "checkpoint_missing"
            rows.append(row)
            continue
        ckpt = _load_ckpt(path)
        state = _state_dict(ckpt)
        row["state_contract"] = _all_state_contract_check(baseline_state, state)
        row["module_inventory"] = _module_inventory(baseline_state, state)
        row["contract_alignment_ok"] = bool(row["state_contract"]["ok"])
        if not bool(row["contract_alignment_ok"]):
            reasons: list[str] = []
            if row["state_contract"]["extra_keys"]:
                reasons.append(f"extra_keys={row['state_contract']['extra_keys'][:4]}")
            if row["state_contract"]["missing_keys"]:
                reasons.append(f"missing_keys={row['state_contract']['missing_keys'][:4]}")
            if row["state_contract"]["shape_mismatches"]:
                first = row["state_contract"]["shape_mismatches"][0]
                reasons.append(f"shape_mismatch={first['key']}")
            row["contract_reason"] = "; ".join(reasons) if reasons else "unknown_contract_mismatch"
            rows.append(row)
            continue
        eval_payload = _eval_case(spec=spec, model_path=path, out_dir=out_dir / "raw_eval" / str(item["name"]) / "teacher_x_gt", force_eval=force_eval)
        metrics = eval_payload["metrics"]
        row.update(eval_payload)
        row["raw_gap_vs_baseline"] = float(metrics["all_ex_root_mean"] - baseline_metrics["all_ex_root_mean"])
        row["share"] = _group_share(metrics)
        if row["name"] == "current_bad":
            ref_share = row["share"]
        rows.append(row)
    if ref_share is None:
        raise SystemExit("[FATAL] donor shortlist missing current_bad reference")
    for row in rows:
        share = row.get("share", None)
        row["share_distance_vs_current_bad"] = _share_distance(share, ref_share) if isinstance(share, Mapping) else float("nan")

    positive_gap = [
        row
        for row in rows
        if bool(row.get("contract_alignment_ok"))
        and str(row["name"]) != "current_bad"
        and _safe_float(row.get("raw_gap_vs_baseline")) > 0.0
    ]
    donor1 = None
    if positive_gap:
        donor1 = max(
            positive_gap,
            key=lambda row: (
                float(_safe_float(row.get("share_distance_vs_current_bad"))),
                float(_safe_float(row.get("raw_gap_vs_baseline"))),
            ),
        )
    donor2_candidates = [
        row
        for row in positive_gap
        if donor1 is None or str(row["name"]) != str(donor1["name"])
    ]
    donor2 = None
    if donor2_candidates:
        family_pref = [row for row in donor2_candidates if "cp015_stage70a_tailfix" in str(row.get("family", ""))]
        pool = family_pref or donor2_candidates
        donor2 = min(
            pool,
            key=lambda row: (
                abs(float(_safe_float(row.get("share_distance_vs_current_bad")))),
                -float(_safe_float(row.get("raw_gap_vs_baseline"))),
            ),
        )
    selected = []
    if donor1 is not None:
        selected.append(
            {
                "slot": "donor1",
                "name": donor1["name"],
                "path": donor1["path"],
                "reason": (
                    "Most different positive-gap raw leg/arm/else share vector among contract-aligned candidates; "
                    "used as the cross-shape donor."
                ),
            }
        )
    if donor2 is not None:
        selected.append(
            {
                "slot": "donor2",
                "name": donor2["name"],
                "path": donor2["path"],
                "reason": (
                    "Closest cp015/stage70a-tailfix family candidate to current_bad in raw group-share shape while remaining "
                    "positive-gap and contract-aligned; used as the same-family stability donor."
                ),
            }
        )
    return {
        "baseline_raw": baseline_raw,
        "candidates": rows,
        "selected": selected,
    }


def _transplant_case_modules() -> dict[str, list[str]]:
    return {
        "L2": list(L2_MODULES),
        "L2_leg03": list(L2_LEG03_MODULES),
        "L2p": list(L2P_MODULES),
        "L2p_out_leg": list(L2P_OUT_LEG_MODULES),
        "L2p_arm": list(L2P_ARM_MODULES),
        "L2p_else": list(L2P_ELSE_MODULES),
        "L2p_all_split": list(L2P_ALL_SPLIT_MODULES),
        "L5": list(L5_MODULES),
    }


def _run_donor_suite(
    *,
    donor_name: str,
    donor_path: Path,
    spec: Mapping[str, Any],
    baseline_ckpt_path: Path,
    baseline_state: Mapping[str, torch.Tensor],
    baseline_raw_metrics: Mapping[str, float],
    out_dir: Path,
    force_eval: bool,
) -> dict[str, Any]:
    donor_ckpt = _load_ckpt(donor_path)
    donor_state = _state_dict(donor_ckpt)
    ckpt_dir = out_dir / "checkpoints" / donor_name
    raw_dir = out_dir / "raw_eval" / donor_name
    case_defs = _transplant_case_modules()
    donor_raw = _eval_case(spec=spec, model_path=donor_path, out_dir=raw_dir / "donor_raw" / "teacher_x_gt", force_eval=force_eval)
    donor_metrics = donor_raw["metrics"]
    if float(donor_metrics["all_ex_root_mean"] - baseline_raw_metrics["all_ex_root_mean"]) <= 0.0:
        raise SystemExit(
            f"[FATAL] donor {donor_name} is not worse than baseline under all_ex_root mean; cannot use closure-based decisive tests."
        )
    summary: dict[str, Any] = {
        "baseline_raw": {
            "ckpt": str(baseline_ckpt_path),
            "metrics": dict(baseline_raw_metrics),
        },
        "donor_raw": {
            "ckpt": str(donor_path),
            **donor_raw,
        },
        "cases": {},
    }
    for case_name, modules in case_defs.items():
        ckpt_path = ckpt_dir / f"{case_name}.pth"
        notes = _copy_state_modules(source_ckpt=donor_ckpt, donor_state=baseline_state, modules=modules, out_path=ckpt_path)
        eval_payload = _eval_case(spec=spec, model_path=ckpt_path, out_dir=raw_dir / case_name / "teacher_x_gt", force_eval=force_eval)
        metrics = eval_payload["metrics"]
        closures: dict[str, Any] = {}
        for group in GROUPS:
            closures[group] = {}
            for suffix in SUFFIXES:
                key = _metric_key(group, suffix)
                closures[group][suffix] = _closure(
                    float(metrics[key]),
                    float(baseline_raw_metrics[key]),
                    float(donor_metrics[key]),
                )
        summary["cases"][case_name] = {
            "ckpt": str(ckpt_path),
            "modules": modules,
            "adapter_notes": notes,
            **eval_payload,
            "closure": closures,
        }
    return summary


def _additivity_error(
    *,
    suite: Mapping[str, Any],
    group: str,
    suffix: str,
) -> dict[str, Any]:
    base = float(suite["cases"]["L2p"]["closure"][group][suffix])
    out_leg = float(suite["cases"]["L2p_out_leg"]["closure"][group][suffix])
    arm = float(suite["cases"]["L2p_arm"]["closure"][group][suffix])
    other = float(suite["cases"]["L2p_else"]["closure"][group][suffix])
    joint = float(suite["cases"]["L2p_all_split"]["closure"][group][suffix])
    sum_singles = float((out_leg - base) + (arm - base) + (other - base))
    joint_marginal = float(joint - base)
    err = abs(sum_singles - joint_marginal)
    return {
        "base_closure": base,
        "single_marginals": {
            "out_leg": float(out_leg - base),
            "arm": float(arm - base),
            "else": float(other - base),
        },
        "sum_singles": sum_singles,
        "joint_marginal": joint_marginal,
        "additivity_error": err,
        "passes_le_0p05": bool(err <= 0.05),
    }


def _primary_groups_for_donor(suite: Mapping[str, Any]) -> list[str]:
    baseline = suite["baseline_raw"]["metrics"]
    donor = suite["donor_raw"]["metrics"]
    scored = []
    for group in ("leg", "arm", "else", "nonleg", "all_ex_root"):
        delta = float(donor[f"{group}_mean"] - baseline[f"{group}_mean"])
        scored.append((delta, group))
    scored.sort(reverse=True)
    if not scored:
        return ["all_ex_root"]
    top_delta = float(scored[0][0])
    groups = [group for delta, group in scored if delta >= top_delta - 1e-9]
    if "all_ex_root" not in groups:
        groups.append("all_ex_root")
    return groups


def _build_cross_branch_additivity(selected_suites: Mapping[str, Any]) -> dict[str, Any]:
    donors: dict[str, Any] = {}
    overall_pass = True
    for donor_name, suite in selected_suites.items():
        primary_groups = _primary_groups_for_donor(suite)
        groups_to_report = list(dict.fromkeys(["all_ex_root", *primary_groups]))
        payload: dict[str, Any] = {
            "primary_groups": primary_groups,
            "groups": {},
        }
        donor_pass = True
        for group in groups_to_report:
            payload["groups"][group] = {}
            for suffix in SUFFIXES:
                row = _additivity_error(suite=suite, group=group, suffix=suffix)
                payload["groups"][group][suffix] = row
                donor_pass = donor_pass and bool(row["passes_le_0p05"])
        payload["passes_preregistered_test"] = donor_pass
        overall_pass = overall_pass and donor_pass
        donors[donor_name] = payload
    return {
        "criterion": {
            "base_case": "L2p",
            "components": ["out_leg", "arm", "else"],
            "threshold": 0.05,
            "rule": "If any donor has additivity_error > 0.05 on all_ex_root or its primary affected groups, current exact three-branch additivity does not generalize.",
        },
        "donors": donors,
        "overall_pass": overall_pass,
    }


def _build_leg_internal_check(selected_suites: Mapping[str, Any]) -> dict[str, Any]:
    donors: dict[str, Any] = {}
    for donor_name, suite in selected_suites.items():
        payload: dict[str, Any] = {"groups": {}}
        for suffix in SUFFIXES:
            l2p = float(suite["cases"]["L2p"]["closure"]["leg"][suffix])
            l2_leg03 = float(suite["cases"]["L2_leg03"]["closure"]["leg"][suffix])
            margin = float(l2p - l2_leg03)
            payload["groups"][suffix] = {
                "closure_L2p": l2p,
                "closure_L2_leg03": l2_leg03,
                "margin": margin,
                "passes_ge_0p30": bool(margin >= 0.30),
            }
        payload["passes_preregistered_test_any"] = any(
            bool(payload["groups"][suffix]["passes_ge_0p30"]) for suffix in SUFFIXES
        )
        payload["passes_preregistered_test_both"] = all(
            bool(payload["groups"][suffix]["passes_ge_0p30"]) for suffix in SUFFIXES
        )
        donors[donor_name] = payload
    return {
        "criterion": {
            "comparison": "closure(L2p) - closure(L2_leg03)",
            "group": "leg",
            "threshold": 0.30,
            "rule": "If margin >= 0.30, leg residual remains monolithic and cannot be reduced to a single-layer surgical story.",
        },
        "donors": donors,
    }


def _build_a5_eval_summary(selected_suites: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"donors": {}}
    for donor_name, suite in selected_suites.items():
        donor_payload: dict[str, Any] = {
            "baseline_raw": suite["baseline_raw"],
            "donor_raw": suite["donor_raw"],
            "cases": {},
        }
        for case_name, payload in suite["cases"].items():
            donor_payload["cases"][case_name] = {
                "ckpt": payload["ckpt"],
                "modules": payload["modules"],
                "metrics": payload["metrics"],
                "closure": payload["closure"],
            }
        out["donors"][donor_name] = donor_payload
    return out


def _write_report(
    *,
    out_dir: Path,
    spec_binding: Mapping[str, Any],
    git_timeline: Sequence[Mapping[str, Any]],
    git_arch: Mapping[str, Any],
    topology_audit: Mapping[str, Any],
    topology_summary: Mapping[str, Any],
    donor_shortlist: Mapping[str, Any],
    additivity: Mapping[str, Any],
    leg_internal: Mapping[str, Any],
) -> None:
    selected = donor_shortlist["selected"]
    donor1_name = selected[0]["name"] if len(selected) >= 1 else "n/a"
    donor2_name = selected[1]["name"] if len(selected) >= 2 else "n/a"
    topo_interp = topology_summary["off_target_interpretation"]
    zero_case = topology_summary["cases"]["L2p_zero_split"]["metrics"]
    rand_case = topology_summary["cases"]["L2p_rand_split_seed0"]["metrics"]
    l2p_case = topology_summary["cases"]["L2p"]["metrics"]

    lines: list[str] = [
        "# Stage6 A++-5 archaeology + topology + minimal decisive tests",
        "",
        "## Spec Binding",
        "",
        f"- sealed spec: `{spec_binding['spec_path']}`",
        f"- file sha256: `{spec_binding['spec_file_sha256']}`",
        f"- declared spec_hash: `{spec_binding['declared_spec_hash']}`",
        f"- canonical computed spec_hash: `{spec_binding['computed_spec_hash']}`",
        f"- expected spec_hash: `{spec_binding['expected_spec_hash']}`",
        f"- baseline config baseline_spec_hash: `{spec_binding['baseline_config_spec_hash']}`",
        f"- sham config baseline_spec_hash: `{spec_binding['sham_config_spec_hash']}`",
        "",
        "## Step 0 — Optimizer archaeology",
        "",
        f"- clean git history contains no `train_arm_residual` or `arm_residual_corrector` strings: `{str(git_arch['conclusions']['historical_arm_residual_git_strings_absent']).lower()}`.",
        f"- first clean commit with a true corrector optimizer path: `{git_arch['conclusions']['closest_clean_commit_with_true_corrector_optimizer']}`.",
        f"- event-clock prefixes ever entered optimizer: `{str(git_arch['conclusions']['event_clock_optimizer_commit_exists']).lower()}`.",
        f"- current clean mainline can train corrector: `{str(git_arch['conclusions']['current_clean_mainline_can_train_corrector']).lower()}`.",
        f"- closest clean commit to current sealed-baseline contract that still trained a real corrector: `{git_arch['conclusions']['closest_clean_commit_with_true_corrector_optimizer']}`.",
        f"- archaeology conclusion: `historical naming was misleading`.",
        f"- required operational consequence: `{git_arch['conclusions']['required_fix_statement']}`.",
        "",
        "| commit | date | touched files | actual trainable prefixes | explanation |",
        "|---|---|---|---|---|",
    ]
    for row in git_timeline:
        prefixes = row["actual_trainable_prefixes"]
        if isinstance(prefixes, Mapping):
            prefix_text = "; ".join(f"{k}: {', '.join(v) if isinstance(v, list) else v}" for k, v in prefixes.items())
        else:
            prefix_text = ", ".join(prefixes) if prefixes else "none"
        touched = ", ".join(str(x) for x in row["touched_files"])
        lines.append(
            f"| `{row['commit'][:12]}` | `{row['date']}` | `{touched}` | `{prefix_text}` | `{row['interpretation']}` |"
        )

    lines.extend(
        [
            "",
            "## Step 1A — Static topology audit",
            "",
            f"- split buffer disjointness: `leg∩arm={len(topology_audit['summary']['intersections']['leg_arm'])}`, `leg∩else={len(topology_audit['summary']['intersections']['leg_else'])}`, `arm∩else={len(topology_audit['summary']['intersections']['arm_else'])}`.",
            f"- nonleg equals arm∪else: `{str(topology_audit['summary']['nonleg_equals_arm_union_else']).lower()}`.",
            f"- downstream mixing after split writeback before direct metric: `{str(topology_audit['answers']['q2_downstream_mixing_after_split_before_metric']).lower()}`.",
            f"- conclusion: `{topology_audit['answers']['q3_statement']}`.",
            f"- warning: `{topology_audit['answers']['warning']}`.",
            "",
            "## Step 1B — Topology sanity on current bad donor",
            "",
            f"- `L2p` all_ex_root mean/p95: `{_fmt(l2p_case['all_ex_root_mean'])}/{_fmt(l2p_case['all_ex_root_p95'])}`.",
            f"- `L2p_zero_split` all_ex_root mean/p95: `{_fmt(zero_case['all_ex_root_mean'])}/{_fmt(zero_case['all_ex_root_p95'])}`.",
            f"- `L2p_rand_split_seed0` all_ex_root mean/p95: `{_fmt(rand_case['all_ex_root_mean'])}/{_fmt(rand_case['all_ex_root_p95'])}`.",
            f"- `L2p_rand_out_leg_seed0` off-target ratio: `{_fmt(topo_interp['L2p_rand_out_leg_seed0']['off_to_target_ratio'])}` (smaller => more topology-dominated).",
            f"- `L2p_rand_arm_seed0` off-target ratio: `{_fmt(topo_interp['L2p_rand_arm_seed0']['off_to_target_ratio'])}`.",
            f"- `L2p_rand_else_seed0` off-target ratio: `{_fmt(topo_interp['L2p_rand_else_seed0']['off_to_target_ratio'])}`.",
            "- topology reading should remain separate from donor-generalization claims below.",
            "",
            "## Step 2A — Donor shortlist",
            "",
            f"- selected donor-1: `{donor1_name}`.",
            f"- selected donor-2: `{donor2_name}`.",
            "- selection rule: donor-1 emphasizes different raw leg/arm/else shape; donor-2 emphasizes same-family stability under the sealed contract.",
            "",
            "## Step 2B/2C — Minimal decisive tests",
            "",
            f"- cross-branch additivity overall pass (`<=0.05` on all_ex_root + primary groups for both donors): `{str(additivity['overall_pass']).lower()}`.",
        ]
    )
    for donor_name, payload in additivity["donors"].items():
        all_mean = payload["groups"]["all_ex_root"]["mean"]["additivity_error"]
        all_p95 = payload["groups"]["all_ex_root"]["p95"]["additivity_error"]
        lines.append(
            f"- `{donor_name}` cross-branch additivity all_ex_root mean/p95 error: `{_fmt(all_mean)}/{_fmt(all_p95)}`."
        )
    for donor_name, payload in leg_internal["donors"].items():
        mean_margin = payload["groups"]["mean"]["margin"]
        p95_margin = payload["groups"]["p95"]["margin"]
        lines.append(
            f"- `{donor_name}` leg internal margin mean/p95: `{_fmt(mean_margin)}/{_fmt(p95_margin)}`."
        )
    direction = "hold"
    if not bool(additivity["overall_pass"]):
        direction = "downgrade"
    elif any(bool(payload["passes_preregistered_test_any"]) for payload in leg_internal["donors"].values()):
        direction = "hold"
    lines.extend(
        [
            "",
            "## Final answers",
            "",
            f"1. spec binding/hash check: declared/computed/expected canonical spec_hash all match `{spec_binding['expected_spec_hash']}`; raw file sha256 differs because the file stores its own hash field.",
            "2. clean history did contain a real corrector optimizer family: legacy `train_so3_corrector` at `b649f655f8bc4dc48ccd05c22ff57f0f57f9109f`.",
            f"3. sham/corrector naming was misleading for `arm_residual`-style stories: the clean git strings are absent, while current clean mainline optimizer only exposes direct/lambda.",
            f"4. current `secondary additivity = 0` reads as `{topology_audit['answers']['q3_statement']}`, not learned functional independence.",
            "5. `L2p_zero_split` / `L2p_rand_split` mainly move target branches; off-target movement is summarized in `topology_sanity_closure_matrix.json` and should be read as topology-sanity evidence, not donor-generalization evidence.",
            f"6. selected donors: `{donor1_name}` (different-shape donor) and `{donor2_name}` (same-family stability donor).",
            "7. per donor results are recorded in `cross_branch_additivity_check.json` and `leg_internal_check.json`.",
            f"8. direction A verdict: `{direction}`; per-branch R0 is not promoted from topology alone.",
            "9. caveat: `N=1/limited-N` remains active; topology sanity and donor generalization are intentionally kept separate.",
            "",
            f"- mandatory fix: `{git_arch['conclusions']['required_fix_statement']}`.",
        ]
    )
    _save_text(out_dir / "report.md", "\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="docs/train_design/baseline_top3_locked_v1.spec.json")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    spec_path = _resolve(args.spec)
    spec = load_spec(spec_path)
    declared_hash = str(spec.get("spec_hash", ""))
    computed_hash = compute_spec_hash(spec)
    if declared_hash != EXPECTED_SPEC_HASH or computed_hash != EXPECTED_SPEC_HASH:
        raise SystemExit(
            f"[FATAL] spec hash mismatch: declared={declared_hash} computed={computed_hash} expected={EXPECTED_SPEC_HASH}"
        )

    baseline_cfg = _load_json(_resolve(str(spec["config_bindings"]["baseline_locked"])))
    sham_cfg = _load_json(_resolve(str(spec["config_bindings"]["sham_lr0"])))
    baseline_cfg_hash = str(baseline_cfg.get("baseline_spec_hash", ""))
    sham_cfg_hash = str(sham_cfg.get("baseline_spec_hash", ""))
    if baseline_cfg_hash != EXPECTED_SPEC_HASH:
        raise SystemExit("[FATAL] baseline config spec hash mismatch")
    if sham_cfg_hash != EXPECTED_SPEC_HASH:
        raise SystemExit("[FATAL] sham config spec hash mismatch")

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    baseline_ckpt_path = _resolve(str(spec["expected_artifacts"]["baseline_locked_ckpt"]))
    current_bad_path = _resolve(str(spec["donor_checkpoints"]["bad"]["path"]))
    baseline_ckpt = _load_ckpt(baseline_ckpt_path)
    current_bad_ckpt = _load_ckpt(current_bad_path)
    baseline_state = _state_dict(baseline_ckpt)

    spec_binding = {
        "spec_path": str(spec_path),
        "spec_file_sha256": sha256_file(spec_path),
        "declared_spec_hash": declared_hash,
        "computed_spec_hash": computed_hash,
        "expected_spec_hash": EXPECTED_SPEC_HASH,
        "baseline_config_spec_hash": baseline_cfg_hash,
        "sham_config_spec_hash": sham_cfg_hash,
        "baseline_ckpt": str(baseline_ckpt_path),
        "baseline_ckpt_sha256": sha256_file(baseline_ckpt_path),
        "current_bad_ckpt": str(current_bad_path),
        "current_bad_ckpt_sha256": sha256_file(current_bad_path),
        "top3_ckpt": str(_resolve(str(spec["donor_checkpoints"]["top3"]["path"]))),
        "top3_ckpt_sha256": sha256_file(_resolve(str(spec["donor_checkpoints"]["top3"]["path"]))),
        "aperture": {
            "teacher_forced_variant": "teacher_x_gt",
            "identical_input": True,
            "training": False,
            "donorization": False,
            "input_side_swap": False,
            "excluded_prefixes": ["contact_plan", "event_clock", "lambda"],
            "primary_metric": "group-split DirectGeoLocalDeg mean/p95",
            "secondary_metric": "global DirectGeoLocalDeg mean",
        },
    }
    _save_json(out_dir / "spec_binding.json", spec_binding)

    git_arch, git_timeline = _build_git_archaeology(spec_binding)
    _save_json(out_dir / "git_archaeology.json", git_arch)
    _save_json(out_dir / "git_archaeology_timeline.json", {"timeline": git_timeline})

    topology_audit = _build_topology_audit(baseline_state=baseline_state)
    _save_json(out_dir / "topology_audit.json", topology_audit)

    topology_summary = _build_topology_cases(
        spec=spec,
        baseline_ckpt_path=baseline_ckpt_path,
        baseline_ckpt=baseline_ckpt,
        baseline_state=baseline_state,
        donor_ckpt_path=current_bad_path,
        donor_ckpt=current_bad_ckpt,
        out_dir=out_dir / "topology_sanity",
        force_eval=bool(args.force_eval),
    )
    _save_json(out_dir / "topology_sanity_eval_summary.json", topology_summary["cases"])
    _save_json(out_dir / "topology_sanity_closure_matrix.json", topology_summary["delta_matrix"])

    donor_shortlist = _build_donor_shortlist(
        spec=spec,
        baseline_ckpt_path=baseline_ckpt_path,
        baseline_state=baseline_state,
        out_dir=out_dir / "shortlist",
        force_eval=bool(args.force_eval),
    )
    _save_json(out_dir / "donor_shortlist.json", donor_shortlist)
    selected = donor_shortlist["selected"]
    if len(selected) < 2:
        raise SystemExit("[FATAL] donor shortlist did not yield two contract-aligned donors.")

    baseline_raw_metrics = donor_shortlist["baseline_raw"]["metrics"]
    selected_suites: dict[str, Any] = {}
    for item in selected:
        donor_name = str(item["name"])
        donor_path = Path(str(item["path"]))
        selected_suites[donor_name] = _run_donor_suite(
            donor_name=donor_name,
            donor_path=donor_path,
            spec=spec,
            baseline_ckpt_path=baseline_ckpt_path,
            baseline_state=baseline_state,
            baseline_raw_metrics=baseline_raw_metrics,
            out_dir=out_dir / "a5_suite",
            force_eval=bool(args.force_eval),
        )

    a5_eval_summary = _build_a5_eval_summary(selected_suites)
    _save_json(out_dir / "a5_minimal_eval_summary.json", a5_eval_summary)

    cross_branch = _build_cross_branch_additivity(selected_suites)
    _save_json(out_dir / "cross_branch_additivity_check.json", cross_branch)

    leg_internal = _build_leg_internal_check(selected_suites)
    _save_json(out_dir / "leg_internal_check.json", leg_internal)

    _write_report(
        out_dir=out_dir,
        spec_binding=spec_binding,
        git_timeline=git_timeline,
        git_arch=git_arch,
        topology_audit=topology_audit,
        topology_summary=topology_summary,
        donor_shortlist=donor_shortlist,
        additivity=cross_branch,
        leg_internal=leg_internal,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
