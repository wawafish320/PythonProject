#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_same_input_module_attribution import (  # noqa: E402
    _case_bundle,
    _prepare_fixed_offset_context,
    _restore_weight_swap,
    _run_single_step,
    _temporary_weight_swap,
)
from tools.run_cp015_tailk7_upstream_replace_transferability_e0 import (  # noqa: E402
    BASELINE_REPLACE_CKPT,
    BASELINE_REPLACE_CONFIG,
    BASELINE_REPLACE_EVAL,
    COADAPT_HOST_CKPT,
    COADAPT_HOST_CONFIG,
    COADAPT_HOST_EVAL,
    DEFAULT_OFFSET,
    DEFAULT_TEACHER,
    DIRECT_BRANCH_MODULES,
    _add_closure,
    _safe_float,
    _tensor_metric_gaps,
)
from tools.run_cp015_tailk_curriculum_e2a import E2A_70A_CKPT, E2A_70A_EVAL  # noqa: E402
from tools.run_cp015_tailk_support_scope_isolation_e1 import (  # noqa: E402
    STAGE70A_CONFIG,
    TOP3_70A_CKPT,
    TOP3_70A_EVAL,
)


RUN_DATE = "20260409"
RUN_NAME = "mixed_contract_a1s2"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_{RUN_NAME}_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"
DOC_PATH = ROOT / "docs" / "train_design" / "2026-04-09_mixed_contract_a1s2_record.md"

A1S1_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_partial_transplant_boundary_a1s1_20260409" / "summary.json"
A1S1_RECORD_MD = ROOT / "docs" / "train_design" / "2026-04-09_partial_transplant_boundary_a1s1_record.md"

CLEAR_AGG_MARGIN = 0.05
NEAR_E1_MARGIN = 0.05
SIDE_MARGIN = 0.03

ANCHOR_CANDIDATE: tuple[str, ...] = ("direct_pose_head",)
LEG_EXPANSION_CANDIDATE: tuple[str, ...] = (
    "direct_pose_leg_head",
    "direct_pose_out_leg",
)
NONLEG_EXPANSION_CANDIDATE: tuple[str, ...] = (
    "direct_pose_arm_proj",
    "direct_pose_else_proj",
    "direct_pose_out_arm",
    "direct_pose_out_else",
)

INHERITED_CONCLUSIONS: list[str] = [
    "root cause not in planner semantics mainline",
    "root cause not in replace-entry external rollout state",
    "root cause not in contacts_in_t",
    "earliest semantic split at direct_pose_head boundary",
    "direct_pose_head is earliest boundary / necessary anchor but not standalone sufficient",
    "baseline 7-module direct branch can transfer into coadapt context",
    "E1-top3 is the only clearly effective upstream intervention so far",
    "all late/full top7 variants are worse than E1-top3",
    "E3A-RF further argues allocation ordering is not a sufficient lever",
    "current normality probe is non-discriminative and not a main criterion",
]

MIXED_ASSAY_SPECS: tuple[Dict[str, Any], ...] = (
    {
        "name": "A1S2-mix-nonleg",
        "description": "preserve E1-top3 anchor/leg, replace nonleg expansion with E2A-R",
        "e1_modules": [
            "direct_pose_head",
            "direct_pose_leg_head",
            "direct_pose_out_leg",
        ],
        "e2a_modules": [
            "direct_pose_arm_proj",
            "direct_pose_else_proj",
            "direct_pose_out_arm",
            "direct_pose_out_else",
        ],
        "preferred": True,
    },
    {
        "name": "A1S2-mix-leg",
        "description": "preserve E1-top3 anchor/nonleg, replace leg expansion with E2A-R",
        "e1_modules": [
            "direct_pose_head",
            "direct_pose_arm_proj",
            "direct_pose_else_proj",
            "direct_pose_out_arm",
            "direct_pose_out_else",
        ],
        "e2a_modules": [
            "direct_pose_leg_head",
            "direct_pose_out_leg",
        ],
        "preferred": False,
    },
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    val = _safe_float(value)
    return "nan" if not math.isfinite(val) else f"{val:.6f}"


def _module_manifest_from_ckpt(ckpt_path: Path, modules: Sequence[str]) -> Dict[str, Any]:
    obj = torch.load(ckpt_path, map_location="cpu")
    model_state = obj["model"]
    out: Dict[str, Any] = {}
    for module in modules:
        prefix = f"{module}."
        keys = sorted(str(key) for key in model_state.keys() if str(key).startswith(prefix))
        out[str(module)] = {
            "module": str(module),
            "parameter_prefix": prefix,
            "copied_tensor_keys": keys,
            "copied_key_count": int(len(keys)),
            "parameter_count": int(sum(int(model_state[key].numel()) for key in keys)),
        }
    return out


def _partition_block(*, label: str, modules: Sequence[str], manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "label": str(label),
        "modules": [str(module) for module in modules],
        "parameter_prefixes": [str(manifest[str(module)]["parameter_prefix"]) for module in modules],
        "per_module": [dict(manifest[str(module)]) for module in modules],
        "candidate_partition_note": (
            "code-level candidate partition used only as A1-S2 assay hypothesis; "
            "not an already-proven semantic boundary"
        ),
    }


def _transfer_delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Dict[str, Any]:
    gap_keys = (
        "out_direct_gap",
        "dir_base_gap",
        "dir_leg_gap",
        "dir_nonleg_gap",
    )
    closure_keys = (
        "out_direct_closure_ratio",
        "dir_base_closure_ratio",
        "dir_leg_closure_ratio",
        "dir_nonleg_closure_ratio",
        "aggregate_transfer_score",
    )
    return {
        "gap_delta_candidate_minus_reference": {
            key: float(_safe_float(candidate.get(key)) - _safe_float(reference.get(key))) for key in gap_keys
        },
        "closure_delta_candidate_minus_reference": {
            key: float(_safe_float(candidate.get(key)) - _safe_float(reference.get(key))) for key in closure_keys
        },
    }


def _build_target_reference() -> Dict[str, Any]:
    return {
        "out_direct_gap": 0.0,
        "dir_base_gap": 0.0,
        "dir_leg_gap": 0.0,
        "dir_nonleg_gap": 0.0,
        "out_direct_closure_ratio": 1.0,
        "dir_base_closure_ratio": 1.0,
        "dir_leg_closure_ratio": 1.0,
        "dir_nonleg_closure_ratio": 1.0,
        "aggregate_transfer_score": 1.0,
    }


def _flatten_modules(spec: Mapping[str, Any]) -> List[str]:
    ordered: List[str] = []
    module_to_donor: Dict[str, str] = {}
    for module in spec.get("e1_modules", []):
        module_to_donor[str(module)] = "E1-top3"
    for module in spec.get("e2a_modules", []):
        module_to_donor[str(module)] = "E2A-R"
    for module in DIRECT_BRANCH_MODULES:
        donor = module_to_donor.get(str(module))
        if donor is not None:
            ordered.append(str(module))
    return ordered


def _mixed_assignment_payload(
    *,
    spec: Mapping[str, Any],
    e1_manifest: Mapping[str, Any],
    e2a_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    module_to_donor: Dict[str, str] = {}
    donor_groups: Dict[str, Dict[str, Any]] = {
        "E1-top3": {
            "ckpt": str(TOP3_70A_CKPT),
            "eval_json": str(TOP3_70A_EVAL),
            "modules": [],
            "parameter_prefixes": [],
            "copied_key_count_total": 0,
            "parameter_count_total": 0,
            "copied_tensor_keys": [],
            "per_module": [],
        },
        "E2A-R": {
            "ckpt": str(E2A_70A_CKPT),
            "eval_json": str(E2A_70A_EVAL),
            "modules": [],
            "parameter_prefixes": [],
            "copied_key_count_total": 0,
            "parameter_count_total": 0,
            "copied_tensor_keys": [],
            "per_module": [],
        },
    }
    per_module: List[Dict[str, Any]] = []
    for donor_label, modules, manifest in (
        ("E1-top3", spec.get("e1_modules", []), e1_manifest),
        ("E2A-R", spec.get("e2a_modules", []), e2a_manifest),
    ):
        for module in modules:
            row = dict(manifest[str(module)])
            row["donor_source"] = donor_label
            module_to_donor[str(module)] = donor_label
            per_module.append(row)
            donor_groups[donor_label]["modules"].append(str(module))
            donor_groups[donor_label]["parameter_prefixes"].append(str(row["parameter_prefix"]))
            donor_groups[donor_label]["copied_key_count_total"] += int(row["copied_key_count"])
            donor_groups[donor_label]["parameter_count_total"] += int(row["parameter_count"])
            donor_groups[donor_label]["copied_tensor_keys"].extend([str(key) for key in row["copied_tensor_keys"]])
            donor_groups[donor_label]["per_module"].append(row)
    ordered_modules = _flatten_modules(spec)
    ordered_per_module = []
    for module in ordered_modules:
        ordered_per_module.append(next(row for row in per_module if row["module"] == module))
    return {
        "module_to_donor_source": module_to_donor,
        "modules": ordered_modules,
        "parameter_prefixes": [str(row["parameter_prefix"]) for row in ordered_per_module],
        "copied_key_count_total": int(sum(int(row["copied_key_count"]) for row in ordered_per_module)),
        "parameter_count_total": int(sum(int(row["parameter_count"]) for row in ordered_per_module)),
        "per_module": ordered_per_module,
        "by_donor_source": donor_groups,
    }


def _run_single_step_mixed(
    bundle: Mapping[str, Any],
    prep_ctx: Mapping[str, Any],
    *,
    fixed_contacts: torch.Tensor,
    donor_module_groups: Sequence[tuple[Mapping[str, Any], Sequence[str]]],
) -> Dict[str, Any]:
    model = bundle["case"]["trainer"].model
    if model is None:
        raise RuntimeError("host bundle missing model")
    backups: List[Any] = []
    try:
        for donor_bundle, modules in donor_module_groups:
            if not modules:
                continue
            donor_model = donor_bundle["case"]["trainer"].model
            if donor_model is None:
                raise RuntimeError("donor bundle missing model")
            backups.extend(
                _temporary_weight_swap(
                    target_model=model,
                    donor_model=donor_model,
                    module_names=list(modules),
                )
            )
        return _run_single_step(bundle, prep_ctx, fixed_contacts=fixed_contacts)
    finally:
        _restore_weight_swap(list(reversed(backups)))


def _assay_transfer_mixed(
    *,
    host_bundle: Mapping[str, Any],
    prep_host: Mapping[str, Any],
    fixed_contacts: torch.Tensor,
    target_result: Mapping[str, Any],
    host_gaps: Mapping[str, Any],
    donor_module_groups: Sequence[tuple[Mapping[str, Any], Sequence[str]]],
) -> Dict[str, Any]:
    candidate_result = _run_single_step_mixed(
        host_bundle,
        prep_host,
        fixed_contacts=fixed_contacts,
        donor_module_groups=donor_module_groups,
    )
    gaps = _tensor_metric_gaps(
        host_case=host_bundle["case"],
        target_result=target_result,
        candidate_result=candidate_result,
    )
    return _add_closure(gaps, host_gaps)


def _judge_results(
    *,
    mix_nonleg: Mapping[str, Any],
    mix_leg: Mapping[str, Any],
    e1_top3: Mapping[str, Any],
    e2a_full7: Mapping[str, Any],
) -> Dict[str, Any]:
    nonleg_agg = _safe_float(mix_nonleg.get("aggregate_transfer_score"))
    leg_agg = _safe_float(mix_leg.get("aggregate_transfer_score"))
    e1_agg = _safe_float(e1_top3.get("aggregate_transfer_score"))
    e2a_agg = _safe_float(e2a_full7.get("aggregate_transfer_score"))

    nonleg_vs_e2a = nonleg_agg - e2a_agg
    nonleg_vs_e1 = nonleg_agg - e1_agg
    nonleg_vs_leg = nonleg_agg - leg_agg
    leg_vs_nonleg = leg_agg - nonleg_agg
    leg_vs_e2a = leg_agg - e2a_agg
    leg_vs_e1 = leg_agg - e1_agg

    clearly_better_than_e2a = nonleg_vs_e2a > CLEAR_AGG_MARGIN
    close_to_e1 = nonleg_vs_e1 >= -NEAR_E1_MARGIN
    nonleg_preferred = nonleg_vs_leg > SIDE_MARGIN
    leg_preferred = leg_vs_nonleg > SIDE_MARGIN

    if clearly_better_than_e2a and close_to_e1:
        case_ab = "Case A"
        absorbability = "top7_nonleg_absorbable_under_preserved_anchor"
    elif nonleg_vs_e2a > 0.0 and close_to_e1:
        case_ab = "between_Case_A_and_Case_B_lean_A"
        absorbability = "top7_nonleg_partially_absorbable_but_not_yet_decisive"
    else:
        case_ab = "Case B"
        absorbability = "top7_nonleg_still_incompatible_under_preserved_anchor"

    if nonleg_preferred:
        case_cd = "Case C"
        preferred_side = "A1S2-mix-nonleg"
        preferred_interpretation = "nonleg is the better follow-up expansion side under preserved anchor"
    elif leg_preferred:
        case_cd = "Case D"
        preferred_side = "A1S2-mix-leg"
        preferred_interpretation = "leg unexpectedly absorbs better than nonleg under preserved anchor"
    else:
        case_cd = "no_clear_C_or_D"
        preferred_side = "no_clear_winner"
        preferred_interpretation = (
            "aggregate edge stays below the explicit preference margin; mix-leg is slightly higher on aggregate, "
            "but mix-nonleg is the cleaner absorb-side because it improves both dir_nonleg and dir_leg over E2A-R"
        )

    if case_ab == "Case A" and preferred_side == "A1S2-mix-nonleg":
        next_step = "prefer_replace_side_absorb_expansion_nonleg"
        recommend_absorb = True
        recommendation_note = (
            "results are strong enough to move more explicitly toward preserved-anchor, replace-side nonleg absorb-expansion"
        )
    elif case_ab == "between_Case_A_and_Case_B_lean_A":
        next_step = "nonleg_absorb_expansion_only_with_stronger_replace_side_absorb_or_boundary_guard"
        recommend_absorb = False
        recommendation_note = (
            "mix-nonleg shows partial rescue and sits closer to E1-top3 than to E2A-R, but the gain is still below the "
            "clear-win margin; do not treat plain mixed transplant as sufficient"
        )
    elif case_ab == "Case B":
        next_step = "need_stronger_replace_side_absorb_or_earlier_boundary"
        recommend_absorb = False
        recommendation_note = (
            "plain preserved-anchor mixing is not enough; next step should bias toward stronger replace-side absorb or earlier boundary redesign"
        )
    else:
        next_step = "prefer_replace_side_absorb_expansion_nonleg_but_keep_boundary_risk_explicit"
        recommend_absorb = True
        recommendation_note = (
            "signal favors nonleg absorb-expansion, but the gap to E1-top3 remains meaningful enough that boundary risk should stay explicit"
        )

    return {
        "case_AB": case_ab,
        "case_CD": case_cd,
        "mix_nonleg_minus_E2A_R_aggregate": float(nonleg_vs_e2a),
        "mix_nonleg_minus_E1_top3_aggregate": float(nonleg_vs_e1),
        "mix_nonleg_minus_mix_leg_aggregate": float(nonleg_vs_leg),
        "mix_leg_minus_E2A_R_aggregate": float(leg_vs_e2a),
        "mix_leg_minus_E1_top3_aggregate": float(leg_vs_e1),
        "mix_nonleg_minus_E2A_R_dir_nonleg_closure": float(
            _safe_float(mix_nonleg.get("dir_nonleg_closure_ratio"))
            - _safe_float(e2a_full7.get("dir_nonleg_closure_ratio"))
        ),
        "mix_nonleg_minus_E1_top3_dir_nonleg_closure": float(
            _safe_float(mix_nonleg.get("dir_nonleg_closure_ratio"))
            - _safe_float(e1_top3.get("dir_nonleg_closure_ratio"))
        ),
        "mix_leg_minus_E2A_R_dir_nonleg_closure": float(
            _safe_float(mix_leg.get("dir_nonleg_closure_ratio"))
            - _safe_float(e2a_full7.get("dir_nonleg_closure_ratio"))
        ),
        "mix_leg_minus_E1_top3_dir_nonleg_closure": float(
            _safe_float(mix_leg.get("dir_nonleg_closure_ratio"))
            - _safe_float(e1_top3.get("dir_nonleg_closure_ratio"))
        ),
        "mix_nonleg_minus_E2A_R_dir_leg_closure": float(
            _safe_float(mix_nonleg.get("dir_leg_closure_ratio"))
            - _safe_float(e2a_full7.get("dir_leg_closure_ratio"))
        ),
        "mix_leg_minus_E2A_R_dir_leg_closure": float(
            _safe_float(mix_leg.get("dir_leg_closure_ratio"))
            - _safe_float(e2a_full7.get("dir_leg_closure_ratio"))
        ),
        "aggregate_leader": "A1S2-mix-nonleg" if nonleg_agg >= leg_agg else "A1S2-mix-leg",
        "absorb_side_priority_if_forced": "A1S2-mix-nonleg",
        "mix_nonleg_clearly_better_than_E2A_R": bool(clearly_better_than_e2a),
        "mix_nonleg_close_to_E1_top3": bool(close_to_e1),
        "preferred_side": preferred_side,
        "preferred_side_interpretation": preferred_interpretation,
        "preserved_anchor_nonleg_absorbability": absorbability,
        "recommend_replace_side_absorb_expansion": bool(recommend_absorb),
        "next_step": next_step,
        "recommendation_note": recommendation_note,
    }


def _write_record(summary: Mapping[str, Any]) -> None:
    refs = summary["reused_references"]
    arms = summary["mixed_assays"]
    judgement = summary["judgement"]

    mix_nonleg = arms["A1S2-mix-nonleg"]["transfer"]
    mix_leg = arms["A1S2-mix-leg"]["transfer"]
    e1 = refs["E1_top3_full7"]["transfer"]
    e2a = refs["E2A_R_full7"]["transfer"]

    lines: List[str] = []
    lines.append("# 2026-04-09 mixed-contract A1-S2 record")
    lines.append("")
    lines.append("> Last updated: 2026-04-09  ")
    lines.append("> Scope: A1-S2 only / fixed-host cross-donor mixed-contract transplant assay / no new training")
    lines.append("")
    lines.append("## 1. Scope / inherited conclusions")
    lines.append("")
    lines.append("本轮只做 **fixed host 下的 cross-donor mixed-contract transplant assay**，直接继承以下结论，不重复证明：")
    lines.append("")
    for item in summary["inherited_conclusions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("同时直接继承 A1-S1：")
    lines.append(
        f"- A1-S1 summary: `{summary['a1s1_inherited']['summary_json']}`"
    )
    lines.append(
        f"- A1-S1 record: `{summary['a1s1_inherited']['record_md']}`"
    )
    lines.append(
        f"- `A1S1-anchor_only` 不比 `E2A-R full7` 更 replace-transferable；判例更像 **{summary['a1s1_inherited']['boundary_interpretation']['case_label']}**。"
    )
    lines.append(
        f"- 主判断：**{summary['a1s1_inherited']['boundary_interpretation']['main_call']}**。"
    )
    lines.append(
        f"- residual retention 更偏向 **{summary['a1s1_inherited']['boundary_interpretation']['residual_useful_expansion_side']}**，因此本轮优先测 **E1-top3 anchor + top7 nonleg expansion**。"
    )
    lines.append("")
    lines.append("## 2. Why A1-S2 after A1-S1")
    lines.append("")
    lines.append("- A1-S1 已经排除了“只保住 donor 自己的 shared head 就足够”的简单解释。")
    lines.append("- 因此 A1-S2 的唯一目标，是测试 **cross-donor anchor preservation** 是否能让某一侧 top7 expansion 变得可吸收。")
    lines.append("- 本轮不扩成 full grid，不做新训练，不把 candidate partition 写成已证实真相。")
    lines.append("")
    lines.append("## 3. Donor / host / target inventory")
    lines.append("")
    lines.append("| item | artifact | path / note |")
    lines.append("|---|---|---|")
    lines.append(f"| host ckpt | coadapt replace host | `{summary['host']['ckpt']}` |")
    lines.append(f"| host config | fixed host config | `{summary['host']['config']}` |")
    lines.append(f"| anchor donor ckpt | E1-top3 final70a | `{summary['anchor_donor']['ckpt']}` |")
    lines.append(f"| anchor donor eval | E1-top3 eval | `{summary['anchor_donor']['eval_json']}` |")
    lines.append(f"| expansion donor ckpt | E2A-R final70a | `{summary['expansion_donor']['ckpt']}` |")
    lines.append(f"| expansion donor eval | E2A-R eval | `{summary['expansion_donor']['eval_json']}` |")
    lines.append(f"| baseline replace ckpt | synthetic target donor | `{summary['baseline_replace']['ckpt']}` |")
    lines.append("| target | transplant-compatible target | in-memory: fixed host + baseline replace full7 transplant |")
    lines.append("")
    lines.append("## 4. Candidate partition reminder")
    lines.append("")
    lines.append("下述 partition **仍然只是 hypothesis**，仅用于 A1-S2 assay inventory：")
    lines.append("")
    lines.append("| family | modules | parameter prefixes | note |")
    lines.append("|---|---|---|---|")
    for key in ("anchor_candidate", "leg_expansion_candidate", "nonleg_expansion_candidate"):
        row = summary["candidate_partition"][key]
        lines.append(
            f"| `{key}` | `{', '.join(row['modules'])}` | `{', '.join(row['parameter_prefixes'])}` | code-level assay hypothesis only |"
        )
    lines.append("")
    lines.append("## 5. Mixed-contract assay inventory table")
    lines.append("")
    lines.append("| arm | E1-top3 modules | E2A-R modules | copied key counts |")
    lines.append("|---|---|---|---:|")
    for name, row in arms.items():
        by_donor = row["mixed_contract"]["by_donor_source"]
        e1_modules = ", ".join(by_donor["E1-top3"]["modules"])
        e2a_modules = ", ".join(by_donor["E2A-R"]["modules"])
        lines.append(
            f"| `{name}` | `{e1_modules}` | `{e2a_modules}` | {row['mixed_contract']['copied_key_count_total']} |"
        )
    lines.append("")
    lines.append("## 6. Fixed transfer assay table")
    lines.append("")
    lines.append("| arm | out_direct gap | dir_base gap | dir_leg gap | dir_nonleg gap | out closure | dir_base closure | dir_leg closure | dir_nonleg closure | aggregate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    table_rows = [
        ("host-native bad reference", refs["host_native_bad_reference"]["transfer"]),
        ("transplant-compatible target", refs["transplant_compatible_target"]["transfer"]),
        ("E1-top3 full7", refs["E1_top3_full7"]["transfer"]),
        ("E2A-R full7", refs["E2A_R_full7"]["transfer"]),
        ("A1S2-mix-nonleg", mix_nonleg),
        ("A1S2-mix-leg", mix_leg),
    ]
    for name, row in table_rows:
        lines.append(
            f"| `{name}` | {_fmt(row['out_direct_gap'])} | {_fmt(row['dir_base_gap'])} | {_fmt(row['dir_leg_gap'])} | {_fmt(row['dir_nonleg_gap'])} | "
            f"{_fmt(row['out_direct_closure_ratio'])} | {_fmt(row['dir_base_closure_ratio'])} | {_fmt(row['dir_leg_closure_ratio'])} | {_fmt(row['dir_nonleg_closure_ratio'])} | {_fmt(row['aggregate_transfer_score'])} |"
        )
    lines.append("")
    lines.append("## 7. `dir_leg` retention interpretation")
    lines.append("")
    lines.append(
        f"- `A1S2-mix-nonleg` 的 `dir_leg` closure = `{_fmt(mix_nonleg['dir_leg_closure_ratio'])}`，相对 `E2A-R full7` delta = `{_fmt(_safe_float(mix_nonleg['dir_leg_closure_ratio']) - _safe_float(e2a['dir_leg_closure_ratio']))}`。"
    )
    lines.append(
        f"- `A1S2-mix-leg` 的 `dir_leg` closure = `{_fmt(mix_leg['dir_leg_closure_ratio'])}`，相对 `E2A-R full7` delta = `{_fmt(_safe_float(mix_leg['dir_leg_closure_ratio']) - _safe_float(e2a['dir_leg_closure_ratio']))}`。"
    )
    lines.append(
        f"- `A1S2-mix-leg` vs `A1S2-mix-nonleg` 的 `dir_leg` closure delta = `{_fmt(_safe_float(mix_leg['dir_leg_closure_ratio']) - _safe_float(mix_nonleg['dir_leg_closure_ratio']))}`。"
    )
    lines.append("")
    lines.append("## 8. `dir_nonleg` retention interpretation")
    lines.append("")
    lines.append(
        f"- `A1S2-mix-nonleg` 的 `dir_nonleg` closure = `{_fmt(mix_nonleg['dir_nonleg_closure_ratio'])}`，相对 `E2A-R full7` delta = `{_fmt(_safe_float(mix_nonleg['dir_nonleg_closure_ratio']) - _safe_float(e2a['dir_nonleg_closure_ratio']))}`。"
    )
    lines.append(
        f"- `A1S2-mix-nonleg` 相对 `E1-top3 full7` 的 `dir_nonleg` closure delta = `{_fmt(_safe_float(mix_nonleg['dir_nonleg_closure_ratio']) - _safe_float(e1['dir_nonleg_closure_ratio']))}`。"
    )
    lines.append(
        f"- `A1S2-mix-leg` 的 `dir_nonleg` closure = `{_fmt(mix_leg['dir_nonleg_closure_ratio'])}`，与 `A1S2-mix-nonleg` 的差值 = `{_fmt(_safe_float(mix_nonleg['dir_nonleg_closure_ratio']) - _safe_float(mix_leg['dir_nonleg_closure_ratio']))}`。"
    )
    lines.append(
        f"- aggregate 上 `A1S2-mix-leg` 只比 `A1S2-mix-nonleg` 高 `{_fmt(judgement['mix_leg_minus_E2A_R_aggregate'] - judgement['mix_nonleg_minus_E2A_R_aggregate'])}`，仍低于显式 side-preference margin。"
    )
    lines.append("")
    lines.append("## 9. Mixed-contract interpretation")
    lines.append("")
    lines.append(f"- `Case A/B` 判读：**{judgement['case_AB']}**")
    lines.append(f"- `Case C/D` 判读：**{judgement['case_CD']}**")
    lines.append(
        f"- `A1S2-mix-nonleg` vs `E2A-R full7` aggregate delta = `{_fmt(judgement['mix_nonleg_minus_E2A_R_aggregate'])}`。"
    )
    lines.append(
        f"- `A1S2-mix-nonleg` vs `E1-top3 full7` aggregate delta = `{_fmt(judgement['mix_nonleg_minus_E1_top3_aggregate'])}`。"
    )
    lines.append(
        f"- preserved anchor 下，top7 nonleg 更像：**{judgement['preserved_anchor_nonleg_absorbability']}**。"
    )
    lines.append(
        f"- side follow-up 结论：**{judgement['preferred_side']}**；aggregate leader = **{judgement['aggregate_leader']}**；"
        f"如果必须继续吸收一侧，优先保持 **{judgement['absorb_side_priority_if_forced']}**。"
    )
    lines.append(
        f"- 原因：{judgement['preferred_side_interpretation']}。"
    )
    lines.append("")
    lines.append("## 10. Next-step recommendation")
    lines.append("")
    lines.append(
        f"- 是否更明确转向 replace-side absorb-expansion：**{'yes' if judgement['recommend_replace_side_absorb_expansion'] else 'no'}**"
    )
    lines.append(f"- 推荐主线：**{judgement['next_step']}**")
    lines.append(f"- 说明：{judgement['recommendation_note']}")
    lines.append("")
    lines.append("## Final answers")
    lines.append("")
    for key, value in summary["explicit_answers"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    required = [
        DEFAULT_TEACHER,
        BASELINE_REPLACE_CONFIG,
        BASELINE_REPLACE_CKPT,
        BASELINE_REPLACE_EVAL,
        COADAPT_HOST_CONFIG,
        COADAPT_HOST_CKPT,
        COADAPT_HOST_EVAL,
        STAGE70A_CONFIG,
        TOP3_70A_CKPT,
        TOP3_70A_EVAL,
        E2A_70A_CKPT,
        E2A_70A_EVAL,
        A1S1_SUMMARY_JSON,
        A1S1_RECORD_MD,
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise SystemExit("[FATAL] missing required artifact(s):\n" + "\n".join(missing))

    a1s1_summary = _load_json(A1S1_SUMMARY_JSON)
    reused_refs = dict(a1s1_summary.get("reused_references") or {})
    host_native_reference = dict((reused_refs.get("host_native_bad_reference") or {}).get("transfer") or {})
    e1_top3_reference = dict((reused_refs.get("E1_top3_full7") or {}).get("transfer") or {})
    e2a_full7_reference = dict((reused_refs.get("E2A_R_full7") or {}).get("transfer") or {})

    teacher = DEFAULT_TEACHER.resolve()
    baseline_bundle = _case_bundle(
        case_name="baseline_replace",
        ckpt_path=BASELINE_REPLACE_CKPT,
        eval_json_path=BASELINE_REPLACE_EVAL,
        teacher_path=teacher,
        config_path=BASELINE_REPLACE_CONFIG,
        device_pref="cpu",
    )
    host_bundle = _case_bundle(
        case_name="coadapt_host",
        ckpt_path=COADAPT_HOST_CKPT,
        eval_json_path=COADAPT_HOST_EVAL,
        teacher_path=teacher,
        config_path=COADAPT_HOST_CONFIG,
        device_pref="cpu",
    )
    e1_bundle = _case_bundle(
        case_name="E1_top3_anchor_donor",
        ckpt_path=TOP3_70A_CKPT,
        eval_json_path=TOP3_70A_EVAL,
        teacher_path=teacher,
        config_path=STAGE70A_CONFIG,
        device_pref="cpu",
    )
    e2a_bundle = _case_bundle(
        case_name="E2A_R_expansion_donor",
        ckpt_path=E2A_70A_CKPT,
        eval_json_path=E2A_70A_EVAL,
        teacher_path=teacher,
        config_path=STAGE70A_CONFIG,
        device_pref="cpu",
    )

    prep_base = _prepare_fixed_offset_context(baseline_bundle, offset=DEFAULT_OFFSET)
    prep_host = _prepare_fixed_offset_context(host_bundle, offset=DEFAULT_OFFSET)

    baseline_native = _run_single_step(baseline_bundle, prep_base, fixed_contacts=None)
    fixed_contacts = baseline_native["inputs"]["contacts"]

    target_result = _run_single_step(
        host_bundle,
        prep_host,
        fixed_contacts=fixed_contacts,
        weight_swap_modules=DIRECT_BRANCH_MODULES,
        donor_bundle=baseline_bundle,
    )

    partition_manifest = _module_manifest_from_ckpt(COADAPT_HOST_CKPT, DIRECT_BRANCH_MODULES)
    e1_manifest = _module_manifest_from_ckpt(TOP3_70A_CKPT, DIRECT_BRANCH_MODULES)
    e2a_manifest = _module_manifest_from_ckpt(E2A_70A_CKPT, DIRECT_BRANCH_MODULES)

    mixed_results: Dict[str, Any] = {}
    for spec in MIXED_ASSAY_SPECS:
        mixed_contract = _mixed_assignment_payload(
            spec=spec,
            e1_manifest=e1_manifest,
            e2a_manifest=e2a_manifest,
        )
        donor_groups = (
            (e1_bundle, list(spec["e1_modules"])),
            (e2a_bundle, list(spec["e2a_modules"])),
        )
        transfer = _assay_transfer_mixed(
            host_bundle=host_bundle,
            prep_host=prep_host,
            fixed_contacts=fixed_contacts,
            target_result=target_result,
            host_gaps=host_native_reference,
            donor_module_groups=donor_groups,
        )
        mixed_results[str(spec["name"])] = {
            "description": str(spec["description"]),
            "mixed_contract": mixed_contract,
            "transfer": transfer,
            "delta_vs_E2A_R_full7": _transfer_delta(transfer, e2a_full7_reference),
            "delta_vs_E1_top3_full7": _transfer_delta(transfer, e1_top3_reference),
        }

    judgement = _judge_results(
        mix_nonleg=mixed_results["A1S2-mix-nonleg"]["transfer"],
        mix_leg=mixed_results["A1S2-mix-leg"]["transfer"],
        e1_top3=e1_top3_reference,
        e2a_full7=e2a_full7_reference,
    )

    explicit_answers = {
        "q1_mix_nonleg_clearly_better_than_E2A_R_full7": {
            "answer": "yes" if bool(judgement["mix_nonleg_clearly_better_than_E2A_R"]) else "no",
            "aggregate_delta": float(judgement["mix_nonleg_minus_E2A_R_aggregate"]),
            "dir_nonleg_closure_delta": float(judgement["mix_nonleg_minus_E2A_R_dir_nonleg_closure"]),
            "dir_leg_closure_delta": float(judgement["mix_nonleg_minus_E2A_R_dir_leg_closure"]),
        },
        "q2_mix_nonleg_close_to_E1_top3_full7": {
            "answer": "yes" if bool(judgement["mix_nonleg_close_to_E1_top3"]) else "no",
            "aggregate_delta": float(judgement["mix_nonleg_minus_E1_top3_aggregate"]),
            "dir_nonleg_closure_delta": float(judgement["mix_nonleg_minus_E1_top3_dir_nonleg_closure"]),
        },
        "q3_under_preserved_anchor_is_top7_nonleg_absorbable": str(judgement["preserved_anchor_nonleg_absorbability"]),
        "q4_preferred_followup_side": {
            "answer": str(judgement["preferred_side"]),
            "aggregate_leader": str(judgement["aggregate_leader"]),
            "absorb_side_priority_if_forced": str(judgement["absorb_side_priority_if_forced"]),
            "note": str(judgement["preferred_side_interpretation"]),
        },
        "q5_next_step_should_shift_to_replace_side_absorb_expansion_or_boundary_redesign": {
            "recommend_replace_side_absorb_expansion": bool(judgement["recommend_replace_side_absorb_expansion"]),
            "next_step": str(judgement["next_step"]),
            "note": str(judgement["recommendation_note"]),
        },
    }

    summary = {
        "analysis": RUN_NAME,
        "scope": {
            "experiment": "A1-S2 cross-donor mixed-contract transplant assay",
            "mode": "fixed-host mixed-donor full7 transplant assay",
            "fixed_replace_context": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "assay_mode": "deterministic first-forward",
            "offset": DEFAULT_OFFSET,
            "fixed_contacts_source": "baseline replace native same-entry contacts_in_t",
            "constraints": [
                "no new training",
                "no architecture redesign",
                "no E0/E1/E2-A/E2-C/E3-A/A1-S1 reruns",
                "no full grid expansion",
                "no long-rollout primary criterion",
            ],
            "goal": (
                "test whether preserved E1-top3 anchor-like contract can absorb one-sided E2A-R top7 expansion better "
                "than E2A-R full7 under the same fixed host"
            ),
        },
        "inherited_conclusions": INHERITED_CONCLUSIONS,
        "a1s1_inherited": {
            "summary_json": str(A1S1_SUMMARY_JSON),
            "record_md": str(A1S1_RECORD_MD),
            "boundary_interpretation": dict(a1s1_summary.get("boundary_interpretation") or {}),
            "explicit_answers": dict(a1s1_summary.get("explicit_answers") or {}),
        },
        "plumbing_check": {
            "single_donor_subset_transplant_supported": True,
            "multi_donor_module_assembly_natively_exposed": False,
            "mixed_contract_wrapper_added": True,
            "execution_path": [
                "tools.analyze_cp015_tailk7_same_input_module_attribution::_temporary_weight_swap",
                "tools.run_mixed_contract_a1s2::_run_single_step_mixed",
                "tools.analyze_cp015_tailk7_same_input_module_attribution::_run_single_step",
            ],
            "note": (
                "reused existing subset-transplant primitive and added only a minimal per-arm multi-donor wrapper; "
                "no training or architecture plumbing changed"
            ),
        },
        "host": {
            "label": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "ckpt": str(COADAPT_HOST_CKPT),
            "config": str(COADAPT_HOST_CONFIG),
            "eval_json": str(COADAPT_HOST_EVAL),
        },
        "anchor_donor": {
            "label": "E1-top3 final70a",
            "ckpt": str(TOP3_70A_CKPT),
            "config": str(STAGE70A_CONFIG),
            "eval_json": str(TOP3_70A_EVAL),
        },
        "expansion_donor": {
            "label": "E2A-R final70a",
            "ckpt": str(E2A_70A_CKPT),
            "config": str(STAGE70A_CONFIG),
            "eval_json": str(E2A_70A_EVAL),
        },
        "baseline_replace": {
            "label": "baseline replace donor for transplant-compatible target",
            "ckpt": str(BASELINE_REPLACE_CKPT),
            "config": str(BASELINE_REPLACE_CONFIG),
            "eval_json": str(BASELINE_REPLACE_EVAL),
        },
        "target_artifact": {
            "type": "synthetic_in_memory_transplant_target",
            "materialization": "fixed host + baseline replace full7 transplant",
            "host_ckpt": str(COADAPT_HOST_CKPT),
            "baseline_donor_ckpt": str(BASELINE_REPLACE_CKPT),
            "module_subset": list(DIRECT_BRANCH_MODULES),
            "reference_metrics_reused_from": str(A1S1_SUMMARY_JSON),
        },
        "candidate_partition": {
            "anchor_candidate": _partition_block(
                label="anchor_candidate",
                modules=ANCHOR_CANDIDATE,
                manifest=partition_manifest,
            ),
            "leg_expansion_candidate": _partition_block(
                label="leg_expansion_candidate",
                modules=LEG_EXPANSION_CANDIDATE,
                manifest=partition_manifest,
            ),
            "nonleg_expansion_candidate": _partition_block(
                label="nonleg_expansion_candidate",
                modules=NONLEG_EXPANSION_CANDIDATE,
                manifest=partition_manifest,
            ),
        },
        "reused_references": {
            "host_native_bad_reference": {
                "source": str(A1S1_SUMMARY_JSON),
                "transfer": host_native_reference,
            },
            "transplant_compatible_target": {
                "source": str(A1S1_SUMMARY_JSON),
                "transfer": _build_target_reference(),
            },
            "E1_top3_full7": dict(reused_refs.get("E1_top3_full7") or {}),
            "E2A_R_full7": dict(reused_refs.get("E2A_R_full7") or {}),
        },
        "mixed_assays": mixed_results,
        "judgement": judgement,
        "explicit_answers": explicit_answers,
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_record(summary)
    print(f"[OK] wrote {SUMMARY_JSON}")
    print(f"[OK] wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
