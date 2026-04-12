#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_same_input_module_attribution import (  # noqa: E402
    _case_bundle,
    _prepare_fixed_offset_context,
    _run_single_step,
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
    _safe_float,
)
from tools.run_cp015_tailk_curriculum_e2a import E2A_70A_CKPT, E2A_70A_EVAL  # noqa: E402
from tools.run_cp015_tailk_support_scope_isolation_e1 import (  # noqa: E402
    STAGE70A_CONFIG,
    TOP3_70A_CKPT,
    TOP3_70A_EVAL,
)
from tools.run_mixed_contract_a1s2 import (  # noqa: E402
    CLEAR_AGG_MARGIN,
    INHERITED_CONCLUSIONS,
    SIDE_MARGIN,
    _assay_transfer_mixed,
    _build_target_reference,
    _fmt,
    _load_json,
    _module_manifest_from_ckpt,
    _partition_block,
    _transfer_delta,
)


RUN_DATE = "20260409"
RUN_NAME = "replace_absorb_boundary_a1s3"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_{RUN_NAME}_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"
DOC_PATH = ROOT / "docs" / "train_design" / "2026-04-09_replace_absorb_boundary_a1s3_record.md"

A1S1_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_partial_transplant_boundary_a1s1_20260409" / "summary.json"
A1S1_RECORD_MD = ROOT / "docs" / "train_design" / "2026-04-09_partial_transplant_boundary_a1s1_record.md"
A1S2_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_mixed_contract_a1s2_20260409" / "summary.json"
A1S2_RECORD_MD = ROOT / "docs" / "train_design" / "2026-04-09_mixed_contract_a1s2_record.md"

PRESERVED_ANCHOR_LEG_BLOCK: tuple[str, ...] = (
    "direct_pose_head",
    "direct_pose_leg_head",
    "direct_pose_out_leg",
)
NONLEG_PROJ_CANDIDATE: tuple[str, ...] = (
    "direct_pose_arm_proj",
    "direct_pose_else_proj",
)
NONLEG_OUT_CANDIDATE: tuple[str, ...] = (
    "direct_pose_out_arm",
    "direct_pose_out_else",
)

A1S1_DIRECT_INHERITED: tuple[str, ...] = (
    "A1S1-anchor_only 不比 E2A-R full7 更 replace-transferable",
    "A1-S1 更像 Case 3",
    "更像 shared head 本身 already compromised",
    "partial add-back 里 anchor_plus_nonleg residual retention 明显好于 anchor_plus_leg",
)

A1S2_DIRECT_INHERITED: tuple[str, ...] = (
    "A1S2-mix-nonleg 比 E2A-R full7 更好但不够 clear-win",
    "A1S2-mix-nonleg aggregate 上接近 E1-top3 full7",
    "A1S2-mix-nonleg 同时改善了相对 E2A-R full7 的 dir_nonleg 和 dir_leg",
    "A1S2-mix-leg aggregate 略高，但 dir_leg 更差",
    "top7 nonleg 更像 partially absorbable but not yet decisive",
    "plain mixed transplant 还不足以支持直接进入 replace-side absorb-expansion 已经 solved",
)

TRI_DONOR_ASSAY_SPECS: tuple[Dict[str, Any], ...] = (
    {
        "name": "A1S3-nonleg-proj-donor_host-out",
        "description": (
            "preserve E1-top3 anchor+leg; give nonleg proj to E2A-R; keep nonleg out on fixed host"
        ),
        "e1_modules": list(PRESERVED_ANCHOR_LEG_BLOCK),
        "e2a_modules": list(NONLEG_PROJ_CANDIDATE),
        "host_modules": list(NONLEG_OUT_CANDIDATE),
        "host_absorb_hypothesis": "host nonleg out/readout side may be the absorb boundary",
        "incompatibility_hypothesis": "main incompatibility may sit at downstream nonleg readout contract",
    },
    {
        "name": "A1S3-nonleg-out-donor_host-proj",
        "description": (
            "preserve E1-top3 anchor+leg; give nonleg out to E2A-R; keep nonleg proj on fixed host"
        ),
        "e1_modules": list(PRESERVED_ANCHOR_LEG_BLOCK),
        "e2a_modules": list(NONLEG_OUT_CANDIDATE),
        "host_modules": list(NONLEG_PROJ_CANDIDATE),
        "host_absorb_hypothesis": "host nonleg proj side may be the absorb boundary",
        "incompatibility_hypothesis": "main incompatibility may sit at upstream nonleg proj contract",
    },
)


def _flatten_modules_three_way(spec: Mapping[str, Any]) -> List[str]:
    module_to_donor: Dict[str, str] = {}
    for module in spec.get("e1_modules", []):
        module_to_donor[str(module)] = "E1-top3"
    for module in spec.get("e2a_modules", []):
        module_to_donor[str(module)] = "E2A-R"
    for module in spec.get("host_modules", []):
        module_to_donor[str(module)] = "fixed-host"
    return [str(module) for module in DIRECT_BRANCH_MODULES if str(module) in module_to_donor]


def _tri_donor_assignment_payload(
    *,
    spec: Mapping[str, Any],
    e1_manifest: Mapping[str, Any],
    e2a_manifest: Mapping[str, Any],
    host_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    donor_groups: Dict[str, Dict[str, Any]] = {
        "E1-top3": {
            "assignment_mode": "swap_from_donor",
            "ckpt": str(TOP3_70A_CKPT),
            "eval_json": str(TOP3_70A_EVAL),
            "modules": [],
            "parameter_prefixes": [],
            "effective_key_count_total": 0,
            "copied_key_count_total": 0,
            "retained_key_count_total": 0,
            "parameter_count_total": 0,
            "effective_tensor_keys": [],
            "copied_tensor_keys": [],
            "retained_tensor_keys": [],
            "per_module": [],
        },
        "E2A-R": {
            "assignment_mode": "swap_from_donor",
            "ckpt": str(E2A_70A_CKPT),
            "eval_json": str(E2A_70A_EVAL),
            "modules": [],
            "parameter_prefixes": [],
            "effective_key_count_total": 0,
            "copied_key_count_total": 0,
            "retained_key_count_total": 0,
            "parameter_count_total": 0,
            "effective_tensor_keys": [],
            "copied_tensor_keys": [],
            "retained_tensor_keys": [],
            "per_module": [],
        },
        "fixed-host": {
            "assignment_mode": "retained_in_host",
            "ckpt": str(COADAPT_HOST_CKPT),
            "eval_json": str(COADAPT_HOST_EVAL),
            "modules": [],
            "parameter_prefixes": [],
            "effective_key_count_total": 0,
            "copied_key_count_total": 0,
            "retained_key_count_total": 0,
            "parameter_count_total": 0,
            "effective_tensor_keys": [],
            "copied_tensor_keys": [],
            "retained_tensor_keys": [],
            "per_module": [],
        },
    }

    per_module: List[Dict[str, Any]] = []
    for donor_label, modules, manifest in (
        ("E1-top3", spec.get("e1_modules", []), e1_manifest),
        ("E2A-R", spec.get("e2a_modules", []), e2a_manifest),
        ("fixed-host", spec.get("host_modules", []), host_manifest),
    ):
        assignment_mode = donor_groups[donor_label]["assignment_mode"]
        for module in modules:
            base_row = dict(manifest[str(module)])
            state_tensor_keys = [str(key) for key in base_row["copied_tensor_keys"]]
            state_key_count = int(base_row["copied_key_count"])
            parameter_count = int(base_row["parameter_count"])
            row = {
                "module": str(base_row["module"]),
                "parameter_prefix": str(base_row["parameter_prefix"]),
                "donor_source": donor_label,
                "assignment_mode": assignment_mode,
                "state_tensor_keys": state_tensor_keys,
                "state_key_count": state_key_count,
                "parameter_count": parameter_count,
                "copied_tensor_keys": state_tensor_keys if donor_label != "fixed-host" else [],
                "copied_key_count": state_key_count if donor_label != "fixed-host" else 0,
                "retained_tensor_keys": state_tensor_keys if donor_label == "fixed-host" else [],
                "retained_key_count": state_key_count if donor_label == "fixed-host" else 0,
            }
            per_module.append(row)
            donor_groups[donor_label]["modules"].append(str(module))
            donor_groups[donor_label]["parameter_prefixes"].append(str(row["parameter_prefix"]))
            donor_groups[donor_label]["effective_key_count_total"] += state_key_count
            donor_groups[donor_label]["copied_key_count_total"] += int(row["copied_key_count"])
            donor_groups[donor_label]["retained_key_count_total"] += int(row["retained_key_count"])
            donor_groups[donor_label]["parameter_count_total"] += parameter_count
            donor_groups[donor_label]["effective_tensor_keys"].extend(state_tensor_keys)
            donor_groups[donor_label]["copied_tensor_keys"].extend([str(key) for key in row["copied_tensor_keys"]])
            donor_groups[donor_label]["retained_tensor_keys"].extend([str(key) for key in row["retained_tensor_keys"]])
            donor_groups[donor_label]["per_module"].append(row)

    ordered_modules = _flatten_modules_three_way(spec)
    ordered_per_module = [next(row for row in per_module if row["module"] == module) for module in ordered_modules]
    return {
        "module_to_donor_source": {str(row["module"]): str(row["donor_source"]) for row in ordered_per_module},
        "modules": ordered_modules,
        "parameter_prefixes": [str(row["parameter_prefix"]) for row in ordered_per_module],
        "assigned_key_count_total": int(sum(int(row["state_key_count"]) for row in ordered_per_module)),
        "copied_key_count_total": int(sum(int(row["copied_key_count"]) for row in ordered_per_module)),
        "retained_key_count_total": int(sum(int(row["retained_key_count"]) for row in ordered_per_module)),
        "parameter_count_total": int(sum(int(row["parameter_count"]) for row in ordered_per_module)),
        "per_module": ordered_per_module,
        "by_donor_source": donor_groups,
    }


def _judge_results(
    *,
    proj_host_out: Mapping[str, Any],
    out_host_proj: Mapping[str, Any],
    mix_nonleg: Mapping[str, Any],
    mix_leg: Mapping[str, Any],
    e1_top3: Mapping[str, Any],
    e2a_full7: Mapping[str, Any],
) -> Dict[str, Any]:
    proj_agg = _safe_float(proj_host_out.get("aggregate_transfer_score"))
    out_agg = _safe_float(out_host_proj.get("aggregate_transfer_score"))
    mix_nonleg_agg = _safe_float(mix_nonleg.get("aggregate_transfer_score"))
    mix_leg_agg = _safe_float(mix_leg.get("aggregate_transfer_score"))
    e1_agg = _safe_float(e1_top3.get("aggregate_transfer_score"))
    e2a_agg = _safe_float(e2a_full7.get("aggregate_transfer_score"))

    proj_vs_mix = proj_agg - mix_nonleg_agg
    out_vs_mix = out_agg - mix_nonleg_agg
    proj_vs_out = proj_agg - out_agg
    proj_vs_e2a = proj_agg - e2a_agg
    out_vs_e2a = out_agg - e2a_agg
    proj_vs_e1 = proj_agg - e1_agg
    out_vs_e1 = out_agg - e1_agg

    proj_clear = proj_vs_mix > CLEAR_AGG_MARGIN
    out_clear = out_vs_mix > CLEAR_AGG_MARGIN
    side_pref_proj = proj_vs_out > SIDE_MARGIN
    side_pref_out = (-proj_vs_out) > SIDE_MARGIN

    if proj_clear and not out_clear:
        case_label = "Case A"
        absorb_boundary_side = "host nonleg out side"
        incompatibility_boundary = "downstream nonleg readout contract"
        recommend_absorb_design = True
        next_step = "preserve_E1_anchor_leg_then_expand_host_nonleg_out_absorb_side"
        interpretation = (
            "keeping host nonleg out while swapping only E2A-R nonleg proj is the only clear step beyond A1S2-mix-nonleg"
        )
    elif out_clear and not proj_clear:
        case_label = "Case B"
        absorb_boundary_side = "host nonleg proj side"
        incompatibility_boundary = "upstream nonleg proj contract"
        recommend_absorb_design = True
        next_step = "preserve_E1_anchor_leg_then_expand_host_nonleg_proj_absorb_side"
        interpretation = (
            "keeping host nonleg proj while swapping only E2A-R nonleg out is the only clear step beyond A1S2-mix-nonleg"
        )
    elif proj_clear and out_clear:
        case_label = "Case D"
        if side_pref_proj:
            absorb_boundary_side = "host nonleg out side"
            incompatibility_boundary = "downstream nonleg readout contract"
        elif side_pref_out:
            absorb_boundary_side = "host nonleg proj side"
            incompatibility_boundary = "upstream nonleg proj contract"
        else:
            proj_leg = _safe_float(proj_host_out.get("dir_leg_closure_ratio"))
            out_leg = _safe_float(out_host_proj.get("dir_leg_closure_ratio"))
            if proj_leg >= out_leg:
                absorb_boundary_side = "host nonleg out side"
                incompatibility_boundary = "downstream nonleg readout contract"
            else:
                absorb_boundary_side = "host nonleg proj side"
                incompatibility_boundary = "upstream nonleg proj contract"
        recommend_absorb_design = True
        next_step = "enter_replace_side_absorb_expansion_design_but_choose_cleaner_A1S3_side"
        interpretation = (
            "both split arms improve clearly over A1S2-mix-nonleg, so replace-side nonleg block looks decomposably absorbable"
        )
    else:
        case_label = "Case C"
        recommend_absorb_design = False
        next_step = "shrink_back_to_earlier_boundary_or_stronger_boundary_guard"
        interpretation = (
            "neither split arm clears the explicit improvement margin over A1S2-mix-nonleg, so plain replace-side splitting is still not decisive"
        )
        if side_pref_proj:
            absorb_boundary_side = "no_clear_winner; weak lean host nonleg out side"
            incompatibility_boundary = "no_clear_single_boundary; weak lean downstream nonleg readout contract"
        elif side_pref_out:
            absorb_boundary_side = "no_clear_winner; weak lean host nonleg proj side"
            incompatibility_boundary = "no_clear_single_boundary; weak lean upstream nonleg proj contract"
        else:
            absorb_boundary_side = "no_clear_winner"
            incompatibility_boundary = "no_clear_single_boundary"

    return {
        "case_label": case_label,
        "A1S3_nonleg_proj_donor_host_out_minus_A1S2_mix_nonleg_aggregate": float(proj_vs_mix),
        "A1S3_nonleg_out_donor_host_proj_minus_A1S2_mix_nonleg_aggregate": float(out_vs_mix),
        "A1S3_nonleg_proj_donor_host_out_minus_A1S2_mix_leg_aggregate": float(proj_agg - mix_leg_agg),
        "A1S3_nonleg_out_donor_host_proj_minus_A1S2_mix_leg_aggregate": float(out_agg - mix_leg_agg),
        "A1S3_nonleg_proj_donor_host_out_minus_E2A_R_full7_aggregate": float(proj_vs_e2a),
        "A1S3_nonleg_out_donor_host_proj_minus_E2A_R_full7_aggregate": float(out_vs_e2a),
        "A1S3_nonleg_proj_donor_host_out_minus_E1_top3_full7_aggregate": float(proj_vs_e1),
        "A1S3_nonleg_out_donor_host_proj_minus_E1_top3_full7_aggregate": float(out_vs_e1),
        "A1S3_nonleg_proj_donor_host_out_minus_A1S2_mix_nonleg_dir_leg_closure": float(
            _safe_float(proj_host_out.get("dir_leg_closure_ratio"))
            - _safe_float(mix_nonleg.get("dir_leg_closure_ratio"))
        ),
        "A1S3_nonleg_out_donor_host_proj_minus_A1S2_mix_nonleg_dir_leg_closure": float(
            _safe_float(out_host_proj.get("dir_leg_closure_ratio"))
            - _safe_float(mix_nonleg.get("dir_leg_closure_ratio"))
        ),
        "A1S3_nonleg_proj_donor_host_out_minus_A1S2_mix_nonleg_dir_nonleg_closure": float(
            _safe_float(proj_host_out.get("dir_nonleg_closure_ratio"))
            - _safe_float(mix_nonleg.get("dir_nonleg_closure_ratio"))
        ),
        "A1S3_nonleg_out_donor_host_proj_minus_A1S2_mix_nonleg_dir_nonleg_closure": float(
            _safe_float(out_host_proj.get("dir_nonleg_closure_ratio"))
            - _safe_float(mix_nonleg.get("dir_nonleg_closure_ratio"))
        ),
        "A1S3_nonleg_proj_donor_host_out_minus_A1S2_mix_nonleg_dir_base_closure": float(
            _safe_float(proj_host_out.get("dir_base_closure_ratio"))
            - _safe_float(mix_nonleg.get("dir_base_closure_ratio"))
        ),
        "A1S3_nonleg_out_donor_host_proj_minus_A1S2_mix_nonleg_dir_base_closure": float(
            _safe_float(out_host_proj.get("dir_base_closure_ratio"))
            - _safe_float(mix_nonleg.get("dir_base_closure_ratio"))
        ),
        "proj_clear_win_over_mix_nonleg": bool(proj_clear),
        "out_clear_win_over_mix_nonleg": bool(out_clear),
        "split_side_preference": (
            "A1S3-nonleg-proj-donor_host-out"
            if side_pref_proj
            else (
                "A1S3-nonleg-out-donor_host-proj"
                if side_pref_out
                else "no_clear_split_side_preference"
            )
        ),
        "absorb_boundary_side": absorb_boundary_side,
        "main_incompatibility_boundary": incompatibility_boundary,
        "recommend_replace_side_absorb_expansion_design": bool(recommend_absorb_design),
        "next_step": next_step,
        "interpretation": interpretation,
    }


def _write_record(summary: Mapping[str, Any]) -> None:
    refs = summary["reused_references"]
    arms = summary["tri_donor_assays"]
    judgement = summary["judgement"]

    proj_arm = arms["A1S3-nonleg-proj-donor_host-out"]["transfer"]
    out_arm = arms["A1S3-nonleg-out-donor_host-proj"]["transfer"]
    mix_nonleg = refs["A1S2_mix_nonleg"]["transfer"]
    mix_leg = refs["A1S2_mix_leg"]["transfer"]
    e1 = refs["E1_top3_full7"]["transfer"]
    e2a = refs["E2A_R_full7"]["transfer"]

    lines: List[str] = []
    lines.append("# 2026-04-09 replace absorb boundary A1-S3 record")
    lines.append("")
    lines.append("> Last updated: 2026-04-09  ")
    lines.append("> Scope: A1-S3 only / fixed-host replace-side nonleg absorb boundary assay / tri-donor / no new training")
    lines.append("")
    lines.append("## 1. Scope / inherited conclusions")
    lines.append("")
    lines.append("本轮只做 **fixed host 下的 replace-side nonleg absorb boundary assay**，直接继承以下结论，不重复证明：")
    lines.append("")
    for item in summary["inherited_conclusions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("直接继承 A1-S1：")
    lines.append(f"- A1-S1 summary: `{summary['a1s1_inherited']['summary_json']}`")
    lines.append(f"- A1-S1 record: `{summary['a1s1_inherited']['record_md']}`")
    for item in summary["a1s1_inherited"]["direct_conclusions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("直接继承 A1-S2：")
    lines.append(f"- A1-S2 summary: `{summary['a1s2_inherited']['summary_json']}`")
    lines.append(f"- A1-S2 record: `{summary['a1s2_inherited']['record_md']}`")
    for item in summary["a1s2_inherited"]["direct_conclusions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 2. Why A1-S3 after A1-S2")
    lines.append("")
    lines.append("- A1-S2 已经把 preserved-anchor mixed transplant 推到 `A1S2-mix-nonleg`，但仍未 clear-win。")
    lines.append("- 因此 A1-S3 的唯一目标，是继续把 nonleg block 只拆成 `proj side` / `out side` 两个 code-level assay split，观察 fixed host 哪一侧更像 absorb boundary。")
    lines.append("- 本轮不做 full grid，不开训练，也不把 candidate partition 写成已证实真相。")
    lines.append("")
    lines.append("## 3. Donor / host / target inventory")
    lines.append("")
    lines.append("| item | artifact | path / note |")
    lines.append("|---|---|---|")
    lines.append(f"| host ckpt | fixed host | `{summary['host']['ckpt']}` |")
    lines.append(f"| host config | fixed host config | `{summary['host']['config']}` |")
    lines.append(f"| anchor donor ckpt | E1-top3 final70a | `{summary['anchor_donor']['ckpt']}` |")
    lines.append(f"| anchor donor eval | E1-top3 eval | `{summary['anchor_donor']['eval_json']}` |")
    lines.append(f"| expansion donor ckpt | E2A-R final70a | `{summary['expansion_donor']['ckpt']}` |")
    lines.append(f"| expansion donor eval | E2A-R eval | `{summary['expansion_donor']['eval_json']}` |")
    lines.append(f"| baseline replace ckpt | transplant target donor | `{summary['baseline_replace']['ckpt']}` |")
    lines.append("| target | transplant-compatible target | in-memory only: fixed host + baseline replace full7 transplant |")
    lines.append("")
    lines.append("## 4. Candidate partition reminder")
    lines.append("")
    lines.append("下述 partition **仍然只是 hypothesis**，仅用于 A1-S3 assay inventory：")
    lines.append("")
    lines.append("| family | modules | parameter prefixes | note |")
    lines.append("|---|---|---|---|")
    for key in ("preserved_anchor_leg_block", "nonleg_proj_candidate", "nonleg_out_candidate"):
        row = summary["candidate_partition"][key]
        lines.append(
            f"| `{key}` | `{', '.join(row['modules'])}` | `{', '.join(row['parameter_prefixes'])}` | code-level assay hypothesis only |"
        )
    lines.append("")
    lines.append("## 5. Tri-donor assay inventory table")
    lines.append("")
    lines.append("| arm | E1-top3 modules | E2A-R modules | fixed-host retained modules | copied key counts |")
    lines.append("|---|---|---|---|---:|")
    for name, row in arms.items():
        by_donor = row["tri_donor_contract"]["by_donor_source"]
        lines.append(
            f"| `{name}` | `{', '.join(by_donor['E1-top3']['modules'])}` | `{', '.join(by_donor['E2A-R']['modules'])}` | "
            f"`{', '.join(by_donor['fixed-host']['modules'])}` | {row['tri_donor_contract']['copied_key_count_total']} |"
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
        ("A1S3-nonleg-proj-donor_host-out", proj_arm),
        ("A1S3-nonleg-out-donor_host-proj", out_arm),
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
        f"- `A1S3-nonleg-proj-donor_host-out` 的 `dir_leg` closure = `{_fmt(proj_arm['dir_leg_closure_ratio'])}`；相对 `A1S2-mix-nonleg` delta = `{_fmt(judgement['A1S3_nonleg_proj_donor_host_out_minus_A1S2_mix_nonleg_dir_leg_closure'])}`。"
    )
    lines.append(
        f"- `A1S3-nonleg-out-donor_host-proj` 的 `dir_leg` closure = `{_fmt(out_arm['dir_leg_closure_ratio'])}`；相对 `A1S2-mix-nonleg` delta = `{_fmt(judgement['A1S3_nonleg_out_donor_host_proj_minus_A1S2_mix_nonleg_dir_leg_closure'])}`。"
    )
    lines.append(
        f"- 两个 A1-S3 arms 的 `dir_leg` closure 差值（proj-host-out 减 out-host-proj）= `{_fmt(_safe_float(proj_arm['dir_leg_closure_ratio']) - _safe_float(out_arm['dir_leg_closure_ratio']))}`。"
    )
    lines.append("")
    lines.append("## 8. `dir_nonleg` retention interpretation")
    lines.append("")
    lines.append(
        f"- `A1S3-nonleg-proj-donor_host-out` 的 `dir_nonleg` closure = `{_fmt(proj_arm['dir_nonleg_closure_ratio'])}`；相对 `A1S2-mix-nonleg` delta = `{_fmt(judgement['A1S3_nonleg_proj_donor_host_out_minus_A1S2_mix_nonleg_dir_nonleg_closure'])}`。"
    )
    lines.append(
        f"- `A1S3-nonleg-out-donor_host-proj` 的 `dir_nonleg` closure = `{_fmt(out_arm['dir_nonleg_closure_ratio'])}`；相对 `A1S2-mix-nonleg` delta = `{_fmt(judgement['A1S3_nonleg_out_donor_host_proj_minus_A1S2_mix_nonleg_dir_nonleg_closure'])}`。"
    )
    lines.append(
        f"- 两个 A1-S3 arms 的 `dir_nonleg` closure 差值（proj-host-out 减 out-host-proj）= `{_fmt(_safe_float(proj_arm['dir_nonleg_closure_ratio']) - _safe_float(out_arm['dir_nonleg_closure_ratio']))}`。"
    )
    lines.append("")
    lines.append("## 9. Replace-side absorb boundary interpretation")
    lines.append("")
    lines.append(f"- A1-S3 判例：**{judgement['case_label']}**")
    lines.append(
        f"- `A1S3-nonleg-proj-donor_host-out` vs `A1S2-mix-nonleg` aggregate delta = `{_fmt(judgement['A1S3_nonleg_proj_donor_host_out_minus_A1S2_mix_nonleg_aggregate'])}`。"
    )
    lines.append(
        f"- `A1S3-nonleg-out-donor_host-proj` vs `A1S2-mix-nonleg` aggregate delta = `{_fmt(judgement['A1S3_nonleg_out_donor_host_proj_minus_A1S2_mix_nonleg_aggregate'])}`。"
    )
    lines.append(
        f"- host absorb boundary call：**{judgement['absorb_boundary_side']}**。"
    )
    lines.append(
        f"- main incompatibility boundary call：**{judgement['main_incompatibility_boundary']}**。"
    )
    lines.append(f"- 解释：{judgement['interpretation']}")
    lines.append("")
    lines.append("## 10. Next-step recommendation")
    lines.append("")
    lines.append(
        f"- 是否支持进入更明确的 replace-side absorb-expansion design：**{'yes' if judgement['recommend_replace_side_absorb_expansion_design'] else 'no'}**"
    )
    lines.append(f"- 推荐主线：**{judgement['next_step']}**")
    lines.append("")
    lines.append("## Final answers")
    lines.append("")
    lines.append(
        f"- `A1S3-nonleg-proj-donor_host-out` 是否明显优于 `A1S2-mix-nonleg`：`{summary['explicit_answers']['q1_A1S3_nonleg_proj_donor_host_out_clearly_better_than_A1S2_mix_nonleg']['answer']}`"
    )
    lines.append(
        f"- `A1S3-nonleg-out-donor_host-proj` 是否明显优于 `A1S2-mix-nonleg`：`{summary['explicit_answers']['q2_A1S3_nonleg_out_donor_host_proj_clearly_better_than_A1S2_mix_nonleg']['answer']}`"
    )
    lines.append(
        f"- host absorb capacity 更像落在哪一侧：`{summary['explicit_answers']['q3_host_absorb_capacity_side']['answer']}`"
    )
    lines.append(
        f"- 是否支持进入更明确的 replace-side absorb-expansion design：`{summary['explicit_answers']['q4_support_enter_replace_side_absorb_expansion_design']['answer']}`"
    )
    lines.append(
        f"- 是否仍应先转向更早 boundary / stronger boundary guard：`{summary['explicit_answers']['q5_should_turn_back_to_earlier_boundary_or_stronger_guard']['answer']}`"
    )
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
        A1S2_SUMMARY_JSON,
        A1S2_RECORD_MD,
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise SystemExit("[FATAL] missing required artifact(s):\n" + "\n".join(missing))

    a1s1_summary = _load_json(A1S1_SUMMARY_JSON)
    a1s2_summary = _load_json(A1S2_SUMMARY_JSON)

    reused_refs = dict(a1s2_summary.get("reused_references") or {})
    host_native_reference = dict((reused_refs.get("host_native_bad_reference") or {}).get("transfer") or {})
    e1_top3_reference = dict((reused_refs.get("E1_top3_full7") or {}).get("transfer") or {})
    e2a_full7_reference = dict((reused_refs.get("E2A_R_full7") or {}).get("transfer") or {})
    mix_nonleg_reference = dict(
        ((a1s2_summary.get("mixed_assays") or {}).get("A1S2-mix-nonleg") or {}).get("transfer") or {}
    )
    mix_leg_reference = dict(
        ((a1s2_summary.get("mixed_assays") or {}).get("A1S2-mix-leg") or {}).get("transfer") or {}
    )

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

    host_manifest = _module_manifest_from_ckpt(COADAPT_HOST_CKPT, DIRECT_BRANCH_MODULES)
    e1_manifest = _module_manifest_from_ckpt(TOP3_70A_CKPT, DIRECT_BRANCH_MODULES)
    e2a_manifest = _module_manifest_from_ckpt(E2A_70A_CKPT, DIRECT_BRANCH_MODULES)

    tri_donor_results: Dict[str, Any] = {}
    for spec in TRI_DONOR_ASSAY_SPECS:
        tri_donor_contract = _tri_donor_assignment_payload(
            spec=spec,
            e1_manifest=e1_manifest,
            e2a_manifest=e2a_manifest,
            host_manifest=host_manifest,
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
        tri_donor_results[str(spec["name"])] = {
            "description": str(spec["description"]),
            "host_absorb_hypothesis": str(spec["host_absorb_hypothesis"]),
            "incompatibility_hypothesis": str(spec["incompatibility_hypothesis"]),
            "tri_donor_contract": tri_donor_contract,
            "transfer": transfer,
            "delta_vs_A1S2_mix_nonleg": _transfer_delta(transfer, mix_nonleg_reference),
            "delta_vs_A1S2_mix_leg": _transfer_delta(transfer, mix_leg_reference),
            "delta_vs_E2A_R_full7": _transfer_delta(transfer, e2a_full7_reference),
            "delta_vs_E1_top3_full7": _transfer_delta(transfer, e1_top3_reference),
        }

    judgement = _judge_results(
        proj_host_out=tri_donor_results["A1S3-nonleg-proj-donor_host-out"]["transfer"],
        out_host_proj=tri_donor_results["A1S3-nonleg-out-donor_host-proj"]["transfer"],
        mix_nonleg=mix_nonleg_reference,
        mix_leg=mix_leg_reference,
        e1_top3=e1_top3_reference,
        e2a_full7=e2a_full7_reference,
    )

    explicit_answers = {
        "q1_A1S3_nonleg_proj_donor_host_out_clearly_better_than_A1S2_mix_nonleg": {
            "answer": "yes" if bool(judgement["proj_clear_win_over_mix_nonleg"]) else "no",
            "aggregate_delta": float(judgement["A1S3_nonleg_proj_donor_host_out_minus_A1S2_mix_nonleg_aggregate"]),
            "dir_leg_closure_delta": float(judgement["A1S3_nonleg_proj_donor_host_out_minus_A1S2_mix_nonleg_dir_leg_closure"]),
            "dir_nonleg_closure_delta": float(
                judgement["A1S3_nonleg_proj_donor_host_out_minus_A1S2_mix_nonleg_dir_nonleg_closure"]
            ),
        },
        "q2_A1S3_nonleg_out_donor_host_proj_clearly_better_than_A1S2_mix_nonleg": {
            "answer": "yes" if bool(judgement["out_clear_win_over_mix_nonleg"]) else "no",
            "aggregate_delta": float(judgement["A1S3_nonleg_out_donor_host_proj_minus_A1S2_mix_nonleg_aggregate"]),
            "dir_leg_closure_delta": float(judgement["A1S3_nonleg_out_donor_host_proj_minus_A1S2_mix_nonleg_dir_leg_closure"]),
            "dir_nonleg_closure_delta": float(
                judgement["A1S3_nonleg_out_donor_host_proj_minus_A1S2_mix_nonleg_dir_nonleg_closure"]
            ),
        },
        "q3_host_absorb_capacity_side": {
            "answer": str(judgement["absorb_boundary_side"]),
            "main_incompatibility_boundary": str(judgement["main_incompatibility_boundary"]),
            "split_side_preference": str(judgement["split_side_preference"]),
        },
        "q4_support_enter_replace_side_absorb_expansion_design": {
            "answer": "yes" if bool(judgement["recommend_replace_side_absorb_expansion_design"]) else "no",
            "next_step": str(judgement["next_step"]),
        },
        "q5_should_turn_back_to_earlier_boundary_or_stronger_guard": {
            "answer": "yes" if not bool(judgement["recommend_replace_side_absorb_expansion_design"]) else "no",
            "next_step": str(judgement["next_step"]),
        },
    }

    summary = {
        "analysis": RUN_NAME,
        "scope": {
            "experiment": "A1-S3 replace-side nonleg absorb boundary assay",
            "mode": "fixed-host tri-donor mixed-contract boundary assay",
            "fixed_replace_context": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "assay_mode": "deterministic first-forward",
            "offset": DEFAULT_OFFSET,
            "fixed_contacts_source": "baseline replace native same-entry contacts_in_t",
            "constraints": [
                "no new training",
                "no architecture redesign",
                "no E0/E1/E2-A/E2-C/E3-A/A1-S1/A1-S2 reruns",
                "no full grid expansion",
                "no long-rollout primary criterion",
            ],
            "goal": (
                "test whether fixed-host absorb capacity is more compatible on nonleg proj side or nonleg out side "
                "while preserving the E1-top3 anchor+leg contract"
            ),
        },
        "inherited_conclusions": INHERITED_CONCLUSIONS,
        "a1s1_inherited": {
            "summary_json": str(A1S1_SUMMARY_JSON),
            "record_md": str(A1S1_RECORD_MD),
            "direct_conclusions": list(A1S1_DIRECT_INHERITED),
            "boundary_interpretation": dict(a1s1_summary.get("boundary_interpretation") or {}),
            "explicit_answers": dict(a1s1_summary.get("explicit_answers") or {}),
        },
        "a1s2_inherited": {
            "summary_json": str(A1S2_SUMMARY_JSON),
            "record_md": str(A1S2_RECORD_MD),
            "direct_conclusions": list(A1S2_DIRECT_INHERITED),
            "judgement": dict(a1s2_summary.get("judgement") or {}),
        },
        "plumbing_check": {
            "single_donor_subset_transplant_supported": True,
            "multi_donor_module_assembly_natively_exposed": False,
            "a1s2_mixed_wrapper_reused": True,
            "tri_donor_contract_manifest_added": True,
            "execution_path": [
                "tools.run_mixed_contract_a1s2::_assay_transfer_mixed",
                "tools.run_replace_absorb_boundary_a1s3::_tri_donor_assignment_payload",
                "tools.analyze_cp015_tailk7_same_input_module_attribution::_run_single_step",
            ],
            "note": (
                "reused the A1-S2 repeated weight-swap runner unchanged; A1-S3 only adds a tri-donor manifest layer "
                "so retained fixed-host modules are explicit in the assay record"
            ),
            "reference_metrics_reused_from": [
                str(A1S1_SUMMARY_JSON),
                str(A1S2_SUMMARY_JSON),
            ],
            "target_tensor_materialization_note": (
                "baseline-native contacts and full7 target tensors were materialized in-memory only to score the new A1-S3 arms; "
                "fixed reference metrics themselves were loaded from prior summaries rather than recomputed as report rows"
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
            "reference_metrics_reused_from": str(A1S2_SUMMARY_JSON),
        },
        "candidate_partition": {
            "preserved_anchor_leg_block": _partition_block(
                label="preserved_anchor_leg_block",
                modules=PRESERVED_ANCHOR_LEG_BLOCK,
                manifest=host_manifest,
            ),
            "nonleg_proj_candidate": _partition_block(
                label="nonleg_proj_candidate",
                modules=NONLEG_PROJ_CANDIDATE,
                manifest=host_manifest,
            ),
            "nonleg_out_candidate": _partition_block(
                label="nonleg_out_candidate",
                modules=NONLEG_OUT_CANDIDATE,
                manifest=host_manifest,
            ),
        },
        "reused_references": {
            "host_native_bad_reference": {
                "source": str(A1S2_SUMMARY_JSON),
                "transfer": host_native_reference,
            },
            "transplant_compatible_target": {
                "source": str(A1S2_SUMMARY_JSON),
                "transfer": _build_target_reference(),
            },
            "E1_top3_full7": dict(reused_refs.get("E1_top3_full7") or {}),
            "E2A_R_full7": dict(reused_refs.get("E2A_R_full7") or {}),
            "A1S2_mix_nonleg": {
                "source": str(A1S2_SUMMARY_JSON),
                "transfer": mix_nonleg_reference,
            },
            "A1S2_mix_leg": {
                "source": str(A1S2_SUMMARY_JSON),
                "transfer": mix_leg_reference,
            },
        },
        "tri_donor_assays": tri_donor_results,
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
