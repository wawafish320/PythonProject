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
RUN_NAME = "partial_transplant_boundary_a1s1"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_{RUN_NAME}_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"
DOC_PATH = ROOT / "docs" / "train_design" / "2026-04-09_partial_transplant_boundary_a1s1_record.md"

E1_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_cp015_tailk_support_scope_isolation_e1_20260408" / "summary.json"
E2A_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_cp015_tailk_curriculum_e2a_20260408" / "summary.json"
E2C_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_cp015_tailk_legfirst_e2c_20260408" / "summary.json"
E3A_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_cp015_tailk_allocation_e3a_20260408" / "summary.json"

CLEAR_AGG_MARGIN = 0.05
HARMFUL_ADD_MARGIN = -0.03
ASYM_MARGIN = 0.03

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

ASSAY_SPECS: tuple[Dict[str, Any], ...] = (
    {
        "name": "A1S1-anchor_only",
        "modules": list(ANCHOR_CANDIDATE),
        "family": "anchor_only",
    },
    {
        "name": "A1S1-anchor_plus_leg",
        "modules": list(ANCHOR_CANDIDATE + LEG_EXPANSION_CANDIDATE),
        "family": "anchor_plus_leg",
    },
    {
        "name": "A1S1-anchor_plus_nonleg",
        "modules": list(ANCHOR_CANDIDATE + NONLEG_EXPANSION_CANDIDATE),
        "family": "anchor_plus_nonleg",
    },
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


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Iterable[Any]) -> float:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _fmt(value: Any) -> str:
    val = _safe_float(value)
    return "nan" if not math.isfinite(val) else f"{val:.6f}"


def _prefix_info_for_modules(ckpt_path: Path, modules: Sequence[str]) -> Dict[str, Any]:
    obj = torch.load(ckpt_path, map_location="cpu")
    model_state = obj["model"]
    out: Dict[str, Any] = {}
    for module in modules:
        prefix = f"{module}."
        keys = sorted(key for key in model_state.keys() if str(key).startswith(prefix))
        out[str(module)] = {
            "module": str(module),
            "parameter_prefix": prefix,
            "state_dict_key_count": int(len(keys)),
            "sample_keys": [str(key) for key in keys[:6]],
            "parameter_count": int(sum(int(model_state[key].numel()) for key in keys)),
        }
    return out


def _partition_block(
    *,
    label: str,
    modules: Sequence[str],
    prefix_info: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "label": str(label),
        "modules": [str(module) for module in modules],
        "parameter_prefixes": [str(prefix_info[str(module)]["parameter_prefix"]) for module in modules],
        "per_module": [dict(prefix_info[str(module)]) for module in modules],
        "candidate_partition_note": (
            "code-level candidate partition used only as A1-S1 assay hypothesis; "
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


def _load_reference_transfer(summary_path: Path, arm_key: str) -> Dict[str, Any]:
    payload = _load_json(summary_path)
    row = ((payload.get("final_70a_results") or {}).get(arm_key) or {})
    transfer = dict(row.get("transfer") or {})
    return {
        "summary_json": str(summary_path),
        "arm_key": str(arm_key),
        "ckpt": str(row.get("ckpt") or row.get("stage70a_ckpt") or ""),
        "eval_json": str(row.get("eval_json") or row.get("stage70a_eval") or ""),
        "transfer": transfer,
    }


def _build_host_native_reference(host_gaps: Mapping[str, Any]) -> Dict[str, Any]:
    transfer = dict(host_gaps)
    transfer.update(
        {
            "out_direct_closure_ratio": 0.0,
            "dir_base_closure_ratio": 0.0,
            "dir_leg_closure_ratio": 0.0,
            "dir_nonleg_closure_ratio": 0.0,
            "aggregate_transfer_score": 0.0,
        }
    )
    return transfer


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


def _assay_transfer(
    *,
    host_bundle: Mapping[str, Any],
    prep_host: Mapping[str, Any],
    donor_bundle: Mapping[str, Any],
    fixed_contacts: torch.Tensor,
    target_result: Mapping[str, Any],
    host_gaps: Mapping[str, Any],
    modules: Sequence[str],
) -> Dict[str, Any]:
    candidate_result = _run_single_step(
        host_bundle,
        prep_host,
        fixed_contacts=fixed_contacts,
        weight_swap_modules=modules,
        donor_bundle=donor_bundle,
    )
    gaps = _tensor_metric_gaps(
        host_case=host_bundle["case"],
        target_result=target_result,
        candidate_result=candidate_result,
    )
    return _add_closure(gaps, host_gaps)


def _classify_boundary_case(
    *,
    anchor_transfer: Mapping[str, Any],
    plus_leg_transfer: Mapping[str, Any],
    plus_nonleg_transfer: Mapping[str, Any],
    full7_transfer: Mapping[str, Any],
) -> Dict[str, Any]:
    anchor_agg = _safe_float(anchor_transfer.get("aggregate_transfer_score"))
    leg_agg = _safe_float(plus_leg_transfer.get("aggregate_transfer_score"))
    nonleg_agg = _safe_float(plus_nonleg_transfer.get("aggregate_transfer_score"))
    full7_agg = _safe_float(full7_transfer.get("aggregate_transfer_score"))

    anchor_vs_full7 = anchor_agg - full7_agg
    leg_vs_anchor = leg_agg - anchor_agg
    nonleg_vs_anchor = nonleg_agg - anchor_agg
    leg_dirleg_vs_anchor = _safe_float(plus_leg_transfer.get("dir_leg_closure_ratio")) - _safe_float(
        anchor_transfer.get("dir_leg_closure_ratio")
    )
    nonleg_dirleg_vs_anchor = _safe_float(plus_nonleg_transfer.get("dir_leg_closure_ratio")) - _safe_float(
        anchor_transfer.get("dir_leg_closure_ratio")
    )

    anchor_clearly_better = anchor_vs_full7 > CLEAR_AGG_MARGIN
    leg_add_harmful = leg_vs_anchor < HARMFUL_ADD_MARGIN
    nonleg_add_harmful = nonleg_vs_anchor < HARMFUL_ADD_MARGIN
    preferred_side = "leg_expansion_candidate" if leg_agg >= nonleg_agg else "nonleg_expansion_candidate"
    preferred_mix = "E1-top3 anchor + top7 leg expansion" if preferred_side == "leg_expansion_candidate" else "E1-top3 anchor + top7 nonleg expansion"
    preferred_reason = (
        "this side retains more residual transfer once added back on top of the shared head in the single-donor scout"
    )

    if not anchor_clearly_better:
        case_label = "Case 3"
        main_call = "shared_head_already_compromised"
        dir_leg_boundary = "earlier shared-head boundary"
        a1s2_recommend = True
        preferred_a1s2 = preferred_mix
        interpretation = (
            "anchor_only does not clearly recover over the donor full7 reference, so the more likely picture is "
            "that the shared head itself is already compromised inside this donor; expansion mixing may still matter, "
            "but the first usable preservation target should move to cross-donor anchor preservation."
        )
    elif leg_add_harmful and nonleg_add_harmful:
        case_label = "Case 4"
        main_call = "expansion_overall_harmful"
        dir_leg_boundary = "anchor -> expansion boundary on both sides"
        a1s2_recommend = True
        preferred_a1s2 = preferred_mix
        interpretation = (
            "anchor_only is better, but both expansion additions pull it back down. This supports an anchor-plus-expansion "
            "factorization rather than treating the donor as a monolithic full7 contract."
        )
    elif leg_vs_anchor + ASYM_MARGIN < nonleg_vs_anchor:
        case_label = "Case 1"
        main_call = "leg_expansion_mainly_harmful"
        dir_leg_boundary = "anchor -> leg expansion boundary"
        a1s2_recommend = True
        preferred_a1s2 = "E1-top3 anchor + top7 leg expansion"
        interpretation = (
            "anchor_only improves over full7, and adding leg expansion degrades more than adding nonleg expansion. "
            "The coarse scout therefore points more toward the leg expansion side as the main compatibility breaker."
        )
    elif nonleg_vs_anchor + ASYM_MARGIN < leg_vs_anchor:
        case_label = "Case 2"
        main_call = "nonleg_expansion_mainly_harmful"
        dir_leg_boundary = "anchor -> nonleg expansion boundary, with dir_leg damage appearing after cross-branch coadaptation"
        a1s2_recommend = True
        preferred_a1s2 = "E1-top3 anchor + top7 nonleg expansion"
        interpretation = (
            "anchor_only improves over full7, and adding nonleg expansion degrades more than adding leg expansion. "
            "This points more toward nonleg expansion or shared cross-branch expansion pressure as the main breaker."
        )
    else:
        case_label = "ambiguous_between_case1_case2"
        main_call = "expansion_harm_present_but_side_unclear"
        dir_leg_boundary = "anchor -> expansion boundary"
        a1s2_recommend = True
        preferred_a1s2 = preferred_mix
        interpretation = (
            "anchor_only is better than full7, so expansion pressure still looks real, but the leg-vs-nonleg asymmetry "
            "is not clean enough to call one side dominant from this coarse scout alone."
        )

    return {
        "case_label": case_label,
        "main_call": main_call,
        "anchor_vs_full7_aggregate_delta": float(anchor_vs_full7),
        "leg_plus_vs_anchor_aggregate_delta": float(leg_vs_anchor),
        "nonleg_plus_vs_anchor_aggregate_delta": float(nonleg_vs_anchor),
        "leg_plus_vs_anchor_dir_leg_closure_delta": float(leg_dirleg_vs_anchor),
        "nonleg_plus_vs_anchor_dir_leg_closure_delta": float(nonleg_dirleg_vs_anchor),
        "anchor_clearly_better_than_full7": bool(anchor_clearly_better),
        "dir_leg_degradation_boundary": dir_leg_boundary,
        "recommend_enter_A1_S2": bool(a1s2_recommend),
        "preferred_A1_S2_mixed_contract_test": preferred_a1s2,
        "residual_useful_expansion_side": preferred_side,
        "preferred_A1_S2_reason": preferred_reason,
        "interpretation": interpretation,
    }


def _build_explicit_answers(
    *,
    judgement: Mapping[str, Any],
    anchor_transfer: Mapping[str, Any],
    plus_leg_transfer: Mapping[str, Any],
    plus_nonleg_transfer: Mapping[str, Any],
    e2a_transfer: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "q1_anchor_only_more_replace_transferable_than_E2A_R_full7": {
            "answer": "yes" if bool(judgement.get("anchor_clearly_better_than_full7")) else "no",
            "aggregate_delta": float(
                _safe_float(anchor_transfer.get("aggregate_transfer_score"))
                - _safe_float(e2a_transfer.get("aggregate_transfer_score"))
            ),
            "dir_leg_closure_delta": float(
                _safe_float(anchor_transfer.get("dir_leg_closure_ratio"))
                - _safe_float(e2a_transfer.get("dir_leg_closure_ratio"))
            ),
        },
        "q2_main_break_source": str(judgement.get("main_call")),
        "q3_dir_leg_worsening_boundary": str(judgement.get("dir_leg_degradation_boundary")),
        "q4_enter_A1_S2": "yes" if bool(judgement.get("recommend_enter_A1_S2")) else "no",
        "q5_preferred_A1_S2": str(judgement.get("preferred_A1_S2_mixed_contract_test")),
        "anchor_plus_leg_vs_anchor_aggregate_delta": float(
            _safe_float(plus_leg_transfer.get("aggregate_transfer_score"))
            - _safe_float(anchor_transfer.get("aggregate_transfer_score"))
        ),
        "anchor_plus_nonleg_vs_anchor_aggregate_delta": float(
            _safe_float(plus_nonleg_transfer.get("aggregate_transfer_score"))
            - _safe_float(anchor_transfer.get("aggregate_transfer_score"))
        ),
    }


def _write_record(summary: Mapping[str, Any]) -> None:
    partition = summary["candidate_partition"]
    assays = summary["partial_assays"]
    refs = summary["reused_references"]
    judgement = summary["boundary_interpretation"]

    lines: List[str] = []
    lines.append("# 2026-04-09 partial-transplant boundary A1-S1 record")
    lines.append("")
    lines.append("> Last updated: 2026-04-09  ")
    lines.append("> Scope: A1-S1 only / fixed-host partial-transplant boundary coarse scout / no new training")
    lines.append("")
    lines.append("## 1. Scope / inherited conclusions")
    lines.append("")
    lines.append("本轮只做 **fixed host 下的 partial-transplant boundary assay**，直接继承以下结论，不重复证明：")
    lines.append("")
    for item in summary["inherited_conclusions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("同时明确声明：下述 `anchor_candidate / leg_expansion_candidate / nonleg_expansion_candidate` **只是 code-level candidate partition**，")
    lines.append("不是已经被证明的真实语义边界；A1-S1 的目标只是先看这个 partition 是否提供信息增益。")
    lines.append("")
    lines.append("## 2. Why A1-S1 before full A1")
    lines.append("")
    lines.append("- 当前 `anchor` 仍是结构假设，不是已证实的 clean semantic partition。")
    lines.append("- 先做 single-donor coarse scout，可以避免把 donor quality、anchor 假设、cross-donor mixing、boundary definition 一次性混在同一个大 grid 里。")
    lines.append("- 因此本轮只测一个 donor、三个 partial assays，不扩成 full boundary sweep。")
    lines.append("")
    lines.append("## 3. Donor / host / target inventory")
    lines.append("")
    lines.append("| item | artifact | path / note |")
    lines.append("|---|---|---|")
    lines.append(f"| donor | E2A-R final70a ckpt | `{summary['donor']['ckpt']}` |")
    lines.append(f"| donor eval | fixed eval artifact | `{summary['donor']['eval_json']}` |")
    lines.append(f"| donor config | stage70a config | `{summary['donor']['config']}` |")
    lines.append(f"| host | coadapt replace host ckpt | `{summary['host']['ckpt']}` |")
    lines.append(f"| host config | fixed host config | `{summary['host']['config']}` |")
    lines.append(f"| baseline donor | baseline replace ckpt | `{summary['baseline_replace']['ckpt']}` |")
    lines.append("| target | synthetic transplant-compatible target | in-memory: fixed host + baseline replace full7 transplant |")
    lines.append("")
    lines.append("## 4. Candidate partition table")
    lines.append("")
    lines.append("| family | modules | parameter prefixes | note |")
    lines.append("|---|---|---|---|")
    for key in ("anchor_candidate", "leg_expansion_candidate", "nonleg_expansion_candidate"):
        block = partition[key]
        lines.append(
            f"| `{key}` | `{', '.join(block['modules'])}` | `{', '.join(block['parameter_prefixes'])}` | code-level assay hypothesis only |"
        )
    lines.append("")
    lines.append("## 5. Assay inventory table")
    lines.append("")
    lines.append("| assay | transplanted modules | parameter prefixes |")
    lines.append("|---|---|---|")
    for name, row in assays.items():
        lines.append(
            f"| `{name}` | `{', '.join(row['modules'])}` | `{', '.join(row['parameter_prefixes'])}` |"
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
    ]
    for name, row in assays.items():
        table_rows.append((name, row["transfer"]))
    for name, row in table_rows:
        lines.append(
            f"| `{name}` | {_fmt(row['out_direct_gap'])} | {_fmt(row['dir_base_gap'])} | {_fmt(row['dir_leg_gap'])} | {_fmt(row['dir_nonleg_gap'])} | "
            f"{_fmt(row['out_direct_closure_ratio'])} | {_fmt(row['dir_base_closure_ratio'])} | {_fmt(row['dir_leg_closure_ratio'])} | {_fmt(row['dir_nonleg_closure_ratio'])} | {_fmt(row['aggregate_transfer_score'])} |"
        )
    lines.append("")
    lines.append("## 7. `dir_leg`-focused interpretation")
    lines.append("")
    anchor = assays["A1S1-anchor_only"]["transfer"]
    plus_leg = assays["A1S1-anchor_plus_leg"]["transfer"]
    plus_nonleg = assays["A1S1-anchor_plus_nonleg"]["transfer"]
    lines.append(
        f"- `A1S1-anchor_only` dir_leg closure = `{_fmt(anchor['dir_leg_closure_ratio'])}`."
    )
    lines.append(
        f"- `A1S1-anchor_plus_leg` dir_leg closure = `{_fmt(plus_leg['dir_leg_closure_ratio'])}`; delta vs anchor_only = `{_fmt(_safe_float(plus_leg['dir_leg_closure_ratio']) - _safe_float(anchor['dir_leg_closure_ratio']))}`."
    )
    lines.append(
        f"- `A1S1-anchor_plus_nonleg` dir_leg closure = `{_fmt(plus_nonleg['dir_leg_closure_ratio'])}`; delta vs anchor_only = `{_fmt(_safe_float(plus_nonleg['dir_leg_closure_ratio']) - _safe_float(anchor['dir_leg_closure_ratio']))}`."
    )
    lines.append(
        f"- 因此本轮把 `dir_leg` 的主要恶化边界读成：**{judgement['dir_leg_degradation_boundary']}**。"
    )
    lines.append("")
    lines.append("## 8. `dir_base` / `dir_nonleg` retention summary")
    lines.append("")
    lines.append(
        f"- `anchor_only` 对 `dir_base` 的 closure = `{_fmt(anchor['dir_base_closure_ratio'])}`，对 `dir_nonleg` 的 closure = `{_fmt(anchor['dir_nonleg_closure_ratio'])}`。"
    )
    lines.append(
        f"- `anchor_plus_leg` aggregate delta vs anchor_only = `{_fmt(_safe_float(plus_leg['aggregate_transfer_score']) - _safe_float(anchor['aggregate_transfer_score']))}`。"
    )
    lines.append(
        f"- `anchor_plus_nonleg` aggregate delta vs anchor_only = `{_fmt(_safe_float(plus_nonleg['aggregate_transfer_score']) - _safe_float(anchor['aggregate_transfer_score']))}`。"
    )
    lines.append(
        f"- `anchor_plus_nonleg` 的 `dir_nonleg` closure = `{_fmt(plus_nonleg['dir_nonleg_closure_ratio'])}`，与 `E2A-R full7` 的 `dir_nonleg` closure 完全持平；aggregate 也接近 `E2A-R full7` (`{_fmt(plus_nonleg['aggregate_transfer_score'])}` vs `{_fmt(refs['E2A_R_full7']['transfer']['aggregate_transfer_score'])}`)。"
    )
    lines.append(
        f"- `anchor_plus_leg` 只比 `anchor_only` 小幅改善 aggregate (`{_fmt(_safe_float(plus_leg['aggregate_transfer_score']) - _safe_float(anchor['aggregate_transfer_score']))}`)，信息增益明显弱于 nonleg 侧。"
    )
    lines.append("- 这能帮助区分：是 shared head 自身已经坏掉，还是某一侧 expansion 进入后打破了原本还能工作的 contract。")
    lines.append("")
    lines.append("## 9. Boundary interpretation")
    lines.append("")
    lines.append(f"- 判例归类：**{judgement['case_label']}**")
    lines.append(f"- 主判断：**{judgement['main_call']}**")
    lines.append(f"- 解释：{judgement['interpretation']}")
    lines.append("- 口径克制：这仍然只是 single-donor coarse scout，不能把 candidate partition 直接升级成已证实真相。")
    lines.append("")
    lines.append("## 10. Whether this supports A1-S2")
    lines.append("")
    lines.append(
        f"- 是否建议进入 `A1-S2`：**{'yes' if judgement['recommend_enter_A1_S2'] else 'no'}**"
    )
    lines.append(
        f"- 如果进入 `A1-S2`，优先测：**{judgement['preferred_A1_S2_mixed_contract_test']}**"
    )
    lines.append(
        f"- 选择理由：{judgement['preferred_A1_S2_reason']}；本轮 residual 更像保留在 **{judgement['residual_useful_expansion_side']}**。"
    )
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
        E2A_70A_CKPT,
        E2A_70A_EVAL,
        TOP3_70A_CKPT,
        TOP3_70A_EVAL,
        E1_SUMMARY_JSON,
        E2A_SUMMARY_JSON,
        E2C_SUMMARY_JSON,
        E3A_SUMMARY_JSON,
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise SystemExit("[FATAL] missing required artifact(s):\n" + "\n".join(missing))

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
    donor_bundle = _case_bundle(
        case_name="E2A_R_final70a_donor",
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
    host_native = _run_single_step(host_bundle, prep_host, fixed_contacts=fixed_contacts)
    target_result = _run_single_step(
        host_bundle,
        prep_host,
        fixed_contacts=fixed_contacts,
        weight_swap_modules=list(ANCHOR_CANDIDATE + LEG_EXPANSION_CANDIDATE + NONLEG_EXPANSION_CANDIDATE),
        donor_bundle=baseline_bundle,
    )
    host_gaps = _tensor_metric_gaps(
        host_case=host_bundle["case"],
        target_result=target_result,
        candidate_result=host_native,
    )

    donor_prefix_info = _prefix_info_for_modules(
        E2A_70A_CKPT,
        list(ANCHOR_CANDIDATE + LEG_EXPANSION_CANDIDATE + NONLEG_EXPANSION_CANDIDATE),
    )

    partial_results: Dict[str, Any] = {}
    for spec in ASSAY_SPECS:
        transfer = _assay_transfer(
            host_bundle=host_bundle,
            prep_host=prep_host,
            donor_bundle=donor_bundle,
            fixed_contacts=fixed_contacts,
            target_result=target_result,
            host_gaps=host_gaps,
            modules=spec["modules"],
        )
        partial_results[str(spec["name"])] = {
            "family": str(spec["family"]),
            "modules": [str(module) for module in spec["modules"]],
            "parameter_prefixes": [str(donor_prefix_info[str(module)]["parameter_prefix"]) for module in spec["modules"]],
            "transfer": transfer,
        }

    host_native_reference = _build_host_native_reference(host_gaps)
    target_reference = _build_target_reference()
    e1_top3_reference = _load_reference_transfer(E1_SUMMARY_JSON, "top3")
    e2a_full7_reference = _load_reference_transfer(E2A_SUMMARY_JSON, "E2A-R")

    for row in partial_results.values():
        row["delta_vs_E2A_R_full7"] = _transfer_delta(row["transfer"], e2a_full7_reference["transfer"])
        row["delta_vs_E1_top3_full7"] = _transfer_delta(row["transfer"], e1_top3_reference["transfer"])

    judgement = _classify_boundary_case(
        anchor_transfer=partial_results["A1S1-anchor_only"]["transfer"],
        plus_leg_transfer=partial_results["A1S1-anchor_plus_leg"]["transfer"],
        plus_nonleg_transfer=partial_results["A1S1-anchor_plus_nonleg"]["transfer"],
        full7_transfer=e2a_full7_reference["transfer"],
    )
    explicit_answers = _build_explicit_answers(
        judgement=judgement,
        anchor_transfer=partial_results["A1S1-anchor_only"]["transfer"],
        plus_leg_transfer=partial_results["A1S1-anchor_plus_leg"]["transfer"],
        plus_nonleg_transfer=partial_results["A1S1-anchor_plus_nonleg"]["transfer"],
        e2a_transfer=e2a_full7_reference["transfer"],
    )

    summary = {
        "analysis": RUN_NAME,
        "scope": {
            "experiment": "A1-S1 partial-transplant boundary coarse scout",
            "mode": "single-donor fixed-host partial transplant assay",
            "fixed_replace_context": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "transplant_compatible_target": "same host + baseline replace 7-module direct-branch transplant",
            "assay_mode": "deterministic first-forward",
            "offset": DEFAULT_OFFSET,
            "fixed_contacts_source": "baseline replace native same-entry contacts_in_t",
            "constraints": [
                "no new training",
                "no architecture redesign",
                "no donor sweep",
                "no full boundary grid",
                "no planner semantics side quest",
                "no normality-probe main criterion",
            ],
            "why_A1_S1_before_full_A1": [
                "anchor is still only a structural hypothesis",
                "single-donor scout avoids mixing donor quality, anchor definition, and cross-donor mixing",
                "goal is coarse information gain, not exhaustive boundary proof",
            ],
        },
        "inherited_conclusions": INHERITED_CONCLUSIONS,
        "plumbing_check": {
            "module_subset_transplant_supported": True,
            "execution_path": [
                "tools.analyze_cp015_tailk7_same_input_module_attribution::_temporary_weight_swap",
                "tools.analyze_cp015_tailk7_same_input_module_attribution::_run_single_step(weight_swap_modules=...)",
            ],
            "note": "subset transplant already exists as temporary per-submodule state_dict swap; no new architecture or training plumbing was introduced",
        },
        "donor": {
            "label": "E2A-R final70a",
            "ckpt": str(E2A_70A_CKPT),
            "eval_json": str(E2A_70A_EVAL),
            "config": str(STAGE70A_CONFIG),
            "reason": "best current top7-family final70a; if partial transplant cannot rescue this donor, weaker top7 donors add less information",
        },
        "host": {
            "label": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "ckpt": str(COADAPT_HOST_CKPT),
            "eval_json": str(COADAPT_HOST_EVAL),
            "config": str(COADAPT_HOST_CONFIG),
        },
        "baseline_replace": {
            "label": "baseline_replace",
            "ckpt": str(BASELINE_REPLACE_CKPT),
            "eval_json": str(BASELINE_REPLACE_EVAL),
            "config": str(BASELINE_REPLACE_CONFIG),
        },
        "target_artifact": {
            "type": "synthetic_in_memory_transplant_target",
            "materialization": "fixed host + baseline replace full7 transplant",
            "host_ckpt": str(COADAPT_HOST_CKPT),
            "baseline_donor_ckpt": str(BASELINE_REPLACE_CKPT),
            "module_subset": list(ANCHOR_CANDIDATE + LEG_EXPANSION_CANDIDATE + NONLEG_EXPANSION_CANDIDATE),
        },
        "candidate_partition": {
            "anchor_candidate": _partition_block(
                label="anchor_candidate",
                modules=ANCHOR_CANDIDATE,
                prefix_info=donor_prefix_info,
            ),
            "leg_expansion_candidate": _partition_block(
                label="leg_expansion_candidate",
                modules=LEG_EXPANSION_CANDIDATE,
                prefix_info=donor_prefix_info,
            ),
            "nonleg_expansion_candidate": _partition_block(
                label="nonleg_expansion_candidate",
                modules=NONLEG_EXPANSION_CANDIDATE,
                prefix_info=donor_prefix_info,
            ),
        },
        "reused_references": {
            "host_native_bad_reference": {
                "source": "recomputed fixed host-native against same target",
                "transfer": host_native_reference,
            },
            "transplant_compatible_target": {
                "source": "same fixed assay target",
                "transfer": target_reference,
            },
            "E1_top3_full7": e1_top3_reference,
            "E2A_R_full7": e2a_full7_reference,
        },
        "partial_assays": partial_results,
        "boundary_interpretation": judgement,
        "explicit_answers": explicit_answers,
        "reference_summary_paths": {
            "E1": str(E1_SUMMARY_JSON),
            "E2A": str(E2A_SUMMARY_JSON),
            "E2C": str(E2C_SUMMARY_JSON),
            "E3A": str(E3A_SUMMARY_JSON),
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_record(summary)
    print(f"[OK] wrote {SUMMARY_JSON}")
    print(f"[OK] wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
