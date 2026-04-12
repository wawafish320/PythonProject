#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_closed_loop_gap import (  # noqa: E402
    DEFAULT_TAIL_CKPT,
    DEFAULT_TAIL_EVAL,
    DEFAULT_TEACHER,
    _load_case,
)
from tools.analyze_cp015_tailk7_hfinal_drift import (  # noqa: E402
    DEFAULT_OUT,
    _association_summary,
    _group_error_series,
    _growth_summary,
    _lead_lag_summary,
    _offset_summary,
    _plan_z_trace,
    _selected_window_bucket_summary,
    _selected_window_cycle_offsets,
    _tensor_to_mean_vec,
    _timing_summaries,
    _trace_metric,
)
from tools.analyze_cp015_tailk7_single_step_rescue import _select_target_steps  # noqa: E402
from train.validate.run_freerun_cycles import _run_freerun_cycles  # noqa: E402


RUN_DATE = "20260404"
DEFAULT_COND_OUT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_cond_drift_audit_{RUN_DATE}" / "summary.json"


def _capture_cond_trace(
    case: Mapping[str, Any],
    *,
    rounds: int,
    teacher_conditioned: bool,
) -> Dict[str, Any]:
    trainer = case["trainer"]
    model = trainer.model
    if model is None:
        raise RuntimeError("trainer.model missing")
    runner = case["runner"]
    captured = []

    def _pre_hook(_module: Any, inputs: Any) -> None:
        cond = None
        if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
            cond = inputs[1]
        captured.append(_tensor_to_mean_vec(cond))

    handle = model.register_forward_pre_hook(_pre_hook)
    try:
        metrics_per_round, per_step, extra = _run_freerun_cycles(
            trainer=trainer,
            sample=case["sample"],
            rounds=int(rounds),
            device=runner.device,
            time_index_mode=str(case["runtime_overrides"]["time_index_mode"]),
            lambda_fusion_apply=bool(case["runtime_overrides"]["lambda_fusion_apply"]),
            export_plan_state_series=True,
            pose_hist_source=("seq" if teacher_conditioned else str(case["runtime_overrides"]["pose_hist_source"])),
            pose_hist_update_source=("gt" if teacher_conditioned else str(case["runtime_overrides"]["pose_hist_update_source"])),
            freerun_x_gt=bool(teacher_conditioned),
        )
    finally:
        handle.remove()
    if len(captured) != len(per_step):
        raise RuntimeError(f"cond trace length mismatch: hook={len(captured)} per_step={len(per_step)}")
    return {
        "metrics_per_round": metrics_per_round,
        "per_step": per_step,
        "extra": extra,
        "cond_vecs": captured,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Teacher-vs-freerun cond drift audit for cp015 tailk7 control.")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_TAIL_CKPT)
    ap.add_argument("--eval", type=Path, default=DEFAULT_TAIL_EVAL)
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--depth-min", type=int, default=10)
    ap.add_argument("--sic-lo", type=int, default=11)
    ap.add_argument("--sic-hi", type=int, default=43)
    ap.add_argument("--drop-wrap", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_COND_OUT)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ckpt = args.ckpt.expanduser().resolve()
    eval_json = args.eval.expanduser().resolve()
    teacher = args.teacher.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    for path in (ckpt, eval_json, teacher):
        if not path.is_file():
            raise SystemExit(f"[FATAL] missing input: {path}")

    case = _load_case(
        case_name="tailk7_factorized_control",
        ckpt_path=ckpt,
        eval_json_path=eval_json,
        teacher_path=teacher,
        device_pref=str(args.device),
    )
    selected_meta = _select_target_steps(
        eval_json,
        depth_min=int(args.depth_min),
        sic_lo=int(args.sic_lo),
        sic_hi=int(args.sic_hi),
        drop_wrap=bool(args.drop_wrap),
    )
    if not selected_meta:
        raise SystemExit("[FATAL] no target steps selected")
    selected_steps = [int(m.step_idx) for m in selected_meta]

    freerun = _capture_cond_trace(case, rounds=int(args.rounds), teacher_conditioned=False)
    teacher_run = _capture_cond_trace(case, rounds=int(args.rounds), teacher_conditioned=True)
    if len(freerun["per_step"]) != len(teacher_run["per_step"]):
        raise RuntimeError(
            f"per_step length mismatch: freerun={len(freerun['per_step'])} teacher={len(teacher_run['per_step'])}"
        )

    cond_trace = _trace_metric(freerun["cond_vecs"], teacher_run["cond_vecs"])
    plan_trace = _trace_metric(_plan_z_trace(freerun["extra"]), _plan_z_trace(teacher_run["extra"]))
    group_errors = _group_error_series(case, freerun["per_step"])

    payload: Dict[str, Any] = {
        "analysis": "cond_drift_audit",
        "case_name": case["case_name"],
        "ckpt_path": case["ckpt_path"],
        "eval_json_path": case["eval_json_path"],
        "teacher_path": case["teacher_path"],
        "runtime_overrides": case["runtime_overrides"],
        "selection": {
            "depth_min": int(args.depth_min),
            "sic_range": [int(args.sic_lo), int(args.sic_hi)],
            "drop_wrap": bool(args.drop_wrap),
            "selected_steps": int(len(selected_steps)),
        },
        "metric_definition": {
            "cond_drift": {
                "norm_l2": "||cond_free - cond_teacher||_2 / sqrt(D)",
                "cosine_distance": "1 - cosine_similarity(cond_free, cond_teacher)",
            },
            "error": "used_local_geo_deg approximated from metrics_per_step[*].KeyBoneGeoLocalDeg and grouped by arm/all_ex_root/leg.",
            "note": "cond trace is captured from the model forward input (2nd positional arg), i.e. the effective cond_input after rollout-side reproject+normalize overrides.",
        },
        "trace_series": {
            "steps": int(cond_trace["steps"]),
            "step_meta": [
                {
                    "step": int(i),
                    "cycle": int(rec.get("cycle", 0) or 0),
                    "step_in_cycle": int(rec.get("step_in_cycle", -1) or -1),
                    "wrap_boundary_step": bool(rec.get("wrap_boundary_step", False)),
                }
                for i, rec in enumerate(freerun["per_step"])
            ],
            "cond": cond_trace,
            "plan_z": plan_trace,
            "used_local_geo_deg": {group: list(vals) for group, vals in group_errors.items()},
        },
        "summary": {
            "cond_offsets": _offset_summary(selected_steps=selected_steps, trace=cond_trace),
            "cond_growth": _growth_summary(selected_steps=selected_steps, trace=cond_trace),
            "cond_future_error_association": _association_summary(
                selected_steps=selected_steps,
                trace=cond_trace,
                group_errors=group_errors,
            ),
            "selected_window_cycle_offsets": _selected_window_cycle_offsets(selected_meta, cond_trace),
            "selected_window_depth_buckets": _selected_window_bucket_summary(
                selected_meta=selected_meta,
                trace=cond_trace,
                group_errors=group_errors,
                key="step_idx",
                specs=(
                    ("d10_20", 10, 20),
                    ("d21_43", 21, 43),
                    ("d87_173", 87, 173),
                    ("d174_433", 174, 433),
                ),
            ),
            "selected_window_sic_buckets": _selected_window_bucket_summary(
                selected_meta=selected_meta,
                trace=cond_trace,
                group_errors=group_errors,
                key="step_in_cycle",
                specs=(
                    ("sic11_21", 11, 21),
                    ("sic22_43", 22, 43),
                ),
            ),
            "timing": _timing_summaries(
                freerun["per_step"],
                cond_trace,
                group_errors,
            ),
            "plan_z_offsets": _offset_summary(selected_steps=selected_steps, trace=plan_trace),
            "plan_z_growth": _growth_summary(selected_steps=selected_steps, trace=plan_trace),
            "plan_z_lead_lag_vs_cond": _lead_lag_summary(plan_trace, cond_trace),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
