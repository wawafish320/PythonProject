#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_closed_loop_gap import (  # noqa: E402
    DEFAULT_TAIL_CKPT,
    DEFAULT_TAIL_EVAL,
    DEFAULT_TEACHER,
    GROUP_KEYS,
    _direct_local_geo_deg,
    _load_case,
    _summary,
)
from train.history import PoseHistState  # noqa: E402
from train.training_MPL import (  # noqa: E402
    RolloutExecutionState,
    RolloutSequenceInputs,
    _new_rollout_prediction_buffers,
)


RUN_DATE = "20260404"
DEFAULT_OUT = (
    ROOT / "debug_output" / f"_tmp_cp015_tailk7_single_step_rescue_audit_{RUN_DATE}" / "summary.json"
)
RESCUE_TYPES: tuple[str, ...] = ("none", "pose_history", "plan_z", "h_final")
HORIZONS: tuple[int, ...] = (5, 20)


@dataclass(frozen=True)
class StepMeta:
    step_idx: int
    cycle: int
    step_in_cycle: int
    wrap_boundary_step: bool


def _clone_tensor(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().clone()
    return x


def _clone_pose_hist_state(state: PoseHistState) -> PoseHistState:
    return PoseHistState(
        enabled=bool(state.enabled),
        length=int(state.length),
        dim=int(state.dim),
        stride=int(state.stride),
        scales=_clone_tensor(state.scales),
        mu=_clone_tensor(state.mu),
        std=_clone_tensor(state.std),
        buffer_norm=_clone_tensor(state.buffer_norm),
        buffer_raw=_clone_tensor(state.buffer_raw),
    )


def _clone_rollout_state(rollout: RolloutExecutionState) -> RolloutExecutionState:
    return RolloutExecutionState(
        batch_size=int(rollout.batch_size),
        total_steps=int(rollout.total_steps),
        mode=str(rollout.mode),
        allow_grad=bool(rollout.allow_grad),
        tf_ratio=float(rollout.tf_ratio),
        ss_chunk_len=int(rollout.ss_chunk_len),
        amp_enabled=bool(rollout.amp_enabled),
        rot6d_slice=rollout.rot6d_slice,
        rot6d_y_slice=rollout.rot6d_y_slice,
        has_time_dim=dict(rollout.has_time_dim),
        cond_norm_mu=_clone_tensor(rollout.cond_norm_mu),
        cond_norm_std=_clone_tensor(rollout.cond_norm_std),
        enable_reprojection=bool(rollout.enable_reprojection),
        plan_enable=bool(rollout.plan_enable),
        time_base_local=_clone_tensor(rollout.time_base_local),
        motion=_clone_tensor(rollout.motion),
        motion_raw_local=_clone_tensor(rollout.motion_raw_local),
        y_raw_local=_clone_tensor(rollout.y_raw_local),
        pose_hist_state=_clone_pose_hist_state(rollout.pose_hist_state),
        ss_sel_hold=_clone_tensor(rollout.ss_sel_hold),
        plan_z=_clone_tensor(rollout.plan_z),
        meas_prev_prob=_clone_tensor(rollout.meas_prev_prob),
        prev_foot_pos_meas=_clone_tensor(rollout.prev_foot_pos_meas),
        reprojection_applied_count=int(rollout.reprojection_applied_count),
        last_attn=_clone_tensor(rollout.last_attn),
        latest_y_raw=_clone_tensor(rollout.latest_y_raw),
        latest_cond_raw_for_env=_clone_tensor(rollout.latest_cond_raw_for_env),
        outs=[],
        delta_preds=[],
        buffers=_new_rollout_prediction_buffers(),
    )


def _metric_from_y_raw(
    *,
    case: Mapping[str, Any],
    y_raw: torch.Tensor,
    gt_seq: torch.Tensor,
    step_idx: int,
) -> Dict[str, float]:
    trainer = case["trainer"]
    gt_frame = trainer._denorm(gt_seq[:, min(int(step_idx) + 1, int(gt_seq.shape[1]) - 1)]).detach().cpu()
    pred_raw = y_raw.detach().cpu()
    geo = _direct_local_geo_deg(
        pred_raw=pred_raw,
        gt_raw=gt_frame,
        rot_slice=case["rot_slice"],
        root_idx=int(case["root_idx"]),
        columns=case["columns"],
    ).detach().cpu().numpy().astype(np.float64, copy=False)
    out: Dict[str, float] = {}
    for group in GROUP_KEYS:
        cols = list(case["groups"][group])
        vals = geo[:, np.asarray(cols, dtype=np.int64)].reshape(-1) if cols else np.asarray([], dtype=np.float64)
        out[group] = float(np.mean(vals)) if vals.size > 0 else float("nan")
    return out


def _select_target_steps(
    eval_json_path: Path,
    *,
    depth_min: int,
    sic_lo: int,
    sic_hi: int,
    drop_wrap: bool,
) -> List[StepMeta]:
    obj = json.loads(eval_json_path.read_text())
    steps = obj.get("metrics_per_step", [])
    if not isinstance(steps, list):
        raise RuntimeError(f"invalid eval json: missing metrics_per_step: {eval_json_path}")
    out: List[StepMeta] = []
    for step_idx, rec in enumerate(steps):
        if not isinstance(rec, Mapping):
            continue
        if int(step_idx) < int(depth_min):
            continue
        try:
            sic = int(rec.get("step_in_cycle", -1) or -1)
        except Exception:
            sic = -1
        if sic < int(sic_lo) or sic > int(sic_hi):
            continue
        wrap = bool(rec.get("wrap_boundary_step", False))
        if bool(drop_wrap) and wrap:
            continue
        try:
            cycle = int(rec.get("cycle", 0) or 0)
        except Exception:
            cycle = 0
        out.append(
            StepMeta(
                step_idx=int(step_idx),
                cycle=int(cycle),
                step_in_cycle=int(sic),
                wrap_boundary_step=bool(wrap),
            )
        )
    return out


def _build_rollout_inputs(case: Mapping[str, Any]) -> RolloutSequenceInputs:
    batched = case["batched"]
    return RolloutSequenceInputs(
        state_seq=batched["motion"],
        cond_seq=batched.get("cond_in"),
        cond_raw_seq=batched.get("cond_tgt_raw"),
        contacts_seq=batched.get("contacts"),
        angvel_seq=batched.get("angvel"),
        pose_hist_seq=batched.get("pose_hist"),
        gt_seq=batched["gt_motion"],
    )


def _register_hidden_rescue_hook(model: Any, teacher_hidden: torch.Tensor):
    target = teacher_hidden.detach().clone()

    def _hook(_module: Any, _inputs: Any, output: Any):
        if not torch.is_tensor(output):
            return output
        out = target.to(device=output.device, dtype=output.dtype)
        if output.dim() == 3 and out.dim() == 2:
            out = out.unsqueeze(1)
        if output.dim() == 2 and out.dim() == 3 and out.size(1) == 1:
            out = out[:, 0]
        if tuple(out.shape) != tuple(output.shape):
            raise RuntimeError(f"h_final rescue shape mismatch: teacher={tuple(out.shape)} runtime={tuple(output.shape)}")
        return out

    return model.coupling_norm.register_forward_hook(_hook)


def _run_rollout_pass(
    *,
    case: Mapping[str, Any],
    rollout_inputs: RolloutSequenceInputs,
    mode: str,
    tf_ratio: float,
    capture_steps: Sequence[int],
    capture_teacher_hidden: bool,
) -> Dict[str, Any]:
    trainer = case["trainer"]
    rollout = trainer._init_rollout_state(
        rollout_inputs,
        cond_norm_mu=case["batched"].get("cond_norm_mu"),
        cond_norm_std=case["batched"].get("cond_norm_std"),
        mode=str(mode),
        tf_ratio=float(tf_ratio),
        time_base=case["batched"].get("start"),
    )
    selected = set(int(x) for x in capture_steps)
    snapshots: Dict[int, Dict[str, Any]] = {}
    errors: List[Dict[str, float]] = []

    for step_idx in range(int(rollout.total_steps)):
        if int(step_idx) in selected:
            snapshots[int(step_idx)] = {
                "rollout": _clone_rollout_state(rollout),
            }
        trainer._rollout_forward_step(rollout, rollout_inputs, step_idx=int(step_idx))
        if rollout.latest_y_raw is None:
            raise RuntimeError(f"step {step_idx}: rollout.latest_y_raw missing")
        err = _metric_from_y_raw(
            case=case,
            y_raw=rollout.latest_y_raw,
            gt_seq=rollout_inputs.gt_seq,
            step_idx=int(step_idx),
        )
        errors.append(err)
        if int(step_idx) in selected and bool(capture_teacher_hidden):
            hidden = None
            if rollout.buffers.hidden_seq:
                hidden = rollout.buffers.hidden_seq[-1]
                if torch.is_tensor(hidden) and hidden.dim() == 3 and hidden.size(1) == 1:
                    hidden = hidden[:, 0]
            snapshots[int(step_idx)]["teacher_h_final"] = _clone_tensor(hidden)
            snapshots[int(step_idx)]["teacher_plan_z"] = _clone_tensor(rollout.plan_z)
            snapshots[int(step_idx)]["teacher_pose_hist_state"] = _clone_pose_hist_state(rollout.pose_hist_state)
        trainer._apply_scheduled_sampling_update(rollout, rollout_inputs, step_idx=int(step_idx))

    return {
        "total_steps": int(rollout.total_steps),
        "snapshots": snapshots,
        "errors": errors,
    }


def _run_rescue_from_snapshot(
    *,
    case: Mapping[str, Any],
    rollout_inputs: RolloutSequenceInputs,
    baseline_snapshot: Mapping[str, Any],
    teacher_snapshot: Mapping[str, Any],
    start_step: int,
    rescue_kind: str,
    horizon: int,
) -> List[Dict[str, float]]:
    trainer = case["trainer"]
    model = trainer.model
    if model is None:
        raise RuntimeError("trainer.model is missing")
    rollout = _clone_rollout_state(baseline_snapshot["rollout"])

    if rescue_kind == "pose_history":
        rollout.pose_hist_state = _clone_pose_hist_state(teacher_snapshot["teacher_pose_hist_state"])
    elif rescue_kind == "plan_z":
        rollout.plan_z = _clone_tensor(teacher_snapshot["teacher_plan_z"])

    end_step = min(int(rollout.total_steps), int(start_step) + int(horizon))
    errors: List[Dict[str, float]] = []
    for step_idx in range(int(start_step), int(end_step)):
        hook_handle = None
        if rescue_kind == "h_final" and int(step_idx) == int(start_step):
            teacher_hidden = teacher_snapshot.get("teacher_h_final", None)
            if not torch.is_tensor(teacher_hidden):
                raise RuntimeError(f"step {start_step}: missing teacher_h_final for hidden rescue")
            hook_handle = _register_hidden_rescue_hook(model, teacher_hidden)
        try:
            trainer._rollout_forward_step(rollout, rollout_inputs, step_idx=int(step_idx))
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        if rollout.latest_y_raw is None:
            raise RuntimeError(f"step {step_idx}: rollout.latest_y_raw missing during rescue")
        err = _metric_from_y_raw(
            case=case,
            y_raw=rollout.latest_y_raw,
            gt_seq=rollout_inputs.gt_seq,
            step_idx=int(step_idx),
        )
        errors.append(err)
        trainer._apply_scheduled_sampling_update(rollout, rollout_inputs, step_idx=int(step_idx))
    return errors


def _finite(vals: Iterable[Any]) -> np.ndarray:
    out: List[float] = []
    for value in vals:
        try:
            fv = float(value)
        except Exception:
            continue
        if math.isfinite(fv):
            out.append(fv)
    return np.asarray(out, dtype=np.float64)


def _aggregate_windows(
    *,
    baseline_errors: Sequence[Dict[str, float]],
    step_meta: Sequence[StepMeta],
    rescue_windows: Mapping[str, Mapping[int, List[Dict[str, float]]]],
    horizons: Sequence[int],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    by_step = {int(m.step_idx): m for m in step_meta}
    for rescue_kind, horizon_map in rescue_windows.items():
        summary[rescue_kind] = {}
        for horizon in horizons:
            per_group_improve: Dict[str, List[float]] = {g: [] for g in GROUP_KEYS}
            per_group_base: Dict[str, List[float]] = {g: [] for g in GROUP_KEYS}
            per_group_rescue: Dict[str, List[float]] = {g: [] for g in GROUP_KEYS}
            positive_counts = {g: 0 for g in GROUP_KEYS}
            window_count = 0
            for start_step, rescue_seq in horizon_map.items():
                base_seq = baseline_errors[int(start_step) : int(start_step) + int(horizon)]
                if len(base_seq) != len(rescue_seq) or not rescue_seq:
                    continue
                window_count += 1
                for group in GROUP_KEYS:
                    base_vals = _finite(row.get(group) for row in base_seq)
                    rescue_vals = _finite(row.get(group) for row in rescue_seq)
                    if base_vals.size <= 0 or rescue_vals.size <= 0:
                        continue
                    base_mean = float(np.mean(base_vals))
                    rescue_mean = float(np.mean(rescue_vals))
                    improve = float(base_mean - rescue_mean)
                    per_group_base[group].append(base_mean)
                    per_group_rescue[group].append(rescue_mean)
                    per_group_improve[group].append(improve)
                    if improve > 0.0:
                        positive_counts[group] += 1
            summary[rescue_kind][f"horizon_{int(horizon)}"] = {
                "windows": int(window_count),
                "groups": {
                    group: {
                        "baseline_window_mean": _summary(_finite(per_group_base[group])),
                        "rescued_window_mean": _summary(_finite(per_group_rescue[group])),
                        "improvement_deg": _summary(_finite(per_group_improve[group])),
                        "positive_rate": (
                            float(positive_counts[group]) / float(max(1, len(per_group_improve[group])))
                            if per_group_improve[group]
                            else float("nan")
                        ),
                    }
                    for group in GROUP_KEYS
                },
            }
    summary["window_meta"] = {
        "count": int(len(step_meta)),
        "first_steps": [
            {
                "step_idx": int(m.step_idx),
                "cycle": int(m.cycle),
                "step_in_cycle": int(m.step_in_cycle),
            }
            for m in list(step_meta)[:10]
        ],
    }
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Single-step rescue audit for cp015 tailk7 control.")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_TAIL_CKPT)
    ap.add_argument("--eval", type=Path, default=DEFAULT_TAIL_EVAL)
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--depth-min", type=int, default=10)
    ap.add_argument("--sic-lo", type=int, default=11)
    ap.add_argument("--sic-hi", type=int, default=43)
    ap.add_argument("--drop-wrap", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
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
    rollout_inputs = _build_rollout_inputs(case)
    target_steps = _select_target_steps(
        eval_json,
        depth_min=int(args.depth_min),
        sic_lo=int(args.sic_lo),
        sic_hi=int(args.sic_hi),
        drop_wrap=bool(args.drop_wrap),
    )
    if not target_steps:
        raise SystemExit("[FATAL] no target steps selected")
    selected = [int(m.step_idx) for m in target_steps]

    baseline_pass = _run_rollout_pass(
        case=case,
        rollout_inputs=rollout_inputs,
        mode="free",
        tf_ratio=0.0,
        capture_steps=selected,
        capture_teacher_hidden=False,
    )
    teacher_pass = _run_rollout_pass(
        case=case,
        rollout_inputs=rollout_inputs,
        mode="mixed",
        tf_ratio=1.0,
        capture_steps=selected,
        capture_teacher_hidden=True,
    )

    rescue_windows: Dict[str, Dict[int, List[Dict[str, float]]]] = {
        kind: {} for kind in RESCUE_TYPES
    }
    baseline_errors = baseline_pass["errors"]
    for rescue_kind in RESCUE_TYPES:
        for meta in target_steps:
            step_idx = int(meta.step_idx)
            if step_idx not in baseline_pass["snapshots"] or step_idx not in teacher_pass["snapshots"]:
                continue
            if rescue_kind == "none":
                rescue_seq = baseline_errors[step_idx : step_idx + max(HORIZONS)]
            else:
                rescue_seq = _run_rescue_from_snapshot(
                    case=case,
                    rollout_inputs=rollout_inputs,
                    baseline_snapshot=baseline_pass["snapshots"][step_idx],
                    teacher_snapshot=teacher_pass["snapshots"][step_idx],
                    start_step=step_idx,
                    rescue_kind=rescue_kind,
                    horizon=max(HORIZONS),
                )
            rescue_windows[rescue_kind][step_idx] = rescue_seq

    horizon_windows: Dict[str, Dict[int, List[Dict[str, float]]]] = {
        kind: {} for kind in RESCUE_TYPES
    }
    for rescue_kind, window_map in rescue_windows.items():
        horizon_windows[rescue_kind] = {}
        for horizon in HORIZONS:
            horizon_windows[rescue_kind][int(horizon)] = {}

    payload = {
        "analysis": "single_step_rescue_audit",
        "case_name": case["case_name"],
        "ckpt_path": case["ckpt_path"],
        "eval_json_path": case["eval_json_path"],
        "teacher_path": case["teacher_path"],
        "runtime_overrides": case["runtime_overrides"],
        "selection": {
            "depth_min": int(args.depth_min),
            "sic_range": [int(args.sic_lo), int(args.sic_hi)],
            "drop_wrap": bool(args.drop_wrap),
            "selected_steps": int(len(target_steps)),
        },
        "metric_definition": {
            "name": "used_local_geo_deg",
            "description": "Local geodesic error in degrees between carried rollout pose y_used_raw and GT y[t+1], grouped by arm/all_ex_root/leg.",
            "horizons": [int(h) for h in HORIZONS],
            "rescue_types": list(RESCUE_TYPES),
            "note": (
                "pose_history / plan_z are replaced with teacher-conditioned state once at the selected step before forward. "
                "h_final rescue uses a one-step forward hook on model.coupling_norm so that the selected step's downstream outputs are generated from teacher hidden, then freerun resumes normally."
            ),
        },
    }

    summary: Dict[str, Any] = {}
    for rescue_kind, window_map in rescue_windows.items():
        summary[rescue_kind] = {}
        for horizon in HORIZONS:
            truncated = {
                step_idx: seq[: int(horizon)]
                for step_idx, seq in window_map.items()
                if len(seq) >= int(horizon)
            }
            summary[rescue_kind][f"horizon_{int(horizon)}"] = {
                "windows": int(len(truncated)),
                "groups": {},
            }
            for group in GROUP_KEYS:
                base_vals: List[float] = []
                rescue_vals: List[float] = []
                improve_vals: List[float] = []
                positive = 0
                for step_idx, seq in truncated.items():
                    base_seq = baseline_errors[int(step_idx) : int(step_idx) + int(horizon)]
                    if len(base_seq) != len(seq):
                        continue
                    base_arr = _finite(row.get(group) for row in base_seq)
                    rescue_arr = _finite(row.get(group) for row in seq)
                    if base_arr.size <= 0 or rescue_arr.size <= 0:
                        continue
                    base_mean = float(np.mean(base_arr))
                    rescue_mean = float(np.mean(rescue_arr))
                    improve = float(base_mean - rescue_mean)
                    base_vals.append(base_mean)
                    rescue_vals.append(rescue_mean)
                    improve_vals.append(improve)
                    if improve > 0.0:
                        positive += 1
                summary[rescue_kind][f"horizon_{int(horizon)}"]["groups"][group] = {
                    "baseline_window_mean": _summary(_finite(base_vals)),
                    "rescued_window_mean": _summary(_finite(rescue_vals)),
                    "improvement_deg": _summary(_finite(improve_vals)),
                    "positive_rate": (
                        float(positive) / float(max(1, len(improve_vals))) if improve_vals else float("nan")
                    ),
                }

    payload["summary"] = summary
    payload["window_meta"] = [
        {
            "step_idx": int(meta.step_idx),
            "cycle": int(meta.cycle),
            "step_in_cycle": int(meta.step_in_cycle),
        }
        for meta in target_steps[:20]
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
