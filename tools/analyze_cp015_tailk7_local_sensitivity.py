#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_closed_loop_gap import (
    DEFAULT_BASELINE_CKPT,
    DEFAULT_BASELINE_EVAL,
    DEFAULT_TAIL_CKPT,
    DEFAULT_TAIL_EVAL,
    DEFAULT_TEACHER,
    GROUP_KEYS,
    _direct_local_geo_deg,
    _group_indices,
    _infer_bone_names,
    _load_case,
    _resolve_device,
    _safe_float,
    _summary,
)
from train.training_MPL import RolloutSequenceInputs


DEFAULT_OUT = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_closed_loop_stability_analysis_20260404"
    / "local_sensitivity.json"
)
INPUT_TARGETS: tuple[str, ...] = ("pose_history", "contacts_meas", "plan")


def _finite(vals: Iterable[Any]) -> np.ndarray:
    out: List[float] = []
    for v in vals:
        fv = _safe_float(v)
        if math.isfinite(fv):
            out.append(fv)
    return np.asarray(out, dtype=np.float64)


def _normalize_delta(shape: torch.Size, eps_l2: float, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    delta = torch.randn(shape, generator=gen, device=device, dtype=dtype)
    flat = delta.reshape(delta.shape[0], -1)
    norm = flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    flat = flat / norm
    flat = flat * float(eps_l2)
    return flat.reshape(shape)


def _group_mean_deg_between(
    *,
    trainer: Any,
    case: Mapping[str, Any],
    base_direct_norm: torch.Tensor,
    pert_direct_norm: torch.Tensor,
) -> Dict[str, float]:
    base_raw = trainer._denorm(base_direct_norm).detach().cpu()
    pert_raw = trainer._denorm(pert_direct_norm).detach().cpu()
    geo = _direct_local_geo_deg(
        pred_raw=pert_raw,
        gt_raw=base_raw,
        rot_slice=case["rot_slice"],
        root_idx=int(case["root_idx"]),
        columns=case["columns"],
    ).detach().cpu().numpy().astype(np.float64, copy=False)
    groups = case["groups"]
    out: Dict[str, float] = {}
    for key in GROUP_KEYS:
        idx = list(groups[key])
        vals = geo[:, np.asarray(idx, dtype=np.int64)].reshape(-1) if idx else np.asarray([], dtype=np.float64)
        out[key] = float(np.mean(vals)) if vals.size > 0 else float("nan")
    return out


def _step_inputs_for_rollout(
    trainer: Any,
    rollout: Any,
    rollout_inputs: RolloutSequenceInputs,
    step_idx: int,
) -> Any:
    return trainer._resolve_rollout_step_inputs(
        SimpleNamespace(
            step_idx=int(step_idx),
            total_steps=rollout.total_steps,
            motion=rollout.motion,
            motion_raw_local=rollout.motion_raw_local,
            y_raw_local=rollout.y_raw_local,
            state_seq=rollout_inputs.state_seq,
            gt_seq=rollout_inputs.gt_seq,
            cond_seq=rollout_inputs.cond_seq,
            cond_raw_seq=rollout_inputs.cond_raw_seq,
            contacts_seq=rollout_inputs.contacts_seq,
            angvel_seq=rollout_inputs.angvel_seq,
            pose_hist_seq=rollout_inputs.pose_hist_seq,
            cond_norm_mu=rollout.cond_norm_mu,
            cond_norm_std=rollout.cond_norm_std,
            has_time_dim=rollout.has_time_dim,
            pose_hist_state=rollout.pose_hist_state,
            plan_enable=rollout.plan_enable,
            mode=rollout.mode,
            enable_reprojection=rollout.enable_reprojection,
            time_base_local=rollout.time_base_local,
            prev_foot_pos_meas=rollout.prev_foot_pos_meas,
        )
    )


def _forward_direct(
    *,
    trainer: Any,
    rollout: Any,
    step_inputs: Any,
) -> Mapping[str, Any]:
    ret = trainer.model(
        rollout.motion,
        step_inputs.cond_input,
        contacts=step_inputs.contacts_in_t,
        angvel=step_inputs.angvel_t,
        pose_history=step_inputs.pose_history_t,
        plan_z=rollout.plan_z,
        meas_logits_prev=rollout.meas_prev_prob,
        time_index=step_inputs.time_index_t,
        rollout_step=step_inputs.rollout_step_t,
    )
    if not isinstance(ret, Mapping):
        raise RuntimeError("model forward must return a mapping")
    if not torch.is_tensor(ret.get("out_direct")):
        raise RuntimeError("model forward missing out_direct")
    return ret


def _run_case(case: Mapping[str, Any], eps_l2: float) -> Dict[str, Any]:
    trainer = case["trainer"]
    model = trainer.model
    if model is None:
        raise RuntimeError(f"{case['case_name']}: missing model")
    model.eval()

    batched = case["batched"]
    rollout_inputs = RolloutSequenceInputs(
        state_seq=batched["motion"],
        cond_seq=batched.get("cond_in"),
        cond_raw_seq=batched.get("cond_tgt_raw"),
        contacts_seq=batched.get("contacts"),
        angvel_seq=batched.get("angvel"),
        pose_hist_seq=batched.get("pose_hist"),
        gt_seq=batched["gt_motion"],
    )
    rollout = trainer._init_rollout_state(
        rollout_inputs,
        cond_norm_mu=batched.get("cond_norm_mu"),
        cond_norm_std=batched.get("cond_norm_std"),
        mode="mixed",
        tf_ratio=1.0,
        time_base=batched.get("start"),
    )

    agg: Dict[str, Dict[str, List[float]]] = {}
    for target in INPUT_TARGETS:
        agg[target] = {"input_l2": []}
        for group in GROUP_KEYS:
            agg[target][f"{group}_gain_deg_per_l2"] = []
            agg[target][f"{group}_delta_deg"] = []
    skip_counts = {target: 0 for target in INPUT_TARGETS}

    with torch.no_grad():
        for step_idx in range(int(rollout.total_steps)):
            step_inputs = _step_inputs_for_rollout(trainer, rollout, rollout_inputs, step_idx)
            base_ret = _forward_direct(trainer=trainer, rollout=rollout, step_inputs=step_inputs)
            base_direct = base_ret["out_direct"]
            base_plan = base_ret.get("contacts_plan")
            base_meas = base_ret.get("contacts_meas")
            base_pose_hist = step_inputs.pose_history_t

            def _do_perturb(
                *,
                target_name: str,
                target_tensor: torch.Tensor | None,
                use_override_attr: str | None,
                pose_history_override: torch.Tensor | None = None,
            ) -> None:
                if not torch.is_tensor(target_tensor) or target_tensor.numel() <= 0:
                    skip_counts[target_name] += 1
                    return
                delta = _normalize_delta(
                    target_tensor.shape,
                    eps_l2=float(eps_l2),
                    seed=10_000 * (1 + int(step_idx)) + hash(target_name) % 997,
                    device=target_tensor.device,
                    dtype=target_tensor.dtype,
                )
                base_input_l2 = float(delta.reshape(delta.shape[0], -1).norm(dim=-1).mean().item())
                if not math.isfinite(base_input_l2) or base_input_l2 <= 0.0:
                    skip_counts[target_name] += 1
                    return

                def _run_variant(sign: float) -> Mapping[str, Any]:
                    if use_override_attr is not None:
                        setattr(model, use_override_attr, target_tensor + float(sign) * delta)
                    ret = trainer.model(
                        rollout.motion,
                        step_inputs.cond_input,
                        contacts=step_inputs.contacts_in_t,
                        angvel=step_inputs.angvel_t,
                        pose_history=pose_history_override if pose_history_override is not None else step_inputs.pose_history_t,
                        plan_z=rollout.plan_z,
                        meas_logits_prev=rollout.meas_prev_prob,
                        time_index=step_inputs.time_index_t,
                        rollout_step=step_inputs.rollout_step_t,
                    )
                    if use_override_attr is not None:
                        setattr(model, use_override_attr, None)
                    return ret

                plus_pose = None
                minus_pose = None
                if pose_history_override is not None:
                    plus_pose = target_tensor + delta
                    minus_pose = target_tensor - delta

                plus_ret = _run_variant(sign=+1.0) if use_override_attr is not None else _forward_direct(
                    trainer=trainer,
                    rollout=rollout,
                    step_inputs=SimpleNamespace(
                        cond_input=step_inputs.cond_input,
                        contacts_in_t=step_inputs.contacts_in_t,
                        angvel_t=step_inputs.angvel_t,
                        pose_history_t=plus_pose,
                        time_index_t=step_inputs.time_index_t,
                        rollout_step_t=step_inputs.rollout_step_t,
                    ),
                )
                minus_ret = _run_variant(sign=-1.0) if use_override_attr is not None else _forward_direct(
                    trainer=trainer,
                    rollout=rollout,
                    step_inputs=SimpleNamespace(
                        cond_input=step_inputs.cond_input,
                        contacts_in_t=step_inputs.contacts_in_t,
                        angvel_t=step_inputs.angvel_t,
                        pose_history_t=minus_pose,
                        time_index_t=step_inputs.time_index_t,
                        rollout_step_t=step_inputs.rollout_step_t,
                    ),
                )

                plus_delta = _group_mean_deg_between(
                    trainer=trainer,
                    case=case,
                    base_direct_norm=base_direct,
                    pert_direct_norm=plus_ret["out_direct"],
                )
                minus_delta = _group_mean_deg_between(
                    trainer=trainer,
                    case=case,
                    base_direct_norm=base_direct,
                    pert_direct_norm=minus_ret["out_direct"],
                )
                agg[target_name]["input_l2"].append(float(base_input_l2))
                for group in GROUP_KEYS:
                    deg = 0.5 * (_safe_float(plus_delta[group]) + _safe_float(minus_delta[group]))
                    gain = deg / max(1e-8, float(base_input_l2))
                    agg[target_name][f"{group}_delta_deg"].append(float(deg))
                    agg[target_name][f"{group}_gain_deg_per_l2"].append(float(gain))

            _do_perturb(
                target_name="pose_history",
                target_tensor=base_pose_hist,
                use_override_attr=None,
                pose_history_override=base_pose_hist,
            )
            _do_perturb(
                target_name="contacts_meas",
                target_tensor=base_meas,
                use_override_attr="direct_pose_meas_override",
            )
            _do_perturb(
                target_name="plan",
                target_tensor=base_plan,
                use_override_attr="direct_pose_plan_override",
            )

            trainer._rollout_forward_step(rollout, rollout_inputs, step_idx=step_idx)
            trainer._apply_scheduled_sampling_update(rollout, rollout_inputs, step_idx=step_idx)

    report: Dict[str, Any] = {
        "case_name": case["case_name"],
        "ckpt_path": case["ckpt_path"],
        "teacher_path": case["teacher_path"],
        "eval_json_path": case["eval_json_path"],
        "runtime_overrides": case["runtime_overrides"],
        "sensitivity_definition": {
            "output": "teacher-working-point out_direct next-pose",
            "metric": "group mean local geodesic delta in degrees between perturbed and base out_direct",
            "gain": "0.5*(d_plus + d_minus) / ||delta_input||_2",
            "perturbation_l2": float(eps_l2),
            "targets": {
                "pose_history": "pose_history input actually fed to the model at each teacher-working-point rollout step",
                "contacts_meas": "direct-pose internal contacts_meas hint via model.direct_pose_meas_override; chosen because eval uses direct_pose_meas_source=model / contacts_meas_source=model",
                "plan": "direct-pose internal contacts_plan hint via model.direct_pose_plan_override; chosen because eval uses direct_pose_plan_source=model",
            },
        },
        "targets": {},
        "skip_counts": {k: int(v) for k, v in skip_counts.items()},
    }
    for target in INPUT_TARGETS:
        report["targets"][target] = {
            "input_l2": _summary(_finite(agg[target]["input_l2"])),
            "groups": {},
        }
        for group in GROUP_KEYS:
            report["targets"][target]["groups"][group] = {
                "delta_deg": _summary(_finite(agg[target][f"{group}_delta_deg"])),
                "gain_deg_per_l2": _summary(_finite(agg[target][f"{group}_gain_deg_per_l2"])),
            }
    return report


def _compare(case_a: Mapping[str, Any], case_b: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for target in INPUT_TARGETS:
        out[target] = {}
        for group in GROUP_KEYS:
            a_targets = case_a.get("targets", {}) or {}
            b_targets = case_b.get("targets", {}) or {}
            a_target = a_targets.get(target, {}) or {}
            b_target = b_targets.get(target, {}) or {}
            a_groups = a_target.get("groups", {}) or {}
            b_groups = b_target.get("groups", {}) or {}
            a_group = (a_groups.get(group, {}) or {})
            b_group = (b_groups.get(group, {}) or {})
            a_gain_stats = (a_group.get("gain_deg_per_l2", {}) or {})
            b_gain_stats = (b_group.get("gain_deg_per_l2", {}) or {})
            a_gain = _safe_float(a_gain_stats.get("mean"))
            b_gain = _safe_float(b_gain_stats.get("mean"))
            a_p95 = _safe_float(a_gain_stats.get("p95"))
            b_p95 = _safe_float(b_gain_stats.get("p95"))
            out[target][group] = {
                "tailk7_factorized_gain_mean": a_gain,
                "baseline_replace_gain_mean": b_gain,
                "delta_tail_minus_base_gain_mean": a_gain - b_gain,
                "tailk7_factorized_gain_p95": a_p95,
                "baseline_replace_gain_p95": b_p95,
                "delta_tail_minus_base_gain_p95": a_p95 - b_p95,
            }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal local sensitivity audit for cp015 tailk7 replace.")
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--tail-ckpt", type=Path, default=DEFAULT_TAIL_CKPT)
    ap.add_argument("--tail-eval", type=Path, default=DEFAULT_TAIL_EVAL)
    ap.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    ap.add_argument("--eps-l2", type=float, default=0.05)
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    teacher = args.teacher.expanduser().resolve()
    tail_ckpt = args.tail_ckpt.expanduser().resolve()
    tail_eval = args.tail_eval.expanduser().resolve()
    base_ckpt = args.baseline_ckpt.expanduser().resolve()
    base_eval = args.baseline_eval.expanduser().resolve()
    out_path = args.out.expanduser().resolve()

    for path in (teacher, tail_ckpt, tail_eval, base_ckpt, base_eval):
        if not path.is_file():
            raise SystemExit(f"[FATAL] missing input: {path}")

    tail_case = _load_case(
        case_name="tailk7_factorized",
        ckpt_path=tail_ckpt,
        eval_json_path=tail_eval,
        teacher_path=teacher,
        device_pref=str(args.device),
    )
    base_case = _load_case(
        case_name="baseline_replace",
        ckpt_path=base_ckpt,
        eval_json_path=base_eval,
        teacher_path=teacher,
        device_pref=str(args.device),
    )

    tail_report = _run_case(tail_case, eps_l2=float(args.eps_l2))
    base_report = _run_case(base_case, eps_l2=float(args.eps_l2))
    payload = {
        "analysis": "local_input_sensitivity",
        "teacher_batch": str(teacher),
        "cases": {
            "tailk7_factorized": tail_report,
            "baseline_replace": base_report,
        },
        "comparison": _compare(tail_report, base_report),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
