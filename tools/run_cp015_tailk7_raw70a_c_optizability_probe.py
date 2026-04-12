#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from run_cp015_oldplan_downstream_chain import safe_float, write_json
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import safe_float, write_json

try:
    import run_cp015_tailk7_replace_efficiency_audit as effprobe
except ModuleNotFoundError:
    from tools import run_cp015_tailk7_replace_efficiency_audit as effprobe

from tools.audit_cp015_tailk7_plan_shortcut_takeover_mechanism import _branch_layout, _first_linear
from train import posttrain as posttrain_mod


RUN_DATE = "20260407"
RUN_NAME = "cp015_tailk7_raw70a_c_optizability_probe"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_{RUN_NAME}_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
LOG_FILE = OUT_ROOT / "probe.log"

TEMPLATE_CFG = effprobe.PROBE_TEMPLATE_CONFIG


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    label: str
    stage_type: str
    ckpt: Path
    purpose: str


CANDIDATES: Tuple[CandidateSpec, ...] = (
    CandidateSpec(
        key="baseline_raw_70a",
        label="baseline raw 70a",
        stage_type="raw70a",
        ckpt=ROOT
        / "models"
        / "__tmp_posttrain_pipeline_from_bestfree_20260317"
        / "70a"
        / "ckpt_last_WalkF_stage7_70a_fromfresh_20260317.pth",
        purpose="baseline raw 70a reference；检验 step0/1 是否更容易被一步更新推向 non-plan regime",
    ),
    CandidateSpec(
        key="tailk7_raw_70a",
        label="tailk7 raw 70a",
        stage_type="raw70a",
        ckpt=ROOT
        / "models"
        / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
        / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth",
        purpose="C 主 probe；检验 raw 70a donor-state 在 replace step0/1 上是否 less optizable",
    ),
    CandidateSpec(
        key="tailk7_baseline_style_adapted_warmstart",
        label="tailk7 baseline-style adapted warmstart",
        stage_type="warmstart/baseline-style adapted",
        ckpt=ROOT
        / "models"
        / "__tmp_cp015_tailk7_warmstart_contract_sentinel_20260402_warmstart_contract_sentinel"
        / "warmstart"
        / "ckpt_last_cp015_tailk7_70a_replace_baseline_style_20260402_warmstart_contract_sentinel.pth",
        purpose="warmstart sentinel；检验 baseline-style surgery 是否足以改善一步 non-plan optizability",
    ),
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{ts}] {msg}"
    print(text, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def fmt(x: Any, digits: int = 6) -> str:
    v = safe_float(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def sgn(x: Any, digits: int = 6) -> str:
    v = safe_float(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:+.{digits}f}"


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required artifact(s):\n" + "\n".join(missing))


def set_probe_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@contextmanager
def zero_input_slice(linear: torch.nn.Linear, zero_slice: Optional[slice]):
    if zero_slice is None:
        yield
        return

    def _pre_hook(_mod: torch.nn.Module, inputs: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
        if not inputs or not torch.is_tensor(inputs[0]):
            return None
        x = inputs[0]
        x_use = x.clone()
        x_use[..., zero_slice] = 0
        return (x_use, *inputs[1:])

    hook = linear.register_forward_pre_hook(_pre_hook)
    try:
        yield
    finally:
        hook.remove()


def _slice_payload(sl: slice) -> List[int]:
    return [int(sl.start or 0), int(sl.stop or 0)]


def _weight_or_grad_block_stats(weight: torch.Tensor, grad: torch.Tensor, sl: slice) -> Dict[str, float]:
    w_block = weight[:, sl].detach().cpu().float()
    g_block = grad[:, sl].detach().cpu().float()
    width = int(w_block.shape[1])
    dim_scale = math.sqrt(float(width)) if width > 0 else float("nan")
    w_fro = float(w_block.norm().item()) if w_block.numel() > 0 else float("nan")
    g_fro = float(g_block.norm().item()) if g_block.numel() > 0 else float("nan")
    cosine = float("nan")
    signed_proj = float("nan")
    signed_proj_per_dim = float("nan")
    if w_block.numel() > 0 and g_block.numel() > 0:
        dot = float((w_block.reshape(-1) * g_block.reshape(-1)).sum().item())
        w_norm = float(w_block.norm().item())
        g_norm = float(g_block.norm().item())
        if w_norm > 1e-12 and g_norm > 1e-12:
            cosine = float(dot / (w_norm * g_norm))
            signed_proj = float(dot / w_norm)
            if dim_scale > 0.0:
                signed_proj_per_dim = float(signed_proj / dim_scale)
    return {
        "width": float(width),
        "weight_fro": w_fro,
        "weight_fro_per_dim": float(w_fro / dim_scale) if dim_scale > 0.0 else float("nan"),
        "grad_fro": g_fro,
        "grad_fro_per_dim": float(g_fro / dim_scale) if dim_scale > 0.0 else float("nan"),
        "grad_weight_cosine": cosine,
        "grad_signed_projection_on_weight": signed_proj,
        "grad_signed_projection_on_weight_per_dim": signed_proj_per_dim,
    }


def build_optimizer(model: torch.nn.Module, cfg: Any) -> Tuple[List[torch.nn.Parameter], List[str], torch.optim.Optimizer]:
    train_mode = posttrain_mod._resolve_train_mode(cfg)
    params, names = posttrain_mod._select_trainable_params(model)
    if not params:
        raise RuntimeError("no trainable params selected")
    overrides = posttrain_mod._combined_optimizer_param_group_overrides(cfg=cfg, model=model, train_mode=train_mode)
    param_groups, _summaries = posttrain_mod._resolve_optimizer_param_groups(
        cfg=cfg,
        params=params,
        names=names,
        overrides=overrides,
    )
    if param_groups is None:
        opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    else:
        opt = torch.optim.AdamW(param_groups, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    return params, names, opt


def run_rollout(
    *,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    rollout_common_kwargs: Dict[str, Any],
    rollout_mode_kwargs: Dict[str, Any],
    linear: torch.nn.Linear,
    zero_slice: Optional[slice],
    seed: int,
    with_grad: bool,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
    set_probe_seed(seed)
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with zero_input_slice(linear, zero_slice), ctx:
        loss, stats, aux = effprobe._lambda_fusion_loss_rollout(
            batch=batch,
            **rollout_common_kwargs,
            **rollout_mode_kwargs,
        )
    stats_out = {str(key): safe_float(value) for key, value in stats.items()}
    return loss, stats_out, aux


def collect_eval_metrics(
    *,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    rollout_common_kwargs: Dict[str, Any],
    rollout_mode_kwargs: Dict[str, Any],
    linear: torch.nn.Linear,
    zero_plan: bool,
    seed: int,
) -> Dict[str, float]:
    loss, stats, _aux = run_rollout(
        model=model,
        batch=batch,
        rollout_common_kwargs=rollout_common_kwargs,
        rollout_mode_kwargs=rollout_mode_kwargs,
        linear=linear,
        zero_slice=_branch_layout(model, linear).plan if zero_plan else None,
        seed=seed,
        with_grad=False,
    )
    out = {
        "loss": float(loss.detach().cpu()),
        "total": safe_float(stats.get("total")),
        "dir_arm_base": safe_float(stats.get("dir_arm_base")),
        "dir_else_base": safe_float(stats.get("dir_else_base")),
        "dir_leg_base": safe_float(stats.get("dir_leg_base")),
    }
    return out


def collect_grad_snapshot(
    *,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    rollout_common_kwargs: Dict[str, Any],
    rollout_mode_kwargs: Dict[str, Any],
    linear: torch.nn.Linear,
    params: Sequence[torch.nn.Parameter],
    seed: int,
) -> Dict[str, Any]:
    model.zero_grad(set_to_none=True)
    loss, stats, _aux = run_rollout(
        model=model,
        batch=batch,
        rollout_common_kwargs=rollout_common_kwargs,
        rollout_mode_kwargs=rollout_mode_kwargs,
        linear=linear,
        zero_slice=None,
        seed=seed,
        with_grad=True,
    )
    loss.backward()
    grad = getattr(linear.weight, "grad", None)
    if grad is None:
        raise RuntimeError("direct_pose_head first linear has no grad")
    layout = _branch_layout(model, linear)
    weight = linear.weight.detach()
    block_stats = {
        "direct_feat": _weight_or_grad_block_stats(weight, grad, layout.direct),
        "plan": _weight_or_grad_block_stats(weight, grad, layout.plan),
        "meas": _weight_or_grad_block_stats(weight, grad, layout.meas),
    }
    direct_g = safe_float(block_stats["direct_feat"]["grad_fro_per_dim"])
    plan_g = safe_float(block_stats["plan"]["grad_fro_per_dim"])
    meas_g = safe_float(block_stats["meas"]["grad_fro_per_dim"])
    return {
        "loss": float(loss.detach().cpu()),
        "stats": {key: safe_float(value) for key, value in stats.items()},
        "grad_norms": {
            "shared_trunk": effprobe.tensor_norms_by_prefix(model, effprobe.MODULE_GROUPS["shared_trunk"]),
            "arm_branch": effprobe.tensor_norms_by_prefix(model, effprobe.MODULE_GROUPS["arm_branch"]),
            "else_branch": effprobe.tensor_norms_by_prefix(model, effprobe.MODULE_GROUPS["else_branch"]),
            "leg_branch": effprobe.tensor_norms_by_prefix(model, effprobe.MODULE_GROUPS["leg_branch"]),
        },
        "blocks": block_stats,
        "ratios": {
            "plan_over_direct_per_dim": float(plan_g / direct_g) if math.isfinite(plan_g) and math.isfinite(direct_g) and abs(direct_g) > 1e-12 else float("nan"),
            "direct_over_plan_per_dim": float(direct_g / plan_g) if math.isfinite(plan_g) and math.isfinite(direct_g) and abs(plan_g) > 1e-12 else float("nan"),
            "meas_over_direct_per_dim": float(meas_g / direct_g) if math.isfinite(meas_g) and math.isfinite(direct_g) and abs(direct_g) > 1e-12 else float("nan"),
        },
    }


def block_update_stats(before: torch.Tensor, after: torch.Tensor, sl: slice) -> Dict[str, float]:
    delta = (after[:, sl] - before[:, sl]).detach().cpu().float()
    width = int(delta.shape[1])
    dim_scale = math.sqrt(float(width)) if width > 0 else float("nan")
    delta_fro = float(delta.norm().item()) if delta.numel() > 0 else float("nan")
    return {
        "delta_fro": delta_fro,
        "delta_fro_per_dim": float(delta_fro / dim_scale) if dim_scale > 0.0 else float("nan"),
    }


def derive_summary(
    *,
    step0_model: Mapping[str, Any],
    step0_zero: Mapping[str, Any],
    step1_model: Mapping[str, Any],
    step1_zero: Mapping[str, Any],
    step0_grad: Mapping[str, Any],
    step1_grad: Mapping[str, Any],
    update_blocks: Mapping[str, Any],
) -> Dict[str, Any]:
    model_impr = safe_float(step0_model.get("dir_arm_base")) - safe_float(step1_model.get("dir_arm_base"))
    zero_impr = safe_float(step0_zero.get("dir_arm_base")) - safe_float(step1_zero.get("dir_arm_base"))
    model_leg_impr = safe_float(step0_model.get("dir_leg_base")) - safe_float(step1_model.get("dir_leg_base"))
    zero_leg_impr = safe_float(step0_zero.get("dir_leg_base")) - safe_float(step1_zero.get("dir_leg_base"))
    step0_gap = safe_float(step0_zero.get("dir_arm_base")) - safe_float(step0_model.get("dir_arm_base"))
    step1_gap = safe_float(step1_zero.get("dir_arm_base")) - safe_float(step1_model.get("dir_arm_base"))
    gap_shrink = step0_gap - step1_gap
    label = "mixed"
    if zero_impr > 0.0 and gap_shrink > 0.0:
        label = "moves toward non-plan basin"
    elif zero_impr <= 0.0 or gap_shrink <= 0.0:
        label = "does not move toward non-plan basin"
    return {
        "dir_arm_model_improvement": model_impr,
        "dir_arm_zero_plan_improvement": zero_impr,
        "dir_leg_model_improvement": model_leg_impr,
        "dir_leg_zero_plan_improvement": zero_leg_impr,
        "dir_arm_plan_gap_step0": step0_gap,
        "dir_arm_plan_gap_step1": step1_gap,
        "dir_arm_plan_gap_shrink": gap_shrink,
        "step0_plan_over_direct_grad": safe_float((step0_grad.get("ratios") or {}).get("plan_over_direct_per_dim")),
        "step1_plan_over_direct_grad": safe_float((step1_grad.get("ratios") or {}).get("plan_over_direct_per_dim")),
        "plan_update_over_direct_update": float(
            safe_float((update_blocks.get("plan") or {}).get("delta_fro_per_dim"))
            / safe_float((update_blocks.get("direct_feat") or {}).get("delta_fro_per_dim"))
        )
        if math.isfinite(safe_float((update_blocks.get("plan") or {}).get("delta_fro_per_dim")))
        and math.isfinite(safe_float((update_blocks.get("direct_feat") or {}).get("delta_fro_per_dim")))
        and abs(safe_float((update_blocks.get("direct_feat") or {}).get("delta_fro_per_dim"))) > 1e-12
        else float("nan"),
        "label": label,
    }


def pairwise_grad_cos(lhs: Mapping[str, Any], rhs: Mapping[str, Any], block: str) -> float:
    lhs_block = (((lhs.get("blocks") or {}).get(block) or {}))
    rhs_block = (((rhs.get("blocks") or {}).get(block) or {}))
    lhs_cos = safe_float(lhs_block.get("grad_signed_projection_on_weight"))
    rhs_cos = safe_float(rhs_block.get("grad_signed_projection_on_weight"))
    if not math.isfinite(lhs_cos) or not math.isfinite(rhs_cos):
        return float("nan")
    return float("nan")


def run_candidate(spec: CandidateSpec) -> Dict[str, Any]:
    log(f"probe start: {spec.label}")
    ctx = effprobe.build_rollout_context(TEMPLATE_CFG, spec.ckpt)
    cfg = ctx["cfg"]
    model = ctx["model"]
    batch = ctx["batch"]
    rollout_common_kwargs = ctx["rollout_common_kwargs"]
    rollout_mode_kwargs = ctx["rollout_mode_kwargs"]
    _module_name, linear = _first_linear(model)
    layout = _branch_layout(model, linear)
    params, names, opt = build_optimizer(model, cfg)
    base_seed = int(getattr(cfg, "seed", 2024) or 2024)

    step0_model = collect_eval_metrics(
        model=model,
        batch=batch,
        rollout_common_kwargs=rollout_common_kwargs,
        rollout_mode_kwargs=rollout_mode_kwargs,
        linear=linear,
        zero_plan=False,
        seed=base_seed,
    )
    step0_zero = collect_eval_metrics(
        model=model,
        batch=batch,
        rollout_common_kwargs=rollout_common_kwargs,
        rollout_mode_kwargs=rollout_mode_kwargs,
        linear=linear,
        zero_plan=True,
        seed=base_seed,
    )
    first_weight_before = linear.weight.detach().clone()
    step0_grad = collect_grad_snapshot(
        model=model,
        batch=batch,
        rollout_common_kwargs=rollout_common_kwargs,
        rollout_mode_kwargs=rollout_mode_kwargs,
        linear=linear,
        params=params,
        seed=base_seed + 1,
    )
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    opt.step()
    model.zero_grad(set_to_none=True)

    step1_model = collect_eval_metrics(
        model=model,
        batch=batch,
        rollout_common_kwargs=rollout_common_kwargs,
        rollout_mode_kwargs=rollout_mode_kwargs,
        linear=linear,
        zero_plan=False,
        seed=base_seed,
    )
    step1_zero = collect_eval_metrics(
        model=model,
        batch=batch,
        rollout_common_kwargs=rollout_common_kwargs,
        rollout_mode_kwargs=rollout_mode_kwargs,
        linear=linear,
        zero_plan=True,
        seed=base_seed,
    )
    step1_grad = collect_grad_snapshot(
        model=model,
        batch=batch,
        rollout_common_kwargs=rollout_common_kwargs,
        rollout_mode_kwargs=rollout_mode_kwargs,
        linear=linear,
        params=params,
        seed=base_seed + 1,
    )
    first_weight_after = linear.weight.detach().clone()
    update_blocks = {
        "direct_feat": block_update_stats(first_weight_before, first_weight_after, layout.direct),
        "plan": block_update_stats(first_weight_before, first_weight_after, layout.plan),
        "meas": block_update_stats(first_weight_before, first_weight_after, layout.meas),
    }
    derived = derive_summary(
        step0_model=step0_model,
        step0_zero=step0_zero,
        step1_model=step1_model,
        step1_zero=step1_zero,
        step0_grad=step0_grad,
        step1_grad=step1_grad,
        update_blocks=update_blocks,
    )
    payload = {
        "candidate": spec.label,
        "candidate_key": spec.key,
        "stage_type": spec.stage_type,
        "checkpoint": str(spec.ckpt),
        "purpose": spec.purpose,
        "template_cfg": str(TEMPLATE_CFG),
        "train_mode": posttrain_mod._resolve_train_mode(cfg),
        "trainable_param_count": int(len(params)),
        "trainable_param_names_preview": list(names[:12]),
        "branch_layout": {
            "direct_feat": _slice_payload(layout.direct),
            "plan": _slice_payload(layout.plan),
            "meas": _slice_payload(layout.meas),
            "direct_feat_dim": int(layout.direct_dim),
            "plan_dim": int(layout.plan_dim),
            "meas_dim": int(layout.meas_dim),
            "total_dim": int(layout.total_dim),
        },
        "step0": {
            "model": step0_model,
            "zero_plan": step0_zero,
            "grad": step0_grad,
        },
        "step1": {
            "model": step1_model,
            "zero_plan": step1_zero,
            "grad": step1_grad,
        },
        "update_blocks": update_blocks,
        "derived": derived,
    }
    detail_path = OUT_ROOT / "candidates" / f"{spec.key}.json"
    write_json(detail_path, payload)
    payload["detail_artifact"] = str(detail_path)
    log(
        f"probe done: {spec.label} "
        f"zero-plan arm_impr={fmt(derived['dir_arm_zero_plan_improvement'])} "
        f"gap_shrink={fmt(derived['dir_arm_plan_gap_shrink'])}"
    )
    return payload


def build_tables(candidate_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    step0_rows: List[Dict[str, Any]] = []
    step1_rows: List[Dict[str, Any]] = []
    one_step_rows: List[Dict[str, Any]] = []
    for row in candidate_rows:
        step0_grad = (((row.get("step0") or {}).get("grad") or {}))
        step1_grad = (((row.get("step1") or {}).get("grad") or {}))
        derived = row.get("derived") or {}
        update_blocks = row.get("update_blocks") or {}
        step0_rows.append(
            {
                "candidate": row.get("candidate"),
                "direct_grad_per_dim": safe_float((((step0_grad.get("blocks") or {}).get("direct_feat") or {}).get("grad_fro_per_dim"))),
                "plan_grad_per_dim": safe_float((((step0_grad.get("blocks") or {}).get("plan") or {}).get("grad_fro_per_dim"))),
                "meas_grad_per_dim": safe_float((((step0_grad.get("blocks") or {}).get("meas") or {}).get("grad_fro_per_dim"))),
                "plan_over_direct": safe_float((step0_grad.get("ratios") or {}).get("plan_over_direct_per_dim")),
                "plan_grad_signed_proj_per_dim": safe_float((((step0_grad.get("blocks") or {}).get("plan") or {}).get("grad_signed_projection_on_weight_per_dim"))),
                "direct_grad_signed_proj_per_dim": safe_float((((step0_grad.get("blocks") or {}).get("direct_feat") or {}).get("grad_signed_projection_on_weight_per_dim"))),
            }
        )
        step1_rows.append(
            {
                "candidate": row.get("candidate"),
                "direct_grad_per_dim": safe_float((((step1_grad.get("blocks") or {}).get("direct_feat") or {}).get("grad_fro_per_dim"))),
                "plan_grad_per_dim": safe_float((((step1_grad.get("blocks") or {}).get("plan") or {}).get("grad_fro_per_dim"))),
                "meas_grad_per_dim": safe_float((((step1_grad.get("blocks") or {}).get("meas") or {}).get("grad_fro_per_dim"))),
                "plan_over_direct": safe_float((step1_grad.get("ratios") or {}).get("plan_over_direct_per_dim")),
                "plan_grad_signed_proj_per_dim": safe_float((((step1_grad.get("blocks") or {}).get("plan") or {}).get("grad_signed_projection_on_weight_per_dim"))),
                "direct_grad_signed_proj_per_dim": safe_float((((step1_grad.get("blocks") or {}).get("direct_feat") or {}).get("grad_signed_projection_on_weight_per_dim"))),
            }
        )
        one_step_rows.append(
            {
                "candidate": row.get("candidate"),
                "step0_model_arm": safe_float((((row.get("step0") or {}).get("model") or {}).get("dir_arm_base"))),
                "step1_model_arm": safe_float((((row.get("step1") or {}).get("model") or {}).get("dir_arm_base"))),
                "model_arm_impr": safe_float(derived.get("dir_arm_model_improvement")),
                "step0_zero_arm": safe_float((((row.get("step0") or {}).get("zero_plan") or {}).get("dir_arm_base"))),
                "step1_zero_arm": safe_float((((row.get("step1") or {}).get("zero_plan") or {}).get("dir_arm_base"))),
                "zero_arm_impr": safe_float(derived.get("dir_arm_zero_plan_improvement")),
                "step0_gap": safe_float(derived.get("dir_arm_plan_gap_step0")),
                "step1_gap": safe_float(derived.get("dir_arm_plan_gap_step1")),
                "gap_shrink": safe_float(derived.get("dir_arm_plan_gap_shrink")),
                "plan_update_over_direct_update": safe_float(derived.get("plan_update_over_direct_update")),
                "label": row.get("derived", {}).get("label"),
                "detail_artifact": row.get("detail_artifact"),
                "update_blocks": update_blocks,
            }
        )
    return {
        "step0_grad_rows": step0_rows,
        "step1_grad_rows": step1_rows,
        "one_step_rows": one_step_rows,
    }


def build_conclusion(candidate_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_key = {str(row.get("candidate_key")): row for row in candidate_rows}
    base = by_key.get("baseline_raw_70a", {})
    tail = by_key.get("tailk7_raw_70a", {})
    adapt = by_key.get("tailk7_baseline_style_adapted_warmstart", {})
    base_d = base.get("derived") or {}
    tail_d = tail.get("derived") or {}
    adapt_d = adapt.get("derived") or {}
    return {
        "baseline_zero_plan_one_step_arm_improvement": safe_float(base_d.get("dir_arm_zero_plan_improvement")),
        "tailk7_zero_plan_one_step_arm_improvement": safe_float(tail_d.get("dir_arm_zero_plan_improvement")),
        "adapted_zero_plan_one_step_arm_improvement": safe_float(adapt_d.get("dir_arm_zero_plan_improvement")),
        "baseline_gap_shrink": safe_float(base_d.get("dir_arm_plan_gap_shrink")),
        "tailk7_gap_shrink": safe_float(tail_d.get("dir_arm_plan_gap_shrink")),
        "adapted_gap_shrink": safe_float(adapt_d.get("dir_arm_plan_gap_shrink")),
        "baseline_step0_plan_over_direct_grad": safe_float(base_d.get("step0_plan_over_direct_grad")),
        "tailk7_step0_plan_over_direct_grad": safe_float(tail_d.get("step0_plan_over_direct_grad")),
        "adapted_step0_plan_over_direct_grad": safe_float(adapt_d.get("step0_plan_over_direct_grad")),
        "supports_c_as_optizability_gap": bool(
            safe_float(base_d.get("dir_arm_zero_plan_improvement")) > safe_float(tail_d.get("dir_arm_zero_plan_improvement"))
            and safe_float(base_d.get("dir_arm_plan_gap_shrink")) > safe_float(tail_d.get("dir_arm_plan_gap_shrink"))
        ),
        "supports_warmstart_not_main_cause": bool(
            safe_float(adapt_d.get("dir_arm_zero_plan_improvement")) <= safe_float(base_d.get("dir_arm_zero_plan_improvement"))
            or safe_float(adapt_d.get("dir_arm_plan_gap_shrink")) <= safe_float(base_d.get("dir_arm_plan_gap_shrink"))
        ),
    }


def render_md(payload: Mapping[str, Any]) -> str:
    step0_rows = list(payload.get("tables", {}).get("step0_grad_rows", []))
    step1_rows = list(payload.get("tables", {}).get("step1_grad_rows", []))
    one_step_rows = list(payload.get("tables", {}).get("one_step_rows", []))
    conc = payload.get("conclusion") or {}
    lines: List[str] = [
        "# cp015 tailk7 raw70a C-only optizability probe",
        "",
        "## Findings",
        "",
        f"- template cfg: `{payload.get('template_cfg')}`",
        "- same-batch / same-loss / same-optimizer one-step probe only; no new 60-step or 240-step lane",
        f"- supports `C as optizability gap`: `{bool(conc.get('supports_c_as_optizability_gap'))}`",
        f"- supports `warmstart not main cause`: `{bool(conc.get('supports_warmstart_not_main_cause'))}`",
        "",
        "## Step0 Gradient Composition",
        "",
        "| candidate | direct_grad/dim | plan_grad/dim | meas_grad/dim | plan/direct | direct_signed_proj/dim | plan_signed_proj/dim |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in step0_rows:
        lines.append(
            f"| {row['candidate']} | {fmt(row['direct_grad_per_dim'])} | {fmt(row['plan_grad_per_dim'])} | "
            f"{fmt(row['meas_grad_per_dim'])} | {fmt(row['plan_over_direct'])} | "
            f"{sgn(row['direct_grad_signed_proj_per_dim'])} | {sgn(row['plan_grad_signed_proj_per_dim'])} |"
        )
    lines.extend(
        [
            "",
            "## Step1 Gradient Composition",
            "",
            "| candidate | direct_grad/dim | plan_grad/dim | meas_grad/dim | plan/direct | direct_signed_proj/dim | plan_signed_proj/dim |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in step1_rows:
        lines.append(
            f"| {row['candidate']} | {fmt(row['direct_grad_per_dim'])} | {fmt(row['plan_grad_per_dim'])} | "
            f"{fmt(row['meas_grad_per_dim'])} | {fmt(row['plan_over_direct'])} | "
            f"{sgn(row['direct_grad_signed_proj_per_dim'])} | {sgn(row['plan_grad_signed_proj_per_dim'])} |"
        )
    lines.extend(
        [
            "",
            "## One-step Non-plan Optizability",
            "",
            "| candidate | step0 model arm | step1 model arm | model impr | step0 zero-plan arm | step1 zero-plan arm | zero-plan impr | step0 gap | step1 gap | gap shrink | plan_update/direct_update | label |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in one_step_rows:
        lines.append(
            f"| {row['candidate']} | {fmt(row['step0_model_arm'])} | {fmt(row['step1_model_arm'])} | {sgn(row['model_arm_impr'])} | "
            f"{fmt(row['step0_zero_arm'])} | {fmt(row['step1_zero_arm'])} | {sgn(row['zero_arm_impr'])} | "
            f"{fmt(row['step0_gap'])} | {fmt(row['step1_gap'])} | {sgn(row['gap_shrink'])} | "
            f"{fmt(row['plan_update_over_direct_update'])} | {row['label']} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- baseline zero-plan one-step arm improvement: `{fmt(conc.get('baseline_zero_plan_one_step_arm_improvement'))}`",
            f"- tailk7 zero-plan one-step arm improvement: `{fmt(conc.get('tailk7_zero_plan_one_step_arm_improvement'))}`",
            f"- adapted zero-plan one-step arm improvement: `{fmt(conc.get('adapted_zero_plan_one_step_arm_improvement'))}`",
            f"- baseline gap shrink: `{sgn(conc.get('baseline_gap_shrink'))}`",
            f"- tailk7 gap shrink: `{sgn(conc.get('tailk7_gap_shrink'))}`",
            f"- adapted gap shrink: `{sgn(conc.get('adapted_gap_shrink'))}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    require_files([TEMPLATE_CFG, *(spec.ckpt for spec in CANDIDATES)])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_rows: List[Dict[str, Any]] = []
    for spec in CANDIDATES:
        candidate_rows.append(run_candidate(spec))
    tables = build_tables(candidate_rows)
    conclusion = build_conclusion(candidate_rows)
    payload = {
        "run_date": RUN_DATE,
        "run_name": RUN_NAME,
        "out_root": str(OUT_ROOT),
        "template_cfg": str(TEMPLATE_CFG),
        "candidates": candidate_rows,
        "tables": tables,
        "conclusion": conclusion,
    }
    write_json(SUMMARY_JSON, payload)
    SUMMARY_MD.write_text(render_md(payload), encoding="utf-8")
    log(f"wrote {SUMMARY_JSON}")
    log(f"wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
