#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from run_cp015_oldplan_downstream_chain import load_json, safe_float, write_json
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import load_json, safe_float, write_json

from train import posttrain


RUN_DATE = "20260409"
RUN_NAME = "replace_handoff_distribution_pathology"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_{RUN_NAME}_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"
DOC_PATH = ROOT / "docs" / "train_design" / "2026-04-09_replace_handoff_distribution_pathology_probe.md"

STAGE70A_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_ep014center_70a_lowlr_sweep_20260328"
    / "configs"
    / "posttrain_70a_lr3e4_from_ep014center_20260328.json"
)
STAGE70B_REPLACE_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_ep014center_replace_lowlr_sweep_20260328"
    / "configs"
    / "posttrain_70b_replace_lowdrift_lr5e5_from_ep014center_20260328.json"
)

DEFAULT_N_BATCHES = 32
TOP_K = 8
CHANNEL_AXIS = -1

ABS_DEAD_STD = 1e-6
ABS_DEAD_SPAN = 1e-5
ABS_DEAD_MAD = 1e-6
REL_DEAD_FACTOR = 0.10
HEAVY_TAIL_EXCESS_KURTOSIS = 10.0
SCALE_OUTLIER_RATIO = 5.0
EPS = 1e-12


INHERITED_CONCLUSIONS = {
    "a1_mainline": [
        "root cause 不在 planner semantics 主线",
        "root cause 不在 replace entry 外部 rollout state",
        "root cause 不在 contacts_in_t",
        "earliest semantic split 在 direct_pose_head boundary",
        "direct_pose_head 是 earliest boundary / necessary anchor，但不是 standalone sufficient",
        "normality probe 在 A1 口径下 non-discriminative，不要当主判据",
        "A1-S5: donor family 的 direct_pose_head output 与 target/baseline manifold 有大幅 drift",
        "A1-S5: donor-vs-donor divergence 很小，但 donor-vs-target divergence 很大",
        "A1-S5: affine guard 对 replace aggregate 无明显帮助",
    ],
    "notail_falsifier": [
        "tail-k 不是 representation drift 的必要原因",
        "notail 的 head-output cosdist to target 仍约 0.785",
        "notail native direct/freerun 指标优于 E1-top3",
        "当前更像 donor-family / 70a->70b handoff 机制问题，而不是 tail-k 特有问题",
    ],
    "replace_closed_loop_falsifier": [
        "step0/1 local optizability 版本的解释已基本不支持",
        "当前更像 initial loss profile -> group-norm EMA seed -> rollout feedback",
        "本轮待答：是什么 distribution shape 把 initial loss profile 弄歪",
    ],
}


@dataclass(frozen=True)
class ArmSpec:
    key: str
    ckpt: Path
    label: str
    optional: bool = False


@dataclass(frozen=True)
class HookSpec:
    key: str
    module: str
    kind: str
    label: str
    reason: str
    fallback_modules: Tuple[str, ...] = ()


ARM_SPECS: Tuple[ArmSpec, ...] = (
    ArmSpec(
        key="baseline_raw70a",
        label="baseline-raw70a",
        ckpt=ROOT
        / "models"
        / "__tmp_posttrain_pipeline_from_bestfree_20260317"
        / "70a"
        / "ckpt_last_WalkF_stage7_70a_fromfresh_20260317.pth",
    ),
    ArmSpec(
        key="e1_top3_raw70a",
        label="E1-top3-raw70a",
        ckpt=ROOT
        / "models"
        / "__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408"
        / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth",
    ),
    ArmSpec(
        key="notail_raw70a",
        label="notail-raw70a",
        ckpt=ROOT
        / "models"
        / "__tmp_cp015_notail_stage70a_from_tailfix_20260409"
        / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_notail_stage6tailfix_20260409.pth",
    ),
    ArmSpec(
        key="e2a_r_raw70a",
        label="E2A-R-raw70a",
        ckpt=ROOT
        / "models"
        / "__tmp_cp015_tailk357ramp_stage70a_from_tailfix_e2a_20260408"
        / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk357ramp_stage6tailfix_e2a_20260408.pth",
        optional=True,
    ),
)

HOOK_SPECS: Tuple[HookSpec, ...] = (
    HookSpec(
        key="direct_pose_head_out",
        module="direct_pose_head",
        kind="output",
        label="direct_pose_head output",
        reason="earliest shared trunk boundary; A1 inherited earliest necessary anchor",
    ),
    HookSpec(
        key="direct_pose_leg_head_out",
        module="direct_pose_leg_head",
        kind="output",
        label="direct_pose_leg_head output",
        reason="leg-only branch hidden state before leg readout",
    ),
    HookSpec(
        key="direct_pose_arm_proj_out",
        module="direct_pose_arm_proj",
        kind="output",
        label="direct_pose_arm_proj output",
        reason="arm-side nonleg adapter/proj before arm readout",
        fallback_modules=("direct_pose_nonleg_proj",),
    ),
    HookSpec(
        key="direct_pose_else_proj_out",
        module="direct_pose_else_proj",
        kind="output",
        label="direct_pose_else_proj output",
        reason="else-side nonleg adapter/proj before else readout",
        fallback_modules=("direct_pose_nonleg_proj",),
    ),
    HookSpec(
        key="direct_pose_out_leg_in",
        module="direct_pose_out_leg",
        kind="input",
        label="direct_pose_out_leg input",
        reason="leg readout contract input; helps separate trunk-vs-leg-readout issues",
    ),
    HookSpec(
        key="direct_pose_out_arm_in",
        module="direct_pose_out_arm",
        kind="input",
        label="direct_pose_out_arm input",
        reason="arm readout contract input; primary nonleg branch readout tap",
        fallback_modules=("direct_pose_out_nonleg",),
    ),
    HookSpec(
        key="direct_pose_out_else_in",
        module="direct_pose_out_else",
        kind="input",
        label="direct_pose_out_else input",
        reason="else readout contract input; complements arm-side nonleg tap",
        fallback_modules=("direct_pose_out_nonleg",),
    ),
)

HOOK_FAMILIES: Mapping[str, str] = {
    "direct_pose_head_out": "shared_trunk",
    "direct_pose_leg_head_out": "leg_branch",
    "direct_pose_out_leg_in": "leg_branch",
    "direct_pose_arm_proj_out": "nonleg_branch",
    "direct_pose_else_proj_out": "nonleg_branch",
    "direct_pose_out_arm_in": "nonleg_branch",
    "direct_pose_out_else_in": "nonleg_branch",
}


def clone_nested(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {str(k): clone_nested(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_nested(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_nested(v) for v in obj)
    return copy.deepcopy(obj)


def assert_exists(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required artifact(s):\n" + "\n".join(missing))


def safe_div(num: float, den: float) -> float:
    if (not math.isfinite(num)) or (not math.isfinite(den)) or abs(den) <= EPS:
        return float("nan")
    return float(num / den)


def tensor_to_rows(tensor: torch.Tensor) -> torch.Tensor:
    x = tensor.detach().cpu().float()
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x.reshape(-1, x.shape[CHANNEL_AXIS])


def tensor_sum(items: Sequence[torch.Tensor], ref: Optional[torch.Tensor] = None) -> torch.Tensor:
    if items:
        return torch.stack(list(items)).sum()
    if ref is not None:
        return ref.new_tensor(0.0)
    return torch.tensor(0.0)


def finite_tensor_median(x: torch.Tensor) -> float:
    if x.numel() <= 0:
        return float("nan")
    mask = torch.isfinite(x)
    if not bool(mask.any().item()):
        return float("nan")
    return float(torch.quantile(x[mask], 0.5).item())


def finite_tensor_mean(x: torch.Tensor) -> float:
    if x.numel() <= 0:
        return float("nan")
    mask = torch.isfinite(x)
    if not bool(mask.any().item()):
        return float("nan")
    return float(x[mask].mean().item())


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() != y.numel() or x.numel() <= 1:
        return float("nan")
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) <= 1:
        return float("nan")
    xa = x[mask]
    ya = y[mask]
    xm = xa - xa.mean()
    ym = ya - ya.mean()
    den = torch.sqrt((xm.pow(2).sum()) * (ym.pow(2).sum()))
    if float(den.item()) <= EPS:
        return float("nan")
    return float((xm * ym).sum().item() / den.item())


class HookRecorder:
    def __init__(self, model: torch.nn.Module, hook_specs: Sequence[HookSpec]) -> None:
        self._model = model
        self._hook_specs = list(hook_specs)
        self._handles: List[Any] = []
        self.records: Dict[str, List[torch.Tensor]] = {spec.key: [] for spec in hook_specs}
        self.meta: Dict[str, Dict[str, Any]] = {}
        modules = dict(model.named_modules())
        for spec in hook_specs:
            selected_name = None
            selected_module = None
            for candidate in (spec.module, *spec.fallback_modules):
                module = modules.get(str(candidate))
                if module is not None:
                    selected_name = str(candidate)
                    selected_module = module
                    break
            self.meta[spec.key] = {
                "label": spec.label,
                "requested_module": spec.module,
                "selected_module": selected_name,
                "fallback_candidates": list(spec.fallback_modules),
                "fallback_used": bool(selected_name is not None and selected_name != spec.module),
                "present": bool(selected_module is not None),
                "hook_kind": "forward_hook_output" if spec.kind == "output" else "forward_pre_hook_input",
                "reason": spec.reason,
                "shape_examples": [],
                "num_calls": 0,
            }
            if selected_module is None:
                continue
            if spec.kind == "output":
                handle = selected_module.register_forward_hook(self._make_output_hook(spec.key))
            elif spec.kind == "input":
                handle = selected_module.register_forward_pre_hook(self._make_input_hook(spec.key))
            else:
                raise ValueError(f"unknown hook kind: {spec.kind}")
            self._handles.append(handle)

    def _record_tensor(self, key: str, tensor: torch.Tensor) -> None:
        self.records[key].append(tensor)
        meta = self.meta[key]
        meta["num_calls"] = int(meta["num_calls"]) + 1
        shape = [int(v) for v in tensor.shape]
        shape_examples = meta["shape_examples"]
        if shape not in shape_examples:
            shape_examples.append(shape)

    def _make_output_hook(self, key: str):
        def _hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
            if torch.is_tensor(output):
                self._record_tensor(key, output)

        return _hook

    def _make_input_hook(self, key: str):
        def _hook(_module: torch.nn.Module, inputs: Tuple[Any, ...]) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                self._record_tensor(key, inputs[0])

        return _hook

    def remove(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()


def build_shared_context(*, n_batches: int) -> Dict[str, Any]:
    replace_payload = load_json(STAGE70B_REPLACE_CONFIG)
    cfg = posttrain._cfg_from_payload(replace_payload)
    posttrain._set_seed(int(cfg.seed))
    device = posttrain._resolve_device(cfg.device)
    norm_spec, ds, _ = posttrain._build_dataset_and_loader(cfg)
    loader = DataLoader(ds, batch_size=int(cfg.batch), shuffle=False, drop_last=True, num_workers=0)
    total_batches = len(loader)
    if total_batches <= 0:
        raise RuntimeError("replace loader produced zero batches")
    actual_n = int(n_batches)
    batches: List[Dict[str, Any]] = []
    loader_iter = posttrain._iter_infinite(loader)
    for _batch_idx in range(actual_n):
        batch = next(loader_iter)
        batches.append(clone_nested(batch))
    train_mode = posttrain._resolve_train_mode(cfg)
    rollout_mode_kwargs = posttrain._build_rollout_mode_kwargs(cfg, train_mode)
    return {
        "replace_payload": replace_payload,
        "replace_cfg": cfg,
        "requested_n_batches": int(n_batches),
        "device": device,
        "norm_spec": norm_spec,
        "dataset": ds,
        "batches": batches,
        "actual_n_batches": int(actual_n),
        "loader_len": int(total_batches),
        "train_mode": str(train_mode),
        "rollout_mode_kwargs": dict(rollout_mode_kwargs),
    }


def build_arm_runtime(shared_ctx: Mapping[str, Any], arm: ArmSpec) -> Dict[str, Any]:
    cfg_payload = dict(shared_ctx["replace_payload"])
    cfg_payload["ckpt_in"] = str(arm.ckpt)
    cfg = posttrain._cfg_from_payload(cfg_payload)
    model, *_meta = posttrain._build_posttrain_model_from_ckpt(
        cfg=cfg,
        ds=shared_ctx["dataset"],
        device=shared_ctx["device"],
    )
    trainer = posttrain._build_model_and_trainer(
        cfg=cfg,
        ds=shared_ctx["dataset"],
        model=model,
        norm_spec=shared_ctx["norm_spec"],
    )
    train_mode = posttrain._resolve_train_mode(cfg)
    posttrain._freeze_all(model)
    posttrain._unfreeze_for_train_mode(model, cfg, train_mode)
    model.train()
    rollout_mode_kwargs = posttrain._build_rollout_mode_kwargs(cfg, train_mode)
    if hasattr(trainer, "_direct_pose_group_norm_ema"):
        setattr(trainer, "_direct_pose_group_norm_ema", {})
    return {
        "arm": arm,
        "cfg_payload": cfg_payload,
        "cfg": cfg,
        "model": model,
        "trainer": trainer,
        "train_mode": str(train_mode),
        "rollout_mode_kwargs": dict(rollout_mode_kwargs),
        "seed": int(getattr(cfg, "seed", 0) or 0),
    }


def build_batch_rollout_components(runtime: Mapping[str, Any], batch: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = runtime["cfg"]
    trainer = runtime["trainer"]
    model = runtime["model"]
    rollout_mode_kwargs = runtime["rollout_mode_kwargs"]
    columns = ("X", "Z")
    prep_ctx = posttrain._lambda_rollout_prepare_context(
        trainer,
        model,
        batch,
        columns=columns,
        rollout_steps=int(getattr(cfg, "rollout_steps", 0) or 0),
        rollout_cycles=int(getattr(cfg, "rollout_cycles", 1) or 1),
        include_boundary=bool(getattr(cfg, "rollout_include_boundary", False)),
        boundary_weight=float(getattr(cfg, "lambda_boundary_weight", 1.0) or 0.0),
        random_offset=bool(getattr(cfg, "rollout_random_offset", False)),
        time_weight_mode=str(getattr(cfg, "lambda_time_weight_mode", "inv") or "inv"),
        time_weight_max=float(getattr(cfg, "lambda_time_weight_max", 1.0) or 1.0),
    )
    objective = str(rollout_mode_kwargs.get("objective", "blend") or "blend")
    nonleg_focus_ctx = posttrain._lambda_rollout_resolve_nonleg_focus(
        trainer,
        objective=objective,
        direct_pose_nonleg_focus_bones=str(getattr(cfg, "direct_pose_nonleg_focus_bones", "") or ""),
        direct_pose_nonleg_focus_weight=float(getattr(cfg, "direct_pose_nonleg_focus_weight", 1.0) or 1.0),
        J=int(prep_ctx["J"]),
        device=prep_ctx["device"],
    )
    reg_ctx = posttrain._lambda_rollout_build_reg_params(
        trainer,
        objective=objective,
        lambda_gate_sup_weight=float(rollout_mode_kwargs.get("lambda_gate_sup_weight", 0.0) or 0.0),
        lambda_gate_sup_start_step=int(rollout_mode_kwargs.get("lambda_gate_sup_start_step", -1) or -1),
        lambda_gate_sup_tau_deg=float(rollout_mode_kwargs.get("lambda_gate_sup_tau_deg", 2.5) or 2.5),
        lambda_gate_sup_margin_deg=float(rollout_mode_kwargs.get("lambda_gate_sup_margin_deg", 1.0) or 1.0),
        direct_pose_loss_group_norm_enable=bool(getattr(cfg, "direct_pose_loss_group_norm_enable", False)),
        direct_pose_loss_group_norm_w_leg=float(getattr(cfg, "direct_pose_loss_group_norm_w_leg", 1.0) or 1.0),
        direct_pose_loss_group_norm_w_nonleg=float(getattr(cfg, "direct_pose_loss_group_norm_w_nonleg", 1.0) or 1.0),
        direct_pose_loss_group_norm_ema_beta=float(
            getattr(cfg, "direct_pose_loss_group_norm_ema_beta", 0.95) or 0.95
        ),
        direct_pose_loss_group_norm_ratio_min=float(
            getattr(cfg, "direct_pose_loss_group_norm_ratio_min", 0.2) or 0.2
        ),
        direct_pose_loss_group_norm_ratio_max=float(
            getattr(cfg, "direct_pose_loss_group_norm_ratio_max", 5.0) or 5.0
        ),
        direct_pose_loss_group_norm_eps=float(getattr(cfg, "direct_pose_loss_group_norm_eps", 1e-6) or 1e-6),
        direct_pose_loss_3way_enable=bool(getattr(cfg, "direct_pose_loss_3way_enable", False)),
        direct_pose_loss_3way_w_leg=float(getattr(cfg, "direct_pose_loss_3way_w_leg", 1.0) or 1.0),
        direct_pose_loss_3way_w_arm=float(getattr(cfg, "direct_pose_loss_3way_w_arm", 1.0) or 1.0),
        direct_pose_loss_3way_w_else=float(getattr(cfg, "direct_pose_loss_3way_w_else", 1.0) or 1.0),
        direct_pose_loss_arm_else_balance_enable=bool(
            getattr(cfg, "direct_pose_loss_arm_else_balance_enable", False)
        ),
        direct_pose_loss_arm_weight=float(getattr(cfg, "direct_pose_loss_arm_weight", 1.0) or 1.0),
        direct_pose_loss_else_weight=float(getattr(cfg, "direct_pose_loss_else_weight", 1.0) or 1.0),
    )
    weights_ctx = {
        "contact_meas_weight": float(getattr(cfg, "contact_meas_weight", 0.0) or 0.0),
        "direct_pose_leg_align_weight": float(getattr(cfg, "direct_pose_leg_align_weight", 0.0) or 0.0),
        "direct_pose_leg_align_oracle_min_deg": float(
            getattr(cfg, "direct_pose_leg_align_oracle_min_deg", 0.0) or 0.0
        ),
        "direct_pose_leg_align_oracle_weight_deg": float(
            getattr(cfg, "direct_pose_leg_align_oracle_weight_deg", 0.0) or 0.0
        ),
        "direct_pose_leg_align_mode": str(getattr(cfg, "direct_pose_leg_align_mode", "cos") or "cos"),
        "direct_pose_leg_align_mag_weight": float(getattr(cfg, "direct_pose_leg_align_mag_weight", 1.0) or 1.0),
        "direct_pose_leg_align_res_weight": float(getattr(cfg, "direct_pose_leg_align_res_weight", 1.0) or 1.0),
        "direct_pose_leg_align_sign_weight": float(getattr(cfg, "direct_pose_leg_align_sign_weight", 0.0) or 0.0),
        "direct_pose_leg_align_cos_thresh": float(getattr(cfg, "direct_pose_leg_align_cos_thresh", 0.0) or 0.0),
        "direct_pose_leg_align_target_joints": getattr(cfg, "direct_pose_leg_align_target_joints", None),
        "direct_pose_leg_align_anchor_joints": getattr(cfg, "direct_pose_leg_align_anchor_joints", None),
        "direct_pose_leg_align_anchor_weight": float(getattr(cfg, "direct_pose_leg_align_anchor_weight", 0.0) or 0.0),
        "direct_pose_leg_gate_sup_weight": float(getattr(cfg, "direct_pose_leg_gate_sup_weight", 0.0) or 0.0),
        "direct_pose_loss_leg_split": bool(getattr(cfg, "direct_pose_loss_leg_split", False)),
        "direct_nonleg_focus_mask_j": nonleg_focus_ctx["direct_nonleg_focus_mask_j"],
        "direct_nonleg_focus_resolved": int(nonleg_focus_ctx["direct_nonleg_focus_resolved"]),
        "direct_nonleg_focus_weight_use": float(nonleg_focus_ctx["direct_nonleg_focus_weight_use"]),
        "direct_pose_loss_3way_enable": bool(getattr(cfg, "direct_pose_loss_3way_enable", False)),
        "direct_pose_loss_3way_w_leg": float(getattr(cfg, "direct_pose_loss_3way_w_leg", 1.0) or 1.0),
        "direct_pose_loss_3way_w_arm": float(getattr(cfg, "direct_pose_loss_3way_w_arm", 1.0) or 1.0),
        "direct_pose_loss_3way_w_else": float(getattr(cfg, "direct_pose_loss_3way_w_else", 1.0) or 1.0),
        "direct_pose_loss_arm_else_balance_enable": bool(
            getattr(cfg, "direct_pose_loss_arm_else_balance_enable", False)
        ),
        "direct_pose_loss_arm_weight": float(getattr(cfg, "direct_pose_loss_arm_weight", 1.0) or 1.0),
        "direct_pose_loss_else_weight": float(getattr(cfg, "direct_pose_loss_else_weight", 1.0) or 1.0),
        "gate_sup_weight": float(reg_ctx["gate_sup_weight"]),
        "gate_sup_start": int(reg_ctx["gate_sup_start"]),
        "tau_rad": float(reg_ctx["tau_rad"]),
        "margin_rad": float(reg_ctx["margin_rad"]),
        "lambda_plan_entropy_weight": float(rollout_mode_kwargs.get("lambda_plan_entropy_weight", 0.0) or 0.0),
        "lambda_plan_dyn_weight": float(rollout_mode_kwargs.get("lambda_plan_dyn_weight", 0.0) or 0.0),
        "lambda_early_weight": float(rollout_mode_kwargs.get("lambda_early_weight", 0.0) or 0.0),
        "lambda_early_steps": int(rollout_mode_kwargs.get("lambda_early_steps", 0) or 0),
        "lambda_entropy_weight": float(rollout_mode_kwargs.get("lambda_entropy_weight", 0.0) or 0.0),
        "lambda_smooth_weight": float(rollout_mode_kwargs.get("lambda_smooth_weight", 0.0) or 0.0),
        "lambda_monotonic_weight": float(rollout_mode_kwargs.get("lambda_monotonic_weight", 0.0) or 0.0),
    }
    state_vars = {
        "meas_used_logits": False,
        "direct_nonleg_focus_applied": float(nonleg_focus_ctx["direct_nonleg_focus_applied"]),
        "lam_prev": None,
        "lam_prev_monot": None,
        "plan_prev": None,
    }
    accum_ctx = posttrain._lambda_fusion_init_accum_ctx()
    runtime_ctx = {
        "trainer": trainer,
        "model": model,
        "batch": batch,
        "prep_ctx": prep_ctx,
        "time_index_mode": str(getattr(cfg, "time_index_mode", "global") or "global"),
        "enable_reprojection": bool(getattr(trainer, "enable_cond_reprojection", True)),
        "detach_rollout_state": bool(getattr(cfg, "detach_rollout_state", True)),
        "columns": columns,
        "objective": objective,
    }
    return {
        "prep_ctx": prep_ctx,
        "nonleg_focus_ctx": nonleg_focus_ctx,
        "reg_ctx": reg_ctx,
        "weights_ctx": weights_ctx,
        "state_vars": state_vars,
        "accum_ctx": accum_ctx,
        "runtime_ctx": runtime_ctx,
    }


def build_finalize_ctx(
    *,
    runtime: Mapping[str, Any],
    components: Mapping[str, Any],
    meas_used_logits: bool,
    direct_nonleg_focus_applied: float,
) -> Dict[str, Any]:
    cfg = runtime["cfg"]
    trainer = runtime["trainer"]
    model = runtime["model"]
    rollout_mode_kwargs = runtime["rollout_mode_kwargs"]
    prep_ctx = components["prep_ctx"]
    reg_ctx = components["reg_ctx"]
    nonleg_focus_ctx = components["nonleg_focus_ctx"]
    return {
        "trainer": trainer,
        "model": model,
        "objective": str(rollout_mode_kwargs.get("objective", "blend") or "blend"),
        "direct_pose_leg_gate_sup_weight": float(getattr(cfg, "direct_pose_leg_gate_sup_weight", 0.0) or 0.0),
        "direct_pose_leg_align_weight": float(getattr(cfg, "direct_pose_leg_align_weight", 0.0) or 0.0),
        "direct_pose_leg_align_anchor_weight": float(getattr(cfg, "direct_pose_leg_align_anchor_weight", 0.0) or 0.0),
        "lambda_entropy_weight": float(rollout_mode_kwargs.get("lambda_entropy_weight", 0.0) or 0.0),
        "lambda_smooth_weight": float(rollout_mode_kwargs.get("lambda_smooth_weight", 0.0) or 0.0),
        "lambda_early_weight": float(rollout_mode_kwargs.get("lambda_early_weight", 0.0) or 0.0),
        "lambda_monotonic_weight": float(rollout_mode_kwargs.get("lambda_monotonic_weight", 0.0) or 0.0),
        "lambda_plan_entropy_weight": float(rollout_mode_kwargs.get("lambda_plan_entropy_weight", 0.0) or 0.0),
        "lambda_plan_dyn_weight": float(rollout_mode_kwargs.get("lambda_plan_dyn_weight", 0.0) or 0.0),
        "contact_meas_weight": float(getattr(cfg, "contact_meas_weight", 0.0) or 0.0),
        "include_boundary": bool(prep_ctx["include_boundary"]),
        "random_offset": bool(getattr(cfg, "rollout_random_offset", False)),
        "offset": int(prep_ctx["offset"]),
        "boundary_weight": float(getattr(cfg, "lambda_boundary_weight", 1.0) or 0.0),
        "boundary_steps": int(prep_ctx["boundary_steps"]),
        "boundary_weighted_sum": float(prep_ctx["boundary_weighted_sum"]),
        "direct_nonleg_focus_requested": int(nonleg_focus_ctx["direct_nonleg_focus_requested"]),
        "direct_nonleg_focus_resolved": int(nonleg_focus_ctx["direct_nonleg_focus_resolved"]),
        "direct_nonleg_focus_weight_use": float(nonleg_focus_ctx["direct_nonleg_focus_weight_use"]),
        "direct_nonleg_focus_applied": float(direct_nonleg_focus_applied),
        "meas_used_logits": bool(meas_used_logits),
        **reg_ctx,
    }


def compute_grads_for_captures(
    *,
    scalar: torch.Tensor,
    recorder: HookRecorder,
    retain_graph: bool,
) -> Dict[int, torch.Tensor]:
    tensors: List[torch.Tensor] = []
    ids: List[int] = []
    for captures in recorder.records.values():
        for tensor in captures:
            if torch.is_tensor(tensor) and bool(tensor.requires_grad):
                tensors.append(tensor)
                ids.append(id(tensor))
    if (not torch.is_tensor(scalar)) or (not bool(scalar.requires_grad)) or (not tensors):
        return {}
    grads = torch.autograd.grad(
        scalar,
        tensors,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    out: Dict[int, torch.Tensor] = {}
    for tensor_id, grad in zip(ids, grads):
        if torch.is_tensor(grad):
            out[tensor_id] = grad
    return out


def apply_ema_update(trainer: Any, aux_payload: Optional[Mapping[str, Any]]) -> None:
    if not isinstance(aux_payload, Mapping):
        return
    update = aux_payload.get("ema_update_payload", None)
    if not isinstance(update, Mapping):
        return
    next_ema: Dict[str, Any] = {}
    tensor_keys = 0
    for key, value in update.items():
        if torch.is_tensor(value):
            if not bool(torch.isfinite(value).all().detach().cpu().item()):
                return
            tensor_keys += 1
            next_ema[str(key)] = value.detach()
        else:
            next_ema[str(key)] = value
    if tensor_keys > 0:
        setattr(trainer, "_direct_pose_group_norm_ema", next_ema)


def analyze_single_batch(
    *,
    runtime: Mapping[str, Any],
    batch: Mapping[str, Any],
    batch_index: int,
) -> Dict[str, Any]:
    trainer = runtime["trainer"]
    model = runtime["model"]
    posttrain._set_seed(int(runtime["seed"]) + int(batch_index))
    model.zero_grad(set_to_none=True)
    components = build_batch_rollout_components(runtime, batch)
    recorder = HookRecorder(model, HOOK_SPECS)
    try:
        meas_used_logits, direct_nonleg_focus_applied = posttrain._lambda_fusion_run_unroll(
            runtime_ctx=components["runtime_ctx"],
            weights_ctx=components["weights_ctx"],
            accum_ctx=components["accum_ctx"],
            state_vars=components["state_vars"],
        )
        accum_ctx = components["accum_ctx"]
        dir_base_tensor = tensor_sum(accum_ctx["dir_base_terms"])
        dir_leg_tensor = tensor_sum(accum_ctx["dir_leg_base_terms"], ref=dir_base_tensor)
        dir_nonleg_tensor = tensor_sum(accum_ctx["dir_nonleg_base_terms"], ref=dir_base_tensor)
        leg_grads = compute_grads_for_captures(
            scalar=dir_leg_tensor,
            recorder=recorder,
            retain_graph=bool(torch.is_tensor(dir_nonleg_tensor) and bool(dir_nonleg_tensor.requires_grad)),
        )
        nonleg_grads = compute_grads_for_captures(
            scalar=dir_nonleg_tensor,
            recorder=recorder,
            retain_graph=False,
        )
        finalize_ctx = build_finalize_ctx(
            runtime=runtime,
            components=components,
            meas_used_logits=meas_used_logits,
            direct_nonleg_focus_applied=direct_nonleg_focus_applied,
        )
        loss, stats, aux_payload = posttrain._lambda_fusion_finalize(
            finalize_ctx=finalize_ctx,
            accum_ctx=accum_ctx,
        )
    finally:
        recorder.remove()
    batch_rows: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    for hook_key, captures in recorder.records.items():
        act_rows: List[torch.Tensor] = []
        leg_rows: List[torch.Tensor] = []
        nonleg_rows: List[torch.Tensor] = []
        for tensor in captures:
            rows = tensor_to_rows(tensor)
            act_rows.append(rows)
            grad_leg = leg_grads.get(id(tensor))
            grad_nonleg = nonleg_grads.get(id(tensor))
            if grad_leg is None:
                grad_leg = torch.zeros_like(tensor)
            if grad_nonleg is None:
                grad_nonleg = torch.zeros_like(tensor)
            leg_rows.append(tensor_to_rows(grad_leg.abs()))
            nonleg_rows.append(tensor_to_rows(grad_nonleg.abs()))
        batch_rows[hook_key] = {
            "activation_rows": act_rows,
            "leg_grad_rows": leg_rows,
            "nonleg_grad_rows": nonleg_rows,
        }
    apply_ema_update(trainer, aux_payload)
    return {
        "loss": float(loss.detach().cpu()),
        "stats": {str(k): safe_float(v) for k, v in stats.items()},
        "batch_rows": batch_rows,
        "hook_meta": copy.deepcopy(recorder.meta),
        "offset": int(components["prep_ctx"]["offset"]),
    }


def summarize_hook(
    *,
    hook_key: str,
    hook_meta: Mapping[str, Any],
    activation_rows: Sequence[torch.Tensor],
    leg_grad_rows: Sequence[torch.Tensor],
    nonleg_grad_rows: Sequence[torch.Tensor],
) -> Dict[str, Any]:
    if not activation_rows:
        return {
            "hook_key": hook_key,
            "hook_present": bool(hook_meta.get("present", False)),
            "selected_module": hook_meta.get("selected_module"),
            "hook_kind": hook_meta.get("hook_kind"),
            "reason": hook_meta.get("reason"),
            "shape_examples": list(hook_meta.get("shape_examples", [])),
            "num_calls": int(hook_meta.get("num_calls", 0)),
            "channel_count": 0,
            "sample_count": 0,
            "per_channel": [],
            "counts": {
                "near_dead": 0,
                "low_diversity": 0,
                "heavy_tail": 0,
                "scale_outlier": 0,
            },
            "alignment": {
                "anomaly_leg_mass": float("nan"),
                "anomaly_nonleg_mass": float("nan"),
                "topk_leg_grad_share_mean": float("nan"),
                "topk_nonleg_grad_share_mean": float("nan"),
                "anomaly_score_vs_leg_grad_share_corr": float("nan"),
                "anomaly_score_vs_nonleg_grad_share_corr": float("nan"),
            },
            "top_anomalous_channels": [],
        }
    act = torch.cat(list(activation_rows), dim=0).to(dtype=torch.float64)
    leg_grad = torch.cat(list(leg_grad_rows), dim=0).to(dtype=torch.float64)
    nonleg_grad = torch.cat(list(nonleg_grad_rows), dim=0).to(dtype=torch.float64)
    mean = act.mean(dim=0)
    std = act.std(dim=0, unbiased=False) if act.shape[0] > 1 else torch.zeros_like(mean)
    min_v = act.min(dim=0).values
    max_v = act.max(dim=0).values
    p01 = torch.quantile(act, 0.01, dim=0)
    p99 = torch.quantile(act, 0.99, dim=0)
    median = torch.quantile(act, 0.50, dim=0)
    mad = torch.quantile((act - median.unsqueeze(0)).abs(), 0.50, dim=0)
    p99_abs = torch.quantile(act.abs(), 0.99, dim=0)
    span = p99 - p01
    centered = act - mean.unsqueeze(0)
    var = centered.pow(2).mean(dim=0)
    kurtosis = torch.full_like(mean, float("nan"))
    valid_var = var > EPS
    if bool(valid_var.any().item()):
        kurtosis[valid_var] = centered[:, valid_var].pow(4).mean(dim=0) / var[valid_var].pow(2) - 3.0
    layer_abs_mean_median = finite_tensor_median(mean.abs())
    layer_std_median = finite_tensor_median(std)
    layer_span_median = finite_tensor_median(span)
    layer_mad_median = finite_tensor_median(mad)
    layer_p99_abs_median = finite_tensor_median(p99_abs)
    dead_std_ref = max(ABS_DEAD_STD, REL_DEAD_FACTOR * max(layer_std_median, ABS_DEAD_STD))
    dead_span_ref = max(ABS_DEAD_SPAN, REL_DEAD_FACTOR * max(layer_span_median, ABS_DEAD_SPAN))
    dead_mad_ref = max(ABS_DEAD_MAD, REL_DEAD_FACTOR * max(layer_mad_median, ABS_DEAD_MAD))
    abs_mean_ratio = mean.abs() / max(layer_abs_mean_median, EPS)
    std_ratio = std / max(layer_std_median, EPS)
    p99_abs_ratio = p99_abs / max(layer_p99_abs_median, EPS)
    dead_ratio = torch.maximum(
        torch.full_like(std, dead_std_ref) / std.clamp_min(EPS),
        torch.full_like(span, dead_span_ref) / span.clamp_min(EPS),
    )
    low_div_ratio = torch.full_like(mad, dead_mad_ref) / mad.clamp_min(EPS)
    heavy_tail_ratio = torch.where(
        torch.isfinite(kurtosis),
        kurtosis / HEAVY_TAIL_EXCESS_KURTOSIS,
        torch.zeros_like(kurtosis),
    )
    anomaly_score = torch.maximum(
        torch.maximum(torch.maximum(abs_mean_ratio, std_ratio), p99_abs_ratio),
        torch.maximum(torch.maximum(dead_ratio, low_div_ratio), heavy_tail_ratio),
    )
    near_dead = (std <= dead_std_ref) & (span <= dead_span_ref)
    low_diversity = mad <= dead_mad_ref
    heavy_tail = torch.isfinite(kurtosis) & (kurtosis >= HEAVY_TAIL_EXCESS_KURTOSIS)
    scale_outlier = (abs_mean_ratio >= SCALE_OUTLIER_RATIO) | (std_ratio >= SCALE_OUTLIER_RATIO) | (
        p99_abs_ratio >= SCALE_OUTLIER_RATIO
    )
    leg_grad_mean = leg_grad.mean(dim=0)
    nonleg_grad_mean = nonleg_grad.mean(dim=0)
    total_grad = leg_grad_mean + nonleg_grad_mean
    leg_share = leg_grad_mean / total_grad.clamp_min(EPS)
    nonleg_share = nonleg_grad_mean / total_grad.clamp_min(EPS)
    anomaly_mass = anomaly_score.sum().clamp_min(EPS)
    anomaly_leg_mass = float((anomaly_score * leg_share).sum().item() / anomaly_mass.item())
    anomaly_nonleg_mass = float((anomaly_score * nonleg_share).sum().item() / anomaly_mass.item())
    order = sorted(
        range(int(act.shape[1])),
        key=lambda idx: float(anomaly_score[idx].item()),
        reverse=True,
    )
    top_indices = order[: min(TOP_K, len(order))]
    per_channel: List[Dict[str, Any]] = []
    for idx in range(int(act.shape[1])):
        reasons: List[str] = []
        if bool(near_dead[idx].item()):
            reasons.append("near_dead")
        if bool(low_diversity[idx].item()):
            reasons.append("low_diversity")
        if bool(heavy_tail[idx].item()):
            reasons.append("heavy_tail")
        if bool(scale_outlier[idx].item()):
            reasons.append("scale_outlier")
        per_channel.append(
            {
                "channel": int(idx),
                "mean": float(mean[idx].item()),
                "std": float(std[idx].item()),
                "min": float(min_v[idx].item()),
                "max": float(max_v[idx].item()),
                "p01": float(p01[idx].item()),
                "p99": float(p99[idx].item()),
                "median": float(median[idx].item()),
                "mad": float(mad[idx].item()),
                "excess_kurtosis_fisher": float(kurtosis[idx].item()) if torch.isfinite(kurtosis[idx]) else float("nan"),
                "p99_abs": float(p99_abs[idx].item()),
                "p99_p01_span": float(span[idx].item()),
                "abs_mean_ratio_to_layer_median": float(abs_mean_ratio[idx].item()),
                "std_ratio_to_layer_median": float(std_ratio[idx].item()),
                "p99_abs_ratio_to_layer_median": float(p99_abs_ratio[idx].item()),
                "anomaly_score": float(anomaly_score[idx].item()),
                "near_dead": bool(near_dead[idx].item()),
                "low_diversity": bool(low_diversity[idx].item()),
                "heavy_tail": bool(heavy_tail[idx].item()),
                "scale_outlier": bool(scale_outlier[idx].item()),
                "leg_grad_mean_abs": float(leg_grad_mean[idx].item()),
                "nonleg_grad_mean_abs": float(nonleg_grad_mean[idx].item()),
                "leg_grad_share": float(leg_share[idx].item()),
                "nonleg_grad_share": float(nonleg_share[idx].item()),
                "reasons": reasons,
            }
        )
    top_anomalous_channels = [per_channel[idx] for idx in top_indices]
    return {
        "hook_key": hook_key,
        "hook_present": bool(hook_meta.get("present", False)),
        "selected_module": hook_meta.get("selected_module"),
        "hook_kind": hook_meta.get("hook_kind"),
        "reason": hook_meta.get("reason"),
        "shape_examples": list(hook_meta.get("shape_examples", [])),
        "num_calls": int(hook_meta.get("num_calls", 0)),
        "channel_count": int(act.shape[1]),
        "sample_count": int(act.shape[0]),
        "layer_medians": {
            "abs_mean": float(layer_abs_mean_median),
            "std": float(layer_std_median),
            "span_p99_p01": float(layer_span_median),
            "mad": float(layer_mad_median),
            "p99_abs": float(layer_p99_abs_median),
        },
        "criteria_refs": {
            "dead_std_ref": float(dead_std_ref),
            "dead_span_ref": float(dead_span_ref),
            "dead_mad_ref": float(dead_mad_ref),
        },
        "per_channel": per_channel,
        "counts": {
            "near_dead": int(near_dead.sum().item()),
            "low_diversity": int(low_diversity.sum().item()),
            "heavy_tail": int(heavy_tail.sum().item()),
            "scale_outlier": int(scale_outlier.sum().item()),
        },
        "alignment": {
            "anomaly_leg_mass": anomaly_leg_mass,
            "anomaly_nonleg_mass": anomaly_nonleg_mass,
            "topk_leg_grad_share_mean": finite_tensor_mean(leg_share[top_indices]) if top_indices else float("nan"),
            "topk_nonleg_grad_share_mean": finite_tensor_mean(nonleg_share[top_indices]) if top_indices else float("nan"),
            "anomaly_score_vs_leg_grad_share_corr": pearson_corr(anomaly_score, leg_share),
            "anomaly_score_vs_nonleg_grad_share_corr": pearson_corr(anomaly_score, nonleg_share),
        },
        "top_anomalous_channels": top_anomalous_channels,
    }


def summarize_arm(
    *,
    runtime: Mapping[str, Any],
    batch_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    hook_meta_union: Dict[str, Dict[str, Any]] = {spec.key: {} for spec in HOOK_SPECS}
    hook_rows: Dict[str, List[torch.Tensor]] = {spec.key: [] for spec in HOOK_SPECS}
    hook_leg_rows: Dict[str, List[torch.Tensor]] = {spec.key: [] for spec in HOOK_SPECS}
    hook_nonleg_rows: Dict[str, List[torch.Tensor]] = {spec.key: [] for spec in HOOK_SPECS}
    batch_sequences: List[Dict[str, Any]] = []
    for batch_idx, batch_result in enumerate(batch_results):
        batch_sequences.append(
            {
                "batch_index": int(batch_idx),
                "offset": int(batch_result["offset"]),
                "loss": float(batch_result["loss"]),
                "dir_base": safe_float(batch_result["stats"].get("dir_base")),
                "dir_geo": safe_float(batch_result["stats"].get("dir_geo")),
                "dir_leg_base": safe_float(batch_result["stats"].get("dir_leg_base")),
                "dir_nonleg_base": safe_float(batch_result["stats"].get("dir_nonleg_base")),
                "dir_nonleg_plain": safe_float(batch_result["stats"].get("dir_nonleg_plain")),
                "dir_group_norm_used": safe_float(batch_result["stats"].get("dir_group_norm_used")),
                "dir_group_norm_leg_raw": safe_float(batch_result["stats"].get("dir_group_norm_leg_raw")),
                "dir_group_norm_nonleg_raw": safe_float(batch_result["stats"].get("dir_group_norm_nonleg_raw")),
                "dir_group_norm_leg": safe_float(batch_result["stats"].get("dir_group_norm_leg")),
                "dir_group_norm_nonleg": safe_float(batch_result["stats"].get("dir_group_norm_nonleg")),
                "dir_group_norm_leg_ema": safe_float(batch_result["stats"].get("dir_group_norm_leg_ema")),
                "dir_group_norm_nonleg_ema": safe_float(batch_result["stats"].get("dir_group_norm_nonleg_ema")),
            }
        )
        for hook_key, meta in batch_result["hook_meta"].items():
            if not hook_meta_union[hook_key]:
                hook_meta_union[hook_key] = copy.deepcopy(meta)
            else:
                hook_meta_union[hook_key]["num_calls"] = int(hook_meta_union[hook_key].get("num_calls", 0)) + int(
                    meta.get("num_calls", 0)
                )
                for shape in meta.get("shape_examples", []):
                    if shape not in hook_meta_union[hook_key]["shape_examples"]:
                        hook_meta_union[hook_key]["shape_examples"].append(shape)
        for hook_key, payload in batch_result["batch_rows"].items():
            hook_rows[hook_key].extend(payload["activation_rows"])
            hook_leg_rows[hook_key].extend(payload["leg_grad_rows"])
            hook_nonleg_rows[hook_key].extend(payload["nonleg_grad_rows"])
    per_hook_summary: Dict[str, Any] = {}
    family_summary: Dict[str, Dict[str, Any]] = {}
    for hook_spec in HOOK_SPECS:
        hook_key = hook_spec.key
        hook_summary = summarize_hook(
            hook_key=hook_key,
            hook_meta=hook_meta_union[hook_key],
            activation_rows=hook_rows[hook_key],
            leg_grad_rows=hook_leg_rows[hook_key],
            nonleg_grad_rows=hook_nonleg_rows[hook_key],
        )
        per_hook_summary[hook_key] = hook_summary
        family_key = HOOK_FAMILIES[hook_key]
        family = family_summary.setdefault(
            family_key,
            {
                "hooks": [],
                "near_dead": 0,
                "low_diversity": 0,
                "heavy_tail": 0,
                "scale_outlier": 0,
                "anomaly_leg_mass_weighted_sum": 0.0,
                "anomaly_nonleg_mass_weighted_sum": 0.0,
                "channel_weight_sum": 0.0,
            },
        )
        family["hooks"].append(hook_key)
        counts = hook_summary["counts"]
        family["near_dead"] += int(counts["near_dead"])
        family["low_diversity"] += int(counts["low_diversity"])
        family["heavy_tail"] += int(counts["heavy_tail"])
        family["scale_outlier"] += int(counts["scale_outlier"])
        weight = float(hook_summary.get("channel_count", 0) or 0)
        family["anomaly_leg_mass_weighted_sum"] += weight * safe_float(
            hook_summary["alignment"].get("anomaly_leg_mass")
        )
        family["anomaly_nonleg_mass_weighted_sum"] += weight * safe_float(
            hook_summary["alignment"].get("anomaly_nonleg_mass")
        )
        family["channel_weight_sum"] += weight
    for family in family_summary.values():
        weight_sum = float(family["channel_weight_sum"])
        family["anomaly_leg_mass"] = safe_div(float(family["anomaly_leg_mass_weighted_sum"]), weight_sum)
        family["anomaly_nonleg_mass"] = safe_div(float(family["anomaly_nonleg_mass_weighted_sum"]), weight_sum)
    return {
        "arm_key": runtime["arm"].key,
        "arm_label": runtime["arm"].label,
        "ckpt": str(runtime["arm"].ckpt),
        "n_batches": int(len(batch_results)),
        "per_batch_sequence": batch_sequences,
        "hooks": per_hook_summary,
        "families": family_summary,
    }


def aggregate_pair_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    hooks_delta: Dict[str, Any] = {}
    for hook_spec in HOOK_SPECS:
        hook_key = hook_spec.key
        left_hook = left["hooks"][hook_key]
        right_hook = right["hooks"][hook_key]
        hooks_delta[hook_key] = {
            "near_dead_delta": int(left_hook["counts"]["near_dead"]) - int(right_hook["counts"]["near_dead"]),
            "low_diversity_delta": int(left_hook["counts"]["low_diversity"])
            - int(right_hook["counts"]["low_diversity"]),
            "heavy_tail_delta": int(left_hook["counts"]["heavy_tail"]) - int(right_hook["counts"]["heavy_tail"]),
            "scale_outlier_delta": int(left_hook["counts"]["scale_outlier"])
            - int(right_hook["counts"]["scale_outlier"]),
            "anomaly_leg_mass_delta": safe_float(left_hook["alignment"]["anomaly_leg_mass"])
            - safe_float(right_hook["alignment"]["anomaly_leg_mass"]),
            "anomaly_nonleg_mass_delta": safe_float(left_hook["alignment"]["anomaly_nonleg_mass"])
            - safe_float(right_hook["alignment"]["anomaly_nonleg_mass"]),
        }
    families_delta: Dict[str, Any] = {}
    for family_key in sorted(set(left["families"].keys()) | set(right["families"].keys())):
        left_family = left["families"].get(family_key, {})
        right_family = right["families"].get(family_key, {})
        families_delta[family_key] = {
            "near_dead_delta": int(left_family.get("near_dead", 0)) - int(right_family.get("near_dead", 0)),
            "low_diversity_delta": int(left_family.get("low_diversity", 0))
            - int(right_family.get("low_diversity", 0)),
            "heavy_tail_delta": int(left_family.get("heavy_tail", 0)) - int(right_family.get("heavy_tail", 0)),
            "scale_outlier_delta": int(left_family.get("scale_outlier", 0))
            - int(right_family.get("scale_outlier", 0)),
            "anomaly_leg_mass_delta": safe_float(left_family.get("anomaly_leg_mass"))
            - safe_float(right_family.get("anomaly_leg_mass")),
            "anomaly_nonleg_mass_delta": safe_float(left_family.get("anomaly_nonleg_mass"))
            - safe_float(right_family.get("anomaly_nonleg_mass")),
        }
    left_seq = left["per_batch_sequence"]
    right_seq = right["per_batch_sequence"]
    shared_steps = min(len(left_seq), len(right_seq))
    step_metrics = {}
    for key in (
        "dir_base",
        "dir_geo",
        "dir_leg_base",
        "dir_nonleg_base",
        "dir_group_norm_leg_raw",
        "dir_group_norm_nonleg_raw",
        "dir_group_norm_leg_ema",
        "dir_group_norm_nonleg_ema",
    ):
        deltas = [
            safe_float(left_seq[idx].get(key)) - safe_float(right_seq[idx].get(key))
            for idx in range(shared_steps)
        ]
        step_metrics[key] = {
            "step0_delta": deltas[0] if deltas else float("nan"),
            "step1_delta": deltas[1] if len(deltas) > 1 else float("nan"),
            "mean_delta": float(sum(deltas) / len(deltas)) if deltas else float("nan"),
            "deltas": deltas,
        }
    return {
        "hooks": hooks_delta,
        "families": families_delta,
        "batch_sequence_deltas": step_metrics,
    }


def classify_relative_pathology(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    delta = aggregate_pair_delta(candidate, baseline)
    family_scores: Dict[str, float] = {}
    anomaly_type_scores = {
        "near_dead": 0.0,
        "low_diversity": 0.0,
        "heavy_tail": 0.0,
        "scale_outlier": 0.0,
    }
    for family_key, payload in delta["families"].items():
        score = (
            max(0.0, float(payload["near_dead_delta"]))
            + max(0.0, float(payload["low_diversity_delta"]))
            + max(0.0, float(payload["heavy_tail_delta"]))
            + max(0.0, float(payload["scale_outlier_delta"]))
        )
        family_scores[family_key] = score
        anomaly_type_scores["near_dead"] += max(0.0, float(payload["near_dead_delta"]))
        anomaly_type_scores["low_diversity"] += max(0.0, float(payload["low_diversity_delta"]))
        anomaly_type_scores["heavy_tail"] += max(0.0, float(payload["heavy_tail_delta"]))
        anomaly_type_scores["scale_outlier"] += max(0.0, float(payload["scale_outlier_delta"]))
    dominant_family = max(family_scores, key=family_scores.get) if family_scores else "none"
    total_relative_signal = float(sum(family_scores.values()))
    dominant_anomaly_type = max(anomaly_type_scores, key=anomaly_type_scores.get)
    dominant_family_payload = candidate["families"].get(dominant_family, {})
    anomaly_nonleg_mass = safe_float(dominant_family_payload.get("anomaly_nonleg_mass"))
    anomaly_leg_mass = safe_float(dominant_family_payload.get("anomaly_leg_mass"))
    grad_alignment = "weak"
    if math.isfinite(anomaly_nonleg_mass) and anomaly_nonleg_mass >= 0.60:
        grad_alignment = "nonleg_aligned"
    elif math.isfinite(anomaly_leg_mass) and anomaly_leg_mass >= 0.60:
        grad_alignment = "leg_aligned"
    elif math.isfinite(anomaly_nonleg_mass) and math.isfinite(anomaly_leg_mass):
        grad_alignment = "mixed"
    if total_relative_signal >= 8.0 and grad_alignment in ("nonleg_aligned", "leg_aligned"):
        support = "supportive_but_not_causal"
    elif total_relative_signal > 0.0:
        support = "weak_support"
    else:
        support = "not_supported"
    if support == "not_supported":
        recommendation = "drop_this_line"
    elif dominant_family == "nonleg_branch":
        if dominant_anomaly_type == "scale_outlier":
            recommendation = "branchwise_rescale_or_whitening_first"
        elif dominant_anomaly_type == "heavy_tail":
            recommendation = "robust_clipping_first"
        else:
            recommendation = "per_branch_optimizer_guard_secondary"
    elif dominant_family == "shared_trunk":
        recommendation = "lightweight_branchwise_fixes_lower_priority"
    else:
        recommendation = "need_stronger_causal_test"
    return {
        "candidate_arm": candidate["arm_label"],
        "relative_signal_total": total_relative_signal,
        "dominant_family": dominant_family,
        "dominant_anomaly_type": dominant_anomaly_type,
        "grad_alignment": grad_alignment,
        "support_level": support,
        "recommendation": recommendation,
        "delta": delta,
    }


def build_summary(*, shared_ctx: Mapping[str, Any], arm_summaries: Mapping[str, Any]) -> Dict[str, Any]:
    baseline = arm_summaries["baseline_raw70a"]
    e1 = arm_summaries["e1_top3_raw70a"]
    notail = arm_summaries["notail_raw70a"]
    pairwise = {
        "baseline_vs_e1_top3": aggregate_pair_delta(baseline, e1),
        "baseline_vs_notail": aggregate_pair_delta(baseline, notail),
        "e1_top3_vs_notail": aggregate_pair_delta(e1, notail),
    }
    if "e2a_r_raw70a" in arm_summaries:
        pairwise["baseline_vs_e2a_r"] = aggregate_pair_delta(baseline, arm_summaries["e2a_r_raw70a"])
    e1_support = classify_relative_pathology(baseline=baseline, candidate=e1)
    notail_support = classify_relative_pathology(baseline=baseline, candidate=notail)
    shared_nonbaseline_support = e1_support["support_level"] in (
        "supportive_but_not_causal",
        "weak_support",
    ) and notail_support["support_level"] in ("supportive_but_not_causal", "weak_support")
    same_dominant_family = e1_support["dominant_family"] == notail_support["dominant_family"]
    pathology_supported = shared_nonbaseline_support and same_dominant_family
    if not pathology_supported:
        overall_recommendation = "do_not_promote_distribution_pathology_as_mainline"
    else:
        family = e1_support["dominant_family"]
        anomaly_type = e1_support["dominant_anomaly_type"]
        if family == "nonleg_branch" and anomaly_type == "scale_outlier":
            overall_recommendation = "prioritize_branchwise_rescale_or_whitening_then_light_optimizer_guard"
        elif family == "nonleg_branch" and anomaly_type == "heavy_tail":
            overall_recommendation = "prioritize_robust_clipping_then_branchwise_rescale"
        elif family == "shared_trunk":
            overall_recommendation = "current_probe_looks_more_shared_trunk_like_than_readout_only"
        else:
            overall_recommendation = "pathology_signal_exists_but_needs_causal_followup"
    answers = {
        "q1_e1_and_notail_relative_to_baseline_have_distribution_pathology": (
            "both_yes_but_cautious" if pathology_supported else "not_both_clearly_supported"
        ),
        "q2_shared_trunk_vs_branch": (
            e1_support["dominant_family"] if pathology_supported else "no_clear_dominant_family"
        ),
        "q3_anomalous_channels_align_with_leg_or_nonleg_grad_share": {
            "e1_top3": e1_support["grad_alignment"],
            "notail": notail_support["grad_alignment"],
        },
        "q4_distribution_pathology_hypothesis_supported": (
            "supportive_but_not_causal" if pathology_supported else "not_supported_as_mainline"
        ),
        "q5_priority_next_step": overall_recommendation,
    }
    return {
        "scope": {
            "task": "70a donor-state -> 70b replace entry activation distribution pathology profile",
            "analysis_only": True,
            "no_training": True,
            "n_batches_requested": int(shared_ctx["requested_n_batches"]),
            "n_batches_used": int(shared_ctx["actual_n_batches"]),
            "loader_len": int(shared_ctx["loader_len"]),
            "rollout_cycles": int(getattr(shared_ctx["replace_cfg"], "rollout_cycles", 0) or 0),
            "rollout_steps": int(getattr(shared_ctx["replace_cfg"], "rollout_steps", 0) or 0),
            "flatten_rule": "per hook tensor is flattened across all non-channel dims; last dim is treated as channel axis",
            "kurtosis_definition": "Fisher excess kurtosis = E[((x-mean)/std)^4] - 3",
        },
        "inherited_conclusions": INHERITED_CONCLUSIONS,
        "artifacts": {
            "summary_json": str(SUMMARY_JSON),
            "doc_path": str(DOC_PATH),
            "stage70a_config": str(STAGE70A_CONFIG),
            "stage70b_replace_config": str(STAGE70B_REPLACE_CONFIG),
        },
        "compared_arms": {
            key: {
                "label": summary["arm_label"],
                "ckpt": summary["ckpt"],
            }
            for key, summary in arm_summaries.items()
        },
        "hook_definitions": [
            {
                "hook_key": spec.key,
                "module": spec.module,
                "kind": "forward_hook_output" if spec.kind == "output" else "forward_pre_hook_input",
                "fallback_modules": list(spec.fallback_modules),
                "reason": spec.reason,
            }
            for spec in HOOK_SPECS
        ],
        "statistics_definition": {
            "per_channel_stats": [
                "mean",
                "std",
                "min",
                "max",
                "p01",
                "p99",
                "median",
                "mad",
                "excess_kurtosis_fisher",
                "p99_abs",
                "p99_p01_span",
                "leg_grad_mean_abs",
                "nonleg_grad_mean_abs",
                "leg_grad_share",
                "nonleg_grad_share",
            ],
        },
        "criteria": {
            "near_dead": {
                "formula": "std <= max(abs_dead_std, rel_dead_factor * layer_median_std) AND (p99-p01) <= max(abs_dead_span, rel_dead_factor * layer_median_span)",
                "abs_dead_std": ABS_DEAD_STD,
                "abs_dead_span": ABS_DEAD_SPAN,
                "rel_dead_factor": REL_DEAD_FACTOR,
            },
            "low_diversity": {
                "formula": "MAD <= max(abs_dead_mad, rel_dead_factor * layer_median_mad)",
                "abs_dead_mad": ABS_DEAD_MAD,
                "rel_dead_factor": REL_DEAD_FACTOR,
            },
            "heavy_tail": {
                "formula": "Fisher excess kurtosis >= heavy_tail_excess_kurtosis",
                "heavy_tail_excess_kurtosis": HEAVY_TAIL_EXCESS_KURTOSIS,
            },
            "scale_outlier": {
                "formula": "max(|mean|/layer_median_|mean|, std/layer_median_std, p99_abs/layer_median_p99_abs) >= scale_outlier_ratio",
                "scale_outlier_ratio": SCALE_OUTLIER_RATIO,
            },
        },
        "arms": arm_summaries,
        "pairwise_deltas": pairwise,
        "judgment": {
            "e1_top3_vs_baseline": e1_support,
            "notail_vs_baseline": notail_support,
            "shared_nonbaseline_support": bool(shared_nonbaseline_support),
            "same_dominant_family": bool(same_dominant_family),
            "distribution_pathology_supported": bool(pathology_supported),
            "overall_recommendation": overall_recommendation,
        },
        "answers": answers,
    }


def render_md(summary: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# 2026-04-09 replace handoff distribution pathology probe")
    lines.append("")
    lines.append("> Scope: B1 only / 70a donor-state -> 70b replace entry activation distribution pathology profile / no-train")
    lines.append("")
    lines.append("## 1. Scope / inherited conclusions")
    lines.append("")
    lines.append("直接继承，不重证：")
    for family, items in summary["inherited_conclusions"].items():
        lines.append(f"- `{family}`:")
        for item in items:
            lines.append(f"  - {item}")
    lines.append("")
    lines.append("## 2. Why this probe after A1 + notail falsifier")
    lines.append("")
    lines.append("- A1 已经把 earliest usable boundary 锁到 `direct_pose_head`，但还没回答 replace entry 的 activation shape 是否已病态。")
    lines.append("- notail falsifier 说明问题不像 tail-k 特有；更像 donor-family handoff 机制。")
    lines.append("- 因此本轮只看 `70a donor -> 70b replace objective` 入口时 direct branch activation distribution / grad-share / early EMA support。")
    lines.append("")
    lines.append("## 3. Compared arms")
    lines.append("")
    lines.append("| arm | ckpt |")
    lines.append("|---|---|")
    for _key, arm in summary["compared_arms"].items():
        lines.append(f"| `{arm['label']}` | `{arm['ckpt']}` |")
    lines.append("")
    lines.append("## 4. Hook definitions")
    lines.append("")
    lines.append("| hook_key | selected module | kind | reason | shapes |")
    lines.append("|---|---|---|---|---|")
    baseline_hooks = summary["arms"]["baseline_raw70a"]["hooks"]
    for hook in summary["hook_definitions"]:
        hook_key = hook["hook_key"]
        ref = baseline_hooks[hook_key]
        lines.append(
            f"| `{hook_key}` | `{ref.get('selected_module')}` | `{ref.get('hook_kind')}` | {hook['reason']} | `{ref.get('shape_examples')}` |"
        )
    lines.append("")
    lines.append("## 5. Statistics definition")
    lines.append("")
    lines.append("- Channel axis = tensor last dim; other dims全部 flatten 后聚合。")
    lines.append("- 主统计：`mean/std/min/max/p01/p99/median/MAD/Fisher excess kurtosis`。")
    lines.append("- Grad 统计：每个 channel 记录 `mean abs grad wrt dir_leg_base` 与 `wrt dir_nonleg_base`，再算 `leg_grad_share / nonleg_grad_share`。")
    lines.append("")
    lines.append("## 6. Dead / heavy-tail criteria")
    lines.append("")
    crit = summary["criteria"]
    lines.append(f"- near-dead: `{crit['near_dead']['formula']}`")
    lines.append(f"- low-diversity: `{crit['low_diversity']['formula']}`")
    lines.append(f"- heavy-tail: `{crit['heavy_tail']['formula']}`")
    lines.append(f"- scale-outlier: `{crit['scale_outlier']['formula']}`")
    lines.append("")
    lines.append("## 7. Per-hook summary table")
    lines.append("")
    lines.append("| arm | hook | near_dead | low_div | heavy_tail | scale_outlier | anomaly_leg_mass | anomaly_nonleg_mass |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for arm_key, arm in summary["arms"].items():
        for hook_key, hook in arm["hooks"].items():
            lines.append(
                f"| `{arm['arm_label']}` | `{hook_key}` | {hook['counts']['near_dead']} | {hook['counts']['low_diversity']} | {hook['counts']['heavy_tail']} | {hook['counts']['scale_outlier']} | "
                f"{safe_float(hook['alignment']['anomaly_leg_mass']):.4f} | {safe_float(hook['alignment']['anomaly_nonleg_mass']):.4f} |"
            )
    lines.append("")
    lines.append("## 8. Anomaly vs grad alignment summary")
    lines.append("")
    for arm_key in ("e1_top3_raw70a", "notail_raw70a"):
        judge = summary["judgment"]["e1_top3_vs_baseline"] if arm_key == "e1_top3_raw70a" else summary["judgment"]["notail_vs_baseline"]
        lines.append(
            f"- `{summary['arms'][arm_key]['arm_label']}`: dominant family=`{judge['dominant_family']}`, dominant anomaly type=`{judge['dominant_anomaly_type']}`, grad alignment=`{judge['grad_alignment']}`, support=`{judge['support_level']}`."
        )
    lines.append("")
    lines.append("## 9. Optional EMA-seed support")
    lines.append("")
    lines.append("- 记录了前 N batch 的 `dir_leg_base / dir_nonleg_base / dir_group_norm_leg_raw / dir_group_norm_nonleg_raw / dir_group_norm_*_ema` 序列。")
    lines.append("- 这些序列在 `summary.json` 的 `arms.*.per_batch_sequence` 下可直接做 step0/step1 / seed trajectory 对比。")
    lines.append("")
    lines.append("## 10. Interpretation")
    lines.append("")
    lines.append(
        f"- overall distribution pathology support: `{summary['judgment']['distribution_pathology_supported']}`"
    )
    lines.append(
        f"- overall recommendation: `{summary['judgment']['overall_recommendation']}`"
    )
    lines.append(
        f"- Q1: `{summary['answers']['q1_e1_and_notail_relative_to_baseline_have_distribution_pathology']}`"
    )
    lines.append(
        f"- Q2: `{summary['answers']['q2_shared_trunk_vs_branch']}`"
    )
    lines.append(
        f"- Q3: `{summary['answers']['q3_anomalous_channels_align_with_leg_or_nonleg_grad_share']}`"
    )
    lines.append(
        f"- Q4: `{summary['answers']['q4_distribution_pathology_hypothesis_supported']}`"
    )
    lines.append(
        f"- Q5: `{summary['answers']['q5_priority_next_step']}`"
    )
    lines.append("")
    lines.append("## 11. Next-step recommendation")
    lines.append("")
    lines.append(f"- `{summary['judgment']['overall_recommendation']}`")
    lines.append("")
    return "\n".join(lines)


def run(*, n_batches: int, include_optional: bool) -> Dict[str, Any]:
    required = [STAGE70A_CONFIG, STAGE70B_REPLACE_CONFIG]
    arms = [arm for arm in ARM_SPECS if include_optional or not arm.optional]
    required.extend([arm.ckpt for arm in arms])
    assert_exists(required)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    shared_ctx = build_shared_context(n_batches=n_batches)
    arm_summaries: Dict[str, Any] = {}
    for arm in arms:
        print(f"[probe] analyzing {arm.label}", flush=True)
        runtime = build_arm_runtime(shared_ctx, arm)
        batch_results: List[Dict[str, Any]] = []
        for batch_idx, batch in enumerate(shared_ctx["batches"]):
            batch_results.append(analyze_single_batch(runtime=runtime, batch=clone_nested(batch), batch_index=batch_idx))
        arm_summaries[arm.key] = summarize_arm(runtime=runtime, batch_results=batch_results)
        del runtime
    summary = build_summary(shared_ctx=shared_ctx, arm_summaries=arm_summaries)
    write_json(SUMMARY_JSON, summary)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(render_md(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="70a->70b replace handoff distribution pathology probe")
    parser.add_argument("--n-batches", type=int, default=DEFAULT_N_BATCHES)
    parser.add_argument("--include-optional", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(n_batches=int(args.n_batches), include_optional=bool(args.include_optional))
    print(f"[probe] wrote {SUMMARY_JSON}")
    print(f"[probe] wrote {DOC_PATH}")
    print(f"[probe] support={summary['judgment']['distribution_pathology_supported']}")


if __name__ == "__main__":
    main()
