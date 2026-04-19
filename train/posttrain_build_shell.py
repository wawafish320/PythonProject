from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from train.checkpoint.compat import (
    DirectPoseBuildConfig,
    DirectPoseBuildOverrides,
    DirectPoseLoadCompatOptions,
    EventClockBuildOverrides,
    LambdaFusionBuildOverrides,
    apply_direct_pose_ckpt_compat as _apply_direct_pose_ckpt_compat,
    attach_motion_encoder_bundle as _attach_motion_encoder_bundle,
    infer_event_clock_build_cfg as _infer_event_clock_build_cfg,
    infer_lambda_fusion_build_cfg as _infer_lambda_fusion_build_cfg,
    resolve_direct_pose_build_cfg as _resolve_direct_pose_build_cfg,
)
from train.models import EventMotionModel
from train.training_MPL import validate_and_fix_model_

if TYPE_CHECKING:
    from train.data.dataset import MotionEventDataset


_DIRECT_POSE_LEG_GATE_ALIAS_MAP: dict[str, str] = {
    "": "none",
    "none": "none",
    "off": "none",
    "false": "none",
    "0": "none",
    "no": "none",
    "n": "none",
    "disable": "none",
    "disabled": "none",
    "learned": "learned",
    "on": "learned",
    "true": "learned",
    "1": "learned",
    "yes": "learned",
    "y": "learned",
    "scale": "scale",
    "mag": "scale",
    "magnitude": "scale",
    "logmag": "scale",
    "log_mag": "scale",
    "exp": "scale",
    "alpha": "scale",
}
_DIRECT_POSE_LEG_GATE_CHOICES: tuple[str, ...] = ("none", "learned", "scale")


@dataclass
class PostTrainModelArtifacts:
    model: EventMotionModel
    direct_pose_feat_source: str
    direct_pose_time_pe_dim: int
    direct_pose_time_pe_base: float
    direct_pose_use_phase_z: bool
    direct_pose_phase_z_mode: str
    direct_pose_split_enable: bool
    direct_pose_nonleg_proj_dim: int
    direct_pose_leg_gate_mode_model: str
    direct_pose_leg_gate_power_model: float


@dataclass(frozen=True)
class PostTrainModelBuildState:
    """Resolved model-build inputs derived from checkpoint, dataset, and CLI policy."""

    ckpt_posttrain_cfg: Optional[dict[str, Any]]
    state_dict: dict[str, Any]
    width: int

    contact_dim: int
    angvel_dim: int
    pose_hist_dim: int

    contact_plan_enable: bool
    contact_plan_hidden: int
    contact_plan_inject: str
    contact_plan_time_pe_dim: int
    contact_plan_init_mode: str
    contact_plan_init_hidden: int
    contact_plan_init_dropout: float

    use_event_clock: bool
    event_clock_hidden_dim: int
    event_clock_gate_hidden_dim: int
    event_clock_max_delta: float
    period_dim_init: int

    direct_pose_cfg: DirectPoseBuildConfig

    lambda_fusion_enable: bool
    lambda_fusion_mode: str
    lambda_fusion_hidden: int
    lambda_fusion_dropout: float
    lambda_fusion_logit_init: float
    lambda_fusion_use_rollout_step: bool

    direct_pose_leg_gate_mode_model: str
    direct_pose_leg_gate_power_model: float


def _resolve_posttrain_model_build_state(*, cfg: Any, ds: MotionEventDataset) -> PostTrainModelBuildState:
    """Resolve checkpoint-backed model build state for posttrain instantiation."""
    ckpt = torch.load(cfg.ckpt_in.expanduser(), map_location="cpu")
    ckpt_posttrain_payload = ckpt.get("posttrain_cfg", None) if isinstance(ckpt, dict) else None
    ckpt_posttrain_cfg: Optional[dict[str, Any]] = ckpt_posttrain_payload if isinstance(ckpt_posttrain_payload, dict) else None
    raw_model_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = {
        key: value for key, value in raw_model_state.items() if not (
            key.startswith("frozen_encoder.")
            or key.startswith("frozen_period_head.")
            or key.startswith("contact_plan_input_proj.")
        )
    }
    shared_encoder_weight = state_dict["shared_encoder.0.weight"]
    width = int(shared_encoder_weight.shape[0])
    shared_encoder_in_dim = int(shared_encoder_weight.shape[1])
    period_dim = int(state_dict["period_encoder.weight"].shape[1]) if "period_encoder.weight" in state_dict else 0
    contact_dim = int(getattr(ds, "contact_dim", 0) or 0)
    angvel_dim = int(getattr(ds, "angvel_dim", 0) or 0)
    pose_hist_dim = int(getattr(ds, "pose_hist_dim", 0) or 0)
    cond_dim = int(ds.Dc)
    dataset_base_in_dim = int(ds.Dx) + int(ds.Dc)
    extra_in_dim = int(max(0, shared_encoder_in_dim - dataset_base_in_dim))
    contact_plan_has_weights = any(key.startswith("contact_plan_cell.") for key in state_dict.keys())
    contact_plan_hidden = None
    if contact_plan_has_weights and "contact_plan_cell.weight_ih" in state_dict:
        try:
            contact_plan_hidden = int(state_dict["contact_plan_cell.weight_ih"].shape[0] // 3)
        except Exception:
            contact_plan_hidden = None
    if contact_plan_hidden is None:
        contact_plan_hidden = int(extra_in_dim) if extra_in_dim > 0 else 64

    contact_plan_inject = "none"
    if extra_in_dim == 0:
        contact_plan_inject = "none"
    elif contact_dim > 0 and extra_in_dim == contact_dim:
        contact_plan_inject = "contacts"
    elif contact_plan_hidden > 0 and extra_in_dim == int(contact_plan_hidden):
        contact_plan_inject = "plan_z"
    elif extra_in_dim > 0 and contact_plan_has_weights:
        contact_plan_inject = "plan_z"
        contact_plan_hidden = int(extra_in_dim)
    contact_plan_time_pe_dim = 0
    try:
        contact_plan_time_head_weight = state_dict.get("contact_plan_time_head.weight", None)
        if torch.is_tensor(contact_plan_time_head_weight) and contact_plan_time_head_weight.ndim == 2:
            contact_plan_time_pe_dim = int(contact_plan_time_head_weight.shape[1])
    except Exception:
        contact_plan_time_pe_dim = 0
    contact_plan_init_mode = str(getattr(cfg, "contact_plan_init_mode", "learnable") or "learnable")
    contact_plan_init_hidden = int(getattr(cfg, "contact_plan_init_hidden", 128) or 128)
    contact_plan_init_dropout = float(getattr(cfg, "contact_plan_init_dropout", 0.0) or 0.0)
    contact_plan_init_has_weights = any(key.startswith("contact_plan_init_head.") for key in state_dict.keys())
    if contact_plan_init_has_weights:
        if str(contact_plan_init_mode).lower().strip() not in ("obs", "learnable+obs", "learnable_obs", "obs+learnable"):
            print("[posttrain][WARN] checkpoint has contact_plan_init_head weights; overriding contact_plan_init_mode -> learnable+obs.")
            contact_plan_init_mode = "learnable+obs"
        contact_plan_init_weight = state_dict.get("contact_plan_init_head.1.weight", None)
        if torch.is_tensor(contact_plan_init_weight) and contact_plan_init_weight.ndim == 2:
            contact_plan_init_hidden = int(contact_plan_init_weight.shape[0])
    direct_pose_cfg = _resolve_direct_pose_build_cfg(
        out_motion_dim=int(ds.Dy),
        state_dict=state_dict,
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        contact_dim=contact_dim,
        cond_dim=cond_dim,
        width=width,
        overrides=DirectPoseBuildOverrides(
            train_direct_pose=bool(getattr(cfg, "train_direct_pose", False)),
            direct_pose_reinit=bool(getattr(cfg, "direct_pose_reinit", False)),
            hidden_override=getattr(cfg, "direct_pose_hidden_override", None),
            meas_mode_override=getattr(cfg, "direct_pose_meas_mode_override", None),
            feat_source=str(getattr(cfg, "direct_pose_feat_source", "auto") or "auto"),
            time_pe_dim=int(getattr(cfg, "direct_pose_time_pe_dim", -1)),
            time_pe_base=float(getattr(cfg, "direct_pose_time_pe_base", 10000.0) or 10000.0),
            use_phase_z=bool(getattr(cfg, "direct_pose_use_phase_z", False)),
            phase_z_mode=str(getattr(cfg, "direct_pose_phase_z_mode", "concat") or "concat"),
            split_enable=bool(getattr(cfg, "direct_pose_split_enable", False)),
            stepc_unified_leg_terminal=bool(getattr(cfg, "direct_pose_stepc_unified_leg_terminal", False)),
            arm_split_enable=bool(getattr(cfg, "direct_pose_arm_split_enable", False)),
            arm_bones=getattr(cfg, "direct_pose_arm_bones", None),
            nonleg_proj_dim=max(0, int(getattr(cfg, "direct_pose_nonleg_proj_dim", 0) or 0)),
        ),
    )
    contact_plan_enable = bool(contact_plan_has_weights or direct_pose_cfg.enable or (extra_in_dim > 0 and contact_dim > 0 and cond_dim > 0))
    event_clock_cfg = _infer_event_clock_build_cfg(
        state_dict=state_dict,
        contact_dim=contact_dim,
        period_dim=period_dim,
        overrides=EventClockBuildOverrides(
            mode=str(getattr(cfg, "event_clock", "auto") or "auto"),
            hidden_dim=getattr(cfg, "event_clock_hidden_dim", None),
            gate_hidden_dim=getattr(cfg, "event_clock_gate_hidden_dim", None),
            max_delta=float(getattr(cfg, "event_clock_max_delta", 0.5) or 0.5),
            has_encoder_bundle=bool(
                cfg.encoder_bundle is not None and cfg.encoder_bundle.expanduser().is_file()
            ),
        ),
    )
    lambda_fusion_cfg = _infer_lambda_fusion_build_cfg(
        state_dict=state_dict,
        width=width,
        contact_dim=contact_dim,
        contact_plan_enable=contact_plan_enable,
        overrides=LambdaFusionBuildOverrides(
            train_lambda_head=bool(getattr(cfg, "train_lambda_head", False)),
            mode=str(getattr(cfg, "lambda_fusion_mode", "per_joint") or "per_joint"),
            hidden=int(getattr(cfg, "lambda_fusion_hidden", 128) or 128),
            dropout=float(getattr(cfg, "lambda_fusion_dropout", 0.0) or 0.0),
            logit_init=float(getattr(cfg, "lambda_fusion_logit_init", -2.0) or -2.0),
            use_rollout_step=bool(getattr(cfg, "lambda_fusion_use_rollout_step", False)),
        ),
    )
    direct_pose_leg_gate_mode_raw = str(getattr(cfg, "direct_pose_leg_gate_mode", "none") or "none").strip().lower()
    direct_pose_leg_gate_mode_model = _DIRECT_POSE_LEG_GATE_ALIAS_MAP.get(direct_pose_leg_gate_mode_raw, direct_pose_leg_gate_mode_raw)
    if direct_pose_leg_gate_mode_model not in _DIRECT_POSE_LEG_GATE_CHOICES:
        direct_pose_leg_gate_mode_model = "none"
        print(
            f"[posttrain][WARN] unrecognized direct_pose_leg_gate_mode={direct_pose_leg_gate_mode_raw!r}; using 'none'. "
            "Set explicit 'none'/'learned'/'scale'."
        )
    else:
        direct_pose_leg_gate_mode_model = str(direct_pose_leg_gate_mode_model)
    try:
        direct_pose_leg_gate_power_model = float(getattr(cfg, "direct_pose_leg_gate_power", 1.0) or 1.0)
    except Exception:
        direct_pose_leg_gate_power_model = 1.0
    if (not math.isfinite(direct_pose_leg_gate_power_model)) or direct_pose_leg_gate_power_model <= 0.0:
        direct_pose_leg_gate_power_model = 1.0
    return PostTrainModelBuildState(
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        state_dict=state_dict,
        width=width,
        contact_dim=contact_dim,
        angvel_dim=angvel_dim,
        pose_hist_dim=pose_hist_dim,
        contact_plan_enable=bool(contact_plan_enable),
        contact_plan_hidden=int(contact_plan_hidden or 64),
        contact_plan_inject=str(contact_plan_inject),
        contact_plan_time_pe_dim=int(contact_plan_time_pe_dim),
        contact_plan_init_mode=str(contact_plan_init_mode),
        contact_plan_init_hidden=int(contact_plan_init_hidden),
        contact_plan_init_dropout=float(contact_plan_init_dropout),
        use_event_clock=bool(event_clock_cfg.use_event_clock),
        event_clock_hidden_dim=int(event_clock_cfg.hidden_dim),
        event_clock_gate_hidden_dim=int(event_clock_cfg.gate_hidden_dim),
        event_clock_max_delta=float(event_clock_cfg.max_delta),
        period_dim_init=int(event_clock_cfg.period_dim_init),
        direct_pose_cfg=direct_pose_cfg,
        lambda_fusion_enable=bool(lambda_fusion_cfg.enable),
        lambda_fusion_mode=str(lambda_fusion_cfg.mode),
        lambda_fusion_hidden=int(lambda_fusion_cfg.hidden),
        lambda_fusion_dropout=float(lambda_fusion_cfg.dropout),
        lambda_fusion_logit_init=float(lambda_fusion_cfg.logit_init),
        lambda_fusion_use_rollout_step=bool(lambda_fusion_cfg.use_rollout_step),
        direct_pose_leg_gate_mode_model=str(direct_pose_leg_gate_mode_model),
        direct_pose_leg_gate_power_model=float(direct_pose_leg_gate_power_model),
    )


def _instantiate_posttrain_model(
    *,
    cfg: Any,
    ds: MotionEventDataset,
    device: torch.device,
    build_state: PostTrainModelBuildState,
) -> EventMotionModel:
    direct_pose_cfg = build_state.direct_pose_cfg
    model = EventMotionModel(
        in_state_dim=int(ds.Dx),
        out_motion_dim=int(ds.Dy),
        cond_dim=int(ds.Dc),
        period_dim=int(build_state.period_dim_init),
        hidden_dim=int(build_state.width),
        num_layers=int(cfg.depth),
        num_heads=int(cfg.num_heads),
        dropout=float(cfg.dropout),
        context_len=int(cfg.context_len),
        contact_dim=int(build_state.contact_dim),
        angvel_dim=int(build_state.angvel_dim),
        pose_hist_dim=int(build_state.pose_hist_dim),
        state_layout=getattr(ds, "state_layout", None),
        bone_names=getattr(ds, "bone_names", None),
        output_layout=getattr(ds, "output_layout", None),
        contact_plan_enable=bool(build_state.contact_plan_enable),
        contact_plan_hidden=int(build_state.contact_plan_hidden),
        contact_plan_dropout=0.0,
        contact_plan_inject=str(build_state.contact_plan_inject),
        contact_plan_inject_detach=True,
        contact_plan_time_pe_dim=int(build_state.contact_plan_time_pe_dim),
        contact_plan_init_mode=str(build_state.contact_plan_init_mode),
        contact_plan_init_hidden=int(build_state.contact_plan_init_hidden),
        contact_plan_init_dropout=float(build_state.contact_plan_init_dropout),
        use_event_clock=bool(build_state.use_event_clock),
        event_clock_max_delta=float(build_state.event_clock_max_delta),
        event_clock_hidden_dim=int(build_state.event_clock_hidden_dim),
        event_clock_gate_hidden_dim=int(build_state.event_clock_gate_hidden_dim),
        direct_pose_enable=bool(direct_pose_cfg.enable),
        direct_pose_hidden=int(direct_pose_cfg.hidden),
        direct_pose_dropout=0.0,
        direct_pose_detach_plan=True,
        direct_pose_meas_mode=str(direct_pose_cfg.meas_mode),
        direct_pose_meas_drop_prob=0.0,
        direct_pose_meas_noise_std=0.0,
        direct_pose_plan_drop_prob=0.0,
        direct_pose_feat_source=str(direct_pose_cfg.feat_source),
        direct_pose_time_pe_dim=int(direct_pose_cfg.time_pe_dim),
        direct_pose_time_pe_base=float(direct_pose_cfg.time_pe_base),
        direct_pose_use_phase_z=bool(direct_pose_cfg.use_phase_z),
        direct_pose_phase_z_mode=str(direct_pose_cfg.phase_z_mode),
        direct_pose_split_enable=bool(direct_pose_cfg.split_enable),
        direct_pose_stepc_unified_leg_terminal=bool(direct_pose_cfg.stepc_unified_leg_terminal),
        direct_pose_nonleg_proj_dim=int(direct_pose_cfg.nonleg_proj_dim),
        direct_pose_arm_split_enable=bool(direct_pose_cfg.arm_split_enable),
        direct_pose_arm_bones=direct_pose_cfg.arm_bones,
        direct_pose_leg_enable=bool(getattr(cfg, "direct_pose_leg_enable", False)),
        direct_pose_leg_bones=getattr(cfg, "direct_pose_leg_bones", None),
        direct_pose_leg_mode=str(getattr(cfg, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add"),
        direct_pose_leg_stopgrad_main=bool(getattr(cfg, "direct_pose_leg_stopgrad_main", False)),
        direct_pose_leg_detach_feat=bool(getattr(cfg, "direct_pose_leg_detach_feat", False)),
        direct_pose_leg_max_deg=float(getattr(cfg, "direct_pose_leg_max_deg", 0.0) or 0.0),
        direct_pose_leg_gate_mode=str(build_state.direct_pose_leg_gate_mode_model),
        direct_pose_leg_gate_power=float(build_state.direct_pose_leg_gate_power_model),
        direct_pose_leg_scale_log_clip=float(getattr(cfg, "direct_pose_leg_scale_log_clip", 4.0) or 4.0),
        direct_pose_leg_scale_clamp_k=float(getattr(cfg, "direct_pose_leg_scale_clamp_k", 0.0) or 0.0),
        lambda_fusion_enable=bool(build_state.lambda_fusion_enable),
        lambda_fusion_mode=str(build_state.lambda_fusion_mode),
        lambda_fusion_hidden=int(build_state.lambda_fusion_hidden),
        lambda_fusion_dropout=float(build_state.lambda_fusion_dropout),
        lambda_fusion_detach_err=True,
        lambda_fusion_logit_init=float(build_state.lambda_fusion_logit_init),
        lambda_fusion_use_rollout_step=bool(build_state.lambda_fusion_use_rollout_step),
    ).to(device)
    validate_and_fix_model_(model, int(ds.Dx), int(ds.Dc))
    return model


def _load_posttrain_checkpoint_into_model(
    *,
    cfg: Any,
    model: EventMotionModel,
    build_state: PostTrainModelBuildState,
) -> None:
    direct_pose_cfg = build_state.direct_pose_cfg
    state_dict = build_state.state_dict
    if cfg.encoder_bundle is not None and cfg.encoder_bundle.expanduser().is_file():
        _attach_motion_encoder_bundle(
            model,
            torch.load(str(cfg.encoder_bundle.expanduser()), map_location="cpu"),
        )

    _apply_direct_pose_ckpt_compat(
        state_dict=state_dict,
        model=model,
        ckpt_posttrain_cfg=build_state.ckpt_posttrain_cfg,
        contact_dim=int(build_state.contact_dim),
        direct_pose_cfg=direct_pose_cfg,
        load_options=DirectPoseLoadCompatOptions(
            train_direct_pose=bool(getattr(cfg, "train_direct_pose", False)),
            leg_enable=bool(getattr(cfg, "direct_pose_leg_enable", False)),
            leg_bones=getattr(cfg, "direct_pose_leg_bones", None),
        ),
    )

    model.load_state_dict(state_dict, strict=False)

    if cfg.train_direct_pose:
        if getattr(model, "direct_pose_head", None) is None:
            raise SystemExit("[FATAL] direct_pose_head is not instantiated; cannot train direct pose expert.")
        leg_only = bool(getattr(cfg, "direct_pose_leg_train_only", False))
        leg_gate_only = bool(getattr(cfg, "direct_pose_leg_gate_train_only", False))
        nonleg_only = bool(getattr(cfg, "direct_pose_nonleg_train_only", False))
        if nonleg_only and (leg_only or leg_gate_only):
            raise SystemExit(
                "[FATAL] direct_pose_nonleg_train_only=true is incompatible with leg train_only modes. "
                "Pick exactly one train_only mode."
            )
        if (leg_only or leg_gate_only) and getattr(model, "direct_pose_leg_head", None) is None:
            raise SystemExit(
                "[FATAL] direct_pose_leg_*_train_only=true but no leg head is instantiated. "
                "Enable direct_pose_leg_enable and provide valid direct_pose_leg_bones."
            )
        has_nonleg_branch = (
            getattr(model, "direct_pose_out_nonleg", None) is not None
            or (
                getattr(model, "direct_pose_out_arm", None) is not None
                and getattr(model, "direct_pose_out_else", None) is not None
            )
        )
        if nonleg_only and (not has_nonleg_branch):
            raise SystemExit(
                "[FATAL] direct_pose_nonleg_train_only=true but no non-leg branch is instantiated. "
                "Enable direct_pose_split_enable (optionally with direct_pose_arm_split_enable)."
            )
        if bool(leg_gate_only):
            has_leg_gate = (getattr(model, "direct_pose_leg_gate_head", None) is not None) or (
                getattr(model, "direct_pose_leg_gate_head_shared", None) is not None
            )
            if not has_leg_gate:
                raise SystemExit(
                    "[FATAL] direct_pose_leg_gate_train_only=true but no leg gate/scale head is instantiated. "
                    "Set direct_pose_leg_gate_mode='learned'/'scale' and enable direct_pose_leg_enable with valid bones."
                )
        if float(getattr(cfg, "direct_pose_leg_gate_sup_weight", 0.0) or 0.0) > 0.0:
            leg_mode = str(getattr(model, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add").strip().lower()
            if leg_mode != "so3":
                raise SystemExit(
                    "[FATAL] direct_pose_leg_gate_sup_weight>0 requires direct_pose_leg_mode='so3' "
                    f"(got {leg_mode!r})."
                )
            has_leg_gate = (getattr(model, "direct_pose_leg_gate_head", None) is not None) or (
                getattr(model, "direct_pose_leg_gate_head_shared", None) is not None
            )
            if not has_leg_gate:
                raise SystemExit(
                    "[FATAL] direct_pose_leg_gate_sup_weight>0 but no learned leg gate head is instantiated. "
                    "Set direct_pose_leg_gate_mode='learned' and enable direct_pose_leg_enable with valid bones."
                )
    if cfg.train_lambda_head:
        if getattr(model, "direct_pose_head", None) is None:
            raise SystemExit("[FATAL] Stage2 needs direct_pose_head (out_direct), but checkpoint/model does not enable it.")
        if getattr(model, "lambda_fusion_head", None) is None:
            raise SystemExit("[FATAL] Stage2 needs lambda_fusion_head, but it is not instantiated.")

    if cfg.so3_corr_gate_logit_reset is not None:
        logit = getattr(model, "so3_corr_gate_logit", None)
        if torch.is_tensor(logit):
            with torch.no_grad():
                logit.fill_(float(cfg.so3_corr_gate_logit_reset))
            print(f"[posttrain] reset so3_corr_gate_logit={float(cfg.so3_corr_gate_logit_reset):.4f}")


def _build_posttrain_model_from_ckpt(
    *,
    cfg: Any,
    ds: MotionEventDataset,
    device: torch.device,
) -> PostTrainModelArtifacts:
    build_state = _resolve_posttrain_model_build_state(cfg=cfg, ds=ds)
    model = _instantiate_posttrain_model(cfg=cfg, ds=ds, device=device, build_state=build_state)
    _load_posttrain_checkpoint_into_model(cfg=cfg, model=model, build_state=build_state)
    direct_pose_cfg = build_state.direct_pose_cfg
    return PostTrainModelArtifacts(
        model=model,
        direct_pose_feat_source=str(direct_pose_cfg.feat_source),
        direct_pose_time_pe_dim=int(direct_pose_cfg.time_pe_dim),
        direct_pose_time_pe_base=float(direct_pose_cfg.time_pe_base),
        direct_pose_use_phase_z=bool(direct_pose_cfg.use_phase_z),
        direct_pose_phase_z_mode=str(direct_pose_cfg.phase_z_mode),
        direct_pose_split_enable=bool(direct_pose_cfg.split_enable),
        direct_pose_nonleg_proj_dim=int(direct_pose_cfg.nonleg_proj_dim),
        direct_pose_leg_gate_mode_model=str(build_state.direct_pose_leg_gate_mode_model),
        direct_pose_leg_gate_power_model=float(build_state.direct_pose_leg_gate_power_model),
    )
