from __future__ import annotations

"""Action-handoff in-betweening — §7.3 §2.4 goal-injection binding gate (reusable parts).

This is the FIRST BINDING gate (can trigger the spec §6 STOP). It is still minimal: a
goal head injected into the base ``EventMotionModel`` + ``L_reach``-only short training +
the hidden_pre reach gate. NO AMP / foot / smooth loss, NO full scheduled sampling, NO
handoff runtime.

Two STEP-0 instrument findings (verified, surfaced by the probe — they shape the design):

1. **Per-step vs full-sequence capture mismatch.** The reach anchors were built from the
   z-probe's FULL-SEQUENCE teacher-forced ``hidden_pre`` (input to ``model._pasa_lnq``).
   Capturing ``hidden_pre`` PER-STEP during an autoregressive free-run (what the §4b cond
   floor probe did) lands in a DIFFERENT space — even feeding a clip's own GT motion
   per-step does NOT self-reach (min_norm ~8 vs ~0.3 teacher-forced). ⇒ the aligned,
   trustworthy capture is a FULL-SEQUENCE forward; the binding gate measures reach on the
   full-sequence ``hidden_pre`` of the goal-conditioned context, not a per-step rollout.

2. **Injection point.** ``hidden_pre`` = ``fw.h_temporal`` is consumed by Q/K/V AND the
   output residual; a delta added at ``_pasa_lnq`` (a LayerNorm, Q-path only) is decoupled
   from the output (Δout ~4e-6) so it cannot be trained by an output ``L_reach``. The
   minimal NON-INTRUSIVE lever is a ``forward_hook`` on ``model.residual_proj`` that adds
   the goal delta to its output: this shifts ``fw.h_temporal`` globally (so reach moves) and
   flows to the output (so ``L_reach`` has gradient). No edit to ``EventMotionModel.forward``.

The goal head is goal-conditioned (one head; the seam window is its input) and trained with
the base frozen. All thresholds (reach 0.7, conv_norm_thr, N, horizon, train steps)
PROVISIONAL.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from train.data.action_handoff_inbetween import (
    EGO_VEL_SLICE,
    POSE_DIM,
    POSE_SLICE,
    STATE_DIM,
    WALK_F,
)

HIDDEN_DIM = 512
GOAL_FLAT_DIM_DEFAULT = 6 * STATE_DIM  # K=6 seam window flattened (spec §1.2)
WALK_L_TO_R = "Walk_L_To_R"
DEFAULT_REACH_RATE_GATE = 0.7  # [PROVISIONAL] spec §6 make-or-break B1 signal
DEFAULT_YAW_MAE_TAU_RAD = 0.25  # [PROVISIONAL] 14.3 deg; separates recorded turns from W1a failures
DEFAULT_SELF_REACH_RATE_LIFT = 0.10  # [PROVISIONAL] with N=20, at least two-start lift over baseline
DEFAULT_POSE_DEGRADATION_TOL = 0.01  # [PROVISIONAL] pose_d slack vs no-goal/pinned baseline


# --------------------------------------------------------------------- torch parts
def _torch():
    import torch  # lazy import so numpy-only callers/tests stay light

    return torch


class GoalHead:
    """Goal-conditioned head: seam window [K,281] (flat) → injection vector.

    Implemented as a thin factory returning an ``nn.Module`` so this module imports without
    torch at definition time. ``mode='additive'`` returns ``[512]``; ``mode='film'`` returns
    ``[1024]`` = ``[gamma, beta]`` for FiLM-style injection.
    """

    @staticmethod
    def build(
        goal_flat_dim: int = GOAL_FLAT_DIM_DEFAULT,
        hidden: int = 256,
        depth: int = 1,
        init_scale: float = 0.0,
        mode: str = "additive",
    ):
        torch = _torch()
        import torch.nn as nn
        mode_l = str(mode or "additive").lower().strip()
        if mode_l not in ("additive", "film"):
            raise ValueError(f"unknown goal head mode: {mode!r}")
        out_dim = HIDDEN_DIM if mode_l == "additive" else 2 * HIDDEN_DIM

        class _GoalHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers = []
                d_in = int(goal_flat_dim)
                for _ in range(max(1, int(depth))):
                    layers.append(nn.Linear(d_in, int(hidden)))
                    layers.append(nn.GELU())
                    d_in = int(hidden)
                layers.append(nn.Linear(d_in, out_dim))
                self.net = nn.Sequential(*layers)
                self.mode = mode_l
                # start near-zero so the goal head is a no-op before training (init scale 0)
                self.scale = nn.Parameter(torch.tensor(float(init_scale)))
                nn.init.zeros_(self.net[-1].bias)
                nn.init.normal_(self.net[-1].weight, std=1e-3)

            def forward(self, goal_flat):  # [*, goal_flat_dim] -> [*, 512] or [*, 1024]
                return self.net(goal_flat) * self.scale

        return _GoalHead()


def register_goal_injection(model, delta):
    """Add ``delta`` ([512] or [B,512]) to ``model.residual_proj`` output → shifts h_temporal.

    Returns the hook handle (call ``.remove()``). Minimal non-intrusive injection (finding #2);
    the default forward path is unchanged when no hook is registered.
    """
    if not hasattr(model, "residual_proj"):
        raise RuntimeError(
            "EventMotionModel has no `residual_proj`; cannot inject goal non-intrusively. "
            "BLOCKER — report the minimal-intrusion alternative, do not force."
        )

    def _hook(_module, _inputs, output):
        d = delta
        if d.dim() == 1:
            d = d.view(1, 1, -1)
        elif d.dim() == 2:
            d = d.unsqueeze(1)
        return output + d

    return model.residual_proj.register_forward_hook(_hook)


class _HookGroup:
    """Small composite handle matching torch hook handles' ``remove`` API."""

    def __init__(self, handles):
        self._handles = list(handles)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()


def _goal_injection_target_modules(model, targets):
    if targets is None:
        target_list = ["shared_encoder.1"]
    elif isinstance(targets, str):
        target_list = [t.strip() for t in targets.split(",") if t.strip()]
    else:
        target_list = [str(t).strip() for t in targets if str(t).strip()]
    if not target_list:
        raise ValueError("at least one goal injection target is required")
    out = []
    for target in target_list:
        spec = target.replace("[", ".").replace("]", "").strip()
        spec_l = spec.lower()
        if spec_l in ("pre_temporal", "shared_encoder", "shared_encoder.1", "act1", "activation1"):
            idx = 1
        elif spec_l in ("early", "shared_encoder.0", "stem0", "linear0"):
            idx = 0
        elif spec_l in ("late", "shared_encoder.2", "stem2", "linear1"):
            idx = 2
        elif spec_l.startswith("shared_encoder."):
            idx = int(spec_l.rsplit(".", 1)[1])
        elif spec_l == "residual_proj":
            if not hasattr(model, "residual_proj"):
                raise RuntimeError("EventMotionModel has no `residual_proj`; cannot inject there.")
            out.append((target, model.residual_proj))
            continue
        else:
            raise ValueError(f"unknown goal injection target: {target!r}")
        if not hasattr(model, "shared_encoder"):
            raise RuntimeError(
                "EventMotionModel has no `shared_encoder`; cannot inject pre-temporal goal "
                "non-intrusively. BLOCKER — report, do not edit models.py."
            )
        try:
            out.append((target, model.shared_encoder[idx]))
        except Exception as exc:
            raise RuntimeError(f"EventMotionModel.shared_encoder[{idx}] is unavailable.") from exc
    return out


def _broadcast_goal_delta(delta, output, width: int):
    d = delta.to(device=output.device, dtype=output.dtype)
    if int(d.shape[-1]) != int(width):
        raise ValueError(f"goal injection width mismatch: expected {width}, got {int(d.shape[-1])}")
    if output.dim() == 3:
        if d.dim() == 1:
            return d.view(1, 1, -1)
        if d.dim() == 2:
            return d.unsqueeze(1)
    elif output.dim() == 2:
        if d.dim() == 1:
            return d.view(1, -1)
    return d


def register_goal_injection_pre_temporal(model, delta, *, targets="shared_encoder.1", mode: str = "additive"):
    """Inject ``delta`` before ``h_temporal`` is formed, without editing ``models.py``.

    By default the hook targets ``shared_encoder[1]`` (the first activation output, shape
    ``[B,T,512]`` or ``[B,512]``), so the goal signal flows through the remaining shared
    encoder before ``fw.h_temporal = h + residual_proj(x)``. ``targets`` may name one or
    more 512-wide modules, e.g. ``"shared_encoder.0"`` or
    ``"shared_encoder.0,shared_encoder.1"`` for the PHASE-1 capacity/injection ablation.

    ``mode='additive'`` consumes ``delta`` [512] or [B,512]. ``mode='film'`` consumes
    ``[1024]`` or ``[B,1024]`` as ``[gamma, beta]`` and applies
    ``output * (1 + gamma) + beta``. Returns a hook handle or composite handle.
    """
    mode_l = str(mode or "additive").lower().strip()
    if mode_l not in ("additive", "film"):
        raise ValueError(f"unknown goal injection mode: {mode!r}")
    target_modules = _goal_injection_target_modules(model, targets)

    def _hook(_module, _inputs, output):
        width = int(output.shape[-1])
        if mode_l == "film":
            d = delta.to(device=output.device, dtype=output.dtype)
            if int(d.shape[-1]) != 2 * width:
                raise ValueError(f"FiLM goal injection width mismatch: expected {2 * width}, got {int(d.shape[-1])}")
            gamma = _broadcast_goal_delta(d[..., :width], output, width)
            beta = _broadcast_goal_delta(d[..., width:], output, width)
            return output * (1.0 + gamma) + beta
        d = _broadcast_goal_delta(delta, output, width)
        return output + d

    handles = [module.register_forward_hook(_hook) for _, module in target_modules]
    return handles[0] if len(handles) == 1 else _HookGroup(handles)


def hidden_pre_anchor_loss(hidden_pre, centroid, *, end_window_k: int = 12, eps: float = 1e-8):
    """Differentiable cosine reach loss in hidden_pre space.

    ``hidden_pre`` is [T,512] or [B,T,512], dtype floating, on any torch device. ``centroid``
    is [512] or [B,512]. The loss is mean ``1-cos`` over the last ``end_window_k`` frames.
    It intentionally does not detach: gradients must flow from the hidden_pre distance back
    into the goal head delta. Norms are clamped by ``eps`` so zero vectors stay finite.
    """
    torch = _torch()
    h = hidden_pre
    if h.dim() == 2:
        h = h.unsqueeze(0)
    if h.dim() != 3 or int(h.shape[-1]) != HIDDEN_DIM:
        raise ValueError(f"hidden_pre must be [T,{HIDDEN_DIM}] or [B,T,{HIDDEN_DIM}], got {tuple(h.shape)}")
    k = int(max(1, min(int(end_window_k), int(h.shape[1]))))
    h = h[:, -k:, :]
    c = torch.as_tensor(centroid, device=h.device, dtype=h.dtype)
    if c.dim() == 1:
        c = c.view(1, 1, -1)
    elif c.dim() == 2:
        c = c.unsqueeze(1)
    if int(c.shape[-1]) != HIDDEN_DIM:
        raise ValueError(f"centroid last dim must be {HIDDEN_DIM}, got {tuple(c.shape)}")
    h_n = h / h.norm(dim=-1, keepdim=True).clamp_min(float(eps))
    c_n = c / c.norm(dim=-1, keepdim=True).clamp_min(float(eps))
    cos = (h_n * c_n).sum(dim=-1).clamp(-1.0, 1.0)
    dist = 1.0 - cos
    return dist.mean()


def egocentric_pose_egovel(out_raw, cond_dir):
    """Generated base-space output → (pose [.,276], ego_vel [.,2]) for ``L_reach`` (torch).

    Heading-invariant pose passes through; ego_vel = world-planar ``root_vel`` rotated into the
    context heading frame (spec §1.1), differentiable in the model's ``root_vel`` output.
    ``yaw_rate``/``contact`` are EXCLUDED from L_reach: they are not base-space model outputs
    (yaw_rate is command-derived from cond, contact is a separate stream).
    """
    torch = _torch()
    pose = out_raw[..., :POSE_DIM]
    rv = out_raw[..., POSE_DIM : POSE_DIM + 2]  # output layout: rot6d[0:276], rootvel[276:278]
    d = cond_dir / torch.clamp(torch.linalg.norm(cond_dir, dim=-1, keepdim=True), min=1e-8)
    ego_fwd = rv[..., 0] * d[..., 0] + rv[..., 1] * d[..., 1]
    ego_lat = -rv[..., 0] * d[..., 1] + rv[..., 1] * d[..., 0]
    ego = torch.stack([ego_fwd, ego_lat], dim=-1)
    return pose, ego


def l_reach(out_raw_endwindow, seam, std, cond_dir, *, w_pose: float = 1.0, w_ego: float = 1.0):
    """Group-normalized weighted L1 of (pose, ego_vel) vs the target seam window (spec §4).

    ``out_raw_endwindow`` [W,278] (raw model output, last frames), ``seam`` [K,281] target.
    Matches the last ``K`` frames; group-std normalization (pose / ego_vel groups). Returns a
    scalar torch loss (differentiable into the goal head).
    """
    torch = _torch()
    K = int(seam.shape[0])
    ew = out_raw_endwindow[-K:]
    pose, ego = egocentric_pose_egovel(ew, cond_dir[-K:] if cond_dir.dim() > 1 else cond_dir)
    seam_t = torch.as_tensor(np.asarray(seam, dtype=np.float32), device=ew.device)
    std_t = torch.as_tensor(np.asarray(std, dtype=np.float32), device=ew.device)
    pose_std = std_t[POSE_SLICE].clamp_min(1e-6)
    ego_std = std_t[EGO_VEL_SLICE].clamp_min(1e-6)
    pose_err = ((pose - seam_t[:, POSE_SLICE]) / pose_std).abs().mean()
    ego_err = ((ego - seam_t[:, EGO_VEL_SLICE]) / ego_std).abs().mean()
    return w_pose * pose_err + w_ego * ego_err


# --------------------------------------------------------------------- pure numpy parts
def calibration_relerr(captured: np.ndarray, saved: np.ndarray) -> float:
    """Relative L2 error between a full-sequence captured hidden_pre and the saved anchor source."""
    c = np.asarray(captured, dtype=np.float64)
    s = np.asarray(saved, dtype=np.float64)
    t = min(c.shape[0], s.shape[0])
    c, s = c[:t], s[:t]
    return float(np.linalg.norm(c - s) / (np.linalg.norm(s) + 1e-12))


def context_window_indices(step: int, context_len: int, seq_len: int, *, mode: str = "edge") -> np.ndarray:
    """Indices for per-step ``context_len`` capture, ending at ``step``.

    ``mode='edge'`` matches recorded non-periodic clips (left-pad with frame 0).
    ``mode='wrap'`` matches arbitrary Walk_F phase sampling on the periodic walk cycle.
    """
    c = int(context_len)
    n = int(seq_len)
    if c < 1:
        raise ValueError(f"context_len must be >= 1, got {context_len}")
    if n < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    raw = int(step) - c + 1 + np.arange(c, dtype=np.int64)
    mode_l = str(mode or "edge").strip().lower()
    if mode_l == "edge":
        return np.clip(raw, 0, n - 1).astype(np.int64, copy=False)
    if mode_l == "wrap":
        return (raw % n).astype(np.int64, copy=False)
    raise ValueError(f"unknown context window mode: {mode!r}")


def calibration_records_all_pass(records: Dict[str, Dict[str, object]], *, conv_norm_thr: float) -> bool:
    """Pure aggregation for LEVER-1 calibration records."""
    if not records:
        return False
    for rec in records.values():
        if not bool(rec.get("context_self_reached", False)):
            return False
        try:
            if float(rec.get("context_self_min_norm", float("inf"))) > float(conv_norm_thr):
                return False
        except Exception:
            return False
    return True


@dataclass
class ReachGateDecision:
    """Legacy radius-gate outcome retained for diagnostics/backward-compatible probes."""

    per_clip_pass: Dict[str, bool]
    l_to_r_pass: bool
    all_pass: bool
    lifted_above_floor: bool
    stop: bool
    reason: str


def reach_gate_decision(
    reach_rates: Dict[str, float],
    *,
    gate: float = DEFAULT_REACH_RATE_GATE,
    floor_rate: float = 0.0,
    l_to_r_clip: str = WALK_L_TO_R,
) -> ReachGateDecision:
    """Apply the legacy radius-normalized reach gate.

    This remains as a diagnostic for old summaries. New W1b binding decisions must use
    ``joint_action_binding_gate_decision`` so radius-normalized reach alone cannot pass B1.
    """
    per_clip_pass = {c: bool(r >= gate) for c, r in reach_rates.items()}
    l_to_r_pass = bool(per_clip_pass.get(l_to_r_clip, False))
    all_pass = bool(all(per_clip_pass.values())) if per_clip_pass else False
    lifted = bool(any(r > floor_rate + 1e-9 for r in reach_rates.values()))
    if not lifted:
        stop, reason = True, (
            f"reach_rate did not lift above the §4b floor ({floor_rate:.2f}) for any target → "
            "goal injection + minimal L_reach is insufficient; STOP and reconsider (spec §6), "
            "do not expand to AMP/curriculum."
        )
    elif not l_to_r_pass:
        stop, reason = True, (
            f"{l_to_r_clip} (zero grounded supervision) below the reach gate ({gate:.2f}) → "
            "the B1 risk concentration fails; STOP per spec §6 (per-clip, not aggregate)."
        )
    elif not all_pass:
        stop, reason = False, (
            f"reach lifted and {l_to_r_clip} passes, but not all targets clear {gate:.2f}; "
            "PROVISIONAL — proceed cautiously / iterate, not a clean STOP."
        )
    else:
        stop, reason = False, f"all targets clear the reach gate ({gate:.2f}); B1 has a foundation."
    return ReachGateDecision(
        per_clip_pass=per_clip_pass,
        l_to_r_pass=l_to_r_pass,
        all_pass=all_pass,
        lifted_above_floor=lifted,
        stop=stop,
        reason=reason,
    )


@dataclass
class JointActionBindingGateDecision:
    """Migrated B1 gate: same-source self-reach plus action-layer evidence."""

    per_clip_pass: Dict[str, bool]
    per_clip_checks: Dict[str, Dict[str, bool]]
    l_to_r_pass: bool
    all_pass: bool
    stop: bool
    reason: str
    thresholds: Dict[str, float]
    hidden_pre_self_reach_is_necessary_not_sufficient: bool = True


def _finite_or_nan(value: object) -> float:
    try:
        v = float(value)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _nested_rate(row: Mapping[str, object], k_label: str) -> float:
    direct_keys = (
        f"self_reach_rate_{k_label.replace('=', '')}",
        "self_reach_rate_k3" if k_label == "k=3" else "",
        "self_reach_rate",
    )
    for key in direct_keys:
        if key and key in row:
            return _finite_or_nan(row.get(key))
    gate = row.get("self_reach_gate", {})
    if isinstance(gate, Mapping):
        rates = gate.get("rate_by_k", {})
        if isinstance(rates, Mapping) and k_label in rates:
            return _finite_or_nan(rates.get(k_label))
    return float("nan")


def _first_finite(row: Mapping[str, object], keys: Sequence[str]) -> float:
    for key in keys:
        if key in row:
            v = _finite_or_nan(row.get(key))
            if np.isfinite(v):
                return v
    return float("nan")


def _baseline_rate_for_clip(
    clip: str,
    *,
    k_label: str,
    baseline_metrics: Sequence[Mapping[str, Mapping[str, object]]],
    fallback: object = None,
) -> float:
    vals: List[float] = []
    fb = _finite_or_nan(fallback)
    if np.isfinite(fb):
        vals.append(fb)
    for table in baseline_metrics:
        row = table.get(clip, {}) if isinstance(table, Mapping) else {}
        if isinstance(row, Mapping):
            v = _nested_rate(row, k_label)
            if np.isfinite(v):
                vals.append(v)
    return float(max(vals)) if vals else 0.0


def _baseline_pose_for_clip(
    clip: str,
    *,
    baseline_metrics: Sequence[Mapping[str, Mapping[str, object]]],
    fallback: object = None,
) -> float:
    vals: List[float] = []
    fb = _finite_or_nan(fallback)
    if np.isfinite(fb):
        vals.append(fb)
    for table in baseline_metrics:
        row = table.get(clip, {}) if isinstance(table, Mapping) else {}
        if isinstance(row, Mapping):
            v = _first_finite(row, ("best_pose_d", "best_pose_d_mean", "mean_best_pose_d"))
            if np.isfinite(v):
                vals.append(v)
    return float(min(vals)) if vals else float("nan")


def joint_action_binding_gate_decision(
    per_clip_metrics: Mapping[str, Mapping[str, object]],
    *,
    baseline_metrics: Sequence[Mapping[str, Mapping[str, object]]] = (),
    k_label: str = "k=3",
    min_reach_lift: float = DEFAULT_SELF_REACH_RATE_LIFT,
    tau_yaw_rad: float = DEFAULT_YAW_MAE_TAU_RAD,
    pose_degradation_tol: float = DEFAULT_POSE_DEGRADATION_TOL,
    l_to_r_clip: str = WALK_L_TO_R,
) -> JointActionBindingGateDecision:
    """Apply the W1b joint B1 criterion per clip.

    Required simultaneously:
    1. same-source self-reach rate at ``k_label`` lifts over no-goal/pinned baselines;
    2. realized-yaw corr > 0 and heading MAE < ``tau_yaw_rad``;
    3. pop_safe_rate > 0;
    4. best_pose_d does not degrade against the best available baseline.
    """
    per_clip_pass: Dict[str, bool] = {}
    per_clip_checks: Dict[str, Dict[str, bool]] = {}
    reasons: List[str] = []
    for clip, row in per_clip_metrics.items():
        if not isinstance(row, Mapping):
            row = {}
        rate = _nested_rate(row, k_label)
        baseline_rate = _baseline_rate_for_clip(
            clip,
            k_label=k_label,
            baseline_metrics=baseline_metrics,
            fallback=row.get("baseline_self_reach_rate"),
        )
        corr = _first_finite(row, ("yaw_corr", "yaw_corr_mean", "realized_yaw_corr"))
        heading_mae = _first_finite(row, ("heading_mae_rad", "yaw_heading_mae_rad_mean", "realized_heading_mae_rad"))
        pop_safe = _first_finite(row, ("pop_safe_rate",))
        best_pose = _first_finite(row, ("best_pose_d", "best_pose_d_mean", "mean_best_pose_d"))
        baseline_pose = _baseline_pose_for_clip(
            clip,
            baseline_metrics=baseline_metrics,
            fallback=row.get("baseline_best_pose_d"),
        )
        reach_available = bool(row.get("reach_available", True))
        if not np.isfinite(baseline_pose):
            pose_ok = bool(np.isfinite(best_pose))
        else:
            pose_ok = bool(np.isfinite(best_pose) and best_pose <= baseline_pose + float(pose_degradation_tol))
        checks = {
            "same_source_reach_available": reach_available,
            f"self_reach_{k_label}_rate_lifted_vs_baseline": bool(
                reach_available
                and np.isfinite(rate)
                and rate >= baseline_rate + float(min_reach_lift)
            ),
            "realized_yaw_corr_positive": bool(np.isfinite(corr) and corr > 0.0),
            "heading_mae_under_tau": bool(np.isfinite(heading_mae) and heading_mae < float(tau_yaw_rad)),
            "pop_safe_positive": bool(np.isfinite(pop_safe) and pop_safe > 0.0),
            "best_pose_not_degraded": pose_ok,
        }
        passed = bool(all(checks.values()))
        per_clip_pass[str(clip)] = passed
        per_clip_checks[str(clip)] = checks
        if not passed:
            failed = [name for name, ok in checks.items() if not ok]
            reasons.append(f"{clip}: failed {', '.join(failed)}")
    l_to_r_pass = bool(per_clip_pass.get(l_to_r_clip, False))
    all_pass = bool(all(per_clip_pass.values())) if per_clip_pass else False
    stop = bool(not l_to_r_pass) if l_to_r_clip in per_clip_pass else bool(not all_pass)
    if all_pass:
        reason = "all clips pass same-source self-reach + action-layer joint criterion"
    elif reasons:
        reason = "; ".join(reasons)
    else:
        reason = "no per-clip metrics available"
    return JointActionBindingGateDecision(
        per_clip_pass=per_clip_pass,
        per_clip_checks=per_clip_checks,
        l_to_r_pass=l_to_r_pass,
        all_pass=all_pass,
        stop=stop,
        reason=reason,
        thresholds={
            "min_reach_lift": float(min_reach_lift),
            "tau_yaw_rad": float(tau_yaw_rad),
            "pose_degradation_tol": float(pose_degradation_tol),
        },
    )


def loss_plateau_status(
    values: Sequence[float],
    *,
    min_steps: int,
    window: int,
    rel_delta: float,
) -> Dict[str, object]:
    """Plateau test on a monitored curve (train hidden loss OR eval min_norm).

    "Plateau" means the curve has gone FLAT — ``|relative_improvement| <= rel_delta`` between
    the previous ``window`` samples and the most recent ``window`` samples. This is the strict
    "持平" reading required for a NEGATIVE upgrade: it excludes a curve that is still
    descending (``rel > thr`` → keep training) AND a curve that is worsening / unstable
    (``rel < -thr`` → lower the LR, do not upgrade). Needs ``max(min_steps, 2*window)`` finite
    samples before it will return ``plateau=True`` at all. Pure / numpy-only.
    """
    finite = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    w = int(max(1, window))
    min_n = int(max(1, min_steps))
    need = int(max(min_n, 2 * w))
    if int(finite.size) < need:
        return {
            "plateau": False,
            "reason": f"insufficient samples for plateau check: n={int(finite.size)} < {need}",
            "n": int(finite.size),
            "prev_window_mean": float("nan"),
            "recent_window_mean": float("nan"),
            "relative_improvement": float("nan"),
        }
    prev_m = float(np.mean(finite[-2 * w : -w]))
    recent_m = float(np.mean(finite[-w:]))
    rel = float((prev_m - recent_m) / max(abs(prev_m), 1e-8))
    thr = float(rel_delta)
    plateau = bool(abs(rel) <= thr)
    if plateau:
        reason = f"plateaued: |relative_improvement|={abs(rel):.4f} <= {thr:.4f}"
    elif rel > thr:
        reason = f"still improving: relative_improvement={rel:.4f} > {thr:.4f}"
    else:
        reason = f"worsened/unstable: relative_improvement={rel:.4f} < -{thr:.4f}"
    return {
        "plateau": plateau,
        "reason": reason,
        "n": int(finite.size),
        "prev_window_mean": prev_m,
        "recent_window_mean": recent_m,
        "relative_improvement": rel,
    }


def summarize_reach_rate(min_norms: Sequence[float], conv_norm_thr: float) -> Dict[str, float]:
    """reach_rate (fraction reached) + reach_min_norm distribution over starts."""
    arr = np.asarray([m for m in min_norms if np.isfinite(m)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "reach_rate": float("nan"), "reach_min_norm_mean": float("nan"),
                "reach_min_norm_median": float("nan"), "reach_min_norm_min": float("nan")}
    return {
        "n": int(arr.size),
        "reach_rate": float(np.mean(arr <= float(conv_norm_thr))),
        "reach_min_norm_mean": float(np.mean(arr)),
        "reach_min_norm_median": float(np.median(arr)),
        "reach_min_norm_min": float(np.min(arr)),
    }
