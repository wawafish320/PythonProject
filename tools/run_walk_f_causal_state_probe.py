#!/usr/bin/env python3
"""Walk_F causal-state scaffold v1 — read-only probes.

Implemented modes are read-only:

* ``yaw_debug`` (Layer 0 / Track 1) — Walk-turn descriptive oracle on
  raw ``RootYaw``. Debug-only; must not be promoted to attractor /
  transition truth.
* ``gauge_check`` (Layer A) — canonical-yaw-quotient + synthetic
  global-yaw rotation grid + ``+/-pi`` wrap boundary, as required by
  ``docs/aperiodic_transition/2026-05-22_walk_f_causal_state_scaffold_v1.md``
  §1.2. Prerequisite for the future causal-state membership track.
  Does NOT estimate causal-state membership, attractor membership,
  phase library, predictive loss, or EventHead targets.
* ``phase_library_check`` (Layer C minimal) — Walk_F-only phase-library
  self-consistency under a fixed estimator grid. It does NOT run query
  leave/return boundary detection and does NOT emit membership labels.
* ``query_boundary_check`` (Layer C.1) — Walk_F self-baseline band
  + query leave/return interval + censoring audit on the 4 turn clips.
  Strictly read-only. It does NOT emit attractor membership,
  cross-attractor claims, EventHead targets, ``handoff_ready``,
  ``transition_done``, or any combined-membership score. ``turn_dyn``
  is forbidden from combined membership by contract; routing it through
  the boundary path raises ``FailFastError``.
* ``pose_reference_scale_check`` (Layer B.1) —
  ``raw_data/processed_data/*.npz`` schema + Walk_F-only ``pose_dyn``
  / ``pose_rel`` reference scale + degeneracy + query inspection
  projected onto Walk_F scale, on the 5-clip set. Strictly read-only.
  It does NOT emit query leave/return intervals, attractor membership,
  combined-membership scores, EventHead targets, ``handoff_ready``,
  ``transition_done``, cross-attractor claims, or transition truth.
  ``turn_dyn`` and any query self-median/MAD rescaling raise
  ``FailFastError``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


CONTRACT = "walk_f_causal_state_scaffold_v1"
TOOL_NAME = "run_walk_f_causal_state_probe"
REFERENCE_FAMILY = ["Walk_F"]

SUPPORTED_MODES = (
    "yaw_debug",
    "gauge_check",
    "reference_scale_check",
    "phase_library_check",
    "query_boundary_check",
    "pose_reference_scale_check",
)

TRACK_BY_MODE = {
    "yaw_debug": "yaw_activity_debug",
    "gauge_check": "walk_f_causal_state_scaffold_prerequisite",
    "reference_scale_check": "walk_f_reference_scale_and_degeneracy",
    "phase_library_check": "walk_f_phase_library_self_consistency",
    "query_boundary_check": "walk_f_query_leave_return_boundary_audit",
    "pose_reference_scale_check": (
        "walk_f_pose_reference_scale_and_degeneracy"
    ),
}
TRACK_ROLE_BY_MODE = {
    "yaw_debug": "debug_only_not_attractor_definition",
    "gauge_check": "gauge_sanity_prerequisite_not_attractor_definition",
    "reference_scale_check": "reference_scale_and_degeneracy_prerequisite_not_membership",
    "phase_library_check": "walk_f_only_self_consistency_not_membership_boundary",
    "query_boundary_check": (
        "query_boundary_audit_not_attractor_membership_not_transition_truth"
    ),
    "pose_reference_scale_check": (
        "pose_reference_scale_and_degeneracy_prerequisite_not_membership"
    ),
}
SCOPE_BY_MODE = {
    "yaw_debug": "walk_turn_yaw_activity_descriptive_oracle",
    "gauge_check": "canonical_yaw_quotient_invariance_sanity",
    "reference_scale_check": "walk_f_per_feature_group_reference_scale_and_degeneracy",
    "phase_library_check": "walk_f_phase_library_self_consistency_single_trajectory",
    "query_boundary_check": (
        "walk_f_self_baseline_band_plus_query_leave_return_with_censoring_audit"
    ),
    "pose_reference_scale_check": (
        "walk_f_pose_dyn_pose_rel_reference_scale_and_degeneracy_processed_data_only"
    ),
}

# yaw_debug constants -----------------------------------------------------
ABS_DEG_PER_S_GRID = (3.0, 5.0, 10.0)
REL_TO_PEAK_GRID = (0.05, 0.10)
STRAIGHT_TOTAL_DELTA_DEG_TOL = 1.0
DEGENERATE_PEAK_DEG_PER_S = 1e-6

# gauge_check constants ---------------------------------------------------
ROTATION_GRID_RAD = (
    math.pi / 6.0,
    math.pi / 2.0,
    math.pi,
    -math.pi,
)
# Synthetic planar translations (meters) applied to root_pos_xy only;
# RootYaw and RootVelocityXY are intentionally not translated. The
# canonical gauge subtracts root_pos_xy[0] so by construction these
# perturbations should leave every quotient feature unchanged. The
# grid is deliberately mixed: axis-aligned and an off-axis, non-round
# value to reduce coincidental cancellation.
TRANSLATION_GRID_M = (
    (1.0, 0.0),
    (0.0, 1.0),
    (-3.7, 2.4),
)
# Numeric tolerance for the SE(2) quotient sanity. Estimator-level
# numeric check ONLY; NOT contract definition. The artifact always
# reports the actual max/median errors so a downstream reviewer can
# re-judge the tolerance. The same tolerance is applied to both the
# yaw rotation grid and the translation grid.
GAUGE_MAX_ABS_ERR_TOL = 1.0e-8
# A frame-to-frame jump in the wrapped yaw signal larger than this is
# treated as a wrap event for diagnostic purposes only.
WRAP_JUMP_DETECT_RAD = math.pi

# reference_scale_check (Layer B) constants -------------------------------
# Per-group absolute MAD thresholds. Below this, the group is marked
# `reference_degenerate=True` on Walk_F. These are estimator-level
# numeric thresholds and are NOT contract definition; the artifact
# always reports the actual MAD / std / ptp / mean_abs so a reviewer
# can re-judge the threshold. Each group's unit is documented inline.
FEATURE_GROUPS_LAYER_B: dict[str, dict[str, Any]] = {
    "root_body": {
        "channels": (
            "body_root_pos_xy.x",
            "body_root_pos_xy.y",
            "body_root_vel_xy.x",
            "body_root_vel_xy.y",
        ),
        "unit_label": "meters_or_meters_per_second",
        "epsilon_mad": 1.0e-6,
        "source": "canonical_quotient_feature_layer_a",
    },
    "turn_dyn": {
        "channels": (
            "yaw_rel_rad",
            "yaw_rate_rad_per_s",
        ),
        "unit_label": "radians_or_radians_per_second",
        "epsilon_mad": 1.0e-6,
        "source": "canonical_quotient_feature_layer_a",
    },
    "contact": {
        "channels": (
            "foot_L_soft_contact_score",
            "foot_R_soft_contact_score",
        ),
        "unit_label": "dimensionless_in_unit_interval",
        "epsilon_mad": 1.0e-4,
        "source": "raw_data_foot_evidence_soft_contact_score",
    },
}
# Feature groups defined by §5.2 but intentionally not run at Layer B.
# pose_dyn requires bone angular velocity summaries (processed_data/*.npz),
# which would add a second data source dependency outside Layer B's scope.
# pose_rel needs relative pose delta summaries with explicit avoidance of
# raw-pose templating; deferred.
NOT_RUN_FEATURE_GROUPS_LAYER_B = ("pose_dyn", "pose_rel")
# Minimum frame count under which phase estimability is not even
# considered; Walk_F currently has 88 frames so this is informational.
MIN_FRAMES_FOR_PHASE_ESTIMABILITY = 4
# MAD-to-Gaussian-equivalent-sigma constant; reported alongside raw MAD
# so a reviewer can compare against a std baseline if they want to.
MAD_TO_GAUSSIAN_SIGMA = 1.4826

# phase_library_check (Layer C minimal) constants --------------------------
LAYER_C_HISTORY_WINDOW_FRAMES = (6, 12)
LAYER_C_FUTURE_HORIZON_FRAMES = (6, 12)
LAYER_C_NEIGHBORHOOD_RADIUS_FRAMES = (4, 8)
LAYER_C_DISTANCE_METRICS = ("z_mse", "z_l1")
LAYER_C_MIN_VALID_QUERY_FRACTION = 0.50
LAYER_C_MIN_VALID_QUERY_COUNT = 12
LAYER_C_IMPROVEMENT_TOL = 0.0
LAYER_C_FEATURE_GROUPS: dict[str, dict[str, Any]] = {
    "root_body_vel_only": {
        "channels": ("body_root_vel_xy.x", "body_root_vel_xy.y"),
        "source": "canonical_quotient_feature_layer_a",
        "group_ablation_role": "preferred_root_body_no_clip_start_displacement",
        "epsilon_mad": FEATURE_GROUPS_LAYER_B["root_body"]["epsilon_mad"],
    },
    "root_body_pos_vel": {
        "channels": (
            "body_root_pos_xy.x",
            "body_root_pos_xy.y",
            "body_root_vel_xy.x",
            "body_root_vel_xy.y",
        ),
        "source": "canonical_quotient_feature_layer_a",
        "group_ablation_role": (
            "reported_ablation_only_clip_start_displacement_may_leak_elapsed_progress"
        ),
        "epsilon_mad": FEATURE_GROUPS_LAYER_B["root_body"]["epsilon_mad"],
    },
    "contact": {
        "channels": (
            "foot_L_soft_contact_score",
            "foot_R_soft_contact_score",
        ),
        "source": "raw_data_foot_evidence_soft_contact_score",
        "group_ablation_role": "independent_contact_self_consistency_view",
        "epsilon_mad": FEATURE_GROUPS_LAYER_B["contact"]["epsilon_mad"],
    },
}
LAYER_C_EXCLUDED_FEATURE_GROUPS: dict[str, str] = {
    "turn_dyn": (
        "excluded because Layer B marks Walk_F turn_dyn degenerate; do not "
        "normalize or combine zero-variance yaw dynamics into phase evidence"
    ),
    "pose_dyn": "not_run; processed_data/*.npz schema not validated in Layer C minimal",
    "pose_rel": "not_run; needs a separate non-template contract",
    "absolute_RootYaw": "excluded by planar translation / global-yaw quotient contract",
    "yaw_activity_debug": "debug oracle only; not phase-library evidence",
}

# query_boundary_check (Layer C.1) constants -------------------------------
# Layer C.1 reuses every Layer C minimal estimator dim and adds three
# boundary-detection dims (band quantile + min consecutive frames for
# leave / return). turn_dyn is intentionally absent from
# LAYER_C1_FEATURE_GROUPS: any combined-membership entry point that reads
# turn_dyn must raise FailFastError (see _layer_c1_forbid_turn_dyn).
LAYER_C1_REFERENCE_CLIP = "Walk_F"
LAYER_C1_QUERY_FAMILY: tuple[str, ...] = (
    "Walk_L_To_L",
    "Walk_L_To_R",
    "Walk_R_To_L",
    "Walk_R_To_R",
)
LAYER_C1_REQUIRED_CLIP_SET: frozenset[str] = frozenset(
    [LAYER_C1_REFERENCE_CLIP, *LAYER_C1_QUERY_FAMILY]
)
LAYER_C1_FEATURE_GROUPS: dict[str, dict[str, Any]] = LAYER_C_FEATURE_GROUPS
LAYER_C1_EXCLUDED_FEATURE_GROUPS: dict[str, str] = LAYER_C_EXCLUDED_FEATURE_GROUPS
LAYER_C1_BAND_QUANTILES: tuple[tuple[str, float], ...] = (
    ("P75", 0.75),
    ("P90", 0.90),
    ("P95", 0.95),
)
LAYER_C1_MIN_CONSEC_FRAMES_FOR_LEAVE: tuple[int, ...] = (2, 4)
LAYER_C1_MIN_CONSEC_FRAMES_FOR_RETURN: tuple[int, ...] = (2, 4)
# Neighbor-consistency tolerance is the smaller of the leave grid. Two
# configs that differ by exactly one grid step in exactly one dim are
# "neighbors". If their leave/return endpoint frames disagree by more
# than this many frames (or their existence boolean differs), the
# per-config return_to_reference_status is overridden to
# INSUFFICIENT_EVIDENCE.
LAYER_C1_NEIGHBOR_CONSISTENCY_TOL_FRAMES: int = int(
    min(LAYER_C1_MIN_CONSEC_FRAMES_FOR_LEAVE)
)
LAYER_C1_MIN_VALID_QUERY_COUNT: int = 12
LAYER_C1_MIN_VALID_QUERY_FRACTION: float = 0.50

# pose_reference_scale_check (Layer B.1) constants ------------------------
# Layer B.1 consumes raw_data/processed_data/<clip>.npz. It does NOT
# re-derive query per-channel medians; Walk_F is the only reference. It
# does NOT compute any combined-membership / combined-score and does
# NOT emit leave/return intervals. turn_dyn is hard-rejected at every
# entry point.
LAYER_B1_REFERENCE_CLIP: str = "Walk_F"
LAYER_B1_QUERY_FAMILY: tuple[str, ...] = (
    "Walk_L_To_L",
    "Walk_L_To_R",
    "Walk_R_To_L",
    "Walk_R_To_R",
)
LAYER_B1_REQUIRED_CLIP_SET: frozenset[str] = frozenset(
    [LAYER_B1_REFERENCE_CLIP, *LAYER_B1_QUERY_FAMILY]
)
LAYER_B1_PROCESSED_DATA_SUBDIR: str = "processed_data"
LAYER_B1_BONE_COUNT: int = 46
LAYER_B1_POSE_DYN_AXES: tuple[str, ...] = ("x", "y", "z")
LAYER_B1_POSE_REL_COMPONENT_LABELS: tuple[str, ...] = (
    "c0",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
)
# Per-group epsilon_mad: bone-local quiescent bones (fingers etc.) can
# sit near zero on a forward walk; threshold is upper-inclusive.
LAYER_B1_FEATURE_GROUPS: dict[str, dict[str, Any]] = {
    "pose_dyn": {
        "source": "processed_data_npz_bone_ang_vel_local_rad_per_sec",
        "source_npz_key": "bone_ang_vel",
        "source_rank": 3,
        "source_last_dim": 3,
        "extracted_channel_count": LAYER_B1_BONE_COUNT * 3,
        "unit_label": "rad_per_sec_bone_local",
        "epsilon_mad": 1.0e-4,
        "group_ablation_role": (
            "bone_local_angular_velocity_pose_dynamics"
        ),
    },
    "pose_rel": {
        "source": (
            "processed_data_npz_bone_rot6d_local_first_difference_times_fps"
        ),
        "source_npz_key": "bone_rot6d",
        "source_rank": 3,
        "source_last_dim": 6,
        "extracted_channel_count": LAYER_B1_BONE_COUNT * 6,
        "unit_label": "local_6d_component_per_sec",
        "epsilon_mad": 1.0e-4,
        "group_ablation_role": (
            "bone_local_rot6d_first_difference_relative_pose_delta_velocity"
        ),
    },
}
LAYER_B1_EXCLUDED_FEATURE_GROUPS: dict[str, str] = {
    "turn_dyn": (
        "excluded by Layer B.1 contract §1 rule 3; turn_dyn is "
        "reference-degenerate on Walk_F per Layer B and MUST NEVER be "
        "folded into combined-membership / combined-score evidence"
    ),
    "root_body": (
        "not_in_scope_layerB1; covered by reference_scale_check (Layer B) "
        "and phase_library_check (Layer C minimal)"
    ),
    "root_body_vel_only": (
        "not_in_scope_layerB1; covered by phase_library_check (Layer C minimal)"
    ),
    "root_body_pos_vel": (
        "not_in_scope_layerB1; covered by phase_library_check (Layer C minimal)"
    ),
    "contact": (
        "not_in_scope_layerB1; covered by reference_scale_check (Layer B) "
        "and phase_library_check (Layer C minimal)"
    ),
    "absolute_RootYaw": (
        "excluded by canonical planar translation / global-yaw quotient "
        "(scaffold v1 §1.2)"
    ),
    "yaw_activity_debug": "debug oracle only; not pose-scale evidence",
}


class FailFastError(RuntimeError):
    """Raised when raw clip / metadata does not satisfy the probe contract."""


def _serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# -----------------------------------------------------------------------
# Raw-clip loading (shared)
# -----------------------------------------------------------------------


def _load_clip(raw_root: Path, clip: str) -> dict[str, Any]:
    """Read a single raw JSON clip and extract the per-frame arrays
    needed by both modes.

    Returns a dict with float64 CPU numpy arrays:

      * ``root_yaw_rad``     shape ``(T,)``
      * ``root_pos_xy_m``    shape ``(T, 2)``  (xy-plane only; z dropped)
      * ``root_vel_xy_mps``  shape ``(T, 2)``

    Fail-fast on missing file, missing ``Frames`` / ``FPS`` / per-frame
    fields, wrong unit, non-scalar yaw, wrong vector shape, non-finite
    values, or ``NumFrames`` / ``len(Frames)`` mismatch.
    """
    path = raw_root / f"{clip}.json"
    if not path.is_file():
        raise FailFastError(
            f"[walk_f_causal_state_probe] raw clip missing: {path}. "
            f"contract={CONTRACT} requires all listed clips to be present; "
            f"no silent skip is allowed."
        )
    raw = json.loads(path.read_text())
    if "Frames" not in raw or not isinstance(raw["Frames"], list) or len(raw["Frames"]) == 0:
        raise FailFastError(
            f"[walk_f_causal_state_probe] clip {clip!r} missing non-empty Frames "
            f"in {path}."
        )
    if "FPS" not in raw:
        raise FailFastError(
            f"[walk_f_causal_state_probe] clip {clip!r} missing FPS in {path}."
        )
    fps = float(raw["FPS"])
    if not np.isfinite(fps) or fps <= 0:
        raise FailFastError(
            f"[walk_f_causal_state_probe] clip {clip!r} non-positive FPS={fps} "
            f"in {path}."
        )

    yaw_units = (
        raw.get("meta", {})
        .get("root_motion", {})
        .get("root_yaw_units")
    )
    if yaw_units != "radians":
        raise FailFastError(
            f"[walk_f_causal_state_probe] clip {clip!r} root_yaw_units="
            f"{yaw_units!r}; this probe contract expects radians. "
            f"Refusing silent unit conversion."
        )

    units = raw.get("meta", {}).get("units")
    if units != "meters":
        raise FailFastError(
            f"[walk_f_causal_state_probe] clip {clip!r} meta.units={units!r}; "
            f"this probe contract expects 'meters' so RootPosition / "
            f"RootVelocityXY are dimensionally consistent. Refusing silent "
            f"unit conversion."
        )

    yaws: list[float] = []
    pos_xy: list[list[float]] = []
    vel_xy: list[list[float]] = []
    for i, frame in enumerate(raw["Frames"]):
        for key in ("RootYaw", "RootPosition", "RootVelocityXY"):
            if key not in frame:
                raise FailFastError(
                    f"[walk_f_causal_state_probe] clip {clip!r} frame {i} "
                    f"missing {key} in {path}."
                )

        v_yaw = frame["RootYaw"]
        if isinstance(v_yaw, (list, tuple)) or not isinstance(v_yaw, (int, float)):
            raise FailFastError(
                f"[walk_f_causal_state_probe] clip {clip!r} frame {i} RootYaw "
                f"is not scalar (got {type(v_yaw).__name__}) in {path}."
            )
        yaws.append(float(v_yaw))

        v_pos = frame["RootPosition"]
        if not isinstance(v_pos, (list, tuple)) or len(v_pos) != 3:
            raise FailFastError(
                f"[walk_f_causal_state_probe] clip {clip!r} frame {i} "
                f"RootPosition shape != (3,) (got {v_pos!r}) in {path}."
            )
        pos_xy.append([float(v_pos[0]), float(v_pos[1])])

        v_vel = frame["RootVelocityXY"]
        if not isinstance(v_vel, (list, tuple)) or len(v_vel) != 2:
            raise FailFastError(
                f"[walk_f_causal_state_probe] clip {clip!r} frame {i} "
                f"RootVelocityXY shape != (2,) (got {v_vel!r}) in {path}."
            )
        vel_xy.append([float(v_vel[0]), float(v_vel[1])])

    if "NumFrames" in raw and int(raw["NumFrames"]) != len(yaws):
        raise FailFastError(
            f"[walk_f_causal_state_probe] clip {clip!r} NumFrames="
            f"{raw['NumFrames']} disagrees with len(Frames)={len(yaws)} in "
            f"{path}."
        )

    yaw_arr = np.asarray(yaws, dtype=np.float64)
    pos_arr = np.asarray(pos_xy, dtype=np.float64)
    vel_arr = np.asarray(vel_xy, dtype=np.float64)

    for name, arr in (
        ("RootYaw", yaw_arr),
        ("RootPosition.xy", pos_arr),
        ("RootVelocityXY", vel_arr),
    ):
        if not np.all(np.isfinite(arr)):
            raise FailFastError(
                f"[walk_f_causal_state_probe] clip {clip!r} contains "
                f"non-finite values in {name} in {path}."
            )

    return {
        "clip": clip,
        "raw_json_path": str(path),
        "fps": fps,
        "frame_count": len(yaws),
        "root_yaw_units_raw": yaw_units,
        "root_units_raw": units,
        "root_yaw_rad": yaw_arr,           # (T,)
        "root_pos_xy_m": pos_arr,          # (T, 2)
        "root_vel_xy_mps": vel_arr,        # (T, 2)
    }


def _load_clip_contact_signals(
    raw_root: Path,
    clip: str,
) -> dict[str, np.ndarray]:
    """Read per-frame ``FootEvidence.{L,R}.soft_contact_score`` for one clip.

    Returns ``{"foot_L_soft_contact_score": (T,), "foot_R_soft_contact_score": (T,)}``
    as float64 CPU numpy arrays. Fail-fast on missing file, missing
    ``Frames``, missing ``FootEvidence``, missing per-side
    ``soft_contact_score``, non-scalar value, or non-finite value. The
    contract (§5.2 contact group) names this field explicitly, so we
    refuse to silently default it.
    """
    path = raw_root / f"{clip}.json"
    if not path.is_file():
        raise FailFastError(
            f"[walk_f_causal_state_probe] raw clip missing for contact load: "
            f"{path}."
        )
    raw = json.loads(path.read_text())
    if "Frames" not in raw or not isinstance(raw["Frames"], list) or len(raw["Frames"]) == 0:
        raise FailFastError(
            f"[walk_f_causal_state_probe] clip {clip!r} missing non-empty Frames "
            f"for contact load in {path}."
        )
    left: list[float] = []
    right: list[float] = []
    for i, frame in enumerate(raw["Frames"]):
        if "FootEvidence" not in frame or not isinstance(frame["FootEvidence"], dict):
            raise FailFastError(
                f"[walk_f_causal_state_probe] clip {clip!r} frame {i} missing "
                f"FootEvidence dict (contract §5.2) in {path}."
            )
        ev = frame["FootEvidence"]
        for side, bucket in (("L", left), ("R", right)):
            if side not in ev or not isinstance(ev[side], dict):
                raise FailFastError(
                    f"[walk_f_causal_state_probe] clip {clip!r} frame {i} "
                    f"missing FootEvidence.{side} dict in {path}."
                )
            side_ev = ev[side]
            if "soft_contact_score" not in side_ev:
                raise FailFastError(
                    f"[walk_f_causal_state_probe] clip {clip!r} frame {i} "
                    f"missing FootEvidence.{side}.soft_contact_score in {path}."
                )
            v = side_ev["soft_contact_score"]
            if isinstance(v, (list, tuple)) or not isinstance(v, (int, float)):
                raise FailFastError(
                    f"[walk_f_causal_state_probe] clip {clip!r} frame {i} "
                    f"FootEvidence.{side}.soft_contact_score is not scalar "
                    f"(got {type(v).__name__}) in {path}."
                )
            bucket.append(float(v))

    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if not (np.all(np.isfinite(left_arr)) and np.all(np.isfinite(right_arr))):
        raise FailFastError(
            f"[walk_f_causal_state_probe] clip {clip!r} contains non-finite "
            f"soft_contact_score in {path}."
        )
    return {
        "foot_L_soft_contact_score": left_arr,
        "foot_R_soft_contact_score": right_arr,
    }


# -----------------------------------------------------------------------
# yaw_debug (Layer 0 / Track 1)
# -----------------------------------------------------------------------


def _compute_yaw_signals(yaw_rad: np.ndarray, fps: float) -> dict[str, Any]:
    yaw_unwrapped = np.unwrap(yaw_rad)
    cumulative_delta_rad = yaw_unwrapped - yaw_unwrapped[0]
    cumulative_delta_deg = np.rad2deg(cumulative_delta_rad)

    yaw_rate_rad = np.zeros_like(yaw_unwrapped)
    if yaw_unwrapped.shape[0] > 1:
        yaw_rate_rad[1:] = np.diff(yaw_unwrapped) * fps
    yaw_rate_deg = np.rad2deg(yaw_rate_rad)
    abs_yaw_rate_deg = np.abs(yaw_rate_deg)

    total_delta_deg = float(cumulative_delta_deg[-1])
    peak_idx = int(np.argmax(abs_yaw_rate_deg))
    peak_abs_rate = float(abs_yaw_rate_deg[peak_idx])

    if total_delta_deg < -STRAIGHT_TOTAL_DELTA_DEG_TOL:
        direction = "left"
    elif total_delta_deg > STRAIGHT_TOTAL_DELTA_DEG_TOL:
        direction = "right"
    else:
        direction = "straight"

    return {
        "yaw_unwrapped_rad": yaw_unwrapped,
        "cumulative_delta_yaw_rad": cumulative_delta_rad,
        "cumulative_delta_yaw_deg": cumulative_delta_deg,
        "yaw_rate_rad_per_s": yaw_rate_rad,
        "yaw_rate_deg_per_s": yaw_rate_deg,
        "abs_yaw_rate_deg_per_s": abs_yaw_rate_deg,
        "total_yaw_delta_deg": total_delta_deg,
        "peak_abs_yaw_rate_deg_per_s": peak_abs_rate,
        "peak_abs_yaw_rate_frame": peak_idx,
        "turn_direction_by_yaw": direction,
        "turn_direction_threshold_deg_estimator_only": STRAIGHT_TOTAL_DELTA_DEG_TOL,
    }


def _activity_window(
    abs_rate_deg: np.ndarray,
    threshold_deg_per_s: float,
) -> dict[str, Any]:
    n = int(abs_rate_deg.shape[0])
    active_mask = abs_rate_deg >= float(threshold_deg_per_s)
    active_frames = [int(i) for i, m in enumerate(active_mask) if bool(m)]

    if not active_frames:
        return {
            "threshold_deg_per_s": float(threshold_deg_per_s),
            "active_frames": [],
            "active_frame_count": 0,
            "activity_start_frame": None,
            "activity_end_frame": None,
            "left_censored_start": False,
            "right_censored_end": False,
            "post_activity_buffer_frames": None,
        }

    start = active_frames[0]
    end = active_frames[-1]
    left_censored = start == 0
    right_censored = end == (n - 1)
    post_buffer = None if right_censored else int(n - 1 - end)

    return {
        "threshold_deg_per_s": float(threshold_deg_per_s),
        "active_frames": active_frames,
        "active_frame_count": int(len(active_frames)),
        "activity_start_frame": int(start),
        "activity_end_frame": int(end),
        "left_censored_start": bool(left_censored),
        "right_censored_end": bool(right_censored),
        "post_activity_buffer_frames": post_buffer,
    }


def _build_threshold_grid(peak_abs_rate: float) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for v in ABS_DEG_PER_S_GRID:
        grid.append(
            {
                "kind": "absolute_deg_per_s",
                "value": float(v),
                "threshold_deg_per_s": float(v),
                "degenerate": False,
            }
        )
    peak_is_degenerate = float(peak_abs_rate) < DEGENERATE_PEAK_DEG_PER_S
    for r in REL_TO_PEAK_GRID:
        grid.append(
            {
                "kind": "relative_to_peak_abs_yaw_rate",
                "value": float(r),
                "threshold_deg_per_s": float(r * peak_abs_rate),
                "degenerate": bool(peak_is_degenerate),
            }
        )
    return grid


def _summarise_stability(
    windows: list[dict[str, Any]],
    direction: str,
) -> dict[str, Any]:
    usable = [w for w in windows if not w.get("degenerate_skipped", False)]
    degenerate_settings = [
        {
            "kind": w["threshold_setting"]["kind"],
            "value": w["threshold_setting"]["value"],
            "threshold_deg_per_s": w["threshold_setting"]["threshold_deg_per_s"],
            "degenerate_reason": w.get("degenerate_reason"),
        }
        for w in windows
        if w.get("degenerate_skipped", False)
    ]

    starts = [w["activity_start_frame"] for w in usable if w["activity_start_frame"] is not None]
    ends = [w["activity_end_frame"] for w in usable if w["activity_end_frame"] is not None]
    nonempty_settings = [
        {
            "kind": w["threshold_setting"]["kind"],
            "value": w["threshold_setting"]["value"],
            "threshold_deg_per_s": w["threshold_setting"]["threshold_deg_per_s"],
        }
        for w in usable
        if w["activity_start_frame"] is not None
    ]
    empty_settings = [
        {
            "kind": w["threshold_setting"]["kind"],
            "value": w["threshold_setting"]["value"],
            "threshold_deg_per_s": w["threshold_setting"]["threshold_deg_per_s"],
        }
        for w in usable
        if w["activity_start_frame"] is None
    ]

    if not starts and not ends:
        if direction == "straight":
            status = "straight_no_yaw_activity_debug_baseline"
        else:
            status = "INSUFFICIENT_EVIDENCE"
        return {
            "status": status,
            "start_frame_min": None,
            "start_frame_max": None,
            "end_frame_min": None,
            "end_frame_max": None,
            "settings_with_activity": [],
            "settings_without_activity": empty_settings,
            "settings_skipped_degenerate": degenerate_settings,
            "thresholds_with_activity_count": 0,
            "thresholds_total": len(windows),
            "thresholds_usable_count": len(usable),
        }

    start_min = int(min(starts))
    start_max = int(max(starts))
    end_min = int(min(ends))
    end_max = int(max(ends))

    if empty_settings:
        status = "INSUFFICIENT_EVIDENCE"
    elif start_min == start_max and end_min == end_max:
        status = "stable"
    else:
        status = "unstable"

    return {
        "status": status,
        "start_frame_min": start_min,
        "start_frame_max": start_max,
        "end_frame_min": end_min,
        "end_frame_max": end_max,
        "settings_with_activity": nonempty_settings,
        "settings_without_activity": empty_settings,
        "settings_skipped_degenerate": degenerate_settings,
        "thresholds_with_activity_count": int(len(nonempty_settings)),
        "thresholds_total": int(len(windows)),
        "thresholds_usable_count": int(len(usable)),
    }


def _process_clip_yaw_debug(clip_info: dict[str, Any]) -> dict[str, Any]:
    yaw_rad: np.ndarray = clip_info["root_yaw_rad"]
    fps: float = clip_info["fps"]
    signals = _compute_yaw_signals(yaw_rad, fps)

    grid = _build_threshold_grid(signals["peak_abs_yaw_rate_deg_per_s"])
    windows: list[dict[str, Any]] = []
    for setting in grid:
        if setting["degenerate"]:
            w = {
                "threshold_deg_per_s": float(setting["threshold_deg_per_s"]),
                "active_frames": [],
                "active_frame_count": 0,
                "activity_start_frame": None,
                "activity_end_frame": None,
                "left_censored_start": False,
                "right_censored_end": False,
                "post_activity_buffer_frames": None,
                "degenerate_skipped": True,
                "degenerate_reason": (
                    "peak_abs_yaw_rate_deg_per_s below "
                    f"{DEGENERATE_PEAK_DEG_PER_S} so relative-to-peak threshold "
                    "would collapse to 0 deg/s and mark every frame active; "
                    "this setting is dropped to avoid a spurious activity claim."
                ),
            }
        else:
            w = _activity_window(signals["abs_yaw_rate_deg_per_s"], setting["threshold_deg_per_s"])
            w["degenerate_skipped"] = False
            w["degenerate_reason"] = None
        w["threshold_setting"] = setting
        windows.append(w)

    stability = _summarise_stability(windows, signals["turn_direction_by_yaw"])

    return {
        "clip": clip_info["clip"],
        "raw_json_path": clip_info["raw_json_path"],
        "fps": fps,
        "frame_count": clip_info["frame_count"],
        "root_yaw_units_raw": clip_info["root_yaw_units_raw"],
        "root_yaw_units_emitted": ["radians", "degrees"],
        "per_frame": {
            "yaw_rad": _serializable(yaw_rad),
            "yaw_unwrapped_rad": _serializable(signals["yaw_unwrapped_rad"]),
            "cumulative_delta_yaw_rad": _serializable(signals["cumulative_delta_yaw_rad"]),
            "cumulative_delta_yaw_deg": _serializable(signals["cumulative_delta_yaw_deg"]),
            "yaw_rate_rad_per_s": _serializable(signals["yaw_rate_rad_per_s"]),
            "yaw_rate_deg_per_s": _serializable(signals["yaw_rate_deg_per_s"]),
            "abs_yaw_rate_deg_per_s": _serializable(signals["abs_yaw_rate_deg_per_s"]),
        },
        "summary": {
            "total_yaw_delta_deg": float(signals["total_yaw_delta_deg"]),
            "peak_abs_yaw_rate_deg_per_s": float(signals["peak_abs_yaw_rate_deg_per_s"]),
            "peak_abs_yaw_rate_frame": int(signals["peak_abs_yaw_rate_frame"]),
            "turn_direction_by_yaw": signals["turn_direction_by_yaw"],
            "turn_direction_threshold_deg_estimator_only": float(
                signals["turn_direction_threshold_deg_estimator_only"]
            ),
        },
        "yaw_activity_windows": windows,
        "window_stability_summary": stability,
        "track": TRACK_BY_MODE["yaw_debug"],
        "track_role": TRACK_ROLE_BY_MODE["yaw_debug"],
        "causal_state_membership_status": "not_implemented_layer0",
    }


# -----------------------------------------------------------------------
# gauge_check (Layer A) — canonical yaw quotient + invariance sanity
# -----------------------------------------------------------------------


def _rotation_matrix_2d(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def _wrap_to_pi(arr: np.ndarray) -> np.ndarray:
    """Wrap an angle array into ``[-pi, pi)``.

    Both ``+pi`` and ``-pi`` map to ``-pi`` (the lower-closed boundary).
    The contract-required wrap-boundary alphas (``+/-pi``) are handled
    explicitly in ``_apply_synthetic_global_rotation`` regardless.
    """
    return (arr + math.pi) % (2.0 * math.pi) - math.pi


def _canonical_quotient_features(
    yaw_rad: np.ndarray,
    root_pos_xy: np.ndarray,
    root_vel_xy: np.ndarray,
    fps: float,
) -> dict[str, np.ndarray]:
    """Apply the canonical SE(2)-style yaw quotient.

    Gauge:
      * planar origin = ``root_pos_xy[0]``;
      * yaw zero      = ``unwrap(RootYaw)[0]``;
      * all root displacement / velocity are rotated by ``R(-yaw0)``
        into a body-aligned frame anchored at the clip's first frame.

    State features (absolute world coordinate eliminated):
      * ``body_root_pos_xy``  shape (T, 2)
      * ``body_root_vel_xy``  shape (T, 2)
      * ``yaw_rel_rad``       shape (T,)   = unwrap(yaw) - yaw[0]
      * ``yaw_rate_rad_per_s``shape (T,)   dynamics-only feature

    These features are by construction invariant to a global planar
    translation and to a global yaw rotation; the ``gauge_check`` mode
    verifies that property numerically.
    """
    yaw_unwrapped = np.unwrap(yaw_rad)
    theta0 = float(yaw_unwrapped[0])
    rot_minus_theta0 = _rotation_matrix_2d(-theta0)

    delta_p = root_pos_xy - root_pos_xy[0:1]
    # row-vector convention: (T, 2) @ R.T applies R to each row.
    body_p = delta_p @ rot_minus_theta0.T
    body_v = root_vel_xy @ rot_minus_theta0.T

    yaw_rel = yaw_unwrapped - theta0
    yaw_rate = np.zeros_like(yaw_unwrapped)
    if yaw_unwrapped.shape[0] > 1:
        yaw_rate[1:] = np.diff(yaw_unwrapped) * fps

    return {
        "body_root_pos_xy": body_p.astype(np.float64),
        "body_root_vel_xy": body_v.astype(np.float64),
        "yaw_rel_rad": yaw_rel.astype(np.float64),
        "yaw_rate_rad_per_s": yaw_rate.astype(np.float64),
    }


def _apply_synthetic_global_rotation(
    yaw_rad: np.ndarray,
    root_pos_xy: np.ndarray,
    root_vel_xy: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Rotate the entire clip by alpha around the world origin.

    * yaw       -> wrap_to_pi(yaw + alpha)
    * root_pos  -> R(alpha) @ root_pos
    * root_vel  -> R(alpha) @ root_vel

    Returns the rotated arrays plus a small diagnostic dict that
    records whether ``+/-pi`` wrap is exercised by this rotation on
    this clip.
    """
    rot_alpha = _rotation_matrix_2d(alpha)
    yaw_pre_wrap = yaw_rad + alpha
    yaw_rotated = _wrap_to_pi(yaw_pre_wrap)
    pos_rotated = root_pos_xy @ rot_alpha.T
    vel_rotated = root_vel_xy @ rot_alpha.T

    pre_min = float(yaw_pre_wrap.min())
    pre_max = float(yaw_pre_wrap.max())
    crosses_pi = bool(pre_max > math.pi or pre_min <= -math.pi)
    wrap_jumps = 0
    if yaw_rotated.shape[0] > 1:
        diffs = np.abs(np.diff(yaw_rotated))
        wrap_jumps = int(np.sum(diffs > WRAP_JUMP_DETECT_RAD))
    # +/-pi is the contract-required wrap boundary; flag the rotation
    # itself, independent of the per-clip yaw values, so Walk_F (yaw=0
    # everywhere) is still recorded as exercising the wrap boundary at
    # alpha=+/-pi.
    alpha_at_wrap_boundary = bool(
        math.isclose(alpha, math.pi, abs_tol=1e-12)
        or math.isclose(alpha, -math.pi, abs_tol=1e-12)
    )
    diag = {
        "alpha_rad": float(alpha),
        "pre_wrap_yaw_min_rad": pre_min,
        "pre_wrap_yaw_max_rad": pre_max,
        "crosses_plus_or_minus_pi_in_pre_wrap": crosses_pi,
        "wrap_jumps_in_post_wrap_yaw": wrap_jumps,
        "alpha_is_required_wrap_boundary": alpha_at_wrap_boundary,
    }
    return yaw_rotated, pos_rotated, vel_rotated, diag


def _apply_synthetic_translation(
    root_pos_xy: np.ndarray,
    beta_xy: tuple[float, float],
) -> np.ndarray:
    """Translate ``root_pos_xy`` by ``beta_xy`` (meters).

    RootYaw and RootVelocityXY are intentionally NOT translated:
    planar translation is a rigid shift of the world origin, which
    does not change angles or per-frame linear velocity vectors. The
    canonical gauge subtracts ``root_pos_xy[0]`` so by construction
    every quotient feature must be invariant under this perturbation.
    """
    beta = np.asarray(beta_xy, dtype=np.float64).reshape(1, 2)
    return root_pos_xy + beta


def _abs_error_stats(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        raise FailFastError(
            f"[walk_f_causal_state_probe] shape mismatch in gauge error: "
            f"{a.shape} vs {b.shape}"
        )
    diff = np.abs(a - b)
    flat = diff.reshape(-1)
    finite_a = int(np.sum(np.isfinite(a)))
    finite_b = int(np.sum(np.isfinite(b)))
    total = int(a.size)
    if flat.size == 0:
        return {
            "max_abs_error": 0.0,
            "median_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "n_values": 0,
            "n_finite_original": finite_a,
            "n_finite_rotated": finite_b,
            "all_finite": bool(finite_a == total and finite_b == total),
        }
    return {
        "max_abs_error": float(np.max(flat)),
        "median_abs_error": float(np.median(flat)),
        "mean_abs_error": float(np.mean(flat)),
        "n_values": total,
        "n_finite_original": finite_a,
        "n_finite_rotated": finite_b,
        "all_finite": bool(finite_a == total and finite_b == total),
    }


def _process_clip_gauge_check(clip_info: dict[str, Any]) -> dict[str, Any]:
    yaw_rad: np.ndarray = clip_info["root_yaw_rad"]
    pos_xy: np.ndarray = clip_info["root_pos_xy_m"]
    vel_xy: np.ndarray = clip_info["root_vel_xy_mps"]
    fps: float = clip_info["fps"]

    feat_orig = _canonical_quotient_features(yaw_rad, pos_xy, vel_xy, fps)

    # --- Yaw rotation grid (incl. +/-pi wrap boundary) ---------------
    per_rotation: list[dict[str, Any]] = []
    wrap_cases: list[dict[str, Any]] = []
    rot_max_err_by_feat: dict[str, float] = {k: 0.0 for k in feat_orig}
    rot_finite_total = 0
    rot_finite_finite = 0

    for alpha in ROTATION_GRID_RAD:
        yaw_rot, pos_rot, vel_rot, wrap_diag = _apply_synthetic_global_rotation(
            yaw_rad, pos_xy, vel_xy, float(alpha)
        )
        feat_rot = _canonical_quotient_features(yaw_rot, pos_rot, vel_rot, fps)
        per_group: dict[str, Any] = {}
        rot_max_err = 0.0
        for k in feat_orig:
            stats = _abs_error_stats(feat_orig[k], feat_rot[k])
            per_group[k] = stats
            rot_max_err = max(rot_max_err, stats["max_abs_error"])
            rot_max_err_by_feat[k] = max(rot_max_err_by_feat[k], stats["max_abs_error"])
            rot_finite_total += stats["n_values"]
            rot_finite_finite += min(stats["n_finite_original"], stats["n_finite_rotated"])
        per_rotation.append(
            {
                "alpha_rad": float(alpha),
                "alpha_deg": float(math.degrees(alpha)),
                "wrap_diagnostic": wrap_diag,
                "by_feature_group": per_group,
                "max_abs_error_over_groups": float(rot_max_err),
            }
        )
        if (
            wrap_diag["crosses_plus_or_minus_pi_in_pre_wrap"]
            or wrap_diag["wrap_jumps_in_post_wrap_yaw"] > 0
            or wrap_diag["alpha_is_required_wrap_boundary"]
        ):
            wrap_cases.append(
                {
                    "alpha_rad": float(alpha),
                    "pre_wrap_yaw_min_rad": wrap_diag["pre_wrap_yaw_min_rad"],
                    "pre_wrap_yaw_max_rad": wrap_diag["pre_wrap_yaw_max_rad"],
                    "wrap_jumps_in_post_wrap_yaw": wrap_diag["wrap_jumps_in_post_wrap_yaw"],
                    "alpha_is_required_wrap_boundary": wrap_diag["alpha_is_required_wrap_boundary"],
                    "crosses_plus_or_minus_pi_in_pre_wrap": wrap_diag["crosses_plus_or_minus_pi_in_pre_wrap"],
                    "max_abs_error_over_groups": float(rot_max_err),
                }
            )

    # --- Planar translation grid -------------------------------------
    per_translation: list[dict[str, Any]] = []
    trans_max_err_by_feat: dict[str, float] = {k: 0.0 for k in feat_orig}
    trans_finite_total = 0
    trans_finite_finite = 0

    for beta in TRANSLATION_GRID_M:
        pos_trans = _apply_synthetic_translation(pos_xy, beta)
        # RootYaw and RootVelocityXY are intentionally NOT translated.
        feat_trans = _canonical_quotient_features(yaw_rad, pos_trans, vel_xy, fps)
        per_group_t: dict[str, Any] = {}
        trans_max_err = 0.0
        for k in feat_orig:
            stats = _abs_error_stats(feat_orig[k], feat_trans[k])
            per_group_t[k] = stats
            trans_max_err = max(trans_max_err, stats["max_abs_error"])
            trans_max_err_by_feat[k] = max(trans_max_err_by_feat[k], stats["max_abs_error"])
            trans_finite_total += stats["n_values"]
            trans_finite_finite += min(stats["n_finite_original"], stats["n_finite_rotated"])
        per_translation.append(
            {
                "beta_xy_m": [float(beta[0]), float(beta[1])],
                "by_feature_group": per_group_t,
                "max_abs_error_over_groups": float(trans_max_err),
            }
        )

    # --- Aggregates --------------------------------------------------
    rot_coverage_fraction = (
        float(rot_finite_finite) / float(rot_finite_total) if rot_finite_total > 0 else 0.0
    )
    trans_coverage_fraction = (
        float(trans_finite_finite) / float(trans_finite_total)
        if trans_finite_total > 0
        else 0.0
    )
    clip_max_err_rotation = max(rot_max_err_by_feat.values()) if rot_max_err_by_feat else 0.0
    clip_max_err_translation = (
        max(trans_max_err_by_feat.values()) if trans_max_err_by_feat else 0.0
    )
    clip_max_err_se2 = max(clip_max_err_rotation, clip_max_err_translation)
    se2_max_err_by_feat = {
        k: float(max(rot_max_err_by_feat[k], trans_max_err_by_feat[k])) for k in feat_orig
    }

    return {
        "clip": clip_info["clip"],
        "raw_json_path": clip_info["raw_json_path"],
        "fps": fps,
        "frame_count": clip_info["frame_count"],
        "root_units_raw": clip_info["root_units_raw"],
        "root_yaw_units_raw": clip_info["root_yaw_units_raw"],
        "track": TRACK_BY_MODE["gauge_check"],
        "track_role": TRACK_ROLE_BY_MODE["gauge_check"],
        # Yaw rotation grid (unchanged Layer A)
        "per_rotation": per_rotation,
        "wrap_boundary_cases": wrap_cases,
        "max_rotation_abs_error_by_feature_group": {
            k: float(v) for k, v in rot_max_err_by_feat.items()
        },
        "max_abs_error_over_rotation_grid": float(clip_max_err_rotation),
        "rotation_finite_coverage": {
            "n_values_compared": int(rot_finite_total),
            "n_finite_min_per_pair": int(rot_finite_finite),
            "coverage_fraction": float(rot_coverage_fraction),
        },
        # Planar translation grid (Polish 2)
        "per_translation": per_translation,
        "max_translation_abs_error_by_feature_group": {
            k: float(v) for k, v in trans_max_err_by_feat.items()
        },
        "max_abs_error_over_translation_grid": float(clip_max_err_translation),
        "translation_finite_coverage": {
            "n_values_compared": int(trans_finite_total),
            "n_finite_min_per_pair": int(trans_finite_finite),
            "coverage_fraction": float(trans_coverage_fraction),
        },
        # Aggregate SE(2) quotient sanity
        "max_abs_error_by_feature_group_se2": se2_max_err_by_feat,
        "max_abs_error_over_se2_grid": float(clip_max_err_se2),
        # Backwards-compatible aliases (yaw-only view kept for callers
        # that read the Layer A pre-polish field names).
        "max_abs_error_by_feature_group": {k: float(v) for k, v in rot_max_err_by_feat.items()},
        "max_abs_error_over_grid": float(clip_max_err_rotation),
        "finite_coverage": {
            "n_values_compared": int(rot_finite_total),
            "n_finite_min_per_pair": int(rot_finite_finite),
            "coverage_fraction": float(rot_coverage_fraction),
        },
        "causal_state_membership_status": "not_implemented_layerA",
    }


def _build_quotient_definition_block() -> dict[str, Any]:
    return {
        "contract_section": (
            "docs/aperiodic_transition/"
            "2026-05-22_walk_f_causal_state_scaffold_v1.md §1.2"
        ),
        "planar_translation_removed": True,
        "global_yaw_removed": True,
        "absolute_root_yaw_is_state_feature": False,
        "absolute_world_position_is_state_feature": False,
        "gauge": {
            "origin_xy": "root_pos_xy[0]",
            "yaw_zero": "unwrap(RootYaw)[0]",
            "rotation_applied_to_displacement_and_velocity": "R(-yaw_zero)",
            "translation_removed_by": "root_pos_xy - root_pos_xy[0]",
        },
        "feature_groups_emitted": [
            "body_root_pos_xy",
            "body_root_vel_xy",
            "yaw_rel_rad",
            "yaw_rate_rad_per_s",
        ],
        "dynamics_features_retained": [
            "yaw_rel_rad",
            "yaw_rate_rad_per_s",
        ],
        "shapes": {
            "yaw": "(T,)",
            "body_root_pos_xy": "(T, 2)",
            "body_root_vel_xy": "(T, 2)",
            "yaw_rel_rad": "(T,)",
            "yaw_rate_rad_per_s": "(T,)",
            "dtype": "float64",
            "device": "cpu_numpy",
        },
    }


def _build_yaw_invariance_sanity(
    per_clip: list[dict[str, Any]],
) -> dict[str, Any]:
    max_err = 0.0
    finite_min_fraction = 1.0
    for entry in per_clip:
        max_err = max(max_err, float(entry["max_abs_error_over_rotation_grid"]))
        finite_min_fraction = min(
            finite_min_fraction,
            float(entry["rotation_finite_coverage"]["coverage_fraction"]),
        )

    if finite_min_fraction < 1.0:
        status = "INSUFFICIENT_EVIDENCE"
        rationale = (
            "At least one clip / rotation produced non-finite values; "
            "yaw-invariance sanity cannot be evaluated numerically until "
            "all compared values are finite."
        )
    elif max_err <= GAUGE_MAX_ABS_ERR_TOL:
        status = "pass"
        rationale = (
            "All compared values are finite and the max absolute error "
            "across clips x rotations x feature groups is within the "
            "estimator-level numeric tolerance. The tolerance is reported "
            "numerically and is NOT a contract definition."
        )
    else:
        status = "fail"
        rationale = (
            "Max absolute error exceeded the estimator-level numeric "
            "tolerance. By §1.2 this indicates the canonical gauge does "
            "not actually quotient out global yaw under at least one "
            "rotation in the grid (including the +/-pi wrap boundary). "
            "This is a CONTRACT failure, not estimator noise."
        )

    return {
        "status": status,
        "tol_max_abs_error_estimator_only": GAUGE_MAX_ABS_ERR_TOL,
        "max_abs_error_overall": float(max_err),
        "min_finite_coverage_fraction": float(finite_min_fraction),
        "rotation_grid_rad": [float(a) for a in ROTATION_GRID_RAD],
        "rotation_grid_deg": [float(math.degrees(a)) for a in ROTATION_GRID_RAD],
        "wrap_boundary_required_alphas_rad": [math.pi, -math.pi],
        "rationale": rationale,
    }


def _build_translation_invariance_sanity(
    per_clip: list[dict[str, Any]],
) -> dict[str, Any]:
    max_err = 0.0
    finite_min_fraction = 1.0
    for entry in per_clip:
        max_err = max(max_err, float(entry["max_abs_error_over_translation_grid"]))
        finite_min_fraction = min(
            finite_min_fraction,
            float(entry["translation_finite_coverage"]["coverage_fraction"]),
        )

    if finite_min_fraction < 1.0:
        status = "INSUFFICIENT_EVIDENCE"
        rationale = (
            "At least one clip / translation produced non-finite values; "
            "translation-invariance sanity cannot be evaluated numerically "
            "until all compared values are finite."
        )
    elif max_err <= GAUGE_MAX_ABS_ERR_TOL:
        status = "pass"
        rationale = (
            "Synthetic planar translations on root_pos_xy (with RootYaw "
            "and RootVelocityXY unchanged) leave every quotient feature "
            "numerically unchanged within the estimator-level tolerance. "
            "The tolerance is reported numerically and is NOT a contract "
            "definition."
        )
    else:
        status = "fail"
        rationale = (
            "Max absolute error under planar translation exceeded the "
            "estimator-level tolerance. By §1.2 this indicates the gauge "
            "does not actually quotient out planar translation; this is "
            "a CONTRACT failure, not estimator noise."
        )

    return {
        "status": status,
        "tol_max_abs_error_estimator_only": GAUGE_MAX_ABS_ERR_TOL,
        "max_abs_error_overall": float(max_err),
        "min_finite_coverage_fraction": float(finite_min_fraction),
        "translation_grid_m": [list(b) for b in TRANSLATION_GRID_M],
        "perturbation_target": "root_pos_xy",
        "fields_intentionally_unperturbed": ["RootYaw", "RootVelocityXY"],
        "rationale": rationale,
    }


def _build_se2_gauge_sanity(
    yaw_sanity: dict[str, Any],
    translation_sanity: dict[str, Any],
    per_clip: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate yaw + translation invariance into the SE(2) quotient
    sanity gate. This is NOT an attractor gate; it only verifies that
    the canonical SE(2) gauge implementation behaves invariantly on
    the current 5 clips under the configured rotation / translation
    grids.
    """
    statuses = (yaw_sanity["status"], translation_sanity["status"])
    if "fail" in statuses:
        status = "fail"
    elif "INSUFFICIENT_EVIDENCE" in statuses:
        status = "INSUFFICIENT_EVIDENCE"
    elif statuses == ("pass", "pass"):
        status = "pass"
    else:
        status = "INSUFFICIENT_EVIDENCE"

    max_err_overall = max(
        float(yaw_sanity["max_abs_error_overall"]),
        float(translation_sanity["max_abs_error_overall"]),
    )
    min_coverage = min(
        float(yaw_sanity["min_finite_coverage_fraction"]),
        float(translation_sanity["min_finite_coverage_fraction"]),
    )
    se2_max_err_by_feat: dict[str, float] = {}
    for entry in per_clip:
        for k, v in entry["max_abs_error_by_feature_group_se2"].items():
            se2_max_err_by_feat[k] = max(se2_max_err_by_feat.get(k, 0.0), float(v))

    return {
        "status": status,
        "name": "SE(2)_quotient_sanity_gate",
        "interpretation": (
            "Numerical sanity that the canonical SE(2) gauge "
            "(planar-translation removal + global-yaw removal + wrap "
            "boundary) leaves all quotient features unchanged on the "
            "current 5 clips. This is a prerequisite for the future "
            "causal-state membership track, NOT attractor evidence."
        ),
        "components": {
            "yaw_invariance_sanity_status": yaw_sanity["status"],
            "translation_invariance_sanity_status": translation_sanity["status"],
        },
        "tol_max_abs_error_estimator_only": GAUGE_MAX_ABS_ERR_TOL,
        "max_abs_error_overall": float(max_err_overall),
        "min_finite_coverage_fraction": float(min_coverage),
        "max_abs_error_by_feature_group": se2_max_err_by_feat,
        "is_attractor_gate": False,
        "is_membership_claim": False,
    }


# -----------------------------------------------------------------------
# reference_scale_check (Layer B) — reference scale + degeneracy
# -----------------------------------------------------------------------


def _compute_channel_scale(arr: np.ndarray) -> dict[str, Any]:
    """Per-channel robust + classical scale summary.

    Returns MAD (median absolute deviation), MAD-derived Gaussian-equivalent
    sigma, classical std, peak-to-peak, mean(|x|), min/max, and finite
    counts. Non-finite handling: every statistic is computed on the
    finite mask; ``all_finite`` is reported separately so Layer B can
    route the channel to ``INSUFFICIENT_EVIDENCE``.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n_total = int(arr.size)
    finite_mask = np.isfinite(arr)
    n_finite = int(finite_mask.sum())
    if n_finite == 0:
        return {
            "n_values": n_total,
            "n_finite": 0,
            "all_finite": False,
            "median": None,
            "mad": None,
            "robust_std_from_mad": None,
            "std": None,
            "ptp": None,
            "mean_abs": None,
            "min": None,
            "max": None,
        }
    finite = arr[finite_mask]
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_std = float(mad * MAD_TO_GAUSSIAN_SIGMA)
    classical_std = float(np.std(finite))
    ptp = float(np.ptp(finite))
    mean_abs = float(np.mean(np.abs(finite)))
    return {
        "n_values": n_total,
        "n_finite": n_finite,
        "all_finite": bool(n_finite == n_total),
        "median": median,
        "mad": mad,
        "robust_std_from_mad": robust_std,
        "std": classical_std,
        "ptp": ptp,
        "mean_abs": mean_abs,
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _extract_layer_b_channels(
    quotient_features: dict[str, np.ndarray],
    contact_signals: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map quotient feature arrays + raw contact arrays into the flat
    channel dict that ``_compute_group_reference_scale`` consumes.
    """
    body_p = quotient_features["body_root_pos_xy"]
    body_v = quotient_features["body_root_vel_xy"]
    yaw_rel = quotient_features["yaw_rel_rad"]
    yaw_rate = quotient_features["yaw_rate_rad_per_s"]
    return {
        "body_root_pos_xy.x": body_p[:, 0],
        "body_root_pos_xy.y": body_p[:, 1],
        "body_root_vel_xy.x": body_v[:, 0],
        "body_root_vel_xy.y": body_v[:, 1],
        "yaw_rel_rad": yaw_rel,
        "yaw_rate_rad_per_s": yaw_rate,
        "foot_L_soft_contact_score": contact_signals["foot_L_soft_contact_score"],
        "foot_R_soft_contact_score": contact_signals["foot_R_soft_contact_score"],
    }


def _classify_phase_structure_status(
    *,
    n_frames: int,
    group_all_finite: bool,
    group_degenerate: bool,
) -> tuple[str, bool]:
    """Return the contract-§6 phase_structure_status enum plus a Layer C
    candidate flag.

    The contract enum (docs/aperiodic_transition/
    2026-05-22_walk_f_causal_state_scaffold_v1.md:255) is exactly
    ``{phase_structured, phase_degenerate, insufficient_evidence}``.
    Layer B cannot return ``phase_structured`` because that requires a
    predictive-loss comparison against a phase-agnostic baseline
    (§3.4), which is Layer C's job. Layer B therefore only emits
    ``phase_degenerate`` or ``insufficient_evidence``; non-degenerate
    groups get ``insufficient_evidence`` plus
    ``layer_c_candidate=True`` so a future Layer C knows where to look.
    """
    if n_frames < MIN_FRAMES_FOR_PHASE_ESTIMABILITY:
        return ("insufficient_evidence", False)
    if not group_all_finite:
        return ("insufficient_evidence", False)
    if group_degenerate:
        return ("phase_degenerate", False)
    # Non-degenerate at Layer B does NOT prove phase-structured. We mark
    # it as insufficient_evidence and flag the group as a Layer C
    # candidate so reviewers see the distinction.
    return ("insufficient_evidence", True)


def _compute_group_reference_scale(
    group_name: str,
    group_def: dict[str, Any],
    channel_arrays: dict[str, np.ndarray],
    n_frames: int,
) -> dict[str, Any]:
    """Aggregate per-channel scale into a group-level reference scale.

    Group-level degeneracy: a channel is degenerate iff its MAD
    ``<= epsilon`` (so the boundary value counts as degenerate); the
    group is degenerate iff every channel is degenerate. The
    phase_structure_status enum is the contract §6 set and does NOT
    involve any predictive distance.
    """
    epsilon = float(group_def["epsilon_mad"])
    channel_summaries: dict[str, Any] = {}
    max_mad = 0.0
    any_not_all_finite = False
    all_channel_degenerate = True
    channel_degenerate_count = 0
    for channel_name in group_def["channels"]:
        if channel_name not in channel_arrays:
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer B channel {channel_name!r} "
                f"missing for group {group_name!r}; refusing silent skip."
            )
        stats = _compute_channel_scale(channel_arrays[channel_name])
        channel_mad = stats["mad"]
        # `<=` not `<`: the boundary value counts as degenerate so the
        # tolerance is upper-inclusive, matching tolerance conventions
        # used elsewhere in this probe (e.g. gauge tol).
        channel_degenerate = bool(channel_mad is not None and channel_mad <= epsilon)
        if channel_degenerate:
            channel_degenerate_count += 1
        if channel_mad is None:
            all_channel_degenerate = False
        elif not channel_degenerate:
            all_channel_degenerate = False
        if channel_mad is not None and channel_mad > max_mad:
            max_mad = float(channel_mad)
        if not stats["all_finite"]:
            any_not_all_finite = True
        channel_summaries[channel_name] = {
            **stats,
            "epsilon_mad_estimator_only": epsilon,
            "channel_degenerate_by_epsilon_mad": channel_degenerate,
        }

    group_degenerate = bool(all_channel_degenerate)
    phase_status, layer_c_candidate = _classify_phase_structure_status(
        n_frames=n_frames,
        group_all_finite=not any_not_all_finite,
        group_degenerate=group_degenerate,
    )
    return {
        "group": group_name,
        "unit_label": group_def["unit_label"],
        "source": group_def["source"],
        "epsilon_mad_estimator_only": epsilon,
        "channels": channel_summaries,
        "group_max_channel_mad": float(max_mad),
        "group_reference_degenerate": group_degenerate,
        "channel_degenerate_count": int(channel_degenerate_count),
        "channel_total_count": int(len(group_def["channels"])),
        "phase_structure_status": phase_status,
        "phase_structure_status_enum_source": (
            "docs/aperiodic_transition/"
            "2026-05-22_walk_f_causal_state_scaffold_v1.md:255"
        ),
        "layer_c_candidate": bool(layer_c_candidate),
        "phase_estimable_candidate": bool(layer_c_candidate),
        "phase_structure_layer_b_definition": (
            "Layer B emits only {phase_degenerate, insufficient_evidence}. "
            "phase_structured requires a predictive-loss comparison vs a "
            "phase-agnostic baseline (§3.4) and is Layer C's job. "
            "Non-degenerate groups get insufficient_evidence with "
            "layer_c_candidate=True."
        ),
    }


def _process_clip_reference_scale_check(
    clip_info: dict[str, Any],
    contact_signals: dict[str, np.ndarray],
    *,
    is_reference: bool,
) -> dict[str, Any]:
    """Compute Layer B per-clip channel summaries.

    For reference-family clips the group reference scale is the
    authoritative output. For non-reference (query) clips we still
    emit per-channel scale numbers so a reviewer can sanity-check the
    extraction, but the artifact must NOT promote them into a
    Walk_F reference scale.
    """
    yaw_rad: np.ndarray = clip_info["root_yaw_rad"]
    pos_xy: np.ndarray = clip_info["root_pos_xy_m"]
    vel_xy: np.ndarray = clip_info["root_vel_xy_mps"]
    fps: float = clip_info["fps"]
    n_frames: int = int(clip_info["frame_count"])

    quotient = _canonical_quotient_features(yaw_rad, pos_xy, vel_xy, fps)
    channels = _extract_layer_b_channels(quotient, contact_signals)

    per_group: dict[str, Any] = {}
    for group_name, group_def in FEATURE_GROUPS_LAYER_B.items():
        per_group[group_name] = _compute_group_reference_scale(
            group_name, group_def, channels, n_frames
        )

    return {
        "clip": clip_info["clip"],
        "raw_json_path": clip_info["raw_json_path"],
        "fps": fps,
        "frame_count": n_frames,
        "track": TRACK_BY_MODE["reference_scale_check"],
        "track_role": TRACK_ROLE_BY_MODE["reference_scale_check"],
        "clip_role": "reference_clip" if is_reference else "query_clip_not_reference",
        "contributes_to_reference_scale": bool(is_reference),
        "feature_groups_layer_b": per_group,
        "not_run_feature_groups_layer_b": list(NOT_RUN_FEATURE_GROUPS_LAYER_B),
        "membership_status": "not_implemented_layerB",
        "predictive_loss_status": "not_implemented_layerB",
    }


# -----------------------------------------------------------------------
# phase_library_check (Layer C minimal) — Walk_F self-consistency only
# -----------------------------------------------------------------------


def _quantile_summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "n": int(arr.size),
            "n_finite": 0,
            "all_finite": False,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": int(arr.size),
        "n_finite": int(finite.size),
        "all_finite": bool(finite.size == arr.size),
        "min": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p25": float(np.quantile(finite, 0.25)),
        "p50": float(np.quantile(finite, 0.50)),
        "p75": float(np.quantile(finite, 0.75)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _metric_loss_1d(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if a.shape != b.shape:
        raise FailFastError(
            f"[walk_f_causal_state_probe] metric shape mismatch: {a.shape} vs {b.shape}"
        )
    diff = np.asarray(a - b, dtype=np.float64)
    if metric == "z_mse":
        return float(np.mean(diff * diff))
    if metric == "z_l1":
        return float(np.mean(np.abs(diff)))
    raise FailFastError(
        f"[walk_f_causal_state_probe] unknown Layer C distance_metric={metric!r}; "
        f"supported={list(LAYER_C_DISTANCE_METRICS)}"
    )


def _metric_loss_rows(query: np.ndarray, candidates: np.ndarray, metric: str) -> np.ndarray:
    if candidates.ndim != 2 or query.ndim != 1 or candidates.shape[1] != query.shape[0]:
        raise FailFastError(
            "[walk_f_causal_state_probe] row metric shape mismatch: "
            f"query={query.shape} candidates={candidates.shape}"
        )
    diff = candidates - query.reshape(1, -1)
    if metric == "z_mse":
        return np.mean(diff * diff, axis=1)
    if metric == "z_l1":
        return np.mean(np.abs(diff), axis=1)
    raise FailFastError(
        f"[walk_f_causal_state_probe] unknown Layer C distance_metric={metric!r}; "
        f"supported={list(LAYER_C_DISTANCE_METRICS)}"
    )


def _build_layer_c_group_matrix(
    *,
    group_name: str,
    group_def: dict[str, Any],
    channel_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Return robust-normalized Walk_F channel matrix for one Layer C group.

    Output ``z_matrix`` is float64 CPU numpy with shape ``(T, C_active)``.
    Channels with non-finite values or MAD at/below the estimator epsilon are
    excluded from the active matrix and reported explicitly.
    """
    epsilon = float(group_def["epsilon_mad"])
    active_names: list[str] = []
    excluded: list[dict[str, Any]] = []
    stats_by_channel: dict[str, Any] = {}
    z_cols: list[np.ndarray] = []
    n_frames: int | None = None

    for channel_name in group_def["channels"]:
        if channel_name not in channel_arrays:
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer C channel {channel_name!r} "
                f"missing for group {group_name!r}; refusing silent skip."
            )
        arr = np.asarray(channel_arrays[channel_name], dtype=np.float64)
        if arr.ndim != 1:
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer C channel {channel_name!r} "
                f"shape must be (T,), got {arr.shape}."
            )
        if n_frames is None:
            n_frames = int(arr.shape[0])
        elif int(arr.shape[0]) != n_frames:
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer C channel length mismatch in "
                f"group {group_name!r}: expected {n_frames}, got {arr.shape[0]} "
                f"for {channel_name!r}."
            )

        stats = _compute_channel_scale(arr)
        mad = stats["mad"]
        robust_std = stats["robust_std_from_mad"]
        stats_by_channel[channel_name] = {
            **stats,
            "epsilon_mad_estimator_only": epsilon,
        }

        if not stats["all_finite"]:
            excluded.append(
                {
                    "channel": channel_name,
                    "reason": "non_finite_values",
                    "mad": mad,
                    "robust_std_from_mad": robust_std,
                }
            )
            continue
        if mad is None or robust_std is None or mad <= epsilon or robust_std <= 0.0:
            excluded.append(
                {
                    "channel": channel_name,
                    "reason": "reference_degenerate_by_mad",
                    "mad": mad,
                    "robust_std_from_mad": robust_std,
                }
            )
            continue

        median = float(stats["median"])
        scale = float(robust_std)
        z_cols.append(((arr - median) / scale).astype(np.float64))
        active_names.append(channel_name)

    if n_frames is None:
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer C group {group_name!r} has no channels."
        )

    if z_cols:
        z_matrix = np.stack(z_cols, axis=1).astype(np.float64)
    else:
        z_matrix = np.zeros((n_frames, 0), dtype=np.float64)

    return {
        "group": group_name,
        "z_matrix": z_matrix,
        "active_channels": active_names,
        "excluded_channels": excluded,
        "channel_scale": stats_by_channel,
        "reference_degenerate": bool(len(active_names) == 0),
        "source": group_def["source"],
        "group_ablation_role": group_def["group_ablation_role"],
    }


def _build_phase_windows(
    z_matrix: np.ndarray,
    history_window_frames: int,
    future_horizon_frames: int,
) -> dict[str, Any]:
    if z_matrix.ndim != 2:
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer C z_matrix must be (T, C), got {z_matrix.shape}."
        )
    t_frames = list(
        range(
            int(history_window_frames) - 1,
            int(z_matrix.shape[0]) - int(future_horizon_frames),
        )
    )
    if not t_frames:
        return {
            "phase_frames": np.asarray([], dtype=np.int64),
            "history_windows": np.zeros((0, 0), dtype=np.float64),
            "future_windows": np.zeros((0, 0), dtype=np.float64),
        }

    history_rows: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    for t in t_frames:
        hist = z_matrix[t - history_window_frames + 1 : t + 1]
        fut = z_matrix[t + 1 : t + future_horizon_frames + 1]
        history_rows.append(hist.reshape(-1))
        future_rows.append(fut.reshape(-1))

    return {
        "phase_frames": np.asarray(t_frames, dtype=np.int64),
        "history_windows": np.stack(history_rows, axis=0).astype(np.float64),
        "future_windows": np.stack(future_rows, axis=0).astype(np.float64),
    }


def _run_layer_c_config(
    *,
    z_matrix: np.ndarray,
    history_window_frames: int,
    future_horizon_frames: int,
    neighborhood_radius_frames: int,
    distance_metric: str,
) -> dict[str, Any]:
    windows = _build_phase_windows(
        z_matrix,
        history_window_frames=history_window_frames,
        future_horizon_frames=future_horizon_frames,
    )
    phase_frames: np.ndarray = windows["phase_frames"]
    history_windows: np.ndarray = windows["history_windows"]
    future_windows: np.ndarray = windows["future_windows"]
    candidate_count = int(phase_frames.shape[0])
    if candidate_count == 0:
        raise FailFastError(
            "[walk_f_causal_state_probe] Layer C produced no valid phase "
            f"candidates for H={history_window_frames}, F={future_horizon_frames}; "
            "the fixed grid is incompatible with the reference clip."
        )

    phase_losses: list[float] = []
    agnostic_losses: list[float] = []
    relative_improvements: list[float] = []
    loss_percentiles: list[float] = []
    confidence_gaps: list[float] = []
    phase_hat_frames: list[int] = []
    matched_offsets: list[int] = []
    loss_curve: list[dict[str, Any]] = []
    loss_percentile_curve: list[dict[str, Any]] = []
    phase_confidence_gap_curve: list[dict[str, Any]] = []
    skipped_queries: list[dict[str, Any]] = []

    for query_i, t in enumerate(phase_frames):
        nonlocal_mask = np.abs(phase_frames - int(t)) > int(neighborhood_radius_frames)
        candidate_indices = np.nonzero(nonlocal_mask)[0]
        if candidate_indices.size == 0:
            skipped_queries.append(
                {
                    "query_frame": int(t),
                    "reason": "no_nonlocal_candidates_after_neighborhood_exclusion",
                }
            )
            continue

        hist_q = history_windows[query_i]
        fut_q = future_windows[query_i]
        hist_candidates = history_windows[candidate_indices]
        fut_candidates = future_windows[candidate_indices]

        hist_distances = _metric_loss_rows(hist_q, hist_candidates, distance_metric)
        order = np.argsort(hist_distances)
        best_local = int(order[0])
        best_candidate_index = int(candidate_indices[best_local])
        best_phase_frame = int(phase_frames[best_candidate_index])
        best_history_distance = float(hist_distances[best_local])
        if order.shape[0] > 1:
            second_history_distance = float(hist_distances[int(order[1])])
            confidence_gap = float(second_history_distance - best_history_distance)
            confidence_gaps.append(confidence_gap)
        else:
            second_history_distance = None
            confidence_gap = None

        phase_loss = _metric_loss_1d(
            fut_q,
            future_windows[best_candidate_index],
            distance_metric,
        )
        baseline_future = np.median(fut_candidates, axis=0)
        agnostic_loss = _metric_loss_1d(fut_q, baseline_future, distance_metric)
        denom = max(abs(agnostic_loss), 1.0e-12)
        rel_improvement = float((agnostic_loss - phase_loss) / denom)

        candidate_future_losses = _metric_loss_rows(fut_q, fut_candidates, distance_metric)
        percentile = float(np.mean(candidate_future_losses <= phase_loss) * 100.0)

        phase_losses.append(float(phase_loss))
        agnostic_losses.append(float(agnostic_loss))
        relative_improvements.append(rel_improvement)
        loss_percentiles.append(percentile)
        phase_hat_frames.append(best_phase_frame)
        matched_offsets.append(int(best_phase_frame - int(t)))

        loss_curve.append(
            {
                "query_frame": int(t),
                "matched_phase_frame": best_phase_frame,
                "matched_phase_offset_frames": int(best_phase_frame - int(t)),
                "phase_loss": float(phase_loss),
                "phase_agnostic_loss": float(agnostic_loss),
                "relative_improvement": rel_improvement,
            }
        )
        loss_percentile_curve.append(
            {
                "query_frame": int(t),
                "phase_loss_percentile_vs_nonlocal_candidate_futures": percentile,
            }
        )
        phase_confidence_gap_curve.append(
            {
                "query_frame": int(t),
                "matched_phase_frame": best_phase_frame,
                "best_history_distance": best_history_distance,
                "second_best_history_distance": second_history_distance,
                "phase_confidence_gap": confidence_gap,
            }
        )

    valid_query_count = int(len(phase_losses))
    valid_query_fraction = float(valid_query_count) / float(candidate_count)
    median_improvement = (
        float(np.median(np.asarray(relative_improvements, dtype=np.float64)))
        if relative_improvements
        else None
    )
    config_valid = (
        valid_query_count >= LAYER_C_MIN_VALID_QUERY_COUNT
        and valid_query_fraction >= LAYER_C_MIN_VALID_QUERY_FRACTION
    )
    beats_baseline = bool(
        config_valid
        and median_improvement is not None
        and median_improvement > LAYER_C_IMPROVEMENT_TOL
    )

    if not config_valid:
        config_status = "insufficient_valid_queries"
    elif beats_baseline:
        config_status = "phase_lookup_beats_phase_agnostic_baseline"
    else:
        config_status = "phase_lookup_does_not_beat_phase_agnostic_baseline"

    return {
        "history_window_frames": int(history_window_frames),
        "future_horizon_frames": int(future_horizon_frames),
        "neighborhood_radius_frames": int(neighborhood_radius_frames),
        "distance_metric": distance_metric,
        "valid_phase_candidate_count": candidate_count,
        "valid_query_count": valid_query_count,
        "valid_query_fraction": valid_query_fraction,
        "min_valid_query_count_estimator_only": LAYER_C_MIN_VALID_QUERY_COUNT,
        "min_valid_query_fraction_estimator_only": LAYER_C_MIN_VALID_QUERY_FRACTION,
        "config_status": config_status,
        "beats_phase_agnostic_baseline": beats_baseline,
        "median_relative_improvement": median_improvement,
        "phase_loss_quantiles": _quantile_summary(phase_losses),
        "phase_agnostic_loss_quantiles": _quantile_summary(agnostic_losses),
        "relative_improvement_quantiles": _quantile_summary(relative_improvements),
        "loss_percentile_quantiles": _quantile_summary(loss_percentiles),
        "phase_confidence_gap_quantiles": _quantile_summary(confidence_gaps),
        "phase_hat_curve_summary": {
            "matched_phase_frame_quantiles": _quantile_summary(phase_hat_frames),
            "matched_phase_offset_frame_quantiles": _quantile_summary(matched_offsets),
            "unique_matched_phase_frame_count": int(len(set(phase_hat_frames))),
        },
        "loss_curve": loss_curve,
        "loss_percentile_curve": loss_percentile_curve,
        "phase_confidence_gap": phase_confidence_gap_curve,
        "skipped_queries": skipped_queries,
    }


def _summarise_layer_c_group(
    *,
    group_payload: dict[str, Any],
    config_results: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_degenerate = bool(group_payload["reference_degenerate"])
    active_channels = list(group_payload["active_channels"])

    if reference_degenerate:
        phase_structure_status = "phase_degenerate"
        evidence_status = "INSUFFICIENT_EVIDENCE"
        self_consistency_signal_status = "not_testable_reference_degenerate"
    else:
        valid_configs = [c for c in config_results if c["valid_query_count"] > 0]
        beat_flags = [bool(c["beats_phase_agnostic_baseline"]) for c in config_results]
        all_configs_valid = all(
            c["config_status"] != "insufficient_valid_queries" for c in config_results
        )
        if not valid_configs or not all_configs_valid:
            self_consistency_signal_status = "insufficient_valid_queries_across_grid"
        elif all(beat_flags):
            self_consistency_signal_status = (
                "consistent_signal_detected_single_trajectory_only"
            )
        elif any(beat_flags):
            self_consistency_signal_status = (
                "mixed_across_estimator_grid_expected_single_trajectory_limitation"
            )
        else:
            self_consistency_signal_status = "no_phase_lookup_advantage_detected"
        phase_structure_status = "insufficient_evidence"
        evidence_status = "INSUFFICIENT_EVIDENCE"

    phase_medians = [
        c["phase_loss_quantiles"]["p50"]
        for c in config_results
        if c["phase_loss_quantiles"]["p50"] is not None
    ]
    agnostic_medians = [
        c["phase_agnostic_loss_quantiles"]["p50"]
        for c in config_results
        if c["phase_agnostic_loss_quantiles"]["p50"] is not None
    ]
    improvement_medians = [
        c["median_relative_improvement"]
        for c in config_results
        if c["median_relative_improvement"] is not None
    ]

    return {
        "phase_structure_status": phase_structure_status,
        "evidence_status": evidence_status,
        "self_consistency_signal_status": self_consistency_signal_status,
        "reference_degenerate": reference_degenerate,
        "active_channels": active_channels,
        "excluded_channels": group_payload["excluded_channels"],
        "channel_scale": group_payload["channel_scale"],
        "source": group_payload["source"],
        "group_ablation_role": group_payload["group_ablation_role"],
        "config_results": config_results,
        "baseline_loss_quantiles": _quantile_summary(agnostic_medians),
        "loss_curve_summary": {
            "phase_loss_median_across_configs": _quantile_summary(phase_medians),
            "phase_agnostic_loss_median_across_configs": _quantile_summary(agnostic_medians),
            "median_relative_improvement_across_configs": _quantile_summary(
                improvement_medians
            ),
        },
        "phase_hat_curve_summary": {
            "unique_matched_phase_frame_count_by_config": [
                c["phase_hat_curve_summary"]["unique_matched_phase_frame_count"]
                for c in config_results
            ],
        },
    }


def _process_clip_phase_library_check(
    clip_info: dict[str, Any],
    contact_signals: dict[str, np.ndarray],
) -> dict[str, Any]:
    yaw_rad: np.ndarray = clip_info["root_yaw_rad"]
    pos_xy: np.ndarray = clip_info["root_pos_xy_m"]
    vel_xy: np.ndarray = clip_info["root_vel_xy_mps"]
    fps: float = clip_info["fps"]

    quotient = _canonical_quotient_features(yaw_rad, pos_xy, vel_xy, fps)
    channel_arrays = _extract_layer_b_channels(quotient, contact_signals)

    per_group: dict[str, Any] = {}
    for group_name, group_def in LAYER_C_FEATURE_GROUPS.items():
        group_payload = _build_layer_c_group_matrix(
            group_name=group_name,
            group_def=group_def,
            channel_arrays=channel_arrays,
        )
        config_results: list[dict[str, Any]] = []
        if not group_payload["reference_degenerate"]:
            z_matrix: np.ndarray = group_payload["z_matrix"]
            for history_window_frames in LAYER_C_HISTORY_WINDOW_FRAMES:
                for future_horizon_frames in LAYER_C_FUTURE_HORIZON_FRAMES:
                    for neighborhood_radius_frames in LAYER_C_NEIGHBORHOOD_RADIUS_FRAMES:
                        for distance_metric in LAYER_C_DISTANCE_METRICS:
                            config_results.append(
                                _run_layer_c_config(
                                    z_matrix=z_matrix,
                                    history_window_frames=history_window_frames,
                                    future_horizon_frames=future_horizon_frames,
                                    neighborhood_radius_frames=neighborhood_radius_frames,
                                    distance_metric=distance_metric,
                                )
                            )
        per_group[group_name] = _summarise_layer_c_group(
            group_payload=group_payload,
            config_results=config_results,
        )

    return {
        "clip": clip_info["clip"],
        "raw_json_path": clip_info["raw_json_path"],
        "fps": fps,
        "frame_count": clip_info["frame_count"],
        "track": TRACK_BY_MODE["phase_library_check"],
        "track_role": TRACK_ROLE_BY_MODE["phase_library_check"],
        "clip_role": "reference_clip",
        "feature_groups_layer_c": per_group,
        "excluded_feature_groups_layer_c": LAYER_C_EXCLUDED_FEATURE_GROUPS,
        "membership_status": "not_implemented_layerC_minimal",
        "query_boundary_status": "not_run_layerC_minimal_reserved_for_layerC1",
    }


def _build_internal_se2_precondition(
    clip_infos: list[dict[str, Any]],
) -> dict[str, Any]:
    per_clip_gauge = [_process_clip_gauge_check(info) for info in clip_infos]
    yaw_sanity = _build_yaw_invariance_sanity(per_clip_gauge)
    translation_sanity = _build_translation_invariance_sanity(per_clip_gauge)
    se2_sanity = _build_se2_gauge_sanity(
        yaw_sanity,
        translation_sanity,
        per_clip_gauge,
    )
    return {
        "status": se2_sanity["status"],
        "source": "internal_lightweight_layerA_rerun_not_external_artifact",
        "yaw_invariance_sanity": yaw_sanity,
        "translation_invariance_sanity": translation_sanity,
        "se2_gauge_sanity": se2_sanity,
        "per_clip": [
            {
                "clip": entry["clip"],
                "max_abs_error_over_rotation_grid": entry["max_abs_error_over_rotation_grid"],
                "max_abs_error_over_translation_grid": entry["max_abs_error_over_translation_grid"],
                "max_abs_error_over_se2_grid": entry["max_abs_error_over_se2_grid"],
                "rotation_finite_coverage": entry["rotation_finite_coverage"],
                "translation_finite_coverage": entry["translation_finite_coverage"],
            }
            for entry in per_clip_gauge
        ],
    }


# -----------------------------------------------------------------------
# query_boundary_check (Layer C.1) — Walk_F self-baseline band +
# query leave/return interval + censoring audit on 4 turn clips
# -----------------------------------------------------------------------


def _layer_c1_forbid_turn_dyn(group_name: str, context: str) -> None:
    """Hard guardrail: turn_dyn MUST NOT enter the combined-membership /
    boundary aggregation path on Walk_F.

    The reason is recorded in
    ``docs/aperiodic_transition/2026-05-22_walk_f_causal_state_scaffold_v1.md``
    §3.3 and in the Layer C.1 contract §1 rule 3. Any caller that hands
    a ``turn_dyn`` group name to this function is treated as a contract
    bug, not a recoverable estimator condition.
    """
    if group_name == "turn_dyn":
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer C.1 refuses to route turn_dyn "
            f"through {context!r}; turn_dyn is excluded from combined membership "
            "by docs/aperiodic_transition/"
            "2026-05-23_walk_f_causal_state_layerc1_query_boundary_contract.md "
            "§1 rule 3 and §3. turn_dyn may only appear as debug / inspection."
        )


def _normalize_with_walk_f_scale(
    *,
    walk_f_group_payload: dict[str, Any],
    query_channel_arrays: dict[str, np.ndarray],
    query_clip_name: str,
    group_name: str,
) -> dict[str, Any]:
    """Apply Walk_F's per-channel median + robust scale to query channels.

    Walk_F's normalization parameters live in
    ``walk_f_group_payload["channel_scale"]``. This function does NOT
    silently fall back to the query clip's own scale if a Walk_F channel
    is degenerate or missing — instead it raises so the contract failure
    is visible. Reference-degenerate groups are filtered upstream and never
    reach this function.
    """
    _layer_c1_forbid_turn_dyn(group_name, "_normalize_with_walk_f_scale")
    active_channels: list[str] = list(walk_f_group_payload["active_channels"])
    channel_scale: dict[str, Any] = walk_f_group_payload["channel_scale"]
    n_frames: int | None = None
    z_cols: list[np.ndarray] = []
    for channel_name in active_channels:
        if channel_name not in query_channel_arrays:
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer C.1 channel {channel_name!r} "
                f"missing for query clip {query_clip_name!r} in group "
                f"{group_name!r}; refusing silent skip "
                "(docs/removal_policy.md §3-§4)."
            )
        arr = np.asarray(query_channel_arrays[channel_name], dtype=np.float64)
        if arr.ndim != 1:
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer C.1 channel {channel_name!r} "
                f"on query {query_clip_name!r} must be 1-D, got {arr.shape}."
            )
        if n_frames is None:
            n_frames = int(arr.shape[0])
        elif int(arr.shape[0]) != n_frames:
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer C.1 channel length mismatch "
                f"on query {query_clip_name!r} group {group_name!r}: expected "
                f"{n_frames}, got {arr.shape[0]} for {channel_name!r}."
            )
        if not np.all(np.isfinite(arr)):
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer C.1 channel {channel_name!r} "
                f"on query {query_clip_name!r} contains non-finite values."
            )
        stats = channel_scale.get(channel_name)
        if stats is None:
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer C.1 missing Walk_F scale "
                f"for channel {channel_name!r} in group {group_name!r}; "
                "refusing to renormalize the query with its own scale."
            )
        median = stats.get("median")
        robust_std = stats.get("robust_std_from_mad")
        if (
            median is None
            or robust_std is None
            or not math.isfinite(float(median))
            or not math.isfinite(float(robust_std))
            or float(robust_std) <= 0.0
        ):
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer C.1 Walk_F scale for "
                f"{channel_name!r} in group {group_name!r} is not usable "
                f"(median={median!r}, robust_std_from_mad={robust_std!r}); "
                "the channel should have been filtered upstream."
            )
        z_cols.append(((arr - float(median)) / float(robust_std)).astype(np.float64))

    if n_frames is None:
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer C.1 group {group_name!r} has no "
            "active channels for the query normalization; this should have "
            "been caught by the Walk_F reference_degenerate guard."
        )
    if z_cols:
        z_matrix = np.stack(z_cols, axis=1).astype(np.float64)
    else:
        z_matrix = np.zeros((n_frames, 0), dtype=np.float64)
    return {
        "z_matrix": z_matrix,
        "active_channels": active_channels,
        "n_frames": int(n_frames),
    }


def _walk_f_self_baseline_loss_distribution(
    *,
    walk_f_z_matrix: np.ndarray,
    history_window_frames: int,
    future_horizon_frames: int,
    neighborhood_radius_frames: int,
    distance_metric: str,
) -> dict[str, Any]:
    """Compute Walk_F leave-one-neighborhood phase-loss distribution.

    Returns the per-Walk_F-phase loss array (1-D float64) plus the
    Walk_F history / future windows and phase-frame indices needed to
    drive query lookup on the same estimator setting.
    """
    windows = _build_phase_windows(
        walk_f_z_matrix,
        history_window_frames=history_window_frames,
        future_horizon_frames=future_horizon_frames,
    )
    phase_frames: np.ndarray = windows["phase_frames"]
    history_windows: np.ndarray = windows["history_windows"]
    future_windows: np.ndarray = windows["future_windows"]
    candidate_count = int(phase_frames.shape[0])
    if candidate_count == 0:
        return {
            "phase_frames": phase_frames,
            "history_windows": history_windows,
            "future_windows": future_windows,
            "self_baseline_loss_array": np.zeros((0,), dtype=np.float64),
            "self_baseline_loss_phase_frames": np.zeros((0,), dtype=np.int64),
            "valid_self_baseline_count": 0,
            "candidate_count": 0,
        }

    losses: list[float] = []
    valid_frames: list[int] = []
    for query_i, t in enumerate(phase_frames):
        nonlocal_mask = np.abs(phase_frames - int(t)) > int(neighborhood_radius_frames)
        candidate_indices = np.nonzero(nonlocal_mask)[0]
        if candidate_indices.size == 0:
            continue
        hist_q = history_windows[query_i]
        hist_candidates = history_windows[candidate_indices]
        hist_distances = _metric_loss_rows(hist_q, hist_candidates, distance_metric)
        best_local = int(np.argmin(hist_distances))
        best_candidate_index = int(candidate_indices[best_local])
        phase_loss = _metric_loss_1d(
            future_windows[query_i],
            future_windows[best_candidate_index],
            distance_metric,
        )
        losses.append(float(phase_loss))
        valid_frames.append(int(t))

    loss_array = np.asarray(losses, dtype=np.float64)
    return {
        "phase_frames": phase_frames,
        "history_windows": history_windows,
        "future_windows": future_windows,
        "self_baseline_loss_array": loss_array,
        "self_baseline_loss_phase_frames": np.asarray(valid_frames, dtype=np.int64),
        "valid_self_baseline_count": int(loss_array.shape[0]),
        "candidate_count": int(candidate_count),
    }


def _query_phase_loss_against_walk_f(
    *,
    walk_f_phase_frames: np.ndarray,
    walk_f_history_windows: np.ndarray,
    walk_f_future_windows: np.ndarray,
    query_z_matrix: np.ndarray,
    history_window_frames: int,
    future_horizon_frames: int,
    distance_metric: str,
) -> dict[str, Any]:
    """For every query phase frame, match H_q(t) to nearest Walk_F H_F(p)
    (no neighborhood exclusion; query and reference are different clips)
    and compute the phase-aware future loss against Y_F(p*).
    """
    query_windows = _build_phase_windows(
        query_z_matrix,
        history_window_frames=history_window_frames,
        future_horizon_frames=future_horizon_frames,
    )
    q_phase_frames: np.ndarray = query_windows["phase_frames"]
    q_H: np.ndarray = query_windows["history_windows"]
    q_Y: np.ndarray = query_windows["future_windows"]
    if q_phase_frames.shape[0] == 0 or walk_f_phase_frames.shape[0] == 0:
        return {
            "query_phase_frames": q_phase_frames,
            "phase_loss_array": np.zeros((0,), dtype=np.float64),
            "matched_walk_f_phase_frames": np.zeros((0,), dtype=np.int64),
        }

    losses: list[float] = []
    matched: list[int] = []
    for query_i in range(q_phase_frames.shape[0]):
        hist_q = q_H[query_i]
        hist_distances = _metric_loss_rows(
            hist_q, walk_f_history_windows, distance_metric
        )
        best_local = int(np.argmin(hist_distances))
        best_walk_f_frame = int(walk_f_phase_frames[best_local])
        phase_loss = _metric_loss_1d(
            q_Y[query_i],
            walk_f_future_windows[best_local],
            distance_metric,
        )
        losses.append(float(phase_loss))
        matched.append(best_walk_f_frame)

    return {
        "query_phase_frames": q_phase_frames,
        "phase_loss_array": np.asarray(losses, dtype=np.float64),
        "matched_walk_f_phase_frames": np.asarray(matched, dtype=np.int64),
    }


def _earliest_consecutive_run(
    mask: np.ndarray, value: bool, min_length: int
) -> tuple[int, int] | None:
    """Return the earliest contiguous run in ``mask`` matching ``value``
    of length at least ``min_length``, as ``[start_idx, end_idx]``
    (closed-interval indices into ``mask``). Returns None if no such run
    exists.
    """
    if mask.size == 0 or int(min_length) <= 0:
        return None
    target = bool(value)
    min_len = int(min_length)
    run_start: int | None = None
    n = int(mask.shape[0])
    for i in range(n):
        if bool(mask[i]) == target:
            if run_start is None:
                run_start = i
            if i - run_start + 1 >= min_len:
                # Extend to the end of the run.
                end = i
                while end + 1 < n and bool(mask[end + 1]) == target:
                    end += 1
                return (run_start, end)
        else:
            run_start = None
    return None


def _detect_leave_return_intervals(
    *,
    query_phase_frames: np.ndarray,
    phase_loss_array: np.ndarray,
    band_value: float,
    min_consecutive_leave: int,
    min_consecutive_return: int,
) -> dict[str, Any]:
    """Apply the Layer C.1 leave/return detection to one query loss curve.

    All boundary frame numbers are CLIP frame indices, mapped through
    ``query_phase_frames``. The earliest reachable phase frame given the
    history window is the censoring anchor (``left_censored_leave =
    leave_interval_starts_at_phase_frames[0]``).
    """
    n = int(phase_loss_array.shape[0])
    if n == 0:
        return {
            "out_of_band_mask": np.zeros((0,), dtype=bool),
            "out_of_band_frame_count": 0,
            "leave_interval": None,
            "return_interval": None,
            "leave_interval_idx": None,
            "return_interval_idx": None,
            "left_censored_leave": False,
            "right_censored_return": False,
        }
    mask = phase_loss_array > float(band_value)
    leave_idx = _earliest_consecutive_run(
        mask, True, int(min_consecutive_leave)
    )
    return_idx: tuple[int, int] | None = None
    if leave_idx is not None:
        leave_end = leave_idx[1]
        post = mask[leave_end + 1 :]
        post_run = _earliest_consecutive_run(
            post, False, int(min_consecutive_return)
        )
        if post_run is not None:
            return_idx = (
                post_run[0] + leave_end + 1,
                post_run[1] + leave_end + 1,
            )

    def _to_clip_frame(idx: int) -> int:
        return int(query_phase_frames[idx])

    leave_interval = (
        None
        if leave_idx is None
        else [_to_clip_frame(leave_idx[0]), _to_clip_frame(leave_idx[1])]
    )
    return_interval = (
        None
        if return_idx is None
        else [_to_clip_frame(return_idx[0]), _to_clip_frame(return_idx[1])]
    )
    left_censored_leave = bool(leave_idx is not None and leave_idx[0] == 0)
    right_censored_return = bool(
        (leave_idx is not None and return_idx is None)
        or bool(mask[-1])
    )
    return {
        "out_of_band_mask": mask.astype(bool),
        "out_of_band_frame_count": int(np.sum(mask)),
        "leave_interval": leave_interval,
        "return_interval": return_interval,
        "leave_interval_idx": list(leave_idx) if leave_idx is not None else None,
        "return_interval_idx": list(return_idx) if return_idx is not None else None,
        "left_censored_leave": left_censored_leave,
        "right_censored_return": right_censored_return,
    }


def _classify_return_to_reference_status_raw(
    *,
    leave_interval: list[int] | None,
    return_interval: list[int] | None,
    valid_query_count: int,
    valid_query_fraction: float,
) -> str:
    """Compute the per-config return_to_reference_status BEFORE neighbor
    consistency is checked. The neighbor-consistency pass (§5 of the
    Layer C.1 contract) may downgrade ``returned`` /
    ``never_left`` / ``never_returned`` to ``INSUFFICIENT_EVIDENCE``.
    """
    if (
        valid_query_count < LAYER_C1_MIN_VALID_QUERY_COUNT
        or valid_query_fraction < LAYER_C1_MIN_VALID_QUERY_FRACTION
    ):
        return "INSUFFICIENT_EVIDENCE"
    if leave_interval is None:
        return "never_left"
    if return_interval is None:
        return "never_returned"
    return "returned"


def _layer_c1_config_id(
    *,
    history_window_frames: int,
    future_horizon_frames: int,
    neighborhood_radius_frames: int,
    distance_metric: str,
    band_quantile_label: str,
    min_consecutive_leave: int,
    min_consecutive_return: int,
) -> str:
    return (
        f"H{int(history_window_frames)}"
        f"_F{int(future_horizon_frames)}"
        f"_N{int(neighborhood_radius_frames)}"
        f"_M{distance_metric}"
        f"_Q{band_quantile_label}"
        f"_L{int(min_consecutive_leave)}"
        f"_R{int(min_consecutive_return)}"
    )


def _layer_c1_neighbor_diff_dims(a: dict[str, Any], b: dict[str, Any]) -> int:
    diffs = 0
    for key in (
        "history_window_frames",
        "future_horizon_frames",
        "neighborhood_radius_frames",
        "distance_metric",
        "band_quantile",
        "min_consecutive_frames_for_leave",
        "min_consecutive_frames_for_return",
    ):
        if a[key] != b[key]:
            diffs += 1
            if diffs > 1:
                return diffs
    return diffs


def _layer_c1_endpoint_conflict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    """Compare the leave/return endpoints of two configs. Returns a small
    diagnostic dict if they disagree (existence boolean OR endpoint
    delta exceeds tolerance), otherwise None.
    """
    tol = int(LAYER_C1_NEIGHBOR_CONSISTENCY_TOL_FRAMES)
    a_leave = a["leave_interval"]
    b_leave = b["leave_interval"]
    a_ret = a["return_interval"]
    b_ret = b["return_interval"]

    if (a_leave is None) != (b_leave is None):
        return {
            "kind": "leave_existence_disagreement",
            "self_leave": a_leave,
            "neighbor_leave": b_leave,
        }
    if (a_ret is None) != (b_ret is None):
        return {
            "kind": "return_existence_disagreement",
            "self_return": a_ret,
            "neighbor_return": b_ret,
        }
    if a_leave is not None and b_leave is not None:
        if (
            abs(int(a_leave[0]) - int(b_leave[0])) > tol
            or abs(int(a_leave[1]) - int(b_leave[1])) > tol
        ):
            return {
                "kind": "leave_endpoint_delta_exceeds_tolerance",
                "tol_frames": tol,
                "self_leave": a_leave,
                "neighbor_leave": b_leave,
            }
    if a_ret is not None and b_ret is not None:
        if (
            abs(int(a_ret[0]) - int(b_ret[0])) > tol
            or abs(int(a_ret[1]) - int(b_ret[1])) > tol
        ):
            return {
                "kind": "return_endpoint_delta_exceeds_tolerance",
                "tol_frames": tol,
                "self_return": a_ret,
                "neighbor_return": b_ret,
            }
    return None


def _apply_neighbor_consistency_override(
    configs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan the per-config list within (query_clip, group) and downgrade
    configs whose leave/return endpoints disagree with a 1-step grid
    neighbor. Returns the mutated config list (in place) and the list of
    conflict pairs recorded for the per-group aggregate.
    """
    n = len(configs)
    conflict_pairs: list[dict[str, Any]] = []
    for i in range(n):
        ci = configs[i]
        # Skip configs whose raw status is already INSUFFICIENT_EVIDENCE due
        # to invalid query counts: nothing to override.
        if ci["return_to_reference_status_pre_neighbor_consistency"] == (
            "INSUFFICIENT_EVIDENCE"
        ):
            ci["return_to_reference_status"] = "INSUFFICIENT_EVIDENCE"
            ci["neighbor_consistency_conflicts"] = []
            continue
        conflicts_for_ci: list[dict[str, Any]] = []
        for j in range(n):
            if i == j:
                continue
            cj = configs[j]
            if _layer_c1_neighbor_diff_dims(ci, cj) != 1:
                continue
            # If the neighbor itself is invalid by query-count, skip — we
            # do not let an INSUFFICIENT_EVIDENCE-by-count neighbor force
            # a healthy config to be downgraded; only well-formed endpoint
            # disagreements count.
            if cj["return_to_reference_status_pre_neighbor_consistency"] == (
                "INSUFFICIENT_EVIDENCE"
            ):
                continue
            conflict = _layer_c1_endpoint_conflict(ci, cj)
            if conflict is not None:
                conflicts_for_ci.append(
                    {
                        "neighbor_config_id": cj["config_id"],
                        **conflict,
                    }
                )
                conflict_pairs.append(
                    {
                        "config_id_a": ci["config_id"],
                        "config_id_b": cj["config_id"],
                        "kind": conflict["kind"],
                    }
                )
        if conflicts_for_ci:
            ci["return_to_reference_status"] = "INSUFFICIENT_EVIDENCE"
            ci["neighbor_consistency_conflicts"] = conflicts_for_ci
        else:
            ci["return_to_reference_status"] = ci[
                "return_to_reference_status_pre_neighbor_consistency"
            ]
            ci["neighbor_consistency_conflicts"] = []
    return configs, conflict_pairs


def _summarize_self_baseline_loss(
    loss_array: np.ndarray,
) -> dict[str, Any]:
    return _quantile_summary(loss_array)


def _layer_c1_group_loss_curve_entry(
    *,
    query_phase_frames: np.ndarray,
    matched_walk_f_phase_frames: np.ndarray,
    phase_loss_array: np.ndarray,
    out_of_band_mask: np.ndarray,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for idx in range(int(query_phase_frames.shape[0])):
        entries.append(
            {
                "query_clip_phase_frame": int(query_phase_frames[idx]),
                "matched_walk_f_phase_frame": int(matched_walk_f_phase_frames[idx]),
                "phase_loss": float(phase_loss_array[idx]),
                "out_of_band": bool(out_of_band_mask[idx]),
            }
        )
    return entries


def _layer_c1_group_loss_percentile_curve(
    *,
    query_phase_frames: np.ndarray,
    phase_loss_array: np.ndarray,
    walk_f_self_baseline_loss_array: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute the percentile of each query phase loss relative to the
    Walk_F self-baseline loss distribution. If the baseline is empty,
    every percentile is None.
    """
    entries: list[dict[str, Any]] = []
    if walk_f_self_baseline_loss_array.shape[0] == 0:
        for idx in range(int(query_phase_frames.shape[0])):
            entries.append(
                {
                    "query_clip_phase_frame": int(query_phase_frames[idx]),
                    "phase_loss_percentile_vs_walk_f_self_baseline": None,
                }
            )
        return entries
    sorted_baseline = np.sort(walk_f_self_baseline_loss_array)
    for idx in range(int(query_phase_frames.shape[0])):
        loss = float(phase_loss_array[idx])
        pct = float(
            np.searchsorted(sorted_baseline, loss, side="right")
        ) / float(sorted_baseline.shape[0]) * 100.0
        entries.append(
            {
                "query_clip_phase_frame": int(query_phase_frames[idx]),
                "phase_loss_percentile_vs_walk_f_self_baseline": pct,
            }
        )
    return entries


def _run_layer_c1_group_for_query(
    *,
    query_clip_name: str,
    group_name: str,
    walk_f_group_payload: dict[str, Any],
    query_channel_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute the full Layer C.1 config sweep for one (query_clip, group)
    pair. Returns a dict containing per-config rows and group-level
    aggregate flags. Reference-degenerate groups are short-circuited.
    """
    _layer_c1_forbid_turn_dyn(group_name, "_run_layer_c1_group_for_query")
    if bool(walk_f_group_payload["reference_degenerate"]):
        return {
            "group": group_name,
            "phase_structure_status": "phase_degenerate",
            "evidence_status": "INSUFFICIENT_EVIDENCE",
            "reference_degenerate": True,
            "active_channels": list(walk_f_group_payload["active_channels"]),
            "excluded_channels": walk_f_group_payload["excluded_channels"],
            "group_ablation_role": walk_f_group_payload["group_ablation_role"],
            "configs": [],
            "neighbor_consistency_conflict_pairs": [],
            "boundary_detection_skipped_reason": (
                "Walk_F reference_degenerate: every channel is at or below the "
                "estimator-level MAD epsilon, so the canonical SE(2) z-score "
                "would diverge under a query rescale. Boundary detection is "
                "skipped (Layer C.1 contract §4)."
            ),
        }

    query_payload = _normalize_with_walk_f_scale(
        walk_f_group_payload=walk_f_group_payload,
        query_channel_arrays=query_channel_arrays,
        query_clip_name=query_clip_name,
        group_name=group_name,
    )
    query_z_matrix: np.ndarray = query_payload["z_matrix"]
    walk_f_z_matrix: np.ndarray = walk_f_group_payload["z_matrix"]

    config_rows: list[dict[str, Any]] = []
    walk_f_per_estimator_cache: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    query_per_estimator_cache: dict[tuple[int, int, str], dict[str, Any]] = {}

    for history_window_frames in LAYER_C_HISTORY_WINDOW_FRAMES:
        for future_horizon_frames in LAYER_C_FUTURE_HORIZON_FRAMES:
            for neighborhood_radius_frames in LAYER_C_NEIGHBORHOOD_RADIUS_FRAMES:
                for distance_metric in LAYER_C_DISTANCE_METRICS:
                    walk_f_cache_key = (
                        int(history_window_frames),
                        int(future_horizon_frames),
                        int(neighborhood_radius_frames),
                        str(distance_metric),
                    )
                    walk_f_baseline = walk_f_per_estimator_cache.get(walk_f_cache_key)
                    if walk_f_baseline is None:
                        walk_f_baseline = _walk_f_self_baseline_loss_distribution(
                            walk_f_z_matrix=walk_f_z_matrix,
                            history_window_frames=history_window_frames,
                            future_horizon_frames=future_horizon_frames,
                            neighborhood_radius_frames=neighborhood_radius_frames,
                            distance_metric=distance_metric,
                        )
                        walk_f_per_estimator_cache[walk_f_cache_key] = walk_f_baseline

                    query_cache_key = (
                        int(history_window_frames),
                        int(future_horizon_frames),
                        str(distance_metric),
                    )
                    query_match = query_per_estimator_cache.get(query_cache_key)
                    if query_match is None:
                        query_match = _query_phase_loss_against_walk_f(
                            walk_f_phase_frames=walk_f_baseline["phase_frames"],
                            walk_f_history_windows=walk_f_baseline[
                                "history_windows"
                            ],
                            walk_f_future_windows=walk_f_baseline[
                                "future_windows"
                            ],
                            query_z_matrix=query_z_matrix,
                            history_window_frames=history_window_frames,
                            future_horizon_frames=future_horizon_frames,
                            distance_metric=distance_metric,
                        )
                        query_per_estimator_cache[query_cache_key] = query_match

                    self_baseline_loss = walk_f_baseline[
                        "self_baseline_loss_array"
                    ]
                    self_baseline_quantiles = _summarize_self_baseline_loss(
                        self_baseline_loss
                    )
                    query_phase_frames = query_match["query_phase_frames"]
                    phase_loss_array = query_match["phase_loss_array"]
                    matched_walk_f = query_match["matched_walk_f_phase_frames"]
                    valid_query_count = int(phase_loss_array.shape[0])
                    candidate_count_walk_f = int(
                        walk_f_baseline["phase_frames"].shape[0]
                    )
                    valid_query_fraction = (
                        float(valid_query_count) / float(candidate_count_walk_f)
                        if candidate_count_walk_f > 0
                        else 0.0
                    )

                    for band_label, band_q in LAYER_C1_BAND_QUANTILES:
                        if self_baseline_loss.shape[0] == 0:
                            band_value: float | None = None
                        else:
                            band_value = float(
                                np.quantile(self_baseline_loss, float(band_q))
                            )

                        for min_leave in LAYER_C1_MIN_CONSEC_FRAMES_FOR_LEAVE:
                            for min_return in LAYER_C1_MIN_CONSEC_FRAMES_FOR_RETURN:
                                config_id = _layer_c1_config_id(
                                    history_window_frames=history_window_frames,
                                    future_horizon_frames=future_horizon_frames,
                                    neighborhood_radius_frames=neighborhood_radius_frames,
                                    distance_metric=distance_metric,
                                    band_quantile_label=band_label,
                                    min_consecutive_leave=min_leave,
                                    min_consecutive_return=min_return,
                                )
                                if (
                                    band_value is None
                                    or phase_loss_array.shape[0] == 0
                                ):
                                    raw_status = "INSUFFICIENT_EVIDENCE"
                                    leave_payload = {
                                        "out_of_band_mask": np.zeros(
                                            (0,), dtype=bool
                                        ),
                                        "out_of_band_frame_count": 0,
                                        "leave_interval": None,
                                        "return_interval": None,
                                        "leave_interval_idx": None,
                                        "return_interval_idx": None,
                                        "left_censored_leave": False,
                                        "right_censored_return": False,
                                    }
                                else:
                                    leave_payload = _detect_leave_return_intervals(
                                        query_phase_frames=query_phase_frames,
                                        phase_loss_array=phase_loss_array,
                                        band_value=band_value,
                                        min_consecutive_leave=min_leave,
                                        min_consecutive_return=min_return,
                                    )
                                    raw_status = (
                                        _classify_return_to_reference_status_raw(
                                            leave_interval=leave_payload[
                                                "leave_interval"
                                            ],
                                            return_interval=leave_payload[
                                                "return_interval"
                                            ],
                                            valid_query_count=valid_query_count,
                                            valid_query_fraction=valid_query_fraction,
                                        )
                                    )

                                if (
                                    band_value is not None
                                    and phase_loss_array.shape[0] > 0
                                ):
                                    loss_curve = _layer_c1_group_loss_curve_entry(
                                        query_phase_frames=query_phase_frames,
                                        matched_walk_f_phase_frames=matched_walk_f,
                                        phase_loss_array=phase_loss_array,
                                        out_of_band_mask=leave_payload[
                                            "out_of_band_mask"
                                        ],
                                    )
                                    loss_percentile_curve = (
                                        _layer_c1_group_loss_percentile_curve(
                                            query_phase_frames=query_phase_frames,
                                            phase_loss_array=phase_loss_array,
                                            walk_f_self_baseline_loss_array=(
                                                self_baseline_loss
                                            ),
                                        )
                                    )
                                else:
                                    loss_curve = []
                                    loss_percentile_curve = []

                                config_rows.append(
                                    {
                                        "config_id": config_id,
                                        "history_window_frames": int(
                                            history_window_frames
                                        ),
                                        "future_horizon_frames": int(
                                            future_horizon_frames
                                        ),
                                        "neighborhood_radius_frames": int(
                                            neighborhood_radius_frames
                                        ),
                                        "distance_metric": str(distance_metric),
                                        "band_quantile": band_label,
                                        "band_quantile_fraction": float(band_q),
                                        "min_consecutive_frames_for_leave": int(
                                            min_leave
                                        ),
                                        "min_consecutive_frames_for_return": int(
                                            min_return
                                        ),
                                        "valid_query_count": valid_query_count,
                                        "valid_query_fraction": (
                                            valid_query_fraction
                                        ),
                                        "min_valid_query_count_estimator_only": (
                                            LAYER_C1_MIN_VALID_QUERY_COUNT
                                        ),
                                        "min_valid_query_fraction_estimator_only": (
                                            LAYER_C1_MIN_VALID_QUERY_FRACTION
                                        ),
                                        "walk_f_self_baseline_candidate_count": (
                                            candidate_count_walk_f
                                        ),
                                        "walk_f_self_baseline_valid_count": int(
                                            walk_f_baseline[
                                                "valid_self_baseline_count"
                                            ]
                                        ),
                                        "band_quantile_value": band_value,
                                        "self_baseline_band_quantiles": (
                                            self_baseline_quantiles
                                        ),
                                        "phase_loss_quantiles_on_query": (
                                            _quantile_summary(phase_loss_array)
                                        ),
                                        "out_of_band_frame_count": int(
                                            leave_payload["out_of_band_frame_count"]
                                        ),
                                        "leave_interval": leave_payload[
                                            "leave_interval"
                                        ],
                                        "return_interval": leave_payload[
                                            "return_interval"
                                        ],
                                        "left_censored_leave": bool(
                                            leave_payload["left_censored_leave"]
                                        ),
                                        "right_censored_return": bool(
                                            leave_payload["right_censored_return"]
                                        ),
                                        "return_to_reference_status_pre_neighbor_consistency": (
                                            raw_status
                                        ),
                                        "return_to_reference_status": raw_status,
                                        "neighbor_consistency_conflicts": [],
                                        "loss_curve": loss_curve,
                                        "loss_percentile_curve": (
                                            loss_percentile_curve
                                        ),
                                    }
                                )

    config_rows, conflict_pairs = _apply_neighbor_consistency_override(config_rows)

    returned_n = sum(
        1
        for c in config_rows
        if c["return_to_reference_status"] == "returned"
    )
    never_left_n = sum(
        1
        for c in config_rows
        if c["return_to_reference_status"] == "never_left"
    )
    never_returned_n = sum(
        1
        for c in config_rows
        if c["return_to_reference_status"] == "never_returned"
    )
    insufficient_n = sum(
        1
        for c in config_rows
        if c["return_to_reference_status"] == "INSUFFICIENT_EVIDENCE"
    )
    left_censored_n = sum(1 for c in config_rows if c["left_censored_leave"])
    right_censored_n = sum(1 for c in config_rows if c["right_censored_return"])

    return {
        "group": group_name,
        "phase_structure_status": "insufficient_evidence",
        "evidence_status": "INSUFFICIENT_EVIDENCE",
        "reference_degenerate": False,
        "active_channels": list(walk_f_group_payload["active_channels"]),
        "excluded_channels": walk_f_group_payload["excluded_channels"],
        "group_ablation_role": walk_f_group_payload["group_ablation_role"],
        "configs": config_rows,
        "returned_config_count": int(returned_n),
        "never_left_config_count": int(never_left_n),
        "never_returned_config_count": int(never_returned_n),
        "insufficient_evidence_config_count": int(insufficient_n),
        "left_censored_leave_config_count": int(left_censored_n),
        "right_censored_return_config_count": int(right_censored_n),
        "neighbor_consistency_conflict_pair_count": int(len(conflict_pairs)),
        "neighbor_consistency_conflict_pairs": conflict_pairs,
        "boundary_detection_skipped_reason": None,
    }


def _build_walk_f_layer_c_group_payloads(
    *,
    walk_f_clip_info: dict[str, Any],
    walk_f_contact_signals: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    quotient = _canonical_quotient_features(
        walk_f_clip_info["root_yaw_rad"],
        walk_f_clip_info["root_pos_xy_m"],
        walk_f_clip_info["root_vel_xy_mps"],
        walk_f_clip_info["fps"],
    )
    channel_arrays = _extract_layer_b_channels(quotient, walk_f_contact_signals)
    payloads: dict[str, dict[str, Any]] = {}
    for group_name, group_def in LAYER_C1_FEATURE_GROUPS.items():
        _layer_c1_forbid_turn_dyn(
            group_name, "_build_walk_f_layer_c_group_payloads"
        )
        payloads[group_name] = _build_layer_c_group_matrix(
            group_name=group_name,
            group_def=group_def,
            channel_arrays=channel_arrays,
        )
    return payloads


def _build_query_channel_arrays(
    *,
    clip_info: dict[str, Any],
    contact_signals: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    quotient = _canonical_quotient_features(
        clip_info["root_yaw_rad"],
        clip_info["root_pos_xy_m"],
        clip_info["root_vel_xy_mps"],
        clip_info["fps"],
    )
    return _extract_layer_b_channels(quotient, contact_signals)


def _process_query_clip_boundary_check(
    *,
    query_clip_info: dict[str, Any],
    query_contact_signals: dict[str, np.ndarray],
    walk_f_group_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    query_channel_arrays = _build_query_channel_arrays(
        clip_info=query_clip_info,
        contact_signals=query_contact_signals,
    )
    per_group: dict[str, Any] = {}
    for group_name, walk_f_payload in walk_f_group_payloads.items():
        _layer_c1_forbid_turn_dyn(
            group_name, "_process_query_clip_boundary_check"
        )
        per_group[group_name] = _run_layer_c1_group_for_query(
            query_clip_name=query_clip_info["clip"],
            group_name=group_name,
            walk_f_group_payload=walk_f_payload,
            query_channel_arrays=query_channel_arrays,
        )
    return {
        "clip": query_clip_info["clip"],
        "raw_json_path": query_clip_info["raw_json_path"],
        "fps": query_clip_info["fps"],
        "frame_count": query_clip_info["frame_count"],
        "track": TRACK_BY_MODE["query_boundary_check"],
        "track_role": TRACK_ROLE_BY_MODE["query_boundary_check"],
        "clip_role": "query_clip",
        "feature_groups_layer_c1": per_group,
        "excluded_feature_groups_layer_c1": LAYER_C1_EXCLUDED_FEATURE_GROUPS,
        "attractor_membership_status": (
            "not_emitted_by_this_tool_layer_c1_query_boundary_only"
        ),
        "event_head_target_status": "not_emitted_by_this_tool",
        "handoff_ready_status": "not_emitted_by_this_tool",
        "transition_done_status": "not_emitted_by_this_tool",
        "cross_attractor_claim_status": "forbidden_by_contract",
        "transition_truth_promotion": "forbidden_by_contract",
    }


# -----------------------------------------------------------------------
# pose_reference_scale_check (Layer B.1) — Walk_F-only pose reference
# scale + degeneracy + processed-data schema audit
# -----------------------------------------------------------------------


def _layer_b1_forbid_turn_dyn(group_name: str, context: str) -> None:
    """Binary guardrail per Layer B.1 contract §1 rule 3.

    Routing ``turn_dyn`` (or any future degenerate group) through a
    Layer B.1 helper that touches combined-membership semantics is a
    contract violation. There is NO tolerance threshold under which
    ``turn_dyn`` is "almost allowed".
    """
    if group_name == "turn_dyn":
        raise FailFastError(
            "[walk_f_causal_state_probe] Layer B.1 contract §1 rule 3 "
            "violation: turn_dyn MUST NEVER be folded into combined-"
            f"membership / combined-score evidence (entry={context!r}). "
            "Walk_F turn_dyn is reference-degenerate per Layer B; routing "
            "it through this entry point would silently produce an "
            "unbounded combined score. No silent fallback "
            "(docs/removal_policy.md §3-§4)."
        )


LAYER_B1_ALLOWED_SCALE_SOURCE: str = "walk_f"


def _layer_b1_forbid_query_self_rescale(
    context: str, *, scale_source: str | None = None
) -> None:
    """Binary guardrail per Layer B.1 contract §1 rule 2.

    Any normalization helper that builds a query z-statistic MUST
    declare ``scale_source = "walk_f"``. Any other value (including
    ``"query_self"``, ``"per_clip"``, ``None``, etc.) raises.
    Helpers that have no business normalising still call this with no
    declared source as a defense-in-depth sentinel.
    """
    raise FailFastError(
        "[walk_f_causal_state_probe] Layer B.1 contract §1 rule 2 "
        "violation: query clips MUST NOT re-derive their own per-channel "
        f"median/MAD scale (entry={context!r}, scale_source="
        f"{scale_source!r}; only {LAYER_B1_ALLOWED_SCALE_SOURCE!r} is "
        "permitted). Walk_F is the only reference scale; query "
        "inspection is Walk_F-projected only. No silent fallback "
        "(docs/removal_policy.md §3-§4)."
    )


def _load_clip_processed_data(
    raw_root: Path,
    clip: str,
    *,
    json_frame_count: int,
    json_fps: float,
) -> dict[str, Any]:
    """Load + schema-validate ``raw_data/processed_data/<clip>.npz``.

    Validates: required keys, ranks, shapes, dtypes, finite values,
    bone count, meta_json units / spaces, FPS vs raw JSON, frame
    count vs raw JSON. Reads the NPZ with ``allow_pickle=True`` only
    because ``meta_json`` and ``bone_names`` are object arrays;
    numeric arrays are validated by explicit shape / dtype checks.
    """
    npz_path = raw_root / LAYER_B1_PROCESSED_DATA_SUBDIR / f"{clip}.npz"
    if not npz_path.is_file():
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 processed-data NPZ "
            f"missing: {npz_path}. contract={CONTRACT} requires every "
            "listed clip's processed_data/*.npz to be present; no silent "
            "skip is allowed (docs/removal_policy.md §3-§4)."
        )
    with np.load(npz_path, allow_pickle=True) as data:
        required_keys = (
            "bone_ang_vel",
            "bone_rot6d",
            "bone_names",
            "parents",
            "FPS",
            "meta_json",
        )
        missing = [k for k in required_keys if k not in data.files]
        if missing:
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
                f"missing required keys: {missing!r}. Refusing silent "
                "fallback to raw JSON for pose features."
            )
        bone_ang_vel = np.asarray(data["bone_ang_vel"])
        bone_rot6d = np.asarray(data["bone_rot6d"])
        bone_names = np.asarray(data["bone_names"])
        parents = np.asarray(data["parents"])
        fps_arr = np.asarray(data["FPS"])
        meta_obj = data["meta_json"].item()

    if bone_ang_vel.dtype != np.float32:
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"bone_ang_vel dtype={bone_ang_vel.dtype} expected float32."
        )
    if bone_rot6d.dtype != np.float32:
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"bone_rot6d dtype={bone_rot6d.dtype} expected float32."
        )
    if parents.dtype != np.int32:
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"parents dtype={parents.dtype} expected int32."
        )
    if fps_arr.dtype != np.int32:
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"FPS dtype={fps_arr.dtype} expected int32."
        )
    if fps_arr.shape != ():
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"FPS shape={fps_arr.shape} expected scalar ()."
        )
    if bone_ang_vel.ndim != 3 or bone_ang_vel.shape[1:] != (
        LAYER_B1_BONE_COUNT,
        3,
    ):
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"bone_ang_vel shape={bone_ang_vel.shape} expected "
            f"(T, {LAYER_B1_BONE_COUNT}, 3)."
        )
    if bone_rot6d.ndim != 3 or bone_rot6d.shape[1:] != (
        LAYER_B1_BONE_COUNT,
        6,
    ):
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"bone_rot6d shape={bone_rot6d.shape} expected "
            f"(T, {LAYER_B1_BONE_COUNT}, 6)."
        )
    if bone_names.shape != (LAYER_B1_BONE_COUNT,):
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"bone_names shape={bone_names.shape} expected "
            f"({LAYER_B1_BONE_COUNT},)."
        )
    if parents.shape != (LAYER_B1_BONE_COUNT,):
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"parents shape={parents.shape} expected "
            f"({LAYER_B1_BONE_COUNT},)."
        )
    if bone_ang_vel.shape[0] != bone_rot6d.shape[0]:
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"frame counts disagree: bone_ang_vel T={bone_ang_vel.shape[0]} "
            f"vs bone_rot6d T={bone_rot6d.shape[0]}."
        )
    t_npz = int(bone_ang_vel.shape[0])
    if t_npz != int(json_frame_count):
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"T={t_npz} disagrees with raw JSON len(Frames)="
            f"{int(json_frame_count)} for clip {clip!r}."
        )
    if t_npz < MIN_FRAMES_FOR_PHASE_ESTIMABILITY:
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"T={t_npz} below MIN_FRAMES_FOR_PHASE_ESTIMABILITY="
            f"{MIN_FRAMES_FOR_PHASE_ESTIMABILITY}; refusing to extract."
        )
    npz_fps = int(fps_arr)
    if npz_fps != int(round(float(json_fps))):
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"FPS={npz_fps} disagrees with raw JSON FPS={float(json_fps)} "
            f"for clip {clip!r}."
        )
    if not np.all(np.isfinite(bone_ang_vel)):
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            "bone_ang_vel contains non-finite values."
        )
    if not np.all(np.isfinite(bone_rot6d)):
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            "bone_rot6d contains non-finite values."
        )

    if isinstance(meta_obj, (bytes, bytearray)):
        meta_obj = meta_obj.decode("utf-8")
    if isinstance(meta_obj, str):
        try:
            meta_obj = json.loads(meta_obj)
        except json.JSONDecodeError as exc:
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
                f"meta_json string is not valid JSON: {exc}"
            ) from exc
    if not isinstance(meta_obj, dict):
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"meta_json is not a dict (got {type(meta_obj).__name__})."
        )
    units = meta_obj.get("units")
    if units != "meters":
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"meta_json.units={units!r} expected 'meters'."
        )
    spaces = meta_obj.get("spaces", {})
    if not isinstance(spaces, dict):
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"meta_json.spaces is not a dict (got "
            f"{type(spaces).__name__})."
        )
    bone_rot_space = spaces.get("bone_rotations")
    if bone_rot_space != "local_6d":
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"meta_json.spaces.bone_rotations={bone_rot_space!r} expected "
            "'local_6d'. Layer B.1 refuses silent space conversion."
        )
    bone_ang_space = spaces.get("bone_angular_velocities")
    if bone_ang_space != "local_rad_per_sec":
        raise FailFastError(
            f"[walk_f_causal_state_probe] Layer B.1 NPZ {npz_path} "
            f"meta_json.spaces.bone_angular_velocities="
            f"{bone_ang_space!r} expected 'local_rad_per_sec'. Layer B.1 "
            "refuses silent unit conversion."
        )

    bone_names_list = [str(b) for b in bone_names.tolist()]
    return {
        "clip": clip,
        "npz_path": str(npz_path),
        "bone_ang_vel_f64": bone_ang_vel.astype(np.float64),
        "bone_rot6d_f64": bone_rot6d.astype(np.float64),
        "bone_names": bone_names_list,
        "parents": parents.astype(np.int64).tolist(),
        "fps_npz": npz_fps,
        "fps_used": float(json_fps),
        "frame_count_npz": t_npz,
        "meta_json_units": units,
        "meta_json_spaces_bone_rotations": bone_rot_space,
        "meta_json_spaces_bone_angular_velocities": bone_ang_space,
    }


def _layer_b1_build_pose_dyn_matrix(
    npz_payload: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    arr = npz_payload["bone_ang_vel_f64"]
    t_npz = arr.shape[0]
    matrix = arr.reshape(t_npz, LAYER_B1_BONE_COUNT * 3)
    channels: list[str] = []
    for bone in npz_payload["bone_names"]:
        for axis in LAYER_B1_POSE_DYN_AXES:
            channels.append(f"bone_ang_vel.{bone}.{axis}")
    if len(channels) != matrix.shape[1]:
        raise FailFastError(
            "[walk_f_causal_state_probe] Layer B.1 pose_dyn channel count "
            f"{len(channels)} != matrix C={matrix.shape[1]}."
        )
    return matrix, channels


def _layer_b1_build_pose_rel_matrix(
    npz_payload: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    arr = npz_payload["bone_rot6d_f64"]
    t_npz = arr.shape[0]
    flat = arr.reshape(t_npz, LAYER_B1_BONE_COUNT * 6)
    fps = float(npz_payload["fps_used"])
    matrix = np.zeros((t_npz, LAYER_B1_BONE_COUNT * 6), dtype=np.float64)
    if t_npz >= 2:
        matrix[1:] = (flat[1:] - flat[:-1]) * fps
    channels: list[str] = []
    for bone in npz_payload["bone_names"]:
        for comp in LAYER_B1_POSE_REL_COMPONENT_LABELS:
            channels.append(f"bone_rot6d_dot.{bone}.{comp}")
    if len(channels) != matrix.shape[1]:
        raise FailFastError(
            "[walk_f_causal_state_probe] Layer B.1 pose_rel channel count "
            f"{len(channels)} != matrix C={matrix.shape[1]}."
        )
    return matrix, channels


def _layer_b1_build_group_matrix(
    group_name: str,
    npz_payload: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    _layer_b1_forbid_turn_dyn(group_name, "_layer_b1_build_group_matrix")
    if group_name == "pose_dyn":
        return _layer_b1_build_pose_dyn_matrix(npz_payload)
    if group_name == "pose_rel":
        return _layer_b1_build_pose_rel_matrix(npz_payload)
    raise FailFastError(
        f"[walk_f_causal_state_probe] Layer B.1 unknown feature group "
        f"{group_name!r}; supported={list(LAYER_B1_FEATURE_GROUPS.keys())}."
    )


def _layer_b1_per_channel_walk_f_scale(
    matrix: np.ndarray,
    channels: list[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for j, ch in enumerate(channels):
        out[ch] = _compute_channel_scale(matrix[:, j])
    return out


def _layer_b1_walk_f_group_reference_scale(
    group_name: str,
    group_def: dict[str, Any],
    matrix: np.ndarray,
    channels: list[str],
    n_frames: int,
) -> dict[str, Any]:
    """Walk_F-only per-group reference scale + degeneracy + phase status.

    Channel-level degeneracy uses ``mad <= epsilon_mad`` (upper-
    inclusive, matches Layer B at line 1367). Group-level degeneracy
    is True iff every channel is degenerate. ``phase_structure_status``
    is the contract §6 enum and is computed via
    ``_classify_phase_structure_status`` (shared with Layer B).
    """
    epsilon = float(group_def["epsilon_mad"])
    channel_summaries: dict[str, Any] = {}
    max_mad = 0.0
    any_not_all_finite = False
    all_channel_degenerate = True
    channel_degenerate_count = 0
    per_channel_scale = _layer_b1_per_channel_walk_f_scale(matrix, channels)
    for ch in channels:
        stats = per_channel_scale[ch]
        channel_mad = stats["mad"]
        channel_degenerate = bool(
            channel_mad is not None and channel_mad <= epsilon
        )
        if channel_degenerate:
            channel_degenerate_count += 1
        if channel_mad is None:
            all_channel_degenerate = False
        elif not channel_degenerate:
            all_channel_degenerate = False
        if channel_mad is not None and channel_mad > max_mad:
            max_mad = float(channel_mad)
        if not stats["all_finite"]:
            any_not_all_finite = True
        channel_summaries[ch] = {
            **stats,
            "epsilon_mad_estimator_only": epsilon,
            "channel_degenerate_by_epsilon_mad": channel_degenerate,
        }
    group_degenerate = bool(all_channel_degenerate)
    phase_status, layer_c_candidate = _classify_phase_structure_status(
        n_frames=n_frames,
        group_all_finite=not any_not_all_finite,
        group_degenerate=group_degenerate,
    )
    return {
        "group": group_name,
        "unit_label": group_def["unit_label"],
        "source": group_def["source"],
        "source_npz_key": group_def["source_npz_key"],
        "extracted_matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "extracted_matrix_dtype": str(matrix.dtype),
        "extracted_matrix_device": "cpu",
        "group_ablation_role": group_def["group_ablation_role"],
        "epsilon_mad_estimator_only": epsilon,
        "channels": channel_summaries,
        "group_max_channel_mad": float(max_mad),
        "group_reference_degenerate": group_degenerate,
        "channel_degenerate_count": int(channel_degenerate_count),
        "channel_total_count": int(len(channels)),
        "phase_structure_status": phase_status,
        "phase_structure_status_enum_source": (
            "docs/aperiodic_transition/"
            "2026-05-22_walk_f_causal_state_scaffold_v1.md:255"
        ),
        "layer_c_candidate": bool(layer_c_candidate),
        "phase_estimable_candidate": bool(layer_c_candidate),
        "phase_structure_layer_b1_definition": (
            "Layer B.1 emits only {phase_degenerate, insufficient_evidence}. "
            "phase_structured requires a predictive-loss comparison vs a "
            "phase-agnostic baseline (scaffold v1 §3.4) and is out of scope. "
            "Non-degenerate groups get insufficient_evidence with "
            "layer_c_candidate=True."
        ),
    }


def _layer_b1_query_channel_walk_f_projected(
    query_value: np.ndarray,
    walk_f_channel_scale: dict[str, Any],
    epsilon_mad: float,
    *,
    scale_source: str,
) -> dict[str, Any]:
    """Per-channel query inspection under Walk_F-projected z-statistics.

    The denominator is ``max(walk_f_robust_std_from_mad, epsilon_mad)``
    so a degenerate Walk_F channel does not blow up to infinite
    z-scores. Caller skips this when Walk_F's group is degenerate
    (see _process_clip_pose_reference_scale_check).

    Layer B.1 contract §1 rule 2: ``scale_source`` MUST equal
    ``"walk_f"``. Any other value raises ``FailFastError`` via
    ``_layer_b1_forbid_query_self_rescale``. This is the only
    normalization entry point on the Layer B.1 query path; the
    sentinel is now load-bearing rather than advisory.
    """
    if scale_source != LAYER_B1_ALLOWED_SCALE_SOURCE:
        _layer_b1_forbid_query_self_rescale(
            "_layer_b1_query_channel_walk_f_projected",
            scale_source=scale_source,
        )
    median_f = walk_f_channel_scale.get("median")
    robust_std = walk_f_channel_scale.get("robust_std_from_mad")
    if median_f is None or robust_std is None:
        return {
            "walk_f_projected_z_median": None,
            "walk_f_projected_z_mad": None,
            "walk_f_projected_z_p05": None,
            "walk_f_projected_z_p50": None,
            "walk_f_projected_z_p95": None,
            "walk_f_projected_z_max_abs": None,
            "n_values": int(query_value.size),
            "n_finite": int(np.sum(np.isfinite(query_value))),
            "all_finite": bool(np.all(np.isfinite(query_value))),
            "denominator_used": None,
            "denominator_floor_epsilon_mad": float(epsilon_mad),
            "walk_f_median": None,
            "walk_f_robust_std_from_mad": None,
        }
    denom = float(max(float(robust_std), float(epsilon_mad)))
    finite_mask = np.isfinite(query_value)
    if int(finite_mask.sum()) == 0:
        return {
            "walk_f_projected_z_median": None,
            "walk_f_projected_z_mad": None,
            "walk_f_projected_z_p05": None,
            "walk_f_projected_z_p50": None,
            "walk_f_projected_z_p95": None,
            "walk_f_projected_z_max_abs": None,
            "n_values": int(query_value.size),
            "n_finite": 0,
            "all_finite": False,
            "denominator_used": denom,
            "denominator_floor_epsilon_mad": float(epsilon_mad),
            "walk_f_median": float(median_f),
            "walk_f_robust_std_from_mad": float(robust_std),
        }
    z = (query_value[finite_mask] - float(median_f)) / denom
    z_median = float(np.median(z))
    z_mad = float(np.median(np.abs(z - z_median)))
    return {
        "walk_f_projected_z_median": z_median,
        "walk_f_projected_z_mad": z_mad,
        "walk_f_projected_z_p05": float(np.quantile(z, 0.05)),
        "walk_f_projected_z_p50": float(np.quantile(z, 0.50)),
        "walk_f_projected_z_p95": float(np.quantile(z, 0.95)),
        "walk_f_projected_z_max_abs": float(np.max(np.abs(z))),
        "n_values": int(query_value.size),
        "n_finite": int(finite_mask.sum()),
        "all_finite": bool(int(finite_mask.sum()) == int(query_value.size)),
        "denominator_used": denom,
        "denominator_floor_epsilon_mad": float(epsilon_mad),
        "walk_f_median": float(median_f),
        "walk_f_robust_std_from_mad": float(robust_std),
    }


def _layer_b1_query_group_inspection(
    *,
    group_name: str,
    group_def: dict[str, Any],
    query_matrix: np.ndarray,
    query_channels: list[str],
    walk_f_reference_block: dict[str, Any],
) -> dict[str, Any]:
    """Build per-channel query inspection block for one group.

    Inspection has two layers:
      * ``walk_f_projected_z_summary``: per-channel z under Walk_F
        median + MAD (the only reference scale).
      * ``query_self_scale_inspection_only``: per-channel raw scale
        on the query clip itself, clearly labelled as inspection-only
        and never aggregated. Computing it via _compute_channel_scale
        is NOT a "rescale" because nothing downstream divides by it;
        the §1 rule 2 guardrail is enforced at the projection-helper
        level (no code path divides by query MAD).

    Returns a single dict matching the contract §7 per-clip-per-group
    block.
    """
    _layer_b1_forbid_turn_dyn(group_name, "_layer_b1_query_group_inspection")
    epsilon = float(group_def["epsilon_mad"])
    walk_f_channels = walk_f_reference_block["channels"]
    walk_f_group_degenerate = bool(
        walk_f_reference_block["group_reference_degenerate"]
    )
    phase_status = (
        "phase_degenerate"
        if walk_f_group_degenerate
        else "insufficient_evidence"
    )
    channel_blocks: dict[str, Any] = {}
    for j, ch in enumerate(query_channels):
        query_col = query_matrix[:, j]
        if walk_f_group_degenerate:
            walk_f_projected = {
                "walk_f_projected_z_median": None,
                "walk_f_projected_z_mad": None,
                "walk_f_projected_z_p05": None,
                "walk_f_projected_z_p50": None,
                "walk_f_projected_z_p95": None,
                "walk_f_projected_z_max_abs": None,
                "n_values": int(query_col.size),
                "n_finite": int(np.sum(np.isfinite(query_col))),
                "all_finite": bool(np.all(np.isfinite(query_col))),
                "denominator_used": None,
                "denominator_floor_epsilon_mad": float(epsilon),
                "walk_f_median": walk_f_channels[ch].get("median"),
                "walk_f_robust_std_from_mad": walk_f_channels[ch].get(
                    "robust_std_from_mad"
                ),
                "skipped_reason": (
                    "walk_f_group_reference_degenerate_skip_z_projection"
                ),
            }
        else:
            walk_f_projected = _layer_b1_query_channel_walk_f_projected(
                query_col,
                walk_f_channels[ch],
                epsilon,
                scale_source=LAYER_B1_ALLOWED_SCALE_SOURCE,
            )
        query_self = _compute_channel_scale(query_col)
        channel_blocks[ch] = {
            "walk_f_projected_z_summary": walk_f_projected,
            "query_self_scale_inspection_only": {
                **query_self,
                "inspection_only_note": (
                    "Per-channel raw scale for the query clip is "
                    "inspection-only. It MUST NOT be aggregated into "
                    "the Walk_F reference scale "
                    "(Layer B.1 contract §1 rule 2)."
                ),
            },
        }
    return {
        "group": group_name,
        "unit_label": group_def["unit_label"],
        "source": group_def["source"],
        "source_npz_key": group_def["source_npz_key"],
        "extracted_matrix_shape": [
            int(query_matrix.shape[0]),
            int(query_matrix.shape[1]),
        ],
        "extracted_matrix_dtype": str(query_matrix.dtype),
        "extracted_matrix_device": "cpu",
        "group_ablation_role": group_def["group_ablation_role"],
        "epsilon_mad_estimator_only": epsilon,
        "channels": channel_blocks,
        "reference_degenerate": walk_f_group_degenerate,
        "reference_degenerate_source": (
            "walk_f_per_group_reference_degenerate_inherited"
        ),
        "phase_structure_status": phase_status,
        "phase_structure_status_enum_source": (
            "docs/aperiodic_transition/"
            "2026-05-22_walk_f_causal_state_scaffold_v1.md:255"
        ),
        "layer_c_candidate": bool(
            not walk_f_group_degenerate
        ),
        "boundary_detection_status": (
            "not_emitted_by_this_tool_layer_b1_reference_scale_only"
        ),
    }


def _layer_b1_processed_data_schema_block(
    npz_payload: dict[str, Any],
) -> dict[str, Any]:
    bone_ang_vel = npz_payload["bone_ang_vel_f64"]
    bone_rot6d = npz_payload["bone_rot6d_f64"]
    t_npz = int(bone_ang_vel.shape[0])
    return {
        "npz_path": npz_payload["npz_path"],
        "arrays": {
            "bone_ang_vel": {
                "key": "bone_ang_vel",
                "rank": 3,
                "shape": [
                    t_npz,
                    LAYER_B1_BONE_COUNT,
                    3,
                ],
                "dtype_on_disk": "float32",
                "dtype_used": "float64",
                "device": "cpu",
                "finite_coverage": {
                    "n_total": int(bone_ang_vel.size),
                    "n_finite": int(np.sum(np.isfinite(bone_ang_vel))),
                    "all_finite": bool(
                        np.all(np.isfinite(bone_ang_vel))
                    ),
                },
            },
            "bone_rot6d": {
                "key": "bone_rot6d",
                "rank": 3,
                "shape": [
                    t_npz,
                    LAYER_B1_BONE_COUNT,
                    6,
                ],
                "dtype_on_disk": "float32",
                "dtype_used": "float64",
                "device": "cpu",
                "finite_coverage": {
                    "n_total": int(bone_rot6d.size),
                    "n_finite": int(np.sum(np.isfinite(bone_rot6d))),
                    "all_finite": bool(np.all(np.isfinite(bone_rot6d))),
                },
            },
            "bone_names": {
                "key": "bone_names",
                "rank": 1,
                "shape": [LAYER_B1_BONE_COUNT],
                "dtype_on_disk": "object",
                "dtype_used": "python_str_list",
                "device": "cpu",
            },
            "parents": {
                "key": "parents",
                "rank": 1,
                "shape": [LAYER_B1_BONE_COUNT],
                "dtype_on_disk": "int32",
                "dtype_used": "python_int_list",
                "device": "cpu",
            },
            "FPS": {
                "key": "FPS",
                "rank": 0,
                "shape": [],
                "dtype_on_disk": "int32",
                "dtype_used": "python_int_scalar",
                "device": "cpu",
            },
        },
        "meta_json_units": npz_payload["meta_json_units"],
        "meta_json_spaces_bone_rotations": npz_payload[
            "meta_json_spaces_bone_rotations"
        ],
        "meta_json_spaces_bone_angular_velocities": npz_payload[
            "meta_json_spaces_bone_angular_velocities"
        ],
        "meta_json_fps": npz_payload["fps_npz"],
        "bone_names_count": int(len(npz_payload["bone_names"])),
        "parents_count": int(len(npz_payload["parents"])),
    }


def _process_clip_pose_reference_scale_check(
    *,
    clip_info: dict[str, Any],
    npz_payload: dict[str, Any],
    is_reference: bool,
    walk_f_reference_scale_per_group: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build per-clip Layer B.1 audit block.

    Reference clips compute Walk_F's per-channel scale + degeneracy
    (the authoritative reference). Query clips reuse Walk_F's
    reference block and produce per-channel Walk_F-projected z
    summaries; their own per-channel raw scale is reported as
    inspection-only and NEVER aggregated.
    """
    t_npz = int(npz_payload["frame_count_npz"])
    schema_block = _layer_b1_processed_data_schema_block(npz_payload)
    feature_groups_block: dict[str, Any] = {}
    for group_name, group_def in LAYER_B1_FEATURE_GROUPS.items():
        matrix, channels = _layer_b1_build_group_matrix(
            group_name, npz_payload
        )
        if is_reference:
            feature_groups_block[group_name] = (
                _layer_b1_walk_f_group_reference_scale(
                    group_name=group_name,
                    group_def=group_def,
                    matrix=matrix,
                    channels=channels,
                    n_frames=t_npz,
                )
            )
        else:
            if walk_f_reference_scale_per_group is None:
                raise FailFastError(
                    "[walk_f_causal_state_probe] Layer B.1 query clip "
                    f"{clip_info['clip']!r} processed before Walk_F "
                    "reference scale was computed; refusing to emit "
                    "query inspection without a reference scale."
                )
            walk_f_block = walk_f_reference_scale_per_group[group_name]
            feature_groups_block[group_name] = (
                _layer_b1_query_group_inspection(
                    group_name=group_name,
                    group_def=group_def,
                    query_matrix=matrix,
                    query_channels=channels,
                    walk_f_reference_block=walk_f_block,
                )
            )
    return {
        "clip": clip_info["clip"],
        "raw_json_path": clip_info["raw_json_path"],
        "npz_path": npz_payload["npz_path"],
        "fps": clip_info["fps"],
        "frame_count": int(clip_info["frame_count"]),
        "track": TRACK_BY_MODE["pose_reference_scale_check"],
        "track_role": TRACK_ROLE_BY_MODE["pose_reference_scale_check"],
        "clip_role": (
            "reference_clip" if is_reference else "query_clip_not_reference"
        ),
        "contributes_to_reference_scale": bool(is_reference),
        "processed_data_schema": schema_block,
        "frame_alignment": {
            "json_frame_count": int(clip_info["frame_count"]),
            "npz_bone_ang_vel_T": t_npz,
            "npz_bone_rot6d_T": t_npz,
            "aligned": True,
        },
        "feature_groups_layer_b1": feature_groups_block,
        "excluded_feature_groups_layer_b1": LAYER_B1_EXCLUDED_FEATURE_GROUPS,
        "attractor_membership_status": "not_emitted_by_this_tool",
        "event_head_target_status": "not_emitted_by_this_tool",
        "handoff_ready_status": "not_emitted_by_this_tool",
        "transition_done_status": "not_emitted_by_this_tool",
        "cross_attractor_claim_status": "forbidden_by_contract",
        "transition_truth_promotion": "forbidden_by_contract",
        "combined_membership_score_status": "forbidden_by_contract",
    }


# -----------------------------------------------------------------------
# Artifact writers
# -----------------------------------------------------------------------


def _write_per_clip(out_dir: Path, per_clip: list[dict[str, Any]], suffix: str) -> list[str]:
    written: list[str] = []
    for entry in per_clip:
        path = out_dir / f"{entry['clip']}_{suffix}.json"
        path.write_text(json.dumps(_serializable(entry), indent=2), encoding="utf-8")
        written.append(str(path))
    return written


def _common_summary_root(
    mode: str,
    raw_root: Path,
    clips: list[str],
    per_clip_paths: list[str],
) -> dict[str, Any]:
    track = TRACK_BY_MODE[mode]
    return {
        "tool": TOOL_NAME,
        "tool_mode": "read_only",
        "contract": CONTRACT,
        "mode": mode,
        "track": track,
        "track_role": TRACK_ROLE_BY_MODE[mode],
        "reference_family": REFERENCE_FAMILY,
        "scope": SCOPE_BY_MODE[mode],
        "raw_root": str(raw_root),
        "clips": list(clips),
        "per_clip_artifact_paths": per_clip_paths,
    }


def _build_summary_yaw_debug(
    raw_root: Path,
    clips: list[str],
    per_clip: list[dict[str, Any]],
    per_clip_paths: list[str],
) -> dict[str, Any]:
    insufficient: list[dict[str, Any]] = []
    for entry in per_clip:
        st = entry["window_stability_summary"]["status"]
        if st == "INSUFFICIENT_EVIDENCE":
            insufficient.append(
                {
                    "clip": entry["clip"],
                    "where": "yaw_activity_debug.window_stability_summary.status",
                    "status": st,
                    "reason": (
                        "yaw activity window endpoints disagree across the "
                        "threshold grid or only part of the grid found "
                        "activity; do not promote any single threshold as "
                        "transition truth."
                    ),
                }
            )

    insufficient.extend(
        [
            {
                "clip": None,
                "where": "causal_state_track_status",
                "status": "not_implemented_layer0",
                "reason": (
                    "Layer 0 / Track 1 only implements yaw_activity_debug. "
                    "Predictive-equivalence membership relative to Walk_F "
                    "(walk_f_causal_state_scaffold §5.1) is not implemented "
                    "by this tool."
                ),
            },
            {
                "clip": None,
                "where": "quotient_definition_status",
                "status": "not_run_layer0",
                "reason": (
                    "Planar translation / global-yaw quotient and yaw "
                    "invariance sanity (§1.2) are not implemented at "
                    "Layer 0. Yaw activity here is computed on absolute "
                    "RootYaw radians as a descriptive debug oracle only. "
                    "Run --mode gauge_check for Layer A."
                ),
            },
            {
                "clip": None,
                "where": "yaw_invariance_sanity",
                "status": "not_run_layer0",
                "reason": (
                    "§1.2 yaw-invariance grid not run by yaw_debug; see "
                    "--mode gauge_check (Layer A)."
                ),
            },
            {
                "clip": None,
                "where": "phase_library / leave_one_phase_baseline",
                "status": "not_implemented_layer0",
                "reason": (
                    "No Walk_F phase candidates, no leave-one-phase "
                    "baseline, no predictive loss are computed by Layer 0 "
                    "or Layer A (§5.3)."
                ),
            },
            {
                "clip": None,
                "where": "transition_truth_promotion",
                "status": "forbidden_by_contract",
                "reason": (
                    "yaw_activity_debug is debug-only and must not be "
                    "promoted to general transition truth, EventHead "
                    "target, handoff_ready, or transition_done "
                    "(§5.1, §7)."
                ),
            },
        ]
    )

    estimation_grid = {
        "absolute_deg_per_s": list(ABS_DEG_PER_S_GRID),
        "relative_to_peak_abs_yaw_rate": list(REL_TO_PEAK_GRID),
        "note": (
            "Thresholds are estimator-level only and are NOT contract "
            "thresholds. A probe setting can only support a conclusion if "
            "nearby settings give the same qualitative result (see "
            "walk_f_causal_state_scaffold §3.2)."
        ),
    }

    definition_layer = {
        "attractor_definition_status": "not_implemented_layer0",
        "causal_state_definition_status": "not_implemented_layer0",
        "current_reference_family": REFERENCE_FAMILY,
        "membership_evidence_status": "not_implemented_layer0",
        "scope_caveat": (
            "Walk_F is a single trajectory; class-level attractor claims "
            "remain INSUFFICIENT_EVIDENCE on current data (§4)."
        ),
    }

    per_clip_summary: list[dict[str, Any]] = []
    for entry in per_clip:
        per_clip_summary.append(
            {
                "clip": entry["clip"],
                "raw_json_path": entry["raw_json_path"],
                "fps": entry["fps"],
                "frame_count": entry["frame_count"],
                "total_yaw_delta_deg": entry["summary"]["total_yaw_delta_deg"],
                "peak_abs_yaw_rate_deg_per_s": entry["summary"]["peak_abs_yaw_rate_deg_per_s"],
                "peak_abs_yaw_rate_frame": entry["summary"]["peak_abs_yaw_rate_frame"],
                "turn_direction_by_yaw": entry["summary"]["turn_direction_by_yaw"],
                "window_stability_status": entry["window_stability_summary"]["status"],
                "window_start_frame_min": entry["window_stability_summary"]["start_frame_min"],
                "window_start_frame_max": entry["window_stability_summary"]["start_frame_max"],
                "window_end_frame_min": entry["window_stability_summary"]["end_frame_min"],
                "window_end_frame_max": entry["window_stability_summary"]["end_frame_max"],
                "any_setting_left_censored_start": any(
                    w["left_censored_start"]
                    for w in entry["yaw_activity_windows"]
                    if w["activity_start_frame"] is not None
                ),
                "any_setting_right_censored_end": any(
                    w["right_censored_end"]
                    for w in entry["yaw_activity_windows"]
                    if w["activity_end_frame"] is not None
                ),
                "yaw_activity_windows": entry["yaw_activity_windows"],
            }
        )

    summary = _common_summary_root("yaw_debug", raw_root, clips, per_clip_paths)
    summary.update(
        {
            "definition_layer": definition_layer,
            "estimation_grid": estimation_grid,
            "yaw_activity_debug": {"per_clip": per_clip_summary},
            "feature_groups": "not_implemented_layer0",
            "quotient_definition": "not_run_layer0",
            "walk_f_baseline": "not_implemented_layer0",
            "per_clip": per_clip_summary,
            "sensitivity_summary": {
                "yaw_activity_threshold_grid_status": "reported_per_clip",
                "predictive_loss_sensitivity_status": "not_implemented_layer0",
                "rotation_grid_sensitivity_status": "not_run_layer0",
            },
            "causal_state_track_status": "not_implemented_layer0",
            "quotient_definition_status": "not_run_layer0",
            "yaw_invariance_sanity": {"status": "not_run_layer0"},
            "attractor_membership_status": "not_implemented_layer0",
            "phase_library_status": "not_implemented_layer0",
            "predictive_loss_status": "not_implemented_layer0",
            "event_head_target_status": "not_emitted_by_this_tool",
            "handoff_ready_status": "not_emitted_by_this_tool",
            "transition_done_status": "not_emitted_by_this_tool",
            "transition_truth_promotion": "forbidden_by_contract",
            "insufficient_evidence": insufficient,
            "notes": [
                "yaw_activity_debug is a Walk-turn descriptive oracle on "
                "raw RootYaw and must not be promoted to attractor "
                "definition or transition truth.",
                "Threshold grid values are estimator-level and not "
                "contract thresholds; see estimation_grid.note.",
                "Layer 0 / Track 1 does not implement causal_state "
                "membership, phase library, leave-one-phase baseline, "
                "predictive loss, or the §1.2 yaw-invariance / quotient "
                "sanity check.",
            ],
        }
    )
    return summary


def _build_summary_gauge_check(
    raw_root: Path,
    clips: list[str],
    per_clip: list[dict[str, Any]],
    per_clip_paths: list[str],
) -> dict[str, Any]:
    quotient_def = _build_quotient_definition_block()
    yaw_sanity = _build_yaw_invariance_sanity(per_clip)
    translation_sanity = _build_translation_invariance_sanity(per_clip)
    se2_sanity = _build_se2_gauge_sanity(yaw_sanity, translation_sanity, per_clip)

    rot_max_err_overall: dict[str, float] = {}
    trans_max_err_overall: dict[str, float] = {}
    se2_max_err_overall: dict[str, float] = {}
    for entry in per_clip:
        for k, v in entry["max_rotation_abs_error_by_feature_group"].items():
            rot_max_err_overall[k] = max(rot_max_err_overall.get(k, 0.0), float(v))
        for k, v in entry["max_translation_abs_error_by_feature_group"].items():
            trans_max_err_overall[k] = max(trans_max_err_overall.get(k, 0.0), float(v))
        for k, v in entry["max_abs_error_by_feature_group_se2"].items():
            se2_max_err_overall[k] = max(se2_max_err_overall.get(k, 0.0), float(v))

    rot_total_compared = sum(int(e["rotation_finite_coverage"]["n_values_compared"]) for e in per_clip)
    rot_total_finite = sum(int(e["rotation_finite_coverage"]["n_finite_min_per_pair"]) for e in per_clip)
    rot_finite_coverage_root = {
        "n_values_compared": int(rot_total_compared),
        "n_finite_min_per_pair": int(rot_total_finite),
        "coverage_fraction": float(rot_total_finite) / float(rot_total_compared) if rot_total_compared else 0.0,
    }
    trans_total_compared = sum(int(e["translation_finite_coverage"]["n_values_compared"]) for e in per_clip)
    trans_total_finite = sum(int(e["translation_finite_coverage"]["n_finite_min_per_pair"]) for e in per_clip)
    trans_finite_coverage_root = {
        "n_values_compared": int(trans_total_compared),
        "n_finite_min_per_pair": int(trans_total_finite),
        "coverage_fraction": float(trans_total_finite) / float(trans_total_compared) if trans_total_compared else 0.0,
    }
    se2_finite_coverage_root = {
        "n_values_compared": int(rot_total_compared + trans_total_compared),
        "n_finite_min_per_pair": int(rot_total_finite + trans_total_finite),
        "coverage_fraction": (
            float(rot_total_finite + trans_total_finite)
            / float(rot_total_compared + trans_total_compared)
            if (rot_total_compared + trans_total_compared)
            else 0.0
        ),
    }

    per_clip_gauge_error: list[dict[str, Any]] = []
    wrap_cases_root: list[dict[str, Any]] = []
    per_clip_summary: list[dict[str, Any]] = []
    for entry in per_clip:
        per_clip_gauge_error.append(
            {
                "clip": entry["clip"],
                "by_rotation": entry["per_rotation"],
                "by_translation": entry["per_translation"],
                "max_rotation_abs_error_by_feature_group": entry["max_rotation_abs_error_by_feature_group"],
                "max_translation_abs_error_by_feature_group": entry["max_translation_abs_error_by_feature_group"],
                "max_abs_error_over_rotation_grid": entry["max_abs_error_over_rotation_grid"],
                "max_abs_error_over_translation_grid": entry["max_abs_error_over_translation_grid"],
                "max_abs_error_over_se2_grid": entry["max_abs_error_over_se2_grid"],
                "rotation_finite_coverage": entry["rotation_finite_coverage"],
                "translation_finite_coverage": entry["translation_finite_coverage"],
            }
        )
        for wc in entry["wrap_boundary_cases"]:
            wrap_cases_root.append({"clip": entry["clip"], **wc})
        per_clip_summary.append(
            {
                "clip": entry["clip"],
                "raw_json_path": entry["raw_json_path"],
                "fps": entry["fps"],
                "frame_count": entry["frame_count"],
                "max_abs_error_over_rotation_grid": entry["max_abs_error_over_rotation_grid"],
                "max_abs_error_over_translation_grid": entry["max_abs_error_over_translation_grid"],
                "max_abs_error_over_se2_grid": entry["max_abs_error_over_se2_grid"],
                "max_rotation_abs_error_by_feature_group": entry["max_rotation_abs_error_by_feature_group"],
                "max_translation_abs_error_by_feature_group": entry["max_translation_abs_error_by_feature_group"],
                "max_abs_error_by_feature_group_se2": entry["max_abs_error_by_feature_group_se2"],
                "rotation_finite_coverage": entry["rotation_finite_coverage"],
                "translation_finite_coverage": entry["translation_finite_coverage"],
            }
        )

    insufficient: list[dict[str, Any]] = []
    if yaw_sanity["status"] == "INSUFFICIENT_EVIDENCE":
        insufficient.append(
            {
                "clip": None,
                "where": "yaw_invariance_sanity.status",
                "status": "INSUFFICIENT_EVIDENCE",
                "reason": yaw_sanity["rationale"],
            }
        )
    if translation_sanity["status"] == "INSUFFICIENT_EVIDENCE":
        insufficient.append(
            {
                "clip": None,
                "where": "translation_invariance_sanity.status",
                "status": "INSUFFICIENT_EVIDENCE",
                "reason": translation_sanity["rationale"],
            }
        )
    if se2_sanity["status"] == "INSUFFICIENT_EVIDENCE":
        insufficient.append(
            {
                "clip": None,
                "where": "se2_gauge_sanity.status",
                "status": "INSUFFICIENT_EVIDENCE",
                "reason": (
                    "Aggregate SE(2) quotient sanity inherits "
                    "INSUFFICIENT_EVIDENCE from at least one component."
                ),
            }
        )
    insufficient.extend(
        [
            {
                "clip": None,
                "where": "causal_state_track_status",
                "status": "not_implemented_layerA",
                "reason": (
                    "Layer A locks the canonical gauge and yaw-invariance "
                    "sanity. Predictive-equivalence membership / phase "
                    "library / leave-one-phase baseline / predictive loss "
                    "(§5.3) are deferred to a later layer."
                ),
            },
            {
                "clip": None,
                "where": "attractor_membership_status",
                "status": "not_implemented_layerA",
                "reason": (
                    "No membership curve is emitted by gauge_check. "
                    "Walk_F single-trajectory class-level attractor claims "
                    "remain INSUFFICIENT_EVIDENCE on current data (§4)."
                ),
            },
            {
                "clip": None,
                "where": "phase_library / leave_one_phase_baseline / predictive_loss",
                "status": "not_implemented_layerA",
                "reason": "Out of Layer A scope.",
            },
            {
                "clip": None,
                "where": "event_head_target_status",
                "status": "not_emitted_by_this_tool",
                "reason": (
                    "gauge_check is read-only and produces no EventHead "
                    "target, no handoff_ready, no transition_done (§7)."
                ),
            },
            {
                "clip": None,
                "where": "transition_truth_promotion",
                "status": "forbidden_by_contract",
                "reason": (
                    "yaw_activity_debug and any quantity derived here "
                    "must not be promoted to general transition truth."
                ),
            },
        ]
    )

    summary = _common_summary_root("gauge_check", raw_root, clips, per_clip_paths)
    summary.update(
        {
            "definition_layer": {
                "attractor_definition_status": "not_implemented_layerA",
                "causal_state_definition_status": "not_implemented_layerA",
                "current_reference_family": REFERENCE_FAMILY,
                "membership_evidence_status": "not_implemented_layerA",
                "scope_caveat": (
                    "Walk_F is a single trajectory; class-level attractor "
                    "claims remain INSUFFICIENT_EVIDENCE on current data "
                    "(§4). Layer A only verifies the gauge prerequisite."
                ),
            },
            "estimation_grid": {
                "rotation_grid_rad": [float(a) for a in ROTATION_GRID_RAD],
                "rotation_grid_deg": [float(math.degrees(a)) for a in ROTATION_GRID_RAD],
                "wrap_boundary_required_alphas_rad": [math.pi, -math.pi],
                "translation_grid_m": [list(b) for b in TRANSLATION_GRID_M],
                "tol_max_abs_error_estimator_only": GAUGE_MAX_ABS_ERR_TOL,
                "note": (
                    "Tolerance is reported numerically and is NOT contract "
                    "definition. Reviewers must judge against the per-clip "
                    "max/median errors in per_clip_gauge_error."
                ),
            },
            "gauge_kind": "SE(2)_quotient_sanity_gate_not_attractor_gate",
            "feature_groups": list(quotient_def["feature_groups_emitted"]),
            "quotient_definition": quotient_def,
            "rotation_grid_rad": [float(a) for a in ROTATION_GRID_RAD],
            "rotation_grid_deg": [float(math.degrees(a)) for a in ROTATION_GRID_RAD],
            "translation_grid_m": [list(b) for b in TRANSLATION_GRID_M],
            "wrap_boundary_cases": wrap_cases_root,
            "per_clip_gauge_error": per_clip_gauge_error,
            "max_rotation_abs_error_by_feature_group": {
                k: float(v) for k, v in rot_max_err_overall.items()
            },
            "max_translation_abs_error_by_feature_group": {
                k: float(v) for k, v in trans_max_err_overall.items()
            },
            "max_abs_error_by_feature_group_se2": {
                k: float(v) for k, v in se2_max_err_overall.items()
            },
            "max_abs_error_overall_rotation": float(yaw_sanity["max_abs_error_overall"]),
            "max_abs_error_overall_translation": float(translation_sanity["max_abs_error_overall"]),
            "max_abs_error_overall_se2": float(se2_sanity["max_abs_error_overall"]),
            "rotation_finite_coverage": rot_finite_coverage_root,
            "translation_finite_coverage": trans_finite_coverage_root,
            "se2_finite_coverage": se2_finite_coverage_root,
            "yaw_invariance_sanity": yaw_sanity,
            "translation_invariance_sanity": translation_sanity,
            "se2_gauge_sanity": se2_sanity,
            # Generic-named root fields. As of the Polish-2 pre-commit
            # cleanup these point to the SE(2) aggregate (rotation +
            # translation), NOT the yaw-only rotation view. The yaw-only
            # numbers live under *_rotation suffixes and in
            # yaw_invariance_sanity. Anyone reading "max_abs_error_overall"
            # should treat it as the SE(2) quotient sanity gate value.
            "max_abs_error_by_feature_group": {
                k: float(v) for k, v in se2_max_err_overall.items()
            },
            "max_abs_error_overall": float(se2_sanity["max_abs_error_overall"]),
            "finite_coverage": se2_finite_coverage_root,
            "max_abs_error_overall_alias_semantics": "se2_aggregate",
            "max_abs_error_by_feature_group_alias_semantics": "se2_aggregate",
            "finite_coverage_alias_semantics": "se2_aggregate",
            "walk_f_baseline": "not_implemented_layerA",
            "per_clip": per_clip_summary,
            "sensitivity_summary": {
                "rotation_grid_sensitivity_status": "reported_per_clip_per_rotation",
                "translation_grid_sensitivity_status": "reported_per_clip_per_translation",
                "predictive_loss_sensitivity_status": "not_implemented_layerA",
                "yaw_activity_threshold_grid_status": "not_emitted_in_this_mode",
            },
            "causal_state_track_status": "not_implemented_layerA",
            "quotient_definition_status": "locked_by_layerA",
            "attractor_membership_status": "not_implemented_layerA",
            "phase_library_status": "not_implemented_layerA",
            "predictive_loss_status": "not_implemented_layerA",
            "event_head_target_status": "not_emitted_by_this_tool",
            "handoff_ready_status": "not_emitted_by_this_tool",
            "transition_done_status": "not_emitted_by_this_tool",
            "transition_truth_promotion": "forbidden_by_contract",
            "insufficient_evidence": insufficient,
            "notes": [
                "Layer A locks an SE(2) quotient sanity gate: planar "
                "translation removal + global-yaw removal + +/-pi wrap "
                "boundary. It is NOT an attractor gate, NOT a membership "
                "claim, and does NOT estimate causal-state membership, "
                "phase library, predictive loss, or EventHead targets.",
                "Tolerance is estimator-level numeric; max/median errors "
                "are always reported so a reviewer can re-judge.",
                "yaw_activity_debug remains debug-only and is intentionally "
                "not re-emitted in this mode.",
                "Synthetic translations perturb root_pos_xy only; RootYaw "
                "and RootVelocityXY are intentionally NOT translated.",
            ],
        }
    )
    return summary


def _build_summary_reference_scale_check(
    raw_root: Path,
    clips: list[str],
    per_clip: list[dict[str, Any]],
    per_clip_paths: list[str],
) -> dict[str, Any]:
    """Layer B summary: reference scale and degeneracy on Walk_F only.

    Reference scale is computed exclusively from clips with
    ``clip_role == "reference_clip"``. Query (non-reference) clips
    are present in per_clip for inspection but DO NOT contribute to
    the reference scale or to the degeneracy / phase_structure
    decision. Layer B emits no membership, no phase library, no
    predictive loss.
    """
    reference_clips = [e for e in per_clip if e["contributes_to_reference_scale"]]
    query_clips = [e for e in per_clip if not e["contributes_to_reference_scale"]]

    # Reference-family aggregation. With v1 data, reference_family =
    # {Walk_F} contains exactly one clip; with future multi-take data,
    # aggregation across reference clips would happen here.
    if len(reference_clips) == 0:
        ref_status = "INSUFFICIENT_EVIDENCE"
        ref_status_reason = (
            "No reference-family clip was supplied. Layer B requires "
            "Walk_F in --clips to compute a reference scale; "
            "reference_family is locked to {Walk_F} by §1.1."
        )
        reference_per_group: dict[str, Any] = {}
    else:
        ref_status = "single_trajectory_reference_only"
        ref_status_reason = (
            "Walk_F is a single trajectory (§4); class-level recurrence "
            "is INSUFFICIENT_EVIDENCE. Layer B reports single-trajectory "
            "reference scale only."
        )
        # With more than one reference clip in the future, we would
        # aggregate per group (e.g. union of per-channel MADs). For now,
        # surface the single reference clip's per-group block verbatim
        # and refuse to silently average if multiple reference clips
        # were somehow supplied (defensive; reference_family is len==1).
        if len(reference_clips) > 1:
            raise FailFastError(
                "[walk_f_causal_state_probe] Layer B received "
                f"{len(reference_clips)} reference clips; reference_family is "
                f"locked to {REFERENCE_FAMILY} (length 1) by §1.1. Refusing "
                "to silently average across multiple reference clips."
            )
        reference_per_group = reference_clips[0]["feature_groups_layer_b"]

    feature_groups_layer_b_meta = {
        name: {
            "channels": list(group_def["channels"]),
            "unit_label": group_def["unit_label"],
            "epsilon_mad_estimator_only": group_def["epsilon_mad"],
            "source": group_def["source"],
        }
        for name, group_def in FEATURE_GROUPS_LAYER_B.items()
    }

    # Summarise the contract §6 phase_structure_status enum per group,
    # plus the Layer-B-only layer_c_candidate flag.
    phase_status_summary: dict[str, Any] = {}
    for name in FEATURE_GROUPS_LAYER_B:
        if name in reference_per_group:
            phase_status_summary[name] = {
                "phase_structure_status": reference_per_group[name][
                    "phase_structure_status"
                ],
                "layer_c_candidate": reference_per_group[name]["layer_c_candidate"],
                "phase_estimable_candidate": reference_per_group[name][
                    "phase_estimable_candidate"
                ],
                "group_reference_degenerate": reference_per_group[name][
                    "group_reference_degenerate"
                ],
                "group_max_channel_mad": reference_per_group[name][
                    "group_max_channel_mad"
                ],
                "epsilon_mad_estimator_only": reference_per_group[name][
                    "epsilon_mad_estimator_only"
                ],
                "channel_degenerate_count": reference_per_group[name][
                    "channel_degenerate_count"
                ],
                "channel_total_count": reference_per_group[name][
                    "channel_total_count"
                ],
            }
        else:
            phase_status_summary[name] = {
                "phase_structure_status": "insufficient_evidence",
                "layer_c_candidate": False,
                "phase_estimable_candidate": False,
                "group_reference_degenerate": None,
                "group_max_channel_mad": None,
                "epsilon_mad_estimator_only": FEATURE_GROUPS_LAYER_B[name][
                    "epsilon_mad"
                ],
                "channel_degenerate_count": None,
                "channel_total_count": len(FEATURE_GROUPS_LAYER_B[name]["channels"]),
            }

    insufficient: list[dict[str, Any]] = []
    if len(reference_clips) == 0:
        insufficient.append(
            {
                "clip": None,
                "where": "reference_family_presence",
                "status": "insufficient_evidence",
                "reason": ref_status_reason,
            }
        )
    for name, summary_entry in phase_status_summary.items():
        if summary_entry["phase_structure_status"] == "insufficient_evidence":
            if summary_entry.get("layer_c_candidate"):
                reason = (
                    "Group is non-degenerate on Walk_F (MAD above "
                    "estimator-level epsilon), but Layer B cannot promote "
                    "this to phase_structured: that requires a "
                    "predictive-loss comparison against a phase-agnostic "
                    "baseline (§3.4). Marked layer_c_candidate=True."
                )
            else:
                reason = (
                    "Either insufficient frames (< "
                    f"{MIN_FRAMES_FOR_PHASE_ESTIMABILITY}) or non-finite "
                    "values, or no reference clip in --clips. No predictive "
                    "loss or phase library is attempted here."
                )
            insufficient.append(
                {
                    "clip": REFERENCE_FAMILY[0] if reference_clips else None,
                    "where": f"feature_groups_layer_b.{name}.phase_structure_status",
                    "status": "insufficient_evidence",
                    "layer_c_candidate": bool(summary_entry.get("layer_c_candidate")),
                    "reason": reason,
                }
            )
        elif summary_entry["phase_structure_status"] == "phase_degenerate":
            insufficient.append(
                {
                    "clip": REFERENCE_FAMILY[0],
                    "where": f"feature_groups_layer_b.{name}.phase_structure_status",
                    "status": "phase_degenerate",
                    "reason": (
                        "Reference scale at or below estimator-level "
                        "absolute MAD epsilon; the group must NOT be used "
                        "as combined membership evidence or normalised by "
                        "a zero-variance reference (§3.3)."
                    ),
                }
            )
    insufficient.extend(
        [
            {
                "clip": None,
                "where": "membership_status",
                "status": "not_implemented_layerB",
                "reason": (
                    "Layer B computes reference scale + degeneracy + "
                    "three-valued phase_structure_status only. Membership "
                    "claims are out of scope; see §5.3 / Layer C."
                ),
            },
            {
                "clip": None,
                "where": "phase_library / leave_one_phase_baseline / predictive_loss",
                "status": "not_implemented_layerB",
                "reason": (
                    "Numerical phase score and leave-one-phase baseline "
                    "are deferred to Layer C."
                ),
            },
            {
                "clip": None,
                "where": "pose_dyn / pose_rel",
                "status": "not_run_layerB",
                "reason": (
                    "pose_dyn requires processed_data/*.npz (bone angular "
                    "velocity summaries); pose_rel requires explicit "
                    "non-templating handling. Both deferred to Layer B.1."
                ),
            },
            {
                "clip": None,
                "where": "event_head_target_status",
                "status": "not_emitted_by_this_tool",
                "reason": (
                    "Layer B is read-only and produces no EventHead "
                    "target, no handoff_ready, no transition_done (§7)."
                ),
            },
            {
                "clip": None,
                "where": "transition_truth_promotion",
                "status": "forbidden_by_contract",
                "reason": (
                    "Reference scale / degeneracy tags MUST NOT be promoted "
                    "to transition truth."
                ),
            },
        ]
    )

    summary = _common_summary_root(
        "reference_scale_check", raw_root, clips, per_clip_paths
    )
    summary.update(
        {
            "definition_layer": {
                "attractor_definition_status": "not_implemented_layerB",
                "causal_state_definition_status": "not_implemented_layerB",
                "current_reference_family": REFERENCE_FAMILY,
                "membership_evidence_status": "not_implemented_layerB",
                "scope_caveat": (
                    "Walk_F is a single trajectory; class-level recurrence "
                    "remains INSUFFICIENT_EVIDENCE on current data (§4). "
                    "Layer B only emits per-feature-group reference scale, "
                    "degeneracy, and a three-valued phase_structure_status."
                ),
            },
            "estimation_grid": {
                "feature_groups_layer_b": feature_groups_layer_b_meta,
                "not_run_feature_groups_layer_b": list(NOT_RUN_FEATURE_GROUPS_LAYER_B),
                "min_frames_for_phase_estimability_estimator_only": (
                    MIN_FRAMES_FOR_PHASE_ESTIMABILITY
                ),
                "mad_to_gaussian_sigma_constant": MAD_TO_GAUSSIAN_SIGMA,
                "note": (
                    "Per-group epsilon_mad values are estimator-level "
                    "numeric thresholds and are NOT contract definition. "
                    "MAD / std / ptp / mean_abs are always reported per "
                    "channel so a reviewer can re-judge."
                ),
            },
            "feature_groups": list(FEATURE_GROUPS_LAYER_B.keys()),
            "feature_groups_meta": feature_groups_layer_b_meta,
            "quotient_definition": "see_gauge_check_mode_summary",
            "reference_family": REFERENCE_FAMILY,
            "reference_clip_names_resolved": [e["clip"] for e in reference_clips],
            "query_clip_names": [e["clip"] for e in query_clips],
            "reference_scale_source": (
                "reference_clips_only"
                if reference_clips
                else "no_reference_clip_present"
            ),
            "reference_clip_status": ref_status,
            "reference_clip_status_reason": ref_status_reason,
            "reference_scale_per_group": reference_per_group,
            "phase_structure_status_per_group": phase_status_summary,
            "phase_structure_status_enum": [
                "phase_structured",
                "phase_degenerate",
                "insufficient_evidence",
            ],
            "phase_structure_status_enum_source": (
                "docs/aperiodic_transition/"
                "2026-05-22_walk_f_causal_state_scaffold_v1.md:255"
            ),
            "walk_f_baseline": "not_implemented_layerB",
            "per_clip": [
                {
                    "clip": e["clip"],
                    "clip_role": e["clip_role"],
                    "frame_count": e["frame_count"],
                    "fps": e["fps"],
                    "feature_groups_layer_b": e["feature_groups_layer_b"],
                }
                for e in per_clip
            ],
            "query_clip_inspection_only": [
                {
                    "clip": e["clip"],
                    "frame_count": e["frame_count"],
                    "note": (
                        "Per-channel scale numbers for non-reference clips "
                        "are present for inspection only. They MUST NOT be "
                        "aggregated into the Walk_F reference scale "
                        "(reference_family is locked to Walk_F by §1.1)."
                    ),
                }
                for e in query_clips
            ],
            "sensitivity_summary": {
                "epsilon_mad_grid_status": "single_point_per_group_layerB",
                "rotation_grid_sensitivity_status": "not_emitted_in_this_mode",
                "translation_grid_sensitivity_status": "not_emitted_in_this_mode",
                "predictive_loss_sensitivity_status": "not_implemented_layerB",
                "yaw_activity_threshold_grid_status": "not_emitted_in_this_mode",
            },
            "causal_state_track_status": "not_implemented_layerB",
            "quotient_definition_status": (
                "inherited_from_layerA_se2_quotient_sanity_gate"
            ),
            "attractor_membership_status": "not_implemented_layerB",
            "phase_library_status": "not_implemented_layerB",
            "predictive_loss_status": "not_implemented_layerB",
            "event_head_target_status": "not_emitted_by_this_tool",
            "handoff_ready_status": "not_emitted_by_this_tool",
            "transition_done_status": "not_emitted_by_this_tool",
            "transition_truth_promotion": "forbidden_by_contract",
            "insufficient_evidence": insufficient,
            "notes": [
                "Layer B emits per-feature-group reference_scale + "
                "reference_degenerate + phase_structure_status drawn from "
                "the contract §6 enum {phase_structured, phase_degenerate, "
                "insufficient_evidence}. Layer B never emits "
                "phase_structured (that needs predictive-loss vs a "
                "phase-agnostic baseline; §3.4 / Layer C). Non-degenerate "
                "groups are emitted as insufficient_evidence with "
                "layer_c_candidate=True.",
                "Reference scale is computed exclusively from "
                "reference_clip_names_resolved (must equal "
                "reference_family). query_clip_names are present for "
                "inspection only and never aggregated.",
                "Epsilon_mad thresholds per group are estimator-level "
                "numeric; raw MAD / std / ptp / mean_abs are always "
                "reported so a reviewer can re-judge.",
                "pose_dyn and pose_rel groups are intentionally not_run at "
                "Layer B; see Layer B.1.",
            ],
        }
    )
    return summary


def _build_summary_phase_library_check(
    raw_root: Path,
    clips: list[str],
    per_clip: list[dict[str, Any]],
    per_clip_paths: list[str],
    internal_se2_precondition: dict[str, Any],
) -> dict[str, Any]:
    """Layer C minimal summary: Walk_F-only phase-library self-consistency.

    This intentionally does not include query leave/return boundary fields.
    Those require membership boundary/censoring semantics and are reserved for
    Layer C.1 by the Layer C minimal contract.
    """
    if clips != REFERENCE_FAMILY:
        raise FailFastError(
            "[walk_f_causal_state_probe] phase_library_check summary requires "
            f"clips exactly {REFERENCE_FAMILY}, got {clips!r}."
        )
    if len(per_clip) != 1 or per_clip[0]["clip"] != REFERENCE_FAMILY[0]:
        raise FailFastError(
            "[walk_f_causal_state_probe] phase_library_check summary received "
            "non-Walk_F per_clip payload; refusing to emit a mixed-reference artifact."
        )

    entry = per_clip[0]
    feature_groups = entry["feature_groups_layer_c"]

    phase_status_summary: dict[str, Any] = {}
    compact_feature_groups: dict[str, Any] = {}
    insufficient: list[dict[str, Any]] = []
    for group_name, group_payload in feature_groups.items():
        phase_status_summary[group_name] = {
            "phase_structure_status": group_payload["phase_structure_status"],
            "evidence_status": group_payload["evidence_status"],
            "self_consistency_signal_status": group_payload[
                "self_consistency_signal_status"
            ],
            "reference_degenerate": group_payload["reference_degenerate"],
            "active_channels": group_payload["active_channels"],
            "excluded_channel_count": len(group_payload["excluded_channels"]),
            "config_count": len(group_payload["config_results"]),
            "configs_beating_baseline_count": int(
                sum(
                    bool(c["beats_phase_agnostic_baseline"])
                    for c in group_payload["config_results"]
                )
            ),
            "group_ablation_role": group_payload["group_ablation_role"],
        }
        compact_feature_groups[group_name] = {
            "phase_structure_status": group_payload["phase_structure_status"],
            "evidence_status": group_payload["evidence_status"],
            "self_consistency_signal_status": group_payload[
                "self_consistency_signal_status"
            ],
            "reference_degenerate": group_payload["reference_degenerate"],
            "active_channels": group_payload["active_channels"],
            "excluded_channels": group_payload["excluded_channels"],
            "group_ablation_role": group_payload["group_ablation_role"],
            "baseline_loss_quantiles": group_payload["baseline_loss_quantiles"],
            "loss_curve_summary": group_payload["loss_curve_summary"],
            "phase_hat_curve_summary": group_payload["phase_hat_curve_summary"],
            "config_results_summary": [
                {
                    "history_window_frames": c["history_window_frames"],
                    "future_horizon_frames": c["future_horizon_frames"],
                    "neighborhood_radius_frames": c["neighborhood_radius_frames"],
                    "distance_metric": c["distance_metric"],
                    "valid_phase_candidate_count": c["valid_phase_candidate_count"],
                    "valid_query_count": c["valid_query_count"],
                    "valid_query_fraction": c["valid_query_fraction"],
                    "config_status": c["config_status"],
                    "beats_phase_agnostic_baseline": c[
                        "beats_phase_agnostic_baseline"
                    ],
                    "median_relative_improvement": c[
                        "median_relative_improvement"
                    ],
                    "phase_loss_quantiles": c["phase_loss_quantiles"],
                    "phase_agnostic_loss_quantiles": c[
                        "phase_agnostic_loss_quantiles"
                    ],
                }
                for c in group_payload["config_results"]
            ],
            "full_curve_artifact_path": per_clip_paths[0],
        }
        if group_payload["evidence_status"] == "INSUFFICIENT_EVIDENCE":
            insufficient.append(
                {
                    "clip": REFERENCE_FAMILY[0],
                    "where": f"feature_groups_layer_c.{group_name}.phase_structure_status",
                    "status": "INSUFFICIENT_EVIDENCE",
                    "phase_structure_status": group_payload["phase_structure_status"],
                    "reason": (
                        "Layer C minimal is a Walk_F single-trajectory "
                        "self-consistency audit. INSUFFICIENT_EVIDENCE is an "
                        "expected contract-pass result when the estimator grid "
                        "is mixed, the group is degenerate, or valid nonlocal "
                        "candidates are insufficient."
                    ),
                }
            )

    insufficient.extend(
        [
            {
                "clip": None,
                "where": "class_level_attractor_claim",
                "status": "INSUFFICIENT_EVIDENCE",
                "reason": (
                    "Walk_F is still one 88-frame trajectory. Layer C minimal "
                    "does not establish class-level recurrence."
                ),
            },
            {
                "clip": None,
                "where": "query_leave_return_boundary",
                "status": "not_run_layerC_minimal_reserved_for_layerC1",
                "reason": (
                    "leave_interval / return_interval / censoring fields require "
                    "query phase lookup and membership boundary logic; omitted "
                    "from Layer C minimal by contract."
                ),
            },
            {
                "clip": None,
                "where": "event_head_target_status",
                "status": "not_emitted_by_this_tool",
                "reason": (
                    "Layer C minimal is read-only and produces no EventHead "
                    "target, no handoff_ready, no transition_done."
                ),
            },
            {
                "clip": None,
                "where": "transition_truth_promotion",
                "status": "forbidden_by_contract",
                "reason": (
                    "Phase-library self-consistency MUST NOT be promoted to "
                    "transition truth or runtime switching behavior."
                ),
            },
        ]
    )

    summary = _common_summary_root(
        "phase_library_check",
        raw_root,
        clips,
        per_clip_paths,
    )
    summary.update(
        {
            "definition_layer": {
                "attractor_definition_status": "not_implemented_layerC_minimal",
                "causal_state_definition_status": "not_implemented_layerC_minimal",
                "current_reference_family": REFERENCE_FAMILY,
                "membership_evidence_status": "not_implemented_layerC_minimal",
                "scope_caveat": (
                    "Walk_F is a single 88-frame trajectory. Layer C minimal "
                    "checks intra-trajectory phase-library self-consistency "
                    "only; class-level recurrence and attractor membership "
                    "remain INSUFFICIENT_EVIDENCE."
                ),
            },
            "layer": "Layer C minimal",
            "layer_c_contract_doc": (
                "docs/aperiodic_transition/"
                "2026-05-23_walk_f_causal_state_layerc_minimal_contract.md"
            ),
            "layer_c_contract_status": "pass",
            "expected_insufficient_evidence_is_contract_pass": True,
            "reference_family": REFERENCE_FAMILY,
            "reference_clip_names_resolved": [REFERENCE_FAMILY[0]],
            "query_clip_names": [],
            "input_clip_policy": "clips_must_equal_reference_family_walk_f_only",
            "internal_se2_gauge_precondition": internal_se2_precondition,
            "estimation_grid": {
                "history_window_frames": list(LAYER_C_HISTORY_WINDOW_FRAMES),
                "future_horizon_frames": list(LAYER_C_FUTURE_HORIZON_FRAMES),
                "neighborhood_radius_frames": list(
                    LAYER_C_NEIGHBORHOOD_RADIUS_FRAMES
                ),
                "distance_metric": list(LAYER_C_DISTANCE_METRICS),
                "grid_point_count": int(
                    len(LAYER_C_HISTORY_WINDOW_FRAMES)
                    * len(LAYER_C_FUTURE_HORIZON_FRAMES)
                    * len(LAYER_C_NEIGHBORHOOD_RADIUS_FRAMES)
                    * len(LAYER_C_DISTANCE_METRICS)
                ),
                "min_valid_query_count_estimator_only": LAYER_C_MIN_VALID_QUERY_COUNT,
                "min_valid_query_fraction_estimator_only": (
                    LAYER_C_MIN_VALID_QUERY_FRACTION
                ),
                "improvement_tol_estimator_only": LAYER_C_IMPROVEMENT_TOL,
                "note": (
                    "All grid points are reported. A single successful setting "
                    "is not sufficient; mixed results remain "
                    "INSUFFICIENT_EVIDENCE."
                ),
            },
            "included_feature_groups": list(LAYER_C_FEATURE_GROUPS.keys()),
            "excluded_feature_groups": LAYER_C_EXCLUDED_FEATURE_GROUPS,
            "phase_structure_status_per_group": phase_status_summary,
            "self_consistency_signal_per_group": {
                group_name: {
                    "self_consistency_signal_status": payload[
                        "self_consistency_signal_status"
                    ],
                    "loss_curve_summary": payload["loss_curve_summary"],
                    "phase_hat_curve_summary": payload["phase_hat_curve_summary"],
                }
                for group_name, payload in feature_groups.items()
            },
            "walk_f_leave_one_neighborhood_baseline": {
                "clip": REFERENCE_FAMILY[0],
                "baseline_type": (
                    "phase_agnostic_median_future_from_same_nonlocal_candidate_set"
                ),
                "neighborhood_exclusion_rule": "abs(candidate_frame - query_frame) > radius",
                "ill_conditioned_single_trajectory_caveat": (
                    "Neighborhood radius choices trade off between too few "
                    "independent candidates and local phase leakage."
                ),
            },
            "feature_groups_layer_c": compact_feature_groups,
            "full_curve_artifact_policy": (
                "Root summary is compact. Full per-config loss_curve, "
                "loss_percentile_curve, and phase_confidence_gap arrays are "
                "stored in the per-clip artifact path."
            ),
            "per_clip": [
                {
                    "clip": entry["clip"],
                    "clip_role": entry["clip_role"],
                    "frame_count": entry["frame_count"],
                    "fps": entry["fps"],
                    "feature_groups_layer_c_detail_artifact_path": per_clip_paths[0],
                    "phase_structure_status_per_group": phase_status_summary,
                    "excluded_feature_groups_layer_c": entry[
                        "excluded_feature_groups_layer_c"
                    ],
                    "membership_status": entry["membership_status"],
                    "query_boundary_status": entry["query_boundary_status"],
                }
            ],
            "sensitivity_summary": {
                "history_window_sensitivity_status": "reported_full_grid",
                "future_horizon_sensitivity_status": "reported_full_grid",
                "neighborhood_radius_sensitivity_status": "reported_full_grid",
                "distance_metric_sensitivity_status": "reported_full_grid",
                "query_boundary_sensitivity_status": "not_run_layerC_minimal",
            },
            "causal_state_track_status": "self_consistency_only_layerC_minimal",
            "quotient_definition_status": "internal_layerA_se2_precondition_passed",
            "attractor_membership_status": "not_implemented_layerC_minimal",
            "phase_library_status": "walk_f_self_consistency_only",
            "predictive_loss_status": "walk_f_leave_one_neighborhood_self_test_only",
            "query_leave_return_status": "not_run_layerC_minimal_reserved_for_layerC1",
            "event_head_target_status": "not_emitted_by_this_tool",
            "handoff_ready_status": "not_emitted_by_this_tool",
            "transition_done_status": "not_emitted_by_this_tool",
            "transition_truth_promotion": "forbidden_by_contract",
            "insufficient_evidence": insufficient,
            "notes": [
                "Layer C minimal does not read external gauge_check artifacts; "
                "it reruns a lightweight internal SE(2) sanity precondition.",
                "INSUFFICIENT_EVIDENCE can be the expected contract-pass "
                "result on the current 88-frame single Walk_F trajectory.",
                "leave/return/censoring fields are intentionally absent and "
                "reserved for Layer C.1.",
                "turn_dyn is excluded from membership / phase evidence because "
                "Layer B found it degenerate on Walk_F.",
            ],
        }
    )
    return summary


def _build_summary_query_boundary_check(
    raw_root: Path,
    clips: list[str],
    walk_f_group_payloads: dict[str, dict[str, Any]],
    per_clip: list[dict[str, Any]],
    per_clip_paths: list[str],
    internal_se2_precondition: dict[str, Any],
) -> dict[str, Any]:
    """Layer C.1 summary: Walk_F self-baseline band + query leave/return
    + censoring audit on the 4 turn clips.

    This builder MUST NOT emit attractor membership, EventHead targets,
    handoff_ready, transition_done, cross-attractor claims, or any
    combined-membership score over feature groups. Every one of those
    fields is recorded as not_emitted_by_this_tool or
    forbidden_by_contract.
    """
    # Defensive: clip set must be exactly {Walk_F + 4 turn clips}. The CLI
    # validator already enforces this BEFORE args.out_dir.mkdir, but we
    # repeat the check here so a future caller cannot bypass it by
    # invoking this builder directly with a partial clip list.
    if frozenset(clips) != LAYER_C1_REQUIRED_CLIP_SET:
        raise FailFastError(
            "[walk_f_causal_state_probe] Layer C.1 summary requires --clips set "
            f"equal to {sorted(LAYER_C1_REQUIRED_CLIP_SET)!r}; got {clips!r}. "
            "Refusing to emit a mixed / partial-family artifact "
            "(docs/removal_policy.md §3-§4)."
        )

    # Walk_F reference summary (small, decoupled from per-query payloads).
    walk_f_group_summary: dict[str, Any] = {}
    for group_name, payload in walk_f_group_payloads.items():
        _layer_c1_forbid_turn_dyn(
            group_name, "_build_summary_query_boundary_check.walk_f_group_summary"
        )
        walk_f_group_summary[group_name] = {
            "reference_degenerate": bool(payload["reference_degenerate"]),
            "active_channels": list(payload["active_channels"]),
            "excluded_channels": payload["excluded_channels"],
            "group_ablation_role": payload["group_ablation_role"],
            "source": payload["source"],
        }

    # Per-query aggregate + censoring summary across (clip, group, config).
    per_query_clip: list[dict[str, Any]] = []
    censoring_rows: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []
    per_group_aggregate: dict[str, Any] = {
        name: {
            "returned_config_count": 0,
            "never_left_config_count": 0,
            "never_returned_config_count": 0,
            "insufficient_evidence_config_count": 0,
            "left_censored_leave_config_count": 0,
            "right_censored_return_config_count": 0,
            "neighbor_consistency_conflict_pair_count": 0,
            "reference_degenerate": False,
            "active_channels": list(
                walk_f_group_payloads[name]["active_channels"]
            ),
            "excluded_channels": walk_f_group_payloads[name]["excluded_channels"],
            "group_ablation_role": (
                walk_f_group_payloads[name]["group_ablation_role"]
            ),
        }
        for name in LAYER_C1_FEATURE_GROUPS
    }

    for entry in per_clip:
        clip_name = entry["clip"]
        groups_payload = entry["feature_groups_layer_c1"]
        compact_groups: dict[str, Any] = {}
        for group_name, group_payload in groups_payload.items():
            _layer_c1_forbid_turn_dyn(
                group_name,
                "_build_summary_query_boundary_check.per_clip_compact",
            )
            ref_degenerate = bool(group_payload["reference_degenerate"])
            per_group_aggregate[group_name]["reference_degenerate"] = (
                per_group_aggregate[group_name]["reference_degenerate"]
                or ref_degenerate
            )
            if ref_degenerate:
                compact_groups[group_name] = {
                    "phase_structure_status": "phase_degenerate",
                    "evidence_status": "INSUFFICIENT_EVIDENCE",
                    "reference_degenerate": True,
                    "active_channels": group_payload["active_channels"],
                    "excluded_channels": group_payload["excluded_channels"],
                    "group_ablation_role": group_payload["group_ablation_role"],
                    "config_count": 0,
                    "returned_config_count": 0,
                    "never_left_config_count": 0,
                    "never_returned_config_count": 0,
                    "insufficient_evidence_config_count": 0,
                    "left_censored_leave_config_count": 0,
                    "right_censored_return_config_count": 0,
                    "neighbor_consistency_conflict_pair_count": 0,
                    "boundary_detection_skipped_reason": group_payload[
                        "boundary_detection_skipped_reason"
                    ],
                    "full_curve_artifact_path": None,
                }
                insufficient.append(
                    {
                        "clip": clip_name,
                        "where": (
                            f"feature_groups_layer_c1.{group_name}."
                            "phase_structure_status"
                        ),
                        "status": "INSUFFICIENT_EVIDENCE",
                        "phase_structure_status": "phase_degenerate",
                        "reason": (
                            "Walk_F reference is degenerate for this group; "
                            "Layer C.1 contract §4 mandates skipping boundary "
                            "detection. This is a contract PASS outcome on "
                            "single-trajectory Walk_F data, not failure."
                        ),
                    }
                )
                continue

            compact_groups[group_name] = {
                "phase_structure_status": "insufficient_evidence",
                "evidence_status": "INSUFFICIENT_EVIDENCE",
                "reference_degenerate": False,
                "active_channels": group_payload["active_channels"],
                "excluded_channels": group_payload["excluded_channels"],
                "group_ablation_role": group_payload["group_ablation_role"],
                "config_count": int(len(group_payload["configs"])),
                "returned_config_count": int(
                    group_payload["returned_config_count"]
                ),
                "never_left_config_count": int(
                    group_payload["never_left_config_count"]
                ),
                "never_returned_config_count": int(
                    group_payload["never_returned_config_count"]
                ),
                "insufficient_evidence_config_count": int(
                    group_payload["insufficient_evidence_config_count"]
                ),
                "left_censored_leave_config_count": int(
                    group_payload["left_censored_leave_config_count"]
                ),
                "right_censored_return_config_count": int(
                    group_payload["right_censored_return_config_count"]
                ),
                "neighbor_consistency_conflict_pair_count": int(
                    group_payload["neighbor_consistency_conflict_pair_count"]
                ),
                "boundary_detection_skipped_reason": None,
                "full_curve_artifact_path": None,
            }
            per_group_aggregate[group_name]["returned_config_count"] += int(
                group_payload["returned_config_count"]
            )
            per_group_aggregate[group_name]["never_left_config_count"] += int(
                group_payload["never_left_config_count"]
            )
            per_group_aggregate[group_name][
                "never_returned_config_count"
            ] += int(group_payload["never_returned_config_count"])
            per_group_aggregate[group_name][
                "insufficient_evidence_config_count"
            ] += int(group_payload["insufficient_evidence_config_count"])
            per_group_aggregate[group_name][
                "left_censored_leave_config_count"
            ] += int(group_payload["left_censored_leave_config_count"])
            per_group_aggregate[group_name][
                "right_censored_return_config_count"
            ] += int(group_payload["right_censored_return_config_count"])
            per_group_aggregate[group_name][
                "neighbor_consistency_conflict_pair_count"
            ] += int(group_payload["neighbor_consistency_conflict_pair_count"])

            if int(group_payload["insufficient_evidence_config_count"]) > 0:
                insufficient.append(
                    {
                        "clip": clip_name,
                        "where": (
                            f"feature_groups_layer_c1.{group_name}."
                            "return_to_reference_status"
                        ),
                        "status": "INSUFFICIENT_EVIDENCE",
                        "phase_structure_status": "insufficient_evidence",
                        "insufficient_evidence_config_count": int(
                            group_payload["insufficient_evidence_config_count"]
                        ),
                        "neighbor_consistency_conflict_pair_count": int(
                            group_payload[
                                "neighbor_consistency_conflict_pair_count"
                            ]
                        ),
                        "reason": (
                            "At least one estimator config returned "
                            "INSUFFICIENT_EVIDENCE due to too few valid "
                            "query phases or neighbor-grid disagreement on "
                            "leave/return endpoints. On single-trajectory "
                            "Walk_F baseline this is contract PASS "
                            "(Layer C.1 contract §1 rule 2)."
                        ),
                    }
                )

            # Per-config censoring rows for the censoring_summary table.
            for c in group_payload["configs"]:
                if c["left_censored_leave"] or c["right_censored_return"]:
                    censoring_rows.append(
                        {
                            "clip": clip_name,
                            "group": group_name,
                            "config_id": c["config_id"],
                            "left_censored_leave": bool(
                                c["left_censored_leave"]
                            ),
                            "right_censored_return": bool(
                                c["right_censored_return"]
                            ),
                            "leave_interval": c["leave_interval"],
                            "return_interval": c["return_interval"],
                            "return_to_reference_status": c[
                                "return_to_reference_status"
                            ],
                        }
                    )

        per_query_clip.append(
            {
                "clip": clip_name,
                "raw_json_path": entry["raw_json_path"],
                "fps": entry["fps"],
                "frame_count": entry["frame_count"],
                "clip_role": entry["clip_role"],
                "feature_groups_layer_c1_compact": compact_groups,
                "full_curve_artifact_paths_note": (
                    "Full per-config loss_curve and loss_percentile_curve "
                    "arrays are stored in the per-clip artifact JSONs."
                ),
            }
        )

    # Resolve per-clip artifact paths into the compact per-group dicts so
    # consumers can find the full curves without hunting through the
    # filesystem.
    clip_to_path: dict[str, str] = {}
    for entry, path in zip(per_clip, per_clip_paths, strict=True):
        clip_to_path[entry["clip"]] = path
    for clip_summary in per_query_clip:
        for compact in clip_summary["feature_groups_layer_c1_compact"].values():
            compact["full_curve_artifact_path"] = clip_to_path.get(
                clip_summary["clip"]
            )

    insufficient.extend(
        [
            {
                "clip": None,
                "where": "class_level_attractor_claim",
                "status": "INSUFFICIENT_EVIDENCE",
                "reason": (
                    "Walk_F remains a single 88-frame trajectory; class-level "
                    "recurrence is not established. Layer C.1 audits query "
                    "boundaries against this single-trajectory baseline only "
                    "(scaffold v1 §4 single-trajectory caveat)."
                ),
            },
            {
                "clip": None,
                "where": "attractor_membership_status",
                "status": "not_emitted_by_this_tool",
                "reason": (
                    "Layer C.1 emits query leave/return intervals + "
                    "censoring flags only. Attractor membership requires a "
                    "separate contract (Layer C.1 contract §1 rule 1)."
                ),
            },
            {
                "clip": None,
                "where": "event_head_target_status",
                "status": "not_emitted_by_this_tool",
                "reason": (
                    "Layer C.1 is read-only and produces no EventHead "
                    "target, no handoff_ready, no transition_done."
                ),
            },
            {
                "clip": None,
                "where": "handoff_ready_status",
                "status": "not_emitted_by_this_tool",
                "reason": "Reserved; see scaffold v1 §2 + §7.",
            },
            {
                "clip": None,
                "where": "transition_done_status",
                "status": "not_emitted_by_this_tool",
                "reason": "Reserved; see scaffold v1 §2 + §7.",
            },
            {
                "clip": None,
                "where": "cross_attractor_claim_status",
                "status": "forbidden_by_contract",
                "reason": (
                    "reference_family is locked to Walk_F. No Walk -> Run / "
                    "Walk -> Idle / etc claims are derivable from this audit "
                    "(scaffold v1 §2)."
                ),
            },
            {
                "clip": None,
                "where": "transition_truth_promotion",
                "status": "forbidden_by_contract",
                "reason": (
                    "Query leave/return intervals MUST NOT be promoted to "
                    "general transition truth, training labels, or runtime "
                    "switching behavior."
                ),
            },
            {
                "clip": None,
                "where": "combined_membership_score",
                "status": "forbidden_by_contract",
                "reason": (
                    "Layer C.1 reports leave/return per feature group "
                    "separately. No combined membership score is emitted, "
                    "and turn_dyn is hard-rejected at every entry point."
                ),
            },
            {
                "clip": None,
                "where": "pose_dyn / pose_rel / processed_data_pose_npz",
                "status": "not_run_layerC1_reserved_for_layerB1",
                "reason": (
                    "pose_dyn and pose_rel require processed_data/*.npz "
                    "schema validation and a non-templating contract; both "
                    "are reserved for Layer B.1."
                ),
            },
        ]
    )

    censoring_summary = {
        "left_censored_leave_total": int(
            sum(1 for r in censoring_rows if r["left_censored_leave"])
        ),
        "right_censored_return_total": int(
            sum(1 for r in censoring_rows if r["right_censored_return"])
        ),
        "rows_truncation_policy": "all_rows_emitted",
        "rows": censoring_rows,
        "expected_insufficient_evidence_is_contract_pass": True,
        "explanation": (
            "Per Layer C.1 contract §1 rule 2 and scaffold v1 §4 + §5.4, "
            "left_censored_leave / right_censored_return / "
            "return_to_reference_status=INSUFFICIENT_EVIDENCE are all "
            "contract PASS outcomes on the current single-trajectory "
            "Walk_F baseline. They are recorded here for audit "
            "transparency and are not implementation failures."
        ),
    }

    estimation_grid = {
        "history_window_frames": list(LAYER_C_HISTORY_WINDOW_FRAMES),
        "future_horizon_frames": list(LAYER_C_FUTURE_HORIZON_FRAMES),
        "neighborhood_radius_frames": list(LAYER_C_NEIGHBORHOOD_RADIUS_FRAMES),
        "distance_metric": list(LAYER_C_DISTANCE_METRICS),
        "band_quantile": [label for label, _ in LAYER_C1_BAND_QUANTILES],
        "band_quantile_fractions": {
            label: float(q) for label, q in LAYER_C1_BAND_QUANTILES
        },
        "min_consecutive_frames_for_leave": list(
            LAYER_C1_MIN_CONSEC_FRAMES_FOR_LEAVE
        ),
        "min_consecutive_frames_for_return": list(
            LAYER_C1_MIN_CONSEC_FRAMES_FOR_RETURN
        ),
        "grid_point_count_per_group": int(
            len(LAYER_C_HISTORY_WINDOW_FRAMES)
            * len(LAYER_C_FUTURE_HORIZON_FRAMES)
            * len(LAYER_C_NEIGHBORHOOD_RADIUS_FRAMES)
            * len(LAYER_C_DISTANCE_METRICS)
            * len(LAYER_C1_BAND_QUANTILES)
            * len(LAYER_C1_MIN_CONSEC_FRAMES_FOR_LEAVE)
            * len(LAYER_C1_MIN_CONSEC_FRAMES_FOR_RETURN)
        ),
        "neighbor_consistency_tol_frames": int(
            LAYER_C1_NEIGHBOR_CONSISTENCY_TOL_FRAMES
        ),
        "min_valid_query_count_estimator_only": LAYER_C1_MIN_VALID_QUERY_COUNT,
        "min_valid_query_fraction_estimator_only": (
            LAYER_C1_MIN_VALID_QUERY_FRACTION
        ),
        "note": (
            "Every grid point is reported. A single config's leave/return "
            "answer is overridden to INSUFFICIENT_EVIDENCE if a 1-step "
            "neighbor in any dimension gives a different existence "
            "boolean or an endpoint that differs by more than the "
            "neighbor_consistency_tol_frames frames."
        ),
    }

    summary = _common_summary_root(
        "query_boundary_check", raw_root, clips, per_clip_paths
    )
    summary.update(
        {
            "layer": "Layer C.1",
            "layer_c1_contract_doc": (
                "docs/aperiodic_transition/"
                "2026-05-23_walk_f_causal_state_layerc1_query_boundary_contract.md"
            ),
            "layer_c1_contract_status": "pass",
            "expected_insufficient_evidence_is_contract_pass": True,
            "definition_layer": {
                "attractor_definition_status": "not_emitted_by_this_tool",
                "causal_state_definition_status": "not_emitted_by_this_tool",
                "current_reference_family": REFERENCE_FAMILY,
                "membership_evidence_status": (
                    "not_emitted_by_this_tool_layer_c1_query_boundary_only"
                ),
                "scope_caveat": (
                    "Walk_F is a single 88-frame trajectory. Layer C.1 "
                    "audits query leave/return relative to this single-"
                    "trajectory baseline only; class-level recurrence and "
                    "attractor membership remain INSUFFICIENT_EVIDENCE."
                ),
            },
            "reference_family": REFERENCE_FAMILY,
            "query_family": list(LAYER_C1_QUERY_FAMILY),
            "reference_clip_names_resolved": [LAYER_C1_REFERENCE_CLIP],
            "query_clip_names_resolved": list(LAYER_C1_QUERY_FAMILY),
            "input_clip_policy": (
                "clips_must_set_equal_walk_f_plus_4_turn_clips_no_duplicates"
            ),
            "internal_se2_gauge_precondition": internal_se2_precondition,
            "walk_f_phase_library_source": (
                "internal_phase_library_check_rerun_not_external_artifact"
            ),
            "estimation_grid": estimation_grid,
            "included_feature_groups": list(LAYER_C1_FEATURE_GROUPS.keys()),
            "excluded_feature_groups": LAYER_C1_EXCLUDED_FEATURE_GROUPS,
            "walk_f_group_payload_summary": walk_f_group_summary,
            "per_query_clip": per_query_clip,
            "per_group_aggregate": per_group_aggregate,
            "censoring_summary": censoring_summary,
            "sensitivity_summary": {
                "history_window_sensitivity_status": "reported_full_grid",
                "future_horizon_sensitivity_status": "reported_full_grid",
                "neighborhood_radius_sensitivity_status": "reported_full_grid",
                "distance_metric_sensitivity_status": "reported_full_grid",
                "band_quantile_sensitivity_status": "reported_full_grid",
                "min_consecutive_leave_sensitivity_status": "reported_full_grid",
                "min_consecutive_return_sensitivity_status": "reported_full_grid",
                "neighbor_consistency_override_status": (
                    "applied_per_query_clip_per_group"
                ),
            },
            "causal_state_track_status": (
                "query_boundary_audit_only_layer_c1"
            ),
            "quotient_definition_status": (
                "internal_layerA_se2_precondition_passed"
            ),
            "phase_library_status": (
                "internal_phase_library_check_rerun_not_external_artifact"
            ),
            "predictive_loss_status": (
                "walk_f_leave_one_neighborhood_plus_query_phase_match"
            ),
            "query_leave_return_status": "emitted_per_clip_per_group_per_config",
            "attractor_membership_status": (
                "not_emitted_by_this_tool_layer_c1_query_boundary_only"
            ),
            "event_head_target_status": "not_emitted_by_this_tool",
            "handoff_ready_status": "not_emitted_by_this_tool",
            "transition_done_status": "not_emitted_by_this_tool",
            "cross_attractor_claim_status": "forbidden_by_contract",
            "transition_truth_promotion": "forbidden_by_contract",
            "combined_membership_score_status": (
                "forbidden_by_contract_groups_reported_separately"
            ),
            "turn_dyn_routing_policy": (
                "fail_fast_on_any_combined_membership_entry_point"
            ),
            "insufficient_evidence": insufficient,
            "notes": [
                "Layer C.1 does not read external gauge_check / phase_library_check "
                "artifacts; it reruns a lightweight internal SE(2) sanity "
                "precondition and rebuilds the Walk_F phase library on-the-fly.",
                "INSUFFICIENT_EVIDENCE is the expected contract-PASS outcome on "
                "the current 88-frame single Walk_F trajectory whenever the "
                "estimator grid is mixed, the query clip is too short for "
                "robust neighbor coverage, or the leave/return endpoints "
                "disagree across the 1-step grid neighborhood.",
                "left_censored_leave and right_censored_return are scaffold v1 "
                "§5.4 outcomes, not implementation failures.",
                "Feature groups are reported separately. No combined membership "
                "score is emitted. turn_dyn is hard-rejected at every "
                "combined-membership entry point by _layer_c1_forbid_turn_dyn.",
                "pose_dyn, pose_rel, and processed_data/*.npz remain reserved "
                "for Layer B.1.",
            ],
        }
    )
    return summary


def _build_summary_pose_reference_scale_check(
    raw_root: Path,
    clips: list[str],
    per_clip: list[dict[str, Any]],
    per_clip_paths: list[str],
) -> dict[str, Any]:
    """Layer B.1 summary: Walk_F pose_dyn / pose_rel reference scale +
    degeneracy + processed-data schema audit on the 5-clip set.

    Reference scale is computed exclusively from clips with
    ``contributes_to_reference_scale == True`` (i.e. Walk_F). Query
    (non-reference) clip per-channel raw scale is present for
    inspection only and is NEVER aggregated into the Walk_F
    reference scale. No combined-membership / combined-score, no
    leave/return interval, no attractor membership, no EventHead
    target, no handoff_ready / transition_done, no cross-attractor
    claim is emitted.
    """
    if frozenset(clips) != LAYER_B1_REQUIRED_CLIP_SET:
        raise FailFastError(
            "[walk_f_causal_state_probe] Layer B.1 summary requires --clips "
            f"set-equal to {sorted(LAYER_B1_REQUIRED_CLIP_SET)!r}; got "
            f"{clips!r}. Refusing to emit a mixed / partial-family artifact "
            "(docs/removal_policy.md §3-§4)."
        )

    reference_clips = [
        e for e in per_clip if e["contributes_to_reference_scale"]
    ]
    query_clips = [
        e for e in per_clip if not e["contributes_to_reference_scale"]
    ]
    if len(reference_clips) != 1:
        raise FailFastError(
            "[walk_f_causal_state_probe] Layer B.1 summary expected exactly "
            f"one reference clip after CLI validation; got "
            f"{len(reference_clips)}. This indicates the CLI guard was "
            "bypassed."
        )

    reference_per_group: dict[str, Any] = reference_clips[0][
        "feature_groups_layer_b1"
    ]

    feature_groups_meta = {
        name: {
            "source": group_def["source"],
            "source_npz_key": group_def["source_npz_key"],
            "extracted_channel_count": int(
                group_def["extracted_channel_count"]
            ),
            "unit_label": group_def["unit_label"],
            "epsilon_mad_estimator_only": float(group_def["epsilon_mad"]),
            "group_ablation_role": group_def["group_ablation_role"],
        }
        for name, group_def in LAYER_B1_FEATURE_GROUPS.items()
    }

    phase_status_summary: dict[str, Any] = {}
    for name in LAYER_B1_FEATURE_GROUPS:
        ref = reference_per_group[name]
        phase_status_summary[name] = {
            "phase_structure_status": ref["phase_structure_status"],
            "layer_c_candidate": bool(ref["layer_c_candidate"]),
            "phase_estimable_candidate": bool(
                ref["phase_estimable_candidate"]
            ),
            "group_reference_degenerate": bool(
                ref["group_reference_degenerate"]
            ),
            "group_max_channel_mad": float(ref["group_max_channel_mad"]),
            "epsilon_mad_estimator_only": float(
                ref["epsilon_mad_estimator_only"]
            ),
            "channel_degenerate_count": int(ref["channel_degenerate_count"]),
            "channel_total_count": int(ref["channel_total_count"]),
        }

    frame_alignment_per_clip: list[dict[str, Any]] = []
    per_query_clip_inspection: list[dict[str, Any]] = []
    for entry in per_clip:
        frame_alignment_per_clip.append(
            {
                "clip": entry["clip"],
                "json_frame_count": int(
                    entry["frame_alignment"]["json_frame_count"]
                ),
                "npz_bone_ang_vel_T": int(
                    entry["frame_alignment"]["npz_bone_ang_vel_T"]
                ),
                "npz_bone_rot6d_T": int(
                    entry["frame_alignment"]["npz_bone_rot6d_T"]
                ),
                "aligned": bool(entry["frame_alignment"]["aligned"]),
            }
        )
        if entry["contributes_to_reference_scale"]:
            continue
        compact_groups: dict[str, Any] = {}
        for group_name, group_payload in entry[
            "feature_groups_layer_b1"
        ].items():
            compact_groups[group_name] = {
                "phase_structure_status": group_payload[
                    "phase_structure_status"
                ],
                "reference_degenerate": bool(
                    group_payload["reference_degenerate"]
                ),
                "reference_degenerate_source": group_payload[
                    "reference_degenerate_source"
                ],
                "layer_c_candidate": bool(group_payload["layer_c_candidate"]),
                "extracted_matrix_shape": group_payload[
                    "extracted_matrix_shape"
                ],
                "extracted_matrix_dtype": group_payload[
                    "extracted_matrix_dtype"
                ],
                "extracted_matrix_device": group_payload[
                    "extracted_matrix_device"
                ],
                "epsilon_mad_estimator_only": float(
                    group_payload["epsilon_mad_estimator_only"]
                ),
                "group_ablation_role": group_payload["group_ablation_role"],
                "source": group_payload["source"],
                "source_npz_key": group_payload["source_npz_key"],
                "boundary_detection_status": group_payload[
                    "boundary_detection_status"
                ],
                "channel_count_inspected": int(
                    len(group_payload["channels"])
                ),
            }
        per_query_clip_inspection.append(
            {
                "clip": entry["clip"],
                "raw_json_path": entry["raw_json_path"],
                "npz_path": entry["npz_path"],
                "fps": entry["fps"],
                "frame_count": int(entry["frame_count"]),
                "clip_role": entry["clip_role"],
                "feature_groups_layer_b1_compact": compact_groups,
                "full_inspection_artifact_path_note": (
                    "Per-channel walk_f_projected_z_summary + "
                    "query_self_scale_inspection_only blocks are stored "
                    "in the per-clip artifact JSON."
                ),
            }
        )

    # Wire per-clip artifact path back into the compact per-query block.
    clip_to_path: dict[str, str] = {}
    for entry, path in zip(per_clip, per_clip_paths, strict=True):
        clip_to_path[entry["clip"]] = path
    for q in per_query_clip_inspection:
        q["full_inspection_artifact_path"] = clip_to_path.get(q["clip"])

    insufficient: list[dict[str, Any]] = []
    for name, st in phase_status_summary.items():
        if st["phase_structure_status"] == "phase_degenerate":
            insufficient.append(
                {
                    "clip": LAYER_B1_REFERENCE_CLIP,
                    "where": (
                        f"per_group_reference_scale.{name}."
                        "phase_structure_status"
                    ),
                    "status": "phase_degenerate",
                    "reason": (
                        "Reference scale at or below estimator-level "
                        "absolute MAD epsilon on every channel; this "
                        "group MUST NOT be used as combined-membership "
                        "evidence or normalised by a zero-variance "
                        "reference (scaffold v1 §3.3)."
                    ),
                }
            )
        elif st["phase_structure_status"] == "insufficient_evidence":
            insufficient.append(
                {
                    "clip": LAYER_B1_REFERENCE_CLIP,
                    "where": (
                        f"per_group_reference_scale.{name}."
                        "phase_structure_status"
                    ),
                    "status": "insufficient_evidence",
                    "layer_c_candidate": bool(st["layer_c_candidate"]),
                    "reason": (
                        "Non-degenerate Walk_F group, but Layer B.1 cannot "
                        "promote this to phase_structured: that requires a "
                        "predictive-loss comparison vs a phase-agnostic "
                        "baseline (scaffold v1 §3.4). Marked "
                        "layer_c_candidate=True."
                    ),
                }
            )
    insufficient.extend(
        [
            {
                "clip": None,
                "where": "class_level_attractor_claim",
                "status": "INSUFFICIENT_EVIDENCE",
                "reason": (
                    "Walk_F remains a single 88-frame trajectory "
                    "(scaffold v1 §4); class-level recurrence is not "
                    "established. Layer B.1 only audits pose reference "
                    "scale + degeneracy on this single-trajectory baseline."
                ),
            },
            {
                "clip": None,
                "where": "membership_status",
                "status": "not_implemented_layerB1",
                "reason": (
                    "Layer B.1 computes per-feature-group reference scale + "
                    "degeneracy + processed-data schema audit only. "
                    "Membership / leave-return / predictive loss are out of "
                    "scope (see Layer C minimal + Layer C.1)."
                ),
            },
            {
                "clip": None,
                "where": "phase_library / leave_one_phase_baseline / predictive_loss",
                "status": "not_implemented_layerB1",
                "reason": (
                    "Numerical phase score and leave-one-phase baseline "
                    "are out of Layer B.1 scope."
                ),
            },
            {
                "clip": None,
                "where": "leave_interval / return_interval / censoring",
                "status": "not_emitted_by_this_tool",
                "reason": (
                    "Layer B.1 contract §9: leave/return intervals and "
                    "censoring flags are reserved for Layer C.1 only."
                ),
            },
            {
                "clip": None,
                "where": "combined_membership_score",
                "status": "forbidden_by_contract",
                "reason": (
                    "Layer B.1 reports per-feature-group reference scale "
                    "and degeneracy separately. No combined-membership / "
                    "combined-score is computed; turn_dyn is hard-rejected "
                    "at every entry point."
                ),
            },
            {
                "clip": None,
                "where": "event_head_target_status",
                "status": "not_emitted_by_this_tool",
                "reason": (
                    "Layer B.1 is read-only and produces no EventHead "
                    "target, no handoff_ready, no transition_done."
                ),
            },
            {
                "clip": None,
                "where": "transition_truth_promotion",
                "status": "forbidden_by_contract",
                "reason": (
                    "Reference scale / degeneracy tags MUST NOT be promoted "
                    "to transition truth, training labels, or runtime "
                    "switching behavior."
                ),
            },
            {
                "clip": None,
                "where": "cross_attractor_claim_status",
                "status": "forbidden_by_contract",
                "reason": (
                    "reference_family is locked to Walk_F. No Walk -> Run / "
                    "Walk -> Idle / etc claims are derivable from this "
                    "audit (scaffold v1 §2)."
                ),
            },
        ]
    )

    estimation_grid = {
        "feature_groups_layer_b1": feature_groups_meta,
        "epsilon_mad_grid_status_per_group": {
            name: "single_point_per_group_layerB1"
            for name in LAYER_B1_FEATURE_GROUPS
        },
        "min_frames_for_phase_estimability_estimator_only": (
            MIN_FRAMES_FOR_PHASE_ESTIMABILITY
        ),
        "mad_to_gaussian_sigma_constant": MAD_TO_GAUSSIAN_SIGMA,
        "note": (
            "Per-group epsilon_mad values are estimator-level numeric "
            "thresholds and are NOT contract definition. MAD / std / ptp "
            "/ mean_abs are always reported per channel so a reviewer can "
            "re-judge."
        ),
    }

    processed_data_root = str(raw_root / LAYER_B1_PROCESSED_DATA_SUBDIR)
    schema_pass = all(
        e["frame_alignment"]["aligned"]
        and all(
            e["processed_data_schema"]["arrays"][k]
            .get("finite_coverage", {})
            .get("all_finite", True)
            for k in ("bone_ang_vel", "bone_rot6d")
        )
        for e in per_clip
    )
    schema_status_reason = (
        "schema_pass_every_clip_aligned_and_finite"
        if schema_pass
        else "schema_fail_unexpected_post_validation_state"
    )

    summary = _common_summary_root(
        "pose_reference_scale_check", raw_root, clips, per_clip_paths
    )
    summary.update(
        {
            "layer": "Layer B.1",
            "layer_b1_contract_doc": (
                "docs/aperiodic_transition/"
                "2026-05-23_walk_f_causal_state_layerb1_pose_reference_scale_contract.md"
            ),
            "layer_b1_contract_status": "pass" if schema_pass else "fail",
            "expected_insufficient_evidence_is_contract_pass": True,
            "definition_layer": {
                "attractor_definition_status": "not_emitted_by_this_tool",
                "causal_state_definition_status": "not_emitted_by_this_tool",
                "current_reference_family": REFERENCE_FAMILY,
                "membership_evidence_status": "not_implemented_layerB1",
                "scope_caveat": (
                    "Walk_F is a single 88-frame trajectory (scaffold v1 "
                    "§4). Layer B.1 only emits per-feature-group reference "
                    "scale + degeneracy + processed-data schema audit. No "
                    "membership / boundary / predictive loss is computed."
                ),
            },
            "estimation_grid": estimation_grid,
            "included_feature_groups": list(LAYER_B1_FEATURE_GROUPS.keys()),
            "excluded_feature_groups": LAYER_B1_EXCLUDED_FEATURE_GROUPS,
            "feature_groups_meta": feature_groups_meta,
            "raw_root": str(raw_root),
            "processed_data_root": processed_data_root,
            "processed_data_schema_status": "pass" if schema_pass else "fail",
            "processed_data_schema_status_reason": schema_status_reason,
            "reference_family": REFERENCE_FAMILY,
            "query_family": list(LAYER_B1_QUERY_FAMILY),
            "reference_clip_names_resolved": [
                e["clip"] for e in reference_clips
            ],
            "query_clip_names": [e["clip"] for e in query_clips],
            "input_clip_policy": (
                "clips_must_set_equal_walk_f_plus_4_turn_clips_no_duplicates"
            ),
            "reference_scale_source": (
                "reference_clips_only"
                if reference_clips
                else "no_reference_clip_present"
            ),
            "reference_clip_status": "single_trajectory_reference_only",
            "reference_clip_status_reason": (
                "Walk_F is a single trajectory (scaffold v1 §4); "
                "class-level recurrence is INSUFFICIENT_EVIDENCE. Layer "
                "B.1 reports single-trajectory pose reference scale only."
            ),
            "per_group_reference_scale": reference_per_group,
            "phase_structure_status_per_group": phase_status_summary,
            "phase_structure_status_enum": [
                "phase_structured",
                "phase_degenerate",
                "insufficient_evidence",
            ],
            "phase_structure_status_enum_source": (
                "docs/aperiodic_transition/"
                "2026-05-22_walk_f_causal_state_scaffold_v1.md:255"
            ),
            "frame_alignment_per_clip": frame_alignment_per_clip,
            "per_query_clip_inspection": per_query_clip_inspection,
            "query_clip_inspection_only_note": (
                "Per-channel query raw scale numbers are present in the "
                "per-clip artifact JSONs for inspection only. They MUST "
                "NOT be aggregated into the Walk_F reference scale. "
                "Enforcement: _layer_b1_query_channel_walk_f_projected "
                "is the only normalization entry point on the query "
                "path and requires scale_source='walk_f'; any other "
                "value raises FailFastError via "
                "_layer_b1_forbid_query_self_rescale. No other code "
                "path in Layer B.1 normalises query channels."
            ),
            "walk_f_baseline": "single_trajectory_walk_f_pose_reference_scale_only",
            "sensitivity_summary": {
                "epsilon_mad_grid_status": "single_point_per_group_layerB1",
                "rotation_grid_sensitivity_status": "not_emitted_in_this_mode",
                "translation_grid_sensitivity_status": "not_emitted_in_this_mode",
                "predictive_loss_sensitivity_status": "not_implemented_layerB1",
                "yaw_activity_threshold_grid_status": "not_emitted_in_this_mode",
                "history_window_sensitivity_status": "not_emitted_in_this_mode",
                "future_horizon_sensitivity_status": "not_emitted_in_this_mode",
                "neighborhood_radius_sensitivity_status": (
                    "not_emitted_in_this_mode"
                ),
                "band_quantile_sensitivity_status": "not_emitted_in_this_mode",
            },
            "causal_state_track_status": (
                "pose_reference_scale_and_degeneracy_layer_b1"
            ),
            "quotient_definition_status": (
                "bone_local_frame_inherent_invariance_no_explicit_se2_reapply"
            ),
            "phase_library_status": (
                "not_implemented_layerB1_see_phase_library_check"
            ),
            "predictive_loss_status": (
                "not_implemented_layerB1_see_phase_library_check"
            ),
            "attractor_membership_status": "not_emitted_by_this_tool",
            "event_head_target_status": "not_emitted_by_this_tool",
            "handoff_ready_status": "not_emitted_by_this_tool",
            "transition_done_status": "not_emitted_by_this_tool",
            "cross_attractor_claim_status": "forbidden_by_contract",
            "transition_truth_promotion": "forbidden_by_contract",
            "combined_membership_score_status": (
                "forbidden_by_contract_groups_reported_separately"
            ),
            "turn_dyn_routing_policy": (
                "fail_fast_on_any_combined_membership_entry_point"
            ),
            "query_self_rescale_policy": (
                "load_bearing_scale_source_param_walk_f_only_fail_fast_otherwise"
            ),
            "query_self_rescale_policy_enforced_at": (
                "_layer_b1_query_channel_walk_f_projected.scale_source_kwarg"
            ),
            "insufficient_evidence": insufficient,
            "notes": [
                "Layer B.1 audits raw_data/processed_data/<clip>.npz "
                "schema + frame alignment + finite coverage and emits "
                "Walk_F-only per-feature-group reference scale + "
                "degeneracy for pose_dyn (bone_ang_vel) and pose_rel "
                "(bone_rot6d first-difference times FPS).",
                "bone_ang_vel and bone_rot6d are in bone-LOCAL frame per "
                "meta_json.spaces; the canonical SE(2) yaw quotient is "
                "NOT re-applied to these groups (doing so would double-"
                "count the local-frame transform).",
                "Walk_F is the only reference. Query clip per-channel "
                "raw scale is reported as inspection_only and never "
                "aggregated into the Walk_F reference scale.",
                "turn_dyn is hard-rejected at every Layer B.1 entry "
                "point via _layer_b1_forbid_turn_dyn.",
                "No combined-membership / combined-score is emitted; no "
                "leave/return interval is emitted; no attractor "
                "membership is emitted.",
            ],
        }
    )
    return summary


# -----------------------------------------------------------------------
# CLI dispatch
# -----------------------------------------------------------------------


def _parse_clip_list(arg: str) -> list[str]:
    parts = [s.strip() for s in arg.split(",") if s.strip()]
    if not parts:
        raise FailFastError(
            "[walk_f_causal_state_probe] --clips produced an empty list; "
            "refuse to run on no clips."
        )
    return parts


def _validate_mode(mode: str) -> str:
    if mode not in SUPPORTED_MODES:
        raise FailFastError(
            f"[walk_f_causal_state_probe] mode={mode!r} is not a supported "
            f"layer. Supported modes: {list(SUPPORTED_MODES)}. "
            "No silent fallback is allowed by docs/removal_policy.md §3-§4."
        )
    return mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Walk_F causal-state scaffold v1 read-only probe. "
            "Modes: yaw_debug (Layer 0 / Track 1, Walk-turn descriptive "
            "oracle on RootYaw); gauge_check (Layer A, canonical SE(2) "
            "yaw quotient + synthetic-rotation invariance + +/-pi wrap "
            "boundary + planar translation invariance); "
            "reference_scale_check (Layer B, Walk_F per-feature-group "
            "reference scale + degeneracy + contract-§6 "
            "phase_structure_status enum); phase_library_check (Layer C "
            "minimal, Walk_F-only leave-one-neighborhood self-consistency); "
            "query_boundary_check (Layer C.1, Walk_F self-baseline band + "
            "query leave/return interval + censoring audit on the 4 turn "
            "clips; emits NO attractor membership, NO EventHead target, NO "
            "handoff_ready, NO transition_done, NO cross-attractor claim, "
            "and rejects turn_dyn at every combined-membership entry point); "
            "pose_reference_scale_check (Layer B.1, processed_data NPZ "
            "schema + Walk_F-only pose_dyn / pose_rel reference scale + "
            "degeneracy + query inspection projected onto Walk_F scale; "
            "emits NO query leave/return, NO membership, NO combined score, "
            "NO transition truth, and rejects turn_dyn and query-self "
            "median/MAD rescaling at every entry point)."
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        help=(
            "yaw_debug | gauge_check | reference_scale_check | "
            "phase_library_check | query_boundary_check | "
            "pose_reference_scale_check. "
            "Unknown mode -> fail-fast (docs/removal_policy.md §3-§4)."
        ),
    )
    parser.add_argument(
        "--clips",
        required=True,
        type=_parse_clip_list,
        help="Comma-separated raw clip stems (matched to <raw_root>/<clip>.json).",
    )
    parser.add_argument(
        "--raw_root",
        required=True,
        type=Path,
        help="Directory containing raw_data JSON clips.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=Path,
        help="Directory to write artifacts into (created AFTER clip validation).",
    )

    args = parser.parse_args(argv)
    mode = _validate_mode(args.mode)

    if not args.raw_root.is_dir():
        raise FailFastError(
            f"[walk_f_causal_state_probe] --raw_root not a directory: "
            f"{args.raw_root}"
        )

    # Layer C.1 clip-set validation MUST happen BEFORE _load_clip and BEFORE
    # args.out_dir.mkdir so an out_dir is never created on a bad clip list.
    # The rules are:
    #   * --clips set-equal to {Walk_F + 4 turn clips}
    #   * order-independent
    #   * duplicates forbidden (Walk_F,Walk_F,... -> exit 2)
    # Each rule maps directly to one fail-fast smoke test in the Layer C.1
    # contract §9. No silent fallback or INSUFFICIENT_EVIDENCE fig-leaf is
    # allowed (docs/removal_policy.md §3-§4).
    if mode == "query_boundary_check":
        if len(args.clips) != len(set(args.clips)):
            duplicates = sorted(
                {c for c in args.clips if args.clips.count(c) > 1}
            )
            raise FailFastError(
                f"[walk_f_causal_state_probe] --mode query_boundary_check "
                f"received duplicate clip entries in --clips: {duplicates!r}. "
                "Layer C.1 contract §2 requires set-equality (no duplicates) "
                f"to {sorted(LAYER_C1_REQUIRED_CLIP_SET)!r}. Refusing to run."
            )
        if frozenset(args.clips) != LAYER_C1_REQUIRED_CLIP_SET:
            missing = sorted(LAYER_C1_REQUIRED_CLIP_SET - set(args.clips))
            extra = sorted(set(args.clips) - LAYER_C1_REQUIRED_CLIP_SET)
            raise FailFastError(
                f"[walk_f_causal_state_probe] --mode query_boundary_check "
                f"requires --clips set-equal to "
                f"{sorted(LAYER_C1_REQUIRED_CLIP_SET)!r} (Layer C.1 contract "
                f"§2). Got {args.clips!r}; missing={missing!r}, "
                f"extra={extra!r}. No silent fallback by "
                "docs/removal_policy.md §3-§4."
            )
    if mode == "pose_reference_scale_check":
        if len(args.clips) != len(set(args.clips)):
            duplicates = sorted(
                {c for c in args.clips if args.clips.count(c) > 1}
            )
            raise FailFastError(
                f"[walk_f_causal_state_probe] --mode pose_reference_scale_check "
                f"received duplicate clip entries in --clips: {duplicates!r}. "
                "Layer B.1 contract §2 requires set-equality (no duplicates) "
                f"to {sorted(LAYER_B1_REQUIRED_CLIP_SET)!r}. Refusing to run."
            )
        if frozenset(args.clips) != LAYER_B1_REQUIRED_CLIP_SET:
            missing = sorted(LAYER_B1_REQUIRED_CLIP_SET - set(args.clips))
            extra = sorted(set(args.clips) - LAYER_B1_REQUIRED_CLIP_SET)
            raise FailFastError(
                f"[walk_f_causal_state_probe] --mode pose_reference_scale_check "
                f"requires --clips set-equal to "
                f"{sorted(LAYER_B1_REQUIRED_CLIP_SET)!r} (Layer B.1 contract "
                f"§2). Got {args.clips!r}; missing={missing!r}, "
                f"extra={extra!r}. No silent fallback by "
                "docs/removal_policy.md §3-§4."
            )

    # Load + validate ALL clips before touching the output directory.
    # This avoids leaving an empty out_dir on a failing raw-data sanity.
    clip_infos = [_load_clip(args.raw_root, clip) for clip in args.clips]

    # Mode-specific pre-mkdir sanity. reference_scale_check requires
    # the reference_family (Walk_F) to be present in --clips; otherwise
    # the run would silently emit INSUFFICIENT_EVIDENCE everywhere.
    if mode == "reference_scale_check":
        missing_reference = [r for r in REFERENCE_FAMILY if r not in args.clips]
        if missing_reference:
            raise FailFastError(
                f"[walk_f_causal_state_probe] --mode reference_scale_check "
                f"requires reference_family clips {REFERENCE_FAMILY} to be "
                f"present in --clips; missing: {missing_reference}. "
                "No silent INSUFFICIENT_EVIDENCE fallback by §1.1 + "
                "docs/removal_policy.md §3-§4."
            )
    if mode == "phase_library_check":
        if args.clips != REFERENCE_FAMILY:
            raise FailFastError(
                f"[walk_f_causal_state_probe] --mode phase_library_check "
                f"requires --clips exactly {REFERENCE_FAMILY}; got {args.clips!r}. "
                "Layer C minimal is Walk_F self-consistency only. Query "
                "leave/return/censoring clips are reserved for Layer C.1; no "
                "silent membership-boundary expansion is allowed."
            )
        internal_se2_precondition = _build_internal_se2_precondition(clip_infos)
        if internal_se2_precondition["status"] != "pass":
            raise FailFastError(
                "[walk_f_causal_state_probe] --mode phase_library_check internal "
                "SE(2) gauge precondition did not pass; refusing to run Layer C "
                "self-consistency because quotient invariance is a prerequisite. "
                f"status={internal_se2_precondition['status']!r}"
            )
    elif mode == "query_boundary_check":
        internal_se2_precondition = _build_internal_se2_precondition(clip_infos)
        if internal_se2_precondition["status"] != "pass":
            raise FailFastError(
                "[walk_f_causal_state_probe] --mode query_boundary_check internal "
                "SE(2) gauge precondition did not pass; refusing to run Layer C.1 "
                "query boundary audit because quotient invariance is a "
                "prerequisite. "
                f"status={internal_se2_precondition['status']!r}"
            )
    else:
        internal_se2_precondition = None

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "yaw_debug":
        per_clip = [_process_clip_yaw_debug(info) for info in clip_infos]
        per_clip_paths = _write_per_clip(args.out_dir, per_clip, "yaw_activity_debug")
        summary = _build_summary_yaw_debug(args.raw_root, args.clips, per_clip, per_clip_paths)
    elif mode == "gauge_check":
        per_clip = [_process_clip_gauge_check(info) for info in clip_infos]
        per_clip_paths = _write_per_clip(args.out_dir, per_clip, "gauge_check")
        summary = _build_summary_gauge_check(args.raw_root, args.clips, per_clip, per_clip_paths)
    elif mode == "reference_scale_check":
        # Layer B needs the FootEvidence.{L,R}.soft_contact_score channel.
        # Fail-fast on missing contact for ANY clip, including query
        # clips — we refuse to silently load partial data.
        contact_by_clip = {
            info["clip"]: _load_clip_contact_signals(args.raw_root, info["clip"])
            for info in clip_infos
        }
        per_clip = [
            _process_clip_reference_scale_check(
                info,
                contact_by_clip[info["clip"]],
                is_reference=info["clip"] in REFERENCE_FAMILY,
            )
            for info in clip_infos
        ]
        per_clip_paths = _write_per_clip(args.out_dir, per_clip, "reference_scale_check")
        summary = _build_summary_reference_scale_check(
            args.raw_root, args.clips, per_clip, per_clip_paths
        )
    elif mode == "phase_library_check":
        contact_by_clip = {
            info["clip"]: _load_clip_contact_signals(args.raw_root, info["clip"])
            for info in clip_infos
        }
        per_clip = [
            _process_clip_phase_library_check(
                info,
                contact_by_clip[info["clip"]],
            )
            for info in clip_infos
        ]
        per_clip_paths = _write_per_clip(args.out_dir, per_clip, "phase_library_check")
        if internal_se2_precondition is None:
            raise FailFastError(
                "[walk_f_causal_state_probe] internal SE(2) precondition missing "
                "for phase_library_check; refusing to emit artifact."
            )
        summary = _build_summary_phase_library_check(
            args.raw_root,
            args.clips,
            per_clip,
            per_clip_paths,
            internal_se2_precondition,
        )
    elif mode == "pose_reference_scale_check":
        npz_by_clip: dict[str, dict[str, Any]] = {}
        for info in clip_infos:
            npz_by_clip[info["clip"]] = _load_clip_processed_data(
                args.raw_root,
                info["clip"],
                json_frame_count=int(info["frame_count"]),
                json_fps=float(info["fps"]),
            )
        # Build Walk_F reference scale FIRST so query clips can reuse it.
        walk_f_info_list = [
            info for info in clip_infos
            if info["clip"] == LAYER_B1_REFERENCE_CLIP
        ]
        if len(walk_f_info_list) != 1:
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer B.1 expected exactly one "
                f"Walk_F clip after CLI validation; got {len(walk_f_info_list)}. "
                "This indicates the CLI clip-set guard was bypassed."
            )
        walk_f_info = walk_f_info_list[0]
        walk_f_per_clip_entry = _process_clip_pose_reference_scale_check(
            clip_info=walk_f_info,
            npz_payload=npz_by_clip[walk_f_info["clip"]],
            is_reference=True,
            walk_f_reference_scale_per_group=None,
        )
        walk_f_reference_scale_per_group = walk_f_per_clip_entry[
            "feature_groups_layer_b1"
        ]
        per_clip = []
        for info in clip_infos:
            if info["clip"] == LAYER_B1_REFERENCE_CLIP:
                per_clip.append(walk_f_per_clip_entry)
            else:
                per_clip.append(
                    _process_clip_pose_reference_scale_check(
                        clip_info=info,
                        npz_payload=npz_by_clip[info["clip"]],
                        is_reference=False,
                        walk_f_reference_scale_per_group=(
                            walk_f_reference_scale_per_group
                        ),
                    )
                )
        per_clip_paths = _write_per_clip(
            args.out_dir, per_clip, "pose_reference_scale_check"
        )
        summary = _build_summary_pose_reference_scale_check(
            args.raw_root, args.clips, per_clip, per_clip_paths
        )
    elif mode == "query_boundary_check":
        contact_by_clip = {
            info["clip"]: _load_clip_contact_signals(args.raw_root, info["clip"])
            for info in clip_infos
        }
        walk_f_info_list = [
            info for info in clip_infos if info["clip"] == LAYER_C1_REFERENCE_CLIP
        ]
        if len(walk_f_info_list) != 1:
            raise FailFastError(
                f"[walk_f_causal_state_probe] Layer C.1 expected exactly one "
                f"Walk_F clip after CLI validation; got {len(walk_f_info_list)}. "
                "This indicates the CLI clip-set guard was bypassed."
            )
        walk_f_info = walk_f_info_list[0]
        walk_f_group_payloads = _build_walk_f_layer_c_group_payloads(
            walk_f_clip_info=walk_f_info,
            walk_f_contact_signals=contact_by_clip[walk_f_info["clip"]],
        )
        per_clip = [
            _process_query_clip_boundary_check(
                query_clip_info=info,
                query_contact_signals=contact_by_clip[info["clip"]],
                walk_f_group_payloads=walk_f_group_payloads,
            )
            for info in clip_infos
            if info["clip"] in LAYER_C1_QUERY_FAMILY
        ]
        per_clip_paths = _write_per_clip(
            args.out_dir, per_clip, "query_boundary_check"
        )
        if internal_se2_precondition is None:
            raise FailFastError(
                "[walk_f_causal_state_probe] internal SE(2) precondition missing "
                "for query_boundary_check; refusing to emit artifact."
            )
        summary = _build_summary_query_boundary_check(
            args.raw_root,
            args.clips,
            walk_f_group_payloads,
            per_clip,
            per_clip_paths,
            internal_se2_precondition,
        )
    else:
        # Unreachable; _validate_mode already raises. Kept as defense in depth.
        raise FailFastError(
            f"[walk_f_causal_state_probe] unhandled mode={mode!r}; refuse "
            "to silently fall through."
        )

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(_serializable(summary), indent=2), encoding="utf-8")

    print(f"[walk_f_causal_state_probe] contract={CONTRACT}")
    print(f"[walk_f_causal_state_probe] mode={mode} track={summary['track']} role={summary['track_role']}")
    print(f"[walk_f_causal_state_probe] summary={summary_path}")
    if mode == "yaw_debug":
        for entry in per_clip:
            print(
                f"[walk_f_causal_state_probe] clip={entry['clip']} "
                f"frames={entry['frame_count']} "
                f"total_yaw_delta_deg={entry['summary']['total_yaw_delta_deg']:.3f} "
                f"peak_abs_yaw_rate_deg_per_s={entry['summary']['peak_abs_yaw_rate_deg_per_s']:.3f} "
                f"peak_frame={entry['summary']['peak_abs_yaw_rate_frame']} "
                f"direction={entry['summary']['turn_direction_by_yaw']} "
                f"window_stability={entry['window_stability_summary']['status']}"
            )
    elif mode == "reference_scale_check":
        print(
            f"[walk_f_causal_state_probe] reference_family={REFERENCE_FAMILY} "
            f"reference_clip_names_resolved={summary['reference_clip_names_resolved']} "
            f"query_clip_names={summary['query_clip_names']} "
            f"reference_scale_source={summary['reference_scale_source']} "
            f"reference_clip_status={summary['reference_clip_status']}"
        )
        for name, st in summary["phase_structure_status_per_group"].items():
            print(
                f"[walk_f_causal_state_probe] group={name} "
                f"phase_structure_status={st['phase_structure_status']} "
                f"layer_c_candidate={st['layer_c_candidate']} "
                f"reference_degenerate={st['group_reference_degenerate']} "
                f"max_channel_mad={st['group_max_channel_mad']!r} "
                f"epsilon_mad_estimator_only={st['epsilon_mad_estimator_only']:.3e} "
                f"degenerate_channels={st['channel_degenerate_count']}/{st['channel_total_count']}"
            )
        for entry in per_clip:
            print(
                f"[walk_f_causal_state_probe] clip={entry['clip']} "
                f"frames={entry['frame_count']} "
                f"role={entry['clip_role']}"
            )
    elif mode == "phase_library_check":
        print(
            f"[walk_f_causal_state_probe] reference_family={REFERENCE_FAMILY} "
            f"reference_clip_names_resolved={summary['reference_clip_names_resolved']} "
            f"query_clip_names={summary['query_clip_names']} "
            f"layer_c_contract_status={summary['layer_c_contract_status']} "
            "expected_insufficient_evidence_is_contract_pass="
            f"{summary['expected_insufficient_evidence_is_contract_pass']}"
        )
        print(
            f"[walk_f_causal_state_probe] internal_se2_precondition="
            f"{summary['internal_se2_gauge_precondition']['status']} "
            f"max_abs_error_overall="
            f"{summary['internal_se2_gauge_precondition']['se2_gauge_sanity']['max_abs_error_overall']:.3e}"
        )
        for name, st in summary["phase_structure_status_per_group"].items():
            print(
                f"[walk_f_causal_state_probe] group={name} "
                f"phase_structure_status={st['phase_structure_status']} "
                f"evidence_status={st['evidence_status']} "
                f"self_consistency_signal_status={st['self_consistency_signal_status']} "
                f"configs_beating_baseline={st['configs_beating_baseline_count']}/"
                f"{st['config_count']} "
                f"reference_degenerate={st['reference_degenerate']} "
                f"active_channels={st['active_channels']}"
            )
    elif mode == "pose_reference_scale_check":
        print(
            f"[walk_f_causal_state_probe] reference_family={REFERENCE_FAMILY} "
            f"query_family={summary['query_family']} "
            f"layer_b1_contract_status={summary['layer_b1_contract_status']} "
            f"processed_data_schema_status="
            f"{summary['processed_data_schema_status']} "
            "expected_insufficient_evidence_is_contract_pass="
            f"{summary['expected_insufficient_evidence_is_contract_pass']}"
        )
        for name, st in summary["phase_structure_status_per_group"].items():
            print(
                f"[walk_f_causal_state_probe] group={name} "
                f"phase_structure_status={st['phase_structure_status']} "
                f"reference_degenerate={st['group_reference_degenerate']} "
                f"layer_c_candidate={st['layer_c_candidate']} "
                f"channel_degenerate={st['channel_degenerate_count']}/"
                f"{st['channel_total_count']} "
                f"group_max_channel_mad={st['group_max_channel_mad']:.3e} "
                f"epsilon_mad={st['epsilon_mad_estimator_only']:.3e}"
            )
        for entry in per_clip:
            print(
                f"[walk_f_causal_state_probe] clip={entry['clip']} "
                f"frames={entry['frame_count']} "
                f"role={entry['clip_role']} "
                f"npz={entry['npz_path']}"
            )
    elif mode == "query_boundary_check":
        print(
            f"[walk_f_causal_state_probe] reference_family={REFERENCE_FAMILY} "
            f"query_family={summary['query_family']} "
            f"layer_c1_contract_status={summary['layer_c1_contract_status']} "
            "expected_insufficient_evidence_is_contract_pass="
            f"{summary['expected_insufficient_evidence_is_contract_pass']}"
        )
        print(
            f"[walk_f_causal_state_probe] internal_se2_precondition="
            f"{summary['internal_se2_gauge_precondition']['status']} "
            f"max_abs_error_overall="
            f"{summary['internal_se2_gauge_precondition']['se2_gauge_sanity']['max_abs_error_overall']:.3e}"
        )
        for entry in per_clip:
            clip_name = entry["clip"]
            for group_name, group_payload in entry[
                "feature_groups_layer_c1"
            ].items():
                if bool(group_payload["reference_degenerate"]):
                    print(
                        f"[walk_f_causal_state_probe] clip={clip_name} "
                        f"group={group_name} "
                        "phase_structure_status=phase_degenerate "
                        "boundary_detection=skipped"
                    )
                    continue
                print(
                    f"[walk_f_causal_state_probe] clip={clip_name} "
                    f"group={group_name} "
                    f"returned={group_payload['returned_config_count']} "
                    f"never_left={group_payload['never_left_config_count']} "
                    f"never_returned={group_payload['never_returned_config_count']} "
                    f"insufficient={group_payload['insufficient_evidence_config_count']} "
                    f"left_censored={group_payload['left_censored_leave_config_count']} "
                    f"right_censored={group_payload['right_censored_return_config_count']} "
                    "neighbor_conflicts="
                    f"{group_payload['neighbor_consistency_conflict_pair_count']}"
                )
        print(
            "[walk_f_causal_state_probe] censoring_summary "
            f"left_censored_total={summary['censoring_summary']['left_censored_leave_total']} "
            f"right_censored_total={summary['censoring_summary']['right_censored_return_total']} "
            "expected_insufficient_evidence_is_contract_pass="
            f"{summary['censoring_summary']['expected_insufficient_evidence_is_contract_pass']}"
        )
    else:
        print(
            f"[walk_f_causal_state_probe] gauge_kind={summary['gauge_kind']}"
        )
        print(
            f"[walk_f_causal_state_probe] yaw_invariance_sanity="
            f"{summary['yaw_invariance_sanity']['status']} "
            f"max_abs_error_overall={summary['yaw_invariance_sanity']['max_abs_error_overall']:.3e} "
            f"min_finite_coverage_fraction={summary['yaw_invariance_sanity']['min_finite_coverage_fraction']:.6f} "
            f"tol_estimator_only={summary['yaw_invariance_sanity']['tol_max_abs_error_estimator_only']:.3e}"
        )
        print(
            f"[walk_f_causal_state_probe] translation_invariance_sanity="
            f"{summary['translation_invariance_sanity']['status']} "
            f"max_abs_error_overall={summary['translation_invariance_sanity']['max_abs_error_overall']:.3e} "
            f"min_finite_coverage_fraction={summary['translation_invariance_sanity']['min_finite_coverage_fraction']:.6f}"
        )
        print(
            f"[walk_f_causal_state_probe] se2_gauge_sanity="
            f"{summary['se2_gauge_sanity']['status']} "
            f"max_abs_error_overall={summary['se2_gauge_sanity']['max_abs_error_overall']:.3e} "
            f"min_finite_coverage_fraction={summary['se2_gauge_sanity']['min_finite_coverage_fraction']:.6f}"
        )
        for entry in per_clip:
            print(
                f"[walk_f_causal_state_probe] clip={entry['clip']} "
                f"frames={entry['frame_count']} "
                f"max_err_rot={entry['max_abs_error_over_rotation_grid']:.3e} "
                f"max_err_trans={entry['max_abs_error_over_translation_grid']:.3e} "
                f"max_err_se2={entry['max_abs_error_over_se2_grid']:.3e} "
                f"rot_coverage={entry['rotation_finite_coverage']['coverage_fraction']:.6f} "
                f"trans_coverage={entry['translation_finite_coverage']['coverage_fraction']:.6f} "
                f"wrap_cases={len(entry['wrap_boundary_cases'])}"
            )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FailFastError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
