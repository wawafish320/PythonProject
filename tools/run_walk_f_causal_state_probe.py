#!/usr/bin/env python3
"""Walk_F causal-state scaffold v1 — read-only probes.

Two modes are implemented, both read-only:

* ``yaw_debug`` (Layer 0 / Track 1) — Walk-turn descriptive oracle on
  raw ``RootYaw``. Debug-only; must not be promoted to attractor /
  transition truth.
* ``gauge_check`` (Layer A) — canonical-yaw-quotient + synthetic
  global-yaw rotation grid + ``+/-pi`` wrap boundary, as required by
  ``docs/aperiodic_transition/2026-05-22_walk_f_causal_state_scaffold_v1.md``
  §1.2. Prerequisite for the future causal-state membership track.
  Does NOT estimate causal-state membership, attractor membership,
  phase library, predictive loss, or EventHead targets.
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

SUPPORTED_MODES = ("yaw_debug", "gauge_check", "reference_scale_check")

TRACK_BY_MODE = {
    "yaw_debug": "yaw_activity_debug",
    "gauge_check": "walk_f_causal_state_scaffold_prerequisite",
    "reference_scale_check": "walk_f_reference_scale_and_degeneracy",
}
TRACK_ROLE_BY_MODE = {
    "yaw_debug": "debug_only_not_attractor_definition",
    "gauge_check": "gauge_sanity_prerequisite_not_attractor_definition",
    "reference_scale_check": "reference_scale_and_degeneracy_prerequisite_not_membership",
}
SCOPE_BY_MODE = {
    "yaw_debug": "walk_turn_yaw_activity_descriptive_oracle",
    "gauge_check": "canonical_yaw_quotient_invariance_sanity",
    "reference_scale_check": "walk_f_per_feature_group_reference_scale_and_degeneracy",
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
            "phase_structure_status enum). NONE of these modes estimate "
            "causal-state membership, attractor membership, phase "
            "library, predictive loss, or EventHead targets."
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        help=(
            "yaw_debug | gauge_check | reference_scale_check. "
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
