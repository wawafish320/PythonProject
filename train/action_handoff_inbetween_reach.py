from __future__ import annotations

"""Action-handoff in-betweening — B1 reach metric in hidden_pre space (§7.3, 3b foundation).

Path A+B (per the plan + the 3b feasibility finding): the binding B1 `reach_rate` is
defined on the base model's `hidden_pre` (512-d) rather than `z`, because (i) the trained
z-head (ZEncoder) was never persisted — only z VALUES were saved — and (ii) the audit A1
result shows hidden_pre carries the same regime/contact information as z (decodability
R² 0.80 vs z 0.84). Using hidden_pre avoids re-deriving an unsaved z-head and still gives
the spec §6 reach metric: `cos_dist(hidden_pre_t, anchor) ≤ CONV_DIST`.

This module owns ONLY the reach metric + target anchor (the measurement half of the
binding gate). It does NOT build/train the goal-conditioned generator and does NOT run the
base model — that is the remaining Slice-2 work (extend `run_freerun_cycles`: base
EventMotionModel init-from-ckpt + goal head + free-run + hidden_pre capture). Anchors are
built from the FROZEN `z_features_per_clip.npz` `{clip}__hidden_pre`; reach on a generated
rollout's captured hidden_pre is computed with `reach_min_norm` / `reached`.

All thresholds PROVISIONAL. Mirrors the audit A2 anchor geometry (end-window centroid +
radius, cosine), but in hidden_pre space instead of z.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

EPS = 1e-8

HIDDEN_DIM = 512
DEFAULT_END_WINDOW_K = 12  # [PROVISIONAL] anchor = mean of the target clip's last K frames
DEFAULT_CONV_NORM_THR = 1.5  # [PROVISIONAL] reach iff min cos_dist ≤ thr × anchor_radius (A2 posture)
DEFAULT_SELF_REACH_K = 3.0  # [PROVISIONAL] binding reach iff abs cos ≤ k × same-model self floor
DEFAULT_SELF_REACH_K_VALUES = (2.0, 3.0, 5.0)
ANCHOR_DIFFUSE_THR = 0.80  # [PROVISIONAL] anchor ill-defined if radius/clip_spread ≥ this (A2)
RADIUS_FLOOR = 1e-6  # below this the anchor radius is degenerate → normalization is unreliable
SELF_REACH_FLOOR = 1e-12  # absolute cos-distance floor for ratio diagnostics only

LOCKED_CLIPS = (
    "Walk_F",
    "Walk_L_To_L",
    "Walk_L_To_R",
    "Walk_R_To_L",
    "Walk_R_To_R",
)
TURN_CLIPS = LOCKED_CLIPS[1:]


def _validate_hidden(arr: np.ndarray, name: str) -> np.ndarray:
    """Fail-fast: hidden_pre must be a finite [T,512] array with T≥1."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != HIDDEN_DIM:
        raise ValueError(f"{name}: expected hidden_pre [T,{HIDDEN_DIM}], got {tuple(np.shape(arr))}")
    if a.shape[0] < 1:
        raise ValueError(f"{name}: empty hidden_pre (T=0)")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: hidden_pre contains non-finite values")
    return a


def cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 − cosine similarity between rows of `a` and rows of `b` → [na, nb]."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    if a64.ndim == 1:
        a64 = a64[None, :]
    if b64.ndim == 1:
        b64 = b64[None, :]
    an = a64 / np.maximum(np.linalg.norm(a64, axis=1, keepdims=True), EPS)
    bn = b64 / np.maximum(np.linalg.norm(b64, axis=1, keepdims=True), EPS)
    return 1.0 - np.clip(an @ bn.T, -1.0, 1.0)


def k_label(k: float) -> str:
    """Stable label for self-reach multiplier tables."""
    v = float(k)
    return f"k={int(v)}" if v.is_integer() else f"k={v:g}"


@dataclass
class HiddenPreAnchor:
    """A target regime's hidden_pre attractor (region/convergence only — D3)."""

    clip: str
    centroid: np.ndarray  # [512]
    radius: float  # mean cos_dist of the end-window frames to the centroid
    clip_spread: float  # mean cos_dist of all clip frames to the clip centroid
    diffuseness: float  # radius / clip_spread
    end_window_k: int
    well_defined: bool
    radius_degenerate: bool = False  # radius < RADIUS_FLOOR → normalization unreliable

    def min_norm(self, hidden_seq: np.ndarray) -> float:
        """Min cos_dist(any frame, centroid) normalized by the anchor radius.

        `hidden_seq` is fail-fast validated as a finite [T,512] rollout; a degenerate
        radius is floored (RADIUS_FLOOR) so normalization cannot blow up.
        """
        seq = _validate_hidden(hidden_seq, "min_norm hidden_seq")
        d = cos_dist(seq, self.centroid).reshape(-1)
        return float(np.min(d) / max(self.radius, RADIUS_FLOOR))

    def min_abs_cos(self, hidden_seq: np.ndarray) -> float:
        """Minimum absolute cosine distance to the centroid over a finite [T,512] sequence."""
        seq = _validate_hidden(hidden_seq, "min_abs_cos hidden_seq")
        d = cos_dist(seq, self.centroid).reshape(-1)
        return float(np.min(d))

    def reached(self, hidden_seq: np.ndarray, conv_norm_thr: float = DEFAULT_CONV_NORM_THR) -> bool:
        """Reach iff the rollout's closest hidden_pre is within conv_norm_thr × radius."""
        return bool(self.min_norm(hidden_seq) <= conv_norm_thr)

    def reached_self_abs(
        self,
        hidden_seq: np.ndarray,
        *,
        self_reach_abs_cos: float,
        k: float = DEFAULT_SELF_REACH_K,
    ) -> bool:
        """Reach iff min absolute cos distance is within k × same-model self-reach floor."""
        return absolute_self_reach_reached(
            self.min_abs_cos(hidden_seq),
            self_reach_abs_cos=float(self_reach_abs_cos),
            k=float(k),
        )


@dataclass
class SameSourceAnchorDiagnostics:
    """Diagnostics proving whether a rebuilt hidden_pre anchor is usable for one model."""

    clip: str
    anchor_hidden_shape: Tuple[int, int]
    self_check_shape: Tuple[int, int]
    self_reach_abs_cos: float
    self_check_abs_cos: float
    self_check_margin_vs_k: float
    self_check_reached: bool
    reach_available: bool
    reason: str


def absolute_self_reach_reached(
    generated_abs_cos: float,
    *,
    self_reach_abs_cos: float,
    k: float = DEFAULT_SELF_REACH_K,
) -> bool:
    """Pure absolute self-reach predicate.

    ``generated_abs_cos`` and ``self_reach_abs_cos`` are scalar cosine distances in the
    same hidden_pre space. The predicate is finite-only: non-finite inputs never pass.
    """
    g = float(generated_abs_cos)
    s = float(self_reach_abs_cos)
    kk = float(k)
    if not (np.isfinite(g) and np.isfinite(s) and np.isfinite(kk)):
        return False
    if kk < 0.0:
        raise ValueError(f"k must be non-negative, got {k}")
    return bool(g <= kk * max(s, SELF_REACH_FLOOR))


def summarize_absolute_self_reach(
    generated_abs_cos: Sequence[float],
    *,
    self_reach_abs_cos: float,
    k_values: Sequence[float] = DEFAULT_SELF_REACH_K_VALUES,
) -> Dict[str, object]:
    """Rate table for the migrated absolute self-reach gate.

    Input values are absolute cosine distances, not radius-normalized distances. Output
    keeps k=2/3/5 side bands so the old radius gate can remain a diagnostic only.
    """
    vals = np.asarray([float(v) for v in generated_abs_cos], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    floor = float(self_reach_abs_cos)
    out: Dict[str, object] = {
        "self_reach_abs_cos": floor,
        "n": int(vals.size),
        "finite_n": int(finite.size),
        "rate_by_k": {},
        "count_by_k": {},
        "threshold_abs_by_k": {},
        "margin_mean_by_k": {},
        "margin_min_by_k": {},
    }
    for k in k_values:
        label = k_label(float(k))
        threshold = float(k) * max(floor, SELF_REACH_FLOOR)
        out["threshold_abs_by_k"][label] = threshold
        if finite.size:
            passed = finite <= threshold
            margins = finite / max(threshold, SELF_REACH_FLOOR)
            out["rate_by_k"][label] = float(np.mean(passed))
            out["count_by_k"][label] = int(np.sum(passed))
            out["margin_mean_by_k"][label] = float(np.mean(margins))
            out["margin_min_by_k"][label] = float(np.min(margins))
        else:
            out["rate_by_k"][label] = float("nan")
            out["count_by_k"][label] = 0
            out["margin_mean_by_k"][label] = float("nan")
            out["margin_min_by_k"][label] = float("nan")
    return out


def build_same_source_hidden_pre_anchors(
    anchor_hidden_by_clip: Mapping[str, np.ndarray],
    *,
    self_check_hidden_by_clip: Mapping[str, np.ndarray] | None = None,
    turn_clips: Sequence[str] = TURN_CLIPS,
    end_window_k: int = DEFAULT_END_WINDOW_K,
    diffuse_thr: float = ANCHOR_DIFFUSE_THR,
    self_reach_k: float = DEFAULT_SELF_REACH_K,
) -> Tuple[Dict[str, HiddenPreAnchor], Dict[str, SameSourceAnchorDiagnostics]]:
    """Build anchors from the evaluated model's own hidden_pre capture.

    ``anchor_hidden_by_clip`` is the same-model full-sequence hidden_pre capture used to
    rebuild each anchor. ``self_check_hidden_by_clip`` is normally the same model's
    context-window/teacher self capture, i.e. the capture path used by rollout metrics.
    If the self-check is farther than ``self_reach_k × self_reach_abs_cos``, reach is
    marked unavailable for that model/clip instead of silently using a stale anchor.
    """
    anchors = build_hidden_pre_anchors(
        dict(anchor_hidden_by_clip),
        turn_clips=turn_clips,
        end_window_k=end_window_k,
        diffuse_thr=diffuse_thr,
    )
    diagnostics: Dict[str, SameSourceAnchorDiagnostics] = {}
    for clip in turn_clips:
        anchor = anchors[clip]
        anchor_hidden = _validate_hidden(anchor_hidden_by_clip[clip], f"same-source anchor '{clip}'")
        check_src = (
            self_check_hidden_by_clip[clip]
            if self_check_hidden_by_clip is not None and clip in self_check_hidden_by_clip
            else anchor_hidden
        )
        self_check = _validate_hidden(check_src, f"same-source self-check '{clip}'")
        self_floor = float(anchor.min_abs_cos(anchor_hidden))
        check_abs = float(anchor.min_abs_cos(self_check))
        threshold = float(self_reach_k) * max(self_floor, SELF_REACH_FLOOR)
        check_reached = bool(check_abs <= threshold)
        reach_available = bool(anchor.well_defined and check_reached)
        if not anchor.well_defined:
            reason = "anchor ill-defined for this model (diffuse or radius-degenerate)"
        elif not check_reached:
            reason = "anchor is miscalibrated for this model; reach unavailable"
        else:
            reason = "same-source anchor self-check passed"
        diagnostics[clip] = SameSourceAnchorDiagnostics(
            clip=clip,
            anchor_hidden_shape=(int(anchor_hidden.shape[0]), int(anchor_hidden.shape[1])),
            self_check_shape=(int(self_check.shape[0]), int(self_check.shape[1])),
            self_reach_abs_cos=self_floor,
            self_check_abs_cos=check_abs,
            self_check_margin_vs_k=float(check_abs / max(threshold, SELF_REACH_FLOOR)),
            self_check_reached=check_reached,
            reach_available=reach_available,
            reason=reason,
        )
    return anchors, diagnostics


def build_hidden_pre_anchors(
    hidden_by_clip: Dict[str, np.ndarray],
    turn_clips: Sequence[str] = TURN_CLIPS,
    end_window_k: int = DEFAULT_END_WINDOW_K,
    diffuse_thr: float = ANCHOR_DIFFUSE_THR,
) -> Dict[str, HiddenPreAnchor]:
    """Build per-turn-clip hidden_pre anchors (end-window centroid + radius)."""
    anchors: Dict[str, HiddenPreAnchor] = {}
    for clip in turn_clips:
        h = _validate_hidden(hidden_by_clip[clip], f"anchor clip '{clip}'")
        t = h.shape[0]
        k = int(min(end_window_k, t))
        end = h[t - k :]
        centroid = end.mean(axis=0)
        radius = float(cos_dist(end, centroid).mean())
        clip_centroid = h.mean(axis=0)
        clip_spread = float(cos_dist(h, clip_centroid).mean())
        diffuseness = float(radius / max(clip_spread, EPS))
        radius_degenerate = bool(radius < RADIUS_FLOOR)
        anchors[clip] = HiddenPreAnchor(
            clip=clip,
            centroid=centroid,
            radius=radius,
            clip_spread=clip_spread,
            diffuseness=diffuseness,
            end_window_k=k,
            # a degenerate (near-zero) radius makes normalized reach unreliable → not usable
            well_defined=bool(diffuseness < diffuse_thr and not radius_degenerate),
            radius_degenerate=radius_degenerate,
        )
    return anchors


def load_hidden_pre(
    z_features: str | Path, clips: Sequence[str] = LOCKED_CLIPS
) -> Dict[str, np.ndarray]:
    """Load `{clip}__hidden_pre` [T,512] from the frozen z-probe artifact."""
    z = np.load(Path(z_features), allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for clip in clips:
        key = f"{clip}__hidden_pre"
        if key not in z.files:
            raise RuntimeError(f"missing hidden_pre key: {key}")
        out[clip] = _validate_hidden(z[key], key)
    return out
