from __future__ import annotations

"""Action-handoff goal-conditioned in-betweening — data/sampling pipeline (§7.2).

Pure data pipeline for the goal-conditioned in-betweening generator described in
`docs/aperiodic_transition/2026-05-29_goal_conditioned_inbetweening_spec.md`. This
module owns the SINGLE source of truth for:

  - the per-frame egocentric (heading-invariant) state `s_t`, D_s = 281
    (spec §1.1): pose_rot6d[276] + root_vel_ego[2] + yaw_rate[1] + contact[2];
  - the full-state Walk_F alignment used for grounded cross-manifold sampling
    (spec §2b / §7.1): pose localizes the cycle-phase neighborhood, contact
    refines within it (ego_vel is phase-flat, yaw_rate≈0 at onset — neither
    disambiguates near onset, matching the re-entry resolver design D3);
  - the groundability gate (spec §2b) deciding whether a turn clip's onset is a
    clean grounded source;
  - the three-type in-betweening sampler (spec §2): (a) within-clip biased gap
    with longer-gap curriculum, (b) grounded cross-manifold with Walk_L_To_R
    fallback, (c) start-state augmentation.

This step produces TENSORS ONLY. It does NOT initialize a model, define a loss,
run scheduled sampling, or touch any base-model checkpoint (spec §0 / staging
§7.2 — model + losses are §7.3). The diagnostic tool
`tools/run_action_handoff_grounded_alignment_check.py` imports the state /
alignment helpers here so the diagnostic and the sampler can never drift apart.

All numeric thresholds are PROVISIONAL (spec preamble): calibrate before any
production use; only the §7.1 alignment outcomes (R_L/L_R/L_L/R_R gate verdicts)
are locked acceptance facts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- schema
FPS: float = 60.0  # [FIXED] spec §0

POSE_DIM = 276  # bone_rot6d.reshape(t, 46*6)
EGO_VEL_DIM = 2  # (forward, lateral)
YAW_RATE_DIM = 1  # heading angular rate, rad/s
CONTACT_DIM = 2  # (right, left)
STATE_DIM = POSE_DIM + EGO_VEL_DIM + YAW_RATE_DIM + CONTACT_DIM  # 281

# Egocentric state channel slices (the canonical 281-d layout).
POSE_SLICE = slice(0, POSE_DIM)  # [0, 276)
EGO_VEL_SLICE = slice(POSE_DIM, POSE_DIM + EGO_VEL_DIM)  # [276, 278)
YAW_RATE_SLICE = slice(POSE_DIM + EGO_VEL_DIM, POSE_DIM + EGO_VEL_DIM + YAW_RATE_DIM)  # [278, 279)
CONTACT_SLICE = slice(STATE_DIM - CONTACT_DIM, STATE_DIM)  # [279, 281)

LOCKED_CLIPS = (
    "Walk_F",
    "Walk_L_To_L",
    "Walk_L_To_R",
    "Walk_R_To_L",
    "Walk_R_To_R",
)
TURN_CLIPS = (
    "Walk_L_To_L",
    "Walk_L_To_R",
    "Walk_R_To_L",
    "Walk_R_To_R",
)
WALK_F = "Walk_F"

# Raw-npz / z-feature layout (spec §0; mirrors the existing audit tools).
RAW_COND_DIR_SLICE = (4, 6)  # cond_in = act_oh(4) + cond_dir(2) + cond_speed(1)
ZFEAT_CONTACT_SLICE = (278, 280)  # future_desc = pose[276] + root_vel[2] + contact[2]

EPS = 1e-8

# --------------------------------------------------------------- provisional knobs
SEAM_LEN_K = 6  # [PROVISIONAL] spec §1.2 — seam-target window ≈100 ms
CONTEXT_LEN_C = 16  # [PROVISIONAL] spec §1.3 — match base model context_len
GAP_MIN = 12  # [PROVISIONAL] spec §2a — longer-gap curriculum start
GAP_MAX = 30  # [PROVISIONAL] spec §2a — curriculum end
POSE_TOPK = 10  # pose-neighborhood size for full-state alignment (spec §7.1 used top-10)
GROUND_CONTACT_THR = 0.30  # [PROVISIONAL] groundability gate: onset contact_d below this
GROUND_POSE_THR = 0.05  # [PROVISIONAL] sanity: onset pose_d below this (authored loop poses)
GROUNDED_GROUNDED_FALLBACK_MAX_ONSET = 8  # later-onset scan budget for fallback (spec §2b)

# Default sampling mix (spec §2 — ratios PROVISIONAL, design FIXED).
SAMPLE_TYPE_WITHIN = "within_clip"
SAMPLE_TYPE_GROUNDED = "grounded_cross_manifold"
SAMPLE_TYPE_AUGMENTED = "start_state_augmented"
DEFAULT_RATIOS = {
    SAMPLE_TYPE_WITHIN: 0.50,
    SAMPLE_TYPE_GROUNDED: 0.35,
    SAMPLE_TYPE_AUGMENTED: 0.15,
}


# ============================================================ egocentric transform
def egocentric_root_vel(root_vel: np.ndarray, cond_dir: np.ndarray) -> np.ndarray:
    """World-planar `root_vel` rotated into the heading frame (spec §1.1).

    `f̂ = normalize(cond_dir)`; forward = v·f̂, lateral = v·f̂⊥ with
    f̂⊥ = (−f̂_y, f̂_x): ``ego_lat = −v_x·f̂_y + v_y·f̂_x``.
    Returns float32 [T, 2] = (forward, lateral).
    """
    rv = np.asarray(root_vel, dtype=np.float64)
    cd = np.asarray(cond_dir, dtype=np.float64)
    fdir = cd / np.maximum(np.linalg.norm(cd, axis=1, keepdims=True), EPS)
    ego_fwd = rv[:, 0] * fdir[:, 0] + rv[:, 1] * fdir[:, 1]
    ego_lat = -rv[:, 0] * fdir[:, 1] + rv[:, 1] * fdir[:, 0]
    return np.stack([ego_fwd, ego_lat], axis=1).astype(np.float32)


def yaw_rate_from_cond_dir(cond_dir: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Heading angular rate from world heading `cond_dir` (spec §1.1, [FIXED]).

    `heading_t = atan2(cond_dir_y, cond_dir_x)`;
    `yaw_rate_t = wrap_[-π,π](heading_t − heading_{t−1}) × fps` → rad/s;
    frame 0 := frame 1's value; shape [T, 1], dtype float32.
    """
    cd = np.asarray(cond_dir, dtype=np.float64)
    t = cd.shape[0]
    heading = np.arctan2(cd[:, 1], cd[:, 0])
    dh = np.diff(heading)
    dh = (dh + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-π, π)
    yaw = np.empty(t, dtype=np.float64)
    if t >= 2:
        yaw[1:] = dh * float(fps)
        yaw[0] = yaw[1]  # frame 0 := frame 1
    else:
        yaw[:] = 0.0
    return yaw.reshape(t, 1).astype(np.float32)


def build_egocentric_state(
    bone_rot6d: np.ndarray,
    root_vel: np.ndarray,
    cond_dir: np.ndarray,
    contact: np.ndarray,
    fps: float = FPS,
) -> np.ndarray:
    """Assemble the per-frame egocentric state `s_t` [T, 281] (spec §1.1).

    Inputs are already front-aligned (frame i ↔ frame i across signals). Caller
    is responsible for truncating each signal to a common `t = min(len)`.
    """
    bone = np.asarray(bone_rot6d, dtype=np.float64)
    t = bone.shape[0]
    pose = bone.reshape(t, -1)
    if pose.shape[1] != POSE_DIM:
        raise ValueError(f"pose dim {pose.shape[1]} != {POSE_DIM} (bone_rot6d must be [T,46,6])")
    ego = egocentric_root_vel(root_vel[:t], cond_dir[:t])
    yaw = yaw_rate_from_cond_dir(cond_dir[:t], fps=fps)
    con = np.asarray(contact[:t], dtype=np.float64)
    if con.shape[1] != CONTACT_DIM:
        raise ValueError(f"contact dim {con.shape[1]} != {CONTACT_DIM}")
    state = np.concatenate(
        [pose.astype(np.float32), ego, yaw, con.astype(np.float32)], axis=1
    )
    if state.shape[1] != STATE_DIM:
        raise ValueError(f"assembled state dim {state.shape[1]} != {STATE_DIM}")
    return state.astype(np.float32)


# ===================================================== full-state Walk_F alignment
def pose_distance(pose_track: np.ndarray, pose_query: np.ndarray) -> np.ndarray:
    """Per-frame pose L2 normalized by sqrt(dim) (design `pose_l2`)."""
    pt = np.asarray(pose_track, dtype=np.float64)
    pq = np.asarray(pose_query, dtype=np.float64).reshape(-1)
    return np.linalg.norm(pt - pq[None, :], axis=1) / np.sqrt(pt.shape[1])


def contact_distance(contact_track: np.ndarray, contact_query: np.ndarray) -> np.ndarray:
    """Per-frame contact L2 (soft-contact channels)."""
    ct = np.asarray(contact_track, dtype=np.float64)
    cq = np.asarray(contact_query, dtype=np.float64).reshape(-1)
    return np.linalg.norm(ct - cq[None, :], axis=1)


@dataclass
class AlignResult:
    """Outcome of aligning a query frame to the Walk_F hub (spec §7.1)."""

    pose_only_phi: int
    pose_only_pose_d: float
    pose_only_contact_d: float
    full_state_phi: int
    full_state_pose_d: float
    full_state_contact_d: float
    pose_topk_frames: List[int]
    groundable: bool


def full_state_align(
    hub_state: np.ndarray,
    query_frame: np.ndarray,
    *,
    topk: int = POSE_TOPK,
    contact_thr: float = GROUND_CONTACT_THR,
    pose_thr: float = GROUND_POSE_THR,
) -> AlignResult:
    """Full-state nearest Walk_F frame to a query frame (spec §2b / §7.1).

    Operationally: pose localizes the cycle-phase neighborhood (pose top-k),
    contact refines within it. Pose-only argmin is reported alongside to expose
    the contact gap it leaves (verified: R_L pose-only f0 leaves contact_d≈0.96;
    full-state f2 lands at contact_d≈0.162). ego_vel/yaw_rate are not used to
    pick the frame (ego_vel phase-flat, yaw_rate≈0 at onset) — they live in the
    matched full state, consistent with re-entry resolver design D3.

    Groundable iff the full-state pick clears both the contact and pose
    thresholds (spec §2b groundability gate).
    """
    hub = np.asarray(hub_state, dtype=np.float64)
    q = np.asarray(query_frame, dtype=np.float64).reshape(-1)
    pose_d = pose_distance(hub[:, POSE_SLICE], q[POSE_SLICE])
    contact_d = contact_distance(hub[:, CONTACT_SLICE], q[CONTACT_SLICE])

    pose_only_phi = int(np.argmin(pose_d))
    k = int(min(max(topk, 1), hub.shape[0]))
    topk_frames = np.argsort(pose_d)[:k]
    full_state_phi = int(topk_frames[int(np.argmin(contact_d[topk_frames]))])

    groundable = bool(
        contact_d[full_state_phi] <= contact_thr and pose_d[full_state_phi] <= pose_thr
    )
    return AlignResult(
        pose_only_phi=pose_only_phi,
        pose_only_pose_d=float(pose_d[pose_only_phi]),
        pose_only_contact_d=float(contact_d[pose_only_phi]),
        full_state_phi=full_state_phi,
        full_state_pose_d=float(pose_d[full_state_phi]),
        full_state_contact_d=float(contact_d[full_state_phi]),
        pose_topk_frames=[int(j) for j in topk_frames.tolist()],
        groundable=groundable,
    )


# ===================================================================== data loading
def _npz_to_str(v: Any) -> str:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    return str(v)


def load_clip_states(
    z_features: str | Path,
    npz_root: str | Path,
    clips: Sequence[str] = LOCKED_CLIPS,
    fps: float = FPS,
) -> Dict[str, np.ndarray]:
    """Build per-clip egocentric state [t, 281] from frozen artifacts (spec §0).

    contact comes from the z-probe `future_desc[:, 278:280]` (pooled-normalized,
    cross-clip comparable); pose/root_vel/cond_dir from the raw processed npz.
    z-features and raw npz are both front-truncated, so frame i ↔ frame i; each
    clip uses `t = min(len of each signal)` (spec §0). No checkpoint dependency.
    """
    z_features = Path(z_features)
    npz_root = Path(npz_root)
    z = np.load(z_features, allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for clip in clips:
        fd_key = f"{clip}__future_desc"
        if fd_key not in z.files:
            raise RuntimeError(f"missing z-feature key: {fd_key}")
        contact_full = np.asarray(z[fd_key], dtype=np.float64)[
            :, ZFEAT_CONTACT_SLICE[0] : ZFEAT_CONTACT_SLICE[1]
        ]
        raw_path = npz_root / f"{clip}.npz"
        if not raw_path.exists():
            raise FileNotFoundError(f"raw processed npz not found: {raw_path}")
        with np.load(raw_path, allow_pickle=True) as d:
            bone = np.asarray(d["bone_rot6d"], dtype=np.float64)
            root_vel = np.asarray(d["root_vel"], dtype=np.float64)
            cond_in = np.asarray(d["cond_in"], dtype=np.float64)
        t = int(min(contact_full.shape[0], bone.shape[0], root_vel.shape[0], cond_in.shape[0]))
        if t < 2:
            raise RuntimeError(f"{clip}: aligned frame count too small ({t})")
        cond_dir = cond_in[:t, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]]
        out[clip] = build_egocentric_state(
            bone[:t], root_vel[:t], cond_dir, contact_full[:t], fps=fps
        )
    return out


# ===================================================================== the sampler
@dataclass
class SamplerConfig:
    """Sampler knobs (all PROVISIONAL per spec §1/§2)."""

    context_len: int = CONTEXT_LEN_C
    seam_len: int = SEAM_LEN_K
    gap_min: int = GAP_MIN
    gap_max: int = GAP_MAX
    pose_topk: int = POSE_TOPK
    ground_contact_thr: float = GROUND_CONTACT_THR
    ground_pose_thr: float = GROUND_POSE_THR
    fallback_max_onset: int = GROUNDED_GROUNDED_FALLBACK_MAX_ONSET
    ratios: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RATIOS))
    aug_noise_std: float = 0.02  # gaussian std on perturbed ctx last-state channels
    aug_phase_shift_max: int = 3  # max |frames| to roll ctx (phase shift)
    hub_clip: str = WALK_F
    turn_clips: Sequence[str] = TURN_CLIPS

    def gap_for_progress(self, progress: float) -> int:
        """Curriculum gap (spec §2a): grows ~12 → ~30 as progress 0 → 1."""
        p = float(np.clip(progress, 0.0, 1.0))
        return int(round(self.gap_min + p * (self.gap_max - self.gap_min)))


def _interest_weights(state: np.ndarray, yaw_std: float) -> np.ndarray:
    """Per-frame sampling interest (spec §2a biased sampling).

    Oversample turn onset/curvature (high |yaw_rate|), contact transitions, and
    clip onset/end. Returns a non-negative weight per frame with a small floor.
    """
    t = state.shape[0]
    yaw = state[:, YAW_RATE_SLICE].reshape(-1)
    yaw_term = np.abs(yaw) / max(float(yaw_std), EPS)

    contact = state[:, CONTACT_SLICE]
    thr = np.median(contact, axis=0, keepdims=True)
    binary = (contact > thr).astype(np.float64)
    trans = np.zeros(t, dtype=np.float64)
    if t >= 2:
        step = np.abs(np.diff(binary, axis=0)).sum(axis=1)  # transitions between t-1,t
        trans[1:] += step
        trans[:-1] += step  # mark both sides of a transition

    edge = np.zeros(t, dtype=np.float64)
    edge_span = max(1, t // 10)
    edge[:edge_span] = 1.0
    edge[-edge_span:] = 1.0

    w = 0.1 + yaw_term + 1.5 * trans + 1.0 * edge
    return np.maximum(w, EPS)


def _slice_window(arr: np.ndarray, start: int, length: int, *, wrap: bool) -> np.ndarray:
    """Take `length` rows from `start`; wrap modulo len if `wrap` (cyclic hub)."""
    t = arr.shape[0]
    if wrap:
        idx = (np.arange(start, start + length)) % t
        return arr[idx].copy()
    return arr[start : start + length].copy()


@dataclass
class Sample:
    """A single in-betweening training sample (spec §2).

    ctx [C, 281], gt_middle [H, 281], seam_target [K, 281]; meta carries the
    requested type, provenance (clip/frames), curriculum gap, and fallback/aug
    flags. H = gap for within-clip, = grounded horizon for grounded.
    """

    ctx: np.ndarray
    gt_middle: np.ndarray
    seam_target: np.ndarray
    meta: Dict[str, Any]


class InbetweenSampler:
    """Three-type goal-conditioned in-betweening sampler (spec §2).

    Pure numpy. `clips` is a mapping clip_name → egocentric state [t, 281]
    (build via `load_clip_states` or synthetic in tests). Produces `Sample`s; the
    optional torch wrapper is `InbetweenTorchDataset`. No model dependency.
    """

    def __init__(
        self,
        clips: Dict[str, np.ndarray],
        config: Optional[SamplerConfig] = None,
    ) -> None:
        self.config = config or SamplerConfig()
        self.clips = {k: np.asarray(v, dtype=np.float32) for k, v in clips.items()}
        for name, st in self.clips.items():
            if st.ndim != 2 or st.shape[1] != STATE_DIM:
                raise ValueError(f"clip '{name}' state must be [t,{STATE_DIM}], got {st.shape}")
        # Pooled yaw std over all clips (group-norm posture, spec §1.1).
        all_yaw = np.concatenate([st[:, YAW_RATE_SLICE].reshape(-1) for st in self.clips.values()])
        self.yaw_std = float(np.std(all_yaw)) or 1.0
        self._interest = {
            name: _interest_weights(st, self.yaw_std) for name, st in self.clips.items()
        }
        self._type_keys = list(self.config.ratios.keys())
        total = float(sum(self.config.ratios.values())) or 1.0
        self._type_probs = [self.config.ratios[k] / total for k in self._type_keys]

    # ---------------------------------------------------------------- type (a)
    def sample_within_clip(
        self,
        rng: np.random.Generator,
        progress: float = 0.0,
        *,
        clip_name: Optional[str] = None,
        gap: Optional[int] = None,
    ) -> Sample:
        cfg = self.config
        if gap is None:
            gap = cfg.gap_for_progress(progress)
        gap = int(gap)
        names = [clip_name] if clip_name else list(self.clips.keys())
        # A clip is eligible if it can fit at least the smallest curriculum gap;
        # the requested gap is then clamped to the clip's feasible maximum so
        # short clips (e.g. Walk_L_To_R, 50 frames) still produce valid samples.
        min_need = cfg.context_len + cfg.gap_min + cfg.seam_len
        eligible = [n for n in names if self.clips[n].shape[0] >= min_need]
        if not eligible:
            raise ValueError(f"no clip long enough for ctx+gap_min+seam={min_need}")
        name = str(rng.choice(eligible))
        st = self.clips[name]
        t = st.shape[0]
        max_gap = t - cfg.context_len - cfg.seam_len
        clamped = min(gap, max_gap)
        gap_clamped = clamped != gap
        gap = clamped
        lo, hi = cfg.context_len, t - gap - cfg.seam_len  # valid ctx-end index i ∈ [lo, hi]
        cand = np.arange(lo, hi + 1)
        # Bias by interest over the masked middle [i, i+gap).
        interest = self._interest[name]
        w = np.array([interest[i : i + gap].mean() for i in cand], dtype=np.float64)
        w = w / w.sum()
        i = int(rng.choice(cand, p=w))
        ctx = st[i - cfg.context_len : i].copy()
        gt_middle = st[i : i + gap].copy()
        seam = st[i + gap : i + gap + cfg.seam_len].copy()
        meta = {
            "sample_type": SAMPLE_TYPE_WITHIN,
            "clip": name,
            "ctx_start": i - cfg.context_len,
            "middle_start": i,
            "seam_start": i + gap,
            "gap": int(gap),
            "gap_clamped": bool(gap_clamped),
            "progress": float(progress),
            "fallback": None,
            "augmented": False,
        }
        return Sample(ctx, gt_middle, seam, meta)

    # ---------------------------------------------------------------- type (b)
    def grounded_alignment(self, turn_clip: str, onset: int = 0) -> AlignResult:
        """Full-state align a turn clip's onset frame to the Walk_F hub."""
        hub = self.clips[self.config.hub_clip]
        turn = self.clips[turn_clip]
        return full_state_align(
            hub,
            turn[onset],
            topk=self.config.pose_topk,
            contact_thr=self.config.ground_contact_thr,
            pose_thr=self.config.ground_pose_thr,
        )

    def sample_grounded(
        self,
        rng: np.random.Generator,
        progress: float = 0.0,
        *,
        turn_clip: Optional[str] = None,
        horizon: Optional[int] = None,
    ) -> Sample:
        cfg = self.config
        hub = self.clips[cfg.hub_clip]
        names = [turn_clip] if turn_clip else list(cfg.turn_clips)
        name = str(rng.choice([n for n in names if n in self.clips]))
        turn = self.clips[name]
        if horizon is None:
            horizon = cfg.gap_for_progress(progress)
        H = int(horizon)

        # Resolve the grounded source onset via the groundability gate, scanning
        # later onsets on failure (spec §2b L_R fallback).
        onset_used = 0
        fallback = None
        align = self.grounded_alignment(name, onset=0)
        if not align.groundable:
            found = False
            max_o = min(cfg.fallback_max_onset, turn.shape[0] - H - cfg.seam_len)
            for o in range(1, max(1, max_o + 1)):
                a = self.grounded_alignment(name, onset=o)
                if a.groundable:
                    align, onset_used, fallback, found = a, o, "later_onset", True
                    break
            if not found:
                # Drop from pure-grounded → within-clip on this turn clip (spec §2b);
                # flag as a possible onset authoring inconsistency.
                sample = self.sample_within_clip(rng, progress, clip_name=name)
                sample.meta.update(
                    {
                        "sample_type": SAMPLE_TYPE_GROUNDED,
                        "fallback": "within_clip",
                        "fallback_reason": "onset_not_groundable",
                        "turn_clip": name,
                        "onset_contact_d": align.full_state_contact_d,
                    }
                )
                return sample

        phi = align.full_state_phi
        if turn.shape[0] < onset_used + H + cfg.seam_len:
            raise ValueError(f"turn clip '{name}' too short for onset+H+K")
        # Hub context wraps (Walk_F is a single periodic cycle) so a small φ
        # still yields a full C-frame ctx (spec §2b: ctx = Walk_F[φ−C, φ]).
        ctx = _slice_window(hub, phi - cfg.context_len, cfg.context_len, wrap=True)
        gt_middle = turn[onset_used : onset_used + H].copy()
        seam = turn[onset_used + H : onset_used + H + cfg.seam_len].copy()
        meta = {
            "sample_type": SAMPLE_TYPE_GROUNDED,
            "hub_clip": cfg.hub_clip,
            "turn_clip": name,
            "phi": int(phi),
            "pose_only_phi": int(align.pose_only_phi),
            "onset_used": int(onset_used),
            "horizon": H,
            "gap": H,
            "progress": float(progress),
            "fallback": fallback,
            "onset_contact_d": align.full_state_contact_d,
            "augmented": False,
        }
        return Sample(ctx, gt_middle, seam, meta)

    # ---------------------------------------------------------------- type (c)
    def _augment_ctx(self, sample: Sample, rng: np.random.Generator) -> Sample:
        """Perturb ctx last-state with noise + phase shift (spec §2c, B3 fix).

        Schema-preserving: only the ctx is perturbed (pose/ego_vel/yaw_rate
        channels get gaussian noise; the whole ctx may be rolled by a small
        phase shift). gt_middle and seam_target are untouched (the goal is what
        the drifted start must still reach).
        """
        cfg = self.config
        ctx = sample.ctx.copy()
        shift = int(rng.integers(-cfg.aug_phase_shift_max, cfg.aug_phase_shift_max + 1))
        if shift != 0:
            ctx = np.roll(ctx, shift, axis=0)
        # Noise on the last state (the start the generator reaches the goal from).
        noise = rng.normal(0.0, cfg.aug_noise_std, size=(POSE_DIM + EGO_VEL_DIM + YAW_RATE_DIM,))
        ctx[-1, : POSE_DIM + EGO_VEL_DIM + YAW_RATE_DIM] += noise.astype(np.float32)
        meta = dict(sample.meta)
        meta["augmented"] = True
        meta["aug_phase_shift"] = shift
        return Sample(ctx.astype(np.float32), sample.gt_middle, sample.seam_target, meta)

    def sample_augmented(self, rng: np.random.Generator, progress: float = 0.0) -> Sample:
        # Augmentation overlaps a/b (spec §2c): build a base then perturb ctx.
        base_type = SAMPLE_TYPE_GROUNDED if rng.random() < 0.5 else SAMPLE_TYPE_WITHIN
        if base_type == SAMPLE_TYPE_GROUNDED:
            base = self.sample_grounded(rng, progress)
        else:
            base = self.sample_within_clip(rng, progress)
        out = self._augment_ctx(base, rng)
        out.meta["sample_type"] = SAMPLE_TYPE_AUGMENTED
        out.meta["base_type"] = base_type
        return out

    # ---------------------------------------------------------------- dispatch
    def sample(self, rng: np.random.Generator, progress: float = 0.0) -> Sample:
        kind = str(rng.choice(self._type_keys, p=self._type_probs))
        if kind == SAMPLE_TYPE_WITHIN:
            return self.sample_within_clip(rng, progress)
        if kind == SAMPLE_TYPE_GROUNDED:
            return self.sample_grounded(rng, progress)
        return self.sample_augmented(rng, progress)

    def sample_batch(
        self, n: int, rng: np.random.Generator, progress: float = 0.0
    ) -> List[Sample]:
        return [self.sample(rng, progress) for _ in range(int(n))]

    # ----------------------------------------------------- goal conditioning iface
    def encode_goal(self, seam_target: np.ndarray, z_anchor: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Encode the seam-window goal `g` → conditioning tensors (spec §1.2).

        This step only PRODUCES tensors — it does not attach a model. Returns the
        per-frame seam tokens [K, 281] and a flattened vector [K*281]; an optional
        `z_anchor` [32] (region/convergence only, NOT used to pick a frame — D3)
        is passed through if supplied.
        """
        seam = np.asarray(seam_target, dtype=np.float32)
        if seam.ndim != 2 or seam.shape[1] != STATE_DIM:
            raise ValueError(f"seam_target must be [K,{STATE_DIM}], got {seam.shape}")
        out: Dict[str, np.ndarray] = {
            "goal_tokens": seam.copy(),
            "goal_flat": seam.reshape(-1).copy(),
        }
        if z_anchor is not None:
            out["z_anchor"] = np.asarray(z_anchor, dtype=np.float32).reshape(-1).copy()
        return out


def sample_to_torch(sample: Sample) -> Dict[str, Any]:
    """Convert a `Sample` to torch tensors (float32), keeping meta as-is.

    Lazy torch import so the numpy sampler/tests carry no hard torch dependency.
    """
    import torch

    return {
        "ctx": torch.as_tensor(sample.ctx, dtype=torch.float32),
        "gt_middle": torch.as_tensor(sample.gt_middle, dtype=torch.float32),
        "seam_target": torch.as_tensor(sample.seam_target, dtype=torch.float32),
        "meta": sample.meta,
    }


class InbetweenTorchDataset:
    """Thin map-style torch Dataset over a fixed draw of `Sample`s.

    Each draw is materialized once (so __getitem__ is reproducible) and returned
    as torch tensors. This is the dataset surface for §7.3 training; it adds no
    model logic here.
    """

    def __init__(
        self,
        sampler: InbetweenSampler,
        length: int,
        seed: int = 0,
        progress: float = 0.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._samples = sampler.sample_batch(int(length), rng, progress=progress)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return sample_to_torch(self._samples[idx])
