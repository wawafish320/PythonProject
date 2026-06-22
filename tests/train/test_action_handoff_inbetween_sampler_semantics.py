from __future__ import annotations

"""Full-state alignment, groundability gate, and 3-type sampler semantics.

Covers spec §2 (sampling) and §2b/§7.1 (grounded full-state φ + gate). Pure
synthetic data; no artifacts, no model.
"""

import numpy as np

from train.data.action_handoff_inbetween import (
    CONTACT_SLICE,
    POSE_DIM,
    POSE_SLICE,
    SAMPLE_TYPE_AUGMENTED,
    SAMPLE_TYPE_GROUNDED,
    SAMPLE_TYPE_WITHIN,
    STATE_DIM,
    InbetweenSampler,
    SamplerConfig,
    full_state_align,
)


def _state(t: int) -> np.ndarray:
    return np.zeros((t, STATE_DIM), dtype=np.float32)


def _hub_with_pose_ramp(t: int) -> np.ndarray:
    """Hub clip with a monotonic pose ramp (distinct phase) + varying contact."""
    st = _state(t)
    st[:, 0] = np.arange(t, dtype=np.float32)  # pose dim0 ramps → distinct frames
    idx = np.arange(t)
    st[:, CONTACT_SLICE] = np.stack(
        [0.5 + 0.4 * np.sin(idx), 0.5 + 0.4 * np.cos(idx)], axis=1
    ).astype(np.float32)
    return st


# --------------------------------------------------------------- full-state φ
def test_full_state_align_prefers_contact_match_over_pose_only_min() -> None:
    # Hub: frame 0 is the global pose-min but has mismatched contact; frame 3 is
    # within the pose neighborhood AND matches contact → full-state must pick f3.
    hub = _state(20)
    hub[0, POSE_SLICE] = 0.0  # exact pose match (global pose-min)
    hub[1, 0] = 0.001
    hub[2, 0] = 0.002
    hub[3, 0] = 0.0015  # in pose top-k but not the min
    hub[4:, 0] = 1.0  # far in pose → excluded from neighborhood
    hub[:, CONTACT_SLICE] = np.array([0.0, 1.0], dtype=np.float32)  # all bad...
    hub[3, CONTACT_SLICE] = np.array([1.0, 0.0], dtype=np.float32)  # ...except f3

    query = _state(1)[0]
    query[POSE_SLICE] = 0.0
    query[CONTACT_SLICE] = np.array([1.0, 0.0], dtype=np.float32)

    res = full_state_align(hub, query, topk=10, contact_thr=0.3, pose_thr=0.05)
    assert res.pose_only_phi == 0
    assert res.full_state_phi == 3
    assert res.full_state_phi != res.pose_only_phi
    assert res.groundable is True
    assert res.full_state_contact_d < 1e-6


def test_groundability_gate_fails_when_no_neighborhood_contact_match() -> None:
    # Onset pose matches a hub frame, but NO hub frame's contact matches → fail.
    hub = _state(20)
    hub[:, 0] = np.linspace(0.0, 1.0, 20)
    hub[:, CONTACT_SLICE] = np.array([0.0, 1.0], dtype=np.float32)  # every frame far

    query = _state(1)[0]
    query[0] = 0.0  # nearest pose frame is f0
    query[CONTACT_SLICE] = np.array([1.0, 0.0], dtype=np.float32)

    res = full_state_align(hub, query, topk=10, contact_thr=0.3, pose_thr=0.05)
    assert res.groundable is False
    assert res.full_state_contact_d > 0.3


# --------------------------------------------------------------- sampler types
def _make_sampler(seed: int = 0) -> InbetweenSampler:
    clips = {
        "Walk_F": _hub_with_pose_ramp(80),
        "Walk_L_To_L": _state(60),
        "Walk_R_To_R": _state(60),
    }
    cfg = SamplerConfig(turn_clips=("Walk_L_To_L", "Walk_R_To_R"))
    return InbetweenSampler(clips, cfg)


def test_curriculum_gap_grows_monotonically() -> None:
    cfg = SamplerConfig()
    assert cfg.gap_for_progress(0.0) == cfg.gap_min == 12
    assert cfg.gap_for_progress(1.0) == cfg.gap_max == 30
    mid = cfg.gap_for_progress(0.5)
    assert cfg.gap_min < mid < cfg.gap_max
    # monotone non-decreasing across the schedule.
    gaps = [cfg.gap_for_progress(p) for p in np.linspace(0, 1, 11)]
    assert all(b >= a for a, b in zip(gaps, gaps[1:]))


def test_within_clip_sample_respects_curriculum_gap_and_schema() -> None:
    sampler = _make_sampler()
    rng = np.random.default_rng(0)
    s0 = sampler.sample_within_clip(rng, progress=0.0)
    s1 = sampler.sample_within_clip(rng, progress=1.0)
    assert s0.meta["gap"] == 12
    assert s1.meta["gap"] == 30
    for s, gap in ((s0, 12), (s1, 30)):
        assert s.ctx.shape == (sampler.config.context_len, STATE_DIM)
        assert s.gt_middle.shape == (gap, STATE_DIM)
        assert s.seam_target.shape == (sampler.config.seam_len, STATE_DIM)
        assert s.ctx.dtype == np.float32


def test_sample_type_ratios_match_config() -> None:
    sampler = _make_sampler()
    rng = np.random.default_rng(123)
    n = 4000
    counts = {SAMPLE_TYPE_WITHIN: 0, SAMPLE_TYPE_GROUNDED: 0, SAMPLE_TYPE_AUGMENTED: 0}
    for _ in range(n):
        s = sampler.sample(rng, progress=0.3)
        counts[s.meta["sample_type"]] += 1
    fracs = {k: v / n for k, v in counts.items()}
    assert abs(fracs[SAMPLE_TYPE_WITHIN] - 0.50) < 0.04
    assert abs(fracs[SAMPLE_TYPE_GROUNDED] - 0.35) < 0.04
    assert abs(fracs[SAMPLE_TYPE_AUGMENTED] - 0.15) < 0.04


def test_augmentation_preserves_schema_and_flags_meta() -> None:
    sampler = _make_sampler()
    rng = np.random.default_rng(7)
    s = sampler.sample_augmented(rng, progress=0.5)
    assert s.meta["sample_type"] == SAMPLE_TYPE_AUGMENTED
    assert s.meta["augmented"] is True
    assert s.meta["base_type"] in (SAMPLE_TYPE_WITHIN, SAMPLE_TYPE_GROUNDED)
    assert s.ctx.shape == (sampler.config.context_len, STATE_DIM)
    assert s.ctx.shape[1] == STATE_DIM == 281
    assert s.seam_target.shape == (sampler.config.seam_len, STATE_DIM)
    assert s.ctx.dtype == np.float32


# ------------------------------------------------------- grounded construction
def test_grounded_sample_uses_full_state_phi_and_wraps_hub_context() -> None:
    hub = _hub_with_pose_ramp(40)
    turn = _state(60)
    # Onset matches hub frame 5 in both pose and contact → groundable, φ small.
    turn[0, 0] = hub[5, 0]
    turn[0, CONTACT_SLICE] = hub[5, CONTACT_SLICE]
    sampler = InbetweenSampler(
        {"Walk_F": hub, "Walk_R_To_R": turn},
        SamplerConfig(turn_clips=("Walk_R_To_R",)),
    )
    rng = np.random.default_rng(0)
    s = sampler.sample_grounded(rng, progress=0.0, turn_clip="Walk_R_To_R", horizon=12)
    assert s.meta["sample_type"] == SAMPLE_TYPE_GROUNDED
    assert s.meta["fallback"] is None
    assert s.meta["phi"] == 5
    # ctx is the C frames before φ on the (periodic) hub, gt_middle is turn[0:H].
    assert s.ctx.shape == (sampler.config.context_len, STATE_DIM)
    assert s.gt_middle.shape == (12, STATE_DIM)
    assert s.seam_target.shape == (sampler.config.seam_len, STATE_DIM)
    assert np.allclose(s.gt_middle, turn[0:12])


def test_grounded_sample_falls_back_when_onset_not_groundable() -> None:
    hub = _hub_with_pose_ramp(40)
    turn = _state(60)
    turn[:, 0] = np.linspace(0.0, 1.0, 60)  # poses on the hub ramp...
    turn[:, CONTACT_SLICE] = np.array([5.0, 5.0], dtype=np.float32)  # ...contact way off
    sampler = InbetweenSampler(
        {"Walk_F": hub, "Walk_L_To_R": turn},
        SamplerConfig(turn_clips=("Walk_L_To_R",)),
    )
    rng = np.random.default_rng(0)
    s = sampler.sample_grounded(rng, progress=0.0, turn_clip="Walk_L_To_R")
    assert s.meta["sample_type"] == SAMPLE_TYPE_GROUNDED
    assert s.meta["fallback"] == "within_clip"
    # fell back to a within-clip gap on the same turn clip; schema intact.
    assert s.meta["clip"] == "Walk_L_To_R"
    assert s.ctx.shape == (sampler.config.context_len, STATE_DIM)
    assert s.seam_target.shape == (sampler.config.seam_len, STATE_DIM)


def test_encode_goal_produces_tensors_without_model() -> None:
    sampler = _make_sampler()
    rng = np.random.default_rng(0)
    s = sampler.sample_within_clip(rng, progress=0.0)
    enc = sampler.encode_goal(s.seam_target, z_anchor=np.zeros(32, dtype=np.float32))
    assert enc["goal_tokens"].shape == (sampler.config.seam_len, STATE_DIM)
    assert enc["goal_flat"].shape == (sampler.config.seam_len * STATE_DIM,)
    assert enc["z_anchor"].shape == (32,)
    assert enc["goal_tokens"].dtype == np.float32
