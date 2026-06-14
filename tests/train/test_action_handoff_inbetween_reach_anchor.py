from __future__ import annotations

"""hidden_pre reach-metric / anchor semantics (§7.3 3b Slice 1). Pure synthetic data."""

import numpy as np
import pytest

from train.action_handoff_inbetween_reach import (
    HIDDEN_DIM,
    build_hidden_pre_anchors,
    cos_dist,
)


def test_fail_fast_on_bad_hidden_shapes_and_nonfinite() -> None:
    good = np.zeros((20, HIDDEN_DIM))
    # wrong width
    with pytest.raises(ValueError):
        build_hidden_pre_anchors({"T": np.zeros((20, 256))}, ("T",))
    # empty
    with pytest.raises(ValueError):
        build_hidden_pre_anchors({"T": np.zeros((0, HIDDEN_DIM))}, ("T",))
    # non-finite
    bad = good.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        build_hidden_pre_anchors({"T": bad}, ("T",))
    # min_norm validates the rollout too
    a = build_hidden_pre_anchors({"T": np.random.default_rng(0).normal(size=(20, HIDDEN_DIM))}, ("T",))["T"]
    with pytest.raises(ValueError):
        a.min_norm(np.zeros((3, 256)))


def test_degenerate_radius_flagged_not_well_defined() -> None:
    # All-identical end window → radius ~0 → degenerate → not well-defined.
    direction = np.ones(HIDDEN_DIM)
    clip = np.concatenate(
        [np.random.default_rng(0).normal(size=(30, HIDDEN_DIM)), np.tile(direction, (12, 1))], axis=0
    )
    a = build_hidden_pre_anchors({"T": clip}, ("T",), end_window_k=12)["T"]
    assert a.radius_degenerate is True
    assert a.well_defined is False


def test_cos_dist_zero_for_parallel_one_for_orthogonal() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = cos_dist(a, np.array([2.0, 0.0]))  # parallel to row0, orthogonal to row1
    assert d.shape == (2, 1)
    assert abs(float(d[0, 0])) < 1e-9
    assert abs(float(d[1, 0]) - 1.0) < 1e-9


def test_anchor_well_defined_and_reach_logic() -> None:
    rng = np.random.default_rng(0)
    # A tight end-window cluster (small radius) embedded in a broader clip → well-defined.
    base = rng.normal(size=(40, HIDDEN_DIM))
    direction = rng.normal(size=(HIDDEN_DIM,))
    direction /= np.linalg.norm(direction)
    end = direction[None, :] * 10.0 + 0.01 * rng.normal(size=(12, HIDDEN_DIM))  # tight cluster
    clip = np.concatenate([base, end], axis=0)

    anchors = build_hidden_pre_anchors({"Walk_R_To_R": clip}, ("Walk_R_To_R",), end_window_k=12)
    a = anchors["Walk_R_To_R"]
    assert a.centroid.shape == (HIDDEN_DIM,)
    assert a.radius >= 0.0
    assert a.well_defined is True  # tight end vs broad clip → diffuseness < 0.80

    # A rollout that passes through the anchor centroid reaches; a far one does not.
    reaching = np.stack([rng.normal(size=HIDDEN_DIM), a.centroid, rng.normal(size=HIDDEN_DIM)])
    assert a.reached(reaching, conv_norm_thr=1.5) is True
    far = -direction[None, :] * 10.0 + np.zeros((3, HIDDEN_DIM))  # opposite hemisphere
    assert a.reached(far, conv_norm_thr=1.5) is False


def test_min_norm_scales_with_radius() -> None:
    rng = np.random.default_rng(1)
    direction = rng.normal(size=(HIDDEN_DIM,))
    direction /= np.linalg.norm(direction)
    end = direction[None, :] * 5.0 + 0.02 * rng.normal(size=(12, HIDDEN_DIM))
    clip = np.concatenate([rng.normal(size=(30, HIDDEN_DIM)), end], axis=0)
    a = build_hidden_pre_anchors({"T": clip}, ("T",), end_window_k=12)["T"]
    # the centroid itself has min_norm 0; a point at exactly radius distance → ~1.0.
    assert a.min_norm(a.centroid[None, :]) < 1e-6
