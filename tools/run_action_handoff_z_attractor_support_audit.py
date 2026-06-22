#!/usr/bin/env python3
"""Action-handoff z attractor-support audit (A1/A2/A3).

Scope-narrow follow-up to the P0-P6 closeout decision record. Reads the frozen
z-probe v1 artifact (`z_features_per_clip.npz`) and answers three open questions
that P2-P4 did NOT cover, WITHOUT retraining and WITHOUT touching any in-basin
pipeline (design §0 lock):

  A1. Is contact phase encoded in z / anchor space?
      P6 used ContactMismatchRate as an *eval* metric, but z internal contact
      encoding was never audited. We test linear decodability of contact from z
      (matched ridge arms vs hidden_pre / energy), contact-state kNN purity in
      z, and whether z-speed spikes at contact transitions.

  A2. For weak rows, is the source z far from the target attractor (source off
      support -> coverage/fallback problem), or is the target anchor itself
      diffuse (attractor-definition problem)?
      Operationalizes the closeout decision rule directly. Uses cosine geometry
      to match the runtime retrieval metric (design §3.4).

  A3. Can turn-end phase be mapped to a Walk_F re-entry phase reliably?
      P2 only established *within-clip* phase locality; cross-clip phase
      correspondence (the runtime "read target phase -> map back to Walk_F
      phase" step) was never validated. We check contact-space re-entry
      sharpness and whether z agrees with contact on the re-entry phase.

Single-cycle limitation (closeout §13): Walk_F is one cycle, so multi-cycle
phase aliasing remains untestable with current data; A3 only assesses within
the single available cycle and reports this caveat.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

# future_desc v1 layout: concat(pose[276], root_vel[2], contact[2]); contact
# channel0=right, channel1=left (train.validate.run_freerun_cycles contract).
POSE_DIM = 276
ROOT_SLICE = (276, 278)
CONTACT_SLICE = (278, 280)

# Known weak pairs from P4-alt failure analysis.
WEAK_PAIRS = (
    ("Walk_L_To_R", "Walk_R_To_L"),
    ("Walk_L_To_R", "Walk_R_To_R"),
)

EPS = 1e-8
DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"

# Provisional thresholds (smoke-only; calibrate before any production use,
# mirroring the P6 threshold contract note posture).
RIDGE_LAMBDA = 1.0
KNN_K = 5
KNN_EXCLUDE_WINDOW = 2
REACH_NORM_THR = 1.5  # d_min <= 1.5 * anchor_radius -> reachable
ANCHOR_DIFFUSE_THR = 0.80  # anchor_radius / clip_spread >= 0.80 -> ill-defined attractor
PHASE_GAP_AGREE_THR = 0.15  # |contact_nn - z_nn| / T_F < 0.15 -> z agrees on re-entry phase


# --------------------------------------------------------------------------- io
def _npz_scalar_to_text(v: Any) -> str:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    return str(v)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _fmt(v: float | None, digits: int = 6) -> str:
    if v is None or not np.isfinite(v):
        return "null"
    return f"{float(v):.{digits}f}"


# ------------------------------------------------------------------ geometry
def _cos_dist_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - cosine similarity between rows of a and rows of b."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    an = a64 / np.maximum(np.linalg.norm(a64, axis=1, keepdims=True), EPS)
    bn = b64 / np.maximum(np.linalg.norm(b64, axis=1, keepdims=True), EPS)
    sim = an @ bn.T
    return 1.0 - np.clip(sim, -1.0, 1.0)


def _cos_dist_to_centroid(x: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return _cos_dist_matrix(x, centroid.reshape(1, -1)).reshape(-1)


def _circular_index_gap(i: int, j: int, period: int) -> int:
    delta = abs(int(i) - int(j))
    return int(min(delta, period - delta))


# ------------------------------------------------------------------- A1 ridge
def _blocked_split_mask(t: int, *, test_frac: float = 0.30) -> np.ndarray:
    """Central contiguous holdout per clip to limit temporal-adjacency leakage.

    Returns a boolean test mask of length t.
    """
    k = max(1, int(round(t * test_frac)))
    start = max(0, (t - k) // 2)
    mask = np.zeros(t, dtype=bool)
    mask[start : start + k] = True
    return mask


_SS_TOT_FLOOR = 1e-4  # below this the test target is ~constant -> R^2 ill-defined


def _fit_ridge(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, lam: float) -> dict[str, Any]:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    xtr = x64[train_mask]
    ytr = y64[train_mask]
    mu = np.mean(xtr, axis=0, keepdims=True)
    sd = np.std(xtr, axis=0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    xtr_s = (xtr - mu) / sd
    ymu = np.mean(ytr, axis=0, keepdims=True)
    d = xtr_s.shape[1]
    gram = xtr_s.T @ xtr_s + lam * np.eye(d, dtype=np.float64)
    beta = np.linalg.solve(gram, xtr_s.T @ (ytr - ymu))
    return {"beta": beta, "mu": mu, "sd": sd, "ymu": ymu, "feature_dim": int(d)}


def _eval_ridge_r2(model: dict[str, Any], x: np.ndarray, y: np.ndarray, eval_mask: np.ndarray) -> dict[str, Any]:
    """Evaluate a fitted ridge model on eval_mask frames; per-dim test R^2.

    R^2 is reported per target dim only where the eval target has non-trivial
    variance (ss_tot >= floor); otherwise that dim is omitted as ill-defined
    (near-constant contact in a short window makes R^2 meaningless).
    """
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    xe = (x64[eval_mask] - model["mu"]) / model["sd"]
    ye = y64[eval_mask]
    pred = xe @ model["beta"] + model["ymu"]
    ss_res = np.sum((ye - pred) ** 2, axis=0)
    ss_tot = np.sum((ye - np.mean(ye, axis=0, keepdims=True)) ** 2, axis=0)
    valid = ss_tot >= _SS_TOT_FLOOR
    r2_per_dim = np.where(valid, 1.0 - ss_res / np.maximum(ss_tot, EPS), np.nan)
    finite = r2_per_dim[np.isfinite(r2_per_dim)]
    return {
        "r2_mean": float(np.mean(finite)) if finite.size else None,
        "r2_per_dim": [None if not np.isfinite(v) else float(v) for v in r2_per_dim.tolist()],
        "n_eval": int(np.sum(eval_mask)),
        "n_dims_well_defined": int(np.sum(valid)),
    }


def _ridge_test_r2(
    x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, lam: float
) -> dict[str, Any]:
    """Fit on train_mask, evaluate on its complement; report per-dim test R^2."""
    model = _fit_ridge(x, y, train_mask, lam)
    out = _eval_ridge_r2(model, x, y, ~train_mask)
    out["n_train"] = int(np.sum(train_mask))
    out["feature_dim"] = model["feature_dim"]
    out["_model"] = model
    return out


def _binarize_contact(contact: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Map each frame to an integer state id from per-channel threshold."""
    bits = (np.asarray(contact, dtype=np.float64) > thr.reshape(1, -1)).astype(np.int64)
    # 2 channels -> state in {0,1,2,3}
    return bits[:, 0] * 2 + bits[:, 1]


def audit_a1_contact_encoding(
    z_by_clip: dict[str, np.ndarray],
    hidden_by_clip: dict[str, np.ndarray],
    energy_by_clip: dict[str, np.ndarray],
    contact_by_clip: dict[str, np.ndarray],
) -> dict[str, Any]:
    # Pool frames across all locked clips with a per-clip central blocked split.
    x_z, x_h, x_e, y_c = [], [], [], []
    train_mask_parts, clip_ids = [], []
    for ci, clip in enumerate(LOCKED_CLIPS):
        t = int(z_by_clip[clip].shape[0])
        test_mask = _blocked_split_mask(t)
        train_mask_parts.append(~test_mask)
        clip_ids.append(np.full(t, ci, dtype=np.int64))
        x_z.append(z_by_clip[clip])
        x_h.append(hidden_by_clip[clip])
        x_e.append(energy_by_clip[clip])
        y_c.append(contact_by_clip[clip])

    Xz = np.concatenate(x_z, axis=0)
    Xh = np.concatenate(x_h, axis=0)
    Xe = np.concatenate(x_e, axis=0)
    Yc = np.concatenate(y_c, axis=0)
    train_mask = np.concatenate(train_mask_parts, axis=0)
    clip_id = np.concatenate(clip_ids, axis=0)

    arms = {
        "z_bottleneck": _ridge_test_r2(Xz, Yc, train_mask, RIDGE_LAMBDA),
        "raw_hidden_pre": _ridge_test_r2(Xh, Yc, train_mask, RIDGE_LAMBDA),
        "energy_scalar": _ridge_test_r2(Xe, Yc, train_mask, RIDGE_LAMBDA),
    }

    # Per-clip diagnostic of the *pooled* z model (per-clip failures cannot hide
    # behind the global mean, mirroring design §3.3 P1 reporting policy). We
    # report normalized RMSE (RMSE / global contact std) instead of per-clip
    # R^2: per-clip test windows are short and contact is often near-constant
    # there, which makes per-clip R^2 numerically degenerate. nRMSE<1 means the
    # pooled model predicts that clip's contact better than the global mean.
    z_model = arms["z_bottleneck"]["_model"]
    global_contact_std = np.std(Yc, axis=0)
    global_contact_std = np.where(global_contact_std > 1e-8, global_contact_std, 1.0)
    per_clip_z_nrmse: dict[str, Any] = {}
    for ci, clip in enumerate(LOCKED_CLIPS):
        sel = (clip_id == ci) & (~train_mask)
        if not np.any(sel):
            per_clip_z_nrmse[clip] = None
            continue
        xe = (Xz[sel] - z_model["mu"]) / z_model["sd"]
        pred = xe @ z_model["beta"] + z_model["ymu"]
        rmse = np.sqrt(np.mean((Yc[sel] - pred) ** 2, axis=0))
        per_clip_z_nrmse[clip] = float(np.mean(rmse / global_contact_std))

    for arm in arms.values():
        arm.pop("_model", None)

    # Contact-state kNN purity in z (cosine), excluding self + temporal window
    # within the same clip.
    contact_thr = np.median(Yc, axis=0)
    state = _binarize_contact(Yc, contact_thr)
    n = state.shape[0]
    unique, counts = np.unique(state, return_counts=True)
    chance = float(np.sum((counts / max(n, 1)) ** 2))

    purity_hits, purity_total = 0, 0
    # build per-clip frame offsets for temporal exclusion
    offset = np.zeros(n, dtype=np.int64)
    base = 0
    clip_local_idx = np.zeros(n, dtype=np.int64)
    for ci, clip in enumerate(LOCKED_CLIPS):
        t = int(z_by_clip[clip].shape[0])
        clip_local_idx[base : base + t] = np.arange(t)
        offset[base : base + t] = base
        base += t

    dmat = _cos_dist_matrix(Xz, Xz)
    for i in range(n):
        exclude = np.zeros(n, dtype=bool)
        exclude[i] = True
        same_clip = clip_id == clip_id[i]
        for j in np.where(same_clip)[0]:
            if abs(int(clip_local_idx[j]) - int(clip_local_idx[i])) <= KNN_EXCLUDE_WINDOW:
                exclude[j] = True
        valid = np.where(~exclude)[0]
        if valid.size == 0:
            continue
        order = valid[np.argsort(dmat[i, valid])][:KNN_K]
        purity_hits += int(np.sum(state[order] == state[i]))
        purity_total += int(order.size)
    knn_purity = float(purity_hits / max(purity_total, 1))

    # z-speed vs contact-transition (per clip, cosine consecutive step)
    change_speeds, nochange_speeds = [], []
    for clip in LOCKED_CLIPS:
        z = z_by_clip[clip]
        c = contact_by_clip[clip]
        st = _binarize_contact(c, contact_thr)
        if z.shape[0] < 2:
            continue
        step = np.array(
            [_cos_dist_matrix(z[t : t + 1], z[t - 1 : t])[0, 0] for t in range(1, z.shape[0])],
            dtype=np.float64,
        )
        changed = st[1:] != st[:-1]
        change_speeds.extend(step[changed].tolist())
        nochange_speeds.extend(step[~changed].tolist())
    mean_change = float(np.mean(change_speeds)) if change_speeds else float("nan")
    mean_nochange = float(np.mean(nochange_speeds)) if nochange_speeds else float("nan")
    transition_lift = (
        float(mean_change / mean_nochange) if (nochange_speeds and mean_nochange > EPS) else float("nan")
    )

    z_r2 = arms["z_bottleneck"]["r2_mean"]
    e_r2 = arms["energy_scalar"]["r2_mean"]
    knn_lift = knn_purity - chance
    if z_r2 > max(0.30, e_r2) and knn_lift > 0.15:
        status = "contact_phase_encoded"
    elif z_r2 < e_r2 or knn_lift <= 0.05:
        status = "contact_phase_weak_or_absent"
    else:
        status = "inconclusive"

    return {
        "question": "Is contact phase encoded in z / anchor space?",
        "contact_layout": {
            "future_desc_slice": list(CONTACT_SLICE),
            "channels": ["contact_right", "contact_left"],
        },
        "split_protocol": "per-clip central contiguous 30% holdout; pooled; train-standardized ridge",
        "ridge_lambda": RIDGE_LAMBDA,
        "decodability_test_r2": {
            "z_bottleneck": arms["z_bottleneck"],
            "raw_hidden_pre": arms["raw_hidden_pre"],
            "energy_scalar": arms["energy_scalar"],
        },
        "per_clip_z_nrmse": per_clip_z_nrmse,
        "knn_contact_state_purity": {
            "k": KNN_K,
            "purity": knn_purity,
            "chance": chance,
            "lift": knn_lift,
            "num_states_observed": int(unique.size),
        },
        "z_speed_at_contact_transition": {
            "mean_z_step_at_change": mean_change,
            "mean_z_step_no_change": mean_nochange,
            "transition_lift_ratio": transition_lift,
        },
        "provisional_gate": {
            "rule": "encoded iff z_test_r2 > max(0.30, energy_r2) and knn_lift > 0.15",
            "interpretation": status,
        },
    }


# ----------------------------------------------------------- A2 reachability
def _clip_intra_step(z: np.ndarray) -> float:
    if z.shape[0] < 2:
        return float("nan")
    steps = np.array(
        [_cos_dist_matrix(z[t : t + 1], z[t - 1 : t])[0, 0] for t in range(1, z.shape[0])],
        dtype=np.float64,
    )
    return float(np.mean(steps))


def audit_a2_reachability(
    z_by_clip: dict[str, np.ndarray], *, end_window_k: int
) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    anchors: dict[str, dict[str, Any]] = {}

    for target in TURN_CLIPS:
        zt = z_by_clip[target]
        t = int(zt.shape[0])
        k = min(end_window_k, t)
        end_win = zt[t - k : t]
        anchor_centroid = np.mean(end_win, axis=0)
        anchor_radius = float(np.mean(_cos_dist_to_centroid(end_win, anchor_centroid)))
        clip_centroid = np.mean(zt, axis=0)
        clip_spread = float(np.mean(_cos_dist_to_centroid(zt, clip_centroid)))
        diffuseness = float(anchor_radius / max(clip_spread, EPS))
        anchors[target] = {
            "end_window_k": int(k),
            "anchor_radius_cos": anchor_radius,
            "clip_spread_cos": clip_spread,
            "anchor_diffuseness": diffuseness,
        }

    for target in TURN_CLIPS:
        zt = z_by_clip[target]
        t = int(zt.shape[0])
        k = min(end_window_k, t)
        anchor_centroid = np.mean(zt[t - k : t], axis=0)
        anchor_radius = anchors[target]["anchor_radius_cos"]
        diffuseness = anchors[target]["anchor_diffuseness"]

        for source in LOCKED_CLIPS:
            if source == target:
                continue
            zs = z_by_clip[source]
            t_s = int(zs.shape[0])
            d_all = _cos_dist_to_centroid(zs, anchor_centroid)
            exit_frame_all = int(np.argmin(d_all))
            d_min_all = float(d_all[exit_frame_all])
            ks = min(end_window_k, t_s)
            end_base = t_s - ks
            d_end = _cos_dist_to_centroid(zs[end_base:], anchor_centroid)
            exit_frame_source_end = int(end_base + int(np.argmin(d_end)))
            d_min_end = float(d_end[exit_frame_source_end - end_base])
            intra = _clip_intra_step(zs)
            d_min_norm = float(d_min_all / max(anchor_radius, EPS))
            gap_in_steps = float(d_min_all / max(intra, EPS)) if np.isfinite(intra) else float("nan")

            # Bridge endpoints implied by the closest-approach exit frame: the
            # entry frame is the target frame nearest (cosine) to the source's
            # exit-frame z. This is the (exit_i, entry_j) pair B is meant to give.
            exit_z = zs[exit_frame_all : exit_frame_all + 1]
            d_to_target = _cos_dist_matrix(exit_z, zt).reshape(-1)
            entry_frame_in_target = int(np.argmin(d_to_target))
            entry_cos_dist = float(d_to_target[entry_frame_in_target])

            if diffuseness >= ANCHOR_DIFFUSE_THR:
                classification = "target_anchor_ill_defined"
            elif d_min_norm <= REACH_NORM_THR:
                classification = "reachable"
            else:
                classification = "source_off_support"

            pair_rows.append(
                {
                    "source": source,
                    "target": target,
                    "is_known_weak_pair": [source, target] in [list(p) for p in WEAK_PAIRS],
                    "anchor_radius_cos": anchor_radius,
                    "anchor_diffuseness": diffuseness,
                    "d_min_to_anchor_all_frames": d_min_all,
                    "d_min_to_anchor_source_end": d_min_end,
                    "d_min_normalized_by_anchor_radius": d_min_norm,
                    "source_intra_step_cos": intra,
                    "gap_in_source_steps": gap_in_steps,
                    "source_frame_count": t_s,
                    "exit_frame_all_frames": exit_frame_all,
                    "exit_frame_source_end_window": exit_frame_source_end,
                    "entry_frame_in_target": entry_frame_in_target,
                    "entry_frame_cos_dist": entry_cos_dist,
                    "target_frame_count": t,
                    "classification": classification,
                }
            )

    weak = [r for r in pair_rows if r["is_known_weak_pair"]]
    normal = [r for r in pair_rows if not r["is_known_weak_pair"]]

    def _mean(rows: list[dict[str, Any]], key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    weak_classes = sorted({r["classification"] for r in weak})
    all_anchors_well_defined = all(
        a["anchor_diffuseness"] < ANCHOR_DIFFUSE_THR for a in anchors.values()
    )
    if not all_anchors_well_defined:
        verdict = "attractor_definition_problem"
    elif weak and all(r["classification"] == "source_off_support" for r in weak):
        verdict = "source_off_support_coverage_or_fallback"
    elif weak and all(r["classification"] == "reachable" for r in weak):
        verdict = "weak_pairs_reachable_failure_elsewhere"
    else:
        verdict = "mixed_inspect_rows"

    return {
        "question": "Weak rows: source off support vs target anchor ill-defined?",
        "metric": "cosine distance (matches runtime retrieval, design §3.4)",
        "end_window_k": int(end_window_k),
        "provisional_thresholds": {
            "reach_norm_thr": REACH_NORM_THR,
            "anchor_diffuse_thr": ANCHOR_DIFFUSE_THR,
        },
        "target_anchors": anchors,
        "pairs": pair_rows,
        "weak_pair_summary": {
            "pairs": [[s, t] for (s, t) in WEAK_PAIRS],
            "classifications": weak_classes,
            "mean_d_min_normalized": _mean(weak, "d_min_normalized_by_anchor_radius"),
            "mean_gap_in_source_steps": _mean(weak, "gap_in_source_steps"),
        },
        "normal_pair_summary": {
            "mean_d_min_normalized": _mean(normal, "d_min_normalized_by_anchor_radius"),
            "mean_gap_in_source_steps": _mean(normal, "gap_in_source_steps"),
        },
        "all_target_anchors_well_defined": bool(all_anchors_well_defined),
        "verdict": verdict,
    }


# ----------------------------------------------------- A3 phase re-entry
def audit_a3_reentry(
    z_by_clip: dict[str, np.ndarray], contact_by_clip: dict[str, np.ndarray]
) -> dict[str, Any]:
    z_f = z_by_clip[WALK_F]
    c_f = contact_by_clip[WALK_F]
    t_f = int(z_f.shape[0])

    per_clip: dict[str, Any] = {}
    phase_gaps, sharpness_vals = [], []

    for clip in TURN_CLIPS:
        zt = z_by_clip[clip]
        ct = contact_by_clip[clip]
        e = int(zt.shape[0]) - 1  # turn-end (re-entry candidate)

        # contact-space re-entry: nearest Walk_F frame by contact distance
        cdist = np.linalg.norm(c_f - ct[e : e + 1], axis=1)
        c_order = np.argsort(cdist)
        c_nn = int(c_order[0])
        top3 = c_order[:3]
        # circular index spread of top-3 contact matches over the Walk_F cycle
        ref = int(top3[0])
        spread = float(
            np.mean([_circular_index_gap(int(j), ref, t_f) for j in top3]) / max(t_f, 1)
        )

        # z-space re-entry: nearest Walk_F frame by cosine
        zdist = _cos_dist_matrix(zt[e : e + 1], z_f).reshape(-1)
        z_nn = int(np.argmin(zdist))

        gap = float(_circular_index_gap(c_nn, z_nn, t_f) / max(t_f, 1))
        phase_gaps.append(gap)
        sharpness_vals.append(spread)

        per_clip[clip] = {
            "turn_end_frame": e,
            "contact_nn_walk_f_frame": c_nn,
            "contact_nn_dist": float(cdist[c_nn]),
            "contact_top3_walk_f_frames": [int(j) for j in top3.tolist()],
            "contact_match_sharpness_circular": spread,
            "z_nn_walk_f_frame": z_nn,
            "z_nn_cos_dist": float(zdist[z_nn]),
            "phase_index_gap_z_vs_contact": gap,
            "z_agrees_with_contact": bool(gap < PHASE_GAP_AGREE_THR),
        }

    mean_gap = float(np.mean(phase_gaps))
    mean_sharp = float(np.mean(sharpness_vals))
    agree_count = int(sum(1 for c in TURN_CLIPS if per_clip[c]["z_agrees_with_contact"]))

    sharp_ok = mean_sharp < 0.15
    if sharp_ok and agree_count >= 3:
        status = "reentry_phase_well_supported"
    elif sharp_ok and agree_count < 2:
        status = "reentry_needs_contact_phase_not_z"
    elif not sharp_ok:
        status = "reentry_phase_ambiguous_from_contact"
    else:
        status = "inconclusive"

    return {
        "question": "Can turn-end phase map to a Walk_F re-entry phase reliably?",
        "walk_f_cycle_frames": t_f,
        "phase_proxy": "soft contact (gait phase); z compared via cosine",
        "single_cycle_caveat": (
            "Walk_F is one cycle; multi-cycle phase aliasing is untestable with "
            "current data (closeout §13)."
        ),
        "per_clip": per_clip,
        "aggregate": {
            "mean_phase_index_gap_z_vs_contact": mean_gap,
            "mean_contact_match_sharpness_circular": mean_sharp,
            "z_agrees_with_contact_count": agree_count,
            "n_turn_clips": len(TURN_CLIPS),
        },
        "provisional_gate": {
            "rule": (
                "well_supported iff mean_sharpness<0.15 and z-agrees>=3; "
                "needs_contact iff sharp but z-agrees<2; "
                "ambiguous iff mean_sharpness>=0.15"
            ),
            "interpretation": status,
        },
    }


# --------------------------------------------------------------------- plots
def _plot_a1(z_f: np.ndarray, contact_f: np.ndarray, thr: np.ndarray, out: Path) -> None:
    z64 = np.asarray(z_f, dtype=np.float64)
    zc = z64 - np.mean(z64, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(zc, full_matrices=False)
    proj = zc @ vt[:2].T
    state = _binarize_contact(contact_f, thr)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=state, cmap="viridis", s=40)
    ax.plot(proj[:, 0], proj[:, 1], "-", color="0.6", linewidth=0.8, alpha=0.6)
    ax.set_title("A1: Walk_F z PCA(2D) colored by contact state")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="contact state id")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_a2(a2: dict[str, Any], out: Path) -> None:
    rows = a2["pairs"]
    labels = [f"{r['source'][:10]}->{r['target'][:10]}" for r in rows]
    vals = [r["d_min_normalized_by_anchor_radius"] for r in rows]
    weak = [r["is_known_weak_pair"] for r in rows]
    colors = ["#d62728" if w else "#1f77b4" for w in weak]
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.barh(range(len(rows)), vals, color=colors)
    ax.axvline(REACH_NORM_THR, color="0.3", linestyle="--", label=f"reach thr={REACH_NORM_THR}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("d_min / anchor_radius (cosine)")
    ax.set_title("A2: source->target anchor reachability (red=known weak pair)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_a3(a3: dict[str, Any], t_f: int, out: Path) -> None:
    clips = list(a3["per_clip"].keys())
    c_nn = [a3["per_clip"][c]["contact_nn_walk_f_frame"] for c in clips]
    z_nn = [a3["per_clip"][c]["z_nn_walk_f_frame"] for c in clips]
    x = np.arange(len(clips))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(x - 0.1, c_nn, c="#2ca02c", s=70, label="contact re-entry frame")
    ax.scatter(x + 0.1, z_nn, c="#9467bd", s=70, marker="^", label="z re-entry frame")
    for i in range(len(clips)):
        ax.plot([x[i] - 0.1, x[i] + 0.1], [c_nn[i], z_nn[i]], color="0.6", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(clips, fontsize=8, rotation=15)
    ax.set_ylabel(f"Walk_F frame index (cycle T={t_f})")
    ax.set_title("A3: re-entry phase (contact vs z); large gap = z lacks cross-clip phase")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def _load_features(npz_path: Path) -> tuple[dict, dict, dict, dict, list[str]]:
    npz = np.load(npz_path, allow_pickle=True)
    clip_order = (
        [_npz_scalar_to_text(x) for x in np.asarray(npz["clip_order"], dtype=object).tolist()]
        if "clip_order" in npz.files
        else list(LOCKED_CLIPS)
    )
    z_by, h_by, e_by, c_by = {}, {}, {}, {}
    for clip in LOCKED_CLIPS:
        for suffix in ("z", "hidden_pre", "future_desc"):
            key = f"{clip}__{suffix}"
            if key not in npz.files:
                raise RuntimeError(f"missing npz key: {key}")
        z = np.asarray(npz[f"{clip}__z"], dtype=np.float64)
        h = np.asarray(npz[f"{clip}__hidden_pre"], dtype=np.float64)
        fd = np.asarray(npz[f"{clip}__future_desc"], dtype=np.float64)
        en_key = f"{clip}__energy"
        en = np.asarray(npz[en_key], dtype=np.float64) if en_key in npz.files else fd[:, ROOT_SLICE[0] : ROOT_SLICE[0] + 1]
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(h)) and np.all(np.isfinite(fd))):
            raise RuntimeError(f"{clip}: non-finite feature values")
        z_by[clip] = z
        h_by[clip] = h
        e_by[clip] = en
        c_by[clip] = fd[:, CONTACT_SLICE[0] : CONTACT_SLICE[1]]
    return z_by, h_by, e_by, c_by, clip_order


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Action-handoff z attractor-support audit (A1/A2/A3); frozen artifact, no retrain."
    )
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--end-window-k", type=int, default=12)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.end_window_k) < 2:
        raise RuntimeError("--end-window-k must be >= 2")

    z_path = Path(args.z_features)
    if not z_path.exists():
        raise FileNotFoundError(f"z-features not found: {z_path}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(
        f"debug_output/_tmp_action_handoff_z_attractor_support_audit_{date_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    z_by, h_by, e_by, c_by, clip_order = _load_features(z_path)

    a1 = audit_a1_contact_encoding(z_by, h_by, e_by, c_by)
    a2 = audit_a2_reachability(z_by, end_window_k=int(args.end_window_k))
    a3 = audit_a3_reentry(z_by, c_by)

    # plots
    contact_thr = np.median(
        np.concatenate([c_by[c] for c in LOCKED_CLIPS], axis=0), axis=0
    )
    a1_png = out_dir / "a1_walk_f_z_contact_state.png"
    a2_png = out_dir / "a2_reachability_bars.png"
    a3_png = out_dir / "a3_reentry_phase.png"
    _plot_a1(z_by[WALK_F], c_by[WALK_F], contact_thr, a1_png)
    _plot_a2(a2, a2_png)
    _plot_a3(a3, int(z_by[WALK_F].shape[0]), a3_png)

    summary = {
        "task": "Action handoff z attractor-support audit (A1/A2/A3)",
        "scope": "research/probe follow-up; frozen artifact; no retrain; no in-basin pipeline change",
        "input_artifact_path": str(z_path.resolve()),
        "locked_clips": list(LOCKED_CLIPS),
        "clip_order_from_npz": clip_order,
        "a1_contact_phase_encoding": a1,
        "a2_source_to_anchor_reachability": a2,
        "a3_cross_clip_phase_reentry": a3,
        "artifacts": {
            "a1_plot": str(a1_png.resolve()),
            "a2_plot": str(a2_png.resolve()),
            "a3_plot": str(a3_png.resolve()),
        },
    }
    json_path = out_dir / "z_attractor_support_audit_summary.json"
    _dump_json(json_path, summary)

    lines: list[str] = []
    lines.append("# Action Handoff z Attractor-Support Audit (A1/A2/A3)")
    lines.append("")
    lines.append(f"- input: {z_path.resolve()}")
    lines.append("- scope: frozen-artifact follow-up to P0-P6 closeout; no retrain; in-basin pipeline locked.")
    lines.append("")
    lines.append("## A1 Contact phase encoded in z?")
    r = a1["decodability_test_r2"]
    lines.append(
        f"- contact decodability test R^2: z={_fmt(r['z_bottleneck']['r2_mean'])} "
        f"(dim {r['z_bottleneck']['feature_dim']}), hidden_pre={_fmt(r['raw_hidden_pre']['r2_mean'])} "
        f"(dim {r['raw_hidden_pre']['feature_dim']}), energy={_fmt(r['energy_scalar']['r2_mean'])} (dim 1)."
    )
    lines.append(
        f"- contact-state kNN purity in z={_fmt(a1['knn_contact_state_purity']['purity'])} "
        f"vs chance={_fmt(a1['knn_contact_state_purity']['chance'])} "
        f"(lift={_fmt(a1['knn_contact_state_purity']['lift'])})."
    )
    lines.append(
        f"- z-step at contact transition vs non-transition: "
        f"{_fmt(a1['z_speed_at_contact_transition']['mean_z_step_at_change'])} vs "
        f"{_fmt(a1['z_speed_at_contact_transition']['mean_z_step_no_change'])} "
        f"(lift={_fmt(a1['z_speed_at_contact_transition']['transition_lift_ratio'])})."
    )
    lines.append(
        "- per-clip z->contact nRMSE (<1 better than global mean): "
        + ", ".join(f"{c}={_fmt(v)}" for c, v in a1["per_clip_z_nrmse"].items())
    )
    lines.append(f"- **interpretation: {a1['provisional_gate']['interpretation']}**")
    lines.append("")
    lines.append("## A2 Weak rows: source off support vs anchor ill-defined?")
    for tgt, an in a2["target_anchors"].items():
        lines.append(
            f"- anchor[{tgt}]: radius_cos={_fmt(an['anchor_radius_cos'])}, "
            f"clip_spread={_fmt(an['clip_spread_cos'])}, diffuseness={_fmt(an['anchor_diffuseness'])}"
        )
    ws = a2["weak_pair_summary"]
    ns = a2["normal_pair_summary"]
    lines.append(
        f"- weak pairs: mean d_min/anchor_radius={_fmt(ws['mean_d_min_normalized'])}, "
        f"mean gap_in_source_steps={_fmt(ws['mean_gap_in_source_steps'])}, classes={ws['classifications']}"
    )
    lines.append(
        f"- normal pairs: mean d_min/anchor_radius={_fmt(ns['mean_d_min_normalized'])}, "
        f"mean gap_in_source_steps={_fmt(ns['mean_gap_in_source_steps'])}"
    )
    lines.append(f"- all target anchors well-defined: {a2['all_target_anchors_well_defined']}")
    lines.append(f"- **verdict: {a2['verdict']}**")
    lines.append("- implied bridge endpoints (closest-approach exit -> nearest target entry):")
    for r in a2["pairs"]:
        if not r["is_known_weak_pair"]:
            continue
        lines.append(
            f"  - {r['source']}->{r['target']}: exit_frame={r['exit_frame_all_frames']}/{r['source_frame_count']} "
            f"(end-window exit={r['exit_frame_source_end_window']}), "
            f"entry_frame={r['entry_frame_in_target']}/{r['target_frame_count']} "
            f"(entry_cos_dist={_fmt(r['entry_frame_cos_dist'])})"
        )
    lines.append("")
    lines.append("## A3 Cross-clip phase re-entry")
    for clip, m in a3["per_clip"].items():
        lines.append(
            f"- {clip}: contact_nn_frame={m['contact_nn_walk_f_frame']}, z_nn_frame={m['z_nn_walk_f_frame']}, "
            f"phase_gap={_fmt(m['phase_index_gap_z_vs_contact'])}, sharpness={_fmt(m['contact_match_sharpness_circular'])}, "
            f"z_agrees={m['z_agrees_with_contact']}"
        )
    ag = a3["aggregate"]
    lines.append(
        f"- aggregate: mean_phase_gap={_fmt(ag['mean_phase_index_gap_z_vs_contact'])}, "
        f"mean_sharpness={_fmt(ag['mean_contact_match_sharpness_circular'])}, "
        f"z_agrees={ag['z_agrees_with_contact_count']}/{ag['n_turn_clips']}"
    )
    lines.append(f"- caveat: {a3['single_cycle_caveat']}")
    lines.append(f"- **interpretation: {a3['provisional_gate']['interpretation']}**")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {json_path.resolve()}")
    lines.append(f"- {a1_png.resolve()}")
    lines.append(f"- {a2_png.resolve()}")
    lines.append(f"- {a3_png.resolve()}")

    md_path = out_dir / "z_attractor_support_audit_summary.md"
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(
        f"[ok] A1={a1['provisional_gate']['interpretation']} "
        f"A2={a2['verdict']} A3={a3['provisional_gate']['interpretation']}"
    )


if __name__ == "__main__":
    main()
