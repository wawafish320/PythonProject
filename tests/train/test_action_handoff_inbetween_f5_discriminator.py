from __future__ import annotations

import numpy as np
import torch

from tools.run_action_handoff_inbetween_f5_discriminator import (
    ARM_AR_CMD,
    ARM_MASKED_CMD,
    ARM_MASKED_CMD_SMOOTH,
    CellSpec,
    _derive_signals,
    classify_ar_drift_from_curve,
    evaluate_shared_rollout_state,
    resolve_precommitted_decision,
    smoothness_delta_mse_weighted,
)
from train.action_handoff_inbetween_model import GateThresholds, evaluate_rollout_state_space
from train.data.action_handoff_inbetween import CONTACT_SLICE, EGO_VEL_SLICE, POSE_SLICE, STATE_DIM, YAW_RATE_SLICE


def test_smoothness_loss_ignores_yaw_and_responds_pose_ego_contact_delta() -> None:
    b, h, d = 2, 6, STATE_DIM
    torch.manual_seed(0)
    pred = torch.randn(b, h, d, dtype=torch.float32)
    target = pred.clone()

    ref = smoothness_delta_mse_weighted(
        pred,
        target,
        pose_w=1.0,
        ego_w=1.0,
        contact_w=1.0,
    )

    target_yaw = target.clone()
    target_yaw[:, 1:, YAW_RATE_SLICE] += 100.0
    yaw_only = smoothness_delta_mse_weighted(
        pred,
        target_yaw,
        pose_w=1.0,
        ego_w=1.0,
        contact_w=1.0,
    )
    assert torch.allclose(ref, yaw_only, atol=1e-9)

    target_body = target.clone()
    target_body[:, 1:, POSE_SLICE] += 0.25
    target_body[:, 1:, EGO_VEL_SLICE] += 0.5
    target_body[:, 1:, CONTACT_SLICE] += 0.75
    body_changed = smoothness_delta_mse_weighted(
        pred,
        target_body,
        pose_w=1.0,
        ego_w=1.0,
        contact_w=1.0,
    )
    assert float(body_changed.item()) > float(ref.item())


def test_shared_pop_decomposition_uses_same_resume_frame_as_state_eval() -> None:
    thr = GateThresholds(tau_pose=0.2, tau_pop=10.0, reach_proxy_thr=10.0)
    roll = np.zeros((4, STATE_DIM), dtype=np.float64)
    goal = np.zeros((2, STATE_DIM), dtype=np.float64)
    std = np.ones((STATE_DIM,), dtype=np.float64)

    roll[:, POSE_SLICE] = 5.0
    goal[:, POSE_SLICE] = 10.0
    roll[2, POSE_SLICE] = 0.0
    goal[1, POSE_SLICE] = 0.0

    roll[2, EGO_VEL_SLICE] = np.asarray([3.0, -1.0], dtype=np.float64)
    goal[1, EGO_VEL_SLICE] = np.asarray([1.0, 1.0], dtype=np.float64)
    roll[2, CONTACT_SLICE] = np.asarray([2.0, -2.0], dtype=np.float64)
    goal[1, CONTACT_SLICE] = np.asarray([1.0, -1.0], dtype=np.float64)

    direct = evaluate_rollout_state_space(roll, goal, std, thr)
    shared = evaluate_shared_rollout_state(roll, goal, std, thr)

    assert int(shared["resume_rollout_frame"]) == int(direct["resume_rollout_frame"])
    assert int(shared["resume_target_frame"]) == int(direct["resume_target_frame"])
    ri = int(direct["resume_rollout_frame"])
    tj = int(direct["resume_target_frame"])
    expected_ego = float(np.mean(np.abs((roll[ri, EGO_VEL_SLICE] - goal[tj, EGO_VEL_SLICE]) / std[EGO_VEL_SLICE])))
    expected_contact = float(np.mean(np.abs((roll[ri, CONTACT_SLICE] - goal[tj, CONTACT_SLICE]) / std[CONTACT_SLICE])))
    assert np.isclose(float(shared["ego_pop"]), expected_ego)
    assert np.isclose(float(shared["contact_pop"]), expected_contact)
    assert np.isclose(float(shared["pop"]), 0.5 * (expected_ego + expected_contact))


def test_capacity_mismatch_blocks_data_conclusion() -> None:
    out = resolve_precommitted_decision(
        {
            "yaw_path_valid": True,
            "capacity_match_found": False,
            "plateau_ok_all": True,
            "drift_evidence_sufficient": True,
            "ar_drift_present": False,
            "continuity_prior_arch_signal": False,
            "ar_arch_signal": False,
            "license_grant_possible": True,
        }
    )
    assert out["primary_decision"] == "INSTRUMENT_INVALID_CAPACITY"
    assert not bool(out["data_or_formulation_license_granted"])


def test_plateau_mismatch_blocks_data_conclusion() -> None:
    out = resolve_precommitted_decision(
        {
            "yaw_path_valid": True,
            "capacity_match_found": True,
            "plateau_ok_all": False,
            "drift_evidence_sufficient": True,
            "ar_drift_present": False,
            "continuity_prior_arch_signal": False,
            "ar_arch_signal": False,
            "license_grant_possible": True,
        }
    )
    assert out["primary_decision"] == "INSTRUMENT_INVALID_PLATEAU"
    assert not bool(out["data_or_formulation_license_granted"])


def test_per_step_increasing_curve_triggers_ar_drift_present() -> None:
    curve = [0.2, 0.22, 0.26, 0.31, 0.38, 0.45]
    drift = classify_ar_drift_from_curve(
        curve,
        n_rollouts=9,
        min_rollouts=6,
        ratio_threshold=1.1,
        slope_threshold=1e-4,
        min_horizon=6,
    )
    assert drift["label"] == "AR_DRIFT_PRESENT"
    assert bool(drift["evidence_sufficient"])


def test_flat_curve_no_drift_requires_sufficient_coverage() -> None:
    flat = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    no_drift = classify_ar_drift_from_curve(
        flat,
        n_rollouts=8,
        min_rollouts=6,
        ratio_threshold=1.1,
        slope_threshold=1e-4,
        min_horizon=6,
    )
    assert no_drift["label"] == "AR_NO_DRIFT_EVIDENCE_STRONG"

    insufficient = classify_ar_drift_from_curve(
        flat,
        n_rollouts=2,
        min_rollouts=6,
        ratio_threshold=1.1,
        slope_threshold=1e-4,
        min_horizon=6,
    )
    assert insufficient["label"] == "DRIFT_EVIDENCE_INSUFFICIENT"
    assert not bool(insufficient["evidence_sufficient"])


def _arm_record(*, pop_safe_rate: float, drift_label: str = "AR_NO_DRIFT_EVIDENCE_STRONG") -> dict:
    per_clip = {
        "Walk_L_To_R": {
            "pop_safe_rate": pop_safe_rate,
            "pop_mean": 0.5,
            "contact_pop_mean": 0.5,
            "yaw_path": {"uses_rollout_free_commanded_yaw": False, "posthoc_yaw_replacement": False},
        },
        "Walk_R_To_L": {
            "pop_safe_rate": pop_safe_rate,
            "pop_mean": 0.5,
            "contact_pop_mean": 0.5,
            "yaw_path": {"uses_rollout_free_commanded_yaw": False, "posthoc_yaw_replacement": False},
        },
    }
    if drift_label:
        per_clip["Walk_L_To_R"]["drift_fingerprint"] = {"label": drift_label}
        per_clip["Walk_R_To_L"]["drift_fingerprint"] = {"label": drift_label}
    return {
        "per_clip": per_clip,
        "train_plateau": {"plateau_ok": True},
        "yaw_body_sensitivity": {
            "command_ignored": False,
            "body_delta_pose_mean": 0.1,
            "body_delta_ego_mean": 0.1,
            "body_delta_contact_mean": 0.1,
        },
    }


def test_clean_all_dead_no_drift_no_arch_signal_grants_license() -> None:
    cells_out = {}
    for cell_name, focus in (("fullsup", "Walk_L_To_R"), ("mirror_r2l", "Walk_R_To_L")):
        spec = CellSpec(
            name=cell_name,
            holdout_policy="none",
            holdout_clip=None,
            focus_clip=focus,
            monitor_clips=("Walk_L_To_R",) if cell_name == "mirror_r2l" else (),
        )
        seeds = {}
        for seed in (0, 1, 2):
            ar = _arm_record(pop_safe_rate=0.0)
            for clip in ar["per_clip"]:
                ar["per_clip"][clip]["yaw_path"] = {
                    "uses_rollout_free_commanded_yaw": True,
                    "posthoc_yaw_replacement": False,
                }
            seeds[str(seed)] = {
                "arms": {
                    ARM_MASKED_CMD: _arm_record(pop_safe_rate=0.0),
                    ARM_MASKED_CMD_SMOOTH: _arm_record(pop_safe_rate=0.0),
                    ARM_AR_CMD: ar,
                }
            }
        cells_out[cell_name] = {"cell": spec.__dict__, "seeds": seeds}

    signals = _derive_signals(
        cells_out=cells_out,
        seeds=(0, 1, 2),
        capacity_info={"capacity_match_found": True},
        pop_safe_gate_threshold=1.0,
        improve_pop_safe_eps=0.02,
        improve_pop_eps=0.01,
    )
    decision = resolve_precommitted_decision(signals)

    assert bool(signals["license_grant_possible"])
    assert decision["primary_decision"] == "LICENSE_DATA_OR_FORMULATION_BOTTLENECK"
    assert bool(decision["data_or_formulation_license_granted"])
