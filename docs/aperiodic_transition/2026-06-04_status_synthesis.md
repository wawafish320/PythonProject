> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §8/§9 under its stated read-only / zero-new-injection scope.

# Action-Handoff Status Synthesis

Date: 2026-06-04

## Goal

Locomotion inbetween generation: given Walk_F start context plus a target turn regime,
generate a physically plausible middle `state281 [H=16]` trajectory. Acceptance is by
motion contract, not GT matching.

Current schedule stance: oracle input schedule. Current implementation stance: debug-only
probes; no production Trainer/runtime/gate/checkpoint mutation.

## Locked Conclusions

- Representation remains flat `state281`; c2 / representation ceiling has been excluded for
  the one-window question.
- Decoder remains deterministic.
- Current debug objective is the 3 causal-item cut:
  `L_articulation`, `L_root_support`, `L_goal`.
- Do not reopen entanglement / lifting / diffusion / yaw prediction for this milestone.
- Root/support failures were optimizer and surrogate-alignment issues in the one-window
  train-fit, not a representation ceiling.

## Feasibility Gate

One-window c1 is confirmed on `Walk_L_To_L:0-15`.

Final artifact:
`debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_lat_hardtol0p01_safe1e6_tau0p005_e7000_20260604/summary.md`

Final one-window adjusted guard:

| metric | value | old band / tolerance | result |
|---|---:|---:|---:|
| `bone_angvel_level_rms_to_target` | `0.35147774` | `0.90655224` | pass |
| `angvel_step_rms_p95` | `0.59888975` | `0.59933325` event-aware | pass |
| `angvel_component_p95_p95` | `0.75034146` | `0.78103643` | pass |
| `rootvel_step_l2_p95` | `0.03490959` | `0.04241988` | pass |
| `yaw_rate_step_abs_p95` | `0.13711222` | `0.13711452` | pass |
| `contact_step_l2_p95` | `0.64826574` | `0.64826574` event-aware | pass |
| `foot_slip_p95_mps` | `1.30187497` | `2.76610528` | pass |
| `heading_error_p95_rad` | `0.00000995` | `0.00001000` | pass |
| `pose_step_l2_p95` | `0.01128157` | `0.01174183` | pass |
| `support_side_correctness` | `0` failures | `0` | pass |

Negative controls under adjusted guard:

- shortcut negative controls still fail: `true`
- command demotion negative controls still fail: `true`

## Band Audit

Band audit artifact:
`debug_output/_tmp_action_handoff_band_audit_20260604/summary.md`

The zero-slack hold is resolved. Accepted percentile relabels:

| target | metric | old band | new band |
|---|---|---:|---:|
| `Walk_L_To_L` | `bone_angvel_level_rms` | `0.90655224` | `0.95781331` |
| `Walk_L_To_L` | `rootvel_step_l2` | `0.04241988` | `0.08486664` |
| `Walk_R_To_L` | `rootvel_step_l2` | `0.04802140` | `0.07909000` |
| `Walk_R_To_R` | `bone_angvel_level_rms` | `0.89982127` | `1.01856762` |
| `Walk_R_To_R` | `rootvel_step_l2` | `0.07419514` | `0.10134733` |
| `Walk_R_To_R` | `foot_slip_contacted_speed_mps` | `2.30164003` | `2.39659229` |

Guard after relabels:

- one-window full-family pass: `true`
- shortcut negative controls still fail: `true`
- command demotion negative controls still fail: `true`
- reconstructed GT acceptance: `1.0000`
- decoder-path-from-GT acceptance: `1.0000`
- `max_abs_seq_delta`: `0.00000000`

Decision: `rootvel_zero_slack_contract_hold_resolved=true`,
`authorize_8window_after_band_audit=true`.

## Generalization Gate

8-window evaluation is now authorized but not yet run in this status note. The next action is
an 8-window debug sweep using the same flat `state281`, deterministic decoder, oracle schedule,
3-item objective, adjusted guard, and accepted band relabels.

Do not treat one-window c1 as recipe generalization. The one-window pass used GT warm start,
7000 epochs, low-LR minimax tail, hard-gate-aligned support surrogate, and `tau=0.005`.
