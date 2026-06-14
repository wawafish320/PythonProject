> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Band Audit: Continuous-Percentile Relabel

Date: 2026-06-04

Scope: debug-only band audit for action-handoff inbetween acceptance. No production Trainer/runtime/gate/checkpoint path was changed.

## Conclusion

- `rootvel_zero_slack_contract_hold_resolved`: `true`
- `authorize_8window_after_band_audit`: `true`
- accepted relabel count: `6`
- rejected relabel count: `0`

## Step 1: Zero-Slack Audit

| target | metric | current band | continuous p50/p95/p99 | verdict basis p99 | verdict | new band |
|---|---|---:|---:|---:|---|---:|
| Walk_L_To_L | `bone_angvel_level_rms` | 0.90655224 | 0.49445716 / 0.85901129 / 0.95781331 | 0.95781331 | zero-slack | 0.95781331 |
| Walk_L_To_L | `angvel_step_rms` | 0.59933325 | 0.12832130 / 0.31595567 / 0.54098654 | 0.44095759 | has-slack | 0.59933325 |
| Walk_L_To_L | `angvel_step_component_p95` | 0.78103643 | 0.21348841 / 0.45822885 / 0.65489564 | 0.65489564 | has-slack | 0.78103643 |
| Walk_L_To_L | `rootvel_step_l2` | 0.04241988 | 0.01017606 / 0.04653953 / 0.08486664 | 0.08486664 | zero-slack | 0.08486664 |
| Walk_L_To_L | `yaw_rate_step_abs` | 0.13711452 | 0.00000000 / 0.09979570 / 0.13637633 | 0.13637633 | has-slack | 0.13711452 |
| Walk_L_To_L | `contact_step_l2` | 0.64826574 | 0.00501935 / 0.27916169 / 1.03706704 | 0.25172432 | has-slack | 0.64826574 |
| Walk_L_To_L | `foot_slip_contacted_speed_mps` | 2.76610528 | 1.48907870 / 2.40602362 / 2.74282146 | 2.74282146 | has-slack | 2.76610528 |
| Walk_L_To_L | `heading_error_rad` | 0.00001000 | 0.00000000 / 0.00000001 / 0.00000002 | 0.00000002 | has-slack | 0.00001000 |
| Walk_L_To_L | `pose_step_l2` | 0.01174183 | 0.00543954 / 0.00916377 / 0.01088832 | 0.01088832 | has-slack | 0.01174183 |
| Walk_L_To_R | `bone_angvel_level_rms` | 1.17852692 | 0.75137148 / 1.11261851 / 1.16978010 | 1.16978010 | audit-only | 1.17852692 |
| Walk_L_To_R | `angvel_step_rms` | 0.44825796 | 0.14488690 / 0.30952081 / 0.43992781 | 0.42878167 | audit-only | 0.44825796 |
| Walk_L_To_R | `angvel_step_component_p95` | 0.58998867 | 0.26408717 / 0.50714992 / 0.56018251 | 0.56018251 | audit-only | 0.58998867 |
| Walk_L_To_R | `rootvel_step_l2` | 0.08347862 | 0.01583235 / 0.04975078 / 0.09763428 | 0.09763428 | audit-only | 0.08347862 |
| Walk_L_To_R | `yaw_rate_step_abs` | 3.18217783 | 0.00000000 / 0.72319241 / 2.49722714 | 2.49722714 | audit-only | 3.18217783 |
| Walk_L_To_R | `contact_step_l2` | 1.19979321 | 0.00417995 / 0.34839994 / 1.05408543 | 0.25322337 | audit-only | 1.19979321 |
| Walk_L_To_R | `foot_slip_contacted_speed_mps` | 3.37338234 | 1.50492758 / 2.41751097 / 3.31639791 | 3.31639791 | audit-only | 3.37338234 |
| Walk_L_To_R | `heading_error_rad` | 0.00001000 | 0.00000000 / 0.00000003 / 0.00000004 | 0.00000004 | audit-only | 0.00001000 |
| Walk_L_To_R | `pose_step_l2` | 0.01461496 | 0.00651983 / 0.01293888 / 0.01429338 | 0.01429338 | audit-only | 0.01461496 |
| Walk_R_To_L | `bone_angvel_level_rms` | 1.13828263 | 0.62141299 / 1.00958120 / 1.06040019 | 1.06040019 | has-slack | 1.13828263 |
| Walk_R_To_L | `angvel_step_rms` | 0.70971101 | 0.15258304 / 0.44360481 / 0.62244291 | 0.64734723 | has-slack | 0.70971101 |
| Walk_R_To_L | `angvel_step_component_p95` | 0.75483635 | 0.24059312 / 0.54997405 / 0.74402431 | 0.74402431 | has-slack | 0.75483635 |
| Walk_R_To_L | `rootvel_step_l2` | 0.04802140 | 0.01622772 / 0.04564989 / 0.07909000 | 0.07909000 | zero-slack | 0.07909000 |
| Walk_R_To_L | `yaw_rate_step_abs` | 1.05991906 | 0.00000000 / 0.34482282 / 1.05991906 | 1.05991906 | has-slack | 1.05991906 |
| Walk_R_To_L | `contact_step_l2` | 0.56536492 | 0.01283675 / 0.42157097 / 0.77950967 | 0.45344730 | has-slack | 0.56536492 |
| Walk_R_To_L | `foot_slip_contacted_speed_mps` | 3.76081495 | 1.56516856 / 3.21158212 / 3.69829576 | 3.69829576 | has-slack | 3.76081495 |
| Walk_R_To_L | `heading_error_rad` | 0.00001000 | 0.00000000 / 0.00000002 / 0.00000004 | 0.00000004 | has-slack | 0.00001000 |
| Walk_R_To_L | `pose_step_l2` | 0.01368798 | 0.00675358 / 0.01189787 / 0.01317200 | 0.01317200 | has-slack | 0.01368798 |
| Walk_R_To_R | `bone_angvel_level_rms` | 0.89982127 | 0.53455947 / 0.90917130 / 1.01856762 | 1.01856762 | zero-slack | 1.01856762 |
| Walk_R_To_R | `angvel_step_rms` | 0.50560947 | 0.11908435 / 0.27086144 / 0.46921032 | 0.43395288 | has-slack | 0.50560947 |
| Walk_R_To_R | `angvel_step_component_p95` | 0.59696580 | 0.20206037 / 0.42889074 / 0.56557086 | 0.56557086 | has-slack | 0.59696580 |
| Walk_R_To_R | `rootvel_step_l2` | 0.07419514 | 0.01064398 / 0.04667915 / 0.10134733 | 0.10134733 | zero-slack | 0.10134733 |
| Walk_R_To_R | `yaw_rate_step_abs` | 0.98427995 | 0.00298936 / 0.26265626 / 0.88997168 | 0.88997168 | has-slack | 0.98427995 |
| Walk_R_To_R | `contact_step_l2` | 1.30524075 | 0.01504992 / 0.52714229 / 1.21021468 | 0.29297250 | has-slack | 1.30524075 |
| Walk_R_To_R | `foot_slip_contacted_speed_mps` | 2.30164003 | 1.24095815 / 2.29290112 / 2.39659229 | 2.39659229 | zero-slack | 2.39659229 |
| Walk_R_To_R | `heading_error_rad` | 0.00001000 | 0.00000000 / 0.00000002 / 0.00000003 | 0.00000003 | has-slack | 0.00001000 |
| Walk_R_To_R | `pose_step_l2` | 0.01226273 | 0.00623281 / 0.01044192 / 0.01217733 | 0.01217733 | has-slack | 0.01226273 |

## Step 2: Continuous-Percentile Relabels

| target | metric | old band | new band | basis |
|---|---|---:|---:|---|
| Walk_L_To_L | `bone_angvel_level_rms` | 0.90655224 | 0.95781331 | continuous p99.0 with no tightening |
| Walk_L_To_L | `rootvel_step_l2` | 0.04241988 | 0.08486664 | continuous p99.0 with no tightening |
| Walk_R_To_L | `rootvel_step_l2` | 0.04802140 | 0.07909000 | continuous p99.0 with no tightening |
| Walk_R_To_R | `bone_angvel_level_rms` | 0.89982127 | 1.01856762 | continuous p99.0 with no tightening |
| Walk_R_To_R | `rootvel_step_l2` | 0.07419514 | 0.10134733 | continuous p99.0 with no tightening |
| Walk_R_To_R | `foot_slip_contacted_speed_mps` | 2.30164003 | 2.39659229 | continuous p99.0 with no tightening |

## Step 3: Guard Results

| relabel | old band | new band | one-window | shortcut neg | command neg | guard identity | accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Walk_L_To_L:bone_angvel_level_rms` | 0.90655224 | 0.95781331 | true | true | true | true | true |
| `Walk_L_To_L:rootvel_step_l2` | 0.04241988 | 0.08486664 | true | true | true | true | true |
| `Walk_R_To_L:rootvel_step_l2` | 0.04802140 | 0.07909000 | true | true | true | true | true |
| `Walk_R_To_R:bone_angvel_level_rms` | 0.89982127 | 1.01856762 | true | true | true | true | true |
| `Walk_R_To_R:rootvel_step_l2` | 0.07419514 | 0.10134733 | true | true | true | true | true |
| `Walk_R_To_R:foot_slip_contacted_speed_mps` | 2.30164003 | 2.39659229 | true | true | true | true | true |

## Final State

- one-window full-family pass under accepted relabels: `true`
- shortcut negative controls still fail: `true`
- command demotion negative controls still fail: `true`
- reconstructed GT acceptance: `1.0000`
- decoder-path-from-GT acceptance: `1.0000`
- `max_abs_seq_delta`: `0.00000000`

Artifacts:

- `debug_output/_tmp_action_handoff_band_audit_20260604/summary.md`
- `debug_output/_tmp_action_handoff_band_audit_20260604/per_metric.csv`
- `debug_output/_tmp_action_handoff_band_audit_20260604/band_audit_summary.json`
