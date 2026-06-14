> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# W1b Direction Decision - Action-Handoff In-Betweening

Date: 2026-05-30

Status: **recommend A: add grounded data**, with **B: masked in-betweening** as the fallback
if data collection/re-authoring is blocked.

Primary evidence:

- W0/W1a locked that the old PHASE2 `Walk_L_To_R` positive was a metric artifact:
  `debug_output/_tmp_action_handoff_inbetween_reach_honesty_20260530_w1a_exact_trained/reach_honesty_probe_summary.json`.
- W1b migrated gate artifact:
  `debug_output/_tmp_action_handoff_inbetween_gate_migration_20260530/gate_migration_eval_summary.json`.
- W1b positive control passed, so the migrated gate is not an always-no gate.

## Decision

Recommend **A. Add grounded data**, specifically:

1. Give `Walk_L_To_R` at least one grounded onset / re-authored onset that clears the current
   full-state groundability gate.
2. Add a small number of locomotion bridge clips around the same onset/contact regimes, so
   the model sees the transition as a supervised action path rather than a latent attractor
   write.

This is the only option with an O(1) marginal fix aimed at the observed failure mode:
`Walk_L_To_R` has the concentrated risk because it lacks grounded onset supervision. The
current evidence says the model can be made to write hidden_pre, but not to realize the turn
as motion under free rollout.

## Why Not Another Injection Point / Fine-Tune

Do **not** run a fourth latent-lever round.

Three rounds now point to the same boundary:

- Old radius gate accepted `Walk_L_To_R` only under the pinned artifact: legacy pinned
  radius reach `0.75`, but yaw corr `-0.80`, heading MAE `43.5 deg`, and `pop_safe=0.00`.
- Under the migrated same-source free rollout gate, PHASE2 `Walk_L_To_R` is rejected:
  self-reach `k=3` rate `0.00`, yaw corr `-0.48`, heading MAE `39.6 deg`,
  `pop_safe=0.00`, joint pass `False`.
- The same gate accepts real recorded `Walk_L_To_R` motion: self-reach `k=3` rate `1.00`,
  yaw corr `1.00`, heading MAE approximately `0 deg`, `pop_safe=1.00`,
  `best_pose_d=0.0`, joint pass `True`.
- Anchor mismatch remains material: trained fullseq hidden_pre vs legacy saved anchor has
  relerr `0.123` for `Walk_L_To_R` and `0.123..0.138` across turn clips. Defaulting back
  to frozen/saved anchors would reintroduce the W1a measurement error.

Mechanism: the latent lever can move `hidden_pre` enough to satisfy a loose/pinned radius
read, but the generated root-velocity heading and contact/pop behavior do not turn. That is
not an injection-depth problem anymore; it is missing grounded action supervision and/or the
wrong generation formulation.

## Minimal Falsifiable Next Step For A

Data requirement:

- Add or re-author one `Walk_L_To_R` grounded onset clip whose onset aligns to `Walk_F`
  with current thresholds: `contact_d <= 0.30` and `pose_d <= 0.05`.
- Add a few locomotion bridge clips that cover the same contact/onset neighborhoods, not
  only terminal turn poses.

Minimum acceptance before any larger run:

- `tools/run_action_handoff_grounded_alignment_check.py` reports `Walk_L_To_R` as
  groundable without within-clip fallback.
- The sampler can draw `Walk_L_To_R` grounded samples with real onset provenance, not only
  fallback metadata.
- A short smoke using the migrated W1b gate must improve `Walk_L_To_R` over no-goal/pinned
  baselines by the joint criterion: self-reach `k=3` lift >= `0.10`, yaw corr `>0`,
  heading MAE `<0.25 rad`, `pop_safe_rate >0`, and `best_pose_d` non-degrading.
- Recorded-turn positive control must remain pass; otherwise the gate or data transform is
  broken.

Fail condition:

- If the new grounded onset clears coverage but the migrated gate still has `Walk_L_To_R`
  yaw corr `<=0` or `pop_safe_rate=0`, stop treating missing data as the sole cause and move
  to the masked-formulation fallback.

## Backup: B Masked In-Betweening

Use B only if A is blocked or A clears coverage but fails the smoke above.

Expected difference on the current 5-clip setting:

- AR free rollout is exposure-bias sensitive and already collapses from pinned reach to free
  reach `0.00` on `Walk_L_To_R`.
- A masked in-betweening smoke should condition on both context and future seam tokens
  directly, so it should reduce middle/seam drift if the issue is generation formulation
  rather than data coverage.

Minimum smoke:

- Same 5 clips, same W1b gate, no B4/seam runtime.
- Compare masked vs current AR on per-clip rows, with `Walk_L_To_R` separate.
- Masked must preserve recorded positive pass and show at least one action-layer lift
  (`yaw corr >0` or `pop_safe_rate >0`) where AR free remains negative; otherwise it is not
  worth expanding before data improves.

## Why Not C Park

Do not park yet. W1b proved the gate is discriminative: it rejects the artifact and accepts
real recorded turns. That gives a clean next falsification path. Parking would be reasonable
only after A or B fails under this gate.

## Largest Risk / Blind Spot

The recommendation assumes the dominant failure is missing grounded `Walk_L_To_R` supervision.
One grounded onset may still be insufficient if the base model's action manifold cannot absorb
the transition, or if the AR formulation is fundamentally wrong for in-betweening. The
provisional `tau_yaw=0.25 rad` is intentionally conservative and must remain reported as
PROVISIONAL until more positive/negative controls exist.
