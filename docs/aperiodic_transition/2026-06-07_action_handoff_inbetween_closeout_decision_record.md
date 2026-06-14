# Action-Handoff Inbetweening Closeout Decision Record

Date: 2026-06-07

Status: CLOSEOUT / DECISION RECORD / READ-ONLY. This record closes the
exploration question. It does not claim the kinematics-only delivery contract
has been validated or shipped.

## 1. Core Question and Scope-Limited Answer

Question, in the investigation scope:

> On the delivered model, with zero new injection and read-only use, can the
> support components of a goal be made controllable?

Decision: **No, under this scope there is no handle.**

The scope qualifier is part of the decision. The claim is:

- Delivered `EventMotionModel` stage7 lambda checkpoint, zero new injection,
  read-only rollout: no controllable support-goal handle was found.
- This is not a claim that support control is impossible in principle. That
  would require either a trained latent probe or a trained goal-injection path,
  which is outside this read-only investigation.

Delivered-model facts:

- The forward boundary has no `goal` argument. Inputs are
  `state`, `cond`, `contacts`, `angvel`, `pose_history`, `plan_z`, `phase_z`,
  event-age/time fields, and rollout step fields
  (`train/models.py:5475`).
- The delivered stage7 lambda log records `contact_plan_inject="plan_z"` and
  `direct_pose_use_phase_z=false`, with phase mode still set to `concat` in the
  config/log (`debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/lambda/checkpoints/posttrain_log_WalkF_stage7_lambda_from_lowlr72_lr1e4_20260504.json:37`,
  `debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/lambda/checkpoints/posttrain_log_WalkF_stage7_lambda_from_lowlr72_lr1e4_20260504.json:86`,
  `debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/lambda/checkpoints/posttrain_log_WalkF_stage7_lambda_from_lowlr72_lr1e4_20260504.json:87`).
- The action-handoff state schema is fixed at `STATE_DIM=281`:
  `pose_rot6d[276] + ego_vel[2] + yaw_rate[1] + contact[2]`
  (`train/data/action_handoff_inbetween.py:9`,
  `train/data/action_handoff_inbetween.py:41`).
- `cond` is `act_oh[4] + cond_dir[2] + cond_speed[1]`; the independent command
  DOF are only `cond_dir[2] + cond_speed[1]`
  (`train/data/action_handoff_inbetween.py:69`).
- `yaw_rate` is the finite-difference derivative of `cond_dir`, not an
  additional independent command DOF (`train/data/action_handoff_inbetween.py:111`).
- The current curriculum gap band is provisional:
  `GAP_MIN=12`, `GAP_MAX=30`
  (`train/data/action_handoff_inbetween.py:77`,
  `train/data/action_handoff_inbetween.py:78`). `gap_for_progress` clips
  progress and interpolates in that range
  (`train/data/action_handoff_inbetween.py:305`).

Tensor contract used by the closeout: per-frame action-handoff state is
`[B,T,281]` or `[1,H,281]`, `float32`, on the caller/evaluation device unless an
artifact explicitly materializes CPU metrics. `cond` is `[B,T,7]` or `[1,H,7]`,
`float32`, on the rollout device. The only free realized output this decision
treats as motion is the rot6d pose slice `[276]`.

## 2. Regime-Basin Statement

The end/goal is not a fixed frame. The accepted formulation already reframed it
as a soft target regime or basin, and stopped reopening endpoint-definition
search (`docs/aperiodic_transition/2026-06-01_middle_generator_acceptance_contract.md:38`,
`docs/aperiodic_transition/2026-05-31_action_handoff_inbetween_soft_endpoint_reframe.md:48`).

The rollout behaves like an attractor system that falls into regime basins:

- Kinematics basin: controllable enough for delivery. `cond_dir[2]` and
  `cond_speed[1]` select heading/speed regimes. This is the real meaning of
  "kinematics controllable" in this model.
- Support/phase basin: basins exist, but there is no goal-associated selector
  in the delivered read-only interface. Support arrival is a function of entry
  phase and gap, weakly parameterized by initial condition and duration. `cond`
  does not select support, and gap selection collapses across starts.

The uncontrollable object is therefore not "endpoint anchoring" in general. The
uncontrollable object is: **the support basin has no controllable goal-linked
selector latent exposed by the delivered model under zero-injection read-only
use.**

## 3. Converged Evidence

1. `cond -> support` closeout is negative.

   The counterfactual closeout found strong command movement but no support
   arrival control. For `alpha=8`, `arrival_changed=0/19` while
   `transition_changed=41/42`; endpoint probability moved by about
   `0.30198883032735735`, but not toward the requested target
   (`debug_output/20260606_cond_contactplan_counterfactual_closeout_summary.md:20`).
   The same artifact records the in-regime sample size
   `n_pairs=21`, `n_source_target_rows=42`, independent source clips `5`, and
   independent source regions `7`
   (`debug_output/20260606_cond_contactplan_counterfactual_closeout_summary.md:6`,
   `debug_output/20260606_cond_contactplan_counterfactual_closeout_summary.md:8`).

2. Gap selection does not survive cross-start evaluation.

   The gap-selection contract summary reports in-regime reachability:
   right support `0/5`, left support `1/5`
   (`debug_output/20260606_gap_selection_goal_contract_summary.json:385`,
   `debug_output/20260606_gap_selection_goal_contract_summary.json:394`,
   `debug_output/20260606_gap_selection_goal_contract_summary.json:362`,
   `debug_output/20260606_gap_selection_goal_contract_summary.json:371`).
   The attractor failure CSV has 53 failure rows excluding the header, with
   failures dominated by `right_to_flight_or_unknown=36` and
   `left_to_flight_or_unknown=10`
   (`debug_output/20260606_gap_selection_goal_contract_attractor_failures.csv:1`,
   `debug_output/20260606_gap_selection_goal_contract_attractor_failures.csv:22`,
   `debug_output/20260606_gap_selection_goal_contract_attractor_failures.csv:45`).
   The artifact's own verdict is that gap selection is weaker than the
   single-start result and must not be treated as a reliable command outside
   calibrated cells
   (`debug_output/20260606_gap_selection_goal_contract_summary.json:1561`).

3. The determinant probe explains the false optimism: support arrival is
   `f(entry phase, gap)`, but the cleanest free-run in-regime evidence has
   `eff_n=1`.

   In the determinant artifact, branch C free-run in-regime has
   `effective_n_units=1`, `gap_min=12`, `gap_max=84`, and target/phase
   prediction accuracy `0.8421052631578947`
   (`debug_output/20260606_support_arrival_determinant_summary.json:244`,
   `debug_output/20260606_support_arrival_determinant_summary.json:245`,
   `debug_output/20260606_support_arrival_determinant_summary.json:246`,
   `debug_output/20260606_support_arrival_determinant_summary.json:256`).
   That is compatible with phase-plus-duration determinism, not with a robust
   cross-start goal handle.

   The phase-matched playback-vs-control red-team probe closes the remaining
   ambiguity. It sampled `{left,right} x {q0,q1,q2,q3}` with two independent
   entries per cell, two seeds per entry, and gap `12..84` step `1`; delivery
   rows are `gap=12..30`. The rows-only recalc reports
   `PLAYBACK-CONFIRMED`, with `row_n=4672`, `pair_n=2336`,
   `delivery_pair_n=608`, and `effective_entry_n=16`
   (`debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:2`,
   `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:6`,
   `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:7`,
   `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:8`,
   `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:9`).
   Free-run support agrees with the teacher model clock at `0.8322368421052632`
   over delivery pairs, or `0.87` when both sides are internally planted; the
   corresponding drift floor is `0.16776315789473684`
   (`debug_output/_tmp_support_playback_vs_control_20260607_full_v2/clock_consistency.json:15`,
   `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/clock_consistency.json:16`,
   `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/clock_consistency.json:17`).
   The side-asymmetry is not new free-run control: absolute side symmetry is low
   for both free and teacher, but free matches the teacher left/right pattern at
   `0.875`, the symmetry-flag match is `0.890625`, and the absolute
   free-vs-teacher symmetry delta is only `0.02176696542893726`
   (`debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:19`,
   `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:20`,
   `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:21`,
   `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:22`,
   `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:23`).
   Clock order is not compressed beyond drift: stress `extra_free_transition_rate`
   is `0.013157894736842105` against a drift-plus-margin budget of
   `0.21776315789473683`
   (`debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:24`,
   `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:25`).
   The clean support metric did not reintroduce binary flight: delivery
   `free_true_flight_rate=0.0`; skate-like rows are diagnostic rendering/FK
   artifacts at `0.1611842105263158`
   (`debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:28`,
   `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:29`).
   Therefore gap is a playback-class readout scheduler for a known entry phase,
   not an independent support-goal steering DOF.

4. `phase_z` is retired as a support-control lever.

   The strict replace boundary probe's zero-new-cols path is numerically better
   than preserving the phase columns: `all_ex_root.mean` improves from
   `0.3291262984275818` to `0.21090209484100342`, and leg mean improves from
   `0.7763916552066803` to `0.43088141083717346`
   (`debug_output/_tmp_strict_replace_phasez_boundary_probe_20260427_211020/baseline/evals/last/group_summary.json:45`,
   `debug_output/_tmp_strict_replace_phasez_boundary_probe_20260427_211020/zero-new-cols/evals/last/group_summary.json:45`,
   `debug_output/_tmp_strict_replace_phasez_boundary_probe_20260427_211020/baseline/evals/last/group_summary.json:13`,
   `debug_output/_tmp_strict_replace_phasez_boundary_probe_20260427_211020/zero-new-cols/evals/last/group_summary.json:13`).

5. `GoalHead` exists only as a trained-injection path, not as read-only evidence.

   The helper is explicitly a goal head injected into a frozen base with
   `L_reach`-only short training, not a production read-only handle
   (`train/action_handoff_inbetween_goal_injection.py:6`,
   `train/action_handoff_inbetween_goal_injection.py:27`).
   The head maps a seam window `[K,281]` into an injection vector
   (`train/action_handoff_inbetween_goal_injection.py:61`,
   `train/action_handoff_inbetween_goal_injection.py:62`). Prior review already
   recorded the small-data mapping trap: Step 2 kept reach-rate at `0` for all
   targets, and the closest Phase 1 result was still about `5.8x` self reach
   (`docs/aperiodic_transition/2026-05-30_action_handoff_inbetween_72_73_review_record.md:106`,
   `docs/aperiodic_transition/2026-05-30_action_handoff_inbetween_72_73_review_record.md:107`,
   `docs/aperiodic_transition/2026-05-30_action_handoff_inbetween_72_73_review_record.md:209`).

6. Natural goal quality is concentrated in a few cells and does not prove
   transition control.

   The canonical inventory concentrates mass in
   `right->right = 0.346590`, `left->right = 0.256230`, and
   `right->left = 0.206190`
   (`debug_output/20260606_canonical_transition_inventory.csv:7`,
   `debug_output/20260606_canonical_transition_inventory.csv:11`,
   `debug_output/20260606_canonical_transition_inventory.csv:8`).
   The largest cell, `right->right`, has
   `without_adjacent_physical_event=True`, so it is a steady-cell artifact, not
   transition-capability evidence
   (`debug_output/20260606_canonical_transition_inventory.csv:7`).

## 4. False Blockers Ruled Out in This Thread

These were the useful reductions from the final red-team pass. They should not
be reopened as support-control blockers without new evidence.

1. FK contact skate/flight was a command-root/FK-prior diagnostic issue, not a
   pose collapse.

   `apply_free_carry_raw` writes the next rot6d pose, then overwrites root
   velocity and root position from the command direction/speed
   (`train/rollout_kernel.py:298`,
   `train/rollout_kernel.py:330`,
   `train/rollout_kernel.py:336`). FK contact measurement is an external
   skeleton-prior diagnostic in world coordinates, and its soft score is a
   product of distance, vertical-velocity, and horizontal-velocity terms
   (`train/validate/contact_meas_whitebox.py:252`,
   `train/validate/contact_meas_whitebox.py:331`).

   The soft full-gap probe demoted the scary binary flight reading: native free
   flight was `54.57%`, but teacher was also `33.79%`; the red-team recalc shows
   the dominant drop driver was horizontal velocity, not distance
   (`debug_output/_tmp_freerun_contact_stability_probe_20260606_soft_fullgap/`,
   `debug_output/_tmp_freerun_contact_stability_probe_20260606_soft_fullgap/20260606_freerun_contact_stability_probe_soft_fullgap_redteam_recalc.json:115`,
   `debug_output/_tmp_freerun_contact_stability_probe_20260606_soft_fullgap/20260606_freerun_contact_stability_probe_soft_fullgap_redteam_recalc.json:116`).

2. "Witness does not drop" was a frozen `CONTACT_SLICE`, not real support.

   `CONTACT_SLICE` is part of the 281 state
   (`train/data/action_handoff_inbetween.py:51`), but `apply_free_carry_raw`
   clones `x_prev` and writes rot6d/root/angvel fields without writing
   `CONTACT_SLICE`
   (`train/rollout_kernel.py:287`,
   `train/rollout_kernel.py:298`,
   `train/rollout_kernel.py:318`,
   `train/rollout_kernel.py:330`,
   `train/rollout_kernel.py:336`). Therefore the state contact slice is a
   carry witness/dead channel in free-run, not motion truth. The real model-side
   contact output for this path is `ret["contacts_plan"]`, and even that remains
   Layer-3 witness-only for this decision.

3. "Free angvel << teacher angvel" was a teacher-reset inflation artifact.

   `apply_free_carry_raw` computes `angvel` from previous/current rot6d
   (`train/rollout_kernel.py:318`). The probe's teacher mode resets the raw
   motion input from ground truth every step, while free-run carries the previous
   rollout state
   (`tools/run_action_handoff_freerun_contact_stability_probe.py:467`,
   `tools/run_action_handoff_freerun_contact_stability_probe.py:509`,
   `tools/run_action_handoff_freerun_contact_stability_probe.py:548`).
   The recomputation closed the native free arm scale:
   `angvel_abs_rms ~= pose_step(rad) * 60`, approximately `0.339 ~= 0.343`.
   Teacher arms were inflated by roughly `10x-18x`; the large z-score near
   `-118` was this reset artifact plus floor amplification, not free-run pose
   damping
   (`debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json`).

4. Pose degradation was not present.

   The FK-free Layer-1 recomputation used `rows=2336`, `valid_rows=2160`,
   `paired_records=1080`, native `eff_n=12`, and goal `eff_n=4`
   (`debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json:5`,
   `debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json:6`,
   `debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json:25`,
   `debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json:33`,
   `debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json:117`).
   Native free vs teacher was not worse:
   `pose_step_geo 0.3268567 vs 0.3349362`,
   `manifold_z 0.3294380 vs 0.3346955`,
   `knn 0.2368162 vs 0.2416067`
   (`debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json:50`,
   `debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json:51`,
   `debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json:57`,
   `debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json:58`,
   `debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json:64`,
   `debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json:65`).
   `contacts_meas_finite_n_total=0`, by design, because this recomputation is
   Layer-1 only and does not use FK/contact values
   (`debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json:26`).

Net effect: the unknown architecture-level stability blocker collapsed. The
remaining conclusion is the older, narrower one: support is not controllable
under the delivered-model / zero-injection / read-only scope.

## 5. Delivery Contract (a): Kinematics-Only

The only delivery artifact recognized by this closeout is the spec-only
kinematics contract:

`docs/aperiodic_transition/2026-06-07_kinematics_only_delivery_contract.md`

Its honest capability claim is:

- Given command-determined kinematics, produce legal on-manifold walk pose.
- The only free realized output is `rot6d` pose `[276]`.
- Root/heading/ego state are command/carry pipeline fields.
- Contact/support channels are not delivered as truth.

The contract explicitly records this model/checkpoint scope
(`docs/aperiodic_transition/2026-06-07_kinematics_only_delivery_contract.md:11`),
the `[B,C,281]`, `[B,H,281]`, `[B,K,281]`, and `[B,T,7]` tensor contract
(`docs/aperiodic_transition/2026-06-07_kinematics_only_delivery_contract.md:14`,
`docs/aperiodic_transition/2026-06-07_kinematics_only_delivery_contract.md:18`),
the no-goal forward boundary
(`docs/aperiodic_transition/2026-06-07_kinematics_only_delivery_contract.md:27`),
and the only-free-output claim
(`docs/aperiodic_transition/2026-06-07_kinematics_only_delivery_contract.md:31`).

The three gates remain:

- Gate A: legality.
- Gate B: command/carry consistency, pipeline-only and zero capability claim.
- Gate C: pose Layer-1 sanity, the only capability gate, with two-sided bands.

Open v0.1 tightening remains outside this closeout: Gate C frozen lower bounds
should become command-conditioned because slow commands naturally have lower
articulation, and the gap band still inherits its provisional status.

## 6. Explicit Scope-Out

R_To_R red-team closeout note:

- `Walk_R_To_R` is not a source-animation or coverage failure under the 5 locked
  clips. The read-only source-vs-teacher-vs-free recalc reports GT-source skate
  `0.08695652173913043`, teacher skate `0.15217391304347827`, and free skate
  `0.5108695652173914`; `free - teacher = 0.3586956521739131`.
- The same table rules out the simple data explanations: `Walk_R_To_R` has the
  largest cond-aligned frame count (`93`), teacher reproduction is mid-pack, and
  kinematics are not the most extreme axis. The failure signature is
  `FREE-RUN-SUPPORT-CARRY-GAP`, not "model did not learn the clip."
- Because the current checkpoint is the fast Walk_F posttrain checkpoint, this
  cell is recorded as `POSTTRAIN-COVERAGE-PENDING`. The closeout test is to rerun
  `debug_output/_tmp_why_r_to_r_special_20260607_v1/why_r_to_r_special_readonly.py --checkpoint <turn_posttrain_ckpt>`
  on a turn-posttrain checkpoint, read the emitted per-clip
  `free_teacher_skate_gaps` map and `other_gap_max`
  (`debug_output/_tmp_why_r_to_r_special_20260607_v1/why_r_to_r_special_readonly.py:680`,
  `debug_output/_tmp_why_r_to_r_special_20260607_v1/why_r_to_r_special_readonly.py:690`),
  and judge by a **relative-outlier** criterion against the turn distribution.
- Closeout criterion (refined, relative): the cell closes when, with all turns
  posttrained, the `Walk_R_To_R` free-teacher skate gap is **no longer an outlier
  within the posttrained-turn gap band** (judged against the same-difficulty
  posttrained turns, not pegged to the Walk_F absolute scale). This supersedes the
  earlier absolute target (`~0.12-0.15`, Walk_F scale): Walk_F is the easiest
  topology and `Walk_R_To_R` is the hardest (densest, longest support switching),
  so the gap can legitimately settle higher than Walk_F after posttrain and still
  be in-band. The embedded `expected_if...` hint string in the verifier
  (`debug_output/_tmp_why_r_to_r_special_20260607_v1/why_r_to_r_special_readonly.py:688`)
  predates this refinement and still names the absolute scale; judge from the
  emitted gap map, not from that string.
- Explicitly two-sided falsifiable:
  - Gap falls into the posttrained-turn band (no longer an outlier) → the
    `FREE-RUN-SUPPORT-CARRY-GAP` / `POSTTRAIN-COVERAGE-PENDING` hypothesis is
    confirmed, `Walk_R_To_R` closes and the cell is deliverable.
  - Gap remains an outlier above the posttrained-turn band → the hypothesis is
    falsified; `Walk_R_To_R` is a genuine free-run-carry weakness that posttrain
    does not fix, and the cell falls back to **drop/accept** (marked as a
    not-delivered cell for the current checkpoint), not reopened as data or
    source-animation work.

The following are not delivered, not pass/fail criteria, and not reopened by
this decision record:

- Contact honesty.
- Support-side phase lock to arbitrary `seam_target`.
- FK contact, skate, flight, or foot-slip quality.
- `contacts_plan`, `plan_z`, `phase_z`, or state `CONTACT_SLICE` as motion
  truth.
- Any seam-goal-following claim.

Layer discipline for future readers:

- Layer 1, rot6d pose `[276]`, is the motion truth used here.
- Layer 2, FK/contact diagnostics, carries an external skeleton prior and is
  diagnostic only.
- Layer 3, `contacts_plan`, `plan_z`, `phase_z`, and `CONTACT_SLICE`, is witness
  only. `CONTACT_SLICE` is frozen under the current free-run carry.

## 7. Optional Future Hook

If support-goal control is reopened later, the entry question should be:

> Is the support basin separable in latent space, and can it be selected by a
> goal?

Answering that requires work outside this closeout scope:

- A trained latent probe, which violates the read-only boundary of this
  investigation; or
- The `GoalHead` path, which is training-dependent and already carries a
  small-data mapping risk
  (`train/action_handoff_inbetween_goal_injection.py:6`,
  `docs/aperiodic_transition/2026-05-30_action_handoff_inbetween_72_73_review_record.md:209`).

This is an optional future hook, not a TODO.

## 8. Exploration Closeout Is Not Delivery Validation

This record closes the exploration question: whether support-goal control can
be extracted from the delivered model under zero-injection read-only use.

It does not validate the kinematics-only delivery contract. That is a separate
delivery task: calibrate GT bands, run the three gates, and issue PASS/FAIL with
artifacts. No such validation is claimed here.

## 9. Anti-Relitigation Discipline

Future work should not reopen the following traps without new evidence from
rows/forward recomputation and independently counted `eff_n`:

- Treating FK contact as the success axis.
- Treating debug/internal channels as realized motion.
- Treating teacher-reset angvel as a free-run stability signal.
- Treating binary contact thresholds as proof of flight.
- Using a two-frame endpoint proxy as the support-goal definition.
- Using oracle labels as a control handle.
- Leaking adjacent windows into transition evidence.
- Counting stride/crop denominators as independent samples.
- Using raw rot6d L2 as a meaningful geometry metric without redundancy checks.
- Letting the three-factor soft contact product self-collapse drive the story.
- Calling root command response a learned command-following proof when the
  carry explicitly writes root velocity/position from `cond`.

Negative conclusions must keep the scope qualifier:

**Delivered model + zero new injection + read-only has no support-goal handle.**
