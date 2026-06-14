# Kinematics-Only Delivery Contract

Date: 2026-06-07

Status: **SPEC-ONLY v0**. This document defines the narrow kinematics-only delivery
contract for action-handoff goal-conditioned inbetweening. It does not train a model,
run a forward pass, write a checkpoint, change a gate, or add a probe.

## 1. Scope

This contract covers only the current `EventMotionModel` stage7 lambda checkpoint
surface under the existing free-run carry:

- Sampling tensors: `ctx [B,C,281]`, `gt_middle [B,H,281]`, `seam_target [B,K,281]`,
  dtype `float32`, on the caller/eval device.
- Canonical 281 layout: `pose_rot6d[276] + ego_vel[2] + yaw_rate[1] + contact[2]`.
  Source: `train/data/action_handoff_inbetween.py:39-51`.
- Command tensor used by the delivered model path: `cond [B,T,7]`, dtype `float32`,
  on the rollout device. Its independent command DOF are `cond_dir[2]` and
  `cond_speed[1]`; the first four channels are action one-hot. Source:
  `train/data/action_handoff_inbetween.py:68-70`.
- `yaw_rate` is not an independent command handle. It is a deterministic derivative
  view of `cond_dir`: `heading_t = atan2(cond_dir_y, cond_dir_x)`,
  `yaw_rate_t = wrap(heading_t - heading_{t-1}) * fps`. Source:
  `train/data/action_handoff_inbetween.py:111-129`.

The delivered model forward surface has no `goal` argument. It accepts
`state/cond/contacts/angvel/pose_history/plan_z/phase_z/...`. Source:
`train/models.py:5475-5505`.

The honest capability claim is therefore narrow:

> Under a command-determined kinematic path, the model's only free realized motion
> output is the rot6d pose slice `[276]`. The root/heading/ego kinematics are command
> application, `angvel` is a carry-derived frame-difference channel, and contact/support
> channels are internal witnesses or carried values, not delivered commands.

## 2. Three-Layer Boundary

The contract must not mix these layers:

- **Layer 1, realized rollout motion:** `next_raw/state_raw` rot6d pose plus root fields
  after carry application. This is the only motion surface used for pass/fail here.
- **Layer 2, measurements on top of motion:** FK contact, foot distance, foot velocity,
  flight, skate, and support honesty. These are diagnostic-only in this contract because
  they import an external skeleton/ground prior.
- **Layer 3, model/carry internals:** `contacts_plan`, `plan_z`, `phase_z`,
  `CONTACT_SLICE`/carry echo, and related debug channels. These are witness-only and
  cannot certify motion success.

## 3. Gate A: Goal Legality Precheck

Type: **legality gate**, FK-free, pre-forward.

A goal is legal only if all conditions hold:

- `gap` is inside the delivery curriculum band. The sampler default is `GAP_MIN=12`,
  `GAP_MAX=30`, and `gap_for_progress` interpolates `12 -> 30`. Source:
  `train/data/action_handoff_inbetween.py:75-79`,
  `train/data/action_handoff_inbetween.py:305-308`.
- The requested goal is expressible by `cond_dir[2]` and `cond_speed[1]`. `yaw_rate`
  may be logged as a derivative of `cond_dir`, but it is not an extra control DOF.
- The goal does not require support-side phase lock, foot-ground contact honesty, or an
  arbitrary contact phase target.
- The requested transition cell belongs to the narrow natural/sampler-supported set.
  Initial whitelist: `right->right`, `left->right`, `right->left`. Diagnostic or
  graylist cells: `flight_or_unknown->right`, `left->flight_or_unknown`.

Canonical inventory evidence:

- `right->right`: sampler frequency `0.346590`, but
  `sampler_endpoint_pair_without_adjacent_physical_event=True`. Treat as a steady-cell
  legality case, not proof of support-transition controllability.
- `left->right`: sampler frequency `0.256230`, physical independent share `0.095238`.
- `right->left`: sampler frequency `0.206190`, physical independent share `0.047619`.
- `flight_or_unknown->right`: sampler frequency `0.079285`.
- `left->flight_or_unknown`: sampler frequency `0.066195`.

Source: `debug_output/20260606_canonical_transition_inventory.csv:3`,
`debug_output/20260606_canonical_transition_inventory.csv:7-11`.

Legality rationale from the full-gap Layer-1 audit:

- Recalc rows: `rows=2336`, `valid_rows=2160`, `paired_records=1080`.
- All `176` invalid rows are structural goal-window failures:
  `goal_window_exceeds_target_clip_max_gap:36` (`96` rows),
  `goal_window_exceeds_target_clip_max_gap:48` (`72` rows), and
  `goal_window_exceeds_target_clip_max_gap:80` (`8` rows).
- This is direct evidence that stress-audit gaps up to `84` must not silently expand the
  delivery band beyond the curriculum contract.

Source:
`debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows_recalc_layer1.json`.

Gate A fail means **illegal/unsupported request**, not model failure.

## 4. Gate B: Command-Application Consistency

Type: **pipeline gate**, FK-free, zero capability claim.

This gate verifies only that the legal command and carry path are applied correctly:

- `cond [B,T,7]` is finite, dtype `float32`, on the rollout device.
- `cond_dir` has finite norm above epsilon before normalization.
- `cond_speed` lies inside the accepted command distribution band for the selected
  target/cell.
- Realized root/heading/ego kinematics equal the command reconstruction within numeric
  tolerance.

This gate is intentionally tautological under the current carry:

- `apply_free_carry_raw` writes only the pose slice from `y_next_raw` into `x_next`.
  Source: `train/rollout_kernel.py:287-299`.
- It computes `vel_world = normalize(cond_dir) * cond_speed`, writes `rootvel`, then
  integrates and writes `rootpos`. Source: `train/rollout_kernel.py:300-342`.
- In canonical 281 construction, `ego_vel[2]` is a projection of `root_vel` under
  `cond_dir`, and `yaw_rate[1]` is derived from `cond_dir`. Source:
  `train/data/action_handoff_inbetween.py:96-108`,
  `train/data/action_handoff_inbetween.py:111-129`.

Therefore Gate B can fail only as a pipeline/carry/config bug. It is not evidence that
the model learned goal following.

Recommended numeric floor:

- `heading_error_rad <= max(GT_continuous_p99, 1e-5)`.
- The `1e-5` rad floor is a numerical tolerance. Band audit true reconstruction errors
  are around `1e-8` rad, e.g. `baseline_p99=2.3915e-08` and
  `effective_current_band=1e-05` for `Walk_L_To_L`.

Source: `debug_output/_tmp_action_handoff_band_audit_20260604/per_metric.csv:9`,
`debug_output/_tmp_action_handoff_band_audit_20260604/per_metric.csv:18`,
`debug_output/_tmp_action_handoff_band_audit_20260604/per_metric.csv:27`,
`debug_output/_tmp_action_handoff_band_audit_20260604/per_metric.csv:36`.

Optional diagnostic, not a gate:

- Log the model's pre-override root prediction in `y_next_raw` versus the command root
  path. This may reveal whether the model would have gone toward the command without
  carry overwrite, but it is not part of acceptance.

## 5. Gate C: Pose Layer-1 Sanity Guard

Type: **capability gate**, FK-free.

This is the only capability gate in the kinematics-only contract. It asks whether the
model produces a legal on-manifold walk pose sequence while the kinematic path is
command-applied.

Inputs for measurement:

- Realized raw rollout state `state_raw [B,T,419]`, dtype `float32` during rollout,
  materialized as CPU `float64` in rows/artifacts.
- Pose slice: `rot6d_x_slice=[5,281]`, width `276`, joint count `46`, columns
  `("X","Z")`.
- Baseline pose distribution: Walk_F raw state `[87,419]`, pose dim `276`, dtype
  `float64`, device `cpu`.
- Z-score floor: `0.05`; in the audit, `pose_std_floored_n=243`.

Source:
`debug_output/_tmp_freerun_layer1_pose_degradation_20260606_fullgap_step1/20260606_freerun_layer1_pose_degradation_fullgap_step1_rows.csv`,
`tools/run_action_handoff_freerun_contact_stability_probe.py:598-657`.

Required metrics:

- `pose_step_geo_deg`: adjacent-frame SO(3) geodesic step from rot6d.
  This must be a **two-sided** band: lower bound prevents frozen/over-smooth pose,
  upper bound prevents jitter/pop.
- `pose_manifold_z_rms`: RMS z distance to the GT Walk_F pose distribution.
  Use an upper band by default; add a lower band if future evidence shows collapse to a
  narrow mean-pose manifold.
- `pose_knn1_z_rms`: nearest-GT pose z distance.
  Use an upper band by default; add a lower band if it becomes a proxy for mode collapse.

Threshold calibration:

- Do not freeze thresholds from this single audit.
- Calibrate final pass/fail bands from GT Walk_F/turn continuous bands, with percentile
  bands per target family and gap bucket.
- `pose_step_geo_deg` must keep both sides. A trajectory with near-zero pose-step fails
  even if it stays on the manifold, because that is over-continuity/frozen motion.
- `manifold_z` and `knn-z` primarily guard off-manifold drift, but should be monitored
  for under-dispersion.

Current read-only evidence, directional only:

- Full-gap FK-free recalc: `rows=2336`, `valid_rows=2160`, `paired_records=1080`.
  `contacts_meas_finite_n_total=0`; `layer1_no_fk_values={"1":2160}`.
- Overall pairs: `n_pairs=1080`, `effective_independent_n=12`, `gap=12..84`.
  `pose_step_geo_deg`: free `0.3269`, teacher `0.3349`, delta `-0.0081`,
  delta p95 `0.2140`.
  `pose_manifold_z_rms`: free `0.3294`, teacher `0.3347`, delta `-0.0053`,
  delta p95 `0.0344`.
  `pose_knn1_z_rms`: free `0.2368`, teacher `0.2416`, delta `-0.0048`,
  delta p95 `0.0579`.
- Goal subset: `n_pairs=204`, `effective_independent_n=4`.
  `pose_step_geo_deg`: free `0.3247`, teacher `0.4309`.
  `pose_manifold_z_rms`: free `0.3307`, teacher `0.4118`.
  `pose_knn1_z_rms`: free `0.2343`, teacher `0.3195`.

These numbers support "no independent pose rollout stability blocker" for this audit.
They do not define permanent thresholds. The goal subset has only four effective
independent units, and the native subset is dominated by Walk_F phase coverage. Treat
single-seed/effective-n-small evidence as directional until GT continuous bands are
locked.

Do not use teacher-reset `angvel` as a Gate C metric:

- Carry computes `angvel` from `x_prev` rot6d and `x_next` rot6d. Source:
  `train/rollout_kernel.py:318-328`.
- The teacher probe resets `motion_raw` each step before carry. Source:
  `tools/run_action_handoff_freerun_contact_stability_probe.py:467-470`,
  `tools/run_action_handoff_freerun_contact_stability_probe.py:507-520`.
- Therefore teacher `angvel` is a one-step mismatch/reset witness, not comparable to
  free-run adjacent-frame `angvel`.

## 6. Explicit Non-Goals

The following are outside pass/fail for this contract:

- Foot-ground contact consistency.
- Support-side phase lock to arbitrary `seam_target`.
- Contact honesty.
- FK contact, flight, skate, foot distance, foot vertical velocity, and foot planar
  velocity. These remain diagnostic-only Layer-2 measurements.
- `contacts_plan`, `contacts_plan_logits`, `plan_z`, `phase_z`, `CONTACT_SLICE`, and
  carry contact echo. These remain Layer-3 witnesses.
- Any claim that the model learned arbitrary goal following. Current forward has no
  `goal` argument, and Gate B is command/carry tautology.

Narrow support scheduling statement:

- If entry phase is already known, `gap` can be used as an in-cycle readout scheduler
  for the model's own gait clock. This is playback-class support scheduling, not a
  support-control DOF. The rows-only playback probe reports `PLAYBACK-CONFIRMED` with
  `effective_entry_n=16`, `delivery_pair_n=608`, delivery free-vs-teacher clock
  agreement `0.8322368421052632`, teacher-relative side-pattern match `0.875`, and
  stress extra-transition rate `0.013157894736842105`
  (`debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:2`,
  `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:8`,
  `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:9`,
  `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:12`,
  `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:21`,
  `debug_output/_tmp_support_playback_vs_control_20260607_full_v2/redteam_recalc.json:24`).

Known posttrain-coverage boundary:

- `Walk_R_To_R` is not treated as a data, animation, or kinematic-extremity
  failure in this contract. The closeout verifier reports GT-source skate
  `0.08695652173913043`, teacher skate `0.15217391304347827`, free skate
  `0.5108695652173914`, and free-teacher gap `0.3586956521739131`.
- The current status for that cell is `POSTTRAIN-COVERAGE-PENDING` /
  `FREE-RUN-SUPPORT-CARRY-GAP`: teacher-forced reproduction is acceptable, but
  free-run support carry is unstable on the same-side dense-switch topology.
- After turn-posttrain exists, rerun
  `debug_output/_tmp_why_r_to_r_special_20260607_v1/why_r_to_r_special_readonly.py --checkpoint <turn_posttrain_ckpt>`
  and read the emitted per-clip `free_teacher_skate_gaps` map and `other_gap_max`
  (`debug_output/_tmp_why_r_to_r_special_20260607_v1/why_r_to_r_special_readonly.py:680`,
  `debug_output/_tmp_why_r_to_r_special_20260607_v1/why_r_to_r_special_readonly.py:690`).
- Closeout criterion (relative, refined): the cell closes when, with all turns
  posttrained, the `Walk_R_To_R` free-teacher gap is no longer an outlier within
  the posttrained-turn gap band. This is a relative judgment against the
  same-difficulty posttrained turns; it is not pegged to the Walk_F absolute scale
  (`~0.12-0.15`), because `Walk_R_To_R` is the hardest topology (densest, longest
  support switching) and may legitimately settle higher than Walk_F yet stay
  in-band. The verifier's embedded absolute hint string
  (`debug_output/_tmp_why_r_to_r_special_20260607_v1/why_r_to_r_special_readonly.py:688`)
  predates this refinement; judge from the emitted gap map.
- Two-sided falsifiable. If the gap falls into the posttrained-turn band, the
  `POSTTRAIN-COVERAGE-PENDING` hypothesis is confirmed and the cell becomes
  deliverable. If the gap stays an outlier above that band, the hypothesis is
  falsified: `Walk_R_To_R` is a posttrain-irreducible free-run-carry weakness and
  the cell falls back to drop/accept (a not-delivered cell for the current
  checkpoint), not reopened as data or animation work.

Known turn under-expression boundary:

- Turn control response is not reopened here. The accepted criterion remains control
  response: changing external `cond_dir/cond_speed` changes realized direction. The
  known boundary is narrower: the current checkpoint under-expresses some turn pose
  amplitude/manifold detail even when teacher-forced, so this is not a free-run carry
  drift issue.
- Read-only mechanism status: under-conditioning is multi-variable. Causal curvature
  explains the slow-turn residual: signed causal yaw-rate splits reduce GT geodesic
  spread in `h00|slow` to ratio `0.5377344030121408` and `h07|slow` to
  `0.34492573213085526`. It does not close fast bins: `h00|fast` stays at
  ratio `0.8705999814871522` with effective `(clip,region)` n `5`, and
  `h07|fast` stays at `0.8880496832718258` with effective n `3`.
  Source:
  `debug_output/_tmp_turn_cond_decomp_20260608_v1/redteam_turn_cond_decomp.json:14`,
  `debug_output/_tmp_turn_cond_decomp_20260608_v1/redteam_turn_cond_decomp.json:53`,
  `debug_output/_tmp_turn_cond_decomp_20260608_v1/redteam_turn_cond_decomp.json:85`,
  `debug_output/_tmp_turn_cond_decomp_20260608_v1/turn_cond_decomp_curvature_refinement_summary.csv:2`,
  `debug_output/_tmp_turn_cond_decomp_20260608_v1/turn_cond_decomp_curvature_refinement_summary.csv:8`,
  `debug_output/_tmp_turn_cond_decomp_20260608_v1/turn_cond_decomp_curvature_refinement_summary.csv:11`,
  `debug_output/_tmp_turn_cond_decomp_20260608_v1/turn_cond_decomp_curvature_refinement_summary.csv:17`.
- Fast/mid residual is phase/progress-like, not curvature-redundant. Clip-local
  progress reduces `h00|fast` to ratio `0.45455239645982226`,
  `h00|mid` to `0.45126085295153817`, and `h07|mid` to
  `0.32512828280347694`; fast-bin base effective n was explicitly checked as
  `h00|fast=5`, `h07|fast=3`, so this is not an effective-n=1 artifact.
  Source:
  `debug_output/_tmp_turn_progress_phase_decomp_20260608_v1/redteam_turn_progress_phase_decomp.json:11`,
  `debug_output/_tmp_turn_progress_phase_decomp_20260608_v1/redteam_turn_progress_phase_decomp.json:12`,
  `debug_output/_tmp_turn_progress_phase_decomp_20260608_v1/redteam_turn_progress_phase_decomp.json:111`,
  `debug_output/_tmp_turn_progress_phase_decomp_20260608_v1/turn_progress_phase_fast_eff_audit.csv:2`,
  `debug_output/_tmp_turn_progress_phase_decomp_20260608_v1/turn_progress_phase_fast_eff_audit.csv:4`,
  `debug_output/_tmp_turn_progress_phase_decomp_20260608_v1/turn_progress_phase_base_refinement_summary.csv:3`,
  `debug_output/_tmp_turn_progress_phase_decomp_20260608_v1/turn_progress_phase_base_refinement_summary.csv:8`,
  `debug_output/_tmp_turn_progress_phase_decomp_20260608_v1/turn_progress_phase_base_refinement_summary.csv:23`.
- Production boundary: `clip_progress = frame/(clip_len-1)` is a diagnosis-only oracle.
  It is an authored clip playback index, not a player command and not a causal
  derivative of the command stream. It must not be added to production direct-pose
  conditioning. GT-pose-derived gait phase is also a read-only oracle/witness because
  it was window-sensitive and is not an online control-equivalent source.
- Current checkpoint status: **not fixed**. The delivered config has
  `direct_pose_feat_source="cond"` and `direct_pose_use_phase_z=false`, so direct pose
  does not receive an explicit clean phase side-channel. Source:
  `debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/lambda/checkpoints/posttrain_log_WalkF_stage7_lambda_from_lowlr72_lr1e4_20260504.json:80`,
  `debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/lambda/checkpoints/posttrain_log_WalkF_stage7_lambda_from_lowlr72_lr1e4_20260504.json:86`.
- B slow-only discriminator status: executed as a non-production experiment, not a
  production fix. `B` adds an internal causal side-channel
  `Δdir_t = cond_dir_t - cond_dir_{t-1}` with tensor shape `(B,T,2)`, `float32`,
  frame0 copied from frame1, and no public/player 7D command interface change. The
  clean control `A` has the same head shape/init/steps/seed/data/lr/freeze/loss and
  feeds zeros on the two extra channels. Both were trained for 120 steps from
  `debug_output/_tmp_b_slow_discriminator_readout_20260611_v1/experiment_init_direct_curvature_sidechannel_dim2_nonproduction.pth`;
  the init report records 125 loaded non-direct keys and 24 fresh direct-pose keys.
  Source:
  `debug_output/_tmp_b_slow_discriminator_readout_20260611_v1/experiment_init_direct_curvature_sidechannel_dim2_report.json:1`,
  `debug_output/_tmp_b_slow_discriminator_readout_20260611_v1/slow_discriminator_cache_manifest.json:1`.
- B readout result: **INCONCLUSIVE-UNDERPOWERED / NULL requires representation
  caveat**, slow scope only. In the primary teacher/R_dir comparison, `B-A` improves
  only 2/6 slow regions on medoid spread, 3/6 on medoid displacement, 2/6 on Karcher
  spread, and 3/6 on Karcher displacement. Median deltas are `+0.0050010631942932204`
  medoid spread abs-error, `-0.0007800604341117179` medoid displacement ratio,
  `+0.002392789909227533` Karcher spread abs-error, and
  `+0.0030488581162899653` Karcher displacement ratio. Since the single-Δdir channel
  does not produce per-region, dual-center, dual-metric agreement and eff-n is only 6
  contiguous slow regions, this does not confirm under-conditioning and does not
  justify an under-fitting claim without a longer causal-window/higher-order retest.
  Source:
  `debug_output/_tmp_b_slow_discriminator_readout_20260611_v1/slow_discriminator_summary.csv:2`,
  `debug_output/_tmp_b_slow_discriminator_readout_20260611_v1/redteam_b_slow_discriminator.json:1`.
- Any future production repair path must treat slow and mid/fast separately. `B`
  currently supplies only a slow-bin discriminator result; it must not be used as
  evidence for mid/fast. A clean phase-like path for fast/mid remains diagnosis-only
  unless it is online, non-authored, and free of contact/TTC/CONTACT_SLICE/current
  `soft_period` sources.
- `soft_period` as currently implemented is not a clean production phase source:
  `InputProjectors.period(inputs)` copies the contact slice into the period input, and
  `soft_period = tanh(period_head(h_period))` inherits that source. Runtime
  `phase_z` is an external input with shape `(B, 2 * contact_dim)` or
  `(B, Tq, 2 * contact_dim)`, dtype/device matched to the rollout tensors; it is clean
  only under a fixed-delta/no-reset path such as `phase_reset_source=none`. Contact
  reset (`contacts_meas`), TTC reset (`ttc_gt`), current frozen-period, or current
  `soft_period` paths are Layer-3-contaminated controls, not production fixes. Source:
  `debug_output/_tmp_phase_z_purity_audit_20260608_v1/redteam_phase_z_purity_audit.md:7`,
  `debug_output/_tmp_phase_z_purity_audit_20260608_v1/redteam_phase_z_purity_audit.md:9`,
  `debug_output/_tmp_phase_z_purity_audit_20260608_v1/redteam_phase_z_purity_audit.md:11`,
  `debug_output/_tmp_phase_z_purity_audit_20260608_v1/redteam_phase_z_purity_audit.md:40`,
  `debug_output/_tmp_phase_z_purity_audit_20260608_v1/redteam_phase_z_purity_audit.md:42`,
  `debug_output/_tmp_phase_z_purity_audit_20260608_v1/redteam_phase_z_purity_audit.md:43`,
  `train/pretrain_mpl_min.py:1556-1565`,
  `train/pretrain_mpl_min.py:2059-2062`,
  `train/validate/run_freerun_cycles.py:10740-10749`.
- Closeout criterion (two-sided falsifiable): if `B + D-clean` repairs the speed-binned
  Layer-1 geodesic spread/displacement pattern without progress, contacts, TTC, or
  current `soft_period`, then the boundary closes as a clean under-conditioning fix.
  If only `C = cond + clip_progress` or a contaminated phase source repairs fast/mid,
  the scope-bound conclusion is negative under this contract: fast turn
  under-expression needs phase-like information, but the current no-memory/no-Layer-3
  production boundary has no accepted clean source. Do not reopen this as a data,
  authored-GT, reverse-NN, or objective-only mean-collapse debate.

## 7. Acceptance Summary

A sample/run is accepted by this kinematics-only contract only if:

1. Gate A accepts the goal as legal: narrow cell, curriculum-band gap, finite command,
   and no support/contact phase-lock requirement.
2. Gate B confirms the command/carry path is internally consistent. This is a pipeline
   regression check with zero capability claim.
3. Gate C confirms the generated rot6d pose sequence is articulated enough and
   on-manifold enough under FK-free Layer-1 metrics, with thresholds calibrated from GT
   continuous bands.

The delivered capability is exactly:

> legal on-manifold walk pose generation under externally commanded kinematics.

The delivered capability is not:

> support-honest, foot-grounded, contact-phase-controllable, arbitrary seam-goal
> inbetweening.
