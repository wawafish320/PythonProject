> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §1/§5/§6 under its stated read-only / zero-new-injection scope.

# Action Handoff Runtime Contract (v0.1, fallback-aware)

Date: 2026-05-29

Status: runtime CONTRACT (provisional, smoke-calibrated) — NOT a training plan.

Scope: defines the runtime decision pipeline for a turn→Walk_F (and Walk_F→turn)
handoff using the frozen z v1 representation. It binds inputs, outputs, decision
rules, fallback policy, and observability.

Non-scope (hard):
- No model posttrain, no scheduled sampling, no in-basin pipeline change (design §0).
- The bridge does NOT learn the i→j in-between; that is external authoring / C prior
  (design §2; epistemologically not derivable from independent clips).
- Production thresholds are NOT established here; every numeric gate below is
  provisional and must be recalibrated on runtime rollouts (same posture as the P6
  threshold contract note).

Evidence base:
- P0–P6 closeout: `docs/aperiodic_transition/2026-05-26_action_handoff_z_probe_closeout_decision_record.md`
- Attractor-support audit (A1/A2/A3): `docs/aperiodic_transition/2026-05-29_action_handoff_z_attractor_support_audit_note.md`
- Re-entry resolver diagnostic (pose-first): `tools/run_action_handoff_reentry_resolver_diag.py`,
  `debug_output/_tmp_action_handoff_reentry_resolver_diag_20260529/`
- z v1 design (A/B/C, runtime metric §3.4): `docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md`

## 0. Pipeline

```
game intent (A: "transition to target X now")
  -> [1] convergence detector   : is the live rollout inside target attractor?
  -> [2] re-entry resolver       : which Walk_F phase do we re-enter at? (pose-first)
  -> [3] bridge owner            : authored blend exit_i -> entry_j
  -> resume Walk_F locomotion loop
        |
        +-- any guard fails -> [4] fallback policy
```

Role split (must hold):
- `z` is used ONLY for component [1] (attractor membership / convergence). It is NOT
  used to pick the re-entry frame — A3 showed z disagrees with contact on the re-entry
  Walk_F frame 0/4 (mean gap ~1/3 of the gait cycle); the resolver diagnostic confirms
  raw pose L2 (not z) gives the sharp cross-clip phase.
- **Pose (rot6d) drives component [2]; contact refines within the pose neighborhood.**
  Velocity is NOT a re-entry signal (egocentric planar velocity is phase-flat — see §2).
- Authored blend / external C prior owns component [3]; the model is never asked to
  produce the in-between.

## 1. Convergence detector

Decides only: "has the live rollout entered the neighborhood of target attractor T?"

Inputs (per rollout frame t):
- `z_t` (frozen z v1).
- target anchor for T: `anchor_centroid_T`, `anchor_radius_T` from the audit
  (`a2_source_to_anchor_reachability.target_anchors`). Anchors are well-defined
  (diffuseness 0.04–0.29, all < 0.80).
- `future_pred_residual_t`: L2 future-pred residual (design §3.4 records this as an
  independent sanity signal, never mixed into retrieval).
- `contact_t` (future_desc contact channels; A1 confirms contact is encoded in z).

Decision (all must hold):
1. **distance**: `cos_dist(z_t, anchor_centroid_T) <= CONV_DIST` AND monotonically
   shrinking over the last `CONV_WINDOW` frames (trajectory is converging, not grazing).
2. **margin**: `cos_dist(z_t, anchor_centroid_T)` is the minimum over all candidate
   attractors by at least `CONV_MARGIN` (convergence is unambiguous).
3. **residual**: `future_pred_residual_t <= RES_MAX` (model not extrapolating).
4. **contact sanity**: `contact_t` binarized state matches the target end contact state.

CRITICAL calibration note: `CONV_DIST` is NOT the offline anchor radius. The audit's
offline cross-clip `d_min/anchor_radius` is ~22–25 for every pair (independent takes
never sit inside another clip's attractor — design §1.1). `CONV_DIST` must be
calibrated on RUNTIME rollouts as the live trajectory contracts toward the anchor; the
offline `anchor_radius_T` (0.01–0.036) is only a lower-bound reference for how tight a
converged cluster can be.

Provisional values (recalibrate on rollouts):
- `CONV_WINDOW = 6` frames, `CONV_MARGIN = TBD-on-rollout`, `CONV_DIST = TBD-on-rollout`,
  `RES_MAX = TBD-on-rollout` (seed from P1 residual distribution).

Output: `converged: bool`, plus the residual/distance/margin for logging.

## 2. Re-entry resolver

Decides: which Walk_F frame (= gait phase) to resume the loop at. **Pose-first; contact
refines; z and velocity never select the frame.**

This replaces the earlier contact-first + velocity-tiebreak design. The resolver
diagnostic (`tools/run_action_handoff_reentry_resolver_diag.py`,
`debug_output/_tmp_action_handoff_reentry_resolver_diag_20260529/`) showed:
- **Contact aliases**: contact-nearest Walk_F frame lands at the wrong phase for 3/4 turn
  clips (e.g. Walk_R_To_L contact-NN = cyc 0.57, a contact coincidence).
- **Egocentric velocity is phase-flat**: Walk_F egocentric velocity has lateral ≈ 0 and
  forward only oscillating ~0.54–0.93, so it carries almost no gait-phase information — a
  velocity "tiebreak" cannot disambiguate anything.
- **Pose localizes phase sharply**: turn-clip end pose is nearest the Walk_F cycle start
  (cyc ~0.00–0.03) for all 4 clips with pose sharpness 0.008–0.011 (these clips are
  authored to return to the loop pose). This resolves Walk_R_To_L cleanly.

Inputs: `pose_t` (bone_rot6d) at convergence; Walk_F reference pose track; `contact_t`,
Walk_F reference contact track; per-target pose sharpness from the resolver diagnostic.

Decision:
1. **pose neighborhood**: `pose_top = top-k argmin_f || pose_t − pose_F[f] || / sqrt(dim)`.
2. **sharpness gate**: if pose sharpness (top-3 circular cycle spread) `>= POSE_SHARP_MAX`
   → phase neighborhood is ambiguous → fallback.
3. **contact refine**: `reentry_frame = argmin_{f ∈ pose_top} || contact_t − contact_F[f] ||`
   (pose picks the phase region; contact picks the exact frame within it).
4. velocity is NOT used; the world-frame turn residual is reconciled by the bridge
   (root-yaw), not by re-entry frame selection.

Known case: Walk_R_To_L resolves to cyc ~0.00 (pose sharpness 0.008) — **no longer a
forced fallback**. All four turn clips pass the sharpness gate (mean pose sharpness 0.009).

Provisional values: `POSE_TOPK = 5`, `POSE_SHARP_MAX = 0.15`.

Output: `reentry_frame: int | None` (None ⇒ pose-phase ambiguous ⇒ fallback).

## 3. Bridge owner (authored blend / external C prior)

Consumes:
- `exit_i` = source exit frame, constrained to the convergence/end window
  (`a2...pairs[*].exit_frame_source_end_window`, NOT the unconstrained
  `exit_frame_all_frames` — see audit §2.1 caveat).
- `entry_j` = target entry frame (`a2...pairs[*].entry_frame_in_target`) and/or the
  resolver's `reentry_frame` for the Walk_F resume.

Produces: the i→j in-between via external authoring (inbetweening prior / physics+RL /
adversarial D — design §2). The bridge is the ONLY place root-yaw / velocity / foot-lock /
upper-vs-lower-body residual is reconciled. The model is never trained to fill this.

Contract: the bridge MUST accept that no learned signal certifies the i→j naturalness;
its acceptance is owned by the external prior, not by z.

## 4. Fallback policy (triggers → actions → owner)

Triggers (any one arms fallback):

| trigger | signal | source |
|---|---|---|
| known-weak source (static gate) | source ∈ {Walk_L_To_R} for target ∈ {Walk_R_To_L, Walk_R_To_R} | closeout §8 weak rows; A2 weak pairs |
| anchor diffuse | `anchor_diffuseness_T >= 0.80` | A2 (none today; guards future clips) |
| convergence residual high | detector [1] residual/distance/margin fails | component 1 |
| re-entry pose-phase ambiguous | resolver [2] pose sharpness `>= POSE_SHARP_MAX` | component 2 (resolver diag: all 4 clips currently pass) |

Note on "source-off-support": offline, EVERY pair is off-support (independent takes), so
it is not a discriminating runtime trigger. It is encoded instead as the **static
known-weak source gate** above (arms fallback on entry), consistent with the closeout
classifying weak rows as `weak_fallback_required_known_risk`.

Actions (design §3.4 fallback set), in escalation order:
1. **extend window**: keep driving toward the attractor for up to `FB_EXTEND` more frames,
   re-check detector.
2. **cut to intermediate clip**: route through a known-safe intermediate rather than
   direct handoff.
3. **refuse**: abort the transition, hold canonical Walk_F.

Owner: the fallback action is a runtime policy decision and MUST be explicit (not a logged
warning) — closeout §10 blocker.

## 5. Observability (required, not optional)

Closeout §10 forbids "no-monitoring deployment". Every handoff attempt logs:
- detector distance / margin / `future_pred_residual` (P1 magnitude risk tracking).
- resolver chosen frame, pose sharpness, pose neighborhood, contact-refine distance.
- fallback trigger fired (if any) and action taken.
- bridge endpoints actually used (`exit_i`, `entry_j`).

P1 magnitude risk (z_bottleneck future_desc point-regression weaker than energy/raw) is
tracked here as a live residual gate, per closeout §10 — it does not block H3 but blocks
unmonitored deployment.

## 6. What this contract is NOT

- Not a posttrain coverage plan. No pair enumeration, no clip-domain training objective.
- Not a production threshold contract. All gates are provisional smoke values; `*-on-rollout`
  fields must be calibrated against runtime rollouts before any production claim.
- Not a naturalness guarantee. Bridge naturalness is owned by the external C prior.

## 7. Open calibration items (before v0.2)

- `CONV_DIST`, `CONV_MARGIN`, `RES_MAX`, `FB_EXTEND` — all from runtime rollouts.
- ~~Whether the velocity/direction tiebreak resolves Walk_R_To_L re-entry~~ — RESOLVED:
  the resolver is pose-first (§2); pose resolves Walk_R_To_L to cyc ~0.00 sharply, velocity
  is dropped (phase-flat). No forced fallback for Walk_R_To_L.
- Validate the pose-first resolver on a clip that does NOT return to the Walk_F loop pose
  (current sharpness depends on these 5 clips terminating near the loop pose).
- Multi-cycle Walk_F reference to test phase aliasing the single-cycle data could not
  (closeout §13).
