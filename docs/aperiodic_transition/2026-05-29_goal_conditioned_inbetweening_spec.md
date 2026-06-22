> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §1/§5/§6 under its stated read-only / zero-new-injection scope.

# Goal-Conditioned In-Betweening — Training Spec v0.1 (FOR REVIEW)

Date: 2026-05-29

Status: minimal training spec, pre-implementation — FOR REVIEW. Every numeric value is
provisional. Decisions marked **[FIXED]** / **[DECISION]** / **[PROVISIONAL]**.

Changelog: v0.1 + Codex review refinements (2026-05-29): yaw_rate units/wrap/norm fixed
(§1.1); grounded cross-manifold alignment made concrete via the §7.1 data check —
full-state φ, real onset GT middle, per-clip groundability gate, Walk_L_To_R fallback (§2b);
B1 gate split into reach/clip_resumable/pop_safe/fallback rates (§6); §7.1 done.

Parent direction: `2026-05-29_action_handoff_goal_conditioned_inbetweening_direction.md`
(architecture, bets B1–B4). This spec fills §7 of that doc and incorporates the Codex
review (biased sampling; full seam state) plus refinements (grounded cross-manifold
sampling; egocentric + short-window seam state; yaw_rate channel; B1-first gate; keep
fallback + style loss).

## 0. Scope

In: a goal-conditioned generator that produces the **short entry transient** (~0.2–0.5 s)
driving an arbitrary in-family (locomotion) state into a target clip's resumable seam,
after which the **authored clip takes over**. Trained on the locked clips, initialized from
the current basetrain+lambda checkpoint, validated via `run_freerun_cycles`.

Out: cross-family (jump) transitions (bet B2, needs new data/physics); full target-motion
generation (clip plays the body); pairwise/per-transition training; production thresholds.

Data facts (verified): FPS=60; pose `bone_rot6d` [T,46,6]; `root_vel` [T,2] world-planar;
`contact` [T,2] (ch0=right, ch1=left); `cond_in` [T,7] = act_oh(4)+cond_dir(2)+cond_speed(1),
`cond_dir`=world heading (cos,sin). Clips: Walk_F(87), Walk_L_To_L(54), Walk_L_To_R(50),
Walk_R_To_L(86), Walk_R_To_R(93). z probe = 32-d on frozen hidden_pre(512).

## 1. State & target schema  [FIXED unless noted]

### 1.1 Per-frame egocentric state `s_t` (heading-invariant), D_s = 281
- `pose_rot6d` [276] — joint-local rotations; already body-local / heading-invariant.
- `root_vel_ego` [2] — (forward, lateral): world `root_vel` rotated by −heading (heading
  from `cond_dir`). `ego_fwd = v·d̂`, `ego_lat = v·d̂⊥`.
- `yaw_rate` [1] — **[FIXED, with rationale + units]** heading angular rate.
  Egocentric velocity is phase-flat (audit verified: Walk_F lateral≈0.000, forward
  0.54–0.93), so it does NOT distinguish "turning" from "straight walk" — the turn signal
  lives in heading rotation. Without `yaw_rate` the turn manifold is not separable in state
  space (verified: Walk_F yaw_rate min/med/max = −0.000/0.000/0.000; turn onsets ramp to
  ~1.2 rad/s).
  Definition (FIXED): `heading_t = atan2(cond_dir_y, cond_dir_x)`;
  `yaw_rate_t = wrap_to_[-π,π](heading_t − heading_{t−1}) × FPS` → **rad/s**;
  frame 0 := frame 1's value; shape [T,1], dtype float32, CPU export.
  Loss/scale (FIXED): standardize `yaw_rate` by the pooled std over all locked clips (same
  group-norm posture as `future_desc`) so it is not drowned by the 276-d pose term.
- `contact` [2] — soft contact (right, left).

World heading/position are deliberately EXCLUDED from `s_t`: they are reconciled by a rigid
yaw transform at handoff (§5), not matched by the generator.

### 1.2 Target / goal `g` = a short SEAM WINDOW, not a single frame  [FIXED]
K-frame window of the target clip's resumable state, same egocentric channels:
- `g.state` [K, 281] (pose, root_vel_ego, yaw_rate, contact)
- `g.z_anchor` [32] — the target regime's `z` anchor (region/convergence only; NOT used to
  pick a frame; see direction D3).
K **[PROVISIONAL]** = 6 (≈100 ms). Rationale: a single keyframe gives C0 (pose) match but
allows C1 (velocity / contact-phase-direction) discontinuity → handoff pop. Matching a
short window enforces velocity + phase-direction continuity (bet B4).

### 1.3 Context  [PROVISIONAL]
`ctx` = last C=16 frames of `s_t` (match base model `context_len`).

## 2. Sampling  [PROVISIONAL ratios; design FIXED]

Combats Codex challenge 1 (avoid degenerating into local interpolation) AND bet B3
(off-manifold) with one combined design. Each minibatch mixes three types:

- **(a) within-clip gap (self-supervised), ~50%.** From any single clip: `ctx`=[i−C, i],
  `g`=[j, j+K], generate the masked span (i, j). **Biased sampling** (not uniform):
  oversample segments with high `|yaw_rate|` (turn onset/curvature), contact transitions,
  and clip onset/end. **Longer-gap curriculum**: gap (j−i) starts ~12 frames, grows to
  ~30 over training.
- **(b) grounded cross-manifold, ~35%.** **[the B1 strengthener — alignment now FIXED via
  §7.1 data check]** Construction:
  - `ctx` = Walk_F (hub) frames `[φ−C, φ]`.
  - **GT middle** = the target turn clip's recorded onset frames `[0, H]` (this IS the real
    walk→turn transition motion — turn clips begin in the walk manifold and ramp `yaw_rate`
    from ~0; verified onset ramps e.g. R_To_L 0→−0.74→−1.15 rad/s). This gives a real middle
    GT, NOT reach-loss-only.
  - **seam target** `g` = turn frames `[H, H+K]`.
  - **Alignment φ = FULL-STATE nearest Walk_F frame to turn[0]** (pose + contact + ego_vel +
    yaw_rate), NOT pose-only. Pose-only is insufficient (verified: pose-only φ leaves a
    contact gap of 0.74/0.96 for L_R / R_L).
  - Cost O(clips), NOT O(pairs); new action adds (walk-hub → its onset), O(1); turn→turn
    composed through the hub. Converts B1 from pure generalization to partially-supervised
    reaching.
  - **Groundability gate (FIXED, per-clip)**: a clip is a clean grounded source only if its
    onset has a Walk_F full-state match below threshold. §7.1 result:
    - Walk_R_To_L: groundable — full-state φ = f2 (cyc 0.02), pose_d 0.011, contact_d 0.162
      (pose-only f0 had contact_d 0.96 → must use full-state φ).
    - Walk_R_To_R, Walk_L_To_L: groundable (low onset contact gap 0.03 / 0.11).
    - **Walk_L_To_R: FAILS the gate** — no Walk_F frame matches onset in pose AND contact
      (pose-top10 min contact_d 0.70); its onset foot-state is not on the walk cycle.
      Fallback: align to a LATER onset frame (after contact settles to walk-compatible) or
      drop from pure-grounded and rely on §2a within-clip + §2c augmentation (accept B1
      generalization there). Flag as a possible onset authoring inconsistency.
- **(c) start-state augmentation, ~15% (overlaps a/b).** Perturb `ctx` last-state with
  noise / use phase-shifted or mildly off-manifold starts, so the model learns to reach `g`
  from arbitrary/drifted starts (control), not reproduce one recorded transition. This is
  the B3 (drift) fix and is coupled to sampling, not a separate stage.

## 3. Model  [DECISION — default + alternative]

Default: continue the current autoregressive base (init from
basetrain+lambda checkpoint), add **goal conditioning** (encode `g` → conditioning tokens
via cross-attention / FiLM). Output = autoregressive next-frame `s_t` over the transition
horizon until convergence. Scheduled sampling on (couples with §2c).

Alternative: masked-token in-betweening (MotionBricks-style) — fill (i, j) given `ctx`+`g`.

Open for review (§8): AR-with-goal vs masked; fine-tune base weights vs separate module
(drift lives in the base → fine-tune likely required).

## 4. Losses  [PROVISIONAL weights]

- `L_reach` (w=1.0) — match `g` over the K-window at horizon end: weighted L on pose,
  root_vel_ego, yaw_rate, contact. Window (not single frame) ⇒ enforces C1.
- `L_imitation` (w=0.5) — **load-bearing for B1 quality (keep, per review).** Stay on the
  data manifold: feature-matching to recorded segments and/or a small adversarial
  discriminator (AMP-style) on the motion distribution. Prevents floaty/hallucinated
  transients, especially under low data.
- `L_foot` (w=0.3) — foot-slide / contact consistency; reuse
  `run_freerun_cycles` foot-slip + contact-mismatch metrics.
- `L_seam_C1` (w=0.5) — at handoff, penalize velocity + `yaw_rate` + contact-phase-direction
  discontinuity between the generator's last frames and `g`'s first frames.
- `L_smooth` (w=0.1) — acceleration/jerk regularizer.

## 5. Handoff (H)  [FIXED design]

- **Timing**: convergence in `z`/anchor region (`cos_dist(z_t, g.z_anchor) ≤ CONV_DIST`,
  shrinking over CONV_WINDOW) → hand over. Thresholds `*-on-rollout` (PROVISIONAL).
- **Resume frame**: pose/contact phase continuity selects the clip frame to resume; `z`
  never selects the frame (direction D3).
- **World heading**: apply a **rigid yaw transform** so the clip resumes in the generator's
  current world heading. Generator never matches absolute heading (§1.1).
- **Fallback (keep, per review)**: if convergence residual stays high or seam pop exceeds
  threshold → runtime-contract fallback (extend window / cut to intermediate / refuse).

## 6. Eval — B1-FIRST gate  [FIXED ordering; PROVISIONAL thresholds]

The first thing built must answer the make-or-break bet B1. Do NOT polish infra before it.

**The gate reports FOUR separate rates (do not collapse — reach must not mask seam
failure):** from N≥20 arbitrary Walk_F start phases, condition on a turn anchor →
- `reach_rate` — fraction where `z` enters the anchor region (`cos_dist ≤ CONV_DIST`).
- `clip_resumable_rate` — fraction where pose ALSO converges to a clip-resumable frame
  (`pose_d ≤ τ_pose`).
- `pop_safe_rate` — fraction where the handoff discontinuity (velocity / `yaw_rate` /
  contact) is below `τ_pop` (this is the B4 seam result; reach can pass while this fails).
- `fallback_rate` — fraction routed to fallback (convergence/seam failed).

**First-round B1 probe**: gate on `reach_rate ≥ 0.7` **[PROVISIONAL]** as the make-or-break
signal, but REPORT all four; a high `reach_rate` with low `pop_safe_rate` means B1 looks ok
but B4 (seam) is unsolved — surface it, do not hide it. **If `reach_rate` fails → STOP and
reconsider; do not expand.**
- **drift metric (B3)**: freerun stability over a fixed horizon, before vs after.
- All thresholds (`CONV_DIST`, `τ_pose`, `τ_pop`, the 0.7) are smoke/probe only — NO
  production threshold (per review); set after the first probe.

Validation harness: extend `train/validate/run_freerun_cycles.py` to (i) start from an
arbitrary phase, (ii) inject a target-anchor goal, (iii) log z-to-anchor distance, pose
convergence, and handoff pop.

## 7. Implementation staging  [FIXED order]

1. **Data check** (cheap) — **DONE (2026-05-29)**: grounded pairs confirmed for 3/4 turn
   clips via full-state alignment (results folded into §2b). Key outcomes: yaw_rate units
   verified (Walk_F≈0, turn onsets ramp to ~1.2 rad/s); full-state φ required (pose-only
   leaves contact gaps); Walk_L_To_R fails the groundability gate (onset foot-state off the
   walk cycle) → fallback path. Reproduce with the one-off check or fold into a tool in
   §7.2.
2. **Sampler + target-conditioning**: implement §2 + §1 schema; unit tests on shapes,
   egocentric transform correctness (Walk_F ego_lat≈0; turn yaw_rate≠0), grounded pairs.
3. **Minimal model + losses**, init from checkpoint; run the **B1 probe (the gate)**.
4. **Only if B1 shows signal**: add curriculum/augmentation breadth, measure pop + drift,
   iterate. No large training run before the gate.

## 8. Open decisions for the reviewer

- Model form: AR-with-goal-conditioning vs masked in-betweening.
- Fine-tune base weights vs separate goal-conditioned module (drift → likely fine-tune).
- `L_imitation`: adversarial (AMP-style) vs feature-matching, given limited data.
- Hyperparameters: K (seam window=6?), C (context=16?), gap curriculum schedule, sample mix.
- Is `yaw_rate` sufficient to separate turn manifolds, or is root angular trajectory over a
  window needed?
- Seam: generator converges exactly onto a clip frame, vs tiny residual blend, vs re-anchor
  the clip to the generator's landing (bet B4).
- B1 thresholds (reachability ≥0.7? τ_pose, τ_pop) — set after the first probe.

## 9. Traceability to bets

- B1 (in-family generalization): addressed by §2b grounded cross-manifold + §2c
  augmentation; gated by §6 B1-first. Quality remains data-bound.
- B2 (cross-family / jump): OUT of scope (§0).
- B3 (drift): §2c augmentation + §3 scheduled sampling + goal-anchoring by construction.
- B4 (seam pop): §1.2 short-window target + §4 L_seam_C1 + §5 handoff design + §6 pop metric.
