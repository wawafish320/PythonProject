> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Action Handoff — Goal-Conditioned In-Betweening Direction & Next-Step Entry

Date: 2026-05-29

Status: direction synthesis + next-step spec entry — FOR REVIEW (not a frozen contract).

Purpose: single source of truth for where the action-handoff line landed after the
P0–P6 probe, the attractor-support audit, and the re-entry resolver work. It states the
target architecture, the design decisions with their evidence, the empirical bets a
reviewer should scrutinize, and the entry point for the training spec. It deliberately
separates **validated facts**, **design decisions**, and **unproven bets**.

## 0. Relationship to prior docs (what this supersedes)

- Closeout P0–P6: `2026-05-26_action_handoff_z_probe_closeout_decision_record.md`
- Attractor-support audit (A1/A2/A3): `2026-05-29_action_handoff_z_attractor_support_audit_note.md`
- Re-entry resolver diagnostic: `tools/run_action_handoff_reentry_resolver_diag.py`,
  `debug_output/_tmp_action_handoff_reentry_resolver_diag_20260529/`
- Runtime contract v0.1: `2026-05-29_action_handoff_runtime_handoff_contract.md`
- z v1 design (A/B/C): `2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md`

**This direction supersedes the runtime contract's component [3] (authored bridge):** the
i→j middle is now a **learned goal-conditioned generator**, not an authored crossfade. The
contract's [1] convergence detector and [2] resolver are re-scoped as the **handoff seam**
(see §3). The reason for the change is in §1.

## 1. What was rejected and why

| Rejected approach | Why |
|---|---|
| Pair / cond-transition posttrain (train each source→target) | O(n²) action pairs × frames; does not scale as actions grow (the user's original objection, message 1). |
| Motion-Matching / matched-point switching (wait for a frame that matches, then cut) | Pushes the burden onto **authoring** (every clip must be authored with matchable seams); cannot start a transition at an arbitrary moment; generates no new motion. |
| Authored-only bridge at matched endpoints | Works only because the current 5 clips were authored as Walk→turn→Walk loops with matched endpoints (resolver sharpness 0.008); this property does NOT generalize to arbitrary clips. |
| Using `z` nearest-frame to pick the re-entry/endpoint frame | A3: `z` is not cross-clip phase-comparable (0/4 agreement with pose/contact). |

## 2. Evidence base (validated, frozen-artifact, teacher-forced)

All teacher-forced / offline on the locked 5 clips (single Walk_F cycle). They define
TARGETS and geometry; they do NOT certify freerun behavior.

- A1: contact is encoded in `z` (decodability R² z=0.84 ≥ hidden_pre 0.80 ≫ energy −0.05; kNN purity 0.83 vs chance 0.38).
- A2: target attractors are well-defined as **regions** (diffuseness 0.04–0.29, all < 0.80); offline every source is "off-support" (independent takes; structural, not a defect).
- A3 + resolver: **pose (rot6d) gives a sharp re-entry phase** (all 4 turn clips land at Walk_F cyc ~0.0–0.03, sharpness 0.008–0.011); contact aliases (3/4 wrong); egocentric velocity is phase-flat (lateral ≈ 0) → not a usable tiebreak.
- Endpoint check: every turn clip's START pose is also sharply on a Walk_F phase (pose_d 0.0002–0.01) → these clips are authored Walk-loop splices; the walk→turn onset is **recorded inside the turn clips**.

## 3. Target architecture (the coherent chain)

```
locomotion (freerun)
  → [G] goal-conditioned in-betweening generator : current state + target → drive into attractor
  → [H] handoff seam : converge to a clip-resumable state, hand over
  → [C] target clip takes over : authored animation plays the body of the motion
  → [H'] exit seam : same mechanism in reverse
  → locomotion
```

- **[G] Generator** — learned, generates ONLY the short entry transition (not the full
  target motion). Conditioned on the target (a clip-resumable keyframe / anchor), drives
  the current state into the target manifold from an arbitrary starting moment.
- **[H] Handoff seam** — decides WHEN to hand over (`z`/anchor region = convergence
  timing) and WHERE the clip resumes (pose/contact phase continuity = the resume frame).
  The generator must converge to a **clip-resumable state**, not merely "near the region",
  or the handoff pops.
- **[C] Clip takeover** — authored animation; high quality; not generated. This bounds the
  generator's job to a ~0.2–0.5 s transient.
- **[H'] Exit** — clip → locomotion uses the same seam mechanism in reverse.

## 4. Design decisions (each is a reviewable claim with rationale)

- **D1 — Train a general goal-reaching skill on WITHIN-CLIP gaps, not pairwise
  transitions.** Sample `(context frames, target keyframe)` from inside ANY single clip,
  mask the middle, learn to fill. Cost = O(total frames), NOT O(action pairs). A new
  action = +1 clip to the pool, O(1) marginal. Transitions are NEVER enumerated. (Resolves
  the message-1 scalability objection.)
- **D2 — Condition on the TARGET STATE, not the action label.** The generator learns
  "drive from current state to goal state". walk→turn emerges at inference from a
  (walk-state, turn-anchor) pair never seen in training.
- **D3 — `z` defines the region/convergence; pose/contact defines the resume frame;
  velocity is not used.** `z` = "am I in the attractor region" (A2). pose/contact = "which
  clip frame resumes" (A3/resolver). Never `z` for the frame (A3 0/4). Velocity is
  phase-flat.
- **D4 — Endpoint is not a pre-selected single frame.** Game INTENT selects the regime
  (discrete); `z`/anchor represents it as a region; phase continuity determines the actual
  landing/resume frame. The generator and the clip MEET at the goal keyframe.
- **D5 — The clip takes over the body; the generator only learns short entry
  transitions.** This preserves authored quality AND reduces the generator's data burden
  (short transients are far more learnable on limited data than full motions).
- **D6 — Continue from the current checkpoint with a new objective; do not greenfield.**
  The base model is already a cond(dir/speed)-conditioned autoregressive generator trained
  on non-flipping windows (design §1.1). The new training adds (a) goal/target conditioning
  and (b) masked in-betweening + scheduled sampling, initialized from the current
  basetrain+lambda checkpoint. This IS the posttrain/scheduled-sampling that design §0
  gated behind P0–P6. Drift is partly handled by construction: the goal keyframe is an
  attractor that anchors the rollout.

## 5. Empirical bets (REVIEW SHOULD SCRUTINIZE THESE)

- **B1 — Within-clip-trained goal-reaching generalizes to cross-clip IN-FAMILY (walk→turn).**
  Founded: the turn clips internally contain turning + root-yaw + the walk→turn onset, so
  the component skills are in the data and goal-conditioning composes them. **Risk:**
  generalization QUALITY scales with data coverage (MotionBricks-class results use ~700 h);
  on 5 clips, in-family transitions may be rough. Mitigation: adding locomotion clips is
  O(1) and improves it monotonically without pairwise blowup.
- **B2 — In-family vs cross-family boundary.** Jump is a different dynamics family
  (airborne, no ground contact); it is NOT in any clip's internal gaps, so the generator
  cannot reach a jump attractor from locomotion. Cross-family needs jump data or
  physics-RL. Explicitly OUT of current scope (acknowledged by user).
- **B3 — Drift is controlled by goal-conditioning + scheduled sampling.** Plausible (goal
  anchors the rollout) but unproven on these clips; must be measured on freerun.
- **B4 — Seam has no pop.** Requires the generator to converge to a clip-RESUMABLE state
  (pose+velocity+contact+phase continuity), not just into the rough region. The generator's
  training objective must therefore target a real clip frame's full state.

## 6. Out of scope / not solved

- Cross-family (e.g., → jump) intermediate generation (B2).
- Production thresholds for convergence/handoff (need freerun rollout calibration; runtime
  contract §7 `*-on-rollout` items).
- Variable/parameterized target clips (current assumption: fixed authored clip plays back).
- Multi-cycle phase aliasing (single Walk_F cycle; closeout §13).

## 7. Next-step spec entry (training spec skeleton — to be filled, then implemented)

This is the entry point, not the full spec. Items to specify:

1. **State / target representation.** Observation-space (pose rot6d + root_vel + contact),
   posttrain-invariant. Target keyframe = a clip-resumable frame's state (+ `z`/anchor for
   region/convergence). Decide exact channels and normalization (reuse z-probe future_desc
   group norm).
2. **Within-clip gap sampling (D1).** How to sample `(context, masked middle, target
   keyframe)` per clip; horizon distribution for the masked window (target the short
   transient length, ~0.2–0.5 s); whether to bias sampling toward clip onsets (recorded
   walk→turn).
3. **Model form.** Continue from current checkpoint (D6): autoregressive vs masked-token
   in-betweening; where the target-conditioning enters; how scheduled sampling is scheduled.
4. **Losses.** reach-target (pose/`z`/contact at the goal) + imitation/style (stay on the
   data manifold, no hallucination) + foot/velocity continuity + (seam) converge-to-
   clip-resumable-state (B4).
5. **Off-manifold robustness (B3).** Noise injection / scheduled sampling / DAgger-style to
   cover drifted starting states; this also addresses freerun drift.
6. **Handoff (H).** Convergence detector in `z`/anchor region for timing; pose/contact phase
   continuity for the resume frame; define the pop-free handover test.
7. **Validation.** Reuse `train/validate/run_freerun_cycles.py`: from arbitrary walk states,
   measure (a) does the generator drive into the target turn manifold (reachability), (b)
   does it converge to a clip-resumable state, (c) seam pop magnitude, (d) does the clip
   take over cleanly. Compare drift before/after.

## 8. Open questions for the reviewer

- Is within-clip masked in-betweening + goal-conditioning the right generalization
  mechanism, or is there a lower-risk route to "reach any in-family attractor" (B1)?
- Should the generator be a fine-tune of the base model or a separate goal-conditioned
  module on top? (Drift lives in the base → fine-tune likely needed; confirm.)
- Is the seam best handled by (i) the generator converging exactly onto a clip frame, or
  (ii) a tiny residual blend at handover, or (iii) the clip itself being re-anchored to the
  generator's landing state? (B4.)
- What is the minimal locomotion data needed for B1 to produce acceptable (not just rough)
  walk↔turn transitions?
