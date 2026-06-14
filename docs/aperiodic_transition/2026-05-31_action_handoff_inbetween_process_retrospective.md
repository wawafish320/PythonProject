> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Action-Handoff Goal-Conditioned In-Betweening — Full-Process Retrospective

Date: 2026-05-31

Status: **STANDALONE COLD-START RETROSPECTIVE — auditable, default-skeptical.** Read-only:
no new experiment, no training, no implementation change. The only artifact produced is this
document. Key numbers were **re-derived from the artifacts** (not copied from the in-loop
self-assessments); where I recomputed a number by hand it is marked **[recomputed]**.

Three-state labels used throughout:
- **[ESTABLISHED]** — holds on the artifacts, survives an independent read.
- **[WITHDRAWN/CORRECTED]** — was once concluded, then refuted by a later probe (the refuter is named).
- **[OPEN]** — neither confirmed nor refuted; genuinely unanswered.

Scope note on naming: the line uses two parallel naming schemes that refer to the same chain.
The **spec staging** scheme (§7.1 → §7.2 → 3a → 3b Slice 1 → §4b → §4c → §4d → PHASE 1 →
PHASE 2) is the AR-with-goal / latent-injection arm. The **W-scheme** (W0/W1a honesty →
W1b direction → W1c fork → W1d LOGO → soft-endpoint reframe) is the post-mortem + masked-arm
arm that runs *after* PHASE 2 and ends the line. F4/F5 are the commanded-yaw control +
discriminator probes that sit alongside the masked arm. This retrospective threads both.

Sources read (design + all records + artifacts; artifact paths are under `debug_output/`):
`2026-05-29_goal_conditioned_inbetweening_spec.md`,
`2026-05-29_action_handoff_goal_conditioned_inbetweening_direction.md`,
`2026-05-29_action_handoff_z_attractor_support_audit_note.md`,
`2026-05-30_action_handoff_inbetween_72_review_record.md`,
`..._72_coverage_note.md`, `..._73_b1_probe_plan.md`, `..._73b_path_ab_plan.md`,
`..._72_73_review_record.md`, `..._reach_metric_honesty.md` (W0/W1a),
`..._direction_decision_w1b.md` (W1b), `..._w1c_fork.md` (W1c/W1d),
`..._independent_review.md`, `2026-05-31_..._soft_endpoint_reframe.md`.

---

## 1. One-page conclusion (TL;DR)

**Where the line is now: PARKED.** The terminal decision is W1d LOGO **PARK**, re-confirmed by
the 2026-05-31 soft-endpoint zero-training probe (**KEEP PARK**). No motion-level walk→turn
transition has been demonstrated for the one generalization target that matters
(`Walk_L_To_R`), under either the AR-with-goal arm or the masked arm.

- **B1 (the make-or-break: does within-clip goal-reaching generalize cross-clip in-family?)** —
  **[OPEN], and now data-bound.** It was never honestly answered "yes". The one apparent
  positive (PHASE 2 `Walk_L_To_R reach_rate 0.75`) is a **radius-normalization instrument
  artifact** (§3, §4 below), withdrawn by W0/W1a. The masked arm's W1d LOGO then showed
  **memorization, not generalization**: every held-out grounded clip fails the action-only gate.
- **F4 (commanded-yaw drive / control) — [ESTABLISHED, with a caveat].** Yaw command is wired
  and read on both the AR/free-rollout path (MinimalGoalAR: `yaw_overwrite_max_abs=0.0`,
  ordered body sensitivity, `plain_rollout ≠ command`) and the masked path (cmd_yaw is a
  first-class `[B,H,1]` input). F4 is "solved" only in the narrow sense of *control plumbing*;
  it does not make the turn happen. New runtime entry points still need re-check.
- **F5 (discriminator: is masked/AR + continuity-prior the lever?) — [the only remaining wall
  is data+architecture].** The capacity-matched clean rerun's primary verdict is
  **`INSTRUMENT_INVALID_PLATEAU`** (the instrument cannot license a clean masked-vs-AR verdict);
  AR shows free-rollout drift on `mirror_r2l`. The two residual entangled causes are **data
  insufficiency vs. one-shot generation architecture being too weak**.
- **Is the data bottleneck LICENSED? NO — [not licensed].** "B1 quality is data-bound" is a
  *plausible, well-motivated hypothesis* (L_R has zero grounded supervision; W1d shows even
  grounded clips fail held-out), but it is **not** an established result: the F5 clean rerun
  explicitly reports `data_or_formulation_license_granted=false`; the masked arm cannot read the
  `hidden_pre` reach metric (space-incompatible); and the LOGO result shows removing a
  *grounded* clip's own supervision also fails — so "just add L_R data" may not unlock it.
- **Decision point: PARK vs un-park.** Current call = **PARK**. The falsifiable un-park trigger
  is fixed and written (§8): build a real `Walk_L_To_R` grounded onset (`contact_d ≤ 0.30 ∧
  pose_d ≤ 0.05`) **plus** locomotion bridge clips, then re-run the **same** W1d LOGO /
  soft-endpoint gate at **unchanged thresholds**; un-park only if the held-out grounded clips
  pass the action-only gate.

---

## 2. The original design

### 2.1 Architecture chain (direction §3)

```
locomotion (freerun)
  → [G] goal-conditioned in-betweening generator : current state + target → drive into attractor
  → [H] handoff seam : converge to a clip-resumable state, hand over
  → [C] target clip takes over : authored animation plays the body
  → [H'] exit seam : same mechanism in reverse
  → locomotion
```
[G] generates ONLY the short entry transient (~0.2–0.5 s); [C] (authored clip) plays the body.
This supersedes the earlier "authored crossfade bridge" (runtime-contract component [3]).

### 2.2 Bets B1–B4 (direction §5)

- **B1** — within-clip-trained goal-reaching generalizes to cross-clip IN-FAMILY (walk→turn).
  *Make-or-break.* Risk: quality scales with data; on 5 clips it may be rough.
- **B2** — in-family vs cross-family boundary; jump is OUT of scope (different dynamics family).
- **B3** — drift controlled by goal-conditioning + scheduled sampling (plausible, unproven).
- **B4** — seam has no pop; requires converging to a clip-RESUMABLE full state, not just the region.

### 2.3 Decisions D1–D6 (direction §4)

- **D1** — train on WITHIN-CLIP gaps, not pairwise transitions (cost O(total frames), not O(pairs)).
- **D2** — condition on the TARGET STATE, not the action label.
- **D3** — `z` = region/convergence; pose/contact = resume frame; velocity not used (A3: z 0/4).
- **D4** — endpoint is not a pre-selected single frame; intent picks the regime region; phase
  continuity picks the landing frame.
- **D5** — the clip takes over the body; the generator only learns short entry transitions.
- **D6** — continue from the basetrain+lambda checkpoint with a new objective; do not greenfield.

### 2.4 §1 state schema + yaw_rate (spec §1.1)

Per-frame **egocentric state `s_t`, D_s = 281** = `pose_rot6d[276]` + `root_vel_ego[2]`
(forward, lateral) + `yaw_rate[1]` + `contact[2]`. World heading/position deliberately EXCLUDED
(reconciled by a rigid yaw transform at handoff). `yaw_rate` is **[FIXED, with rationale]**:
`heading_t = atan2(cond_dir_y, cond_dir_x)`, `yaw_rate_t = wrap_[-π,π](Δheading)×FPS` (rad/s),
frame0:=frame1, standardized by pooled std. Rationale (verified, §3.1): egocentric velocity is
phase-flat (Walk_F lateral ≈ 0, forward 0.54–0.93), so the turn signal lives only in heading
rotation; without `yaw_rate` the turn manifold is not separable. Goal `g` = a K=6-frame seam
window (+ `z_anchor` for region only). Context C=16.

### 2.5 §6 acceptance gate (spec §6)

B1-FIRST gate, from N≥20 arbitrary Walk_F start phases conditioned on a turn anchor, reports
**FOUR separate rates** (never collapsed): `reach_rate` (z/anchor region entry),
`clip_resumable_rate` (pose converges, `pose_d ≤ τ_pose`), `pop_safe_rate` (handoff
discontinuity ≤ `τ_pop`; the B4 result), `fallback_rate`. First-round gate: `reach_rate ≥ 0.7`
[PROVISIONAL]; **STOP if reach fails — do not expand**. All thresholds smoke-only.

---

## 3. Timeline experiment ledger (tested / found / withdrawn-or-corrected)

Notation: each step lists **测了什么 / 发现什么 / 后来是否被撤回或修正**.

### 3.0 Evidence base A1/A2/A3 (frozen, teacher-forced; `z_attractor_support_audit_note`) — [ESTABLISHED as geometry; they do NOT certify freerun]
- **A1** contact IS encoded in `z`: decodability R² z **0.84** ≥ hidden_pre **0.80** ≫ energy
  **−0.05**; kNN purity **0.83** vs chance 0.38. (Used to justify Path A+B reach in `hidden_pre`.)
- **A2** target attractors are well-defined **regions** (diffuseness 0.035–0.289, all < 0.80);
  every source is off-support OFFLINE (~22–25× radius) — structural (independent takes), not a defect.
- **A3** pose gives a sharp re-entry phase (sharpness 0.008–0.011; R_L aliased 0.276);
  **z agrees 0/4** cross-clip; ego-velocity phase-flat. → D3.

### 3.1 §7.1 grounded alignment check — [ESTABLISHED]
- Tested: per-clip onset alignment of each turn clip to a Walk_F frame (full-state φ).
- Found: groundable = {L_L, R_L, R_R}; **`Walk_L_To_R` FAILS** the groundability gate
  (`contact_d 0.703 ≫ 0.30`). Full-state φ = **pose-localize + contact-refine** (a genuine
  standardized 281-d L2 is pose-dominated and collapses to pose-only — e.g. R_L pose-only f0
  `contact_d 0.96` vs full-state f2 `contact_d 0.162`). Walk_F sanity: yaw_rate min/med/max
  = −0/0/+0, ego_lat |max| ≈ 1.5e-15 → **yaw_rate channel is load-bearing**.
- Withdrawn? No. This is the **single most solid empirical result in the line** and it correctly
  concentrated all B1 risk on `Walk_L_To_R`.

### 3.2 §7.2 sampler + schema — [ESTABLISHED as a pipeline; later found largely unused by the actual generator]
- Tested: 3-type sampler (within-clip biased+curriculum 0.50 / grounded cross-manifold 0.35 /
  start-state augment 0.15), egocentric transform, groundability gate; 14 unit tests pass.
- Found: ratios reproduce (≈0.51/0.35/0.14 → 0.50/0.35/0.15); **L_R = 100% within-clip
  fallback (`grounded_ok_rate 0.0`)** = zero grounded supervision; gap 12→30 clamps to 28 for
  L_R (50-frame clip); biased-interest lift washes out at long gaps (1.32→1.02).
- Corrected later: the egocentric 281-d tensor pipeline is **not** what the base generator
  trains on (representation gap, §3.5b/independent-review): only the sampling *indices* +
  groundability gate survive downstream.

### 3.3 §7.3 3a wiring smoke (from-scratch, NON-BINDING) — [ESTABLISHED as harness]
- Tested: MinimalGoalAR residual model + losses + free-rollout, no checkpoint.
- Found: loss 1.46→0.22; **caught a real harness bug** (clip_resumable matched the whole clip
  incl. the walk-like onset → trivially 1.00); fixed → discriminating (L_R clip_resumable
  0.21/fallback 0.79). Confirms per-clip reporting is mandatory.

### 3.4 §7.3 3b Slice 1 reach metric in `hidden_pre(512)` (NON-BINDING) — [ESTABLISHED]
- Tested: anchors + offline separation in hidden_pre (z-head was never persisted; Path A+B).
- Found: all 4 anchors well-defined (diffuseness 0.08–0.22); CONV_DIST = 1.5×radius;
  radius_cos = L_L **0.00332**, L_R **0.01169**, R_L **0.00498**, R_R **0.00756**; Walk_F
  recorded frames 5–11× radius away → reach only meaningful on GENERATED rollouts.
- These radii are the load-bearing input to the radius artifact (§4).

### 3.5 §4b cond-driven baseline floor (NON-BINDING) — [WITHDRAWN/CORRECTED]
- Tested: base ckpt free-run from N=20 Walk_F phases with target turn cond, reach via Slice-1.
- Found (initial): `reach_floor_rate 0.00` all four; `pop_safe 0.00`. Honest sub-findings:
  `act_oh` identical `[0,1,0,0]` across all 5 clips (action one-hot is a no-op); `cond_in`
  per-window normalized (constant override collapses to ≈Walk_F); pose heading-invariant →
  `clip_resumable`/`fallback` **degenerate** (1.00/0.00, carry no info).
- **Withdrawn by §4c STEP 0:** the §4b 0.00 floor used the **per-step** `hidden_pre` capture,
  which STEP 0 proved is **space-misaligned** (a clip's own per-step capture does NOT self-reach:
  6.7–20.5× radius, vs full-seq self-reach 0.11–0.31). → *"曾下结论 X = base cond has no floor"
  → 被 Y = per-step capture artifact 推翻*. The 0.00 is partly an instrument artifact, not a clean
  negative. (This is 假阳性 #2: a per-step capture confound.)

### 3.6 §4c goal-injection BINDING probe — [ESTABLISHED negative, but NARROW scope]
- Tested: GoalHead + minimal **output-space** L_reach (pose+ego_vel), base frozen, full-seq reach gate.
- Found: `reach_rate 0.00` all four; trained min_norm **42–283×** radius — *WORSE* than the
  no-goal baseline (4.9–11×); STOP=True. The additive delta dominates and points AWAY from the anchor.
- Scope (correctly narrowed by the in-loop record): the turn-discriminating signal (yaw/heading)
  is NOT in the base OUTPUT space; output-L_reach has **no term rewarding hidden_pre reach** →
  the negative is structural-by-construction, LOW evidence about B1. The §4c lever was *mis-wired*;
  §4d re-wires it.

### 3.7 §4d reach-aware rewire (3 levers, frozen base, CONVERGED) — [ESTABLISHED: reach DIRECTION moves, gate NOT crossed]
- Tested: LEVER 1 per-step context-window calibration (now self-reaches 0.106/0.266/0.238/0.307);
  LEVER 2 **hidden_pre-space** reach loss (not detached); LEVER 3 pre-temporal injection at
  `shared_encoder[1]`.
- Found: genuine plateau (train hidden `|rel|=0.0030`, eval min_norm `|rel|=0.0178`, both ≤0.02;
  flattened while LR was still 5e-4). Goal head roughly **halves** min_norm vs no-goal (L_L
  6.73 vs 13.82; **L_R 2.26 vs 6.77**; R_L 5.13 vs 16.65; R_R 3.74 vs 7.37) — but `reach_rate
  0.00` for all (gate min_norm ≤ 1.5); `pop_safe 0.00`. L_R floor 1.81 vs 1.5.
- Withdrawn? Not withdrawn, but explicitly **not sufficient** to claim a frozen-base ceiling on
  its own (single config `small_add_s1`) → gated to PHASE 1.

### 3.8 PHASE 1 head/injection ablation (6 configs, frozen base) — [ESTABLISHED ceiling; mild goalpost in framing]
- Tested: head capacity (h=256/512/1024, depth 1/2/3), additive vs FiLM, hook s0/s1/multi.
- Found: **no config crosses 1.5**; best `mid_add_multi_s0_s1` L_R min_norm **1.531**, still
  ~5.8× the L_R self-reach (0.266); reach_rate 0.00 everywhere.
- Correction noted by independent review (§2.5): the ceiling criterion was **relaxed** from the
  §4d dual-gate (train-loss AND eval-min_norm plateau) to **eval-min_norm only**. This is a real
  goalpost move, but **empirically backstopped** (the one non-plateaued config was extended to
  1800 steps and reach still did not move: L_R stuck 1.81–1.83). Conclusion stands; framing was
  rewritten as a general rule rather than "we extended it and it didn't move".

### 3.9 PHASE 2 guarded base fine-tune + "real §6 AR gate" — [WITHDRAWN: the one positive is an artifact]
- Tested: tail unfreeze (6 tensors, 515,072 params; `shared_encoder.4/5` + `residual_proj`),
  L2-to-init 3.5e-6; goal head + hidden_pre reach loss; Walk_F drift guard.
- Found (headline): **`Walk_L_To_R reach_rate 0.75`** (min_norm mean 1.43 / min 1.33), others
  0.00; `pop_safe 0.00` all four; Walk_F drift guard passed (pose +0.2%, pop −0.8%, root_speed
  +5.5%). Decision recorded as "partial_success_l_r…". eval min_norm was still improving (0.49),
  i.e. not converged.
- **Withdrawn by W0/W1a + independent review** as a **radius-normalization artifact** (§4 below)
  AND a self-written latent proxy. This is 假阳性 #1 (the central one). The honest outward
  statement is: *frozen base is a confirmed ceiling; one guarded fine-tune moved a latent proxy
  toward all four anchors without collapsing Walk_F, but produced no motion-level walk→turn
  transition on any target.*

### 3.10 W0/W1a reach-metric honesty — [ESTABLISHED: the §4c/PHASE-2 L_R "positive" is WITHDRAWN]
- Tested: (A) absolute self-reach gate (`generated_abs_cos ≤ k·self_reach_abs_cos`, k∈{2,3,5});
  (B) pinned-vs-free no-goal base; (C) realized-yaw/heading; + exact PHASE-2 trained replay.
- Found: under the absolute gate **L_R fails k=2/3/5** (k=5 margin 1.003 — fails even the loose
  gate). Exact trained free rollout (target cond + self-carried contacts + trained goal): L_R
  free `k=3` self-reach **0.00** (pinned 1.00), realized yaw corr **−0.48**, heading MAE
  **39.6°**, `pop_safe 0.00`. State round-trip exact (`max_abs_delta=0.0`). → "L_R 有正信号" is
  **WITHDRAWN**; B4/seam remains blocked.

### 3.11 W1b direction decision — [ESTABLISHED routing]
- Found: the migrated gate is **discriminative** (rejects the artifact; accepts real recorded
  turn: yaw_corr 1.0, heading MAE ≈0, pop_safe 1.0). Recommend **A (add grounded L_R data +
  bridge clips)**; **B (masked in-betweening)** as fallback. Do NOT run a 4th latent-lever round.

### 3.12 W1c fork + masked smoke — [ESTABLISHED: A blocked, B selected, B is anti-B1]
- Found: STEP-0 triage → A blocked by data authoring on L_R (`onset0 contact_d 0.703`, sampler
  `grounded_ok_rate 0.0`); selected B. Masked smoke: L_R `yaw_corr −0.78`, `pop_safe 0`, pose
  degraded → grounded clips reconstruct, **ungrounded L_R fails** = anti-B1 evidence
  (`memorization_suspected=true`). reach component is **non-binding for masked** (281-d state
  vs hidden_pre(512) space-incompatible).

### 3.13 W1d LOGO — [ESTABLISHED terminal: memorization, not generalization → PARK]
- Tested: leave-one-grounded-clip-out (MIRROR-L_R and FULL-HOLDOUT) under the **action-only**
  gate (`yaw_corr>0 ∧ heading_MAE_rad<0.25 ∧ pop_safe>0 ∧ pose not degraded`).
- Found: full-sup L_L/R_L/R_R **pass**, L_R fail. **Every held-out grounded clip FAILS** under
  both MIRROR-L_R and FULL-HOLDOUT (e.g. R_R MIRROR yaw_corr −0.84). `recorded_identity_pass`
  7/7 (gate not broken). → **H1 confirmed (memorization / no in-family generalization)**;
  **A alone may not unlock** (removing a grounded clip's own supervision already fails). PARK.

### 3.14 Soft-endpoint reframe (zero-training) — [ESTABLISHED: KEEP PARK]
- Tested: re-score parked masked bridges under a SOFT caliper (resume-candidate set widened to
  the turn-regime span; thresholds **unchanged**; 4 honesty guards + pos/neg controls).
- Found: gate valid (pos control pass, neg control holds); **0 of 3 held-out clips revived** —
  L_L/R_L `pop_safe` stays 0 even at the pose-optimal in-regime landing; R_R `yaw_corr −0.84`
  (wrong-way, caliper-invariant). Widening even *lowers* pop_safe on the full-sup table (R_L
  0.35→0.10, R_R 0.45→0.00). → **KEEP PARK**; the reframe is formalized into the spec but does
  not break the data ceiling.

### 3.15 F4 commanded-yaw control + AR wiring — [ESTABLISHED: control plumbed on both paths]
- `ar_commanded_yaw_wiring` (MinimalGoalAR, AR/free-rollout): verdict
  **`AR_YAW_WIRING_CONNECTED_AND_READ`**; `yaw_overwrite_max_abs = 0.0` (command written
  exactly), `yaw_overwrite_connected=true`; **body sensitivity non-zero and ordered**
  (target_vs_zero pose/ego/contact 0.020/0.023/0.028 < target_vs_neg_target 0.041/0.045/0.055);
  `plain_rollout` yaw-vs-command MAE **1.778** (plain rollout ≠ command → yaw genuinely follows
  the overwrite, not free-run).
- Masked path: `cmd_yaw` is a **first-class `[B,H,1]` input** (cmd_yaw_middle from the target
  middle yaw trajectory). Three states are kept distinct and must not be conflated: **AR path**
  (MinimalGoalAR, now wired) vs **masked path** (cmd_yaw first-class) vs **MinimalGoalAR
  baseline**.
- Caveat: this proves control *plumbing*, not that the command produces a coordinated turn.

### 3.16 F5 discriminator — [ESTABLISHED: instrument cannot license a clean masked-vs-AR verdict]
- `f5_discriminator` (initial, capacity NOT matched): decision_labels
  `['AR_HELPFUL_NO_DRIFT','INCONCLUSIVE']`; flags `continuity_prior_helpful=False`,
  `ar_helpful_no_drift=True`, `ar_drift_present=False`, `plateau_ok_all=False`,
  `any_command_ignored=False`, `capacity_comparable=True`; but **param counts masked 2,518,316
  vs ar 1,187,097** (a >2× gap — so that initial "AR helpful, no drift" read was
  capacity-confounded). fullsup pop_safe: masked 0.10 / smooth 0.117 / ar 0.233; mirror_r2l:
  masked 0.00 / smooth 0.00 / ar 0.05.
- `f5_discriminator_clean` (capacity-matched + plateau-matched + per-step-vs-t drift rerun;
  arms `masked_cmd / masked_cmd_smooth / ar_cmd_capacity_matched` with `selected_ar_hidden=448`
  → capacity ratio ≈1.03; seeds 0/1/2; cells `fullsup`+`mirror_r2l`) — **[recomputed/confirmed
  from JSON top-level]**: **`primary_decision = INSTRUMENT_INVALID_PLATEAU`**;
  `validity_gates.capacity_matched = true`, `plateau_matched = false`,
  `drift_evidence_sufficient = true`; `decision_signals.ar_drift_present = true` with
  `drift_seed_cell_labels.mirror_r2l` = **AR_DRIFT_PRESENT for all 3 seeds** (fullsup all
  `AR_NO_DRIFT_EVIDENCE_STRONG`); `all_arms_fail_gate = true`, `license_grant_possible = false`;
  **`data_or_formulation_license_granted = false`**; `decision_labels =
  ["AR_DRIFT_CONFOUNDED","AR_DRIFT_PRESENT","INSTRUMENT_INVALID_PLATEAU"]`.
  → masked-vs-AR and continuity-prior questions are not cleanly decidable on the current
  instrument; AR additionally carries free-rollout drift the masked arm does not; and the
  **data bottleneck is explicitly NOT licensed**.

---

## 4. Independent recompute — the radius-normalization artifact (假阳性 #1)

This is the linchpin: it is why PHASE 2's "L_R reach_rate 0.75" / "L_R 正信号" is **WITHDRAWN**.

**Mechanism.** The gate is `min_norm = min_t cos_dist(hidden_pre_t, centroid) / anchor_radius
≤ 1.5` — it divides by **each anchor's own radius**. But the training objective
(`hidden_pre_anchor_loss`) minimizes **raw `1−cos` with no radius normalization**. So the
optimizer drives every target to roughly the same *absolute* cosine proximity, and the gate then
"passes" whichever anchor has the **largest** radius.

**[recomputed]** from PHASE 2 gate `reach_min_norm_min` × Slice-1 `anchor_radius_cos`
(abs_cos = min_norm × radius):

| target | anchor radius_cos | gate min_norm (min) | **abs_cos reached** | reach_rate | passes 1.5? |
|---|---:|---:|---:|---:|:--:|
| Walk_R_To_L | 0.004975 | 2.11 | **0.01050 (physically CLOSEST)** | 0.00 | ✗ |
| Walk_L_To_R | 0.011691 | 1.33 | 0.01555 | **0.75** | ✓ |
| Walk_L_To_L | 0.003319 | 4.72 | 0.01567 | 0.00 | ✗ |
| Walk_R_To_R | 0.007563 | 2.80 | 0.02118 | 0.00 | ✗ |

Reading the table: **L_R and L_L reach essentially identical absolute proximity (0.01555 vs
0.01567)** yet L_R "passes" (1.33) and L_L "fails badly" (4.72) — purely because L_R's anchor
radius is **3.52×** L_L's. **R_L gets physically closest of all (0.01050) but fails** because its
anchor is tight. The "success" clip is the one with the **loosest** attractor — which is exactly
the clip flagged as having **zero grounded supervision** (its onset is off the walk cycle, so its
turn-end cluster is the most diffuse). My recompute matches the independent review's table to
3–4 significant figures (it used the W1a exact min 1.32691 → 0.015514).

**Corroboration (same artifacts):** the generated L_R rollout reaches the L_R anchor at abs_cos
**0.0155**, which is *closer than any real independent turn clip* gets to it — Slice-1 offline:
nearest real clip R_L→L_R sits at min_norm 2.13 (abs 0.0249), Walk_F→L_R at 4.91 (abs 0.0574). A
Walk_F-seeded synthetic landing closer to the L_R attractor than a recorded turn is not plausible
as genuine regime membership; it is the signature of a **self-written latent proxy** (the goal
head emits one constant 512-d additive bias per clip, injected upstream of where `hidden_pre` is
measured, and LEVER 2 trains that bias on the very cosine the gate reads).

**Net:** "L_R 过门 0.75" is the **loosest-anchor + self-written-proxy** artifact, **not** "L_R is
physically closer". Every motion-level observable contradicts it: `pop_safe 0.00`, and L_R's
`mean_best_pose_d 0.117` is the **worst** of the four targets. **[WITHDRAWN]** (refuters:
W0/W1a absolute-self-reach gate k=2/3/5 all fail; exact free rollout L_R k3 0.00, yaw corr −0.48).

Also recomputed/confirmed: **§4b cond floor 0.00 is partly a per-step capture artifact** (per-step
self-reach 6.7–20.5× radius vs full-seq 0.11–0.31; STEP-0) — 假阳性 #2.

---

## 5. F5 cause-narrowing

**Masked arm — these candidate causes are EXCLUDED for `Walk_L_To_R` failing the action-only gate:**
- *yaw-wiring* — cmd_yaw is a first-class input and is read (F4 §3.15).
- *capacity* — F5 clean rerun is capacity-matched (`validity_gates.capacity_matched=true`).
- *continuity-prior* — `masked_cmd_smooth` does not rescue (F5 clean
  `continuity_prior_arch_signal=false`; soft-endpoint widening even lowers pop_safe).
- *endpoint rigidity* — soft-endpoint reframe (region + re-anchor) revives 0/3 held-out clips.
- *metric/gate validity* — recorded-identity positive control passes 7/7; W1b gate accepts real
  turns and rejects the artifact.

**AR arm has an ADDITIONAL failure the masked arm does not:** free-rollout drift
(`AR_DRIFT_PRESENT` on `mirror_r2l`, all 3 seeds; W1a free rollout L_R k3 0.00 vs pinned 1.00).

**The two residual, still-entangled causes:** **(i) data insufficiency** (L_R has zero grounded
supervision; W1d shows even grounded clips fail held-out) vs **(ii) the one-shot generation
architecture is too weak** (5 clips / single Walk_F cycle; the F5 instrument cannot plateau
cleanly → `INSTRUMENT_INVALID_PLATEAU`, `data_or_formulation_license_granted=false`). These are
**not** separated by any current artifact. Do not collapse them.

---

## 6. Design assertion vs final verdict (per-item, three-state)

| Assertion | Verdict | Note / refuter |
|---|---|---|
| §1.1 schema + `yaw_rate` load-bearing | **✓ [ESTABLISHED]** | Walk_F ego-vel phase-flat, yaw_rate ≈0; preserved unchanged by the soft-endpoint reframe. |
| §1.1 egocentric 281-d state is the generator I/O | **✗ [CORRECTED]** | Non-invertible (drops world heading), cannot feed the base model; survives only as index/anchor selection (73b plan / representation gap). |
| §2 3-type sampling | **partial [ESTABLISHED design, unused tensors]** | Sampler real + tested; PHASE 1/2 never train on its tensors. |
| §2b groundability gate; L_R fails | **✓ [ESTABLISHED]** | The most solid result; L_R contact_d 0.703 ≫ 0.30. |
| D1 within-clip gaps, O(1) marginal | **✓ [ESTABLISHED as principle]** | But "O(1)" only when grounded onset assets exist; L_R needs authoring (M2). |
| D2 condition on target state | **✓ [ESTABLISHED]** | Retained across the reframe. |
| D3 z=region / pose+contact=frame / no velocity | **✓ [ESTABLISHED]** | A3 z 0/4; used as a constraint, not over-claimed. |
| D4 soft endpoint (not a pre-selected frame) | **✓ [ESTABLISHED + formalized]** | Soft-endpoint reframe formalized it into the spec (but it did not rescue the bridges). |
| D5 clip plays the body | **✓ [ESTABLISHED, untested at runtime]** | Never reached the handoff/runtime stage. |
| D6 fine-tune from checkpoint, not greenfield | **partial ✓ [ESTABLISHED]** | The from-scratch 3a model was abandoned for the base; guarded tail fine-tune is the live mechanism. The frozen-base ceiling is confirmed; guarded fine-tune moves a latent proxy without collapsing Walk_F (B3 weak-positive). D6's specific FiLM/cross-attn mechanism was replaced by a crude constant additive bias. |
| §4 L_reach (output-space) can aim the turn regime | **✗ [WITHDRAWN]** | §4c binding-negative (min_norm 42–283×); no output term rewards hidden_pre/yaw. Correctly retired. The latent-space reach (§4d) moves the proxy but is self-written and radius-gamed → not motion. |
| §6 latent-radius reach gate | **✗ [CORRECTED] → use a motion truth gate** | Radius-normalization artifact (§4); replaced by the W1b/W1d action-only motion gate (yaw_corr/heading_MAE/pop_safe/pose). |
| §3 default AR-with-goal + upstream injection | **✗ [CORRECTED] → masked is more on-path; no upstream latent injection** | AR hit the frozen-base ceiling + free-rollout drift; spec §3 corrected to default=masked, latent read downstream only. |
| B1 in-family generalization | **[OPEN], data-bound** | Never demonstrated at motion level; W1d = memorization. |
| B2 cross-family (jump) | **out of scope** | Unchanged. |
| B3 drift controlled | **partial ✓ [ESTABLISHED, thin]** | PHASE-2 Walk_F drift guard passed but loose (OR-band) and only on Walk_F freerun, never under a real turn. |
| B4 seam no pop | **✗ [OPEN/unsolved]** | `pop_safe = 0.00` for all targets, all phases, all arms. |
| endpoint soft-reframe | **✓ formalized, [not the bottleneck]** | KEEP PARK; 0 revived. |

---

## 7. Methodology assets and the traps they caught

**Assets worth keeping:**
- **W1b motion-truth gate + positive/negative controls** — a gate with a recorded-identity
  positive control (must pass) and a straight-Walk_F negative control (must fail), at **unchanged
  thresholds**. This is the discipline that exposed the radius artifact.
- **Two-layer pre-commitment** — `raw_reach_stop` vs `upgrade_negative`, requiring BOTH train and
  eval curves to plateau (strict `|rel| ≤ 0.02`) before "upgrading" a negative to "needs fine-tune".
- **Cheap-before-expensive / instrument-before-mechanism** — data check → sampler → reach metric
  → cond floor → injection, each cheaper than the next; no large run before the B1 gate.
- **Per-clip reporting with `Walk_L_To_R` on its own row** — without it the L_R artifact would
  have been averaged into a falsely rosy aggregate.

**Self-deceptions this discipline caught (honest list):**
1. **Radius artifact** (假阳性 #1) — "L_R passes the §6 gate 0.75" → caught by recomputing
   *absolute* cosines across targets (§4). The artifacts contained the disproof; no in-loop doc
   did the cross-target absolute comparison until W0/W1a/independent review.
2. **20-step / per-step false negative** — §4b "base cond has no floor (0.00)" → caught as a
   space-misaligned per-step capture (STEP-0; full-seq self-reaches, per-step does not).
3. **Per-step confound** — the same capture misalignment also threatened §4d's calibration until
   LEVER 1 (context-window calibration) fixed it.
4. **Premature data conclusion** — "B1 is data-bound, just add L_R" → tempered by W1d showing
   even grounded clips fail held-out, and by F5's `data_or_formulation_license_granted=false`.
5. **AR↔masked extrapolation** — early "AR not wired" reads must NOT be carried to the masked
   path (different state, different gate); the reach metric is space-incompatible for masked.

**Why any success criterion must have a check independent of the training objective:** the
central failure was a metric (`hidden_pre` cosine, radius-normalized) that the optimizer could
**write into directly** via a free per-target additive bias. A proxy the trainer controls is not
evidence. The fix — and the standing rule — is that the binding check must be a **motion-level
observable** (realized yaw / heading integral / pop_safe / contact continuity) that the latent
lever cannot fabricate, plus a recorded-identity positive control and a straight-walk negative
control so the gate is provably neither always-yes nor always-no.

---

## 8. Current status + decision point

**Status.**
- **F4 (control):** SOLVED as plumbing — yaw command wired and read on both AR and masked paths
  (§3.15). New runtime entry points still need re-check; control ≠ turn generation.
- **F5 (the only remaining wall):** the masked-vs-AR / continuity-prior question is
  **`INSTRUMENT_INVALID_PLATEAU`**; AR additionally drifts (`AR_DRIFT_PRESENT` on mirror_r2l).
  Cheap probes are **exhausted** (latent levers, frozen ablation, guarded fine-tune, masked
  smoke, LOGO, soft-endpoint — all run). The F5 contract explicitly says: do not add another
  "re-clean" instrument; the only next steps are data/formulation decision or PARK.
- **Bottleneck = data + generation architecture**, entangled and **not** separated; "data
  bottleneck" is **NOT licensed** (`data_or_formulation_license_granted=false`; masked reach
  space-incompatible; W1d shows grounded clips also fail held-out).

**Decision: PARK (current call).** Do not run a further latent-lever / injection / fine-tune
round on the current 5 clips; do not relax the action-only thresholds.

**Un-park is allowed only on this falsifiable trigger (unchanged from W1d / soft-endpoint):**
1. Build/author a real `Walk_L_To_R` grounded onset that clears the **unchanged** groundability
   gate (`contact_d ≤ 0.30 ∧ pose_d ≤ 0.05`), drawable with real onset provenance (not fallback);
   **and** add a few locomotion bridge clips covering the same onset/contact regimes (esp. the
   R_L / mirror onset neighborhood).
2. Re-run the **same** W1d LOGO + soft-endpoint probe at **unchanged thresholds**
   (`yaw_corr>0 ∧ heading_MAE_rad<0.25 ∧ pop_safe>0 ∧ pose not degraded`,
   `recorded_identity_pass=true`).
3. **Un-park iff** the MIRROR-L_R held-out grounded clips now **pass** the action-only gate (and,
   for the soft caliper, ≥1 held-out clip is revived with motion consistency intact). Otherwise
   hold PARK.
4. Architecture lever (parallel, optional): if added data clears coverage but the gate still
   fails, treat it as evidence the **one-shot generation architecture** is the binding cause and
   swap to a stronger generator — judged by the **same** motion-truth gate, never by a latent
   proxy.

**Residual over-claims / contradictions to keep flagged (honest):**
- The phrase "**the real §6 AR gate**" in `..._72_73_review_record.md` §5 overstates PHASE 2: it
  is a **Walk_F-pinned** context-AR rollout (cond/contacts/angvel teacher-forced from Walk_F; the
  turn enters only as the constant latent bias), so it is structurally incapable of producing turn
  contacts/heading — which is *why* `pop_safe`/contact rates are 0/degenerate. The spec §6
  free-run-into-the-turn-manifold gate **remains un-run**. Do not cite PHASE 2 as the spec §6 gate.
- The 72_73 record's staging table still says "§6 AR gate … L_R partial pass (0.75)"; that row is
  **withdrawn** by W0/W1a and must be read with the W1c fork note (the record itself points to it).
- "No edits to `run_freerun_cycles.py` / `models.py`" is true **for the inbetween line** (it uses
  pre-existing public helpers + hooks); the file is modified on this branch for an unrelated
  contact-metric feature. Keep the qualifier.

---

### Appendix — key numbers (all re-read from artifacts; [recomputed] = derived here)

- Clips (frames): Walk_F 87, L_L 54, **L_R 50**, R_L 86, R_R 93. `act_oh` uniform `[0,1,0,0]`.
- §7.1 groundable {L_L, R_L, R_R}; L_R fail (contact_d 0.703 > 0.30). R_L pose-only f0 contact_d
  0.960 → full-state f2 0.162.
- A1 R²: z 0.84 / hidden_pre 0.80 / energy −0.05; kNN purity 0.83 vs 0.38.
- Slice-1 radius_cos: L_L 0.00332, L_R 0.01169, R_L 0.00498, R_R 0.00756 (diffuseness 0.13/0.22/0.08/0.12).
- §4c reach 0.00 all; trained min_norm 42–283× vs no-goal 4.9–11×.
- §4d goal-vs-no-goal min_norm: L_L 6.73/13.82, L_R 2.26/6.77, R_L 5.13/16.65, R_R 3.74/7.37; reach 0.00.
- PHASE 1 best L_R min_norm 1.531 (mid_add_multi_s0_s1); none cross 1.5; L_R self-reach 0.266.
- PHASE 2 §6 gate reach_rate: L_L 0.00, **L_R 0.75 (min 1.33)**, R_L 0.00, R_R 0.00; pop_safe 0 all;
  drift guard pass (pose +0.2%, pop −0.8%, root_speed +5.5%); eval min_norm not converged (0.49).
- **[recomputed] radius artifact abs_cos** (min_norm×radius): R_L 0.01050 (closest, fails), L_R
  0.01555 (passes), L_L 0.01567 (fails), R_R 0.02118 (fails). Offline real reach of L_R anchor:
  R_L→L_R 0.0249, Walk_F→L_R 0.0574 — both farther than the synthetic 0.0155.
- W1a absolute gate: L_R k=2/3/5 all fail (k=5 margin 1.003); free rollout L_R k3 0.00, yaw corr
  −0.48, heading MAE 39.6°, pop_safe 0.
- W1d full-sup action-only: L_L pass (yaw 0.965, pop 0.35), R_L pass (0.978/0.35), R_R pass
  (0.841/0.45), L_R fail (−0.780/0.00). Held-out (MIRROR-L_R & FULL-HOLDOUT): all 3 fail;
  recorded identity 7/7 pass.
- Soft-endpoint: KEEP PARK, 0/3 revived; gate valid.
- F4 AR wiring: `yaw_overwrite_max_abs=0.0`, plain-rollout yaw-vs-cmd MAE 1.778, body sensitivity
  ordered (0.020/0.023/0.028 → 0.041/0.045/0.055), verdict `AR_YAW_WIRING_CONNECTED_AND_READ`.
- F5 clean: `primary_decision=INSTRUMENT_INVALID_PLATEAU`, `capacity_matched=true` (ar_hidden 448,
  ratio ≈1.03), `AR_DRIFT_PRESENT` on mirror_r2l (3/3 seeds), `data_or_formulation_license_granted
  =false`. F5 initial (capacity-unmatched): params masked 2,518,316 vs ar 1,187,097; fullsup
  pop_safe masked 0.10 / smooth 0.117 / ar 0.233.
