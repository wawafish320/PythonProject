> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Action-Handoff In-Betweening — §7.2 + §7.3 Round Review Record

Date: 2026-05-30

Status: **FOR REVIEW.** Single auditable record for the §7.2 data pipeline, the §7.3 3a
wiring smoke, the §7.3 3b reach-metric foundation (Slice 1), the §2.4 goal-injection binding
probe (§4c), the §7.3 3b reach-aware rewire probe (§4d), PHASE 1 frozen-base
capacity/injection ablation (§4e), and PHASE 2 guarded base fine-tune with the real §6 AR gate
(§5). Honest about what is done vs. what remains.

Spec: `2026-05-29_goal_conditioned_inbetweening_spec.md`. Companion notes:
`..._72_review_record.md`, `..._72_coverage_note.md`, `..._73_b1_probe_plan.md`,
`..._73b_path_ab_plan.md`.

External conclusion pointer (W1d LOGO lock): see
`2026-05-30_action_handoff_inbetween_w1c_fork.md` section
`W1d LOGO — Binding Final Decision (pre-committed branch)` for the current final
H1/H2 verdict and forced terminal direction.
Do not carry forward the older "L_R has signal" narrative without that fork note.

---

## 1. Where the line stands (staging spec §7)

| Step | Scope | Status |
|---|---|---|
| 1 data check | §7.1 grounded alignment | DONE (productized) |
| 2 sampler + schema | §7.2 pure data pipeline + tests | **DONE** (§2) |
| 3a wiring smoke | minimal AR+goal, state-space, NON-BINDING | **DONE** (§3) |
| 3b reach metric | hidden_pre anchor + reach, frozen-data | **DONE — Slice 1** (§4) |
| 3b cond floor | cond-driven base free-run, hidden_pre reach (NON-BINDING) | **DONE but CONTAMINATED** (§4b) |
| 3b binding gate (minimal) | full-seq goal injection + output-L_reach, hidden_pre reach | **DONE — binding NEGATIVE, narrow scope** (§4c) |
| 3b rewire (3 levers, frozen) | per-step calibration + hidden_pre reach loss + pre-temporal inject | **DONE-levers; small-head/shared_encoder[1] CONVERGED negative, narrow scope** (§4d) |
| PHASE 1 head/injection ablation | frozen base, ≤6 goal-head capacity/injection configs | **DONE — eval-min_norm ceiling confirmed for §4d gate; PHASE 2 allowed** (§4e) |
| §6 AR gate (real) | arbitrary-phase AR free-run + per-step reach | **RUN — L_R partial pass (0.75), all-target not yet; Walk_F drift guard passed** (§5) |

Nothing fake was produced for any gate; 3a / 3b-Slice-1 / 3b-Slice-2 are explicitly
NON-BINDING and cannot trigger the spec §6 STOP.

## 2. §7.2 — data pipeline (DONE)

`train/data/action_handoff_inbetween.py`: egocentric state s_t [281] (pose276 + ego_vel2 +
yaw_rate1 + contact2), full-state φ (pose-localize + contact-refine), groundability gate,
3-type sampler (within-clip biased + curriculum, grounded cross-manifold + L_R fallback,
start-state augmentation), `encode_goal`, torch dataset. Diagnostics:
`run_action_handoff_grounded_alignment_check.py` (reproduces §7.1: R_L f2/0.162,
pose-only f0/0.96; L_R fail 0.703; L_L/R_R pass; + standardized-281d comparator + L_R
later-onset scan), `run_action_handoff_inbetween_sampler_coverage.py` (real-clip mix/gap/
fallback). Key locked facts: full-state φ = pose-localize+contact-refine (genuine 281-d L2
degenerates to pose-only — artifact-backed); **L_R has zero grounded supervision** (100%
within-clip fallback) ⇒ B1 risk concentrated there.

## 3. §7.3 3a — wiring smoke (DONE, NON-BINDING)

`train/action_handoff_inbetween_model.py` (pooled-std group normalizer, `MinimalGoalAR`
residual AR+goal, L_middle/L_reach/L_seam_C1, free-rollout, state-space metrics) +
`tools/run_action_handoff_inbetween_b1_probe.py`. From-scratch, no checkpoint, no z-reach.
Smoke: loss 1.46→0.22; free-rollout per-clip state-space metrics. **3a caught a real
harness bug** (clip_resumable matched the whole clip incl. walk-like onset → trivially
1.00); fixed to match the resumable seam region → now discriminating: **L_R
clip_resumable 0.21 / fallback 0.79** vs 1.00 for the others (aggregate 0.80 would mask
L_R — confirms per-clip reporting is mandatory).

## 4. §7.3 3b Slice 1 — reach metric in hidden_pre space (DONE, NON-BINDING)

`train/action_handoff_inbetween_reach.py` + `run_action_handoff_inbetween_reach_anchor_check.py`.
Path A+B: reach in hidden_pre(512), NOT z (z-head unsaved; hidden_pre carries the same
regime info per A1). Validated on frozen hidden_pre: all 4 turn anchors **well-defined**
(diffuseness 0.08–0.22 ≪ 0.80); provisional CONV_DIST = 1.5×radius (0.005–0.018); Walk_F
recorded frames 5–11× the radius away (off-support OFFLINE, mirrors A2) ⇒ reach only
meaningful on GENERATED rollouts.

## 4b. §7.3 3b Slice 2 — cond-driven baseline floor (DONE experiment, NON-BINDING)

`train/action_handoff_inbetween_cond_probe.py` (pure parts) +
`tools/run_action_handoff_inbetween_b1_cond_baseline_probe.py` + 15 tests. Base ckpt
free-run from N=20 arbitrary Walk_F phases with the target turn's cond, hidden_pre captured
at `_pasa_lnq`, reach via Slice-1 metric. No edits to run_freerun_cycles (hooks + per-call
`cond_reprojection="off"`). Result: **reach_floor_rate 0.00 all four targets**
(reach_min_norm 5.6–22× radius); pop_safe 0.00. Three honest findings surfaced (not faked):
(1) `act_oh` is identical `[0,1,0,0]` across all 5 clips → action one-hot override is a
no-op (only `cond_dir` distinguishes a turn); (2) `cond_in` is per-window robust-normalized
→ a constant `cond_dir` override collapses to ≈0 (≈Walk_F), so the turn's recorded cond
*trajectory* is injected; (3) pose is heading-invariant → `clip_resumable`/`fallback` are
**degenerate** (1.00/0.00) at any seam offset and carry NO information here — the
informative floor signals are `reach_*` and `pop_safe`.

**Honest read — DOWNGRADED / CONTAMINATED (per §4c STEP 0):** the §4b 0.00 floor used the
**per-step** hidden_pre capture, which §4c STEP 0 then proved is **space-misaligned** (a
clip's own per-step capture does not self-reach: 6.7–20.5× radius). So the §4b 0.00 is
**partly a capture artifact, not a clean "base cond has no floor" negative** — do not cite
it as a clean result. The calibrated capture is full-seq (§4c). Other caveats remain
(cond trajectory phase-misaligned with seed; `pop_safe`=0 downstream of non-reach).

## 4c. §7.3 §2.4 — goal-injection BINDING probe (DONE, binding NEGATIVE, NARROW scope)

`train/action_handoff_inbetween_goal_injection.py` (GoalHead zero-init no-op,
`register_goal_injection` = non-intrusive `residual_proj` forward_hook, `l_reach`,
`reach_gate_decision` STOP semantics, calibration/aggregation) +
`tools/run_action_handoff_inbetween_b1_goal_injection_probe.py` + 13 tests. No edits to
`models.py` / `run_freerun_cycles.py`.

- **STEP 0 calibration:** full-seq capture reproduces saved anchors (relerr 0) and
  self-reaches (min_norm 0.11–0.31); **per-step capture does NOT self-reach (6.7–20.5×) →
  space-misaligned** ⇒ gate uses full-seq; also retroactively contaminates §4b.
- **STEP 1:** goal head + minimal L_reach (base frozen): L_reach 1.254 → 0.633.
- **STEP 2 BINDING gate:** `reach_rate` **0.00 all four targets**; trained min_norm
  (42–283×) is WORSE than the in-path no-goal baseline (4.9–11×). Gate decision **STOP =
  True** (per-clip, Walk_L_To_R own row).

**Scope of this STOP (narrow):** it applies ONLY to the lever "`residual_proj` hook + base
frozen + **output-space** L_reach(pose+ego_vel) + full-seq hidden_pre reach". It is NOT a
verdict on B1, for two structural reasons:
1. **Objective/metric mismatch is baked in.** The turn-discriminating signal (yaw_rate /
   heading rotation) is NOT in the base model's OUTPUT space — it lives in cond (input) and
   hidden_pre (latent). Output-L_reach on pose (heading-invariant) + ego_vel (phase-flat)
   has **no term that rewards hidden_pre reach**, so it cannot, even in principle, move the
   gated metric toward the anchor. The negative is structural-by-construction → LOW evidence
   about B1 capability; it is a design lesson.
2. **Not the spec §6 gate.** STEP 2 is a full-seq goal-conditioned ENCODE, not an
   arbitrary-phase AR free-run; and per-step AR capture is uncalibrated (STEP 0). The real
   §6 AR gate has not been run.

**→ Both structural levers this section identified (per-step calibration; a hidden_pre /
latent reach loss; pre-temporal injection) are now wired and run in §4d — the §4c STOP was
a mis-wired lever, §4d is the re-wired one trained to convergence.**

## 4d. §7.3 3b — reach-aware rewire probe (DONE-levers, CONVERGED, reach gate NOT crossed)

`tools/run_action_handoff_inbetween_reach_aware_rewire_probe.py` (frozen base; no
`models.py` / `run_freerun_cycles.py` edits) + the pure parts in
`train/action_handoff_inbetween_goal_injection.py` (`loss_plateau_status`,
`register_goal_injection_pre_temporal`, `hidden_pre_anchor_loss`). Three ordered levers, all
wired and CORRECT — this fixes exactly the §4c structural mismatch:

1. **LEVER 1 — per-step AR hidden_pre calibration.** Each AR step feeds a `context_len=16`
   teacher window and keeps the LAST hidden_pre; all four turn clips self-reach
   (`context_self_min_norm` 0.106 / 0.266 / 0.238 / 0.307, all ≤ gate) ⇒ the per-step reach
   metric is calibrated (resolves the §4c/STEP-0 space-misalignment, no anchor redefinition).
2. **LEVER 2 — hidden_pre-space reach loss.** Differentiable cos-reach on the rollout's
   hidden_pre vs the anchor centroid (NOT detached — gradient flows to the goal head). This is
   the term §4c lacked (output-L_reach had no hold on the gated metric); output-L_reach is
   kept only as a tiny 0.02-weight auxiliary.
3. **LEVER 3 — pre-temporal injection.** Goal delta injected at `shared_encoder[1]` (before
   `fw.h_temporal`), shaping regime formation rather than an end broadcast that perturbs
   hidden_pre away. Goal travels through the head + pre-temporal hook, NOT the cond channel.

**Two-layer decision (implemented + verified):** `raw_reach_stop` = the binary reach gate
did not pass; `upgrade_negative` is allowed to assert "frozen base needs fine-tune" ONLY when
BOTH curves have genuinely plateaued — **(a)** train-batch hidden loss AND **(b)** EVAL
min_norm (the quantity the gate actually reads off the gate-rollout, which decouples from the
train-batch loss). `plateau` is the strict "持平" test `|relative_improvement| ≤ 0.02` — it
excludes a curve still descending (keep training) AND a curve worsening/unstable (lower LR,
do not upgrade).

**Converged run (`frozen_step1e3_1200`; lr 1e-3, step-decay 400/×0.5, grad-clip 1.0, 1200
steps):**
- Training is a GENUINE plateau, not an LR→0 artifact: eval min_norm flattened progressively
  while LR was still 5e-4 — trajectory (step:min_norm) `0:11.34 → 480:6.26 → 560:5.95 →
  640:5.79 → 800:5.55 → 960:5.55 → 1040:5.48 → 1200:5.39`; train hidden loss
  `|rel|=0.0030`, eval min_norm `|rel|=0.0178`, both ≤ 0.02.
- **Reach lifts but plateaus SHORT of the gate.** The goal head roughly HALVES min_norm vs
  the no-goal baseline (per-clip gate min_norm mean — goal vs no-goal): L_L 6.73 vs 13.82,
  **L_R 2.26 vs 6.77**, R_L 5.13 vs 16.65, R_R 3.74 vs 7.37. So the continuous reach
  DIRECTION is correct and the levers move the gated metric — but `reach_rate = 0.00` for all
  four (gate = min_norm ≤ 1.5). `pop_safe = 0.00`; `clip_resumable`/`fallback` reported
  DEGENERATE (1.00/0.00, heading-invariant — carry no info, per §4b finding 3); reach +
  pop_safe are the informative signals.
- **Walk_L_To_R (single column, zero grounded supervision, anchor nearest Walk_F):** the
  nearest miss — gate min_norm **floor 1.81** vs the 1.5 gate. A true miss, not a too-strict
  gate: self-reach lands at 0.1–0.31 and the 1.5×radius gate is already ~5× looser than the
  recorded turn frames (threshold sanity, §5). It does NOT clear the gate on its own.

**Verdict after PHASE-1 reinterpretation:** this is a valid plateau negative for exactly the
`small_add_s1` configuration (`GoalHead hidden=256 depth=1 additive`, hook
`shared_encoder[1]`). It is NOT sufficient by itself to claim the frozen-base ceiling,
because it does not exclude "head too weak / injection point too narrow". That claim is gated
by the PHASE 1 ablation in §4e. Base stayed frozen throughout; `models.py` /
`run_freerun_cycles.py` untouched.

## 4e. PHASE 1 — goal-head capacity / injection ablation (DONE; ceiling confirmed for §4d gate)

Goal: close the "head too weak / injection too narrow" escape hatch before touching base
weights. Scope stayed frozen-base; only goal-head capacity, additive vs FiLM, and hook target
changed. All runs used hidden_pre reach loss (not detached), output-L_reach auxiliary 0.02,
per-step context-window calibration, grad clip 1.0, and context-window AR gate. Artifacts:
`debug_output/_tmp_action_handoff_inbetween_phase1_head_ablation_20260530/`.

| run | head / injection | plateau | L_R reach_rate mean/min | per-clip min_norm (L_L / L_R / R_L / R_R) |
|---|---|---|---|---|
| `small_add_s1` | h=256 d=1 additive, `shared_encoder[1]` | yes | 0.00, 2.26 / 1.81 | 5.62 / 1.81 / 4.32 / 3.04 |
| `mid_add_s1` | h=512 d=2 additive, `shared_encoder[1]` | eval near-plateau (`eval rel=0.0210`) | 0.00, 2.30 / 1.83 | 5.41 / 1.83 / 4.21 / 2.91 |
| `large_add_s1` | h=1024 d=3 additive, `shared_encoder[1]` | yes | 0.00, 2.29 / 1.83 | 4.82 / 1.83 / 4.25 / 2.97 |
| `mid_film_s1` | h=512 d=2 FiLM, `shared_encoder[1]` | yes | 0.00, 2.40 / 2.04 | 5.95 / 2.04 / 4.66 / 3.07 |
| `mid_add_early_s0` | h=512 d=2 additive, `shared_encoder[0]` | yes | 0.00, 2.02 / 1.59 | 4.43 / 1.59 / 3.70 / 2.88 |
| `mid_add_multi_s0_s1` | h=512 d=2 additive, `shared_encoder[0,1]` | yes | 0.00, 1.98 / 1.53 | 4.79 / 1.53 / 3.69 / 2.97 |

Clarification runs for the only train-loss non-plateau grid point (`mid_add_s1`, same config,
lr floor 1e-4) did not cross the gate, and showed the reach metric itself had stabilized:

- `mid_add_s1_1400_lrfloor`: L_R min_norm 1.83, reach_rate 0.00; eval min_norm plateaued
  (`eval rel=0.0167`); train hidden loss was still improving (`train rel=0.0323 > 0.02`).
- `mid_add_s1_extend1800_lrfloor`: L_R min_norm 1.81, reach_rate 0.00; eval min_norm
  plateaued (`eval rel=0.0040`); train hidden loss worsened/unstable
  (`train rel=-0.0381 < -0.02`).

**PHASE 1 decision:** frozen-base ceiling is confirmed for the §4d context-window-AR rewire
gate. The clarified criterion keys the ceiling call on eval min_norm because that is the
quantity the reach gate reads; train-batch hidden loss remains an optimization-health
diagnostic, not a hard veto when eval min_norm has stabilized. No configuration reached the
provisional gate (`min_norm <= 1.5`, `reach_rate > 0`); the closest result was
`mid_add_multi_s0_s1` with L_R min_norm 1.531. L_R self-reach is 0.266, so the best
frozen-base result is still about 5.8× the recorded-turn self-reach distance. Continuing
frozen-base injection tuning at this point is likely threshold-gaming, not solving the latent
regime transition. PHASE 2 guarded base fine-tune is allowed, with the final verdict still
reserved for the real §6 AR gate + Walk_F drift guard.

## 5. PHASE 2 — guarded base fine-tune + real §6 AR gate (RUN)

Trigger: PHASE 1 closed the cheap frozen-base escape hatch under the §4d context-window gate;
base weights were touched only after that. Artifact:
`debug_output/_tmp_action_handoff_inbetween_phase2_guarded_finetune_20260530_tail500/`.

**Guarded update actually used:**
- Unfreeze policy `tail`: 6 tensors / 515,072 params
  (`shared_encoder.4.{weight,bias}`, `shared_encoder.5.{weight,bias}`,
  `residual_proj.{weight,bias}`), not full-network fine-tune.
- Base LR 2e-5 with floor 5e-6; head LR 5e-4 with floor 1e-4; grad clip 0.5; L2-to-init
  regularizer weight 10.0. Final L2-to-init = **3.50e-6**.
- Kept goal head + hidden_pre reach loss + output `L_reach` auxiliary 0.02. The train reach
  loss consumes graph-preserving `hidden_pre` tensors with shape `[24, 512]`, dtype
  `torch.float32`, device `cpu`; it is not detached.
- Walk_F no-goal freerun drift was measured before/after as a hard guard.

**Training trajectory (500-step guarded run, not a plateau claim):** eval hidden loss
0.1373 → 0.0340; eval min_norm mean 11.34 → 3.58. Eval min_norm was still improving at the
end (`relative_improvement=0.4938 > 0.02`; trajectory
`0:11.34, 100:7.00, 200:5.31, 300:4.32, 400:4.06, 500:3.58`), so this is a guarded first
probe, not a convergence endpoint.

**Walk_F drift guard PASSED:** best_pose_d_mean 0.0582 → 0.0584
(abs +0.0001, rel +0.0021), pop_mean 0.5680 → 0.5635 (abs -0.0045, rel -0.0079),
root_speed_mean 0.5880 → 0.6202 (abs +0.0322, rel +0.0548). All three checks passed under
rel_tol 0.20 / abs_tol 0.03, so the reach lift did not come by collapsing Walk_F freerun.

**Real §6 AR gate (arbitrary Walk_F phase, context-window AR, per-step calibrated hidden_pre):**

| turn target | N | reach_rate | reach_min_norm mean/min | no-goal baseline mean/min | pop_safe |
|---|---:|---:|---:|---:|---:|
| Walk_L_To_L | 20 | 0.00 | 5.50 / 4.72 | 9.16 / 7.45 | 0.00 |
| Walk_L_To_R | 20 | **0.75** | **1.43 / 1.33** | 4.74 / 4.04 | 0.00 |
| Walk_R_To_L | 20 | 0.00 | 2.45 / 2.11 | 11.88 / 8.93 | 0.00 |
| Walk_R_To_R | 20 | 0.00 | 3.16 / 2.80 | 4.32 / 3.89 | 0.00 |

**PHASE 2 decision:** `partial_success_l_r_passes_walk_f_stable_all_targets_not_yet`.
Reach lifted above the zero floor and **Walk_L_To_R passes** the provisional reach gate
(0.75 ≥ 0.70; min_norm ≤ 1.5 for 15/20 starts) while Walk_F drift stays stable. This is not
all-target success: L_L / R_L / R_R still have reach_rate 0.00, and `pop_safe` remains 0.00 for
all clips. So the real §6 gate is no longer "NOT YET RUN"; it now says L_R has a guarded base
fine-tune foundation, but the all-target handoff/regime transition remains open. Thresholds
are still PROVISIONAL.

Representation gap (carry): §7.2 sampler is egocentric; the base generator needs base-space
training tensors (teacher path), with the egocentric sampler kept only for target/anchor +
biased/grounded INDICES.

## 6. Tests + reproduction

Unit tests (all green): `test_action_handoff_inbetween_*` = **73 passed** (goal_injection now
24, incl. `loss_plateau_status` "持平" cases and goal-head depth / FiLM / multi-target hook
coverage; PHASE 1 ablation aggregation 2; PHASE 2 guarded helpers 9; cond_probe 15; reach_anchor / sampler / state /
b1_probe_smoke as before). The plateau judge is a pure numpy function in
`train/action_handoff_inbetween_goal_injection.py`; PHASE 1 / PHASE 2 training probes are
exercised end-to-end by artifact runs, not by unit tests.

```bash
# §7.1 + standardized-281d + later-onset scan
python3 tools/run_action_handoff_grounded_alignment_check.py
# §7.2 sampler coverage on real clips
python3 tools/run_action_handoff_inbetween_sampler_coverage.py
# §7.3 3a wiring smoke (NON-BINDING)
python3 tools/run_action_handoff_inbetween_b1_probe.py
# §7.3 3b Slice 1 reach-anchor check (NON-BINDING)
python3 tools/run_action_handoff_inbetween_reach_anchor_check.py
# §7.3 3b reach-aware rewire probe (BINDING, frozen base, 3 levers → plateau) — §4d
python3 tools/run_action_handoff_inbetween_reach_aware_rewire_probe.py \
  --lr 1e-3 --lr-schedule step --lr-step-size 400 --lr-step-gamma 0.5 --grad-clip 1.0 \
  --train-steps 1200 --eval-every 80 --min-upgrade-train-steps 600 --plateau-window 80 \
  --eval-plateau-window 3 --eval-plateau-min-samples 6
# PHASE 1 frozen-base head/injection ablation (≤6 run grid)
python3 tools/run_action_handoff_inbetween_phase1_head_ablation.py \
  --out-dir debug_output/_tmp_action_handoff_inbetween_phase1_head_ablation_20260530 \
  --train-steps 1200 --eval-every 80 --lr 1e-3 --lr-schedule step \
  --lr-step-size 400 --lr-step-gamma 0.5 --lr-floor 1e-4 --grad-clip 1.0 \
  --min-upgrade-train-steps 600 --plateau-window 80 \
  --eval-plateau-window 3 --eval-plateau-min-samples 6 --max-runs 6
# PHASE 2 guarded base fine-tune + real §6 AR gate
python3 tools/run_action_handoff_inbetween_phase2_guarded_finetune_probe.py \
  --out-dir debug_output/_tmp_action_handoff_inbetween_phase2_guarded_finetune_20260530_tail500 \
  --train-steps 500 --eval-every 100 --n-starts 20 --train-horizon 24 --gate-horizon 72 \
  --lr-head 5e-4 --lr-base 2e-5 --lr-schedule step --lr-step-size 250 --lr-step-gamma 0.5 \
  --lr-floor-head 1e-4 --lr-floor-base 5e-6 --grad-clip 0.5 --base-l2-weight 10.0 \
  --unfreeze-policy tail --goal-head-hidden 512 --goal-head-depth 2 --goal-head-mode additive \
  --goal-injection-targets shared_encoder.0,shared_encoder.1
# tests
python3 -m pytest tests/train/test_action_handoff_inbetween_*.py -q
```
Inputs frozen/read-only: `..._z_probe_v1_20260524/z_features_per_clip.npz`,
`raw_data/processed_data/*.npz`, and the basetrain+lambda checkpoint used by the frozen-base
and guarded-base probes. PHASE 2 writes only debug artifacts here, not a production checkpoint.

## 7. Open decisions for the reviewer

- Confirm full-state φ interpretation (pose-localize + contact-refine) — §7.2.
- Confirm hidden_pre-space reach (Path A+B) as the binding `reach_rate` surrogate for z
  (z-head unsaved; A1 parity) — or require re-deriving/persisting the z-head.
- Confirm the Slice-2 ordering: cond-driven baseline probe BEFORE training a goal head.
- All thresholds (CONV_DIST=1.5×radius, τ_pose, τ_pop, reach 0.7, K, C, gap, ratios,
  group weights) PROVISIONAL — revisit after more guarded §6 runs, not from a single L_R pass.
- Decide whether the next PHASE 2 action is to continue the same guarded tail policy to
  eval-min_norm plateau or change the target schedule/objective for all-target reach; do not
  treat the current L_R partial pass as production handoff success.
