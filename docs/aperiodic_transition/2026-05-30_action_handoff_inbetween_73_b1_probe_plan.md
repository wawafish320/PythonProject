> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §9 under its stated read-only / zero-new-injection scope.

# §7.3 Minimal B1 Probe — Plan (LOCKED before scaffold)

Date: 2026-05-30

Status: implementation plan for spec §7.3 (staging step 3), scoped down to the **minimal
B1 probe** — NOT full goal-conditioned productionized training. FOR REVIEW. Builds on the
frozen §7.2 data pipeline; does not modify §7.2 logic unless the probe exposes a sampler
bug.

> **SUPERSEDED (reach metric) by `..._73b_path_ab_plan.md`.** The 3b feasibility finding
> showed the z-head (ZEncoder) was never persisted and the egocentric state can't be fed
> back through the base model, so the binding `reach_rate` is now measured in **hidden_pre
> (512-d)** space, NOT via a frozen z-encoder / z-region. Section **D-B below is obsolete**
> wherever it says "z-encoder / z-reach" — read it as **hidden_pre reach** (see §0/§2 of
> the 73b plan). The 3a/3b NON-BINDING staging (D-A) still stands.

- Spec: `2026-05-29_goal_conditioned_inbetweening_spec.md` (§3 model default, §4 losses,
  §6 B1-first gate, §7 staging step 3).
- §7.2 boundary: `2026-05-30_action_handoff_inbetween_72_review_record.md`.
- Sampler interfaces consumed: `train/data/action_handoff_inbetween.py`
  (`InbetweenSampler`, `Sample` = ctx[C,281]/gt_middle[H,281]/seam_target[K,281],
  `encode_goal`).

## 0. Why scope down (the make-or-break)

§7.2 locked the data + risk surface. The unanswered make-or-break is spec B1: from an
arbitrary Walk_F phase, conditioned on a turn goal, **can the generator reach the target
manifold?** Spec §6 requires reporting `reach_rate / clip_resumable_rate / pop_safe_rate /
fallback_rate` and **STOP if reach fails — do not expand**. So the first thing built must
be a *fair* B1 test, nothing more.

## 1. Two locked decisions (and why they compose)

**D-A — staged init.** The probe is built in two stages:
- **3a wiring smoke** — from-scratch tiny model, **no checkpoint**. Validates the harness:
  losses decrease, free-rollout runs, state-space metrics + JSON/MD gate emit. Its reach
  numbers are **NON-BINDING** and **cannot trigger the spec STOP** (a randomly-initialized
  model on ~370 frames in a minute-scale smoke fails to reach by construction — that is a
  false negative, not evidence against B1).
- **3b binding gate** — generator **initialized from the basetrain+lambda checkpoint**
  (spec D6/§3: drift lives in the base; the line is "continue the existing AR generator").
  Only 3b's gate is binding; only 3b can trigger STOP.

**D-B — reach_rate rides on the base model forward [REVISED → hidden_pre, see 73b plan].**
`reach_rate` was specified on `z` (`cos_dist(z_t, g.z_anchor) ≤ CONV_DIST`, spec §6/§5),
and computing it for a *generated* frame needs `hidden_pre = base_model(frame)` — so it is
intrinsically a 3b metric (the base model is only loaded in 3b). **Revision:** the z-head
was never persisted, so the binding metric is now `cos_dist(hidden_pre_t, hidden_pre
anchor) ≤ CONV_DIST` (hidden_pre carries z's regime info per A1). z-reach wording below is
obsolete. This composes exactly with D-A:
- 3a reports **state-space metrics only** (no model): `clip_resumable_rate`,
  `pop_safe_rate`, `fallback_rate`, and a state-space *reach proxy* explicitly labelled
  NOT the z-reach.
- 3b adds the real `reach_rate` (z-region) once the base model is loaded.

## 2. Module ownership (new files only; §7.2 untouched)

- `train/action_handoff_inbetween_model.py` — minimal AR-with-goal model + losses +
  pooled-std group normalizer + free-rollout + state-space metric fns. (Model belongs in
  `train/`, not `train/data/`.)
- `tools/run_action_handoff_inbetween_b1_probe.py` — probe runner: load real clips →
  normalize → short teacher-forced smoke train → free-rollout eval → per-clip JSON/MD gate.
- `tests/train/test_action_handoff_inbetween_b1_probe_smoke.py` — shapes, loss-decreases-
  on-overfit, rollout/metric schema, per-clip gate keys incl. Walk_L_To_R.

## 3. Model — minimal AR-with-goal (spec §3 default)

Per step, predict the next frame from the current state, the context, and the goal:
- `ctx_emb` = small encoder over ctx[C,281] (GRU last hidden or MLP over flattened ctx).
- `goal_emb` = MLP over `encode_goal(seam_target)` (seam window [K,281] pooled/flattened).
- step: `s_{t+1} = s_t + Δ(s_t, ctx_emb, goal_emb)` (residual next-frame).
All states normalized by the pooled mean/std group-norm (so the 276-d pose term does not
drown ego_vel/yaw_rate/contact — spec §1.1 posture). Tiny (hidden≈128); minute-scale smoke
on CPU. AR default per spec §3; masked-token alternative deferred.

## 4. Losses — minimal three (teacher-forced training)

Operating in normalized (std) space; channel groups pose[276]/ego_vel[2]/yaw_rate[1]/
contact[2] weighted (default equal in std units; tune later).
- **L_middle** = MSE over the teacher-forced middle prediction vs `gt_middle[H,281]`. This
  is the AR-reconstruction / data-manifold term — the **minimal stand-in for L_imitation**;
  the heavier AMP / feature-matching imitation (spec §4) is **deferred**.
- **L_reach** (spec §4, w=1.0) = weighted L over the horizon-end K-window prediction vs
  `seam_target[K,281]` (pose + ego_vel + yaw_rate + contact). Window (not single frame) ⇒
  enforces C1.
- **L_seam_C1** (light) = continuity penalty on (ego_vel, yaw_rate, contact) between the
  last generated frame and the seam's first frame.
Deferred (NOT in the minimal probe): L_foot (full foot-slide/contact), L_smooth, AMP
imitation, full scheduled-sampling schedule.

## 5. Eval — FREE rollout (not teacher-forced), per-clip gate

Training is teacher-forced; **the gate measures free AR rollout** (spec §6 freerun). From
N≥20 arbitrary Walk_F start phases (ctx = Walk_F[i−C, i], wrapping the periodic cycle),
condition on each turn target's seam anchor, roll ~H+K frames with NO teacher forcing.

Per (start, target) metrics:
- **clip_resumable_rate** — min pose_d(rollout frame, target-clip frame) ≤ `τ_pose`
  (state-space; no model).
- **pop_safe_rate** — at the best-resumable frame, handoff discontinuity in
  (ego_vel, yaw_rate, contact) ≤ `τ_pop` (state-space; this is the B4 seam result —
  reach can pass while this fails, so it is reported separately).
- **fallback_rate** — routed to fallback (not resumable).
- **reach proxy (3a)** — state-space distance into the target seam-window region, labelled
  NOT the z-reach. **(3b)** replaces it with the real z-region `reach_rate`.

**Reporting is PER-CLIP, with Walk_L_To_R on its own row** (review record: L_R has zero
grounded supervision, so an aggregate `reach_rate` would mask it). An aggregate row is
allowed but never the sole signal.

**Gate semantics:** 3a is NON-BINDING (harness validation). 3b gates on
`reach_rate ≥ 0.7` [PROVISIONAL] as the make-or-break, reports all four rates, **STOPs if
reach fails** (spec §6) — and reports L_R separately so a high aggregate cannot hide an
L_R failure.

## 6. Thresholds — all PROVISIONAL (smoke/probe only)

`τ_pose`, `τ_pop`, `CONV_DIST`, the 0.7 reach gate, group weights, N, smoke train steps —
all provisional, set/calibrated after the first fair (3b) probe. `g.z_anchor` (3b) reuses
the audit A2 end-window centroid + radius.

## 7. Out of scope (unchanged; later than this step)

Full L_imitation (AMP/feature-matching), L_foot, L_smooth, full scheduled-sampling
training, handoff runtime contract wiring, freerun runner integration, production
thresholds, any large training run. Per spec §7 step 4: breadth is added **only if B1
shows signal**.

## 8. Definition of done for 3a

Harness runs end-to-end: smoke train loss decreases; free rollout produces per-clip
state-space metrics (incl. L_R row); JSON/MD gate written to
`debug_output/_tmp_action_handoff_inbetween_b1_probe_<date>/`; unit test green; clearly
marked NON-BINDING / no-checkpoint / no-z-reach. No checkpoint, no base-model dependency in
3a.
