> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §3.2/§9 under its stated read-only / zero-new-injection scope.

# §7.3 3b — Path A+B Plan (base-integrated generator, reach via hidden_pre)

Date: 2026-05-30

Status: FOR REVIEW. Plan for the binding B1 gate after the 3b feasibility finding. Records
what is DONE (Slice 1, real) and scopes the remaining days-level build (Slice 2) honestly.

Decision (locked): **Path A+B** — the goal-conditioned generator is the base
`EventMotionModel` (init from the basetrain+lambda checkpoint, spec D6) + a goal head,
free-running in the base model's native space; `reach_rate` is measured in **hidden_pre
(512-d)** space, not `z`.

## 0. The 3b feasibility finding (why literal 3b was blocked)

Confirmed by inspection:
1. **The z-head (ZEncoder) was never persisted** — it is trained per-run inside
   `tools/run_action_handoff_z_probe_v1.py`; only z VALUES were saved
   (`z_features_per_clip.npz`). The checkpoint holds only the base model
   (`bone_adapters`, `_pasa_lnq`, `period_encoder`…), no z-head.
2. **The base model consumes native world-frame multi-stream input** —
   `model(motion, cond_in, contacts=, angvel=, pose_history=)`; `hidden_pre` is captured at
   `model._pasa_lnq`. The §7.2 generator's 281-d egocentric state is heading-invariant and
   carries no angvel/cond, so it **cannot be fed back through the base model** (the
   egocentric transform deliberately drops world heading, §1.1 — not invertible).
3. **`run_freerun_cycles` is the base-model AR rollout harness** (init from this ckpt,
   `EventMotionModel`, autoregressive) — spec §6 explicitly says the B1 gate harness =
   extend it to inject a goal + log z-to-anchor.

⇒ The spec-faithful generator must live in the **base model's space**, not as a bolt-on to
the standalone 281-d 3a scaffold. Path A+B resolves (1) by using hidden_pre (A1: hidden_pre
decodes contact R² 0.80 ≈ z 0.84 — same regime info) so the unsaved z-head is not needed.

## 1. DONE — Slice 1: the reach metric in hidden_pre space (real, validated)

- `train/action_handoff_inbetween_reach.py` — `HiddenPreAnchor`, `build_hidden_pre_anchors`
  (end-window centroid + radius, A2 geometry but in hidden_pre), `reached` /`min_norm`,
  `load_hidden_pre`.
- `tools/run_action_handoff_inbetween_reach_anchor_check.py` — validates anchors + offline
  separation, JSON/MD. NON-BINDING (no model).
- `tests/train/test_action_handoff_inbetween_reach_anchor.py` — 3 tests.

Validated on frozen hidden_pre: all four turn anchors **well-defined** (diffuseness
0.08–0.22, far below the A2 0.80 bar); provisional CONV_DIST = 1.5×radius
(0.005–0.018); recorded Walk_F frames sit **5–11× the anchor radius** away — off-support
OFFLINE (mirrors audit A2 in z-space), confirming reach is only meaningful on GENERATED
rollouts.

## 2. TODO — Slice 2: the goal-conditioned base generator (the days-level remainder)

### 2.1 Representation-gap finding (must be handled)
The §7.2 sampler produces **egocentric 281-d** tensors; the base generator needs
**base-space** training data (`motion/cond/contacts/angvel/pose_history`, normalized +
tanh-compressed). So the egocentric sampler does NOT directly feed the base generator. Use:
- the base-space teacher data path (reuse `run_teacher_rollout` / the z-probe input
  pipeline) for the tensors fed to the model;
- the §7.2 egocentric sampler for the **heading-invariant target/anchor definition** and
  the **biased / grounded sampling INDICES** (which frames to pair) — its design value is
  the sampling policy + groundability gate, not the tensor format here.

### 2.2 Recommended first measurement — NO-TRAINING cond-driven baseline probe (NON-BINDING floor)
Before building/training any goal head, get the cheapest real *floor* signal — explicitly
**NON-BINDING** (it cannot trigger the spec §6 STOP):
- The base model is already conditioned on `cond_in` (act_oh + cond_dir + cond_speed).
- From N≥20 arbitrary Walk_F phases, **override cond to the target turn's action/direction**,
  free-run the base model (extend `run_freerun_cycles`), capture hidden_pre, apply the
  Slice-1 reach metric.
- This is a **floor diagnostic**: it tells us whether the EXISTING base cond conditioning
  already carries signal toward the turn anchor from an arbitrary start. It does NOT define
  the gate — there is no goal head yet, so a low floor is informative but NOT evidence
  against B1, and a high floor does not "pass" B1. **The binding gate remains §2.4** (the
  base-space free-run AFTER goal head / goal injection). Report per-clip, Walk_L_To_R
  separate, marked NON-BINDING floor.

### 2.3 Goal head + fine-tune (if 2.2 shows a foundation)
- Inject the goal seam (`encode_goal` window) into `EventMotionModel` via FiLM /
  cross-attention conditioning tokens; init base from ckpt, goal head new.
- Losses: minimal three from the 3a plan, ported to base-space output (L_middle AR recon,
  L_reach to the seam, light L_seam_C1). AMP/foot/smooth deferred.
- Scheduled sampling minimal; teacher-forced train, **free rollout** eval.

### 2.4 Binding gate (Slice 2 output)
Extend `run_freerun_cycles`: arbitrary-phase start + goal injection + hidden_pre capture →
reach via Slice 1. Report `reach_rate / clip_resumable_rate / pop_safe_rate /
fallback_rate`, **per-clip with Walk_L_To_R on its own row**. THIS run is binding: gate on
`reach_rate ≥ 0.7` [PROVISIONAL], **STOP if reach fails** (spec §6). 3a and Slice 1 are
NON-BINDING.

## 3. Staging order for Slice 2

1. Reach metric (Slice 1) — **DONE**.
2. cond-driven baseline probe (§2.2) — cheapest NON-BINDING floor diagnostic, no training.
3. Goal head + minimal fine-tune (§2.3) — only if §2.2 shows a foundation.
4. Binding gate report (§2.4); STOP/iterate per spec §6.

## 4. Out of scope (unchanged)

Full L_imitation (AMP), L_foot, L_smooth, full scheduled-sampling schedule, handoff runtime
contract, production thresholds, large training. z-head re-derivation is avoided entirely
(hidden_pre reach). All thresholds PROVISIONAL.
