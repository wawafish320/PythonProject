> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Action-Handoff Goal-Conditioned In-Betweening — Independent External Review

Date: 2026-05-30

Reviewer posture: cold-start external reviewer, default-skeptical, falsification-first. I
re-derived every number I cite from the artifacts/code; I did **not** trust the in-loop
review record's self-assessment. No code or experiment logic was changed — review-only. I
re-ran the test suite and re-read the binding artifacts.

Scope reviewed: spec (`2026-05-29_goal_conditioned_inbetweening_spec.md`) + direction
(`..._goal_conditioned_inbetweening_direction.md`) → §7.2 data pipeline → 3a wiring smoke →
3b Slice-1 reach metric → §4b cond floor → §4c goal-injection binding probe → §4d 3-lever
rewire → PHASE 1 head/injection ablation → PHASE 2 guarded tail fine-tune. Primary record
under audit: `2026-05-30_action_handoff_inbetween_72_73_review_record.md`.

**One-line verdict.** The line is methodologically unusually disciplined (per-clip
reporting, NON-BINDING labels, degenerate-metric flags, two-layer plateau judge, a real
drift guard). But the headline outward claim — *"B1 has signal: L_R passes the real §6 AR
gate"* — does **not** survive an independent read. The single positive (L_R reach_rate
0.75) is, on the numbers, an **artifact of radius-normalised gating on the loosest anchor**
combined with a reach metric that the optimiser writes into directly. Every motion-level
observable (pop_safe, pose, contact) shows no seam. The honest status is **"frozen base is a
confirmed ceiling; one guarded fine-tune moved a latent proxy but produced no motion-level
transition on any of 4 targets."** Direction (fine-tune the base, not a bolt-on head) is
probably right; the *evidence that it worked* is over-claimed.

---

## 1. Design-vs-Result Scorecard

Legend: ✅ confirmed by evidence · ⚠️ partial/under-specified/unproven · ❌ refuted ·
🔵 untested.

| # | Assumption / bet | Verdict | Evidence pointer |
|---|---|---|---|
| §1.1 | `yaw_rate` is load-bearing (Walk_F ego-vel phase-flat; turn signal in heading) | ✅ | `72_coverage_note.md:45-47`; Walk_F yaw_rate ±0.000, ego_lat 0.000. Real and reproduced. |
| §1.1 | Egocentric 281-d state is the right generator I/O | ❌ (for the actual generator) | `73b_path_ab_plan.md:50-58` + record `:257-259`: the egocentric state is non-invertible (drops world heading) and **cannot be fed back through the base model**. The base generator is trained in base-space; the 281-d sampler survives only as index/anchor selection. The headline schema is not the schema the model uses. |
| §2 | 3-type sampling (within / grounded / augment) | ⚠️ implemented, not exercised in any binding run | `action_handoff_inbetween.py` sampler is real and tested, but PHASE 1/2 do not train on its tensors (representation gap). Its design value (groundability gate, biased indices) is retained; its tensor pipeline is unused downstream. |
| §2b | Groundability gate; L_R fails it | ✅ | `72_coverage_note.md:38-49`; reproduced by me via the test suite + alignment tool. L_R contact_d 0.703 ≫ 0.30 gate. **This is the most solid empirical result in the line.** |
| §2b | L_R has zero grounded supervision → B1 risk concentrated there | ✅ | `72_coverage_note.md:61-67` (100% within-clip fallback). Correctly drove per-clip reporting. |
| §3/D6 | Continue base AR + goal conditioning; fine-tune not greenfield | ⚠️→leaning ✅ | The from-scratch 281-d 3a model was **abandoned** for Path A+B (base model in its own space). Fine-tune is now the live path, but D6's specific mechanism (goal cross-attn/FiLM tokens) is replaced by a crude **constant additive latent bias** (see §2 below). |
| §4 | `L_reach` in base output space can aim the turn regime | ❌ | `goal_injection_probe_summary.md:30` + record `:108-116`: output-L_reach has **no term** that rewards the turn-discriminating channel (yaw/heading lives in cond+latent, not output). §4c binding-NEGATIVE (min_norm grew to 42–283×). This design assumption is **refuted** and correctly retired. |
| §5/D3 | `z` = region; pose/contact = resume frame; never `z` for frame | ✅ (as a *finding*) | A3 audit `:113-131`, z-agrees 0/4. Used as a constraint, not over-claimed. |
| §6 | B1-first gate: report 4 rates, STOP if reach fails | ✅ discipline / ⚠️ execution | The 4 rates are reported and L_R is on its own row. BUT 2 of 4 (`clip_resumable`, `fallback`) are **degenerate (1.0/0.0)** under heading-invariant pose and carry no info, and `pop_safe` is **0.00 everywhere** — so the gate effectively rides on one latent proxy (`reach_rate`). Honestly flagged, but it hollows the "4-rate" gate. |
| B1 | Within-clip goal-reaching generalises cross-clip in-family | 🔵 still unproven | No motion-level walk→turn transition exists in any artifact. The only positive is a latent-space proxy (§2 below). B1 is **neither confirmed nor refuted** — it has not actually been tested at the motion level. |
| B3 | Drift controlled by goal-conditioning | ⚠️ weak-positive | PHASE 2 Walk_F drift guard passed (`phase2…json:386-430`), but the guard is 3 scalar aggregates with loose OR-tolerance (see §2.4); and the goal rollout pins contacts/cond to Walk_F, so it never stress-tests drift under a real turn. |
| B4 | Seam has no pop | ❌ unsolved | `pop_safe_rate = 0.00` for **all 4 targets, all phases** (`phase2…json` per-clip). B4 is wholly unsolved and correctly surfaced. |

---

## 2. Integrity Findings (instrument artifact / over-claim / moved goalpost)

I went looking for places where the conclusion was manufactured. Findings, strongest first.

### 2.1 ❌ OVER-CLAIM / INSTRUMENT ARTIFACT — "L_R passes the §6 gate" is a radius-normalisation artifact

This is the central problem and it is decisive on the numbers.

The reach gate is `min_norm = min_t cos_dist(hidden_pre_t, centroid) / anchor_radius ≤ 1.5`
(`train/action_handoff_inbetween_reach.py:87-99`). The gate **divides by each anchor's own
radius**. But the *training objective* — `hidden_pre_anchor_loss`
(`goal_injection.py:227-254`) — minimises **raw `1−cos` with no radius normalisation**. So
the optimiser drives every target to roughly the same *absolute* cosine proximity, and the
gate then rewards whichever anchor happens to have the largest radius.

Re-derived from `phase2_guarded_finetune_summary.json` (goal min_norm) ×
`reach_anchor_check_summary.json` (anchor_radius_cos):

| target | anchor radius | gate `min_norm` (min) | **absolute** cos_dist reached | passes 1.5? |
|---|---:|---:|---:|---:|
| Walk_R_To_L | 0.00498 | 2.11 | **0.01051 (physically CLOSEST)** | ❌ |
| Walk_L_To_R | 0.01169 | **1.33** | 0.01551 | ✅ |
| Walk_L_To_L | 0.00332 | 4.72 | 0.01566 | ❌ |
| Walk_R_To_R | 0.00756 | 2.80 | 0.02115 | ❌ |

Read the middle two rows: **L_R and L_L reach essentially identical absolute proximity
(0.01551 vs 0.01566)**, yet L_R "passes" at 1.33 and L_L "fails badly" at 4.72 — purely
because L_R's anchor radius (0.0117) is **3.5× L_L's** (0.0033). And **R_L gets physically
closest of all (0.0105) but fails** because its anchor is tight. The chosen "success" is the
clip with the **loosest** attractor, which — not coincidentally — is the one clip the
pipeline flagged as having **zero grounded supervision** (its onset is off the walk cycle,
hence its turn-end cluster is the most diffuse). The narrative "L_R has a guarded fine-tune
foundation" inverts the geometry: L_R is the *easiest target to nominally reach*, not a
genuine success.

Corroborating red flag (same artifacts): the generated L_R rollout reaches the L_R anchor at
abs cos **0.0155**, which is **closer than any real independent turn clip** gets to it —
`reach_anchor_check_summary.json:64-75` shows the nearest real clip (R_L→L_R) sits at
min_norm 2.13 (abs cos 0.0249), and Walk_F itself at 4.91. A Walk_F-seeded synthetic rollout
landing *closer to the L_R attractor than a real recorded turn* is not plausible as genuine
regime membership; it is the signature of the next finding.

### 2.2 ⚠️ INSTRUMENT ARTIFACT — the reach metric is the training target, written into directly

The goal head emits **one constant 512-d vector per clip**
(`delta = goal_head(goal_flat[clip])`, `phase2…probe.py:467`) injected additively at
`shared_encoder.0/1`, upstream of `_pasa_lnq` where `hidden_pre` is measured
(`register_goal_injection_pre_temporal`, `goal_injection.py:193-224`). LEVER 2
(`hidden_pre_anchor_loss`) then trains that delta **on the very cosine distance the gate
reads**. The in-loop work was aware of this — `goal_injection_probe_summary.md:39`:
*"The goal delta enters the measured hidden_pre directly → reach is meaningful only if the
output-trained delta aligns with the anchor."* In §4c the delta was trained on *output*
L_reach and the latent reach went the *wrong* way (min_norm 42–283). The §4d/PHASE-2 "fix"
was to **train the delta directly on the latent reach metric**. So "reach improved" is close
to tautological: a free per-target additive bias optimised to reduce cos-to-centroid reduces
cos-to-centroid. The only open quantity was *how far* it could push, and the answer is
uniformly abs cos ~0.01–0.02 — i.e. **5× the self-reach absolute distance** (L_R self-reach
abs cos 0.0031 vs generated 0.0155), nowhere near where real turn frames sit.

Why this matters: a latent proxy that the optimiser can write into is not independent
evidence of a motion transition. The independent check is the **output motion**, and it
says no: `pop_safe = 0.00` everywhere; `mean_best_pose_d` for the "passing" L_R is **0.117**
— the *worst* of the four targets (`phase2…json:461`). The one target that "passes" the
latent gate has the worst pose match. That is the opposite of what a real seam would show.

### 2.3 ⚠️ OVER-CLAIM — "the real §6 AR gate" is a Walk_F-pinned partial rollout, not the spec §6 gate

Spec §6 / §7 define the gate as: from arbitrary phase, **inject a target-anchor goal,
free-run into the turn manifold**, and report z/anchor reach + pose-resumable + pop. What
PHASE 2 actually runs (`_run_context_ar_rollout`,
`reach_aware_rewire_probe.py:217-343`) is:

- pose + root-vel are genuinely free-carried (real AR on the motion channel) ✅;
- **but `cond_in`, `cond_tgt_raw`, `contacts`, `angvel` for every future step are pulled
  from the recorded Walk_F clip** (`:322-329`) — teacher-forced, not generated;
- the turn target enters **only** as the constant latent bias, never as cond or contact.

So the model is conditioned on Walk_F throughout and its contact stream is pinned to Walk_F.
It is structurally incapable of producing turn contacts or turn heading dynamics — which is
*why* `pop_safe` and the contact-based rates are 0/degenerate. Calling this "the real §6 AR
gate" (record `:240`, `:30`) over-states it. It is materially better than §4c's full-seq
encode (motion is now AR), but it is a **hidden_pre-reach proxy on a Walk_F-conditioned
rollout**, not the spec's free-run-into-the-turn-manifold test. The spec §6 gate remains
un-run.

### 2.4 ⚠️ WEAK GUARD (not refuted) — drift guard tolerance is loose, but the measured drift is small

`_drift_guard` passes a check if `abs_delta ≤ 0.03` **OR** `rel_delta ≤ 0.20`
(`phase2…probe.py:194`). That OR with a 20% relative band is a generous guard. The only
non-trivial mover is `root_speed_mean` +5.5% (0.588→0.620), which fails the abs band
(Δ0.032 > 0.03) and passes only on the relative band. Pose drift is +0.2% and pop is −0.8%.
**Judgment:** the guard *passed legitimately* because the actual deltas are small — I do not
dispute B3-stability for this 500-step run. But (a) the band is loose enough that it is a
weak guarantee, and (b) the guard only watches 3 scalar aggregates of *Walk_F* freerun, with
`best_pose_d` heading-invariant; it never measures drift under an actual turn rollout (which
this setup can't produce anyway). Credible-but-thin, not an artifact.

### 2.5 🔬 GOALPOST — PHASE-1 ceiling criterion was relaxed from dual-gate to eval-min_norm-only

This is the one the brief most wanted an outside read on. Verdict: **a real relaxation,
empirically backstopped, but documented as if it were always the rule.**

- §4d's two-layer rule (record `:142-148`) requires **BOTH** (a) train-batch hidden loss
  AND (b) eval min_norm to plateau before asserting "frozen base needs fine-tune". The
  strict `|rel| ≤ 0.02` test explicitly says a *still-descending* train curve → **keep
  training**, a *worsening* train curve → **lower LR, do not upgrade**.
- PHASE 1 (record `:200-210`, `phase1_final_decision.md:5,33-36`) re-keys the ceiling on
  **eval min_norm only**, demoting train-batch loss to "optimization-health diagnostic, not
  a hard veto." The grid point that forced this, `mid_add_s1`, had train loss **still
  improving** (`rel=0.0323 > 0.02`) at 1400 steps and **worsening** (`rel=−0.0381`) at 1800
  — i.e. under the *original* rule, neither clarification run licenses the ceiling claim
  (one says "keep training", the other "lower LR").

Is it principled or convenience? Both, partly. **Principled side:** eval min_norm *is* the
quantity the gate reads, and they did the honest thing — *extended* the non-plateaued config
to 1800 steps and reach did not move (L_R stuck at 1.81–1.83). So empirically the escape
hatch ("train the head longer") is closed. **Convenience side:** the *stated criterion* was
rewritten to retire a rule that was deliberately conservative, and the rewrite is what
unlocked PHASE 2. The clean, non-goalpost framing would have been: *"we extended the one
non-plateaued config and reach still didn't cross, so we treat the ceiling as empirically
confirmed despite train-loss not formally plateauing."* Instead the doc asserts a new general
criterion. **Net: defensible conclusion, mild goalpost in the framing.** It does not change
the PHASE-1 bottom line (no config crossed; best 1.531 ≈ 5.8× self-reach), so the ceiling
call itself stands.

### 2.6 ✅ VERIFIED-HONEST items (no artifact found)

- **`clip_resumable`/`fallback` degeneracy under heading-invariant pose** is correctly
  caught (the 3a harness bug where it matched the whole clip → trivially 1.00, record `:54`),
  re-flagged everywhere with `_DEGENERATE` suffixes in the JSON, and explicitly excluded from
  the signal. Good.
- **§4b cond-floor 0.00 was retroactively downgraded/contaminated** once STEP-0 proved
  per-step capture is space-misaligned (record `:82-87`). They did not cite the contaminated
  negative as clean. Good.
- **STEP-0 per-step vs full-seq calibration** is real: per-step capture does **not**
  self-reach (6.7–20.5×, `goal_injection_probe_summary.md:11-16`), full-seq does (0.11–0.31);
  the context-window per-step calibration self-reaches (0.106–0.307,
  `phase2…json:7-66`) — I confirmed these from the JSON. The calibration is legitimate.
- **"No edits to run_freerun_cycles.py / models.py":** the file *is* modified on this branch
  (+421 lines), but those are an unrelated `injection_contracts`/contact-metric feature; the
  inbetween probes use only pre-existing public helpers (`FreeRunCycleRunner`,
  `_build_full_cycle_sample`) + forward hooks. **The claim holds for the inbetween line.**
- **`73 passed`** — reproduced exactly (`pytest … -q` → 73 passed in ~7s).
- **Per-clip reporting with L_R on its own row** is enforced end-to-end and is the single
  best discipline decision in the line — without it the L_R artifact in §2.1 would have been
  averaged into a falsely rosy aggregate.

---

## 3. Direction Judgment

**Is the line viable to continue? Yes, but with the scoreboard reset.** The genuinely
established facts are narrow and mostly *negative/foundational*:

1. Output-space `L_reach` cannot aim a turn regime (refuted, retired). ✅
2. Frozen base + latent goal injection is a ceiling — no head capacity/injection point
   crosses the gate without threshold-gaming. ✅ (this is solid; PHASE-1 ablation is good
   work regardless of §2.5's framing wrinkle).
3. A guarded tail fine-tune *can move a latent proxy* without collapsing Walk_F. ✅
4. **No motion-level walk→turn transition has been demonstrated for any target.** The B1
   make-or-break is **still unanswered**, not "has signal."

**The next wall to hit.** The brief proposes *B4 seam + tighten reach*. I partly disagree on
ordering. Tightening reach and chasing the seam both presume the latent proxy is a faithful
target — and §2.1/§2.2 show it currently is not (it's radius-gamed and directly written). So
the **first** wall is **measurement**, then **B4**:

- **W0 (measure before you build):** replace the radius-normalised, self-written gate with a
  metric that cannot be satisfied by a constant latent bias. Concretely: (a) gate on
  **absolute** hidden_pre proximity *relative to the self-reach floor* (e.g. require
  generated abs-cos ≤ k× self-reach, not ≤ 1.5× radius), so a target's loose anchor can't buy
  a pass; (b) require the reach to be **carried by the output motion** — score reach on a
  hidden_pre re-derived from a *fully* free rollout (turn cond + generated contacts), not a
  Walk_F-pinned one; (c) cross-check with a motion-space turn statistic (realised yaw_rate /
  heading integral over the rollout), which is the actual thing B1 needs and which no current
  metric reports.
- **W1 (then B4 seam):** only meaningful once a rollout demonstrably enters a turn at the
  motion level. Right now `pop_safe=0` is over-determined (contacts are Walk_F), so seam work
  would be optimising an instrument, not the seam.
- **Data, not levers:** the audit and coverage notes both say B1 quality is data-bound and
  L_R specifically has *zero* grounded supervision. The highest-leverage move is likely
  **+locomotion clips / a grounded L_R onset**, which is O(1) marginal by design (D1), rather
  than more injection-point tuning on 5 clips. The team's own §2.6/closeout says this; the
  PHASE-1/2 effort drifted into latent-lever tuning instead.

So: **agree the seam/B4 is unsolved and important, but it is not the next wall.** The next
wall is an honest motion-level reach metric; B4 comes after.

---

## 4. Risks / Blind Spots (outside view)

- **Proxy capture.** The whole binding chain is measured in `hidden_pre`, a space the goal
  head writes into directly. An in-loop reviewer who trusts "reach_rate" inherits a metric
  that is partly self-fulfilling. The radius-normalisation cross-target inversion (§2.1) is
  invisible unless you compute absolute cosines across targets — which the artifacts contain
  but no doc does. **This is the single most important thing to internalise.**
- **"L_R is the success" is exactly backwards.** L_R was correctly flagged as the *highest
  risk* (no grounded supervision). It then "passes" — and the reason it passes (loosest
  anchor) is *because* it's the worst-supervised clip. There is a real danger of reading the
  one weak clip's normalised pass as the foundation to build on.
- **Representation-gap drift.** The headline design artifact (egocentric 281-d sampler, full
  schema, 3-type sampling, from-scratch 3a) is largely **decorative** w.r.t. the actual
  generator, which trains in base-space off the teacher path. Months of schema/sampler design
  feed only index selection. Worth an explicit decision: keep investing in the egocentric
  pipeline or treat it as a sunk scaffold.
- **5-clip / single-Walk_F-cycle ceiling.** Everything (anchors, self-reach, phase aliasing)
  is calibrated on one Walk_F cycle and 50–93-frame turn clips. L_R is 50 frames — too short
  for the C16+gap+K6 window at long gaps (`72_coverage_note.md:74-78`). Any "tighten reach"
  effort will collide with this data floor before it collides with a modelling limit.
- **Thresholds are provisional but load-bearing.** `conv_norm_thr=1.5`, `reach_gate=0.70`,
  `tau_pop`, `tau_pose` are all smoke-set. The line is honest that they're provisional, but
  the *only* positive result in the entire line depends on exactly one of them (1.5×radius)
  applied to exactly one anchor. That is a thin reed for an outward "B1 has signal" claim.
- **Good practices to keep.** Per-clip + L_R-own-row reporting; NON-BINDING labelling and the
  STOP discipline; degenerate-metric flagging; the no-goal baseline column; L2-to-init guard
  on fine-tune; running the non-plateaued config longer instead of declaring victory. These
  are above the bar for this kind of exploratory line and should be preserved.

---

## 5. Bottom line for the record

The in-loop record's own *internal* hedges are mostly accurate ("partial", "not all-target",
"pop_safe=0", "thresholds provisional"). The problem is the **emphasis**: a reader comes away
believing L_R is a beachhead. Independently, L_R's pass is a radius-normalisation artifact on
a self-written latent proxy, with every motion-level observable negative. The correct outward
sentence is:

> *"Frozen base is a confirmed ceiling. A guarded tail fine-tune moved the latent reach proxy
> toward all four anchors without collapsing Walk_F, but produced no motion-level walk→turn
> transition on any target (pop_safe 0/4; the one gate 'pass' is the loosest-anchor clip and
> has the worst pose match). B1 remains unanswered. Next: an honest motion-level reach metric
> and more grounded data — not more latent-injection tuning."*

That is a *useful* result — a clean ceiling and a working guarded-fine-tune harness are real
progress — but it is a foundation-laying negative, not "B1 has signal."
</content>
</invoke>
