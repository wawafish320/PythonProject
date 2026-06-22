> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Action-Handoff In-Betweening — Soft-Endpoint Reframe (formalization + zero-training probe)

Date: 2026-05-31

Status: **DESIGN FORMALIZATION + ZERO-TRAINING FALSIFIABILITY PROBE (binding decision recorded).**
No training, no base unfreeze, no latent injection. Re-uses only existing artifacts.

Parent / prior records:
- W1d LOGO PARK lock: `2026-05-30_action_handoff_inbetween_w1c_fork.md`
  (§ `W1d LOGO — Binding Final Decision`) and the round review
  `2026-05-30_action_handoff_inbetween_72_73_review_record.md`.
- Spec being amended: `2026-05-29_goal_conditioned_inbetweening_spec.md`.

This note (1) formalizes the reframe, (2) records the spec passages it **corrects** and those it
**preserves**, (3) pre-commits a decision rule, and (4) records the probe result against that rule.

---

## 1. The reframe (locked design conclusion — formalized, not re-litigated)

1. **Switch signal ≠ endpoint; they are decoupled.** The switch signal is a discrete intent that
   selects a turn regime's **latent REGION** (direction D3, attractor-as-region). It does **not**
   specify a landing frame.
2. **The endpoint is soft and emergent.** The transition's latent enters the region naturally, at
   an **arbitrary phase**; the downstream clip **re-anchors to wherever the bridge lands**, instead
   of matching a fixed seam frame.
3. **The live part is the middle BRIDGE.** Walk_F is 87 frames; cutting from an arbitrary phase
   (e.g. frame 21) generally does not align with the turn entry phase, so there is a transition
   state that is **neither walk nor turn** and **must be generated** — it cannot be produced by an
   MM-cut or interpolation. The bridge's shape **varies with the starting phase difference**.
4. **The difficulty is decoupled coordination.** The bridge is a coordinated real state =
   root-heading ramp (cond / yaw) + pose mid-stride + contact switch, all moving together.
5. **Red line (do not repeat W1a).** The latent SUPPORTS a soft endpoint, but the latent must be
   **read from the generated motion (downstream); it is never injected upstream**. Motion
   consistency (foot-slide / realized-yaw / contact continuity) is the bridge's **verification**,
   **not the endpoint's definition**.

Decoupling, stated precisely:
- **switch signal → region**: intent picks the regime region, full stop.
- **endpoint emerges + re-anchor**: the landing phase is whatever the generated bridge reaches in
  that region; the clip re-anchors there (pose/contact phase continuity selects the resume frame;
  a rigid yaw reconciles world heading).
- **bridge = phase-difference → transition state**: the learnable object is the map
  `(start phase difference) → (neither-walk-nor-turn transition state sequence)`.

## 2. Spec corrections (this note AMENDS the following passages)

- **§1.2 target / goal `g`** — *was*: `g` = a fixed K-frame SEAM WINDOW, matched exactly.
  *Corrected to*: `g` denotes a **soft regime region endpoint**, not a fixed seam window. The
  switch signal selects the region (D3 / `z_anchor` = region membership only); the landing frame is
  not specified up front. The K-window remains useful **only** as a C1-continuity *verification*
  device at whatever frame the bridge re-anchors to — not as the endpoint definition.
- **§5 handoff** — *was*: converge exactly onto a fixed clip seam frame.
  *Corrected to*: **soft landing + re-anchor.** Timing = region membership read off the *generated*
  rollout (downstream); resume frame = the **arbitrary in-regime phase** the bridge actually
  reaches, selected by pose/contact continuity; world heading reconciled by a rigid yaw transform
  (unchanged). "Match a fixed seam frame" is explicitly retired.
- **§3 model** — *was*: default = AR-with-goal-conditioning; alternative = masked in-betweening.
  *Corrected to*: **default = masked in-betweening** (the W1c/W1d candidate; the AR-with-goal lever
  hit the frozen-base ceiling, §4d/§4e, and W1b proved metric migration, not generation). And
  **goal conditioning must NOT be an upstream latent injection** — the goal selects the region; the
  latent is read downstream from the bridge's own states. (This bans the W1a-style upstream
  `z`-radius lever.)

## 3. Spec passages PRESERVED (explicitly NOT changed)

- **§1.1 state schema + `yaw_rate`** — the 281-d egocentric state and the `yaw_rate` channel
  (heading-invariant, turn-separating) stand unchanged.
- **D6** and **D1r sampling** — preserved as locked.
- **Staging discipline** (§7: data-check → sampler → minimal model + B1 gate → only-then breadth) —
  preserved; this step adds no training and does not jump stages.
- **W1b joint gate** — preserved as the binding criterion. For the masked (281-d) candidate the
  binding clause is the **action-only** subset (`self_reach`/`hidden_pre` reach is space-incompatible
  and stays non-binding, per the W1c fork note): `yaw_corr>0 ∧ heading_MAE<τ_yaw ∧ pop_safe>0 ∧
  pose not degraded`, with the recorded-identity positive control required to pass.

## 4. Operationalization of "soft" (so it cannot be faked)

The reframe changes **exactly one knob** in the scoring: the **resume-frame candidate set** the
downstream re-anchor may pick from. Everything else (thresholds, metric function, realized-yaw
verification) is held identical to the W1d precise caliper.

- **PRECISE caliper** (W1d): candidate set = the fixed K-frame seam window `target[g0 : g0+K]`.
- **SOFT caliper** (this reframe): candidate set = the **turn-regime span** = the contiguous frames
  from `g0` up to the last in-regime frame (`|yaw_rate| ≥ frac·max|yaw_rate|`). Re-anchor = the
  best pose/contact-continuity frame within that span.

Four guards keep "soft" honest (implemented in `train/action_handoff_inbetween_soft_endpoint.py`):
- **Soft ≠ threshold relaxation.** `τ_pose`, `τ_pop`, `τ_yaw` are identical between calipers; only
  the candidate set widens. (W1a-relapse guard: soft is *region + re-anchor*, not a looser radius.)
- **`precise ⊆ soft`.** The soft span starts at `g0`, so it gives the re-anchor strictly MORE
  freedom (the reframe's best fair chance); soft can only improve pose-match, never worsen it.
- **Region = turn regime, not the post-turn walk return.** Turn clips return to the Walk_F loop
  pose at their tail (re-entry resolver); the regime mask stops at the last in-regime frame so a
  straight-walk bridge cannot cheat by pose-matching that walk-like tail.
- **Realized-yaw is caliper-invariant.** Re-anchor moves only pose-match + pop; the heading-ramp
  verification (`yaw_corr`/`heading_MAE`) is unchanged, so a wrong-way ramp (`yaw_corr<0`) can never
  be revived by re-anchoring. Plus a positive control (recorded turn must pass) and a negative
  control (straight Walk_F must still FAIL soft) guard against an always-yes gate.

## 5. Pre-committed decision rule (written BEFORE seeing the numbers)

Let the W1d LOGO held-out clips be `{Walk_L_To_L, Walk_R_To_L, Walk_R_To_R}` (the three grounded
clips; `Walk_L_To_R` is ungrounded → never a LOGO holdout). A held-out clip is **REVIVED** iff it
**fails the precise caliper, passes the soft caliper, AND keeps motion consistency under soft**
(realized-yaw correct + pop genuinely safe).

- **Gate-validity guard first.** If the positive control fails OR the negative control passes (soft
  caliper is always-yes), the gate is broken → draw **no** conclusion, **hold PARK**, fix the gate.
- **UNPARK** iff the gate is valid AND ≥1 held-out clip is revived → the reframe materially changes
  the W1d wall → update the spec and **un-park** (bridge training stays data-gated, but the soft
  endpoint lowers the data requirement, so a restart is warranted).
- **KEEP PARK** iff the gate is valid AND no held-out clip is revived (motion genuinely inconsistent
  / foot-slide) → the reframe does **not** break the data ceiling → **hold PARK**, but the soft
  endpoint is formalized into the spec for use once data arrives.
- Angle A (recorded MM-cut: fixed-frame vs re-anchor pop) is **mechanism side-evidence only**
  (non-binding).

## 6. Probe + result

Probe: `tools/run_action_handoff_inbetween_soft_endpoint_probe.py` (zero training). It re-scores
the **W1d LOGO parked masked bridges** (full-supervision table + the three per-clip MIRROR-L_R
held-out states) under both calipers, runs the two controls, and computes Angle A on recorded data.
Artifacts: `debug_output/_tmp_action_handoff_inbetween_soft_endpoint_probe_20260531/`.

**Controls (gate validity):** recorded-identity positive control **pass**; soft positive control
(recorded turn re-anchored stays pop-safe) **pass**; soft negative control holds (straight Walk_F
still **fails** soft) **True** ⇒ **gate valid**.

**Angle B — held-out (MIRROR-L_R), soft vs precise (single rows):**

| held-out clip | precise pass | soft pass | motion-consistent (soft) | revived | note |
|---|---|---|---|---|---|
| Walk_L_To_L | False | False | False | False | `pop_safe=0` precise→soft (0.00→0.00) |
| Walk_R_To_L | False | False | False | False | `pop_safe=0`; pose improves (0.095→0.081) but pop stays unsafe |
| Walk_R_To_R | False | False | False | False | `yaw_corr=-0.84` (wrong-way heading ramp — caliper-invariant) |

Soft re-anchor monotonically improves pose (confirming `precise ⊆ soft`), but the binding failure is
elsewhere: `pop_safe` stays 0 for L_L/R_L (genuine velocity/contact discontinuity at the
best-pose landing), and R_R's heading ramp is simply the wrong way (`yaw_corr<0`), which re-anchoring
cannot touch. On the full-supervision table, widening to the regime even lowers `pop_safe` for some
clips (R_L 0.35→0.10, R_R 0.45→0.00): the pose-optimal in-regime landing is *less* velocity/contact
continuous, i.e. the bridges do not produce a coordinated turn-regime landing.

**Angle A (mechanism, non-binding):** arbitrary-phase MM-cut leaves substantial pop under both
calipers; soft re-anchor reduces pop on only 1/4 clips (it can increase pop where the genuine turn
regime sits farther from a straight-walk cut frame), and always leaves residual unsafe pop with a
large cut→region gap ⇒ re-anchoring a raw cut does **not** substitute for a generated bridge;
MM-cut/interpolation is insufficient (consistent with the re-entry resolver:
`tools/run_action_handoff_reentry_resolver_diag.py`).

## 7. Decision (per §5, honestly triggered)

**DECISION: KEEP PARK.** Gate valid; **0 held-out clips revived** under the soft caliper with motion
consistency intact. The soft-endpoint reframe is now **formalized into the spec** (§2 corrections
above), but on the current artifacts it does **not** break the W1d data ceiling — the parked bridges
genuinely fail motion consistency (pop / wrong-way yaw) even when allowed to land at any in-regime
phase. This is the "soft口径下仍 fail → 保持 PARK，但形式化进 spec" branch, not a fake revival.

**What would flip it (unchanged falsifiable next gate, now in the soft口径):** re-run this probe (no
threshold change, controls must stay valid) after the W1d data-unlock contract is met (real onset
samples with `contact_d≤0.30 ∧ pose_d≤0.05`, and MIRROR-L_R held-out passing the action-only gate);
if the soft caliper then revives the held-out clips with motion consistency intact, un-park.

## 8. Scope / red lines honored

- No training, no base unfreeze, no upstream latent injection; in 281-d state space there is no
  latent to inject — region membership is read downstream from the bridge's own states.
- Recorded-identity positive control passes (gate not broken); negative control holds (gate not
  always-yes); no faked revival.
- `run_freerun_cycles` and §7.2 paths untouched; only new files added
  (`train/action_handoff_inbetween_soft_endpoint.py`,
  `tools/run_action_handoff_inbetween_soft_endpoint_probe.py`, tests).

## 9. Reproduction

```bash
# zero-training soft-endpoint probe (A + B, soft vs precise, controls, pre-committed verdict)
python3 tools/run_action_handoff_inbetween_soft_endpoint_probe.py
# pure-function tests for the soft caliper + decision rule
python3 -m pytest tests/train/test_action_handoff_inbetween_soft_endpoint.py -q
# full in-betweening suite (was 79; +13 soft-endpoint = 92)
python3 -m pytest tests/train/test_action_handoff_inbetween_*.py -q
```

Inputs are frozen/read-only (z-probe npz, raw processed npz, W1b gate-migration summary, and the
W1d LOGO parked `masked_smoke_state.pt` artifacts). The probe writes only debug artifacts.
