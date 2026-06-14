> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# §7.2 Goal-Conditioned In-Betweening — Round Review Record

Date: 2026-05-30

Status: **FOR REVIEW.** Complete, auditable record of the §7.2 implementation round
(data pipeline + reproducible diagnostics + unit tests). No training, no model, no loss,
no scheduled sampling, no freerun runner wiring, no checkpoint dependency
(spec §0 scope lock + staging §7.2).

- Spec (source of truth): `2026-05-29_goal_conditioned_inbetweening_spec.md`
- Direction: `2026-05-29_action_handoff_goal_conditioned_inbetweening_direction.md`
- Coverage findings (companion): `2026-05-30_action_handoff_inbetween_72_coverage_note.md`

---

## 1. Task vs delivery

§7.2 (staging step 2): implement the §2 sampling + §1 schema as a pure data/sampling
pipeline; productize the one-off §7.1 data check; unit tests on shapes, egocentric
transform, grounded pairs. All four acceptance criteria below are met.

| Acceptance criterion | Status | Evidence (§) |
|---|---|---|
| Diagnostic reproduces §7.1 numbers (R_L/L_R/L_L/R_R gate verdicts) | ✅ | §4 |
| All new unit tests pass | ✅ 14/14 | §5 |
| No base-model checkpoint dependency (pure data pipeline) | ✅ | §3 |
| Deliverables: file inventory, test results, diagnostic MD summary | ✅ | §2, §5, §4/§6 |

---

## 2. File inventory (new this round)

| LOC | Path | Role |
|---|---|---|
| 631 | `train/data/action_handoff_inbetween.py` | Core pipeline (single source of truth) |
| 240 | `tools/run_action_handoff_grounded_alignment_check.py` | Productized §7.1 alignment check |
| 268 | `tools/run_action_handoff_inbetween_sampler_coverage.py` | Sampler behavior on real 5 clips |
| 106 | `tests/train/test_action_handoff_inbetween_state_semantics.py` | State/schema/yaw_rate tests (5) |
| 193 | `tests/train/test_action_handoff_inbetween_sampler_semantics.py` | Align/gate/sampler tests (9) |
| 90 | `docs/.../2026-05-30_..._72_coverage_note.md` | Coverage findings note |
| (this) | `docs/.../2026-05-30_..._72_review_record.md` | This review record |

Generated artifacts (gitignored debug_output):
`debug_output/_tmp_action_handoff_grounded_alignment_check_<date>/` and
`debug_output/_tmp_action_handoff_inbetween_sampler_coverage_<date>/` (JSON + MD each).

---

## 3. Design summary + the reviewable claims

### 3.1 Egocentric state `s_t`, D_s = 281 (spec §1.1)
`pose_rot6d[276]` (bone_rot6d.reshape) + `root_vel_ego[2]` (fwd,lat via heading rotate)
+ `yaw_rate[1]` (rad/s) + `contact[2]`. Canonical contiguous slices [0,276)/[276,278)/
[278,279)/[279,281). yaw_rate: `wrap_[-π,π](Δatan2(cond_dir))×FPS`, frame0:=frame1,
float32. contact from z-probe `future_desc[:,278:280]` (cross-clip comparable);
pose/root_vel/cond_dir from raw npz; front-aligned, `t = min(len)` per clip.

### 3.2 Full-state φ — **the one judgment call to scrutinize**
"Full-state φ" is implemented as **pose top-k neighborhood (cycle-phase localization)
refined by min contact distance**, NOT a genuine 281-d distance. Rationale, now
**artifact-backed** by the alignment tool's `standardized_281d_comparator` block
(group-normed pooled-std 281-d nearest frame, per clip):

| clip | std-281d φ (contact_d) | == pose-only φ? | full-state φ |
|---|---|---|---|
| Walk_L_To_L | f41 (0.113) | True | f37 |
| Walk_L_To_R | f40 (0.743) | True | f43 |
| Walk_R_To_L | f0 (**0.960**) | True | **f2** |
| Walk_R_To_R | f0 (0.107) | False (vs pose-only f1) | f82 |

The genuine 281-d L2 is pose-dominated → collapses to the pose-only pick (3/4 clips
exactly; R_R picks the adjacent f0, still contact_d 0.107), leaving the contact gap — most
sharply R_L f0 contact_d 0.960 vs full-state f2 0.162. Only pose-localize + contact-refine
reproduces the locked §7.1 numbers. This matches re-entry resolver design D3: ego_vel is
phase-flat and yaw_rate ≈ 0 at onset, so neither disambiguates the frame — they live in
the matched full state but do not pick it. Recorded in the diagnostic JSON/MD
(`standardized_281d_comparator`) and code docstring. **Reviewer: confirm this
interpretation of "full state" is acceptable, or specify a different combination.**

### 3.3 Groundability gate (spec §2b)
Groundable iff full-state φ clears `contact_d ≤ 0.30` AND `pose_d ≤ 0.05` (both
PROVISIONAL). 0.30 sits between R_L 0.162 (pass) and L_R 0.703 (fail) — wide margin, but
the number itself awaits freerun calibration.

### 3.4 Three-type sampler (spec §2)
- (a) within-clip biased gap: anchor sampled ∝ mean interest over the masked middle
  (interest = |yaw_rate|/pooled-std + contact-transition + clip-edge); curriculum gap
  `12 + progress·18`, **clamped to each clip's feasible max** (C+gap+K ≤ t).
- (b) grounded cross-manifold: ctx = Walk_F`[φ−C,φ]` (wraps — Walk_F is one periodic
  cycle), gt_middle = turn`[0,H]`, seam = turn`[H,H+K]`. On gate failure: scan later
  onsets, else fall back to within-clip on that turn clip with `meta.fallback` flagged.
- (c) start-state augmentation: perturb ctx last-state (noise + small phase roll) over an
  (a)/(b) base; gt_middle and seam untouched (the goal the drifted start must reach).
- Goal-conditioning interface `encode_goal(seam)` → `{goal_tokens[K,281], goal_flat,
  z_anchor?}`. Produces tensors only; no model attached.

K=6, C=16, gap 12→30, ratios 0.50/0.35/0.15 — all PROVISIONAL (spec §1/§2).

---

## 4. §7.1 reproduction (locked acceptance) — diagnostic MD summary

Run: `python3 tools/run_action_handoff_grounded_alignment_check.py`

Walk_F egocentric sanity: yaw_rate min/med/max = −0.000/0.000/0.000 rad/s; ego
lateral |max| = 0.0000 (straight walk phase-flat ⇒ yaw_rate channel is load-bearing).

| clip | pose-only φ (contact_d) | full-state φ (cyc) | pose_d | contact_d | yaw onset peak | groundable |
|---|---|---|---|---|---|---|
| Walk_L_To_L | f41 (0.113) | f37 (0.43) | 0.009 | 0.074 | 0.46 | True |
| Walk_L_To_R | f40 (0.743) | f43 (0.49) | 0.016 | **0.703** | 0.10 | **False** |
| Walk_R_To_L | f0 (**0.960**) | **f2** (0.02) | 0.011 | **0.162** | 2.26 | True |
| Walk_R_To_R | f1 (0.029) | f82 (0.94) | 0.020 | 0.013 | 0.30 | True |

groundable = {L_L, R_L, R_R}; FAILS = {L_R}. **Exact match to spec §2b** (R_L f2
contact_d 0.162 / pose-only f0 0.96; L_R fail 0.70; L_L 0.11 / R_R 0.03 pass).

---

## 5. Unit tests — 14/14 pass

Run: `python3 -m pytest tests/train/test_action_handoff_inbetween_state_semantics.py tests/train/test_action_handoff_inbetween_sampler_semantics.py -q` → `14 passed`.

State/schema (5):
- `test_schema_dims_and_slices` — 281 = 276+2+1+2; contiguous ordered slices.
- `test_walk_f_like_clip_has_zero_lateral_and_zero_yaw_rate` — ego_lat≈0, yaw_rate≈0.
- `test_turn_onset_has_nonzero_yaw_rate` — turn heading ramp ⇒ |yaw_rate| > 0.5.
- `test_yaw_rate_definition_units_wrap_frame0_dtype_shape` — ×FPS, wrap to [−π,π),
  frame0==frame1, float32, [T,1].
- `test_build_egocentric_state_shape_and_dtype` — [T,281] float32; pose/contact passthrough.

Align/gate/sampler (9):
- `test_full_state_align_prefers_contact_match_over_pose_only_min` — pose-only picks f0,
  full-state picks the contact-matching neighbor f3 (the §7.1 mechanism).
- `test_groundability_gate_fails_when_no_neighborhood_contact_match` — gate → False.
- `test_curriculum_gap_grows_monotonically` — 12→30, monotone.
- `test_within_clip_sample_respects_curriculum_gap_and_schema` — gap & shapes.
- `test_sample_type_ratios_match_config` — 0.50/0.35/0.15 within ±0.04 over 4000 draws.
- `test_augmentation_preserves_schema_and_flags_meta` — schema intact, flags set.
- `test_grounded_sample_uses_full_state_phi_and_wraps_hub_context` — φ used, hub wrap.
- `test_grounded_sample_falls_back_when_onset_not_groundable` — fallback path.
- `test_encode_goal_produces_tensors_without_model` — tensors only.

Suite still collects clean: `tests/train` 390 tests collected, no import breakage.

---

## 6. Sampler coverage on real clips (companion note §2)

Run: `python3 tools/run_action_handoff_inbetween_sampler_coverage.py`
(n=6000/progress, n_grounded=4000, seed=0). Clip frames: Walk_F 87, L_L 54, L_R 50,
R_L 86, R_R 93.

- Type mix matches config across progress (within ~0.50, grounded ~0.35, aug ~0.15).
- Curriculum gap 12→21→30; at progress 1.0 **min=28** (L_R clamped: 50 frames < C+30+K).
- Biased interest lift 1.32 → 1.02 as gap 12 → 30 (bias dilutes at long gaps).
- Grounded fallback: L_L/R_L/R_R 100% grounded_ok; **L_R 100% within-clip fallback**
  (within the configured scan window onsets 1..8, best contact_d = 0.473 at onset 8 > 0.30;
  scanning past the window contact_d clears 0.30 but pose_d then crosses the 0.05 pose gate
  — failure reason shifts contact→pose, clip stays non-groundable. Artifact: alignment
  tool `later_onset_scan_failed_clips`).

---

## 7. Findings that gate / shape §7.3

1. **B1 risk concentrated on Walk_L_To_R** — zero grounded supervision; served only by
   within-clip + augmentation. The §6 B1 gate **must report per-clip** (an explicit L_R
   row), else 3 groundable clips' reach_rate masks it.
2. **`gap_max` bounded by the shortest clip** — L_R (50 frames) clamps to 28; C=16/K=6/
   gap_max=30 interact with clip length. Set §7.3 horizon hyperparameters against the
   50-frame floor.
3. **Biased sampling washes out at long gaps** — if onset-focused supervision must persist
   at gap=30, bias the anchor position rather than the masked-middle mean.

---

## 8. Open decisions (unchanged from spec §8; gate §7.3, NOT in this round)

- Full-state φ interpretation (§3.2) — confirm or respecify.
- Model form: AR-with-goal-conditioning vs masked-token in-betweening.
- Fine-tune base weights vs separate goal-conditioned module.
- Loss weights; B1 thresholds (reach_rate≥0.7?, τ_pose, τ_pop); gate 0.30, K, C,
  gap schedule, sample ratios — all PROVISIONAL, set after the first B1 probe.

## 9. How to reproduce everything

```bash
# §7.1 alignment check (reproduces the §4 table)
python3 tools/run_action_handoff_grounded_alignment_check.py
# sampler coverage on real clips (§6)
python3 tools/run_action_handoff_inbetween_sampler_coverage.py
# unit tests (§5)
python3 -m pytest tests/train/test_action_handoff_inbetween_state_semantics.py \
                  tests/train/test_action_handoff_inbetween_sampler_semantics.py -q
```
Inputs (frozen, read-only): `debug_output/_tmp_action_handoff_z_probe_v1_20260524/
z_features_per_clip.npz` (contact) + `raw_data/processed_data/*.npz` (pose/root_vel/
cond_dir). No checkpoint.
