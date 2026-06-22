> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Root/Pose decouple measurement — is `between` a low-dim root problem?

Date: 2026-06-05

Status: **read-only / measurement only.** No training; no production trainer/runtime/gate/
checkpoint touched; no loss/model change. No reopening of the representation-conditioning
debate (layer-2 is settled). This measures **problem dimension / decoupling**, not "anchored
advantage".

Tool: `tools/run_action_handoff_root_pose_decouple_measure.py`
Artifacts: `debug_output/_tmp_action_handoff_root_pose_decouple_measure_20260605/`
(`summary.md`, `summary.json`, `per_clip.csv`, `per_window_a.csv`, `decouple_usability.csv`).

## 0. Claim under test

> `between` may be a **low-dimensional root problem, with `pose == Walk_F`**: walk is "go
> forward", turn is "rotate the root"; the local joint articulation stays the same walk cycle.

Two sub-questions, measured on existing data only:
- **(a) representation**: in the root-excluded local representation, how different are Walk_F
  and the turns, and how much of that difference is *root* vs *local residual*?
- **(b) usability**: even if local is near-invariant, does driving the Walk_F local cycle along
  a *real turn root path* keep the feet honest in world frame (foot_slip / support_side)?

## 1. `directlocal` construction — confirmed before measuring

The canonical egocentric state (`train/data/action_handoff_inbetween.py`) is
`state281 = pose_rot6d[0:276] + ego_vel[276:278] + yaw_rate[278:279] + contact[279:281]`. The
raw npz (`raw_data/processed_data/<clip>.npz`, `state_layout_json`) carries
`RootPosition[3] + RootVelocity[2] + BoneRotations6D[276] + BoneAngularVelocities[138]`.

Empirically verified frame conventions (these decide what "root-excluded" means here):

1. **`rot6d` is a heading-canonical (root-excluded) local pose.** The root bone (dims 0:6) has
   tiny temporal std (~0.1) and barely rotates, while the world path curves up to ~35° over a
   clip and `root_pos` is recentred to the origin at frame 0 for every clip. The world heading
   is **not** carried in the bone rotations.
2. **World heading is reapplied at reconstruction time via `cond_dir`.** The canonical
   reconstruction (`_world_root_vel_from_ego` + `_integrate_root_pos`, the path that scores GT
   at acceptance 1.0) rotates the egocentric `ego_vel` into the `cond_dir` frame and integrates
   it to `root_pos`; FK (`fk_positions_from_rot6d`) then uses `rot6d` directly + `root_pos`.
3. **`bone_ang_vel` is per-bone local angular velocity** (bone0 rms 0.25–0.37 ≈ the limb bones,
   i.e. no dominant shared world-rotation offset → not a world-frame signal).

Operational split used throughout:
- **root channels**: `ego_vel(2) + yaw_rate(1) + cond_dir heading + rot6d root-bone[0:6] +
  bone_angvel bone0[0:3]`. The world root *path* is fully determined by `{ego_vel, yaw_rate,
  cond_dir, root_pos[0]}`.
- **local (root-excluded) channels**: `rot6d non-root bones[6:276]` (270) + `bone_angvel
  non-root[3:138]` (135).

This matches the prompt's `directlocal = {rot6d local pose, root-local bone_angvel}`; the only
clarification is that the *heading/root path lives outside `state281`* (in `cond_dir` +
`root_pos`, reapplied at reconstruction), not inside the pose channels.

## 2. Path identity (preflight)

Items, reconstruction, GT reconstruction, band calibration and acceptance scoring are imported
verbatim from `run_action_handoff_oracle_schedule_trajectory_decoder_smoke`, with the flags
under which reconstructed GT acceptance == 1.0: `oracle_contact_passthrough=True`,
`command_align_root_vel=False`, `reconstructed_baseline_quantile=100.0`, `horizon=16`,
`context_len=16`, `stride=1`, `min_run_frames=2`.

- **GT reconstructed acceptance = 1.000** across all 188 matched windows → results below are on
  the same calibrated yardstick.
- Matched windows = 3 groundable turn clips `Walk_L_To_L`(39)/`Walk_R_To_L`(71)/
  `Walk_R_To_R`(78) = **188**; `Walk_L_To_R`(35, ungroundable) excluded.
- Walk_F phase alignment per window: `full_state_align(Walk_F_state, turn_onset_frame)`
  (pose top-k → contact refine), the spec §7.1 rule.

## 3. (a) Root vs local residual

Each turn frame is matched to its **phase-optimal** Walk_F frame; residual = the local
difference Walk_F cannot represent at *any* phase. The **null yardstick** `WALKF_SELF_FLOOR` is
Walk_F matched to its own nearest non-adjacent frame (the manifold's own thickness). (A
phase-drift confound was ruled out: an onset-aligned-then-freerun Walk_F window gives the same
residual, 0.111 vs 0.110.)

| metric | Walk_F self-floor (null) | turn (ALL) | ratio |
|---|---|---|---|
| pose_d (post-match) | 0.021 | **0.109** | **5.2×** |
| rot6d local residual rms | 0.021 | **0.110** | **5.2×** |
| rot6d root-bone residual rms | 0.019 | 0.081 | 4.3× |
| bone_angvel total Δ rms (rad/s) | 0.426 | **0.664** | 1.56× |
| bone_angvel **root-explained** frac | 0.010 | **0.0085** | ~1% root |
| normalized root-explained frac† | — | **0.061** | ~6% root / 94% local |

Per clip: mild `Walk_L_To_L` is closest to Walk_F (pose_d 0.061); the `R_*` turns articulate
more (0.12–0.13). Within-window heading change is small (mean ≈ −0.035 rad, p95 ≈ 0.21 rad).

Findings:
- **`pose == Walk_F` (verbatim) is falsified.** The phase-optimal turn-pose residual is **~5×
  the Walk_F self-floor** and ~2× the groundability pose threshold (0.05). Turns articulate
  differently (lean/bank/foot placement) — but the residual is still *modest* in absolute pose
  terms, so Walk_F remains a strong backbone.
- The **`bone_angvel` 0.6–0.7 rad/s "regime delta"** is mostly Walk_F's *own* frame-to-frame
  angvel spread (turn 0.664 vs floor 0.426 → only 1.56× excess), and is **~99% in non-root
  (local) bones** — the *same* root/local split as the Walk_F null. **The regime difference is
  not a root phenomenon.**
- †The normalized "root ≈ 6%" energy figure is **dimensionality-confounded** (13 root dims vs
  405 local dims) — a low-dim root is *expected* to be low-energy/high-leverage, so this is not
  evidence that root is unimportant. The load-bearing, dimension-free results are the
  ratio-to-floor (local residual 5× the null) and (b).

## 4. (b) Decouple usability

Walk_F local cycle (phase-aligned at the turn onset, run coherently over the 16-frame horizon)
substituted into `rot6d`/`bone_angvel`, driven along the **real turn root path** (`ego_vel`,
`yaw_rate`, `cond_dir`, `root_pos[0]` from the turn), reconstructed to world `state281` through
the GT-identical path and scored against the GT-calibrated contract bands.

| family (ALL, n=188) | pass rate |
|---|---|
| acceptance (all families) | **0.112** |
| support_honesty / **foot_slip** | **1.000** ✅ |
| pose_continuity | **1.000** ✅ |
| regime_reached | 1.000 |
| rate_budget | 0.984 |
| command_response | 1.000 |
| **support_side_correctness** | **0.112** ❌ binding |

- foot_slip p95 mean 1.95 m/s, **to-band ratio 0.69 (< 1, within GT band)**.
- **164/188** windows fail **only** `support_side_correctness`; 21 fully pass.
- Binding features: **foot position relative to root** (`left_rel_x_mean` 70, `right_rel_x_mean`
  59, `right_rel_z_mean` 36, `*_rel_y_mean` …) and single-support claimed-speed ratios.

Finding: **foot-slip and pose-continuity decouple cleanly** — a coherent Walk_F gait driven
along the turn root path does not skate and stays continuous. What fails is **support-foot
placement relative to the turning root**: Walk_F's stance geometry is not the turn's. This is
the root↔gait-phase↔foot-placement coupling, the saga's perennial hard bone (support).

## 5. Decision-tree verdict

- (a) **observable local residual** (not near-invariant) → rules out the pure-root branch.
- (b) decouple **fails the contract**, bound by **support_side** (foot-slip and pose-continuity
  pass).

**Hit: branch 3 — "root + small local correction" — with a branch-2 mechanism.** The local
correction that matters is **support-foot placement coordinated with the root / gait phase**,
not a free 281-dim pose.

Reconciliation with the saga (consistent, not contradictory): the local articulation difference
is **large in energy but easy** (Stage1 fits pose to ~5e-10, "白送"); the root↔support coupling
is **small in energy but binding** (support_side / rootvel / heading are always the hard bone).
This measurement and the saga agree: the difficulty is the root–support coupling, not free pose
generation.

## 6. Recommended architecture

Collapse the 16×281 minimax trajectory generator to:

1. **Walk_F pose backbone** — local articulation anchored to the phase-aligned Walk_F cycle
   (cheap; it is ~5× the noise floor but still modest and easy to fit).
2. **Low-dim root path** — decoder outputs `{ego_vel/forward+lateral, yaw_rate, heading}` →
   integrated `root_pos`. This is the low-dimensional steering signal.
3. **Gait-phase-coordinated support-foot-placement correction** — a *small* local residual on
   top of Walk_F, concentrated on the support-foot `*_rel_*` geometry, so the correct foot is
   planted where the turning root needs it (the only family that fails the decouple).

This is ~an order of magnitude smaller than a free 281-dim generator: pose is anchored to
Walk_F, the decoder owns the low-dim root path plus the support-foot placement the turn demands.

## 7. Caveats / discipline

- Read-only; existing contract scoring + reconstruction reused verbatim; GT path = 1.0
  confirmed before interpreting any number.
- Did not reopen representation/conditioning (layer-2); this is dimension/decoupling, not
  anchored advantage.
- Used existing `directlocal` data; the construction matched "root-excluded" with one
  clarification (heading/root path lives in `cond_dir`+`root_pos`, reapplied at reconstruction)
  — reported in §1.
- The normalized root-energy fraction is dimensionality-confounded (§3†); conclusions rest on
  ratio-to-Walk_F-floor and on the path-identical (b) contract scoring.
