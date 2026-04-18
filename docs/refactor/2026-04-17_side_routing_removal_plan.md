# [2026-04-17] Side-Routing Removal Plan

Date: 2026-04-17  
Status: Draft / Planned  
Owner: posttrain cleanup  
Scope: `train/models.py`, posttrain checkpoint contract, freerun validation, posttrain config surface, active config cleanup, documentation handoff.  
Goal: remove the retired `direct_pose_leg_side_*` / shared side-routing path as dead code, keep the current plain direct-pose leg path behavior unchanged, and prepare `EventMotionModel` for a later structural split.  
Non-goals: do not redesign direct-pose loss, do not change active experiment behavior, do not delete archive configs, do not split `EventMotionModel` in this cleanup round.

---

## 0. One-Page Decision

Side-routing is retired and should be removed from the live codepath.

The safe order is:

1. Freeze/reject new side-routing configs.
2. Prove canonical checkpoints do not contain side-routing weights.
3. Remove `models.py` side-routing construction and forward logic atomically.
4. Bump the posttrain checkpoint contract to v2 and make v1 failure explicit.
5. Clean posttrain/freerun/tools/config/docs surfaces.

Important adjustment from the initial roadmap: `models.py` construction removal and forward removal should be one atomic phase. If the `direct_pose_leg_head_shared` sentinel disappears while forward still references it, the model can fail with `AttributeError` even when active configs have `side_routing=false`.

---

## 1. Document Structure

This plan is the single source of truth. Execution evidence and final closure should be stored separately:

- Main plan: `docs/refactor/2026-04-17_side_routing_removal_plan.md`
- P0/P1 evidence: `docs/changes/2026-04-17_side_routing_removal_p0_scan.md`
- Final report: `docs/changes/2026-04-17_side_routing_removal_report.md`
- Inventory update: `docs/Problems/active/2026-03-07_trainbase_posttrain_unused_branch_inventory.md`
- Contract cross-link: `docs/refactor/posttrain_checkpoint_contract_reset_cleanup_plan.md`

The P0/P1 evidence doc should be created before model deletion. The final report should be created after smoke/freerun validation.

---

## 2. Current Truth Snapshot

### 2.1 Live side-routing surfaces

Primary removal targets:

- `train/models.py`
  - `EventMotionModel.__init__` side-routing args/attrs.
  - `_init_direct_pose_routing_metadata` side-routing metadata and side buffers.
  - `_build_direct_pose_modules` shared per-side leg heads and gate/sign-gate heads.
  - `forward` side-routing branch, side cue assembly, side embedding, rank-1/sign-gate path, per-side scatter.
  - `_apply_direct_pose_leg_side_plan_other_ablation`.
  - `_compute_direct_pose_leg_cross_leg_ablation` if it is only reachable from side-routing diagnostics.
- `train/posttrain.py`
  - config rejector/listing surface.
  - shared-leg-head gradient probe/reporting remnants.
- `train/validate/run_freerun_cycles.py`
  - shared-head hook/reporting remnants.
  - `direct_pose_leg_side_cue` phase-age dependency remnant.
- `tools/`
  - archaeology/probe scripts that still reference `direct_pose_leg_head_shared`.
  - helper scripts passing `direct_pose_leg_side_plan_other_ablate`.
- `config/`
  - active configs carrying default-false side-routing keys.

### 2.2 Already-converged surfaces in current checkout

These initial roadmap items appear already resolved in the current tree and should not be treated as pending implementation work:

- `train/model_ckpt_contract.py` currently has no `side_*` fields in `DirectPoseLegBuildConfig`.
- `resolve_direct_pose_leg_build_cfg` is not present.
- `train/posttrain.py` build-state copy/serialization no longer carries direct-pose leg `side_*` fields.
- `train/validate/run_freerun_cycles.py` does not currently expose a `direct_pose_leg_side_plan_other_ablate` CLI surface.

### 2.3 Output-key classification

Do not delete all direct-leg diagnostic keys blindly.

Remove as side-routing-only:

- `direct_leg_side_sign_gate`

Keep unless proven side-only in the current plain path:

- `direct_leg_omega`
- `direct_leg_omega_raw`
- `direct_leg_gate`
- `direct_leg_gate_logits`
- `direct_leg_scale`
- `direct_leg_scale_log`
- `direct_leg_scale_log_raw`

Reason: the plain direct-pose leg branch also emits several gate/scale/raw diagnostic keys, and `posttrain.py` uses `direct_leg_gate_logits` for gate supervision.

---

## 3. Blocking Gates

### P0. Canonical Checkpoint Scan

Goal: prove current canonical posttrain checkpoints contain no side-routing tensors.

Required prefixes:

- `direct_pose_leg_head_shared.`
- `direct_pose_leg_gate_head_shared.`
- `direct_pose_leg_side_sign_gate_head.`
- `direct_pose_leg_side_embed.`
- `direct_pose_leg_side_pos_r_tensor`
- `direct_pose_leg_side_pos_l_tensor`

Rules:

- Scan the canonical ckpt manifest, not only ad-hoc local glob results.
- Include `state_dict`, `model`, and whole-dict state-dict layouts.
- Treat unreadable checkpoint files as blocking until classified.
- Record exact command, scanned count, hit count, unreadable count, and hit filenames in `docs/changes/2026-04-17_side_routing_removal_p0_scan.md`.

Acceptance:

- `HIT_FILES=0` for canonical checkpoints.
- `ERROR_FILES=0` or each error is explicitly classified as non-canonical/irrelevant.

### P1. Active Config Scan

Goal: prove active configs do not enable side-routing.

Commands:

- `rg -n "direct_pose_leg_side" config/ -g '!config/archive*/**'`
- `rg -n --pcre2 '"direct_pose_leg_side_(routing|plan_other|phase_other|phase_rel|sign_gate|rank1)"\s*:\s*true|"direct_pose_leg_side_embed_dim"\s*:\s*[1-9][0-9]*|"direct_pose_leg_side_sign_gate_reg_weight"\s*:\s*(?!0(?:\.0+)?\b)[0-9.]+' config/ -g '!config/archive*/**'`

Acceptance:

- No active config has enabled/non-default side-routing keys.
- Archive configs are preserved as historical evidence.

### P2. Downstream Output-Key Scan

Goal: avoid deleting output keys used by losses/metrics/eval scripts.

Commands:

- `rg -n "direct_leg_side_sign_gate|direct_leg_gate_logits|direct_leg_scale_log_raw|direct_leg_omega_raw|direct_leg_gate|direct_leg_scale" train/ -g '!train/models.py'`
- `rg -n "direct_pose_leg_head_shared|direct_pose_leg_side_plan_other_ablate|direct_pose_leg_side_cue" train/ tools/`

Acceptance:

- `direct_leg_side_sign_gate` has no live downstream dependency.
- Shared-head instrumentation is removed or moved to retired tooling.
- Plain-path diagnostic keys are kept if downstream uses them.

---

## 4. Execution Phases

### E1. Freeze / Reject

Status: partially present.

Actions:

- Split side-routing checks out of `_cfg_reject_retired_direct_pose_highorder` into `_cfg_reject_side_routing`, or keep the current rejector but rename/scope the side-routing part clearly.
- Ensure the rejector is connected at `_cfg_from_payload`.
- Reject enabled/non-default side-routing fields:
  - `direct_pose_leg_side_routing`
  - `direct_pose_leg_side_plan_other`
  - `direct_pose_leg_side_phase_other`
  - `direct_pose_leg_side_phase_rel`
  - `direct_pose_leg_side_sign_gate`
  - `direct_pose_leg_side_rank1`
  - `direct_pose_leg_side_embed_dim > 0`
  - `direct_pose_leg_side_cue != none/off/disabled`
  - `direct_pose_leg_side_sign_gate_reg_weight > 0`

Acceptance:

- Active configs pass.
- Archived side-routing configs would be rejected if run, but archive configs are not edited or rerun.

### E2. Atomic `models.py` Removal

This phase combines the original construction and forward deletion.

Remove:

- `EventMotionModel.__init__` side-routing args and side attrs.
- Shared side-routing modules:
  - `direct_pose_leg_head_shared`
  - `direct_pose_leg_gate_head_shared`
  - `direct_pose_leg_side_sign_gate_head`
  - `direct_pose_leg_side_embed`
- Side metadata and buffers:
  - `direct_pose_leg_side_k`
  - `direct_pose_leg_side_pos_r`
  - `direct_pose_leg_side_pos_l`
  - `direct_pose_leg_side_pos_r_tensor`
  - `direct_pose_leg_side_pos_l_tensor`
- Side-routing `forward` branch.
- Side-only result key:
  - `direct_leg_side_sign_gate`

Keep:

- Plain `direct_pose_leg_head`.
- Plain direct-leg gate/scale/raw diagnostic outputs used by current training/eval.
- Current direct-pose leg SO(3) behavior for active configs.

Acceptance:

- `rg -n "direct_pose_leg_head_shared|direct_pose_leg_gate_head_shared|direct_pose_leg_side_" train/models.py` has no matches.
- `python3 -m py_compile train/models.py` passes.
- A minimal model import/build smoke passes.

### E3. Helper Removal

Remove side-routing-only helpers after E2:

- `_apply_direct_pose_leg_side_plan_other_ablation`
- `_compute_direct_pose_leg_cross_leg_ablation` if no non-side caller remains.

Acceptance:

- `rg -n "side_plan_other|cross_leg_ablate|direct_pose_leg_side" train/models.py` has no unintended matches.
- Import graph has no missing references.

### E4. Checkpoint Contract v2

Actions:

- Set `POSTTRAIN_CHECKPOINT_CONTRACT_VERSION = 2`.
- Keep strict loading for v2 checkpoints.
- Reject v1 checkpoints with an explicit migration error, not a generic unsupported-version message.
- Error text should mention that side-routing/shared leg-head state_dict compatibility has been retired.
- Do not add a silent v1-to-v2 adapter unless a canonical v1 checkpoint must remain loadable.

Acceptance:

- Current v2 save/load path works.
- v1 checkpoint load fails with a clear error.
- `rg -n "side_" train/model_ckpt_contract.py` remains 0 unless the only match is an intentional error message.

### E5. Posttrain Sync

Actions:

- Remove shared-head gradient probe preference.
- Remove `leg_head_shared` gradient monitor buckets.
- Keep plain `direct_pose_leg_head` monitoring.
- Keep `direct_leg_gate_logits` supervision if plain path still emits it.

Acceptance:

- `rg -n "direct_pose_leg_head_shared|direct_pose_leg_side_" train/posttrain.py` has no matches except the side-routing rejector if intentionally retained there.
- Posttrain smoke completes.

### E6. Freerun / Tools Sync

Actions:

- Remove `direct_pose_leg_head_shared` forward-hook/reporting text in freerun.
- Remove `direct_pose_leg_side_cue` phase-age dependency if side-routing is the only reason it exists.
- Remove or retire scripts that inspect shared side-routing heads.
- Remove `direct_pose_leg_side_plan_other_ablate` kwargs in tools.

Acceptance:

- `rg -n "direct_pose_leg_head_shared|direct_pose_leg_side_|side_plan_other_ablate" train/validate tools/` has no live-code matches.
- Freerun smoke completes.

### E7. Active Config Cleanup

Actions:

- Remove default side-routing keys from active configs.
- Do not edit archive configs.

Acceptance:

- `rg -n "direct_pose_leg_side" config/ -g '!config/archive*/**'` has no matches.
- Archive configs still preserve old experiments.

### E8. Documentation Closure

Actions:

- Add P0/P1 scan evidence doc.
- Add final cleanup report.
- Update unused-branch inventory status to closed.
- Add cross-link from checkpoint contract cleanup plan to this side-routing plan/report.

Acceptance:

- Final report includes grep evidence, smoke commands, and before/after behavior statement.
- The final report explicitly states whether loss/metric comparison is no-op.

---

## 5. Acceptance Matrix

| Area | Check | Required result |
| --- | --- | --- |
| Checkpoints | canonical prefix scan | 0 side-routing tensor hits |
| Active config | `direct_pose_leg_side` scan excluding archive | 0 matches after E7 |
| `models.py` | shared/side-routing grep | 0 matches |
| Contract | version constant | `POSTTRAIN_CHECKPOINT_CONTRACT_VERSION = 2` |
| Contract | v1 load | explicit failure message |
| Posttrain | smoke | passes |
| Freerun | smoke | passes |
| Behavior | active configs | no expected loss/metric change |
| Docs | final report | grep + smoke + comparison evidence recorded |

---

## 6. Risk Register

### R1. Canonical checkpoint invalidation

Risk: a canonical checkpoint still contains side-routing weights.

Mitigation:

- P0 is blocking.
- If hits exist, either regenerate those checkpoints under the current contract or explicitly retire them before E2.

### R2. Output-key contract breakage

Risk: downstream code reads a key incorrectly classified as side-only.

Mitigation:

- P2 scan before E2.
- Only remove `direct_leg_side_sign_gate` by default.
- Preserve plain-path gate/scale/raw keys unless proven dead.

### R3. Tooling archaeology noise

Risk: final grep keeps failing because old probe scripts reference shared heads.

Mitigation:

- Decide per script: update, move to `docs/retired_directions` evidence, or delete if one-off.
- Include `tools/` in E6, not as an afterthought.

### R4. Contract version blast radius

Risk: all v1 checkpoint loads fail after bump.

Mitigation:

- Make failure explicit and actionable.
- Record affected checkpoint paths from P0/P1 evidence.
- Do not silently ignore state_dict structure changes.

### R5. EventMotionModel split interference

Risk: file splitting while side-routing glue still exists creates noisy cross-file imports.

Mitigation:

- Keep EventMotionModel split out of scope.
- Revisit split only after E1-E8 are complete and smoke-tested.

---

## 7. Rollback Policy

Preferred rollback is commit-level, not compatibility shims.

If E2 breaks active training:

1. Revert the atomic `models.py` removal commit.
2. Keep P0/P1 evidence docs.
3. Fix classification of the missed dependency.
4. Re-attempt E2 with a narrower deletion boundary.

Do not reintroduce side-routing config support after E1 unless a canonical checkpoint or active experiment is explicitly reclassified as live.

---

## 8. Deferred Follow-Up: EventMotionModel Split

After side-routing removal is closed, open a separate plan for structural splitting:

- `train/models_event_motion.py`: `EventMotionModel` shell.
- `train/models_event_encoder.py`: encoder, FiLM, period/contact-event clock pieces.
- `train/models_direct_pose.py`: direct-pose module construction and forward helpers.
- `train/models_plan_corrector.py`: plan-z correction path.

This must be a new roadmap with its own baseline metrics. It should not share acceptance criteria with side-routing deletion.
