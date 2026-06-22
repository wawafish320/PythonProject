> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Walk_F rollout-eval checkpoint selection — provenance memo (2026-05-24)

Status: provenance / audit trail only.  
Scope: this memo records how the first post-removal current-surface compatible local checkpoint surfaced for the Walk_F turn-cycle rollout-eval pilot. It is **not** a quality findings memo. It does **not** populate the `docs/aperiodic_transition/...rollout_eval_pilot_contract.md` §4 failure taxonomy and does **not** assert any performance verdict.

## 1. Hard block from the old surface

Before retrain, the only Walk_F-eligible local checkpoint set carried the legacy `contact_plan_init_head.*` parameter family. That family is now a removed surface fence and is treated as a `LEGACY_REMOVED_FIELD` fail-fast by `tools/select_walk_f_rollout_eval_checkpoint.py` (`LEGACY_REMOVED_MARKERS = ("[Removed]", "contact_plan_init_head")`, `tools/select_walk_f_rollout_eval_checkpoint.py:30-31`).

Removed-field policy: `docs/removal_policy.md` (do not modify; fail-fast on contact).

## 2. 1271 metadata gate numbers

The 1271-checkpoint metadata sweep (pre-retrain census of historical local ckpts considered against the current surface) decomposed as:

- 1157 — removed-field hits (legacy `contact_plan_init_head.*` and other removed parameters).
- 70 — teacher direct-pose blocker (teacher artifact write blocked under direct-pose surface).
- 36 — no usable `state_dict` payload at all.
- 5 — metadata survivors: cleared the metadata gate but failed downstream smoke.
- 3 — `load_error` (schema / shape mismatch raised during `load_event_motion_ckpt_payload`).

Total: 1157 + 70 + 36 + 5 + 3 = 1271. The 5 "survivors" are the load-bearing population for §3 below.

## 3. The 5 metadata-survivors pattern

Each of the 5 survivors shows the same fingerprint when re-driven through the current-surface smoke:

- `*_freerun_cycles.json` exists (free-run subprocess wrote its artifact),
- `*_teacher_pred.json` is missing,
- teacher stdout/stderr carries `[ERR]` / `[FATAL]` markers (now promoted to `PilotError`, see commit `2f5657d`).

This pattern is what `tools/select_walk_f_rollout_eval_checkpoint.py` would classify as `TEACHER_ARTIFACT_MISSING` with non-empty `teacher_fatal_marker_hits` — i.e. the checkpoint passed metadata gating but the current-surface teacher rollout could not complete.

## 4. Fresh basetrain checkpoint, first selection attempt: real root cause

After retrain produced:

`models/_current_surface_retrain_20260524/walk_f_rollout_eval_current_surface_20260524/ckpt_last_walk_f_rollout_eval_current_surface_20260524.pth`

the first selection-smoke pass against it still **FAIL**ed. The true root cause was **not** the checkpoint — it was a `train/validate/run_teacher_rollout.py` glue bug: the teacher CLI was not feeding `cond_raw_seq` into `Trainer._rollout_sequence` (`train/training_MPL.py:1194-1208`), so the rollout kernel received `cond_raw_seq=None` and the carry path collapsed.

This is a validation-glue bug, *not* a checkpoint compatibility bug.

## 5. Commit 81bc95e — validation glue fix, not a shim

`81bc95e fix(validate): pass raw cond into teacher rollout carry`

- Touches only `train/validate/run_teacher_rollout.py`.
- Does **not** modify the trainer, the checkpoint loader, the model build, the data schema, or any normalization template.
- Does **not** add a backwards-compat fallback for old-surface checkpoints.
- Does **not** mutate training semantics.
- Effect: teacher rollout now carries a raw condition tensor into the rollout kernel, matching the teacher batch contract emitted by `train/validate/export_teacher_batches.py:251-253` where `teacher.cond = npz.cond_in` (raw).

For audit purposes: `teacher_block.keys()` for `validate/teacher_batches/Walk_F_teacher.json` is `['cond', 'state_norm', 'target_norm']`. There is no `cond_raw` or `cond_tgt_raw` key in the current teacher JSON contract, so the carry source resolves to `teacher.cond_as_raw_carry` (see §7).

## 6. Selection-smoke comparability

Selection-smoke output produced **before** 81bc95e and **after** 81bc95e are not directly comparable: same checkpoint can move from `TEACHER_ARTIFACT_MISSING` to `PASS` purely because the validation-glue path changed, with zero change to the checkpoint payload. Treat any pre-81bc95e selection summary as a different artifact class and do not cross-compare with post-81bc95e summaries.

The post-fix selection summary on the fresh ckpt:

`debug_output/walk_f_rollout_eval_checkpoint_selection_20260524_retrain_after_teacher_fix/summary.json`

- candidate_count_scanned: 1
- pass_count: 1
- fail_count: 0
- hard_block_no_compatible_checkpoint: false

This is the first time, on this branch, that a local checkpoint passes the current-surface selection smoke. It is the first **post-removal / current-surface compatible** local checkpoint for Walk_F rollout-eval.

## 7. Explicit `cond_raw` source marker — selection-tool contract

Follow-up to 81bc95e makes the carry-source resolution explicit and auditable:

- `train/validate/run_teacher_rollout.py:resolve_cond_raw_carry` resolves the raw carry condition in declared priority `teacher.cond_raw → teacher.cond_tgt_raw → teacher.cond_as_raw_carry`, raises if none are present, and stamps the source into each per-clip artifact under `teacher_runner_contract.cond_raw_source` (resolver version `explicit_cond_raw_source_v1`).
- `tools/select_walk_f_rollout_eval_checkpoint.py` reads that field back from each candidate's teacher artifact and writes it into `result.json`, plus an aggregate top-level `summary.json` field set: `teacher_runner_contract_marker`, `selection_semantics_note`, `teacher_runner_cond_raw_source_counts`.
- Selection tool still **does not** compute performance metrics, **does not** read band artifacts, and **does not** classify against the §4 rollout-eval failure taxonomy.

## 8. 2-row paired smoke — execution gate only

`debug_output/_tmp_walk_f_turn_cycle_rollout_eval_compatible_ckpt_smoke/summary.json`:

- valid_rows = 2
- total_rows = 2
- rollout_failures_count = 0
- failure_taxonomy_verdict = `TRAINING_MECHANISM_FAIL.EXPOSURE_BIAS_DRIFT`

This is **execution-gate only**. A 2-row valid-rows verdict is not a full-baseline verdict for the §4 taxonomy and must not be treated as one. The verdict string is a hint emitted by the pilot runner against the tiny sample; whether it is the right full-population class is a question only the locked full baseline can answer.

## 9. What is next, and what is not next

Next: locked full Walk_F turn-cycle rollout-eval baseline against the fresh checkpoint, with the locked rollout config from the pilot contract.

**Not** next: any form of posttrain, any change to the trainer / checkpoint loader / model build, any modification to the teacher batch schema, or any expansion of selection-smoke scope into performance metrics. Those are outside the provenance objective.
