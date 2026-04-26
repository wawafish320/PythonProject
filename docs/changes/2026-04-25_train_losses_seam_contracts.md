# [2026-04-25] `train/losses.py` seam contracts

Date: 2026-04-25
Scope: `train/losses.py` seam retention only.
Non-goal: no new helper extraction, no further method-count reduction, no loss formula change.

---

## Intentional Seams

- `MotionJointLoss._invalidate_weight_cache`
  - Kept as an explicit cache-reset seam for tests and cache lifecycle control.
- `MotionJointLoss._collect_limb_local_stats`
  - Kept as a diagnostics seam; `train/diagnostics.py` resolves it reflectively via `getattr(trainer.loss_fn, "_collect_limb_local_stats", None)`.
- `MotionJointLoss._compute_direct_pose_group_base_payload`
  - Kept as the shared direct-pose base-payload seam used by `train/posttrain.py`.
- `MotionJointLoss._compute_direct_pose_group_norm_shared`
  - Kept as the public tuple-return compatibility seam; core logic stays in `_compute_direct_pose_group_norm_result(...)`.

## Current Contract

- Group-norm helper stack is intentionally `shared -> result`, not a single public `request`-typed entrypoint.
- Diagnostics and posttrain continue to depend on these seams as external integration points even when `train/losses.py` has no internal callers for some of them.
- Future cleanup should not classify the four methods above as dead code without first removing the dependent diagnostics/posttrain/test contracts.

## Regression Coverage

- `tests/train/test_train_models_failfast.py`
  - Locks cache invalidation semantics for `_invalidate_weight_cache`.
- `tests/train/test_diagnostics_runtime_config.py`
  - Locks reflective diagnostics access to `_collect_limb_local_stats`.
- `tests/train/test_posttrain_direct_group_norm_phase5.py`
  - Locks runtime delegation through `_compute_direct_pose_group_base_payload` and `_compute_direct_pose_group_norm_shared`.
- `tests/train/test_posttrain_finalize_shared_helpers.py`
  - Locks AST-level delegation in `_finalize_direct_group_norm`.
