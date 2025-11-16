## Adaptive Loss & Scheduler Plan

Goal: build on the cleaned config pipeline so the trainer can react to metrics
inside the run (online loss re-weighting, TF/Free-run adjustment) without
breaking the current stage schedule / CLI flow.

### Phase 1 – Loss Reporting (minimal)
1. **MotionJointLoss**: add a `return_detailed_losses` flag. When enabled,
   `forward()` returns `(total_loss, {"fk_pos": raw_fk, ...})` where each value
   is the *unweighted* component (divide by `w_*` when appropriate).
2. **Trainer wiring**: expose a CLI flag `--adaptive_loss {dwa,uncertainty}`.
   - Default `None` leaves behavior unchanged.
   - When set, toggle `loss_fn.return_detailed_losses = True` and instantiate
     `AdaptiveLossWeighting`.

### Phase 2 – AdaptiveLossWeighting (dependency‑free)
Implementation file: `train/adaptive_loss.py`
- Support methods that don’t touch autograd graphs:
  - `dwa`: Dynamic Weight Averaging with 2-epoch history + temperature.
  - `uncertainty`: learnable `log_vars` (few parameters).
- Each call receives raw losses, outputs `total_loss`, `final_weights`, and
  diagnostics. Base weights from config should be applied *before* calling the
  adaptive module; AdaptiveLoss only produces a scale `±Δ` (clamped via
  `adjustment_range`, e.g. ±30%).
- Log every N steps so we can monitor stability before enabling by default.

### Phase 3 – Adaptive Scheduler (micro adjustments)
Implementation file: `train/adaptive_scheduler.py`
- Inputs: stage schedule’s base parameters (`freerun_horizon`, `freerun_weight`,
  current TF ratio). Scheduler stores `base_values` per epoch and emits a small
  delta (e.g. ±30%).
- Data sources:
  - Batch-level: loss + grad norm (for quick reaction).
  - Epoch-level: `trainer.metric_history` (new cache) to detect trends
    (YawAbsDeg up, loss trending, etc.).
- CLI flags:
  ```
  --adaptive_scheduler
  --adaptive_adjustment_rate 0.1
  --adaptive_update_interval 50
  --adaptive_use_epoch_metrics (default True)
  ```
- Application order inside training loop:
  1. `_apply_stage_schedule(ep)` → base values.
  2. Scheduler `set_base_values(...)`.
  3. After each batch, call `step_batch(loss, grad_norm)`; recompute
     `freerun_horizon` etc. as `base + delta`.
  4. After eval, push `trainer.latest_epoch_metrics('valfree')` into scheduler
     via `step_epoch`.

### Phase 4 – Testing & Safety
1. Start with dry-run mode inside AdaptiveLoss/Scheduler (compute weights/deltas
   but don’t apply) to verify signals/stability.
2. Add logging hooks so adjustments go into `train/logs/*.txt` for inspection.
3. Document the new flags in `docs/CONFIG_WORKFLOW.md` once stable.

### Long‑term
- GradNorm support (requires careful autograd usage).
- Persist scheduler state per run (append to `train/bayes_history.json`?).
- Surface adaptive stats back into `train_configurator` for future tuning.
