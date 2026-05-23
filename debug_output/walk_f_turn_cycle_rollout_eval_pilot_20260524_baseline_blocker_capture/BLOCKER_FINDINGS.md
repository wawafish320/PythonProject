# Baseline Blocker Capture Note (2026-05-24)

## Scope

- This note is an artifact-local blocker record for:
  - `debug_output/walk_f_turn_cycle_rollout_eval_pilot_20260524_baseline_blocker_capture`
- It is **not** a research findings memo and does **not** claim turn-cycle return behavior conclusions.

## Observed Blocker

- Baseline run is blocked by strict valid-row gate:
  - `summary.json`: `exit_status=2`
  - `summary.json`: `baseline_blocked_no_valid_paired_rows=true`
  - `summary.json`: `valid_rows=0`, `invalid_rows=12`

## Primary Evidence

- `run_manifest.json.rollout_failures[*]` shows rollout subprocesses with:
  - `returncode=0`
  - `artifact_missing=true`
  - empty expected artifact directory listing for teacher/free-run output dirs.
- Example (Walk_F, teacher stage) stdout includes fatal markers:
  - `[ERR] ... resolve_contact_plan_build_cfg ...`
  - `[Removed] checkpoint/config field contact_plan_init_head.* entered a retired strict branch ...`

## Infrastructure Conclusion (Only)

- Baseline is currently blocked because selected checkpoint surface contains retired `contact_plan_init_head.*` content rejected by current code path.
- This is an evaluator infrastructure compatibility blocker, not a turn-cycle mechanism conclusion.

## Next Action Boundary

- Unblock requires a post-removal (current-surface compatible) checkpoint selection.
- Do not interpret this blocker artifact as evidence for `EXPOSURE_BIAS_DRIFT` / `CAPACITY` / `OBJECTIVE_BLIND_TO_BAND`.
