# Turn-Aware Posttrain Validation And New-Environment Timing Plan

Date: 2026-06-15

Status: EXECUTION DIRECTION / HANDOFF PLAN. This document is intended to be
committed and pushed with the current PR branch so a new environment can
continue from a stable written plan. It does not introduce production code.

## 1. Decision

Do not use a full basetrain to answer the next mechanism question first.

The next mechanism question is:

> Can the turn residual idea move from ignored harness artifacts into a
> reproducible, production-adjacent posttrain path without breaking Walk_F zero
> accumulation?

The recommended next training direction is:

1. Keep the current C' R3 anchor frozen.
2. Add turn clips to a turn-aware posttrain residual path.
3. Train only a fixed-gate SO(3) residual head, with recurrent residual as the
   primary candidate and per-frame residual as the baseline.
4. Keep Walk_F in the run only as a zero-accumulation/no-leak witness.
5. Run a separate full-flow timing pass in the new Windows environment, but
   treat it as an environment capacity and reproducibility audit, not as the
   mechanism verdict.

This means "turn enters posttrain" is the right direction, but "restart
basetrain and then run complete posttrain" is not the first validation step.

## 2. Why Full Basetrain Is Not The First Mechanism Test

Full basetrain would mix at least four variables:

- anchor representation quality,
- turn data curriculum,
- posttrain objective shape,
- free-run own-carry stability.

That makes failure hard to attribute. The current evidence says the interesting
object is narrower: a large but smooth correction on top of a frozen anchor.
The residual should be zero on Walk_F and active only under causal turn gate.

Therefore the first mechanism run should freeze the anchor and answer whether
posttrain can learn the turn-active residual under free rollout. Only if this
fails for representation reasons should turn be moved back into basetrain
curriculum.

## 3. Current Evidence To Preserve

The latest local harness artifact was intentionally ignored by Git:

- `debug_output/_tmp_gated_residual_recurrent_20260614_v1/`

If this directory is unavailable in the new environment, do not treat its
absence as a PR failure. Use the following numbers as the handoff hypothesis and
rerun the tracked harness or the new turn-aware posttrain equivalent.

Target correction witness:

| clip | target mag p95 deg | target frame-mean delta p95 deg/frame |
|---|---:|---:|
| Walk_L_To_L | 30.002 | 1.150 |
| Walk_L_To_R | 24.578 | 1.081 |
| Walk_R_To_L | 16.600 | 0.930 |
| Walk_R_To_R | 16.388 | 0.846 |

Phase A per-frame cap-up:

| cap | repair | free band | mean repair | mean free step-p95 | verdict |
|---:|---:|---:|---:|---:|---|
| 12 | 3/4 | 2/4 | 0.391 | 1.534 | partial |
| 16 | 3/4 | 3/4 | 0.366 | 1.489 | band-closed baseline |
| 24 | 3/4 | 0/4 | 0.393 | 1.624 | jitter |
| 30 | 3/4 | 0/4 | 0.405 | 1.813 | jitter |

Phase B recurrent at cap 30:

| metric | value |
|---|---:|
| repair hits | 4/4 |
| free band hits | 3/4 |
| mean repair | 0.714 |
| mean free step-p95 | 1.372 |
| same-cap step-p95 gain | 4/4 |
| same-cap gated-omega delta gain | 4/4 |
| Walk_F fmj abs diff vs anchor | 0.0 |
| Walk_F gate mean/p95 | 0.0 / 0.0 |
| Walk_F gated omega p95 | 0.0 |
| recurrent state stability | pass |

Interpretation: the current best hypothesis is not "more basetrain". It is
"posttrain needs an explicitly turn-active recurrent residual objective with
Walk_F zero-leak guardrails".

## 4. Refactor Recommendation

Before productionizing, refactor the ignored harness idea into a tracked,
reproducible runner. This should be treated as a bridge step, not as final
production implementation.

Recommended bridge shape:

- new tracked tool under `tools/`, for example
  `tools/run_action_handoff_turn_residual_posttrain_probe.py`;
- no production checkpoint schema change in the first bridge pass;
- import existing train helpers instead of copying SO(3), rollout, or history
  logic;
- output artifacts under
  `debug_output/20260615_turn_residual_posttrain_probe/`;
- write `preregistration.json`, `sweep_summary.json`, `closeout.md`, and
  `cache_recompute_selfcheck.json`;
- make the runner resume-safe and cache-readable so Windows can rerun the same
  experiment.

If the bridge reproduces the harness result, then productionization can move
into `train/posttrain.py` and model/config/checkpoint contracts with a smaller
diff.

## 5. Turn-Aware Posttrain Spec

### Owner Boundary

Expected owner placement when moving beyond the bridge:

- posttrain objective, training loop, artifact save:
  `train/posttrain.py`;
- model head definition only if a production residual head is required:
  `train/models.py`;
- rollout carry and free-run stepping:
  `train/rollout_kernel.py`;
- shared SO(3), rot6d, geodesic, and exp/log math:
  `train/geometry.py`;
- checkpoint shape/schema only:
  `train/checkpoint/load_schema.py` and related checkpoint modules.

Do not put SO(3), FK, rollout carry, or history state-machine logic directly
into a new entry script.

### Data And Roles

Use Walk_F plus turn clips with explicit roles:

- `Walk_F`: zero-accumulation witness only, eff-n=1 diagnostic.
- `Walk_L_To_L`, `Walk_L_To_R`, `Walk_R_To_L`, `Walk_R_To_R`: turn verdict,
  eff-n=4.
- `Walk_L_To_L`: keep single-column hard-clip reporting because it is the most
  cap-starved case.

### Model

Frozen anchor plus residual:

```text
R_out[t] = Exp(gate_target[t] * cap(omega_res[t])) @ R_anchor[t]
```

Residual candidates:

1. Per-frame residual baseline at cap 16 and cap 30.
2. Recurrent residual primary candidate at cap 30.

The fixed gate must remain causal and non-learned in the first production
candidate. It should come from command yaw-rate semantics, not teacher-forced
future pose.

Tensor contracts to keep explicit:

- `R_gt`, `R_anchor`, `R_out`: `[T,46,3,3]`, `float32`, CPU in cache, rollout
  device during training.
- `omega_res`, `gated_omega`: `[T,46,3]`, `float32`, rotvec in radians during
  training, reported in degrees.
- `gate_target`: `[T]` or `[B,1]`, `float32`, causal command-derived scalar.
- recurrent state: `[B,Hr]`, `float32`, reset at sequence boundary and cached
  as `[T,Hr]`.

### Loss Policy

Use teacher and free rollout columns, but the verdict is free-run first.

Losses should be gate-active:

- geodesic repair term on turn clips;
- gated delta/smoothness term weighted by `max(gate_t, gate_t-1)`;
- omega regularization on active frames;
- optional recurrent state norm/delta regularization only if free state
  stability fails.

Walk_F must remain zero because gate is zero. Walk_F free step-p95 is only an
anchor-inherited diagnostic and must not fail the residual verdict.

### Acceptance

Minimum acceptance for a candidate:

- Walk_F zero accumulation:
  - `R_out` free Walk_F fmj equals anchor within tolerance;
  - Walk_F gate mean/p95 is `0`;
  - Walk_F gated omega p95 is `0`.
- Turn repair:
  - free turn fmj reduction versus anchor is at least 30 percent on at least
    3/4 turn clips.
- Turn stability:
  - free step-p95 in `[0.5,1.5] * GT` on at least 3/4 turn clips; or, for an
    intermediate bridge result, out-of-band excess compressed by at least 50
    percent versus the same-cap per-frame baseline on at least 3/4 clips.
- Recurrence gain:
  - same-cap recurrent repair is not worse than per-frame;
  - same-cap recurrent free step-p95 and gated-omega delta-p95 improve on at
    least 3/4 clips.
- Recurrent state:
  - finite state norm and finite state delta on all free clips;
  - no monotonic drift or obvious sequence-length amplification.

Verdict labels:

- `TURN-RESIDUAL-CANDIDATE(band-closed)`;
- `FRONTIER-MOVED-NOT-CLOSED`;
- `RECURRENCE-NULL`;
- `RECURRENT-STATE-UNSTABLE`.

## 6. New Windows Environment Timing Plan

This lane answers a different question:

> How long does a clean environment need to run the project end to end?

It should not be used as the first turn-residual mechanism verdict.

Create a timing output directory:

```bash
mkdir -p debug_output/20260615_new_env_fullflow_timing
```

Record immutable context:

```bash
git rev-parse HEAD > debug_output/20260615_new_env_fullflow_timing/git_sha.txt
git status --short > debug_output/20260615_new_env_fullflow_timing/git_status.txt
python --version > debug_output/20260615_new_env_fullflow_timing/python_version.txt
python -m pip freeze > debug_output/20260615_new_env_fullflow_timing/pip_freeze.txt
```

On Windows PowerShell, wrap each command with `Measure-Command` and write the
elapsed seconds to a small text or JSON file. On Unix-like shells, use
`/usr/bin/time -p`.

Suggested timing stages:

| stage | purpose | command template | verdict role |
|---|---|---|---|
| env smoke | dependency/import sanity | `python -m pytest tests/train/test_geometry_shared_helpers.py` | required |
| rollout smoke | free carry sanity | `python -m pytest tests/train/test_rollout_kernel_free_carry.py` | required |
| config smoke | entry/config compatibility | `python -m pytest tests/train/test_training_mpl_entry_config_compat.py` | required |
| action-handoff smoke | new scaffold sanity | `python -m pytest tests/train/test_action_handoff_inbetween_commanded_yaw.py` | required if present |
| preprocess | raw to processed timing | `python -m train.convert_json_to_npz raw_data/Walk_F.json --out raw_data/processed_data` | timing only unless data missing |
| basetrain smoke | trainer can step/save | use the smallest current config-supported smoke run | timing and environment risk |
| full basetrain | clean full anchor timing | run current canonical basetrain config | timing and reproducibility |
| posttrain smoke | posttrain entry can load/freeze/save | `PYTHONPATH=. python -m train.posttrain --config <config_json> --ckpt_in <ckpt> --out_dir debug_output/20260615_new_env_fullflow_timing/posttrain_smoke --run_name posttrain_smoke` | timing and contract risk |
| turn residual bridge | mechanism rerun | tracked bridge runner once implemented | actual next mechanism verdict |

For Windows PowerShell:

```powershell
$t = Measure-Command { python -m pytest tests/train/test_geometry_shared_helpers.py }
$t.TotalSeconds | Out-File debug_output/20260615_new_env_fullflow_timing/pytest_geometry_seconds.txt
```

For Unix-like shells:

```bash
/usr/bin/time -p python -m pytest tests/train/test_geometry_shared_helpers.py \
  2> debug_output/20260615_new_env_fullflow_timing/pytest_geometry_time.txt
```

The full-flow timing report should include:

- wall time per stage;
- CPU/GPU/MPS/CUDA device;
- peak memory if available;
- checkpoint path and size;
- final basetrain metrics path;
- final posttrain metrics path;
- whether any stage used reduced epochs/steps.

## 7. Recommended Next Work Order

1. Commit this document to the current PR branch.
2. On Windows, pull the PR branch and run the timing smoke stages first.
3. Do not start full basetrain until the smoke stages pass.
4. Port the recurrent residual harness into a tracked bridge runner.
5. Run per-frame cap16/cap30 and recurrent cap30 from the frozen C' R3 anchor.
6. Only after the bridge reproduces Walk_F zero plus turn repair/stability,
   move the path into production posttrain code.
7. Run full basetrain/posttrain timing as a separate environment report.

## 8. Escalation Conditions

Escalate back to owner decision before productionizing if any of these happen:

- Walk_F gate or gated omega is nonzero.
- Walk_F residual free fmj differs from anchor beyond tolerance.
- Recurrent state free norm/delta is non-finite or grows with rollout length.
- Recurrent repair is worse than same-cap per-frame.
- Full basetrain changes the anchor behavior enough that the frozen-anchor
  residual conclusion no longer applies.
- Windows full-flow timing shows dependency or hardware behavior materially
  different from the current Mac run.

## 9. One-Line Handoff

The correct next validation is not "train everything longer"; it is a
turn-aware, frozen-anchor posttrain residual bridge with explicit Walk_F
zero-leak guards, plus a separate clean-environment full-flow timing report.
