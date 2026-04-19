# [2026-04-18] Posttrain compat cleanup gate

Date: 2026-04-18  
Status: Draft / audit-backed execution gate  
Owner: train refactor cleanup  
Scope: checkpoint compat, posttrain rebuild, runtime attach, caller-local overlays, live trainer attrs  
Related:

- `docs/refactor/2026-04-18_posttrain_training_mpl_zone_map.md`
- `docs/refactor/2026-04-18_runtime_attach_api_draft.md`
- `docs/refactor/2026-04-18_posttrain_training_mpl_commonization_execution_plan.md`
- `docs/refactor/2026-04-18_train_folder_relayout_execution_plan.md`

---

## 0. One-page conclusion

This gate turns the current compat audit into an executable cleanup policy.

Current recommendation:

- Do **not** start broad compat deletion now.
- Treat current state as **A+**:
  - audit-only documentation;
  - plus a repeatable posttrain execution gate;
  - plus explicit cleanup eligibility rules.
- Any future cleanup must pass this gate before deletion and again after deletion.

The important distinction:

- `compat inventory` answers what still exists.
- `compat cleanup gate` answers when it is safe to remove one surface.

At the current refactor boundary, most remaining compat surfaces still protect one of:

- checkpoint load / save behavior;
- posttrain checkpoint rebuild;
- `runtime_attach` shared-half behavior;
- caller-local overlay ownership;
- `trainbase_*` / `posttrain_*` live trainer attrs;
- eval/export tools that still load historical checkpoints.

Therefore the default cleanup decision is `defer` unless a candidate passes the eligibility rules below.

---

## 1. Hard boundaries

This gate inherits the current refactor boundaries.

Do not clean a compat surface if the cleanup requires any of:

- changing checkpoint top-level keys;
- changing `posttrain_cfg` round-trip behavior;
- changing `checkpoint_contract` semantics;
- renaming `trainbase_*` or `posttrain_*` live trainer attrs;
- moving caller-local overlay ownership into `train/runtime_attach.py`;
- touching `train/models.py`;
- changing train/posttrain/eval runtime semantics.

Immediate stop-rule:

> If a cleanup candidate touches ckpt contract, live attrs, or overlay owner drift, stop and re-run the audit instead of deleting.

---

## 2. Current compat inventory at gate granularity

| Surface | Owner | Current readers | Coverage | Cleanup status |
|---|---|---|---|---|
| EventMotion checkpoint load/build compat | `train/checkpoint/compat.py` | `training_MPL`, `posttrain_build_shell`, `export_onnx_from_ckpt`, `run_teacher_rollout`, `run_freerun_cycles` | partial unit + downstream smoke/tool coverage | `must-keep-now` |
| Direct-pose split / stepc tensor upgrade | `train/checkpoint/compat.py` | `EventMotionModel.load_state_dict`, train resume, posttrain/eval loaders | `tests/train/test_event_motion_model_refactor_phase_d.py` | `must-keep-now` |
| Posttrain checkpoint rebuild shell | `train/posttrain_build_shell.py` | `train/posttrain.py`, tools via direct `train.posttrain_build_shell` imports | `tools/run_posttrain_build_shell_smoke.py` | `must-keep-now` |
| Posttrain save-side `{"model","posttrain_cfg"}` payload | `train/posttrain.py` | posttrain reload, eval/export tools, historical tools | build-shell smoke top-level key checks | `must-keep-now` |
| Contract v2 / retired-v1 guard | `train/checkpoint/contract.py` | currently mostly dormant except public normalizers/tests | mode-normalization unit test | `defer` |
| Shared runtime attach layer | `train/runtime_attach.py` | basetrain and posttrain builders | pose-history tests + entry-shell smoke | `must-keep-now` |
| Basetrain caller-local overlay | `train/training_MPL.py` | basetrain trainer construction | entry-shell smoke | `defer` |
| Posttrain caller-local overlay | `train/posttrain.py` | posttrain trainer construction and objective runtime | indirect only | `defer` |
| `trainbase_*` / `posttrain_*` live attr bridge | `train/training_MPL.py` | `Trainer._predict_pretrain_contacts_from_frozen` | indirect runtime coverage | `must-keep-now` |
| Old `train.posttrain` helper namespace | `train/posttrain.py` entry/runtime helpers | tools/tests still import `_cfg_from_payload`, `_build_dataset_and_loader`, `_save_posttrain_outputs`, and rollout/runtime helpers; build-shell re-export subset already removed | mixed tool/runtime coverage | `defer` |
| Eval-only retired/legacy adapters | `train/validate/run_teacher_rollout.py`, `train/validate/run_freerun_cycles.py` | validation / whitebox runners | limited targeted tests | `defer` |
| Package marker | `train/checkpoint/__init__.py` | no direct callers found | none needed | `likely-removable`, but zero-value |

No high-value, definitely-dead shim currently qualifies for immediate removal.

---

## 3. Cleanup eligibility rules

A compat candidate is eligible for deletion only if **all** conditions are true:

1. `rg` shows no real caller in active `train/`, `tools/`, or `tests/` code.
2. It does not participate in checkpoint load, save, rebuild, or round-trip.
3. It does not write or read `trainbase_*` / `posttrain_*` live attrs.
4. It does not change `runtime_attach` owner boundaries.
5. It does not change model constructor kwargs or `train/models.py`.
6. It has a clearly identified gate that will fail if the deletion is wrong.
7. The diff is small and local.

If any answer is uncertain, classify as `defer`, not `likely-removable`.

Recommended candidate states:

- `must-keep-now`: actively protects runtime or checkpoint behavior.
- `defer`: may be removable later, but current owner/caller/gate is not stable enough.
- `likely-removable`: no real caller, no contract effect, gate exists, small diff.

---

## 4. Required pre-cleanup scan

Before any compat cleanup branch, run a static scan and paste the result into the cleanup note.

Minimum scan:

```bash
rg -n "compat|legacy|shim|re-export|overlay|contract|trainbase_|posttrain_" train tests tools
rg -n "_build_posttrain_model_from_ckpt\\(|_resolve_posttrain_model_build_state\\(|_load_posttrain_checkpoint_into_model\\(" train tools
rg -n "from train\\.checkpoint\\.(compat|contract)|from \\.checkpoint\\.(compat|contract)" train tests tools
rg -n "from train\\.checkpoint import|import train\\.checkpoint\\b" train tests tools
```

Expected interpretation:

- Any hit in checkpoint load/save/rebuild means `defer` unless the change is purely documentation.
- Any hit in `trainbase_*` / `posttrain_*` live attrs means `defer`.
- Any hit in tools is not automatically dead; first decide whether the tool is active, retired, or archive-only.

---

## 5. Required execution gates

### 5.1 Syntax gate

Run after any doc-adjacent import rewiring or compat deletion:

```bash
python3 -m py_compile \
  train/checkpoint/compat.py \
  train/checkpoint/contract.py \
  train/runtime_attach.py \
  train/posttrain_build_shell.py \
  train/posttrain.py \
  train/training_MPL.py
```

### 5.2 Checkpoint rebuild gate

Run before and after any cleanup touching checkpoint load/build code:

```bash
python3 tools/run_posttrain_build_shell_smoke.py
```

This gate protects:

- checkpoint top-level key expectations;
- checkpoint-derived build-state resolution;
- direct/lambda model instantiation;
- frozen encoder bundle attach;
- direct-pose and lambda-head presence after load.

If this fails, revert the cleanup and reclassify the candidate as `must-keep-now`.

### 5.3 Runtime attach gate

Run before and after any cleanup touching runtime attach, overlay, or live attrs:

```bash
python3 tools/run_training_mpl_entry_shell_smoke.py
```

This gate protects:

- dataset runtime attach;
- shared pose-history runtime attach;
- `trainbase_contacts_pretrain_affine_stats_spec` raw-spec mapping;
- loss/trainer runtime synchronization.

For posttrain-side overlay cleanup, this gate is necessary but not sufficient; also run the checkpoint rebuild gate and a real posttrain stop-before-training smoke if available.

### 5.3b Posttrain overlay smoke

Run before and after any cleanup touching posttrain trainer build, posttrain local overlay, or shared-half runtime attach consumed by posttrain:

```bash
python3 tools/run_posttrain_runtime_overlay_smoke.py
```

This gate protects:

- shared pose-history runtime attach on posttrain `Trainer`;
- posttrain-local `posttrain_contacts_pretrain_clamp` mapping;
- posttrain-local `posttrain_contacts_pretrain_affine_stats_spec` raw-spec mapping;
- posttrain-local parsed `posttrain_contacts_pretrain_affine` payload;
- representative posttrain overlay fields such as contact measurement mode and lambda reliability mode.

### 5.4 Unit gate

Run the focused unit suite for compat and pose-history surfaces:

```bash
python3 -m unittest \
  tests.train.test_event_motion_model_refactor_phase_d \
  tests.train.test_event_motion_model_phase_d_mode_normalization \
  tests.train.test_training_mpl_pose_history_phase2 \
  tests.train.test_posttrain_pose_history_phase3 \
  tests.train.test_run_freerun_cycles_pose_history_phase4
```

This gate protects:

- direct-pose split upgrade;
- direct-pose stepc terminal upgrade;
- public mode normalizers;
- shared pose-history semantics across basetrain, posttrain, and eval.

### 5.5 Real posttrain run gate

For any cleanup that touches posttrain rebuild, runtime overlay, or checkpoint compatibility, run at least one real posttrain command far enough to prove:

1. config parses into `PostTrainConfig`;
2. dataset and norm spec build;
3. checkpoint-derived model build succeeds;
4. `Trainer` builds and receives shared-half attach;
5. posttrain-local overlay attrs exist;
6. a checkpoint saved by posttrain can be loaded by the same build path.

Record:

- input ckpt path;
- input ckpt top-level keys;
- output ckpt top-level keys;
- build-state summary;
- whether `posttrain_contacts_pretrain_*` attrs exist;
- whether `pose_hist_mu` / `pose_hist_std` are attached.

This is the gate that decides whether a cleanup is truly safe for posttrain, not just import-clean.

---

## 6. Decision log template

For each cleanup candidate, add a short decision row before changing code:

| Candidate | Owner | Proposed action | Pre-scan callers | Gate required | Decision | Reason |
|---|---|---|---|---|---|---|
| Example | `train/foo.py` | delete alias | `0` active callers | syntax + focused unit | `likely-removable` | no runtime/ckpt/live-attr contract |

Decision rules:

- `must-keep-now`: record why it protects runtime or ckpt behavior.
- `defer`: record what must become true before cleanup.
- `likely-removable`: record the exact gate that will catch a bad deletion.

---

## 7. Minimal cleanup order for later

If cleanup happens after this gate is in use, use this order:

1. Rewire active tools away from old `train.posttrain` helper namespace.
2. Delete only tool-path shims with zero active callers.
3. Clean eval-only retired adapters after validation runners have replacement coverage.
4. Revisit caller-local overlays only after overlay ownership is explicitly redesigned.
5. Revisit `trainbase_*` / `posttrain_*` bridge only after live attr names are frozen or intentionally migrated.
6. Revisit checkpoint compat only after repeated posttrain round-trip gates pass on current and historical fixtures.
7. Revisit `train/checkpoint/contract.py` only after deciding whether contract-v2 is active, archived, or intentionally superseded.

Do not skip directly to checkpoint compat deletion.

---

## 8. Stop-rules

Stop the cleanup immediately if any of these happen:

- `tools/run_posttrain_build_shell_smoke.py` fails;
- a saved posttrain ckpt changes top-level keys;
- `posttrain_cfg` disappears or changes schema unexpectedly;
- `trainbase_*` / `posttrain_*` attrs are renamed or removed;
- runtime attach behavior changes from shared-half + caller-local overlay;
- cleanup requires editing `train/models.py`;
- eval/export ckpt loaders lose the ability to load current runbook checkpoints;
- a cleanup diff becomes broad enough that it is no longer a local shim deletion.

When a stop-rule triggers:

1. revert the cleanup;
2. mark the candidate `defer`;
3. update this gate with the missing test/smoke needed before retrying.

---

## 9. Baseline gate run — 2026-04-18 16:47

Purpose:

- Validate the current canonical runbook as the posttrain acceptance gate before any compat cleanup.
- Scope was posttrain sub-chain only: `stage6 -> 70a -> warmstart copy -> replace -> 70R -> 71 -> 72 -> lambda`.
- Fresh basetrain was not rerun because a fresh donor existed and was readable.

Run artifacts:

- Gate root: `debug_output/_tmp_posttrain_baseline_gate_20260418_164700`
- Model root: `models/__tmp_posttrain_baseline_gate_20260418_164700`
- Donor ckpt: `models/__tmp_tail_top7_fresh_chain_20260413_195656/basetrain/fresh_tail_top7_basetrain_20260413_195656/ckpt_last_fresh_tail_top7_basetrain_20260413_195656.pth`
- Donor top-level keys: `["model", "config"]`
- Structured summary: `debug_output/_tmp_posttrain_baseline_gate_20260418_164700/posttrain_baseline_gate_summary.json`

Lightweight gate result:

| Gate | Result | Notes |
|---|---|---|
| `py_compile` | pass | `train/checkpoint/compat.py`, `train/checkpoint/contract.py`, `train/runtime_attach.py`, `train/posttrain_build_shell.py`, `train/posttrain.py`, `train/training_MPL.py` |
| `tools/run_posttrain_build_shell_smoke.py` | pass | direct fixture keys `["config", "model"]`; lambda fixture keys `["model", "posttrain_cfg"]`; frozen encoder attached |
| `tools/run_training_mpl_entry_shell_smoke.py` | pass | shared runtime attach present; `pose_hist_mu/std` attached; `trainbase_contacts_pretrain_*` overlay present |
| focused unittest | pass | 17 tests: compat / mode normalization / pose-history surfaces |

Runbook Step 6 result:

| Stage | Input ckpt | Output ckpt | Result | Failure point | Contract exercised |
|---|---|---|---|---|---|
| `stage6` | donor ckpt above | `models/__tmp_posttrain_baseline_gate_20260418_164700/stage6_stepc_handoff/ckpt_last_stage6_tailfix_top7_stepc_clean_fromfresh_20260418_164700.pth` | fail | `train.posttrain` argparse rejects runbook CLI arg `--posttrain_contacts_source` | no; failed before config/runtime/checkpoint load |
| `70a` | stage6 output | `models/__tmp_posttrain_baseline_gate_20260418_164700/70a_clean/ckpt_last_WalkF_stage7_70a_from_fresh_tailk7_stage6stepc_clean_20260418_164700.pth` | not run | blocked by `stage6` | no |
| `warmstart copy` | 70a output | `models/__tmp_posttrain_baseline_gate_20260418_164700/warmstart_clean/ckpt_last_fresh_tail_top7_70a_replace_zerophase_cleanstepc_20260418_164700.pth` | not run | blocked by `stage6` | no |
| `replace` | warmstart output | `models/__tmp_posttrain_baseline_gate_20260418_164700/replace_clean/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_from_fresh_tailk7_70a_cleanstepc_20260418_164700.pth` | not run | blocked by `stage6` | no |
| `70R` | replace output | `models/__tmp_posttrain_baseline_gate_20260418_164700/70R_clean/ckpt_last_WalkF_stage7_70R_from_fresh_tailk7_replace_cleanstepc_s180_20260418_164700.pth` | not run | blocked by `stage6`; static inspection also shows likely runner/interface drift | no |
| `71` | 70R output | `models/__tmp_posttrain_baseline_gate_20260418_164700/71_clean/ckpt_last_WalkF_stage7_71_from_fresh_70R_cleanstepc_lr3e4_20260418_164700.pth` | not run | blocked by `stage6` | no |
| `72` | 71 output | `models/__tmp_posttrain_baseline_gate_20260418_164700/72_clean/ckpt_last_WalkF_stage7_72_from_fresh_71_cleanstepc_lr1e4_20260418_164700.pth` | not run | blocked by `stage6` | no |
| `lambda` | 72 output | `models/__tmp_posttrain_baseline_gate_20260418_164700/lambda_clean/ckpt_last_WalkF_stage7_lambda_from_fresh_72_cleanstepc_20260418_164700.pth` | not run | blocked by `stage6` | no |

Classification:

- Primary blocker: runbook CLI drift / path drift.
- The failure is not a checkpoint compat failure: checkpoint load was never reached.
- The failure is not a runtime attach / overlay failure: trainer construction was never reached.
- The failure is not a missing donor issue: donor ckpt exists and has `["model", "config"]` top-level keys.
- Static 70R coupling risk remains: `tools/run_posttrain_nonleg_trunk_ablation.py` still unpacks `_build_posttrain_model_from_ckpt(...)` as a tuple and calls `_save_posttrain_outputs(...)` with the pre-artifacts keyword protocol, while current `train.posttrain` exposes `PostTrainModelArtifacts`.

Cleanup readiness:

- Do **not** start compat cleanup from this state.
- First fix or clarify the canonical runbook command surface, especially the obsolete `--posttrain_contacts_source` CLI argument.
- After runbook Step 6 reaches `70R`, separately verify whether the current `70R` runner protocol is still compatible with `train.posttrain`.

Recommended next step:

- **A. First fix runbook / path drift.**

---

## 10. Compat-focused rerun — 2026-04-18 17:03

Purpose:

- Rerun after the runbook / `70R` runner fix.
- Scope intentionally narrowed to compat-layer detection, with cutoff at `70R`.
- `71 -> 72 -> lambda` were not required for this compat-only pass.

Run artifacts:

- Gate root: `debug_output/_tmp_posttrain_baseline_gate_rerun_20260418_170306`
- Model root: `models/__tmp_posttrain_baseline_gate_rerun_20260418_170306`
- Structured summary: `debug_output/_tmp_posttrain_baseline_gate_rerun_20260418_170306/posttrain_baseline_gate_rerun_summary.json`

Lightweight gate result:

| Gate | Result |
|---|---|
| `py_compile` | pass |
| `tools/run_posttrain_build_shell_smoke.py` | pass |
| `tools/run_training_mpl_entry_shell_smoke.py` | pass |
| focused unittest | pass |

Compat-focused stage result:

| Stage | Input ckpt | Output ckpt | Result | Contract exercised |
|---|---|---|---|---|
| `stage6` | fresh donor `ckpt_last_fresh_tail_top7_basetrain_20260413_195656.pth` | `models/__tmp_posttrain_baseline_gate_rerun_20260418_170306/stage6_stepc_handoff/ckpt_last_stage6_tailfix_top7_stepc_clean_fromfresh_20260418_170306.pth` | pass | `train.posttrain` ckpt load/build/runtime attach/save |
| `70a` | stage6 output | `models/__tmp_posttrain_baseline_gate_rerun_20260418_170306/70a_clean/ckpt_last_WalkF_stage7_70a_from_fresh_tailk7_stage6stepc_clean_20260418_170306.pth` | pass | posttrain ckpt reload and save-side `posttrain_cfg` |
| `warmstart copy` | 70a output | `models/__tmp_posttrain_baseline_gate_rerun_20260418_170306/warmstart_clean/ckpt_last_fresh_tail_top7_70a_replace_zerophase_cleanstepc_20260418_170306.pth` | pass | top-level key preservation by copy |
| `replace` | warmstart output | `models/__tmp_posttrain_baseline_gate_rerun_20260418_170306/replace_clean/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_from_fresh_tailk7_70a_cleanstepc_20260418_170306.pth` | pass | phase-z / direct-pose compat adaptation and save |
| `70R` | replace output | `models/__tmp_posttrain_baseline_gate_rerun_20260418_170306/70R_clean/ckpt_last_WalkF_stage7_70R_from_fresh_tailk7_replace_cleanstepc_s180_20260418_170306.pth` | pass | direct `tools/run_posttrain_nonleg_trunk_ablation.py` coupling to current `train.posttrain` artifacts API |
| `71` | 70R output | not required | intentionally interrupted | not part of compat-focused cutoff |

Checkpoint key findings:

- Donor ckpt keys: `["model", "config"]`.
- All posttrain outputs from `stage6`, `70a`, `warmstart copy`, `replace`, and `70R` have top-level keys `["model", "posttrain_cfg"]`.
- `posttrain_contacts_source` is no longer serialized in produced `posttrain_cfg`; clamp and affine stats remain present.
- `70R` saved both step ckpts and final `ckpt_last`, confirming direct runner save coupling works without the old shim.

Contract findings:

- checkpoint load/save compat: pass through donor -> stage6 -> 70a -> warmstart -> replace -> 70R.
- posttrain build shell: pass; both smoke and real chain use the current `PostTrainModelArtifacts` path.
- runtime attach shared-half: pass by entry-shell smoke and real posttrain stages reaching training.
- caller-local overlay: pass by real posttrain stages resolving pretrain clamp / affine overlay.
- `trainbase_*` / `posttrain_*` live attrs: no rename drift observed by gates.
- `70R` runner coupling: pass; direct runner executed current artifacts API and saved `ckpt_last`.

Cleanup readiness:

- Broad compat cleanup is still **not** appropriate because checkpoint/runtime surfaces are active contract surfaces.
- Very small dead-shim cleanup is now reasonable if the candidate is zero-caller and does not touch checkpoint keys, `posttrain_cfg`, runtime attach, live attrs, or `train/models.py`.

Recommended next step:

- **D. Can start very small dead-shim cleanup.**

---

## 11. Very small dead-shim scan — 2026-04-18

Scope:

- Static caller scan only across active `train/`, `tools/`, and `tests/`, plus this gate note and the fresh-chain runbook.
- No runtime/checkpoint cleanup patch was applied in this round.

Scan findings:

- `train/checkpoint/compat.py` remains an active contract surface:
  - callers exist in `train/posttrain_build_shell.py`, `train/export_onnx_from_ckpt.py`, `train/validate/run_teacher_rollout.py`, `train/validate/run_freerun_cycles.py`, and `tests/train/test_event_motion_model_refactor_phase_d.py`.
- `train/checkpoint/contract.py` is still a `defer` surface:
  - public mode normalizers are still imported by `tests/train/test_event_motion_model_phase_d_mode_normalization.py`;
  - bundle-attach behavior is still exercised via `train/posttrain_build_shell.py` and `train/training_MPL.py`;
  - even sparse callers are enough to keep this in the checkpoint-contract boundary for now.
- The old `train.posttrain` helper namespace is still active in tools/tests:
  - callers still import `_cfg_from_payload`, `_build_dataset_and_loader`, `_build_posttrain_model_from_ckpt`, and/or `_save_posttrain_outputs` from `train.posttrain`;
  - this includes `tools/run_posttrain_nonleg_trunk_ablation.py`, `tools/run_cp015_tailk7_replace_efficiency_audit.py`, multiple `ep014center_*` tools, `tools/run_posttrain_build_shell_smoke.py`, and `tests/train/test_posttrain_pose_history_phase3.py`.
- Historical root-level shim targets are already absent from the current `train/` tree:
  - no current files matched `train/model_ckpt_contract.py`, `train/model_ckpt_compat.py`, `train/posttrain_common.py`, or `train/rotvec_semantics.py`.
- Zero-caller / near-zero-caller hits found in code:
  - `train/checkpoint/__init__.py` is only a package marker; no direct callers found.
  - `train/contracts/asset_semantics.py` keeps `LEGACY_ROTVEC_SEMANTICS`, but no active code caller was found; only a historical relayout note still mentions it.

Cleanup decision:

- Gate stays green, but there is still **no safe, high-value deletion target**.
- No code deletion was made in this round.
- Do not delete `train/checkpoint/__init__.py` or `LEGACY_ROTVEC_SEMANTICS` just for churn reduction:
  - both are low-value removals;
  - neither meaningfully shrinks the active compat/runtime surface;
  - deleting them now would add package/import or historical-contract review cost with little payoff.

---

## 10. Follow-up update — 2026-04-18 tools-side build-shell migration

Status:

- The tools-side migration for the posttrain build-shell surface is now complete.
- `train/posttrain.py` no longer re-exports `_build_posttrain_model_from_ckpt`, `_instantiate_posttrain_model`, `_load_posttrain_checkpoint_into_model`, or `_resolve_posttrain_model_build_state` from its top-level import surface.
- Active tools now import the build-shell helper directly from `train/posttrain_build_shell.py`.

Representative rewired callers:

- `tools/run_posttrain_nonleg_trunk_ablation.py`
- `tools/run_cp015_tailk7_replace_efficiency_audit.py`
- `tools/run_ep014center_per_head_merged_stage.py`
- `tools/run_ep014center_70a_to71_plain_leg_cleanup.py`
- `tools/run_ep014center_stage6_early_live_cleanup_bias.py`
- `tools/run_ep014center_replace_redesign.py`

What changed in cleanup status:

- Gate Step 1 from Section 7 (“rewire active tools away from old `train.posttrain` helper namespace”) is complete for the **build-shell subset**.
- This makes the old `train.posttrain -> posttrain_build_shell` re-export shim eligible for removal, and that shim has now been removed.
- This does **not** mean the whole `train.posttrain` helper namespace is dead:
  - tools still use `_cfg_from_payload`, `_build_dataset_and_loader`, and `_save_posttrain_outputs`;
  - tests still use posttrain runtime/rollout helpers directly;
  - caller-local overlay and live-attr boundaries are unchanged.

Validation run after rewiring:

- `python3 -m py_compile train/posttrain.py train/posttrain_build_shell.py tools/run_posttrain_nonleg_trunk_ablation.py tools/run_ep014center_per_head_merged_stage.py tools/run_ep014center_70a_to71_plain_leg_cleanup.py tools/run_ep014center_stage6_early_live_cleanup_bias.py tools/run_ep014center_replace_redesign.py tools/run_cp015_tailk7_replace_efficiency_audit.py`
- `python3 tools/run_posttrain_build_shell_smoke.py`

Updated cleanup recommendation:

- It is now reasonable to treat the removed build-shell re-export as a completed local cleanup.
- The next `defer` boundary remains the broader `train.posttrain` entry/runtime helper surface, not checkpoint compat or overlay removal.

---

## 10b. Follow-up update — 2026-04-18 neutral contact-pretrain runtime attrs

Status:

- The contact-pretrain runtime contract is now split into:
  - caller-local owner attrs that keep their existing prefixes;
  - shared neutral attrs used by shared `Trainer` runtime code.
- Dual-write is now centralized in `train/runtime_attach.py:106` via
  `apply_contacts_pretrain_runtime(...)`, so basetrain/posttrain no longer
  hand-maintain separate prefixed + neutral setattr sequences.
- Basetrain owner now writes both:
  - `trainbase_contacts_pretrain_clamp`
  - `trainbase_contacts_pretrain_affine_stats_spec`
  - `trainbase_contacts_pretrain_affine`
  - plus neutral `contacts_pretrain_clamp`
  - `contacts_pretrain_affine_stats_spec`
  - `contacts_pretrain_affine`
- Posttrain owner now writes both:
  - `posttrain_contacts_pretrain_clamp`
  - `posttrain_contacts_pretrain_affine_stats_spec`
  - `posttrain_contacts_pretrain_affine`
  - plus the same neutral `contacts_pretrain_*` attrs.
- `Trainer._predict_pretrain_contacts_from_frozen(...)` no longer reads across prefixes; it now reads only the neutral attrs.
- `contacts_pretrain_runtime_attached` is now the explicit activation marker:
  - missing / false => legal inactive path, shared reader may return `None`;
  - true + missing neutral attr => loud `RuntimeError` instead of silent skip.

What changed in cleanup status:

- The specific cross-prefix fallback previously used inside shared `Trainer` runtime code is now retired.
- This should be treated as a runtime-contract cleanup, not as a live-attr rename:
  - prefixed attrs remain owned by basetrain/posttrain callers;
  - shared runtime code no longer knows those prefixes.
- The earlier inventory row for the `trainbase_*` / `posttrain_*` live-attr bridge reflects the pre-update state; this note supersedes that specific bridge detail.
- The broader caller-local live-attr surface is still active, so this does **not** authorize broad live-attr cleanup.

Validation run after landing neutral attrs:

- `python3 -m py_compile train/runtime_attach.py train/training_MPL.py train/posttrain.py tools/run_training_mpl_entry_shell_smoke.py tools/run_posttrain_runtime_overlay_smoke.py tests/train/test_contacts_pretrain_runtime_attach.py`
- `python3 tools/run_training_mpl_entry_shell_smoke.py --out-dir debug_output/_training_mpl_entry_shell_smokes_attach_helper_20260418`
- `python3 tools/run_posttrain_runtime_overlay_smoke.py --out-dir debug_output/_posttrain_runtime_overlay_smokes_attach_helper_20260418`
- `python3 -m unittest tests.train.test_contacts_pretrain_runtime_attach`

Updated cleanup recommendation:

- It is now reasonable to treat the contact-pretrain cross-prefix bridge retirement as a completed local runtime-contract cleanup.
- The dual-write / activation rule is now explicit enough to treat “shared reads owner-local prefix” as a regression.
- The next `defer` boundary remains broader overlay / live-attr redesign, not checkpoint compat deletion.

---

## 10c. Follow-up update — 2026-04-18 neutral contact-pretrain payload plumbing

Status:

- The already-parsed contact-pretrain runtime payload now flows as the neutral
  `ContactPretrainRuntime` value object through:
  - `train/runtime_attach.py::apply_contacts_pretrain_runtime(...)`;
  - `train/training_MPL.py::TrainerRuntimeConfig`;
  - `train/posttrain.py::PosttrainLocalRuntimeOverlay`.
- Caller-facing config keys and live attrs are unchanged:
  - basetrain still owns `trainbase_contacts_pretrain_*`;
  - posttrain still owns `posttrain_contacts_pretrain_*`;
  - shared `Trainer` still reads only `contacts_pretrain_*`.
- `owner_prefix` remains only at the final attach boundary where the helper
  dual-writes owner-local attrs plus neutral attrs.

Why this cleanup qualifies:

- It does not touch checkpoint load/save/rebuild contracts.
- It does not rename live trainer attrs or posttrain config keys.
- It reduces internal caller-prefix coupling by removing the resolved
  `trainbase_*` / `posttrain_*` contact-pretrain triplets from runtime carrier
  dataclasses.

Validation run after landing:

- `python3 -m py_compile train/checkpoint/compat.py train/checkpoint/contract.py train/runtime_attach.py train/posttrain_build_shell.py train/posttrain.py train/training_MPL.py`
- `python3 -m unittest tests.train.test_contacts_pretrain_runtime_attach`
- `python3 tools/run_training_mpl_entry_shell_smoke.py`
- `python3 tools/run_posttrain_runtime_overlay_smoke.py`

Updated cleanup recommendation:

- This is a completed small runtime-contract cleanup.
- Further broad overlay/live-attr redesign remains `defer`.

---

## 10d. Follow-up update — 2026-04-18 next-direction scan result

Status:

- After the neutral contact-pretrain payload cleanup, a follow-up scan of
  `train/runtime_attach.py`, `train/training_MPL.py`, `train/rollout_kernel.py`,
  and `train/posttrain.py` did **not** find another equally strong
  shared/core cleanup seam in the same family.
- Shared/core runtime no longer reads `posttrain_*` or other caller-prefixed
  contact-pretrain attrs; remaining owner naming is confined to the final
  dual-write attach point.

Most visible remaining duplication:

- `train/validate/run_freerun_cycles.py` still mirrors parts of the
  posttrain-local overlay contract:
  - `contact_meas_*` parsing / normalization;
  - `lambda_reliability_*` parsing / trainer attach.
- A small number of probe/follow-up tools also set `trainer.lambda_reliability_*`
  directly after reconstructing posttrain-like runtime state.

Why this is `defer`, not the next cleanup:

- The compat inventory already classifies eval-only legacy adapters as `defer`.
- Rewiring this surface now would mainly clean up eval/tool-local duplication,
  not reduce shared `Trainer` / shared runtime coupling in a meaningful way.
- Pulling this path into a more formal shared overlay contract would approach the
  currently blocked “broad overlay owner redesign” boundary.

Updated cleanup recommendation:

- Treat the next-direction scan result as `defer`.
- Re-open this seam only if eval/runtime-adapter cleanup becomes an explicit
  goal, or if a future step intentionally promotes more of the posttrain-local
  overlay into a stable shared contract.

---

## 10e. Follow-up update — 2026-04-18 shared loss runtime sync helper

Status:

- A small shared/runtime seam was still duplicated across basetrain and
  posttrain:
  - basetrain synced `loss_fn.mu_y` / `loss_fn.std_y` from `Trainer`, and
    optionally copied `_bundle_meta`;
  - posttrain separately synced the same runtime stats onto `loss_fn`.
- This has now been centralized in
  `train/runtime_attach.py::apply_loss_runtime_from_trainer(...)`.

What changed:

- `train/training_MPL.py` now uses the shared helper for train-entry loss runtime
  sync, including the existing optional `_bundle_meta -> loss_fn.meta` copy.
- `train/posttrain.py` now uses the same shared helper for posttrain loss runtime
  stats attach.
- No caller-facing config keys, live trainer attrs, checkpoint keys, or overlay
  ownership rules changed.

Why this cleanup qualifies:

- It is a narrow shared/runtime seam, not a broad overlay redesign.
- It removes a duplicated attach rule from two runtime builders.
- It keeps the shared contract explicit: dataset/runtime attach hydrates both
  `Trainer` and `loss_fn` via shared helpers instead of caller-local ad hoc sync.

Validation run after landing:

- `python3 -m py_compile train/runtime_attach.py train/training_MPL.py train/posttrain.py tests/train/test_contacts_pretrain_runtime_attach.py`
- `python3 -m unittest tests.train.test_contacts_pretrain_runtime_attach`
- `python3 tools/run_training_mpl_entry_shell_smoke.py`
- `python3 tools/run_posttrain_runtime_overlay_smoke.py`

Updated cleanup recommendation:

- This is another completed small shared/runtime cleanup.
- After this, the remaining high-visibility duplication is still concentrated in
  eval/runtime-adapter surfaces, which remain `defer` unless that direction
  becomes an explicit goal.

---

## 10f. Shared runtime hydrator seam map

This table tracks `train/runtime_attach.py` as the shared runtime hydrator,
separate from caller-local overlay ownership.

| Seam | Current status | Blast radius | Next action |
|---|---|---|---|
| Shared trainer runtime (`pose_hist_mu/std/scales`, yaw axis/offset, run metadata) | `closed` via `SharedTrainerRuntime` + `apply_shared_trainer_runtime(...)` | basetrain + posttrain trainer runtime attach | Keep; use entry-shell and posttrain overlay smokes as regression gates. |
| Contact-pretrain runtime (`contacts_pretrain_*` neutral attrs + owner-local dual-write) | `closed` via `ContactPretrainRuntime` + `apply_contacts_pretrain_runtime(...)` | basetrain + posttrain owner-local attrs plus shared `Trainer` reader | Keep; treat shared reads of owner-prefixed attrs as regression. |
| Loss runtime (`loss_fn.mu_y/std_y`, optional basetrain `_bundle_meta -> loss_fn.meta`) | `closed` via `apply_loss_runtime_from_trainer(...)` | basetrain + posttrain loss runtime sync | Keep; helper now loud-fails only on partial `mu_y` / `std_y` pairs, while allowing both missing. |
| Eval/runtime adapter overlay mirror (`run_freerun_cycles.py` contact-meas + lambda reliability attach) | `defer` | eval adapter and probe tools | Do not fold into shared hydrator unless eval/runtime-adapter cleanup becomes an explicit goal. |
| Frozen encoder/contact-head attach | `defer-scan` | checkpoint/build shell, basetrain runtime prepare, tools | Re-open only with a focused caller/callsite scan; do not touch checkpoint compat or model constructor. |
| Angvel/contact-meas state slices | `defer-scan` | dataset layout, model runtime, rollout contact-pretrain input | Re-open only if a clear duplicated attach helper exists and smokes can cover it. |

Working rule:

- Add to `train/runtime_attach.py` only when the seam is a shared hydration rule
  already used by basetrain and posttrain.
- Keep caller-local policy, overlay parsing, and eval adapter compatibility out
  of this module unless a later gate explicitly promotes them.
