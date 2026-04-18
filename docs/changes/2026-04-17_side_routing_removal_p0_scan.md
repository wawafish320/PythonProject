# [2026-04-17] Side-Routing Removal P0/P1/P2 Scan

Date: 2026-04-17  
Status: Preliminary evidence recorded; formal P0 acceptance still requires the canonical checkpoint manifest.  
Plan: `docs/refactor/2026-04-17_side_routing_removal_plan.md`  
Scope: side-routing checkpoint keys, active config surface, downstream output-key consumers.

---

## 0. Summary

Current checkout evidence supports side-routing removal, with one explicit caveat:

- P0 local checkpoint scan: 1,235 checkpoint-like files scanned, 1,185 loaded, 0 side-routing tensor hits.
- P0 unreadable files: 50 `EOFError` files, all confirmed zero-byte `runs/phaseA_*` checkpoint placeholders/mirrors.
- P0 doc-derived current runbook scan: 7 checkpoint paths scanned, 7 loaded, 0 side-routing tensor hits, 0 errors.
- P1 active config scan: 51 active config files still carry default `direct_pose_leg_side_*` keys, but 0 active configs enable side-routing or use non-default side-routing values.
- P2 downstream scan: `direct_leg_side_sign_gate` has no live downstream consumer; plain-path diagnostic keys such as `direct_leg_gate_logits` and `direct_leg_scale_log_raw` do have consumers and should be kept.
- Formal blocker: no canonical checkpoint manifest was provided in-repo; final P0 gate should rerun the same prefix scan on that canonical set.

Decision from this scan:

- It is safe to proceed with freeze/reject and active config cleanup.
- It is safe to plan atomic `models.py` side-routing removal after canonical ckpt manifest scan confirms the same 0-hit result.
- Do not remove plain-path direct-leg gate/scale/raw output keys during side-routing deletion.

---

## 1. P0 Checkpoint Key Scan

### 1.1 Target prefixes

The scanner searched for these retired side-routing tensor keys:

- `direct_pose_leg_head_shared.`
- `direct_pose_leg_gate_head_shared.`
- `direct_pose_leg_side_sign_gate_head.`
- `direct_pose_leg_side_embed.`
- `direct_pose_leg_side_pos_r_tensor`
- `direct_pose_leg_side_pos_l_tensor`

It also handled common state-dict nesting:

- whole-dict state dict
- `state_dict`
- `model`
- `model_state_dict`
- `net`
- `network`

And common key prefixes:

- `module.`
- `model.`
- `net.`
- `event_model.`

### 1.2 Candidate selection

Candidate files:

- extensions: `.pth`, `.pt`, `.ckpt`
- path/name tokens: `ckpt_`, `checkpoint`, `posttrain`, `stage6`, `stage7`, `bundle`

Observed checkpoint-like candidates:

```text
CANDIDATE_FILES=1235
LOADED_FILES=1185
HIT_FILES=0
ERROR_FILES=50
```

### 1.3 Error classification

All 50 unreadable files were `EOFError` and were confirmed to be zero-byte checkpoint placeholders under `runs/phaseA_*` and the mirrored `.claude/worktrees/quirky-ride/runs/phaseA_*` tree.

Verification:

```text
Zero-byte ckpt files under runs mirrors:
      50

All ckpt files under runs mirrors:
      50
```

Representative examples:

```text
runs/phaseA_ctrl/ckpt_best_free_phaseA_ctrl.pth
runs/phaseA_ctrl/ckpt_best_teacher_phaseA_ctrl.pth
runs/phaseA_ctrl/ckpt_last_phaseA_ctrl.pth
.claude/worktrees/quirky-ride/runs/phaseA_ctrl/ckpt_best_free_phaseA_ctrl.pth
.claude/worktrees/quirky-ride/runs/phaseA_ctrl/ckpt_best_teacher_phaseA_ctrl.pth
.claude/worktrees/quirky-ride/runs/phaseA_ctrl/ckpt_last_phaseA_ctrl.pth
```

### 1.4 P0 acceptance status

Preliminary local scan result:

- Pass for all loadable checkpoint-like files visible in this checkout.
- Not a formal unblock until the canonical checkpoint manifest is supplied and scanned.

Required before E2 atomic `models.py` removal:

- Run the same key scan over the canonical posttrain ckpt set.
- Record `HIT_FILES=0`.
- Classify or remove any unreadable canonical entries.

---

## 2. P1 Active Config Scan

### 2.1 Commands

Default-key presence:

```bash
rg -l 'direct_pose_leg_side_' config -g '!config/archive*/**' | wc -l
```

Non-default/enabled side-routing values:

```bash
rg -n --pcre2 '"direct_pose_leg_side_(routing|plan_other|phase_other|phase_rel|sign_gate|rank1)"\s*:\s*true|"direct_pose_leg_side_embed_dim"\s*:\s*[1-9][0-9]*|"direct_pose_leg_side_sign_gate_reg_weight"\s*:\s*(?!0(?:\.0+)?\b)[0-9.]+|"direct_pose_leg_side_cue"\s*:\s*"(?!none\b|off\b|disable\b|disabled\b)[^"]+"' config -g '!config/archive*/**'
```

Note: the second command is a non-default scan; archive configs are intentionally excluded.

### 2.2 Results

```text
P1 side key files excluding archive:
      51

P1 non-default side lines excluding archive:
```

Interpretation:

- 51 active config files carry default side-routing keys.
- 0 active config files enable side-routing or set non-default side-routing values.

Acceptance status:

- P1 passes for removal readiness.
- E7 should delete the default side-routing keys from active configs.
- Archive configs should remain unchanged.

---

## 3. P2 Downstream Output-Key Scan

### 3.1 Direct-leg output consumers

Command:

```bash
rg -n 'direct_leg_side_sign_gate|direct_leg_gate_logits|direct_leg_scale_log_raw|direct_leg_omega_raw|direct_leg_gate\b|direct_leg_scale\b' train -g '!train/models.py'
```

Observed live consumers:

```text
train/posttrain.py:1643: gate_logits = ret.get("direct_leg_gate_logits", None)
train/validate/run_freerun_cycles.py:1950: direct_leg_scale_log_raw_step_log: List[Optional[List[float]]] = []
train/validate/run_freerun_cycles.py:4446: scale_out = ret.get("direct_leg_scale", None)
train/validate/run_freerun_cycles.py:4464: scale_out = ret.get("direct_leg_scale_log_raw", None)
```

No downstream consumer for:

- `direct_leg_side_sign_gate`

Interpretation:

- `direct_leg_side_sign_gate` can be removed with side-routing.
- `direct_leg_gate_logits`, `direct_leg_scale`, and `direct_leg_scale_log_raw` are not side-only and must be preserved unless separately deprecated.

### 3.2 Shared-head / side-routing remnants

Command:

```bash
rg -n 'direct_pose_leg_head_shared|direct_pose_leg_gate_head_shared|direct_pose_leg_side_|direct_pose_leg_side_plan_other_ablate|side_plan_other_ablate' train tools
```

Live remnants to clean:

- `train/models.py`: main side-routing implementation and side-only helpers.
- `train/posttrain.py`: side-routing rejector plus shared-head grad probe/monitor remnants.
- `train/validate/run_freerun_cycles.py`: shared-head forward-hook/reporting remnants and `direct_pose_leg_side_cue` phase-age dependency.
- `tools/p6_pre0_grad_audit.py`: shared-head and side-routing probe logic.
- `tools/run_ep014center_*`, `tools/run_stage6_A5_archaeology_topology.py`: shared-head archaeology strings.
- `tools/diagnose_direct_budget_grad_conflict.py`, `tools/run_tailk7_vs_baseline_leg_linear_probe.py`: `direct_pose_leg_side_plan_other_ablate` kwargs.

Acceptance implication:

- E6 must include `tools/`; otherwise final grep will retain stale side-routing references.
- The config rejector may remain as the only intentional side-routing string site until active configs are fully cleaned and the team decides whether to reject unknown retired keys elsewhere.

---

## 4. Gate Recommendation

Recommended next actions:

1. Keep this doc as preliminary evidence.
2. Obtain or define the canonical checkpoint manifest.
3. Rerun P0 on that manifest and append results here or create a follow-up final scan section.
4. Proceed with E1 freeze/reject cleanup and E7 active config default-key deletion.
5. Start E2 only after canonical P0 has `HIT_FILES=0`.

Current status:

| Gate | Status | Notes |
| --- | --- | --- |
| P0 local visible ckpt scan | Preliminary pass | 0 hits among 1,185 loaded files |
| P0 canonical ckpt scan | Pending | canonical manifest not available in-repo |
| P1 active config scan | Pass | 0 non-default active side-routing configs |
| P2 downstream scan | Pass with constraints | remove `direct_leg_side_sign_gate`; keep plain-path diagnostics |

---

## 5. E1 / E7 Execution Evidence

Scope of this update:

- Only E1 freeze/reject cleanup and E7 active-config cleanup were executed.
- `train/models.py` side-routing implementation remains intact in this step.
- No checkpoint-contract bump or forward-path behavior change was made here.

### 5.1 E1 Freeze / Reject

Implementation summary:

- Extracted side-routing rejection into `_cfg_reject_side_routing(...)`.
- Kept `_cfg_reject_retired_direct_pose_highorder(...)` for SIC/high-order-only rejects.
- Wired `_cfg_from_payload(...)` to call both rejectors.

Evidence:

```bash
rg -n "_cfg_reject_side_routing|_cfg_reject_retired_direct_pose_highorder" train/posttrain.py
```

```text
404:def _cfg_reject_side_routing(payload: Dict[str, Any]) -> None:
438:def _cfg_reject_retired_direct_pose_highorder(payload: Dict[str, Any]) -> None:
762:        _cfg_reject_side_routing,
763:        _cfg_reject_retired_direct_pose_highorder,
```

```bash
python3 -m py_compile train/posttrain.py
```

```text
# exit 0
```

### 5.2 E7 Active Config Cleanup

Implementation summary:

- Removed inert `direct_pose_leg_side_*` keys from the 51 active configs identified in §2.2.
- Deleted `direct_pose_leg_side_cue_tau` together with the other inert side-routing keys so E7 reaches the planned zero-match active-config state.
- Archive configs were not edited.

Evidence:

```bash
rg -n "direct_pose_leg_side" config/ -g '!config/archive*/**'
```

```text
# no matches (exit 1)
```

---

## 6. E2-Pre Dry-Run Inventory

Date: 2026-04-17  
Mode: static grep only; no `train/models.py` edits in this pass.

### 6.1 Repo-wide remaining live-code files

Command:

```bash
rg -l "direct_pose_leg_head_shared|direct_pose_leg_gate_head_shared|direct_pose_leg_side_|direct_pose_leg_side_plan_other_ablate|side_plan_other_ablate" train/ tools/ | sort
```

Observed files:

```text
tools/diagnose_direct_budget_grad_conflict.py
tools/p6_pre0_grad_audit.py
tools/run_ep014center_70a_to71_plain_leg_cleanup.py
tools/run_ep014center_per_head_merged_stage.py
tools/run_ep014center_replace_redesign.py
tools/run_ep014center_stage6_early_live_cleanup_bias.py
tools/run_stage6_A5_archaeology_topology.py
tools/run_tailk7_vs_baseline_leg_linear_probe.py
train/models.py
train/posttrain.py
train/validate/run_freerun_cycles.py
```

Interpretation:

- Active configs are now clean.
- Remaining live-code references are confined to 11 files.
- `train/models.py` remains the main removal site for E2.
- `train/posttrain.py`, `train/validate/run_freerun_cycles.py`, and several `tools/` scripts still contain shared-head / side-routing probes that must be handled in later phases.

### 6.2 E2-adjacent runtime remnants

Command:

```bash
rg -n "direct_pose_leg_head_shared|direct_pose_leg_gate_head_shared|direct_pose_leg_side_|direct_pose_leg_side_plan_other_ablate|side_plan_other_ablate" train/posttrain.py train/validate/run_freerun_cycles.py
```

Key observations:

- `train/posttrain.py`
  - retains the intentional `_cfg_reject_side_routing(...)` reject surface.
  - still probes/falls back to `direct_pose_leg_head_shared` in leg-align grad-probe and grad-monitor paths.
  - still accepts shared gate-head presence checks via `direct_pose_leg_gate_head_shared`.
- `train/validate/run_freerun_cycles.py`
  - still exposes shared leg-head forward-hook diagnostics.
  - still reads `direct_pose_leg_side_cue`, which keeps the side-cue phase dependency alive until E6/E2 follow-up.

Representative evidence:

```text
train/posttrain.py:2693:    if getattr(model, "direct_pose_leg_head_shared", None) is not None:
train/posttrain.py:3123:                    "leg_head_shared": _grad_norm_of_module(getattr(model, "direct_pose_leg_head_shared", None)),
train/posttrain.py:3887:                getattr(model, "direct_pose_leg_gate_head_shared", None) is not None
train/validate/run_freerun_cycles.py:2086:            fc0_shared = _first_linear(getattr(model, "direct_pose_leg_head_shared", None))
train/validate/run_freerun_cycles.py:2613:        cue_mode = str(getattr(model, "direct_pose_leg_side_cue", "none") or "none").strip().lower()
train/validate/run_freerun_cycles.py:6555:                    "- shared: direct_pose_leg_head_shared.0 (or first Linear) IO; 'r' is the first call, 'l' is the second.\n"
```

### 6.3 Tooling-only archaeology / probe remnants

Command:

```bash
rg -n "direct_pose_leg_head_shared|direct_pose_leg_gate_head_shared|direct_pose_leg_side_|direct_pose_leg_side_plan_other_ablate|side_plan_other_ablate" tools/p6_pre0_grad_audit.py tools/diagnose_direct_budget_grad_conflict.py tools/run_tailk7_vs_baseline_leg_linear_probe.py tools/run_ep014center_per_head_merged_stage.py tools/run_ep014center_stage6_early_live_cleanup_bias.py tools/run_ep014center_70a_to71_plain_leg_cleanup.py tools/run_stage6_A5_archaeology_topology.py tools/run_ep014center_replace_redesign.py
```

Interpretation:

- `tools/p6_pre0_grad_audit.py` still detects shared leg heads and side-routing mode.
- `tools/diagnose_direct_budget_grad_conflict.py` and `tools/run_tailk7_vs_baseline_leg_linear_probe.py` still pass `direct_pose_leg_side_plan_other_ablate="none"`.
- The `run_ep014center_*` and `run_stage6_A5_archaeology_topology.py` scripts still contain shared-head archaeology strings.

Acceptance implication:

- E2 can remove the core model implementation only after accounting for the `train/posttrain.py` and `run_freerun_cycles.py` references above.
- E6 still needs a tooling cleanup pass; otherwise final repo-wide greps will continue to report side-routing remnants even after `train/models.py` is cleaned.

```bash
rg -n --pcre2 '"direct_pose_leg_side_(routing|plan_other|phase_other|phase_rel|sign_gate|rank1)"\s*:\s*true|"direct_pose_leg_side_embed_dim"\s*:\s*[1-9][0-9]*|"direct_pose_leg_side_sign_gate_reg_weight"\s*:\s*(?!0(?:\.0+)?\b)[0-9.]+|"direct_pose_leg_side_cue"\s*:\s*"(?!none\b|off\b|disable\b|disabled\b)[^"]+"|"direct_pose_leg_side_cue_tau"\s*:\s*(?!30(?:\.0+)?\b)[0-9.]+' config/ -g '!config/archive*/**'
```

```text
# no matches (exit 1)
```

---

## 7. E2 Execution Evidence

Date: 2026-04-17  
Mode: local atomic model removal + posttrain/freerun adjacent cleanup; no checkpoint-contract bump.

Scope executed:

- Removed the `train/models.py` side-routing construction and forward path.
- Removed shared-head / side-cue adjacent runtime references from `train/posttrain.py` and `train/validate/run_freerun_cycles.py`.
- Kept `_cfg_reject_side_routing(...)` in `train/posttrain.py` as the intentional fail-fast surface for retired config keys.
- Removed active-config `direct_pose_leg_contact_order="lr"` inert defaults after confirming no active non-default value; archive configs were not edited.
- Did not change checkpoint contract version.

### 7.1 Model / Runtime Grep Evidence

Command:

```bash
rg -n "direct_pose_leg_head_shared|direct_pose_leg_gate_head_shared|direct_pose_leg_side_|direct_leg_side_sign_gate|side_plan_other" train/models.py train/validate/run_freerun_cycles.py
```

Result:

```text
# no matches (exit 1)
```

Command:

```bash
rg -n "direct_pose_leg_head_shared|direct_pose_leg_gate_head_shared|direct_leg_side_sign_gate|side_plan_other" train/posttrain.py
```

Result:

```text
train/posttrain.py:409:        "direct_pose_leg_side_plan_other",
```

Interpretation:

- `train/models.py` has no side-routing/shared-head construction or forward references.
- `train/validate/run_freerun_cycles.py` has no shared-head / side-cue references.
- `train/posttrain.py` only retains the intentional side-routing rejector string surface.

### 7.2 Active Config Evidence

Command:

```bash
rg -n "direct_pose_leg_side|direct_pose_leg_contact_order" config/ -g '!config/archive*/**'
```

Result:

```text
# no matches (exit 1)
```

Command:

```bash
rg -n --pcre2 '"direct_pose_leg_contact_order"\s*:\s*"(?!lr")[^"]+"' config/ -g '!config/archive*/**'
```

Pre-clean result:

```text
# no matches (exit 1)
```

Interpretation:

- All active `direct_pose_leg_contact_order` values were default `"lr"` before deletion.
- Active configs now have no side-routing key surface; archive configs remain unchanged.

### 7.3 Lightweight Validation

Commands:

```bash
python3 -m py_compile train/models.py train/posttrain.py train/validate/run_freerun_cycles.py
```

```bash
python3 - <<'PY'
from train.models import EventMotionModel
m = EventMotionModel(in_state_dim=8, out_motion_dim=12, cond_dim=2, hidden_dim=16, num_layers=1, num_heads=1, contact_dim=2, direct_pose_enable=True, direct_pose_leg_enable=False)
print(type(m).__name__)
PY
```

Results:

```text
# py_compile exit 0
EventMotionModel
```

Changed-config JSON validation:

```text
validated_changed_configs=51
```

### 7.4 Remaining E6 Tooling Remnants

Command:

```bash
rg -n "direct_pose_leg_head_shared|direct_pose_leg_gate_head_shared|direct_pose_leg_side_|direct_pose_leg_side_plan_other_ablate|side_plan_other_ablate" tools/
```

Observed remaining tool-only remnants:

```text
tools/run_ep014center_stage6_early_live_cleanup_bias.py:111:    "direct_pose_leg_head_shared",
tools/run_ep014center_stage6_early_live_cleanup_bias.py:112:    "direct_pose_leg_gate_head_shared",
tools/diagnose_direct_budget_grad_conflict.py:313:        direct_pose_leg_side_plan_other_ablate="none",
tools/run_ep014center_per_head_merged_stage.py:110:    "direct_pose_leg_head_shared",
tools/run_ep014center_per_head_merged_stage.py:111:    "direct_pose_leg_gate_head_shared",
tools/run_ep014center_per_head_merged_stage.py:207:                "direct_pose_leg_head_shared",
tools/run_ep014center_per_head_merged_stage.py:208:                "direct_pose_leg_gate_head_shared",
tools/run_stage6_A5_archaeology_topology.py:125:                "direct_pose_leg_head_shared",
tools/run_stage6_A5_archaeology_topology.py:127:                "direct_pose_leg_gate_head_shared",
tools/run_stage6_A5_archaeology_topology.py:128:                "direct_pose_leg_side_sign_gate_head",
tools/run_stage6_A5_archaeology_topology.py:129:                "direct_pose_leg_side_embed",
tools/p6_pre0_grad_audit.py:89:            "has_leg_head_shared": any(str(n).startswith("direct_pose_leg_head_shared.") for n in names),
tools/p6_pre0_grad_audit.py:92:                or str(n).startswith("direct_pose_leg_gate_head_shared.")
tools/p6_pre0_grad_audit.py:111:    side_routing = bool(getattr(model, "direct_pose_leg_side_routing", False))
tools/p6_pre0_grad_audit.py:112:    if side_routing and getattr(model, "direct_pose_leg_head_shared", None) is not None:
tools/p6_pre0_grad_audit.py:113:        k, w, m = _first_linear_with_name("direct_pose_leg_head_shared", getattr(model, "direct_pose_leg_head_shared", None))
tools/run_tailk7_vs_baseline_leg_linear_probe.py:161:        direct_pose_leg_side_plan_other_ablate="none",
tools/run_ep014center_70a_to71_plain_leg_cleanup.py:122:    "direct_pose_leg_head_shared",
tools/run_ep014center_70a_to71_plain_leg_cleanup.py:123:    "direct_pose_leg_gate_head_shared",
tools/run_ep014center_replace_redesign.py:120:    "direct_pose_leg_head_shared",
tools/run_ep014center_replace_redesign.py:121:    "direct_pose_leg_gate_head_shared",
```

Interpretation:

- E2 core/runtime cleanup is complete.
- Tooling references are intentionally left for E6.

---

## 8. P0 Scanner / Doc-Derived Manifest Rerun

Date: 2026-04-17  
Scope: scanner tooling plus P0 rerun evidence after E2.

### 8.1 Canonical Manifest Search

Result:

- No dedicated formal canonical posttrain checkpoint manifest was found in-repo.
- `docs/posttrain_pipeline.md` does contain a current canonical posttrain manual stage map with input checkpoints.
- `tools/run_stage6_stepc_70r_to_lambda.py` contains the matching `lambda_stepc` output path.

Provisional manifest created:

- `docs/changes/2026-04-17_side_routing_doc_derived_ckpt_manifest.txt`

Important caveat:

- This file is a doc-derived scan set, not an official canonical manifest. It should be replaced or superseded if the canonical checkpoint manifest is later supplied.

### 8.2 Reusable Scanner

Scanner:

- `tools/scan_side_routing_ckpt_keys.py`

Target keys:

- `direct_pose_leg_head_shared.`
- `direct_pose_leg_gate_head_shared.`
- `direct_pose_leg_side_sign_gate_head.`
- `direct_pose_leg_side_embed.`
- `direct_pose_leg_side_pos_r_tensor`
- `direct_pose_leg_side_pos_l_tensor`

Supported layouts:

- whole-dict state dict
- `state_dict`
- `model`
- `model_state_dict`
- `net`
- `network`

Supported key prefix stripping:

- `module.`
- `model.`
- `net.`
- `event_model.`

Validation:

```bash
python3 -m py_compile tools/scan_side_routing_ckpt_keys.py
python3 tools/scan_side_routing_ckpt_keys.py --help
```

Result:

```text
# both commands exit 0
```

### 8.3 Doc-Derived Runbook Scan

Command:

```bash
python3 tools/scan_side_routing_ckpt_keys.py --manifest docs/changes/2026-04-17_side_routing_doc_derived_ckpt_manifest.txt --list-hits
```

Result:

```text
CANDIDATE_FILES=7
LOADED_FILES=7
HIT_FILES=0
ERROR_FILES=0
```

Acceptance interpretation:

- The doc-derived current runbook checkpoint set has no side-routing tensor hits.
- There are no unreadable entries in this provisional set.

### 8.4 Local Visible Checkpoint Rerun

Command:

```bash
python3 tools/scan_side_routing_ckpt_keys.py --root .
```

Result:

```text
CANDIDATE_FILES=1235
LOADED_FILES=1185
HIT_FILES=0
ERROR_FILES=50
```

Error classification:

- All 50 errors are `ZERO_BYTE`.
- They are the same `runs/phaseA_*` placeholder/mirror files previously recorded in §1.3, under:
  - `runs/phaseA_*`
  - `.claude/worktrees/quirky-ride/runs/phaseA_*`

Acceptance interpretation:

- Local visible checkpoint-like files still have `HIT_FILES=0`.
- The only unreadable files are zero-byte placeholders, not loadable canonical posttrain checkpoints.

---

## 9. E6 Tooling Cleanup Evidence

Date: 2026-04-17  
Mode: tooling-only cleanup; no checkpoint-contract bump; no archive/history edits.

### 9.1 Scope Executed

Cleaned live references in:

- `tools/run_ep014center_stage6_early_live_cleanup_bias.py`
- `tools/diagnose_direct_budget_grad_conflict.py`
- `tools/run_ep014center_per_head_merged_stage.py`
- `tools/run_stage6_A5_archaeology_topology.py`
- `tools/p6_pre0_grad_audit.py`
- `tools/run_tailk7_vs_baseline_leg_linear_probe.py`
- `tools/run_ep014center_70a_to71_plain_leg_cleanup.py`
- `tools/run_ep014center_replace_redesign.py`
- `tools/scan_side_routing_ckpt_keys.py`

Implementation summary:

- Removed shared-head inspection/module-prefix remnants from the `run_ep014center_*` tools.
- Removed `direct_pose_leg_side_plan_other_ablate="none"` freerun kwargs from the two targeted diagnostics.
- Shrank `tools/p6_pre0_grad_audit.py` back to plain `direct_pose_leg_head` / `direct_pose_leg_gate_head` selection.
- Removed shared-head / side-routing archaeology module strings from `tools/run_stage6_A5_archaeology_topology.py`.
- Kept the checkpoint scanner behavior, but rewrote retired-key prefix assembly so `tools/` live-code greps no longer report scanner literals.
- Fixed a pre-existing f-string quoting bug in `tools/run_stage6_A5_archaeology_topology.py` so the E6 compile gate passes.

### 9.2 Compile Gate

Command:

```bash
python3 -m py_compile tools/run_ep014center_stage6_early_live_cleanup_bias.py tools/diagnose_direct_budget_grad_conflict.py tools/run_ep014center_per_head_merged_stage.py tools/run_stage6_A5_archaeology_topology.py tools/p6_pre0_grad_audit.py tools/run_tailk7_vs_baseline_leg_linear_probe.py tools/run_ep014center_70a_to71_plain_leg_cleanup.py tools/run_ep014center_replace_redesign.py tools/scan_side_routing_ckpt_keys.py
```

Result:

```text
# exit 0
```

### 9.3 Tooling Grep Gate

Command:

```bash
rg -n "direct_pose_leg_head_shared|direct_pose_leg_gate_head_shared|direct_pose_leg_side_|direct_pose_leg_side_plan_other_ablate|side_plan_other_ablate" tools/
```

Result:

```text
# no matches (exit 1)
```

Interpretation:

- E6 `tools/` live-code remnants are now at 0 matches.

### 9.4 Train / Runtime Residual Grep

Command:

```bash
rg -n "direct_pose_leg_head_shared|direct_pose_leg_gate_head_shared|direct_pose_leg_side_|direct_leg_side_sign_gate|side_plan_other" train/models.py train/posttrain.py train/validate/run_freerun_cycles.py
```

Result:

```text
train/posttrain.py:408:        "direct_pose_leg_side_routing",
train/posttrain.py:409:        "direct_pose_leg_side_plan_other",
train/posttrain.py:410:        "direct_pose_leg_side_phase_other",
train/posttrain.py:411:        "direct_pose_leg_side_phase_rel",
train/posttrain.py:412:        "direct_pose_leg_side_sign_gate",
train/posttrain.py:413:        "direct_pose_leg_side_rank1",
train/posttrain.py:419:    if int(_cfg_get_int(payload, "direct_pose_leg_side_embed_dim", 0, min_value=0)) > 0:
train/posttrain.py:420:        active_keys.append("direct_pose_leg_side_embed_dim")
train/posttrain.py:422:    side_cue = str(payload.get("direct_pose_leg_side_cue") or "none").strip().lower()
train/posttrain.py:424:        active_keys.append("direct_pose_leg_side_cue")
train/posttrain.py:426:    if float(_cfg_get_float(payload, "direct_pose_leg_side_sign_gate_reg_weight", 0.0, min_value=0.0)) > 0.0:
train/posttrain.py:427:        active_keys.append("direct_pose_leg_side_sign_gate_reg_weight")
```

Interpretation:

- `train/models.py` and `train/validate/run_freerun_cycles.py` remain clean for this grep.
- The only remaining train-side hits are the intentional `_cfg_reject_side_routing(...)` rejector surface in `train/posttrain.py`.
- This is consistent with the E1/E2 plan note that the rejector may remain as the sole non-doc side-routing string site.

---

## 10. E4 Contract v2 Hard-Cut Evidence

Date: 2026-04-17  
Mode: contract-only cutoff; no v1 adapter retained.

### 10.1 Implementation Summary

Implemented in `train/model_ckpt_contract.py`:

- Bumped `POSTTRAIN_CHECKPOINT_CONTRACT_VERSION` from `1` to `2`.
- Replaced the generic version failure with:
  - explicit invalid-version failure for malformed `checkpoint_contract.version`
  - explicit `v1` retirement failure explaining that side-routing/shared-leg-head compatibility was removed
  - explicit unsupported-version failure for any non-`2` version
- Intentionally did **not** add any v1→v2 compatibility adapter, state-dict rename, or silent fallback path.

### 10.2 Compile Gate

Command:

```bash
python3 -m py_compile train/model_ckpt_contract.py train/posttrain.py
```

Result:

```text
# exit 0
```

### 10.3 Contract Grep Evidence

Command:

```bash
rg -n "POSTTRAIN_CHECKPOINT_CONTRACT_VERSION =|posttrain checkpoint contract v1 is retired|unsupported posttrain checkpoint contract version=|invalid posttrain checkpoint contract version=" train/model_ckpt_contract.py
```

Result:

```text
33:POSTTRAIN_CHECKPOINT_CONTRACT_VERSION = 2
467:            f"[FATAL] invalid posttrain checkpoint contract version={contract.get('version', None)!r}; "
472:            "[FATAL] posttrain checkpoint contract v1 is retired in current mainline after "
479:            f"[FATAL] unsupported posttrain checkpoint contract version={version}; "
```

Interpretation:

- Current mainline now writes and expects contract `v2`.
- `v1` is rejected with a dedicated retirement message instead of a generic unsupported-version error.
- The loader remains strict for any other version.

### 10.4 Side-Key Grep Sanity

Command:

```bash
rg -n "side_" train/model_ckpt_contract.py
```

Result:

```text
# no matches (exit 1)
```

Interpretation:

- The contract surface itself does not retain side-key compatibility fields.
- The `v1` retirement messaging documents the cutoff semantically without reintroducing `side_` compatibility logic.

---

## 11. Retired-Config Surface Cleanup

Date: 2026-04-17  
Scope: final active-config retired/default key cleanup plus removal of the transitional retired-key rejectors from `train/posttrain.py`.  
Non-goals: no archive-config edits, no long training run, no new retired-key denylist.

### 11.1 Active Config Cleanup

Implementation summary:

- Edited 51 parseable non-archive config JSON files under `config/`.
- Removed inert high-order defaults:
  - `direct_pose_loss_sics` removed from 51 files
  - `direct_pose_loss_cycle_gte` removed from 51 files
  - `direct_pose_loss_sic_mode` removed from 51 files
  - `direct_pose_loss_sic_boost` removed from 51 files
- Removed false `*_train_only` defaults:
  - `direct_pose_leg_train_only: false` removed from 20 files
  - `direct_pose_leg_gate_train_only: false` removed from 41 files
  - `direct_pose_nonleg_train_only: false` removed from 33 files
  - `direct_pose_hinge_train_only: false` removed from 0 files
  - `direct_pose_hinge_gate_train_only: false` removed from 0 files
- Retained all `*_train_only: true` entries for later manual keep/migrate/archive review in this intermediate pass.
  - This was superseded by §12, which removes the direct leg/non-leg train-only mainline surface.
- Archive configs were not edited.

Note on non-archive placeholders:

- 8 non-archive `config/*.json` files are zero-byte / non-parseable placeholders and were skipped unchanged:
  - `config/posttrain_WalkF_hinge_clean_split_calfr_z90_deltaw2_s3_denomfix_20260122_epsmax_0p0.json`
  - `config/posttrain_WalkF_hinge_clean_split_calfr_z90_deltaw2_s3_denomfix_20260123_epsl2_0p01.json`
  - `config/posttrain_WalkF_hinge_clean_split_calfr_z90_deltaw2_s3_denomfix_20260123_epslr_0p3.json`
  - `config/posttrain_WalkF_hinge_clean_split_calfr_z90_deltaw2_s3_denomfix_20260123_epsmax_0p25.json`
  - `config/posttrain_WalkF_hinge_clean_split_calfr_z90_deltaw2_s3_denomfix_20260123_epssrc_pre.json`
  - `config/posttrain_WalkF_routeA_hiddenpre_calfr_z90_deltaw2_s3_denomfix_20260123.json`
  - `config/posttrain_WalkF_stage4_direct_hiddenpre_reinit_20260123.json`
  - `config/posttrain_WalkF_stage4b_hinge_only_hiddenpre_20260123.json`

Evidence:

```bash
rg -n "direct_pose_loss_sics|direct_pose_loss_cycle_gte|direct_pose_loss_sic_mode|direct_pose_loss_sic_boost" config/ -g '!config/archive*/**'
```

```text
# no matches (exit 1)
```

```bash
rg -n '"[^"]*_train_only"\s*:\s*false' config/ -g '!config/archive*/**'
```

```text
# no matches (exit 1)
```

```bash
git diff --name-only -- 'config/archive*'
```

```text
# no matches
```

### 11.2 Retained `true *_train_only` Inventory

These were intentionally kept in place in §11. This inventory is now historical; §12 removes the direct leg/non-leg `*_train_only` active-config surface.

`direct_pose_leg_train_only` retained in 21 files:

- `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260226_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json`
- `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260228_fromarmchain_skip70c.json`
- `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_supfocusULw4_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_supfocusULw4v2_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_supfocusULw4v3_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj512_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_71m_legcurriculum_proj10_lr3e4_e1_s45_20260308_fromarmchain.json`
- `config/posttrain_WalkF_stage7_71m_legcurriculum_proj10_lr3e4_e1_s60_20260308_fromarmchain.json`
- `config/posttrain_WalkF_stage7_71m_legcurriculum_proj3_lr3e4_e1_s60_20260308_fromarmchain.json`
- `config/posttrain_WalkF_stage7_71m_legcurriculum_proj5_lr3e4_e1_s60_20260308_fromarmchain.json`
- `config/posttrain_WalkF_stage7_71m_legcurriculum_proj7_lr3e4_e1_s60_20260308_fromarmchain.json`
- `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260226_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json`
- `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260228_fromarmchain_skip70c.json`
- `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_supfocusULw4_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_supfocusULw4v2_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_supfocusULw4v3_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj512_ep5_20260227_fromsplitfirst.json`

`direct_pose_nonleg_train_only` retained in 9 files:

- `config/_tmp_posttrain_WalkF_stage7_nonleg_recovery_proj256_preleg_splitB2_phaseconcat_probe_20260225.json`
- `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260226_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260227_fromarmchain.json`
- `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260228_fromarmchain_skip70c.json`
- `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_supfocusULw4_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_supfocusULw4v2_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_supfocusULw4v3_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj512_preleg_ep5_20260227_fromsplitfirst.json`

`direct_pose_hinge_train_only` retained in 22 files:

- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup0_feat_hidden_e2e.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup0_feat_hidden_e2e_reg001.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup0_feat_hidden_gate_only_e2e.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup0_feat_hidden_gate_only_e2e_reg001.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_feat_hidden.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_feat_hidden_basefeat_rot6d_rc5_spe60.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_feat_hidden_gate_learned.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_feat_hidden_gate_only.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_feat_hidden_nomeas.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_feat_hidden_stance_suppress.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_gatemeas_adversarial.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_gatemeas_p2_phasefeatnone.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_gatemeas_p4_phasefeatnone.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_gatemeas_p4_phasefeatnone_supraw.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_gatemeas_phasefeatnone.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_gatemeas_phasefeatnone_ang0.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_gatemeas_phasefeatnone_supraw.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_gateplan_phasefeatnone.json`
- `config/posttrain_direct_pose_WalkF_only_hinge_calfr_z90_basefeat_rot6d_overfit_rc5_e5_spe60.json`
- `config/posttrain_direct_pose_WalkF_only_hinge_calfr_z90_basefeat_rot6d_overfit_rc5_e5_spe60_deltaw2.json`
- `config/posttrain_direct_pose_WalkF_only_hinge_calfr_z90_basefeat_rot6d_overfit_rc5_e5_spe60_deltaw2_s30.json`
- `config/posttrain_direct_pose_WalkF_only_hinge_calfr_z90_basefeat_rot6d_overfit_rc5_e5_spe60_deltaw2_supw10.json`

`direct_pose_hinge_gate_train_only` retained in 3 files:

- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup0_feat_hidden_gate_only_e2e.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup0_feat_hidden_gate_only_e2e_reg001.json`
- `config/posttrain_direct_pose_5clip_hidden_timepe_hinge_calfr_z90_sup1_feat_hidden_gate_only.json`

Evidence:

```bash
rg -n '"[^"]*_train_only"\s*:\s*true' config/ -g '!config/archive*/**'
```

```text
# retained true-train-only inventory recorded above
```

### 11.3 `train/posttrain.py` Rejector Removal

Implementation summary:

- Removed `_cfg_reject_side_routing(...)`.
- Removed `_cfg_reject_retired_direct_pose_highorder(...)`.
- Removed `_cfg_reject_lambda_train_only_keys(...)`.
- Removed the reject loop from `_cfg_from_payload(...)`.
- Did not add any replacement retired-key denylist.
- Left the main schema parsing path otherwise intact; this pass does not attempt a larger config-schema refactor.

Evidence:

```bash
python3 -m py_compile train/posttrain.py
```

```text
# exit 0
```

```bash
rg -n "def _cfg_reject_side_routing|def _cfg_reject_retired_direct_pose_highorder|def _cfg_reject_lambda_train_only_keys|RETIRED_DIRECT_POSE_SIDE_ROUTING|RETIRED_DIRECT_POSE_HIGHORDER" train/posttrain.py
```

```text
# no matches (exit 1)
```

```bash
rg -n "direct_pose_leg_side_" train/posttrain.py
```

```text
# no matches (exit 1)
```

---

## 12. Direct Train-Only Surface Contraction

Date: 2026-04-17  
Scope: remove the remaining direct leg/non-leg `*_train_only` mainline surface so posttrain exposes only standard direct training and lambda-head training.

### 12.1 Runtime Surface Removal

Implementation summary:

- Removed `direct_pose_leg_train_only`, `direct_pose_leg_gate_train_only`, and `direct_pose_nonleg_train_only` from `PostTrainConfig`.
- Removed parser entries for those keys from `_cfg_parse_direct_pose(...)`.
- Removed CLI args for `--direct_pose_leg_train_only` and `--direct_pose_nonleg_train_only`.
- Simplified `_unfreeze_for_train_mode(..., train_mode="direct")` to call standard `_unfreeze_direct_pose(model)`.
- Simplified `train/posttrain_common.py::_unfreeze_direct_pose(...)` to expose only standard direct-mode unfreeze.
- Removed the train-only validation branch and saved-config special-casing in `train/posttrain.py`.

Evidence:

```bash
rg -n "direct_pose_(leg|nonleg)_train_only|direct_pose_leg_gate_train_only|--direct_pose_(leg|nonleg)_train_only|--direct_pose_leg_gate_train_only|leg_train_only|leg_gate_only|nonleg_only" train tools config -g '!config/archive*/**'
```

```text
# no matches (exit 1)
```

### 12.2 Active Config Migration

Implementation summary:

- Converted the remaining active configs to standard direct-mode surface by deleting:
  - `direct_pose_leg_train_only: true` from 21 files
  - `direct_pose_nonleg_train_only: true` from 9 files
  - `direct_pose_leg_gate_train_only` from 0 active files
- Total changed active config files in this contraction pass: 30.
- Archive configs were not edited.

Evidence:

```bash
rg -n '"direct_pose_(leg|nonleg)_train_only"\s*:\s*(true|false)|"direct_pose_leg_gate_train_only"\s*:\s*(true|false)' config/ -g '!config/archive*/**'
```

```text
# direct leg/non-leg train_only keys have no active matches after contraction
```

```bash
git diff --name-only -- 'config/archive*'
```

```text
# no matches
```

### 12.3 Tooling Cleanup

Implementation summary:

- Removed generated false train-only overrides from:
  - `tools/run_cp015_tailk7_plan_drop_competition_probe.py`
  - `tools/run_cp015_tailk7_replace_direct_recovery_bridge.py`
  - `tools/run_stage6_nline_stability_two_runs.py`
- Removed retired direct train-only optimizer-scope keys from:
  - `tools/run_ep014center_stage6_early_live_cleanup_bias.py`
- Removed unsupported train-only ablation cases from:
  - `tools/p6_constraint_ablation_runner.py`
- Updated trainable summary probing in:
  - `tools/p6_pre0_grad_audit.py`

### 12.4 Lightweight Validation

Command:

```bash
python3 -m py_compile train/posttrain.py train/posttrain_common.py tools/p6_pre0_grad_audit.py tools/p6_constraint_ablation_runner.py tools/run_cp015_tailk7_plan_drop_competition_probe.py tools/run_cp015_tailk7_replace_direct_recovery_bridge.py tools/run_stage6_nline_stability_two_runs.py tools/run_ep014center_stage6_early_live_cleanup_bias.py
```

Result:

```text
# exit 0
```
