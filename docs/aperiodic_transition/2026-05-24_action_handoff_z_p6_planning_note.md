> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §3.4/§9 under its stated read-only / zero-new-injection scope.

# Action Handoff z P6 Planning Note (2026-05-24)

## 1. Status

- H3 is partially supported under the recalibrated P4-alt yardstick.
- v1 is not fully passed.
- P1 magnitude-regression risk remains open.
- P6 is not implemented yet.

## 2. Why P6 planning is now justified

- P4-alt sweep global stability is pass-like in 7/10 configs (`debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.md`).
- All five sources are majority pass-like in the same sweep (`Walk_F` 10/10, `Walk_L_To_L` 10/10, `Walk_L_To_R` 7/10, `Walk_R_To_L` 10/10, `Walk_R_To_R` 9/10).
- Internal diagnostics indicate z has usable structure, not pure noise (`debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_20260524/internal_structure_v2_summary.md`):
  - phase locality: strong
  - low-dimensionality: strong
  - turn-end monotonic convergence: strong
  - cross-turn end tightness: strong

## 3. Known weak-source risk

Stable weakest source in P4-alt sweep is `Walk_L_To_R` (`debug_output/_tmp_action_handoff_z_probe_v1_walk_l_to_r_failure_analysis_20260524/walk_l_to_r_failure_analysis.md`):

- `mean_ratio=0.940338`
- `mean_spearman=0.232160`
- `near_random=0/10`

Long-horizon degradation:

- short-horizon `mean_spearman=0.326221`
- long-horizon `mean_spearman=0.012683`

Target-specific weak pairs:

- `Walk_L_To_R -> Walk_R_To_L`: `mean_ratio=1.011248`, `mean_hit_lift=0.010950`
- `Walk_L_To_R -> Walk_R_To_R`: `mean_hit_lift=-0.016090`

## 4. Source-level hypotheses (data/representation)

These are root-cause hypotheses for planning and diagnostics, not implementation fixes:

- direction-change ambiguity / mirrored turn-family asymmetry: `L_To_R` source may map to multiple mirrored futures with weak separability in current descriptors.
- insufficient long-horizon predictive signal in `hidden_pre` for `L_To_R` source: short-horizon ranking signal survives, but long-horizon ordering collapses.
- target-pair `future_desc` divergence at `N>=24`: target futures for `Walk_R_To_L` / `Walk_R_To_R` may branch in ways not preserved by current z neighborhood geometry.
- contact/foot/root vs `future_desc` mismatch: current future-equivalence descriptor may underrepresent the cues that dominate safe turn-direction entry decisions.
- single-clip / no multi-cycle limitation: per-clip temporal support may be insufficient for stable cycle-closure-style or long-range branch discrimination.
- z objective bias toward short-horizon equivalence: `L_predict + InfoNCE` weighting may emphasize near-term equivalence more than long-horizon branching discrimination.

## 5. P6 stress-case plan (planning only)

First P6 planning scope should explicitly include:

- normal cases where P4-alt is strong
- `Walk_L_To_R -> Walk_R_To_L`
- `Walk_L_To_R -> Walk_R_To_R`
- `N>=24` long-horizon warning/fallback logging

Required P6 report fields:

- selected `source/target` pair
- z distance / rank / margin
- future-equivalence score (if available)
- fallback / no-good-candidate flag
- contact / foot / root / pose safety metrics from existing P6 priority contract

## 6. Decision boundary

- If P6 failures mainly concentrate on known weak pairs (`Walk_L_To_R -> Walk_R_To_L`, `Walk_L_To_R -> Walk_R_To_R`), treat this as source-specific failure, not global framework rejection.
- If P6 also fails on strong P4-alt pairs, framework/runtime route is not ready.
- If P6 passes strong cases while weak pairs are controllable via explicit fallback policy, v1 can proceed with fallback policy.

## 7. Non-goals

- No P6 code implementation in this note.
- No z retraining in this note.
- No beta/Dz ablation in this note.
- No C naturalness-prior family decision in this note.

## Artifact Cross-links

- `debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.md`
- `debug_output/_tmp_action_handoff_z_probe_v1_walk_l_to_r_failure_analysis_20260524/walk_l_to_r_failure_analysis.md`
- `debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_20260524/internal_structure_v2_summary.md`
- `debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md`
