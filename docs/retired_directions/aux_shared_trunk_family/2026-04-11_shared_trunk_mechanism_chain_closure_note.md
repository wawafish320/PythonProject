# 2026-04-11 shared trunk mechanism chain closure note

> Status: retired closure memo / historical mechanism record
> Current role: archaeology / mechanism evidence only
> Do not use this document as the current shipping decision.
> Later downgrade / reevaluation:
> - `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

## 1. Final action

Historical action after E1–E6:

- at that time, the chain closed with `aux_detach` as the family-local default inside the aux-family mechanism track

Why the chain closes here:

- E6 already resolves the only remaining practical question
- `shared_attach_aux trunkscale05` did not produce a positive shipping case over `aux_detach`
- further mechanism refinements would not change the historical ship decision inside that chain

This is therefore a **closure memo**, not a new experiment record.

## 2. Final mechanism hierarchy

Final working hierarchy for this chain:

1. **Primary near mechanism:** gradient-path-mediated harm when aux pressure lands on the shared trunk
2. **Most plausible local form:** shared-trunk plasticity / capacity sink
3. **Important modulator:** attach mismatch can redirect the sink into a less costly parameter pool
4. **Not primary:** per-step sign conflict
5. **Downgraded:** rollout/objective mismatch as the main explanation

Seed-sensitivity note from E5d + E6:

- the local `scale 1.0 -> 0.5` improvement direction replicated across seed A and seed B
- but the effect size (`~0.05 .. 0.08` on `all_ex_root p95`) is much smaller than the observed cross-seed baseline shift (`~0.23 .. 0.26`)
- therefore the quantitative scale story is seed-sensitive and was not enough, even at the time, to move away from `aux_detach`

## 3. Paused / de-scoped directions

The following directions are now paused unless a future action-relevant trigger appears:

- `E5c` rollout-aware objective redesign
- `E5d-seed-scale2`
- `E6 Stage B` (`seed-B aux_detach`) as a mechanism-only completion
- `E7+` aux-scale sweeps / attach sweeps / further mechanism refinement

Reason:

- they may refine explanation quality
- but they were not expected to change the historical family-local ship decision

## 4. Reopen criteria

Reopen this chain only if all of the following become true:

- downstream work shows `aux_detach` is insufficient for the concrete target
- the remaining gap is action-relevant rather than purely explanatory
- root-cause evidence points back to shared-trunk plasticity / routing, rather than some unrelated downstream bottleneck

Concrete example:

- if a future `70a` / `70b` objective requires better Stage6 behavior than `aux_detach` can provide, and the deficit is traced back to shared-trunk aux routing, then this chain can be reopened with a new decision target

## 5. Closure summary

The historical practical conclusion inside this chain was:

- keep `aux_detach` as the aux-family-local default
- stop the mechanism chain here
- do not reopen it just to finish an explanatory tree that no longer affects action

Current reading:

- this memo is still useful as mechanism evidence
- it is **not** the current mainline ship decision
- the later downstream reevaluation is the document that settled non-mainline status for the feature family
