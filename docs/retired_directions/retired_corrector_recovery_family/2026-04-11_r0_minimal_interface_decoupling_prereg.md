# 2026-04-11 R0 minimal interface-decoupling prereg

> Archived on 2026-04-12.  
> Current role: prereg kept only as historical scope control for the retired R0 minimal-corrector track.  
> Reader guidance: any “current ...” wording below refers to the then-live old-boundary recipe around 2026-04-11, not current canonical posttrain policy.

> Status: prereg / do not reinterpret after seeing results  
> Scope: minimal R0 only; not full universal redesign

## 1. R0 scope

- Donor: `E1-top3` only.
- Corrector: fresh-init residual corrector.
- Donor: frozen.
- Observable: detached before the corrector.
- Forbidden inputs: no donor hidden, no trunk activation, no new semantic channel.
- Out of scope: no `E2A-R`, no bad `top7`, no mixed donor, no multi-donor matrix.

## 2. Zero-correction start

Rule:

```text
y = y_donor + gate * residual(obs)
```

- Use a learned residual gate initialized at exactly `0`.
- Residual body/head are normally initialized.
- The gate is the only identity-start mechanism for R0.
- Step0 forward must be strictly equal to donor/base under the residual path.
- Implementation guard: when all residual omegas are exactly zero, the SO(3) adjustment helper must short-circuit and return the unmodified base tensor, avoiding reproject-only drift.

## 3. Observable tensor set

R0 may only use detached versions of behavior-space tensors already produced/consumed by the then-live locked replace path:

- `arm_base_rot6d` from donor `out_direct`
- `contacts_plan`
- `contacts_meas`
- `contacts_err`
- `event_clock_delta_meas`
- `event_clock_lambda_corr`
- `event_clock_delta_z`
- existing period / phase-event / direct time-PE observables

No new observable channel is introduced in R0. New channels, canonicalizers, donor hidden features, or trunk activations are reserved for R1+ only if R0 passes.

## 4. Arms

- `baseline_locked`: then-current locked-contract replace recipe, run on the `E1-top3` donor warmstart.
- `sham_lr0`: residual corrector structure present; corrector/gate parameter group `lr=0`.
- `branch`: same residual corrector structure; corrector/gate train normally.

## 5. No-op identity sanity

- Primary sanity: `baseline_locked` vs `sham_lr0` must be numerically equivalent within tolerance on headline metrics.
- Tolerance for headline metrics: `<= max(1%, 0.002)`.
- If no-op identity fails, classify as `pipeline/data-path leak` and do not interpret branch gains.
- Supplemental mechanistic check: `donor_noop` (same frozen donor warmstart, no residual training) vs `sham_lr0` should also be near-equal; if this fails it is direct evidence of frozen-path / data-path leakage even before comparing to the locked baseline.

## 6. Gradient-path hard assert

- Donor / shared trunk / shared head parameters must remain frozen in residual arms.
- All corrector observables must be detached before entering the corrector.
- After branch backward, donor/shared non-corrector params must have `grad is None` or all-zero gradients.
- If gradient reaches donor/shared trunk/shared head/non-corrector params, R0 is invalid.

## 7. Acceptance gates

- Headline metrics: `all_ex_root mean`, `all_ex_root p95`.
- Veto metric: `leg p95`.
- Identity sanity: `baseline_locked` vs `sham_lr0` headline metric difference `<= max(1%, 0.002)`.
- Branch positive: `branch` vs `baseline_locked` improves `>= 3%`, and `branch` vs `sham_lr0` improves `>= max(2%, 0.005)`.
- Veto: if `branch` leg p95 worsens by `> 3%`, do not promote.

## 8. Branch-vs-sham readout

- Branch positive vs both `baseline_locked` and `sham_lr0`: pass; can expand to the donor matrix later.
- Branch improves over `baseline_locked` but is approximately equal to `sham_lr0`: structural fragility branch, not observable fail.
- Branch worse than both `baseline_locked` and `sham_lr0`: objective / observable / minimal capacity fail.
- `baseline_locked` not approximately equal to `sham_lr0`: pipeline/data-path leak; stop and debug.

## 9. Planned commands / artifacts

- Donor source: `models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth`
- Locked baseline recipe source: `debug_output/_tmp_cp015_tailk7_replace_schedule_ablation_20260402/configs/posttrain_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.json`
- New R0 artifact root: `debug_output/_tmp_r0_minimal_interface_decoupling_20260411`
- New model root: `models/__tmp_r0_minimal_interface_decoupling_20260411`
