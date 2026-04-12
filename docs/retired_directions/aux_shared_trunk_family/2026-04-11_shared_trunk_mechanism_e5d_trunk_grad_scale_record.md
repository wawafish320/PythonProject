# 2026-04-11 shared trunk mechanism E5d trunk-grad scale record

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Status: completed  
> Scope: only `E5d shared-trunk aux-grad scale probe`; reuse `aux_detach` as `0.0`, reuse `shared_attach_aux` as `1.0`, add only matched reruns `0.5` and `2.0`  
> Result: **actual shared-trunk aux-grad size is cleanly monotonic; the `0.0 -> 0.5 -> 1.0` segment supports a monotonic sink-size ↔ harm relation within one seed, while the `1.0 -> 2.0` inversion is single-seed and unresolved against measured seed noise**

## 1. Fixed question

This round asks only:

> if we change only the aux-loss gradient strength that flows back into the `shared_trunk`, does rollout harm change monotonically with that actual shared-trunk aux-grad size?

Per instruction, this record **reuses** the existing E4/E5a/E5b conclusions and does not re-litigate them:

- `docs/retired_directions/aux_shared_trunk_family/2026-04-11_shared_trunk_mechanism_e4_epochwise_aux_rollout_mismatch_record.md`
- `docs/retired_directions/aux_shared_trunk_family/2026-04-11_shared_trunk_mechanism_e5a_seed_reversal_check_record.md`
- `docs/retired_directions/aux_shared_trunk_family/2026-04-11_shared_trunk_mechanism_e5b_gradient_path_probe_record.md`

The new points here are only:

1. `0.5` shared-trunk aux-grad scale
2. `2.0` shared-trunk aux-grad scale

Reference points:

1. `0.0 = aux_detach`
2. `1.0 = shared_attach_aux`

## 2. Artifact check first

Searched for existing E5d-style artifacts under:

- `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_trunkscale05_epochsnap`
- `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_trunkscale20_epochsnap`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale`

Result:

- no complete reusable E5d artifact set existed before this round
- so exactly two matched reruns were required: `0.5`, `2.0`

No extra seed sweep, objective redesign, attach sweep, or downstream run was added.

## 3. Minimal implementation added

There was no existing trunk-only aux-grad scale knob, so a minimal one was added.

### 3.1 Knob semantics

New knob:

- `direct_pose_aux_leg_trunk_grad_scale`
- default: `1.0`

Semantics:

- forward remains identity
- backward scales only the gradient flowing from `direct_pose_aux_leg` back into the `shared_trunk`
- aux-head forward value is unchanged
- aux-head parameter gradients are therefore kept on the original forward value
- `leg_boundary` attach is unaffected
- default behavior is unchanged for all existing runs

Implementation path:

- when `direct_pose_aux_leg_attach == "shared_trunk"`, the aux input is wrapped by an identity-forward / scaled-backward transform before entering `direct_pose_aux_leg_head`
- `direct_pose_aux_leg_detach_feat=true` still zeroes the upstream path as before

### 3.2 Code changes

Changed files:

- `train/models.py`
- `train/posttrain.py`
- `tests/train/test_posttrain_direct_pose_aux_leg.py`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_trunk_grad_scale_analysis.py`

Focused validation:

```bash
python3 -m py_compile train/models.py train/posttrain.py tests/train/test_posttrain_direct_pose_aux_leg.py
python3 -m unittest tests.train.test_posttrain_direct_pose_aux_leg
python3 -m py_compile debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_trunk_grad_scale_analysis.py
```

Result:

- all passed

## 4. Actual commands run

### 4.1 Train rerun: `0.5`

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_trunkscale05_epochsnap.json \
  --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_trunkscale05_epochsnap \
  --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sharedaux_trunkscale05_epochsnap_20260411 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 4.2 Train rerun: `2.0`

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_trunkscale20_epochsnap.json \
  --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_trunkscale20_epochsnap \
  --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sharedaux_trunkscale20_epochsnap_20260411 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 4.3 Eval + probe + aggregation

```bash
PYTHONPATH=. python3 debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_trunk_grad_scale_analysis.py \
  --run-evals \
  --ckpt-steps 420,480 \
  --steps 60

PYTHONPATH=. python3 debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_trunk_grad_scale_analysis.py \
  --ckpt-steps 420,480 \
  --steps 60
```

The first command generated missing freerun summaries for the new `0.5` / `2.0` arms at steps `420` and `480`; the second regenerated the final summary after a debug-only csv-writer fix in the analysis helper.

## 5. Primary artifacts

Configs:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_trunkscale05_epochsnap.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_trunkscale20_epochsnap.json`

Model roots:

- `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_trunkscale05_epochsnap`
- `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_trunkscale20_epochsnap`

Eval / probe / plots:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_trunkscale_metrics.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_trunkscale_summary.md`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_final_summary.csv`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_gradient_summary.csv`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_probe_step_rows.csv`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_configured_scale_vs_leg_p95.png`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_aux_loss_and_leg_p95_by_scale.png`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_configured_scale_vs_all_ex_root_p95.png`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/e5d_configured_scale_vs_main_leg_cosine.png`

New eval roots:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/shared_attach_aux_trunkscale05`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5d_trunkscale/shared_attach_aux_trunkscale20`

## 6. Method note

The reported cosine remains the same E5b methodology:

- **full-vector cosine**
- not per-tensor averaged cosine

So each group cosine is equivalent to flattening the whole parameter group into one long vector and computing one cosine on that concatenated vector.

## 7. Final endpoint table (`ckpt_step_000480`)

| scale | arm | aux_leg_loss | aux_leg_over_main | leg mean | leg p95 | all_ex_root mean | all_ex_root p95 | shared ratio mean | shared cos median | shared cos<0 frac | shared aux grad mean | main_leg ratio mean | main_leg cos median | main_leg cos<0 frac |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0` | `aux_detach` | `0.054265` | `0.028465` | `0.597226` | `1.316565` | `0.262125` | `0.937262` | `0.000000` | `nan` | `nan` | `0.000000` | `0.000000` | `nan` | `nan` |
| `0.5` | `shared_attach_aux_trunkscale05` | `0.054436` | `0.028668` | `0.583757` | `1.366734` | `0.249303` | `0.909538` | `0.085791` | `0.019403` | `0.4167` | `0.122716` | `0.025174` | `0.007894` | `0.4833` |
| `1.0` | `shared_attach_aux` | `0.054306` | `0.028285` | `0.708246` | `1.793149` | `0.268797` | `0.990646` | `0.167507` | `0.024988` | `0.4167` | `0.238674` | `0.048622` | `0.005224` | `0.4667` |
| `2.0` | `shared_attach_aux_trunkscale20` | `0.053218` | `0.028618` | `0.675842` | `1.608886` | `0.244532` | `0.907650` | `0.405979` | `0.082672` | `0.2333` | `0.485031` | `0.101267` | `-0.031666` | `0.5833` |

## 8. Gradient-path summary (`420`, `480`)

| scale | ckpt | shared ratio mean | shared cos median | shared cos<0 frac | shared aux grad mean | main_leg ratio mean | main_leg cos median | main_leg cos<0 frac | leg p95 | all_ex_root p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0` | `420` | `0.000000` | `nan` | `nan` | `0.000000` | `0.000000` | `nan` | `nan` | `1.953524` | `1.118872` |
| `0.0` | `480` | `0.000000` | `nan` | `nan` | `0.000000` | `0.000000` | `nan` | `nan` | `1.316565` | `0.937262` |
| `0.5` | `420` | `0.085288` | `0.032553` | `0.4167` | `0.117790` | `0.024265` | `0.000909` | `0.4833` | `1.585469` | `1.095694` |
| `0.5` | `480` | `0.085791` | `0.019403` | `0.4167` | `0.122716` | `0.025174` | `0.007894` | `0.4833` | `1.366734` | `0.909538` |
| `1.0` | `420` | `0.173939` | `-0.006195` | `0.5000` | `0.231698` | `0.045513` | `0.016904` | `0.4333` | `1.488769` | `0.938242` |
| `1.0` | `480` | `0.167507` | `0.024988` | `0.4167` | `0.238674` | `0.048622` | `0.005224` | `0.4667` | `1.793149` | `0.990646` |
| `2.0` | `420` | `0.347972` | `0.019935` | `0.4167` | `0.472044` | `0.095579` | `0.005135` | `0.4833` | `1.402979` | `0.923881` |
| `2.0` | `480` | `0.405979` | `0.082672` | `0.2333` | `0.485031` | `0.101267` | `-0.031666` | `0.5833` | `1.608886` | `0.907650` |

## 9. Readout

### 9.1 What is clearly supported

Supported:

1. the new knob really changes the **actual shared-trunk aux-grad size**
   - endpoint shared ratio is monotonic: `0.000 -> 0.0858 -> 0.1675 -> 0.4060`
   - endpoint shared aux-grad mean is also monotonic: `0.000 -> 0.1227 -> 0.2387 -> 0.4850`
   - same direction appears at the `420` sanity checkpoint
2. `aux_leg_loss` remains readable at all four points
   - endpoint range is tight: `0.0532 .. 0.0544`
   - so the scale manipulation is **not** just “turning aux off”
3. from `0.0 -> 0.5 -> 1.0`, `leg p95` does worsen with actual shared-trunk aux-grad size
   - `1.3166 -> 1.3667 -> 1.7931`
4. over that same `0.0 -> 0.5 -> 1.0` segment, `shared_trunk main_leg_vs_aux_leg` cosine is still not negative
   - `0.5 @480`: `+0.0079`
   - `1.0 @480`: `+0.0052`
   - so that local worsening still points away from strong sign conflict

### 9.2 What is *not* robustly supported yet

Not robustly supported yet:

1. a clean global monotonic harm law over `0.0 / 0.5 / 1.0 / 2.0`
   - within seed A, `2.0` keeps increasing actual shared-trunk ratio strongly
   - but `leg p95` improves relative to `1.0`:
     - `1.7931 -> 1.6089`
2. a matching monotonic `all_ex_root p95` law
   - endpoint `all_ex_root p95`: `0.9373 -> 0.9095 -> 0.9906 -> 0.9077`
3. a strong sign-conflict rescue at high scale
   - `2.0 @480` gives only a **mild** negative `main_leg` cosine: `-0.0317`
   - but the worst rollout harm is still at `1.0`, not `2.0`
   - so this does not produce a coherent sign-conflict story

### 9.3 Important seed-noise caveat on the `1.0 -> 2.0` inversion

This record must be read together with E5a-seed.

Relevant comparison:

- `scale 1.0 seed A vs seed B` final `leg p95` gap from E5a-seed:
  - `1.793149 - 1.669341 = 0.1238`
- `scale 1.0 vs scale 2.0`, both on seed A in this record:
  - `1.793149 - 1.608886 = 0.1843`

So the observed `1.0 -> 2.0` improvement in E5d is only modestly larger than the measured seed-to-seed spread at `scale = 1.0` itself.

Therefore E5d cannot yet distinguish:

1. a true non-monotonic dose-response curve with `1.0` near a pessimal crossing
2. a single-seed trajectory idiosyncrasy where seed A happens to place `scale = 1.0` in a worse basin

This matters because all four scale points in E5d use the same seed-A chain.

So the most defensible read is:

- the `0.0 -> 0.5 -> 1.0` segment is real same-seed evidence
- the `1.0 -> 2.0` inversion is currently **single-seed and unresolved**

Any `scale = 2.0` interpretation should therefore be treated as single-seed and deferred pending a seed-B replication of `scale = 2.0`.

## 10. Explicit judgment on `main_leg_vs_aux_leg`

Required call:

- for `0.5` and `1.0`, `shared_trunk main_leg_vs_aux_leg` remains **micro-positive / near zero**
- for `2.0`, `420` is still micro-positive (`+0.0051`) and `480` becomes only mildly negative (`-0.0317`)

Interpretation:

- the key `0.0 -> 0.5 -> 1.0` worsening segment still happens **without** negative `main_leg` cosine
- that continues to support **capacity / representation interference / sink** over direct sign conflict
- the mild `2.0 @480` negativity is too small and too late to explain the whole pattern, especially because `2.0` is *less* harmful than `1.0`

## 11. Answers to the round's key questions

### Q1. Does actual shared-trunk aux/main ratio increase with configured scale?

Yes.

At `480`:

- `0.0`: `0.000000`
- `0.5`: `0.085791`
- `1.0`: `0.167507`
- `2.0`: `0.405979`

So the actual gradient-path readout gives a clean size axis.

### Q2. Does `leg p95` worsen monotonically with actual shared-trunk aux-grad size?

Within seed A, it worsens monotonically on `0.0 -> 0.5 -> 1.0`.

Observed endpoint sequence:

- `1.316565 -> 1.366734 -> 1.793149 -> 1.608886`

So:

- **yes**, on the `0.0 -> 0.5 -> 1.0` segment
- **unresolved**, for the `1.0 -> 2.0` step, because that inversion is measured on the same seed and is only modestly larger than the already-measured `scale = 1.0` seed spread from E5a-seed

### Q3. Does `all_ex_root p95` follow the same direction?

No.

It is non-monotonic and, on this seed, nonzero scales can even outperform `aux_detach`:

- `0.937262 -> 0.909538 -> 0.990646 -> 0.907650`

That makes `all_ex_root p95` a potentially interesting practical signal, but it is also single-seed and should not be over-read before replication.

### Q4. Does `aux_leg_loss` remain readable rather than getting destroyed?

Yes.

The endpoint aux loss stays almost flat:

- `0.054265 / 0.054436 / 0.054306 / 0.053218`

So the manipulation keeps aux active.

### Q5. Did capacity sink gain a clean quantitative size axis?

Only partially, and in a segment-local sense.

- **yes** for the actual shared-trunk gradient-path magnitude
- **yes** for a same-seed monotonic harm relation on `0.0 -> 0.5 -> 1.0`
- **no / unresolved** for a global monotonic harm law across `0.0 .. 2.0`

So E5d supports a real aux-grad size axis in the shared trunk, and locally supports a quantitative sink reading on `0.0 -> 0.5 -> 1.0`; it does **not** yet show that the full `0.0 .. 2.0` range obeys a clean global dose-response law.

### Q6. Should E5c now be redefined as “sink redirection” rather than “objective mismatch fix”?

E5c should remain paused.

The practical reason is now stronger than “evidence not yet perfect”:

- from E1–E5d, aux-leg supervision on the shared trunk does **not** already justify an expensive rollout-aware objective redesign
- there are already two cheap alternatives that avoid harm without touching the objective:
  - `aux_detach`
  - `late_attach_aux`
- any rollout-aware aux objective would be a materially larger engineering investment
- its expected value is therefore bounded above by the best cheap alternative unless a new result shows that `detach` / `late_attach` cannot deliver the improvement we want

If E5c is discussed later, the safe phrasing is still:

- “possibly redirect a shared-trunk sink that is real, but whose quantitative response curve is not yet settled”

### Q7. Does the non-monotonic result imply the sink is more qualitative than simple size control?

Not decisively.

More precise read:

- a real shared-trunk sink size axis in the probe
- a same-seed monotonic harm relation on `0.0 -> 0.5 -> 1.0`
- and an unresolved `1.0 -> 2.0` inversion that could reflect either:
  - a true nonlinear / regime-dependent response curve
  - or seed-specific trajectory noise

So a simple “bigger shared-trunk aux-grad => strictly worse rollout” rule is not supported yet, but neither is a strong claim that the sink is merely qualitative.

## 12. Bottom line

E5d lands as:

- **robustly positive** for “the shared trunk really sees a controllable aux-grad size axis”
- **robustly positive, but segment-local** for “within one seed, harm tracks that size on `0.0 -> 0.5 -> 1.0`”
- **unresolved** for the `1.0 -> 2.0` inversion, because its magnitude is only modestly larger than the measured scale-1 seed variance from E5a-seed

So the strongest honest update is:

1. sign conflict is still not the best explanation
2. on `0.0 -> 0.5 -> 1.0`, a shared-trunk sink is strengthened by same-seed quantitative evidence while `main_leg` cosine remains non-negative
3. the `1.0 -> 2.0` inversion should be treated as **single-seed and unresolved**, not as a strong falsification of a size axis
4. therefore E5c should remain paused, both because the evidence is not yet strong enough to justify redesign and because cheaper alternatives (`aux_detach`, `late_attach_aux`) already exist

## 13. Post-E5a + E5b + E5d hierarchy

Current combined hierarchy across the recent chain is:

1. **Primary near mechanism:** gradient-path-mediated harm when aux attaches to the shared trunk
   - supported by E1, E5b cross-arm path placement, and the E5d `0.0 -> 0.5 -> 1.0` segment
2. **Mechanistic form, locally supported:** shared-trunk capacity / plasticity sink
   - supported locally by the E5d `0.0 -> 0.5 -> 1.0` quantitative segment
   - not yet promoted to a full global dose-response law because `1.0 -> 2.0` is unresolved under seed noise
3. **Mechanistic modulator:** attach mismatch as sink redirection into a less costly parameter pool
   - supported by E3 and E5b `late_attach_aux`
4. **Actively rejected as primary:** per-step sign conflict
   - weakened by E5b full-vector cosines and further weakened by E5d's mildly-positive `main_leg` cosine on the `0.0 -> 0.5 -> 1.0` segment
5. **Downgraded:** supervision–rollout objective mismatch
   - E5a-seed removed the robust within-arm late-reversal pillar
   - what remains is mainly the E4 cross-arm endpoint mismatch, which is also well explained by items (1)–(3)
6. **Not primary:** pure capacity saturation / no-usable-signal; structural fork / head-side competition
