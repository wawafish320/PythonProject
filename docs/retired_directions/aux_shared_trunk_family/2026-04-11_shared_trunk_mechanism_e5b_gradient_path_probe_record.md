# 2026-04-11 shared trunk mechanism E5b gradient-path probe record

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Status: completed  
> Scope: reuse existing `stage6 native` snapshot artifacts only; no new training; no objective change; no downstream  
> Result: **more consistent with shared-trunk capacity / plasticity sink than with strong sign conflict**

## 1. Fixed question

This round asks the minimal E5b question:

> in the already-matched E4 arms, does the harmful `shared_attach_aux` arm show evidence of **direct gradient sign conflict** on the shared trunk, or is the evidence better explained by a **shared-trunk plasticity / capacity sink**?

Arms reused from E4:

1. `shared_attach_aux`
2. `aux_detach`
3. `late_attach_aux`

No new rerun was launched.

## 2. Important blocker found first: step-0 probe is structurally invalid

The initial step-0 probe looked broken because `aux_leg_loss` only updated `direct_pose_aux_leg_head` and produced **zero upstream aux gradients**.

Root cause:

- all three `ckpt_step_000000` snapshots have
  - `direct_pose_aux_leg_head.weight` norm = `0.0`
- therefore at step 0:
  - `d(aux_loss)/d(aux_head_weight) != 0`
  - but `d(aux_loss)/d(aux_in) = W^T g = 0`
  - so no aux gradient can reach the shared trunk yet

Observed zero-init norms:

| arm | `ckpt_step_000000` aux-head weight norm |
| --- | ---: |
| `shared_attach_aux` | `0.0` |
| `aux_detach` | `0.0` |
| `late_attach_aux` | `0.0` |

So E5b must probe **later existing snapshots**, not step 0.

## 3. Actual commands run

Main late-phase probe:

```bash
PYTHONPATH=. python3 debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5b_gradprobe/e5b_aux_gradient_conflict_probe.py \
  --steps 60 \
  --ckpt-steps 420,480
```

The script probes each existing snapshot with teacher-forced single-step autograd over `60` steps and reports:

- `shared_trunk` / `leg_branch` / `main_readout` / `aux_head` grad norms
- `main_nonleg` vs `aux_leg` gradient ratio and cosine
- matched epoch metrics from existing E4 summaries

Primary artifacts:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5b_gradprobe/e5b_gradprobe_metrics.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5b_gradprobe/e5b_gradprobe_metrics.md`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5b_gradprobe/e5b_gradprobe_summary.csv`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5b_gradprobe/e5b_gradprobe_step_rows.csv`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5b_gradprobe/e5b_shared_trunk_ratio_vs_leg_p95.png`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5b_gradprobe/e5b_shared_trunk_cosine_summary.png`

Probe script:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5b_gradprobe/e5b_aux_gradient_conflict_probe.py`

## 3.1 Methodology note: cosine is full-vector, not per-tensor averaged

The reported group cosine is the **full-vector cosine** over the entire parameter group:

- for a parameter group such as `shared_trunk`, each tensor's dot product and squared norm is accumulated first
- then one single cosine is computed from the aggregated totals
- equivalently, this is the same as flattening all gradients in the group, concatenating them into one long vector, and taking one cosine

So the probe does **not** do:

- “compute cosine per parameter tensor, then average cosines”

That averaging scheme could artificially wash out localized conflict toward zero; this probe does not use it.
Local opposition cannot be diluted by tensor-level averaging because there is no tensor-level averaging here.

Implementation note:

- `_pair_stats(...)` in `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5b_gradprobe/e5b_aux_gradient_conflict_probe.py`

## 3.2 Methodology note: zero-init makes step-0 upstream aux gradients identically zero

This was not a probe bug; it is a consequence of the aux-head initialization.

Because `direct_pose_aux_leg_head.weight = 0` at `ckpt_step_000000`:

- aux output starts at zero
- aux loss still gives a non-zero gradient on aux-head weights
- but the Jacobian from aux output back to aux input is multiplied by the zero weight matrix
- therefore `d(aux_loss) / d(aux_hidden) = 0` at step 0

So the aux effect on the shared trunk is **ramped in gradually as the aux head learns non-zero weights**; it is not “full on” from step 0. This is exactly why E5b had to probe later existing snapshots rather than step 0.

## 4. Late-phase checkpoint summary

Key summary (`main_nonleg` vs `aux_leg` on `shared_trunk`):

| arm | ckpt step | epoch | attach | detach | aux-head w norm | aux_leg_loss | leg p95 | all_ex_root p95 | shared aux/main ratio mean | shared cos median | shared cos<0 frac | shared aux grad mean |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_attach_aux` | `420` | `7` | `shared_trunk` | `false` | `2.432092` | `0.059068` | `1.488769` | `0.938242` | `0.173939` | `-0.006195` | `0.5000` | `0.231698` |
| `shared_attach_aux` | `480` | `8` | `shared_trunk` | `false` | `2.597601` | `0.054306` | `1.793149` | `0.990646` | `0.167507` | `0.024988` | `0.4167` | `0.238674` |
| `aux_detach` | `420` | `7` | `shared_trunk` | `true` | `2.445740` | `0.059939` | `1.953524` | `1.118872` | `0.000000` | `nan` | `nan` | `0.000000` |
| `aux_detach` | `480` | `8` | `shared_trunk` | `true` | `2.619786` | `0.054265` | `1.316565` | `0.937262` | `0.000000` | `nan` | `nan` | `0.000000` |
| `late_attach_aux` | `420` | `7` | `leg_boundary` | `false` | `2.614565` | `0.096982` | `1.421675` | `0.931515` | `0.000000` | `nan` | `nan` | `0.000000` |
| `late_attach_aux` | `480` | `8` | `leg_boundary` | `false` | `2.774359` | `0.094440` | `1.438880` | `0.913264` | `0.000000` | `nan` | `nan` | `0.000000` |

Late-attach control on the leg boundary:

| arm | ckpt step | leg_branch aux grad mean |
| --- | ---: | ---: |
| `late_attach_aux` | `420` | `0.223633` |
| `late_attach_aux` | `480` | `0.239714` |

So:

- `shared_attach_aux` carries a persistent aux gradient on the **shared trunk**
- `aux_detach` blocks that path completely while keeping a similar learned aux head
- `late_attach_aux` moves aux pressure off the shared trunk and onto the **leg boundary**

## 5. Readout

### 5.1 What is supported

Supported:

1. `shared_attach_aux` does have a **real late-phase shared-trunk aux gradient path**
   - ratio vs `main_nonleg` is about `0.17`
   - aux-grad mean is about `0.23-0.24`
2. `aux_detach` removes that shared-trunk aux path entirely
   - shared-trunk aux grad = `0`
   - yet epoch-8 `aux_leg_loss` is almost identical to `shared_attach_aux`
3. `late_attach_aux` also removes shared-trunk aux pressure entirely
   - shared-trunk aux grad = `0`
   - aux gradients instead land on `leg_branch`
4. The most harmful endpoint remains `shared_attach_aux` epoch 8:
   - `leg p95 = 1.793149`
   - vs `1.316565` for `aux_detach`
   - vs `1.438880` for `late_attach_aux`

This is the strongest E5b evidence:

- harm tracks **where the aux gradient goes**
- not simply whether the aux head is trainable
- and not simply whether `aux_leg_loss` becomes small

Important limit:

- this cross-arm relation is cleanest at the **epoch-8 endpoint** and in the **gradient-path placement** itself
- it is not perfectly monotonic at every intermediate checkpoint (e.g. `aux_detach` epoch 7 is still worse than `shared_attach_aux` epoch 7)
- so the supported claim is about the **late shared-trunk path mechanism**, not a strict checkpoint-by-checkpoint total ordering

Statistical honesty note for `cos<0 frac`:

- `n = 60` teacher steps per checkpoint
- for a Bernoulli fraction near `0.5`, the binomial standard error is about `sqrt(0.5 * 0.5 / 60) ≈ 0.065`
- therefore `0.5000` and `0.4167` are comfortably within the noise scale of a 50/50 split
- both observed values fall within `1 SE` of `0.5` and are statistically indistinguishable from random

So these fractions should be read as:

- **within noise of 50/50**
- this is a clean null result for sign conflict, not a weak negative-direction trend

not as evidence for a meaningful negative-direction tendency.

### 5.2 What is *not* supported

The data do **not** support a strong sign-conflict reading:

- `shared_attach_aux` shared-trunk cosine medians are near zero, not strongly negative
  - epoch 7: `-0.006`
  - epoch 8: `+0.025`
- negative-cos fraction is only mixed, not dominant
  - epoch 7: `0.50`
  - epoch 8: `0.4167`
- the more leg-specific pair is also not negative:
  - `shared_trunk × main_leg_vs_aux_leg`
  - epoch 7: cos median `+0.016904`, `cos<0 frac = 0.4333`, ratio mean `0.045513`
  - epoch 8: cos median `+0.005224`, `cos<0 frac = 0.4667`, ratio mean `0.048622`

So the observed late shared-trunk aux gradients are:

- **non-zero**
- **persistent**
- but **not strongly anti-aligned**

This is much weaker than the signature expected from a clean “aux gradient directly fights the main gradient” story.
In fact, the `main_leg` pair is a harder refutation than the `main_nonleg` pair:

- if leg harm came from direct directional opposition against leg updates, the shared-trunk `main_leg_vs_aux_leg` cosine should be negative
- instead it is mildly positive at both late checkpoints
- yet `shared_attach_aux` is still the worst rollout arm on `leg p95`

So the harm must come from something beyond first-order directional opposition, which is exactly why the capacity / plasticity sink reading is stronger.

## 6. Interpretation

Current best read, updated in light of later E5a and E5d follow-ups, is hierarchical rather than “one explanation replaces another”:

1. **Primary near mechanism:** harm is mediated by the aux gradient path landing on the shared trunk  
   → this is the cleanest E5b contribution: `shared_attach_aux` is the only harmful arm with persistent late shared-trunk aux pressure, while `aux_detach` and `late_attach_aux` remove that pressure and avoid the same endpoint harm
2. **Mechanistic form:** shared-trunk capacity / plasticity sink  
   → E5b supports this as the leading form because the shared-trunk aux/main ratio is non-trivial (`≈ 0.17`) while full-vector cosine stays near zero and the `main_leg_vs_aux_leg` pair is mildly positive rather than negative
3. **Mechanistic modulator confirmed:** attach mismatch redirects the sink into a less harmful parameter pool  
   → `late_attach_aux` behaves like a natural experiment: the sink is moved from `shared_trunk` to `leg_branch`, and harm weakens correspondingly
4. **Actively rejected as primary:** per-step sign conflict  
   → E5b directly argues against a strong PCGrad / gradient-surgery explanation via full-vector group cosine `≈ 0` and the mildly positive `shared_trunk × main_leg_vs_aux_leg` cosine
5. **Downgraded upstream framing:** supervision–rollout objective mismatch  
   → this remains plausible only as a higher-level story about where the sink lands, but it was weakened by E5a because the E4 `7 -> 8` late reversal did not replicate on seed B
6. **Not primary:** pure saturation / no-usable-signal; broad structural fork / head-side competition

Why this hierarchy holds:

1. `shared_attach_aux` is the only arm with persistent late shared-trunk aux pressure.
2. `aux_detach` and `late_attach_aux` both remove that shared-trunk pressure and also avoid the same level of final harm.
3. The late shared-trunk cosine signal is near zero rather than strongly negative.
4. `late_attach_aux` shows that changing attach does not remove the sink; it **redirects** the sink to a less costly sub-module.
5. Later E5d adds a useful local quantitative confirmation on the `0.0 -> 0.5 -> 1.0` segment: shared-trunk aux-grad size increases, `leg p95` worsens, and `main_leg` cosine remains non-negative throughout. E5d's `1.0 -> 2.0` inversion is currently single-seed and unresolved against seed noise, so it does not overturn the primary E5b path argument.

So E5b moves the explanation away from:

- “the aux gradient is mostly pointing in the wrong direction”

and toward:

- “the aux objective occupies shared-trunk update budget / representational plasticity in a way that does not help rollout.”

This also sharpens the relationship to E3:

- E3 established attach mismatch as a **modulator**
- E5b supplies the mechanism:
  - attach changes **where the sink lands**
  - `late_attach_aux` helps because it redirects the sink away from the shared trunk and into a less harmful sub-module

Honest caveat:

- the `cos ≈ 0` + non-trivial-ratio + worse-rollout combination makes capacity / plasticity sink the **leading** interpretation
- but it is not mathematically exclusive
- for example, curvature / Hessian effects could still modulate update efficiency without appearing as strong first-order sign conflict

So E5b should be read as:

- **shared-trunk path mediation is directly supported**
- **capacity / plasticity sink is the leading mechanistic form**
- **sign conflict is actively downgraded**

## 7. Decision

Updated chain decision after E5a + E5d:

- `E5a-downstream`: **skip**
  - E5a already removed the robust within-arm reversal pillar
  - E5b/E5d are mechanism diagnostics, not downstream-improvement evidence
- `E5c`: **remain paused**
  - not just because evidence is incomplete
  - also because there are already cheaper alternatives that avoid harm without objective redesign:
    - `aux_detach`
    - `late_attach_aux`
  - a rollout-aware aux objective is a materially larger engineering investment whose expected value is bounded above by the best cheap alternative unless a new result specifically shows those alternatives cannot deliver

Bottom line:

> E5b does not support a strong sign-conflict hypothesis.  
> It most strongly supports a shared-trunk gradient-path-mediated harm story, with capacity / plasticity sink as the leading mechanistic form, attach mismatch as sink redirection, and supervision–rollout mismatch downgraded to a weaker upstream framing rather than a load-bearing pillar.
