# Phase Disambiguation Bridge: Paper Fourier-Phase vs Our Phase Hints (`contacts_*` / `phase_z_in`)

> Last updated: 2026-01-25  
> Context: MLPL2_DirectBranch_v1 + Stage2 λ + contact loop (plan/meas/err) + Stage4 direct (hinge is debug-only) + td_hazard phase reset + prev_phase_vec clock (contact_phase_state) + `phase_z_in -> direct` bridge
> Status (2026-02-28): **legacy repro-only**. For current default pipeline, use `docs/posttrain_pipeline.md`.
> Update (2026-03-01): legacy posttrain targets are retired on current mainline `train.posttrain`; commands in this doc that rely on `train_so3_corrector/train_contact_plan*/train_contact_meas/train_contact_td_hazard` are archival and may fail unless replayed on historical snapshots.

This document “bridges” the conceptual gap between:

1) **Paper-style phase conditioning**: `phase = FourierFit(GT)` and `direct(phase) -> pose`  
2) **Our system**: `contacts_plan` (GRU(cond)) + `contacts_meas` (learned `contact_meas_head`) used for **Round0 phase disambiguation** and **long-horizon drift suppression**

It also records the *concrete* experiment evidence from this repo (paths + numbers) that motivates the next iterations.

---

## 0. Problem Statement (What “Round0 accuracy” actually means here)

In `freerun_cycles`, **Round0** starts from a GT-aligned initial state (teacher clip start). In this regime:

- **Incremental** (`inc`) is effectively a *one-step conditional predictor* from a correct `y_prev` → it is naturally strong early.
- **Direct** (`dir`) in our current architecture is a *prior*:
  - `direct_pose_head` does **not** see `y_prev`/full pose history
  - it relies on `(cond, contacts_plan)` (and implicitly whatever “phase proxy” plan encodes)

Therefore, if direct lacks a reliable phase disambiguation input at t=0, it will learn a **multi-modal mean** (“average phase”), which is systematically worse than inc in Round0.

> Note (2026-01-23): the "direct is a prior" framing above is the **cond-only** route (`direct_pose_feat_source="cond"`). In the current posttrain pipeline we typically use `direct_pose_feat_source="hidden_pre"` (pre-PASA `h_temporal`) for direct; hinge is treated as **debug-only**. Keep this setting consistent across Stage4/5 to avoid semantic mismatch when retraining λ.

---

## 1. Paper Baseline vs Our Setup (Key Differences)

### 1.1 Paper: clean phase is *primary input*

- Phase source: `phase = FourierFit(GT)` (always “clean”, drift-free)
- Direct target: learn a mapping `pose = f(phase)` (phase is the core input)

Implications:
- Round0: phase is correct → direct can be correct
- Round1+: phase stays correct (because it’s from GT / fitted trajectory) → direct remains stable

### 1.2 Our system: `contacts_meas` is *derived*, and phase is *implicit*

We have two phase-like signals:

- `contacts_plan`: produced by a GRUCell from `cond` (optionally + time-PE bias)
- `contacts_meas`: produced by `contact_meas_head` from current `state_t` (causal; no pose history).  
  As of 2026-01, this repo uses the redesigned lower-body/no-hist meas head by default
  (see `docs/contact_meas_head_redesign_lowerbody_nohist.md`).

Key difference to paper:
- **We do not have a clean, explicit phase label**; we only have proxies (contacts) that can be:
  - clean at Round0 (because pose is GT-aligned at t=0)
  - drift-corrupted at later rounds if derived from predicted pose

---

## 2. The User’s Hypothesis (and why it’s mostly correct)

Hypothesis:
- `contacts_meas` is a *low-dimensional phase indicator* (2D for L/R foot), so it can disambiguate “left-stance vs right-stance” at Round0.
- Feeding `contacts_meas` to `direct` can solve Round0 ambiguity without giving direct the whole `y_prev` (which would break error-orthogonality assumptions).

This is correct **under two conditions**:

1) **`contacts_meas` must be stable / near-memoryless** at the timescale we use it (avoid hidden-state-like drift amplification).
2) **The direct head must be trained to treat the phase hint as *high-bandwidth* signal** for disambiguation (not just a weak “micro adjustment”).

---

## 3. FiLM “bandwidth” concern: is `contacts_meas` only a weak hint?

Your analysis is correct:

- If we implement: `direct = FiLM(backbone(cond, plan), hint=meas)`  
  and training makes the model solve most of the task via `(cond, plan)` with `meas` as tiny correction,
  then FiLM can become a **low-bandwidth bottleneck** and direct will not fully “snap” to the correct stance at Round0.

In the paper, the model is trained as:
- “Decode pose *from phase*” (phase is the task-defining conditioning variable)

Whereas in our naive adaptation, the model can become:
- “Predict pose from cond+plan; meas only adjusts”

### 3.1 When FiLM is enough

FiLM is sufficient if:
- phase hint cleanly toggles between two modes and the representation has learned two separable manifolds
- the optimization forces the model to *need* the hint (e.g., by removing other shortcuts)

### 3.2 When FiLM is not enough

FiLM often fails when:
- the backbone already explains most variance, and the hint only shifts a few channels slightly
- the hint is noisy, so gradients encourage “ignore hint” behavior

---

## 4. Recommended “Bridge” Designs (from least to most structural change)

### Design D0 (Ablation): concatenate `contacts_meas` into direct input

Direct head becomes:

```
direct_pose_head([cond, contacts_plan, contacts_meas]) -> y_dir
```

Pros:
- Maximum “bandwidth” without feeding full pose
- Direct is still i.i.d.-ish w.r.t. pose drift compared to feeding `y_prev`

Cons:
- If `contacts_meas` drifts later, direct can drift in a correlated way unless we gate it.

### Design D0.5 (Recommended): use explicit phase state `phase_z_in` as direct phase hint

Empirically, the 2D `contacts_meas` / `contacts_plan` probabilities can collapse to a high‑entropy mid‑range output
(≈0.5), providing almost no bandwidth for **fine phase** (e.g. SIC49 vs SIC54 within stance/swing).

When `contact_phase_state_enable=True`, the model already maintains an explicit, step‑stateful clock:

- `phase_z_in`: `2 * contact_dim` (per foot `[sinφ, cosφ]`)

We can route this higher‑bandwidth phase representation into the direct head:

**Append mode (legacy “add phase”)**

```
direct_in = [direct_feat(+time_pe), contacts_plan, contacts_meas, phase_z_in]
```

Enable:
- `direct_pose_use_phase_z=true`
- `direct_pose_phase_z_mode=concat` (default)

**Replace mode (new; preferred when 2D contacts are degenerate)**

```
direct_in = [direct_feat(+time_pe), phase_z_in]   # replace (contacts_plan, contacts_meas)
```

Enable:
- `direct_pose_use_phase_z=true`
- `direct_pose_phase_z_mode=replace_contacts`

Notes:
- `replace_contacts` is supported only for `direct_pose_meas_mode='concat'` (it replaces the contact hint path).
- This does **not** remove the contact system; it only removes low‑bandwidth `contacts_*` features from *direct conditioning*.

### Design D1 (Safer): phase-hint as **mode selector** (not continuous regressor)

Instead of using meas as a small modulator, force it to select between two modes:

- learn two direct decoders (or two last-layer heads): `direct_left`, `direct_right`
- compute `p_left = contacts_meas[0]`, `p_right = contacts_meas[1]`
- blend: `y_dir = p_left * y_left + p_right * y_right`

This matches the “multi-modal disambiguation” nature directly and avoids FiLM bandwidth limits.

### Design D2 (Robust): hint dropout / corruption during training

To prevent the model from learning “ignore hint”:

- randomly zero out `contacts_meas` with probability `p`
- add noise to `contacts_meas` (small Beta noise / Gaussian before sigmoid clamp)
- optionally also corrupt `contacts_plan` so the model cannot use plan as a shortcut

Goal:
- make the hint *necessary* for early-phase disambiguation
- teach the network to handle hint noise gracefully (avoid hard dependence on a brittle white-box threshold)

### Design D3 (Long-horizon safety): gate direct usage by reliability

Because `contacts_meas` may be drift-corrupted at Round1+:

- compute a deterministic reliability score `r_meas(t)` from white-box internals (or from contact_err stability)
- only allow meas to affect direct when `r_meas` is high
- fall back to cond+plan-only direct when `r_meas` is low

This preserves the “Round0 helps, later don’t get poisoned” property.

---

## 5. What we learned from repo experiments (evidence)

### 5.1 Baseline: direct is catastrophically worse than inc in Round0

From `debug_output/freerun_cycles_verify_nofusion/Walk_F_freerun_cycles.json`:

- Round0 mean: `inc=11.36°`, `direct=22.63°`
- Round0 first10: `inc=3.00°`, `direct=24.78°`
- Even with per-joint bias align at step0 (`--direct_align_inc0`), Round0 mean only drops to `align=18.81°`

Interpretation:
- direct is not “a constant offset away” from inc; it is a wrong-mode solution early.

### 5.2 Training contact_plan dynamics helps early direct, but doesn’t win Round0

We added a posttrain mode to supervise full `contacts_plan` dynamics (GRU+heads) against GT contacts.

Checkpoint:
- `output_models/posttrain_contact_plan_walkf/ckpt_last_b1_s200.pth`

Verified in:
- `debug_output/freerun_cycles_verify_planpost_nofusion/Walk_F_freerun_cycles.json`

Numbers:
- Round0 mean: `direct=21.50°` (down from 22.63°)
- Round0 first10: `direct=15.99°` (down from 24.78°)
- Round0 first10 align: `align=12.59°` (down from 20.39°)

Also, `contacts_plan` is no longer nearly-constant in Round0:
- std increased significantly (proxy for “time-varying / phase-like” plan).

Interpretation:
- Plan dynamics was indeed a bottleneck, but direct still lacks a strong phase disambiguation input.

---

## 6. Practical next step (what to do next)

If the goal is **Round0 direct ~= inc** (or direct > inc), then we need to add an **early-clean (preferably higher‑bandwidth) phase hint** to direct and train direct to use it strongly:

1) Prefer **D0.5**: feed `phase_z_in` into direct (`direct_pose_use_phase_z=true`). When 2D contacts are degenerate, use `direct_pose_phase_z_mode=replace_contacts`.
2) If you still rely on `contacts_meas` as the hint, implement **D0** and add **D2** (hint corruption) to avoid “ignore hint”.
3) If you want paper-like multi-modal behavior, prioritize **D1** (mode selector) over pure FiLM.
4) Protect long horizon / non-periodic segments with **D3** (reliability gate / fallback).

This preserves your desired property: direct is not a function of full `y_prev`, so its errors can remain more orthogonal to inc drift than a state-conditioned residual direct.

---

## 7. Commands used in this repo (repro)

### 7.1 Baseline diagnostics (no apply)

```
python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_so3corr.pth \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --rounds 5 --time-index-mode cycle \
  --direct_align_inc0 \
  --log_contacts --log_contacts_whitebox --log_contacts_whitebox_first_steps 8 \
  --out debug_output/freerun_cycles_verify_nofusion --force
```

### 7.2 Train contact plan dynamics (teacher supervision)

```
python -m train.posttrain \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_so3corr.pth \
  --out_dir output_models/posttrain_contact_plan_walkf \
  --run_name b1_s200 \
  --data raw_data/processed_data --paths raw_data/processed_data/Walk_F.npz \
  --bundle_json raw_data/processed_data/norm_template.json \
  --pretrain_template models/pretrain_template.json \
  --encoder_bundle models/motion_encoder_equiv_stageA.pt \
  --device cpu --batch 1 --seq_len 87 --epochs 1 --steps_per_epoch 200 \
  --lr 2e-4 --weight_decay 0 \
  --train_so3_corrector false --train_contact_plan true --train_contact_plan_init false --train_lambda_head false \
  --contact_plan_weight 1.0 --time_index_mode cycle
```

### 7.3 Posttrain checkpoint diagnostics (no apply)

```
python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model output_models/posttrain_contact_plan_walkf/ckpt_last_b1_s200.pth \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --rounds 5 --time-index-mode cycle \
  --direct_align_inc0 \
  --log_contacts --log_contacts_whitebox --log_contacts_whitebox_first_steps 8 \
  --out debug_output/freerun_cycles_verify_planpost_nofusion --force
```

### 7.4 TD hazard head posttrain + loop stability check

Hazard-only posttrain（训练期用 `ttc_gt` 固定 phase_z；推理期用 `td_hazard` 闭环 reset）:

```
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_curriculum_4_4_10_ss1_lambda_cycles2_after_direct_pose.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_DirectBranch_v1_d1_curriculum_4_4_10_ss1_td_hazard_pt \
  --data raw_data/processed_data \
  --bundle_json raw_data/processed_data/norm_template.json \
  --pretrain_template models/pretrain_template.json \
  --encoder_bundle models/motion_encoder_equiv_stageA.pt \
  --device cpu \
  --seq_len 87 --batch 8 --epochs 1 --steps_per_epoch 200 \
  --rollout_cycles 5 --rollout_random_offset true --time_index_mode cycle \
  --train_so3_corrector false \
  --train_contact_plan_init false --train_contact_plan false \
  --train_direct_pose false --train_lambda_head false \
  --train_contact_meas false --train_contact_ttc false \
  --train_contact_td_hazard true \
  --phase_reset_source ttc_gt \
  --contact_td_hazard_bce_weight 1.0 \
  --contact_td_hazard_event_weight 86 \
  --contact_td_hazard_mass_weight 0.15 \
  --contact_td_hazard_unimodal_weight 0.01 \
  --contact_td_hazard_entropy_weight 0.01 \
```

Free-run verify (must see `phase_reset_source_applied=td_hazard` in the output JSON):

```
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_curriculum_4_4_10_ss1_td_hazard_pt.pth \
  --rounds 5 \
  --log_contacts \
  --phase_reset_source td_hazard \
  --device cpu \
  --out debug_output/freerun_td_hazard_verify --force
```

Summarize `events/cycle`, `period mean±std`, and `mass/cycle mean±std`:

```
python tools/summarize_freerun_ttc_loop.py \
  --event_key TDHazardEventPerC \
  --json debug_output/freerun_td_hazard_verify/*_freerun_cycles.json
```

---

## 8. Update (2025-12-30): D0 (concat meas) training result + remaining error source

We implemented the bridge heads (D0/D1) and ran a full retrain for **D0**:

- ckpt: `models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d0/ckpt_best_teacher_exp_phase_DirectBranch_v1_d0.pth`
- validation outputs:
  - `debug_output/freerun_cycles_verify_exp_phase_d0_depth3_joint_v2/Walk_F_freerun_cycles.json`
  - `debug_output/freerun_cycles_verify_exp_phase_d0_depth3_joint_v2/Walk_F_lower_body_geolocal.json`

### 8.1 What improved (evidence)

Command used (note `--depth 3` must match training):

```
python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d0/ckpt_best_teacher_exp_phase_DirectBranch_v1_d0.pth \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --rounds 2 --time-index-mode cycle --depth 3 \
  --direct_align_inc0 \
  --export_joint_geolocal \
  --log_contacts --log_contacts_whitebox --log_contacts_whitebox_first_steps 20 \
  --out debug_output/freerun_cycles_verify_exp_phase_d0_depth3_joint_v2 --force
```

Compared against the baseline ckpt:
`models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1/ckpt_best_teacher_exp_phase_DirectBranch_v1.pth`
(validated at `debug_output/freerun_cycles_verify_exp_phase_base_depth3_joint_v2/Walk_F_freerun_cycles.json`)

Observed changes on Walk_F (GeoLocalDeg, degrees):
- Round0 step0 `DirectGeoLocalDeg`: **5.77° → 4.14°**
- Round0 first10 mean `DirectGeoLocalDeg`: **16.71° → 14.15°**
- Round0 first20 mean `DirectGeoLocalDeg`: **18.30° → 12.82°** (D0 curve becomes much flatter after ~t=3)

### 8.2 Why “step0 is still not enough” (root cause, measured)

In Round0, `contacts_meas` is extremely clean at t=0 (near one-hot stance), but in the D0 ckpt
the **plan is initialized into the opposite stance** at t=0, so the direct head sees two conflicting phase hints.

From `debug_output/freerun_cycles_verify_exp_phase_d0_depth3_joint_v2/Walk_F_freerun_cycles.json` step0:
- `ContactMeasPerC ≈ [0.900, 0.000]`  (white-box meas, clean)
- `ContactGTPerC   ≈ [0.860, 0.012]`  (teacher GT soft contacts)
- `ContactPlanPerC ≈ [0.330, 0.679]`  (**wrong phase at t=0**)
- `ContactErrPerC  ≈ [-0.570, 0.679]` (plan - meas)

This means D0’s “concat meas” is necessary but not sufficient: the direct head must learn to
**trust meas over plan at t=0**, *or* plan must be cold-started correctly so plan/meas agree.

The clearest architectural reason this happens:
- D0 ckpt **does not have** `contact_plan_init_head.*` weights (no obs-conditioned init head).
- baseline ckpt **does have** `contact_plan_init_head.*` weights.

So the dominant error source is **contact_plan cold-start (plan_z0 initialization)**, not λ-fusion or SO(3) corrector.

### 8.3 “What exactly to check next” (verification checklist)

We added per-channel contact debug fields in `run_freerun_cycles` to make this measurable:
`ContactPlanPerC`, `ContactMeasPerC`, `ContactGTPerC`, `ContactErrPerC` (plus the existing `ContactMeasWhitebox`).

Use these checks to localize remaining errors:

1) **t=0 stance agreement**
   - Expect `argmax(ContactPlanPerC)` ≈ `argmax(ContactMeasPerC)` ≈ `argmax(ContactGTPerC)`.
   - If plan disagrees, Round0 direct will be systematically off even if meas is perfect.

2) **Is plan simply phase-shifted (cycle offset) rather than “random”?**
   - Compute best circular shift between `ContactPlanPerC[c]` and `ContactGTPerC[c]` over one cycle.
   - Large best-shift indicates the plan learned the gait waveform but is *misaligned* at t=0 (classic cold-start issue).

3) **Lower-body-only vs full-body**
   - Read `Walk_F_lower_body_geolocal.json` to separate “true gait phase” errors (legs/feet) from
     “underconstrained DOF” errors (arms/fingers).

### 8.4 Action items implied by the evidence

If the goal is “Round0 direct ~= inc”, the immediate bottleneck is plan_z0 disambiguation:
- Ensure `contact_plan_init_mode = learnable+obs` and the corresponding init head is trained,
  so `contacts_plan(t=0)` aligns with the observed stance (meas/GT) before the GRU dynamics takes over.

After plan_z0 is fixed, re-run the same `run_freerun_cycles` and check that:
- step0 `ContactPlanPerC` agrees with `ContactMeasPerC`,
- step0 `DirectGeoLocalDeg` drops further without relying on `--direct_align_inc0`.

### 8.5 Posttrain pipeline (updated paths for D0 ckpt)

If you follow the “3-stage posttrain” pipeline (plan-init → λ head → so3 corr),
then the only thing that changes for the D0 run is the **input checkpoint path** (and therefore the derived output paths).

> Note (2026-01-03): `train.posttrain` now preserves Event-Clock v3 weights (`event_clock_*`) by default (`--event_clock auto`).
> If you want an ablation that *drops* Event-Clock from the posttrain ckpt, add `--event_clock off` to **every** posttrain stage.
> Posttrain rollouts also maintain `meas_logits_prev` internally so `Δmeas` is non-zero even when unrolling with `T=1` per step.
>
> Note (2026-01-05): this repo now uses `contact_meas_head` v1 only (lower-body + no-hist; see `docs/contact_meas_head_redesign_lowerbody_nohist.md`).
> - Legacy `pose_hist`-based meas heads are no longer supported; please retrain if you need posttrain on an old checkpoint.
> - The `contact_meas_ground_z_*` args in the commands below only affect the *white-box* meas path (`_contact_meas_whitebox`).
>   If your checkpoint enables learned meas (`contact_meas_enable=true`), those args are ignored.
>
> Note (2026-01-06): `train.posttrain` preserves the **prev_phase_vec clock** (`contact_phase_state_*`) and its plan-logit injection (`contact_plan_phase_head.*`) when the input ckpt contains those weights.
> - Rollouts maintain step-stateful `phase_z` (like `plan_z`) across steps; if you reset plan state on cycle boundaries, both `plan_z/phase_z` reset.
> - Posttrain cannot “add” the clock to an old ckpt that never had it; you must enable it in full-train to get those weights.
>
> Note (2026-01-10): 新增 touchdown hazard clock-anchor（`td_hazard`，integrate-to-1）。它不是一个“推理阈值调参器”，而是：
> - **训练侧**把 hazard 约束成“每 cycle 单峰 + 总质量≈1”（BCE + mass + unimodality）。
> - **推理侧**用 accumulator `acc+=sigmoid(hz_logit)`，当 `acc>=1` 触发 event，并 `acc-=1`，从而 deterministic 地每周期触发 1 次 reset。
> - 跑法：先用 `train.posttrain --train_contact_td_hazard true` 训练 hazard head，然后 `run_freerun_cycles --phase_reset_source td_hazard`；验收用 `tools/summarize_freerun_ttc_loop.py --event_key TDHazardEventPerC ...`。

Below is the same pipeline, but starting from the D0 full-train checkpoint:
`models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d0/ckpt_best_free_exp_phase_DirectBranch_v1_d0.pth`

1) **Posttrain plan_z0 init head** (learnable+obs):

```
python -m train.posttrain \
  --config config/posttrain_lambda_fusion.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d0/ckpt_best_free_exp_phase_DirectBranch_v1_d0.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_DirectBranch_v1_d0_planinit_obs \
  --train_contact_plan_init true \
  --train_lambda_head false \
  --train_so3_corrector false \
  --contact_plan_init_mode learnable+obs \
  --contact_plan_init_hidden 128 \
  --seq_len 87 \
  --time_index_mode cycle \
  --contact_meas_ground_z_mode window \
  --contact_meas_ground_z_window 5 \
  --contact_meas_ground_z_quantile 0.2 \
  --contact_meas_ground_z_slew_down_cm 1.0 \
  --contact_meas_ground_z_slew_up_cm 0.2
```

This writes:
`models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d0_planinit_obs.pth`

2) **Posttrain λ head** (cycles2 + warmup reliability):

```
python -m train.posttrain \
  --config config/posttrain_lambda_fusion.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d0_planinit_obs.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_DirectBranch_v1_d0_lambda_cycles2_rwarmup10 \
  --train_lambda_head true \
  --train_contact_plan_init false \
  --train_so3_corrector false \
  --rollout_cycles 2 \
  --time_index_mode cycle \
  --lambda_reliability_mode warmup \
  --lambda_reliability_warmup_steps 10 \
  --contact_meas_ground_z_mode window \
  --contact_meas_ground_z_window 5 \
  --contact_meas_ground_z_quantile 0.2 \
  --contact_meas_ground_z_slew_down_cm 1.0 \
  --contact_meas_ground_z_slew_up_cm 0.2
```

This writes:
`models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d0_lambda_cycles2_rwarmup10.pth`

3) **Posttrain SO(3) corrector**:

```
python -m train.posttrain \
  --config config/posttrain_directbranch_so3_corr.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d0_lambda_cycles2_rwarmup10.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_DirectBranch_v1_d0_so3corr \
  --contact_meas_ground_z_mode window \
  --contact_meas_ground_z_window 5 \
  --contact_meas_ground_z_quantile 0.2 \
  --contact_meas_ground_z_slew_down_cm 1.0 \
  --contact_meas_ground_z_slew_up_cm 0.2
```

This writes:
`models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d0_so3corr.pth`

---

## 9. Update (2025-12-31): Direct Head Not Using Plan/Meas → train_direct_pose Mode

After running the 3-stage posttrain (planinit_obs → lambda_cycles2 → so3corr), we found `BlendGeoLocalDeg` still around 11°, far from the target <8°. This section documents the root cause diagnosis and the **train_direct_pose** solution that achieved **11° → 4°** improvement.

### 9.1 Problem Diagnosis Summary

Three bottlenecks were identified through systematic analysis:

1. **Plan amplitude collapse** (soft outputs, not wrong phase)
   - MSE loss on (L,R) marginals encourages "hedge" predictions
   - Observed: `mean(|L-R|) ≈ 0.37` vs GT `0.79` (47% amplitude)
   - Solution: supervise plan **logits** with `binary_cross_entropy_with_logits` (sigmoid head), not MSE on probabilities

2. **Whitebox meas crash after step0**
   - `VxyScoreMean` drops to 0 in free-run (FK velocity thresholds fail on predicted pose)
   - Meas becomes uninformative after the first step
   - Solution: learned `contact_meas_head` from pose features

3. **Direct head ignoring plan/meas** (root cause of the 11° plateau)
   - Gradient analysis revealed: `||∂L/∂cond|| >> ||∂L/∂plan||` (~30x ratio)
   - Direct learned to solve the task from `cond` alone, treating plan/meas as noise
   - The concat architecture `[cond, plan, meas]` allows this shortcut

### 9.2 The train_direct_pose Solution

We added a new posttrain stage that forces the direct_pose_head to use plan/meas:

**Key changes in `train/posttrain.py`:**
- New flag: `--train_direct_pose` (line ~1996)
- New objective parameter in `_lambda_fusion_loss_rollout`: `objective="direct"`
- Supervises only the direct branch output (not blend), forcing it to match GT
- Combined with BCE-supervised plan (sigmoid head) + learned meas head

**Why this works:**
- When direct is supervised directly (not blended with inc), gradient flows cleanly to plan/meas inputs
- The model cannot shortcut via cond alone because plan/meas carry phase information GT-correlated

### 9.3 Final Results (2025-12-31)

Validation on Walk_F with `run_freerun_cycles`, comparing:
- **Baseline**: ckpt after 3-stage posttrain (planinit → lambda → so3corr), ~11° blend error
- **After train_direct_pose**: additional direct head finetuning (**direct expert improved**)
- **After retraining λ**: re-train `lambda_fusion_head` once **after** direct improves (**re-calibrate Blend**)

| Metric | Baseline | After train_direct_pose | Improvement |
|--------|----------|-------------------------|-------------|
| DirectGeoLocalDeg R1-4 mean | 11.67° | 3.48° | -8.19° (70%) |
| BlendGeoLocalDeg R1-4 mean | 11.06° | 4.25° | -6.81° (62%) |

Note: R1-4 means Rounds 1-4 (excluding Round0 which has GT-aligned init).

#### Important: why “direct improves” ≠ “Blend improves”

`BlendGeoLocalDeg` depends on **both**:

1) direct expert quality (`out_direct`) and  
2) Stage2 selection (`λ_eff = λ * r_t`) which decides how much to trust direct vs incremental.

In practice, `train_direct_pose` changes the *direct expert*, so the previously trained `λ` head can become stale:
- before: direct is worse → λ learns to stay small (protect early)
- after: direct becomes good → λ should increase in drift rounds (use direct), but it will not unless retrained.

If you observe `DirectGeoLocalDeg` ≈ 3–4° but `BlendGeoLocalDeg` still high (and `LambdaMean` ~0.1–0.2),
that is the signature of **“direct is fixed, but λ is still conservative”**.

### 9.4 Recommended Checkpoint

Best performing checkpoint for **final Blend** (λ+SO(3) apply):
```
models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d0_lambda_cycles2_after_direct_pose.pth
```

**Important**:
- The Stage4 ckpt is an intermediate artifact (direct expert). For final Blend, retrain λ once (Stage5).
- Hinge is treated as **debug-only** and is not part of the default posttrain pipeline (keep as an ablation/reference).
- Keep `--so3_corr_gate_logit_reset null` when running `train_direct_pose`; resetting gate (e.g. `-3.0`) can hurt when `--so3_corr_apply` is enabled.

### 9.5 Commands for Reproduction

#### Stage 4: train_direct_pose (direct-only / reinit / hidden_pre)

Use the Stage3 checkpoint as input (e.g. the `*_so3corr.pth` output from the first 3 posttrain stages).

```
python -m train.posttrain \
  --config config/posttrain_directbranch_so3_corr.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d0_so3corr.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_direct_pose_rerun_keepgate \
  --train_direct_pose true \
  --direct_pose_feat_source hidden_pre \
  --direct_pose_hinge_enable false \
  --direct_pose_reinit true \
  --train_lambda_head false --train_so3_corrector false \
  --train_contact_plan false --train_contact_plan_init false \
  --train_contact_meas false --train_contact_td_hazard false --train_contact_ttc false \
  --phase_reset_source td_hazard \
  --so3_corr_gate_logit_reset null
```

**Stage4 sanity checks**

- stdout:
  - must include: `[posttrain][INFO] dropped ... direct_pose_* hinge tensors ... (reinit/override)` (expected: `direct_pose_reinit=true` drops direct+hinge weights from input ckpt and retrains).
  - `trainable=...` must include `direct_pose_head.*`; must NOT include `lambda_fusion_head.*` / `so3_*` / `contact_*`.
- output ckpt (sanity only):
  - `state_dict` must have `direct_pose_head.*`; must NOT have `direct_pose_hinge_*` (expected: Stage4 reinit + direct-only).
  - `posttrain_cfg.direct_pose_feat_source == "hidden_pre"` (ensure direct routing is pre-PASA).

#### (Debug) Stage 4b: hinge-only (clean-split / eps_source=hidden_pre)

This is kept as a **debug / ablation** recipe and should **not** be part of the default posttrain flow.

Run **after Stage4** (hinge-only expects the Stage4 direct head + `direct_pose_feat_source` to be finalized first).

```
python -m train.posttrain \
  --config config/posttrain_directbranch_so3_corr.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_direct_pose_rerun_keepgate.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_direct_pose_hinge_only_hiddenpre_clean \
  --train_direct_pose true \
  --direct_pose_feat_source hidden_pre \
  --direct_pose_reinit false \
  --direct_pose_hinge_enable true \
  --direct_pose_hinge_train_only true \
  --direct_pose_hinge_clean true \
  --direct_pose_hinge_eps_source hidden_pre \
  --direct_pose_hinge_bones calf_r \
  --train_lambda_head false --train_so3_corrector false \
  --train_contact_plan false --train_contact_plan_init false \
  --train_contact_meas false --train_contact_td_hazard false --train_contact_ttc false \
  --phase_reset_source td_hazard \
  --so3_corr_gate_logit_reset null
```

**Stage4b sanity checks**

- stdout:
  - `trainable=...` should ONLY include `direct_pose_hinge_*` (for clean split: especially `direct_pose_hinge_nonhidden_head.*` + `direct_pose_hinge_eps_head.*`).
  - must NOT include `direct_pose_head.*` (hinge-only).
- output ckpt:
  - `state_dict` must include `direct_pose_hinge_nonhidden_head.*` and `direct_pose_hinge_eps_head.*`.
  - `posttrain_cfg` must satisfy:
    - `direct_pose_feat_source == "hidden_pre"`
    - `direct_pose_hinge_clean == true`
    - `direct_pose_hinge_eps_source == "hidden_pre"`
    - `direct_pose_hinge_bones == "calf_r"` (or your actual setting)

#### Stage 5: retrain λ head (recommended for final Blend; run after Stage4)

This step recalibrates `lambda_fusion_head` after the direct expert quality changed.

```
python -m train.posttrain \
  --config config/posttrain_lambda_fusion.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_direct_pose_rerun_keepgate.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_DirectBranch_v1_d0_lambda_cycles2_after_direct_pose \
  --train_lambda_head true --train_direct_pose false --train_so3_corrector false \
  --train_contact_plan false --train_contact_plan_init false \
  --train_contact_meas false --train_contact_td_hazard false --train_contact_ttc false \
  --rollout_cycles 2 \
  --time_index_mode cycle \
  --lambda_reliability_mode warmup \
  --lambda_reliability_warmup_steps 10 \
  --phase_reset_source td_hazard
```

**Stage5 sanity checks**

- stdout:
  - `trainable=...` should ONLY include `lambda_fusion_head.*`.
- output ckpt:
  - `state_dict` must still include `direct_pose_head.*` (must NOT be dropped).
  - `posttrain_cfg` must still have `direct_pose_feat_source == "hidden_pre"` (no semantic mismatch vs Stage4).

##### λ gate supervision (logits) — use when Blend stays conservative

When `DirectGeoLocalDeg` is already low but `BlendGeoLocalDeg` stays conservative, the root cause is often
that the λ policy is not aligned with the per-step “which expert is better” signal.

We add a Stage2 supervision term on `lambda_fusion_logits`:

- compute per-joint errors (radians):  
  `err_inc = geodesic(R_inc, R_gt)` and `err_dir = geodesic(R_dir, R_gt)`
- oracle soft label:  
  `λ* = sigmoid((err_inc - err_dir) / τ)`
- apply a margin mask: only supervise when `|err_inc - err_dir| >= δ` (default `δ=1°`)
- loss: `BCEWithLogits(lambda_fusion_logits, λ*)` (soft targets), added to the Stage2 objective

Flags:
- `--lambda_gate_sup_weight`: supervision weight (0 disables; start with `0.1`)
- `--lambda_gate_sup_tau_deg`: soft-label temperature τ in degrees (start with `2.5`)
- `--lambda_gate_sup_margin_deg`: margin δ in degrees (default `1.0`; set `0` to disable)
- `--lambda_gate_sup_start_step`: start rollout step; `-1` auto uses `lambda_reliability_warmup_steps` when warmup is enabled

Posttrain logs/stats include:
- `gate_sup_loss`: weighted supervision loss (per rollout)
- `gate_sup_acc@0.5`: agreement rate between `sigmoid(logits)>0.5` and `(err_inc>err_dir)` on supervised entries
- `gate_sup_frac`: supervised fraction (after margin mask)

#### Validation command

```
python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d0_lambda_cycles2_after_direct_pose.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --rounds 5 --time-index-mode auto \
  --so3_corr_apply --lambda_fusion_apply \
  --log_contacts --log_contacts_whitebox --log_contacts_whitebox_first_steps 8 \
  --out debug_output/posttrain_direct_pose_rerun/after_keepgate_lambda_retrained --force
```

### 9.6 Complete Posttrain Pipeline Summary

For future reference, the complete posttrain pipeline is:

| Stage | Target | Key Flags | Purpose |
|-------|--------|-----------|---------|
| 1 | plan_z0 init | `--train_contact_plan_init true` | Correct cold-start stance |
| 2 | λ head | `--train_lambda_head true --rollout_cycles 2` | Learn inc/direct blending |
| 3 | SO(3) corr | `--train_so3_corrector true` | Orientation drift correction |
| 4 | direct head (direct-only) | `--train_direct_pose true --direct_pose_reinit true --direct_pose_feat_source hidden_pre --direct_pose_hinge_enable false` | Reinit + train direct expert on pre-PASA features |
| 5 | λ head (retrain) | `--train_lambda_head true --rollout_cycles 2 --time_index_mode cycle` | Re-calibrate Blend after direct changes |

Stage 4 is the key addition that fixes direct quality (and avoids the "cond-only shortcut" when using the legacy `direct_pose_feat_source="cond"` route);
Stage 5 is the practical step that makes `BlendGeoLocalDeg` follow the improved direct expert under `--lambda_fusion_apply`.

Important: keep `direct_pose_feat_source` **identical** across Stage4/5. Otherwise Stage5 will train λ against a different "direct semantics" than what you evaluate at inference.

Stage 6 turns on a **stable phase reset source** that does not rely on `contacts_meas` threshold crossings (which can jitter under drift):
the model predicts a per-step touchdown **hazard** and inference uses an **integrate-to-1 accumulator** to produce exactly-one-event-per-cycle
reset events (`--phase_reset_source td_hazard`). Training is done with **`ttc_gt` reset** to stabilize `phase_z`, plus an **alignment loss**
to force hazard peaks to coincide with GT touchdown.

1) **Posttrain hazard head** (run on the checkpoint you actually plan to use; this is usually Stage 5 output):

```
python -m train.posttrain \
  --config config/posttrain.json \
  --ckpt_in <CKPT_FROM_STAGE5.pth> \
  --out_dir <OUT_DIR> --run_name posttrain_td_hazard \
  --train_so3_corrector false \
  --train_contact_plan_init false --train_contact_plan false \
  --train_direct_pose false --train_lambda_head false \
  --train_contact_meas false --train_contact_ttc false \
  --train_contact_td_hazard true \
  --phase_reset_source ttc_gt \
  --contact_td_hazard_bce_weight 1.0 \
  --contact_td_hazard_event_weight <seq_len-1> \
  --contact_td_hazard_mass_weight 0.1 \
  --contact_td_hazard_unimodal_weight 0.01 \
  --contact_td_hazard_entropy_weight 0.01 \
  --seq_len <cycle_len>
```

2) **Validate hazard-driven reset**:

```
python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model <CKPT_WITH_TD_HAZARD.pth> \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --rounds 5 --cond-reprojection auto --event_clock auto \
  --phase_reset_source td_hazard \
  --log_contacts \
  --out <OUT_DIR> --force
```

3) **Acceptance summary (events/cycle and period)**:

```
python tools/summarize_freerun_ttc_loop.py \
  --event_key TDHazardEventPerC \
  --json <OUT_DIR>/*_freerun_cycles.json
```

Notes:
- Event-Clock v3 is **not** a separate posttrain stage; it is a contact_plan-internal correction that should be preserved across stages (default `--event_clock auto`).
- The prev_phase_vec clock (`contact_phase_state_*`, step-state `phase_z`) is also **not** a separate posttrain stage; if your ckpt has it, it should be preserved across stages and rollouts must cache `phase_z` like `plan_z`.
- Keep `encoder_bundle` configured (all provided posttrain JSON configs already set it) so `period_encoder`/frozen encoder wiring stays consistent across posttrain checkpoints.

### 9.7 Lessons Learned

1. **Concat ≠ usage**: Just concatenating plan/meas to direct input doesn't guarantee the model uses them. Gradient shortcuts through cond can dominate.

2. **MSE on soft targets encourages hedging**: Use logits-space BCE (`binary_cross_entropy_with_logits`) for contact stance instead.

3. **Whitebox features are fragile in free-run**: FK-derived contact scores (VxyScore, etc.) fail on predicted pose. Learned heads are more robust.

4. **Direct supervision breaks shortcuts**: Training the direct branch directly (not blended) forces it to learn proper phase conditioning.

5. **After changing an expert, retrain the selector**: `lambda_fusion_head` is a *policy* over experts. If the direct expert changes, the old λ policy can be suboptimal.

---

## 10. Update (2026-01-01): D0 (concat meas) Impact Table (Walk_F)

This section records the measured impact of D0 (“direct input includes `contacts_meas`”) at different stages.

Two different notions of “D0 effect”:

1) **Train-time effect**: compare a non-D0 ckpt vs a D0 ckpt (weights differ globally; this is *not* an isolated ablation).
2) **Runtime usage ablation**: for a *fixed D0 ckpt*, toggle `--direct_pose_meas_force_zero` in `run_freerun_cycles`
   to force the direct head to ignore the meas hint (**concat → zeros**). This does **not** change `contacts_err` / λ inputs
   (it only changes the direct hint), so it isolates “is the model actually using `contacts_meas`?”.

All metrics below are **GeoLocalDeg (degrees)** on Walk_F.

### 10.1 Train-time effect (base vs D0 ckpt)

Evaluations:
- Base: `debug_output/freerun_cycles_verify_exp_phase_base_depth3_joint_v2/Walk_F_freerun_cycles.json`
- D0: `debug_output/freerun_cycles_verify_exp_phase_d0_depth3_joint_v2/Walk_F_freerun_cycles.json`

| ckpt | D0 | Round0 step0 `DirectGeoLocalDeg` | Round0 first20 mean `DirectGeoLocalDeg` | Round0 mean `DirectGeoLocalDeg` | Round1 mean `DirectGeoLocalDeg` |
|------|----|----------------------------------:|----------------------------------------:|--------------------------------:|--------------------------------:|
| `exp_phase_DirectBranch_v1/ckpt_best_teacher_exp_phase_DirectBranch_v1.pth` | no | 5.77 | 18.30 | 21.13 | 22.62 |
| `exp_phase_DirectBranch_v1_d0/ckpt_best_teacher_exp_phase_DirectBranch_v1_d0.pth` | yes | 4.14 | 12.82 | 16.79 | 15.65 |

### 10.2 Runtime usage ablation (`--direct_pose_meas_force_zero`)

#### A) D0 train ckpt (no Stage2 apply; isolate “meas hint used?”)

Evaluations:
- D0 (meas enabled): `debug_output/freerun_cycles_verify_exp_phase_d0_depth3_joint_v2/Walk_F_freerun_cycles.json`
- D0 (force direct meas=0): `debug_output/freerun_cycles_verify_exp_phase_d0_depth3_joint_v2_no_meas/Walk_F_freerun_cycles.json`

| ckpt | `--direct_pose_meas_force_zero` | Round0 step0 `DirectGeoLocalDeg` | Round0 mean `DirectGeoLocalDeg` | Round1 mean `DirectGeoLocalDeg` |
|------|--------------------------------:|----------------------------------:|--------------------------------:|--------------------------------:|
| `exp_phase_DirectBranch_v1_d0/ckpt_best_teacher_exp_phase_DirectBranch_v1_d0.pth` | false | 4.14 | 16.79 | 15.65 |
| `exp_phase_DirectBranch_v1_d0/ckpt_best_teacher_exp_phase_DirectBranch_v1_d0.pth` | true  | 4.91 | 16.83 | 15.65 |

Interpretation: on this ckpt, meas-hint mainly helps **step0 / very-early** (~+0.77° at step0 when removed), but has near‑zero effect on the per-round mean.

#### B) Posttrain λ ckpt (λ apply; isolate effect on final `Blend`)

Evaluations:
- meas enabled: `debug_output/bridge_table_lambda_apply_r5/Walk_F_freerun_cycles.json`
- force direct meas=0: `debug_output/bridge_table_lambda_apply_zero_directmeas_r5/Walk_F_freerun_cycles.json`

| ckpt | `--direct_pose_meas_force_zero` | Round0 mean `BlendGeoLocalDeg` | R1+ mean `BlendGeoLocalDeg` | R1+ mean `DirectGeoLocalDeg` |
|------|--------------------------------:|-------------------------------:|----------------------------:|-----------------------------:|
| `ckpt_last_posttrain_DirectBranch_v1_d0_lambda_cycles2_rwarmup10.pth` | false | 9.81 | 11.28 | 12.07 |
| `ckpt_last_posttrain_DirectBranch_v1_d0_lambda_cycles2_rwarmup10.pth` | true  | 9.86 | 11.47 | 12.33 |

#### C) Final ckpt (λ+SO(3) apply; isolate effect near the current ~4° plateau)

Evaluations:
- meas enabled: `debug_output/posttrain_direct_pose_rerun/after_keepgate_lambda_retrained/Walk_F_freerun_cycles.json`
- force direct meas=0: `debug_output/posttrain_direct_pose_rerun/after_keepgate_lambda_retrained_no_meas/Walk_F_freerun_cycles.json`

| ckpt | `--direct_pose_meas_force_zero` | Round0 mean `BlendGeoLocalDeg` | R1+ mean `BlendGeoLocalDeg` | R1+ mean `DirectGeoLocalDeg` |
|------|--------------------------------:|-------------------------------:|----------------------------:|-----------------------------:|
| `ckpt_last_posttrain_DirectBranch_v1_d0_lambda_cycles2_after_direct_pose.pth` | false | 3.22 | 4.97 | 3.63 |
| `ckpt_last_posttrain_DirectBranch_v1_d0_lambda_cycles2_after_direct_pose.pth` | true  | 3.13 | 4.73 | 3.54 |

Interpretation: for the current final ckpt, disabling D0 meas-hint changes `Blend` by only ~0.2–0.3° on R1+ mean (sign may vary),
so D0 is **not** the dominant bottleneck for “~5° → ~2°”, but it is also **not** negligible.

### 10.3 Repro commands

Quick helper (base vs D0 train ckpts only):

```bash
python tools/eval_d0_concat_ablation.py --force --log_contacts
```

To reproduce the runtime ablation for any checkpoint:

```bash
python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model <CKPT.pth> \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --rounds 5 --time-index-mode auto --depth 3 \
  --lambda_fusion_apply --so3_corr_apply \
  --out debug_output/<out_dir> --force
```

Then rerun with `--direct_pose_meas_force_zero` added, and compare the two JSON outputs.

---

## 11. Update (2026-01-06): `d1_phaseclk` (prev_phase_vec clock) end-to-end commands

This section records the exact commands used for the new training flow:

- Full-train: enable `contact_phase_state_enable` (prev_phase_vec clock) + Event-Clock v3 + `w_direct_pose`
- Posttrain: same 5-stage pipeline as before, but **starting from the `d1_phaseclk` full-train ckpt**

### 11.1 Full-train command

```bash
python -m train.training_MPL \
  --config_json config/exp_phase_mpl.json \
  --run_name exp_phase_DirectBranch_v1_d1_phaseclk \
  --out ./models/MLPL2_DirectBranch_v1 \
  --depth 3 \
  --encoder_path ./models/motion_encoder_equiv_stageA.pt \
  --contact_plan_enable \
  --contact_plan_init_mode learnable+obs --contact_plan_init_hidden 128 \
  --contact_phase_state_enable \
  --contact_phase_state_init_mode obs \
  --contact_phase_state_event_kind touchdown --contact_phase_state_event_thr 0.5 \
  --direct_pose_enable --w_direct_pose 0.2 \
  --contact_plan_time_pe_dim 16 \
  --direct_pose_meas_mode concat \
  --direct_pose_meas_drop_prob 0.1 --direct_pose_plan_drop_prob 0.1 --direct_pose_meas_noise_std 0.03 \
  --use_event_clock \
  --event_clock_max_delta 0.5 \
  --event_clock_hidden_dim 64 \
  --event_clock_gate_hidden_dim 32 \
  --event_clock_lambda_entropy_weight 0.01 \
  --event_clock_lambda_prior_weight 0.01 \
  --event_clock_delta_z_l2_weight 0.001
```

Outputs:
- `models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1_phaseclk/ckpt_best_free_exp_phase_DirectBranch_v1_d1_phaseclk.pth`
- `models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1_phaseclk/ckpt_best_teacher_exp_phase_DirectBranch_v1_d1_phaseclk.pth`

### 11.2 Posttrain pipeline (stages 1–6; `d1_phaseclk_resid_20260108`)

Stage 1: plan_z0 init

```bash
python -m train.posttrain \
  --config config/posttrain_lambda_fusion.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1_phaseclk_resid_20260108/ckpt_best_free_exp_phase_DirectBranch_v1_d1_phaseclk_resid_20260108.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_planinit_obs \
  --train_contact_plan_init true \
  --train_lambda_head false \
  --train_so3_corrector false \
  --contact_plan_init_mode learnable+obs \
  --contact_plan_init_hidden 128 \
  --seq_len 87 \
  --time_index_mode cycle
```

Stage 2: λ head (cycles2 + warmup reliability)

```bash
python -m train.posttrain \
  --config config/posttrain_lambda_fusion.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_planinit_obs.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_lambda_cycles2_rwarmup10 \
  --train_lambda_head true \
  --train_contact_plan_init false \
  --train_so3_corrector false \
  --rollout_cycles 2 \
  --time_index_mode cycle \
  --lambda_reliability_mode warmup \
  --lambda_reliability_warmup_steps 10
```

Stage 3: SO(3) corrector

```bash
python -m train.posttrain \
  --config config/posttrain_directbranch_so3_corr.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_lambda_cycles2_rwarmup10.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_so3corr
```

Stage 4: train_direct_pose (keep the learned SO(3) gate)

```bash
python -m train.posttrain \
  --config config/posttrain_directbranch_so3_corr.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_so3corr.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_direct_pose_keepgate \
  --train_so3_corrector false \
  --train_contact_plan_init false \
  --train_contact_plan false \
  --train_lambda_head false \
  --train_direct_pose true \
  --so3_corr_gate_logit_reset null
```

Stage 5: retrain λ (final Blend)

```bash
python -m train.posttrain \
  --config config/posttrain_lambda_fusion.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_direct_pose_keepgate.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_lambda_cycles2_after_direct_pose \
  --train_lambda_head true \
  --train_contact_plan_init false \
  --train_so3_corrector false \
  --rollout_cycles 2 \
  --time_index_mode cycle \
  --lambda_reliability_mode warmup \
  --lambda_reliability_warmup_steps 10 \
  --lambda_gate_sup_weight 0.1 \
  --lambda_gate_sup_tau_deg 2.5 \
  --lambda_gate_sup_margin_deg 1.0 \
  --lambda_gate_sup_start_step -1
```

Stage 6: TD hazard head（训练期用 `ttc_gt` 稳定 phase_z；推理期用 `td_hazard` 闭环）

```bash
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_lambda_cycles2_after_direct_pose.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_td_hazard_pt \
  --data raw_data/processed_data \
  --bundle_json raw_data/processed_data/norm_template.json \
  --pretrain_template models/pretrain_template.json \
  --encoder_bundle models/motion_encoder_equiv_stageA.pt \
  --seq_len 87 --batch 8 --epochs 1 --steps_per_epoch 200 \
  --train_so3_corrector false \
  --train_contact_plan_init false --train_contact_plan false \
  --train_direct_pose false --train_lambda_head false \
  --train_contact_meas false --train_contact_ttc false \
  --train_contact_td_hazard true \
  --phase_reset_source ttc_gt \
  --contact_td_hazard_rollout true \
  --contact_td_hazard_bce_weight 1.0 \
  --contact_td_hazard_mass_weight 0.1 \
  --contact_td_hazard_unimodal_weight 0.01 \
  --contact_td_hazard_entropy_weight 0.01 \
```

Final ckpt:
- Blend (Stage 5): `models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_lambda_cycles2_after_direct_pose.pth`
- Deploy (Stage 6): `models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_td_hazard_pt.pth`

### 11.3 Validation (Walk_F; `--cond-reprojection auto`)

```bash
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_td_hazard_pt.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --rounds 5 --time-index-mode auto --depth 3 \
  --lambda_fusion_apply --so3_corr_apply \
  --cond-reprojection auto --event_clock auto \
  --contacts_meas_source model \
  --phase_reset_source td_hazard \
  --log_contacts \
  --out debug_output/posttrain_d1_phaseclk_resid_20260108/after_direct_pose_lambda_retrained_td_hazard --force
```

Output JSON:
- `debug_output/posttrain_d1_phaseclk_resid_20260108/after_direct_pose_lambda_retrained_td_hazard/Walk_F_freerun_cycles.json`

To print the per-round markdown tables (same format as our “5-stage” tables):

```bash
python tools/print_freerun_cycles_tables.py \
  --metric BlendGeoLocalDeg \
  --stage "FINAL=debug_output/posttrain_d1_phaseclk_resid_20260108/after_direct_pose_lambda_retrained_td_hazard/Walk_F_freerun_cycles.json"
```

如果你已经把各 stage 的 `run_freerun_cycles` JSON 都跑出来了（例如 Base/noapply、λ/apply、FINAL/apply），可以直接拼成 “bridge table”：

```bash
python tools/print_freerun_cycles_tables.py \
  --metric BlendGeoLocalDeg \
  --stage "Base (noapply)=<...>/Walk_F_freerun_cycles.json" \
  --stage "planinit_obs (noapply)=<...>/Walk_F_freerun_cycles.json" \
  --stage "lambda_fusion (apply)=<...>/Walk_F_freerun_cycles.json" \
  --stage "so3corr ckpt (lambda apply)=<...>/Walk_F_freerun_cycles.json" \
  --stage "FINAL (lambda+so3 apply)=<...>/Walk_F_freerun_cycles.json"
```

#### 11.3.1 Acceptance: GT-X(except rot6d)（排除 X-side 干扰）

为了隔离「rot6d / pose 自身」误差、排除 root/state (X-side) drift 对 long-horizon 的干扰，可以在 freerun 打开：

- `--freerun_x_gt_except_rot6d`：每步把 rollout state **X** 覆盖为 teacher GT（除了 `BoneRotations6D` slice）。

跑法（同一 ckpt / 同一套 apply flags，分别跑 baseline 和 GT-X）：

```bash
# Baseline
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model <CKPT.pth> \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --rounds 5 --time-index-mode auto --depth 3 \
  --lambda_fusion_apply --so3_corr_apply \
  --cond-reprojection auto --event_clock auto \
  --out <OUT_BASE> --force

# GT-X(except r6d)
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model <CKPT.pth> \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --rounds 5 --time-index-mode auto --depth 3 \
  --lambda_fusion_apply --so3_corr_apply \
  --cond-reprojection auto --event_clock auto \
  --freerun_x_gt_except_rot6d \
  --out <OUT_GTX> --force
```



```bash
python tools/print_freerun_cycles_tables.py \
  --compare \
  --baseline <OUT_BASE>/Walk_F_freerun_cycles.json \
  --after <OUT_GTX>/Walk_F_freerun_cycles.json \
  --compare-after-label "After GT-X(except r6d)" \
  --compare-metric BlendGeoLocalDeg \
  --compare-metric DirectGeoLocalDeg
```

> Note: 这是诊断用 ablation（用于排除干扰/定位误差来源），不是 deployment 评估路径。

### 11.4 TTC clock-anchor (legacy / debug): make it show up in the bridge tables (2026-01-10)

**Key point**: 你看到的 5-stage 表默认走的是 `phase_reset_source=contacts_meas`（旧路径）；要对照 TTC，必须在 `run_freerun_cycles` 里显式指定 `--phase_reset_source`，并检查输出 JSON 里的 `phase_reset_source_applied`，避免 “ttc_pred 但 ckpt 没有 TTC head → silent fallback”。

最小对照（同一套 `--cond-reprojection auto`，不要用 `--cond-reprojection on`）：

```bash
# Oracle sanity: ttc_gt（不依赖 TTC head）
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model <CKPT.pth> \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --rounds 5 --time-index-mode auto --depth 3 \
  --lambda_fusion_apply --so3_corr_apply \
  --cond-reprojection auto --event_clock auto \
  --contacts_meas_source model \
  --phase_reset_source ttc_gt \
  --log_contacts \
  --out debug_output/<out_dir> --force

# Legacy deploy path: ttc_pred（更容易受 round/阈值抖动影响；稳定 reset 推荐用 11.5 的 `td_hazard`）
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model <CKPT_WITH_TTC_HEAD.pth> \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --rounds 5 --time-index-mode auto --depth 3 \
  --lambda_fusion_apply --so3_corr_apply \
  --cond-reprojection auto --event_clock auto \
  --contacts_meas_source model \
  --phase_reset_source ttc_pred --ttc_update_alpha 0.25 \
  --log_contacts \
  --out debug_output/<out_dir> --force
```

事件稳定性统计（建议一起跑，避免只看 GeoLocalDeg 误判）：

```bash
python tools/summarize_freerun_ttc_loop.py --json debug_output/<out_dir>/*_freerun_cycles.json
```

**如果你要把 `ttc_pred` 也做成“5-stage bridge table”的同款格式**：

- 不建议把 `ttc_pred` 当成“每个中间 stage ckpt 都必须稳定”的硬指标；工程目标是 **最终上线 ckpt** 的 reset 稳定。中间 stage 更适合用 `ttc_gt`（oracle）来验证机制上限/排查逻辑。
- 如果 `ttc_pred` 出现 `2/5`、`0/0`、`10/9`，优先在最终 ckpt 上做一次 **短** 的 `train_contact_ttc_only` 校准，并开启 `--contact_ttc_event_weight / --contact_ttc_small_weight` 去强化 touchdown 附近（而不是把 `steps_per_epoch` 从 200 一路堆到 500+ 造成过拟合）。详见 `docs/changes/2026-01-10_contact_loop_ttc_clock_anchor.md` 的“ttc_pred 没达到预期”小节。

### 11.5 TD hazard clock-anchor (`td_hazard`): deterministic integrate-to-1 reset (2026-01-10)

#### 11.5.1 Motivation (why we moved away from `ttc_pred`)

旧路线把 `ttc_pred` 当作 “`0..cycle_len` 倒计时” 在推理侧 `round()` 并做阈值/最小间隔逻辑，很容易出现 **round 抖动 → reset 事件爆炸**（短间隔重复触发），导致 `phase_z` 锚点不稳定。

新路线改成：模型每步输出 touchdown **hazard**（概率质量/强度），推理侧用 **integrate-to-1 accumulator** 把它变成 deterministic event（理论上每 cycle 触发 1 次），并用该 event 去 reset `phase_z`（clock anchor），不依赖阈值/最小间隔等推理超参。

#### 11.5.2 Runtime behavior (integrate-to-1)

对每个 contact channel（Walk 典型是 L/R 两个 channel）维护一个 accumulator：

```
hazard_t = sigmoid(hz_logit_t)         # in [0, 1]
acc += hazard_t
event = (acc >= 1)
acc = acc - event.float()             # keep in [0, 1)
```

`event==1` 的那一步用于做 `phase_z` reset（clock anchor），因此是 deterministic、无阈值调参，并且天然具有 “总质量≈1 → 每周期 1 次触发” 的可解释性。

#### 11.5.3 Model interfaces / logs (what to look for)

**Hazard head outputs** (from model `ret`):
- `ret["contacts_td_hazard_logit"]`: `(B,T,C)` (or `(B,C)` for single-step)
- `ret["contacts_td_hazard_prob"] = sigmoid(logit)`: `(B,T,C)`

**Free-run logging**: `run_freerun_cycles --log_contacts` 会写出 per-step：
- `TDHazardLogitPerC`, `TDHazardProbPerC`
- `TDHazardAccPerC` (sawtooth accumulator)
- `TDHazardEventPerC` (0/1 event per step)

**Summary script**: `tools/summarize_freerun_ttc_loop.py` 支持指定 event key，hazard 用：
`--event_key TDHazardEventPerC`.

#### 11.5.4 Train: posttrain hazard-only (copy-paste)

你需要先把 hazard head 训练进 checkpoint（否则 freerun 会 fallback 到 `contacts_meas`）。

关键点（新版）：**训练期用 `--phase_reset_source ttc_gt` 稳定 phase_z**，让 hazard 学 “相位→touchdown 对齐”；推理期再切回 `td_hazard` 做闭环 reset。

下面命令只训练 hazard head（关掉其它 posttrain 目标），`ckpt_in` 用你最终要 freerun 的 ckpt（这里用 11.2 Stage 5 输出），并用 `seq_len=87` 覆盖一个 cycle（Walk_F 的 cycle_len=87；如果你换数据，确保 `seq_len >= cycle_len`）：

```bash
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain.json \
  --ckpt_in models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_lambda_cycles2_after_direct_pose.pth \
  --out_dir models/MLPL2_DirectBranch_v1 \
  --run_name posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_td_hazard_pt \
  --data raw_data/processed_data \
  --bundle_json raw_data/processed_data/norm_template.json \
  --pretrain_template models/pretrain_template.json \
  --encoder_bundle models/motion_encoder_equiv_stageA.pt \
  --seq_len 87 --batch 8 --epochs 1 --steps_per_epoch 200 \
  --train_so3_corrector false \
  --train_contact_plan_init false --train_contact_plan false \
  --train_direct_pose false --train_lambda_head false \
  --train_contact_meas false --train_contact_ttc false \
  --train_contact_td_hazard true \
  --phase_reset_source ttc_gt \
  --contact_td_hazard_bce_weight 1.0 \
  --contact_td_hazard_event_weight 86 \
  --contact_td_hazard_mass_weight 0.1 \
  --contact_td_hazard_unimodal_weight 0.01 \
  --contact_td_hazard_entropy_weight 0.01 \
```

Outputs:
- ckpt: `models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_td_hazard_pt.pth`
- log: `models/MLPL2_DirectBranch_v1/posttrain_log_posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_td_hazard_pt.json`

Quick self-check:
- `contact_td_hazard_mass_pred_mean` 是否逐步接近 `contact_td_hazard_mass_tgt_mean`（每 cycle≈1）。  
  （如果用 mixed supervision：`--contact_td_hazard_rollout_weight > 0`，字段会变成 `rollout_contact_td_hazard_*` / `teacher_contact_td_hazard_*`。）

#### 11.5.5 Free-run: use hazard as reset source

```bash
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_resid_20260108_td_hazard_pt.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --rounds 5 \
  --cond-reprojection auto --event_clock auto \
  --phase_reset_source td_hazard \
  --log_contacts \
  --out debug_output/freerun_td_hazard_d1_phaseclk_resid_20260108 --force
```

关键检查点：输出 JSON 里 `phase_reset_source_applied` 必须是 `td_hazard`（不是 `contacts_meas`）。

#### 11.5.6 Acceptance (avoid slipping back into “tuning”)

A) **事件稳定性（第一优先）**：

```bash
python tools/summarize_freerun_ttc_loop.py \
  --event_key TDHazardEventPerC \
  --json debug_output/freerun_td_hazard_d1_phaseclk_resid_20260108/*_freerun_cycles.json
```

期望：
- `TDHazardEventPerC` events (L/R) ≈ rounds/rounds（每 cycle 1 次）
- `TDHazardEventPerC` period mean (L/R) ≈ cycle_len（允许小抖动，但不能像 `ttc_pred` 那样一堆短间隔）

B) **形态检查（第二优先）**：看 freerun JSON 每步的：
- `TDHazardAccPerC`: sawtooth（0→接近1→event→减1回到[0,1)）
- `TDHazardEventPerC`: 每 cycle 触发 1 次为目标
- `TDHazardProbPerC`: 单峰/集中（unimodal weight 生效后更明显）

C) **任务指标（第三优先）**：`BlendGeoLocalDeg`（R1-4）不应明显退化。

#### 11.5.7 Common failures / fixes

- `phase_reset_source_applied = contacts_meas`：ckpt 里没有 `contact_td_hazard_head.*` 权重 → 先跑一遍 11.5.4 的 hazard-only posttrain 再 freerun。
- events/cycle << 1（event starvation）：hazard 总质量太小
  - 提高 `--contact_td_hazard_mass_weight` 或 `--contact_td_hazard_bce_weight`
  - 确保 `seq_len` 覆盖完整 cycle（窗内真的包含 touchdown event）
- events/cycle > 1（over-trigger）：hazard 总质量 > 1（accumulator 多次跨 1）
  - 提高 `--contact_td_hazard_mass_weight`（把总质量压回 1）
  - 提高 `--contact_td_hazard_unimodal_weight` 抑制多峰
- hazard 变成“常数率/散峰”（`TDHazardProbPerC` 很平，或一堆小峰）：
  - 增大 `--contact_td_hazard_event_weight`（经验上先试 `≈ seq_len-1`）
  - 或加入 `--contact_td_hazard_entropy_weight`（鼓励更尖的单峰；和 `unimodal` 互补）

---

## 12. Update (2026-01-24): Stage7 direct objective de-dilution (soft tail + swing-state)

This is **not** a LUT/hinge fix. It changes the **credit assignment** of the Stage6 direct rollout objective to address
phase-locked spikes that were effectively invisible under `(joint x step) mean` reduction.

> Update (2026-01-25): We now treat this tail reweight path as **deprecated** for Stage7.
> The current repo version removes the Median/Quadratic tail focus and any auto-balance/reweighting inside the Stage7
> direct objective (back to base mean; optional leg/nonleg split). The knobs
> `direct_pose_loss_tail_*` / `direct_pose_loss_state_*` remain in the CLI/config for backward compatibility but are
> effectively **no-ops** for the Stage7 direct objective.
>
> IMPORTANT: when running `run_freerun_cycles` on Stage6/7 checkpoints, always pass the correct `--depth` (e.g. `--depth 3`)
> to match the checkpoint architecture; otherwise the runner may instantiate a depth=2 model and silently ignore
> `shared_encoder.8.*` residual block weights.

### 12.1 Background: why some joints/spikes are “information exists but not learned”

In Stage6 direct (cond anchor), the core objective effectively does:

- per-step per-sample: `mean_joints(e_dir[B,J])`
- then average across `~400+` rollout steps

So a single “(joint, SIC)” spike gets a gradient coefficient roughly `~ 1/(J * total_steps)` (can be ~`5e-5`), i.e.
**strongly diluted**. This matches the observed behavior for some legs/feet spikes (e.g. `calf/foot` at few SIC points),
and is orthogonal to “axis upper bound” issues (e.g. `foot_l@SIC=13` type failures).

### 12.2 Fix: smooth, stop-grad weighting (no per-phase/bone LUT; low memory risk)

We modify the **direct** rollout loss accumulation to:

- **Soft tail over joints (stop-grad)**:
  - `w_tail = softmax(e_dir.detach() / T)`
- **State-aware swing boost (stop-grad)**:
  - `w_state = 1 + a * (1 - contact.detach())` (per-joint side inferred by `_l/_r` suffix; scoped to `legs|limbs|all`)
- **Scale-stable normalize**:
  - `w_eff = w_eff / mean(w_eff)` (per-sample per-step)
- **Convex mix** (avoid whack-a-mole / keep baseline behavior):
  - `L = (1-λ) * mean(e_dir) + λ * mean(e_dir * w_eff)`

Config knobs (all default to 0/off in `train.posttrain`):
- `direct_pose_loss_tail_mix` (λ, recommended start 0.3)
- `direct_pose_loss_tail_temp_deg` (T in degrees, recommended start 3.0)
- `direct_pose_loss_state_swing_boost` (a, recommended start 1.0)
- `direct_pose_loss_state_contact_source` (`gt|plan|meas`, recommended `gt`)
- `direct_pose_loss_state_scope` (`legs|limbs|all`, recommended `legs`)

### 12.3 Stage7 configs + train commands (Walk_F; current)

Configs:
- BASE (direct pose only): `config/posttrain_WalkF_stage7_directpose_medianquadtail_20260125.json`
- NEW  (leg head + SO(3) residual; leg-only): `config/posttrain_WalkF_stage7_directpose_legso3split_legonly_20260125.json`

Train:

```bash
python -m train.posttrain --config config/posttrain_WalkF_stage7_directpose_medianquadtail_20260125.json
python -m train.posttrain --config config/posttrain_WalkF_stage7_directpose_legso3split_legonly_20260125.json
```

Output ckpts:
- `models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage7_directpose_medianquadtail_20260125.pth`
- `models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage7_directpose_legso3split_legonly_20260125.pth`

(Older Stage7 configs like `*_direct_tail_state_weight_*` are kept only as historical artifacts.)

### 12.4 Evaluation commands (important: applyfull)

If you want the **final composed output** to reflect improvements (not only the direct expert),
evaluate with `--lambda_fusion_apply --so3_corr_apply` (**no hinge**).

Note: hinge is treated as a **debug-only** tool; do not include it in the default posttrain/eval pipeline.

Stage6 baseline (historical):

```bash
python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage6_direct_cond_anchor_20260124.pth \
  --depth 3 \
  --rounds 5 \
  --time-index-mode cycle --time-index-cycle-minus1 \
  --lambda_fusion_apply --so3_corr_apply \
  --out debug_output/_stage6_baseline_tidxm1_apply_lamso3/Walk_F_freerun_cycles.json
```

Stage7 (BASE, depth=3, applyfull):

```bash
python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage7_directpose_medianquadtail_20260125.pth \
  --depth 3 \
  --rounds 5 \
  --time-index-mode cycle --time-index-cycle-minus1 \
  --contact_plan_init_mode learnable+obs --contact_plan_init_hidden 128 --contact_plan_init_dropout 0.0 \
  --lambda_fusion_apply --so3_corr_apply \
  --export_joint_direct_geolocal_series \
  --out debug_output/_stage7_directpose_medianquadtail_fixbase_depth3_apply_lamso3_export_series46_jointdirect/Walk_F_freerun_cycles.json
```

Stage7 (NEW, depth=3, applyfull):

```bash
python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage7_directpose_legso3split_legonly_20260125.pth \
  --depth 3 \
  --rounds 5 \
  --time-index-mode auto \
  --contact_plan_init_mode learnable+obs --contact_plan_init_hidden 128 --contact_plan_init_dropout 0.0 \
  --lambda_fusion_apply --so3_corr_apply \
  --export_joint_direct_geolocal_series \
  --out debug_output/_stage7_directpose_legso3split_legonly_depth3_apply_lamso3_export_series46_jointdirect/Walk_F_freerun_cycles.json
```

### 12.5 Result summary (HISTORICAL; Stage7 tail reweight; 2026-01-24 run)

Overall (mean over rounds):
- `DirectGeoLocalDeg`: `0.64° -> 0.29°` (**-0.34°**, **-53%**, direct expert learned better)
- `BlendGeoLocalDeg`: `2.09° -> 1.94°` (**-0.15°**, **-7.2%**, final composed output improved)
- `GeoLocalDeg`: `2.61° -> 2.51°` (**-0.10°**, **-3.8%**, final output improved)

Diluted bones (direct expert):
- `calf_r`: direct mean `9.20° -> 0.98°`; worst SIC reduced (`SIC56 mean 26.27° -> SIC55 mean 5.39°`)
- `foot_r`: direct mean `1.53° -> 0.68°`; worst SIC reduced (`SIC54 mean 6.68° -> SIC86 mean 3.85°`)

Note: final improvement for a specific bone is mediated by `λ_eff` (e.g. `calf_r` often has low `λ_eff`,
so even if direct improves a lot, the final output may remain inc-dominated unless λ shifts).

### 12.6 Result summary (Stage7 leg head + SO(3) residual; leg-only; depth=3 eval; 2026-01-25 run)

Walk_F (A/B on `DirectGeoLocalDeg`, computed from `per_step_direct_geolocal_deg` with:
`cycle>=1` + `drop wrap_boundary_step` + `exclude root_idx`):

- BASE: `debug_output/_stage7_directpose_medianquadtail_fixbase_depth3_apply_lamso3_export_series46_jointdirect/Walk_F_freerun_cycles.json`
- NEW : `debug_output/_stage7_directpose_legso3split_legonly_depth3_apply_lamso3_export_series46_jointdirect/Walk_F_freerun_cycles.json`

Global (flatten joints):
- mean: `0.1957 -> 0.1805` (Δ `-0.0153`)
- p99:  `1.2417 -> 1.0185` (Δ `-0.2232`)
- max:  `3.4629 -> 3.1730` (Δ `-0.2899`)

Key bones (p99):
- `calf_l/calf_r` improve strongly; `foot_r` is the only clear regression point in this run (needs targeted follow-up).

---

## 13. Update (2026-01-25): replace 2D contact hint with `phase_z_in` in direct conditioning (Stage7 no-hinge)

### 13.1 Motivation (why D0 is not enough in practice)

In Walk_F Stage7 no-hinge export, the exported contact probabilities can be **low-bandwidth / high-entropy**:

- `contacts_meas` can collapse near `0.5` (high entropy, little stance/swing information).
- `contacts_plan` can sync with GT in phase order but stay in a compressed mid-range (weak within-stance/swing resolution).

This makes “2D contact probs as phase hint” a bottleneck for **fine phase‑locked spikes** (e.g. fixed SIC foot/ball z‑twist).

### 13.2 Change: `direct_pose_phase_z_mode=replace_contacts`

We added a routing knob to the direct head:

- `direct_pose_use_phase_z=true`: capture the pre-update `phase_z_in` (dim=`2*contact_dim`) and feed it to direct.
- `direct_pose_phase_z_mode=replace_contacts`: in `direct_pose_meas_mode='concat'`, use:

```
direct_in = [direct_feat(+time_pe), phase_z_in]    # replace (contacts_plan, contacts_meas)
```

This keeps the phase hint dimension `2*C` but removes the degenerate `contacts_*` features from direct conditioning.

### 13.3 Walk_F results (cycle>=1 && !wrap_boundary; n=344; λ+SO3 apply)

Comparison (step-mean):

- Baseline Stage7 no-hinge:
  - `GeoLocalDeg=2.5107`, `BlendGeoLocalDeg=1.9427`, `DirectGeoLocalDeg=0.2936`, `LambdaMean=0.7486`
- +`phase_z_in` (append mode; direct-only posttrain):
  - `GeoLocalDeg=2.5439`, `BlendGeoLocalDeg=1.9642`, `DirectGeoLocalDeg=0.2880`, `LambdaMean=0.7486`
- **+`phase_z_in` replace contacts (this update):**
  - `GeoLocalDeg=2.5073`, `BlendGeoLocalDeg=1.9142`, **`DirectGeoLocalDeg=0.1644`**, `LambdaMean=0.7486`

Key spike (Direct per-bone GeoLocalDeg):
- `ball_r`:
  - max: `6.1046°@SIC49 -> 5.4753°@SIC54 -> 5.1410°@SIC54`
  - SIC49 mean (4 samples): `6.0987 -> 4.2786 -> 3.7841`
  - SIC54 mean (4 samples): `5.7755 -> 5.4753 -> 5.1359`
- `foot_l` max: `5.3342°@SIC13 -> 3.3591°@SIC42 -> 3.1513°@SIC42`

Interpretation:
- Replacing 2D contact hint with higher-bandwidth `phase_z_in` improves direct’s fine phase locking substantially.
- `λ` stayed ~constant in this run, so final Blend improvements are modest unless `λ` is retrained/calibrated.

### 13.4 Artifacts (paths)

- ckpt:
  - `models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage7_direct_tail_state_weight_nohinge_phasezin_replacecontacts_nopt_20260125.pth`
- JSON (λ+SO3 apply; **full** per-step per-joint DirectGeoLocalDeg under `per_step_direct_geolocal_deg`):
  - `debug_output/_stage7_tail_state_nohinge_phasezin_replacecontacts_apply_lamso3_export_series46_full/Walk_F_freerun_cycles.json/Walk_F_freerun_cycles.json`
- JSON (λ+SO3 apply; compact step metrics only):
  - `debug_output/_stage7_tail_state_nohinge_phasezin_replacecontacts_apply_lamso3_export_series46/Walk_F_freerun_cycles.json`
- JSON (λ+SO3 apply; keybone per-step dict series):
  - `debug_output/_stage7_tail_state_nohinge_phasezin_replacecontacts_apply_lamso3_export_series46_jointdirect/Walk_F_freerun_cycles.json`

Repro (full export):

```bash
python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage7_direct_tail_state_weight_nohinge_phasezin_replacecontacts_nopt_20260125.pth \
  --depth 3 \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --out debug_output/_stage7_tail_state_nohinge_phasezin_replacecontacts_apply_lamso3_export_series46_full/Walk_F_freerun_cycles.json \
  --rounds 5 \
  --time-index-mode cycle \
  --time-index-cycle-minus1 \
  --lambda_fusion_apply \
  --so3_corr_apply \
  --contact_plan_init_mode learnable+obs \
  --contact_plan_inject_scale 1.0 \
  --contact_plan_time_bias_scale 1.0 \
  --phase_reset_source contacts_meas \
  --contacts_meas_source model \
  --direct_pose_meas_source model \
  --direct_pose_plan_source model \
  --export_joint_direct_geolocal_series
```

---

## 14. Walk_F workflow: leg-ω signed-scale head (oracle table supervised; posttrain)

This is the concrete “new workflow” used in `Walk_F` to **learn** a per-(sic,bone) ω scale (instead of applying an
oracle table only at freerun time). It relies on the higher-bandwidth phase hint routing (`phase_z_in -> direct`)
to make phase-locked behavior linearly accessible to the head.

Fixed eval mask (do not change): `cycle>=1` + `drop_wrap` + `excl_root`  
Metric: `per_step_direct_geolocal_deg["DirectGeoLocalDeg"]`

### 14.1 What it does

Enable a per-joint signed scale on leg ω (SO(3) mode only):

- `scale = sign * exp(clip * tanh(log_mag_raw/clip))`
- `sign = 2*sigmoid(sign_logit) - 1`  (soft ±1 and also “off” via sign≈0)
- `omega_eff = omega_raw * scale`

This single head can cover: `off / identity / amplify / flip`.

### 14.2 Recommended config (keep `direct_pose_leg_scale_sup_weight: 1.0`)

Oracle signed alpha-table:
- `debug_output/_posttrain_sicboost10_ext3234_20260204/oracle_alpha_tables/Walk_F_oracle_alpha_table_signed_v2_keep14_noflip_nocalf55.json`

Posttrain config (train only `direct_pose_leg_gate_head.*`):
- `config/posttrain_WalkF_stage7_legomega_baseline_alignproj_min0p5_sicboost10_ext3234_signedScaleV2_keep14_noflip_nocalf55_W1_clip4_20260205.json`

Key knobs (excerpt; do not change semantics across runs):

```json
{
  "direct_pose_use_phase_z": true,
  "direct_pose_phase_z_mode": "replace_contacts",
  "direct_pose_leg_mode": "so3",
  "direct_pose_leg_stopgrad_main": true,
  "direct_pose_leg_detach_feat": true,

  "direct_pose_leg_gate_mode": "signed_scale",
  "direct_pose_leg_scale_log_clip": 4.0,

  "direct_pose_leg_gate_train_only": true,
  "direct_pose_leg_scale_sup_weight": 1.0,
  "direct_pose_leg_scale_sup_alpha_table_json": "debug_output/_posttrain_sicboost10_ext3234_20260204/oracle_alpha_tables/Walk_F_oracle_alpha_table_signed_v2_keep14_noflip_nocalf55.json"
}
```

### 14.3 Commands (posttrain → freerun → eval)

Posttrain:

```bash
python -m train.posttrain \
  --config config/posttrain_WalkF_stage7_legomega_baseline_alignproj_min0p5_sicboost10_ext3234_signedScaleV2_keep14_noflip_nocalf55_W1_clip4_20260205.json
```

Freerun (same args as PREV-NEW; export DirectGeoLocal series):

```bash
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage7_legomega_baseline_alignproj_min0p5_sicboost10_ext3234_signedScaleV2_keep14_noflip_nocalf55_W1_clip4_20260205.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --out debug_output/_posttrain_sicboost10_ext3234_signedScaleV2_keep14_noflip_nocalf55_W1_clip4_20260205/new/C1_ttc_pred \
  --rounds 5 --time-index-mode auto \
  --phase_reset_source ttc_pred \
  --so3_corr_apply --lambda_fusion_apply \
  --export_joint_direct_geolocal_series
```

Eval (fixed mask):

```bash
python tools/summarize_direct_geolocal_masked.py \
  debug_output/_posttrain_sicboost10_ext3234_20260204/new/C1_ttc_pred/Walk_F_freerun_cycles.json \
  debug_output/_posttrain_sicboost10_ext3234_signedScaleV2_keep14_noflip_nocalf55_W1_clip4_20260205/new/C1_ttc_pred/Walk_F_freerun_cycles.json \
  --cycle_gte 1 --drop_wrap

python tools/report_ab_worstpoints.py --topn 25 --min-cycle 1 \
  --base debug_output/_posttrain_sicboost10_ext3234_20260204/new/C1_ttc_pred/Walk_F_freerun_cycles.json \
  --new  debug_output/_posttrain_sicboost10_ext3234_signedScaleV2_keep14_noflip_nocalf55_W1_clip4_20260205/new/C1_ttc_pred/Walk_F_freerun_cycles.json
```

Posttrain log sanity check (printed every step):
- `leg_scale=...` should decrease
- `logμ(tgt/pred)=.../...` should move pred toward tgt
- `sign=... tgt=... pred=...` should move `pred` toward `tgt` (y in [0,1])
