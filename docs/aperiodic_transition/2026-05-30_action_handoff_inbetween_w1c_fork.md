> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# W1c Fork Decision — A (grounded data) vs B (masked in-betweening)

Date: 2026-05-30

Update: 2026-05-31 (W1d LOGO decision lock)

## W1d LOGO — Binding Final Decision (pre-committed branch)

目标（按预承诺）:
- 对每个 grounded clip `g ∈ {Walk_L_To_L, Walk_R_To_L, Walk_R_To_R}` 做 LOGO；
- 绑定判据只用 action-only gate（masked 无 `hidden_pre`，reach non-binding）:
  `yaw_corr>0` + `heading_MAE_rad<0.25` + `pop_safe>0` + `best_pose_d` 不退化；
- 两种口径并报：`MIRROR-L_R`（去掉 `g` 的 grounded cross-manifold 监督，保留 `g` within-clip）
  与 `FULL-HOLDOUT`（`g` 完全不参与训练）；
- 每个 run 必存 state，recorded identity 正控必须继续通过。

W1d artifacts:
- `debug_output/_tmp_action_handoff_w1d_logo_20260531_fullsup/masked_smoke_summary.json`
- `debug_output/_tmp_action_handoff_w1d_logo_20260531_mirror_Walk_L_To_L/masked_smoke_summary.json`
- `debug_output/_tmp_action_handoff_w1d_logo_20260531_fullhold_Walk_L_To_L/masked_smoke_summary.json`
- `debug_output/_tmp_action_handoff_w1d_logo_20260531_mirror_Walk_R_To_L/masked_smoke_summary.json`
- `debug_output/_tmp_action_handoff_w1d_logo_20260531_fullhold_Walk_R_To_L/masked_smoke_summary.json`
- `debug_output/_tmp_action_handoff_w1d_logo_20260531_mirror_Walk_R_To_R/masked_smoke_summary.json`
- `debug_output/_tmp_action_handoff_w1d_logo_20260531_fullhold_Walk_R_To_R/masked_smoke_summary.json`

### Full-supervision reference (W1c v2 aligned)

| clip | yaw_corr | heading_MAE_rad | pop_safe | best_pose_d_mean | action-only pass |
|---|---:|---:|---:|---:|---|
| Walk_L_To_L | 0.9648 | 0.0058 | 0.35 | 0.0688 | True |
| Walk_L_To_R | -0.7797 | 0.0197 | 0.00 | 0.1342 | False |
| Walk_R_To_L | 0.9780 | 0.0625 | 0.35 | 0.0597 | True |
| Walk_R_To_R | 0.8414 | 0.0050 | 0.45 | 0.0535 | True |

### LOGO held-out clip rows (single-row verdict per run)

| holdout clip g | policy | yaw_corr (Δ vs fullsup-g) | heading_MAE_rad (Δ) | pop_safe (Δ) | best_pose_d_mean (Δ) | held-out action-only pass |
|---|---|---:|---:|---:|---:|---|
| Walk_L_To_L | MIRROR-L_R | 0.3018 (-0.6630) | 0.0182 (+0.0124) | 0.00 (-0.35) | 0.0855 (+0.0167) | **False** |
| Walk_L_To_L | FULL-HOLDOUT | -0.1018 (-1.0666) | 0.0217 (+0.0159) | 0.05 (-0.30) | 0.0880 (+0.0192) | **False** |
| Walk_R_To_L | MIRROR-L_R | 0.1360 (-0.8420) | 0.1142 (+0.0517) | 0.00 (-0.35) | 0.0953 (+0.0356) | **False** |
| Walk_R_To_L | FULL-HOLDOUT | 0.0209 (-0.9571) | 0.1250 (+0.0625) | 0.00 (-0.35) | 0.1009 (+0.0412) | **False** |
| Walk_R_To_R | MIRROR-L_R | -0.8365 (-1.6779) | 0.0368 (+0.0318) | 0.15 (-0.30) | 0.0997 (+0.0462) | **False** |
| Walk_R_To_R | FULL-HOLDOUT | -0.9626 (-1.8040) | 0.0559 (+0.0508) | 0.10 (-0.35) | 0.0946 (+0.0412) | **False** |

细化（MIRROR-L_R 的 fail 组件）:
- `Walk_L_To_L`: `pop_safe=0` 且 pose 退化（`best_pose_d` +0.0167）。
- `Walk_R_To_L`: `pop_safe=0`。
- `Walk_R_To_R`: `yaw_corr<0`。

旁证（MIRROR-L_R vs FULL-HOLDOUT）:
- 三个 `g` 在 MIRROR-L_R 全部不通过，`within-clip` 单独并不能让 held-out `g` 过门；
- FULL-HOLDOUT 同样不通过，且通常更差，说明 `g` 的 grounded cross-manifold 监督对 `g` 本身动作层结果是必需项。

Recorded identity 正控:
- 7/7 runs `recorded_identity_pass=true`（门本身未坏，结论不由 gate 退化造成）。

### H1/H2 binding verdict

- **H1 confirmed (memorization / no in-family generalization under holdout).**
- 按预承诺分支，终局必须落：**PARK**（B 判定数据受限/死路，当前不继续投入 B）。
- 并明确：**A(仅补 L_R 数据)单独也未必解锁**，因为连已 grounded 的 clip 去掉自身 grounded 监督后都不能稳定通过 action-only gate。

### Final direction + unlock spec

终局决策:
- **PARK**（不进 B4/seam、不改 action-only 阈值、不引入新 latent lever/injection/fine-tune）。

数据解锁规格（仅作为 future unpark contract，不是当前执行项）:
- 至少需构造真实 onset 样本满足 `contact_d<=0.30` 且 `pose_d<=0.05`；
- 且必须在同一 LOGO 协议下证明“去掉 `g` 自身 grounded 监督后，held-out `g` 仍能过 action-only gate”。

唯一可证伪下一步:
- 只接受一个反证门：在**不改阈值**（`yaw_corr>0`, `heading_MAE_rad<0.25`, `pop_safe>0`, pose 不退化）
  且 `recorded_identity_pass=true` 前提下，重跑本 W1d LOGO，若 `MIRROR-L_R` 三个 held-out `g`
  全部通过，则推翻本次 H1/PARK；否则保持 PARK。

重申:
- reach 分量对 masked 仍是 **non-binding**（space incompatibility: `281d state` vs `hidden_pre(512)`）。
- `tau_yaw=0.25 rad` 仍 **PROVISIONAL**，未在真实生成中间区间完成校准。

## Headline Conclusions (binding)

1. W1b proves metric/gate migration correctness, **not** turn-generation capability:
   M1 = G-B is a recorded/oracle **identity non-degeneracy control** (L_R: `best_pose_d=0`,
   `yaw_corr=1.0`, `heading_mae_rad=1.286e-8`, `pop_safe=1`, `self_reach_k3=1`), so the gate
   is not always-no; it **does not** calibrate `tau_yaw=0.25 rad` on real generated middle
   intervals. `tau_yaw` remains **PROVISIONAL**.
2. M2 stays in the headline: A's "O(1)" only applies when grounded onset assets already
   exist; if grounded `Walk_L_To_R` onset is absent, cost is animation/data authoring and is
   a blocking risk, not sampling cost.
3. STEP-0 read-only triage verdict is **(b)**: A is blocked by data authoring on `Walk_L_To_R`;
   selected path = **B (masked in-betweening smoke)**.

Artifacts:
- `debug_output/_tmp_action_handoff_w1c_step0_20260530/grounded_alignment_check_summary.json`
- `debug_output/_tmp_action_handoff_w1c_step0_20260530/inbetween_sampler_coverage_summary.json`
- `debug_output/_tmp_action_handoff_w1c_gate_migration_20260530/gate_migration_eval_summary.json`
- `debug_output/_tmp_action_handoff_w1c_step2b_masked_20260531_v2/masked_smoke_summary.json`
- `debug_output/_tmp_action_handoff_w1c_step2b_masked_20260531_v2/masked_smoke_state.pt`

## STEP 0 (read-only) — grounded provenance triage

判据（A viable）:
- 存在真实录制 `Walk_L_To_R` onset，且 `contact_d <= 0.30`、`pose_d <= 0.05`；
- sampler 可用真实 onset provenance 抽到（非 fallback metadata）。

### 三情形判定

| 情形 | 判据 | 本次数字 | 结论 |
|---|---|---|---|
| (a) 现有录制可清门 | onset 清 `contact<=0.30 && pose<=0.05` 且 provenance=grounded_ok | `Walk_L_To_R onset0: contact_d=0.7031, pose_d=0.0162, groundable=false`; sampler `grounded_ok_rate=0.0` | 否 |
| (b) 只有 fallback / later-onset 越 pose 门 | fallback window 内无清门；超窗可清 contact 但越 pose | 窗内 best: onset8 `contact_d=0.4730 > 0.30`; 窗外 onset10 起 `contact_d=0.2533` 但 `pose_d=0.0880 > 0.05`（onset16: `contact_d=0.0055`, `pose_d=0.1346`）; sampler `within_clip_fallback_rate=1.0` | **是** |
| (c) 边界含糊 | 接近门限需补证据 | 非含糊（和门限有明确间隔） | 否 |

补充:
- 其余三条 turn clip 全部 grounded 可抽到（`grounded_ok_rate=1.0`，fallback=0）。
- 因此 A 在当前数据下是 **A blocked (data authoring blocker)**。

## Selected Path: STEP 2(B) masked smoke (minimal, no B4/seam runtime)

执行约束:
- 5 clip、零新数据、同 W1b 联合字段（`self_reach/yaw/pop/pose`）、不进入 B4/seam/handoff runtime。
- 必存 state：`masked_smoke_state.pt` 已保存。

实现摘要:
- `tools/run_action_handoff_inbetween_masked_smoke.py`：context + future seam -> direct middle fill（non-AR）。
- 训练 tensor：`ctx [B,16,281]`, `seam [B,6,281]`, `middle [B,12,281]`, `float32`, `cpu`。
- 产出 joint-gate 所需字段并与 W1b AR-free baseline 并排比较。

训练 smoke:
- loss `0.8132 -> 0.0043`（min `0.0033`）。

### W1b fields per clip (candidate = masked, 281-d space)

| clip | self_reach_k3 | yaw_corr | heading_MAE_rad | pop_safe_rate | best_pose_d_mean | joint_pass |
|---|---:|---:|---:|---:|---:|---|
| Walk_L_To_L | 0.00 | 0.9648 | 0.0058 | 0.35 | 0.0688 | False |
| **Walk_L_To_R** | **0.00** | **-0.7797** | **0.0197** | **0.00** | **0.1342** | **False** |
| Walk_R_To_L | 0.00 | 0.9780 | 0.0625 | 0.35 | 0.0597 | False |
| Walk_R_To_R | 0.00 | 0.8414 | 0.0050 | 0.45 | 0.0535 | False |

观察:
- 动作层表象不是“仅 R_R 抬升”，而是 3/4 clip 都出现 `yaw_corr>0` 且 `pop_safe>0`
  （L_L/R_L/R_R 分别 `0.965/0.35`, `0.978/0.35`, `0.841/0.45`）。
- 唯一失败仍是 `Walk_L_To_R`（`yaw_corr=-0.780`, `pop_safe=0`, pose 退化）。
- 若按 W1b 全联合门（含 reach）直接判，`all_pass=false`, `l_to_r_pass=false`, `stop=true`。
  但该 reach 分量对 masked 口径不兼容（见下文“空间兼容性”）。

### pinned vs free (W1b migrated baseline, Walk_L_To_R)

|口径|self_k3|yaw_corr|heading_MAE_rad|pop_safe|best_pose_d / mean|
|---|---:|---:|---:|---:|---:|
| pinned_goal (AR baseline) | 0.00 | -0.8012 | 0.7598 | 0.00 | 0.1166 |
| free_goal (AR baseline) | 0.00 | -0.4796 | 0.6918 | 0.00 | 0.1166 |
| recorded identity control | 1.00 | 1.0000 | 1.286e-8 | 1.00 | 0.0000 |
| masked (this step) | 0.00 | -0.7797 | 0.0197 | 0.00 | 0.1342 |

## STEP 2(B) acceptance check (corrected)

- (i) recorded identity control must pass: **PASS** (`all_pass=true`)。
- (ii) 旧判据（any AR-free negative clip 有 lift）:
  - AR-free-negative clips: `Walk_L_To_R`, `Walk_R_To_R`
  - lifted: `Walk_R_To_R` -> **PASS（nonbinding）**

升格后的绑定判据（防 memorization 污染）:
- 先按 STEP0 provenance 分组：grounded=`{L_L,R_L,R_R}`，ungrounded=`{L_R}`。
- 只把 **ungrounded** 上的动作层 lift 记为泛化证据。
- 本次 `lifted_ungrounded_clips=[]`，`memorization_suspected=true`（grounded 有 lift，ungrounded 无）。
- 因此：`step2_pass_generalization_binding=false`。

绑定结论:
- `Walk_L_To_R`（唯一 ungrounded/泛化目标）仍 `yaw_corr<0`、`pop_safe=0`、pose 退化。
- B 当前表现是“grounded clip 可重建、ungrounded L_R 失败”，属于反 B1 证据，不构成可行性通过。

## 空间兼容性（reach 分量升格）

- masked 候选是 from-scratch `281-d state` 模型，没有 `hidden_pre`。
- W1b reach 条款在 `hidden_pre(512)` 空间；与 masked 的 `self_reach_k3`（281-d state）不可比。
- 结论：对 masked 的绑定判定应使用“动作层门”（`yaw_corr + heading_mae + pop_safe + pose`），
  reach 条款在该 artifact 上标记为 `reach_component_comparable=false`。

## Minimal falsifiable next gate for handoff

在不改 W1b 阈值前提下，B 的下一次最小验收改为两步：
1. 对 masked 先用动作层门（不含 hidden_pre reach）做泛化判定：L_R 必须同时 `yaw_corr>0`、`heading_mae<0.25`、`pop_safe>0`、`pose` 不劣化。
2. 若要回到 W1b 全联合门，需把 masked 建到 base/hidden_pre 空间再评 reach。
