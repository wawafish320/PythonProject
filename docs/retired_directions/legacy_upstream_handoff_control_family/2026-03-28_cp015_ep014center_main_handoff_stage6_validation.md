# CP015 ep014center：basetrain -> offline selector -> exact Stage6 主链验证

> Status: archived legacy upstream / handoff / control record
> Reader note: this file belongs to the old-boundary upstream-control investigation; any `current`, `default`, `canonical`, `recommend`, or `mainline` wording below is historical context, not present-tense repo policy.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

> Last updated: 2026-03-28  
> 目的：在 `main worktree` 上验证一条最小链路是否可用：
>
> `basetrain -> offline selector scoring -> exact Stage6 ranking`
>
> 约束：
>
> - 不把评分内化到训练主循环
> - 不新增 `best_handoff` 训练内保存逻辑
> - 不修改 canonical downstream posttrain 语义
> - 只围绕一个 `cp015 ep014center` basetrain 配置做验证

---

## 0. TL;DR

本轮最关键的结论有四条：

1. 这条最小链路已经在主工作区跑通：
   - basetrain 成功训练并额外保存 `epoch12-15`
   - offline selector 成功消费 `ckpt_epoch_*`
   - exact Stage6 ranking 成功产出最终排序
2. offline selector 最终选中的是 `epoch012`
3. exact Stage6 ranking 的最终 top-1 是 `last`，对应 `basetrain epoch015`
4. 从这颗 Stage6 winner 继续往 downstream 走时，当前固定的 `70a` 选择应为 `lr=3e-4`

因此本轮支持的主结论是：

> **建议保留 `12-15` 作为后续标准 handoff scoring 准备窗口；但不建议把窗口内某一个 epoch 直接固化为默认 handoff，仍需要 exact Stage6 ranking 做最终裁决。**

并且在当前 `main worktree` 上，可以进一步固定一条当前可执行的 downstream 结论：

> **这条 `cp015 ep014center` 主链目前应写成：`last(epoch015) -> exact Stage6 winner -> 70a(lr=3e-4)`。**

---

## 1. 实验配置与产物路径

### 1.1 配置

- `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_seed2024_20260324.json`

### 1.2 输出目录

- `out_dir`: `models/cp015_corridor_hold_phasea050_fixedsched_ep014center_seed2024_20260324`
- `exp_dir`: `models/cp015_corridor_hold_phasea050_fixedsched_ep014center_seed2024_20260324/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_seed2024_20260324`

### 1.3 关键结果文件

- basetrain summary:
  - `models/cp015_corridor_hold_phasea050_fixedsched_ep014center_seed2024_20260324/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_seed2024_20260324/basetrain_keybone_group_summary.json`
- offline selector summary:
  - `debug_output/_tmp_ep014center_main_selector_check/handoff_selector_summary.json`
- exact Stage6 ranking:
  - `debug_output/_tmp_ep014center_main_anchor_check/stage6_anchor_table.json`
- downstream `70a(lr=3e-4)` config:
  - `debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json`
- downstream `70a(lr=3e-4)` ckpt:
  - `models/__tmp_ep014center_70a_lowlr_sweep_20260328/lr3e4/ckpt_last_WalkF_stage7_70a_lr3e4_from_ep014center_stage6winner_20260328.pth`
- downstream `70a(lr=3e-4)` eval summary:
  - `debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/eval_model_source/lr3e4_group_summary.json`

---

## 2. Checkpoint 保存结果

本轮要求的训练期 checkpoint 均已生成：

| artifact | status |
|---|---|
| `ckpt_best_free_*` | OK |
| `ckpt_best_teacher_*` | OK |
| `ckpt_last_*` | OK |
| `ckpt_epoch_012.pth` | OK |
| `ckpt_epoch_013.pth` | OK |
| `ckpt_epoch_014.pth` | OK |
| `ckpt_epoch_015.pth` | OK |

这说明：

- basetrain 主流程可以在不改 `best_free / best_teacher / last` 语义的前提下，额外导出 `ep12-15` 的 exact handoff 候选；
- offline selector 与 exact Stage6 ranking 后续都能直接消费这些 `ckpt_epoch_*`。

---

## 3. Basetrain 全程数据

### 3.1 逐 epoch 训练 / teacher / valfree 指标

| epoch | train_loss | teacher_geo_local_deg | valfree_geo_local_deg | valfree_geo_deg | root_vel_mae | ang_vel_mae |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.231042 | 0.711849 | 7.206639 | 7.191139 | 0.313253 | 0.829448 |
| 2 | 0.157075 | 0.362607 | 4.923769 | 5.424226 | 0.314470 | 0.583591 |
| 3 | 0.142202 | 0.186235 | 4.801520 | 5.052631 | 0.339200 | 0.601967 |
| 4 | 0.128307 | 0.137882 | 3.865810 | 4.312442 | 0.318614 | 0.493103 |
| 5 | 0.122221 | 0.127103 | 3.963114 | 4.440526 | 0.301853 | 0.527740 |
| 6 | 0.090373 | 0.094930 | 4.038974 | 4.306288 | 0.325440 | 0.532557 |
| 7 | 0.086586 | 0.085112 | 4.078839 | 4.268487 | 0.334686 | 0.533348 |
| 8 | 0.092244 | 0.078350 | 4.051696 | 4.298395 | 0.314184 | 0.516938 |
| 9 | 0.082408 | 0.085178 | 3.968090 | 4.372337 | 0.329870 | 0.527979 |
| 10 | 0.077156 | 0.068268 | 3.788931 | 3.994953 | 0.296173 | 0.479216 |
| 11 | 0.075055 | 0.071165 | 3.806179 | 4.124241 | 0.315903 | 0.492129 |
| 12 | 0.062145 | 0.071000 | 3.924356 | 4.191607 | 0.298749 | 0.507394 |
| 13 | 0.061826 | 0.070779 | 3.876409 | 4.013767 | 0.277487 | 0.492997 |
| 14 | 0.061417 | 0.068339 | 3.843380 | 4.054204 | 0.318405 | 0.490654 |
| 15 | 0.061354 | 0.067293 | 3.814029 | 4.100449 | 0.296299 | 0.495153 |

### 3.2 basetrain summary 抽取

- `best_teacher_by_GeoLocalDeg = epoch015`
  - `GeoLocalDeg = 0.067293`
  - `KeyBoneGeoLocalDegMean = 0.095969`
  - `group_mean.leg = 0.116282`
  - `group_mean.arm = 0.221309`
  - `group_mean.trunk = 0.068299`

- `best_free_by_GeoDriftSlopeProxy = epoch010`
  - `GeoLocalDeg = 3.788931`
  - `FreeRunGeoLocalDeg = 3.788931`
  - `KeyBoneGeoLocalDegMean = 6.599779`
  - `GeoDriftSlopeProxy = 0.514429`
  - `group_mean.leg = 7.467869`
  - `group_mean.arm = 3.915375`
  - `group_mean.trunk = 1.776074`

### 3.3 basetrain 视角下的窗口读数

从 `10-15` 看：

- teacher 最优点清楚落在 `15`
- valfree broad scalar 最优点落在 `10`
- `12-15` 是明显的 late stable window，但内部 winner 并不单一

这正是后面 selector / exact Stage6 必须继续判别的原因。

---

## 4. Offline selector 完整结果

### 4.1 结果摘要

- `selected_candidate = epoch012`
- `exact_epoch_candidates = [epoch012, epoch013, epoch014]`
- proxy epoch top-5:
  1. `epoch013`
  2. `epoch010`
  3. `epoch015`
  4. `epoch011`
  5. `epoch014`

这说明：

- proxy scan 已经明显把注意力拉到 `12-15` 附近；
- 但它还没有“自动收敛到 ep014”，而是更偏向 `13 / 15 / 14` 这一带；
- 真正的 exact selector winner 最终是 `epoch012`。

### 4.2 selector candidate 表

| candidate | selector | epoch | guardrail_pass | score | contact_err | plan_gt | phase_shift | leg8 | SIC12-15 foot_l/ball_l | calf_r_global |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `epoch012` | `epoch_exact` | 12 | yes | 1.55 | 0.287352 | 0.400326 | 2.000000 | 10.8013 | 14.5840 | 15.7303 |
| `best_free` | `best_free` | 10 | yes | 2.95 | 0.290468 | 0.401505 | 4.666667 | 10.8507 | 14.6108 | 15.9216 |
| `best_teacher` | `best_teacher` | 15 | yes | 3.00 | 0.319753 | 0.404413 | 4.000000 | 10.6375 | 14.5439 | 15.3827 |
| `epoch014` | `epoch_exact` | 14 | yes | 3.55 | 0.299462 | 0.398781 | 4.000000 | 10.8836 | 14.6300 | 16.1086 |
| `epoch013` | `epoch_exact` | 13 | yes | 3.95 | 0.290468 | 0.401505 | 4.666667 | 10.8507 | 14.6108 | 15.9216 |
| `last` | `last` | 15 | no | invalid | 0.300190 | 0.394619 | 4.333333 | 10.9250 | 14.6484 | 16.2147 |

### 4.3 selector 解读

#### `epoch012` 为什么会赢 offline selector

它拿到了最强的 basetrain-side coarse handoff profile：

- 最低 `contact_err_abs_mean = 0.287352`
- 第二低 `contact_plan_gt_abs_mean = 0.400326`
- 最低 `phase_shift_contact_plan_abs_mean = 2.0`
- `leg8 / SIC12-15 / calf_r_global` 也都维持在前列

这让它在当前 v1 rank formula 下获得最优综合分 `1.55`。

#### `last` 为什么没有赢 offline selector

`last` 并不是 broad 指标完全差，而是：

- 触发了 `geo_local_guardrail`
- 导致 `score_valid = false`

也就是说，这轮 offline selector 不是把 `last` 判成“绝对最差”，而是认为它**不满足 basetrain-side coarse promote guardrail**。

这也是本轮最重要的结构性现象之一：

> basetrain-side selector 与 exact Stage6 final ranking 的 winner 并不一致。

---

## 5. Exact Stage6 ranking 完整结果

### 5.1 结果摘要

- `anchor = last`
- 即：exact Stage6 top-1 来自 `selector=last`, `basetrain_epoch=15`
- `stage6_good_set = [last]`
- `stage6_bad_set = [epoch014, best_teacher, epoch012, best_free, epoch013]`

### 5.2 exact Stage6 排名表

| rank | candidate | selector | epoch | all_ex_root_mean | leg_mean | nonleg_mean | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 | red_flag |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `last` | `last` | 15 | 0.312873 | 0.847077 | 0.197369 | 1.292959 | 0.450189 | none |
| 2 | `epoch014` | `epoch_exact` | 14 | 0.367248 | 0.854386 | 0.261921 | 1.344749 | 1.351731 | `calf_r` |
| 3 | `best_teacher` | `best_teacher` | 15 | 0.380183 | 0.901717 | 0.267419 | 1.055943 | 1.554634 | `leg`, `calf_r` |
| 4 | `epoch012` | `epoch_exact` | 12 | 0.383953 | 0.953613 | 0.260784 | 1.310002 | 0.403901 | `leg` |
| 5 | `best_free` | `best_free` | 10 | 0.393234 | 0.890161 | 0.285790 | 2.084045 | 0.489073 | `nonleg`, `foot` |
| 6 | `epoch013` | `epoch_exact` | 13 | 0.393234 | 0.890161 | 0.285790 | 2.084045 | 0.489073 | `nonleg`, `foot` |

### 5.3 exact Stage6 解读

#### 为什么 `last(epoch015)` 是最终 top-1

`last` 是唯一一个：

- `all_ex_root_mean` 排名第 1
- `leg_mean` 排名第 1
- `nonleg_mean` 排名第 1
- 且 **无 red flag**

它在 broad exit quality 上拿到了最稳的 Stage6-only outcome，因此被 exact Stage6 ranking promote 为 anchor。

#### 为什么 `epoch014` 没能超过 `last`

`epoch014` 的 broad 指标已经非常接近最终最优：

- `all_ex_root_mean = 0.367248`（第 2）
- `leg_mean = 0.854386`（第 2）

但它的：

- `calf_r_sic2_4_mean = 1.351731`

进入了 red-flag 区，因此最终止步于第 2。

#### 为什么 `epoch012` 没能保持 selector 第一

`epoch012` 在 Stage6-only 下的强项是：

- `calf_r_sic2_4_mean = 0.403901`（第 1）
- `nonleg_mean = 0.260784`（第 2）

但它的：

- `leg_mean = 0.953613`

在 ranking 中成为 red-flag 项，导致它只能排到第 4。

#### `epoch013` 的特殊现象

`epoch013` 与 `best_free` 共享同一个 Stage6 replay 结果：

- `shared_stage6_alias_of = tmp_ep014center_main_selector_check__best_free`
- `state_dict_digest` 也相同

也就是说，本轮 `epoch013` 没有提供额外 downstream value，而是 alias 到 `best_free`。

---

## 6. Downstream 70a 固定结论

### 6.1 当前固定的 70a 路径与学习率

对这条 `cp015 ep014center` 主链，当前不建议继续沿用默认 `70a(lr=1e-3)`。

当前固定的 downstream `70a` 选择应为：

- learning rate: `3e-4`
- config:
  - `debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json`
- model ckpt:
  - `models/__tmp_ep014center_70a_lowlr_sweep_20260328/lr3e4/ckpt_last_WalkF_stage7_70a_lr3e4_from_ep014center_stage6winner_20260328.pth`

### 6.2 为什么固定到 `70a(lr=3e-4)`

从同一颗 exact Stage6 winner 出发的 `70a` 小 sweep 结果如下：

| variant | lr | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `70a` default | `1e-3` | 0.283011 | 0.650486 | 0.203557 | 0.225091 | 1.239521 | 0.559893 |
| `70a` lowlr | `5e-4` | 0.209380 | 0.463853 | 0.154359 | 0.179431 | 0.540162 | 1.478219 |
| `70a` lowlr | `3e-4` | 0.192487 | 0.443163 | 0.138287 | 0.160615 | 0.686596 | 0.286713 |

这说明：

- `5e-4` 会把 `foot_l/ball_l@SIC12-15` 明显压低，但会明显拉坏 `calf_r@SIC2-4`
- `3e-4` 是当前最平衡的点
- 相比默认 `1e-3`，`3e-4` 同时改善了：
  - `all_ex_root`
  - `leg`
  - `nonleg`
  - `arm`
  - `foot_l/ball_l@SIC12-15`
  - `calf_r@SIC2-4`

因此当前推荐不是“继续使用默认 70a”，而是明确固定到：

> `70a(lr=3e-4)`

---

## 7. 这轮数据说明了什么

### 7.1 `12-15` 作为准备窗口是合理的

这轮证据是明确支持 `12-15` 窗口的：

- offline selector winner = `epoch012`
- exact Stage6 top-1 = `epoch015`
- exact Stage6 top-2 = `epoch014`

也就是说，真正有用的候选已经集中在 `12-15` 这一段，而不是继续偏向更早或更晚的大范围 sweep。

### 7.2 不能把 basetrain proxy / coarse selector 当 final ranker

本轮最关键的实证就是：

- basetrain-side coarse selector winner = `epoch012`
- exact Stage6 final winner = `epoch015`

因此 current handoff process 的正确分层应该是：

1. basetrain 负责导出 `12-15` 等 exact candidate window
2. offline selector 负责 coarse prefilter / candidate narrowing
3. exact Stage6 ranking 负责 final promote

### 7.3 `best_free` 单点不足以代表 downstream best handoff

这轮里：

- `best_free = epoch010`
- 但 exact Stage6 排名只到第 5

说明只依赖 `best_free` 做 downstream handoff 已经明显不够。

---

## 8. 本轮对后续流程的直接建议

### 建议 1：保留 `12-15` 为标准 handoff scoring 准备窗口

后续 basetrain 默认应继续：

- 保存 `epoch12-15`
- 不需要把窗口扩大成更大的 dense sweep
- 先把 `12-15` 作为 canonical handoff preparation window

### 建议 2：不要把单个 epoch 直接写死

当前不建议把：

- `epoch012`
- 或 `epoch014`
- 或 `epoch015`

直接写死成默认 handoff。

因为这轮已经证明：

- selector winner 与 exact Stage6 winner 会分离
- 同一个 window 内仍需要 exact downstream-aware 复判

### 建议 3：继续把 exact Stage6 ranking 作为 final tie-break

即使 offline selector 已经很有用，它当前更合适的定位仍然是：

- `candidate discovery`
- `coarse prefilter`

而 exact Stage6 ranking 才应继续承担：

- `final exact handoff decision`

### 建议 4：这条 lane 的 downstream 先固定 `70a(lr=3e-4)`

当前这条 lane 在 exact Stage6 winner 之后，建议先固定：

- Stage6 winner ckpt:
  - `models/__tmp_ep014center_main_anchor_check/tmp_ep014center_main_selector_check__last/ckpt_last_tmp_ep014center_main_selector_check__last_stage6_anchor_ep014center_main_check.pth`
- 70a winner ckpt:
  - `models/__tmp_ep014center_70a_lowlr_sweep_20260328/lr3e4/ckpt_last_WalkF_stage7_70a_lr3e4_from_ep014center_stage6winner_20260328.pth`
- 70a learning rate:
  - `3e-4`

---

## 9. 最终结论

本轮 `main worktree` 主链验证给出的最终结论是：

> **是否建议把 `12-15` 作为后续标准 handoff scoring 准备窗口？**
>
> **建议。**

但需要补一句同样重要的 operational caveat：

> **不要把 `12-15` 中某一个 epoch 直接固化为默认 handoff；应继续使用 exact Stage6 ranking 做最终裁决。**

更具体地说：

- `12-15 window` 是对的
- `best_free single-point` 不够
- `offline selector` 有价值，但更适合作为 prefilter
- `exact Stage6 ranking` 仍是 final promote authority
- downstream `70a` 在这条 lane 上当前应固定为 `lr=3e-4`
