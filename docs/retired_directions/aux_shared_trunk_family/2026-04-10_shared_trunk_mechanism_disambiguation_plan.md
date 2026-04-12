# 2026-04-10 shared trunk mechanism disambiguation plan

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Purpose: 用**最少实验数**区分当前主问题更像  
> `capacity saturation`、`gradient conflict / supervision redundancy`、还是 `attach mismatch`
>
> Scope: 只做机制判别，不做新 feature 设计，不做新大 sweep  
> Rule: 优先复用现有 `DSN aux-leg` 链路与现成开关；先在 `stage6` 做最小判别，**不要**把每个诊断臂都一路推进到 `70a -> 70b`

## 1. Current state of evidence

截至 2026-04-10，可以认为下面两件事已经基本成立：

1. **不是 branch-side capacity 不足**
   - 现有 matched sham + aux 结果更支持问题位于 `shared trunk`
   - branch expansion / extra head 的历史解释需要整体降级

2. **但 trunk-side 机制还没有分清**
   - 当前仍未区分：
     - `H1. capacity saturation`
     - `H2. gradient conflict / supervision redundancy`
     - `H3. attach mismatch`

换句话说，已知答案是：

> “主矛盾在 shared trunk，不在 split 后 branch 容量”

尚未知的答案是：

> “shared trunk 里的主导失败机制到底是哪一种”

## 2. Decision target

本计划不追求一次性证明全部真相，而是要用最少实验把问题压缩到下面的决策树：

```text
Q1. harm 是否依赖 aux gradient 真的进入 shared trunk？
  ├─ 否 → 结构扰动 / head-side competition 更可疑
  └─ 是 → 进入 Q2

Q2. 当前 shared trunk hidden 本身是否已经包含“可被小 readout 读出”的 leg 信息？
  ├─ 是 → 更像 gradient conflict / supervision redundancy
  └─ 否 → 进入 Q3

Q3. 如果把 aux attach 移到更晚、更 leg-specific 的边界，是否恢复？
  ├─ 是 → attach mismatch
  └─ 否 → 更像 capacity saturation / no usable leg signal
```

## 3. Recommended experiment count

建议：

- **3 个核心实验**
- **+ 1 个可选确认实验**

其中：

- 前 **2 个** 可以直接用当前仓库已有能力完成
- 第 **3 个** 只有在前两步仍留下 `capacity vs attach mismatch` 歧义时才需要
- 第 **4 个** 只是下游确认，不是必须

## 4. Fixed protocol

所有核心判别实验都固定在 **`stage6 native`** 完成：

- 原因：
  - 这是最早出现有效信息的 stage
  - 成本最低
  - 若 `stage6` 都没有正窗口，没必要把每个诊断臂都继续推到 `70a/70b`

统一固定 recipe：

| item | value |
| --- | --- |
| donor family | `cp015 tailk7 rankmix tw020 canonical donor` |
| config | `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_20260401.json` |
| ckpt_in | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth` |
| epochs | `8` |
| steps_per_epoch | `60` |
| lr | `3e-4` |
| encoder_bundle | `models/motion_encoder_equiv.pt.best.pt` |
| direct_pose_use_phase_z | `true` |
| direct_pose_phase_z_mode | `concat` |

统一比较指标：

- `DirectGeoLocalDeg`
- `all_ex_root`
- `leg`
- `nonleg`
- `arm`
- `else`

附加诊断指标：

- `aux leg loss` / `aux leg geo loss` 训练曲线
- 若日志里已有：`direct_pose_aux_leg_weight`
- 若需要：最终 `direct_pose_aux_leg_head` 训练 loss 降幅

统一控制臂：

- `baseline`
- `sham`
- `aux`

已存在时直接复用当前 `DSN aux-leg` 记录，不重复训练。

## 5. Core experiment set

### E1. `aux_detach` — test whether trunk-directed aux gradient is necessary

#### Goal

回答：

> 当前伤害是否必须依赖 aux gradient 真正回传进 `shared trunk`？

#### Config diff

在现有 `aux` 基础上只改：

```json
{
  "direct_pose_aux_leg_enable": true,
  "direct_pose_aux_leg_weight": 0.2,
  "direct_pose_aux_leg_detach_feat": true,
  "direct_pose_aux_leg_log_enable": true
}
```

其余保持和 `aux` 臂完全 matched。

#### Why this is valid

当前实现里，`direct_pose_aux_leg` 固定接在 `direct_pose_head` shared trunk output；  
`direct_pose_aux_leg_detach_feat=true` 会让 aux head 仍然存在、仍然 forward、仍然优化自己的参数，但**不再把 aux loss 梯度送回 shared trunk**。

所以它正好回答：

- 是 `trunk-directed companion gradient` 在伤害主路？
- 还是只要“多一个头、多一个损失、多一条优化分支”就会坏？

#### Readout

比较：

- `baseline`
- `sham`
- `aux`
- `aux_detach`

#### Decision rule

若：

- `aux_detach ≈ sham`
- 且显著优于 `aux`

则说明：

- 伤害**依赖** aux gradient 真正进入 shared trunk
- `pure structure-only perturbation` 不是主要伤害来源
- 问题更偏向 `gradient conflict / redundancy / attach mismatch`

若：

- `aux_detach ≈ aux`

则说明：

- 就算不把 aux gradient 回传 trunk，额外头/额外优化分支本身也足以带来主要伤害
- 更像 `structural fork / head-side competition`

> 结合现有 sham 结果，预期更可能看到 `aux_detach ≈ sham`，而不是 `aux_detach ≈ aux`

---

### E2. `frozen_trunk_aux_readability` — test whether current shared trunk hidden is already leg-readable

#### Goal

回答：

> 当前 `shared trunk hidden` 本身，是否已经包含足够的 leg 信息，能被一个小 aux readout 读出来？

这是区分：

- `gradient conflict / redundancy`
- vs `capacity saturation / no usable signal`

的关键一步。

#### Design

保持 aux 结构与 aux loss 打开，但**冻结 shared trunk**：

- `direct_pose_head.*` param group `lr=0.0`

aux head 正常训练：

- `direct_pose_aux_leg_head.*` 正常 lr

推荐只冻结 shared trunk，不冻结 aux head；其余是否冻结主 readout 可以保持默认，先不额外加复杂度。

#### Minimal config mechanism

通过 `optimizer_param_group_overrides` 完成，不改代码。

示意：

```json
{
  "direct_pose_aux_leg_enable": true,
  "direct_pose_aux_leg_weight": 0.2,
  "direct_pose_aux_leg_detach_feat": false,
  "direct_pose_aux_leg_log_enable": true,
  "optimizer_param_group_overrides": [
    {
      "name": "freeze_shared_trunk",
      "lr": 0.0,
      "module_prefixes": ["direct_pose_head"]
    }
  ]
}
```

#### Readout

主要不看 downstream group summary 是否立刻变好，优先看：

- aux loss 是否显著下降
- 下降速度是否接近当前正常 `aux` 臂

#### Decision rule

若：

- shared trunk 冻结后，aux head 仍能把 aux loss 明显降下来

则说明：

- 当前 shared trunk hidden **已经包含可读的 leg 信息**
- 问题更不像“纯 capacity 不足 / 根本没有 leg signal”
- 更像 `gradient conflict / redundancy`

若：

- shared trunk 冻结后，aux loss 几乎不降，或明显弱于正常 `aux`

则说明：

- 当前 attach 点看到的 shared hidden **本身就不够 leg-readable**
- 剩余歧义压缩为：
  - `attach mismatch`
  - 或 `capacity saturation / no usable leg signal`

---

### E3. `late_attach_probe` — only run if E2 still leaves `attach vs capacity` ambiguity

#### Goal

回答：

> 当前失败到底是“shared trunk output 这个 attach 点不对”，还是“整个 trunk 到这里就已经没有足够 leg signal”？

#### Scope

这是本计划里唯一可能需要**最小代码扩展**的一步。  
如果 E1/E2 已经足够把问题收敛到 `gradient conflict / redundancy`，则**不要跑**这一步。

#### Minimal implementation

给 aux head 增加一个**纯诊断用** attach 选项：

- `shared_trunk`（当前默认）
- `leg_boundary`

推荐 `leg_boundary` 候选：

- `direct_pose_leg_head` hidden
- 或 `direct_pose_out_leg` input

注意：

- 这不是 redesign
- 只是在同一个 aux objective 下换一个更晚的 attach tap
- 仍然要求有 matched `sham` / `branch`

#### Compare

只比较两组：

1. `late_attach_sham`
2. `late_attach_aux`

不需要重做一大堆 matrix。

#### Decision rule

若：

- `shared_attach` 失败
- 但 `late_attach_aux` 相对 `late_attach_sham` 出现清晰净增益

则说明：

- 更像 `attach mismatch`

若：

- `late_attach_aux` 依然没有净增益

则说明：

- 更像 `capacity saturation / no usable leg signal in current trunk pipeline`

## 6. Optional confirmation

### E4. Promote only the single surviving explanation to `70a -> 70b`

只有当 E1-E3 中有一条臂在 `stage6` 给出**明确机制信号**时，才把它推进：

- `70a native`
- `new70b_replace_lowdrift`

目的不是重新做全面比较，而只是确认：

> 这个机制判别结论是否会在 downstream handoff 后保留

如果 `stage6` 没有机制信号，就不要浪费预算下推。

## 7. Expected outcome map

### Pattern A

- `E1 aux_detach ≈ sham`
- `E2 frozen_trunk_aux_readability = yes`

Interpretation:

- 最像 `gradient conflict / supervision redundancy`
- 当前 trunk 不是没信号，而是**一旦让 aux gradient 参与 trunk 更新，就开始竞争/覆盖**

### Pattern B

- `E1 aux_detach ≈ sham`
- `E2 frozen_trunk_aux_readability = no`
- `E3 late_attach = yes`

Interpretation:

- 最像 `attach mismatch`
- 当前 aux attach 点没有作用到真正决定 leg 的表示边界

### Pattern C

- `E1 aux_detach ≈ sham`
- `E2 frozen_trunk_aux_readability = no`
- `E3 late_attach = no`

Interpretation:

- 最像 `capacity saturation / no usable leg signal`
- 继续加监督、换权重、换 schedule 都意义不大

### Pattern D

- `E1 aux_detach ≈ aux`

Interpretation:

- 更像 `extra-head structural fork / head-side competition`
- 这条结果会比当前直觉更反常；若出现，优先先复查日志与 control matching

### Pattern E (observed after E4)

- `E1 aux_detach ≈ sham`
- `E2 frozen_trunk_aux_readability = partial yes`
- `E3 late_attach = readability yes, rollout no`
- `E4 full-epoch aux-vs-freerun correlation = positive`
- `E4 shared_attach epoch 7 -> 8 = late reversal`
- `E4 shared_attach vs aux_detach endpoint = nearly same aux loss, large freerun gap`

Interpretation:

- primary near mechanism is `trunk-directed aux-gradient interference`
- `supervision–rollout mismatch` is retained but weaker, because E4 did **not** produce the hoped-for clean within-arm monotone divergence
- inside that primary bucket, the remaining unresolved split is:
  - `sign conflict`
  - vs `capacity / plasticity sink`
- next step should prefer:
  - `seed-first` late-phase confirmation
  - then a small gradient-path probe that distinguishes `conflict` vs `sink`
  - before direct rollout-aware objective redesign

## 8. Recommended execution order

建议严格按下面顺序：

1. 复用现有 `baseline / sham / aux`
2. 跑 `E1 aux_detach`
3. 跑 `E2 frozen_trunk_aux_readability`
4. 只有 `E2` 留下歧义时，才跑 `E3 late_attach_probe`
5. 只有出现清晰机制信号时，才跑 `E4` downstream confirmation

## 9. Why this plan is minimal

这份计划故意避免：

- 新 trunk 设计
- 新 branch 设计
- recipe sweep
- attach point 大网格
- aux weight 大 sweep
- downstream 全链路重复开火

因为当前真正缺的不是更多 trial，而是：

> 一个能把 `capacity`、`conflict/redundancy`、`attach mismatch` 真正分开的最小判别序列

## 10. Bottom-line recommendation

如果只允许先做 **2 个实验**，就做：

1. `E1 aux_detach`
2. `E2 frozen_trunk_aux_readability`

这两步已经足够回答：

- 问题是不是 `pure branch capacity`
- aux gradient 是否真的是主要伤害源
- 当前 trunk hidden 里是否已经存在可读的 leg signal

如果这两步之后还剩唯一歧义，再加：

3. `E3 late_attach_probe`

---

## Final recommendation

> **建议按 “2 + 1” 的结构执行：先做 `aux_detach` 和 `frozen_trunk_aux_readability`；只有它们没法把问题压缩到单一解释时，再做一个最小 `late_attach` probe。**
