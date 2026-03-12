# 2026-03-06 TrainBase 改动范围冻结：仅做 Stage6 前置拆分

Last updated: 2026-03-06

## 1) 结论（Scope Freeze）

本轮改动只做一件事：

- 将 Stage6 的 `direct split-first` 思路前置到 trainbase（即：在主训练阶段提前引入 split 训练口径）。

本轮明确不做：

- 不改 Stage7（70a/70b/70c/70R/71/72）策略。
- 不改 lambda final 校准逻辑。
- 不改 contacts source / provider / pretrain_contact 路由策略。
- 不改 event_clock 机制与参数语义。
- 不改 posttrain 主流程编排与运行顺序。

---

## 2) 方向证据（Data Evidence）

提前拆分（在 trainbase 前置 split-head）只由下面两类信号驱动：

- 触发条件 A（Basetrain 终点 group 收敛状态）：
  若 `arm/else` validation error 明显高于 `leg`，且可排除“任务本身难度差异”，则说明动态权重未充分消除梯度竞争，应考虑提前拆分。
- 触发条件 B（Post-train 初始化效率）：
  若 post-train 前几个 epoch 主要在“纠正”basetrain 学到的不平衡表征，则说明 basetrain 起点对 split-head 不友好，应考虑提前拆分。

### 2.1 数据来源（仅使用起跑前诊断链路）

- `debug_output/__tmp_basetrain_bestfree_groupdist_20260305/group_summary.json`
- `debug_output/__tmp_basetrain_bestfree_groupdist_20260305/posttrain_stage6_init_stats.json`
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`

### 2.2 触发条件 A 证据：Basetrain 终点 group 收敛状态

基于 `group_summary.json`（`cycle>=1`, `drop_wrap=true`）：

- `leg mean = 10.6688°`
- `arm mean = 6.6670°`
- `else mean = 2.0487°`
- `nonleg mean = 5.2940°`
- `arm/leg = 0.625x`
- `else/leg = 0.192x`

判定：

- 当前样例没有出现“`arm/else` 明显高于 `leg`”的形态。
- 因此本次样例对触发条件 A 的判定是：**未触发**。
- 这也意味着：不能用 A 作为“立即提前拆分”的直接依据。

### 2.3 触发条件 B 证据：Post-train 前期是否主要在纠偏

基于 `posttrain_stage6_init_stats.json`：

- step1: `dir_leg_base=0.1919`, `dir_nonleg_base=0.0624`, `leg_over_nonleg=3.075x`
- head20 mean: `leg_over_nonleg=3.550x`
- step1 gradient: `direct_grad_norm_out_arm / direct_grad_norm_out_else = 7.709x`

基于同一 run 的 `posttrain_log_*`（epoch 1-5 均值）：

- `leg_over_nonleg`：`3.787 -> 4.113 -> 3.852 -> 3.486 -> 3.715`（ep1->ep5 仅 `-1.9%`）
- `arm/else` 输出梯度比：`7.192 -> 5.805 -> 5.502 -> 4.771 -> 5.006`（ep1->ep5 `-30.4%`）
- 同期 `dir_leg_base` 与 `dir_nonleg_base` 都下降约 `82.7%` / `82.8%`，但两者比例长期维持在 `~3.5-4.1x`

解读：

- 不平衡表征在 Stage6 起点即存在，并在前 5 个 epoch 内持续。
- 前期优化中出现明显的“梯度再平衡动作”（`arm/else` 梯度比明显回落）。
- 因而本次样例对触发条件 B 的判定是：**有触发迹象（中等置信）**，说明存在可观的初始化纠偏成本。

### 2.4 本次样例结论（是否支持提前拆分）

- 条件 A：未触发。
- 条件 B：有触发迹象（中等置信）。
- 结论：本样例支持“先做 trainbase 前置拆分 A/B 验证”，但不支持“仅凭当前一次观测就直接全量切换”。

执行口径（固定）：

1. 仅以 A/B 两类信号决定是否提前拆分（不再用其它链路结果替代）。
2. A、B 都不触发：不提前拆分。
3. 仅 B 触发：先做受控 A/B（小流量或单配置验证）。
4. A+B 同时触发：优先级最高，可进入提前拆分方案。

> 注：以上证据用于“是否值得前置拆分”的决策，不等价于单一因果证明；最终以 trainbase A/B 实验结果为准。

---

## 3) 本轮允许改动边界

仅允许以下类型改动：

1. trainbase 入口补齐 Stage6 前置拆分所需配置开关（参数可见、可解析、可落盘）。
2. trainbase 训练损失中的 direct 分支按 split 口径计算（仅限 direct loss 相关路径）。
3. 增加与 split 相关的最小必要日志/统计字段，便于检查是否生效。

---

## 4) 本轮禁止改动边界

以下全部保持现状：

1. posttrain 的阶段目标切换（`train_direct_pose` / `train_lambda_head`）不变。
2. 70R/71/72 的 train-only 冻结策略不前置到 trainbase。
3. lambda reliability / gate supervision / boundary weighting 等 lambda 相关策略不迁移。
4. contact 相关 contract（whitebox 默认、validate A/B 来源口径）不改。
5. 任何历史链路兼容性清理动作（如 phase/provider 退役）不在本轮内追加。

---

## 5) 验收口径（仅针对“前置拆分是否生效”）

最小验收：

1. trainbase 运行时能识别并记录 split 前置配置。
2. direct loss 日志可区分 leg / non-leg 统计（至少能判断 split 生效与否）。
3. 不触发 Stage7 / lambda / contact contract 的行为变化。

失败即回滚到“只保留参数壳，不启用新损失逻辑”。

---

## 6) 备注

- 本文档只定义“本轮范围边界”，不替代具体实现设计文档。
- 若后续需要引入 70R/71/72 或 lambda 相关机制，需单独开新问题文档并重新过 scope 审核。
