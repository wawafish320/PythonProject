# 2026-04-28 `legacy_checkpoint_compat` 删除后的完整 rerun / 对比回报计划

Date: 2026-04-28  
Status: Draft / code-side strict-current cut landed locally; Stage 1 completed locally; Stage 2 full posttrain rerun completed locally; Stage 3 structured compare completed locally  
Scope: post-cut validation for `legacy_checkpoint_compat` dual-track removal  
Goal: 不只验证“代码删完能跑”，还要验证：

1. 迁移后的 legacy checkpoint 能被 strict 路径消费
2. 完整 `posttrain` rerun 能跑通
3. 与既有基线相比，没有出现需要人工阻断的明显行为回退

Non-goal:

- 本文档**不预设通过阈值**
- 本文档不替代删除 plan
- 本文档不顺手定义新的 metric taxonomy

---

## 0) TL;DR

post-cut rerun 的完成标准不是单个 smoke test 通过，而是：

1. legacy ckpt 先迁移成 strict ckpt
2. strict contract smoke 通过
3. 完整 posttrain 跑完
4. 结构化对比以下 4 组结果：
   - `2026-04-25 baseline`
   - `2026-04-26 strict-contract-fullchain-preflight`
   - `2026-04-27 resolved-config rerun`
   - `post-cut rerun`
5. 对每个 group 回报：
   - `mean`
   - `p50`
   - `p90`
   - `p95`
6. 再回报：
   - `delta_vs_0425`
   - `delta_vs_0426`
   - `delta_vs_current`

这里先**不定 hard threshold**，先把对比情况完整报出来，由人判断是否接受。

当前本地执行状态（2026-04-28）：

- 已完成：code-side strict/current 单轨切换已落地，本 doc 依赖的 parse/load/manifest fail-fast 边界已在本地实现。
- 已完成：`python3 tools/check_strict_checkpoint_contract_smoke.py` 通过；`python3 -m unittest tests.train.test_checkpoint_compat_removal` 通过。
- 已完成：代表性 ckpt 的 `Stage 1` migrate / strict smoke / 最小 posttrain 启动已执行并在本页回填；运行根目录为 `debug_output/_tmp_legacy_ckpt_gateA1_stage1_20260428_220911/`。
- 已完成：`Stage 1` 幂等性检查已收口为 fail-fast；对已 strict 的 `ckpt_gateA1_stage1_strict.pth` 再跑 migrate 会报 `[FATAL][AlreadyStrict]`。
- 已完成：`Stage 2` 已在 `debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/` 下跑完完整 lambda posttrain、freerun eval 与 `group_summary.json` 产物。
- 已完成：`Stage 3` 已生成结构化 compare 产物 `debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/stage3_compare_lambda_step200.json`。

---

## 1) 参考对比组

### A. 2026-04-25 baseline

产物：

- `debug_output/_tmp_tail_top7_fresh_chain_step360_20260425_030401/lambda_lr_branch_cmp/from72_lr1e4_s20_s120_s150/evals_lambda_apply/step_000200/group_summary.json`

执行前必须补齐：

- `seed=0`
- `config=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_tail_top7_fresh_chain_step360_20260425_030401/lambda_lr_branch_cmp/from72_lr1e4_s20_s120_s150/configs/lambda_from72_lr1e4_s20_s120_s150.json`
- `ckpt=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_tail_top7_fresh_chain_step360_20260425_030401/72_lr_branch_cmp/from71_lr1e4_s20_s120/checkpoints/72_main/ckpt_step_000150_WalkF_stage7_72_from71_lr1e4_s20_s120_20260425_030401.pth`

### B. 2026-04-26 strict-contract-fullchain-preflight

产物：

- `debug_output/_tmp_strict_contract_fullchain_preflight_20260426_173158/lambda_lr_branch_cmp/from72_lr1e4_s20_s120_s150/evals_lambda_apply/step_000200/compare_vs_0425_lambda_step200.json`

说明：

- 上方 `产物` 列出的 compare 文件是 derived compare，不是 raw `group_summary.json`。
- raw `group_summary.json` 路径已定位并写入下方 `raw_group_summary=` 字段（位于同一 run-root 下的 `evals_lambda_apply/step_000200/group_summary.json`，2345 bytes）。
- 0426 现在可以与 0425 / 0427 / post-cut 同口径填入主表；compare 文件保留为历史参考。

执行前必须补齐：

- `seed=0`
- `config=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_strict_contract_fullchain_preflight_20260426_173158/lambda_lr_branch_cmp/from72_lr1e4_s20_s120_s150/configs/lambda_from72_lr1e4_s20_s120_s150.json`
- `ckpt=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_strict_contract_fullchain_preflight_20260426_173158/72_lr_branch_cmp/from71_lr1e4_s20_s120/checkpoints/72_main/ckpt_step_000150_WalkF_stage7_72_from71_lr1e4_s20_s120_20260426_173158.pth`
- `raw_group_summary=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_strict_contract_fullchain_preflight_20260426_173158/lambda_lr_branch_cmp/from72_lr1e4_s20_s120_s150/evals_lambda_apply/step_000200/group_summary.json`

### C. 2026-04-27 resolved-config rerun

产物：

- `debug_output/_tmp_strict_stageB_resolvedcfg_rerun_20260427_224340/evals/lambda/step_000200/group_summary.json`

执行前必须补齐：

- `seed=0`
- `config=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_strict_stageB_resolvedcfg_rerun_20260427_224340/configs/lambda.json`
- `ckpt=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_strict_stageB_resolvedcfg_rerun_20260427_224340/lambda/checkpoints/ckpt_step_000200_lambda_resolvedcfg_20260427_224340.pth`

### D. Post-cut rerun

已执行并回填：

- `run_root=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017`
- `group_summary=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/evals/lambda/step_000200/group_summary.json`
- `compare_report=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/stage3_compare_lambda_step200.json`
- `seed=0`
- `config=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/configs/lambda_postcut.json`
- `migrated_strict_ckpt=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_gateA1_stage1_20260428_220911/checkpoints/ckpt_gateA1_stage1_strict.pth`
- `legacy_ckpt=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_tail_top7_fresh_chain_step360_20260425_030401/72_lr_branch_cmp/from71_lr1e4_s20_s120/checkpoints/72_main/ckpt_step_000150_WalkF_stage7_72_from71_lr1e4_s20_s120_20260425_030401.pth`（== §1.A `0425 baseline` 的 `ckpt=`，== 删除 plan `Gate A1` 使用的代表 ckpt；三处必须严格相同）

---

## 2) 删除后的必跑验证链

### Stage 1. 迁移代表性 legacy checkpoint

输入（已固定，不允许执行人临时挑选）：

- 路径：`/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_tail_top7_fresh_chain_step360_20260425_030401/72_lr_branch_cmp/from71_lr1e4_s20_s120/checkpoints/72_main/ckpt_step_000150_WalkF_stage7_72_from71_lr1e4_s20_s120_20260425_030401.pth`
- 该路径同时是本 doc §1.A `0425 baseline` 的 `ckpt=` 字段、删除 plan `Gate A1` 的代表 ckpt、以及 Stage 2 完整 posttrain rerun 的输入。三处必须严格相同；这里引用的是删除 plan `Gate A1`，不是旧版 Gate A 宽口径。

属性要求（基于删除 plan `Gate A0` 审计事实与 `Gate A-scope` 作用域豁免后的 `Gate A1` manifest-only 口径）：

- 缺 `resolved_build_manifest` / `resolved_build_manifest_hash`（仓内 ckpt 普世满足）
- `frozen_encoder.*` / `contact_plan_input_proj.*` key 命中**不再是 Stage 1 强制输入要求**：这不是漏写，而是删除 plan `Gate A0` 已提供现场审计事实依据，删除 plan `Gate A-scope` 已提供 post-cut 作用域豁免依据；因此本轮真正阻断删除的 live smoke 只剩删除 plan `Gate A1`。

Gate 口径对齐：

- `Gate A0` 提供审计事实依据
- `Gate A-scope` 提供作用域豁免依据
- `Gate A1` 才是本轮真正阻断删除的 live smoke

迁移产物路径（执行后即写入此处，不再使用占位符）：

- migrated strict ckpt：`/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_gateA1_stage1_20260428_220911/checkpoints/ckpt_gateA1_stage1_strict.pth`
- migrate report：`/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_gateA1_stage1_20260428_220911/reports/migrate_gateA1_stage1.log`
- strict contract smoke output：`/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_gateA1_stage1_20260428_220911/reports/check_strict_checkpoint_contract_smoke.log`
- 最小 posttrain smoke log：`/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_gateA1_stage1_20260428_220911/reports/posttrain_stage1_smoke.log`

当前状态（2026-04-28）：

- code-side prerequisite 已具备，且代表性 ckpt 的本节执行结果已回填到上方 4 条路径。
- migrate `dry_run` 显示 `legacy_hash == strict_hash` 且 config diff 为 `<none>`。
- migrated strict ckpt 已通过 strict/current load、fingerprint compare、1-step lambda smoke。
- 对该 strict ckpt 再次运行 migrate 现在会 fail-fast 报 `[FATAL][AlreadyStrict]`；不得再做 double-migrate 写出。

执行：

1. `tools/migrate_legacy_posttrain_ckpt.py`
2. `tools/check_strict_checkpoint_contract_smoke.py`
3. 最小 `train.posttrain` 启动
4. migrate 幂等性检查

执行后产物路径必须回填到本节上方"迁移产物路径"占位处。

要求：

- 报告 strict load 可吃
- 不再借助 legacy 开关
- 对同一 legacy ckpt 连续跑两次 migrate，结果必须满足二选一：
  - 第二次产物与第一次等价
  - 或第二次在“已是 strict / 不应 double-migrate”处 fail-fast
- 不允许出现 silent double-migrate 漂移

### Stage 2. 完整 posttrain rerun

要求：

- 使用固定数据 / 固定 config / 固定 seed / 固定代表性输入
- 跑完完整 posttrain，而不是只跑启动 smoke
- 产出 checkpoint、posttrain log、eval group summary
- `seed/config/legacy_ckpt` 三者必须在执行前写死进本文档；post-cut 不允许临时选别的对象

建议同时保存：

- config snapshot
- migrated ckpt path
- run root
- posttrain log json
- eval group summary

当前状态（2026-04-28）：**已完成**

产物：

- `run_root=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017`
- `config=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/configs/lambda_postcut.json`
- `input_strict_ckpt=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_gateA1_stage1_20260428_220911/checkpoints/ckpt_gateA1_stage1_strict.pth`
- `posttrain_log_json=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/lambda/checkpoints/posttrain_log_lambda_postcut_20260428_223017.json`
- `posttrain_main_log=<gap: this run was launched directly without tee; no separate raw stdout log file was persisted for Stage 2. Use the posttrain_log_json above as the authoritative per-step record.>`
- `final_checkpoint=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/lambda/checkpoints/ckpt_step_000200_lambda_postcut_20260428_223017.pth`
- `final_checkpoint_alias=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/lambda/checkpoints/ckpt_last_lambda_postcut_20260428_223017.pth`
- `eval_log=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/evals/lambda/step_000200/eval.log`
- `eval_cycles_json=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/evals/lambda/step_000200/eval_model_source/Walk_F_freerun_cycles.json`
- `group_summary=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/evals/lambda/step_000200/group_summary.json`

结果：

- strict/current path 可直接消费 `Gate A1` 的 migrated strict ckpt，无需 `legacy_checkpoint_compat` runtime branch。
- Stage 2 最终 freerun 指标与 `0425 baseline` 完全一致：
  - `all_ex_root`: `mean=0.1203484249`, `p50=0.0705285445`, `p90=0.2912226021`, `p95=0.4068961143`
  - `leg`: `mean=0.1781741089`, `p50=0.1373052895`, `p90=0.3665178716`, `p95=0.4564554095`
  - `nonleg`: `mean=0.1078455743`, `p50=0.0579889007`, `p90=0.2700274587`, `p95=0.3813521564`
  - `arm`: `mean=0.1258135008`, `p50=0.0629186258`, `p90=0.3237527907`, `p95=0.4414300323`
  - `else`: `mean=0.0653759300`, `p50=0.0520666875`, `p90=0.1429784149`, `p95=0.1783065200`

### Stage 3. 结构化对比回报

把 `post-cut rerun` 与 `0425/0426/0427` 同口径对比。

当前状态（2026-04-28）：**已完成**

结构化产物：

- `compare_json=/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/stage3_compare_lambda_step200.json`

结果摘要：

- `post-cut rerun` 与 `0425 baseline` 在 `all_ex_root / leg / nonleg / arm / else` 的 `mean/p50/p90/p95` 全量 20 个槽位完全一致，因此全量 `delta_vs_0425 = 0.0`。
- 相对 `0426` / `current(=0427 resolved-config rerun)` 的差值如下：
  - `all_ex_root`: `delta_vs_0426(mean/p50/p90/p95)=+0.0120251853/+0.0008249953/+0.0354283750/+0.0680672526`; `delta_vs_current=+0.0086903696/-0.0003928319/+0.0301561058/+0.0536658466`
  - `leg`: `delta_vs_0426(mean/p50/p90/p95)=+0.0096267363/+0.0039639920/+0.0408453643/+0.0369590819`; `delta_vs_current=+0.0108351863/+0.0044134259/+0.0491558909/+0.0319275260`
  - `nonleg`: `delta_vs_0426(mean/p50/p90/p95)=+0.0125437689/+0.0014107861/+0.0346337110/+0.0628352165`; `delta_vs_current=+0.0082266254/-0.0004087240/+0.0269665569/+0.0444848537`
  - `arm`: `delta_vs_0426(mean/p50/p90/p95)=+0.0187008451/+0.0044458881/+0.0526809692/+0.0887112916`; `delta_vs_current=+0.0141108180/+0.0020112395/+0.0285676718/+0.0595085919`
  - `else`: `delta_vs_0426(mean/p50/p90/p95)=-0.0020093203/+0.0001822338/-0.0109786838/-0.0057075322`; `delta_vs_current=-0.0056814660/-0.0038577504/-0.0239337981/-0.0219782293`

---

## 3) 必回报指标

按你确认的 5 组：

- `all_ex_root`
- `leg`
- `nonleg`
- `arm`
- `else`

每组都要报 4 个统计量：

- `mean`
- `p50`
- `p90`
- `p95`

另外为方便人工读数，再回报：

- `delta_vs_0425`
- `delta_vs_0426`
- `delta_vs_current`

说明：

- `current` 在本文统一指 `2026-04-27 resolved-config rerun`
- 这里只做对比回报，不提前规定阈值

---

## 4) 推荐回报格式

建议最终汇报固定成下面这种形式。

### 4.1 结构化产物优先

禁止手工抄一张 140+ 填空位的 markdown 表作为唯一数据源。

必做项：

1. 用一个小脚本读取 4 组输入路径，统一产出结构化 JSON / CSV
2. JSON / CSV 必须覆盖：
   - 5 个 group
   - `mean/p50/p90/p95`
   - `delta_vs_0425`
   - `delta_vs_0426`
   - `delta_vs_current`
3. markdown 只作为人工总结展示，不作为唯一事实来源

建议结构：

```json
{
  "all_ex_root": {
    "mean": {"0425": 0.0, "0426": 0.0, "current": 0.0, "post_cut": 0.0, "delta_vs_0425": 0.0, "delta_vs_0426": 0.0, "delta_vs_current": 0.0},
    "p50": {...},
    "p90": {...},
    "p95": {...}
  },
  "leg": {},
  "nonleg": {},
  "arm": {},
  "else": {}
}
```

### 4.2 人工总结表

在结构化 JSON 产出之后，再附一份易读版 markdown 总结。

#### `all_ex_root`

- `mean`
  - `0425`: `<value>`
  - `0426`: `<value>`
  - `current`: `<value>`
  - `post-cut`: `<value>`
  - `delta_vs_0425`: `<value>`
  - `delta_vs_0426`: `<value>`
  - `delta_vs_current`: `<value>`
- `p50`
  - `0425`: `<value>`
  - `0426`: `<value>`
  - `current`: `<value>`
  - `post-cut`: `<value>`
  - `delta_vs_0425`: `<value>`
  - `delta_vs_0426`: `<value>`
  - `delta_vs_current`: `<value>`
- `p90`
  - `0425`: `<value>`
  - `0426`: `<value>`
  - `current`: `<value>`
  - `post-cut`: `<value>`
  - `delta_vs_0425`: `<value>`
  - `delta_vs_0426`: `<value>`
  - `delta_vs_current`: `<value>`
- `p95`
  - `0425`: `<value>`
  - `0426`: `<value>`
  - `current`: `<value>`
  - `post-cut`: `<value>`
  - `delta_vs_0425`: `<value>`
  - `delta_vs_0426`: `<value>`
  - `delta_vs_current`: `<value>`

- 同上

其余 `leg / nonleg / arm / else` 同口径展开。

### 4.3 文字总结

最终文字总结建议只回答 3 件事：

1. 删除后 strict 路径是否完整跑通
2. migrated legacy ckpt 是否已被 strict 路径消费
3. `mean/p50/p90/p95` 对比下，相对 `0425/0426/0427` 的整体变化趋势

这里不需要硬判 “pass/fail threshold”，但必须把变化讲清楚。

---

## 5) 推荐补充产物

为了让最终是否提交远程可复核，建议保留：

- `post-cut` run root
- migrated checkpoint path
- strict contract smoke output
- posttrain log
- posttrain log json
- step checkpoint paths
- final checkpoint path
- eval `group_summary.json`
- 如有现成 compare script，也保存 compare json

---

## 6) 切后必须额外检查的非指标项

除了数值对比，还要 grep / 检查：

### 6.1 仓内代码 / 配置

确认不再存在 live入口：

- `legacy_checkpoint_compat=true`
- `--strict_current_model_build false`
- `_apply_direct_pose_ckpt_compat`
- `_LEGACY_STRIPPED_CHECKPOINT_PREFIXES`

### 6.2 新 run root / logs

确认不再出现 legacy operator guidance：

- `legacy_checkpoint_compat=true`
- `shape/posttrain_cfg inference`
- `chain_hop-waiver`
- `chain-hop-report-only` 以外的任何 `waiver` 词根 operator hint
- `fingerprint waiver`
- `direct_pose temp compat`

### 6.3 新 ckpt contract

确认新 ckpt：

- 带 `resolved_build_manifest`
- 带 `resolved_build_manifest_hash`
- strict contract smoke 可通过

---

## 7) 提交远程前 checklist

只有下面都完成后，才建议统一提交到远程：

- [ ] 删除代码已经 landed
- [ ] 代表性 legacy ckpt 已 migrate 成 strict ckpt
- [ ] strict contract smoke 通过
- [ ] 最小 posttrain smoke 通过
- [ ] 完整 posttrain rerun 通过
- [ ] 已产出 `post-cut rerun` 的 `group_summary.json`
- [ ] 已完成 `0425 / 0426 / 0427 / post-cut` 的 `mean/p50/p90/p95` 对比回报
- [ ] 已检查新 run root / logs 不再出现 legacy operator guidance
- [ ] 已确认无需要阻断提交的明显回退

当前状态（2026-04-28）：

- 已完成：代表性 legacy ckpt migrate、strict contract smoke、最小 posttrain smoke、完整 posttrain rerun、`group_summary.json` 产出、`0425 / 0426 / 0427 / post-cut` 对比回报、以及新 run root forbidden-guidance grep。
- 尚未完成：是否存在“需要阻断提交的明显回退”的最终人工判定仍由 compare 结果与 Gate B inventory 一并收口；`posttrain_main_log` 本轮没有单独 tee 到文件，只保留 `posttrain_log_json`。

---

## 8) 必做自动化

本轮必须有一个小脚本 / 命令，把 4 组输入路径读入后统一吐出结构化 compare 产物。

最低要求：

- 输入 4 组 raw `group_summary` 路径
- 输出 JSON 或 CSV
- 覆盖：
  - `mean/p50/p90/p95`
  - `delta_vs_0425`
  - `delta_vs_0426`
  - `delta_vs_current`

允许的结果：

- 30 行脚本
- shell + jq 小工具
- 现有 compare 工具扩充

不允许的结果：

- 只靠人工抄表完成最终结论

当前状态（2026-04-28）：

- 已完成：结构化 compare 产物已落盘到 `debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/stage3_compare_lambda_step200.json`，覆盖 4 组输入路径、5 个 group、`mean/p50/p90/p95` 与 `delta_vs_0425 / delta_vs_0426 / delta_vs_current`。
