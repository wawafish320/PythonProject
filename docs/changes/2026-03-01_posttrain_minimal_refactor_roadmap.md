# [2026-03-01] `train/posttrain.py` 最小重构路线图（v2.1）

Date: 2026-03-01  
Status: Active v2.1（加入净减法硬约束）  
Scope: `train/posttrain.py`（本轮不新增脚本文件，不跨文件迁移）  
Goal: 在**不改变训练语义**前提下，先做重复收敛，再拆巨函数，降低维护成本与回归风险。  
Non-goal: 不改 loss 数学定义、不改默认超参行为、不引入新算法。

---

## 0) 当前策略（先减法，再分层）

统一执行顺序：

1. **Phase A: 去重与映射收敛（低风险）**
   - 先做重复代码块删除、常量/映射合并。
   - 再做小规模函数拆分（仅在重复已删除后）。
   - 要求净减法：抽出后必须删掉原重复实现。
2. **Phase B: 拆分巨函数（中/高风险）**
   - 只按“职责边界”拆，不按行数硬拆。
   - 一次只拆一个子流程，保证可回滚。

核心原则：
- one step, one commit
- 每步必须可回归、可回退
- 任何一步失败直接回到上一步 commit

新增硬约束（自本版本起强制执行）：
- 只做**净减法重构**：`train/posttrain.py` 总行数必须下降。
- 去重阶段（Phase A）`def` 总数不得增加。
- 拆分阶段（Phase B）`def` 可小幅增加，但必须同时满足：总行数净下降、最大函数长度下降。
- 禁止“只抽函数不删旧逻辑”。
- 禁止纯包装函数（仅搬一层调用、无边界收敛/重复删除价值的 helper）。
- 不新增文件，不跨文件迁移，不改行为。
- 每步必须附 before/after 指标（至少包含 LOC 与 `def` 数）。
- 推荐记录命令：
  - `wc -l train/posttrain.py`
  - `rg -n "^def " train/posttrain.py | wc -l`

---

## 1) 已落地（v2.1 当前）

已完成（仅低风险改动）：
- 合并 CLI override key 列表，消除 `main` 中重复清单维护。
- 合并 direct leg gate 同义词映射（配置解析与建模处复用同一映射）。
- 合并 optional CSV / SIC 字段归一化逻辑。
- 合并 int/float clamp 重复片段。
- 合并 `direct_pose_feat_source` 归一化规则。
- 合并多处分散的 `requires_grad_(True)` 循环（解冻路径）。
- 收敛 `main` 中 payload 覆写样板，抽成单一入口。

本轮新增（Phase A / Step A3 + A4，Phase B / Step B1-1 + B1-2 + B1-3 + B2）：
- 继续收敛 `main` 的 payload 覆写样板，保持 parser 参数集合不变。
- 将 `main` 内 payload 构建调用收口到单一入口，减少入口样板分散。
- 合并 CLI 覆写 special key 集合（模块级常量），删除运行时 `special_keys.update(...)` 样板。
- 去掉 `main` 内 checkpoint `posttrain_cfg` 提取的冗余 try/except 样板，改为等价显式判型提取。
- Step B1-1：将 `main` 中训练循环与保存输出拆分为编排 helper：
  - `_run_training_loop(...)`
  - `_save_posttrain_outputs(...)`
- Step B1-2：将 `main` 中数据集/加载器构建拆分为编排 helper：
  - `_build_dataset_and_loader(...)`
- Step B1-3：将 `main` 中 trainer 构建与 dataset-normalizer 绑定拆分为编排 helper：
  - `_build_model_and_trainer(...)`
- Step B2：将 `_cfg_from_payload` 按职责边界拆分：
  - `_cfg_reject_retired_targets(...)`
  - `_cfg_parse_path_basic(...)`
  - `_cfg_parse_direct_pose(...)`
  - `_cfg_parse_lambda_rollout(...)`
  - `_cfg_from_payload(...)`（仅保留编排组装）

Step A3 指标（before/after）：
- `main` 中 payload 构建样板：5 行 -> 1 行（调用点）。
- `_apply_cli_overrides` 调用参数：3 个 -> 2 个（去掉调用点显式 `args_map` 传递）。

Step A4 指标（before/after）：
- LOC：`5069 -> 5055`（-14）
- `def` 数：`59 -> 58`（-1）
- 最大函数长度：`1733 -> 1726`（`main`，-7）
- 目标重复块 #1：`special_keys.update(...)`：`2 -> 0`
- 目标重复块 #2：`ckpt_posttrain_cfg` try/except 提取样板：`1 -> 0`

Step B1-1 指标（before/after）：
- LOC：`5055 -> 5044`（-11）
- `def` 数：`58 -> 60`（+2）
- 最大函数长度：`1726 -> 1605`（`main`，-121）
- 编排 helper 数：`0 -> 2`（训练循环 / 输出保存）

Step B1-2 指标（before/after）：
- LOC：`5044 -> 5030`（-14）
- `def` 数：`60 -> 61`（+1）
- 最大函数长度：`1605 -> 1571`（`main`，-34）
- 编排 helper 数：`2 -> 3`（新增数据集/加载器构建）

Step B1-3 指标（before/after）：
- LOC：`5030 -> 4981`（-49）
- `def` 数：`61 -> 62`（+1）
- 最大函数长度：`1571 -> 1444`（`main`，-127）
- 编排 helper 数：`3 -> 4`（新增 model/trainer 构建）

Step B2 指标（before/after）：
- LOC：`4981 -> 4941`（-40）
- `def` 数：`62 -> 66`（+4）
- 最大函数长度：`1444 -> 1440`（`main`，-4）
- `_cfg_from_payload`：巨函数解析逻辑 -> 4 个职责 helper + 1 个编排入口

已验证：
- `python -m py_compile train/posttrain.py` 通过。
- `python -m train.posttrain --help`（退出码 0）通过。

---

## 2) 下一步执行计划

## Phase A（本轮）完成状态

- Step A3、A4 已完成，且满足 Phase A 结构门禁：`def_after <= def_before` 且 `LOC_after < LOC_before`。
- parser 参数集合未改；payload 覆写与 checkpoint cfg 提取逻辑保持语义等价。

---

## Phase B：职责边界拆分（B1 + B2 已完成）

### Step B1 — 先拆 `main` 的编排层（中风险）

建议拆成 3~4 个编排 helper（只搬运，不改计算）：
- `_build_dataset_and_loader(...)`
- `_build_model_and_trainer(...)`
- `_run_training_loop(...)`
- `_save_outputs(...)`

当前进度：
- 已完成：`_run_training_loop(...)`、`_save_posttrain_outputs(...)`、`_build_dataset_and_loader(...)`、`_build_model_and_trainer(...)`
- Step B1 完成。

约束：
- 不改变任何已有参数传递。
- 不改日志 key，不改 checkpoint key。
- 可小幅增加 `def`，但必须降低最大函数长度，且总 LOC 继续净减。
- 禁止纯包装函数；新增 helper 必须同时删除原地重复/冗余逻辑。

回归门：
- `py_compile`
- direct/lambda 各 1 组最小 smoke（`epochs=1, steps_per_epoch=20`）

---

### Step B2 — 拆 `_cfg_from_payload`（中风险，已完成）

目标：降低配置解析维护成本（已达成）。

已落地边界：
- path/basic 解析
- direct-pose 解析
- lambda/rollout 解析
- 最终 dataclass 组装

当前结构：
- `_cfg_reject_retired_targets(...)`：TTC / legacy target fail-fast
- `_cfg_parse_path_basic(...)`：路径与 run metadata
- `_cfg_parse_direct_pose(...)`：direct-pose + gate supervision 相关解析
- `_cfg_parse_lambda_rollout(...)`：核心训练 + lambda/rollout 相关解析
- `_cfg_from_payload(...)`：仅做编排 merge + dataclass 构建

约束：
- 历史上下文：当时保留了旧 alias（含 `direct_pose_leg_gate_loss_weight` 兼容 alias）；该 alias 已于 `2026-04-15` 从 mainline 删除，现仅接受 `direct_pose_leg_gate_sup_weight`。
- default 行为不变（含 `rollout_include_boundary` 的 `rollout_cycles>1` 自动默认）。

回归门：
- `py_compile` / `--help` 通过。
- 关键 alias/default/fail-fast 行为 spot-check 通过。

---

### Step B3 — 最后拆 `_lambda_fusion_loss_rollout`（高风险）

这是最后做的步骤。

建议边界：
- rollout 初始化
- 单步 forward + 误差
- 正则项汇总
- stats 汇总

强制要求：
- 提供 legacy/new A-B 对照（同 batch、同 seed）。
- 关键统计 key 集合一致。

当前进度（2026-03-01）：
- 已修复 `_lambda_rollout_unroll_steps(...)` 上下文解包问题（`locals()` 字典按字符串 key 读取）。
- 已完成最小 A-B 对照（同 seed，同 batch replay）：
  - direct smoke：`keyset_match=1`, `max_abs_diff=0.000000e+00`
  - lambda smoke：`keyset_match=1`, `max_abs_diff=0.000000e+00`
- 已清理 B3 临时 A-B debug 回放路径（`POSTTRAIN_ROLLOUT_AB_COMPARE` 相关分支）与
  `direct_grad_norm_out_nonleg_legacy` 统计键。
- legacy 目标 fail-fast 兼容骨架保留（`_LEGACY_TARGET_KEYS` / `_enabled_legacy_targets` /
  `[FATAL][LEGACY_TARGET_RETIRED]` / 隐藏 CLI 兼容入口），并通过静态 guard。

Step B3 指标（before/after，本次工作树）：
- LOC：`5347 -> 5275`（-72）
- `def` 数：`74 -> 71`（-3）
- 最大函数长度：`1440 -> 1440`（`main`，0）

---

## 3) 不可触碰区（本轮）

以下内容本轮不做语义改动：
- geodesic / SO(3) 计算公式
- lambda 融合与 reliability 数学定义
- checkpoint 兼容策略（key 命名与保存结构）
- 默认训练超参与默认分支行为

---

## 4) 回归与验收标准

每个 commit 最低门禁：
1. `python -m py_compile train/posttrain.py`
2. `python -m train.posttrain --help`

每个 commit 必报结构指标（before/after）：
1. `train/posttrain.py` 总行数（LOC）
2. `train/posttrain.py` `def` 数量
3. `train/posttrain.py` 最大函数长度（按行数）
4. 本步目标重复块计数（自定义，例如某重复字面量/模式出现次数）

分阶段结构门禁：
- Phase A（去重）：`def_after <= def_before`，且 `LOC_after < LOC_before`
- Phase B（拆分）：允许 `def` 小幅上升，但必须 `LOC_after < LOC_before` 且 `max_func_len_after < max_func_len_before`

当前基线（2026-03-01，约束生效点）：
- LOC = 5069
- `def` 数 = 59

当前工作树（A4 完成后）：
- LOC = 5055
- `def` 数 = 58
- 最大函数长度 = 1726（`main`）

当前工作树（B1-1 后）：
- LOC = 5044
- `def` 数 = 60
- 最大函数长度 = 1605（`main`）

当前工作树（B1-2 后）：
- LOC = 5030
- `def` 数 = 61
- 最大函数长度 = 1571（`main`）

当前工作树（B1-3 后）：
- LOC = 4981
- `def` 数 = 62
- 最大函数长度 = 1444（`main`）

当前工作树（B2 后）：
- LOC = 4941
- `def` 数 = 66
- 最大函数长度 = 1440（`main`）

每个阶段门禁（A/B 分界处）：
1. direct smoke（最小步数）
2. lambda smoke（最小步数）
3. 能正常写出 `ckpt_last_*.pth` 与 `posttrain_log_*.json`

建议关注指标（同配置对比）：
- `total`
- `dir_geo`
- `inc_geo`
- `lambda_mean`
- `gate_sup_loss`（若启用）

---

## 5) 提交建议（commit plan）

1. `refactor(posttrain): dedupe constants and alias maps`
2. `refactor(posttrain): dedupe parser/payload override boilerplate`
3. `refactor(posttrain): split main orchestration into helper stages`
4. `refactor(posttrain): split config payload parsing by concern`
5. `refactor(posttrain): stage lambda_fusion rollout into subroutines`

说明：任何时候只要 smoke 波动异常，立即停在当前 commit，不继续下一步。
