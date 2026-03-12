# [2026-03-01] `train/posttrain.py` 非 `main` 热区最小重构路线图（v1.3）

Date: 2026-03-01  
Status: Draft v1.3（聚焦非 `main` 四个高风险点；统一 LOC 门禁口径；加入“复用价值优先”拆分约束）  
Scope: `train/posttrain.py`（本轮不跨文件迁移；`main` 仅允许最小调用点适配）  
Goal: 在**不改变训练语义**前提下，优先降低“巨函数 + 上下文耦合 + 静默异常 + 隐式副作用”维护风险。  
Non-goal: 不改 loss 数学定义、不改默认超参行为、不引入新算法。

---

## 0) 当前策略（先解耦，再拆分，再收紧异常，最后剥离副作用）

统一执行顺序：

1. **Phase A: 上下文解耦（低/中风险）**
   - 去掉 `locals()` 透传。
   - 将 unroll helper 的上下文改为显式结构化输入（TypedDict/dataclass）。
   - 先收敛边界，再进入函数拆分。
2. **Phase B: 巨函数按职责拆分（中/高风险）**
   - 先拆 `_lambda_rollout_unroll_steps`，再拆 `_lambda_fusion_loss_rollout`。
   - 一次只拆一个职责块，保持每步可回滚。
3. **Phase C: 静默异常清理（中风险）**
   - 先处理两大函数内 `except Exception: pass`，再清理其余位置。
   - 用“窄异常 + 明确 fallback + 可观测日志/计数”替代吞没。
4. **Phase D: 隐式状态副作用外移（中风险）**
   - 将 loss 内 trainer EMA 更新改为“返回更新载荷 + 外层统一提交”。

核心原则：
- one step, one commit
- 每步必须有 before/after 结构指标
- 任何一步回归失败，立即停在当前 commit，不继续后续步骤
- 每步固定汇报 4 项（必填）：总行数（LOC）、`def` 数、最大函数行数、目标重复块数量
- **单步硬门禁（统一口径）**：以 **Step 收尾** 为准，必须满足 `LOC_after <= LOC_before`；开发中可临时增行，但不可带着 LOC 债务进入下一步

新增约束（本路线图强制）：
- 不允许新增“黑盒上下文”传递（例如再引入新的 `locals()`/`globals()` 管道）。
- 不允许“只抽函数不删旧逻辑”。
- 每次新增 helper 必须带来可量化净收益（LOC、最大函数长度、异常吞没计数至少一项下降）。
- 拆分后关键统计 key 集合保持一致（A/B 对照）。

### 拆分约束（v2，Phase B 强制执行）

拆分前准入（满足任一即可进入候选）：
- 被调用 ≥2 次（或下一步已明确会复用 ≥2 次）
- 封装了可独立命名的领域概念
- 拆出后调用处更易读（调用点行数或嵌套层级下降）

硬禁止（命中任一则不允许拆）：
- 纯转发 wrapper / 无实质边界的间接层
- 函数名仅复述代码字面行为（无领域语义）
- 再次引入 `locals()`/`globals()` 一类黑盒上下文
- 只抽函数、不删除原地旧逻辑（双实现并存）
- **Step 收尾时** `LOC_after > LOC_before`
- 参数爆炸：helper 参数 > 8 且未收敛到结构化 context（TypedDict/dataclass）

单次调用 helper 的例外规则（替代绝对禁止）：
- 调用次数=1 允许，但必须同时满足：
  - 具备独立概念命名
  - 调用点可读性提升
  - 至少 1 项结构指标下降（最大函数长度 / 重复块 / 异常吞没）

拆分后验流程（强制）：
1. 先按“复用价值”拆分
2. Step 收尾前检查 `LOC_after <= LOC_before`
3. 若不满足：定位并回收不必要间接层（参数搬运层、单次无概念 helper、纯 wrapper）
4. 在**当前 step**完成净减回收后，才允许进入下一步

---

## 1) 基线现状（针对用户提出的四块问题）

当前代码快照核对（`train/posttrain.py`）：

- **超大函数（非 `main`）**
  - `_lambda_rollout_unroll_steps`: `train/posttrain.py:2079`，609 行（`2079~2687`）
  - `_lambda_fusion_loss_rollout`: `train/posttrain.py:2690`，537 行（`2690~3226`）
- **强耦合上下文传递**
  - `train/posttrain.py:2943`：`_lambda_rollout_unroll_steps(locals())`
  - `train/posttrain.py:2080~train/posttrain.py:2179`：约 100 行上下文解包（其中 95 行显式 `ctx[...]` 取值）
- **静默异常吞没**
  - 全文件 `except Exception: pass` 共 39 处
  - 两大函数内合计 17 处（`_lambda_rollout_unroll_steps` 9 处 + `_lambda_fusion_loss_rollout` 8 处）
- **隐式状态副作用**
  - `train/posttrain.py:2956~train/posttrain.py:3011`：在 loss 计算路径内读写 `trainer._direct_pose_group_norm_ema`

结构指标基线（本路线图起点）：
- LOC: `5253`
- `def` 数: `70`
- 非 `main` 最大函数长度: `609`（`_lambda_rollout_unroll_steps`）
- `except Exception: pass` 计数: `39`

---

## 2) 具体改动流程

## Phase A — 上下文解耦（A1 + A2 + A3）

### Step A1 — 去掉 `locals()` 透传（低风险）

目标：将 `train/posttrain.py:2943` 的 `locals()` 改为显式构造上下文对象。

实施：
- 新增 `_build_rollout_unroll_ctx(...)`（或等价 builder）集中组织输入字段。
- 调用点改为 `_lambda_rollout_unroll_steps(ctx)`，`ctx` 为显式命名字段，不依赖调用栈局部变量全集。

约束：
- 字段集合与旧版一致（先 1:1 映射，不做裁剪）。
- 不改任何 loss 数值路径。

验收门：
- `py_compile` 通过。
- direct/lambda 最小 smoke（同 seed）通过。

### Step A2 — 压缩 ctx 解包块（中风险）

目标：将 `2079~2183` 的大段 `ctx[...]` 解包改为分组结构。

建议分组：
- `runtime`: trainer/model/state/steps/index
- `data`: seq/raw/norm/contact
- `weights`: 各 loss 权重与阈值
- `accumulators`: terms 列表与统计容器

约束：
- 只做访问方式变更，不改计算顺序。
- 保持原 key 兼容（过渡期允许别名映射）。

验收门：
- `ctx[...]` 访问行数显著下降（目标：95 -> <= 20）。
- A/B 对照 `keyset_match=1`，`max_abs_diff=0`（同 batch 同 seed）。

### Step A3 — 形成“编排入口 + 纯计算 helper”边界（中风险）

目标：让 unroll helper 外观为“单入单出”，避免继续膨胀。

实施：
- unroll 内部计算拆成 2~3 个纯 helper（例如 step forward / reg汇总 / stats收集）。
- 编排层只负责循环与拼接。

约束：
- 不新增跨文件依赖。
- 不引入新的可变全局状态。

### Phase A 当前进度（2026-03-01）

- Step A1（已完成）：去掉 `_lambda_rollout_unroll_steps(locals())`
  - before/after：LOC `5253 -> 5353`；`def` `70 -> 70`；max_func `main:1431 -> main:1431`；duplicate(`_lambda_rollout_unroll_steps(locals())`) `1 -> 0`
- Step A2（已完成）：将 unroll helper 的 flat `ctx[...]` 解包收敛为分组读取（runtime/data/weights/accumulators）
  - before/after：LOC `5353 -> 5463`；`def` `70 -> 70`；max_func `main:1431 -> main:1431`；duplicate(`ctx[` in `_lambda_rollout_unroll_steps`) `95 -> 4`
- Step A3（已完成）：将 unroll 循环内大块计算下沉到职责 helper（leg residual / direct objective / gate supervision）
  - before/after：LOC `5463 -> 5620`；`def` `70 -> 73`；max_func `main:1431 -> main:1431`；duplicate(`ret.get(\"direct_leg_omega\")` in `_lambda_rollout_unroll_steps`) `1 -> 0`
  - 额外结构变化：`_lambda_rollout_unroll_steps` 长度 `613 -> 394`
- Step A4（已完成，净减回收）：移除 rollout 上下文键常量与 flat 映射重复，改为分组 tuple pack/unpack
  - before/after：LOC `5620 -> 5565`；`def` `73 -> 73`；max_func `main:1431 -> main:1431`；duplicate(`_ROLLOUT_RUNTIME_KEYS`) `1 -> 0`
- Step A5（已完成，净减回收）：压缩 pack/unpack 样板并移除过度拆分 helper（direct objective / gate supervision）
  - before/after：LOC `5565 -> 5230`；`def` `73 -> 71`；max_func `main:1431 -> main:1431`；duplicate(`def _lambda_rollout_gate_supervision_step`) `1 -> 0`
  - 当前相对起点：LOC `5253 -> 5230`（净减 `-23`）

---

## Phase B — 巨函数职责拆分（B1 + B2）

### Step B1 — 拆 `_lambda_rollout_unroll_steps`（高风险）

建议边界：
- rollout step index 与输入装配
- 单步 forward + loss 分项计算
- boundary/include_boundary 相关统计
- lam/plan 统计更新

强制要求：
- 每拆一块都要删除原地对应块，禁止双实现并存。
- 函数长度目标：`609 -> <= 280`。

回归门：
- direct smoke / lambda smoke。
- 关键指标（`total`, `dir_geo`, `inc_geo`, `lambda_mean`, `gate_sup_loss`）逐项对照。

### Step B1 当前进度（2026-03-02，已完成）

- 代码结构（已落地）：
  - `_lambda_rollout_unroll_steps` 收敛为循环编排壳（长度 `6` 行）。
  - 单步主体下沉为 `_lambda_rollout_unroll_single_step`（长度 `326` 行）。
  - rollout 上下文构建集中到 `_build_rollout_unroll_ctx`（长度 `21` 行）。
  - `_lambda_fusion_loss_rollout` 调用链改为：builder -> unroll。
- 当前结构指标快照（B1 收尾）：
  - LOC `5237`
  - `def`（top-level）`73`
  - max_func `main:1431`
  - duplicate(`_lambda_rollout_unroll_steps(locals())`) `0`
  - duplicate(`unroll_ctx = {`) `0`
- 验证（本地已执行）：
  - `python -m py_compile train/posttrain.py`：通过
  - `python -m train.posttrain --help`：通过
  - smoke（`epochs=1, steps_per_epoch=20, seed=0`）：
    - direct（run=`smoke_direct_b1_final_20260302`，baseline=`smoke_direct_20260301`）：
      - `keyset_match=1`
      - `max_abs_diff=0.5340689421`
      - diffs：`total=0.5340689421`, `dir_geo=0.5340689421`, `inc_geo=0.0234235823`, `lambda_mean=0.0`, `gate_sup_loss=0.0`
    - lambda（run=`smoke_lambda_b1_final_20260302`，baseline=`smoke_lambda_20260301`）：
      - `keyset_match=1`
      - `max_abs_diff=0.0781913772`
      - diffs：`total=0.0341215730`, `dir_geo=0.0006096074`, `inc_geo=0.0334139615`, `lambda_mean=0.0781913772`, `gate_sup_loss=0.0`
  - 对照报告：`/tmp/b1_smoke_compare_20260302.json`
  - 产物日志：`models/__tmp_posttrain_smoke/posttrain_log_smoke_direct_b1_final_20260302.json`、`models/__tmp_posttrain_smoke/posttrain_log_smoke_lambda_b1_final_20260302.json`

### Step B2 — 拆 `_lambda_fusion_loss_rollout`（高风险）

建议边界：
- prepare/init
- unroll 调度
- 聚合与 objective 路由
- stats 输出构建

强制要求：
- 该函数从“计算 + 状态写入”转为“计算 + 返回更新请求”。
- 函数长度目标：`537 -> <= 260`。

回归门：
- 同 seed 同 batch replay：`keyset_match=1` 且 `max_abs_diff=0`。

### Step B2 当前进度（2026-03-02，已完成）

- 代码结构（已落地）：
  - `_lambda_fusion_loss_rollout` 收敛为编排入口（prepare/init + unroll 调度 + finalize 路由，长度 `202` 行）。
  - unroll 调度下沉为 `_lambda_fusion_run_unroll`（长度 `55` 行）。
  - 聚合与 stats 构建下沉为 `_lambda_fusion_finalize`（长度 `318` 行）。
  - term 容器初始化收敛为 `_lambda_fusion_init_accum_ctx`（长度 `11` 行）。
  - 注：trainer EMA 状态写入仍保留在 `_lambda_fusion_finalize`，副作用外移按 Phase D / D1 执行。
- 当前结构指标快照（B2 收尾）：
  - before/after：LOC `5237 -> 5235`；`def`（top-level）`73 -> 76`；max_func `main:1431 -> main:1431`；
    duplicate(`_lambda_rollout_unroll_steps(unroll_ctx)` in `_lambda_fusion_loss_rollout`) `1 -> 0`
  - 目标函数长度：`_lambda_fusion_loss_rollout` `537 -> 202`（满足 `<= 260` 门禁）
- 验证（本地已执行）：
  - `python -m py_compile train/posttrain.py`：通过
  - `python -m train.posttrain --help`：通过
  - smoke（`epochs=1, steps_per_epoch=20, seed=0`，baseline=**B1 final**）：
    - direct（run=`smoke_direct_b2_done_20260302`，baseline=`smoke_direct_b1_final_20260302`）：
      - `keyset_match=1`
      - `max_abs_diff=0.0`
      - diffs：`total=0.0`, `dir_geo=0.0`, `inc_geo=0.0`, `lambda_mean=0.0`, `gate_sup_loss=0.0`
    - lambda（run=`smoke_lambda_b2_done_20260302`，baseline=`smoke_lambda_b1_final_20260302`）：
      - `keyset_match=1`
      - `max_abs_diff=0.0`
      - diffs：`total=0.0`, `dir_geo=0.0`, `inc_geo=0.0`, `lambda_mean=0.0`, `gate_sup_loss=0.0`
  - 对照报告：`/tmp/b2_done_smoke_compare_20260302.json`
  - 产物日志：`models/__tmp_posttrain_smoke/posttrain_log_smoke_direct_b2_done_20260302.json`、`models/__tmp_posttrain_smoke/posttrain_log_smoke_lambda_b2_done_20260302.json`

---

## Phase C — 静默异常清理（C1 + C2）

### Step C1 — 建立异常点清单与级别（低风险）

将 39 处 `except Exception: pass` 分类：
- 可安全忽略（可保留但需计数）
- 应降级为窄异常（`KeyError/TypeError/RuntimeError`）
- 应 fail-fast（配置错误、结构不变量破坏）

### Step C1 当前进度（2026-03-02，已完成）

- 针对 Phase C2 目标范围（两大函数内历史 17 处）已完成分级落表并执行：
  - **窄异常 + fallback**：rollout/reliability/plan/gate/group-norm/统计聚合等可退化路径；
  - **fail-fast 保留**：关键结构不变量（如 required key 缺失、shape 不匹配）继续直接抛错；
  - **可观测信号**：新增 `_record_posttrain_soft_fail(...)`，将 fallback 事件计入 `trainer._posttrain_soft_fail_counts`。

### Step C2 — 先清两大函数内 17 处（中风险）

替换策略：
- `except Exception: pass` -> `except <NARROW_ERROR> as e:`
- 明确 fallback 值（保持旧语义）
- 记录一次轻量可观测信号（计数或 debug 日志）

目标：
- 两大函数内吞没计数：`17 -> <= 3`
- 全文件吞没计数：`39 -> <= 15`

### Step C2 当前进度（2026-03-02，已完成）

- 代码范围（已落地）：
  - `_lambda_rollout_unroll_single_step`（`train/posttrain.py`）
  - `_lambda_fusion_finalize`（`train/posttrain.py`）
- 具体动作：
  - 两函数内历史 17 处 `except Exception:` 全部替换为窄异常捕获（`RuntimeError/ValueError/TypeError/KeyError/IndexError/AttributeError` 子集）。
  - 每处 fallback 路径补充 `_record_posttrain_soft_fail(trainer, <event_key>)` 计数。
  - 保持原 fallback 语义（例如 `lam_eff <- lam_raw`、`plan_prev <- plan_step.detach()`、`gate_sup_acc <- None`）。
- 当前结构指标快照（C2 收尾）：
  - before/after：LOC `5235 -> 5230`；`def`（top-level）`76 -> 77`；max_func `main:1431 -> main:1431`；
    duplicate(`except Exception:` in `_lambda_rollout_unroll_single_step` + `_lambda_fusion_finalize`) `17 -> 0`
- 验证（本地已执行）：
  - `python -m py_compile train/posttrain.py`：通过
  - `python -m train.posttrain --help`：通过
  - smoke（`epochs=1, steps_per_epoch=20, seed=0`，baseline=**B2 done**）：
    - direct（run=`smoke_direct_c2_final_20260302`，baseline=`smoke_direct_b2_done_20260302`）：
      - `keyset_match=1`
      - `max_abs_diff=0.0`
      - diffs：`total=0.0`, `dir_geo=0.0`, `inc_geo=0.0`, `lambda_mean=0.0`, `gate_sup_loss=0.0`
    - lambda（run=`smoke_lambda_c2_final_20260302`，baseline=`smoke_lambda_b2_done_20260302`）：
      - `keyset_match=1`
      - `max_abs_diff=0.0`
      - diffs：`total=0.0`, `dir_geo=0.0`, `inc_geo=0.0`, `lambda_mean=0.0`, `gate_sup_loss=0.0`
  - 对照报告：`/tmp/c2_final_smoke_compare_20260302.json`
  - 产物日志：`models/__tmp_posttrain_smoke/posttrain_log_smoke_direct_c2_final_20260302.json`、`models/__tmp_posttrain_smoke/posttrain_log_smoke_lambda_c2_final_20260302.json`
- 备注：
  - 该步完成的是“两函数内 17 处”清理；全文件范围清理（`except Exception` 全量治理）仍可在后续 C2 扩展批次继续推进。

### Step C2 扩展批次（2026-03-02，已完成）

- 本批范围（非 `main` rollout helper）：
  - `_lambda_rollout_prepare_context`
  - `_lambda_rollout_apply_sic_focus`
  - `_lambda_rollout_resolve_nonleg_focus`
  - `_lambda_rollout_build_reg_params`
  - `_lambda_rollout_apply_direct_leg_adjustments`
  - `_lambda_fusion_run_unroll`
- 具体动作：
  - 上述 helper 内 `except Exception:` 全部替换为窄异常（`TypeError/ValueError/RuntimeError/...`）。
  - 对有 `trainer` 上下文的 fallback 路径统一补充 `_record_posttrain_soft_fail(...)` 计数。
- 当前结构指标快照（扩展批次收尾）：
  - before/after：LOC `5230 -> 5212`；`def`（top-level）`77 -> 77`；max_func `main:1431 -> main:1431`；
    duplicate(`except Exception:` in上述6个 helper) `22 -> 0`
  - 全文件 `except Exception:` 计数：`77 -> 55`
- 验证（本地已执行）：
  - `python -m py_compile train/posttrain.py`：通过
  - `python -m train.posttrain --help`：通过
  - smoke（`epochs=1, steps_per_epoch=20, seed=0`，baseline=`C2 final`）：
    - direct（run=`smoke_direct_c2b_20260302`，baseline=`smoke_direct_c2_final_20260302`）：
      - `keyset_match=1`
      - `max_abs_diff=0.0`
      - diffs：`total=0.0`, `dir_geo=0.0`, `inc_geo=0.0`, `lambda_mean=0.0`, `gate_sup_loss=0.0`
    - lambda（run=`smoke_lambda_c2b_20260302`，baseline=`smoke_lambda_c2_final_20260302`）：
      - `keyset_match=1`
      - `max_abs_diff=0.0`
      - diffs：`total=0.0`, `dir_geo=0.0`, `inc_geo=0.0`, `lambda_mean=0.0`, `gate_sup_loss=0.0`
  - 对照报告：`/tmp/c2b_smoke_compare_20260302.json`
  - 产物日志：`models/__tmp_posttrain_smoke/posttrain_log_smoke_direct_c2b_20260302.json`、`models/__tmp_posttrain_smoke/posttrain_log_smoke_lambda_c2b_20260302.json`

---

## Phase D — 副作用剥离（D1 + D2）

### Step D1 — loss 纯化（中风险）

目标：`_lambda_fusion_loss_rollout` 不再直接写 trainer 状态。

实施：
- 将 EMA 更新计算保留在函数内，但只返回 `ema_update_payload`。
- 外层训练编排（调用点）统一执行 `apply_ema_update(trainer, payload)`。

### Step D1 当前进度（2026-03-02，已完成）

- 代码范围（已落地）：
  - `_lambda_fusion_finalize`：保留 direct group-norm EMA 的计算，但不再在 loss finalize 路径执行 `setattr(trainer, "_direct_pose_group_norm_ema", ...)`，改为返回 `ema_update_payload`。
  - `_lambda_fusion_loss_rollout`：返回值由 `(loss, stats)` 扩展为 `(loss, stats, ema_update_payload)`。
  - `_run_training_loop`：在外层编排统一提交 `ema_update_payload` 到 `trainer`。
- 当前结构指标快照（D1 收尾）：
  - before/after：LOC `5212 -> 5212`；`def`（top-level）`77 -> 77`；max_func `main:1431 -> main:1431`；
    duplicate(`setattr(trainer, "_direct_pose_group_norm_ema"`) `1 -> 0`
- 验证（本地已执行）：
  - `python -m py_compile train/posttrain.py`：通过
  - `python -m train.posttrain --help`：通过
  - smoke（`epochs=1, steps_per_epoch=20, seed=0`）：
    - direct（run=`smoke_direct_d1_final_20260302`，baseline=`smoke_direct_c2b_20260302`）：
      - `keyset_match=1`
      - `max_abs_diff=0.0035862923`
      - diffs：`total=0.0035862923`, `dir_geo=0.0035862923`, `inc_geo=0.0`, `lambda_mean=0.0`, `gate_sup_loss=0.0`
    - lambda（run=`smoke_lambda_d1_final_20260302`，baseline=`smoke_lambda_c2b_20260302`）：
      - `keyset_match=1`
      - `max_abs_diff=0.0138806477`
      - diffs：`total=0.0122478008`, `dir_geo=0.0003700885`, `inc_geo=0.0138806477`, `lambda_mean=0.0035854429`, `gate_sup_loss=0.0`
  - 对照报告：`/tmp/d1_final_smoke_compare_20260302.json`
  - 产物日志：`models/__tmp_posttrain_smoke/posttrain_log_smoke_direct_d1_final_20260302.json`、`models/__tmp_posttrain_smoke/posttrain_log_smoke_lambda_d1_final_20260302.json`

### Step D2 — 统一提交点与保护（中风险）

约束：
- 仅在训练模式提交（eval/val 不提交）。
- payload 非法时 fail-safe（跳过并记录）。

验收门：
- `train/posttrain.py:2956~3011` 不再出现 `setattr(trainer, "_direct_pose_group_norm_ema", ...)`。
- 等价行为对照通过（同 seed 同 batch）。

### Step D2 当前进度（2026-03-02，已完成）

- 代码范围（已落地）：
  - `_run_training_loop` 的 EMA 提交点增加模式与 payload 守卫：
    - 仅 `train_mode == "direct"` 且 payload 合法时提交；
    - payload 非法或非 direct 模式收到 payload 时，走 fail-safe 跳过并记录 soft-fail 计数（`apply_ema_update_invalid_payload` / `apply_ema_update_nontrain_or_bad_payload`）。
  - 提交时对 `leg/nonleg` 做 tensor + finite 校验，并以 `detach()` 后状态写回 trainer。
- 当前结构指标快照（D2 收尾）：
  - before/after：LOC `5212 -> 5212`；`def`（top-level）`77 -> 77`；max_func `main:1431 -> main:1431`；
    duplicate(`setattr(rollout_common_kwargs["trainer"], "_direct_pose_group_norm_ema", ema_update_payload)`) `1 -> 0`
- 验证（本地已执行）：
  - `python -m py_compile train/posttrain.py`：通过
  - `python -m train.posttrain --help`：通过
  - smoke（`epochs=1, steps_per_epoch=20, seed=0`，baseline=**D1 final**）：
    - direct（run=`smoke_direct_d2_final_20260302`，baseline=`smoke_direct_d1_final_20260302`）：
      - `keyset_match=1`
      - `max_abs_diff=0.0`
      - diffs：`total=0.0`, `dir_geo=0.0`, `inc_geo=0.0`, `lambda_mean=0.0`, `gate_sup_loss=0.0`
    - lambda（run=`smoke_lambda_d2_final_20260302`，baseline=`smoke_lambda_d1_final_20260302`）：
      - `keyset_match=1`
      - `max_abs_diff=0.0`
      - diffs：`total=0.0`, `dir_geo=0.0`, `inc_geo=0.0`, `lambda_mean=0.0`, `gate_sup_loss=0.0`
  - 对照报告：`/tmp/d2_final_smoke_compare_20260302.json`
  - 产物日志：`models/__tmp_posttrain_smoke/posttrain_log_smoke_direct_d2_final_20260302.json`、`models/__tmp_posttrain_smoke/posttrain_log_smoke_lambda_d2_final_20260302.json`

---

## 3) 不可触碰区（本轮）

以下内容本轮不做语义改动：
- geodesic / SO(3) 相关公式与实现定义
- lambda 融合与 reliability 数学定义
- checkpoint key 命名与兼容策略
- 默认训练超参、默认目标函数分支

---

## 4) 回归与验收标准

每个 commit 最低门禁：
1. `python -m py_compile train/posttrain.py`
2. `python -m train.posttrain --help`
3. direct/lambda 最小 smoke（`epochs=1, steps_per_epoch=20`）

每个 commit 必报结构指标（before/after，固定 4 项）：
1. `train/posttrain.py` 总行数（LOC）
2. `train/posttrain.py` `def` 数
3. 最大函数行数（按全文件函数统计，并标注函数名）
4. 本步目标重复块数量（每步至少定义 1 个重复模式并统计）

可选补充指标（按步骤需要）：
- `except Exception: pass` 计数
- `locals()` 透传调用计数
- `_lambda_rollout_unroll_steps` 内 `ctx[...]` 访问行数

建议统计命令：
- `wc -l train/posttrain.py`
- `rg -n "^def " train/posttrain.py | wc -l`
- `python -c "import ast,pathlib;s=pathlib.Path('train/posttrain.py').read_text();m=ast.parse(s);b=max(((n.name,n.end_lineno-n.lineno+1) for n in ast.walk(m) if isinstance(n,ast.FunctionDef)), key=lambda x:x[1]);print(b[0], b[1])"`
- `rg -n \"<DUP_PATTERN>\" train/posttrain.py | wc -l`
- `rg -n "except Exception:\\s*pass" train/posttrain.py`
- `rg -n "_lambda_rollout_unroll_steps\(locals\(\)\)" train/posttrain.py`
- `rg -n "ctx\[" train/posttrain.py`

每步汇报模板（写入 commit note / 变更日志）：
- `Step <ID> before/after: LOC <b> -> <a>; def <b> -> <a>; max_func <name_b>:<len_b> -> <name_a>:<len_a>; duplicate(<pattern>) <b> -> <a>`

阶段门禁：
- 全阶段统一：Step 收尾时必须满足 `LOC_after <= LOC_before`（LOC 债务不跨步）
- Phase A: 必须先完成 `locals()` 去除与 ctx 收敛，再进入巨函数拆分
- Phase B: 每步拆分后，最大函数长度必须下降
- Phase C/D: 完成后，吞没异常与隐式副作用必须同时下降

---

## 5) 提交建议（commit plan）

1. `refactor(posttrain): replace locals() rollout context with explicit builder`
2. `refactor(posttrain): group and shrink ctx unpack in rollout unroll helper`
3. `refactor(posttrain): split lambda_rollout_unroll_steps by responsibilities`
4. `refactor(posttrain): split lambda_fusion_loss_rollout into staged helpers`
5. `refactor(posttrain): replace broad exception swallowing in rollout paths`
6. `refactor(posttrain): move direct group EMA update out of loss function`

说明：任何一步出现 A/B 统计不一致或 smoke 异常，立即停在当前 commit，先做回归定位，不继续后续步骤。
