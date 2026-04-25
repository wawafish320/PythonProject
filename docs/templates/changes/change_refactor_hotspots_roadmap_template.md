# [YYYY-MM-DD] `[目标文件/模块]` 热点重构路线图（v1）

Date: [YYYY-MM-DD]  
Status: Draft / Active v1（[一句话说明当前版本关注点与约束变化]）  
Scope: `[文件/模块范围]`（[本轮允许与禁止的改动边界]）  
Goal: 在**不改变语义/行为**前提下，优先降低“重复样板 + 巨函数 + 上下文耦合 + 静默异常 + 隐式副作用”维护风险。  
Non-goal: 不改核心算法/数学定义、不改默认超参行为、不引入新架构。

---

## 0) 当前策略（先去重/收边界，再拆分，再收紧异常，最后剥离副作用）

统一执行顺序：

1. **Phase A: 重复收敛与边界解耦（低/中风险）**
   - 先移除重复块、重复映射、重复样板。
   - 去掉 `locals()` / `globals()` / 黑盒 `ctx` 一类隐式输入。
   - 先收敛边界，再进入巨函数拆分。
2. **Phase B: 巨函数按职责拆分（中/高风险）**
   - 一次只拆一个职责块。
   - 先拆编排壳，再拆纯计算块，再拆聚合/提交块。
3. **Phase C: 静默异常清理（中风险）**
   - 先建立异常点清单与级别。
   - 用“窄异常 + 明确 fallback + 可观测计数/日志”替代吞没。
4. **Phase D: 隐式状态副作用外移（中风险）**
   - 将计算路径中的状态写入改为“返回更新请求 + 外层统一提交”。

核心原则：
- one step, one commit
- 每步必须有 before/after 结构指标
- 任何一步回归失败，立即停在当前 commit，不继续后续步骤
- 每步固定汇报 4 项：总行数（LOC）、`def`/函数数、最大函数行数、目标重复块数量
- 每步还必须汇报至少 **1 项本轮主题相关的结构债指标**，用于证明不是“为了拆分而拆分”
- **单步硬门禁**：以 **Step 收尾** 为准，必须满足 `LOC_after <= LOC_before`；开发中可临时增行，但不可带着 LOC 债务进入下一步

新增约束（本路线图强制）：
- 不允许新增新的黑盒上下文传递（例如 `locals()` / `globals()` / 无边界字典透传）。
- 不允许“只抽函数不删旧逻辑”。
- 每次新增 helper 必须带来可量化净收益（LOC、最大函数长度、重复块计数、异常吞没计数、上下文耦合计数等至少一项下降）。
- 拆分后关键统计 key 集合保持一致（A/B 对照）。

### 拆分约束（强制执行）

拆分前准入（满足任一即可进入候选）：
- 被调用 ≥2 次（或下一步已明确会复用 ≥2 次）
- 封装了可独立命名的领域概念
- 拆出后调用处更易读（调用点行数或嵌套层级下降）

硬禁止（命中任一则不允许拆）：
- 纯转发 wrapper / 无实质边界的间接层
- 函数名仅复述代码字面行为（无领域语义）
- 再次引入 `locals()` / `globals()` / 黑盒上下文
- 只抽函数、不删除原地旧逻辑（双实现并存）
- **Step 收尾时** `LOC_after > LOC_before`
- helper 参数 > 8 且未收敛到结构化 context（TypedDict / dataclass / namedtuple / 等价结构）

单次调用 helper 的例外规则（替代绝对禁止）：
- 调用次数 = 1 允许，但必须同时满足：
  - 具备独立概念命名
  - 调用点可读性提升
  - 至少 1 项结构指标下降（最大函数长度 / 重复块 / 异常吞没 / 上下文耦合）

拆分后验流程（强制）：
1. 先按“复用价值”拆分
2. Step 收尾前检查 `LOC_after <= LOC_before`
3. 若不满足：定位并回收不必要间接层（参数搬运层、单次无概念 helper、纯 wrapper）
4. 在**当前 step**完成净减回收后，才允许进入下一步

---

## 1) 基线现状（针对本轮热点问题）

当前代码快照核对（`[目标文件/模块]`）：

- **热点问题 1**：[`[path:line]` + 简述现状]
- **热点问题 2**：[`[path:line]` + 简述现状]
- **热点问题 3**：[`[path:line]` + 简述现状]
- **热点问题 4**：[`[path:line]` + 简述现状]

结构指标基线（本路线图起点）：
- LOC: `[数字]`
- `def` / 函数数: `[数字]`
- 最大函数长度: ``[函数名]`` = `[数字]`
- 目标重复块计数: ``[pattern]`` = `[数字]`
- 本轮主题结构债指标 #1: ``[pattern / metric name]`` = `[数字]`
- 本轮主题结构债指标 #2: ``[pattern / metric name]`` = `[数字，可选是否启用由本轮定义，但一旦选定必须持续跟踪]`

---

## 2) 具体改动流程

## Phase A — 重复收敛与边界解耦（A1 + A2 + A3）

### Step A1 — 去掉黑盒上下文透传（低风险）

目标：将 `[path:line]` 的黑盒上下文改为显式命名输入。

实施：
- 新增 `[builder / context factory / typed container]` 集中组织输入字段。
- 调用点改为 `[helper(ctx)]`，`ctx` 为显式命名字段，不依赖调用栈局部变量全集。

约束：
- 字段集合与旧版一致（先 1:1 映射，不做语义裁剪）。
- 不改任何核心计算路径。

验收门：
- `[语法 / import / build]` 通过。
- `[最小 smoke / replay]` 通过。
- 黑盒上下文计数下降到目标值。

### Step A2 — 收敛重复块 / 参数解包 / 映射样板（中风险）

目标：将 `[重复块类型]` 从多处分散维护，收敛到单一入口。

建议对象：
- 重复映射 / alias map
- 重复 key 列表 / special keys
- 重复 clamp / normalize / fallback 片段
- 重复 payload / config / override 样板

约束：
- 只做边界与访问方式收敛，不改计算顺序。
- 删除原地重复实现，禁止新旧双轨并存。

验收门：
- 目标重复块计数显著下降。
- 关键 key 集合 / 输出字段集合保持一致。

### Step A3 — 形成“编排入口 + 纯计算 helper”边界（中风险）

目标：让热点函数外观收敛为“单入单出”，避免继续膨胀。

实施：
- 将循环体 / 计算块 / 聚合块下沉到职责 helper。
- 编排层只负责调度、拼接、路由。

约束：
- 不新增跨文件依赖。
- 不引入新的可变全局状态。

验收门：
- 热点函数长度下降。
- 调用点嵌套层级下降。
- 至少 1 项主题结构债指标下降。

### Phase A 当前进度（[YYYY-MM-DD]）

- Step A1（[状态]）：[一句话总结]
  - before/after：LOC `[b] -> [a]`；`def` `[b] -> [a]`；max_func ``[name_b]:[len_b] -> [name_a]:[len_a]``；duplicate(``[pattern]``) `[b] -> [a]`；structural(``[metric]``) `[b] -> [a]`
- Step A2（[状态]）：[一句话总结]
  - before/after：LOC `[b] -> [a]`；`def` `[b] -> [a]`；max_func ``[name_b]:[len_b] -> [name_a]:[len_a]``；duplicate(``[pattern]``) `[b] -> [a]`；structural(``[metric]``) `[b] -> [a]`
- Step A3（[状态]）：[一句话总结]
  - before/after：LOC `[b] -> [a]`；`def` `[b] -> [a]`；max_func ``[name_b]:[len_b] -> [name_a]:[len_a]``；duplicate(``[pattern]``) `[b] -> [a]`；structural(``[metric]``) `[b] -> [a]`

---

## Phase B — 巨函数职责拆分（B1 + B2）

### Step B1 — 拆 `[hotspot_func_a]`（高风险）

建议边界：
- 输入装配 / 索引调度
- 单步 forward / compute 主体
- boundary / include_boundary / mask 统计
- 聚合前的中间结果收集

强制要求：
- 每拆一块都要删除原地对应块，禁止双实现并存。
- 函数长度目标：``[原长度] -> <= [目标长度]``。

回归门：
- `[smoke / replay / A/B compare]`
- 关键指标（`[metric_a]`, `[metric_b]`, `[metric_c]`）逐项对照。

### Step B1 当前进度（[YYYY-MM-DD]，[状态]）

- 代码结构（已落地）：
  - `[原函数]` 收敛为 `[编排壳 / 入口函数]`（长度 `[数字]` 行）
  - 单步主体下沉为 `[helper_name]`（长度 `[数字]` 行）
  - 上下文构建集中到 `[builder_name]`（长度 `[数字]` 行）
- 当前结构指标快照（B1 收尾）：
  - LOC `[数字]`
  - `def` / 函数数 `[数字]`
  - max_func ``[name]:[len]``
  - duplicate(``[pattern]``) `[数字]`
  - structural(``[metric]``) `[数字]`
- 验证（本地已执行）：
  - `[command]`：通过
  - `[command]`：通过
  - `[smoke / replay artifact path]`

### Step B2 — 拆 `[hotspot_func_b]`（高风险）

建议边界：
- prepare / init
- 调度 / unroll / dispatch
- 聚合与 objective 路由
- stats / output / commit request 构建

强制要求：
- 该函数从“计算 + 状态写入”转为“计算 + 返回更新请求”。
- 函数长度目标：``[原长度] -> <= [目标长度]``。

回归门：
- 同 seed / 同 batch replay：`keyset_match=1` 且 `[max_abs_diff 目标]`。

### Step B2 当前进度（[YYYY-MM-DD]，[状态]）

- 代码结构（已落地）：
  - `[原函数]` 收敛为编排入口（长度 `[数字]` 行）
  - `[调度 helper]`（长度 `[数字]` 行）
  - `[finalize / aggregate helper]`（长度 `[数字]` 行）
  - `[init ctx helper]`（长度 `[数字]` 行）
- 当前结构指标快照（B2 收尾）：
  - before/after：LOC `[b] -> [a]`；`def` `[b] -> [a]`；max_func ``[name_b]:[len_b] -> [name_a]:[len_a]``；duplicate(``[pattern]``) `[b] -> [a]`；structural(``[metric]``) `[b] -> [a]`
- 验证（本地已执行）：
  - `[command]`：通过
  - `[command]`：通过
  - `[compare report path / log path]`

---

## Phase C — 静默异常清理（C1 + C2）

### Step C1 — 建立异常点清单与级别（低风险）

目标：将广义吞没异常按风险等级分层。

分类建议：
- 可安全忽略（可保留但需计数）
- 应降级为窄异常（如 `KeyError` / `TypeError` / `RuntimeError`）
- 应 fail-fast（配置错误、结构不变量破坏）

验收门：
- 形成清单：位置、异常类型、fallback、可观测手段。

### Step C1 当前进度（[YYYY-MM-DD]，[状态]）

- before/after：LOC `[b] -> [a]`；`def` `[b] -> [a]`；max_func ``[name_b]:[len_b] -> [name_a]:[len_a]``；duplicate(``[pattern]``) `[b] -> [a]`；structural(``except Exception: pass``) `[b] -> [a]`

### Step C2 — 清理热点路径中的广义异常吞没（中风险）

目标：优先处理本轮核心热点函数中的 `except Exception: pass` / 宽泛 fallback。

约束：
- 先改热点路径，再改外围路径。
- 每处替换必须明确 fallback 语义。

验收门：
- 广义吞没异常计数下降。
- 回归路径保持一致。

### Step C2 当前进度（[YYYY-MM-DD]，[状态]）

- before/after：LOC `[b] -> [a]`；`def` `[b] -> [a]`；max_func ``[name_b]:[len_b] -> [name_a]:[len_a]``；duplicate(``[pattern]``) `[b] -> [a]`；structural(``except Exception: pass``) `[b] -> [a]`

---

## Phase D — 副作用剥离（D1 + D2）

### Step D1 — 纯化计算路径（中风险）

目标：将计算函数中的状态更新 / 外部写入剥离到外层。

实施：
- 计算函数仅返回结果与更新请求。
- 外层统一执行 commit / apply / update。

验收门：
- 隐式副作用计数下降。
- 调用链更清晰，A/B 输出一致。

### Step D1 当前进度（[YYYY-MM-DD]，[状态]）

- before/after：LOC `[b] -> [a]`；`def` `[b] -> [a]`；max_func ``[name_b]:[len_b] -> [name_a]:[len_a]``；duplicate(``[pattern]``) `[b] -> [a]`；structural(``[副作用 pattern]``) `[b] -> [a]`

### Step D2 — 统一提交点与保护（中风险）

目标：将所有状态变更集中到明确的提交点，并为提交路径加保护。

验收门：
- 提交点数量下降。
- 状态写入路径可追踪。

### Step D2 当前进度（[YYYY-MM-DD]，[状态]）

- before/after：LOC `[b] -> [a]`；`def` `[b] -> [a]`；max_func ``[name_b]:[len_b] -> [name_a]:[len_a]``；duplicate(``[pattern]``) `[b] -> [a]`；structural(``[提交点 pattern]``) `[b] -> [a]`

---

## 3) 不可触碰区（本轮）

- 不改 `[核心算法 / 数学定义 / loss 公式]`
- 不改 `[默认超参 / 配置含义 / schema 对外语义]`
- 不改 `[数据格式 / checkpoint 契约 / 日志字段契约]`
- 不跨文件迁移 `[若本轮禁止]`
- 不引入 `[新依赖 / 新脚本 / 新运行模式]`

---

## 4) 回归与验收标准

每个 commit 最低门禁：
1. `[语法 / import / build check command]`
2. `[CLI/help / module import / config parse check command]`
3. `[最小 smoke / replay / tiny train command]`

每个 commit 必报结构指标（before/after，固定 4 项）：
1. `[目标文件/模块]` 总行数（LOC）
2. `[目标文件/模块]` `def` / 函数数
3. 最大函数行数（按全文件函数统计，并标注函数名）
4. 本步目标重复块数量（每步至少定义 1 个重复模式并统计）

每个 commit 还必须报 **主题结构债指标**（至少 1 项，推荐 2 项）：
- 广义异常吞没计数（例如 `except Exception: pass`）
- 黑盒上下文透传计数（例如 `locals()` / `globals()` / 无边界 `ctx`）
- 热点函数内上下文访问行数（例如 `ctx[...]`）
- 热点路径副作用写入点计数
- 热点样板计数（例如 alias map / special keys / fallback 片段）

建议统计命令：
- `wc -l [target_path]`
- `rg -n "^def " [target_path] | wc -l`
- `python -c "import ast,pathlib;s=pathlib.Path('[target_path]').read_text();m=ast.parse(s);b=max(((n.name,n.end_lineno-n.lineno+1) for n in ast.walk(m) if isinstance(n,ast.FunctionDef)), key=lambda x:x[1]);print(b[0], b[1])"`
- `rg -n "[DUP_PATTERN]" [target_path] | wc -l`
- `rg -n "[STRUCTURAL_PATTERN]" [target_path] | wc -l`

每步汇报模板（写入 commit note / 变更日志）：
- `Step <ID> before/after: LOC <b> -> <a>; def <b> -> <a>; max_func <name_b>:<len_b> -> <name_a>:<len_a>; duplicate(<pattern>) <b> -> <a>; structural(<metric>) <b> -> <a>`

阶段门禁：
- 全阶段统一：Step 收尾时必须满足 `LOC_after <= LOC_before`（LOC 债务不跨步）
- Phase A：必须先完成重复收敛 / 黑盒上下文去除 / 边界收口，再进入巨函数拆分
- Phase B：每步拆分后，最大函数长度必须下降
- Phase C：完成后，广义异常吞没必须下降
- Phase D：完成后，隐式副作用写入点必须下降

---

## 5) 提交建议（commit plan）

1. `refactor([module]): remove duplicated maps and boilerplate before split`
2. `refactor([module]): replace black-box context with explicit builder`
3. `refactor([module]): split [hotspot_func_a] by responsibilities`
4. `refactor([module]): split [hotspot_func_b] into staged helpers`
5. `refactor([module]): replace broad exception swallowing in hotspot paths`
6. `refactor([module]): move side effects out of compute path`

说明：任何一步出现 A/B 统计不一致、结构指标不降、或 smoke 异常，立即停在当前 commit，先做回归定位，不继续后续步骤。
