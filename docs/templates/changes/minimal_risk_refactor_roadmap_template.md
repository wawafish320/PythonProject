# [YYYY-MM-DD] `<scope>` 最小风险重构路线图模版（v0.1）

Date: YYYY-MM-DD  
Status: Draft  
Scope: `<file/function/module>`  
Owner: `<optional>`  

Goal: 在**不改变业务/训练语义**的前提下，降低 `<scope>` 的维护风险：隐式契约、巨函数、隐藏副作用、异常吞没、测试缺口。  
Non-goal:
- 不改算法/数学定义。
- 不改默认超参、CLI 行为或配置兼容性。
- 不改 checkpoint save-load contract。
- 不做跨文件迁移，除非本路线图明确批准。
- 不为了“函数变短”而拆分。

---

## 0) 当前判断

### 为什么需要重构

当前痛点不是“代码看起来长”，而是：

- `<pain_1>`：例如隐式 dict contract / string key 耦合。
- `<pain_2>`：例如函数内读写外层 mutable state。
- `<pain_3>`：例如 nested closure 捕获过多外层变量。
- `<pain_4>`：例如 hot-path 缺少数值 smoke。

本轮重构的判断标准：

> 只有当改动能降低 contract coupling、缩小副作用边界、提升可验证性，或降低 hot-path 修改风险时，才允许拆分。

---

## 1) 基线快照

执行前记录结构指标，后续每一步都更新。

### 结构指标

- Target file LOC: `<N>`
- Top-level `def` count: `<N>`
- 最大函数:
  - `<function_name>`: `<N>` lines
- 目标函数长度:
  - `<function_a>`: `<N>` lines
  - `<function_b>`: `<N>` lines
- 隐式 contract 计数:
  - `ctx[...]`: `<N>`
  - `accum[...]`: `<N>`
  - `finalize_ctx[...]`: `<N>`
  - `locals()` / `globals()` pass-through: `<N>`
- 异常吞没:
  - `except Exception: pass`: `<N>`
- Nested closure:
  - `<nested_fn>`: `<N>` lines, captures `<rough_count>` outer vars

### 扫描命令

```bash
python3 -m py_compile <target_file>

rg 'locals\(|globals\(|ctx\[|accum\[|finalize_ctx\[|except Exception: pass' <target_file>

python3 - <<'PY'
from pathlib import Path

p = Path("<target_file>")
lines = p.read_text().splitlines()

for i, line in enumerate(lines, start=1):
    if line.startswith("def "):
        name = line.split("(", 1)[0].replace("def ", "")
        end = len(lines)
        for j in range(i, len(lines)):
            if j + 1 > i and lines[j].startswith(("def ", "class ")):
                end = j
                break
        print(f"{name}: line={i}, len={end - i}")
PY
```

---

## 2) 覆盖门禁

重构前必须确认测试/烟测覆盖。

### 必须有的验证层

| 覆盖层 | 是否已有 | 命令 | 备注 |
|---|---:|---|---|
| import/build smoke | `<yes/no>` | `<cmd>` | 检查循环 import / parser / build shell |
| compile smoke | `<yes/no>` | `python3 -m py_compile <file>` | 只保证语法 |
| focused unit test | `<yes/no>` | `<cmd>` | pin contract shape/keyset |
| numerical smoke | `<yes/no>` | `<cmd>` | pin loss/stats 数值 |
| real entry smoke | `<yes/no>` | `<cmd>` | 依赖本地数据/ckpt 可选 |

### Stop-rule

- 如果 hot-path 没有 numerical smoke：**先补 smoke，再动 hot-path**。
- 如果 real entry smoke 因本地 artifact 缺失无法跑，必须至少保留：
  - compile smoke
  - import/build smoke
  - focused contract/unit test
  - hermetic numerical smoke
- 如果 smoke 失败且原因不是本地 artifact 缺失：先修 smoke 或回滚，不继续重构。

---

## 3) 拆分准入规则

任何 helper 拆分前，必须满足至少一条：

- 被调用 ≥2 次，或下一步明确会复用。
- 封装了独立领域概念，例如 `finalize_group_norm`、`build_runtime_overlay`。
- 调用点可读性明显提升，嵌套层级下降。
- 将隐式输入变成显式参数/context。
- 将副作用从计算路径中隔离出来。
- 为 smoke/unit test 提供稳定 seam。

### 禁止拆分

命中任一条则不允许拆：

- 纯转发 wrapper。
- helper 名字只是复述代码字面行为，没有领域语义。
- 参数列表 > 8 且没有收敛成合理 typed context。
- 只抽函数但原地旧逻辑不删除。
- 新增 `locals()` / `globals()` / 黑盒 dict contract。
- 为了降低单个函数 LOC 而牺牲整体可读性。
- 没有对应验证手段。

### 单次调用 helper 例外

调用次数 = 1 可以接受，但必须同时满足：

- 名字表达独立概念。
- 调用点更像“流程编排”。
- 至少一项指标下降：
  - 最大函数长度
  - 隐式 contract 计数
  - nested closure 捕获范围
  - 异常吞没点
  - 副作用位置

---

## 4) 执行顺序

原则：**先低风险 setup，后 hot-path；先显式 contract，后拆函数；先 smoke，后大改。**

### Phase 0 — Stabilization / Coverage

目标：确认当前 snapshot 可验证。

改动：
- 不改业务代码，最多补最小 smoke/test。
- pin 住关键 stats key / payload shape / 数值 smoke。

验证：

```bash
python3 -m py_compile <target_file>
python3 -m unittest <focused_test>
<optional smoke command>
```

Stop-rule:
- 缺少 hot-path numerical smoke 时，不进入 hot-path refactor。
- smoke 失败且原因不是本地 artifact 缺失时，先修 smoke 或回滚。

---

### Phase 1 — Low-risk Setup Path Refactor

目标：处理非 hot-loop、顺序执行、阶段边界清晰的大函数。

候选：
- `<setup_function>`: `<N>` lines

允许拆分边界：
- config / norm spec resolve
- model build
- runtime attach
- loss wiring
- trainer construction
- final validation

禁止：
- 新增跨阶段大 context 容器。
- 为了少传参而引入黑盒 dict。
- 改训练语义或默认配置解析。

Stop-rule:
- 如果需要新建“大而全 dataclass”才能传参，说明 seam 切错，停手。
- 如果 helper 只是把 5 行代码搬出去，回收。

验证：

```bash
python3 -m py_compile <target_file>
python3 -c "from <module> import <builder>; print('import_ok')"
<runtime_overlay_smoke_if_available>
```

---

### Phase 2 — Flatten Nested Closures

目标：把大 nested closure 的隐式 outer-scope 依赖显式化。

候选：
- `<nested_fn_a>`: `<N>` lines
- `<nested_fn_b>`: `<N>` lines

流程：
1. 先列出 closure 读取的 outer variables。
2. 判断是否能减少参数到 ≤8。
3. 如果参数 >8：
   - 先拆 nested function 内部职责
   - 或保留 nested，不硬提
4. 提升后补最小测试或复用现有 smoke。

Stop-rule:
- 参数列表 >8 且没有清晰 context：不提升。
- 提升后调用点比原来更难读：回滚。
- 只是“从 nested 变 top-level”但没有降低 coupling：不做。

---

### Phase 3 — Hot-path Step Function Refactor

目标：只在 smoke 可靠后，拆 per-step hot-path。

候选：
- `<hot_step_fn>`: `<N>` lines

推荐阶段边界：
- decode / forward
- geometry compute
- objective accumulate
- regularization accumulate
- carry state advance

强制要求：
- 每次只拆一个阶段。
- 拆完立即跑 hot-path smoke。
- stats key 不变。
- aux payload 结构不变。
- loss 数学不变。

Stop-rule:
- 任意数值差异超出预期：回滚该 slice。
- 出现 NaN/grad norm 变化：回滚该 slice。
- helper 参数爆炸：先 typed context，再继续。

验证：

```bash
python3 -m py_compile <target_file>
python3 -m unittest <focused_hotpath_test>
<numerical_smoke>
```

---

### Phase 4 — Remaining Large Helper Refactor

目标：在 Phase 3 稳定后，处理同类 150+ 行函数。

候选：
- `<function_a>`: `<N>` lines
- `<function_b>`: `<N>` lines

规则：
- 一次只处理一个函数。
- 不跨文件。
- 不顺手改算法。
- 每个函数必须有单独 before/after 指标。

---

## 5) Per-step 记录模版

每一步完成后追加一段。

### Step `<ID>` — `<short title>`

Status: `<Done / Blocked / Reverted>`  
Risk: `<Low / Medium / High>`  
Scope:
- `<file>:<line>`
- `<function_name>`

Intent:
- `<why this step exists>`

Changes:
- `<change_1>`
- `<change_2>`

Explicit non-changes:
- loss math unchanged
- stats keys unchanged
- aux payload unchanged
- CLI/checkpoint/training loop unchanged

Before/after:
- LOC: `<before> -> <after>`
- top-level def count: `<before> -> <after>`
- max target function length: `<before> -> <after>`
- implicit contract count: `<before> -> <after>`
- nested closure count/capture: `<before> -> <after>`

Validation:

```bash
<cmd_1>
<cmd_2>
<cmd_3>
```

Result:
- `<pass/fail>`
- Known limitation: `<e.g. real smoke blocked by missing ckpt>`

Rollback rule:
- Revert this step if `<specific condition>`.

---

## 6) Final Acceptance Criteria

本轮结束必须满足：

- 所有 touched functions 有明确责任边界。
- 没有新增黑盒 dict / `locals()` / `globals()` contract。
- hot-path stats keyset 不变。
- aux payload structure 不变。
- compile + focused tests 通过。
- numerical smoke 通过，或明确记录 blocked artifact。
- 没有“只为缩短函数”的 helper。
- 没有跨文件迁移，除非 roadmap 明确批准。

---

## 7) Final Report 模版

最终汇报必须包含：

- 新增/修改的 typed context 或 helper。
- 修改的关键函数。
- 明确说明业务/训练语义是否保持。
- 明确说明 stats / aux payload / CLI / checkpoint 是否保持。
- 残留 contract 清单。
- 验证结果。
- blocked smoke 原因，如果有。

---

## 8) 使用备注

这个文件不是“拆分任务清单”，而是“允许拆分的审查协议”。  
执行每一步前先问：

> 这个 helper 是否降低了某种风险？

如果答案只是“函数短一点”，则不拆。
