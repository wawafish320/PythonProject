# [YYYY-MM-DD] `[topic]` change report

Date: [YYYY-MM-DD]  
Status: Draft / Active / Done / Blocked  
Scope: `[files/modules]`  
Owner: `[optional]`  

Goal: [一句话说明这次 change 要解决什么]  
Non-goal:
- [不改算法/数学定义/训练语义等]
- [不改 checkpoint / CLI / config contract，除非本报告明确说明]

---

## 0) 一页版结论

- **结论**: [Done / Partial / Blocked]
- **核心变化**: [1-3 条]
- **行为影响**: [语义保持 / 有意变化，说明影响面]
- **验证状态**: [已跑命令 + 是否通过；blocked 要写原因]
- **后续动作**: [none / follow-up doc path]

---

## 1) 背景与决策

为什么需要这次改动：

- [背景 1]
- [背景 2]

关键决策：

| 决策 | 原因 | 替代方案 | 取舍 |
|---|---|---|---|
| [decision] | [why] | [alternative] | [tradeoff] |

---

## 2) 改动摘要

| 区域 | 文件/位置 | 改动 | 风险 |
|---|---|---|---|
| [area] | `[path:line]` | [change] | [low/medium/high] |
| [area] | `[path:line]` | [change] | [low/medium/high] |

明确未改：

- [loss math / stats keys / payload shape / CLI / checkpoint contract unchanged]
- [其他保持不变的关键契约]

---

## 3) 证据与产物

| 类型 | 路径/命令 | 说明 |
|---|---|---|
| baseline | `[path]` | [before 证据] |
| output | `[path]` | [after 产物] |
| log | `[path]` | [运行日志] |

关键命令：

```bash
[cmd_1]
[cmd_2]
```

---

## 4) Before / After

| 指标 | Before | After | 说明 |
|---|---:|---:|---|
| LOC | `[N]` | `[N]` | [scope] |
| max function length | `[N]` | `[N]` | [function] |
| duplicate block count | `[N]` | `[N]` | [pattern] |
| contract key count | `[N]` | `[N]` | [pattern] |

---

## 5) 验证

| 验证层 | 命令 | 结果 |
|---|---|---|
| compile/import | `[cmd]` | [pass/fail/blocked] |
| focused test | `[cmd]` | [pass/fail/blocked] |
| numerical smoke | `[cmd]` | [pass/fail/blocked] |
| real entry smoke | `[cmd]` | [pass/fail/blocked] |

Blocked smoke:

- [如果有，写清楚缺少 artifact / 数据 / 环境，而不是省略]

---

## 6) 风险与回退

| 风险 | 触发信号 | 回退策略 |
|---|---|---|
| [risk] | [signal] | [rollback] |

---

## 7) Follow-up

- [ ] [后续动作 1]
- [ ] [后续动作 2]

