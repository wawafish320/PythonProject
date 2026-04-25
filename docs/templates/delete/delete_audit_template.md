# [YYYY-MM-DD] `[scope]` deletion audit / cleanup inventory

Date: [YYYY-MM-DD]  
Status: Draft / Static-audit ready / Checklist-ready / Done / Blocked  
Scope: `[files/modules]`  
Method: 静态引用分析（`rg` / AST / 代码阅读） + [runtime rerun / checkpoint scan / config scan，如适用]  

Goal: 为后续删除 legacy / compat / dead code 提供**可执行清单**，降低维护风险，同时保持主链行为不变。  
Non-goal:
- 不改算法/数学定义。
- 不改默认训练行为。
- 不改 checkpoint/load contract，除非本删除项明确包含 contract bump。
- 不删除 archive / retired evidence，除非本轮明确批准。

---

## 0) 一页版结论

按当前证据，候选项分为：

### A. Remove-Now

1. `[symbol/path]` — [为什么可以删]

### B. Remove-If-Clean

1. `[symbol/path]` — [还需要补哪类证据]

### C. Dedup-First

1. `[symbol/path]` — [需要先公共化/收敛重复逻辑]

### D. Keep / Revisit

1. `[symbol/path]` — [保留原因或复查条件]

---

## 1) 状态标记

- `Remove-Now`：静态引用基本清空，删除前只需轻量 smoke。
- `Remove-If-Clean`：看起来是 dead/cold，但还需要确认 docs/tools/tests/config/外部入口。
- `Dedup-First`：不是直接删除项；先抽公共 helper，确认行为一致，再清重复块。
- `Keep-Guard`：承担 fail-fast 或用户友好报错，不应直接删。
- `Keep-Compat`：薄 wrapper / 老名字仍是外部稳定 API。
- `Keep-Active`：训练、posttrain、validate、checkpoint 或 config 仍依赖。
- `Revisit-With-Rerun`：静态冷，但需要 runtime hits / fresh-chain rerun 佐证。

---

## 2) 证据扫描

### 2.1 静态引用

```bash
rg -n "[symbol_a]|[symbol_b]|[symbol_c]" train tools tests docs config
```

记录：

- 总命中数: `[N]`
- 代码命中: `[N]`
- 文档命中: `[N]`
- 只剩本 audit 文档命中: `[yes/no]`

### 2.2 Runtime / config / checkpoint 证据（如适用）

```bash
[runtime_smoke_or_checkpoint_scan_cmd]
```

记录：

- scanned files / runs: `[N]`
- hit files / runtime hits: `[N]`
- unreadable / blocked: `[N]`
- blocked 分类: `[none / details]`

---

## 3) Inventory 总表

| Item | Location | 类型 | 当前证据 | 建议 | 删除前门禁 |
|---|---|---|---|---|---|
| `[symbol]` | `[path:line]` | [dead/compat/dup/guard] | [evidence] | [status] | [smoke/check] |
| `[symbol]` | `[path:line]` | [dead/compat/dup/guard] | [evidence] | [status] | [smoke/check] |

---

## 4) 推荐执行顺序

### Phase A — no-behavior dead cleanup

目标：先删静态确定的 dead private code，降低噪音，不碰主链 contract。

候选：

- `[path:line]` `[symbol]`

删除前 checklist：

- [ ] `rg -n "[symbol]" train tools tests docs config`
- [ ] `python3 -m py_compile [touched_files]`
- [ ] import/build smoke: `[cmd]`
- [ ] 如涉及 checkpoint/config：完成 scan 且 `HIT_FILES=0`

最终勾选：

- [ ] Remove
- [ ] Keep
- [ ] Revisit

### Phase B — compat / contract cleanup

目标：删除或收口兼容壳，避免旧入口继续扩散。

候选：

- `[path:line]` `[symbol]`

额外门禁：

- [ ] 有明确迁移路径或 fail-fast 信息
- [ ] 旧入口 repo 内无有效调用
- [ ] 如果是 checkpoint/config contract，已记录版本/错误信息变化

### Phase C — duplicate / cold branch cleanup

目标：先公共化重复逻辑，再删除旧分支，避免行为漂移。

候选：

- `[path:line]` `[symbol/block]`

额外门禁：

- [ ] before/after keyset 一致
- [ ] focused numerical smoke 通过
- [ ] 没有新旧双实现并存

---

## 5) 单项记录模板

### `[symbol/path]`

**位置**

- `[path:line]`

**它是什么**

- [一句话说明用途]

**静态证据**

- [rg / AST / code reading 结论]

**风险点**

- [checkpoint / config / external CLI / docs / tests 风险]

**当前建议**

- `[Remove-Now / Remove-If-Clean / Dedup-First / Keep-* / Revisit-With-Rerun]`

**删除前 checklist**

- [ ] `[cmd]`
- [ ] `[cmd]`
- [ ] `[cmd]`

**执行回填**

- [ ] 删除完成
- [ ] 验证完成
- [ ] 如未删除，记录保留原因

---

## 6) 验证门禁

| 验证层 | 命令 | 必须/可选 | 结果 |
|---|---|---:|---|
| compile | `python3 -m py_compile [files]` | 必须 | [pass/fail] |
| import smoke | `[cmd]` | 必须 | [pass/fail] |
| focused unit | `[cmd]` | 按风险 | [pass/fail/blocked] |
| runtime smoke | `[cmd]` | 按风险 | [pass/fail/blocked] |
| checkpoint/config scan | `[cmd]` | 按风险 | [pass/fail/blocked] |

Stop-rule:

- smoke 失败且不是本地 artifact 缺失：停止删除，先修验证或回滚。
- checkpoint/config 有命中：停止删除，先分类并更新 contract/迁移策略。
- 删除后出现新旧双轨：回收旧实现或回滚，不进入下一 phase。

---

## 7) Final report 回填

- 删除项: `[list]`
- 保留项: `[list + reason]`
- 验证结果: `[commands + pass/fail]`
- 行为/contract 影响: `[unchanged / intentional change]`
- 后续 cleanup: `[follow-up docs or none]`

