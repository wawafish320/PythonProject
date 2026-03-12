# Phase A 模板：语义冻结与基线固化（不改逻辑）

适用范围：任何“先固化证据、后做删除/重构”的改动收敛任务。  
目标：在进入 Phase B/C 前，固定可复跑 baseline 与可定位 key 引用清单，避免后续改动失去锚点。

---

## 0) 使用方式（建议）

1. 在主计划文档的 Phase A 区块引用本模板。  
2. 每次任务都落一份 `phaseA_report.md`（必选，不建议省略）。  
3. 报告里必须包含：实际执行命令、固定路径、关键指标、引用清单统计。

---

## 1) 标准动作（A1/A2/A3）

### A1. 固化 baseline 命令并实际执行 1 次

- 选择“主链默认命令”（不带 fallback 参数）。
- 命令必须可直接复跑（包含完整入参，不依赖口头上下文）。
- 产物输出到固定目录（不要写临时目录）。

建议记录字段：
- 命令全文
- 执行日期
- 运行环境（可选）
- 输出路径（json/log）

---

### A2. 产出关键 key 引用清单（文件 + 行号）

- 用固定关键词集合（按本任务定义）。
- 用固定文件范围（至少覆盖本轮要改的核心文件）。
- 产出 raw 文本文件，保留原始命中行，便于后续逐条消解。

建议命令模板：

```bash
rg -n "[key1]|[key2]|[key3]" [file_scope_a] [file_scope_b] [file_scope_c] \
  > [PHASE_A_KEY_REFS_RAW_TXT]
```

建议统计维度：
- 总命中行数
- 按文件命中数
- 按 key 命中数

---

### A3. 固化关键指标快照（用于回归对照）

- 从 A1 的 baseline run 结果中提取固定口径指标。
- 明确 mask/切片规则与 metric source。
- 同时输出机器可读（JSON）与人类可读（TXT/MD）。

建议至少包含：
- global 指标
- 本任务关注 hotspot 指标（如特定 bone / SIC）
- count（样本数）

---

## 2) 产出物规范（建议设为必选）

- `docs/delete/[DATE]_[topic]_phaseA_report.md`
- `docs/delete/[DATE]_[topic]_phaseA_key_refs_raw.txt`
- `debug_output/[topic]_phaseA_[DATE]/mainchain_baseline_rerun/[baseline_run_json]`
- `debug_output/[topic]_phaseA_[DATE]/mainchain_baseline_rerun/phaseA_metrics_snapshot.json`
- `debug_output/[topic]_phaseA_[DATE]/mainchain_baseline_rerun/phaseA_metrics_snapshot.txt`

备注：
- 报告是索引页（single entry point），其余是证据文件（source of truth）。

---

## 3) 验收清单（可直接复制）

- [ ] baseline 可复跑（至少已实际执行 1 次）
- [ ] baseline 路径固定且在报告中可点击定位
- [ ] key 引用清单覆盖本轮核心改动文件
- [ ] 指标快照口径明确（mask + metric source）
- [ ] 报告与 raw 证据路径一致

---

## 4) `phaseA_report.md` 建议骨架

```md
# [DATE] [topic] Phase A 执行结果

## A1. baseline 固化
- 固化命令（已实际执行）
- 固定产出路径

## A3. baseline 指标快照
- mask
- metric source
- 关键 slice 指标表

## A2. key 引用清单
- raw 文件路径
- 生成命令
- 统计（总数/按文件/按 key）
- 快速定位锚点（代表 line anchors）

## 验收结论
- baseline 可复跑
- 清单覆盖核心文件
- 指标快照已固化
```

---

## 5) 与主计划文档的关联写法

在主计划文档 Phase A 小节增加：

```md
已完成产出见：`docs/delete/[DATE]_[topic]_phaseA_report.md`
模板参考：`docs/templates/change_refactor_phaseA_template.md`
```

这样后续 Phase B/C 的执行者可直接沿同一锚点推进。
