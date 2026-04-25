# [2026-04-25] Checkpoint fingerprint / manifest rollout plan

Date: 2026-04-25  
Status: rollout tracking only (not a machine-enforced gate); Phase 3 landed; Phase 4 (compare + report) landed at posttrain build shell; Phase 5 (context-aware enforce) landed at posttrain build shell, with caller wiring still required  
Owner: train / checkpoint contract cleanup  
Scope: `train/training_MPL.py`, `train/posttrain_build_shell.py`, `train/checkpoint/`, `docs/refactor/`, `docs/removal_policy.md` alignment  
Goal: 为 basetrain / posttrain 建立一套可回放、可分段比对、可 fail-fast 的 checkpoint fingerprint 协议，用来抓住“能跑但语义/组装顺序/消费信号已漂移”的隐性错误。  
Non-goals: 本轮不做 legacy lane，不把 fingerprint 当成 file-layout hash，不替代现有 contract version gate，不把 training policy 强行提升为 architecture mismatch。

Related:

- `docs/removal_policy.md`
- `train/training_MPL.py`
- `train/posttrain_build_shell.py`
- `train/checkpoint/contract.py`

---

## 0. 一页版结论

这轮 fingerprint 工程的核心目标不是“多一个 checksum”，而是补齐 **silent semantic drift observability**：

- 同一个 checkpoint 仍然能 load / train / rollout
- 但实际已经不是原来那条语义路径
- 变化点可能来自：
  - head / branch 拓扑变化
  - build / attach 顺序变化
  - 某个模块开始消费新的信号，或不再消费旧信号
  - runtime attach 拓扑变化

因此本轮不采用“单一 hash”方案，而采用：

1. **canonical manifest**
2. **4 段 fingerprint**
3. **regularization**
4. **分级 compare / enforce**

其中最重要的约束是：

- `module_graph_hash` **对 file/import layout 不敏感**
- `build_order_hash` **只看 semantic build skeleton，不看运行噪声**
- **不提供 legacy lane**
- 未来新增 hash 段时，`missing optional field = no-check`，不是 mismatch

---

## 1. 为什么现在需要这件事

当前树里的几个事实决定了“只看能不能跑”已经不够：

1. basetrain save 侧目前主要保存 `model` + `config`，还没有正式的多段 fingerprint / manifest 协议。  
2. posttrain build/load 侧会从 checkpoint tensor shape、现有 cfg override、runtime wiring 反推当前 build state。  
3. 现有回归里虽然已经有 `state_dict` fingerprint 思路，但它主要覆盖的是 **weights 内容稳定性**，不是 **语义拓扑稳定性**。  

这意味着当前最危险的错误类型不是：

- shape mismatch
- strict load 直接报错

而是：

- **shape 仍然对**
- **strict=False 仍然能过**
- **训练仍然能跑**
- 但语义已经漂移，导致后续排错时间被大量浪费

本轮 fingerprint / manifest 的目标就是把这类错误前移到 checkpoint/save/load boundary。

---

## 2. 设计原则（本文件锁死的决策）

### 2.1 先 freeze，再建 baseline

当前 de-maze 刚完成，真实训练里的拓扑还没有稳定一轮。  
因此必须先过一个 **freeze gate**，再刷 fingerprint baseline：

- 先完成 `e_t / contacts_meas` 的 micro-fix
- 再跑一轮真实 basetrain/posttrain smoke
- 确认拓扑真正静止后，再进入 fingerprint write-only

否则当前刷出来的 baseline 很可能在下一轮微修后整体失效。

### 2.2 `module_graph_hash` 不看 import path / file layout

`module_graph_hash` 的目标是刻画 **语义模块图**，不是刻画“代码今天在哪个文件里”。

因此本文件明确规定：

- Phase E 跨文件迁移 **不应** 让 `module_graph_hash` 自动变化
- `module_graph_hash` **不得**直接基于 import path / source file path / 原始 `repr(module)` 计算
- 推荐基于以下稳定语义字段计算：
  - `component_slot`
  - `component_kind`
  - normalized submodule structure
  - normalized input / output contract
  - normalized consumes / produces 集合

如果只是 file move / import 路径变化，hash 应保持不变；  
只有真实语义变化（消费信号、模块结构、拓扑接线变化）才应触发 mismatch。

### 2.3 `build_order_hash` 只看 semantic build skeleton

`build_order_hash` 不负责描述具体 Python 执行细节，只负责描述以下信息：

- 构建阶段顺序
- 每一步 attach / inject 了哪些语义字段
- 每一步消费了哪些上游产物
- 每一步产出了哪些下游可见运行时对象

它要抓的是：

- “先 attach runtime 再 build loss” vs “先 build loss 再 attach runtime”
- “现在 `direct_pose_head` 多消费了 `contacts_meas`”
- “某个 runtime attr 不再被 attach”

它**不应该**抓：

- 临时路径
- run name
- out dir
- Python object id
- RNG state
- dict / set 的原始插入顺序

### 2.4 不提供 legacy lane

本计划与 `docs/removal_policy.md` 保持一致：

- enforce 之后，**没有** `--allow_legacy` / `--accept_old_fingerprint` / `legacy lane`
- 对于缺少 fingerprint block 的旧 checkpoint，行为是：
  - write-only / compare-only 阶段：允许 load，但打印明确说明
  - enforce 阶段：直接 fail-fast

错误信息应明确说明：

- checkpoint 缺少 fingerprint metadata
- 该 checkpoint 早于 fingerprint policy 引入日期
- 当前主线不再提供 legacy lane
- 需要 regenerate with current mainline

### 2.5 contract version gate 继续保留

fingerprint **不替代**现有 `checkpoint_contract.version` 机制。

两者职责不同：

- `contract version`：定义 retired boundary / schema retirement
- `fingerprint`：定义 semantic equivalence / drift detection

因此 load 侧顺序应保持：

1. 先过 contract version gate
2. 再做 fingerprint compare / enforce

---

## 3. 协议拆分：4 段 required + 1 段 log-only

### 3.1 Required segments

#### A. `io_signature_hash`

含义：

- `forward` 输入 / 输出边界
- 输入输出 key、shape 语义、dtype、可选性

用途：

- 最稳定、最严格的下游 contract
- mismatch 时直接 fail-fast

#### B. `module_graph_hash`

含义：

- 语义模块图
- head / branch / terminal / shared block 的 normalized graph
- 每个模块的 normalized consumes / produces

用途：

- 抓“IO 没变，但内部 semantic graph 已变”
- mismatch 时直接 fail-fast

#### C. `build_order_hash`

含义：

- basetrain / posttrain entry shell 的 semantic build skeleton
- build / attach 顺序
- 每一步 inject / attach 的 attr 集合

用途：

- 抓“组装顺序错了”“attach 拓扑变了”
- mismatch 时直接 fail-fast

#### D. `weights_hash`

含义：

- normalized `state_dict` 权重指纹

用途：

- 区分“语义同构但权重不同”的合法场景
- mismatch 默认 warning，不阻断合法 finetune / resume / donor replacement

### 3.2 Log-only segment

#### E. `train_policy_hash`

含义：

- `freerun_stage_schedule`
- trainer-side policy knobs
- architecture 外的 training-time scheduling policy

用途：

- 保留审计性
- 不把 training policy 误当成 architecture drift

默认行为：

- save 时写入
- load 时 compare + log
- 不参与 required enforce

---

## 4. Canonical manifest 设计

hash 只是快速 gate；真正给人排错用的，是 **manifest**。

每个 checkpoint 至少应携带一个可读的 `manifest_summary`，用于回答：

- 这个模型有哪些语义组件？
- 每个组件消费什么信号？
- 每个组件产出什么结果？
- build / attach 的先后顺序是什么？

### 4.1 推荐最小字段

```json
{
  "fingerprint_schema_version": 1,
  "required_segments": [
    "io_signature_hash",
    "module_graph_hash",
    "build_order_hash",
    "weights_hash"
  ],
  "optional_segments": [
    "train_policy_hash"
  ],
  "io_signature": {
    "inputs": [],
    "outputs": []
  },
  "module_graph": {
    "components": []
  },
  "build_trace": {
    "steps": []
  },
  "train_policy": {
    "schedule_keys": []
  }
}
```

### 4.2 `module_graph` 推荐字段

每个 component 建议至少记录：

- `component_slot`
- `component_kind`
- `enabled`
- `consumes`
- `produces`
- `normalized_config`
- `children`

其中：

- `component_slot` 表示语义位置，例如 `shared_encoder`、`direct_pose_head`、`lambda_fusion_head`
- `component_kind` 表示语义类别，例如 `mlp_head`、`gru_cell`、`fusion_head`
- `normalized_config` 必须先 regularize

### 4.3 `build_trace` 推荐字段

每个 step 建议至少记录：

- `step_name`
- `step_order`
- `consumes`
- `produces`
- `attached_attrs`
- `notes`

`build_trace` 的粒度采用“两层并存”：

- `module_graph_hash`：per-head / per-component 粒度
- `build_order_hash`：big-chunk / build-skeleton 粒度

这是为了避免 `build_order_hash` 对过细内部实现过敏。

---

## 5. Regularization 规则（必须实现）

没有 regularization，fingerprint 会被噪声淹没。

本轮强制要求以下规则：

1. `dict` / `set` / attr 集合一律按稳定 key 排序  
2. 不把 `id(...)` / 内存地址 / 原始 `repr` hex addr 算进 hash  
3. 不把临时路径、`out_dir`、`run_name`、`bundle_json_path` 之类环境噪声算进 hash  
4. 不把 `torch.Generator` 当前 state 算进 hash  
5. 所有 list-like config 先 canonicalize，再序列化  
6. 所有 `None` / 缺省值统一成固定 canonical 表示  
7. manifest 序列化时使用稳定 key 顺序与稳定编码格式  

特别说明：

- `train/runtime_attach.py` 中面向运行时的路径型 / run-meta 字段属于 **volatile runtime metadata**
- 这些字段可以保留在 live runtime 上，但**不得**进入 required fingerprint input

---

## 6. Rollout phases

### Phase 0 — Freeze gate

目标：

- 先让 de-maze 后拓扑稳定

动作：

- 完成 `e_t / contacts_meas` micro-fix
- 跑一轮真实 basetrain / posttrain smoke
- 确认 build graph 不再因为当前已知小问题继续抖动

影响：

- 指纹落地会稍晚一点
- 但能避免 baseline 在一周内被反复重刷

### Phase 1 — Policy only

目标：

- 先把语义边界写死，再开始实现

动作：

- 新增本 policy doc
- 明确 required / optional segments
- 明确 Phase E 对 fingerprint 的不变量
- 明确 no legacy lane
- 明确 mismatch / missing 的处理矩阵

影响：

- 无运行时行为变化
- 为后续实现提供稳定边界

### Phase 2 — Canonical manifest + hash core

目标：

- 建公共 fingerprint 核

动作：

- 新增集中模块，建议放在 `train/checkpoint/fingerprint.py`
- 实现：
  - manifest builder
  - regularize utils
  - segment hash calculator
  - comparison helpers

影响：

- 新增公共基础层
- 主要风险是 regularization 不全导致 false-positive

实现回填（2026-04-25）：

- 已落地 `train/checkpoint/fingerprint.py`
- 已实现 canonical manifest dataclass、regularization helpers、segment hash core、compare/report helpers
- 当前已覆盖 `EventMotionModel` component manifest、basetrain/posttrain build skeleton manifest、checkpoint fingerprint metadata builder
- focused tests 已补到 `tests/train/test_checkpoint_fingerprint_phase2.py`

### Phase 3 — Save-side write-only

目标：

- checkpoint 开始落盘 fingerprint / manifest

动作：

- 在 basetrain ckpt save 路径写入：
  - `fingerprint_schema_version`
  - `fingerprints`
  - `manifest_summary`
- posttrain contract ckpt 也建议带同样字段

当前落地点：

- `train/training_MPL.py`
- `train/posttrain.py`

影响：

- checkpoint 体积略增
- 训练行为不变
- 旧 reader 通常可忽略新增 metadata

实现回填（2026-04-25）：

- basetrain save 路径已写入 `fingerprint_schema_version`、`fingerprints`、`manifest_summary`
- posttrain final / step checkpoint 已写入 fingerprint metadata，并带 `checkpoint_contract`、`build_cfg`
- focused tests 已补到 `tests/train/test_checkpoint_fingerprint_phase3_save.py`

### Phase 4 — Load-side compare + report

目标：

- 先获得可见性，不急着拦截

动作：

- 在 load / build shell 入口计算“当前期望 fingerprint”
- 与 checkpoint 内的 fingerprint 做分段 compare
- 输出结构化 compare summary

当前推荐接入点：

- `train/posttrain_build_shell.py`
- 如有需要，补到 basetrain resume/load 路径

影响：

- 立刻能看到：
  - 哪段 match
  - 哪段 mismatch
  - 哪段 missing
- 但暂不阻断现有实验

### Phase 5 — Enforce

目标：

- 把 silent drift 正式前移到 boundary

动作：

- `load_context=resume` 时：
  - `io_signature_hash` mismatch → fail-fast
  - `module_graph_hash` mismatch → fail-fast
  - `build_order_hash` mismatch → fail-fast
- `load_context=chain_hop` 时：
  - `io_signature_hash` mismatch → compare + report only
  - `module_graph_hash` mismatch → compare + report only
  - `build_order_hash` mismatch → compare + report only
- `weights_hash` mismatch → warning
- `train_policy_hash` mismatch → log-only
- fingerprint block 整体缺失 → fail-fast（`resume` / `chain_hop` 均如此）

影响：

- `chain_hop` 当前只是 phase-1 policy waiver，不是 drift detection complete
- 当前 `chain_hop` 还不能区分 `expected stage delta` 与 `wrong upstream stage`
- 因此在 `chain_hop` 下，compare report 是主要可见信号，operator 需要人工检查
- `chain_hop` 不是 legacy / pre-policy checkpoint 的通行证；缺 fingerprint block 仍然会被明确挡住
- 这不是“完成态”或“完整保护”；它只是 caller-side context-aware policy 的第一版
- 在 caller 显式提供 `load_context` 之前，对应 entrypoint 会按设计 fail-fast；caller wiring 是本 rollout 的关键路径，不是可选后续项

### Phase 4.5 — Policy-waived lineage handling

目标：

- 明确处理当前 `chain_hop` policy-waived lineage，避免把临时 waiver lane 误写成 canonical protected baseline

动作：

- 对当前 policy-waived lineage（包括待处理的 `70R` downstream rerun / continuation 入口）必须显式二选一：
  - 退役并按显式 `resume` / `chain_hop` caller policy 重跑
  - 保留为带标签的人工对照 lineage
- 不允许把 policy-waived lineage 静默升级为“已完整受保护”的 canonical 主链

影响：

- operator 需要尽快给出 lineage disposition，而不是无限期依赖 `chain_hop` waiver
- compare report 继续是 `chain_hop` lane 的主要可见信号，直到更完整的 drift detection 落地

### Phase 6 — Tests / diff tooling / operator UX

目标：

- 降低 false-positive，提高排错效率

动作：

- 加 determinism tests
- 加 save/load round-trip tests
- 加 mismatch 分类测试
- 为常见 mismatch 输出短 diff hint

影响：

- 测试成本上升
- 但 operator 调试收益显著更高

---

## 7. Mismatch / missing 处理矩阵

| Segment | `match` | `mismatch` | `missing_required` | `missing_optional` | 默认处置 |
|---|---|---|---|---|---|
| `io_signature_hash` | pass | fail | fail | n/a | 最严格 |
| `module_graph_hash` | pass | fail | fail | n/a | 语义已变 |
| `build_order_hash` | pass | fail | fail | n/a | 顺序/attach 已变 |
| `weights_hash` | pass | warn | fail | n/a | 合法 finetune 可不同 |
| `train_policy_hash` | pass | log | n/a | no-check | 审计，不阻断 |

说明：

- `missing_required` 与 `mismatch` 都应输出明确 segment 名称
- `missing_optional` 明确记为 `no-check`，不是 `mismatch`
- 缺整块 fingerprint metadata 的旧 checkpoint，在 enforce 阶段视为 `missing_required`

---

## 8. 错误消息格式（必须标准化）

如果只打印 “fingerprint mismatch” 而不指出哪一段变了，排错价值会大幅下降。

因此要求统一错误格式至少包含：

1. `segment`
2. `status`
3. `ckpt_hash`
4. `current_hash`
5. `short_diff_hint`
6. `next_action`

建议格式：

```text
[FATAL] checkpoint fingerprint mismatch.
- segment: module_graph_hash
- status: mismatch
- ckpt_hash: <...>
- current_hash: <...>
- hint: direct_pose_head consumes changed: [cond, plan_z] -> [cond, plan_z, contacts_meas]
- action: regenerate checkpoint with current mainline or revert to the last supported semantic graph.
```

对于缺 fingerprint block 的旧 checkpoint：

```text
[FATAL] checkpoint missing required fingerprint metadata.
- segment: fingerprint_block
- status: missing_required
- hint: checkpoint predates fingerprint policy introduced on 2026-04-25
- action: regenerate this checkpoint with current mainline; no legacy lane is provided.
```

---

## 9. Future extensibility

未来新增 hash 段时，不能要求所有历史 checkpoint 立刻全量 regenerate。

因此协议层应显式区分：

- `required_segments`
- `optional_segments`

规则固定如下：

1. 新增 segment 默认先进入 `optional_segments`
2. `optional_segments` 缺失时，状态记为 `missing_optional`
3. `missing_optional` 的含义是 `no-check`，不是 `mismatch`
4. 只有当某个 segment 升级为 required 时，才需要新的 schema 版本与新的 enforce 边界

这样可以做到：

- 协议可扩展
- 不会因为“新加一个 hash 段”就强迫全量旧 ckpt 立刻失效

但这条规则**不适用于 pre-policy checkpoint**：

- 它们缺的是整块 fingerprint protocol
- 在 enforce 阶段必须 fail-fast

---

## 10. 推荐实现落点

### 10.1 新增模块

- `train/checkpoint/fingerprint.py`

建议职责：

- `regularize_*`
- `build_*_manifest`
- `compute_*_hash`
- `compare_fingerprints`
- `format_fingerprint_mismatch`

### 10.2 Save-side 接入

- `train/training_MPL.py`
- `train/checkpoint/contract.py`

### 10.3 Load-side 接入

- `train/posttrain_build_shell.py`
- 如有必要，补到 basetrain resume/load path

### 10.4 文档

- 本文：计划 / policy / rollout
- 后续如果进入 enforce，可再补一份 `docs/fingerprint_policy.md` 或在本文中转正为 standing policy

---

## 11. Acceptance criteria

进入 Phase 3 之前，应满足：

- [ ] de-maze 后真实训练拓扑已稳定一轮
- [ ] `module_graph_hash` 的 file-layout-insensitive 语义已写死
- [ ] `build_order_hash` 的 build-skeleton 边界已写死
- [ ] no legacy lane 决策已写死
- [ ] required / optional segments 已写死

进入 Phase 5 enforce 之前，应满足：

- [ ] save-side fingerprint 已稳定落盘一轮
- [ ] load-side compare log-only 已跑过真实样本
- [ ] mismatch error 格式已定型
- [ ] regularization determinism tests 已通过
- [ ] 至少一个真实“能跑但语义错”的历史案例能被新协议抓住

当前实现进度（2026-04-25）：

- [x] Phase 2 canonical manifest + hash core
- [x] Phase 3 save-side write-only
- [ ] Phase 4 load-side compare + report
- [ ] Phase 5 enforce

补充说明：

- 上面的 status / checklist 只用于 rollout tracking，不是 machine-enforced gate
- 实际 load-side 行为必须由 caller 显式提供 `load_context` 决定，不能从 stage 编号、checkpoint metadata 或其他上下文自动推断
- caller wiring 不是 nice-to-have：在对应入口显式传入 `load_context` 之前，fail-fast block 是预期行为

---

## 12. 当前建议的执行顺序

按优先级推荐：

1. 先做 `e_t / contacts_meas` micro-fix，冻结当前 de-maze 拓扑  
2. 写清 policy（本文）  
3. 实现 canonical manifest + 4 段 fingerprint 核  
4. save-side write-only  
5. load-side compare + report  
6. 跑真实样本，确认 false-positive 可控  
7. 再切 enforce  

一句话总结：

> 这轮 fingerprint 工程的目标不是“让旧 checkpoint 更兼容”，而是把“能跑但语义漂了”的错误尽早、明确、可回放地暴露出来。
