# TrainBase v2 流程重构设计（Core / Patch 分层）

> Last updated: 2026-03-09  
> 目标：在不破坏 Stage6→Stage7 主链质量的前提下，降低 train(base) 维护复杂度，明确“哪些属于 base core、哪些属于 patch 实验层”。
> Update (2026-03-09): `whitebox` runtime/validate lane 已从当前 mainline 退休；本文若提到 `whitebox`，除非明确标成 historical/archive，否则都应按当前 contract 替换为 `pretrain_contact`。

关联输入文档：
- `docs/Problems/active/2026-03-02_trainbase_simplify_review.md`
- `docs/contact_loop_closure_design.md`
- `docs/contact_meas_head_redesign_lowerbody_nohist.md`
- `docs/contact_meas_whitebox_stability.md`
- `docs/contact_phase_state_prevphase_tta.md`（历史路线图，主链已移除）

---

## 0) TL;DR（结论先行）

1. train(base) 需要从“按 ckpt 隐式激活功能”改为“显式配置 + 兼容层”。
2. **Core 保留**：inc/direct 双专家 + λ fusion + 统一评估口径（apply on/off）+ 明确 meas source contract。
3. **Patch 保留**：`event_clock`、`learned/whitebox meas 细节稳定化`（这些目前仍有场景耦合和不确定性）。
4. 简化迁移顺序按风险执行（2026-03-04 更新）：
   - Step C：`contact_phase_state` 已从主链移除（完成）
   - Step B：`posttrain contact_meas_provider` 已移除，主链固定 `pretrain_contact`（`whitebox` runtime 已退休）
   - Step A：`event_clock` 归因后再决策（最后）

---

## 1) 当前复杂度根因（为什么难维护）

### 1.1 运行行为被 ckpt key 隐式驱动
历史上 active chain 的 ckpt 同时包含：
- `contact_meas_head.*`
- `event_clock_*`
- `contact_phase_state_*`

结果是：即使 config 没“主动训练”这些分支，它们仍可能在 rollout 路径参与运行，导致“看起来像 base，其实混着 patch 行为”。
2026-03-03 起主链 runtime + active config 已移除 `contact_phase_state*`。

### 1.2 同名概念在训练/验证/部署语义不一致
`contacts_meas` 同时存在：
- learned head 输出（model）
- white-box（由预测 pose 推导）
- 外部 override（deploy 期的真实传感器/上游）

若不显式钉死来源，指标可解释性会快速下降。

### 1.3 闭环模块之间存在耦合放大
`contacts_meas` 不稳会同时影响：
- `contacts_err`
- phase reset（历史上也包含 `contact_phase_state` 分支）
- `event_clock`
- λ reliability（`contacts_err` 模式）

所以 trainbase 需要“分层治理”，不能继续把所有机制放在同一条默认路径里联动调参。

---

## 2) TrainBase v2 分层定义

### 2.1 Core（默认主链，必须长期稳定）

Core 只保留“稳定可解释、跨实验可复用”的能力：

1. **双专家骨架**：incremental + direct。
2. **Stage2 λ fusion 主干**：支持 apply off（专家诊断）与 apply on（系统表现）。
3. **统一时域口径**：multi-cycle 固定按 cycle 内 transition 切片，time-index 使用 `cycle` 模式。
4. **显式 meas source contract**：posttrain 固定 `pretrain_contact`，来源对比仅在 validate lane 显式进行（`contacts_meas_source`）。

### 2.2 Patch（实验增量层，可插拔）

1. `event_clock`（plan-z 校正与 gate）
2. `contacts_meas` 的实现细节强化：
   - learned meas freerun-robust 训练
   - white-box 的 `hit_flag/ground_z` 稳定化策略
3. `contact_phase_state`（prev_phase / TTA）转为历史研究路线，不再属于主链 patch 开关。

Patch 默认不作为 trainbase 的“必经流程”，只在目标实验开启。

---

## 3) 模块归类：哪些并入 Core，哪些保留 Patch

| 模块 | 当前状态 | 归类建议 | 原因 |
|---|---|---|---|
| 双专家（inc/direct）+ λ fusion | 主链核心 | **Merge -> Core** | 这是 Stage6→Stage7 主链的基础能力 |
| `lambda_fusion_apply` 双口径评估 | 已可用 | **Merge -> Core** | 能稳定区分“专家质量”与“闭环系统质量” |
| time-index `cycle` 口径 | 已验证 | **Merge -> Core** | 避免 multi-cycle time-PE OOD，降低伪回归 |
| `contacts_meas` 来源命令契约 | posttrain 已固定 `pretrain_contact`；validate 保留 `pretrain_contact|model|gt|zero` | **Merge -> Core（接口）** | 主链可解释性与复现性前提；运行入口职责清晰 |
| historical white-box meas 内部细节（gate/ground_z 等） | 已退出 mainline | **Archive reference** | 仅供旧实验复盘，不再属于当前 core/patch 执行面 |
| learned meas head（含 meas-only / rollout 训练） | teacher 与 freerun 存 gap | **Patch** | 依赖数据域与闭环漂移形态，需专项训练 |
| `contact_phase_state` | 主链已移除（2026-03-03） | **Retired from mainline** | 仅保留历史分支/复现实验，不再进入主链配置与运行 |
| `event_clock` | 关停会出现质量回退 | **Patch（最后治理）** | 仍需 attribution，暂不适合先删 |

---

## 4) 新的 trainbase 执行流程（推荐）

### 4.1 默认 Base Lane（Core only）

1. 按 `docs/posttrain_pipeline.md` 的 Stage6→Stage7 主链训练。
2. 每个阶段固定执行两类评估：
   - **apply off**：看 `inc vs direct` 专家质量
   - **apply on**：看融合后的系统质量
3. `contacts_meas` 主链语义固定为：posttrain 使用 `pretrain_contact`；来源 A/B 在 validate 通过 `contacts_meas_source` 显式声明。
4. `contact_phase_state` 已从主链移除；`event_clock` 仅在 profile 允许时启用。

### 4.2 Legacy Lane A：Phase Anchor（contact_phase_state / prev_phase / TTA）

适用场景：仅历史复现实验需要显式 phase 状态建模时。

要求：
- 不进入当前主链；必须使用隔离分支/快照执行。
- 必做 `t=0` init 策略与弱事件 fallback 验收。

### 4.3 Patch Lane B：Contact Meas 研究（learned / white-box）

适用场景：需要替换/增强 meas 信号质量时。

要求：
- 先钉死 `contacts_meas_source`，再做任何结论。
- learned meas 必须区分 teacher 与 freerun 指标。
- historical white-box 研究若要复现，必须切到历史快照；当前 mainline 不再保留这条执行路径。

### 4.4 Patch Lane C：Event Clock

适用场景：需要验证 plan-z 动态纠偏收益时。

要求：
- 先完成 attribution（收益来自哪里，副作用在哪）。
- 未完成归因前，不并入默认简化动作。

---

## 5) 配置收敛建议（trainbase 统一入口）

建议将 trainbase 主配置收敛为“显式模式”：

```yaml
# TrainBase v2（建议语义，不要求一次性改完）
trainbase_profile: core           # core | meas_patch | eventclock_patch | full
posttrain_contact_signal: pretrain_contact_fixed   # fixed in posttrain (no provider knob)
validate_contacts_meas_source: model       # model | pretrain_contact | gt | zero
event_clock_mode: auto            # auto | on | off
```

建议解释：
- `core` profile 下默认：
  - `contact_phase_state` 不可用（主链已移除）
  - posttrain 不暴露 `contact_meas_provider*`，rollout contacts 走固定 `pretrain_contact`
  - `event_clock_mode=auto`（暂不强改，避免质量回退）
- 来源对比统一放在 validate lane（`contacts_meas_source`）。

---

## 6) 迁移计划（低风险 -> 高风险）

### Step C（已完成，2026-03-03）：`contact_phase_state` 从主链移除

目标：把 phase-state 从 trainbase 主链入口、模型实现、posttrain/freerun/runtime config 中彻底移除。

验收：
- 主链代码不再出现 `contact_phase_state*`。
- active config 不再包含 `contact_phase_state_* / contact_phase_state_mode`。
- phase 移除后主指标对齐基线（见 2026-03-03 执行记录）。

### Step B（已完成，2026-03-09 口径更新）：`posttrain contact_meas_provider` 移除 + 来源契约固定

目标：统一 posttrain/validate 语义边界，去除 provider 策略层歧义。

验收：
- posttrain 入口已移除 `contact_meas_provider*` 语义（旧入口不再存在）。
- posttrain rollout contacts 固定 `pretrain_contact`。
- validate 来源对比统一在 `pretrain_contact|model|gt|zero`。

### Step A（最后做）：`event_clock` 归因与收敛

目标：在保留质量的前提下决定 event_clock 的最终定位。

验收：
- 明确其增益来源（Round0/Round1、哪些关节、哪些时段）。
- 若要默认关闭，需先证明关键场景回退可接受。

---

## 7) 验收指标（统一看板）

最小指标集建议固定为：

1. **专家质量（apply off）**
   - `GeoLocalDeg`（inc）
   - `DirectGeoLocalDeg`（direct）
2. **系统质量（apply on）**
   - `BlendGeoLocalDeg`
   - Round0 vs Round1 的误差曲线
3. **闭环信号质量**
   - `ContactMeasGtAbsMean`
   - `ContactErrAbsMean`
   - `ContactsMeasSourceApplied`（来源追踪）
4. **性能/代价**
   - 单步推理耗时（至少重复多次取均值）

---

## 8) 文档治理建议（本目录用途）

`docs/trainbase_design/` 作为 trainbase 设计收敛目录，建议只放两类文档：

1. **流程级文档**（如本文）：定义 core/patch 边界、配置语义、迁移顺序。
2. **决策记录文档（ADR）**：每次决定“并入 core / 保留 patch / 下线”都补一页短记录。

这样可以把“实验细节”留在各专项文档，把“trainbase 主流程决策”集中管理，避免知识分叉。

---

## 9) 与现有文档的分工

- 主运行入口与命令链：`docs/posttrain_pipeline.md`
- 本文：trainbase v2 分层与治理（core/patch）
- 闭环机制细节：`docs/contact_loop_closure_design.md`
- learned meas 结构与调试：`docs/contact_meas_head_redesign_lowerbody_nohist.md`
- historical white-box 复盘：`docs/contact_meas_whitebox_stability.md`
- phase/TTA 历史路线图：`docs/contact_phase_state_prevphase_tta.md`（主链已移除）
