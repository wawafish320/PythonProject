# [2026-03-16] `train/models.py` EventMotionModel / direct-pose 重复逻辑收敛计划（v1）

Date: 2026-03-16  
Status: Completed（Phase A / Phase B / Phase C1 / Phase C2 / Phase C3 / Phase D 已落地）  
Scope: `train/models.py`（聚焦 `EventMotionModel`、`EventMotionModel.forward`、direct-pose head 初始化、局部 override/ablate helper）  
Goal: 在**不改变训练/推理语义**前提下，优先收敛 `forward` 与 direct-pose 初始化中的重复骨架，降低 event-clock on/off、split/non-split 分支漂移风险。  
Non-goal: 不改 loss 数学定义、不改默认配置行为、不改 checkpoint key 兼容策略、不做跨文件迁移。

---

## 0) 基线现状（针对本轮热点）

注：本节保留 Phase A 基线口径，用于后续 before/after 对照；最新执行状态以 Phase A / Phase B 报告为准。

当前代码快照核对（`train/models.py`）：

- 文件总长度：`6197` 行
- `EventMotionModel`：`train/models.py:513` 起，约 `3961` 行
- `EventMotionModel.forward`：`train/models.py:2256` 起，约 `2218` 行
- `EventMotionModel` 内 `except Exception: pass`：`39` 处
- 本轮已确认热点重复块：至少 `9` 组

已确认的重复热点：

- `contacts_input` shape canonicalization / pad / expand：
  - `train/models.py:2579`
  - `train/models.py:2942`
- `meas_logits_prev -> (B, C)` canonicalization：
  - `train/models.py:2603`
  - `train/models.py:2966`
- `phase_z / phase_event_age` 初始化骨架：
  - `train/models.py:2645`
  - `train/models.py:3000`
- contact-plan 逐时间步 loop 骨架：
  - `train/models.py:2777`
  - `train/models.py:3097`
- direct-pose trunk 定义：
  - `train/models.py:1481`
  - `train/models.py:1516`
- leg residual / gate / shared / shared-gate head 的同构 MLP + last-layer init：
  - `train/models.py:1533`
  - `train/models.py:1556`
  - `train/models.py:1614`
  - `train/models.py:1637`
- `rot6d` joint count 解析重复实现（已存在 helper 却未复用）：
  - helper：`train/models.py:133`
  - 重复手写：`train/models.py:1680`
  - 重复手写：`train/models.py:1725`
- `plan_in` / `meas_in` override 的 `(B,T,C)` 对齐：
  - `train/models.py:3501`
  - `train/models.py:3560`
- `_ablate` layout A/B 两版近似实现：
  - `train/models.py:4102`
  - `train/models.py:4143`

---

## 1) 一页版结论（先看这个）

**当前最值得先做的三件事（P0）**：

1. 抽出 `contacts/meas/override/phase` 的 tensor canonicalization helper，统一 `(B,T,C)` / `(B,C)` / `(B,2C)` 对齐逻辑。  
2. 抽出 direct-pose 通用 MLP builder + last-layer init helper，并强制复用 `_resolve_rot6d_joint_count(...)`。  
3. 将 contact-plan per-step loop 收敛为单一 scaffold，把 event-clock on/off 差异压缩到 hook 或分支块，而不是复制整段循环。  

**执行顺序**：`Phase A 语义冻结 -> Phase B 低风险 helper 收敛 -> Phase C forward/direct-pose 主体重构 -> Phase D 清理与防回归`  
**验收原则**：每一步都必须保持语义等价，并提供 before/after 结构指标与最小 smoke 对照。

**状态更新（2026-03-17）**：

- Phase A 已完成，详见 `docs/changes/2026-03-16_event_motion_model_refactor_phaseA_report.md`
- Phase B 代码已完成，详见 `docs/changes/2026-03-16_event_motion_model_refactor_phaseB_report.md`
- Phase C1 已完成，详见 `docs/changes/2026-03-16_event_motion_model_refactor_phaseC1_report.md`
- Phase C2 已完成，详见 `docs/changes/2026-03-16_event_motion_model_refactor_phaseC2_report.md`
- Phase C3 已完成，详见 `docs/changes/2026-03-16_event_motion_model_refactor_phaseC3_report.md`
- Phase D 已完成，详见 `docs/changes/2026-03-16_event_motion_model_refactor_phaseD_report.md`
- 本轮 roadmap 已闭环；如需继续清理新热点，建议另起一份后续 roadmap

---

## 2) 当前问题（按优先级）

### 2.1 P0（必须先解决）

| 类别 | 位置 | 当前现象 | 风险 |
|---|---|---|---|
| Tensor canonicalization 重复 | `train/models.py:2579`, `train/models.py:2603`, `train/models.py:2942`, `train/models.py:2966`, `train/models.py:3501`, `train/models.py:3560` | 同一类 `(B,T,C)` / `(B,C)` pad-expand-trim 逻辑散落多处 | event-clock on/off、plan/meas override 行为容易静默漂移 |
| Phase state 初始化重复 | `train/models.py:2645`, `train/models.py:3000` | `phase_z` 规范化、anchor fallback、`phase_event_age` 初始化骨架重复 | 一侧修 bug 后另一侧漏改，导致 phase reset 语义分叉 |
| Contact-plan loop 骨架重复 | `train/models.py:2777`, `train/models.py:3097` | 两段 `for _t in range(Tq)` 共享大段状态推进与缓存逻辑，只在 event-clock 路径多 gate/corrector | 修复/新增状态时必须双改，最易引入状态推进不一致 |
| direct-pose trunk/head 重复 | `train/models.py:1481`, `train/models.py:1516`, `train/models.py:1533`, `train/models.py:1556`, `train/models.py:1614`, `train/models.py:1637` | 同构 `Linear-ReLU-Dropout-Linear-ReLU-Dropout-Linear` 与 zero/bias init 反复手写 | 初始化规则、dropout、层宽调整时容易出现分支不一致 |
| rot6d joint count 多源实现 | `train/models.py:133`, `train/models.py:1680`, `train/models.py:1725` | 已有 `_resolve_rot6d_joint_count(...)`，但 `lambda_fusion` 与 `so3_corrector` 仍手写解析 | 未来 layout 规则变动时会出现 source-of-truth 冲突 |

### 2.2 P1（建议随后解决）

| 类别 | 位置 | 当前现象 | 风险 |
|---|---|---|---|
| `_ablate` 近似双实现 | `train/models.py:4102`, `train/models.py:4143` | Layout A/B 只有 phase/contact 切片布局不同，但 ablation 操作模式高度一致 | 后续新增 ablation mode 时容易只补一边 |
| `forward` 体积过大 | `train/models.py:2256` | 单函数约 `2218` 行，局部 helper 不足、状态变量密集 | 很难局部证明“只改了一个职责” |

### 2.3 P2（清理与治理）

| 类别 | 位置 | 当前现象 | 风险 |
|---|---|---|---|
| 静默异常过多 | `train/models.py:513` 之后 | `EventMotionModel` 内 `except Exception: pass` 仍较多 | 重构阶段若 helper 边界不清，容易继续隐藏错误来源 |
| phase/direct/contact 语义混杂 | `train/models.py:2256` 之后 | forward 同时承担 canonicalization、state init、rollout、direct injection、debug cache | 后续任何小改动都需要大范围回归 |

---

## 3) 目标状态（Done 后应满足）

1. `contacts_input`、`meas_logits_prev`、`plan_in/meas_in override` 的 shape 对齐逻辑分别只有一处 source of truth。  
2. `phase_z` / `phase_event_age` 初始化只保留一套入口，event-clock on/off 只决定“是否使用额外修正”，不复制完整初始化。  
3. direct-pose trunk 与 leg/head/gate/shared-gate 的 MLP 搭建统一走公共 builder，last-layer init 语义可复用且可读。  
4. `lambda_fusion_joint_count` 与 `so3_corr_joint_count` 只通过 `_resolve_rot6d_joint_count(...)` 或其单一包装入口解析。  
5. contact-plan per-step loop 只保留一个主 scaffold，event-clock 特有逻辑以 hook/辅助块注入。  
6. checkpoint key、默认参数行为、输出 tensor shape 不变。  

---

## 4) 分阶段执行（每阶段都有输入/输出/验收）

### Phase A — 语义冻结与基线固化（不改逻辑）

模板参考：`docs/templates/changes/change_refactor_phaseA_template.md`  
执行状态：已完成（2026-03-17）  
本次执行报告：`docs/changes/2026-03-16_event_motion_model_refactor_phaseA_report.md`  
热点 raw 清单：`docs/changes/2026-03-16_event_motion_model_refactor_phaseA_key_refs_raw.txt`  
基线输出目录：`debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline`

**完成说明**

- 已实际执行并冻结：
  - `python -m py_compile train/models.py`（通过）
  - `python train/debug_contact_loop.py`（按 roadmap 原文冻结；当前在 repo root 下因 `ModuleNotFoundError: No module named 'train'` 失败）
  - `python -m train.debug_contact_loop`（补充为可复跑 smoke baseline；当前通过）
- 已固化结构指标快照：
  - `train/models.py` LOC = `6177`
  - `EventMotionModel` 长度 = `3961`
  - `EventMotionModel.forward` 长度 = `2130`
  - `EventMotionModel` 内 `except Exception: pass` 计数 = `39`
  - 本轮 hotspot group 数量 = `9`
- 已固化热点 key refs raw 命中统计：`21` 条（文件范围：`train/models.py`）

**要做什么**

- [x] A1. 固化 baseline 命令，至少覆盖：
  - `python -m py_compile train/models.py`
  - `python train/debug_contact_loop.py`
- [x] A2. 产出热点引用清单，覆盖本轮所有重复锚点：
  - `docs/changes/2026-03-16_event_motion_model_refactor_phaseA_key_refs_raw.txt`
- [x] A3. 固化结构指标快照，至少包含：
  - `train/models.py` LOC
  - `EventMotionModel` 长度
  - `EventMotionModel.forward` 长度
  - `EventMotionModel` 内 `except Exception: pass` 计数
  - 本轮 hotspot 数量

**产出物**

- [x] `docs/changes/2026-03-16_event_motion_model_refactor_phaseA_report.md`
- [x] `docs/changes/2026-03-16_event_motion_model_refactor_phaseA_key_refs_raw.txt`
- [x] `debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline/phaseA_metrics_snapshot.json`
- [x] `debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline/phaseA_metrics_snapshot.txt`

**验收**

- [x] baseline 命令与输出路径已固化（含 roadmap 原命令失败现状冻结 + module-form smoke baseline）
- [x] key refs 覆盖本轮核心重复块
- [x] 结构指标口径固定，可用于后续 step 对照

---

### Phase B — 内部公共化/接口补齐（低风险重构）

执行状态：代码已完成（2026-03-17）  
执行报告：`docs/changes/2026-03-16_event_motion_model_refactor_phaseB_report.md`

**完成说明**

- 已新增并复用 tensor canonicalization helper：
  - `contacts_input -> (B,T,C)`
  - `meas_logits_prev -> (B,C)`
  - direct override -> `(B,T,C)`
- 已新增并复用 phase-state init helper：
  - `phase_z` 外部输入规范化
  - anchor fallback
  - `phase_event_age` 默认初始化
- 已将 `lambda_fusion_joint_count` / `so3_corr_joint_count` 统一到 `_resolve_rot6d_joint_count(...)`
- 已完成验证：
  - `python -m py_compile train/models.py`（通过）
  - `python -m train.debug_contact_loop`（通过）
  - targeted smoke：`direct_pose + phase_state + override`，覆盖 `use_event_clock=False/True`（通过）
- 已知说明：
  - `python train/debug_contact_loop.py` 继续保持 Phase A 冻结的脚本路径失败现状；当前在 repo root 下仍因 `ModuleNotFoundError: No module named 'train'` 失败，不属于本轮 refactor 回归

**要做什么**

- [x] B1. 新增 tensor canonicalization helper，统一：
  - `contacts_input -> (B,T,C)`
  - `meas_logits_prev -> (B,C)`
  - `override -> (B,T,C)`
- [x] B2. 新增 phase-state init helper，统一：
  - `phase_z` 外部输入规范化
  - anchor fallback
  - `phase_event_age` 默认初始化
- [x] B3. 将 `lambda_fusion_joint_count` / `so3_corr_joint_count` 改为复用 `_resolve_rot6d_joint_count(...)`

**验收**

- [x] P0 中三类 canonicalization 逻辑在代码中各只剩单一入口
- [ ] `python train/debug_contact_loop.py` 通过（当前仍保持 Phase A 已冻结的脚本路径失败现状）
- [x] `python -m py_compile train/models.py` 通过
- [x] `python -m train.debug_contact_loop` 通过（作为当前可复跑 smoke baseline）
- [x] 同 seed / 同输入下关键输出 shape 不变

---

### Phase C — 主体收敛（forward + direct-pose）

#### C1: direct-pose 初始化收敛

执行状态：代码已完成（2026-03-17）  
执行报告：`docs/changes/2026-03-16_event_motion_model_refactor_phaseC1_report.md`

- [x] 抽出 trunk builder（两层 `Linear-ReLU-Dropout`）
- [x] 抽出通用 head builder（支持输出层、zero-init、bias-init）
- [x] 收敛 `direct_pose_leg_head` / `direct_pose_leg_gate_head`
- [x] 收敛 `direct_pose_leg_head_shared` / `direct_pose_leg_gate_head_shared`

#### C2: `forward` 中 phase/contact scaffold 收敛

执行状态：代码已完成（2026-03-17）  
执行报告：`docs/changes/2026-03-16_event_motion_model_refactor_phaseC2_report.md`

- [x] 抽出统一的 phase/contact precompute block
- [x] 抽出单一 contact-plan per-step scaffold
- [x] 将 event-clock on/off 差异压缩到局部 hook（如 gate/corrector/extra cache）

#### C3: override / direct input 入口收敛

执行状态：代码已完成（2026-03-17）  
执行报告：`docs/changes/2026-03-16_event_motion_model_refactor_phaseC3_report.md`

- [x] `plan_in` override 与 `meas_in` override 统一到单一 helper
- [x] 保持 `concat` / `mode_select` / phase-z replace 路径行为不变

**阶段验收**

- [x] `EventMotionModel.forward` 行数显著下降
- [x] direct-pose 初始化中的同构 MLP 搭建不再重复手写
- [x] `debug_contact_loop` 与 direct-pose 最小 init/forward smoke 通过

---

### Phase D — 清理与防回归

执行状态：代码已完成（2026-03-17）  
执行报告：`docs/changes/2026-03-16_event_motion_model_refactor_phaseD_report.md`

**要做什么**

- [x] D1. 删除旧重复实现，禁止 helper 与旧块双实现并存
- [x] D2. 视 Phase C 收敛情况决定是否统一 `_ablate` 两版实现
- [x] D3. 增加 focused regression check（至少覆盖 event-clock on/off、split/non-split、override path）

**验收**

- [x] 本轮热点重复块数显著下降
- [x] checkpoint 兼容加载路径未破坏
- [x] 文档与 smoke 路径可复跑

---

## 5) 风险与回退

| 风险 | 触发信号 | 回退策略 |
|---|---|---|
| direct-pose builder 收敛后参数名或 shape 漂移 | checkpoint load mismatch / head shape mismatch | 保持 attribute 名称不变；一步一 commit；必要时先只抽 builder 不改调用层级 |
| phase helper 抽取改变默认 fallback | `phase_z_next` / `phase_event_age_next` 与 baseline 不一致 | 在固定 seed + 固定输入下做 old/new 对照；若不一致仅回退当前 step |
| contact-plan 单 loop 收敛引入 event-clock 分支偏差 | `contacts_plan` / `contacts_meas` / `omega_hat` 形状或数值异常 | 先以行为等价为目标，不同时收紧异常处理；保留 hook 边界，避免一次合并太多职责 |
| `_ablate` 过早合并导致 layout A/B 语义被抹平 | 某一 layout 的 ablation mode 缺失或切片错位 | `_ablate` 放到 Phase D，只有在前面主链稳定后才处理 |

---

## 6) 建议提交拆分（Commit Plan）

1. Commit 1: Phase A 基线固化与 key refs 清单  
2. Commit 2: tensor canonicalization / phase init helper 收敛  
3. Commit 3: direct-pose trunk/head builder 收敛 + rot6d joint count 统一  
4. Commit 4: contact-plan loop / override 入口收敛  
5. Commit 5: 删除旧重复块 + 增补 smoke / 文档更新  

---

## 7) 本轮优先级

- **P0（立即）**: `contacts/meas/phase/override` canonicalization helper、direct-pose builder helper、rot6d joint-count 单一入口  
- **P1**: contact-plan 单 loop scaffold、`forward` 结构降噪  
- **P2**: `_ablate` 收敛、异常处理治理  

---

## 8) 与配套文档关系

- 本文件聚焦：`train/models.py` 的重复逻辑收敛路线图。  
- Phase A 模板：`docs/templates/changes/change_refactor_phaseA_template.md`  
- Phase A 执行报告：`docs/changes/2026-03-16_event_motion_model_refactor_phaseA_report.md`
- Phase B 执行报告：`docs/changes/2026-03-16_event_motion_model_refactor_phaseB_report.md`
- 主计划模板：`docs/templates/changes/change_refactor_plan_template.md`  
- 推荐执行顺序：先按本文件完成 Phase A/B，再进入 `forward` 主循环收敛与清理。
