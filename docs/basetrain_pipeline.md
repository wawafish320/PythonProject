# Basetrain Pipeline (Global Canonical / 设计思路)

> Last updated: 2026-04-21
> Status: current global canonical basetrain design
> Scope: 解释当前 `train/training_MPL.py` 这条 basetrain 入口在干什么、由哪些模块协作、与 posttrain 的边界在哪里
> 不包含：训练命令手册（见 `train/TRAINING_GUIDE.md`）、posttrain canonical chain（见 `docs/posttrain_pipeline.md`）、代码归属红线（见 `train/MODULE_BOUNDARIES.md`）

---

## 1) TL;DR

basetrain 这一层负责：

- 从 raw data 训出一个**可被 posttrain 消费的 base ckpt**——即包含 `inc / direct` 双专家 + `λ fusion` 头的 `EventMotionModel`
- 锁定 **contacts source contract**：basetrain 生成的 ckpt 必须能在 `pretrain_contact` source 下做 free-run，不依赖任何 deploy-time 外部 contact 信号
- 把 **runtime contract**（norm spec / pose history / contact-pretrain hydration）写入 ckpt 与 bundle，使得 posttrain 入口能用 `posttrain_build_shell` 反推出一致的 model + runtime

basetrain 的正常出口是：**Stage6-StepC handoff**（一个 StepC-compatible 的 base ckpt + bundle 组合）。
posttrain 的第一站 `70a` 直接消费这个 handoff。

basetrain **不**做：

- 不做 lambda 头的 posttrain finetune（那在 `train.posttrain --train_lambda_head=true`）
- 不做 direct-pose 的 leg residual / leg gate 后训（那在 `train.posttrain --train_direct_pose / direct_pose_leg_*`）
- 不做 checkpoint-derived rebuild orchestration（那在 `train/posttrain_build_shell.py`）

---

## 2) Source of Truth

### 2.1 入口

- 主入口：`train/training_MPL.py`
- 顶层函数：`main()` → `train_entry()`
- CLI 模板：`train/TRAINING_GUIDE.md` §3

### 2.2 编排骨架

`train_entry()` 是稳定的 6 步线性编排，**不要**在 entry 文件之外重写它：

```
_build_train_components(argv)           # parser / defaults / TrainEntryContext
  → _build_train_loaders(ctx)           # norm_spec + dataset + loader + DatasetRuntimeArtifacts
  → _build_train_model(ctx, data)       # EventMotionModel + encoder bundle attach
  → _prepare_train_model_runtime(...)   # history runtime / contacts pretrain runtime
  → _build_train_loss_and_trainer(...)  # MotionJointLoss + Trainer
  → _attach_train_entry_runtime(...)    # runtime_attach.* 把中性字段绑到 Trainer / loss
  → trainer.fit(...)                    # epoch loop / stage schedule / validation / ckpt
  → _run_postfit_actions(...)           # ONNX export 等 postfit
```

### 2.3 关联模块（详细归属契约见 `train/MODULE_BOUNDARIES.md`）

| 角色 | 模块 | 说明 |
|---|---|---|
| basetrain entry + Trainer | `train/training_MPL.py` | parser / `train_entry()` / `Trainer` class / fit / ONNX export |
| 模型本体 | `train/models.py` | `EventMotionModel`、各个 head、leg-bone 默认表 |
| Loss | `train/models.py` 中 `MotionJointLoss` | 不在 basetrain 里 inline 任何 loss 数学 |
| Rollout 低层 state | `train/rollout_kernel.py` | DTOs / cond-contact-time prepare / carry / buffer |
| Pose history 状态 | `train/history.py` | `PoseHistState` + advance / `AdaptiveHistoryModule` |
| Trainer/loss runtime 绑定 | `train/runtime_attach.py` | 中性字段 attach；caller-prefix 在 entry 文件 |
| 几何数学 | `train/geometry.py` | rotation / SO(3) / FK 唯一来源 |
| 在线诊断 | `train/diagnostics.py` | free-run / norm-debug / grad probe / nan-grad |
| Teacher / free-run 评估 | `train/eval_utils.py` | `evaluate_teacher` / `evaluate_freerun` |
| 配置 / 归一 | `train/configuration/norm_spec.py` | `merge_norm_spec` / `ContactPretrainRuntime` |
| 数据 | `train/data/dataset.py` | `MotionEventDataset` / `DatasetRuntimeArtifacts` |

---

## 3) basetrain 当前 mental model

`train/training_MPL.py` 不是"一段大训练脚本"，更准确地说它是 **basetrain entry + 共享 `Trainer` runtime substrate**。

- entry shell（parser / defaults / build / postfit）服务的是 basetrain CLI policy。
- `Trainer` class 同时被 basetrain 用来跑 `fit()`，也被 posttrain 用来承载 rollout / runtime / loss 调用契约。
- `_lambda_*` family 不在这里——它是 posttrain-only objective，住在 `train/posttrain.py`。

posttrain 入口（`train.posttrain`）会反过来 **import `Trainer`** 并直接消费 basetrain 暴露的 runtime attr / 行为；但 posttrain **不**复用 `Trainer.fit()`、`evaluate_teacher` 调度、ONNX export 这些 entry-shell。

---

## 4) Core 模块（必须长期稳定）

下面列出当前默认 base lane 上**必须存在**的核心模块。删除或 silently bypass 任何一项都视为破坏 basetrain canonical contract。

| 模块 | 职责 | 不可被 silently 替换的原因 |
|---|---|---|
| `inc` 专家 | 增量预测（基于上一帧 + delta） | downstream `λ fusion` 的两个输入之一 |
| `direct` 专家 | 直接姿态预测（cond + plan(+meas) → 绝对 pose） | downstream `λ fusion` 的另一个输入；也是 posttrain `_lambda_*` policy 的工作面 |
| `λ fusion` 头 | 学到何时信任 `inc` vs `direct` | basetrain 训出的 init lambda 是 posttrain `train_lambda_head` 的起点 |
| `pretrain_contact` runtime | basetrain rollout 时的 contact source | 锁死 contact contract，避免 train/infer 不一致 |
| `pose_hist` 显式缓冲 | 历史窗口 → 模型输入 | 与 ONNX 单步推理对齐，保证 train/deploy 同源 |
| stage schedule (`freerun_stage_schedule`) | TF → Mixed → Free-run 渐进 | 决定 `teacher_forcing_ratio` / `lr` / history dropout 等 per-stage override |

---

## 5) Patch 模块（实验增量层，可插拔）

下列模块属于 patch lane，**默认可关、不参与 base canonical**：

- `event_clock`（contact_plan 残差校正）——auto/on/off，basetrain 默认不强制启用
- learned `contact_meas` 头 —— `whitebox` runtime 已在 2026-03-09 退休，learned 头作为研究 patch 保留

启用任一 patch 模块都属于偏离 default base lane；要走 `docs/trainbase_design/2026-03-02_trainbase_v2_core_patch_flow.md` 描述的 patch lane 决策。

---

## 6) Stage Schedule 心智模型

`freerun_stage_schedule` 是 basetrain 的核心训练调度。`Trainer._apply_stage_schedule(epoch)` 在每个 epoch 入口按 schedule 应用：

- `teacher_forcing_ratio`（TF→Mixed→Free-run 的连续过渡）
- `optimizer_lr`（按阶段切 lr）
- `history_dropout_prob`（防止过度依赖 pose history）
- `direct_pose_trunk_trainable`（在某些阶段冻结 direct pose trunk）

**不要**在 `_apply_stage_schedule` 之外手工写 stage 切换；新增 stage knob 必须走 schedule entry，而不是在 `Trainer.fit` 里 inline 判断 epoch。

退休的 stage 字段：见 `Trainer.__init__` 中的 `_assert_no_removed_trainbase_stage_keys` —— 这是 fail-fast 红线，任何回流都会在启动时被拒绝。

---

## 7) basetrain → posttrain 的 boundary contract

basetrain 的 ckpt + bundle 必须满足以下契约，posttrain 入口才能稳定消费：

1. **state_dict shape**：包含 `inc / direct / lambda_fusion_head` 的全部权重；`event_clock_*`、`contact_meas_*` 等 patch 权重要么不存在，要么 posttrain 端走 `auto/on/off` 兼容。
2. **runtime metadata**：`norm_template_path`、`bundle_json_path`、`contacts_pretrain_*` 字段已通过 `runtime_attach.apply_*_runtime` 写入并落盘。
3. **pose_hist contract**：`pose_hist_len / pose_hist_dim / pose_hist_scales / pose_hist_mu / pose_hist_std` 与 dataset 一致——posttrain `posttrain_build_shell` 按这套字段反推 build state。
4. **contacts source = `pretrain_contact`**：与 `docs/posttrain_pipeline.md` §3.2 锁定的 runtime contract 一致；clamp 与 affine_stats 见同节。
5. **StepC compatibility**：basetrain 的 Stage6 出口必须是 StepC-compatible，即 `70a` 在 load 时能直接 absorb，不需要额外 partial_load + tensor_upgrade。

posttrain 怎么消费这个 handoff、怎么续训 `70a → replace → 70R → 71 → 72 → lambda`，见 **`docs/posttrain_pipeline.md`**——本文档不重复。

---

## 8) 不要做的事

- 不要在 `train/training_MPL.py` 里写 rotation / SO(3) 数学（→ `train/geometry.py`）
- 不要在 `Trainer` 内部写 free-run 诊断本体或 grad probe（→ `train/diagnostics.py`）
- 不要在 `Trainer` 内部新增 rollout DTO / 低层 carry helper（→ `train/rollout_kernel.py`）
- 不要在 entry 里手动 setattr 中性 runtime 字段（→ `train/runtime_attach.py`）
- 不要在 basetrain entry 里 inline 解析 ckpt 反推 build state（→ `train/posttrain_build_shell.py`，且这是 posttrain-only 路径）
- 不要把 `_lambda_*` family 移回 basetrain（policy/objective 必须留在 posttrain 入口）
- 不要在不修改 `freerun_stage_schedule` 的前提下手工调 `teacher_forcing_ratio` / `lr` / `history_dropout_prob`

---

## 9) Caveats

- `Trainer` class 体量仍然很大（约 1900 行），是已知结构债——拆分时序见 `docs/refactor/2026-04-18_posttrain_training_mpl_zone_map.md`，本文不重复。
- `train_architecture_overview.md`（2025-12-25）是早期总览，**不是当前 canonical**——其中提到的 `dataset.py` 平铺、`train_configurator.py`、缺失的 `rollout_kernel` / `runtime_attach` 等都已经变化。如果两份文档冲突，**以本文档为准**。
- patch lane（`event_clock` / learned `contact_meas`）的 mainline 决策见 `docs/trainbase_design/2026-03-02_trainbase_v2_core_patch_flow.md`，那是 v2 迁移期决策记录，**不是当前架构总览**。

---

## 10) 关联文档

| 文档 | 角色 | 与本文的关系 |
|---|---|---|
| `train/MODULE_BOUNDARIES.md` | 代码归属红线 | 写代码前自检 |
| `train/TRAINING_GUIDE.md` | 训练命令手册 | 怎么跑 |
| `docs/posttrain_pipeline.md` | posttrain 当前 canonical | basetrain handoff 之后的下游 |
| `docs/refactor/2026-04-18_posttrain_training_mpl_zone_map.md` | 拆分时序 / shared seam | 历史依据 |
| `docs/trainbase_design/2026-03-02_trainbase_v2_core_patch_flow.md` | v2 Core/Patch 迁移决策 | 历史决策记录，非当前架构 |
| `docs/train_architecture_overview.md` | 早期系统总览（2025-12-25） | 已过时，仅历史参考 |
