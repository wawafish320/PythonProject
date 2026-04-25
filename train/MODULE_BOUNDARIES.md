# `train/` 模块边界契约

> **这是一份开发者契约，不是重构计划。**
> 阅读对象：任何要往 `train/` 下加新代码或改 `posttrain.py` / `training_MPL.py` 的人。
> 目的：防止已经拆出去的逻辑（geometry / diagnostics / history / rollout_kernel / runtime_attach / posttrain_shared / posttrain_build_shell / eval_utils）回流到两个 entry 文件。
>
> 拆分时序与历史依据见 `docs/refactor/2026-04-18_posttrain_training_mpl_zone_map.md`，本文件不重复。

---

## 1. import 拓扑（自下而上）

下层模块**不允许**反向 import 上层模块。新代码如果迫使你违反这个方向，几乎一定是放错了文件。

| Tier | 模块 | 允许 import 的 `train/` 子模块 |
|---|---|---|
| 0（叶子） | `geometry.py`, `utils.py`, `posttrain_shared.py`, `runtime/freeze.py` | 无（仅 torch / numpy / std） |
| 1 | `history.py`, `configuration/*`, `contracts/*` | `data.normalizers`（仅 history） |
| 2 | `rollout_kernel.py`, `runtime_attach.py` | Tier 0/1 + `data.dataset` / `data.normalizers` |
| 3 | `diagnostics.py`, `eval_utils.py` | Tier 0/1/2 + `models`, `data.io` |
| 4 | `training_MPL.py`, `posttrain_build_shell.py` | Tier 0–3 + `checkpoint.*`, `models` |
| 5（entry） | `posttrain.py` | Tier 0–4 |

**例外**：`posttrain_build_shell.py` 当前从 `training_MPL.py` 引入 `validate_and_fix_model_`。这是历史 re-export，**不要扩大**——新增 helper 走 `train/utils.py`。

---

## 2. 各支持模块的归属契约

每个表格三栏：**属于这里** / **不属于这里（写到哪）** / **import 方向**。

### 2.1 `train/geometry.py`

| 属于这里 | 不属于这里（写到哪） |
|---|---|
| 6D rotation ↔ matrix 转换、Gram-Schmidt、reproject | trainer attr 读取（→ caller） |
| SO(3) exp/log、geodesic、axis-angle | 评估指标聚合（→ `diagnostics.py`） |
| delta 旋转合成 / blend / leg-correction | per-step rollout 状态（→ `rollout_kernel.py`） |
| root-yaw / root-relative / parent-relative 矩阵 | normalizer 反归一化（→ `data.normalizers`） |
| `wrap_to_pi_*`、FK position、numpy 镜像版本 | bone-name / joint-spec 解析（→ `utils.py`） |

**Import 方向**：`geometry.py` 只能 import torch/numpy/math，**不准** import `train/` 内任何模块。

### 2.2 `train/diagnostics.py`

| 属于这里 | 不属于这里（写到哪） |
|---|---|
| free-run / teacher 在线诊断 (`diagnose_free_run`, `history_drift_debug`) | rotation 数学（→ `geometry.py`） |
| norm-debug / dataset-index 优化 (`_norm_debug_once`, `_maybe_optimize_dataset_index`) | 评估循环本身（→ `eval_utils.py`） |
| grad probe / nan-grad report (`test_gradient_connection`, `dump_nan_grad_report`) | 训练 loop 控制流（→ `posttrain.py` / `Trainer.fit`） |
| stage schedule 解析 (`_parse_stage_schedule`) | runtime attr 注入（→ `runtime_attach.py`） |
| direct-pose grad stats、free-run debug payload 写盘 | rollout 低层 carry（→ `rollout_kernel.py`） |

**Import 方向**：可以 import `geometry` / `rollout_kernel` / `utils` / `models` / `data.io`；**不准** import `training_MPL` 或 `posttrain` 或 `eval_utils`。

### 2.3 `train/history.py`

| 属于这里 | 不属于这里（写到哪） |
|---|---|
| `PoseHistState` 数据结构与 in-place advance | rollout step 编排（→ `rollout_kernel.py`） |
| `init_pose_hist_state` / `resolve_pose_hist_input` / `advance_pose_hist_state*` | trainer attr 持有（→ `Trainer` + `runtime_attach.py`） |
| `AdaptiveHistoryModule`（nn.Module） | rotation 数学（→ `geometry.py`） |
| `attach_adaptive_history_runtime` / `resolve_pose_hist_runtime_tensors` | 模型构建（→ `models.py` / `posttrain_build_shell.py`） |

**Import 方向**：只能 import `data.normalizers`。

### 2.4 `train/rollout_kernel.py`

| 属于这里 | 不属于这里（写到哪） |
|---|---|
| Rollout DTOs：`RolloutPredictionBuffers` / `RolloutSequenceInputs` / `RolloutStepInputs` / `RolloutExecutionState` | 高层 objective / loss 聚合（→ `posttrain.py` 的 `_lambda_*`） |
| Runtime config resolvers：`resolve_rollout_cond_runtime_config` / `resolve_free_carry_runtime_config` / `resolve_pose_hist_runtime_config` | epoch/teacher 调度（→ `Trainer.fit` / `eval_utils.py`） |
| 单步 carry / buffer / cond-pose-time input prepare、`apply_free_carry_raw`、`finalize_rollout_prediction_buffers` | 诊断指标记录（→ `diagnostics.py`） |
| pose history 在 rollout 中的 resolve/advance glue（薄层） | rotation 数学本体（→ `geometry.py`） |

**Import 方向**：可以 import `geometry` / `history` / `data.normalizers`；**不准** import `diagnostics` / `eval_utils` / `training_MPL` / `posttrain*`。

### 2.5 `train/runtime_attach.py`

| 属于这里 | 不属于这里（写到哪） |
|---|---|
| `SharedTrainerRuntime` dataclass + `resolve_*` / `apply_*` 中性 attach | caller-specific 前缀映射（保留在 caller 的 entry 文件里） |
| `apply_contacts_pretrain_runtime`、`apply_loss_runtime_from_trainer` | runtime 字段计算逻辑（→ `configuration/norm_spec.py` / `history.py`） |
| Dataset runtime → trainer 的中性绑定 | CLI flag 解析（→ entry 文件） |

**Import 方向**：可以 import `configuration.norm_spec` / `data.dataset` / `history`；**不准** import `training_MPL` / `posttrain*` / `diagnostics`。

### 2.6 `train/posttrain_shared.py`

| 属于这里 | 不属于这里（写到哪） |
|---|---|
| posttrain rollout 内的 reduce / aggregate / safe-scalar helper（`reduce_optional_term_totals`, `safe_float_scalar`, `summarize_*`） | rotation 数学（→ `geometry.py`） |
| 不依赖 `Trainer` 实例的纯函数 reducer | 触碰 `Trainer` runtime attr（→ `posttrain.py` 本体） |
| | basetrain 也用得到的通用 helper（→ `utils.py`） |

**Import 方向**：只能 import torch / std。**不准** import `train/` 内任何模块。

### 2.7 `train/posttrain_build_shell.py`

| 属于这里 | 不属于这里（写到哪） |
|---|---|
| `PostTrainModelBuildState` / `PostTrainModelArtifacts` | 训练 loop / objective（→ `posttrain.py`） |
| 从 ckpt 反推 build-state、实例化 `EventMotionModel`、apply compat load | dataset / loader 构建（→ `posttrain.py` 的 `_build_dataset_and_loader`） |
| direct-pose / lambda-fusion / event-clock build override 解析 | trainer runtime attach（→ `runtime_attach.py` + `posttrain.py` 本体） |
| selective runtime guard / encoder bundle attach | basetrain resume 路径（→ `training_MPL.py`，不共用入口） |

**Import 方向**：可以 import `checkpoint.compat` / `models`；当前从 `training_MPL` 引入 `validate_and_fix_model_` 是历史 re-export，**不要扩大**。

### 2.8 `train/eval_utils.py`

| 属于这里 | 不属于这里（写到哪） |
|---|---|
| `evaluate_teacher` / `evaluate_freerun` 入口 | per-batch 诊断指标本体（→ `diagnostics.py`，这里只 orchestrate） |
| `FreeRunSettings` 配置 dataclass | rollout 低层 step / carry（→ `rollout_kernel.py`） |
| 评估 batch 循环 + 调用 `diagnose_free_run` / `collect_freerun_step_debug_record` | 训练 epoch 调度（→ `Trainer.fit`） |

**Import 方向**：可以 import `diagnostics` / `geometry` / `rollout_kernel`；**不准** import `training_MPL` / `posttrain*`。

### 2.9 `train/utils.py`

| 属于这里 | 不属于这里（写到哪） |
|---|---|
| CLI / config helper：`cfg_get_*`、`apply_cli_overrides`、`parse_int_set_spec`、`as_path` / `as_bool` / `as_float_list` | rotation 数学（→ `geometry.py`） |
| Grad helper：`grad_list_norm`, `grad_list_cosine`, `grad_norm_of_module` | trainer runtime attr 读取（→ caller） |
| `build_mlp` / `safe_int_scalar` / `warn_once` / `iter_infinite` / `pick_first_present` | model 构造（→ `models.py`） |
| `set_global_args` / `get_global_arg`、`expand_paths_from_specs` | 诊断 / metrics（→ `diagnostics.py`） |
| `_build_pretrain_contact_encoder_input`、`validate_and_fix_model_`、`sanity_check_model_dims` | dataset 加载（→ `data.*`） |

**Import 方向**：只能 import torch/nn/std。**不准** import `train/` 内任何模块。

### 2.10 `train/runtime/freeze.py`

| 属于这里 | 不属于这里（写到哪） |
|---|---|
| `_freeze_all` / `_unfreeze_for_train_mode` / `_select_trainable_params` | train_mode 决策（→ `posttrain.py` 的 `_resolve_train_mode`） |
| 任何只依赖 `nn.Module` + 字符串 mode 的 freeze policy | 模型 head 构造（→ `models.py`） |

**Import 方向**：只能 import torch。

---

## 3. `posttrain.py` 红线清单

`train/posttrain.py` 现在仍是 entry 文件，但**这些类别的代码不允许再写入**——已经有专属归属：

| 类别 | 已经搬到 | 红线 |
|---|---|---|
| 旋转 / SO(3) / 6D 数学 | `geometry.py` | 不准在 posttrain.py 里再实现 `rot6d_to_matrix` / `so3_*` / `geodesic_*` 之类。直接 import。 |
| free-run / teacher 诊断、norm-debug、grad probe、nan-grad report | `diagnostics.py` | 不准在 posttrain.py 里再写 per-step 指标聚合或 grad 体检。 |
| pose history 状态机 | `history.py` | 不准在 posttrain.py 里直接更新 `pose_hist_*` tensor；走 `advance_pose_hist_state*`。 |
| rollout DTO / cond-input prepare / carry / buffer | `rollout_kernel.py` | 不准在 posttrain.py 里再定义新的 rollout dataclass 或重写 cond/contacts 输入打包。 |
| Trainer / loss runtime 字段绑定 | `runtime_attach.py` | 不准在 posttrain.py 里手动 setattr 中性 runtime 字段；走 `apply_*_runtime`。 |
| Checkpoint 反推 build-state / 实例化 model / compat load | `posttrain_build_shell.py` | 不准在 posttrain.py 里直接 `torch.load` + 手工构造 `EventMotionModel` 或 apply 兼容补丁。 |
| reduce / aggregate / safe-scalar helper | `posttrain_shared.py` | 不准在 posttrain.py 里再 inline 写 `safe_float_scalar` / 类似的 reducer。 |
| Free / freeze policy（按 mode 切换 trainable） | `runtime/freeze.py` | 不准在 posttrain.py 里再写新的 `_freeze_all` 变体。 |

`posttrain.py` **可以**继续承接：

- `PostTrainConfig` 与 `_cfg_*` 解析（posttrain 专属 CLI/JSON 表达）
- `_lambda_*` family（lambda-fusion / direct-pose 的 posttrain-only objective）
- `_run_training_loop` 自定义优化器循环、step snapshot、L2-SP、grad probe orchestration
- `_save_posttrain_outputs` artifact 写盘
- `main()` / `_build_posttrain_arg_parser()`

---

## 4. `training_MPL.py` 红线清单

`train/training_MPL.py` 是 basetrain entry + `Trainer` class 宿主。**这些类别的代码不允许再写入**：

| 类别 | 已经搬到 | 红线 |
|---|---|---|
| 旋转 / SO(3) / 6D 数学 | `geometry.py` | 同 posttrain：直接 import，不在 `Trainer` 方法里再实现一遍。 |
| pose history 状态机本体 | `history.py` | `Trainer` 持有 tensor、调用 `advance_*`，但不写 buffer 推进逻辑本体。 |
| Rollout DTO / 低层 step state | `rollout_kernel.py` | 不准在 `Trainer` 内部再定义新的 rollout dataclass；统一走 `rollout_kernel`。 |
| free-run / teacher 诊断指标本体 | `diagnostics.py` / `eval_utils.py` | `Trainer.fit` 调度评估，但不在 `Trainer` 内部实现 free-run debug 本体。 |
| Norm-debug、grad probe、nan-grad report | `diagnostics.py` | 同上。 |
| Trainer runtime 字段中性绑定 | `runtime_attach.py` | `_attach_train_entry_runtime` 调用 `apply_*_runtime`，不再 inline setattr 中性字段。 |
| Stage schedule 解析（`_parse_stage_schedule`） | `diagnostics.py`（已搬） | 不准回流。 |

`training_MPL.py` **可以**继续承接：

- `Trainer` class 本体（`__init__`、`step`、`_run_*`、teacher_forcing 调度、stage advance）
- `Trainer.fit` epoch 管理、validation 调度、ckpt finalize
- basetrain CLI parser / defaults / `train_entry()` shell
- `_build_train_loaders` / `_build_train_model` / `_build_train_loss_and_trainer` 顺序编排
- ONNX export (`_export_postfit_onnx`, `export_onnx_step_stateful_nophase`)
- `TrainerRuntimeConfig` 在本文件内的定义（这是 basetrain owner 的 runtime view）

---

## 5. 写新代码前的决策树

**"我要加一段代码，应该写到哪？"** 按这个顺序问自己；命中第一个 yes 就停。

1. **是不是纯几何 / 旋转 / SO(3) / FK 数学？**（不依赖任何 trainer attr） → `geometry.py`
2. **是不是 CLI / config 解析、grad helper、`build_mlp` 这类通用工具？**（basetrain 和 posttrain 都可能用） → `utils.py`
3. **是不是 pose history 状态本身的初始化 / 推进 / 反归一化？** → `history.py`
4. **是不是 rollout 单步 cond / contacts / time / carry / buffer 的低层结构？**（不含 objective） → `rollout_kernel.py`
5. **是不是 trainer / loss runtime 字段的中性绑定？**（owner-prefix 之外的部分） → `runtime_attach.py`
6. **是不是 free-run / teacher 诊断指标、norm-debug、grad probe、nan-grad report、stage schedule 解析？** → `diagnostics.py`
7. **是不是 teacher / free-run 评估 batch 循环入口？** → `eval_utils.py`
8. **是不是从 checkpoint 反推 build-state、实例化 posttrain model、apply compat load？** → `posttrain_build_shell.py`
9. **是不是 freeze / unfreeze / 选 trainable param？** → `runtime/freeze.py`
10. **是不是 posttrain rollout 内不依赖 `Trainer` 的 reduce / aggregate / safe-scalar？** → `posttrain_shared.py`
11. **是不是 posttrain 专属的 objective / loss / training loop / artifact save？** → `posttrain.py`（entry 本体）
12. **是不是 basetrain 专属的 `Trainer` 行为 / fit / parser / ONNX export？** → `training_MPL.py`（entry 本体）

如果一段代码同时命中 11 和 12，停下来——它大概率是 shared seam，看 `docs/refactor/2026-04-18_posttrain_training_mpl_zone_map.md` 里 §1.2 的 seam 表，或者考虑加到 `rollout_kernel.py` / `runtime_attach.py`。

---

## 6. 评审 checklist（提 PR 前自检）

- [ ] 新代码没有在 `posttrain.py` / `training_MPL.py` 重新实现 §3 / §4 红线类别。
- [ ] 没有反向 import（Tier N 不 import Tier M>N，见 §1）。
- [ ] 没有在 `geometry.py` / `utils.py` / `posttrain_shared.py` 引入 `train/` 内部 import。
- [ ] 没有在 `posttrain_build_shell.py` 扩大对 `training_MPL` 的依赖（除现有 `validate_and_fix_model_`）。
- [ ] 如果是 shared seam，先翻 zone map §1.2 / §1.3 确认是否仍属 keep-local。

---

## 7. 这份文档不做的事

- 不写未完成拆分的下一步（见 zone map Step 5/6）。
- 不写"如何使用"——那是 `TRAINING_GUIDE.md` 的职责。
- 不写训练语义、loss 配方、stage schedule 内容。
- 不引用具体行号（行号会漂；用模块名 + 函数名）。
- 不替代 `docs/refactor/` 下的历史拆分记录。

新加 / 删除 / 重命名一个 `train/` 子模块时，**同步更新本文档的 §1（拓扑）和 §2（归属表）**，否则边界就开始失真。
