# [2026-04-18] Step 1 API draft — `train/runtime_attach.py`

Date: 2026-04-18  
Status: Draft / Step 1 interface proposal  
Related:

- `docs/refactor/2026-04-18_posttrain_training_mpl_zone_map.md`
- `docs/refactor/2026-04-18_posttrain_training_mpl_commonization_execution_plan.md`

---

## 0. 目标

这份草案只回答 Step 1：

> 如何把 `posttrain.py` / `training_MPL.py` 的 **runtime attach seam** 收到一个可执行边界里，
> 同时不改训练语义，不改 checkpoint contract，不引入早期 attr rename churn。

这里的 `runtime_attach` 特指：

- `DatasetRuntimeArtifacts` 已经构建完成之后，
- 如何把 pose-history runtime、yaw/forward-axis override、contact-pretrain hydration、
  basetrain/posttrain 的 trainer runtime overlay 注入到 `Trainer`。

这份草案**不**试图处理：

- rollout kernel 抽取
- checkpoint rebuild shell
- `models.py`
- `Trainer.fit(...)`
- posttrain objective/loss

### 0.1 shared-half 落地备注

截至当前最小薄片实现：

- `train/runtime_attach.py` 已落地 shared-half：
  - `SharedTrainerRuntime`
  - `resolve_shared_trainer_runtime(...)`
  - `apply_shared_trainer_runtime(...)`
- 当前已接入：
  - `train/training_MPL.py`
  - `train/posttrain.py`
- `posttrain` 目前只接 shared-half；posttrain-specific overlay 仍保持在 caller 本地
- basetrain-specific overlay 仍故意保留在 `train/training_MPL.py`

### 0.2 当前边界选择

本轮已明确选择：

- `train/runtime_attach.py` **只承载 shared-half**
- basetrain overlay 继续留在 `train/training_MPL.py` 的本地 adapter
- posttrain overlay 继续留在 `train/posttrain.py` 的 posttrain-local helper

这意味着：

- 本轮**没有**把 `tf_mode`、`freerun_stage_schedule`、`hyperparam_scheduler`、`enable_grad_connection_test`
  这类更接近 basetrain shell / rollout management 的字段抬进 shared module
- 也**没有**把 `lambda_reliability_*` / `contact_meas_*` 这类 posttrain-only policy 配置提升为共享契约

### 0.3 overlay 草案的契约说明

- 下文出现的 `BasetrainRuntimeOverlay` / `PosttrainRuntimeOverlay` 更接近 deferred sketch，不是当前已落地契约
- 即使后续恢复这两个 dataclass，它们的字段组成也不是永久边界；到 zone map Step 4 抽 rollout kernel 时，字段很可能会再次重分配
- 因此不应把这些 overlay dataclass 视为稳定公共 API

### 0.4 已记录验证

Verified on 2026-04-18:

- `python3 -m py_compile train/runtime_attach.py train/training_MPL.py`
- `python3 -m py_compile train/runtime_attach.py train/training_MPL.py train/posttrain.py`
- helper-level attach smoke pass：
  - basetrain 路径可解析 `SharedTrainerRuntime` 并保留 `TrainerRuntimeConfig.shared`
  - posttrain 路径可解析并应用 posttrain-local overlay
  - pose-history shared attach 现已在 basetrain / posttrain 两侧对称接线
- dataset-backed entry-build smoke pass：
  - basetrain fixture：`config/exp_phase_mpl.clean.json` + `raw_data/processed_data/Walk_F.npz` + `models/motion_encoder_equiv_stageA.pt`
  - posttrain fixture：`config/posttrain_direct_pose_walkf.json` + override ckpt `models/MLPL2_DirectBranch_v1_20260317/exp_phase_DirectBranch_v1_d1_20260317/ckpt_best_free_exp_phase_DirectBranch_v1_d1_20260317.pth` + `raw_data/processed_data/Walk_F.npz`
  - stop point：两边都止步于 dataset/model/trainer/runtime attach 完成，不进入训练循环
  - summary artifact：`debug_output/_runtime_attach_entry_smokes_20260418/entry_build_smoke_summary.json`

---

## 1. 先锁边界：谁留在原处，谁归新模块

### 1.1 继续留在原模块

#### `train/data/dataset.py` 继续拥有

- `DatasetRuntimeArtifacts`
- `build_dataset_artifacts(...)`
- `attach_dataset_runtime_to_trainer(...)`
- `build_and_attach_dataset_runtime(...)`

原因：

- 这些 helper 的职责是 **dataset → normalized runtime artifacts / trainer base attach**。
- 它们已经是 dataset-owned API，不应在 Step 1 里再搬一次 owner。

#### `train/configuration/norm_spec.py` 继续拥有

- `ContactPretrainRuntime`
- `resolve_contact_pretrain_runtime(...)`
- `merge_norm_spec(...)`

原因：

- 这些是 config/spec 解析，不是 trainer attach owner。

#### `train/history.py` 继续拥有

- `resolve_pose_hist_runtime_tensors(...)`

原因：

- 这是 pose-history runtime tensor 的生产者，`runtime_attach.py` 只消费它。

### 1.2 `train/runtime_attach.py` 新增后拥有

新增模块 `train/runtime_attach.py` 在当前落地形态下只拥有 **shared-half attach**：

1. 把 `DatasetRuntimeArtifacts` 之外的 shared runtime 补齐到 `Trainer`
   - `pose_hist_scales`
   - `pose_hist_mu`
   - `pose_hist_std`
   - `yaw_forward_axis`
   - `yaw_forward_axis_offset`
   - optional output metadata
2. 通过中性字段映射保持现有 live attr contract
3. caller-specific overlay 继续留在 owner-local helper / adapter 中

---

## 2. 当前代码里已经暴露出来的 Step 1 seam

### 2.1 `training_MPL.py` 当前做了两层 attach

第一层：

- `build_and_attach_dataset_runtime(...)`  
  见 `train/training_MPL.py:3850`

第二层：

- `_resolve_trainer_runtime_config(...)`  
  见 `train/training_MPL.py:3291`
- `_apply_trainer_runtime_config(...)`  
  见 `train/training_MPL.py:3375`

这说明 basetrain 目前已经事实分成：

- dataset-owned attach
- trainer runtime overlay attach

### 2.2 `posttrain.py` 原先只做了“半层 attach”

它做了：

- `build_and_attach_dataset_runtime(...)`  
  见 `train/posttrain.py:3175`
- `resolve_contact_pretrain_runtime(...)` + 一串 trainer attrs 直接赋值  
  见 `train/posttrain.py:3200`

但它**没有**像 basetrain 那样补上：

- `pose_hist_scales`
- `pose_hist_mu`
- `pose_hist_std`
- resolved `yaw_forward_axis`
- resolved `yaw_forward_axis_offset`

这会带来一个重要不对称：

- posttrain 的 `_lambda_rollout_prepare_context(...)` 用 `trainer._pose_hist_params`
  初始化 pose history，见 `train/posttrain.py:1134`
- 而 `Trainer._pose_hist_params(...)` 依赖
  `trainer.pose_hist_scales / pose_hist_mu / pose_hist_std`，见 `train/training_MPL.py:588`

这也是 Step 1 的一个隐藏收益：

> 把 posttrain 也接到同一套 pose-history runtime attach 上，消除当前 basetrain / posttrain 的 attach 不对称。

当前这个不对称已经由 shared-half 接线消除。

---

## 3. Step 1 设计原则

### 3.1 attr 命名策略

采用：

> **保留前缀 + 中性映射层**

即：

- 新模块内部 dataclass / helper 使用中性字段名
- 最终 apply 到 `Trainer` 时，再映射到
  - basetrain: `trainbase_*`
  - posttrain: `posttrain_*`

本阶段**不**重命名 live trainer attrs。

### 3.2 owner split

Step 1 必须避免“双重 owner”：

- dataset.py 已经 attach 的字段，`runtime_attach.py` 不再重复定义 owner
- `runtime_attach.py` 只负责 dataset.py **还没 attach** 的 runtime overlay

### 3.3 public API 不直接吃 `argparse.Namespace`

为了避免把 CLI schema 固化进 shared module：

- `training_MPL.py` 本地负责 `args -> kwargs`
- `posttrain.py` 本地负责 `cfg -> kwargs`
- `train/runtime_attach.py` 的 public API 只接收 typed kwargs / dataclass

---

## 4. 拟议 public API

## 4.1 数据结构

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from train.configuration.norm_spec import ContactPretrainRuntime


@dataclass(frozen=True)
class SharedTrainerRuntime:
    pose_hist_scales: Optional[torch.Tensor]
    pose_hist_mu: Optional[torch.Tensor]
    pose_hist_std: Optional[torch.Tensor]
    yaw_forward_axis: int
    yaw_forward_axis_offset: float

    # Optional metadata: basetrain currently uses these; posttrain may leave them None.
    norm_template_path: Optional[str] = None
    bundle_json_path: Optional[str] = None
    out_dir: Optional[str] = None
    full_config: Optional[Dict[str, Any]] = None
    current_run_name: Optional[str] = None


@dataclass(frozen=True)
class BasetrainRuntimeOverlay:
    shared: SharedTrainerRuntime
    contact_pretrain: ContactPretrainRuntime
    direct_pose_grad_monitor_enable: bool
    direct_pose_grad_ratio_gate: float
    diag_topk: int
    diag_thr: float
    teacher_eval_max_batches: Optional[int]
    ss_chunk_len: int
    tf_mode: str
    tf_start_epoch: int
    tf_end_epoch: int
    tf_max: float
    tf_min: float
    history_debug_steps: int
    history_dropout_prob: float
    history_dropout_prob_min: float
    history_dropout_prob_max: float
    freerun_stage_schedule: list[Any]
    hyperparam_scheduler: Any
    freerun_debug_path: Optional[str]
    enable_grad_connection_test: bool


@dataclass(frozen=True)
class PosttrainRuntimeOverlay:
    shared: SharedTrainerRuntime
    contact_pretrain: ContactPretrainRuntime
    contact_meas_gate_by_hit_override: Optional[bool]
    contact_meas_vxy_mode: str
    contact_meas_ground_z_mode: str
    contact_meas_ground_z_beta: float
    contact_meas_ground_z_window: int
    contact_meas_ground_z_quantile: float
    contact_meas_ground_z_max_up_m: float
    contact_meas_ground_z_max_down_m: float
    lambda_reliability_mode: str
    lambda_reliability_warmup_steps: int
    lambda_reliability_contact_err_max: float
    lambda_reliability_warmup_joint_scales: Any
```

### 4.2 shared resolution

```python
from pathlib import Path
from typing import Any, Dict, Optional

from train.data.dataset import DatasetRuntimeArtifacts


def resolve_shared_trainer_runtime(
    *,
    dataset_artifacts: DatasetRuntimeArtifacts,
    trainer_default_yaw_forward_axis: int,
    yaw_forward_axis_override: Optional[int] = None,
    yaw_forward_offset_deg_override: Optional[float] = None,
    norm_template_path: Optional[Path] = None,
    bundle_json_path: Optional[str] = None,
    out_dir: Optional[Path] = None,
    full_config: Optional[Dict[str, Any]] = None,
    current_run_name: Optional[str] = None,
) -> SharedTrainerRuntime: ...
```

职责：

- 调 `resolve_pose_hist_runtime_tensors(dataset_artifacts.dataset)`
- 统一 forward-axis 解析规则：
  - override 优先
  - 其次 `dataset_artifacts.forward_axis`
  - 否则 fallback 到 caller 给的 trainer default
- 统一 offset 解析规则：
  - degree override 统一转 radians
  - 否则继承 `dataset_artifacts.forward_axis_offset`
- 只返回 **中性 runtime**，不带 `trainbase_*` / `posttrain_*` 前缀

### 4.3 basetrain overlay resolution（deferred sketch，非当前落地 API）

```python
def resolve_basetrain_runtime_overlay(
    *,
    shared: SharedTrainerRuntime,
    contact_pretrain_clamp_raw: Any,
    contact_pretrain_affine_stats_raw: Any,
    direct_pose_grad_monitor_enable: bool,
    direct_pose_grad_ratio_gate: float,
    diag_topk: int,
    diag_thr: float,
    teacher_eval_max_batches: Optional[int],
    ss_chunk_len: int,
    tf_mode: str,
    tf_start_epoch: int,
    tf_end_epoch: int,
    tf_max: float,
    tf_min: float,
    history_debug_steps: int,
    history_dropout_prob: float,
    history_dropout_prob_min: float = 0.05,
    history_dropout_prob_max: float = 0.30,
    freerun_stage_schedule: Optional[list[Any]] = None,
    hyperparam_scheduler: Any = None,
    freerun_debug_path: Optional[str] = None,
    enable_grad_connection_test: bool = True,
    warn_contact_pretrain: bool = True,
    warn_prefix: str = "[MPL]",
) -> BasetrainRuntimeOverlay: ...
```

职责：

- 内部调 `resolve_contact_pretrain_runtime(...)`
- 把 basetrain 目前 `_resolve_trainer_runtime_config(...)` 的逻辑收到 typed overlay
- 不直接写 trainer attrs

### 4.4 posttrain overlay resolution（deferred sketch，非当前落地 API）

```python
def resolve_posttrain_runtime_overlay(
    *,
    shared: SharedTrainerRuntime,
    contact_pretrain_clamp_raw: Any,
    contact_pretrain_affine_stats_raw: Any,
    contact_meas_gate_by_hit_raw: Any,
    contact_meas_vxy_mode: str,
    contact_meas_ground_z_mode: str,
    contact_meas_ground_z_beta: float,
    contact_meas_ground_z_window: int,
    contact_meas_ground_z_quantile: float,
    contact_meas_ground_z_slew_up_cm: float,
    contact_meas_ground_z_slew_down_cm: float,
    lambda_reliability_mode: str,
    lambda_reliability_warmup_steps: int,
    lambda_reliability_contact_err_max: float,
    lambda_reliability_warmup_joint_scales: Any,
    warn_contact_pretrain: bool = False,
    warn_prefix: str = "[posttrain]",
) -> PosttrainRuntimeOverlay: ...
```

职责：

- 内部调 `resolve_contact_pretrain_runtime(...)`
- 统一 posttrain 的 contact-meas 解析
- 统一 `cm -> meter` 换算
- 统一 lambda reliability overlay

### 4.5 apply helpers（仅 `apply_shared_trainer_runtime(...)` 已落地）

```python
def apply_shared_trainer_runtime(
    trainer: Any,
    runtime: SharedTrainerRuntime,
) -> Any: ...


def apply_basetrain_runtime_overlay(
    trainer: Any,
    overlay: BasetrainRuntimeOverlay,
) -> Any: ...


def apply_posttrain_runtime_overlay(
    trainer: Any,
    overlay: PosttrainRuntimeOverlay,
) -> Any: ...
```

#### `apply_shared_trainer_runtime(...)` 只设置这些字段

```python
trainer.pose_hist_scales
trainer.pose_hist_mu
trainer.pose_hist_std
trainer.yaw_forward_axis
trainer.yaw_forward_axis_offset
trainer._norm_template_path
trainer._bundle_json_path
trainer.out_dir
trainer.full_config
trainer._current_run_name
```

注意：

- **不**重复设置 dataset.py 已经 attach 过的字段：
  - `pose_hist_len`
  - `pose_hist_dim`
  - `fps`
  - `bone_hz`
  - `forward_axis`
  - `forward_axis_offset`
  - `mu_x/std_x/mu_y/std_y`
  - layout / slices / normalizer / `_bundle_meta`

#### `apply_basetrain_runtime_overlay(...)` 保留现有 attr contract（deferred）

它内部应先调 `apply_shared_trainer_runtime(...)`，再设置：

```python
trainer.trainbase_contacts_pretrain_clamp
trainer.trainbase_contacts_pretrain_affine
trainer.trainbase_contacts_pretrain_affine_stats_spec
trainer.direct_pose_grad_monitor_enable
trainer.direct_pose_grad_ratio_gate
trainer.diag_topk
trainer.diag_thr
trainer.teacher_eval_max_batches
trainer.ss_chunk_len
trainer.tf_mode
trainer.tf_start_epoch
trainer.tf_end_epoch
trainer.tf_max
trainer.tf_min
trainer.history_debug_steps
trainer.history_dropout_prob
trainer.history_dropout_prob_min
trainer.history_dropout_prob_max
trainer.freerun_stage_schedule
trainer.hyperparam_scheduler
trainer.freerun_debug_path
trainer.enable_grad_connection_test
```

#### `apply_posttrain_runtime_overlay(...)` 也保留现有 attr contract（deferred）

它内部应先调 `apply_shared_trainer_runtime(...)`，再设置：

```python
trainer.posttrain_contacts_pretrain_clamp
trainer.posttrain_contacts_pretrain_affine
trainer.posttrain_contacts_pretrain_affine_stats_spec
trainer.contact_meas_gate_by_hit_override
trainer.contact_meas_vxy_mode
trainer.contact_meas_ground_z_mode
trainer.contact_meas_ground_z_beta
trainer.contact_meas_ground_z_window
trainer.contact_meas_ground_z_quantile
trainer.contact_meas_ground_z_max_up_m
trainer.contact_meas_ground_z_max_down_m
trainer.lambda_reliability_mode
trainer.lambda_reliability_warmup_steps
trainer.lambda_reliability_contact_err_max
trainer.lambda_reliability_warmup_joint_scales
```

---

## 5. caller 侧最小接入方式

## 5.1 `training_MPL.py`

### 当前 owner

- `_resolve_trainer_runtime_config(...)`
- `_apply_trainer_runtime_config(...)`

### 当前已落地做法

保留 `TrainerRuntimeConfig`，只把 shared-half 收进 `runtime_cfg.shared`：

```python
runtime_cfg = _resolve_trainer_runtime_config(...)
apply_shared_trainer_runtime(trainer, runtime_cfg.shared)
# basetrain-specific fields still applied locally
```

### deferred 版本可改成

后续若要继续收敛，可再引入本地 adapter：

```python
def _args_to_basetrain_runtime_kwargs(args: argparse.Namespace, trainer: Trainer) -> dict[str, Any]: ...
```

再决定是否包装为 `BasetrainRuntimeOverlay`；这一步当前并未落地。

## 5.2 `posttrain.py`

### 当前 owner

`_build_model_and_trainer(...)` 里以下块是 Step 1 候选：

- `build_and_attach_dataset_runtime(...)` 之后的 runtime overlay 赋值
- `resolve_contact_pretrain_runtime(...)`
- contact meas config
- lambda reliability config

### 当前已落地做法

```python
dataset_artifacts = build_and_attach_dataset_runtime(...)

shared = resolve_shared_trainer_runtime(...)
apply_shared_trainer_runtime(trainer, shared)

_apply_posttrain_local_runtime_overlay(
    trainer,
    _resolve_posttrain_local_runtime_overlay(cfg),
)
```

### deferred 版本可改成

后续若真要把 posttrain overlay 做成显式 payload，可再考虑 `PosttrainRuntimeOverlay`；
但它应继续是 posttrain-owned，而不是 shared runtime contract。

### Step 1 对 posttrain 的直接收益

- posttrain 终于也会 attach：
  - `pose_hist_scales`
  - `pose_hist_mu`
  - `pose_hist_std`
- 从而让 `trainer._pose_hist_params` 与 basetrain 一致

---

## 6. 不建议塞进 `train/runtime_attach.py` 的内容

以下内容虽然“看起来相关”，但在 Step 1 不应进入新模块：

- `build_and_attach_dataset_runtime(...)` 本体  
  - 这是 dataset owner，不是 runtime overlay owner。
- `merge_norm_spec(...)`
- `resolve_contact_pretrain_runtime(...)`
- `Trainer.fit(...)`
- `evaluate_teacher(...)`
- posttrain `_lambda_fusion_loss_rollout(...)`
- checkpoint build/load compat
- loss_fn bone_names/meta 细节

尤其不要把 `training_MPL.py` 的 parser/defaults 或 `posttrain.py` 的 config parse 搬进来；
否则 Step 1 会立刻从 runtime attach 变成 shell merge。

---

## 7. 精确搬运边界

## 7.1 从 `training_MPL.py` 搬什么

已搬到 `train/runtime_attach.py`：

- `_resolve_trainer_runtime_config(...)` 的核心逻辑
  - 当前只拆出了 `resolve_shared_trainer_runtime(...)`
- `_apply_trainer_runtime_config(...)`
  - 当前只拆出了 `apply_shared_trainer_runtime(...)`

留在 `training_MPL.py`：

- `args -> kwargs` adapter
- `_resolve_freerun_stage_schedule(...)`
- entry shell / parser
- basetrain-specific overlay resolve/apply

## 7.2 从 `posttrain.py` 搬什么

已搬到 `train/runtime_attach.py`：

- `_build_model_and_trainer(...)` 中的 shared-half：
  - pose-history runtime tensors
  - resolved yaw runtime
  - optional metadata

留在 `posttrain.py`：

- `Trainer` / `loss_fn` 构造
- bone_names 处理
- `build_and_attach_dataset_runtime(...)` 调用
- posttrain-local overlay resolve/apply
  - contact-pretrain overlay
  - contact-meas overlay
  - lambda reliability overlay
- posttrain-only objective / training loop

---

## 8. Step 1 验收条件

## 8.1 结构验收

- 新增 `train/runtime_attach.py`
- `training_MPL.py` 已真实调用 `resolve_shared_trainer_runtime(...)` / `apply_shared_trainer_runtime(...)`
- `posttrain.py` 已真实调用 `resolve_shared_trainer_runtime(...)` / `apply_shared_trainer_runtime(...)`
- caller-specific overlay 仍允许保留在 owner-local helper / adapter

## 8.2 行为验收

- basetrain runtime attr 名保持不变：
  - `trainbase_*`
- posttrain runtime attr 名保持不变：
  - `posttrain_*`
- 不改 checkpoint top-level key
- 不改训练 loop 语义

## 8.3 关键 smoke

最低 smoke 应覆盖：

1. basetrain helper path 仍能完成 dataset attach + shared-half attach
2. posttrain helper path 仍能完成 dataset attach + shared-half attach + posttrain-local overlay attach
3. posttrain 在启用 pose history 时，`trainer._pose_hist_params(...)` 不再依赖未初始化的 `pose_hist_scales`

---

## 9. 一句总结

Step 1 的最佳切法不是“把 dataset attach 全搬走”，而是：

> 保留 `dataset.py` 作为 base artifacts owner，新增 `train/runtime_attach.py` 作为
> **shared runtime attach owner**，让 basetrain / posttrain 都先通过同一个中性 shared-half layer
> 注入公共 runtime，再由各自 owner-local overlay 写入前缀化 trainer attrs。
