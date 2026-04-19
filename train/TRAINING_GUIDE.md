# 运动生成训练指南（与当前 `train/` 代码对齐）

> 本文档已按当前代码快照整理（`train/training_MPL.py` / `train/models.py` / `train/dataset.py`）。
> 旧版中出现的 `w_cond_yaw`、`w_rot_log`、`w_limb_geo`、`best_model.pt` 等内容不再作为当前主流程。

---

## 1. 项目概述

本项目是一个基于 PyTorch 的角色运动生成/预测系统，核心特性：

- 6D rotation 表示（训练 loss 内会做几何约束）
- Teacher Forcing + Free-run 渐进式训练
- 支持分阶段调度（`freerun_stage_schedule`）
- 支持 contact plan / contact meas / direct pose 等扩展头
- 支持在线 Teacher / ValFree 指标输出

主训练入口：

- `train/training_MPL.py`

---

## 2. 环境准备

推荐：

- Python 3.10+
- PyTorch 2.x
- CUDA（如使用 GPU）

最小依赖（示例）：

```bash
pip install torch torchvision torchaudio
pip install numpy scipy tqdm
```

---

## 3. 快速开始（推荐流程）

以下命令默认在项目根目录执行（`PythonProject/`）。

### Step 1) JSON 转 NPZ

```bash
python -m train.convert_json_to_npz raw_data/Walk_F.json --out raw_data/processed_data
```

常用可选项：

```bash
python -m train.convert_json_to_npz raw_data --out raw_data/processed_data --merge --export-norm
```

说明：

- soft contact 来自 `Frames[*].FootEvidence.{L,R}.soft_contact_score`
- `yaw` 不作为显式训练特征写入
- NPZ 会保存 `source_json`，训练阶段会复用该源 JSON 的接触标注

### Step 2) 生成/更新训练配置（profile + build）

```bash
python -m tools.config_builder --profile \
  --base-config config/exp_phase_mpl.json \
  --output config/exp_phase_mpl.json
```

只预览不落盘：

```bash
python -m tools.config_builder --profile --dry-run \
  --base-config config/exp_phase_mpl.json \
  --output config/exp_phase_mpl.json
```

### Step 3) 启动训练

```bash
python -m train.training_MPL --config_json config/exp_phase_mpl.json
```

### Step 4) （可选）先做预训练编码器

```bash
python -m train.pretrain_mpl_min \
  --in_glob "raw_data/processed_data/*.npz" \
  --out models/motion_encoder_pretrained.pt \
  --epochs 50 \
  --batch_size 16 \
  --lr 5e-4
```

再在主训练中加载：

```bash
python -m train.training_MPL \
  --config_json config/exp_phase_mpl.json \
  --config_override encoder_path="models/motion_encoder_pretrained.pt"
```

---

## 4. 配置文件关键项（当前版本）

默认配置参考：`config/exp_phase_mpl.json`

### 4.1 必需路径

- `data`: NPZ 目录（含 `*.npz`）
- `bundle_json`: normalizer/template JSON
- `out`: 输出根目录

> 使用 `--config_json` 时，配置里的键必须是训练脚本可识别参数；未知键会直接报错。

### 4.2 基础训练参数

- `epochs`, `batch`, `lr`, `seq_len`
- `grad_clip`, `weight_decay`, `accum_steps`
- `amp`（自动混合精度）

### 4.3 Teacher Forcing 参数

- `tf_mode`: 当前支持 `global` / `epoch_linear`
- `tf_start_epoch`, `tf_end_epoch`, `tf_max`, `tf_min`
- `ss_chunk_len`（scheduled sampling chunk）

### 4.4 调度 / 调试参数

- `freerun_stage_schedule`
- `history_debug_steps`
- `teacher_eval_max_batches`
- `freerun_debug_path`

### 4.5 当前主损失相关参数

- `w_rot_ortho`
- `w_rot_local`
- `w_root_vel`, `w_root_speed`
- （可选）`w_contact_plan`, `w_direct_pose`

注：旧版 `freerun_weight/horizon`、teacher/input noise、`eval_horizon/eval_warmup/monitor_batches`
等 trainbase 配置已移除，不再属于当前 `training_MPL.py` 契约。

### 4.6 模型结构参数

- `width`（hidden dim）
- `depth`（layer depth）
- `num_heads`, `context_len`, `dropout`

> 注意：当前训练脚本主干使用 `width/depth`，不是 `hidden_dim/num_layers`。

### 4.7 数据增强参数（当前）

- `yaw_aug_deg`

---

## 5. 分阶段调度（`freerun_stage_schedule`）

训练脚本支持阶段调度，并在每个 epoch 自动应用当前阶段覆盖项。

可用写法（建议 JSON 结构）：

```json
{
  "freerun_stage_schedule": [
    {
      "range": [1, 4],
      "label": "stage1_warmup_tf1",
      "params": {
        "tf_max": 1.0,
        "tf_min": 1.0,
        "ss_chunk_len": 1,
        "opt_lr": 0.001
      }
    },
    {
      "range": [5, 8],
      "label": "stage2_transition",
      "params": {
        "tf_max": 1.0,
        "tf_min": 0.5,
        "opt_lr": 0.0005
      },
      "loss": {
        "w_rot_local": 0.25
      }
    },
    {
      "range": [9, 18],
      "label": "stage3_stable",
      "params": {
        "tf_max": 0.5,
        "tf_min": 0.5,
        "opt_lr": 0.0002,
        "history_dropout_prob": 0.15
      }
    }
  ]
}
```

说明：

- `range` 可写 `[start, end]`
- `params` 会直接覆盖 Trainer/Loss 可写属性（`opt_lr` 会改 optimizer lr）
- `loss` / `loss_groups` 会映射到 `loss_fn`
- `trainer.freerun_weight/horizon` 与旧 teacher/input noise 分支已经删除；请不要继续写入 schedule

---

## 6. 训练命令与常用覆盖方式

### 6.1 基础训练

```bash
python -m train.training_MPL --config_json config/exp_phase_mpl.json
```

### 6.2 临时覆盖配置（不改 JSON 文件）

```bash
python -m train.training_MPL \
  --config_json config/exp_phase_mpl.json \
  --config_override lr=3e-4 \
  --config_override batch=8 \
  --config_override run_name="exp_tmp_lr3e4"
```

### 6.3 `--resume` 语义（重要）

```bash
python -m train.training_MPL \
  --config_json config/exp_phase_mpl.json \
  --resume models/exp_xxx/ckpt_last_exp_xxx.pth
```

当前行为是：

- 只加载 `model state_dict`（会跳过 shape 不匹配项）
- 不恢复 optimizer/scheduler/epoch 状态

它更接近“初始化权重继续训练”，不是严格的“完整断点续训”。

---

## 7. 输出产物与指标

假设配置里：

- `out = ./models/exp_phase_e2e_sc`
- `run_name = exp_phase_MLP`

则输出目录通常为：

- `models/exp_phase_e2e_sc/exp_phase_MLP/`

### 7.1 Checkpoint

训练过程中会保存：

- `ckpt_best_teacher_<run_name>.pth`
- `ckpt_best_free_<run_name>.pth`
- `ckpt_last_<run_name>.pth`

### 7.2 Metrics

每个 epoch 会写 JSON：

- `metrics/teacher_epXXX.json`
- `metrics/valfree_epXXX.json`

典型字段包含：

- `GeoDeg`, `GeoLocalDeg`, `RootVelMAE`, `AngVelMAE`
- `KeyBone/*`
- `loss_group/core|aux|long`

---

## 8. 评估与诊断

### 8.1 在线评估

当前训练期可通过 `teacher_eval_max_batches` 限制 teacher 评估 batch 数；更细的 freerun
诊断建议使用 `train/validate/` 下的独立脚本。

### 8.2 离线复用评估函数

`train/eval_utils.py` 提供函数接口：

```python
from train import training_MPL
from train.eval_utils import evaluate_teacher, evaluate_freerun, FreeRunSettings

trainer = training_MPL.Trainer(...)
teacher_stats = evaluate_teacher(trainer, val_loader, mode='teacher')
free_stats = evaluate_freerun(
    trainer,
    val_loader,
    settings=FreeRunSettings(warmup_steps=4, horizon=12)
)
print(teacher_stats.get('GeoDeg'), free_stats.get('FreeRun/GeoDeg'))
```

---

## 9. 导出与推理

### 9.1 从 checkpoint 提取模型参数

```python
import torch

ckpt = torch.load('models/exp_xxx/ckpt_last_exp_xxx.pth', map_location='cpu')
state = ckpt.get('model', ckpt)
torch.save(state, 'exported_model_state.pt')
```

### 9.2 重新导出 ONNX（推荐使用现有脚本）

```bash
python -m train.export_onnx_from_ckpt \
  --ckpt models/exp_xxx/ckpt_last_exp_xxx.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --data raw_data/processed_data \
  --out-onnx models/exp_xxx/model_step_stateful_nophase.onnx
```

---

## 10. 常见问题（当前版本）

### Q1. `--config_json` 报“存在未识别字段”

原因：配置里有当前 CLI 不认识的键。

处理：

1. 删除旧字段（如历史版本遗留参数）
2. 对照 `training_MPL.py` 的 `add_argument(...)` 更新键名
3. 重新运行

---

### Q2. Free-run 漂移大（Teacher 好，ValFree 差）

优先调整：

1. 增加后期 `freerun_weight`
2. 适当增大 `freerun_horizon`
3. 放缓 TF 衰减（`tf_end_epoch` 更晚）
4. 检查 `w_rot_local` / `w_rot_vel` 是否过低

---

### Q3. 小数据集容易过拟合

建议：

- 适度增大 `dropout`、`weight_decay`
- 开启轻度 yaw 增强：`yaw_aug_deg`
- 适当减小模型容量（`width`、`depth`）

---

### Q4. 显存不足

可按顺序尝试：

1. 减小 `batch`
2. 减小 `seq_len`
3. 增大 `accum_steps`
4. 开启 `amp`

---

## 11. 代码索引（当前）

| 功能 | 文件路径 |
|------|----------|
| 主训练入口 | `train/training_MPL.py` |
| 主模型与损失 | `train/models.py` |
| 数据集与增强 | `train/dataset.py` |
| 几何工具 | `train/geometry.py` |
| 评估工具 | `train/eval_utils.py` |
| 配置 CLI | `tools/config_builder/cli.py` |
| 数据 profile | `tools/config_builder/profile.py` |
| 阶段构建 | `tools/config_builder/stages.py` |
| JSON→NPZ 转换 | `train/convert_json_to_npz.py` |
| ONNX 导出 | `train/export_onnx_from_ckpt.py` |

---

## 12. 迁移提示（从旧文档配置迁移）

如果你手里有旧配置，建议按下列原则迁移：

1. 先保留最小可跑参数：`data/bundle_json/out/epochs/batch/lr/seq_len`
2. 仅保留当前可识别 loss：`w_rot_ortho/w_rot_local/w_rot_vel/w_root_vel/w_root_speed`
3. 用 `freerun_stage_schedule` 管理阶段，而不是散落脚本参数
4. 先跑通并观察 `metrics/*.json`，再逐步加 contact/direct 等模块

这样可以最快避免“参数名存在但代码未使用”或“配置直接报错”的问题。
