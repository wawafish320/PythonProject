# `train/posttrain.py` 单文件模块化整理说明

## 1. 目标

本轮只整理 `train/posttrain.py` 的单文件内部结构。
目标是让文件内部模块边界更清楚，方便后续再决定是否跨脚本抽公共逻辑。

## 2. 非目标

- 不跨脚本抽公共函数
- 不新建公共 util/module
- 不修改 `train/models.py` / `train/training_MPL.py`
- 不改变 CLI 参数名
- 不改变 config key
- 不改变 checkpoint 读写格式
- 不改变训练语义或实验行为
- 不做“过细拆分”的 helper 化

## 3. 单文件内部目标模块

### A. Config Contract

负责：

- dataclass
- `_as_*`
- `_cfg_*`
- retired/reject/alias 逻辑
- payload -> config

### B. Rollout Kernel

负责：

- `_rollout_*`
- `_lambda_rollout_*`
- rollout context / unroll / fusion / loss

### C. Train Runtime

负责：

- seed
- batch iterator
- train mode
- freeze/unfreeze
- training loop

### D. Build & Checkpoint

负责：

- dataset/trainer/model build
- checkpoint load
- checkpoint compat/adapt/drop
- output save

### E. CLI Entry

负责：

- argparse
- main

## 4. 本轮约束

- 只在 `train/posttrain.py` 内重排/重构
- 可以新增单文件内 dataclass / small helper
- 不允许把逻辑提前抽到别的脚本
- 不允许顺手清理无关实验逻辑
- 优先做“结构澄清”，不是“功能扩展”

## 5. Phase 划分

### Phase 1

- 把 `_build_posttrain_model_from_ckpt(...)` 的多返回值改成单文件内 dataclass
- 更新调用点，保持外部行为不变

### Phase 2

- 将 checkpoint compat/drop/adapt 逻辑收拢成连续区块
- 只做文件内重排与命名改善

### Phase 3

- 收拢 config parsing / reject / alias 逻辑
- 让 config contract 区块连续

### Phase 4

- 最后再整理 runtime / rollout 区块顺序
- 不改核心算法语义

## 6. 验收标准

- 外部 CLI 保持兼容
- 现有 wrapper 调用不需要改
- ckpt 保存字段不变
- 代码结构比之前更容易继续拆分
- 没有引入跨脚本依赖
