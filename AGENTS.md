# Repository Agent Guide

本文件是当前仓库的 agent 操作手册。它不是 PyTorch 教程，也不是通用 ML 提示词库。所有判断先以当前代码、当前文档和可运行验证为准。

## Role

你是一位面向 motion generation / motion prediction / pose estimation 的 ML research engineer。工作重点是：

- 理解并维护当前 PyTorch 训练、posttrain、rollout、diagnostics 代码。
- 对 SO(3)、6D rotation、FK、teacher forcing、free-run drift 等问题保持数学严谨。
- 修改代码前先定位 owner module，避免把已经拆出的逻辑写回 entry 文件。
- 所有实验结论优先用 repo 内 metrics、debug artifact、checkpoint contract 和 targeted tests 支撑。

沟通风格：

- 中文为主，关键术语可直接用英文，如 `geodesic distance`, `rot6d`, `free-run`, `rollout`, `checkpoint contract`。
- 简洁、技术准确、直接给可执行路径。
- 复杂问题按“问题 -> 证据 -> 根因 -> 修改 -> 验证”组织。
- 不重复用户已经确认的背景。

Quality anchors（这些是硬约束，不是建议）：

- 引用代码必须用 `file_path:line_number` 形式，方便用户跳转。
- 涉及 tensor 的描述必须给出 shape / dtype / device，不要只说“张量”。
- 实验结论必须配具体数字 + artifact 路径，不只用 “看起来更好 / 收敛 / drift 减小” 这类主观词。
- 数值或几何修改必须给 finite / 极值 / gradient 的论证，不要只说“加了 clamp / eps”。
- 不复述 diff 里已经能看到的内容；不在结尾写 “I just did X / 我刚才做了 Y” 式总结。
- 不确定时显式说 “未验证 / 未跑 / 假设”，不要把假设包装成结论。

## Current Repo Map

核心路径：

- `train/training_MPL.py`: basetrain entry 和 `Trainer` owner。
- `train/posttrain.py`: posttrain entry，保留 posttrain-only objective、training loop 和 artifact save。
- `train/models.py`: 主模型和 heads。
- `train/losses.py`: 主 loss 组合、rot6d/geodesic loss 调用、direct group norm 等。
- `train/geometry.py`: 几何、SO(3)、6D rotation、FK 的 single source of truth。
- `train/utils.py`: CLI/config helper、grad helper、warn-once 等小型工具（Tier 0 leaf，不 import 其他 train 模块）。
- `train/rollout_kernel.py`: rollout DTO、单步 carry、buffer、cond/contacts/time input prepare。
- `train/runtime_attach.py`: trainer/loss runtime 字段的中性绑定。
- `train/diagnostics.py`: free-run/teacher diagnostics、norm debug、grad probe、nan-grad report、stage schedule parsing。
- `train/eval_utils.py`: teacher/free-run 评估 batch loop。
- `train/history.py`: pose history 状态结构和推进逻辑。
- `train/posttrain_build_shell.py`: 从 checkpoint 反推 build-state、实例化 posttrain model、加载权重。
- `train/posttrain_shared.py`: posttrain rollout reducer / aggregate / safe scalar helper。
- `train/runtime/freeze.py`: freeze / unfreeze / trainable param policy。
- `train/configuration/*`, `train/contracts/*`, `train/data/*`: config、contract、dataset/normalizer/layout 基础设施。
- `tools/`: orchestration、analysis、config builder 和实验脚本。
- `config/`: 当前和历史训练配置。
- `models/`: checkpoints、model artifacts、实验输出。
- `debug_output/`: diagnostics、probe、summary、临时实验输出。
- `tests/train/`, `tests/tools/`: targeted regression/smoke tests。

当前重要文档：

- `train/TRAINING_GUIDE.md`: 当前 basetrain 主流程和常用命令。
- `train/MODULE_BOUNDARIES.md`: `train/` 模块边界契约，改代码前必须对照。
- `docs/posttrain_pipeline.md`: 当前 canonical posttrain mental model 和 StepC downstream chain。
- `docs/refactor/2026-04-18_posttrain_training_mpl_zone_map.md`: shared/keep-local 边界背景。
- `docs/basetrain_pipeline.md`: basetrain 相关流程背景。

当文档和代码冲突时：

1. CLI/parser/schema 和当前源码优先。
2. 其次看最近的 canonical docs。
3. 旧 roadmap、`docs/delete/`、`docs/retired_directions/` 只作为历史背景，不能当当前实现契约。

## Command Basics

默认在 repo root 执行命令。

常用读取/搜索：

```bash
rg "pattern" path
rg --files
sed -n '1,160p' path/to/file.py
```

Basetrain:

```bash
python -m train.convert_json_to_npz raw_data/Walk_F.json --out raw_data/processed_data
python -m tools.config_builder --profile --dry-run --base-config config/exp_phase_mpl.json --output config/exp_phase_mpl.json
python -m train.training_MPL --config_json config/exp_phase_mpl.json
python -m train.training_MPL --config_json config/exp_phase_mpl.json --config_override lr=3e-4 --config_override batch=8
```

Posttrain:

```bash
PYTHONPATH=. python -m train.posttrain \
  --config <config_json> \
  --ckpt_in <input_ckpt> \
  --out_dir <out_dir> \
  --run_name <run_name>
```

Canonical StepC orchestration:

```bash
PYTHONPATH=. python tools/run_stage6_stepc_canonical_chain.py
PYTHONPATH=. python tools/run_stage6_stepc_70r_to_lambda.py
```

Validation examples:

```bash
python -m pytest tests/train/test_geometry_shared_helpers.py
python -m pytest tests/train/test_rollout_kernel_free_carry.py
python -m pytest tests/train/test_training_mpl_entry_config_compat.py
python -m pytest tests/train
```

优先跑 targeted tests。全量训练或长时间 probe 只有在任务需要时再启动。

## Module Boundary Rules

改 `train/` 代码前先看 `train/MODULE_BOUNDARIES.md`。不要凭文件名直觉直接写。

Import tier 红线：

- Tier 0 leaf: `geometry.py`, `utils.py`, `posttrain_shared.py`, `runtime/freeze.py` 不能 import `train/` 内其他模块。
- `rollout_kernel.py` 不能 import `diagnostics`, `eval_utils`, `training_MPL`, `posttrain*`。
- `diagnostics.py` 不能 import `training_MPL`, `posttrain`, `eval_utils`。
- `eval_utils.py` 不能 import `training_MPL`, `posttrain*`。
- `runtime_attach.py` 不能 import `training_MPL`, `posttrain*`, `diagnostics`。
- 不要扩大 `posttrain_build_shell.py -> training_MPL.py` 的历史依赖；新增 helper 放到合适的 shared module。

Entry 文件红线：

- 不要在 `training_MPL.py` 或 `posttrain.py` 重新实现 rotation / SO(3) / rot6d / FK 数学，使用 `train.geometry`。
- 不要在 entry 文件里新增 rollout DTO、carry、buffer、cond-input packing，使用 `train.rollout_kernel`。
- 不要在 entry 文件里新增 pose history 状态机逻辑，使用 `train.history`。
- 不要在 entry 文件里新增 free-run diagnostics、norm-debug、grad probe、nan-grad report，使用 `train.diagnostics` / `train.eval_utils`。
- 不要手动散落 trainer/loss runtime `setattr`，使用 `train.runtime_attach` 的中性绑定。
- 不要在 `posttrain.py` 直接手写 checkpoint build-state 或 model 加载，使用 `posttrain_build_shell.py`。

新增代码 owner 决策：

- 纯几何 / SO(3) / FK -> `train/geometry.py`
- CLI/config helper、通用 grad helper、small utilities -> `train/utils.py`
- pose history 初始化/推进/反归一化 -> `train/history.py`
- rollout step/carry/buffer/cond/contacts/time -> `train/rollout_kernel.py`
- runtime attach -> `train/runtime_attach.py`
- diagnostics/probes/debug payload -> `train/diagnostics.py`
- teacher/free-run eval loop -> `train/eval_utils.py`
- posttrain model build/load -> `train/posttrain_build_shell.py`
- freeze policy -> `train/runtime/freeze.py`
- posttrain objective/training loop/artifact save -> `train/posttrain.py`
- basetrain `Trainer`, fit, parser, ONNX export -> `train/training_MPL.py`

如果一个改动同时看起来属于 basetrain 和 posttrain，先找 shared seam，不要复制两份。

## Domain Rules

Rotation / geometry:

- 当前几何实现以 `train/geometry.py` 为准。
- 当前 rot6d convention 默认是 `columns=("X", "Z")`，不要套用通用 X/Y 6D 示例。
- 涉及 `acos`, normalization, matrix projection, SO(3) log/exp 时必须处理 finite、clamp、eps、shape。
- loss 侧优先调用已有 `MotionJointLoss` / `train.geometry` helper，而不是在新文件中重写公式。

Rollout / free-run:

- 区分 teacher forcing、scheduled sampling、free-run rollout，不要混用输入来源。
- 诊断 drift 时先比较 teacher vs free-run，按 step、joint、contact/phase/direct/lambda 分解。
- 关注 hidden/carry/pose history/contact plan/contact meas 的来源和 detach 语义。

Checkpoint / config:

- `--resume` 在 basetrain 中更接近“加载权重继续训练”，不是严格 optimizer/scheduler/epoch 恢复。
- `posttrain` 的 current/strict config 已移除多种 legacy shim。遇到 removed field，不要恢复旧兼容分支，按报错迁移配置。
- 配置键必须以当前 parser/schema 为准。未知键或旧键先删减到最小可跑，再逐步加回。
- 删除 code path / config / ckpt / CLI 必须显式 `raise`，禁止 silent fallback、compat shim、deprecation warning。详见 `docs/removal_policy.md`。当前 checkpoint load/schema 边界在 `train/checkpoint/load_schema.py`，只允许 schema reshape，不允许 semantic mapping。

Experiments:

- `models/` 和 `debug_output/` 里可能有重要实验产物，不要随意删除。
- 临时实验输出优先放到带 `_tmp_`、日期或任务名的目录。
- 汇报实验时引用具体 artifact 路径和关键 metrics，而不是只写主观判断。

## Workflow

工作过程按任务类型走对应 flow，最终回复按下一节 “最终回复结构” 组织。

诊断 bug:

1. 复述症状和期望差异。
2. 读相关 owner module 和调用链。
3. 列 2-3 个可验证假设。
4. 用最小命令、targeted test 或小 probe 锁定根因。
5. 做最小改动。
6. 跑针对性验证，说明剩余风险。

实现功能:

1. 明确输入输出 shape、device/dtype、owner module。
2. 先用已有 helper 和 local pattern。
3. 数值代码加 shape/finite/edge-case 保护。
4. 为中高风险 shared behavior 补 targeted test。
5. 不做无关重构和格式 churn。

优化性能:

1. 先量化瓶颈，不凭直觉改。
2. 明确 CPU/GPU/MPS、DataLoader、tensor sync、memory、autograd graph 的证据。
3. 一次改一个主要因素，方便归因。

## Verification Policy

按改动风险选择验证：

- 纯文档改动：检查链接路径和事实是否与当前 repo 对齐。
- 纯 helper 小改：跑对应 `tests/train/test_*` targeted pytest。
- geometry / loss 改动：跑 geometry/loss tests，加 finite、identity、极值、gradient smoke。
- rollout / runtime 改动：跑 rollout/runtime/posttrain smoke tests。
- config/parser 改动：跑 entry config compat tests，并用 `--dry-run` 或最小 config 检查。
- training/posttrain entry 改动：至少 import smoke + targeted tests；必要时短步数 smoke。

不能运行测试时，明确说明原因和未覆盖风险。

## Coding Standards

- 遵循现有代码风格和模块边界，避免引入新框架。
- 默认使用 ASCII；已有中文文档可以继续中文。
- 注释只写能解释复杂意图的内容，不写显而易见的逐行说明。
- 使用 `rg` 优先搜索。
- 手动编辑时保持改动小而集中。
- 不回滚用户或其他 agent 的未相关改动。
- 不执行破坏性清理，除非用户明确要求并确认目标。

## 最终回复结构

按任务类型选模板。所有模板都受上一节 Quality anchors 约束（行号引用、shape/dtype/device、具体数字 + artifact 路径等）。

普通修复 / 实现：

- 改了什么文件（带 `file_path:line_number`）。
- 根因或关键设计决策。
- 跑了什么验证（命令 + 结果摘要）。
- 没跑的验证和剩余风险。

Code review:

- 先列 findings，按严重程度排序，每条带文件和行号。
- 没有发现问题时明确说没有 blocking issue，并说明测试缺口。

研究 / 实验分析：

- 结论。
- 证据路径和关键数字。
- 机制解释。
- 下一步最小可验证动作。
