# AI Motion Model Project

## 项目概述

本项目包含：
- Python 训练代码（基于 PyTorch）
- Unreal Engine C++ 推理代码
- 数据处理和验证工具

## 重要文档

### C++ 和 Python 对接

**⚠️ 必读：** 如果你要修改训练或推理代码，请先阅读：

📄 **[C++ 推理与 Python 训练对接文档](docs/CPP_PYTHON_INFERENCE_ALIGNMENT.md)**

这个文档详细说明了：
- C++ 推理和 Python 训练的数据流程
- 关键对接点（归一化、6D 旋转、delta 组合）
- 常见错误和调试方法
- 代码修改指南

## 目录结构

```
.
├── train/                      # Python 训练代码
│   ├── geometry.py            # 6D 旋转相关函数（关键！）
│   ├── validate/
│   │   └── run_teacher_rollout.py  # 推理验证脚本
│   └── ...
├── reasoning/                  # Unreal Engine C++ 代码
│   ├── EnemyAnimInstance.h
│   └── EnemyAnimInstance.cpp  # 核心推理逻辑
├── raw_data/                   # 原始动画数据
│   └── processed_data/
│       └── norm_template.json  # 归一化参数（MuY, StdY）
├── validate/                   # 验证数据
│   └── teacher_predictions/    # Teacher forcing 参考数据
├── docs/                       # 项目文档
│   └── CPP_PYTHON_INFERENCE_ALIGNMENT.md  # 对接文档
└── README.md                   # 本文件
```

## 快速开始

### Python 训练

```bash
# TODO: 添加训练命令
```

### C++ 推理（Unreal Engine）

1. 打开 Unreal Engine 项目
2. 编译 C++ 代码
3. 配置 `EnemyAnimInstance`:
   - 加载模型：设置 `ModelPath`
   - 加载归一化参数：设置 `NormTemplatePath`
   - （可选）Teacher 模式：启用 `bEnableTeacherPlayback`

## 验证

使用 Teacher Forcing 模式验证 C++ 推理是否正确：

1. 设置 `bEnableTeacherPlayback = true`
2. 设置 `bUseTeacherForcingEval = true`
3. 运行并对比 C++ 输出与 `validate/teacher_predictions/*.json`

详见：[对接文档 - 验证方法](docs/CPP_PYTHON_INFERENCE_ALIGNMENT.md#5-验证方法)

## 已知问题

✅ **已解决：** 骨骼旋转约 90 度偏移问题（Commits 45a8b54, 49c7341）
   - 根因：C++ 反归一化逻辑与 Python 不一致
   - 修复：部分反归一化 + 添加 reproject 步骤
   - 详见：[对接文档 - 修改历史](docs/CPP_PYTHON_INFERENCE_ALIGNMENT.md#7-修改历史)

## 维护指南

**修改代码前请阅读：** [维护指南](docs/CPP_PYTHON_INFERENCE_ALIGNMENT.md#8-维护指南)

## License

[添加许可信息]
