# Post-Training（后训练）入口：`train/posttrain.py`

目标：保持主训练流程精简，把“长 free-run 分布暴露 / 动作切换模拟 / reset 策略”等更贴近游戏的校准逻辑，放到独立的 post-train 阶段。

当前实现：**冻结 base model，只训练 SO(3) corrector head**（`omega_hat` 分支 + 可选 gate logit）。

补充：为了让 contact-loop 真正“闭环”，`train/posttrain.py` 现在也支持可选地 **finetune `contact_meas_head`**（监督来自 GT soft contacts，但不把 GT contacts 喂进 forward，避免 train/infer mismatch）。详细设计见：

- `docs/contact_loop_closure_design.md`

## 用法

1) 准备配置：

- 示例配置：`config/posttrain.json`
- 你可以用 `paths` 限定参与后训练的 `.npz`（更可控、更快）：
  - `paths: ["raw_data/processed_data/xxx.npz", ...]`

2) 运行后训练：

```bash
python -m train.posttrain --config config/posttrain.json
```

也可以直接用命令行覆盖关键参数（不用改 JSON）：

```bash
python -m train.posttrain \
  --ckpt_in models/xxx/ckpt_last_xxx.pth \
  --out_dir models/posttrain \
  --run_name posttrain_corr_only
```

### 可选：只训练 / 同时训练 contact_meas_head

只训练 `contact_meas_head`（建议先做这步，把 `contacts_meas` 训成“像观测”的信号，让 `plan - meas` 有意义）：

```bash
python -m train.posttrain \
  --config config/posttrain_contactloop_corr.json \
  --run_name posttrain_meas_only \
  --train_so3_corrector false \
  --train_contact_meas true \
  --contact_meas_weight 1.0
```

联合训练（meas + so3 corrector）：

```bash
python -m train.posttrain \
  --config config/posttrain_contactloop_corr.json \
  --run_name posttrain_meas_plus_corr \
  --train_so3_corrector true \
  --train_contact_meas true \
  --contact_meas_weight 0.5
```

## 常见问题：gate 太小学不动

如果日志里长期出现 `gate≈0.0067` 且 `omega_l2≈0`，说明 ckpt 里的 `so3_corr_gate_logit` 基本把纠偏关死了（梯度被大幅缩小）。

推荐两种方式（二选一）：

- **Warmup 强制 gate**（更稳）：`--gate_warmup_steps 200 --gate_warmup_value 0.1`
- **重置 gate logit**（更“干净”）：`--so3_corr_gate_logit_reset -2.2`（sigmoid≈0.1）

产物：
- `models/posttrain/ckpt_last_<run_name>.pth`
- `models/posttrain/posttrain_log_<run_name>.json`

3) 用 free-run 验证脚本做成果展示/回归：

```bash
python -m train.validate.run_freerun_cycles --model models/posttrain/ckpt_last_<run_name>.pth ...
```
