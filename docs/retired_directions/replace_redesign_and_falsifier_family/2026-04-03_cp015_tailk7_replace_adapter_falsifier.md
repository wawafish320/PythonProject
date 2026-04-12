# CP015 tailk7 replace：zero-init residual adapter falsifier（附 routing follow-up）

> Status: retired falsifier record / historical positive-then-downgraded evidence
> Current role: archaeology / interface-translation evidence with later sham downgrade
> Important later downgrade:
> - `docs/retired_directions/replace_redesign_and_falsifier_family/2026-04-10_branch_sham_audit_replace_adapter_record.md`

> Last updated: 2026-04-04
> 目的：只做一个最小 falsifier，回答当前 `tailk7 replace/freerun` 不佳是否主要因为 downstream interface 无法平滑读取 donor geometry；以及一个最小 zero-init residual adapter 能否明显改善。

本文主记录 `adapter falsifier`，并在末尾追加 2026-04-04 的最小 `routing / leg-bypass falsifier` follow-up：

- 基线：`tailk7 replace best schedule e3x60 lr=5e-5`
- 唯一变量：在 frozen donor trunk 输出到现有 downstream 入口之间，加一个 shape-preserving zero-init residual adapter
- 其余保持不变：
  - 不改 replace / 70R 语义
  - 不换 donor
  - 不改 warmstart
  - 不改 loss
  - 不改 optimizer / weight_decay / data / batch / seq_len / seed / epochs / rollout_cycles / event_clock / phase_reset_source / contacts_source
  - 不做 hyperparameter matrix

---

## 0. TL;DR

本轮固定结论：

1. 最小 adapter 确实能带来 **freerun 改善**，不是只改 train。
2. 相对当前主对照 `tailk7 e3x60 lr=5e-5`：
   - `arm p95`: `0.554550 -> 0.550443`
   - `all_ex_root p95`: `0.622943 -> 0.611407`
   - `leg p95`: `0.890515 -> 0.882486`
3. `arm p90` 没有改善，略升：
   - `0.414565 -> 0.415701`
4. 但判定标准里关键的 `arm p95`、`all_ex_root p95` 都下降，且 `leg` 没有变坏，因此这轮**在当时的约定判据下**判：
   - **A. adapter 明显改善，支持 interface translation / geometry mismatch 假设**
5. 这仍然不是“已经足够”的最终解：
   - 它只是说明 downstream interface translation 确实是主要矛盾之一
   - 同时也说明 `adapter-only` 比“单纯改 LR”更接近问题核心
6. Current reading after the later sham audit:
   - do **not** read this file as branch-only causal proof
   - read it as historical positive evidence whose global branch-effect interpretation was later downgraded

---

## 1. 固定输入与产物

### 1.1 模板 / 对照

- 模板 config:
  - `debug_output/_tmp_cp015_tailk7_replace_schedule_ablation_20260402/configs/posttrain_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.json`
- 主对照 ckpt:
  - `models/__tmp_cp015_tailk7_replace_schedule_ablation_20260402/e3x60/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.pth`
- 主对照 summary:
  - `debug_output/_tmp_cp015_tailk7_replace_schedule_ablation_20260402/eval_model_source/e3x60_group_summary.json`

### 1.2 新 case

- 新 config:
  - `debug_output/_tmp_cp015_tailk7_replace_adapter_falsifier_20260403/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403.json`
- 新 model dir:
  - `models/__tmp_cp015_tailk7_replace_adapter_falsifier_20260403/e3x60_adapter`
- 新 eval dir:
  - `debug_output/_tmp_cp015_tailk7_replace_adapter_falsifier_20260403/eval_model_source/e3x60_adapter`
- 新 ckpt:
  - `models/__tmp_cp015_tailk7_replace_adapter_falsifier_20260403/e3x60_adapter/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403.pth`
- 新 train log:
  - `models/__tmp_cp015_tailk7_replace_adapter_falsifier_20260403/e3x60_adapter/posttrain_log_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403.json`
- 新 eval json:
  - `debug_output/_tmp_cp015_tailk7_replace_adapter_falsifier_20260403/eval_model_source/e3x60_adapter/Walk_F_freerun_cycles.json`
- 新 group summary:
  - `debug_output/_tmp_cp015_tailk7_replace_adapter_falsifier_20260403/eval_model_source/e3x60_adapter_group_summary.json`

---

## 2. 唯一实现改动

### 2.1 插入点

adapter 插在：

- frozen donor trunk 产出的 `direct_feat`
- 进入现有 `direct_pose_head` 之前

当前这个 case 的 `direct_pose_feat_source=cond`，所以实际是：

- `cond -> adapter -> 现有 downstream direct head input`

代码位置：

- `train/models.py`
- `direct_feat = cond/...` 之后
- `if self.direct_pose_input_adapter is not None: direct_feat = self.direct_pose_input_adapter(direct_feat)`

### 2.2 结构与 shape

结构固定为：

`LayerNorm -> Linear(D,256) -> SiLU -> Linear(256,D)`

输出固定为：

`y = x + adapter(x)`

当前 case 的输入输出维度：

- `D = 7`
- batch rollout 态：`(B, Tq, 7)`
- 单步 freerun 态：`(B, 7)`

### 2.3 zero-init

最后一层 `Linear(256,D)` 做 strict zero-init：

- `weight = 0`
- `bias = 0`

因此 step0 时 adapter 是严格 identity start。

### 2.4 参数量

adapter 参数量：

- `LayerNorm(7)`: `7 + 7 = 14`
- `Linear(7,256)`: `7*256 + 256 = 2048`
- `Linear(256,7)`: `256*7 + 7 = 1799`
- 合计：`14 + 2048 + 1799 = 3861`

总 trainable 参数变化：

| case | trainable params |
|---|---:|
| 原 `e3x60 lr=5e-5` | `929070` |
| `e3x60_adapter` | `932931` |
| delta | `+3861` |

---

## 3. 实际运行命令

### 3.1 Train

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_replace_adapter_falsifier_20260403/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403.json \
  --ckpt_in /Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_schedule_ablation_20260402/warmstart/ckpt_last_cp015_tailk7_70a_replace_zerophase_20260402.pth \
  --out_dir /Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_adapter_falsifier_20260403/e3x60_adapter \
  --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle /Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 3.2 Freerun Eval

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model /Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_adapter_falsifier_20260403/e3x60_adapter/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403.pth \
  --rounds 5 \
  --depth 3 \
  --time-index-mode cycle \
  --event_clock auto \
  --phase_reset_source none \
  --contacts_meas_source model \
  --lambda_fusion_apply \
  --log_contacts \
  --export_direct_arm_probe \
  --export_joint_direct_geolocal_series \
  --out /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_replace_adapter_falsifier_20260403/eval_model_source/e3x60_adapter \
  --force
```

### 3.3 Group Summary

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  tools/phasea_group_summary.py \
  /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_replace_adapter_falsifier_20260403/eval_model_source/e3x60_adapter/Walk_F_freerun_cycles.json \
  --cycle_gte 1 \
  --drop_wrap \
  --out /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_replace_adapter_falsifier_20260403/eval_model_source/e3x60_adapter_group_summary.json
```

---

## 4. Train 结果

### 4.1 stdout 关键日志

```text
[posttrain] trainable=26 params: direct_pose_input_adapter.norm.weight, ...
[posttrain] direct_pose_input_adapter in_dim=7 hidden=256 params=3861 last_weight_absmax_init=0.000000 last_bias_absmax_init=0.000000
[posttrain][epoch 1] avg_total=1.528294 ok_steps=60 skipped=0
[posttrain][epoch 2] avg_total=1.889729 ok_steps=60 skipped=0
[posttrain][epoch 3] avg_total=1.954992 ok_steps=60 skipped=0
```

### 4.2 final train 指标

按 epoch 平均：

| epoch | avg_total |
|---|---:|
| 1 | `1.528294` |
| 2 | `1.889729` |
| 3 | `1.954992` |

最后一步：

- `step=180`
- `total=2.131518`
- `dir_geo=2.131518`
- `blend_loss=0.321705`

对照原 `e3x60 lr=5e-5`：

| case | epoch1 avg_total | epoch2 avg_total | epoch3 avg_total |
|---|---:|---:|---:|
| 原 `e3x60 lr=5e-5` | `1.488669` | `1.907731` | `1.933253` |
| `e3x60_adapter` | `1.528294` | `1.889729` | `1.954992` |

解释：

- train 端没有出现稳定的“显著更好”
- 这轮 improvement 不是 train loss 下降驱动的误判

---

## 5. Freerun 对照结果

统一记为 `mean / p90 / p95`。

| case | arm | all_ex_root | leg | else |
|---|---|---|---|---|
| `tailk7 70a` | `0.177822 / 0.445496 / 0.629841` | `0.213743 / 0.566093 / 0.783662` | `0.502952 / 0.970880 / 1.175182` | `0.088314 / 0.201317 / 0.259110` |
| `tailk7 e3x60 lr=5e-5` | `0.160056 / 0.414565 / 0.554550` | `0.182774 / 0.461205 / 0.622943` | `0.379237 / 0.745277 / 0.890515` | `0.093588 / 0.200817 / 0.288886` |
| `tailk7 e3x60 lr=1e-5 falsifier` | `0.168610 / 0.430735 / 0.608705` | `0.192825 / 0.484252 / 0.690631` | `0.407328 / 0.808446 / 0.970266` | `0.094060 / 0.201514 / 0.296829` |
| `baseline replace` | `0.116105 / 0.303279 / 0.423788` | `0.152126 / 0.402586 / 0.567555` | `0.391665 / 0.795561 / 0.998558` | `0.063055 / 0.140461 / 0.177474` |
| `tailk7 e3x60 adapter` | `0.156052 / 0.415701 / 0.550443` | `0.175801 / 0.438493 / 0.611407` | `0.364820 / 0.738802 / 0.882486` | `0.085013 / 0.187801 / 0.261784` |

### 5.1 相对当前主对照 `tailk7 e3x60 lr=5e-5` 的 delta

| group | mean delta | p90 delta | p95 delta |
|---|---:|---:|---:|
| arm | `-0.004004` | `+0.001137` | `-0.004107` |
| all_ex_root | `-0.006972` | `-0.022712` | `-0.011536` |
| leg | `-0.014418` | `-0.006475` | `-0.008029` |
| else | `-0.008575` | `-0.013016` | `-0.027102` |

关键观察：

1. `arm p95` 下降：
   - `0.554550 -> 0.550443`
2. `all_ex_root p95` 下降：
   - `0.622943 -> 0.611407`
3. `leg` 没变坏，反而三项都下降：
   - `0.379237 / 0.745277 / 0.890515`
   - `-> 0.364820 / 0.738802 / 0.882486`
4. `arm p90` 没降，略升：
   - `0.414565 -> 0.415701`
   - 说明改善不是全分布均匀的，而更像在右尾和全局接口对齐上起作用

---

## 6. “adapter 确实启用且 zero-init 生效”的证据

### 6.1 train log config 证据

文件：

- `models/__tmp_cp015_tailk7_replace_adapter_falsifier_20260403/e3x60_adapter/posttrain_log_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403.json`

关键字段：

- `direct_pose_input_adapter_enable = true`
- `direct_pose_input_adapter_enabled = true`
- `direct_pose_input_adapter_in_dim = 7`
- `direct_pose_input_adapter_hidden_dim = 256`
- `direct_pose_input_adapter_param_count = 3861`
- `direct_pose_input_adapter_last_weight_absmax_init = 0.0`
- `direct_pose_input_adapter_last_bias_absmax_init = 0.0`

### 6.2 stdout 证据

训练启动时直接打印：

```text
[posttrain] trainable=26 params: direct_pose_input_adapter.norm.weight, ...
[posttrain] direct_pose_input_adapter in_dim=7 hidden=256 params=3861 last_weight_absmax_init=0.000000 last_bias_absmax_init=0.000000
```

这两条已经足够说明：

1. adapter 被加入 trainable set
2. 输入维度是当前实际 downstream 入口维度 `D=7`
3. 最后一层确实是 strict zero-init

### 6.3 freerun load 证据

freerun 运行时没有出现 adapter unexpected keys，被正常 runtime 重建并加载：

```text
[FreeRun][WARN] state_dict mismatch: missing=[frozen_encoder..., frozen_period_head...], unexpected=[]
```

这里 `unexpected=[]` 很关键，说明 adapter 没被 eval 侧丢掉。

---

## 7. 最小结论

按预设判定标准：

- 若 adapter 相对 `tailk7 e3x60 lr=5e-5` 让 `arm p95` 和 `all_ex_root p95` 明显下降，并且 `leg` 没明显变坏，判 `A`
- 若 train 有改善但 freerun 持平或更差，尤其 `arm p95` 不降反升，判 `B`

本轮结果：

- `arm p95`: `0.554550 -> 0.550443`，下降
- `all_ex_root p95`: `0.622943 -> 0.611407`，下降
- `leg p95`: `0.890515 -> 0.882486`，未变坏
- 同时 train 端没有明显变好，因此这不是“train 小幅变化误判”

因此本轮结论固定为：

> **A. adapter 明显改善，支持 interface translation / geometry mismatch 假设**

但也要同时固定一个边界：

> 这只能说明 `adapter-only` 已经足以 falsify “主因只是 LR / 优化太猛”。
> 它不等于已经证明更复杂的 routing / bypass 永远不需要，只是说明下一步不该再优先回到 LR 争论。

---

## 8. 2026-04-04 follow-up：minimal routing / leg-bypass falsifier

这一轮不是重开设计，而是在当前 `e3x60_adapter` control 上只做一个最小 routing 变量，回答：

> adapter 已经证明 interface translation 有帮助后，剩余 gap 是否主要来自 downstream group coupling；一个最小 leg-bypass / group-decoupling 能不能继续改善 freerun？

### 8.1 固定输入与新产物

- control config：
  - `debug_output/_tmp_cp015_tailk7_replace_adapter_falsifier_20260403/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403.json`
- control ckpt：
  - `models/__tmp_cp015_tailk7_replace_adapter_falsifier_20260403/e3x60_adapter/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403.pth`
- 新 config：
  - `debug_output/_tmp_cp015_tailk7_replace_routing_falsifier_20260403/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_legbypass_lr5e5_from_cp015_tailk7_70a_20260403.json`
- 新 model dir：
  - `models/__tmp_cp015_tailk7_replace_routing_falsifier_20260403/e3x60_adapter_legbypass`
- 新 eval dir：
  - `debug_output/_tmp_cp015_tailk7_replace_routing_falsifier_20260403/eval_model_source/e3x60_adapter_legbypass`
- 新 ckpt：
  - `models/__tmp_cp015_tailk7_replace_routing_falsifier_20260403/e3x60_adapter_legbypass/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_legbypass_lr5e5_from_cp015_tailk7_70a_20260403.pth`
- 新 train log：
  - `models/__tmp_cp015_tailk7_replace_routing_falsifier_20260403/e3x60_adapter_legbypass/posttrain_log_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_legbypass_lr5e5_from_cp015_tailk7_70a_20260403.json`
- 新 eval json：
  - `debug_output/_tmp_cp015_tailk7_replace_routing_falsifier_20260403/eval_model_source/e3x60_adapter_legbypass/Walk_F_freerun_cycles.json`
- 新 group summary：
  - `debug_output/_tmp_cp015_tailk7_replace_routing_falsifier_20260403/eval_model_source/e3x60_adapter_legbypass_group_summary.json`

这一轮没有改源码，只新增了一个 config，复用仓库里现成的 `direct_pose_leg_head` / `direct_pose_leg_stopgrad_main` 语义。

### 8.2 精确 routing / bypass 语义

#### 8.2.1 插入点

现有 `adapter` 保持不变：

- `cond -> direct_pose_input_adapter -> direct_feat`

随后形成现有 downstream 入口：

- `direct_feat + time_pe + plan + meas -> direct_flat`

共享 downstream path 仍然是：

- `direct_flat -> direct_pose_head(shared trunk) -> direct_pose_out_leg`
- `direct_flat -> direct_pose_head(shared trunk) -> direct_pose_arm_proj -> direct_pose_out_arm`
- `direct_flat -> direct_pose_head(shared trunk) -> direct_pose_else_proj -> direct_pose_out_else`

本轮启用的最小 bypass 是：

- `direct_flat -> direct_pose_leg_head -> direct_leg_omega`
- 然后在 raw-space compose 时，把 `direct_leg_omega` 作用到 `R_leg_base`

关键切断点在 leg compose 前：

- `direct_pose_leg_stopgrad_main = true`
- 因此 `R_leg_base = R_leg_base.detach()`

这意味着：

1. leg residual 仍然读取 donor+adapter 后的 `direct_flat`
2. arm/else 继续完全走原来的共享 downstream path
3. leg loss 不再主要通过 `direct_pose_head -> direct_pose_out_leg` 这条共享主更新路径回传
4. 这次没有启用 `direct_pose_leg_detach_feat`，所以 leg residual 仍可回到 `direct_flat` 上游；被绕开的是 shared main leg update path，不是把 leg 支路完全孤立成冻结输入

#### 8.2.2 Shape

当前 case 的关键 shape：

- adapter 输入输出：`(B, Tq, 7)`，单步是 `(B, 7)`
- `direct_flat`：`(B, Tq, 43)`
- shared trunk hidden：`(B*Tq, 512)`
- shared leg readout：`(B, Tq, 48)`，对应 `8 * 6`
- arm readout：`(B, Tq, 156)`，对应 `26 * 6`
- else readout：`(B, Tq, 74)`
- bypass leg residual 输出：`(B, Tq, 24)`，对应 `8 * 3`
- compose 前实际 leg omega：`(B, Tq, 8, 3)`

#### 8.2.3 参数量

本轮没有新增模块，只改 routing flag：

| case | trainable params |
|---|---:|
| `e3x60_adapter` | `932931` |
| `e3x60_adapter_legbypass` | `932931` |
| delta | `0` |

其中 adapter 仍是：

- `3861` params

### 8.3 实际运行命令

#### 8.3.1 Train

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_replace_routing_falsifier_20260403/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_legbypass_lr5e5_from_cp015_tailk7_70a_20260403.json \
  --ckpt_in /Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_schedule_ablation_20260402/warmstart/ckpt_last_cp015_tailk7_70a_replace_zerophase_20260402.pth \
  --out_dir /Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_routing_falsifier_20260403/e3x60_adapter_legbypass \
  --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_legbypass_lr5e5_from_cp015_tailk7_70a_20260403 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle /Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

#### 8.3.2 Freerun Eval

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model /Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_routing_falsifier_20260403/e3x60_adapter_legbypass/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_legbypass_lr5e5_from_cp015_tailk7_70a_20260403.pth \
  --rounds 5 \
  --depth 3 \
  --time-index-mode cycle \
  --event_clock auto \
  --phase_reset_source none \
  --contacts_meas_source model \
  --lambda_fusion_apply \
  --log_contacts \
  --export_direct_arm_probe \
  --export_joint_direct_geolocal_series \
  --out /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_replace_routing_falsifier_20260403/eval_model_source/e3x60_adapter_legbypass \
  --force
```

#### 8.3.3 Group Summary

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  tools/phasea_group_summary.py \
  /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_replace_routing_falsifier_20260403/eval_model_source/e3x60_adapter_legbypass/Walk_F_freerun_cycles.json \
  --cycle_gte 1 \
  --drop_wrap \
  --out /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_replace_routing_falsifier_20260403/eval_model_source/e3x60_adapter_legbypass_group_summary.json
```

### 8.4 Train 结果

epoch 平均：

| epoch | avg_total |
|---|---:|
| 1 | `1.541961` |
| 2 | `1.885475` |
| 3 | `1.933667` |

最后一步：

- `step=180`
- `total=2.151028`
- `dir_geo=2.151028`
- `blend_loss=0.321705`

对照 `e3x60_adapter`：

| case | epoch1 avg_total | epoch2 avg_total | epoch3 avg_total |
|---|---:|---:|---:|
| `tailk7 e3x60 adapter` | `1.528294` | `1.889729` | `1.954992` |
| `tailk7 e3x60 adapter leg-bypass` | `1.541961` | `1.885475` | `1.933667` |

结论：

- train 有变化，但不是更强的正向信号
- 这一轮仍然必须看 freerun，而不能按 train loss 判

### 8.5 Freerun 对照结果

统一记为 `mean / p90 / p95`。

| case | arm | all_ex_root | leg | else |
|---|---|---|---|---|
| `tailk7 70a` | `0.177822 / 0.445496 / 0.629841` | `0.213743 / 0.566093 / 0.783662` | `0.502952 / 0.970880 / 1.175182` | `0.088314 / 0.201317 / 0.259110` |
| `tailk7 e3x60 lr=5e-5` | `0.160056 / 0.414565 / 0.554550` | `0.182774 / 0.461205 / 0.622943` | `0.379237 / 0.745277 / 0.890515` | `0.093588 / 0.200817 / 0.288886` |
| `tailk7 e3x60 lr=1e-5 falsifier` | `0.168610 / 0.430735 / 0.608705` | `0.192825 / 0.484252 / 0.690631` | `0.407328 / 0.808446 / 0.970266` | `0.094060 / 0.201514 / 0.296829` |
| `tailk7 e3x60 adapter` | `0.156052 / 0.415701 / 0.550443` | `0.175801 / 0.438493 / 0.611407` | `0.364820 / 0.738802 / 0.882486` | `0.085013 / 0.187801 / 0.261784` |
| `baseline replace` | `0.116105 / 0.303279 / 0.423788` | `0.152126 / 0.402586 / 0.567555` | `0.391665 / 0.795561 / 0.998558` | `0.063055 / 0.140461 / 0.177474` |
| `tailk7 e3x60 adapter leg-bypass` | `0.154278 / 0.400737 / 0.562478` | `0.178262 / 0.442584 / 0.630359` | `0.384593 / 0.775811 / 0.971611` | `0.084890 / 0.190240 / 0.274835` |

#### 8.5.1 相对 `tailk7 e3x60 adapter` 的 delta

| group | mean delta | p90 delta | p95 delta |
|---|---:|---:|---:|
| arm | `-0.001774` | `-0.014965` | `+0.012035` |
| all_ex_root | `+0.002460` | `+0.004091` | `+0.018951` |
| leg | `+0.019774` | `+0.037009` | `+0.089124` |
| else | `-0.000123` | `+0.002438` | `+0.013051` |

关键观察：

1. `arm p95` 没有继续下降，反而变坏：
   - `0.550443 -> 0.562478`
2. `all_ex_root p95` 也变坏：
   - `0.611407 -> 0.630359`
3. `leg` 没守住，明显变坏：
   - `0.882486 -> 0.971611`
4. 虽然 `arm mean` / `arm p90` 略有改善，但这不满足预设判定标准

### 8.6 “routing / bypass 确实启用”的证据

#### 8.6.1 train log config 证据

文件：

- `models/__tmp_cp015_tailk7_replace_routing_falsifier_20260403/e3x60_adapter_legbypass/posttrain_log_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_legbypass_lr5e5_from_cp015_tailk7_70a_20260403.json`

关键字段：

- `direct_pose_leg_stopgrad_main = true`
- `direct_pose_leg_detach_feat = false`
- `direct_pose_input_adapter_enable = true`
- `direct_pose_split_enable = true`
- `direct_pose_arm_split_enable = true`

这已经说明当前 case 不是去掉 adapter，而是在 `adapter` 保持开启的基础上，只把 leg compose 的主链梯度改成 stopgrad。

#### 8.6.2 现有代码语义证据

这一轮复用的是现成代码，不是新造第二套架构：

- `train/models.py`
  - `direct_feat` 先过 adapter
  - 然后形成 `direct_flat`
  - `direct_pose_leg_head(leg_in)` 直接从 `direct_flat` 读入
- `train/posttrain.py`
  - raw-space leg compose 时，若 `direct_pose_leg_stopgrad_main=true`，则 `R_leg_base = R_leg_base.detach()`

因此本轮的 leg-bypass 精确语义是：

- **leg 路径绕开的是 shared main leg update path**
- **arm/else 仍然走 shared downstream path**
- **新增参数量为 0**

### 8.7 最小结论

按约定判据：

- 若相对 `e3x60_adapter`，`arm p95` 和 `all_ex_root p95` 继续下降，且 `leg` 不明显变坏，判 `A`
- 若 train 有变化但 freerun 持平或更差，尤其 `arm p95` 不降，判 `B`

本轮结果是：

- `arm p95`: `0.550443 -> 0.562478`，不降反升
- `all_ex_root p95`: `0.611407 -> 0.630359`，变差
- `leg p95`: `0.882486 -> 0.971611`，明显变坏

因此这轮 follow-up 的结论固定为：

> **B. routing / bypass 也基本救不回来，simple decoupling 不是主要剩余问题**

同时需要把 adapter 结论和这轮 routing 结论分开：

1. `adapter falsifier` 作为 historical signal 仍然成立：
   - **A. interface translation / geometry mismatch likely had some help signal**
2. 但在 adapter 之上，再加这个最小 `leg-bypass / group-decoupling`：
   - **没有继续改善 freerun**
   - 因此不能把剩余主因简单归结为“arm/leg downstream gradient contamination”这一个问题

Current reading:

- keep this file as historical positive evidence with limited scope
- use the later sham audit when discussing whether the branch gain was a clean objective effect
