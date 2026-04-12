# 2026-04-10 DSN auxiliary leg supervision implementation checklist

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Status: engineering checklist / no-code spec  
> Scope: `train/models.py` + `train/posttrain.py` + config plumbing  
> Goal: 在 `stage6` 引入 training-only `auxiliary leg head`，但保持最终 `70a/70b` downstream contract 为 baseline-compatible

## 1. Implementation target

本 checklist 对应的不是一个新 inference architecture，而是一个 **training-time scaffold**：

- `stage6` 训练时：有 `auxiliary leg head`
- `70a / 70b / deploy`：没有 `auxiliary leg head`

一句话：

- **训练时多一个 ghost head，导出时仍是 baseline。**

---

## 2. Existing code touchpoints

## 2.1 Model construction

核心模型入口在：

- `train/models.py`
- `EventMotionModel(...)`

当前 direct-pose 相关结构已经集中在这里：

- `direct_pose_head`
- `direct_pose_out_leg`
- `direct_pose_leg_head`
- `direct_pose_arm_proj`
- `direct_pose_else_proj`

因此 auxiliary head 也应该放在同一个模型类中统一管理，而不是在 trainer 外部拼接临时 module。

## 2.2 Posttrain config parsing

配置主入口在：

- `train/posttrain.py`

当前 direct-pose 相关开关已经走完整 schema：

- dataclass fields
- payload parsing
- arg parser / `config_json`
- `_build_posttrain_model_from_ckpt(...)`

因此 auxiliary-leg 配置也应按同一路径接入，而不是做 ad-hoc local variable。

## 2.3 Trainable param selection

当前 freeze / unfreeze 逻辑集中在：

- `train/posttrain_common.py`
- `train/posttrain.py`

包括：

- `_freeze_all(...)`
- `_unfreeze_direct_pose(...)`
- `_unfreeze_for_train_mode(...)`
- `_select_trainable_params(...)`

auxiliary head 应被视作 **direct-pose train mode 的一部分**。

## 2.4 Loss / rollout objective

当前 direct objective 的主入口在：

- `train/posttrain.py`
- `_lambda_fusion_loss_rollout(...)`

leg/nonleg split、3-way、group norm、focus weight 也都在这一路径汇合。

auxiliary loss 应复用这里已有的 group mask / leg slice 语义，而不是另起一套不兼容 target。

## 2.5 Checkpoint compatibility

当前 direct-pose 相关 state-dict 清理逻辑也在：

- `train/posttrain.py`

这里已经有大量 `direct_pose_*` tensor 的 drop / reinit / shape-guard。

auxiliary head 必须接入同样的 compatibility shell：

- 旧 ckpt load 不应被破坏
- deploy/export 时 auxiliary tensor 不应成为 required contract

---

## 3. Proposed new config fields

建议新增一组最小字段，命名保持和现有 `direct_pose_*` 风格一致。

## 3.1 Enable + structure

- `direct_pose_aux_leg_enable: bool`
- `direct_pose_aux_leg_variant: str`
- `direct_pose_aux_leg_hidden: int`
- `direct_pose_aux_leg_detach_feat: bool`

建议默认值：

```text
direct_pose_aux_leg_enable=false
direct_pose_aux_leg_variant=linear
direct_pose_aux_leg_hidden=0
direct_pose_aux_leg_detach_feat=false
```

variant 第一轮只建议支持：

- `linear`
- `mlp`

不要第一轮就做：

- residual stack
- BN/GN-heavy aux tower
- branch-specific adapter cascade

## 3.2 Loss knobs

- `direct_pose_aux_leg_weight: float`
- `direct_pose_aux_leg_loss_mode: str`
- `direct_pose_aux_leg_warmup_steps: int`
- `direct_pose_aux_leg_hold_steps: int`
- `direct_pose_aux_leg_decay_steps: int`
- `direct_pose_aux_leg_min_weight: float`

建议默认值：

```text
direct_pose_aux_leg_weight=0.0
direct_pose_aux_leg_loss_mode=geo
direct_pose_aux_leg_warmup_steps=0
direct_pose_aux_leg_hold_steps=0
direct_pose_aux_leg_decay_steps=0
direct_pose_aux_leg_min_weight=0.0
```

第一轮只建议保留一个 loss mode：

- `geo`

含义：

- auxiliary head 只预测 leg subset 对应的 pose output
- loss 用和主 direct objective 同语义的 geodesic / rotation loss

## 3.3 Logging knobs

- `direct_pose_aux_leg_log_enable: bool`

用于打印：

- `aux_leg_weight`
- `aux_leg_loss`
- `aux_leg_to_main_ratio`
- optional `aux_leg_grad_norm`

---

## 4. Model-side checklist

## 4.1 Add module members

在 `EventMotionModel.__init__` 中新增：

- `self.direct_pose_aux_leg_enable`
- `self.direct_pose_aux_leg_head`

要求：

- 默认 `None`
- 只有当 `direct_pose_enable=true` 且 `direct_pose_aux_leg_enable=true` 时才实例化

## 4.2 Attach point

第一轮固定 attach 到：

- `direct_pose_head` 的 shared trunk output

不要接到：

- `direct_pose_out_leg` 之后
- `direct_pose_leg_head` 之后
- `arm/else` branch 内部

原因：

- 要让 companion gradient 作用到 shared trunk
- 而不是只训一个更晚的 side branch

## 4.3 Output dimension

aux head 的输出维度应直接对齐：

- `direct_pose_leg_out_idx`

即：

- auxiliary 只预测 leg subset 对应的 output slice
- 不预测 full output

这可以最大限度避免：

- 新的非 leg contract
- 不必要的 nonleg interference

## 4.4 No persistent downstream dependency

aux head 不能：

- 改写 `direct_pose_out_leg`
- 改写 `direct_pose_head`
- 改写 main output tensor schema
- 改写 `has_direct_pose_readout()` 的部署语义

换句话说：

- 它是额外的 readout，不是替代 readout

---

## 5. Forward-path checklist

## 5.1 Recommended forward contract

推荐在 model forward 返回值中新增一个 optional field：

```python
ret["direct_pose_aux_leg"] = aux_leg_pred
```

但前提是：

- 只有 enable 时返回
- 不影响现有调用方对主字段的访问

## 5.2 Avoid main-path mutation

禁止做法：

- 用 aux output 去 residual-add main output
- 用 aux output 去 gate main output
- 用 aux output 替换 leg output

否则它就不再是 DSN-style companion objective，而变成新的 permanent path。

## 5.3 Sham control compatibility

`sham aux-head` 臂要求：

- 模块被实例化
- forward 也产生 aux output
- 但 loss 权重为 0

这样 control 才能真正隔离：

- “有个头” vs “有 companion objective”

---

## 6. Loss-side checklist

## 6.1 Reuse existing leg mask semantics

auxiliary loss 不应自己重新推 joint mapping。

优先复用：

- `direct_pose_leg_out_idx`
- 或现有 `loss_fn._resolve_direct_group_masks(...)`

确保：

- auxiliary leg slice
- 主 direct loss leg slice
- probe / eval leg group

三者语义一致。

## 6.2 Loss form

第一轮推荐：

```text
L_total = L_main + w_aux(step) * L_aux_leg
```

其中：

- `L_main` 保持不变
- `L_aux_leg` 只看 aux leg output vs gt leg slice

不要第一轮就引入：

- 额外 KL
- consistency loss
- cross-head agreement loss
- feature distillation loss

先做最小 companion objective。

## 6.3 Weight schedule

新增一个 helper，例如：

- `_direct_pose_aux_leg_weight(step, cfg)`

推荐支持：

1. warmup
2. hold
3. decay

目标：

- 前期给 trunk leg gradient
- 后期逐步交还主目标

## 6.4 Logging

每步或每 N 步至少记录：

- `aux_leg_weight`
- `aux_leg_loss`
- `aux_leg_loss_weighted`
- `aux_leg_over_main`

可选记录：

- `aux_leg_grad_norm`
- `trunk_grad_with_aux`

---

## 7. Freeze / optimizer checklist

## 7.1 Direct-pose training mode

在 `train_direct_pose=true` 下：

- auxiliary head 参数应默认 `requires_grad=true`
- shared trunk 参数维持当前 direct-pose 训练逻辑

## 7.2 No new standalone train mode

第一轮不建议新增：

- `train_aux_leg_only`
- `train_aux_leg_then_main`

原因：

- 会把最小 falsifier 变复杂
- 当前要回答的问题只是：`companion objective` 是否有帮助

## 7.3 Param group policy

第一轮建议：

- auxiliary head 与 direct head 同 lr

不要第一轮就做：

- 独立 lr
- head/trunk differential lr

这些都可以留到正信号之后。

---

## 8. Checkpoint / export checklist

## 8.1 Load compatibility

旧 checkpoint 加载新模型时：

- auxiliary head 相关 key 缺失必须是允许的
- 不得要求历史 ckpt 拥有 aux tensors

## 8.2 Save compatibility

需要明确区分两类 artifact：

1. **train artifact**
   - 可包含 auxiliary tensors
2. **deploy / handoff artifact**
   - 不应要求 auxiliary tensors

## 8.3 Recommended rule

建议在保存 `70a` / downstream handoff ckpt 时：

- strip 掉 `direct_pose_aux_leg_*`

这样可以把 downstream contract 写死成：

- baseline-compatible direct-pose state only

## 8.4 State-dict prefix

建议统一 prefix：

- `direct_pose_aux_leg_head.*`

这样好处是：

- drop / keep / audit 都容易做
- 不会和现有 `direct_pose_leg_head.*` 混淆

---

## 9. Validation checklist

## 9.1 Structural sanity

- baseline config 不开 aux 时，模型结构与当前一致
- enable aux 时，main output shape 完全不变
- `has_direct_pose_readout()` 语义不变

## 9.2 Training sanity

- `baseline` / `sham` / `aux` 三臂都能正常启动
- `sham` 臂 `aux_leg_loss` 有值但 weighted contribution 为 0
- `aux` 臂 `aux_leg_weight` 按 schedule 变化

## 9.3 Checkpoint sanity

- old ckpt -> new code 可加载
- aux-trained ckpt -> stripped deploy ckpt 可生成
- stripped ckpt -> `70a / 70b replace` 可按 baseline 路径继续加载

## 9.4 Metric sanity

至少检查：

- `stage6 native`
- `70a native`
- `70b replace`

并额外记录：

- leg / nonleg / all_ex_root

---

## 10. Minimal experiment execution plan

## 10.1 Arm A: baseline

- 当前 recipe 原样跑

## 10.2 Arm B: sham aux-head

- `direct_pose_aux_leg_enable=true`
- `direct_pose_aux_leg_weight=0`

## 10.3 Arm C: actual DSN aux-leg

- `direct_pose_aux_leg_enable=true`
- `direct_pose_aux_leg_weight>0`
- 其余与 baseline 保持一致

## 10.4 Hold constant

第一轮必须固定：

- donor family
- `epochs`
- `steps_per_epoch`
- `lr`
- `encoder_bundle`
- `direct_pose_use_phase_z`
- `direct_pose_phase_z_mode`

否则结论会被 recipe drift 污染。

---

## 11. Suggested implementation order

### Step 1

只加 config schema + model member + forward optional output  
不接 loss  
先确认：

- enable/disable 都能跑
- state-dict 不炸

### Step 2

接 `L_aux_leg` 与 logging  
但先不做 strip export  
先确认：

- `baseline/sham/aux` 三臂 train-time 行为可区分

### Step 3

补 deploy-strip 逻辑  
确认：

- aux-trained `stage6/70a` ckpt 导出后仍可走 baseline downstream
- 最小静态 sanity 通过后，再进入 Step 4 的真实链路验证

### Step 4

再做完整 `stage6 -> 70a -> 70b replace` 链路验证

---

## 12. Hard rules

1. 不把 aux output 混入 main forward result 的部署语义  
2. 不让 aux module 成为 downstream required contract  
3. 不把第一轮实验做成多因素 sweep  
4. 不在第一轮引入新的 normalization / EMA-heavy submodule  
5. 不把 “aux improves train loss” 当成成功  
6. 不经过 `70b replace` 就宣布方向成立  

---

## 13. One-screen build checklist

- [x] 在 `EventMotionModel` 新增 `direct_pose_aux_leg_head`
- [x] attach 到 `direct_pose_head` shared trunk output
- [x] 输出维度对齐 `direct_pose_leg_out_idx`
- [x] 新增 `direct_pose_aux_leg_*` config fields
- [x] 在 posttrain config parser 中接入 schema
- [x] 在 train-time forward 返回 optional `direct_pose_aux_leg`
- [x] 在 rollout loss 中加入 `L_aux_leg`
- [x] 增加 `aux_leg_weight` schedule helper
- [x] 增加 `aux_leg_loss` logging
- [x] `sham` 臂可通过 `weight=0` 实现
- [x] old ckpt load 不受影响
- [x] deploy/export 可 strip `direct_pose_aux_leg_head.*`
- [x] `70a -> 70b replace` 仍按 baseline contract 继续

Step 4 record:

- `docs/retired_directions/aux_shared_trunk_family/2026-04-10_dsn_auxiliary_leg_supervision_step4_record.md`

## 14. Final engineering bet

> 如果这条线是对的，那么最先出现的正信号，不应该是“aux head 很会做 leg”，而应该是“删掉 aux head 之后，baseline contract 仍然更能服务 70b replace”。
