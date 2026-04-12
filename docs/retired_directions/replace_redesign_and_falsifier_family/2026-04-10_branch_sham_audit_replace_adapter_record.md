# 2026-04-10 branch sham audit: replace zero-init residual adapter

> Status: retired methodology-caution record
> Current role: branch/sham reading discipline, not a live recipe candidate
> Use this file to avoid over-reading structural branch gains as clean objective effects.

> Date: 2026-04-10  
> Scope: 只做 matched sham audit；不改 trunk / attach point / optimizer / loss semantics；不扩成新的 recipe sweep  
> Methodology note: 本文是独立的 branch-sham 方法学记录，不混入 DSN 成败判断

## 1. Why this object

本轮只固定 **1 个** 审计对象：

- `replace zero-init residual adapter`

选择理由：

- 它是近期最明确被读成 **“加一个结构本身有效”** 的正结论之一
- 原始结论口径比较强：`adapter` 被判为支持 `interface translation / geometry mismatch`
- 它是一个很典型的 **extra structural branch / module** 干预
- 可以在**不改代码**的前提下补一个 matched sham：保持 adapter 结构存在，但把 adapter param group 的 `lr=0.0`

本轮**不加第二对象**，是为了严格控 scope。`factorized readout` 虽然也是结构干预，但它的历史结论本来就只是“局部 tail 改善，不是全局解”，优先级低于 `adapter`。

## 2. Matched three-arm design

三臂定义：

1. `baseline`
   - 历史对照：`tailk7 replace best schedule e3x60 lr=5e-5`
   - 无 adapter
2. `sham`
   - adapter 结构存在
   - adapter 参数仍进入 optimizer param group
   - 但 `direct_pose_input_adapter.*` 的 param group `lr=0.0`
   - 目的：隔离 **extra module presence / optimizer trajectory fork / param-group perturbation**
3. `branch`
   - 历史真实改动臂：`zero-init residual adapter`

这一定义刻意对齐当前 `DSN aux-leg` 的 sham 语义：

- 结构在场
- 不给该结构真实训练收益
- 只看 structural perturbation 到底有多大

## 3. Recipe inventory

### 3.1 Shared donor / warmstart

| item | value |
| --- | --- |
| donor family | `cp015 tailk7 canonical top7 donor -> replace e3x60` |
| donor stage coverage | **replace-stage only** |
| upstream `70a` source | `models/__tmp_cp015_tailk7_replace_schedule_ablation_20260402/warmstart/ckpt_last_cp015_tailk7_70a_replace_zerophase_20260402.pth` |
| scope note | 该对象只覆盖 `new70b_replace_lowdrift`；`stage6 native` / `70a native` 不存在三臂分叉，三臂共享同一个 `70a` warmstart donor |

### 3.2 Arm-by-arm config

| arm | config | epochs | steps_per_epoch | lr | encoder_bundle | direct_pose_use_phase_z | direct_pose_phase_z_mode |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `baseline` | `debug_output/_tmp_cp015_tailk7_replace_schedule_ablation_20260402/configs/posttrain_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.json` | `3` | `60` | `5e-5` | `models/motion_encoder_equiv.pt.best.pt` | `true` | `concat` |
| `sham` | `debug_output/_tmp_adapter_sham_audit_20260410/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_sham_lr0_20260410.json` | `3` | `60` | `5e-5` | `models/motion_encoder_equiv.pt.best.pt` | `true` | `concat` |
| `branch` | `debug_output/_tmp_cp015_tailk7_replace_adapter_falsifier_20260403/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403.json` | `3` | `60` | `5e-5` | `models/motion_encoder_equiv.pt.best.pt` | `true` | `concat` |

唯一额外差异：

- `baseline`: `direct_pose_input_adapter_enable = false`
- `sham`: `direct_pose_input_adapter_enable = true`, 但 `optimizer_param_group_overrides=[{"name":"adapter_frozen","lr":0.0,"module_prefixes":["direct_pose_input_adapter"]}]`
- `branch`: `direct_pose_input_adapter_enable = true`, adapter 正常训练

## 4. Actual commands

### 4.1 Historical baseline command

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_cp015_tailk7_replace_schedule_ablation_20260402/configs/posttrain_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.json \
  --ckpt_in models/__tmp_cp015_tailk7_replace_schedule_ablation_20260402/warmstart/ckpt_last_cp015_tailk7_70a_replace_zerophase_20260402.pth \
  --out_dir models/__tmp_cp015_tailk7_replace_schedule_ablation_20260402/e3x60 \
  --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 4.2 Historical branch command

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_cp015_tailk7_replace_adapter_falsifier_20260403/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403.json \
  --ckpt_in models/__tmp_cp015_tailk7_replace_schedule_ablation_20260402/warmstart/ckpt_last_cp015_tailk7_70a_replace_zerophase_20260402.pth \
  --out_dir models/__tmp_cp015_tailk7_replace_adapter_falsifier_20260403/e3x60_adapter \
  --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 4.3 This-round sham command

先生成临时 sham config：

```bash
python3 - <<'PY'
import json
from pathlib import Path
src=Path('debug_output/_tmp_cp015_tailk7_replace_adapter_falsifier_20260403/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_lr5e5_from_cp015_tailk7_70a_20260403.json')
out=Path('debug_output/_tmp_adapter_sham_audit_20260410/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_sham_lr0_20260410.json')
out.parent.mkdir(parents=True, exist_ok=True)
d=json.loads(src.read_text())
d['optimizer_param_group_overrides']=[{
  'name':'adapter_frozen',
  'lr':0.0,
  'module_prefixes':['direct_pose_input_adapter']
}]
out.write_text(json.dumps(d, indent=2) + '\n')
print(out)
PY
```

再跑 sham train：

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_adapter_sham_audit_20260410/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_sham_lr0_20260410.json \
  --ckpt_in models/__tmp_cp015_tailk7_replace_schedule_ablation_20260402/warmstart/ckpt_last_cp015_tailk7_70a_replace_zerophase_20260402.pth \
  --out_dir models/__tmp_adapter_sham_audit_20260410/e3x60_adapter_sham \
  --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_sham_lr0_20260410 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

sham eval：

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_adapter_sham_audit_20260410/e3x60_adapter_sham/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_sham_lr0_20260410.pth \
  --rounds 5 --depth 3 --time-index-mode cycle --event_clock auto --phase_reset_source none \
  --contacts_meas_source model --lambda_fusion_apply --log_contacts --export_direct_arm_probe --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_adapter_sham_audit_20260410/eval_model_source/e3x60_adapter_sham \
  --force
```

group summary：

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  tools/phasea_group_summary.py \
  debug_output/_tmp_adapter_sham_audit_20260410/eval_model_source/e3x60_adapter_sham/Walk_F_freerun_cycles.json \
  --cycle_gte 1 --drop_wrap \
  --out debug_output/_tmp_adapter_sham_audit_20260410/eval_model_source/e3x60_adapter_sham_group_summary.json
```

## 5. Results

### 5.1 Stage coverage note

- 本对象只覆盖 `new70b_replace_lowdrift`
- `stage6 native` / `70a native` 在此对象上**没有三臂分叉**，因为三臂共享同一个 `70a` warmstart donor

### 5.2 Mean table

> 本表里 `DirectGeoLocalDeg` 与 `all_ex_root` 同列同值；沿用当前 `phasea_group_summary` 口径

| arm | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | else |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `0.182774` | `0.182774` | `0.379237` | `0.140295` | `0.160056` | `0.093588` |
| `sham` | `0.176999` | `0.176999` | `0.369397` | `0.135399` | `0.156491` | `0.085546` |
| `branch` | `0.175801` | `0.175801` | `0.364820` | `0.134932` | `0.156052` | `0.085013` |

### 5.3 Mean gain decomposition

> lower is better; therefore negative delta means improved

| metric | baseline -> sham | sham -> branch | baseline -> branch | sham share of branch gain |
| --- | ---: | ---: | ---: | ---: |
| `DirectGeoLocalDeg` | `-0.005775` | `-0.001198` | `-0.006972` | `82.8%` |
| `all_ex_root` | `-0.005775` | `-0.001198` | `-0.006972` | `82.8%` |
| `leg` | `-0.009840` | `-0.004578` | `-0.014418` | `68.3%` |
| `nonleg` | `-0.004896` | `-0.000467` | `-0.005363` | `91.3%` |
| `arm` | `-0.003565` | `-0.000439` | `-0.004004` | `89.0%` |
| `else` | `-0.008042` | `-0.000533` | `-0.008575` | `93.8%` |

### 5.4 Tail metrics audit (`p95`)

这一步很关键，因为原始 `adapter` 正结论主要就是靠 `p95` 尾部指标成立。

| arm | all_ex_root p95 | leg p95 | nonleg p95 | arm p95 | else p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `0.622943` | `0.890515` | `0.476347` | `0.554550` | `0.288886` |
| `sham` | `0.609356` | `0.910383` | `0.467279` | `0.549664` | `0.259119` |
| `branch` | `0.611407` | `0.882486` | `0.463465` | `0.550443` | `0.261784` |

对应拆账：

| metric | baseline -> sham | sham -> branch | baseline -> branch | readout |
| --- | ---: | ---: | ---: | --- |
| `all_ex_root p95` | `-0.013587` | `+0.002051` | `-0.011536` | sham **已超过** 全 branch gap；branch 反而比 sham 略差 |
| `arm p95` | `-0.004886` | `+0.000779` | `-0.004107` | sham **已解释超过 100%**；branch 比 sham 略差 |
| `else p95` | `-0.029767` | `+0.002665` | `-0.027102` | sham **已解释超过 100%**；branch 比 sham 略差 |
| `nonleg p95` | `-0.009069` | `-0.003813` | `-0.012882` | sham 解释约 `70.4%`，branch 还有一小段净增益 |
| `leg p95` | `+0.019868` | `-0.027897` | `-0.008029` | sham 本身**更差**，branch 的真实净增益主要集中在这里 |

## 6. Direct answers to the five required questions

### Q1. `baseline -> sham` 的 gap 有多大？

很大，而且方向并不完全中性：

- mean 指标上，`sham` 对全部 5 个主指标都是改善
- `DirectGeoLocalDeg/all_ex_root` mean 已经吃掉 `82.8%` 的全 branch gap
- `nonleg/arm/else` mean 分别吃掉 `91.3% / 89.0% / 93.8%`
- 但在 `leg p95` 上，`sham` 反而更差：`0.890515 -> 0.910383`

因此，这个 `sham` **绝不是中性 control**，它已经足够大到重写解释口径。

### Q2. `sham -> branch` 的净增益还有多少？

mean 上只剩很小净增益：

- `DirectGeoLocalDeg/all_ex_root`: `-0.001198`
- `leg`: `-0.004578`
- `nonleg`: `-0.000467`
- `arm`: `-0.000439`
- `else`: `-0.000533`

真正还明显保留 objective-like 净增益的地方，主要在 **`leg p95`**：

- `sham -> branch`: `0.910383 -> 0.882486`，净改善 `-0.027897`

也就是说，adapter 不是完全没有 objective effect；但这个 effect **高度集中在 leg tail recovery**，而不是广泛覆盖 root/nonleg/arm/else。

### Q3. 历史上宣称的 `baseline -> branch` 改善里，有多少被 sham-level perturbation 吃掉？

如果看 mean 主表：

- `all_ex_root`: `82.8%`
- `leg`: `68.3%`
- `nonleg`: `91.3%`
- `arm`: `89.0%`
- `else`: `93.8%`

如果看原始结论最依赖的 `p95`：

- `all_ex_root p95`: sham 已解释 **超过 100%**
- `arm p95`: sham 已解释 **超过 100%**
- `else p95`: sham 已解释 **超过 100%**
- `nonleg p95`: sham 已解释 `70.4%`
- 只有 `leg p95` 不是 sham 吃掉，而是 branch 真正追回

因此，历史上把 `adapter` 读成“一个全局 interface translation 解”的口径，明显过强。

### Q4. 加入 sham control 后，历史结论是否仍成立？

需要**降级重写**。

原结论：

- `adapter 明显改善，支持 interface translation / geometry mismatch`

加入 sham 后，更准确的重写应是：

- `adapter` 的**大部分 nonleg/root/arm/else 改善**，已经能被 **sham-level structural perturbation** 解释
- `adapter` 仍然保留一个**更窄的真实净效应**：主要集中在 `leg` 尾部恢复，尤其 `leg p95`
- 因此它不再支持“adapter 本身在广义指标上 clearly effective”这类强口径
- 它最多支持一个更弱、更局部的结论：
  - **adapter 可能对 leg-tail-specific recovery 有真实帮助，但历史上看到的多数全局改进并不能直接归因给 adapter objective**

### Q5. 这是否强化了 notail falsifier 的解释力？

**是，但要限口径。**

强化的部分：

- 它强化了一个方法学方向：**历史上很多 branch-side positive readout 可能被 structural perturbation 高估**
- 因而 notail falsifier 所代表的那类“不要轻易把 observed gap 读成 branch/objective 本体效果”的怀疑，现在更有支撑

不该过度延伸的部分：

- 这轮 adapter sham audit **并不直接证明** notail 是 adapter 结果的因果解释
- 它只证明：**branch positive gap 里有很大一块本来就不是 clean objective effect**

## 7. Conclusion status update

### 7.1 Still stands after sham

- `adapter` 不是完全无效
- 它在 `leg` 尾部，尤其 `leg p95`，仍然有真实净增益

### 7.2 Must be downgraded / rewritten

- “`adapter` 在 root/nonleg/arm/else 上给出明确 branch-positive improvement”
- “`adapter` 的全局 freerun 改善可以直接归因于 interface translation objective 本身”

### 7.3 Did sham explain most of the branch gap?

**Yes.**

至少在本对象上：

- mean 主指标里，sham 已解释掉大多数 gap
- 原始最关键的 `all_ex_root p95 / arm p95 / else p95` 改善，sham 已经解释到 **约等于甚至超过** 全 branch gap

## 8. Methodology takeaway

本轮最重要的结论不是 adapter 成败本身，而是方法学；这也是它现在被保留在 retired 目录里的主要原因：

> 以后凡是 `branch / auxiliary / adapter / replace-side structural module` 一类实验，只要历史结论依赖 `baseline -> branch` gap，就**必须默认带 matched sham control**。

默认标准应是：

1. `baseline`
2. `sham`
   - 结构在场
   - 但不给该结构真实训练收益
3. `branch`
   - 真实改动臂

否则：

- 不能再把 `baseline -> branch` 的 observed gap 直接读成 objective effect
- 至少要先回答：里面有多少已经被 sham-level structural perturbation 吃掉

## 9. Repo-change note

本轮：

- **没有改代码**
- 只新增了
  - 1 个临时 sham config
  - 1 份独立 audit record 文档
