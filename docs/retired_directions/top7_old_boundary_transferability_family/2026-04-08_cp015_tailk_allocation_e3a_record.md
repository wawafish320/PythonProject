# 2026-04-08 CP015 tailk allocation E3-A record

> Archived on 2026-04-12.  
> Current role: historical old-boundary `top7` transferability record inside the archived `E0/E1/E2A/E2C/E3A/A1-S1..S5` family, not current design policy.  
> Reader guidance: any `主线` / `推荐` / `默认下一步` / `canonical` wording below is preserved as family-local historical language.

> Last updated: 2026-04-08  
> Scope: **E3-A only** / fixed top7 support / allocation-only readout-first / fixed replace context / deterministic first-forward  
> Machine summary: `debug_output/_tmp_cp015_tailk_allocation_e3a_20260408/summary.json`

## 1. Scope / inherited conclusions

本轮只执行 **E3-A: freeze head, train readouts/adapters first**，直接继承以下结论，不重复证明：

- root cause 不在 planner semantics 主线
- root cause 不在 replace entry 外部 rollout state
- root cause 不在 `contacts_in_t`
- earliest semantic split 在 `direct_pose_head` boundary
- first-step split 最像 whole direct-branch contract mismatch
- `direct_pose_head` 是 earliest boundary / necessary anchor，但不是 standalone sufficient module
- high closure 需要 7-module direct-branch joint contract
- baseline 的 7-module direct-branch transplant 能在 coadapt context work，所以不是 “top7 impossible”
- E0 已显示：当前 top7 path 在最早可用 stage6 exact `epoch013` 就已偏差，不是 final-only 才坏
- E0 已显示：`epoch014/015` 明显优于最终 checkpoint；最坏拐点在 `epoch015 -> stage6_tailfix`
- E1 / E2-A / E2-C 已显示：当前改善主要集中在 `dir_base` / `dir_nonleg`，`dir_leg` 仍几乎不动
- normality probe 在当前 transplant assay 下已经多次不区分；若再次不区分，必须标为 `normality_probe_non_discriminative`

本轮唯一要回答的是：

> 如果不再改 supervision target / support path，而只改 direct branch 内部的 co-adaptation allocation，能否让 produced checkpoint 比 `E1-top3` / `E2A-R` / `E2C-L` 更 replace-transferable，尤其是不再继续牺牲 `dir_leg`？

---

## 2. 为什么在 E2-C 之后优先开 E3，而不是继续 E2-family

`E2C-L` 已经给出一个相当直接的否定信号：

- 固定 top7 support 的 leg-first path 仍没有把 final `70a` 的 `dir_leg` 抬起来
- 为了尝试走 leg-first，还发生了不可接受的 nonleg giveback

因此当前更优先的问题不再是：

- “leg-first path 还不够强吗？”

而更像是：

- **head / adapters / readouts 同时自由 co-adapt，会不会把 early compatibility 拉回 non-transferable basin？**

所以 E3-A 优先测的是 allocation，而不是继续改 support-width 或 loss shaping。

---

## 3. E3-A design / controls / invariants

### 3.1 Single new arm

本轮只新增一个 arm：

- `E3A-RF`: matched readout-first allocation arm

### 3.2 Fixed assay

与 E1 / E2-A / E2-C 完全对齐：

- host replace context: `coadapt_allrot_interface_bestlr_longer_4x_20260406`
- transplant-compatible target: 同 host + baseline replace 的 7-module direct-branch transplant
- mode: deterministic / single-step / first-forward
- offset: `45`
- contacts: baseline replace native same-entry `contacts_in_t`

### 3.3 Strict controls / invariants

保持不变：

- same basetrain pipeline / optimizer / lr / wd / data / seed / init
- same `save_fit_ckpt_epochs = 12-15`
- same `rot_local_tail_reduce = rank_linear_mix`
- same `rot_local_tail_uniform_mix = 0.4`
- same `rot_local_tail_rank_mix = 0.6`
- same stage6 tailfix config
- same final `70a` config
- same seed policy
- same top-level / phase-b / phase-c / phase-d `rot_local_tail_k = 7`

本轮明确不做：

- support-width 改动
- leg/nonleg loss target 改动
- E2-family curriculum 复刻
- loss family / optimizer family / architecture 改动

### 3.4 Trainable-scope plumbing inventory

本轮先盘点现有代码里真正可用的 allocation surface：

- 已有 whole-head freeze surface：`direct_pose_trunk_trainable`
- 本轮最小新增 stage-schedule surface：`direct_pose_head_train_scope`
- 新增 scope mode：
  - `readout_only`
  - `readout_plus_midhead`
  - `full`
  - `frozen`

关键约束是 canonical matched top7 basetrain 的 direct branch 仍是 monolithic `direct_pose_head`：

- 没有单独实例化的 basetrain-time adapters
- 因此 phase-a 的 “readouts/adapters only” 在实际代码里等价为 **readout-only**
- phase-b 再恢复 late hidden block
- phase-c / phase-d 回到 full direct-head co-adaptation

这轮 **不是** `degraded_e3a_variant`：

- `degraded_e3a_variant = false`
- 原因不是 adapter 家族真的在 basetrain 中独立存在，而是现有 schedule surface 已能 cleanly 表达 staged allocation 语义
- 不需要重造 scheduler machinery

---

## 4. Arm inventory

| arm | provenance | schedule | basetrain `epoch014` | final `70a` |
|---|---|---|---|---|
| `E1-top7` | reuse existing | `7 -> 7 -> 7` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk7_stage70a_from_tailfix_20260402/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth` |
| `E1-top3` | reuse existing | `3 -> 3 -> 3` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk3_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth` |
| `E2A-R` | reuse existing | `3 -> 5 -> 7` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk357ramp_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk357ramp_stage70a_from_tailfix_e2a_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk357ramp_stage6tailfix_e2a_20260408.pth` |
| `E2C-L` | reuse existing | `7 -> 7 -> 7 (leg-first -> nonleg expansion)` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_legfirst_nonlegexp_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk7_legfirst_stage70a_from_tailfix_e2c_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_legfirst_stage6tailfix_e2c_20260408.pth` |
| `E3A-RF` | new allocation arm | `7 -> 7 -> 7 (readout-first allocation)` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_e3a_rf_readoutfirst_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk7_e3a_rf_stage70a_from_tailfix_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_e3a_rf_stage6tailfix_20260408.pth` |

E3A-RF 新增中间点：

- stage6 tailfix final: `models/__tmp_cp015_tailk7_e3a_rf_stage6_tailfix_20260408/lr3e4_e8x60_wd1e4_reinit1/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_e3a_rf_20260408.pth`
- stage6 eval: `debug_output/_tmp_cp015_tailk_allocation_e3a_20260408/stage6_tailfix/stage6_freerun/Walk_F_freerun_cycles.json`
- final `70a` eval: `debug_output/_tmp_cp015_tailk_allocation_e3a_20260408/stage70a/eval_model_source/Walk_F_freerun_cycles.json`

---

## 5. Config diff table

| field | `E1-top7` | `E3A-RF` | note |
|---|---:|---:|---|
| `rot_local_tail_k` | `7` | `7` | kept fixed at top7 |
| `phase_b.core.rot_local_tail_k` | `7` | `7` | kept fixed at top7 |
| `phase_c.core.rot_local_tail_k` | `7` | `7` | kept fixed at top7 |
| `phase_d.core.rot_local_tail_k` | `7` | `7` | kept fixed at top7 |
| `phase_a.params.direct_pose_head_train_scope` | `null` | `readouts_adapters_only` | early freeze main head / readout-first allocation |
| `phase_b.params.direct_pose_head_train_scope` | `null` | `readouts_adapters_plus_midhead` | mid ramp / restore late hidden block |
| `phase_c.params.direct_pose_head_train_scope` | `null` | `full` | return to full direct-head co-adaptation |
| `phase_d.params.direct_pose_head_train_scope` | `null` | `full` | late full direct-branch target |
| `save_fit_ckpt_epochs` | `12-15` | `12-15` | fixed |

结论上，这轮唯一新增实质变量就是：

- direct-branch **co-adaptation allocation**

而不是：

- support width
- leg/nonleg loss family

---

## 6. Direct-branch module family mapping table

### 6.1 Basetrain shared-head top7 matched mapping

| family | modules | parameter prefixes | meaning |
|---|---|---|---|
| `head` | `direct_pose_head` | `direct_pose_head.0`, `direct_pose_head.3` | shared hidden layers / main head block |
| `adapters` | none | none | canonical matched basetrain config does not instantiate standalone direct adapters |
| `readouts` | `direct_pose_head` | `direct_pose_head.6` | monolithic direct head final linear readout |

### 6.2 Stage6 / final `70a` transfer-contract mapping

| family | modules | parameter prefixes | meaning |
|---|---|---|---|
| `head` | `direct_pose_head` | `direct_pose_head.` | shared split-head trunk / earliest anchor |
| `adapters` | `direct_pose_arm_proj`, `direct_pose_else_proj` | `direct_pose_arm_proj.`, `direct_pose_else_proj.` | nonleg branch adapters / amplifiers |
| `readouts` | `direct_pose_out_leg`, `direct_pose_out_arm`, `direct_pose_out_else`, `direct_pose_leg_head` | `direct_pose_out_leg.`, `direct_pose_out_arm.`, `direct_pose_out_else.`, `direct_pose_leg_head.` | leg/nonleg direct readout heads |

完整 7-module set：

- `direct_pose_head`
- `direct_pose_leg_head`
- `direct_pose_arm_proj`
- `direct_pose_else_proj`
- `direct_pose_out_leg`
- `direct_pose_out_arm`
- `direct_pose_out_else`

这也解释了一个关键限制：

- E3A-RF basetrain 的 “readouts/adapters first” 在 basetrain-time 实际只能做到 **readout-first**
- 并不意味着 transfer-chain 里的 adapter family 不重要

---

## 7. Basetrain allocation schedule table

| stage | epoch range | top7 support | allocation mode | effective trainable families |
|---|---|---:|---|---|
| top-level | global default | `7` | fixed top7 target | n/a |
| `phase_a` | `1-5` | `7` | freeze main head / readout-first | `readouts` |
| `phase_b` | `6-9` | `7` | restore late hidden block | `head_late_block + readouts` |
| `phase_c` | `10-11` | `7` | full direct-head co-adaptation | `head + readouts` |
| `phase_d` | `12-15` | `7` | late full top7 target | `head + readouts` |

对应 config：

- `config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_e3a_rf_readoutfirst_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408.json`

---

## 8. Stage6 tailfix / final `70a` result summary

### 8.1 E3A-RF own-chain fixed transfer summary

| checkpoint | `out_direct` gap | `dir_base` gap | `dir_leg` gap | `dir_nonleg` gap | aggregate transfer score |
|---|---:|---:|---:|---:|---:|
| `E3A-RF stage6 tailfix final` | `0.562763` | `1.215861` | `2.661159` | `0.903364` | `-0.063878` |
| `E3A-RF final 70a` | `0.468347` | `1.095939` | `2.137026` | `0.870839` | `0.081860` |

链内读法：

- `70a` 相比 E3A-RF 自身 stage6 确实改善了 aggregate（`+0.145737`）
- 但即使到了 final `70a`，aggregate 仍低于 `E1-top3 = 0.232733`、`E2A-R = 0.156738`、`E2C-L = 0.128023`
- stage6 尤其明显地把 `dir_leg` 拉坏：`dir_leg_closure = -0.323468`

### 8.2 Native freerun group summary

| checkpoint | `all_ex_root` | `leg` | `nonleg` | `arm` | `else` |
|---|---:|---:|---:|---:|---:|
| `E3A-RF stage6 tailfix final` | `0.279978` | `0.696402` | `0.189940` | `0.219516` | `0.120035` |
| `E3A-RF final 70a` | `0.217346` | `0.524547` | `0.150925` | `0.175026` | `0.093958` |

这说明：

- E3A-RF 自身链内 native freerun 是改善的
- 但主判据仍然是 fixed replace-transferability，而不是 native freerun

---

## 9. Fixed transfer assay table

### 9.1 Final `70a` raw gaps

| arm | `out_direct` gap | `dir_base` gap | `dir_leg` gap | `dir_nonleg` gap |
|---|---:|---:|---:|---:|
| `E1-top7` | `0.469847` | `1.293004` | `2.010747` | `1.137816` |
| `E1-top3` | `0.475508` | `0.794442` | `1.931073` | `0.548684` |
| `E2A-R` | `0.461871` | `0.960091` | `2.021634` | `0.730569` |
| `E2C-L` | `0.469371` | `0.997315` | `2.120300` | `0.754508` |
| `E3A-RF` | `0.468347` | `1.095939` | `2.137026` | `0.870839` |

### 9.2 Final `70a` closure ratios vs transplant-compatible target

| arm | `out_direct` closure | `dir_base` closure | `dir_leg` closure | `dir_nonleg` closure | aggregate |
|---|---:|---:|---:|---:|---:|
| `E1-top7` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `E1-top3` | `-0.012050` | `0.385584` | `0.039624` | `0.517774` | `0.232733` |
| `E2A-R` | `0.016976` | `0.257472` | `-0.005414` | `0.357920` | `0.156738` |
| `E2C-L` | `0.001012` | `0.228683` | `-0.054484` | `0.336881` | `0.128023` |
| `E3A-RF` | `0.003192` | `0.152408` | `-0.062802` | `0.234640` | `0.081860` |

### 9.3 Delta summary

`E3A-RF - E1-top7`:

- `aggregate_transfer_score`: `+0.081860`
- `dir_base_closure_ratio`: `+0.152408`
- `dir_leg_closure_ratio`: `-0.062802`
- `dir_nonleg_closure_ratio`: `+0.234640`
- `out_direct_closure_ratio`: `+0.003192`

`E3A-RF - E1-top3`:

- `aggregate_transfer_score`: `-0.150874`
- `dir_base_closure_ratio`: `-0.233176`
- `dir_leg_closure_ratio`: `-0.102426`
- `dir_nonleg_closure_ratio`: `-0.283134`
- `out_direct_closure_ratio`: `+0.015242`

`E3A-RF - E2A-R`:

- `aggregate_transfer_score`: `-0.074879`
- `dir_base_closure_ratio`: `-0.105064`
- `dir_leg_closure_ratio`: `-0.057388`
- `dir_nonleg_closure_ratio`: `-0.123280`
- `out_direct_closure_ratio`: `-0.013783`

`E3A-RF - E2C-L`:

- `aggregate_transfer_score`: `-0.046163`
- `dir_base_closure_ratio`: `-0.076275`
- `dir_leg_closure_ratio`: `-0.008318`
- `dir_nonleg_closure_ratio`: `-0.102241`
- `out_direct_closure_ratio`: `+0.002181`

---

## 10. `dir_leg`-focused interpretation

本轮最核心的问题是：`dir_leg` 有没有终于从 prior E2-family 的“几乎不动”模式中脱离？

答案是否定的，而且更差。

final `70a` 上：

- `E3A-RF dir_leg gap = 2.137026`
- `E1-top7 dir_leg gap = 2.010747`
- `E1-top3 dir_leg gap = 1.931073`
- `E2A-R dir_leg gap = 2.021634`
- `E2C-L dir_leg gap = 2.120300`

对应 `dir_leg` closure 也同样更差：

- vs `E1-top7`: `-0.062802`
- vs `E1-top3`: `-0.102426`
- vs `E2A-R`: `-0.057388`
- vs `E2C-L`: `-0.008318`

因此这轮不能判成：

- “首次出现明确 leg closure improvement”
- “至少不再继续恶化”

更准确的读法是：

- E3A-RF 没有建立更好的 leg-side transfer-compatible basin
- final `70a` 的 `dir_leg` 甚至比 `E2A-R` / `E2C-L` 更差

---

## 11. Nonleg retention / giveback summary

这轮的 nonleg 保留也不理想。

相对 `E1-top3`：

- `dir_base` closure retention = `0.395266`
- `dir_nonleg` closure retention = `0.453170`
- `dir_base` closure delta = `-0.233176`
- `dir_nonleg` closure delta = `-0.283134`

相对 `E2A-R`：

- `dir_base` closure retention = `0.591941`
- `dir_nonleg` closure retention = `0.655565`
- `dir_base` closure delta = `-0.105064`
- `dir_nonleg` closure delta = `-0.123280`

相对 `E2C-L`：

- `dir_base` closure delta = `-0.076275`
- `dir_nonleg` closure delta = `-0.102241`

结论：

- `unacceptable_nonleg_giveback = true`
- E3A-RF 不仅没有换来 leg-side closure
- 还回吐了 prior arms 已拿到的相当一部分 `dir_base` / `dir_nonleg` gain

---

## 12. Replace-normality summary

本轮 replace-normality readout 再次完全不区分：

- host-native bad reference
- baseline-transplant target
- `E1-top7`
- `E1-top3`
- `E2A-R`
- `E2C-L`
- `E3A-RF`

所有 signature 完全相同：

- `label = plan_compensatory`
- `plan_over_direct_sensitivity = 0.385554`
- `plan_zero_delta_geolocal_deg = 0.145226`
- `direct_zero_delta_geolocal_deg = 3.951890`
- `meas_zero_delta_geolocal_deg = 0.085150`

因此必须明确标记：

- `normality_probe_non_discriminative`

本轮对 “是否更正常进入 replace” 的判断，只能继续保守依赖 fixed transferability；不应过度解释 normality probe。

---

## 13. Proxy telemetry summary

`direct_pose_head.0` input-block telemetry 继续记录，但本轮仍只应作为 coarse concurrent readout。

| arm | direct `/dim` | plan `/dim` | meas `/dim` | `plan/direct` | `plan/meas` | `plan/(direct+meas)` |
|---|---:|---:|---:|---:|---:|---:|
| `E1-top7` | `2.038038` | `2.011084` | `1.948133` | `0.986775` | `1.032313` | `0.504515` |
| `E1-top3` | `2.033001` | `2.009971` | `2.028098` | `0.988672` | `0.991062` | `0.494933` |
| `E2A-R` | `2.034409` | `2.009031` | `2.024326` | `0.987525` | `0.992445` | `0.494989` |
| `E2C-L` | `2.033732` | `2.010425` | `2.023941` | `0.988540` | `0.993322` | `0.495463` |
| `E3A-RF` | `2.036670` | `2.010439` | `1.967798` | `0.987121` | `1.021669` | `0.502049` |

读法：

- E3A-RF 的 proxy 位置仍落在 prior arms 的狭窄带内
- 它没有给出能提前解释 fixed transfer worsening 的 leading signal
- 这再次符合已有结论：`direct_pose_head.0` proxy useful but not leading indicator / not root cause

---

## 14. Interpretation

本轮应判为 **Case 3**：

- `E3A-RF` 明显优于 `E1-top7`
- 但不如 `E1-top3`
- 也不如 `E2A-R`
- 也不如 `E2C-L`
- `dir_leg` 不仅没有明确抬升，反而比 `E2A-R` / `E2C-L` 更差
- 同时发生了不可接受的 nonleg giveback

因此当前不能支持：

- `co_adaptation_allocation_is_missing_lever = false`
- `top7_viable_under_staged_allocation_compatible_coadaptation = false`

更克制、也更准确的结论是：

- **allocation 有可能仍然相关**
- 但当前 first allocation arm (`E3A-RF`) 还不足以建立更好的 transfer-compatible basin
- 现阶段没有证据支持 “只要 readout-first staged allocation，top7 就可行”

---

## 15. 如果不够，为什么下一步应是 `E3-B` 而不是 `E4`

下一步更应该优先开：

- `E3-B`

而不是直接跳：

- `E4`

原因是：

1. 当前失败仍然是 **first-order allocation ordering** 问题，而不是 second-order optimizer tuning 问题  
   `E3A-RF` 已经说明 readout-first ordering 不 work，但这不等于 head-first / alternate allocation ordering 不 work。

2. `E4` 的 knob（LR scale / weighting / regularization / clipping）信息价值更低  
   在 “allocation ordering 是否强因果” 还没回答完时就转 `E4`，容易把问题过早改写成调参问题。

3. normality probe 继续 non-discriminative  
   这意味着当前真正有信息量的主读数仍是 fixed transferability；最值得继续压榨的是另一个 allocation ordering，而不是 second-order tuning。

所以本轮结论是：

- **next_step_recommendation = `E3-B`**
