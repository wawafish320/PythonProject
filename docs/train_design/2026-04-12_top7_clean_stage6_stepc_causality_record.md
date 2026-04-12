# 2026-04-12 top7 clean stage6-StepC 因果验证记录

> 范围：针对 canonical `top7` donor 的 clean handoff 因果验证  
> 决策目标：判断 `70a/replace` 的 early downstream 不干净，主要来自旧 `stage6` handoff / boundary contract，还是来自 `top7` donor 本身  
> 固定 donor：`cp015_tailk7_rankmix_tw020 ... ckpt_epoch_014.pth`  
> 固定评估口径：model-source contract，`N=5 / limited-N`，Step A 必要但不充分，排序按 Step B'  
> 非目标：不重跑 basetrain，不做 tail-k sweep，不做 LR sweep，不改架构，不做 recipe search

## 1. 本轮为什么要做

上一轮已经确认：

- `old stage6 handoff -> downstream StepC compatibility` 相比 `old stage6 handoff -> old-cut downstream`，从 `70R` 开始存在明显 rescue
- 但 `70a` / `replace` 在 pseudo-StepC lane 里仍然不干净
- 当时最合理的解释是 `two-layer interaction`
  - `top7` donor 自身带着 early downstream 负担
  - 旧 `stage6` handoff / old boundary fragmentation 又进一步放大了这个负担

但上一轮还不是 clean `stage6-StepC` chain。它的 Phase 1 仍然建立在旧 `stage6` tailfix handoff 上，只是 downstream lane 打开了 StepC compatibility。

所以这轮只回答一个更精确的问题：

> 如果 `stage6` 本身就按 StepC unified leg terminal 语义产出 handoff，`70a / replace / 70R` 会不会比当前 pseudo-StepC lane 更进一步变好？

换句话说，这轮不是再证明 “StepC downstream 比 old-cut 好”，而是验证：

> **clean `stage6-StepC` handoff 本身，是不是 early drag 的关键因果来源。**

## 2. 验证设计

### 2.1 三条 lane

本轮把 lane 明确拆成三类：

| lane | 含义 | 动作 |
| --- | --- | --- |
| `O` | `old stage6 handoff -> old-cut downstream` | 只复用 reference |
| `P` | `old stage6 handoff -> downstream StepC compatibility` | 只复用 reference |
| `C` | `top7 basetrain donor -> clean stage6-StepC handoff -> downstream StepC` | 本轮新跑 |

本轮唯一允许的 donor 为：

- ckpt: `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth`
- config: `config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json`
- donor report: `debug_output/_tmp_cp015_tailk7_rankmix_tw020_20260401/final_report.json`

Lane `O/P` 的 reference 直接复用：

- `debug_output/_tmp_top7_posttrain_oldcut_vs_stepc_20260412/summary.json`
- `debug_output/_tmp_top7_posttrain_oldcut_vs_stepc_20260412/decision.md`

### 2.2 Phase 划分

Phase 0：

- 生成 / 确认 clean `stage6-StepC` handoff
- 验证 provenance 来自同一个 canonical `top7` donor
- 验证 layout：
  - `has_direct_pose_leg_terminal = true`
  - `has_direct_pose_out_leg = false`
  - `cfg_direct_pose_stepc_unified_leg_terminal = true`

Phase 1：

- 跑 `C-70a`
- 跑 copy-only `replace` warmstart 和 `C-replace`
- 跑 `C-70R`
- 比较两组：
  - `C` vs `O`
  - `C` vs `P`

Phase 2 只在满足以下条件时继续：

- `C-70R` 明确优于 `O-70R`
- `C-70R` 不弱于 `P-70R`
- 最好 `C-70a` 或 `C-replace` 相对 `P` 已出现改善

本轮满足继续条件，因此继续跑了 locked chain：

- `C-71`
- `C-72`
- `C-lambda`

### 2.3 判定口径

全程使用 model-source contract：

- Step A gate 仍然是 necessary-but-not-sufficient
- Step B' 排序规则：
  - primary：`all_ex_root_mean`
  - tie-break1：当 `|delta_mean| < 0.002` 时看 `all_ex_root_p95`
  - tie-break2：当 `|delta_p95| < 0.01` 时看 `leg_mean`
  - hard reject：固定 incumbent 的 `nonleg_p95` 阈值
- incumbent 固定为 `current_bad.teacher_x_gt`
- 所有结论都带 `N=5 / limited-N`

## 3. Artifact 与实现说明

### 3.1 本轮新增 focused runner

本轮新增 runner：

- `tools/run_top7_clean_stage6_stepc_chain.py`

职责严格限定为：

- 从 canonical `top7` donor 生成 clean `stage6-StepC` handoff
- 复用 Lane `O/P` reference artifacts
- 跑 Lane `C`
- 执行 model-source eval 和 group summary
- 输出 comparison / decision

产出汇总：

- `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/summary.json`
- `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/summary.md`
- `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/decision.md`

### 3.2 本轮唯一必要的 runtime fix

`C-70R` 第一次执行时撞到一个 wrapper runtime bug：

- `tools/run_posttrain_nonleg_trunk_ablation.py` 还在解包旧版 `_build_posttrain_model_from_ckpt(...)` 返回签名

因此做了一个最小修复：

- 把 wrapper 的解包改为包含 `direct_pose_use_phase_z` 与 `direct_pose_phase_z_mode`
- 同时把这两个字段传给 `_save_posttrain_outputs(...)`

这里没有改 `train/` 的 StepC 语义逻辑，只是修了一个现成 wrapper 的兼容性问题。

## 4. Phase 0：clean stage6-StepC handoff

clean stage6 artifact：

- ckpt: `models/__tmp_top7_clean_stage6_stepc_chain_20260412/stage6_stepc_handoff/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stepc_clean_20260412.pth`
- config: `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_stage6_tailfix_top7_clean_stepc_20260412.json`
- eval: `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/stage6_stepc_handoff/eval_model_source/Walk_F_freerun_cycles.json`
- group summary: `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/stage6_stepc_handoff/eval_model_source_group_summary.json`

layout check：

| check | value |
| --- | --- |
| `has_direct_pose_leg_terminal` | `true` |
| `has_direct_pose_out_leg` | `false` |
| `cfg_direct_pose_stepc_unified_leg_terminal` | `true` |

stage6 指标：

| handoff | `all_ex_root_mean` | `all_ex_root_p95` | `leg_mean` | `leg_p95` | `nonleg_p95` | `arm_mean` | `arm_p95` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old stage6 tailfix | `0.250873` | `0.836182` | `0.566078` | `1.324971` | `0.604531` | `0.201131` | `0.687248` |
| clean stage6-StepC | `0.252103` | `0.869592` | `0.590828` | `1.389266` | `0.642695` | `0.198233` | `0.699600` |

这一阶段最关键的读法是：

- clean `stage6-StepC` **不是** 单纯靠 stage6 指标变好来解释后续收益
- 相反，它在若干 direct 指标上还略差于 old stage6
- 所以后续如果 downstream 变好，解释重点必须放在：
  - handoff contract
  - layout semantics
  - boundary compatibility

而不是放在 “stage6 本身先数值变强了”

## 5. Phase 1 结果

下面的 delta 采用 `candidate - reference`，负值更好。`C` 表示新的 clean `stage6-StepC` lane。

| compare | Δ `all_ex_root_mean` | Δ `all_ex_root_p95` | Δ `leg_mean` | Δ `leg_p95` | Δ `nonleg_p95` | Step B' |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `C-70a` vs `O-70a` | `+0.026282` | `+0.092572` | `+0.085434` | `+0.227605` | `+0.038794` | `lose` |
| `C-replace` vs `O-replace` | `-0.009474` | `-0.066715` | `-0.058892` | `-0.145397` | `+0.016352` | `win` |
| `C-70R` vs `O-70R` | `-0.214888` | `-0.898800` | `-0.740512` | `-1.770212` | `-0.371724` | `win` |
| `C-70a` vs `P-70a` | `-0.014601` | `+0.059567` | `+0.036714` | `+0.071479` | `-0.083009` | `win` |
| `C-replace` vs `P-replace` | `-0.013567` | `-0.075329` | `-0.041969` | `-0.165128` | `-0.004770` | `win` |
| `C-70R` vs `P-70R` | `-0.124858` | `-0.477006` | `-0.386469` | `-1.059974` | `-0.233862` | `win` |

Phase 1 的所有 Step B' verdict 都由 `all_ex_root_mean` 的 primary 直接触发；没有触发 tie-break，也没有 hard reject。

### 5.1 `70a` 阶段分析

`C-70a` 是一个 **mixed 但有方向性** 的结果：

- 相比 `O-70a`，它仍然输：
  - `all_ex_root_mean`: `+0.026282`
  - `leg_mean`: `+0.085434`
  - `leg_p95`: `+0.227605`
- 相比 `P-70a`，它已经赢了 Step B' primary：
  - `all_ex_root_mean`: `-0.014601`
  - `nonleg_p95`: `-0.083009`
- 但相对 `P-70a`，它还不是全维度更好：
  - `all_ex_root_p95`: `+0.059567`
  - `leg_mean`: `+0.036714`
  - `leg_p95`: `+0.071479`

这一阶段的结论是：

- clean stage6-StepC 已经开始救回 pseudo-StepC 的 early drag
- 但 raw `70a` 仍然没有完全干净
- 这说明 residual donor burden / early recipe burden 还残留一部分
- 因此不能把结论写成“全部都是 boundary 问题”

### 5.2 `replace` 阶段分析

`C-replace` 是本轮 early chain 里第一个 **明确干净转正** 的阶段：

- 相比 `O-replace`：
  - `all_ex_root_mean`: `-0.009474`
  - `all_ex_root_p95`: `-0.066715`
  - `leg_mean`: `-0.058892`
  - `leg_p95`: `-0.145397`
- 相比 `P-replace`：
  - `all_ex_root_mean`: `-0.013567`
  - `all_ex_root_p95`: `-0.075329`
  - `leg_mean`: `-0.041969`
  - `leg_p95`: `-0.165128`

这里的关键因果含义是：

- 一旦 clean stage6-StepC handoff 接上 locked replace recipe，pseudo-StepC lane 里的旧 handoff 包袱就明显缩小
- 这已经不是 “70R 才开始 rescue”
- `replace` 本身就出现了 clean extra rescue

### 5.3 `70R` 阶段分析

`C-70R` 是 Phase 1 的决定性证据：

- 相比 `O-70R`：
  - `all_ex_root_mean`: `-0.214888`
  - `all_ex_root_p95`: `-0.898800`
  - `leg_mean`: `-0.740512`
  - `leg_p95`: `-1.770212`
  - `nonleg_p95`: `-0.371724`
- 相比 `P-70R`：
  - `all_ex_root_mean`: `-0.124858`
  - `all_ex_root_p95`: `-0.477006`
  - `leg_mean`: `-0.386469`
  - `leg_p95`: `-1.059974`
  - `nonleg_p95`: `-0.233862`

这一阶段说明：

- clean stage6-StepC 不只是“不弱于 pseudo-StepC”
- 它在 `70R` 已经明显强于 `P`
- 因此 Phase 2 的 locked continuation 是合理的

这也意味着旧的表述需要更新：

- 旧说法：StepC 从 `70R` 开始才变清楚
- 新说法：
  - `70a`：部分 rescue，但仍 mixed
  - `replace`：已经出现 clear rescue
  - `70R`：出现 strong rescue

## 6. Phase 2 结果

由于 Phase 1 满足继续条件，因此继续跑了 locked `71/72/lambda` chain。

| compare | Δ `all_ex_root_mean` | Δ `all_ex_root_p95` | Δ `leg_mean` | Δ `leg_p95` | Δ `nonleg_p95` | Step B' |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `C-71` vs `O-71` | `-0.148604` | `-0.504866` | `-0.327336` | `-0.727575` | `-0.422427` | `win` |
| `C-72` vs `O-72` | `-0.146594` | `-0.489268` | `-0.316029` | `-0.741765` | `-0.422427` | `win` |
| `C-lambda` vs `O-lambda` | `-0.146594` | `-0.489268` | `-0.316029` | `-0.741765` | `-0.422427` | `win` |
| `C-71` vs `P-71` | `-0.094337` | `-0.315494` | `-0.200226` | `-0.365294` | `-0.289998` | `win` |
| `C-72` vs `P-72` | `-0.110947` | `-0.381435` | `-0.293658` | `-0.613466` | `-0.289998` | `win` |
| `C-lambda` vs `P-lambda` | `-0.110947` | `-0.381435` | `-0.293658` | `-0.613466` | `-0.289998` | `win` |

Phase 2 的所有 Step B' verdict 同样都由 `all_ex_root_mean` 的 primary 直接触发；没有 tie-break，也没有 hard reject。

### 6.1 `71` 阶段分析

`C-71` 说明 `70R` 的收益不是单阶段偶然现象：

- 同时优于 `O-71` 和 `P-71`
- aggregate、leg、nonleg p95 一起改善
- 因此可以排除 “只是 70R 恰好有效” 的解释

### 6.2 `72` 阶段分析

`C-72` 持续维持 clean handoff 的收益：

- 相比 `P-72`：
  - `leg_mean`: `-0.293658`
  - `leg_p95`: `-0.613466`
  - `nonleg_p95`: `-0.289998`

说明：

- clean stage6-StepC handoff 与 locked `72` recipe 是兼容的
- 这条链并不是靠“牺牲 nonleg 换 leg”来获胜

### 6.3 `lambda` 阶段分析

本轮 artifact 里，`C-lambda` 的 direct-group 指标与 `C-72` 相同，并且相对 `O/P lambda` 都继续保持 win。

因此这里更合理的读法是：

- `lambda` 主要承担 chain closure 作用
- 它说明 clean handoff 的收益没有在 late chain 崩掉
- 不应过度解读成单独的 lambda-specific mechanism 胜利

## 7. 因果解释

### 7.1 核心回答

clean `stage6-StepC` 相比 pseudo-StepC，是否进一步 rescue 了 `70a/replace`？

- 是
- 而且程度可以写成：**明显 rescue**

依据是：

- `C-70a` 已经赢 `P-70a`
- `C-replace` 已经同时赢 `O-replace` 与 `P-replace`
- `C-70R` 对 `O/P` 都是大幅 clean win

但这里需要保留一个限定：

- `70a` 仍然输给 `O-70a`
- 所以不能写成“top7 完全没有 intrinsic downstream burden”

更准确的说法应是：

- 旧 `stage6` handoff / fragmented boundary contract 是这轮实验暴露出的 dominant early drag
- 但在 raw `70a` 仍能看到一部分 residual donor / early recipe burden

### 7.2 这轮之后更支持哪种解释

这轮会把 preferred explanation 从：

- `two-layer interaction`

明显推向：

- **`old-stage6-handoff / downstream-boundary-induced` 为主**

更精确一点可以写成：

> `two-layer interaction` 仍然可以作为 `70a` 局部残留现象的 caveat，但它已经不再是最好的主叙事。clean stage6-StepC handoff 在 `replace` 就开始清晰缩小 early drag，并在 `70R -> lambda` 对 `O/P` 全面获胜，因此主因果来源更像是旧 stage6 handoff / boundary contract，而不是一个根本不可用的 top7 donor。

### 7.3 `top7` 的更精确表述

不要写：

- `top7 太 aggressive`

应该写成：

> `top7` 超出了 legacy stage6 handoff / old boundary contract 在 early downstream 能干净吸收的范围。在旧 handoff 下它看起来像 compromised donor；但当 handoff 本身改成 clean StepC unified-leg-terminal 语义后，`70a/replace/70R` 的 downstream regression 明显收缩。

### 7.4 `top3` 的更精确表述

不要写：

- `top3 天然最优`

更准确的表述是：

> `top3` 更像是旧 stage6 handoff + old boundary contract 仍能 handle 的 donor 范围。它是 old-boundary-compatible operating range，而不一定是 handoff contract 修正之后的最优 semantic scope。

## 8. 最终判断

这轮 clean causality test 支持：

- `old-stage6-handoff / downstream-boundary-induced` 是主要解释
- `top7` 不应再被粗糙地写成“太 aggressive”
- 旧 handoff / boundary contract 让 `top7` 在 early downstream 看起来比实际更坏

因此这轮之后不建议继续扩训练链：

- 不需要因此重跑 basetrain
- 不需要做新的 tail-k sweep
- 不需要做新的 LR sweep
- clean main chain 已经跑到 `lambda`

如果还要补一点置信度，唯一值得做的最小下一步应该是 eval-only：

- 对现有 `O/P/C` ckpt 增加 model-source eval 次数
- 不再打开新的 recipe search
