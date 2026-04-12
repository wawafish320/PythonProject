# CP015 upstream：`basetrain -> Stage6` 最小 handoff probe 与 export contract 设计

> Status: archived legacy upstream / handoff / control record
> Reader note: this file belongs to the old-boundary upstream-control investigation; any `current`, `default`, `canonical`, `recommend`, or `mainline` wording below is historical context, not present-tense repo policy.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

> Last updated: 2026-03-26
> 目标：在不改 canonical posttrain 文档、不继续调 downstream `71/72`、不启动新长链训练的前提下，定义 upstream 第一支可执行工作：
> 1. 一个轻量 `Stage6-only probe`
> 2. 一个最小 `basetrain true-sample export contract`
>
> 这两者只服务于后续 selector upgrade，不在本轮扩展成新的 train objective 或 full-chain replay 计划。

关联输入：

- `debug_output/_tmp_cp015_stage6_checkpoint_correlation_20260323/summary.md`
- `debug_output/_tmp_cp015_stage6_exit_tail_map_20260323/summary.md`
- `docs/posttrain_pipeline.md`
- `debug_output/_tmp_cp015_next_basetrain_target_20260324/stage6_handoff_recommendation.md`
- `debug_output/_tmp_cp015_next_basetrain_target_20260324/minimal_export_fields.csv`

---

## 0) TL;DR

1. 当前 upstream 的核心问题不是“没有更好的 basin”，而是 **已有 `Stage6-friendly mid-training basin`，但 current selector / proxy 选不稳也解释不稳**。
2. 本轮最小目标不应是改训练，而应是先把 `basetrain -> Stage6` interface 看清楚：
   - basetrain 侧先补齐 true-sample export；
   - Stage6 侧只跑 light probe，不进 `70a -> 70R -> 71 -> 72 -> lambda`。
3. `basetrain export` 的职责是 **预筛 / audit / window finding**，不是 final ranker。
4. final promote 仍应由 **exact Stage6 actual** 决定；若 exact Stage6 已存在，则它优先级高于 basetrain-side inference。

---

## 1) 背景与问题定义

`2026-03-26` 的新结论已经把优先级重排讲清楚了：

- 在当前锁定的 downstream stage semantics 下，继续只调 `71/72` 的 LR / steps，收益已经明显递减；
- official chain 仍然是当前最优可用链路；
- 真正更大的杠杆已经转到 upstream，具体是 `basetrain -> Stage6` handoff 质量。

更关键的是，`debug_output/_tmp_cp015_stage6_checkpoint_correlation_20260323/summary.md` 已经证明：

- `fixedsched` 的最佳 Stage6 handoff 在 `ep014`
- `phasebextend` 的最佳 Stage6 handoff 在 `ep012`
- `phasebcp018` 的最佳 Stage6 handoff 在 `ep014`
- 三条 run 都 **不是** late `ep019` 最优

因此这轮要解决的问题不是：

- “basetrain late mean 再压一点”

而是：

- “如何稳定找到并验证更 `Stage6-friendly` 的 mid-training basin”

这正是 P0 / P1 交界处该做的最小工作。

---

## 2) selector upgrade 的目标

本轮 selector upgrade 的目标应明确写成：

> 不再问“哪个 ckpt 的 basetrain freerun scalar 最好”，
> 而是问“哪个 ckpt 在锁定 handoff contract 下最适合作为 `Stage6` 起点”。

这意味着第一步不应该直接改训练逻辑，而应该先冻结一个新的接口定义：

1. `best_handoff` 的语义先定义为 **offline promoted artifact**
   - 不是立即改写训练主循环里的 canonical selector
   - 先把判别口径做对
2. basetrain 侧只承担 **candidate generation + prefilter**
   - 找 decision window
   - 淘汰明显坏方向
3. exact Stage6 侧承担 **final ranking**
   - 尤其在 near-tie 候选之间
   - 不再让 basetrain proxy 单独拍板

本轮的 deliverable 也应围绕这个目标收缩：

- 一个最小 `Stage6-only probe`
- 一个最小 basetrain export schema
- 一条清晰的 promote 顺序

而不是完整 selector v2 implementation。

---

## 3) 为什么 current proxy 不够

当前问题不是“完全没有 selector”，而是 **现有 selector / proxy 仍不足以选出 `Stage6-friendly mid-training basin`**。

### 3.1 current candidate prior 仍偏向 pose/free-run scalar

`tools/run_basetrain_handoff_selector.py` 当前候选发现与扩展逻辑仍以：

- `best_free`
- `last`
- `best_teacher`
- `proxy_topk` epoch

为中心；其中 proxy epoch scan 的排序仍主要依赖：

- `GeoDegSlope`
- `GeoDriftSlopeProxy`
- freerun leg / teacher geo local

这意味着 candidate pool 虽然已经比“只看 `best_free`”前进了一步，但搜索 prior 仍旧偏向 basetrain 内部 pose/free-run scalar，而不是 handoff-friendly quality。

### 3.2 exact checkpoint correlation 已证明 current proxy 相关性不够

`debug_output/_tmp_cp015_stage6_checkpoint_correlation_20260323/summary.md` 的结论已经足够强：

- `ep012` 的更优 Stage6 basin 会被 current proxy 漏掉
- late `phasebcp018` 的 tail-heavy failure 会被 current proxy 弱化
- strongest proxy correlation 也只有 modest / unstable 量级

因此 current proxy 最多能做“粗筛”，不能稳定做 final selector。

### 3.3 当前很多 tail readout 仍是 `curve_proxy`，不是真实 sample tail

checkpoint correlation 明确指出：

- 当前 Phase-1 `p95` 很多仍来自 `GeoLocalDegCurveBones`
- 它们不是与 `Stage6` 同口径的 true-sample `p95`

这会直接带来两个问题：

1. broad floor shift 和 tail-heavy failure 容易混在一起；
2. arm / nonleg 的 late tail risk 会在 basetrain mean 或 curve proxy 里被冲淡。

`debug_output/_tmp_cp015_stage6_exit_tail_map_20260323/summary.md` 的 `phasebcp018` 就是这个典型反例：

- basetrain late arm mean 看起来不差
- exact Stage6 arm / nonleg p95 却最差

### 3.4 current selector v1 偏 semantic，对 calibration / dynamic range 保护不够

`debug_output/_tmp_d0_phasea8_selector_backfill_20260322/plan_metric_audit.md` 已给出额外补充：

- current selector v1 主要由 semantic alignment metrics 主导；
- 它没有显式保护 contact-plan 的 calibration / dynamic range；
- `teacher-free calibration gap` 仍缺失。

这意味着 current selector v1 即使在语义上不差，也可能挑到一个：

- `ContactErrAbsMean` 看似还行
- 但 plan-stack continuous calibration 不够 handoff-ready

的 checkpoint。

### 3.5 current `Stage6` probe 也还不够完整

`tools/run_stage6_probe_selector.py` 已经是很好的 light probe 原型，但它当前 summary 只把以下指标作为单值排序主体：

- `all_ex_root`
- `leg`
- `nonleg`

而没有把当前最关键的 hotspot / tail 信息显式列为 gate：

- `foot_l/ball_l@SIC12-15`
- `calf_r@SIC2-4`
- `arm/nonleg true_p95`

所以它已经能做 “light Stage6 run”，但还不是足够 sharp 的 final promotion table。

### 3.6 结论

当前 proxy / selector 的问题可以浓缩成一句话：

> 它们已经能帮助缩小候选池，但还不能可靠地区分
> “mid-training Stage6-friendly basin”
> 与
> “late broad/tail failure 在 basetrain 侧被伪装成还行的候选”。

---

## 4) `Stage6-only probe` 最小设计

## 4.1 目标

`Stage6-only probe` 的目标不是证明 full chain 最终谁赢，而是更小、更快地回答：

> “这个 basetrain ckpt 进入 canonical `Stage6` 后，接口是否已经偏斜？”

因此它必须：

- 保持 Stage6 canonical semantics 不变；
- 只跑 `Stage6`，不自动接 downstream `70a/70R/71/72/lambda`；
- 输出足以做 promote / reject 的 exit 指标；
- 只把 init 指标作为诊断，不拿 init 替代 exit。

## 4.2 输入

最小输入应固定为：

1. 候选 ckpt 列表
   - 优先来自已有 saved selectors 与 dense epoch window
   - 当前 CP015 推荐窗口应先锁在 `ep12-15`
   - `ep019` 仅保留为 late failure sentinel，不应做默认 promote 候选
2. canonical Stage6 config
   - `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
3. locked eval contract
   - `teacher = validate/teacher_batches/Walk_F_teacher.json`
   - `contacts_meas_source = pretrain_contact`
   - `affine = affine_mix08 canonical stats`
   - `encoder_bundle = canonical motion encoder`
   - `rounds = 5`
   - `depth = 3`
   - `time_index_mode = cycle`
   - `event_clock = auto`
   - `phase_reset_source = none`
   - mask: `cycle >= 1` and `drop_wrap = true`

## 4.3 最小输出指标

最小 probe 建议分三层输出。

### A. init diagnostics

这些指标只用于看 interface skew，不用于替代 final rank：

- `step1 dir_leg_base`
- `step1 dir_nonleg_base`
- `step1 leg_over_nonleg`
- `head20 dir_leg_base`
- `head20 dir_nonleg_base`
- `head20 leg_over_nonleg`
- 可选：`step1/head20 arm_over_else`

这些指标 `tools/run_stage6_probe_selector.py` 已能直接产出。

### B. exit gate metrics

这些是最小排序主表，应该显式出现在 summary 里：

- `all_ex_root`
- `leg`
- `nonleg`
- `foot_l/ball_l@SIC12-15`
- `calf_r@SIC2-4`

理由：

- 前三项保留 broad exit quality；
- 后两项把当前最顽固的 localized hotspot 直接纳入 gate；
- 这样可以避免 probe 只看 broad mean，把真正 downstream-sensitive pocket 放掉。

### C. exit tail guardrails

这些不一定要成为主排序第一列，但至少要成为 reject / tie-break 信息：

- `leg true_p95`
- `nonleg true_p95`
- `arm true_p95`
- 可选：对应 group 的 `trim_top10`

理由：

- current proxy 最大漏检点之一就是 broad-vs-tail 错分；
- 如果 probe 不把 tail guardrail 补上，`phasebcp018` 这类 late tail-heavy 风险仍可能被低估。

## 4.4 probe 输出形态

每个 candidate 至少应输出：

- candidate metadata
  - `family`
  - `epoch`
  - `selector`
  - `ckpt`
- init diagnostics
- exit gate metrics
- exit tail guardrails
- raw Stage6 eval artifact 路径
  - `stage6_freerun/Walk_F_freerun_cycles.json`
  - `stage6_group_summary.json`
  - `posttrain_stage6_init_stats.json`

最终 summary 至少应能回答：

1. 哪个 candidate 是本轮 `Stage6-only` best
2. 哪些 candidate 明显 fail
3. 哪些 candidate 只是 near-tie，需要 exact downstream replay 再判

## 4.5 直接可复用/轻改的脚本路径

- `tools/run_stage6_probe_selector.py`
  - 已能跑 canonical Stage6-only probe
  - 已能输出 `step1/head20` readiness 统计
  - 下一步只需要 very small patch，把 exit hotspot / tail 指标补进 summary
- `tools/phasea_group_summary.py`
  - 已能对 freerun json 做 group summary
- `tools/run_basetrain_handoff_selector.py`
  - 其 `_mean_direct_deg(...)` 逻辑可以直接复用到 Stage6 probe 的 `foot/calf` hotspot 聚合
- `tools/export_cp015_basetrain_transfer_compat.py`
  - 其 true-sample tail / SIC hotspot 汇总逻辑可作为 Stage6-side postprocess 模板

---

## 5) basetrain export 最小 contract

## 5.1 设计目标

这个 export contract 的目标不是取代 exact Stage6，而是：

1. 把 basetrain-side readout 从 `curve_proxy` 升级到 true-sample readout；
2. 用统一 contract 保证不同 checkpoint / family 可 apple-to-apple 对比；
3. 为后续 selector v2 提供稳定输入 schema。

当前最合适的起点是：

- `tools/export_cp015_basetrain_transfer_compat.py`
- `debug_output/_tmp_cp015_next_basetrain_target_20260324/minimal_export_fields.csv`

也就是说，本轮不需要重新发明 schema，直接沿 `cp015_transfer_compat_v1` 收紧即可。

## 5.2 contract metadata

最小 export contract 必须显式记录以下元信息：

### A. candidate identity

- `schema_version`
- `candidate.name`
- `candidate.source_kind`
- `candidate.source_path`

### B. eval contract

- `model`
- `teacher_json`
- `encoder_bundle`
- `rounds`
- `depth`
- `time_index_mode`
- `event_clock`
- `phase_reset_source`
- `contacts_meas_source`
- `contacts_meas_pretrain_clamp`
- `contacts_meas_pretrain_affine_stats_spec`
- `mask_cycle_gte = 1`
- `mask_drop_wrap = true`

说明：

- 当前 `transfer_contract.json` 已记录大部分字段；
- 设计上建议把 `depth` 也显式写进 contract payload，而不是只在 runner 参数里隐式约束。

### C. contract checks

- `contacts_meas_source_ok`
- `rounds_ok`
- `depth_ok`
- `time_index_mode_ok`
- `phase_reset_source_ok`
- `event_clock_ok`
- `contacts_meas_pretrain_clamp_ok`
- `affine_stats_ok`
- `teacher_json_ok`
- `encoder_bundle_ok`
- `mask_cycle_gte_ok`
- `mask_drop_wrap_ok`
- `contract_ok`

## 5.3 raw sample payload

最小 raw payload 必须保留与后续统计一一对应的 masked sample pool：

- `direct_geolocal_deg[K, J]`
- `bone_names[J]`
- `root_idx`
- `step_index[K]`
- `step[K]`
- `cycle[K]`
- `step_in_cycle[K]`
- `time_index[K]`

理由：

- 只有保留原始 masked sample pool，后续才可以重新计算：
  - true `p50/p90/p95`
  - trimmed means
  - LR abs asymmetry
  - `step_in_cycle` hotspot mass

如果没有这层 raw payload，schema 很快又会退回到“先算死的 proxy summary”。

## 5.4 summary tables

最小 summary 层建议固定为四张表。

### A. `transfer_group_stats`

group 范围：

- `all_ex_root`
- `leg`
- `nonleg`
- `arm`

每组最少字段：

- `samples`
- `true_mean`
- `true_p50`
- `true_p90`
- `true_p95`
- `true_trim_top1`
- `true_trim_top5`
- `true_trim_top10`

### B. `transfer_joint_stats`

joint 范围至少保留：

- `thigh_l/r`
- `calf_l/r`
- `foot_l/r`
- `ball_l/r`
- `upperarm_l/r`
- `lowerarm_l/r`
- `hand_l/r`

每个 joint 仍使用：

- `samples`
- `true_mean`
- `true_p50`
- `true_p90`
- `true_p95`
- `true_trim_top1`
- `true_trim_top5`
- `true_trim_top10`

### C. `transfer_pair_abs_stats`

pair 范围：

- `thigh`
- `calf`
- `foot`
- `ball`
- `upperarm`
- `lowerarm`
- `hand`

最少字段：

- `samples`
- `abs_true_mean`
- `abs_true_p50`
- `abs_true_p90`
- `abs_true_p95`
- `abs_true_trim_top10`

这层是最小 asymmetry guardrail。

### D. `transfer_sic_hotspot_rows`

至少同时保留：

- group scope
- joint scope

最少字段：

- `tail_thr_true_p95`
- `sic_mean`
- `sic_true_p95`
- `sic_tail_hit_count`
- `sic_tail_hit_share`
- `sic_tail_excess_sum`
- `sic_tail_excess_share`

这层的作用不是花哨分析，而是把“phase-locked hotspot”从 broad mean 里拆出来。

## 5.5 这个 contract 在 selector 闭环里的职责

这个 contract 只应该承担：

1. `decision window` 定位
   - 例如确认 `ep12-15` 才是值得密集比较的窗口
2. broad reject
   - 例如 `phasebextend` 式 broad floor degradation
3. tail / hotspot audit
   - 尤其 arm / nonleg late tail-heavy 风险
4. selector v2 特征输入
   - 后续才考虑是否把它进入 selector code path

它 **不应该** 承担：

- final ranking between near-tied checkpoints once exact Stage6 actual already exists

## 5.6 直接可复用的脚本路径

- `tools/export_cp015_basetrain_transfer_compat.py`
  - 已经输出：
    - `transfer_contract.json`
    - `transfer_raw_masked_direct_geolocal.npz`
    - `transfer_group_stats.csv`
    - `transfer_joint_stats.csv`
    - `transfer_pair_abs_stats.csv`
    - `transfer_sic_hotspot_rows.csv`
- 现有 smoke / batch artifact：
  - `debug_output/_tmp_cp015_next_basetrain_target_20260324/export_smoke/`
  - `debug_output/_tmp_cp015_next_basetrain_target_20260324/export_batch_ep12_ep14_ep15_ep19/`

这意味着本轮 schema 设计可以直接建立在已有产物上，不需要大改实现。

---

## 6) selector 闭环建议

建议把 upstream selector 闭环明确拆成四层。

### Layer 0：candidate pool freeze

先冻结候选池，不要再让 late checkpoint 自动占优势：

- 每个 family 至少保留：
  - `best_free`
  - `last`
  - `best_teacher`
- 若有 dense checkpoints，则优先比较：
  - `ep12`
  - `ep13`
  - `ep14`
  - `ep15`
- `ep19` 保留为 late sentinel，不做默认 promote 候选

推荐复用：

- `tools/run_basetrain_handoff_selector.py`

但在本轮语义下，它更适合做：

- candidate discovery
- locked-boundary basetrain eval runner

而不是 final selector。

### Layer 1：basetrain export prefilter

对同一批 candidate 先做 true-sample export，并只做 coarse prefilter：

- reject clear broad degradation
- reject obvious hotspot concentration worsening
- reject obvious late drift

可以沿用 `2026-03-24` recommendation 里的保守 reject 规则，例如：

- `leg true_p50` 明显高于参考
- `leg true_trim_top10` 明显高于参考
- SIC hotspot mass 明显向差方向集中

但这里仍然只做 **pre-filter**，不做 near-tie final rank。

### Layer 2：optional plan calibration audit

如果 semantic 层已经过线但候选仍近似并列，则再补一层 calibration audit：

- `plan_gap_ratio_to_gt_mean`
- `plan_std_ratio_to_gt_mean`
- `plan_peak_trough_gap_mean`
- `plan_mid35_65_occupancy_mean`
- `contact_plan_lr_abs_diff_mean`
- `phase_shift_direct_geo_gain`

推荐复用：

- `tools/audit_selector_plan_metrics.py`

这一层的目标不是替代 Stage6 probe，而是防止 selector v1 只看 semantic alignment 就过早放行。

### Layer 3：Stage6-only final ranking

通过 prefilter 的候选才进入 `Stage6-only probe`：

- exact Stage6 actual 是 final authority
- 排名应以 Stage6 exit main table 为主
- init readiness 只做诊断附表

只有通过这一步的候选，才值得进入 downstream canonical chain。

---

## 7) 下一步执行顺序

建议按以下顺序推进，且每一步都尽量复用已有 artifact。

1. 先冻结当前 upstream 比较窗口
   - family 先围绕 `fixedsched / phasebcp018 / phasebextend`
   - epoch 先锁 `ep12-15`
   - `ep19` 仅作 late reject sentinel
2. 直接复用现有 basetrain export 样例
   - 先把 `debug_output/_tmp_cp015_next_basetrain_target_20260324/export_batch_ep12_ep14_ep15_ep19/` 作为 contract 参考
   - 不先重跑同类导出
3. 把 `Stage6-only probe` 的最小输出表头冻结
   - 主表必须含 `all_ex_root / leg / nonleg / foot_l/ball_l@SIC12-15 / calf_r@SIC2-4`
   - 附表必须含 `step1/head20 leg_over_nonleg`
   - guardrail 至少含 `leg/nonleg/arm true_p95`
4. 如果后续真的要写第一支代码
   - 优先补 `tools/run_stage6_probe_selector.py`
   - 让它在已有 Stage6 eval json 基础上多产出：
     - `foot_l/ball_l@SIC12-15`
     - `calf_r@SIC2-4`
     - `leg/nonleg/arm true_p95`
   - 不先去改训练逻辑或 downstream pipeline
5. 只有在 `Stage6-only` 排名稳定之后
   - 才考虑是否把这些指标正式并入 `run_basetrain_handoff_selector.py`
   - 或是否需要新的 selector v2 code path

---

## 8) 明确“不做什么”

本轮明确不做以下事项：

- 不修改 `docs/posttrain_pipeline.md`
- 不继续做 `71/72` 的 LR 或 step sweep
- 不把任务扩大成新的 full-chain replay 计划
- 不启动新的长链 basetrain 训练
- 不改 canonical Stage6 / downstream semantics
- 不把 basetrain export 误用成 final Stage6 predictor
- 不默认把 late `ep019` 当成 promote 目标

---

## 9) 本轮结论

对 upstream 来说，当前最值得做的第一步不是再调 downstream，而是：

1. 把 basetrain-side true-sample export contract 冻住；
2. 用它只做 coarse prefilter；
3. 再用 light `Stage6-only probe` 做 exact handoff ranking；
4. 最终只把 Stage6-pass 的少数候选送进后续 selector upgrade / canonical downstream。

换句话说：

> 先把 `basetrain -> Stage6` interface 讲清楚，
> 再谈训练和 selector 的下一轮升级。
