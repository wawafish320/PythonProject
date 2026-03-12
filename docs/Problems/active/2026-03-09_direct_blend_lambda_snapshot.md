# 2026-03-09 当前 `direct / blend / lambda` 快照整理

## 1) 目的

把当前 accepted `s180 low-LR trunkfull -> 71 -> 72 -> lambda final` 链路里，
最常被反复引用的三组信号单独收口：

- `DirectGeoLocalDeg`
- `BlendGeoLocalDeg`
- `LambdaMean`

并统一整理：

- 当前值（`mean / p50 / p90 / p95 / p99 / max`）
- 相对两份基线的变化
- 当前口径里需要特别注意的 source / affine 差异

---

## 2) 数据源

### 当前 accepted chain 评估 artifact

- current eval：`debug_output/_tmp_lambda_from_s180_72_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
- compare vs baseline A：
  - direct：`debug_output/_tmp_chain_s180promote_20260308/compare_vs_accepted_r5_direct/global_signal_summary.txt`
  - blend：`debug_output/_tmp_chain_s180promote_20260308/compare_vs_accepted_r5_blend/summary_metrics.txt`
- compare vs baseline B：
  - direct：`debug_output/_tmp_chain_s180promote_20260308/compare_vs_evalon_20260307_direct/global_signal_summary.txt`
  - blend：`debug_output/_tmp_chain_s180promote_20260308/compare_vs_evalon_20260307_blend/summary_metrics.txt`

### 基线 A

- `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305_r5/new_fullchain_pretrain/Walk_F_freerun_cycles.json`

### 基线 B

- `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_Walk_F_series/Walk_F_freerun_cycles.json`

### 统一统计口径

- mask：`cycle>=1, drop_wrap`
- 即：去掉第 0 个 cycle，并去掉每个 cycle wrap step
- 当前 3 份 JSON 在该口径下的 `n=344`

---

## 3) 先说清楚当前口径差异

这里有一个必须单独标注的事实：

- `docs/posttrain_pipeline.md` 当前主线 validate 命令要求：`pretrain_contact + clamp1 + affine_mix08`
- 但当前被 `s180 promote` checklist / compare 直接引用的 final eval artifact：
  - `contacts_meas_source = model`
  - `encoder_bundle = models/motion_encoder_equiv_stageA.pt`
  - `contacts_meas_pretrain_affine_stats_spec = None`

因此本文档整理的是：

- **当前 checklist / compare 实际使用的 final eval 快照**

而不是：

- **严格按主线 validate lane（`pretrain_contact + affine_mix08`）重跑后的新快照**

换句话说，这份文档适合回答“当前 compare artifact 上 `direct / blend / lambda` 到底是什么数”，
不适合直接替代“主线 validate contract 是否已经完全对齐”。

---

## 4) 当前三组核心信号：绝对值快照

### 4.1 当前 accepted chain（current）

| Metric | mean | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| `DirectGeoLocalDeg` | 0.112947 | 0.108628 | 0.167492 | 0.179196 | 0.199292 | 0.210580 |
| `BlendGeoLocalDeg` | 0.491534 | 0.469119 | 0.647434 | 0.736259 | 0.807640 | 0.821959 |
| `GeoLocalDeg` | 0.955117 | 0.924820 | 1.241557 | 1.382117 | 1.562999 | 1.570373 |
| `LambdaMean` | 0.973549 | 0.973612 | 0.973852 | 0.973918 | 0.974033 | 0.974067 |

补充：

- `LambdaEffMean` 与 `LambdaMean` 在当前 artifact 中相同
- `LambdaRelMean` 在当前 artifact 中恒为 `1.0`

### 4.2 基线 A（accepted `new_fullchain_pretrain` r5）

| Metric | mean | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| `DirectGeoLocalDeg` | 0.147802 | 0.141568 | 0.195531 | 0.217293 | 0.288073 | 0.321332 |
| `BlendGeoLocalDeg` | 0.531568 | 0.511545 | 0.696104 | 0.825876 | 0.878033 | 0.901127 |
| `GeoLocalDeg` | 1.030833 | 0.990132 | 1.354518 | 1.523022 | 1.676727 | 1.683177 |
| `LambdaMean` | 0.973585 | 0.973638 | 0.973801 | 0.973861 | 0.973922 | 0.973971 |

### 4.3 基线 B（`2026-03-07 eval_on`）

| Metric | mean | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| `DirectGeoLocalDeg` | 0.131316 | 0.127829 | 0.172214 | 0.179830 | 0.203642 | 0.221648 |
| `BlendGeoLocalDeg` | 0.497677 | 0.479911 | 0.612207 | 0.703717 | 0.795526 | 0.824611 |
| `GeoLocalDeg` | 0.961235 | 0.919981 | 1.233885 | 1.347935 | 1.547076 | 1.574776 |
| `LambdaMean` | 0.972715 | 0.972846 | 0.973101 | 0.973131 | 0.973276 | 0.973315 |

---

## 5) 相对基线 A：当前值怎么变

### 5.1 direct / blend / lambda 主指标

| Metric | baseline A | current | delta | 结论 |
|---|---:|---:|---:|---|
| `DirectGeoLocalDeg` | 0.147802 | 0.112947 | -0.034855 | 明显变好 |
| `BlendGeoLocalDeg` | 0.531568 | 0.491534 | -0.040034 | 明显变好 |
| `GeoLocalDeg` | 1.030833 | 0.955117 | -0.075716 | 明显变好 |
| `LambdaMean` | 0.973585 | 0.973549 | -0.000036 | 基本不变 |

### 5.2 分位数层面变化（baseline A -> current）

| Metric | p50 delta | p90 delta | p95 delta | p99 delta | max delta |
|---|---:|---:|---:|---:|---:|
| `DirectGeoLocalDeg` | -0.032940 | -0.028039 | -0.038097 | -0.088780 | -0.110752 |
| `BlendGeoLocalDeg` | -0.042426 | -0.048670 | -0.089617 | -0.070393 | -0.079168 |
| `GeoLocalDeg` | -0.065311 | -0.112961 | -0.140905 | -0.113728 | -0.112804 |
| `LambdaMean` | -0.000026 | +0.000051 | +0.000057 | +0.000111 | +0.000096 |

### 5.3 解释

- 相对旧 accepted baseline，当前 chain 的 `direct` / `blend` / `global geo` 都是稳定更好
- `LambdaMean` 几乎不动，说明这里的收益不是靠简单把 λ 平均值大幅推高换来的
- 因此“通过”主要来自 geometry 改善，而不是 λ 分布整体漂移

---

## 6) 相对基线 B：当前值怎么变

### 6.1 direct / blend / lambda 主指标

| Metric | baseline B | current | delta | 结论 |
|---|---:|---:|---:|---|
| `DirectGeoLocalDeg` | 0.131316 | 0.112947 | -0.018368 | 变好 |
| `BlendGeoLocalDeg` | 0.497677 | 0.491534 | -0.006143 | 小幅变好 |
| `GeoLocalDeg` | 0.961235 | 0.955117 | -0.006118 | 小幅变好 |
| `LambdaMean` | 0.972715 | 0.973549 | +0.000834 | 略升 |

### 6.2 分位数层面变化（baseline B -> current）

| Metric | p50 delta | p90 delta | p95 delta | p99 delta | max delta |
|---|---:|---:|---:|---:|---:|
| `DirectGeoLocalDeg` | -0.019201 | -0.004722 | -0.000634 | -0.004349 | -0.011068 |
| `BlendGeoLocalDeg` | -0.010792 | +0.035227 | +0.032541 | +0.012114 | -0.002652 |
| `GeoLocalDeg` | +0.004839 | +0.007672 | +0.034182 | +0.015923 | -0.004403 |
| `LambdaMean` | +0.000766 | +0.000751 | +0.000788 | +0.000757 | +0.000752 |

### 6.3 解释

- 相对 `2026-03-07 eval_on` baseline，`direct` 仍然是明确更好
- `blend` / `global geo` 虽然均值仍赢，但高分位（特别是 `p90/p95/p99`）没有像相对 baseline A 那样全面变好
- 这和 checklist 里的 `pass with watchlist` 结论一致：整体通过，但 residual hotspot 仍在

---

## 7) 当前最简结论

如果只看当前可直接引用的 artifact，可以收口为下面 5 句：

1. 当前 accepted `s180 -> 71 -> 72 -> lambda` 链，`DirectGeoLocalDeg` 已经明确优于两份基线。
2. `BlendGeoLocalDeg` 相对 baseline A 明显更好；相对 baseline B 只是小幅更好。
3. `GeoLocalDeg` 的均值层面也更好，但高分位仍保留 residual regression 风险。
4. `LambdaMean` 几乎不变或仅轻微上升，说明主要收益不是来自 λ 均值整体抬升。
5. 当前 checklist / compare 所依赖的 final eval artifact，并不是 `pretrain_contact + affine_mix08` validate lane，而是 `contacts_meas_source=model` 口径；如果后续要统一主线 contract，应该补一份严格按主线 validate 命令重跑的同口径快照。

---

## 8) 建议后续动作

- 若目标是**继续讨论主线 accept/watchlist**：直接引用本文档即可。
- 若目标是**统一主线 contract**：补跑一份 `lambda final` 的显式 `pretrain_contact + clamp1 + affine_mix08` eval，并按本文同格式补一版 snapshot。
- 若目标是**和 71m / stageA bundle 诊断线对照**：不要直接混用本页数值，先注明 source / affine / bundle 口径差异。
