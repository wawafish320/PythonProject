# 2026-03-11 `exp_phase_DirectBranch_v1_d1 -> posttrain mainline` pointwise 快照

## 1) 目的

这份文档只整理当前这条主链：

- trainbase 入口使用 `exp_phase_DirectBranch_v1_d1`；
- downstream 口径按 `docs/posttrain_pipeline.md` 当前 accepted mainline；
- 旧 baseline 包可能失效，所以这里不再展开 baseline section；
- 直接以 `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F` 对应的 `new_json` 作为 trainbase->posttrain anchor；
- pointwise 展示形式参考 `2026-03-09_direct_blend_lambda_pointwise_snapshot.md`，但只保留单源结果。

---

## 2) 主链口径

### 2.1 trainbase 入口

```bash
python -m train.training_MPL   --config_json config/exp_phase_mpl.clean.json   --run_name exp_phase_DirectBranch_v1_d1   --out ./models/MLPL2_DirectBranch_v1   --depth 3   --encoder_path ./models/motion_encoder_equiv_stageA.pt   --contact_plan_enable   --contact_plan_init_mode learnable+obs --contact_plan_init_hidden 128   --direct_pose_enable --w_direct_pose 0.2   --contact_plan_time_pe_dim 16   --direct_pose_meas_mode concat   --direct_pose_meas_drop_prob 0.1 --direct_pose_plan_drop_prob 0.1 --direct_pose_meas_noise_std 0.03   --use_event_clock   --event_clock_max_delta 0.5   --event_clock_hidden_dim 64   --event_clock_gate_hidden_dim 32   --event_clock_lambda_entropy_weight 0.01   --event_clock_lambda_prior_weight 0.01   --event_clock_delta_z_l2_weight 0.001
```

补充事实：

- `config/exp_phase_mpl.clean.json` 当前已经内置 `trainbase_contacts_source=pretrain_contact`；
- 同时固定 `trainbase_contacts_pretrain_clamp=1.0`；
- 并指向 `debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`。

### 2.2 downstream accepted mainline

当前 `docs/posttrain_pipeline.md` 主链是：

`Stage6 -> 70a -> 70b_concat -> 70c_replacecontacts (historical reference shell) -> promoted 70R (low-LR trunkfull s180) -> 71 -> 72 -> lambda final`

accepted downstream 证据：

- `debug_output/_tmp_70R_lowlr_trunkfull_s180_rounds5_20260308/s180_verdict.md`
- `debug_output/_tmp_chain_s180promote_20260308/chain_verdict.md`
- `debug_output/_tmp_chain_s180promote_20260308/compare_vs_evalon_20260307_direct/global_signal_summary.txt`
- `debug_output/_tmp_chain_s180promote_20260308/compare_vs_evalon_20260307_blend/summary_metrics.txt`

---

## 3) 数据源

### 3.1 eval-on compare pack

- compare dir: `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F`
- summary: `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F/summary_metrics.txt`
- global signal: `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F/global_signal_summary.txt`
- gate: `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F/gate_metrics.json`
- anchor json: `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_Walk_F_series/Walk_F_freerun_cycles.json`

### 3.2 2026-03-11 full trainbase rerun

这次不是 smoke，而是把整条 trainbase 命令按完整 epoch 跑完，出于避免覆盖旧产物的原因只改了 `run_name`：

- run dir: `models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1_fullrerun_20260311`
- best teacher: `models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1_fullrerun_20260311/ckpt_best_teacher_exp_phase_DirectBranch_v1_d1_fullrerun_20260311.pth`
- best free: `models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1_fullrerun_20260311/ckpt_best_free_exp_phase_DirectBranch_v1_d1_fullrerun_20260311.pth`
- last: `models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1_fullrerun_20260311/ckpt_last_exp_phase_DirectBranch_v1_d1_fullrerun_20260311.pth`
- ONNX: `models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1_fullrerun_20260311/exp_phase_DirectBranch_v1_d1_fullrerun_20260311_step_stateful_nophase.onnx`

full rerun 末尾结果：

- `train_ep018 loss = 0.413923`
- `teacher_ep018 loss = 0.395048`
- `teacher_ep018 GeoDeg = 0.184363 deg`
- `teacher_ep018 GeoLocalDeg = 0.077797 deg`
- `valfree_ep018 GeoDeg = 4.044082 deg`
- `valfree_ep018 GeoLocalDeg = 3.784185 deg`
- `valfree_ep018 RootVelMAE = 0.321580`
- `valfree_ep018 AngVelMAE = 0.493949 rad/s`

相对旧的 `exp_phase_DirectBranch_v1_d1` 目录：

- freerun `GeoDeg` 略好：`4.257098 -> 4.044082`
- freerun `GeoLocalDeg` 略好：`3.812674 -> 3.784185`
- `RootVelMAE` 基本持平：`0.321375 -> 0.321580`
- teacher 指标没有严格追平旧目录，但 trainbase 全流程当前代码仍然可完整跑通并稳定导出 ckpt/onnx。

### 3.3 fresh downstream chain（active handoff）

我继续把 fresh best-free ckpt 按当前 active downstream handoff 真跑到了 final：

- Stage6:
  - `models/__tmp_fullchain_from_exp_phase_DirectBranch_v1_d1_fullrerun_20260311/ckpt_last_WalkF_stage6_from_fullrerun_20260311.pth`
- 70a:
  - `models/__tmp_fullchain_from_exp_phase_DirectBranch_v1_d1_fullrerun_20260311/ckpt_last_WalkF_stage7_70a_from_fullrerun_20260311.pth`
- 70a replace-control warm-start:
  - `models/__tmp_fullchain_from_exp_phase_DirectBranch_v1_d1_fullrerun_20260311/ckpt_last_WalkF_stage7_70a_replacecontacts_zerophase_from_fullrerun_20260311.pth`
- 70b_replace:
  - `models/__tmp_70b_replace_from_fullrerun_20260311/ckpt_last_WalkF_stage7_70b_replacecontacts_from_fullrerun_20260311.pth`
- 70R promoted trunkfull s180:
  - `models/__tmp_70R_from_fullrerun_trunkfull_s180_20260311/ckpt_last_WalkF_stage7_70R_from_fullrerun_trunkfull_s180_20260311.pth`
- 71:
  - `models/__tmp_71_from_fullrerun_20260311/ckpt_last_WalkF_stage7_71_from_fullrerun_20260311.pth`
- 72:
  - `models/__tmp_72_from_fullrerun_20260311/ckpt_last_WalkF_stage7_72_from_fullrerun_20260311.pth`
- lambda final:
  - `models/__tmp_lambda_from_fullrerun_20260311/ckpt_last_WalkF_stage7_lambda_from_fullrerun_20260311.pth`

说明：

- 这里走的是当前 accepted downstream 真正依赖的 active handoff：
  `Stage6 -> 70a -> new70b_replace -> promoted 70R(s180) -> 71 -> 72 -> lambda final`
- 历史 `70b_concat / 70c_replacecontacts` reference shell 本次 fresh rerun 没单独复跑，
  因为当前 accepted downstream promote 并不从那两个 historical shell 继续 handoff。

### 3.4 fresh final freerun eval

fresh final ckpt 的 freerun eval 产物：

- eval json: `debug_output/_tmp_lambda_from_fullrerun_eval_20260311_Walk_F/Walk_F_freerun_cycles.json`
- compare note: `debug_output/_tmp_lambda_from_fullrerun_eval_20260311_Walk_F/summary_vs_anchor.md`

运行口径：

- `rounds=5`
- `phase_reset_source=none`
- `contacts_meas_source=pretrain_contact`
- `contacts_meas_pretrain_clamp=1.0`
- `contacts_meas_pretrain_affine_stats=affine_mix08`
- `lambda_fusion_apply=true`

masked step metrics（`cycle>=1, drop_wrap=True`）：

- `DirectGeoLocalDeg mean = 0.121730`
- `BlendGeoLocalDeg mean = 0.121224`
- `GeoLocalDeg mean = 0.515706`
- `LambdaMean mean = 0.953183`

相对 anchor `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_Walk_F_series/Walk_F_freerun_cycles.json`：

- 下面这些都是 error mean，数值越低越好；

- direct global mean：`0.131316 -> 0.121730`（`-7.2995%`）
- leg8 mean：`0.292003 -> 0.289441`
- non-leg mean：`0.096573 -> 0.085469`
- `SIC12-15 foot_l/ball_l`：`0.572571 -> 0.577879`
- `calf_r global`：`0.288574 -> 0.245514`
- `calf_r SIC2-4`：`0.131135 -> 0.092557`
- `calf_r SIC35-42`：`0.139700 -> 0.230735`
- `calf_r SIC53-63`：`0.408190 -> 0.353921`

结论：

- basetrain -> downstream fullchain -> final eval 这条 fresh rerun 已经真正跑通；
- overall direct / non-leg / calf_r 主窗口是改善的；
- 但 watchlist 还在：`foot_l/ball_l @ SIC12-15` 有轻微回退，`calf_r @ SIC35-42` 也更差；
- 因此这条 fresh chain 更像“pass with watchlist”，还不能直接宣称严格支配所有已有 anchor。

### 3.5 和 accepted old baseline / accepted final 的关系

如果按更严格的 accepted 口径，不应只看 `eval_on` anchor，还要看：

- accepted old baseline：`debug_output/_tmp_chain_s180promote_20260308/compare_vs_accepted_r5_direct/global_signal_summary.txt`
- accepted final：同一文件里记录的 `new_json=debug_output/_tmp_lambda_from_s180_72_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`

repo 当前可直接读取到的 direct global mean 对比是：

- accepted old baseline：`0.147802`
- current accepted final：`0.112947`
- fresh full-rerun final：`0.121730`

所以更准确的结论应写成：

- fresh full-rerun final 相对 accepted old baseline 仍然改善：`0.147802 -> 0.121730`（约 `-17.64%`）
- 但 fresh full-rerun final **没有追平** 当前 accepted final：`0.112947 -> 0.121730`（约 `+7.78%` 回退）
- 对应 `non_leg_mean` 也是同样结论：
  - vs accepted old baseline：`0.111934 -> 0.085469`（改善）
  - vs current accepted final：`0.078048 -> 0.085469`（回退）

因此：

- 如果比较对象是 `eval_on` anchor，这次 fresh rerun 是改善；
- 如果比较对象是你说的 accepted baseline / accepted final 口径，这次 fresh rerun 还不够好，不能替代当前 accepted final。

---

## 4) accepted mainline 相对这个 anchor 的位置

从 `debug_output/_tmp_chain_s180promote_20260308/compare_vs_evalon_20260307_direct/global_signal_summary.txt`
与 `debug_output/_tmp_chain_s180promote_20260308/compare_vs_evalon_20260307_blend/summary_metrics.txt` 看，
当前 accepted `lambda final` 相对这个 anchor 的 masked 结果是：

- `DirectGeoLocalDeg mean`: `0.131316 -> 0.112947` (`-13.9879%`)
- `leg8_mean`: `0.292003 -> 0.274360`
- `non_leg_mean`: `0.096573 -> 0.078048`
- `improved_ratio`: `0.615698`
- `BlendGeoLocalDeg mean`: `0.497677 -> 0.491534`
- `GeoLocalDeg mean`: `0.961235 -> 0.955117`
- `LambdaMean mean`: `0.972715 -> 0.973549`

因此这份文档里的 anchor 可以继续作为“trainbase 出发点”的 reference pack；
但如果要描述当前 accepted full-chain 终点，应以上面 `s180-promote -> 71 -> 72 -> lambda final` 的 compare 包为准。

---

## 5) anchor 的 masked 全局摘要（只看 new_json）

### 5.1 Global (all bones excl root)

- `steps_kept = 344`
- `samples_kept = 15480`
- `mean_deg = 0.131316`
- `p50_deg = 0.079918`
- `p90_deg = 0.332772`
- `p95_deg = 0.451996`
- `p99_deg = 0.705909`
- `max_deg = 1.414769`

per-bone mean top 8:

1. `foot_l`: `0.3305`
2. `foot_r`: `0.3275`
3. `thigh_l`: `0.3246`
4. `calf_l`: `0.3145`
5. `thumb_01_l`: `0.3056`
6. `thigh_r`: `0.2987`
7. `hand_r`: `0.2983`
8. `calf_r`: `0.2886`

### 5.2 Focus windows

- `SIC12-15 foot_l/ball_l`: mean=`0.572571`, p95=`0.807285`, max=`0.849918`
- `calf_r global`: mean=`0.288574`, p95=`0.700232`, max=`1.414769`
- `calf_r @ SIC2-4`: mean=`0.131135`, p95=`0.191791`, max=`0.194397`
- `calf_r @ SIC35-42`: mean=`0.139700`, p95=`0.327516`, max=`0.340846`
- `calf_r @ SIC53-63`: mean=`0.408190`, p95=`0.859333`, max=`0.921687`

这里最值得记住的是：

- `foot_l/ball_l` 在 `sic=12-15` 仍明显偏高；
- `calf_r` 真正重的窗口不是 `sic=2-4`，而是 `sic=53-63`；
- 所以这个 anchor 的 lower-body 问题不是单点，而是“左脚窗口 + 右小腿后段窗口”双热点。

---

## 6) pointwise 口径说明

统一 mask：

- `cycle >= 1`
- `drop_wrap = True`
- 有效 step 数：`n = 344`
- `cycle_len = 87`

quantile 继续使用 nearest-rank 实际点：

- 不做插值；
- 直接取 `ceil(q * n)` 对应的真实 step；
- 每个点都记录 `global_step / cycle / sic / value`。

---

## 7) anchor：实际索引点

### 7.1 `DirectGeoLocalDeg`
| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 336 | 3 | 75 | 0.127522 |
| `p90` | 310 | 178 | 2 | 4 | 0.172469 |
| `p95` | 327 | 259 | 2 | 85 | 0.179868 |
| `p99` | 341 | 87 | 1 | 0 | 0.212547 |
| `max` | - | 261 | 3 | 0 | 0.221648 |

### 7.2 `BlendGeoLocalDeg`
| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 411 | 4 | 63 | 0.479669 |
| `p90` | 310 | 315 | 3 | 54 | 0.613184 |
| `p95` | 327 | 183 | 2 | 9 | 0.704609 |
| `p99` | 341 | 360 | 4 | 12 | 0.796576 |
| `max` | - | 99 | 1 | 12 | 0.824611 |

### 7.3 `GeoLocalDeg`
| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 394 | 4 | 46 | 0.918294 |
| `p90` | 310 | 315 | 3 | 54 | 1.236477 |
| `p95` | 327 | 101 | 1 | 14 | 1.348158 |
| `p99` | 341 | 273 | 3 | 12 | 1.547721 |
| `max` | - | 99 | 1 | 12 | 1.574776 |

### 7.4 `LambdaMean`
| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 250 | 2 | 76 | 0.972846 |
| `p90` | 310 | 229 | 2 | 55 | 0.973102 |
| `p95` | 327 | 361 | 4 | 13 | 0.973131 |
| `p99` | 341 | 261 | 3 | 0 | 0.973286 |
| `max` | - | 273 | 3 | 12 | 0.973315 |

---

## 8) 当前最值得记住的点

1. `DirectGeoLocalDeg`：
   - `p95` 在 `cycle=2, sic=85`
   - `p99` 在 `cycle=1, sic=0`
   - `max` 在 `cycle=3, sic=0`

2. `BlendGeoLocalDeg`：
   - `p95` 在 `cycle=2, sic=9`
   - `p99` 在 `cycle=4, sic=12`
   - `max` 在 `cycle=1, sic=12`

3. `GeoLocalDeg`：
   - `p95` 在 `cycle=1, sic=14`
   - `p99` 在 `cycle=3, sic=12`
   - `max` 在 `cycle=1, sic=12`

4. `LambdaMean`：
   - `p95` 在 `cycle=4, sic=13`
   - `p99` 在 `cycle=3, sic=0`
   - `max` 在 `cycle=3, sic=12`

5. 如果只选 watch steps，优先看：
   - `step 99` / `step 273` / `step 360`：反复出现的 `sic=12` mixed hotspot
   - `step 87` / `step 261`：cycle-start lower-body hotspot
   - `step 361`：lambda 高位紧跟在 `sic=12` 后面的拖尾点

---

## 9) 后续

joint-level hotspot 解释见：

- `docs/Problems/active/2026-03-11_trainbase_posttrain_mainline_evalon_hotspots.md`
