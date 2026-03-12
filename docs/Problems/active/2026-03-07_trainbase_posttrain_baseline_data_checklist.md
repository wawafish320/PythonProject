# 2026-03-07 Trainbase / Posttrain 基准数据清单

## 目标

给后续 trainbase / posttrain 清理提供一份**固定基准数据清单**，避免删完分支后失去
比较锚点。

本清单主要参考以下现成 artifact：

- `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/old_whitebox_vs_new_fullchain_pretrain/gate_metrics.json`
- `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/old_whitebox_vs_new_fullchain_pretrain/global_signal_summary.txt`
- `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/old_whitebox_vs_new_fullchain_pretrain/summary_metrics.txt`

其中：

- `old_lambda_whitebox` = 历史 whitebox 参考线；
- `new_fullchain_pretrain` = 当前 `pretrain_contact + affine_mix08` 接受基线。

---

## 1) Authoritative baseline pack

### A. 基线 compare 包（历史 whitebox -> 当前 accepted baseline）

- 目录：`debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/old_whitebox_vs_new_fullchain_pretrain`
- gate：`debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/old_whitebox_vs_new_fullchain_pretrain/gate_metrics.json`
- global signal：`debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/old_whitebox_vs_new_fullchain_pretrain/global_signal_summary.txt`
- summary metrics：`debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/old_whitebox_vs_new_fullchain_pretrain/summary_metrics.txt`

### B. 当前 accepted baseline 原始 JSON

- `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/new_fullchain_pretrain/Walk_F_freerun_cycles.json`

### C. 本轮 `lambda_fusion_apply=on` 对 accepted baseline 的同格式 compare 包

- 目录：`debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F`
- gate：`debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F/gate_metrics.json`
- global signal：`debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F/global_signal_summary.txt`
- summary metrics：`debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F/summary_metrics.txt`

---

## 2) Accepted baseline snapshot (`new_fullchain_pretrain`)

以下数值全部来自：

- `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/old_whitebox_vs_new_fullchain_pretrain/summary_metrics.txt`

### A. Global (all bones excl root)

- `mean_deg = 0.1464529355537977`
- `p50_deg = 0.08172890916466713`
- `p90_deg = 0.369600659608841`
- `p95_deg = 0.5138191908597944`
- `p99_deg = 0.8987995332479471`
- `max_deg = 1.7128169536590576`

### B. SIC12-15 + `foot_l,ball_l`

- `mean_deg = 0.5361018516123295`
- `p50_deg = 0.5437473058700562`
- `p90_deg = 0.7539978921413422`
- `p95_deg = 0.7721778601408005`
- `p99_deg = 0.80157011449337`
- `max_deg = 0.8089181780815125`

### C. `calf_r` global

- `mean_deg = 0.26531818303344556`
- `p50_deg = 0.24677465856075287`
- `p90_deg = 0.500238510966301`
- `p95_deg = 0.5525934875011442`
- `p99_deg = 0.6739819395542139`
- `max_deg = 0.7327119708061218`

### D. `calf_r @ SIC2-4`

- `mean_deg = 0.13048052539428076`
- `p50_deg = 0.13126190751791`
- `p90_deg = 0.15918438881635666`
- `p95_deg = 0.15961572900414467`
- `p99_deg = 0.15996080115437508`
- `max_deg = 0.16004706919193268`

### E. `calf_r @ SIC35-42`

- `mean_deg = 0.22464306897018105`
- `p50_deg = 0.20510689914226532`
- `p90_deg = 0.43833182752132416`
- `p95_deg = 0.48859233409166336`
- `p99_deg = 0.4984718218445778`
- `max_deg = 0.5009416937828064`

### F. `calf_r @ SIC53-63`

- `mean_deg = 0.24309141459790143`
- `p50_deg = 0.26207469403743744`
- `p90_deg = 0.35523604452610025`
- `p95_deg = 0.364609794318676`
- `p99_deg = 0.3664527177810669`
- `max_deg = 0.3668515086174011`

---

## 3) Historical control snapshot (`old_lambda_whitebox`)

这组保留的意义是：后面如果清理 `pretrain_contact` 主链分支，仍然可以回头判断是
“退回了 old whitebox 行为”还是“偏离了 accepted baseline”。

### A. Global (all bones excl root)

- `mean_deg = 0.1575153519833121`
- `p50_deg = 0.0844406932592392`
- `p90_deg = 0.4111690849065781`
- `p95_deg = 0.5322088003158559`
- `p99_deg = 0.9153735309839237`
- `max_deg = 2.5822157859802246`

### B. SIC12-15 + `foot_l,ball_l`

- `mean_deg = 0.4998738747090101`
- `p50_deg = 0.5296328514814377`
- `p90_deg = 0.6942403614521027`
- `p95_deg = 0.7499096691608429`
- `p99_deg = 0.7654249250888825`
- `max_deg = 0.7693037390708923`

### C. `calf_r` global

- `mean_deg = 0.31679838602322824`
- `p50_deg = 0.25630390644073486`
- `p90_deg = 0.6098024487495423`
- `p95_deg = 0.7112807422876357`
- `p99_deg = 0.9872807180881475`
- `max_deg = 1.2649134397506714`

### D. `calf_r @ SIC2-4`

- `mean_deg = 0.3662948856751124`
- `p50_deg = 0.32627008855342865`
- `p90_deg = 0.47267091274261475`
- `p95_deg = 0.4784783869981766`
- `p99_deg = 0.483124366402626`
- `max_deg = 0.4842858612537384`

### E. `calf_r @ SIC35-42`

- `mean_deg = 0.18143795838113874`
- `p50_deg = 0.21417807787656784`
- `p90_deg = 0.2497124969959259`
- `p95_deg = 0.2528001666069031`
- `p99_deg = 0.2574933886528015`
- `max_deg = 0.2586666941642761`

### F. `calf_r @ SIC53-63`

- `mean_deg = 0.40202517502687196`
- `p50_deg = 0.3695749044418335`
- `p90_deg = 0.6193354487419128`
- `p95_deg = 0.6652199804782867`
- `p99_deg = 0.703939705491066`
- `max_deg = 0.7136082053184509`

---

## 4) Gate baseline (历史 whitebox -> accepted baseline)

来源：

- `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/old_whitebox_vs_new_fullchain_pretrain/gate_metrics.json`

关键 gate 数值：

- `leg8_mean_delta = -0.015139309907004517`
- `non_leg_mean_delta = -0.010180925947894909`
- `global_mean_old = 0.1575153519833121`
- `global_mean_new = 0.1464529355537977`
- `global_mean_rel_delta_pct = -7.023071903928699`
- `sic12_15_footl_balll_old = 0.4998738747090101`
- `sic12_15_footl_balll_new = 0.5361018516123295`
- `calf_r_global_old = 0.31679838602322824`
- `calf_r_global_new = 0.26531818303344556`
- `calf_r_sic2_4_old = 0.3662948856751124`
- `calf_r_sic2_4_new = 0.13048052539428076`

---

## 5) 当前 `lambda_fusion_apply=on` 的 Walk_F compare 路径

如果后续清理完分支后，需要快速对照这次 2026-03-07 结果，直接看：

- `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F/global_signal_summary.txt`
- `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F/summary_metrics.txt`
- `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F/gate_metrics.json`

这组 compare 的结论是：

- `global_mean_old = 0.14780221704995444`
- `global_mean_new = 0.13131584089519935`
- `global_mean_rel_delta_pct = -11.15434970043988`
- `leg8_mean_delta = -0.02168916104972929`
- `non_leg_mean_delta = -0.015361449690976886`

但 `calf_r_global_old -> new = 0.27078741867376793 -> 0.2885736009690824`，
说明这次 `lambda_on` 虽然全局更好，`calf_r global` 仍有局部回升，需要保留为后续清理后的重点观察点。
