# 2026-03-02 trainbase Step0 / Step1 执行记录（独立文档）

Last updated: 2026-03-02

## 1) 目标与口径

基于以下文档落地最小闭环：
- docs/Problems/active/2026-03-02_trainbase_simplify_review.md
- docs/trainbase_design/2026-03-02_trainbase_v2_core_patch_flow.md

本次只执行两步：
- Step 0: 冻结 baseline 评估口径（contacts_meas_source=model）
- Step 1: 当时主链显式固定 historical whitebox provider（该口径现已退休）

统一运行条件：
- active whitelist 8 个 config 对应 ckpt
- teacher: validate/teacher_batches/Walk_F_teacher.json
- rounds=5，统计 round>=1 均值
- 固定 event_clock=auto、phase_reset_source=none、direct_pose_meas_source=model
- 固定 lambda_fusion_apply=on（闭环系统口径）

## 2) 执行命令（脚本化批量）

执行脚本：
- debug_output/step01_contact_provider_20260302/run_step01_contact_provider.py

脚本会按 whitelist 逐个运行，核心命令形态如下（Step0/Step1 仅 contacts_meas_source 不同）：

python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model <ckpt> \
  --rounds 5 \
  --time-index-mode cycle \
  --depth 3 \
  --event_clock auto \
  --phase_reset_source none \
  --contacts_meas_source <model|whitebox> \
  --direct_pose_meas_source model \
  --lambda_fusion_apply \
  --log_contacts \
  --out <out_dir> \
  --force

## 3) 产物路径

总目录：
- debug_output/step01_contact_provider_20260302/

Step 0 产物：
- debug_output/step01_contact_provider_20260302/step0_baseline_model/<stage>/Walk_F_freerun_cycles.json

Step 1 产物：
- debug_output/step01_contact_provider_20260302/step1_provider_whitebox/<stage>/Walk_F_freerun_cycles.json

聚合产物：
- debug_output/step01_contact_provider_20260302/step01_metrics_round_ge1.csv
- debug_output/step01_contact_provider_20260302/step01_compare_whitebox_minus_model_round_ge1.csv
- debug_output/step01_contact_provider_20260302/summary.md

## 4) 结果摘要（8 stage 均值，whitebox-model）

| metric | Step0 model | Step1 whitebox | delta |
|---|---:|---:|---:|
| GeoLocalDegWeighted | 63.379988 | 63.364724 | -0.015264 |
| GeoDeg | 77.372551 | 77.465822 | +0.093271 |
| BlendGeoLocalDegWeighted | 63.379988 | 63.364724 | -0.015264 |
| DirectGeoLocalDegWeighted | 1.070783 | 1.085391 | +0.014608 |
| ContactErrAbsMean | 0.094922 | 0.338358 | +0.243436 |
| ContactMeasGtAbsMean | 0.415979 | 0.442926 | +0.026946 |
| runtime_s (per run) | 4.863819 | 5.408772 | +11.24% |

## 5) 当前结论（对应 Step0/Step1）

1. Step0 baseline 口径已固定并落盘，可作为后续 core/patch 与降耦合改造的统一对照。
2. Step1 显式 whitebox provider 在主质量指标上与 model 接近（GeoLocalDegWeighted 变化很小），但 Direct 指标与 contact 相关误差统计有可观变化。
3. 推理耗时在本次单次测量口径下 whitebox 平均约 +11.24%，后续需要重复 benchmark 做稳定性确认。

## 6) 下一步（对齐你提出的顺序）

- 保持 Step1 显式 provider 语义，进入 core vs patch 大模块拆分。
- 拆分后再做降耦合：contacts_err 接口化、去隐式注入、逐步移除 fallback，并用 Step0 固定口径回归验收。
