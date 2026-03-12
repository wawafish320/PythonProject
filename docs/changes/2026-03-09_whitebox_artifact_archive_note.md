# 2026-03-09 `whitebox` 历史资产归档说明

自 2026-03-09 起，`whitebox` runtime/validate lane 已从当前 mainline 退休。

因此，仓库中仍带有 `whitebox` 命名的 checkpoint、JSON、输出目录，均应按下面口径理解：

- 它们是历史 control / baseline / debug artifact，不代表当前可执行主线。
- 它们用于复盘旧结论，或作为和当前 accepted `pretrain_contact` 主线的对照基线。
- 当前主线复跑时，不应再把这些目录名当成可继续沿用的 route contract。

常见历史资产形态：

- `models/__tmp_phaseD_posttrain_ab_20260304_fullrerun/*whitebox*`
- `models/__tmp_phaseD_posttrain_ab_20260305/*whitebox*`
- `models/__tmp_phaseD_route_smoke/*whitebox*`
- `debug_output/step01_contact_provider_20260302/step1_provider_whitebox/`
- `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305*/old_whitebox_vs_new_*`
- `debug_output/_tmp_phaseD_posttrain_ab_20260305*/eval_*_whitebox/`
- `debug_output/stepB_contactmeas_ab_20260302/*_whitebox/`

阅读这些资产时，统一按以下语义解释：

- `*_whitebox*`: 历史 `whitebox` control / source baseline
- `old_*whitebox*`: 用于和当前 accepted baseline 做 old-vs-new compare 的旧参考线
- `phaseD_*whitebox*`: 清理前 A/B 或 smoke 产物，不再对应当前 runtime contract

如果后续需要保留历史结果，请继续保留这些目录名；但新增文档/脚本引用它们时，应显式标注 `historical` / `archive`，避免被误判为现役路径。
