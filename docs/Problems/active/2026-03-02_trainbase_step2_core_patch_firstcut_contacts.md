# 2026-03-02 trainbase Step2（core vs patch）第一刀：contacts provider 接口化

Last updated: 2026-03-03

## 1) 目标

在不改变主链默认行为的前提下，先把 contacts provider 相关路径拆成“core 决策 + patch 覆写”两层，作为后续降耦合（去隐式注入 / 去 fallback）的落脚点。

## 2) 本次改动

### 2.1 posttrain（train base）侧：显式 provider 接口

新增配置/CLI：
- `contact_meas_provider`: `auto|whitebox|learned`
- `contact_meas_provider_strict`: `off|on`

代码落点：
- `train/posttrain.py`
  - `PostTrainConfig` 增加字段
  - `_canon_contact_meas_provider` / `_flag_on`
  - `_cfg_parse_lambda_rollout` 解析配置
  - `_prepare_rollout_contacts_input` 改为按 provider 路由
  - `_build_model_and_trainer` 增加 fail-fast + provider 日志
  - arg parser 增加 `--contact_meas_provider*`

行为：
- `auto`：兼容历史逻辑（含 init/fallback）
- `whitebox`：固定白盒
- `learned`：要求 learned head；`strict=on` 时缺 head 直接失败

### 2.2 freerun 评估侧：core/patch 拆层 + strict

新增 core helper：
- `_resolve_contacts_meas_source_core(...)`
- `_contacts_meas_source_match(...)`

新增 CLI：
- `--contacts_meas_source_strict off|on`

代码落点：
- `train/validate/run_freerun_cycles.py`
  - core helper 负责基础源选择（model/whitebox/gt/zero）
  - patch 逻辑（GT override sic、model post-process）保留在 core 之后
  - strict 检查放在最终 applied 源上

新增 per-step 追踪字段（log_contacts 时可见）：
- `ContactsMeasSourceAppliedCore`
- `ContactsMeasSourceAppliedPatch`

### 2.3 freerun 评估侧：把 `whitebox_init` 从 core 挪到 patch 开关

新增 CLI：
- `--contact_plan_init_bootstrap_mode legacy|off`

代码落点：
- `train/validate/run_freerun_cycles.py`
  - `_resolve_contacts_meas_source_core(...)` 不再隐式注入 `whitebox_init`
  - rollout patch 段新增 t=0 bootstrap 分支（仅在 `bootstrap_mode=legacy` 时触发）
  - 产物 meta 新增 `contact_plan_init_bootstrap_mode`

行为：
- `legacy`：保持历史行为，`contacts_meas_source=model` 且 `init_mode=obs|learnable+obs` 时，t=0 可能出现 `applied=whitebox_init`
- `off`（默认）：关闭该隐式注入，`whitebox_init` 不再出现（除非你显式请求 whitebox）

## 3) 兼容性

- posttrain/trainbase 默认配置不变：`contact_meas_provider=auto`、`contacts_meas_source_strict=off`。
- freerun 评估侧默认已调整为：`contact_plan_init_bootstrap_mode=off`（如需历史行为可显式传 `legacy`）。
- strict 仅在显式开启时生效，用于阻止“请求源 != 实际源”的隐式回退。

## 4) 本地 smoke 验证

### 4.1 语法/参数

- `python -m py_compile train/posttrain.py train/validate/run_freerun_cycles.py` ✅
- `python -m train.posttrain --help`（可见 `--contact_meas_provider*`）✅
- `python -m train.validate.run_freerun_cycles --help`（可见 `--contacts_meas_source_strict`）✅
- `python -m train.validate.run_freerun_cycles --help`（可见 `--contact_plan_init_bootstrap_mode`）✅

### 4.2 freerun 行为

- 默认（strict=off）可跑通：
  - `debug_output/__tmp_step2_core_patch_contacts_log/Walk_F_freerun_cycles.json`
- strict=on 会在隐式 init 注入时 fail-fast（符合预期）：
  - `contacts_meas_source=model` 但实际 `applied=whitebox_init` -> 报错退出
  - 产物：`debug_output/__tmp_step2_core_patch_contacts_strict/`
- `contact_plan_init_bootstrap_mode=off` + `strict=on` 可跑通（`applied=model`，不再触发 `whitebox_init`）：
  - `debug_output/__tmp_step2p3_bootstrap_off_strict_ok/Walk_F_freerun_cycles.json`

### 4.3 Step0 baseline 口径 strict 巡检（active whitelist 8 ckpt）

执行脚本：
- `debug_output/step2_contacts_strict_scan_20260303/run_step2_contacts_strict_scan.py`

统一口径：
- `contacts_meas_source=model`
- `contacts_meas_source_strict=on`
- 固定 Step0 baseline 其余参数（`event_clock=auto`、`phase_reset_source=none`、`direct_pose_meas_source=model`、`lambda_fusion_apply=on`、`rounds=5`）
- 对比 `contact_plan_init_bootstrap_mode=legacy|off`

产物：
- `debug_output/step2_contacts_strict_scan_20260303/strict_scan_raw.csv`
- `debug_output/step2_contacts_strict_scan_20260303/strict_scan_summary.csv`
- `debug_output/step2_contacts_strict_scan_20260303/summary.md`

结果：
- `bootstrap=legacy`：8/8 全部 strict 失败，失败原因为 `applied=whitebox_init`
- `bootstrap=off`：8/8 全部通过，step0 记录为 `requested=model / applied=model / core=model / patch=none`
- 未发现 `whitebox_fallback` 依赖（在本次 active whitelist + Walk_F teacher 口径下）

## 5) 下一步建议

1. trainbase/posttrain 侧对齐同样的 bootstrap 分层开关，消除跨入口语义差异。
2. 为 `contacts_err` 引入显式输入接口（requested/applied/fallback_reason）并在 summary 固化。
3. 在 Step0/Step1 聚合脚本里增加 `ContactsMeasSourceApplied{Core,Patch}` 的汇总列，固定巡检口径。
