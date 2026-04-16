# 2026-04-13 `train/training_MPL.py` 热点重构路线图（v1）

Date: 2026-04-13  
Status: Draft / Active v1 / update-3（截至 2026-04-13 已完成 Phase A、Phase B2、Phase B2.5，并启动 Phase B1 第一轮 rollout 壳层化；不改训练语义，不改默认 CLI，不改 checkpoint/metrics schema）  
Scope: `train/training_MPL.py`（本轮只做 trainer/rollout/diagnostics/train-entry 的职责拆分与边界收口，不改 `train/models.py` 的模型语义，不改 loss 数学定义）  
Goal: 在**不改变语义/行为**前提下，优先降低“超大 `Trainer` 类 + rollout/验证链路耦合 + CLI/build 样板膨胀 + 宽泛异常吞没 + I/O 副作用穿透主路径”的维护风险。  
Non-goal: 不改默认超参、不改 `--help` flag 集合、不改 bundle/norm template 约定、不改 `best_teacher/best_free/last` checkpoint 命名。

关联参考文档：
- 参考格式：`docs/delete/2026-03-10_train_models_refactor_hotspots_roadmap.md`

---

## 0) 当前策略（先收 config/runtime/metrics 边界，再拆 Trainer 热点，再收紧异常，最后剥离副作用）

本轮总原则先明确：

- **第一优先策略不是马上把 `Trainer` 拆成多个文件，而是先收 runtime/config/metrics 的重复面和边界。**
- `training_MPL.py` 当前的问题不是单个 2000 行函数，而是**一个 3865 行的 `Trainer` 类 + 多个 100~450 行热点 helper + 训练/诊断/导出/CLI 副作用交织**。
- 如果不先把 `args -> runtime_cfg -> trainer attr`、metrics/checkpoint 落盘模板、stage-schedule 参数写回这些边界收紧，后续拆 rollout / validation / diagnostics 会非常容易引入行为漂移。

统一执行顺序：

1. **Phase A: 重复收敛与边界解耦（低/中风险）**
   - 先收敛 runtime config 读取 / trainer 属性写回样板。
   - 先收敛 parser/build/train-entry 的 `args` 展开样板。
   - 先收敛 metrics / checkpoint / JSON 落盘模板。
2. **Phase B: 巨热点按职责拆分（中/高风险）**
   - 优先拆 rollout/freerun 核心链路。
   - 其次拆 fit/validation/checkpoint orchestration。
   - 最后拆 diagnostics/contact-meas/keybone 纯分析块。
3. **Phase C: 静默异常清理（中风险）**
   - 建立宽泛异常清单，先区分 active path / debug-only / I/O fallback。
   - 再缩窄热路径中的 `except Exception`。
4. **Phase D: 隐式副作用外移（中风险）**
   - 把 metrics/checkpoint/export/diag dump 从主训练路径剥离成明确提交点。
   - 让 `train_entry` / `main` 保持薄 orchestration 壳。

核心原则：

- one step, one commit
- 每步必须有 before/after 结构指标
- 任何一步回归失败，立即停在当前 commit，不继续后续步骤
- 每步固定汇报 4 项：LOC、`def`/函数数、最大函数长度、`except Exception` 数量
- 每步还必须汇报至少 **1 项本轮主题相关的结构债指标**
- **单步硬门禁**：以 Step 收尾为准，必须满足 `LOC_after <= LOC_before`

新增约束（本路线图强制）：

- 不允许继续把新逻辑直接堆进 `Trainer`
- 不允许新增新的“parser helper + build helper + runtime apply helper”三处重复配置读写
- 不允许“只抽函数不删旧逻辑”
- 新增 helper 必须带来至少 1 项可量化净收益（长度下降 / 重复块下降 / 异常吞没下降 / 属性写回下降）

### 拆分约束（v1，Phase B 强制执行）

拆分前准入（满足任一即可进入候选）：

- 被调用 >=2 次（或下一步已明确会复用 >=2 次）
- 封装了可独立命名的 trainer 领域概念（例如 rollout state、validation snapshot、diag export payload）
- 拆出后调用处更易读（调用点行数或嵌套层级下降）

硬禁止（命中任一则不允许拆）：

- 纯转发 wrapper / 无实质边界的间接层
- 函数名仅复述代码字面行为（无 trainer 领域语义）
- 再次引入黑盒上下文（隐式读写大量 `self.*` / `args.*` 却不声明输入）
- 只抽函数、不删除原地旧逻辑（双实现并存）
- **Step 收尾时** `LOC_after > LOC_before`
- 参数爆炸：helper 参数 > 8 且未收敛到结构化 request/context（dataclass / TypedDict）

单次调用 helper 的例外规则（替代绝对禁止）：

- 调用次数 = 1 允许，但必须同时满足：
  - 具备独立概念命名
  - 调用点可读性提升
  - 至少 1 项结构指标下降（最大函数长度 / `except Exception` / `args.` 访问 / `trainer.*=` 写回）

拆分后验流程（强制）：

1. 先按“复用价值 + 副作用可见性”拆分
2. Step 收尾前检查 `LOC_after <= LOC_before`
3. 若不满足：回收无概念 helper / 参数搬运层 / 纯 wrapper
4. 在**当前 step**完成净减回收后，才允许进入下一步

---

## 0.5) 本轮进展（2026-04-13 / baseline-0）

当前状态仍是**路线图建立阶段，尚未开始代码修改**；先记录基线快照：

- 文件总行数：`6978`
- `def` / 函数数：`154`
- `class` 数：`18`
- 最大类跨度：`Trainer` = `3865` 行（`train/training_MPL.py:483` 起）
- `Trainer` 方法数：`85`
- 最大函数长度：`_compute_contact_and_angvel_metrics = 451` 行（`train/training_MPL.py:4534`）
- 宽泛异常吞没：`except Exception = 91`
- `_phasec_warn_once(...)` 调用：`36`
- `print(...)` 调用：`88`
- `args.` 访问计数：`132`
- `trainer.* = ...` 写回计数：`54`
- `add_argument(...)` 调用：`133`

局部热点长度（基线）：

- `train/training_MPL.py:4534` `_compute_contact_and_angvel_metrics`：`451`
- `train/training_MPL.py:3919` `Trainer._contact_meas_whitebox`：`318`
- `train/training_MPL.py:4987` `_compute_keybone_metrics`：`187`
- `train/training_MPL.py:5834` `_resolve_trainer_runtime_config`：`156`
- `train/training_MPL.py:2888` `Trainer._run_epoch_validation`：`153`
- `train/training_MPL.py:6790` `export_onnx_step_stateful_nophase`：`153`
- `train/training_MPL.py:5611` `_parser_add_model_args`：`143`
- `train/training_MPL.py:1018` `Trainer._lambda_fusion_apply_reliability`：`139`
- `train/training_MPL.py:2708` `Trainer._run_one_train_epoch`：`135`
- `train/training_MPL.py:6505` `_build_train_loss_and_trainer`：`133`
- `train/training_MPL.py:2175` `Trainer._apply_stage_schedule`：`130`
- `train/training_MPL.py:3479` `Trainer._history_drift_debug`：`127`
- `train/training_MPL.py:1771` `Trainer._rollout_forward_step`：`121`

当前结论：

- `train/models.py` 已有独立路线图；下一顺位结构热点确实已经落到 `train/training_MPL.py`
- 真正需要先处理的不是单点算法，而是 `Trainer` 内部的**rollout / fit / validation / diag / metrics / schedule / runtime config** 七类职责耦合

### update-3（2026-04-13 / after Phase A + B2 + B2.5 + partial B1）

截至 `2026-04-13`，已完成：

- `A1`：runtime config 读取 / trainer 属性写回样板收敛
- `A2`：parser / build kwargs 样板收敛
- `A2.5`：metrics / checkpoint / JSON 落盘模板收敛
- `A3`：validation / checkpoint 子系统边界收口
- `B2`：`fit / validation / checkpoint` 主循环第一轮壳层化
- `B2.5`：`free-run diagnostics / contact_meas` 主路径收尾；纯计算、debug fallback、trainer state write-back 边界继续拉开
- `B1`（第一轮 / in progress）：`rollout step input / init / forward / sequence / lambda reliability` 第一轮壳层化

当前结构快照（相对 baseline-0）：

- 文件总行数：`6944`（`-34`）
- `def` / 函数数：`179`（`+25`；B1 收口后仍以 helper 换更薄 orchestration 壳）
- `class` 数：`20`（`+2`；新增 free-run rotation carrier + rollout step input carrier）
- 最大类跨度：`Trainer = 3924` 行（`train/training_MPL.py:506` 起，`+59`）
- `Trainer` 方法数：`117`（`+32`；已回收部分过细粒度 rollout helper）
- 最大函数长度：`export_onnx_step_stateful_nophase = 153` 行（`train/training_MPL.py:6664`，`-298`）
- 宽泛异常吞没：`except Exception = 77`（`-14`）
- `_phasec_warn_once(...)` 调用：`36`（`=`）
- `print(...)` 调用：`87`（`-1`）
- `args.` 访问计数：`135`（`+3`）
- `trainer.* = runtime_cfg.*` 写回计数：`0`（`-54`）

已收敛的局部热点（current）：

- `train/training_MPL.py:5783` `_resolve_trainer_runtime_config`：`116`
- `train/training_MPL.py:2957` `Trainer._run_epoch_validation`：`76`
- `train/training_MPL.py:3433` `Trainer.fit`：`37`
- `train/training_MPL.py:1126` `Trainer._lambda_fusion_apply_reliability`：`35`
- `train/training_MPL.py:1536` `Trainer._resolve_rollout_step_inputs`：`42`
- `train/training_MPL.py:1675` `Trainer._init_rollout_state`：`72`
- `train/training_MPL.py:1972` `Trainer._rollout_forward_step`：`37`
- `train/training_MPL.py:2049` `Trainer._rollout_sequence`：`51`
- `train/training_MPL.py:4320` `Trainer._contact_meas_whitebox`：`44`
- `train/training_MPL.py:5028` `_compute_contact_and_angvel_metrics`：`30`
- `train/training_MPL.py:5060` `_compute_keybone_metrics`：`145`
- `train/training_MPL.py:5266` `_diagnose_free_run_impl`：`41`
- `train/training_MPL.py:5678` `_build_train_parser`：`13`
- `train/training_MPL.py:5692` `_parse_train_entry_args`：`33`
- `train/training_MPL.py:5935` `_build_direct_pose_options`：`12`

### update-4（2026-04-13 / after Phase B1 hotspot pass-2）

本轮继续留在 `train/training_MPL.py` 内推进 `B1`，重点收口 rollout / freerun 链路里尚偏重的“frozen contact fallback + SO(3) residual compose”两段逻辑，并保持 `_rollout_sequence(...)` 继续作为 orchestration 壳。

当前结构快照（相对 update-3）：

- 文件总行数：`6926`（相对 update-3 再降 `18`；相对 baseline-0 为 `-52`）
- `def` / 函数数：`187`（`+8`；新增 helper 限制在文件内、未继续膨胀 `Trainer` 方法数）
- `class` 数：`20`（`=`）
- 最大类跨度：`Trainer = 3833` 行（`train/training_MPL.py:579` 起，较 update-3 收窄）
- `Trainer` 方法数：`117`（`=`）
- 最大函数长度：`export_onnx_step_stateful_nophase = 153` 行（`train/training_MPL.py:6738`，`=`）
- 宽泛异常吞没：`except Exception = 74`（相对 update-3 再降 `3`）
- `_phasec_warn_once(...)` 调用：`36`（`=`）
- `print(...)` 调用：`87`（`=`）
- `args.` 访问计数：`135`（`=`）
- `trainer.* = runtime_cfg.*` 写回计数：`0`（`=`）

本轮 rollout 热点收口结果：

- `train/training_MPL.py:885` `Trainer._predict_pretrain_contacts_from_frozen`：`114 -> 64`
- `train/training_MPL.py:950` `Trainer._compose_delta_to_raw`：`114 -> 78`
- `train/training_MPL.py:1566` `Trainer._update_rollout_carry_state`：`58 -> 53`
- `train/training_MPL.py:2031` `Trainer._rollout_sequence`：维持 `51`，未反向变厚

本轮边界变化：

- frozen-contact 路径里把 **encoder input 组装** 与 **encoder/head forward** 分开：新增 `_build_pretrain_contact_encoder_input(...)`，沿用 `_parse_pretrain_contact_affine_spec(...)` 做 affine/fallback 规整
- delta compose 路径里把 **SO(3) correction 几何变换** 与 **StdY 缩放 / residual compose** 分开：新增 `_apply_so3_correction_to_delta_raw(...)`
- `_update_rollout_carry_state(...)` 仅做轻量收口，继续维持“free carry → scheduled sampling blend → pose history advance → state write-back”顺序，不再引入新的 rollout 参数中转层

### update-5（2026-04-13 / after B2.5 diagnostics follow-up）

本轮从 B1 切到 B2.5 follow-up，优先处理 `_history_drift_debug` 与 `_compute_keybone_metrics`，继续只在 `train/training_MPL.py` 内重构。

当前结构快照（相对 update-4）：

- 文件总行数：`6920`（相对 update-4 再降 `6`；相对 baseline-0 为 `-58`）
- `def` / 函数数：`196`（`+9`；新增 file-local / nested diagnostics helper，`Trainer` 方法数未增加）
- `class` 数：`20`（`=`）
- 最大类跨度：`Trainer = 3747` 行（`train/training_MPL.py:579` 起，较 update-4 收窄）
- `Trainer` 方法数：`117`（`=`）
- 最大函数长度：`export_onnx_step_stateful_nophase = 153` 行（`train/training_MPL.py:6732`，`=`）
- 宽泛异常吞没：`except Exception = 73`（相对 update-4 再降 `1`）
- `_phasec_warn_once(...)` 调用：`36`（`=`）
- `print(...)` 调用：`86`（`-1`）
- `args.` 访问计数：`135`（`=`）
- `trainer.* = runtime_cfg.*` 写回计数：`0`（`=`）

本轮 diagnostics 热点收口结果：

- `train/training_MPL.py:3545` `Trainer._history_drift_debug`：`127 -> 41`
- `train/training_MPL.py:5083` `_compute_keybone_metrics`：`145 -> 53`
- 新增 `_run_history_drift_rollout(...)`、`_compute_history_drift_geo_local_stats(...)`、`_emit_history_drift_debug_lines(...)`
- 新增 `_compute_keybone_detail_metrics(...)`、`_build_keybone_group_summary(...)`

本轮边界变化：

- `_history_drift_debug(...)` 分成 **train_free rollout 执行**、**rot-local geodesic 纯计算**、**HistDrift 文本输出** 三段
- `_compute_keybone_metrics(...)` 分成 **per-keybone detail 纯计算**、**group_mean summary 计算**、**result / diag metric 写回** 三段
- `Trainer` 方法数不变；未把 diagnostics helper 继续塞回 `Trainer`

当前仍为首要剩余热点：

- `train/training_MPL.py:6732` `export_onnx_step_stateful_nophase`：`153`
- `train/training_MPL.py:5579` `_parser_add_model_args`：`143`
- `train/training_MPL.py:3973` `_compute_contact_meas_whitebox_state`：`137`
- `train/training_MPL.py:2792` `Trainer._run_one_train_epoch`：`135`
- `train/training_MPL.py:6448` `_build_train_loss_and_trainer`：`132`
- `train/training_MPL.py:2259` `Trainer._apply_stage_schedule`：`130`
- `train/training_MPL.py:5851` `_resolve_trainer_runtime_config`：`116`

---

## 1) 基线现状（针对本轮热点问题）

当前代码快照核对（`train/training_MPL.py`）：

- **热点问题 1**：`train/training_MPL.py:483` 起的 `Trainer` 类跨度达到 `3865` 行，rollout、fit、validation、metrics、stage schedule、diagnostics、NaN guard 全堆在一个类里。
- **热点问题 2**：rollout/freerun 链路不是一个超长函数，而是一串互相强耦合的 70~140 行 helper：`_lambda_fusion_apply_reliability`、`_compose_delta_to_raw`、`_resolve_rollout_step_inputs`、`_init_rollout_state`、`_rollout_forward_step`、`_rollout_sequence`。
- **热点问题 3**：训练/验证 orchestration 仍然偏重：`_run_one_train_epoch`、`_run_epoch_validation`、`fit`、`_update_best_ckpts`、`_persist_epoch_validation_outputs` 之间混合了优化、评估、落盘、ckpt 更新和日志。
- **热点问题 4**：诊断热点偏重：`_contact_meas_whitebox`、`_compute_contact_and_angvel_metrics`、`_compute_keybone_metrics` 这几个函数已经接近“子系统”级别，但还挂在同一文件的主训练语境里。
- **热点问题 5**：CLI / build / runtime config 面很宽：6 个 `_parser_add_*` helper、`_parse_train_entry_args`、`_resolve_trainer_runtime_config`、`_apply_trainer_runtime_config`、`_build_train_*` 系列和 `train_entry()` 串在一起，导致 `args` 扩散面很大。
- **热点问题 6**：宽泛异常吞没仍然很多。当前 `except Exception` 出现 `91` 次，其中不少落在训练热路径、rollout 热路径和 validation / export 过渡路径。
- **热点问题 7**：I/O 副作用散落：metrics JSON、summary JSON、teacher/free diag dump、checkpoint 保存、post-fit valfree、ONNX export 都与训练主逻辑交织。

结构指标基线（本路线图起点）：

- LOC: `6978`
- `def` / 函数数: `154`
- `class` 数: `18`
- 最大类跨度: `Trainer = 3865`
- 最大函数长度: `_compute_contact_and_angvel_metrics = 451`
- 主题结构债指标 #1: `except Exception = 91`
- 主题结构债指标 #2: `args.` access count = `132`
- 主题结构债指标 #3: `trainer.*=` write count = `54`
- 主题结构债指标 #4: `print(...)` count = `88`

为什么现在轮到 `train/training_MPL.py`：

- `train/models.py` 已经有独立路线图，且边界收口已经开始。
- `training_MPL.py` 现在是 basetrain 侧最强的“编排 + 诊断 + 输出 + CLI”多职责耦合点。
- 如果不先给 `training_MPL.py` 建边界，后续无论是继续整理 `posttrain` 还是抽共享 eval/runtime，都容易被 basetrain 的上下文噪音反向污染。

---

## 2) 具体改动流程

## Phase A — 重复收敛与边界解耦（A1 + A2 + A2.5 + A3）

### Step A1 — 收敛 runtime config 读取 / trainer 属性写回样板（低/中风险）

目标：将 `train/training_MPL.py:5834` `_resolve_trainer_runtime_config` 与 `train/training_MPL.py:5992` `_apply_trainer_runtime_config` 的大段字段读写收敛成分组化边界。

实施：

- 先按子系统分组：
  - normalizer / layout
  - contacts / contact_meas
  - eval / monitor
  - history / schedule
  - output / metadata
- 将“parse/coerce default” 与 “apply to trainer attributes” 明确拆开。
- 优先引入字段绑定表或小型 grouped helper，减少平铺式 `trainer.xxx = runtime_cfg.xxx`。

约束：

- 不改 `TrainerRuntimeConfig` 对外字段语义
- 不改 trainer 运行时属性名
- 不改 `train_entry()` 对 runtime config 的调用时机

验收门：

- `_apply_trainer_runtime_config` 长度显著下降
- `trainer.*=` 写回计数下降
- runtime 字段集合保持一致

### Step A2 — 收敛 parser / config override / build kwargs 样板（中风险）

目标：把 parser / `args` 展开 / build kwargs 的样板先收口，再拆更大的训练路径。

建议对象：

- `_build_train_parser`
- `_parse_train_entry_args`
- `_build_train_model`
- `_build_train_loss_and_trainer`
- `_build_direct_pose_options`

实施：

- 保留 6 个 parser adders 的分组语义，但把重复的 `add_argument` / choices / bool parser / help 组织成更明确的 subsystem block。
- `model_kwargs` / `loss_fn` kwargs / trainer kwargs 分成命名 builder。
- `_parse_train_entry_args` 只保留“config defaults -> parse -> override -> validate required”的主流程。

约束：

- 不改变 CLI flag / default / help 文本语义
- 不改变 `--config_json` / `--config_override` 行为
- 不提前删除历史兼容报错入口（例如 retired `whitebox` 报错）

验收门：

- parser/build 路径的 `args.` 扩散下降
- `train/training_MPL.py --help` 输出语义保持一致
- `model_kwargs` / loss / trainer build 更容易独立阅读

### Step A2.5 — 收敛 metrics / checkpoint / JSON 落盘模板（中风险）

目标：优先收敛 metrics / checkpoint / summary / diag dump 的重复模板，让训练主路径不再承载过多 I/O 细节。

建议对象：

- `_persist_epoch_metrics_artifacts`
- `_dump_metrics_json`
- `_save_val_metrics`
- `_write_basetrain_keybone_group_summary`
- `_fit_checkpoint_payload`
- `_save_fit_checkpoint_payload`
- `_run_epoch_validation` 内 teacher/free diag JSON dump

实施：

- 先引入统一的 metrics sink / checkpoint sink helper（不要求跨文件）
- 主调用点尽量只保留：
  - payload 构造
  - tag / epoch / target path
  - 是否执行的策略判断

约束：

- 不改变 JSON key set / ckpt filename / best ckpt 选择语义
- 不改变 `metrics/*.json` 与 `basetrain_keybone_group_summary.json` schema（后续已去掉内存侧历史缓冲）

验收门：

- 写文件模板重复块数量下降
- `metrics/*.json` schema 保持一致
- `best_teacher` / `best_free` / `last` 行为保持一致

### Step A3 — 建立 `Trainer` 子系统边界（中风险）

目标：先在同文件内把 `Trainer` 划成明确子系统，而不是立刻跨文件搬迁。

建议边界：

- rollout state / step transition
- fit / validation / checkpoint
- stage schedule / adaptive tuning
- diagnostics / contact-meas / drift

实施：

- 优先用 request/result dataclass 收口隐式上下文
- 减少 helper 对大量 `self.*` 的黑盒读取
- 把“纯计算”和“日志 / 落盘 / debug side effect”从同一个 helper 中分开

约束：

- 暂不改 `Trainer` 对外接口
- 暂不跨文件拆分

验收门：

- `Trainer` 内部调用图更清晰
- 至少 1 个子系统形成“orchestration 壳 + 纯 helper”边界
- 至少 1 项主题结构债指标下降

### Phase A 当前进度（2026-04-13）

- Step A1（Completed / 2026-04-13）
  - `runtime config` 按 `output / normalizer-layout / contacts / eval-monitor / history-schedule` 分组收敛
  - `trainer.* = runtime_cfg.*` 写回计数：`54 -> 0`
- Step A2（Completed / 2026-04-13）
  - parser adders 改为分组循环；`_parse_train_entry_args` / `direct-pose build` 样板缩短
  - `_build_direct_pose_options`：`22 -> 12`
- Step A2.5（Completed / 2026-04-13）
  - 新增统一 `metrics/json/checkpoint` sink
  - `fit` 中 train/teacher/valfree metrics 与 checkpoint finalize 模板显著下降
- Step A3（Completed / 2026-04-13）
  - 建立 `ValidationRuntimeContext`
  - validation / checkpoint 主路径形成“orchestration 壳 + helper”边界
- Phase A 累计结果：
  - LOC：`6978 -> 6945`
  - `Trainer._run_epoch_validation`：`153 -> 76`
  - `Trainer.fit`：`83 -> 73`（Phase A 收尾时）

---

## Phase B — 巨热点职责拆分（B1 + B2 + B2.5）

### Step B1 — 拆 rollout / freerun 核心链路（高风险）

建议边界：

- 输入解析：`_resolve_rollout_step_inputs`
- 状态初始化：`_init_rollout_state`
- 单步推进：`_rollout_forward_step`
- carry 更新：`_update_rollout_carry_state`
- λ / Δpose 几何合成：`_lambda_fusion_apply_reliability`、`_apply_lambda_fusion_to_raw`、`_compose_delta_to_raw`
- 序列编排：`_rollout_sequence`

强制要求：

- `_rollout_sequence` 收敛成 orchestration 壳
- 不允许保留旧逻辑镜像副本
- 输出 tensor keyset / shape / dtype 语义不变

回归门：

- `python3 -m py_compile train/training_MPL.py train/models.py train/posttrain.py`
- `python3 tools/check_lambda_fusion_blend_geometry.py`
- rollout 相关最小 smoke 路径通过

### Step B2 — 拆 fit / validation / checkpoint 主循环（高风险）

建议边界：

- batch 执行：`_run_one_train_batch`
- epoch orchestration：`_run_one_train_epoch`
- validation orchestration：`_run_epoch_validation`
- metrics 持久化：`_persist_epoch_validation_outputs`
- best-ckpt 选择：`_update_best_ckpts`
- 顶层训练壳：`fit`

强制要求：

- `fit` 只保留 epoch 编排壳
- `_run_epoch_validation` 不再同时承担“评估 + console 汇总 + JSON dump + best-source 决策”四类职责
- checkpoint 更新必须显式依赖 validation payload，不再隐式读多个旁路字段

回归门：

- train / teacher / valfree 指标 key 集合一致
- `best_teacher` / `best_free` 选择语义一致
- 允许浮点微小扰动，不允许 tag/keyset 漂移

### Step B2.5 — 拆 diagnostics / contact-meas / keybone 分析块（高风险）

建议边界：

- `_contact_meas_whitebox`
- `_compute_input_drift_metrics`
- `_compute_contact_and_angvel_metrics`
- `_compute_keybone_metrics`
- `_diagnose_free_run_impl`
- `_history_drift_debug`

强制要求：

- diagnostics 计算与日志/文件输出分离
- debug-only fallback 与 active path fail-fast 分离
- contact-meas 诊断逻辑不再继续反向侵入训练主路径

回归门：

- free-run 诊断 key 集合一致
- keybone/contact/angvel 曲线字段保持兼容
- 允许 debug payload 内容顺序调整，不允许字段缺失

### Phase B 当前进度（2026-04-13 / update-5）

- Step B1（In Progress / 2026-04-13）
  - `_lambda_fusion_apply_reliability`：`139 -> 35`
  - `_resolve_rollout_step_inputs`：`105 -> 42`
  - `_init_rollout_state`：`105 -> 72`
  - `_predict_pretrain_contacts_from_frozen`：`114 -> 64`
  - `_compose_delta_to_raw`：`114 -> 78`
  - `_rollout_forward_step`：`121 -> 37`
  - `_rollout_sequence`：`70 -> 51`
  - `_update_rollout_carry_state`：`58 -> 53`
  - 新增 `_resolve_rollout_step_angvel(...)`、`_resolve_rollout_step_cond_raw(...)`、`_run_rollout_model_step(...)`、`_maybe_apply_rollout_lambda_fusion(...)`、`_finalize_rollout_outputs(...)`
  - 新增 `_build_pretrain_contact_encoder_input(...)`、`_apply_so3_correction_to_delta_raw(...)`
  - `_apply_scheduled_sampling_update(...)` 不再用 `SimpleNamespace` 人工打包中转；`_update_rollout_carry_state(...)` 直接接收 `rollout + rollout_inputs + step_idx`
  - `_resolve_rollout_ss_chunk_len(...)` / `_resolve_rollout_rot6d_slices(...)` / `_resolve_rollout_time_dim_flags(...)` / `_resolve_rollout_time_base_local(...)` 已回收到 `_init_rollout_state(...)`
  - 本轮完成后：
    - `_rollout_sequence(...)` 继续保持 orchestration 壳；未反向长回去
    - frozen contact fallback 与 SO(3) compose 几何已从主流程里拆出纯计算边界
    - `_update_rollout_carry_state(...)` 仍有进一步拆“free carry / scheduled sampling blend / pose-hist advance”的空间，但已不是当前最大结构热点
- Step B2（Completed / 2026-04-13）
  - 新增 `_run_fit_epoch_cycle(...)` 与 `_finalize_fit_checkpoints(...)`
  - `Trainer.fit`：`83 -> 37`
  - checkpoint 更新继续显式依赖 validation payload
- Step B2.5（Completed / 2026-04-13）
  - `_diagnose_free_run_impl`：`215 -> 41`
  - `Trainer._contact_meas_whitebox`：`318 -> 44`
  - `_compute_contact_and_angvel_metrics`：`451 -> 30`
  - `_compute_keybone_metrics`：`187 -> 53`
  - `Trainer._history_drift_debug`：`127 -> 41`
  - 新增 `_compute_contact_meas_whitebox_state(...)`、`_record_geodesic_diag_metrics(...)`、`_record_angvel_diag_metrics(...)`、`_record_period_contact_hint_metrics(...)`
  - 新增 `_run_history_drift_rollout(...)`、`_compute_history_drift_geo_local_stats(...)`、`_emit_history_drift_debug_lines(...)`
  - 新增 `_compute_keybone_detail_metrics(...)`、`_build_keybone_group_summary(...)`
  - diagnostics 纯计算与 debug/log fallback 继续分离；free-run diagnostics keyset / curve 字段保持兼容

---

## Phase C — 静默异常清理（C1 + C2）

### Step C1 — 建立异常点清单与级别（低风险）

目标：把 `training_MPL.py` 里的 `except Exception` 先分类，不立即全部删除。

优先清单：

- rollout active path：
  - `_predict_pretrain_contacts_from_frozen`
  - `_compose_delta_to_raw`
  - `_run_one_train_batch`
- validation / fit path：
  - `_run_epoch_validation`
  - `_run_one_train_epoch`
  - `_save_adjusted_config`
- entry / export path：
  - `_prepare_motion_encoder_and_contacts_runtime`
  - `_run_postfit_actions`
  - `_export_postfit_onnx`
  - `export_onnx_step_stateful_nophase`

分类建议：

- 应 fail-fast（训练主路径 / rollout 主路径）
- 可保留窄异常（数值标量化 / JSON 序列化 / 文件写入）
- debug-only fallback（history drift / export probe / diag dump）

验收门：

- 形成位置 / 原因 / fallback 语义清单
- 明确每个异常块属于 hot path 还是 debug path

### Step C2 — 清理热点路径中的广义异常吞没（中风险）

目标：优先减少训练热路径与 rollout 热路径中的 `except Exception`。

约束：

- 先处理 active path，再处理外围 debug path
- 每处替换必须明确 fallback 语义
- `warn_once` 仅保留给可恢复但非核心语义路径

验收门：

- `except Exception` 计数下降
- fit / rollout / validation 主路径中的广义异常显著下降
- 训练和导出回归路径保持一致

---

## Phase D — 副作用剥离（D1 + D2）

### Step D1 — 将 metrics / checkpoint / diag dump 从主计算路径剥离（中风险）

目标：避免 `fit` / `_run_epoch_validation` / `_persist_epoch_validation_outputs` 同时承担“算结果”和“写结果”。

方向：

- 主路径先产出标准化 payload
- 统一在 finalize/sink 阶段决定：
  - 是否记录 metrics
  - 是否写 JSON
  - 是否更新 checkpoint
  - 是否输出 diag dump

### Step D2 — 将 CLI / post-fit / ONNX export 收敛成薄入口（中风险）

目标：让 `train_entry` / `main` 保持薄壳，post-fit 和 export 成为明确附加动作。

方向：

- `train_entry` 只保留 build -> runtime -> fit -> postfit orchestration
- `_run_postfit_actions` 和 `_export_postfit_onnx` 进一步收口
- `export_onnx_step_stateful_nophase` 长期可作为独立导出 helper 候选，但本轮优先先在文件内收边界

---

## 3) 本轮建议的实际起手顺序

建议顺序：

1. `A1`: 收敛 runtime config 读取 / trainer 属性写回样板
2. `A2.5`: 收敛 metrics / checkpoint / JSON 落盘模板
3. `A3`: 建立 `Trainer` 子系统边界
4. `A2`: 收敛 parser / build kwargs 样板
5. `B1`: 拆 rollout / freerun 核心链路
6. `B2`: 拆 fit / validation / checkpoint 主循环
7. `B2.5`: 拆 diagnostics / contact-meas / keybone 分析块
8. `C1/C2`: 清理广义异常吞没
9. `D1/D2`: 外移副作用并收薄入口

理由：

- `training_MPL.py` 当前最大的阻塞不是算法，而是**主路径周围的配置/落盘/诊断噪音太重**。
- 如果直接先拆 rollout，会被 runtime config、metrics sink、validation side effect 的上下文耦合拖慢。
- 先做 `A1 + A2.5 + A3`，能先把“训练主路径到底依赖哪些状态、哪些输出是计算、哪些是 I/O”收清楚。
- 这样再做 `B1/B2`，更容易把 `Trainer` 收敛成一层真正的 orchestration shell。

截至 `2026-04-13` 的**实际执行顺序**：

1. `A1`
2. `A2.5`
3. `A3`
4. `A2`
5. `B2`
6. `B2.5`（partial）

当前建议的**后续顺序**：

1. `B2.5` 收尾（优先 `_contact_meas_whitebox` / `_compute_contact_and_angvel_metrics`）
2. `B1`
3. `C1/C2`
4. `D1/D2`

---

## 4) 后续优先级（文件级）

当前建议的仓内清理优先级：

1. `train/training_MPL.py`
2. `train/eval_utils.py`
3. `train/posttrain.py`

说明：

- `train/models.py` 已经有独立路线图，本轮不混做。
- `training_MPL.py` 一旦把 rollout / validation contract 收紧，`train/eval_utils.py` 会成为下一步自然的共享边界承接点。
- `train/posttrain.py` 建议等 basetrain runtime contract 稳定后再继续整理，避免两边同时漂移。

---

## 5) 建议验证命令（每步最小固定集）

```bash
python3 -m py_compile train/training_MPL.py train/models.py train/posttrain.py
python3 tools/check_standard_rotvec_semantics.py
python3 tools/check_lambda_fusion_blend_geometry.py
python3 -m train.training_MPL --help
```

补充说明：

- 当前仓内 `train/training_MPL.py` 使用相对导入，直接执行 `python3 train/training_MPL.py --help` 会失败。
- `--help` 回归应统一使用 module invocation：`python3 -m train.training_MPL --help`。

如果某一步已经触及导出/入口编排，再补：

```bash
python3 -m train.posttrain --help
```

---

## 6) 备注

- 这份路线图已开始记录实际推进结果，不再只是“待执行清单”。
- 当前最重要的不是跨文件搬迁，而是先把 `Trainer` 里的 runtime / rollout / validation / diagnostics / I/O 边界收清楚。
- `A1 + A2 + A2.5 + A3` 已完成；当前优先级已转到 `B2.5` 收尾与 `B1` rollout 核心链路。
