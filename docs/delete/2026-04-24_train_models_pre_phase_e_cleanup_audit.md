# [2026-04-24] `train/models.py` pre-Phase-E deletion audit / cleanup inventory

Date: 2026-04-24  
Status: Static-audit ready  
Scope: `train/models.py`（以 `EventMotionModel.forward(...)` Phase D 收尾后的单文件去重 / cleanup 为主），必要时引用 `tests/*`、`docs/*`、`train/eval_utils.py`、`train/rollout_kernel.py`、`train/diagnostics.py` 作为调用面证据。  
Method: 静态引用分析（`rg` / AST / 代码阅读） + 引用 2026-04-24 Phase D final validation baseline（未在本 audit pass 内重跑 fresh rerun）  

Goal: 为 Phase E 之前的“单文件内结构化收口”提供**可执行清单**，优先处理重复代码、冷分支和可收窄 helper，确保后续跨文件迁移尽量是 mechanical move。  
Non-goal:
- 不改算法/数学定义。
- 不改默认训练行为。
- 不改 checkpoint/load contract。
- 不在本轮直接拆文件。

关联参考：

- 模板：`docs/templates/delete/delete_audit_template.md`
- 当前结构基线：`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`
- fail-fast / broad handler 基线：`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`
- 历史 cleanup inventory：`docs/delete/2026-04-17_train_models_cleanup_inventory.md`

---

## 0) 一页版结论

按当前证据，候选项分为：

### A. Remove-Now

1. 暂无。Phase D 刚收尾，当前最值钱的下一步不是直接删，而是继续把单文件内重复控制流压平。

### B. Remove-If-Clean

1. `EventMotionModel._apply_direct_pose_leg_side_plan_other_ablation` — `train/models.py:2902`，当前只有一个 runtime caller；如果 side-routing cleanup 重新开启，且该 caller 被一并移除，可作为第一批删除候选。

### C. Dedup-First

1. direct-pose leg gate / scale 双分支重复块 — `train/models.py:4306`, `train/models.py:4534`, `train/models.py:4573`, `train/models.py:4652`, `train/models.py:4676`  
   先公共化 learned-gate / scale-gate 逻辑，再删重复块。
2. `EventMotionModel.forward(...)` tail post-processing shell — `train/models.py:4715`, `train/models.py:4806`, `train/models.py:4819`  
   先抽 `lambda_fusion` / `so3 corrector` / `period_pred` 的尾段壳，再考虑更细 cleanup。

### D. Keep / Revisit

1. `EventMotionModel._compute_direct_pose_leg_cross_leg_ablation` — `train/models.py:2922`，仍有真实 runtime caller，不能当 dead helper 处理。
2. `EventMotionModel._canonicalize_contacts_meas_inputs` — `train/models.py:2797`，Phase C 已完成去重且有 focused tests，当前应保留为 canonical helper。
3. direct-hint override pair — `train/models.py:2748`, `train/models.py:2771`，已经是去重后的公共入口，不应回退。
4. contact-plan debug helper quartet — `train/models.py:2681`, `train/models.py:2694`, `train/models.py:2711`, `train/models.py:2731`，虽然偏 debug，但 forward 与 tests 均活跃依赖。
5. `period_pred` / `so3_delta_corrector` tail outputs — `train/models.py:4806`, `train/models.py:4819`，repo 外围 eval / rollout / diagnostics 仍消费，不应直接删。

---

## 1) 状态标记

- `Remove-Now`：静态引用基本清空，删除前只需轻量 smoke。
- `Remove-If-Clean`：看起来是 cold/deletable，但还需要确认当前 caller / docs / tests / config。
- `Dedup-First`：不是直接删除项；先公共化/收敛重复逻辑，再删除旧块。
- `Keep-Guard`：承担 fail-fast 或用户友好报错，不应直接删。
- `Keep-Compat`：薄 wrapper / 老名字仍是外部稳定 API。
- `Keep-Active`：训练、posttrain、validate、checkpoint 或 diagnostics 仍依赖。
- `Revisit-With-Rerun`：静态偏冷，但需要 fresh rerun / contract review 才能继续推进。

---

## 2) 证据扫描

### 2.1 静态引用

```bash
rg -n "_apply_direct_pose_leg_side_plan_other_ablation|_compute_direct_pose_leg_cross_leg_ablation|_canonicalize_contacts_meas_inputs|_canonicalize_direct_hint_override|_apply_direct_hint_override|_init_contact_plan_debug_buffers|_append_contact_plan_debug_logits|_finalize_contact_plan_debug_logits|_write_contact_plan_debug_logits|period_pred|so3_delta_corrector|lambda_fusion forward failed" train tests docs tools config
```

记录：

- 总命中数: `87`
- 代码命中: `63`
- 测试命中: `14`
- 文档命中: `10`
- 只剩本 audit 文档命中: `no`

补充局部扫描：

- `_apply_direct_pose_leg_side_plan_other_ablation`：总命中 `4`，其中 runtime caller 仅 `train/models.py:4306`
- `_compute_direct_pose_leg_cross_leg_ablation`：总命中 `10`，其中 runtime caller 仅 `train/models.py:4592`
- `_canonicalize_contacts_meas_inputs`：总命中 `8`，runtime callers 为 `train/models.py:3560` / `train/models.py:3718`
- direct-hint override pair：总命中 `5`，runtime callers 为 `train/models.py:4011` / `train/models.py:4044`
- contact-plan debug helpers：总命中 `15`，forward + tests 均活跃
- tail outputs（`period_pred` / `so3_delta_corrector` / lambda_fusion tail）命中 `45`，外围 `train/eval_utils.py`、`train/diagnostics.py`、`train/rollout_kernel.py` 仍消费

### 2.2 Runtime / config / checkpoint 证据（引用当前基线）

```bash
python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_phase_d_final_smoke_20260424 --run_name train_models_phase_d_final_smoke_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0
```

记录：

- scanned runs: `2`（引用 2026-04-24 Phase D final baseline）
- hit files / runtime hits: `N/A`（本 audit pass 未新增 instrumentation）
- unreadable / blocked: `0`
- blocked 分类: `none`

说明：

- 本 audit 文档本身未重跑 fresh rerun；这里引用的是当前最新 Phase D closeout baseline，用于判断“哪些 helper/output 仍明显活跃”。

---

## 3) Inventory 总表

| Item | Location | 类型 | 当前证据 | 建议 | 删除前门禁 |
|---|---|---|---|---|---|
| direct-pose leg gate / scale duplicated runtime blocks | `train/models.py:4306`, `train/models.py:4534`, `train/models.py:4573`, `train/models.py:4652`, `train/models.py:4676` | dup | side-routed 与 non-side 两套 learned/scale gate 高度相似，仅 routing 输入不同 | `Dedup-First` | snapshot + stage6 smoke + failfast tests |
| forward tail post-processing shell | `train/models.py:4715`, `train/models.py:4806`, `train/models.py:4819` | dup/orchestration hotspot | tail 仍以内联块承接 `lambda_fusion` / `so3` / `period_pred` | `Dedup-First` | snapshot + stage6 smoke |
| `_apply_direct_pose_leg_side_plan_other_ablation` | `train/models.py:2902` | cold helper | 仅 `1` 个 runtime caller + `2` 个 docs 命中，无 tests 直接覆盖自身语义 | `Remove-If-Clean` | side-routing caller audit + snapshot + smoke |
| `_compute_direct_pose_leg_cross_leg_ablation` | `train/models.py:2922` | active helper | `1` 个 runtime caller + `4` 个 failfast tests；仍参与 non-side leg residual path | `Keep-Active` | 若 caller 归零再重审 |
| `_canonicalize_contacts_meas_inputs` | `train/models.py:2797` | canonical helper | 已被两处 runtime caller 复用，且有 `tests/train/test_event_motion_model_phase_c_contacts_meas.py` focused coverage | `Keep-Active` | 保持 helper 单一实现 |
| direct-hint override pair | `train/models.py:2748`, `train/models.py:2771` | canonical helper | 当前仅 `2` 个 runtime caller，helper 化收益已兑现 | `Keep-Active` | 不回退为内联重复逻辑 |
| contact-plan debug helper quartet | `train/models.py:2681`, `train/models.py:2694`, `train/models.py:2711`, `train/models.py:2731` | debug contract | forward 与 `tests/train/test_train_models_failfast.py` 均活跃依赖 | `Keep-Active` | 如 retire debug logits contract，再单独开 audit |
| `period_pred` / `so3_delta_corrector` tail outputs | `train/models.py:4806`, `train/models.py:4819` | active output contract | `train/eval_utils.py` / `train/diagnostics.py` / `train/rollout_kernel.py` 仍消费 | `Keep-Active` | 任何改动都要连带外围调用面 scan |

---

## 4) 推荐执行顺序

### Phase A — no-behavior single-file dedup

目标：继续留在 `train/models.py` 内，把最明显的 orchestration / duplicate runtime block 压平，不进入跨文件迁移。

候选：

- `train/models.py:4715` tail shell（`lambda_fusion` / `so3` / `period_pred`）
- `train/models.py:4306` + `train/models.py:4534`–`train/models.py:4676` direct-pose leg gate/scale 重复块

删除前 / 改动前 checklist：

- [ ] `python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`
- [ ] `python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`
- [ ] forward snapshot: `tests.train.test_event_motion_model_refactor_phase_d.test_forward_output_snapshot_deterministic_regression`
- [ ] state_dict fingerprint: `tests.train.test_event_motion_model_refactor_phase_d.test_state_dict_fingerprint_repeated_construction_regression`
- [ ] stage6 smoke

最终勾选：

- [ ] Dedup landed
- [ ] Keep as-is
- [ ] Revisit

### Phase B — side-routing cold helper cleanup

目标：只在 side-routing caller 明确收窄后，再决定是否删 `_apply_direct_pose_leg_side_plan_other_ablation(...)`。

候选：

- `train/models.py:2902` `_apply_direct_pose_leg_side_plan_other_ablation`

额外门禁：

- [ ] `rg -n "_apply_direct_pose_leg_side_plan_other_ablation" train tests docs tools config`
- [ ] 确认 `train/models.py:4306` caller 是否仍保留
- [ ] side-routing removal plan 与当前 forward 真实路径一致
- [ ] snapshot / stage6 smoke 通过

### Phase C — Phase E preflight cleanup boundary

目标：在真正进入跨文件迁移前，把“应保留的 active helpers”和“未来可删的 cold helpers”边界写清楚。

候选：

- `train/models.py:2922` `_compute_direct_pose_leg_cross_leg_ablation`
- `train/models.py:2797` `_canonicalize_contacts_meas_inputs`
- `train/models.py:2681`–`train/models.py:2731` contact-plan debug helper quartet

额外门禁：

- [ ] before/after keyset 一致
- [ ] focused numerical smoke 通过
- [ ] 没有 helper 被一边保留一边重新内联，造成新旧双轨

---

## 5) 单项记录模板

### direct-pose leg gate / scale duplicate blocks

**位置**

- `train/models.py:4306`
- `train/models.py:4534`
- `train/models.py:4573`
- `train/models.py:4652`
- `train/models.py:4676`

**它是什么**

- direct-pose leg residual forward 中，side-routed 与 non-side 两条路径各自维护一套 learned gate / scale gate 逻辑。

**静态证据**

- 相同类别的错误标签出现 4 次：side-routed learned/scale、non-side learned/scale。
- 逻辑都围绕 `gate/scale head -> logits -> sigmoid/exp -> optional clamp/power -> apply omega` 展开。

**风险点**

- 直接删任一块都可能影响 `direct_leg_gate*` / `direct_leg_scale*` stats contract。

**当前建议**

- `Dedup-First`

**删除前 checklist**

- [ ] forward snapshot
- [ ] state_dict fingerprint
- [ ] stage6 smoke

**执行回填**

- [ ] 去重完成
- [ ] 验证完成
- [ ] 如未推进，记录保留原因

### `_apply_direct_pose_leg_side_plan_other_ablation`

**位置**

- `train/models.py:2902`

**它是什么**

- side-routing path 专用的 plan-other ablation helper。

**静态证据**

- repo 内总命中 `4`；其中 runtime caller 仅 `train/models.py:4306`，另外 `2` 个命中来自历史 side-routing removal plan。

**风险点**

- 当前仍有 live caller；若误删会直接影响 side-routing branch。

**当前建议**

- `Remove-If-Clean`

**删除前 checklist**

- [ ] `rg -n "_apply_direct_pose_leg_side_plan_other_ablation" train tests docs tools config`
- [ ] caller audit
- [ ] forward snapshot + stage6 smoke

**执行回填**

- [ ] 删除完成
- [ ] 验证完成
- [ ] 如未删除，记录保留原因

---

## 6) 验证门禁

| 验证层 | 命令 | 必须/可选 | 结果 |
|---|---|---:|---|
| compile | `python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py` | 必须 | [passed] |
| focused unit | `python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d` | 必须 | [passed] |
| forward snapshot | `python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression` | 必须 | [passed] |
| state_dict fingerprint | `python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression` | 必须 | [passed] |
| runtime smoke | `PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_pre_phase_e_simplify_batch2_20260424 --run_name train_models_pre_phase_e_simplify_batch2_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0` | 按 `forward` 风险必跑 | [passed] |
| static audit grep | `git diff -U0 -- train/models.py tests/train/test_event_motion_model_refactor_phase_d.py docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md docs/changes/2026-04-21_train_models_fail_fast_inventory.md docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md \| rg -n "<removal-policy §6 patterns>"` | 必须 | [passed] |

Stop-rule:

- snapshot 或 stage6 smoke 失败：停止 cleanup，先回滚或收窄改动面。
- 删除后出现新旧双轨：回收旧实现或回滚，不进入 Phase E。
- 若 caller 仍活跃：停止删除，改为 `Keep-Active` 或 `Dedup-First`。

---

## 7) Final report 回填

- 删除项: `none`
- 保留项: `forward tail outputs 与 direct-pose leg gate/scale contract 均继续保留为 active output contract；只做单文件壳层收口`
- 验证结果: `passed（compile / focused unit / snapshot / fingerprint / stage6 smoke / diff-grep）`
- 行为/contract 影响: `expected unchanged；output key set、state_dict fingerprint、stage6 smoke 保持一致`
- 后续 cleanup: `继续 simplification batch 3，优先 side-routed / non-side leg feature assembly 与 omega pre-gate shell；暂不进入 Phase E`

### Implementation Update — 2026-04-24 pre-Phase-E simplification batch 1 forward tail shell

- **本 batch 目标**：落实本 audit 在 `forward tail post-processing shell` 条目里的 `Dedup-First` 第一批，只收 `lambda_fusion` / `so3_delta_corrector` / `period_pred` 的尾段壳层，不碰 direct-pose leg gate/scale duplicated runtime block，不进入跨文件迁移。
- **实际完成项**：`train/models.py` 新增 `_lambda_fusion_rollout_step_feature(...)`、`_write_forward_lambda_fusion_outputs(...)`、`_write_forward_so3_delta_outputs(...)`、`_write_forward_period_output(...)`；`forward(...)` 尾段现在从 direct-pose final writeback 线性进入这三段 tail helper dispatch。未改 output key set、lambda/so3/period 数值语义、checkpoint contract、默认超参。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：简化前 tail 仍以内联块承接 `lambda_fusion` / `so3` / `period_pred`；简化后 tail 结构为 `_write_forward_lambda_fusion_outputs(...)` → `_write_forward_so3_delta_outputs(...)` → `_write_forward_period_output(...)`。`forward tail post-processing shell` 这一项从 audit 视角已完成第一批 `Dedup-First`，但 direct-pose leg gate/scale duplication 仍未动。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：更新 `tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_shell_dispatch_smoke_regression`，新增 tail helper dispatch 顺序断言；snapshot / state_dict fingerprint regression 直接复用。
- **forward snapshot 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`。
- **state_dict fingerprint 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`。
- **stage6 deterministic smoke 验证方式**：使用输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch1_20260424` 运行 5-step deterministic smoke，结果 `ok_steps=5 skipped=0`。
- **运行过的命令**：compile / focused unit / snapshot / state_dict fingerprint / AST broad-count / stage6 deterministic smoke / touched diff removal-policy grep。
- **验证结果**：全部通过；snapshot、fingerprint、stage6 smoke 均证明行为一致。
- **阻塞项 / 风险**：tail shell 已压平，但 direct-pose leg gate / scale duplicated runtime block 仍是更大的下一处 mechanical simplification 热点；若直接进入 Phase E，会把这部分单文件内可验证的结构压平与跨文件迁移耦合到同一批。
- **下一步建议动作**：继续 simplification batch 2，优先 direct-pose leg gate / scale duplicated runtime block；暂不进入 Phase E。

### Implementation Update — 2026-04-24 pre-Phase-E simplification batch 2 direct-pose leg gate/scale shell

- **本 batch 目标**：落实本 audit 中 `direct-pose leg gate / scale duplicate blocks` 的 `Dedup-First` 第一批，统一 side-routed 与 non-side 两条路径的 learned / scale gate apply 壳层；不进入跨文件迁移，不改 output key set、数值语义、checkpoint contract。
- **实际完成项**：`train/models.py` 新增 `_DirectPoseLegGateOutputs` 与 `_apply_direct_pose_leg_gate_outputs(...)`。side-routed 与 non-side 两条路径中重复的 `gate head -> sigmoid/power`、`scale head -> exp/clamp/log`、`direct_leg_gate*` / `direct_leg_scale*` 写回、`omega_eff` 生成逻辑已统一走该 helper；分支内只保留各自 feature assembly、side scatter index 与 `omega_leg` 准备。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：简化前 `train/models.py:4663`–`train/models.py:4734` 与 `train/models.py:4790`–`train/models.py:4837` 是两套几乎同形的 gate/scale block；简化后两条路径都统一调用 `train/models.py:2852` 的 `_apply_direct_pose_leg_gate_outputs(...)`。`direct-pose leg gate / scale duplicate blocks` 这一审计项已完成 apply-shell 级别的第一批 dedup，但 feature assembly duplication 仍未动。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_event_motion_model_refactor_phase_d.py:587` 新增 `test_forward_leg_gate_helper_dispatch_regression`；`tests/train/test_event_motion_model_refactor_phase_d.py:643` 新增 `test_forward_leg_scale_helper_dispatch_regression`。两条测试都覆盖 `side_routing=False/True`，锁定 `_apply_direct_pose_leg_gate_outputs(...)` dispatch 与 `direct_leg_gate` / `direct_leg_scale` 输出 contract。
- **forward snapshot 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`。
- **state_dict fingerprint 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`。
- **stage6 deterministic smoke 验证方式**：使用输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch2_20260424` 运行 5-step deterministic smoke，结果 `ok_steps=5 skipped=0`。
- **运行过的命令**：compile / focused Phase D unit / full unit / snapshot / state_dict fingerprint / AST broad-count / stage6 deterministic smoke / touched ranges removal-policy grep。
- **验证结果**：全部通过；联合 `unittest` 总数更新为 `129`（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=11`），snapshot、fingerprint、stage6 smoke 均证明行为一致。
- **阻塞项 / 风险**：gate/scale apply 已收口，但 side-routed 与 non-side 的 leg feature assembly、per-side cue/embedding、`omega_leg` scatter 仍是剩余热点；若现在直接进 Phase E，会把这部分仍可单文件机械整理的代码与跨文件迁移耦合。
- **下一步建议动作**：继续 simplification batch 3，优先 side-routed / non-side leg feature assembly 与 `omega_leg` pre-gate shell；暂不进入 Phase E。

### Implementation Update — 2026-04-24 pre-Phase-E simplification batch 3 leg assembly / cue-embed / omega pre-gate shell

- **本 batch 目标**：落实本 audit 中 batch 2 结尾的下一步，把 `forward(...)` 内剩余的 side-routed / non-side leg feature assembly、side-routed cue / embedding shell、以及 `omega_leg` pre-gate 准备继续压平；不删 helper，不改数值语义，不进入 Phase E。
- **实际完成项**：`train/models.py` 新增 `_DirectPoseSideLegAssembly`、`_prepare_direct_pose_leg_head_input(...)`、`_prepare_direct_pose_side_cues(...)`、`_prepare_direct_pose_side_embeddings(...)`、`_prepare_direct_pose_leg_omega(...)`、`_assemble_direct_pose_side_leg_features(...)`。side-routed 分支的 plan/meas canonicalization、phase view、cue clamp、embedding broadcast、feature concat/flatten 现统一收进 `_assemble_direct_pose_side_leg_features(...)`；non-side 分支改为通过 `_prepare_direct_pose_leg_head_input(...)` 显式走 shared leg-input contract；side/non-side 两条路径进入 gate/scale helper 之前的 `omega_leg` reshape/scatter/max-rad clamp 现统一经过 `_prepare_direct_pose_leg_omega(...)`。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：简化前 `forward(...)` 在 side-routed 分支内同时承载 plan/meas/phase/cue/embedding assembly、leg input flatten、omega scatter/max-rad clamp；non-side 分支单独保留 `direct_flat.detach()` 与 `leg_delta.view(..., 3)` + max-rad clamp。简化后 duplicated 的 assembly/pre-gate 准备分别压到 `_assemble_direct_pose_side_leg_features(...)`、`_prepare_direct_pose_leg_head_input(...)`、`_prepare_direct_pose_leg_omega(...)`；branch body 只保留 head forward、optional sign gate、cross-leg ablation 与 writeback dispatch。
- **是否符合“少而硬”的 helper 原则**：符合。`_prepare_direct_pose_leg_head_input(...)` 和 `_prepare_direct_pose_leg_omega(...)` 是共享 contract helper；`_prepare_direct_pose_side_cues(...)` 与 `_prepare_direct_pose_side_embeddings(...)` 明确收拢 side-routed 的 shape/error context；`_assemble_direct_pose_side_leg_features(...)` 是一处真实的去重型 assembly helper。没有新增只包 1–2 行的单行 wrapper，也没有把 side-routed / non-side 强行统一成大而复杂的 mega-helper。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_event_motion_model_refactor_phase_d` 新增 `test_forward_leg_input_helper_dispatch_regression`、`test_forward_side_leg_cue_embedding_shell_dispatch_regression`、`test_forward_leg_omega_pre_gate_helper_dispatch_regression`；加上 batch 2 已有的 gate/scale dispatch regression，最小覆盖 shared leg-input helper、cue/embedding 入口、omega pre-gate 入口，以及 learned / scale gate output contract。Phase D tests 总数增至 `14`，联合 `tests.train.test_train_models_failfast` 总数增至 `132`。
- **forward snapshot 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`。
- **state_dict fingerprint 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`。
- **stage6 deterministic smoke 验证方式**：使用输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch3_20260424` 运行 5-step deterministic smoke，结果 `ok_steps=5 skipped=0`。
- **运行过的命令**：compile / focused Phase D unit / full unit / snapshot / state_dict fingerprint / AST broad-count / stage6 deterministic smoke / touched code/test ranges removal-policy grep。
- **验证结果**：全部通过；联合 `unittest` 总数更新为 `132`（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=14`），snapshot、fingerprint、stage6 smoke 均证明行为一致。
- **阻塞项 / 风险**：assembly / cue / embedding / omega pre-gate 已收口，但 side-routed rank1/sign-gate 后段与 non-side cross-leg ablation/readout 仍在 `forward(...)` 内保留为 branch-specific 算法块；若现在直接进入 Phase E，会把这些尚可在单文件里继续机械压平的热点与跨文件迁移耦合。
- **为什么此时仍适合继续单文件 simplification**：本轮再次证明 direct-pose leg residual 的剩余热点可以用单文件内机械 helper 收口，并且现有 snapshot / fingerprint / stage6 smoke 足以锁定不变性；因此继续 batch 4 的边际成本仍低于直接跨文件迁移。
- **下一步建议动作**：继续 simplification batch 4，优先看 side-routed sign-gate / rank1 / residual writeback 与 non-side cross-leg ablation/readout 之后还能否再压出一层 dispatch；暂不进入 Phase E。

### Implementation Update — 2026-04-24 pre-Phase-E simplification batch 4 side omega / non-side delta dispatch shell

- **本 batch 目标**：落实本 audit 中 batch 3 结尾的下一步，把 side-routed sign-gate / rank1 / side-omega resolver，以及 non-side cross-leg ablation / head fallback / `rot6d_add` writeback 继续压平成更线性的 dispatch shell；不删 helper，不改数值语义，不进入 Phase E。
- **实际完成项**：`train/models.py` 新增 `_DirectPoseSideLegOmegaOutputs`、`_resolve_direct_pose_side_leg_omegas(...)`、`_resolve_direct_pose_non_side_leg_delta(...)`、`_apply_direct_pose_rot6d_leg_delta(...)`。side-routed 分支的 rank1/per-joint/sign-gate 分叉现在统一收进 `_resolve_direct_pose_side_leg_omegas(...)`；non-side 分支的 cross-leg ablation 与 `direct_pose_leg_head(...)` fallback 统一收进 `_resolve_direct_pose_non_side_leg_delta(...)`；`rot6d_add` 的 additive residual writeback 统一收进 `_apply_direct_pose_rot6d_leg_delta(...)`。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：简化前 `forward(...)` 里 side-routed 分支还内联保留 rank1/per-joint/sign-gate，non-side 分支还内联保留 cross-leg ablation/head fallback 与 `rot6d_add` writeback。简化后这些 branch-specific tail 分别压到 `_resolve_direct_pose_side_leg_omegas(...)`、`_resolve_direct_pose_non_side_leg_delta(...)`、`_apply_direct_pose_rot6d_leg_delta(...)`；`forward(...)` 外层只保留 branch dispatch 与 output writeback。
- **是否符合“少而硬”的 helper 原则**：符合。`_resolve_direct_pose_side_leg_omegas(...)` 和 `_resolve_direct_pose_non_side_leg_delta(...)` 都承载真实分支 dispatch 与 contract 收拢；`_apply_direct_pose_rot6d_leg_delta(...)` 承载 non-side `rot6d_add` writeback contract。没有新增纯包装型单行 helper，也没有把 side/non-side 混成 mega-helper。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_event_motion_model_refactor_phase_d` 新增 `test_forward_side_leg_omega_resolver_dispatch_regression`、`test_forward_non_side_leg_delta_dispatch_regression`、`test_forward_non_side_rot6d_residual_writeback_dispatch_regression`；加上 batch 2/3 既有回归后，最小覆盖 side omega resolver、non-side leg delta resolver、rot6d writeback 壳层。Phase D tests 总数增至 `17`，联合 `tests.train.test_train_models_failfast` 总数增至 `135`。
- **forward snapshot 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`。
- **state_dict fingerprint 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`。
- **stage6 deterministic smoke 验证方式**：使用输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch4_20260424` 运行 5-step deterministic smoke，结果 `ok_steps=5 skipped=0`。
- **运行过的命令**：compile / focused Phase D unit / full unit / snapshot / state_dict fingerprint / AST broad-count / stage6 deterministic smoke / touched code/test ranges removal-policy grep。
- **验证结果**：全部通过；联合 `unittest` 总数更新为 `135`（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=17`），snapshot、fingerprint、stage6 smoke 均证明行为一致。
- **阻塞项 / 风险**：side omega resolver 与 non-side delta dispatch 已收口，但 final direct-pose leg residual result writeback / result assignment 仍在 `forward(...)` 内保留为 branch-specific tail；若现在直接进入 Phase E，会把这些仍可单文件继续机械压平的热点与跨文件迁移耦合。
- **为什么此时仍适合继续单文件 simplification**：本轮再次证明 side/non-side tail dispatch 仍能在单文件内机械化收口，并且 snapshot / fingerprint / stage6 smoke 足够锁定不变性；因此继续 batch 5 的边际风险仍低于直接跨文件迁移。
- **下一步建议动作**：继续 simplification batch 5，优先看 final direct-pose leg residual writeback / result assignment 是否还能再压出一层 dispatch；暂不进入 Phase E。

### Implementation Update — 2026-04-24 pre-Phase-E simplification batch 5 leg residual final writeback / result-assignment shell

- **本 batch 目标**：落实本 audit 中 batch 4 结尾的下一步，把 direct-pose leg residual 的 final writeback / result assignment 再压平成一层共享 dispatch 壳；不删 helper，不改数值语义，不进入 Phase E。
- **实际完成项**：`train/models.py` 新增 `_DirectPoseLegWritebackOutputs` 与 `_dispatch_direct_pose_leg_residual_writeback(...)`，并让 `_write_forward_direct_pose_outputs(...)` 改为消费统一 writeback contract。side-routed 路径在 `_resolve_direct_pose_side_leg_omegas(...)` / gate helper 之后进入这层 shell；non-side 路径在 `_resolve_direct_pose_non_side_leg_delta(...)` 之后按 `leg_mode='so3'|'rot6d_add'` 进入同一层 shell。`direct_leg_omega*` / `direct_leg_gate*` / `direct_leg_scale*` / `direct_leg_side_sign_gate` / `out_direct` 的最终写回不再由两条分支各自手工展开。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：简化前 `forward(...)` 里 side-routed 分支在 omega/gate helper 之后仍要手工回填 `direct_leg_omega` / `direct_leg_gate` / `direct_leg_scale*` locals，non-side 分支则在 `leg_mode` 分叉里各自决定 `so3` / `rot6d_add` final writeback，再把一长串参数传给 `_write_forward_direct_pose_outputs(...)`。简化后两条路径都只负责 resolver / gate / mode dispatch，最终统一收敛到 `_dispatch_direct_pose_leg_residual_writeback(...)`；最终 `result['out_direct']` 与 `result['direct_leg_*']` 的写入统一由 `_DirectPoseLegWritebackOutputs` contract 驱动。
- **这轮具体压平了哪些 duplicated shell / dispatch 壳**：压平了 side-routed / non-side 两条路径在最终输出写回前的 `direct_leg_*` local 回填、writer 参数展开、以及 non-side `rot6d_add` final writeback 与 `so3` output contract 的最后一层分发。当前 remaining hotspot 已从“branch tail writeback”收窄到“direct-pose 主体 readout/orchestration 体量”。
- **是否符合“少而硬”的 helper 原则**：符合。本轮只增加 `1` 个 dataclass + `1` 个 dispatch helper，且二者都承担真实 contract 收拢；没有新增单行 wrapper，也没有把 side-routed / non-side 做成大而复杂的 mega-helper。看起来不像“为了拆分而拆分”。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_event_motion_model_refactor_phase_d` 新增 `test_forward_leg_residual_final_writeback_shell_regression`，以最小 dispatch / branch-shape regression 锁定 side-routed `scale` / `sign-gate` / `rank1`、non-side `learned` / `rot6d_add` 在 resolver 之后都会进入 `_dispatch_direct_pose_leg_residual_writeback(...)`，且 output key contract 保持不变。Phase D tests 总数增至 `18`，联合 `tests.train.test_train_models_failfast` 总数增至 `136`。
- **forward snapshot 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`。
- **state_dict fingerprint 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`。
- **stage6 deterministic smoke 验证方式**：使用输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch5_20260424` 运行 5-step deterministic smoke，结果 `ok_steps=5 skipped=0`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；snapshot 单测命令；state_dict fingerprint 单测命令；AST broad-handler inline Python；stage6 deterministic smoke 命令；touched code/test ranges removal-policy grep。
- **验证结果**：全部通过；联合 `unittest` 总数更新为 `136`（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=18`），snapshot、fingerprint、stage6 smoke 均证明行为一致。
- **阻塞项 / 风险**：final writeback 壳已收口，但 direct-pose 主体 readout / orchestration 仍偏长；若此时直接进入 Phase E，会把仍可在单文件里继续机械压平的 `forward(...)` 局部热点与跨文件迁移耦合。
- **为什么此时仍适合继续单文件 simplification**：本轮再次证明 direct-pose leg residual 尾段 contract 仍能在单文件内机械化收口，并且 snapshot / fingerprint / stage6 smoke 足够锁定不变性；因此继续 batch 6 的风险仍低于立刻进入 Phase E。
- **下一步建议动作**：继续 simplification batch 6，优先看 direct-pose readout / `leg_outputs` 初始化 / final writer 之间是否还能再压出一层不改变数值语义的 orchestration 壳；暂不进入 Phase E。

### Implementation Update — 2026-04-24 pre-Phase-E simplification batch 6 corrective shell + thin-helper retirement

- **本 batch 目标**：纠正 batch 5 的“长换成散”倾向，回退不值得保留的薄 helper / dispatch contract，把 direct-pose leg residual 再组织成两个 branch-sized shell；不改数值语义，不新增 compat / warning-only debt，不进入 Phase E。
- **实际完成项**：`train/models.py` 退休 `_prepare_direct_pose_side_cues(...)`、`_prepare_direct_pose_side_embeddings(...)`、`_prepare_direct_pose_leg_head_input(...)`、`_apply_direct_pose_rot6d_leg_delta(...)`、`_dispatch_direct_pose_leg_residual_writeback(...)`、`_DirectPoseLegWritebackOutputs`；新增 `_forward_side_routed_leg_residual(...)` 与 `_forward_non_side_leg_residual(...)`。side-routed shell 内仍保留 `_assemble_direct_pose_side_leg_features(...) -> _resolve_direct_pose_side_leg_omegas(...) -> _prepare_direct_pose_leg_omega(...) -> _apply_direct_pose_leg_gate_outputs(...)` 这条可闭合的局部阅读路径；non-side shell 内则用 guard clause 展平旧的深层分支，再局部完成 `so3` / `rot6d_add` writeback。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：batch 5 结束时，读同一条 residual path 需要在 `_assemble_direct_pose_side_leg_features(...)`、多个 `_prepare_*` 薄 helper、`_dispatch_direct_pose_leg_residual_writeback(...)` 与 writer 之间多次跳转；branch-local duplication 变少了，但 distributed cognition 增加。batch 6 后，side-routed / non-side 两条路径重新收回各自 shell；散掉的是“跨 1000 行找 control flow”的成本，保留下来的是少量 branch-local result assignment duplication。
- **这轮具体压平了哪些 duplicated shell / dispatch 壳**：压平了 batch 5 新增的 final writeback / result-assignment dispatch 壳、rot6d residual 单行写回壳、leg-input 单行壳，以及 side cue / embedding 的单-call-site 预处理壳；保留 `_assemble_direct_pose_side_leg_features(...)` 是因为它通过了“extract-for-host-readability”门槛，而不是 dedup 门槛。
- **是否符合“少而硬”的 helper 原则**：符合。看起来像“为了拆分而拆分”的 helper 都已退休；保留下来的 helper 要么承担真实去重/contract 收益，要么显著减少 `forward(...)` 外层物理长度。当前没有新增需要特别点名的“可疑 helper”。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_event_motion_model_refactor_phase_d` 删除旧的薄 helper dispatch regression，改为新增 `test_forward_side_routed_leg_residual_shell_dispatch_regression` 与 `test_forward_non_side_leg_residual_shell_dispatch_regression`。Phase D tests 当前为 `16`，联合 `tests.train.test_train_models_failfast` 当前为 `134`；新测试只锁定 shell dispatch 与 branch-shape/output contract，不耦合 helper 内部实现细节。
- **forward snapshot 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`。
- **state_dict fingerprint 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`。
- **stage6 deterministic smoke 验证方式**：使用输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch6_20260424` 运行 5-step deterministic smoke，结果 `ok_steps=5 skipped=0`。
- **运行过的命令**：compile / focused Phase D unit / full unit / snapshot / state_dict fingerprint / AST broad-count / touched code/test ranges removal-policy grep / stage6 deterministic smoke。
- **验证结果**：全部通过；联合 `unittest` 总数更新为 `134`（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=16`），snapshot、fingerprint、stage6 smoke 均证明 corrective pass 前后一致。
- **阻塞项 / 风险**：batch 6 之后，显而易见的“薄壳 retirement + branch shell 压平”收益基本耗尽；再继续单文件 batching 的风险，不是数值漂移，而是重新滑回“为了拆分而拆分”。这一点比 batch 5 前更需要自律。
- **为什么此时仍适合继续单文件 simplification**：原因已从“还有很多 duplicated logic 可收”变成“只剩极少量宿主壳可局部搬运”；也就是说，继续单文件 simplification 仍然可行，但只适合最后一小步、且必须保持 branch-sized shell 粒度。当前已经接近 Phase E 入口，而不是还应连续开很多 batch。
- **下一步建议动作**：优先做 Phase E readiness 评估；如果强制再开一批单文件 simplification，最推荐只看 direct-pose 主 readout/orchestration 的单个宿主壳，完成后就停止，不建议继续扩展 batch 7 及之后的薄 helper 化。

### Implementation Update — 2026-04-24 pre-Phase-E simplification batch 7 event-clock loop shell

- **本 batch 目标**：落实 batch 6 结尾“如果还强制再开一批，只挑一个宿主壳”的约束，把 `forward(...)` 中 event-clock on contact-plan GRU loop body 压平成单个 step shell；不碰 direct-pose leg residual、`__init__`、`_canonicalize_contacts_meas_inputs(...)`、forward 大 stage 切分，也不引入新 helper 网络。
- **实际完成项**：`train/models.py` 在 `forward(...)` 内新增 `_append_contact_plan_direct_step_inputs(...)` 与 `_step_contact_plan_event_clock(...)` 两个 nested def。前者统一 phase/cue per-step shape 校验与错误包装；后者把 event-clock on loop body 收成 `append -> raw/logits/err -> gate -> corrector -> time bias -> debug/logit/prob -> lambda/dyn/delta append` 的单个壳层。event-clock off loop 只复用 append helper，因此没有把“长”换成“散”。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：batch 6 后 remaining hotspot 已从 direct-pose leg residual 转为 event-clock loop body：phase/cue append try/except、gate/corrector、time-bias 壳混在一起，宿主阅读路径被打断。batch 7 后，event-clock on loop 宿主基本只剩 `for _t in range(Tq): _step_contact_plan_event_clock(_t)`；可读性提升来自 host-path 压缩，而不是把同一段逻辑散射到多个跨文件 helper。
- **这轮具体压平了哪些 nested / try 壳**：压平了 `phase_in_direct_seq.append(...)` 前的 per-step shape + try/except 壳、`leg_side_cue_seq.append(...)` 前的 per-step shape + try/except 壳、`contact_plan_time_head(...)` 的 per-step time-bias try/except 壳，以及 loop 内 gate/corrector/readout/debug append 的主阅读路径。没有单独抽出 `_compute_time_bias(...)` 之类的薄包装。
- **是否符合“少而硬”的 helper 原则**：符合。只有 2 个 nested def，且都留在 `forward(...)` closure 内；`_append_contact_plan_direct_step_inputs(...)` 明确收敛 contract，`_step_contact_plan_event_clock(...)` 明确压平宿主路径。看起来像“为了拆分而拆分”的 helper 没有新增。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests/train/test_event_motion_model_refactor_phase_d.py` 新增 `test_event_clock_loop_shell_phase_cue_time_bias_regression`，以最小 branch-shape regression 锁定 event-clock on 路径仍会同时产出 phase/cue/time-bias 相关 contract，并保持 output key set 子集不变。Phase D tests 更新为 `17`，联合 `tests.train.test_train_models_failfast` 更新为 `135`。
- **forward snapshot 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`。
- **state_dict fingerprint 验证方式**：`tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`。
- **stage6 deterministic smoke 验证方式**：使用输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch7_20260424` 运行 5-step deterministic smoke，结果 `ok_steps=5 skipped=0`。
- **运行过的命令**：compile / focused event-clock regression / full unit / snapshot / state_dict fingerprint / AST broad-count / stage6 deterministic smoke / touched ranges removal-policy grep。
- **验证结果**：全部通过；联合 `unittest` 总数更新为 `135`（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=17`），snapshot、fingerprint、stage6 smoke 均证明 batch 7 前后一致，AST 计数保持 broad=`0` / exact=`0` / as_exc=`0`。
- **阻塞项 / 风险**：batch 7 之后，audit 里“还能用一个 branch-sized shell 解决的问题”几乎已经消耗完。剩余可见热点要么太小，不值得抽；要么再抽就会重新制造 distributed cognition，这与本 audit 的 stop rule 冲突。
- **为什么此时仍适合继续单文件 simplification**：适合的前提只对 batch 7 成立——event-clock loop 仍是一个 closure-heavy、错误上下文密集、且能在单文件内机械验证的热点。完成它之后，继续单文件 simplification 的性价比已经明显下降。
- **下一步建议动作**：把 batch 7 视为更合理的停点，优先准备 Phase E readiness；若一定要再挤出一批，也只能审慎评估 contact-plan non-event-clock / finalize handoff 的单个宿主壳，且完成后立即停止，不建议继续扩展 batch 8+。
