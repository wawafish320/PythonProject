> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §1/§7 under its stated read-only / zero-new-injection scope.

# Action Handoff z P6 Runner-Invoke Readiness Note

Date: 2026-05-24
Status: readiness-only (no code implementation in this note)

## 0. Scope / Guardrail

Scope:
- 为 standalone P6 tool 的 `--execution-mode runner_invoke` 做 readiness 契约定义。
- 目标是防止 standalone P6 退化为隐式 orchestration 脚本。

Hard guardrails:
- 本 note 不改代码。
- 不改 `train/training_MPL.py`。
- 不改 `train/posttrain.py`。
- 不改 `train/validate/run_freerun_cycles.py`。
- 当前结论仍是：schema/report/replay plumbing passed，**not P6 passed**。

---

## 1. Runner-Invoke 唯一调用面（owner 决策）

## 1.1 Canonical subprocess target (v1 proposal)

唯一允许调用：
- `python -m train.validate.run_freerun_cycles`

不允许：
- 并行调用多个 runner。
- 在 v1 使用其他注入 wrapper/临时脚本作为隐式入口。

## 1.2 Invocation envelope（由 standalone P6 负责）

standalone P6 只负责：
1. 生成 trial 级 CLI 参数。
2. 启动 subprocess。
3. 检查产物完整性。
4. 将产物回填到当前 P6 report schema。

standalone P6 不负责：
- 改写 runner 行为。
- 在运行时修改训练/模型配置。
- 动态 patch 任何 `train/validate/*` 模块。

---

## 2. P6 Row -> Runner Trial Config 映射契约

## 2.1 输入行（来自现有 row schema）

必须字段：
- `trial_id`
- `source_clip`
- `target_clip`
- `horizon_N`
- `case_type` (`normal|weak_stress`)
- `p6_retrieval_metadata.*`（按现有 schema）
- `p6_fallback.*`（按现有 schema）

## 2.2 生成的 runner config（v1）

新增内部对象（执行期）：

```json
{
  "runner_trial_config": {
    "mode": "runner_invoke",
    "runner_entry": "python -m train.validate.run_freerun_cycles",
    "teacher_json": "validate/teacher_batches/Walk_F_teacher.json",
    "turn_npz": "raw_data/processed_data/<target_clip>.npz",
    "rounds": 4,
    "inject_from_step": 0,
    "inject_at_step": "mapped_from_horizon_bucket",
    "inject_fields": "rootvel,rot6d,angvel",
    "inject_pose_hist_full": true,
    "pose_hist_source": "buffer",
    "log_contacts": true,
    "export_keybone_pos_err": true
  }
}
```

映射规则（v1）：
- `horizon_N in {12,24}` -> bucket 映射：
  - `N=12` -> `inject_at_step=40`
  - `N=24` -> `inject_at_step=80`
- `target_clip` 决定 `turn_npz`。
- `source_clip` 在 v1 不直接映射 runner teacher（保持 `Walk_F_teacher.json`），仅用于 P6 row 语义与报告分组。

Fail-fast:
- `horizon_N` 非 `{12,24}`：直接 exit 2。
- `target_clip` 对应 npz 缺失：exit 2。
- runner config 任意必填键缺失：exit 2。

---

## 3. Runner Output -> P6 Report 回填契约

## 3.1 产物要求

每个 trial 必须产出：
- `<trial_out_dir>/Walk_F_freerun_cycles.json`

可选但推荐：
- `<trial_out_dir>/paired_delta.json`（若后续引入 paired-delta reducer）

## 3.2 回填目标

目标字段：`row.p6_safety_metrics`

v1 从 `Walk_F_freerun_cycles.json` 回填：
- `ContactMismatchRate`
- `FootSlipBallL`
- `FootSlipBallR`
- `RootStepDispErr`
- `GeoLocalDeg`

Canonical metric definition contract（runner_invoke v1）：
- `ContactMismatchRate`：per-step 在 `ContactGTPerC` 与 `ContactMeasPerC` 上按阈值 `>0.5` 二值化后，逐通道不一致比例（`mismatch_channels / valid_channels`）。
- `ContactMismatchFrameOr`：per-step 是否存在任一通道不一致（0/1）。
- `FootSlipBallL/R`：per-step 使用 `ContactMeasWhitebox.VxyCmpsMean` 对应脚通道的速度（cm/s）换算到 m/s（除以 100）；仅当该脚在 `t` 与 `t+1` 两帧 `ContactGTPerC > 0.5` 时记为有效样本，否则记为 `null`（`no_dual_frame_gt_contact`）。

Extractor aggregation contract（row 级）：
- `ContactMismatchRate` 必须优先使用 canonical 字段，不得以 `ContactErrAbsMean` 替代。
- `FootSlipBallL/R` 聚合仅使用有效（非零）样本；若无有效样本则 row 级为 `null`，并视为 canonical metric missing（不得写成 0）。

建议聚合口径（必须固定并写入 report）：
- `window_policy = "post_inject_fixed_window_v1"`
- `agg = "mean"`
- 若无窗口支持则使用明确 fallback（如 full-seq mean），并在 `decision.note` 标记 `window_fallback_used=true`。

类型约束：
- 标量，shape=`[]`，JSON number 或 `null`。
- `null` 仅允许 foot-slip 条件跳过；其他指标 `null` 视为失败。
- 所有 number 必须 finite（非 NaN/Inf）。

---

## 4. Fail-Fast / Timeout / Partial-Output Policy

## 4.1 Fail-fast（exit code=2）

- subprocess 非零退出码。
- 期望输出 JSON 缺失。
- 输出 JSON schema 缺关键字段。
- required metric 非 finite / 非法 null。
- 执行行数超出 v1 smoke 限制。

## 4.2 Timeout policy

- 每 trial 设置硬超时（建议 `timeout_s=900`）。
- 超时立即 kill 子进程并标记 trial failed。
- 不做自动重试（v1）。

## 4.3 Partial-output policy

- 允许单行失败后继续执行剩余行（便于比较 normal vs weak）。
- 失败行写：
  - `p6_safety_metrics.status="runner_invoke_failed"`
  - `decision.status="runner_invoke_failed"`
  - `decision.note` 写明 `error_type/error_message/artifact_path`
- summary 必须包含：
  - `runner_invoke_total_rows`
  - `runner_invoke_success_rows`
  - `runner_invoke_failed_rows`

---

## 5. v1 执行范围锁定（必须）

只允许 `1 normal + 1 weak_stress`：
- normal: `Walk_F -> Walk_R_To_L`, `N=12`
- weak_stress: `Walk_L_To_R -> Walk_R_To_R`, `N=24`

禁止：
- v1 直接跑 full matrix。
- v1 并行多 trial。
- v1 加入自动参数搜索/自适应重试。

---

## 6. Readiness Gate（进入实现前必须全满足）

1. 调用面唯一化（仅 `python -m train.validate.run_freerun_cycles`）。
2. row->runner config 映射表冻结。
3. output->report 回填表冻结。
4. fail-fast/timeout/partial-output 策略冻结。
5. smoke-only 执行范围冻结。

未满足上述任一项：不进入 `--execution-mode runner_invoke` 实现。
