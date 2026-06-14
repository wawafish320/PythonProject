> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §1/§7 under its stated read-only / zero-new-injection scope.

# Action Handoff z P6 Minimal Execution Wiring Design Note

Date: 2026-05-24
Status: design-only (no code change in this note)

## 0. Scope / Non-goals

Scope:
- 定义 standalone P6 tool 从 dry-run scaffold 进入 **minimal execution wiring** 的最小可评审契约。
- 限定 v1 execution 为 smoke 级别：`1 normal + 1 weak_stress`。

Non-goals:
- 不改 `train/training_MPL.py`。
- 不改 `train/posttrain.py`。
- 不改 `train/validate/run_freerun_cycles.py`。
- 不在本 note 里实现 runner 接线或启动新训练。

---

## 1. Standalone Tool 将调用的现有 substrate/runner

### 1.1 v1 canonical execution backend（先锁定）

v1 不直接发起新的 rollout 子进程；只做 **artifact replay**：

- substrate 根目录：`debug_output/_tmp_turn_a_to_b_entry_probe_20260515`
- 必需输入：
  - `sweep_config.json`
  - `contract_check_report.json`
  - `p2_entry_probe_check_report.json`
  - `trials/*/paired_delta.json`
  - `trials/*/Walk_F_freerun_cycles.json`

理由：
- 该 substrate 已有可复用的 trial 级安全指标统计（entry/post-recovery）。
- 当前 HEAD 里不引入新的 runner/inject CLI 变更，避免 boundary 污染。

### 1.2 runner-invoke 路径（保留到 v2）

v2 才允许从 standalone tool 直接调用 rollout runner（例如 `python -m train.validate.run_freerun_cycles`）生成新 trial artifact。
v1 不开放该路径；若请求 runner-invoke，直接 fail-fast。

---

## 2. 从 dry-run row 到 real trial config 的映射

## 2.1 新增 execution binding（设计字段）

为每条可执行 row 增加 `execution_binding`（仅 execution 模式需要）：

```json
{
  "execution_binding": {
    "mode": "artifact_replay",
    "substrate_trial_id": "trial_003_Walk_R_To_L_M0_N40",
    "trial_dir": "debug_output/_tmp_turn_a_to_b_entry_probe_20260515/trials/trial_003_Walk_R_To_L_M0_N40",
    "paired_delta_json": ".../paired_delta.json",
    "freerun_json": ".../Walk_F_freerun_cycles.json",
    "metric_window": "entry_window"
  }
}
```

`mode` 在 v1 只能是 `artifact_replay`。

## 2.2 row -> binding 映射规则（v1）

1. `trial_id/source_clip/target_clip/horizon_N/case_type` 继续沿用 scaffold row。
2. `horizon_N` 在 v1 仅作 bucket 标签（short/long），不强制等于 substrate 的 `inject_at_step`。
3. `source_clip/target_clip` 与 substrate trial 的语义在 v1 允许“proxy 绑定”，但必须显式写在 `execution_binding`，不能隐式推断。
4. 每条执行 row 必须指向一个唯一 `trial_dir`，不允许一条 row 绑定多个 trial。

## 2.3 v1 smoke 固定样本

- normal smoke（proxy）：
  - row: `Walk_F -> Walk_R_To_L`, `N=12`, `case_type=normal`
  - binding: `trial_003_Walk_R_To_L_M0_N40`
- weak_stress smoke（proxy）：
  - row: `Walk_L_To_R -> Walk_R_To_R`, `N=24`, `case_type=weak_stress`
  - binding: `trial_002_Walk_L_To_R_M0_N80`

说明：这是 plumbing smoke，不是 full semantic equivalence 验证。

---

## 3. 真实 metrics 回填 `p6_safety_metrics`

## 3.1 数据源

优先源：`paired_delta.json` 的 `<metric_window>.metric_summary`。
回退源：`Walk_F_freerun_cycles.json` 的 `metrics_per_step`（仅当 paired_delta 缺失时）。

## 3.2 指标回填契约（row 级）

回填字段：
- `ContactMismatchRate`
- `FootSlipBallL`
- `FootSlipBallR`
- `RootStepDispErr`
- `GeoLocalDeg`

取值规则（v1）：
- 默认取 `metric_summary[metric].mean`（`float`）。
- 若 `n==0` 且指标为 `FootSlipBallL/R`，可设 `null`，并记录 `skip_reason`（例如 `no_dual_frame_gt_contact`）。
- 其他指标若 `n<=0` 或 `mean` 非 finite，直接 fail-fast。

类型/shape/dtype/device 约束：
- 所有 `p6_safety_metrics.*` 都是标量：shape=`[]`，dtype=`float64`（JSON number），device=`cpu`（artifact replay）。
- `null` 仅允许在 foot-slip 条件跳过场景。

回填后状态：
- `p6_safety_metrics.status = "executed_artifact_replay_v1"`
- `decision.status = "executed_smoke_not_pass_gate"`

---

## 4. `--allow-execute` 最小安全开关与 fail-fast

execution 入口必须同时满足：
- `--allow-execute=true`
- `--dry-run=false`
- `--execution-mode=smoke_v1`（新增枚举，默认 `dry_run_only`）
- `--trial-matrix` 显式提供，且仅 2 rows

v1 fail-fast 条件（exit code=2）：
1. row 数量不是 2。
2. row 组成不是 `1 normal + 1 weak_stress`。
3. 缺少 `execution_binding` 或 `mode != artifact_replay`。
4. 任一绑定文件不存在或 JSON schema 缺键。
5. required safety metric 非 finite / 非法 null。
6. 请求 `runner_invoke`。

执行后强制写入：
- `execution_status = "executed_smoke_v1_artifact_replay"`
- `known_risks` 增加 `"semantic_proxy_binding_used_in_smoke_v1"`

---

## 5. v1 执行面：只跑 1 normal + 1 weak_stress smoke

固定策略：
- 不跑全矩阵。
- 不自动展开 horizons。
- 不做 pass/fail 结论，只验证：
  - row->binding 映射可执行；
  - safety metrics 回填可用；
  - fail-fast 与执行状态写盘正确。

建议 smoke 输出新增字段：
- `smoke_scope = "2_rows_only"`
- `smoke_rows = [trial_id_normal, trial_id_weak]`
- `smoke_limit_enforced = true`

---

## 6. 建议实施顺序（仍是设计）

1. 在 standalone P6 tool 里先实现 `execution_binding` schema 校验与 fail-fast（不调用 runner）。
2. 加入 artifact-replay executor（只读 JSON，不改 train 代码）。
3. 加 `smoke_v1` 门控（2 rows only）。
4. 评审 smoke artifact 后，再决定是否进入 v2 runner-invoke 设计。

当前结论保持：
- standalone P6 scaffold dry-run passed
- no evaluator called（截至本 note）
- P6 未执行通过（not passed）
