> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Action Handoff z P6 Injection Wiring Decision Note

Date: 2026-05-25
Status: Decision note (Phase A only)

## 1. Current blocker

- `runner_invoke` subprocess smoke 已通过：`p6_synthetic_boundary_eval_summary.json` 显示 `execution_status="executed_runner_smoke_v1"`，且 row 的 `execution_binding.run_result.ok=true`（`debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260524_runner_real_smoke_normal1_v4/p6_synthetic_boundary_eval_summary.json`）。
- metric extraction/backfill plumbing 已通过：同一 summary 中 `p6_safety_metrics` 已回填 `ContactMismatchRate/RootStepDispErr/GeoLocalDeg`，并记录 `metric_source_used`。
- 但当前 `run_freerun_cycles` CLI 没有真实 `--inject_*` 参数：`parse_args()` 无 `--inject-turn-npz/--inject-at-step/--inject-from-step/--inject-fields`（`train/validate/run_freerun_cycles.py:10049`）。
- 上一轮 `target_clip/horizon` 仅是 report-level metadata，不是 injected execution：
  - P6 tool 明确写了 `inject_args_supported_by_runner_cli=false` 与 `runner_smoke_proxy_note`（`tools/run_action_handoff_p6_synthetic_boundary_eval.py:621`）。
  - command construction 注释明确“current CLI has no --inject_* args”（`tools/run_action_handoff_p6_synthetic_boundary_eval.py:605`）。

结论：当前已验证的是 runner 调用链和 metrics 回填链，不是 P6 synthetic-boundary 实注入执行。

## 2. Candidate decisions

### A. 给 `train/validate/run_freerun_cycles.py` 加最小 injection CLI

目标：仅新增 CLI + metadata plumbing，把 `--inject-turn-npz/--inject-at-step/--inject-from-step/--inject-fields` 接到“已存在的注入路径”。

### B. 新建 wrapper 复用现有 substrate artifact replay，不做新 execution

目标：维持现有 artifact replay 方式，把报告做完整，但不新增真实 runner 注入执行。

### C. 继续只做 artifact replay，延后真实 execution

目标：冻结现状，明确 P6 仍未进入 injected execution，等待后续专门的 runner 注入能力设计与实现。

## 3. 方案评估

### A 评估

- 是否能真实注入 `target rootvel/rot6d/angvel`：**当前不能保证**。
  证据：`_run_freerun_cycles(...)` 参数集中没有任何注入输入（`turn_npz/inject_at_step/inject_fields`）可复用入口（`train/validate/run_freerun_cycles.py:2234`）。
- 是否保持训练入口不污染：可以（仅改 `train/validate/run_freerun_cycles.py` + tools，不改 `training_MPL.py`/`posttrain.py`）。
- 是否最小改动：**当前不满足**。
  因为不是“CLI 未暴露但内部路径已存在”，而是“内部执行路径缺失”；若硬做 A，需新增 mid-rollout state splice 逻辑，已经不是 minimal plumbing。
- 是否能回填现有 safety metrics：理论可行（runner JSON 已有 `metrics_per_step` 回填路径），但前提是注入真实生效。
- 风险：高。
  在 `run_freerun_cycles` 主循环中新增注入行为会触及 sample/state carry/pose_history/contact plan 的时序一致性，容易越过“不得重构 evaluator 主逻辑”边界。

### B 评估

- 是否能真实注入 `target rootvel/rot6d/angvel`：不能（只 replay 既有 artifact）。
- 是否保持训练入口不污染：可以。
- 是否最小改动：可以（几乎零改动）。
- 是否能回填现有 safety metrics：可以（已有 replay 回填）。
- 风险：中。
  风险不是代码稳定性，而是语义误判：容易被误解为“执行过 injected P6”。

### C 评估

- 是否能真实注入 `target rootvel/rot6d/angvel`：不能（明确不做）。
- 是否保持训练入口不污染：可以。
- 是否最小改动：最好（零或文档级改动）。
- 是否能回填现有 safety metrics：可以，但仅限 replay 或 runner non-injected smoke。
- 风险：低到中。
  主要风险是进度延后；但能避免把 proxy metadata 误写成真实 synthetic-boundary execution。

## 4. Recommendation

推荐 **C（停止在 artifact replay / non-injected runner smoke，暂不进入真实 execution）**。

原因：

1. 当前代码中未发现“可复用的注入执行路径仅缺 CLI 暴露”。
   `run_freerun_cycles` 既无 `--inject_*` CLI，也无对应执行期注入参数接点（`train/validate/run_freerun_cycles.py:10049`, `train/validate/run_freerun_cycles.py:2234`）。
2. 现有 P6 tool 已明确标注当前 runner 仅为 proxy（metadata 级映射），非真实注入（`tools/run_action_handoff_p6_synthetic_boundary_eval.py:605`, `tools/run_action_handoff_p6_synthetic_boundary_eval.py:621`）。
3. 若强行推进 A，将不可避免新增 evaluator 主逻辑，不符合“只允许最小 injection CLI / metadata plumbing；不得重构 evaluator 主逻辑”的硬约束。

## 5. Decision boundary

- 只有在 **推荐 A 且 owner module 清楚、并且已确认存在可复用 injection path（仅 CLI 未暴露）** 时，才进入 Phase B。
- 当前结论为 **推荐 C**，因此 **不进入 Phase B**。
  阻塞原因：`run_freerun_cycles` 缺失可直接接线的内部注入执行路径；现阶段无法在“不重构 evaluator 主逻辑”的前提下完成真实 injected execution。
