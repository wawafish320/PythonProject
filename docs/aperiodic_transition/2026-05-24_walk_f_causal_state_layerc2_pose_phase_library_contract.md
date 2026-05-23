# Walk_F Causal-State Scaffold v1 Layer C.2 Pose Phase-Library Check Contract (2026-05-24)

> 本 memo 是 **CONTRACT + IMPL scope** for Layer C.2 `pose_phase_library_check`。  
> Layer C.2 严格 read-only：不训练、不产出 EventHead target、不定义
> `handoff_ready`/`transition_done`、不产出 query leave/return、不产出
> attractor membership、不产出 combined membership score、不写
> checkpoint/fingerprint/train config/raw_data。

## §0 Relation to Existing Scaffold / Layers

Layer C.2 是以下契约的 pose-only 补层，不替代它们：

- scaffold v1: `docs/aperiodic_transition/2026-05-22_walk_f_causal_state_scaffold_v1.md`
- Layer C minimal: `docs/aperiodic_transition/2026-05-23_walk_f_causal_state_layerc_minimal_contract.md`
- Layer B.1 pose scale/degeneracy:
  `docs/aperiodic_transition/2026-05-23_walk_f_causal_state_layerb1_pose_reference_scale_contract.md`

Layer C minimal 已覆盖 `root_body`/`contact` 的 Walk_F self-consistency。  
Layer B.1 已覆盖 `pose_dyn`/`pose_rel` 的 processed-data schema + Walk_F
reference scale/degeneracy。  
Layer C.2 在此基础上，仅补 `pose_dyn`/`pose_rel` 的 Layer C 风格 phase-library
self-consistency 检查，不扩展 membership/boundary/training 语义。

## §1 Hard-Locked Rules

以下 3 条为二进制硬约束；任何违反必须 `FailFastError`，禁止 silent fallback：

1. **Walk_F-only pose phase-library self-consistency。**  
   Layer C.2 只评估 Walk_F 自一致性，不做 query boundary / membership /
   training labels。
2. **单条 88-frame Walk_F 的 contract 语义不变。**  
   `phase_structure_status=insufficient_evidence` 在 C.2 是 contract PASS；
   C.2 允许汇报 `self_consistency_signal_status` 作为候选证据，但不得提升为
   `phase_structured`。
3. **`turn_dyn` 永不进入 combined membership / combined score。**  
   任意路径尝试路由 `turn_dyn` 至 combined-membership 语义必须 fail-fast；
   C.2 需新增二进制 guard（`_layer_c2_forbid_turn_dyn`），并保持与 C.1/B.1
   同级措辞强度。

## §2 Mode / Clip Policy / Fail-Fast Ordering

Tool mode:

```text
pose_phase_library_check
```

Allowed `--clips`（精确匹配，顺序敏感，与 Layer C minimal 一致）：

```text
["Walk_F"]
```

Fail-fast 顺序必须在 `args.out_dir.mkdir(...)` 之前完成：

- mode 不支持；
- clip 列表非法（非 `["Walk_F"]`）；
- duplicate clips；
- raw JSON 缺失或 `_load_clip` 失败；
- processed-data NPZ 缺失、schema mismatch、non-finite；
- pose group active channel 数为 0；
- 任意 `turn_dyn` 路由到 C.2 combined-membership 语义路径。

## §3 Data Source / Shapes / Dtype / Device

Layer C.2 使用 Layer B.1 owner 逻辑从
`raw_data/processed_data/Walk_F.npz` 提取：

- `pose_dyn`:
  - source: `bone_ang_vel` (`(T,46,3)` on disk `float32`)
  - extracted matrix: **`(T,138)`**, `float64`, `cpu`
- `pose_rel`:
  - source: `bone_rot6d` first-difference `* FPS`
  - extracted matrix: **`(T,276)`**, `float64`, `cpu`

Active set 规则（复用 B.1 阈值语义）：

- 按 Walk_F channel MAD，`mad <= 1.0e-4` 视作排除；
- 输出 `excluded_channels_by_walk_f_mad` + `active_channels`；
- 当前已知 artifact 期望值：`pose_dyn` 66/138 active，`pose_rel`
  128/276 active（实现不得 hardcode，必须按 MAD 计算）。

`pose_rel` caveat（必须在 artifact 与 memo 中显式重复）：  
`pose_rel` 是 rot6d 分量的一阶差分速度，不是 SO(3) log/geodesic
angular velocity。

## §4 Estimator Grid (Locked Reuse)

Layer C.2 必须逐字复用 Layer C minimal 的 2x2x2x2 网格（16 configs）：

- `history_window_frames`: `[6, 12]`
- `future_horizon_frames`: `[6, 12]`
- `neighborhood_radius_frames`: `[4, 8]`
- `distance_metric`: `["z_mse", "z_l1"]`

不得新增/删减网格维度、不得改默认常量、不得改变
`phase_library_check` 既有行为。

## §5 Output Contract

每个 group（`pose_dyn`, `pose_rel`）必须输出：

- `channel_total_count`
- `active_channel_count`
- `active_channels`
- `excluded_channels_by_walk_f_mad`
- `estimator_grid`（16 configs 明细）
- `configs_beating_baseline_count`
- `self_consistency_signal_status`
- `phase_structure_status = "insufficient_evidence"`
- `evidence_status = "INSUFFICIENT_EVIDENCE"`

summary root 必须显式声明下列字段为 not emitted / forbidden：

- `attractor_membership_status`
- `event_head_target_status`
- `handoff_ready_status`
- `transition_done_status`
- `query_leave_return_status`
- `cross_attractor_claim_status`
- `combined_membership_score_status`

并显式包含：

- `expected_insufficient_evidence_is_contract_pass = true`
- `layer_c2_contract_status = "pass"`（在 schema/finite/guard 条件满足时）

## §6 Expected Current-Data Result

当前数据下的预期结果：

- `phase_structure_status=insufficient_evidence` 是 **PASS**；
- C.2 允许报告 16-config 的 `self_consistency_signal_status`，
  并与既有 root/contact C-min artifact 做并列比较；
- 但不得据此提升为 `phase_structured`，不得产生 membership/boundary/
  transition/training 语义。

## §7 Non-Goals (Explicit)

Layer C.2 不输出、不定义、不暗示：

- query leave/return/censoring；
- attractor membership；
- transition truth；
- EventHead target；
- `handoff_ready`；
- `transition_done`；
- cross-attractor claims；
- checkpoints / fingerprints / train config / raw_data writes。
