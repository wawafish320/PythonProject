# EventMotionModel component slot inventory

Date: 2026-04-25  
Status: Draft / pre-`compute_module_graph_hash()` paper artifact  
Scope: `train/models.py:560` `EventMotionModel` semantic slots  
Goal: 在真正落 `compute_module_graph_hash()` 之前，先冻结“哪些东西算语义 slot、哪些东西只是实现细节 / runtime 噪声”。

---

## 0. 约定

### 0.1 这份 inventory 记录的是什么

这里的 `component_slot` 指 **语义 slot identity**，不是 import path，也不是某个 `.py` 文件里的类布局。

它的用途是给后续 `module_graph_hash` 提供稳定输入：

- Phase E 跨文件迁移时，slot identity 不应变化
- 只有 slot 的 **语义角色 / consumes / produces / normalized config** 变化时，才应触发 hash 变化

### 0.2 粒度约定

这份 inventory **不**按“每个 leaf `nn.Linear` 一项”建模。  
如果一组 attr 共同构成一个稳定的语义块，则用一个 `component_slot` 表示，并在 `backing_attrs` 中列出现有实现。

例子：

- `pasa_attention_block` 是一个语义 slot
- 当前 backing attrs 是 `_pasa_q / _pasa_k / _pasa_v / _pasa_o / _pasa_lnq / _pasa_film / coupling_norm`

### 0.3 `component_kind` 词表（第一版）

- `encoder_trunk`
- `residual_bypass_proj`
- `attention_coupler`
- `motion_readout`
- `phase_projection`
- `residual_adapter_bank`
- `recurrent_plan_core`
- `state_seed`
- `obs_seed_adapter`
- `contact_logit_head`
- `time_bias_head`
- `periodicity_gate`
- `plan_state_corrector`
- `pose_trunk`
- `pose_terminal`
- `pose_branch_readout`
- `feature_bottleneck`
- `leg_residual_head`
- `leg_gate_head`
- `side_routed_leg_head`
- `side_routed_leg_gate_head`
- `side_embedding`
- `side_sign_gate`
- `fusion_gate_head`
- `so3_delta_head`
- `scalar_gate_parameter`
- `adaptive_history_encoder`
- `external_frozen_encoder`
- `external_frozen_period_head`
- `external_frozen_contact_head`

### 0.4 Hash-input conventions

以下规则决定下文表格里哪些列真正进入 `module_graph_hash`：

- **Slot presence 永远入 hash**：slot 的 `enabled / absent` 状态是 graph fact，无论 `hash_scope` 怎么写，slot 的 enable 状态变化必然触发 hash 变化。`hash_scope` 只约束"当 enabled 时 config detail 是否入 hash"。
- **`backing_attrs` 只做 inventory 文档**，不是 hash 输入。attr 改名（如 `_pasa_q` → `_pasa_query`）只要不改语义，hash 不变。hash 输入只来自 `component_slot` / `component_kind` / normalized `consumes` / normalized `produces` / `normalized_config`。
- **`consumes` / `produces` 默认按 sorted set hash**；若顺序是语义的一部分（例如 feature concat 次序），须在该 slot 的 `notes` 里显式标 `order-sensitive`，并按 ordered tuple hash。

---

## 1. Core backbone slots

| component_slot | component_kind | backing_attrs | enabled_when | consumes | produces | hash_scope | notes |
|---|---|---|---|---|---|---|---|
| `shared_encoder` | `encoder_trunk` | `shared_encoder` | always | `state`, `cond`, optional contact-plan injection | `h` / pre-temporal hidden seed | required | 主 encoder trunk；depth / residual-vs-plain 属于 normalized config |
| `residual_proj` | `residual_bypass_proj` | `residual_proj` | always | same encoder input as `shared_encoder` | residual addend for `h_temporal` | required | 决定 PASA 前 residual coupling 路径 |
| `pasa_attention_block` | `attention_coupler` | `_pasa_q`, `_pasa_k`, `_pasa_v`, `_pasa_o`, `_pasa_lnq`, `_pasa_film`, `coupling_norm` | always | `h_temporal`, `cond` | `attn`, `h_final` | required | 这是一个复合 slot；不要拆成 import/layout-sensitive leaf hash |
| `motion_head` | `motion_readout` | `motion_head` | always | `h_final` | baseline motion output `out` / `delta` | required | 不含 direct-pose / λ-fusion / so3-corr 分支 |
| `period_encoder` | `phase_projection` | `period_encoder` | `period_dim > 0` | `soft_period` / period feature | `period_emb` | required when present | 用于 contact-plan / event-clock 路径的 period feature 投影 |
| `bone_residual_adapter_bank` | `residual_adapter_bank` | `_bone_adapters` + `_bone_adapter_slices/_names` metadata | target bones resolve successfully | `h_final`, selected output slices | per-bone additive residuals | required when present | slot identity 是 adapter bank，不是某个单独 adapter 的当前位置 |

---

## 2. Contact-plan / event-clock slots

| component_slot | component_kind | backing_attrs | enabled_when | consumes | produces | hash_scope | notes |
|---|---|---|---|---|---|---|---|
| `contact_plan_cell` | `recurrent_plan_core` | `contact_plan_cell` | `contact_plan_enable` | per-step `cond`, previous `plan_z` | next `plan_z_raw` | required when present | 这是 contact-plan 的 recurrent core |
| `contact_plan_init_z` | `state_seed` | `contact_plan_init_z` | `contact_plan_enable` | none | learnable initial `plan_z` | required when present | 参数 slot，不是 module slot；仍属于 semantic graph |
| `contact_plan_init_head` | `obs_seed_adapter` | `contact_plan_init_head` | `contact_plan_init_mode in {'obs','learnable+obs'}` and `obs_dim > 0` | `contacts`, `angvel`, `pose_history` | observation-conditioned init delta | required when present | cold-start disambiguation 入口 |
| `contact_plan_head` | `contact_logit_head` | `contact_plan_head` | `contact_plan_enable` | `plan_z` | `contacts_plan_logits` / `contacts_plan` | required when present | 主 contact readout |
| `contact_plan_time_head` | `time_bias_head` | `contact_plan_time_head` | `contact_plan_time_pe_dim > 0` | time positional encoding | additive time bias on contact logits | required when present | time bias 是语义性输入，不应丢到 runtime 噪声里 |
| `event_clock_gate` | `periodicity_gate` | `event_clock_gate` | `use_event_clock` | contact delta obs, LR diff, optional period feat | `lambda_corr`, `lambda_logit`, `dynamic_prior` | required when present | event-clock 的 gating / prior 语义入口 |
| `event_clock_corrector` | `plan_state_corrector` | `event_clock_corrector` | `use_event_clock` | `plan_z_raw`, delta obs, optional period feat, gate outputs | corrected `plan_z`, `delta_z` | required when present | 负责对 contact-plan latent 做 residual correction |

---

## 3. Direct-pose family slots

| component_slot | component_kind | backing_attrs | enabled_when | consumes | produces | hash_scope | notes |
|---|---|---|---|---|---|---|---|
| `direct_pose_head` | `pose_trunk` | `direct_pose_head` | `contact_plan_enable and direct_pose_enable` | selected direct features: `cond/hidden`, `contacts_plan`, optional `contacts_meas`, optional `phase_z`, optional time PE | `out_direct` trunk or mode-select logits | required when present | 这是 direct-pose 主 trunk；`feat_source` / `meas_mode` / `phase_z_mode` / `direct_pose_leg_side_routing` / `direct_pose_leg_mode` 都属于 normalized config，确保 side-routed vs non-side 的 graph 选择对 hash 可见；`consumes` 为 `order-sensitive`（feature concat 次序是语义的一部分） |
| `direct_pose_leg_terminal` | `pose_terminal` | `direct_pose_leg_terminal` | `direct_pose_split_enable` | shared direct trunk feature | leg output slice | required when present | split-head leg terminal |
| `direct_pose_out_nonleg` | `pose_branch_readout` | `direct_pose_out_nonleg` | `direct_pose_split_enable and not direct_pose_arm_split_enable` | shared direct trunk feature or projected feature | non-leg output slice | required when present | 与 arm/else 分支互斥 |
| `direct_pose_nonleg_proj` | `feature_bottleneck` | `direct_pose_nonleg_proj` | `direct_pose_split_enable and direct_pose_nonleg_proj_dim > 0 and not direct_pose_arm_split_enable` | direct trunk feature | projected non-leg feature | required when present | projection presence 影响 non-leg branch graph |
| `direct_pose_out_arm` | `pose_branch_readout` | `direct_pose_out_arm` | `direct_pose_arm_split_enable` | shared direct trunk feature or projected arm feature | arm output slice | required when present | three-way split 的 arm branch |
| `direct_pose_out_else` | `pose_branch_readout` | `direct_pose_out_else` | `direct_pose_arm_split_enable` | shared direct trunk feature or projected else feature | else output slice | required when present | three-way split 的 else branch |
| `direct_pose_arm_proj` | `feature_bottleneck` | `direct_pose_arm_proj` | `direct_pose_arm_split_enable and direct_pose_nonleg_proj_dim > 0` | direct trunk feature | projected arm feature | required when present | arm branch bottleneck |
| `direct_pose_else_proj` | `feature_bottleneck` | `direct_pose_else_proj` | `direct_pose_arm_split_enable and direct_pose_nonleg_proj_dim > 0` | direct trunk feature | projected else feature | required when present | else branch bottleneck |
| `direct_pose_leg_head` | `leg_residual_head` | `direct_pose_leg_head` | `direct_pose_leg_enable` and resolved leg joints exist | direct feature stream | per-leg residual (`rot6d_add` or `so3`) | required when present | 即使 side-routing 打开，当前实现中它也仍可能被实例化。active vs dead-weight 由 `direct_pose_leg_side_routing` 决定——该 flag 已进入 `direct_pose_head` 的 normalized_config，故本 slot 的 presence 与 active 状态对 hash 都是可见的 |
| `direct_pose_leg_gate_head` | `leg_gate_head` | `direct_pose_leg_gate_head` | `direct_pose_leg_enable and direct_pose_leg_gate_mode in {'learned','scale'}` | direct feature stream | per-leg gate / scale | required when present | 与 `direct_pose_leg_head` 配对 |
| `direct_pose_leg_head_shared` | `side_routed_leg_head` | `direct_pose_leg_head_shared` | `direct_pose_leg_side_routing` and `direct_pose_leg_side_k > 0` | side-routed per-leg feature stream | per-side shared residual output | required when present | 同侧 joint coupling 的 shared head |
| `direct_pose_leg_gate_head_shared` | `side_routed_leg_gate_head` | `direct_pose_leg_gate_head_shared` | `direct_pose_leg_side_routing` and gate mode is `learned/scale` | side-routed per-leg feature stream | per-side shared gate / scale | required when present | side-routed gate path |
| `direct_pose_leg_side_embed` | `side_embedding` | `direct_pose_leg_side_embed` | `direct_pose_leg_side_routing and direct_pose_leg_side_embed_dim > 0` | side id (`L/R`) | side embedding feature | required when present | side identity 进入 shared leg head 的显式 cue |
| `direct_pose_leg_side_sign_gate_head` | `side_sign_gate` | `direct_pose_leg_side_sign_gate_head` | `direct_pose_leg_side_routing and direct_pose_leg_side_sign_gate` | side-routed feature stream | side-shared sign gate | required when present | 目标是抓 same-side co-flip failures |

---

## 4. Posttrain / auxiliary slots

| component_slot | component_kind | backing_attrs | enabled_when | consumes | produces | hash_scope | notes |
|---|---|---|---|---|---|---|---|
| `lambda_fusion_head` | `fusion_gate_head` | `lambda_fusion_head` | `lambda_fusion_enable` | `h_final`, optional `contacts_plan`, optional rollout step | λ logits / fusion weights | required when present | Stage2/posttrain 关键语义分支 |
| `so3_delta_corrector` | `so3_delta_head` | `so3_delta_corrector` | resolved rot-joint count > 0 | `h_final`, optional `contacts_plan` | `omega_hat` | required when present | post-train friendly SO(3) delta branch |
| `so3_corr_gate_logit` | `scalar_gate_parameter` | `so3_corr_gate_logit` | resolved rot-joint count > 0 | none | scalar gate on so3 corrector | required when present | 参数 slot，和 `so3_delta_corrector` 成对出现 |

---

## 5. Runtime-attached / external-bundle slots

这些 slot 不是 base constructor 永久自带，但它们一旦 attach，就会改变真实 forward / rollout 语义。  
因此它们应在 inventory 中显式出现；后续 hash 时可以标记为“present-sensitive optional slot”，而不是把它们当噪声忽略掉。

| component_slot | component_kind | backing_attrs | enabled_when | consumes | produces | hash_scope | notes |
|---|---|---|---|---|---|---|---|
| `adaptive_history_module` | `adaptive_history_encoder` | `adaptive_history_module` | `attach_adaptive_history_runtime(...)` 成功 attach | raw `pose_history` | transformed / windowed pose-history feature | present-sensitive | 这是 runtime attach 到 model 上的语义模块，不是单纯 trainer attr |
| `frozen_encoder` | `external_frozen_encoder` | `frozen_encoder` | external encoder bundle attached | encoder input assembled from rollout state | hidden summary for contact / period inference | present-sensitive | 来源于 bundle，不应 hash file path，只 hash “存在 + semantic kind + normalized dims” |
| `frozen_period_head` | `external_frozen_period_head` | `frozen_period_head` | bundle contains period head | frozen encoder hidden summary | `soft_period` | present-sensitive | 与 `period_encoder` 是不同 slot：一个是 external predictor，一个是 in-model projection |
| `frozen_contact_head` | `external_frozen_contact_head` | `frozen_contact_head` | bundle contains contact head | frozen encoder hidden summary | external contact hint | present-sensitive | contact-plan rollout contract 的外部依赖 |

---

## 6. Excluded / non-slot items

下列内容**不**作为 component slot 单独建模，但仍可能进入 normalized config / metadata：

- `direct_pose_leg_out_idx`
- `direct_pose_nonleg_out_idx`
- `direct_pose_arm_out_idx`
- `direct_pose_else_out_idx`
- `direct_pose_leg_joint_idx_tensor`
- `direct_pose_leg_side_pos_r_tensor`
- `direct_pose_leg_side_pos_l_tensor`
- `_bone_adapter_slices`
- `_bone_adapter_names`

原因：

- 它们是 **routing metadata / index metadata**
- 会影响 wiring，但不属于“独立可消费信号的 component”
- 更适合进入对应 slot 的 `normalized_config`

另外：

- `contact_plan_phase_head` 当前只被声明为 placeholder，没有在现行 build path 中实例化  
- `contact_plan_input_proj` 当前不是 `EventMotionModel` 的 live attr，不应写成 model-owned slot

---

## 7. 对后续 `compute_module_graph_hash()` 的直接约束

后续实现时应遵守：

1. `component_slot` 取上表中的稳定 slot 名，不取 import path  
2. runtime-attached external slots 不得按 bundle path / checkpoint path hash  
3. routing/index metadata 进入对应 slot 的 normalized config，不单独当 component  
4. `pasa_attention_block` 这类复合 block 按语义块 hash，不按 leaf `Linear` 顺序逐层散列  
5. `present-sensitive` slot 的”是否存在”本身就是 graph 事实，应参与 hash  
6. `consumes` / `produces` 默认按 sorted set hash；order-sensitive slot 必须在 `notes` 里显式标注并按 ordered tuple hash  
7. `component_kind` 词表封闭：新增 slot 必须复用 §0.3 中已有的 kind，或显式 bump `fingerprint_schema_version`；不得在实现中临时扩表
