# 2026-04-17 `train/models.py` cleanup inventory / execution plan

Date: 2026-04-17  
Status: Draft / static-audit ready; no code deletion landed in this doc pass  
Scope: `train/models.py` 为主，必要时只引用 `train/model_ckpt_contract.py`、`train/training_MPL.py`、`train/posttrain.py`、`train/validate/*`、`tools/*`、`tests/*` 作为调用面证据。  
Method: 静态引用分析（`rg` + AST method scan + 代码阅读）。本轮未做 fresh-chain runtime rerun，因此所有 `Remove-*` 结论都需要按 checklist 再过一次 smoke。  
Goal: 给后续清理 `train/models.py` 里的 dead code、重复逻辑、compat / transition shell 提供一份可执行清单，优先降低维护风险，不改变模型数学语义、不改变 checkpoint/load 合约、不改变默认训练行为。  
Non-goal: 不在本轮直接删除 active checkpoint compat、不拆新 Python 模块、不做全文件格式化、不修改模型结构。

关联参考文档：

- 参考格式：`docs/delete/2026-04-14_train_legacy_compat_deletion_audit.md`
- 参考路线图：`docs/delete/2026-04-14_train_models_refactor_hotspots_roadmap.md`
- 当前承载文件：`train/models.py`
- 相关 contract：`train/model_ckpt_contract.py`

---

## 0. 结论先看

按当前静态扫描，`train/models.py` 的清理项可以分成 5 类：

### A. 基本可视为 dead / cold code，优先清理

1. `ContactMeasHeadLowerBodyNoHistV1`
2. `LOWER_BODY_INDICES_V1`
3. `numpy as np` import
4. `EventMotionModel._direct_pose_first_linear`
5. `EventMotionModel._direct_pose_last_linear`
6. `MotionJointLoss._angvel_hz`

### B. 只剩 re-export / thin wrapper 的兼容壳

1. `_build_pretrain_contact_encoder_input` 的 `models.py` re-export（已于 `2026-04-18` 删除；canonical home 为 `train.utils`）
2. `EventMotionModel.attach_motion_encoder(...)` 薄 wrapper
3. `MotionJointLoss.__init__(**legacy_kwargs)` deprecated loss key reject guard

### C. 明显重复逻辑，建议先抽 helper 再删重复块

1. `contacts_meas` canonicalize + pad/expand + `delta_meas` 计算，在 `forward` 里有两份近似实现
2. direct-pose feature source / phase mode / leg mode / gate mode 的参数归一化，在 model / contract / posttrain 多处重复
3. direct-pose leg/arm group 解析，在 model routing 和 `MotionJointLoss` 中重复
4. template hint formatting 在 `MotionJointLoss` 和 `Trainer` 侧重复

### D. 过渡层，当前不建议直接删

1. `direct_pose_out_leg` 与 `direct_pose_leg_terminal` 双轨
2. direct-pose split state / StepC leg-terminal checkpoint upgrade 相关字段
3. `attach_motion_encoder` 对外入口名称

### E. 模型内已经冷，但外层仍活跃的字段

1. `phase_reset_source` 在 `EventMotionModel` 内只做归一化，不参与 forward；真正逻辑在 posttrain / freerun runner 外层
2. 这类字段适合先做 contract 边界收口，不适合直接无提示删除

---

## 1. 状态标记

- `Remove-Now`：静态引用基本清空，删除前只需轻量 smoke。
- `Remove-If-Clean`：当前像 dead/cold，但需要先确认 docs/tools/tests/config 外部调用面。
- `Dedup-First`：不是删除项；先抽公共 helper，确认行为一致，再清掉重复块。
- `Keep-Guard`：虽然是 legacy/compat，但承担 fail-fast 或用户友好报错。
- `Keep-Compat`：薄 wrapper / 老名字仍是外部稳定 API。
- `Keep-Active`：checkpoint、训练、posttrain、validate 仍可能依赖，不能作为 cleanup 第一批删除。
- `Fix-or-Remove`：helper 本身未被调用，但其维护的状态/cache 仍活跃；先判断应接入调用点还是删除。
- `Revisit-With-Rerun`：静态冷，但需要 fresh basetrain->posttrain runtime hits 佐证。

---

## 2. 总表

| Item | Location | 类型 | 当前证据 | 建议 |
|---|---|---|---|---|
| `ContactMeasHeadLowerBodyNoHistV1` | `train/models.py` former symbol | unused module class | 已于 `2026-04-17` 删除；仓内剩余命中仅文档文本 | `Removed (2026-04-17)` |
| `LOWER_BODY_INDICES_V1` | `train/models.py` former symbol | unused const | 已于 `2026-04-17` 删除；仓内剩余命中仅文档文本 | `Removed (2026-04-17)` |
| `import numpy as np` | `train/models.py` former import | unused import | 已于 `2026-04-17` 删除 | `Removed (2026-04-17)` |
| `_build_pretrain_contact_encoder_input` re-export | `train/models.py` former import / `__all__` entry | compat re-export | 已于 `2026-04-18` 删除；canonical impl 保留在 `train/utils.py` | `Removed (2026-04-18)` |
| `_direct_pose_first_linear` | `train/models.py:1318` | unused private helper | repo 内仅定义，无调用 | `Remove-Now` |
| `_direct_pose_last_linear` | `train/models.py:1326` | unused private helper | repo 内仅定义，无调用 | `Remove-Now` |
| `_invalidate_weight_cache` | `train/models.py` former helper | unused private helper | 已于 `2026-04-18` 删除；`_joint_weight_cache` 仍由 active path 维护 | `Removed (2026-04-18)` |
| `_angvel_hz` | `train/models.py` former helper | unused private helper | 已于 `2026-04-17` 删除 | `Removed (2026-04-17)` |
| `MotionJointLoss._format_template_hint` | `train/models.py` | removed unused helper | loss-side 无任何调用；trainer 侧已有独立 `_format_template_hint` | `Removed (Phase F)` |
| `template_hint`, `bundle_hint` attrs | `train/models.py` | removed stale attrs | loss-side 无读者；`training_MPL.py` 的无意义写入已同步删除 | `Removed (Phase F)` |
| `phase_reset_source` in model | `train/models.py:465`, `train/models.py:600` | cold model attr | `EventMotionModel` 内只归一化；外层 runner 自己处理 phase reset | `Revisit-With-Rerun` |
| `direct_pose_out_leg` / `direct_pose_leg_terminal` dual path | `train/models.py:671`, `train/models.py:672`, `train/models.py:1123` | transition compat | tests / tools / checkpoint upgrade 仍覆盖老新 key | `Keep-Active` |
| `attach_motion_encoder` wrapper | `train/models.py:2871` | thin API shell | training / posttrain / validate 仍调用 model method | `Keep-Compat` |
| `legacy_kwargs` reject guard | `train/models.py:2915` | fail-fast guard | 当前未命中，但能阻止旧 loss keys 静默进入 | `Keep-Guard` |
| contacts meas canonicalization duplicate | `train/models.py:1972`, `train/models.py:2177` | duplicated forward block | expand/pad/prev/delta 逻辑重复 | `Dedup-First` |
| direct-pose mode normalization duplicate | `train/models.py:626`, `train/models.py:646`, `train/models.py:699`, `train/models.py:717` | duplicated normalization | contract 侧也有 normalize helper | `Dedup-First` |
| direct group resolver duplicate | `train/models.py:900`, `train/models.py:3118` | duplicated joint grouping | model 和 loss 各自解析 leg/arm/defaults | `Dedup-First` |

---

## 3. Recommended execution order

### Phase A — no-behavior dead cleanup

目标：先删静态确定的 dead private code，降低噪音，不碰 checkpoint / forward 主路径。

候选：

- `train/models.py:10` 删除 `import numpy as np`
- `train/models.py:1318` 删除 `_direct_pose_first_linear`
- `train/models.py:1326` 删除 `_direct_pose_last_linear`
- `train/models.py:3707` 删除 `_angvel_hz`

删除前 checklist：

- [x] `rg -n "_direct_pose_first_linear|_direct_pose_last_linear|_angvel_hz|np\\." train tools tests docs`
- [x] `python3 -m py_compile train/models.py`
- [x] `python3 - <<'PY'` import smoke：`import train.models`
- [x] object smoke：`EventMotionModel(in_state_dim=1, out_motion_dim=1, num_heads=1, hidden_dim=8)`

最终勾选：

- [x] Remove
- [ ] Keep
- [ ] Revisit

执行回填（2026-04-17）：

- 已删除 `train/models.py:10` 的 `import numpy as np`
- 已删除 `EventMotionModel._direct_pose_first_linear`
- 已删除 `EventMotionModel._direct_pose_last_linear`
- 已删除 `MotionJointLoss._angvel_hz`
- 删除前 repo 级搜索结果：上述 3 个 private helper 仅在 `train/models.py` 定义、以及本 inventory 文档中出现；`train/models.py` 内无 `np.` 使用
- 删除后复查：`rg -n "_direct_pose_first_linear|_direct_pose_last_linear|_angvel_hz" train tools tests docs` 仅剩文档文本；`rg -n "\\bnp\\." train/models.py` 与 `rg -n "import numpy as np" train/models.py` 均无命中
- `python3 -m py_compile train/models.py` 通过
- import smoke 通过：`import train.models` 输出 `ok`
- instantiate smoke 通过：`EventMotionModel` 可成功构造；环境中出现一次 PyTorch `Initializing zero-element tensors is a no-op` warning，但不影响实例化结果

### Phase B — contact-meas retired head audit

目标：确认 `ContactMeasHeadLowerBodyNoHistV1` 是否只是设计文档残留；如果没有任何 checkpoint / config / tool 需要它，则删除 class + const。

候选：

- `train/models.py:46` `LOWER_BODY_INDICES_V1`
- `train/models.py:68` `ContactMeasHeadLowerBodyNoHistV1`

风险点：

- 文档 `docs/contact_meas_head_redesign_lowerbody_nohist.md` 仍明确提到该 head。
- 如果后续要复活 lower-body no-history contact-meas head，删除会让文档失去对应实现。

删除前 checklist：

- [x] `rg -n "ContactMeasHeadLowerBodyNoHistV1|LOWER_BODY_INDICES_V1" .`
- [x] 确认没有 checkpoint state dict 包含 `ContactMeasHeadLowerBodyNoHistV1` 相关 module key
- [x] 更新或标注 `docs/contact_meas_head_redesign_lowerbody_nohist.md`
- [x] `python3 -m py_compile train/models.py train/model_ckpt_contract.py`

最终勾选：

- [x] Remove
- [ ] Keep as archived implementation
- [ ] Move note to retired docs

执行回填（2026-04-17）：

- 已删除 `train/models.py` 中的 `LOWER_BODY_INDICES_V1`
- 已删除 `train/models.py` 中的 `ContactMeasHeadLowerBodyNoHistV1`
- repo 级搜索结果：代码侧未发现任何实例化/消费；删除后 `train/models.py` 内已无上述符号，仓内剩余命中仅为文档文本
- checkpoint audit：扫描 `models/` 与 `debug_output/` 下 1299 个 checkpoint-like 文件，`torch.load(..., weights_only=True)` 均可读取；未发现相关 top-level key / `state_dict` key 命中
- 二进制字符串扫描：`rg -a -n "ContactMeasHeadLowerBodyNoHistV1|LOWER_BODY_INDICES_V1" models debug_output` 无命中
- 文档处理：`docs/contact_meas_head_redesign_lowerbody_nohist.md` 已标注为 archived design / debugging note，并注明这两个符号已于 2026-04-17 从 `train/models.py` 删除
- `python3 -m py_compile train/models.py train/model_ckpt_contract.py` 通过
- import smoke 通过：`import train.models` 输出 `ok`
- instantiate smoke 通过：`EventMotionModel` 可成功构造；环境中出现一次 PyTorch `Initializing zero-element tensors is a no-op` warning，但不影响实例化结果

### Phase C — `contacts_meas` canonicalization dedup

目标：把 forward 中重复的 contacts input 规范化逻辑收成单 helper，减少 event-clock / non-event-clock 两条路径漂移。

重复位置：

- event-clock branch: `train/models.py:1972`
- non-event-clock branch: `train/models.py:2177`

建议 helper 形态：

- 输入：`contacts_input`, `meas_logits_prev`, `B`, `Tq`, `contact_dim`, `device`, `dtype`
- 输出：`contacts_meas`, `delta_meas`, `meas_prev_t`
- 行为要求：
  - 支持 `(C)`, `(B,C)`, `(B,T,C)`
  - 支持 batch broadcast 和 time broadcast
  - 不吞 shape mismatch；至少只吞 tensor conversion 失败
  - pad/truncate 逻辑保持原语义

验收指标：

- [x] `EventMotionModel.forward` 行数下降
- [x] `contacts_meas` canonicalization 逻辑只剩一个实现
- [x] `contacts_plan` / `event_clock_delta_meas` 输出 key 不变
- [x] `tests/train/test_event_motion_model_refactor_phase_d.py` 通过

最终勾选：

- [x] Dedup landed
- [ ] Keep duplicated due to risk
- [ ] Revisit after runtime rerun

执行回填（2026-04-17）：

- 已新增 helper：`train/models.py:1575` `_canonicalize_contacts_meas_inputs(...)`
- event-clock branch 与 non-event-clock branch 已改为共用 helper：
  - `train/models.py:1987`
  - `train/models.py:2109`
- helper 行为：
  - 支持 `contacts_input` 的 `(C)`, `(B,C)`, `(B,T,C)`
  - 支持 batch/time singleton broadcast
  - 保留 channel 维 pad/truncate 语义
  - 对 `contacts_input` / `meas_logits_prev` 的 tensor conversion failure 保持容错
  - 对 batch/time shape mismatch 改为显式 `ValueError`，不再静默回退到零 meas
- 输出兼容性：
  - `contacts_plan` 仍由 `train/models.py:2155` 生成
  - `event_clock_delta_meas` 输出 key 仍保留在 `train/models.py:2389`
- 新增 targeted regression tests：`tests/train/test_event_motion_model_phase_c_contacts_meas.py`
  - helper broadcast + `delta_meas` 计算
  - channel pad/truncate
  - shape mismatch raises
  - conversion failure fallback
  - event-clock / non-event-clock forward integration
- 验证结果：
  - `python3 -m py_compile train/models.py tests/train/test_event_motion_model_phase_c_contacts_meas.py` 通过
  - `python3 -m unittest -v tests.train.test_event_motion_model_phase_c_contacts_meas`：6 tests 全部通过
  - import smoke 通过：`import train.models` 输出 `ok`
  - instantiate smoke 通过：`EventMotionModel` 可成功构造；环境中出现一次 PyTorch `Initializing zero-element tensors is a no-op` warning，但不影响实例化结果
  - `python3 -m unittest -v tests.train.test_event_motion_model_refactor_phase_d`：4 tests 全部通过
  - 兼容性 blocker 已清理：恢复 `_maybe_upgrade_direct_pose_split_state_dict`、`_maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict` 与 `load_state_dict(...)` 的 pre-load compat 调用

### Phase D — mode normalization single source

目标：把字符串 alias / canonicalization 从 `EventMotionModel.__init__` 内联块收口，尽量复用 `train/model_ckpt_contract.py` 的规范化定义或移动到现有 utility 边界。

重复对象：

- `direct_pose_feat_source`
- `direct_pose_phase_z_mode`
- `direct_pose_leg_mode`
- `direct_pose_leg_gate_mode`
- `contact_plan_init_mode`
- `lambda_fusion_mode`

当前重复证据：

- model ctor 内联归一化：`train/models.py:592`, `train/models.py:626`, `train/models.py:646`, `train/models.py:699`, `train/models.py:717`, `train/models.py:759`
- contract helper：`train/model_ckpt_contract.py:99`, `train/model_ckpt_contract.py:232`, `train/model_ckpt_contract.py:340`, `train/model_ckpt_contract.py:582`
- posttrain config parsing 也会再次归一化：`train/posttrain.py:405`, `train/posttrain.py:542`, `train/posttrain.py:551`

建议顺序：

1. 先只替换 `direct_pose_leg_gate_mode`，因为已有 public helper `normalize_direct_pose_leg_gate_mode`
2. 再处理 `direct_pose_phase_z_mode` / `direct_pose_feat_source`
3. 最后处理 `contact_plan_init_mode` / `lambda_fusion_mode`

风险点：

- `train/model_ckpt_contract.py` 当前有 `TYPE_CHECKING` 引用模型；若 `models.py` 运行时 import contract helper，需要确认不会引入循环导入。
- 不建议新增新模块；如果循环导入风险高，则先在 `model_ckpt_contract.py` 中导出纯常量/纯函数，保持低耦合。

验收指标：

- [x] alias map 只保留一个 source of truth
- [x] unsupported values 的报错语义不变或更严格
- [x] active configs 能 instantiate `EventMotionModel`

最终勾选：

- [x] Dedup landed
- [ ] Keep duplicated due to import-cycle risk
- [ ] Revisit after contract boundary cleanup

执行回填（2026-04-17）：

- 已在 `train/model_ckpt_contract.py` 导出统一 public canonicalizers：
  - `normalize_contact_plan_init_mode` at `train/model_ckpt_contract.py:145`
  - `normalize_direct_pose_phase_z_mode` at `train/model_ckpt_contract.py:164`
  - `normalize_direct_pose_feat_source` at `train/model_ckpt_contract.py:183`
  - `normalize_direct_pose_leg_mode` at `train/model_ckpt_contract.py:209`
  - `normalize_direct_pose_leg_gate_mode` at `train/model_ckpt_contract.py:122`
  - `normalize_lambda_fusion_mode` at `train/model_ckpt_contract.py:228`
- `EventMotionModel.__init__` 已切到同一 source of truth：
  - `train/models.py:532`
  - `train/models.py:567`
  - `train/models.py:581`
  - `train/models.py:630`
  - `train/models.py:646`
  - `train/models.py:682`
- `resolve_posttrain_build_state_from_contract(...)` 已切到相同 canonicalizers：
  - `train/model_ckpt_contract.py:760`
  - `train/model_ckpt_contract.py:766`
  - `train/model_ckpt_contract.py:772`
  - `train/model_ckpt_contract.py:778`
  - `train/model_ckpt_contract.py:784`
- `train/posttrain.py` config parsing 也已改为复用同一组 helper，避免 build-time drift：
  - `train/posttrain.py:483`
  - `train/posttrain.py:489`
  - `train/posttrain.py:666`
  - `train/posttrain.py:672`
  - `train/posttrain.py:678`
  - `train/posttrain.py:684`
- 语义说明：
  - 默认 fallback 语义保持与 `EventMotionModel.__init__` 原行为一致（invalid -> default）
  - strict 模式下对 unsupported 值改为统一 fatal rejection，覆盖 contract / posttrain 入口
  - `direct_pose_leg_gate_mode` alias 集补齐了 `mlp|net|nn|learn|gate -> learned`，保持旧模型构造兼容
- 新增 regression tests：`tests/train/test_event_motion_model_phase_d_mode_normalization.py`
  - public canonicalizer alias coverage
  - strict mode invalid rejection
  - model ctor alias canonicalization
  - invalid value default fallback
- 验证结果：
  - `python3 -m py_compile train/models.py train/model_ckpt_contract.py train/posttrain.py tests/train/test_event_motion_model_phase_d_mode_normalization.py` 通过
  - `python3 -m unittest -v tests.train.test_event_motion_model_phase_d_mode_normalization tests.train.test_event_motion_model_refactor_phase_d tests.train.test_event_motion_model_phase_c_contacts_meas`：14 tests 全部通过
  - import smoke 通过：`import train.models` 输出 `ok`
  - instantiate smoke 通过：`EventMotionModel` 可成功构造；环境中出现一次 PyTorch `Initializing zero-element tensors is a no-op` warning，但不影响实例化结果

### Phase E — direct group resolver dedup

目标：model direct-pose split routing 和 loss direct-pose group masks 使用同一套 joint spec 解析语义，避免 leg/arm/default bones 漂移。

重复位置：

- model routing：`train/models.py:900`
- loss group masks：`train/models.py:3118`

当前共享基础：

- `_resolve_joint_spec_indices` 已在 `train/utils.py`，且 `models.py` 已 import。
- `DEFAULT_DIRECT_POSE_LEG_BONES` 和 `STAGE6_3WAY_ARMCHAIN_BONES` 已是 shared constants。

建议：

- 不新增模块。
- 在 `train/utils.py` 或 `train/models.py` 内已有 helper 基础上，收一个“joint indices -> output dim mask / out_idx”的小 helper。
- 确保 model side 和 loss side 都通过同一 resolver 得到 leg/arm indices。

验收指标：

- [x] leg/arm/default bone 解析逻辑不再手写两套
- [x] direct split index coverage check 保留
- [x] `tests/train/test_event_motion_model_refactor_phase_d.py` 通过

最终勾选：

- [x] Dedup landed
- [ ] Keep duplicated due to output-mask semantics
- [ ] Revisit

执行回填（2026-04-18）：

- 已新增 shared resolver：`train/models.py:68` `_resolve_direct_group_indices(...)`
- `EventMotionModel` side 已切到 shared resolver：
  - leg residual metadata 解析：`train/models.py:850`
  - split routing leg/arm group 解析：`train/models.py:878`
- `MotionJointLoss` side 已切到 shared resolver：
  - `train/models.py:3199` `_resolve_direct_group_masks(...)`
  - `train/models.py:3204` 调用 `_resolve_direct_group_indices(...)`
- 保留的语义边界：
  - model side 的 `build_split_out_index(...)` / output-dim coverage check 仍在本地保留，不与 loss mask 逻辑硬合并
  - loss side 的 root exclusion (`root_idx`) 仍保留在 `_resolve_direct_group_masks(...)`，不污染 model routing
  - `direct_pose_leg_enable` 分支继续允许自定义 lower-body 默认顺序（ball/foot/calf/thigh），split/loss 共享的仍是 `DEFAULT_DIRECT_POSE_LEG_BONES` / `STAGE6_3WAY_ARMCHAIN_BONES`
- 结果：
  - model split routing 与 loss group masks 现在通过同一 leg/arm joint-spec resolver 获取 indices，避免 arm-vs-leg overlap filtering 漂移
  - overlap 语义统一：arm indices 会先减去 leg indices，再生成 arm/else 分组
- 新增 regression test：`tests/train/test_event_motion_model_phase_e_direct_groups.py`
  - 校验 model `direct_pose_leg_out_idx` / `direct_pose_arm_out_idx` / `direct_pose_else_out_idx`
  - 与 loss-side `leg` / `arm` / `else` masks 保持同一 membership 语义
- 验证结果：
  - `python3 -m py_compile train/models.py tests/train/test_event_motion_model_phase_e_direct_groups.py` 通过
  - `python3 -m unittest -v tests.train.test_event_motion_model_phase_e_direct_groups tests.train.test_event_motion_model_refactor_phase_d tests.train.test_event_motion_model_phase_c_contacts_meas tests.train.test_event_motion_model_phase_d_mode_normalization`：15 tests 全部通过
  - import smoke 通过：`import train.models` 输出 `ok`
  - instantiate smoke 通过：`EventMotionModel` 可成功构造；环境中出现一次 PyTorch `Initializing zero-element tensors is a no-op` warning，但不影响实例化结果

### Phase F — cold compat attr review

目标：处理模型内已经冷、但外部仍有同名概念的 compat attrs，避免误删 active runner 逻辑。

候选：

- `phase_reset_source` in `EventMotionModel.__init__`
- `_build_pretrain_contact_encoder_input` re-export
- `MotionJointLoss._format_template_hint`
- `template_hint` / `bundle_hint`

建议：

- `phase_reset_source`：先不要删。先做 runtime hit 统计，确认没有外部依赖 `model.phase_reset_source`。
- `_build_pretrain_contact_encoder_input`：repo 内 active imports 已切到 `train.utils`；`train.models` compat re-export 已于 `2026-04-18` 删除。
- `MotionJointLoss._format_template_hint`：如果没有任何 loss-side normalizer error path 需要它，可以和 `template_hint` / `bundle_hint` attrs 一起删；但要确认 `training_MPL.py:3822` 设置这些字段不会变成无意义写入。

删除前 checklist：

- [x] `rg -n "model\\.phase_reset_source|getattr\\(model, ['\\\"]phase_reset_source" .`
- [x] `rg -n "_build_pretrain_contact_encoder_input" .`
- [x] `rg -n "template_hint|bundle_hint|_format_template_hint" train tools tests docs`
- [ ] fresh basetrain -> posttrain runtime hits 确认

最终勾选：

- [x] Remove
- [x] Keep-Compat
- [ ] Keep-Guard
- [ ] Revisit-With-Rerun

执行回填（2026-04-18）：

- `phase_reset_source`：
  - 精确 grep `rg -n "model\\.phase_reset_source|getattr\\(model, ['\\\"]phase_reset_source" .` 仅命中本 inventory 文档，无 repo 内 runtime 读取方。
  - 但广义 `phase_reset_source` 在 `tools/` / `docs/` / `tests/` / validate/posttrain CLI 中仍大量活跃，且 `tests/train/test_event_motion_model_refactor_phase_d.py:80` 仍通过 ctor 传入该参数。
  - 结论：本轮不删；inventory 状态维持 `Revisit-With-Rerun`，实现上继续保留 keep-compat 的 ctor 归一化与 attr 存储，避免外层 active runner / config 语义漂移。
- `_build_pretrain_contact_encoder_input` re-export：
  - `rg -n "_build_pretrain_contact_encoder_input" train tools tests docs` 显示 active 实现在 `train/utils.py:116`，repo 内 active 调用已走 `train/training_MPL.py:79,656`。
  - `train/models.py` 在 follow-up patch 前只剩 import + `__all__` re-export，repo 内已无内部消费者。
  - follow-up（`2026-04-18`）：已删除 `train/models.py` 中对应 compat import / re-export；仓内剩余代码引用仅保留 `train.utils.py:116` 与 `train/training_MPL.py:79,656`。
- `MotionJointLoss._format_template_hint` / `template_hint` / `bundle_hint`：
  - `rg -n "template_hint|bundle_hint|_format_template_hint" train tools tests docs` 显示：
    - `train/models.py` 仅有定义与 attr 自读；
    - `train/training_MPL.py` 有独立 trainer-side `_format_template_hint(...)`；
    - `train/training_MPL.py:3822-3823` 对 `loss_fn.template_hint` / `loss_fn.bundle_hint` 的赋值无任何后续读取方。
  - 已删除：
    - `train/models.py` 中 `MotionJointLoss._format_template_hint(...)`
    - `train/models.py` 中 `self.template_hint` / `self.bundle_hint`
    - `train/training_MPL.py` 中对 `loss_fn.template_hint` / `loss_fn.bundle_hint` 的无意义写入
- 验证结果：
  - `python3 -m py_compile train/models.py train/training_MPL.py` 通过
  - `python3 -m unittest -v tests.train.test_event_motion_model_phase_e_direct_groups tests.train.test_event_motion_model_refactor_phase_d tests.train.test_event_motion_model_phase_c_contacts_meas tests.train.test_event_motion_model_phase_d_mode_normalization`：15 tests 全部通过
  - import smoke 通过：`import train.models` 输出 `ok`
  - instantiate smoke 通过：`EventMotionModel` 可成功构造；环境中出现一次 PyTorch `Initializing zero-element tensors is a no-op` warning，但不影响实例化结果
  - loss smoke 通过：`MotionJointLoss(output_layout={})` 可成功构造

---

## 4. Do-not-delete list for this cleanup round

这些项虽然看起来像兼容层，但当前不建议作为第一批删除：

1. `EventMotionModel.attach_motion_encoder(...)`
   - 外部 active 调用仍在 `train/training_MPL.py`、`train/posttrain.py`、`train/validate/run_freerun_cycles.py`、`train/validate/run_teacher_rollout.py`
   - 当前只是把主实现放在 `train/model_ckpt_contract.py`，model method 是稳定 API

2. `direct_pose_out_leg` / `direct_pose_leg_terminal` 双路径
   - StepC 过渡仍有 tests 覆盖
   - 多个 tools 仍按 module name 做 probe / transplant / attribution
   - checkpoint upgrade 仍需要识别老 key

3. `MotionJointLoss(**legacy_kwargs)` reject guard
   - 当前不是 active path，但它是 fail-fast guard
   - 删除后旧配置可能变成更深层、更难定位的错误

4. checkpoint build/load compat
   - 不在本文件里直接清理
   - 任何涉及 state dict key rewrite / upgrade 的逻辑必须单独做 runtime rerun

---

## 5. Suggested validation matrix

最小验证：

- [ ] `python3 -m py_compile train/models.py`
- [ ] `python3 - <<'PY'` import smoke：`import train.models`
- [ ] `python3 - <<'PY'` instantiate smoke：`EventMotionModel(in_state_dim=1, out_motion_dim=1, hidden_dim=8, num_heads=1)`
- [ ] `python3 - <<'PY'` loss smoke：`MotionJointLoss(output_layout={})`

定向测试：

- [ ] `python3 -m pytest tests/train/test_event_motion_model_refactor_phase_d.py`
- [ ] direct-pose split enable smoke
- [ ] direct-pose StepC unified leg terminal smoke
- [ ] contact-plan enable + event-clock enable smoke

如果进入 compat attr 删除：

- [ ] basetrain minimal run
- [ ] posttrain load from fresh checkpoint
- [ ] freerun validate load from posttrain checkpoint
- [ ] export path smoke if ONNX/export remains required

---

## 6. Open questions

1. `phase_reset_source` 是否还需要作为 `EventMotionModel` 构造参数存在，还是应完全归 posttrain / validate runner 管？
2. direct-pose mode normalization 是否允许从 `models.py` import `model_ckpt_contract.py` 的 helper，还是需要把纯 normalize helper 下沉到已有 `train/utils.py`？

---

## 7. Recommended first PR / commit slice

该建议切片现已完成并过时；当前代码状态已包含下列净减：

1. 删除 `np` unused import
2. 删除 `_direct_pose_first_linear`
3. 删除 `_direct_pose_last_linear`
4. 删除 `_angvel_hz`
5. 删除 `ContactMeasHeadLowerBodyNoHistV1` + `LOWER_BODY_INDICES_V1`
6. 删除 `_build_pretrain_contact_encoder_input` 的 `train.models` compat re-export

第一刀不要碰：

- `phase_reset_source`
- `direct_pose_out_leg` / `direct_pose_leg_terminal`
- checkpoint compat / state dict upgrade
- `attach_motion_encoder`
- `MotionJointLoss(**legacy_kwargs)` guard
