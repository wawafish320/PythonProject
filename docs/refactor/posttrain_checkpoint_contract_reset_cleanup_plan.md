# `train.posttrain` Checkpoint Contract Reset 清理路线图

Date: 2026-04-17  
Status: Draft  
Scope: `train/posttrain.py`, `train/model_ckpt_compat.py`, `train/models.py`, posttrain configs, validate/export loaders  
Premise: **不再兼容旧 checkpoint**；只支持当前 mainline 重新产出的 canonical checkpoint。  

---

## 1. 背景

当前 `train.posttrain` 已经基本收敛到 newflow 主链：

- `train_direct_pose`：Stage6/70/71/72 direct expert continuation
- `train_lambda_head`：lambda final
- `encoder_bundle`：posttrain mainline 唯一 encoder bundle 字段
- 旧高阶 direct-pose 分支：active config 直接 fail-fast

但代码里仍然保留一层 checkpoint compatibility：

- 从 checkpoint tensor shape 反推模型结构
- 加载前 drop / adapt 老 tensor
- direct-pose split / StepC terminal 旧形状升级
- `strict=False` 容忍 missing/unexpected keys
- lambda-final config 里仍有历史命名后缀

如果目标是“完全清干净”，最核心的策略不是继续削 compat，而是定义新 checkpoint contract，然后让所有 loader 只接受这个 contract。

---

## 2. 新策略

### 2.1 新原则

1. **No legacy checkpoint support**
   - 没有 contract version 的 checkpoint 视为 archive。
   - 不再主链加载。

2. **No tensor-shape archaeology**
   - 不再从 `state_dict` tensor 形状猜测 direct-pose / lambda / event-clock 拓扑。
   - 模型重建只读显式 `build_cfg`。

3. **Strict loading**
   - 主链 loader 使用 `strict=True`。
   - missing/unexpected key 是 checkpoint contract 错误，不是 silent fallback。

4. **No inert config keys**
   - `train_lambda_head=true` 时，不允许出现 `*_train_only`。
   - 默认 `contact_plan_init_*` 不再写入输出 config。

5. **Archive separation**
   - 旧 checkpoint 复现放到 archived lane / historical script。
   - 主链不保留兼容壳。

---

## 3. 新 checkpoint contract

建议 posttrain 保存 checkpoint 时写入：

```python
{
    "model": model.state_dict(),
    "posttrain_cfg": cfg_jsonable,
    "checkpoint_contract": {
        "name": "posttrain_newflow",
        "version": 1,
        "created_by": "train.posttrain",
    },
    "build_cfg": {
        "in_state_dim": int(ds.Dx),
        "out_motion_dim": int(ds.Dy),
        "cond_dim": int(ds.Dc),
        "hidden_dim": int(model.hidden_dim),
        "depth": int(cfg.depth),
        "num_heads": int(cfg.num_heads),
        "dropout": float(cfg.dropout),
        "context_len": int(cfg.context_len),
        "contact_dim": int(build_state.contact_dim),
        "angvel_dim": int(build_state.angvel_dim),
        "pose_hist_dim": int(build_state.pose_hist_dim),
        "period_dim": int(model.period_dim),
        "contact_plan": {...},
        "direct_pose": {...},
        "event_clock": {...},
        "lambda_fusion": {...},
    },
}
```

### 3.1 Loader rule

新 loader 必须先检查：

```python
contract = ckpt.get("checkpoint_contract")
if not isinstance(contract, dict) or contract.get("name") != "posttrain_newflow":
    raise SystemExit("[FATAL] unsupported checkpoint contract; regenerate with current train.posttrain.")
if int(contract.get("version", -1)) != 1:
    raise SystemExit("[FATAL] unsupported posttrain checkpoint contract version.")
```

然后从 `build_cfg` 实例化模型，不再调用 compat inference。

---

## 4. 推荐改动顺序

### Phase A — 写入新 contract（低风险）

目标：先让新产出的 checkpoint 带完整 contract，不马上删除旧 loader。

#### A1. 在 posttrain 保存时加入 contract

修改点：

- `train/posttrain.py` 的 `_save_posttrain_outputs(...)`

动作：

- 写入 `checkpoint_contract`
- 写入显式 `build_cfg`
- 确保 `posttrain_cfg` 不再带 lambda-mode inert keys

验收：

- `python3 -m py_compile train/posttrain.py`
- 新跑一个最小 posttrain smoke，确认 ckpt 里有 `checkpoint_contract` 和 `build_cfg`

#### A2. 新增 contract reader

建议新增函数：

- `load_posttrain_contract_ckpt_payload(...)`
- `resolve_posttrain_build_state_from_contract(...)`

建议文件：

- 如果保留旧文件名：`train/model_ckpt_compat.py`
- 如果想重命名：新建 `train/model_ckpt_contract.py`

验收：

- 对新 ckpt 能完整实例化模型
- 不依赖 tensor shape inference

---

### Phase B — strict loader 切换（中风险）

目标：所有主链 loader 都走新 contract + strict load。

#### B1. posttrain 主入口 strict load

修改点：

- `train/posttrain.py`

动作：

- 替换 `prepare_event_motion_ckpt_state_for_load(...)`
- 直接读取 contract build state
- `model.load_state_dict(state_dict, strict=True)`

验收：

- 新 ckpt 可加载
- 旧 ckpt 明确 fail-fast
- missing/unexpected key 不再 silent

#### B2. validate/export loader 同步

修改点：

- `train/validate/run_freerun_cycles.py`
- `train/validate/run_teacher_rollout.py`
- `train/export_onnx_from_ckpt.py`

动作：

- 同步改为 contract reader
- `strict=True`
- 对无 contract checkpoint 明确报错

验收：

- 当前 canonical ckpt 的 freerun / teacher rollout / export 能跑通
- 旧 ckpt 报错信息明确指向“需要重新生成”

---

### Phase C — 删除 compat 核心（高风险）

目标：删除只服务旧 checkpoint 的 upgrade / drop / adapt。

#### C1. 删除 direct-pose load compat

可删对象：

- `train/model_ckpt_compat.py` 中的 `apply_direct_pose_ckpt_compat(...)`
- direct-pose phase-z input adapt
- retired high-order tensor drop
- incompatible leg tensor drop
- leg bones mismatch re-init / shape mismatch drop

注意：

- 如果 current model 仍保留 side-routing 模块代码，应单独决定是否删除模型结构。
- 本 phase 只删 checkpoint load 兼容，不一定删模型能力。

验收：

- `rg -n "apply_direct_pose_ckpt_compat|prepare_event_motion_ckpt_state_for_load" train`
- 主链不再引用上述函数

#### C2. 删除 old-shape upgrade 函数

可删对象：

- `maybe_upgrade_direct_pose_split_state_dict(...)`
- `maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict(...)`
- `train/models.py` 中对应 wrapper:
  - `_maybe_upgrade_direct_pose_split_state_dict(...)`
  - `_maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict(...)`

验收：

- `rg -n "maybe_upgrade_direct_pose" train` 无主链引用
- `python3 -m py_compile train/models.py train/model_ckpt_compat.py`

#### C3. 精简 / 重命名 compat 文件

如果 `train/model_ckpt_compat.py` 只剩 current contract 逻辑，建议：

- 重命名为 `train/model_ckpt_contract.py`
- 保留：
  - ckpt payload load
  - contract validation
  - build_cfg dataclasses
  - attach encoder bundle helper
- 删除：
  - `compat`
  - `upgrade`
  - `legacy`
  - `retired tensor drop`

验收：

- `rg -n "compat|legacy|retired|upgrade" train/model_ckpt_contract.py` 只剩注释中必要说明，最好为 0

当前状态（2026-04-17，C3b / C3c）：

- `train/model_ckpt_contract.py` 已承担 current posttrain checkpoint contract 主实现：
  - contract validation
  - ckpt payload load
  - build_cfg dataclasses / resolve helpers
  - encoder bundle attach helper
- `train/posttrain.py`、`train/models.py`、`train/validate/*`、`train/export_onnx_from_ckpt.py` 已直接从 `train/model_ckpt_contract.py` import，不再依赖 compat shim。
- `train/model_ckpt_compat.py` 已收口为 resume-only legacy shim，只保留：
  - `ResumeLoadReport`
  - `resume_load_weights_compat(...)`
- `train/training_MPL.py` 仍依赖 `resume_load_weights_compat(...)` 做 basetrain resume，因此当前还不能直接删除 `train/model_ckpt_compat.py`。

---

### Phase D — 配置与命名去历史化（低/中风险）

目标：repo 中 current posttrain configs 不再暗示旧兼容路径。

#### D1. lambda-final config 去旧后缀

对象：

- current active lambda-final configs under `config/posttrain_WalkF_stage7_lambda_final_calib_*.json`
- historical `stage72` lambda-final entry renamed to `config/posttrain_WalkF_stage7_lambda_final_calib_20260226_frombase.json`

动作：

- 文件名改为 `lambda_final_*`
- `run_name` 同步去掉旧后缀
- docs / tools 中引用同步修改

验收：

- current config / tools / train 不再引用旧 lambda-final 后缀
- 只允许 historical docs / archive 中残留

#### D2. lambda config 禁止 train-only 字段

当前已清理方向：

- `train_lambda_head=true` 时输出不写 `*_train_only`
- 静态 lambda configs 不再带 `*_train_only`

下一步可加强：

- 输入 config 如果 `train_lambda_head=true` 且存在任何 `*_train_only`，直接 fatal

验收：

- `rg -n "_train_only" config/posttrain*lambda*.json`
- 主链 lambda config 中应为 0

#### D3. posttrain config defaults 精简

可考虑移除默认显式字段：

- `contact_plan_init_mode=learnable`
- `contact_plan_init_hidden=128`
- `contact_plan_init_dropout=0.0`

保留规则：

- 非默认值可以保留
- 默认值不写

验收：

- `rg -n "contact_plan_init_(mode|hidden|dropout)" config/posttrain*.json`
- current mainline config 中默认值不再反复出现

---

## 5. 明确不建议同步做的事

### 5.1 不要在同一轮删除模型结构

例如：

- `direct_pose_leg_head_shared`
- `direct_pose_leg_gate_head_shared`
- `direct_pose_leg_side_embed`
- `direct_pose_leg_side_sign_gate_head`

这些虽然和旧高阶路径有关，但属于模型能力层，不只是 checkpoint loader。

建议顺序：

1. 先删除 checkpoint compat
2. 验证 current strict ckpt 全链路
3. 再决定是否删除模型结构

### 5.2 不要同时改 basetrain 的 `encoder_path`

`train.posttrain` 已收敛到 `encoder_bundle`。

但 `train/training_MPL.py` 当前仍使用 `encoder_path` 作为 basetrain contract。  
这应另开一轮 repo-wide naming cleanup，不要和 checkpoint contract reset 混在一起。

### 5.3 不要保留新旧双 loader 太久

过渡可以短期存在：

- `load_legacy_ckpt_for_archive(...)`
- `load_contract_ckpt(...)`

但主链入口只能走 `load_contract_ckpt(...)`。  
否则 compat 壳会重新长回来。

---

## 6. 建议最终结构

### 6.1 文件职责

`train/posttrain.py`

- CLI
- config parse
- current model build
- train loop
- save current contract checkpoint

`train/model_ckpt_contract.py`

- validate checkpoint contract
- load `model` state_dict
- read `build_cfg`
- attach encoder bundle
- no shape upgrade
- no tensor drop

`train/models.py`

- model definition
- no checkpoint migration wrapper

`train/validate/*`

- only load current contract checkpoint
- old checkpoint evaluation requires archive script

### 6.2 理想 grep 状态

主链代码里应接近：

```bash
rg -n "compat|legacy|upgrade|retired|strict=False" train/posttrain.py train/model_ckpt_contract.py train/models.py
```

期望：

- `strict=False`：0
- `maybe_upgrade_direct_pose`：0
- `apply_direct_pose_ckpt_compat`：0
- lambda-final legacy suffix：0
- `encoder_path`：仅 basetrain 文件中存在，posttrain 不存在

---

## 7. 风险与回滚

### 7.1 最大风险

最大风险不是代码编译失败，而是当前仍在用的实验 checkpoint 没有新 contract。

因此必须先产出一批新 canonical ckpt：

- Stage6-StepC
- 70a
- replace
- 70R
- 71
- 72
- lambda final

并确认 validate/export 都能 strict load。

### 7.2 回滚策略

推荐分支策略：

- `main`: current stable
- `cleanup/posttrain-contract-v1`: contract reset
- archive loader 如需保留，放到单独脚本，不接 mainline

如果 strict load 发现缺字段：

1. 不恢复 tensor-shape inference
2. 在 `build_cfg` 中补显式字段
3. 重新生成 checkpoint

---

## 8. 最小执行清单

按最小可执行顺序：

1. 在 `_save_posttrain_outputs` 写 `checkpoint_contract` + `build_cfg`
2. 新增 contract reader
3. `train.posttrain` 切到 contract reader + `strict=True`
4. 重新生成一条 canonical chain
5. validate/export 切到 contract reader + `strict=True`
6. 删除 `prepare_event_motion_ckpt_state_for_load`
7. 删除 `apply_direct_pose_ckpt_compat`
8. 删除 `maybe_upgrade_direct_pose_*`
9. 删除 `train/models.py` 中 upgrade wrapper
10. 去掉 lambda-final 旧后缀文件名 / run_name
11. grep 验收

---

## 9. 判断标准

完成后，应该能回答：

- 当前 ckpt 为什么能加载？因为有明确 `checkpoint_contract` 和 `build_cfg`。
- 旧 ckpt 为什么不能加载？因为没有 current contract。
- 模型为什么这么构建？因为 `build_cfg` 显式声明，而不是 tensor shape 猜测。
- lambda config 为什么干净？因为 lambda 模式不再携带 direct/leg train-only 历史字段。

这就是“清干净”的边界。

---

## 10. 后续独立清理：side-routing model-structure removal

2026-04-17 起，`direct_pose_leg_side_*` / `direct_pose_leg_head_shared` / `direct_pose_leg_gate_head_shared` 的模型结构删除已从本 contract reset 文档中拆出为独立计划。

原因：

- 本文档的边界是 checkpoint contract reset 与 strict loader 收敛。
- side-routing 删除会触及 `train/models.py` 构造面、forward 面、freerun/tooling/config surface，blast radius 大于 loader reset。
- 当前执行策略要求先做 ckpt/config/downstream scan，再 atomic 删除 `models.py` side-routing 构造与 forward 分支。

关联文档：

- 主控计划：`docs/refactor/2026-04-17_side_routing_removal_plan.md`
- P0/P1/P2 证据：`docs/changes/2026-04-17_side_routing_removal_p0_scan.md`

关系说明：

- 本文档仍是 posttrain checkpoint contract reset 的历史主文档。
- side-routing removal 是 contract reset 之后的结构性 follow-up。
- `POSTTRAIN_CHECKPOINT_CONTRACT_VERSION = 2` 的执行细节以后续 side-routing removal plan 为准。
