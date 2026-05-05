# Removal Policy: 删除 / 移除分支的契约

> Last updated: 2026-04-21
> Status: standing policy
> Scope: 任何 `train/` / `tools/` / `config/` 下删除 code path、config key、ckpt field、CLI flag、runtime attr 的操作
> 阅读对象：动手删代码的人（人类或 LLM）

---

## 1) 为什么有这份文档

具体故障模式（参考 commit `c26ea1c0` "remove legacy direct pose leg split path"）：

- 删掉了 legacy 入口的实现
- **没有**在 entry / load 路径加显式 reject
- 老 ckpt 加载时不报错，**silent 地走到 current canonical path**
- 语义漂移：跑出来的结果和 legacy 时期"看起来一样能跑"，但实际行为已经变了
- 后续调试时无法判断"这次跑挂是新路径有 bug，还是老 ckpt 不该被这条路径吃"

对比正确做法（已经存在于代码里的范本）：

- `train/training_MPL.py:_assert_no_legacy_loss_keys_in_schedule(...)`
- `train/training_MPL.py:_assert_no_removed_trainbase_stage_keys(...)`

这两个函数在 stage schedule 解析入口显式抛错，老 config 进来就 fail-fast，不会 silent 漂到新路径。**所有删除分支都应该按这个范本做。**

---

## 2) 删除分支时必须做的事（顺序不可换）

### Step 1: 标识被删分支消费的所有状态

在删之前先列出：

- config key（JSON / TOML / argparse）
- CLI flag
- ckpt state_dict 中的 key（`*_head.weight` 等）
- ckpt 顶层 metadata key（`posttrain_cfg` 子字段等）
- runtime attr（`Trainer.xxx` / `loss_fn.xxx`）
- 环境变量

### Step 2: 在每个入口加 fail-fast reject

每一类状态都要在它**第一次被读到的入口**加显式 raise。不允许"在内部某层默认走新分支"。

模板：

```python
# config / schedule / argparse 入口
if any(k in cfg for k in REMOVED_KEYS):
    hits = sorted(set(cfg) & REMOVED_KEYS)
    raise ValueError(
        f"[Removed] config 中包含已退休字段 {hits}. "
        f"该分支已在 <commit/PR ref> 移除，迁移到 <new path/'no replacement'>. "
        f"请清理 config 或回滚到 <last-supported-commit>."
    )
```

```python
# ckpt load 入口
removed = sorted(k for k in state_dict if k.startswith(REMOVED_PREFIX))
if removed:
    raise RuntimeError(
        f"[Removed] ckpt 含已退休 head 权重 {removed[:5]}{'...' if len(removed)>5 else ''}. "
        f"该 head 已在 <commit/PR ref> 移除. "
        f"请用 <last-supported-commit> 训练新 ckpt 或显式声明 --drop_removed_heads."
    )
```

错误信息**必须**包含三件事：哪个字段、哪次提交移除、迁移到哪里（或 "no replacement"）。

### Step 3: 不加任何 silent 兜底

详见 §4。

### Step 4: 在 commit / PR 描述里写 removal boundary

模板：

```
removal boundary:
- removed: <which path / config keys / ckpt keys>
- rejection enforced at: <file:function>
- migration: <new path | "no replacement">
- last-supported-commit: <sha>
```

没有这段说明的删除 PR 不算完成。

---

## 3) Fail-fast 范本（直接照抄）

| 范本 | 位置 | 触发点 |
|---|---|---|
| Stage schedule 中已退休的 loss key | `train/training_MPL.py:_assert_no_legacy_loss_keys_in_schedule` | `_resolve_freerun_stage_schedule` 解析时 |
| Stage schedule 中已退休的 trainbase key | `train/training_MPL.py:_assert_no_removed_trainbase_stage_keys` | 同上 |
| Posttrain config 中已退休的 direct-pose high-order key | `train/posttrain.py:_cfg_reject_retired_direct_pose_highorder` | `PostTrainConfig` 解析时 |

新增 fail-fast 时，命名沿用 `_assert_no_<scope>_<what>_keys` / `_<scope>_reject_<what>` 这两套既有约定，不要发明新风格。

---

## 4) 反模式清单（不允许出现）

| 反模式 | 形态 | 为什么不允许 |
|---|---|---|
| Silent fallback | `if 'old_key' in cfg: cfg['new_key'] = cfg.pop('old_key')` | 老 config 被无声改写，调试时无法区分"老 config 还是新 config" |
| Default 兜底 | `value = cfg.get('new_key', cfg.get('old_key', DEFAULT))` | 老字段被当成新字段消费，语义漂移 |
| Deprecation warning | `warnings.warn('old_key is deprecated, use new_key')` | 用户 / pipeline 长时间忽略 warning，最后等同 silent |
| Ckpt key rename in load | `if 'old_head.weight' in sd: sd['new_head.weight'] = sd.pop('old_head.weight')` | 见 §5 |
| 兼容字段 | "为了让老 ckpt 不挂，加一个 `compat_*` 字段在 model 上" | 兼容字段会被新代码继续消费，永远删不掉 |
| Try-except 吞错 | `try: load_legacy(); except Exception: load_new()` | 异常被吞，根因不可见 |
| 注释保留 | `# old_key removed, kept here for reference` | 死代码 + 误导未来的 grep |

如果你正在写以上任何一种，**停手**——这意味着边界没设清楚，先回到 §2 Step 1。

---

## 5) Ckpt load schema 边界（`train/checkpoint/load_schema.py`）

load schema 层**只允许** schema reshape，**不允许** semantic mapping。

| 允许 | 不允许 |
|---|---|
| Tensor shape 升级（如 `[N, 3] → [N, 6]` 的 padding rule 在 contract 里写明） | "老 ckpt 没有这个 head，silent 跳过" |
| 已经契约化的 key rename（`old_name → new_name`，且新旧语义**位精确**等价） | "老 head 的输出可以接到新 head 的输入" |
| Tensor dtype 升降级（fp32 ↔ fp16） | "老的 contact_meas 当成新的 pretrain_contact 用" |
| 显式 `partial_load` 时按白名单跳过指定 key | "老 ckpt 缺这个 key，用默认初始化" |

判定标准：**如果 reshape 后两份 ckpt 在所有输入下输出位精确相同，是 schema reshape；否则是 semantic mapping，不允许进 load schema 层**。

semantic mapping 的正确归宿是 fail-fast（§2 Step 2）+ 显式迁移工具（`tools/migrate_*.py`），**不是** load schema 层。

---

## 6) LLM 协作特别约束

LLM 在被要求"删除某分支"时，会倾向于：

- 自动加 `cfg.get('new', cfg.get('old', default))` 风格的兜底
- 自动加 deprecation warning
- 在 ckpt load 时 silent rename old key 到 new key
- 加 `# kept for backwards compat` 注释保留死代码

这些都属于 §4 反模式。给 LLM 的删除 prompt 必须显式包含：

> 不要为这个删除加任何兼容 shim、silent fallback、default 兜底、deprecation warning 或 ckpt key rename。
> 老 config / 老 ckpt 进入新路径必须 fail-fast raise。
> 参考 `docs/removal_policy.md` 与 `_assert_no_legacy_loss_keys_in_schedule` 范本。

PR review 第一步必须 grep diff 看是否新增了：

```
\.get\(.*\.get\(            # 嵌套兜底
warnings?\.warn\(            # warning-only
state_dict\[.*\]\s*=\s*state_dict\.pop\(  # silent rename
# .* compat                  # 注释保留
# .* legacy                  # 注释保留
```

命中任意一条，回到 §2 Step 1 重做。

---

## 7) 评审 checklist（PR 自检）

- [ ] 已列出被删分支消费的所有状态（config / ckpt / CLI / runtime attr / env）
- [ ] 每个状态在第一次被读取的入口都有显式 raise
- [ ] 错误信息包含：字段名 + 移除提交 + 迁移路径（或 "no replacement"）
- [ ] 没有新增 §4 任何反模式
- [ ] compat 层（如果改了）只做 schema reshape，没做 semantic mapping
- [ ] commit / PR 描述里写了 removal boundary 段落
- [ ] LLM 生成的 diff 已 grep 过 §6 的反模式正则

---

## 8) 关联文档

| 文档 | 关系 |
|---|---|
| `train/MODULE_BOUNDARIES.md` | 代码归属红线（写代码前自检），与本政策正交 |
| `docs/basetrain_pipeline.md` | basetrain canonical；删除 basetrain 行为前先确认是否破坏 boundary contract（§7） |
| `docs/posttrain_pipeline.md` | posttrain canonical；删除 posttrain stage 行为同理 |
| `docs/changes/2026-04-24_train_models_forward_single_file_de_maze_plan.md` | `train/models.py` 在 Phase E 前的 single-file de-maze 计划；即使只是“去迷宫化”，仍不得借整理结构之名引入 silent fallback / compat debt |
| `docs/delete/` | 历史删除审计记录（per-feature），格式可作为新删除 PR 描述模板 |

---

## 9) 这份文档不做的事

- 不审计已经存在的 silent fallback——那是单独 scope 的 audit，应另开（grep `cfg.get(...,cfg.get(`、`if old_ in cfg`、`warnings.warn(`、`state_dict[...] = state_dict.pop(` 等）。
- 不规定何时**应该**删除一个分支——那是产品 / 设计决策。
- 不替代 `train/MODULE_BOUNDARIES.md` 的代码归属规则。
