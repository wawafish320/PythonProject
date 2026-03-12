# 2026-03-07 posttrain `direct_pose` 高阶支线清理轮（side routing / SIC focus / sign gate / rank1）

Last updated: 2026-03-07

## 1) 目标

对应 `docs/Problems/active/2026-03-07_trainbase_posttrain_unused_branch_inventory.md` 第 5 节第 4 项：

- 对 `direct_pose` 高阶支线单开一轮；
- 本轮只处理 `train.posttrain` mainline runtime；
- 目标是把这组“本轮未有效进入训练目标”的分支从当前 mainline 训练 contract 中摘掉，
  同时避免误伤 checkpoint compat / archived validate lane。

本轮覆盖的高阶支线：

- `direct_pose_leg_side_routing`
- `direct_pose_leg_side_sign_gate`
- `direct_pose_leg_side_rank1`
- `direct_pose_loss_sics` / `direct_pose_loss_cycle_gte` / `direct_pose_loss_sic_mode` / `direct_pose_loss_sic_boost`

## 2) 本轮口径

统一按当前 `docs/posttrain_pipeline.md` mainline：

- `posttrain_contacts_source=pretrain_contact`
- `Stage6 -> 70a -> 70b -> 70c -> 70R -> 71 -> 72 -> lambda final`
- `direct_pose` 当前 active 的仍然只有：split / arm-split / leg head / leg gate / leg align / group norm / grad monitor
- side routing / sign gate / rank1 / SIC focus 不再视为 current mainline contract

## 3) 代码变更边界

### 已做

1. `train/posttrain.py`
   - 新增 parser guard：如果 active config/CLI 试图真正启用 side routing / sign gate / rank1 / SIC focus，runtime 直接 fail-fast。
   - 对应 config 字段在 mainline parse 后统一钉成 inert defaults，避免继续往 rollout / train-mode 主链传播。
   - direct rollout loss 中移除 SIC focus weight 重写逻辑。
   - direct rollout loss 中移除 side sign gate regularizer 支线。
   - direct train-mode 的 unfreeze / expected-trainable 前缀不再包含 routed/shared leg modules。
   - `_build_posttrain_model_from_ckpt` 中显式剥离 routed/shared leg ckpt tensors，避免 retired shell 悄悄混进当前 mainline build。

2. `train/posttrain_common.py`
   - 统一把 direct leg train-only / leg-gate-only 的 helper 收敛到标准 `direct_pose_leg_head` / `direct_pose_leg_gate_head`。

3. `docs/posttrain_pipeline.md`
   - mainline policy 增补：这组高阶 direct-pose 开关已退休；当前主线只接受 inert defaults。

### 本轮明确不做

- 不删除 `train/models.py` / `train/validate/run_freerun_cycles.py` 中为 archived checkpoint / validate lane 保留的历史建模兼容。
- 不处理 `lambda final` compat-read 字段。
- 不处理 `contacts_meas` runtime 输入与 `contact_phase_state` state core 这两个重要例外。

## 4) 为什么这样收

原因是这组分支 spread 很广，但当前 mainline 根本不用：

- parser / rollout loss / train-only unfreeze 都有入口；
- routed/shared leg head 还会牵到 checkpoint tensor 命名；
- 如果直接粗删 `train/models.py`，很容易连 archived ckpt validate 一起打断。

因此本轮采用的策略是：

- **先把 mainline contract 收紧**：禁止 active 启用；
- **再把训练期 dead branch 拆掉**：不再参与 rollout / unfreeze；
- **兼容层只保留到 ckpt drop / validate lane**，不再让它进入当前 posttrain 主训练口径。

## 5) 当前结论

完成本轮后，这组分支在当前 posttrain mainline 中的口径变为：

- `retired-from-mainline, compat-shell-only`

更具体地说：

- active configs / CLI 不允许再真正启用；
- inert default 值仍可被读取，以兼容现有 config 形状；
- 历史 ckpt 中 routed/shared tensors 会在 mainline build 时被显式剥离；
- archived reproduce / validate lane 暂不在本轮删除。

## 6) 建议的后续顺序

接下来仍按 inventory 文档原顺序：

1. 本轮完成后，主线可以继续把注意力放回 3 个重要例外：
   - `contacts_meas` runtime 输入
   - `contact_phase_state` state core
   - `lambda final` compat-read 字段
2. 如果后面要做更彻底的历史清库，再单独开一轮处理：
   - `train/models.py` 中 routed/shared leg head 结构
   - `train/validate/run_freerun_cycles.py` 的旧 ckpt 推断路径
   - archive configs / archive docs 的最终归档策略
