# 2026-03-10 `train/history.py` pose-history 共享 API 清单

## 目标

把当前分散在

- `train/training_MPL.py`
- `train/posttrain.py`

里的 pose-history 初始化/读取/滚动更新逻辑，收口到 `train/history.py`。

这轮先做 **shared API 设计清单**，不直接扩展到 scheduled sampling、rollout policy、train/posttrain 专属状态机。

---

## 2026-03-11 执行进展

- Phase 1 已完成：共享 pose-history contract 已落到 `train/history.py`。
- Phase 2 已完成：`train/training_MPL.py` 已切到共享 pose-history helper。
- Phase 3 已完成：`train/posttrain.py` 已切到共享 pose-history helper。
- Phase 4 已完成：`train/validate/run_freerun_cycles.py` 已切到共享 pose-history state/helper，validation / ablation 侧完成收口。
- 已新增并导出：
  - `PoseHistState`
  - `pose_hist_transform_vec`
  - `pose_hist_inverse_vec`
  - `init_pose_hist_state`
  - `resolve_pose_hist_input`
  - `advance_pose_hist_state`
  - `advance_pose_hist_state_with_tail`
- 落地口径采用本文后半段的 Phase 1 具体签名：
  - `init_pose_hist_state(...)` 不再显式接收 `has_time_dim`
  - `resolve_pose_hist_input(...)` 只接收 `state / pose_hist_seq / idx`
  - time-vs-static seq 由 `pose_hist_seq.dim()` 推断
- 已补最小回归到 `tests/train/test_pose_history_helpers.py`，覆盖：
  - transform/inverse round-trip
  - init from time seq
  - init from static seq
  - fallback from `y_prev_raw[..., rot_slice]`
  - resolve buffer-vs-seq
  - advance roll+tail write
  - `force_disable` / params missing / invalid dim / missing `rot_slice` 边界
- 已补 Phase 2 接入回归到 `tests/train/test_training_mpl_pose_history_phase2.py`，覆盖：
  - `_prepare_pose_hist_state(...)` 与共享 helper 初始化一致
  - `_resolve_rollout_step_inputs(...)` 的 buffer-vs-seq 选择一致
  - `_update_rollout_carry_state(...)` 会基于最终 carry raw pose 推进 history buffer
- 已补 Phase 3 接入回归到 `tests/train/test_posttrain_pose_history_phase3.py`，覆盖：
  - `_lambda_rollout_prepare_context(...)` 的 offset init parity
  - `_rollout_step_common(...)` 的 buffer-vs-seq 选择一致
  - `_apply_rollout_carry_state(...)` 的 shared advance 行为一致
- 已补 Phase 4 接入回归到 `tests/train/test_run_freerun_cycles_pose_history_phase4.py`，覆盖：
  - eval 侧 step-specific init 与共享 helper 一致
  - eval 侧 zero fallback 仍保持 buffer contract 可用
  - `pose_hist_source = buffer / seq / zero` 三种输入选择一致
- 本地已执行验证：
  - `python - <<'PY' ... from train.history import ...`：通过
  - `python -m unittest tests.train.test_pose_history_helpers tests.train.test_training_mpl_pose_history_phase2 tests.train.test_posttrain_pose_history_phase3 tests.train.test_run_freerun_cycles_pose_history_phase4`：通过
- `train/training_MPL.py` 当前已完成的接入点：
  - 删除本地 `PoseHistState`，改为从 `train.history` import
  - `_prepare_pose_hist_state(...)` 转调 `init_pose_hist_state(...)`
  - `_resolve_rollout_step_inputs(...)` 转调 `resolve_pose_hist_input(...)`
  - `_update_rollout_carry_state(...)` 转调 `advance_pose_hist_state(...)`
  - 删除本地 `_pose_hist_transform_vec(...)` / `_pose_hist_inverse_vec(...)`
- `train/posttrain.py` 当前已完成的接入点：
  - 删除 `_init_rollout_pose_hist_state(...)`
  - 改为统一维护 `state["pose_hist_state"]`
  - `_rollout_step_common(...)` 转调 `resolve_pose_hist_input(...)`
  - `_apply_rollout_carry_state(...)` 转调 `advance_pose_hist_state(...)`
  - `_lambda_rollout_prepare_context(...)` 转调 `init_pose_hist_state(...)`
- `train/validate/run_freerun_cycles.py` 当前已完成的接入点：
  - 新增 eval 侧 `_init_eval_pose_hist_state(...)` / `_resolve_eval_pose_hist_input(...)`，统一转调共享 helper
  - 初始化 / cycle-start reset / sync reset 改为复用 `init_pose_hist_state(...)`
  - `pose_hist_source = buffer / seq / zero` 改为复用 `resolve_pose_hist_input(...)` 做输入解析
  - `pose_hist_update_source = pred / gt / zero / freeze` 的 buffer 推进改为复用 `advance_pose_hist_state_with_tail(...)`
  - donor / hybrid boundary carry 改为统一维护 `donor_state["pose_hist_state"]`
- 当前剩余工作：
  - 本 checklist 4 个 phase 已全部完成
  - 若后续还要扩展 validation 侧 pose-history policy，建议单开新 checklist，不再回滚 shared contract

---

## 当前重复点

当前至少有下面两块逻辑已经明显重复：

- `train/training_MPL.py::_prepare_pose_hist_state`
- `train/posttrain.py::_init_rollout_pose_hist_state`

并且它们后续消费的 buffer 更新逻辑也高度同源：

- `train/training_MPL.py::_update_rollout_carry_state` 内的 pose-history buffer 滚动更新片段
- `train/posttrain.py::_apply_rollout_carry_state` 内的 pose-history buffer 滚动更新片段

因此，共享 API 不应只覆盖 init，还应至少覆盖：

1. state 类型
2. init
3. rollout 时取当前 pose-history 输入
4. rollout 后推进 buffer

---

## 为什么放到 `train/history.py`

`train/history.py` 已经承载 `AdaptiveHistoryModule`，语义上就是仓内的 history-related 模块。

相较之下，这部分逻辑放到：

- `train/training_MPL.py`：范围太窄，只适合 trainbase method
- `train/posttrain_common.py`：更像 posttrain helper，不够中立

所以 `train/history.py` 是当前更自然的共享落点。

---

## 推荐共享 API

### 1. `PoseHistState`

把当前 `train/training_MPL.py` 里的本地 dataclass 提升为共享类型，并作为 posttrain 的统一状态对象，替代散落的 dict 字段。

```python
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PoseHistState:
    enabled: bool
    length: int
    dim: int
    stride: int
    scales: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    buffer_norm: Optional[torch.Tensor] = None
    buffer_raw: Optional[torch.Tensor] = None
```

用途：

- 统一 `training_MPL` 和 `posttrain` 的 pose-history state contract
- 不再在 `posttrain.py` 里散落维护
  `pose_hist_enabled` / `pose_hist_stride` / `pose_hist_buffer_raw` / `pose_hist_buffer_norm`

---

### 2. `pose_hist_transform_vec`

统一对 flattened pose-history raw vector 做 normalizer transform。

```python
def pose_hist_transform_vec(
    raw_flat: torch.Tensor,
    scales: Optional[torch.Tensor],
    mu: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor: ...
```

对应来源：

- `train/training_MPL.py::_pose_hist_transform_vec`

---

### 3. `pose_hist_inverse_vec`

统一从 normalized pose-history vector 还原 raw vector。

```python
def pose_hist_inverse_vec(
    norm_flat: torch.Tensor,
    scales: Optional[torch.Tensor],
    mu: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor: ...
```

对应来源：

- `train/training_MPL.py::_pose_hist_inverse_vec`

---

### 4. `init_pose_hist_state`

这是本轮最核心的抽取点，合并 trainbase / posttrain 两套初始化逻辑。

```python
from typing import Callable, Optional


def init_pose_hist_state(
    *,
    ref_tensor: torch.Tensor,
    pose_hist_seq: Optional[torch.Tensor],
    y_prev_raw: Optional[torch.Tensor],
    rot_slice: Optional[slice],
    pose_hist_len: int,
    pose_hist_dim: int,
    params_fn: Callable[
        [torch.Tensor],
        tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
    ],
    offset: int = 0,
    force_disable: bool = False,
    require_rot_slice_for_fallback: bool = False,
) -> PoseHistState: ...
```

合并来源：

- `train/training_MPL.py::_prepare_pose_hist_state`
- `train/posttrain.py::_init_rollout_pose_hist_state`

设计要点：

- 不直接依赖 `Trainer`
- 用 `params_fn(ref_tensor)` 注入 `scales / mu / std`
- 兼容两种初始化来源：
  - 直接用 `pose_hist_seq`
  - 当 `pose_hist_seq` 缺失时，从 `y_prev_raw[..., rot_slice]` fallback 构造
- 保留 `offset`，兼容 posttrain 那种从指定时刻取 buffer 的做法
- 保留 `force_disable`，兼容 `training_MPL.py` 里的 `force_pose_hist_seq` 关闭逻辑

---

### 5. `resolve_pose_hist_input`

统一 rollout 单步里“优先使用 buffer_norm，否则退回原始 pose_hist_seq”的逻辑。

```python
def resolve_pose_hist_input(
    *,
    state: PoseHistState,
    pose_hist_seq: Optional[torch.Tensor],
    idx: int,
) -> Optional[torch.Tensor]: ...
```

合并来源：

- `train/training_MPL.py::_resolve_rollout_step_inputs` 中 `pose_history_t` 选择片段
- `train/posttrain.py::_rollout_step_common` 中 `pose_hist_t` 选择片段

---

### 6. `advance_pose_hist_state`

统一 rollout 之后 pose-history buffer 的滚动更新。

```python
def advance_pose_hist_state(
    state: PoseHistState,
    *,
    y_next_raw: torch.Tensor,
    rot_slice: Optional[slice],
) -> PoseHistState: ...
```

合并来源：

- `train/training_MPL.py::_update_rollout_carry_state` 内 pose-history buffer 更新片段
- `train/posttrain.py::_apply_rollout_carry_state` 内 pose-history buffer 更新片段

设计要点：

- 输入 old state，返回 new state
- 保持 pure helper 风格，避免函数内部依赖 train/posttrain 其余 runtime 状态
- 内部调用共享的 `pose_hist_transform_vec`

---

## 建议保留在调用侧、不放进 `train/history.py` 的部分

以下内容暂时不建议一起搬：

- `Trainer._pose_hist_params`
  - 这是 runtime config / normalizer state 的读取接口，保留在 `Trainer` 更自然
  - 共享层通过 `params_fn` 注入即可
- scheduled sampling 的 GT/raw blend
- freerun `motion_raw` / `motion` carry 更新
- cond reprojection
- train/posttrain 各自的 rollout state machine

换句话说，`train/history.py` 只负责：

- pose-history state 的创建
- pose-history input 的解析
- pose-history buffer 的推进

不负责整个 rollout policy。

---

## 调用侧迁移建议

### `train/training_MPL.py`

建议替换为：

- 删除本地 `PoseHistState` 定义，改为从 `train.history` import
- `self._prepare_pose_hist_state(...)` 改为调用 `init_pose_hist_state(...)`
- rollout 单步里的 pose-history 选择改为 `resolve_pose_hist_input(...)`
- carry 更新里的 pose-history buffer 滚动改为 `advance_pose_hist_state(...)`

### `train/posttrain.py`

建议替换为：

- 删除 `_init_rollout_pose_hist_state(...)`
- 不再散落维护 `pose_hist_enabled` / `pose_hist_stride` / `pose_hist_buffer_*`
- 改为统一维护 `state["pose_hist_state"]`
- rollout 单步里通过 `resolve_pose_hist_input(...)` 取 `pose_hist_t`
- carry 更新里通过 `advance_pose_hist_state(...)` 推进 state

---

## 推荐 `__all__`

```python
__all__ = [
    "AdaptiveHistoryModule",
    "PoseHistState",
    "pose_hist_transform_vec",
    "pose_hist_inverse_vec",
    "init_pose_hist_state",
    "resolve_pose_hist_input",
    "advance_pose_hist_state",
    "advance_pose_hist_state_with_tail",
]
```

---

## 推荐实施顺序（至少 3 个 phase）

不建议一次性同时改 `train/training_MPL.py` 和 `train/posttrain.py`。
更稳妥的做法是先把共享 contract 固定，再分调用侧逐步迁移。

### Phase 1: 先落共享 helper，不改外部语义（已完成，2026-03-11）

目标：

- 只在 `train/history.py` 新增共享类型和 helper
- 先固定 shared API contract
- 尽量做到行为不变

范围：

- 新增：
  - `PoseHistState`
  - `pose_hist_transform_vec`
  - `pose_hist_inverse_vec`
  - `init_pose_hist_state`
  - `resolve_pose_hist_input`
  - `advance_pose_hist_state`
- `train/training_MPL.py` / `train/posttrain.py` 暂时可以先不删旧入口
- 允许旧入口内部转调新 helper，但不要顺手改 rollout state machine

本 phase 不做：

- 不改 scheduled sampling
- 不改 cond reprojection
- 不改 `motion_raw` / `motion` carry
- 不碰 `train/validate/run_freerun_cycles.py`

退出条件：

- `train/history.py` 可独立 import
- 新 helper 的最小单测通过：
  - init from `pose_hist_seq`
  - init from `y_prev_raw[..., rot_slice]` fallback
  - resolve buffer-vs-seq
  - advance 后 buffer 末帧写入正确

### Phase 1 当前进度（2026-03-11，已完成）

- 代码已落地到 `train/history.py`，未改 `train/training_MPL.py` / `train/posttrain.py` 外部语义。
- Phase 1 约定的 6 个共享 helper 已全部补齐，并已纳入 `__all__`。
- 最小单测已补到 `tests/train/test_pose_history_helpers.py`，覆盖主路径与 4 个边界 case。
- Phase 1 退出条件已满足：
  - `train/history.py` 可独立 import
  - helper 最小回归通过
- 当前剩余工作收敛为：
  - Phase 2：接 `train/training_MPL.py`
  - Phase 3：接 `train/posttrain.py`

### Phase 2: 接入 `train/training_MPL.py`

目标：

- 让 trainbase 侧只消费共享 pose-history API
- 不改变当前 trainbase rollout 策略

范围：

- 删除本地 `PoseHistState` 定义，改为从 `train.history` import
- `self._prepare_pose_hist_state(...)` 改为调用 `init_pose_hist_state(...)`
- rollout 单步的 pose-history 选择改为 `resolve_pose_hist_input(...)`
- carry 更新里的 buffer 推进改为 `advance_pose_hist_state(...)`

本 phase 不做：

- 不改 `Trainer._pose_hist_params`
- 不改 scheduled sampling 的 GT/raw blend 逻辑
- 不改 `plan_z` / `phase_z` / `cond_raw_for_env` 等其他 rollout 状态

退出条件：

- trainbase rollout smoke test 通过
- old/new 行为 parity：
  - `pose_history_t` shape 一致
  - `buffer_norm` / `buffer_raw` 首步初始化一致
  - 一步 advance 后结果一致

### Phase 2 当前进度（2026-03-11，已完成）

- `train/training_MPL.py` 已只消费 `train.history` 中的共享 pose-history API。
- 已完成的替换点：
  - 删除本地 `PoseHistState`
  - `_prepare_pose_hist_state(...) -> init_pose_hist_state(...)`
  - `_resolve_rollout_step_inputs(...) -> resolve_pose_hist_input(...)`
  - `_update_rollout_carry_state(...) -> advance_pose_hist_state(...)`
  - 删除本地 `_pose_hist_transform_vec(...)` / `_pose_hist_inverse_vec(...)`
- 已补回归到 `tests/train/test_training_mpl_pose_history_phase2.py`，覆盖本 phase 约定的 3 个 parity 点。
- 本地验证：
  - `python -m unittest tests.train.test_pose_history_helpers tests.train.test_training_mpl_pose_history_phase2`：通过
- 当前剩余工作收敛为：
  - Phase 3：接 `train/posttrain.py`

### Phase 3: 接入 `train/posttrain.py`

目标：

- 用共享 `PoseHistState` 替换 posttrain 里散落的 pose-history dict 字段
- 保持 posttrain rollout 行为不变

范围：

- 删除 `_init_rollout_pose_hist_state(...)`
- 改为统一维护 `state["pose_hist_state"]`
- `_rollout_step_common(...)` 通过 `resolve_pose_hist_input(...)` 取 `pose_hist_t`
- `_apply_rollout_carry_state(...)` 通过 `advance_pose_hist_state(...)` 推进 buffer

本 phase 重点回归：

- `offset` 初始化是否与当前 posttrain 一致
- boundary / multi-cycle 情况下初始 pose-history 是否不回归
- posttrain contacts 输入是否仍然拿到相同的 `pose_hist_t`

退出条件：

- posttrain rollout smoke test 通过
- 随机 offset 场景下 init parity 通过
- 多步 rollout 后 `pose_hist_state.buffer_*` 与旧实现一致

### Phase 3 当前进度（2026-03-11，已完成）

- `train/posttrain.py` 已切到 `train.history` 中的共享 pose-history API。
- 已完成的替换点：
  - 删除 `_init_rollout_pose_hist_state(...)`
  - 统一改为维护 `state["pose_hist_state"]`
  - `_rollout_step_common(...) -> resolve_pose_hist_input(...)`
  - `_apply_rollout_carry_state(...) -> advance_pose_hist_state(...)`
  - `_lambda_rollout_prepare_context(...) -> init_pose_hist_state(...)`
- 已补回归到 `tests/train/test_posttrain_pose_history_phase3.py`，覆盖本 phase 约定的 3 个 parity 点。
- 本地验证：
  - `python -m unittest tests.train.test_pose_history_helpers tests.train.test_training_mpl_pose_history_phase2 tests.train.test_posttrain_pose_history_phase3`：通过
- 当前剩余工作收敛为：
  - Phase 4：验证 / ablation 侧收口

### Phase 4: 验证 / ablation 侧单独收口（已完成，2026-03-11）

`train/validate/run_freerun_cycles.py` 里也有明显同源逻辑，但它比 train/posttrain 多出：

- `pose_hist_source = buffer / seq / zero`
- `pose_hist_update_source = pred / gt / zero / freeze`
- donor / hybrid update

因此不建议和前三个 phase 混做。
实际落地时，shared contract 继续放在 `train/history.py`，但 eval policy 仍保留在 validation 侧：

- eval 初始化 / cycle-start reseed 复用 `init_pose_hist_state(...)`
- eval buffer-vs-seq-vs-zero 选择复用 `resolve_pose_hist_input(...)`
- eval `pred / gt / zero` buffer 推进复用 `advance_pose_hist_state_with_tail(...)`
- donor / hybrid boundary carry 统一改为维护 `PoseHistState`

### Phase 4 当前状态（2026-03-11，已完成）

- `train/validate/run_freerun_cycles.py` 已完成接入，不再依赖已删除的 `Trainer._pose_hist_transform_vec(...)` / `_pose_hist_inverse_vec(...)`。
- eval 侧新增局部 helper：
  - `_init_eval_pose_hist_state(...)`
  - `_resolve_eval_pose_hist_input(...)`
- 已补回归到 `tests/train/test_run_freerun_cycles_pose_history_phase4.py`，覆盖：
  - step-specific init parity
  - zero fallback contract
  - `buffer / seq / zero` source 选择
- 本地验证：
  - `python -m unittest tests.train.test_pose_history_helpers tests.train.test_training_mpl_pose_history_phase2 tests.train.test_posttrain_pose_history_phase3 tests.train.test_run_freerun_cycles_pose_history_phase4`：通过

---

## API contract 补充约束

为了避免 Phase 2/3 接入时语义漂移，建议先把下面几条写死：

1. `pose_hist_dim % pose_hist_len == 0`
   - 否则视为 invalid config，不能只依赖 `stride > 0`

2. `advance_pose_hist_state(..., y_next_raw=...)` 的输入语义
   - 这里的 `y_next_raw` 指“本步真正写回 carry 的 raw pose”
   - 不是任意中间预测值
   - 对 trainbase 来说，应与 scheduled sampling blend 之后写入 state 的那个 raw pose 一致

3. fail-soft 规则
   - `params_fn(ref_tensor)` 返回 `None`：默认 disable，而不是 raise
   - `pose_hist_seq` 缺失但 `rot_slice` 可用：允许 fallback 初始化
   - `require_rot_slice_for_fallback=True` 且 `rot_slice` 缺失：允许显式 raise

4. `resolve_pose_hist_input(...)` 的职责边界
   - 只负责“优先 buffer，否则退回 seq”
   - 不负责 `unsqueeze`
   - 不负责 contacts / cond / angvel 的任何选择逻辑

---

## Phase 1 具体函数签名（建议稿）

下面这版签名是给 Phase 1 直接落地用的，目标是接口尽量收敛、调用侧依赖尽量少。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, TypeAlias

import torch


PoseHistParamsFn: TypeAlias = Callable[
    [torch.Tensor],
    tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
]


@dataclass
class PoseHistState:
    enabled: bool
    length: int
    dim: int
    stride: int
    scales: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    buffer_norm: Optional[torch.Tensor] = None
    buffer_raw: Optional[torch.Tensor] = None


def pose_hist_transform_vec(
    raw_flat: torch.Tensor,
    scales: Optional[torch.Tensor],
    mu: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor: ...


def pose_hist_inverse_vec(
    norm_flat: torch.Tensor,
    scales: Optional[torch.Tensor],
    mu: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor: ...


def init_pose_hist_state(
    *,
    ref_tensor: torch.Tensor,
    pose_hist_seq: Optional[torch.Tensor],
    y_prev_raw: Optional[torch.Tensor],
    rot_slice: Optional[slice],
    pose_hist_len: int,
    pose_hist_dim: int,
    params_fn: PoseHistParamsFn,
    offset: int = 0,
    force_disable: bool = False,
    require_rot_slice_for_fallback: bool = False,
) -> PoseHistState: ...


def resolve_pose_hist_input(
    *,
    state: PoseHistState,
    pose_hist_seq: Optional[torch.Tensor],
    idx: int,
) -> Optional[torch.Tensor]: ...


def advance_pose_hist_state(
    state: PoseHistState,
    *,
    y_next_raw: torch.Tensor,
    rot_slice: Optional[slice],
) -> PoseHistState: ...
```

说明：

- `init_pose_hist_state(...)` 不再显式接收 `has_time_dim`
  - 直接从 `pose_hist_seq.dim()` 推断 `[B, T, D]` vs `[B, D]`
- `resolve_pose_hist_input(...)` 只管“buffer 优先，否则回退到 seq”
- `advance_pose_hist_state(...)` 只管 buffer 滚动，不掺入 rollout 其他 state

---

## Phase 1 伪代码

### 1. `pose_hist_transform_vec`

```python
def pose_hist_transform_vec(raw_flat, scales, mu, std):
    if scales is None or raw_flat.numel() == 0:
        return raw_flat

    norm = VectorTanhNormalizerTorch(scales, mu, std)
    norm = norm.to(device=raw_flat.device, dtype=raw_flat.dtype)
    return norm(raw_flat)
```

### 2. `pose_hist_inverse_vec`

```python
def pose_hist_inverse_vec(norm_flat, scales, mu, std):
    if scales is None or norm_flat.numel() == 0:
        return norm_flat

    norm = VectorTanhNormalizerTorch(scales, mu, std)
    norm = norm.to(device=norm_flat.device, dtype=norm_flat.dtype)
    return norm.inverse(norm_flat)
```

### 3. `init_pose_hist_state`

```python
def init_pose_hist_state(...):
    if pose_hist_len <= 0 or pose_hist_dim <= 0:
        return PoseHistState(enabled=False, length=pose_hist_len, dim=pose_hist_dim, stride=0)

    if pose_hist_dim % pose_hist_len != 0:
        raise ValueError("pose_hist_dim must be divisible by pose_hist_len")

    stride = pose_hist_dim // pose_hist_len
    state = PoseHistState(
        enabled=not force_disable,
        length=pose_hist_len,
        dim=pose_hist_dim,
        stride=stride,
    )
    if not state.enabled:
        return state

    scales, mu, std = params_fn(ref_tensor)
    if scales is None:
        state.enabled = False
        return state
    state.scales = scales
    state.mu = mu
    state.std = std

    initial_norm = None
    if torch.is_tensor(pose_hist_seq) and pose_hist_seq.numel() > 0:
        seq = pose_hist_seq.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        if seq.dim() == 3:
            idx = min(max(int(offset), 0), int(seq.shape[1]) - 1)
            initial_norm = seq[:, idx]
        else:
            initial_norm = seq

    with torch.no_grad():
        if initial_norm is not None:
            state.buffer_norm = initial_norm
            state.buffer_raw = pose_hist_inverse_vec(initial_norm, scales, mu, std)
            return state

        if require_rot_slice_for_fallback and not isinstance(rot_slice, slice):
            raise RuntimeError("pose_hist enabled but rot slice missing for fallback init")

        if (not torch.is_tensor(y_prev_raw)) or (not isinstance(rot_slice, slice)):
            state.enabled = False
            return state

        base_rot = y_prev_raw[..., rot_slice]
        state.buffer_raw = (
            base_rot.unsqueeze(1)
            .repeat(1, pose_hist_len, 1)
            .reshape(base_rot.shape[0], pose_hist_dim)
        )
        state.buffer_norm = pose_hist_transform_vec(state.buffer_raw, scales, mu, std)
        return state
```

### 4. `resolve_pose_hist_input`

```python
def resolve_pose_hist_input(*, state, pose_hist_seq, idx):
    if state.enabled and state.buffer_norm is not None:
        return state.buffer_norm

    if (not torch.is_tensor(pose_hist_seq)) or pose_hist_seq.numel() == 0:
        return None

    if pose_hist_seq.dim() == 3:
        step_idx = min(max(int(idx), 0), int(pose_hist_seq.shape[1]) - 1)
        return pose_hist_seq[:, step_idx]

    return pose_hist_seq
```

### 5. `advance_pose_hist_state`

```python
def advance_pose_hist_state(state, *, y_next_raw, rot_slice):
    if (
        (not state.enabled)
        or state.stride <= 0
        or state.buffer_raw is None
        or (not isinstance(rot_slice, slice))
    ):
        return state

    with torch.no_grad():
        next_buffer_raw = torch.roll(state.buffer_raw, shifts=-state.stride, dims=-1)
        next_buffer_raw[..., -state.stride:] = y_next_raw[..., rot_slice]
        next_buffer_norm = pose_hist_transform_vec(
            next_buffer_raw,
            state.scales,
            state.mu,
            state.std,
        )

    return PoseHistState(
        enabled=state.enabled,
        length=state.length,
        dim=state.dim,
        stride=state.stride,
        scales=state.scales,
        mu=state.mu,
        std=state.std,
        buffer_norm=next_buffer_norm,
        buffer_raw=next_buffer_raw,
    )
```

---

## Phase 1 最小测试清单

建议先补一个轻量测试文件，例如：

- `tests/train/test_pose_history_helpers.py`

最小必测项建议 6 个：

### 1. transform / inverse round-trip

目标：

- 验证 `pose_hist_transform_vec` 和 `pose_hist_inverse_vec` 与 `VectorTanhNormalizerTorch` 语义一致

断言：

- `inverse(transform(x)) ~= x`
- shape 不变
- device / dtype 不漂移

### 2. init: 从 time-seq 正常初始化

输入：

- `pose_hist_seq.shape == [B, T, D]`
- `offset = k`

断言：

- `state.enabled is True`
- `state.buffer_norm == pose_hist_seq[:, k]`
- `state.buffer_raw.shape == [B, D]`
- `state.length / dim / stride` 正确

### 3. init: 从 static seq 正常初始化

输入：

- `pose_hist_seq.shape == [B, D]`

断言：

- `state.buffer_norm == pose_hist_seq`
- 不依赖 `offset`

### 4. init: `pose_hist_seq` 缺失时 fallback 到 `y_prev_raw[..., rot_slice]`

输入：

- `pose_hist_seq = None`
- `y_prev_raw` 和合法 `rot_slice`

断言：

- `state.enabled is True`
- `state.buffer_raw` 等于把 `base_rot` 重复 `pose_hist_len` 次后的 flatten
- `state.buffer_norm` 等于对 `buffer_raw` 做共享 transform 的结果

### 5. resolve: buffer 优先，否则回退 seq

分两组：

- `state.buffer_norm is not None`
- `state.buffer_norm is None`

断言：

- 第一组返回 `buffer_norm`
- 第二组对 `[B, T, D]` 返回 `pose_hist_seq[:, idx]`
- 第二组对 `[B, D]` 返回 `pose_hist_seq`

### 6. advance: 正确 roll 并写入尾帧

输入：

- 构造一个已知 `buffer_raw`
- 构造 `y_next_raw[..., rot_slice]`

断言：

- `next_state.buffer_raw[..., :-stride] == old_state.buffer_raw[..., stride:]`
- `next_state.buffer_raw[..., -stride:] == y_next_raw[..., rot_slice]`
- `next_state.buffer_norm` 等于对 `next_state.buffer_raw` 调用共享 transform 的结果

---

## Phase 1 建议补的边界测试（可选但很值）

如果你想把 helper 层一次钉得更牢，建议再补下面 4 个边界 case：

1. `force_disable=True`
   - 应直接返回 disabled state

2. `params_fn(ref_tensor)` 返回 `(None, None, None)`
   - 应 fail-soft，返回 disabled state

3. `pose_hist_dim % pose_hist_len != 0`
   - 应抛 `ValueError`

4. `require_rot_slice_for_fallback=True` 且 fallback 时 `rot_slice is None`
   - 应显式抛错，而不是静默 disable

---

## 本轮结论

如果只考虑 `train/training_MPL.py` 和 `train/posttrain.py` 之间最值得先抽的 history 共享层，那么
`train/history.py` 里的最小可行 API 仍然是：

1. `PoseHistState`
2. `pose_hist_transform_vec`
3. `pose_hist_inverse_vec`
4. `init_pose_hist_state`
5. `resolve_pose_hist_input`
6. `advance_pose_hist_state`

这是一个比较稳妥的切分边界：

- 足够覆盖当前重复点
- 不会把整个 rollout runtime 一次性拖进 `history.py`
- 也不会让 `history.py` 反向依赖 `Trainer` 或 posttrain runtime 细节

但实施上应明确拆成至少 3 个 phase：

1. 先落共享 helper 和 contract
2. 再迁 `train/training_MPL.py`
3. 最后迁 `train/posttrain.py`

必要时把 validation / ablation 侧作为单独后续 phase 处理，避免一次改动面过大。
