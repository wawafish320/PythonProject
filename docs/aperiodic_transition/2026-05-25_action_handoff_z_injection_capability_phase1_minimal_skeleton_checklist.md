> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §1/§7 under its stated read-only / zero-new-injection scope.

# Action Handoff z Injection Capability Phase-1 Minimal Skeleton Checklist

Date: 2026-05-25
Status: Design-only (no implementation)

## 0. Phase-1 boundary

- Phase-1 只定义最小代码骨架与接口契约，不写实现。
- Phase-1 明确 **不接 P6**，不改 `tools/run_action_handoff_p6_synthetic_boundary_eval.py`。
- 目标是钉死 owner/seam/contract，避免后续把注入逻辑塞回 P6 tool 或训练 entry。

## 1. 新增文件与 owner module

### 1.1 `train/validate/injection_contracts.py`（owner: validate/injection contracts）

职责：
- dataclass 契约定义。
- shape/dtype/device 校验规则。
- 注入字段解析与 fail-fast（纯 contract，不依赖 rollout 执行）。

### 1.2 `train/validate/injection_runtime.py`（owner: validate/injection runtime）

职责：
- 从 target npz 装载注入 payload。
- 在单步 rollout seam 上应用注入（仅 runtime helper）。
- 产出注入应用 metadata（applied_log）。

### 1.3 `train/validate/injection_windows.py`（owner: validate/injection windowing）

职责：
- `entry_window` / `post_inject_recovery` 的窗口切片与元数据计算。
- 从 per-step metrics 聚合 `metric_summary`（不负责跑模型）。

### 1.4 `train/validate/run_freerun_injection_eval.py`（owner: standalone injection evaluator entry）

职责：
- 独立 evaluator CLI（只服务 injection capability）。
- 调用 `run_freerun_cycles` 可复用路径或共享 helper，输出 `freerun + paired_delta + run.log`。

## 2. 公开函数签名（草案）

```python
# train/validate/injection_contracts.py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence
import torch

InjectField = Literal["rootvel", "rot6d", "angvel"]

@dataclass(frozen=True)
class InjectionRequest:
    source_npz: Path
    inject_at_step: int
    inject_from_step: int
    inject_fields: tuple[InjectField, ...]
    inject_label: str
    inject_pose_hist_full: bool
    length: int  # Phase-1 fixed to 1; >1 fail-fast

@dataclass(frozen=True)
class InjectionTensorSpec:
    rot6d_shape: tuple[int, int]      # (1, J*6)
    rootvel_shape: tuple[int, int]    # (1, Vr)
    angvel_shape: tuple[int, int]     # (1, J*3)
    dtype: torch.dtype
    device: torch.device

@dataclass
class InjectionPayload:
    rot6d_raw: Optional[torch.Tensor]
    rootvel_raw: Optional[torch.Tensor]
    angvel_raw: Optional[torch.Tensor]
    source_frame_index: int
    spec: InjectionTensorSpec

@dataclass
class InjectionApplyRecord:
    step: int
    step_in_cycle: int
    fields_applied: list[dict]
    pose_hist_tail_rewrite: Optional[dict]

def parse_inject_fields(raw: str) -> tuple[InjectField, ...]: ...
def validate_injection_request(request: InjectionRequest) -> None: ...
def validate_payload_against_slices(
    payload: InjectionPayload,
    *,
    rot6d_x_slice: slice,
    angvel_x_slice: Optional[slice],
    rootvel_x_slice: slice,
) -> None: ...
```

```python
# train/validate/injection_runtime.py
from pathlib import Path
from typing import Any, Optional
import torch
from train.validate.injection_contracts import InjectionRequest, InjectionPayload, InjectionApplyRecord

def load_injection_payload_from_npz(
    *,
    request: InjectionRequest,
    device: torch.device,
    dtype: torch.dtype,
    state_layout_json: str,
) -> InjectionPayload: ...

def apply_injection_to_y_used_raw(
    *,
    y_used_raw: torch.Tensor,
    payload: InjectionPayload,
    fields: tuple[str, ...],
    rot6d_y_slice: slice,
    output_rootvel_y_slice: Optional[slice],  # 如果 y 含 rootvel，否则为 None
) -> tuple[torch.Tensor, list[dict]]: ...

def maybe_rewrite_pose_hist_tail(
    *,
    pose_hist_enabled: bool,
    pose_hist_state: Any,
    payload: InjectionPayload,
    inject_pose_hist_full: bool,
    pose_hist_stride: int,
) -> Optional[dict]: ...
```

```python
# train/validate/injection_windows.py
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass(frozen=True)
class WindowSpec:
    entry_window_pre_k: int
    entry_window_post_k: int
    recovery_window_k: int

def compute_window_bounds(
    *,
    inject_at_step: int,
    total_steps: int,
    spec: WindowSpec,
) -> dict[str, dict]: ...

def summarize_window_metrics(
    *,
    per_step_metrics: List[Dict[str, Any]],
    bounds: dict[str, dict],
    required_metrics: list[str],
) -> dict[str, Any]: ...
```

```python
# train/validate/run_freerun_injection_eval.py
import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace: ...
def run_trial(args: argparse.Namespace) -> Path: ...
def main() -> None: ...
```

## 3. I/O dataclass 契约

### 3.1 输入契约（Phase-1）

- `InjectionRequest.source_npz`: 目标 clip npz 路径，必须存在。
- `InjectionRequest.inject_at_step`: 全局 rollout step（0-based）。
- `InjectionRequest.inject_from_step`: source clip frame index（0-based）。
- `InjectionRequest.inject_fields`: `{rootvel, rot6d, angvel}` 非空子集。
- `InjectionRequest.length`: Phase-1 必须等于 `1`。

### 3.2 输出契约（Phase-1）

- `InjectionPayload`: 仅包含本次注入所需 raw tensor，不做多步缓存。
- `InjectionApplyRecord`: 每次应用一条记录，至少含：
  - `step`, `step_in_cycle`
  - `fields_applied`（field + slice + applied）
  - `pose_hist_tail_rewrite`（requested/applied/mode/stride/frame_indices）

## 4. Tensor shape / dtype / device contract

结合当前数据布局（`state_layout_json`）：
- `RootVelocity: [3,5)`，维度 `Vr=2`
- `BoneRotations6D: [5,281)`，维度 `J*6=276`
- `BoneAngularVelocities: [281,419)`，维度 `J*3=138`

运行时强约束：
- `rot6d_raw.shape == (1, 276)`，`dtype=torch.float32`
- `rootvel_raw.shape == (1, 2)`，`dtype=torch.float32`
- `angvel_raw.shape == (1, 138)`，`dtype=torch.float32`
- device 必须与 rollout `y_used_raw.device` 一致；不一致时显式 `.to(device, dtype)`，失败则 fatal。
- 所有注入 tensor 必须 finite；出现 `NaN/Inf` 直接 fatal。

## 5. 注入 seam 的最小 patch 点（仅设计，不改代码）

seam 位置（现有主循环）：
- `y_used_raw` 产生后：`train/validate/run_freerun_cycles.py:6650`
- `apply_free_carry_raw` 前：`train/validate/run_freerun_cycles.py:6668`

Phase-1 最小 seam 定义：
1. 仅在 `step == inject_at_step` 时触发一次注入（`length=1`）。
2. 先 override `y_used_raw` 指定字段，再进入 `apply_free_carry_raw`。
3. 如 `inject_pose_hist_full=true`，在 pose history 更新段同步写入 tail（现有 pose_hist 更新段在 `train/validate/run_freerun_cycles.py:6718` 一带）。

## 6. 测试骨架文件名

### 6.1 Unit tests

- `tests/train/test_injection_contracts.py`
- `tests/train/test_injection_runtime_apply.py`
- `tests/train/test_injection_windows.py`

### 6.2 Smoke tests（capability-only）

- `tests/train/test_injection_eval_smoke_normal.py`
- `tests/train/test_injection_eval_smoke_weak.py`

说明：
- smoke 只验证 capability 产物完整性（`freerun json + paired_delta + run.log`）。
- 不读取/不判断 P6 status，不接 P6 tool。

## 7. 不允许改动文件（Phase-1）

- `tools/run_action_handoff_p6_synthetic_boundary_eval.py`
- `train/training_MPL.py`
- `train/posttrain.py`
- `docs/aperiodic_transition/*P6*runner*`（除非单独文档同步）

## 8. Phase-1 明确不接 P6

- 不新增任何 P6 CLI 参数透传。
- 不在 P6 summary 写 `executed_runner_injected_smoke_v1_*`。
- 仅产出 injection capability 自身 artifact 与测试结果，后续由单独任务决定如何接 P6。
