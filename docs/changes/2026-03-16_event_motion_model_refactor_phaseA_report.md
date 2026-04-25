# [2026-03-16] EventMotionModel refactor Phase A 执行结果

模板参考：`docs/templates/changes/change_refactor_phaseA_template.md`  
路线图来源：`docs/changes/2026-03-16_event_motion_model_refactor_roadmap.md`

## A1. baseline 固化

- 固定输出目录：`debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline`
- 命令状态清单：`debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline/phaseA_baseline_commands.txt`
- 机器可读命令记录：`debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline/phaseA_command_status.json`
- 实际执行日期：2026-03-17（Asia/Shanghai）

| label | command | exit | 说明 | log |
|---|---|---:|---|---|
| `py_compile` | `python -m py_compile train/models.py` | 0 | roadmap Phase A required baseline command | `debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline/py_compile.log` |
| `debug_contact_loop_script_path` | `python train/debug_contact_loop.py` | 1 | exact roadmap spelling; frozen as-is even though package import fails from repo root | `debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline/debug_contact_loop_script_path.log` |
| `debug_contact_loop_module_path` | `python -m train.debug_contact_loop` | 0 | package-form replay command used as working smoke equivalent for post-refactor comparison | `debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline/debug_contact_loop_module_path.log` |

- `python -m py_compile train/models.py` 当前通过。
- `python train/debug_contact_loop.py` 当前固定失败，原因已冻结到 `debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline/debug_contact_loop_script_path.log`：repo root 下以脚本路径执行时 `from train.models import EventMotionModel` 无法解析包名。
- 为了保留一个可复跑的 smoke baseline，额外固化 `python -m train.debug_contact_loop`；当前通过，stdout 已写入 `debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline/debug_contact_loop_module_path.log`。

## A3. baseline 结构指标快照

- JSON：`debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline/phaseA_metrics_snapshot.json`
- TXT：`debug_output/event_motion_model_refactor_phaseA_20260316/mainchain_baseline/phaseA_metrics_snapshot.txt`

| metric | value |
|---|---:|
| `train/models.py` LOC | 6177 |
| `EventMotionModel` line range | `513-4473` |
| `EventMotionModel` length | 3961 |
| `EventMotionModel.forward` line range | `2256-4385` |
| `EventMotionModel.forward` length | 2130 |
| `except Exception: pass` count inside `EventMotionModel` | 39 |
| hotspot group count | 9 |
| hotspot anchor hit count | 21 |

## A2. key 引用清单

- raw 文件：`docs/changes/2026-03-16_event_motion_model_refactor_phaseA_key_refs_raw.txt`
- 固定文件范围：`train/models.py`
- 固定方法：对 roadmap hotspot anchor 做 exact-substring 扫描
- 总命中行数：21
- 按文件命中数：`train/models.py=21`

| key | hits |
|---|---:|
| `contacts_input_canonicalization_block_a` | 1 |
| `contacts_input_canonicalization_block_b` | 1 |
| `meas_logits_prev_canonicalization` | 2 |
| `phase_state_init_event_clock_on` | 1 |
| `phase_state_init_event_clock_off` | 1 |
| `contact_plan_loop_scaffold` | 2 |
| `direct_pose_trunk` | 2 |
| `direct_pose_leg_head` | 1 |
| `direct_pose_leg_gate_head` | 1 |
| `direct_pose_leg_head_shared` | 1 |
| `direct_pose_leg_gate_head_shared` | 1 |
| `rot6d_joint_count_helper` | 1 |
| `lambda_fusion_joint_count_manual_parse` | 1 |
| `so3_corr_joint_count_manual_parse` | 1 |
| `direct_pose_plan_override_canonicalization` | 1 |
| `direct_pose_meas_override_canonicalization` | 1 |
| `ablate_layout_helper` | 2 |

代表性锚点：

- `train/models.py:1481` / `train/models.py:1516`：direct-pose trunk 两处 `self.direct_pose_head = nn.Sequential(...)`
- `train/models.py:2578` / `train/models.py:2941`：`contacts_input` canonicalization 两套骨架起点
- `train/models.py:2603` / `train/models.py:2966`：`meas_logits_prev` canonicalization 两套骨架起点
- `train/models.py:2646` / `train/models.py:3001`：phase-state init 两套入口
- `train/models.py:2777` / `train/models.py:3097`：contact-plan per-step loop 两处主 scaffold
- `train/models.py:3493` / `train/models.py:3548`：plan/meas override canonicalization 两处入口
- `train/models.py:4102` / `train/models.py:4143`：`_ablate` layout A/B 双实现

## 验收结论

- baseline 命令与输出路径已经固化；其中 roadmap 原始脚本路径命令的失败现状也已冻结。
- 可复跑 smoke baseline 已补充 package-form 等价命令，便于 Phase B/C 后做 before/after 对照。
- key refs raw 清单已覆盖本轮 9 组热点，结构指标口径已固定，可直接作为后续重构对照基线。
