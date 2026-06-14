> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §5/§6/§8 under its stated read-only / zero-new-injection scope.

# Between feature-bridge experiment design

Date: 2026-06-05

Status: design + no-FK debug preflight/train-fit. No production trainer/runtime/gate/checkpoint
path is changed.

## 0. Reframe

`between` should be treated as a context-conditioned bridge:

```text
previous clip context + soft contact/cycle + target intent -> middle bridge features -> eval state
```

It should not be treated as:

```text
target -> free 16x281 regression
```

and it should not be treated as:

```text
Walk_F template pose + target root path
```

The current `state281` layout is useful as an observed/evaluation interface, but it mixes
different semantics:

```text
state281 = rot6d[0:276] + ego_vel[276:278] + yaw_rate[278:279] + contact[279:281]
```

For this experiment, `state281 [B,H,281] float32 cpu` is only the final assembled state used by
the existing reconstruction/contract scorer. It is not the primary modeling space.

## 1. Variable roles

### Causal context

`ctx [B,C,281] float32` comes from the previous clip. It is the causal state of the bridge:
phase, recent support, root velocity/yaw, local pose history, and contact history.

The model may encode this into features, but the experiment should not pretend the bridge starts
from an isolated Walk_F phase.

### Soft contact / cycle

`soft_contact [B,H,2] float32` is a continuous cycle/control signal. It should be used as a
feature/clock, not immediately collapsed into hard hand-authored support rules.

Important constraint: **do not put FK in the generator objective or feature definition.** FK
anchoring makes the bridge learn through a hand-authored skeleton geometry constraint and risks
poor generalization. Existing FK-derived `foot_slip` / `support_side` contracts can remain
legacy audits, but they are not the feature-bridge learning target.

Hard support labels/tokens are kept only as a debug ablation, not as the default design.

### Target / seam intent

The target should be represented as intent/arrival features, not as a full future pose leak.

First-round low-dimensional target features:

- `cond_dir [B,H,2] float32`
- `yaw_rate [B,H,1] float32`
- root displacement from start to end `[B,3] float32`
- endpoint root channels `ego_vel/yaw_rate [B,3] float32`

The existing `endpoint state prefix state281[-1,:279]` can remain a debug reference because the
old smoke harness used it, but it should not be the first feature-bridge target.

### Evaluation state

The bridge can still assemble:

- `rot6d [B,H,276] float32`
- `ego_vel/yaw_rate [B,H,3] float32`
- `contact [B,H,2] float32`
- optional `bone_angvel [B,H,138] float32` witness

into `state281 [B,H,281] float32` for the existing scorer.

## 2. First ablation matrix

Use a debug-only train-fit or read-only preflight before touching production code.

| Variant | Inputs | Purpose |
|---|---|---|
| A `ctx_target_lowdim` | `ctx + target_lowdim` | Can target intent alone steer a previous-context continuation? |
| B `ctx_soft_cycle` | `ctx + soft_contact_cycle` | How much does the soft cycle explain without target intent? |
| C `ctx_target_soft_cycle` | `ctx + target_lowdim + soft_contact_cycle` | Main hypothesis. |
| D `ctx_target_hard_support_debug` | `ctx + target_lowdim + hard support tokens` | Debug comparison only; checks whether hard labels are hiding the soft-cycle problem. |

The main comparison is C vs A/B. D is not a proposed production route.

## 3. Metrics

Primary no-FK audits:

- `state_mse`
- `pose_rot6d_mse`
- `ego_vel_mse`
- `yaw_rate_mse`
- `bone_angvel_aux_mse`
- `pose_step_mse`
- `root_intent_mse`
- `root_pos_mse`
- `root_vel_mse`

`foot_slip` must be read narrowly: it is contacted-foot speed under the claimed soft-contact
mask/band. It is not foot trajectory equality and not proof of support placement correctness.
For this no-FK experiment it is a legacy audit only, not a success metric.

`support_side_correctness` is an audit metric, not a hand-written training target. Do not train
directly on `left_rel_x_mean`/`right_rel_x_mean` bands in the first experiment.

## 4. Loss sketch

The first train-fit should avoid hand-defined support placement targets:

```text
L = L_feature_arrival
  + L_root_continuity
  + L_pose_continuity
  + optional L_bone_angvel_witness
```

Where:

- `L_feature_arrival` compares predicted terminal/bridge features to target intent features.
- `L_root_continuity` keeps root velocity/yaw continuous.
- `L_pose_continuity` keeps local pose/residual continuous.
- `L_bone_angvel_witness` is auxiliary only; `bone_angvel [B,H,138]` is not a `state281` field.
- No FK-derived loss is allowed in this experiment.

## 5. Preflight

Read-only probe:

```bash
python3 tools/run_action_handoff_between_feature_bridge_preflight.py
```

Expected artifacts:

```text
debug_output/_tmp_action_handoff_between_feature_bridge_preflight_20260605/
  summary.json
  summary.md
  feature_groups.csv
  variants.csv
  feature_target_guard_rows.csv
```

The preflight must confirm:

- feature groups have fixed shape/dtype and finite values;
- raw target and target-lowdim features are finite on matched windows;
- the proposed ablations are explicit about which inputs are runtime-safe, oracle/debug-only, or
  leakage-prone.

## 6. Decision rule

Proceed to a small debug train-fit only if the preflight is clean:

- all first-round feature groups finite
- raw target and target-lowdim features finite
- C has a clear, non-leaky input contract

The first no-FK train-fit success criterion is:

```text
root_intent_mse decreases
root_pos_mse / root_vel_mse decrease
pose_step_mse stays controlled
state/pose MSE are reported only as coarse reconstruction diagnostics
```

## 7. First execution result

Command:

```bash
python3 tools/run_action_handoff_between_feature_bridge_preflight.py --run-train-fit --epochs 300 --hidden-dim 256 --torch-num-threads 8
```

Artifacts:

```text
debug_output/_tmp_action_handoff_between_feature_bridge_preflight_20260605/
  summary.md
  summary.json
  feature_groups.csv
  variants.csv
  feature_target_guard_rows.csv
  train_fit_rows.csv
  train_fit_summary.csv
```

Preflight:

- matched windows: `188` from `Walk_L_To_L` / `Walk_R_To_L` / `Walk_R_To_R`
- `Walk_L_To_R`: `35` windows, excluded as unmatched diagnostic target
- feature target guard: raw target finite `1.000`; target-lowdim finite `1.000`
- all feature groups finite:
  - `ctx_state`: dim `4496`
  - `soft_contact_cycle`: dim `128`
  - `target_lowdim`: dim `54`
  - `hard_support_tokens_debug`: dim `102`
  - `endpoint_prefix_debug`: dim `279`

300-epoch no-FK debug train-fit, contiguous-block test partition (`n=53`):

| Variant | state MSE | pose MSE | root intent MSE | root pos MSE | root vel MSE |
|---|---:|---:|---:|---:|---:|
| `ctx_target_lowdim` | `0.01182565` | `0.00774114` | `0.37518908` | `0.00013717` | `0.01003067` |
| `ctx_soft_cycle` | `0.01186953` | `0.00777561` | `0.37582692` | `0.00013719` | `0.01007369` |
| `ctx_target_soft_cycle` | `0.01183466` | `0.00774469` | `0.37573205` | `0.00013704` | `0.01002615` |
| `ctx_target_hard_support_debug` | `0.01203943` | `0.00792838` | `0.37777747` | `0.00013709` | `0.01032912` |

Reading:

- The previous FK-loss run is invalid for this design direction and should not be used as a
  conclusion.
- The valid no-FK result shows the four input variants are nearly indistinguishable under a
  direct `6704`-dim raw-output MLP. Adding `soft_contact_cycle` to this output formulation does
  not create a meaningful advantage.
- Next experiment should stop asking the MLP to freely emit `6704` raw outputs. The output should
  be split into root/arrival + local feature residual, with soft contact kept as a cycle feature
  rather than an FK anchor.
