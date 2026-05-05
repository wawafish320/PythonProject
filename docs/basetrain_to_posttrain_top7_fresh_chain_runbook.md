# Fresh Basetrain -> Top7 Clean-StepC Runbook

> Last updated: 2026-04-22  
> Scope: `tail top7 basetrain -> fresh ckpt_last donor -> stage6 -> 70a -> replace -> 70R -> 71 -> 72 -> lambda`, plus optional experimental selective-continuation branch from `stage6 step360`  
> This runbook is for the **fresh basetrain donor** path and does **not** include legacy old-boundary detailed comparison.

---

## 1. TL;DR

这条链路的固定约束是：

- `basetrain` 使用  
  `config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json`
- `basetrain` donor 固定使用 fresh `ckpt_last_<run_name>.pth`，**不是** `ckpt_epoch_014.pth`
- `posttrain` 主链固定使用  
  `stage6 -> 70a -> warmstart copy -> replace -> 70R -> 71 -> 72 -> lambda`
- 上面这条仍然是 **canonical / default** posttrain 主链；如果要保留新的诊断分支，应该作为 **parallel experimental branch** 记录，而不是替换主链
- 运行入口固定使用 CPU / 禁用 MPS wrapper：  
  `debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py`
- 所有产物写入新的临时目录，避免覆盖旧结果

参考：

- canonical top7 StepC 文档：`docs/posttrain_pipeline_top7_clean_stepc.md`
- 已成功 fresh 链路上下文：`debug_output/_tmp_tail_top7_fresh_chain_20260413_195656/run_context.json`

---

## 2. Locked Runtime Contract

这条 fresh 链路默认锁定：

- posttrain contact source 在当前 mainline 中隐式固定为 `pretrain_contact`
  - 不再通过 `--posttrain_contacts_source` CLI 显式传入
- `CONTACT_CLAMP=1.0`
- `ENCODER_BUNDLE=models/motion_encoder_equiv.pt.best.pt`
- `AFFINE_STATS=debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`
- `CPU_WRAPPER=debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py`

---

## 3. Step 0 — 初始化变量

在 repo root 执行，并在**同一个 shell session**里继续后续步骤：

```bash
set -euo pipefail
set -a

cd /Users/xingzhaorui/PycharmProjects/PythonProject
ROOT="$(pwd)"
PYTHONPATH="$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="tail_top7_fresh_chain_${STAMP}"
RUN_ROOT="$ROOT/debug_output/_tmp_${RUN_TAG}"
MODEL_ROOT="$ROOT/models/__tmp_${RUN_TAG}"

CPU_WRAPPER="$ROOT/debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py"
BASE_CONFIG="$ROOT/config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json"
ENCODER_BUNDLE="$ROOT/models/motion_encoder_equiv.pt.best.pt"
AFFINE_STATS="$ROOT/debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json"
CONTACT_CLAMP="1.0"

# Important: use the frozen 2026-04-12 stage configs that reproduced the
# 2026-04-13 clean chain. Do not switch these to current repo `config/*.json`,
# otherwise `replace` may drift (notably `direct_pose_phase_z_mode`).
PT_CFG_STAGE6_BASE="$ROOT/debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_stage6_tailfix_top7_clean_stepc_20260412.json"
PT_CFG_70A_BASE="$ROOT/debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_70a_from_top7_stage6_clean_stepc_20260412.json"
PT_CFG_REPLACE_BASE="$ROOT/debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_replace_from_top7_70a_clean_stepc_20260412.json"
PT_CFG_70R_BASE="$ROOT/debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_70R_from_top7_replace_clean_stepc_20260412.json"
PT_CFG_71_BASE="$ROOT/debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_71_from_top7_70R_clean_stepc_20260412.json"
PT_CFG_72_BASE="$ROOT/debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_72_from_top7_71_clean_stepc_20260412.json"
PT_CFG_LAMBDA_BASE="$ROOT/debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_lambda_from_top7_72_clean_stepc_20260412.json"

mkdir -p "$RUN_ROOT/configs" "$RUN_ROOT/logs" "$MODEL_ROOT"
```

---

## 4. Step 1 — 生成 basetrain runtime config

当前 active `BASE_CONFIG` 已经清理为 `train.training_MPL` 可直接解析的合法 config；这一步主要生成 fresh run 的 runtime copy：

- 改写输出目录到新的临时根目录
- 改写 `run_name` 与 `freerun_debug_path`
- 强制 `amp=false`
- 旧键清理仅作 **backward-compatible defensive cleanup only**：
  - `save_fit_ckpt_epochs`
  - `seed`
  - `rot_local_tail_rank_mix`
  - `rot_local_tail_reduce`
  - `rot_local_tail_uniform_mix`
  - `trainbase_contacts_source`

```bash
BASETRAIN_CFG="$RUN_ROOT/configs/basetrain_runtime.json"
BASETRAIN_RUN_NAME="fresh_tail_top7_basetrain_${STAMP}"
BASETRAIN_OUT_ROOT="$MODEL_ROOT/basetrain"
BASETRAIN_OUT_DIR="$BASETRAIN_OUT_ROOT/$BASETRAIN_RUN_NAME"
BASETRAIN_CKPT="$BASETRAIN_OUT_DIR/ckpt_last_${BASETRAIN_RUN_NAME}.pth"
BASETRAIN_FREERUN_DIAG="$RUN_ROOT/basetrain/freerun_diag_${BASETRAIN_RUN_NAME}.pt"

python3 - <<'PY'
import json
import os
from pathlib import Path

base_config = Path(os.environ["BASE_CONFIG"])
out_json = Path(os.environ["BASETRAIN_CFG"])
payload = json.loads(base_config.read_text(encoding="utf-8"))

# Backward-compatible defensive cleanup only.
# Active BASE_CONFIG should already be parser-legal without these pops.
for key in [
    "save_fit_ckpt_epochs",
    "seed",
    "rot_local_tail_rank_mix",
    "rot_local_tail_reduce",
    "rot_local_tail_uniform_mix",
    "trainbase_contacts_source",
]:
    payload.pop(key, None)

payload["out"] = os.environ["BASETRAIN_OUT_ROOT"]
payload["run_name"] = os.environ["BASETRAIN_RUN_NAME"]
payload["freerun_debug_path"] = os.environ["BASETRAIN_FREERUN_DIAG"]
payload["amp"] = False
payload["config_json"] = os.environ["BASE_CONFIG"]

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(out_json)
PY
```

---

## 5. Step 2 — 跑 fresh basetrain

```bash
"$CPU_WRAPPER" -m train.training_MPL \
  --config_json "$BASETRAIN_CFG" \
  2>&1 | tee "$RUN_ROOT/logs/basetrain.log"

test -f "$BASETRAIN_CKPT"
echo "$BASETRAIN_CKPT"
```

成功后 donor 固定取：

```bash
echo "$BASETRAIN_CKPT"
```

---

## 6. Step 3 — 定义 posttrain 路径变量

```bash
STAGE6_RUN_NAME="stage6_tailfix_top7_stepc_clean_fromfresh_${STAMP}"
STAGE6_OUT_DIR="$MODEL_ROOT/stage6_stepc_handoff"
STAGE6_CFG="$RUN_ROOT/configs/stage6_${STAMP}.json"
STAGE6_CKPT="$STAGE6_OUT_DIR/ckpt_last_${STAGE6_RUN_NAME}.pth"
STAGE6_GROUP="$RUN_ROOT/stage6_stepc_handoff/eval_model_source_group_summary.json"

S70A_RUN_NAME="WalkF_stage7_70a_from_fresh_tailk7_stage6stepc_clean_${STAMP}"
S70A_OUT_DIR="$MODEL_ROOT/70a_clean"
S70A_CFG="$RUN_ROOT/configs/70a_${STAMP}.json"
S70A_CKPT="$S70A_OUT_DIR/ckpt_last_${S70A_RUN_NAME}.pth"
S70A_GROUP="$RUN_ROOT/70a_clean/eval_model_source_group_summary.json"

WARMSTART_OUT_DIR="$MODEL_ROOT/warmstart_clean"
WARMSTART_CKPT="$WARMSTART_OUT_DIR/ckpt_last_fresh_tail_top7_70a_replace_zerophase_cleanstepc_${STAMP}.pth"

REPLACE_RUN_NAME="WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_from_fresh_tailk7_70a_cleanstepc_${STAMP}"
REPLACE_OUT_DIR="$MODEL_ROOT/replace_clean"
REPLACE_CFG="$RUN_ROOT/configs/replace_${STAMP}.json"
REPLACE_CKPT="$REPLACE_OUT_DIR/ckpt_last_${REPLACE_RUN_NAME}.pth"
REPLACE_GROUP="$RUN_ROOT/replace_clean/eval_model_source_group_summary.json"

S70R_RUN_NAME="WalkF_stage7_70R_from_fresh_tailk7_replace_cleanstepc_lr1e4_s180_${STAMP}"
S70R_OUT_DIR="$MODEL_ROOT/70R_clean"
S70R_CFG="$RUN_ROOT/configs/70R_${STAMP}.json"
S70R_CKPT="$S70R_OUT_DIR/ckpt_last_${S70R_RUN_NAME}.pth"
S70R_GROUP="$RUN_ROOT/70R_clean/eval_model_source_group_summary.json"

S71_RUN_NAME="WalkF_stage7_71_from_fresh_70R_cleanstepc_lr3e4_${STAMP}"
S71_OUT_DIR="$MODEL_ROOT/71_clean"
S71_CFG="$RUN_ROOT/configs/71_${STAMP}.json"
S71_CKPT="$S71_OUT_DIR/ckpt_last_${S71_RUN_NAME}.pth"
S71_GROUP="$RUN_ROOT/71_clean/eval_model_source_group_summary.json"

S72_RUN_NAME="WalkF_stage7_72_from_fresh_71_cleanstepc_lr1e4_${STAMP}"
S72_OUT_DIR="$MODEL_ROOT/72_clean"
S72_CFG="$RUN_ROOT/configs/72_${STAMP}.json"
S72_CKPT="$S72_OUT_DIR/ckpt_last_${S72_RUN_NAME}.pth"
S72_GROUP="$RUN_ROOT/72_clean/eval_model_source_group_summary.json"

LAMBDA_RUN_NAME="WalkF_stage7_lambda_from_fresh_72_cleanstepc_${STAMP}"
LAMBDA_OUT_DIR="$MODEL_ROOT/lambda_clean"
LAMBDA_CFG="$RUN_ROOT/configs/lambda_${STAMP}.json"
LAMBDA_CKPT="$LAMBDA_OUT_DIR/ckpt_last_${LAMBDA_RUN_NAME}.pth"
LAMBDA_GROUP="$RUN_ROOT/lambda_clean/eval_model_source_group_summary.json"
```

---

## 7. Step 4 — 从冻结且已验证的 stage config 生成 posttrain config

这一步只做路径和少量 recipe override，不改 repo 主代码；base config 必须来自上面的冻结 stage configs，而不是当前 repo `config/*.json`。

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

def load(path_env: str):
    return json.loads(Path(os.environ[path_env]).read_text(encoding="utf-8"))

def dump(path_env: str, payload):
    path = Path(os.environ[path_env])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

common = {
    "encoder_bundle": os.environ["ENCODER_BUNDLE"],
    "device": "cpu",
    "posttrain_contacts_pretrain_clamp": os.environ["CONTACT_CLAMP"],
    "posttrain_contacts_pretrain_affine_stats": os.environ["AFFINE_STATS"],
}

specs = [
    (
        "PT_CFG_STAGE6_BASE",
        "STAGE6_CFG",
        {
            "ckpt_in": os.environ["BASETRAIN_CKPT"],
            "out_dir": os.environ["STAGE6_OUT_DIR"],
            "run_name": os.environ["STAGE6_RUN_NAME"],
        },
    ),
    (
        "PT_CFG_70A_BASE",
        "S70A_CFG",
        {
            "ckpt_in": os.environ["STAGE6_CKPT"],
            "out_dir": os.environ["S70A_OUT_DIR"],
            "run_name": os.environ["S70A_RUN_NAME"],
            "lr": 3e-4,
        },
    ),
    (
        "PT_CFG_REPLACE_BASE",
        "REPLACE_CFG",
        {
            "ckpt_in": os.environ["WARMSTART_CKPT"],
            "out_dir": os.environ["REPLACE_OUT_DIR"],
            "run_name": os.environ["REPLACE_RUN_NAME"],
            "epochs": 3,
            "steps_per_epoch": 60,
            "lr": 5e-5,
        },
    ),
    (
        "PT_CFG_70R_BASE",
        "S70R_CFG",
        {
            "ckpt_in": os.environ["REPLACE_CKPT"],
            "out_dir": os.environ["S70R_OUT_DIR"],
            "run_name": os.environ["S70R_RUN_NAME"],
            "epochs": 1,
            "lr": 1e-4,
        },
    ),
    (
        "PT_CFG_71_BASE",
        "S71_CFG",
        {
            "ckpt_in": os.environ["S70R_CKPT"],
            "out_dir": os.environ["S71_OUT_DIR"],
            "run_name": os.environ["S71_RUN_NAME"],
            "lr": 3e-4,
        },
    ),
    (
        "PT_CFG_72_BASE",
        "S72_CFG",
        {
            "ckpt_in": os.environ["S71_CKPT"],
            "out_dir": os.environ["S72_OUT_DIR"],
            "run_name": os.environ["S72_RUN_NAME"],
            "lr": 1e-4,
        },
    ),
    (
        "PT_CFG_LAMBDA_BASE",
        "LAMBDA_CFG",
        {
            "ckpt_in": os.environ["S72_CKPT"],
            "out_dir": os.environ["LAMBDA_OUT_DIR"],
            "run_name": os.environ["LAMBDA_RUN_NAME"],
        },
    ),
]

for base_env, out_env, overrides in specs:
    payload = load(base_env)
    payload.update(common)
    payload.update(overrides)
    dump(out_env, payload)
PY
```

---

## 8. Step 5 — 70R 直接调用当前 runner

当前本地代码已经验证：`tools/run_posttrain_nonleg_trunk_ablation.py` 可以直接配合
当前 `train.posttrain` 运行，**不需要额外 shim**。

```bash
test -f "$ROOT/tools/run_posttrain_nonleg_trunk_ablation.py"
```

---

## 9. Step 6 — 跑 `stage6 -> 70a -> replace -> 70R -> 71 -> 72 -> lambda`

### 9.1 Stage6

```bash
"$CPU_WRAPPER" -m train.posttrain \
  --config "$STAGE6_CFG" \
  --ckpt_in "$BASETRAIN_CKPT" \
  --out_dir "$STAGE6_OUT_DIR" \
  --run_name "$STAGE6_RUN_NAME" \
  --posttrain_contacts_pretrain_clamp "$CONTACT_CLAMP" \
  --encoder_bundle "$ENCODER_BUNDLE" \
  --posttrain_contacts_pretrain_affine_stats "$AFFINE_STATS" \
  2>&1 | tee "$RUN_ROOT/logs/stage6.log"

test -f "$STAGE6_CKPT"
```

### 9.2 70a

```bash
"$CPU_WRAPPER" -m train.posttrain \
  --config "$S70A_CFG" \
  --ckpt_in "$STAGE6_CKPT" \
  --out_dir "$S70A_OUT_DIR" \
  --run_name "$S70A_RUN_NAME" \
  --posttrain_contacts_pretrain_clamp "$CONTACT_CLAMP" \
  --encoder_bundle "$ENCODER_BUNDLE" \
  --posttrain_contacts_pretrain_affine_stats "$AFFINE_STATS" \
  2>&1 | tee "$RUN_ROOT/logs/70a.log"

test -f "$S70A_CKPT"
```

### 9.3 Warmstart copy

```bash
mkdir -p "$WARMSTART_OUT_DIR"
cp "$S70A_CKPT" "$WARMSTART_CKPT"
test -f "$WARMSTART_CKPT"
```

### 9.4 replace

```bash
"$CPU_WRAPPER" -m train.posttrain \
  --config "$REPLACE_CFG" \
  --ckpt_in "$WARMSTART_CKPT" \
  --out_dir "$REPLACE_OUT_DIR" \
  --run_name "$REPLACE_RUN_NAME" \
  --posttrain_contacts_pretrain_clamp "$CONTACT_CLAMP" \
  --encoder_bundle "$ENCODER_BUNDLE" \
  --posttrain_contacts_pretrain_affine_stats "$AFFINE_STATS" \
  2>&1 | tee "$RUN_ROOT/logs/replace.log"

test -f "$REPLACE_CKPT"
```

### 9.5 70R

`S70R_CFG` 在 Step 4 已固定写入 `lr=1e-4`；这里的直接启动命令不再额外覆写学习率。

```bash
"$CPU_WRAPPER" "$ROOT/tools/run_posttrain_nonleg_trunk_ablation.py" \
  --config "$S70R_CFG" \
  --trunk-mode full \
  --out-dir "$S70R_OUT_DIR" \
  --run-name "$S70R_RUN_NAME" \
  --epochs 1 \
  --steps-per-epoch 180 \
  --save-step-ckpts 0,1,5,20,60,180 \
  2>&1 | tee "$RUN_ROOT/logs/70R.log"

test -f "$S70R_CKPT"
```

### 9.6 71

```bash
"$CPU_WRAPPER" -m train.posttrain \
  --config "$S71_CFG" \
  --ckpt_in "$S70R_CKPT" \
  --out_dir "$S71_OUT_DIR" \
  --run_name "$S71_RUN_NAME" \
  --posttrain_contacts_pretrain_clamp "$CONTACT_CLAMP" \
  --encoder_bundle "$ENCODER_BUNDLE" \
  --posttrain_contacts_pretrain_affine_stats "$AFFINE_STATS" \
  2>&1 | tee "$RUN_ROOT/logs/71.log"

test -f "$S71_CKPT"
```

### 9.7 72

```bash
"$CPU_WRAPPER" -m train.posttrain \
  --config "$S72_CFG" \
  --ckpt_in "$S71_CKPT" \
  --out_dir "$S72_OUT_DIR" \
  --run_name "$S72_RUN_NAME" \
  --posttrain_contacts_pretrain_clamp "$CONTACT_CLAMP" \
  --encoder_bundle "$ENCODER_BUNDLE" \
  --posttrain_contacts_pretrain_affine_stats "$AFFINE_STATS" \
  2>&1 | tee "$RUN_ROOT/logs/72.log"

test -f "$S72_CKPT"
```

### 9.8 lambda

```bash
"$CPU_WRAPPER" -m train.posttrain \
  --config "$LAMBDA_CFG" \
  --ckpt_in "$S72_CKPT" \
  --out_dir "$LAMBDA_OUT_DIR" \
  --run_name "$LAMBDA_RUN_NAME" \
  --posttrain_contacts_pretrain_clamp "$CONTACT_CLAMP" \
  --encoder_bundle "$ENCODER_BUNDLE" \
  --posttrain_contacts_pretrain_affine_stats "$AFFINE_STATS" \
  2>&1 | tee "$RUN_ROOT/logs/lambda.log"

test -f "$LAMBDA_CKPT"
```

---

## 10. Step 7 — 生成 `run_context.json`

这一步会把每个阶段的：

- 启动命令
- 输入 ckpt
- 输出 ckpt
- 是否成功结束
- 若已生成评估文件，则附上 `eval_model_source_group_summary.json` 的核心指标

统一写到：

- `$RUN_ROOT/run_context.json`

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])

def metrics_from_group(group_path):
    if not group_path:
        return None
    path = Path(group_path)
    if not path.is_file():
        return None
    groups = json.loads(path.read_text(encoding="utf-8")).get("groups", {})
    out = {}
    for name in ("all_ex_root", "leg", "nonleg", "arm"):
        row = groups.get(name, {})
        out[name] = {
            "mean": row.get("mean"),
            "p50": row.get("p50"),
            "p90": row.get("p90"),
            "p95": row.get("p95"),
            "samples": row.get("samples"),
        }
    return out

def stage_row(name, launch_command, input_ckpt, output_ckpt, group_summary):
    return {
        "launch_command": launch_command,
        "input_ckpt": input_ckpt,
        "output_ckpt": output_ckpt,
        "group_summary": group_summary,
        "metrics": metrics_from_group(group_summary),
        "success": bool(output_ckpt) and Path(output_ckpt).is_file(),
    }

summary = {
    "run_root": os.environ["RUN_ROOT"],
    "model_root": os.environ["MODEL_ROOT"],
    "base_config": os.environ["BASE_CONFIG"],
    "compat_notes": [
        "basetrain runtime copy: rewrote out / run_name / freerun_debug_path and forced amp=false for CPU/no-MPS",
        "basetrain runtime copy: backward-compatible defensive cleanup only for save_fit_ckpt_epochs / seed / rot_local_tail_rank_mix / rot_local_tail_reduce / rot_local_tail_uniform_mix / trainbase_contacts_source",
        "70R launched directly via tools/run_posttrain_nonleg_trunk_ablation.py (no shim)",
    ],
    "stages": {
        "basetrain": stage_row(
            "basetrain",
            f'PYTHONPATH={os.environ["PYTHONPATH"]} {os.environ["CPU_WRAPPER"]} -m train.training_MPL --config_json {os.environ["BASETRAIN_CFG"]}',
            None,
            os.environ["BASETRAIN_CKPT"],
            None,
        ),
        "stage6": stage_row(
            "stage6",
            f'PYTHONPATH={os.environ["PYTHONPATH"]} {os.environ["CPU_WRAPPER"]} -m train.posttrain --config {os.environ["STAGE6_CFG"]} --ckpt_in {os.environ["BASETRAIN_CKPT"]} --out_dir {os.environ["STAGE6_OUT_DIR"]} --run_name {os.environ["STAGE6_RUN_NAME"]} --posttrain_contacts_pretrain_clamp {os.environ["CONTACT_CLAMP"]} --encoder_bundle {os.environ["ENCODER_BUNDLE"]} --posttrain_contacts_pretrain_affine_stats {os.environ["AFFINE_STATS"]}',
            os.environ["BASETRAIN_CKPT"],
            os.environ["STAGE6_CKPT"],
            os.environ["STAGE6_GROUP"],
        ),
        "70a": stage_row(
            "70a",
            f'PYTHONPATH={os.environ["PYTHONPATH"]} {os.environ["CPU_WRAPPER"]} -m train.posttrain --config {os.environ["S70A_CFG"]} --ckpt_in {os.environ["STAGE6_CKPT"]} --out_dir {os.environ["S70A_OUT_DIR"]} --run_name {os.environ["S70A_RUN_NAME"]} --posttrain_contacts_pretrain_clamp {os.environ["CONTACT_CLAMP"]} --encoder_bundle {os.environ["ENCODER_BUNDLE"]} --posttrain_contacts_pretrain_affine_stats {os.environ["AFFINE_STATS"]}',
            os.environ["STAGE6_CKPT"],
            os.environ["S70A_CKPT"],
            os.environ["S70A_GROUP"],
        ),
        "warmstart_clean": stage_row(
            "warmstart_clean",
            f'cp {os.environ["S70A_CKPT"]} {os.environ["WARMSTART_CKPT"]}',
            os.environ["S70A_CKPT"],
            os.environ["WARMSTART_CKPT"],
            None,
        ),
        "replace": stage_row(
            "replace",
            f'PYTHONPATH={os.environ["PYTHONPATH"]} {os.environ["CPU_WRAPPER"]} -m train.posttrain --config {os.environ["REPLACE_CFG"]} --ckpt_in {os.environ["WARMSTART_CKPT"]} --out_dir {os.environ["REPLACE_OUT_DIR"]} --run_name {os.environ["REPLACE_RUN_NAME"]} --posttrain_contacts_pretrain_clamp {os.environ["CONTACT_CLAMP"]} --encoder_bundle {os.environ["ENCODER_BUNDLE"]} --posttrain_contacts_pretrain_affine_stats {os.environ["AFFINE_STATS"]}',
            os.environ["WARMSTART_CKPT"],
            os.environ["REPLACE_CKPT"],
            os.environ["REPLACE_GROUP"],
        ),
        "70R": stage_row(
            "70R",
            f'PYTHONPATH={os.environ["PYTHONPATH"]} {os.environ["CPU_WRAPPER"]} {Path(os.environ["ROOT"]) / "tools" / "run_posttrain_nonleg_trunk_ablation.py"} --config {os.environ["S70R_CFG"]} --trunk-mode full --out-dir {os.environ["S70R_OUT_DIR"]} --run-name {os.environ["S70R_RUN_NAME"]} --epochs 1 --steps-per-epoch 180 --save-step-ckpts 0,1,5,20,60,180',
            os.environ["REPLACE_CKPT"],
            os.environ["S70R_CKPT"],
            os.environ["S70R_GROUP"],
        ),
        "71": stage_row(
            "71",
            f'PYTHONPATH={os.environ["PYTHONPATH"]} {os.environ["CPU_WRAPPER"]} -m train.posttrain --config {os.environ["S71_CFG"]} --ckpt_in {os.environ["S70R_CKPT"]} --out_dir {os.environ["S71_OUT_DIR"]} --run_name {os.environ["S71_RUN_NAME"]} --posttrain_contacts_pretrain_clamp {os.environ["CONTACT_CLAMP"]} --encoder_bundle {os.environ["ENCODER_BUNDLE"]} --posttrain_contacts_pretrain_affine_stats {os.environ["AFFINE_STATS"]}',
            os.environ["S70R_CKPT"],
            os.environ["S71_CKPT"],
            os.environ["S71_GROUP"],
        ),
        "72": stage_row(
            "72",
            f'PYTHONPATH={os.environ["PYTHONPATH"]} {os.environ["CPU_WRAPPER"]} -m train.posttrain --config {os.environ["S72_CFG"]} --ckpt_in {os.environ["S71_CKPT"]} --out_dir {os.environ["S72_OUT_DIR"]} --run_name {os.environ["S72_RUN_NAME"]} --posttrain_contacts_pretrain_clamp {os.environ["CONTACT_CLAMP"]} --encoder_bundle {os.environ["ENCODER_BUNDLE"]} --posttrain_contacts_pretrain_affine_stats {os.environ["AFFINE_STATS"]}',
            os.environ["S71_CKPT"],
            os.environ["S72_CKPT"],
            os.environ["S72_GROUP"],
        ),
        "lambda": stage_row(
            "lambda",
            f'PYTHONPATH={os.environ["PYTHONPATH"]} {os.environ["CPU_WRAPPER"]} -m train.posttrain --config {os.environ["LAMBDA_CFG"]} --ckpt_in {os.environ["S72_CKPT"]} --out_dir {os.environ["LAMBDA_OUT_DIR"]} --run_name {os.environ["LAMBDA_RUN_NAME"]} --posttrain_contacts_pretrain_clamp {os.environ["CONTACT_CLAMP"]} --encoder_bundle {os.environ["ENCODER_BUNDLE"]} --posttrain_contacts_pretrain_affine_stats {os.environ["AFFINE_STATS"]}',
            os.environ["S72_CKPT"],
            os.environ["LAMBDA_CKPT"],
            os.environ["LAMBDA_GROUP"],
        ),
    },
}

out_json = run_root / "run_context.json"
out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(out_json)
PY
```

---

## 11. 快速结果查看

### 11.1 fresh basetrain donor

```bash
echo "$BASETRAIN_CKPT"
```

### 11.2 全链路输出 ckpt

```bash
printf '%s\n' \
  "$STAGE6_CKPT" \
  "$S70A_CKPT" \
  "$WARMSTART_CKPT" \
  "$REPLACE_CKPT" \
  "$S70R_CKPT" \
  "$S71_CKPT" \
  "$S72_CKPT" \
  "$LAMBDA_CKPT"
```

### 11.3 打开结构化汇总

```bash
echo "$RUN_ROOT/run_context.json"
```

---

## 12. Experimental Branch — `stage6 step360` selective continuation

这一节记录一条**并行实验分支**，用于保留当前观察到的 trade-off；它**不替代**上面的 canonical 主链。

### 12.1 分支定位

- canonical 主链仍然是：`stage6 -> 70a -> warmstart copy -> replace -> 70R -> 71 -> 72 -> lambda`
- 实验分支的目标不是生成新的默认 baseline，而是显式保留一条：
  - `all_ex_root` 更优
  - `nonleg / arm / else` 更优
  - 但 `leg` 明显回退
  的候选链路，方便后续继续诊断 stage7 的 leg recovery 问题

### 12.2 当前已验证的实验链路（2026-04-25 update）

- 上游入口仍是 `stage6 step360`，但 fresh run root 切到本轮：
  - `debug_output/_tmp_tail_top7_fresh_chain_step360_20260425_030401/run_context.json`
- 完整链路（warmstart copy 仍是显式一步，与 §9 主链定义一致）：
  - `stage6(step360) -> 70a -> warmstart copy -> replace -> 70R(lr=1e-4, pick step20) -> 71(pick step120) -> 72(pick step150) -> lambda`
- 各 stage handoff 摘要：
  - `70R lr probe`（详见 §12.2a）：
    - `debug_output/_tmp_tail_top7_fresh_chain_step360_20260425_030401/70R_lr_probe/sweep_step20_summary.md`
  - `71` main（lr=1e-4 dense）：
    - `debug_output/_tmp_tail_top7_fresh_chain_step360_20260425_030401/71_lr_branch_cmp/summary_71s120.md`
  - `72 from 71@120` dense：
    - `debug_output/_tmp_tail_top7_fresh_chain_step360_20260425_030401/72_lr_branch_cmp/summary_72s150.md`
  - `lambda from 72@150`：
    - `debug_output/_tmp_tail_top7_fresh_chain_step360_20260425_030401/lambda_lr_branch_cmp/summary_branch_lambda_eval.md`
- `lambda` 在这条分支上**仍为 downstream no-op**（同表 `Δ(s200-72) = 0.00000` 全维成立），因此**有效最终端点仍记作 `72_step150`**；`lambda` 仅作"若未来 stage7 末端动力学发生变化时可重新激活"的占位。

> 备注：本节口语化描述 regression onset 时常会把 warmstart copy 略去（因为它不是训练阶段、只是 ckpt handoff）。但本 runbook 内对链路完整性的引用应**保留 warmstart copy**，与 §9 canonical 主链的 stage 命名严格一致。

### 12.2a 70R lr probe finding（2026-04-25 新增）

本轮最有操作性的新结论与 70R lr 默认值有关：

- 现 default `lr=3e-4` 在当前 refactored 代码线下，是 stage7 regression 的明确 onset 来源
- `lr=1e-4` 在 step20 处全维优于 `lr=3e-4`（数据见 `70R_lr_probe/sweep_step20_summary.md`）
- 因此本 experimental chain 已固定 70R `lr=1e-4`；`lr=3e-4` 视为 refactor 期间未及更新的 stale default
- 2026-04-29 补充：新的正式入口 `tools/run_stage6final_canonical_downstream.py` 已把 `70R` 默认值收敛到 `lr=1e-4`；旧的 `3e-4` 仅保留为显式 ablation / override 值，不再建议作为 canonical downstream default。

### 12.3 和 healthy final / historical optima 的当前对比

参考基线（**注意基线属于哪条代码线**）：

- `_tmp_tail_top7_fresh_chain_20260418_074813/lambda_clean` —— **同代码线 healthy final**，本节 ship-gate anchor
- `_tmp_tail_top7_fresh_chain_step360_20260425_030401/lambda_lr_branch_cmp/summary_branch_lambda_eval.md` —— 本分支评估
- `old_lambda_20260416` —— **老代码线 historical optimum**，仅历史对照，**不参与 ship gate**

完整四列对比（`branch_lambda_s200` / `current_lambda_s200` / `old_lambda_20260418` / `old_lambda_20260416`）：

| group | metric | branch | current | 0418 | 0416 | best |
|---|---:|---:|---:|---:|---:|---|
| all_ex_root | mean | 0.12035 | 0.14299 | 0.13188 | 0.11660 | 0416 |
| all_ex_root | p50  | 0.07053 | 0.08881 | 0.08135 | 0.07370 | branch |
| all_ex_root | p90  | 0.29122 | 0.34828 | 0.32050 | 0.28167 | 0416 |
| all_ex_root | p95  | 0.40690 | 0.45967 | 0.43957 | 0.37617 | 0416 |
| leg | mean | 0.17817 | 0.20328 | 0.17306 | 0.16532 | 0416 |
| leg | p50  | 0.13731 | 0.15942 | 0.13229 | 0.12452 | 0416 |
| leg | p90  | 0.36652 | 0.41422 | 0.32997 | 0.34027 | 0418 |
| leg | p95  | 0.45646 | 0.52292 | 0.47462 | 0.46136 | branch |
| nonleg | mean | 0.10785 | 0.12995 | 0.12298 | 0.10606 | 0416 |
| nonleg | p50  | 0.05799 | 0.06975 | 0.06952 | 0.06245 | branch |
| nonleg | p90  | 0.27003 | 0.33680 | 0.31706 | 0.26694 | 0416 |
| nonleg | p95  | 0.38135 | 0.44496 | 0.43405 | 0.35823 | 0416 |
| arm | mean | 0.12581 | 0.14846 | 0.14320 | 0.12061 | 0416 |
| arm | p50  | 0.06292 | 0.07292 | 0.07853 | 0.06646 | branch |
| arm | p90  | 0.32375 | 0.38766 | 0.37485 | 0.30808 | 0416 |
| arm | p95  | 0.44143 | 0.49163 | 0.50520 | 0.40665 | 0416 |
| else | mean | 0.06538 | 0.08620 | 0.07519 | 0.07167 | branch |
| else | p50  | 0.05207 | 0.06628 | 0.05615 | 0.05548 | branch |
| else | p90  | 0.14298 | 0.20573 | 0.17480 | 0.16561 | branch |
| else | p95  | 0.17831 | 0.23343 | 0.20925 | 0.20425 | branch |

读这张表的方式：

- **vs 同代码线 healthy final（0418）**：
  - `all_ex_root` / `nonleg` / `arm` / `else` 全部优于
  - `leg`：`mean / p50` 略差（~+3% / +4%），`p90` 略差（~+11%），`p95` 反超（~-4%）
  - 已显著优于 refactor 后未优化的 `current_lambda_s200`（默认链路），证明 70R lr=1e-4 + step360 入口的确是 refactored 代码线下当前最佳已知 continuation
- **vs 历史最优（0416，老代码线）**：
  - `else` 全 4 项反超；`leg p95` 反超；多个 p50 反超
  - `leg / nonleg / arm` 的 mean / p90 / p95 仍略差，量级集中在 ~3–8% 区间
  - 这个 gap 属于 **refactor-cost gap**（当前代码线、配置默认尚未重新调优），不是 stage7 regression 重新出现
  - 因此 `0416` 仅作 historical reference，不构成本分支的 ship-gate failure

### 12.4 后续复跑建议

- **不建议继续在 71/72 上扫 schedule 去追 leg tail**。本分支已经基本把 71/72 在 `stage6 step360` 入口、70R lr=1e-4 配置下的局部下界摸到：
  - `71` lr=1e-4 dense 的 leg p95 拐点落在 `step120`（数据见 `71_lr_branch_cmp/summary_71s120.md`），不是单调下降
  - `72 from 71@120` lr=1e-4 dense 的 leg p95 在 `step150` 取最低，late-step 仍出现轻微回退
  - 当前 refactored 配置下继续扫 71/72 schedule，大概率只能带来低幅波动，难以再消化对 0418 的 leg mean/p90 残差
- 如果只是要**复跑验证下界没漂**（不是找更优点）：
  - `71` 保持 lr=1e-4 dense ckpt `(0,1,5,20,...,120,150,180)`，重点核 `step120` 的 leg p90/p95
  - `72 from 71@120` 保持 lr=1e-4 dense ckpt，重点核 `step150` 的 leg p90/p95
  - 复核 `lambda` 是否仍为 downstream no-op
- 若要**主动改善 leg gap**：按当前证据，下一步应进 §12.6 stage6 cutpoint shift；71/72 / 70a / replace 内部不要再先动
- 任何复跑都**不要覆盖** canonical fresh chain 的 `71 / 72 / lambda`；保留主链，单独新建 `_tmp_*` run root 承接 continuation

### 12.5 Ship criteria（作为 experimental candidate 固化）

**Ship-gate anchor 显式声明**：

- gate 基线 = `_tmp_tail_top7_fresh_chain_20260418_074813/lambda_clean`（**同代码线 healthy final**）
- `old_lambda_20260416` 仅作 historical reference，**不参与 ship gate**（跨代码线对比不构成 apples-to-apples）

把这条分支固化为 experimental candidate（仍不替代 canonical 主链）的条件，全部对 anchor（0418）：

- `leg p95` 对 anchor 的 gap ≤ ~5%，`leg mean` 对 anchor 的 gap ≤ ~7%
  - 当前 `leg p95` ≈ -3.8%（反超），`leg mean` ≈ +3.0%，**满足**
- `all_ex_root` / `nonleg` / `arm` / `else` 每一项都**至少不劣于** anchor
  - 当前全部**优于** anchor，**满足**

下列任一情况出现，则应重新审视这条分支是否仍值得保留为候选：

- 复跑后非 leg 任一组（`nonleg` / `arm` / `else` / `all_ex_root`）从"优于 anchor"掉回"劣于 anchor"
- `leg mean` 或 `leg p95` 对 anchor 的 gap 扩大超过 ~8%
- `lambda` 不再为 downstream no-op（`step0 != step_end`），意味着 stage7 末端动力学变化，现有"有效终点 = 72_step150"的记账方式失效
- 70R `lr=1e-4` 对 `lr=3e-4` 的全维优势在新一轮复跑中失去——若该 lr finding 不稳，本分支的入口配置需从 70R lr 重新评估

这条分支**不**作为 canonical 主链替代的前提：

- 入口是 `stage6 step360`，**不经过** canonical `stage6 ckpt_last`
- 仍受 refactor-cost gap 约束（vs `0416`），表明当前代码线尚未重新调优；判断"是否替代 canonical"的窗口应等到 refactor 收口、配置 default 重新对齐之后再开

### 12.6 下一实验 — stage6 cutpoint shift（唯一推荐动作）

目标：用本分支在非 leg 维度相对 healthy 的 headroom，换 leg 起点改善，验证能否关掉当前 `+0.02` 级别的 leg p95 残差。

前置假设（必须在实验前核对，否则不划算）：

- 当前本分支 `all_ex_root` / `nonleg` / `arm` / `else` 仍然**全部优于** healthy final —— 这是"可以用 headroom 换 leg"的前提；如果这个前提不再成立，这个实验直接不起

做法：

- 基于同一 fresh basetrain donor，stage6 dense ckpt 至少保存 `360 / 380 / 400`
- 三个 cutpoint 各自起完整 continuation：`-> 70a -> warmstart copy -> replace -> 70R(lr=1e-4) -> 71 (dense, pick step120) -> 72 (dense, pick step150) -> lambda`
- 70R 复用 §12.2a 的 `lr=1e-4` 结论，不再保留 `lr=3e-4` 作为 cutpoint 之外的混淆因子
- 除 stage6 cutpoint 外，**所有其它参数冻结**（lr / steps_per_epoch / save_step_ckpts / 71/72 选点规则），保证唯一变量就是 cutpoint

三方对比：

- `cutpoint=360`（当前分支）
- `cutpoint=380 / 400`（新）
- `healthy final`（`debug_output/_tmp_tail_top7_fresh_chain_20260418_074813/lambda_clean/eval_model_source_group_summary.json`）

验收通过的条件（必须同时满足）：

- `leg mean` 和 `leg p95` 对 healthy final 的 gap **都缩小**
- `all_ex_root` / `nonleg` / `arm` / `else` 每一项都**仍然不劣于** healthy final

验收失败 / 需要撤回的条件（任一成立即回退到 `cutpoint=360`）：

- 任一非 leg 组从"不劣于 healthy"退化为"劣于 healthy"
- leg gap 没有缩小，或反而扩大

在这个实验得出结论**之前**，下列动作都不起：

- 71/72 的 full sweep
- 70a / replace 的 leg reinjection（引入 leg 到 nonleg_train_only 路径的改动）
- 重新选择 stage6 `ckpt_last` vs step-cut 的默认入口

---

## 13. Notes / Caveats

- 这份 runbook 是 **fresh basetrain ckpt_last donor** 版本，不是 canonical `ep014 donor` 版本。
- `70R` 当前 runbook 直接调用 `tools/run_posttrain_nonleg_trunk_ablation.py`；本文件默认以当前 `PostTrainModelArtifacts` / `_save_posttrain_outputs(...)` API 为准，如接口再次漂移，再重新评估是否需要临时 shim。
- 如果你后续想把这条链路再封成单个 runner，优先把本文件中的变量名和 stage 命名直接复用。

---

## 14. Canonical Downstream Entrypoint (2026-04-29 rewrite)

如果你的起点已经不是 fresh basetrain，而是一个**现成的 strict/current `stage6-final` checkpoint**，后续不要再手工敲
`70a -> replace -> 70R -> 71 -> 72 -> lambda` 这串命令。直接用正式入口：

```bash
python3 tools/run_stage6final_canonical_downstream.py
```

默认输入是本轮 0425 成功路径对应的 strict donor：

- `debug_output/_tmp_legacy_ckpt_stage6final_rerun0425_20260429_001234/migrated_ckpts/stage6_final_strict_from0425.pth`

默认会新建一个独立 run root：

- `debug_output/_tmp_stage6final_canonical_downstream_<timestamp>/`

并固定编排：

- `stage6-final -> 70a -> replace -> 70R -> 71 -> 72 -> lambda`

关键约束已经内置在 runner 里：

- `replace` 明确保持 `direct_pose_phase_z_mode='concat'`
- `70R` 默认学习率固定为 `1e-4`；如需复旧配方，可显式传 `--lr-70r 3e-4`
- `replace -> 70R` 不再直接拿 replace ckpt 硬 handoff
- runner 会先 strip replace source 里的 `direct_pose_*`
- 然后用 `--allow-missing-prefix direct_pose_`
- 同时配合 donor step0 + `--transplant-prefix direct_pose_` 补回 coherent full `direct_pose_*` bundle

常用显式写法：

```bash
python3 tools/run_stage6final_canonical_downstream.py \
  --source-stage6 /abs/path/to/stage6_final_strict.pth \
  --run-root /abs/path/to/debug_output/_tmp_my_stage6final_downstream
```

产物重点看：

- `run_result.json`
- `config_manifest.json`
- `handoffs/`
- `evals/`
- `replace_vs_ref_compare.json`
- `lambda_vs_ref_compare.json`

如果只想先起训练、不跑评估：

```bash
python3 tools/run_stage6final_canonical_downstream.py --skip-eval
```

---

## 15. Experimental Branch — 用 `70R(1e-4 / 60 / last)` 替换默认 `70R`（2026-05-04）

这个分支**不要直接覆盖**上面的 canonical fresh chain。推荐作为独立 `_tmp_*` run root 并行记录。

本节对应的实验含义是：

- `stage6 -> 70a -> replace` 保持不变
- 只把 `70R` 改成：
  - `lr=1e-4`
  - `epochs=1`
  - `steps_per_epoch=60`
  - `save_step_ckpts=0,1,20`
  - downstream 入口固定取 `70R ckpt_last`
- 然后继续跑 `71 -> 72 -> lambda`

`71` 的配置需要单独记账。首版 downstream helper 使用的是旧默认 `71 lr=3e-4`，后续 2026-05-04 选点/降 LR 实验证明：

- `71` 不是 early-stop 问题：在现有 `lr=3e-4` checkpoint 里，`step120 == last` 是最佳点；`step40/60/90` 都不能解决 mean 回吐。
- 更优的第一改动是把 `71 lr` 从 `3e-4` 降到 `1e-4`，其它 leg-only 语义保持不变。
- 对本节的 dense `70R(1e-4 / 60 / last)` 输入，`71 lr=1e-4` 的最佳点是 `step90`，不是 `last`。

这条分支已经完成过一次完整验证，核心结论是：

- `70R last` 相比 fresh-chain `70R last` 更好，尤其是 `all_ex_root / leg`
- 首版 `71 lr=3e-4` 改善会传到 downstream：
  - `71 final`: `all_ex_root mean=0.120385`, `leg mean=0.292303`
  - `lambda final`: `all_ex_root mean=0.096919`, `leg mean=0.160309`
- follow-up `71 lr=1e-4` 在 71 本层显著更强：
  - dense `70R -> 71 lr1e4 step90`: `all_ex_root mean=0.102371`, `p95=0.336560`, `leg mean=0.190973`, `leg p95=0.504251`

对应已完成产物：

- `70R`: `debug_output/_tmp_70r_next_denseckpt_20260504_r6`
- downstream: `debug_output/_tmp_70r_dense_to_lambda_20260504_r2`
- `71 checkpoint select`: `debug_output/_tmp_71_ckpt_select_20260504/summary.md`
- `71 lr1e4 probe`: `debug_output/_tmp_71_lr1e4_probe_20260504/summary.md`

### 15.1 推荐入口

如果你的起点已经是一个**现成的 strict/current `replace` checkpoint**，不要手工改 §9 的命令串。直接分成两步：

1. 先生成新的 `70R(1e-4 / 60 / last)`
2. 再从这个 `70R ckpt_last` 继续跑 `71 -> 72 -> lambda`
3. 如果要采用当前更优 71，必须把 71 recipe 显式改成 §15.4 的 `lr=1e-4` 版本

### 15.2 Step A — 起 `70R(1e-4 / 60 / last)`

```bash
STAMP="$(date +%Y%m%d_%H%M%S)"

STRICT_REPLACE_CKPT="$ROOT/debug_output/_tmp_strict_stageB_finalstate_20260427_080658/stageB_strict/replace/checkpoints/ckpt_last_replace_strictB_20260427_080803.pth"
DIRECT_POSE_DONOR_STEP0="$ROOT/debug_output/_tmp_strict_contract_fullchain_preflight_20260426_173158/70R_lr_probe/lr1e4_step20/checkpoints/ckpt_step_000000_WalkF_stage7_70R_lr1e4_step20_20260426_173158.pth"

BRANCH70R_ROOT="$ROOT/debug_output/_tmp_70r_next_denseckpt_${STAMP}"
BRANCH70R_NAME="70R_next_denseckpt_${STAMP}"

python3 tools/run_strict_70r_trunkfull_probe.py \
  --source-replace-ckpt "$STRICT_REPLACE_CKPT" \
  --run-root "$BRANCH70R_ROOT" \
  --run-name "$BRANCH70R_NAME" \
  --lr 1e-4 \
  --epochs 1 \
  --steps-per-epoch 60 \
  --save-step-ckpts 0,1,20 \
  --eval-steps 20,last \
  --tensor-donor-ckpt "$DIRECT_POSE_DONOR_STEP0" \
  --transplant-prefix direct_pose_
```

关键产物：

- `70R last ckpt`:
  - `"$BRANCH70R_ROOT/checkpoints/ckpt_last_${BRANCH70R_NAME}.pth"`
- `70R eval`:
  - `"$BRANCH70R_ROOT/evals/step_last/group_summary.json"`
- `70R summary`:
  - `"$BRANCH70R_ROOT/probe_summary.json"`

### 15.3 Step B — 从新的 `70R ckpt_last` 继续到 `lambda`（首版复现）

这一步复现的是首版 downstream：`71` 仍走 helper 默认 `lr=3e-4`。它适合复现 `debug_output/_tmp_70r_dense_to_lambda_20260504_r2`，但不是当前推荐的最优 71 配置。

```bash
NEXT70R_CKPT="$BRANCH70R_ROOT/checkpoints/ckpt_last_${BRANCH70R_NAME}.pth"
BRANCHDOWN_ROOT="$ROOT/debug_output/_tmp_70r_dense_to_lambda_${STAMP}"

python3 tools/run_strict_70r_to_lambda_downstream.py \
  --source-70r "$NEXT70R_CKPT" \
  --run-root "$BRANCHDOWN_ROOT"
```

这个 helper 现在已经内置处理 strict/current handoff 所需的 fingerprint refresh，覆盖：

- `70R -> 71`
- `71(last) -> 72`
- `72 -> lambda`

关键产物：

- `71 final`:
  - `"$BRANCHDOWN_ROOT/evals/71/group_summary.json"`
- `lambda final`:
  - `"$BRANCHDOWN_ROOT/evals/lambda/group_summary.json"`
- 全链汇总:
  - `"$BRANCHDOWN_ROOT/run_result.json"`

### 15.4 71 配置补充 — 推荐 `lr=1e-4`

如果目标是继续优化 71，而不是只复现首版 downstream，`71` 固定使用下面这组字段：

```json
{
  "lr": 0.0001,
  "epochs": 1,
  "steps_per_epoch": 120,
  "save_step_ckpts": "0,1,5,20,40,60,90,120",
  "direct_pose_leg_train_only": true,
  "direct_pose_nonleg_train_only": false,
  "direct_pose_leg_stopgrad_main": true,
  "direct_pose_leg_align_weight": 0.0,
  "strict_current_model_build": true
}
```

保留的 71 语义：

- `train_direct_pose=true`
- `train_lambda_head=false`
- `direct_pose_leg_enable=true`
- `direct_pose_leg_mode=so3`
- `direct_pose_leg_detach_feat=true`
- `direct_pose_loss_leg_split=true`
- `direct_pose_stepc_unified_leg_terminal=true`

选点规则：

- dense `70R(1e-4 / 60 / last)` 输入：优先取 `71 lr1e4 step90`
- sibling lowlr `70R(7e-5 / 60 / last)` 输入：优先取 `71 lr1e4 step120/last`
- 不建议把 `steps_per_epoch` 先缩到 `60`；已评估的 `step60` 不如后续 checkpoint

已验证指标：

| lane | 71 ckpt | all mean | all p95 | leg mean | leg p95 |
|---|---|---:|---:|---:|---:|
| dense `70R 1e-4/60/last` | `lr1e4 step90` | 0.102371 | 0.336560 | 0.190973 | 0.504251 |
| lowlr `70R 7e-5/60/last` | `lr1e4 step120/last` | 0.097820 | 0.325507 | 0.183066 | 0.464128 |
| lowlr + downstream `72/lambda` | `lambda final` | 0.093424 | 0.309132 | 0.158343 | 0.417908 |

对应产物：

- dense/lowlr 71 lr1e4 probe:
  - `debug_output/_tmp_71_lr1e4_probe_20260504/summary.md`
- lowlr best downstream:
  - `debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/run_result.json`

### 15.5 如何嵌回 fresh chain 记账

如果你只是想在这份 fresh runbook 语义下记录这个实验，推荐把链路记成：

- 首版复现：`stage6 -> 70a -> warmstart copy -> replace -> 70R(1e-4, 60, pick last) -> 71(lr=3e-4, pick last) -> 72 -> lambda`
- 当前推荐：`stage6 -> 70a -> warmstart copy -> replace -> 70R(1e-4, 60, pick last) -> 71(lr=1e-4, pick step90 for dense) -> 72 -> lambda`

这里有两个边界要注意：

- 这不是把 §9 的 canonical `70R s180` 原地改掉，而是新增一个 parallel branch
- downstream 入口仍然用 `70R ckpt_last`，不需要改成 `step20`

### 15.6 当前建议

基于 2026-05-04 这轮结果，如果只在 `70R` 层做一个最小替换，当前最值得保留的是：

- `70R = lr=1e-4, epochs=1, steps_per_epoch=60, pick last`

如果允许同时改 71，当前更推荐：

- `70R = lr=1e-4, epochs=1, steps_per_epoch=60, pick last`
- `71 = lr=1e-4, epochs=1, steps_per_epoch=120`
- dense 70R 输入下 pick `71 step90`

原因是它满足三点：

- `70R` 本层比 fresh-chain 旧 `70R last` 更强
- 改善可以稳定传到 `71` 和 `lambda`，不是只停留在 `70R` 单点
- `71 lr=1e-4` 同时改善 `all mean / all p95 / leg mean / leg p95`，比 `lr=3e-4` 的 tail-only calibrator 更稳
