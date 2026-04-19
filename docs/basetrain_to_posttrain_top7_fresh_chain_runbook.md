# Fresh Basetrain -> Top7 Clean-StepC Runbook

> Last updated: 2026-04-14  
> Scope: `tail top7 basetrain -> fresh ckpt_last donor -> stage6 -> 70a -> replace -> 70R -> 71 -> 72 -> lambda`  
> This runbook is for the **fresh basetrain donor** path and does **not** include legacy old-boundary detailed comparison.

---

## 1. TL;DR

这条链路的固定约束是：

- `basetrain` 使用  
  `config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json`
- `basetrain` donor 固定使用 fresh `ckpt_last_<run_name>.pth`，**不是** `ckpt_epoch_014.pth`
- `posttrain` 主链固定使用  
  `stage6 -> 70a -> warmstart copy -> replace -> 70R -> 71 -> 72 -> lambda`
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

S70R_RUN_NAME="WalkF_stage7_70R_from_fresh_tailk7_replace_cleanstepc_s180_${STAMP}"
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
    "direct_pose_stepc_unified_leg_terminal": True,
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
            "lr": 3e-4,
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

## 12. Notes / Caveats

- 这份 runbook 是 **fresh basetrain ckpt_last donor** 版本，不是 canonical `ep014 donor` 版本。
- `70R` 当前 runbook 直接调用 `tools/run_posttrain_nonleg_trunk_ablation.py`；本文件默认以当前 `PostTrainModelArtifacts` / `_save_posttrain_outputs(...)` API 为准，如接口再次漂移，再重新评估是否需要临时 shim。
- 如果你后续想把这条链路再封成单个 runner，优先把本文件中的变量名和 stage 命名直接复用。
