from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_smoke_injection_metadata_contract_angvel_target_slice_missing() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "debug_output" / "_tmp_action_handoff_injection_cli_smoke_test_20260525"
    cmd = [
        sys.executable,
        "-m",
        "train.validate.run_freerun_cycles",
        "--model",
        "debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/lambda/checkpoints/ckpt_last_WalkF_stage7_lambda_from_lowlr72_lr1e4_20260504.pth",
        "--teacher",
        "validate/teacher_batches/Walk_F_teacher.json",
        "--bundle",
        "raw_data/processed_data/norm_template.json",
        "--npz-root",
        "raw_data/processed_data",
        "--out",
        str(out_dir),
        "--rounds",
        "4",
        "--device",
        "auto",
        "--force",
        "--lambda_fusion_apply",
        "--pose_hist_source",
        "buffer",
        "--log_contacts",
        "--export_keybone_pos_err",
        "--inject-turn-npz",
        "raw_data/processed_data/Walk_R_To_L.npz",
        "--inject-at-step",
        "40",
        "--inject-from-step",
        "0",
        "--inject-fields",
        "rootvel,rot6d,angvel",
        "--inject-label",
        "Walk_F_to_Walk_R_To_L_N40",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"

    result_json = out_dir / "Walk_F_freerun_cycles.json"
    assert result_json.is_file(), f"missing output json: {result_json}"
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    records = payload.get("injection_apply_records")
    assert isinstance(records, list) and len(records) == 1
    rec = records[0]
    fields = {str(item.get("field")): item for item in rec.get("fields_applied", [])}

    rootvel = fields.get("rootvel")
    rot6d = fields.get("rot6d")
    angvel = fields.get("angvel")
    assert rootvel is not None and rot6d is not None and angvel is not None

    assert rootvel.get("requested") is True
    assert rootvel.get("applied") is True
    assert rootvel.get("reason") == "applied"

    assert rot6d.get("requested") is True
    assert rot6d.get("applied") is True
    assert rot6d.get("reason") == "applied"

    assert angvel.get("requested") is True
    assert angvel.get("applied") is False
    assert angvel.get("reason") == "target_slice_missing"
    assert angvel.get("target_slice") is None
