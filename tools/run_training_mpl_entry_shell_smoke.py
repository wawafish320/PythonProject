#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.configuration.io import dump_json
import train.training_MPL as mpl


def _default_argv(out_dir: Path) -> list[str]:
    return [
        "--config_json",
        "config/exp_phase_mpl.clean.json",
        "--train_files",
        "raw_data/processed_data/Walk_F.npz",
        "--bundle_json",
        "raw_data/processed_data/norm_template.json",
        "--encoder_path",
        "models/motion_encoder_equiv_stageA.pt",
        "--out",
        str(out_dir),
        "--run_name",
        "basetrain_entry_shell_smoke",
        "--epochs",
        "1",
        "--batch",
        "1",
        "--num_workers",
        "0",
    ]


def _assert_shell_artifacts(*, train_ctx: Any, train_data: Any, model_artifacts: Any, build_artifacts: Any) -> dict[str, Any]:
    trainer = build_artifacts.trainer
    loss_fn = build_artifacts.loss_fn
    model = build_artifacts.model

    if int(train_data.dx) <= 0 or int(train_data.dy) <= 0:
        raise SystemExit("[FATAL] basetrain shell smoke resolved invalid train dims.")
    if getattr(model, "shared_encoder", None) is None:
        raise SystemExit("[FATAL] basetrain shell smoke failed to instantiate EventMotionModel core.")
    if bool(getattr(model, "contact_plan_enable", False)) and getattr(model, "frozen_encoder", None) is None:
        raise SystemExit("[FATAL] basetrain shell smoke lost frozen encoder bundle attach.")
    if getattr(trainer, "pose_hist_dim", None) is None:
        raise SystemExit("[FATAL] basetrain shell smoke lost dataset runtime attach on trainer.")
    if getattr(trainer, "pose_hist_mu", None) is None or getattr(trainer, "pose_hist_std", None) is None:
        raise SystemExit("[FATAL] basetrain shell smoke lost shared pose-history runtime attach.")
    if getattr(trainer, "trainbase_contacts_pretrain_affine_stats_spec", None) != getattr(
        train_ctx.args,
        "trainbase_contacts_pretrain_affine_stats",
        None,
    ):
        raise SystemExit("[FATAL] basetrain shell smoke drifted trainbase contact-pretrain raw spec mapping.")
    if float(getattr(trainer, "contacts_pretrain_clamp", 0.0)) != float(
        getattr(trainer, "trainbase_contacts_pretrain_clamp", 0.0)
    ):
        raise SystemExit("[FATAL] basetrain shell smoke drifted neutral contact-pretrain clamp mapping.")
    if getattr(trainer, "contacts_pretrain_affine_stats_spec", None) != getattr(
        trainer,
        "trainbase_contacts_pretrain_affine_stats_spec",
        None,
    ):
        raise SystemExit("[FATAL] basetrain shell smoke drifted neutral contact-pretrain raw spec mapping.")
    if getattr(trainer, "contacts_pretrain_affine", None) != getattr(trainer, "trainbase_contacts_pretrain_affine", None):
        raise SystemExit("[FATAL] basetrain shell smoke drifted neutral parsed contact-pretrain affine mapping.")
    if not bool(getattr(trainer, "contacts_pretrain_runtime_attached", False)):
        raise SystemExit("[FATAL] basetrain shell smoke lost contacts_pretrain_runtime_attached marker.")
    if getattr(loss_fn, "mu_y", None) is not getattr(trainer, "mu_y", None):
        raise SystemExit("[FATAL] basetrain shell smoke failed to sync loss_fn.mu_y from trainer.")
    if getattr(loss_fn, "std_y", None) is not getattr(trainer, "std_y", None):
        raise SystemExit("[FATAL] basetrain shell smoke failed to sync loss_fn.std_y from trainer.")

    return {
        "config_json": str(getattr(train_ctx.args, "config_json", None)),
        "train_files": list(train_ctx.train_paths),
        "run_name": str(train_ctx.run_name),
        "bundle_json": str(getattr(train_ctx.args, "bundle_json", "")),
        "encoder_path": str(getattr(train_ctx.args, "encoder_path", "")),
        "dims": {
            "dx": int(train_data.dx),
            "dy": int(train_data.dy),
            "dc": int(train_data.dc),
            "pose_hist_dim_raw": int(model_artifacts.pose_hist_dim_raw),
            "pose_hist_len_raw": int(model_artifacts.pose_hist_len_raw),
        },
        "model_paths": {
            "contact_plan_enable": bool(getattr(model, "contact_plan_enable", False)),
            "direct_pose_enable": bool(getattr(model, "direct_pose_enable", False)),
            "direct_pose_head": getattr(model, "direct_pose_head", None) is not None,
            "frozen_encoder_attached": getattr(model, "frozen_encoder", None) is not None,
            "frozen_contact_head_attached": getattr(model, "frozen_contact_head", None) is not None,
        },
        "trainer_runtime": {
            "yaw_forward_axis": int(getattr(trainer, "yaw_forward_axis", -1)),
            "pose_hist_dim": int(getattr(trainer, "pose_hist_dim", 0) or 0),
            "has_pose_hist_mu": getattr(trainer, "pose_hist_mu", None) is not None,
            "has_pose_hist_std": getattr(trainer, "pose_hist_std", None) is not None,
            "trainbase_contacts_pretrain_clamp": float(getattr(trainer, "trainbase_contacts_pretrain_clamp", 0.0)),
            "trainbase_contacts_pretrain_affine_stats_spec": getattr(trainer, "trainbase_contacts_pretrain_affine_stats_spec", None),
            "contacts_pretrain_clamp": float(getattr(trainer, "contacts_pretrain_clamp", 0.0)),
            "contacts_pretrain_affine_stats_spec": getattr(trainer, "contacts_pretrain_affine_stats_spec", None),
            "has_contacts_pretrain_affine": getattr(trainer, "contacts_pretrain_affine", None) is not None,
            "contacts_pretrain_runtime_attached": bool(getattr(trainer, "contacts_pretrain_runtime_attached", False)),
            "teacher_eval_max_batches": getattr(trainer, "teacher_eval_max_batches", None),
        },
        "loss_runtime": {
            "mu_y_synced": getattr(loss_fn, "mu_y", None) is getattr(trainer, "mu_y", None),
            "std_y_synced": getattr(loss_fn, "std_y", None) is getattr(trainer, "std_y", None),
            "meta_from_bundle": bool(getattr(loss_fn, "meta", None)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Step 2 basetrain entry shell smoke.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("debug_output/_training_mpl_entry_shell_smokes_20260418"),
        help="Directory for smoke summary artifact.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_argv = _default_argv(out_dir)

    train_ctx = mpl._build_train_components(smoke_argv)
    train_data = mpl._build_train_loaders(train_ctx)
    model_artifacts = mpl._build_train_model(train_ctx, train_data)
    mpl._prepare_train_model_runtime(train_ctx, train_data, model_artifacts)
    build_artifacts = mpl._build_train_loss_and_trainer(train_ctx, train_data, model_artifacts)
    mpl._attach_train_entry_runtime(train_ctx, train_data, build_artifacts)

    summary = _assert_shell_artifacts(
        train_ctx=train_ctx,
        train_data=train_data,
        model_artifacts=model_artifacts,
        build_artifacts=build_artifacts,
    )
    summary_path = out_dir / "training_mpl_entry_shell_smoke_summary.json"
    dump_json(summary_path, summary)
    print(f"[smoke] wrote summary: {summary_path}")
    print(
        "[smoke] "
        f"run_name={summary['run_name']} "
        f"dx={summary['dims']['dx']} dy={summary['dims']['dy']} dc={summary['dims']['dc']} "
        f"contact_plan={summary['model_paths']['contact_plan_enable']} "
        f"frozen_encoder={summary['model_paths']['frozen_encoder_attached']}"
    )


if __name__ == "__main__":
    main()
