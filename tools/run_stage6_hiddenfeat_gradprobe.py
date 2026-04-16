#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import training_MPL as tm

RUN_DATE = "20260313"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_stage6_hiddenfeat_gradprobe_{RUN_DATE}"
MODEL_OUT = OUT_ROOT / "__probe_models"


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    family: str
    source_config: Path
    resume_ckpt: Path
    detach_feat: bool


SPECS: Sequence[ProbeSpec] = (
    ProbeSpec(
        name="old_hidden_gradon",
        family="old",
        source_config=ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1" / "config_resolved.json",
        resume_ckpt=ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1" / "ckpt_last_exp_phase_DirectBranch_v1_d1.pth",
        detach_feat=False,
    ),
    ProbeSpec(
        name="old_hidden_gradoff",
        family="old",
        source_config=ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1" / "config_resolved.json",
        resume_ckpt=ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1" / "ckpt_last_exp_phase_DirectBranch_v1_d1.pth",
        detach_feat=True,
    ),
    ProbeSpec(
        name="cp015_hidden_gradon",
        family="cp015",
        source_config=ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1_cp015_tailk3" / "config_resolved.json",
        resume_ckpt=ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1_cp015_tailk3" / "ckpt_last_exp_phase_DirectBranch_v1_d1_cp015_tailk3.pth",
        detach_feat=False,
    ),
    ProbeSpec(
        name="cp015_hidden_gradoff",
        family="cp015",
        source_config=ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1_cp015_tailk3" / "config_resolved.json",
        resume_ckpt=ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1_cp015_tailk3" / "ckpt_last_exp_phase_DirectBranch_v1_d1_cp015_tailk3.pth",
        detach_feat=True,
    ),
)


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _fmt(v: Any, nd: int = 6) -> str:
    x = _safe_float(v)
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def _module_grad_norm(module: Optional[torch.nn.Module]) -> float:
    if module is None:
        return float("nan")
    total = None
    for param in module.parameters(recurse=True):
        if param.grad is None:
            continue
        g2 = param.grad.detach().float().pow(2).sum()
        total = g2 if total is None else total + g2
    if total is None:
        return 0.0
    return float(total.sqrt().detach().cpu())


def _param_grad_norm(param: Optional[torch.Tensor]) -> float:
    if param is None or (not torch.is_tensor(param)):
        return float("nan")
    if param.grad is None:
        return 0.0
    g = param.grad.detach().float()
    return float(g.pow(2).sum().sqrt().cpu())


def _merge_grad_norm(*vals: float) -> float:
    finite = [float(v) for v in vals if math.isfinite(_safe_float(v))]
    if not finite:
        return float("nan")
    return float(math.sqrt(sum(v * v for v in finite)))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_probe_config(spec: ProbeSpec) -> Path:
    cfg = json.loads(spec.source_config.read_text(encoding="utf-8"))
    cfg.pop("_trainbase_contacts_source_resolved", None)
    cfg["out"] = str(MODEL_OUT)
    cfg["run_name"] = spec.name
    cfg["resume"] = str(spec.resume_ckpt)
    cfg["direct_pose_enable"] = True
    cfg["direct_pose_detach_plan"] = True
    cfg["direct_pose_feat_source"] = "hidden"
    cfg["direct_pose_detach_feat"] = bool(spec.detach_feat)
    cfg_path = OUT_ROOT / "configs" / f"{spec.name}.json"
    _write_json(cfg_path, cfg)
    return cfg_path


def _build_train_ctx_from_config(cfg_path: Path) -> Any:
    saved = list(sys.argv)
    try:
        sys.argv = ["training_MPL.py", "--config_json", str(cfg_path)]
        return tm._build_train_components()
    finally:
        sys.argv = saved


def _first_batch(loader: Any) -> Any:
    try:
        return next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("empty train loader; cannot run grad probe") from exc


def _pick_batch_tensor(batch: Dict[str, Any], keys: Sequence[str], device: torch.device) -> Optional[torch.Tensor]:
    for key in keys:
        value = batch.get(key)
        if value is None:
            continue
        try:
            tensor = value.to(device)
            return tensor if tensor.dtype == torch.float32 else tensor.float()
        except Exception:
            continue
    return None


def _run_one_probe(spec: ProbeSpec) -> Dict[str, Any]:
    cfg_path = _build_probe_config(spec)
    train_ctx = _build_train_ctx_from_config(cfg_path)
    train_data = tm._build_train_loaders(train_ctx)
    model_artifacts = tm._build_train_model(train_ctx, train_data)
    tm._prepare_train_model_runtime(train_ctx, train_data, model_artifacts)
    build_artifacts = tm._build_train_loss_and_trainer(train_ctx, train_data, model_artifacts)
    dataset_artifacts = tm.build_and_attach_dataset_runtime(
        build_artifacts.trainer,
        train_data.ds_train,
        bundle_path=train_ctx.args.bundle_json,
    )
    runtime_cfg = tm._resolve_trainer_runtime_config(
        args=train_ctx.args,
        trainer=build_artifacts.trainer,
        dataset_artifacts=dataset_artifacts,
        norm_template_path=train_ctx.norm_template_path,
        bundle_json_path=build_artifacts.bundle_json_path,
        out_dir=train_ctx.out_dir,
        resolved_config=build_artifacts.resolved_config,
        run_name=train_ctx.run_name,
    )
    tm._apply_trainer_runtime_config(build_artifacts.trainer, runtime_cfg)

    model = build_artifacts.model
    loss_fn = build_artifacts.loss_fn
    trainer = build_artifacts.trainer
    batch = _first_batch(train_data.train_loader)
    device = train_ctx.device

    state_seq = _pick_batch_tensor(batch, ("motion", "X", "x_in_features"), device)
    gt_seq = _pick_batch_tensor(batch, ("gt_motion", "Y", "y_out_features", "y_out_seq"), device)
    if state_seq is None or gt_seq is None:
        raise RuntimeError(f"{spec.name}: batch missing motion/gt tensors")

    cond_seq = _pick_batch_tensor(batch, ("cond_in",), device)
    cond_raw_seq = _pick_batch_tensor(batch, ("cond_tgt_raw",), device)
    contacts_seq = _pick_batch_tensor(batch, ("contacts",), device)
    angvel_seq = _pick_batch_tensor(batch, ("angvel",), device)
    pose_hist_seq = _pick_batch_tensor(batch, ("pose_hist",), device)
    cond_norm_mu = _pick_batch_tensor(batch, ("cond_norm_mu",), device)
    cond_norm_std = _pick_batch_tensor(batch, ("cond_norm_std",), device)
    time_base = _pick_batch_tensor(batch, ("start",), device)

    model.train()
    model.zero_grad(set_to_none=True)
    preds_dict, _ = trainer._rollout_sequence(
        state_seq,
        cond_seq,
        cond_raw_seq,
        contacts_seq=contacts_seq,
        angvel_seq=angvel_seq,
        pose_hist_seq=pose_hist_seq,
        gt_seq=gt_seq,
        cond_norm_mu=cond_norm_mu,
        cond_norm_std=cond_norm_std,
        mode="mixed",
        tf_ratio=1.0,
        time_base=time_base,
    )

    direct = preds_dict.get("out_direct", None) if isinstance(preds_dict, dict) else None
    if not torch.is_tensor(direct):
        raise RuntimeError(f"{spec.name}: rollout produced no out_direct tensor")
    payload = loss_fn._compute_direct_pose_payload(direct, gt_seq, deg_per_rad=(180.0 / math.pi))
    if payload is None:
        raise RuntimeError(f"{spec.name}: direct-only payload is None")
    direct_objective, extra = payload
    if not torch.is_tensor(direct_objective):
        raise RuntimeError(f"{spec.name}: direct-only objective is not a tensor")

    direct_objective.backward()

    contact_plan_grad = _merge_grad_norm(
        _module_grad_norm(getattr(model, "contact_plan_cell", None)),
        _module_grad_norm(getattr(model, "contact_plan_head", None)),
        _module_grad_norm(getattr(model, "contact_plan_time_head", None)),
        _module_grad_norm(getattr(model, "contact_plan_phase_head", None)),
        _module_grad_norm(getattr(model, "contact_plan_init_head", None)),
        _param_grad_norm(getattr(model, "contact_plan_init_z", None)),
    )
    direct_head = getattr(model, "direct_pose_head", None)
    head_in_features = None
    if isinstance(direct_head, torch.nn.Sequential) and len(direct_head) > 0 and isinstance(direct_head[0], torch.nn.Linear):
        head_in_features = int(direct_head[0].in_features)

    result = {
        "name": spec.name,
        "family": spec.family,
        "source_config": str(spec.source_config),
        "probe_config": str(cfg_path),
        "resume_ckpt": str(spec.resume_ckpt),
        "feat_source": str(getattr(model, "direct_pose_feat_source", "")),
        "detach_plan": bool(getattr(model, "direct_pose_detach_plan", False)),
        "detach_feat": bool(getattr(model, "direct_pose_detach_feat", False)),
        "direct_head_in_features": head_in_features,
        "cond_dim": int(getattr(model, "cond_dim", 0) or 0),
        "hidden_dim": int(getattr(model, "hidden_dim", 0) or 0),
        "contact_dim": int(getattr(model, "contact_dim", 0) or 0),
        "direct_only_loss": float(direct_objective.detach().cpu().item()),
        "direct_extra": {k: _safe_float(v) for k, v in dict(extra or {}).items()},
        "grad_norms": {
            "direct_pose_head": _module_grad_norm(getattr(model, "direct_pose_head", None)),
            "shared_encoder": _module_grad_norm(getattr(model, "shared_encoder", None)),
            "contact_plan": contact_plan_grad,
        },
    }
    grads = result["grad_norms"]
    shared_grad = _safe_float(grads.get("shared_encoder"))
    head_grad = _safe_float(grads.get("direct_pose_head"))
    if spec.detach_feat:
        passed = bool(head_grad > 0.0 and shared_grad <= 1e-12)
    else:
        passed = bool(head_grad > 0.0 and shared_grad > 1e-8)
    result["pass"] = passed
    return result


def _render_md(results: Sequence[Dict[str, Any]], family_summary: Dict[str, Any], overall_pass: bool) -> str:
    lines: List[str] = []
    lines.append("# Hidden-feature direct grad probe")
    lines.append("")
    lines.append("- purpose: verify `direct_pose_feat_source=hidden` creates a real backbone gradient path, and `direct_pose_detach_feat=true` removes only that path")
    lines.append("- direct-only loss: yes")
    lines.append(f"- overall_pass: {'yes' if overall_pass else 'no'}")
    lines.append("")
    lines.append("| lane | family | detach_feat | feat_source | grad direct_pose_head | grad shared_encoder | grad contact_plan | pass |")
    lines.append("|---|---|---|---|---:|---:|---:|---|")
    for row in results:
        grads = row["grad_norms"]
        lines.append(
            f"| {row['name']} | {row['family']} | {row['detach_feat']} | {row['feat_source']} | "
            f"{_fmt(grads.get('direct_pose_head'))} | {_fmt(grads.get('shared_encoder'))} | {_fmt(grads.get('contact_plan'))} | "
            f"{'yes' if row.get('pass') else 'no'} |"
        )
    lines.append("")
    lines.append("## Family checks")
    lines.append("")
    for family, payload in family_summary.items():
        on = payload["gradon"]
        off = payload["gradoff"]
        on_shared = _safe_float(on["grad_norms"]["shared_encoder"])
        off_shared = _safe_float(off["grad_norms"]["shared_encoder"])
        ratio = float("nan")
        if math.isfinite(on_shared) and abs(on_shared) > 1e-12 and math.isfinite(off_shared):
            ratio = off_shared / on_shared
        head_live = "yes" if _safe_float(off["grad_norms"]["direct_pose_head"]) > 0.0 else "no"
        lines.append(
            f"- `{family}`: gradon shared={_fmt(on_shared)}; gradoff shared={_fmt(off_shared)}; "
            f"off/on={_fmt(ratio)}; head stays live={head_live}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    results = [_run_one_probe(spec) for spec in SPECS]
    by_name = {row["name"]: row for row in results}
    family_summary = {
        "old": {"gradon": by_name["old_hidden_gradon"], "gradoff": by_name["old_hidden_gradoff"]},
        "cp015": {"gradon": by_name["cp015_hidden_gradon"], "gradoff": by_name["cp015_hidden_gradoff"]},
    }

    overall_pass = bool(all(row.get("pass") for row in results))
    payload = {
        "run_date": RUN_DATE,
        "out_root": str(OUT_ROOT),
        "overall_pass": overall_pass,
        "criteria": {
            "gradon": "direct_pose_head > 0 and shared_encoder > 1e-8",
            "gradoff": "direct_pose_head > 0 and shared_encoder <= 1e-12",
        },
        "results": results,
        "family_summary": family_summary,
    }
    _write_json(OUT_ROOT / "hiddenfeat_backbone_grad_probe.json", payload)
    (OUT_ROOT / "hiddenfeat_backbone_grad_probe.md").write_text(
        _render_md(results, family_summary, overall_pass),
        encoding="utf-8",
    )
    print(f"[OK] wrote {OUT_ROOT / 'hiddenfeat_backbone_grad_probe.json'}")
    print(f"[OK] wrote {OUT_ROOT / 'hiddenfeat_backbone_grad_probe.md'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
