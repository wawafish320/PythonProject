from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .profile import compute_total_epochs, compute_batch_size, compute_base_lr

REMOVED_TOPLEVEL_KEYS: tuple[str, ...] = (
    "diag_input_stats",
    "eval_horizon",
    "eval_warmup",
    "foot_contact_threshold",
    "freerun_horizon",
    "freerun_horizon_ramp_epochs",
    "freerun_weight",
    "monitor_batches",
    "patience",
    "tf_warmup_epochs",
)

REMOVED_STAGE_ROOT_KEYS: tuple[str, ...] = (
    "teacher_rot_noise_deg_start",
    "teacher_rot_noise_deg_end",
    "teacher_rot_noise_prob_start",
    "teacher_rot_noise_prob_end",
    "targets",
)

REMOVED_STAGE_PARAM_KEYS: tuple[str, ...] = (
    "freerun_horizon",
    "freerun_weight",
    "teacher_rot_noise_deg",
    "teacher_rot_noise_prob",
    "input_step_noise_prob",
    "input_noise_profile",
)

REMOVED_STAGE_TRAINER_KEYS: tuple[str, ...] = (
    "freerun_horizon",
    "freerun_weight",
)

STAGE_TEMPLATE: List[Dict[str, Any]] = [
    {
        "name": "stage1_teacher",
        "ratio": 0.3,
        "w_rot_local": 0.07,
        "tf": {"max": 1.0, "min": 1.0},
    },
    {
        "name": "stage2_mixed",
        "ratio": 0.4,
        "w_rot_local": 0.18,
        "tf": {"max": 0.75, "min": 0.35},
    },
    {
        "name": "stage3_stable",
        "ratio": 0.3,
        "w_rot_local": 0.35,
        "tf": {"max": 0.5, "min": 0.1},
    },
]


def _sanitize_stage_schedule(base_schedule: Any) -> list[Any]:
    if not isinstance(base_schedule, Sequence) or isinstance(base_schedule, (str, bytes)):
        return []
    cleaned: list[Any] = []
    for raw_stage in base_schedule:
        if not isinstance(raw_stage, Mapping):
            cleaned.append(raw_stage)
            continue

        stage = deepcopy(dict(raw_stage))
        for key in REMOVED_STAGE_ROOT_KEYS:
            stage.pop(key, None)

        params = stage.get("params")
        if isinstance(params, Mapping):
            params_clean = {k: v for k, v in dict(params).items() if k not in REMOVED_STAGE_PARAM_KEYS}
            if params_clean:
                stage["params"] = params_clean
            else:
                stage.pop("params", None)

        trainer_cfg = stage.get("trainer")
        if isinstance(trainer_cfg, Mapping):
            trainer_clean = {k: v for k, v in dict(trainer_cfg).items() if k not in REMOVED_STAGE_TRAINER_KEYS}
            if trainer_clean:
                stage["trainer"] = trainer_clean
            else:
                stage.pop("trainer", None)

        cleaned.append(stage)
    return cleaned


def _sanitize_base_cfg(base_cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    cfg = deepcopy(dict(base_cfg or {}))
    for key in REMOVED_TOPLEVEL_KEYS:
        cfg.pop(key, None)
    if "freerun_stage_schedule" in cfg:
        cfg["freerun_stage_schedule"] = _sanitize_stage_schedule(cfg.get("freerun_stage_schedule"))
    return cfg


class TrainingConfigBuilder:
    def __init__(self, base_cfg: Optional[Mapping[str, Any]] = None):
        self.base_cfg = _sanitize_base_cfg(base_cfg)

    def build(self, profile: Mapping[str, Any]) -> Dict[str, Any]:
        total_epochs = self.base_cfg.get("epochs") or compute_total_epochs(int(profile["total_frames"]))
        batch_size = self.base_cfg.get("batch") or compute_batch_size(float(profile["avg_seq_len"]))
        lr = self.base_cfg.get("lr") or compute_base_lr(
            int(profile["total_frames"]), float(profile["complexity"]), batch_size
        )

        base_schedule = self.base_cfg.get("freerun_stage_schedule")
        if base_schedule:
            stages = _sanitize_stage_schedule(base_schedule)
            refs = self._compute_reference_targets(profile)
        else:
            stages, refs = self._build_stage_schedule(profile, total_epochs)

        cfg = dict(self.base_cfg)
        cfg["dataset_profile"] = dict(profile)
        cfg["epochs"] = int(total_epochs)
        cfg["batch"] = int(batch_size)
        cfg["lr"] = float(round(lr, 6))
        cfg["freerun_stage_schedule"] = stages

        first_stage = stages[0] if stages else {}
        loss_cfg = (first_stage.get("loss") or {}) if isinstance(first_stage, Mapping) else {}
        loss_group_core = {}
        if isinstance(first_stage, Mapping):
            loss_groups = first_stage.get("loss_groups") or {}
            if isinstance(loss_groups, Mapping):
                core_cfg = loss_groups.get("core")
                if isinstance(core_cfg, Mapping):
                    loss_group_core = core_cfg

        cfg["w_rot_local"] = float(
            loss_cfg.get("w_rot_local", loss_group_core.get("w_rot_local", cfg.get("w_rot_local", 0.0)))
        )
        cfg.setdefault("tf_mode", "epoch_linear")
        cfg["tf_start_epoch"] = 1
        cfg["tf_end_epoch"] = max(2, int(total_epochs * 0.65))
        tf_cfg = (first_stage.get("tf") or {}) if isinstance(first_stage, Mapping) else {}
        cfg["tf_max"] = float(tf_cfg.get("max", cfg.get("tf_max", 1.0)))
        cfg["tf_min"] = float(tf_cfg.get("min", cfg.get("tf_min", 0.0)))
        cfg.setdefault("seq_len", int(profile["avg_seq_len"]))
        cfg.setdefault("strategy_meta", {})["reference_targets"] = refs
        return cfg

    def _build_stage_schedule(self, profile: Mapping[str, Any], total_epochs: int) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        dataset_refs = self._compute_reference_targets(profile)

        stages: List[Dict[str, Any]] = []
        cursor = 1
        for template in STAGE_TEMPLATE:
            length = max(1, round(total_epochs * template["ratio"]))
            start = cursor
            end = min(total_epochs, cursor + length - 1)
            cursor = end + 1

            stages.append(
                {
                    "range": [start, end],
                    "label": template["name"],
                    "loss": {"w_rot_local": float(template["w_rot_local"])},
                    "tf": dict(template["tf"]),
                }
            )

        stages[-1]["range"][1] = total_epochs
        return stages, dataset_refs

    @staticmethod
    def _compute_reference_targets(profile: Mapping[str, Any]) -> Dict[str, float]:
        posture_ref = max(1.2, float(profile["bone_angle_mean_deg"]) * 0.04)
        return {
            "yaw": float(profile["yaw_mean_deg"]),
            "root": float(profile["speed_mean"]),
            "rot": posture_ref,
        }
