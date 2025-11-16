from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .profile import compute_total_epochs, compute_batch_size, compute_base_lr

STAGE_TEMPLATE: List[Dict[str, Any]] = [
    {
        "name": "stage1_teacher",
        "ratio": 0.3,
        "motion": {
            "freerun_weight_scale": 0.0,
            "freerun_horizon_scale": 0.6,
            "latent_scale": 0.1,
        },
        "posture": {"scale": 0.2},
        "tf_max": 1.0,
        "yaw_ratio": (0.75, 1.25),
        "root_ratio": (0.02, 0.04),
        "rot_ratio": (0.9, 1.4),
    },
    {
        "name": "stage2_mixed",
        "ratio": 0.4,
        "motion": {
            "freerun_weight_scale": 0.35,
            "freerun_horizon_scale": 0.9,
            "latent_scale": 0.4,
        },
        "posture": {"scale": 0.65},
        "tf_max": 0.75,
        "yaw_ratio": (0.65, 1.1),
        "root_ratio": (0.018, 0.035),
        "rot_ratio": (0.75, 1.1),
    },
    {
        "name": "stage3_freerun",
        "ratio": 0.3,
        "motion": {
            "freerun_weight_scale": 0.65,
            "freerun_horizon_scale": 1.2,
            "latent_scale": 0.8,
        },
        "posture": {"scale": 1.0},
        "tf_max": 0.5,
        "yaw_ratio": (0.55, 0.95),
        "root_ratio": (0.015, 0.03),
        "rot_ratio": (0.65, 1.0),
    },
]


class TrainingConfigBuilder:
    def __init__(self, base_cfg: Optional[Mapping[str, Any]] = None):
        self.base_cfg = dict(base_cfg or {})

    def build(self, profile: Mapping[str, Any]) -> Dict[str, Any]:
        total_epochs = self.base_cfg.get("epochs") or compute_total_epochs(int(profile["total_frames"]))
        batch_size = self.base_cfg.get("batch") or compute_batch_size(float(profile["avg_seq_len"]))
        lr = self.base_cfg.get("lr") or compute_base_lr(
            int(profile["total_frames"]), float(profile["complexity"]), batch_size
        )

        stages, refs = self._build_stage_schedule(profile, total_epochs)
        cfg = dict(self.base_cfg)
        cfg["dataset_profile"] = dict(profile)
        cfg["epochs"] = int(total_epochs)
        cfg["batch"] = int(batch_size)
        cfg["lr"] = float(round(lr, 6))
        cfg["freerun_stage_schedule"] = stages
        cfg["freerun_weight"] = stages[0]["trainer"]["freerun_weight"]
        cfg["w_latent_consistency"] = stages[0]["trainer"]["w_latent_consistency"]
        cfg["w_fk_pos"] = stages[0]["loss"]["w_fk_pos"]
        cfg["w_rot_local"] = stages[0]["loss"]["w_rot_local"]
        cfg.setdefault("tf_mode", "epoch_linear")
        cfg["tf_start_epoch"] = 1
        cfg["tf_end_epoch"] = max(2, int(total_epochs * 0.65))
        cfg["tf_max"] = stages[0]["tf"]["max"]
        cfg["tf_min"] = 0.0
        cfg.setdefault("seq_len", int(profile["avg_seq_len"]))
        cfg.setdefault("freerun_horizon", stages[0]["trainer"]["freerun_horizon"])
        cfg.setdefault("freerun_horizon_ramp_epochs", max(4, int(total_epochs * 0.15)))
        cfg.setdefault("strategy_meta", {})["reference_targets"] = refs
        return cfg

    def _build_stage_schedule(self, profile: Mapping[str, Any], total_epochs: int) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        avg_seq_len = float(profile["avg_seq_len"])
        base_horizon = max(6, int(round(avg_seq_len * 0.2)))
        posture_ref = max(1.2, float(profile["bone_angle_mean_deg"]) * 0.04)
        dataset_refs = {
            "yaw": float(profile["yaw_mean_deg"]),
            "root": float(profile["speed_mean"]),
            "rot": posture_ref,
        }

        stages: List[Dict[str, Any]] = []
        cursor = 1
        for template in STAGE_TEMPLATE:
            length = max(1, round(total_epochs * template["ratio"]))
            start = cursor
            end = min(total_epochs, cursor + length - 1)
            cursor = end + 1
            motion = template["motion"]
            posture = template["posture"]

            freerun_horizon = max(4, int(round(base_horizon * motion["freerun_horizon_scale"])))
            freerun_weight = min(0.8, max(0.0, motion["freerun_weight_scale"] * 0.5))
            latent = min(0.6, max(0.0, motion["latent_scale"] * 0.3))
            posture_weight = min(0.8, max(0.01, posture["scale"] * 0.35))

            yaw_lo_ratio, yaw_hi_ratio = template["yaw_ratio"]
            root_lo_ratio, root_hi_ratio = template["root_ratio"]
            rot_lo_ratio, rot_hi_ratio = template["rot_ratio"]

            stages.append(
                {
                    "range": [start, end],
                    "label": template["name"],
                    "trainer": {
                        "freerun_weight": round(freerun_weight, 4),
                        "freerun_horizon": freerun_horizon,
                        "w_latent_consistency": round(latent, 4),
                    },
                    "loss": {
                        "w_fk_pos": round(posture_weight, 4),
                        "w_rot_local": round(posture_weight, 4),
                    },
                    "tf": {"max": template["tf_max"]},
                    "targets": {
                        "yaw": {"ref": dataset_refs["yaw"], "lo_ratio": yaw_lo_ratio, "hi_ratio": yaw_hi_ratio},
                        "root": {"ref": dataset_refs["root"], "lo_ratio": root_lo_ratio, "hi_ratio": root_hi_ratio},
                        "rot": {"ref": dataset_refs["rot"], "lo_ratio": rot_lo_ratio, "hi_ratio": rot_hi_ratio},
                    },
                }
            )

        stages[-1]["range"][1] = total_epochs
        return stages, dataset_refs
