"""Top-level helpers for the train package."""

# 暴露常用工具，外部可通过 `from train import xxx` 直接获取
__all__ = [
    # 几何工具
    "reproject_rot6d",
    "rot6d_to_matrix",
    "angvel_vec_from_R_seq",
    "geodesic_R",
    "compose_rot6d_delta",
    "root_relative_matrices",
    "so3_log_map",
    # 评估工具
    "evaluate_teacher",
    "evaluate_freerun",
    "FreeRunSettings",
    # 布局工具
    "parse_layout_entry",
    "normalize_layout",
    "canonicalize_state_layout",
    # IO 工具
    "load_soft_contacts_from_json",
    "direction_yaw_from_array",
    "velocity_yaw_from_array",
]


def __getattr__(name):
    if name in __all__:
        if name in {
            "reproject_rot6d",
            "rot6d_to_matrix",
            "angvel_vec_from_R_seq",
            "geodesic_R",
            "compose_rot6d_delta",
            "root_relative_matrices",
            "so3_log_map",
        }:
            from .geometry import (
                reproject_rot6d,
                rot6d_to_matrix,
                angvel_vec_from_R_seq,
                geodesic_R,
                compose_rot6d_delta,
                root_relative_matrices,
                so3_log_map,
            )

            values = {
                "reproject_rot6d": reproject_rot6d,
                "rot6d_to_matrix": rot6d_to_matrix,
                "angvel_vec_from_R_seq": angvel_vec_from_R_seq,
                "geodesic_R": geodesic_R,
                "compose_rot6d_delta": compose_rot6d_delta,
                "root_relative_matrices": root_relative_matrices,
                "so3_log_map": so3_log_map,
            }
            return values[name]

        if name in {"evaluate_teacher", "evaluate_freerun", "FreeRunSettings"}:
            from .eval_utils import evaluate_teacher, evaluate_freerun, FreeRunSettings

            values = {
                "evaluate_teacher": evaluate_teacher,
                "evaluate_freerun": evaluate_freerun,
                "FreeRunSettings": FreeRunSettings,
            }
            return values[name]

        if name in {"parse_layout_entry", "normalize_layout", "canonicalize_state_layout"}:
            from .layout import (
                parse_layout_entry,
                normalize_layout,
                canonicalize_state_layout,
            )

            values = {
                "parse_layout_entry": parse_layout_entry,
                "normalize_layout": normalize_layout,
                "canonicalize_state_layout": canonicalize_state_layout,
            }
            return values[name]

        if name in {
            "load_soft_contacts_from_json",
            "direction_yaw_from_array",
            "velocity_yaw_from_array",
        }:
            from .io import (
                load_soft_contacts_from_json,
                direction_yaw_from_array,
                velocity_yaw_from_array,
            )

            values = {
                "load_soft_contacts_from_json": load_soft_contacts_from_json,
                "direction_yaw_from_array": direction_yaw_from_array,
                "velocity_yaw_from_array": velocity_yaw_from_array,
            }
            return values[name]

    raise AttributeError(f"module 'train' has no attribute {name}")
