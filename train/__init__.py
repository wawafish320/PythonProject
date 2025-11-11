"""Top-level helpers for the train package."""

__all__ = ["reproject_rot6d", "rot6d_to_matrix", "angvel_vec_from_R_seq"]


def __getattr__(name):
    if name in __all__:
        from .training_MPL import reproject_rot6d, rot6d_to_matrix, angvel_vec_from_R_seq

        values = {
            "reproject_rot6d": reproject_rot6d,
            "rot6d_to_matrix": rot6d_to_matrix,
            "angvel_vec_from_R_seq": angvel_vec_from_R_seq,
        }
        return values[name]
    raise AttributeError(f"module 'train' has no attribute {name}")
