"""Small numeric validators used before hardware operations."""

from functools import cache
from numbers import Real

import numpy as np


def finite_scalar(value, name: str, *, minimum: float | None = None, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name}: expected a finite number")  # noqa: TRY004 — uniform validation API
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name}: expected a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name}: must be positive")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}: must be >= {minimum}")
    return value


def finite_vector(value, size: int, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if raw.shape != (size,) or raw.dtype.kind not in "iuf":
            raise ValueError
        out = raw.astype(float, copy=True)
        if not np.all(np.isfinite(out)):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name}: expected exactly {size} finite numeric values in a flat vector") from None
    return out


@cache
def vendor_joint_limits(arm_type: str, gripper: str) -> np.ndarray:
    """The pinned SDK's configured raw-coordinate bounds; no hardware is opened."""
    from i2rt.robots.get_robot import get_yam_joint_limits
    from i2rt.robots.utils import ArmType, GripperType

    limits = np.asarray(get_yam_joint_limits(ArmType.from_string_name(arm_type), GripperType.from_string_name(gripper)), dtype=float)
    if limits.shape != (6, 2) or not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("invalid vendored joint bounds")
    limits.setflags(write=False)
    return limits
