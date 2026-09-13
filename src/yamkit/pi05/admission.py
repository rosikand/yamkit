"""Host/rig-bound native π0.5 qualification checks; no hardware activation."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np

from yamkit.inference.mapping import YAM_NAMES
from yamkit.inference.standalone_service import stable_host_id
from yamkit.validation import finite_vector, vendor_joint_limits

from .contract import ACTION_TRANSFORM, CONTRACT_ID, PROFILE, build_id


def _projection_accounting_valid(result: dict) -> bool:
    """Expected endpoint transformations are distinct from unexpected SDK edits."""
    fields = ("projected_rows", "projected_gripper_values", "executed_projected_rows", "executed_projected_gripper_values")
    if any(type(result.get(name)) is not int or not 0 <= result[name] <= 3000 for name in fields):
        return False
    rows, values = result["projected_rows"], result["projected_gripper_values"]
    maximum = result.get("maximum_gripper_projection")
    lower, upper = ACTION_TRANSFORM["raw_anomaly_range"]
    return (rows <= 1500 and rows <= values <= 2 * rows
            and result["executed_projected_rows"] == rows and result["executed_projected_gripper_values"] == values
            and type(maximum) in (float, int) and 0 <= maximum <= max(-lower, upper - 1.0)
            and math.isfinite(maximum)
            and (maximum == 0) == (values == 0))


def rig_binding(rig_path: Path) -> dict:
    from yamkit.config import RigConfig

    rig = RigConfig.load(rig_path)
    schema = rig_observation_schema(rig)
    return {"host_id": stable_host_id(), "rig_sha256": hashlib.sha256(rig_path.read_bytes()).hexdigest(),
            "observation_schema": schema}


def rig_observation_schema(rig) -> dict:
    """Exact reviewed arm mapping and configured RGB payload shapes, no devices."""
    if rig.validate():
        raise ValueError("π0.5 requires a valid rig configuration before software qualification")
    if rig.control.home_speed <= 0:
        raise ValueError("π0.5 startup requires configured home motion to be enabled")
    if set(rig.cameras) != set(PROFILE.image_keys):
        raise ValueError("π0.5 reference requires exactly top, left_wrist and right_wrist rig cameras")
    images = {}
    for name in PROFILE.image_keys:
        config = rig.cameras[name]
        height, width = config.get("height"), config.get("width")
        if (type(height) is not int or type(width) is not int or not 1 <= height <= 720
                or not 1 <= width <= 1280 or config.get("fps") != 30):
            raise ValueError("π0.5 reference cameras need explicit bounded RGB dimensions at 30 Hz")
        images[name] = [height, width, 3]
    for side in ("left", "right"):
        arm = rig.arm(side + "_follower")
        if (arm.role != "follower" or arm.side not in (None, side)
                or arm.arm_type != "yam" or arm.gripper != "linear_4310"
                or arm.gripper_limits is None):
            raise ValueError("π0.5 mapping requires correctly sided YAM followers with calibrated LINEAR_4310 grippers")
    return {"state_names": list(YAM_NAMES), "images": images, "image_encoding": "rgb8", "crop": "none"}


def passive_target_validator(rig_path: Path):
    """Read configured SDK bounds only; never construct a robot or open a bus."""
    from yamkit.config import RigConfig

    rig = RigConfig.load(rig_path)
    rig_observation_schema(rig)
    bounds = []
    for name in ("left_follower", "right_follower"):
        arm = rig.arm(name)
        if arm.role != "follower" or not arm.has_motor_gripper or arm.gripper_limits is None:
            raise ValueError("π0.5 needs two configured followers with existing calibrated motor grippers")
        limits = vendor_joint_limits(arm.arm_type, arm.gripper).copy()
        if arm.joint_offsets is not None:
            limits += np.asarray(arm.joint_offsets)[:, None]
        bounds.extend(limits.tolist())
        bounds.append([0.0, 1.0])
    limits = np.array(bounds)

    def validate(target):
        if set(target) != set(YAM_NAMES):
            raise ValueError("π0.5 target must contain exactly both arms' fourteen values")
        values = finite_vector([target[name] for name in YAM_NAMES], 14, "π0.5 target")
        if np.any(values < limits[:, 0]) or np.any(values > limits[:, 1]):
            raise ValueError("π0.5 native target exceeds configured joint/gripper bounds; no clipping")

    return validate


def validate_qualification(report: dict, metadata: dict, *, task: str, rig_path: Path) -> None:
    """Admission never equates a model card or old cached ready flag with proof."""
    from .transport import validate_readiness

    validate_readiness(metadata)
    if (not isinstance(report, dict) or any(not isinstance(report.get(key, {}), dict)
                                           for key in ("integrated", "stop_proof", "direct_warm_round_trip_s", "robot_host"))):
        raise ValueError("π0.5 needs current passing native qualification for this host, rig, task and model session")
    now, result, stop = time.time(), report.get("integrated", {}), report.get("stop_proof", {})
    expires = report.get("expires_at")
    p95 = report.get("direct_warm_round_trip_s", {}).get("p95")
    if (report.get("qualified") is not True or report.get("reasons") != []
            or report.get("profile") != PROFILE.id or report.get("model_revision") != PROFILE.revision
            or report.get("controller_mode") != CONTRACT_ID or report.get("pi05_build_id") != build_id()
            or report.get("action_transform") != ACTION_TRANSFORM
            or result.get("action_transform") != ACTION_TRANSFORM or not _projection_accounting_valid(result)
            or report.get("instance_id") != metadata.get("instance_id") or report.get("task") != task
            or report.get("robot_host") != rig_binding(rig_path)
            or report.get("observation_schema") != report.get("robot_host", {}).get("observation_schema")
            or report.get("hardware_tested") is not False or report.get("bounds_checked") is not True
            or report.get("completed_warm_samples") != 50 or result.get("completed_chunks") != 50
            or result.get("predicted_rows") != 1500 or result.get("completed_rows") != 1500
            or result.get("attempted_rows") != 1500
            or any(result.get(key) != 0 for key in ("dropped_rows", "modified_commands", "coherence_violations", "faults",
                                                   "unknown_partial_dispatches"))
            or stop.get("stop_requested_during_inflight_rpc") is not True or stop.get("commands_after_stop") != 0
            or stop.get("all_fake_robots_released") is not True
            or type(p95) not in (float, int) or not math.isfinite(p95) or not 0 <= p95 <= 1.6
            or type(expires) not in (float, int) or not math.isfinite(expires) or expires <= now
            or expires != metadata.get("http_session_expires_at")
            or type(report.get("completed_at")) not in (float, int)
            or not 0 <= now - report["completed_at"] < 86400):
        raise ValueError("π0.5 needs current passing native qualification for this host, rig, task and model session")
