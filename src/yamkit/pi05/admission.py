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

from .contract import CONTRACT_ID, PROFILE, build_id


def rig_binding(rig_path: Path) -> dict:
    return {"host_id": stable_host_id(), "rig_sha256": hashlib.sha256(rig_path.read_bytes()).hexdigest()}


def passive_target_validator(rig_path: Path):
    """Read configured SDK bounds only; never construct a robot or open a bus."""
    from yamkit.config import RigConfig

    rig = RigConfig.load(rig_path)
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
    now, result, stop = time.time(), report.get("integrated", {}), report.get("stop_proof", {})
    expires = report.get("expires_at")
    p95 = report.get("direct_warm_round_trip_s", {}).get("p95")
    if (report.get("qualified") is not True or report.get("reasons") != []
            or report.get("profile") != PROFILE.id or report.get("model_revision") != PROFILE.revision
            or report.get("controller_mode") != CONTRACT_ID or report.get("pi05_build_id") != build_id()
            or report.get("instance_id") != metadata.get("instance_id") or report.get("task") != task
            or report.get("robot_host") != rig_binding(rig_path)
            or report.get("hardware_tested") is not False or report.get("bounds_checked") is not True
            or report.get("completed_warm_samples") != 50 or result.get("completed_chunks") != 50
            or result.get("predicted_rows") != 1500 or result.get("completed_rows") != 1500
            or any(result.get(key) != 0 for key in ("dropped_rows", "modified_commands", "coherence_violations", "faults"))
            or stop.get("stop_requested_during_inflight_rpc") is not True or stop.get("commands_after_stop") != 0
            or stop.get("all_fake_robots_released") is not True
            or type(p95) not in (float, int) or not math.isfinite(p95) or not 0 <= p95 <= 1.6
            or type(expires) not in (float, int) or not math.isfinite(expires) or expires <= now
            or expires != metadata.get("http_session_expires_at")
            or type(report.get("completed_at")) not in (float, int)
            or not 0 <= now - report["completed_at"] < 86400):
        raise ValueError("π0.5 needs current passing native qualification for this host, rig, task and model session")
