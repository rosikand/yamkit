"""Passive OpenPI/YAM configuration, immutable statistics and host qualification."""

from __future__ import annotations

import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from yamkit.inference.mapping import YAM_NAMES
from yamkit.paths import ROOT

from .yam_candidate import CandidateQuantiles, CandidateStatistics

STATISTICS_RELATIVE = Path("data/openpi/yam/normalization.json")


def service_binding(metadata):
    return {key: value for key, value in metadata.items()
            if key not in ("warm_signatures", "busy", "runtime_provenance")}


def load_statistics(path: Path | None = None) -> CandidateStatistics:
    from yamkit.backend_workflow import _local_path

    path = _local_path(str(path or ROOT / STATISTICS_RELATIVE), root=ROOT)
    if not path.is_file() or not 0 < path.stat().st_size < 65536:
        raise ValueError("Install the reviewed repository-local experimental OpenPI YAM statistics")
    value = json.loads(path.read_text())
    result = CandidateStatistics(CandidateQuantiles(**value["state"]), CandidateQuantiles(**value["actions"]))
    if result.metadata()["candidate_statistics_sha256"] != value.get("candidate_statistics_sha256"):
        raise ValueError("Experimental YAM statistics identity changed")
    return result


def adapter_build_id() -> str:
    """Qualify only this adapter and its existing physical/capture seams, not MA2."""
    names = ("yam_candidate.py", "interface.py", "executor.py", "admission.py", "qualification.py",
             "rollout.py", "workflow.py", "artifacts.py")
    digest = hashlib.sha256(b"openpi-experimental-yam-execution-v1\0")
    for name in names:
        digest.update(name.encode() + b"\0" + Path(__file__).with_name(name).read_bytes())
    for name in ("src/yamkit/arm.py", "src/yamkit/config.py", "src/yamkit/validation.py",
                 "plugins/lerobot_robot_yamkit/lerobot_robot_yamkit/yam_follower.py"):
        digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
    return digest.hexdigest()


def rig_contract(rig_path: Path) -> dict:
    """Read YAML/XML only. No robot, encoder, CAN, discovery or camera constructor."""
    from yamkit.config import RigConfig
    from yamkit.inference.standalone_service import stable_host_id

    rig = RigConfig.load(rig_path)
    if rig.validate() or rig.control.home_speed <= 0:
        raise ValueError("OpenPI requires the valid configured rig and its bounded startup home")
    if any(not math.isfinite(v) or not 0 < v <= 3 for v in
           (rig.control.max_joint_speed, rig.control.max_gripper_speed)):
        raise ValueError("OpenPI interface requires configured joint/gripper speeds in (0,3]")
    cameras = ("top", "left_wrist", "right_wrist")
    if set(rig.cameras) != set(cameras) or any(
        (rig.cameras[n].get("height"), rig.cameras[n].get("width"), rig.cameras[n].get("fps")) != (480, 640, 30)
        for n in cameras
    ):
        raise ValueError("OpenPI YAM requires the existing three 640x480 RGB cameras at 30 Hz")
    source = ROOT / "third_party/i2rt/i2rt/robot_models/arm/yam/v1/yam.xml"
    elements = {e.get("name"): e for e in ET.parse(source).getroot().iter("joint") if e.get("range")}
    native = np.array([list(map(float, elements[f"joint{i}"].get("range").split())) for i in range(1, 7)])
    # Exact current get_yam_joint_limits convention. No narrowed/widened candidate bounds.
    native += np.array([-0.15, 0.15])
    bounds = []
    for side in ("left", "right"):
        arm = rig.arm(side + "_follower")
        if (arm.role != "follower" or arm.side not in (None, side) or arm.arm_type != "yam"
                or arm.gripper != "linear_4310" or arm.gripper_limits is None):
            raise ValueError("OpenPI requires the mapped, already calibrated YAM LINEAR_4310 followers")
        limits = native.copy()
        if arm.joint_offsets is not None:
            limits += np.asarray(arm.joint_offsets)[:, None]
        bounds.extend(limits.tolist())
        bounds.append([0., 1.])
    return {"host_id": stable_host_id(), "rig_sha256": hashlib.sha256(rig_path.read_bytes()).hexdigest(),
            "bounds_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "state_names": list(YAM_NAMES), "image_shapes": {n: [480, 640, 3] for n in cameras},
            "bounds": bounds, "max_joint_speed": rig.control.max_joint_speed,
            "max_gripper_speed": rig.control.max_gripper_speed}


def target_validator(binding: dict):
    limits = np.asarray(binding["bounds"], dtype=float)
    if limits.shape != (14, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Invalid passive YAM target envelope")

    def validate(target):
        if set(target) != set(YAM_NAMES):
            raise ValueError("OpenPI command needs exactly fourteen named YAM values")
        values = np.asarray([target[n] for n in YAM_NAMES])
        if (values.shape != (14,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all()
                or np.any(values < limits[:, 0]) or np.any(values > limits[:, 1])):
            raise ValueError("OpenPI target violates configured YAM joint/gripper bounds")

    return validate


def _exact_counts(value, expected):
    return all(type(value.get(key)) is int and value[key] == count for key, count in expected)


def complete_execution_proof(proof):
    """A chunk label cannot hide partial rows, extra RPCs, or unknown sends.

    Actuator subdivision is valid and expected: it must have internally exact
    accounting, not a fabricated zero-substep requirement. These are checks of
    collected software evidence, not additional operator preparation steps.
    """
    if not isinstance(proof, dict):
        return False
    expected = (("completed_chunks", 50), ("admitted_chunks", 50), ("inference_calls", 50),
                ("predicted_rows", 2500), ("admitted_rows", 1250), ("completed_rows", 1250),
                ("intended_unused_rows", 1250), ("uncompleted_committed_rows", 0),
                ("rows_discarded_on_stop_or_deadline", 0), ("rows_discarded_on_fault", 0),
                ("faults", 0), ("modified_commands", 0), ("coherence_violations", 0),
                ("unknown_partial_dispatches", 0), ("invalid_chunks", 0), ("reordered_rows", 0),
                ("prefix_dropped_rows", 0), ("late_response_rows_discarded", 0),
                ("stopped_rpc_exceptions", 0))
    if not _exact_counts(proof, expected) or proof.get("stop_requested") is not False:
        return False
    if any(type(proof.get(key)) is not int or proof[key] < 0 for key in
           ("completed_points", "attempted_points", "inserted_transition_points")):
        return False
    return (proof["attempted_points"] == proof["completed_points"]
            and proof["completed_points"] == proof["completed_rows"] + proof["inserted_transition_points"])


def stopped_execution_proof(proof):
    """Exactly one in-flight request is invalidated; no actuator call completes."""
    if not isinstance(proof, dict):
        return False
    if (not _exact_counts(proof, (("inference_calls", 1), ("completed_points", 0),
                                 ("attempted_points", 0), ("faults", 0)))
            or proof.get("stop_requested") is not True):
        return False
    invalidations = proof.get("stopped_rpc_exceptions")
    discarded = proof.get("late_response_rows_discarded")
    # A cancelled transport may raise, or a callback can return the now-unusable
    # complete 50-row result. Either path is valid; neither is a physical retry.
    return (type(invalidations) is int and type(discarded) is int
            and (invalidations, discarded) in ((1, 0), (0, 50)))


def validate_qualification(report, metadata, *, task, rig_path, statistics):
    """No cached ready flag can substitute for matching execution/Stop evidence."""
    if not isinstance(report, dict):
        raise ValueError("Current OpenPI execution qualification is required")  # noqa: TRY004 — uniform admission failure
    if any(not isinstance(report.get(key), dict) for key in
           ("integrated", "stop_proof", "direct_warm_round_trip_s", "robot_host")):
        raise ValueError("Malformed OpenPI qualification evidence")
    proof, stop = report.get("integrated", {}), report.get("stop_proof", {})
    completed, expiry = report.get("completed_at"), report.get("expires_at")
    p95 = report.get("direct_warm_round_trip_s", {}).get("p95")
    from .interface import CONTRACT_ID

    if (report.get("qualified") is not True or report.get("reasons") != []
            or report.get("hardware_tested") is not False or report.get("profile") != "pi05-base"
            or report.get("controller_mode") != CONTRACT_ID or report.get("adapter_build_id") != adapter_build_id()
            or report.get("statistics_sha256") != statistics.metadata()["candidate_statistics_sha256"]
            or report.get("service_identity") != service_binding(metadata) or report.get("task") != task
            or report.get("robot_host") != rig_contract(rig_path)
            or not _exact_counts(report, (("completed_warm_samples", 50), ("direct_chunks_validated", 50)))
            or report.get("bounds_checked") is not True or not complete_execution_proof(proof)
            or not _exact_counts(report["direct_warm_round_trip_s"], (("sample_count", 50),))
            or stop.get("stop_requested_during_inflight_rpc") is not True
            or not _exact_counts(stop, (("commands_after_stop", 0), ("total_fake_commands", 0)))
            or stop.get("cancelled_before_transport_return") is not True
            or stop.get("all_fake_robots_released") is not True
            or not stopped_execution_proof(stop.get("execution"))
            or type(p95) not in (float, int) or not math.isfinite(p95) or not 0 <= p95 <= 1.6
            or type(completed) not in (float, int) or not 0 <= time.time() - completed < 86400
            or type(expiry) not in (float, int) or not math.isfinite(expiry) or expiry <= time.time()
            or expiry != metadata.get("session_expires_at")):
        raise ValueError("OpenPI qualification does not match this host, rig, task, service and adapter")
