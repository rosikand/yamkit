"""Passive exact-host/statistics/service admission; no robot or camera access."""

import copy
import hashlib
import json

import numpy as np
import pytest

from yamkit.inference.mapping import YAM_NAMES
from yamkit.openpi import admission
from yamkit.openpi.interface import CONTRACT_ID
from yamkit.openpi.yam_candidate import CandidateQuantiles, CandidateStatistics


def statistics():
    state = CandidateQuantiles(np.zeros(14), np.ones(14), "unit-test paired saved corpus", "a" * 64)
    return CandidateStatistics(state, state)


@pytest.fixture
def evidence(monkeypatch, tmp_path):
    now = 10000.0
    metadata = {"instance_id": "test-official-instance", "session_expires_at": now + 1000,
                "checkpoint": "gs://openpi-assets/checkpoints/pi05_base", "runtime_revision": "pinned",
                "warm_signatures": ["test-task-shape"], "busy": False, "runtime_provenance": {"unused": True}}
    binding = {"host_id": "test-lenovo", "rig_sha256": "a" * 64,
               "bounds_source_sha256": "b" * 64, "max_joint_speed": 3., "max_gripper_speed": 3.}
    stats = statistics()
    report = {
        "profile": "pi05-base", "qualified": True, "reasons": [], "hardware_tested": False,
        "controller_mode": CONTRACT_ID, "adapter_build_id": "test-build",
        "statistics_sha256": stats.metadata()["candidate_statistics_sha256"],
        "service_identity": admission.service_binding(metadata), "task": "test canonical task",
        "robot_host": binding, "completed_warm_samples": 50, "direct_chunks_validated": 50,
        "bounds_checked": True, "completed_at": now - 1, "expires_at": now + 1000,
        "direct_warm_round_trip_s": {"sample_count": 50, "p95": 0.2},
        "integrated": {"completed_chunks": 50, "admitted_chunks": 50, "inference_calls": 50,
                       "stop_requested": False, "predicted_rows": 2500, "admitted_rows": 1250,
                       "completed_rows": 1250, "intended_unused_rows": 1250, "uncompleted_committed_rows": 0,
                       "attempted_points": 1300, "completed_points": 1300, "faults": 0,
                       "inserted_transition_points": 50, "rows_discarded_on_stop_or_deadline": 0,
                       "rows_discarded_on_fault": 0, "late_response_rows_discarded": 0, "stopped_rpc_exceptions": 0,
                       "modified_commands": 0, "coherence_violations": 0, "unknown_partial_dispatches": 0,
                       "invalid_chunks": 0, "reordered_rows": 0, "prefix_dropped_rows": 0},
        "stop_proof": {"stop_requested_during_inflight_rpc": True, "commands_after_stop": 0,
                       "total_fake_commands": 0, "cancelled_before_transport_return": True,
                       "all_fake_robots_released": True,
                       "execution": {"inference_calls": 1, "stop_requested": True, "completed_points": 0,
                                     "attempted_points": 0, "faults": 0, "stopped_rpc_exceptions": 1,
                                     "late_response_rows_discarded": 0}},
    }
    monkeypatch.setattr(admission.time, "time", lambda: now)
    monkeypatch.setattr(admission, "adapter_build_id", lambda: "test-build")
    monkeypatch.setattr(admission, "rig_contract", lambda _: copy.deepcopy(binding))
    args = {"task": "test canonical task", "rig_path": tmp_path / "rig.yaml", "statistics": stats}
    return report, metadata, args


def test_complete_current_evidence_matches_exact_host_task_service_and_statistics(evidence):
    report, metadata, args = evidence
    admission.validate_qualification(report, metadata, **args)


@pytest.mark.parametrize("key,value", [
    ("qualified", False), ("qualified", 1), ("reasons", ["failed"]), ("hardware_tested", True),
    ("profile", "pi05-yam"), ("controller_mode", "reference"), ("adapter_build_id", "old-build"),
    ("statistics_sha256", "b" * 64), ("service_identity", {}), ("task", "another task"),
    ("robot_host", {}), ("completed_warm_samples", 49), ("direct_chunks_validated", 49),
    ("bounds_checked", False), ("completed_at", 10001), ("completed_at", -100000),
    ("completed_at", float("nan")), ("expires_at", 9999), ("expires_at", 12000),
    ("expires_at", float("inf")), ("expires_at", True),
])
def test_stale_mismatched_or_unqualified_records_cannot_pass(evidence, key, value):
    report, metadata, args = evidence
    report[key] = value
    with pytest.raises(ValueError, match="qualification"):
        admission.validate_qualification(report, metadata, **args)


@pytest.mark.parametrize("key,value", [
    ("completed_chunks", 49), ("predicted_rows", 2499), ("admitted_rows", 1249), ("completed_rows", 1249),
    ("intended_unused_rows", 1251), ("uncompleted_committed_rows", 1), ("completed_points", 1249),
    ("completed_points", True), ("attempted_points", 1301), ("faults", 1), ("modified_commands", 1),
    ("coherence_violations", 1), ("unknown_partial_dispatches", 1), ("invalid_chunks", 1),
    ("reordered_rows", 1), ("prefix_dropped_rows", 1), ("admitted_chunks", 49), ("inference_calls", 51),
    ("stop_requested", True), ("rows_discarded_on_stop_or_deadline", 1), ("rows_discarded_on_fault", 1),
    ("late_response_rows_discarded", 50), ("stopped_rpc_exceptions", 1), ("inserted_transition_points", 49),
])
def test_execution_evidence_must_prove_complete_unmodified_prefixes(evidence, key, value):
    report, metadata, args = evidence
    report["integrated"][key] = value
    with pytest.raises(ValueError, match="qualification"):
        admission.validate_qualification(report, metadata, **args)


@pytest.mark.parametrize("key,value", [
    ("stop_requested_during_inflight_rpc", False), ("commands_after_stop", 1), ("total_fake_commands", 1),
    ("cancelled_before_transport_return", False), ("all_fake_robots_released", False),
])
def test_stop_proof_cannot_be_replaced_with_ready_flag(evidence, key, value):
    report, metadata, args = evidence
    report["stop_proof"][key] = value
    with pytest.raises(ValueError, match="qualification"):
        admission.validate_qualification(report, metadata, **args)


@pytest.mark.parametrize("key,value", [("inference_calls", 0), ("stop_requested", False),
                                      ("completed_points", 1), ("attempted_points", 1), ("faults", 1),
                                      ("stopped_rpc_exceptions", 0)])
def test_actual_stop_executor_proof_is_required(evidence, key, value):
    report, metadata, args = evidence
    report["stop_proof"]["execution"][key] = value
    with pytest.raises(ValueError, match="qualification"):
        admission.validate_qualification(report, metadata, **args)


def test_stop_may_reject_returned_rows_without_transport_exception(evidence):
    report, metadata, args = evidence
    proof = report["stop_proof"]["execution"]
    proof["stopped_rpc_exceptions"] = 0
    proof["late_response_rows_discarded"] = 50
    admission.validate_qualification(report, metadata, **args)


@pytest.mark.parametrize("value", [-1, 1.60001, float("nan"), float("inf"), True, "0.2"])
def test_rpc_deadline_margin_is_not_relaxed(evidence, value):
    report, metadata, args = evidence
    report["direct_warm_round_trip_s"]["p95"] = value
    with pytest.raises(ValueError, match="qualification"):
        admission.validate_qualification(report, metadata, **args)


@pytest.mark.parametrize("key", ["integrated", "stop_proof", "direct_warm_round_trip_s", "robot_host"])
def test_malformed_evidence_rejected_cleanly(evidence, key):
    report, metadata, args = evidence
    report[key] = None
    with pytest.raises(ValueError, match="Malformed"):
        admission.validate_qualification(report, metadata, **args)


def test_service_warm_cache_is_not_identity_but_new_instance_is(evidence):
    report, metadata, args = evidence
    metadata.update(warm_signatures=["new prompt shape"], busy=True, runtime_provenance={"different": True})
    admission.validate_qualification(report, metadata, **args)
    metadata["instance_id"] = "replacement-service"
    with pytest.raises(ValueError, match="qualification"):
        admission.validate_qualification(report, metadata, **args)


def test_statistics_load_checks_immutable_content_digest(tmp_path):
    stats = statistics()
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps(stats.metadata()))
    loaded = admission.load_statistics(path)
    assert loaded.metadata()["candidate_statistics_sha256"] == stats.metadata()["candidate_statistics_sha256"]
    data = stats.metadata()
    data["state"]["q01"][0] += 0.1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity changed"):
        admission.load_statistics(path)


def test_statistics_missing_path_does_not_fetch_or_invent_values(tmp_path):
    with pytest.raises(ValueError, match="repository-local"):
        admission.load_statistics(tmp_path / "not-created.json")


def test_target_envelope_exact_order_and_bounds():
    limits = np.tile([-3., 3.], (14, 1))
    limits[[6, 13]] = [0, 1]
    validate = admission.target_validator({"bounds": limits.tolist()})
    target = dict(zip(YAM_NAMES, [0.] * 14, strict=True))
    validate(target)
    validate(dict(reversed(list(target.items()))))
    for key, value in ((YAM_NAMES[0], 3.001), (YAM_NAMES[6], -0.001), (YAM_NAMES[13], 1.001),
                       (YAM_NAMES[2], np.nan), (YAM_NAMES[4], float("inf"))):
        with pytest.raises(ValueError, match="bounds"):
            validate({**target, key: value})
    with pytest.raises(ValueError, match="fourteen"):
        validate({**target, "extra_joint.pos": 0.})


@pytest.mark.parametrize("bounds", [np.zeros((14, 2)), np.zeros((7, 2)), np.full((14, 2), np.nan)])
def test_malformed_target_envelope_never_sends(bounds):
    with pytest.raises(ValueError, match="envelope"):
        admission.target_validator({"bounds": bounds})


@pytest.fixture
def configured_rig(rig, tmp_path, monkeypatch):
    from yamkit.inference import standalone_service
    rig.cameras = {name: {"type": "opencv", "height": 480, "width": 640, "fps": 30,
                          "index_or_path": "/dev/never-open-this-camera"}
                   for name in ("top", "left_wrist", "right_wrist")}
    for side in ("left", "right"):
        rig.arm(side + "_follower").gripper_limits = [0., 6.5]
    path = tmp_path / "rig.yaml"
    rig.save(path)
    monkeypatch.setattr(standalone_service, "stable_host_id", lambda: "test-robot-host")
    return rig, path


def test_rig_contract_is_passive_and_uses_exact_current_limits(configured_rig, monkeypatch):
    from yamkit.arm import YamArm
    from yamkit.validation import vendor_joint_limits
    _rig, path = configured_rig
    monkeypatch.setattr(YamArm, "connect", lambda *_a, **_kw: pytest.fail("passive rig admission opened hardware"))
    value = admission.rig_contract(path)
    limits = np.asarray(value["bounds"])
    np.testing.assert_array_equal(limits[:6], vendor_joint_limits("yam", "linear_4310"))
    np.testing.assert_array_equal(limits[7:13], vendor_joint_limits("yam", "linear_4310"))
    np.testing.assert_array_equal(limits[[6, 13]], [[0, 1], [0, 1]])
    assert value["state_names"] == list(YAM_NAMES)
    assert value["rig_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert value["host_id"] == "test-robot-host"


@pytest.mark.parametrize("change", ["missing_calibration", "wrong_side", "wrong_camera_shape", "missing_camera",
                                     "disabled_home", "excess_joint_speed", "excess_gripper_speed"])
def test_unsupported_rig_rejected_before_device_construction(configured_rig, change):
    rig, path = configured_rig
    if change == "missing_calibration":
        rig.arm("right_follower").gripper_limits = None
    elif change == "wrong_side":
        rig.arm("right_follower").side = "left"
    elif change == "wrong_camera_shape":
        rig.cameras["top"]["height"] = 360
    elif change == "missing_camera":
        del rig.cameras["right_wrist"]
    elif change == "disabled_home":
        rig.control.home_speed = 0
    elif change == "excess_joint_speed":
        rig.control.max_joint_speed = 3.1
    else:
        rig.control.max_gripper_speed = 3.1
    rig.save(path)
    with pytest.raises(ValueError, match="OpenPI"):
        admission.rig_contract(path)
