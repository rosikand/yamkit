"""Independent offline qualification/admission regressions; no GPU or devices."""

import copy
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from yamkit.inference.mapping import YAM_NAMES
from yamkit.openpi import admission, qualification
from yamkit.openpi.executor import OpenPiYamExecutor
from yamkit.openpi.interface import CONTRACT_ID, EXECUTION_CONTRACT
from yamkit.openpi.yam_candidate import CandidateQuantiles, CandidateStatistics


def observation(joint=0., pixel=0):
    state = np.full(14, joint, dtype=np.float64)
    state[[6, 13]] = .5
    return {"state": state, **{name: np.full((480, 640, 3), pixel, dtype=np.uint8)
                              for name in ("top", "left_wrist", "right_wrist")}}


@pytest.fixture
def statistics():
    low, high = np.full(14, -.01), np.full(14, .01)
    low[[6, 13]], high[[6, 13]] = 0., 1.
    return CandidateStatistics(CandidateQuantiles(np.full(14, -1), np.ones(14), "test-state", "a" * 64),
                               CandidateQuantiles(low, high, "test-action", "a" * 64))


@pytest.fixture
def host(monkeypatch):
    bounds = [[-3., 3.] for _ in range(14)]
    bounds[6] = bounds[13] = [0., 1.]
    value = {"host_id": "fake-unit-host", "rig_sha256": "rig-test", "bounds_source_sha256": "bounds-test",
             "state_names": list(YAM_NAMES), "bounds": bounds,
             "image_shapes": {name: [480, 640, 3] for name in ("top", "left_wrist", "right_wrist")},
             "max_joint_speed": 3., "max_gripper_speed": 3.}
    for module in (admission, qualification):
        monkeypatch.setattr(module, "rig_contract", lambda _path: copy.deepcopy(value))
        monkeypatch.setattr(module, "adapter_build_id", lambda: "adapter-test")
    return value


def metadata():
    return {"instance_id": "fake-unit-instance", "session_expires_at": time.time() + 3600,
            "profile": "pi05-base", "ready": True}


def full_proof():
    return {"controller_mode": CONTRACT_ID, "execution_contract": dict(EXECUTION_CONTRACT),
            "predicted_rows": 2500, "admitted_rows": 1250, "completed_rows": 1250,
            "completed_chunks": 50, "admitted_chunks": 50, "intended_unused_rows": 1250,
            "uncompleted_committed_rows": 0, "rows_discarded_on_stop_or_deadline": 0,
            "rows_discarded_on_fault": 0, "prefix_dropped_rows": 0, "reordered_rows": 0,
            "attempted_points": 1250, "completed_points": 1250, "inserted_transition_points": 0,
            "unknown_partial_dispatches": 0, "modified_commands": 0, "coherence_violations": 0,
            "faults": 0, "invalid_chunks": 0, "inference_calls": 50, "stop_requested": False,
            "late_response_rows_discarded": 0, "stopped_rpc_exceptions": 0,
            "projected_gripper_values": 0, "executed_projected_gripper_values": 0,
            "maximum_gripper_projection": 0., "roundoff_receipts": 0, "maximum_receipt_roundoff": 0.}


def passing_report(host, statistics, current):
    return {"qualified": True, "reasons": [], "hardware_tested": False, "profile": "pi05-base",
            "controller_mode": CONTRACT_ID, "adapter_build_id": "adapter-test",
            "statistics_sha256": statistics.metadata()["candidate_statistics_sha256"],
            "service_identity": admission.service_binding(current), "task": "cube", "robot_host": host,
            "completed_warm_samples": 50, "direct_chunks_validated": 50, "bounds_checked": True, "completed_at": time.time(),
            "expires_at": current["session_expires_at"], "direct_warm_round_trip_s": {"sample_count": 50, "p95": .1},
            "integrated": full_proof(), "stop_proof": {
                "stop_requested_during_inflight_rpc": True, "cancelled_before_transport_return": True,
                "commands_after_stop": 0,
                "total_fake_commands": 0, "all_fake_robots_released": True,
                "execution": {"inference_calls": 1, "stop_requested": True, "completed_points": 0, "attempted_points": 0,
                              "faults": 0, "stopped_rpc_exceptions": 1, "late_response_rows_discarded": 0}}}


def test_matching_complete_proof_is_accepted(host, statistics):
    current = metadata()
    admission.validate_qualification(passing_report(host, statistics, current), current,
                                     task="cube", rig_path=Path("unused"), statistics=statistics)


def test_consistent_nonzero_transition_substeps_are_allowed(host, statistics):
    current = metadata()
    evidence = passing_report(host, statistics, current)
    evidence["integrated"].update(completed_points=1300, attempted_points=1300, inserted_transition_points=50)
    admission.validate_qualification(evidence, current, task="cube", rig_path=Path("unused"), statistics=statistics)


@pytest.mark.parametrize("field,value", [
    ("predicted_rows", 2499), ("admitted_rows", 1251), ("completed_rows", 1249),
    ("admitted_chunks", 51), ("intended_unused_rows", 0), ("uncompleted_committed_rows", 1),
    ("rows_discarded_on_stop_or_deadline", 1), ("rows_discarded_on_fault", 1),
    ("prefix_dropped_rows", 1), ("reordered_rows", 1), ("unknown_partial_dispatches", 1),
    ("modified_commands", 1), ("coherence_violations", 1), ("faults", 1),
    ("invalid_chunks", 1), ("inference_calls", 51), ("stop_requested", True),
    ("late_response_rows_discarded", 50), ("stopped_rpc_exceptions", 1),
    ("attempted_points", 1251), ("completed_points", 1249), ("inserted_transition_points", 1),
])
def test_counter_inconsistency_cannot_hide_behind_50_complete_chunks(host, statistics, field, value):
    current = metadata()
    evidence = passing_report(host, statistics, current)
    evidence["integrated"][field] = value
    with pytest.raises(ValueError):
        admission.validate_qualification(evidence, current, task="cube", rig_path=Path("unused"), statistics=statistics)


@pytest.mark.parametrize("field,value", [
    ("qualified", False), ("bounds_checked", False), ("hardware_tested", True),
    ("completed_warm_samples", 49), ("controller_mode", "reference"),
    ("adapter_build_id", "old"), ("statistics_sha256", "changed"),
    ("completed_at", time.time() - 90000),
])
def test_incomplete_or_different_qualification_rejected(host, statistics, field, value):
    current = metadata()
    evidence = passing_report(host, statistics, current)
    evidence[field] = value
    with pytest.raises(ValueError):
        admission.validate_qualification(evidence, current, task="cube", rig_path=Path("unused"), statistics=statistics)


@pytest.mark.parametrize("field,value", [
    ("stop_requested_during_inflight_rpc", False), ("commands_after_stop", 1),
    ("total_fake_commands", 1), ("all_fake_robots_released", False),
])
def test_stop_proof_requires_actual_no_send_and_release(host, statistics, field, value):
    current = metadata()
    evidence = passing_report(host, statistics, current)
    evidence["stop_proof"][field] = value
    with pytest.raises(ValueError):
        admission.validate_qualification(evidence, current, task="cube", rig_path=Path("unused"), statistics=statistics)


@pytest.mark.parametrize("field,value", [
    ("inference_calls", 0), ("inference_calls", 2), ("stop_requested", False),
    ("completed_points", 1), ("faults", 1),
])
def test_stop_probe_counters_must_match_one_invalidated_request(host, statistics, field, value):
    current = metadata()
    evidence = passing_report(host, statistics, current)
    evidence["stop_proof"]["execution"][field] = value
    with pytest.raises(ValueError):
        admission.validate_qualification(evidence, current, task="cube", rig_path=Path("unused"), statistics=statistics)


def test_saved_fake_never_confuses_archived_state_with_fake_tracking(host):
    samples = [observation(.1, 1), observation(.2, 2)]
    fake = qualification.SavedFake(samples, admission.target_validator(host))
    first = fake.observe_policy_input()
    target = dict(zip(YAM_NAMES, observation(.3)["state"].tolist(), strict=True))
    fake.send(target, lambda: None)
    tracked = fake.observe()
    second = fake.observe_policy_input()
    assert np.array_equal(first["state"], samples[0]["state"])
    assert np.array_equal(tracked["state"], observation(.3)["state"])
    assert np.array_equal(tracked["top"], samples[0]["top"])
    assert np.array_equal(second["state"], samples[1]["state"])
    assert np.array_equal(second["top"], samples[1]["top"])
    assert second["source_observation_index"] == 1
    assert np.array_equal(second["fake_actuator_state_at_request"], observation(.3)["state"])
    assert np.array_equal(fake.observe()["state"], observation(.3)["state"])
    fake.release()
    with pytest.raises(RuntimeError):
        fake.observe()
    with pytest.raises(RuntimeError):
        fake.send(target, lambda: None)
    with pytest.raises(RuntimeError):
        fake.observe_policy_input()


def test_response_anchor_correlation_and_raw_padding_retained(statistics):
    sample = observation(.1)
    raw = np.zeros((50, 32), dtype=np.float32)
    raw[:, 14:] = np.arange(18, dtype=np.float32)
    decoded = qualification.decoded_response({"state": sample["state"].tolist(), "raw_normalized_chunk": raw},
                                             sample, statistics)
    np.testing.assert_array_equal(decoded["raw_normalized_chunk"], raw)
    np.testing.assert_array_equal(decoded["audit"]["unused_normalized_model_dimensions"], raw[:, 14:])
    assert decoded["audit"]["clipping_applied_to_model_or_joints"] is False
    wrong = sample["state"].copy(); wrong[1] += .01
    with pytest.raises(ValueError, match="exact measured"):
        qualification.decoded_response({"state": wrong, "raw_normalized_chunk": raw}, sample, statistics)


class FastClock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now += max(seconds, 1e-10)


class FastExecutor(OpenPiYamExecutor):
    """Real executor logic with a fake clock, never wall-clock qualification."""

    def __init__(self, **kwargs):
        clock = FastClock()
        super().__init__(**kwargs, clock=clock, wait=clock.wait)


class FakeTransport:
    def __init__(self, bad_call=None, bad_column=0, bad_value=None):
        self.metadata = metadata()
        self.calls = 0
        self.bad_call, self.bad_column, self.bad_value = bad_call, bad_column, bad_value
        self.request_sent = threading.Event()
        self.cancelled = threading.Event()
        self.sent_calls = []

    def ready(self):
        return self.metadata

    def warm(self, _request):
        return None

    def ensure_session_active(self):
        return None

    def cancel(self):
        self.cancelled.set()

    def predict_chunk(self, request, timeout_s=2):
        index = self.calls
        self.calls += 1
        self.request_sent.clear()
        self.sent_calls.append(index)
        self.request_sent.set()
        if index == 100:  # Direct 50, integrated 50, then one explicit Stop probe.
            if not self.cancelled.wait(1):
                raise TimeoutError("Unit Stop watcher did not cancel")
            raise RuntimeError("Invalidated unit request; private transport text excluded")
        raw = np.zeros((50, 32), dtype=np.float32)
        if index == self.bad_call:
            raw[0, self.bad_column] = self.bad_value
        return {"state": request["state"], "raw_normalized_chunk": raw}


@pytest.fixture
def fake_executor(monkeypatch):
    monkeypatch.setattr(qualification, "OpenPiYamExecutor", FastExecutor)


def test_collect_full_workflow_retains_direct_and_integrated_raw_evidence(host, statistics, fake_executor, tmp_path):
    transport = FakeTransport()
    samples = [observation(.1, 1), observation(.2, 2)]
    evidence = qualification.collect(transport, statistics=statistics, observations=samples,
                                     task="cube", rig_path=Path("unused"), directory=tmp_path / "qualification")
    assert evidence["qualified"] is True, evidence["reasons"]
    assert evidence["hardware_tested"] is False
    assert "perfect target tracking" in evidence["fake_scope"]
    assert evidence["qualification_input_mode"] == "paired_saved_state_rgb_replay_v1"
    assert evidence["closed_loop_world_simulation"] is False
    assert evidence["fake_actuator_reset_on_policy_input"] is False
    assert evidence["integrated"]["completed_chunks"] == 50
    assert evidence["integrated"]["completed_rows"] == 1250
    assert evidence["stop_proof"]["total_fake_commands"] == 0
    assert transport.calls == 101
    files = list((tmp_path / "qualification").glob("*.npz"))
    assert len([path for path in files if path.name.startswith("direct-")]) == 50
    assert len([path for path in files if path.name.startswith("integrated-")]) == 50
    for path in files:
        with np.load(path, allow_pickle=False) as stored:
            assert stored["state"].shape == (14,)
            assert stored["raw_normalized_chunk"].shape == (50, 32)
            assert stored["proposed_yam_chunk"].shape == (50, 14)
            assert all(stored[name].shape == (480, 640, 3) for name in ("top", "left_wrist", "right_wrist"))
            if path.name.startswith("integrated-"):
                index = int(path.stem.split("-")[1])
                source = samples[index % len(samples)]
                assert stored["source_observation_index"] == index % len(samples)
                for key in ("state", "top", "left_wrist", "right_wrist"):
                    np.testing.assert_array_equal(stored[key], source[key])
                if index == 1:
                    assert not np.array_equal(stored["fake_actuator_state_at_request"], source["state"])
    input_events = [event for event in json.loads((tmp_path / "qualification/integrated-events.json").read_text())
                    if event["kind"] == "saved_policy_input"]
    assert len(input_events) == 50
    assert all(event["policy_images_generated_by_fake_actuators"] is False for event in input_events)


@pytest.mark.parametrize("bad", [1000., float("nan"), float("inf")])
def test_direct_fault_retains_raw_reply_and_never_qualifies(host, statistics, fake_executor, tmp_path, bad):
    transport = FakeTransport(bad_call=0, bad_value=bad)
    evidence = qualification.collect(transport, statistics=statistics, observations=[observation()],
                                     task="cube", rig_path=Path("unused"), directory=tmp_path / "qualification")
    assert evidence["qualified"] is False
    assert evidence["reasons"]
    assert transport.calls == 1
    raw_files = list((tmp_path / "qualification").glob("direct-*.npz"))
    assert len(raw_files) == 1
    with np.load(raw_files[0], allow_pickle=False) as stored:
        actual = stored["raw_normalized_chunk"][0, 0]
        assert np.isnan(actual) if np.isnan(bad) else actual == bad
    text = (tmp_path / "qualification/qualification.json").read_text()
    assert "private transport text" not in text
    json.loads(text)


def test_integrated_fault_releases_fake_robot_and_retains_received_sample(host, statistics, fake_executor, monkeypatch, tmp_path):
    transport = FakeTransport(bad_call=50, bad_value=1000.)
    created = []
    original = qualification.SavedFake

    class TrackedFake(original):
        def __init__(self, *args):
            super().__init__(*args)
            created.append(self)

    monkeypatch.setattr(qualification, "SavedFake", TrackedFake)
    evidence = qualification.collect(transport, statistics=statistics, observations=[observation()],
                                     task="cube", rig_path=Path("unused"), directory=tmp_path / "qualification")
    assert evidence["qualified"] is False and evidence["integrated"]["faults"] == 1
    assert len(created) == 1 and created[0].released
    assert not created[0].sent and transport.calls == 51
    with np.load(tmp_path / "qualification/integrated-000.npz", allow_pickle=False) as stored:
        assert stored["raw_normalized_chunk"][0, 0] == 1000.
