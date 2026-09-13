"""Software-only PI readiness and physical adapter tests with fake hardware."""

import json
import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from yamkit.inference.mapping import YAM_NAMES
from yamkit.pi05 import admission, qualification, rollout
from yamkit.pi05.admission import rig_observation_schema, validate_qualification
from yamkit.pi05.contract import CONTRACT, PROFILE, build_id
from yamkit.pi05.executor import Pi05ExecutionFault, Pi05ReferenceExecutor
from yamkit.pi05.qualification import load_saved_observation, make_request, save_qualification
from yamkit.pi05.transport import validate_readiness


def metadata():
    return {"profile": PROFILE.id, "model_revision": PROFILE.revision, "model": PROFILE.repo_id,
            "execution_mode": "eager", "pi05_build_id": build_id(), "controller_contract": CONTRACT,
            "ready": True, "instance_id": "pi05-test-instance", "http_session_expires_at": time.time() + 3600,
            "execution_identity": {"profile": PROFILE.id, "model_revision": PROFILE.revision,
                                   "controller_contract": CONTRACT["id"], "model_dtype": "bfloat16",
                                   "chunk_size": 30, "action_width": 14, "num_inference_steps": 10,
                                   "native_rtc_enabled": False, "strict_weights_restored": True,
                                   "compile_model": False}}


def observation():
    return {"state": np.zeros(14), **{name: np.zeros((8, 12, 3), dtype=np.uint8) for name in PROFILE.image_keys}}


def test_exact_pi05_readiness_not_old_generic_base():
    valid = metadata()
    validate_readiness(valid)
    for key, value in (("profile", "pi05"), ("model_revision", "wrong"), ("pi05_build_id", "old"),
                       ("execution_mode", "cuda_graph10"), ("controller_contract", {"id": "reference"})):
        with pytest.raises(ValueError, match="pinned native"):
            validate_readiness({**valid, key: value})


@pytest.mark.parametrize("bad", [None, [], {"execution_identity": None, "controller_contract": {}},
                                  {"execution_identity": {}, "controller_contract": None}])
def test_malformed_readiness_is_uniform_validation_failure(bad):
    with pytest.raises(ValueError, match="readiness"):
        validate_readiness(bad)


def rig():
    arms = {side + "_follower": SimpleNamespace(role="follower", side=side, arm_type="yam",
                                               gripper="linear_4310", gripper_limits=[0, 1])
            for side in ("left", "right")}
    return SimpleNamespace(cameras={name: {"width": 640, "height": 480, "fps": 30}
                                    for name in PROFILE.image_keys}, arm=arms.__getitem__,
                           validate=list, control=SimpleNamespace(home_speed=0.25))


@pytest.mark.parametrize("field,value", [("side", "right"), ("arm_type", "yam_pro"),
                                         ("gripper", "linear_3507"), ("gripper_limits", None)])
def test_native_rig_mapping_rejects_wrong_robot_side_and_gripper(field, value):
    fixture = rig()
    setattr(fixture.arm("left_follower"), field, value)
    with pytest.raises(ValueError, match="mapping"):
        rig_observation_schema(fixture)


@pytest.mark.parametrize("mutation", ["missing", "extra", "fps", "width", "height"])
def test_native_rig_camera_contract_is_explicit(mutation):
    fixture = rig()
    if mutation == "missing":
        fixture.cameras.pop("left_wrist")
    elif mutation == "extra":
        fixture.cameras["unused"] = dict(fixture.cameras["top"])
    else:
        fixture.cameras["top"][mutation] = None
    with pytest.raises(ValueError, match="camera"):
        rig_observation_schema(fixture)


def test_invalid_rig_cannot_qualify_when_plugin_would_reject_startup():
    fixture = rig()
    fixture.validate = lambda: ["duplicate CAN adapters"]
    with pytest.raises(ValueError, match="valid rig"):
        rig_observation_schema(fixture)


def test_disabled_startup_home_cannot_qualify():
    fixture = rig()
    fixture.control.home_speed = 0
    with pytest.raises(ValueError, match="home motion"):
        rig_observation_schema(fixture)


def test_tiny_saved_images_cannot_qualify_full_rig_payload(monkeypatch):
    calls = []
    monkeypatch.setattr(admission, "rig_binding", lambda _: {"observation_schema": rig_observation_schema(rig())})
    transport = SimpleNamespace(ready=lambda *_: calls.append("ready"))
    with pytest.raises(ValueError, match="actual full camera payload"):
        qualification.collect_qualification(transport, observations=[observation()], task="move cube",
                                            requests=50, rig_path="unused-test-rig")
    assert calls == []


@pytest.mark.parametrize("field,value", [("strict_weights_restored", False), ("native_rtc_enabled", True),
                                         ("compile_model", True), ("chunk_size", 50),
                                         ("action_width", 32), ("model_dtype", "float32")])
def test_changed_native_execution_cannot_be_ready(field, value):
    valid = metadata()
    valid["execution_identity"][field] = value
    with pytest.raises(ValueError, match="pinned native"):
        validate_readiness(valid)


def test_saved_recording_loader_and_wire_keep_pixels_state_exact(tmp_path):
    source = observation()
    source["state"] = np.arange(14, dtype=np.float32) / 14
    source["left_wrist"][:] = 42
    path = tmp_path / "observation.npz"
    np.savez(path, **source)
    saved = load_saved_observation(path)
    for name in source:
        np.testing.assert_array_equal(saved[name], source[name])
    request = make_request(saved, task="move cube", session_id="session", sequence_id=0, timeout_s=2)
    assert request["profile"] == "pi05-yam"
    assert request["state"] == source["state"].astype(np.float64).tolist()
    assert request["images"]["left_wrist"]["data"] == source["left_wrist"].tobytes()


def test_saved_loader_never_accepts_pickle_or_missing_camera(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, state=np.zeros(14), top=np.zeros((8, 12, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="three"):
        load_saved_observation(path)
    np.savez(path, **{**observation(), "state": np.array([object()] * 14)})
    with pytest.raises(ValueError, match="Object arrays"):
        load_saved_observation(path)


class Transport:
    def __init__(self):
        self.metadata = metadata()
        self.closed = False

    def ready(self, timeout):
        return self.metadata

    def predict_chunk(self, request, timeout):
        time.sleep(0.02)  # The Stop timer really fires while this fake RPC is in flight.
        return {"chunk": np.full((30, 14), 0.25).tolist()}

    def ensure_session_active(self):
        if self.closed:
            raise ValueError("closed")

    def cancel(self):
        pass

    def close(self):
        self.closed = True


def test_diagnostic_qualification_never_claims_physical_ready_without_full_host_proof(monkeypatch):
    class FastExecutor(Pi05ReferenceExecutor):
        def _wait_until(self, deadline):
            return self.check()

    monkeypatch.setattr(qualification, "Pi05ReferenceExecutor", FastExecutor)
    result = qualification.collect_qualification(Transport(), observations=[observation()], task="move cube", requests=1)
    assert result["qualified"] is False
    assert result["hardware_tested"] is False
    assert result["completed_warm_samples"] == 1
    assert result["integrated"]["completed_rows"] == 30
    assert result["stop_proof"]["commands_after_stop"] == 0
    assert result["stop_proof"]["stop_requested_during_inflight_rpc"] is True
    assert result["all_fake_robots_released"] is True
    assert any("50-sample" in reason for reason in result["reasons"])
    assert any("host and rig" in reason for reason in result["reasons"])


class FailingChunkTransport(Transport):
    def __init__(self, *, rejected_sample, chunk=None, error=None):
        super().__init__()
        self.calls = 0
        self.rejected_sample, self.chunk, self.error = rejected_sample, chunk, error

    def predict_chunk(self, request, timeout):
        index = self.calls - 1  # First request is the excluded cold sample.
        self.calls += 1
        if index == self.rejected_sample:
            if self.error is not None:
                raise self.error
            # No arbitrary response keys may enter diagnostic evidence.
            return {"chunk": self.chunk, "untrusted_private_field": "fixture-secret-do-not-retain"}
        return {"chunk": np.full((30, 14), 0.25).tolist()}


@pytest.mark.parametrize("column,value", [(6, 1.000123456789), (13, -0.000123456789),
                                          (6, 1.000895619392395)])  # Captured real-GPU overshoot, unchanged.
def test_rejected_native_gripper_preserves_exact_response_and_sample_without_clipping(column, value):
    chunk = np.full((30, 14), 0.25).tolist()
    chunk[17][column] = value
    transport = FailingChunkTransport(rejected_sample=38, chunk=chunk)
    result = qualification.collect_qualification(transport, observations=[observation()] * 50,
                                                task="move cube", requests=50)
    failure = result["failure"]
    assert result["qualified"] is False
    assert result["hardware_tested"] is False
    assert result["completed_warm_samples"] == len(result["direct_chunks"]) == 38
    assert transport.calls == 40  # Cold, 38 passing warm calls, one rejected response; no retry.
    assert failure["phase"] == "direct_warm"
    assert failure["sample_index"] == failure["observation_index"] == 38
    assert failure["request_sequence_id"] == 39
    assert failure["native_response_received"] is True
    assert failure["reason"] == "π0.5 gripper output is outside [0,1]; no automatic clipping is allowed"
    evidence = failure["native_response"]
    assert evidence["raw_chunk"] == chunk
    assert chunk[17][column] == value  # Reporting never mutates native output.
    assert evidence["shape"] == [30, 14]
    assert evidence["truncated"] is False
    assert evidence["gripper_bound_violations"] == [{"row_index": 17, "column_index": column,
        "name": YAM_NAMES[column], "value": value, "minimum": 0.0, "maximum": 1.0}]
    assert evidence["column_ranges"][column]["maximum"] == max(0.25, value)
    assert evidence["column_ranges"][column]["minimum"] == min(0.25, value)
    assert "fixture-secret-do-not-retain" not in json.dumps(result, allow_nan=False)
    assert "integrated" not in result


def test_failed_rpc_does_not_misattribute_previous_response_or_expose_error_text():
    transport = FailingChunkTransport(rejected_sample=1, error=RuntimeError("fixture-private-credential"))
    result = qualification.collect_qualification(transport, observations=[observation()], task="move cube", requests=2)
    assert result["completed_warm_samples"] == 1
    assert result["failure"]["sample_index"] == 1
    assert result["failure"]["request_sequence_id"] == 2
    assert result["failure"]["native_response_received"] is False
    assert result["failure"]["native_response"] is None
    assert result["failure"]["reason"] is None
    assert "fixture-private-credential" not in json.dumps(result, allow_nan=False)


def test_only_exact_known_execution_fault_messages_are_retained():
    transport = FailingChunkTransport(rejected_sample=0, error=Pi05ExecutionFault("fixture-private-credential"))
    result = qualification.collect_qualification(transport, observations=[observation()], task="move cube", requests=1)
    assert result["failure"]["error_type"] == "Pi05ExecutionFault"
    assert result["failure"]["reason"] is None
    assert "fixture-private-credential" not in json.dumps(result, allow_nan=False)


def test_configured_bound_failure_reports_row_and_exact_native_chunk_without_error_text():
    chunk = np.full((30, 14), 0.25).tolist()
    chunk[7][0] = 12.3456789
    transport = FailingChunkTransport(rejected_sample=0, chunk=chunk)

    def reject_joint(target):
        if target[YAM_NAMES[0]] > 10:
            raise ValueError("fixture-private-credential")

    result = qualification.collect_qualification(transport, observations=[observation()], task="move cube",
                                                requests=1, validate_target=reject_joint)
    assert result["failure"]["validation_row_index"] == 7
    assert result["failure"]["native_response"]["raw_chunk"] == chunk
    assert result["failure"]["native_response"]["column_ranges"][0]["maximum"] == 12.3456789
    assert result["qualified"] is False
    assert result["direct_chunks"] == []
    assert "fixture-private-credential" not in json.dumps(result, allow_nan=False)


def test_integrated_failure_retains_its_response_and_releases_fake_arms(monkeypatch):
    class FastExecutor(Pi05ReferenceExecutor):
        def _wait_until(self, deadline):
            return self.check()

    monkeypatch.setattr(qualification, "Pi05ReferenceExecutor", FastExecutor)
    chunk = np.full((30, 14), 0.25).tolist()
    chunk[0][13] = 1.001
    transport = FailingChunkTransport(rejected_sample=3, chunk=chunk)
    result = qualification.collect_qualification(transport, observations=[observation()], task="move cube", requests=2)
    assert result["failure"]["phase"] == "integrated"
    assert result["failure"]["sample_index"] == 1
    assert result["failure"]["native_response"]["raw_chunk"] == chunk
    assert result["integrated"]["completed_chunks"] == 1
    assert result["integrated"]["completed_rows"] == 30
    assert result["integrated"]["faults"] == 1
    assert result["all_fake_robots_released"] is True
    assert result["qualified"] is False


@pytest.mark.parametrize("value,kind", [(float("nan"), "NaN"), (float("inf"), "Infinity"),
                                       (float("-inf"), "-Infinity")])
def test_nonfinite_native_failure_is_exactly_identified_and_json_safe(value, kind):
    chunk = np.full((30, 14), 0.25).tolist()
    chunk[4][2] = value
    transport = FailingChunkTransport(rejected_sample=0, chunk=chunk)
    result = qualification.collect_qualification(transport, observations=[observation()], task="move cube", requests=1)
    evidence = result["failure"]["native_response"]
    assert evidence["raw_chunk"][4][2] is None
    assert evidence["nonfinite_values"] == [{"row_index": 4, "column_index": 2, "kind": kind}]
    assert evidence["column_ranges"][2]["finite_count"] == 29
    assert result["qualified"] is False
    json.dumps(result, allow_nan=False)


def test_response_diagnostics_are_bounded_and_never_serialize_unsupported_values():
    class NeverInspect:
        def __repr__(self):
            pytest.fail("Untrusted values must never be stringified")

    row = [0.25] * 14 + ["fixture-private-credential"]
    evidence = qualification._failed_chunk_evidence([row] * 31)
    assert evidence["truncated"] is True
    assert len(evidence["raw_chunk"]) == 30
    assert all(len(values) == 14 for values in evidence["raw_chunk"])
    unsupported = qualification._failed_chunk_evidence(
        [[NeverInspect(), "fixture-private-credential", 2 ** 2048, (1 << 1024) - 1]])
    assert unsupported["raw_chunk"] == [[None, None, None, None]]
    assert unsupported["unsupported_values_omitted"] is True
    assert "fixture-private-credential" not in json.dumps([evidence, unsupported], allow_nan=False)


def test_old_or_unbound_readiness_cannot_admit_robot(tmp_path):
    path = tmp_path / "rig.yaml"
    path.write_text("arms: []\n")
    with pytest.raises(ValueError, match="current passing"):
        validate_qualification({"qualified": True}, metadata(), task="move cube", rig_path=path)


@pytest.mark.parametrize("report", [[], None, {"integrated": None}, {"stop_proof": []},
                                     {"robot_host": None}, {"direct_warm_round_trip_s": []}])
def test_malformed_cached_qualification_is_actionable_without_touching_rig(report, tmp_path):
    with pytest.raises(ValueError, match="current passing"):
        validate_qualification(report, metadata(), task="move cube", rig_path=tmp_path / "does-not-exist")


class FakeRobot:
    def __init__(self):
        self.events = []
        self.action = dict(zip(YAM_NAMES, [0.0] * 14, strict=True))

    def connect(self):
        self.events.append("connect")

    def get_observation(self):
        return {**self.action, **{key: value for key, value in observation().items() if key != "state"}}

    def validate_action_target(self, target):
        assert set(target) == set(YAM_NAMES)

    def send_reference_action(self, target, dispatch_check):
        dispatch_check()
        self.events.append("send")
        self.action = dict(target)
        return dict(target)

    def disconnect(self, home):
        assert home is False
        self.events.append("release")


def test_physical_entrypoint_requires_explicit_confirmation_before_constructing_any_robot(tmp_path):
    called = []
    with pytest.raises(ValueError, match="supervised"):
        rollout.run_rollout(Transport(), task="move cube", duration_s=1, rig_path=tmp_path / "rig",
                            qualification={}, artifact_dir=tmp_path / "run",
                            robot_factory=lambda *_: called.append(True))
    assert called == []


def test_physical_entrypoint_rejects_stale_qualification_before_robot_constructor(tmp_path):
    called = []
    with pytest.raises(ValueError, match="current passing"):
        rollout.run_rollout(Transport(), task="move cube", duration_s=1, rig_path=tmp_path / "rig",
                            qualification={}, accept_mapping=True, confirm_supervised=True,
                            artifact_dir=tmp_path / "run", robot_factory=lambda *_: called.append(True))
    assert called == []


def test_delayed_confirmation_near_session_expiry_cannot_construct_robot(monkeypatch, tmp_path):
    monkeypatch.setattr(rollout, "validate_qualification", lambda *_a, **_kw: None)
    transport, called = Transport(), []
    transport.metadata["http_session_expires_at"] = time.time() + 60.5
    with pytest.raises(RuntimeError, match="failed"):
        rollout.run_rollout(transport, task="move cube", duration_s=5, rig_path=tmp_path / "rig",
                            qualification={}, accept_mapping=True, confirm_supervised=True,
                            artifact_dir=tmp_path / "run", robot_factory=lambda *_: called.append(True))
    assert called == []
    assert transport.closed is True
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["released"] is True
    assert report["home_attempted"] is False


def run_fake(monkeypatch, tmp_path, *, stop_after_send=False, fail=False):
    monkeypatch.setattr(rollout, "validate_qualification", lambda *_a, **_kw: None)
    robot, stop, transport = FakeRobot(), Event(), Transport()
    original_send = robot.send_reference_action

    def send(target, dispatch_check):
        sent = original_send(target, dispatch_check)
        if stop_after_send:
            stop.set()
        if fail:
            raise ValueError("SDK fault")
        return sent

    robot.send_reference_action = send
    destination = tmp_path / "run"
    return robot, destination, lambda: rollout.run_rollout(
        transport, task="move cube", duration_s=0.07, rig_path=tmp_path / "rig", qualification={},
        accept_mapping=True, confirm_supervised=True, artifact_dir=destination,
        shutdown_event=stop, robot_factory=lambda *_: robot,
        home=lambda *_: robot.events.append("home"))


def test_fake_physical_lifecycle_homes_only_on_healthy_completion_then_releases(monkeypatch, tmp_path):
    robot, destination, run = run_fake(monkeypatch, tmp_path)
    result = run()
    assert robot.events[0] == "connect"
    assert "send" in robot.events
    assert robot.events[-2:] == ["home", "release"]
    assert result["released"] is True and result["hardware_tested"] is False
    assert json.loads((destination / "report.json").read_text())["status"] == "completed"


def test_stop_releases_without_home_and_writes_artifacts_after_release(monkeypatch, tmp_path):
    robot, destination, run = run_fake(monkeypatch, tmp_path, stop_after_send=True)
    result = run()
    assert robot.events == ["connect", "send", "release"]
    assert result["status"] == "stopped"
    assert result["home_attempted"] is False
    assert (destination / "trace.json").is_file()


def test_fault_releases_without_retry_or_home(monkeypatch, tmp_path):
    robot, destination, run = run_fake(monkeypatch, tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="failed"):
        run()
    assert robot.events == ["connect", "send", "release"]
    report = json.loads((destination / "report.json").read_text())
    assert report["released"] is True
    assert report["execution"]["completed_rows"] == 0
    assert report["execution"]["attempted_rows"] == report["execution"]["unknown_partial_dispatches"] == 1


def test_qualification_save_preserves_historical_files(tmp_path):
    observation_path = tmp_path / "observation.npz"
    np.savez(observation_path, **observation())
    report = tmp_path / "qualification.json"
    save_qualification({"hardware_tested": False}, report, observation_paths=[observation_path])
    with pytest.raises(FileExistsError):
        save_qualification({"hardware_tested": True}, report, observation_paths=[observation_path])
    assert json.loads(report.read_text())["hardware_tested"] is False
