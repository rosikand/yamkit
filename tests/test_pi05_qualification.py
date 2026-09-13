"""Software-only PI readiness and physical adapter tests with fake hardware."""

import json
import time
from threading import Event

import numpy as np
import pytest

from yamkit.inference.mapping import YAM_NAMES
from yamkit.pi05 import qualification, rollout
from yamkit.pi05.admission import validate_qualification
from yamkit.pi05.contract import CONTRACT, PROFILE, build_id
from yamkit.pi05.executor import Pi05ReferenceExecutor
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


def test_old_or_unbound_readiness_cannot_admit_robot(tmp_path):
    path = tmp_path / "rig.yaml"
    path.write_text("arms: []\n")
    with pytest.raises(ValueError, match="current passing"):
        validate_qualification({"qualified": True}, metadata(), task="move cube", rig_path=path)


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
    assert json.loads((destination / "report.json").read_text())["released"] is True


def test_qualification_save_preserves_historical_files(tmp_path):
    observation_path = tmp_path / "observation.npz"
    np.savez(observation_path, **observation())
    report = tmp_path / "qualification.json"
    save_qualification({"hardware_tested": False}, report, observation_paths=[observation_path])
    with pytest.raises(FileExistsError):
        save_qualification({"hardware_tested": True}, report, observation_paths=[observation_path])
    assert json.loads(report.read_text())["hardware_tested"] is False
