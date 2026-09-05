import time

import numpy as np
import pytest

from yamkit import probe_runner, probes
from yamkit.inference.profiles import get_profile
from yamkit.inference.service import ModelRuntime


@pytest.fixture
def profile():
    return get_profile("molmoact2")


@pytest.fixture
def observation(profile):
    return probes.ProbeObservation(
        state=np.array([0.0] * 6 + [0.5] + [0.0] * 6 + [0.5]),
        state_names=profile.state_names,
        images={name: np.zeros((6, 8, 3), dtype=np.uint8) for name in profile.image_keys},
        source="dataset:episode:17", captured_at=time.time() - 3600,
    )


@pytest.fixture
def calibrated_rig(rig, profile):
    for spec in rig.followers():
        spec.gripper_limits = [0.0, 6.5]
    rig.cameras = {name: {"type": "opencv", "index_or_path": i} for i, name in enumerate(profile.image_keys)}
    rig.save()
    return rig


class FakeRuntime:
    def __init__(self, profile):
        self.profile = profile
        self.events = []
        self.requests = []
        self.metadata = {**profile.metadata(), "ready": True, "saved_processors": True, "fresh_chunk": True}
        self.response_changes = {}

    def ready(self):
        self.events.append("ready")
        return self.metadata

    def predict_chunk(self, request):
        self.events.append("predict")
        self.requests.append(request)
        chunk = [request["state"]]
        raw = [list(request["state"])]
        raw[0][6] = -0.75
        response = {key: request[key] for key in (
            "protocol_version", "profile", "model_revision", "session_id", "sequence_id", "observation_time",
        )}
        response.update(
            action_units="robot", action_names=list(self.profile.action_names), chunk=chunk,
            unclipped_chunk=raw, unclipped_action_units="robot", saved_postprocessor_clamp=True,
            timing={"preprocess_s": 0.001, "inference_s": 0.01, "postprocess_s": 0.001, "total_s": 0.012},
            transforms={"left_wrist": {"crop": request["crop"]}},
        )
        response.update(self.response_changes)
        return response

    def reset(self, session_id):
        self.events.append("reset")
        assert self.requests[-1]["session_id"] == session_id


@pytest.fixture
def runtime(profile, monkeypatch):
    runtime = FakeRuntime(profile)

    def load(profile, *, device):
        runtime.events.append("load")
        return runtime

    monkeypatch.setattr(ModelRuntime, "load", load)
    return runtime


def save(tmp_path, observation):
    return probes.save_observation(tmp_path / "sample.npz", observation)


def run(rig_path, **kwargs):
    return probe_runner.run_profile_probe("molmoact2", rig_path, task="pick up the object", **kwargs)


def test_saved_local_probe_uses_saved_pipeline_before_clamp_with_no_hardware(tmp_path, observation, runtime, monkeypatch):
    monkeypatch.setattr(probe_runner, "capture_live_observation", lambda *a, **k: pytest.fail("activated"))
    result = run(tmp_path / "nonexistent-rig.yaml", saved=save(tmp_path, observation))
    assert runtime.events == ["load", "ready", "predict", "reset"]
    request = runtime.requests[0]
    assert request["mode"] == "saved_probe"
    assert request["observation_age_s"] > 3600
    assert list(request["images"]) == list(observation.images)
    assert request["crop"] == "none"
    assert request["continuation"] is None
    assert result["first_targets"][6] == -0.75
    assert result["chunk_min"][6] == -0.75
    assert result["clipped"] is False
    assert "observation_stale" in result["issues"]
    assert result["saved_postprocessor_clamp"] is True
    assert result["backend"] == "local"
    assert result["motion_approved"] is False
    assert result["physical_validation"] == "not performed"
    assert "source conventions only" in result["mapping_validation_basis"]


def test_live_capture_is_after_confirmed_readiness(calibrated_rig, observation, runtime, monkeypatch):
    def capture(rig, selected, **kwargs):
        assert runtime.events == ["load", "ready"]
        assert kwargs["approved"] is True
        assert kwargs["expected_state_names"] == observation.state_names
        assert selected == ["left_follower", "right_follower"]
        observation.mode = "live"
        observation.captured_at = time.time()
        observation.captured_monotonic = time.monotonic()
        runtime.events.append("capture")
        return observation

    monkeypatch.setattr(probe_runner, "capture_live_observation", capture)
    result = run(calibrated_rig.path, live=True, approved=True)
    assert runtime.events == ["load", "ready", "capture", "predict", "reset"]
    assert runtime.requests[0]["mode"] == "live_probe"
    assert result["activation"] == probes.ACTIVE_READ_LABEL


def test_live_rejects_approval_before_loading_or_capture(calibrated_rig, runtime):
    with pytest.raises(PermissionError, match="explicit operator approval"):
        run(calibrated_rig.path, live=True)
    assert runtime.events == []


def test_all_arm_calibration_preflight_before_paid_readiness(calibrated_rig, runtime):
    calibrated_rig.arm("right_follower").gripper_limits = None
    calibrated_rig.save()
    with pytest.raises(ValueError, match="right_follower.*gripper_limits"):
        run(calibrated_rig.path, live=True, approved=True)
    assert runtime.events == []


@pytest.mark.parametrize("change", ["wrong_revision", "not_ready", "no_processors", "native_units"])
def test_readiness_failure_prevents_live_capture(calibrated_rig, runtime, monkeypatch, change):
    if change == "wrong_revision":
        runtime.metadata["revision"] = "wrong"
    elif change == "not_ready":
        runtime.metadata["ready"] = False
    elif change == "no_processors":
        runtime.metadata["saved_processors"] = False
    else:
        runtime.metadata["action_units"] = "checkpoint_native"
    monkeypatch.setattr(probe_runner, "capture_live_observation", lambda *a, **k: pytest.fail("activated"))
    with pytest.raises(ValueError):
        run(calibrated_rig.path, live=True, approved=True)
    assert runtime.events == ["load", "ready"]


@pytest.mark.parametrize("policy", ["smolvla", "pi05"])
def test_native_only_profiles_never_activate_or_load(policy, calibrated_rig, runtime):
    with pytest.raises(ValueError, match="no verified physical YAM mapping"):
        probe_runner.run_profile_probe(policy, calibrated_rig.path, live=True, approved=True, task="test")
    assert runtime.events == []


@pytest.mark.parametrize("change", ["state", "camera"])
def test_saved_mapping_mismatch_before_loading(tmp_path, observation, runtime, change):
    if change == "state":
        observation.state_names = tuple(reversed(observation.state_names))
    else:
        observation.images["left"] = observation.images.pop("left_wrist")
    with pytest.raises(ValueError, match="observation"):
        run(tmp_path / "unused.yaml", saved=save(tmp_path, observation))
    assert runtime.events == []


def test_bgr_camera_rejected_before_activating_or_loading(calibrated_rig, runtime):
    calibrated_rig.cameras["left_wrist"]["color_mode"] = "bgr"
    calibrated_rig.save()
    with pytest.raises(ValueError, match="require RGB"):
        run(calibrated_rig.path, live=True, approved=True)
    assert runtime.events == []


@pytest.mark.parametrize("field,value", [("gripper", "linear_3507"), ("arm_type", "yam_pro")])
def test_other_hardware_geometry_rejected_before_loading(calibrated_rig, runtime, field, value):
    setattr(calibrated_rig.arm("right_follower"), field, value)
    calibrated_rig.save()
    with pytest.raises(ValueError, match="requires YAM with a LINEAR_4310"):
        run(calibrated_rig.path, live=True, approved=True)
    assert runtime.events == []


def test_modal_uses_outbound_service_only_and_transforms_once(tmp_path, observation, runtime, monkeypatch):
    from yamkit import modal_ops

    monkeypatch.setattr(ModelRuntime, "load", lambda *a, **k: pytest.fail("loaded weights locally"))
    monkeypatch.setattr(modal_ops, "owned_service", lambda: {"app_name": "yamkit-vla-test"})
    monkeypatch.setattr(modal_ops, "service_handle", lambda *a: runtime)
    timeouts = []

    def call(method, *args, timeout):
        timeouts.append(timeout)
        return method(*args)

    monkeypatch.setattr(modal_ops, "call", call)
    result = run(tmp_path / "unused.yaml", saved=save(tmp_path, observation), backend="modal", center_crop=True)
    assert runtime.events == ["ready", "predict", "reset"]
    assert timeouts == [300, 120, 5]
    assert result["backend"] == "modal"
    assert runtime.requests[0]["images"]["left_wrist"]["height"] == 6
    assert runtime.requests[0]["crop"] == "center_16_9"
    assert result["image_transforms"] == {"left_wrist": {"crop": "center_16_9"}}


def test_missing_preclipping_robot_units_fails_honestly_and_retires(tmp_path, observation, runtime):
    runtime.response_changes["unclipped_action_units"] = "normalized"
    with pytest.raises(ValueError, match="pre-clipping robot-unit diagnostics"):
        run(tmp_path / "unused.yaml", saved=save(tmp_path, observation))
    assert runtime.events[-1] == "reset"


def test_wrong_session_response_rejected_and_retired(tmp_path, observation, runtime):
    runtime.response_changes["session_id"] = "another-session"
    with pytest.raises(ValueError, match="session_id mismatch"):
        run(tmp_path / "unused.yaml", saved=save(tmp_path, observation))
    assert runtime.events[-1] == "reset"


def test_interruption_during_inference_retires_session(tmp_path, observation, runtime, monkeypatch):
    original = runtime.predict_chunk

    def stop(request):
        original(request)
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime, "predict_chunk", stop)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "unused.yaml", saved=save(tmp_path, observation))
    assert runtime.events[-1] == "reset"
