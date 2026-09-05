import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from yamkit import arm as arm_module
from yamkit import probes


def snapshot(**changes):
    values = {
        "state": np.array([0.25] * 6 + [0.5]),
        "state_names": tuple([f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]),
        "images": {"left_wrist": np.zeros((6, 8, 3), dtype=np.uint8)},
        "source": "episode:2/frame:17", "captured_at": time.time(),
    }
    values.update(changes)
    return probes.ProbeObservation(**values)


def no_hardware(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("hardware was accessed")

    monkeypatch.setattr(arm_module.YamArm, "connect", forbidden)
    monkeypatch.setattr(arm_module, "resolve_channel", forbidden)
    monkeypatch.setattr(probes, "_make_cameras", forbidden)


def report(observation, actions, **kwargs):
    return probes.probe_observation(
        observation, lambda _: actions, action_names=observation.state_names,
        profile_id="fixture", revision="pinned", mapping_validated=True, **kwargs,
    )


def test_saved_roundtrip_preserves_source_age_images_and_never_opens_hardware(tmp_path, monkeypatch):
    no_hardware(monkeypatch)
    original = snapshot(captured_at=time.time() - 3600)
    path = probes.save_observation(tmp_path / "observation.npz", original)
    saved = probes.load_saved_observation(path)
    result = report(saved, np.tile(saved.state, (4, 1)))
    assert saved.state_names == original.state_names
    np.testing.assert_array_equal(saved.images["left_wrist"], original.images["left_wrist"])
    assert "episode:2/frame:17" in result["source"]
    assert result["observation_age_before_s"] >= 3600
    assert "observation_stale" in result["issues"]
    assert result["activation"] == probes.SAVED_LABEL
    assert result["motion_approved"] is False
    assert result["replay_permitted"] is False


def test_saved_format_rejects_pickle(tmp_path):
    path = tmp_path / "malicious.npz"
    np.savez(path, metadata=np.array({"not": "json"}, dtype=object))
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        probes.load_saved_observation(path)


def test_saved_format_rejects_large_uncompressed_payload(tmp_path, monkeypatch):
    path = probes.save_observation(tmp_path / "large.npz", snapshot(images={"top": np.zeros((100, 100, 3), dtype=np.uint8)}))
    monkeypatch.setattr(probes, "MAX_SNAPSHOT_BYTES", 4000)
    with pytest.raises(ValueError, match="uncompressed"):
        probes.load_saved_observation(path)


@pytest.mark.parametrize("limits", [None, [], [0], [0, 0], [0, float("nan")], [0, float("inf")], [0, 1, 2], [False, True]])
def test_all_arms_calibration_preflight_before_any_hardware(rig, monkeypatch, limits):
    no_hardware(monkeypatch)
    rig.arm("left_follower").gripper_limits = [0, 6.5]
    rig.arm("right_follower").gripper_limits = limits
    with pytest.raises(ValueError, match="right_follower.*gripper_limits"):
        probes.capture_live_observation(rig, approved=True)


def test_reversed_calibration_endpoints_preserve_direction(rig):
    rig.arm("left_follower").gripper_limits = [6.5, 0]
    specs, names = probes.preflight_live_probe(rig, ["left_follower"])
    assert specs[0].gripper_limits == [6.5, 0]
    assert names[-1] == "gripper.pos"


def test_live_requires_explicit_approval_before_any_hardware(rig, monkeypatch):
    no_hardware(monkeypatch)
    with pytest.raises(PermissionError, match=probes.ACTIVE_READ_LABEL):
        probes.capture_live_observation(rig)


def test_incompatible_native_policy_rejected_before_any_hardware(rig, monkeypatch):
    no_hardware(monkeypatch)
    for spec in rig.followers():
        spec.gripper_limits = [0, 6.5]
    with pytest.raises(ValueError, match="physical state mapping differs"):
        probes.capture_live_observation(rig, approved=True, expected_state_names=["native"] * 32)


def test_live_uses_gravity_read_close_with_no_position_or_home_commands(rig, fake_connect, monkeypatch):
    for spec in rig.followers():
        spec.gripper_limits = [0, 6.5]
    calls = []
    original_connect = arm_module.YamArm.connect

    def connect(spec, channel, **kwargs):
        calls.append((spec.name, kwargs))
        return original_connect(spec, channel, **kwargs)

    monkeypatch.setattr(arm_module.YamArm, "connect", connect)
    monkeypatch.setattr(probes, "_make_cameras", lambda configs: {})
    obs = probes.capture_live_observation(rig, approved=True)
    assert [name for name, _ in calls] == ["left_follower", "right_follower"]
    assert all(kwargs["zero_gravity"] is True for _, kwargs in calls)
    assert len(obs.state) == 14
    assert obs.state_names[0] == "left_joint_1.pos"
    assert obs.state_names[6] == "left_gripper.pos"
    assert obs.state_names[7] == "right_joint_1.pos"
    assert all(robot.closed and not robot.commands for robot in fake_connect.values())
    result = report(obs, np.tile(obs.state, (2, 1)), max_age_s=10)
    assert result["activation"] == probes.ACTIVE_READ_LABEL
    assert result["age_basis"] == "local monotonic"


def test_partial_arm_failure_closes_previously_opened_without_homing(rig, fake_connect, monkeypatch):
    for spec in rig.followers():
        spec.gripper_limits = [0, 6.5]
    original_connect = arm_module.YamArm.connect

    def connect(spec, channel, **kwargs):
        if spec.name == "right_follower":
            raise RuntimeError("connect failed")
        return original_connect(spec, channel, **kwargs)

    monkeypatch.setattr(arm_module.YamArm, "connect", connect)
    monkeypatch.setattr(probes, "_make_cameras", lambda configs: {})
    with pytest.raises(RuntimeError, match="connect failed"):
        probes.capture_live_observation(rig, approved=True)
    assert fake_connect["left_follower"].closed
    assert not fake_connect["left_follower"].commands


class FakeHub:
    def __init__(self, running=False, owner=None):
        self.suspended_by = owner
        self.cams = {"top": SimpleNamespace(running=running)}
        self.events = []

    def suspend(self, owner):
        self.events.append("suspend")
        self.suspended_by = owner

    def resume(self):
        self.events.append("resume")
        self.suspended_by = None


def test_unreleased_preview_prevents_activation_and_camera_open(rig, monkeypatch):
    for spec in rig.followers():
        spec.gripper_limits = [0, 6.5]
    monkeypatch.setattr(arm_module, "resolve_channel", lambda spec: "fakecan")
    monkeypatch.setattr(arm_module.YamArm, "connect", lambda *a, **kw: pytest.fail("activated"))
    monkeypatch.setattr(probes, "_make_cameras", lambda _: pytest.fail("opened competing camera"))
    hub = FakeHub(running=True)
    with pytest.raises(RuntimeError, match="did not release"):
        probes.capture_live_observation(rig, approved=True, camera_hub=hub)
    assert hub.events == ["suspend", "resume"]


def test_different_camera_owner_not_resumed():
    hub = FakeHub(owner="record")
    with pytest.raises(RuntimeError, match="already owned"), probes._preview_ownership(hub):
        pytest.fail("ownership stolen")
    assert hub.suspended_by == "record"
    assert hub.events == []


def test_camera_failure_cleans_up_and_does_not_activate(rig, monkeypatch):
    for spec in rig.followers():
        spec.gripper_limits = [0, 6.5]
    monkeypatch.setattr(arm_module, "resolve_channel", lambda spec: "fakecan")
    monkeypatch.setattr(arm_module.YamArm, "connect", lambda *a, **kw: pytest.fail("activated"))
    events = []

    class Camera:
        is_connected = False

        def connect(self):
            events.append("connect")
            raise RuntimeError("camera busy")

        def disconnect(self):
            events.append("disconnect")

    monkeypatch.setattr(probes, "_make_cameras", lambda _: {"top": Camera()})
    hub = FakeHub()
    with pytest.raises(RuntimeError, match="camera busy"):
        probes.capture_live_observation(rig, approved=True, camera_hub=hub)
    assert events == ["connect", "disconnect"]
    assert hub.events == ["suspend", "resume"]


def test_live_probe_camera_lease_denial_prevents_activation(rig, fake_connect, monkeypatch):
    for spec in rig.followers():
        spec.gripper_limits = [0, 6.5]
    monkeypatch.setattr(probes, "_make_cameras", lambda _: {"top": SimpleNamespace(
        connect=lambda: pytest.fail("opened a camera without ownership"))})

    def denied(names):
        assert names == ["top"]
        raise RuntimeError("preview still owns camera")

    monkeypatch.setattr("yamkit.camera_ownership.claim_from_env", denied)
    with pytest.raises(RuntimeError, match="preview still owns"):
        probes.capture_live_observation(rig, approved=True)
    assert not fake_connect


@pytest.mark.parametrize("failed_cleanup", [False, True])
def test_live_probe_releases_camera_lease_only_after_confirmed_cleanup(
    rig, fake_connect, monkeypatch, failed_cleanup,
):
    for spec in rig.followers():
        spec.gripper_limits = [0, 6.5]
    events = []

    class Camera:
        is_connected = False

        def connect(self):
            assert events == ["claim"]
            events.append("connect")
            self.is_connected = True

        def async_read(self, **kwargs):
            return np.zeros((8, 8, 3), dtype=np.uint8)

        read_latest = async_read

        def disconnect(self):
            assert all(robot.closed for robot in fake_connect.values())
            events.append("disconnect")
            if failed_cleanup:
                raise RuntimeError("camera close failed")
            self.is_connected = False

    def claim(names):
        assert names == ["top"]
        events.append("claim")
        return SimpleNamespace(release=lambda: events.append("release"))

    monkeypatch.setattr(probes, "_make_cameras", lambda _: {"top": Camera()})
    monkeypatch.setattr("yamkit.camera_ownership.claim_from_env", claim)
    hub = FakeHub()
    if failed_cleanup:
        with pytest.raises(RuntimeError, match="camera close failed"):
            probes.capture_live_observation(rig, approved=True, camera_hub=hub)
        assert events == ["claim", "connect", "disconnect"]
        assert hub.suspended_by == "policy-probe-live"
    else:
        probes.capture_live_observation(rig, approved=True, camera_hub=hub)
        assert events == ["claim", "connect", "disconnect", "release"]
        assert hub.suspended_by is None
    assert all(robot.closed and not robot.commands for robot in fake_connect.values())


@pytest.mark.parametrize("defect", ["connected", "reader_alive"])
def test_silent_camera_close_failure_retains_owners_and_closes_every_other_resource(
    rig, fake_connect, monkeypatch, defect,
):
    for spec in rig.followers():
        spec.gripper_limits = [0, 6.5]
    events = []

    class Camera:
        def __init__(self, name):
            self.name = name
            self.is_connected = False
            self.reader_alive = False
            self.thread = SimpleNamespace(is_alive=lambda: self.reader_alive)

        def connect(self):
            self.is_connected = self.reader_alive = True
            events.append("connect:" + self.name)

        def async_read(self, **kwargs):
            return np.zeros((8, 8, 3), dtype=np.uint8)

        read_latest = async_read

        def disconnect(self):
            assert len(fake_connect) == 2 and all(robot.closed for robot in fake_connect.values())
            events.append("disconnect:" + self.name)
            self.is_connected = self.name == "broken" and defect == "connected"
            self.reader_alive = self.name == "broken" and defect == "reader_alive"
            self.thread = None  # pinned cameras drop their reference even if the reader survives

    cameras = {name: Camera(name) for name in ("healthy", "broken")}
    monkeypatch.setattr(probes, "_make_cameras", lambda _: cameras)
    monkeypatch.setattr("yamkit.camera_ownership.claim_from_env", lambda _: SimpleNamespace(
        release=lambda: events.append("release")))
    hub = FakeHub()
    with pytest.raises(RuntimeError, match="camera release could not be confirmed"):
        probes.capture_live_observation(rig, approved=True, camera_hub=hub)
    assert events == ["connect:healthy", "connect:broken", "disconnect:broken", "disconnect:healthy"]
    assert hub.events == ["suspend"] and hub.suspended_by == "policy-probe-live"
    assert all(robot.closed and not robot.commands for robot in fake_connect.values())


def test_camera_cleaned_during_failed_connect_releases_lease_without_hardware_activation(
    rig, fake_connect, monkeypatch,
):
    from lerobot.utils.errors import DeviceNotConnectedError

    for spec in rig.followers():
        spec.gripper_limits = [0, 6.5]
    events = []

    class Camera:
        is_connected = False
        thread = None

        def connect(self):
            raise RuntimeError("camera setup failed after own cleanup")

        def disconnect(self):
            events.append("disconnect")
            raise DeviceNotConnectedError("already disconnected")

    monkeypatch.setattr(probes, "_make_cameras", lambda _: {"top": Camera()})
    monkeypatch.setattr("yamkit.camera_ownership.claim_from_env", lambda _: SimpleNamespace(
        release=lambda: events.append("release")))
    hub = FakeHub()
    with pytest.raises(RuntimeError, match="camera setup failed after own cleanup"):
        probes.capture_live_observation(rig, approved=True, camera_hub=hub)
    assert events == ["disconnect", "release"] and not fake_connect
    assert hub.events == ["suspend", "resume"] and hub.suspended_by is None


@pytest.mark.parametrize("retry_succeeds", [False, True])
@pytest.mark.parametrize("reader_survives", [False, True])
def test_probe_tracks_readers_detached_by_internal_warmup_cleanup(
    rig, fake_connect, monkeypatch, retry_succeeds, reader_survives,
):
    from lerobot.utils.errors import DeviceNotConnectedError

    for spec in rig.followers():
        spec.gripper_limits = [0, 6.5]
    events = []

    class Camera:
        def __init__(self, name):
            self.name, self.thread, self.is_connected = name, None, False

        def _stop_read_thread(self):
            self.thread = None
            self.is_connected = False

        def connect(self):
            self.is_connected = True
            if self.name == "broken":
                self.thread = SimpleNamespace(is_alive=lambda: reader_survives)
                self._stop_read_thread()  # pinned warmup cleanup detaches the first reader
                if not retry_succeeds:
                    raise ConnectionError("synthetic warmup failure")
                self.is_connected = True
                self.thread = SimpleNamespace(is_alive=lambda: False)  # successful later retry

        def async_read(self, **kwargs):
            return np.zeros((8, 8, 3), dtype=np.uint8)

        read_latest = async_read

        def disconnect(self):
            events.append("disconnect:" + self.name)
            assert all(arm.closed for arm in fake_connect.values())
            if not self.is_connected:
                raise DeviceNotConnectedError("already cleaned up internally")
            self._stop_read_thread()

    cameras = {name: Camera(name) for name in ("healthy", "broken")}
    monkeypatch.setattr(probes, "_make_cameras", lambda _: cameras)
    monkeypatch.setattr("yamkit.camera_ownership.claim_from_env", lambda _: SimpleNamespace(
        release=lambda: events.append("release")))
    hub = FakeHub()
    if reader_survives:
        with pytest.raises(RuntimeError, match="camera release could not be confirmed"):
            probes.capture_live_observation(rig, approved=True, camera_hub=hub)
        assert events == ["disconnect:broken", "disconnect:healthy"]
        assert hub.events == ["suspend"]
    else:
        if retry_succeeds:
            probes.capture_live_observation(rig, approved=True, camera_hub=hub)
        else:
            with pytest.raises(ConnectionError, match="synthetic warmup failure"):
                probes.capture_live_observation(rig, approved=True, camera_hub=hub)
        assert events == ["disconnect:broken", "disconnect:healthy", "release"]
        assert hub.events == ["suspend", "resume"]
    assert all("_stop_read_thread" not in vars(camera) for camera in cameras.values())
    # Probe camera setup completes before any arms activate, including retries.
    assert len(fake_connect) == (2 if retry_succeeds and not reader_survives else 0)
    assert all(arm.closed and not arm.commands for arm in fake_connect.values())


def test_report_keeps_unclipped_targets_signed_deltas_and_full_chunk_extrema():
    obs = snapshot()
    actions = np.tile(obs.state, (3, 1))
    actions[0, 0] = -0.75
    actions[0, 6] = -0.5
    actions[2, 2] = 20.0
    actions[2, 6] = 2.0
    result = report(obs, actions)
    assert result["first_targets"][0] == -0.75
    assert result["first_signed_deltas"][0] == -1.0
    assert result["chunk_max"][2] == 20.0
    assert result["grippers"]["gripper.pos"] == {"state": 0.5, "first_target": -0.5, "min": -0.5, "max": 2.0}
    assert "gripper_target_out_of_range:gripper.pos" in result["issues"]
    assert "large_joint_delta:joint_3.pos" in result["issues"]
    assert result["clipped"] is False
    assert "chunk" not in result


def test_predictor_called_once_preserving_images_for_shared_profile_transform():
    obs = snapshot()
    seen = []

    def predict(value):
        seen.append(value)
        assert list(value.images) == ["left_wrist"]
        assert value.images["left_wrist"].shape == (6, 8, 3)
        return np.tile(value.state, (1, 3, 1))

    result = probes.probe_observation(
        obs, predict, action_names=obs.state_names, profile_id="fixture", revision="pinned",
        transforms=["server:center_16_9", "saved:rename_map"], mapping_validated=True,
    )
    assert seen == [obs]
    assert result["transforms"] == ["server:center_16_9", "saved:rename_map"]
    assert result["passed_diagnostics"] is True
    assert result["motion_approved"] is False


@pytest.mark.parametrize("shape", [(7,), (0, 7), (2, 6), (2, 3, 7)])
def test_malformed_chunks_are_flagged_without_padding_or_truncation(shape):
    result = report(snapshot(), np.zeros(shape))
    assert "malformed_action_shape" in result["issues"]
    assert "first_targets" not in result


def test_nonfinite_actions_visible_and_report_remains_valid_json():
    actions = np.ones((3, 7))
    actions[0, 1] = np.nan
    actions[2, 0] = np.inf
    result = report(snapshot(), actions)
    assert "nonfinite_actions" in result["issues"]
    assert result["finite"] is False
    assert result["first_targets"][1] is None
    assert result["chunk_max"][0] is None
    assert json.loads(probes.format_probe_report(result))["clipped"] is False


def test_expected_mapping_checked_before_prediction():
    obs = snapshot()
    with pytest.raises(ValueError, match="state names differ"):
        probes.probe_observation(
            obs, lambda _: pytest.fail("predicted with incompatible mapping"),
            action_names=obs.state_names, profile_id="fixture", revision="pinned",
            expected_state_names=tuple(reversed(obs.state_names)),
        )


@pytest.mark.parametrize("captured_at,issue", [(None, "observation_age_unknown"), (time.time() + 500, "observation_timestamp_in_future")])
def test_unknown_and_future_saved_age_flagged(captured_at, issue):
    obs = snapshot(captured_at=captured_at)
    assert issue in report(obs, np.tile(obs.state, (2, 1)))["issues"]


def test_arbitrary_array_protocol_objects_are_rejected_before_conversion():
    class Arbitrary:
        def __array__(self, *args, **kwargs):
            pytest.fail("arbitrary array conversion was invoked")

    with pytest.raises(TypeError, match="bounded NumPy"):
        snapshot(state=Arbitrary()).validate()
    with pytest.raises(TypeError, match="bounded NumPy"):
        snapshot(images={"top": Arbitrary()}).validate()


def test_large_tensor_chunk_rejected_before_device_transfer():
    class Tensor:
        shape = (1, 1000, 1000)

        def detach(self):
            pytest.fail("oversized tensor was copied from device")

    with pytest.raises(ValueError, match="exceeds diagnostic bounds"):
        report(snapshot(), Tensor())


def test_nested_oversized_lists_rejected_before_numpy_conversion(monkeypatch):
    monkeypatch.setattr(probes, "MAX_ACTION_VALUES", 10)
    with pytest.raises(ValueError, match="bounded numerical arrays"):
        probes._bounded_chunk([[0.0] * 7, [0.0] * 7])
