from unittest.mock import Mock

import numpy as np
import pytest

from yamkit.agent_robot import (
    GRIPPER_KEY,
    JOINT_KEYS,
    METADATA_KEY,
    FixtureRobot,
    LiveIntegrationError,
    ObservationError,
    RobotAdapter,
    make_live_robot,
    validate_rig,
)


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


def sample(clock=None):
    return FixtureRobot(clock=clock or Clock()).get_observation()


def test_fixture_uses_named_rgb_and_copies_frames():
    clock = Clock()
    robot = FixtureRobot(clock=clock)
    adapter = RobotAdapter(robot, clock=clock)
    first = adapter.observe()
    assert first.source == "fixture"
    assert first.sequence == 1
    assert first.captured_at == clock.now
    assert first.state[GRIPPER_KEY] == 0.5
    assert list(first.images) == ["fixture_top"]
    assert first.images["fixture_top"].shape == (120, 160, 3)
    assert first.images["fixture_top"].dtype == np.uint8
    np.testing.assert_array_equal(first.images["fixture_top"][50, 30], [230, 25, 25])
    clock.now += 0.1
    second = adapter.observe(after=first.captured_at)
    assert second.sequence == 2
    assert not np.shares_memory(first.images["fixture_top"], second.images["fixture_top"])


def test_returned_command_is_separate_from_readback():
    robot = Mock()
    raw = sample()
    robot.get_observation.return_value = raw
    target = {key: 0.2 for key in JOINT_KEYS} | {GRIPPER_KEY: 0.7}
    bounded = {key: 0.01 for key in JOINT_KEYS} | {GRIPPER_KEY: 0.51}
    robot.send_action.return_value = bounded
    adapter = RobotAdapter(robot, clock=Clock())
    assert adapter.send(target) == bounded
    robot.send_action.assert_called_once_with(target)
    robot.get_observation.assert_not_called()
    observed = adapter.observe()
    assert observed.state[JOINT_KEYS[0]] == 0.0
    assert observed.state[GRIPPER_KEY] == 0.5


@pytest.mark.parametrize("value", [True, False, np.bool_(True), float("nan"), float("inf"), "0.1", None])
def test_action_rejects_nonfinite_and_non_numeric_before_send(value):
    robot = Mock()
    target = {key: 0.0 for key in JOINT_KEYS} | {GRIPPER_KEY: 0.5}
    target[JOINT_KEYS[0]] = value
    with pytest.raises(ValueError, match="finite number"):
        RobotAdapter(robot).send(target)
    robot.send_action.assert_not_called()


@pytest.mark.parametrize("change", ["missing", "extra", "bad_gripper"])
def test_action_exact_keys_and_gripper_range(change):
    robot = Mock()
    target = {key: 0.0 for key in JOINT_KEYS} | {GRIPPER_KEY: 0.5}
    if change == "missing":
        target.pop(JOINT_KEYS[0])
    elif change == "extra":
        target["velocity"] = 1.0
    else:
        target[GRIPPER_KEY] = 1.01
    with pytest.raises(ValueError):
        RobotAdapter(robot).send(target)
    robot.send_action.assert_not_called()


def test_bare_plugin_observation_cannot_be_labeled_fresh():
    raw = sample()
    raw.pop(METADATA_KEY)
    robot = Mock(get_observation=Mock(return_value=raw))
    with pytest.raises(ObservationError, match="acquisition freshness contract"):
        RobotAdapter(robot, clock=Clock()).observe()


@pytest.mark.parametrize(
    "field,value,pattern",
    [
        ("captured_at", float("nan"), "finite number"),
        ("captured_at", 11.0, "future"),
        ("captured_at", 8.0, "stale"),
        ("captured_at", True, "finite number"),
        ("sequence", True, "nonnegative integer"),
        ("sequence", -1, "nonnegative integer"),
        ("sequence", 1.5, "nonnegative integer"),
        ("source", "", "nonempty label"),
        ("contract", "wall-clock", "freshness contract"),
    ],
)
def test_invalid_metadata(field, value, pattern):
    raw = sample()
    raw[METADATA_KEY][field] = value
    robot = Mock(get_observation=Mock(return_value=raw))
    with pytest.raises(ObservationError, match=pattern):
        RobotAdapter(robot, clock=Clock()).observe()


def test_after_boundary_is_strict():
    clock = Clock()
    robot = FixtureRobot(clock=clock)
    with pytest.raises(ObservationError, match="after the requested boundary"):
        RobotAdapter(robot, clock=clock).observe(after=clock.now)


@pytest.mark.parametrize("change", ["timestamp", "sequence", "source"])
def test_feedback_must_advance_in_same_source(change):
    clock = Clock()
    first = sample(clock)
    clock.now += 0.1
    second = sample(clock)
    second[METADATA_KEY]["sequence"] = 2
    if change == "timestamp":
        second[METADATA_KEY]["captured_at"] = first[METADATA_KEY]["captured_at"]
    elif change == "sequence":
        second[METADATA_KEY]["sequence"] = 1
    else:
        second[METADATA_KEY]["source"] = "another fixture"
    robot = Mock(get_observation=Mock(side_effect=[first, second]))
    adapter = RobotAdapter(robot, clock=clock)
    adapter.observe()
    with pytest.raises(ObservationError, match="changed|did not advance"):
        adapter.observe()


@pytest.mark.parametrize(
    "frame", [np.zeros((5, 5)), np.zeros((5, 5, 3)), np.zeros((0, 5, 3), dtype=np.uint8), "cached"]
)
def test_images_must_be_nonempty_uint8_rgb(frame):
    raw = sample()
    raw["fixture_top"] = frame
    robot = Mock(get_observation=Mock(return_value=raw))
    with pytest.raises(ObservationError, match="uint8 RGB"):
        RobotAdapter(robot, clock=Clock()).observe()


def test_camera_required():
    raw = sample()
    del raw["fixture_top"]
    robot = Mock(get_observation=Mock(return_value=raw))
    with pytest.raises(ObservationError, match="at least one named RGB"):
        RobotAdapter(robot, clock=Clock()).observe()


def test_fixture_cleanup_is_idempotent_and_blocks_further_io():
    robot = FixtureRobot()
    adapter = RobotAdapter(robot)
    adapter.close()
    adapter.close()
    assert robot.closed
    assert robot.commands == []
    with pytest.raises(RuntimeError, match="closed"):
        adapter.observe()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.send(robot.state)


def test_adapter_never_calls_unverified_physical_teardown():
    robot = Mock()
    RobotAdapter(robot).close()
    robot.disconnect.assert_not_called()
    robot.close.assert_not_called()


def test_live_construction_blocked_before_import_or_open(rig, monkeypatch):
    rig.cameras = {"top": {"type": "opencv", "index_or_path": 0}}
    rig.save()
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("lerobot", "i2rt", "cv2", "yamkit.arm")):
            pytest.fail(f"hardware module imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(LiveIntegrationError, match="sensor acquisition freshness is unavailable") as error:
        make_live_robot(rig.path, "left_follower")
    assert "read_latest()" in str(error.value)
    assert "No arm or camera was opened" in str(error.value)


def test_validate_live_config_before_integration_block(rig):
    with pytest.raises(ValueError, match="at least one camera"):
        make_live_robot(rig.path, "left_follower")
    with pytest.raises(ValueError, match="requires one follower"):
        make_live_robot(rig.path, "left_leader")
    with pytest.raises(KeyError, match="not in rig"):
        make_live_robot(rig.path, "unknown")


@pytest.mark.parametrize("speed", [0, -1, float("nan"), True])
def test_live_config_preserves_valid_speed_limits(rig, speed):
    rig.cameras = {"top": {"type": "opencv", "index_or_path": 0}}
    rig.control.max_joint_speed = speed
    rig.save()
    with pytest.raises(ValueError, match="control.max_joint_speed"):
        validate_rig(rig.path, "left_follower")


@pytest.mark.parametrize("content", ["[", "plain scalar", "[nonempty]", "arms: [invalid]"])
def test_malformed_rig_fails_with_configuration_error(tmp_path, content):
    path = tmp_path / "rig.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match="invalid rig YAML/structure"):
        validate_rig(path, "left_follower")


@pytest.mark.parametrize("cameras", [["top"], {GRIPPER_KEY: {}}, {"top": "invalid"}])
def test_invalid_camera_structure_is_a_configuration_error(rig, cameras):
    rig.cameras = cameras
    rig.save()
    with pytest.raises(ValueError, match="map names to camera configuration"):
        validate_rig(rig.path, "left_follower")


@pytest.mark.parametrize("bimanual", [False, True])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_public_plugin_no_home_cleanup_releases_arms_despite_camera_fault(
    rig, fake_connect, monkeypatch, bimanual, failure,
):
    from lerobot_robot_yamkit import BiYamFollowerConfig, YamFollowerConfig
    from lerobot_robot_yamkit.yam_follower import BiYamFollower, YamFollower

    rig.control.home_speed = 0
    rig.save()
    robot = (BiYamFollower(BiYamFollowerConfig(rig=str(rig.path))) if bimanual else
             YamFollower(YamFollowerConfig(rig=str(rig.path), arm="left_follower")))
    robot.connect()
    handles = list(robot._sides.values()) if bimanual else [robot._h]
    for handle in handles:
        handle.home_speed = 1  # cleanup must skip an otherwise enabled home move
        monkeypatch.setattr(handle.arm, "go_home", Mock(side_effect=AssertionError("unexpected home")))
    healthy = Mock(is_connected=False, thread=None)
    broken = Mock(disconnect=Mock(side_effect=failure("camera failed")), thread=None)
    robot._opened_cameras[:] = [broken, healthy]
    preview = robot._preview = Mock()
    commands_before = {name: len(arm.commands) for name, arm in fake_connect.items()}

    with pytest.raises(failure, match="camera failed"):
        robot.disconnect(home=False)

    healthy.disconnect.assert_called_once()
    broken.disconnect.assert_called_once()
    preview.close.assert_called_once()
    assert all(arm.closed for arm in fake_connect.values())
    assert {name: len(arm.commands) for name, arm in fake_connect.items()} == commands_before
    assert all(handle.arm is None for handle in handles)


def test_public_plugin_observation_cannot_prove_cached_frame_freshness(rig, fake_connect):
    from lerobot_robot_yamkit import YamFollowerConfig
    from lerobot_robot_yamkit.yam_follower import YamFollower

    rig.control.home_speed = 0
    rig.save()
    robot = YamFollower(YamFollowerConfig(rig=str(rig.path), arm="left_follower"))
    robot.connect()
    cached = np.zeros((5, 5, 3), dtype=np.uint8)
    robot.cameras = {"top": Mock(read_latest=Mock(return_value=cached), is_connected=True)}
    try:
        raw = robot.get_observation()
        assert raw["top"] is cached
        assert METADATA_KEY not in raw
        with pytest.raises(ObservationError, match="acquisition freshness contract"):
            RobotAdapter(robot, clock=Clock()).observe()
        assert fake_connect["left_follower"].commands == []
    finally:
        robot.disconnect(home=False)
