import numpy as np
import pytest


def _register():
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()


def test_plugins_discovered_by_lerobot():
    _register()
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

    assert {"yam_follower", "bi_yam_follower"} <= set(RobotConfig.get_known_choices())
    assert {"yam_leader", "bi_yam_leader"} <= set(TeleoperatorConfig.get_known_choices())


def test_follower_features_and_io(rig, fake_connect, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lr"))
    from lerobot.robots.utils import make_robot_from_config
    from lerobot_robot_yamkit import YamFollowerConfig

    robot = make_robot_from_config(YamFollowerConfig(rig=str(rig.path), arm="left_follower"))
    assert list(robot.action_features) == [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
    assert robot.observation_features == robot.action_features  # no cameras in the test rig
    robot.connect()
    obs = robot.get_observation()
    assert set(obs) == set(robot.observation_features)
    sent = robot.send_action({**{f"joint_{i}.pos": 0.0 for i in range(1, 7)}, "gripper.pos": 0.5})
    assert set(sent) == set(robot.action_features)
    robot.disconnect()
    assert fake_connect["left_follower"].closed


def test_bimanual_follower_and_leader(rig, fake_connect, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lr"))
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.teleoperators.utils import make_teleoperator_from_config
    from lerobot_robot_yamkit import BiYamFollowerConfig
    from lerobot_teleoperator_yamkit import BiYamLeaderConfig

    robot = make_robot_from_config(BiYamFollowerConfig(rig=str(rig.path)))
    teleop = make_teleoperator_from_config(BiYamLeaderConfig(rig=str(rig.path)))
    assert set(teleop.action_features) == set(robot.action_features)  # leader keys drive follower keys 1:1
    assert "left_gripper.pos" in robot.action_features and "right_joint_6.pos" in robot.action_features
    teleop.connect()
    robot.connect()
    fake_connect["left_leader"].pos = np.array([0.3, 0, 0, 0, 0, 0])
    act = teleop.get_action()
    assert act["left_joint_1.pos"] == pytest.approx(0.3)
    sent = robot.send_action(act)
    assert set(sent) == set(robot.action_features)
    robot.disconnect()
    teleop.disconnect()


def test_role_mismatch(rig):
    from lerobot_robot_yamkit import YamFollowerConfig
    from lerobot_robot_yamkit.yam_follower import YamFollower

    with pytest.raises(ValueError):
        YamFollower(YamFollowerConfig(rig=str(rig.path), arm="left_leader"))


def test_plugins_park_arms_on_connect_and_disconnect(rig, fake_connect, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lr"))
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.teleoperators.utils import make_teleoperator_from_config
    from lerobot_robot_yamkit import YamFollowerConfig
    from lerobot_teleoperator_yamkit import YamLeaderConfig

    rig.control.home_speed = rig.control.leader_home_speed = 50.0
    rig.save()
    fake_connect.presets["left_follower"] = np.array([0.3] * 6 + [0.5])
    fake_connect.presets["left_leader"] = np.full(6, 0.3)
    robot = make_robot_from_config(YamFollowerConfig(rig=str(rig.path), arm="left_follower"))
    teleop = make_teleoperator_from_config(YamLeaderConfig(rig=str(rig.path), arm="left_leader"))
    robot.connect()
    teleop.connect()
    f, l = fake_connect["left_follower"], fake_connect["left_leader"]
    assert np.allclose(f.pos[:6], 0) and f.pos[6] == pytest.approx(0.5)  # follower homed, gripper untouched
    assert np.allclose(l.pos, 0) and l.idle_calls >= 1 and np.all(l.kp == 80.0)  # leader homed compliantly, released
    f.pos[:6] = 0.4
    l.pos[:] = 0.4
    robot.disconnect()
    teleop.disconnect()
    assert np.allclose(f.pos[:6], 0) and np.allclose(l.pos, 0) and f.closed and l.closed


def test_plugins_do_not_move_arms_when_home_speed_is_zero(rig, fake_connect, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lr"))
    from lerobot.robots.utils import make_robot_from_config
    from lerobot_robot_yamkit import YamFollowerConfig

    rig.control.home_speed = 0.0
    rig.save()
    fake_connect.presets["left_follower"] = np.array([0.3] * 6 + [0.5])
    robot = make_robot_from_config(YamFollowerConfig(rig=str(rig.path), arm="left_follower"))
    robot.connect()
    robot.disconnect()
    f = fake_connect["left_follower"]
    assert f.commands == [] and np.allclose(f.pos[:6], 0.3) and f.closed


def test_bimanual_plugins_park_both_arms_together(rig, fake_connect, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lr"))
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.teleoperators.utils import make_teleoperator_from_config
    from lerobot_robot_yamkit import BiYamFollowerConfig
    from lerobot_teleoperator_yamkit import BiYamLeaderConfig

    rig.control.home_speed = rig.control.leader_home_speed = 50.0
    rig.save()
    for n in ("left_leader", "left_follower", "right_leader", "right_follower"):
        fake_connect.presets[n] = np.full(6, 0.3)
    robot = make_robot_from_config(BiYamFollowerConfig(rig=str(rig.path)))
    teleop = make_teleoperator_from_config(BiYamLeaderConfig(rig=str(rig.path)))
    robot.connect()
    teleop.connect()
    for n, r in fake_connect.items():
        assert np.allclose(r.pos[:6], 0), n
        r.pos[:6] = 0.4
    robot.disconnect()
    teleop.disconnect()
    for n, r in fake_connect.items():
        assert np.allclose(r.pos[:6], 0) and r.closed, n


@pytest.fixture
def make_plugin(rig, fake_connect, tmp_path, monkeypatch):
    """Construct any of the four plugins without homing or touching hardware."""
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lr"))
    from lerobot_robot_yamkit import BiYamFollowerConfig, YamFollowerConfig
    from lerobot_robot_yamkit.yam_follower import BiYamFollower, YamFollower
    from lerobot_teleoperator_yamkit import BiYamLeaderConfig, YamLeaderConfig
    from lerobot_teleoperator_yamkit.yam_leader import BiYamLeader, YamLeader

    rig.control.home_speed = rig.control.leader_home_speed = 0.0
    rig.save()
    choices = {
        "follower": (YamFollower, YamFollowerConfig, {"arm": "left_follower"}),
        "bi_follower": (BiYamFollower, BiYamFollowerConfig, {}),
        "leader": (YamLeader, YamLeaderConfig, {"arm": "left_leader"}),
        "bi_leader": (BiYamLeader, BiYamLeaderConfig, {}),
    }

    def make(kind, **kwargs):
        cls, config, defaults = choices[kind]
        return cls(config(rig=str(rig.path), **(defaults | kwargs)))

    return make


def _handles(plugin):
    return list(plugin._sides.values()) if hasattr(plugin, "_sides") else [plugin._h]


@pytest.mark.parametrize("kind", ["follower", "bi_follower"])
@pytest.mark.parametrize("defect", ["missing_joint", "missing_gripper", "extra_joint", "nan", "inf", "vector", "text", "numeric_text", "boolean", "bounds"])
def test_rejected_plugin_actions_never_command_or_restore_gains(make_plugin, kind, defect):
    plugin = make_plugin(kind)
    plugin.connect()
    handles = _handles(plugin)
    for h in handles:
        h.arm.zero_torque()
    action = dict.fromkeys(plugin.action_features, 0.1)
    prefix = "right_" if kind == "bi_follower" else ""
    key = prefix + "joint_1.pos"
    if defect == "missing_joint":
        action.pop(key)
    elif defect == "missing_gripper":
        action.pop(prefix + "gripper.pos")
    elif defect == "extra_joint":
        action[prefix + "joint_7.pos"] = 0.0
    else:
        action[key] = {"nan": np.nan, "inf": np.inf, "vector": [0.1], "text": "bad", "numeric_text": "0.1", "boolean": True, "bounds": 100.0}[defect]
    try:
        with pytest.raises(ValueError):
            plugin.send_action(action)
        for h in handles:
            assert h.arm.robot.commands == []
            assert np.all(h.arm.robot.kp == 0), "rejecting an action must not restore gains"
    finally:
        plugin.disconnect(home=False)


@pytest.mark.parametrize("defect", ["measured_nan", "measured_bounds", "previous_nan", "previous_shape"])
def test_bimanual_prevalidates_right_state_before_left_command(make_plugin, defect):
    plugin = make_plugin("bi_follower")
    plugin.connect()
    left, right = _handles(plugin)
    left.arm.zero_torque()
    right.arm.zero_torque()
    if defect.startswith("measured"):
        right.arm.robot.pos[0] = np.nan if defect == "measured_nan" else 100.0
    else:
        right.arm._last_cmd = np.full(7, np.nan) if defect == "previous_nan" else np.zeros(8)
    try:
        with pytest.raises(ValueError):
            plugin.send_action(dict.fromkeys(plugin.action_features, 0.1))
        assert left.arm.robot.commands == right.arm.robot.commands == []
        assert np.all(left.arm.robot.kp == 0) and np.all(right.arm.robot.kp == 0)
    finally:
        plugin.disconnect(home=False)


@pytest.mark.parametrize("kind", ["follower", "bi_follower", "leader", "bi_leader"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_plugins_cleanup_homing_startup_failure(make_plugin, fake_connect, monkeypatch, kind, error_type):
    from yamkit.arm import YamArm

    plugin = make_plugin(kind)
    for h in _handles(plugin):
        h.home_speed = 0.25

    def fail_home(*args, **kwargs):
        raise error_type("home failed")

    monkeypatch.setattr(YamArm, "go_home", fail_home)
    with pytest.raises(error_type, match="home failed"):
        plugin.connect()
    assert fake_connect and all(r.closed for r in fake_connect.values())
    assert all(h.arm is None for h in _handles(plugin))
    plugin.disconnect()  # partial startup was fully torn down


@pytest.mark.parametrize("kind", ["bi_follower", "bi_leader"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_plugins_cleanup_first_arm_when_second_connect_fails(make_plugin, fake_connect, monkeypatch, kind, error_type):
    from yamkit.arm import YamArm

    plugin = make_plugin(kind)
    connect = YamArm.connect

    def fail_second(spec, channel, **kwargs):
        if spec.name.startswith("right"):
            raise error_type("second open failed")
        return connect(spec, channel, **kwargs)

    monkeypatch.setattr(YamArm, "connect", fail_second)
    with pytest.raises(error_type, match="second open failed"):
        plugin.connect()
    assert len(fake_connect) == 1 and all(r.closed for r in fake_connect.values())
    plugin.disconnect(home=False)


class _FakeCamera:
    def __init__(self, *, connect_error=None, disconnect_error=None):
        self.is_connected = False
        self.connect_error = connect_error
        self.disconnect_error = disconnect_error
        self.disconnect_calls = 0

    def connect(self):
        self.is_connected = True  # camera may acquire its device before raising
        if self.connect_error is not None:
            raise self.connect_error

    def disconnect(self):
        self.disconnect_calls += 1
        self.is_connected = False
        if self.disconnect_error is not None:
            error, self.disconnect_error = self.disconnect_error, None
            raise error


@pytest.mark.parametrize("kind", ["follower", "bi_follower"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_plugins_cleanup_partial_camera_startup(make_plugin, fake_connect, kind, error_type):
    plugin = make_plugin(kind)
    first = _FakeCamera(disconnect_error=RuntimeError("camera close failed"))
    second = _FakeCamera(connect_error=error_type("camera open failed"))
    unused = _FakeCamera()
    plugin.cameras = {"first": first, "second": second, "unused": unused}
    with pytest.raises(error_type, match="camera open failed"):
        plugin.connect()
    assert first.disconnect_calls == second.disconnect_calls == 1
    assert unused.disconnect_calls == 0
    assert all(r.closed and not r.commands for r in fake_connect.values())
    plugin.disconnect()
    assert first.disconnect_calls == 2  # retry the camera that reported a cleanup failure
    assert second.disconnect_calls == 1


@pytest.mark.parametrize("kind", ["follower", "bi_follower"])
def test_camera_that_cleans_own_failed_connect_can_reconnect(make_plugin, fake_connect, kind):
    from lerobot.utils.errors import DeviceNotConnectedError

    class Camera(_FakeCamera):
        def connect(self):
            if self.connect_error is not None:
                error, self.connect_error = self.connect_error, None
                raise error  # LeRobot cleaned up internally before surfacing the error.
            super().connect()

        def disconnect(self):
            if not self.is_connected:
                raise DeviceNotConnectedError("already closed")
            super().disconnect()

    plugin = make_plugin(kind)
    camera = Camera(connect_error=RuntimeError("camera setup failed"))
    plugin.cameras = {"camera": camera}
    with pytest.raises(RuntimeError, match="camera setup failed"):
        plugin.connect()
    assert all(r.closed for r in fake_connect.values())
    assert plugin._opened_cameras == []
    plugin.disconnect(home=False)
    plugin.connect()
    assert plugin.is_connected
    plugin.disconnect(home=False)
    assert not camera.is_connected


@pytest.mark.parametrize("kind", ["follower", "bi_follower", "leader", "bi_leader"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_disconnect_attempts_every_arm_after_close_error(make_plugin, fake_connect, monkeypatch, kind, error_type):
    plugin = make_plugin(kind)
    plugin.connect()
    first = _handles(plugin)[0].arm
    close = first.close

    def fail_close(*args, **kwargs):
        close(*args, **kwargs)
        raise error_type("close failed")

    monkeypatch.setattr(first, "close", fail_close)
    with pytest.raises(error_type, match="close failed"):
        plugin.disconnect(home=False)
    assert all(r.closed for r in fake_connect.values())
    assert all(h.arm is None for h in _handles(plugin))
    plugin.disconnect(home=False)


@pytest.mark.parametrize("kind", ["bi_follower", "bi_leader"])
def test_failed_sdk_close_retains_arm_for_retry(make_plugin, monkeypatch, kind):
    plugin = make_plugin(kind)
    plugin.connect()
    left, right = _handles(plugin)
    arm = right.arm
    close = arm.robot.close

    def fail_close():
        raise RuntimeError("SDK still running")

    monkeypatch.setattr(arm.robot, "close", fail_close)
    with pytest.raises(RuntimeError, match="SDK still running"):
        plugin.disconnect(home=False)
    assert left.arm is None and right.arm is arm and not arm._closed
    with pytest.raises(RuntimeError, match="previous"):
        plugin.connect()
    assert left.arm is None, "reconnect must not open another arm while cleanup is incomplete"
    monkeypatch.setattr(arm.robot, "close", close)
    plugin.disconnect(home=False)
    assert right.arm is None and arm._closed


@pytest.mark.parametrize("kind", ["follower", "bi_follower"])
def test_camera_disconnect_failure_does_not_skip_other_resources(make_plugin, fake_connect, kind):
    plugin = make_plugin(kind)
    first = _FakeCamera()
    second = _FakeCamera(disconnect_error=KeyboardInterrupt("camera close interrupted"))
    plugin.cameras = {"first": first, "second": second}
    plugin.connect()
    with pytest.raises(KeyboardInterrupt, match="camera close interrupted"):
        plugin.disconnect()
    assert first.disconnect_calls == second.disconnect_calls == 1
    assert all(r.closed for r in fake_connect.values())


@pytest.mark.parametrize("kind", ["follower", "bi_follower", "leader", "bi_leader"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_home_disconnect_failure_still_closes_every_arm(make_plugin, fake_connect, monkeypatch, kind, error_type):
    plugin = make_plugin(kind)
    plugin.connect()

    def fail_home(*args, **kwargs):
        raise error_type("home interrupted")

    for h in _handles(plugin):
        h.home_speed = 0.25
        monkeypatch.setattr(h.arm, "go_home", fail_home)
    with pytest.raises(error_type, match="home interrupted"):
        plugin.disconnect()
    assert all(r.closed for r in fake_connect.values())
    plugin.disconnect()


@pytest.mark.parametrize("kind", ["follower", "bi_follower", "leader", "bi_leader"])
def test_no_home_disconnect_is_idempotent(make_plugin, fake_connect, monkeypatch, kind):
    plugin = make_plugin(kind)
    plugin.connect()
    for h in _handles(plugin):
        h.home_speed = 0.25
        monkeypatch.setattr(h.arm, "go_home", lambda **kw: pytest.fail("no-home cleanup must not move"))
    plugin.disconnect(home=False)
    plugin.disconnect()
    assert all(r.closed and r.commands == [] for r in fake_connect.values())


@pytest.mark.parametrize("kind", ["follower", "bi_follower"])
@pytest.mark.parametrize("field", ["max_joint_speed", "max_gripper_speed"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_plugin_override_rejected_before_open(make_plugin, fake_connect, kind, field, value):
    with pytest.raises(ValueError):
        make_plugin(kind, **{field: value})
    assert fake_connect == {}


@pytest.mark.parametrize("kind", ["follower", "bi_follower", "leader", "bi_leader"])
def test_mutated_invalid_control_rejected_before_open(make_plugin, fake_connect, kind):
    plugin = make_plugin(kind)
    plugin.rig.control.home_speed = np.nan
    with pytest.raises(ValueError):
        plugin.connect()
    assert fake_connect == {}


@pytest.mark.parametrize("kind", ["bi_follower", "bi_leader"])
def test_bimanual_duplicate_arm_rejected_before_open(make_plugin, fake_connect, kind):
    arm = "left_follower" if kind == "bi_follower" else "left_leader"
    with pytest.raises(ValueError, match="two different arms"):
        make_plugin(kind, left=arm, right=arm)
    assert fake_connect == {}


@pytest.mark.parametrize("kind", ["leader", "bi_leader"])
def test_missing_leader_trigger_cannot_become_open_gripper_action(make_plugin, kind):
    plugin = make_plugin(kind)
    plugin.connect()
    _handles(plugin)[-1].arm.robot.encoder = None
    try:
        with pytest.raises(ValueError, match="trigger is missing"):
            plugin.get_action()
    finally:
        plugin.disconnect(home=False)
