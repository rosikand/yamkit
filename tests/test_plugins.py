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
