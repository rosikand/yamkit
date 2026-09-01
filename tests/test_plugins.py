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
