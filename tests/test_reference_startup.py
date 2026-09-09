"""Reference startup uses real home/validation with fake motors and transport."""
# ruff: noqa: F811 -- imported pytest fixtures are injected by parameter name

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from lerobot_robot_yamkit import BiYamFollowerConfig
from lerobot_robot_yamkit.yam_follower import BiYamFollower, _home_together

from tests.conftest import FakeRobot
from tests.test_http_policy_readiness import rollout_config, transport  # noqa: F401
from yamkit import arm as arm_module
from yamkit.config import RigConfig
from yamkit.inference.client import InvalidatedRequest
from yamkit.remote_rollout import run_remote_rollout


@pytest.fixture
def startup_robot(rig, fake_connect):
    rig.control.home_speed = .5
    rig.arms["left_follower"].rest_pose = [.02] * 6
    rig.arms["right_follower"].rest_pose = [.03] * 6
    rig.save()
    fake_connect.presets = {"left_follower": np.array([0.] * 6 + [.365]),
                            "right_follower": np.array([0.] * 6 + [.72])}
    config = BiYamFollowerConfig(rig=str(rig.path))
    config._reference_startup = True
    config._session_shutdown_event = threading.Event()
    robot = BiYamFollower(config)
    try:
        yield SimpleNamespace(robot=robot, config=config, motors=fake_connect)
    finally:
        robot.disconnect(home=False)


def test_reference_opens_both_grippers_before_first_observation_and_caches_sent_home(startup_robot):
    case = startup_robot
    assert case.robot._reference_start_action is None
    case.robot.connect()
    observed = case.robot.get_observation()
    expected = {f"{side}_{name}.pos": float(value)
                for side, handle in case.robot._sides.items()
                for name, value in zip(handle.names, [*handle.spec.home_pose, 1.0], strict=True)}
    assert observed == pytest.approx(expected)
    assert case.robot._reference_start_action == expected
    for side, handle in case.robot._sides.items():
        commands = case.motors[handle.spec.name].commands
        assert len(commands) > 1
        assert commands[0][-1] < commands[-1][-1] == 1.0
        np.testing.assert_allclose(commands[-1], [expected[f"{side}_{name}.pos"] for name in handle.names])
        assert handle.home_job[1] == {"speed": .5}  # No persistent open-on-home override.


def test_default_connect_and_later_home_preserve_current_gripper(startup_robot):
    case = startup_robot
    case.config._reference_startup = False
    case.robot.connect()
    assert case.robot._reference_start_action is None
    for name, initial in [("left_follower", .365), ("right_follower", .72)]:
        assert all(command[-1] == initial for command in case.motors[name].commands)
    # A later ordinary home, including completion-home callers, preserves the
    # newly measured grip even after a reference startup opened it earlier.
    case.config._reference_startup = True
    _home_together(case.robot._sides.values(), gripper=1.0)
    for handle in case.robot._sides.values():
        handle.arm.robot.pos[-1] = .4
    _home_together(case.robot._sides.values())
    for handle in case.robot._sides.values():
        assert handle.arm.robot.commands[-1][-1] == .4
        assert handle.home_job[1] == {"speed": .5}


@pytest.mark.parametrize("when", ["before_connect", "during_home", "after_home"])
def test_stopped_reference_startup_never_publishes_cache(startup_robot, monkeypatch, when):
    case = startup_robot
    stop = case.config._session_shutdown_event
    case.robot._reference_start_action = {"obsolete": 1.0}
    if when == "before_connect":
        stop.set()
    elif when == "during_home":
        original = FakeRobot.command_joint_pos

        def stop_after_command(motor, command):
            original(motor, command)
            stop.set()

        monkeypatch.setattr(FakeRobot, "command_joint_pos", stop_after_command)
    else:
        original = arm_module.YamArm.go_home

        def stop_after_home(arm, *args, **kwargs):
            value = original(arm, *args, **kwargs)
            stop.set()
            return value

        monkeypatch.setattr(arm_module.YamArm, "go_home", stop_after_home)
    with pytest.raises(RuntimeError, match="stopped"):
        case.robot.connect()
    assert case.robot._reference_start_action is None
    assert all(motor.closed for motor in case.motors.values())
    if when == "before_connect":
        assert not case.motors


def test_reference_disabled_home_rejected_before_cameras_or_arms(startup_robot, monkeypatch):
    case = startup_robot
    case.robot._sides["right"].home_speed = 0
    monkeypatch.setattr(case.robot, "_connect_cameras", lambda: pytest.fail("must reject before cameras"))
    with pytest.raises(ValueError, match="enabled home"):
        case.robot.connect()
    assert not case.motors and case.robot._reference_start_action is None


def test_invalid_second_arm_state_prevents_all_startup_commands(startup_robot):
    case = startup_robot
    case.motors.presets["right_follower"][-1] = 1.1
    with pytest.raises(ValueError, match="gripper"):
        case.robot.connect()
    assert all(not motor.commands and motor.closed for motor in case.motors.values())
    assert case.robot._reference_start_action is None


def test_home_prevalidates_explicit_gripper_before_starting_any_worker(startup_robot):
    case = startup_robot
    for handle in case.robot._sides.values():
        handle.connect(home=False)
    left, right = [handle.arm for handle in case.robot._sides.values()]
    with pytest.raises(ValueError, match="gripper"):
        arm_module.go_home_all([(left, {"gripper": 1.0}), (right, {"gripper": 1.1})])
    assert all(not motor.commands for motor in case.motors.values())


def test_explicit_home_gripper_retains_existing_gripper_speed_bound(startup_robot, monkeypatch):
    case = startup_robot
    handle = case.robot._sides["left"]
    handle.connect(home=False)
    arm = handle.arm
    arm.max_gripper_speed = .2
    times = []
    clock = [100.0]
    original_send = arm.robot.command_joint_pos

    def sleep(seconds):
        clock[0] += seconds

    def send(command):
        times.append(clock[0])
        original_send(command)

    monkeypatch.setattr(arm_module, "time", SimpleNamespace(
        monotonic=lambda: clock[0], time=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(arm.robot, "command_joint_pos", send)
    arm.go_home(.5, gripper=1.0)
    values = np.array([.365, *[command[-1] for command in arm.robot.commands]])
    dt = np.diff([100.0, *times])
    assert np.all(np.abs(np.diff(values)) <= .2 * dt + 1e-9)
    assert clock[0] - 100 >= (1 - .365) / .2 - 1e-9
    assert values[-1] == 1.0


def test_reference_runtime_rejects_disabled_home_before_warmup(transport, rollout_config, fake_connect):
    rollout_config.policy.controller_mode = "reference"
    with pytest.raises(ValueError, match="enabled home"):
        run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert not fake_connect and transport.ready_count == 0 and transport.requests == []


def test_async_runtime_clears_a_preexisting_reference_startup_flag(
        transport, rollout_config, fake_connect, monkeypatch):
    rollout_config.robot._reference_startup = True
    rollout_config.policy.controller_mode = "async"
    captured = []

    def context_boundary(config, shutdown_event):
        captured.append(config.robot._reference_startup)
        raise RuntimeError("captured fake context boundary")

    monkeypatch.setattr("yamkit.remote_rollout.build_rollout_context", context_boundary)
    with pytest.raises(RuntimeError, match="fake context boundary"):
        run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert captured == [False] and not fake_connect and not transport.requests


@pytest.mark.parametrize("stop_during_home", [False, True])
def test_real_factory_starts_policy_only_after_open_reset(
        transport, rollout_config, fake_connect, monkeypatch, stop_during_home):
    rig = RigConfig.load(rollout_config.robot.rig)
    rig.control.home_speed = .5
    rig.save()
    rollout_config.policy.controller_mode = "reference"
    fake_connect.presets = {"left_follower": np.array([0.] * 6 + [.365]),
                            "right_follower": np.array([0.] * 6 + [.72])}
    stop = threading.Event()
    original = FakeRobot.command_joint_pos

    def command(motor, target):
        original(motor, target)
        if stop_during_home:
            stop.set()

    monkeypatch.setattr(FakeRobot, "command_joint_pos", command)

    def halt_first_policy_reply(response):
        assert all(motor.commands[-1][-1] == 1.0 for motor in fake_connect.values())
        stop.set()
        raise InvalidatedRequest("Stop after validating fake first policy state")

    transport.response_hook = halt_first_policy_reply
    if stop_during_home:
        with pytest.raises(RuntimeError, match="stopped"):
            run_remote_rollout(rollout_config, shutdown_event=stop)
        assert rollout_config.robot._runtime_robot._reference_start_action is None
        assert len(transport.requests) == 1  # Pre-hardware native warmup only.
    else:
        run_remote_rollout(rollout_config, shutdown_event=stop)
        assert len(transport.requests) == 2
        assert transport.requests[-1]["state"] == [0.] * 6 + [1.] + [0.] * 6 + [1.]
    assert all(motor.closed for motor in fake_connect.values())
