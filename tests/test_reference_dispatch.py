"""Dedicated literal reference dispatch through real YamArm validation and fake motors."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from lerobot_robot_yamkit import BiYamFollowerConfig
from lerobot_robot_yamkit.yam_follower import BiYamFollower

from yamkit import arm as arm_module
from yamkit import remote_rollout
from yamkit.inference.client import RemoteFault
from yamkit.teleop_control import GatedAction, LeaderAction


@pytest.fixture
def reference_robot(rig, fake_connect, monkeypatch):
    rig.control.home_speed = 0.0
    rig.save()
    config = BiYamFollowerConfig(rig=str(rig.path))
    stop = threading.Event()
    config._session_shutdown_event = stop
    robot = BiYamFollower(config)
    robot.connect()  # fake_connect intercepts every arm; this rig has no cameras.
    clock = [100.0]
    monkeypatch.setattr(arm_module, "time", SimpleNamespace(
        monotonic=lambda: clock[0], time=lambda: clock[0], sleep=lambda _: None))
    case = SimpleNamespace(robot=robot, stop=stop, clock=clock,
                           arms={side: handle.arm for side, handle in robot._sides.items()},
                           motors={side: handle.arm.robot for side, handle in robot._sides.items()})
    try:
        yield case
    finally:
        robot.disconnect(home=False)


def target(robot):
    return {name: .8 if "gripper" in name else .2 for name in robot.action_features}


def assert_no_commands(case):
    assert all(motor.commands == [] for motor in case.motors.values())


@pytest.mark.parametrize("command_age", [.001, .88])
def test_reference_sends_exact_targets_without_fast_or_stale_clamp(reference_robot, monkeypatch, command_age):
    case = reference_robot
    events = []
    for side, arm in case.arms.items():
        # Measured lag and cached commands intentionally differ. A normal stale
        # send would rebase; a rapid send would earn only 3 * 1ms of movement.
        arm.robot.pos[:] = .05
        arm._last_cmd = np.full(7, .1)
        arm._last_cmd_t = case.clock[0] - command_age
        original_validate, original_command = arm.validate_command, arm.command

        def validate(q, gripper, *, limit_speed, side=side, original=original_validate):
            events.append((side, "validate", limit_speed))
            return original(q, gripper, limit_speed=limit_speed)

        def command(q, gripper, *, limit_speed, side=side, original=original_command):
            events.append((side, "command", limit_speed))
            return original(q, gripper, limit_speed=limit_speed)

        monkeypatch.setattr(arm, "validate_command", validate)
        monkeypatch.setattr(arm, "command", command)

    requested = target(case.robot)
    sent = case.robot.send_reference_action(requested)
    assert sent == requested
    assert events == [("left", "validate", False), ("right", "validate", False),
                      ("left", "command", False), ("right", "command", False)]
    for side, arm in case.arms.items():
        expected = [requested[f"{side}_{name}.pos"] for name in case.robot._sides[side].names]
        np.testing.assert_array_equal(case.motors[side].commands, [expected])
        np.testing.assert_array_equal(arm._last_cmd, expected)
        assert arm._last_cmd_t == case.clock[0]
        assert arm.max_joint_speed == arm.max_gripper_speed == 3.0
    assert arm_module.MAX_COMMAND_DT == .01 and arm_module.STALE_COMMAND_S == .5


def test_normal_dispatch_still_uses_original_speed_clamps(reference_robot):
    case = reference_robot
    requested = target(case.robot)
    sent = case.robot.send_action(requested)
    assert sent != requested
    assert all(value == pytest.approx(.03) for value in sent.values())
    assert all(arm.max_joint_speed == arm.max_gripper_speed == 3.0 for arm in case.arms.values())


@pytest.mark.parametrize("kind", ["gated", "leader", "list"])
def test_reference_rejects_operator_metadata_and_non_plain_actions(reference_robot, kind):
    case = reference_robot
    action = target(case.robot)
    acknowledgements = []
    if kind == "gated":
        action = GatedAction(action, capture_hold={"left_", "right_"}, on_sent=acknowledgements.append)
    elif kind == "leader":
        action = LeaderAction(action, buttons={"left_": (True,), "right_": (True,)})
    else:
        action = list(action.values())
    with pytest.raises(TypeError, match="plain dictionary"):
        case.robot.send_reference_action(action)
    assert_no_commands(case)
    assert acknowledgements == []


@pytest.mark.parametrize("defect", ["missing", "extra", "nan", "bounds", "gripper", "boolean"])
def test_reference_rejects_invalid_right_target_before_either_send(reference_robot, defect):
    case = reference_robot
    action = target(case.robot)
    if defect == "missing":
        action.pop("right_gripper.pos")
    elif defect == "extra":
        action["right_joint_7.pos"] = 0.0
    elif defect == "gripper":
        action["right_gripper.pos"] = 1.01
    else:
        action["right_joint_1.pos"] = {"nan": np.nan, "bounds": 100.0, "boolean": True}[defect]
    for arm in case.arms.values():
        arm.zero_torque()
    with pytest.raises(ValueError):
        case.robot.send_reference_action(action)
    assert_no_commands(case)
    assert all(np.all(motor.kp == 0) for motor in case.motors.values())


@pytest.mark.parametrize("defect", ["measured_nan", "measured_bounds", "measured_gripper", "previous_nan"])
def test_reference_prevalidates_right_measured_and_previous_state(reference_robot, defect):
    case = reference_robot
    right = case.arms["right"]
    if defect == "previous_nan":
        right._last_cmd = np.full(7, np.nan)
    elif defect == "measured_gripper":
        right.robot.pos[6] = 1.01
    else:
        right.robot.pos[0] = np.nan if defect == "measured_nan" else 100.0
    with pytest.raises(ValueError):
        case.robot.send_reference_action(target(case.robot))
    assert_no_commands(case)


@pytest.mark.parametrize("after_left", [False, True])
def test_reference_observes_session_stop_before_each_arm(reference_robot, monkeypatch, after_left):
    case = reference_robot
    if after_left:
        original = case.motors["left"].command_joint_pos

        def send_then_stop(value):
            original(value)
            case.stop.set()

        monkeypatch.setattr(case.motors["left"], "command_joint_pos", send_then_stop)
    else:
        case.stop.set()
    with pytest.raises(RuntimeError, match="stopped"):
        case.robot.send_reference_action(target(case.robot))
    assert len(case.motors["left"].commands) == int(after_left)
    assert case.motors["right"].commands == []


def test_reference_returns_actual_command_feedback(reference_robot, monkeypatch):
    case = reference_robot
    original = case.arms["right"].command

    def changed_feedback(*args, **kwargs):
        result = original(*args, **kwargs)
        result[0] += .001
        return result

    monkeypatch.setattr(case.arms["right"], "command", changed_feedback)
    requested = target(case.robot)
    sent = case.robot.send_reference_action(requested)
    assert sent["right_joint_1.pos"] == pytest.approx(requested["right_joint_1.pos"] + .001)
    # The controller must see the intervention and decide whether to invalidate.
    assert sent != requested


def test_reference_surfaces_partial_dispatch_failure(reference_robot, monkeypatch):
    case = reference_robot

    def fail_send(value):
        raise RuntimeError("injected right SDK failure")

    monkeypatch.setattr(case.motors["right"], "command_joint_pos", fail_send)
    with pytest.raises(RuntimeError, match="right SDK failure"):
        case.robot.send_reference_action(target(case.robot))
    assert len(case.motors["left"].commands) == 1 and case.motors["right"].commands == []


def test_reference_rechecks_phase_deadline_after_both_arm_validation(reference_robot, monkeypatch):
    case = reference_robot
    monkeypatch.setattr(remote_rollout, "time", SimpleNamespace(monotonic=lambda: case.clock[0]))
    wrapper = remote_rollout._StoppableRobot(case.robot, case.stop, reference_dispatch=True)
    wrapper.action_deadline = lambda: 100.01
    original = case.arms["right"].validate_command
    validated = []

    def delayed_right_validation(*args, **kwargs):
        result = original(*args, **kwargs)
        validated.append(True)
        case.clock[0] = 100.02
        return result

    monkeypatch.setattr(case.arms["right"], "validate_command", delayed_right_validation)
    with pytest.raises(RemoteFault, match="expired before hardware dispatch"):
        wrapper.send_action(target(case.robot))
    assert validated == [True] and case.stop.is_set()
    assert_no_commands(case)


def test_reference_rechecks_session_between_left_and_right_commands(reference_robot, monkeypatch):
    case = reference_robot
    expired, checks = [], []
    wrapper = remote_rollout._StoppableRobot(case.robot, case.stop, reference_dispatch=True)

    def session_check():
        checks.append(True)
        if expired:
            raise RemoteFault("injected session expiry after left command")

    wrapper.session_check = session_check
    original = case.motors["left"].command_joint_pos

    def send_then_expire(value):
        original(value)
        expired.append(True)

    monkeypatch.setattr(case.motors["left"], "command_joint_pos", send_then_expire)
    with pytest.raises(RemoteFault, match="session expiry after left"):
        wrapper.send_action(target(case.robot))
    assert len(checks) == 3  # Wrapper entry, left dispatch, then rejected right dispatch.
    assert len(case.motors["left"].commands) == 1 and case.motors["right"].commands == []
    assert case.stop.is_set()
