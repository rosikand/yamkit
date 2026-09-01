import numpy as np
import pytest

from yamkit.arm import YamArm
from yamkit.config import ArmSpec

from .conftest import FakeRobot


@pytest.fixture
def follower():
    spec = ArmSpec(name="f", role="follower", gripper="linear_4310", can_serial="x")
    robot = FakeRobot(7, gripper=True)
    return YamArm(spec, "can0", robot, max_joint_speed=1.0, max_gripper_speed=2.0), robot


def test_read_follower(follower):
    arm, robot = follower
    robot.pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    st = arm.read()
    assert np.allclose(st.q, robot.pos[:6]) and st.gripper == pytest.approx(0.7) and st.buttons is None
    assert st.vector().shape == (7,)


def test_speed_clamp_first_command(follower):
    arm, robot = follower
    sent = arm.command(np.full(6, 1.0), gripper=0.0)
    # first command: dt=0.01 s -> max 0.01 rad step from measured zeros; gripper 0.02 from 1.0? no: measured gripper is 0.0
    assert np.all(np.abs(sent[:6]) <= 0.01 + 1e-9)
    assert robot.commands[-1].shape == (7,)


def test_unlimited_command_and_move_to(follower):
    arm, robot = follower
    sent = arm.command(np.full(6, 0.5), gripper=0.25, limit_speed=False)
    assert np.allclose(sent, [0.5] * 6 + [0.25])
    arm.move_to(np.zeros(6), 1.0, duration=0.05, hz=100)
    assert np.allclose(robot.pos, [0.0] * 6 + [1.0], atol=1e-6)


def test_gripper_default_keeps_last(follower):
    arm, _robot = follower
    arm.command(np.zeros(6), gripper=0.3, limit_speed=False)
    sent = arm.command(np.zeros(6), None, limit_speed=False)
    assert sent[-1] == pytest.approx(0.3)


def test_gains_and_close(follower):
    arm, robot = follower
    arm.zero_torque()
    assert arm._gains_zeroed and np.all(robot.kp == 0)
    arm.command(np.zeros(6), limit_speed=False)  # restores gains before commanding
    assert np.all(robot.kp == 80.0)
    arm.scale_gains(0.2, 0.0)
    assert np.allclose(robot.kp, 16.0) and np.all(robot.kd == 0)
    arm.close(settle_s=0)
    assert robot.closed and robot.idle_calls == 1


def test_leader_read_handle():
    spec = ArmSpec(name="l", role="leader", gripper="yam_teaching_handle", can_serial="x")
    robot = FakeRobot(6, gripper=False, handle=True)
    robot.encoder[0].position = 0.25
    robot.encoder[0].io_inputs = [1, 0]
    arm = YamArm(spec, "can1", robot)
    st = arm.read()
    assert st.gripper == pytest.approx(0.75) and st.buttons == (True, False)
    assert arm.command(np.ones(6), limit_speed=False).shape == (6,)
