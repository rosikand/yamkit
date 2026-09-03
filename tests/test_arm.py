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


def test_joint_offsets_put_a_leader_in_its_followers_frame():
    spec = ArmSpec(name="l", role="leader", gripper="yam_teaching_handle", can_serial="x", joint_offsets=[0.1, 0, 0, 0, 0, 0])
    robot = FakeRobot(6, gripper=False, handle=True)
    arm = YamArm(spec, "can1", robot)
    assert arm.read().q[0] == pytest.approx(0.1)  # motors at raw 0 read as +0.1 (the follower's angle)
    arm.command(np.zeros(6), limit_speed=False)  # "same angle as the follower's zero"
    assert robot.commands[-1][0] == pytest.approx(-0.1)  # motors receive the raw target
    assert arm.read().q[0] == pytest.approx(0.0)
    assert np.allclose(arm.home_pose, 0)


def test_go_home_default_pose_rest_pose_and_gripper(follower, monkeypatch):
    from yamkit import arm as arm_mod

    monkeypatch.setattr(arm_mod, "HOME_MIN_S", 0.01)
    monkeypatch.setattr(arm_mod, "HOME_SETTLE_S", 0.0)
    arm, robot = follower
    robot.pos = np.array([0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.3])
    assert arm.go_home(speed=50.0) == pytest.approx(0.5)
    assert np.allclose(robot.pos[:6], 0) and robot.pos[6] == pytest.approx(0.3)  # gripper untouched
    assert robot.idle_calls == 0  # not released
    arm.spec.rest_pose = [0.1] * 6
    arm.go_home(speed=50.0, release=True)
    assert np.allclose(robot.pos[:6], 0.1) and robot.idle_calls == 1


def test_go_home_compliant_and_interrupt(monkeypatch):
    from yamkit import arm as arm_mod

    monkeypatch.setattr(arm_mod, "HOME_MIN_S", 0.01)
    monkeypatch.setattr(arm_mod, "HOME_SETTLE_S", 0.0)
    spec = ArmSpec(name="l", role="leader", gripper="yam_teaching_handle", can_serial="x")
    robot = FakeRobot(6, gripper=False, handle=True)
    arm = YamArm(spec, "can1", robot)
    robot.pos = np.ones(6)
    seen_kp = []
    orig = robot.command_joint_pos
    robot.command_joint_pos = lambda q: (seen_kp.append(robot.kp.copy()), orig(q))
    arm.go_home(speed=50.0, compliant=True, release=True)
    assert np.allclose(robot.pos, 0) and robot.idle_calls == 1
    assert np.allclose(seen_kp[0], 80.0 * arm_mod.COMPLIANT_KP_SCALE)  # moved with low gains
    assert np.all(robot.kp == 80.0)  # restored afterwards

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(arm, "move_to", interrupted)
    with pytest.raises(KeyboardInterrupt):
        arm.go_home(speed=50.0, compliant=True)
    assert robot.idle_calls == 2 and np.all(robot.kp == 80.0)  # released, gains restored, then re-raised


def test_go_home_all_runs_arms_together_and_ctrl_c_releases_all(monkeypatch):
    import _thread
    import threading
    import time

    from yamkit import arm as arm_mod

    monkeypatch.setattr(arm_mod, "HOME_MIN_S", 0.01)
    monkeypatch.setattr(arm_mod, "HOME_SETTLE_S", 0.0)
    arms = []
    for i in range(4):
        spec = ArmSpec(name=f"a{i}", role="follower", gripper="linear_4310", can_serial=str(i))
        robot = FakeRobot(7, gripper=True)
        robot.pos[:6] = 0.3
        arms.append((YamArm(spec, f"can{i}", robot), robot))
    t0 = time.monotonic()
    arm_mod.go_home_all([(a, {"speed": 1.0, "release": True}) for a, _ in arms])  # 0.3 s each
    assert time.monotonic() - t0 < 0.9
    assert all(np.allclose(r.pos[:6], 0) and r.idle_calls == 1 for _, r in arms)

    for _, r in arms:
        r.pos[:6] = 0.3
    threading.Timer(0.15, _thread.interrupt_main).start()
    with pytest.raises(KeyboardInterrupt):
        arm_mod.go_home_all([(a, {"speed": 0.01}) for a, _ in arms])  # would take 30 s
    assert all(r.pos[0] > 0.25 and r.idle_calls == 2 for _, r in arms)  # all stopped early and released
