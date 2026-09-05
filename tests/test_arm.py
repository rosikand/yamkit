from itertools import pairwise

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
    robot.pos = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.3])
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
    assert np.allclose(seen_kp, 80.0 * arm_mod.COMPLIANT_KP_SCALE)  # moved with low gains
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


@pytest.mark.parametrize('limit_speed', [True, False])
@pytest.mark.parametrize('q', [np.zeros(5), np.zeros(7), np.zeros((2, 3)), [0] * 5 + [np.nan],
                              [0] * 5 + [np.inf], ['0'] * 6, [True] * 6, [0j] * 6, None])
def test_rejected_targets_never_command_or_restore_gains(follower, q, limit_speed):
    arm, robot = follower
    arm.zero_torque()
    with pytest.raises(ValueError):
        arm.command(q, limit_speed=limit_speed)
    assert not robot.commands
    assert np.all(robot.kp == 0) and arm._gains_zeroed


@pytest.mark.parametrize('gripper', [np.nan, np.inf, -0.1, 1.1, '0.5', [0.5], True])
def test_invalid_gripper_never_changes_gains(follower, gripper):
    arm, robot = follower
    arm.zero_torque()
    with pytest.raises(ValueError):
        arm.command(np.zeros(6), gripper, limit_speed=False)
    assert not robot.commands and np.all(robot.kp == 0)


@pytest.mark.parametrize('field,value', [('joint_pos', np.zeros(7)), ('joint_pos', [np.nan] * 6),
                                        ('joint_vel', [np.inf] * 6), ('joint_eff', [[0] * 6]),
                                        ('gripper_pos', [np.nan]), ('gripper_pos', [1.01])])
def test_bad_measurements_rejected_even_without_speed_limit(follower, monkeypatch, field, value):
    arm, robot = follower
    arm.zero_torque()
    obs = robot.get_observations()
    obs[field] = value
    monkeypatch.setattr(robot, 'get_observations', lambda: obs)
    with pytest.raises(ValueError):
        arm.command(np.zeros(6), 0.5, limit_speed=False)
    assert not robot.commands and np.all(robot.kp == 0)


@pytest.mark.parametrize('previous', [np.zeros(6), np.full(7, np.nan), [0, -1, 0, 0, 0, 0, 0], [0] * 6 + [2]])
def test_previous_state_is_validated_before_gains(follower, previous):
    arm, robot = follower
    arm.zero_torque()
    arm._last_cmd = np.asarray(previous)
    with pytest.raises(ValueError):
        arm.command(np.zeros(6), 0, limit_speed=False)
    assert not robot.commands and np.all(robot.kp == 0)


def test_invalid_default_gains_rejected_before_sending(follower):
    arm, robot = follower
    arm.zero_torque()
    arm.default_kp[0] = np.nan
    with pytest.raises(ValueError):
        arm.command(np.zeros(6), 0)
    assert not robot.commands and np.all(robot.kp == 0)


@pytest.mark.parametrize('which', ['target', 'measured'])
def test_raw_bounds_reject_instead_of_sdk_clipping(follower, which):
    arm, robot = follower
    arm.zero_torque()
    q = np.zeros(6)
    if which == 'target':
        q[1] = -0.2  # below vendor joint 2 lower bound (-0.15)
    else:
        robot.pos[1] = -0.2
    with pytest.raises(ValueError, match='vendor joint bounds'):
        arm.command(q, limit_speed=False)
    assert not robot.commands and np.all(robot.kp == 0)


def test_aligned_bounds_shift_with_offsets_and_return_exact_raw_target():
    spec = ArmSpec(name='l', role='leader', gripper='yam_teaching_handle', joint_offsets=[0.1, 0, 0, 0, 0, 0])
    robot = FakeRobot(6, gripper=False, handle=True)
    arm = YamArm(spec, 'can1', robot)
    upper = arm._raw_limits[0, 1]
    q = np.zeros(6)
    q[0] = upper + 0.1
    arm.command(q, limit_speed=False)
    assert robot.commands[-1][0] == pytest.approx(upper)
    q[0] += 0.001
    before = len(robot.commands)
    with pytest.raises(ValueError, match='vendor joint bounds'):
        arm.command(q, limit_speed=False)
    assert len(robot.commands) == before


class FakeClock:
    def __init__(self):
        self.now = 10.0
        self.late = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds + self.late
        self.late = 0.0


@pytest.fixture
def clock(monkeypatch):
    from yamkit import arm as arm_mod
    clock = FakeClock()
    monkeypatch.setattr(arm_mod.time, 'monotonic', clock.monotonic)
    monkeypatch.setattr(arm_mod.time, 'sleep', clock.sleep)
    return clock


def test_no_minimum_dt_or_catchup_budget(follower, clock):
    arm, robot = follower
    first = arm.command(np.ones(6), 1)
    for _ in range(10):
        assert np.array_equal(arm.command(np.ones(6), 1), first)
    clock.now += 0.2
    second = arm.command(np.ones(6), 1)
    assert np.max(second[:6] - first[:6]) <= 0.01 + 1e-12
    robot.pos[:6] = 0.2
    clock.now += 1
    third = arm.command(np.ones(6), 1)
    assert np.allclose(third[:6], 0.21)  # stale ramp starts at measurement


def test_move_extends_short_duration_and_late_wakeup_cannot_jump(follower, clock):
    arm, robot = follower
    records = [(clock.now, robot.pos.copy())]
    orig = robot.command_joint_pos

    def record(q):
        records.append((clock.now, q.copy()))
        orig(q)
        if len(records) == 3:
            clock.late = 0.3

    robot.command_joint_pos = record
    arm.move_to(np.full(6, 0.2), 1, duration=0.001, hz=100)
    assert np.allclose(robot.pos, [0.2] * 6 + [1])
    assert clock.now - records[0][0] >= 0.5
    for (t0, q0), (t1, q1) in pairwise(records):
        assert np.max(np.abs(q1[:6] - q0[:6])) <= min(t1 - t0, 0.01) * arm.max_joint_speed + 1e-12
        assert abs(q1[-1] - q0[-1]) <= min(t1 - t0, 0.01) * arm.max_gripper_speed + 1e-12


@pytest.mark.parametrize('kwargs', [{'duration': 0}, {'duration': float('nan')}, {'hz': 0}, {'hz': -1}])
def test_bad_move_options_never_send(follower, kwargs):
    arm, robot = follower
    arm.zero_torque()
    with pytest.raises(ValueError):
        arm.move_to(np.zeros(6), **kwargs)
    assert not robot.commands and np.all(robot.kp == 0)


def test_home_checks_target_before_compliant_gains(follower):
    arm, robot = follower
    arm.spec.rest_pose = [0, -0.2, 0, 0, 0, 0]
    arm.zero_torque()
    with pytest.raises(ValueError, match='vendor joint bounds'):
        arm.go_home(compliant=True)
    assert not robot.commands and np.all(robot.kp == 0)


def test_home_preserves_measured_gripper_and_respects_speed(follower, clock):
    arm, robot = follower
    arm.command(np.zeros(6), 1, limit_speed=False)
    robot.pos = np.array([0.5] * 6 + [0.2])
    started = clock.now
    arm.go_home(speed=100)
    assert clock.now - started >= 0.5
    assert robot.pos[-1] == pytest.approx(0.2)


def test_hold_replaces_obsolete_target_before_restoring_gains(follower):
    arm, robot = follower
    arm.command(np.ones(6), 1, limit_speed=False)
    arm.zero_torque()
    robot.pos = np.array([0.2] * 6 + [0.3])
    held = robot.pos.copy()
    updates = []
    update = robot.update_kp_kd

    def check(kp, kd):
        updates.append(robot.commands[-1].copy())
        update(kp, kd)

    robot.update_kp_kd = check
    arm.hold()
    assert np.array_equal(robot.commands[-1], held)
    assert updates and all(np.array_equal(q, held) for q in updates)


def test_close_attempts_sdk_despite_idle_cancellation_and_repeats_are_safe(follower, monkeypatch):
    arm, robot = follower

    def fail():
        raise KeyboardInterrupt

    monkeypatch.setattr(robot, 'enter_gravity_comp_idle', fail)
    with pytest.raises(KeyboardInterrupt):
        arm.close(settle_s=0)
    assert robot.closed
    arm.close(settle_s=0)
    with pytest.raises(RuntimeError, match='closed'):
        arm.command(np.zeros(6))


def test_home_all_rejects_later_state_before_any_arm_moves(follower):
    from yamkit.arm import go_home_all
    arm, robot = follower
    other_robot = FakeRobot()
    other = YamArm(ArmSpec(name='other', role='follower'), 'can1', other_robot)
    other_robot.pos[1] = -1
    with pytest.raises(ValueError, match='operator recovery'):
        go_home_all([(arm, {}), (other, {})])
    assert not robot.commands and not other_robot.commands


def test_cancellation_after_worker_start_never_moves_or_leaves_active_worker(follower, monkeypatch):
    import threading

    from yamkit.arm import go_home_all

    arm, robot = follower
    threads = []
    start = threading.Thread.start

    def interrupted(thread):
        start(thread)
        threads.append(thread)
        raise KeyboardInterrupt

    monkeypatch.setattr(threading.Thread, 'start', interrupted)
    with pytest.raises(KeyboardInterrupt):
        go_home_all([(arm, {})])
    assert not robot.commands
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()
