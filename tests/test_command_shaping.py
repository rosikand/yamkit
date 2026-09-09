"""Remote-only shaping against fake commands; never activate real hardware."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from yamkit.inference.command_shaping import (
    ACTION_NAMES,
    JOINT_NAMES,
    CommandShapingFault,
    JointCommandShaper,
)


def action(joint=0.0, gripper=0.8):
    return {name: gripper if "gripper" in name else joint for name in ACTION_NAMES}


def limits(lower=-2.0, upper=2.0, max_step=0.03):
    return {name: {"lower": lower, "upper": upper, "max_step": max_step} for name in JOINT_NAMES}


def advance(shaper, target, now):
    step = shaper.prepare(target, now=now)
    shaper.commit(step, step.shaped, deadline_monotonic_s=now + 0.03)
    return step


def test_first_command_is_bounded_from_initial_pose_and_grippers_pass_through():
    shaper = JointCommandShaper(action(0.2), limits())
    step = advance(shaper, action(1.0, gripper=0.1), 50.0)
    assert step.shaped[JOINT_NAMES[0]] == pytest.approx(0.2 + 2 / 30**2)
    assert step.shaped["left_gripper.pos"] == 0.1
    assert step.shaped["right_gripper.pos"] == 0.1
    assert shaper.metrics()["samples"][0]["requested"] == action(1.0, gripper=0.1)
    assert shaper.metrics()["maximum_command_acceleration_rad_s2"] == pytest.approx(2.0)


def test_reversal_decelerates_through_zero_with_unchanged_speed_step_and_acceleration_bounds():
    shaper = JointCommandShaper(action(), limits())
    positions, velocities = [], []
    for index in range(100):
        step = advance(shaper, action(1.0 if index < 35 else -1.0), 10 + index / 30)
        positions.append(step.shaped[JOINT_NAMES[0]])
        velocities.append(shaper.velocity[0])
    assert velocities[35] > 0  # Direction cannot snap when the new target reverses.
    assert any(value < 0 for value in velocities[36:])
    assert np.max(np.abs(velocities)) <= 0.6 + 1e-8
    assert np.max(np.abs(np.diff(velocities))) * 30 <= 2 + 1e-8
    assert np.max(np.abs(np.diff(positions))) <= 0.03
    assert shaper.metrics()["modified_count"] > 0


def test_joint_boundary_braking_stays_inside_original_bounds():
    shaper = JointCommandShaper(action(), limits(lower=-0.1, upper=0.1))
    for index in range(200):
        advance(shaper, action(0.1), 10 + index / 30)
        assert np.all(shaper.position <= 0.1)
        assert np.all(shaper.position >= -0.1)
        assert np.all(shaper.velocity >= -1e-8)
    assert np.all(shaper.position > 0.095)


@pytest.mark.parametrize("initial,target", [(0.0, 0.05), (0.0, 0.2), (0.0, -0.05),
                                            (0.0, -0.2), (0.8, 0.85), (-0.8, -0.6),
                                            (0.0, 1.2), (0.0, -1.2)])
def test_stationary_target_from_rest_converges_without_overshoot_or_reversal(initial, target):
    shaper = JointCommandShaper(action(initial), limits())
    direction = np.sign(target - initial)
    for index in range(180):
        previous = shaper.position.copy()
        advance(shaper, action(target, gripper=0.1), 10 + index / 30)
        assert np.all(direction * (shaper.position - previous) >= -1e-12)
        assert np.all(direction * (target - shaper.position) >= -1e-12)
        assert np.max(np.abs(shaper.position - previous)) <= 0.03 + 1e-8
    np.testing.assert_allclose(shaper.position, target, atol=1e-6)
    assert np.max(np.abs(shaper.velocity)) < 1e-6
    assert shaper.metrics()["maximum_command_velocity_rad_s"] <= 0.6 + 1e-8
    assert shaper.metrics()["maximum_command_acceleration_rad_s2"] <= 2.0 + 1e-8
    assert all(row["shaped"]["left_gripper.pos"] == 0.1 for row in shaper.metrics()["samples"])


@pytest.mark.parametrize("target", [0.05, -0.05, 0.2, -0.2])
@pytest.mark.parametrize("intervals", [(0.02, 0.035, 0.049, 0.08, 0.099),
                                     (0.099, 0.007, 0.013, 0.097, 0.025)])
def test_stationary_target_braking_tolerates_variable_dispatch_intervals(target, intervals):
    shaper = JointCommandShaper(action(), limits())
    now = 10.0
    for index in range(160):
        now += intervals[index % len(intervals)]
        previous, velocity = shaper.position.copy(), shaper.velocity.copy()
        step = advance(shaper, action(target), now)
        assert np.all(np.sign(target) * (shaper.position - previous) >= -1e-12)
        assert np.all(np.sign(target) * (target - shaper.position) >= -1e-12)
        assert np.max(np.abs(shaper.position - previous)) <= 0.03 + 1e-8
        assert np.max(np.abs(shaper.velocity - velocity)) <= 2 * min(step.dt_s, 0.05) + 1e-8
    np.testing.assert_allclose(shaper.position, target, atol=1e-6)
    assert np.max(np.abs(shaper.velocity)) < 1e-6


@pytest.mark.parametrize("remaining_distance", [0.0, 0.001])
def test_suddenly_nearer_target_inside_stopping_distance_keeps_acceleration_protection(remaining_distance):
    shaper = JointCommandShaper(action(), limits())
    for index in range(20):
        advance(shaper, action(1.5), 10 + index / 30)
    previous, velocity = shaper.position.copy(), shaper.velocity.copy()
    target = shaper.position[0] + remaining_distance
    step = advance(shaper, action(target), shaper.last_at + 1 / 30)
    assert np.all(shaper.position > target)  # Stopping in 1 mrad would violate acceleration.
    assert np.all(shaper.position > previous)
    np.testing.assert_allclose(shaper.velocity, velocity - 2 * step.dt_s, atol=1e-8)
    assert shaper.valid
    for _ in range(160):
        advance(shaper, action(target), shaper.last_at + 1 / 30)
    np.testing.assert_allclose(shaper.position, target, atol=1e-6)
    assert shaper.metrics()["maximum_command_acceleration_rad_s2"] <= 2 + 1e-8


def test_zero_and_tiny_target_errors_stay_finite_and_do_not_generate_jitter():
    shaper = JointCommandShaper(action(0.3), limits())
    target = action(0.3)
    target[JOINT_NAMES[1]] += 1e-9
    target[JOINT_NAMES[2]] -= 1e-9
    expected = np.array([target[name] for name in JOINT_NAMES])
    direction = np.sign(expected - shaper.position)
    for index in range(120):
        previous = shaper.position.copy()
        advance(shaper, target, 10 + index / 30)
        assert np.all(np.isfinite(shaper.velocity))
        assert np.all(direction * (shaper.position - previous) >= 0)
        assert np.all(direction * (expected - shaper.position) >= 0)
        assert shaper.position[0] == 0.3 and shaper.velocity[0] == 0
    np.testing.assert_allclose(shaper.position, expected, atol=1e-14, rtol=0)
    assert np.max(np.abs(shaper.velocity)) < 1e-12


def test_infeasible_joint_boundary_braking_fails_before_candidate_is_returned():
    shaper = JointCommandShaper(action(0.099), limits(lower=-0.1, upper=0.1))
    shaper.velocity.fill(0.6)  # An inconsistent incoming controller state must not be clamped away.
    with pytest.raises(CommandShapingFault, match="bounded braking"):
        shaper.prepare(action(-0.1), now=10)
    assert shaper.generation == 0


@pytest.mark.parametrize("dt", [0, -0.001, 0.101, float("inf"), float("nan")])
def test_bad_or_stalled_clock_never_accumulates_catchup(dt):
    shaper = JointCommandShaper(action(), limits())
    advance(shaper, action(0.4), 10.0)
    previous = shaper.position.copy()
    with pytest.raises(ValueError):
        shaper.prepare(action(0.4), now=10.0 + dt)
    np.testing.assert_array_equal(shaper.position, previous)
    assert shaper.generation == 1


def test_long_gap_at_speed_fails_instead_of_violating_existing_arm_step_cap():
    shaper = JointCommandShaper(action(), limits())
    for index in range(15):
        advance(shaper, action(1.0), 10 + index / 30)
    with pytest.raises(CommandShapingFault, match="step-budget"):
        shaper.prepare(action(1.0), now=shaper.last_at + 0.08)


def test_lower_rig_speed_and_irregular_intervals_keep_existing_clamp_and_acceleration():
    shaper = JointCommandShaper(action(), limits(max_step=0.005))
    now = 10.0
    for index in range(80):
        now += (0.02, 0.035, 0.049, 0.08)[index % 4]
        previous = shaper.position.copy()
        advance(shaper, action(0.5), now)
        assert np.max(np.abs(shaper.position - previous)) <= 0.005 + 1e-8
        assert np.max(np.abs(shaper.velocity)) <= 0.1 + 1e-8
    assert shaper.metrics()["maximum_command_acceleration_rad_s2"] <= 2 + 1e-8


@pytest.mark.parametrize("name, value", [(JOINT_NAMES[0], float("nan")), (JOINT_NAMES[4], 5.0),
                                         ("right_gripper.pos", -0.01), (JOINT_NAMES[3], True)])
def test_invalid_original_targets_cannot_be_hidden_by_smoothing(name, value):
    shaper = JointCommandShaper(action(), limits())
    target = action()
    target[name] = value
    with pytest.raises(ValueError):
        shaper.prepare(target, now=10)
    assert shaper.generation == 0


def test_exact_action_shape_required():
    shaper = JointCommandShaper(action(), limits())
    target = action()
    target.pop("right_gripper.pos")
    with pytest.raises(CommandShapingFault, match="exactly 14"):
        shaper.prepare(target, now=10)


def test_successful_postclamp_feedback_is_the_next_state_and_recorded_separately():
    shaper = JointCommandShaper(action(), limits())
    step = shaper.prepare(action(0.5), now=10)
    actual = dict(step.shaped)
    actual.update({name: 0.001 for name in JOINT_NAMES})
    shaper.commit(step, actual, deadline_monotonic_s=10.03)
    np.testing.assert_allclose(shaper.position, 0.001)
    np.testing.assert_allclose(shaper.velocity, 0.03)
    sample = shaper.metrics()["samples"][0]
    assert sample["sent"] == actual and sample["shaped"] != actual
    assert sample["postclamp_modified"]
    next_step = shaper.prepare(action(0.5), now=10 + 1 / 30)
    assert next_step.shaped[JOINT_NAMES[0]] == pytest.approx(0.001 + (0.03 + 2 / 30) / 30)


def test_postclamp_violation_is_recorded_and_permanently_invalidates_shaper():
    shaper = JointCommandShaper(action(), limits())
    step = shaper.prepare(action(0.5), now=10)
    with pytest.raises(CommandShapingFault, match="Postclamp"):
        shaper.commit(step, action(0.02), deadline_monotonic_s=10.03)
    assert not shaper.valid and shaper.generation == 1
    assert shaper.metrics()["samples"][0]["postclamp_bounds_exceeded"]
    with pytest.raises(CommandShapingFault, match="invalidated"):
        shaper.prepare(action(), now=10.04)


class FakeBoundary:
    def __init__(self):
        self.sent = []
        self.validated = []
        self.validation_hook = None
        self.snapshot_hook = None
        self.snapshot_count = 0

    def get_joint_state(self):
        self.snapshot_count += 1
        if self.snapshot_hook:
            self.snapshot_hook()
        return action()

    def validate_action_target(self, target):
        self.validated.append(dict(target))
        if self.validation_hook:
            self.validation_hook()
        if any(abs(value) > 2 for name, value in target.items() if "gripper" not in name):
            raise ValueError("original joint bounds")

    def send_action(self, target):
        self.sent.append(dict(target))
        return dict(target)


def wrapper(monkeypatch):
    from yamkit import remote_rollout

    clock = [10.0]
    monkeypatch.setattr(remote_rollout, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    robot = FakeBoundary()
    stop = threading.Event()
    shaper = JointCommandShaper(action(), limits())
    wrapped = remote_rollout._StoppableRobot(robot, stop, command_shaper=shaper)
    wrapped.action_deadline = lambda: 10.03
    return wrapped, robot, stop, shaper, clock


@pytest.mark.parametrize("race", ["stop", "deadline", "session", "invalidate"])
def test_stop_and_expiry_rechecked_after_preparation_before_any_send(monkeypatch, race):
    wrapped, robot, stop, shaper, clock = wrapper(monkeypatch)

    def race_hook():
        if race == "stop":
            stop.set()
        elif race == "deadline":
            clock[0] = 10.04
        elif race == "invalidate":
            shaper.invalidate()
        else:
            wrapped.session_check = lambda: (_ for _ in ()).throw(RuntimeError("session expired"))

    # Race at the shaped-target validation, after prepare has succeeded.
    robot.validation_hook = lambda: race_hook() if len(robot.validated) == 2 else None
    with pytest.raises(RuntimeError):
        wrapped.send_action(action(0.5))
    assert robot.sent == [] and not shaper.valid and stop.is_set()


def test_wrapper_original_robot_bounds_checked_before_shaper_and_both_validated(monkeypatch):
    wrapped, robot, stop, shaper, _clock = wrapper(monkeypatch)
    with pytest.raises(ValueError, match="original joint bounds"):
        wrapped.send_action(action(3.0))
    assert robot.validated == [action(3.0)] and robot.sent == []
    assert stop.is_set() and not shaper.valid


def test_reset_allows_initial_setup_but_never_restarts_a_used_or_invalidated_filter(monkeypatch):
    wrapped, robot, _stop, shaper, _clock = wrapper(monkeypatch)
    wrapped.reset_shaping()
    wrapped.send_action(action(0.5))
    assert len(robot.sent) == 1 and shaper.valid
    wrapped.reset_shaping()
    assert not shaper.valid
    wrapped.reset_shaping()
    with pytest.raises(Exception, match="invalidated"):
        wrapped.send_action(action(0.5))
    assert len(robot.sent) == 1


@pytest.mark.parametrize("race", ["stop", "deadline", "session", "invalidate"])
def test_first_snapshot_cannot_bypass_stop_expiry_or_invalidation(monkeypatch, race):
    wrapped, robot, stop, shaper, clock = wrapper(monkeypatch)

    def snapshot_hook():
        if race == "stop":
            stop.set()
        elif race == "deadline":
            clock[0] = 10.04
        elif race == "invalidate":
            shaper.invalidate()
        else:
            wrapped.session_check = lambda: (_ for _ in ()).throw(RuntimeError("session expired"))

    robot.snapshot_hook = snapshot_hook
    with pytest.raises((RuntimeError, CommandShapingFault)):
        wrapped.send_action(action(0.5))
    assert not robot.sent and not shaper.valid and stop.is_set()
    wrapped.reset_shaping()
    with pytest.raises(RuntimeError):
        wrapped.send_action(action(0.5))
    assert robot.snapshot_count == 1 and not robot.sent


@pytest.mark.parametrize("field,value", [("right_joint_1.pos", float("nan")),
                                         ("right_joint_1.pos", 3.0), ("right_gripper.pos", -0.01)])
def test_invalid_first_measured_snapshot_rejects_both_arms(monkeypatch, field, value):
    wrapped, robot, stop, shaper, _clock = wrapper(monkeypatch)
    position = action()
    position[field] = value
    robot.get_joint_state = lambda: position
    with pytest.raises(ValueError):
        wrapped.send_action(action(0.5))
    assert not robot.sent and not shaper.valid and stop.is_set()


def test_first_snapshot_does_not_earn_extra_command_time(monkeypatch):
    wrapped, robot, _stop, shaper, clock = wrapper(monkeypatch)
    robot.snapshot_hook = lambda: clock.__setitem__(0, 10.02)
    result = wrapped.send_action(action(0.5))
    assert result[JOINT_NAMES[0]] == pytest.approx(2 / 30**2)
    assert shaper.metrics()["samples"][0]["dt_s"] == pytest.approx(1 / 30)
    with pytest.raises(CommandShapingFault, match="reinitialize"):
        shaper.initialize_position(action(1.0))


def test_plugin_original_target_validator_and_limits_never_read_or_command_hardware(rig, fake_connect):
    from lerobot_robot_yamkit import BiYamFollowerConfig
    from lerobot_robot_yamkit.yam_follower import BiYamFollower

    rig.control.home_speed = 0
    rig.save()
    robot = BiYamFollower(BiYamFollowerConfig(rig=str(rig.path), cameras={}))
    robot.connect()
    try:
        def forbidden():
            raise AssertionError("Pure target validation read hardware")

        for fake in fake_connect.values():
            fake.get_observations = forbidden
        configured = robot.joint_command_limits()
        assert set(configured) == set(JOINT_NAMES)
        robot.validate_action_target(action(0.2))
        bad = action(0.2)
        bad["right_joint_2.pos"] = configured["right_joint_2.pos"]["upper"] + 0.01
        with pytest.raises(ValueError, match="bounds"):
            robot.validate_action_target(bad)
        assert all(not fake.commands for fake in fake_connect.values())
    finally:
        robot.disconnect_no_home()


@pytest.mark.parametrize("previous_command_age", [0.1, 1.24])
def test_first_dispatch_uses_settled_pose_without_reseeding_later_commands(
        monkeypatch, rig, fake_connect, previous_command_age):
    """Reproduce physical startup drift while the first inference chunk is discarded."""
    from lerobot_robot_yamkit import BiYamFollowerConfig
    from lerobot_robot_yamkit.yam_follower import BiYamFollower

    from yamkit import arm, remote_rollout

    clock = [10.0]
    for module in (arm, remote_rollout):
        monkeypatch.setattr(module, "time", SimpleNamespace(
            monotonic=lambda: clock[0], time=lambda: clock[0], sleep=lambda _: None))
    rig.control.home_speed = 0
    rig.save()
    robot = BiYamFollower(BiYamFollowerConfig(rig=str(rig.path), cameras={}))
    robot.connect()
    snapshots = []

    original_snapshot = robot.get_joint_state

    def snapshot():
        snapshots.append(clock[0])
        return original_snapshot()

    monkeypatch.setattr(robot, "get_joint_state", snapshot)
    monkeypatch.setattr(robot, "_camera_observation", lambda *_: pytest.fail("Startup snapshot acquired cameras"))
    try:
        initial = action(gripper=0.8)
        initial["left_joint_5.pos"] = -0.05092698558022413
        for handle in robot._sides.values():
            handle.arm._robot.pos[-1] = 0.8
            handle.arm._last_cmd = handle.arm._robot.pos.copy()
            handle.arm._last_cmd_t = clock[0] - previous_command_age
        left = fake_connect["left_follower"]
        left.pos[4] = -0.014305333028152845  # Settled while waiting for admission.
        shaper = JointCommandShaper(initial, robot.joint_command_limits())
        wrapped = remote_rollout._StoppableRobot(robot, threading.Event(), command_shaper=shaper)
        wrapped.action_deadline = lambda: clock[0] + 0.03
        requested = action(gripper=0.8)
        requested["left_joint_5.pos"] = -0.0025453269481658936
        measured = left.pos[4]
        first = wrapped.send_action(requested)
        assert first["left_joint_5.pos"] == pytest.approx(measured + 2 / 30**2)
        assert shaper.metrics()["postclamp_modified_count"] == 0
        assert shaper.metrics()["maximum_command_acceleration_rad_s2"] <= 2.0 + 1e-8
        assert snapshots == [10.0]

        # Measured tracking drift must not reset the ongoing command trajectory.
        left.pos[4] = 0.7
        clock[0] += 1 / 30
        second = wrapped.send_action(requested)
        assert snapshots == [10.0]
        assert abs(second["left_joint_5.pos"] - first["left_joint_5.pos"]) <= 4 / 30**2 + 1e-8
        assert shaper.generation == 2 and shaper.valid
    finally:
        robot.disconnect_no_home()
