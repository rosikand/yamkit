"""Pure literal reference points and validation lifecycle; no hardware or model."""

from dataclasses import replace

import numpy as np
import pytest

from yamkit.inference.reference import (
    ACTION_NAMES,
    GRIPPER_NAMES,
    JOINT_NAMES,
    ReferenceCommandGuard,
    ReferenceInterpolationFault,
    plan_reference_row,
    reference_row,
)


def action(joint=0.0, gripper=.8):
    return {name: gripper if name in GRIPPER_NAMES else joint for name in ACTION_NAMES}


def limits(step=.03):
    return {name: {"lower": -2.0, "upper": 2.0, "max_step": step} for name in JOINT_NAMES}


def grips(step=.03):
    return dict.fromkeys(GRIPPER_NAMES, step)


def guard(initial=None):
    return ReferenceCommandGuard(initial or action(), limits(), grips())


def commit(value, target, now):
    step = value.prepare(target, now=now)
    value.commit(step, step.shaped, deadline_monotonic_s=now + .5)
    return step


@pytest.mark.parametrize("delta", [0, .009, .019, .02, .029999, .037, .2, 1.4])
def test_plan_equals_literal_pinned_linspace_without_added_holds(delta):
    start, target = action(), action()
    target[JOINT_NAMES[0]] = delta
    count = min(int(delta / .01), 100)
    expected = (np.linspace(list(start.values()), list(target.values()), count) if count > 1
                else np.array([list(target.values())]))
    plan = plan_reference_row(start, target, joint_limits=limits(), gripper_max_step=grips())
    np.testing.assert_array_equal([list(row.values()) for row in plan.commands], expected)
    assert plan.commands == reference_row(start, target)
    assert plan.ticks == plan.reference_count == max(1, count)
    assert plan.added_ticks == 0 and plan.commands[-1] == target
    assert plan.duration_s == pytest.approx(max(1, count) / 30)
    if count > 1:
        assert plan.commands[0] == start and plan.commands[-2] != target
        assert plan.progress == tuple(np.linspace(0, 1, count))
    else:
        assert plan.progress == (1.0,)


@pytest.mark.parametrize("configured_step", [.03, .001, 1e-12])
def test_all_14_coordinates_share_literal_progress_independent_of_speed_settings(configured_step):
    start, target = action(.1, .8), action(-.23, .15)
    target["left_joint_4.pos"] = .31
    target["right_gripper.pos"] = .45
    plan = plan_reference_row(start, target, joint_limits=limits(configured_step),
                              gripper_max_step=grips(configured_step))
    original = np.array(list(start.values()))
    delta = np.array(list(target.values())) - original
    assert plan.commands == reference_row(start, target)
    assert plan.added_ticks == 0
    for point, fraction in zip(plan.commands, plan.progress, strict=True):
        np.testing.assert_allclose(list(point.values()), original + fraction * delta, atol=1e-14)
    target["left_gripper.pos"] = .9
    assert plan.commands[-1]["left_gripper.pos"] == .15


def test_every_row_endpoint_is_preserved_without_zero_velocity_stop():
    value = guard()
    now = 10.0
    for target in [action(.2, .3), action(-.15, .6), action(.4, .1)]:
        plan = plan_reference_row(value.last_action, target, joint_limits=limits(), gripper_max_step=grips())
        for command in plan.commands:
            now += .001  # Reference overrun spacing is not changed by this guard.
            step = commit(value, command, now)
            assert step.requested == step.shaped == command
        assert value.last_action == target
        assert np.any(value.velocity != 0)  # No synthetic repeated endpoint/zero slope.
        last_at, velocity, generation = value.last_at, value.velocity.copy(), value.generation
        value.begin_inference_wait(now + 2)
        now += .88
        value.end_inference_wait(now)
        assert value.last_at == last_at and value.generation == generation
        np.testing.assert_array_equal(value.velocity, velocity)
    settings = value.metrics()["settings"]
    assert settings["custom_joint_shaping"] is settings["yamarm_speed_clamp"] is False
    assert settings["max_joint_velocity_rad_s"] is None
    assert settings["max_joint_acceleration_rad_s2"] is None


@pytest.mark.parametrize("gap", [.0002, .001, .02, .1001, .88, 4.0])
def test_actual_positive_intervals_are_not_retimed_or_speed_limited(gap):
    value = guard()
    first = commit(value, action(), 10)
    assert first.dt_s is None
    target = action(.8, .1)
    step = commit(value, target, 10 + gap)
    assert step.dt_s == pytest.approx(gap)
    assert step.requested == step.shaped == value.last_action == target
    assert value.valid and value.postclamp_modified_count == 0
    assert value.metrics()["maximum_command_velocity_rad_s"] == pytest.approx(.8 / gap)


def test_actual_command_slopes_are_metrics_not_acceleration_limits():
    value = guard()
    commit(value, action(), 10)
    commit(value, action(.2, .1), 10.001)
    commit(value, action(-.2, .9), 10.002)
    assert value.valid
    assert value.metrics()["maximum_command_velocity_rad_s"] == pytest.approx(400)
    assert value.metrics()["maximum_command_acceleration_rad_s2"] == pytest.approx(600000)


@pytest.mark.parametrize("key,bad", [(JOINT_NAMES[0], float("nan")), (JOINT_NAMES[1], float("inf")),
                                     (JOINT_NAMES[2], True), (JOINT_NAMES[3], 2.01),
                                     (GRIPPER_NAMES[0], -.01), (GRIPPER_NAMES[1], 1.01)])
def test_invalid_original_target_is_rejected_before_any_point_can_mask_it(key, bad):
    target = action()
    target[key] = bad
    with pytest.raises(ValueError):
        plan_reference_row(action(), target, joint_limits=limits(), gripper_max_step=grips())
    value = guard()
    with pytest.raises(ValueError):
        value.prepare(target, now=10)
    assert not value.valid and value.generation == 0


@pytest.mark.parametrize("bad_time", [float("nan"), float("inf"), True, 10.0, 9.0])
def test_dispatch_clock_must_remain_finite_and_strictly_forward(bad_time):
    value = guard()
    commit(value, action(), 10)
    with pytest.raises(ValueError):
        value.prepare(action(), now=bad_time)
    assert not value.valid and value.generation == 1


@pytest.mark.parametrize("key", [JOINT_NAMES[0], GRIPPER_NAMES[0]])
def test_changed_postclamp_feedback_is_recorded_then_faults(key):
    value = guard()
    step = value.prepare(action(), now=10)
    actual = dict(step.shaped)
    actual[key] += .0001
    with pytest.raises(ReferenceInterpolationFault, match="Postclamp"):
        value.commit(step, actual, deadline_monotonic_s=10.5)
    assert value.last_action == actual and value.generation == 1 and not value.valid
    assert value.metrics()["samples"][0]["postclamp_modified"]


@pytest.mark.parametrize("bad", [float("nan"), 3.0])
def test_invalid_returned_joint_state_is_not_accepted(bad):
    value = guard()
    step = value.prepare(action(), now=10)
    sent = dict(step.shaped)
    sent[JOINT_NAMES[0]] = bad
    with pytest.raises(ValueError):
        value.commit(step, sent, deadline_monotonic_s=10.5)
    assert not value.valid and value.generation == 0


@pytest.mark.parametrize("operation", ["invalidate", "copy", "mutate", "double"])
def test_pending_command_identity_cannot_be_changed_or_reused(operation):
    value = guard()
    step = value.prepare(action(), now=10)
    if operation == "invalidate":
        value.invalidate()
    elif operation == "copy":
        step = replace(step)
    elif operation == "mutate":
        step.shaped[JOINT_NAMES[0]] += .1
    else:
        value.commit(step, step.shaped, deadline_monotonic_s=10.5)
    with pytest.raises(ReferenceInterpolationFault):
        value.commit(step, step.shaped, deadline_monotonic_s=10.5)
    assert not value.valid


def test_wait_keeps_actual_dispatch_clock_and_never_rebases_cached_state():
    value = guard()
    commit(value, action(), 10)
    commit(value, action(.1, .7), 10.001)
    velocity, anchor = value.velocity.copy(), value.last_action
    value.begin_inference_wait(12)
    value.end_inference_wait(10.88)
    assert value.last_at == 10.001 and value.last_action == anchor and value.generation == 2
    np.testing.assert_array_equal(value.velocity, velocity)
    step = commit(value, action(.2, .6), 10.881)
    assert step.dt_s == pytest.approx(.88)  # No synthetic 1/30 restart or accumulated catch-up commands.
    assert value.metrics()["inference_waits"][-1]["command_dispatches_during_wait"] == 0


def test_wait_forbids_maintenance_commands_and_cannot_extend_its_deadline():
    value = guard()
    commit(value, action(), 10)
    value.begin_inference_wait(11)
    with pytest.raises(ReferenceInterpolationFault, match="paused"):
        value.prepare(action(), now=10.5)
    assert not value.valid and value.generation == 1
    value = guard()
    commit(value, action(), 10)
    value.begin_inference_wait(11)
    with pytest.raises(ReferenceInterpolationFault):
        value.begin_inference_wait(12)
    assert value._wait_deadline == 11 and not value.valid


@pytest.mark.parametrize("bad_time", [float("nan"), float("inf"), 9.0, 11.0, 12.0])
def test_wait_expiry_or_bad_time_never_revives_guard(bad_time):
    value = guard()
    commit(value, action(), 10)
    value.begin_inference_wait(11)
    with pytest.raises(ValueError):
        value.end_inference_wait(bad_time)
    with pytest.raises(ReferenceInterpolationFault):
        value.initialize_position(action(.2))
    assert not value.valid and value.generation == 1


def test_pending_command_cannot_start_a_wait_or_be_reinitialized():
    for operation in [lambda value: value.begin_inference_wait(11),
                      lambda value: value.initialize_position(action(.1))]:
        value = guard()
        value.prepare(action(), now=10)
        with pytest.raises(ReferenceInterpolationFault):
            operation(value)
        assert not value.valid


def test_first_wait_allows_one_fresh_seed_and_does_not_invent_first_interval():
    value = guard()
    value.begin_inference_wait(11)
    value.end_inference_wait(10.8)
    value.initialize_position(action(.2, .5))
    step = commit(value, value.last_action, 10.81)
    assert step.dt_s is None
    assert value.metrics()["initial_action"] == action(.2, .5)
    assert value.metrics()["samples"][0]["dt_s"] is None
    with pytest.raises(ReferenceInterpolationFault):
        value.initialize_position(action())


def test_mutating_input_or_cached_action_copy_cannot_change_pending_target():
    value = guard()
    target = action(.2)
    step = value.prepare(target, now=10)
    target[JOINT_NAMES[0]] = .7
    value.commit(step, step.shaped, deadline_monotonic_s=10.5)
    copied = value.last_action
    copied[JOINT_NAMES[0]] = .8
    assert value.last_action == action(.2)


def test_plan_requires_exact_names_valid_bounds_and_period():
    bad = action()
    del bad[JOINT_NAMES[0]]
    with pytest.raises(ValueError, match="14"):
        reference_row(action(), bad)
    invalid_limits = limits()
    invalid_limits[JOINT_NAMES[0]]["lower"] = 3
    with pytest.raises(ValueError, match="bounds"):
        plan_reference_row(action(), action(), joint_limits=invalid_limits, gripper_max_step=grips())
    with pytest.raises(ValueError):
        plan_reference_row(action(), action(), joint_limits=limits(), gripper_max_step=grips(), period_s=0)
