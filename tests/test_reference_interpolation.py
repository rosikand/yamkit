"""Pure reference path/guard tests: no robot, camera, model or network objects."""

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


def action(joint=0.0, gripper=0.8):
    return {n: gripper if n in GRIPPER_NAMES else joint for n in ACTION_NAMES}


def limits(step=.03):
    return {n: {"lower": -2.0, "upper": 2.0, "max_step": step} for n in JOINT_NAMES}


def gripper_limits(step=.03):
    return dict.fromkeys(GRIPPER_NAMES, step)


def guard(initial=None, step=.03, grip_step=.03):
    return ReferenceCommandGuard(initial or action(), limits(step), gripper_limits(grip_step))


def commit(g, target, now):
    prepared = g.prepare(target, now=now)
    g.commit(prepared, prepared.shaped, deadline_monotonic_s=now + .03)
    return prepared


@pytest.mark.parametrize("delta", [0.0, .009, .019, .02, .037, .2, 1.4])
def test_literal_reference_matches_pinned_formula_and_endpoint_inclusive_linspace(delta):
    start, target = action(), action()
    target[JOINT_NAMES[0]] = delta
    count = min(int(delta / .01), 100)
    expected = (np.linspace(list(start.values()), list(target.values()), count) if count > 1
                else np.array([list(target.values())]))
    result = reference_row(start, target)
    np.testing.assert_array_equal([list(row.values()) for row in result], expected)
    assert result[-1] == target
    if count > 1:
        assert result[0] == start and len(result) == count


def test_literal_reference_uses_gripper_in_common_count_and_does_not_alias_inputs():
    start, target = action(), action()
    target["left_gripper.pos"] = .1
    target[JOINT_NAMES[0]] = .04
    result = reference_row(start, target)
    assert len(result) == min(int(.7 / .01), 100)
    target["left_gripper.pos"] = .9
    assert result[-1]["left_gripper.pos"] == .1


@pytest.mark.parametrize("joint_step,grip_step", [(.03, .03), (.004, .03), (.03, .004)])
def test_plan_keeps_one_progress_fraction_exact_endpoint_and_all_existing_bounds(joint_step, grip_step):
    start = action(.1, .8)
    target = {n: (.05 + .65 * i / 13 if n in GRIPPER_NAMES else .1 + (-1)**i * (.05 + i / 30))
              for i, n in enumerate(ACTION_NAMES)}
    plan = plan_reference_row(start, target, joint_limits=limits(joint_step),
                              gripper_max_step=gripper_limits(grip_step))
    g = guard(start, joint_step, grip_step)
    initial = np.array(list(start.values()))
    delta = np.array(list(target.values())) - initial
    previous_time = 10.0
    for index, (command, fraction) in enumerate(zip(plan.commands, plan.progress, strict=True)):
        np.testing.assert_allclose(np.array(list(command.values())), initial + delta * fraction, atol=1e-14)
        assert 0 <= fraction <= 1
        step = commit(g, command, previous_time + index * plan.period_s)
        assert step.requested == step.shaped == command  # The guard never filters a coordinate.
    assert plan.commands[0] == start
    assert plan.commands[-1] == plan.commands[-2] == target
    assert plan.progress[-2:] == (1.0, 1.0)
    assert all(b >= a for a, b in zip(plan.progress, plan.progress[1:]))
    assert plan.ticks >= plan.reference_count
    assert plan.added_ticks == plan.ticks - len(reference_row(start, target))
    assert plan.duration_s == pytest.approx(plan.ticks / 30)
    assert g.last_action == target
    np.testing.assert_array_equal(g.velocity, 0)
    assert g.metrics()["maximum_command_acceleration_rad_s2"] <= 1 + 1e-8
    assert g.metrics()["maximum_command_velocity_rad_s"] <= min(.6, joint_step / .05) + 1e-8


def test_every_original_row_is_reached_before_turning_and_pauses_only_after_zero_slope():
    g = guard()
    now = 10.0
    targets = [action(.23, .2), action(-.18, .73), action(.31, .15), action(.31, .15)]
    for target in targets:
        plan = plan_reference_row(g.last_action, target, joint_limits=limits(), gripper_max_step=gripper_limits())
        for command in plan.commands:
            now += plan.period_s
            commit(g, command, now)
        assert g.last_action == target
        assert np.max(np.abs(g.velocity)) == 0
        generation, anchor = g.generation, g.last_action
        g.begin_inference_wait(now + 1)
        for _ in range(12):
            now += 1 / 30
            commit(g, anchor, now)
        g.end_inference_wait(now)
        assert g.valid and g.generation == generation + 12 and g.last_action == anchor
        assert g.last_at == now
        assert g.metrics()["inference_waits"][-1]["hold_count"] == 12
    assert len(g.metrics()["inference_waits"]) == len(targets)


def test_identical_row_is_one_exact_hold():
    start = action(.2, .5)
    plan = plan_reference_row(start, start, joint_limits=limits(), gripper_max_step=gripper_limits())
    assert plan.commands == (start,) and plan.progress == (1.0,)
    assert plan.ticks == plan.reference_count == 1 and plan.added_ticks == 0


@pytest.mark.parametrize("key,value", [(JOINT_NAMES[0], float("nan")), (JOINT_NAMES[1], float("inf")),
                                       (JOINT_NAMES[2], True), (JOINT_NAMES[3], 2.01),
                                       (GRIPPER_NAMES[0], -.01), (GRIPPER_NAMES[1], 1.01)])
def test_invalid_original_target_is_rejected_before_interpolation_or_dispatch(key, value):
    target = action()
    target[key] = value
    with pytest.raises(ValueError):
        plan_reference_row(action(), target, joint_limits=limits(), gripper_max_step=gripper_limits())
    g = guard()
    with pytest.raises(ValueError):
        g.prepare(target, now=10)
    assert g.generation == 0 and not g.valid


def test_invalid_names_bounds_and_unbounded_plan_are_rejected():
    bad = action()
    del bad[JOINT_NAMES[0]]
    with pytest.raises(ValueError, match="14"):
        reference_row(action(), bad)
    with pytest.raises(ValueError, match="both gripper"):
        plan_reference_row(action(), action(.1), joint_limits=limits(), gripper_max_step={})
    with pytest.raises(ValueError, match="capacity"):
        plan_reference_row(action(), action(.1), joint_limits=limits(1e-12), gripper_max_step=gripper_limits())


@pytest.mark.parametrize("gap", [-.01, 0.0, .02, .1001])
def test_unexpected_early_or_stalled_dispatch_fails_without_catchup(gap):
    g = guard()
    commit(g, action(), 10)
    with pytest.raises(ReferenceInterpolationFault, match="clock"):
        g.prepare(action(), now=10 + gap)
    assert not g.valid and g.generation == 1


def test_actual_time_acceleration_and_gripper_steps_are_checked_before_dispatch():
    g = guard()
    commit(g, action(.002), 10)
    with pytest.raises(ReferenceInterpolationFault, match="acceleration"):
        g.prepare(action(.01), now=10 + 1 / 30)
    assert g.generation == 1
    g = guard()
    with pytest.raises(ReferenceInterpolationFault, match="gripper step"):
        g.prepare(action(gripper=.7), now=10)
    assert g.generation == 0


@pytest.mark.parametrize("changed", [JOINT_NAMES[0], GRIPPER_NAMES[0]])
def test_even_bounded_postclamp_changes_fault_instead_of_breaking_whole_vector_path(changed):
    g = guard()
    step = g.prepare(action(), now=10)
    actual = dict(step.shaped)
    actual[changed] += .0001
    with pytest.raises(ReferenceInterpolationFault, match="Postclamp"):
        g.commit(step, actual, deadline_monotonic_s=10.03)
    assert not g.valid and g.generation == 1
    assert g.last_action == actual
    sample = g.metrics()["samples"][0]
    assert sample["postclamp_modified"] and not sample["postclamp_bounds_exceeded"]


def test_pending_command_cannot_survive_invalidation_or_be_committed_twice():
    g = guard()
    step = g.prepare(action(), now=10)
    g.invalidate()
    with pytest.raises(ReferenceInterpolationFault):
        g.commit(step, step.shaped, deadline_monotonic_s=10.03)
    assert g.generation == 0
    g = guard()
    step = commit(g, action(), 10)
    with pytest.raises(ReferenceInterpolationFault):
        g.commit(step, step.shaped, deadline_monotonic_s=10.03)
    assert g.generation == 1


def test_inference_wait_rejects_joint_or_gripper_motion_and_expiry_cannot_revive_guard():
    for target in [action(.001), action(gripper=.79)]:
        g = guard()
        commit(g, target, 10)
        with pytest.raises(ReferenceInterpolationFault, match="stationary"):
            g.begin_inference_wait(11)
        assert not g.valid
    g = guard()
    commit(g, action(), 10)
    g.begin_inference_wait(11)
    with pytest.raises(ReferenceInterpolationFault, match="expired"):
        g.end_inference_wait(11)
    with pytest.raises(ReferenceInterpolationFault):
        g.initialize_position(action())
    assert not g.valid and g.generation == 1


def test_pause_requires_explicit_resume_and_cached_action_is_not_mutable_from_outside():
    g = guard()
    copy = g.last_action
    copy[JOINT_NAMES[0]] = 1
    assert g.last_action == action() and g.anchor_initialized
    g.begin_inference_wait(20)
    with pytest.raises(ReferenceInterpolationFault, match="paused"):
        g.prepare(action(), now=10)
    assert not g.valid


@pytest.mark.parametrize("bad_time", [float("nan"), float("inf"), 9.0])
def test_wait_clock_faults_do_not_reset_or_revive_committed_state(bad_time):
    g = guard()
    commit(g, action(), 10)
    g.begin_inference_wait(11)
    with pytest.raises(ValueError):
        g.end_inference_wait(bad_time)
    assert not g.valid and g.generation == 1 and g.last_action == action()


def test_dispatch_after_explicit_wait_must_not_precede_recorded_resume_time():
    g = guard()
    commit(g, action(), 10)
    g.begin_inference_wait(11)
    commit(g, action(), 10 + 1 / 30)
    commit(g, action(), 10 + 2 / 30)
    g.end_inference_wait(10.08)
    with pytest.raises(ReferenceInterpolationFault, match="backwards"):
        g.prepare(action(), now=10.07)
    assert not g.valid and g.generation == 3


def test_exact_inference_holds_preserve_real_command_clock_and_fixed_deadline():
    g = guard(action(.2, .6))
    anchor = g.last_action
    commit(g, anchor, 10)
    g.begin_inference_wait(11.2)
    for index in range(1, 28):
        now = 10 + index / 30
        commit(g, anchor, now)
        assert g.last_at == now and g.last_action == anchor
        assert g._wait_deadline == 11.2
        assert g.metrics()["samples"][-1]["dispatch_role"] == "inference_hold"
        np.testing.assert_array_equal(g.velocity, 0)
    generation, last_at = g.generation, g.last_at
    g.end_inference_wait(10.91)
    assert g.generation == generation and g.last_at == last_at and g.last_action == anchor
    next_step = commit(g, anchor, 10 + 28 / 30)
    assert next_step.dt_s == pytest.approx(1 / 30)
    metrics = g.metrics()
    assert metrics["inference_hold_count"] == 27
    assert metrics["samples"][-1]["dispatch_role"] == "interpolation"
    assert metrics["inference_waits"] == [{
        "last_dispatch_monotonic_s": 10, "deadline_monotonic_s": 11.2,
        "resumed_monotonic_s": 10.91, "stationary": True, "hold_count": 27,
    }]


@pytest.mark.parametrize("key", [JOINT_NAMES[0], GRIPPER_NAMES[0]])
def test_inference_hold_rejects_even_tiny_changed_target_before_commit(key):
    g = guard()
    commit(g, action(), 10)
    g.begin_inference_wait(11)
    target = g.last_action
    target[key] += 1e-12
    with pytest.raises(ReferenceInterpolationFault, match="exact committed"):
        g.prepare(target, now=10 + 1 / 30)
    assert not g.valid and g.generation == 1 and g.last_at == 10
    assert g.metrics()["inference_waits"][-1]["hold_count"] == 0


@pytest.mark.parametrize("operation", ["hold", "resume"])
def test_wait_cannot_hide_actual_command_stall(operation):
    g = guard()
    commit(g, action(), 10)
    g.begin_inference_wait(11)
    with pytest.raises(ReferenceInterpolationFault, match="clock"):
        if operation == "hold":
            g.prepare(action(), now=10.1001)
        else:
            g.end_inference_wait(10.1001)
    assert not g.valid and g.generation == 1 and g.last_at == 10
    assert g.metrics()["inference_waits"][-1]["resumed_monotonic_s"] is None


def test_hold_deadline_is_not_renewed_by_successful_maintenance_commands():
    g = guard()
    commit(g, action(), 10)
    g.begin_inference_wait(10.1)
    commit(g, action(), 10 + 1 / 30)
    commit(g, action(), 10 + 2 / 30)
    with pytest.raises(ReferenceInterpolationFault, match="deadline expired"):
        g.prepare(action(), now=10.1)
    assert not g.valid and g.generation == 3
    assert g.metrics()["inference_waits"][-1]["hold_count"] == 2
    assert g.metrics()["inference_waits"][-1]["deadline_monotonic_s"] == 10.1


@pytest.mark.parametrize("operation", ["resume", "replace", "initialize"])
def test_pending_hold_cannot_be_rebased_or_cross_a_wait_transition(operation):
    g = guard()
    commit(g, action(), 10)
    g.begin_inference_wait(11)
    step = g.prepare(action(), now=10 + 1 / 30)
    with pytest.raises(ReferenceInterpolationFault):
        if operation == "resume":
            g.end_inference_wait(10.04)
        elif operation == "replace":
            g.begin_inference_wait(12)
        else:
            g.initialize_position(action(.1))
    with pytest.raises(ReferenceInterpolationFault):
        g.commit(step, step.shaped, deadline_monotonic_s=11)
    assert not g.valid and g.generation == 1 and g.last_at == 10
    assert g.last_action == action() and g._wait_deadline == 11


def test_first_wait_without_committed_commands_retains_fresh_initial_dispatch():
    g = guard()
    g.begin_inference_wait(11)
    g.end_inference_wait(10.8)
    assert g.generation == 0 and g.last_at is None
    g.initialize_position(action(.2, .5))
    step = commit(g, g.last_action, 10.81)
    assert step.dt_s == pytest.approx(1 / 30)
    assert g.metrics()["inference_hold_count"] == 0
    assert g.metrics()["inference_waits"][-1]["hold_count"] == 0
    assert g.metrics()["samples"][-1]["dispatch_role"] == "interpolation"


@pytest.mark.parametrize("key", [JOINT_NAMES[0], GRIPPER_NAMES[0]])
def test_hold_postclamp_changes_cannot_accumulate_below_normal_tolerance(key):
    g = guard()
    commit(g, action(), 10)
    g.begin_inference_wait(11)
    step = g.prepare(g.last_action, now=10 + 1 / 30)
    sent = dict(step.shaped)
    sent[key] += 1e-12
    with pytest.raises(ReferenceInterpolationFault, match="Postclamp"):
        g.commit(step, sent, deadline_monotonic_s=11)
    assert not g.valid and g.generation == 2 and g.last_action == sent
    sample = g.metrics()["samples"][-1]
    assert sample["dispatch_role"] == "inference_hold" and sample["postclamp_modified"]
    assert g.metrics()["inference_hold_count"] == 1
