"""Long reference diagnostics remain bounded without changing dispatched points."""

import threading
from copy import deepcopy
from types import SimpleNamespace

from yamkit import reference_rollout
from yamkit.inference.command_shaping import ACTION_NAMES, JOINT_NAMES, MAX_SAMPLES, JointCommandShaper
from yamkit.inference.reference import (
    GRIPPER_NAMES,
    MAX_REFERENCE_SAMPLES,
    ReferenceCommandGuard,
    plan_reference_row,
    reference_row,
)


def action(joint=0.0, gripper=.8):
    return {name: gripper if name in GRIPPER_NAMES else joint for name in ACTION_NAMES}


def limits():
    return {name: {"lower": -2.0, "upper": 2.0, "max_step": .03} for name in JOINT_NAMES}


def test_reference_retains_2048_commands_then_evicts_without_changing_literal_points(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(reference_rollout, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    initial = action(-.8, .2)
    grips = dict.fromkeys(GRIPPER_NAMES, .03)
    guard = ReferenceCommandGuard(initial, limits(), grips)
    policy = SimpleNamespace(transport=SimpleNamespace())
    engine = reference_rollout.ReferenceRemoteInferenceEngine(
        policy=policy, preprocessor=None, postprocessor=None,
        robot_wrapper=SimpleNamespace(command_shaper=guard), task="put the red cube into the black container", fps=30,
        shutdown_event=threading.Event(), duration=120, gripper_max_step=grips)
    # Thirty already-admitted original rows exercise the real dequeue/commit path
    # without importing a hardware adapter, observing cameras, or invoking a model.
    targets = [action(.8 if i % 2 == 0 else -.8, .8 if i % 2 == 0 else .2) for i in range(30)]
    original_targets = deepcopy(targets)
    plans, expected, previous = [], [], initial
    for target in targets:
        plans.append(plan_reference_row(previous, target, joint_limits=limits(), gripper_max_step=grips))
        expected.extend(reference_row(previous, target))
        previous = target
    assert len(expected) == 3000
    engine._plans = plans
    engine.predictions.append({"chunk_index": 0})
    engine.predicted_steps = engine.admitted_steps = len(targets)
    engine.phase_deadline = engine._plan_deadline = clock[0] + 120
    assert MAX_REFERENCE_SAMPLES == 2048
    for index, original in enumerate(expected):
        clock[0] += 1 / 30
        returned = engine.get_action({})
        requested = dict(zip(ACTION_NAMES, returned.tolist(), strict=True))
        assert requested == original
        step = guard.prepare(requested, now=clock[0])
        guard.commit(step, requested, deadline_monotonic_s=engine.action_deadline)
        engine.record_execution()
        engine.record_commit()
        assert guard.last_action == original
        if index + 1 in (1001, 1803, 2048, 2049, 3000):
            count = index + 1
            retained = min(count, MAX_REFERENCE_SAMPLES)
            commands, dispatch = guard.metrics(), engine.metrics()
            assert commands["sample_count"] == dispatch["interpolation_dispatches"] == count
            assert len(commands["samples"]) == len(dispatch["dispatch_samples"]) == retained
            assert commands["samples_dropped"] == dispatch["dispatch_samples_dropped"] == count - retained
            assert commands["samples"][0]["dispatch_index"] == count - retained
            assert dispatch["dispatch_samples"][0]["dispatch_index"] == count - retained
            for command, point in zip(commands["samples"], dispatch["dispatch_samples"], strict=True):
                original = expected[command["dispatch_index"]]
                assert command["requested"] == command["shaped"] == command["sent"] == original
                assert point["dispatch_index"] == command["dispatch_index"]
                assert point["target"] == targets[point["row_index"]]
                assert command["deadline_monotonic_s"] == point["deadline_monotonic_s"] == engine.phase_deadline
    assert targets == original_targets
    assert engine.completed_steps == 30 and engine.completed_chunks == 1
    assert guard.valid and guard.postclamp_modified_count == engine.expired_before_dispatch == 0


def test_async_command_retention_remains_1000():
    assert MAX_SAMPLES == 1000
    target = action()
    guard = JointCommandShaper(target, limits())
    for index in range(1002):
        step = guard.prepare(target, now=10 + index / 30)
        guard.commit(step, target, deadline_monotonic_s=100)
    metrics = guard.metrics()
    assert metrics["sample_count"] == 1002
    assert len(metrics["samples"]) == 1000
    assert metrics["samples_dropped"] == 2
    assert metrics["samples"][0]["dispatch_index"] == 2
