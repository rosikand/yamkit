"""Startup admission uses original action deadlines and never re-arms an underrun."""

import threading
from types import SimpleNamespace

import pytest
import torch

from yamkit.inference.client import RemoteFault
from yamkit.remote_rollout import InvalidatableActionQueue, UnguidedRemoteInferenceEngine, _ObservationSlot


@pytest.fixture
def clock(monkeypatch):
    value = [100.0]
    monkeypatch.setattr("yamkit.remote_rollout.time", SimpleNamespace(monotonic=lambda: value[0]))
    return value


def _actions(offset=0, count=30):
    return torch.arange(offset, offset + count).float().unsqueeze(1).repeat(1, 14)


def _queue(clock, *, max_age_s=2.0, startup_horizon_s=0.5):
    observation = [clock[0]]
    queue = InvalidatableActionQueue(
        max_steps=45, max_age_s=max_age_s, fps=30,
        observation_time=lambda: observation[0], startup_horizon_s=startup_horizon_s,
    )
    return queue, observation


def _engine(clock, *, max_age_s=2.0, timeout_s=10.0):
    stop = threading.Event()
    session = SimpleNamespace(samples=[], _closed=False)
    policy = SimpleNamespace(
        config=SimpleNamespace(prediction_queue_threshold=None, max_observation_age_s=max_age_s,
                               request_timeout_s=timeout_s),
        profile=SimpleNamespace(chunk_size=30), session=session,
        transport=SimpleNamespace(ensure_session_active=lambda: None),
        _observation_time=clock[0],
    )

    def close():
        session._closed = True

    policy.close = close
    engine = UnguidedRemoteInferenceEngine(
        policy=policy, preprocessor=SimpleNamespace(steps=[]), postprocessor=SimpleNamespace(),
        robot_wrapper=SimpleNamespace(), hw_features={}, task="saved fake observation", fps=30,
        shutdown_event=stop,
    )
    engine._action_queue = engine._new_queue()
    engine._started_at = clock[0]
    return engine, policy, stop


def test_recorded_slow_first_chunk_is_discarded_then_fresh_warm_chunk_keeps_deadlines(clock):
    engine, policy, stop = _engine(clock)
    queue = engine.action_queue
    clock[0] = 100.73
    queue.merge(_actions(), _actions(), 22)
    assert queue.qsize() == 8
    assert engine.get_action(None) is None
    assert queue.valid and queue.qsize() == 0
    assert queue.startup_discarded_chunks == 1 and queue.startup_discarded_actions == 8
    assert queue.last_action_deadline is None
    assert not engine._ever_had_action and engine.dequeued_actions == engine.executed_actions == 0
    assert not stop.is_set() and not policy.session._closed

    policy._observation_time = 100.8
    clock[0] = 101.09
    queue.merge(_actions(100), _actions(100), 9)
    original_deadlines = [100.8 + (i + 1) / 30 for i in range(9, 30)]
    assert queue._deadlines == pytest.approx(original_deadlines)
    assert engine.get_action(None)[0].item() == 109
    assert queue.last_action_deadline == pytest.approx(original_deadlines[0])
    assert queue._deadlines == pytest.approx(original_deadlines)
    assert queue.startup_accepted_horizon_s == pytest.approx(21 / 30)
    assert queue.expired_prefix_dropped == 31 and queue.overlap_prefix_dropped == 0
    assert engine._ever_had_action and engine.dequeued_actions == 1


def test_startup_checks_actual_deadline_horizon_when_depth_overstates_freshness(clock):
    queue, _ = _queue(clock, max_age_s=0.4)
    clock[0] += 0.1
    queue.merge(_actions(), _actions(), 3)
    snapshot = queue.timing_snapshot()
    assert snapshot["depth"] / 30 > 0.5
    assert snapshot["deadline_horizon_s"] < 0.5
    assert queue.get() is None
    assert queue.startup_discarded_actions == snapshot["depth"]
    assert queue.valid and queue.qsize() == 0


def test_startup_checks_depth_horizon_when_deadline_overstates_available_actions(clock):
    queue, _ = _queue(clock)
    clock[0] += 0.1
    queue.merge(_actions(), _actions(), 21)
    snapshot = queue.timing_snapshot()
    assert snapshot["deadline_horizon_s"] > 0.5
    assert snapshot["depth"] == 9
    assert queue.get() is None
    assert queue.startup_discarded_actions == 9


@pytest.mark.parametrize("minimum,accepted", [(0.5, True), (0.500001, False)])
def test_startup_horizon_boundary_is_not_rounded_up(clock, minimum, accepted):
    queue, _ = _queue(clock, startup_horizon_s=minimum)
    clock[0] += 0.49
    queue.merge(_actions(), _actions(), 15)
    result = queue.get()
    assert (result is not None) is accepted
    if accepted:
        assert result[0].item() == 15
        assert queue.startup_accepted_horizon_s == pytest.approx(0.5)
    else:
        assert queue.startup_discarded_actions == 15


def test_admission_does_not_restart_when_an_admitted_queue_becomes_short(clock):
    queue, observation = _queue(clock)
    queue.merge(_actions(), _actions(), 0)
    for i in range(30):
        assert queue.get()[0].item() == i
    assert queue.startup_discarded_chunks == 0
    # A subsequent short chunk remains governed by the existing running rules.
    observation[0] = clock[0]
    queue.merge(_actions(100, count=1), _actions(100, count=1), 0)
    assert queue.get()[0].item() == 100
    assert queue.startup_discarded_chunks == 0


def test_post_admission_underrun_still_faults_immediately(clock):
    engine, policy, stop = _engine(clock)
    engine.action_queue.merge(_actions(), _actions(), 0)
    for _ in range(30):
        assert engine.get_action(None) is not None
    with pytest.raises(RemoteFault, match="underrun"):
        engine.get_action(None)
    assert engine.underruns == 1 and engine.failed and stop.is_set()
    assert policy.session._closed and not engine.action_queue.valid
    assert engine.action_queue.startup_discarded_chunks == 0


def test_original_expiry_faults_are_not_reclassified_as_startup_discard(clock):
    queue, _ = _queue(clock)
    queue.merge(_actions(), _actions(), 0)
    clock[0] = queue._deadlines[0]
    with pytest.raises(RemoteFault, match="expired"):
        queue.get()
    assert queue.startup_discarded_chunks == 0
    assert queue.expired_queued_actions == 1


@pytest.mark.parametrize("kind", ["nonfinite", "capacity", "fully_expired"])
def test_invalid_chunks_still_fault_before_startup_admission(clock, kind):
    queue, _ = _queue(clock)
    actions = _actions(count=46 if kind == "capacity" else 30)
    if kind == "nonfinite":
        actions[0, 0] = float("nan")
    if kind == "fully_expired":
        clock[0] += 1.1
    with pytest.raises(RemoteFault, match="Nonfinite|capacity|expired"):
        queue.merge(actions, actions, 0)
    assert queue.qsize() == 0 and queue.startup_discarded_chunks == 0


@pytest.mark.parametrize("max_age_s,expected", [(2.0, 0.5), (0.6, 0.3)])
def test_engine_derives_startup_reserve_from_native_and_freshness_horizons(clock, max_age_s, expected):
    engine, _, _ = _engine(clock, max_age_s=max_age_s)
    assert engine.startup_min_horizon_s == pytest.approx(expected)


def test_populated_queue_cannot_escape_original_startup_timeout(clock):
    engine, policy, stop = _engine(clock, timeout_s=2)
    original_started_at = engine._started_at
    clock[0] = policy._observation_time = 102.1
    engine.action_queue.merge(_actions(), _actions(), 0)
    assert engine.action_queue.qsize() == 30
    with pytest.raises(RemoteFault, match="startup|underrun"):
        engine.get_action(None)
    assert engine._started_at == original_started_at
    assert engine.dequeued_actions == engine.executed_actions == 0
    assert engine.failed and stop.is_set() and policy.session._closed
    assert not engine.action_queue.valid


def test_initial_queue_get_crossing_startup_deadline_is_rejected_before_dispatch(clock, monkeypatch):
    engine, policy, stop = _engine(clock, timeout_s=2)
    clock[0] = policy._observation_time = 101.99
    queue = engine.action_queue
    queue.merge(_actions(), _actions(), 0)
    original_get = queue.get

    def blocked_get():
        # Simulate waiting for the queue lock across the fixed startup deadline.
        # The queued actions themselves are still fresh when the lock is obtained.
        clock[0] = 102.01
        return original_get()

    monkeypatch.setattr(queue, "get", blocked_get)
    with pytest.raises(RemoteFault, match="startup"):
        engine.get_action(None)
    assert queue.startup_accepted_horizon_s is not None
    assert engine.dequeued_actions == engine.executed_actions == 0
    assert not engine._ever_had_action
    assert engine.failed and stop.is_set() and policy.session._closed
    assert not queue.valid and queue.qsize() == 0


def test_repeated_short_rejections_never_extend_startup_timeout(clock):
    engine, policy, stop = _engine(clock, timeout_s=2)
    original_started_at = engine._started_at
    for observation_at in [100.0, 100.8]:
        policy._observation_time = observation_at
        clock[0] = observation_at + 0.73
        engine.action_queue.merge(_actions(), _actions(), 22)
        assert engine.get_action(None) is None
    assert engine.action_queue.startup_discarded_chunks == 2
    assert engine._started_at == original_started_at
    clock[0] = 102.01
    with pytest.raises(RemoteFault, match="startup|underrun"):
        engine.get_action(None)
    assert stop.is_set() and policy.session._closed and engine.dequeued_actions == 0


@pytest.mark.parametrize("kind", ["operator_stop", "session_expiry"])
def test_stop_and_session_expiry_prevent_initial_dequeue(clock, kind):
    engine, policy, stop = _engine(clock)
    engine.action_queue.merge(_actions(), _actions(), 0)
    if kind == "operator_stop":
        stop.set()
    else:
        def expired():
            raise RemoteFault("HTTP inference session expired")

        policy.transport.ensure_session_active = expired
    with pytest.raises(RemoteFault, match="stopped|expired"):
        engine.get_action(None)
    assert engine.dequeued_actions == engine.executed_actions == 0
    if kind == "session_expiry":
        assert engine.failed and stop.is_set() and policy.session._closed


def test_rejected_observation_is_not_reused_by_upstream_observation_slot(clock):
    policy = SimpleNamespace(_last_requested_observation_time=100.0)
    slot = _ObservationSlot(policy, "fake")
    slot["obs"] = {"sample": "rejected"}
    slot.timestamp = 100.0
    assert slot.get("obs") is None
    slot["obs"] = {"sample": "fresh"}
    slot.timestamp = 100.8
    assert slot.get("obs") == {"sample": "fresh"}
    assert policy._observation_time == 100.8


def test_all_short_startup_ends_failed_with_zero_policy_sends_and_fake_hardware_released():
    from scripts.benchmark_remote import run_scenario

    result = run_scenario("test_all_short_startup", [0.65], duration=1.6, image_hw=(8, 8))
    assert result["error"] is not None and "startup admitted" in result["error"]
    assert result["failed"] and not result["duration_completed"]
    assert result["executed_actions"] == result["dequeued_actions"] == 0
    assert result["startup_queue"]["discarded_chunks"] >= 1
    assert result["startup_queue"]["accepted_horizon_s"] is None
    assert not result["home_attempted"] and result["all_fake_robots_released"]


def test_startup_admission_waits_for_complete_locked_merge(clock, monkeypatch):
    queue, _ = _queue(clock)
    append_entered = threading.Event()
    allow_append = threading.Event()
    get_entered = threading.Event()
    get_finished = threading.Event()
    results, failures = [], []
    original_append = queue._append_actions_queue

    def paused_append(*args):
        append_entered.set()
        assert allow_append.wait(2)
        original_append(*args)

    def merge():
        try:
            queue.merge(_actions(), _actions(), 0)
        except BaseException as exc:  # noqa: BLE001 — report background failures in the test thread
            failures.append(exc)

    def get():
        get_entered.set()
        try:
            results.append(queue.get())
        except BaseException as exc:  # noqa: BLE001 — report background failures in the test thread
            failures.append(exc)
        finally:
            get_finished.set()

    monkeypatch.setattr(queue, "_append_actions_queue", paused_append)
    producer, consumer = threading.Thread(target=merge), threading.Thread(target=get)
    producer.start()
    try:
        assert append_entered.wait(2)
        consumer.start()
        assert get_entered.wait(2)
        assert not get_finished.wait(0.03)
    finally:
        allow_append.set()
        producer.join(2)
        if consumer.ident is not None:
            consumer.join(2)
    assert not producer.is_alive() and not consumer.is_alive()
    assert not failures and len(results) == 1 and results[0][0].item() == 0
    assert queue.startup_discarded_chunks == 0 and queue.qsize() == 29
