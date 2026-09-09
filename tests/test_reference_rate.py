"""Literal reference timing through LeRobot helpers; fake clocks/robots only."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.rollout.configs import BaseStrategyConfig

from yamkit import reference_strategy
from yamkit.inference.client import RemoteFault
from yamkit.inference.command_shaping import ACTION_NAMES


class FakeStop:
    def __init__(self, clock):
        self.clock = clock
        self.stopped = False
        self.waits = []
        self.on_wait = None

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, seconds):
        self.waits.append(seconds)
        if self.on_wait is not None:
            self.on_wait(seconds)
        if not self.stopped:
            self.clock[0] += seconds
        return self.stopped


def test_rate_waits_after_send_and_resets_its_origin_after_rpc_overrun(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(reference_strategy, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    stop = FakeStop(clock)
    rate = reference_strategy.ReferenceRate(30, shutdown_event=stop, keep_running=lambda: not stop.is_set())
    clock[0] += .88  # An already completed send after a long synchronous RPC.
    assert rate.sleep() is True
    assert rate.last == clock[0] and stop.waits == []
    first_origin = rate.last
    clock[0] += .003  # Camera copy and the upstream multipoint sleep precede the next send.
    assert rate.sleep() is True
    assert first_origin + 1 / 30 <= rate.last < first_origin + 1 / 30 + .000101
    assert stop.waits and all(value == .0001 for value in stop.waits)
    last = rate.last
    stop.set()
    assert rate.sleep() is False and rate.last == last


@pytest.fixture
def run_case(monkeypatch):
    def build(*, duration=1.6, actions=(True, True, False), rpc_delays=None):
        clock = [100.0]
        stop = FakeStop(clock)
        events, reads, sends, notifications, rpc_inputs, seeds, telemetry = [], [], [], [], [], [], []
        monkeypatch.setattr(reference_strategy, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        start_action = {name: 1.0 if "gripper" in name else .07 for name in ACTION_NAMES}
        inner = SimpleNamespace(_reference_start_action=start_action)
        previous_observer = lambda: None
        engine = SimpleNamespace(observe_row=previous_observer, phase_deadline=None, index=0,
                                 rate_steps=0, multipoint_extra_sleep_calls=0,
                                 last_step_multipoint=False, session_fault_at=None, get_calls=0)
        delays = {0: .88, 2: .42} if rpc_delays is None else rpc_delays

        def check():
            if engine.session_fault_at is not None and clock[0] >= engine.session_fault_at:
                raise RemoteFault("injected session expiry")

        def observe():
            role = getattr(inner, "_reference_observation_role", "ordinary")
            if role == "row_anchor" and getattr(inner, "fail_row_read", False):
                raise TimeoutError("injected row camera failure")
            clock[0] += .002  # A camera/encoder observation takes measurable time.
            sample = dict.fromkeys(ACTION_NAMES, (len(reads) + 1) / 100)
            reads.append((role, clock[0], sample))
            events.append(("read", role, len(reads)))
            return sample

        def notify(observation):
            value = observation[ACTION_NAMES[0]]
            notifications.append(value)
            events.append(("notify", value))

        def seed(observation):
            seeds.append(dict(observation))
            events.append(("seed", observation[ACTION_NAMES[0]]))

        def get_action(frame):
            engine.get_calls += 1
            if engine.index >= len(actions):
                return None
            if engine.index in delays:
                rpc_inputs.append(frame["observation.state"].copy())
                events.append(("rpc", engine.index))
                clock[0] += delays[engine.index]
            if engine.index == 0 or engine.index >= 2:
                engine.observe_row()
            events.append(("select", engine.index))
            return torch.full((14,), (engine.index + 1) / 10, dtype=torch.float64)

        def send(action):
            sends.append((clock[0], dict(action)))
            events.append(("send", engine.index))
            engine.last_step_multipoint = actions[engine.index]
            engine.index += 1
            return dict(action)

        def process(observation):
            events.append(("process", observation[ACTION_NAMES[0]]))
            return observation

        engine._check = check
        engine.resume = lambda: setattr(engine, "phase_deadline", clock[0] + duration)
        engine.reset = engine.start = lambda: None
        engine.notify_observation = notify
        engine.get_action = get_action
        robot = SimpleNamespace(inner=inner, command_shaper=SimpleNamespace(initialize_position=seed),
                                get_observation=observe, send_action=send)
        cfg = SimpleNamespace(fps=30, duration=duration, interpolation_multiplier=1,
                              use_torch_compile=False, display_data=False)
        ctx = SimpleNamespace(runtime=SimpleNamespace(cfg=cfg, shutdown_event=stop),
                              hardware=SimpleNamespace(robot_wrapper=robot), policy=SimpleNamespace(inference=engine),
                              processors=SimpleNamespace(robot_observation_processor=process,
                                                         robot_action_processor=lambda pair: pair[0]),
                              data=SimpleNamespace(ordered_action_keys=ACTION_NAMES, dataset_features={
                                  "observation.state": {"dtype": "float32", "shape": (14,),
                                                        "names": ACTION_NAMES}}))
        strategy = reference_strategy.ReferenceStrategy(BaseStrategyConfig())
        strategy.setup(ctx)  # Real LeRobot interpolator and send_next_action are exercised.
        monkeypatch.setattr(strategy, "_log_telemetry", lambda *args: telemetry.append(clock[0]))
        return SimpleNamespace(clock=clock, stop=stop, ctx=ctx, engine=engine, inner=inner,
                               strategy=strategy, events=events, reads=reads, sends=sends,
                               notifications=notifications, rpc_inputs=rpc_inputs, seeds=seeds,
                               telemetry=telemetry, previous_observer=previous_observer)

    return build


def test_strategy_preserves_post_rpc_short_pair_and_post_step_policy_inputs(run_case):
    case = run_case()
    case.strategy.run(case.ctx)
    assert len(case.sends) == 3
    assert case.sends[1][0] - case.sends[0][0] == pytest.approx(.003)
    assert case.engine.rate_steps == 3 and case.engine.multipoint_extra_sleep_calls == 2
    assert [role for role, _, _ in case.reads] == ["ordinary", "row_anchor", "ordinary",
                                                 "ordinary", "row_anchor", "ordinary"]
    assert case.notifications == [.01, .03, .04, .06]
    np.testing.assert_allclose(case.rpc_inputs, [[.01] * 14, [.04] * 14])
    assert len(case.seeds) == 1 and case.seeds[0] == case.inner._reference_start_action
    assert case.seeds[0] != case.reads[0][2]  # Measured monitoring remains separate from reset-command state.
    assert case.events.index(("seed", .07)) < case.events.index(("rpc", 0))
    assert case.events.index(("send", 0)) < case.events.index(("read", "ordinary", 3))
    assert len(case.telemetry) == 3
    assert case.engine.observe_row is case.previous_observer
    assert not hasattr(case.inner, "_reference_observation_role")


def test_missing_completed_startup_cache_prevents_first_policy_request(run_case):
    case = run_case()
    del case.inner._reference_start_action
    with pytest.raises(RemoteFault, match="startup reset did not complete"):
        case.strategy.run(case.ctx)
    assert not case.rpc_inputs and not case.sends and not case.seeds
    assert case.engine.observe_row is case.previous_observer


@pytest.mark.parametrize("during_extra_sleep", [False, True])
def test_stop_interrupts_rate_or_extra_sleep_without_another_send(run_case, during_extra_sleep):
    case = run_case(duration=.2, actions=(True,), rpc_delays={0: 0.0})

    def stop_wait(seconds):
        if (seconds == .001) == during_extra_sleep:
            case.stop.set()

    case.stop.on_wait = stop_wait
    case.strategy.run(case.ctx)
    assert len(case.sends) == 1
    assert case.engine.rate_steps == int(during_extra_sleep)
    assert case.engine.multipoint_extra_sleep_calls == int(during_extra_sleep)
    assert len(case.reads) == 2 + int(during_extra_sleep)
    assert case.engine.observe_row is case.previous_observer


def test_phase_expiry_during_rate_wait_prevents_later_observation_or_send(run_case):
    case = run_case(duration=.01, actions=(True,), rpc_delays={0: 0.0})
    case.strategy.run(case.ctx)
    assert len(case.sends) == 1 and len(case.reads) == 2
    assert case.engine.rate_steps == case.engine.multipoint_extra_sleep_calls == 0
    assert case.engine.phase_deadline <= case.clock[0] < case.engine.phase_deadline + .000101
    assert not case.stop.is_set()


def test_session_expiry_during_rate_wait_propagates_without_more_motion(run_case):
    case = run_case(duration=.2, actions=(True,), rpc_delays={0: 0.0})
    case.engine.session_fault_at = 100.01
    with pytest.raises(RemoteFault, match="session expiry"):
        case.strategy.run(case.ctx)
    assert len(case.sends) == 1 and len(case.reads) == 2
    assert case.engine.rate_steps == 0
    assert case.engine.observe_row is case.previous_observer


@pytest.mark.parametrize("prior_role", [None, "previous_role"])
def test_row_anchor_role_and_callback_restore_after_observation_error(run_case, prior_role):
    case = run_case(rpc_delays={0: 0.0})
    case.inner.fail_row_read = True
    if prior_role is not None:
        case.inner._reference_observation_role = prior_role
    with pytest.raises(TimeoutError, match="row camera failure"):
        case.strategy.run(case.ctx)
    assert case.sends == [] and len(case.reads) == 1
    assert getattr(case.inner, "_reference_observation_role", None) == prior_role
    assert case.engine.observe_row is case.previous_observer


def test_none_action_tail_waits_without_busy_spin_or_extra_reads(run_case):
    case = run_case(duration=.11, actions=(), rpc_delays={})
    case.strategy.run(case.ctx)
    assert case.sends == [] and len(case.reads) == 1
    assert 1 < case.engine.get_calls <= 4 and 1 < len(case.stop.waits) <= 4
    assert case.engine.rate_steps == case.engine.multipoint_extra_sleep_calls == 0
    assert case.clock[0] == pytest.approx(case.engine.phase_deadline)
