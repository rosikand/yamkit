"""Maintained inference endpoints through real command clamps and fake motors only."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yamkit import arm as arm_module
from yamkit import reference_rollout, remote_rollout
from yamkit.config import ArmSpec
from yamkit.inference.client import RemoteFault
from yamkit.inference.command_shaping import ACTION_NAMES
from yamkit.inference.profiles import get_profile
from yamkit.inference.reference import ReferenceCommandGuard


class LaggingMotor:
    """Measured joints follow each fake command with a constant joint-3 offset."""

    def __init__(self, target, lag, main_thread):
        self.lag = lag
        self.main_thread = main_thread
        self.measured = target.copy()
        self.measured[2] -= lag
        self.writes = []
        self.access_threads = []

    def _main_only(self):
        thread = threading.get_ident()
        self.access_threads.append(thread)
        assert thread == self.main_thread, "RPC worker accessed a hardware method"

    def num_dofs(self):
        self._main_only()
        return 7

    def get_robot_info(self):
        self._main_only()
        return {"kp": np.full(7, 80.0), "kd": np.full(7, 5.0)}

    def get_observations(self):
        self._main_only()
        return {"joint_pos": self.measured[:6].copy(), "joint_vel": np.zeros(6),
                "joint_eff": np.zeros(6), "gripper_pos": self.measured[6:7].copy()}

    def command_joint_pos(self, target):
        self._main_only()
        self.writes.append(np.array(target).copy())
        self.measured = np.array(target).copy()
        self.measured[2] -= self.lag


class FakeBimanual:
    robot_type = "bi_yam_follower"

    def __init__(self, arms, main_thread):
        self.arms = arms
        self.main_thread = main_thread
        self.postclamp_delta = 0.0

    def _main_only(self):
        assert threading.get_ident() == self.main_thread

    def get_joint_state(self):
        self._main_only()
        return {name: float(value) for side, arm in self.arms.items()
                for name, value in zip(self.names(side), arm.read().vector(), strict=True)}

    @staticmethod
    def names(side):
        return [name for name in ACTION_NAMES if name.startswith(side + "_")]

    def joint_command_limits(self):
        self._main_only()
        return {f"{side}_joint_{i}.pos": {"lower": float(bounds[0]), "upper": float(bounds[1]), "max_step": .03}
                for side, arm in self.arms.items() for i, bounds in enumerate(arm.target_bounds(), start=1)}

    def validate_action_target(self, action):
        self._main_only()
        for side, arm in self.arms.items():
            arm.validate_target([action[f"{side}_joint_{i}.pos"] for i in range(1, 7)],
                                action[f"{side}_gripper.pos"])

    def send_action(self, action):
        self._main_only()
        for side, arm in self.arms.items():
            arm.validate_command([action[f"{side}_joint_{i}.pos"] for i in range(1, 7)],
                                 action[f"{side}_gripper.pos"])
        result = {}
        for side, arm in self.arms.items():
            sent = arm.command([action[f"{side}_joint_{i}.pos"] for i in range(1, 7)],
                               action[f"{side}_gripper.pos"])
            result.update(dict(zip(self.names(side), sent.tolist(), strict=True)))
        result["left_gripper.pos"] += self.postclamp_delta
        return result


@pytest.fixture
def maintained(monkeypatch):
    main_thread = threading.get_ident()
    clock = [100.0]
    fake_time = SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0])
    for module in (arm_module, reference_rollout, remote_rollout):
        monkeypatch.setattr(module, "time", fake_time)

    def forbidden_connection(*args, **kwargs):
        raise AssertionError("No hardware may connect in this test")

    monkeypatch.setattr(arm_module.YamArm, "connect", forbidden_connection)
    target = {name: .8 if "gripper" in name else .4 for name in ACTION_NAMES}
    arms, motors = {}, {}
    for side, lag in (("left", .0355), ("right", .0337)):
        motor = LaggingMotor(np.array([target[name] for name in FakeBimanual.names(side)]), lag, main_thread)
        spec = ArmSpec(name=f"{side}_follower", role="follower", side=side,
                       gripper="linear_4310", can_serial="FAKE-OFFLINE-ONLY")
        arms[side] = arm_module.YamArm(spec, "FAKE-NO-DEVICE", motor, max_joint_speed=3, max_gripper_speed=3)
        motors[side] = motor
    inner = FakeBimanual(arms, main_thread)
    guard = ReferenceCommandGuard(inner.get_joint_state(), inner.joint_command_limits(),
                                  {"left_gripper.pos": .03, "right_gripper.pos": .03})
    stop, entered, release = threading.Event(), threading.Event(), threading.Event()
    calls, worker_threads = [], []

    def predict(batch):
        worker_threads.append(threading.get_ident())
        calls.append(batch["observation.state"].clone())
        if len(calls) == 2:
            entered.set()
            assert release.wait(5), "Test did not release its fake RPC worker"
        values = [value + (.02 if len(calls) > 1 and "joint_" in name else 0)
                  for name, value in target.items()]
        return torch.tensor([[values] * 30], dtype=torch.float32)

    policy = SimpleNamespace(transport=SimpleNamespace(), config=SimpleNamespace(
        request_timeout_s=10, max_observation_age_s=2), _last_prediction_timing={},
        predict_action_chunk=predict, reset=lambda: None, close=release.set)
    robot = remote_rollout._StoppableRobot(inner, stop, command_shaper=guard)
    engine = reference_rollout.ReferenceRemoteInferenceEngine(
        policy=policy, preprocessor=lambda value: value, postprocessor=lambda value: value,
        robot_wrapper=robot, task="put the red cube into the black container", fps=30,
        shutdown_event=stop, duration=30,
        gripper_max_step={"left_gripper.pos": .03, "right_gripper.pos": .03})
    robot.on_action, robot.on_commit = engine.record_execution, engine.record_commit
    robot.on_fault, robot.on_dispatch = engine._fault, engine.record_dispatch
    robot.action_deadline = lambda: engine.action_deadline
    observation = {"observation.state": np.array(list(inner.get_joint_state().values()), dtype=np.float32),
                   **{f"observation.images.{name}": np.zeros((2, 2, 3), dtype=np.uint8)
                      for name in get_profile("molmoact2").image_keys}}
    engine.start()
    engine.resume()

    def tick(dt=.034, *, dispatch=True):
        clock[0] += dt
        engine.notify_observation(observation)
        action = engine.get_action(observation)
        if action is not None and dispatch:
            robot.send_action(dict(zip(ACTION_NAMES, action.tolist(), strict=True)))
        return action

    # First worker has no simulated latency. Wait explicitly for scheduling;
    # test clock progression and wall-clock thread scheduling stay separate.
    tick()
    if engine._rpc is not None:
        assert engine._rpc["done"].wait(2)
    for _ in range(300):
        if engine.completed_chunks:
            break
        tick()
    assert engine.completed_chunks == 1 and engine.completed_steps == 30
    anchor = guard.last_action
    prior_interpolation = engine.interpolation_dispatches
    prior_writes = {side: len(motor.writes) for side, motor in motors.items()}
    tick()  # Begins the blocked second request and commits its first hold.
    assert entered.wait(2)
    assert engine._rpc is not None and not engine._rpc["done"].is_set()
    result = SimpleNamespace(clock=clock, engine=engine, guard=guard, robot=robot, inner=inner,
                             arms=arms, motors=motors, stop=stop, release=release, tick=tick,
                             worker_threads=worker_threads, main_thread=main_thread,
                             anchor=anchor, prior_interpolation=prior_interpolation,
                             prior_writes=prior_writes, calls=calls)
    try:
        yield result
    finally:
        slot = engine._rpc
        release.set()
        if slot is not None:
            assert slot["done"].wait(2)
        engine.invalidate()


def test_lagging_real_arm_clamps_survive_maintained_880ms_rpc(maintained):
    case = maintained
    started = case.engine._rpc["event"]["prediction_started_monotonic_s"]
    while case.clock[0] - started < .88:
        case.tick()
        assert case.engine.completed_chunks == 1 and case.engine.completed_steps == 30
        assert case.engine.interpolation_dispatches == case.prior_interpolation
        assert case.guard.last_action == case.anchor
    assert case.clock[0] - started == pytest.approx(.884)
    assert case.engine.inference_hold_dispatches == 27
    assert arm_module.STALE_COMMAND_S == .5 and arm_module.MAX_COMMAND_DT == .01
    for side, arm in case.arms.items():
        assert arm.max_joint_speed == arm.max_gripper_speed == 3
        assert case.clock[0] - arm._last_cmd_t < .1
        assert case.anchor[f"{side}_joint_3.pos"] - arm.read().q[2] == pytest.approx(case.motors[side].lag)
        assert len(case.motors[side].writes) - case.prior_writes[side] == 27
        assert all(thread == case.main_thread for thread in case.motors[side].access_threads)
    assert case.guard.postclamp_modified_count == 0
    assert all(thread != case.main_thread for thread in case.worker_threads)
    slot = case.engine._rpc
    case.release.set()
    assert slot["done"].wait(2)
    case.tick()  # Next chunk's initial endpoint, preserving the maintained clock.
    assert case.engine.admitted_steps == 60 and case.engine.completed_steps == 30
    assert case.engine.interpolation_dispatches == case.prior_interpolation + 1
    assert case.guard.valid and not case.engine.failed and not case.stop.is_set()
    event = case.engine.predictions[-1]
    assert event["actions_executed_during_prediction"] == 0
    assert event["maintenance_holds_during_prediction"] == 27
    assert torch.equal(case.calls[-1], torch.tensor([list(case.anchor.values())], dtype=torch.float32))
    assert case.engine.executed_actions == case.engine.interpolation_dispatches + case.engine.inference_hold_dispatches
    hold_events = [event for event in case.engine.dispatch_samples if event["dispatch_role"] == "inference_hold"]
    assert len(hold_events) == 27
    assert all(event["target"] == case.anchor and "row_index" not in event for event in hold_events)


def test_maintained_rpc_stall_faults_before_any_hardware_send(maintained):
    case = maintained
    counts = {side: len(motor.writes) for side, motor in case.motors.items()}
    with pytest.raises(ValueError, match="stalled|gap|clock"):
        case.tick(.101)
    assert not case.guard.valid and case.stop.is_set() and case.engine.failed
    assert counts == {side: len(motor.writes) for side, motor in case.motors.items()}
    assert case.engine.completed_steps == 30


def test_late_worker_reply_cannot_extend_maintained_wait_deadline(maintained):
    case = maintained
    slot = case.engine._rpc
    counts = {side: len(motor.writes) for side, motor in case.motors.items()}
    case.clock[0] = slot["deadline"] + .001
    case.release.set()
    assert slot["done"].wait(2)
    with pytest.raises(RemoteFault, match="deadline|expired"):
        case.tick(0)
    assert not case.guard.valid and case.stop.is_set()
    assert case.engine.admitted_steps == case.engine.completed_steps == 30
    assert counts == {side: len(motor.writes) for side, motor in case.motors.items()}


def test_stop_between_hold_selection_and_send_prevents_hardware(maintained):
    case = maintained
    action = case.tick(dispatch=False)
    counts = {side: len(motor.writes) for side, motor in case.motors.items()}
    case.stop.set()
    with pytest.raises(RemoteFault, match="stopped"):
        case.robot.send_action(dict(zip(ACTION_NAMES, action.tolist(), strict=True)))
    assert not case.guard.valid
    assert counts == {side: len(motor.writes) for side, motor in case.motors.items()}
    assert case.engine.completed_steps == 30


def test_postclamp_modified_hold_faults_without_advancing_model_row(maintained):
    case = maintained
    completed_holds = case.engine.inference_hold_dispatches
    case.inner.postclamp_delta = .001  # Simulated feedback intervention, with no real device.
    with pytest.raises(ValueError, match="Postclamp reference"):
        case.tick()
    assert not case.guard.valid and case.stop.is_set() and case.engine.failed
    assert case.guard.postclamp_modified_count == 1
    assert case.engine.inference_hold_dispatches == completed_holds + 1  # Sent, then faulted.
    assert case.engine.completed_steps == 30 and case.engine.completed_chunks == 1
    assert case.engine.interpolation_dispatches == case.prior_interpolation
