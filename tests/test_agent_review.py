"""Independent fault-path checks for the fixture-only agent operation."""

import json

import pytest

from yamkit.agent import AgentConfig, JsonlLog, run_episode
from yamkit.agent_openai import Decision, ToolCall
from yamkit.agent_robot import GRIPPER_KEY, JOINT_KEYS, FixtureRobot, RobotAdapter


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        self.now += 0.001  # model the time needed to acquire fixture observations
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Provider:
    def __init__(self, name="move_joints", arguments=None, result_error=False):
        self.name = name
        self.arguments = arguments if arguments is not None else {"delta": [0.1, 0, 0, 0, 0, 0]}
        self.calls = 0
        self.result_error = result_error

    def decide(self, observation, *, timeout_s):
        self.calls += 1
        return Decision(
            f"response-{self.calls}",
            [ToolCall(f"call-{self.calls}", self.name, json.dumps(self.arguments))],
        )

    def record_result(self, decision, result):
        if self.result_error:
            raise RuntimeError("provider history failed")


def run(tmp_path, robot_type=FixtureRobot, provider=None, *, cancelled=None):
    clock = Clock()
    robot = robot_type(clock=clock)
    provider = provider or Provider()
    adapter = RobotAdapter(robot, clock=clock)
    config = AgentConfig(model="mock", task="fixture test", max_steps=2, settle_s=0)
    result = run_episode(
        adapter, provider, config, tmp_path / "review.jsonl", clock=clock, sleep=clock.sleep,
        cancelled=(lambda: cancelled(robot)) if cancelled else (lambda: False),
    )
    return result, robot, provider


def test_invalid_config_closes_owned_fixture(tmp_path):
    robot = FixtureRobot()
    config = AgentConfig(model="", task="fixture test")
    with pytest.raises(ValueError):
        run_episode(RobotAdapter(robot), Provider(), config, tmp_path / "review.jsonl")
    assert robot.closed
    assert robot.commands == []


def test_existing_log_closes_owned_fixture_without_overwriting(tmp_path):
    path = tmp_path / "review.jsonl"
    path.write_text("existing log\n")
    robot = FixtureRobot()
    config = AgentConfig(model="mock", task="fixture test")
    with pytest.raises(FileExistsError):
        run_episode(RobotAdapter(robot), Provider(), config, path)
    assert robot.closed
    assert robot.commands == []
    assert path.read_text() == "existing log\n"


def test_invalid_feedback_after_send_stops_without_another_action(tmp_path):
    class BrokenFeedback(FixtureRobot):
        def get_observation(self):
            raw = super().get_observation()
            if self.commands:
                raw[JOINT_KEYS[0]] = float("nan")
            return raw

    result, robot, provider = run(tmp_path, BrokenFeedback)
    assert result["status"] == "error"
    assert "ObservationError" in result["reason"]
    assert len(robot.commands) == provider.calls == 1
    assert robot.closed


def test_sent_gripper_target_does_not_count_as_measured_completion(tmp_path):
    class NontrackingRobot(FixtureRobot):
        def send_action(self, action):
            self.commands.append(dict(action))
            return dict(action)  # acknowledge command but measured state never moves

    result, robot, provider = run(tmp_path, NontrackingRobot, Provider("open_gripper", {}))
    assert result["status"] == "error"
    assert len(robot.commands) == provider.calls == 1
    assert robot.commands[0][GRIPPER_KEY] == 1.0
    assert robot.state[GRIPPER_KEY] == 0.5
    assert robot.closed


def test_failure_recording_action_result_does_not_replay_action(tmp_path):
    result, robot, provider = run(tmp_path, provider=Provider(result_error=True))
    assert result["status"] == "error"
    assert len(robot.commands) == provider.calls == 1
    assert robot.state[JOINT_KEYS[0]] == pytest.approx(0.1)
    assert robot.closed


def test_log_failure_after_send_stops_before_next_decision(tmp_path, monkeypatch):
    original = JsonlLog.write

    def write(self, event, **fields):
        if event == "readback":
            raise OSError("log disk error")
        return original(self, event, **fields)

    monkeypatch.setattr(JsonlLog, "write", write)
    result, robot, provider = run(tmp_path)
    assert result["status"] == "error"
    assert len(robot.commands) == provider.calls == 1
    assert robot.closed


def test_cancellation_after_send_never_sends_again(tmp_path):
    result, robot, provider = run(tmp_path, cancelled=lambda robot: bool(robot.commands))
    assert result["status"] == "cancelled"
    assert len(robot.commands) == provider.calls == 1
    assert robot.closed
