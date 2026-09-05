import json
from dataclasses import replace

import pytest

from yamkit.agent import AgentConfig, JsonlLog, run_episode, validate_tool
from yamkit.agent_openai import Decision, ToolCall
from yamkit.agent_robot import FixtureRobot, RobotAdapter


class Clock:
    def __init__(self):
        self.now = 1.0

    def __call__(self):
        self.now += 0.0001  # acquisition/call overhead, distinct monotonic capture times
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Script:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.results = []
        self.observations = []

    def decide(self, obs, *, timeout_s):
        self.observations.append(obs)
        calls = next(self.calls)
        return Decision("resp_test", calls, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    def record_result(self, decision, result):
        self.results.append(result)


def call(name, args=None, call_id="call_1"):
    return ToolCall(call_id, name, json.dumps(args or {}))


def finish(call_id="done"):
    return call("finish", {"success": True, "reason": "fixture only"}, call_id)


@pytest.fixture
def episode(tmp_path):
    def run(calls, robot_type=FixtureRobot, config=None, cancelled=lambda: False):
        clock = Clock()
        robot = robot_type(clock=clock)
        provider = Script(calls)
        log = tmp_path / "episode.jsonl"
        result = run_episode(RobotAdapter(robot, clock=clock), provider,
                             config or AgentConfig("test-model", "fixture task", settle_s=0.2), log,
                             clock=clock, sleep=clock.sleep, cancelled=cancelled)
        return result, robot, provider, [json.loads(line) for line in log.read_text().splitlines()]
    return run


@pytest.mark.parametrize("name,arguments", [
    ("shell", "{}"), ("observe", '{"x": 1}'), ("open_gripper", "[]"),
    ("close_gripper", '{"target": 0}'), ("move_joints", '{"delta":[0,0,0,0,0]}'),
    ("move_joints", '{"delta":[true,0,0,0,0,0]}'),
    ("move_joints", '{"delta":[NaN,0,0,0,0,0]}'),
    ("move_joints", '{"delta":[1e9999,0,0,0,0,0]}'),
    ("move_joints", '{"delta":["0",0,0,0,0,0]}'),
    ("move_joints", '{"delta":null}'), ("move_joints", '{"delta":[0,0,0,0,0,0],"x":0}'),
    ("observe", "null"), ("observe", "not json"),
    ("finish", '{"success":1,"reason":"no"}'), ("finish", '{"success":true,"reason":""}'),
    ("finish", '{"success":true,"success":false,"reason":"no"}'),
])
def test_reject_invalid_tools(name, arguments):
    with pytest.raises(ValueError):
        validate_tool(name, arguments)


@pytest.mark.parametrize("change", [
    {"max_steps": 0}, {"max_steps": True}, {"max_steps": 1001}, {"model": ""}, {"task": " "},
    {"settle_s": -1}, {"settle_s": float("nan")}, {"max_joint_delta": .11},
    {"motion_timeout_s": 0}, {"api_timeout_s": float("inf")}, {"episode_timeout_s": -1},
])
def test_invalid_configuration(change):
    with pytest.raises(ValueError):
        replace(AgentConfig("test", "task"), **change).validate()


def test_relative_delta_is_clamped_and_single_fixed_target(episode):
    class Slow(FixtureRobot):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.state["joint_1.pos"] = .4

        def send_action(self, action):
            self.commands.append(dict(action))
            self.state = {k: self.state[k] + max(-.025, min(.025, v - self.state[k]))
                          for k, v in action.items()}
            return dict(self.state)

    result, robot, provider, events = episode([
        [call("move_joints", {"delta": [100, -.2, 0, 0, 0, 0]})], [finish()],
    ], Slow)
    assert result["status"] == "finished" and robot.closed
    assert result["success_basis"] == "model_declared"
    assert len(robot.commands) >= 4
    assert all(target == robot.commands[0] for target in robot.commands)
    assert robot.commands[0]["joint_1.pos"] == pytest.approx(.5)
    assert robot.commands[0]["joint_2.pos"] == pytest.approx(-.1)
    assert all(t["gripper.pos"] == .5 for t in robot.commands)
    assert provider.observations[1].captured_at > provider.observations[0].captured_at + .2
    assert provider.results[0]["post_settle"]["sequence"] > provider.observations[0].sequence
    assert {e["event"] for e in events} >= {"target", "readback", "action_complete", "termination"}
    assert events[-1]["usage"]["total_tokens"] == 30


@pytest.mark.parametrize("name,target", [("open_gripper", 1.), ("close_gripper", 0.)])
def test_gripper_preserves_all_starting_joints(episode, name, target):
    result, robot, _, _ = episode([[call(name)], [finish()]])
    assert result["status"] == "finished"
    assert robot.commands[0] == {**dict.fromkeys([f"joint_{i}.pos" for i in range(1, 7)], 0.),
                                 "gripper.pos": target}


def test_multiple_duplicate_malformed_and_empty_decisions_consume_budget(episode):
    move = call("move_joints", {"delta": [.01, 0, 0, 0, 0, 0]}, "same")
    result, robot, provider, _ = episode([
        [move, call("open_gripper", call_id="other")], [move], [],
        [ToolCall("bad", "observe", '{"no":true}')], [call("observe", call_id="obs")],
    ], config=AgentConfig("test", "task", max_steps=5))
    assert result["status"] == "max_steps" and result["steps"] == 5
    assert robot.commands == [] and robot.closed
    assert len(provider.results) == 5
    assert all(not r["ok"] for r in provider.results[:4])


def test_duplicate_action_call_never_replays(episode):
    move = call("move_joints", {"delta": [.01, 0, 0, 0, 0, 0]})
    result, robot, _, _ = episode([[move], [move], [finish()]])
    assert result["status"] == "finished"
    assert len(robot.commands) == 1


def test_returned_target_is_not_measured_completion(episode):
    class Stuck(FixtureRobot):
        def send_action(self, action):
            self.commands.append(dict(action))
            return dict(action)

    result, robot, _, events = episode([[call("move_joints", {"delta": [.1, 0, 0, 0, 0, 0]})]],
                                     Stuck, AgentConfig("test", "task", motion_timeout_s=.35))
    assert result["status"] == "deadline" and robot.closed
    assert 1 <= len(robot.commands) <= 4
    assert not any(e["event"] == "action_complete" for e in events)


def test_excessive_tracking_error_stops_immediately(episode):
    class Diverged(FixtureRobot):
        def send_action(self, action):
            self.commands.append(dict(action))
            self.state["joint_1.pos"] = 2.
            return dict(action)

    result, robot, _, _ = episode([[call("open_gripper")]], Diverged)
    assert result["status"] == "error" and robot.closed
    assert len(robot.commands) == 1


def test_stale_feedback_aborts_and_releases_fixture(episode):
    class Stale(FixtureRobot):
        def get_observation(self):
            obs = super().get_observation()
            if self.commands:
                obs["__yamkit_agent_observation__"]["captured_at"] -= 10
            return obs

    result, robot, _, _ = episode([[call("open_gripper")]], Stale)
    assert result["status"] == "error" and robot.closed and len(robot.commands) == 1


def test_drift_during_settle_stops_before_next_decision(episode):
    class Drifts(FixtureRobot):
        reads_after_send = 0

        def get_observation(self):
            if self.commands:
                self.reads_after_send += 1
                if self.reads_after_send > 1:
                    self.state["joint_1.pos"] += 1
            return super().get_observation()

    result, robot, provider, events = episode([[call("open_gripper")], [finish()]], Drifts)
    assert result["status"] == "error" and "after settling" in result["reason"]
    assert robot.closed and len(robot.commands) == 1 and len(provider.observations) == 1
    assert not any(e["event"] == "action_complete" for e in events)


def test_cancellation_does_not_start_action(episode):
    result, robot, _, events = episode([[finish()]], cancelled=lambda: True)
    assert result["status"] == "cancelled" and robot.closed and robot.commands == []
    assert events[-1]["event"] == "termination"


def test_api_error_body_is_never_logged_and_no_action_retry(tmp_path):
    class Broken:
        def decide(self, *_args, **_kwargs):
            raise RuntimeError("SECRET data:image/jpeg;base64,PRIVATE")

    robot = FixtureRobot()
    path = tmp_path / "error.jsonl"
    result = run_episode(RobotAdapter(robot), Broken(), AgentConfig("test", "task"), path)
    assert result["status"] == "error" and robot.closed and robot.commands == []
    assert "SECRET" not in path.read_text() and "base64" not in path.read_text()


@pytest.mark.parametrize("api,episode_timeout", [(.1, 3.), (3., .1)])
def test_late_api_response_cannot_execute(tmp_path, api, episode_timeout):
    clock = Clock()
    robot = FixtureRobot(clock=clock)

    class Late(Script):
        def decide(self, *args, **kwargs):
            clock.sleep(.2)
            return super().decide(*args, **kwargs)

    provider = Late([[call("open_gripper")]])
    result = run_episode(RobotAdapter(robot, clock=clock), provider,
                         AgentConfig("test", "task", api_timeout_s=api, episode_timeout_s=episode_timeout),
                         tmp_path / "late.jsonl", clock=clock, sleep=clock.sleep)
    assert result["status"] == "deadline" and robot.commands == [] and robot.closed


def test_log_redacts_only_allowed_credentials_and_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMKIT_OPENAI_API_KEY", "dummy-super-secret")
    path = tmp_path / "log.jsonl"
    log = JsonlLog(path)
    log.write("start", task="dummy-super-secret")
    log.close()
    assert "dummy-super-secret" not in path.read_text()
    with pytest.raises(FileExistsError):
        JsonlLog(path)
