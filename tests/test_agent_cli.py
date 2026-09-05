"""CLI validation and hardware isolation; all provider calls are deterministic and mocked."""

import builtins
import json

import pytest
from typer.testing import CliRunner

from yamkit import agent as controller
from yamkit import agent_openai, agent_robot, cli, paths


@pytest.fixture
def agent_cli_rig(rig, tmp_path, monkeypatch):
    rig.cameras = {"top": {"type": "opencv", "index_or_path": 0}}
    rig.save()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    return rig


def args(rig, *extra):
    return [
        "agent", "--model", "test-model", "--task", "inspect the fixture",
        "--arm", "left_follower", "--rig", str(rig.path), *extra,
    ]


def forbid_hardware_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith(("lerobot_robot_yamkit", "i2rt", "yamkit.arm", "openai")):
            raise AssertionError(f"physical module imported: {name}")
        if name == "arm" and level and globals and globals.get("__package__") == "yamkit":
            raise AssertionError("physical arm module imported")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


def forbidden(*args, **kwargs):
    raise AssertionError("construction must not happen")


def test_offline_dry_run_never_imports_hardware_or_sdk(agent_cli_rig, monkeypatch, tmp_path):
    forbid_hardware_imports(monkeypatch)
    monkeypatch.setattr(agent_robot, "make_live_robot", forbidden)
    monkeypatch.setattr(agent_openai, "OpenAIProvider", forbidden)
    monkeypatch.setattr(agent_openai, "credential_status", forbidden)
    result = CliRunner().invoke(cli.app, args(
        agent_cli_rig, "--dry-run", "--offline", "--settle-s", "0",
        "--max-steps", "5", "--log-path", "outputs/agent/offline.jsonl",
    ))
    assert result.exit_code == 0, result.output
    assert "Hardware dry-run" in result.output
    assert "Offline mocked provider" in result.output
    assert "model-declared" in result.output
    rows = [json.loads(line) for line in (tmp_path / "outputs/agent/offline.jsonl").read_text().splitlines()]
    assert rows[0]["event"] == "start"
    assert rows[-1]["event"] == "termination"
    assert rows[-1]["status"] == "finished"
    assert rows[-1]["success"] is False
    assert rows[-1]["success_basis"] == "model_declared"
    assert any(row["event"] == "readback" for row in rows)
    observations = [row for row in rows if row["event"] == "observation"]
    assert observations
    assert all(row["source"] == "fixture" and row["camera_names"] == ["fixture_top"] for row in observations)


def test_paid_dry_run_uses_fixtures_and_explicit_provider(agent_cli_rig, monkeypatch):
    forbid_hardware_imports(monkeypatch)
    monkeypatch.setattr(agent_robot, "make_live_robot", forbidden)
    monkeypatch.setenv("YAMKIT_OPENAI_API_KEY", "dummy-cli-key")
    seen = []

    def fake_provider(model, task):
        seen.append((model, task))
        return agent_openai.MockProvider()

    monkeypatch.setattr(agent_openai, "OpenAIProvider", fake_provider)
    result = CliRunner().invoke(cli.app, args(agent_cli_rig, "--dry-run", "--settle-s", "0"))
    assert result.exit_code == 0, result.output
    assert seen == [("test-model", "inspect the fixture")]
    assert "Hardware dry-run" in result.output
    assert "Paid OpenAI API mode" in result.output
    assert "Credential: SET" in result.output
    assert "dummy-cli-key" not in result.output


@pytest.mark.parametrize("extra", [
    [], ["--dry-run", "--execute"], ["--offline"], ["--execute", "--offline"],
    ["--dry-run", "--max-steps", "0"], ["--dry-run", "--max-steps", "1001"],
    ["--dry-run", "--settle-s", "-1"], ["--dry-run", "--settle-s", "nan"],
    ["--dry-run", "--max-joint-delta", "0.11"],
    ["--dry-run", "--motion-timeout-s", "0"],
    ["--dry-run", "--api-timeout-s", "inf"],
    ["--dry-run", "--episode-timeout-s", "0"],
    ["--dry-run", "--model", ""], ["--dry-run", "--task", " "],
    ["--dry-run", "--arm", "left_leader"], ["--dry-run", "--arm", "unknown"],
    ["--dry-run", "--arm", "left_follower,right_follower"],
    ["--dry-run", "--log-path", "../outside.jsonl"],
])
def test_invalid_input_precedes_robot_or_provider(agent_cli_rig, monkeypatch, extra):
    for module, names in (
        (agent_robot, ("FixtureRobot", "RobotAdapter", "make_live_robot")),
        (agent_openai, ("OpenAIProvider", "MockProvider")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    result = CliRunner().invoke(cli.app, args(agent_cli_rig, *extra))
    assert result.exit_code == 2, result.output
    assert "construction must not happen" not in result.output


@pytest.mark.parametrize("required", ["--model", "--task", "--arm"])
def test_required_options(agent_cli_rig, required):
    command = args(agent_cli_rig, "--dry-run", "--offline")
    index = command.index(required)
    del command[index:index + 2]
    result = CliRunner().invoke(cli.app, command)
    assert result.exit_code == 2
    assert required in result.output


def test_execute_reports_blockers_before_provider_or_hardware(agent_cli_rig, monkeypatch):
    forbid_hardware_imports(monkeypatch)
    monkeypatch.setattr(agent_openai, "OpenAIProvider", forbidden)
    monkeypatch.setattr(agent_robot, "FixtureRobot", forbidden)
    result = CliRunner().invoke(cli.app, args(agent_cli_rig, "--execute"))
    assert result.exit_code == 1, result.output
    output = " ".join(result.output.split())
    assert "Live agent execution is disabled" in output
    assert "no public no-home option" in output
    assert "No arm or camera was opened" in output


def test_missing_rig_precedes_factories(agent_cli_rig, monkeypatch):
    agent_cli_rig.path.unlink()
    monkeypatch.setattr(agent_openai, "OpenAIProvider", forbidden)
    monkeypatch.setattr(agent_robot, "FixtureRobot", forbidden)
    result = CliRunner().invoke(cli.app, args(agent_cli_rig, "--dry-run"))
    assert result.exit_code == 2
    assert "rig file not found" in result.output


def test_invalid_rig_precedes_factories(agent_cli_rig, monkeypatch):
    agent_cli_rig.control.max_joint_speed = 0
    agent_cli_rig.save()
    monkeypatch.setattr(agent_openai, "OpenAIProvider", forbidden)
    monkeypatch.setattr(agent_robot, "FixtureRobot", forbidden)
    result = CliRunner().invoke(cli.app, args(agent_cli_rig, "--dry-run"))
    assert result.exit_code == 2
    assert "control.max_joint_speed" in result.output


def test_existing_log_is_preserved_before_provider(agent_cli_rig, monkeypatch, tmp_path):
    destination = tmp_path / "previous.jsonl"
    destination.write_text("previous episode\n")
    monkeypatch.setattr(agent_openai, "OpenAIProvider", forbidden)
    result = CliRunner().invoke(cli.app, args(agent_cli_rig, "--dry-run", "--log-path", str(destination)))
    assert result.exit_code == 2
    assert destination.read_text() == "previous episode\n"


def test_unwritable_log_has_clean_error(agent_cli_rig):
    # The rig YAML is a file, so it cannot also be a directory containing logs.
    destination = agent_cli_rig.path / "episode.jsonl"
    result = CliRunner().invoke(cli.app, args(
        agent_cli_rig, "--dry-run", "--offline", "--log-path", str(destination),
    ))
    assert result.exit_code == 1, result.output
    assert "Episode could not run" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(("status", "code"), [("error", 1), ("deadline", 1), ("max_steps", 1), ("cancelled", 130)])
def test_episode_failure_status_reaches_shell(agent_cli_rig, monkeypatch, status, code):
    def run_episode(adapter, provider, config, log_path):
        adapter.close()
        return {"status": status, "reason": "stopped", "success": None, "success_basis": "unverified"}

    monkeypatch.setattr(controller, "run_episode", run_episode)
    result = CliRunner().invoke(cli.app, args(agent_cli_rig, "--dry-run", "--offline"))
    assert result.exit_code == code, result.output
    assert f"Episode: {status}" in result.output


def test_options_reach_controller(agent_cli_rig, monkeypatch):
    seen = []

    def run_episode(adapter, provider, config, log_path):
        seen.append(config)
        adapter.close()
        return {"status": "finished", "reason": "done", "success": False, "success_basis": "model_declared"}

    monkeypatch.setattr(controller, "run_episode", run_episode)
    result = CliRunner().invoke(cli.app, args(
        agent_cli_rig, "--dry-run", "--offline", "--max-steps", "7", "--settle-s", "0.2",
        "--max-joint-delta", "0.05", "--motion-timeout-s", "2", "--api-timeout-s", "10",
        "--episode-timeout-s", "25",
    ))
    assert result.exit_code == 0, result.output
    assert seen == [controller.AgentConfig("test-model", "inspect the fixture", 7, 0.2, 0.05, 2, 10, 25)]


def test_agent_help_explains_paid_dry_run():
    result = CliRunner().invoke(cli.app, ["agent", "--help"])
    assert result.exit_code == 0
    assert "--offline" in result.output
    assert "paid" in result.output
