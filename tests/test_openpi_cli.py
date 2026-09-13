"""Exact official-base CLI routing; all service and physical delegates are mocked."""

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from yamkit import backend_workflow, cli, inference_workflow, workflow_lock
from yamkit.backend_workflow import WorkflowError
from yamkit.openpi import workflow
from yamkit.openpi.interface import CONTRACT_ID

TASK = "put the red cube into the black container"
COMMAND = ["rollout", "--backend", "lambda", "--policy", "pi05_base", "--task", TASK, "--duration", "60"]


@pytest.fixture
def official(tmp_path, monkeypatch):
    from yamkit import fake_inference, fake_ma2_workflow, pi05_workflow

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(inference_workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow_lock, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "_rig_arms", lambda *_a, **_kw: pytest.fail("no device resolution in CLI tests"))
    preparations, runs, idle = [], [], []
    monkeypatch.setattr(backend_workflow, "assert_ui_idle", lambda **kwargs: idle.append(kwargs))

    def prepare(**kwargs):
        preparations.append(kwargs)
        selection = workflow.OpenPiSelection(kwargs["task"], str(kwargs["rig"]), kwargs["duration"],
                                              "lambda-openpi", kwargs["config"], tmp_path / "unopened-proof.json")
        return selection, {"ready": True, "hardware_tested": False, "policy": "pi05-base",
                           "controller_mode": CONTRACT_ID, "motion_approval_received": False}

    def run(selection, **kwargs):
        runs.append((selection, kwargs))
        return {"exit_status": 0, "hardware_tested": False, "policy": selection.policy,
                "status": "explicit mocked delegate only"}

    monkeypatch.setattr(workflow, "prepare", prepare)
    monkeypatch.setattr(workflow, "run_prepared", run)
    monkeypatch.setattr(inference_workflow, "ensure_backend", lambda *_a, **_kw: pytest.fail("no MA2 fallback"))
    monkeypatch.setattr(pi05_workflow, "prepare_pi05", lambda *_a, **_kw: pytest.fail("no PI-YAM substitute"))
    monkeypatch.setattr(pi05_workflow, "run_prepared_pi05", lambda *_a, **_kw: pytest.fail("no PI-YAM executor"))
    monkeypatch.setattr(fake_ma2_workflow, "run_fake_molmoact2", lambda *_a, **_kw: pytest.fail("no MA2 fake executor"))
    monkeypatch.setattr(fake_inference, "run_fake_pi05", lambda *_a, **_kw: pytest.fail("no PI-YAM fake executor"))
    return SimpleNamespace(root=tmp_path, preparations=preparations, runs=runs, idle=idle)


def test_exact_deliverable_prepares_then_requires_fresh_prompt_approval(official):
    result = CliRunner().invoke(cli.app, COMMAND, input="n\n")
    assert result.exit_code != 0
    assert len(official.preparations) == 1 and not official.runs
    assert official.preparations[0]["task"] == TASK and official.preparations[0]["duration"] == 60
    assert "Pending one supervised command" in result.output
    assert "secure mounts" in result.output and "Stop/power cutoff" in result.output
    assert "I am on site" in result.output


def test_exact_deliverable_yes_reaches_only_official_delegate_with_both_flags(official):
    result = CliRunner().invoke(cli.app, COMMAND, input="y\n")
    assert result.exit_code == 0, result.output
    assert len(official.runs) == 1
    selection, values = official.runs[0]
    assert selection.policy == "pi05-base" and selection.task == TASK and selection.duration == 60
    assert selection.controller_mode == CONTRACT_ID
    assert values["confirm_supervised"] is True and values["accept_mapping"] is True
    assert "fake_hardware" not in values
    assert official.idle == [{"require_cameras_idle": True}]


def test_explicit_physical_flags_are_forwarded_without_second_prompt(official):
    result = CliRunner().invoke(cli.app, [*COMMAND, "--confirm-supervised", "--accept-mapping"])
    assert result.exit_code == 0, result.output
    assert len(official.runs) == 1 and "I am on site" not in result.output


@pytest.mark.parametrize("partial", ["--confirm-supervised", "--accept-mapping"])
def test_partial_flags_still_require_fresh_interactive_approval(official, partial):
    result = CliRunner().invoke(cli.app, [*COMMAND, partial], input="n\n")
    assert result.exit_code != 0 and not official.runs
    assert "I am on site" in result.output


def test_fake_deliverable_runs_saved_software_branch_without_any_approval(official):
    result = CliRunner().invoke(cli.app, [*COMMAND, "--fake-hardware", "--capture-trace",
                                         "--upload-repo-id", "example/private-rollouts"])
    assert result.exit_code == 0, result.output
    selection, values = official.runs[0]
    assert selection.policy == "pi05-base"
    assert values["fake_hardware"] is True and values["capture_trace"] is True
    assert values["upload_repo_id"] == "example/private-rollouts"
    assert values["artifact_dir"].is_relative_to(official.root / "outputs/ui/deployments")
    assert "confirm_supervised" not in values and "accept_mapping" not in values
    assert "I am on site" not in result.output
    assert '"hardware_tested": false' in result.output and '"physical_task_success": null' in result.output


@pytest.mark.parametrize("extra", ["--confirm-supervised", "--accept-mapping", "--dry-run"])
def test_fake_mode_cannot_mix_physical_approval_or_dry_run(official, extra):
    result = CliRunner().invoke(cli.app, [*COMMAND, "--fake-hardware", extra])
    assert result.exit_code == 2 and not official.preparations and not official.runs


@pytest.mark.parametrize("arguments", [["--controller-mode", "reference"], ["--execution-mode", "cuda_graph10"],
                                      ["--call-mode", "remote"], ["--fps", "50"], ["--rtc"],
                                      ["--center-crop"], ["--image-encoding", "jpeg"]])
def test_official_native_defaults_cannot_be_silently_overridden(official, arguments):
    result = CliRunner().invoke(cli.app, [*COMMAND, *arguments])
    assert result.exit_code == 2 and not official.preparations and not official.runs


def test_official_dry_run_and_inference_are_software_only(official):
    dry = CliRunner().invoke(cli.app, [*COMMAND, "--dry-run"])
    inference = CliRunner().invoke(cli.app, ["inference", "--backend", "lambda", "--policy", "pi05_base",
                                            "--task", TASK, "--duration", "60"])
    assert dry.exit_code == inference.exit_code == 0
    assert len(official.preparations) == 2 and not official.runs
    assert "no rollout was launched" in dry.output and "No hardware" in inference.output


def test_prepare_failure_does_not_prompt_or_delegate(official, monkeypatch):
    def fail(**kwargs):
        raise WorkflowError("Official OpenPI qualification needs matching retained evidence")

    monkeypatch.setattr(workflow, "prepare", fail)
    result = CliRunner().invoke(cli.app, COMMAND, input="y\n")
    assert result.exit_code == 2 and not official.runs
    assert "I am on site" not in result.output


def test_fake_delegate_exception_does_not_leak_private_text(official, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("fake-private-value-not-safe-to-display")

    monkeypatch.setattr(workflow, "run_prepared", fail)
    result = CliRunner().invoke(cli.app, [*COMMAND, "--fake-hardware"])
    assert result.exit_code == 2
    assert "Software-only fake execution failed" in result.output
    assert "fake-private-value" not in result.output
