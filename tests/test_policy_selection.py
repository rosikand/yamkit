"""Configured names cannot confuse official OpenPI and the YAM fine-tune."""

import json

import pytest
from typer.testing import CliRunner

from yamkit import cli, inference_workflow
from yamkit.backend_workflow import WorkflowError
from yamkit.policy_selection import OPENPI_YAM_BLOCKER, canonical_policy


@pytest.mark.parametrize("name,expected", [
    ("pi05", "pi05-yam"), ("pi05_yam", "pi05-yam"), ("pi05-yam", "pi05-yam"),
    ("pi05_base", "pi05-base"), ("pi05-base", "pi05-base"),
    ("molmoact2", "molmoact2"), ("lerobot/pi05_base", "lerobot/pi05_base"),
])
def test_only_explicit_configured_aliases(name, expected):
    assert canonical_policy(name) == expected


@pytest.mark.parametrize("name", ["pi05_base", "pi05-base"])
def test_official_base_blocked_before_backend_contact_or_rig_read(name, monkeypatch):
    monkeypatch.setattr(inference_workflow, "load_target", lambda *_a, **_k: pytest.fail("must not contact backend"))
    with pytest.raises(WorkflowError, match="Official OpenPI.*normalization"):
        inference_workflow._prepare_inference(backend="lambda", policy=name, task="cube", rig="missing.yaml",
                                             duration=60, arms=(), config=None, progress=lambda _: None, force=False)


def test_exact_requested_base_cli_explains_contract_without_confirmation(monkeypatch, tmp_path):
    from yamkit import workflow_lock

    monkeypatch.setattr(workflow_lock, "ROOT", tmp_path)
    monkeypatch.setattr(inference_workflow, "ROOT", tmp_path)
    result = CliRunner().invoke(cli.app, ["rollout", "--backend", "lambda", "--policy", "pi05_base",
                                         "--task", "put the red cube into the black container", "--duration", "60"])
    assert result.exit_code == 2
    assert "Official OpenPI" in result.output and "normalization" in result.output
    # Rich wraps messages to terminal width; assert precise semantics separately.
    assert "grippers" in result.output and "transitions" in result.output
    assert "out-of-range grippers" in OPENPI_YAM_BLOCKER
    assert "initial joint transitions" in OPENPI_YAM_BLOCKER
    assert "OPENPI_YAM_EXPERIMENT_2026-09-13.md" in OPENPI_YAM_BLOCKER
    assert "I am on site" not in result.output
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("same", ["service", "endpoint"])
def test_policy_config_cannot_share_model_identity_or_listener(same, monkeypatch, tmp_path):
    from yamkit import backend_workflow

    monkeypatch.setattr(backend_workflow, "ROOT", tmp_path)
    first = {"service": "ma2", "endpoint": "http://127.0.0.1:8765"}
    second = {"service": "pi05-yam", "endpoint": "http://127.0.0.1:8766"}
    second[same] = first[same]
    path = tmp_path / "backends.json"
    path.write_text(json.dumps({"version": 1, "backends": {"lambda": {"policies": {
        "molmoact2": first, "pi05-yam": second}}}}))
    path.chmod(0o600)
    with pytest.raises(WorkflowError, match="own service name and loopback port"):
        backend_workflow.load_target("lambda", "pi05_yam", config=path)
