"""Configured names cannot confuse official OpenPI and the YAM fine-tune."""

import json

import pytest
from typer.testing import CliRunner

from yamkit import cli, inference_workflow
from yamkit.backend_workflow import WorkflowError
from yamkit.policy_selection import canonical_policy


@pytest.mark.parametrize("name,expected", [
    ("pi05", "pi05-yam"), ("pi05_yam", "pi05-yam"), ("pi05-yam", "pi05-yam"),
    ("pi05_base", "pi05-base"), ("pi05-base", "pi05-base"),
    ("molmoact2", "molmoact2"), ("lerobot/pi05_base", "lerobot/pi05_base"),
])
def test_only_explicit_configured_aliases(name, expected):
    assert canonical_policy(name) == expected


@pytest.mark.parametrize("name", ["pi05_base", "pi05-base"])
def test_official_base_routes_only_to_official_workflow(name, monkeypatch):
    from yamkit.openpi import workflow

    monkeypatch.setattr(inference_workflow, "load_target", lambda *_a, **_k: pytest.fail("no MA2 backend substitution"))
    calls = []
    sentinel = object()
    monkeypatch.setattr(workflow, "prepare", lambda **kwargs: calls.append(kwargs) or sentinel)
    result = inference_workflow._prepare_inference(backend="lambda", policy=name, task="cube", rig="unopened.yaml",
                                                 duration=60, arms=(), config=None, progress=lambda _: None, force=False)
    assert result is sentinel and len(calls) == 1
    assert calls[0]["task"] == "cube" and calls[0]["duration"] == 60 and calls[0]["rig"] == "unopened.yaml"


def test_failed_official_preparation_still_blocks_before_confirmation(monkeypatch, tmp_path):
    from yamkit import workflow_lock
    from yamkit.openpi import workflow

    monkeypatch.setattr(workflow_lock, "ROOT", tmp_path)
    monkeypatch.setattr(inference_workflow, "ROOT", tmp_path)

    def failed(**_kwargs):
        raise WorkflowError("Official OpenPI software qualification failed; no motion was started")

    monkeypatch.setattr(workflow, "prepare", failed)
    monkeypatch.setattr(workflow, "run_prepared", lambda *_a, **_kw: pytest.fail("no physical delegate after failure"))
    result = CliRunner().invoke(cli.app, ["rollout", "--backend", "lambda", "--policy", "pi05_base",
                                         "--task", "put the red cube into the black container", "--duration", "60"])
    assert result.exit_code == 2
    assert "Official OpenPI" in result.output and "qualification failed" in result.output
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
