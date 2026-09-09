"""The bundle command reads finalized files only; no hardware or real Hub calls."""

import builtins
import json

import pytest
from typer.testing import CliRunner

from yamkit import cli


@pytest.fixture
def no_hardware(monkeypatch):
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith(("i2rt", "lerobot_robot_yamkit", "yamkit.arm")):
            raise AssertionError("Bundle command imported a hardware module")
        if name == "arm" and level and globals and globals.get("__package__") == "yamkit":
            raise AssertionError("Bundle command imported the arm wrapper")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_bundle_cli_packages_finalized_files_without_network(tmp_path, monkeypatch, no_hardware):
    from yamkit import rollout_artifacts

    run = tmp_path / "20260101-120000-rollout-fixture"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({
        "id": run.name, "kind": "rollout", "started_at": 1, "ended_at": 2,
        "returncode": 0, "status": "success",
    }))
    (run / "summary.json").write_text('{"resources_released":true}')
    (run / "log.txt").write_text("fixture run completed\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("Local bundling contacted the Hub")

    monkeypatch.setattr(rollout_artifacts, "upload_rollout", forbidden)
    result = CliRunner().invoke(cli.app, ["bundle-rollout", str(run)])
    assert result.exit_code == 0, result.output
    assert (run / "bundle" / "manifest.json").is_file()
    assert (run / "bundle" / "README.md").is_file()
    assert (run / "log.txt").read_text() == "fixture run completed\n"


def test_bundle_cli_refuses_active_run_before_upload(tmp_path, monkeypatch, no_hardware):
    from yamkit import rollout_artifacts

    run = tmp_path / "20260101-120000-rollout-live"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({
        "id": run.name, "kind": "rollout", "status": "running", "returncode": None,
    }))
    seen = []
    monkeypatch.setattr(rollout_artifacts, "upload_rollout", lambda *a, **kw: seen.append(kw))
    result = CliRunner().invoke(cli.app, ["bundle-rollout", str(run), "--upload-to", "fixture/private"])
    assert result.exit_code == 1
    assert not seen
    assert not (run / "bundle").exists()


def test_bundle_cli_upload_failure_does_not_echo_credentials(tmp_path, monkeypatch, no_hardware):
    from yamkit import rollout_artifacts

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "README.md").write_text("Retained artifact fixture")
    monkeypatch.setattr(rollout_artifacts, "package_rollout", lambda *a, **kw: bundle)

    def fail(*args, **kwargs):
        raise RuntimeError("Authorization: Bearer hf_SYNTHETIC_SECRET_MUST_NOT_ECHO")

    monkeypatch.setattr(rollout_artifacts, "upload_rollout", fail)
    result = CliRunner().invoke(cli.app, ["bundle-rollout", str(tmp_path), "--upload-to", "fixture/private"])
    assert result.exit_code == 1
    assert "SYNTHETIC_SECRET" not in result.output
    assert "RuntimeError" in result.output
    assert (bundle / "README.md").is_file()
