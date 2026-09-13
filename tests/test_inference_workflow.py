"""Simple workflow remains software-only until a separately confirmed legacy rollout."""

import fcntl
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest
from typer.testing import CliRunner

from tests.test_external_ops import attach
from tests.test_external_ops import attachment as _attachment
from yamkit import backend_workflow as backend
from yamkit import cli
from yamkit import inference_workflow as workflow
from yamkit import workflow_lock as locks
from yamkit.backend_workflow import WorkflowError

attachment = _attachment


@pytest.fixture
def legacy_attachment(attachment, tmp_path, monkeypatch):
    """A genuinely validated saved attachment, with its readiness HTTP stubbed."""
    monkeypatch.setattr(backend, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(locks, "ROOT", tmp_path)
    receipt = attach(attachment)
    attachment.calls.clear()
    return SimpleNamespace(root=tmp_path, receipt=receipt, calls=attachment.calls)


def test_missing_default_configuration_retains_existing_attachment_compatibility(legacy_attachment):
    fixture = legacy_attachment
    target = backend.load_target("lambda", "molmoact2")
    assert target.service == fixture.receipt["name"]
    assert target.endpoint == fixture.receipt["http_endpoint"]
    assert target.policy == "molmoact2" and target.backend == "lambda"
    assert target.token_file is target.ssh is target.remote is None
    assert not fixture.calls and not (fixture.root / backend.CONFIG_RELATIVE).exists()


@pytest.mark.parametrize("filename", ["data/inference/typo.json", backend.CONFIG_RELATIVE])
def test_explicit_missing_configuration_never_searches_existing_attachment(legacy_attachment, monkeypatch, filename):
    from yamkit import external_ops

    fixture = legacy_attachment
    assert backend.load_target("lambda", "molmoact2").service == fixture.receipt["name"]
    monkeypatch.setattr(external_ops, "owned_service", lambda *_a: pytest.fail("An explicit path forbids fallback search"))
    with pytest.raises(WorkflowError, match="Explicit backend configuration file does not exist"):
        backend.load_target("lambda", "molmoact2", config=fixture.root / filename)
    assert not fixture.calls and not (fixture.root / filename).exists()


def test_cli_missing_explicit_config_is_actionable_before_backend_or_confirmation(legacy_attachment, monkeypatch):
    monkeypatch.setattr(workflow, "ensure_backend", lambda *_a, **_k: pytest.fail("No backend connection or startup"))
    fixture = legacy_attachment
    result = CliRunner().invoke(cli.app, ["rollout", "--backend", "lambda", "--policy", "molmoact2",
                                         "--task", "cube", "--rig", str(fixture.root / "unread-rig.yaml"),
                                         "--backend-config", str(fixture.root / "missing.json"), "--fake-hardware"])
    assert result.exit_code == 2
    assert "Explicit backend configuration file does not exist" in result.output
    assert "--backend-config" in result.output and "I am on site" not in result.output
    assert not fixture.calls and not (fixture.root / "outputs").exists()


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(locks, "ROOT", tmp_path)
    path = tmp_path / backend.CONFIG_RELATIVE
    path.parent.mkdir(parents=True)
    entry = {"service": "test-gpu", "endpoint": "http://127.0.0.1:8765", "token_file": "data/inference/test.token",
             "remote": {"repo": "/home/example/yamkit", "token_file": "data/inference/test.token", "region": "test-region"}}
    value = {"version": 1, "backends": {"lambda": {"ssh": {"host": "gpu-alias"}, "policies": {"molmoact2": entry}}}}

    def save():
        path.write_text(json.dumps(value))
        path.chmod(0o600)

    save()
    return SimpleNamespace(root=tmp_path, path=path, value=value, entry=entry, save=save)


def test_config_uses_standard_ssh_semantics_without_hardcoded_identity(configured):
    target = backend.load_target("lambda", "molmoact2")
    assert target.ssh == {"host": "gpu-alias"}
    assert target.remote["session_seconds"] == 28800
    args = backend.ssh_args(target)
    assert "StrictHostKeyChecking=yes" in args and "BatchMode=yes" in args
    assert "-F" not in args and "IdentitiesOnly=yes" not in args and "-i" not in args
    assert target.token_file == configured.root / "data/inference/test.token"


@pytest.mark.parametrize("field,value", [("token_file", "../secret"), ("endpoint", "http://example.com:8765"),
                                        ("service", "x; pwd"), ("extra", True)])
def test_invalid_config_is_rejected_without_connecting(configured, field, value):
    configured.entry[field] = value
    configured.save()
    with pytest.raises(ValueError):
        backend.load_target("lambda", "molmoact2")


def test_config_credentials_are_paths_only_and_file_private(configured):
    configured.path.chmod(0o644)
    with pytest.raises(WorkflowError, match="chmod 600"):
        backend.load_target("lambda", "molmoact2")


def test_pi05_alias_cannot_fall_back_to_base_or_molmo(configured):
    with pytest.raises((ValueError, OSError)):
        workflow.prepare_inference(backend="lambda", policy="pi05", task="test", rig=configured.root / "missing-rig.yaml")


def test_shell_characters_in_ssh_host_rejected(configured):
    configured.value["backends"]["lambda"]["ssh"]["host"] = "gpu; touch x"
    configured.save()
    with pytest.raises(WorkflowError, match="SSH host"):
        backend.load_target("lambda", "molmoact2")


def test_ui_active_and_wrong_listener_fail_without_any_changes(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, _): return json.dumps({"active": True, "cameras_owned": False}).encode()
    monkeypatch.setattr(backend, "urlopen", lambda *_a, **_k: Response())
    with pytest.raises(WorkflowError, match="UI session"):
        backend.assert_ui_idle()
    monkeypatch.setattr(Response, "read", lambda *_: b'{"not_yamkit":true}')
    with pytest.raises(WorkflowError, match="Port 8400"):
        backend.assert_ui_idle()
    monkeypatch.setattr(Response, "read", lambda *_: b"\xff")
    with pytest.raises(WorkflowError, match="Cannot verify"):
        backend.assert_ui_idle()


def test_terminal_only_ui_absence_allowed_but_timeout_not_allowed(monkeypatch):
    def refused(*_a, **_k): raise URLError(ConnectionRefusedError())
    monkeypatch.setattr(backend, "urlopen", refused)
    backend.assert_ui_idle()
    def timeout(*_a, **_k): raise TimeoutError()
    monkeypatch.setattr(backend, "urlopen", timeout)
    with pytest.raises(WorkflowError, match="Cannot verify"):
        backend.assert_ui_idle()


def test_direct_preview_blocks_cli_but_owned_active_preparation_can_continue(monkeypatch):
    directory = "/repo/.context/inference-preparation/example"
    state = {"active": False, "cameras_owned": False, "direct_cameras_open": ["top"],
             "mode": "inference-prepare", "pid": os.getpid(), "meta": {"preparation_dir": directory}}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, _): return json.dumps(state).encode()
    monkeypatch.setattr(backend, "urlopen", lambda *_a, **_kw: Response())
    backend.assert_ui_idle()  # Software-only inference does not need to close existing previews.
    with pytest.raises(WorkflowError, match="close Live/camera previews or use UI Start"):
        backend.assert_ui_idle(require_cameras_idle=True)
    with pytest.raises(WorkflowError, match="direct camera previews"):
        backend.assert_ui_idle(own_preparation_dir=directory, require_cameras_idle=True)  # stale PID cannot claim this exception
    state["active"] = True
    backend.assert_ui_idle(own_preparation_dir=directory, require_cameras_idle=True)


def test_ready_backend_keeps_receipt_and_does_not_start_or_attach(configured, monkeypatch):
    from yamkit import external_ops

    target = backend.load_target("lambda", "molmoact2")
    receipt = {"status": "ready", "metadata": {"instance_id": "instance"}}
    monkeypatch.setattr(backend, "assert_ui_idle", lambda: None)
    monkeypatch.setattr(external_ops, "owned_service", lambda *_: receipt)
    monkeypatch.setattr(external_ops, "http_credentials", lambda *_: {"endpoint_url": target.endpoint, "token": "private"})
    monkeypatch.setattr(external_ops, "_validated_metadata", lambda _a, _b, value, _c: value)
    monkeypatch.setattr(backend, "_probe_service", lambda *_: dict(receipt["metadata"]))
    monkeypatch.setattr(external_ops, "attach_service", lambda *_: pytest.fail("must reuse exact receipt"))
    monkeypatch.setattr(backend, "_ssh", lambda *_a, **_k: pytest.fail("must not start anything"))
    assert backend.ensure_backend(target, "task") is receipt


def test_changed_live_graph_metadata_is_refreshed_before_cached_proof(configured, monkeypatch):
    from yamkit import external_ops

    target = backend.load_target("lambda", "molmoact2")
    monkeypatch.setattr(backend, "assert_ui_idle", lambda: None)
    monkeypatch.setattr(external_ops, "owned_service", lambda *_: {"status": "ready", "metadata": {"instance_id": "instance", "task": "old"}})
    monkeypatch.setattr(external_ops, "http_credentials", lambda *_: {"endpoint_url": target.endpoint, "token": "private"})
    monkeypatch.setattr(external_ops, "_validated_metadata", lambda _a, _b, value, _c: value)
    monkeypatch.setattr(backend, "_probe_service", lambda *_: {"instance_id": "instance", "task": "new"})
    seen = []
    monkeypatch.setattr(external_ops, "update_ready", lambda *a, **kw: seen.append((a, kw)) or {"updated": True})
    assert backend.ensure_backend(target, "task") == {"updated": True}
    assert seen[0][0][1]["task"] == "new" and seen[0][1]["expected_instance_id"] == "instance"


def test_unknown_local_listener_never_starts_gpu_or_tunnel(configured, monkeypatch):
    from yamkit import external_ops

    target = backend.load_target("lambda", "molmoact2")
    monkeypatch.setattr(backend, "assert_ui_idle", lambda: None)
    monkeypatch.setattr(external_ops, "owned_service", lambda *_: None)
    monkeypatch.setattr(external_ops, "_read_private", lambda *_: "private")
    def failed(*_): raise RuntimeError("secret value from endpoint")
    monkeypatch.setattr(backend, "_probe_service", failed)
    monkeypatch.setattr(backend, "_listening", lambda *_: True)
    monkeypatch.setattr(backend, "_managed_forward_present", lambda *_: False)
    monkeypatch.setattr(backend, "_ssh", lambda *_a, **_kw: pytest.fail("unknown port is not permission to start"))
    with pytest.raises(WorkflowError, match="unrecognized listener") as caught:
        backend.ensure_backend(target, "task")
    assert "secret" not in str(caught.value)


def test_exact_production_defaults_do_not_alter_legacy_options():
    options = workflow.reference_options(policy="molmoact2", task="test", service="test", rig=Path("configs/rig.yaml"))
    assert options.backend == "external" and options.controller_mode == "reference"
    assert options.execution_mode == "cuda_graph10" and options.call_mode == "http"
    assert not options.supervised_confirmed and not options.mapping_accepted
    from yamkit.deployment import InferenceOptions
    assert InferenceOptions(policy="molmoact2").controller_mode == "async"


def test_reference_reuses_exact_ui_context_without_helper(configured, monkeypatch):
    from yamkit.config import RigConfig
    from yamkit.inference import qualification
    from yamkit.ui import server

    options = workflow.reference_options(policy="molmoact2", task="test", service="test")
    monkeypatch.setattr(workflow, "assert_ui_idle", lambda: None)
    monkeypatch.setattr(RigConfig, "load", lambda *_: object())
    monkeypatch.setattr(server, "_prompt_preparation_context", lambda *_: {"expires_at": time.time() + 3600})
    monkeypatch.setattr(qualification, "settings_from_rig", lambda *_: {})
    monkeypatch.setattr(qualification, "validate_qualification", lambda *_: {"created_unix_s": time.time(), "assessment": {"qualified": True}})
    result = workflow.prepare_reference(options)
    assert result["ready"] and result["reused"] and not result["hardware_tested"]
    assert not (configured.root / ".context/inference-preparation").exists()


def test_workflow_lock_readonly_probe_and_nested_ownership(configured):
    locks.assert_workflow_available()
    assert not (configured.root / ".context").exists()
    with locks.workflow_lock():
        with locks.workflow_lock():
            assert True
        with pytest.raises(WorkflowError, match="Another"):
            locks.assert_workflow_available()
    locks.assert_workflow_available()


def test_workflow_lock_blocks_independent_owner(configured):
    path = configured.root / ".context/inference-workflow.lock"
    path.parent.mkdir(parents=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(WorkflowError, match="Another"):
            locks.workflow_lock().__enter__()
    finally:
        os.close(descriptor)


def test_workflow_lock_rejects_symlink_before_any_directory_write(configured, tmp_path):
    other = tmp_path / "other-owned-fixture"
    other.mkdir()
    (configured.root / ".context").symlink_to(other, target_is_directory=True)
    with pytest.raises(WorkflowError, match="without symlinks"), locks.workflow_lock():
        pytest.fail("must reject symlink")
    assert list(other.iterdir()) == []


def test_ui_child_keeps_lock_after_parent_guard_exits(configured):
    import sys

    from yamkit.ui.sessions import SessionManager

    manager = SessionManager()
    code = "from pathlib import Path; import sys,time; from yamkit.workflow_lock import workflow_lock; " \
           "\nwith workflow_lock(root=Path(sys.argv[1])):\n print('inherited ownership',flush=True); time.sleep(1)"
    try:
        with locks.workflow_lock() as descriptor:
            manager.start("inference-prepare", [sys.executable, "-c", code, str(configured.root)],
                          inference_lock_fd=descriptor)
        with pytest.raises(WorkflowError, match="Another"):
            locks.assert_workflow_available()
        assert manager.wait(timeout=5) == 0
        assert "inherited ownership" in manager.log
        locks.assert_workflow_available()
    finally:
        manager.close()


def test_cli_inference_and_rollout_dry_run_never_resolve_hardware(configured, monkeypatch):
    calls = []
    def prepare(**kwargs):
        calls.append(kwargs)
        return workflow.reference_options(policy="molmoact2", task=kwargs["task"], service="test"), {"ready": True, "hardware_tested": False}
    monkeypatch.setattr(workflow, "prepare_inference", prepare)
    monkeypatch.setattr(cli, "_rig_arms", lambda *_: pytest.fail("no hardware setup in software mode"))
    runner = CliRunner()
    result = runner.invoke(cli.app, ["inference", "--backend", "lambda", "--policy", "molmoact2", "--task", "test"])
    assert result.exit_code == 0, result.output
    assert "No hardware" in result.output and len(calls) == 1
    result = runner.invoke(cli.app, ["rollout", "--backend", "lambda", "--policy", "molmoact2", "--task", "test", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "no rollout was launched" in result.output and len(calls) == 2


def test_simple_rollout_refuses_confirmation_or_nonreference_before_hardware(configured, monkeypatch):
    monkeypatch.setattr(workflow, "prepare_inference", lambda **kw: (
        workflow.reference_options(policy="molmoact2", task=kw["task"], service="test"), {"ready": True}))
    monkeypatch.setattr(cli, "_rig_arms", lambda *_: pytest.fail("no hardware without confirmation"))
    runner = CliRunner()
    args = ["rollout", "--backend", "lambda", "--policy", "molmoact2", "--task", "test"]
    result = runner.invoke(cli.app, args, input="n\n")
    assert result.exit_code != 0 and "secure mounts" in result.output
    result = runner.invoke(cli.app, [*args, "--controller-mode", "async"])
    assert result.exit_code == 2 and "--controller-mode" in result.output and "reference" in result.output
    assert "Pending one" not in result.output


def test_remote_bootstrap_has_no_takeover_and_task_is_quoted(configured, monkeypatch):
    target = backend.load_target("lambda", "molmoact2")
    seen = []
    monkeypatch.setattr(backend, "_ssh", lambda _t, argv: seen.append(argv) or '{"status":"started","pid":123}')
    assert backend._start_remote(target, "cube'; $(false)") == "started"
    assert seen[0][0] == "gpu-alias" and "StrictHostKeyChecking=no" not in str(seen)
    assert "os.kill" not in backend._REMOTE_BOOTSTRAP and "terminate" not in backend._REMOTE_BOOTSTRAP
    assert "pass_fds=(fd,)" in backend._REMOTE_BOOTSTRAP


def test_delayed_terminal_confirmation_rechecks_lease_before_any_hardware(configured, monkeypatch):
    monkeypatch.setattr(workflow, "prepare_inference", lambda **kw: (
        workflow.reference_options(policy="molmoact2", task=kw["task"], service="test"), {"ready": True}))
    monkeypatch.setattr(backend, "assert_ui_idle", lambda **_: None)
    def expired(_options):
        raise WorkflowError("The attached model session expires too soon")
    monkeypatch.setattr(workflow, "require_prepared_current", expired)
    monkeypatch.setattr(cli, "_rig_arms", lambda *_: pytest.fail("expired approval wait cannot open hardware"))
    result = CliRunner().invoke(cli.app, ["rollout", "--backend", "lambda", "--policy", "molmoact2",
                                         "--task", "test"], input="y\n")
    assert result.exit_code == 2 and "expires too soon" in result.output


@pytest.mark.parametrize("service_expiry,qualification_remaining,passes", [(2000, 1000, True), (1070, 1000, False), (2000, 70, False)])
def test_after_confirmation_margin_uses_both_service_and_qualification_expiry(
        monkeypatch, service_expiry, qualification_remaining, passes):
    from yamkit.config import RigConfig
    from yamkit.inference import qualification
    from yamkit.ui import server

    options = workflow.reference_options(policy="molmoact2", task="test", service="test", duration=60)
    monkeypatch.setattr(RigConfig, "load", lambda *_: object())
    monkeypatch.setattr(workflow.time, "time", lambda: 1000)
    monkeypatch.setattr(server, "_prompt_preparation_context", lambda *_: {"expires_at": service_expiry})
    monkeypatch.setattr(qualification, "settings_from_rig", lambda *_: {})
    monkeypatch.setattr(qualification, "validate_qualification", lambda *_: {
        "created_unix_s": 1000 - qualification.MAX_AGE_S + qualification_remaining})
    if passes:
        workflow.require_prepared_current(options)
    else:
        with pytest.raises(WorkflowError, match="expires too soon after confirmation"):
            workflow.require_prepared_current(options)


@pytest.mark.parametrize("failure", [OSError, KeyError, TypeError])
def test_after_confirmation_changed_metadata_is_actionable_without_private_details(monkeypatch, failure):
    from yamkit.config import RigConfig

    options = workflow.reference_options(policy="molmoact2", task="test", service="test")
    def changed(*_):
        raise failure("private local metadata details")
    monkeypatch.setattr(RigConfig, "load", changed)
    with pytest.raises(WorkflowError, match="changed after confirmation") as caught:
        workflow.require_prepared_current(options)
    assert "private local" not in str(caught.value)
