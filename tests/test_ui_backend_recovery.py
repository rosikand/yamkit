"""Explicit UI Start can recover configured software, but never starts hardware implicitly."""

import dataclasses
import json
import os
import time
from contextlib import contextmanager

import pytest

from tests.test_ui_prompt_preparation import (
    NAME,
    NEW_TASK,
    body,
    fake_child,
    helper,
)
from tests.test_ui_prompt_preparation import (
    attachment as _attachment,
)
from tests.test_ui_prompt_preparation import (
    inference_ui as _inference_ui,
)
from tests.test_ui_prompt_preparation import (
    prompt_ui as _prompt_ui,
)
from yamkit import backend_workflow as backend
from yamkit import external_ops, modal_qualification
from yamkit import inference_workflow as workflow
from yamkit import workflow_lock as locks
from yamkit.deployment import InferenceOptions
from yamkit.ui import server

attachment = _attachment
inference_ui = _inference_ui
prompt_ui = _prompt_ui


@pytest.fixture
def recovery_ui(prompt_ui, monkeypatch):
    ui = prompt_ui
    monkeypatch.setattr(backend, "ROOT", ui.root)
    monkeypatch.setattr(workflow, "ROOT", ui.root)
    path = ui.root / backend.CONFIG_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "backends": {"lambda": {"policies": {"molmoact2": {
        "service": NAME, "endpoint": "http://127.0.0.1:8765", "token_file": "token"}}}}}))
    path.chmod(0o600)
    receipt = external_ops.owned_service(NAME)
    receipt["http_session_expires_at"] = receipt["metadata"]["http_session_expires_at"] = time.time() - 1
    external_ops._save(external_ops._directory(NAME) / "receipt.json", receipt)
    return ui


def test_expired_configured_ui_preflight_is_local_recoverable_not_ready(recovery_ui, monkeypatch):
    monkeypatch.setattr(backend, "ensure_backend", lambda *_a, **_kw: pytest.fail("preflight must not contact GPU"))
    result = recovery_ui.client.post("/api/inference/preflight", json=body(NEW_TASK)).json()
    assert not result["ready"] and result["can_prepare"]
    assert not recovery_ui.manager.active and not recovery_ui.seen


def test_expired_service_start_only_creates_managed_nonhardware_recovery_child(recovery_ui):
    fake_child(recovery_ui, "import time; print('[yamkit-prepare] connecting',flush=True); time.sleep(30)")
    result = recovery_ui.client.post("/api/inference/prepare", json=body(NEW_TASK))
    assert result.status_code == 200, result.text
    state = recovery_ui.client.get("/api/session").json()
    assert state["mode"] == "inference-prepare" and state["active"] and not state["cameras_owned"]
    request = json.loads((__import__("pathlib").Path(state["meta"]["preparation_dir"]) / "request.json").read_text())
    assert request["expected"]["recovery"] and not request["options"]["supervised_confirmed"]
    assert not recovery_ui.seen


def test_recovery_never_bypasses_mismatched_camera_shape(recovery_ui):
    recovery_ui.rig.cameras["top"]["width"] = 320
    recovery_ui.rig.save()
    result = recovery_ui.client.post("/api/inference/preflight", json=body(NEW_TASK)).json()
    assert not result["ready"] and not result["can_prepare"]
    assert recovery_ui.client.post("/api/inference/prepare", json=body(NEW_TASK)).status_code == 422
    assert not recovery_ui.manager.active


def test_recovery_helper_refreshes_before_qualification_and_preserves_expired_receipt(recovery_ui, monkeypatch):
    options = InferenceOptions(**body(NEW_TASK), rig_path=str(recovery_ui.rig.path))
    directory = recovery_ui.root / ".context/inference-preparation" / ("a" * 32)
    directory.mkdir(parents=True)
    request = directory / "request.json"
    request.write_text(json.dumps({"options": dataclasses.asdict(options), "capture_trace": False,
                                   "expected": server._preparation_selection_context(options, recovery_ui.rig)}))
    events = []

    def recover(_options, _rig, _expected, **kwargs):
        assert kwargs["directory"] == directory
        old = json.loads((directory / "previous-attachment.json").read_text())
        assert old["http_session_expires_at"] < time.time()
        receipt = external_ops.owned_service(NAME)
        receipt["http_session_expires_at"] = receipt["metadata"]["http_session_expires_at"] = time.time() + 3600
        external_ops._save(external_ops._directory(NAME) / "receipt.json", receipt)
        monkeypatch.setattr(external_ops, "_probe_ready", lambda *_: receipt["metadata"])
        events.append("recovered")

    def collect(*_args, **_kwargs):
        assert events == ["recovered"]
        events.append("qualified")
        from tests.test_http_runtime_binding import runtime_metadata

        receipt = external_ops.owned_service(NAME)
        receipt["metadata"]["graph_warmup"] = runtime_metadata(task=NEW_TASK, image_hw=(480, 640))["graph_warmup"]
        external_ops.update_ready(NAME, receipt["metadata"], expected_instance_id=receipt["metadata"]["instance_id"])
        return {"hardware_tested": False, "assessment": {"qualified": True}}

    monkeypatch.setattr(workflow, "recover_reference_backend", recover)
    monkeypatch.setattr(modal_qualification, "collect_qualification", collect)
    monkeypatch.setattr(external_ops, "_probe_ready", lambda *_: external_ops.owned_service(NAME)["metadata"])
    assert helper.execute(request) == 0
    assert events == ["recovered", "qualified"]
    result = json.loads((directory / "result.json").read_text())
    assert result["ready"] and not result["hardware_tested"]
    assert not recovery_ui.manager.active and not recovery_ui.seen


def test_self_session_exception_requires_pid_mode_directory_and_no_cameras(monkeypatch):
    directory = "/repo/.context/inference-preparation/example"
    state = {"active": True, "mode": "inference-prepare", "pid": os.getpid(), "cameras_owned": False,
             "meta": {"preparation_dir": directory}}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, _): return json.dumps(state).encode()
    monkeypatch.setattr(backend, "urlopen", lambda *_a, **_kw: Response())
    backend.assert_ui_idle(own_preparation_dir=directory)
    for field, value in (("pid", -1), ("mode", "rollout"), ("cameras_owned", True)):
        previous = state[field]
        state[field] = value
        with pytest.raises(backend.WorkflowError, match="UI session"):
            backend.assert_ui_idle(own_preparation_dir=directory)
        state[field] = previous
    with pytest.raises(backend.WorkflowError, match="UI session"):
        backend.assert_ui_idle(own_preparation_dir="/different")


def test_unreadable_lock_fails_actionably_without_child(prompt_ui, monkeypatch):
    @contextmanager
    def denied(**_kwargs):
        raise PermissionError("private filesystem details")
        yield
    monkeypatch.setattr(locks, "workflow_lock", denied)
    response = prompt_ui.client.post("/api/inference/prepare", json=body(NEW_TASK))
    assert response.status_code == 409 and "lock is unavailable" in response.text
    assert "private filesystem" not in response.text and not prompt_ui.manager.active


def test_cross_process_lock_blocks_ui_preflight_and_start(prompt_ui):
    # An independent file description models another CLI process, not context reentrancy.
    import fcntl
    path = prompt_ui.root / ".context/inference-workflow.lock"
    path.parent.mkdir(exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = prompt_ui.client.post("/api/inference/preflight", json=body(NEW_TASK)).json()
        assert not result["ready"] and not result["can_prepare"] and "Another" in result["reason"]
        response = prompt_ui.client.post("/api/inference/prepare", json=body(NEW_TASK))
        assert response.status_code == 409 and not prompt_ui.manager.active
    finally:
        os.close(descriptor)
