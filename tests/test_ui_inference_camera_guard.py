"""Direct-preview/terminal ownership races with inert capture threads only."""

import fcntl
import os
import threading
from contextlib import contextmanager

import pytest

from tests.test_inference_ui import inference_ui as _inference_ui
from yamkit import workflow_lock as locks
from yamkit.backend_workflow import WorkflowError
from yamkit.ui import server
from yamkit.ui.camstream import CameraHub, _Camera
from yamkit.ui.preview_proxy import PreviewStreamingResponse

inference_ui = _inference_ui
_real_ensure_running = _Camera.ensure_running


@contextmanager
def independent_workflow(root):
    """Separate file description, exactly like an unrelated CLI process."""
    path = root / ".context/inference-workflow.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


@pytest.fixture
def inert_camera(inference_ui, monkeypatch):
    opened = []

    def loop(camera):
        opened.append(camera.name)
        camera._stop.wait(5)  # No OpenCV, hardware modules or real device opens.

    monkeypatch.setattr(_Camera, "_loop", loop)
    monkeypatch.setattr(_Camera, "ensure_running", _real_ensure_running)
    route = next(route for route in inference_ui.client.app.routes if route.path == "/api/cameras/{name}/stream")
    response = route.endpoint("top")
    camera = response._camera
    try:
        yield inference_ui, camera, response, opened
    finally:
        camera.stop(join=True)


def test_session_status_passively_exposes_direct_preview_without_claiming_robot(inert_camera):
    ui, camera, _response, _opened = inert_camera
    assert ui.client.get("/api/session").json()["direct_cameras_open"] == []
    assert camera.ensure_running()
    state = ui.client.get("/api/session").json()
    assert state["direct_cameras_open"] == ["top"]
    assert not state["active"] and not state["cameras_owned"]
    assert not ui.seen
    camera.stop(join=True)
    assert ui.client.get("/api/session").json()["direct_cameras_open"] == []


def test_response_created_before_cli_lock_cannot_open_on_late_subscription(inert_camera):
    ui, camera, response, opened = inert_camera
    with independent_workflow(ui.root):
        assert list(response._frames) == []
        assert not camera.running and opened == []
        blocked = ui.client.get("/api/cameras/top/stream")
        assert blocked.status_code == 409
    assert "inference operation" in camera.error


def test_actual_capture_thread_start_is_inside_workflow_guard(inert_camera, monkeypatch):
    ui, camera, _response, _opened = inert_camera
    start = threading.Thread.start
    checks = []

    def checked_start(thread):
        if thread.name == "cam-top":
            with pytest.raises(WorkflowError, match="Another"):
                locks.assert_workflow_available(root=ui.root)
            checks.append(True)
        return start(thread)

    monkeypatch.setattr(threading.Thread, "start", checked_start)
    assert camera.ensure_running() and checks == [True]
    # The CLI acquires only after the thread is visible to its passive status check.
    with independent_workflow(ui.root):
        assert ui.client.get("/api/session").json()["direct_cameras_open"] == ["top"]


def test_existing_preview_is_not_stopped_by_software_check_and_ui_handoff_still_releases(inert_camera):
    ui, camera, _response, opened = inert_camera
    assert camera.ensure_running()
    with independent_workflow(ui.root):
        assert camera.running  # The guard never terminates a user's already-running preview.
        assert camera.stop(join=True, disable=True)  # Simulate existing UI ownership acknowledgement.
        assert not camera.ensure_running()
        camera.allow()
        assert not camera.ensure_running()  # Child still owns inference, no competing direct restart.
    assert camera.ensure_running()
    assert len(opened) == 2


def test_reload_keeps_guard_for_new_cameras(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(_Camera, "_loop", lambda _: pytest.fail("guard must reject before capture"))
    hub = CameraHub({}, start_guard=server._direct_preview_start_guard)
    hub.reload({"new": {"index_or_path": "fake"}})
    with independent_workflow(tmp_path):
        assert not hub.get("new").ensure_running()
    hub.close()


def test_ui_owned_preview_proxy_bypasses_direct_capture_fence(inert_camera, monkeypatch):
    ui, camera, _response, opened = inert_camera
    registration = object()
    calls = []
    monkeypatch.setattr(ui.manager, "preview_registration", lambda name: registration)
    monkeypatch.setattr(server, "open_stream", lambda reg, name, current: calls.append((reg, name)) or iter(()))
    assert ui.manager.on_camera_acquire("fake-owned-child")
    try:
        route = next(route for route in ui.client.app.routes if route.path == "/api/cameras/{name}/stream")
        with independent_workflow(ui.root):
            response = route.endpoint("top")
            assert isinstance(response, PreviewStreamingResponse)
            assert calls == [(registration, "top")]
            assert not camera.running and opened == []
    finally:
        ui.manager.on_camera_release("fake-owned-child")
