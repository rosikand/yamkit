"""Camera ownership and authenticated proxy checks using pipes/fake sources, never hardware."""

from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from yamkit.camera_ownership import claim_from_env
from yamkit.ui import camstream, preview_proxy
from yamkit.ui.camstream import CameraHub
from yamkit.ui.sessions import PreviewRegistration, SessionManager


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def child_manager(body, **kwargs):
    manager = SessionManager(**kwargs)
    manager.start("unexpected-command-name", [sys.executable, "-u", "-c", body])
    return manager


CLAIM = "from yamkit.camera_ownership import claim_from_env; lease=claim_from_env(['top']); "
ANNOUNCE = (
    "import json; print('@yamkit-preview/1 '+json.dumps({'v':1,'session':lease.session,"
    "'owner':lease.owner,'port':12345,'cameras':['top']}), flush=True); "
)


def test_claim_is_optional_and_timeout_fails_closed():
    output = io.StringIO()
    assert claim_from_env(["top"], environ={}, output=output).owner == ""
    assert claim_from_env([], environ={"YAMKIT_PREVIEW_SESSION": "s", "YAMKIT_PREVIEW_TOKEN": "t"}).owner == ""
    read, write = os.pipe()
    try:
        with os.fdopen(read) as source, pytest.raises(RuntimeError, match="timed out"):
            claim_from_env(
                ["top"], environ={"YAMKIT_PREVIEW_SESSION": "s", "YAMKIT_PREVIEW_TOKEN": "secret"},
                input=source, output=output, timeout=0.02,
            )
    finally:
        os.close(write)
    assert "@yamkit-cameras/1 " in output.getvalue()
    assert "secret" not in output.getvalue()


def test_ownership_waits_for_release_confirmation_then_releases_during_upload():
    entered, allow = threading.Event(), threading.Event()
    callbacks = []

    def acquire(owner):
        entered.set()
        assert allow.wait(3)
        callbacks.append(("acquire", owner))
        return True

    manager = child_manager(
        CLAIM + ANNOUNCE + "print('CAMERA_OPEN'); import time; time.sleep(.15); "
        "lease.release(); lease.release(); print('[yamkit] recording finished upload'); time.sleep(30)",
        on_camera_acquire=acquire, on_camera_release=lambda owner: callbacks.append(("release", owner)),
    )
    try:
        assert entered.wait(3)
        assert not any("CAMERA_OPEN" == line for line in manager.log)
        allow.set()
        wait_for(lambda: manager.preview_registration() is not None)
        reg = manager.preview_registration("top")
        assert reg is not None and reg.cameras == ("top",)
        assert manager.preview_registration("unknown") is None
        assert manager.preview_is_current(reg)
        assert reg.token not in repr(reg)
        wait_for(lambda: manager.parsed.get("phase") == "upload")
        assert manager.active and not manager.cameras_owned
        assert not manager.preview_is_current(reg)
        assert manager.preview_registration() is None
        assert callbacks == [("acquire", reg.owner), ("release", reg.owner)]
        assert reg.token not in json.dumps(manager.status())
        assert not any(line.startswith("@yamkit-") for line in manager.log)
    finally:
        allow.set()
        manager.stop(grace_s=.1)
        manager.wait(5)


def test_denied_acquisition_never_opens_cameras():
    callbacks = []
    manager = child_manager(
        CLAIM + "print('CAMERA_OPEN')",
        on_camera_acquire=lambda owner: False,
        on_camera_release=callbacks.append,
    )
    assert manager.wait(5) != 0
    assert "CAMERA_OPEN" not in manager.log
    assert any("acquisition denied" in line for line in manager.log)
    assert len(callbacks) == 1
    assert not manager.cameras_owned


def test_no_command_name_ownership_and_popen_failure(monkeypatch):
    acquired = []
    manager = SessionManager(on_camera_acquire=acquired.append)
    manager.start("record", [sys.executable, "-c", "print('not opening cameras')"])
    assert manager.wait(5) == 0
    assert acquired == [] and not manager.cameras_owned

    def fail(*args, **kwargs):
        raise OSError("cannot start")

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(OSError, match="cannot start"):
        manager.start("record", ["missing"])
    assert not manager.active and not manager.cameras_owned
    assert manager.preview_registration() is None


def test_registration_requires_current_owner_and_is_accepted_once():
    class Proc:
        stdin = io.StringIO()

        def poll(self):
            return None

    manager = SessionManager()
    manager._proc = proc = Proc()
    manager._session, manager._token = "session", "secret"

    def message(prefix="@yamkit-preview/1 ", **kwargs):
        body = {"v": 1, "session": "session", "owner": "owner", "port": 1234, "cameras": ["top"]}
        body.update(kwargs)
        manager._control_line(prefix + json.dumps(body), proc)

    message()
    assert manager.preview_registration() is None
    message("@yamkit-cameras/1 ", event="acquire")
    assert json.loads(proc.stdin.getvalue())["ok"] is True
    for change in ({"session": "old"}, {"owner": "old"}, {"port": True}, {"port": 65536},
                   {"port": "http://example.com"}, {"cameras": ["unknown"]}, {"cameras": ["../top"]}):
        message(**change)
        assert manager.preview_registration() is None
    message()
    reg = manager.preview_registration()
    assert reg is not None and reg.port == 1234
    message(port=4321)
    assert manager.preview_registration() is reg
    message("@yamkit-cameras/1 ", event="release", owner="stale")
    assert manager.preview_registration() is reg
    message("@yamkit-cameras/1 ", event="release")
    assert not manager.preview_is_current(reg)
    message()
    assert manager.preview_registration() is None
    message("@yamkit-cameras/1 ", event="acquire", owner="next")
    message(owner="owner")
    assert manager.preview_registration() is None
    message(owner="next")
    assert manager.preview_registration().owner == "next"


def test_rapid_sessions_invalidate_registration_and_redact_token():
    manager = SessionManager()
    previous = None
    for _ in range(3):
        manager.start("teleop", [sys.executable, "-u", "-c", CLAIM + ANNOUNCE
                      + "import os,time; print(os.environ['YAMKIT_PREVIEW_TOKEN']); time.sleep(30)"])
        try:
            wait_for(lambda: manager.preview_registration() is not None)
            reg = manager.preview_registration()
            if previous:
                assert not manager.preview_is_current(previous)
                assert reg.session != previous.session and reg.token != previous.token
            wait_for(lambda: "[redacted]" in manager.log)
            assert reg.token not in json.dumps(manager.status())
        finally:
            manager.stop(grace_s=.1)
            manager.wait(5)
        assert not manager.active and not manager.cameras_owned
        previous = reg


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="Linux process group/zombie inspection")
def test_launcher_crash_holds_ownership_until_descendant_exits(tmp_path):
    ready = tmp_path / "descendant-ready"
    child = (
        "import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"pathlib.Path({str(ready)!r}).write_text('ready'); time.sleep(30)"
    )
    body = (
        CLAIM + "import subprocess,sys,time,os,pathlib; "
        f"p=subprocess.Popen([sys.executable,'-u','-c',{child!r}],stdout=subprocess.DEVNULL); "
        "print('DESCENDANT '+str(p.pid),flush=True); "
        f"ready=pathlib.Path({str(ready)!r}); "
        "\nwhile not ready.exists(): time.sleep(.01)\nos._exit(4)"
    )
    released = []
    manager = child_manager(body, on_camera_release=lambda owner: released.append(time.monotonic()))
    try:
        wait_for(lambda: ready.exists())
        wait_for(lambda: manager._proc.poll() is not None)
        assert manager.cameras_owned and manager.active and released == []
        with pytest.raises(RuntimeError, match="already running"):
            manager.start("record", [sys.executable, "-c", "pass"])
        assert manager.wait(5) == 4
        wait_for(lambda: not manager.active)
        assert len(released) == 1 and not manager.cameras_owned
    finally:
        manager._signal(manager._proc, signal.SIGKILL)
        manager.wait(5)


def test_hub_blocks_delayed_generators_and_stale_resume(monkeypatch):
    hub = CameraHub({"top": {"index_or_path": "fake"}})
    cam = hub.get("top")
    opened = []
    monkeypatch.setattr(cam, "_loop", lambda: opened.append(True))
    delayed = cam.frames(threading.Event())
    assert hub.suspend("owner")
    assert list(delayed) == []
    assert not cam.ensure_running() and opened == []
    assert not hub.resume("old")
    assert not cam.ensure_running()
    hub.reload({"top": {"index_or_path": "fake"}, "new": {"index_or_path": "fake2"}})
    assert not hub.get("new").ensure_running()
    assert hub.resume("owner")
    assert cam.ensure_running()
    cam.stop(join=True)
    assert opened == [True]


def test_hub_refuses_acquire_when_capture_join_times_out(monkeypatch):
    hub = CameraHub({"top": {"index_or_path": "fake"}})
    cam = hub.get("top")
    held = threading.Event()
    monkeypatch.setattr(camstream, "STOP_JOIN_S", .02)
    monkeypatch.setattr(cam, "_loop", lambda: held.wait(3))
    cam.ensure_running()
    try:
        assert not hub.suspend("owner")
        assert hub.suspended_by == "owner" and not cam.ensure_running()
    finally:
        held.set()
        cam.stop(join=True)
        hub.resume("owner")


def test_closed_hub_cannot_resume_from_delayed_release():
    hub = CameraHub({"top": {"index_or_path": "fake"}})
    assert hub.suspend("owner")
    assert hub.close()
    assert not hub.resume("owner")
    assert not hub.get("top").ensure_running()


@pytest.fixture
def endpoint():
    state = {"code": 200, "payload": {}, "requests": [], "stream": b"--yamkitframe\r\n"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append((self.path, self.headers.get(preview_proxy.TOKEN_HEADER)))
            code = state["code"]
            self.send_response(code)
            if code == 302:
                self.send_header("Location", "http://example.invalid/not-allowed")
            if self.path == "/status":
                body = json.dumps(state["payload"]).encode()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(state.get("length", len(body))))
            else:
                body = state["stream"]
                self.send_header("Content-Type", preview_proxy.MJPEG_MEDIA_TYPE)
            self.end_headers()
            try:
                if state.get("drip"):
                    for byte in body:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(.01)
                else:
                    self.wfile.write(body)
            except OSError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    worker.start()
    reg = PreviewRegistration("session", "owner", server.server_port, ("top",), "secret")
    state["payload"] = {"v": 1, "session": reg.session, "cameras": {"top": {"state": "live", "age_s": .1}}}
    yield reg, state
    server.shutdown()
    server.server_close()
    worker.join(1)


def test_proxy_auth_status_limits_and_redirect_rejection(endpoint):
    reg, state = endpoint
    current = lambda candidate: candidate is reg
    status = preview_proxy.fetch_status(reg, current)
    assert status["top"]["state"] == "live" and status["top"]["age_s"] >= .1
    assert state["requests"] == [("/status", "secret")]
    state["payload"]["cameras"]["unknown"] = {"token": "secret"}
    state["payload"]["cameras"]["top"]["token"] = "secret"
    assert "secret" not in json.dumps(preview_proxy.fetch_status(reg, current))
    state["code"] = 302
    with pytest.raises(preview_proxy.PreviewUnavailable):
        preview_proxy.fetch_status(reg, current)
    assert all(path == "/status" for path, _ in state["requests"])
    state["code"], state["length"] = 200, preview_proxy.MAX_STATUS_BYTES + 1
    with pytest.raises(preview_proxy.PreviewUnavailable, match="size"):
        preview_proxy.fetch_status(reg, current)
    state.pop("length")
    state["payload"]["session"] = "stale"
    with pytest.raises(preview_proxy.PreviewUnavailable, match="session"):
        preview_proxy.fetch_status(reg, current)


def test_proxy_stream_is_bounded_unknown_and_stale_rejected(endpoint, monkeypatch):
    reg, state = endpoint
    active = True
    current = lambda candidate: active and candidate is reg
    monkeypatch.setattr(preview_proxy, "_stream_permits", threading.BoundedSemaphore(1))
    with pytest.raises(preview_proxy.PreviewUnavailable, match="unknown"):
        preview_proxy.open_stream(reg, "http://elsewhere", current)
    stream = preview_proxy.open_stream(reg, "top", current)
    try:
        with pytest.raises(preview_proxy.PreviewUnavailable, match="too many"):
            preview_proxy.open_stream(reg, "top", current)
        assert next(stream) == state["stream"]
        active = False
        assert list(stream) == []
    finally:
        stream.close()
        stream.close()
    active = True
    stream = preview_proxy.open_stream(reg, "top", current)
    stream.close()
    assert all(path == "/cameras/top/stream" and token == "secret" for path, token in state["requests"])


def test_status_slow_drip_is_bounded_by_total_deadline(endpoint, monkeypatch):
    reg, state = endpoint
    state["drip"] = True
    monkeypatch.setattr(preview_proxy, "REQUEST_TIMEOUT_S", .06)
    started = time.monotonic()
    with pytest.raises(preview_proxy.PreviewUnavailable):
        preview_proxy.fetch_status(reg, lambda candidate: candidate is reg)
    assert time.monotonic() - started < .5


def test_malformed_status_values_are_contained(endpoint):
    reg, state = endpoint
    state["payload"]["cameras"]["top"] = {"state": {"bad": True}, "seq": 10 ** 500, "age_s": float("nan")}
    status = preview_proxy.fetch_status(reg, lambda candidate: candidate is reg)
    assert status == {"top": {"state": "unavailable", "age_s": None}}


def test_downstream_send_timeout_releases_proxy_permit(endpoint, monkeypatch):
    reg, _state = endpoint
    monkeypatch.setattr(preview_proxy, "_stream_permits", threading.BoundedSemaphore(1))
    monkeypatch.setattr(preview_proxy, "SEND_TIMEOUT_S", .02)
    stream = preview_proxy.open_stream(reg, "top", lambda candidate: candidate is reg)
    response = preview_proxy.PreviewStreamingResponse(stream)

    async def blocked_send(message):
        await asyncio.sleep(30)

    asyncio.run(response.stream_response(blocked_send))
    assert stream._closed
    next_stream = preview_proxy.open_stream(reg, "top", lambda candidate: candidate is reg)
    next_stream.close()
