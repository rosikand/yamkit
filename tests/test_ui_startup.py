"""A duplicate UI launch must never reconcile the running dashboard's saved jobs."""

import errno
import http.client
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn

from yamkit.ui import server


def forbid_app(monkeypatch):
    monkeypatch.setattr(server, "create_app", lambda *_: pytest.fail("Occupied port must be detected before app construction"))


@pytest.mark.parametrize("is_dashboard", [False, True])
def test_occupied_port_preserves_existing_listener_and_pending_receipts(tmp_path, monkeypatch, capsys, is_dashboard):
    run_dir = tmp_path / "outputs/ui/deployments/current-run"
    run_dir.mkdir(parents=True)
    receipts = {"hf-upload.json": b'{"status":"uploading"}\n',
                "rollout-progress.json": b'{"phase":"packaging","outcome":"completed"}\n'}
    for name, content in receipts.items():
        (run_dir / name).write_bytes(content)
    before = {name: (path.read_bytes(), path.stat().st_mtime_ns)
              for name in receipts if (path := run_dir / name).is_file()}
    forbid_app(monkeypatch)
    probes = []
    monkeypatch.setattr(server, "_existing_dashboard", lambda host, port: probes.append((host, port)) or is_dashboard)
    with socket.create_server(("127.0.0.1", 0)) as existing:
        port = existing.getsockname()[1]
        if is_dashboard:
            server.run(tmp_path / "rig.yaml", port=port)
            assert "already running" in capsys.readouterr().out
        else:
            with pytest.raises(SystemExit, match="already in use.*left untouched"):
                server.run(tmp_path / "rig.yaml", port=port)
        assert existing.fileno() >= 0 and existing.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
        assert probes == [("127.0.0.1", port)]
        assert before == {name: ((run_dir / name).read_bytes(), (run_dir / name).stat().st_mtime_ns) for name in receipts}


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0", "::1"])
def test_listening_socket_is_reserved_before_app_and_passed_to_server(monkeypatch, host):
    captured = []
    original_create = socket.create_server

    def create(*args, **kwargs):
        try:
            listener = original_create(*args, **kwargs)
        except OSError as exc:
            if host == "::1" and exc.errno in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                pytest.skip("IPv6 loopback unavailable")
            raise
        captured.append(listener)
        return listener

    def app(rig):
        assert len(captured) == 1 and captured[0].getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
        assert captured[0].getsockname()[1] != 0
        with pytest.raises(OSError) as error:
            original_create((host, captured[0].getsockname()[1]), family=captured[0].family)
        assert error.value.errno == errno.EADDRINUSE
        return "fixture-app"

    def config(value, **kwargs):
        assert value == "fixture-app" and kwargs["host"] == host
        assert kwargs["port"] == captured[0].getsockname()[1]
        return kwargs

    def run(*, sockets):
        assert sockets == captured and sockets[0].fileno() >= 0

    monkeypatch.setattr(socket, "create_server", create)
    monkeypatch.setattr(server, "create_app", app)
    monkeypatch.setattr(uvicorn, "Config", config)
    monkeypatch.setattr(uvicorn, "Server", lambda _config: SimpleNamespace(run=run, started=True))
    server.run(host=host, port=0)
    assert captured[0].fileno() == -1


@pytest.mark.parametrize("failure", ["app", "config", "run", "startup", "interrupt"])
def test_reserved_listener_closes_on_every_startup_failure(monkeypatch, failure):
    captured = []
    original_create = socket.create_server

    def create(*args, **kwargs):
        listener = original_create(*args, **kwargs)
        captured.append(listener)
        return listener

    def step(name):
        if failure == name:
            raise RuntimeError("fake startup failure")
        return object()

    def run(**kwargs):
        if failure == "interrupt":
            raise KeyboardInterrupt
        step("run")

    monkeypatch.setattr(socket, "create_server", create)
    monkeypatch.setattr(server, "create_app", lambda *_: step("app"))
    monkeypatch.setattr(uvicorn, "Config", lambda *_args, **_kwargs: step("config"))
    monkeypatch.setattr(uvicorn, "Server", lambda _config: SimpleNamespace(run=run, started=failure != "startup"))
    if failure == "interrupt":
        server.run(port=0)
    else:
        with pytest.raises(SystemExit if failure == "startup" else RuntimeError):
            server.run(port=0)
    assert captured[0].fileno() == -1


def test_other_bind_error_does_not_probe_or_construct_application(monkeypatch):
    forbid_app(monkeypatch)

    def denied(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(socket, "create_server", denied)
    monkeypatch.setattr(server, "_existing_dashboard", lambda *_: pytest.fail("Not an occupied-port failure"))
    with pytest.raises(SystemExit, match="Cannot bind yamkit UI.*Permission denied"):
        server.run()


@pytest.mark.parametrize("body,identified", [
    (b'<title>yamkit</title> robot console <script src="app.js"></script>', True),
    (b'<title>yamkit</title> robot console <script src="app.js?v=1789300123"></script>', True),
    (b'<title>another service</title>', False),
    (b'<title>yamkit</title>', False),
])
def test_existing_ui_probe_only_requests_static_shell(body, identified):
    requests = []
    with socket.create_server(("127.0.0.1", 0)) as existing:
        existing.settimeout(2)

        def reply():
            connection, _address = existing.accept()
            with connection:
                connection.settimeout(2)
                requests.append(connection.recv(4096))
                connection.sendall(b'HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n' + body)

        worker = threading.Thread(target=reply, daemon=True)
        worker.start()
        assert server._existing_dashboard("127.0.0.1", existing.getsockname()[1]) is identified
        worker.join(timeout=2)
    assert len(requests) == 1 and requests[0].startswith(b"GET / HTTP/1.1\r\n")
    assert b"/api/" not in requests[0]


def test_real_prebound_uvicorn_serves_only_a_trivial_asgi_fixture(monkeypatch):
    captured, running, requests, errors = [], [], [], []
    original_create, original_server = socket.create_server, uvicorn.Server

    def create(*args, **kwargs):
        listener = original_create(*args, **kwargs)
        captured.append(listener)
        return listener

    async def fixture_app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        else:
            requests.append(scope["path"])
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"safe fixture"})

    def make_server(config):
        value = original_server(config)
        running.append(value)
        return value

    def run():
        try:
            server.run(port=0)
        except BaseException as exc:  # noqa: BLE001 — return worker failures to the assertion thread.
            errors.append(type(exc).__name__)

    monkeypatch.setattr(socket, "create_server", create)
    monkeypatch.setattr(server, "create_app", lambda *_: fixture_app)
    monkeypatch.setattr(uvicorn, "Server", make_server)
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not errors and not (running and running[0].started) and time.monotonic() < deadline:
            time.sleep(.01)
        assert not errors and running and running[0].started
        connection = http.client.HTTPConnection("127.0.0.1", captured[0].getsockname()[1], timeout=2)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            assert response.status == 200 and response.read() == b"safe fixture"
        finally:
            connection.close()
    finally:
        if running:
            running[0].should_exit = True
        worker.join(timeout=5)
    assert not worker.is_alive() and not errors
    assert captured[0].fileno() == -1 and requests == ["/"]


def test_dashboard_probe_has_total_deadline_not_an_unbounded_slow_read(monkeypatch):
    now = [0.0]
    received = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def sendall(self, _):
            pass

        def settimeout(self, timeout):
            assert 0 < timeout <= 1

        def recv(self, count):
            received.append(count)
            now[0] += .4
            return b"x"

    monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: Connection())
    monkeypatch.setattr(server.time, "monotonic", lambda: now[0])
    assert not server._existing_dashboard("127.0.0.1", 8400)
    assert len(received) == 3
