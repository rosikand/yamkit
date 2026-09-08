"""Tunnel lifecycle and factory boundaries use CPU fakes and no Modal calls."""

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tests.test_http_service import TOKEN, Runtime, rpc
from yamkit.inference import http_tunnel, modal_service
from yamkit.inference.http_service import HTTP_TOKEN_ENV, create_http_app

ENDPOINT = "https://owned-endpoint.a.modal.host"


@pytest.fixture
def fake_listener(monkeypatch):
    events = []
    config = SimpleNamespace(fail_start=False, fail_forward=False, fail_revoke=False,
                             blocked=False, release=threading.Event(), timer=None, server=None)

    class Server:
        def __init__(self, server_config):
            config.server = self
            self.config = server_config
            self.started = self.should_exit = False

        def run(self):
            events.append("listener_start")
            if config.fail_start:
                return
            self.started = True
            while not self.should_exit:
                time.sleep(0.001)
            events.append("listener_stop")
            if config.blocked:
                assert config.release.wait(2)

    class Forward:
        def __enter__(self):
            events.append("forward")
            assert config.server.started
            if config.fail_forward:
                raise RuntimeError("private SDK error")
            return SimpleNamespace(url=ENDPOINT)

        def __exit__(self, *args):
            events.append("revoke")
            assert not config.server.should_exit
            if config.fail_revoke:
                raise RuntimeError("private SDK revocation error")

    def forward(port, **kwargs):
        assert port == 8000 and kwargs == {"unencrypted": False, "h2_enabled": False}
        return Forward()

    class Timer:
        def __init__(self, delay, callback):
            self.delay, self.callback = delay, callback
            self.cancelled = False
            config.timer = self

        def start(self):
            events.append("timer")

        def cancel(self):
            self.cancelled = True

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(forward=forward))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(Config=lambda *a, **k: k, Server=Server))
    monkeypatch.setattr(http_tunnel.threading, "Timer", Timer)
    monkeypatch.setattr(http_tunnel, "STARTUP_TIMEOUT_S", 0.1)
    monkeypatch.setattr(http_tunnel, "REVOKE_TIMEOUT_S", 0.1)
    monkeypatch.setattr(http_tunnel, "JOIN_TIMEOUT_S", 0.05)
    return config, events


def test_one_server_tls_only_and_revoke_precedes_listener_shutdown(fake_listener):
    config, events = fake_listener
    application = create_http_app(Runtime(), token=TOKEN)
    server = http_tunnel.HttpTunnelServer(application, expires_at=time.time() + 60).start()
    try:
        assert server.endpoint == ENDPOINT and server.application is application
        server.ensure_running()
        assert config.server.config["workers"] == 1 and config.server.config["loop"] == "asyncio"
        assert config.server.config["http"] == "h11" and not config.server.config["access_log"]
        assert config.server.config["proxy_headers"] is False
    finally:
        assert server.close() == {"tunnel_revoked": True, "server_thread_stopped": True}
    assert events.index("revoke") < events.index("listener_stop") and config.timer.cancelled
    assert server.close() == {"tunnel_revoked": True, "server_thread_stopped": True}
    assert events.count("revoke") == 1
    with pytest.raises(RuntimeError, match="unavailable"):
        server.ensure_running()


@pytest.mark.parametrize("failure", ["fail_start", "fail_forward"])
def test_startup_failure_cleans_partial_listener_without_leaking_errors(fake_listener, failure):
    config, events = fake_listener
    setattr(config, failure, True)
    server = http_tunnel.HttpTunnelServer(object(), expires_at=time.time() + 60)
    with pytest.raises(RuntimeError, match="startup failed") as error:
        server.start()
    assert "private" not in str(error.value) and not server.thread.is_alive()
    if failure == "fail_start":
        assert "forward" not in events
    else:
        assert "revoke" in events and "listener_stop" in events


def test_failed_revocation_does_not_skip_listener_shutdown_or_claim_success(fake_listener):
    config, events = fake_listener
    server = http_tunnel.HttpTunnelServer(object(), expires_at=time.time() + 60).start()
    config.fail_revoke = True
    with pytest.raises(RuntimeError, match="unverified"):
        server.close()
    assert "listener_stop" in events and not server.thread.is_alive()


def test_lingering_server_thread_cannot_be_reported_as_stopped(fake_listener):
    config, _ = fake_listener
    config.blocked = True
    server = http_tunnel.HttpTunnelServer(object(), expires_at=time.time() + 60).start()
    try:
        with pytest.raises(RuntimeError, match="unverified"):
            server.close()
        assert server.thread.is_alive()
    finally:
        config.release.set()
        server.thread.join(timeout=1)
    assert server.close()["server_thread_stopped"] is True


def test_expiry_callback_revokes_without_waiting_for_owner_input(fake_listener):
    config, events = fake_listener
    server = http_tunnel.HttpTunnelServer(object(), expires_at=time.time() + 60).start()
    assert 0 < config.timer.delay <= 60
    config.timer.callback()
    assert "revoke" in events and not server.thread.is_alive()


@pytest.fixture
def factory(monkeypatch):
    events, classes = [], []

    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def pip_install_from_requirements(self, path):
            events.append("requirements")
            return self

        def pip_install(self, *packages):
            assert packages == ("uvicorn==0.52.4", "h11==0.16.0")
            events.append("server_packages")
            return self

        def env(self, values):
            return self

        def add_local_dir(self, *args, **kwargs):
            events.append("source")
            return self

        def add_local_file(self, *args):
            return self

    class App:
        def __init__(self, name):
            pass

        def cls(self, **kwargs):
            def decorate(cls):
                classes.append(cls)
                return cls
            return decorate

    runtime = Runtime()
    runtime._lock = threading.Lock()

    class Tunnel:
        def __init__(self, application, *, expires_at):
            self.application, self.expires_at = application, expires_at
            self.endpoint = ENDPOINT

        def start(self):
            events.append("tunnel_start")

        def ensure_running(self):
            events.append("tunnel_health")

        def close(self):
            events.append("tunnel_close")

    def decorator(label):
        def make():
            def decorate(fn):
                events.append(label)
                return fn
            return decorate
        return make

    fake = SimpleNamespace(App=App, Image=Image,
        Volume=SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(commit=lambda: None)),
        Secret=SimpleNamespace(from_dict=lambda value: value), enter=lambda: lambda f: f,
        method=lambda: lambda f: f, asgi_app=decorator("asgi_endpoint"), exit=decorator("exit_hook"))
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setenv(HTTP_TOKEN_ENV, TOKEN)
    monkeypatch.setattr("yamkit.inference.service.ModelRuntime.load", lambda *a, **k: runtime)
    monkeypatch.setattr("yamkit.inference.identity.inference_build_id", lambda: "current-build")
    monkeypatch.setattr(modal_service, "_prepare_http_graph_gc", lambda runtime: {"strategy": "prepared"})
    monkeypatch.setattr(http_tunnel, "HttpTunnelServer", Tunnel)
    return SimpleNamespace(events=events, classes=classes, runtime=runtime)


def test_tunnel_factory_has_one_ingress_guarded_readiness_and_no_sdk_mutations(factory):
    expiry = time.time() + 600
    modal_service.create_app("molmoact2", transport="http", execution_mode="cuda_graph10", http_token=TOKEN,
                             http_ingress="tunnel", min_containers=1, http_session_expires_at=expiry)
    assert factory.events.index("requirements") < factory.events.index("server_packages") < factory.events.index("source")
    assert "asgi_endpoint" not in factory.events and "exit_hook" in factory.events
    service = factory.classes[0]()
    assert "http" not in factory.classes[0].__dict__
    service.load()
    ready = service.ready()
    assert ready["http_ingress"] == "tunnel" and ready["http_session_expires_at"] == expiry
    assert ready["http_endpoint"] == ENDPOINT and ready["instance_id"] == "same-instance"
    assert ready["supported_call_modes"] == ["http"]
    with factory.runtime._lock, pytest.raises(RuntimeError, match="busy"):
        service.ready()
    for call in (lambda: service.predict_chunk({"pixels": b"x"}), lambda: service.reset("session")):
        with pytest.raises(ValueError, match="HTTP endpoint"):
            call()
    # HTTP readiness and model calls use distinct adapter/runtime locks; no double-lock deadlock.
    assert asyncio.run(rpc(service._http_tunnel.application))[0] == 200
    assert asyncio.run(rpc(service._http_tunnel.application, {
        "method": "predict_chunk", "payload": {"pixels": b"x"}}))[0] == 200
    service.stop_http()
    assert factory.events[-1] == "tunnel_close"


def test_default_factory_keeps_asgi_without_tunnel_packages_or_exit_hook(factory):
    modal_service.create_app("molmoact2", transport="http", http_token=TOKEN)
    assert "asgi_endpoint" in factory.events
    assert "server_packages" not in factory.events and "exit_hook" not in factory.events
    service = factory.classes[0]()
    service.load()
    assert service.ready()["http_ingress"] == "asgi"
    assert service.ready()["http_session_expires_at"] is None


@pytest.mark.parametrize("options", [
    {"http_ingress": "unknown"}, {"http_ingress": "tunnel"},
    {"http_ingress": "tunnel", "transport": "sdk"},
    {"http_ingress": "asgi", "http_session_expires_at": 123},
    *({"http_ingress": "tunnel", "min_containers": 1, "http_session_expires_at": value}
      for value in (True, None, float("nan"), float("inf"), 0, 1e30)),
])
def test_invalid_tunnel_factory_options_fail_before_modal_objects(factory, options):
    kwargs = {"transport": "http", "http_token": TOKEN, **options}
    with pytest.raises(ValueError):
        modal_service.create_app(**kwargs)
    assert factory.events == []
