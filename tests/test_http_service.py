"""The HTTP adapter serves one inert runtime without blocking its ASGI loop."""

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from yamkit.inference.http_service import HTTP_TOKEN_ENV, create_http_app
from yamkit.inference.http_wire import MAX_WIRE_BYTES, decode_message, encode_message

TOKEN = "test-only-dedicated-bearer-token-0123456789"


class Runtime:
    def __init__(self):
        self.calls = []

    def ready(self):
        self.calls.append("ready")
        return {"ready": True, "instance_id": "same-instance"}

    def predict_chunk(self, request):
        self.calls.append(request)
        return {"instance_id": "same-instance", "pixels": request["pixels"]}

    def reset(self, session_id):
        self.calls.append(("reset", session_id))


async def rpc(app, message=None, *, body=None, headers=None, events=None, method="POST", path="/rpc",
              query=b""):
    wire = encode_message(message or {"method": "ready", "payload": None}) if body is None else body
    pending = list(events) if events is not None else [{"type": "http.request", "body": wire}]
    sent, reads = [], []

    async def receive():
        reads.append(True)
        assert pending, "ASGI read past the end of this request"
        return pending.pop(0)

    async def send(value):
        sent.append(value)

    await app({"type": "http", "method": method, "path": path, "query_string": query,
               "headers": [(b"authorization", ("Bearer " + TOKEN).encode())] if headers is None else headers},
              receive, send)
    if not sent:
        return None, None, reads
    assert len(sent) == 2
    assert (b"cache-control", b"no-store") in sent[0]["headers"]
    return sent[0]["status"], decode_message(sent[1]["body"]), reads


def test_authenticated_methods_share_runtime_and_preserve_binary_values():
    async def scenario():
        runtime = Runtime()
        app = create_http_app(runtime, token=TOKEN, ready=lambda: {**runtime.ready(), "transport": "http"})
        status, value, _ = await rpc(app)
        assert status == 200 and value["result"]["transport"] == "http"
        assert value["result"]["instance_id"] == "same-instance"
        pixels = bytes(640 * 480 * 3)
        status, value, _ = await rpc(app, {"method": "predict_chunk", "payload": {"pixels": pixels}})
        assert status == 200 and value == {"ok": True, "result": {
            "instance_id": "same-instance", "pixels": pixels}}
        status, value, _ = await rpc(app, {"method": "reset", "payload": {"session_id": "session-1"}})
        assert status == 200 and value == {"ok": True, "result": {}}
        assert runtime.calls == ["ready", {"pixels": pixels}, ("reset", "session-1")]

    asyncio.run(scenario())


@pytest.mark.parametrize("headers", [[], [(b"authorization", b"Bearer wrong")],
                                     [(b"authorization", ("Bearer " + TOKEN).encode())] * 2])
def test_authentication_precedes_any_body_read(headers):
    runtime = Runtime()
    status, value, reads = asyncio.run(rpc(create_http_app(runtime, token=TOKEN), headers=headers,
                                          events=[]))
    assert status == 401 and not reads and not runtime.calls
    assert value == {"ok": False, "error_type": "Rejected"}


@pytest.mark.parametrize("message", [
    {"method": "ready", "payload": {}}, {"method": "ready", "payload": None, "extra": True},
    {"method": "predict_chunk", "payload": None}, {"method": "predict_chunk", "payload": {}},
    {"method": "reset", "payload": {"session_id": ""}},
    {"method": "reset", "payload": {"session_id": "a", "extra": True}},
    {"method": "reset", "payload": {"session_id": 7}}, {"method": "unknown", "payload": None},
    {"method": ["ready"], "payload": None}, {"payload": None},
])
def test_strict_envelopes_are_rejected_before_runtime(message):
    runtime = Runtime()
    status, _, _ = asyncio.run(rpc(create_http_app(runtime, token=TOKEN), message))
    assert status == 400 and not runtime.calls


@pytest.mark.parametrize("overrides", [{"method": "GET"}, {"path": "/"}, {"query": b"secret=x"}])
def test_only_post_rpc_without_query_is_available(overrides):
    status, _, reads = asyncio.run(rpc(create_http_app(Runtime(), token=TOKEN), events=[], **overrides))
    assert status == 404 and not reads


def test_declared_and_streamed_payloads_are_bounded_and_disconnect_is_inert():
    async def scenario():
        runtime = Runtime()
        app = create_http_app(runtime, token=TOKEN)
        auth = (b"authorization", ("Bearer " + TOKEN).encode())
        status, _, reads = await rpc(app, headers=[auth, (b"content-length", b"99999999")], events=[])
        assert status == 413 and not reads
        status, _, reads = await rpc(app, events=[
            {"type": "http.request", "body": bytes(MAX_WIRE_BYTES), "more_body": True},
            {"type": "http.request", "body": b"x", "more_body": True},
        ])
        assert status == 413 and len(reads) == 2
        for length in (b"-1", b"oops", b"000000000000", b"0"):
            status, _, _ = await rpc(app, headers=[auth, (b"content-length", length)])
            assert status == 400
        status, _, _ = await rpc(app, body=b"invalid binary frame")
        assert status == 400
        status, _, _ = await rpc(app, events=[{"type": "http.disconnect"}])
        assert status is None and not runtime.calls

    asyncio.run(scenario())


def test_runtime_errors_and_unencodable_results_do_not_leak_details():
    runtime = Runtime()

    def fail():
        raise RuntimeError("private payload, account token, or server traceback")

    for ready in (fail, lambda: {"bad": object()}, lambda: ["not a dictionary"]):
        status, value, _ = asyncio.run(rpc(create_http_app(runtime, token=TOKEN, ready=ready)))
        assert status == 500 and value == {"ok": False, "error_type": "Failed"}


@pytest.mark.parametrize("finish", ["cancel", "timeout", "complete"])
def test_worker_does_not_block_loop_and_keeps_ownership_until_it_finishes(finish):
    entered, release = threading.Event(), threading.Event()

    def blocking_ready():
        entered.set()
        assert release.wait(2)
        return {"ready": True}

    async def scenario():
        app = create_http_app(Runtime(), token=TOKEN, ready=blocking_ready,
                              request_timeout_s=0.05 if finish == "timeout" else 1)
        worker = asyncio.create_task(rpc(app))
        try:
            async with asyncio.timeout(1):
                while not entered.is_set():
                    await asyncio.sleep(0.001)
            # A blocked model thread must not prevent another HTTP request from
            # promptly reaching the endpoint's single-flight rejection.
            status, value, _ = await asyncio.wait_for(rpc(app), 0.1)
            assert status == 409 and value["error_type"] == "Busy"
            if finish == "cancel":
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker
            elif finish == "timeout":
                status, value, _ = await worker
                assert status == 504 and value["error_type"] == "Deadline"
            if finish != "complete":
                status, _, _ = await rpc(app)
                assert status == 409, "Cancelled await must not release unfinished model work"
        finally:
            release.set()
            if finish == "complete":
                status, value, _ = await worker
                assert status == 200 and value["ok"] is True
        async with asyncio.timeout(1):
            while (await rpc(app))[0] == 409:
                await asyncio.sleep(0.001)

    asyncio.run(scenario())


def test_lifespan_handshake():
    async def scenario():
        events = iter([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}])
        replies = []

        async def receive():
            return next(events)

        async def send(event):
            replies.append(event)

        await create_http_app(Runtime(), token=TOKEN)({"type": "lifespan"}, receive, send)
        assert replies == [{"type": "lifespan.startup.complete"}, {"type": "lifespan.shutdown.complete"}]

    asyncio.run(scenario())


def test_cancel_before_executor_start_does_not_strand_model_ownership(monkeypatch):
    original_to_thread = asyncio.to_thread

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def queued(function, *args):
            entered.set()
            await release.wait()
            return await original_to_thread(function, *args)

        monkeypatch.setattr(asyncio, "to_thread", queued)
        runtime = Runtime()
        app = create_http_app(runtime, token=TOKEN)
        caller = asyncio.create_task(rpc(app))
        await asyncio.wait_for(entered.wait(), 1)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert (await rpc(app))[0] == 409
        assert not runtime.calls
        release.set()
        async with asyncio.timeout(1):
            while (await rpc(app))[0] == 409:
                await asyncio.sleep(0.001)
        assert runtime.calls == ["ready", "ready"]

    asyncio.run(scenario())


@pytest.mark.parametrize("token", [None, "short", "x" * 257, "x" * 32 + "\n", "x" * 32 + "é"])
def test_invalid_secret_rejected_without_echo(token):
    with pytest.raises(ValueError, match="dedicated") as exc:
        create_http_app(Runtime(), token=token)
    if token:
        assert token not in str(exc.value)


def test_expired_session_rejects_before_reading_body_or_running_model():
    runtime = Runtime()
    app = create_http_app(runtime, token=TOKEN, session_expires_at=time.time() - 1)
    status, value, reads = asyncio.run(rpc(app, events=[]))
    assert status == 410 and value["error_type"] == "Expired"
    assert not reads and not runtime.calls


@pytest.mark.parametrize("expiry", [True, "tomorrow", float("nan"), float("inf"), 0])
def test_malformed_session_expiry_is_rejected(expiry):
    with pytest.raises(ValueError, match="expiry"):
        create_http_app(Runtime(), token=TOKEN, session_expires_at=expiry)


def test_session_expiring_while_executor_is_queued_never_calls_runtime(monkeypatch):
    original_to_thread = asyncio.to_thread

    async def scenario():
        queued, release = asyncio.Event(), asyncio.Event()
        finished = asyncio.Event()

        async def delayed(function, *args):
            queued.set()
            await release.wait()
            try:
                return await original_to_thread(function, *args)
            finally:
                finished.set()

        monkeypatch.setattr(asyncio, "to_thread", delayed)
        runtime = Runtime()
        app = create_http_app(runtime, token=TOKEN, request_timeout_s=1,
                              session_expires_at=time.time() + 0.05)
        request = asyncio.create_task(rpc(app))
        await asyncio.wait_for(queued.wait(), 1)
        status, value, _ = await request
        assert status == 410 and value["error_type"] == "Expired"
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        assert not runtime.calls

    asyncio.run(scenario())


@pytest.mark.parametrize("min_containers", [None, 1])
def test_factory_adds_opt_in_endpoint_to_same_class_without_forwarding_account_secrets(monkeypatch, min_containers):
    from yamkit.inference.identity import FOLLOWER_SOURCE_RELATIVE
    from yamkit.inference.modal_service import create_app
    from yamkit.paths import ROOT

    captured = {"secrets": [], "classes": [], "files": []}

    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def pip_install_from_requirements(self, path):
            return self

        def env(self, values):
            captured["env"] = values
            return self

        def add_local_dir(self, *args, **kwargs):
            return self

        def add_local_file(self, *args):
            captured["files"].append(args)
            return self

    class App:
        def __init__(self, name):
            self.name = name

        def cls(self, **kwargs):
            captured["config"] = kwargs

            def decorate(cls):
                captured["classes"].append(cls)
                return cls
            return decorate

    def secret(values):
        captured["secrets"].append(values)
        return values

    fake = SimpleNamespace(
        App=App, Image=Image, Volume=SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(commit=lambda: None)),
        Secret=SimpleNamespace(from_dict=secret), enter=lambda: lambda f: f, method=lambda: lambda f: f,
        asgi_app=lambda: lambda f: f,
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setitem(sys.modules, "yamkit.inference.identity", SimpleNamespace(
        inference_build_id=lambda: "build-hash", FOLLOWER_SOURCE_RELATIVE=FOLLOWER_SOURCE_RELATIVE))
    runtime = Runtime()

    def load(profile, *, device, execution_mode):
        assert (profile, device, execution_mode) == ("molmoact2", "cuda", "cuda_graph10")
        return runtime

    monkeypatch.setattr("yamkit.inference.service.ModelRuntime.load", load)
    # This factory runs in pytest; only the separate fake-GC tests exercise the
    # cloud startup guard. Never freeze this process's real objects.
    monkeypatch.setattr("yamkit.inference.modal_service._prepare_http_graph_gc",
                        lambda runtime: {"strategy": "fake_gc_for_factory_test"})
    monkeypatch.setenv("HF_TOKEN", "test-hf-secret")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "must-never-forward")
    monkeypatch.setenv(HTTP_TOKEN_ENV, TOKEN)
    pool_options = {} if min_containers is None else {"min_containers": min_containers}
    create_app("molmoact2", transport="http", execution_mode="cuda_graph10", http_token=TOKEN,
               gpu="H100!", **pool_options)
    assert len(captured["classes"]) == 1
    cls = captured["classes"][0]
    assert "__init__" not in cls.__dict__
    assert {"ready", "predict_chunk", "reset", "http"} <= cls.__dict__.keys()
    assert captured["config"]["max_containers"] == 1
    assert captured["config"]["min_containers"] == (min_containers or 0)
    assert captured["secrets"] == [{"HF_TOKEN": "test-hf-secret"}, {HTTP_TOKEN_ENV: TOKEN}]
    assert not {"MODAL_TOKEN_SECRET", "HF_TOKEN", HTTP_TOKEN_ENV} & captured["env"].keys()
    assert captured["files"] == [
        (str(ROOT / "configs/modal-requirements.txt"), "/opt/yamkit/configs/modal-requirements.txt"),
        (str(ROOT / FOLLOWER_SOURCE_RELATIVE), f"/opt/yamkit/{FOLLOWER_SOURCE_RELATIVE}"),
        (str(ROOT / "scripts/benchmark_remote.py"), "/opt/yamkit/scripts/benchmark_remote.py"),
        (str(ROOT / "scripts/setup_inference.sh"), "/opt/yamkit/scripts/setup_inference.sh"),
    ]
    assert captured["env"]["PYTHONPATH"] == "/opt/yamkit/src"  # Plugin source is data, not an import path.
    service = cls()
    service.load()
    sdk = service.ready()
    status, response, _ = asyncio.run(rpc(service.http()))
    assert status == 200 and response["result"] == sdk
    assert sdk["supported_call_modes"] == ["remote", "http"]
    assert sdk["preferred_call_mode"] == sdk["transport"] == "http"
    assert sdk["execution_mode"] == "cuda_graph10" and sdk["inference_build_id"] == "build-hash"
    assert sdk["http_wire_version"] == 1
    assert sdk["min_containers"] == (min_containers or 0)
    assert sdk["max_containers"] == 1
    create_app()
    assert captured["config"]["min_containers"] == 0
    assert "http" not in captured["classes"][-1].__dict__
    for kwargs in ({"transport": "invalid"}, {"http_token": TOKEN}, {"transport": "http"},
                   {"execution_mode": "invalid"}, {"execution_mode": "cuda_graph10"}):
        with pytest.raises(ValueError):
            create_app(**kwargs)


@pytest.mark.parametrize("min_containers", [-1, 2, True, False, 0.0, 1.0, "1", None])
def test_factory_rejects_invalid_minimum_before_constructing_modal_objects(min_containers):
    from yamkit.inference.modal_service import create_app

    with pytest.raises(ValueError, match="min_containers"):
        create_app(min_containers=min_containers)
