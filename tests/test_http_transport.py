"""Fake HTTP verifies cancellation, true local deadlines, bounds and credential isolation."""

import threading
import time

import httpx
import pytest

from yamkit.inference import http_transport
from yamkit.inference.client import InvalidatedRequest, RemoteFault, RemoteSession
from yamkit.inference.http_transport import HttpTransport, validate_endpoint_url
from yamkit.inference.http_wire import MAX_MESSAGE_BYTES, WIRE_CODEC, decode_message, encode_message
from yamkit.inference.profiles import get_profile

TOKEN = "SECRET_SENTINEL_" + "x" * 40
ENDPOINT = "https://test-app.modal.run"


def reply(result=None, **kwargs):
    return httpx.Response(200, stream=httpx.ByteStream(encode_message({"ok": True, "result": result or {}})),
                          headers={"content-type": "application/octet-stream"}, **kwargs)


@pytest.fixture
def factory(monkeypatch):
    clients = []

    def install(handler):
        def make(token):
            client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False,
                                  headers={"Authorization": "Bearer " + token,
                                           "Content-Type": "application/octet-stream",
                                           "Accept-Encoding": "identity"})
            clients.append(client)
            return client
        monkeypatch.setattr(http_transport, "_make_client", make)
        return clients
    yield install
    for client in clients:
        client.close()


def transport(**kwargs):
    return HttpTransport("yamkit-vla-test", "molmoact2", endpoint_url=ENDPOINT, token=TOKEN, **kwargs)


def background(call):
    result = {}
    done = threading.Event()

    def run():
        try:
            result["value"] = call()
        except BaseException as exc:  # noqa: BLE001 — retain thread failures for the parent test assertion
            result["error"] = exc
        finally:
            done.set()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return result, done, thread


def retired(value):
    assert value._busy.acquire(timeout=1), "HTTP worker did not retire"
    value._busy.release()


def test_factory_disables_proxies_retries_redirects_and_uses_one_connection(monkeypatch):
    seen = {}
    marker = object()
    monkeypatch.setattr(httpx, "HTTPTransport", lambda **kwargs: seen.setdefault("transport", kwargs) or marker)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: seen.setdefault("client", kwargs))
    http_transport._make_client(TOKEN)
    assert seen["transport"]["retries"] == 0
    assert seen["transport"]["trust_env"] is False
    assert seen["transport"]["limits"].max_connections == 1
    assert seen["transport"]["limits"].max_keepalive_connections == 1
    assert seen["client"]["trust_env"] is False
    assert seen["client"]["follow_redirects"] is False
    assert seen["client"]["headers"]["Accept-Encoding"] == "identity"


@pytest.mark.parametrize("endpoint", [
    "http://test.modal.run", "https://modal.run", "https://test.modal.run.attacker.test",
    "https://test.modal.run@attacker.test", "https://user:password@test.modal.run", "https://127.0.0.1",
    "https://test.modal.run:8443", "https://test.modal.run/rpc", "https://test.modal.run?token=secret",
    "https://test.modal.run?", "https://test.modal.run#", "https://test.modal.run#fragment",
    "https://test.modal.run\n", "https://test.modal.run/../", "https://a..modal.run",
    "https://-bad.modal.run", "https://bad_.modal.run", "https://\u00e9.modal.run",
    "https://test.modal.run:bogus", "https://test.modal.run\\@attacker.test",
])
def test_rejects_unsafe_endpoint_before_client_creation(endpoint, monkeypatch):
    monkeypatch.setattr(http_transport, "_make_client", lambda *_: pytest.fail("Client created during validation"))
    with pytest.raises(ValueError) as error:
        HttpTransport("yamkit-vla-test", "molmoact2", endpoint_url=endpoint, token=TOKEN)
    assert endpoint not in str(error.value)


@pytest.mark.parametrize("token", [None, "short", "x" * 257, "x" * 40 + "\nInjected: header", "x" * 40 + "\u00e9"])
def test_rejects_invalid_credentials_without_echoing_them(token):
    with pytest.raises(ValueError) as error:
        HttpTransport("yamkit-vla-test", "molmoact2", endpoint_url=ENDPOINT, token=token)
    assert token is None or token not in str(error.value)


def test_lazy_persistent_roundtrip_preserves_raw_bytes_and_reset(factory):
    requests = []

    def handler(request):
        assert str(request.url) == ENDPOINT + "/rpc"
        assert request.headers["authorization"] == "Bearer " + TOKEN
        assert request.headers["content-type"] == "application/octet-stream"
        envelope = decode_message(request.content)
        requests.append(envelope)
        return reply({"method": envelope["method"], "payload": envelope["payload"]})

    clients = factory(handler)
    value = transport()
    assert clients == []
    assert TOKEN not in repr(value) and ENDPOINT not in repr(value)
    assert validate_endpoint_url(ENDPOINT.upper() + ":443/") == ENDPOINT
    assert value.ready(1) == {"method": "ready", "payload": None}
    first = dict(value.last_timing)
    value.cancel()  # A completed generation is invalidated, but the next call is allowed.
    payload = {"images": {"top": {"data": bytes(range(256)) * 3600}}, "state": [0.125] * 14}
    assert value.predict_chunk(payload, 1)["payload"] == payload
    assert value.reset("retired-session", 1)["payload"] == {"session_id": "retired-session"}
    assert len(clients) == 1 and len(requests) == 3
    assert first["client_reused"] is False and value.last_timing["client_reused"] is True
    assert value.last_timing["wire_codec"] == WIRE_CODEC
    assert value.last_timing["wire_request_bytes"] > 0
    assert value.last_timing["wire_response_bytes"] > 0
    assert TOKEN not in str(value.last_timing) and ENDPOINT not in str(value.last_timing)


@pytest.mark.parametrize("mode", ["cancel", "shutdown"])
def test_stop_returns_promptly_but_late_inflight_response_cannot_enable_next_call(factory, mode):
    entered, release = threading.Event(), threading.Event()
    shutdown = threading.Event()

    def handler(request):
        entered.set()
        assert release.wait(2)
        return reply({"late": True})

    factory(handler)
    value = transport(shutdown_event=shutdown)
    result, done, thread = background(lambda: value.ready(1))
    try:
        assert entered.wait(1)
        begin = time.monotonic()
        (value.cancel if mode == "cancel" else shutdown.set)()
        assert time.monotonic() - begin < 0.05
        assert done.wait(0.3)
        assert isinstance(result.get("error"), InvalidatedRequest)
        assert "value" not in result
        if mode == "cancel":
            with pytest.raises(RemoteFault, match="still in flight"):
                value.ready(0.1)
        else:
            with pytest.raises(InvalidatedRequest, match="stopped"):
                value.ready(0.1)
    finally:
        release.set()
        thread.join(1)
        retired(value)
    shutdown.clear()
    assert value.ready(1) == {"late": True}


def test_total_deadline_survives_underlying_http_that_ignores_phase_timeout(factory):
    entered, release = threading.Event(), threading.Event()

    def handler(request):
        entered.set()
        assert release.wait(2)
        return reply({"late": True})

    factory(handler)
    value = transport()
    started = time.monotonic()
    try:
        with pytest.raises(RemoteFault, match="deadline"):
            value.ready(0.04)
        assert entered.is_set()
        assert time.monotonic() - started < 0.3
        with pytest.raises(RemoteFault, match="still in flight"):
            value.ready(0.05)
    finally:
        release.set()
        retired(value)
    assert value.ready(1) == {"late": True}


@pytest.mark.parametrize("stage", ["encode", "client", "decode"])
def test_deadline_covers_codec_and_client_setup_not_only_network(factory, monkeypatch, stage):
    entered, release = threading.Event(), threading.Event()
    sent = []
    factory(lambda request: sent.append(request) or reply({"accepted": True}))
    target = {"encode": "encode_message", "client": "_make_client", "decode": "decode_message"}[stage]
    original = getattr(http_transport, target)

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(http_transport, target, blocked)
    value = transport()
    try:
        with pytest.raises(RemoteFault, match="deadline"):
            value.ready(0.04)
        assert entered.is_set()
        with pytest.raises(RemoteFault, match="still in flight"):
            value.ready(0.05)
    finally:
        release.set()
        retired(value)
    if stage in ("encode", "client"):
        assert sent == []


@pytest.mark.parametrize("status", [301, 307, 401, 409, 500, 503])
def test_http_errors_are_single_attempt_and_never_leak_body_url_or_token(factory, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, text=TOKEN + " " + ENDPOINT,
                              headers={"location": "https://attacker.test/" + TOKEN})

    factory(handler)
    value = transport()
    with pytest.raises(RemoteFault) as error:
        value.ready(1)
    assert len(requests) == 1
    assert TOKEN not in str(error.value) and ENDPOINT not in str(error.value)
    assert error.value.__context__ is None


def test_http_exception_details_never_cross_worker_boundary(factory):
    def handler(request):
        raise httpx.ReadError(TOKEN + " " + ENDPOINT, request=request)

    factory(handler)
    with pytest.raises(RemoteFault) as error:
        transport().ready(1)
    assert TOKEN not in str(error.value) and ENDPOINT not in str(error.value)
    assert error.value.__context__ is None


@pytest.mark.parametrize("envelope", [
    {"ok": False, "error_type": TOKEN}, {"ok": 1, "result": {}}, {"ok": True, "result": []},
    {"ok": True, "result": {}, "unexpected": TOKEN}, {"ok": True},
])
def test_response_envelope_is_strict_and_server_error_text_is_not_exposed(factory, envelope):
    factory(lambda request: httpx.Response(200, stream=httpx.ByteStream(encode_message(envelope)),
                                           headers={"content-type": "application/octet-stream"}))
    with pytest.raises(RemoteFault) as error:
        transport().ready(1)
    assert TOKEN not in str(error.value)


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.yielded = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk


def test_stream_rejects_oversize_before_reading_later_chunks(factory):
    stream = Chunks([b"x" * MAX_MESSAGE_BYTES, b"x", TOKEN.encode()])
    factory(lambda request: httpx.Response(200, stream=stream,
                                           headers={"content-type": "application/octet-stream"}))
    with pytest.raises(RemoteFault):
        transport().ready(1)
    assert stream.yielded == 2


@pytest.mark.parametrize("headers", [
    {"content-length": str(MAX_MESSAGE_BYTES + 1)}, {"content-length": "-1"}, {"content-length": "invalid"},
    {"content-encoding": "gzip"}, {"content-type": "text/html"},
])
def test_unsafe_response_headers_rejected_before_stream_read(factory, headers):
    stream = Chunks([b"x"])
    factory(lambda request: httpx.Response(200, stream=stream,
                                           headers={"content-type": "application/octet-stream", **headers}))
    with pytest.raises(RemoteFault):
        transport().ready(1)
    assert stream.yielded == 0


def test_truncated_or_trailing_binary_message_is_rejected(factory):
    good = encode_message({"ok": True, "result": {}})
    for body in (good[:-1], good + b"trailing"):
        factory(lambda request, body=body: httpx.Response(200, stream=httpx.ByteStream(body),
                                                       headers={"content-type": "application/octet-stream"}))
        with pytest.raises(RemoteFault):
            transport().ready(1)


def test_failed_call_has_fresh_timing_not_previous_success(factory):
    fail = False

    def handler(request):
        if fail:
            raise httpx.ReadError(TOKEN, request=request)
        return reply({"ready": True})

    factory(handler)
    value = transport()
    value.ready(1)
    assert "wire_response_bytes" in value.last_timing
    fail = True
    with pytest.raises(RemoteFault):
        value.ready(1)
    assert "wire_response_bytes" not in value.last_timing


def test_remote_session_reset_rejects_late_actions_then_accepts_new_generation(factory):
    profile = get_profile("molmoact2")
    entered, release = threading.Event(), threading.Event()
    requests = []

    def handler(request):
        payload = decode_message(request.content)["payload"]
        requests.append(payload)
        entered.set()
        assert release.wait(2)
        result = {key: payload[key] for key in (
            "protocol_version", "profile", "model_revision", "session_id", "sequence_id", "observation_time")}
        result.update(action_units="robot", instance_id="fake", action_names=list(profile.action_names),
                      mode=payload.get("mode", "robot"),
                      execution_mode=payload.get("execution_mode", "eager"),
                      chunk=[[0.0] * 14], timing=dict.fromkeys(
                          ("preprocess_s", "inference_s", "postprocess_s", "total_s"), 0.0))
        return reply(result)

    factory(handler)
    value = transport()
    session = RemoteSession(value, profile, timeout_s=1)
    images = {name: {"encoding": "rgb8", "height": 1, "width": 1, "data": b"\0\0\0"}
              for name in profile.image_keys}

    def predict():
        return session.predict(state=[0.0] * 14, images=images, task="test generation",
                               observation_time=time.monotonic())

    result, done, thread = background(predict)
    try:
        assert entered.wait(1)
        prior = session.session_id
        session.reset()
        assert done.wait(0.3)
        assert isinstance(result.get("error"), InvalidatedRequest)
        assert len(session.samples) == 0
    finally:
        release.set()
        thread.join(1)
        retired(value)
    response = predict()
    assert response["session_id"] == session.session_id != prior
    assert len(requests) == 2 and len(session.samples) == 1


def test_terminal_close_before_first_call_is_permanent_and_idempotent(factory):
    clients = factory(lambda request: pytest.fail("Closed transport sent a request"))
    value = transport()
    value.close()
    value.close()
    value.cancel()
    with pytest.raises(InvalidatedRequest, match="closed"):
        value.ready(1)
    assert clients == []


def test_idle_terminal_close_cleans_pool_once_without_waiting_for_cleanup(factory, monkeypatch):
    clients = factory(lambda request: reply({"ready": True}))
    value = transport()
    value.ready(1)
    client = clients[0]
    close_started, allow_close, closed = threading.Event(), threading.Event(), threading.Event()
    closes = []
    original = client.close

    def slow_close():
        closes.append(True)
        close_started.set()
        assert allow_close.wait(2)
        original()
        closed.set()

    monkeypatch.setattr(client, "close", slow_close)
    try:
        begin = time.monotonic()
        value.close()
        assert time.monotonic() - begin < 0.1
        assert close_started.wait(1)
        value.close()
        value.cancel()
        with pytest.raises(InvalidatedRequest, match="closed"):
            value.ready(1)
        assert closes == [True]
    finally:
        allow_close.set()
        assert closed.wait(1)
        monkeypatch.setattr(client, "close", original)
    assert client.is_closed


def test_terminal_close_during_read_defers_pool_close_until_underlying_read_ends(factory, monkeypatch):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def handler(request):
        entered.set()
        assert release.wait(2)
        return reply({"late": True})

    clients = factory(handler)
    value = transport()
    result, done, thread = background(lambda: value.ready(1))
    assert entered.wait(1)
    client = clients[0]
    original = client.close
    closes = []

    def record_close():
        assert release.is_set(), "Pool closed concurrently with an active read"
        closes.append(True)
        original()
        closed.set()

    monkeypatch.setattr(client, "close", record_close)
    try:
        value.close()
        value.close()
        assert done.wait(0.3)
        assert isinstance(result.get("error"), InvalidatedRequest)
        assert "value" not in result
        assert not closed.is_set() and not client.is_closed
        with pytest.raises(InvalidatedRequest, match="closed"):
            value.ready(1)
    finally:
        release.set()
        thread.join(1)
        retired(value)
        assert closed.wait(1)
        monkeypatch.setattr(client, "close", original)
    assert closes == [True] and client.is_closed
