"""Exact tunnel identity and lifetime are required before any fake hardware use."""

import copy
import hashlib
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from tests.test_http_runtime_binding import IMAGE_HW, TASK, runtime_metadata
from tests.test_http_runtime_binding import http_evidence as _http_evidence
from yamkit import modal_ops
from yamkit.inference import http_transport, identity, qualification
from yamkit.inference.client import InvalidatedRequest, RemoteFault
from yamkit.inference.http_transport import HttpTransport, validate_endpoint_url
from yamkit.inference.http_wire import decode_message, encode_message

APP = "yamkit-vla-tunnel-test"
URL = "https://instance-123.relay-us-west.modal.host"
OTHER = "https://instance-456.relay-us-west.modal.host"
TOKEN = "test-only-tunnel-secret-01234567890123456789"
NOW = 2000.0
EXPIRY = NOW + 600.0


@pytest.fixture(autouse=True)
def fixed_time_and_build(monkeypatch):
    monkeypatch.setattr(identity.time, "time", lambda: NOW)
    monkeypatch.setattr(identity, "inference_build_id", lambda: "b" * 64)


def route_metadata(**changes):
    return {"http_ingress": "tunnel", "http_endpoint": URL, "http_session_expires_at": EXPIRY, **changes}


def transport(**changes):
    return HttpTransport(APP, "molmoact2", endpoint_url=URL, token=TOKEN,
                         **{"http_ingress": "tunnel", "http_session_expires_at": EXPIRY, **changes})


def install_http(monkeypatch, handler):
    client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False, follow_redirects=False)
    monkeypatch.setattr(http_transport, "_make_client", lambda token: client)
    return client


def reply(value):
    return httpx.Response(200, stream=httpx.ByteStream(encode_message({"ok": True, "result": value})),
                          headers={"content-type": "application/octet-stream"})


def test_tunnel_urls_are_explicit_and_accept_provider_dns_labels():
    assert validate_endpoint_url(URL.upper() + ":443/", http_ingress="tunnel") == URL
    assert validate_endpoint_url("https://abc.r1.modal.host", http_ingress="tunnel")
    with pytest.raises(ValueError):
        validate_endpoint_url(URL)
    with pytest.raises(ValueError):
        validate_endpoint_url("https://app.modal.run", http_ingress="tunnel")


@pytest.mark.parametrize("url", [
    "https://modal.host", "https://one.modal.host", "https://one.two.three.modal.host",
    "https://one.two.modal.host.attacker.test", "https://-one.two.modal.host",
    "https://one-.two.modal.host", "https://one.two_.modal.host", "https://one..modal.host",
    "https://" + "a" * 64 + ".two.modal.host", "http://one.two.modal.host",
    "https://one.two.modal.host:8443", "https://user:password@one.two.modal.host",
    "https://one.two.modal.host/rpc", "https://one.two.modal.host?", "https://one.two.modal.host#",
    "https://one.two.modal.host\n", "https://one.two.modal.host\\@attacker.test",
])
def test_tunnel_url_rejects_unsafe_or_unrecognized_origins(url):
    with pytest.raises(ValueError) as error:
        validate_endpoint_url(url, http_ingress="tunnel")
    assert url not in str(error.value)


@pytest.mark.parametrize("expiry", [None, True, "2600", float("nan"), float("inf"), 10**400,
                                      NOW, NOW - 1, NOW + 901])
def test_tunnel_constructor_rejects_missing_or_unbounded_expiry_before_client_creation(monkeypatch, expiry):
    monkeypatch.setattr(http_transport, "_make_client", lambda token: pytest.fail("No HTTP client may be created"))
    with pytest.raises(ValueError):
        transport(http_session_expires_at=expiry)


def test_cheap_guard_is_nonblocking_and_wall_rollback_cannot_extend_lifetime(monkeypatch):
    clock = {"monotonic": 100.0, "wall": NOW}
    monkeypatch.setattr(http_transport.time, "monotonic", lambda: clock["monotonic"])
    monkeypatch.setattr(http_transport.time, "time", lambda: clock["wall"])
    value = transport()
    with value._state_lock, value._busy:
        value.ensure_session_active()
    clock.update(monotonic=700.0, wall=NOW - 100)
    with pytest.raises(RemoteFault, match="expired"):
        value.ensure_session_active()
    with pytest.raises(RemoteFault, match="expired"):
        value.ready(1)
    value.close()
    with pytest.raises(InvalidatedRequest, match="closed"):
        value.ensure_session_active()


def test_wall_clock_jump_also_expires_tunnel_before_monotonic_deadline(monkeypatch):
    value = transport()
    monkeypatch.setattr(http_transport.time, "time", lambda: EXPIRY)
    with pytest.raises(RemoteFault, match="expired"):
        value.ensure_session_active()


def test_tunnel_lifetime_sets_shared_startup_stop_without_any_http_or_hardware(monkeypatch):
    monkeypatch.setattr(http_transport, "_make_client", lambda token: pytest.fail("No network in lifetime timer"))
    stopped = threading.Event()
    value = transport(http_session_expires_at=NOW + 0.03, shutdown_event=stopped)
    try:
        assert value._expiry_timer.daemon
        assert stopped.wait(1)
        with pytest.raises(InvalidatedRequest, match="stopped"):
            value.ensure_session_active()
    finally:
        value.close()


def test_close_cancels_owned_expiry_timer_and_no_timer_for_direct_benchmark(monkeypatch):
    timers = []

    class Timer:
        def __init__(self, delay, callback):
            self.delay, self.callback, self.cancelled, self.started = delay, callback, False, False
            timers.append(self)

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(http_transport.threading, "Timer", Timer)
    direct = transport()
    assert not timers
    direct.close()
    event = threading.Event()
    value = transport(shutdown_event=event)
    assert len(timers) == 1 and timers[0].started and timers[0].daemon
    assert 0 < timers[0].delay <= 600
    value.close()
    value.close()
    assert timers[0].cancelled and value._expiry_timer is None and not event.is_set()


def test_tunnel_roundtrip_preserves_wire_and_measures_exact_route(monkeypatch):
    seen = []

    def handler(request):
        assert str(request.url) == URL + "/rpc"
        envelope = decode_message(request.content)
        seen.append(envelope)
        return reply(route_metadata() if envelope["method"] == "ready" else {"payload": envelope["payload"]})

    client = install_http(monkeypatch, handler)
    try:
        value = transport()
        assert value.ready(1) == route_metadata()
        payload = {"images": {"top": {"data": bytes(range(256))}}, "state": [0.125] * 14}
        assert value.predict_chunk(payload, 1)["payload"] == payload
        assert value.last_timing["http_ingress"] == "tunnel"
        assert value.last_timing["http_session_expires_at"] == EXPIRY
        assert value.last_timing["http_endpoint_sha256"] == hashlib.sha256(URL.encode()).hexdigest()
        assert URL not in str(value.last_timing) and TOKEN not in str(value.last_timing)
        assert len(seen) == 2
    finally:
        client.close()


@pytest.mark.parametrize("change", [{"http_endpoint": OTHER}, {"http_ingress": "asgi"},
                                    {"http_session_expires_at": EXPIRY + 1},
                                    {"http_session_expires_at": None}, {"http_endpoint": None}])
def test_changed_ready_route_retires_transport_before_prediction(monkeypatch, change):
    calls = []

    def handler(request):
        calls.append(request)
        return reply(route_metadata(**change))

    client = install_http(monkeypatch, handler)
    value = transport()
    try:
        with pytest.raises(RemoteFault, match="readiness"):
            value.ready(1)
        with pytest.raises(InvalidatedRequest, match="closed"):
            value.predict_chunk({}, 1)
        assert len(calls) == 1
    finally:
        client.close()


def test_http_return_after_wall_expiry_is_discarded(monkeypatch):
    def handler(request):
        monkeypatch.setattr(http_transport.time, "time", lambda: EXPIRY)
        return reply({"late": True})

    client = install_http(monkeypatch, handler)
    try:
        with pytest.raises(RemoteFault, match="expired"):
            transport().predict_chunk({}, 1)
    finally:
        client.close()


def test_runtime_binding_requires_exact_explicit_tunnel_metadata():
    metadata = {**runtime_metadata(), **route_metadata()}
    result = identity.http_runtime_binding("molmoact2", metadata, execution_mode="cuda_graph10",
                                           task=TASK, image_hw=IMAGE_HW, endpoint_url=URL)
    assert all(result[key] == val for key, val in route_metadata().items())
    with pytest.raises(ValueError, match="origin"):
        identity.http_runtime_binding("molmoact2", metadata, execution_mode="cuda_graph10",
                                      task=TASK, image_hw=IMAGE_HW, endpoint_url=OTHER)


@pytest.fixture
def receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(modal_ops, "receipt_path", lambda: tmp_path / "owned.json")
    monkeypatch.setattr(modal_ops.subprocess, "run", lambda *a, **k: pytest.fail("No cloud commands"))
    metadata = {**runtime_metadata(), **route_metadata()}
    value = {"app_name": APP, "status": "ready", "transport": "http", "profile_id": "molmoact2",
             "revision": metadata["revision"], "execution_mode": "cuda_graph10", "region": "us-west",
             "routing_region": "us-west", "metadata": metadata, **route_metadata()}
    modal_ops._save(value)
    modal_ops._save_http_auth(APP, URL, TOKEN, http_ingress="tunnel")
    return value


def test_private_tunnel_credentials_need_no_modal_account_and_keep_original_schema(receipt, monkeypatch):
    for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(name, raising=False)
    private = json.loads(modal_ops.http_auth_path().read_text())
    assert set(private) == {"app_name", "endpoint_url", "token"}
    assert modal_ops.http_auth_path().stat().st_mode & 0o777 == 0o600
    credentials = modal_ops.http_credentials(APP)
    assert credentials["token"] == TOKEN and credentials["endpoint_url"] == URL
    assert credentials["http_ingress"] == "tunnel" and credentials["http_session_expires_at"] == EXPIRY
    assert TOKEN not in modal_ops.receipt_path().read_text()


@pytest.mark.parametrize("key,value", [("http_ingress", "asgi"), ("http_session_expires_at", EXPIRY + 1),
                                      ("http_endpoint", OTHER)])
def test_credentials_reject_receipt_and_runtime_route_mismatch(receipt, key, value):
    receipt[key] = value
    modal_ops._save(receipt)
    with pytest.raises(ValueError):
        modal_ops.http_credentials(APP)


def test_expired_credentials_remain_removable_after_owned_shutdown(receipt, monkeypatch):
    monkeypatch.setattr(identity.time, "time", lambda: EXPIRY)
    with pytest.raises(ValueError, match="unexpired"):
        modal_ops.http_credentials(APP)
    modal_ops._remove_http_auth(APP)
    assert not modal_ops.http_auth_path().exists()


@pytest.mark.parametrize("key,value", [("http_endpoint", OTHER), ("http_session_expires_at", EXPIRY + 1),
                                      ("instance_id", "another-container"), ("http_ingress", "asgi")])
def test_fresh_readiness_must_match_owned_origin_expiry_and_instance_before_hardware(receipt, key, value):
    config = SimpleNamespace(profile="molmoact2", modal_app=APP, call_mode="http", execution_mode="cuda_graph10",
                             task=TASK, image_encoding="rgb8", jpeg_quality=85, center_crop=False,
                             prediction_queue_threshold=30, max_observation_age_s=2.0)
    metadata = copy.deepcopy(receipt["metadata"])
    assert qualification.current_settings(config, image_hw=IMAGE_HW)["http_ingress"] == "tunnel"
    metadata[key] = value
    with pytest.raises(qualification.QualificationError):
        qualification.current_settings(config, image_hw=IMAGE_HW, metadata=metadata)


@pytest.fixture
def tunnel_evidence(monkeypatch):
    settings, direct, integrated = _http_evidence.__wrapped__(monkeypatch)
    settings.update(route_metadata())
    for report in (direct, integrated):
        report["readiness"].update(route_metadata())
        for row in report["samples"]:
            row["transport_timing"].update(http_ingress="tunnel", http_session_expires_at=EXPIRY,
                http_endpoint_sha256=hashlib.sha256(URL.encode()).hexdigest())
    return settings, direct, integrated


def test_tunnel_evidence_qualifies_only_during_exact_live_session(tunnel_evidence, tmp_path, monkeypatch):
    settings, direct, integrated = tunnel_evidence
    record = qualification.build_qualification(settings, direct=direct, integrated=integrated)
    assert record["assessment"]["qualified"], record["assessment"]["reasons"]
    path = qualification.save_qualification(record, tmp_path / "qualification.json")
    assert qualification.validate_qualification(settings, path=path)["assessment"]["qualified"]
    monkeypatch.setattr(identity.time, "time", lambda: EXPIRY)
    with pytest.raises(qualification.QualificationError):
        qualification.validate_qualification(settings, path=path)


@pytest.mark.parametrize("report", [1, 2])
@pytest.mark.parametrize("key,value", [("http_ingress", "asgi"), ("http_ingress", None),
                                      ("http_endpoint_sha256", "x" * 64),
                                      ("http_session_expires_at", EXPIRY + 1)])
def test_every_tunnel_wire_sample_must_prove_qualified_route_and_lifetime(tunnel_evidence, report, key, value):
    settings, direct, integrated = tunnel_evidence
    tunnel_evidence[report]["samples"][25]["transport_timing"][key] = value
    record = qualification.build_qualification(settings, direct=direct, integrated=integrated)
    assert not record["assessment"]["qualified"]
    assert any("ingress" in reason for reason in record["assessment"]["reasons"])


@pytest.mark.parametrize("report", [1, 2])
@pytest.mark.parametrize("key,value", [("http_ingress", "tunnel"), ("http_session_expires_at", EXPIRY),
                                      ("http_endpoint_sha256", "0" * 64)])
def test_legacy_asgi_omissions_do_not_allow_explicit_route_contradictions(monkeypatch, report, key, value):
    evidence = _http_evidence.__wrapped__(monkeypatch)
    settings, direct, integrated = evidence
    evidence[report]["samples"][25]["transport_timing"][key] = value
    record = qualification.build_qualification(settings, direct=direct, integrated=integrated)
    assert not record["assessment"]["qualified"]
    assert any("ingress" in reason for reason in record["assessment"]["reasons"])
