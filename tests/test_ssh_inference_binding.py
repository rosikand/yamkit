"""Owned SSH endpoints stay explicit and cannot inherit Modal's identity rules."""

import copy
import time

import httpx
import pytest

from tests.test_http_runtime_binding import IMAGE_HW, TASK, runtime_metadata
from yamkit.inference import http_transport, identity
from yamkit.inference.client import RemoteFault
from yamkit.inference.http_transport import HttpTransport, validate_endpoint_url
from yamkit.inference.http_wire import encode_message

URL = "http://127.0.0.1:8765"
TOKEN = "test-only-ssh-token-01234567890123456789"


def provenance():
    return {"python": "3.12.12", "packages": {"lerobot": "0.6.1", "torch": "2.11.0+cu128",
                                             "transformers": "5.5.4"},
            "torch_cuda": "12.8", "gpu": {"device": "cuda:0", "name": "NVIDIA H100 SXM5"}}


def ssh_metadata():
    result = runtime_metadata()
    for key in ("requested_compute_region", "compute_region", "routing_region"):
        result.pop(key)
    result.update(http_ingress="ssh", http_endpoint=URL, http_session_expires_at=time.time() + 28800,
                  external_service={"provider": "lambda", "service_id": "lambda-georgia", "host_id": "a" * 64,
                                    "region": "Georgia", "region_source": "operator_declared"},
                  runtime_provenance=provenance())
    return result


@pytest.fixture(autouse=True)
def fixed_source(monkeypatch):
    monkeypatch.setattr(identity, "inference_build_id", lambda: "b" * 64)


def test_ssh_endpoint_accepts_only_explicit_literal_loopback():
    assert validate_endpoint_url(URL, http_ingress="ssh") == URL
    assert validate_endpoint_url("http://127.0.0.1:65535", http_ingress="ssh")
    with pytest.raises(ValueError):
        validate_endpoint_url(URL)
    with pytest.raises(ValueError):
        validate_endpoint_url(URL, http_ingress="tunnel")
    assert validate_endpoint_url("https://a.modal.run") == "https://a.modal.run"
    assert validate_endpoint_url("https://a.b.modal.host", http_ingress="tunnel") == "https://a.b.modal.host"


@pytest.mark.parametrize("url", [
    None, "http://localhost:8765", "http://127.0.0.2:8765", "http://[::1]:8765", "http://2130706433:8765",
    "http://0.0.0.0:8765", "http://192.0.2.1:8765", "https://127.0.0.1:8765", "http://127.0.0.1",
    "http://127.0.0.1:0", "http://127.0.0.1:65536", "http://127.0.0.1:08765", URL + "/", URL + "/rpc",
    URL + "?", URL + "#token", URL + "\n", "http://user:secret@127.0.0.1:8765", "HTTP://127.0.0.1:8765",
])
def test_ssh_endpoint_rejects_aliases_public_http_and_unbounded_authority(url):
    with pytest.raises(ValueError) as exc:
        validate_endpoint_url(url, http_ingress="ssh")
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("seconds", [28800, 86400])
def test_owned_ssh_lifetime_does_not_extend_modal_tunnel_limit(monkeypatch, seconds):
    monkeypatch.setattr(identity.time, "time", lambda: 1000.0)
    metadata = {"http_ingress": "ssh", "http_endpoint": URL, "http_session_expires_at": 1000.0 + seconds}
    assert identity.http_ingress_binding(metadata)["http_session_expires_at"] == 1000 + seconds
    metadata.update(http_ingress="tunnel", http_endpoint="https://a.b.modal.host")
    with pytest.raises(ValueError, match="tunnel"):
        identity.http_ingress_binding(metadata)


@pytest.mark.parametrize("expiry", [None, True, float("nan"), float("inf"), 10**400, -1])
def test_ssh_requires_finite_future_session(expiry):
    with pytest.raises(ValueError):
        HttpTransport("lambda-georgia", "molmoact2", endpoint_url=URL, token=TOKEN,
                      http_ingress="ssh", http_session_expires_at=expiry)


def test_ssh_runtime_binding_carries_independent_provider_and_installed_runtime():
    metadata = ssh_metadata()
    result = identity.http_runtime_binding("molmoact2", metadata, execution_mode="cuda_graph10", task=TASK,
                                           image_hw=IMAGE_HW, endpoint_url=URL)
    assert result["external_service"] == metadata["external_service"]
    assert result["runtime_provenance"] == metadata["runtime_provenance"]
    result["external_service"]["region"] = "elsewhere"
    result["runtime_provenance"]["packages"]["torch"] = "different"
    assert metadata["external_service"]["region"] == "Georgia"
    assert metadata["runtime_provenance"]["packages"]["torch"] == "2.11.0+cu128"


@pytest.mark.parametrize("path,value", [
    (("external_service", "provider"), "modal"), (("external_service", "host_id"), "raw-hostname"),
    (("external_service", "service_id"), "bad/name"), (("external_service", "region_source"), "provider_verified"),
    (("external_service", "region"), "Georgia\nsecret"), (("runtime_provenance", "packages", "torch"), "2.11.0+cpu"),
    (("runtime_provenance", "packages", "lerobot"), "0.7.0"),
    (("runtime_provenance", "packages", "transformers"), "unknown"),
    (("runtime_provenance", "torch_cuda"), None), (("runtime_provenance", "python"), "3.13.0"),
    (("runtime_provenance", "gpu", "name"), ""), (("inference_build_id",), "old-source"),
])
def test_changed_ssh_identity_or_unpinned_runtime_is_rejected(path, value):
    metadata = ssh_metadata()
    target = metadata
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        identity.http_runtime_binding("molmoact2", metadata, execution_mode="cuda_graph10", task=TASK,
                                      image_hw=IMAGE_HW, endpoint_url=URL)


def test_owned_provider_cannot_masquerade_as_modal():
    metadata = ssh_metadata()
    metadata.update(http_ingress="asgi", http_endpoint="https://a.modal.run", http_session_expires_at=None)
    with pytest.raises(ValueError, match="external service"):
        identity.http_runtime_binding("molmoact2", metadata, execution_mode="cuda_graph10", task=TASK,
                                      image_hw=IMAGE_HW)


def test_ssh_transport_verifies_exact_expiry_on_every_ready_response(monkeypatch):
    metadata = ssh_metadata()
    replies = [metadata, {**copy.deepcopy(metadata), "http_session_expires_at": metadata["http_session_expires_at"] + 1}]

    def handler(request):
        assert str(request.url) == URL + "/rpc"
        return httpx.Response(200, stream=httpx.ByteStream(encode_message({"ok": True, "result": replies.pop(0)})),
                              headers={"content-type": "application/octet-stream"})

    client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    monkeypatch.setattr(http_transport, "_make_client", lambda token: client)
    transport = HttpTransport("lambda-georgia", "molmoact2", endpoint_url=URL, token=TOKEN,
                              http_ingress="ssh", http_session_expires_at=metadata["http_session_expires_at"])
    try:
        assert transport.ready(1)["external_service"]["provider"] == "lambda"
        with pytest.raises(RemoteFault, match="readiness"):
            transport.ready(1)
        assert transport._closed
    finally:
        transport.close()
